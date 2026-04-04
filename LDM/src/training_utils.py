from dataclasses import dataclass, replace
import tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import lagrdataset
import latn
import latn_globals
import distributed

@dataclass
class TrainDesc:
    optimizer: any
    learning_rate: any
    scheduler: any
    epochs: int
    savepath: str
    batch_size: int
    loss_fn: any
    save_every: int
    world_size: int  # world_size = num accelerators

def construct_dataloader(ds: lagrdataset.LagrDataset,
                         train_desc: TrainDesc,
                         sampler=None):
    """
    Args:
       ds, test_ds: datasets
       sampler <: torch.utils.data.sampler.Sampler

    Specialized dataloader that relies on LagrDataset().__getitems__(idxs)
     to return an /already collated/ tuple of (inputs, outputs) that can
     be consumed by the model.
    If this assumption changes, collate_fn=no_action should be removed
    """
    def no_action(already_collated_samples):
        return already_collated_samples

    dl = DataLoader(ds, batch_size=train_desc.batch_size,
                    shuffle=(sampler is None),
                    sampler=None if (sampler is None) else sampler(ds),
                    collate_fn=no_action)
    return dl


def load_train_objs(data_desc: lagrdataset.DataDesc,
                    network_desc: latn.LATNDesc,
                    train_desc: TrainDesc):
    """
    Encapsulate interfacing with LagrDataset & LATN
    """
    ff_type = latn.FFN# if data_desc.target_name == "vis" else latn.Skip_FFN
    train_ds, test_ds = lagrdataset.LagrDataset.from_file(data_desc)
    model = latn.LATN(data_desc, network_desc, latn.ConstrainedTensorHistoryConv, ff_type)
    model.set_timescale(train_ds.timescale)
    return dict({'train_ds': train_ds,
                 'test_ds': test_ds,
                 'model': model})

def load_node_train_objs(data_desc: lagrdataset.DataDesc,
                         network_desc: latn.LATNDesc,
                         train_desc: TrainDesc):
    train_ds, test_ds = lagrdataset.LagrDataset.from_file(data_desc)

    network_desc = replace(network_desc,
                           output_len=latn_globals.NUM_PIJ_OUTPUTS)
    ph_model = latn.LATN(data_desc,
                         network_desc,
                         latn.ConstrainedTensorHistoryConv,
                         latn.FFN)#Skip_FFN)
    ph_model_sd = torch.load(train_desc.savepath + "../ph/apriori_model_state_dict.pt")
    ph_model.load_state_dict(ph_model_sd)

    network_desc = replace(network_desc,
                           output_len=latn_globals.NUM_VIS_OUTPUTS)
    vis_model = latn.LATN(data_desc,
                          network_desc,
                          latn.ConstrainedTensorHistoryConv,
                          latn.FFN)
    vis_model_sd = torch.load(train_desc.savepath + "../vis/apriori_model_state_dict.pt")
    vis_model.load_state_dict(vis_model_sd)
    
    model = latn.LATN_NODE(data_desc, ph_model, vis_model);
    return dict({'train_ds': train_ds,
                 'test_ds': test_ds,
                 'model': model})

class Trainer:
    """
    Lifted from pytorch tutorials
    """
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            test_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            learning_rate,
            scheduler,
            device: any,  # device is int for gpu, str otherwise
            save_every: int,
            savepath: str) -> None:
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.scheduler = scheduler(self.optimizer, factor=0.8, threshold=1e-2, patience=15)
        self.save_every = save_every
        self.savepath = savepath
        self.normalization = 1e5
        if isinstance(device, int):
            self.gpu_id = device
            self.model = DDP(model, device_ids=[self.gpu_id])
            if (self.gpu_id == 0):
                self.writer = SummaryWriter(log_dir=self.savepath)
        else:
            self.gpu_id = device
            self.writer = SummaryWriter()

    def _write_train_loss(self, train_loss):
        with open(self.savepath + "/train_loss.csv", "a") as fp:
            fp.write(f"{train_loss:.6e}\n")

    def _write_test_loss(self, test_loss):
        with open(self.savepath + "/test_loss.csv", "a") as fp:
            fp.write(f"{test_loss:.6e}\n")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.functional.mse_loss(output, targets)/self.normalization
        loss.backward()  # synchronizes distributed
        self.optimizer.step()
        return loss

    def _run_eval_test_data(self, sources, targets):
        sources = sources.to(self.gpu_id)
        sources = self.test_data.dataset.reinflate_input(sources)
        targets = targets.to(self.gpu_id)
        targets = self.test_data.dataset.reinflate_output(targets)[0]
        outputs = self.model(sources)
        return outputs, targets
    
    def _run_test(self, epoch):
        self.model.eval()
        loss = 0
        for source, targets in self.test_data:
            outputs, targets = self._run_eval_test_data(source, targets)
            loss += torch.nn.functional.mse_loss(outputs, targets)/self.normalization
        loss /= len(self.test_data)
        tensor_list = [torch.tensor(0.0, device=self.gpu_id) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list, loss)
        loss = sum(tensor_list)/len(tensor_list)
        if (distributed.save_process(self.gpu_id)):
            print(f"Test loss  = {loss:.3e}")
            self.writer.add_scalar(f"Loss/test", loss, epoch)
            self._write_test_loss(loss)
        return loss
        

    def _run_epoch(self, epoch):
        if (distributed.save_process(self.gpu_id)):
            print(f"Epoch: {epoch}")
        b_sz = len(next(iter(self.train_data))[0])
        
        if isinstance(self.train_data.sampler,
                      torch.utils.data.DistributedSampler):
            self.train_data.sampler.set_epoch(epoch)

        loss = 0
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            source = self.test_data.dataset.reinflate_input(source)
            targets = targets.to(self.gpu_id)
            targets = self.test_data.dataset.reinflate_output(targets)[0]
            loss += self._run_batch(source, targets)
        loss /= len(self.train_data)
        tensor_list = [torch.tensor(0.0, device=self.gpu_id) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list, loss)
        loss = sum(tensor_list)/len(tensor_list)
        if (self.gpu_id == 0):
            print(f"Train loss = {loss:.3e}")
            self.writer.add_scalar(f"Loss/train", loss, epoch)
            self._write_train_loss(loss)
        return loss

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = f"{self.savepath}/checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _save_best(self):
        ckp = self.model.state_dict()
        PATH = f"{self.savepath}/checkpoint_best_model.pt"
        torch.save(ckp, PATH)

    def _load_checkpoint(self, path, model):
        ckp = torch.load(path, weights_only=True)
        model.load_state_dict(ckp)

    def _set_normalization(self):
        train_loss = 0
        for source, targets in self.train_data:
            targets = targets.to(self.gpu_id)
            targets = self.test_data.dataset.reinflate_output(targets)[0]
            output = torch.zeros_like(targets)
            train_loss += torch.nn.functional.mse_loss(output, targets)
        train_loss /= len(self.test_data)
        train_tensor_list = [torch.tensor(0.0, device=self.gpu_id) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(train_tensor_list, train_loss)
        train_loss = sum(train_tensor_list)/len(train_tensor_list)

        test_loss = 0
        for source, targets in self.test_data:
            targets = targets.to(self.gpu_id)
            targets = self.test_data.dataset.reinflate_output(targets)[0]
            output = torch.zeros_like(targets)
            test_loss += torch.nn.functional.mse_loss(output, targets)
        test_loss /= len(self.test_data)
        test_tensor_list = [torch.tensor(0.0, device=self.gpu_id) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(test_tensor_list, test_loss)
        test_loss = sum(test_tensor_list)/len(test_tensor_list)
            
        self.normalization = test_loss + train_loss
        print(f"normalization = {self.normalization}")

    def train(self, max_epochs: int):
        min_test_loss = torch.inf
        self._set_normalization()
        
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            test_loss = self._run_test(epoch)
            self.scheduler.step(test_loss)
            if distributed.save_process(self.gpu_id) \
               and ((epoch+1) % self.save_every == 0):
                self._save_checkpoint(epoch)
            if distributed.save_process(self.gpu_id) \
               and ((epoch+1) % 5 == 0) \
               and (test_loss < min_test_loss):
                self._save_best()
                min_test_loss = test_loss
            if distributed.save_process(self.gpu_id) \
               and ((epoch+1) % 5 == 0):
                print(f"lr = {self.scheduler.get_last_lr()}")
