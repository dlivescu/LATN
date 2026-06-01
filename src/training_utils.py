import os
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
    resume: bool = False  # resume from checkpoint_resume.pt in savepath if present

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


# def train_network(train_ds, test_ds, model,
#                   train_desc: TrainDesc, device='cpu'):
#     """Standardized function to:
#     0. construct dataloaders
#         - encapsulate device(s) specific bits
#     1. compute loss (after moving data to correct device
#     2. compute gradient
#     3. update parameters
#     4. log results using TensorBoard
#     """
#     # 0. Construct dataloaders:
#     is_distributed = train_desc.world_size > 1
#     sampler = torch.utils.data.DistributedSampler if is_distributed \
#         else torch.utils.data.RandomSampler
#     train_dl = construct_dataloader(train_ds, train_desc, sampler)
#     test_dl = construct_dataloader(test_ds, train_desc, sampler)

#     # 1. Compute loss
#     # def train_loop(_train_ds, _opt, _loss
#     # return

def load_train_objs(data_desc: lagrdataset.DataDesc,
                    network_desc: latn.LATNDesc,
                    train_desc: TrainDesc):
    """
    Encapsulate interfacing with LagrDataset & LATN
    """
    ff_type = latn.FFN# if data_desc.target_name == "vis" else latn.Skip_FFN
    train_ds, test_ds = lagrdataset.LagrDataset.from_file(data_desc)
    model = latn.LATN(data_desc, network_desc,
                      latn.ConstrainedTensorHistoryConv, ff_type)
    model.set_timescale(train_ds.timescale)
    return dict({'train_ds': train_ds,
                 'test_ds': test_ds,
                 'model': model})


def load_node_train_objs(data_desc: lagrdataset.DataDesc,
                         network_desc: latn.LATNDesc,
                         train_desc: TrainDesc):
    """Assemble a LATN_NODE from the a-priori-trained ph/vis models.

    Loads each sub-model's ``apriori_model_state_dict.pt`` (saved by
    ``gpu_tangent_learning`` as a bare, DDP-unwrapped state dict) from
    ``../ph`` and ``../vis`` relative to the node savepath.
    """
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
            # source = source.to(self.gpu_id)
            # source = self.test_data.dataset.reinflate_input(source)
            # targets = targets.to(self.gpu_id)
            # targets = self.test_data.dataset.reinflate_output(targets)[0]
            # output = self.model(source)
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

        sub_epoch_count = 0
        loss = 0
        for source, targets in self.train_data:
            #print(f"\n[GPU{self.gpu_id}] Epoch {epoch} \
            #| Batchsize: {b_sz} | Step: {sub_epoch_count}")
            sub_epoch_count += 1

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

    def _resume_path(self):
        return f"{self.savepath}/checkpoint_resume.pt"

    def _save_resume_state(self, epoch, min_test_loss):
        """Full training state for spot-interruption resume.

        Distinct from `_save_checkpoint`, which writes a bare model
        state_dict that external consumers (notebooks) expect. This bundle
        additionally carries optimizer/scheduler state, the epoch counter,
        the running best test loss, and the loss normalization so a resumed
        run is numerically continuous.
        """
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'min_test_loss': min_test_loss,
            'normalization': self.normalization,
        }
        torch.save(state, self._resume_path())

    def _load_resume_state(self):
        """Restore from `_save_resume_state`. Returns (start_epoch, min_test_loss)."""
        map_location = (f"cuda:{self.gpu_id}" if isinstance(self.gpu_id, int)
                        else self.gpu_id)
        state = torch.load(self._resume_path(), map_location=map_location,
                           weights_only=False)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.normalization = state['normalization']
        return state['epoch'] + 1, state['min_test_loss']

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

    def train(self, max_epochs: int, resume: bool = False):
        start_epoch = 0
        min_test_loss = torch.inf

        # Resume bundle lives on the instance's local disk, so every rank
        # reads the same file and starts from identical weights (required by
        # DDP). If resume is requested but no bundle exists yet (fresh launch,
        # or the second model hasn't started), fall through to a fresh start.
        if resume and os.path.exists(self._resume_path()):
            start_epoch, min_test_loss = self._load_resume_state()
            if distributed.save_process(self.gpu_id):
                print(f"Resuming at epoch {start_epoch} "
                      f"(min_test_loss={min_test_loss:.3e}, "
                      f"normalization={self.normalization})")
        else:
            self._set_normalization()

        for epoch in range(start_epoch, max_epochs):
            self._run_epoch(epoch)
            test_loss = self._run_test(epoch)
            self.scheduler.step(test_loss)
            if distributed.save_process(self.gpu_id) \
               and ((epoch+1) % 5 == 0) \
               and (test_loss < min_test_loss):
                self._save_best()
                min_test_loss = test_loss
            if distributed.save_process(self.gpu_id) \
               and ((epoch+1) % 5 == 0):
                print(f"lr = {self.scheduler.get_last_lr()}")
            # full-state checkpoint last, so the resume bundle captures the
            # freshest min_test_loss from the best-model block above.
            if distributed.save_process(self.gpu_id) \
               and ((epoch+1) % self.save_every == 0):
                self._save_checkpoint(epoch)
                self._save_resume_state(epoch, min_test_loss)
