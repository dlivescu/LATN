import os
import numpy
import unittest as ut
import torch
import utils
import lagrdataset
import latn
import training_utils
import distributed

DATA_DESC = lagrdataset.DataDesc(
    os.path.dirname(os.path.abspath(__file__)) + '/test_data',
    (1000, 100, 3, 3),  # datashape = (num_samples, num_tsteps, 3, 3)
    "pij",  # target_name,
    3e-4,  # dt,
    5,  # history_timestep,
    25,  # history_length,
    0.3)  # percent_test)
NETWORK_DESC = latn.LATNDesc(
    2,  # num_layers
    20,  # num_units
    torch.nn.ReLU,  # activation
    10,  # input_len
    10,  # output_len
    0.2)  # dropout_rate
TRAIN_DESC = training_utils.TrainDesc(
    torch.optim.Adam,  # optimizer
    torch.optim.lr_scheduler.ReduceLROnPlateau,  # scheduler
    10,  # epochs
    "./test_output",  # savepath
    32,  # batch_size
    torch.nn.functional.mse_loss,  # loss_fn
    5,  # save_every
    0)  # world_size

def multiprocess_setup(rank, 
                       train_desc: training_utils.TrainDesc,
                       sampler):
    distributed.ddp_setup(rank, train_desc.world_size)
    train_ds, test_ds = lagrdataset.\
        LagrDataset.from_file(DATA_DESC)
    train_dl = training_utils.construct_dataloader(train_ds,
                                                   train_desc,
                                                   sampler)
    ut.TestCase().assertIsInstance(train_dl, torch.utils.data.DataLoader)
    torch.distributed.destroy_process_group()

def multiprocess_train(rank: int,
                       train_desc: training_utils.TrainDesc,
                       data_desc: lagrdataset.DataDesc,
                       network_desc: latn.LATNDesc):
    distributed.ddp_setup(rank, train_desc.world_size)
    train_objs = training_utils.load_train_objs(
        data_desc, network_desc, train_desc)
    model = distributed.distribute_model(train_objs['model'].to(rank),
                                         rank)
    sampler = torch.utils.data.distributed.DistributedSampler
    train_dl = training_utils.construct_dataloader(train_objs['train_ds'],
                                                   train_desc,
                                                   sampler)
    test_dl = training_utils.construct_dataloader(train_objs['test_ds'],
                                                  train_desc,
                                                  sampler)
    trainer = training_utils.Trainer(
        model, train_dl, test_dl, train_desc.optimizer, rank, train_desc.save_every)
    trainer.train(train_desc.epochs)
    torch.distributed.destroy_process_group()



class Test_helpers(ut.TestCase):
    def setUp(self):
        """create datasets to act on with LATN models"""
        self.train_ds, self.test_ds = lagrdataset.\
            LagrDataset.from_file(DATA_DESC)
        
    def test_construct_dataloader(self):
        dl = training_utils.construct_dataloader(self.train_ds, TRAIN_DESC)
        self.assertIsInstance(dl, torch.utils.data.DataLoader)
        
        if (torch.cuda.device_count() > 1):
            TRAIN_DESC.world_size = torch.cuda.device_count()
            
            torch.multiprocessing.spawn(
                multiprocess_setup,
                args=(TRAIN_DESC, torch.utils.data.distributed.DistributedSampler),
                nprocs=TRAIN_DESC.world_size)
            TRAIN_DESC.world_size = 0

        else:
            sampler = None
            dl = training_utils.\
                construct_dataloader(self.train_ds,
                                     TRAIN_DESC,
                                     sampler=sampler)
            self.assertIsInstance(dl.sampler, torch.utils.data.sampler.RandomSampler)

    def test_load_train_objs(self):
        d = training_utils.load_train_objs(DATA_DESC,
                                           NETWORK_DESC,
                                           TRAIN_DESC)
        self.assertIsInstance(d, dict)
        self.assertIsInstance(d['train_ds'], lagrdataset.LagrDataset)
        self.assertIsInstance(d['test_ds'], lagrdataset.LagrDataset)
        self.assertIsInstance(d['model'], latn.LATN)


class TestTrainer(ut.TestCase):
    def setUp(self):
        """create datasets to act on with LATN models"""
        train_objs = training_utils.load_train_objs(DATA_DESC,
                                                    NETWORK_DESC,
                                                    TRAIN_DESC)
        train_dl = training_utils.construct_dataloader(train_objs['train_ds'],
                                                       TRAIN_DESC)
        test_dl = training_utils.construct_dataloader(train_objs['test_ds'],
                                                      TRAIN_DESC)

        model = train_objs['model']
        device = distributed.get_available_device()
        self.trainer = training_utils.\
            Trainer(model,
                    train_dl,
                    test_dl,
                    torch.optim.Adam,
                    device,  # device
                    5)  # save_every
            
    def test__init__(self):
        #TODO test gpu, multigpu
        self.assertIsInstance(self.trainer, training_utils.Trainer)

    def test_run_batch(self):
        it = iter(self.trainer.train_data)
        flat_inputs, flat_outputs = next(it)
        flat_inputs = flat_inputs.to(self.trainer.gpu_id)
        flat_outputs = flat_outputs.to(self.trainer.gpu_id)
        self.assertEqual(flat_inputs.shape[0], TRAIN_DESC.batch_size)
        self.assertEqual(flat_outputs.shape[0], TRAIN_DESC.batch_size)

        inputs = self.trainer.train_data.dataset.reinflate_input(flat_inputs)
        T = (DATA_DESC.history_length//DATA_DESC.history_timestep) + 1
        outputs = self.trainer.train_data.dataset.reinflate_output(flat_outputs)
        self.assertEqual(inputs[0].shape, (TRAIN_DESC.batch_size, T, 9))
        self.assertEqual(inputs[1].shape, (TRAIN_DESC.batch_size, 5))
        self.assertEqual(inputs[2].shape, (TRAIN_DESC.batch_size, 3, 3, 10))
        self.assertEqual(outputs[0].shape, (TRAIN_DESC.batch_size, 3, 3))

        self.assertEqual(self.trainer.model(inputs).shape,
                         outputs[0].shape)
        self.trainer._run_batch(inputs, outputs[0])

    def test_run_test(self):
        self.trainer._run_test()

    def test_run_epoch(self):
        self.trainer._run_epoch(1)

    def test_saveload_checkopint(self):
        model_state_1 = dict()
        for key in self.trainer.model.state_dict():
            model_state_1[key] = self.trainer.model.state_dict()[key].\
                clone().detach()

        self.trainer._save_checkpoint(0)
        
        self.trainer._run_epoch(1)
        for key in model_state_1:
            self.assertFalse(torch.allclose(model_state_1[key],
                                            self.trainer.model.state_dict()[key]))

        self.trainer._load_checkpoint('./checkpoint.pt',
                                      self.trainer.model)
        for key in model_state_1:
            self.assertTrue(torch.allclose(model_state_1[key],
                                           self.trainer.model.state_dict()[key]))

    def test_train(self):
        self.trainer.train(10)

    def test_multigpu_train(self):
        if (torch.cuda.device_count() > 1):
            TRAIN_DESC.world_size = torch.cuda.device_count()
            torch.multiprocessing.spawn(
                multiprocess_train,
                args=(TRAIN_DESC, DATA_DESC, NETWORK_DESC),
                nprocs=TRAIN_DESC.world_size)
            TRAIN_DESC.world_size = 0
