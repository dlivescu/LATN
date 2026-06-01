import os
import json
import argparse
from pathlib import Path
from dataclasses import replace
import torch
import utils
import lagrdataset
import latn
import latn_globals
import training_utils
import distributed

NUM_PIJ_OUTPUTS = latn_globals.NUM_PIJ_OUTPUTS
NUM_VIS_OUTPUTS = latn_globals.NUM_VIS_OUTPUTS
NUM_INVARIANTS = latn_globals.NUM_INVARIANTS


def run_slug(args):
    """Readable, path/S3-safe tag identifying a model+data configuration.

    Used to namespace each run's outputs (``<savepath>/<slug>/{ph,vis}/``) so a
    hyperparameter sweep doesn't let one config's checkpoints clobber another's.
    Excludes run-control knobs (max_epochs, save_every, resume) on purpose: the
    SAME experiment can be extended or resumed without changing its prefix.
    """
    parts = [f"nu{args.num_units}", f"nl{args.num_layers}",
             f"nf{args.num_filters}", f"lr{args.learning_rate:g}",
             f"dr{args.dropout_rate:g}", f"hl{args.history_length}",
             f"ht{args.history_timestep}", f"pt{args.percent_test:g}"]
    if args.num_samples:
        parts.append(f"ns{args.num_samples}")
    return "_".join(parts)

def single_threaded_main(train_desc: training_utils.TrainDesc,
                         data_desc: lagrdataset.DataDesc,
                         network_desc: latn.LATNDesc):
    datasets, model, optimizer = training_utils.load_train_objs(
        data_desc, network_desc, train_desc)
    train_dl = training_utils.construct_dataloader(datasets[0],
                                                   train_desc)
    test_dl = training_utils.construct_dataloader(datasets[1],
                                                  train_desc)
    trainer = training_utils.Trainer(model,
                                     train_dl,
                                     test_dl,
                                     optimizer,
                                     rank,
                                     train_desc.save_every,
                                     train_desc.savepath)
    trainer.train(train_desc.epochs, resume=train_desc.resume)

def gpu_tangent_learning(rank: int,
                         train_desc: training_utils.TrainDesc,
                         data_desc: lagrdataset.DataDesc,
                         network_desc: latn.LATNDesc):
    """
    Trains ph/vis LATN networks individually using ph/vis data from finite
     differences of computed fields in DNS.
    Meant to be called with at least one gpu.
    Args:
       rank: Unique identifier of each process, e.g. gpu
    """
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
    trainer = training_utils.Trainer(model,
                                     train_dl,
                                     test_dl,
                                     train_desc.optimizer,
                                     train_desc.learning_rate,
                                     train_desc.scheduler,
                                     rank,
                                     train_desc.save_every,
                                     train_desc.savepath)
    trainer.train(train_desc.epochs, resume=train_desc.resume)

    # bare (DDP-unwrapped) state dict the NODE stage loads as its sub-model
    if rank == 0:  # only save on one process
        torch.save(model.module.state_dict(),
                   train_desc.savepath + "apriori_model_state_dict.pt")

    # generate data for apriori evaluation, just pij/vis | aij?
    print(f"test_ds.shape = {len(train_objs['test_ds'])}")
    preds = []
    gts = []
    model.eval()
    with torch.no_grad(): # Disable gradient calculations
        for source, targets in trainer.test_data:
            outputs, targets = trainer._run_eval_test_data(source, targets)
            preds.append(outputs)
            gts.append(targets)
            print(f"outputs shape = {outputs.shape}")

            
    all_preds = torch.cat(preds)
    all_gts = torch.cat(gts)
    #N, i, j = all_preds.shape
    per_gpu_shape = all_preds.shape
    if rank == 0: # gather on 0
        gathered_preds = [torch.zeros(per_gpu_shape, device=rank)
                          for _ in range(train_desc.world_size)]
        gathered_gt = [torch.zeros(per_gpu_shape, device=rank)
                          for _ in range(train_desc.world_size)]
        torch.distributed.gather(all_preds, gather_list=gathered_preds, dst=0)
        torch.distributed.gather(all_gts, gather_list=gathered_gt, dst=0)
        torch.distributed.barrier(device_ids=[rank])
        ###---------------Synchronize-------------###
        gathered_preds = torch.cat(gathered_preds)
        gathered_gt = torch.cat(gathered_gt)
        torch.save({'pred ' + data_desc.target_name: gathered_preds.to('cpu'),
                    'gt ' + data_desc.target_name: gathered_gt.to('cpu')},
                   train_desc.savepath + f"/{data_desc.target_name}_apriori_eval.pt")
    else:
        torch.distributed.gather(all_preds, dst=0)
        torch.distributed.gather(all_gts, dst=0)
        torch.distributed.barrier(device_ids=[rank])
        ###---------------Synchronize-------------###

    ################# THIS MUST BE LAST CALL ####################
    torch.distributed.destroy_process_group()


def gpu_node_learning(rank: int,
                      train_desc: training_utils.TrainDesc,
                      data_desc: lagrdataset.DataDesc,
                      network_desc: latn.LATNDesc):
    """
    Trains joint neural ode (NODE) model - comprised of pressure Hessian
     and viscous Laplacian LATN networks - toward minimizing the MSE on
     trajectories of velocity gradient tensor (aij).
    This is done first by mean (no stochasticity), then by tuning the parameters
     of the noise to account of any remaining residual.
    """
    distributed.ddp_setup(rank, train_desc.world_size)
    train_objs = training_utils.load_node_train_objs(data_desc,
                                                     network_desc,
                                                     train_desc)
    train_objs['model'].set_timescale(train_objs['train_ds'].timescale)

    model = distributed.distribute_model(train_objs['model'].to(rank),
                                         rank)
    sampler = torch.utils.data.distributed.DistributedSampler
    train_dl = training_utils.construct_dataloader(train_objs['train_ds'],
                                                   train_desc,
                                                   sampler)
    test_dl = training_utils.construct_dataloader(train_objs['test_ds'],
                                                  train_desc,
                                                  sampler)

    trainer = training_utils.Trainer(model,
                                     train_dl,
                                     test_dl,
                                     train_desc.optimizer,
                                     train_desc.learning_rate,
                                     train_desc.scheduler,
                                     rank,
                                     train_desc.save_every,
                                     train_desc.savepath)
    trainer.train(train_desc.epochs, resume=train_desc.resume)

    num_test_samples, hl = train_objs['test_ds'].aij_series.shape[:2]
    new_batch_size = int(num_test_samples/train_desc.world_size)+1
    per_gpu_sample_size = [min(num_test_samples-i*new_batch_size, new_batch_size) for i in range(train_desc.world_size)]
    # generate data for posteriori evaluation, aij(T=100\tau=1000dt)
    T = 1000
    save_every = 200
    model.eval()
    with torch.no_grad(): # Disable gradient calculations
        start_ind = sum(per_gpu_sample_size[:rank])
        end_ind = sum(per_gpu_sample_size[:rank+1])
        posteriori_result = model.module.forward_eval(train_objs['test_ds'].aij_series[start_ind:end_ind,...], T, rank)[:, hl:save_every:T+hl, ...]
        torch.save({f"aij_{rank}": posteriori_result.to('cpu')},
                   train_desc.savepath + f"/aij_posteriori_eval_{rank}.pt")

    ################# THIS MUST BE LAST CALL ####################
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(prog='LATN',
                                     epilog='Good luck!')
    parser.add_argument('-dp', '--datapath', 
                        help="path to directory containing data files, e.g., aij.bin", type=str)
    parser.add_argument('-sp', '--savepath', help="path to directory to save the trained model in", type=str)
    parser.add_argument('-hl', '--history_length', help="length of Lagrangian history", type=int, default=50)
    parser.add_argument('-ht', '--history_timestep', help="multiple of DNS timestep seperating history snapshots", type=int, default=1)
    parser.add_argument('-pt', '--percent_test', help="percentage of samples to reserve for testing", type=float)
    parser.add_argument('-nl', '--num_layers', help="number of hidden layers", type=int, default=3)
    parser.add_argument('-nu', '--num_units', help="number of units per hidden layer in the ff portion", type=int, default=30)
    parser.add_argument('-nf', '--num_filters', help="number of convolutional filters for Lagrangian attention", type=int)
    parser.add_argument('-me', '--max_epochs', help="max number of training epochs", type=int, default=200)
    parser.add_argument('-dr', '--dropout_rate', help="dropout rate of dropout layers", type=float, default=0.0)
    parser.add_argument('-lr', '--learning_rate', help="initial learning rate of optimizer", type=float, default=0.3)
    parser.add_argument('-ns', '--num_samples', help="cap on number of trajectories used (default: all)", type=int, default=None)
    parser.add_argument('-bs', '--batch_size', help="override training batch size (default: 1<<17 for pij, 1<<16 for vis)", type=int, default=None)
    parser.add_argument('--resume', help="resume each model from checkpoint_resume.pt in its savepath if present (for spot interruptions)", action='store_true')
    parser.add_argument('-se', '--save_every', help="epochs between checkpoints / resume-state saves (lower for spot resilience)", type=int, default=50)
    parser.add_argument('-rn', '--run_name', help="explicit run name for output namespacing (default: auto-derived config slug)", type=str, default=None)
    parser.add_argument('-rl', '--rollout_len', help="rollout length (in dt) for the neural ODE (NODE) stage", type=int, default=1)

    args = parser.parse_args()
    print(args)

    # Namespace this run by its config so a sweep doesn't clobber checkpoints.
    run_name = args.run_name or run_slug(args)
    base = os.path.join(args.savepath, run_name)
    Path(base + "/ph").mkdir(parents=True, exist_ok=True)
    Path(base + "/vis").mkdir(parents=True, exist_ok=True)
    Path(base + "/node").mkdir(parents=True, exist_ok=True)
    with open(base + "/config.json", "w") as fp:
        json.dump(vars(args), fp, indent=2, default=str)
    print(f"Run outputs -> {base}")

    print("Creating pij dataset")
    PIJ_DATA_DESC = lagrdataset.DataDesc(
        args.datapath,
        (131072, 1000, 3, 3),  # datashape = (num_samples, num_tsteps, 3, 3)
        "pij",  # target_name,
        3e-4,  # dt,
        args.history_timestep,  # history_timestep,
        args.history_length,  # history_length,
        args.percent_test,  # percent_test)
        args.num_samples)  # num_samples

    print("Creating pij network description")
    PIJ_NETWORK_DESC = latn.LATNDesc(
        args.num_layers,  # num_layers
        args.num_units,  # num_units
        torch.nn.ReLU,  # activation
        NUM_INVARIANTS + args.num_filters,  # input_len
        NUM_PIJ_OUTPUTS,  # output_len
        args.dropout_rate)  # dropout_rate

    print("Creating pij train description")
    PIJ_TRAIN_DESC = training_utils.TrainDesc(
        torch.optim.Adam,  # optimizer
        args.learning_rate,  # learning_rate
        torch.optim.lr_scheduler.ReduceLROnPlateau,  # scheduler
        args.max_epochs,  # epochs
        base + "/ph/",  # savepath
        args.batch_size or (1<<17),  # batch_size
        torch.nn.functional.mse_loss,  # loss_fn
        args.save_every,  # save_every
        torch.cuda.device_count(),  # world_size
        args.resume)  # resume

    print("**** Pressure training beginning ****")
    print(PIJ_DATA_DESC.path_to_data)
    torch.multiprocessing.spawn(
        gpu_tangent_learning,
        args=(PIJ_TRAIN_DESC, PIJ_DATA_DESC, PIJ_NETWORK_DESC),
        nprocs=PIJ_TRAIN_DESC.world_size)

    VIS_DATA_DESC = lagrdataset.DataDesc(
        args.datapath,
        (131072, 1000, 3, 3),  # datashape = (num_samples, num_tsteps, 3, 3)
        "vis",  # target_name,
        3e-4,  # dt,
        args.history_timestep,  # history_timestep,
        args.history_length,  # history_length,
        args.percent_test,  # percent_test)
        args.num_samples)  # num_samples
    VIS_NETWORK_DESC = latn.LATNDesc(
        args.num_layers,  # num_layers
        args.num_units,  # num_units
        torch.nn.ReLU,  # activation
        NUM_INVARIANTS + args.num_filters,  # input_len
        NUM_VIS_OUTPUTS,  # output_len
        args.dropout_rate)  # dropout_rate
    VIS_TRAIN_DESC = training_utils.TrainDesc(
        torch.optim.Adam,  # optimizer
        args.learning_rate,  # learning_rate
        torch.optim.lr_scheduler.ReduceLROnPlateau,  # scheduler
        args.max_epochs,  # epochs
        base + "/vis/",  # savepath
        args.batch_size or (1<<16),  # batch_size
        torch.nn.functional.mse_loss,  # loss_fn
        args.save_every,  # save_every
        torch.cuda.device_count(),  # world_size
        args.resume)  # resume
    print(VIS_DATA_DESC.path_to_data)
    torch.multiprocessing.spawn(
        gpu_tangent_learning,
        args=(VIS_TRAIN_DESC, VIS_DATA_DESC, VIS_NETWORK_DESC),
        nprocs=VIS_TRAIN_DESC.world_size)

    # polish using the NODE model: load the freshly-trained ph/vis a-priori
    #  models and fit the joint stochastic neural ODE to VGT trajectories.
    NODE_DATA_DESC = lagrdataset.DataDesc(
        args.datapath,
        (131072, 1000, 3, 3),  # datashape = (num_samples, num_tsteps, 3, 3)
        "dA",  # target_name,
        3e-4,  # dt,
        args.history_timestep,  # history_timestep,
        args.history_length,  # history_length,
        args.percent_test,  # percent_test
        args.num_samples,  # num_samples
        args.rollout_len)  # rollout_len
    NODE_NETWORK_DESC = latn.LATNDesc(
        args.num_layers,  # num_layers
        args.num_units,  # num_units
        torch.nn.ReLU,  # activation
        NUM_INVARIANTS + args.num_filters,  # input_len
        args.num_filters,  # output_len (overridden per sub-model in load_node_train_objs)
        args.dropout_rate)  # dropout_rate
    NODE_TRAIN_DESC = training_utils.TrainDesc(
        torch.optim.Adam,  # optimizer
        1e-5,  # learning_rate (NODE polish uses a small lr)
        torch.optim.lr_scheduler.ReduceLROnPlateau,  # scheduler
        args.max_epochs,  # epochs
        base + "/node/",  # savepath
        args.batch_size or (1<<14),  # batch_size
        torch.nn.functional.mse_loss,  # loss_fn
        args.save_every,  # save_every
        torch.cuda.device_count(),  # world_size
        args.resume)  # resume
    print("**** NODE training beginning ****")
    torch.multiprocessing.spawn(
        gpu_node_learning,
        args=(NODE_TRAIN_DESC, NODE_DATA_DESC, NODE_NETWORK_DESC),
        nprocs=NODE_TRAIN_DESC.world_size)
