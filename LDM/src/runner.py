import os
import argparse
from pathlib import Path
import torch
import utils
import lagrdataset
import latn
import training_utils
import distributed
import latn_globals
from dataclasses import replace


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
    trainer.train(train_desc.epochs)

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
    trainer.train(train_desc.epochs)
    
    num_test_samples,hl = train_objs['test_ds'].aij_series.shape[:2]
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
    trainer.train(train_desc.epochs)

    if rank == 0: #only save on one process
        torch.save(model.module.state_dict(), train_desc.savepath + "apriori_model_state_dict.pt")
    
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

if __name__ == "__main__":
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
    parser.add_argument('-rl', '--rollout_len', help="rollout length for neural ODE", type=int, default=1)

    args = parser.parse_args()
    savepath = args.savepath + \
        f"hl{args.history_length}_" + \
        f"ht{args.history_timestep}_" +\
        f"nl{args.num_layers}_" +\
        f"nu{args.num_units}_" +\
        f"nf{args.num_filters}_" +\
        f"lr{args.learning_rate}/"
    print(savepath)
    Path(savepath).mkdir(parents=True, exist_ok=True)

    DATA_DESC = lagrdataset.DataDesc(
        args.datapath,
        (131072, 1000, 3, 3),  # datashape = (num_samples, num_tsteps, 3, 3)
        "",  # target_name,
        3e-4,  # dt,
        args.history_timestep,  # history_timestep,
        args.history_length,  # history_length,
        args.percent_test,  # percent_test
        args.rollout_len) # rollout_len
    NETWORK_DESC = latn.LATNDesc(
        args.num_layers,  # num_layers
        args.num_units,  # num_units
        torch.nn.ReLU,  # activation
        latn_globals.NUM_INVARIANTS + args.num_filters,  # input_len
        latn_globals.NUM_PIJ_OUTPUTS,  # output_len
        args.dropout_rate)  # dropout_rate
    TRAIN_DESC = training_utils.TrainDesc(
        torch.optim.Adam,  # optimizer
        args.learning_rate, # learning rate
        torch.optim.lr_scheduler.ReduceLROnPlateau,  # scheduler
        100,#args.max_epochs,  # epochs
        savepath,  # savepath
        1<<17,  # batch_size
        torch.nn.functional.mse_loss,  # loss_fn
        50,  # save_every
        torch.cuda.device_count())  # world_size

    PIJ_DATA_DESC = replace(DATA_DESC,
                            target_name="pij")
    PIJ_NETWORK_DESC = replace(NETWORK_DESC,
                               output_len=latn_globals.NUM_PIJ_OUTPUTS)
    PIJ_TRAIN_DESC = replace(TRAIN_DESC,
                             savepath=savepath + "/ph/")
    Path(PIJ_TRAIN_DESC.savepath).mkdir(parents=True, exist_ok=True)
    # torch.multiprocessing.spawn(
    #     gpu_tangent_learning,
    #     args=(PIJ_TRAIN_DESC, PIJ_DATA_DESC, PIJ_NETWORK_DESC),
    #     nprocs=PIJ_TRAIN_DESC.world_size)

    VIS_DATA_DESC = replace(DATA_DESC,
                            target_name="vis")
    VIS_NETWORK_DESC = replace(NETWORK_DESC,
                               output_len=latn_globals.NUM_VIS_OUTPUTS)
    VIS_TRAIN_DESC = replace(TRAIN_DESC,
                             savepath=savepath + "/vis/")
    Path(VIS_TRAIN_DESC.savepath).mkdir(parents=True, exist_ok=True)
    # torch.multiprocessing.spawn(
    #     gpu_tangent_learning,
    #     args=(VIS_TRAIN_DESC, VIS_DATA_DESC, VIS_NETWORK_DESC),
    #     nprocs=VIS_TRAIN_DESC.world_size)

    # polish using NODE model
    NODE_DATA_DESC = replace(DATA_DESC,
                             target_name="dA")
    NODE_NETWORK_DESC = replace(NETWORK_DESC,
                                output_len=args.num_filters)
    NODE_TRAIN_DESC = replace(TRAIN_DESC,
                              savepath=savepath + "/node/",
                              batch_size=1<<14,
                              learning_rate=1e-5,
                              epochs=10)
    Path(NODE_TRAIN_DESC.savepath).mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(
        gpu_node_learning,
        args=(NODE_TRAIN_DESC, NODE_DATA_DESC, NODE_NETWORK_DESC),
        nprocs=NODE_TRAIN_DESC.world_size)


    # generate data for posteriori evaluation, just aij(t=1000dt)?
