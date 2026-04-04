import argparse;
import numpy as np;
import os;
from pathlib import Path
import torch;
from torch import nn;
from torch.utils.data import Dataset, DataLoader;
import matplotlib.pyplot as plt
from collections import OrderedDict
from utils import *

class TBNNDataset(Dataset):
    def __init__(self, invars, tb, target):
        self.invars = invars;
        self.tb = tb;
        self.target = target;
        self.numSamples = invars.shape[0] if (len(invars.shape) > 1) else 1
        self.normalized = True;

    def __len__(self):
        return self.numSamples;

    def __getitem__(self, idx):
        return TBNNDataset(self.invars[idx,:], self.tb[idx,:,:,:], self.target[idx,:,:])
    
    def pin_memory(self):
        self.invars = self.invars.pin_memory()
        self.tb = self.tb.pin_memory()
        self.target = self.target.pin_memory()
        
    def to(self, device):
        self.invars = self.invars.to(device);
        self.tb = self.tb.to(device);
        self.target = self.target.to(device);
        
    def __sizeof__(self):
        bytes_in_invars = self.invars.element_size()*np.prod(self.invars.shape);
        bytes_in_tb = self.tb.element_size()*np.prod(self.tb.shape);
        bytes_in_target = self.target.element_size()*np.prod(self.target.shape);
        return bytes_in_invars+bytes_in_tb+bytes_in_target;
    
    def shuffle(self):
        shuffled_inds = np.random.choice(self.numSamples, self.numSamples, replace=False);
        self.invars = self.invars[shuffled_inds, :];
        self.tb = self.tb[shuffled_inds, :, :, :];
        self.target = self.target[shuffled_inds, :, :];

        

def create_ph_datasets(datapath, N=131072, num_train_folds=3, num_test_folds=2, numSamples=131072, numTestSamples=131072, T=1000, timestep=10, M=200, dt=3e-4):
    aij = np.fromfile(datapath + "aij_1024_dns.bin");
    aij = aij.reshape([N, T, 3, 3]);

    pij = np.fromfile(datapath + "pij_1024_dns.bin");
    pij = pij.reshape([N, T, 3, 3]);

    if (( num_train_folds + num_test_folds)*M > T):
        print('algorithm isn\'t smart enough..')
        print('folds reset to 1')
        num_train_folds = 1
        num_test_folds = 1
    
    train_inds = [range(M*i, M*(i+1), timestep) for i in range(num_train_folds)];
    train_pij = pij[:, train_inds[0][-1], :, :];
    train_aij = aij[:, train_inds[0][-1], :, :];
    for i in range(1, len(train_inds)):
        train_pij = np.concatenate([train_pij, pij[:, train_inds[i][-1], :,:]], axis=0);
        train_aij = np.concatenate([train_aij, aij[:, train_inds[i][-1], :,:]], axis=0);

    test_inds = [range(T-M*(i+1), T-M*i, timestep) for i in range(num_test_folds)];
    test_pij = pij[:, test_inds[0][-1], :, :];
    test_aij = aij[:, test_inds[0][-1], :, :];
    for i in range(1, len(test_inds)):
        test_pij = np.concatenate([test_pij, pij[:, test_inds[i][-1], :, :]], axis=0);
        test_aij = np.concatenate([test_aij, aij[:, test_inds[i][-1], :, :]], axis=0);
    test_pij = torch.tensor(test_pij);
    test_aij = torch.tensor(test_aij);

    # convert to torch objects
    train_aij = torch.tensor(train_aij);
    train_pij = torch.tensor(train_pij);

    # remove trace from PH
    train_pij = train_pij - (1/3) * torch.einsum('i,ijk->ijk', torch.einsum('nii->n', train_pij), torch.diag_embed(torch.ones(train_pij.shape[0], 3)));
    test_pij = test_pij - (1/3) * torch.einsum('i,ijk->ijk', torch.einsum('nii->n', test_pij), torch.diag_embed(torch.ones(test_pij.shape[0], 3)));

    # normalize inputs
    tau = calcCharacteristicTimescale(train_aij);
    train_aij = train_aij*tau;
    test_aij = test_aij*tau;

    # calculate tb/invariants from normalized data
    train_invars = calcInvariants(train_aij).clone().detach();
    test_invars = calcInvariants(test_aij).clone().detach();
    train_tb = calcSymTensorBasis(train_aij).clone().detach();
    test_tb = calcSymTensorBasis(test_aij).clone().detach();

    train_ds = TBNNDataset(train_invars, train_tb, train_pij);
    test_ds = TBNNDataset(test_invars, test_tb, test_pij);

    return train_ds, test_ds;

def create_vis_datasets(datapath, N=131072, num_train_folds=3, num_test_folds=2, numSamples=131072, numTestSamples=131072, T=1000, timestep=10, M=200, dt=3e-4):
    aij = np.fromfile(datapath + "aij_1024_dns.bin");
    aij = aij.reshape([N, T, 3, 3]);

    vis = np.fromfile(datapath + "vis_1024_dns.bin");
    vis = vis.reshape([N, T, 3, 3]);

    if ( (num_train_folds + num_test_folds) * M > T):
        print('algorithm is not smart enough to handle that many folds, TEST AND TRAIN DATA WILL OVERLAP!');
        print('num_train_folds set to 1, num_test_folds set to 1')
        num_train_folds = 1;
        num_test_folds = 1

    train_inds = [range(M*i, M*(i+1), timestep) for i in range(num_train_folds)];
    train_vis = vis[:, train_inds[0][-1], :, :];
    train_aij = aij[:, train_inds[0][-1], :, :];
    for i in range(1, len(train_inds)):
        train_vis = np.concatenate([train_vis, vis[:, train_inds[i][-1], :,:]], axis=0);
        train_aij = np.concatenate([train_aij, aij[:, train_inds[i][-1], :,:]], axis=0);
        
    test_inds = [range(T-M*(i+1), T-M*i, timestep) for i in range(num_test_folds)];
    test_vis = vis[:, test_inds[0][-1], :, :];
    test_aij = aij[:, test_inds[0][-1], :, :];
    for i in range(1, len(test_inds)):
        test_vis = np.concatenate([test_vis, vis[:, test_inds[i][-1], :, :]], axis=0);
        test_aij = np.concatenate([test_aij, aij[:, test_inds[i][-1], :, :]], axis=0);
    test_vis = torch.tensor(test_vis);
    test_aij = torch.tensor(test_aij);

    # convert to torch objects
    train_aij = torch.tensor(train_aij);
    train_vis = torch.tensor(train_vis);

    # normalize inputs
    tau = calcCharacteristicTimescale(train_aij);
    train_aij = train_aij*tau;
    test_aij = test_aij*tau;

    # calculate tb/invariants from normalized data
    train_invars = calcInvariants(train_aij[:,:,:]).clone().detach();
    test_invars = calcInvariants(test_aij[:,:,:]).clone().detach();
    train_tb = calcFullTensorBasis(train_aij[:,:,:]).clone().detach();
    test_tb = calcFullTensorBasis(test_aij[:,:,:]).clone().detach();
    
    train_ds = TBNNDataset(train_invars, train_tb, train_vis);
    test_ds = TBNNDataset(test_invars, test_tb, test_vis);

    return train_ds, test_ds;


class TBNN(nn.Module):
    def __init__(self, num_layers=1, num_units=10, activation=nn.ReLU(), num_invars=5, num_tb=10, device='cuda', dropout_rate=0.2):
        super().__init__()
        self.layers = OrderedDict()
        self.layers['lin1'] = nn.Linear(num_invars, num_units)
        self.layers['act1'] = activation
        for i in range(1,num_layers):
            self.layers['lin'+str(i+1)] = nn.Linear(num_units, num_units);
            self.layers['act'+str(i+1)] = activation;
        self.layers['lin'+str(num_layers+1)] = nn.Linear(num_units, num_tb)
        self.ff = nn.Sequential(self.layers)
        self.to(device)
            
    def forward(self, _invars, _tb):
        g = self.ff(_invars);
        return torch.einsum('ij,iklj->ikl', g, _tb);

def train_network(train_ds, test_ds, model, optimizer, scheduler, epochs, savepath, device='cuda', batch_size = 1<<10):
    def train_loop(_train_ds, _opt, _loss_fn, _batch_size):
        num_batches = len(_train_ds)/_batch_size;
        epoch_loss = 0.0
        for i in range(0, len(_train_ds), _batch_size):
            pred = model(_train_ds.invars[i:i+_batch_size, :], _train_ds.tb[i:i+_batch_size,:,:,:]);
            loss = _loss_fn(pred, _train_ds.target[i:i+_batch_size,:,:]);
            _opt.zero_grad()
            loss.backward()
            _opt.step()
            epoch_loss += loss;
        return epoch_loss/num_batches;
    def test_loop(_test_ds, _loss_fn):
        pred = model(_test_ds.invars, _test_ds.tb);
        return _loss_fn(pred, _test_ds.target);
    
    train_ds.to(device);
    test_ds.to(device);
    
    baseline_loss = nn.functional.mse_loss(torch.zeros(test_ds.target.shape).to(device), test_ds.target);
    print(f"baseline loss = {baseline_loss:>4f}");
    def loss_fn(pred, gt):
        return nn.functional.mse_loss(pred, gt)/baseline_loss;
    train_loss = np.zeros(epochs);
    test_loss = np.zeros(epochs);
    minLoss = np.inf;
    best_result_path = savepath + 'intermed_result.pt';

    for i in range(epochs):
        train_ds.shuffle()
        train_loss[i] = train_loop(train_ds, optimizer, loss_fn, batch_size);
        with torch.no_grad():
            model.eval()
            test_loss[i] = test_loop(test_ds, loss_fn);
            model.train()
            scheduler.step(test_loss[i])
    
        if (test_loss[i] < minLoss):
            save_trained_model(best_result_path, model, optimizer, train_loss, test_loss);
            minLoss = test_loss[i]
            print(f"new min achieved, epoch = {i}, test loss = {test_loss[i]:>4f}, train loss = {train_loss[i]:>4f}")

        if (i % 50 == 0 and i > 0):
            print(f"test loss = {np.mean(test_loss[i-50:i]):>4f}, loss = {np.mean(train_loss[i-50:i]):>4f}, epoch = {i}")
            if (i > 3000 and np.min(test_loss[range(i-1000,i)]) > (1 - 1e-5)*np.min(test_loss[range(i-1000)])):
                train_loss = train_loss[range(i)];
                test_loss = test_loss[range(i)];
                print(f"early termination at epoch = {i}");
                break;
    print(f"finished training, best test loss = {minLoss:>4f}");
    model.load_state_dict(torch.load(best_result_path)['model'].state_dict()); # load best epoch params
    save_trained_model(savepath, model, optimizer, train_loss, test_loss, train_dataset = train_ds, test_dataset = test_ds);
    return;

def train_tbnn_ph(args):
    train_ds, test_ds = create_ph_datasets(args.datapath)
    net = TBNN(num_layers=args.num_layers, num_units=args.num_units, dropout_rate=args.dropout_rate).double()
    savepath = args.savepath + f"/tbnn_ph_nl{args.num_layers}_nu{args.num_units}_dr{args.dropout_rate}.pt"
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-7, verbose=True)
    train_network(train_ds, test_ds, net, optimizer, scheduler, 1000, savepath, 'cuda')

def train_tbnn_vis(args):
    savepath = args.savepath + f"/tbnn_vis_nl{args.num_layers}_nu{args.num_units}_dr{args.dropout_rate}.pt"
    train_ds, test_ds = create_vis_datasets(args.datapath)
    net = TBNN(num_layers=args.num_layers, num_units=args.num_units, dropout_rate=args.dropout_rate, num_tb=16).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True)
    train_network(train_ds, test_ds, net, optimizer, scheduler, args.max_epochs, savepath, 'cuda')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TBNN',
                                     description='Training utilities for the tensor basis model',
                                     epilog='Good luck!')
    parser.add_argument('-me', '--max_epochs', help="max number of training epochs", type=int, default=200)
    parser.add_argument('-dp', '--datapath', 
                        help="path to directory containing data files, e.g., aij.bin", type=str)
    parser.add_argument('-sp', '--savepath', help="path to directory to save the trained model in", type=str)
    parser.add_argument('-nu', '--num_units', help="number of units per hidden layer in the ff portion", type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', help="initial learning rate of optimizer", type=float, default=0.3)
    parser.add_argument('-nl', '--num_layers', help="number of hidden layers", type=int, default=3)
    parser.add_argument('-dr', '--dropout_rate', help="dropout rate of dropout layers", type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    if (args.savepath):
        Path(args.savepath).mkdir(parents=True, exist_ok=True)

    train_tbnn_ph(args)
    train_tbnn_vis(args)

