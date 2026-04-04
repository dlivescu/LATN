import argparse;
import numpy as np;
import os;
from pathlib import Path
import torch;
from torch import nn;
from torch.utils.data import Dataset, DataLoader;
from collections import OrderedDict
from utils import *


class LDM_TBNN_Dataset(Dataset):
    def __init__(self, aij_series, invars, tb, target, timescale, history_length, history_timestep):
        self.aij_series = aij_series.flatten(start_dim=2);
        self.invars = invars;
        self.tb = tb;
        self.target = target;
        self.numSamples = invars.shape[0] if (len(invars.shape) > 1) else 1
        self.normalized = True;
        self.normalization = timescale;
        self.history_length = history_length;
        self.history_timestep = history_timestep;

    def __len__(self):
        return self.numSamples;
            
    def to(self, device):
        self.aij_series = self.aij_series.to(device);
        self.invars = self.invars.to(device);
        self.tb = self.tb.to(device);
        self.target = self.target.to(device);
        
    def concatenate(self, ds):
        aij_series = torch.cat((self.aij_series, ds.aij_series), dim=0);
        invars = torch.cat((self.invars, ds.invars), dim=0);
        tb = torch.cat((self.tb, ds.tb), dim=0);
        target = torch.cat((self.target, ds.target), dim=0);
        return LDM_TBNN_Dataset(aij_series, invars, tb, target, self.normalization, self.history_length, self.history_timestep);

    def rotate(self):
        pi = torch.tensor(np.pi)
        alpha = torch.rand(1)*2*pi;
        beta = torch.rand(1)*pi;
        gamma = torch.rand(1)*2*pi;
        R_yaw = torch.tensor([[torch.cos(alpha), -torch.sin(alpha), 0], [torch.sin(alpha), torch.cos(alpha), 0], [0, 0, 1]])
        R_pitch = torch.tensor([[torch.cos(beta), 0, torch.sin(beta)], [0,1,0], [-torch.sin(beta), 0, torch.cos(beta)]])
        R_roll = torch.tensor([[1, 0, 0], [0, torch.cos(gamma), -torch.sin(gamma)], [0, torch.sin(gamma), torch.cos(gamma)]])
        R = torch.einsum('ij,jk,kl',R_yaw, R_pitch, R_roll).double();

        aij_series = torch.einsum('ij,ntjk->ntik',R, self.aij_series.reshape(self.aij_series.shape[0], self.aij_series.shape[1], 3,3)).flatten(start_dim=2)
        tb = torch.einsum('ij,njkm,kl->nilm', R, self.tb, R.transpose(0,1))
        target = torch.einsum('ij,njk,kl->nil', R, self.target, R.transpose(0,1))
        return LDM_TBNN_Dataset(aij_series, self.invars, tb, target, self.normalization, self.history_length, self.history_timestep);
        # def __init__(self, aij_series, invars, tb, target, timescale, history_length, history_timestep):
        # self.aij_series = torch.einsum('ij,ntjk->ntik',R, self.aij_series.reshape(self.aij_series.shape[0], self.aij_series.shape[1], 3,3)).flatten(start_dim=2)
        # self.tb = torch.einsum('ij,njkm,kl->nilm', R, self.tb, R.transpose(0,1))
        # self.target = torch.einsum('ij,njk,kl->nil', R, self.target, R.transpose(0,1))

        
    def __sizeof__(self):
        bytes_in_aij_series = self.aij_series.element_size()*np.prod(self.aij_series.shape);
        bytes_in_invars = self.invars.element_size()*np.prod(self.invars.shape);
        bytes_in_tb = self.tb.element_size()*np.prod(self.tb.shape);
        bytes_in_target = self.target.element_size()*np.prod(self.target.shape);
        return bytes_in_aij_series+bytes_in_invars+bytes_in_tb+bytes_in_target;
    
    def shuffle(self):
        num_samples = self.aij_series.shape[0]
        shuffled_inds = np.random.choice(num_samples, num_samples, replace=False);
        if (self.aij_series.shape[0] == num_samples):
            self.aij_series = self.aij_series[shuffled_inds, :, :];
        if (self.invars.shape[0] == num_samples):
            self.invars = self.invars[shuffled_inds, :];
        if (self.tb.shape[0] == num_samples):
            self.tb = self.tb[shuffled_inds, :, :, :];
        if (self.target.shape[0] == num_samples):
            self.target = self.target[shuffled_inds, :, :];
        
def create_ph_datasets(datapath, N=131072, num_train_folds=3, num_test_folds=2, T=1000, timestep=10, M=200, dt=3e-4):
    aij = np.fromfile(datapath + "aij_1024_dns.bin");
    aij = aij.reshape([N, T, 3, 3]);

    pij = np.fromfile(datapath + "pij_1024_dns.bin");
    pij = pij.reshape([N, T, 3, 3])

    if (( num_train_folds + num_test_folds)*M > T):
        print('algorithm isn\'t smart enough..')
        print('folds reset to 1')
        num_train_folds = 1
        num_test_folds = 1

    train_inds = [range(M*i, M*(i+1), timestep) for i in range(num_train_folds)];
    train_pij = pij[:, train_inds[0][-1], :, :];
    train_aij = aij[:, train_inds[0], :, :];
    for i in range(1, len(train_inds)):
        train_pij = np.concatenate([train_pij, pij[:, train_inds[i][-1], :,:]], axis=0);
        train_aij = np.concatenate([train_aij, aij[:, train_inds[i], :,:]], axis=0);
        
    test_inds = [range(T-M*(i+1), T-M*i, timestep) for i in range(num_test_folds)];
    test_pij = pij[:, test_inds[0][-1], :, :];
    test_aij = aij[:, test_inds[0], :, :];
    for i in range(1, len(test_inds)):
        test_pij = np.concatenate([test_pij, pij[:, test_inds[i][-1], :, :]], axis=0);
        test_aij = np.concatenate([test_aij, aij[:, test_inds[i], :, :]], axis=0);
    test_pij = torch.tensor(test_pij);
    test_aij = torch.tensor(test_aij);

    # convert to torch objects
    train_aij = torch.tensor(train_aij);
    train_pij = torch.tensor(train_pij);

    # remove trace from PH
    train_pij = train_pij - (1/3) * torch.einsum('i,ijk->ijk', torch.einsum('nii->n', train_pij), torch.diag_embed(torch.ones(train_pij.shape[0], 3)));
    test_pij = test_pij - (1/3) * torch.einsum('i,ijk->ijk', torch.einsum('nii->n', test_pij), torch.diag_embed(torch.ones(test_pij.shape[0], 3)));

    def get_good_inds(arr, cutoff):
        def bad_cond(_x):
            is_too_big = torch.abs(_x) > cutoff
            return is_too_big
        def good_cond(_x):
            return torch.logical_not(bad_cond(_x));
        flat_good = torch.flatten(good_cond(arr))
        good_inds = torch.nonzero(flat_good).flatten()
        return good_inds

    train_aij_sqnorm_arr = torch.einsum('nij,nij->n',train_aij[:,-1,:,:],train_aij[:,-1,:,:])
    test_aij_sqnorm_arr = torch.einsum('nij,nij->n',test_aij[:,-1,:,:],test_aij[:,-1,:,:])
    cutoff = torch.mean(train_aij_sqnorm_arr) + 10*torch.std(train_aij_sqnorm_arr)

    train_good_inds = get_good_inds(train_aij_sqnorm_arr, cutoff);
    test_good_inds = get_good_inds(test_aij_sqnorm_arr, cutoff);
    print(f"original length of aij = {train_aij.shape[0]}")
    print(f"new length of aij = {len(train_good_inds)}")

    train_aij = train_aij[train_good_inds,...]
    train_pij = train_pij[train_good_inds,...]
    test_aij = test_aij[test_good_inds,...]
    test_pij = test_pij[test_good_inds,...]

    # normalize inputs
    tau = calc_characteristic_timescale(train_aij[:,0,:,:]);
    print(tau);
    train_aij = train_aij*tau;
    test_aij = test_aij*tau;

    # calculate tb/invariants from normalized data
    train_invars = calcInvariants(train_aij[:,-1,:,:]).clone().detach();
    test_invars = calcInvariants(test_aij[:,-1,:,:]).clone().detach();
    train_tb = calcSymTensorBasis(train_aij[:,-1,:,:]).clone().detach();
    test_tb = calcSymTensorBasis(test_aij[:,-1,:,:]).clone().detach();

    train_ds = LDM_TBNN_Dataset(train_aij, train_invars, train_tb, train_pij, tau, M, timestep*dt);
    test_ds = LDM_TBNN_Dataset(test_aij, test_invars, test_tb, test_pij, tau, M, timestep*dt);

    # for i in range(2):
    #     rot_ds = train_ds.rotate();
    #     train_ds = train_ds.concatenate(rot_ds);
    # for i in range(2):
    #     rot_ds = test_ds.rotate();
    #     test_ds = test_ds.concatenate(rot_ds);

    print(train_ds.invars.shape)

    return train_ds, test_ds;

def create_dA_datasets(datapath, N=131072, num_train_folds=3, num_test_folds=2, num_tsteps=5, T=1000, timestep=10, M=200, dt=3e-4):
    aij = np.fromfile(datapath + "aij_1024_dns.bin");
    aij = aij.reshape([N, T, 3, 3]);

    stride = M + num_tsteps;
    if (( num_train_folds + num_test_folds)*(stride) > T):
        print('algorithm isn\'t smart enough..')
        print('folds reset to 1')
        num_train_folds = 1
        num_test_folds = 1

    train_inds = [range(stride*i, stride*(i+1), 1) for i in range(num_train_folds)];
    train_aij = aij[:, train_inds[0], :, :];
    for i in range(1, len(train_inds)):
        train_aij = np.concatenate([train_aij, aij[:, train_inds[i], :,:]], axis=0);
        
    test_inds = [range(T-stride*(i+1), T-stride*i, 1) for i in range(num_test_folds)];
    #debug
    print(train_inds)
    print(test_inds)
    test_aij = aij[:, test_inds[0], :, :];
    for i in range(1, len(test_inds)):
        test_aij = np.concatenate([test_aij, aij[:, test_inds[i], :, :]], axis=0);

    # convert to torch objects
    train_aij = torch.tensor(train_aij);
    test_aij = torch.tensor(test_aij);

    # remove outliers or loss will go to inf in few steps
    def get_good_inds(arr, cutoff):
        def bad_cond(_x):
            is_too_big = torch.abs(_x) > cutoff
            return is_too_big
        def good_cond(_x):
            return torch.logical_not(bad_cond(_x));
        flat_good = torch.flatten(good_cond(arr))
        good_inds = torch.nonzero(flat_good).flatten()
        return good_inds

    train_aij_sqnorm_arr = torch.einsum('nij,nij->n',train_aij[:,-1,:,:],train_aij[:,-1,:,:])
    test_aij_sqnorm_arr = torch.einsum('nij,nij->n',test_aij[:,-1,:,:],test_aij[:,-1,:,:])
    cutoff = torch.mean(train_aij_sqnorm_arr) + 5*torch.std(train_aij_sqnorm_arr)

    train_good_inds = get_good_inds(train_aij_sqnorm_arr, cutoff);
    test_good_inds = get_good_inds(test_aij_sqnorm_arr, cutoff);
    print(f"original length of aij = {train_aij.shape[0]}")
    print(f"new length of aij = {len(train_good_inds)}")
    train_aij = train_aij[train_good_inds,...]
    test_aij = test_aij[test_good_inds,...]

    print(test_aij.shape)
    # split last num_tsteps into target dataset
    train_target = train_aij[:,-num_tsteps:,:,:]
    test_target = test_aij[:,-num_tsteps:,:,:]
    train_aij = train_aij[:,:-num_tsteps,:,:]
    test_aij = test_aij[:,:-num_tsteps,:,:]
    print(test_aij.shape)

    empty = torch.tensor([])
    train_ds = LDM_TBNN_Dataset(train_aij, empty, empty, train_target, empty, empty, empty);
    test_ds = LDM_TBNN_Dataset(test_aij, empty, empty, test_target, empty, empty, empty);

    return train_ds, test_ds;


def create_vis_datasets(datapath, N=131072, num_train_folds=3, num_test_folds=2, T=1000, timestep=10, M=291, dt=3e-4):
    aij = np.fromfile(datapath + "aij_1024_dns.bin");
    aij = aij.reshape([N, T, 3, 3]);

    vis = np.fromfile(datapath + "vis_1024_dns.bin");
    vis = vis.reshape([N, T, 3, 3])

    if ( (num_train_folds + num_test_folds) * M > T):
        print('algorithm is not smart enough to handle that many folds, TEST AND TRAIN DATA WILL OVERLAP!');
        print('num_train_folds set to 1, num_test_folds set to 1')
        num_train_folds = 1;
        num_test_folds = 1

    train_inds = [range(M*i, M*(i+1), timestep) for i in range(num_train_folds)];
    train_vis = vis[:, train_inds[0][-1], :, :];
    train_aij = aij[:, train_inds[0], :, :];
    for i in range(1, len(train_inds)):
        train_vis = np.concatenate([train_vis, vis[:, train_inds[i][-1], :,:]], axis=0);
        train_aij = np.concatenate([train_aij, aij[:, train_inds[i], :,:]], axis=0);
        
    test_inds = [range(T-M*(i+1), T-M*i, timestep) for i in range(num_test_folds)];
    test_vis = vis[:, test_inds[0][-1], :, :];
    test_aij = aij[:, test_inds[0], :, :];
    for i in range(1, len(test_inds)):
        test_vis = np.concatenate([test_vis, vis[:, test_inds[i][-1], :, :]], axis=0);
        test_aij = np.concatenate([test_aij, aij[:, test_inds[i], :, :]], axis=0);
    test_vis = torch.tensor(test_vis);
    test_aij = torch.tensor(test_aij);

    # convert to torch objects
    train_aij = torch.tensor(train_aij);
    train_vis = torch.tensor(train_vis);

    def get_good_inds(vis, cutoff):
        def bad_cond(_x):
            is_too_big = torch.abs(_x) > cutoff
            return is_too_big
        def good_cond(_x):
            return torch.logical_not(bad_cond(_x));
        flat_good = torch.flatten(good_cond(vis))
        good_inds = torch.nonzero(flat_good).flatten()
        return good_inds

    train_vis_sqnorm_arr = torch.einsum('nij,nij->n',train_vis,train_vis)
    test_vis_sqnorm_arr = torch.einsum('nij,nij->n',test_vis,test_vis)
    cutoff = torch.mean(train_vis_sqnorm_arr) + 10*torch.std(train_vis_sqnorm_arr)

    train_good_inds = get_good_inds(train_vis_sqnorm_arr, cutoff);
    test_good_inds = get_good_inds(test_vis_sqnorm_arr, cutoff);

    # train_aij = train_aij[train_good_inds,...]
    # train_vis = train_vis[train_good_inds,...]
    # test_aij = test_aij[test_good_inds,...]
    # test_vis = test_vis[test_good_inds,...]


    # normalize inputs
    tau = calc_characteristic_timescale(train_aij[:,0,:,:]);
    train_aij = train_aij*tau;
    test_aij = test_aij*tau;

    # calculate tb/invariants from normalized data
    train_invars = calcInvariants(train_aij[:,-1,:,:]).clone().detach();
    test_invars = calcInvariants(test_aij[:,-1,:,:]).clone().detach();
    train_tb = calcFullTensorBasis(train_aij[:,-1,:,:]).clone().detach();
    test_tb = calcFullTensorBasis(test_aij[:,-1,:,:]).clone().detach();
    
    train_ds = LDM_TBNN_Dataset(train_aij, train_invars, train_tb, train_vis, tau, M, timestep*dt);
    test_ds = LDM_TBNN_Dataset(test_aij, test_invars, test_tb, test_vis, tau, M, timestep*dt);

    return train_ds, test_ds;

class RicherTensorConv(nn.Module):
    def __init__(self, _num_tsteps, _num_filters=3):
        super().__init__();
        self.ps = torch.nn.Parameter(torch.rand(_num_tsteps, _num_filters, 9));
    def forward(self, x):
        #i->samples, j->timestep, m->filters, k->tensor entry
        # returns a (i,m) tensor
        # num_steps = self.ps.shape[0]
        # num_filters = self.ps.shape[1]
        # num_sym_filters = int(num_filters/2)
        # S = self.ps[:,:num_sym_filters,...].reshape(num_steps, num_sym_filters, 3,3)
        # W = self.ps[:,num_sym_filters:,...].reshape(num_steps, num_filters-num_sym_filters, 3,3)
        # sym_kernels = (0.5*(S + S.transpose(-2,-1))).reshape(num_steps, num_sym_filters, 9);
        # asym_kernels = (0.5*(W - W.transpose(-2,-1))).reshape(num_steps, num_filters-num_sym_filters, 9);
        # return nn.functional.sigmoid(torch.einsum('ijk,jmk->im',x,torch.cat((sym_kernels, asym_kernels), 1)));
        return nn.functional.sigmoid(torch.einsum('ijk,jmk->im',x,self.ps))

class EvenRicherTensorConv(nn.Module):
    def __init__(self, _num_tsteps, _num_filters=3):
        super().__init__();
        self.lin1 = torch.nn.Linear(1, 30)
        self.lin2 = torch.nn.Linear(30, 9);

    def build_kernel(self, t):
        return self.lin2(torch.nn.functional.relu(self.lin1(t))).reshape(3,3)

    def forward(self, x):
        kernels = torch.stack([self.build_kernel(t) for t in ts]);
        
        
            
class TensorConv(nn.Module):
    def __init__(self, _num_tsteps, _num_filters=1):
        super().__init__()
        self.ps = torch.nn.Parameter(torch.rand(_num_tsteps, _num_filters));
        # self.ps.requires_grad = False; #debugging

    def forward(self, x):
        tmp = torch.einsum('ijk,jm->imk', x, self.ps)
        tmp = torch.sum(tmp, 2)
        #return nn.Dropout(0.2)(nn.functional.sigmoid(tmp)); #convolve along temporal dim
        return nn.functional.sigmoid(tmp); #convolve along temporal dim
    
class ConvTBNN(nn.Module):
    def __init__(self, _num_tsteps, history_length, history_dt, num_layers=1, num_units=10, activation=nn.ReLU(), input_len=14, output_len=10, device='cuda', dropout_rate=0.2):
        super().__init__()
        self.history_length = history_length; #M
        self.history_dt = history_dt; # float, timestep*simulation_dt
        
        # use parameters to create layer specification
        self.num_hidden_layers = num_layers;
        self.layers = OrderedDict()
        if (num_layers == 0):
            self.layers['lin1'] = nn.Linear(input_len, output_len);
        else:
            self.layers['lin1'] = nn.Linear(input_len, num_units)
            self.layers['act1'] = activation
            for i in range(1,num_layers):
                self.layers['lin'+str(i+1)] = nn.Linear(num_units, num_units);
                self.layers['act'+str(i+1)] = activation;
#                self.layers['drop'+str(i+1)] = nn.Dropout(dropout_rate);
            current_layer_dim = self.layers[list(self.layers.keys())[-2]].out_features;
            self.layers['lin'+str(num_layers+1)] = nn.Linear(current_layer_dim, output_len);
        # compile spec into function
        self.ff = nn.Sequential(self.layers)
        self.conv = RicherTensorConv(_num_tsteps, input_len-5);
        self.to(device)

    def forward(self, _invars, _timeseries, _tb):
        convolvedTimeseries = self.conv(_timeseries);
        # adding in skip connections
        # layer_count = 0;
        # output = torch.cat([_invars, convolvedTimeseries], dim=1);
        # for submodel in self.ff:
        #     output = submodel.forward(output);
        #     if (issubclass(type(submodel), torch.nn.Dropout)):
        #         layer_count += 1;
        #         if (layer_count == 1):
        #             input = output;
        #         elif (len(self.layers.keys()) > 3 and layer_count > 1 and layer_count < len(self.layers.keys())-2):
        #             output = output + input;
        #             input = output;
        g = self.ff(torch.cat([_invars, convolvedTimeseries], dim=1));
        return torch.einsum('ij,iklj->ikl', g, _tb);
        #return torch.einsum('ij,iklj->ikl', output, _tb);

class NODEConvTBNN(nn.Module):
    def __init__(self, ldm_ph_model, ldm_vis_model, normalization=0.0468):
        super().__init__()
        assert ldm_ph_model.history_length == ldm_vis_model.history_length, "NODE constructor: history lengths different"
        assert ldm_ph_model.history_dt == ldm_vis_model.history_dt, "NODE constructor: history dt different"
        self.ph_model = ldm_ph_model;
        self.vis_model = ldm_vis_model;
        update_freq = int(self.ph_model.history_dt / 3e-4);
        self.time_history_inds = range(0, ldm_ph_model.history_length, update_freq);
        self.normalization = normalization;
        
    def forward(self, aij_series):
        # assume VG unnormalized!
        normalized_aij_series = aij_series*self.normalization
        invars = calcInvariants(normalized_aij_series[:,-1,:,:])
        tb = calcFullTensorBasis(normalized_aij_series[:,-1,:,:])
        # all outputs are in unnormalized space, so use unnormalized for RE
        # returns 'dA'
        return (get_restricted_euler(aij_series[:,-1,:,:]) + 
                get_ldm_dev_ph(self.ph_model, invars, normalized_aij_series[:,self.time_history_inds,...].flatten(start_dim=2), tb[...,:10]) + 
                get_ldm_vis(self.vis_model, invars, normalized_aij_series[:,self.time_history_inds,...].flatten(start_dim=2), tb));


def train_network(train_ds, test_ds, model, optimizer, scheduler, epochs, savepath, device='cuda', batch_size = 1<<9, unnormalized_loss=torch.nn.functional.mse_loss):
    def train_loop(_train_ds, _opt, _loss_fn, _batch_size):
#        _train_ds.rotate();
        num_batches = int(np.ceil(len(_train_ds)/_batch_size));
        epoch_loss = 0.0
        for i in range(0, len(_train_ds), _batch_size):
            pred = model(_train_ds.invars[i:i+_batch_size, :], _train_ds.aij_series[i:i+_batch_size,:,:],_train_ds.tb[i:i+_batch_size,:,:,:]);
            loss = _loss_fn(pred, _train_ds.target[i:i+_batch_size,:,:]) + 0.001*torch.mean(torch.abs(model.conv.ps));
            _opt.zero_grad()
            loss.backward()
            _opt.step()
            epoch_loss += loss;
        return epoch_loss/num_batches;
    def test_loop(_test_ds, _loss_fn):
#        _test_ds.rotate();
        model.eval()
        pred = model(_test_ds.invars, _test_ds.aij_series, _test_ds.tb);
        model.train()
        return _loss_fn(pred, _test_ds.target)
    
    train_ds.to(device);
    test_ds.to(device);
    baseline_loss = unnormalized_loss(torch.zeros(test_ds.target.shape).to(device), test_ds.target);
    print(f"baseline loss = {baseline_loss:>4f}");
    def loss_fn(pred, gt):
        return unnormalized_loss(pred, gt)/baseline_loss;
    
    train_loss = np.zeros(epochs);
    test_loss = np.zeros(epochs);
    minLoss = np.inf;
    best_result_path = savepath + 'intermed_result.pt';

    for i in range(epochs):
        train_loss[i] = train_loop(train_ds, optimizer, loss_fn, batch_size);
        train_ds.shuffle()
        with torch.no_grad():
            model.eval()
            test_loss[i] = test_loop(test_ds, loss_fn);
            model.train()
            scheduler.step(test_loss[i])
    
        if (test_loss[i] < minLoss):
            save_trained_model(best_result_path, model, optimizer, train_loss, test_loss);
            minLoss = test_loss[i]
            print(f"new min achieved, epoch = {i}, test loss = {test_loss[i]:>4f}, train loss = {train_loss[i]:>4f}")

        if (i % 1 == 0 and i > 0):
            print(f"epoch = {i}, test loss = {test_loss[i]:>4f}, train loss = {train_loss[i]:>4f}")
            print(optimizer.param_groups[0]['lr'])
            if (optimizer.param_groups[0]['lr'] < 1e-5): #lr too small to make any more progress
                train_loss = train_loss[range(i)];
                test_loss = test_loss[range(i)];
                print(f"early termination at epoch = {i}");
                break;
    print(f"finished training, best test loss = {minLoss:>4f}");
    model.load_state_dict(torch.load(best_result_path)['model'].state_dict()); # load best epoch params
    save_trained_model(savepath, model, optimizer, train_loss, test_loss, train_dataset = train_ds, test_dataset = test_ds);
    return;

def train_ldmtbnn_ph(args):
    save_dir = args.savepath + "/ph/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    savepath = save_dir + f"/ldtn_ph_nl{args.num_layers}_nu{args.num_units}_dr{args.dropout_rate}_M{args.history_length}.pt"
    train_ds, test_ds = create_ph_datasets(args.datapath, M=args.history_length, timestep=args.history_timestep)
    net = ConvTBNN(train_ds.aij_series.shape[1], train_ds.history_length, train_ds.history_timestep, num_layers=args.num_layers, num_units=args.num_units, dropout_rate=args.dropout_rate).double()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, cooldown=20, min_lr=1e-7, verbose=True)
    train_network(train_ds, test_ds, net, optimizer, scheduler, args.max_epochs, savepath, 'cuda')
    print("finished training: " + savepath)
    return savepath;

def train_ldmtbnn_vis(args):
    save_dir = args.savepath + "/vis/";
    Path(save_dir).mkdir(parents=True, exist_ok=True);
    savepath = save_dir + f"/conv_tb_vis_nl{args.num_layers}_nu{args.num_units}_dr{args.dropout_rate}_M{args.history_length}.pt"
    train_ds, test_ds = create_vis_datasets(args.datapath, M=args.history_length, timestep=args.history_timestep)
    net = ConvTBNN(train_ds.aij_series.shape[1], train_ds.history_length, train_ds.history_timestep, num_layers=args.num_layers, num_units=args.num_units, dropout_rate=args.dropout_rate, output_len=16).double()    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, cooldown=20, min_lr=1e-7, verbose=True)
    train_network(train_ds, test_ds, net, optimizer, scheduler, args.max_epochs, savepath, 'cuda')#, unnormalized_loss=torch.nn.functional.l1_loss)
    print("finished training: " + savepath)
    return savepath;

def train_ldmtbnn_dA(args):
    save_dir = args.savepath + "/dA/";
    Path(save_dir).mkdir(parents=True, exist_ok=True);
    savepath = save_dir + f"/ldtn_ph_dA_nl{args.num_layers}_nu{args.num_units}_dr{args.dropout_rate}_M{args.history_length}.pt"
    train_ds, test_ds = create_dA_datasets(args.datapath, M=args.M, timestep=args.history_timestep, num_train_folds=15, num_test_folds=4)
    net = ConvTBNN(train_ds.aij_series.shape[1], train_ds.history_length, train_ds.history_timestep, num_layers=args.num_layers, num_units=args.num_units, dropout_rate=args.dropout_rate, output_len=16).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, cooldown=20, min_lr=1e-7, verbose=True)
    train_network(train_ds, test_ds, net, optimizer, scheduler, args.max_epochs, savepath, 'cuda', batch_size=1<<10)#, unnormalized_loss=torch.nn.functional.l1_loss)

def ldm_apriori_eval(model, savepath):
    save_dir = savepath[:savepath.rfind('/')+1];

    # save train and test losses
    torch.save(model['train_loss'], save_dir + 'train_loss.pt')
    torch.save(model['test_loss'], save_dir + 'test_loss.pt')
    
    # save eigenvector alignments
    pred = get_ldm_dev_ph(model['model'], model['test_dataset'].invars, model['test_dataset'].aij_series, model['test_dataset'].tb).detach();
    gt = get_gt_ph(model['test_dataset'].target.detach())
    eig_align = torch.tensor(calcEigenvectorAlignment(gt.cpu().numpy(), pred.cpu().numpy()));
    torch.save(eig_align, save_dir + 'eigenvector_alignment.pt');

    # save memory kernels
    torch.save(model['model'].conv.ps, save_dir + 'apriori_memory_kernels.pt')

    # save QR CMT contributions
    gt_vgt = model['test_dataset'].aij_series[:,-1,:].detach();
    gt_vgt = gt_vgt.reshape((gt_vgt.shape[0], 3,3));
    quiver_info = calcdQdR(gt_vgt, pred);
    torch.save(quiver_info, save_dir + 'quiver_info.pt')

def create_full_ldm_model(ph_savepath, vis_savepath):
    ldm_ph_model = torch.load(ph_savepath)['model'];
    ldm_vis_model = torch.load(vis_savepath)['model'];
    return NODEConvTBNN(ldm_ph_model, ldm_vis_model);
    
def node_polishing(ph_savepath, vis_savepath, datapath, results_savepath, batch_size=1<<16, lr=1e-2, max_epochs=200, device='cuda'):

    def node_loss(model, ds, num_steps, dt=3e-4):
        loss = 0.0;
        num_samples = ds.aij_series.shape[0]
        aij_series = ds.aij_series.detach().clone().reshape(ds.aij_series.shape[0], ds.aij_series.shape[1], 3,3)
        for i in range(num_steps):
            dA = dt*model.forward(aij_series)
            aij_series = torch.cat((aij_series[:,1:,:,:], (aij_series[:,-1,:,:] + dA).reshape(num_samples,1,3,3)), 1);
            loss += torch.nn.functional.mse_loss(aij_series[:,-1,:,:], ds.target[:,i,:,:])
        return loss;

    def train_loop(epochs, model, train_ds, test_ds, lr=1e-2, batch_size = 1<<11, num_tsteps=10):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10, verbose=True)
        losses = torch.zeros(epochs)
        test_losses = torch.zeros(epochs)
        empty = torch.tensor([])
        batch_inds = range(0, len(train_ds), batch_size);
        test_batch_inds = range(0, len(test_ds), batch_size);
        num_batches = len(batch_inds);
        num_test_batches = len(test_batch_inds);
        for i in range(epochs):
            for j in batch_inds:
                batch_ds = LDM_TBNN_Dataset(train_ds.aij_series[j:j+batch_size,...], empty, empty, 
                                            train_ds.target[j:j+batch_size,...], empty, empty, empty);
                batch_ds.to(device);
                loss = node_loss(model, batch_ds, num_tsteps);
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses[i] += loss.detach().to(losses.device);
            losses[i] /= num_batches
            print(f"epoch = {i}, train loss = {losses[i]}")
            if (i % 1 == 0):
                test_loss = 0;
                for j in test_batch_inds:
                    with torch.no_grad():
                        batch_ds = LDM_TBNN_Dataset(test_ds.aij_series[j:j+batch_size,...], empty, empty, 
                                                    test_ds.target[j:j+batch_size,...], empty, empty, empty);
                        batch_ds.to(device);
                        test_loss += node_loss(model, batch_ds, num_tsteps);
                test_losses[i] = test_loss.to(test_losses.device)/num_test_batches
                print(f"test loss = {test_losses[i]}");
                scheduler.step(test_losses[i])
                train_ds.shuffle()
        return losses, test_losses;

    model = create_full_ldm_model(ph_savepath, vis_savepath);
    train_ds, test_ds = create_dA_datasets(datapath, num_train_folds=5, num_test_folds=3, num_tsteps=10, M=model.ph_model.history_length)
    train_ds.to('cpu') #going to use a lot of memory, so have to page these onto gpu
    test_ds.to('cpu')

    losses, test_losses = train_loop(max_epochs, model, train_ds, test_ds, lr=lr, batch_size=batch_size)
    torch.save(losses, results_savepath + 'train_loss.pt')
    torch.save(test_losses, results_savepath + 'test_loss.pt')

    model_savepath = results_savepath + f'ldm_node_M{model.ph_model.history_length}.pt'
    torch.save(model, model_savepath)
    return model_savepath;

def evolve_trajectories(full_model_path, datapath, results_savepath, start_ind=200, fold_length=200, T=100*3e-3, dt=3e-4, save_every=50):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    torch.set_default_device(device); #hacky
    full_model = torch.load(full_model_path)
    ldm_ph_model = full_model.ph_model;
    ldm_vis_model = full_model.vis_model;

    dns_aij = np.fromfile(datapath + 'aij_1024_dns.bin').reshape([131072,1000,3,3])
    dns_aij = np.concatenate([dns_aij[:,start_ind+i:start_ind+(i+ldm_ph_model.history_length),:,:] for i in range(0,fold_length,ldm_ph_model.history_length)])
    print(dns_aij.shape)
    A_0 = torch.tensor(dns_aij)
    tau = torch.sqrt(calc_characteristic_timescale(A_0[:,-1,:,:]));

    N = A_0.shape[0];
    delta = torch.diag_embed(torch.ones(N,3)) #\delta_{ij}
    del_1 = torch.einsum('nij,nkl->nijkl', delta, delta);
    del_2 = torch.einsum('nik,njl->nijkl', delta, delta);
    del_3 = torch.einsum('nil,njk->nijkl', delta, delta);

    # currently forcing does not depend on sample, so can save time by precomputing
    Da = 0.1/tau**2
    Ds = 0.1/tau**2;
    stoch_forcing_tensor = -(1/3)*torch.sqrt(Ds/5)*del_1 + (1/2)*(torch.sqrt(Ds/5)+torch.sqrt(Da/3))*del_2 + (1/2)*(torch.sqrt(Ds/5)-torch.sqrt(Da/3))*del_3;
    def forcing(_A):
        return stoch_forcing_tensor;

    num_tsteps=int(T//dt) + 1;
    print(f'num_tsteps = {num_tsteps}, save_every = {save_every}')
    ldm_result = compute_aij_trajs_ldmtbnn(A_0, ldm_ph_model, ldm_vis_model.double(), forcing, num_tsteps, tau, dt=dt, save_every=save_every, accelerate_mixing=False)
    torch.save(ldm_result, results_savepath + f'evolved_dns_initial_conditions_T{T}_dt{dt}.pt')
    return;
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='LDM_TBNN',
                                     description='Training utilities for the augmented tensor basis model',
                                     epilog='Good luck!')
    parser.add_argument('-me', '--max_epochs', help="max number of training epochs", type=int, default=200)
    parser.add_argument('-dp', '--datapath', 
                        help="path to directory containing data files, e.g., aij.bin", type=str)
    parser.add_argument('-sp', '--savepath', help="path to directory to save the trained model in", type=str)
    parser.add_argument('-hl', '--history_length', help="length of Lagrangian history", type=int, default=50)
    parser.add_argument('-nu', '--num_units', help="number of units per hidden layer in the ff portion", type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', help="initial learning rate of optimizer", type=float, default=0.3)
    parser.add_argument('-nl', '--num_layers', help="number of hidden layers", type=int, default=3)
    parser.add_argument('-dr', '--dropout_rate', help="dropout rate of dropout layers", type=float, default=0.0)
    parser.add_argument('-ht', '--history_timestep', help="multiple of DNS timestep seperating history snapshots", type=int, default=1)
    args = parser.parse_args()
    print(args)
    if (args.savepath):
        Path(args.savepath).mkdir(parents=True, exist_ok=True)

    # tangent space learning
    ph_savepath = train_ldmtbnn_ph(args);
    vis_savepath = train_ldmtbnn_vis(args);

    # a priori evaluation
    ph_model = torch.load(ph_savepath);
    ldm_apriori_eval(ph_model, ph_savepath)
    vis_model = torch.load(vis_savepath);
    ldm_apriori_eval(vis_model, vis_savepath)
    print("finished apriori postprocessing")

    # polish using node
    node_savepath = args.savepath + '/node/'
    Path(node_savepath).mkdir(parents=True, exist_ok=True);
    full_model_path = node_polishing(ph_savepath, vis_savepath, args.datapath, node_savepath, batch_size=1<<16, lr=1e-2, max_epochs=2);
    print("finished node polishing!")

    # evolve and save trajectories
    dns_traj_savepath = args.savepath + '/posteriori/'
    Path(dns_traj_savepath).mkdir(parents=True, exist_ok=True);
    evolve_trajectories(full_model_path, args.datapath, dns_traj_savepath, start_ind=200, fold_length=200, T=1*3e-3, dt=3e-4, save_every=1);

