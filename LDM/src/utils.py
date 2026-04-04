import numpy as np;
import os;
import torch;
from torch import nn;
from torch.utils.data import Dataset, DataLoader;

# unless otherwise noted, _aij.shape assumed to be (N,3,3)

def calcInvariants(_aij):
    S = 0.5*(_aij + _aij.transpose(-2,-1));
    W = 0.5*(_aij - _aij.transpose(-2,-1));
    l1 = torch.einsum('n...ij,n...ji->n...', S, S);
    l2 = torch.einsum('n...ij,n...ji->n...', W, W);
    l3 = torch.einsum('n...ij,n...jk,n...ki->n...', S, S, S);
    l4 = torch.einsum('n...ij,n...jk,n...ki->n...', W, W, S);
    l5 = torch.einsum('n...ij,n...jk,n...kl,n...li->n...', W, W, S, S);
    return torch.stack([l1, l2, l3, l4, l5], dim=-1);

# def calcSymTensorBasis(_aij):
#     I = torch.diag_embed(torch.ones(_aij.shape[0], 3));
#     S = 0.5*(_aij + _aij.transpose(-2,-1));
#     S2 = torch.einsum('...ij,...jk->...ik',S,S)
#     W = 0.5*(_aij - _aij.transpose(-2,-1));
#     W2 = torch.einsum('...ij,...jk->...ik',W,W)

#     t1 = S;
#     t2 = torch.einsum('...ij,...jk',S,W)-torch.einsum('...ij,...jk',W,S)
#     t3 = S2 - (1/3)*torch.einsum('i,ijk->ijk', torch.einsum('nii->n',S2), I);
#     t4 = W2 - (1/3)*torch.einsum('i,ijk->ijk', torch.einsum('nii->n',W2), I);
#     t5 = torch.einsum('...ij,...jk',W, S2) - torch.einsum('...ij,...jk',S2,W);
#     t6 = torch.matmul(W2, S) + torch.matmul(S, W2) - (2/3)*torch.einsum('i,ijk->ijk', torch.einsum('nij,nji->n',S , W2), I);
#     t7 = torch.matmul(W, torch.matmul(S, W2)) - torch.matmul(W2, torch.matmul(S, W));
#     t8 = torch.matmul(S, torch.matmul(W, S2)) - torch.matmul(S2, torch.matmul(W, S));
#     t9 = torch.matmul(W2, S2) + torch.matmul(S2, W2) - (2/3)*torch.einsum('i,ijk->ijk', torch.einsum('nij,nji->n', S2, W2), I);
#     t10 = torch.matmul(W, torch.matmul(S2, W2)) - torch.matmul(W2, torch.matmul(S2, W));
#     return torch.stack([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10], dim=-1);


def calcSymTensorBasis(_aij):
    device = _aij.device;
    I = torch.diag_embed(torch.ones(_aij.shape[0], 3, device=device));
    S = 0.5*(_aij + _aij.transpose(1,2));
    S2 = torch.matmul(S,S);
    W = 0.5*(_aij - _aij.transpose(1,2));
    W2 = torch.matmul(W,W);

    t1 = S;
    t2 = torch.matmul(S,W)-torch.matmul(W,S);
    t3 = S2 - (1/3)*torch.einsum('i,ijk->ijk', torch.einsum('nii->n',S2), I);
    t4 = W2 - (1/3)*torch.einsum('i,ijk->ijk', torch.einsum('nii->n',W2), I);
    t5 = torch.matmul(W, S2) - torch.matmul(S2,W);
    t6 = torch.matmul(W2, S) + torch.matmul(S, W2) - (2/3)*torch.einsum('i,ijk->ijk', torch.einsum('nij,nji->n',S , W2), I);
    t7 = torch.matmul(W, torch.matmul(S, W2)) - torch.matmul(W2, torch.matmul(S, W));
    t8 = torch.matmul(S, torch.matmul(W, S2)) - torch.matmul(S2, torch.matmul(W, S));
    t9 = torch.matmul(W2, S2) + torch.matmul(S2, W2) - (2/3)*torch.einsum('i,ijk->ijk', torch.einsum('nij,nji->n', S2, W2), I);
    t10 = torch.matmul(W, torch.matmul(S2, W2)) - torch.matmul(W2, torch.matmul(S2, W));
    return torch.stack([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10], dim=-1);

def calcSkewSymTensorBasis(_aij):
    device = _aij.device
    I = torch.diag_embed(torch.ones(_aij.shape[0], 3, device=device));
    S = 0.5*(_aij + _aij.transpose(1,2));
    S2 = torch.matmul(S,S);
    W = 0.5*(_aij - _aij.transpose(1,2));
    W2 = torch.matmul(W,W);

    t1 = W;
    t2 = torch.matmul(S, W) + torch.matmul(W, S);
    t3 = torch.matmul(S2, W) + torch.matmul(W, S2);
    t4 = torch.matmul(W2, S) - torch.matmul(S, W2);
    t5 = torch.matmul(W2, S2) - torch.matmul(S2, W2);
    t6 = torch.matmul(S, torch.matmul(W2, S2)) - torch.matmul(S2, torch.matmul(W2, S))
    return torch.stack([t1, t2, t3, t4, t5, t6], dim=-1)

def calcFullTensorBasis(_aij):
    return torch.cat([calcSymTensorBasis(_aij), calcSkewSymTensorBasis(_aij)], axis=-1);

def calc_characteristic_timescale(_aij):
    def get_good_inds(sde_result):
        def bad_cond(_x):
            is_nan = torch.logical_or(torch.isinf(_x),torch.isnan(_x));
            is_too_big = torch.abs(_x) > 5e3
            return torch.logical_or(is_nan, is_too_big)
        def good_cond(_x):
            return torch.logical_not(bad_cond(_x));
        flat_good = torch.flatten(good_cond(sde_result), start_dim=1)
        good_inds = torch.nonzero(torch.min(flat_good, dim=1).values == True).flatten()
        return good_inds
    good_inds = get_good_inds(_aij)
    S = 0.5*(_aij[good_inds,...] + _aij[good_inds,...].transpose(1,2));
    norm_arr = torch.einsum('nij,nij->n', S, S)
    tau = 1/torch.sqrt(torch.mean(norm_arr));
    return tau;

def get_restricted_euler(_A):
    I = torch.zeros(3,3, device=_A.device)
    I[0,0]=I[1,1]=I[2,2]=1.0;
    return -torch.einsum('nik,nkj->nij', _A, _A) + (1/3)*torch.einsum('nml,nlm,ij->nij', _A,_A,I);
def get_gt_ph(ph):
    return -ph;
def get_gt_vis(vis):
    return vis;

def get_latn_ph(ph_model, inputs):
    return -ph_model(inputs)
def get_latn_vis(vis_model, inputs):
    return vis_model(inputs)

def get_ldm_dev_ph(ph_model, invars, aij_series, sym_tb):
    return -ph_model(invars, aij_series, sym_tb);
def get_ldm_vis(vis_model, invars, aij_series, full_tb):
    return vis_model(invars, aij_series, full_tb);
def get_ldm_stochastic_forcing(forcing_model, _A):
    return forcing_model(_A);

def get_tbnn_dev_ph(ph_model, invars, sym_tb):
    return -ph_model(invars, sym_tb);
def get_tbnn_vis(vis_model, invars, full_tb):
    return vis_model(invars, full_tb);
def get_tbnn_stochastic_forcing(forcing_model, _A):
    return forcing_model(_A);

def compute_aij_trajs_tbnn(A_0, ph_model, vis_model, forcing_model, num_tsteps, tau, dt=3e-4, save_every=1,):
    ph_model.eval()
    vis_model.eval()
    num_trajs = A_0.shape[0];
    num_saves = (num_tsteps//save_every) + 1;
    result = torch.zeros(num_trajs, num_saves, 3,3).double();
    result[:,0,:,:] = A_0;
    def det_RHS(A):
        _A = A;
        invars = calcInvariants(A);
        full_tb = calcFullTensorBasis(A);
        sym_tb = full_tb[:,:,:,:10].clone();

        # print(f"norm(A) = {torch.nanmean(torch.einsum('nij,nij->n',A,A))}")
        # print(f"norm(RE) = {torch.mean(get_restricted_euler(A/tau))}")
        # print(f"norm(PH) = {torch.mean(get_tbnn_dev_ph(ph_model, invars, sym_tb))}")
        # print(f"norm(VL) = {torch.mean(get_tbnn_vis(vis_model, invars, full_tb))}")
        # print(f"mean(A) = {torch.mean(A)}")
        # print(f"mean(invars) = {torch.mean(invars)}")
        # print(f"mean(tb) = {torch.mean(full_tb)}")

        dA = get_restricted_euler(A/tau) + get_tbnn_dev_ph(ph_model, invars, sym_tb) + get_tbnn_vis(vis_model, invars, full_tb);
        return dA;
    def stoch_RHS(A):
        _dW = torch.normal(mean=torch.zeros(num_trajs,3,3),std=np.sqrt(dt)*torch.ones(num_trajs,3,3));
        return torch.einsum('nijkl,nkl->nij',get_tbnn_stochastic_forcing(forcing_model, A), _dW)
    
    cache = result[:,0,:,:].detach().clone();
    for i in range(1, num_tsteps+1):
        A = cache.detach().clone();
        A *= tau
        cache = cache + (dt*det_RHS(A) + stoch_RHS(A)).detach();
        if (i % save_every == 0):
            result[:,(i//save_every),:,:] = cache.detach().clone();
            print(i)
    return result;

def advance_random_initial_condition(A_0, forcing_model, num_tsteps, dt):
    num_trajs = A_0.shape[0];
    def stoch_RHS(A):
        _dW = torch.normal(mean=torch.zeros(num_trajs,3,3),std=np.sqrt(dt)*torch.ones(num_trajs,3,3));
        return torch.einsum('nijkl,nkl->nij', get_ldm_stochastic_forcing(forcing_model, A), _dW)
    result = torch.zeros(A_0.shape[0], num_tsteps, 3,3).double();
    result[:,0,:,:] = A_0;
    for i in range(1, num_tsteps):
        result[:,i,:,:] = result[:,i-1,:,:] + stoch_RHS(result[:,i-1,:,:]);
    return result;

def compute_aij_trajs_ldmtbnn(A_0, ph_model, vis_model, forcing_model, num_tsteps, tau, dt=3e-4, save_every=1, accelerate_mixing=False):
    ph_model.eval()
    vis_model.eval()
    num_trajs = A_0.shape[0];
    num_saves = int(num_tsteps//save_every) + 1;
    result = torch.zeros(num_trajs, num_saves, 3,3).double();
    result[:,0,:,:] = A_0[:,-1,:,:].detach().clone();
    update_freq = int(ph_model.history_dt / dt)

    def det_RHS(A, aij_series, _tau):
        _aij_series = aij_series.reshape(aij_series.shape[0], aij_series.shape[1], 9);#.flatten(start_dim=2);
        invars = calcInvariants(A);
        full_tb = calcFullTensorBasis(A);
        sym_tb = full_tb[:,:,:,:10].clone();

        # print(f"norm(A) = {torch.mean(torch.einsum('nij,nij->n',A,A))}")
        # print(f"norm(A_series) = {torch.mean(torch.einsum('nkij,nkij->n',aij_series,aij_series))}")
        # print(f"norm(RE) = {torch.mean(get_restricted_euler(A/_tau))}")
        # print(f"norm(PH) = {torch.mean(get_ldm_dev_ph(ph_model, invars, _aij_series, sym_tb))}")
        # print(f"norm(VL) = {torch.mean(get_ldm_vis(vis_model, invars, _aij_series, full_tb))}")
        # print(f"mean(A) = {torch.mean(A)}")
        # print(f"mean(invars) = {torch.mean(invars)}")
        # print(f"mean(tb) = {torch.mean(full_tb)}")

        dA = get_restricted_euler(A/_tau) + get_ldm_dev_ph(ph_model, invars, _aij_series, sym_tb) + get_ldm_vis(vis_model, invars, _aij_series, full_tb);
        return dA
    def stoch_RHS(A, aij_series):
        _dW = torch.normal(mean=torch.zeros(num_trajs,3,3),std=np.sqrt(dt)*torch.ones(num_trajs,3,3));
        return torch.einsum('nijkl,nkl->nij', get_ldm_stochastic_forcing(forcing_model, A), _dW)

    temporal_cache = A_0.clone().detach();#torch.zeros(A_0.shape[0], (A_0.shape[1]-1)*cache_rot_freq, 3, 3);
    temporal_cache *= tau;#calcCharacteristicTimescale(temporal_cache.flatten(start_dim=0, end_dim=1));
    temporal_inds = range(0, A_0.shape[1], update_freq);
    
    cache = result[:,0,:,:].clone().detach();
    for i in range(1, num_tsteps+1):
        A = cache.detach().clone();
        _tau = tau;#calcCharacteristicTimescale(A);
        A *= _tau;
        cache = cache + (dt*det_RHS(A, temporal_cache[:, temporal_inds, :, :], _tau) + stoch_RHS(A, temporal_cache[:, temporal_inds, :, :])).detach();
        # if (accelerate_mixing and i < 50):
        #     temporal_cache[:,0:-update_freq,:,:] = temporal_cache[:,update_freq:,:,:].detach().clone();
        #     for j in range(-update_freq, temporal_cache.shape[1]):
        #         temporal_cache[:,j,:,:] = cache.detach().clone()*tau;#calcCharacteristicTimescale(cache);
        # elif (i % 1 == 0):#update_freq == 0):
        temporal_cache[:,0:-1,:,:] = temporal_cache[:,1:,:,:].detach().clone();
        temporal_cache[:,-1,:,:] = cache.detach().clone()*tau;#calcCharacteristicTimescale(cache);
        if (i % save_every == 0):
            result[:,(i//save_every),:,:] = cache.detach().clone();
            print(i)
    return result;


def save_trained_model(save_path, model, optimizer, train_loss, test_loss, train_dataset=None, test_dataset=None):
    torch.save({
        'model' : model,
        'optimizer': optimizer,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
    }, save_path);


def calcEigenvectorAlignment(gt, pred):
    num_samples = gt.shape[0];
    gt_eigs = [np.linalg.eigh(gt[i,:,:])[1] for i in range(num_samples)];
    pred_eigs = [np.linalg.eigh(pred[i,:,:])[1] for i in range(num_samples)];
    e3_alignment = np.array([np.arccos(np.abs(np.dot(pred_eigs[i][:,0], gt_eigs[i][:,0]))) for i in range(num_samples)]);
    e2_alignment = np.array([np.arccos(np.abs(np.dot(pred_eigs[i][:,1], gt_eigs[i][:,1]))) for i in range(num_samples)]);
    e1_alignment = np.array([np.arccos(np.abs(np.dot(pred_eigs[i][:,2], gt_eigs[i][:,2]))) for i in range(num_samples)]);
    return np.stack((e3_alignment, e2_alignment, e1_alignment), axis=0)

def compute_W_norm(aij):
    if (aij.shape[0] == aij.shape[1]):
        W_arr = 0.5*(aij-torch.transpose(aij, dim0=0, dim1=1));
        W_norm = torch.sqrt(torch.mean(torch.einsum('ijn,ijn->n', W_arr, W_arr)))
    elif (aij.shape[-2] == aij.shape[-1]):
        W_arr = 0.5*(aij-torch.transpose(aij, dim0=-2, dim1=-1));
        W_norm = torch.sqrt(torch.mean(torch.einsum('nij,nij->n', W_arr, W_arr)))
    return W_norm;

# force_grad a stand-in for pressure Hessian, viscous Laplacian, restricted Euler
#  sample indices should align, e.g., vgt.shape = (3,3,T,N) = H.shape
def calcdQdR(vgt, force_grad):
    W_norm = compute_W_norm(vgt)
    if (vgt.shape[0] == vgt.shape[1]):
        Q = -0.5*torch.einsum('ij...,ji...', vgt, vgt)/(W_norm**2);
        R = -(1/3)*torch.einsum('ij...,jk...,ki...', vgt, vgt, vgt)/(W_norm**3);
        dQ = -torch.einsum('ij...,ji...', vgt, force_grad)/W_norm**3;
        dR = -torch.einsum('ij...,jk...,ki...', vgt, vgt, force_grad)/W_norm**4;
    elif (vgt.shape[-2] == vgt.shape[-1]):
        Q = -0.5*torch.einsum('...ij,...ji', vgt, vgt)/(W_norm**2);
        R = -(1/3)*torch.einsum('...ij,...jk,...ki', vgt, vgt, vgt)/(W_norm**3);
        dQ = -torch.einsum('...ij,...ji', vgt, force_grad)/W_norm**3;
        dR = -torch.einsum('...ij,...jk,...ki', vgt, vgt, force_grad)/W_norm**4;

    return torch.stack((Q,R,dQ,dR), dim=-1)


def second_order_backward_fd(arr, dt):
    """Calculates first derivative using backward FD, i.e.
    A(t) -> arr={A(t_{n-2}), A(t_{n-1}), A(t_{n})}, returns
    \\partial_t A(t=t_n).

    arr - shape = (N,3,...); N = num_samples, 3 = timesteps
    returns deriv with shape=(N,...)
    """
    coef_m2 = 1/(2*dt)
    coef_m1 = -2/dt
    coef_0 = 3/(2*dt)
    coefs = torch.tensor([coef_m2, coef_m1, coef_0], dtype=float)
    return torch.einsum('nt...,t->n...', arr, coefs)


def calc_trace(arr: torch.tensor):
    """calculates trace assuming arr.shape=(..., 3, 3)"""
    return torch.einsum('...ii', arr)


def remove_trace(arr: torch.tensor):
    """returns new torch.tensor that is arr - (1/3)*tr(arr)
    assumes arr.shape=(..., 3, 3)
    """
    diag_shape = list(arr.shape[:-2])
    diag_shape.append(3)
    return arr - (1/3) * \
        torch.einsum('..., ...jk', calc_trace(arr),
                     torch.diag_embed(torch.ones(diag_shape)))
