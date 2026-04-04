import glob
import csv
import matplotlib.pyplot as plt;
import numpy as np;
import os;
import torch;
from torch import nn;
from torch.utils.data import Dataset, DataLoader;
from utils import *

# arr_pred_ph is an array of pressure hessian predictions, each having size of gt_ph
#   i.e. arr_pred_ph[i].shape == gt_ph.shape for i in range(len(arr_pred_ph));
def comparePressureEigenvectorAlignment(gt_ph, arr_pred_ph, labels=[], title=''):
    num_comps = len(arr_pred_ph)
    num_samples = gt_ph.shape[0];
    fig, axs = plt.subplots(1,3, sharey=True, dpi=300);
    nbins = 30;
    for j in range(num_comps):
        eig_align = calcEigenvectorAlignment(gt_ph, arr_pred_ph[j]);
        axs[0].hist(eig_align[0,:], bins=nbins, density=True, histtype='step')
        axs[1].hist(eig_align[1,:], bins=nbins, density=True, histtype='step')
        axs[2].hist(eig_align[2,:], bins=nbins, density=True, histtype='step')
    fig.supxlabel(r'$\theta$')
    fig.supylabel(r'PDF')
    fig.suptitle(title)
    axs[0].set_title('$e_1$')
    axs[1].set_title('$e_2$')
    axs[2].set_title('$e_3$')
    axs[1].legend()
    return plt;

def plotPressureEigenvectorAlignment2d(ax, gt, pred, eig1=0, eig2=2, density=True):
    eig_align = calcEigenvectorAlignment(gt, pred);
    ax.hist2d(eig_align[eig1,:], eig_align[eig2,:], density=density);
    ax.set_xlabel(f'$\theta_{eig1+1}$')
    ax.set_ylabel(f'$\theta_{eig2+1}$')
    #ax.colorbar()
    return;

def plotLongitudinalPDF(vgt):
#    fig, axs = plt.subplots(1,1);
    num_samples = vgt.shape[0];
    denom = 1/np.sum(vgt[:,0,0]**2)**(0.5)
    plt.hist(vgt[:,0,0]/denom, bins=500, density=True, histtype='step', log=True)
    plt.ylim((1e-4, 1e0))
    return plt.show()

def plotTransversePDF(vgt):
    num_samples = vgt.shape[0];
    denom = 1/np.sum(vgt[:,0,1]**2)**(0.5)
    plt.hist(vgt[:,0,0]/denom, bins=500, density=True, histtype='step', log=True)
    plt.ylim((1e-4, 1e0))
    return plt.show()
    
def assign_QR_phase_space_indx(aij, num_Qbins, num_Rbins, Rlims=(-5,5), Qlims=(-5,5)):
    W_norm = compute_W_norm(aij);
    Qbin_width = (Qlims[-1]-Qlims[0])/num_Qbins;
    Rbin_width = (Rlims[-1]-Rlims[0])/num_Rbins;
    Q = -0.5*torch.einsum('ijn,jin->n', aij, aij)/(W_norm**2);
    R = -(1/3)*torch.einsum('ijn,jkn,kin->n', aij, aij, aij)/(W_norm**3);
    Qind = torch.floor((Q-Qlims[0])/Qbin_width);
    Rind = torch.floor((R-Rlims[0])/Rbin_width);
    Qind = torch.where(Qind < 0, -1, Qind);
    Qind = torch.where(Qind >= num_Qbins, -1, Qind);
    Rind = torch.where(Rind < 0, -1, Rind);
    Rind = torch.where(Rind >= num_Rbins, -1, Rind);
    return (Qind, Rind);

def assign_qr_phase_space_indx(aij, num_qbins=50, num_rbins=50):
    qlims = (-0.5, 0.5);
    rlims = (-np.sqrt(3)/9, np.sqrt(3)/9);
    qbin_width = (qlims[-1]-qlims[0])/num_qbins;
    rbin_width = (rlims[-1]-rlims[0])/num_rbins;
    A = torch.sqrt(torch.einsum('ij...,ij...', aij, aij));
    b = torch.div(aij,A);
    q = -0.5*torch.einsum('ij...,ji...', b, b);
    r = -(1/3)*torch.einsum('ij...,jk...,ki...', b, b, b);
    qind = torch.floor((q-qlims[0])/qbin_width);
    rind = torch.floor((r-rlims[0])/rbin_width);
    return (qind, rind)

def plot_QR_seperatix(ax, R_range=(-4,4)):
    seperatix_R = np.linspace(R_range[0], R_range[-1], num=1000);
    seperatix_Q = -((27.0/4.0)*(seperatix_R**2))**(1.0/3.0)
    ax.plot(seperatix_R, seperatix_Q, color='b', alpha=0.4);

def plot_QRCMT(ax, vgt, ph, num_Qbins=20, num_Rbins=20, label="", color='b', cutoff=50, scale=1.0, save=False, title_prefix='Pressure Hessian', show_title=True):
    W_norm = compute_W_norm(vgt)
    R_range = np.linspace(-5,5, num=num_Rbins+1)/W_norm.cpu().numpy()**3
    Q_range = np.linspace(-5,5, num=num_Qbins+1)/W_norm.cpu().numpy()**2
    
    Qind, Rind = assign_QR_phase_space_indx(vgt, num_Qbins, num_Rbins);
    dR = -torch.einsum('ijn,jkn,kin->n', vgt, vgt, ph)/W_norm**4;
    dQ = -torch.einsum('ijn,jin->n', vgt, ph)/W_norm**3;

    R_mid = (R_range[:-1] + R_range[1:])/2;
    Q_mid = (Q_range[:-1] + Q_range[1:])/2;

    quiver_x = np.zeros(R_mid.shape[0]*Q_mid.shape[0]);
    quiver_y = np.zeros(R_mid.shape[0]*Q_mid.shape[0]);
    quiver_dx = np.zeros(R_mid.shape[0]*Q_mid.shape[0]);
    quiver_dy = np.zeros(R_mid.shape[0]*Q_mid.shape[0]);
    samples_in_bin = np.zeros(R_mid.shape[0]*Q_mid.shape[0]);

    for i in range(R_mid.shape[0]):
        for j in range(Q_mid.shape[0]):
            ind = i*Q_mid.shape[0]+j;
            quiver_x[ind] = R_mid[i];
            quiver_y[ind] = Q_mid[j];
    
    Qind = Qind.cpu().numpy()
    Rind = Rind.cpu().numpy()
    dR = dR.cpu().numpy()
    dQ = dQ.cpu().numpy()
    for i in range(vgt.shape[-1]):
        if (Qind[i] > -1 and Rind[i] > -1):
            ind = int(Rind[i]*Q_mid.shape[0]+Qind[i])
            quiver_dx[ind] += dR[i];
            quiver_dy[ind] += dQ[i];
            samples_in_bin[ind] += 1;

    for i in range(quiver_dx.shape[0]):
        if (samples_in_bin[i] > cutoff):
            quiver_dx[i] /= samples_in_bin[i]
            quiver_dy[i] /= samples_in_bin[i]
        else:
            quiver_dx[i] = 0.0;
            quiver_dy[i] = 0.0;

    ax.quiver(quiver_x, quiver_y, quiver_dx, quiver_dy, scale=scale, color=color, label=label, angles='xy')
    return;

# def plot_qrCMT(ax, vgt, ph, num_qbins=20, num_rbins=20, label="", color='b', cutoff=50, scale=1.0, save=False, title_prefix='Pressure Hessian', show_title=True):
#     A = torch.einsum('ijn,ijn->n', vgt, vgt)
#     b = torch.div(vgt, A);
#     norm_ph = torch.div(ph, A**2);
#     r_range = np.linspace(-np.sqrt(3)/9,np.sqrt(3)/9, num=num_rbins+1)
#     q_range = np.linspace(-0.5,0.5, num=num_rbins+1)
    
#     qind, rind = assign_qr_phase_space_indx(b, num_qbins, num_rbins);
#     q = -0.5*torch.einsum('ij...,ji...', b, b);
#     r = -(1/3)*torch.einsum('ij...,jk...,ki...', b, b, b);
#     dq = torch.einsum('ij...,ji...', norm_ph, b-2*q*b);
#     dr = torch.einsum('ij...,jk...,ki...', norm_ph, b, b, norm_ph)


#     r_mid = (r_range[:-1] + r_range[1:])/2;
#     q_mid = (q_range[:-1] + q_range[1:])/2;

#     quiver_x = np.zeros(r_mid.shape[0]*q_mid.shape[0]);
#     quiver_y = np.zeros(r_mid.shape[0]*q_mid.shape[0]);
#     quiver_dx = np.zeros(r_mid.shape[0]*q_mid.shape[0]);
#     quiver_dy = np.zeros(r_mid.shape[0]*q_mid.shape[0]);
#     samples_in_bin = np.zeros(r_mid.shape[0]*q_mid.shape[0]);

#     for i in range(r_mid.shape[0]):
#         for j in range(q_mid.shape[0]):
#             ind = i*q_mid.shape[0]+j;
#             quiver_x[ind] = r_mid[i];
#             quiver_y[ind] = q_mid[j];
    
#     qind = qind.cpu().numpy()
#     rind = rind.cpu().numpy()
#     dr = dr.cpu().numpy()
#     dq = dq.cpu().numpy()
#     for i in range(b.shape[-1]):
#         if (qind[i] > -1 and rind[i] > -1):
#             ind = int(rind[i]*q_mid.shape[0]+qind[i])
#             quiver_dx[ind] += dr[i];
#             quiver_dy[ind] += dq[i];
#             samples_in_bin[ind] += 1;

#     for i in range(quiver_dx.shape[0]):
#         if (samples_in_bin[i] > cutoff):
#             quiver_dx[i] /= samples_in_bin[i]
#             quiver_dy[i] /= samples_in_bin[i]
#         else:
#             quiver_dx[i] = 0.0;
#             quiver_dy[i] = 0.0;

#     ax.quiver(quiver_x, quiver_y, quiver_dx, quiver_dy, scale=scale, color=color, label=label)
#     return;

    
def plotPressureQRCMT(ax, vgt, ph_arr, labels=[], cutoff=50, scale=1.0, save=False, title_prefix='Pressure Hessian', show_title=True):
    num_samples = vgt.shape[0];
    W_arr = [0.5*(vgt[i,:,:]-vgt[i,:,:].transpose(0,1)) for i in range(num_samples)];
    wnorm = np.sqrt(np.mean([np.linalg.norm(W_arr[i]) for i in range(num_samples)]));
    
    #fig, ax = plt.subplots(figsize=fig_size);
    #ax = fig.add_subplot(111);
    ax.set_ylabel=('Q');
    ax.set_xlabel=('R');
    if show_title:
        ax.set_title(title_prefix + ' contribution to Q-R CMTs');
    r = [-np.trace(np.linalg.matrix_power(vgt[i,:,:], 3))/3.0 for i in range(num_samples)]/wnorm**3;
    q = [-np.trace(np.linalg.matrix_power(vgt[i,:,:], 2))/2.0 for i in range(num_samples)]/wnorm**2;

    r_range = np.linspace(-5, 5, num=21);
    q_range = np.linspace(-5, 5, num=21);
    r_mid = (r_range[:-1] + r_range[1:])/2;
    q_mid = (q_range[:-1] + q_range[1:])/2;

    seperatix_r = np.linspace(r_range[0], r_range[-1], num=1000);
    seperatix_q = -((27.0/4.0)*(seperatix_r**2))**(1.0/3.0)
    ax.plot(seperatix_r, seperatix_q, color='b', alpha=0.4);
    
    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    
    for ph_itr in range(len(ph_arr)):
        ph = ph_arr[ph_itr]

        dr = [np.trace(np.matmul(np.matmul(vgt[i,:,:],vgt[i,:,:]),ph[i,:,:])) for i in range(num_samples)]/wnorm**4;
        dq = [np.trace(np.matmul(vgt[i,:,:],ph[i,:,:])) for i in range(num_samples)]/wnorm**3;

        dq_array = np.zeros((r_mid.shape[0], q_mid.shape[0]));
        dr_array = np.zeros(dq_array.shape);

        for i in range(r_mid.shape[0]):
            ind_r = [k for k in range(r.shape[0]) if (r_range[i+1] > r[k] > r_range[i])];
            active_q = q[ind_r];
            for j in range(q_mid.shape[0]):
                ind_q = [k for k in range(active_q.shape[0]) if (q_range[j+1] > active_q[k] > q_range[j])];
                if (len(ind_q) > cutoff):
                    dq_array[i,j] = np.mean(dq[ind_r][ind_q]);
                    dr_array[i,j] = np.mean(dr[ind_r][ind_q]);

        shape_prod = r_mid.shape[0]*q_mid.shape[0]
        quiver_x = np.zeros(shape_prod);
        quiver_y = np.zeros(shape_prod);
        quiver_dx = np.zeros(shape_prod);
        quiver_dy = np.zeros(shape_prod);
        for i in range(r_mid.shape[0]):
            for j in range(q_mid.shape[0]):
                ind = i*q_mid.shape[0] + j;
                quiver_x[ind] = r_mid[i];
                quiver_y[ind] = q_mid[j];
                quiver_dx[ind] = dr_array[i,j];
                quiver_dy[ind] = dq_array[i,j];

        if (save):
            np.save(labels[ph_itr] + '_quiver_x.npy', quiver_x);
            np.save(labels[ph_itr] + '_quiver_y.npy', quiver_y);
            np.save(labels[ph_itr] + '_quiver_dx.npy', quiver_dx);
            np.save(labels[ph_itr] + '_quiver_dy.npy', quiver_dy);

        ax.quiver(quiver_x, quiver_y, quiver_dx, quiver_dy, scale=scale, label=labels[ph_itr], color=colormap[ph_itr])

    ax.legend()
    return ax;

def plotPressureQRCMT_fromFile(labels=[], cutoff=50, scale=1.0, save=False):
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.set_ylabel=('Q');
    ax.set_xlabel=('R');
    ax.set_title('Pressure Hessian contribution to Q-R CMTs');
    r_range = np.linspace(-5, 5, num=21);
    q_range = np.linspace(-5, 5, num=21);
    seperatix_r = np.linspace(r_range[0], r_range[-1], num=1000);
    seperatix_q = -((27.0/4.0)*(seperatix_r**2))**(1.0/3.0)
    plt.plot(seperatix_r, seperatix_q);
    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    
    for l in range(len(labels)):
        quiver_x = np.load(labels[l] + '_quiver_x.npy');
        quiver_y = np.load(labels[l] + '_quiver_y.npy');
        quiver_dx = np.load(labels[l] + '_quiver_dx.npy');
        quiver_dy = np.load(labels[l] + '_quiver_dy.npy');
        plt.quiver(quiver_x, quiver_y, quiver_dx, quiver_dy, scale=scale, label=labels[l], color=colormap[l])

    plt.legend()
    return plt.show()

def plotQRPDF(ax, vgt, num_bins=100, levels = 7, linestyle='solid', Rlims=(-5,5), Qlims=(-5,5)):
    W_norm = compute_W_norm(vgt);
    Q = -0.5*torch.einsum('...ij,...ji', vgt, vgt)/(W_norm**2);
    R = -(1/3)*torch.einsum('...ij,...jk,...ki', vgt, vgt, vgt)/(W_norm**3);
    Q = Q.cpu().numpy()
    R = R.cpu().numpy()

    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    ax.set_ylabel=('Q');
    ax.set_xlabel=('R');
    ax.set_title('QR PDF');
    R_range = np.linspace(Rlims[0], Rlims[1], num=num_bins);
    Q_range = np.linspace(Qlims[0], Qlims[1], num=num_bins);
    seperatix_R = np.linspace(R_range[0], R_range[-1], num=1000);
    seperatix_Q = -((27.0/4.0)*(seperatix_R**2))**(1.0/3.0)

    h, xedges, yedges = np.histogram2d(R, Q, bins=[R_range, Q_range], density=True)
    x_mids = (xedges[1:]+xedges[:-1])/2
    y_mids = (yedges[1:]+yedges[:-1])/2
    CS = ax.contour(x_mids, y_mids, h.T, levels, linestyles=linestyle, colors=colormap)
    ax.plot(seperatix_R, seperatix_Q, color='k', alpha=0.15)
    ax.set_xlim(R_range[0], R_range[-1])
    ax.set_ylim(Q_range[0], Q_range[-1])
    return

def plot_aligncondqr(ax, vgt, num_bins=50, levels=[0.1*i for i in range(11)] , linestyle='solid'):
    rlims = (-np.sqrt(3)/9, np.sqrt(3)/9);
    qlims = (-1/2, 1/2);
    r_range = np.linspace(rlims[0], rlims[1], num=num_bins);
    q_range = np.linspace(qlims[0], qlims[1], num=num_bins);
    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];

    def new_calc_alignment(vgt):
        S = 0.5*(vgt + vgt.transpose(1,2))
        W = 0.5*(vgt - vgt.transpose(1,2))
        epsilon = torch.zeros(3,3,3)
        epsilon[2,1,0] = epsilon[0,2,1] = epsilon[1,0,2] = 1.0
        omega = torch.einsum('nij,ijk->nk',W,epsilon);
        vals, vecs = torch.linalg.eigh(S);
        #align = torch.abs(torch.einsum('nk,nk->n', omega, vecs[:,:,1]))/torch.linalg.norm(omega);
        align = torch.abs(torch.div(torch.einsum('nk,nk->n', omega, vecs[:,:,1]), torch.sqrt(torch.einsum('nk,nk->n',omega,omega))));
        return align;
    # def calc_alignment(vgt):
    #     S = 0.5*(vgt + vgt.transpose(0,1));
    #     W = 0.5*(vgt - vgt.transpose(0,1));
    #     omega = np.array([W[2,1], W[0,2], W[1,0]])
        
    #     eigenvalues, eigenvectors = np.linalg.eig(S);
    #     idx = eigenvalues.argsort()[::-1]   
    #     eigenvectors = eigenvectors[:,idx]
        
    #     return np.abs(np.dot(omega,eigenvectors[:,1])/np.linalg.norm(omega));
    
    SRE_arr = new_calc_alignment(vgt)#np.array([calc_alignment(vgt[i,:,:]) for i in range(num_samples)]);
    perm_vgt = torch.permute(vgt, (1,2,0));
    A_arr = torch.sqrt(torch.einsum('ijn, ijn->n',perm_vgt,perm_vgt))
    norm_vgt = torch.div(perm_vgt, A_arr);
    q = -0.5*torch.einsum('ijn,jin->n', norm_vgt, norm_vgt).cpu().numpy();
    r = -(1/3)*torch.einsum('ijn,jkn,kin->n', norm_vgt, norm_vgt, norm_vgt).cpu().numpy();
    h, xedges, yedges = np.histogram2d(r, q, bins=[r_range, q_range], weights=SRE_arr.cpu().numpy(), density=True);
    h_samples, __xedges, __yedges = np.histogram2d(r,q,bins=[r_range, q_range]);    
    h = (h)/(h_samples+1); #add one to avoid div by 0

    x_mids = (xedges[1:] + xedges[:-1])/2;
    y_mids = (yedges[1:] + yedges[:-1])/2;
    CS = ax.contour(x_mids, y_mids, h.T*1000, levels, linestyles=linestyle, cmap='viridis');
    ax.set_xlim(r_range[0], r_range[-1]);
    ax.set_ylim(q_range[0], q_range[-1]);
    return;

def plot_interSREcondqr(ax, vgt, num_bins=100, levels=7, linestyle='solid'):
    num_samples = vgt.shape[-1];
    rlims = (-np.sqrt(3)/9, np.sqrt(3)/9);
    qlims = (-1/2, 1/2);
    r_range = np.linspace(rlims[0], rlims[1], num=num_bins);
    q_range = np.linspace(qlims[0], qlims[1], num=num_bins);
    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];
        
    perm_vgt = torch.permute(vgt, (2,0,1));
    S = 0.5 * (perm_vgt + perm_vgt.transpose(1,2));
    SRE_arr = torch.linalg.eigh(S).eigenvalues[:,1];

    A_arr = torch.sqrt(torch.einsum('ijn, ijn->n',vgt,vgt))
    norm_vgt = torch.div(vgt, A_arr);
    q = -0.5*torch.einsum('ijn,jin->n', norm_vgt, norm_vgt).cpu().numpy();
    r = -(1/3)*torch.einsum('ijn,jkn,kin->n', norm_vgt, norm_vgt, norm_vgt).cpu().numpy();
    h, xedges, yedges = np.histogram2d(r, q, bins=[r_range, q_range], weights=SRE_arr.cpu().numpy(), density=True);
    h_samples, __xedges, __yedges = np.histogram2d(r,q,bins=[r_range, q_range]);    
    h = h/(h_samples+1); #add one to avoid div by 0
    h *= 10;
    
    x_mids = (xedges[1:] + xedges[:-1])/2;
    y_mids = (yedges[1:] + yedges[:-1])/2;
    CS = ax.contour(x_mids, y_mids, h.T, levels, linestyles=linestyle, cmap='seismic');
    ax.set_xlim(r_range[0], r_range[-1]);
    ax.set_ylim(q_range[0], q_range[-1]);
    return;

def plot_Acondqr(ax, vgt, num_bins=100, levels=7, linestyle='solid'):
    num_samples = vgt.shape[0];
    rlims = (-np.sqrt(3)/9, np.sqrt(3)/9);
    qlims = (-1/2, 1/2);
    r_range = np.linspace(rlims[0], rlims[1], num=num_bins);
    q_range = np.linspace(qlims[0], qlims[1], num=num_bins);
    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    
    A_arr = torch.sqrt(torch.einsum('ijn, ijn->n',vgt,vgt))
    norm_vgt = torch.div(vgt, A_arr);
    q = -0.5*torch.einsum('ijn,jin->n', norm_vgt, norm_vgt).cpu().numpy();
    r = -(1/3)*torch.einsum('ijn,jkn,kin->n', norm_vgt, norm_vgt, norm_vgt).cpu().numpy();
    A_arr = A_arr.cpu().numpy()
    h, xedges, yedges = np.histogram2d(r, q, bins=[r_range, q_range], weights=A_arr**2, density=True);
    h_samples, __xedges, __yedges = np.histogram2d(r,q,bins=[r_range, q_range]);
    h = h/(h_samples+1); #add one to avoid div by 0
    
    x_mids = (xedges[1:] + xedges[:-1])/2;
    y_mids = (yedges[1:] + yedges[:-1])/2;
    CS = ax.contour(x_mids, y_mids, h.T, levels, linestyles=linestyle, colors=colormap);
    ax.set_xlim(r_range[0], r_range[-1]);
    ax.set_ylim(q_range[0], q_range[-1]);
    return;

def plot_Hcondqr(ax, vgt, ph, num_bins=100, levels=7, linestyle='solid'):
    num_samples = vgt.shape[0];
    rlims = (-np.sqrt(3)/9, np.sqrt(3)/9);
    qlims = (-1/2, 1/2);
    r_range = np.linspace(rlims[0], rlims[1], num=num_bins);
    q_range = np.linspace(qlims[0], qlims[1], num=num_bins);
    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    
    A_arr = torch.sqrt(torch.einsum('ijn, ijn->n',vgt,vgt))
    H_arr = torch.sqrt(torch.einsum('nij, nij->n',ph, ph)).cpu().numpy()
    norm_vgt = torch.div(vgt, A_arr);
    q = -0.5*torch.einsum('ijn,jin->n', norm_vgt, norm_vgt).cpu().numpy();
    r = -(1/3)*torch.einsum('ijn,jkn,kin->n', norm_vgt, norm_vgt, norm_vgt).cpu().numpy();
    A_arr = A_arr.cpu().numpy()
    h, xedges, yedges = np.histogram2d(r, q, bins=[r_range, q_range], weights=H_arr**2, density=True);
    h_samples, __xedges, __yedges = np.histogram2d(r,q,bins=[r_range, q_range]);
    h = h/(h_samples+1); #add one to avoid div by 0
    
    x_mids = (xedges[1:] + xedges[:-1])/2;
    y_mids = (yedges[1:] + yedges[:-1])/2;
    CS = ax.contour(x_mids, y_mids, h.T, levels, linestyles=linestyle, colors=colormap);
    ax.set_xlim(r_range[0], r_range[-1]);
    ax.set_ylim(q_range[0], q_range[-1]);
    return;


def plot_qr_pdf(ax, vgt, num_bins=100, levels=7, linestyle='solid'):
    colormap = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    rlims = (-np.sqrt(3)/9, np.sqrt(3)/9);
    qlims = (-1/2, 1/2);
    r_range = np.linspace(rlims[0], rlims[1], num=num_bins);
    q_range = np.linspace(qlims[0], qlims[1], num=num_bins);
    #num_samples = vgt.shape[0];
    A_arr = torch.sqrt(torch.einsum('ijn, ijn->n',vgt,vgt))
    norm_vgt = torch.div(vgt, A_arr);
    #A_arr = [np.linalg.norm(vgt[i,:,:]) for i in range(num_samples)];
    #norm_vgt = [vgt[i,:,:]/A_arr[i] for i in range(num_samples)];
    q = -0.5*torch.einsum('ijn,jin->n', norm_vgt, norm_vgt);
    r = -(1/3)*torch.einsum('ijn,jkn,kin->n', norm_vgt, norm_vgt, norm_vgt);

    h, xedges, yedges = np.histogram2d(r.cpu().numpy(), q.cpu().numpy(), bins=[r_range, q_range], density=True);
    x_mids = (xedges[1:] + xedges[:-1])/2;
    y_mids = (yedges[1:] + yedges[:-1])/2;
    CS = ax.contour(x_mids, y_mids, h.T, levels, linestyles=linestyle, colors=colormap);
    ax.set_xlim(r_range[0], r_range[-1]);
    ax.set_ylim(q_range[0], q_range[-1]);
    return;

# def plot_losses(search_str, root_dir=None):
#     paths = glob.glob(search_str, root_dir=root_dir, recursive=True)
#     losses = {}
#     for 
