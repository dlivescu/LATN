from dataclasses import dataclass
from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import utils
import lagrdataset

class ScalarHistoryConv(nn.Module):
    def __init__(self, num_tsteps, num_filters):
        super().__init__()
        self.ps = torch.nn.Parameter(torch.rand(num_tsteps, num_filters))

    def forward(self, x):
        conv = torch.einsum('...jk,jm->...m', x, self.ps)
        return nn.functional.sigmoid(conv)


class TensorHistoryConv(nn.Module):
    def __init__(self, num_tsteps, num_filters):
        super().__init__()
        self.ps = torch.nn.Parameter(torch.rand(num_tsteps, num_filters, 9))

    def forward(self, x):
        conv = torch.einsum('...jk, jmk -> ...m', x, self.ps)
        return nn.functional.sigmoid(conv)


class ConstrainedTensorHistoryConv(TensorHistoryConv):
    def _get_conv_filters(self):
        num_steps, num_filters = self.ps.shape[:2]
        num_sym_filters = int(num_filters/2)
        sym_params = self.ps[:, :num_sym_filters, ...]\
                         .reshape(num_steps, num_sym_filters, 3, 3)
        asym_params = self.ps[:, num_sym_filters:, ...]\
                          .reshape(num_steps,
                                   num_filters-num_sym_filters, 3, 3)
        sym_kernels = (0.5*(sym_params + sym_params.transpose(-2, -1)))\
            .reshape(num_steps, num_sym_filters, 9)
        asym_kernels = (0.5*(asym_params - asym_params.transpose(-2, -1)))\
            .reshape(num_steps, num_filters-num_sym_filters, 9)
        return torch.cat((sym_kernels, asym_kernels), 1)

    def forward(self, x):
        # i->samples, j->timestep, m->filters, k->tensor entry
        # returns a (i,m) tensor
        filters = self._get_conv_filters()
        conv = torch.einsum('...jk,jmk->...m',
                            x,
                            filters)
        return nn.functional.sigmoid(conv)


@dataclass
class LATNDesc:
    """Holds all user-defined properties of LATN model.
    Meant to standardize interface to model re/construction
    """
    num_layers: int
    num_units: int
    activation: any  # <torch.nn.modules.activation>
    input_len: int
    output_len: int
    dropout_rate: float

class FFN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers,
                 num_hidden_units,
                 activation):
        super().__init__()
        self.layers = OrderedDict()
        if (num_layers == 0):
            self.layers['lin1'] = nn.Linear(input_dim,
                                            output_dim)
        else:
            self.layers['lin1'] = nn.Linear(input_dim,
                                            num_hidden_units)
            self.layers['act1'] = activation()
            #self.layers['drop'] = nn.Dropout(network_desc.dropout_rate)
            for i in range(1, num_layers):
                self.layers[f'lin{i+1}'] = nn.Linear(num_hidden_units,
                                                     num_hidden_units)
                self.layers[f'act{i+1}'] = activation()
            current_layer_dim = self.layers[list(self.layers.keys())[-2]]\
                                    .out_features
            self.layers['lin'+str(num_layers+1)] =\
                nn.Linear(current_layer_dim, output_dim)
            # compile spec into function
        self.ff = nn.Sequential(self.layers)

    def forward(self, inputs):
        return self.ff(inputs);

class Skip_FFN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers,
                 num_hidden_units,
                 activation):
        super().__init__()
        self.layers = OrderedDict()
        intermediate_output_len = 10
        self.layers['lin1'] = nn.Linear(input_dim,
                                        num_hidden_units)
        self.layers['act1'] = torch.nn.Tanh()#activation()
        self.layers['lin2'] = nn.Linear(num_hidden_units,
                                        num_hidden_units)
        self.layers['act2'] = activation()
        self.ff = nn.Sequential(self.layers)

        self.layers2 = OrderedDict()
        ff2_units = num_hidden_units
        self.layers2['lin1'] = nn.Linear(num_hidden_units,
                                         num_hidden_units)
        self.layers2['act1'] = activation()
        self.layers2['lin2'] = nn.Linear(num_hidden_units,
                                         num_hidden_units)
        self.layers2['act2'] = activation()
        self.ff2 = nn.Sequential(self.layers2)

        self.layers3 = OrderedDict()
        self.layers3['lin1'] = nn.Linear(num_hidden_units,
                                         num_hidden_units)
        self.layers3['act1'] = activation()
        self.layers3['lin2'] = nn.Linear(num_hidden_units,
                                         num_hidden_units)
        self.layers3['act2'] = activation()
        self.layers3['lin3'] = nn.Linear(num_hidden_units,
                                         output_dim)
        self.ff3 = nn.Sequential(self.layers3)

    def forward(self, inputs):
        x1 = self.ff(inputs)
        x2 = self.ff2(x1)
        outputs = self.ff3(x1+x2)
        return outputs;

class LATN(nn.Module):
    def __init__(self,
                 data_desc: lagrdataset.DataDesc,
                 network_desc: LATNDesc,
                 history_conv_type: type,
                 ff_type: type, #Skip_FFN or FFN
                 device='cpu'):
        super().__init__()
        self.data_desc = data_desc
        self.ff = ff_type(network_desc.input_len,
                          network_desc.output_len,
                          network_desc.num_layers,
                          network_desc.num_units,
                          network_desc.activation)

        # calculate dimensions for Lagrangian attention
        num_invars = 5
        self.num_filters = network_desc.input_len - num_invars
        self.num_tsteps = (self.data_desc.history_length
                           // self.data_desc.history_timestep) + 1
        self.conv = history_conv_type(self.num_tsteps, self.num_filters)
        self.to(device)

    def set_device(self, device):
        self.device=device

    def set_timescale(self, timescale):
        return

    def forward(self, inputs):
        """inputs should be exactly the first entry of LagrDataset.__getitem__
        Enforced in tests.
        """
        aij_series, invars, tb = inputs
        conv_chars = self.conv(aij_series)
        stacked_inputs = torch.cat([invars, conv_chars], dim=1)
        gs = self.ff(stacked_inputs)
        #gs = self.ff2(torch.cat([gs, stacked_inputs], dim=1))
        return torch.einsum('...j,...klj->...kl', gs, tb)


class LATN_NODE(nn.Module):
    def __init__(self,
                 data_desc: lagrdataset.DataDesc,
                 ph_model: LATN,
                 vis_model: LATN,
                 device='cpu'):
        super().__init__()
        assert ph_model.data_desc.history_length == \
            vis_model.data_desc.history_length == \
            data_desc.history_length
        assert ph_model.data_desc.history_timestep == \
            vis_model.data_desc.history_timestep == \
            data_desc.history_timestep

        self.ph_model = ph_model
        self.vis_model = vis_model
        self.data_desc = data_desc
        self.normalization_timescale = 0.0;
        self.Da = 0.0#1/tau**2
        self.Ds = 0.0#1/tau**2
        self.device = device
        self.to(device)

    def set_device(self, device):
        self.device=device

    def get_forcing(self, N, _device):
        delta = torch.diag_embed(torch.ones(N,3,device=_device)) #\delta_{ij}
        del_1 = torch.einsum('nij,nkl->nijkl', delta, delta);
        del_2 = torch.einsum('nik,njl->nijkl', delta, delta);
        del_3 = torch.einsum('nil,njk->nijkl', delta, delta);

        # currently forcing does not depend on sample, so can save time by precomputing
        return -(1/3)*torch.sqrt(self.Ds/5)*del_1 + (1/2)*(torch.sqrt(self.Ds/5)+torch.sqrt(self.Da/3))*del_2 + (1/2)*(torch.sqrt(self.Ds/5)-torch.sqrt(self.Da/3))*del_3;
        
    def set_timescale(self, timescale):
        self.normalization_timescale = timescale
        self.Da = 100.0/timescale**2
        self.Ds = 100.0/timescale**2

    def tangent(self, aij_series):
        """Returns dA/dt = E + H + T
        Assume aij normalized.
        """
        aij = aij_series[:, [-1], ...].reshape(aij_series.shape[0], 3, 3)
        # aij_history = aij_series[:, history_sample_inds, ...].\
        #     flatten(start_dim=2)
        invars = utils.calcInvariants(aij)
        tb = utils.calcFullTensorBasis(aij)
        ph_inputs = [aij_series, invars, tb[..., :10]]
        vis_inputs = [aij_series, invars, tb]

        # print(f"aij isnan = {torch.any(torch.isnan(aij))}")
        # print(f"invars isnan = {torch.any(torch.isnan(invars))}")
        # print(f"tb isnan = {torch.any(torch.isnan(tb))}")
        # print(f"timescale = {self.normalization_timescale}")
        # restricted euler uses un-normalized VGT
        return utils.get_restricted_euler(aij/self.normalization_timescale) \
            + utils.get_latn_ph(self.ph_model, ph_inputs) \
            + utils.get_latn_vis(self.vis_model, vis_inputs)


    def forward(self, inputs):
        """inputs should be exactly the first entry of LagrDataset.__getitem__
        Enforced in tests.
        """
        assert self.normalization_timescale > 0.0, "normalization not set!"
        history_sample_inds = lagrdataset._create_inds(
            self.data_desc.history_length,
            self.data_desc.history_timestep,
            0, self.data_desc.history_length+1)
        aij_cont_series, invars, tb = inputs

        # [0, 1, ..., T] -> [1, ..., T, T + Euler_approx]
        # [1, ..., T, T+Euler] -> [1, ..., T, T+Heun_approx]
        #  Resulting [0, 1, ..., T] -> [1, ..., T, T+Heun]
        num_samples, num_tsteps = aij_cont_series.shape[:2]
        aij_series = aij_cont_series.clone()
        dev = aij_cont_series.device
        for step in range(self.data_desc.rollout_len):
            # Use Heun's method to advance
            # First approx using forward Euler
            daij_1 = self.tangent(
                aij_series[:, history_sample_inds, ...].flatten(end_dim=1)).reshape(num_samples, 9)
            dA = (aij_series[:, -1, ...] + self.data_desc.dt * daij_1).reshape(num_samples, 1, 9)
            aij_series = torch.cat((aij_series[:, 1:, ...], dA), 1)
            # Second approx using predicted
            daij_2 = self.tangent(
                aij_series[:, history_sample_inds, ...].flatten(end_dim=1)).reshape(num_samples, 9)
            # # now instead of advancing whole timeseries, just replace
            # #  last entry
            # # [-2] because we replaced [-1] with Euler approx above
            dW = torch.normal(mean=torch.zeros(num_samples,3,3,device=dev),std=(self.data_desc.dt**(1/2))*torch.ones(num_samples,3,3,device=dev));
            forcing = torch.einsum('nijkl,nkl->nij', self.get_forcing(num_samples, dev), dW).reshape(num_samples, 1, 9)
            dA = (aij_series[:, -2, ...] + (self.data_desc.dt/2) * (daij_1 + daij_2)).reshape(num_samples, 1, 9) + forcing
            aij_series = torch.cat((aij_series[:, :-1, ...], dA), 1)
        return (aij_series.reshape(num_samples, num_tsteps, 3,3))[:, -self.data_desc.rollout_len:, ...]

    def forward_eval(self, A0, T, rank):
        """
        A0 - a starting sample of aij time-history. A0.shape=(num_samples, history_length, 3,3)
        """
        assert self.normalization_timescale > 0.0, "normalization not set!"
        history_sample_inds = torch.tensor(lagrdataset._create_inds(
            self.data_desc.history_length,
            self.data_desc.history_timestep,
            0, self.data_desc.history_length+1))
        num_samples, hl = A0.shape[:2]
        result_aij = torch.zeros((num_samples, hl+T, 3,3), device=rank)
        result_aij[:,:hl,...] = A0.reshape(num_samples, hl, 3, 3)
        for step in range(T):
            daij_1 = self.tangent(
                result_aij[:, history_sample_inds+step, ...].flatten(end_dim=1).flatten(start_dim=-2))
            result_aij[:,step+hl,...] = result_aij[:, step+hl-1, ...] + self.data_desc.dt*daij_1
        return result_aij

    def forward_multistep(self, aij_cont_series, timescale, num_steps):
        """Because we will walk forward individual timesteps,
        (that probably don't align with history_timestep),
        aij_cont_series "continuous series" must have full
        temporal resolution.

        num_steps - number of dt steps to advance
        """
        history_sample_inds = lagrdataset._create_inds(
            self.data_desc.history_length,
            self.data_desc.history_timestep,
            0, self.data_desc.history_length+1)

        # [0, 1, ..., T] -> [1, ..., T, T + Euler_approx]
        # [1, ..., T, T+Euler] -> [1, ..., T, T+Heun_approx]
        #  Resulting [0, 1, ..., T] -> [1, ..., T, T+Heun]
        num_samples, num_tsteps = aij_cont_series.shape[:2]
        aij_series = aij_cont_series.clone()
        for step in range(num_steps):
            # Use Heun's method to advance
            # First approx using forward Euler
            daij_1 = self.forward(
                aij_series[:, history_sample_inds, ...].flatten(end_dim=1),
                timescale).reshape(num_samples, 9)
            aij_series = torch.cat((aij_series[:, 1:, ...],
                                    aij_series[:, [-1], ...] + self.data_desc.dt * daij_1), 1)
            # Second approx using predicted
            daij_2 = self.forward(
                aij_series[:, history_sample_inds, ...].flatten(end_dim=1),
                timescale).reshape(num_samples, 9)
            # now instead of advancing whole timeseries, just replace
            #  last entry
            # [-2] because we replaced [-1] with Euler approx above
            aij_series = torch.cat((aij_series[:, 0:, ...],
                                    aij_series[:, [-2], ...] +
                                    (self.data_desc.dt/2) * (daij_1 + daij_2)), 1)
        return aij_series
