
import glob
from dataclasses import dataclass
import utils
import torch
from torch.utils.data import Dataset
import numpy as np

def get_good_inds(arr, nsigma):
    """
    arr.shape = (N,...), apply good_cond to each arr[i,...],
     return inds for first index
    nsigma = number of sigma distances from mean squared norm to include
    """
    def bad_cond(_x, cutoff):
        nentries = _x.shape[0]
        is_too_big_entry = torch.abs(_x) > cutoff
        is_too_big_ind = torch.tensor([torch.any(is_too_big_entry[i,...]) for i in range(nentries)])
        return is_too_big_ind
    def good_cond(_x, cutoff):
        return torch.logical_not(bad_cond(_x, cutoff));
    sq_norm_arr = torch.einsum('n...ij,n...ij->n...',arr,arr)
    cutoff = torch.mean(sq_norm_arr) + nsigma*torch.std(sq_norm_arr)
    good_inds = torch.nonzero(good_cond(sq_norm_arr, cutoff)).flatten()
    return good_inds

def _create_inds(history_length: int, history_tstep: int, start_ind: int,
                 end_ind: int):
    """Handles logic to index datasets along temporal dimension.
    Returns non-overlapping sets of temporal indices, each
    spanning `history_length` with subsequent indices spaced
    `history_tstep` apart.

    To create an index list for "no memory" LATN models, set `history_length`
    equal to spacing between samples, and `history_tstep` to zero

    history_length - length of each index range
    history_tstep - number of timesteps between each element in a range
                    must evenly divide history_length
    start_ind - index to start at, inclusive
    end_ind - index to stop at, inclusive (except when `history_length=0`)
      this policy ensures number of indice sets matches between e.g.
      _create_inds(10, 5, 2, 35) and _create_inds(10, 0, 2, 35)

    ex::
    >>>_create_inds(10, 5, 2, 33)
    [[2, 7, 12],
     [13, 18, 23]]
    >>>_create_inds(10, 5, 2, 34)
    [[2, 7, 12],
    [13, 18, 23],
    [24, 29, 34]]
    >>>_create_inds(10, 0, 2, 34) # for TBNN
    [[2],
    [12],
    [22]]
    """
    num_folds = (end_ind+1-start_ind)//(history_length+1)
    if history_tstep == 0:
        num_steps_per_fold = 1
    else:
        num_steps_per_fold = (history_length//history_tstep) + 1
    retval = [[start_ind + ((history_length+1)*i) + (j*history_tstep)
               for j in range(num_steps_per_fold)]
              for i in range(num_folds)]

    err_str = f"""Empty indice set requested.
    history_length = {history_length}, history_tstep = {history_tstep}
    start_ind = {start_ind}, end_ind = {end_ind}
    """
    assert (len(retval) > 0), err_str
    return retval


def _get_terminal_inds(inds):
    """Enforces consistent way to get indices for target
    after creating full sets for aij"""
    return [inds[i][-1] for i in range(len(inds))]


def _load_pij(path, data_shape, inds):
    """Encapsulates specific logic of loading pressure Hessian
    removes trace"""
    terminal_inds = _get_terminal_inds(inds)
    pij = torch.tensor(np.fromfile(path).reshape(data_shape)
                       [:, terminal_inds, ...])
    pij = pij.flatten(end_dim=1)
    pij = utils.remove_trace(pij)
    return pij


def _load_vis(path, data_shape, inds):
    """Encapsulates specific logic of loading viscous Laplacian"""
    terminal_inds = _get_terminal_inds(inds)
    vis = torch.tensor(np.fromfile(path).reshape(data_shape)
                       [:, terminal_inds, ...])
    vis = vis.flatten(end_dim=1)
    return vis


def _load_da(aij_timeseries, dt, inds):
    """Calculates dA using finite difference
    aij_timeseries - full timeseries, with spacing dt. shape=(N,timesteps,...)
    dt - spacing between sequential entries in aij_timeseries
    inds - temporal indices to calculate dA at, min(inds) >= 2
    """
    inds_for_fd = [[inds[i][-1]-j for j in [2, 1, 0]]
                   for i in range(len(inds))]
    flat_aij = aij_timeseries[:, inds_for_fd, ...].flatten(end_dim=1)
    return utils.second_order_backward_fd(flat_aij, dt)

def _load_a(aij_timeseries, dt, inds, rollout_length:int):
    """
    aij_timeseries - full timeseries, with spacing dt. shape=(N,timesteps,...)
    dt - spacing between sequential entries in aij_timeseries
    inds - temporal indices to calculate dA at, min(inds) >= 2
    rollout_length - how long prediction window to prepare

    inds[-1] + rollout_length < aij_timeseries.shape[1]
    """
    terminal_inds = _get_terminal_inds(inds)
    assert terminal_inds[-1] + rollout_length < aij_timeseries.shape[1]
    inds_for_rollout = [[terminal_inds[i]+j for j in range(rollout_length)]
                        for i in range(len(terminal_inds))]
    rolled_out_aij = aij_timeseries[:, inds_for_rollout, ...].flatten(end_dim=1)
    print(f"rolled out aij.shape = {rolled_out_aij.shape}")
    return rolled_out_aij


@dataclass
class DataDesc:
    """Holds all user-defined properties of dataset.
    Meant to allow easy dataset recreation via `from_file`
    """
    path_to_data: str
    data_shape: tuple
    target_name: str
    dt: float
    history_timestep: int  # in multiples of dt
    history_length: int  # in multiples of dt
    percent_test: float  # in [0, 1)
    rollout_len:int #in multiples of dt. Ignored except for dA datasets

class LagrDataset(Dataset):
    """
    Main interface to create train/test datasets is via
    `LagrDataset.fromfile(path_to_data)`
    """
    def __init__(self, aij_timeseries: torch.tensor, target: torch.tensor,
                 data_desc: DataDesc, sym: bool = True):
        """Construct LagrDataset from tensors.

        aij_timeseries - tensor of shape (N,T,3,3)
        target - tensor of shape (N,3,3)
        sym - bool indicating symmetric (true) or full tensor
           basis expansion (false)
        """
        # check shapes to make sure data makes sense
        if aij_timeseries.shape[0] != target.shape[0]:
            error_str = \
                f"aij and target have incompatible shapes\n \
                aij.shape={aij_timeseries.shape},\
                target.shape={target.shape}\n"
            raise AssertionError(error_str)

        if (data_desc.target_name == "dA"):
            good_inds = get_good_inds(target, 0.5)
        else:
            good_inds = get_good_inds(target, 5)
        print(f"good_inds.shape = {good_inds.shape}")
        # normalize aij by (empirical) Kolmogorov timescale
        self.timescale = \
            utils.calc_characteristic_timescale(aij_timeseries[:, -1, :, :])
        self.aij_series = self.timescale * aij_timeseries[good_inds, ...]
        aij = self.aij_series[:, -1, :, :]
        self.aij_series = self.aij_series.flatten(start_dim=2)

        # calculate and set invariants and tensor basis elements
        self.invars = utils.calcInvariants(aij)
        self.sym = sym
        if self.sym is True:
            self.tb = utils.calcSymTensorBasis(aij)
        else:
            self.tb = utils.calcFullTensorBasis(aij)

        # record target
        self.target = target[good_inds, ...]
        self.data_desc = data_desc

    def __getitem__(self, idx: int) -> tuple:
        # use [idx] to ensure shape is (1, ...) to conform with access
        #  of more than one sample at a time, e.g. (N, ...)
        inp = [self.aij_series[[idx], ...],
               self.invars[[idx], ...],
               self.tb[[idx], ...]]
        out = self.target[[idx], ...].flatten(start_dim=1)
        return (self._flatten_input(inp), out)

    def __len__(self) -> int:
        return self.aij_series.shape[0]

    def __getitems__(self, idxs: list) -> list:
        """
        Consumes idxs and returns a tuple of stacked samples.
        Relies on dataloader to implement a "do nothing" collate_fn,
         otherwise the default collate_fn will try to stack inputs and
         outputs, which is nonsensical.
        If default behavoir is desired, do instead:
         `return [(inp[i, :], out[i, :]) for i in range(len(idxs))]`
        """
        inp = self._flatten_input([self.aij_series[idxs, ...],
                                   self.invars[idxs, ...],
                                   self.tb[idxs, ...]])
        out = self.target[idxs, ...].flatten(start_dim=1)
        return (inp, out)
        

    def _flatten_input(self, inputs):
        """ inputs = self.__getitem__(idx)[0] """
        return torch.cat((inputs[0].flatten(start_dim=1),
                         inputs[1].flatten(start_dim=1),
                          inputs[2].flatten(start_dim=1)), dim=1)
    def reinflate_input(self, flat_inputs):
        """
        flat_inputs = self._flatten_input(...)
        flat_inputs.shape == (num_samples, sample_size)
        """
        num_samples = flat_inputs.shape[0]
        if (self.data_desc.target_name == "dA"):
            aij_history_len = self.data_desc.history_length+1
        else:
            aij_history_len = ((self.data_desc.history_length
                                // self.data_desc.history_timestep) + 1)
        aij_len = (3*3) * aij_history_len
        num_tb = 10 if self.sym else 16
        aij = flat_inputs[:, :aij_len].reshape(num_samples, aij_history_len, 9)
        invars = flat_inputs[:, aij_len:(aij_len+5)].reshape(num_samples, 5)
        tb = flat_inputs[:, (aij_len+5):].reshape(num_samples, 3, 3, num_tb)
        return [aij, invars, tb]

    def reinflate_output(self, flat_outputs):
        if (self.data_desc.target_name == "dA"):
            return [flat_outputs.reshape(flat_outputs.shape[0],
                                         self.data_desc.rollout_len,
                                         3, 3)]
        else:
            return [flat_outputs.reshape(flat_outputs.shape[0], 3, 3)]

    def to(self, device):
        self.aij_series = self.aij_series.to(device)
        self.invars = self.invars.to(device)
        self.tb = self.tb.to(device)
        self.target = self.target.to(device)
        return self

    @classmethod
    def from_file(cls, data_desc: DataDesc, dtype=torch.float32, device='cpu'):
        """ Method encapsulates functionality to read data
        (VGT, PH, VL) from a directory, and process it into
        a LagrDataset

        directory_path - path to numpy files aij, pij, vis
        data_shape - tuple of data sizes, e.g. (N, T, 3, 3)
        target_name - one of ['vis', 'pij', 'dA'] that will be
           training target

        1. Create train/test indices
        2. Read aij from file
        3. Normalize aij
        4. Create target
        5. Create invariants, tensor basis elements
        6. Stack samples together for model consumption
        """
        # 0. Check path uniqueness
        # aij_path = glob.glob(directory_path + "/aij*.bin")
        # if ((len(aij_path) > 1) or (len(target_path) > 1)):
        #     err_str = f"aij or target path is not unique.\n \
        #     len(aij_path)={len(aij_path)}, \
        #     len(target_path)={len(target_path)}\n"
        #     raise AssertionError(err_str)

        # 1. Create train/test indices
        num_tsteps = data_desc.data_shape[1]
        num_train_steps = int((1 - data_desc.percent_test) * num_tsteps)
        train_inds = _create_inds(data_desc.history_length,
                                  data_desc.history_timestep,
                                  0, num_train_steps)
        test_inds = _create_inds(data_desc.history_length,
                                 data_desc.history_timestep,
                                 num_train_steps+1, num_tsteps)

        # 2. Read aij from file
        aij_path = glob.glob(data_desc.path_to_data + "/aij*.bin")
        aij_timeseries = np.fromfile(aij_path[0]).reshape(data_desc.data_shape)
        aij_timeseries = torch.tensor(aij_timeseries)

        # 4 Create target
        search_path = data_desc.path_to_data + "/"\
            + data_desc.target_name + "*.bin"
        print(f"search path = {search_path}")
        target_path = glob.glob(data_desc.path_to_data + "/"
                                + data_desc.target_name + "*.bin")
        if data_desc.target_name == "pij":
            train_target = _load_pij(target_path[0],
                                     data_desc.data_shape, train_inds)
            test_target = _load_pij(target_path[0],
                                    data_desc.data_shape, test_inds)
        elif data_desc.target_name == "vis":
            train_target = _load_vis(target_path[0],
                                     data_desc.data_shape, train_inds)
            test_target = _load_vis(target_path[0],
                                    data_desc.data_shape, test_inds)
        else:  # target_name == "dA":
            # 
            __num_tsteps = data_desc.data_shape[1]-(data_desc.rollout_len+1)
            __num_train_steps = int((1 - data_desc.percent_test) * __num_tsteps)
            # in the following we take all aij timesteps,
            #  as we perform the rollout prediction, the
            #  temporal kernel will land between samples otherwise.
            train_inds = _create_inds(data_desc.history_length,
                                      1, # need dense history
                                      0,
                                      __num_train_steps)
            test_inds = _create_inds(data_desc.history_length,
                                     1,
                                     __num_train_steps+1,
                                     __num_tsteps)
            train_target = _load_a(aij_timeseries,
                                   data_desc.dt,
                                   train_inds,
                                   data_desc.rollout_len)
            test_target = _load_a(aij_timeseries,
                                  data_desc.dt,
                                  test_inds,
                                  data_desc.rollout_len)

        # 6 Stack samples together for model consumption
        train_aij = aij_timeseries[:, train_inds, ...]\
            .detach().clone().flatten(end_dim=1)
        test_aij = aij_timeseries[:, test_inds, ...]\
            .detach().clone().flatten(end_dim=1)

        # 7 Return train and test LagrDatasets
        train_ds = cls(train_aij.to(dtype),
                       train_target.to(dtype),
                       data_desc,
                       sym=(data_desc.target_name == "pij"))
        test_ds = cls(test_aij.to(dtype),
                      test_target.to(dtype),
                      data_desc,
                      sym=(data_desc.target_name == "pij"))

        train_ds.to(device)
        test_ds.to(device)
        return train_ds, test_ds
