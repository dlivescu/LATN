import os
import unittest as ut
import torch
from torch.linalg import matrix_norm
from dataclasses import replace
import utils
import lagrdataset
import latn
import latn_globals
import distributed

DATA_DESC = lagrdataset.\
    DataDesc(
        os.path.dirname(os.path.abspath(__file__)) + '/test_data',
             (16, 100, 3, 3), #(num_samples, num_tsteps, 3, 3)
             "pij", # target_name,
             3e-4, # dt,
             5, # history_timestep,
             25, # history_length,
             0.3) # percent_test)
NETWORK_DESC = latn.LATNDesc(
    2,  # num_layers
    20,  # num_units
    torch.nn.ReLU,  # activation
    10,  # input_len
    10,  # output_len
    0.2)  # dropout_rate


class TestScalarHistoryConv(ut.TestCase):
    def setUp(self):
        """create datasets to act on with LATN models"""
        self.train_ds, self.test_ds = lagrdataset.\
            LagrDataset.from_file(DATA_DESC)
        self.num_tsteps = self.train_ds.data_desc.history_length\
            // self.train_ds.data_desc.history_timestep + 1
        self.num_filters = 10
        self.conv = latn.ScalarHistoryConv(self.num_tsteps, self.num_filters)

    def test__init__(self):
        self.assertIsInstance(self.conv, latn.ScalarHistoryConv)

    def test_forward(self):
        self.assertEqual(self.conv.forward(self.train_ds.aij_series).shape,
                         (self.train_ds.aij_series.shape[0], self.num_filters))


class TestTensorHistoryConv(ut.TestCase):
    def setUp(self):
        self.train_ds, self.test_ds = lagrdataset.\
            LagrDataset.from_file(DATA_DESC)
        self.num_tsteps = self.train_ds.data_desc.history_length\
            // self.train_ds.data_desc.history_timestep + 1
        self.num_filters = 10
        self.conv = latn.TensorHistoryConv(self.num_tsteps, self.num_filters)

    def test__init__(self):
        self.assertIsInstance(self.conv, latn.TensorHistoryConv)

    def test_forward(self):
        self.assertEqual(self.conv.forward(self.train_ds.aij_series).shape,
                         (self.train_ds.aij_series.shape[0], self.num_filters))


class TestConstrainedTensorHistoryConv(ut.TestCase):
    def setUp(self):
        self.train_ds, self.test_ds = lagrdataset.\
            LagrDataset.from_file(DATA_DESC)
        self.num_tsteps = self.train_ds.data_desc.history_length\
            // self.train_ds.data_desc.history_timestep + 1
        self.num_filters = 10
        self.conv = latn.ConstrainedTensorHistoryConv(
            self.num_tsteps, self.num_filters)

    def test__init__(self):
        self.assertIsInstance(self.conv, latn.ConstrainedTensorHistoryConv)

    def test_forward(self):
        self.assertEqual(self.conv.forward(self.train_ds.aij_series).shape,
                         (self.train_ds.aij_series.shape[0], self.num_filters))

    def test_constraint(self):
        """The constrained conv must split its filters into a purely symmetric
        and a purely antisymmetric half."""
        filters = self.conv._get_conv_filters().clone().detach()
        self.assertEqual(filters.shape, (self.num_tsteps, self.num_filters, 9))
        filters = filters.reshape((self.num_tsteps * self.num_filters, 3, 3))
        def sym_metric(mat):
            sym_mat = 0.5*(mat + mat.transpose(-2, -1))
            asym_mat = 0.5*(mat - mat.transpose(-2, -1))
            return (matrix_norm(sym_mat) - matrix_norm(asym_mat)) / \
                (matrix_norm(sym_mat) + matrix_norm(asym_mat))
        sym_metrics = [sym_metric(filters[i, :, :])
                       for i in range(filters.shape[0])]
        tol = 1e-7
        close_to_one = [torch.abs(sym_metrics[i]-1) < tol
                        for i in range(len(sym_metrics))]
        close_to_negative_one = [torch.abs(sym_metrics[i]+1) < tol
                                 for i in range(len(sym_metrics))]
        results = [close_to_one[i] or close_to_negative_one[i]
                   for i in range(len(sym_metrics))]
        for i in range(len(results)):
            self.assertTrue(results[i])


class TestFFN(ut.TestCase):
    def test_ffn_shape(self):
        ff = latn.FFN(10, 7, 2, 20, torch.nn.ReLU)
        out = ff(torch.randn(13, 10))
        self.assertEqual(tuple(out.shape), (13, 7))

    def test_ffn_zero_layers(self):
        ff = latn.FFN(10, 7, 0, 20, torch.nn.ReLU)
        self.assertEqual(tuple(ff(torch.randn(4, 10)).shape), (4, 7))

    def test_skip_ffn_shape(self):
        ff = latn.Skip_FFN(10, 7, 2, 20, torch.nn.ReLU)
        out = ff(torch.randn(13, 10))
        self.assertEqual(tuple(out.shape), (13, 7))


class TestLATN(ut.TestCase):
    def setUp(self):
        self.device = distributed.get_available_device()
        ph_data_desc = DATA_DESC
        vis_data_desc = replace(DATA_DESC, target_name="vis")
        vis_network_desc = replace(NETWORK_DESC, output_len=16)
        self.ph_train_ds, self.ph_test_ds = lagrdataset.\
            LagrDataset.from_file(ph_data_desc, device=self.device)
        self.vis_train_ds, self.vis_test_ds = lagrdataset.\
            LagrDataset.from_file(vis_data_desc, device=self.device)

        self.network_desc = NETWORK_DESC
        self.ph_model = latn.LATN(self.ph_train_ds.data_desc,
                                  self.network_desc,
                                  latn.TensorHistoryConv,
                                  latn.FFN,
                                  device=self.device)
        self.vis_model = latn.LATN(self.vis_train_ds.data_desc,
                                   vis_network_desc,
                                   latn.TensorHistoryConv,
                                   latn.FFN,
                                   device=self.device)

    def test__init__(self):
        self.assertIsInstance(self.ph_model, latn.LATN)
        for ps in self.ph_model.parameters():
            self.assertEqual(ps.device.type, self.device.type)
        for ps in self.vis_model.parameters():
            self.assertEqual(ps.device.type, self.device.type)

    def test_forward(self):
        sample_input, sample_output = self.ph_train_ds.__getitem__(0)
        sample_input = self.ph_train_ds.reinflate_input(sample_input)
        sample_output = self.ph_train_ds.reinflate_output(sample_output)[0]
        self.assertEqual(sample_input[0].device.type, self.device.type)
        self.assertEqual(self.ph_model.forward(sample_input).shape,
                         sample_output.shape)

        sample_input, sample_output = self.vis_train_ds.__getitem__(0)
        sample_input = self.vis_train_ds.reinflate_input(sample_input)
        sample_output = self.vis_train_ds.reinflate_output(sample_output)[0]
        self.assertEqual(self.vis_model.forward(sample_input).shape,
                         sample_output.shape)


class TestLATN_NODE(ut.TestCase):
    """The stochastic neural-ODE assembly. CPU-only so the rollout is
    deterministic under a fixed seed (no DDP / nccl)."""
    def setUp(self):
        self.device = 'cpu'
        self.rollout = 2
        self.data_desc = lagrdataset.DataDesc(
            DATA_DESC.path_to_data, DATA_DESC.data_shape, "dA",
            DATA_DESC.dt, DATA_DESC.history_timestep,
            DATA_DESC.history_length, DATA_DESC.percent_test,
            16,             # num_samples (keep it small/fast)
            self.rollout)   # rollout_len
        self.train_ds, self.test_ds = lagrdataset.\
            LagrDataset.from_file(self.data_desc, device=self.device)
        nd = latn.LATNDesc(2, 20, torch.nn.ReLU,
                           latn_globals.NUM_INVARIANTS + 10, 0, 0.0)
        self.ph_model = latn.LATN(self.data_desc,
                                  replace(nd, output_len=latn_globals.NUM_PIJ_OUTPUTS),
                                  latn.ConstrainedTensorHistoryConv, latn.FFN)
        self.vis_model = latn.LATN(self.data_desc,
                                   replace(nd, output_len=latn_globals.NUM_VIS_OUTPUTS),
                                   latn.ConstrainedTensorHistoryConv, latn.FFN)
        self.node = latn.LATN_NODE(self.data_desc, self.ph_model, self.vis_model)
        self.node.set_timescale(self.train_ds.timescale)
        # one collated, reinflated batch (the exact Trainer input)
        src, _ = self.train_ds.__getitems__(list(range(8)))
        self.inp = self.train_ds.reinflate_input(src)

    def test__init__(self):
        self.assertIsInstance(self.node, latn.LATN_NODE)

    def test_set_timescale_sets_noise(self):
        self.assertGreater(float(self.node.normalization_timescale), 0.0)
        self.assertGreater(float(self.node.Da), 0.0)
        self.assertGreater(float(self.node.Ds), 0.0)

    def test_tangent_shape(self):
        # tangent expects the strided (history_timestep) sub-sampled history
        hsi = lagrdataset._create_inds(self.data_desc.history_length,
                                       self.data_desc.history_timestep,
                                       0, self.data_desc.history_length + 1)
        sub = self.inp[0][:, hsi, ...].flatten(end_dim=1)
        self.assertEqual(tuple(self.node.tangent(sub).shape), (8, 3, 3))

    def test_get_forcing_shape_and_isotropy(self):
        f = self.node.get_forcing(5, 'cpu')
        self.assertEqual(tuple(f.shape), (5, 3, 3, 3, 3))
        # forcing does not depend on the sample index
        for n in range(1, 5):
            self.assertTrue(torch.allclose(f[0], f[n]))

    def test_forward_shape(self):
        out = self.node.forward(self.inp)
        self.assertEqual(tuple(out.shape), (8, self.rollout, 3, 3))

    def test_forward_seeded_determinism(self):
        torch.manual_seed(42)
        a = self.node.forward(self.inp)
        torch.manual_seed(42)
        b = self.node.forward(self.inp)
        self.assertTrue(torch.allclose(a, b))
        # different seed -> different stochastic realization
        torch.manual_seed(43)
        c = self.node.forward(self.inp)
        self.assertFalse(torch.allclose(a, c))

    def test_forward_noiseless_matches_reference_heun(self):
        # With Da=Ds=0 the rollout is a deterministic Heun step; verify it
        # against an independent hand-rolled Heun built from tangent().
        self.node.Da = torch.tensor(0.0)
        self.node.Ds = torch.tensor(0.0)
        torch.manual_seed(0)
        out = self.node.forward(self.inp)

        hsi = lagrdataset._create_inds(self.data_desc.history_length,
                                       self.data_desc.history_timestep,
                                       0, self.data_desc.history_length + 1)
        dt = self.data_desc.dt
        series = self.inp[0].clone()
        n = series.shape[0]
        ref = []
        for _ in range(self.rollout):
            d1 = self.node.tangent(series[:, hsi, ...].flatten(end_dim=1)).reshape(n, 9)
            euler = (series[:, -1, ...] + dt * d1).reshape(n, 1, 9)
            series = torch.cat((series[:, 1:, ...], euler), 1)
            d2 = self.node.tangent(series[:, hsi, ...].flatten(end_dim=1)).reshape(n, 9)
            heun = (series[:, -2, ...] + (dt/2)*(d1 + d2)).reshape(n, 1, 9)
            series = torch.cat((series[:, :-1, ...], heun), 1)
            ref.append(heun.reshape(n, 3, 3))
        ref = torch.stack(ref, dim=1)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_forward_eval_shape(self):
        T = 3
        A0 = self.test_ds.aij_series[:4, ...]   # (4, hl+1, 9)
        hl = A0.shape[1]
        out = self.node.forward_eval(A0, T, 'cpu')
        self.assertEqual(tuple(out.shape), (4, hl + T, 3, 3))


if __name__ == '__main__':
    ut.main()
