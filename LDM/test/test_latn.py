import os
import unittest as ut
import torch
from torch.linalg import matrix_norm
import utils
import lagrdataset
import latn
import distributed

DATA_DESC = lagrdataset.\
    DataDesc(
        os.path.dirname(os.path.abspath(__file__)) + '/test_data',
             (1000, 100, 3, 3), #(num_samples, num_tsteps, 3, 3)
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

class TestLATN(ut.TestCase):
    def setUp(self):
        self.device = distributed.get_available_device()
        ph_data_desc = DATA_DESC
        vis_data_desc = lagrdataset.DataDesc(DATA_DESC.path_to_data,
                                             DATA_DESC.data_shape,
                                             "vis",
                                             DATA_DESC.dt,
                                             DATA_DESC.history_timestep,
                                             DATA_DESC.history_length,
                                             DATA_DESC.percent_test)
        vis_network_desc = latn.LATNDesc(NETWORK_DESC.num_layers,
                                         NETWORK_DESC.num_units,
                                         NETWORK_DESC.activation,
                                         NETWORK_DESC.input_len,
                                         16,
                                         NETWORK_DESC.dropout_rate)
        self.ph_train_ds, self.ph_test_ds = lagrdataset.\
            LagrDataset.from_file(ph_data_desc, device=self.device)
        self.vis_train_ds, self.vis_test_ds = lagrdataset.\
            LagrDataset.from_file(vis_data_desc, device=self.device)
        
        self.network_desc = NETWORK_DESC
        self.ph_model = latn.LATN(self.ph_train_ds.data_desc,
                                  self.network_desc,
                                  latn.TensorHistoryConv,
                                  device=self.device)
        self.vis_model = latn.LATN(self.vis_train_ds.data_desc,
                                   vis_network_desc,
                                   latn.TensorHistoryConv,
                                   device=self.device)

    def test__init__(self):
        self.assertIsInstance(self.ph_model, latn.LATN)
        for ps in self.ph_model.parameters():
            self.assertEqual(ps.device.type, self.device.type)
        for ps in self.vis_model.parameters():
            self.assertEqual(ps.device.type, self.device.type)


    def test_forward(self):
        sample_input, sample_output = self.ph_train_ds.__getitem__(0)
        self.assertEqual(sample_input[0].device.type, self.device.type)
        self.assertEqual(self.ph_model.forward(sample_input).shape,
                         sample_output[0].shape)

        sample_input, sample_output = self.vis_train_ds.__getitem__(0)
        self.assertEqual(sample_input[0].device.type, self.device.type)
        self.assertEqual(self.vis_model.forward(sample_input).shape,
                         sample_output[0].shape)

class TestLATN(ut.TestCase):
    def setUp(self):
        self.device = distributed.get_available_device()
        ph_data_desc = DATA_DESC
        vis_data_desc = lagrdataset.DataDesc(DATA_DESC.path_to_data,
                                             DATA_DESC.data_shape,
                                             "vis",
                                             DATA_DESC.dt,
                                             DATA_DESC.history_timestep,
                                             DATA_DESC.history_length,
                                             DATA_DESC.percent_test)
        vis_network_desc = latn.LATNDesc(NETWORK_DESC.num_layers,
                                         NETWORK_DESC.num_units,
                                         NETWORK_DESC.activation,
                                         NETWORK_DESC.input_len,
                                         16,
                                         NETWORK_DESC.dropout_rate)
        self.ph_train_ds, self.ph_test_ds = lagrdataset.\
            LagrDataset.from_file(ph_data_desc, device=self.device)
        self.vis_train_ds, self.vis_test_ds = lagrdataset.\
            LagrDataset.from_file(vis_data_desc, device=self.device)

        self.network_desc = NETWORK_DESC
        self.ph_model = latn.LATN(self.ph_train_ds.data_desc,
                                  self.network_desc,
                                  latn.TensorHistoryConv,
                                  device=self.device)
        self.vis_model = latn.LATN(self.vis_train_ds.data_desc,
                                   vis_network_desc,
                                   latn.TensorHistoryConv,
                                   device=self.device)
        self.latn_node = latn.LATN_NODE(self.ph_train_ds.data_desc,
                                        self.ph_model,
                                        self.vis_model)

    def test__init__(self):
        self.assertIsInstance(self.latn_node, latn.LATN_NODE)

    def test_forward(self):
        dA = self.latn_node.forward(self.ph_train_ds.aij_series,
                                    self.ph_train_ds.timescale)
        num_samples = self.ph_train_ds.aij_series.shape[0]
        self.assertEqual(dA.shape, (num_samples, 3, 3))
                         

    def test_forward_multistep(self):
        num_samples = self.ph_train_ds.aij_series.shape[0]
        aij_cont_series = torch.rand(num_samples,
                                     self.ph_train_ds.data_desc.history_length + 1,
                                     9).to(self.device)
        self.latn_node.forward_multistep(aij_cont_series,
                                         self.ph_train_ds.timescale,
                                         5,
                                         3e-4)
