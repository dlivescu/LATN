import os
import shutil
import unittest as ut
import numpy as np
import torch
import utils
import lagrdataset as ld
import distributed

class LagrDatasetTest(ut.TestCase):
    """Tests functionality in LagrDataset.py"""
    tmp_dir = "./__tmp"
    cwd = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):
        self.device = distributed.get_available_device()
        try:
            os.mkdir(self.tmp_dir)
        except OSError as e:
            if e == OSError(60):
                pass
        return
    
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        return

    def test_create_inds(self):
        self.assertEqual(ld._create_inds(10, 5, 2, 33),
                         [[2, 7, 12],
                          [13, 18, 23]])
        self.assertEqual(ld._create_inds(10, 5, 2, 34),
                         [[2, 7, 12],
                          [13, 18, 23],
                          [24, 29, 34]])
        self.assertEqual(ld._create_inds(10, 5, 2, 35),
                         [[2, 7, 12],
                          [13, 18, 23],
                          [24, 29, 34]])
        self.assertEqual(ld._create_inds(10, 10, 2, 12),
                         [[2, 12]])
        # TBNN cases
        self.assertEqual(ld._create_inds(10, 0, 2, 34),
                         [[2], [13], [24]])
        self.assertEqual(ld._create_inds(10, 0, 2, 35),
                         [[2], [13], [24]])
        self.assertEqual(ld._create_inds(10, 0, 2, 36),
                         [[2], [13], [24]])
        self.assertEqual(len(ld._create_inds(10, 0, 2, 35)),
                         len(ld._create_inds(10, 5, 2, 35)))

    def test_get_terminal_inds(self):
        inds = ld._create_inds(10, 0, 2, 35)
        last_inds = [inds[i][-1] for i in range(len(inds))]
        self.assertEqual(last_inds, ld._get_terminal_inds(inds))

    def test_load_pij(self):
        path_to_data = self.tmp_dir + '/pij.npy'
        num_samples = 200
        num_tsteps = 30
        data_shape = (num_samples, num_tsteps, 3, 3)
        sample_data = np.random.random(data_shape)
        sample_data.tofile(path_to_data)
        self.assertIsInstance(ld._load_pij(path_to_data, data_shape, [[1]]),
                              torch.Tensor)
        self.assertEqual(ld._load_pij(path_to_data, data_shape, [[1], [2]]).shape,
                         (num_samples*2, 3, 3))
        pij = ld._load_pij(path_to_data, data_shape, [[1], [2]])
        traces = utils.calc_trace(pij)
        self.assertTrue(torch.allclose(traces, torch.zeros_like(traces)))

    def test_load_vis(self):
        path_to_data = self.tmp_dir + '/vis.npy'
        num_samples = 200
        num_tsteps = 30
        data_shape = (num_samples, num_tsteps, 3, 3)
        sample_data = np.random.random(data_shape)
        sample_data.tofile(path_to_data)
        sample_data = torch.tensor(sample_data)
        self.assertIsInstance(ld._load_vis(path_to_data, data_shape, [[1]]),
                              torch.Tensor)
        self.assertEqual(ld._load_vis(path_to_data, data_shape, [[1], [2]]).shape,
                         (num_samples*2, 3, 3))
        vis = ld._load_vis(path_to_data, data_shape, [[1], [2]])
        self.assertTrue(
            torch.allclose(vis,
                           sample_data[:, [[1], [2]], ...].flatten(end_dim=2)))

    def test_load_da(self):
        num_samples = 200
        num_tsteps = 30
        data_shape = (num_samples, num_tsteps, 3, 3)
        sample_aij = torch.rand(data_shape)
        sample_aij = utils.remove_trace(sample_aij)
        traces = utils.calc_trace(sample_aij)
        self.assertTrue(torch.allclose(traces, torch.zeros_like(traces), atol=1e-6))

        da = ld._load_da(sample_aij, 3e-4, [[2]])
        self.assertIsInstance(da, torch.Tensor)

    def test_LagrDataset_init(self):
        num_samples = 200
        num_tsteps = 30
        aij_shape = (num_samples, num_tsteps, 3, 3)
        target_shape = (num_samples, 3, 3)
        sample_aij = utils.remove_trace(torch.rand(aij_shape))
        sample_target = torch.rand(target_shape)
        data_desc = ld.DataDesc("", aij_shape, "pij", 3e-4, 10, 20, 0.3)
        sample_ds = ld.LagrDataset(sample_aij, sample_target, data_desc)

        self.assertIsInstance(sample_ds,
                              ld.LagrDataset)
        self.assertIsInstance(sample_ds.aij_series, torch.Tensor)
        self.assertIsInstance(sample_ds.target, torch.Tensor)
        self.assertIsInstance(sample_ds.invars, torch.Tensor)
        self.assertIsInstance(sample_ds.tb, torch.Tensor)
        self.assertEqual(sample_ds.data_desc, data_desc)

        self.assertEqual(sample_ds.aij_series.shape, (num_samples, num_tsteps, 9))
        self.assertEqual(sample_ds.target.shape, target_shape)
        self.assertEqual(sample_ds.invars.shape, (num_samples, 5))
        self.assertEqual(sample_ds.tb.shape, (num_samples, 3, 3, 10))

        sample_ds = ld.LagrDataset(sample_aij, sample_target,
                                   data_desc, sym=False)
        self.assertEqual(sample_ds.tb.shape, (num_samples, 3, 3, 16))
        return

    def test_LagrDataset__len__(self):
        num_samples = 200
        num_tsteps = 30
        aij_shape = (num_samples, num_tsteps, 3, 3)
        target_shape = (num_samples, 3, 3)
        sample_aij = utils.remove_trace(torch.rand(aij_shape))
        sample_target = torch.rand(target_shape)
        data_desc = ld.DataDesc("", aij_shape, "pij", 3e-4, 10, 20, 0.3)
        sample_ds = ld.LagrDataset(sample_aij, sample_target, data_desc)
        self.assertEqual(len(sample_ds), num_samples)
        
    def test_LagrDataset_to(self):
        num_samples = 1000
        num_tsteps = 100
        test_data_path = self.cwd + '/test_data/'
        data_shape = (num_samples, num_tsteps, 3, 3)
        target_name = "pij"
        dt = 3e-4
        history_timestep = 5
        history_length = 25
        percent_test = 0.3
        data_desc = ld.DataDesc(test_data_path,
                                data_shape,
                                target_name,
                                dt,
                                history_timestep,
                                history_length,
                                percent_test)
        train_ds, test_ds = ld.LagrDataset.from_file(data_desc)
        self.assertEqual(train_ds.aij_series.device.type, 'cpu')
        if self.device.type != 'cpu':
            train_ds.to(self.device)
            self.assertEqual(train_ds.aij_series.device.type,
                             self.device.type)


    def test_LagrDataset_fromfile(self):
        num_samples = 1000
        num_tsteps = 100
        test_data_path = self.cwd + '/test_data/'
        data_shape = (num_samples, num_tsteps, 3, 3)
        target_name = "pij"
        dt = 3e-4
        history_timestep = 5
        history_length = 25
        percent_test = 0.3
        data_desc = ld.DataDesc(test_data_path,
                                data_shape,
                                target_name,
                                dt,
                                history_timestep,
                                history_length,
                                percent_test)
        train_ds, test_ds = ld.LagrDataset.from_file(data_desc)
        self.assertIsInstance(train_ds, ld.LagrDataset)
        self.assertIsInstance(test_ds, ld.LagrDataset)

        train_ds, test_ds = ld.LagrDataset.from_file(data_desc)
        self.assertIsInstance(train_ds, ld.LagrDataset)
        self.assertIsInstance(test_ds, ld.LagrDataset)
        return

    def test_LagrDataset__getitem__(self):
        num_samples = 1000
        num_tsteps = 100
        test_data_path = self.cwd + '/test_data/'
        data_shape = (num_samples, num_tsteps, 3, 3)
        target_name = "pij"
        dt = 3e-4
        history_timestep = 5
        history_length = 25
        percent_test = 0.3
        data_desc = ld.DataDesc(test_data_path,
                                data_shape,
                                target_name,
                                dt,
                                history_timestep,
                                history_length,
                                percent_test)
        train_ds, test_ds = ld.LagrDataset.from_file(data_desc)
        train_input, train_output = train_ds.__getitem__(34)
        test_input, test_output = test_ds.__getitem__(72)
        train_input = train_ds.reinflate_input(train_input)
        test_input = test_ds.reinflate_input(test_input)
        train_output = train_ds.reinflate_output(train_output)
        test_output = test_ds.reinflate_output(test_output)
        self.assertTrue(torch.allclose(train_input[0],
                                       train_ds.aij_series[34, ...]))
        self.assertTrue(torch.allclose(train_input[1],
                                       train_ds.invars[34, ...]))
        self.assertTrue(torch.allclose(train_input[2],
                                       train_ds.tb[34, ...]))
        self.assertTrue(torch.allclose(train_output[0],
                                       train_ds.target[34, ...]))

        self.assertTrue(torch.allclose(test_input[0],
                                       test_ds.aij_series[72, ...]))
        self.assertTrue(torch.allclose(test_input[1],
                                       test_ds.invars[72, ...]))
        self.assertTrue(torch.allclose(test_input[2],
                                       test_ds.tb[72, ...]))
        self.assertTrue(torch.allclose(test_output[0],
                                       test_ds.target[72, ...]))

    def test_LagrDataset__getitems__(self):
        num_samples = 1000
        num_tsteps = 100
        test_data_path = self.cwd + '/test_data/'
        data_shape = (num_samples, num_tsteps, 3, 3)
        target_name = "pij"
        dt = 3e-4
        history_timestep = 5
        history_length = 25
        percent_test = 0.3
        data_desc = ld.DataDesc(test_data_path,
                                data_shape,
                                target_name,
                                dt,
                                history_timestep,
                                history_length,
                                percent_test)
        train_ds, test_ds = ld.LagrDataset.from_file(data_desc)
        idxs = [34, 72]
        train_input, train_output = train_ds.__getitems__(idxs)
        test_input, test_output = test_ds.__getitems__(idxs)
        train_input = train_ds.reinflate_input(train_input)
        test_input = test_ds.reinflate_input(test_input)
        train_output = train_ds.reinflate_output(train_output)
        test_output = test_ds.reinflate_output(test_output)
        self.assertTrue(torch.allclose(train_input[0],
                                       train_ds.aij_series[idxs, ...]))
        self.assertTrue(torch.allclose(train_input[1],
                                       train_ds.invars[idxs, ...]))
        self.assertTrue(torch.allclose(train_input[2],
                                       train_ds.tb[idxs, ...]))
        self.assertTrue(torch.allclose(train_output[0],
                                       train_ds.target[idxs, ...]))

        self.assertTrue(torch.allclose(test_input[0],
                                       test_ds.aij_series[idxs, ...]))
        self.assertTrue(torch.allclose(test_input[1],
                                       test_ds.invars[idxs, ...]))
        self.assertTrue(torch.allclose(test_input[2],
                                       test_ds.tb[idxs, ...]))
        self.assertTrue(torch.allclose(test_output[0],
                                       test_ds.target[idxs, ...]))
