import os
import shutil
import unittest as ut
import numpy as np
import torch
import utils


class UtilsTest(ut.TestCase):
    """Tests functionality in LagrDataset.py"""
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_calc_trace(self):
        num_samples = 30
        num_tsteps = 50
        another_dim = 60
        data_shape = (num_samples, num_tsteps, another_dim, 3, 3)
        arr = torch.ones(data_shape)
        self.assertTrue(
            torch.allclose(utils.calc_trace(arr),
                           3*torch.ones(data_shape[:-2]),
                           atol=5e-7))

    def test_remove_trace(self):
        num_samples = 30
        num_tsteps = 50
        another_dim = 60
        data_shape = (num_samples, num_tsteps, another_dim, 3, 3)
        arr = torch.rand(data_shape)
        self.assertTrue(
            torch.allclose(utils.calc_trace(utils.remove_trace(arr)),
                           torch.zeros(data_shape[:-2]),
                           atol=5e-7))
        num_samples = 30
        data_shape = (num_samples, 3, 3)
        arr = torch.rand(data_shape)
        self.assertTrue(
            torch.allclose(utils.calc_trace(utils.remove_trace(arr)),
                           torch.zeros(data_shape[:-2]),
                           atol=5e-7))
