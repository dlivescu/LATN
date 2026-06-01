import os
import shutil
import unittest as ut
import numpy as np
import torch
import utils

torch.manual_seed(0)


def _rand_aij(n, dtype=torch.float64):
    """Random VGT batch (N,3,3)."""
    return torch.randn(n, 3, 3, dtype=dtype)


def _rand_orthogonal(n, dtype=torch.float64):
    """Batch of orthogonal matrices via QR of random matrices."""
    q, _ = torch.linalg.qr(torch.randn(n, 3, 3, dtype=dtype))
    return q


def _sym(a):
    return 0.5 * (a + a.transpose(-2, -1))


def _skew(a):
    return 0.5 * (a - a.transpose(-2, -1))


class UtilsTest(ut.TestCase):
    """Analytic and property-based checks of the physics/math kernels that sit
    on the training path (utils.py). All in float64 on CPU."""

    # ------------------------------------------------------------------ trace
    def test_calc_trace(self):
        data_shape = (30, 50, 60, 3, 3)
        arr = torch.ones(data_shape)
        self.assertTrue(torch.allclose(utils.calc_trace(arr),
                                       3 * torch.ones(data_shape[:-2]),
                                       atol=5e-7))

    def test_remove_trace(self):
        for data_shape in [(30, 50, 60, 3, 3), (30, 3, 3)]:
            arr = torch.rand(data_shape)
            detraced = utils.remove_trace(arr)
            # trace of the de-traced array is ~0
            self.assertTrue(torch.allclose(utils.calc_trace(detraced),
                                           torch.zeros(data_shape[:-2]),
                                           atol=5e-7))
            # and it equals arr - (1/3) tr(arr) I exactly
            I = torch.eye(3)
            expected = arr - (1/3) * utils.calc_trace(arr)[..., None, None] * I
            self.assertTrue(torch.allclose(detraced, expected, atol=1e-6))

    # ------------------------------------------------------------- invariants
    def test_invariants_shape(self):
        self.assertEqual(utils.calcInvariants(_rand_aij(17)).shape, (17, 5))

    def test_invariants_pure_symmetric(self):
        # A symmetric => W=0 => l2=l4=l5=0, l1 = |S|_F^2 = tr(S^2)
        S = _sym(_rand_aij(64))
        l = utils.calcInvariants(S)
        self.assertTrue(torch.allclose(l[:, 1], torch.zeros(64, dtype=torch.float64), atol=1e-10))
        self.assertTrue(torch.allclose(l[:, 3], torch.zeros(64, dtype=torch.float64), atol=1e-10))
        self.assertTrue(torch.allclose(l[:, 4], torch.zeros(64, dtype=torch.float64), atol=1e-10))
        self.assertTrue(torch.allclose(l[:, 0],
                                       torch.einsum('nij,nij->n', S, S),
                                       atol=1e-9))

    def test_invariants_pure_antisymmetric(self):
        # A antisymmetric => S=0 => l1=l3=l4=l5=0, l2 = -|W|_F^2 <= 0
        W = _skew(_rand_aij(64))
        l = utils.calcInvariants(W)
        for k in (0, 2, 3, 4):
            self.assertTrue(torch.allclose(l[:, k], torch.zeros(64, dtype=torch.float64), atol=1e-10))
        self.assertTrue(torch.allclose(l[:, 1],
                                       -torch.einsum('nij,nij->n', W, W),
                                       atol=1e-9))
        self.assertTrue(torch.all(l[:, 1] <= 1e-12))

    def test_invariants_orthogonal_invariance(self):
        # invariants are unchanged under A -> Q A Q^T for orthogonal Q
        A = _rand_aij(64)
        Q = _rand_orthogonal(64)
        Arot = torch.einsum('nij,njk,nlk->nil', Q, A, Q)
        self.assertTrue(torch.allclose(utils.calcInvariants(A),
                                       utils.calcInvariants(Arot),
                                       atol=1e-8))

    # ----------------------------------------------------------- tensor basis
    def test_sym_basis_shape_and_symmetry(self):
        A = _rand_aij(32)
        tb = utils.calcSymTensorBasis(A)
        self.assertEqual(tb.shape, (32, 3, 3, 10))
        # every symmetric-basis tensor must itself be symmetric
        for k in range(10):
            tk = tb[..., k]
            self.assertTrue(torch.allclose(tk, tk.transpose(-2, -1), atol=1e-9))
        # t3, t4 are the deviatoric (traceless) S^2, W^2 terms
        for k in (2, 3):
            self.assertTrue(torch.allclose(utils.calc_trace(tb[..., k]),
                                           torch.zeros(32, dtype=torch.float64), atol=1e-9))

    def test_skew_basis_shape_and_antisymmetry(self):
        A = _rand_aij(32)
        tb = utils.calcSkewSymTensorBasis(A)
        self.assertEqual(tb.shape, (32, 3, 3, 6))
        for k in range(6):
            tk = tb[..., k]
            self.assertTrue(torch.allclose(tk, -tk.transpose(-2, -1), atol=1e-9))

    def test_full_basis_shape_and_concat(self):
        A = _rand_aij(8)
        full = utils.calcFullTensorBasis(A)
        self.assertEqual(full.shape, (8, 3, 3, 16))
        self.assertTrue(torch.allclose(full[..., :10],
                                       utils.calcSymTensorBasis(A)))
        self.assertTrue(torch.allclose(full[..., 10:],
                                       utils.calcSkewSymTensorBasis(A)))

    def test_basis_device_propagation(self):
        A = _rand_aij(4)
        self.assertEqual(utils.calcSymTensorBasis(A).device, A.device)

    # ------------------------------------------------------- restricted euler
    def test_restricted_euler_traceless(self):
        A = _rand_aij(64)
        re = utils.get_restricted_euler(A)
        self.assertTrue(torch.allclose(utils.calc_trace(re),
                                       torch.zeros(64, dtype=torch.float64), atol=1e-9))

    def test_restricted_euler_value(self):
        A = _rand_aij(64)
        I = torch.eye(3, dtype=A.dtype)
        trA2 = torch.einsum('nik,nki->n', A, A)
        expected = -torch.matmul(A, A) + (1/3) * trA2[:, None, None] * I
        self.assertTrue(torch.allclose(utils.get_restricted_euler(A),
                                       expected, atol=1e-9))

    # ------------------------------------------------- characteristic timescale
    def test_timescale_known_value(self):
        # all samples equal a fixed symmetric S => tau = 1/|S|_F
        S = _sym(_rand_aij(1)).repeat(50, 1, 1)
        normS = torch.sqrt(torch.einsum('nij,nij->n', S, S)[0])
        tau = utils.calc_characteristic_timescale(S)
        self.assertTrue(torch.allclose(tau, 1.0 / normS, atol=1e-9))
        self.assertGreater(float(tau), 0.0)

    def test_timescale_filters_outliers(self):
        good = _sym(_rand_aij(100))
        # tau computed on the clean batch
        tau_clean = utils.calc_characteristic_timescale(good)
        # inject inf and a huge (>5e3) sample; result must ignore them
        polluted = good.clone()
        polluted[3] = float('inf')
        polluted[7] = 1e4
        tau_filtered = utils.calc_characteristic_timescale(polluted)
        # recompute clean tau without the two polluted rows
        keep = [i for i in range(100) if i not in (3, 7)]
        tau_ref = utils.calc_characteristic_timescale(good[keep])
        self.assertTrue(torch.allclose(tau_filtered, tau_ref, atol=1e-9))
        self.assertTrue(torch.isfinite(tau_filtered))

    # ----------------------------------------------- second-order backward FD
    def test_backward_fd_exact_on_quadratic(self):
        # 2nd-order backward difference is exact for polynomials up to degree 2.
        # A(t) = a + b t + c t^2 sampled at [t0-2dt, t0-dt, t0];
        # derivative at t0 is b + 2 c t0.
        dt = 3e-4
        n = 16
        a = torch.randn(n, 3, 3, dtype=torch.float64)
        b = torch.randn(n, 3, 3, dtype=torch.float64)
        c = torch.randn(n, 3, 3, dtype=torch.float64)
        t0 = 0.37
        ts = torch.tensor([t0 - 2*dt, t0 - dt, t0], dtype=torch.float64)
        # arr shape (n, 3 timesteps, 3, 3)
        arr = (a[:, None] + b[:, None]*ts[None, :, None, None]
               + c[:, None]*(ts[None, :, None, None]**2))
        deriv = utils.second_order_backward_fd(arr, dt)
        expected = b + 2*c*t0
        self.assertTrue(torch.allclose(deriv, expected, atol=1e-6))

    def test_backward_fd_exact_on_linear(self):
        dt = 1e-3
        n = 8
        a = torch.randn(n, 3, 3, dtype=torch.float64)
        b = torch.randn(n, 3, 3, dtype=torch.float64)
        ts = torch.tensor([-2*dt, -dt, 0.0], dtype=torch.float64)
        arr = a[:, None] + b[:, None]*ts[None, :, None, None]
        deriv = utils.second_order_backward_fd(arr, dt)
        self.assertTrue(torch.allclose(deriv, b, atol=1e-7))


if __name__ == '__main__':
    ut.main()
