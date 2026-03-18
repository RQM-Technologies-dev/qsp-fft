"""Tests for qsp.fft.qdft — direct O(N²) QDFT reference implementation."""

import numpy as np
import pytest

from qsp.fft.qdft import iqdft, qdft


class TestQdftOutputShape:
    def test_shape(self):
        q = np.random.default_rng(0).standard_normal((8, 4))
        Q = qdft(q, np.array([1.0, 0.0, 0.0]))
        assert Q.shape == (8, 4)

    def test_shape_non_power_of_two(self):
        q = np.random.default_rng(1).standard_normal((6, 4))
        Q = qdft(q, np.array([0.0, 1.0, 0.0]))
        assert Q.shape == (6, 4)


class TestQdftImpulse:
    """Scalar unit impulse at n=0 should give a flat spectrum."""

    def test_scalar_impulse_flat_spectrum(self):
        N = 8
        q = np.zeros((N, 4))
        q[0, 0] = 1.0  # q[0] = 1 (unit scalar)
        Q = qdft(q, np.array([1.0, 0.0, 0.0]))
        expected = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
        np.testing.assert_allclose(Q, expected, atol=1e-12)

    def test_i_impulse_gives_exp_minus_u_theta(self):
        """q[0] = i should give Q[k] = i for all k (since exp term is 1)."""
        N = 4
        q = np.zeros((N, 4))
        q[0, 1] = 1.0  # q[0] = i
        u = np.array([0.0, 0.0, 1.0])  # k-axis
        Q = qdft(q, u)
        expected = np.tile([0.0, 1.0, 0.0, 0.0], (N, 1))
        np.testing.assert_allclose(Q, expected, atol=1e-12)


class TestQdftConstant:
    """A constant quaternion signal should concentrate energy in bin 0."""

    def test_constant_scalar_signal(self):
        N = 8
        q = np.zeros((N, 4))
        q[:, 0] = 1.0  # all ones (scalar)
        Q = qdft(q, np.array([1.0, 0.0, 0.0]))
        # Bin 0 should be N·e₀
        np.testing.assert_allclose(Q[0], [float(N), 0.0, 0.0, 0.0], atol=1e-10)
        # All other bins should be ~0
        np.testing.assert_allclose(Q[1:], 0.0, atol=1e-10)

    def test_constant_pure_signal(self):
        """Constant q[n] = j: Q[0] = N·j, Q[k≠0] = 0."""
        N = 6
        q = np.zeros((N, 4))
        q[:, 2] = 1.0  # all j
        u = np.array([1.0, 0.0, 0.0])  # i-axis
        Q = qdft(q, u)
        np.testing.assert_allclose(Q[0], [0.0, 0.0, float(N), 0.0], atol=1e-10)
        np.testing.assert_allclose(Q[1:], 0.0, atol=1e-10)


class TestQdftLinearity:
    def test_linearity(self):
        rng = np.random.default_rng(5)
        q1 = rng.standard_normal((8, 4))
        q2 = rng.standard_normal((8, 4))
        u = np.array([1.0, 0.0, 0.0])
        alpha = 3.14
        Q12 = qdft(alpha * q1 + q2, u)
        expected = alpha * qdft(q1, u) + qdft(q2, u)
        np.testing.assert_allclose(Q12, expected, atol=1e-10)


class TestQdftAxisNormalization:
    def test_unnormalised_axis_same_result(self):
        rng = np.random.default_rng(6)
        q = rng.standard_normal((8, 4))
        u_unit = np.array([1.0, 0.0, 0.0])
        u_scaled = np.array([5.0, 0.0, 0.0])
        np.testing.assert_allclose(qdft(q, u_unit), qdft(q, u_scaled), atol=1e-12)


class TestIqdftInverse:
    def test_round_trip_small(self):
        rng = np.random.default_rng(7)
        q = rng.standard_normal((8, 4))
        u = np.array([1.0, 0.0, 0.0])
        reconstructed = iqdft(qdft(q, u), u)
        np.testing.assert_allclose(reconstructed, q, atol=1e-10)

    def test_round_trip_j_axis(self):
        rng = np.random.default_rng(8)
        q = rng.standard_normal((6, 4))
        u = np.array([0.0, 1.0, 0.0])
        reconstructed = iqdft(qdft(q, u), u)
        np.testing.assert_allclose(reconstructed, q, atol=1e-10)

    def test_output_shape(self):
        q = np.random.default_rng(9).standard_normal((5, 4))
        u = np.array([0.0, 0.0, 1.0])
        assert iqdft(qdft(q, u), u).shape == (5, 4)
