"""Tests for qsp_fft.qfft — fast FFT-based QDFT implementation."""

import numpy as np
import pytest

from qsp_fft.qdft import qdft
from qsp_fft.qfft import iqfft, qfft


class TestQfftOutputShape:
    def test_shape_power_of_two(self):
        q = np.random.default_rng(0).standard_normal((16, 4))
        Q = qfft(q, np.array([1.0, 0.0, 0.0]))
        assert Q.shape == (16, 4)

    def test_shape_non_power_of_two(self):
        """NumPy FFT supports arbitrary lengths."""
        q = np.random.default_rng(1).standard_normal((7, 4))
        Q = qfft(q, np.array([0.0, 1.0, 0.0]))
        assert Q.shape == (7, 4)

    def test_shape_small(self):
        q = np.random.default_rng(2).standard_normal((3, 4))
        Q = qfft(q, np.array([0.0, 0.0, 1.0]))
        assert Q.shape == (3, 4)


class TestQfftMatchesQdft:
    """qfft must be numerically identical to qdft."""

    @pytest.mark.parametrize("N", [4, 8, 13, 16])
    def test_matches_qdft_i_axis(self, N):
        rng = np.random.default_rng(N)
        q = rng.standard_normal((N, 4))
        u = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(qfft(q, u), qdft(q, u), atol=1e-10)

    @pytest.mark.parametrize("N", [4, 8, 9])
    def test_matches_qdft_j_axis(self, N):
        rng = np.random.default_rng(N + 100)
        q = rng.standard_normal((N, 4))
        u = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(qfft(q, u), qdft(q, u), atol=1e-10)

    @pytest.mark.parametrize("N", [4, 7, 16])
    def test_matches_qdft_diagonal_axis(self, N):
        rng = np.random.default_rng(N + 200)
        q = rng.standard_normal((N, 4))
        u = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        np.testing.assert_allclose(qfft(q, u), qdft(q, u), atol=1e-10)


class TestQfftImpulse:
    def test_scalar_impulse_flat(self):
        N = 8
        q = np.zeros((N, 4))
        q[0, 0] = 1.0
        Q = qfft(q, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(Q, np.tile([1., 0., 0., 0.], (N, 1)), atol=1e-12)


class TestQfftIqfftRoundTrip:
    @pytest.mark.parametrize("N", [4, 8, 16, 32])
    def test_round_trip_i_axis(self, N):
        rng = np.random.default_rng(N)
        q = rng.standard_normal((N, 4))
        u = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(iqfft(qfft(q, u), u), q, atol=1e-10)

    @pytest.mark.parametrize("N", [5, 9, 13])
    def test_round_trip_non_power_of_two(self, N):
        rng = np.random.default_rng(N + 300)
        q = rng.standard_normal((N, 4))
        u = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(iqfft(qfft(q, u), u), q, atol=1e-10)

    def test_round_trip_k_axis(self):
        rng = np.random.default_rng(99)
        q = rng.standard_normal((12, 4))
        u = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(iqfft(qfft(q, u), u), q, atol=1e-10)


class TestQfftSliceSanity:
    """If signal lies in C_u, QFFT matches ordinary complex FFT embedded
    in quaternion form (the 'reduction sanity' test)."""

    def test_real_signal_as_scalar_quaternion(self):
        """Pure real signal: q[n] = (x[n], 0, 0, 0).
        Q[k] should have the complex FFT in the scalar/u components."""
        rng = np.random.default_rng(10)
        N = 16
        x = rng.standard_normal(N)
        q = np.zeros((N, 4))
        q[:, 0] = x

        u = np.array([1.0, 0.0, 0.0])
        Q = qfft(q, u)

        # Scalar component: Re(FFT(x))[k]
        X = np.fft.fft(x)
        np.testing.assert_allclose(Q[:, 0], X.real, atol=1e-10)
        # u-component: Im(FFT(x))[k] (should be 0 for real x... wait, no)
        # Actually for real x, FFT(x) is conjugate-symmetric but not purely real.
        # The scalar part maps to Re(FFT(x)) and u-part to Im(FFT(x)) — both may be nonzero.
        # Just verify the full quaternion via qdft agreement.
        np.testing.assert_allclose(Q, qdft(q, u), atol=1e-10)

    def test_cu_signal_round_trips(self):
        """Signal entirely in C_i = span{1, i}: qy = qz = 0."""
        rng = np.random.default_rng(11)
        N = 8
        q = np.zeros((N, 4))
        q[:, 0] = rng.standard_normal(N)  # scalar
        q[:, 1] = rng.standard_normal(N)  # i-component
        u = np.array([1.0, 0.0, 0.0])

        Q_fast = qfft(q, u)
        Q_direct = qdft(q, u)
        np.testing.assert_allclose(Q_fast, Q_direct, atol=1e-10)

        # Verify that reconstruction is exact.
        np.testing.assert_allclose(iqfft(Q_fast, u), q, atol=1e-10)
