"""Tests for round-trip inversion (qdft/iqdft and qfft/iqfft)."""

import numpy as np
import pytest

from qsp_fft.qdft import iqdft, qdft
from qsp_fft.qfft import iqfft, qfft


AXES = [
    ("i", np.array([1.0, 0.0, 0.0])),
    ("j", np.array([0.0, 1.0, 0.0])),
    ("k", np.array([0.0, 0.0, 1.0])),
    ("diag", np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
]

SIZES = [1, 4, 8, 16]


class TestQdftInverse:
    @pytest.mark.parametrize("N", SIZES)
    @pytest.mark.parametrize("axis_name,axis", AXES)
    def test_round_trip(self, N, axis_name, axis):
        rng = np.random.default_rng(hash((N, axis_name)) % 2**32)
        q = rng.standard_normal((N, 4))
        reconstructed = iqdft(qdft(q, axis), axis)
        np.testing.assert_allclose(
            reconstructed, q, atol=1e-9,
            err_msg=f"N={N}, axis={axis_name}",
        )

    def test_single_sample(self):
        q = np.array([[1.0, 2.0, 3.0, 4.0]])
        u = np.array([1.0, 0.0, 0.0])
        reconstructed = iqdft(qdft(q, u), u)
        np.testing.assert_allclose(reconstructed, q, atol=1e-12)


class TestQfftInverse:
    @pytest.mark.parametrize("N", SIZES)
    @pytest.mark.parametrize("axis_name,axis", AXES)
    def test_round_trip(self, N, axis_name, axis):
        rng = np.random.default_rng(hash((N, axis_name, "qfft")) % 2**32)
        q = rng.standard_normal((N, 4))
        reconstructed = iqfft(qfft(q, axis), axis)
        np.testing.assert_allclose(
            reconstructed, q, atol=1e-10,
            err_msg=f"N={N}, axis={axis_name}",
        )

    def test_zero_signal(self):
        q = np.zeros((8, 4))
        u = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(iqfft(qfft(q, u), u), q, atol=1e-12)

    def test_qdft_matches_qfft_inverse(self):
        """iqdft and iqfft should reconstruct the same signal."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((10, 4))
        u = np.array([1.0, 0.0, 0.0])
        Q = qfft(q, u)
        np.testing.assert_allclose(iqdft(Q, u), iqfft(Q, u), atol=1e-10)

    def test_non_power_of_two(self):
        rng = np.random.default_rng(123)
        q = rng.standard_normal((11, 4))
        u = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(iqfft(qfft(q, u), u), q, atol=1e-10)
