"""Tests for the Parseval energy relation and validation helpers."""

import numpy as np
import pytest

from qsp_fft.qfft import qfft
from qsp_fft.qdft import qdft
from qsp_fft.validation import check_parseval, compare_qdft_qfft, reconstruction_error


class TestCheckParseval:
    """Parseval: Σ_n ||q[n]||² = (1/N) · Σ_k ||Q[k]||²."""

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_parseval_qfft_i_axis(self, N):
        rng = np.random.default_rng(N * 10)
        q = rng.standard_normal((N, 4))
        Q = qfft(q, np.array([1.0, 0.0, 0.0]))
        assert check_parseval(q, Q), f"Parseval failed for N={N}"

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_parseval_qfft_j_axis(self, N):
        rng = np.random.default_rng(N * 20)
        q = rng.standard_normal((N, 4))
        Q = qfft(q, np.array([0.0, 1.0, 0.0]))
        assert check_parseval(q, Q)

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_parseval_qdft(self, N):
        rng = np.random.default_rng(N * 30)
        q = rng.standard_normal((N, 4))
        Q = qdft(q, np.array([0.0, 0.0, 1.0]))
        assert check_parseval(q, Q)

    def test_parseval_zero_signal(self):
        q = np.zeros((8, 4))
        Q = qfft(q, np.array([1.0, 0.0, 0.0]))
        assert check_parseval(q, Q)

    def test_parseval_diagonal_axis(self):
        rng = np.random.default_rng(777)
        q = rng.standard_normal((12, 4))
        u = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        Q = qfft(q, u)
        assert check_parseval(q, Q)

    def test_parseval_fails_for_mismatched(self):
        """Should return False if spectrum is completely wrong."""
        rng = np.random.default_rng(88)
        q = rng.standard_normal((8, 4))
        Q_wrong = rng.standard_normal((8, 4)) * 100
        assert not check_parseval(q, Q_wrong, atol=1e-8)


class TestReconstructionError:
    def test_identical_inputs_zero_error(self):
        q = np.eye(4, 4)
        assert reconstruction_error(q, q) == 0.0

    def test_zero_signal_returns_absolute(self):
        q = np.zeros((4, 4))
        r = np.ones((4, 4)) * 0.1
        err = reconstruction_error(q, r)
        assert err > 0

    def test_known_error(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        r = np.array([[2.0, 0.0, 0.0, 0.0]])
        # ||q - r|| / ||q|| = 1 / 1 = 1.0
        assert pytest.approx(reconstruction_error(q, r), abs=1e-12) == 1.0

    def test_round_trip_error_small(self):
        from qsp_fft.qfft import iqfft, qfft
        rng = np.random.default_rng(55)
        q = rng.standard_normal((16, 4))
        u = np.array([1.0, 0.0, 0.0])
        r = iqfft(qfft(q, u), u)
        assert reconstruction_error(q, r) < 1e-10


class TestCompareQdftQfft:
    def test_agree_on_random_signal(self):
        rng = np.random.default_rng(0)
        q = rng.standard_normal((8, 4))
        assert compare_qdft_qfft(q, np.array([1.0, 0.0, 0.0]))

    def test_agree_on_diagonal_axis(self):
        rng = np.random.default_rng(1)
        q = rng.standard_normal((8, 4))
        u = np.array([1.0, 2.0, 3.0])
        assert compare_qdft_qfft(q, u)

    def test_agree_on_zero_signal(self):
        q = np.zeros((8, 4))
        assert compare_qdft_qfft(q, np.array([0.0, 1.0, 0.0]))
