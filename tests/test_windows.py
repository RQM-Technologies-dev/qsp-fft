"""Tests for qsp.fft.windows."""

import math

import numpy as np
import pytest

from qsp.fft.windows import hamming_window, hann_window, rectangular_window


# ---------------------------------------------------------------------------
# rectangular_window
# ---------------------------------------------------------------------------

class TestRectangularWindow:
    def test_length(self):
        w = rectangular_window(10)
        assert len(w) == 10

    def test_all_ones(self):
        w = rectangular_window(8)
        np.testing.assert_array_equal(w, np.ones(8))

    def test_length_one(self):
        w = rectangular_window(1)
        assert w[0] == 1.0

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            rectangular_window(0)
        with pytest.raises(ValueError):
            rectangular_window(-3)


# ---------------------------------------------------------------------------
# hann_window
# ---------------------------------------------------------------------------

class TestHannWindow:
    def test_length(self):
        w = hann_window(16)
        assert len(w) == 16

    def test_first_and_last_are_zero(self):
        w = hann_window(9)
        assert w[0] == pytest.approx(0.0, abs=1e-12)
        assert w[-1] == pytest.approx(0.0, abs=1e-12)

    def test_centre_is_one_for_odd_length(self):
        n = 9
        w = hann_window(n)
        assert w[n // 2] == pytest.approx(1.0, rel=1e-6)

    def test_values_between_zero_and_one(self):
        w = hann_window(32)
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0 + 1e-12)

    def test_symmetry(self):
        w = hann_window(11)
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)

    def test_length_one(self):
        w = hann_window(1)
        assert w[0] == 1.0

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            hann_window(0)

    def test_formula_spot_check(self):
        n = 5
        w = hann_window(n)
        expected = 0.5 * (1.0 - math.cos(2 * math.pi * 2 / (n - 1)))
        assert w[2] == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# hamming_window
# ---------------------------------------------------------------------------

class TestHammingWindow:
    def test_length(self):
        w = hamming_window(16)
        assert len(w) == 16

    def test_first_sample_approx_008(self):
        """Standard Hamming first coefficient is 0.08."""
        w = hamming_window(9)
        assert w[0] == pytest.approx(0.08, abs=1e-10)

    def test_values_positive(self):
        w = hamming_window(32)
        assert np.all(w > 0)

    def test_symmetry(self):
        w = hamming_window(11)
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)

    def test_length_one(self):
        w = hamming_window(1)
        assert w[0] == 1.0

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            hamming_window(0)

    def test_formula_spot_check(self):
        n = 5
        w = hamming_window(n)
        expected = 0.54 - 0.46 * math.cos(2 * math.pi * 1 / (n - 1))
        assert w[1] == pytest.approx(expected, rel=1e-10)
