"""Tests for qsp_fft.spectrum."""

import numpy as np
import pytest

from qsp_fft.spectrum import frequency_bins, magnitude_spectrum, power_spectrum


# ---------------------------------------------------------------------------
# magnitude_spectrum
# ---------------------------------------------------------------------------

class TestMagnitudeSpectrum:
    def test_output_length(self):
        n = 16
        signal = np.zeros(n)
        mag = magnitude_spectrum(signal)
        assert len(mag) == n // 2 + 1

    def test_zero_signal_is_zero(self):
        mag = magnitude_spectrum(np.zeros(32))
        np.testing.assert_array_equal(mag, 0.0)

    def test_dc_component(self):
        """A constant signal should place all energy in bin 0 (DC)."""
        n = 64
        signal = np.ones(n)
        mag = magnitude_spectrum(signal)
        assert mag[0] == pytest.approx(n, rel=1e-6)
        assert np.all(mag[1:] < 1e-10)

    def test_sinusoid_peak(self):
        """A pure sinusoid should produce a single dominant peak."""
        n = 256
        freq_bin = 10
        t = np.arange(n)
        signal = np.sin(2 * np.pi * freq_bin * t / n)
        mag = magnitude_spectrum(signal)
        peak = int(np.argmax(mag))
        assert peak == freq_bin

    def test_output_is_non_negative(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(128)
        mag = magnitude_spectrum(signal)
        assert np.all(mag >= 0)


# ---------------------------------------------------------------------------
# power_spectrum
# ---------------------------------------------------------------------------

class TestPowerSpectrum:
    def test_equals_magnitude_squared(self):
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(64)
        pwr = power_spectrum(signal)
        expected = magnitude_spectrum(signal) ** 2
        np.testing.assert_allclose(pwr, expected)

    def test_output_length(self):
        n = 32
        pwr = power_spectrum(np.ones(n))
        assert len(pwr) == n // 2 + 1

    def test_zero_signal_is_zero(self):
        pwr = power_spectrum(np.zeros(16))
        np.testing.assert_array_equal(pwr, 0.0)


# ---------------------------------------------------------------------------
# frequency_bins
# ---------------------------------------------------------------------------

class TestFrequencyBins:
    def test_output_length(self):
        freqs = frequency_bins(32)
        assert len(freqs) == 32 // 2 + 1

    def test_first_bin_is_zero(self):
        freqs = frequency_bins(64)
        assert freqs[0] == 0.0

    def test_last_bin_is_nyquist(self):
        n = 64
        sr = 1000.0
        freqs = frequency_bins(n, sample_rate=sr)
        assert freqs[-1] == pytest.approx(sr / 2)

    def test_known_values(self):
        freqs = frequency_bins(4, sample_rate=4.0)
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(freqs, expected)

    def test_default_sample_rate(self):
        freqs = frequency_bins(4)
        assert freqs[0] == 0.0
        assert freqs[-1] == pytest.approx(0.5)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            frequency_bins(0)

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError):
            frequency_bins(8, sample_rate=-1.0)
