"""Tests for qsp_fft.analysis."""

import numpy as np
import pytest

from qsp_fft.analysis import (
    dominant_frequency_index,
    dominant_frequency_value,
    spectral_energy,
)


def _sinusoid(freq_bin: int, n: int = 256) -> np.ndarray:
    """Return a pure sinusoid with *freq_bin* cycles in *n* samples."""
    t = np.arange(n)
    return np.sin(2 * np.pi * freq_bin * t / n)


# ---------------------------------------------------------------------------
# dominant_frequency_index
# ---------------------------------------------------------------------------

class TestDominantFrequencyIndex:
    def test_pure_sinusoid(self):
        signal = _sinusoid(freq_bin=8)
        assert dominant_frequency_index(signal) == 8

    def test_dc_signal(self):
        signal = np.ones(64)
        assert dominant_frequency_index(signal) == 0

    def test_higher_frequency(self):
        signal = _sinusoid(freq_bin=32)
        assert dominant_frequency_index(signal) == 32

    def test_returns_int(self):
        signal = _sinusoid(freq_bin=5)
        idx = dominant_frequency_index(signal)
        assert isinstance(idx, int)


# ---------------------------------------------------------------------------
# dominant_frequency_value
# ---------------------------------------------------------------------------

class TestDominantFrequencyValue:
    def test_known_frequency(self):
        """Signal with 10 Hz tone sampled at 256 Hz over 256 samples."""
        n = 256
        sample_rate = 256.0
        t = np.arange(n) / sample_rate
        signal = np.sin(2 * np.pi * 10.0 * t)
        freq = dominant_frequency_value(signal, sample_rate=sample_rate)
        assert freq == pytest.approx(10.0, abs=0.5)

    def test_normalised_frequency(self):
        """Default sample_rate=1 returns normalised frequency."""
        n = 64
        freq_bin = 4
        signal = _sinusoid(freq_bin=freq_bin, n=n)
        freq = dominant_frequency_value(signal)
        expected = freq_bin / n  # normalised
        assert freq == pytest.approx(expected, rel=1e-3)

    def test_returns_float(self):
        signal = _sinusoid(freq_bin=3)
        val = dominant_frequency_value(signal)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# spectral_energy
# ---------------------------------------------------------------------------

class TestSpectralEnergy:
    def test_zero_signal_is_zero(self):
        assert spectral_energy(np.zeros(32)) == 0.0

    def test_positive_for_nonzero_signal(self):
        signal = _sinusoid(freq_bin=5)
        energy = spectral_energy(signal)
        assert energy > 0.0

    def test_returns_float(self):
        energy = spectral_energy(np.ones(16))
        assert isinstance(energy, float)

    def test_increases_with_amplitude(self):
        signal = _sinusoid(freq_bin=5, n=128)
        e1 = spectral_energy(signal)
        e2 = spectral_energy(2.0 * signal)
        assert e2 == pytest.approx(4.0 * e1, rel=1e-9)
