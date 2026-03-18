"""qsp.fft — v1 spectral engine for the RQM Technologies QSP ecosystem.

This package provides both the classical (real-signal) spectral helpers
that have always lived here and the new **Quaternionic FFT** stack
introduced in v1.

v1 Quaternionic FFT public API
------------------------------
axis.py
    normalize_axis, is_unit_axis, canonical_axes

qdft.py
    qdft, iqdft   — direct O(N²) reference implementation

qfft.py
    qfft, iqfft   — fast O(N log N) slice-based implementation

spectrum.py (quaternionic helpers)
    spectrum_magnitude, spectrum_energy, total_energy, dominant_bins

validation.py
    reconstruction_error, check_parseval, compare_qdft_qfft

Classical (real-signal) API — preserved for backward compatibility
-----------------------------------------------------------------
spectrum.py
    magnitude_spectrum, power_spectrum, frequency_bins

windows.py
    rectangular_window, hann_window, hamming_window

analysis.py
    dominant_frequency_index, dominant_frequency_value, spectral_energy

utils.py
    next_power_of_two, normalise_signal
"""

# v1 Quaternionic FFT API
from qsp.fft.axis import canonical_axes, is_unit_axis, normalize_axis
from qsp.fft.qdft import iqdft, qdft
from qsp.fft.qfft import iqfft, qfft
from qsp.fft.spectrum import (
    dominant_bins,
    spectrum_energy,
    spectrum_magnitude,
    total_energy,
)
from qsp.fft.validation import check_parseval, compare_qdft_qfft, reconstruction_error

# Classical (real-signal) API — backward-compatible
from qsp.fft.spectrum import frequency_bins, magnitude_spectrum, power_spectrum
from qsp.fft.windows import hamming_window, hann_window, rectangular_window
from qsp.fft.analysis import (
    dominant_frequency_index,
    dominant_frequency_value,
    spectral_energy,
)
from qsp.fft.utils import next_power_of_two, normalise_signal

__all__ = [
    # v1 quaternionic FFT — axis
    "normalize_axis",
    "is_unit_axis",
    "canonical_axes",
    # v1 quaternionic FFT — transforms
    "qdft",
    "iqdft",
    "qfft",
    "iqfft",
    # v1 quaternionic FFT — spectrum helpers
    "spectrum_magnitude",
    "spectrum_energy",
    "total_energy",
    "dominant_bins",
    # v1 quaternionic FFT — validation
    "reconstruction_error",
    "check_parseval",
    "compare_qdft_qfft",
    # classical helpers (backward-compatible)
    "magnitude_spectrum",
    "power_spectrum",
    "frequency_bins",
    "rectangular_window",
    "hann_window",
    "hamming_window",
    "dominant_frequency_index",
    "dominant_frequency_value",
    "spectral_energy",
    "next_power_of_two",
    "normalise_signal",
]
