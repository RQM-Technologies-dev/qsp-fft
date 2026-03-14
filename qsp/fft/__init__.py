"""qsp.fft — spectral-transform layer of the RQM Technologies ecosystem.

Public API
----------
spectrum.py
    magnitude_spectrum, power_spectrum, frequency_bins

windows.py
    rectangular_window, hann_window, hamming_window

analysis.py
    dominant_frequency_index, dominant_frequency_value, spectral_energy

utils.py
    next_power_of_two, normalise_signal
"""

from qsp.fft.spectrum import frequency_bins, magnitude_spectrum, power_spectrum
from qsp.fft.windows import hamming_window, hann_window, rectangular_window
from qsp.fft.analysis import (
    dominant_frequency_index,
    dominant_frequency_value,
    spectral_energy,
)
from qsp.fft.utils import next_power_of_two, normalise_signal

__all__ = [
    # spectrum
    "magnitude_spectrum",
    "power_spectrum",
    "frequency_bins",
    # windows
    "rectangular_window",
    "hann_window",
    "hamming_window",
    # analysis
    "dominant_frequency_index",
    "dominant_frequency_value",
    "spectral_energy",
    # utils
    "next_power_of_two",
    "normalise_signal",
]
