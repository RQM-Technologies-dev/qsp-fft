"""qsp.fft.analysis — spectral analysis helpers.

Operates on magnitude or power spectra produced by :mod:`qsp.fft.spectrum`.
"""

import numpy as np

from qsp.fft.spectrum import frequency_bins, magnitude_spectrum

__all__ = [
    "dominant_frequency_index",
    "dominant_frequency_value",
    "spectral_energy",
]


def dominant_frequency_index(signal: np.ndarray) -> int:
    """Return the index of the dominant (highest-magnitude) frequency bin.

    Parameters
    ----------
    signal:
        Real-valued 1-D time-domain signal.

    Returns
    -------
    int
        Index into the one-sided magnitude spectrum where the magnitude
        is greatest.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft import dominant_frequency_index
    >>> t = np.arange(64) / 64.0
    >>> signal = np.sin(2 * np.pi * 4 * t)   # 4 cycles in 64 samples
    >>> dominant_frequency_index(signal)
    4
    """
    mag = magnitude_spectrum(np.asarray(signal, dtype=float))
    return int(np.argmax(mag))


def dominant_frequency_value(
    signal: np.ndarray,
    sample_rate: float = 1.0,
) -> float:
    """Return the frequency (in Hz) of the dominant spectral component.

    Parameters
    ----------
    signal:
        Real-valued 1-D time-domain signal.
    sample_rate:
        Sampling rate in Hz.  Defaults to ``1.0`` (normalised frequency).

    Returns
    -------
    float
        Frequency of the dominant bin in Hz.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft import dominant_frequency_value
    >>> t = np.arange(256) / 256.0
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> dominant_frequency_value(signal, sample_rate=256.0)
    10.0
    """
    signal = np.asarray(signal, dtype=float)
    idx = dominant_frequency_index(signal)
    freqs = frequency_bins(len(signal), sample_rate)
    return float(freqs[idx])


def spectral_energy(signal: np.ndarray) -> float:
    """Return the total spectral energy of *signal*.

    Computed as the sum of squared magnitude values across all one-sided
    frequency bins (Parseval-equivalent for the one-sided spectrum).

    Parameters
    ----------
    signal:
        Real-valued 1-D time-domain signal.

    Returns
    -------
    float
        Sum of ``|FFT(signal)[k]|²`` over non-negative frequency bins.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft import spectral_energy
    >>> spectral_energy(np.zeros(8))
    0.0
    """
    mag = magnitude_spectrum(np.asarray(signal, dtype=float))
    return float(np.sum(mag ** 2))
