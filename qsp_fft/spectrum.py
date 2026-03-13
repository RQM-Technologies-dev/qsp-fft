"""qsp_fft.spectrum — core spectral-transform utilities.

All functions operate on real-valued 1-D NumPy arrays and return
one-sided (non-negative frequency) results unless otherwise stated.
"""

import numpy as np

__all__ = ["magnitude_spectrum", "power_spectrum", "frequency_bins"]


def magnitude_spectrum(signal: np.ndarray) -> np.ndarray:
    """Return the one-sided magnitude spectrum of *signal*.

    Computes ``|FFT(signal)|`` and returns only the non-negative
    frequency components (first ``N//2 + 1`` bins).

    Parameters
    ----------
    signal:
        Real-valued 1-D array of time-domain samples.

    Returns
    -------
    numpy.ndarray
        1-D array of magnitude values with length ``len(signal) // 2 + 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft import magnitude_spectrum
    >>> mag = magnitude_spectrum(np.array([1.0, 0.0, -1.0, 0.0]))
    >>> mag.shape
    (3,)
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    return np.abs(spectrum)


def power_spectrum(signal: np.ndarray) -> np.ndarray:
    """Return the one-sided power spectrum of *signal*.

    Computes ``|FFT(signal)|²`` for non-negative frequency bins.

    Parameters
    ----------
    signal:
        Real-valued 1-D array of time-domain samples.

    Returns
    -------
    numpy.ndarray
        1-D array of power values with length ``len(signal) // 2 + 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft import power_spectrum
    >>> pwr = power_spectrum(np.ones(8))
    >>> pwr.shape
    (5,)
    """
    return magnitude_spectrum(signal) ** 2


def frequency_bins(n: int, sample_rate: float = 1.0) -> np.ndarray:
    """Return the one-sided frequency bin centres for an *n*-point FFT.

    Parameters
    ----------
    n:
        Number of time-domain samples (FFT length).
    sample_rate:
        Sampling rate in Hz.  Defaults to ``1.0`` (normalised frequency).

    Returns
    -------
    numpy.ndarray
        1-D array of length ``n // 2 + 1`` containing frequencies in Hz
        (or normalised units when *sample_rate* is 1).

    Examples
    --------
    >>> from qsp_fft import frequency_bins
    >>> frequency_bins(4, sample_rate=4.0)
    array([0., 1., 2.])
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    return np.fft.rfftfreq(n, d=1.0 / sample_rate)
