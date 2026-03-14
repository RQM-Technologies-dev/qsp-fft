"""qsp.fft.windows — reusable window functions for spectral analysis.

Each function returns a 1-D NumPy array of the requested length.
"""

import math

import numpy as np

__all__ = ["rectangular_window", "hann_window", "hamming_window"]


def _check_length(n: int) -> None:
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Window length must be a positive integer, got {n!r}")


def rectangular_window(n: int) -> np.ndarray:
    """Return a rectangular (boxcar) window of length *n*.

    All coefficients are ``1.0``.  This is the identity window—equivalent
    to applying no windowing at all.

    Parameters
    ----------
    n:
        Number of samples.

    Returns
    -------
    numpy.ndarray
        1-D array of ones with shape ``(n,)``.

    Examples
    --------
    >>> from qsp.fft import rectangular_window
    >>> rectangular_window(4)
    array([1., 1., 1., 1.])
    """
    _check_length(n)
    return np.ones(n)


def hann_window(n: int) -> np.ndarray:
    """Return a Hann (raised-cosine) window of length *n*.

    The Hann window reduces spectral leakage by tapering the signal
    to zero at both ends.

    Parameters
    ----------
    n:
        Number of samples.

    Returns
    -------
    numpy.ndarray
        1-D array with shape ``(n,)`` where coefficients follow
        ``0.5 * (1 - cos(2π k / (n-1)))`` for ``k = 0, …, n-1``.

    Examples
    --------
    >>> from qsp.fft import hann_window
    >>> w = hann_window(5)
    >>> round(float(w[0]), 6), round(float(w[2]), 6)
    (0.0, 1.0)
    """
    _check_length(n)
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    return 0.5 * (1.0 - np.cos(2.0 * math.pi * k / (n - 1)))


def hamming_window(n: int) -> np.ndarray:
    """Return a Hamming window of length *n*.

    The Hamming window is similar to the Hann window but does not taper
    fully to zero, which reduces the maximum side-lobe level.

    Parameters
    ----------
    n:
        Number of samples.

    Returns
    -------
    numpy.ndarray
        1-D array with shape ``(n,)`` where coefficients follow
        ``0.54 - 0.46 * cos(2π k / (n-1))`` for ``k = 0, …, n-1``.

    Examples
    --------
    >>> from qsp.fft import hamming_window
    >>> w = hamming_window(5)
    >>> round(float(w[0]), 6)
    0.08
    """
    _check_length(n)
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    return 0.54 - 0.46 * np.cos(2.0 * math.pi * k / (n - 1))
