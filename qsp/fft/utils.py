"""qsp.fft.utils — shared low-level helpers for the spectral layer.

Only utilities that clearly belong in the spectral domain live here.
Quaternion helpers belong in qsp-core.
"""

import math

import numpy as np

__all__ = ["next_power_of_two", "normalise_signal"]


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two that is greater than or equal to *n*.

    Parameters
    ----------
    n:
        A positive integer.

    Returns
    -------
    int
        Smallest ``2**k`` such that ``2**k >= n``.

    Raises
    ------
    ValueError
        If *n* is not a positive integer.

    Examples
    --------
    >>> from qsp.fft.utils import next_power_of_two
    >>> next_power_of_two(1)
    1
    >>> next_power_of_two(5)
    8
    >>> next_power_of_two(8)
    8
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return 1 << (n - 1).bit_length()


def normalise_signal(signal: np.ndarray) -> np.ndarray:
    """Return *signal* scaled to the range ``[-1, 1]``.

    If *signal* is entirely zero the original array is returned unchanged.

    Parameters
    ----------
    signal:
        Real-valued 1-D NumPy array.

    Returns
    -------
    numpy.ndarray
        Array with the same shape as *signal* where the maximum absolute
        value is ``1.0`` (or the array is all zeros).

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.utils import normalise_signal
    >>> normalise_signal(np.array([0.0, 2.0, -4.0, 2.0]))
    array([ 0.  ,  0.5 , -1.  ,  0.5 ])
    """
    signal = np.asarray(signal, dtype=float)
    peak = np.max(np.abs(signal))
    if peak == 0.0:
        return signal.copy()
    return signal / peak
