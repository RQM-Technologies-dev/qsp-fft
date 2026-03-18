"""qsp.fft.spectrum — core spectral-transform utilities.

This module contains two families of helpers:

**Classical (real-signal) helpers** — operate on real-valued 1-D NumPy
arrays and return one-sided (non-negative frequency) results:

* :func:`magnitude_spectrum`, :func:`power_spectrum`, :func:`frequency_bins`

**Quaternionic spectrum helpers** — operate on QDFT/QFFT output arrays of
shape ``(N, 4)`` in ``(w,x,y,z)`` order:

* :func:`spectrum_magnitude`, :func:`spectrum_energy`,
  :func:`total_energy`, :func:`dominant_bins`
"""

import numpy as np

__all__ = [
    # classical helpers
    "magnitude_spectrum",
    "power_spectrum",
    "frequency_bins",
    # quaternionic helpers
    "spectrum_magnitude",
    "spectrum_energy",
    "total_energy",
    "dominant_bins",
]


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
    >>> from qsp.fft import magnitude_spectrum
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
    >>> from qsp.fft import power_spectrum
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
    >>> from qsp.fft import frequency_bins
    >>> frequency_bins(4, sample_rate=4.0)
    array([0., 1., 2.])
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    return np.fft.rfftfreq(n, d=1.0 / sample_rate)


# ---------------------------------------------------------------------------
# Quaternionic spectrum helpers
# ---------------------------------------------------------------------------


def spectrum_magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Return the per-bin quaternion norm of a QDFT/QFFT spectrum.

    Parameters
    ----------
    spectrum:
        Quaternion spectrum of shape ``(N, 4)`` in ``(w,x,y,z)`` order,
        as returned by :func:`qsp.fft.qfft.qfft` or
        :func:`qsp.fft.qdft.qdft`.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(N,)`` with the Euclidean norm of each bin.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.spectrum import spectrum_magnitude
    >>> Q = np.zeros((4, 4)); Q[0, 0] = 2.0
    >>> spectrum_magnitude(Q)
    array([2., 0., 0., 0.])
    """
    from qsp.fft.quaternion import as_quaternion_array, quaternion_norm

    spectrum = as_quaternion_array(spectrum)
    return quaternion_norm(spectrum)


def spectrum_energy(spectrum: np.ndarray) -> np.ndarray:
    """Return the per-bin squared quaternion norm ``|Q[k]|²``.

    Parameters
    ----------
    spectrum:
        Quaternion spectrum of shape ``(N, 4)`` in ``(w,x,y,z)`` order.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(N,)`` with ``||Q[k]||²`` for each bin.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.spectrum import spectrum_energy
    >>> Q = np.zeros((4, 4)); Q[1, 1] = 3.0
    >>> spectrum_energy(Q)
    array([0., 9., 0., 0.])
    """
    return spectrum_magnitude(spectrum) ** 2


def total_energy(signal: np.ndarray) -> float:
    """Return the total energy ``Σ_n ||q[n]||²`` of a quaternion signal.

    Parameters
    ----------
    signal:
        Quaternion signal of shape ``(N, 4)`` in ``(w,x,y,z)`` order.

    Returns
    -------
    float
        Sum of squared norms over all samples.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.spectrum import total_energy
    >>> q = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    >>> total_energy(q)
    2.0
    """
    from qsp.fft.quaternion import as_quaternion_array, quaternion_norm

    signal = as_quaternion_array(signal)
    return float(np.sum(quaternion_norm(signal) ** 2))


def dominant_bins(
    spectrum: np.ndarray,
    k: int | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Return indices of the dominant spectral bins.

    Bins are ranked by energy (``|Q[k]|²``).  Selection rules:

    * If *k* is provided (and *threshold* is also provided, *k* takes
      precedence): return the *k* bin indices with the highest energy,
      sorted descending by energy.
    * If only *threshold* is provided: return indices of all bins whose
      energy exceeds *threshold*, sorted ascending by index.
    * If both are ``None``: return all bin indices sorted descending by
      energy.

    Parameters
    ----------
    spectrum:
        Quaternion spectrum of shape ``(N, 4)``.
    k:
        Number of top bins to return.  Takes precedence over *threshold*.
    threshold:
        Energy threshold.  Bins with ``|Q[k]|² > threshold`` are returned.

    Returns
    -------
    numpy.ndarray
        1-D integer array of bin indices.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.spectrum import dominant_bins
    >>> Q = np.zeros((8, 4)); Q[3, 0] = 5.0; Q[7, 0] = 2.0
    >>> dominant_bins(Q, k=1)
    array([3])
    >>> dominant_bins(Q, threshold=1.0)
    array([3, 7])
    """
    from qsp.fft.quaternion import as_quaternion_array

    spectrum = as_quaternion_array(spectrum)
    energies = np.sum(spectrum ** 2, axis=1)

    if k is not None:
        return np.argsort(energies)[::-1][:k]
    if threshold is not None:
        return np.where(energies > threshold)[0]
    # Both None: all bins sorted by descending energy.
    return np.argsort(energies)[::-1]
