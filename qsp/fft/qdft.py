"""qsp.fft.qdft — direct O(N²) reference implementation of the QDFT.

Implements the right-sided fixed-axis Quaternionic Discrete Fourier
Transform (QDFT) exactly as defined in the v1 specification:

    Forward:
        Q_u[k] = Σ_{n=0}^{N-1}  q[n] · exp(-u · 2πkn/N)

    Inverse:
        q[n]   = (1/N) Σ_{k=0}^{N-1}  Q_u[k] · exp(+u · 2πkn/N)

where ``exp(u·θ) = cos(θ) + u·sin(θ)`` and the exponential sits on the
**right** of each quaternion sample (right-sided convention).

These functions are intentionally the brute-force O(N²) reference
implementations, designed for correctness verification rather than
performance.  Use :mod:`qsp.fft.qfft` for production use.
"""

from __future__ import annotations

import numpy as np

from qsp.fft.axis import normalize_axis
from qsp.fft.quaternion import as_quaternion_array, quaternion_exp_pure, quaternion_multiply

__all__ = ["qdft", "iqdft"]


def qdft(signal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Compute the right-sided fixed-axis QDFT directly (O(N²)).

    For each output bin *k*:

    .. math::
        Q_u[k] = \\sum_{n=0}^{N-1} q[n] \\cdot \\exp\\!\\left(-u\\,
        \\frac{2\\pi k n}{N}\\right)

    where multiplication is the Hamilton product and the exponential
    lies on the **right** of ``q[n]``.

    Parameters
    ----------
    signal:
        Quaternion-valued signal of shape ``(N, 4)`` in ``(w,x,y,z)``
        order.
    axis:
        Pure-quaternion analysis axis as a nonzero 3-vector ``(u1,u2,u3)``.
        Need not be pre-normalised.

    Returns
    -------
    numpy.ndarray
        Quaternion spectrum of shape ``(N, 4)`` in ``(w,x,y,z)`` order.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.qdft import qdft
    >>> q = np.zeros((4, 4)); q[0, 0] = 1.0  # scalar impulse at n=0
    >>> Q = qdft(q, np.array([1., 0., 0.]))
    >>> np.allclose(Q, np.tile([1., 0., 0., 0.], (4, 1)))
    True
    """
    signal = as_quaternion_array(signal)
    N = len(signal)
    axis = normalize_axis(axis)

    n_indices = np.arange(N, dtype=float)
    result = np.zeros((N, 4))

    for k in range(N):
        theta = 2.0 * np.pi * k * n_indices / N      # shape (N,)
        exp_neg = quaternion_exp_pure(axis, -theta)   # shape (N, 4)
        products = quaternion_multiply(signal, exp_neg)  # shape (N, 4)
        result[k] = products.sum(axis=0)

    return result


def iqdft(spectrum: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Compute the inverse right-sided fixed-axis QDFT directly (O(N²)).

    For each time-domain sample *n*:

    .. math::
        q[n] = \\frac{1}{N} \\sum_{k=0}^{N-1} Q_u[k] \\cdot
        \\exp\\!\\left(+u\\,\\frac{2\\pi k n}{N}\\right)

    Parameters
    ----------
    spectrum:
        Quaternion spectrum of shape ``(N, 4)`` in ``(w,x,y,z)`` order,
        as returned by :func:`qdft`.
    axis:
        Same analysis axis used in the forward transform.

    Returns
    -------
    numpy.ndarray
        Reconstructed signal of shape ``(N, 4)`` in ``(w,x,y,z)`` order.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.qdft import qdft, iqdft
    >>> rng = np.random.default_rng(0)
    >>> q = rng.standard_normal((8, 4))
    >>> axis = np.array([0., 1., 0.])
    >>> np.allclose(iqdft(qdft(q, axis), axis), q, atol=1e-10)
    True
    """
    spectrum = as_quaternion_array(spectrum)
    N = len(spectrum)
    axis = normalize_axis(axis)

    k_indices = np.arange(N, dtype=float)
    result = np.zeros((N, 4))

    for n in range(N):
        theta = 2.0 * np.pi * k_indices * n / N     # shape (N,)
        exp_pos = quaternion_exp_pure(axis, theta)   # shape (N, 4)
        products = quaternion_multiply(spectrum, exp_pos)  # shape (N, 4)
        result[n] = products.sum(axis=0) / N

    return result
