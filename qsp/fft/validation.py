"""qsp.fft.validation — correctness and consistency helpers for QDFT/QFFT.

These utilities allow callers to verify that a QDFT/QFFT computation is
self-consistent (inversion, Parseval energy balance, and agreement between
the direct and fast implementations).
"""

from __future__ import annotations

import numpy as np

from qsp.fft.quaternion import as_quaternion_array

__all__ = ["reconstruction_error", "check_parseval", "compare_qdft_qfft"]


def reconstruction_error(
    signal: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Return the relative L2 reconstruction error.

    Computes::

        ||signal - reconstructed||_F / ||signal||_F

    where ``||·||_F`` is the Frobenius norm over all samples and
    quaternion components.  If *signal* is identically zero the
    numerator (absolute error) is returned instead.

    Parameters
    ----------
    signal:
        Original quaternion signal, shape ``(N, 4)``.
    reconstructed:
        Reconstructed signal (e.g. result of ``iqfft(qfft(signal))``),
        shape ``(N, 4)``.

    Returns
    -------
    float
        Relative L2 error in ``[0, ∞)``.  A value below ``1e-10`` is
        considered numerically perfect for double-precision inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.validation import reconstruction_error
    >>> q = np.eye(4, 4)
    >>> reconstruction_error(q, q)
    0.0
    """
    signal = as_quaternion_array(signal)
    reconstructed = as_quaternion_array(reconstructed)
    num = float(np.linalg.norm(signal - reconstructed))
    den = float(np.linalg.norm(signal))
    if den == 0.0:
        return num
    return num / den


def check_parseval(
    signal: np.ndarray,
    spectrum: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    """Return ``True`` if the Parseval energy relation holds.

    For the right-sided QDFT with the normalisation used in this library
    (no ``1/N`` on the forward transform, ``1/N`` on the inverse), the
    Parseval relation is::

        Σ_n ||q[n]||² = (1/N) · Σ_k ||Q[k]||²

    where ``||·||`` denotes the quaternion norm.

    Parameters
    ----------
    signal:
        Time-domain quaternion signal, shape ``(N, 4)``.
    spectrum:
        Corresponding QDFT/QFFT spectrum, shape ``(N, 4)``.
    atol:
        Absolute tolerance for the comparison.

    Returns
    -------
    bool

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.qfft import qfft
    >>> from qsp.fft.validation import check_parseval
    >>> rng = np.random.default_rng(0)
    >>> q = rng.standard_normal((16, 4))
    >>> Q = qfft(q, np.array([1., 0., 0.]))
    >>> check_parseval(q, Q)
    True
    """
    signal = as_quaternion_array(signal)
    spectrum = as_quaternion_array(spectrum)
    N = len(signal)
    signal_energy = float(np.sum(signal ** 2))
    spectrum_energy = float(np.sum(spectrum ** 2)) / N
    return abs(signal_energy - spectrum_energy) <= atol


def compare_qdft_qfft(
    signal: np.ndarray,
    axis: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    """Return ``True`` if :func:`qdft` and :func:`qfft` agree on *signal*.

    Runs both the direct O(N²) reference implementation and the fast
    O(N log N) FFT-based implementation and checks that every quaternion
    component of every output bin agrees within *atol*.

    Parameters
    ----------
    signal:
        Quaternion signal, shape ``(N, 4)``.
    axis:
        Analysis axis, shape ``(3,)``.
    atol:
        Absolute tolerance (component-wise) for the comparison.

    Returns
    -------
    bool

    Examples
    --------
    >>> import numpy as np
    >>> from qsp.fft.validation import compare_qdft_qfft
    >>> rng = np.random.default_rng(7)
    >>> q = rng.standard_normal((8, 4))
    >>> compare_qdft_qfft(q, np.array([1., 0., 0.]))
    True
    """
    from qsp.fft.qdft import qdft
    from qsp.fft.qfft import qfft

    Q_direct = qdft(signal, axis)
    Q_fast = qfft(signal, axis)
    return bool(np.max(np.abs(Q_direct - Q_fast)) <= atol)
