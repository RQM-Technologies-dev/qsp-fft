"""qsp_fft.axis — axis utilities for quaternionic spectral analysis.

Provides helpers for validating, normalising, and constructing the fixed
analysis axis used by the QDFT/QFFT transforms.
"""

from __future__ import annotations

import numpy as np

__all__ = ["normalize_axis", "is_unit_axis", "canonical_axes"]


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    """Return the unit-length version of *axis*.

    Parameters
    ----------
    axis:
        A nonzero 3-vector specifying the pure-quaternion analysis axis.
        Shape must be ``(3,)``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(3,)`` with unit norm.

    Raises
    ------
    ValueError
        If *axis* does not have shape ``(3,)`` or is the zero vector.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.axis import normalize_axis
    >>> normalize_axis(np.array([2.0, 0.0, 0.0]))
    array([1., 0., 0.])
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,):
        raise ValueError(
            f"axis must have shape (3,), got {axis.shape!r}"
        )
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("axis must be nonzero; cannot normalise the zero vector")
    return axis / norm


def is_unit_axis(axis: np.ndarray, atol: float = 1e-8) -> bool:
    """Return ``True`` if *axis* is already a unit 3-vector.

    Parameters
    ----------
    axis:
        Candidate axis vector.  Must have shape ``(3,)``; any other
        shape returns ``False`` without raising.
    atol:
        Absolute tolerance on ``|norm(axis) - 1|``.

    Returns
    -------
    bool

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.axis import is_unit_axis
    >>> is_unit_axis(np.array([1.0, 0.0, 0.0]))
    True
    >>> is_unit_axis(np.array([1.0, 1.0, 0.0]))
    False
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,):
        return False
    return bool(abs(float(np.linalg.norm(axis)) - 1.0) <= atol)


def canonical_axes() -> dict[str, np.ndarray]:
    """Return the three canonical unit pure-quaternion axes.

    Returns
    -------
    dict[str, numpy.ndarray]
        A mapping with keys ``"i"``, ``"j"``, ``"k"`` and values that
        are unit 3-vectors (the imaginary parts of the standard quaternion
        basis elements).

    Examples
    --------
    >>> from qsp_fft.axis import canonical_axes
    >>> axes = canonical_axes()
    >>> axes["i"]
    array([1., 0., 0.])
    >>> axes["k"]
    array([0., 0., 1.])
    """
    return {
        "i": np.array([1.0, 0.0, 0.0]),
        "j": np.array([0.0, 1.0, 0.0]),
        "k": np.array([0.0, 0.0, 1.0]),
    }
