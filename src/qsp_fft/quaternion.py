"""qsp_fft.quaternion — quaternion helpers for the QDFT/QFFT stack.

All quaternions are represented as ``(w, x, y, z)`` float arrays in
``(N, 4)`` shape, corresponding to ``q = w + x·i + y·j + z·k``.

These helpers are purposely narrow in scope: they implement only what is
needed to support the fixed-axis QDFT/QFFT transforms.  General quaternion
algebra lives in ``qsp-core``; these functions are spectral-computation
utilities.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "as_quaternion_array",
    "quaternion_norm",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_exp_pure",
]


def as_quaternion_array(x: np.ndarray) -> np.ndarray:
    """Coerce *x* to a ``(N, 4)`` float array in ``(w, x, y, z)`` order.

    A single quaternion given as a 1-D array of length 4 is promoted to
    shape ``(1, 4)``.

    Parameters
    ----------
    x:
        Input array.  Accepted shapes: ``(4,)`` or ``(N, 4)``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(N, 4)``.

    Raises
    ------
    ValueError
        If *x* cannot be interpreted as a quaternion array.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.quaternion import as_quaternion_array
    >>> as_quaternion_array(np.array([1.0, 0.0, 0.0, 0.0])).shape
    (1, 4)
    >>> as_quaternion_array(np.zeros((5, 4))).shape
    (5, 4)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if x.shape[0] == 4:
            return x[np.newaxis, :]
        raise ValueError(
            f"1-D input must have length 4 (one quaternion), got {x.shape[0]}"
        )
    if x.ndim == 2 and x.shape[1] == 4:
        return x
    raise ValueError(
        f"Cannot interpret shape {x.shape!r} as a quaternion array; "
        "expected (4,) or (N, 4)"
    )


def quaternion_norm(q: np.ndarray) -> np.ndarray:
    """Return the sample-wise Euclidean norm of quaternion array *q*.

    Parameters
    ----------
    q:
        Quaternion array of shape ``(N, 4)`` or ``(4,)``.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(N,)`` with per-sample norms.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.quaternion import quaternion_norm
    >>> quaternion_norm(np.array([[1.0, 0.0, 0.0, 0.0]]))
    array([1.])
    >>> quaternion_norm(np.array([[0.0, 3.0, 4.0, 0.0]]))
    array([5.])
    """
    q = as_quaternion_array(q)
    return np.linalg.norm(q, axis=1)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of quaternion array *q*.

    For ``q = w + x·i + y·j + z·k`` the conjugate is
    ``q* = w - x·i - y·j - z·k``.

    Parameters
    ----------
    q:
        Quaternion array of shape ``(N, 4)`` or ``(4,)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 4)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.quaternion import quaternion_conjugate
    >>> quaternion_conjugate(np.array([[1.0, 2.0, 3.0, 4.0]]))
    array([[ 1., -2., -3., -4.]])
    """
    q = as_quaternion_array(q)
    result = q.copy()
    result[:, 1:] = -result[:, 1:]
    return result


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of *q1* and *q2* (row-wise).

    Computes ``q1 * q2`` using the standard Hamilton product rule.
    Inputs are paired row-by-row; shapes must be broadcast-compatible.

    Parameters
    ----------
    q1:
        First quaternion array, shape ``(N, 4)`` or ``(4,)``.
    q2:
        Second quaternion array, shape ``(N, 4)`` or ``(4,)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 4)`` containing the Hamilton product for
        each corresponding pair of rows.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.quaternion import quaternion_multiply
    >>> i = np.array([[0., 1., 0., 0.]])
    >>> j = np.array([[0., 0., 1., 0.]])
    >>> quaternion_multiply(i, j)  # i*j = k
    array([[0., 0., 0., 1.]])
    """
    q1 = as_quaternion_array(q1)
    q2 = as_quaternion_array(q2)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.column_stack([w, x, y, z])


def quaternion_exp_pure(
    axis: np.ndarray,
    theta: float | np.ndarray,
) -> np.ndarray:
    """Return ``exp(u·θ) = cos(θ) + u·sin(θ)`` for a pure unit axis *u*.

    Parameters
    ----------
    axis:
        Unit pure-quaternion axis as a 3-vector ``(u1, u2, u3)``.
    theta:
        Angle(s) in radians.  May be a scalar or a 1-D array.

    Returns
    -------
    numpy.ndarray
        If *theta* is a scalar: shape ``(4,)`` quaternion in ``(w,x,y,z)``
        form.
        If *theta* is a 1-D array of length *M*: shape ``(M, 4)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.quaternion import quaternion_exp_pure
    >>> import math
    >>> q = quaternion_exp_pure(np.array([1.,0.,0.]), math.pi / 2)
    >>> q.round(10)
    array([0., 1., 0., 0.])
    """
    axis = np.asarray(axis, dtype=float)
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    theta = np.atleast_1d(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    w = cos_t
    x = sin_t * axis[0]
    y = sin_t * axis[1]
    z = sin_t * axis[2]
    result = np.column_stack([w, x, y, z])
    if scalar_input:
        return result[0]  # shape (4,)
    return result         # shape (M, 4)
