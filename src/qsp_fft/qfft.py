"""qsp_fft.qfft — fast FFT-based implementation of the QDFT.

Implements the same right-sided fixed-axis Quaternionic DFT as
:mod:`qsp_fft.qdft`, but reduces the computation to two ordinary complex
FFTs via the *slice decomposition* of the quaternion algebra.

Slice decomposition
-------------------
Given a fixed unit pure-quaternion axis ``u``, every quaternion ``q`` can
be written as::

    q = a + b·v

where ``a, b`` lie in the complex slice ``C_u = {p + q·u : p,q ∈ ℝ}``
(isomorphic to ℂ), and ``v`` is any fixed unit pure-quaternion orthogonal
to ``u`` (so that ``{u, v, w=u×v}`` is an oriented orthonormal frame for
the imaginary subspace ℝ³).

Under this decomposition the right-sided QDFT becomes::

    Q_u[k] = Σ_n (a[n] + b[n]·v) · exp(-u·2πkn/N)
            = A[k] + B̃[k]·v

where:

* ``A[k] = FFT(z_a)[k]``  — ordinary complex FFT of the C_u component
* ``B̃[k] = conj(FFT(conj(z_b))[k])``  — equivalent to
  ``Σ_n b[n]·exp(+u·2πkn/N)``

and ``z_a[n], z_b[n]`` are complex numbers encoding ``a[n]`` and ``b[n]``
respectively (see source for the coordinate formulas).

The inverse ``iqfft`` reverses the decomposition using:

* ``c₁[n] = IFFT(Z_A)[n]``
* ``c₂[n] = FFT(Z_C2)[n] / N``  — reconstructs the C_u coefficient for
  the *v*-component

Key algebraic identities used
------------------------------
* Elements of ``C_u`` commute with ``exp(±u·θ)``.
* ``v · exp(-u·θ) = exp(+u·θ) · v``  (swap rule, proven from
  anticommutativity ``u·v = -v·u``).
* ``q · v · exp(u·θ) = q · exp(-u·θ) · v``  for ``q ∈ C_u``.
"""

from __future__ import annotations

import numpy as np

from .axis import normalize_axis
from .quaternion import as_quaternion_array

__all__ = ["qfft", "iqfft"]


def _perp_axis(u: np.ndarray) -> np.ndarray:
    """Return a unit 3-vector perpendicular to the unit axis *u*.

    Selects the coordinate axis *e_i* whose dot product with *u* is
    smallest (most perpendicular), then Gram-Schmidt orthogonalises.
    """
    i = int(np.argmin(np.abs(u)))
    e = np.zeros(3)
    e[i] = 1.0
    v = e - np.dot(e, u) * u
    return v / np.linalg.norm(v)


def qfft(signal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Compute the right-sided fixed-axis QDFT via two complex FFTs.

    Numerically identical to :func:`qsp_fft.qdft.qdft` but runs in
    *O(N log N)* by reducing the quaternionic transform to two ordinary
    complex FFTs through the slice decomposition described in the module
    docstring.

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
    >>> from qsp_fft.qfft import qfft
    >>> from qsp_fft.axis import canonical_axes
    >>> axes = canonical_axes()
    >>> q = np.zeros((8, 4)); q[0, 0] = 1.0
    >>> Q = qfft(q, axes["i"])
    >>> np.allclose(Q, np.tile([1., 0., 0., 0.], (8, 1)))
    True
    """
    signal = as_quaternion_array(signal)
    N = len(signal)
    u = normalize_axis(axis)
    v = _perp_axis(u)
    w = np.cross(u, v)  # w = u × v, completes the orthonormal frame

    # Extract scalar and vector parts.
    scalar = signal[:, 0]          # q_w,  shape (N,)
    vec = signal[:, 1:]            # (q_x, q_y, q_z), shape (N, 3)

    # Decompose each quaternion: q = a + b·v, where a, b ∈ C_u.
    # Represent a and b as ordinary complex numbers with u ↔ i:
    #   a[n] = q_w[n]           + i·(vec[n]·u)
    #   b[n] = (vec[n]·v)       + i·(vec[n]·w)
    z_a = scalar + 1j * (vec @ u)   # shape (N,), complex
    z_b = (vec @ v) + 1j * (vec @ w)  # shape (N,), complex

    # Two complex FFTs.
    # A[k]  = Σ_n a[n]·exp(-2πi·kn/N)  (standard forward DFT of z_a)
    Z_A = np.fft.fft(z_a)

    # B̃[k]  = Σ_n b[n]·exp(+2πi·kn/N)  (DFT with *positive* exponent)
    #        = conj(FFT(conj(z_b)))[k]
    Z_B = np.conj(np.fft.fft(np.conj(z_b)))

    # Recombine: Q[k] = A[k] + B̃[k]·v, expanding back into (w,x,y,z).
    # A[k] ∈ C_u:   real part → scalar, imag part → component along u.
    # B̃[k]·v ∈ ℍ:  Re(B̃)·v + Im(B̃)·(u×v) = Re(B̃)·v + Im(B̃)·w.
    result = np.zeros((N, 4))
    result[:, 0] = Z_A.real
    result[:, 1:] = (
        Z_A.imag[:, np.newaxis] * u
        + Z_B.real[:, np.newaxis] * v
        + Z_B.imag[:, np.newaxis] * w
    )
    return result


def iqfft(spectrum: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Compute the inverse right-sided fixed-axis QDFT via two complex FFTs.

    Reconstructs the time-domain quaternion signal from the spectrum
    produced by :func:`qfft`:

    .. math::
        q[n] = \\frac{1}{N} \\sum_{k=0}^{N-1}
               Q_u[k] \\cdot \\exp\\!\\left(+u\\,\\frac{2\\pi kn}{N}\\right)

    The inverse slice decomposition uses:

    * ``c₁[n] = IFFT(Z_A)[n]``
    * ``c₂[n] = FFT(Z_C2)[n] / N``  — the C_u coefficient for the
      ``v``-component, recovered by applying ``FFT`` (not IFFT) to the
      ``C₂`` bins extracted from the spectrum.

    Parameters
    ----------
    spectrum:
        Quaternion spectrum of shape ``(N, 4)`` in ``(w,x,y,z)`` order,
        as returned by :func:`qfft`.
    axis:
        Same analysis axis used in the forward transform.

    Returns
    -------
    numpy.ndarray
        Reconstructed signal of shape ``(N, 4)`` in ``(w,x,y,z)`` order.

    Examples
    --------
    >>> import numpy as np
    >>> from qsp_fft.qfft import qfft, iqfft
    >>> rng = np.random.default_rng(42)
    >>> q = rng.standard_normal((16, 4))
    >>> axis = np.array([0., 1., 0.])
    >>> np.allclose(iqfft(qfft(q, axis), axis), q, atol=1e-10)
    True
    """
    spectrum = as_quaternion_array(spectrum)
    N = len(spectrum)
    u = normalize_axis(axis)
    v = _perp_axis(u)
    w = np.cross(u, v)

    # Extract C₁[k] and C₂[k] from spectrum bins Q[k].
    # Q[k] = C₁[k] + C₂[k]·v, so:
    #   C₁[k] = Q_scalar[k] + i·(Q_vec[k]·u)
    #   C₂[k] = (Q_vec[k]·v) + i·(Q_vec[k]·w)
    Q_scalar = spectrum[:, 0]
    Q_vec = spectrum[:, 1:]

    Z_A = Q_scalar + 1j * (Q_vec @ u)      # C₁[k], complex, shape (N,)
    Z_C2 = (Q_vec @ v) + 1j * (Q_vec @ w)  # C₂[k], complex, shape (N,)

    # Inverse:
    #   c₁[n] = IFFT(Z_A)[n]          = (1/N)·Σ_k C₁[k]·exp(+2πi·kn/N)
    #   c₂[n] = FFT(Z_C2)[n] / N      = (1/N)·Σ_k C₂[k]·exp(-2πi·kn/N)
    c1 = np.fft.ifft(Z_A)
    c2 = np.fft.fft(Z_C2) / N

    # Reconstruct quaternion from (c₁, c₂):
    #   q[n] = c₁[n] + c₂[n]·v
    result = np.zeros((N, 4))
    result[:, 0] = c1.real
    result[:, 1:] = (
        c1.imag[:, np.newaxis] * u
        + c2.real[:, np.newaxis] * v
        + c2.imag[:, np.newaxis] * w
    )
    return result
