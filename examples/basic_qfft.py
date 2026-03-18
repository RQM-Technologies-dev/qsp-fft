"""basic_qfft.py — minimal demonstration of the v1 QFFT API.

Creates a small quaternionic signal, computes the QFFT with a canonical
axis, prints per-bin spectrum magnitudes, inverts, and reports the
reconstruction error.

Usage::

    python examples/basic_qfft.py
"""

import numpy as np

from qsp.fft import (
    canonical_axes,
    iqfft,
    qfft,
    reconstruction_error,
    spectrum_magnitude,
)


def main() -> None:
    rng = np.random.default_rng(0)

    N = 16
    q = rng.standard_normal((N, 4))

    axes = canonical_axes()
    u = axes["i"]

    print(f"Signal shape : {q.shape}")
    print(f"Analysis axis: {u}  (i-axis)")
    print()

    Q = qfft(q, u)
    mag = spectrum_magnitude(Q)

    print("Per-bin spectrum magnitudes:")
    for k, m in enumerate(mag):
        print(f"  bin {k:2d}: {m:.6f}")
    print()

    q_rec = iqfft(Q, u)
    err = reconstruction_error(q, q_rec)
    print(f"Reconstruction relative L2 error: {err:.2e}")


if __name__ == "__main__":
    main()
