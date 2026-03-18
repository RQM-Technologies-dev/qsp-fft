"""spectral_peaks.py — synthesise a slice sinusoid and show dominant bins.

Constructs a quaternion signal by embedding a complex sinusoid of known
frequency into the C_i complex slice, runs qfft, and demonstrates that
dominant_bins correctly identifies the active bins.

Usage::

    python examples/spectral_peaks.py
"""

import numpy as np

from qsp_fft import canonical_axes, dominant_bins, qfft, spectrum_magnitude


def main() -> None:
    N = 32
    target_bin = 5          # integer frequency bin
    axes = canonical_axes()
    u = axes["i"]           # analysis axis = i

    # Build a sinusoid in C_i: q[n] = cos(2π·f·n/N) + i·sin(2π·f·n/N)
    # This lies entirely in the C_i slice (qy = qz = 0).
    n = np.arange(N)
    theta = 2.0 * np.pi * target_bin * n / N
    q = np.zeros((N, 4))
    q[:, 0] = np.cos(theta)   # scalar (w) part
    q[:, 1] = np.sin(theta)   # i-component

    print(f"Signal length    : N = {N}")
    print(f"Embedded bin     : f = {target_bin}")
    print(f"Analysis axis    : i")
    print()

    Q = qfft(q, u)
    mag = spectrum_magnitude(Q)

    # Top-3 bins by energy
    top3 = dominant_bins(Q, k=3)
    print("Top-3 dominant bins (by energy):")
    for idx in top3:
        print(f"  bin {int(idx):3d}: |Q| = {mag[idx]:.4f}")

    print()
    print("All bins with magnitude > 0.1:")
    above = dominant_bins(Q, threshold=0.01)  # energy threshold
    for idx in sorted(above.tolist()):
        print(f"  bin {int(idx):3d}: |Q| = {mag[idx]:.4f}")


if __name__ == "__main__":
    main()
