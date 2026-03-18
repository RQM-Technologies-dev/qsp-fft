"""axis_comparison.py — same signal transformed under i, j, k axes.

Shows how the QFFT spectrum magnitude varies when the same quaternion
signal is analysed along each of the three canonical axes.

Usage::

    python examples/axis_comparison.py
"""

import numpy as np

from qsp.fft import canonical_axes, qfft, spectrum_magnitude


def main() -> None:
    rng = np.random.default_rng(7)

    N = 16
    q = rng.standard_normal((N, 4))
    axes = canonical_axes()

    print(f"Signal length : N = {N}")
    print()

    results = {}
    for name, u in axes.items():
        Q = qfft(q, u)
        mag = spectrum_magnitude(Q)
        results[name] = mag
        total = float(np.sum(mag))
        dominant = int(np.argmax(mag))
        print(
            f"Axis {name!r}: total magnitude = {total:.4f}, "
            f"dominant bin = {dominant}  (|Q[{dominant}]| = {mag[dominant]:.4f})"
        )

    print()
    # Per-bin comparison across axes
    print("Per-bin magnitude comparison (i | j | k):")
    print(f"{'bin':>4}  {'i':>10}  {'j':>10}  {'k':>10}")
    for k in range(N):
        vals = [results[name][k] for name in ("i", "j", "k")]
        print(f"{k:4d}  {vals[0]:10.4f}  {vals[1]:10.4f}  {vals[2]:10.4f}")


if __name__ == "__main__":
    main()
