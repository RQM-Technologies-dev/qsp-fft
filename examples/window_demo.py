"""examples/window_demo.py — compare window functions with qsp-fft.

Run:
    python examples/window_demo.py
"""

import numpy as np

from qsp.fft import hamming_window, hann_window, rectangular_window
from qsp.fft.utils import normalise_signal


def _stats(name: str, w: np.ndarray) -> None:
    print(f"  {name:<20}  len={len(w)}  min={w.min():.4f}  max={w.max():.4f}"
          f"  sum={w.sum():.4f}")


def main() -> None:
    n = 64

    windows = {
        "rectangular": rectangular_window(n),
        "hann": hann_window(n),
        "hamming": hamming_window(n),
    }

    print("=== Window Demo ===")
    print(f"Window length: {n}\n")
    print("Statistics:")
    for name, w in windows.items():
        _stats(name, w)

    # -----------------------------------------------------------------------
    # Apply each window to a noisy sinusoid and compare spectral leakage
    # -----------------------------------------------------------------------
    from qsp.fft import magnitude_spectrum, frequency_bins

    sample_rate = float(n)
    t = np.arange(n) / sample_rate
    signal = np.sin(2 * np.pi * 5 * t)   # 5 Hz tone in a 64-sample frame

    freqs = frequency_bins(n, sample_rate=sample_rate)

    print("\nPeak magnitude at 5 Hz after each window:")
    for name, w in windows.items():
        mag = magnitude_spectrum(signal * w)
        peak = float(mag[5])   # bin 5 == 5 Hz for our setup
        print(f"  {name:<20}  {peak:.4f}")

    # -----------------------------------------------------------------------
    # Demonstrate normalise_signal helper
    # -----------------------------------------------------------------------
    noisy = np.array([0.0, 3.0, -6.0, 1.5])
    normed = normalise_signal(noisy)
    print("\nnormalise_signal([0, 3, -6, 1.5]):", normed)


if __name__ == "__main__":
    main()
