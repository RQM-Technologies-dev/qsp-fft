"""examples/spectrum_demo.py — basic spectrum analysis with qsp-fft.

Run:
    python examples/spectrum_demo.py
"""

import numpy as np

from qsp_fft import (
    dominant_frequency_value,
    frequency_bins,
    hann_window,
    magnitude_spectrum,
    spectral_energy,
)


def main() -> None:
    # -----------------------------------------------------------------------
    # Build a synthetic signal: two tones at 50 Hz and 120 Hz
    # -----------------------------------------------------------------------
    sample_rate = 1000.0   # Hz
    duration = 1.0         # seconds
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate

    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    # -----------------------------------------------------------------------
    # Apply a Hann window to reduce spectral leakage
    # -----------------------------------------------------------------------
    window = hann_window(n)
    windowed = signal * window

    # -----------------------------------------------------------------------
    # Compute spectrum
    # -----------------------------------------------------------------------
    mag = magnitude_spectrum(windowed)
    freqs = frequency_bins(n, sample_rate=sample_rate)

    # -----------------------------------------------------------------------
    # Report results
    # -----------------------------------------------------------------------
    peak_idx = int(np.argmax(mag))
    dominant = dominant_frequency_value(windowed, sample_rate=sample_rate)
    energy = spectral_energy(windowed)

    print("=== Spectrum Demo ===")
    print(f"Signal length    : {n} samples  ({duration:.1f} s @ {sample_rate:.0f} Hz)")
    print(f"Spectral bins    : {len(freqs)}")
    print(f"Peak bin index   : {peak_idx}")
    print(f"Dominant freq    : {dominant:.1f} Hz")
    print(f"Spectral energy  : {energy:.2f}")

    # Show top-5 peaks
    top5_idx = np.argsort(mag)[-5:][::-1]
    print("\nTop-5 frequency components:")
    for i in top5_idx:
        print(f"  {freqs[i]:7.1f} Hz  → magnitude {mag[i]:.2f}")


if __name__ == "__main__":
    main()
