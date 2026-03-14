# API Overview — qsp-fft

`qsp-fft` exposes a flat, stable public API through its top-level package.
All public names are importable directly from `qsp_fft`.  Internal helpers
(e.g. `qsp_fft.windows._check_length`) are not part of the public contract
and may change between releases.

---

## Installation

```bash
pip install qsp-core qsp-fft
```

Local development:

```bash
pip install -e ".[dev]"
```

---

## Top-level imports

```python
from qsp_fft import (
    # spectrum helpers
    magnitude_spectrum,
    power_spectrum,
    frequency_bins,
    # window functions
    rectangular_window,
    hann_window,
    hamming_window,
    # analysis helpers
    dominant_frequency_index,
    dominant_frequency_value,
    spectral_energy,
    # utility helpers
    next_power_of_two,
    normalise_signal,
)
```

All names listed in `qsp_fft.__all__` are public.  Everything else is
considered internal.

---

## Spectrum helpers

These are the core building blocks for frequency-domain analysis.  They wrap
`numpy.fft.rfft` / `numpy.fft.rfftfreq` with a consistent one-sided
convention so downstream callers do not need to handle FFT symmetry manually.

### `magnitude_spectrum(signal) → ndarray`

Returns the one-sided `|FFT(signal)|` with shape `(n//2 + 1,)`.

The result is the raw (un-normalised) magnitude.  Callers that need amplitude
accuracy should account for the FFT length.

### `power_spectrum(signal) → ndarray`

Returns `|FFT(signal)|²` (one-sided), i.e. `magnitude_spectrum(signal)**2`.

Useful for energy comparisons and spectral density estimates.

### `frequency_bins(n, sample_rate=1.0) → ndarray`

Returns the one-sided frequency bin centres for an `n`-point FFT.

```python
freqs = frequency_bins(512, sample_rate=1000.0)
# freqs[0] == 0.0,  freqs[-1] == 500.0 Hz (Nyquist)
```

Setting `sample_rate=1.0` (the default) returns normalised frequencies.

---

## Window functions

Window functions reduce spectral leakage by tapering the signal before the
FFT.  All window functions accept an integer length `n` and return a 1-D array
of shape `(n,)`.

### `rectangular_window(n) → ndarray`

All-ones array.  Identity window — equivalent to applying no windowing at all.

### `hann_window(n) → ndarray`

Raised-cosine taper.  Tapers to zero at both ends.
Formula: `0.5 * (1 - cos(2π k / (n-1)))` for `k = 0, …, n-1`

Good general-purpose choice for reducing sidelobes when amplitude accuracy
is not critical.

### `hamming_window(n) → ndarray`

Similar to Hann but does not reach zero at the endpoints.
Formula: `0.54 - 0.46 * cos(2π k / (n-1))`

Provides a lower maximum sidelobe level than the Hann window.

---

## Analysis helpers

Higher-level helpers built on top of the spectrum helpers.  Intended for
common inspection tasks in downstream pipelines.

### `dominant_frequency_index(signal) → int`

Index of the highest-magnitude bin in the one-sided spectrum.  Useful when
the caller needs to look up additional per-bin metadata.

### `dominant_frequency_value(signal, sample_rate=1.0) → float`

Frequency (Hz) of the dominant spectral component.  Combines
`dominant_frequency_index` with `frequency_bins`.

### `spectral_energy(signal) → float`

Sum of `|FFT(signal)[k]|²` over all non-negative bins.  A scalar measure of
total spectral energy; downstream systems can use this for signal-strength
comparisons.

---

## Utility helpers

Small, reusable helpers for spectral workflows.

### `next_power_of_two(n) → int`

Smallest `2**k >= n`.  Useful for zero-padding a signal before FFT to achieve
a fast, power-of-two transform length.

### `normalise_signal(signal) → ndarray`

Scales the signal so the peak absolute value is `1.0`.  If the signal is
entirely zero the original array is returned unchanged.

---

## Downstream reuse guidance

These helpers are designed to be composed freely by downstream systems:

- **Spectrum inspection:** apply a window → compute `magnitude_spectrum` →
  use `frequency_bins` to label the x-axis.
- **Energy monitoring:** call `spectral_energy` periodically to track signal
  power over time.
- **Dominant-frequency tracking:** call `dominant_frequency_value` on
  successive frames to observe frequency drift.
- **Pre-processing:** use `normalise_signal` and `next_power_of_two` to
  prepare a signal before passing it to `magnitude_spectrum`.

For integration patterns with `qsp-filter` and `qsp-modulation`, see
[downstream-usage.md](downstream-usage.md).

---

## Quick example

```python
import numpy as np
from qsp_fft import hann_window, magnitude_spectrum, frequency_bins, dominant_frequency_value

sample_rate = 1000.0
n = 512
t = np.arange(n) / sample_rate
signal = np.sin(2 * np.pi * 50 * t)

mag = magnitude_spectrum(signal * hann_window(n))
freqs = frequency_bins(n, sample_rate=sample_rate)

print(f"Dominant: {dominant_frequency_value(signal * hann_window(n), sample_rate=sample_rate):.1f} Hz")
```
