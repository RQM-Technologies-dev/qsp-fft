# API Overview — qsp-fft

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

All public names are importable directly from `qsp_fft`:

```python
from qsp_fft import (
    # spectrum
    magnitude_spectrum,
    power_spectrum,
    frequency_bins,
    # windows
    rectangular_window,
    hann_window,
    hamming_window,
    # analysis
    dominant_frequency_index,
    dominant_frequency_value,
    spectral_energy,
    # utils
    next_power_of_two,
    normalise_signal,
)
```

---

## spectrum

### `magnitude_spectrum(signal) → ndarray`

Returns the one-sided `|FFT(signal)|` with shape `(n//2 + 1,)`.

### `power_spectrum(signal) → ndarray`

Returns `|FFT(signal)|²` (one-sided), i.e. `magnitude_spectrum(signal)**2`.

### `frequency_bins(n, sample_rate=1.0) → ndarray`

Returns the one-sided frequency bin centres for an `n`-point FFT.

```python
freqs = frequency_bins(512, sample_rate=1000.0)
# freqs[0] == 0.0,  freqs[-1] == 500.0 Hz (Nyquist)
```

---

## windows

All window functions accept an integer length `n` and return a 1-D array.

### `rectangular_window(n) → ndarray`
All-ones array.  Identity window.

### `hann_window(n) → ndarray`
Raised-cosine taper.  Tapers to zero at both ends.
Formula: `0.5 * (1 - cos(2π k / (n-1)))`

### `hamming_window(n) → ndarray`
Similar to Hann but does not reach zero.
Formula: `0.54 - 0.46 * cos(2π k / (n-1))`

---

## analysis

### `dominant_frequency_index(signal) → int`
Index of the highest-magnitude bin in the one-sided spectrum.

### `dominant_frequency_value(signal, sample_rate=1.0) → float`
Frequency (Hz) of the dominant spectral component.

### `spectral_energy(signal) → float`
Sum of `|FFT(signal)[k]|²` over all non-negative bins.

---

## utils

### `next_power_of_two(n) → int`
Smallest `2**k >= n`.  Useful for zero-padding before FFT.

### `normalise_signal(signal) → ndarray`
Scales the signal so the peak absolute value is `1.0`.

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
