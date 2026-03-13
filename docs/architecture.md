# Architecture — qsp-fft

## Overview

`qsp-fft` is the **spectral-transform layer** of the RQM Technologies
ecosystem.  It sits directly on top of `qsp-core` and provides the
windowing, spectral, and frequency-analysis utilities required by higher-level
packages (`qsp-filter`, `qsp-modulation`, …).

```
┌──────────────────────────────┐
│       Application code       │
└──────────────┬───────────────┘
               │ uses
┌──────────────▼───────────────┐
│  qsp-filter / qsp-modulation │
└──────────────┬───────────────┘
               │ uses
┌──────────────▼───────────────┐
│           qsp-fft            │   ← THIS PACKAGE
│  spectrum · windows · analysis│
└──────────────┬───────────────┘
               │ imports
┌──────────────▼───────────────┐
│           qsp-core           │
│  Quaternion · SU(2) helpers  │
└──────────────────────────────┘
```

---

## Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `spectrum.py` | `magnitude_spectrum`, `power_spectrum`, `frequency_bins` |
| `windows.py` | `rectangular_window`, `hann_window`, `hamming_window` |
| `analysis.py` | `dominant_frequency_index`, `dominant_frequency_value`, `spectral_energy` |
| `utils.py` | `next_power_of_two`, `normalise_signal` |

---

## Dependency philosophy

- **qsp-fft imports from qsp-core** for any quaternion types or SU(2) helpers.
- **qsp-fft does NOT reimplement** `Quaternion`, quaternion arithmetic, or
  rotation primitives.
- All spectral logic (FFT wrappers, windows, analysis) lives exclusively in
  this package.

---

## Data-flow example

```
raw signal (np.ndarray)
    │
    ▼
[windows.py]   ← apply a window function to reduce leakage
    │
    ▼
[spectrum.py]  ← compute magnitude / power spectrum via np.fft.rfft
    │
    ▼
[analysis.py]  ← extract dominant frequency, spectral energy
```

---

## Design constraints

1. **Pure Python + NumPy** — no C extensions, no Cython.
2. **One-sided spectra** — all functions return non-negative frequency
   components (`np.fft.rfft` / `np.fft.rfftfreq`).
3. **Small, single-purpose functions** — no stateful objects.
4. **No UI, no deployment code.**
