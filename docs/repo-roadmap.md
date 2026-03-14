# Repository Roadmap — qsp-fft

This document describes appropriate future growth for `qsp-fft` and clarifies
what future work should go elsewhere.

`qsp-fft` is a **spectral-analysis primitives library**.  Future additions
must remain within that scope.

---

## Appropriate future additions

These are in-scope for `qsp-fft`:

### Additional window functions

- Blackman window (`0.42 - 0.5 * cos(2πk/(n-1)) + 0.08 * cos(4πk/(n-1))`)
- Kaiser window (parameterised by shape factor β)
- Flat-top window (for accurate amplitude measurement)
- Bartlett (triangular) window

Each new window function should follow the existing pattern in `windows.py`
and include a NumPy-style docstring and at least one test.

### Spectral-density helpers

- `power_spectral_density(signal, sample_rate)` — normalised by sample rate
- `normalised_power_spectrum(signal)` — divided by FFT length

### Short-time / block-based FFT utilities (STFT)

- `stft(signal, window_fn, hop_size, fft_length)` — short-time Fourier transform
- `stft_magnitude(…)` — magnitude version of the STFT
- These should return 2-D arrays (time × frequency) and stay stateless

### Spectral peak extraction helpers

- `find_spectral_peaks(magnitude, threshold)` — indices of local maxima above threshold
- `peak_frequencies(signal, sample_rate, n_peaks)` — top-N dominant frequencies

### Normalisation and scaling

- `amplitude_spectrum(signal, n)` — `magnitude_spectrum` divided by `n` for
  amplitude-accurate results
- `db_spectrum(signal)` — `20 * log10(magnitude_spectrum + eps)`

### Spectral comparison utilities

- `spectral_distance(signal_a, signal_b)` — L2 distance between magnitude spectra
- `spectral_correlation(signal_a, signal_b)` — normalised cross-spectrum

---

## What should NOT be added here

The following categories of work do **not** belong in `qsp-fft` and should be
implemented in the appropriate sibling or downstream repository instead.

| Category | Where it belongs |
|----------|-----------------|
| Full plotting / reporting suites | downstream application repos |
| Detection / classification systems | downstream application repos |
| Equalizers, synchronization loops | downstream application repos |
| FIR/IIR filter design or application | `qsp-filter` |
| Modulation / demodulation / IQ logic | `qsp-modulation` |
| Orientation / IMU / rotation logic | `qsp-orientation` |
| Quaternion arithmetic or `Quaternion` class | `qsp-core` |
| Hardware or SDR interfaces | downstream application repos |
| End-to-end modem or receiver applications | `quaternionic-modem` or equivalent |
| Communications protocol logic | downstream application repos |
| ML or statistical analysis layers | downstream application repos |

---

## Versioning guidance

- **Patch releases** (0.x.y): bug fixes, docstring improvements, new window
  functions that do not change existing signatures.
- **Minor releases** (0.x.0): new public functions (STFT helpers, peak
  extraction, spectral density).
- **Major releases** (x.0.0): breaking changes to existing public signatures
  or spectral conventions (e.g., changing FFT normalisation).  These require
  explicit documentation of migration steps.

Breaking changes to the public API or spectral conventions should be avoided
unless there is a strong correctness reason.  Downstream callers rely on
the stability of the one-sided FFT convention, frequency-bin layout, and
function signatures.

---

## Contributing new features

Before adding a new public function to `qsp-fft`:

1. Confirm it is a **spectral primitive or reusable spectral helper**.
2. Confirm it does **not** duplicate logic that belongs in `qsp-core`,
   `qsp-filter`, or `qsp-modulation`.
3. Write a NumPy-style docstring.
4. Add at least one test in `tests/`.
5. Export the function from `qsp_fft/__init__.py` and add it to `__all__`.
6. Update `docs/api-overview.md` with a description.
