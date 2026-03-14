# Downstream Usage — qsp-fft

This document explains how downstream systems may use `qsp-fft` as a
spectral-analysis primitives layer.  It is aimed at developers building
higher-level QSP repositories, analysis pipelines, or application-level tools
on top of `qsp-fft`.

---

## Guiding principle

`qsp-fft` provides building blocks.  Downstream systems are responsible for:

- Choosing which spectral helpers to call
- Combining spectral results with filtering, modulation, or classification logic
- Interpreting spectral outputs in their application context

`qsp-fft` does not make those decisions.

---

## Basic pattern: windowed spectrum inspection

The most common usage pattern is to window a signal and then compute its
magnitude or power spectrum for inspection or further processing.

```python
import numpy as np
from qsp_fft import hann_window, magnitude_spectrum, frequency_bins

sample_rate = 1000.0          # Hz
n = 512
t = np.arange(n) / sample_rate
signal = np.sin(2 * np.pi * 50 * t)   # 50 Hz tone

windowed = signal * hann_window(n)
mag = magnitude_spectrum(windowed)
freqs = frequency_bins(n, sample_rate=sample_rate)

peak_idx = int(np.argmax(mag))
print(f"Dominant frequency: {freqs[peak_idx]:.1f} Hz")
```

---

## Combining with qsp-filter

A typical signal-processing pipeline applies `qsp-fft` for spectral
inspection and `qsp-filter` for the actual filtering step.

```
raw signal
    │
    ├─▶ [qsp-fft]    magnitude_spectrum → inspect before filtering
    │
    ├─▶ [qsp-filter] apply_filter       → filter the signal
    │
    └─▶ [qsp-fft]    magnitude_spectrum → inspect after filtering
```

`qsp-fft` only owns the spectral inspection steps.  `qsp-filter` owns the
filter-design and application logic.

---

## Combining with qsp-modulation

When inspecting a modulated waveform, `qsp-modulation` generates the signal
and `qsp-fft` analyses its spectral content.

```python
# Hypothetical downstream analysis script — not part of qsp-fft
from qsp_modulation import bpsk_modulate          # owned by qsp-modulation
from qsp_fft import hann_window, power_spectrum, frequency_bins

bits = [1, 0, 1, 1, 0, 1]
waveform = bpsk_modulate(bits, sample_rate=1000.0)

pwr = power_spectrum(waveform * hann_window(len(waveform)))
freqs = frequency_bins(len(waveform), sample_rate=1000.0)
```

`qsp-fft` provides the spectral view; `qsp-modulation` provides the
symbol-level logic.  Neither repo reaches into the other's domain.

---

## Preparing for quaternionic-modem

`quaternionic-modem` (or equivalent downstream repos) combines modulation,
spectral analysis, filtering, and application-level receiver logic.  `qsp-fft`
serves as the spectral-analysis layer inside that pipeline.

Typical integration points:

- `magnitude_spectrum` / `power_spectrum` for channel diagnostics
- `frequency_bins` for frequency-axis labelling in spectrum monitors
- `dominant_frequency_value` for carrier-frequency estimation
- `spectral_energy` for signal-strength or SNR approximations

---

## Energy monitoring

`spectral_energy` provides a scalar measure of signal power that downstream
systems can use for strength comparisons or threshold decisions.

```python
from qsp_fft import spectral_energy

energy_before = spectral_energy(raw_signal)
energy_after  = spectral_energy(processed_signal)
ratio_db = 10 * np.log10(energy_after / energy_before + 1e-12)
```

---

## Zero-padding for higher spectral resolution

Use `next_power_of_two` to zero-pad a signal to the next fast FFT length.

```python
import numpy as np
from qsp_fft import next_power_of_two, magnitude_spectrum, frequency_bins

n_original = 300
n_padded = next_power_of_two(n_original)   # → 512
signal_padded = np.zeros(n_padded)
signal_padded[:n_original] = original_signal

mag = magnitude_spectrum(signal_padded)
freqs = frequency_bins(n_padded, sample_rate=sample_rate)
```

---

## What downstream repos should NOT expect from qsp-fft

- Filter design or application — use `qsp-filter`
- Modulation / demodulation — use `qsp-modulation`
- Orientation / rotation — use `qsp-orientation`
- Quaternion arithmetic — use `qsp-core`
- Plotting or dashboards — implement in the application layer
- Detection / classification logic — implement in the application layer

`qsp-fft` is a reusable spectral primitives library.  Keep it that way.
