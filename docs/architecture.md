# Architecture — qsp-fft

## Role in the QSP Ecosystem

`qsp-fft` is the **Layer-1 QSP spectral-analysis primitives library** of the
RQM Technologies ecosystem.  It sits directly on top of `qsp-core` and
provides the windowing, spectral, and frequency-analysis utilities required by
higher-level packages (`qsp-filter`, `qsp-modulation`, …) and downstream
application systems.

```
┌─────────────────────────────────────────────┐
│          Downstream / Application layer      │
│  (quaternionic-modem, sensing pipelines, …)  │
└─────────────────┬───────────────────────────┘
                  │ uses
┌─────────────────▼───────────────────────────┐
│   qsp-filter · qsp-modulation · qsp-orient.  │
└─────────────────┬───────────────────────────┘
                  │ uses
┌─────────────────▼───────────────────────────┐
│                qsp-fft                       │   ← THIS PACKAGE
│  spectrum · windows · analysis               │
└─────────────────┬───────────────────────────┘
                  │ imports
┌─────────────────▼───────────────────────────┐
│               qsp-core                       │
│      Quaternion · SU(2) helpers              │
└─────────────────────────────────────────────┘
```

`qsp-fft` is intended to remain **focused and composable**: a building block
that downstream systems can rely on without dragging in application-level logic.

---

## QSP Perspective

Within the QSP ecosystem, spectral analysis is not just a convenience for
plotting frequency content.  It is one of the primary ways structured signal
behavior becomes measurable and comparable across domains.  `qsp-fft` provides
the reusable spectral primitives needed by downstream communication, sensing,
and analysis systems without embedding high-level application logic at the
library layer.

---

## Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `spectrum.py` | `magnitude_spectrum`, `power_spectrum`, `frequency_bins` |
| `windows.py` | `rectangular_window`, `hann_window`, `hamming_window` |
| `analysis.py` | `dominant_frequency_index`, `dominant_frequency_value`, `spectral_energy` |
| `utils.py` | `next_power_of_two`, `normalise_signal` |

---

## Boundary

### Belongs in qsp-fft

- Reusable FFT wrappers and spectrum-extraction helpers
- One-sided magnitude and power spectrum calculations
- Frequency-axis generation
- Simple, stateless window functions
- Small spectral utilities (`next_power_of_two`, `normalise_signal`)
- Spectral-analysis demos and examples

### Does NOT belong in qsp-fft

| What to avoid | Where it belongs |
|---------------|-----------------|
| Quaternion arithmetic / `Quaternion` class | `qsp-core` |
| SU(2) rotation primitives | `qsp-core` |
| Filtering / normalization pipelines | `qsp-filter` |
| Modulation / IQ / symbol-level logic | `qsp-modulation` |
| IMU / orientation / rotation logic | `qsp-orientation` |
| Plotting-heavy dashboards | downstream application repos |
| Detection / classification systems | downstream application repos |
| SDR / hardware integrations | downstream application repos |
| End-to-end modem or receiver applications | `quaternionic-modem` or equivalent |
| Large statistical or ML analysis layers | downstream application repos |

---

## Relationship to qsp-core

`qsp-core` is the intended home for shared quaternion primitives and
foundational math utilities across the entire QSP ecosystem.  `qsp-fft` builds
spectral-analysis helpers on top of that foundation.

Current spectral utilities operate on real-valued NumPy arrays and rely on
`np.fft.rfft` / `np.fft.rfftfreq`.  Direct use of quaternion types is not
required for these operations, but the architectural dependency on `qsp-core`
is declared in `pyproject.toml` so that future features (e.g., quaternion-
valued FFT analysis) can import from `qsp-core` without introducing a new
dependency at that time.

**qsp-fft must not** reimplement `Quaternion`, quaternion arithmetic, or SU(2)
rotation primitives.  All spectral logic (FFT wrappers, windows, analysis)
lives exclusively in this package.

---

## Downstream Systems

`qsp-fft` is a building block for Layer-2 repositories and application-level
systems.  Examples:

- **`quaternionic-modem`** — combines modulation, spectral analysis, and
  filtering into a cohesive communications pipeline
- **Communication-analysis pipelines** — use magnitude/power spectra to
  characterize channel conditions
- **Waveform inspection tools** — use frequency-bin helpers to label spectral data
- **Spectral diagnostics in sensor systems** — feed `spectral_energy` and
  `dominant_frequency_value` into classification layers
- **Signal-processing experiments** — combine `qsp-fft`, `qsp-filter`, and
  `qsp-modulation` to prototype novel algorithms

`qsp-fft` provides the spectral layer; it does not own the application logic
above it.

---

## Spectral Conventions

All functions follow these conventions consistently.  **These conventions must
remain stable** unless a change is intentional, versioned, and documented.

| Convention | Detail |
|-----------|--------|
| **One-sided FFT** | `np.fft.rfft` / `np.fft.rfftfreq`; all outputs have length `n // 2 + 1` |
| **Frequency bins** | `frequency_bins(n, sample_rate)` returns bin centres in Hz; `sample_rate=1.0` gives normalised frequency |
| **Magnitude spectrum** | Raw `|FFT(signal)|` — not divided by `n` |
| **Power spectrum** | Element-wise square of the magnitude spectrum |
| **Input type** | Real-valued 1-D NumPy arrays; complex input is not a primary target |
| **Window formula** | Symmetric formula using `(n-1)` as denominator so both endpoints are included |
| **Sample rate default** | `sample_rate=1.0` (normalised units) where applicable |

---

## Data-flow example

```
raw signal (np.ndarray)
    │
    ▼
[windows.py]   ← apply a window function to reduce spectral leakage
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
5. **No application logic** — this is a primitives library, not a framework.
