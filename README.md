# qsp-fft

**Spectral-transform primitives library for the RQM Technologies QSP ecosystem.**

`qsp-fft` is the **Layer-1 QSP library** for reusable spectral-analysis
primitives.  It provides windowing, spectral analysis, and frequency utilities
as a focused, composable building block for downstream signal-analysis systems.

---

## Role in the QSP Ecosystem

`qsp-fft` is one layer in the RQM Technologies Quaternionic Signal Processing
(QSP) platform:

```
┌─────────────────────────────────────────────┐
│          Downstream / Application layer      │
│  (quaternionic-modem, sensing pipelines, …)  │
└─────────────────┬───────────────────────────┘
                  │ uses
┌─────────────────▼───────────────────────────┐
│       qsp-filter  ·  qsp-modulation          │
│       qsp-orientation  ·  …                  │
└─────────────────┬───────────────────────────┘
                  │ uses
┌─────────────────▼───────────────────────────┐
│                qsp-fft                       │  ← THIS REPO
│  spectrum · windows · frequency bins         │
└─────────────────┬───────────────────────────┘
                  │ imports
┌─────────────────▼───────────────────────────┐
│               qsp-core                       │
│      Quaternion · SU(2) helpers              │
└─────────────────────────────────────────────┘
```

This repository provides:

- **FFT-based spectrum helpers** — magnitude spectrum, power spectrum
- **Frequency-bin generation** — one-sided frequency axes from FFT length and sample rate
- **Lightweight window functions** — rectangular, Hann, Hamming
- **Reusable building blocks** for downstream signal-analysis, communications, and sensing systems

`qsp-fft` is intended to support higher-level systems without becoming a full
signal-analysis application framework.  It stays small, composable, and focused.

---

## Why This Repo Matters

Frequency-domain analysis is central to many signal-processing workflows.
`qsp-fft` provides the reusable spectral primitives that make the following
practical across the QSP ecosystem:

- **Waveform inspection** — verify spectral content of generated or received signals
- **Signal diagnostics** — identify dominant frequencies and energy distribution
- **Spectrum-based experiments** — prototype analysis algorithms on top of stable helpers
- **Communications prototyping** — inspect modulated waveforms before and after transmission
- **Downstream receiver and sensing pipelines** — feed clean spectral representations into classification and detection systems

Without a shared spectral layer, every downstream package would reimplement
the same FFT wrappers, window functions, and frequency-bin generators
inconsistently.  `qsp-fft` is the single source of truth for these primitives.

---

## QSP Perspective

Within the QSP ecosystem, spectral analysis is not just a convenience for
plotting frequency content.  It is one of the primary ways structured signal
behavior becomes measurable and comparable across domains.  `qsp-fft` provides
the reusable spectral primitives needed by downstream communication, sensing,
and analysis systems without embedding high-level application logic at the
library layer.

---

## What belongs here vs. what does not

### Belongs in qsp-fft

| Category | Examples |
|----------|---------|
| FFT wrappers / helpers | `magnitude_spectrum`, `power_spectrum` |
| Frequency-axis helpers | `frequency_bins` |
| Window functions | `rectangular_window`, `hann_window`, `hamming_window` |
| Spectral analysis primitives | `dominant_frequency_value`, `spectral_energy` |
| Utility helpers for spectral work | `next_power_of_two`, `normalise_signal` |
| Spectral-analysis demos / examples | `examples/spectrum_demo.py`, `examples/window_demo.py` |

### Does NOT belong in qsp-fft

| What | Where it belongs |
|------|-----------------|
| Generic quaternion algebra | `qsp-core` |
| Filtering and normalization pipelines | `qsp-filter` |
| Digital modulation / IQ / symbol schemes | `qsp-modulation` |
| IMU / orientation / rotation logic | `qsp-orientation` |
| Plotting-heavy dashboards | downstream application repos |
| Complete signal-analysis frameworks | downstream application repos |
| Application-specific detection / classification | downstream application repos |
| SDR / hardware integrations | downstream application repos |
| End-to-end modem or communication receiver logic | `quaternionic-modem` or equivalent |
| Large statistical analysis layers | downstream application repos |

`qsp-fft` is a **building-block spectral library**, not a full analysis application.

---

## Relationship to qsp-core

`qsp-core` is the intended home for shared quaternion primitives and
foundational math utilities.  `qsp-fft` builds spectral-analysis helpers on
top of that foundation.

```
qsp-core   →   Quaternion class, SU(2) helpers, quaternion arithmetic
qsp-fft    →   FFT wrappers, window functions, spectral analysis helpers
```

Current implementations in `qsp-fft` primarily rely on NumPy and standard
signal-processing math (`np.fft.rfft`, `np.fft.rfftfreq`).  This is
acceptable: the spectral domain does not always require quaternion types
directly.  However, when a future feature does require quaternion types (for
example, quaternion-valued FFT analysis), `qsp-fft` will import from
`qsp-core` rather than reimplement those primitives.

The dependency is declared in `pyproject.toml` so the architectural
relationship is explicit even where it is not yet exercised in code.

Install both packages:

```bash
pip install qsp-core qsp-fft
```

Or, for a local development install of this repo:

```bash
pip install -e ".[dev]"
```

---

## Relationship to qsp-filter and qsp-modulation

These three repositories are **complementary and often used together**, but
their responsibilities are distinct:

| Repository | Responsibility |
|------------|---------------|
| `qsp-fft` | Spectral transforms, spectrum extraction, frequency-bin generation, windowing |
| `qsp-filter` | Filtering pipelines, FIR/IIR design, normalization workflows |
| `qsp-modulation` | Modulation schemes, IQ generation, symbol-level primitives |

A typical analysis pipeline might apply `qsp-modulation` to generate a
waveform, run `qsp-fft` to inspect its spectrum, and then pass results through
`qsp-filter` — but each repo keeps its own scope clean.

---

## Downstream Systems

`qsp-fft` is a building block for Layer-2 repositories and application-level
systems.  Examples of downstream use include:

- **`quaternionic-modem`** — combines modulation, spectral analysis, and filtering into a cohesive communications pipeline
- **Communication-analysis pipelines** — use magnitude/power spectra to characterize channel conditions
- **Waveform inspection tools** — use frequency-bin helpers to label and display spectral data
- **Spectral diagnostics in sensor systems** — feed `spectral_energy` and `dominant_frequency_value` results into classification layers
- **Downstream experiments** — combine `qsp-fft`, `qsp-filter`, and `qsp-modulation` to prototype novel signal-processing algorithms

See [docs/downstream-usage.md](docs/downstream-usage.md) for concrete usage patterns.

---

## Spectral Conventions

All functions in `qsp-fft` follow a consistent set of conventions:

| Convention | Detail |
|-----------|--------|
| **One-sided FFT** | All spectrum functions use `np.fft.rfft` / `np.fft.rfftfreq` and return only non-negative frequency components of length `n // 2 + 1` |
| **Frequency bins** | `frequency_bins(n, sample_rate)` returns bin centres in Hz; `sample_rate=1.0` gives normalised frequency |
| **Magnitude spectrum** | Raw `|FFT(signal)|` — not divided by `n`.  Downstream callers should normalise if needed. |
| **Power spectrum** | `|FFT(signal)|²`, i.e., element-wise square of the magnitude spectrum |
| **Input signals** | All helpers accept real-valued 1-D NumPy arrays.  Complex-valued input is not a primary target in the current release. |
| **Window conventions** | Window functions use the symmetric formula `0.5 * (1 - cos(2π k / (n-1)))` so that both endpoints are included in the taper |
| **Sample rate default** | Functions that accept `sample_rate` default to `1.0` (normalised units) so they are usable without a known sample rate |

These conventions should remain stable across releases.  If they must change,
the change should be intentional, versioned, and documented.

---

## Future Extensions

See [docs/repo-roadmap.md](docs/repo-roadmap.md) for a detailed forward-looking
guide.  In brief, appropriate future additions include:

- Additional window functions (Blackman, Kaiser, flat-top, …)
- Spectral-density helpers
- Short-time / block-based FFT utilities (STFT)
- Spectral peak extraction helpers
- More robust normalization and scaling options
- Spectral comparison utilities

Additions that do **not** belong here (they belong in downstream or sibling repos):

- Full plotting or reporting suites
- Detection or classification systems
- Equalizers, synchronization loops, or filtering pipelines
- Communications protocol logic
- Hardware or SDR interfaces
- End-to-end modem applications

---

## Quick start

```python
import numpy as np
from qsp_fft import magnitude_spectrum, frequency_bins, hann_window

# Build a simple sinusoidal signal
sample_rate = 1000        # Hz
n = 512
t = np.arange(n) / sample_rate
signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz tone

# Apply a Hann window and compute the spectrum
window = hann_window(n)
mag = magnitude_spectrum(signal * window)
freqs = frequency_bins(n, sample_rate)

peak_idx = int(np.argmax(mag))
print(f"Peak frequency: {freqs[peak_idx]:.1f} Hz")  # → 50.0 Hz
```

---

## Running tests

```bash
pytest
```

---

## Running examples

```bash
python examples/spectrum_demo.py
python examples/window_demo.py
```

---

## Repository layout

```
qsp-fft
├── AGENTS.md
├── README.md
├── pyproject.toml
├── qsp_fft/
│   ├── __init__.py
│   ├── spectrum.py
│   ├── windows.py
│   ├── analysis.py
│   └── utils.py
├── tests/
│   ├── test_spectrum.py
│   ├── test_windows.py
│   ├── test_analysis.py
│   └── test_package_api.py
├── examples/
│   ├── spectrum_demo.py
│   └── window_demo.py
└── docs/
    ├── architecture.md
    ├── api-overview.md
    ├── dependency-on-qsp-core.md
    ├── downstream-usage.md
    └── repo-roadmap.md
```

---

## License

See [LICENSE](LICENSE).