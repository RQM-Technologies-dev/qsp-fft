<img src="https://github.com/RQM-Technologies-dev/qsp-fft/actions/workflows/ci.yml/badge.svg" alt="CI Status">

# qsp-fft

`qsp-fft` provides the Quaternionic Discrete Fourier Transform (QDFT) and
Quaternionic Fast Fourier Transform (QFFT) for quaternion-valued signals.
It implements a fixed-axis, right-sided spectral decomposition that extends
classical FFT ideas into quaternionic signal space, enabling frequency
analysis of signals carrying phase, orientation, and polarisation as a
single coherent structure.

---

## Role in the QSP Ecosystem

`qsp-fft` is the **spectral engine** in the RQM Technologies Quaternionic
Signal Processing (QSP) platform:

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
│  QDFT/QFFT · spectrum · windows · freq bins  │
└─────────────────┬───────────────────────────┘
                  │ imports
┌─────────────────▼───────────────────────────┐
│               qsp-core                       │
│      Quaternion · SU(2) helpers              │
└─────────────────────────────────────────────┘
```

It is responsible for:

- Quaternionic spectral decomposition (QDFT / QFFT)
- Inverse reconstruction (iqdft / iqfft)
- Spectral magnitude/energy helpers
- Correctness/validation tools
- Classical (real-signal) FFT helpers and window functions (backward-compatible)

---

## v1 Scope and Positioning

v1 is intentionally narrow:

| Dimension | v1 choice |
|-----------|-----------|
| Convention | **Right-sided** only |
| Axis | **Fixed** analysis axis per transform call |
| Signal dimension | **1D** quaternion-valued sequences |
| Implementation | **NumPy-first**, no C extensions or GPU code |
| Reference | Direct O(N²) **qdft** for correctness verification |
| Production | Fast O(N log N) **qfft** via slice decomposition |

---

## Mathematical Definition

### Forward QDFT (right-sided, fixed axis u)

For quaternion samples `q[0], …, q[N-1]` and a fixed unit pure
quaternion axis `u` with `||u|| = 1`, `u² = -1`:

```
Q_u[k] = Σ_{n=0}^{N-1}  q[n] · exp(-u · 2πkn/N),   k = 0, …, N-1
```

where `exp(u·θ) = cos(θ) + u·sin(θ)` and multiplication is the Hamilton
product (the exponential lies on the **right** of `q[n]`).

### Inverse QDFT

```
q[n] = (1/N) Σ_{k=0}^{N-1}  Q_u[k] · exp(+u · 2πkn/N),   n = 0, …, N-1
```

---

## Quick Start

```python
import numpy as np
from qsp_fft import canonical_axes, qfft, iqfft, spectrum_magnitude, reconstruction_error

# Create a small quaternionic signal: shape (N, 4) in (w, x, y, z) order
rng = np.random.default_rng(0)
N = 16
q = rng.standard_normal((N, 4))

# Choose an analysis axis
u = canonical_axes()["i"]   # [1, 0, 0]

# Forward transform
Q = qfft(q, u)               # shape (N, 4)

# Per-bin spectrum magnitudes
mag = spectrum_magnitude(Q)  # shape (N,)
print("Magnitudes:", mag.round(3))

# Invert
q_rec = iqfft(Q, u)          # shape (N, 4)
print("Reconstruction error:", reconstruction_error(q, q_rec))
```

---

## Design Principles

| Principle | Detail |
|-----------|--------|
| **Right-sided convention** | `exp(-u·2πkn/N)` multiplies on the right of each sample |
| **Fixed-axis slice analysis** | One unit pure-quaternion axis per call; the algebra decomposes into two complex FFTs in the slice `C_u` |
| **Signal format** | Quaternion signals as `(N, 4)` NumPy arrays in `(w, x, y, z)` order |
| **Two implementations** | `qdft`/`iqdft` for reference correctness; `qfft`/`iqfft` for production |
| **NumPy-first** | No C extensions, GPU, or heavy runtime dependencies |
| **Stateless functions** | All public functions are pure transforms with no mutable state |

---

## Public API

### v1 Quaternionic FFT

```python
# Axis utilities
from qsp_fft import normalize_axis, is_unit_axis, canonical_axes

# Transforms
from qsp_fft import qdft, iqdft    # direct O(N²) reference
from qsp_fft import qfft, iqfft    # fast O(N log N)

# Spectrum helpers
from qsp_fft import spectrum_magnitude, spectrum_energy, total_energy, dominant_bins

# Validation
from qsp_fft import reconstruction_error, check_parseval, compare_qdft_qfft
```

### Classical helpers (backward-compatible)

```python
from qsp_fft import magnitude_spectrum, power_spectrum, frequency_bins
from qsp_fft import rectangular_window, hann_window, hamming_window
from qsp_fft import dominant_frequency_index, dominant_frequency_value, spectral_energy
from qsp_fft import next_power_of_two, normalise_signal
```

---

## Repository Layout

```
qsp-fft
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/
│   └── qsp_fft/
│       ├── __init__.py     ← public API exports
│       ├── py.typed        ← PEP 561 marker
│       ├── axis.py         ← normalize_axis, is_unit_axis, canonical_axes
│       ├── quaternion.py   ← quaternion helpers (w,x,y,z convention)
│       ├── qdft.py         ← direct O(N²) QDFT reference
│       ├── qfft.py         ← fast O(N log N) QFFT via slice decomposition
│       ├── spectrum.py     ← classical + quaternionic spectrum helpers
│       ├── validation.py   ← reconstruction_error, check_parseval, compare_qdft_qfft
│       ├── windows.py      ← rectangular, Hann, Hamming windows
│       ├── analysis.py     ← dominant frequency helpers
│       └── utils.py        ← next_power_of_two, normalise_signal
├── tests/
│   ├── test_api.py
│   ├── test_axis.py
│   ├── test_quaternion.py
│   ├── test_qdft.py
│   ├── test_qfft.py
│   ├── test_inverse.py
│   ├── test_parseval.py
│   ├── test_spectrum.py
│   ├── test_windows.py
│   ├── test_analysis.py
│   └── test_package_api.py
├── examples/
│   ├── basic_qfft.py
│   ├── compare_qdft_vs_qfft.py
│   ├── axis_comparison.py
│   ├── spectral_peaks.py
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

## Spectral Conventions

| Convention | Detail |
|------------|--------|
| **Right-sided QDFT** | `exp(-u·2πkn/N)` on the right; inverse has `1/N` normalisation |
| **Quaternion format** | `(N, 4)` arrays in `(w, x, y, z)` order |
| **Parseval relation** | `Σ_n ‖q[n]‖² = (1/N) · Σ_k ‖Q[k]‖²` |
| **Classical one-sided FFT** | Classical helpers use `np.fft.rfft`/`rfftfreq` (non-negative bins) |
| **Frequency bins** | `frequency_bins(n, sample_rate)` for real-signal helpers |
| **Window conventions** | Symmetric formula `0.5·(1 − cos(2πk/(n−1)))` for Hann |

---

## What Belongs Here vs. What Does Not

### Belongs in qsp-fft

| Category | Examples |
|----------|----------|
| QDFT/QFFT transforms | `qdft`, `iqdft`, `qfft`, `iqfft` |
| Axis helpers | `normalize_axis`, `canonical_axes` |
| Quaternionic spectrum helpers | `spectrum_magnitude`, `dominant_bins` |
| Validation tools | `reconstruction_error`, `check_parseval` |
| Classical FFT wrappers | `magnitude_spectrum`, `power_spectrum` |
| Window functions | `hann_window`, `hamming_window` |
| Spectral-analysis demos | files under `examples/` |

### Does NOT belong in qsp-fft

| What | Where it belongs |
|------|-----------------|
| `Quaternion` class, generic quaternion arithmetic | `qsp-core` |
| SU(2) rotation primitives | `qsp-core` |
| Filtering / FIR / IIR pipelines | `qsp-filter` |
| Digital modulation / IQ / symbol schemes | `qsp-modulation` |
| IMU / orientation / rotation logic | `qsp-orientation` |
| GUI, web apps, plotting dashboards | downstream application repos |
| SDR / hardware integrations | downstream application repos |
| End-to-end modem applications | `quaternionic-modem` or equivalent |

---

## Roadmap

Future additions being considered (not yet scoped):

- **Multi-axis spectra** — two-sided or bilateral QDFT
- **Quaternionic STFT** — short-time quaternionic spectral analysis
- **SU(2) harmonic transforms** — spherical-harmonic decompositions
- **Additional window functions** — Blackman, Kaiser, flat-top
- **Spectral-density helpers** — PSD estimation for quaternion signals

---

## Installation

```bash
pip install qsp-core qsp-fft
```

For a local development install:

```bash
pip install -e ".[dev]"
```

---

## Running Tests

```bash
pytest
```

---

## Running Examples

```bash
python examples/basic_qfft.py
python examples/compare_qdft_vs_qfft.py
python examples/axis_comparison.py
python examples/spectral_peaks.py
```

---

## Publishing

Releases are automatically published to PyPI via GitHub Actions and PyPI
Trusted Publishing.

---

## License

See [LICENSE](LICENSE).
