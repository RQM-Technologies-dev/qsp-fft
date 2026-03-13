# AGENTS.md — qsp-fft

## Role

`qsp-fft` is the **spectral-transform layer** of the RQM Technologies ecosystem.

It provides:
- Windowing functions (rectangular, Hann, Hamming, …)
- Spectral utilities (magnitude spectrum, power spectrum, frequency bins)
- Minimal analysis helpers (dominant frequency, spectral energy)

It is built **on top of `qsp-core`**, which owns all quaternion math and basic
SU(2) helpers.

---

## Ecosystem map

```
RQM-Technologies
├── qsp-core          ← quaternion math, SU(2) primitives  (NOT here)
├── qsp-fft           ← spectral transforms               (THIS repo)
├── qsp-filter        ← filtering, built on qsp-core
├── qsp-modulation    ← modulation, built on qsp-core
└── …
```

---

## Boundary rules for agents working here

| Belongs in qsp-fft | Belongs in qsp-core |
|--------------------|---------------------|
| FFT / IFFT wrappers | `Quaternion` class |
| Window functions | SU(2) rotation helpers |
| Spectral analysis helpers | Quaternion norm / conjugate |
| Frequency-bin utilities | Any quaternion arithmetic |

**Do not** reimplement `Quaternion` or SU(2) primitives in this repository.
**Do** import from `qsp_core` when quaternion types or helpers are needed.

---

## Coding standards

1. Pure Python (plus `numpy`).  No C extensions, no Cython.
2. Every public function must have a NumPy-style docstring.
3. Every new behaviour must have a corresponding test in `tests/`.
4. Functions should be small and single-purpose.
5. No UI code, no deployment code.

---

## Directory layout

```
qsp-fft
├── AGENTS.md           ← this file
├── README.md
├── pyproject.toml
├── qsp_fft/
│   ├── __init__.py     ← public API
│   ├── spectrum.py     ← magnitude_spectrum, power_spectrum, frequency_bins
│   ├── windows.py      ← rectangular_window, hann_window, hamming_window
│   ├── analysis.py     ← dominant_frequency_*, spectral_energy
│   └── utils.py        ← shared low-level helpers
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
    └── dependency-on-qsp-core.md
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
