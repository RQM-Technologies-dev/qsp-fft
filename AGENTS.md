# AGENTS.md — qsp-fft

## Role

`qsp-fft` is the **Layer-1 QSP spectral-analysis primitives library** of the
RQM Technologies ecosystem.

It provides:
- Windowing functions (rectangular, Hann, Hamming, …)
- Spectral utilities (magnitude spectrum, power spectrum, frequency bins)
- Minimal analysis helpers (dominant frequency, spectral energy)
- Small reusable utilities related to spectral workflows

It is built **on top of `qsp-core`**, which owns all quaternion math and basic
SU(2) helpers.  `qsp-fft` must remain **focused and composable** — it is a
building block, not a full analysis application.

---

## Ecosystem map

```
RQM-Technologies
│
├── qsp-core          ← quaternion math, SU(2) primitives            (NOT here)
│
├── qsp-fft           ← spectral-analysis primitives                 (THIS repo)
│   spectrum · windows · frequency bins · analysis helpers
│
├── qsp-filter        ← filtering / normalization pipelines          (NOT here)
│
├── qsp-modulation    ← modulation schemes, IQ, symbol primitives    (NOT here)
│
├── qsp-orientation   ← IMU / rotation / orientation logic           (NOT here)
│
└── downstream        ← quaternionic-modem, sensing pipelines, etc.  (NOT here)
```

---

## Boundary rules for agents working here

### Belongs in qsp-fft

| Category | Examples |
|----------|---------|
| FFT / IFFT wrappers | `magnitude_spectrum`, `power_spectrum` |
| Frequency-axis helpers | `frequency_bins` |
| Window functions | `rectangular_window`, `hann_window`, `hamming_window` |
| Spectral analysis helpers | `dominant_frequency_index`, `dominant_frequency_value`, `spectral_energy` |
| Small spectral utilities | `next_power_of_two`, `normalise_signal` |
| Spectral-analysis demos / examples | files under `examples/` |

### Does NOT belong in qsp-fft

| What to avoid | Where it belongs instead |
|---------------|--------------------------|
| `Quaternion` class or quaternion arithmetic | `qsp-core` |
| SU(2) rotation primitives | `qsp-core` |
| Filtering / FIR / IIR pipelines | `qsp-filter` |
| Digital modulation / IQ / symbol schemes | `qsp-modulation` |
| IMU / orientation / rotation logic | `qsp-orientation` |
| Plotting-heavy dashboards or reporting suites | downstream application repos |
| Detection / classification systems | downstream application repos |
| Equalizers, synchronization loops | downstream application repos |
| SDR / hardware integrations | downstream application repos |
| End-to-end modem or receiver applications | `quaternionic-modem` or equivalent |
| Large statistical or ML analysis layers | downstream application repos |

**Do not** reimplement `Quaternion` or SU(2) primitives in this repository.
**Do** import from `qsp_core` when quaternion types or helpers are needed.
**Do not** absorb filtering, modulation, orientation, or application-level
analysis logic — those belong in their respective sibling repos.

---

## Boundary preservation instructions

These rules must be followed by all contributors and AI agents:

1. **Keep qsp-fft small and composable.**  If a proposed feature cannot be
   described as a spectral primitive or reusable spectral helper, it does not
   belong here.
2. **Do not add stateful objects** (classes that hold signal history, filter
   state, etc.).  All public functions must be stateless transforms.
3. **Preserve the public API contract.**  Public function signatures in
   `qsp_fft/__init__.py` must not be changed in a breaking way without a
   version bump and corresponding test updates.
4. **Spectral conventions must remain stable** unless the change is
   intentional, versioned, and documented.  The one-sided FFT convention,
   frequency-bin generation method, and normalisation behaviour are load-
   bearing for downstream callers.
5. **Do not introduce UI, plotting, or deployment code** anywhere in this repo.
6. **New behaviour requires a test.**  Every new public function must have
   corresponding tests in `tests/`.
7. **Every public function must have a NumPy-style docstring.**

---

## Coding standards

1. Pure Python (plus `numpy`).  No C extensions, no Cython.
2. Every public function must have a NumPy-style docstring.
3. Every new behaviour must have a corresponding test in `tests/`.
4. Functions should be small and single-purpose.
5. No UI code, no deployment code.
6. No stateful objects in the public API.

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
    ├── dependency-on-qsp-core.md
    ├── downstream-usage.md
    └── repo-roadmap.md
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
