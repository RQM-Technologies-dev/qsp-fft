# Dependency on qsp-core

## Why qsp-core?

`qsp-core` is the shared foundation of the RQM Technologies ecosystem.  It
owns all **quaternion math** and **basic SU(2) helpers** so that downstream
packages (`qsp-fft`, `qsp-filter`, `qsp-modulation`, …) do not re-invent the
same primitives.

---

## What qsp-core provides

- `Quaternion` — the core quaternion type with arithmetic operators
- SU(2) rotation helpers (e.g., `rotation_matrix`, `su2_from_axis_angle`)
- Quaternion norms, conjugates, and interpolation utilities
- Any other shared math that is quaternion-specific

See the [qsp-core repository](https://github.com/RQM-Technologies-dev/qsp-core)
for the canonical list.

---

## What qsp-fft does NOT contain

Because `qsp-core` is the authoritative source, **qsp-fft must not**:

- Define a `Quaternion` class or struct.
- Implement quaternion multiplication, norm, or conjugate.
- Implement SU(2) rotation matrices from scratch.
- Duplicate any utility already present in `qsp-core`.

---

## How qsp-fft uses qsp-core

When a future feature of `qsp-fft` requires quaternion types (for example,
quaternion-valued FFT or SU(2)-based spectral analysis), the import pattern is:

```python
from qsp_core import Quaternion          # the shared type
from qsp_core.su2 import rotation_matrix # SU(2) helpers
```

Current spectral utilities (`magnitude_spectrum`, `hann_window`, …) operate on
real-valued NumPy arrays and do not yet require quaternion types, but the
dependency is declared in `pyproject.toml` so the foundation is in place.

---

## Dependency declaration

`pyproject.toml`:

```toml
[project]
dependencies = [
    "qsp-core>=0.1.0",
    "numpy>=1.24",
]
```

---

## Boundary summary

| Layer | Package | What it owns |
|-------|---------|--------------|
| Quaternion math | `qsp-core` | `Quaternion`, SU(2), norms |
| Spectral transforms | `qsp-fft` | FFT wrappers, windows, analysis |
| Filtering | `qsp-filter` | FIR/IIR filters (builds on both) |
| Modulation | `qsp-modulation` | modulation schemes (builds on both) |
