# qsp-fft

**Spectral-transform library for the RQM Technologies ecosystem.**

`qsp-fft` provides windowing, spectral analysis, and frequency utilities as a
clean layer on top of [`qsp-core`](https://github.com/RQM-Technologies-dev/qsp-core).

---

## What is qsp-fft?

`qsp-fft` handles everything related to spectral transforms:

- **Window functions** — rectangular, Hann, Hamming
- **Spectral utilities** — magnitude spectrum, power spectrum, frequency bins
- **Analysis helpers** — dominant frequency, spectral energy

It intentionally contains *no* quaternion math.  Quaternion primitives live in
`qsp-core` and are imported from there when needed.

---

## Dependency on qsp-core

```
qsp-core   →   quaternion math, SU(2) helpers
qsp-fft    →   spectral transforms  (imports from qsp-core)
```

Install both packages:

```bash
pip install qsp-core qsp-fft
```

Or, for a local development install of this repo:

```bash
pip install -e ".[dev]"
```

---

## What belongs here vs. in qsp-core

| qsp-fft | qsp-core |
|---------|----------|
| FFT wrappers | `Quaternion` class |
| Window functions | SU(2) rotation helpers |
| Spectral analysis | Quaternion arithmetic |
| Frequency bins | Quaternion norm / conjugate |

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
    └── dependency-on-qsp-core.md
```

---

## License

See [LICENSE](LICENSE).