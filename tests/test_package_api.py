"""Tests for the public API surface of qsp_fft."""

import importlib

import pytest


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------

def test_package_importable():
    import qsp_fft  # noqa: F401


def test_all_public_names_importable():
    import qsp_fft

    expected = [
        "magnitude_spectrum",
        "power_spectrum",
        "frequency_bins",
        "rectangular_window",
        "hann_window",
        "hamming_window",
        "dominant_frequency_index",
        "dominant_frequency_value",
        "spectral_energy",
        "next_power_of_two",
        "normalise_signal",
    ]
    for name in expected:
        assert hasattr(qsp_fft, name), f"qsp_fft.{name} not found in package"


def test_all_list_matches_importable_names():
    """Every name in __all__ must be importable at the top level."""
    import qsp_fft

    for name in qsp_fft.__all__:
        assert hasattr(qsp_fft, name), f"{name!r} is in __all__ but not in package"


# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "qsp_fft.spectrum",
    "qsp_fft.windows",
    "qsp_fft.analysis",
    "qsp_fft.utils",
])
def test_submodule_importable(module):
    importlib.import_module(module)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

def test_frequency_bins_smoke():
    from qsp_fft import frequency_bins
    freqs = frequency_bins(16, sample_rate=100.0)
    assert len(freqs) == 9


def test_magnitude_spectrum_smoke():
    import numpy as np
    from qsp_fft import magnitude_spectrum
    mag = magnitude_spectrum(np.ones(32))
    assert mag.shape == (17,)


def test_windows_smoke():
    from qsp_fft import hamming_window, hann_window, rectangular_window
    for fn in (rectangular_window, hann_window, hamming_window):
        w = fn(16)
        assert len(w) == 16


def test_utils_smoke():
    from qsp_fft import next_power_of_two, normalise_signal
    import numpy as np
    assert next_power_of_two(7) == 8
    arr = normalise_signal(np.array([0.0, 4.0, -2.0]))
    assert arr[1] == pytest.approx(1.0)
