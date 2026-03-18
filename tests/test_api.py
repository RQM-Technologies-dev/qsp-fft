"""Smoke tests for the top-level public API of qsp.fft (v1 + backward-compat)."""

import importlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestV1ApiImports:
    """All v1 public names must be importable from the top-level package."""

    v1_names = [
        # axis
        "normalize_axis",
        "is_unit_axis",
        "canonical_axes",
        # transforms
        "qdft",
        "iqdft",
        "qfft",
        "iqfft",
        # spectrum helpers
        "spectrum_magnitude",
        "spectrum_energy",
        "total_energy",
        "dominant_bins",
        # validation
        "reconstruction_error",
        "check_parseval",
        "compare_qdft_qfft",
    ]

    def test_all_v1_names_in_package(self):
        import qsp.fft
        for name in self.v1_names:
            assert hasattr(qsp.fft, name), f"qsp.fft.{name} not found in package"

    def test_all_v1_names_in_dunder_all(self):
        import qsp.fft
        for name in self.v1_names:
            assert name in qsp.fft.__all__, f"{name!r} missing from __all__"


class TestBackwardCompatImports:
    """Pre-v1 names must still be importable."""

    legacy_names = [
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

    def test_legacy_names_still_present(self):
        import qsp.fft
        for name in self.legacy_names:
            assert hasattr(qsp.fft, name), f"qsp.fft.{name} missing (backward compat)"


# ---------------------------------------------------------------------------
# Submodule imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "qsp.fft.axis",
    "qsp.fft.quaternion",
    "qsp.fft.qdft",
    "qsp.fft.qfft",
    "qsp.fft.spectrum",
    "qsp.fft.validation",
    "qsp.fft.windows",
    "qsp.fft.analysis",
    "qsp.fft.utils",
])
def test_submodule_importable(module):
    importlib.import_module(module)


# ---------------------------------------------------------------------------
# Output shape smoke tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def setup_method(self):
        self.N = 16
        rng = np.random.default_rng(0)
        self.q = rng.standard_normal((self.N, 4))
        self.u = np.array([1.0, 0.0, 0.0])

    def test_qfft_shape(self):
        from qsp.fft import qfft
        Q = qfft(self.q, self.u)
        assert Q.shape == (self.N, 4)

    def test_iqfft_shape(self):
        from qsp.fft import iqfft, qfft
        Q = qfft(self.q, self.u)
        q_rec = iqfft(Q, self.u)
        assert q_rec.shape == (self.N, 4)

    def test_spectrum_magnitude_shape(self):
        from qsp.fft import qfft, spectrum_magnitude
        Q = qfft(self.q, self.u)
        assert spectrum_magnitude(Q).shape == (self.N,)

    def test_spectrum_energy_shape(self):
        from qsp.fft import qfft, spectrum_energy
        Q = qfft(self.q, self.u)
        assert spectrum_energy(Q).shape == (self.N,)

    def test_total_energy_scalar(self):
        from qsp.fft import total_energy
        e = total_energy(self.q)
        assert isinstance(e, float)

    def test_dominant_bins_top_k(self):
        from qsp.fft import dominant_bins, qfft
        Q = qfft(self.q, self.u)
        idx = dominant_bins(Q, k=3)
        assert idx.shape == (3,)

    def test_dominant_bins_threshold(self):
        from qsp.fft import dominant_bins, qfft
        Q = qfft(self.q, self.u)
        idx = dominant_bins(Q, threshold=0.0)
        assert idx.ndim == 1

    def test_canonical_axes(self):
        from qsp.fft import canonical_axes
        axes = canonical_axes()
        assert set(axes.keys()) == {"i", "j", "k"}
        for ax in axes.values():
            assert ax.shape == (3,)
