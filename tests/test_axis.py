"""Tests for qsp.fft.axis — axis utilities."""

import numpy as np
import pytest

from qsp.fft.axis import canonical_axes, is_unit_axis, normalize_axis


class TestNormalizeAxis:
    def test_unit_vector_unchanged(self):
        ax = np.array([1.0, 0.0, 0.0])
        result = normalize_axis(ax)
        np.testing.assert_allclose(result, ax)

    def test_non_unit_vector_normalised(self):
        ax = np.array([3.0, 0.0, 0.0])
        result = normalize_axis(ax)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_arbitrary_vector(self):
        ax = np.array([1.0, 2.0, 2.0])
        result = normalize_axis(ax)
        assert pytest.approx(float(np.linalg.norm(result)), abs=1e-12) == 1.0

    def test_negative_axis(self):
        ax = np.array([-2.0, 0.0, 0.0])
        result = normalize_axis(ax)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0])

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="nonzero"):
            normalize_axis(np.array([0.0, 0.0, 0.0]))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            normalize_axis(np.array([1.0, 0.0]))

    def test_returns_float_array(self):
        ax = np.array([1, 0, 0])
        result = normalize_axis(ax)
        assert result.dtype == float

    def test_output_shape(self):
        ax = np.array([1.0, 2.0, 3.0])
        assert normalize_axis(ax).shape == (3,)


class TestIsUnitAxis:
    def test_unit_axes_return_true(self):
        for ax in [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]:
            assert is_unit_axis(ax) is True

    def test_non_unit_returns_false(self):
        assert is_unit_axis(np.array([1.0, 1.0, 0.0])) is False
        assert is_unit_axis(np.array([2.0, 0.0, 0.0])) is False

    def test_zero_vector_returns_false(self):
        assert is_unit_axis(np.array([0.0, 0.0, 0.0])) is False

    def test_wrong_shape_returns_false(self):
        assert is_unit_axis(np.array([1.0, 0.0])) is False
        assert is_unit_axis(np.array([1.0, 0.0, 0.0, 0.0])) is False

    def test_atol_respected(self):
        ax = np.array([1.0 + 1e-9, 0.0, 0.0])
        assert is_unit_axis(ax, atol=1e-8) is True
        assert is_unit_axis(ax, atol=1e-12) is False


class TestCanonicalAxes:
    def test_keys(self):
        axes = canonical_axes()
        assert set(axes.keys()) == {"i", "j", "k"}

    def test_i_axis(self):
        np.testing.assert_array_equal(canonical_axes()["i"], [1.0, 0.0, 0.0])

    def test_j_axis(self):
        np.testing.assert_array_equal(canonical_axes()["j"], [0.0, 1.0, 0.0])

    def test_k_axis(self):
        np.testing.assert_array_equal(canonical_axes()["k"], [0.0, 0.0, 1.0])

    def test_all_unit(self):
        for name, ax in canonical_axes().items():
            assert is_unit_axis(ax), f"axis '{name}' is not unit"

    def test_returns_float_arrays(self):
        for ax in canonical_axes().values():
            assert ax.dtype == float
