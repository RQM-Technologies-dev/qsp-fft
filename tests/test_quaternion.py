"""Tests for qsp_fft.quaternion — quaternion helper functions."""

import math

import numpy as np
import pytest

from qsp_fft.quaternion import (
    as_quaternion_array,
    quaternion_conjugate,
    quaternion_exp_pure,
    quaternion_multiply,
    quaternion_norm,
)


class TestAsQuaternionArray:
    def test_single_quaternion_promoted(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = as_quaternion_array(q)
        assert result.shape == (1, 4)

    def test_batch_unchanged(self):
        q = np.zeros((5, 4))
        result = as_quaternion_array(q)
        assert result.shape == (5, 4)

    def test_dtype_float(self):
        q = np.array([1, 0, 0, 0])
        assert as_quaternion_array(q).dtype == float

    def test_wrong_1d_length_raises(self):
        with pytest.raises(ValueError):
            as_quaternion_array(np.array([1.0, 2.0, 3.0]))

    def test_wrong_2d_columns_raises(self):
        with pytest.raises(ValueError):
            as_quaternion_array(np.zeros((3, 3)))

    def test_values_preserved(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = as_quaternion_array(q)
        np.testing.assert_array_equal(result[0], q)


class TestQuaternionNorm:
    def test_unit_quaternion(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(quaternion_norm(q), [1.0])

    def test_pythagorean(self):
        q = np.array([[0.0, 3.0, 4.0, 0.0]])
        np.testing.assert_allclose(quaternion_norm(q), [5.0])

    def test_zero_quaternion(self):
        q = np.zeros((1, 4))
        np.testing.assert_allclose(quaternion_norm(q), [0.0])

    def test_batch(self):
        q = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
        norms = quaternion_norm(q)
        np.testing.assert_allclose(norms, [1., 1., 1.])

    def test_output_shape(self):
        q = np.random.default_rng(0).standard_normal((7, 4))
        assert quaternion_norm(q).shape == (7,)


class TestQuaternionConjugate:
    def test_conjugate_scalar(self):
        q = np.array([[2.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(quaternion_conjugate(q), q)

    def test_conjugate_pure(self):
        q = np.array([[0.0, 1.0, 2.0, 3.0]])
        result = quaternion_conjugate(q)
        expected = np.array([[0.0, -1.0, -2.0, -3.0]])
        np.testing.assert_array_equal(result, expected)

    def test_conjugate_general(self):
        q = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = quaternion_conjugate(q)
        expected = np.array([[1.0, -2.0, -3.0, -4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_double_conjugate_identity(self):
        q = np.array([[5.0, -3.0, 1.0, 2.0]])
        np.testing.assert_array_equal(quaternion_conjugate(quaternion_conjugate(q)), q)

    def test_norm_preserved(self):
        rng = np.random.default_rng(1)
        q = rng.standard_normal((10, 4))
        np.testing.assert_allclose(quaternion_norm(q), quaternion_norm(quaternion_conjugate(q)))


class TestQuaternionMultiply:
    def test_i_times_i_minus_one(self):
        """i * i = -1 (scalar -1)."""
        i = np.array([[0., 1., 0., 0.]])
        result = quaternion_multiply(i, i)
        np.testing.assert_allclose(result, [[-1., 0., 0., 0.]], atol=1e-12)

    def test_i_times_j_is_k(self):
        """i * j = k."""
        i = np.array([[0., 1., 0., 0.]])
        j = np.array([[0., 0., 1., 0.]])
        result = quaternion_multiply(i, j)
        np.testing.assert_allclose(result, [[0., 0., 0., 1.]], atol=1e-12)

    def test_j_times_i_is_minus_k(self):
        """j * i = -k (non-commutative)."""
        i = np.array([[0., 1., 0., 0.]])
        j = np.array([[0., 0., 1., 0.]])
        result = quaternion_multiply(j, i)
        np.testing.assert_allclose(result, [[0., 0., 0., -1.]], atol=1e-12)

    def test_j_times_k_is_i(self):
        j = np.array([[0., 0., 1., 0.]])
        k = np.array([[0., 0., 0., 1.]])
        result = quaternion_multiply(j, k)
        np.testing.assert_allclose(result, [[0., 1., 0., 0.]], atol=1e-12)

    def test_k_times_i_is_j(self):
        k = np.array([[0., 0., 0., 1.]])
        i = np.array([[0., 1., 0., 0.]])
        result = quaternion_multiply(k, i)
        np.testing.assert_allclose(result, [[0., 0., 1., 0.]], atol=1e-12)

    def test_identity_multiplication(self):
        e = np.array([[1., 0., 0., 0.]])
        rng = np.random.default_rng(2)
        q = rng.standard_normal((1, 4))
        np.testing.assert_allclose(quaternion_multiply(e, q), q, atol=1e-12)
        np.testing.assert_allclose(quaternion_multiply(q, e), q, atol=1e-12)

    def test_norm_multiplicative(self):
        """||q1 * q2|| = ||q1|| * ||q2||."""
        rng = np.random.default_rng(3)
        q1 = rng.standard_normal((5, 4))
        q2 = rng.standard_normal((5, 4))
        prod = quaternion_multiply(q1, q2)
        np.testing.assert_allclose(
            quaternion_norm(prod),
            quaternion_norm(q1) * quaternion_norm(q2),
            atol=1e-10,
        )

    def test_output_shape(self):
        rng = np.random.default_rng(4)
        q1 = rng.standard_normal((8, 4))
        q2 = rng.standard_normal((8, 4))
        assert quaternion_multiply(q1, q2).shape == (8, 4)


class TestQuaternionExpPure:
    def test_zero_angle(self):
        """exp(u·0) = 1 (unit scalar quaternion)."""
        ax = np.array([1.0, 0.0, 0.0])
        q = quaternion_exp_pure(ax, 0.0)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_pi_over_two(self):
        """exp(i·π/2) = i."""
        ax = np.array([1.0, 0.0, 0.0])
        q = quaternion_exp_pure(ax, math.pi / 2)
        np.testing.assert_allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1e-12)

    def test_pi(self):
        """exp(j·π) = -1."""
        ax = np.array([0.0, 1.0, 0.0])
        q = quaternion_exp_pure(ax, math.pi)
        np.testing.assert_allclose(q, [-1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_unit_norm(self):
        """||exp(u·θ)|| = 1 for any θ."""
        ax = np.array([1.0, 0.0, 0.0])
        for theta in [0.0, 0.5, 1.0, math.pi, 2 * math.pi]:
            q = quaternion_exp_pure(ax, theta)
            assert pytest.approx(float(np.linalg.norm(q)), abs=1e-12) == 1.0

    def test_scalar_output_shape(self):
        ax = np.array([0.0, 0.0, 1.0])
        q = quaternion_exp_pure(ax, 1.0)
        assert q.shape == (4,)

    def test_array_output_shape(self):
        ax = np.array([1.0, 0.0, 0.0])
        theta = np.array([0.0, 1.0, 2.0])
        q = quaternion_exp_pure(ax, theta)
        assert q.shape == (3, 4)

    def test_cos_sin_formula(self):
        """exp(u·θ) = (cos θ, u·sin θ)."""
        ax = np.array([0.0, 1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        theta = 1.23
        q = quaternion_exp_pure(ax, theta)
        expected = np.array([
            math.cos(theta),
            0.0,
            math.sin(theta) / math.sqrt(2),
            math.sin(theta) / math.sqrt(2),
        ])
        np.testing.assert_allclose(q, expected, atol=1e-12)
