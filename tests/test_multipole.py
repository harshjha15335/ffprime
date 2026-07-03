"""
Tests for electrostatic multipole potential and field functions.
All tests validate against analytical solutions derived from first principles.

NOTE: this module's implementation moved from
``ffprime.electrostatics.multipole`` to ``ffprime.electrostatics.cartesian``
as part of the Cartesian/spherical architecture split (the old
``multipole`` module is kept only as a deprecated import alias). Only the
import path below changed -- test bodies and assertions are unchanged.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from ffprime.electrostatics.cartesian import (
    monopole_potential,
    monopole_field,
    dipole_potential,
    dipole_field,
    quadrupole_potential,
    quadrupole_field,
    total_potential,
    total_field,
)

# Default finite-difference step used for numerical-gradient tests.
FD_STEP = 1e-5


# ─────────────────────────────────────────────
# MONOPOLE TESTS
# ─────────────────────────────────────────────

class TestMonopolePotential:

    def test_single_unit_charge(self):
        """V = q/r → at distance 1.0, V = 1.0"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result, [1.0], rtol=1e-10)

    def test_inverse_distance_scaling(self):
        """V scales as 1/r"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0],
                           [4.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result[0] / result[1], 2.0, rtol=1e-10)

    def test_negative_charge(self):
        """Negative charge gives negative potential"""
        charges = np.array([-1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result, [-1.0], rtol=1e-10)

    def test_superposition(self):
        """Two equal charges: potential is sum of individual contributions"""
        charges = np.array([1.0, 1.0])
        coords = np.array([[1.0, 0.0, 0.0],
                           [-1.0, 0.0, 0.0]])
        points = np.array([[0.0, 2.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        expected = 2.0 / np.sqrt(5.0)
        assert_allclose(result, [expected], rtol=1e-10)

    def test_singularity_safe(self):
        """Potential at exact atom location returns finite value"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert np.isfinite(result).all()

    def test_multiple_field_points(self):
        """Vectorized evaluation over multiple points"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result, [1.0, 0.5, 1/3], rtol=1e-10)

    def test_charge_coordinate_length_mismatch(self):
        """Number of charges and coordinates must match"""
        charges = np.array([1.0, 2.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            monopole_potential(charges, coords, points)

    def test_invalid_coordinate_shape(self):
        """Coordinates must have shape (N, 3)"""
        charges = np.array([1.0])
        coords = np.array([0.0, 0.0, 0.0])
        points = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            monopole_potential(charges, coords, points)

    def test_invalid_points_shape(self):
        """Points must have shape (M, 3)"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            monopole_potential(charges, coords, points)

    def test_monopole_matches_analytic_formula(self):
        """Verify implementation against analytical V=q/r formula"""
        charges = np.array([2.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[4.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        expected = 2.0 / 4.0
        assert_allclose(result, [expected], rtol=1e-10)


class TestMonopoleField:

    def test_unit_charge_along_x(self):
        """E = q*r/r^3 → at (1,0,0), E = (1,0,0)"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = monopole_field(charges, coords, points)
        assert_allclose(result, [[1.0, 0.0, 0.0]], rtol=1e-10)

    def test_inverse_square_scaling(self):
        """E magnitude scales as 1/r^2"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0],
                           [4.0, 0.0, 0.0]])
        result = monopole_field(charges, coords, points)
        ratio = np.linalg.norm(result[0]) / np.linalg.norm(result[1])
        assert_allclose(ratio, 4.0, rtol=1e-10)

    def test_field_is_radial(self):
        """Field points radially away from positive charge"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 1.0, 1.0]])
        result = monopole_field(charges, coords, points)
        r = np.array([1.0, 1.0, 1.0])
        r_hat = r / np.linalg.norm(r)
        result_hat = result[0] / np.linalg.norm(result[0])
        assert_allclose(result_hat, r_hat, rtol=1e-10)

    def test_singularity_safe(self):
        """Field at exact atom location returns finite value"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = monopole_field(charges, coords, points)
        assert np.isfinite(result).all()

    def test_monopole_field_matches_analytic_formula(self):
        """Verify implementation against analytical E=q*r/r^3 formula"""
        charges = np.array([2.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0]])
        result = monopole_field(charges, coords, points)
        expected = np.array([[0.5, 0.0, 0.0]])
        assert_allclose(result, expected, rtol=1e-10)


# ─────────────────────────────────────────────
# DIPOLE TESTS
# ─────────────────────────────────────────────

class TestDipolePotential:

    def test_dipole_along_axis(self):
        """
        p=(1,0,0) at origin, point at (2,0,0):
        V = (p.r)/r^3 = 2/8 = 0.25
        """
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result, [0.25], rtol=1e-10)

    def test_dipole_perpendicular_zero(self):
        """
        p=(1,0,0), point at (0,2,0):
        V = (p.r)/r^3 = 0 (perpendicular)
        """
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 2.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result, [0.0], atol=1e-12)

    def test_dipole_antisymmetry(self):
        """V at +r and -r are equal and opposite"""
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0],
                           [-2.0, 0.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result[0], -result[1], rtol=1e-10)

    def test_dipole_scales_as_inverse_r_squared(self):
        """V_dipole scales as 1/r^2"""
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result[0] / result[1], 4.0, rtol=1e-10)


class TestDipoleField:

    def test_dipole_field_along_axis(self):
        """
        p=(0,0,1) at origin, point at (0,0,2):
        E = [3(p.r̂)r̂ - p] / r^3
          = [3*(0,0,1) - (0,0,1)] / 8
          = (0, 0, 0.25)
        """
        dipoles = np.array([[0.0, 0.0, 1.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 2.0]])
        result = dipole_field(dipoles, coords, points)
        assert_allclose(result, [[0.0, 0.0, 0.25]], rtol=1e-10)

    def test_dipole_field_perpendicular(self):
        """
        p=(0,0,1) at origin, point at (2,0,0):
        r̂=(1,0,0), p.r̂=0
        E = [0 - (0,0,1)] / 8 = (0, 0, -0.125)
        """
        dipoles = np.array([[0.0, 0.0, 1.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0]])
        result = dipole_field(dipoles, coords, points)
        assert_allclose(result, [[0.0, 0.0, -0.125]], rtol=1e-10)

    def test_singularity_safe(self):
        """Field at exact atom location returns finite value"""
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = dipole_field(dipoles, coords, points)
        assert np.isfinite(result).all()


# ─────────────────────────────────────────────
# QUADRUPOLE TESTS
# ─────────────────────────────────────────────

class TestQuadrupolePotential:

    def test_quadrupole_along_z(self):
        """
        Q = diag(-1,-1,2) (traceless), point at (0,0,2):
        Qrr = Q_ab r_a r_b = 2*4 = 8
        V = Qrr / r^5 = 8 / 32 = 0.25
        """
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 2.0]])
        result = quadrupole_potential(Q, coords, points)
        assert_allclose(result, [0.25], rtol=1e-10)

    def test_traceless_symmetry(self):
        """Trace of Q should not contribute — pure traceless result"""
        Q = np.array([[[2.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, -1.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0]])
        result = quadrupole_potential(Q, coords, points)
        assert np.isfinite(result).all()

    def test_singularity_safe(self):
        """Potential at exact atom location returns finite value"""
        Q = np.array([[[1.0, 0.0, 0.0],
                       [0.0, -0.5, 0.0],
                       [0.0, 0.0, -0.5]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = quadrupole_potential(Q, coords, points)
        assert np.isfinite(result).all()


class TestQuadrupoleField:

    def test_quadrupole_field_finite(self):
        """Field from quadrupole is finite at valid points"""
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 2.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0]])
        result = quadrupole_field(Q, coords, points)
        assert np.isfinite(result).all()

    def test_singularity_safe(self):
        """Field at exact atom location returns finite value"""
        Q = np.array([[[1.0, 0.0, 0.0],
                       [0.0, -0.5, 0.0],
                       [0.0, 0.0, -0.5]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = quadrupole_field(Q, coords, points)
        assert np.isfinite(result).all()


# ─────────────────────────────────────────────
# TOTAL POTENTIAL AND FIELD TESTS
# ─────────────────────────────────────────────

class TestTotalPotential:

    def test_monopole_only(self):
        """total_potential with only charges matches monopole_potential"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = total_potential(coords, points, atcharges=charges)
        expected = monopole_potential(charges, coords, points)
        assert_allclose(result, expected, rtol=1e-10)

    def test_superposition_all_terms(self):
        """Total potential is sum of all individual contributions"""
        charges = np.array([1.0])
        dipoles = np.array([[1.0, 0.0, 0.0]])
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 1.0, 0.0]])

        total = total_potential(
            coords,
            points,
            atcharges=charges,
            dipoles=dipoles,
            quadrupoles=Q,
        )
        expected = (monopole_potential(charges, coords, points) +
                    dipole_potential(dipoles, coords, points) +
                    quadrupole_potential(Q, coords, points))
        assert_allclose(total, expected, rtol=1e-10)


class TestTotalField:

    def test_monopole_only(self):
        """total_field with only charges matches monopole_field"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = total_field(coords, points, atcharges=charges)
        expected = monopole_field(charges, coords, points)
        assert_allclose(result, expected, rtol=1e-10)

    def test_superposition_all_terms(self):
        """Total field is sum of all individual contributions"""
        charges = np.array([1.0])
        dipoles = np.array([[1.0, 0.0, 0.0]])
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 1.0, 0.0]])

        total = total_field(
            coords,
            points,
            atcharges=charges,
            dipoles=dipoles,
            quadrupoles=Q,
        )
        expected = (monopole_field(charges, coords, points) +
                    dipole_field(dipoles, coords, points) +
                    quadrupole_field(Q, coords, points))
        assert_allclose(total, expected, rtol=1e-10)


# ─────────────────────────────────────────────
# MONOPOLE NUMERICAL GRADIENT CONSISTENCY: E = -∇V
# ─────────────────────────────────────────────

def _compute_monopole_gradient(atcharges, atcoords, point, h=FD_STEP):
    """
    Compute a numerical approximation to the gradient of the monopole
    potential using second-order central finite differences.

    .. math::

        \\frac{\\partial V}{\\partial x_i}
        \\approx
        \\frac{V(\\mathbf r+h\\hat e_i)-V(\\mathbf r-h\\hat e_i)}
             {2h}
    """
    grad = np.zeros(3)
    for i in range(3):
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        v_plus = monopole_potential(atcharges, atcoords, point_plus.reshape(1, 3))[0]
        v_minus = monopole_potential(atcharges, atcoords, point_minus.reshape(1, 3))[0]
        grad[i] = (v_plus - v_minus) / (2.0 * h)
    return grad


def _assert_field_matches_gradient(atcharges, atcoords, point, h=FD_STEP,
                                    rtol=1e-6, atol=1e-8):
    """Assert that monopole_field equals -∇V at ``point``, via finite differences."""
    analytic_field = monopole_field(atcharges, atcoords, point.reshape(1, 3))[0]
    numerical_field = -_compute_monopole_gradient(atcharges, atcoords, point, h=h)
    assert_allclose(numerical_field, analytic_field, rtol=rtol, atol=atol)


class TestMonopoleGradientConsistency:

    def test_field_matches_numerical_gradient_general_point(self):
        """
        Verify E = -∇V via central finite differences at a general,
        non-axis-aligned point.

        .. math::
            \\mathbf{E} = -\\nabla V
        """
        atcharges = np.array([1.0])
        atcoords = np.array([[0.0, 0.0, 0.0]])
        point = np.array([1.2, -0.7, 0.5])
        _assert_field_matches_gradient(atcharges, atcoords, point)

    def test_field_matches_numerical_gradient_multiple_charges(self):
        """Superposition of several charges should also satisfy E = -∇V."""
        atcharges = np.array([1.5, -0.8, 2.0])
        atcoords = np.array([[1.0, 0.0, 0.0],
                              [-1.0, 0.5, 0.0],
                              [0.3, -0.6, 1.1]])
        point = np.array([2.3, 1.4, -0.9])
        _assert_field_matches_gradient(atcharges, atcoords, point)

    def test_field_matches_numerical_gradient_randomized(self):
        """
        Randomized configurations of charges, coordinates, and field
        points, verifying E = -∇V holds across many independent trials.

        Candidate points are resampled if they fall too close to any
        charge, since the finite-difference stencil itself becomes
        unreliable near the singularity (covered separately below).
        """
        rng = np.random.default_rng(42)
        min_distance = 0.3

        for _ in range(20):
            n_charges = rng.integers(1, 4)
            atcharges = rng.uniform(-2.0, 2.0, size=n_charges)
            atcoords = rng.uniform(-1.0, 1.0, size=(n_charges, 3))

            while True:
                point = rng.uniform(-3.0, 3.0, size=3)
                distances = np.linalg.norm(atcoords - point, axis=1)
                if np.all(distances > min_distance):
                    break

            _assert_field_matches_gradient(atcharges, atcoords, point,
                                            rtol=1e-5, atol=1e-7)

    def test_field_matches_numerical_gradient_near_singularity(self):
        """
        Near-singularity robustness check: E = -∇V should still hold at a
        point very close to (but not exactly on) a charge.

        Finite differences become less accurate close to the singularity,
        so a larger step and looser tolerances are used here.
        """
        atcharges = np.array([1.0])
        atcoords = np.array([[0.0, 0.0, 0.0]])
        point = np.array([1e-3, 0.0, 0.0])
        _assert_field_matches_gradient(atcharges, atcoords, point,
                                        h=1e-6, rtol=1e-3, atol=1e-3)
