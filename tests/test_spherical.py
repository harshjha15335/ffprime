import numpy as np
import pytest
from ffprime.electrostatics.spherical import (
    dipole_cartesian_to_spherical,
    dipole_spherical_to_cartesian,
    quadrupole_cartesian_to_spherical,
    quadrupole_spherical_to_cartesian,
)
from ffprime.electrostatics.multipole import dipole_potential
from ffprime.electrostatics.spherical import spherical_dipole_potential
from ffprime.electrostatics.multipole import quadrupole_potential
from ffprime.electrostatics.spherical import spherical_quadrupole_potential


# ---------------------------------------------------------------------------
# Dipole
# ---------------------------------------------------------------------------

def test_dipole_z_aligned():
    """Pure z-dipole maps entirely to Q_10, others zero."""
    p = np.array([0.0, 0.0, 1.0])
    sph = dipole_cartesian_to_spherical(p)
    assert np.isclose(sph[0], 1.0)
    assert np.isclose(sph[1], 0.0)
    assert np.isclose(sph[2], 0.0)


def test_dipole_roundtrip():
    """Cartesian -> spherical -> Cartesian recovers original."""
    p = np.array([1.5, -2.0, 3.0])
    assert np.allclose(
        dipole_spherical_to_cartesian(dipole_cartesian_to_spherical(p)), p
    )


def test_dipole_potential_consistency():
    q = np.array([1.0, 2.0, 3.0])

    cart = dipole_spherical_to_cartesian(q)

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    v_cart = dipole_potential(
        np.array([cart]),
        coords,
        points,
    )

    v_sph = spherical_dipole_potential(
        np.array([q]),
        coords,
        points,
    )

    assert np.allclose(v_cart, v_sph)


def test_dipole_potential_along_z():
    """
    Pure Q10 dipole evaluated on the z-axis.

    Q10 = 1, r = (0, 0, 2)

    V = z / r³
      = 2 / 8
      = 0.25
    """
    dipoles = np.array([[1.0, 0.0, 0.0]])

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    result = spherical_dipole_potential(
        dipoles,
        coords,
        points,
    )

    assert np.allclose(result, [0.25])


# ---------------------------------------------------------------------------
# Quadrupole
# ---------------------------------------------------------------------------

def test_quadrupole_traceless_check():
    """Non-traceless tensor raises ValueError."""
    bad_tensor = np.eye(3)
    with pytest.raises(ValueError, match="traceless"):
        quadrupole_cartesian_to_spherical(bad_tensor)


def test_quadrupole_z_axial():
    """Axially symmetric quadrupole along z has only Q_20 nonzero."""
    theta = np.array([
        [-0.5,  0.0,  0.0],
        [ 0.0, -0.5,  0.0],
        [ 0.0,  0.0,  1.0]
    ])
    sph = quadrupole_cartesian_to_spherical(theta)
    assert np.isclose(sph[0], 1.0)
    assert np.allclose(sph[1:], 0.0)


def test_quadrupole_roundtrip():
    """Cartesian -> spherical -> Cartesian recovers original tensor."""
    theta = np.array([
        [-0.5,  0.3,  0.1],
        [ 0.3, -0.5,  0.2],
        [ 0.1,  0.2,  1.0]
    ])
    recovered = quadrupole_spherical_to_cartesian(
        quadrupole_cartesian_to_spherical(theta)
    )
    assert np.allclose(recovered, theta)


def test_quadrupole_potential_consistency():
    q = np.array([1.0, 0.3, -0.2, 0.4, 0.1])

    theta = quadrupole_spherical_to_cartesian(q)

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    v_cart = quadrupole_potential(
        np.array([theta]),
        coords,
        points,
    )

    v_sph = spherical_quadrupole_potential(
        np.array([q]),
        coords,
        points,
    )

    assert np.allclose(v_cart, v_sph)


def test_quadrupole_potential_along_z():
    """
    Pure Q20 quadrupole evaluated on the z-axis.

    Q20 = 1, r = (0, 0, 2)

    V = (z² - 0.5(x² + y²)) / r⁵
      = 4 / 32
      = 0.125
    """
    quadrupoles = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    result = spherical_quadrupole_potential(
        quadrupoles,
        coords,
        points,
    )

    assert np.allclose(result, [0.125])