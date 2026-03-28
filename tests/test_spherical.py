import numpy as np
import pytest
from ffprime.electrostatics.spherical import (
    dipole_cartesian_to_spherical,
    dipole_spherical_to_cartesian,
    quadrupole_cartesian_to_spherical,
    quadrupole_spherical_to_cartesian,
)


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