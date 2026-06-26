"""
Spherical multipole conversions and electrostatic potential functions.

Conversion between Cartesian and real spherical (Stone convention) forms
for dipole and quadrupole moments, plus routines to evaluate the
electrostatic potential from collections of spherical multipoles.

Stone convention:

Dipole components:

    Q_10  = pz
    Q_11c = px
    Q_11s = py

Quadrupole components:

    Q_20  = Θzz
    Q_21c = Θxz
    Q_21s = Θyz
    Q_22c = (Θxx - Θyy) / 2
    Q_22s = Θxy

References
----------
Stone, A.J. "The Theory of Intermolecular Forces",
Oxford University Press.
"""
import numpy as np

from ffprime.electrostatics.multipole import quadrupole_field


def _validate_shapes(coords, points):
    """Validate coordinate and point array shapes."""
    coords = np.asarray(coords)
    points = np.asarray(points)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords must have shape (N, 3), got {coords.shape}"
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (M, 3), got {points.shape}"
        )


def dipole_cartesian_to_spherical(p: np.ndarray) -> np.ndarray:
    """
    Convert a Cartesian dipole moment to real spherical form.

    Uses the Stone convention:

        Q_10  = pz
        Q_11c = px
        Q_11s = py
    """
    p = np.asarray(p)

    if p.shape != (3,):
        raise ValueError(
            f"dipole must have shape (3,), got {p.shape}"
        )

    px, py, pz = p
    return np.array([pz, px, py])


def dipole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """
    Convert real spherical dipole components to Cartesian form.

    Input ordering:

        [Q_10, Q_11c, Q_11s]
    """
    q = np.asarray(q)

    if q.shape != (3,):
        raise ValueError(
            f"dipole must have shape (3,), got {q.shape}"
        )

    Q_10, Q_11c, Q_11s = q
    return np.array([Q_11c, Q_11s, Q_10])


def spherical_dipole_potential(dipoles, coords, points):
    """
    Compute electrostatic potential directly from spherical dipole moments.

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Spherical dipole components [Q_10, Q_11c, Q_11s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    potential : np.ndarray, shape (M,)
    """
    dipoles = np.asarray(dipoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    _validate_shapes(coords, points)

    if len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)

    x = r_vecs[:, :, 0]
    y = r_vecs[:, :, 1]
    z = r_vecs[:, :, 2]

    Q10 = dipoles[:, 0]
    Q11c = dipoles[:, 1]
    Q11s = dipoles[:, 2]

    p_dot_r = (
        Q11c[np.newaxis, :] * x
        + Q11s[np.newaxis, :] * y
        + Q10[np.newaxis, :] * z
    )

    return np.sum(p_dot_r / safe_r**3, axis=1)


def spherical_monopole_field(charges, coords, points):
    """
    Compute electric field directly from point charges.

    Parameters
    ----------
    charges : np.ndarray, shape (N,)
        Point charges in atomic units.
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    charges = np.asarray(charges)
    coords = np.asarray(coords)
    points = np.asarray(points)

    _validate_shapes(coords, points)

    if len(charges) != len(coords):
        raise ValueError("charges and coords must have same length")

    if charges.ndim != 1:
        raise ValueError(
            f"charges must have shape (N,), got {charges.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)

    field = charges[np.newaxis, :, np.newaxis] * r_vecs / safe_r[:, :, np.newaxis] ** 3
    return np.sum(field, axis=1)


def spherical_dipole_field(dipoles, coords, points):
    """
    Compute electric field directly from spherical dipole moments.

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Spherical dipole components [Q_10, Q_11c, Q_11s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    dipoles = np.asarray(dipoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    _validate_shapes(coords, points)

    if len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)
    r_hat = r_vecs / safe_r[:, :, np.newaxis]

    Q10 = dipoles[:, 0]
    Q11c = dipoles[:, 1]
    Q11s = dipoles[:, 2]

    # Reconstruct p.r_hat using spherical components:
    # p = (Q11c, Q11s, Q10) in Cartesian, so p.r_hat = Q11c*rx + Q11s*ry + Q10*rz
    rx = r_hat[:, :, 0]
    ry = r_hat[:, :, 1]
    rz = r_hat[:, :, 2]

    p_dot_rhat = (
        Q11c[np.newaxis, :] * rx
        + Q11s[np.newaxis, :] * ry
        + Q10[np.newaxis, :] * rz
    )

    # p vector broadcast to (M, N, 3)
    p = np.stack([
        Q11c[np.newaxis, :] * np.ones_like(rx),
        Q11s[np.newaxis, :] * np.ones_like(rx),
        Q10[np.newaxis, :] * np.ones_like(rx),
    ], axis=-1)

    term = (3 * p_dot_rhat[:, :, np.newaxis] * r_hat - p)
    return np.sum(term / safe_r[:, :, np.newaxis] ** 3, axis=1)


def spherical_quadrupole_potential(quadrupoles, coords, points):

    """
    Compute electrostatic potential directly from spherical quadrupole moments.

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 5)
        Spherical quadrupole components
        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    potential : np.ndarray, shape (M,)
    """
    quadrupoles = np.asarray(quadrupoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    _validate_shapes(coords, points)

    if len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    if quadrupoles.ndim != 2 or quadrupoles.shape[1] != 5:
        raise ValueError(
            f"quadrupoles must have shape (N, 5), got {quadrupoles.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]

    x = r_vecs[:, :, 0]
    y = r_vecs[:, :, 1]
    z = r_vecs[:, :, 2]

    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)

    Q20 = quadrupoles[:, 0]
    Q21c = quadrupoles[:, 1]
    Q21s = quadrupoles[:, 2]
    Q22c = quadrupoles[:, 3]
    Q22s = quadrupoles[:, 4]

    numerator = (
        Q20[np.newaxis, :] * (z**2 - 0.5 * (x**2 + y**2))
        + Q22c[np.newaxis, :] * (x**2 - y**2)
        + 2.0 * Q22s[np.newaxis, :] * x * y
        + 2.0 * Q21c[np.newaxis, :] * x * z
        + 2.0 * Q21s[np.newaxis, :] * y * z
    )

    return np.sum(numerator / safe_r**5, axis=1)


def spherical_quadrupole_field(quadrupoles, coords, points):
    """
    Compute electric field directly from spherical quadrupole moments.

    Reuses the same infrastructure as ``spherical_quadrupole_potential``:
    the spherical components are converted back to a Cartesian traceless
    quadrupole tensor via ``quadrupole_spherical_to_cartesian``, and the
    field is then evaluated using the existing Cartesian
    ``quadrupole_field`` routine (E = -grad V).

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 5)
        Spherical quadrupole components
        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    quadrupoles = np.asarray(quadrupoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    _validate_shapes(coords, points)

    if len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    if quadrupoles.ndim != 2 or quadrupoles.shape[1] != 5:
        raise ValueError(
            f"quadrupoles must have shape (N, 5), got {quadrupoles.shape}"
        )

    cartesian = np.array(
        [
            quadrupole_spherical_to_cartesian(q)
            for q in quadrupoles
        ]
    )

    return quadrupole_field(
        cartesian,
        coords,
        points,
    )


def quadrupole_cartesian_to_spherical(theta: np.ndarray) -> np.ndarray:
    """
    Convert a traceless Cartesian quadrupole tensor to real spherical form.

    Parameters
    ----------
    theta : np.ndarray, shape (3, 3)
        Symmetric traceless Cartesian quadrupole tensor.

    Returns
    -------
    np.ndarray, shape (5,)
        Real spherical components ordered as

        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s]

    Notes
    -----
    Stone convention:

    Q_20  = Θzz
    Q_21c = Θxz
    Q_21s = Θyz
    Q_22c = (Θxx - Θyy)/2
    Q_22s = Θxy
    """
    theta = np.asarray(theta)

    if theta.shape != (3, 3):
        raise ValueError(
            f"quadrupole tensor must have shape (3, 3), got {theta.shape}"
        )

    if not np.allclose(theta, theta.T, atol=1e-10):
        raise ValueError("quadrupole tensor must be symmetric")

    if not np.isclose(np.trace(theta), 0.0, atol=1e-10):
        raise ValueError(
            f"quadrupole tensor must be traceless, got trace = {np.trace(theta):.6e}"
        )

    Q_20 = theta[2, 2]
    Q_21c = theta[0, 2]
    Q_21s = theta[1, 2]
    Q_22c = (theta[0, 0] - theta[1, 1]) / 2.0
    Q_22s = theta[0, 1]

    return np.array([Q_20, Q_21c, Q_21s, Q_22c, Q_22s])


def quadrupole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """
    Convert real spherical quadrupole moments to a Cartesian tensor.

    Parameters
    ----------
    q : np.ndarray, shape (5,)
        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s]

    Returns
    -------
    np.ndarray, shape (3, 3)
        Symmetric traceless Cartesian quadrupole tensor.
    """
    q = np.asarray(q)

    if q.shape != (5,):
        raise ValueError(
            f"quadrupole must have shape (5,), got {q.shape}"
        )

    Q_20, Q_21c, Q_21s, Q_22c, Q_22s = q

    theta = np.zeros((3, 3))
    theta[2, 2] = Q_20
    theta[0, 0] = Q_22c - Q_20 / 2.0
    theta[1, 1] = -Q_22c - Q_20 / 2.0
    theta[0, 2] = theta[2, 0] = Q_21c
    theta[1, 2] = theta[2, 1] = Q_21s
    theta[0, 1] = theta[1, 0] = Q_22s

    return theta