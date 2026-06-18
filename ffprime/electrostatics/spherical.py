import numpy as np


def dipole_cartesian_to_spherical(p: np.ndarray) -> np.ndarray:
    """
    Convert a Cartesian dipole moment to real spherical (solid harmonic) form.
    """
    px, py, pz = p
    return np.array([pz, px, py])


def dipole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """
    Convert real spherical dipole components back to Cartesian form.
    """
    Q_10, Q_11c, Q_11s = q
    return np.array([Q_11c, Q_11s, Q_10])


def spherical_dipole_potential(
    dipoles: np.ndarray,
    coords: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Compute electrostatic potential directly from spherical dipole moments.

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Spherical dipole components [Q_10, Q_11c, Q_11s].
    coords : np.ndarray, shape (N, 3)
        Atomic coordinates.
    points : np.ndarray, shape (M, 3)
        Evaluation points.

    Returns
    -------
    np.ndarray, shape (M,)
        Electrostatic potential.
    """
    dipoles = np.asarray(dipoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]

    x = r_vecs[:, :, 0]
    y = r_vecs[:, :, 1]
    z = r_vecs[:, :, 2]

    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)

    Q10 = dipoles[:, 0]
    Q11c = dipoles[:, 1]
    Q11s = dipoles[:, 2]

    numerator = (
        Q11c[np.newaxis, :] * x
        + Q11s[np.newaxis, :] * y
        + Q10[np.newaxis, :] * z
    )

    return np.sum(numerator / safe_r**3, axis=1)


def quadrupole_cartesian_to_spherical(theta: np.ndarray) -> np.ndarray:
    """
    Convert a traceless Cartesian quadrupole tensor to real spherical form.
    """
    if not np.isclose(np.trace(theta), 0.0, atol=1e-10):
        raise ValueError(
            f"Quadrupole tensor must be traceless, got trace = {np.trace(theta):.6e}"
        )

    Q_20 = theta[2, 2]
    Q_21c = theta[0, 2]
    Q_21s = theta[1, 2]
    Q_22c = (theta[0, 0] - theta[1, 1]) / 2.0
    Q_22s = theta[0, 1]

    return np.array([Q_20, Q_21c, Q_21s, Q_22c, Q_22s])


def quadrupole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """
    Convert real spherical quadrupole components back to Cartesian tensor form.
    """
    Q_20, Q_21c, Q_21s, Q_22c, Q_22s = q

    theta = np.zeros((3, 3))
    theta[2, 2] = Q_20
    theta[0, 0] = Q_22c - Q_20 / 2.0
    theta[1, 1] = -Q_22c - Q_20 / 2.0
    theta[0, 2] = theta[2, 0] = Q_21c
    theta[1, 2] = theta[2, 1] = Q_21s
    theta[0, 1] = theta[1, 0] = Q_22s

    return theta