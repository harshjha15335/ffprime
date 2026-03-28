import numpy as np


def dipole_cartesian_to_spherical(p: np.ndarray) -> np.ndarray:
    """
    Convert a Cartesian dipole moment to real spherical (solid harmonic) form.

    Parameters
    ----------
    p : np.ndarray, shape (3,)
        Cartesian dipole moment [px, py, pz] in atomic units (e * bohr).

    Returns
    -------
    np.ndarray, shape (3,)
        Spherical components [Q_10, Q_11c, Q_11s].

    Notes
    -----
    Convention follows Stone, A.J. "Theory of Intermolecular Forces", 2nd ed.
    Q_10 = pz,  Q_11c = px,  Q_11s = py
    """
    px, py, pz = p
    return np.array([pz, px, py])


def dipole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """
    Convert real spherical dipole components back to Cartesian form.

    Parameters
    ----------
    q : np.ndarray, shape (3,)
        Spherical components [Q_10, Q_11c, Q_11s].

    Returns
    -------
    np.ndarray, shape (3,)
        Cartesian dipole moment [px, py, pz] in atomic units (e * bohr).
    """
    Q_10, Q_11c, Q_11s = q
    return np.array([Q_11c, Q_11s, Q_10])


def quadrupole_cartesian_to_spherical(theta: np.ndarray) -> np.ndarray:
    """
    Convert a traceless Cartesian quadrupole tensor to real spherical form.

    Parameters
    ----------
    theta : np.ndarray, shape (3, 3)
        Traceless symmetric quadrupole tensor in atomic units (e * bohr^2).
        Must satisfy np.trace(theta) == 0.

    Returns
    -------
    np.ndarray, shape (5,)
        Spherical components [Q_20, Q_21c, Q_21s, Q_22c, Q_22s].

    Raises
    ------
    ValueError
        If the input tensor is not traceless within numerical tolerance.

    Notes
    -----
    Convention follows Stone, A.J. "Theory of Intermolecular Forces", 2nd ed.
    Q_20  = Θzz
    Q_21c = Θxz
    Q_21s = Θyz
    Q_22c = (Θxx - Θyy) / 2
    Q_22s = Θxy
    """
    if not np.isclose(np.trace(theta), 0.0, atol=1e-10):
        raise ValueError(
            f"Quadrupole tensor must be traceless, got trace = {np.trace(theta):.6e}"
        )

    Q_20  = theta[2, 2]
    Q_21c = theta[0, 2]
    Q_21s = theta[1, 2]
    Q_22c = (theta[0, 0] - theta[1, 1]) / 2.0
    Q_22s = theta[0, 1]

    return np.array([Q_20, Q_21c, Q_21s, Q_22c, Q_22s])


def quadrupole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """
    Convert real spherical quadrupole components back to Cartesian tensor form.

    Parameters
    ----------
    q : np.ndarray, shape (5,)
        Spherical components [Q_20, Q_21c, Q_21s, Q_22c, Q_22s].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Traceless symmetric quadrupole tensor in atomic units (e * bohr^2).
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