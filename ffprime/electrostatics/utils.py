"""
Shared helpers for the electrostatics package.

``cartesian.py`` and ``spherical.py`` both need to (a) validate the shapes
of the coordinate/point arrays they're given, and (b) compute pairwise
displacement vectors and singularity-safe distances between multipole
sites and field points. Both pieces of logic were previously duplicated
verbatim in each module; they now live here as the single source of
truth.

Nothing in this module performs a physics calculation on its own -- it
only prepares/validates the inputs that ``cartesian.py`` and
``spherical.py`` build their formulas on top of.
"""

import numpy as np


def validate_shapes(coords, points):
    """Validate coordinate and point array shapes.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Multipole-site coordinates.
    points : np.ndarray, shape (M, 3)
        Field points at which a potential/field will be evaluated.

    Raises
    ------
    ValueError
        If ``coords`` or ``points`` does not have shape ``(*, 3)``.
    """
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


def compute_displacement(coords, points):
    """Compute pairwise displacement vectors and singularity-safe distances.

    This is the ``r_vecs`` / ``r`` / ``safe_r`` block that every potential
    and field routine in this package starts with:

        r_vecs = points[:, None, :] - coords[None, :, :]
        r      = ||r_vecs||
        safe_r = r, with any distance below 1e-12 replaced by inf

    Replacing near-zero distances with ``inf`` keeps evaluation at (or
    extremely close to) a multipole's own site finite, rather than
    raising a division-by-zero warning.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    r_vecs : np.ndarray, shape (M, N, 3)
        Displacement vectors ``points[m] - coords[n]``.
    r : np.ndarray, shape (M, N)
        Euclidean distances corresponding to ``r_vecs``.
    safe_r : np.ndarray, shape (M, N)
        ``r`` with near-zero entries replaced by ``inf``.
    """
    coords = np.asarray(coords)
    points = np.asarray(points)

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)

    return r_vecs, r, safe_r