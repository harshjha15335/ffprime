"""
Electrostatic potential and electric field from atom-centered multipoles,
expressed in standard Cartesian form.

Implements monopole, dipole, and quadrupole contributions following the
multipole expansion of the electrostatic potential in atomic units.

This module contains *only* Cartesian-representation routines. Real
spherical harmonic (Stone-convention) multipoles, their conversions
to/from Cartesian form, and the corresponding spherical potential/field
routines live in :mod:`ffprime.electrostatics.spherical`. Shared
validation and geometry helpers live in
:mod:`ffprime.electrostatics.utils`.

References
----------
Stone, A.J. "The Theory of Intermolecular Forces", Oxford University Press.
"""

import numpy as np

from ffprime.electrostatics.utils import compute_displacement, validate_shapes


def monopole_potential(atcharges, atcoords, points):
    """
    Compute electrostatic potential from atomic monopoles (charges).

    .. math::

        V(\\mathbf{r}) = \\sum_i \\frac{q_i}{|\\mathbf{r} - \\mathbf{r}_i|}

    Parameters
    ----------
    atcharges : np.ndarray, shape (N,)
        Atomic charges in atomic units.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the potential.

    Returns
    -------
    potential : np.ndarray, shape (M,)
        Electrostatic potential at each field point in atomic units.
    """
    atcharges = np.asarray(atcharges)
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if len(atcharges) != len(atcoords):
        raise ValueError("atcharges and atcoords must have same length")

    _, _, safe_r = compute_displacement(atcoords, points)
    return np.sum(atcharges[np.newaxis, :] / safe_r, axis=1)


def monopole_field(atcharges, atcoords, points):
    """
    Compute electric field from atomic monopoles (charges).

    .. math::

        \\mathbf{E}(\\mathbf{r}) =
        \\sum_i q_i \\frac{\\mathbf{r} - \\mathbf{r}_i}{|\\mathbf{r} - \\mathbf{r}_i|^3}

    Parameters
    ----------
    atcharges : np.ndarray, shape (N,)
        Atomic charges in atomic units.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the field.

    Returns
    -------
    field : np.ndarray, shape (M, 3)
        Electric field at each field point in atomic units.
    """
    atcharges = np.asarray(atcharges)
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if len(atcharges) != len(atcoords):
        raise ValueError("atcharges and atcoords must have same length")

    r_vecs, _, safe_r = compute_displacement(atcoords, points)
    field = atcharges[np.newaxis, :, np.newaxis] * r_vecs / safe_r[:, :, np.newaxis] ** 3
    return np.sum(field, axis=1)


def dipole_potential(dipoles, atcoords, points):
    """
    Compute electrostatic potential from atomic dipoles.

    .. math::

        V(\\mathbf{r}) =
        \\sum_i \\frac{\\mathbf{p}_i \\cdot (\\mathbf{r} - \\mathbf{r}_i)}{|\\mathbf{r} - \\mathbf{r}_i|^3}

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Atomic dipole moment vectors in atomic units.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the potential.

    Returns
    -------
    potential : np.ndarray, shape (M,)
        Electrostatic potential at each field point in atomic units.
    """
    dipoles = np.asarray(dipoles)
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if len(dipoles) != len(atcoords):
        raise ValueError("dipoles and atcoords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    r_vecs, _, safe_r = compute_displacement(atcoords, points)
    p_dot_r = np.einsum("ij,mij->mi", dipoles, r_vecs)
    return np.sum(p_dot_r / safe_r ** 3, axis=1)


def dipole_field(dipoles, atcoords, points):
    """
    Compute electric field from atomic dipoles.

    .. math::

        \\mathbf{E}(\\mathbf{r}) =
        \\sum_i \\frac{3(\\mathbf{p}_i \\cdot \\hat{\\mathbf{r}})\\hat{\\mathbf{r}} - \\mathbf{p}_i}{|\\mathbf{r} - \\mathbf{r}_i|^3}

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Atomic dipole moment vectors in atomic units.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the field.

    Returns
    -------
    field : np.ndarray, shape (M, 3)
        Electric field at each field point in atomic units.
    """
    dipoles = np.asarray(dipoles)
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if len(dipoles) != len(atcoords):
        raise ValueError("dipoles and atcoords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    r_vecs, _, safe_r = compute_displacement(atcoords, points)
    r_hat = r_vecs / safe_r[:, :, np.newaxis]
    p_dot_rhat = np.einsum("ij,mij->mi", dipoles, r_hat)
    term = (3 * p_dot_rhat[:, :, np.newaxis] * r_hat
            - dipoles[np.newaxis, :, :])
    return np.sum(term / safe_r[:, :, np.newaxis] ** 3, axis=1)


def quadrupole_potential(quadrupoles, atcoords, points):
    """
    Compute electrostatic potential from atomic quadrupoles (traceless).

    .. math::

        V(\\mathbf{r}) = \\sum_i \\frac{\\mathbf{Q}_i : \\mathbf{r}\\mathbf{r}}{|\\mathbf{r} - \\mathbf{r}_i|^5}

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 3, 3)
        Traceless quadrupole moment tensors in atomic units.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the potential.

    Returns
    -------
    potential : np.ndarray, shape (M,)
        Electrostatic potential at each field point in atomic units.
    """
    quadrupoles = np.asarray(quadrupoles)
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if len(quadrupoles) != len(atcoords):
        raise ValueError("quadrupoles and atcoords must have same length")

    if quadrupoles.ndim != 3 or quadrupoles.shape[1:] != (3, 3):
        raise ValueError(
            f"quadrupoles must have shape (N, 3, 3), got {quadrupoles.shape}"
        )

    r_vecs, _, safe_r = compute_displacement(atcoords, points)
    Qrr = np.einsum("nab,mna,mnb->mn", quadrupoles, r_vecs, r_vecs)
    return np.sum(Qrr / (safe_r ** 5), axis=1)


def quadrupole_field(quadrupoles, atcoords, points):
    """
    Compute electric field from atomic quadrupoles (traceless).

    .. math::

        \\mathbf{E}(\\mathbf{r}) =
        \\sum_i \\left[ \\frac{5(\\mathbf{Q}_i : \\mathbf{r}\\mathbf{r})\\,\\mathbf{r}}{r^7} - \\frac{\\mathbf{Q}_i \\cdot \\mathbf{r}}{r^5} \\right]

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 3, 3)
        Traceless quadrupole moment tensors in atomic units.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the field.

    Returns
    -------
    field : np.ndarray, shape (M, 3)
        Electric field at each field point in atomic units.
    """
    quadrupoles = np.asarray(quadrupoles)
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if len(quadrupoles) != len(atcoords):
        raise ValueError("quadrupoles and atcoords must have same length")

    if quadrupoles.ndim != 3 or quadrupoles.shape[1:] != (3, 3):
        raise ValueError(
            f"quadrupoles must have shape (N, 3, 3), got {quadrupoles.shape}"
        )

    r_vecs, _, safe_r = compute_displacement(atcoords, points)
    Qrr = np.einsum("nab,mna,mnb->mn", quadrupoles, r_vecs, r_vecs)
    Qr = np.einsum("nab,mnb->mna", quadrupoles, r_vecs)
    term1 = (5 * Qrr[:, :, np.newaxis] * r_vecs
             / (safe_r[:, :, np.newaxis] ** 7))
    term2 = Qr / safe_r[:, :, np.newaxis] ** 5
    return np.sum(term1 - term2, axis=1)


def total_potential(atcoords, points, atcharges=None, dipoles=None, quadrupoles=None):
    """
    Compute total electrostatic potential from all multipole contributions.

    Parameters
    ----------
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the potential.
    atcharges : np.ndarray, shape (N,), optional
        Atomic charges in atomic units.
    dipoles : np.ndarray, shape (N, 3), optional
        Atomic dipole moment vectors in atomic units.
    quadrupoles : np.ndarray, shape (N, 3, 3), optional
        Traceless quadrupole moment tensors in atomic units.

    Returns
    -------
    potential : np.ndarray, shape (M,)
        Electrostatic potential at each field point in atomic units.
    """
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if atcharges is not None and len(atcharges) != len(atcoords):
        raise ValueError("atcharges and atcoords must have same length")

    if dipoles is not None and len(dipoles) != len(atcoords):
        raise ValueError("dipoles and atcoords must have same length")

    if quadrupoles is not None and len(quadrupoles) != len(atcoords):
        raise ValueError("quadrupoles and atcoords must have same length")

    potential = np.zeros(points.shape[0])
    if atcharges is not None:
        potential += monopole_potential(atcharges, atcoords, points)
    if dipoles is not None:
        potential += dipole_potential(dipoles, atcoords, points)
    if quadrupoles is not None:
        potential += quadrupole_potential(quadrupoles, atcoords, points)
    return potential


def total_field(atcoords, points, atcharges=None, dipoles=None, quadrupoles=None):
    """
    Compute total electric field from all multipole contributions.

    Parameters
    ----------
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the field.
    atcharges : np.ndarray, shape (N,), optional
        Atomic charges in atomic units.
    dipoles : np.ndarray, shape (N, 3), optional
        Atomic dipole moment vectors in atomic units.
    quadrupoles : np.ndarray, shape (N, 3, 3), optional
        Traceless quadrupole moment tensors in atomic units.

    Returns
    -------
    field : np.ndarray, shape (M, 3)
        Electric field at each field point in atomic units.
    """
    atcoords = np.asarray(atcoords)
    points = np.asarray(points)
    validate_shapes(atcoords, points)

    if atcharges is not None and len(atcharges) != len(atcoords):
        raise ValueError("atcharges and atcoords must have same length")

    if dipoles is not None and len(dipoles) != len(atcoords):
        raise ValueError("dipoles and atcoords must have same length")

    if quadrupoles is not None and len(quadrupoles) != len(atcoords):
        raise ValueError("quadrupoles and atcoords must have same length")

    field = np.zeros((points.shape[0], 3))
    if atcharges is not None:
        field += monopole_field(atcharges, atcoords, points)
    if dipoles is not None:
        field += dipole_field(dipoles, atcoords, points)
    if quadrupoles is not None:
        field += quadrupole_field(quadrupoles, atcoords, points)
    return field
