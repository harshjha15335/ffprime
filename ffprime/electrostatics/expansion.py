"""
Container class for atom-centered multipole expansions.

This module defines :class:`MultipoleExpansion`, a small container that
holds the per-atom multipole data (charges, dipoles, quadrupoles) for a
single molecule or system, along with the representation in which those
moments are stored. The container does not yet evaluate any potential
or field -- evaluation is delegated to the existing routines in
:mod:`ffprime.electrostatics.cartesian` and
:mod:`ffprime.electrostatics.spherical`.
"""

import numpy as np

__all__ = ["MultipoleExpansion"]


_VALID_REPRESENTATIONS = ("cartesian", "spherical")


class MultipoleExpansion:
    """Container for an atom-centered multipole expansion.

    Parameters
    ----------
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates of the multipole sites, in atomic units.
    atcharges : np.ndarray, shape (N,), optional
        Atomic monopole (charge) moments, in atomic units.
    dipoles : np.ndarray, optional
        Dipole moments. Shape depends on ``representation``: ``(N, 3)``
        for both ``"cartesian"`` and ``"spherical"``.
    quadrupoles : np.ndarray, optional
        Quadrupole moments. Shape depends on ``representation``:
        ``(N, 3, 3)`` for ``"cartesian"`` (symmetric traceless tensor)
        and ``(N, 5)`` for ``"spherical"`` (real spherical components).
    representation : str, optional
        Either ``"cartesian"`` or ``"spherical"``. Defaults to
        ``"cartesian"``.

    Attributes
    ----------
    representation : str
        Representation in which ``dipoles`` and ``quadrupoles`` are
        stored. One of ``"cartesian"`` or ``"spherical"``.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates of the multipole sites.
    atcharges : np.ndarray, shape (N,) or None
        Atomic monopole (charge) moments.
    dipoles : np.ndarray or None
        Dipole moments, in the stored representation.
    quadrupoles : np.ndarray or None
        Quadrupole moments, in the stored representation.

    Raises
    ------
    TypeError
        If ``atcoords`` is ``None``.
    ValueError
        If ``representation`` is not one of ``"cartesian"`` or
        ``"spherical"``.
    """

    def __init__(
        self,
        atcoords: np.ndarray,
        atcharges: np.ndarray | None = None,
        dipoles: np.ndarray | None = None,
        quadrupoles: np.ndarray | None = None,
        representation: str = "cartesian",
    ) -> None:
        """Store an atom-centered multipole expansion in the chosen form."""
        if atcoords is None:
            raise TypeError("atcoords cannot be None")

        if representation not in _VALID_REPRESENTATIONS:
            raise ValueError(
                "representation must be one of "
                f"{_VALID_REPRESENTATIONS}, got {representation!r}"
            )

        self.representation: str = representation
        self.atcoords: np.ndarray = np.asarray(atcoords)
        self.atcharges: np.ndarray | None = (
            None if atcharges is None else np.asarray(atcharges)
        )
        self.dipoles: np.ndarray | None = (
            None if dipoles is None else np.asarray(dipoles)
        )
        self.quadrupoles: np.ndarray | None = (
            None if quadrupoles is None else np.asarray(quadrupoles)
        )

    @classmethod
    def from_cartesian(
        cls,
        atcoords: np.ndarray,
        atcharges: np.ndarray | None = None,
        dipoles: np.ndarray | None = None,
        quadrupoles: np.ndarray | None = None,
    ) -> "MultipoleExpansion":
        """Build a :class:`MultipoleExpansion` from Cartesian multipoles.

        Parameters
        ----------
        atcoords : np.ndarray, shape (N, 3)
            Atomic coordinates of the multipole sites, in atomic units.
        atcharges : np.ndarray, shape (N,), optional
            Atomic monopole (charge) moments, in atomic units.
        dipoles : np.ndarray, shape (N, 3), optional
            Cartesian dipole moment vectors, in atomic units.
        quadrupoles : np.ndarray, shape (N, 3, 3), optional
            Symmetric traceless Cartesian quadrupole moment tensors, in
            atomic units.

        Returns
        -------
        MultipoleExpansion
            Container storing the moments in ``"cartesian"``
            representation.
        """
        return cls(
            atcoords=atcoords,
            atcharges=atcharges,
            dipoles=dipoles,
            quadrupoles=quadrupoles,
            representation="cartesian",
        )

    @classmethod
    def from_spherical(
        cls,
        atcoords: np.ndarray,
        atcharges: np.ndarray | None = None,
        dipoles: np.ndarray | None = None,
        quadrupoles: np.ndarray | None = None,
    ) -> "MultipoleExpansion":
        """Build a :class:`MultipoleExpansion` from real spherical multipoles.

        Parameters
        ----------
        atcoords : np.ndarray, shape (N, 3)
            Atomic coordinates of the multipole sites, in atomic units.
        atcharges : np.ndarray, shape (N,), optional
            Atomic monopole (charge) moments, in atomic units. Charges
            have no orientation and are identical in both
            representations.
        dipoles : np.ndarray, shape (N, 3), optional
            Real spherical dipole components (Stone convention).
        quadrupoles : np.ndarray, shape (N, 5), optional
            Real spherical quadrupole components (Stone convention).

        Returns
        -------
        MultipoleExpansion
            Container storing the moments in ``"spherical"``
            representation.
        """
        return cls(
            atcoords=atcoords,
            atcharges=atcharges,
            dipoles=dipoles,
            quadrupoles=quadrupoles,
            representation="spherical",
        )
