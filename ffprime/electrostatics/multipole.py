"""
Deprecated compatibility shim.

This module has been superseded by :mod:`ffprime.electrostatics.cartesian`
as part of splitting the package into dedicated Cartesian and spherical
modules (see ``cartesian.py``, ``spherical.py``, ``utils.py``). It is kept
in place purely so that pre-existing imports of the form

    from ffprime.electrostatics.multipole import monopole_potential

continue to work unmodified. New code should import from
:mod:`ffprime.electrostatics.cartesian` instead. This shim may be removed
in a future release once downstream call sites have migrated.
"""

from ffprime.electrostatics.cartesian import (  # noqa: F401
    monopole_potential,
    monopole_field,
    dipole_potential,
    dipole_field,
    quadrupole_potential,
    quadrupole_field,
    total_potential,
    total_field,
)
