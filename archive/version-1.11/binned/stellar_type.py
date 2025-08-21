"""Enumeration of stellar types used in the legacy binned code.

The enumeration values are carried over verbatim from the original
implementation to avoid breaking user code that references these
constants.  The meaning of each type is documented in the COMPAS
framework and is not reproduced here.  This enumeration is provided
solely for compatibility and the dasein rewrite does not depend on it.
"""

from __future__ import annotations

from enum import Enum, auto


class STELLAR_TYPE(Enum):
    MS_LTE_07 = 0
    MS_GT_07 = auto()
    HERTZSPRUNG_GAP = auto()
    FIRST_GIANT_BRANCH = auto()
    CORE_HELIUM_BURNING = auto()
    EARLY_ASYMPTOTIC_GIANT_BRANCH = auto()
    THERMALLY_PULSING_ASYMPTOTIC_GIANT_BRANCH = auto()
    NAKED_HELIUM_STAR_MS = auto()
    NAKED_HELIUM_STAR_HERTZSPRUNG_GAP = auto()
    NAKED_HELIUM_STAR_GIANT_BRANCH = auto()
    HELIUM_WHITE_DWARF = auto()
    CARBON_OXYGEN_WHITE_DWARF = auto()
    OXYGEN_NEON_WHITE_DWARF = auto()
    NEUTRON_STAR = auto()
    BLACK_HOLE = auto()
    NAKED_HELIUM_STAR_EARLY_ASYMPTOTIC_GIANT_BRANCH = auto()
    HYBRID_WHITE_DWARF = auto()
    NAKED_HELIUM_STAR_THERMALLY_PULSING_ASYMPTOTIC_GIANT_BRANCH = auto()
    PROTO_NEUTRON_STAR = auto()
    UNKNOWN = auto()


class BH(STELLAR_TYPE):
    """Alias for black hole type."""
    pass


class NS(STELLAR_TYPE):
    """Alias for neutron star type."""
    pass


class WD(STELLAR_TYPE):
    """Alias for white dwarf types."""
    pass