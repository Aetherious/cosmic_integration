"""Enumerations for stellar types used in binary populations.

The original COMPAS code defines a large enumeration of stellar
evolutionary stages.  In the binned integrator we only need to
identify black holes (BH), neutron stars (NS) and white dwarfs (WD)
when classifying double compact object types.  The full enumeration
is provided for completeness.
"""

from __future__ import annotations

from enum import Enum, auto


class StellarType(Enum):
    """Enumeration of stellar types.

    The values correspond to those used by COMPAS.  Only a subset
    (black holes, neutron stars and white dwarfs) is used by the
    binned integrator.
    """

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
    MASSLESS_REMNANT = auto()
    CHEMICALLY_HOMOGENEOUS = auto()
    STAR = auto()
    BINARY_STAR = auto()
    NONE = auto()


# Convenience aliases for compact object categories
BH = [StellarType.BLACK_HOLE]
NS = [StellarType.NEUTRON_STAR]
WD = [
    StellarType.HELIUM_WHITE_DWARF,
    StellarType.CARBON_OXYGEN_WHITE_DWARF,
    StellarType.OXYGEN_NEON_WHITE_DWARF,
]
