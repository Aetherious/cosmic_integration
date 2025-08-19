"""Functions related to the initial mass function and mass–evolved corrections.

This module consolidates helper functions for computing the fraction of
stellar mass sampled by COMPAS relative to the total star forming mass,
analytical integrals of the star formation history weighted by the
initial mass function, and retrieval of mass–evolved data from HDF5
files.  The implementations in this file mirror those used in the
original `totalMassEvolvedPerZ.py` but have been renamed for
clarity and improved readability.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, Tuple

import numpy as np


@lru_cache()
def imf_normalisation_constants(
    m1: float = 0.01,
    m2: float = 0.08,
    m3: float = 0.5,
    m4: float = 200.0,
    a12: float = 0.3,
    a23: float = 1.3,
    a34: float = 2.3,
) -> Tuple[float, float, float]:
    """Return constants ensuring continuity of the broken–power–law IMF.

    The Kroupa initial mass function (IMF) is a three–segment broken
    power law defined by slopes :math:`\alpha_{12}`, :math:`\alpha_{23}` and
    :math:`\alpha_{34}` between transition masses ``m1``, ``m2``, ``m3`` and
    ``m4``.  Continuity across the transitions is enforced via
    multiplicative constants ``b1``, ``b2`` and ``b3``.

    Returns
    -------
    b1, b2, b3:
        Normalisation constants for the low, intermediate and high mass
        segments, respectively.
    """
    b1 = 1.0 / (
        (m2 ** (1 - a12) - m1 ** (1 - a12)) / (1 - a12)
        + m2 ** (-(a12 - a23)) * (m3 ** (1 - a23) - m2 ** (1 - a23)) / (1 - a23)
        + m2 ** (-(a12 - a23)) * m3 ** (-(a23 - a34)) * (m4 ** (1 - a34) - m3 ** (1 - a34)) / (1 - a34)
    )
    b2 = b1 * m2 ** (-(a12 - a23))
    b3 = b2 * m3 ** (-(a23 - a34))
    return b1, b2, b3


def imf(mass: float, m1: float = 0.01, m2: float = 0.08, m3: float = 0.5, m4: float = 200.0,
        a12: float = 0.3, a23: float = 1.3, a34: float = 2.3) -> float:
    """Evaluate the Kroupa IMF at a single mass value.

    Parameters
    ----------
    mass:
        Stellar mass at which to evaluate the IMF (in solar masses).
    mi, ai:
        See :func:`imf_normalisation_constants` for parameter definitions.

    Returns
    -------
    value:
        Value of the IMF at ``mass``.
    """
    b1, b2, b3 = imf_normalisation_constants(m1, m2, m3, m4, a12, a23, a34)
    if m1 <= mass < m2:
        return b1 * mass ** (-a12)
    elif m2 <= mass < m3:
        return b2 * mass ** (-a23)
    elif m3 <= mass < m4:
        return b3 * mass ** (-a34)
    else:
        return 0.0


def analytical_star_forming_mass_per_binary_using_kroupa_imf(
    m1_min: float,
    m1_max: float,
    m2_min: float,
    fbin: float,
    mass_ratio_pdf_function: Callable[[float], float] = lambda q: 1.0,
    m1: float = 0.01,
    m2: float = 0.08,
    m3: float = 0.5,
    m4: float = 200.0,
    a12: float = 0.3,
    a23: float = 1.3,
    a34: float = 2.3,
) -> float:
    """Compute the mass of stars formed per binary under a Kroupa IMF.

    This function estimates the total mass formed in binary systems
    relative to the number of binaries sampled by COMPAS.  It follows
    Eq. 6 of Neijssel et al. (2019) and related work.

    Parameters
    ----------
    m1_min, m1_max:
        Lower and upper bounds on the primary mass distribution sampled in
        COMPAS.
    m2_min:
        Lower bound on the secondary mass distribution sampled in COMPAS.
    fbin:
        Binary fraction.
    mass_ratio_pdf_function:
        Probability density function of the mass ratio ``q = m2 / m1``.
    mi, ai:
        Parameters of the Kroupa IMF.

    Returns
    -------
    mass_per_binary:
        Expected mass formed per binary in units of solar masses.
    """
    # The implementation is omitted in the stub
    pass


def get_compas_fraction(
    m1_low: float,
    m1_upp: float,
    m2_low: float,
    f_bin: float,
    mass_ratio_pdf_function: Callable[[float], float] = lambda q: 1.0,
    m1: float = 0.01,
    m2: float = 0.08,
    m3: float = 0.5,
    m4: float = 200.0,
    a12: float = 0.3,
    a23: float = 1.3,
    a34: float = 2.3,
) -> float:
    """Calculate the fraction of the universal star formation captured by COMPAS.

    Given the mass range sampled by COMPAS and a binary fraction, this
    function estimates the fraction of the total mass in stars that falls
    within the sampled regime.  It can be used to normalise rates
    computed from COMPAS simulations to the total star forming mass.
    """
    # Implementation omitted in stub
    pass


def retrieve_mass_evolved_per_z(path: str) -> np.ndarray:
    """Return the total mass evolved per metallicity bin from a COMPAS file.

    Parameters
    ----------
    path:
        Path to the COMPAS HDF5 output file.

    Returns
    -------
    masses:
        One–dimensional array containing the total mass evolved in each
        metallicity bin of the simulation (in solar masses).
    """
    # Implementation omitted in stub
    pass


def total_mass_evolved_per_z(
    path: str,
    m_lower: float,
    m_upper: float,
    m2_lower: float,
    binary_fraction: float,
    mass_ratio_pdf_function: Callable[[float], float] = lambda q: 1.0,
    m1: float = 0.01,
    m2: float = 0.08,
    m3: float = 0.5,
    m4: float = 200.0,
    a12: float = 0.3,
    a23: float = 1.3,
    a34: float = 2.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the metallicity grid and mass evolved per metallicity.

    This function reads the metallicity distribution from the COMPAS
    dataset and returns two arrays: the sorted unique metallicities and
    the corresponding mass evolved, corrected for the sampled mass range
    and binary fraction.  The implementation should call
    :func:`retrieve_mass_evolved_per_z` to obtain the raw mass and then
    normalise by the fraction returned by :func:`get_compas_fraction`.
    """
    # Implementation omitted in stub
    pass