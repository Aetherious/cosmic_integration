#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mass per Redshift
=================

Calculates the total mass evolved per metallicity as a function of redshift for a COMPAS-simulated dataset.

*Refactored to adhere with PEP 8 formatting guidelines and NumPy-style docstrings.

Author
------
Unknown (TeamCOMPAS)

Reviewed & Refactored
---------------------
Suoi-Nguon Pham <aetheriofficial@gmail.com>

Date
----
2025-08-10 (ONGOING)

License
-------
MIT
"""

# ====== Standard Library ======
from dataclasses import dataclass
import functools
from typing import NamedTuple, Callable

# ====== External Packages ======
import h5py as h5
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ====== Global Constants ======
DEFAULT_M1: float = 0.01
DEFAULT_M2: float = 0.08
DEFAULT_M3: float = 0.5
DEFAULT_M4: float = 200.0
DEFAULT_A12: float = 0.3
DEFAULT_A23: float = 1.3
DEFAULT_A34: float = 2.3


@dataclass(frozen=True)
class IMFParameters:
    """
    Parameters defining a segmented initial mass function (IMF).
    
    Default follows Kroupa (2001), Eqns 1-2.
    https://arxiv.org/abs/astro-ph/0009005

    Attributes
    ----------
    m1, m2, m3, m4 : `float`
        Mass breakpoints for the IMF.
    a12, a23, a34  : `float`
        Power-law slopes for each IMF segment.    
    """
    m1: float = DEFAULT_M1
    m2: float = DEFAULT_M2
    m3: float = DEFAULT_M3
    m4: float = DEFAULT_M4
    a12: float = DEFAULT_A12
    a23: float = DEFAULT_A23
    a34: float = DEFAULT_A34


class IMFNormalizationValues(NamedTuple):
    """
    Normalization constants for the three initial mass function (IMF) segments.
    
    Attributes
    ----------
    b1, b2, b3 : `float`
        Normalization constants ensuring IMF continuity.
    """
    b1: float
    b2: float
    b3: float


@functools.lru_cache()
def _compute_imf_normalization_constants(params: IMFParameters) -> IMFNormalizationValues:
    """
    Compute normalization constants for the initial mass function (IMF).
    
    Parameters
    ----------
    params : `IMFParameters`
        IMF mass bounds and slope parameters.

    Returns
    -------
    `IMFNormalizationValues`
        Named tuple containing normalization constants for the three IMF segments.
    """
    m1, m2, m3, m4 = params.m1, params.m2, params.m3, params.m4
    a12, a23, a34 = params.a12, params.a23, params.a34

    b1 = 1 / (
        (m2 ** (1 - a12) - m1 ** (1 - a12)) / (1 - a12)
        + m2 ** (-(a12 - a23)) * (m3 ** (1 - a23) - m2 ** (1 - a23)) / (1 - a23)
        + m2 ** (-(a12 - a23)) * m3 ** (-(a23 - a34)) * (m4 ** (1 - a34) - m3 ** (1 - a34)) / (1 - a34)
    )
    b2 = b1 * m2 ** (-(a12 - a23))
    b3 = b2 * m3 ** (-(a23 - a34))

    return IMFNormalizationValues(b1, b2, b3)


@np.vectorize
def IMF(m: float, params: IMFParameters = IMFParameters()) -> float:
    """
    Evaluate the initial mass function (IMF) at a given mass for a segmented power law.
    
    Calculates the fraction of stellar mass between m and m + dm for a three-part broken power-law. 
    
    Parameters
    ----------
    m : `float` or `np.ndarray`
        Stellar mass at which to evaluate the IMF.
    params : `IMFParameters`, optional
        IMF parameter set.

    Returns
    -------
    `float`
        IMF evaluated at the given masses.
    """
    # Compute normalization constants to ensure the IMF is continuous.
    b1, b2, b3 = _compute_imf_normalization_constants(params)

    # Evaluate IMF at a point or list of points.
    if params.m1 <= m < params.m2:
        return b1 * m ** (-params.a12)
    elif params.m2 <= m < params.m3:
        return b2 * m ** (-params.a23)
    elif params.m3 <= m < params.m4:
        return b3 * m ** (-params.a34)
    return 0.0


def fractionalize_mass(
    m1_low: float,
    m1_upp: float,
    m2_low: float,
    f_bin: float,
    mass_ratio_pdf_function: Callable[[float], float] = lambda q: 1,
    params: IMFParameters = IMFParameters()
) -> float:
    """
    Calculate the mass in a COMPAS population relative to that of the total universal population.
    
    Can be used to normalize the rates of objects from COMPAS simulations.

    Parameters
    ----------
    m1_low : `float`
        Lower limit on the sampled primary mass.
    m1_upp : `float`
        Upper limit on the sampled primary mass.
    m2_low : `float`
        Lower limit on the sampled secondary mass.
    f_bin : `float`
        Binary fraction.
    mass_ratio_pdf_function : `Callable`, optional
        Function to calculate the mass ratio PDF. Default is uniform distribution.
    params : `IMFParameters`, optional
        IMF parameter set.

    Returns
    -------
    `float`
        Mass in a COMPAS population relative to total universal population.
    """

    # For normalization purposes, first compute the integral with no COMPAS cuts.
    def _full_integral(mass: float) -> float:
        primary_mass: float = IMF(mass, params) * mass
        
        # Compute the expected secondary mass given the mass ratio pdf function.
        expected_secondary_mass: float = quad(lambda q: q * mass_ratio_pdf_function(q), 0, 1)[0] * primary_mass
        
        single_stars: float = (1 - f_bin) * primary_mass
        binary_stars: float = f_bin * (primary_mass + expected_secondary_mass)
        return single_stars + binary_stars
    
    full_mass: float = quad(_full_integral, m1, m4, args=(m1, m2, m3, m4, a12, a23, a34))[0]
    
    # Then, compute the integral for the COMPAS regime.
    def _compas_integral(mass, m2_low, f_bin, m1, m2, m3, m4, a12, a23, a34):
        primary_mass: float = IMF(mass, params) * mass
        
        # find the fraction that are below the m2 mass cut
        f_below_m2low: float = quad(mass_ratio_pdf_function, 0, m2_low / mass)[0]
        
        # Compute the expected companion mass given the m2 cut and mass ratio pdf function.
        # expectation value of the secondary mass given the m2 cut and mass ratio pdf function
        expected_secondary_mass: float = quad(lambda q: q * mass_ratio_pdf_function(q), m2_low / mass, 1)[0] * primary_mass
        
        # return total mass of binary stars that have m2 above the cut
        return f_bin * (1 - f_below_m2low) * (primary_mass + expected_secondary_mass)
    
    compas_mass: float  = quad(_compas_integral, m1_low, m1_upp, args=(m2_low, f_bin, m1, m2, m3, m4, a12, a23, a34))[0]
    
    return compas_mass / full_mass


def retrieve_evolved_mass(path: str) -> np.ndarray:
    """
    Retrieve total stellar mass evolved as function of metallicity from COMPAS HDF5 output.

    Parameters
    ----------
    path : `str`
        COMPAS HDF5 file path.
    
    Returns
    -------
    `np.ndarray`
        Bin for total stellar mass evolved as a function of metallicity.
    """
    with h5.File(path, 'r') as f:
        all_systems = f['BSE_System_Parameters']
        metals = all_systems['Metallicity@ZAMS(1)'][()]
        m1s = all_systems['Mass@ZAMS(1)'][()]
        m2s = all_systems['Mass@ZAMS(2)'][()]

        unique_metals = np.unique(metals)
        total = np.zeros(len(unique_metals))

        for i, Z in enumerate(unique_metals):
            mask = metals == Z
            total[i] = np.sum(m1s[mask]) + np.sum(m2s[mask])

    return total

def mass_per_z(path, Mlower, Mupper, m2_low, binaryFraction, mass_ratio_pdf_function=lambda q: 1,
                         m1=DEFAULT_M1, m2=DEFAULT_M2, m3=DEFAULT_M3, m4=DEFAULT_M4, a12=DEFAULT_A12, a23=DEFAULT_A23, a34=DEFAULT_A34):
    """
    Calculate the total mass evolved per metallicity as a function of redshift in a COMPAS simulation.
    """

    # calculate the fraction of mass in the COMPAS simulation vs. the real population without sample cuts
    fraction = fractionalize_mass(m1_low=Mlower, m1_upp=Mupper, m2_low=m2_low, f_bin=binaryFraction,
                                   mass_ratio_pdf_function=mass_ratio_pdf_function,
                                   m1=m1, m2=m2, m3=m3, m4=m4, a12=a12, a23=a23, a34=a34)
    multiplicationFactor = 1 / fraction

    # get the mass evolved for each metallicity bin and convert to a total mass using the fraction
    MassEvolvedPerZ = retrieve_evolved_mass(path)

    totalMassEvolvedPerMetallicity = MassEvolvedPerZ / fraction

    return multiplicationFactor, totalMassEvolvedPerMetallicity

def star_forming_mass_per_binary(
        path,
        Mlower, Mupper, m2_low, binaryFraction, mass_ratio_pdf_function=lambda q: 1,
        m1=DEFAULT_M1, m2=DEFAULT_M2, m3=DEFAULT_M3, m4=DEFAULT_M4, a12=DEFAULT_A12, a23=DEFAULT_A23, a34=DEFAULT_A34):
    """
    Calculate the total mass of stars formed per binary star formed within the COMPAS simulation.
    """
    multiplicationFactor, _ = mass_per_z(**locals())

    # get the total mass in COMPAS and number of binaries
    with h5.File(path, 'r') as f:
        all_systems = f['BSE_System_Parameters']
        m1s = (all_systems['Mass@ZAMS(1)'])[()]
        m2s = (all_systems['Mass@ZAMS(2)'])[()]
        n_binaries = len(m1s)
        total_star_forming_mass_in_COMPAS = sum(m1s) + sum(m2s)

    total_star_forming_mass = total_star_forming_mass_in_COMPAS * multiplicationFactor
    return total_star_forming_mass / n_binaries

def inverse_sample_IMF(
        n_samples = int(1e5),
        m_min=0.01, m_max=200,
        m1=DEFAULT_M1, m2=DEFAULT_M2, m3=DEFAULT_M3, m4=DEFAULT_M4, a12=DEFAULT_A12, a23=DEFAULT_A23, a34=DEFAULT_A34,
        cdf_pts=int(1e4)
        ):
    m = np.linspace(m_min, m_max, cdf_pts)
    imf_values = IMF(m, m1, m2, m3, m4, a12, a23, a34)
    cumulative = np.cumsum(imf_values)
    cumulative -= cumulative.min()
    f = interp1d(cumulative/cumulative.max(), m)
    return f(np.random.random(n_samples))

def draw_samples_from_kroupa_imf(
        Mlower, Mupper, m2_low,
        m1=DEFAULT_M1, m2=DEFAULT_M2, m3=DEFAULT_M3, m4=DEFAULT_M4, a12=DEFAULT_A12, a23=DEFAULT_A23, a34=DEFAULT_A34,
        n_samples = int(1e5)
):
    """
    Draw samples from the Kroupa IMF
    """
    m1_samples = inverse_sample_IMF(n_samples=n_samples,
        m_min=Mlower, m_max=Mupper,
        m1=m1, m2=m2, m3=m3, m4=m4, a12=a12, a23=a23, a34=a34
    )
    m2_samples = m1_samples * np.random.random(n_samples)
    mask = (Mlower < m1_samples) & (m1_samples <= Mupper) & (m2_low < m2_samples)
    return m1_samples[mask] , m2_samples[mask]

def analytical_star_forming_mass_per_binary_using_kroupa_imf(
        m1_min, m1_max, m2_min, fbin=1., imf_mass_bounds=[0.01,0.08,0.5,200]
):
    """
    Analytical computation of the mass of stars formed per binary star formed within the
    [m1 min, m1 max] and [m2 min, ..] rage,
    using the Kroupa IMF:

        p(M) \propto M^-0.3 for M between m1 and m2
        p(M) \propto M^-1.3 for M between m2 and m3;
        p(M) = alpha * M^-2.3 for M between m3 and m4;

    @Ilya Mandel's derivation
    """
    m1, m2, m3, m4 = imf_mass_bounds
    if m1_min < m3:
        raise ValueError(f"This analytical derivation requires IMF break m3  < m1_min ({m3} !< {m1_min})")
    alpha = (-(m4**(-1.3)-m3**(-1.3))/1.3 - (m3**(-0.3)-m2**(-0.3))/(m3*0.3) + (m2**0.7-m1**0.7)/(m2*m3*0.7))**(-1)
    # average mass of stars (average mass of all binaries is a factor of 1.5 larger)
    m_avg = alpha * (-(m4**(-0.3)-m3**(-0.3))/0.3 + (m3**0.7-m2**0.7)/(m3*0.7) + (m2**1.7-m1**1.7)/(m2*m3*1.7))
    # fraction of binaries that COMPAS simulates
    fint = -alpha / 1.3 * (m1_max ** (-1.3) - m1_min ** (-1.3)) + alpha * m2_min / 2.3 * (m1_max ** (-2.3) - m1_min ** (-2.3))
    # mass represented by each binary simulated by COMPAS
    m_rep = (1/fint) * m_avg * (1.5 + (1-fbin)/fbin)
    return m_rep
