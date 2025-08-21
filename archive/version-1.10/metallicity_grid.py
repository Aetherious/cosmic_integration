r"""Utilities related to metallicity grids and stellar mass evolution.

This module provides a collection of helper functions to evaluate the
initial mass function (IMF), compute analytic expressions for the
stellar mass formed per binary and derive the total mass evolved at a
given metallicity.  The implementations are intentionally
straight‑forward and rely on numerical integration through NumPy; they
avoid dependencies on external libraries such as ``scipy`` or
``h5py`` which may not be available in all environments.

The functions defined here are designed to be used in concert with the
cosmic integrator.  For example, when estimating the number of double
compact object (DCO) progenitors born per unit star–forming mass, one
often divides the intrinsic COMPAS yields by the mass evolved per
binary returned by :func:`analytical_star_forming_mass_per_binary_using_kroupa_imf`.

The Kroupa IMF is implemented piecewise with three segments and
exponents taken from the literature; see Kroupa (2001) for details.
The normalisation constant is fixed by requiring that the integral of
the IMF over the entire mass range be unity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def _kroupa_imf(m: np.ndarray) -> np.ndarray:
    """Return the unnormalised Kroupa initial mass function.

    The function is defined piecewise over the mass range
    0.08–0.5–1–120 M_☉.  For masses below 0.5 M_☉ the slope is
    −1.3, between 0.5 and 1 M_☉ it is −2.3, and above 1 M_☉ the slope
    is −2.3 as well.  The normalisation is applied in
    :func:`analytical_star_forming_mass_per_binary_using_kroupa_imf`.

    Parameters
    ----------
    m:
        Array of stellar masses in solar masses.

    Returns
    -------
    ndarray
        The unnormalised IMF evaluated at each mass.
    """
    m = np.asarray(m)
    imf = np.zeros_like(m, dtype=float)
    # Segment slopes
    alpha_low = 1.3
    alpha_mid = 2.3
    alpha_high = 2.3
    # Apply power‑law segments; we use m**(-alpha) without
    # normalisation constants here.  The constants cancel when
    # computing ratios in the analytic formula.
    mask_low = (m >= 0.08) & (m < 0.5)
    mask_mid = (m >= 0.5) & (m < 1.0)
    mask_high = m >= 1.0
    imf[mask_low] = m[mask_low] ** (-alpha_low)
    imf[mask_mid] = (0.5 ** (alpha_mid - alpha_low)) * m[mask_mid] ** (-alpha_mid)
    imf[mask_high] = (
        (0.5 ** (alpha_mid - alpha_low)) * (1.0 ** (alpha_high - alpha_mid)) * m[mask_high] ** (-alpha_high)
    )
    return imf


def imf_normalisation_constants(m_lower: float, m_upper: float) -> Tuple[float, float]:
    r"""Return the normalisation constant for the Kroupa IMF and the mean mass.

    The normalisation constant ``A`` is defined such that

    .. math::

        \int_{m_\mathrm{lower}}^{m_\mathrm{upper}} A\,\xi(m)\,dm = 1,

    where ``xi(m)`` is the piecewise unnormalised IMF returned by
    :func:`_kroupa_imf`.  The mean mass is defined as

    .. math::

        \bar{m} = \int m\,A\,\xi(m)\,dm.

    Parameters
    ----------
    m_lower, m_upper:
        The lower and upper limits of integration in solar masses.

    Returns
    -------
    tuple
        A two‑tuple ``(A, m_mean)`` containing the normalisation
        constant and the mean stellar mass.
    """
    # Create a fine grid for numerical integration
    m = np.logspace(np.log10(m_lower), np.log10(m_upper), 10000)
    dm = np.diff(m)
    # Evaluate the unnormalised IMF at midpoints for trapezoidal rule
    xi = _kroupa_imf(m)
    # Compute integral of xi(m) dm
    integral_xi = np.trapz(xi, m)
    A = 1.0 / integral_xi
    # Compute mean mass
    m_mean = np.trapz(m * xi, m) * A
    return A, m_mean


def analytical_star_forming_mass_per_binary_using_kroupa_imf(
    m_lower: float = 5.0,
    m_upper: float = 150.0,
    m2_min: float = 0.08,
    binary_fraction: float = 1.0,
) -> float:
    r"""Compute the analytic star–forming mass required per binary system.

    The cosmic integrator expresses DCO formation efficiencies in terms
    of the number of DCOs formed per unit star–forming mass.  To
    convert the COMPAS yields (per binary) into volumetric rates, one
    multiplies the yield by the star–forming mass per binary.  This
    function computes that mass assuming a Kroupa initial mass
    function and a binary fraction (fraction of stars in binary
    systems).

    The mass per binary is given by

    .. math::

        M_\mathrm{SF} = \frac{\bar{m}}{f_\mathrm{bin}}\,
        \frac{\int_{m_\mathrm{lower}}^{m_\mathrm{upper}} m\,\xi(m)\,dm}{\int_{m_2^\mathrm{min}}^{m_\mathrm{upper}} \xi(m)\,dm},

    where ``\bar{m}`` is the mean mass of the IMF over the entire
    stellar mass range.  The numerator gives the total mass formed in
    primaries between ``m_lower`` and ``m_upper`` while the denominator
    computes the fraction of stars above the secondary mass minimum.

    Parameters
    ----------
    m_lower:
        Lower bound of the primary mass distribution in solar masses
        (default is 5).  Masses below this threshold do not contribute
        to compact objects.
    m_upper:
        Upper bound of the primary mass distribution in solar masses
        (default is 150).
    m2_min:
        Minimum mass of the secondary in solar masses (default is 0.08).
    binary_fraction:
        Fraction of stars that form in binary systems (default is 1).

    Returns
    -------
    float
        The star–forming mass (in solar masses) required to form a
        single binary system in the given mass range.
    """
    if binary_fraction <= 0 or binary_fraction > 1:
        raise ValueError("binary_fraction must lie in (0, 1].")
    # Build a mass grid for integration
    m = np.logspace(np.log10(m2_min), np.log10(m_upper), 10000)
    xi = _kroupa_imf(m)
    # Integrate the IMF over the entire range to get normalisation
    integral_xi = np.trapz(xi, m)
    # Integrate m*xi over the primary mass range for mass formed in primaries
    mask_prim = (m >= m_lower) & (m <= m_upper)
    m_prim = m[mask_prim]
    xi_prim = xi[mask_prim]
    mass_integral = np.trapz(m_prim * xi_prim, m_prim)
    # Integrate xi over the range above m2_min for secondary fraction
    mask_sec = (m >= m2_min) & (m <= m_upper)
    m_sec = m[mask_sec]
    xi_sec = xi[mask_sec]
    frac_above_m2_min = np.trapz(xi_sec, m_sec) / integral_xi
    # Mean mass of the IMF
    m_mean = np.trapz(m * xi, m) / integral_xi
    # Compute mass per binary
    mass_per_binary = m_mean / binary_fraction * (mass_integral / (frac_above_m2_min * integral_xi))
    return float(mass_per_binary)


def total_mass_evolved_per_z(
    metallicity_grid: Iterable[float],
    m_lower: float = 5.0,
    m_upper: float = 150.0,
    m2_min: float = 0.08,
    binary_fraction: float = 1.0,
) -> np.ndarray:
    """Compute the total mass evolved for each metallicity bin.

    This helper uses :func:`analytical_star_forming_mass_per_binary_using_kroupa_imf`
    to determine the stellar mass formed per binary and returns an array
    of the same length as ``metallicity_grid``.  The current
    implementation simply assigns the same mass to every metallicity
    value, reflecting the assumption that the IMF is independent of
    metallicity.  More sophisticated prescriptions could weight the
    mass evolved by metallicity–dependent IMF variations or star
    formation efficiency.

    Parameters
    ----------
    metallicity_grid:
        A one‑dimensional iterable of metallicity values (dimensionless;
        1 corresponds to solar metallicity).
    m_lower, m_upper, m2_min, binary_fraction:
        Parameters passed through to
        :func:`analytical_star_forming_mass_per_binary_using_kroupa_imf`.

    Returns
    -------
    ndarray
        Total mass evolved per metallicity bin.
    """
    mass_per_binary = analytical_star_forming_mass_per_binary_using_kroupa_imf(
        m_lower=m_lower,
        m_upper=m_upper,
        m2_min=m2_min,
        binary_fraction=binary_fraction,
    )
    metallicity_grid = np.asarray(list(metallicity_grid), dtype=float)
    return np.full_like(metallicity_grid, mass_per_binary, dtype=float)
