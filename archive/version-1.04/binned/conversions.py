"""Mass conversion utilities for compact binaries.

This module provides simple helper functions to convert between the
component masses of a binary and derived quantities such as the chirp
mass and symmetric mass ratio.  These conversions are used throughout
the binned integrator and detection calculations.
"""

from __future__ import annotations

import numpy as np


def m1_m2_to_chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Return the chirp mass of a binary system.

    The chirp mass :math:`\mathcal{M} = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5}`
    governs the phasing of the gravitational waveform.

    Parameters
    ----------
    m1, m2:
        Arrays of primary and secondary masses (solar masses).

    Returns
    -------
    chirp_mass:
        Array of chirp masses with the same shape as the inputs.
    """
    # Implementation omitted in stub
    pass


def m1_m2_to_eta(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Return the symmetric mass ratio of a binary system.

    The symmetric mass ratio :math:`\eta = m_1 m_2 / (m_1 + m_2)^2` is
    dimensionless and takes values in the interval (0, 0.25].

    Parameters
    ----------
    m1, m2:
        Arrays of primary and secondary masses (solar masses).

    Returns
    -------
    eta:
        Array of symmetric mass ratios with the same shape as the inputs.
    """
    # Implementation omitted in stub
    pass


def m1_m2_to_eta_chirp_mass(m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return both the symmetric mass ratio and chirp mass for arrays of masses.

    This convenience function computes :math:`\eta` and :math:`\mathcal{M}`
    in a single call.  It is often used when converting from component
    masses to the parameters used in signal–to–noise grids.

    Parameters
    ----------
    m1, m2:
        Arrays of primary and secondary masses (solar masses).

    Returns
    -------
    eta, chirp_mass:
        Tuple of arrays containing the symmetric mass ratio and chirp mass.
    """
    # Implementation omitted in stub
    return np.zeros_like(m1), np.zeros_like(m1)


def chirp_mass_eta_to_m1_m2(chirp_mass: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert chirp mass and symmetric mass ratio back to component masses.

    Parameters
    ----------
    chirp_mass:
        Array of chirp masses (solar masses).
    eta:
        Array of symmetric mass ratios.

    Returns
    -------
    m1, m2:
        Arrays of primary and secondary masses corresponding to the inputs.
    """
    # Implementation omitted in stub
    return np.zeros_like(chirp_mass), np.zeros_like(chirp_mass)