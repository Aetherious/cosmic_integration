"""Utility conversion functions for binned analyses.

This module reproduces a minimal subset of the conversions used in the
original binned cosmic integration pipeline.  The functions convert
between component masses, chirp mass, symmetric mass ratio and total
mass.  These routines are provided for completeness and may be useful
when working with summary outputs.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "m1_m2_to_chirp_mass",
    "m1_m2_to_eta",
    "m1_m2_to_eta_chirp_mass",
    "chirp_mass_eta_to_total_mass",
    "total_mass_eta_to_m1_m2",
]


def m1_m2_to_chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Return the chirp mass of a binary given component masses.

    Parameters
    ----------
    m1, m2 : numpy.ndarray
        Arrays of primary and secondary masses.

    Returns
    -------
    numpy.ndarray
        The chirp mass array.
    """
    return np.power(m1 * m2, 3.0 / 5.0) / np.power(m1 + m2, 1.0 / 5.0)


def m1_m2_to_eta(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Return the symmetric mass ratio of a binary given component masses."""
    return (m1 * m2) / np.square(m1 + m2)


def m1_m2_to_eta_chirp_mass(m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the pair (eta, chirp_mass) for given component masses."""
    return m1_m2_to_eta(m1, m2), m1_m2_to_chirp_mass(m1, m2)


def chirp_mass_eta_to_total_mass(chirp_mass: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Return the total mass given chirp mass and symmetric mass ratio."""
    return chirp_mass / np.power(eta, 3.0 / 5.0)


def total_mass_eta_to_m1_m2(total_mass: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the component masses given total mass and symmetric mass ratio.

    The quadratic formula is used to solve for the mass ratio.
    """
    q = (1.0 - np.sqrt(1.0 - 4.0 * eta)) / (2.0 * eta)
    m1 = total_mass * q / (1.0 + q)
    m2 = total_mass - m1
    return m1, m2