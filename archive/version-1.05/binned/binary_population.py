"""Binary population representation for the binned cosmic integrator.

This module defines a :class:`BinaryPopulation` class that wraps the
component masses, delay times, metallicities and other metadata of
double compact object populations.  Instances are typically created
from a COMPAS HDF5 file via the :meth:`from_compas_h5` class method.
The class stores the minimal information required by the binned
integrator and provides basic helper methods for derived quantities.

The stub implementation below defines the public API without any
internal logic.  See the original ``binary_population.py`` for a
complete reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class BinaryPopulation:
    """General double compact object population class.

    Parameters
    ----------
    m1, m2:
        Arrays of primary and secondary masses (solar masses).
    t_delay:
        Array of delay times (Myr).
    z_zams:
        Array of metallicities at zero–age main sequence (mass fraction of metals).
    n_systems:
        Total number of systems in the underlying COMPAS population.
    dcos_included:
        List of strings indicating which DCO types were selected (e.g. ["BBH"]).
    m1_min, m1_max, m2_min:
        Mass limits used in the population synthesis simulation.
    binary_fraction:
        Binary fraction assumed in the simulation.
    """

    m1: np.ndarray
    m2: np.ndarray
    t_delay: np.ndarray
    z_zams: np.ndarray
    n_systems: int
    dcos_included: List[str]
    m1_min: float
    m1_max: float
    m2_min: float
    binary_fraction: float

    @classmethod
    def from_compas_h5(
        cls,
        path: str,
        dcos_included: Sequence[str] = ("BBH",),
        m1_min: float = 5.0,
        m1_max: float = 150.0,
        m2_min: float = 0.1,
        binary_fraction: float = 0.7,
    ) -> "BinaryPopulation":
        """Load a binary population from a COMPAS HDF5 file.

        Parameters
        ----------
        path:
            Path to the COMPAS output file.
        dcos_included:
            Sequence of DCO types to include (default is ["BBH"]).
        m1_min, m1_max, m2_min:
            Mass limits applied to the population.
        binary_fraction:
            Binary fraction assumed in the simulation.

        Returns
        -------
        population:
            Instance of :class:`BinaryPopulation` containing the selected systems.
        """
        # Implementation omitted in stub
        raise NotImplementedError