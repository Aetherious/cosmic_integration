"""Cosmology helpers for the binned integrator.

This module defines a :class:`CosmologicalModel` that wraps an
`astropy.cosmology.FLRW` instance and precomputes arrays of lookback
times, luminosity distances and comoving volumes on a redshift grid.
These tables are then consumed by the binned integrator to avoid
repeated calls into Astropy.  The stub defines the public API but
leaves the implementation empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.cosmology import FLRW


@dataclass
class CosmologicalModel:
    """Precomputed cosmological relations on a redshift grid.

    Parameters
    ----------
    cosmology:
        Astropy cosmology instance used for all conversions.
    z_min, z_max:
        Minimum and maximum redshift of the table.
    n_bins:
        Number of points in the redshift grid.
    """

    cosmology: FLRW
    z_min: float = 0.0
    z_max: float = 10.0
    n_bins: int = 1000

    # Precomputed arrays
    redshifts: Optional[np.ndarray] = None
    lookback_times: Optional[np.ndarray] = None
    luminosity_distances: Optional[np.ndarray] = None
    comoving_volumes: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Implementation would fill in the precomputed arrays
        pass