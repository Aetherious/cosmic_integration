"""Compute binned detection rates for compact binary populations.

This module contains functions that take a binary population, a
precomputed cosmological model, and a detector signal–to–noise grid
and return a two–dimensional matrix of detection rates.  The
computation typically involves calculating the formation rate of
systems at each redshift, evolving each system through a delay–time
distribution to determine when it merges, and folding in the
detection probability.  The stub below defines the interface but
omits the numerical integration.

Functions
---------
compute_binned_detection_rates(dco_population, cosmological_model, snr_grid,
                               chirp_mass_bins, redshift_bins,
                               max_detectable_redshift=1.0)
    Compute a detection rate matrix on the supplied chirp mass and
    redshift grids.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .binary_population import BinaryPopulation
from .cosmological_model import CosmologicalModel
from .snr_grid import SNRGrid


def compute_binned_detection_rates(
    dco_population: BinaryPopulation,
    cosmological_model: CosmologicalModel,
    snr_grid: SNRGrid,
    chirp_mass_bins: Sequence[float],
    redshift_bins: Sequence[float],
    max_detectable_redshift: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Return the detection rate as a 2D array binned in chirp mass and redshift.

    Parameters
    ----------
    dco_population:
        Binary population containing masses, delay times and metallicities.
    cosmological_model:
        Precomputed cosmological relations such as the cosmic star formation
        history and metallicity distribution.
    snr_grid:
        Grid of signal–to–noise values and detection probabilities as a
        function of chirp mass and symmetric mass ratio.
    chirp_mass_bins, redshift_bins:
        Edges of the bins in chirp mass and redshift on which to compute
        the rates.
    max_detectable_redshift:
        Maximum redshift at which to evaluate the rate (defaults to 1.0).
    verbose:
        If ``True``, emit progress information via the logging module.

    Returns
    -------
    rate_matrix:
        A two–dimensional array of shape ``(len(chirp_mass_bins), len(redshift_bins))``
        containing the number of detections per year per unit comoving volume.
        The stub implementation returns a zeros array of the expected shape.
    """
    n_mc_bins = len(chirp_mass_bins)
    n_z_bins = len(redshift_bins)
    return np.zeros((n_mc_bins, n_z_bins))
