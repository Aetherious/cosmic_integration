"""Compute detection probabilities on a grid of binary parameters.

The :class:`DetectionMatrix` class evaluates the probability of
detecting binary mergers on a binned parameter grid of chirp mass and
symmetric mass ratio.  It optionally utilises GPU acceleration via
CuPy when available.  The stub defines the interface only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .conversions import m1_m2_to_chirp_mass, m1_m2_to_eta
from ..selection_effects import detection_probability


@dataclass
class DetectionMatrix:
    """Tabulate detection probability on a two–dimensional parameter grid.

    Parameters
    ----------
    m1_bins, m2_bins:
        Arrays defining the edges of the mass bins for the primary and
        secondary masses.
    redshift_bins:
        Array of redshift bin edges.
    snr_threshold:
        Detection threshold on the matched–filter signal–to–noise ratio.
    sensitivity:
        Detector sensitivity curve identifier.
    """

    m1_bins: np.ndarray
    m2_bins: np.ndarray
    redshift_bins: np.ndarray
    snr_threshold: float
    sensitivity: str = "design"

    def compute(self) -> None:
        """Evaluate detection probability on the parameter grid.

        This method fills in a three–dimensional array indexed by mass
        bins and redshift bin with the probability that a system in
        that bin would be detected.  GPU acceleration may be used
        internally if available.
        """
        pass