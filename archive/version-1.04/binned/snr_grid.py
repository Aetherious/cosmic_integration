"""Grid of signal–to–noise values and detection probabilities.

This module defines the :class:`SNRGrid`, which precomputes the
signal–to–noise ratio (SNR) of compact binary mergers as a function
of chirp mass and symmetric mass ratio at a reference distance, along
with the corresponding detection probability as a function of SNR.
The grid is used to look up detection probabilities without
recomputing waveform SNRs for each binary.
The stub implementation defines the interface but leaves the heavy
computations unimplemented.

Classes
-------
SNRGrid
    Represent a grid of SNR values and detection probabilities.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .conversions import chirp_mass_eta_to_m1_m2, m1_m2_to_eta_chirp_mass  # type: ignore
from ..selection_effects import SNRinterpolator, detection_probability_from_snr


@dataclass
class SNRGrid:
    """Precompute SNR values and detection probabilities on a parameter grid.

    Parameters
    ----------
    sensitivity:
        Name of the detector sensitivity curve (e.g., ``"O1"`` or ``"design"``).
    mc_max, mc_step:
        Maximum chirp mass and step size used when constructing the grid.
    eta_max, eta_step:
        Maximum symmetric mass ratio and step size used when constructing the grid.
    snr_threshold, snr_max, snr_step:
        Parameters defining the SNR sampling for computing detection
        probabilities.
    """

    sensitivity: str = "O1"
    mc_max: float = 300.0
    mc_step: float = 0.1
    eta_max: float = 0.25
    eta_step: float = 0.01
    snr_threshold: float = 8.0
    snr_max: float = 1000.0
    snr_step: float = 0.1

    # Precomputed arrays; set in __post_init__
    chirp_mass: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    eta: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    m1: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    m2: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    snr_grid_at_1Mpc: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    snr: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    pdetection: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialise the parameter grids.  The real implementation would
        # compute m1, m2 on a mesh, call the SNR interpolator and compute
        # detection probabilities.
        self.chirp_mass = np.arange(self.mc_step, self.mc_max + self.mc_step, self.mc_step)
        self.eta = np.arange(self.eta_step, self.eta_max + self.eta_step, self.eta_step)
        # Placeholder arrays matching expected shapes
        self.m1 = np.zeros((len(self.eta), len(self.chirp_mass)))
        self.m2 = np.zeros((len(self.eta), len(self.chirp_mass)))
        self.snr_grid_at_1Mpc = np.zeros((len(self.eta), len(self.chirp_mass)))
        self.snr = np.arange(self.snr_step, self.snr_max + self.snr_step, self.snr_step)
        self.pdetection = np.zeros_like(self.snr)

    def plot(self) -> object:
        """Return a figure representing the SNR grid.

        The real implementation would produce a colour plot of SNR as a
        function of component masses and overlay the detection probability.
        The stub returns ``None``.
        """
        return None

    @property
    def label(self) -> str:
        """Return a label suitable for saving plots associated with this grid."""
        return f"snr_{self.sensitivity}"
