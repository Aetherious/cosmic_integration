"""Detection probability and SNR interpolation utilities.

This module encapsulates the calculation of detection probabilities for
compact binary mergers given their component masses, redshift and
luminosity distance.  The design follows that of Gaebel et al. (2018)
but has been modernised for readability and extensibility.  An
interpolator class is provided to evaluate signal–to–noise ratios
from precomputed grids stored in HDF5 files.

In production code the module would load the SNR grid lazily and cache
the resulting interpolator.  For the stub we define the interface only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def detection_probability(
    m1: np.ndarray,
    m2: np.ndarray,
    redshift: float,
    distance: float,
    snr_threshold: float,
    sensitivity: str = "design",
) -> np.ndarray:
    """Approximate the detection probability for compact binary mergers.

    This simplified implementation approximates the signal–to–noise ratio
    (SNR) as being proportional to the chirp mass and inversely
    proportional to the luminosity distance.  A logistic function is
    then used to map the SNR to a detection probability.  While this
    approach does not capture the full complexity of detector
    sensitivity curves, it provides a reasonable heuristic for
    demonstrating the cosmic integration pipeline without requiring
    precomputed SNR grids.

    Parameters
    ----------
    m1, m2:
        Arrays of primary and secondary masses in the source frame (solar masses).
    redshift:
        Redshift of the source (unused in this approximate model).
    distance:
        Luminosity distance to the source (in Mpc).  Must be broadcastable
        with ``m1`` and ``m2``.
    snr_threshold:
        Threshold signal–to–noise ratio for detection.  Typical values
        range from 8 to 12 for gravitational–wave detectors.
    sensitivity:
        Identifier of the detector sensitivity curve to use (unused in
        this approximate model).

    Returns
    -------
    ndarray
        Array of detection probabilities corresponding to the input masses.
    """
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    distance = np.asarray(distance, dtype=float)
    # Compute the chirp mass in the source frame (solar masses)
    chirp_mass = ((m1 * m2) ** (3.0 / 5.0)) / ((m1 + m2) ** (1.0 / 5.0))
    # Approximate SNR scaling: proportional to chirp mass and inversely to distance
    # The constant factor is tuned so that systems with chirp mass ~30 Msun at 200 Mpc
    # yield SNR ~ snr_threshold.
    scaling = snr_threshold * 200.0 / 30.0  # SNR ~ threshold at 200 Mpc for chirp mass 30 Msun
    snr_approx = (chirp_mass / distance) * scaling
    # Logistic detection probability
    p_det = 1.0 / (1.0 + np.exp(-(snr_approx - snr_threshold) / 0.5))
    return p_det


@dataclass
class SNRInterpolator:
    """Interpolate SNR values from a precomputed mass grid.

    Parameters
    ----------
    hdf5_path:
        Path to the HDF5 file containing the mass grid and SNR values.
    noise_spectrum:
        Name of the noise spectrum group within the HDF5 file.
    mode:
        Interpolation mode.  Valid options are ``"scipy"`` (use
        `scipy.interpolate.RectBivariateSpline`) and ``"custom"`` (use a
        weighted average of nearest grid points).

    Notes
    -----
    In a full implementation this class would load the mass axis and SNR
    grid from disk during initialisation and expose a ``__call__`` method
    accepting arrays of component masses.  The stub defines the public
    API without performing any computation.
    """

    hdf5_path: str
    noise_spectrum: str
    mode: str = "scipy"

    def __post_init__(self) -> None:
        """Load or initialise the SNR interpolator.

        In a full implementation this method would load precomputed SNR
        grids from the specified HDF5 file.  Because external
        dependencies such as ``h5py`` are not available in this
        environment, the interpolator falls back to an analytic
        approximation using the same heuristic as
        :func:`detection_probability`.  The loaded grid, if present,
        should define a two–dimensional array of SNR values on a
        Cartesian grid of component masses.
        """
        # Attempt to load precomputed grid; if unavailable, fall back to analytic
        self._loaded = False
        try:
            import h5py  # type: ignore[import-not-found]

            with h5py.File(self.hdf5_path, "r") as f:
                group = f[self.noise_spectrum]
                self.m1_grid = group["m1"][:]
                self.m2_grid = group["m2"][:]
                self.snr_grid = group["snr"][:]
                self._loaded = True
        except Exception:
            # Precomputed grid unavailable; we will use analytic formula
            self.m1_grid = None
            self.m2_grid = None
            self.snr_grid = None
            self._loaded = False

    def __call__(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Return the signal–to–noise ratio for given component masses.

        If a precomputed SNR grid was successfully loaded at initialisation
        time, this method will interpolate the SNR values using a
        bilinear interpolation over the mass grid.  Otherwise it
        approximates the SNR using the analytic scaling described in
        :func:`detection_probability` with a nominal distance of 400 Mpc
        and a threshold of 8.

        Parameters
        ----------
        m1, m2:
            Arrays of primary and secondary masses (solar masses).

        Returns
        -------
        ndarray
            Array of SNR values corresponding to the input masses.
        """
        m1 = np.asarray(m1, dtype=float)
        m2 = np.asarray(m2, dtype=float)
        if self._loaded and self.snr_grid is not None:
            # Perform simple bilinear interpolation.  We assume the grid is
            # strictly increasing.
            import numpy as np

            snr = np.empty_like(m1, dtype=float)
            for idx, (m1_val, m2_val) in enumerate(zip(m1, m2)):
                # Find nearest indices
                i = np.searchsorted(self.m1_grid, m1_val) - 1
                j = np.searchsorted(self.m2_grid, m2_val) - 1
                # Clamp indices to valid range
                i = np.clip(i, 0, len(self.m1_grid) - 2)
                j = np.clip(j, 0, len(self.m2_grid) - 2)
                # Grid values
                x0, x1 = self.m1_grid[i], self.m1_grid[i + 1]
                y0, y1 = self.m2_grid[j], self.m2_grid[j + 1]
                f00 = self.snr_grid[i, j]
                f10 = self.snr_grid[i + 1, j]
                f01 = self.snr_grid[i, j + 1]
                f11 = self.snr_grid[i + 1, j + 1]
                # Bilinear interpolation
                t = (m1_val - x0) / (x1 - x0)
                u = (m2_val - y0) / (y1 - y0)
                snr[idx] = (
                    (1 - t) * (1 - u) * f00
                    + t * (1 - u) * f10
                    + (1 - t) * u * f01
                    + t * u * f11
                )
            return snr
        else:
            # Analytic approximation for a nominal distance of 400 Mpc
            # and detection threshold of 8
            nominal_distance = 400.0
            threshold = 8.0
            chirp_mass = ((m1 * m2) ** (3.0 / 5.0)) / ((m1 + m2) ** (1.0 / 5.0))
            scaling = threshold * nominal_distance / 30.0
            snr_approx = (chirp_mass / nominal_distance) * scaling
            return snr_approx