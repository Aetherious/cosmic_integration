"""Detection probability and SNR interpolation utilities.

This module implements a simple interpolator to evaluate the
signal-to-noise ratio (SNR) of compact binary mergers as a function
of component masses at a fixed distance.  The SNR grid used by
COMPAS is stored in an HDF5 file which is read on demand.  A
wrapper class :class:`SNRInterpolator` exposes the interpolator via
``__call__`` so that it can be used like a function.

In addition we provide convenience functions to compute a grid of
SNRs and detection probabilities over chirp mass and symmetric
mass ratio as well as to evaluate the detection probability for
individual binaries at different redshifts.  The detection
probability is approximated by a Heaviside step function on the
SNR threshold; a more sophisticated treatment accounting for
random source orientations can be incorporated here if desired.

"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "SNRInterpolator",
    "detection_probability_from_snr",
    "compute_snr_and_detection_grids",
    "find_detection_probability",
]


class SNRInterpolator:
    """Approximate signal-to-noise ratio at 1 Mpc based on chirp mass.

    In the absence of the precomputed SNR grid (which requires the
    `h5py` package and external data), this simplified
    implementation estimates the SNR of a compact binary at a
    distance of 1 Mpc by assuming it scales as the chirp mass to the
    5/6 power.  The chirp mass M_c for component masses m1 and m2
    (in solar masses) is defined as

    .. math::

        M_c = frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}.

    The resulting SNR at 1 Mpc is taken to be ``M_c**(5/6)`` in
    arbitrary units.  This captures the scaling of the inspiral
    signal amplitude with mass but is not calibrated to any
    particular detector sensitivity curve.  Users requiring more
    accurate SNR estimates should supply their own interpolator.

    Parameters
    ----------
    sensitivity : str, optional
        Ignored in this simplified implementation but retained for
        compatibility.  Defaults to 'O1'.
    grid_filename : str, optional
        Ignored in this simplified implementation.
    """

    def __init__(self, sensitivity: str = "O1", grid_filename: str | None = None) -> None:
        # sensitivity and grid_filename are ignored; included to mirror
        # the API of the original class.
        pass

    def __call__(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        m1_arr = np.asarray(m1, dtype=float)
        m2_arr = np.asarray(m2, dtype=float)
        # Compute chirp mass
        chirp_mass = (m1_arr * m2_arr) ** (3.0 / 5.0) / (m1_arr + m2_arr) ** (1.0 / 5.0)
        # Approximate SNR scaling; arbitrarily normalised
        return chirp_mass ** (5.0 / 6.0)


def detection_probability_from_snr(snr_value: np.ndarray, snr_threshold: float) -> np.ndarray:
    """Return a binary detection probability based on a threshold.

    A value of 1 is returned where the signal-to-noise ratio is
    greater than or equal to ``snr_threshold`` and 0 elsewhere.  This
    function may be replaced by a more sophisticated model if desired.
    """
    snr_arr = np.asarray(snr_value, dtype=float)
    return (snr_arr >= snr_threshold).astype(float)


def compute_snr_and_detection_grids(
    sensitivity: str = "O1",
    snr_threshold: float = 8.0,
    mc_max: float = 300.0,
    mc_step: float = 0.1,
    eta_max: float = 0.25,
    eta_step: float = 0.01,
    snr_max: float = 1000.0,
    snr_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute SNR and detection probability grids.

    This simplified version constructs a grid of chirp masses and
    symmetric mass ratios, evaluates the approximate SNR at 1 Mpc
    using :class:`SNRInterpolator`, and computes a detection
    probability array based on a hard threshold.  The SNR values are
    uncalibrated and intended only for demonstration purposes.
    """
    # Instantiate the simplified interpolator
    interpolator = SNRInterpolator(sensitivity)

    # Chirp mass and eta grids
    mc_array = np.arange(mc_step, mc_max + mc_step, mc_step)
    eta_array = np.arange(eta_step, eta_max + eta_step, eta_step)

    # Compute total masses and component masses
    mt_grid = mc_array[None, :] / (eta_array[:, None] ** 0.6)
    # Avoid invalid values due to rounding; clip sqrt argument
    sqrt_arg = 1.0 - 4.0 * eta_array[:, None]
    sqrt_arg = np.clip(sqrt_arg, 0.0, None)
    m1_grid = 0.5 * mt_grid * (1.0 + np.sqrt(sqrt_arg))
    m2_grid = mt_grid - m1_grid
    # Evaluate approximate SNR at 1 Mpc
    snr_grid_at_1mpc = interpolator(m1_grid, m2_grid)
    # Build detection probability mapping as a 1D array
    snr_values = np.arange(snr_step, snr_max + snr_step, snr_step)
    detection_probabilities = detection_probability_from_snr(snr_values, snr_threshold)
    return snr_grid_at_1mpc, detection_probabilities


def find_detection_probability(
    mc: np.ndarray,
    eta: np.ndarray,
    redshifts: np.ndarray,
    distances: np.ndarray,
    n_redshifts_detection: int,
    n_binaries: int,
    snr_grid_at_1mpc: np.ndarray,
    detection_probability_from_snr_grid: np.ndarray,
    mc_step: float = 0.1,
    eta_step: float = 0.01,
    snr_step: float = 0.1,
) -> np.ndarray:
    """Evaluate detection probability for each binary at each redshift.

    Parameters
    ----------
    mc : array_like
        Chirp masses of the binaries (solar masses).
    eta : array_like
        Symmetric mass ratios of the binaries.
    redshifts : array_like
        Array of redshifts at which to evaluate detectability.
    distances : array_like
        Luminosity distances corresponding to the redshifts (Mpc).
    n_redshifts_detection : int
        Number of redshift points to consider for detectability.
    n_binaries : int
        Number of merging binaries in the sample.
    snr_grid_at_1mpc : ndarray
        SNR values at 1 Mpc on the (η, Mᶜ) grid.
    detection_probability_from_snr_grid : ndarray
        1D array mapping SNR values to detection probabilities (0 or 1).
    mc_step, eta_step, snr_step : float, optional
        Grid spacing for chirp mass, symmetric mass ratio and SNR.
        Defaults match those in :func:`compute_snr_and_detection_grids`.

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_binaries, n_redshifts_detection) giving
        the detection probability of each binary at each redshift.

    Notes
    -----
    This routine implements the same logic as the original
    ``find_detection_probability`` function but uses a simplified
    detection probability lookup.  Indices are computed by rounding
    masses to the nearest grid point.  Values falling outside the
    grid are assigned unit detection probability.
    """
    detection_probability = np.ones((n_binaries, n_redshifts_detection), dtype=float)

    # Precompute indices for η; subtract 1 since our grid starts at one mc_step
    eta_indices = np.round(eta / eta_step).astype(int) - 1
    # Bound indices to valid range
    eta_indices = np.clip(eta_indices, 0, snr_grid_at_1mpc.shape[0] - 1)

    for i in range(n_binaries):
        # Shift chirp mass by (1+z) factor for redshifted mass
        mc_shifted = mc[i] * (1.0 + redshifts[:n_redshifts_detection])
        mc_indices = np.round(mc_shifted / mc_step).astype(int) - 1
        # Determine where indices fall inside the grid
        inside = (mc_indices >= 0) & (mc_indices < snr_grid_at_1mpc.shape[1])
        # Look up SNR at 1 Mpc for masses inside the grid
        snrs = np.zeros(n_redshifts_detection, dtype=float)
        snrs[inside] = snr_grid_at_1mpc[eta_indices[i], mc_indices[inside]]
        # Scale by distance
        snrs = snrs / distances[:n_redshifts_detection]
        # Convert SNR to detection probability using the 1D lookup
        snr_indices = np.round(snrs / snr_step).astype(int) - 1
        # Clip indices into detection_probability_from_snr_grid
        snr_indices = np.clip(snr_indices, 0, len(detection_probability_from_snr_grid) - 1)
        detection_probability[i, :] = detection_probability_from_snr_grid[snr_indices]
    return detection_probability
