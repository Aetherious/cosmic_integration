"""Core rate calculation routines.

This module provides the high-level function
:func:`find_detection_rate` which orchestrates the entire
integration pipeline: reading COMPAS data, constructing the cosmic
star formation and metallicity histories, computing formation and
merger rates for each binary, evaluating detection probabilities
and assembling the final detection rate as a function of redshift.

To assist with these tasks a number of helper functions are
provided, each of which can be used independently if more fine
grained control is desired.  Parallel execution over binary
systems is supported via the ``n_workers`` parameter to
:func:`find_detection_rate`.

"""
from __future__ import annotations

import math
import warnings
import os
from typing import Any, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm as _NormalDist

try:
    import astropy.units as u
except Exception:
    u = None  # pragma: no cover

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

from .cosmology import get_cosmology
from .compas_data import CompasData
from .detection import (
    compute_snr_and_detection_grids,
    find_detection_probability,
    detection_probability_from_snr,
)

__all__ = [
    "calculate_redshift_related_params",
    "find_sfr",
    "find_metallicity_distribution",
    "find_formation_and_merger_rates",
    "find_detection_rate",
]


def calculate_redshift_related_params(
    max_redshift: float = 10.0,
    max_redshift_detection: float = 1.0,
    redshift_step: float = 0.001,
    z_first_sf: float = 10.0,
    cosmology: Optional[Any] = None,
) -> Tuple[np.ndarray, int, np.ndarray, float, np.ndarray, np.ndarray]:
    """Construct arrays of redshift and cosmological quantities.

    Parameters
    ----------
    max_redshift : float, optional
        Maximum redshift to include in the array.  Defaults to 10.
    max_redshift_detection : float, optional
        Maximum redshift up to which detection rates are calculated.
        Must be less than or equal to ``max_redshift``.  Defaults to 1.
    redshift_step : float, optional
        Increment in redshift.  Defaults to 0.001.
    z_first_sf : float, optional
        Redshift of the onset of star formation.  Used when
        computing merger rates.  Defaults to 10.
    cosmology : astropy.cosmology.FLRW or None, optional
        Cosmology to use.  If None the default cosmology is used.

    Returns
    -------
    tuple
        A tuple ``(redshifts, n_redshifts_detection, times, time_first_sf,
        distances, shell_volumes)`` where:

        * ``redshifts``: array of redshift values.
        * ``n_redshifts_detection``: integer index delimiting the
          redshifts used for detection (i.e., number of redshifts with
          z <= max_redshift_detection).
        * ``times``: array of cosmic ages corresponding to the redshifts
          (in Myr).
        * ``time_first_sf``: age of the universe at ``z_first_sf``.
        * ``distances``: array of luminosity distances (in Mpc).
        * ``shell_volumes``: array of comoving shell volumes (in Gpc^3)
          between adjacent redshift bins.
    """
    cosm = get_cosmology(cosmology)
    # Redshift grid
    if redshift_step <= 0:
        raise ValueError("redshift_step must be positive")
    redshifts = np.arange(0.0, max_redshift + redshift_step, redshift_step)
    if max_redshift_detection > max_redshift:
        raise ValueError("max_redshift_detection cannot exceed max_redshift")
    n_redshifts_detection = int(max_redshift_detection / redshift_step) + 1
    # Convert to cosmic ages (Myr) handling absence of astropy units
    age_obj = cosm.age(redshifts)
    if hasattr(age_obj, "to") and u is not None:
        times = age_obj.to(u.Myr).value
        time_first_sf = cosm.age(z_first_sf).to(u.Myr).value
    else:
        # Age returned in Gyr; convert to Myr
        times = np.asarray(age_obj, dtype=float) * 1e3
        time_first_sf = float(cosm.age(z_first_sf)) * 1e3
    # Luminosity distances (Mpc)
    dist_obj = cosm.luminosity_distance(redshifts)
    if hasattr(dist_obj, "to") and u is not None:
        distances = dist_obj.to(u.Mpc).value
    else:
        distances = np.asarray(dist_obj, dtype=float)
    # Avoid zero distance at z=0 to prevent divide by zero
    if len(distances) > 0:
        distances[0] = 1e-3
    # Comoving volumes (Gpc^3)
    vol_obj = cosm.comoving_volume(redshifts)
    if hasattr(vol_obj, "to") and u is not None:
        volumes = vol_obj.to(u.Gpc ** 3).value
    else:
        volumes = np.asarray(vol_obj, dtype=float)
    shell_volumes = np.diff(volumes)
    shell_volumes = np.append(shell_volumes, shell_volumes[-1])
    return redshifts, n_redshifts_detection, times, time_first_sf, distances, shell_volumes


def find_sfr(
    redshifts: np.ndarray,
    a: float = 0.01,
    b: float = 2.77,
    c: float = 2.90,
    d: float = 4.70,
) -> np.ndarray:
    """Star formation rate density as a function of redshift.

    The functional form follows the parameterisation of Madau &
    Dickinson (2014) as used in Neijssel et al. (2019).  It
    describes the cosmic star formation rate per comoving volume.

    Parameters
    ----------
    redshifts : array_like
        Redshift values at which to evaluate the star formation rate.
    a, b, c, d : float, optional
        Parameters controlling the normalisation, early time power
        law slope, turnover redshift and high redshift decline
        respectively.

    Returns
    -------
    numpy.ndarray
        Star formation rate density at each redshift in units of
        ``Msun yr^−1 Gpc^−3``.
    """
    sfr = a * ((1.0 + redshifts) ** b) / (1.0 + ((1.0 + redshifts) / c) ** d)
    # Convert to Msun/yr/Gpc^3; handle absence of astropy units
    if u is not None:
        return (sfr * u.Msun / u.yr / u.Mpc ** 3).to(u.Msun / u.yr / u.Gpc ** 3).value
    # 1 Gpc^3 = (1000 Mpc)^3 = 1e9 Mpc^3
    return sfr / 1e9


def find_metallicity_distribution(
    redshifts: np.ndarray,
    min_logz_compas: float,
    max_logz_compas: float,
    mu0: float = 0.035,
    muz: float = -0.23,
    sigma0: float = 0.39,
    sigmaz: float = 0.0,
    alpha: float = 0.0,
    min_logz: float = -12.0,
    max_logz: float = 0.0,
    step_logz: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the metallicity probability distribution over redshift.

    This routine implements the log‑skew‑normal distribution used in
    Neijssel et al. (2019) and van Son et al. (2022) to describe the
    distribution of stellar metallicities as a function of redshift.
    The distribution is normalised such that integrating over
    metallicities yields unity at each redshift.  A flat prior on
    logZ is assumed for the COMPAS sampling; the returned
    ``p_draw_metallicity`` accounts for this when converting from the
    probability of a given metallicity to the probability of drawing
    that metallicity from the simulation.

    Parameters
    ----------
    redshifts : array_like
        Redshift values at which to evaluate the metallicity
        distribution.
    min_logz_compas, max_logz_compas : float
        Minimum and maximum log metallicities sampled by the COMPAS
        simulation.  These are used to compute the probability of
        drawing a metallicity from a flat log distribution.
    mu0, muz : float, optional
        Mean (location parameter) at z=0 and its redshift evolution.
    sigma0, sigmaz : float, optional
        Width (scale parameter) at z=0 and its redshift evolution.
    alpha : float, optional
        Skewness of the log‑skew‑normal.  A value of zero
        corresponds to a log‑normal distribution.
    min_logz, max_logz, step_logz : float, optional
        Range and step size for the metallicity grid on which the
        distribution is computed.

    Returns
    -------
    tuple (dPdlogZ, metallicities, p_draw_metallicity)
        * ``dPdlogZ``: 2D array of shape (n_redshifts, n_metallicity_bins)
          giving the probability density in logZ at each redshift.
        * ``metallicities``: 1D array of metallicity values (not
          log‑space).
        * ``p_draw_metallicity``: float giving the probability of
          drawing a particular metallicity under the assumption of a
          uniform distribution in logZ in the simulation.
    """
    # Redshift dependent sigma and mean metallicity
    sigma = sigma0 * 10.0 ** (sigmaz * redshifts)
    mean_metallicities = mu0 * 10.0 ** (muz * redshifts)
    # Convert to mu parameter of log‑skew‑normal using expected value
    beta = alpha / math.sqrt(1.0 + alpha ** 2)
    phi = _NormalDist.cdf(beta * sigma)
    mu_metallicities = np.log(
        mean_metallicities / 2.0 / (np.exp(0.5 * sigma ** 2) * phi)
    )
    # Grid of log metallicities
    log_metallicities = np.arange(min_logz, max_logz + step_logz, step_logz)
    metallicities = np.exp(log_metallicities)
    # Evaluate log‑skew‑normal distribution (without 1/Z factor)
    dPdlogZ = (
        2.0
        / sigma[:, np.newaxis]
        * _NormalDist.pdf((log_metallicities - mu_metallicities[:, np.newaxis]) / sigma[:, np.newaxis])
        * _NormalDist.cdf(alpha * (log_metallicities - mu_metallicities[:, np.newaxis]) / sigma[:, np.newaxis])
    )
    # Normalise over metallicity dimension
    norm = dPdlogZ.sum(axis=1) * step_logz
    dPdlogZ = dPdlogZ / norm[:, np.newaxis]
    # Probability of drawing a metallicity from a flat log distribution
    p_draw_metallicity = 1.0 / (max_logz_compas - min_logz_compas)
    return dPdlogZ, metallicities, p_draw_metallicity


def _compute_single_formation_merger_row(
    i: int,
    n_formed: np.ndarray,
    dPdlogZ: Any,
    metallicities: Optional[np.ndarray],
    p_draw_metallicity: float,
    compas_metallicities: np.ndarray,
    compas_delay_times: np.ndarray,
    times: np.ndarray,
    time_first_sf: float,
    redshifts: np.ndarray,
    redshift_step: float,
    compas_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper computing formation and merger rates for a single binary.

    The arguments correspond to slices of the arrays in
    :func:`find_formation_and_merger_rates` and are passed verbatim
    when parallelising over binaries.
    """
    n_redshifts = len(redshifts)
    formation_rate_row = np.zeros(n_redshifts, dtype=float)
    merger_rate_row = np.zeros(n_redshifts, dtype=float)
    # Formation rate either uniform or weighted by metallicity distribution
    if metallicities is None:
        formation_rate_row[:] = n_formed * compas_weights[i]
    else:
        # Find index of closest metallicity for this system
        idx = np.digitize(compas_metallicities[i], metallicities)
        formation_rate_row[:] = n_formed * dPdlogZ[:, idx] / p_draw_metallicity * compas_weights[i]
    # Time of formation given delay time
    time_of_formation = times - compas_delay_times[i]
    # Only consider formation times after star formation begins
    first_valid = np.digitize(time_first_sf, time_of_formation)
    # Adjust index if it goes out of range
    if first_valid == n_redshifts:
        first_valid = n_redshifts - 1
    # Convert times to redshifts via interpolation (times is monotonic decreasing with redshift)
    times_to_redshifts = interp1d(times, redshifts, bounds_error=False, fill_value=(redshifts[0], redshifts[-1]))
    if first_valid > 0:
        z_of_formation = times_to_redshifts(time_of_formation[: first_valid - 1])
        # Determine indices in redshift grid
        z_indices = np.ceil(z_of_formation / redshift_step).astype(int)
        # Map formation redshifts to merger redshift indices
        merger_rate_row[: first_valid - 1] = formation_rate_row[z_indices]
    return formation_rate_row, merger_rate_row


def find_formation_and_merger_rates(
    n_binaries: int,
    redshifts: np.ndarray,
    times: np.ndarray,
    time_first_sf: float,
    n_formed: np.ndarray,
    dPdlogZ: Any,
    metallicities: Optional[np.ndarray],
    p_draw_metallicity: float,
    compas_metallicities: np.ndarray,
    compas_delay_times: np.ndarray,
    compas_weights: np.ndarray,
    n_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute formation and merger rate matrices for all binaries.

    Parameters
    ----------
    n_binaries : int
        Number of binaries in the simulation.
    redshifts : array_like
        Redshift values over which to evaluate the rates.
    times : array_like
        Cosmic ages corresponding to the redshifts.
    time_first_sf : float
        Age of the universe at which star formation begins.
    n_formed : array_like
        Star forming mass per year per cubic Gpc divided by the
        representative mass per binary, yielding the number of
        binaries formed per year per Gpc^3.
    dPdlogZ : array_like or scalar
        2D array giving the probability of a metallicity at each
        redshift, or a scalar if metallicity weighting is not used.
    metallicities : array_like or None
        Metallicities at which ``dPdlogZ`` is evaluated.  If
        ``None`` metallicity weighting is disabled.
    p_draw_metallicity : float
        Probability of drawing a particular metallicity from a flat
        log distribution.
    compas_metallicities : array_like
        Metallicities of the binaries in the simulation.
    compas_delay_times : array_like
        Delay times (formation + coalescence) of the binaries.
    compas_weights : array_like
        Adaptive sampling weights for each binary (unity if not
        provided).
    n_workers : int, optional
        Number of worker processes to use for parallel execution.  If
        less than two the computation is performed sequentially.

    Returns
    -------
    tuple (formation_rate, merger_rate)
        Each array has shape (n_binaries, n_redshifts) giving the
        formation or merger rate of every binary as a function of
        redshift.

    Notes
    -----
    When ``n_workers`` is greater than one the computation is
    performed in parallel using ``joblib``.  This can provide
    significant speedups for large binary populations at the cost
    of increased memory consumption.
    """
    n_redshifts = len(redshifts)
    formation_rate = np.zeros((n_binaries, n_redshifts), dtype=float)
    merger_rate = np.zeros((n_binaries, n_redshifts), dtype=float)
    redshift_step = redshifts[1] - redshifts[0] if len(redshifts) > 1 else 1.0
    if n_workers and n_workers > 1 and Parallel is not None:
        # Prepare arguments for each binary index
        tasks = [
            delayed(_compute_single_formation_merger_row)(
                i,
                n_formed,
                dPdlogZ,
                metallicities,
                p_draw_metallicity,
                compas_metallicities,
                compas_delay_times,
                times,
                time_first_sf,
                redshifts,
                redshift_step,
                compas_weights,
            )
            for i in range(n_binaries)
        ]
        # Execute in parallel; prefer 'loky' backend for process based parallelism
        results = Parallel(n_jobs=n_workers, backend="loky")(tasks)
        for i, (form_row, merge_row) in enumerate(results):
            formation_rate[i] = form_row
            merger_rate[i] = merge_row
    else:
        # Sequential execution
        for i in range(n_binaries):
            form_row, merge_row = _compute_single_formation_merger_row(
                i,
                n_formed,
                dPdlogZ,
                metallicities,
                p_draw_metallicity,
                compas_metallicities,
                compas_delay_times,
                times,
                time_first_sf,
                redshifts,
                redshift_step,
                compas_weights,
            )
            formation_rate[i] = form_row
            merger_rate[i] = merge_row
    return formation_rate, merger_rate


def find_detection_rate(
    path: str,
    dco_type: str = "BBH",
    merger_output_filename: Optional[str] = None,
    weight_column: Optional[str] = None,
    merges_hubble_time: bool = True,
    pessimistic_cee: bool = True,
    no_rlof_after_cee: bool = True,
    max_redshift: float = 10.0,
    max_redshift_detection: float = 1.0,
    redshift_step: float = 0.001,
    z_first_sf: float = 10.0,
    use_sampled_mass_ranges: bool = True,
    m1_min: float = 5.0,
    m1_max: float = 150.0,
    m2_min: float = 0.1,
    fbin: float = 0.7,
    a_sf: float = 0.01,
    b_sf: float = 2.77,
    c_sf: float = 2.90,
    d_sf: float = 4.70,
    mu0: float = 0.035,
    muz: float = -0.23,
    sigma0: float = 0.39,
    sigmaz: float = 0.0,
    alpha: float = 0.0,
    min_logz: float = -12.0,
    max_logz: float = 0.0,
    step_logz: float = 0.01,
    sensitivity: str = "O1",
    snr_threshold: float = 8.0,
    mc_max: float = 300.0,
    mc_step: float = 0.1,
    eta_max: float = 0.25,
    eta_step: float = 0.01,
    snr_max: float = 1000.0,
    snr_step: float = 0.1,
    cosmology: Optional[Any] = None,
    n_workers: int = 1,
    save_npz: bool = True,
    output_filename: str = "results.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CompasData]:
    """High level wrapper to compute detection, formation and merger rates.

    Parameters
    ----------
    path : str
        Path to the COMPAS HDF5 output file.
    dco_type : {'BBH', 'BHNS', 'BNS', 'all'}, optional
        Type of double compact objects to include.  Defaults to 'BBH'.
    merger_output_filename : str or None, optional
        If provided the merger rate matrix will be saved to a text
        file with this name in the same directory as ``path``.  The
        file will list the component masses, merger redshift and
        merger rate for each system.  Defaults to ``None`` (no
        output).
    weight_column : str or None, optional
        Name of the column in ``BSE_Double_Compact_Objects`` that
        contains adaptive sampling weights.  If ``None`` unity
        weights are assumed.  Defaults to ``None``.
    merges_hubble_time, pessimistic_cee, no_rlof_after_cee : bool, optional
        Flags controlling masking of binaries.  See
        :meth:`CompasData.set_dco_mask` for details.  Defaults are
        True, True and True respectively.
    max_redshift, max_redshift_detection, redshift_step, z_first_sf : float, optional
        Parameters controlling the redshift grid.  See
        :func:`calculate_redshift_related_params`.  Note that
        ``max_redshift_detection`` must not exceed ``max_redshift``.
    use_sampled_mass_ranges : bool, optional
        If True the minimum and maximum primary and secondary masses
        are derived from the data rather than taken from the
        ``m1_min``, ``m1_max`` and ``m2_min`` arguments.  Defaults to
        True.
    m1_min, m1_max, m2_min : float, optional
        Bounds on the sampled masses.  Only used when
        ``use_sampled_mass_ranges`` is False.  Defaults correspond
        roughly to typical black hole ranges.
    fbin : float, optional
        Binary fraction used in the sampling.  Defaults to 0.7.
    a_sf, b_sf, c_sf, d_sf : float, optional
        Parameters controlling the star formation rate density.  See
        :func:`find_sfr`.  Defaults follow Neijssel et al. (2019).
    mu0, muz, sigma0, sigmaz, alpha : float, optional
        Parameters controlling the metallicity distribution.  See
        :func:`find_metallicity_distribution`.
    min_logz, max_logz, step_logz : float, optional
        Range and resolution of the metallicity grid.  Defaults span
        from 10^{−12} to 1.
    sensitivity : {'design', 'O1', 'O3'}, optional
        Detector sensitivity for the SNR grid.  Defaults to 'O1'.
    snr_threshold : float, optional
        Detection threshold on the SNR.  Defaults to 8.0.
    mc_max, mc_step, eta_max, eta_step, snr_max, snr_step : float, optional
        Parameters controlling the SNR and detection probability grids.
    cosmology : astropy.cosmology.FLRW or None, optional
        Cosmological model.  See :func:`get_cosmology`.
    n_workers : int, optional
        Number of processes to use for parallel computations.  If
        greater than one and joblib is available the formation and
        merger rates will be computed in parallel.  Defaults to 1
        (serial execution).
    save_npz : bool, optional
        If True the detection, formation and merger rate matrices
        together with the redshift array are saved in a numpy
        compressed file with name ``output_filename``.  Defaults to True.
    output_filename : str, optional
        Filename (without directory) for the saved numpy archive.  The
        file will be created in the same directory as ``path``.
        Defaults to "results.npz".

    Returns
    -------
    tuple (detection_rate, formation_rate, merger_rate, redshifts, compas)
        The three rate matrices (each of shape (n_binaries, n_redshifts))
        the array of redshifts and the :class:`CompasData` instance
        used to load the simulation.

    Raises
    ------
    AssertionError
        If any of the input sanity checks fail (e.g., negative
        values).

    Notes
    -----
    The function prints a few progress messages to standard output
    detailing the representative star forming mass per binary and
    any warnings about the sampled mass ranges.
    """
    # Set up cosmology early to ensure astropy units are available
    cosm = get_cosmology(cosmology)
    # Input sanity checks
    assert max_redshift_detection <= max_redshift, "max_redshift_detection cannot exceed max_redshift"
    assert mc_step < mc_max, "mc_step must be less than mc_max"
    assert eta_step < eta_max, "eta_step must be less than eta_max"
    assert snr_step < snr_max, "snr_step must be less than snr_max"
    # Negative parameter checks
    for val, name in [
        (max_redshift, "max_redshift"),
        (max_redshift_detection, "max_redshift_detection"),
        (m1_min, "m1_min"),
        (m1_max, "m1_max"),
        (m2_min, "m2_min"),
        (mu0, "mu0"),
        (sigma0, "sigma0"),
        (step_logz, "step_logz"),
        (snr_threshold, "snr_threshold"),
        (mc_max, "mc_max"),
        (mc_step, "mc_step"),
        (eta_max, "eta_max"),
        (eta_step, "eta_step"),
        (snr_max, "snr_max"),
        (snr_step, "snr_step"),
    ]:
        if val < 0.0:
            raise ValueError(f"{name} must be non‑negative")
    # Instantiate data container and apply masks
    compas = CompasData(path, m1_min if not use_sampled_mass_ranges else None, m1_max if not use_sampled_mass_ranges else None, m2_min if not use_sampled_mass_ranges else None, fbin)
    compas.set_dco_mask(types=dco_type, within_hubble_time=merges_hubble_time, pessimistic=pessimistic_cee, no_rlof_after_cee=no_rlof_after_cee)
    compas.load()
    compas.set_weights(weight_column)
    # Derive sampled mass ranges from data if requested
    if use_sampled_mass_ranges:
        # Use masses from the Double Compact Objects table
        m1_all = compas.mass1
        m2_all = compas.mass2
        # Exclude equal masses due to RLOF at ZAMS when inferring m1_min
        m1_mask = m1_all[m1_all != m2_all]
        compas.m1_min = float(np.min(m1_mask)) if len(m1_mask) > 0 else float(np.min(m1_all))
        compas.m1_max = float(np.max(m1_all))
        compas.m2_min = float(np.min(m2_all))
    # Compute star forming mass per binary
    compas.find_star_forming_mass_per_binary_sampling()
    # Basic warnings
    chirp_masses = (compas.mass1_masked * compas.mass2_masked) ** (3.0 / 5.0) / (compas.mass1_masked + compas.mass2_masked) ** (1.0 / 5.0)
    max_observed_mc = np.max(chirp_masses) * (1.0 + max_redshift_detection)
    if max_observed_mc > mc_max:
        warnings.warn(
            "Maximum chirp mass on grid is below max observed chirp mass * (1+z). Increase mc_max for accuracy.",
            RuntimeWarning,
        )
    # Construct redshift arrays
    redshifts, n_redshifts_detection, times, time_first_sf, distances, shell_volumes = calculate_redshift_related_params(
        max_redshift, max_redshift_detection, redshift_step, z_first_sf, cosmology
    )
    # Star formation rate density (Msun/yr/Gpc^3)
    sfr = find_sfr(redshifts, a_sf, b_sf, c_sf, d_sf)
    # Convert to number of binaries formed per year per Gpc^3
    # Representative star forming mass per binary stored on compas.mass_evolved_per_binary
    n_formed = sfr / compas.mass_evolved_per_binary
    # Metallicities
    if np.log(np.min(compas.metallicity_masked)) != np.log(np.max(compas.metallicity_masked)):
        dPdlogZ, metallicities, p_draw_metallicity = find_metallicity_distribution(
            redshifts,
            min_logz_compas=float(np.log(np.min(compas.metallicity_masked))),
            max_logz_compas=float(np.log(np.max(compas.metallicity_masked))),
            mu0=mu0,
            muz=muz,
            sigma0=sigma0,
            sigmaz=sigmaz,
            alpha=alpha,
            min_logz=min_logz,
            max_logz=max_logz,
            step_logz=step_logz,
        )
    else:
        metallicities = None
        dPdlogZ = 1.0
        p_draw_metallicity = 1.0
    # Formation and merger rates
    formation_rate, merger_rate = find_formation_and_merger_rates(
        n_binaries=len(chirp_masses),
        redshifts=redshifts,
        times=times,
        time_first_sf=time_first_sf,
        n_formed=n_formed,
        dPdlogZ=dPdlogZ,
        metallicities=metallicities,
        p_draw_metallicity=p_draw_metallicity,
        compas_metallicities=compas.metallicity_masked,
        compas_delay_times=compas.delay_times_masked,
        compas_weights=compas.sw_weights,
        n_workers=n_workers,
    )
    # SNR and detection probability grids
    snr_grid_at_1mpc, detection_probability_from_snr_grid = compute_snr_and_detection_grids(
        sensitivity=sensitivity,
        snr_threshold=snr_threshold,
        mc_max=mc_max,
        mc_step=mc_step,
        eta_max=eta_max,
        eta_step=eta_step,
        snr_max=snr_max,
        snr_step=snr_step,
    )
    # Detection probabilities per binary and redshift
    etas = compas.mass1_masked * compas.mass2_masked / (compas.mass1_masked + compas.mass2_masked) ** 2
    detection_probability = find_detection_probability(
        mc=chirp_masses,
        eta=etas,
        redshifts=redshifts,
        distances=distances,
        n_redshifts_detection=n_redshifts_detection,
        n_binaries=len(chirp_masses),
        snr_grid_at_1mpc=snr_grid_at_1mpc,
        detection_probability_from_snr_grid=detection_probability_from_snr_grid,
        mc_step=mc_step,
        eta_step=eta_step,
        snr_step=snr_step,
    )
    # Final detection rate (per year) using Neijssel+19 Eq. 2
    detection_rate = np.zeros((len(chirp_masses), n_redshifts_detection), dtype=float)
    # Multiply merger rate by detection probability and volume factor
    detection_rate[:, :] = merger_rate[:, :n_redshifts_detection] * detection_probability * shell_volumes[:n_redshifts_detection] / (
        1.0 + redshifts[:n_redshifts_detection]
    )
    # Optionally save results
    if save_npz:
        output_path = os.path.join(os.path.dirname(path), output_filename)
        np.savez_compressed(
            output_path,
            detection_rate=detection_rate,
            formation_rate=formation_rate,
            merger_rate=merger_rate,
            redshifts=redshifts,
        )
    # Optional merger output file
    if merger_output_filename:
        merge_path = os.path.join(os.path.dirname(path), merger_output_filename)
        with open(merge_path, "w", encoding="utf-8") as out:
            out.write("Mass1atMerger\tMass2atMerger\tMergerRedshift\tMergerRate\n")
            out.write("Msun\tMsun\t--\tGpc^{-3} yr^{-1}\n")
            # List only non‑zero merger rates up to n_redshifts_detection
            for i in range(n_redshifts_detection):
                for j in range(len(chirp_masses)):
                    if merger_rate[j, i] > 0.0:
                        out.write(
                            f"{compas.mass1_masked[j]:.5f}\t{compas.mass2_masked[j]:.5f}\t{redshifts[i]:.5f}\t{merger_rate[j, i]:.10f}\n"
                        )
    return detection_rate, formation_rate, merger_rate, redshifts, compas
