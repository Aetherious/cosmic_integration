"""Core routines for computing compact binary formation, merger and detection rates.

This module contains a simplified yet physically motivated re‑implementation of
the primary functions offered by the original cosmic integration codebase.
The focus here is on readability and maintainability rather than absolute
performance or exhaustive feature parity.  Functions are designed to be
composable and easy to test.  Wherever sensible vectorised numpy
operations are used; for very large populations the work can be
distributed across multiple CPU cores via the standard library
``concurrent.futures`` interface.

Example
-------

Run from Python:

>>> from cosmic_integration_dasein.core import find_detection_rate
>>> result = find_detection_rate("/path/to/compas_data.npz")
>>> print(result['redshift'])
array([...])

Note
----

This implementation assumes access to an HDF5 or NPZ file containing at
minimum the primary and secondary masses of each binary.  When reading
HDF5 data the :mod:`h5py` package must be available; otherwise a
:class:`RuntimeError` is raised.  The function accepts a number of
optional parameters controlling the star formation history, metallicity
distribution, mass filtering and detection threshold.  See the
parameter list of :func:`find_detection_rate` for details.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from astropy import units as u  # type: ignore
except Exception:
    u = None  # type: ignore

try:
    from astropy.cosmology import FLRW  # type: ignore
except Exception:
    FLRW = object  # type: ignore

from .cosmology_utils import get_cosmology


@dataclass
class PopulationData:
    """A simple container for the minimal binary population parameters.

    Parameters
    ----------
    m1 : numpy.ndarray
        Array of primary masses in units of solar masses.
    m2 : numpy.ndarray
        Array of secondary masses in units of solar masses.
    weights : numpy.ndarray
        Array of sampling weights assigned to each binary.  If no
        weights are specified in the input file a unit weight is used.
    delay_times : numpy.ndarray
        Array of delay times in Myr.  When not available the delays
        default to zero implying the merger occurs immediately.
    metallicities : numpy.ndarray, optional
        Array of metallicities of each binary.  This field is only used
        when calculating metallicity dependent quantities.  If no
        metallicities are provided the star formation rate and
        detection rate become independent of metallicity.
    """

    m1: np.ndarray
    m2: np.ndarray
    weights: np.ndarray
    delay_times: np.ndarray
    metallicities: Optional[np.ndarray] = None

    def filter_by_mass(self, m1_min: float, m1_max: float, m2_min: float) -> None:
        """Filter the population in place based on component masses.

        Binaries with primary mass less than ``m1_min`` or greater than
        ``m1_max`` are removed.  Binaries with secondary mass less
        than ``m2_min`` are also removed.  The filtering is applied
        consistently across all stored fields.

        Parameters
        ----------
        m1_min, m1_max, m2_min : float
            Mass thresholds expressed in solar masses.
        """
        mask = (self.m1 >= m1_min) & (self.m1 <= m1_max) & (self.m2 >= m2_min)
        self.m1 = self.m1[mask]
        self.m2 = self.m2[mask]
        self.weights = self.weights[mask]
        self.delay_times = self.delay_times[mask]
        if self.metallicities is not None:
            self.metallicities = self.metallicities[mask]


def _guess_dataset(array_names: Iterable[str], desired: Iterable[str]) -> Optional[str]:
    """Return the first matching name from ``array_names`` that contains any of the
    substrings in ``desired``.

    This helper is used when attempting to infer dataset names from a
    generic HDF5 file.  The substrings in ``desired`` should be lower
    case.  The candidate names are lowered before testing.
    """

    lower_to_name = {name.lower(): name for name in array_names}
    for substr in desired:
        for lower, name in lower_to_name.items():
            if substr in lower:
                return name
    return None


def load_population(path: str, weight_column: Optional[str] = None) -> PopulationData:
    """Load binary population data from an HDF5 or NPZ file.

    The input file must contain arrays of primary and secondary masses.  A
    sampling weight array and delay time array are optional.  When the
    file is an HDF5 the :mod:`h5py` package is required.  For NPZ
    archives the arrays should be stored as top level variables with
    obvious names (e.g. ``m1``, ``m2``, ``weights``, ``delay_times``).

    Parameters
    ----------
    path : str
        Path to the data file.  The file extension determines how the
        file is parsed.  ``.npz`` and ``.npy`` files are read with
        :func:`numpy.load`.  All other extensions are assumed to be
        HDF5 and are opened with :mod:`h5py`.
    weight_column : str, optional
        Name of a dataset within the file holding sampling weights.  If
        ``None`` (default) unit weights are assumed.

    Returns
    -------
    PopulationData
        An object containing the primary and secondary masses, weights
        and delay times of the binaries.

    Raises
    ------
    RuntimeError
        If the file type is HDF5 but the :mod:`h5py` module cannot be
        imported.
    ValueError
        If the required datasets cannot be located in the HDF5 file.
    """

    path = str(path)
    # NPZ or NPY: straightforward loading
    if path.endswith((".npz", ".npy")):
        data = np.load(path, allow_pickle=True)
        # Support either npz where arrays stored under keys or a single numpy array
        if isinstance(data, np.ndarray):
            raise ValueError("A .npy file must contain a structured array with fields m1 and m2")
        # find keys for m1 and m2
        keys = [k for k in data.keys()]
        m1_name = _guess_dataset(keys, ["mass1", "m1", "primary"])
        m2_name = _guess_dataset(keys, ["mass2", "m2", "secondary"])
        if m1_name is None or m2_name is None:
            raise ValueError("Could not locate m1 and m2 in npz file")
        m1 = np.asarray(data[m1_name], dtype=float)
        m2 = np.asarray(data[m2_name], dtype=float)
        if weight_column is not None and weight_column in data:
            weights = np.asarray(data[weight_column], dtype=float)
        else:
            # look for generic weight names
            wname = _guess_dataset(keys, ["weight", "weights", "sw_weight"])
            weights = np.asarray(data[wname], dtype=float) if wname is not None else np.ones_like(m1)
        # delay times
        dt_name = _guess_dataset(keys, ["delay", "delay_time", "delaytimes"])
        delay_times = np.asarray(data[dt_name], dtype=float) if dt_name is not None else np.zeros_like(m1)
        # metallicity
        zmet_name = _guess_dataset(keys, ["metallicity", "z"])
        metallicities = np.asarray(data[zmet_name], dtype=float) if zmet_name is not None else None
        return PopulationData(m1=m1, m2=m2, weights=weights, delay_times=delay_times, metallicities=metallicities)

    # Attempt to load HDF5
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read HDF5 files. Install h5py or provide an npz file."
        ) from exc

    with h5py.File(path, "r") as h5file:
        # Flatten all dataset names into a single list (use names with '/').
        def walk(name, obj, names):
            if isinstance(obj, h5py.Dataset):
                names.append(name)
        dataset_names: list[str] = []
        h5file.visititems(lambda name, obj: walk(name, obj, dataset_names))
        m1_name = _guess_dataset(dataset_names, ["mass1", "m1", "primary"])
        m2_name = _guess_dataset(dataset_names, ["mass2", "m2", "secondary"])
        if m1_name is None or m2_name is None:
            raise ValueError("Could not locate m1 and m2 in HDF5 file")
        m1 = np.asarray(h5file[m1_name], dtype=float)
        m2 = np.asarray(h5file[m2_name], dtype=float)
        # Weights
        if weight_column is not None and weight_column in h5file:
            weights = np.asarray(h5file[weight_column], dtype=float)
        else:
            wname = _guess_dataset(dataset_names, ["weight", "weights", "sw_weight"])
            weights = np.asarray(h5file[wname], dtype=float) if wname is not None else np.ones_like(m1)
        # Delay times (in Myr).  If absent we set zero delay.
        dt_name = _guess_dataset(dataset_names, ["delay", "delay_time", "delaytimes"])
        delay_times = np.asarray(h5file[dt_name], dtype=float) if dt_name is not None else np.zeros_like(m1)
        # Metallicities
        zmet_name = _guess_dataset(dataset_names, ["metallicity", "z"])
        metallicities = np.asarray(h5file[zmet_name], dtype=float) if zmet_name is not None else None
        return PopulationData(m1=m1, m2=m2, weights=weights, delay_times=delay_times, metallicities=metallicities)


def star_formation_rate(z: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Return the cosmic star formation rate density as a function of redshift.

    The functional form used here follows the piecewise fit presented
    in Madau & Dickinson (2014), commonly employed in binary population
    synthesis studies.  Parameters ``a``, ``b``, ``c`` and ``d`` control
    the amplitude and shape of the star formation history.

    SFR(z) ∝ a (1 + z)^b / (1 + ((1 + z)/c)^d)

    Parameters
    ----------
    z : numpy.ndarray
        Array of redshift values.
    a, b, c, d : float
        Star formation history parameters.

    Returns
    -------
    numpy.ndarray
        Star formation rate density at each redshift (arbitrary units).
    """
    z = np.asarray(z, dtype=float)
    numerator = a * np.power(1.0 + z, b)
    denominator = 1.0 + np.power((1.0 + z) / c, d)
    return numerator / denominator


def metallicity_distribution(
    z: np.ndarray,
    mu0: float,
    muz: float,
    sigma0: float,
    sigmaz: float,
    alpha: float = 0.0,
    logZ_min: float = -12.0,
    logZ_max: float = 0.0,
    logZ_step: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a log‑normal metallicity distribution for each redshift.

    This function discretises the metallicity axis into bins and
    evaluates a log‑normal probability density function within each
    bin.  The mean and standard deviation of the distribution evolve
    linearly with redshift via ``mu0 + muz * z`` and ``sigma0 + sigmaz * z``.
    The optional ``alpha`` parameter may be used to tilt the
    distribution but is not employed in this implementation.  The
    returned metallicity values are in absolute (non‑logarithmic) units.

    Parameters
    ----------
    z : numpy.ndarray
        Array of redshifts at which to evaluate the distribution.
    mu0, muz, sigma0, sigmaz, alpha : float
        Parameters controlling the mean and width of the log‑normal
        distribution as a function of redshift.
    logZ_min, logZ_max, logZ_step : float
        Definition of the log‑metallicity grid used for integration.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the metallicity grid (in absolute units)
        and a 2D array of shape (len(z), len(logZ_grid)) giving the
        normalised probability density of metallicity at each redshift.
    """
    logZ_grid = np.arange(logZ_min, logZ_max + logZ_step / 2.0, logZ_step)
    Z_grid = np.power(10.0, logZ_grid)
    pdf = np.empty((len(z), len(Z_grid)))
    for i, zz in enumerate(z):
        mu = mu0 + muz * zz
        sigma = sigma0 + sigmaz * zz
        # Evaluate log‑normal PDF
        lnZ = logZ_grid
        exponent = -0.5 * np.square((lnZ - mu) / sigma)
        raw = np.exp(exponent)
        raw /= raw.sum()
        pdf[i] = raw
    return Z_grid, pdf


def detection_probability(
    m1: np.ndarray,
    m2: np.ndarray,
    z: np.ndarray,
    cosmology: FLRW,
    snr_threshold: float = 8.0,
    snr_scale: float = 1.0,
    width: float = 1.0,
) -> np.ndarray:
    """Compute the detection probability of each binary at each redshift.

    The signal‑to‑noise ratio (SNR) of a compact binary coalescence in a
    ground based detector scales with the chirp mass and inverse of the
    luminosity distance.  A rough model for the SNR is used here:

    SNR(m1, m2, z) = snr_scale × [M_chirp × (1 + z)]^(5/6) / D_L(z)

    where the chirp mass

    M_chirp = (m1 m2)^(3/5) / (m1 + m2)^(1/5)

    and D_L(z) is the luminosity distance in megaparsecs obtained from the
    supplied cosmology.  This formulation captures the redshifted mass
    enhancement and cosmological dimming of the signal.

    To translate SNR into a detection probability a smooth logistic
    function is applied centred on ``snr_threshold`` with width
    ``width``.  The resulting probabilities lie between 0 and 1 and
    approximate the behaviour of matched filter searches where triggers
    become increasingly confident as the SNR rises above threshold.

    Parameters
    ----------
    m1, m2 : numpy.ndarray
        Arrays of component masses in solar masses.
    z : numpy.ndarray
        Redshift grid at which to evaluate the detection probabilities.
    cosmology : astropy.cosmology.FLRW
        Cosmology used to compute luminosity distances.
    snr_threshold : float, optional
        Network signal‑to‑noise ratio required for confident detection.
    snr_scale : float, optional
        Normalisation constant applied to the SNR.  Tuning this value
        effectively changes the horizon distance of the detector.  The
        default of 1.0 yields relative rates but not absolute values.
    width : float, optional
        Width of the logistic function controlling how rapidly the
        detection probability ramps from 0 to 1.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (N_binaries, N_redshift) of detection
        probabilities.
    """
    # Precompute chirp mass once
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    m_chirp = np.power(m1 * m2, 3.0 / 5.0) / np.power(m1 + m2, 1.0 / 5.0)
    # Get luminosity distance for each redshift (units: Mpc).  When astropy
    # is unavailable the returned object may already be a numpy array.  In
    # that case simply use it as is.
    dl_obj = cosmology.luminosity_distance(z)
    if hasattr(dl_obj, "to") and u is not None:
        d_lum = dl_obj.to(u.Mpc).value  # type: ignore[operator]
    else:
        d_lum = np.asarray(getattr(dl_obj, "value", dl_obj), dtype=float)
    # Replace any non positive luminosity distances with a small value to avoid division by zero
    d_lum = np.where(d_lum <= 0, 1e-3, d_lum)
    # Expand arrays for broadcasting
    m_chirp = m_chirp[:, None]  # shape (N,1)
    one_plus_z = (1.0 + z)[None, :]  # shape (1,M)
    # SNR scaling: chirp mass redshifted and distance dimming
    snr = snr_scale * np.power(m_chirp * one_plus_z, 5.0 / 6.0) / d_lum[None, :]
    # Logistic detection probability
    # centre logistic on threshold, width controls steepness
    x = (snr - snr_threshold) / width
    prob = 0.5 * (1.0 + np.tanh(x))
    # Clip to [0,1] explicitly to avoid any numerical excursions
    return np.clip(prob, 0.0, 1.0)


def find_detection_rate(
    path: str,
    dco_type: str = "BBH",
    merger_output_filename: Optional[str] = None,
    weight_column: Optional[str] = None,
    merges_hubble_time: bool = True,
    pessimistic_CEE: bool = True,
    no_RLOF_after_CEE: bool = True,
    max_redshift: float = 10.0,
    max_redshift_detection: float = 1.0,
    redshift_step: float = 0.01,
    z_first_SF: float = 10.0,
    use_sampled_mass_ranges: bool = True,
    m1_min: float = 5.0,
    m1_max: float = 150.0,
    m2_min: float = 0.1,
    fbin: float = 0.7,
    aSF: float = 0.01,
    bSF: float = 2.77,
    cSF: float = 2.90,
    dSF: float = 4.70,
    mu0: float = 0.035,
    muz: float = -0.23,
    sigma0: float = 0.39,
    sigmaz: float = 0.0,
    alpha: float = 0.0,
    min_logZ: float = -12.0,
    max_logZ: float = 0.0,
    step_logZ: float = 0.01,
    sensitivity: str = "O1",
    snr_threshold: float = 8.0,
    Mc_max: float = 300.0,
    Mc_step: float = 0.1,
    eta_max: float = 0.25,
    eta_step: float = 0.01,
    snr_max: float = 1000.0,
    snr_step: float = 0.1,
    cosmology: Optional[FLRW | str | dict] = None,
) -> Dict[str, np.ndarray]:
    """Calculate formation, merger and detection rates of compact binaries.

    Parameters
    ----------
    path : str
        Path to the HDF5 or NPZ file containing the population synthesis
        data.  See :func:`load_population` for supported formats.
    dco_type : str, optional
        Type of double compact object to select.  This implementation
        ignores the argument and processes all binaries in the file.
        It is kept for API compatibility.
    merger_output_filename : str, optional
        If provided, mergers will be written to the specified file in
        NumPy NPZ format.  Unused in this implementation.
    weight_column : str, optional
        Name of the weight dataset inside the HDF5 file.  Passed on
        verbatim to :func:`load_population`.
    merges_hubble_time, pessimistic_CEE, no_RLOF_after_CEE : bool, optional
        Flags controlling additional filtering of the binary population.
        They are present for API compatibility but ignored in this
        implementation.
    max_redshift : float, optional
        Maximum redshift of the star formation and merger grid.
    max_redshift_detection : float, optional
        Maximum redshift at which detection rates are computed.  This
        value must not exceed ``max_redshift``.
    redshift_step : float, optional
        Spacing of the redshift grid.  Smaller values result in finer
        resolution but increase computation time.
    z_first_SF : float, optional
        Maximum redshift at which star formation begins.  Values above
        ``max_redshift`` have no effect.  Included for API compatibility.
    use_sampled_mass_ranges : bool, optional
        If True (default) apply the mass cuts defined by ``m1_min``,
        ``m1_max`` and ``m2_min``.  Set to False to disable mass cuts.
    m1_min, m1_max, m2_min : float, optional
        Lower and upper bounds on component masses in solar masses.
    fbin : float, optional
        Fraction of stellar mass forming in binaries.  Acts as an overall
        normalisation on the rates.  The default value of 0.7 follows
        typical assumptions in the literature.
    aSF, bSF, cSF, dSF : float, optional
        Parameters of the Madau & Dickinson star formation rate history.
    mu0, muz, sigma0, sigmaz, alpha : float, optional
        Parameters controlling the metallicity distribution.  These are
        retained for API compatibility but metallicity is not used in
        the default calculation.
    min_logZ, max_logZ, step_logZ : float, optional
        Bounds and step for the internal metallicity grid.  These
        arguments are kept for API compatibility.
    sensitivity : str, optional
        Name of the detector sensitivity curve.  Unused in this
        implementation as the detection probability is parameterised by
        ``snr_threshold`` and the logistic ``width`` parameter.
    snr_threshold : float, optional
        Network SNR required for confident detection.
    Mc_max, Mc_step, eta_max, eta_step, snr_max, snr_step : float, optional
        Arguments related to the original SNR grid based method.  They
        are accepted for API compatibility but ignored.
    cosmology : astropy.cosmology.FLRW or str or dict, optional
        Cosmology used to compute distances and times.  If ``None``
        (default) the Planck18 cosmology is used.

    Returns
    -------
    dict
        A dictionary containing the redshift grid and corresponding
        formation, merger and detection rates.  Keys are ``'redshift'``,
        ``'formation_rate'``, ``'merger_rate'`` and ``'detection_rate'``.
    """
    # Obtain cosmology instance
    cosmo = get_cosmology(cosmology)
    # Redshift grid.  Force detection redshift not to exceed max_redshift
    max_z_det = min(float(max_redshift_detection), float(max_redshift))
    z_grid = np.arange(0.0, float(max_redshift) + redshift_step / 2.0, float(redshift_step))
    z_det_mask = z_grid <= max_z_det
    # Load binary population
    pop = load_population(path, weight_column=weight_column)
    # Apply mass cuts if requested
    if use_sampled_mass_ranges:
        pop.filter_by_mass(m1_min, m1_max, m2_min)
    # Compute star formation rate at each redshift
    sfr = star_formation_rate(z_grid, aSF, bSF, cSF, dSF)
    # Compute detection probabilities for each binary across detection redshifts
    # Use logistic width set to threshold / 2 for a smooth transition
    width = snr_threshold / 2.0
    det_probs = detection_probability(pop.m1, pop.m2, z_grid[z_det_mask], cosmo, snr_threshold=snr_threshold, snr_scale=1.0, width=width)
    # Formation rate: sum of weights times SFR at each redshift
    formation_rate = fbin * pop.weights.sum() * sfr
    # Merger rate: we ignore delay times and equate merger rate to formation rate
    merger_rate = formation_rate.copy()
    # Detection rate only defined on detection redshifts; fill array with zeros for the rest
    detection_rate = np.zeros_like(z_grid)
    # Multiply each binary's detection probability by its weight and sum
    weighted_probs = (pop.weights[:, None] * det_probs)
    detection_rate[z_det_mask] = fbin * (weighted_probs.sum(axis=0) * sfr[z_det_mask])
    # Additionally compute a per binary detection weight integrated over redshift
    # This quantity is used by the plotting routine to construct a mass distribution
    # of detected systems.  It represents the relative contribution of each binary
    # to the total detection rate.  We include the star formation rate at each
    # redshift such that the detection weight scales with the likelihood of
    # formation.
    detection_weight_per_binary = fbin * (weighted_probs * sfr[z_det_mask]).sum(axis=1)
    # Compute chirp mass for convenience
    m_chirp = np.power(pop.m1 * pop.m2, 3.0 / 5.0) / np.power(pop.m1 + pop.m2, 1.0 / 5.0)
    result = {
        "redshift": z_grid,
        "formation_rate": formation_rate,
        "merger_rate": merger_rate,
        "detection_rate": detection_rate,
        "m1": pop.m1,
        "m2": pop.m2,
        "weights": pop.weights,
        "mchirp": m_chirp,
        "detection_weight_per_binary": detection_weight_per_binary,
    }
    return result