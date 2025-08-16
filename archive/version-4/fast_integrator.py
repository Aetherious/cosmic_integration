"""Fast cosmic integration and command-line interface.

This module bundles a set of helper functions and a command-line
interface for quickly estimating the detection rate of compact binary
mergers across cosmological distances.  The interface mirrors that of
the original ``FastCosmicIntegration.py`` script but has been split
into smaller functions for clarity.  Users can import the helper
functions directly or run the module as a script to parse command
line arguments and perform the calculation.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Callable, Sequence, Tuple

import numpy as np

from astropy.cosmology import FLRW
from astropy import units as u

from .cosmology import get_cosmology
from .compas_data import CompasData
from .selection_effects import detection_probability


def calculate_redshift_related_params(
    max_redshift: float = 10.0,
    max_redshift_detection: float = 1.0,
    redshift_step: float = 0.001,
    z_first_sf: float = 10.0,
    cosmology: Optional[FLRW] = None,
) -> Tuple[np.ndarray, int, np.ndarray, float, np.ndarray, np.ndarray]:
    """Return arrays of redshifts, ages, distances and volumes for integration.

    Parameters
    ----------
    max_redshift:
        Maximum redshift to consider.
    max_redshift_detection:
        Maximum redshift at which to compute detection rates (must be
        smaller than or equal to ``max_redshift``).
    redshift_step:
        Bin size in redshift.
    z_first_sf:
        Redshift at which the first star formation occurs.
    cosmology:
        Astropy cosmology instance.  If ``None``, the default cosmology
        from :func:`~updated_cosmic_integration.cosmology.get_cosmology` is used.

    Returns
    -------
    redshifts, n_redshifts_detection, ages, age_first_sf, distances, volumes:
        Arrays of redshifts, number of detection bins, cosmic ages (Myr),
        age of first star formation (Myr), luminosity distances (Mpc) and
        shell volumes (Gpc³).
    """
    # Implementation omitted in stub
    pass


def find_sfr(
    redshifts: Sequence[float],
    a: float = 0.01,
    b: float = 2.77,
    c: float = 2.90,
    d: float = 4.70,
) -> np.ndarray:
    """Return the cosmic star formation rate as a function of redshift.

    The functional form follows Madau & Dickinson (2014) and subsequent
    work.  The returned array has the same shape as ``redshifts`` and is
    expressed in solar masses per year per cubic Gigaparsec.
    """
    # Implementation omitted in stub
    pass


def find_metallicity_distribution(
    redshifts: Sequence[float],
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
    """Return the metallicity distribution dP/dlogZ as a function of redshift.

    Parameters
    ----------
    redshifts:
        Array of redshifts.
    min_logz_compas, max_logz_compas:
        Minimum and maximum log10 metallicities sampled in the COMPAS simulation.
    mu0, muz:
        Mean of the log–normal distribution at z=0 and its redshift dependence.
    sigma0, sigmaz:
        Standard deviation of the log–normal distribution at z=0 and its redshift dependence.
    alpha:
        Skewness parameter of the log–skew–normal distribution (alpha=0 reduces to log–normal).
    min_logz, max_logz, step_logz:
        Bounds and step size for the metallicity grid used in the integration.

    Returns
    -------
    dPdlogZ, metallicities, p_draw:
        Two–dimensional array of probabilities for each redshift and metallicity,
        array of metallicity values and probability of drawing a metallicity in
        the COMPAS sampling range.
    """
    # Implementation omitted in stub
    pass


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse command–line arguments for the fast integration script.

    Parameters
    ----------
    argv:
        Optional sequence of command line arguments to parse (defaults to
        ``sys.argv[1:]``).  See the full documentation for a description
        of the supported options.

    Returns
    -------
    args:
        Namespace object containing the parsed options.
    """
    parser = ArgumentParser(description="Fast cosmic integration options")
    # Implementation would add arguments here
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the command line interface.

    This function parses the command line, runs the integration and may
    output the results to disk or the console.  It is designed to be
    called from a ``__main__`` block.
    """
    args = parse_cli_args(argv)
    # Implementation omitted in stub
    pass


if __name__ == "__main__":  # pragma: no cover
    main()