"""Metallicity specific star–formation rate modelling.

The :class:`MSSFR` class encapsulates prescriptions for the
metallicity–specific star formation rate (MSSFR).  It combines a
metallicity distribution (either derived from a galaxy mass–metallicity
relation or a parametrized log–normal), a galaxy stellar mass function
and a cosmic star formation history to calculate the mass of stars
formed in a given metallicity bin at a specific cosmic time.

This class is intentionally minimal in the stub implementation.  In
production code one would implement methods to set different
prescriptions and evaluate the MSSFR.  Here we define the interface
only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from astropy.cosmology import FLRW, WMAP9


@dataclass
class MSSFR:
    """Model of the metallicity specific star formation rate.

    Parameters
    ----------
    metallicity_grid:
        Sequence of metallicities (mass fraction of metals) at which to
        evaluate the MSSFR.  Must be strictly increasing.
    bin_in_log_space:
        If ``True``, the metallicity bins will be defined in logarithmic
        space.  Otherwise bins are linear in metallicity.
    metallicity_lower_limit, metallicity_upper_limit:
        Lower and upper limits of the metallicity domain.  These limits
        are used when computing metallicity bin edges.
    cosmology:
        Astropy cosmology instance.  If omitted, :class:`~astropy.cosmology.WMAP9`
        will be used.  The cosmology is used when converting cosmic time
        into redshift and vice versa.
    verbose:
        If ``True``, the model may emit informational messages via the
        logging infrastructure.

    Notes
    -----
    Many options for the mass–metallicity relation, galaxy stellar mass
    function and cosmic star formation history exist in the literature.
    Implementations should provide methods to set these prescriptions
    individually.  The stub defines the public API but leaves the
    implementation to future work.
    """

    metallicity_grid: Optional[Sequence[float]] = None
    bin_in_log_space: bool = True
    metallicity_lower_limit: float = 1e-5
    metallicity_upper_limit: float = 1.0
    cosmology: FLRW = WMAP9
    verbose: bool = False

    # Internal attributes initialised at runtime
    metallicity_bin_edges: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def calculate_metallicity_bin_edges(self) -> None:
        """Compute the boundaries of the metallicity bins.

        Given the user supplied metallicity grid and bounds, this method
        constructs an array of metallicity bin edges.  In logarithmic
        mode, the edges are the geometric means between successive
        metallicities; in linear mode, they are arithmetic means.  The
        lower and upper limits are prepended and appended respectively.
        The resulting array has length ``len(metallicity_grid) + 1``.
        """
        pass

    def set_z_distribution(self, prescription: str, **kwargs: object) -> None:
        """Select a metallicity distribution prescription.

        Parameters
        ----------
        prescription:
            Name of the distribution (e.g. ``"lognormal"`` or ``"MZ_GSMF"``).
        **kwargs:
            Additional keyword arguments controlling the chosen prescription.
        """
        pass

    def set_gsmf(self, prescription: str, **kwargs: object) -> None:
        """Select a galaxy stellar mass function (GSMF) prescription.

        Parameters
        ----------
        prescription:
            Name of the GSMF (e.g. ``"Panter2004"`` or ``"Furlong2015"``).
        **kwargs:
            Additional keyword arguments controlling the GSMF.
        """
        pass

    def set_sfr_history(self, prescription: str, **kwargs: object) -> None:
        """Select a cosmic star formation history prescription.

        Parameters
        ----------
        prescription:
            Name of the star formation rate model (e.g. ``"Madau2014"``).
        **kwargs:
            Additional keyword arguments controlling the SFR model.
        """
        pass

    def return_mssfr(
        self,
        metallicity: float,
        ages_birth: np.ndarray,
        redshifts_birth: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the metallicity specific star formation rate.

        Given a metallicity bin and arrays of birth ages and redshifts,
        this method returns the MSSFR in units of solar masses per year per
        cubic Gigaparsec for each system in the sample.  The precise
        behaviour depends on the selected prescriptions for the
        metallicity distribution, GSMF and SFR history.

        Parameters
        ----------
        metallicity:
            The metallicity (mass fraction of metals) for which to compute
            the MSSFR.
        ages_birth:
            Array of ages of the universe at formation (in Gyr) for the
            systems in the population.
        redshifts_birth:
            Array of redshifts at formation corresponding to
            ``ages_birth``.

        Returns
        -------
        mssfr:
            Array of MSSFR values with the same shape as ``ages_birth``.
        """
        pass