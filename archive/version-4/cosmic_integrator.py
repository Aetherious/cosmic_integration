"""Cosmological integration of compact binary formation and detection rates.

The :class:`CosmicIntegrator` couples a population synthesis dataset, a
metallicity specific star–formation rate model and a cosmology to
calculate merger and detection rates as a function of redshift.  It
implements the steps described in Neijssel et al. (2019) and related
work, but the interface is general enough to accommodate alternative
prescriptions.

Usage example
-------------

.. code-block:: python

    from updated_cosmic_integration import CompasData, CosmicIntegrator, MSSFR

    compas_data = CompasData(path=Path("COMPAS_Output.h5"))
    compas_data.load_data()
    compas_data.set_dco_mask(["BBH"])
    compas_data.compute_delay_times()
    compas_data.compute_metallicity_grid()

    mssfr = MSSFR(metallicity_grid=[0.0002, 0.002, 0.02])

    integrator = CosmicIntegrator(compas_data=compas_data, mssfr=mssfr)
    integrator.create_redshift_shells()
    integrator.compute_birth_arrays()
    integrator.integrate()

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .compas_data import CompasData
from .mssfr import MSSFR
from .cosmology import get_cosmology
from .selection_effects import detection_probability


@dataclass
class CosmicIntegrator:
    """Perform cosmological integration of compact binary formation and detection.

    Parameters
    ----------
    compas_data:
        A populated :class:`CompasData` instance containing the population
        synthesis data and masks.
    mssfr:
        A configured :class:`MSSFR` instance capable of returning
        metallicity specific star formation rates.
    redshift_min:
        Minimum redshift at which to evaluate the merger rate.
    redshift_max:
        Maximum redshift at which to evaluate the merger rate.
    n_redshift_bins:
        Number of redshift bins used in the integration.
    gw_snr_threshold:
        Signal–to–noise threshold above which an event is considered
        detectable.
    gw_sensitivity:
        Name of the detector sensitivity curve to use (e.g., ``"design"`` or
        ``"O1"``).  The valid options are defined in
        :func:`~updated_cosmic_integration.selection_effects.detection_probability`.
    cosmology_name:
        Name of an `astropy.cosmology` known cosmology, or ``None`` to
        use the default cosmology.  See :func:`~updated_cosmic_integration.cosmology.get_cosmology`.

    Attributes
    ----------
    cosmology:
        The instantiated cosmology (an `astropy.cosmology.FLRW` instance).
    shell_redshift_edges, shell_center_redshifts, shell_dz:
        Arrays describing the redshift bins.
    shell_luminosity_distances:
        Luminosity distances corresponding to the shell centers (in Mpc).
    shell_volumes:
        Comoving volumes of each redshift shell (in Gpc³).
    per_system_age_birth, per_system_redshift_birth:
        Two–dimensional arrays of birth ages and birth redshifts for each
        system and shell.
    intrinsic_rates, observed_rates:
        Two–dimensional arrays of intrinsic and observed rates per system
        per shell.
    """

    compas_data: CompasData
    mssfr: MSSFR
    redshift_min: float = 0.0
    redshift_max: float = 2.0
    n_redshift_bins: int = 20
    gw_snr_threshold: float = 8.0
    gw_sensitivity: str = "design"
    cosmology_name: Optional[str] = None

    # Runtime attributes not set via the constructor
    cosmology: object = field(init=False, repr=False)
    shell_redshift_edges: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    shell_center_redshifts: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    shell_dz: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    shell_luminosity_distances: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    shell_volumes: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    per_system_age_birth: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    per_system_redshift_birth: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    intrinsic_rates: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    observed_rates: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Instantiate the cosmology used for all conversions."""
        self.cosmology = get_cosmology(self.cosmology_name)

    def create_redshift_shells(self) -> None:
        """Define concentric redshift bins for integration.

        This method initialises arrays describing the redshift grid used in
        the integration.  It sets the following attributes:

        * ``shell_redshift_edges`` – the boundaries of the bins
        * ``shell_center_redshifts`` – the midpoint of each bin
        * ``shell_dz`` – the width of each bin in redshift
        * ``shell_luminosity_distances`` – luminosity distance at each bin
        * ``shell_volumes`` – comoving volume of each bin in Gpc³

        Implementations must convert redshift into luminosity distance and
        comoving volume using the chosen cosmology.  See the astropy
        cosmology documentation for details.
        """
        pass

    def compute_birth_arrays(self) -> None:
        """Compute formation ages and redshifts for each system in each shell.

        Based on the delay time distribution from :attr:`compas_data`
        (in Myr) and the cosmology, this method calculates when and at
        which redshift each system would have formed if observed to merge
        at the center of each redshift shell.  The results are stored in
        ``per_system_age_birth`` (in Gyr) and ``per_system_redshift_birth``.
        Systems that would form before the onset of star formation are
        marked with a sentinel value (e.g. ``-1``).
        """
        pass

    def integrate(self) -> None:
        """Populate intrinsic and observed rates for each system in each shell.

        The cosmological integration proceeds by iterating over redshift
        shells and metallicity bins.  For each combination, the metallicity
        specific star formation rate (MSSFR) is evaluated at the birth
        redshift for the systems in that bin.  The rate per system is
        normalised by the mass evolved per metallicity (from the COMPAS
        dataset) and then converted to an intrinsic rate per comoving
        volume.  To obtain the observed rate, the intrinsic rate is
        multiplied by the comoving volume of the shell, corrected for
        cosmological time dilation, and weighted by the detection
        probability.
        """
        pass