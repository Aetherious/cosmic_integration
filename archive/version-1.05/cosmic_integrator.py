"""Cosmological integration of compact binary formation and detection rates.

The :class:`CosmicIntegrator` couples a population synthesis catalogue
loaded via :class:`CompasData`, a metallicity specific star formation
rate model (:class:`MSSFR`) and a cosmology (:class:`~.cosmology.Cosmology`)
to estimate the cosmic merger rate density of double compact objects
and their detection rate by gravitational–wave observatories.  The
integration follows the standard steps described in Neijssel et al.
(2019) and subsequent works but is simplified for clarity and to avoid
external dependencies.

The main workflow proceeds as follows:

1. Instantiate :class:`CompasData` and call :meth:`~CompasData.load_data`,
   :meth:`~CompasData.set_dco_mask`, :meth:`~CompasData.compute_delay_times`
   and :meth:`~CompasData.compute_metallicity_grid`.
2. Create an :class:`MSSFR` object.  The default SFR history and
   metallicity distribution are adequate for demonstration purposes.
3. Instantiate :class:`CosmicIntegrator` with the above objects and
   optional cosmological and detector parameters.
4. Call :meth:`create_redshift_shells` to define the grid in redshift.
5. Call :meth:`compute_birth_arrays` to determine the formation times
   and redshifts of the selected systems.
6. Call :meth:`integrate` to compute the intrinsic and observed
   merger rate density as a function of redshift.

Parallelisation
---------------

The integration can be computationally expensive when large numbers of
systems are considered.  To mitigate this, the :meth:`integrate`
method divides the calculation over redshift shells and executes
independent shells in parallel using a process pool.  This parallel
execution is particularly beneficial when processing binary black hole
(BBH) populations, which are typically the largest.  Users can adjust
the number of workers via the ``n_workers`` argument of
:meth:`integrate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .compas_data import CompasData
from .mssfr import MSSFR
from .cosmology import Cosmology, get_cosmology
from .selection_effects import detection_probability


@dataclass
class CosmicIntegrator:
    """Perform cosmological integration of compact binary formation and detection.

    Parameters
    ----------
    compas_data:
        A populated :class:`CompasData` instance containing the population
        synthesis data and masks.  Only systems selected by the DCO
        mask are used in the integration.
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
        Name of the detector sensitivity curve to use.  Currently
        unused but retained for compatibility.
    cosmology_params:
        Optional dictionary of cosmology parameters (``H0``, ``omega_m``,
        ``omega_lambda``) passed to :func:`~.cosmology.get_cosmology`.
    """

    compas_data: CompasData
    mssfr: MSSFR
    redshift_min: float = 0.0
    redshift_max: float = 4.0
    n_redshift_bins: int = 30
    gw_snr_threshold: float = 8.0
    gw_sensitivity: str = "design"
    cosmology_params: Optional[dict] = None

    # Runtime attributes (initialised in methods)
    cosmology: Cosmology = field(init=False, repr=False)
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
        # Instantiate cosmology from parameters
        self.cosmology = get_cosmology(self.cosmology_params)

    def create_redshift_shells(self) -> None:
        """Define concentric redshift bins and compute their volumes and distances."""
        # Create equally spaced bins in redshift
        edges = np.linspace(self.redshift_min, self.redshift_max, self.n_redshift_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dz = edges[1:] - edges[:-1]
        # Compute comoving volumes (Mpc^3) and convert to Gpc^3
        vol_elements = self.cosmology.comoving_volume_element(centers)  # Mpc^3 per dz
        volumes = vol_elements * dz / 1e9  # Gpc^3
        # Compute luminosity distances: d_L = (1+z) * d_C
        d_c = self.cosmology.comoving_distance(centers)
        d_L = (1 + centers) * d_c
        # Store
        self.shell_redshift_edges = edges
        self.shell_center_redshifts = centers
        self.shell_dz = dz
        self.shell_volumes = volumes
        self.shell_luminosity_distances = d_L

    def compute_birth_arrays(self, n_interp: int = 2000) -> None:
        """Compute formation times and redshifts for all systems and shells.

        This method computes a two–dimensional array ``per_system_age_birth``
        of shape ``(n_systems, n_bins)`` giving the lookback time to
        formation for each system if it merges at the centre redshift of
        each shell.  It also fills ``per_system_redshift_birth`` with the
        corresponding formation redshift obtained by inverting the
        lookback time relation.  Systems whose delay times would place
        their formation before the Big Bang are assigned a sentinel redshift
        of ``-1``.

        Parameters
        ----------
        n_interp:
            Number of samples used to build the redshift–lookback time
            inversion grid.  Increasing this value improves the accuracy
            of the inverted redshift at the cost of additional memory and
            computation.
        """
        if self.compas_data.delay_times is None:
            raise RuntimeError("compute_delay_times must be called on the CompasData before computing birth arrays.")
        if self.shell_center_redshifts is None:
            raise RuntimeError("create_redshift_shells must be called before computing birth arrays.")
        # Precompute lookback time at shell centres (Gyr)
        t_lb_merge = self.cosmology.lookback_time(self.shell_center_redshifts)
        # Convert delay times to Gyr
        delay_times = self.compas_data.delay_times.astype(float) / 1000.0  # Myr to Gyr
        n_systems = delay_times.size
        n_bins = self.shell_center_redshifts.size
        # Compute birth lookback times: t_birth = t_merge + delay
        # Shape (n_systems, n_bins)
        t_birth = t_lb_merge[np.newaxis, :] + delay_times[:, np.newaxis]
        self.per_system_age_birth = t_birth
        # Build inversion grid: redshift vs lookback time
        z_grid = np.linspace(0.0, max(self.redshift_max * 1.5, 10.0), n_interp)
        t_grid = self.cosmology.lookback_time(z_grid)
        # Ensure monotonicity for interpolation
        # t_grid is increasing with z, so we can invert it directly
        # Interpolate for each element of t_birth
        per_system_z_birth = np.empty_like(t_birth)
        max_t = t_grid[-1]
        for i in range(n_systems):
            # Vectorised interpolation for each row
            tb = t_birth[i]
            # Clip at max lookback time; assign -1 for those beyond
            mask_valid = tb <= max_t
            per_system_z_birth[i, ~mask_valid] = -1.0
            per_system_z_birth[i, mask_valid] = np.interp(tb[mask_valid], t_grid, z_grid)
        self.per_system_redshift_birth = per_system_z_birth

    def _compute_rates_for_shell(self, j: int) -> Tuple[float, float]:
        """Compute intrinsic and observed rate contributions for a single shell.

        This helper method is designed to be executed in parallel across
        shells.  It accesses instance attributes to gather the necessary
        data for shell ``j`` and returns the summed intrinsic and
        observed rates for that shell.

        Parameters
        ----------
        j:
            Index of the redshift shell.

        Returns
        -------
        tuple
            (intrinsic_rate, observed_rate) for shell ``j``.
        """
        # Skip systems with negative formation redshift
        birth_z = self.per_system_redshift_birth[:, j]
        valid = birth_z >= 0.0
        if not np.any(valid):
            return 0.0, 0.0
        # Retrieve metallicities and masses of valid systems
        df = self.compas_data.data.loc[self.compas_data.dco_mask]
        m1 = df.loc[valid, "mass1"].values.astype(float)
        m2 = df.loc[valid, "mass2"].values.astype(float)
        Z_sys = df.loc[valid, "metallicity"].values.astype(float)
        mass_evolved = np.empty_like(Z_sys, dtype=float)
        # Map each system's metallicity to the nearest metallicity grid value
        grid = self.compas_data.metallicity_grid
        total_mass_evolved = self.compas_data.total_mass_evolved_per_z
        # Precompute indices for mapping metallicity to mass evolved
        indices = np.abs(Z_sys[:, None] - grid[None, :]).argmin(axis=1)
        mass_evolved = total_mass_evolved[indices]
        # Evaluate MSSFR for each system at its birth redshift
        mssfr_values = self.mssfr.mssfr_for_bin(Z_sys, birth_z[valid])
        # Intrinsic rate per system: MSSFR divided by mass evolved
        intrinsic_per_system = mssfr_values / mass_evolved
        # Sum intrinsic rates and multiply by shell comoving volume
        intrinsic_rate_shell = intrinsic_per_system.sum() * self.shell_volumes[j]
        # Detection probability for each system
        d_L = self.shell_luminosity_distances[j]
        p_det = detection_probability(
            m1,
            m2,
            self.shell_center_redshifts[j],
            d_L,
            self.gw_snr_threshold,
            sensitivity=self.gw_sensitivity,
        )
        # Observed rate: intrinsic rate weighted by detection and time dilation
        # Note: divide by (1+z) to account for time dilation of merger rate
        observed_per_system = intrinsic_per_system * p_det / (1 + self.shell_center_redshifts[j])
        observed_rate_shell = observed_per_system.sum() * self.shell_volumes[j]
        return intrinsic_rate_shell, observed_rate_shell

    def integrate(self, n_workers: int = 1) -> None:
        """Compute intrinsic and observed merger rates across redshift shells.

        Parameters
        ----------
        n_workers:
            Number of parallel processes to use when evaluating different
            redshift shells.  If set to 1, the computation runs in the
            main process.  Increasing this value can significantly
            accelerate the integration for large BBH populations.
        """
        if self.per_system_redshift_birth is None or self.shell_volumes is None:
            raise RuntimeError("Birth arrays and shell volumes must be computed before integrating.")
        n_bins = self.shell_center_redshifts.size
        intrinsic_rates = np.zeros(n_bins, dtype=float)
        observed_rates = np.zeros(n_bins, dtype=float)
        # Use a process pool to compute each shell independently
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self._compute_rates_for_shell, j): j for j in range(n_bins)}
                for future in as_completed(futures):
                    j = futures[future]
                    intrinsic_rate_shell, observed_rate_shell = future.result()
                    intrinsic_rates[j] = intrinsic_rate_shell
                    observed_rates[j] = observed_rate_shell
        else:
            for j in range(n_bins):
                intrinsic_rate_shell, observed_rate_shell = self._compute_rates_for_shell(j)
                intrinsic_rates[j] = intrinsic_rate_shell
                observed_rates[j] = observed_rate_shell
        self.intrinsic_rates = intrinsic_rates
        self.observed_rates = observed_rates

    def save_results(self, filename: str) -> None:
        """Save the computed rates to a comma–separated file.

        A header row is written containing the redshift bin centres and
        the intrinsic and observed rates.  Units are Gpc⁻³ yr⁻¹ for the
        intrinsic rate and yr⁻¹ for the observed rate (per shell).

        Parameters
        ----------
        filename:
            Path to the output CSV file.  Existing files will be
            overwritten.
        """
        if self.intrinsic_rates is None or self.observed_rates is None:
            raise RuntimeError("Integration has not been run; no results to save.")
        import pandas as pd

        df = pd.DataFrame(
            {
                "redshift": self.shell_center_redshifts,
                "intrinsic_rate_Gpc3_yr": self.intrinsic_rates,
                "observed_rate_yr": self.observed_rates,
            }
        )
        df.to_csv(filename, index=False)
