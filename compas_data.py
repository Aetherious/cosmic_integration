"""Utilities for loading and filtering COMPAS simulation data.

This module defines a lightweight :class:`CompasData` class which
reads the relevant quantities from an HDF5 output file produced by
the COMPAS binary population synthesis code.  Only those fields
required to compute formation, merger and detection rates are
extracted; many of the optional fields and complex masking options
from the original code have been omitted for clarity.

The primary purpose of this class is to provide easy access to
component masses, metallicities and delay times, together with
simple filtering on the compact binary type (BBH, BHNS, BNS or all).
It also computes the representative star forming mass per binary
system which is required to normalise the cosmic star formation
history when converting star formation rates into formation rates.

Example
-------

>>> from cosmic_integration_dasein.compas_data import CompasData
>>> compas = CompasData("/path/to/COMPAS_Output.h5", m2_min=0.1, binary_fraction=0.7)
>>> compas.set_dco_mask(types="BBH")
>>> compas.load()
>>> compas.find_star_forming_mass_per_binary_sampling()

"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple, Union

try:  # h5py is a required dependency.
    import h5py as h5
except Exception as exc:
    # Immediately raise an informative error if h5py cannot be imported.
    raise ImportError(
        "The h5py package is required to read COMPAS HDF5 files. "
        "Install h5py or provide an alternative data loader."
    ) from exc

import numpy as np

try: # SciPy is a required dependency.
    from scipy.integrate import quad
except Exception as exc:
    raise ImportError(
        "The SciPy package is required to accurately calculate the COMPAS mass fraction. "
        "Please install SciPy and its dependencies."
    ) from exc

class CompasData:
    """Container for reading and storing COMPAS simulation data.

    Parameters
    ----------
    path : str
        Path to the COMPAS HDF5 output file.
    m1_min : float, optional
        Minimum primary mass sampled in the simulation.  If provided
        together with ``m1_max`` and ``m2_min`` this will be used
        when computing the star forming mass per binary.  Defaults to
        ``None``.
    m1_max : float, optional
        Maximum primary mass sampled in the simulation.  Defaults to
        ``None``.
    m2_min : float, optional
        Minimum secondary mass sampled in the simulation.  Defaults
        to ``None``.
    binary_fraction : float, optional
        Fraction of stars born in binary systems used by the COMPAS
        simulation.  Defaults to ``None``.
    suppress_reminder : bool, optional
        If ``True`` no reminder about calling :meth:`set_dco_mask` and
        :meth:`load` is printed.  Defaults to ``False``.

    Notes
    -----
    The COMPAS HDF5 format consists of multiple groups storing
    different aspects of the simulation.  This class accesses only
    those groups required for the integration: ``BSE_Double_Compact_Objects``
    for compact object binaries and ``BSE_System_Parameters`` for
    initial conditions.  Other groups are ignored.
    """

    def __init__(
        self,
        path: str,
        m1_min: Optional[float] = None,
        m1_max: Optional[float] = None,
        m2_min: Optional[float] = None,
        binary_fraction: Optional[float] = None,
        suppress_reminder: bool = False,
    ) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        self.path = path

        # Simulation sampling parameters used to estimate the total
        # star forming mass per binary.  These may be overwritten by
        # the caller after inspecting the loaded masses.  None
        # indicates that no constraints are applied.
        self.m1_min = m1_min
        self.m1_max = m1_max
        self.m2_min = m2_min
        self.binary_fraction = binary_fraction

        # Arrays loaded from the HDF5 file after calling load().
        self.mass1: Optional[np.ndarray] = None
        self.mass2: Optional[np.ndarray] = None
        self.delay_times: Optional[np.ndarray] = None
        self.initial_metallicities: Optional[np.ndarray] = None
        self.sw_weights: Optional[np.ndarray] = None
        self.n_systems: Optional[int] = None

        # Mask identifying which rows correspond to the desired
        # double compact object type.  Initially unset; the user
        # should call set_dco_mask() before load().
        self.dco_mask: Optional[np.ndarray] = None

        # Resulting arrays after applying the DCO mask.  These will
        # contain only the selected systems.
        self.mass1_masked: Optional[np.ndarray] = None
        self.mass2_masked: Optional[np.ndarray] = None
        self.delay_times_masked: Optional[np.ndarray] = None
        self.metallicity_masked: Optional[np.ndarray] = None

        # Representative star forming mass per binary.  Computed
        # lazily by find_star_forming_mass_per_binary_sampling().
        self.mass_evolved_per_binary: Optional[float] = None

        if not suppress_reminder:
            print(
                "CompasData: remember to call set_dco_mask() and load() before using the data."
            )

    # ------------------------------------------------------------------
    # Internal helper functions
    # ------------------------------------------------------------------
    def _get_variables(
        self, group_name: str, variable_names: Union[str, Iterable[str]]
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Retrieve one or more variables from an HDF5 group.

        Parameters
        ----------
        group_name : str
            Name of the group in the HDF5 file.
        variable_names : str or iterable of str
            Dataset names to return from the group.  If a single
            name is supplied the corresponding array is returned
            directly; otherwise a tuple of arrays is returned in
            the same order as ``variable_names``.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The requested variable(s).  Datasets are read into memory
            in their entirety.  For large simulations this may
            consume significant memory; however this class is
            intended for moderate sized populations.
        """
        with h5.File(self.path, "r") as h5file:
            group = h5file[group_name]

            if isinstance(variable_names, str):
                return group[variable_names][()]
            
            return tuple(group[name][()] for name in variable_names)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_dco_mask(
        self,
        types: str = "BBH",
        within_hubble_time: bool = True,
        pessimistic: bool = True,
        no_rlof_after_cee: bool = True,
    ) -> None:
        """Select a mask on the compact binary types.

        The available types are ``'BBH'`` for binary black holes,
        ``'BHNS'`` for black hole - neutron star binaries, ``'BNS'`` for
        double neutron stars and ``'all'`` to select every compact
        binary.  Optional boolean flags allow one to further filter
        out systems that do not merge within a Hubble time, systems
        undergoing optimistic common envelope events or immediate
        Roche lobe overflow after common envelope.

        Parameters
        ----------
        types : {'BBH', 'BHNS', 'BNS', 'all'}, optional
            Which double compact object type to retain.  Defaults
            to 'BBH'.
        within_hubble_time : bool, optional
            If ``True`` only binaries that merge within a Hubble time
            are kept.  Defaults to ``True``.
        pessimistic : bool, optional
            If ``True`` binaries experiencing an optimistic common
            envelope are removed.  Defaults to ``True``.
        no_rlof_after_cee : bool, optional
            If ``True`` binaries that overflow their Roche lobe
            immediately after a common envelope event are removed.
            Defaults to ``True``.

        Notes
        -----
        This method does not read the full data arrays; it only
        extracts the necessary flags to build the mask.  The mask is
        stored in :attr:`dco_mask` and applied by :meth:`load`.
        """
        # Retrieve basic DCO properties:
        stellar_type_1, stellar_type_2, merges_hubble, seeds = self._get_variables(
            "BSE_Double_Compact_Objects",
            ["Stellar_Type(1)", "Stellar_Type(2)", "Merges_Hubble_Time", "SEED"],
        )
        seeds = seeds.flatten()

        # Determine type specific mask.  In the COMPAS output the
        # final stellar types are encoded as integers: 14 for BH,
        # 13 for NS.  Black hole – neutron star systems can appear
        # in either order of the components.
        type_masks = {
            "all": np.repeat(True, len(seeds)),
            "BBH": np.logical_and(stellar_type_1 == 14, stellar_type_2 == 14),
            "BHNS": np.logical_or(
                np.logical_and(stellar_type_1 == 14, stellar_type_2 == 13),
                np.logical_and(stellar_type_1 == 13, stellar_type_2 == 14),
            ),
            "BNS": np.logical_and(stellar_type_1 == 13, stellar_type_2 == 13),
        }
        if types not in type_masks:
            raise ValueError(f"Unknown DCO type '{types}'.")
        mask_type = type_masks[types]

        # Apply Hubble time mask, if requested:
        mask_hubble = merges_hubble.astype(bool) if within_hubble_time else np.repeat(True, len(seeds))

        # The optimistic common envelope and Roche lobe overflow flags
        # are stored in the common envelope group.  To avoid loading
        # entire groups we read only the relevant columns.
        ce_seeds = self._get_variables("BSE_Common_Envelopes", "SEED")
        ce_mask = np.isin(ce_seeds, seeds)
        if no_rlof_after_cee:
            rlof_flag = self._get_variables("BSE_Common_Envelopes", "Immediate_RLOF>CE")[ce_mask].astype(bool)
            rlof_seeds = np.unique(ce_seeds[ce_mask][rlof_flag])
            mask_rlof = np.logical_not(np.isin(seeds, rlof_seeds))
        else:
            mask_rlof = np.repeat(True, len(seeds))
        if pessimistic:
            pessimistic_flag = self._get_variables("BSE_Common_Envelopes", "Optimistic_CE")[ce_mask].astype(bool)
            pessimistic_seeds = np.unique(ce_seeds[ce_mask][pessimistic_flag])
            mask_pessimistic = np.logical_not(np.isin(seeds, pessimistic_seeds))
        else:
            mask_pessimistic = np.repeat(True, len(seeds))

        # Combine masks:
        self.dco_mask = mask_type & mask_hubble & mask_rlof & mask_pessimistic

    def load(self) -> None:
        """Load mass, delay time and metallicity arrays and apply mask.

        After calling this method the attributes :attr:`mass1_masked`,
        :attr:`mass2_masked`, :attr:`delay_times_masked` and
        :attr:`metallicity_masked` will be populated.  The full arrays
        are also stored on :attr:`mass1`, :attr:`mass2`,
        :attr:`delay_times` and :attr:`initial_metallicities`.

        If :meth:`set_dco_mask` has not been called a ValueError
        will be raised.
        """
        if self.dco_mask is None:
            raise ValueError("DCO mask has not been set.  Call set_dco_mask() first.")

        # Load masses, times, and seeds for DCOs:
        primary_masses, secondary_masses, formation_times, coalescence_times, seeds = self._get_variables(
            "BSE_Double_Compact_Objects",
            ["Mass(1)", "Mass(2)", "Time", "Coalescence_Time", "SEED"],
        )
        seeds = seeds.flatten()

        # Load initial metallicities for all systems; later we pick
        # out those corresponding to the DCO seeds:
        initial_seeds, initial_Z = self._get_variables(
            "BSE_System_Parameters", ["SEED", "Metallicity@ZAMS(1)"]
        )

        # Delay time is the sum of formation time and coalescence time:
        delay_times = formation_times + coalescence_times

        # Apply mask:
        mask = self.dco_mask
        self.mass1 = primary_masses
        self.mass2 = secondary_masses
        self.delay_times = delay_times
        self.initial_metallicities = initial_Z

        self.mass1_masked = primary_masses[mask]
        self.mass2_masked = secondary_masses[mask]
        self.delay_times_masked = delay_times[mask]

        # Identify the metallicity for each DCO by matching seeds:
        if len(initial_seeds.shape) > 1:
            initial_seeds = initial_seeds.flatten()
        mask_met = np.isin(initial_seeds, seeds[mask])
        self.metallicity_masked = initial_Z[mask_met]
        self.n_systems = len(initial_seeds)

        # Initialise weights to unity; these may be overridden
        # externally via set_weights() or by find_star_forming_mass_per_binary_sampling():
        self.sw_weights = np.ones_like(self.mass1_masked, dtype=float)

    def set_weights(self, weight_column: Optional[str] = None) -> None:
        """Load adaptive sampling weights for each DCO if available.

        Parameters
        ----------
        weight_column : str, optional
            Name of the column in the ``DoubleCompactObjects`` table that
            contains sampling weights.  If ``None`` weights of unity
            are used.  Defaults to ``None``.

        Notes
        -----
        Adaptive sampling weights allow one to correct the intrinsic
        distribution of DCOs back to the underlying population when
        sampling non-uniformly.  If the weight column is not present
        or unspecified weights of unity are retained.
        """
        if weight_column is None:
            return
        
        try:
            weights = self._get_variables("BSE_Double_Compact_Objects", weight_column)
            # Apply DCO mask:
            self.sw_weights = weights[self.dco_mask]
        except Exception:
            # Fall back to unity weights:
            self.sw_weights = np.ones_like(self.mass1_masked, dtype=float)

    # ------------------------------------------------------------------
    # Star forming mass per binary
    # ------------------------------------------------------------------
    def _imf(self, m: float) -> float:
        """Kroupa (2001) initial mass function with three segments.

        This is a private helper used by
        :meth:`find_star_forming_mass_per_binary_sampling` to compute
        the representative star forming mass per binary.  The IMF is
        defined piecewise with slopes ``a12``, ``a23`` and ``a34``
        between the mass boundaries ``m1``, ``m2``, ``m3`` and ``m4``.

        Parameters
        ----------
        m : float
            Stellar mass at which to evaluate the IMF.

        Returns
        -------
        float
            Value of the IMF at mass ``m``.  Units are arbitrary and
            cancel in the normalisation.
        """
        # Default parameters following Kroupa (2001):
        m1, m2, m3, m4 = 0.01, 0.08, 0.5, 200.0
        a12, a23, a34 = 0.3, 1.3, 2.3

        # Compute normalisation constants on first call:
        b1 = 1 / (
            (m2 ** (1 - a12) - m1 ** (1 - a12)) / (1 - a12)
            + m2 ** (-(a12 - a23)) * (m3 ** (1 - a23) - m2 ** (1 - a23)) / (1 - a23)
            + m2 ** (-(a12 - a23))
            * m3 ** (-(a23 - a34))
            * (m4 ** (1 - a34) - m3 ** (1 - a34))
            / (1 - a34)
        )
        b2 = b1 * m2 ** (-(a12 - a23))
        b3 = b2 * m3 ** (-(a23 - a34))

        # Piecewise IMF:
        if m1 <= m < m2:
            return b1 * m ** (-a12)
        if m2 <= m < m3:
            return b2 * m ** (-a23)
        if m3 <= m <= m4:
            return b3 * m ** (-a34)
        
        return 0.0

    def _compas_mass_fraction(self) -> float:
        """Compute the fraction of total stellar mass represented by the COMPAS sampling.

        The COMPAS simulation typically evolves only a subset of the
        full stellar mass spectrum.  To convert between the mass
        evolved in COMPAS and the universal star formation rate the
        ratio of masses must be estimated.  This helper performs the
        integrations required to obtain that fraction.

        Returns
        -------
        float
            The fraction of mass in the COMPAS population relative to
            the total underlying stellar population.

        Notes
        -----
        The current implementation uses numerical integration from
        ``scipy.integrate.quad`` if SciPy is available. For
        scientific applications, SciPy is strongly recommended.
        """
        # Unpack sampling limits; use defaults if None:
        m1_low = self.m1_min if self.m1_min is not None else 0.01
        m1_upp = self.m1_max if self.m1_max is not None else 200.0
        m2_low = self.m2_min if self.m2_min is not None else 0.01
        f_bin = self.binary_fraction if self.binary_fraction is not None else 1.0

        # Define mass ratio PDF as uniform between 0 and 1:
        def qpdf(q: float) -> float:
            return 1.0

        # Integrand for the full mass (no COMPAS cut):
        def full_integral(mass: float) -> float:
            imf_val = self._imf(mass)
            primary_mass = imf_val * mass
            # expected secondary mass given uniform mass ratio distribution
            expected_secondary_mass = quad(lambda q: q * qpdf(q), 0.0, 1.0)[0] * primary_mass
            single_stars = (1.0 - f_bin) * primary_mass
            binary_stars = f_bin * (primary_mass + expected_secondary_mass)
            return single_stars + binary_stars

        full_mass, _ = quad(full_integral, 0.01, 200.0)

        # Integrand for COMPAS sampled mass:
        def compas_integral(mass: float) -> float:
            imf_val = self._imf(mass)
            primary_mass = imf_val * mass
            # Fraction of secondaries below m2_low:
            f_below = quad(qpdf, 0.0, m2_low / mass if mass > 0 else 0.0)[0]
            expected_secondary_mass = quad(
                lambda q: q * qpdf(q), m2_low / mass if mass > 0 else 0.0, 1.0
            )[0] * primary_mass
            return f_bin * (1.0 - f_below) * (primary_mass + expected_secondary_mass)

        compas_mass, _ = quad(compas_integral, m1_low, m1_upp)

        return compas_mass / full_mass

    def find_star_forming_mass_per_binary_sampling(self) -> None:
        """Compute and store the representative star forming mass per binary.

        This method estimates the ratio of the total mass of stars
        formed in the simulated mass range to the number of simulated
        systems.  The resulting mass per binary is stored in
        :attr:`mass_evolved_per_binary` and can be used to convert
        cosmological star formation rates into formation rates for
        the simulated population.

        Notes
        -----
        The computation relies on the sampling limits ``m1_min``,
        ``m1_max`` and ``m2_min``, together with the ``binary_fraction``
        specifying the fraction of stars born in binaries.  These
        should be set before calling this method.  If any of the
        sampling parameters are ``None`` sensible defaults are used.
        """
        # Ensure we have loaded the data first to know the number of systems:
        if self.mass1 is None or self.mass2 is None:
            raise RuntimeError("Call load() before computing star forming mass per binary.")

        # Compute the fraction of total mass represented by the COMPAS sampling:
        fraction = self._compas_mass_fraction()
        if fraction <= 0.0:
            raise RuntimeError("Computed COMPAS mass fraction is non positive.")

        # Calculate the total mass evolved per metallicity bin from the HDF5 file:
        with h5.File(self.path, "r") as h5file:
            all_systems = h5file["BSE_System_Parameters"]
            m1s = all_systems["Mass@ZAMS(1)"][()]
            m2s = all_systems["Mass@ZAMS(2)"][()]
            total_mass_compas = float(np.sum(m1s) + np.sum(m2s))
            n_binaries = len(m1s)

        # Total mass formed in the universe represented by the simulation:
        total_mass_universe = total_mass_compas / fraction
        self.mass_evolved_per_binary = total_mass_universe / float(n_binaries)
