"""Data access and masking for COMPAS population synthesis results.

This module defines a :class:`CompasData` class that encapsulates the
various datasets required by the cosmic integration pipeline.  The class
provides methods to load information from an HDF5 file produced by
COMPAS, construct masks for different classes of double compact objects
(DCOs), compute delay times and metallicity grids, and recalculate the
total mass evolved under different initial mass function assumptions.

The design emphasises lazy loading and clear separation of concerns:

* File system interactions and HDF5 handling are hidden behind the
  :meth:`load_data` method.
* Mask construction is performed via :meth:`set_dco_mask`.
* Computation of auxiliary quantities such as delay times and metallicity
  grids is delegated to specific methods.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .metallicity_grid import total_mass_evolved_per_z


@dataclass
class CompasData:
    """Container for COMPAS binary population synthesis data.

    The :class:`CompasData` class acts as a thin wrapper around a
    tabular dataset containing the outputs of a population synthesis
    calculation.  Rather than rely on HDF5 or proprietary formats
    directly, this implementation expects the data to be supplied as a
    comma–separated values (CSV) file, a pickle of a Pandas
    ``DataFrame`` or an ``npz`` archive containing named arrays.  The
    data must include at minimum the following columns:

    * ``mass1`` and ``mass2``: the primary and secondary BH/NS masses in solar masses.
    * ``stellar_type_1`` and ``stellar_type_2``: integer stellar type codes (BH=14, NS=13).
    * ``merges_hubble_time``: boolean flag indicating whether the system merges within a Hubble time.
    * ``optimistic_ce``: boolean flag indicating whether the optimistic common envelope prescription applies.
    * ``immediate_rlof``: boolean flag for systems that experience immediate Roche–lobe overflow post–CEE.
    * ``metallicity``: initial metallicity of the system (mass fraction of metals).
    * ``formation_time`` and ``merger_time`` or ``delay_time``: either the formation and merger times (in Myr) or a precomputed delay time.  If both are available the delay time is computed as the difference.

    Additional columns may be present and will be stored on the instance.

    Parameters
    ----------
    path:
        Path to the input file.  The file type is inferred from the
        extension: ``.csv`` for CSV, ``.pkl`` or ``.pickle`` for
        pickled ``DataFrame`` and ``.npz`` for NumPy archives.
    m_lower, m_upper, m2_min, binary_fraction:
        Parameters of the initial mass function used to compute the
        mass evolved per metallicity.  These may be overridden later
        with :meth:`recalculate_mass_evolved`.
    """

    path: Path
    m_lower: Optional[float] = None
    m_upper: Optional[float] = None
    m2_min: Optional[float] = None
    binary_fraction: Optional[float] = None

    # Attributes set after loading
    data: Optional[pd.DataFrame] = None
    n_systems: Optional[int] = None
    dco_mask: Optional[np.ndarray] = None
    bbh_mask: Optional[np.ndarray] = None
    bhns_mask: Optional[np.ndarray] = None
    dns_mask: Optional[np.ndarray] = None
    metallicity_grid: Optional[np.ndarray] = None
    total_mass_evolved_per_z: Optional[np.ndarray] = None

    def load_data(self) -> None:
        """Load the simulation dataset from disk.

        The file specified by :attr:`path` is read and converted into a
        Pandas ``DataFrame``.  A variety of file formats are supported:

        ``.csv``
            Parsed via :func:`pandas.read_csv`.
        ``.pkl`` or ``.pickle``
            Loaded via :func:`pandas.read_pickle`.
        ``.npz``
            Interpreted as a NumPy archive containing arrays for each
            required column.  All arrays must have the same length.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is unsupported or missing required columns.
        """
        path_str = str(self.path)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"Input data file not found: {path_str}")
        if path_str.endswith(".csv"):
            df = pd.read_csv(path_str)
        elif path_str.endswith(".pkl") or path_str.endswith(".pickle"):
            df = pd.read_pickle(path_str)
        elif path_str.endswith(".npz"):
            with np.load(path_str, allow_pickle=True) as archive:
                columns = archive.files
                data_dict = {col: archive[col] for col in columns}
                df = pd.DataFrame(data_dict)
        else:
            raise ValueError(
                "Unsupported file format. Please provide a CSV, pickle or NPZ file."
            )
        # Validate required columns
        required = [
            "mass1",
            "mass2",
            "stellar_type_1",
            "stellar_type_2",
            "merges_hubble_time",
            "optimistic_ce",
            "immediate_rlof",
            "metallicity",
        ]
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            raise ValueError(f"Missing required columns in input data: {missing}")
        # Delay time may be given directly or derived from formation and merger times
        if "delay_time" not in df.columns:
            if {"formation_time", "merger_time"}.issubset(df.columns):
                df["delay_time"] = df["merger_time"] - df["formation_time"]
            else:
                raise ValueError(
                    "Dataset must contain either 'delay_time' or both 'formation_time' and 'merger_time'."
                )
        # Store DataFrame and number of systems
        self.data = df
        self.n_systems = len(df)

    def set_dco_mask(
        self,
        dco_types: Sequence[str],
        within_hubble_time: bool = True,
        pessimistic: bool = True,
        forbid_immediate_rlof: bool = True,
    ) -> None:
        """Construct boolean masks for different double compact object types.

        This method sets masks on the dataset to select specific types
        of compact object binaries.  The selection is controlled by
        several boolean flags which mirror common choices in the COMPAS
        community.

        Parameters
        ----------
        dco_types:
            Sequence of type identifiers.  Valid entries include ``"BBH"``
            for binary black holes, ``"BHNS"`` for black hole–neutron
            star binaries and ``"BNS"`` for binary neutron stars.  If
            multiple types are supplied the union of all matching
            systems is selected.
        within_hubble_time:
            If ``True``, only systems flagged as merging within a Hubble
            time are included.
        pessimistic:
            If ``True``, exclude systems with ``optimistic_ce`` set to
            ``True``.
        forbid_immediate_rlof:
            If ``True``, exclude systems with ``immediate_rlof`` set to
            ``True``.

        Notes
        -----
        On completion the instance attributes ``dco_mask``, ``bbh_mask``,
        ``bhns_mask`` and ``dns_mask`` are defined.  The union of all
        selected types is stored in ``dco_mask``.
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before setting DCO masks.")
        df = self.data
        # Masks based on stellar types: BH=14, NS=13
        bbh = (df["stellar_type_1"] == 14) & (df["stellar_type_2"] == 14)
        bhns = (
            (df["stellar_type_1"] == 14) & (df["stellar_type_2"] == 13)
        ) | (
            (df["stellar_type_1"] == 13) & (df["stellar_type_2"] == 14)
        )
        dns = (df["stellar_type_1"] == 13) & (df["stellar_type_2"] == 13)
        # Apply flags
        mask = np.zeros(len(df), dtype=bool)
        for dtype in dco_types:
            if dtype.upper() == "BBH":
                mask |= bbh.values
            elif dtype.upper() == "BHNS":
                mask |= bhns.values
            elif dtype.upper() in {"BNS", "DNS"}:
                mask |= dns.values
            else:
                raise ValueError(f"Unknown DCO type: {dtype}")
        if within_hubble_time and "merges_hubble_time" in df.columns:
            mask &= df["merges_hubble_time"].astype(bool).values
        if pessimistic and "optimistic_ce" in df.columns:
            mask &= ~df["optimistic_ce"].astype(bool).values
        if forbid_immediate_rlof and "immediate_rlof" in df.columns:
            mask &= ~df["immediate_rlof"].astype(bool).values
        # Save masks
        self.dco_mask = mask
        self.bbh_mask = bbh.values & mask
        self.bhns_mask = bhns.values & mask
        self.dns_mask = dns.values & mask

    def compute_delay_times(self) -> None:
        """Compute delay times for each system in the selected DCO mask.

        The delay time is the time between the formation of the binary
        system and its eventual merger.  If a column ``delay_time`` is
        present in the loaded dataset it is used directly; otherwise
        the difference between ``merger_time`` and ``formation_time`` is
        computed.  Results are stored in Myr.
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before computing delay times.")
        if self.dco_mask is None:
            raise RuntimeError("DCO mask must be set before computing delay times.")
        df = self.data
        if "delay_time" in df.columns:
            delay = df.loc[self.dco_mask, "delay_time"].values.astype(float)
        else:
            delay = (
                df.loc[self.dco_mask, "merger_time"].values
                - df.loc[self.dco_mask, "formation_time"].values
            ).astype(float)
        self.delay_times = delay

    def compute_metallicity_grid(self) -> None:
        """Construct a metallicity grid and compute mass evolved per metallicity.

        The metallicity grid is set to the sorted unique metallicity
        values in the selected systems.  The mass evolved per metallicity
        is computed using the IMF parameters (``m_lower``, ``m_upper``,
        ``m2_min`` and ``binary_fraction``).  If these parameters are
        not set on the instance they default to [5, 150, 0.08, 1] as
        commonly used in the literature.
        """
        if self.data is None or self.dco_mask is None:
            raise RuntimeError("Data and DCO mask must be initialised before computing metallicity grid.")
        df = self.data
        # Unique metallicities of selected systems
        unique_Z = np.unique(df.loc[self.dco_mask, "metallicity"].astype(float))
        self.metallicity_grid = np.sort(unique_Z)
        # Use defaults if attributes are None
        m_lower = self.m_lower if self.m_lower is not None else 5.0
        m_upper = self.m_upper if self.m_upper is not None else 150.0
        m2_min = self.m2_min if self.m2_min is not None else 0.08
        binary_fraction = self.binary_fraction if self.binary_fraction is not None else 1.0
        self.total_mass_evolved_per_z = total_mass_evolved_per_z(
            self.metallicity_grid,
            m_lower=m_lower,
            m_upper=m_upper,
            m2_min=m2_min,
            binary_fraction=binary_fraction,
        )

    def recalculate_mass_evolved(
        self,
        m_lower: float,
        m_upper: float,
        binary_fraction: float,
    ) -> None:
        """Update IMF parameters and recompute mass evolved per metallicity.

        Parameters
        ----------
        m_lower, m_upper:
            New primary mass limits for the IMF.
        binary_fraction:
            New binary fraction.

        Updating these parameters triggers recomputation of
        :attr:`total_mass_evolved_per_z`.
        """
        self.m_lower = m_lower
        self.m_upper = m_upper
        self.binary_fraction = binary_fraction
        # If the metallicity grid has already been computed, update the mass evolved
        if self.metallicity_grid is not None:
            self.total_mass_evolved_per_z = total_mass_evolved_per_z(
                self.metallicity_grid,
                m_lower=m_lower,
                m_upper=m_upper,
                m2_min=self.m2_min if self.m2_min is not None else 0.08,
                binary_fraction=binary_fraction,
            )