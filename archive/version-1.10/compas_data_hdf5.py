"""Specialised CompasData class for HDF5 input files.

This module defines :class:`CompasDataHDF5`, a subclass of
:class:`~cosmic_integration_dasein.compas_data.CompasData` that
overrides the :meth:`load_data` method to preferentially read HDF5
datasets.  If the HDF5 file cannot be parsed via Pandas, the class
attempts to fall back to the :mod:`h5py` library, extracting all
top–level datasets into a :class:`pandas.DataFrame`.  Users must ensure
that the HDF5 file contains one–dimensional datasets of equal length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .compas_data import CompasData


@dataclass
class CompasDataHDF5(CompasData):
    """Subclass of :class:`CompasData` for HDF5 input files."""

    def load_data(self) -> None:
        path_str = str(self.path)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"HDF5 file not found: {path_str}")
        if not (path_str.endswith(".h5") or path_str.endswith(".hdf5")):
            raise ValueError("CompasDataHDF5 can only read HDF5 files.")
        # Try pandas first; support partial reads via self.max_rows
        try:
            if self.max_rows is not None:
                df = pd.read_hdf(path_str, start=0, stop=self.max_rows)
            else:
                df = pd.read_hdf(path_str)
        except Exception:
            try:
                import h5py  # type: ignore[import-not-found]
                with h5py.File(path_str, "r") as f:
                    data_dict = {key: f[key][()] for key in f.keys()}
                df = pd.DataFrame(data_dict)
                if self.max_rows is not None:
                    df = df.iloc[: self.max_rows]
            except Exception as e:
                raise ValueError(
                    f"Unable to parse HDF5 file; ensure it contains simple datasets: {e}"
                )
        self.data = df
        self.n_systems = len(df)