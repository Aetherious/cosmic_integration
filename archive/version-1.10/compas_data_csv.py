"""Specialised CompasData class for CSV input files.

This module defines a :class:`CompasDataCSV` class that inherits
from :class:`~cosmic_integration_dasein.compas_data.CompasData` but
overrides the :meth:`~cosmic_integration_dasein.compas_data.CompasData.load_data`
method to exclusively read CSV files.  It is intended for users who
have pre–processed their COMPAS output into comma–separated values
format to avoid repeated HDF5 parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .compas_data import CompasData


@dataclass
class CompasDataCSV(CompasData):
    """Subclass of :class:`CompasData` for CSV input files."""

    def load_data(self) -> None:
        path_str = str(self.path)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"CSV file not found: {path_str}")
        if not path_str.endswith(".csv"):
            raise ValueError("CompasDataCSV can only read CSV files.")
        # Use nrows argument to limit memory usage if max_rows is set
        if self.max_rows is not None:
            df = pd.read_csv(path_str, nrows=self.max_rows)
        else:
            df = pd.read_csv(path_str)
        self.data = df
        self.n_systems = len(df)