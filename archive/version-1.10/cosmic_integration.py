"""Command line interface for the cosmic integration pipeline.

This script provides a user‐friendly wrapper around the core classes of
``cosmic_integration_dasein``.  It allows researchers to compute
merger and detection rates from a population synthesis catalogue in a
single command.  The input data should be in CSV, pickle or NPZ
format and must include the columns described in
``compas_data.py``.  The default settings reproduce a simple
analysis of binary black holes using a Kroupa IMF and the Madau &
Dickinson (2014) SFR history.

Example usage::

    python -m cosmic_integration_dasein.cosmic_integration \
        --input-file my_population.csv \
        --dco-type BBH \
        --output-file rates.csv \
        --n-workers 4

The computed rates are written to the specified output file in CSV
format.  Use the ``--help`` flag to see all available options.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from .compas_data import CompasData
from .compas_data_csv import CompasDataCSV
from .compas_data_hdf5 import CompasDataHDF5
from .mssfr import MSSFR
from .cosmic_integrator import CosmicIntegrator


def parse_arguments(args: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments for the cosmic integration CLI."""
    parser = argparse.ArgumentParser(description="Cosmic integration of compact binary merger rates")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the population synthesis data file (CSV, pickle or NPZ)",
    )
    parser.add_argument(
        "--dco-type",
        type=str,
        nargs="+",
        default=["BBH"],
        help="Types of double compact objects to include (BBH, BHNS, BNS)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="rates.csv",
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.0,
        help="Minimum redshift for integration",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=4.0,
        help="Maximum redshift for integration",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=30,
        help="Number of redshift bins",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--sfr-model",
        type=str,
        default="MadauDickinson2014",
        help="Name of the star formation history model (unused in this simple implementation)",
    )
    parser.add_argument(
        "--metallicity-grid",
        type=float,
        nargs="*",
        default=None,
        help="List of metallicities (mass fraction) defining the metallicity grid",
    )
    parser.add_argument(
        "--gw-threshold",
        type=float,
        default=8.0,
        help="SNR threshold for gravitational wave detection",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the command line interface."""
    args = parse_arguments(argv if argv is not None else [])
    # Load population synthesis data using appropriate subclass
    input_path = Path(args.input_file)
    if input_path.suffix.lower() == ".csv":
        compas_data = CompasDataCSV(path=input_path)
    elif input_path.suffix.lower() in {".h5", ".hdf5"}:
        compas_data = CompasDataHDF5(path=input_path)
    else:
        compas_data = CompasData(path=input_path)
    compas_data.load_data()
    # Set DCO mask
    compas_data.set_dco_mask(args.dco_type)
    # Compute delay times and metallicity grid
    compas_data.compute_delay_times()
    compas_data.compute_metallicity_grid()
    # Configure MSSFR
    mssfr = MSSFR(metallicity_grid=args.metallicity_grid)
    # Instantiate integrator
    integrator = CosmicIntegrator(
        compas_data=compas_data,
        mssfr=mssfr,
        redshift_min=args.z_min,
        redshift_max=args.z_max,
        n_redshift_bins=args.n_bins,
        gw_snr_threshold=args.gw_threshold,
    )
    # Build shells and birth arrays
    integrator.create_redshift_shells()
    integrator.compute_birth_arrays()
    # Integrate
    integrator.integrate(n_workers=args.n_workers)
    # Save results
    integrator.save_results(args.output_file)
    print(f"Rates saved to {args.output_file}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1:])
