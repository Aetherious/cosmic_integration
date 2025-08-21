"""Command line interface for the cosmic integration dasein package.

The CLI provides a thin wrapper around :func:`~cosmic_integration_dasein.core.find_detection_rate`.
It accepts a handful of arguments controlling the population selection,
cosmology and detector threshold and writes the computed rates to disk
in NPZ format.  To inspect or plot the results use the companion
script in :mod:`~cosmic_integration_dasein.plotting`.

Example
-------

.. code-block:: console

    $ python -m cosmic_integration_dasein.cli \
        --input compas_data.npz \
        --output results.npz \
        --max-redshift 10 \
        --max-redshift-detection 1.0 \
        --redshift-step 0.01

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .core import find_detection_rate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Optionally override the list of arguments.  When ``None`` the
        arguments are taken from :data:`sys.argv`.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compute compact binary formation, merger and detection rates."
    )
    parser.add_argument("--input", required=True, help="Path to the population synthesis file (HDF5 or NPZ).")
    parser.add_argument("--output", default="results.npz", help="Path to save the results NPZ file.")
    parser.add_argument("--max-redshift", type=float, default=10.0, help="Maximum redshift in the grid.")
    parser.add_argument(
        "--max-redshift-detection", type=float, default=1.0, help="Maximum redshift for detection rates."
    )
    parser.add_argument(
        "--redshift-step", type=float, default=0.01, help="Step size of the redshift grid."
    )
    parser.add_argument("--m1-min", type=float, default=5.0, help="Minimum primary mass to consider (Msun).")
    parser.add_argument("--m1-max", type=float, default=150.0, help="Maximum primary mass to consider (Msun).")
    parser.add_argument("--m2-min", type=float, default=0.1, help="Minimum secondary mass to consider (Msun).")
    parser.add_argument(
        "--fbin", type=float, default=0.7, help="Fraction of stellar mass forming in binaries."
    )
    parser.add_argument(
        "--snr-threshold", type=float, default=8.0, help="Network SNR threshold for detection."
    )
    parser.add_argument(
        "--cosmology",
        default=None,
        help=(
            "Specify an astropy cosmology by name, or provide a JSON dictionary of parameters"
            " to construct a FlatLambdaCDM instance.  When omitted the Planck18 cosmology is used."
        ),
    )
    parser.add_argument(
        "--weight-column",
        default=None,
        help="Name of the dataset containing weights within the input file."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    This function is separated from the argument parser to facilitate
    testing.  It loads the input file, executes the detection rate
    calculation and writes the results to disk.

    Parameters
    ----------
    argv : list of str, optional
        List of command line arguments.  When ``None`` (default) the
        arguments are taken from :data:`sys.argv`.

    Returns
    -------
    int
        Exit status code.  ``0`` on success and non‑zero on error.
    """
    args = _parse_args(argv)
    # Parse cosmology argument: either JSON or string
    cosmo_arg: Any = None
    if args.cosmology:
        try:
            cosmo_arg = json.loads(args.cosmology)
        except json.JSONDecodeError:
            cosmo_arg = args.cosmology
    # Compute rates
    try:
        result: Dict[str, np.ndarray] = find_detection_rate(
            path=args.input,
            max_redshift=args.max_redshift,
            max_redshift_detection=args.max_redshift_detection,
            redshift_step=args.redshift_step,
            m1_min=args.m1_min,
            m1_max=args.m1_max,
            m2_min=args.m2_min,
            fbin=args.fbin,
            snr_threshold=args.snr_threshold,
            cosmology=cosmo_arg,
            weight_column=args.weight_column,
        )
    except Exception as exc:
        print(f"Error computing detection rate: {exc}", file=sys.stderr)
        return 1
    # Write results
    output_path = Path(args.output)
    np.savez(output_path, **result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())