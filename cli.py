"""Command line interface for cosmic_integration_dasein.

This script exposes the :func:`~cosmic_integration_dasein.rate.find_detection_rate`
function via a set of command line arguments.  The interface is
deliberately minimal: the user specifies the path to the COMPAS
output file and optional parameters controlling the simulation and
integration.  The script will compute detection, formation and
merger rates and save the results to a compressed NumPy archive.

Example
-------

Run with default settings on a COMPAS file::

    python -m cosmic_integration_dasein.cli --path /path/to/COMPAS_Output.h5

Specify a different DCO type and detection threshold::

    python -m cosmic_integration_dasein.cli --path /path/to/COMPAS_Output.h5 \
        --dco-type BHNS --snr-threshold 12

Use parallel processing with four workers::

    python -m cosmic_integration_dasein.cli --path /path/to/COMPAS_Output.h5 \
        --n-workers 4

"""
from __future__ import annotations

import argparse
from typing import Any

try: # NumPy is a required dependency.
    import numpy as np
except Exception as exc:
    # Immediately raise an informative error if NumPy cannot be imported.
    raise ImportError(
        "The NumPy package is required for calculating total detection rate."
        "Please install NumPy and its dependencies."
    ) from exc

from .rate import find_detection_rate


def _positive_float(value: str) -> float:
    try:
        val = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid float") from exc
    if val < 0.0:
        raise argparse.ArgumentTypeError(f"{value!r} must be non-negative")
    return val


def _positive_int(value: str) -> int:
    try:
        val = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc
    if val < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be non-negative")
    return val


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute formation, merger and detection rates for a COMPAS simulation.\n"
            "The results are written to a NumPy .npz file containing the rate matrices and redshift array."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", required=True, help="Path to the COMPAS HDF5 output file.")
    parser.add_argument("--dco-type", default="BBH", choices=["BBH", "BHNS", "BNS", "all"], help="Type of compact binaries to include.")
    parser.add_argument("--weight-column", default=None, help="Column in BSE_Double_Compact_Objects containing sampling weights.")
    parser.add_argument("--no-hubble-mask", action="store_true", help="Disable the within Hubble time filter.")
    parser.add_argument("--pessimistic-cee", action="store_false", dest="pessimistic_cee", help="Allow optimistic common envelope events.")
    parser.add_argument("--allow-rlof", action="store_false", dest="no_rlof_after_cee", help="Allow immediate RLOF after common envelope.")
    parser.add_argument("--max-redshift", type=_positive_float, default=10.0, help="Maximum redshift to integrate to.")
    parser.add_argument("--max-redshift-detection", type=_positive_float, default=1.0, help="Maximum redshift for detectability calculation.")
    parser.add_argument("--redshift-step", type=_positive_float, default=0.001, help="Step size in redshift.")
    parser.add_argument("--z-first-sf", type=_positive_float, default=10.0, help="Redshift at which star formation begins.")
    parser.add_argument("--fbin", type=_positive_float, default=0.7, help="Binary fraction used in the sampling.")
    parser.add_argument("--snr-threshold", type=_positive_float, default=8.0, help="Detection threshold on the SNR.")
    parser.add_argument("--sensitivity", default="O1", choices=["design", "O1", "O3"], help="Detector sensitivity PSD.")
    parser.add_argument("--n-workers", type=_positive_int, default=1, help="Number of parallel workers to use.")
    parser.add_argument("--output", default="results.npz", help="Filename for the output NPZ archive.")
    return parser


def main(argv: Any | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Map boolean options:
    merges_hubble_time = not args.no_hubble_mask
    detection_rate, formation_rate, merger_rate, redshifts, _ = find_detection_rate(
        path=args.path,
        dco_type=args.dco_type,
        weight_column=args.weight_column,
        merges_hubble_time=merges_hubble_time,
        pessimistic_cee=args.pessimistic_cee,
        no_rlof_after_cee=args.no_rlof_after_cee,
        max_redshift=args.max_redshift,
        max_redshift_detection=args.max_redshift_detection,
        redshift_step=args.redshift_step,
        z_first_sf=args.z_first_sf,
        fbin=args.fbin,
        snr_threshold=args.snr_threshold,
        sensitivity=args.sensitivity,
        n_workers=args.n_workers,
        output_filename=args.output,
    )
    
    # Summarise results:
    total_detection_rate = np.sum(detection_rate, axis=0)
    print(
        f"Computed detection rates for {detection_rate.shape[0]} binaries over {detection_rate.shape[1]} redshift bins."
    )
    print(f"Total detection rate at z=0: {total_detection_rate[0]:.3e} yr^-1")


if __name__ == "__main__":
    main()
