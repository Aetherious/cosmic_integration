"""Simple plotting utilities for cosmic_integration_dasein.

This module defines helper functions and a command line interface to
visualise the output of :func:`~cosmic_integration_dasein.rate.find_detection_rate`.
It expects a NumPy ``.npz`` file containing the arrays ``detection_rate``,
``formation_rate``, ``merger_rate`` and ``redshifts`` as saved by
the integration.  A summary plot showing the total rates as
functions of redshift is produced and saved as ``plot.png`` in
the current working directory.

Example
-------

To plot results saved in ``/path/to/results.npz``::

    python -m cosmic_integration_dasein.plotting --input /path/to/results.npz

"""
from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(npz_path: str) -> dict:
    """Load rate arrays from a NumPy archive.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` file created by the rate calculation.

    Returns
    -------
    dict
        Dictionary of arrays contained in the file.  Keys include
        ``detection_rate``, ``formation_rate``, ``merger_rate`` and
        ``redshifts``.
    """
    with np.load(npz_path) as data:
        return {key: data[key] for key in data}


def plot_rates(results: dict, output_filename: str = "plot.png") -> str:
    """Generate a summary plot of the total rates vs redshift.

    Parameters
    ----------
    results : dict
        Dictionary containing at least the keys ``detection_rate``,
        ``formation_rate``, ``merger_rate`` and ``redshifts``.
    output_filename : str, optional
        Name of the output image file.  Defaults to 'plot.png'.

    Returns
    -------
    str
        Path to the saved image.

    Notes
    -----
    The plot shows the sum of the rates over all binaries as a
    function of redshift.  The x‑axis extends over the full redshift
    grid while the y‑axis is logarithmic to capture the wide range of
    values.  Users can customise the plot by modifying this
    function.
    """
    detection_rate = results.get("detection_rate")
    formation_rate = results.get("formation_rate")
    merger_rate = results.get("merger_rate")
    redshifts = results.get("redshifts")
    if any(x is None for x in [detection_rate, formation_rate, merger_rate, redshifts]):
        raise ValueError("Incomplete results dictionary; required keys are missing.")
    # Sum over binaries
    total_detection = np.sum(detection_rate, axis=0)
    total_formation = np.sum(formation_rate, axis=0)
    total_merger = np.sum(merger_rate, axis=0)
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(redshifts[: len(total_detection)], total_detection, label="Detection rate")
    ax.plot(redshifts, total_merger, label="Merger rate", ls="--")
    ax.plot(redshifts, total_formation, label="Formation rate", ls=":")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Rate [yr$^{-1}$ Gpc$^{-3}$]")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
    # Save figure
    fig.savefig(output_filename, dpi=300)
    return os.path.abspath(output_filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the total formation, merger and detection rates as a function of redshift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to the NPZ file containing rate arrays.")
    parser.add_argument("--output", default="plot.png", help="Filename for the saved plot image.")
    return parser


def main(argv: Any | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = load_results(args.input)
    output_path = plot_rates(results, args.output)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
