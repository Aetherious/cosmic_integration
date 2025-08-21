"""Top level package for the cosmic integration dasein rewrite.

This package provides a refactored and simplified API for computing formation,
merger and detection rates of compact binary populations from HDF5 input
datasets.  The goal of this rewrite is to expose a clean, well‑documented
interface and to separate concerns between the computational core, I/O
handling, plotting utilities and optional binned routines.  All code is
written to be PEP 8 compliant, uses explicit type hints and includes
minimal docstrings to aid discovery and maintenance.

The primary entry point of the package is :func:`~cosmic_integration_dasein.core.find_detection_rate`.  A
companion command line interface living in :mod:`~cosmic_integration_dasein.cli` allows
rate calculations to be executed from the shell.  A dedicated plotting module
:mod:`~cosmic_integration_dasein.plotting` consumes the numpy archive produced by the
core and generates diagnostic figures.

The :mod:`~cosmic_integration_dasein.binned` subpackage contains a lightly
refactored copy of the legacy binned implementation.  This code exists
primarily for backwards compatibility and will not receive further
optimisation.  It is exposed here to allow reproducibility of previous
analyses and should be considered deprecated.

"""

from .core import find_detection_rate  # noqa: F401

__all__ = [
    "find_detection_rate",
]