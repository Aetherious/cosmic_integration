"""Top level imports for the cosmic_integration_dasein package.

This package provides a minimalist re-implementation of the core
functionality found in the original ``cosmic_integration`` code base.
The goal is to preserve the essential behaviours required to compute
intrinsic formation, merger and detection rates for compact binary
populations while simplifying the code structure and enforcing
PEP 8 compliance.  Optional binned utilities are exposed through
the :mod:`cosmic_integration_dasein.binned` submodule but are
otherwise untouched.

Users should primarily interact with the :func:`find_detection_rate`
function in :mod:`cosmic_integration_dasein.rate` which wraps the
entire calculation pipeline.  To build command line interfaces
or generate plots see :mod:`cosmic_integration_dasein.cli` and
``plotting.py`` respectively.

"""

from .compas_data import CompasData
from .rate import find_detection_rate
from .cosmology import get_cosmology

__all__ = [
    "CompasData",
    "find_detection_rate",
    "get_cosmology",
]
