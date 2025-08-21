"""Deprecated binary population container.

This stub replaces the original heavy weight :class:`BinaryPopulation` class used
in the binned cosmic integration implementation.  The original class
provided extensive support for loading populations from HDF5, drawing
samples from an initial mass function, computing chirp masses and
mass ratios, and filtering populations based on various criteria.

The dasein rewrite does not expose a fully featured binned pipeline.
Instead, users should employ the high level API in
:mod:`cosmic_integration_dasein.core` for rate calculations and
utilise external libraries for more advanced binned analyses.

The class defined here raises :class:`NotImplementedError` when
instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BinaryPopulation:
    """Placeholder for the deprecated BinaryPopulation class.

    Attempting to use this class will result in an exception.  The
    original implementation can be found in the upstream
    ``cosmic_integration`` package.  This stub is provided solely to
    satisfy import statements within the binned subpackage.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "BinaryPopulation is not implemented in the dasein rewrite. "
            "Please use cosmic_integration_original for binned analyses."
        )