"""Deprecated cosmological model for binned analyses.

In the original implementation this module provided a class
encapsulating the star formation history, metallicity distribution
and cosmological distance measures on a fine redshift grid.  The
dasein rewrite eschews this complexity in favour of a simplified
pipeline.  As such the class below exists solely to satisfy imports
from legacy code and will raise an exception upon use.
"""

from __future__ import annotations


class CosmologicalModel:
    """Placeholder class for the deprecated binned cosmological model."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "CosmologicalModel is not implemented in the dasein rewrite. "
            "Use the top level API in cosmic_integration_dasein.core."
        )