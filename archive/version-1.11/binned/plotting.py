"""Deprecated plotting routines for binned analyses.

The original implementation included a number of routines for visualising
detected populations, including corner plots and rate matrices.  In the
dasein rewrite these functions have been removed.  Use the high level
plotting utilities in :mod:`cosmic_integration_dasein.plotting` instead.
"""

from __future__ import annotations

from typing import Any


def plot_detection_rate_matrix(*args: Any, **kwargs: Any) -> None:
    """Placeholder for the deprecated detection rate matrix plot."""
    raise NotImplementedError(
        "plot_detection_rate_matrix has been removed in the dasein rewrite. "
        "Refer to cosmic_integration_dasein.plotting for supported plots."
    )