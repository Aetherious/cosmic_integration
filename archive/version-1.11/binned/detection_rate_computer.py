"""Deprecated binned detection rate computation.

The original code provided a function to compute a two dimensional
detection rate grid over chirp mass and redshift.  In the dasein
rewrite this function is not supported.  The placeholder defined here
raises :class:`NotImplementedError` whenever called.
"""

from __future__ import annotations

from typing import Any


def compute_binned_detection_rates(*args: Any, **kwargs: Any) -> None:
    """Placeholder for the deprecated binned detection rate routine."""
    raise NotImplementedError(
        "compute_binned_detection_rates is not available in the dasein rewrite. "
        "Please refer to the original cosmic_integration package for binned analyses."
    )