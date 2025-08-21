"""Deprecated detection matrix utilities.

The detection matrix computed the expected number of detections in
each bin of chirp mass and redshift.  This functionality has been
dropped from the dasein rewrite.  The :class:`DetectionMatrix` defined
here raises an exception when constructed.
"""

from __future__ import annotations


class DetectionMatrix:
    """Placeholder for the deprecated DetectionMatrix class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "DetectionMatrix has been removed in the dasein rewrite. "
            "Use cosmic_integration_original for binned analyses."
        )