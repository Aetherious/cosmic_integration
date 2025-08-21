"""Deprecated SNR grid utilities for binned analyses.

This stub replaces the SNRGrid class from the original binned module.
Computing pre tabulated grids of signal to noise ratio is beyond the
scope of the dasein rewrite.  Users interested in these advanced
features should revert to the original cosmic integration code.
"""

from __future__ import annotations


class SNRGrid:
    """Placeholder for the deprecated SNRGrid class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "SNRGrid is not available in the dasein rewrite. Use the original package for SNR grids."
        )