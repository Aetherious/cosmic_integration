"""Deprecated 2D binning utilities.

This module previously provided functions to bin two‑dimensional data
according to a one‑dimensional axis.  The full implementation has
been removed in this refactor.  Users requiring the original
functionality should rely on the upstream ``cosmic_integration``
package.  All functions defined here raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["bin_2d_data"]


def bin_2d_data(*args: Any, **kwargs: Any) -> None:
    """Placeholder for the deprecated 2D binning routine.

    Raises
    ------
    NotImplementedError
        Always raised to indicate that the function is not available
        in the dasein rewrite.
    """
    raise NotImplementedError(
        "bin_2d_data has been removed in the dasein rewrite. Use numpy.histogram2d or ``cosmic_integration_original`` for binned computations."
    )