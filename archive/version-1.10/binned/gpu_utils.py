"""Unified CPU/GPU array module selection.

This module attempts to import CuPy in order to perform numerical
computations on a GPU when available.  If CuPy is unavailable, it
falls back to NumPy.  It exposes a single public attribute ``xp``
which aliases the imported module.  The stub always sets ``xp`` to
``numpy``.
"""

from __future__ import annotations

import numpy as np

try:
    # In a full implementation this would attempt to import cupy.
    import cupy as cp  # type: ignore  # noqa: F401
    xp = cp  # pragma: no cover
except Exception:
    # If CuPy is not available, fall back to NumPy.
    xp = np
