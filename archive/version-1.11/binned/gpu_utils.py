"""Lightweight GPU/CPU array backend selection.

The original code attempted to use CuPy for GPU acceleration if
available and fell back to NumPy otherwise.  This behaviour is
retained here as a minimal convenience for users porting binned
pipelines.  When CuPy is unavailable or fails to import the module
exposes ``xp`` as NumPy.
"""

from __future__ import annotations

try:
    import cupy as cp  # type: ignore
    xp = cp  # expose CuPy as the array namespace
    gpu_available = True
except Exception:
    import numpy as np  # type: ignore
    xp = np
    gpu_available = False

__all__ = ["xp", "gpu_available"]