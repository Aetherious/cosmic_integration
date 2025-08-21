"""Utilities for binning two–dimensional data.

This module defines a simple function for binning a two–dimensional
array along one axis according to a set of one–dimensional bin
edges.  It is used by the binned detection rate computer to
coarsely bin formation and merger rate arrays.  The implementation
here is a stub and does not perform any computation.

Functions
---------
bin_2d_data(data2d, data1d, bins, axis=0)
    Bin a 2D array along one axis according to a 1D array of values.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bin_2d_data(
    data2d: np.ndarray,
    data1d: np.ndarray,
    bins: Sequence[float],
    axis: int = 0,
) -> np.ndarray:
    """Bin a two–dimensional array along a single axis.

    Parameters
    ----------
    data2d:
        Two–dimensional array to be binned.  The axis along which the
        binning is performed must have the same length as ``data1d``.
    data1d:
        One–dimensional array used to determine which row or column of
        ``data2d`` falls into which bin.  Typically this is a vector of
        parameters corresponding to the rows or columns of ``data2d``.
    bins:
        Sequence of bin edges.  The number of bins is ``len(bins)``.
    axis:
        Axis along which to sum the data.  If ``0`` (default), each row
        of ``data2d`` corresponds to an entry in ``data1d``; if ``1``,
        columns correspond.

    Returns
    -------
    binned_data:
        A two–dimensional array with the same number of columns (if
        ``axis=0``) or rows (if ``axis=1``) as the input, and a length
        along the binned axis equal to ``len(bins)``.  Each slice
        contains the sum of entries in ``data2d`` whose corresponding
        value in ``data1d`` falls into the appropriate bin.  The stub
        implementation returns an empty array of the correct shape.
    """
    # The real implementation would digitise ``data1d`` into ``bins``
    # and sum ``data2d`` accordingly.  Here we construct an array of
    # zeros with the expected shape.
    binned_shape = list(data2d.shape)
    binned_shape[axis] = len(bins)
    return np.zeros(binned_shape)
