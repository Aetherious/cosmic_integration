"""Plotting utilities for the binned cosmic integrator.

This module collects functions that produce common diagnostic plots
used when analysing detection rate matrices, star formation histories
and metallicity distributions.  Matplotlib is used for all figures.
The stub defines the public API but does not implement any plotting
logic.

Functions
---------
plot_detection_rate_matrix(detection_rate, chirp_masses, redshifts, normalise=True, annotation=None)
    Draw a two–dimensional heatmap of detection rates with marginal
    histograms.
plot_sfr_and_metallicity(redshift, sfr, metallicities, dPdlogZ, p_draw_metallicity,
                          metallicity_label, sf_label, redshift_range, logZ_range)
    Plot cosmic star formation history and metallicity distribution.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_detection_rate_matrix(
    detection_rate: np.ndarray,
    chirp_masses: Sequence[float],
    redshifts: Sequence[float],
    normalise: bool = True,
    annotation: Optional[str] = None,
) -> plt.Figure:
    """Plot a detection rate matrix as a heatmap with marginal histograms.

    Parameters
    ----------
    detection_rate:
        Two–dimensional array of detection rates indexed by chirp mass and
        redshift.
    chirp_masses, redshifts:
        Bin centres corresponding to the axes of ``detection_rate``.
    normalise:
        Whether to normalise the heatmap relative to its maximum.
    annotation:
        Optional string to annotate the plot (e.g. total detection rate).

    Returns
    -------
    fig:
        Matplotlib figure object containing the plot.  The stub
        implementation returns an empty figure.
    """
    return plt.figure()


def plot_sfr_and_metallicity(
    redshift: Sequence[float],
    sfr: Sequence[float],
    metallicities: Sequence[float],
    dPdlogZ: np.ndarray,
    p_draw_metallicity: Sequence[float],
    metallicity_label: str,
    sf_label: str,
    redshift_range: Sequence[float],
    logZ_range: Sequence[float],
) -> plt.Figure:
    """Plot the cosmic star formation history and metallicity distribution.

    Parameters
    ----------
    redshift, sfr:
        Redshift values and corresponding star formation rate history.
    metallicities, dPdlogZ:
        Metallicities and metallicity distribution evaluated on a grid.
    p_draw_metallicity:
        Normalisation factors used when drawing metallicities from a flat
        distribution in logarithmic space.
    metallicity_label, sf_label:
        Strings summarising the metallicity and star formation models.
    redshift_range, logZ_range:
        Ranges of redshift and logarithmic metallicity plotted on the axes.

    Returns
    -------
    fig:
        Matplotlib figure containing three subplots.  The stub returns
        an empty figure.
    """
    return plt.figure()
