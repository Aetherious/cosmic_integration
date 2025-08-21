"""Utilities for generating frame files with injected gravitational waves.

This module defines convenience functions to create GW frame files by
injecting signals into simulated interferometer data.  It relies on
the `bilby` and `gwpy` libraries, which are not imported in this stub
module.  The functions provide a high level interface suitable for
generating synthetic data for training or validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


def multiple_injections(
    path: str,
    filename: str,
    dz: float,
    observation_time: float,
    t0: float,
) -> int:
    """Inject multiple events into a detector network.

    Parameters
    ----------
    path:
        Directory containing a text file of merger parameters.
    filename:
        Name of the file within ``path`` listing events to inject.
    dz:
        Size of the redshift bin used to compute the injection probability.
    observation_time:
        Total observation time in days.
    t0:
        Reference GPS time.

    Returns
    -------
    n_injections:
        Number of successful injections written to the frame file.
    """
    # Implementation omitted in stub
    pass


def one_injection(
    m1: float,
    m2: float,
    z: float,
    distance: float,
    t0: float,
    observation_time: float,
) -> float:
    """Inject a single compact binary merger into the detector network.

    Parameters
    ----------
    m1, m2:
        Source-frame component masses (solar masses).
    z:
        Redshift of the source.
    distance:
        Luminosity distance to the source (in Mpc).
    t0:
        Reference GPS time.
    observation_time:
        Duration over which to place the injection (in days).

    Returns
    -------
    start_time:
        GPS start time of the injection within the observation window.
    """
    # Implementation omitted in stub
    pass