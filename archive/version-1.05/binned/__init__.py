"""Binned cosmic integration subpackage.

This subpackage contains an alternative implementation of the cosmic
integration framework that precomputes rates on a two–dimensional grid
of delay time and redshift.  The binned approach can accelerate
scenarios where the same population is evaluated repeatedly under
different cosmological models or detector sensitivities.  Only the
public API is defined in the stubs provided here.
"""

from .binary_population import BinaryPopulation
from .conversions import m1_m2_to_chirp_mass, m1_m2_to_eta
from .cosmological_model import CosmologicalModel
from .detection_matrix import DetectionMatrix
from .detection_rate_computer import compute_binned_detection_rates as DetectionRateComputer
from .gpu_utils import xp as gpu_array_module
from .io import (
    recursively_load_dict_contents_from_group,
    recursively_save_dict_contents_to_group,
)
from .plotting import (
    plot_detection_rate_matrix,
    plot_sfr_and_metallicity,
)
from .snr_grid import SNRGrid
from .stellar_type import BH, NS, WD

__all__ = [
    "BinaryPopulation",
    "m1_m2_to_chirp_mass",
    "m1_m2_to_eta",
    "CosmologicalModel",
    "DetectionMatrix",
    "DetectionRateComputer",
    "gpu_array_module",
    "recursively_load_dict_contents_from_group",
    "recursively_save_dict_contents_to_group",
    "plot_detection_rate_matrix",
    "plot_sfr_and_metallicity",
    "SNRGrid",
    "BH",
    "NS",
    "WD",
]