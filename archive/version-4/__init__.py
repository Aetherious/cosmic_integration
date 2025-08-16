"""Top level package for the restructured cosmic integration utilities.

This package exposes the core public API of the cosmic integration codebase.
It provides convenient accessors for the main data container, the cosmic
integrator and the metallicity-specific star-formation rate class.  To
instantiate a workflow, import the desired classes from this package:

.. code-block:: python

    from updated_cosmic_integration import CompasData, CosmicIntegrator, MSSFR
    from pathlib import Path

    # Load your population synthesis data
    compas_data = CompasData(path=Path("/path/to/compas_output.h5"))
    compas_data.load_data()

    # Configure a metallicity specific star formation model
    mssfr = MSSFR(metallicity_grid=[0.0002, 0.002, 0.02])

    # Set up and run the cosmological integration
    integrator = CosmicIntegrator(compas_data=compas_data, mssfr=mssfr)
    integrator.create_redshift_shells()
    integrator.compute_birth_arrays()
    integrator.integrate()

The remainder of the modules in this package are considered internal
implementation details and their APIs may change without notice.
"""

from .compas_data import CompasData
from .cosmic_integrator import CosmicIntegrator
from .mssfr import MSSFR
from .cosmology import get_cosmology, set_cosmology

# Expose the binned integrator subpackage as a namespace
from . import binned

__all__ = [
    "CompasData",
    "CosmicIntegrator",
    "MSSFR",
    "get_cosmology",
    "set_cosmology",
    "binned",
]