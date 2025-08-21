"""Top level package for the restructured cosmic integration utilities.

This package exposes the core public API of the cosmic integration codebase.
It provides convenient accessors for the main data container, the cosmic
integrator and the metallicity–specific star–formation rate class.  To
instantiate a workflow, import the desired classes from this package:

.. code-block:: python

    from cosmic_integration_dasein import CompasData, CosmicIntegrator, MSSFR
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
# The binned subpackage provides cosmological integration on discretised grids.
# It depends on optional third-party libraries (e.g. astropy) that may not
# be available in all deployment environments.  To avoid import errors when
# those dependencies are missing, we attempt to import the subpackage here
# but fall back to a ``None`` placeholder if it fails.  Client code should
# check for ``None`` before accessing ``binned``.
try:
    from . import binned  # type: ignore[import]
except Exception:
    binned = None  # type: ignore[assignment]

__all__ = [
    "CompasData",
    "CosmicIntegrator",
    "MSSFR",
    "get_cosmology",
    "set_cosmology",
    # 'binned' may be None if optional dependencies are missing
    "binned",
]