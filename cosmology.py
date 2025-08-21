"""Cosmology helper functions.

This module provides a thin wrapper around the `astropy.cosmology`
package to retrieve a cosmological model.  The function
:func:`get_cosmology` accepts either an existing
`~astropy.cosmology.FLRW` instance, the name of a built in cosmology,
or a dictionary of parameters.  It returns an appropriate cosmology
object which can then be used to convert redshifts to ages,
distances and volumes.

The default cosmology is Planck18.  Users may override this
by supplying a different cosmology to functions in this package.

Example
-------

>>> from cosmic_integration_dasein.cosmology import get_cosmology
>>> cosmo = get_cosmology()  # returns Planck18 by default
>>> cosmo.age(1.0).value  # get age of the universe at z=1

"""
from typing import Union, Dict, Optional, Any

# NOTE:
# This module requires the `astropy` package.  The original
# implementation included a fallback `SimpleCosmology` class to
# approximate a flat ΛCDM universe when `astropy` was unavailable.
# In the dasein rewrite astropy is treated as a hard dependency and
# the fallback is no longer selected.

try:  # astropy is a required dependency.
    from astropy import cosmology as _cosmo_mod
except Exception as exc:
    # Immediately raise an informative error if astropy cannot be imported.
    raise ImportError(
        "The astropy package is required by cosmic_integration_dasein. "
        "Please install astropy and its dependencies."
    ) from exc


COSMO_TYPE = Union[_cosmo_mod.FLRW, str, Dict[str, float]]
DEFAULT_COSMOLOGY = _cosmo_mod.Planck18
COSMOLOGY = [DEFAULT_COSMOLOGY, DEFAULT_COSMOLOGY.name]

def get_cosmology(cosmology: Optional[COSMO_TYPE] = None) -> Any:
    """Return a cosmology object using astropy.

    Parameters
    ----------
    cosmology : None, astropy.cosmology.FLRW, str or dict, optional
        The desired cosmology.  If ``None`` the default Planck18
        cosmology is returned.  If an instance of
        :class:`~astropy.cosmology.FLRW` is provided it is used
        directly.  A string input should correspond to the name
        of a built in astropy cosmology (e.g. ``'Planck18'`` or
        ``'FlatLambdaCDM'``).  A dictionary will be passed to
        :class:`~astropy.cosmology.FlatLambdaCDM` or
        :class:`~astropy.cosmology.LambdaCDM` depending on the
        keys present.  Any other type will raise a ``TypeError``.

    Returns
    -------
    astropy.cosmology.FLRW
        The cosmology instance.
    """
    # Resolve input into an astropy cosmology:
    if cosmology is None:
        cosm = DEFAULT_COSMOLOGY
    elif isinstance(cosmology, _cosmo_mod.FLRW):
        cosm = cosmology
    elif isinstance(cosmology, str):
        try:
            cosm = getattr(_cosmo_mod, cosmology)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown cosmology '{cosmology}'. Check astropy.cosmology for valid names."
            ) from exc
    elif isinstance(cosmology, dict):
        # Determine which class to use based on parameter names:
        if "Ode0" in cosmology:
            # Non flat cosmology:
            if "w0" in cosmology:
                cosm = _cosmo_mod.wCDM(**cosmology)
            else:
                cosm = _cosmo_mod.LambdaCDM(**cosmology)
        else:
            cosm = _cosmo_mod.FlatLambdaCDM(**cosmology)
    else:
        raise TypeError(
            "cosmology must be None, an astropy.cosmology.FLRW instance, a string or a dict"
        )
    
    # Cache the cosmology and its name:
    COSMOLOGY[0] = cosm
    COSMOLOGY[1] = cosm.name if getattr(cosm, "name", None) else repr(cosm)
    return cosm

def set_cosmology(cosmology: Optional[COSMO_TYPE] = None) -> None:
    """Set the default cosmology used by this package."""
    _ = get_cosmology(cosmology)
