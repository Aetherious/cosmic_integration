"""Utility functions to instantiate cosmological models with Astropy.

This module provides helper functions to create instances of
`astropy.cosmology.FLRW` subclasses given a variety of input
representations.  Users may pass ``None`` to obtain the default
cosmology, a string corresponding to a built-in cosmology name, an
existing cosmology instance, or a dictionary of parameters for
`~astropy.cosmology.LambdaCDM` or `~astropy.cosmology.wCDM`.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from astropy import cosmology as cosmo

COSMOLOGY_CACHE = {
    "object": cosmo.Planck18,
    "name": cosmo.Planck18.name,
}


def get_cosmology(cosmology: Optional[Union[cosmo.FLRW, str, Dict[str, float]]] = None) -> cosmo.FLRW:
    """Return a cosmology instance from various descriptions.

    Parameters
    ----------
    cosmology:
        Description of the desired cosmology.  One of

        * ``None`` – Return the default cosmology (Planck18).
        * Instance of :class:`astropy.cosmology.FLRW` – Returned verbatim.
        * ``str`` – Name of a known Astropy cosmology (e.g. ``"Planck18"``).
        * ``dict`` – Parameter dictionary used to instantiate a
          `~astropy.cosmology.LambdaCDM` or `~astropy.cosmology.wCDM`.

    Returns
    -------
    cosmology:
        Instance of :class:`astropy.cosmology.FLRW` matching the
        description.
    """
    # If no cosmology is given return the cached default
    if cosmology is None:
        result = COSMOLOGY_CACHE["object"]
    elif isinstance(cosmology, cosmo.FLRW):
        result = cosmology
    elif isinstance(cosmology, str):
        result = getattr(cosmo, cosmology)
    elif isinstance(cosmology, dict):
        # Select between wCDM and LambdaCDM depending on presence of w0
        if "Ode0" in cosmology:
            if "w0" in cosmology:
                result = cosmo.wCDM(**cosmology)
            else:
                result = cosmo.LambdaCDM(**cosmology)
        else:
            result = cosmo.FlatLambdaCDM(**cosmology)
    else:
        raise TypeError(f"Unsupported cosmology specification: {type(cosmology)!r}")

    # Cache the chosen cosmology for later reference
    COSMOLOGY_CACHE["object"] = result
    COSMOLOGY_CACHE["name"] = result.name or repr(result)
    return result


def set_cosmology(cosmology: Optional[Union[cosmo.FLRW, str, Dict[str, float]]] = None) -> None:
    """Set the default cosmology used by :func:`get_cosmology`.

    Parameters
    ----------
    cosmology:
        New cosmology description (see :func:`get_cosmology` for accepted
        types).  If ``None``, the default Planck18 cosmology is restored.
    """
    get_cosmology(cosmology)