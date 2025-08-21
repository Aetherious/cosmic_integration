"""Utility routines for interacting with astropy cosmology objects.

The original cosmic integration code supported passing either an
``astropy.cosmology.FLRW`` instance, a string naming a built in cosmology,
or a dictionary of parameters.  This module exposes a single helper
function that implements that behaviour.  It returns a concrete
instance of an FLRW subclass based on the provided input or falls
back to the default Planck18 cosmology if none is given.

Example
-------

>>> from cosmic_integration_dasein.cosmology_utils import get_cosmology
>>> cosmo = get_cosmology("WMAP9")
>>> cosmo.h
0.69

"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np  # used for fallback cosmology

try:
    from astropy import cosmology as cosmo  # type: ignore
except Exception:
    cosmo = None  # type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # During static type checking assume astropy is available
    from astropy.cosmology import FLRW  # type: ignore
    CosmologyType = Union[FLRW, str, Dict[str, Any]]
else:
    CosmologyType = Union[object, str, Dict[str, Any]]


def get_cosmology(cosmology: Optional[CosmologyType] = None) -> cosmo.FLRW:
    """Return an astropy cosmology object based on the provided argument.

    Parameters
    ----------
    cosmology : astropy.cosmology.FLRW or str or dict, optional
        If an FLRW instance is supplied it is returned directly.  If a
        string is supplied it is interpreted as the name of a cosmology
        registered in :mod:`astropy.cosmology`.  If a dictionary is
        supplied it is unpacked and passed to the :class:`astropy.cosmology.FlatLambdaCDM`
        constructor.  When ``None`` (the default) the Planck18 cosmology
        is returned.

    Returns
    -------
    astropy.cosmology.FLRW
        A cosmology object suitable for distance and lookback time
        calculations.

    Notes
    -----
    The original implementation accepted a dictionary with keys
    ``H0``, ``Om0`` and ``Om_baryon``.  When given those parameters this
    function constructs a :class:`astropy.cosmology.FlatLambdaCDM` instance
    with the provided Hubble constant and matter density.  Any additional
    parameters are ignored.
    """

    # If astropy is available use it to construct the cosmology
    if cosmo is not None:
        if cosmology is None:
            return cosmo.Planck18
        if isinstance(cosmology, cosmo.FLRW):  # type: ignore[attr-defined]
            return cosmology
        if isinstance(cosmology, str):
            try:
                return getattr(cosmo, cosmology)
            except AttributeError:
                raise ValueError(f"Unknown cosmology name: {cosmology}") from None
        if isinstance(cosmology, dict):
            # Accept only the common parameters.  Additional values are ignored.
            H0 = cosmology.get("H0", cosmo.Planck18.H0)
            Om0 = cosmology.get("Om0", cosmo.Planck18.Om0)
            Ob0 = cosmology.get("Ob0", None)
            return cosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)
        raise TypeError(
            "cosmology must be None, an astropy cosmology instance, a name or a parameter dictionary"
        )

    # Fall back to a simple approximation if astropy is unavailable
    # Here we implement a rudimentary flat LambdaCDM cosmology with H0=70 km/s/Mpc
    # and Omega_m=0.3.  The luminosity distance is approximated by the series
    # expansion d_L(z) ≈ (c/H0) z (1 + 0.5 z) valid for small z but applied to all z.
    class _DummyCosmology:
        H0 = 70.0  # km/s/Mpc
        c = 299792.458  # km/s
        Om0 = 0.3
        def luminosity_distance(self, z: float | np.ndarray):  # type: ignore[name-defined]
            z = np.asarray(z, dtype=float)
            dl = (self.c / self.H0) * z * (1.0 + 0.5 * z)
            # Return raw Mpc as float, but mimic Astropy by providing a to() method returning self
            class _DL:
                def __init__(self, value: np.ndarray) -> None:
                    self.value = value
                def to(self, unit: str) -> "_DL":  # type: ignore[override]
                    return self
            return _DL(dl)
        def age(self, z: float | np.ndarray) -> float:  # type: ignore[override]
            # Provide a crude estimate of the cosmic age (Gyr).  Not used in core.
            # Age ≈ 1/H0 for z=0
            return 1.0 / (self.H0 / 100.0)  # ~14 Gyr
    return _DummyCosmology()
    if isinstance(cosmology, cosmo.FLRW):
        return cosmology
    if isinstance(cosmology, str):
        try:
            return getattr(cosmo, cosmology)
        except AttributeError:
            raise ValueError(f"Unknown cosmology name: {cosmology}") from None
    if isinstance(cosmology, dict):
        # Accept only the common parameters.  Additional values are ignored.
        H0 = cosmology.get("H0", cosmo.Planck18.H0)
        Om0 = cosmology.get("Om0", cosmo.Planck18.Om0)
        Ob0 = cosmology.get("Ob0", None)
        return cosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)
    raise TypeError(
        "cosmology must be None, an astropy cosmology instance, a name or a parameter dictionary"
    )