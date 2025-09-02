
from __future__ import annotations
import numpy as np, astropy.units as u
from astropy.cosmology import Planck18 as _Planck18, FlatLambdaCDM
from astropy.cosmology import Cosmology as _Cosmo

def get_cosmology(cosmology=None) -> _Cosmo:
    if isinstance(cosmology, _Cosmo): return cosmology
    if cosmology is None: return _Planck18
    if isinstance(cosmology, str):
        return getattr(__import__("astropy.cosmology", fromlist=[cosmology]), cosmology)
    if isinstance(cosmology, dict): return FlatLambdaCDM(**cosmology)
    raise TypeError("Unsupported cosmology argument.")

def z_grid(z_max: float, dz: float) -> np.ndarray:
    n = int(np.floor(z_max/dz)); return np.linspace(0.0, n*dz, n+1)

def cosmology_arrays(cosmo: _Cosmo, z: np.ndarray):
    d_l = cosmo.luminosity_distance(z).to(u.Gpc).value
    dVdz_dsr = cosmo.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value
    dVdz = 4.0*np.pi*dVdz_dsr
    t_lb = cosmo.lookback_time(z).to(u.Gyr).value
    return d_l, dVdz, t_lb
