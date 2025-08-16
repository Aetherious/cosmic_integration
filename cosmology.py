"""Minimal cosmology utilities for the cosmic integration pipeline.

This module implements a very lightweight cosmology class with a few
helper functions needed by the cosmic integrator.  The goal is to
provide enough functionality to convert between redshift and lookback
time and to compute comoving volume elements without requiring
heavyweight dependencies such as Astropy.  The formulas are
approximate but adequate for order of magnitude estimates.

The cosmological model assumes a flat Universe with a matter density
``omega_m`` and a cosmological constant ``omega_lambda = 1 -
omega_m``.  Radiation and curvature terms are neglected.  The Hubble
constant ``H0`` is expressed in units of km/s/Mpc.  A global
cosmology instance can be set via :func:`set_cosmology` and queried
via :func:`get_cosmology`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

__all__ = ["Cosmology", "get_cosmology", "set_cosmology"]


@dataclass
class Cosmology:
    """Simple flat ΛCDM cosmology.

    Parameters
    ----------
    H0:
        Hubble constant in km s⁻¹ Mpc⁻¹.
    omega_m:
        Matter density parameter.
    omega_lambda:
        Dark energy density parameter.  If ``None`` it is assumed
        ``1 - omega_m``, corresponding to a flat Universe.

    Notes
    -----
    The functions defined on this class operate on NumPy arrays and
    return values in convenient astrophysical units: lookback times in
    Gyr and comoving distances in Mpc.
    """

    H0: float = 67.74
    omega_m: float = 0.3089
    omega_lambda: Optional[float] = None

    def __post_init__(self) -> None:
        if self.omega_lambda is None:
            self.omega_lambda = 1.0 - self.omega_m
        # Speed of light in km/s
        self.c = 299792.458
        # Precompute H0 in s⁻¹
        # 1 Mpc = 3.0856775814913673e19 km
        self.H0_s = (self.H0 * 1000) / (3.0856775814913673e22)

    def E(self, z: np.ndarray) -> np.ndarray:
        """Dimensionless Hubble parameter E(z).

        E(z) = sqrt(omega_m (1+z)^3 + omega_lambda).

        Parameters
        ----------
        z:
            Redshift array.

        Returns
        -------
        ndarray
            Dimensionless expansion rate at each redshift.
        """
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.omega_m * (1 + z) ** 3 + self.omega_lambda)

    def lookback_time(self, z: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """Return the lookback time to a given redshift in Gyr.

        The lookback time is computed via numerical integration of

        .. math::

            t(z) = \frac{1}{H_0} \int_0^z \frac{dz'}{(1+z') E(z')}.

        Parameters
        ----------
        z:
            One–dimensional array of redshifts.
        n_steps:
            Number of integration points for the trapezoidal rule.

        Returns
        -------
        ndarray
            Lookback time in Gyr for each redshift.
        """
        z = np.asarray(z, dtype=float)
        # Set up integration grid
        # We integrate from 0 to z for each value independently.
        t_lb = np.zeros_like(z, dtype=float)
        # Precompute constant conversion to Gyr: 1/H0 in seconds times 1e-9 for Gyr
        sec_to_gyr = 1.0 / (self.H0_s) / (3.15576e16)  # seconds to Gyr
        for i, z_max in enumerate(z):
            if z_max == 0:
                t_lb[i] = 0.0
                continue
            zz = np.linspace(0.0, z_max, n_steps)
            dz = zz[1] - zz[0]
            integrand = 1.0 / ((1 + zz) * self.E(zz))
            integral = np.trapz(integrand, zz)
            t_lb[i] = sec_to_gyr * integral
        return t_lb

    def comoving_distance(self, z: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """Comoving radial distance to redshift ``z`` in Mpc.

        Parameters
        ----------
        z:
            One–dimensional array of redshifts.
        n_steps:
            Number of integration steps.

        Returns
        -------
        ndarray
            Comoving distance in Mpc.
        """
        z = np.asarray(z, dtype=float)
        d_c = np.zeros_like(z, dtype=float)
        for i, z_max in enumerate(z):
            if z_max == 0:
                d_c[i] = 0.0
                continue
            zz = np.linspace(0.0, z_max, n_steps)
            integrand = 1.0 / self.E(zz)
            integral = np.trapz(integrand, zz)
            d_c[i] = (self.c / self.H0) * integral
        return d_c

    def comoving_volume_element(self, z: np.ndarray) -> np.ndarray:
        """Return the differential comoving volume element dV/dz in Mpc³ per unit redshift.

        The differential comoving volume in a flat Universe is given by

        .. math::

            \frac{dV}{dz} = 4\pi \frac{c}{H_0} \frac{D_C(z)^2}{E(z)}.

        Parameters
        ----------
        z:
            One–dimensional array of redshifts.

        Returns
        -------
        ndarray
            Differential comoving volume in Mpc³ per unit redshift.
        """
        z = np.asarray(z, dtype=float)
        d_c = self.comoving_distance(z)
        return 4.0 * np.pi * (self.c / self.H0) * (d_c ** 2) / self.E(z)


# Global cosmology instance and cache
_COSMOLOGY: Cosmology = Cosmology()


def get_cosmology(params: Optional[Dict[str, Any]] = None) -> Cosmology:
    """Return the current cosmology or create a new one from a parameter dictionary.

    Parameters
    ----------
    params:
        Optional dictionary containing keys ``H0``, ``omega_m`` and
        ``omega_lambda``.  Missing ``omega_lambda`` will be inferred as
        ``1 - omega_m``.  If ``None``, the current global cosmology
        instance is returned.

    Returns
    -------
    Cosmology
        A cosmology instance matching the supplied parameters.
    """
    global _COSMOLOGY
    if params is None:
        return _COSMOLOGY
    # Update the cosmology using supplied parameters
    H0 = params.get("H0", _COSMOLOGY.H0)
    omega_m = params.get("omega_m", _COSMOLOGY.omega_m)
    omega_lambda = params.get("omega_lambda", None)
    _COSMOLOGY = Cosmology(H0=H0, omega_m=omega_m, omega_lambda=omega_lambda)
    return _COSMOLOGY


def set_cosmology(params: Optional[Dict[str, Any]] = None) -> None:
    """Update the global cosmology instance.

    Parameters
    ----------
    params:
        Dictionary of parameters.  See :func:`get_cosmology` for valid keys.
        Passing ``None`` resets the cosmology to the default values.
    """
    if params is None:
        # Reset to defaults
        get_cosmology({"H0": 67.74, "omega_m": 0.3089})
    else:
        get_cosmology(params)
