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
import math

# NOTE:
# This module requires the `astropy` package.  The original
# implementation included a fallback `SimpleCosmology` class to
# approximate a flat ΛCDM universe when `astropy` was unavailable.
# In the dasein rewrite astropy is treated as a hard dependency and
# the fallback is no longer selected.  The SimpleCosmology class is
# retained below purely for legacy reasons but is not used by
# `get_cosmology`.

try:  # astropy is now a required dependency
    from astropy import cosmology as _cosmo_mod  # type: ignore
    from astropy import units as _u  # type: ignore
except Exception as exc:  # pragma: no cover
    # Immediately raise an informative error if astropy cannot be imported.
    raise ImportError(
        "The astropy package is required by cosmic_integration_dasein. "
        "Please install astropy and its dependencies."
    ) from exc


class SimpleCosmology:
    """Fallback cosmology for when astropy is not installed.

    This class implements only the methods required by the rest of the
    package: :meth:`age`, :meth:`luminosity_distance` and
    :meth:`comoving_volume`.  It assumes a flat ΛCDM universe with
    parameters H0, Om0 and Ode0.  Radiation and curvature are
    neglected.  The calculations are approximate and intended
    primarily for testing and environments lacking astropy.  The
    output units follow the convention of astropy if it is available;
    otherwise plain floats are returned with implicit units (Gyr
    for age, Mpc for distances and Gpc³ for volumes).

    Parameters
    ----------
    H0 : float, optional
        Hubble constant in km/s/Mpc.  Defaults to 67.74.
    Om0 : float, optional
        Matter density parameter.  Defaults to 0.3089.
    Ode0 : float, optional
        Dark energy density parameter.  Defaults to 0.6911 (i.e.
        1 - Om0 for a flat universe).
    """

    def __init__(self, H0: float = 67.74, Om0: float = 0.3089, Ode0: float | None = None) -> None:
        self.H0 = float(H0)  # km/s/Mpc
        self.Om0 = float(Om0)
        if Ode0 is None:
            self.Ode0 = 1.0 - self.Om0
        else:
            self.Ode0 = float(Ode0)
        # Speed of light in km/s
        self.c = 299792.458

    def _E(self, z: float) -> float:
        """Dimensionless Hubble parameter E(z)."""
        return math.sqrt(self.Om0 * (1.0 + z) ** 3 + self.Ode0)

    def _comoving_distance(self, z: float) -> float:
        """Line of sight comoving distance in Mpc."""
        # Integrate 0..z of c/H0 / E(z')
        from scipy.integrate import quad  # type: ignore

        integrand = lambda zp: 1.0 / self._E(zp)
        val, _err = quad(integrand, 0.0, z)
        return (self.c / self.H0) * val

    def luminosity_distance(self, z: Any) -> Any:
        """Luminosity distance in Mpc.

        This returns either a float or a numpy array depending on
        the input type.  If astropy units are available the result
        will carry units of Mpc; otherwise a plain float or array is
        returned.
        """
        # Vectorise over numpy arrays
        import numpy as np

        z_arr = np.asarray(z, dtype=float)
        # Compute comoving distances for each element
        dc_vec = np.vectorize(self._comoving_distance)(z_arr)
        dl = (1.0 + z_arr) * dc_vec
        if _u is not None:
            return dl * _u.Mpc
        return dl

    def comoving_volume(self, z: Any) -> Any:
        """Comoving volume enclosed within redshift z in Gpc^3.

        Assumes a flat universe where the volume element is 4π r^2 dr.
        The result has units of Gpc^3 if astropy units are available.
        """
        import numpy as np

        z_arr = np.asarray(z, dtype=float)
        # Compute comoving distance (Mpc)
        r = np.vectorize(self._comoving_distance)(z_arr)
        # Convert to Gpc for cubic volume
        r_gpc = r / 1e3
        vol = (4.0 / 3.0) * math.pi * (r_gpc ** 3)
        if _u is not None:
            return vol * _u.Gpc ** 3
        return vol

    def age(self, z: Any) -> Any:
        """Age of the universe at redshift z.

        This method computes the cosmic age by evaluating the
        difference between the integral of ``1/((1+z') E(z'))``
        from 0 to a large cutoff and the integral from ``z`` to the
        same cutoff.  The result is multiplied by ``3.08567758e19 / H0``
        to convert from dimensionless units to seconds (using
        ``1/H0`` expressed in seconds) and then converted to Gyr.  If
        astropy units are available a Quantity in Gyr is returned.
        Otherwise a plain float or array of floats is returned.
        """
        from scipy.integrate import quad  # type: ignore
        import numpy as np

        def integrand(zp: float) -> float:
            return 1.0 / ((1.0 + zp) * self._E(zp))

        # Integrate to a finite upper limit as approximation to infinity
        z_max = 1e4
        # Compute I0 = ∫_0^{z_max} integrand(z') dz'
        I0, _ = quad(integrand, 0.0, z_max)
        z_arr = np.asarray(z, dtype=float)
        ages_gyr = np.empty_like(z_arr, dtype=float)
        # Precompute conversion from dimensionless integral to seconds
        conv_sec = 3.0856775814913673e19 / self.H0
        # 1 Gyr in seconds
        sec_in_gyr = 3.15576e16
        for idx, zz in np.ndenumerate(z_arr):
            # Compute I(z) = ∫_z^{z_max} integrand dz'
            if zz >= z_max:
                # At very high redshift age tends to zero
                I_z = 0.0
            else:
                I_z, _ = quad(integrand, zz, z_max)
            delta_I = I0 - I_z
            # Age in seconds
            age_sec = delta_I * conv_sec
            ages_gyr[idx] = age_sec / sec_in_gyr
        if _u is not None:
            return ages_gyr * _u.Gyr
        return ages_gyr


if _cosmo_mod is not None:
    # Type alias for the allowed input types when astropy is available
    COSMO_TYPE = Union[_cosmo_mod.FLRW, str, Dict[str, float]]
    DEFAULT_COSMOLOGY = _cosmo_mod.Planck18
    COSMOLOGY = [DEFAULT_COSMOLOGY, DEFAULT_COSMOLOGY.name]

    def get_cosmology(cosmology: Optional[COSMO_TYPE] = None) -> Any:
        # """Return a cosmology object.

        # When astropy is installed this simply returns an instance of
        # ``astropy.cosmology.FLRW`` using the supplied arguments.
        # Otherwise it returns a :class:`SimpleCosmology` instance.
        # """
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
        # Resolve input into an astropy cosmology
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
            # Determine which class to use based on parameter names
            if "Ode0" in cosmology:
                # Non flat cosmology
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
        # Cache the cosmology and its name
        COSMOLOGY[0] = cosm
        COSMOLOGY[1] = cosm.name if getattr(cosm, "name", None) else repr(cosm)
        return cosm

    def set_cosmology(cosmology: Optional[COSMO_TYPE] = None) -> None:
        """Set the default cosmology used by this package."""
        _ = get_cosmology(cosmology)
else:
    # Define fallback stubs when astropy is absent
    COSMO_TYPE = Union[SimpleCosmology, str, Dict[str, float]]
    DEFAULT_COSMOLOGY = SimpleCosmology()
    COSMOLOGY = [DEFAULT_COSMOLOGY, "SimpleCosmology"]

    def get_cosmology(cosmology: Optional[COSMO_TYPE] = None) -> SimpleCosmology:
        """Return a fallback SimpleCosmology instance.

        If a dictionary is provided its keys may include ``H0``,
        ``Om0`` and ``Ode0`` to override the default parameters.  A
        string input is ignored and results in the default
        cosmology.  Instances of :class:`SimpleCosmology` are passed
        through unchanged.  Any other type will raise an error.
        """
        if cosmology is None:
            cosm = DEFAULT_COSMOLOGY
        elif isinstance(cosmology, SimpleCosmology):
            cosm = cosmology
        elif isinstance(cosmology, dict):
            # Extract supported parameters; ignore unknown keys
            H0 = cosmology.get("H0", DEFAULT_COSMOLOGY.H0)
            Om0 = cosmology.get("Om0", DEFAULT_COSMOLOGY.Om0)
            Ode0 = cosmology.get("Ode0", DEFAULT_COSMOLOGY.Ode0)
            cosm = SimpleCosmology(H0=H0, Om0=Om0, Ode0=Ode0)
        elif isinstance(cosmology, str):
            # Without astropy we cannot interpret strings; use default
            cosm = DEFAULT_COSMOLOGY
        else:
            raise TypeError(
                "cosmology must be None, a SimpleCosmology instance, a dict, or a string when astropy is unavailable"
            )
        # Update cache
        COSMOLOGY[0] = cosm
        COSMOLOGY[1] = "SimpleCosmology"
        return cosm

    def set_cosmology(cosmology: Optional[COSMO_TYPE] = None) -> None:
        """Set the default fallback cosmology."""
        _ = get_cosmology(cosmology)
