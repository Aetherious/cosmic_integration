"""Modelling of the metallicity specific star formation rate (MSSFR).

The cosmic integration pipeline requires an estimate of how the cosmic
star formation rate (SFR) is distributed across metallicity.  A
``MSSFR`` instance encapsulates simple prescriptions for the cosmic
SFR history and the metallicity distribution at a given redshift.

This implementation offers two primary components:

* A Madau & Dickinson (2014) star formation history
  (:func:`_sfr_madau_dickinson`).  This widely used fit reproduces
  observations of the cosmic SFR density over a wide range of
  redshift.
* A log–normal metallicity distribution whose mean decreases with
  redshift.  The functional form of the mean metallicity as a
  function of redshift is loosely based on observed mass–metallicity
  relations but is not tied to a specific dataset.

More sophisticated implementations could include a galaxy stellar mass
function and a metallicity–dependent star formation efficiency, but
these are beyond the scope of this example.  Users may override the
default SFR and metallicity distribution by assigning custom callables
to the attributes :attr:`sfr_history` and :attr:`metallicity_distribution`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


def _sfr_madau_dickinson(z: np.ndarray) -> np.ndarray:
    """Return the cosmic star formation rate density of Madau & Dickinson (2014).

    The functional form is

    .. math::

        \mathrm{SFR}(z) = 0.015 \frac{(1+z)^{2.7}}{1 + \left(\frac{1+z}{2.9}\right)^{5.6}}

    in units of solar masses per year per cubic megaparsec.  The
    numerical prefactor has been tuned to match observational
    constraints but can be freely rescaled.

    Parameters
    ----------
    z:
        Redshift array.

    Returns
    -------
    ndarray
        SFR density for each redshift.
    """
    z = np.asarray(z, dtype=float)
    return 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)


def _lognormal_metallicity_distribution(
    z: np.ndarray, metallicity_grid: np.ndarray, mu0: float = -1.0, alpha: float = 0.3, sigma: float = 0.5
) -> np.ndarray:
    """Compute a redshift–dependent log–normal metallicity distribution.

    The distribution is expressed as a probability density over the
    metallicity grid.  The mean of the underlying log–normal in
    :math:`\log_{10}(Z/\mathrm{Z}_\odot)` is parametrised as

    .. math::

        \mu(z) = \mu_0 - \alpha z,

    where ``mu0`` is the mean at ``z=0`` and ``alpha`` controls how
    quickly the mean decreases with redshift.  The standard deviation
    ``sigma`` is assumed constant with redshift.  Solar metallicity
    corresponds to ``Z=0.02`` in these units.

    Parameters
    ----------
    z:
        Array of redshifts at which to evaluate the distribution.
    metallicity_grid:
        One–dimensional array of metallicities (mass fraction of metals).
    mu0:
        Mean of log10(Z/Z_solar) at z=0.
    alpha:
        Linear slope of the mean metallicity with redshift.
    sigma:
        Standard deviation of log10(Z/Z_solar).

    Returns
    -------
    ndarray
        A two–dimensional array of shape ``(len(z), len(metallicity_grid))``
        where each row contains the metallicity probability density
        normalised to unity.
    """
    z = np.asarray(z, dtype=float)
    metallicity_grid = np.asarray(metallicity_grid, dtype=float)
    if np.any(metallicity_grid <= 0):
        raise ValueError("Metallicity values must be strictly positive.")
    logZ = np.log10(metallicity_grid / 0.02)
    # Preallocate output
    pdf = np.zeros((z.size, metallicity_grid.size))
    for i, redshift in enumerate(z):
        mu = mu0 - alpha * redshift
        exponent = -0.5 * ((logZ - mu) / sigma) ** 2
        # Log–normal PDF in log10 space: f(Z) = 1/(Z ln 10 sigma sqrt(2pi)) * exp(-((log10 Z - mu)^2)/(2 sigma^2))
        p = np.exp(exponent) / (metallicity_grid * np.log(10) * sigma * np.sqrt(2 * np.pi))
        # Normalise
        pdf[i] = p / p.sum()
    return pdf


@dataclass
class MSSFR:
    """Representation of the metallicity specific star formation rate.

    The MSSFR combines a cosmic star formation history with a
    redshift–dependent metallicity distribution to assign weights to
    systems born at different cosmic times and metallicities.  Users
    may supply custom callables for the SFR history and the
    metallicity distribution; sensible defaults are provided.

    Parameters
    ----------
    metallicity_grid:
        Sequence of metallicity values (mass fraction of metals) at
        which the MSSFR will be evaluated.  If ``None``, a default
        three–bin grid ``[0.0002, 0.002, 0.02]`` is used.
    sfr_history:
        Callable returning the cosmic SFR density as a function of
        redshift.  If ``None``, the Madau–Dickinson 2014 fit is used.
    metallicity_distribution:
        Callable returning an array of metallicity distribution
        functions ``p(Z|z)``.  It must accept a redshift array and the
        metallicity grid, and return a 2D array with shape
        ``(len(z), len(metallicity_grid))``.  If ``None``, a
        log–normal distribution with a mean decreasing linearly with
        redshift is used.
    """

    metallicity_grid: Optional[Sequence[float]] = None
    sfr_history: Optional[Callable[[np.ndarray], np.ndarray]] = None
    metallicity_distribution: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None

    # Internal cache for the default grid converted to NumPy
    _grid: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Set default metallicity grid if not provided
        if self.metallicity_grid is None:
            self.metallicity_grid = [0.0002, 0.002, 0.02]
        # Convert to numpy array and check monotonicity
        self._grid = np.asarray(self.metallicity_grid, dtype=float)
        if not np.all(np.diff(self._grid) > 0):
            raise ValueError("metallicity_grid must be strictly increasing.")
        # Assign default SFR history
        if self.sfr_history is None:
            self.sfr_history = _sfr_madau_dickinson
        # Assign default metallicity distribution
        if self.metallicity_distribution is None:
            # Partially apply the grid to avoid repeated conversions
            def _dist(z: np.ndarray, grid: np.ndarray = self._grid) -> np.ndarray:
                return _lognormal_metallicity_distribution(z, grid)

            self.metallicity_distribution = _dist

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        """Return the MSSFR at each redshift and metallicity bin.

        The MSSFR is defined here as the product of the cosmic SFR
        density and the metallicity probability density.  Specifically,

        .. math::

            \mathrm{MSSFR}(z, Z_i) = \mathrm{SFR}(z) \times p(Z_i | z),

        where ``p`` is the metallicity distribution.

        Parameters
        ----------
        z:
            Array of redshifts.

        Returns
        -------
        ndarray
            A 2D array of shape ``(len(z), len(metallicity_grid))``.
        """
        z = np.asarray(z, dtype=float)
        sfr = self.sfr_history(z)  # shape (len(z),)
        pz = self.metallicity_distribution(z, self._grid)  # shape (len(z), len(grid))
        # Multiply broadcast to combine sfr[:, None] with pz
        return sfr[:, None] * pz

    def mssfr_for_bin(self, metallicity: float | np.ndarray, z: np.ndarray) -> np.ndarray:
        r"""Return the MSSFR for one or more metallicity values across redshift.

        This helper supports both scalar and array inputs for ``metallicity``.
        When a single metallicity value is provided, the MSSFR grid is
        evaluated at all supplied redshifts and the nearest metallicity bin
        is selected.  When a one–dimensional array of metallicities is
        passed (of the same length as ``z``), the MSSFR is computed for
        each event individually by selecting the nearest metallicity bin
        for the corresponding metallicity value and redshift.

        Parameters
        ----------
        metallicity:
            Metallicity value(s) (mass fraction of metals).  If an array
            is supplied its shape must match that of ``z``.
        z:
            One–dimensional array of redshifts.

        Returns
        -------
        ndarray
            The MSSFR evaluated at the provided redshifts.  If
            ``metallicity`` is an array, the returned array has the same
            shape; otherwise it has length ``len(z)``.
        """
        z = np.asarray(z, dtype=float)
        # If metallicity is array–like, handle elementwise
        if np.ndim(metallicity) > 0 and not np.isscalar(metallicity):
            metallicity_arr = np.asarray(metallicity, dtype=float)
            if metallicity_arr.shape != z.shape:
                raise ValueError(
                    "When providing an array of metallicities, its shape must match that of the redshift array."
                )
            # Evaluate the MSSFR grid for each redshift (len(z) x len(grid))
            mssfr_grid = self.evaluate(z)
            # Find nearest metallicity bin index for each metallicity value
            # self._grid shape (n_grid,), metallicity_arr shape (n,)
            idxs = np.abs(self._grid[None, :] - metallicity_arr[:, None]).argmin(axis=1)
            # Select MSSFR value for each event
            return mssfr_grid[np.arange(z.size), idxs]
        else:
            # Scalar metallicity: find the nearest metallicity bin
            idx = int(np.argmin(np.abs(self._grid - float(metallicity))))
            mssfr_grid = self.evaluate(z)  # shape (len(z), len(grid))
            return mssfr_grid[:, idx]
