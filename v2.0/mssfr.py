
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SFRDParams:
    a: float = 0.01; b: float = 2.77; c: float = 2.9; d: float = 4.7
def sfrd_md14(z, p: SFRDParams):
    z = np.asarray(z, dtype=float); return p.a*(1+z)**p.b / (1 + ((1+z)/p.c)**p.d)

@dataclass
class LognormalMetallicityParams:
    mu0: float = 0.035; muz: float = -0.23; sigma0: float = 0.39; sigmaz: float = 0.0
    alpha: float = 0.0; Z_sun: float = 0.0142; use_log10: bool = True

def _ln_moments(z, p: LognormalMetallicityParams):
    z = np.asarray(z, dtype=float)
    mu_Z = p.Z_sun * (10.0**(p.mu0 + p.muz*np.log10(1+z)) if p.use_log10 else (p.mu0 + p.muz*z))
    mu_ln = np.log(np.clip(mu_Z, 1e-12, None)); sigma = np.sqrt(max(p.sigma0,1e-6)**2 + (p.sigmaz*z)**2)
    return mu_ln, sigma

def lognormal_mdf(Z, z, p: LognormalMetallicityParams):
    Z = np.asarray(Z, dtype=float); z = np.asarray(z, dtype=float)
    mu_ln, sigma = _ln_moments(z, p); Z = np.clip(Z, 1e-12, None)
    Z2 = Z[None,:]; mu = mu_ln[:,None]; s = sigma[:,None]
    pdf = (1.0/(Z2*s*np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(Z2)-mu)/s)**2)
    if abs(p.alpha)>0.0: pdf *= (Z2/p.Z_sun)**p.alpha
    norm = np.trapz(pdf, Z, axis=1, initial=0.0); pdf = pdf / norm[:,None]
    return pdf

def mssfr(Z, z, sfr_params: SFRDParams, mdf_params: LognormalMetallicityParams):
    return sfrd_md14(z, sfr_params)[:,None] * lognormal_mdf(Z, z, mdf_params)
