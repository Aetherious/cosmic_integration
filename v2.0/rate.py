
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np, astropy.units as u
from astropy.cosmology import z_at_value
from .cosmology import get_cosmology, cosmology_arrays, z_grid
from .compas_data import CompasData
from .detection import SNRGridCalibrator
from .mssfr import SFRDParams, LognormalMetallicityParams, mssfr

@dataclass
class DetectionRateConfig:
    z_max: float = 10.0; dz: float = 0.05; dz_detect: float = 0.005
    snr_grid_path: Optional[str] = None; detector_key: str = "Aplus.txt"
    rho_th: float = 8.0; logistic_sigma: float = 0.5; grid_distance_unit_gpc: float = 0.001
    sfr_a: float = 0.01; sfr_b: float = 2.77; sfr_c: float = 2.9; sfr_d: float = 4.7
    mu0: float = 0.035; muz: float = -0.23; sigma0: float = 0.39; sigmaz: float = 0.0; alpha: float = 0.0
    Z_sun: float = 0.0142; use_log10_mu: bool = True
    Z_min: float = 1e-4; Z_max: float = 0.03; n_Z: int = 50
    compas_time_unit: str = "Myr"

def _ensure_time_unit(delay_times: np.ndarray, unit: str) -> np.ndarray:
    arr = np.asarray(delay_times, dtype=float); u = unit.lower()
    return arr if u.startswith("g") else arr/1e3 if u.startswith("m") else arr/1e6 if u.startswith("k") else arr/1e9 if u.startswith("y") else arr

def _birth_redshift(cosmo, z_merge: np.ndarray, delay_gyr: float) -> np.ndarray:
    tlb_merge = cosmo.lookback_time(z_merge).to(u.Gyr).value
    tlb_birth = np.clip(tlb_merge + delay_gyr, 0.0, cosmo.age(0).to(u.Gyr).value - 1e-6)
    return np.array([float(z_at_value(cosmo.lookback_time, t*u.Gyr, zmin=0.0, zmax=20.0)) for t in tlb_birth])

def find_detection_rate(path: str, cosmology: Optional[Any] = None, config: Optional[DetectionRateConfig] = None,
                        n_workers: int = 1, output_filename: Optional[str] = "results.npz"):
    if config is None: config = DetectionRateConfig()
    cosmo = get_cosmology(cosmology)
    z = z_grid(config.z_max, config.dz); z_det = z_grid(min(config.z_max,2.0), config.dz_detect)
    d_l_gpc, dVdz_gpc3, _ = cosmology_arrays(cosmo, z)
    d_l_gpc_det, dVdz_gpc3_det, _ = cosmology_arrays(cosmo, z_det)

    compas = CompasData(path); compas.set_dco_mask("BBH"); compas.load()
    delay_gyr = _ensure_time_unit(compas.delay_times_masked, config.compas_time_unit)
    Z_sys = compas.initial_metallicities_masked
    weights = getattr(compas, "sw_weights", None); 
    if weights is None: weights = np.ones_like(Z_sys)

    sfr_par = SFRDParams(config.sfr_a, config.sfr_b, config.sfr_c, config.sfr_d)
    mdf_par = LognormalMetallicityParams(config.mu0, config.muz, config.sigma0, config.sigmaz,
                                         config.alpha, config.Z_sun, config.use_log10_mu)
    Z_grid = np.geomspace(config.Z_min, config.Z_max, config.n_Z)

    calib = SNRGridCalibrator(grid_path=config.snr_grid_path, detector_key=config.detector_key,
                              rho_th=config.rho_th, logistic_sigma=config.logistic_sigma,
                              grid_distance_unit_gpc=config.grid_distance_unit_gpc)

    n_bin, n_z = Z_sys.size, z.size
    merger_rate = np.zeros((n_bin, n_z), dtype=float)
    for i in range(n_bin):
        zb = _birth_redshift(cosmo, z, delay_gyr[i])
        m = mssfr(Z_grid, zb, sfr_par, mdf_par)  # [n_z, n_Z]
        norm = np.trapz(m, Z_grid, axis=1); m = (m.T/np.where(norm>0, norm, 1.0)).T
        m_at_Zi = np.interp(Z_sys[i], Z_grid, m.T, left=0.0, right=0.0)
        if getattr(compas, "mass_evolved_per_binary", None) is None: compas.find_star_forming_mass_per_binary_sampling()
        R_form_i = m_at_Zi / float(compas.mass_evolved_per_binary)
        merger_rate[i,:] = R_form_i * float(weights[i])

    formation_rate = np.trapz(mssfr(Z_grid, z, sfr_par, mdf_par), Z_grid, axis=1)

    n_zd = z_det.size; m1 = compas.mass1_masked; m2 = compas.mass2_masked
    m1_grid = np.broadcast_to(m1[:,None], (n_bin, n_zd))
    m2_grid = np.broadcast_to(m2[:,None], (n_bin, n_zd))
    z_grid_det = np.broadcast_to(z_det[None,:], (n_bin, n_zd))
    d_l_grid = np.broadcast_to(d_l_gpc_det[None,:], (n_bin, n_zd))
    p_det = calib.p_det(m1_grid, m2_grid, z_grid_det, d_l_grid)

    merger_rate_det = np.empty((n_bin, n_zd), dtype=float)
    for i in range(n_bin):
        merger_rate_det[i,:] = np.interp(z_det, z, merger_rate[i,:], left=0.0, right=0.0)

    detection_rate = (merger_rate_det / (1.0 + z_det)[None,:]) * dVdz_gpc3_det[None,:] * p_det

    if output_filename:
        np.savez_compressed(
            output_filename,
            z=z, z_det=z_det, Z_grid=Z_grid,
            formation_rate=formation_rate, merger_rate=merger_rate, detection_rate=detection_rate,
            d_l_gpc=d_l_gpc, d_l_gpc_det=d_l_gpc_det, dVdz_gpc3=dVdz_gpc3, dVdz_gpc3_det=dVdz_gpc3_det,
            masses=np.vstack([m1, m2]).T, weights=weights,
            meta=np.array([("cosmology", getattr(cosmo, "name", "custom"))], dtype=object),
            sfr_params=np.array([config.sfr_a, config.sfr_b, config.sfr_c, config.sfr_d]),
            mdf_params=np.array([config.mu0, config.muz, config.sigma0, config.sigmaz, config.alpha, config.Z_sun, int(config.use_log10_mu)]),
        )
    return detection_rate, formation_rate, merger_rate, z, compas
