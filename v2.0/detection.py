
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os, h5py, numpy as np
from importlib import resources

def _find_default_grid() -> str:
    env = os.getenv("COSMIC_INTEGRATION_SNR_GRID")
    if env and os.path.isfile(env):
        return env
    try:
        data_pkg = "cosmic_integration_dasein.data"
        try:
            for p in resources.files(data_pkg).iterdir():
                if str(p).lower().endswith((".h5", ".hdf5")):
                    return str(p)
        except AttributeError:
            with resources.path(data_pkg, "__init__.py") as base:
                base_dir = os.path.dirname(str(base))
            for fn in os.listdir(base_dir):
                if fn.lower().endswith((".h5", ".hdf5")):
                    return os.path.join(base_dir, fn)
    except Exception:
        pass
    raise FileNotFoundError("No bundled SNR grid found in cosmic_integration_dasein/data/. "
                            "Place a *.h5 there or set COSMIC_INTEGRATION_SNR_GRID.")

def _bilinear_interp(xg: np.ndarray, yg: np.ndarray, Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float)); y = np.atleast_1d(np.asarray(y, dtype=float))
    x = np.clip(x, xg[0], xg[-1]); y = np.clip(y, yg[0], yg[-1])
    ix = np.searchsorted(xg, x, side="right") - 1; iy = np.searchsorted(yg, y, side="right") - 1
    ix = np.clip(ix, 0, len(xg)-2); iy = np.clip(iy, 0, len(yg)-2)
    x0 = xg[ix]; x1 = xg[ix+1]; y0 = yg[iy]; y1 = yg[iy+1]
    tx = (x-x0)/(x1-x0+1e-15); ty = (y-y0)/(y1-y0+1e-15)
    Z00 = Z[ix,iy]; Z10=Z[ix+1,iy]; Z01=Z[ix,iy+1]; Z11=Z[ix+1,iy+1]
    return (1-tx)*(1-ty)*Z00 + tx*(1-ty)*Z10 + (1-tx)*ty*Z01 + tx*ty*Z11

@dataclass
class SNRGridCalibrator:
    grid_path: Optional[str] = None
    detector_key: str = "Aplus.txt"
    rho_th: float = 8.0
    logistic_sigma: float = 0.5
    grid_distance_unit_gpc: float = 0.001  # distance attr in Mpc -> Gpc

    def __post_init__(self) -> None:
        if not self.grid_path:
            self.grid_path = _find_default_grid()
        self._load_grid()

    def _load_grid(self) -> None:
        with h5py.File(self.grid_path, "r") as f:
            masses = np.array(f["mass_axis"][:], dtype=float)
            if "snr_values" not in f: raise KeyError("Missing group 'snr_values' in SNR grid HDF5.")
            group = f["snr_values"]
            if self.detector_key not in group:
                raise KeyError(f"detector_key '{self.detector_key}' not found; available: {list(group.keys())}")
            snr_grid = np.array(group[self.detector_key][:], dtype=float)
            d_ref = float(f.attrs.get("distance", 1.0)) * float(self.grid_distance_unit_gpc)
        if snr_grid.shape != (masses.size, masses.size):
            raise ValueError("SNR grid shape must be (N_mass, N_mass) matching 'mass_axis'.")
        self.mass_axis = masses; self.snr_grid_ref = snr_grid; self.distance_ref_gpc = d_ref

    def optimal_snr_at_distance(self, m1_det, m2_det, d_l_gpc):
        m1_det = np.asarray(m1_det, dtype=float); m2_det = np.asarray(m2_det, dtype=float)
        d_l_gpc = np.asarray(d_l_gpc, dtype=float)
        m1 = np.maximum(m1_det, m2_det); m2 = np.minimum(m1_det, m2_det)
        snr_ref = _bilinear_interp(self.mass_axis, self.mass_axis, self.snr_grid_ref, m1, m2)
        scale = np.where(d_l_gpc>0, self.distance_ref_gpc/d_l_gpc, 0.0)
        return snr_ref * scale

    def p_det(self, m1_src, m2_src, z, d_l_gpc):
        z = np.asarray(z, dtype=float)
        m1_det = (1.0 + z) * np.asarray(m1_src, dtype=float)
        m2_det = (1.0 + z) * np.asarray(m2_src, dtype=float)
        rho_opt = self.optimal_snr_at_distance(m1_det, m2_det, d_l_gpc)
        rho_eff = rho_opt / 2.26
        x = (rho_eff - self.rho_th) / max(self.logistic_sigma, 1e-6)
        return 1.0 / (1.0 + np.exp(-x))
