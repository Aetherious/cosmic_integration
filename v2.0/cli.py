
from __future__ import annotations
import argparse, numpy as np
from .rate import find_detection_rate, DetectionRateConfig
from .cosmology import get_cosmology

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cosmic_integration_dasein")
    p.add_argument("--path", required=True, help="Path to COMPAS HDF5 file.")
    p.add_argument("--detector", default="Aplus.txt")
    p.add_argument("--rho-threshold", type=float, default=8.0)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--grid-distance-unit", type=float, default=0.001)
    p.add_argument("--z-max", type=float, default=10.0)
    p.add_argument("--dz", type=float, default=0.05)
    p.add_argument("--dz-detect", type=float, default=0.005)
    p.add_argument("--sfr-a", type=float, default=0.01)
    p.add_argument("--sfr-b", type=float, default=2.77)
    p.add_argument("--sfr-c", type=float, default=2.9)
    p.add_argument("--sfr-d", type=float, default=4.7)
    p.add_argument("--mu0", type=float, default=0.035)
    p.add_argument("--muz", type=float, default=-0.23)
    p.add_argument("--sigma0", type=float, default=0.39)
    p.add_argument("--sigmaz", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--Zsun", type=float, default=0.0142)
    p.add_argument("--mu-log10", action="store_true")
    p.add_argument("--mu-linear", dest="mu_log10", action="store_false")
    p.set_defaults(mu_log10=True)
    p.add_argument("--Z-min", type=float, default=1e-4)
    p.add_argument("--Z-max", type=float, default=0.03)
    p.add_argument("--nZ", type=int, default=50)
    p.add_argument("--compas-time-unit", default="Myr")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--output", default="results.npz")
    p.add_argument("--cosmology", default=None)
    return p

def main() -> None:
    args = build_parser().parse_args()
    cosmo = get_cosmology(args.cosmology) if args.cosmology else None
    cfg = DetectionRateConfig(
        z_max=args.z_max, dz=args.dz, dz_detect=args.dz_detect,
        snr_grid_path=None, detector_key=args.detector,
        rho_th=args.rho_threshold, logistic_sigma=args.sigma,
        grid_distance_unit_gpc=args.grid_distance_unit,
        sfr_a=args.sfr_a, sfr_b=args.sfr_b, sfr_c=args.sfr_c, sfr_d=args.sfr_d,
        mu0=args.mu0, muz=args.muz, sigma0=args.sigma0, sigmaz=args.sigmaz, alpha=args.alpha,
        Z_sun=args.Zsun, use_log10_mu=args.mu_log10,
        Z_min=args.Z_min, Z_max=args.Z_max, n_Z=args.nZ,
        compas_time_unit=args.compas_time_unit,
    )
    detection_rate, formation_rate, merger_rate, z, _ = find_detection_rate(
        path=args.path, cosmology=cosmo, config=cfg, n_workers=args.n_workers, output_filename=args.output
    )
    total_det = np.sum(detection_rate, axis=0)
    print(f"[ok] wrote {args.output}")
    print(f"z-grid (det): {total_det.size} bins, z_max={args.z_max}")
    print(f"Total detection rate at z~0: {total_det[0]:.3e} yr^-1 dz^-1")

if __name__ == "__main__":
    main()
