#!/usr/bin/env python3

"""
description here
"""

from cli import parse_args


from calculations import find_detection_rate
import io_utils import append_rates, delete_rates
from plotting import plot_rates

import astropy.units as u
import importlib
import numpy as np

def main():
    args = parse_args()
    
    #####################################
    # Run the cosmic integration

    # if args.Cosmology == "Planck18":
    #     print("USING PLANCK18 AS COSMOLOGY! if working with Illustris TNG data please use Planck15 instead")
    # else:
    #     print("Using %s as cosmology!"%args.Cosmology)
    cosmology = getattr(importlib.import_module('astropy.cosmology'), args.Cosmology)

    print("Calculate detection rates")
    
    start_CI = time.time()

    detection_rate, formation_rate, merger_rate, redshifts, COMPAS, Average_SF_mass_needed, shell_volumes = \
        find_detection_rate(
            args.path,
            dco_type=args.dco_type,
            weight_column=args.weight_column,
            max_redshift=args.max_redshift,
            max_redshift_detection=args.max_redshift_detection,
            redshift_step=args.redshift_step,
            z_first_SF= args.z_first_SF,
            m1_min=args.m1_min*u.Msun,
            m1_max=args.m1_max*u.Msun,
            m2_min=args.m2_min*u.Msun,
            fbin=args.fbin,
            aSF = args.aSF,
            bSF = args.bSF,
            cSF = args.cSF,
            dSF = args.dSF, 
            mu0=args.mu0,
            muz=args.muz,
            sigma0=args.sigma0,
            sigmaz=args.sigmaz,
            alpha=args.alpha, 
            sensitivity=args.sensitivity,
            snr_threshold=args.snr_threshold, 
            min_logZ=-12.0,
            max_logZ=0.0,
            step_logZ=0.01,
            Mc_max=300.0,
            Mc_step=0.1,
            eta_max=0.25,
            eta_step=0.01,
            snr_max=1000.0,
            snr_step=0.1)
    
    end_CI = time.time()
    
    print("rates calculatd, now appending rates")

    #####################################
    # Plot your result
    start_plot = time.time()
    chirp_masses = (COMPAS.mass1*COMPAS.mass2)**(3./5.) / (COMPAS.mass1 + COMPAS.mass2)**(1./5.)
    print('almost finished, just plotting your results now')
    plot_rates(args.path, formation_rate, merger_rate, detection_rate, redshifts, chirp_masses, show_plot = False, mu0=args.mu0, muz=args.muz, sigma0=args.sigma0, sigmaz=args.sigmaz, alpha=args.alpha, aSF = args.aSF,  bSF = args.bSF , cSF = args.cSF , dSF = args.dSF ,)
    end_plot = time.time()

    print('CI took ', end_CI - start_CI, 's')
    print('Appending rates took ', end_append - start_append, 's')
    print('plot took ', end_plot - start_plot, 's')

if __name__ == "__main__":
    main()
