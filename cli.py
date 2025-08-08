import argparse

from constants import (
    DEFAULT_COMPAS_FILENAME,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_COSMOLOGY,
    DEFAULT_MAX_REDSHIFT,
    DEFAULT_MAX_REDSHIFT_DETECTION,
    DEFAULT_REDSHIFT_STEP,
    DEFAULT_Z_FIRST_SF,
    DEFAULT_SENSITIVITY,
    DEFAULT_SNR_THRESHOLD,
    DEFAULT_M1_MIN,
    DEFAULT_M1_MAX,
    DEFAULT_M2_MIN,
    DEFAULT_FBIN,
    DEFAULT_MU0,
    DEFAULT_MUZ,
    DEFAULT_SIGMA0,
    DEFAULT_SIGMAZ,
    DEFAULT_ALPHA,
    DEFAULT_ASF,
    DEFAULT_BSF,
    DEFAULT_CSF,
    DEFAULT_DSF,
    DEFAULT_MIN_LOGZ,
    DEFAULT_MAX_LOGZ,
    DEFAULT_STEP_LOGZ,
    DEFAULT_MC_MAX,
    DEFAULT_MC_STEP,
    DEFAULT_ETA_MAX,
    DEFAULT_ETA_STEP,
    DEFAULT_SNR_MAX,
    DEFAULT_SNR_STEP,
    DEFAULT_APPEND_RATES,
    DEFAULT_BINNED_RATES,
    DEFAULT_REDSHIFT_BIN_SIZE,
    DEFAULT_DCO_TYPE,
    DEFAULT_WEIGHT_COLUMN
)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    
    # Define command line options for the most commonly varied options
    parser.add_argument("--path", dest= 'path',  help="Path to the COMPAS file that contains the output",type=str, default = './')
    parser.add_argument("--filename", dest= 'fname',  help="Name of the COMPAS file",type=str, default = "COMPAS_Output.h5")
    parser.add_argument("--outfname", dest= 'outfname',  help="Name of the output file where you store the rates, default is append to COMPAS output",type=str, default = "COMPAS_Output.h5")

    # For what DCO would you like the rate?  options: ALL, BHBH, BHNS NSNS
    parser.add_argument("--dco_type", dest= 'dco_type',  help="Which DCO type you used to calculate rates, one of: ['all', 'BBH', 'BHNS', 'BNS'] ",type=str, default = "BBH")
    parser.add_argument("--weight", dest= 'weight_column',  help="Name of column w AIS sampling weights, i.e. 'mixture_weight'(leave as None for unweighted samples) ",type=str, default = None)

    # Options for the redshift evolution and detector sensitivity
    parser.add_argument("--maxz", dest= 'max_redshift',  help="Maximum redshift to use in array",type=float, default=10)
    parser.add_argument("--zSF", dest= 'z_first_SF',  help="redshift of first star formation",type=float, default=10)
    parser.add_argument("--maxzdet", dest= 'max_redshift_detection',  help="Maximum redshift to calculate detection rates",type=float, default=1)
    parser.add_argument("--zstep", dest= 'redshift_step',  help="size of step to take in redshift",type=float, default=0.001)
    parser.add_argument("--sens", dest= 'sensitivity',  help="Which detector sensitivity to use: one of ['design', 'O1', 'O3']",type=str, default = "O3")
    parser.add_argument("--snr", dest= 'snr_threshold',  help="What SNR threshold required for a detection",type=float, default=8)

    # Parameters to calculate the representing SF mass (make sure these match YOUR simulation!)
    parser.add_argument("--m1min", dest= 'm1_min',  help="Minimum primary mass sampled by COMPAS",type=float, default=5.) 
    parser.add_argument("--m1max", dest= 'm1_max',  help="Maximum primary mass sampled by COMPAS",type=float, default=150.) 
    parser.add_argument("--m2min", dest= 'm2_min',  help="Minimum secondary mass sampled by COMPAS",type=float, default=0.1) 
    parser.add_argument("--fbin", dest= 'fbin',  help="Binary fraction used by COMPAS",type=float, default=0.7) 

    # Parameters determining dP/dZ and SFR(z), default options from Neijssel 2019
    parser.add_argument("--mu0", dest= 'mu0',  help="mean metallicity at redshhift 0",type=float, default=0.035)
    parser.add_argument("--muz", dest= 'muz',  help="redshift evolution of mean metallicity, dPdlogZ",type=float, default=-0.23)
    parser.add_argument("--sigma0", dest= 'sigma0',  help="variance in metallicity density distribution, dPdlogZ",type=float, default=0.39)
    parser.add_argument("--sigmaz", dest= 'sigmaz',  help="redshift evolution of variance, dPdlogZ",type=float, default=0.0)
    parser.add_argument("--alpha", dest= 'alpha',  help="skewness of mtallicity density distribution, dPdlogZ",type=float, default=0.0)
    parser.add_argument("--aSF", dest= 'aSF',  help="Parameter for shape of SFR(z)",type=float, default=0.01) 
    parser.add_argument("--bSF", dest= 'bSF',  help="Parameter for shape of SFR(z)",type=float, default=2.77)
    parser.add_argument("--cSF", dest= 'cSF',  help="Parameter for shape of SFR(z)",type=float, default=2.90)
    parser.add_argument("--dSF", dest= 'dSF',  help="Parameter for shape of SFR(z)",type=float, default=4.70)
 
     # Options for saving your data
    parser.add_argument("--dontAppend", dest= 'append_rates',  help="Prevent the script from appending your rates to the hdf5 file.", action='store_false', default=True)
    parser.add_argument("--BinAppend", dest= 'binned_rates',  help="Append your rates in more crude redshift bins to save space.", action='store_true', default=False)
    parser.add_argument("--redshiftBinSize", dest= 'zBinSize',  help="How big should the crude redshift bins be", type=float, default=0.05)
    parser.add_argument("--delete", dest= 'delete_rates',  help="Delete the rate group from your hdf5 output file (groupname based on dP/dZ parameters)", action='store_true', default=False)
    parser.add_argument("--cosmology", dest='Cosmology', help="Cosmology that is used for cosmic integration", type=str, default="Planck18")
    
    return parser.parse_args()
