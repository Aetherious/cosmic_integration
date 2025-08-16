# constants.py
"""
Default values and project-wide constants for CosmicIntegration.
"""

# File names
DEFAULT_COMPAS_FILENAME = "COMPAS_Output.h5"
DEFAULT_OUTPUT_FILENAME = "COMPAS_Output.h5"

# Cosmology
DEFAULT_COSMOLOGY = "Planck18"

# Redshift parameters
DEFAULT_MAX_REDSHIFT = 10.0
DEFAULT_MAX_REDSHIFT_DETECTION = 1.0
DEFAULT_REDSHIFT_STEP = 0.001
DEFAULT_Z_FIRST_SF = 10.0

# Detector sensitivity
DEFAULT_SENSITIVITY = "O3"
DEFAULT_SNR_THRESHOLD = 8.0

# Mass parameters
DEFAULT_M1_MIN = 5.0         # Msun
DEFAULT_M1_MAX = 150.0       # Msun
DEFAULT_M2_MIN = 0.1         # Msun
DEFAULT_FBIN = 0.7

# Metallicity distribution (from Neijssel+19 or your defaults)
DEFAULT_MU0 = 0.035
DEFAULT_MUZ = -0.23
DEFAULT_SIGMA0 = 0.39
DEFAULT_SIGMAZ = 0.0
DEFAULT_ALPHA = 0.0

# Star formation rate parameters (from Madau & Dickinson 2014 / Neijssel+19)
DEFAULT_ASF = 0.01
DEFAULT_BSF = 2.77
DEFAULT_CSF = 2.90
DEFAULT_DSF = 4.70

# Metallicity grid
DEFAULT_MIN_LOGZ = -12.0
DEFAULT_MAX_LOGZ = 0.0
DEFAULT_STEP_LOGZ = 0.01

# Detection probability/SNR grid
DEFAULT_MC_MAX = 300.0
DEFAULT_MC_STEP = 0.1
DEFAULT_ETA_MAX = 0.25
DEFAULT_ETA_STEP = 0.01
DEFAULT_SNR_MAX = 1000.0
DEFAULT_SNR_STEP = 0.1

# Output/plotting
DEFAULT_APPEND_RATES = True
DEFAULT_BINNED_RATES = False
DEFAULT_REDSHIFT_BIN_SIZE = 0.05

# DCO type
DEFAULT_DCO_TYPE = "BBH"
