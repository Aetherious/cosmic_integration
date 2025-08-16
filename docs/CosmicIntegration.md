# cosmic_integration Module

## find_detection_rate()

### Parameters

#### Finding and Masking COMPAS File
path : str
dco_type : str
merger_output_filename : str
weight_column : str

merges_in_hubble_time (bool)
no_ROLF_after_CEE (bool)
pessimistic_CEE (bool)

#### Creating Redshift Array
max_redshift (float)
max_redshift_detection (float)
redshift_step (float)

#### Determining Star Forming Mass per Sampled Binary
use_sampled_mass_ranges (bool)
m1_min (float)
m1_max (float)
m2_min (float)
fbin (float)

#### Creating Metallicity Distribution and Probabilities
mu0 (float)
muz (float)
sigma0 (float)
sigmaz (float)
alpha (float)
min_logZ (float)
max_logZ (float)
step_logZ (float)

#### Determining Detection Probabilities
sensitivity (string)
snr_threshold (float)
Mc_max (float)
Mc_step (float)
eta_max (float)
eta_step (float)
snr_max (float)
snr_step (float)

### Returns

