#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal profiler for COMPAS Cosmic Integration scripts.

Purpose:
- Quickly identify slow spots in CosmicIntegration for rewrite.
- Prints results to terminal only.

Usage:
    python profile_pipeline.py
    pip install line-profiler  # for function-level profiling
"""

import cProfile
import pstats

from ClassCOMPAS import setCOMPASData
import CosmicIntegration
import selection_effects

# -------- SETTINGS --------
TEST_FILE = "/mnt/c/Users/Dasein/Downloads/Zuniform.h5"  # <-- CHANGE THIS
SENSITIVITY = "design"
SNR_THRESHOLD = 8.0


def run_pipeline():
    """Pipeline subset focused on CosmicIntegration usage."""
    # Load a small COMPAS dataset
    data = setCOMPASData(TEST_FILE)

    # Optional: slice down for speed
    sample_size = min(5000, len(data.mass1))
    m1 = data.mass1[:sample_size]
    m2 = data.mass2[:sample_size]
    z = data.redshift[:sample_size]
    dist = data.luminosity_distance[:sample_size]

    # Preload selection effects
    selection_effects._interpolator = selection_effects.SNRinterpolator(SENSITIVITY)
    selection_effects._sens = SENSITIVITY
    selection_effects._random_thetas = None

    # Compute detection probabilities
    for i in range(sample_size):
        selection_effects.detection_probability(
            m1[i], m2[i], z[i], dist[i],
            snr_threshold=SNR_THRESHOLD,
            sensitivity=SENSITIVITY
        )

    # Run the cosmic integration module
    CosmicIntegration.runCosmicIntegration(
        data,
        sensitivity=SENSITIVITY,
        snr_threshold=SNR_THRESHOLD
    )


if __name__ == "__main__":
    print(">>> Running CPU profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    run_pipeline()
    profiler.disable()

    # Show top 20 slowest functions
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)

    # Optional: Uncomment to profile specific functions in detail
    """
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp.add_function(CosmicIntegration.runCosmicIntegration)
    lp_wrapper = lp(run_pipeline)
    lp_wrapper()
    lp.print_stats()
    """