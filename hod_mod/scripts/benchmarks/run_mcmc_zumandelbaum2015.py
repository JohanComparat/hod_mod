#!/usr/bin/env python
"""Run MCMC for the Zu & Mandelbaum 2015 joint benchmark starting from the saved MAP params.

This bypasses the MAP optimization step (which can stall with Powell + JAX) and
initializes walkers directly from the benchmark_result.json MAP values.

Usage
-----
    python hod_mod/scripts/benchmarks/run_mcmc_zumandelbaum2015.py
    python hod_mod/scripts/benchmarks/run_mcmc_zumandelbaum2015.py --force  # rerun even if chain exists
"""

import argparse
import json
import os
import sys

import numpy as np
from hod_mod.paths import results_root

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

CONFIG_FILE  = os.path.join(_REPO_ROOT, "configs/benchmarks/benchmark_zumandelbaum2015.yml")
RESULT_FILE  = os.path.join(results_root(), "benchmarks/zumandelbaum2015_sdss/benchmark_result.json")
OUT_DIR      = os.path.join(results_root(), "benchmarks/zumandelbaum2015_sdss")
CHAIN_FILE   = os.path.join(OUT_DIR, "flatchain.npz")

INIT_SCATTER = 1e-3   # fractional scatter around MAP for walker initialisation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun even if chain exists")
    args = parser.parse_args()

    if os.path.exists(CHAIN_FILE) and not args.force:
        print(f"Chain already exists: {CHAIN_FILE}")
        print("Pass --force to rerun.")
        return

    print("Loading saved MAP params …")
    with open(RESULT_FILE) as fh:
        saved = json.load(fh)
    map_params = saved["params"]
    print(f"  chi2/ndof = {saved['chi2_ndof']:.4g}  status = {saved['status']}")

    print("\nBuilding JointFitter (JAX compilation ~60 s) …")
    from hod_mod.fitting import load_config, JointFitter
    config = load_config(CONFIG_FILE)
    fitter = JointFitter(config)
    print(f"  wp bins: {len(fitter.rp_arr)}  ΔΣ bins: {len(fitter.R_arr)}")

    # Build theta0 from saved MAP params in free-param order
    theta0 = np.array([map_params[p] for p in config.free_params])
    print(f"\n  Free params ({len(theta0)}): {config.free_params}")
    print(f"  theta0 = {theta0}")

    # Verify MAP log-prob is finite
    lp = fitter._log_prob(theta0)
    print(f"  log_prob(MAP) = {lp:.4f}")
    if not np.isfinite(lp):
        print("ERROR: MAP log_prob is not finite — check the saved params.")
        sys.exit(1)

    # Initialise walkers in a tiny ball around MAP
    rng = np.random.default_rng(42)
    scale = np.maximum(np.abs(theta0) * INIT_SCATTER, 1e-4)
    initial_pos = theta0[None, :] + rng.normal(0, scale, (config.n_walkers, len(theta0)))
    # clip to bounds
    for i, pname in enumerate(config.free_params):
        lo, hi = config.param_bounds[pname]
        initial_pos[:, i] = np.clip(initial_pos[:, i], lo, hi)

    print(f"\nRunning MCMC: {config.n_walkers} walkers × {config.n_burnin} burn-in + {config.n_steps} steps …")
    os.makedirs(OUT_DIR, exist_ok=True)
    fitter.sample(initial_pos=initial_pos, progress=True)
    print(f"\nDone. Chain saved → {CHAIN_FILE}")


if __name__ == "__main__":
    main()
