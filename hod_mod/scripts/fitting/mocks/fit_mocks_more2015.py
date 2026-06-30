"""Fit the More+2015 HOD model to Uchuu mock wp(rp) data with Planck 3σ prior.

Reads the projected correlation function measured on Uchuu N-body simulation
mock galaxy catalogues.  The mocks share the same selection as the LS10/BGS
data but with known input cosmology (Planck 2018), enabling model validation.

Cosmological parameters are varied with **Gaussian priors** from Planck 2018
(:mod:`~hod_mod.fitting.planck_prior`):

.. math::

    \\ln\\pi(\\theta) = -\\frac{1}{2} \\sum_i
    \\left(\\frac{\\theta_i - \\mu_i^{\\rm P18}}{\\sigma_i^{\\rm P18}}\\right)^2

with hard bounds at ±3σ.  This tight prior tests whether the HOD fitting can
recover the correct galaxy–halo connection given a precise cosmology.

Inputs
------
- /path/to/sum_stat/data/mocks/twopcf/MOCK_VLIM_ANY_Mstar{XX}_z{ZZ}-wp-pimax100.h5
  HDF5 projected correlation function from mock measurements.

Outputs
-------
- results/mocks/mstar{XX}/map_result.json
- results/mocks/mstar{XX}/flatchain.npz
- results/mocks/mstar{XX}/wp_bestfit.pdf   (with --plot)

Usage
-----
Single bin::

    python scripts/fitting/mocks/fit_mocks_more2015.py --mstar 10.24 --plot

All bins::

    bash scripts/fitting/mocks/run_mocks_batch.sh

References
----------
More et al. 2015, ApJ 806, 2 (arXiv:1407.1856)
Ishiyama et al. 2021 (Uchuu simulations, arXiv:2007.14720)
Planck Collaboration 2020, A&A 641, A6 (arXiv:1807.06209)
"""

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from hod_mod.fitting import WpFitConfig, WpFitter
from hod_mod.fitting.planck_prior import PLANCK18_MEANS, PLANCK18_SIGMAS, PLANCK18_3SIGMA
from hod_mod.paths import results_root

SUM_STAT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                 "..", "sum_stat", "data")
)

# Uchuu mock stellar mass bins
MOCK_BINS = {
    9.29:  {"z_min": 0.05, "z_max": 0.08, "z_eff": 0.065, "log10mmin_init": 11.5},
    9.78:  {"z_min": 0.05, "z_max": 0.12, "z_eff": 0.085, "log10mmin_init": 11.8},
    10.24: {"z_min": 0.05, "z_max": 0.18, "z_eff": 0.115, "log10mmin_init": 12.0},
    10.45: {"z_min": 0.05, "z_max": 0.22, "z_eff": 0.135, "log10mmin_init": 12.2},
    10.65: {"z_min": 0.05, "z_max": 0.26, "z_eff": 0.155, "log10mmin_init": 12.4},
    10.84: {"z_min": 0.05, "z_max": 0.31, "z_eff": 0.180, "log10mmin_init": 12.6},
    11.03: {"z_min": 0.05, "z_max": 0.35, "z_eff": 0.200, "log10mmin_init": 12.8},
    11.22: {"z_min": 0.05, "z_max": 0.35, "z_eff": 0.200, "log10mmin_init": 13.0},
    11.39: {"z_min": 0.05, "z_max": 0.35, "z_eff": 0.200, "log10mmin_init": 13.2},
}


def mock_wp_file(mstar_lo: float, info: dict, sum_stat_dir: str) -> str:
    zlo  = f"{info['z_min']:.2f}"
    zhi  = f"{info['z_max']:.2f}"
    fname = f"MOCK_VLIM_ANY_Mstar{mstar_lo:.2f}_z{zlo}-{zhi}-wp-pimax100.h5"
    return os.path.join(sum_stat_dir, "mocks", "twopcf", fname)


def build_config(mstar_lo: float, info: dict, sum_stat_dir: str, output_root: str,
                 method: str, n_walkers: int, n_steps: int, n_burnin: int,
                 wide_cosmo: bool = False) -> WpFitConfig:
    """Build a WpFitConfig with Planck 3σ Gaussian cosmological prior."""
    z_eff   = info["z_eff"]
    log10m0 = info["log10mmin_init"]
    path    = mock_wp_file(mstar_lo, info, sum_stat_dir)

    mstar_str  = f"{mstar_lo:.2f}"
    output_dir = os.path.join(output_root, f"mstar{mstar_str}")

    # HOD parameters — always varied with uniform priors
    free_params = ["log10mmin", "sigma_logm", "log10m1", "alpha", "kappa"]
    param_init  = {
        "log10mmin":  log10m0,
        "sigma_logm": 0.38,
        "log10m1":    log10m0 + 1.1,
        "alpha":      1.0,
        "kappa":      1.0,
        "alpha_inc":  1.0,
        "log10m_inc": log10m0 - 0.5,
    }
    param_bounds = {
        "log10mmin":  (log10m0 - 1.5, log10m0 + 1.5),
        "sigma_logm": (0.05, 1.5),
        "log10m1":    (log10m0 - 0.5, log10m0 + 2.5),
        "alpha":      (0.5, 3.0),
        "kappa":      (0.1, 5.0),
    }
    param_prior_types  = {p: "uniform" for p in free_params}
    param_prior_means  = {}
    param_prior_sigmas = {}

    # Cosmological parameters — Gaussian Planck 2018 prior (default) or wide (for testing)
    cosmo_params = ["h", "Omega_m", "n_s", "ln10^{10}A_s"]
    for cp in cosmo_params:
        free_params.append(cp)
        param_init[cp] = PLANCK18_MEANS[cp]
        if wide_cosmo:
            lo, hi = PLANCK18_3SIGMA[cp]
            width  = hi - lo
            param_bounds[cp]      = (lo - width, hi + width)  # ±6σ
            param_prior_types[cp] = "uniform"
        else:
            param_bounds[cp]       = PLANCK18_3SIGMA[cp]
            param_prior_types[cp]  = "gaussian"
            param_prior_means[cp]  = PLANCK18_MEANS[cp]
            param_prior_sigmas[cp] = PLANCK18_SIGMAS[cp]

    return WpFitConfig(
        data_file          = path,
        data_format        = "hdf5",
        rp_min             = 0.3,
        rp_max             = 50.0,
        hod_model          = "MoreHODModel",
        hmf_backend        = "csst",
        z                  = z_eff,
        pi_max             = 100.0,
        free_params        = free_params,
        param_bounds       = param_bounds,
        param_init         = param_init,
        method             = method,
        n_walkers          = n_walkers,
        n_steps            = n_steps,
        n_burnin           = n_burnin,
        output_dir         = output_dir,
        repo_root          = "",
        param_prior_types  = param_prior_types,
        param_prior_means  = param_prior_means,
        param_prior_sigmas = param_prior_sigmas,
    )


def main():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mstar", type=float, required=True,
                        choices=list(MOCK_BINS.keys()),
                        help="Stellar mass threshold of the mock sample.")
    parser.add_argument("--sum-stat-dir", default=SUM_STAT_DIR)
    parser.add_argument("--wide-cosmo", action="store_true",
                        help="Use wide uniform cosmological priors instead of Planck 3σ.")
    parser.add_argument("--map-only",  action="store_true")
    parser.add_argument("--mcmc-only", action="store_true")
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--n-steps",   type=int, default=2000)
    parser.add_argument("--n-burnin",  type=int, default=500)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir",
                        default=os.path.join(results_root(), "mocks"))
    args = parser.parse_args()

    info   = MOCK_BINS[args.mstar]
    method = "both"
    if args.map_only:  method = "map"
    if args.mcmc_only: method = "emcee"

    cfg    = build_config(args.mstar, info, args.sum_stat_dir, args.output_dir,
                          method, args.n_walkers, args.n_steps, args.n_burnin,
                          wide_cosmo=args.wide_cosmo)
    fitter = WpFitter(cfg)

    print(f"\nMock More+2015 HOD fit (Planck 3σ prior)")
    print(f"  Stellar mass threshold:  log10(M*/M_sun) > {args.mstar}")
    print(f"  Redshift range:  z = {info['z_min']:.2f}–{info['z_max']:.2f}  "
          f"(z_eff = {info['z_eff']:.3f})")
    print(f"  Data file:  {cfg.data_file}")
    print(f"  N_rp bins:  {len(fitter.rp_arr)}")
    cosmo_prior = "Planck 3σ Gaussian" if not args.wide_cosmo else "wide uniform"
    print(f"  Cosmological prior: {cosmo_prior}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    if not args.mcmc_only:
        result = fitter.map_fit()
        print("\n=== MAP result ===")
        for name, val in zip(cfg.free_params, result["theta"]):
            print(f"  {name:25s} = {val:.4f}")
        print(f"  chi2/dof = {result['chi2']:.2f} / {result['ndof']}")

        out_json = os.path.join(cfg.output_dir, "map_result.json")
        with open(out_json, "w") as fh:
            json.dump({
                "params": result["params"], "chi2": result["chi2"],
                "ndof": result["ndof"], "success": result["success"],
                "mstar_lo": args.mstar, "z_eff": info["z_eff"],
                "cosmo_prior": "planck3sigma" if not args.wide_cosmo else "wide",
            }, fh, indent=2)
        print(f"MAP result saved → {out_json}")

        if args.plot:
            import matplotlib.pyplot as plt
            wp_pred = fitter.predict_wp(result["params"])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(fitter.rp_arr, fitter.wp_obs, fitter.wp_err,
                        fmt="o", color="k", ms=4, label="Uchuu mock")
            ax.loglog(fitter.rp_arr, wp_pred, color="C1", label="More+2015 MAP")
            ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
            ax.set_ylabel(r"$w_p(r_p)$ [Mpc/$h$]")
            ax.legend()
            ax.set_title(f"Mock  $\\log M_* > {args.mstar}$  Planck 3σ prior")
            plt.tight_layout()
            out_fig = os.path.join(cfg.output_dir, "wp_bestfit.pdf")
            plt.savefig(out_fig)
            print(f"Figure saved → {out_fig}")

    if not args.map_only and method in ("emcee", "both"):
        sampler = fitter.sample(progress=True)
        flat    = sampler.get_chain(flat=True)
        print(f"\nMCMC acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(f"Chain shape: {flat.shape}")


if __name__ == "__main__":
    main()
