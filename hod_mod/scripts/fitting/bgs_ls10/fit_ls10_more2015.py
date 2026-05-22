"""Fit the More+2015 HOD model to LS10/BGS wp(rp) data.

Reads the projected correlation function measured by ``sum_stat`` from the
DESI Legacy Survey DR10 (LS10) volume-limited galaxy samples.  Each sample
selects galaxies above a stellar mass threshold (``--mstar``) and is fit
with the :class:`~hod_mod.galaxies.hod.MoreHODModel`.

Cosmological parameters are held fixed at Planck 2018 values unless
``--vary-cosmo`` is set, in which case h, Ω_m, Ω_b, n_s, and ln10As are
freed with Gaussian priors from Planck 2018 (±3σ hard bounds).

Inputs
------
- /path/to/sum_stat/data/twopcf/LS10_VLIM_ANY_Mstar{XX}-12.0_z{ZZ}-wp-pimax100-sys-comb.h5
  HDF5 projected correlation function from sum_stat.
  Distances in Mpc; h-conversion is applied automatically.
  Full covariance matrix from TreeCorr variance estimate.
- /path/to/sum_stat/data/lf_smf/LS10_VLIM_ANY_Mstar{XX}-12.0_z{ZZ}-smf-vmax.h5
  Stellar mass function (for galaxy number density prior, optional).

Outputs
-------
- results/bgs_ls10/mstar{XX}/map_result.json     best-fit parameters, χ²/dof
- results/bgs_ls10/mstar{XX}/flatchain.npz        emcee posterior samples
- results/bgs_ls10/mstar{XX}/wp_bestfit.pdf        comparison figure (with --plot)

Usage
-----
Single bin::

    python scripts/fitting/bgs_ls10/fit_ls10_more2015.py --mstar 10.0 --plot

All bins (see run_ls10_batch.sh)::

    bash scripts/fitting/bgs_ls10/run_ls10_batch.sh

References
----------
More et al. 2015, ApJ 806, 2 (arXiv:1407.1856)
DESI BGS: Hahn et al. 2023 (arXiv:2208.08512)
"""

import argparse
import json
import os
from dataclasses import dataclass, field

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies.hod import MoreHODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.fitting.hod_wp import WpFitConfig, WpFitter
from hod_mod.fitting.planck_prior import PLANCK18_MEANS, PLANCK18_SIGMAS, PLANCK18_3SIGMA
from hod_mod.data_io.sum_stat_reader import SumStatReader

# ---------------------------------------------------------------------------
# Lookup table: LS10 bins with sys-comb wp files
# format: mstar_lo → (mstar_hi, z_min, z_max, z_eff, log10mmin_init)
# ---------------------------------------------------------------------------

SUM_STAT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                 "..", "sum_stat", "data")
)

LS10_BINS = {
    9.0:  {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.08, "z_eff": 0.065, "log10mmin_init": 11.5},
    9.5:  {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.12, "z_eff": 0.085, "log10mmin_init": 11.8},
    10.0: {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.18, "z_eff": 0.115, "log10mmin_init": 12.0},
    11.0: {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.35, "z_eff": 0.200, "log10mmin_init": 12.8},
    11.25:{"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.35, "z_eff": 0.200, "log10mmin_init": 13.0},
    11.5: {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.35, "z_eff": 0.200, "log10mmin_init": 13.2},
}


def wp_file(mstar_lo: float, info: dict, sum_stat_dir: str, variant: str = "sys-comb") -> str:
    mlo = f"{mstar_lo:.1f}" if mstar_lo == int(mstar_lo) else f"{mstar_lo}"
    mhi = f"{info['mstar_hi']:.1f}"
    zlo = f"{info['z_min']:.2f}"
    zhi = f"{info['z_max']:.2f}"
    suffix = f"-{variant}" if variant else ""
    fname = f"LS10_VLIM_ANY_Mstar{mlo}-{mhi}_z{zlo}-{zhi}-wp-pimax100{suffix}.h5"
    return os.path.join(sum_stat_dir, "twopcf", fname)


def build_config(mstar_lo: float, info: dict, sum_stat_dir: str,
                 vary_cosmo: bool, output_root: str, method: str,
                 n_walkers: int, n_steps: int, n_burnin: int) -> WpFitConfig:
    """Construct a WpFitConfig for a single LS10 stellar mass bin."""
    z_eff   = info["z_eff"]
    log10m0 = info["log10mmin_init"]
    path    = wp_file(mstar_lo, info, sum_stat_dir)

    mstar_str  = f"{mstar_lo:.2f}".rstrip("0").rstrip(".")
    output_dir = os.path.join(output_root, f"mstar{mstar_str}")

    # HOD free parameters — uniform priors unless Gaussian specified below
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

    if vary_cosmo:
        cosmo_params = ["h", "Omega_m", "n_s", "ln10^{10}A_s"]
        for cp in cosmo_params:
            free_params.append(cp)
            param_init[cp]         = PLANCK18_MEANS[cp]
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
        hmf_backend        = "tinker08",
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
                        choices=list(LS10_BINS.keys()),
                        help="Lower stellar mass threshold of the sample.")
    parser.add_argument("--sum-stat-dir", default=SUM_STAT_DIR,
                        help="Path to sum_stat/data/ directory.")
    parser.add_argument("--vary-cosmo", action="store_true",
                        help="Free h, Omega_m, n_s, ln10As with Planck 3σ Gaussian prior.")
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--mcmc-only", action="store_true")
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--n-steps",   type=int, default=2000)
    parser.add_argument("--n-burnin",  type=int, default=500)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "results", "bgs_ls10"))
    args = parser.parse_args()

    info = LS10_BINS[args.mstar]
    method = "both"
    if args.map_only:  method = "map"
    if args.mcmc_only: method = "emcee"

    cfg    = build_config(args.mstar, info, args.sum_stat_dir, args.vary_cosmo,
                          args.output_dir, method, args.n_walkers,
                          args.n_steps, args.n_burnin)
    fitter = WpFitter(cfg)

    print(f"\nLS10 More+2015 HOD fit")
    print(f"  Stellar mass threshold: log10(M*/M_sun) > {args.mstar}")
    print(f"  Redshift range:  z = {info['z_min']:.2f}–{info['z_max']:.2f}  (z_eff = {info['z_eff']:.3f})")
    print(f"  Data file:  {cfg.data_file}")
    print(f"  N_rp bins:  {len(fitter.rp_arr)}")
    print(f"  Free params: {cfg.free_params}")

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
                "params":  result["params"],
                "chi2":    result["chi2"],
                "ndof":    result["ndof"],
                "success": result["success"],
                "mstar_lo": args.mstar,
                "z_eff":   info["z_eff"],
            }, fh, indent=2)
        print(f"MAP result saved → {out_json}")

        if args.plot:
            import matplotlib.pyplot as plt
            wp_pred = fitter.predict_wp(result["params"])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(fitter.rp_arr, fitter.wp_obs, fitter.wp_err,
                        fmt="o", color="k", ms=4,
                        label=rf"LS10  $M_* > 10^{{{args.mstar}}}$")
            ax.loglog(fitter.rp_arr, wp_pred, color="C0", label="More+2015 MAP")
            ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
            ax.set_ylabel(r"$w_p(r_p)$ [Mpc/$h$]")
            ax.legend()
            ax.set_title(f"BGS/LS10  $\\log M_* > {args.mstar}$  "
                         f"$z \\in [{info['z_min']:.2f},{info['z_max']:.2f}]$")
            out_fig = os.path.join(cfg.output_dir, "wp_bestfit.pdf")
            plt.tight_layout()
            plt.savefig(out_fig)
            print(f"Figure saved → {out_fig}")

    if not args.map_only and method in ("emcee", "both"):
        sampler = fitter.sample(progress=True)
        flat    = sampler.get_chain(flat=True)
        acc     = np.mean(sampler.acceptance_fraction)
        print(f"\nMCMC acceptance fraction: {acc:.3f}")
        print(f"Chain shape: {flat.shape}")


if __name__ == "__main__":
    main()
