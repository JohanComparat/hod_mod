"""Fit an HOD model to wp(rp) from a YAML config file.

Usage
-----
    python scripts/fit_hod_wp.py <config.yml> [--plot] [--map-only] [--mcmc-only]

Examples
--------
    python scripts/fit_hod_wp.py configs/hod_fit_more2015_cmass.yml --plot
    python scripts/fit_hod_wp.py configs/hod_fit_more2015_cmass.yml --map-only
"""

import argparse
import os
import sys

# sys.path.insert removed — hod_mod is installed

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.fitting import load_config, WpFitter


def parse_args():
    parser = argparse.ArgumentParser(description="Fit HOD to wp(rp)")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--plot", action="store_true", help="Save best-fit plot")
    parser.add_argument("--map-only",  action="store_true", dest="map_only")
    parser.add_argument("--mcmc-only", action="store_true", dest="mcmc_only")
    return parser.parse_args()


def print_params(label, params, free_names):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    for name in free_names:
        print(f"  {name:20s} = {params[name]:.4f}")


def run_map(fitter: WpFitter) -> dict:
    print("\n[MAP] Running Nelder-Mead optimisation …")
    result = fitter.map_fit()
    print_params("MAP best-fit parameters", result["params"], fitter.config.free_params)
    ndof = result["ndof"]
    print(f"\n  chi2 / ndof = {result['chi2']:.2f} / {ndof} = {result['chi2']/max(ndof,1):.2f}")
    print(f"  Converged:  {result['success']} ({result['message']})")
    return result


def run_mcmc(fitter: WpFitter, map_result: dict | None = None) -> dict:
    print("\n[MCMC] Running emcee …")
    initial_pos = None
    if map_result is not None:
        n_free    = len(fitter.config.free_params)
        n_walkers = fitter.config.n_walkers
        x_map     = map_result["theta"]
        bounds    = [fitter.config.param_bounds[p] for p in fitter.config.free_params]
        widths    = [0.02 * (b[1] - b[0]) for b in bounds]
        initial_pos = np.clip(
            x_map + np.random.randn(n_walkers, n_free) * widths,
            [b[0] for b in bounds],
            [b[1] for b in bounds],
        )

    sampler = fitter.sample(initial_pos=initial_pos)
    flatchain = sampler.get_chain(flat=True)

    med = np.median(flatchain, axis=0)
    lo  = np.percentile(flatchain, 16, axis=0)
    hi  = np.percentile(flatchain, 84, axis=0)

    print(f"\n{'─'*50}")
    print("  MCMC marginalized parameters (median ± 1σ)")
    print(f"{'─'*50}")
    for i, name in enumerate(fitter.config.free_params):
        print(f"  {name:20s} = {med[i]:.4f} + {hi[i]-med[i]:.4f} - {med[i]-lo[i]:.4f}")

    from hod_mod.fitting.hod_wp import _assemble_hod_params
    med_params = _assemble_hod_params(
        med, fitter.config.free_params,
        {k: v for k, v in fitter.config.param_init.items()
         if k not in fitter.config.free_params}
    )
    chi2_med = fitter.chi2(med_params)
    ndof = len(fitter.rp_arr) - len(fitter.config.free_params)
    print(f"\n  chi2(median) / ndof = {chi2_med:.2f} / {ndof} = {chi2_med/max(ndof,1):.2f}")
    return {"sampler": sampler, "median_params": med_params, "flatchain": flatchain}


def make_plot(fitter: WpFitter, best_params: dict, output_dir: str, tag: str = ""):
    wp_best = fitter.predict_wp(best_params)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
                             sharex=True)

    ax, ax_res = axes

    # data
    ax.errorbar(
        fitter.rp_arr, fitter.wp_obs, yerr=fitter.wp_err,
        fmt="ko", ms=5, lw=1, capsize=3, label="More+2015 data",
    )
    ax.loglog(fitter.rp_arr, wp_best, "r-", lw=2, label=f"{fitter.config.hod_model} (best fit)")

    ax.set_ylabel(r"$w_p(r_p)$ [$h^{-1}$ Mpc]")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        fr"BOSS CMASS $z_{{eff}}={fitter.config.z}$, "
        fr"$\pi_{{max}}={fitter.config.pi_max:.0f}$ $h^{{-1}}$ Mpc"
    )

    # residuals
    ratio = (fitter.wp_obs - wp_best) / fitter.wp_err
    ax_res.axhline(0, color="r", lw=1.5, ls="--")
    ax_res.errorbar(fitter.rp_arr, ratio, yerr=1.0, fmt="ko", ms=5, capsize=3)
    ax_res.axhspan(-2, 2, color="gray", alpha=0.15)
    ax_res.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
    ax_res.set_ylabel(r"$(w_p^{\rm obs} - w_p^{\rm mod}) / \sigma$")
    ax_res.set_ylim(-4, 4)
    ax_res.grid(True, alpha=0.3)

    ax.set_xscale("log")
    ax.set_yscale("log")

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"wp_bestfit{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.close(fig)


def main():
    args = parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    print(f"  HOD model    : {config.hod_model}")
    print(f"  z            : {config.z}")
    print(f"  pi_max       : {config.pi_max} Mpc/h")
    print(f"  rp range     : [{config.rp_min}, {config.rp_max}] Mpc/h")
    print(f"  Free params  : {config.free_params}")
    print(f"  Method       : {config.method}")

    fitter = WpFitter(config)
    print(f"  Data points  : {len(fitter.rp_arr)}")

    method = config.method
    if args.map_only:
        method = "map"
    elif args.mcmc_only:
        method = "emcee"

    map_result  = None
    mcmc_result = None
    best_params = config.param_init.copy()

    if method in ("map", "both"):
        map_result  = run_map(fitter)
        best_params = map_result["params"]

    if method in ("emcee", "both"):
        mcmc_result = run_mcmc(fitter, map_result)
        best_params = mcmc_result["median_params"]

    if args.plot:
        tag = "_map" if method == "map" else "_mcmc" if method == "emcee" else ""
        make_plot(fitter, best_params, config.output_dir, tag=tag)


if __name__ == "__main__":
    main()
