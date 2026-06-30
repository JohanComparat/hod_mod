"""run_fit_More15.py — Fit the More+2015 HOD model to wp(rp), ΔΣ(R), or both.

Wraps the validated :class:`~hod_mod.fitting.WpFitter`,
:class:`~hod_mod.fitting.JointFitter`, and
:class:`~hod_mod.fitting.DeltaSigmaFitter` classes.
The probe mode (wp-only, ESD-only, or joint wp+ESD) is inferred automatically
from the YAML configuration: include a ``data:`` section for wp, and/or a
``joint:`` / ``ds:`` section for ΔΣ.

Usage
-----
::

    # MAP fit only (fast)
    python hod_mod/scripts/fitting/run_fit_More15.py  config.yml --map-only

    # MAP + emcee MCMC
    python hod_mod/scripts/fitting/run_fit_More15.py  config.yml --mcmc

    # Reload saved MAP result and regenerate figures
    python hod_mod/scripts/fitting/run_fit_More15.py  config.yml --plot-only

    # Override output directory
    python hod_mod/scripts/fitting/run_fit_More15.py  config.yml --output-dir /tmp/test/

See ``docs/fitting_more2015.rst`` for the full YAML configuration reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Plotting helpers (same functions used by run_benchmark.py)
# ---------------------------------------------------------------------------

from hod_mod.scripts.benchmarks.benchmark_plots import (
    _COL_DATA,
    _COL_MAP,
    _COL_PUB,
    load_flatchain,
    mcmc_bands,
    add_bands,
    residual_panel,
    plot_hod,
    plot_corner,
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_result(map_result: dict, label: str) -> None:
    chi2_ndof = (map_result["chi2"] / map_result["ndof"]
                 if map_result["ndof"] > 0 else float("nan"))
    print(f"\n{'='*60}")
    print(f"Fit: {label}")
    print(f"{'='*60}")
    print(f"  chi2 / ndof = {map_result['chi2']:.3f} / {map_result['ndof']}"
          f"  →  chi2/dof = {chi2_ndof:.3f}")
    print(f"  Optimizer:  {'OK' if map_result['success'] else 'WARNING: ' + map_result['message']}")
    print(f"\n  Best-fit parameters:")
    for name, val in map_result["params"].items():
        marker = "  (free)" if name in map_result.get("free_params", []) else ""
        print(f"    {name:20s} = {val:10.5g}{marker}")


def _make_plots(
    fitter,
    params: dict,
    chi2_ndof: float,
    label: str,
    output_dir: str,
    published: dict | None = None,
) -> None:
    """Save wp, ΔΣ, combined, HOD, and (if flatchain exists) corner figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # Build published-params overlay (used when user supplies published_params: in YAML)
    pub_params = None
    if published:
        pub_params = dict(params)
        for pname, pentry in published.items():
            pub_val = pentry[0] if isinstance(pentry, (list, tuple)) else float(pentry)
            pub_params[pname] = pub_val

    # Load MCMC flatchain if it exists
    flatchain, fc_names = load_flatchain(output_dir)

    has_wp = hasattr(fitter, "rp_arr") and fitter.rp_arr is not None
    has_ds = hasattr(fitter, "predict_ds")

    rp = wp_obs = wp_err = wp_pred = wp_pub = wp_bands = None
    R  = ds_obs = ds_err = ds_pred = ds_pub = ds_bands = None

    # ---- wp ---------------------------------------------------------------
    if has_wp:
        rp      = np.array(fitter.rp_arr)
        wp_obs  = np.array(fitter.wp_obs)
        wp_err  = (np.sqrt(np.diag(np.linalg.inv(fitter.icov_wp)))
                   if hasattr(fitter, "icov_wp") else np.array(fitter.wp_err))
        wp_pred = np.array(fitter.predict_wp(params))
        if pub_params:
            try:
                wp_pub = np.array(fitter.predict_wp(pub_params))
            except Exception:
                pass
        wp_bands = (mcmc_bands(fitter.predict_wp, params, flatchain, fc_names)
                    if flatchain is not None else None)

        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        add_bands(axes[0], rp, wp_bands, _COL_MAP)
        axes[0].errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=4,
                         color=_COL_DATA, zorder=5, label="Data")
        axes[0].loglog(rp, wp_pred, "-", color=_COL_MAP, lw=1.8,
                       label=f"MAP (χ²/dof={chi2_ndof:.2f})")
        if wp_bands is not None:
            axes[0].plot([], [], "--", color=_COL_MAP, lw=1.5,
                         label="MCMC median ± 68/95%")
        if wp_pub is not None:
            axes[0].loglog(rp, wp_pub, "--", color=_COL_PUB, lw=1.5,
                           label="Reference best-fit")
        axes[0].set_ylabel(r"$w_p(r_p)$ [$h^{-1}$ Mpc]")
        axes[0].legend(fontsize=9)
        axes[0].set_title(label, fontsize=10)
        residual_panel(axes[1], rp, wp_obs, wp_pred, wp_err,
                       pub=wp_pub, bands=wp_bands, fmt="o")
        axes[1].set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
        fig.tight_layout()
        out = os.path.join(output_dir, "fit_wp.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- ΔΣ ---------------------------------------------------------------
    if has_ds:
        R       = np.array(fitter.R_arr)
        ds_obs  = np.array(fitter.ds_obs)
        ds_err  = (np.sqrt(np.diag(np.linalg.inv(fitter.icov_ds)))
                   if hasattr(fitter, "icov_ds") else np.array(fitter.ds_err))
        ds_pred = np.array(fitter.predict_ds(params))
        if pub_params:
            try:
                ds_pub = np.array(fitter.predict_ds(pub_params))
            except Exception:
                pass
        ds_bands = (mcmc_bands(fitter.predict_ds, params, flatchain, fc_names)
                    if flatchain is not None else None)

        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
        add_bands(axes2[0], R, ds_bands, _COL_MAP)
        axes2[0].errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=4,
                          color=_COL_DATA, zorder=5, label="Data")
        axes2[0].loglog(R, ds_pred, "-", color=_COL_MAP, lw=1.8,
                        label=f"MAP (χ²/dof={chi2_ndof:.2f})")
        if ds_bands is not None:
            axes2[0].plot([], [], "--", color=_COL_MAP, lw=1.5,
                          label="MCMC median ± 68/95%")
        if ds_pub is not None:
            axes2[0].loglog(R, ds_pub, "--", color=_COL_PUB, lw=1.5,
                            label="Reference best-fit")
        axes2[0].set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
        axes2[0].legend(fontsize=9)
        axes2[0].set_title(label + (" — ΔΣ only" if not has_wp else " — ΔΣ"), fontsize=10)
        residual_panel(axes2[1], R, ds_obs, ds_pred, ds_err,
                       pub=ds_pub, bands=ds_bands, fmt="s")
        axes2[1].set_xlabel(r"$R$ [$h^{-1}$ Mpc]")
        fig2.tight_layout()
        out2 = os.path.join(output_dir, "fit_ds.png")
        fig2.savefig(out2, dpi=150)
        plt.close(fig2)
        print(f"  Saved: {out2}")

    # ---- combined (rp×wp and ΔΣ side by side) -----------------------------
    if has_wp or has_ds:
        n_cols = int(has_wp) + int(has_ds)
        fig3, axes3 = plt.subplots(
            2, n_cols, figsize=(6 * n_cols + 0.5, 8),
            sharex="col", squeeze=False,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05,
                         "wspace": 0.3 if n_cols > 1 else 0.0},
        )
        col = 0
        if has_wp:
            ax0, ax1 = axes3[0, col], axes3[1, col]
            add_bands(ax0, rp, wp_bands, _COL_MAP, scale=rp)
            ax0.errorbar(rp, rp * wp_obs, yerr=rp * wp_err, fmt="o", ms=4,
                         color=_COL_DATA, zorder=5, label="Data")
            ax0.loglog(rp, rp * wp_pred, "-", color=_COL_MAP, lw=1.8,
                       label=f"MAP (χ²/dof={chi2_ndof:.2f})")
            if wp_pub is not None:
                ax0.loglog(rp, rp * wp_pub, "--", color=_COL_PUB, lw=1.5,
                           label="Reference best-fit")
            ax0.set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}{\rm Mpc}^2$]")
            ax0.legend(fontsize=8)
            ax0.set_title(r"$w_p$", fontsize=10)
            residual_panel(ax1, rp, wp_obs, wp_pred, wp_err,
                           pub=wp_pub, bands=wp_bands, fmt="o")
            ax1.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
            col += 1
        if has_ds:
            ax0, ax1 = axes3[0, col], axes3[1, col]
            add_bands(ax0, R, ds_bands, _COL_MAP)
            ax0.errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=4,
                         color=_COL_DATA, zorder=5, label="Data")
            ax0.loglog(R, ds_pred, "-", color=_COL_MAP, lw=1.8,
                       label="MAP" if has_wp else f"MAP (χ²/dof={chi2_ndof:.2f})")
            if ds_pub is not None:
                ax0.loglog(R, ds_pub, "--", color=_COL_PUB, lw=1.5,
                           label="Reference best-fit")
            ax0.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
            ax0.legend(fontsize=8)
            ax0.set_title(r"$\Delta\Sigma$" if has_wp else label, fontsize=10)
            residual_panel(ax1, R, ds_obs, ds_pred, ds_err,
                           pub=ds_pub, bands=ds_bands, fmt="s")
            ax1.set_xlabel(r"$R$ [$h^{-1}$ Mpc]")
        fig3.suptitle(label, fontsize=11)
        out3 = os.path.join(output_dir, "fit_combined.png")
        fig3.savefig(out3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: {out3}")

    # ---- HOD ---------------------------------------------------------------
    # plot_hod saves as benchmark_{model_key}_hod.png; rename to fit_hod.png
    import re
    _slug = re.sub(r"[^\w]+", "_", label).strip("_") or "fit"
    plot_hod(fitter, params, pub_params, _slug, output_dir,
             flatchain=flatchain, param_names=fc_names)
    _hod_src = os.path.join(output_dir, f"benchmark_{_slug}_hod.png")
    _hod_dst = os.path.join(output_dir, "fit_hod.png")
    if os.path.exists(_hod_src):
        os.replace(_hod_src, _hod_dst)
        print(f"  Saved: {_hod_dst}")

    # ---- corner (MCMC only) ------------------------------------------------
    if flatchain is not None and published is not None:
        fixed = {k: float(v) for k, v in params.items() if k not in fc_names}
        plot_corner(flatchain, fc_names, published, _slug, output_dir,
                    fixed_params=fixed or None)
        _cor_src = os.path.join(output_dir, f"benchmark_{_slug}_corner.png")
        _cor_dst = os.path.join(output_dir, "fit_corner.png")
        if os.path.exists(_cor_src):
            os.replace(_cor_src, _cor_dst)
            print(f"  Saved: {_cor_dst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to YAML configuration file.")
    p.add_argument("--mcmc",       action="store_true",
                   help="Run emcee MCMC after MAP optimisation.")
    p.add_argument("--map-only",   action="store_true",
                   help="Run MAP only (skip MCMC).")
    p.add_argument("--plot-only",  action="store_true",
                   help="Skip fitting; reload fit_result.json and regenerate figures.")
    p.add_argument("--output-dir", metavar="DIR", default=None,
                   help="Override the output directory from the config.")
    args = p.parse_args()

    from hod_mod.fitting import load_config, WpFitter, JointFitter, DeltaSigmaFitter
    import yaml
    import dataclasses

    # --- load config --------------------------------------------------------
    config = load_config(args.config)
    if args.output_dir:
        config = dataclasses.replace(config, output_dir=args.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    # Read optional top-level keys not handled by load_config
    with open(args.config) as fh:
        raw_yaml = yaml.safe_load(fh)
    label     = raw_yaml.get("label", os.path.splitext(os.path.basename(args.config))[0])
    published = raw_yaml.get("published_params", None)   # {name: [value, error]} or None

    # --- select fitter -------------------------------------------------------
    has_wp = bool(config.data_file and os.path.isfile(config.data_file))
    has_ds = config.ds_file is not None
    if has_wp and has_ds:
        fitter = JointFitter(config)
        probe_str = "wp+esd"
    elif has_ds:
        fitter = DeltaSigmaFitter(config)
        probe_str = "esd"
    else:
        fitter = WpFitter(config)
        probe_str = "wp"

    print(f"\n{'='*60}")
    print(f"run_fit_More15  [{label}]")
    print(f"  Config:   {args.config}")
    print(f"  Probes:   {probe_str}")
    print(f"  Free params ({len(config.free_params)}): {config.free_params}")
    print(f"  Output:   {config.output_dir}")

    result_file = os.path.join(config.output_dir, "fit_result.json")

    # --- plot-only mode ------------------------------------------------------
    if args.plot_only:
        if not os.path.exists(result_file):
            sys.exit(f"ERROR: {result_file} not found — run without --plot-only first.")
        with open(result_file) as fh:
            saved = json.load(fh)
        params    = saved["params"]
        chi2_ndof = saved["chi2_ndof"]
        print(f"\n  Loaded MAP params from {result_file}  (χ²/dof={chi2_ndof:.3f})")
        print("\n=== Generating figures ===")
        _make_plots(fitter, params, chi2_ndof, label, config.output_dir, published)
        return

    # --- MAP fit -------------------------------------------------------------
    map_result = fitter.map_fit()
    # Attach free_params list for pretty-printing
    map_result["free_params"] = list(config.free_params)

    ndof      = map_result["ndof"]
    chi2      = map_result["chi2"]
    chi2_ndof = chi2 / ndof if ndof > 0 else float("nan")
    params    = map_result["params"]

    _print_result(map_result, label)

    # --- save JSON result ---------------------------------------------------
    result = {
        "config":     os.path.abspath(args.config),
        "label":      label,
        "probes":     probe_str,
        "chi2":       float(chi2),
        "ndof":       int(ndof),
        "chi2_ndof":  float(chi2_ndof),
        "success":    bool(map_result["success"]),
        "params":     {k: float(v) for k, v in params.items()},
        "free_params": list(config.free_params),
    }
    with open(result_file, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n  Result saved → {result_file}")

    # --- optional MCMC ------------------------------------------------------
    run_mcmc = (args.mcmc or config.method in ("mcmc", "both")) and not args.map_only
    if run_mcmc:
        flatchain_path = os.path.join(config.output_dir, "flatchain.npz")
        print("\n=== MCMC sampling ===")
        theta0    = np.asarray(map_result["theta"])
        scale     = np.maximum(np.abs(theta0) * 1e-3, 1e-4)
        rng       = np.random.default_rng(42)
        init_pos  = theta0[None, :] + rng.normal(0, scale, (config.n_walkers, len(theta0)))
        fitter.sample(initial_pos=init_pos)
        print(f"  Chain saved → {flatchain_path}")

    # --- plots ---------------------------------------------------------------
    print("\n=== Generating figures ===")
    _make_plots(fitter, params, chi2_ndof, label, config.output_dir, published)


if __name__ == "__main__":
    main()
