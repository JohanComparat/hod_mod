#!/usr/bin/env python
"""Verify the ZuMandelbaum15HODModel against the digitized data from ZM15 Fig. 6.

Plots predicted WPRP and ESD at the published iHOD Table 2 parameters against the
digitized measurements for all 7 WPRP and 5 ESD bins in a single multi-panel figure.

This is the primary "does the code reproduce the paper?" check.

Output
------
  results/benchmarks/zumandelbaum2015_verification/zm15_verification_all_bins.png

Usage
-----
    python hod_mod/scripts/benchmarks/plot_zm15_verification.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hod_mod.paths import results_root

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

OUT_DIR = os.path.join(results_root(), "benchmarks/zumandelbaum2015_verification")

# Per-bin config files (same order as BIN_LABELS)
BIN_LABELS = [
    "9.4-9.8", "9.8-10.2", "10.2-10.6", "10.6-11.0",
    "11.0-11.2", "11.2-11.4", "11.4-12.0",
]
BIN_CONFIGS = [
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_9p4_9p8.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_9p8_10p2.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_10p2_10p6.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_10p6_11p0.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p0_11p2.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p2_11p4.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p4_12p0.yml",
]

# Published iHOD Table 2 parameters
IHOD_PARAMS = {
    "log10m_star_thresh": 10.2,   # placeholder — overridden per bin
    "log10m_star_max":    None,   # overridden per bin
    "lg_m1h":        12.10,
    "lg_m0star":     10.31,
    "beta":          0.33,
    "delta":         0.42,
    "gamma":         1.21,
    "sigma_lnmstar": 0.50,
    "eta":           -0.04,
    "fc":            0.86,
    "bsat":          8.98,
    "beta_sat":      0.90,
    "bcut":          0.86,
    "beta_cut":      0.41,
    "alpha_sat":     1.00,
}

# Colour per bin (7 colours)
COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def build_fitters():
    from hod_mod.fitting import load_config, WpFitter, JointFitter
    fitters = []
    for cfg_rel, label in zip(BIN_CONFIGS, BIN_LABELS):
        cfg_path = os.path.join(_REPO_ROOT, cfg_rel)
        config   = load_config(cfg_path)
        has_wp   = bool(config.data_file and os.path.isfile(config.data_file))
        has_ds   = config.ds_file is not None
        f = JointFitter(config) if (has_wp and has_ds) else WpFitter(config)
        # Stash per-bin fixed params
        f._bin_label = label
        f._thresh    = config.param_init["log10m_star_thresh"]
        f._max_thresh = config.param_init.get("log10m_star_max")
        f._has_ds    = has_ds
        fitters.append(f)
    return fitters


def predict(fitter, params_override=None):
    """Return (rp, wp_pred) and optionally (R, ds_pred) at given params."""
    import jax.numpy as jnp
    params = dict(fitter._fixed_params)
    if params_override:
        params.update(params_override)

    theta_cosmo = fitter._theta_cosmo_call(params)
    wp_pred = np.asarray(fitter.predictor.wp(
        jnp.array(fitter.rp_arr), fitter.config.pi_max,
        fitter.config.z, theta_cosmo, params))

    ds_pred = None
    if fitter._has_ds:
        ds_pred = np.asarray(fitter.predict_ds(params))
    return wp_pred, ds_pred


def make_figure(fitters):
    n_bins   = len(fitters)
    n_ds_bins = sum(1 for f in fitters if f._has_ds)

    # 2 rows (wp, ds) × 7 columns; ds row has empty slots for first 2
    fig, axes = plt.subplots(2, n_bins, figsize=(3.0 * n_bins, 7),
                              sharex="col",
                              gridspec_kw={"hspace": 0.05})

    for col, (fitter, color) in enumerate(zip(fitters, COLORS)):
        label = fitter._bin_label

        # Build per-bin published params (override thresh / max)
        params = dict(IHOD_PARAMS)
        params["log10m_star_thresh"] = fitter._thresh
        params["log10m_star_max"]    = fitter._max_thresh

        try:
            wp_pred, ds_pred = predict(fitter, params)
        except Exception as exc:
            print(f"  [{label}] prediction failed: {exc}")
            continue

        rp     = np.array(fitter.rp_arr)
        wp_obs = np.array(fitter.wp_obs)
        wp_err = np.array(fitter.wp_err)
        chi2_wp = float(np.sum(((wp_pred - wp_obs) / wp_err) ** 2))

        # --- wp panel ---
        ax = axes[0, col]
        ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=4, color="k",
                    lw=0.8, capsize=2, label="data")
        ax.loglog(rp, wp_pred, "-", color=color, lw=2.0,
                  label=f"iHOD  χ²={chi2_wp:.1f}")
        ax.set_title(rf"$\log M_*\in[{label}]$", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlim(0.04, 30)
        if col == 0:
            ax.set_ylabel(r"$w_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=9)
        else:
            ax.set_yticklabels([])

        # --- ds panel ---
        ax2 = axes[1, col]
        if fitter._has_ds and ds_pred is not None:
            R      = np.array(fitter.R_arr)
            ds_obs = np.array(fitter.ds_obs)
            ds_err = np.array(fitter.ds_err)
            chi2_ds = float(np.sum(((ds_pred - ds_obs) / ds_err) ** 2))
            ax2.errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=4, color="k",
                         lw=0.8, capsize=2)
            ax2.loglog(R, ds_pred, "-", color=color, lw=2.0,
                       label=f"χ²={chi2_ds:.1f}")
            ax2.legend(fontsize=6, loc="upper right")
            ax2.set_xlim(0.04, 20)
            if col == 0:
                ax2.set_ylabel(r"$\Delta\Sigma\ [M_\odot\,h\,\mathrm{pc}^{-2}]$",
                                fontsize=9)
            else:
                ax2.set_yticklabels([])
        else:
            ax2.text(0.5, 0.5, "ESD not used", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=8, color="gray")
            ax2.set_axis_off()

        ax2.set_xlabel(r"$r_p$ or $R\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=8)

    fig.suptitle(
        "Zu & Mandelbaum 2015 — iHOD model at published Table 2 parameters\n"
        r"vs. digitized WPRP and ESD from Figure 6",
        fontsize=11, y=1.01,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "zm15_verification_all_bins.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main():
    print("Building fitters (JAX compilation ~60 s) …")
    fitters = build_fitters()
    print(f"  {len(fitters)} bins loaded")

    print("\nGenerating verification figure …")
    out = make_figure(fitters)
    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
