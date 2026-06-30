#!/usr/bin/env python
"""Generate MCMC figures for the Zu & Mandelbaum 2015 joint benchmark.

Produces three files in results/benchmarks/zumandelbaum2015_sdss/:
  - corner_zumandelbaum2015.png          : posterior corner plot (9 free params)
  - benchmark_zm15_wp_ds_mcmc.png        : wp + ΔΣ with MCMC band + published iHOD/cHOD
  - comparison_zm15_published.png        : focused MCMC median vs iHOD/cHOD published params

Usage
-----
    python hod_mod/scripts/benchmarks/plot_zumandelbaum2015_mcmc.py
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

CHAIN_FILE  = os.path.join(results_root(), "benchmarks/zumandelbaum2015_sdss/flatchain.npz")
CONFIG_FILE = os.path.join(_REPO_ROOT, "configs/benchmarks/benchmark_zumandelbaum2015.yml")
OUT_DIR     = os.path.join(results_root(), "benchmarks/zumandelbaum2015_sdss")

# Fixed params (not sampled in MCMC)
FIXED_PARAMS = {
    "log10m_star_thresh": 10.2,
    "beta_sat":  0.90,
    "bcut":      0.86,
    "beta_cut":  0.41,
    "alpha_sat": 1.00,
}

# Published iHOD best-fit — Table 2 of ZM15
IHOD_PARAMS = {
    "lg_m1h":        12.10,
    "lg_m0star":     10.31,
    "beta":          0.33,
    "delta":         0.42,
    "gamma":         1.21,
    "sigma_lnmstar": 0.50,
    "eta":           -0.04,
    "fc":            0.86,
    "bsat":          8.98,
    **FIXED_PARAMS,
}

IHOD_ERRORS = {
    "lg_m1h":        0.17,
    "lg_m0star":     0.10,
    "beta":          0.21,
    "delta":         0.04,
    "gamma":         0.20,
    "sigma_lnmstar": 0.04,
    "eta":           0.02,
    "fc":            0.14,
    "bsat":          1.18,
}

CHOD_PARAMS = {
    **IHOD_PARAMS,
    "lg_m1h":    12.32,
    "lg_m0star": 10.47,
    "beta":      0.54,
    "delta":     0.42,
    "gamma":     1.05,
}

# MAP params — filled from benchmark_result.json after the MAP run
MAP_PARAMS = None   # updated by load_map_params()

FREE_PARAMS = [
    "lg_m1h", "lg_m0star", "beta", "delta", "gamma",
    "sigma_lnmstar", "eta", "fc", "bsat",
]

PARAM_LABELS = {
    "lg_m1h":        r"$\log_{10}M_{1h}$",
    "lg_m0star":     r"$\log_{10}M_{*0}$",
    "beta":          r"$\beta$",
    "delta":         r"$\delta$",
    "gamma":         r"$\gamma$",
    "sigma_lnmstar": r"$\sigma_{\ln M_*}$",
    "eta":           r"$\eta$",
    "fc":            r"$f_c$",
    "bsat":          r"$B_{\rm sat}$",
}

N_SAMPLES = 400
RNG_SEED  = 42


# ---------------------------------------------------------------------------
# Load chain and MAP params
# ---------------------------------------------------------------------------

def load_chain():
    d = np.load(CHAIN_FILE)
    return d["flatchain"], list(d["param_names"])


def load_map_params():
    import json
    result_file = os.path.join(OUT_DIR, "benchmark_result.json")
    if not os.path.exists(result_file):
        print(f"WARNING: {result_file} not found — MAP overlay will be skipped")
        return None
    with open(result_file) as f:
        result = json.load(f)
    map_p = result.get("params", {})
    return {**FIXED_PARAMS, **map_p}


# ---------------------------------------------------------------------------
# MCMC band helpers — wp and ΔΣ
# ---------------------------------------------------------------------------

def mcmc_band_wp(chain, names, fitter, n_samples=N_SAMPLES):
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(chain), size=n_samples, replace=False)
    wp_samples = []
    for row in chain[idx]:
        params = {**FIXED_PARAMS, **dict(zip(names, row))}
        try:
            wp_samples.append(np.array(fitter.predict_wp(params)))
        except Exception:
            pass
    wp_arr = np.array(wp_samples)
    return (
        np.percentile(wp_arr, 16, axis=0),
        np.median(wp_arr, axis=0),
        np.percentile(wp_arr, 84, axis=0),
    )


def mcmc_band_ds(chain, names, fitter, n_samples=N_SAMPLES):
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(chain), size=n_samples, replace=False)
    ds_samples = []
    for row in chain[idx]:
        params = {**FIXED_PARAMS, **dict(zip(names, row))}
        try:
            ds_samples.append(np.array(fitter.predict_ds(params)))
        except Exception:
            pass
    ds_arr = np.array(ds_samples)
    return (
        np.percentile(ds_arr, 16, axis=0),
        np.median(ds_arr, axis=0),
        np.percentile(ds_arr, 84, axis=0),
    )


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

def _hist2d(ax, x, y, bins=30):
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    H = H.T
    H = gaussian_filter(H, sigma=1.0)
    levels = _contour_levels(H, [0.393, 0.865])
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    ax.contourf(xc, yc, H, levels=[levels[1], H.max()], colors=["C0"], alpha=0.25)
    ax.contourf(xc, yc, H, levels=[levels[0], H.max()], colors=["C0"], alpha=0.40)
    ax.contour(xc, yc, H, levels=levels, colors=["C0", "C0"], linewidths=[0.8, 1.2])


def _contour_levels(H, fractions):
    h_sorted = np.sort(H.ravel())[::-1]
    cumsum = np.cumsum(h_sorted)
    cumsum /= cumsum[-1]
    levels = []
    for f in fractions:
        idx = np.searchsorted(cumsum, f)
        levels.append(h_sorted[idx] if idx < len(h_sorted) else h_sorted[-1])
    return sorted(levels)


def make_corner(chain, names):
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    map_p = MAP_PARAMS or {}

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            pname_j = names[j]
            pname_i = names[i]
            xi = chain[:, j]
            yi = chain[:, i]
            pub_j = IHOD_PARAMS[pname_j]
            pub_i = IHOD_PARAMS[pname_i]

            if i == j:
                ax.hist(xi, bins=50, color="C0", alpha=0.7, density=True)
                if pname_j in map_p:
                    ax.axvline(map_p[pname_j], color="C3", lw=1.5, ls="--")
                ax.axvline(pub_j, color="C1", lw=1.5, ls="-")
                p16, p84 = np.percentile(xi, [16, 84])
                ax.axvline(p16, color="C0", lw=0.8, ls=":")
                ax.axvline(p84, color="C0", lw=0.8, ls=":")
                ax.set_yticks([])
            else:
                _hist2d(ax, xi, yi)
                if pname_j in map_p and pname_i in map_p:
                    ax.axvline(map_p[pname_j], color="C3", lw=0.9, ls="--", alpha=0.8)
                    ax.axhline(map_p[pname_i], color="C3", lw=0.9, ls="--", alpha=0.8)
                    ax.plot(map_p[pname_j], map_p[pname_i], "C3+", ms=8, mew=1.5, zorder=5)
                ax.axvline(pub_j, color="C1", lw=0.9, ls="-", alpha=0.8)
                ax.axhline(pub_i, color="C1", lw=0.9, ls="-", alpha=0.8)
                ax.plot(pub_j, pub_i, "C1*", ms=9, zorder=5)

            if i == n - 1:
                ax.set_xlabel(PARAM_LABELS.get(pname_j, pname_j), fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(PARAM_LABELS.get(pname_i, pname_i), fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=6)

    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], color="C3", ls="--", lw=1.5, label="MAP"),
        Line2D([0], [0], color="C1", ls="-",  lw=1.5, label="Published iHOD (ZM15 Table 2)"),
        Line2D([0], [0], color="C0", ls="-",  lw=4,   alpha=0.6,
               label=r"Posterior $1\sigma$/$2\sigma$"),
    ]
    axes[0, 0].legend(handles=proxy, fontsize=7, loc="upper left",
                      bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    fig.suptitle(
        r"Zu & Mandelbaum 2015 iHOD — SDSS DR7 posterior ($w_p + \Delta\Sigma$)" + "\n"
        r"(dashed red = MAP, solid orange = published iHOD, shaded = $1\sigma$/$2\sigma$)",
        fontsize=11, y=0.995,
    )
    out = os.path.join(OUT_DIR, "corner_zumandelbaum2015.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# wp + ΔΣ with MCMC band
# ---------------------------------------------------------------------------

def make_wp_ds_mcmc(chain, names, fitter):
    print("  Computing wp MCMC band …")
    wp_lo, wp_med, wp_hi = mcmc_band_wp(chain, names, fitter)
    print("  Computing ΔΣ MCMC band …")
    ds_lo, ds_med, ds_hi = mcmc_band_ds(chain, names, fitter)

    rp     = np.array(fitter.rp_arr)
    wp_obs = np.array(fitter.wp_obs)
    wp_err = np.array(fitter.wp_err)
    R      = np.array(fitter.R_arr)
    ds_obs = np.array(fitter.ds_obs)
    ds_err = np.array(fitter.ds_err)

    wp_ihod = np.array(fitter.predict_wp(IHOD_PARAMS))
    wp_chod = np.array(fitter.predict_wp(CHOD_PARAMS))
    ds_ihod = np.array(fitter.predict_ds(IHOD_PARAMS))
    ds_chod = np.array(fitter.predict_ds(CHOD_PARAMS))

    map_p = MAP_PARAMS
    wp_map = np.array(fitter.predict_wp(map_p)) if map_p else None
    ds_map = np.array(fitter.predict_ds(map_p)) if map_p else None

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06, "wspace": 0.30},
    )

    # ---- wp main ----
    ax = axes[0, 0]
    ax.fill_between(rp, wp_lo, wp_hi, color="C0", alpha=0.25,
                    label=r"MCMC $1\sigma$ band")
    ax.loglog(rp, wp_med, "--", color="C0", lw=1.5, label="MCMC median")
    if wp_map is not None:
        ax.loglog(rp, wp_map, "-", color="C0", lw=2.2, label="MAP")
    ax.loglog(rp, wp_ihod, "-",  color="C1", lw=2.0, label="Published iHOD")
    ax.loglog(rp, wp_chod, "--", color="C2", lw=1.8, label="Published cHOD")
    ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=5, color="k",
                label="SDSS DR7 data", zorder=5)
    ax.set_ylabel(r"$w_p(r_p)\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        r"Zu & Mandelbaum 2015 iHOD — SDSS DR7, $\log_{10}(M_*/h^{-2}M_\odot)>10.2$",
        fontsize=10,
    )

    # ---- wp residuals ----
    ax2 = axes[1, 0]
    ax2.fill_between(rp, wp_obs / wp_hi - 1, wp_obs / wp_lo - 1,
                     color="C0", alpha=0.25)
    ax2.axhline(0, color="k", lw=0.9, ls="--")
    if wp_map is not None:
        ax2.errorbar(rp, wp_obs / wp_map - 1, yerr=wp_err / wp_map,
                     fmt="o", ms=4, color="C0", label="data/MAP − 1")
    ax2.plot(rp, wp_obs / wp_ihod - 1, "-",  color="C1", lw=1.6,
             label="data/iHOD − 1")
    ax2.plot(rp, wp_obs / wp_chod - 1, "--", color="C2", lw=1.4,
             label="data/cHOD − 1")
    ax2.axhline( 0.15, color="gray", lw=0.5, ls=":")
    ax2.axhline(-0.15, color="gray", lw=0.5, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$r_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax2.set_ylabel(r"data/model $-1$", fontsize=11)
    ax2.set_ylim(-0.55, 0.55)
    ax2.legend(fontsize=7, loc="upper right")

    # ---- ΔΣ main ----
    ax3 = axes[0, 1]
    ax3.fill_between(R, ds_lo, ds_hi, color="C0", alpha=0.25,
                     label=r"MCMC $1\sigma$ band")
    ax3.loglog(R, ds_med, "--", color="C0", lw=1.5, label="MCMC median")
    if ds_map is not None:
        ax3.loglog(R, ds_map, "-", color="C0", lw=2.2, label="MAP")
    ax3.loglog(R, ds_ihod, "-",  color="C1", lw=2.0, label="Published iHOD")
    ax3.loglog(R, ds_chod, "--", color="C2", lw=1.8, label="Published cHOD")
    ax3.errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=5, color="k",
                 label="SDSS lensing data", zorder=5)
    ax3.set_ylabel(r"$\Delta\Sigma(R)\ [M_\odot\,h\,\mathrm{pc}^{-2}]$", fontsize=12)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_title(r"$z_{\rm eff}=0.1$, Mandelbaum+2006 lensing", fontsize=10)

    # ---- ΔΣ residuals ----
    ax4 = axes[1, 1]
    ax4.fill_between(R, ds_obs / ds_hi - 1, ds_obs / ds_lo - 1,
                     color="C0", alpha=0.25)
    ax4.axhline(0, color="k", lw=0.9, ls="--")
    if ds_map is not None:
        ax4.errorbar(R, ds_obs / ds_map - 1, yerr=ds_err / ds_map,
                     fmt="s", ms=4, color="C0", label=r"data/MAP $-1$")
    ax4.plot(R, ds_obs / ds_ihod - 1, "-",  color="C1", lw=1.6,
             label=r"data/iHOD $-1$")
    ax4.plot(R, ds_obs / ds_chod - 1, "--", color="C2", lw=1.4,
             label=r"data/cHOD $-1$")
    ax4.axhline( 0.20, color="gray", lw=0.5, ls=":")
    ax4.axhline(-0.20, color="gray", lw=0.5, ls=":")
    ax4.set_xscale("log")
    ax4.set_xlabel(r"$R\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax4.set_ylabel(r"data/model $-1$", fontsize=11)
    ax4.set_ylim(-0.55, 0.55)
    ax4.legend(fontsize=7, loc="upper right")

    out = os.path.join(OUT_DIR, "benchmark_zm15_wp_ds_mcmc.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Focused comparison: MCMC median vs published iHOD / cHOD
# ---------------------------------------------------------------------------

def make_comparison_published(chain, names, fitter):
    wp_lo, wp_med, wp_hi = mcmc_band_wp(chain, names, fitter)
    ds_lo, ds_med, ds_hi = mcmc_band_ds(chain, names, fitter)

    rp     = np.array(fitter.rp_arr)
    wp_obs = np.array(fitter.wp_obs)
    wp_err = np.array(fitter.wp_err)
    R      = np.array(fitter.R_arr)
    ds_obs = np.array(fitter.ds_obs)
    ds_err = np.array(fitter.ds_err)

    wp_ihod = np.array(fitter.predict_wp(IHOD_PARAMS))
    ds_ihod = np.array(fitter.predict_ds(IHOD_PARAMS))

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06, "wspace": 0.30},
    )

    for ax, x, obs, err, med, lo, hi, pub, xlabel, ylabel, title in [
        (
            axes[0, 0], rp, wp_obs, wp_err, wp_med, wp_lo, wp_hi, wp_ihod,
            None,
            r"$w_p(r_p)\ [h^{-1}\,\mathrm{Mpc}]$",
            r"$w_p$: MCMC median vs published iHOD",
        ),
        (
            axes[0, 1], R, ds_obs, ds_err, ds_med, ds_lo, ds_hi, ds_ihod,
            None,
            r"$\Delta\Sigma(R)\ [M_\odot\,h\,\mathrm{pc}^{-2}]$",
            r"$\Delta\Sigma$: MCMC median vs published iHOD",
        ),
    ]:
        ax.fill_between(x, lo, hi, color="C0", alpha=0.25,
                        label=r"Our MCMC $1\sigma$ band")
        ax.loglog(x, med, "-", color="C0", lw=2.0, label="Our MCMC median")
        ax.loglog(x, pub, "-", color="C1", lw=2.0, label="Published iHOD (ZM15 Table 2)")
        ax.errorbar(x, obs, yerr=err, fmt="o", ms=5, color="k",
                    label="SDSS DR7 data", zorder=5)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

    for ax2, x, obs, med, lo, hi, pub, err, xlabel in [
        (axes[1, 0], rp, wp_obs, wp_med, wp_lo, wp_hi, wp_ihod, wp_err,
         r"$r_p\ [h^{-1}\,\mathrm{Mpc}]$"),
        (axes[1, 1], R,  ds_obs, ds_med, ds_lo, ds_hi, ds_ihod, ds_err,
         r"$R\ [h^{-1}\,\mathrm{Mpc}]$"),
    ]:
        ax2.fill_between(x, obs / hi - 1, obs / lo - 1, color="C0", alpha=0.25)
        ax2.axhline(0, color="k", lw=0.9, ls="--")
        ax2.errorbar(x, obs / med - 1, yerr=err / med,
                     fmt="o", ms=4, color="C0", label="data/MCMC median − 1")
        ax2.plot(x, obs / pub - 1, "-", color="C1", lw=1.6,
                 label="data/Published − 1")
        ax2.axhline( 0.15, color="gray", lw=0.5, ls=":")
        ax2.axhline(-0.15, color="gray", lw=0.5, ls=":")
        ax2.set_xscale("log")
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel(r"data/model $-1$", fontsize=11)
        ax2.set_ylim(-0.55, 0.55)
        ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        r"Zu & Mandelbaum 2015 — SDSS DR7 reproduction vs published iHOD",
        fontsize=12, y=1.01,
    )
    out = os.path.join(OUT_DIR, "comparison_zm15_published.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MAP_PARAMS
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading MCMC chain …")
    chain, names = load_chain()
    print(f"  chain shape: {chain.shape}  param_names: {names}")

    MAP_PARAMS = load_map_params()
    if MAP_PARAMS:
        print(f"  MAP params loaded: {MAP_PARAMS}")

    print("\nLoading JointFitter …")
    from hod_mod.fitting import load_config, JointFitter
    config = load_config(CONFIG_FILE)
    fitter = JointFitter(config)
    print(f"  wp bins: {len(fitter.rp_arr)}, ΔΣ bins: {len(fitter.R_arr)}")

    print("\nMaking corner plot …")
    make_corner(chain, names)

    print("\nMaking wp + ΔΣ MCMC band figure …")
    make_wp_ds_mcmc(chain, names, fitter)

    print("\nMaking focused published comparison figure …")
    make_comparison_published(chain, names, fitter)

    print("\nDone.")


if __name__ == "__main__":
    main()
