#!/usr/bin/env python
"""Generate MCMC figures for the kravtsov2004 benchmark.

Produces two files in results/benchmarks/kravtsov2004_cmass/:
  - corner_kravtsov2004.png        : posterior corner plot
  - benchmark_kravtsov2004_wp_mcmc.png : wp with MCMC uncertainty band

Usage
-----
    python hod_mod/scripts/benchmarks/plot_kravtsov2004_mcmc.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHAIN_FILE  = os.path.join(_REPO_ROOT, "results/benchmarks/kravtsov2004_cmass/flatchain.npz")
CONFIG_FILE = os.path.join(_REPO_ROOT, "configs/benchmarks/benchmark_kravtsov2004.yml")
OUT_DIR     = os.path.join(_REPO_ROOT, "results/benchmarks/kravtsov2004_cmass")

MAP_PARAMS = {
    "log10mmin":  13.193282137252126,
    "sigma_logm": 0.40123431017189365,
    "log10m0":    14.334179733334896,
    "log10m1":    13.404553559785294,
    "alpha":      1.0752246579296978,
}

PARAM_LABELS = {
    "log10mmin":  r"$\log_{10}M_{\min}$",
    "sigma_logm": r"$\sigma_{\log M}$",
    "log10m0":    r"$\log_{10}M_0$",
    "log10m1":    r"$\log_{10}M_1$",
    "alpha":      r"$\alpha$",
}

N_SAMPLES_WP = 400   # chain draws used for the wp uncertainty band
RNG_SEED     = 42

MORE2015_CHAIN_FILE  = os.path.join(_REPO_ROOT, "results/more2015_cmass/flatchain.npz")
MORE2015_CONFIG_FILE = os.path.join(_REPO_ROOT, "configs/benchmarks/benchmark_more2015.yml")

MORE2015_MAP_PARAMS = {
    "log10mmin":  12.891854619247049,
    "sigma_logm": 0.050000207755272114,
    "log10m1":    13.683177504021286,
    "alpha":      1.4153494635487072,
    "kappa":      0.565173084427479,
    "alpha_inc":  1.0,
    "log10m_inc": 13.0,
}


# ---------------------------------------------------------------------------
# Load chain
# ---------------------------------------------------------------------------

def load_chain():
    d = np.load(CHAIN_FILE)
    chain = d["flatchain"]          # (N, 5)
    names = list(d["param_names"])  # ['log10mmin', ...]
    return chain, names


# ---------------------------------------------------------------------------
# Corner plot (pure matplotlib, lower-triangle)
# ---------------------------------------------------------------------------

def _hist2d(ax, x, y, bins=30, smooth=True):
    """2D histogram with optional smoothing."""
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    H = H.T
    if smooth:
        H = gaussian_filter(H, sigma=1.0)
    levels = _contour_levels(H, [0.393, 0.865])  # 1σ and 2σ filled fraction
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    ax.contourf(xc, yc, H, levels=[levels[1], H.max()], colors=["C0"], alpha=0.25)
    ax.contourf(xc, yc, H, levels=[levels[0], H.max()], colors=["C0"], alpha=0.40)
    ax.contour(xc, yc, H, levels=levels, colors=["C0", "C0"], linewidths=[0.8, 1.2])


def _contour_levels(H, fractions):
    """Find iso-density levels enclosing given probability fractions."""
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
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            xi = chain[:, j]
            yi = chain[:, i]
            map_j = MAP_PARAMS[names[j]]
            map_i = MAP_PARAMS[names[i]]

            if i == j:
                ax.hist(xi, bins=50, color="C0", alpha=0.7, density=True)
                ax.axvline(map_j, color="C3", lw=1.5, ls="--")
                p16, p84 = np.percentile(xi, [16, 84])
                ax.axvline(p16, color="C0", lw=0.8, ls=":")
                ax.axvline(p84, color="C0", lw=0.8, ls=":")
                ax.set_yticks([])
            else:
                _hist2d(ax, xi, yi)
                ax.axvline(map_j, color="C3", lw=1.0, ls="--", alpha=0.8)
                ax.axhline(map_i, color="C3", lw=1.0, ls="--", alpha=0.8)
                ax.plot(map_j, map_i, "C3+", ms=8, mew=1.5)

            # Labels only on outer edges
            if i == n - 1:
                ax.set_xlabel(PARAM_LABELS[names[j]], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(PARAM_LABELS[names[i]], fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Kravtsov+2004 HOD — BOSS CMASS posterior\n"
        r"(dashed red = MAP, shaded = $1\sigma$/$2\sigma$)",
        fontsize=11, y=0.995,
    )
    out = os.path.join(OUT_DIR, "corner_kravtsov2004.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# wp with MCMC band
# ---------------------------------------------------------------------------

def make_wp_mcmc(chain, names, fitter):
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(chain), size=N_SAMPLES_WP, replace=False)
    samples = chain[idx]

    wp_samples = []
    for row in samples:
        params = dict(zip(names, row))
        try:
            wp_samples.append(np.array(fitter.predict_wp(params)))
        except Exception:
            pass

    wp_samples = np.array(wp_samples)      # (N_ok, n_rp)
    wp_med = np.median(wp_samples, axis=0)
    wp_lo  = np.percentile(wp_samples, 16, axis=0)
    wp_hi  = np.percentile(wp_samples, 84, axis=0)

    rp     = np.array(fitter.rp_arr)
    wp_obs = np.array(fitter.wp_obs)
    wp_err = np.array(fitter.wp_err)
    wp_map = np.array(fitter.predict_wp(MAP_PARAMS))

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.fill_between(rp, wp_lo, wp_hi, color="C0", alpha=0.25,
                    label=r"MCMC $1\sigma$ band")
    ax.loglog(rp, wp_med, "--", color="C0", lw=1.4, label="MCMC median")
    ax.loglog(rp, wp_map, "-",  color="C0", lw=2.0,
              label=r"MAP ($\chi^2/\mathrm{dof}=1.91$)")
    ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=5, color="k",
                label="BOSS CMASS data (White+2014)")
    ax.set_ylabel(r"$w_p(r_p)\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_title("Kravtsov+2004 HODModel — BOSS CMASS z=0.52", fontsize=11)

    ax2 = axes[1]
    ratio_map = wp_obs / wp_map - 1
    ratio_err = wp_err / wp_map
    ratio_lo  = wp_obs / wp_hi - 1   # band edges flip in ratio
    ratio_hi  = wp_obs / wp_lo - 1
    ax2.fill_between(rp, ratio_lo, ratio_hi, color="C0", alpha=0.25)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.errorbar(rp, ratio_map, yerr=ratio_err, fmt="o", ms=5, color="k")
    ax2.axhline( 0.1, color="gray", lw=0.5, ls=":")
    ax2.axhline(-0.1, color="gray", lw=0.5, ls=":")
    ax2.set_xlabel(r"$r_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax2.set_ylabel(r"data/MAP $- 1$", fontsize=11)
    ax2.set_ylim(-0.55, 0.55)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "benchmark_kravtsov2004_wp_mcmc.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Model comparison figure (Kravtsov+2004 vs More+2015)
# ---------------------------------------------------------------------------

def _mcmc_band(chain, names, fitter, n_samples=N_SAMPLES_WP, fixed_params=None):
    """Return (wp_lo, wp_med, wp_hi) from random chain draws."""
    if fixed_params is None:
        fixed_params = {}
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(chain), size=n_samples, replace=False)
    wp_samples = []
    for row in chain[idx]:
        params = {**fixed_params, **dict(zip(names, row))}
        try:
            wp_samples.append(np.array(fitter.predict_wp(params)))
        except Exception:
            pass
    wp_samples = np.array(wp_samples)
    return (
        np.percentile(wp_samples, 16, axis=0),
        np.median(wp_samples, axis=0),
        np.percentile(wp_samples, 84, axis=0),
    )


def make_model_comparison(krav_chain, krav_names, krav_fitter,
                          more_chain, more_names, more_fitter):
    """Side-by-side comparison of both HOD models on the same BOSS CMASS data."""
    rp     = np.array(krav_fitter.rp_arr)
    wp_obs = np.array(krav_fitter.wp_obs)
    wp_err = np.array(krav_fitter.wp_err)

    wp_map_krav = np.array(krav_fitter.predict_wp(MAP_PARAMS))
    wp_map_more = np.array(more_fitter.predict_wp(MORE2015_MAP_PARAMS))

    klo, kmed, khi = _mcmc_band(krav_chain, krav_names, krav_fitter)
    mlo, mmed, mhi = _mcmc_band(more_chain, more_names, more_fitter,
                                fixed_params={"alpha_inc": 1.0, "log10m_inc": 13.0})

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    # Kravtsov+2004
    ax.fill_between(rp, klo, khi, color="C0", alpha=0.20)
    ax.loglog(rp, kmed, "--", color="C0", lw=1.2)
    ax.loglog(rp, wp_map_krav, "-", color="C0", lw=2.0,
              label=r"Kravtsov+2004 MAP ($\chi^2/\mathrm{dof}=1.91$)")
    # More+2015
    ax.fill_between(rp, mlo, mhi, color="C1", alpha=0.20)
    ax.loglog(rp, mmed, "--", color="C1", lw=1.2)
    ax.loglog(rp, wp_map_more, "-", color="C1", lw=2.0,
              label=r"More+2015 MAP ($\chi^2/\mathrm{dof}=1.54$)")
    # Data
    ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=5, color="k", zorder=5,
                label="BOSS CMASS data (White+2014)")
    ax.set_ylabel(r"$w_p(r_p)\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_title("Model comparison — BOSS CMASS z=0.52 (WMAP7)", fontsize=11)

    ax2 = axes[1]
    # Residuals relative to data
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.fill_between(rp, wp_map_krav / wp_obs - 1, 0, color="C0", alpha=0.15)
    ax2.fill_between(rp, wp_map_more / wp_obs - 1, 0, color="C1", alpha=0.15)
    ax2.plot(rp, wp_map_krav / wp_obs - 1, "-", color="C0", lw=1.8,
             label="Kravtsov+2004")
    ax2.plot(rp, wp_map_more / wp_obs - 1, "-", color="C1", lw=1.8,
             label="More+2015")
    ax2.axhline( 0.1, color="gray", lw=0.5, ls=":")
    ax2.axhline(-0.1, color="gray", lw=0.5, ls=":")
    ax2.set_xlabel(r"$r_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax2.set_ylabel(r"model/data $- 1$", fontsize=11)
    ax2.set_ylim(-0.55, 0.55)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "comparison_kravtsov2004_vs_more2015_wp.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from hod_mod.fitting import load_config, WpFitter

    print("Loading Kravtsov+2004 chain …")
    chain, names = load_chain()
    print(f"  shape: {chain.shape}, params: {names}")

    print("Building Kravtsov+2004 WpFitter …")
    config = load_config(CONFIG_FILE)
    fitter = WpFitter(config)

    print("Loading More+2015 chain …")
    d_more = np.load(MORE2015_CHAIN_FILE)
    more_chain = d_more["flatchain"]
    more_names = list(d_more["param_names"])
    print(f"  shape: {more_chain.shape}, params: {more_names}")

    print("Building More+2015 WpFitter …")
    more_config = load_config(MORE2015_CONFIG_FILE)
    more_fitter = WpFitter(more_config)

    print("Generating corner plot …")
    make_corner(chain, names)

    print(f"Generating wp+MCMC figure ({N_SAMPLES_WP} samples) …")
    make_wp_mcmc(chain, names, fitter)

    print("Generating model comparison figure …")
    make_model_comparison(chain, names, fitter, more_chain, more_names, more_fitter)

    print("Done.")


if __name__ == "__main__":
    main()
