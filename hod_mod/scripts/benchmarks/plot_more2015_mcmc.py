#!/usr/bin/env python
"""Generate MCMC figures for the more2015 benchmark.

Produces three files in results/benchmarks/more2015_cmass/:
  - corner_more2015.png                  : posterior corner plot with MAP + published values
  - benchmark_more2015_wp_mcmc.png       : wp with MCMC band and published model
  - comparison_more2015_published_wp.png : focused residual comparison vs published params

Usage
-----
    python hod_mod/scripts/benchmarks/plot_more2015_mcmc.py
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
# Paths and constants
# ---------------------------------------------------------------------------

CHAIN_FILE  = os.path.join(_REPO_ROOT, "results/more2015_cmass/flatchain.npz")
CONFIG_FILE = os.path.join(_REPO_ROOT, "configs/benchmarks/benchmark_more2015.yml")
OUT_DIR     = os.path.join(_REPO_ROOT, "results/benchmarks/more2015_cmass")

FIXED_PARAMS = {"alpha_inc": 1.0, "log10m_inc": 13.0}

MAP_PARAMS = {
    "log10mmin":  12.891854619247049,
    "sigma_logm": 0.050000207755272114,
    "log10m1":    13.683177504021286,
    "alpha":      1.4153494635487072,
    "kappa":      0.565173084427479,
    **FIXED_PARAMS,
}

PUBLISHED_PARAMS = {
    "log10mmin":  13.03,
    "sigma_logm": 0.38,
    "log10m1":    13.80,
    "alpha":      1.17,
    "kappa":      0.51,
    **FIXED_PARAMS,
}

PUBLISHED_ERRORS = {
    "log10mmin":  0.02,
    "sigma_logm": 0.05,
    "log10m1":    0.05,
    "alpha":      0.10,
    "kappa":      0.20,
}

FREE_PARAMS = ["log10mmin", "sigma_logm", "log10m1", "alpha", "kappa"]

PARAM_LABELS = {
    "log10mmin":  r"$\log_{10}M_{\min}$",
    "sigma_logm": r"$\sigma_{\log M}$",
    "log10m1":    r"$\log_{10}M_1$",
    "alpha":      r"$\alpha$",
    "kappa":      r"$\kappa$",
}

N_SAMPLES = 400
RNG_SEED  = 42


# ---------------------------------------------------------------------------
# Load chain
# ---------------------------------------------------------------------------

def load_chain():
    d = np.load(CHAIN_FILE)
    return d["flatchain"], list(d["param_names"])


# ---------------------------------------------------------------------------
# MCMC band helper
# ---------------------------------------------------------------------------

def mcmc_band(chain, names, fitter, n_samples=N_SAMPLES):
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
    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

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
            map_j  = MAP_PARAMS[pname_j]
            map_i  = MAP_PARAMS[pname_i]
            pub_j  = PUBLISHED_PARAMS[pname_j]
            pub_i  = PUBLISHED_PARAMS[pname_i]

            if i == j:
                ax.hist(xi, bins=50, color="C0", alpha=0.7, density=True)
                ax.axvline(map_j, color="C3",  lw=1.5, ls="--", label="MAP")
                ax.axvline(pub_j, color="C1",  lw=1.5, ls="-",  label="Published")
                p16, p84 = np.percentile(xi, [16, 84])
                ax.axvline(p16, color="C0", lw=0.8, ls=":")
                ax.axvline(p84, color="C0", lw=0.8, ls=":")
                ax.set_yticks([])
            else:
                _hist2d(ax, xi, yi)
                ax.axvline(map_j, color="C3", lw=0.9, ls="--", alpha=0.8)
                ax.axhline(map_i, color="C3", lw=0.9, ls="--", alpha=0.8)
                ax.plot(map_j, map_i, "C3+", ms=8, mew=1.5, zorder=5)
                ax.axvline(pub_j, color="C1", lw=0.9, ls="-", alpha=0.8)
                ax.axhline(pub_i, color="C1", lw=0.9, ls="-", alpha=0.8)
                ax.plot(pub_j, pub_i, "C1*", ms=9, zorder=5)

            if i == n - 1:
                ax.set_xlabel(PARAM_LABELS[pname_j], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(PARAM_LABELS[pname_i], fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    # Legend via proxy artists on the first diagonal panel
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], color="C3", ls="--", lw=1.5, label="MAP (degenerate valley)"),
        Line2D([0], [0], color="C1", ls="-",  lw=1.5, label="Published More+2015"),
        Line2D([0], [0], color="C0", ls="-",  lw=4,   alpha=0.6, label=r"Posterior $1\sigma/2\sigma$"),
    ]
    axes[0, 0].legend(handles=proxy, fontsize=7, loc="upper left",
                      bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    fig.suptitle(
        "More+2015 MoreHODModel — BOSS CMASS posterior\n"
        r"(dashed red = MAP, solid orange = published More+2015, shaded = $1\sigma$/$2\sigma$)",
        fontsize=11, y=0.995,
    )
    out = os.path.join(OUT_DIR, "corner_more2015.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# wp with MCMC band (+ MAP + published)
# ---------------------------------------------------------------------------

def make_wp_mcmc(chain, names, fitter):
    wp_lo, wp_med, wp_hi = mcmc_band(chain, names, fitter)

    rp     = np.array(fitter.rp_arr)
    wp_obs = np.array(fitter.wp_obs)
    wp_err = np.array(fitter.wp_err)
    wp_map = np.array(fitter.predict_wp(MAP_PARAMS))
    wp_pub = np.array(fitter.predict_wp(PUBLISHED_PARAMS))

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.fill_between(rp, wp_lo, wp_hi, color="C0", alpha=0.25,
                    label=r"MCMC $1\sigma$ band")
    ax.loglog(rp, wp_med, "--", color="C0", lw=1.4, label="MCMC median")
    ax.loglog(rp, wp_map, "-",  color="C0", lw=2.0,
              label=r"MAP ($\chi^2/\mathrm{dof}=1.54$, degenerate)")
    ax.loglog(rp, wp_pub, "-",  color="C1", lw=2.0,
              label="Published More+2015 params")
    ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=5, color="k",
                label="BOSS CMASS data (White+2014)")
    ax.set_ylabel(r"$w_p(r_p)\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_title("More+2015 MoreHODModel — BOSS CMASS z=0.52", fontsize=11)

    ax2 = axes[1]
    ax2.fill_between(rp, wp_obs / wp_hi - 1, wp_obs / wp_lo - 1,
                     color="C0", alpha=0.25)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.errorbar(rp, wp_obs / wp_map - 1, yerr=wp_err / wp_map,
                 fmt="o", ms=5, color="C0", label="data/MAP − 1")
    ax2.plot(rp, wp_obs / wp_pub - 1, "-", color="C1", lw=1.6,
             label="data/Published − 1")
    ax2.axhline( 0.1, color="gray", lw=0.5, ls=":")
    ax2.axhline(-0.1, color="gray", lw=0.5, ls=":")
    ax2.set_xlabel(r"$r_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax2.set_ylabel(r"data/model $- 1$", fontsize=11)
    ax2.set_ylim(-0.55, 0.55)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "benchmark_more2015_wp_mcmc.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Focused comparison with paper: MCMC median vs published
# ---------------------------------------------------------------------------

def make_comparison_published(chain, names, fitter):
    wp_lo, wp_med, wp_hi = mcmc_band(chain, names, fitter)

    rp     = np.array(fitter.rp_arr)
    wp_obs = np.array(fitter.wp_obs)
    wp_err = np.array(fitter.wp_err)
    wp_pub = np.array(fitter.predict_wp(PUBLISHED_PARAMS))

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.fill_between(rp, wp_lo, wp_hi, color="C0", alpha=0.25,
                    label=r"Our MCMC $1\sigma$ band")
    ax.loglog(rp, wp_med, "-", color="C0", lw=2.0, label="Our MCMC median")
    ax.loglog(rp, wp_pub, "-", color="C1", lw=2.0, label="Published More+2015")
    ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=5, color="k",
                label="BOSS CMASS data (White+2014)")
    ax.set_ylabel(r"$w_p(r_p)\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_title(
        "More+2015 HOD: reproduction vs published best-fit\n"
        r"BOSS CMASS z=0.52, WMAP7",
        fontsize=11,
    )

    ax2 = axes[1]
    ax2.fill_between(rp, wp_obs / wp_hi - 1, wp_obs / wp_lo - 1,
                     color="C0", alpha=0.25)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.errorbar(rp, wp_obs / wp_med - 1, yerr=wp_err / wp_med,
                 fmt="o", ms=5, color="C0", label="data/MCMC median − 1")
    ax2.plot(rp, wp_obs / wp_pub - 1, "-", color="C1", lw=1.6,
             label="data/Published − 1")
    ax2.axhline( 0.1, color="gray", lw=0.5, ls=":")
    ax2.axhline(-0.1, color="gray", lw=0.5, ls=":")
    ax2.set_xlabel(r"$r_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax2.set_ylabel(r"data/model $- 1$", fontsize=11)
    ax2.set_ylim(-0.55, 0.55)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "comparison_more2015_published_wp.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from hod_mod.fitting import load_config, WpFitter

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading More+2015 chain …")
    chain, names = load_chain()
    print(f"  shape: {chain.shape}, params: {names}")

    print("Building WpFitter …")
    config = load_config(CONFIG_FILE)
    fitter = WpFitter(config)

    print("Generating corner plot …")
    make_corner(chain, names)

    print(f"Generating wp+MCMC figure ({N_SAMPLES} samples) …")
    make_wp_mcmc(chain, names, fitter)

    print("Generating published-comparison figure …")
    make_comparison_published(chain, names, fitter)

    print("Done.")


if __name__ == "__main__":
    main()
