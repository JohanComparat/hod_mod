#!/usr/bin/env python
"""iHOD vs cHOD comparison figure for the Zu & Mandelbaum 2015 benchmark.

Produces:
  results/benchmarks/zumandelbaum2015_sdss/comparison_ihod_chod_wp_ds.png

2×2 layout (wp row, ΔΣ row; main panel + residual panel each):
  Top-left:    wp(rp)  data + iHOD + cHOD
  Top-right:   wp residuals
  Bottom-left: ΔΣ(R)  data + iHOD + cHOD
  Bottom-right: ΔΣ residuals

Usage
-----
    python hod_mod/scripts/benchmarks/plot_zumandelbaum2015_comparison.py
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

CONFIG_FILE = os.path.join(_REPO_ROOT, "configs/benchmarks/benchmark_zumandelbaum2015.yml")
OUT_DIR     = os.path.join(_REPO_ROOT, "results/benchmarks/zumandelbaum2015_sdss")

# Published best-fit parameters — Table 2 of ZM15 (arXiv:1505.02781)
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
    "log10m_star_thresh": 10.2,
    "beta_sat":  0.90,
    "bcut":      0.86,
    "beta_cut":  0.41,
    "alpha_sat": 1.00,
}

# cHOD: Table 2 SHMR params, satellite fixed at iHOD best-fit values
CHOD_PARAMS = {
    **IHOD_PARAMS,
    "lg_m1h":    12.32,
    "lg_m0star": 10.47,
    "beta":      0.54,
    "delta":     0.42,
    "gamma":     1.05,
}


def load_fitter():
    from hod_mod.fitting import load_config, JointFitter
    config = load_config(CONFIG_FILE)
    fitter = JointFitter(config)
    return fitter


def make_comparison(fitter):
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

    chi2_wp_ihod = float(np.sum(((wp_ihod - wp_obs) / wp_err) ** 2))
    chi2_ds_ihod = float(np.sum(((ds_ihod - ds_obs) / ds_err) ** 2))
    chi2_wp_chod = float(np.sum(((wp_chod - wp_obs) / wp_err) ** 2))
    chi2_ds_chod = float(np.sum(((ds_chod - ds_obs) / ds_err) ** 2))
    n_wp, n_ds = len(wp_obs), len(ds_obs)

    print(f"iHOD  chi2_wp={chi2_wp_ihod:.2f}/{n_wp-9}  chi2_ds={chi2_ds_ihod:.2f}/{n_ds-9}  "
          f"total={chi2_wp_ihod+chi2_ds_ihod:.2f}/{n_wp+n_ds-9}")
    print(f"cHOD  chi2_wp={chi2_wp_chod:.2f}/{n_wp-9}  chi2_ds={chi2_ds_chod:.2f}/{n_ds-9}  "
          f"total={chi2_wp_chod+chi2_ds_chod:.2f}/{n_wp+n_ds-9}")

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.28},
    )

    # ---- wp main panel -------------------------------------------------------
    ax = axes[0, 0]
    ax.fill_between([], [], color="none")   # dummy
    ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=5, color="k",
                label="SDSS DR7 data (ZM15 Fig. 6)", zorder=4)
    ax.loglog(rp, wp_ihod, "-",  color="C0", lw=2.2,
              label=f"iHOD (Table 2)  "
                    r"$\chi^2_{\rm wp}$" + f"={chi2_wp_ihod:.1f}")
    ax.loglog(rp, wp_chod, "--", color="C1", lw=2.2,
              label=f"cHOD (Table 2)  "
                    r"$\chi^2_{\rm wp}$" + f"={chi2_wp_chod:.1f}")
    ax.set_ylabel(r"$w_p(r_p)\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax.set_title(
        r"Zu & Mandelbaum 2015 — SDSS DR7, $\log_{10}(M_*/h^{-2}M_\odot)>10.2$",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xticklabels([])

    # ---- wp residuals --------------------------------------------------------
    ax2 = axes[1, 0]
    ax2.errorbar(rp, (wp_obs - wp_ihod) / wp_err, yerr=np.ones_like(rp),
                 fmt="o", ms=5, color="C0", label="(data−iHOD)/err")
    ax2.plot(rp, (wp_obs - wp_chod) / wp_err, "s--", ms=5, color="C1",
             label="(data−cHOD)/err")
    ax2.axhline(0,    color="k",    lw=0.9, ls="--")
    ax2.axhline( 1.0, color="gray", lw=0.6, ls=":")
    ax2.axhline(-1.0, color="gray", lw=0.6, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$r_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax2.set_ylabel(r"residual / $\sigma$", fontsize=11)
    ax2.set_ylim(-3, 3)
    ax2.legend(fontsize=8, loc="upper right")

    # ---- ΔΣ main panel -------------------------------------------------------
    ax3 = axes[0, 1]
    ax3.errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=5, color="k",
                 label="SDSS DR7 lensing data", zorder=4)
    ax3.loglog(R, ds_ihod, "-",  color="C0", lw=2.2,
               label=f"iHOD (Table 2)  "
                     r"$\chi^2_{\Delta\Sigma}$" + f"={chi2_ds_ihod:.1f}")
    ax3.loglog(R, ds_chod, "--", color="C1", lw=2.2,
               label=f"cHOD (Table 2)  "
                     r"$\chi^2_{\Delta\Sigma}$" + f"={chi2_ds_chod:.1f}")
    ax3.set_ylabel(r"$\Delta\Sigma(R)\ [M_\odot\,h\,\mathrm{pc}^{-2}]$", fontsize=12)
    ax3.set_title(r"$z_{\rm eff}=0.1$, Mandelbaum+2006 lensing", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.set_xticklabels([])

    # ---- ΔΣ residuals --------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.errorbar(R, (ds_obs - ds_ihod) / ds_err, yerr=np.ones_like(R),
                 fmt="s", ms=5, color="C0", label=r"(data−iHOD)/$\sigma$")
    ax4.plot(R, (ds_obs - ds_chod) / ds_err, "^--", ms=5, color="C1",
             label=r"(data−cHOD)/$\sigma$")
    ax4.axhline(0,    color="k",    lw=0.9, ls="--")
    ax4.axhline( 1.0, color="gray", lw=0.6, ls=":")
    ax4.axhline(-1.0, color="gray", lw=0.6, ls=":")
    ax4.set_xscale("log")
    ax4.set_xlabel(r"$R\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax4.set_ylabel(r"residual / $\sigma$", fontsize=11)
    ax4.set_ylim(-3, 3)
    ax4.legend(fontsize=8, loc="upper right")

    out = os.path.join(OUT_DIR, "comparison_ihod_chod_wp_ds.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading JointFitter …")
    fitter = load_fitter()
    print(f"  wp bins: {len(fitter.rp_arr)}  (rp=[{fitter.rp_arr[0]:.3f}, {fitter.rp_arr[-1]:.3f}] h-1 Mpc)")
    print(f"  ΔΣ bins: {len(fitter.R_arr)}   (R=[{fitter.R_arr[0]:.3f}, {fitter.R_arr[-1]:.3f}] h-1 Mpc)")
    print("\nGenerating iHOD vs cHOD comparison figure …")
    make_comparison(fitter)
