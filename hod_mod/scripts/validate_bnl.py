"""Validation figures for the BNL (Beyond-Linear Bias) implementation.

Mead & Verde (2021), arXiv:2011.08858
https://github.com/alexander-mead/BNL

Produces four panels:
1. β^NL diagonal (nu1=nu2) vs k for all 8 mass bins
2. 2-halo boost ratio P_gg^{2h,BNL} / P_gg^{2h,lin} - 1 vs k
3. Full P_gg(k): with vs without BNL + ratio
4. Full P_gm(k): with vs without BNL + ratio

Usage
-----
    cd $HOD_MOD_REPO
    python -m hod_mod.scripts.validate_bnl
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.core import BeyondLinearBiasMead21
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Cosmology and HOD setup
# ---------------------------------------------------------------------------
_THETA = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "flat": True,
    "H0": _THETA["h"] * 100.0,
    "Om0": _THETA["Omega_m"],
    "Ob0": _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns": _THETA["n_s"],
}
_Z = 0.3
_HOD_PARAMS = MoreHODModel.default_params()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_predictors(n_k=256):
    pk_lin = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod = MoreHODModel(hmf, hmf.bias)
    bnl = BeyondLinearBiasMead21()

    pred_lin = FullHaloModelPrediction(pk_lin, hod, hp, n_k=n_k)
    pred_bnl = FullHaloModelPrediction(pk_lin, hod, hp, n_k=n_k, bnl_model=bnl)
    return bnl, pred_lin, pred_bnl


# ---------------------------------------------------------------------------
# Figure 1: β^NL diagonal (nu1 = nu2) for each of the 8 mass bins
# ---------------------------------------------------------------------------

def fig_beta_nl_diagonal(bnl):
    k_plot = np.logspace(np.log10(bnl._k_ref[0]) - 0.05,
                          np.log10(bnl._k_ref[-1]) + 0.05, 200)
    beta = bnl.beta_nl(k_plot, bnl._nu_ref, bnl._nu_ref)   # (Nk, 8, 8)
    diag = np.array([beta[:, i, i] for i in range(8)])       # (8, Nk)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 8))

    for i in range(8):
        label = rf"$\nu={bnl._nu_ref[i]:.2f}$  ($\log_{{10}}M_i$)"
        ax.plot(k_plot, diag[i], color=colors[i], label=label)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    # Mark the tabulated k range
    ax.axvline(bnl._k_ref[0],  color="grey", lw=0.6, ls=":")
    ax.axvline(bnl._k_ref[-1], color="grey", lw=0.6, ls=":")

    ax.set_xscale("log")
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$\beta^\mathrm{NL}(k,\,\nu,\,\nu)$")
    ax.set_title(r"Beyond-linear bias $\beta^\mathrm{NL}$ diagonal — Mead \& Verde (2021)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(1e-3, 2.0)

    out = _FIG_DIR / "bnl_beta_nl_diagonal.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: 2-halo boost ratio
# ---------------------------------------------------------------------------

def fig_2h_boost(pred_lin, pred_bnl):
    t_lin = pred_lin._pk_tables_full(_Z, _THETA, _HOD_PARAMS)
    t_bnl = pred_bnl._pk_tables_full(_Z, _THETA, _HOD_PARAMS)

    k = np.exp(np.array(t_lin["log_k"]))
    p2h_lin = np.exp(np.array(t_lin["log_pgg_2h"]))
    p2h_bnl = np.exp(np.array(t_bnl["log_pgg_2h"]))
    ratio = p2h_bnl / p2h_lin - 1.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, ratio, color="C0", lw=1.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 5.0)
    ax.set_ylim(-0.5, 3.0)
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_{gg,\mathrm{BNL}}^{2h} / P_{gg,\mathrm{lin}}^{2h} - 1$")
    ax.set_title(rf"2-halo BNL boost — $z={_Z}$, More+2015 HOD")

    out = _FIG_DIR / "bnl_2h_boost_ratio.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Full P_gg comparison
# ---------------------------------------------------------------------------

def fig_pgg_comparison(pred_lin, pred_bnl):
    t_lin = pred_lin._pk_tables_full(_Z, _THETA, _HOD_PARAMS)
    t_bnl = pred_bnl._pk_tables_full(_Z, _THETA, _HOD_PARAMS)

    k = np.exp(np.array(t_lin["log_k"]))
    pgg_lin = np.exp(np.array(t_lin["log_pgg"]))
    pgg_bnl = np.exp(np.array(t_bnl["log_pgg"]))
    p1h     = np.exp(np.array(t_lin["log_pgg_1h"]))
    p2h_lin = np.exp(np.array(t_lin["log_pgg_2h"]))
    p2h_bnl = np.exp(np.array(t_bnl["log_pgg_2h"]))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_r = axes

    ax.loglog(k, pgg_lin, color="C0", lw=2.0, label=r"$P_{gg}$ linear bias")
    ax.loglog(k, pgg_bnl, color="C1", lw=2.0, ls="--", label=r"$P_{gg}$ + BNL")
    ax.loglog(k, p1h,     color="grey", lw=1.0, ls=":", label="1-halo")
    ax.loglog(k, p2h_lin, color="C0", lw=1.0, alpha=0.5, label="2-halo (linear)")
    ax.loglog(k, p2h_bnl, color="C1", lw=1.0, alpha=0.5, ls="--", label="2-halo (BNL)")
    ax.set_ylabel(r"$P_{gg}(k)\ [(h^{-1}\mathrm{Mpc})^3]$")
    ax.set_title(rf"Galaxy power spectrum — $z={_Z}$, More+2015 HOD")
    ax.legend(fontsize=8)
    ax.set_xlim(1e-3, 5.0)

    ratio = pgg_bnl / pgg_lin
    ax_r.semilogx(k, ratio, color="C1", lw=1.8)
    ax_r.axhline(1, color="k", lw=0.8, ls="--")
    ax_r.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax_r.set_ylabel(r"$P_{gg}^\mathrm{BNL} / P_{gg}^\mathrm{lin}$")
    ax_r.set_xlim(1e-3, 5.0)

    fig.tight_layout()
    out = _FIG_DIR / "bnl_pgg_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Full P_gm comparison
# ---------------------------------------------------------------------------

def fig_pgm_comparison(pred_lin, pred_bnl):
    t_lin = pred_lin._pk_tables_full(_Z, _THETA, _HOD_PARAMS)
    t_bnl = pred_bnl._pk_tables_full(_Z, _THETA, _HOD_PARAMS)

    k = np.exp(np.array(t_lin["log_k"]))
    pgm_lin = np.exp(np.array(t_lin["log_pgm"]))
    pgm_bnl = np.exp(np.array(t_bnl["log_pgm"]))
    p1h     = np.exp(np.array(t_lin["log_pgm_1h"]))
    p2h_lin = np.exp(np.array(t_lin["log_pgm_2h"]))
    p2h_bnl = np.exp(np.array(t_bnl["log_pgm_2h"]))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_r = axes

    ax.loglog(k, pgm_lin, color="C2", lw=2.0, label=r"$P_{gm}$ linear bias")
    ax.loglog(k, pgm_bnl, color="C3", lw=2.0, ls="--", label=r"$P_{gm}$ + BNL")
    ax.loglog(k, p1h,     color="grey", lw=1.0, ls=":", label="1-halo")
    ax.loglog(k, p2h_lin, color="C2", lw=1.0, alpha=0.5, label="2-halo (linear)")
    ax.loglog(k, p2h_bnl, color="C3", lw=1.0, alpha=0.5, ls="--", label="2-halo (BNL)")
    ax.set_ylabel(r"$P_{gm}(k)\ [(h^{-1}\mathrm{Mpc})^3]$")
    ax.set_title(rf"Galaxy-matter cross power — $z={_Z}$, More+2015 HOD")
    ax.legend(fontsize=8)
    ax.set_xlim(1e-3, 5.0)

    ratio = pgm_bnl / pgm_lin
    ax_r.semilogx(k, ratio, color="C3", lw=1.8)
    ax_r.axhline(1, color="k", lw=0.8, ls="--")
    ax_r.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax_r.set_ylabel(r"$P_{gm}^\mathrm{BNL} / P_{gm}^\mathrm{lin}$")
    ax_r.set_xlim(1e-3, 5.0)

    fig.tight_layout()
    out = _FIG_DIR / "bnl_pgm_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building halo model predictors ...")
    bnl, pred_lin, pred_bnl = _build_predictors(n_k=256)

    print("Figure 1: β^NL diagonal ...")
    fig_beta_nl_diagonal(bnl)

    print("Figure 2: 2-halo boost ratio ...")
    fig_2h_boost(pred_lin, pred_bnl)

    print("Figure 3: P_gg comparison ...")
    fig_pgg_comparison(pred_lin, pred_bnl)

    print("Figure 4: P_gm comparison ...")
    fig_pgm_comparison(pred_lin, pred_bnl)

    print(f"\nAll figures saved to {_FIG_DIR}/")
