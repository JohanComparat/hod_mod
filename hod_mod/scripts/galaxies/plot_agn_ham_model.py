"""Verification plots for HamAGNModel.

Generates seven figures:
  fig_agn_ham_01_mapping.png     — HAM hard luminosity vs halo/stellar mass
  fig_agn_ham_02_hard_xlf.png    — Predicted hard XLF vs Aird+2015 / Ueda+2014
  fig_agn_ham_03_soft_xlf.png    — Predicted soft XLF vs Hasinger+2005 data points
  fig_agn_ham_04_obscuration.png — Type fractions vs log10(L_X)
  fig_agn_ham_06_duty_cycle.png  — AGN duty cycle f(>LX|M*) vs M* with G17
  fig_agn_ham_07_lsar.png        — Specific accretion rate distribution with G17/A17
  fig_agn_ham_08_hgsmf.png       — AGN host galaxy SMF vs Bongiorno+2016

Run with:
    JAX_PLATFORMS=cpu python3 -m hod_mod.scripts.galaxies.plot_agn_ham_model
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.special import erfc

from hod_mod.agn.ham import (
    HamAGNModel,
    _aird15_lade_np,
    _ueda14_ldde_np,
    obscured_fraction,
    _f_compton_thick,
    _duty_cycle_interp,
    _HARD_TO_SOFT_RATIO,
)


_OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "_images")
_AGN_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agn")
os.makedirs(_OUT_DIR, exist_ok=True)


def _lit(fname: str) -> str | None:
    """Return path to a bundled literature data file, or env-var fallback."""
    p = os.path.join(_AGN_DATA, fname)
    if os.path.isfile(p):
        return p
    env = os.environ.get("GIT_STMOD_DATA", "")
    fb  = os.path.join(env, "data/validation/validation_AGN/literature_data", fname)
    return fb if os.path.isfile(fb) else None


def _model_ingredients(model, z):
    """Return (log10mh, log10mstar, lx_ham, dndlogmh, f_dc, sigma_lx) at redshift z."""
    from hod_mod.connection.hod import _mstar_from_mh_zu15
    log10mh  = model._log10mh_grid
    mh_arr   = 10.0 ** log10mh
    theta    = model._theta_cosmo
    dndm     = np.array(model._hmf.dndm(jnp.array(mh_arr), z, theta))
    dndlogmh = dndm * mh_arr * np.log(10.0)
    p        = model._zu15_params
    log10mstar = np.array(_mstar_from_mh_zu15(
        jnp.array(log10mh),
        p["lg_m1h"], p["lg_m0star"], p["beta"], p["delta"], p["gamma"],
    ))
    lx_ham   = model.ham_log10lx_hard(log10mh, z)

    # Exclude halos clamped to the XLF floor — they have no genuine HAM assignment
    # and their large HMF weight (abundant low-mass halos) contaminates all statistics.
    valid      = lx_ham > 41.05
    log10mh    = log10mh[valid]
    log10mstar = log10mstar[valid]
    lx_ham     = lx_ham[valid]
    dndlogmh   = dndlogmh[valid]

    f_dc     = model._duty_cycle if model._duty_cycle is not None else _duty_cycle_interp(z)
    return log10mh, log10mstar, lx_ham, dndlogmh, float(f_dc), float(model._scatter_lx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xlf_from_model(
    model: HamAGNModel,
    log10lx_grid: np.ndarray,
    z: float,
    band: str = "hard",
) -> np.ndarray:
    """Predict XLF at redshift z by weighting the HMF with the HAM mapping.

    band = 'hard' returns predicted hard XLF.
    band = 'soft' additionally applies the K-correction fraction.
    """
    from hod_mod.core.power_spectrum import LinearPowerSpectrum

    theta   = model._theta_cosmo
    log10mh = model._log10mh_grid
    mh_arr  = 10.0 ** log10mh
    dlogmh  = np.gradient(log10mh)

    dndm_arr = np.array(
        model._hmf.dndm(jnp.array(mh_arr), z, theta)
    )
    dndlogmh = dndm_arr * mh_arr * np.log(10.0)   # h³/Mpc³/dex

    if model._duty_cycle is not None:
        f_dc = model._duty_cycle
    else:
        f_dc = _duty_cycle_interp(z)

    # HAM hard luminosity for each halo
    lx_hard = model.ham_log10lx_hard(log10mh, z)   # (n_mh,)

    weights = dndlogmh * f_dc                         # (n_mh,)

    if band == "soft":
        # Bin by soft luminosity: lx_soft = lx_hard + log10(k_eff)
        k_eff = model._mean_k_eff(lx_hard, z)         # (n_mh,)
        lx_bin = lx_hard + np.log10(np.maximum(k_eff, 1e-30))  # log10(L_soft)
    else:
        lx_bin = lx_hard

    # Bin the weighted halos into the LX grid
    bin_edges = np.concatenate([[log10lx_grid[0] - 0.5*(log10lx_grid[1]-log10lx_grid[0])],
                                 0.5*(log10lx_grid[:-1] + log10lx_grid[1:]),
                                 [log10lx_grid[-1] + 0.5*(log10lx_grid[-1]-log10lx_grid[-2])]])
    bin_width = np.diff(bin_edges)

    xlf = np.zeros(len(log10lx_grid))
    for j, (lx_lo, lx_hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (lx_bin >= lx_lo) & (lx_bin < lx_hi)
        if mask.any():
            xlf[j] = np.sum(weights[mask] * dlogmh[mask]) / bin_width[j]

    return xlf


# ---------------------------------------------------------------------------
# Figure 1 — HAM mapping: log10(L_X^hard) vs log10(M_h) and log10(M_*)
# ---------------------------------------------------------------------------

def make_fig01_ham_mapping(model: HamAGNModel) -> None:
    log10mh = model._log10mh_grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["C0", "C1", "C2"]
    zs     = [0.1, 0.5, 1.0]

    for z, col in zip(zs, colors):
        lbl      = f"z={z}"
        lx_hard  = model.ham_log10lx_hard(log10mh, z)

        axes[0].plot(log10mh, lx_hard, color=col, label=lbl)

        # Also get M_* from ZuMandelbaum SHMR
        from hod_mod.connection.hod import _mstar_from_mh_zu15
        p = model._zu15_params
        log10mstar = np.array(
            _mstar_from_mh_zu15(
                jnp.array(log10mh),
                p["lg_m1h"], p["lg_m0star"], p["beta"], p["delta"], p["gamma"],
            )
        )
        axes[1].plot(log10mstar, lx_hard, color=col, label=lbl)

    for ax, xlabel in zip(axes, [r"$\log_{10}(M_h\,[M_\odot/h])$",
                                   r"$\log_{10}(M_*\,[M_\odot/h])$"]):
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r"$\log_{10}(L_X^{2-10\,\mathrm{keV}}\,[\mathrm{erg/s}])$",
                      fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(40, 47)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"HAM mapping (xlf={model._xlf_name})", fontsize=12)
    axes[1].set_title("via ZuMandelbaum15 SHMR", fontsize=12)

    fig.tight_layout()
    path = os.path.join(_OUT_DIR, "fig_agn_ham_01_mapping.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Hard XLF check: model vs Aird+2015 / Ueda+2014
# ---------------------------------------------------------------------------

def make_fig02_hard_xlf(model: HamAGNModel) -> None:
    log10lx = np.linspace(41.0, 46.5, 60)
    fig, ax  = plt.subplots(figsize=(8, 6))
    colors   = ["C0", "C1", "C2"]
    zs       = [0.1, 0.5, 1.0]

    for z, col in zip(zs, colors):
        xlf_pred = _xlf_from_model(model, log10lx, z, band="hard")
        xlf_a15  = _aird15_lade_np(log10lx, z)
        xlf_u14  = _ueda14_ldde_np(log10lx, z)
        h_planck = model._theta_cosmo.get("h", 0.6736)
        h_fac    = (0.70 / h_planck) ** 3  # same conversion as precomputation

        lbl = f"z={z}"
        ax.semilogy(log10lx, xlf_pred,          color=col, lw=2,
                    label=lbl + " model (HAM)")
        ax.semilogy(log10lx, xlf_a15 * h_fac,   color=col, lw=1.5, ls="--",
                    label=lbl + " Aird+15")
        ax.semilogy(log10lx, xlf_u14 * h_fac,   color=col, lw=1.0, ls=":",
                    label=lbl + " Ueda+14")

    ax.set_xlabel(r"$\log_{10}(L_X^{2-10\,\mathrm{keV}}\,[\mathrm{erg/s}])$",
                  fontsize=12)
    ax.set_ylabel(r"$\Phi\,[(h^{-1}\mathrm{Mpc})^{-3}\,\mathrm{dex}^{-1}]$",
                  fontsize=12)
    ax.set_title(f"Hard XLF — HAM model vs references (xlf={model._xlf_name})",
                 fontsize=12)
    ax.set_ylim(1e-9, 1e-4)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.03,
            "Model solid; Aird+15 dashed; Ueda+14 dotted\n"
            "Hard XLF should match XLF used for HAM by construction",
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, "fig_agn_ham_02_hard_xlf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Soft XLF check: model vs Hasinger+2005
# ---------------------------------------------------------------------------

def make_fig03_soft_xlf(model: HamAGNModel) -> None:
    # Grid in L_soft (0.5-2 keV) — same range as Hasinger+2005
    log10lx_soft = np.linspace(41.0, 46.0, 60)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["C0", "C1", "C2"]
    zs      = [0.1, 0.5, 1.0]

    # Load Hasinger+2005 observed data points (z=0.015-0.2, published with h=0.70)
    _H05_PATH = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "data", "agn", "hasinger05_z015-02.ascii",
    )
    H05 = None
    if os.path.isfile(_H05_PATH):
        H05 = np.loadtxt(_H05_PATH, unpack=True)

    for z, col in zip(zs, colors):
        xlf_pred_soft = _xlf_from_model(model, log10lx_soft, z, band="soft")

        ax.semilogy(log10lx_soft, xlf_pred_soft, color=col, lw=2,
                    label=f"z={z} model (HAM soft)")

    # Overplot actual Hasinger+2005 data points (only valid for z≈0.1 curve)
    if H05 is not None:
        h_model = float(model._theta_cosmo.get("h", 0.6736))
        h_ratio = h_model / 0.70
        # Shift Lx bins from h=0.70 to model h: L ∝ d_L^2 ∝ h^{-2}
        x_h05    = (H05[2] + H05[3]) * 0.5 - 2.0 * np.log10(h_ratio)
        xerr     = (H05[3] - H05[2]) / 2.0
        # Shift phi: volume ∝ h^{-3}
        y_h05    = H05[6] * H05[9] * h_ratio ** 3
        y_h05_up = (H05[6] + H05[7]) * H05[9] * h_ratio ** 3
        y_h05_lo = (H05[6] - H05[8]) * H05[9] * h_ratio ** 3
        ax.errorbar(
            x_h05, y_h05,
            yerr=[y_h05 - y_h05_lo, y_h05_up - y_h05],
            xerr=xerr,
            fmt="ko", ms=5, capsize=3, lw=1,
            label="Hasinger+2005 data (z=0.015–0.2)", zorder=5,
        )

    ax.set_xlabel(r"$\log_{10}(L_X^{0.5-2\,\mathrm{keV}}\,[\mathrm{erg/s}])$",
                  fontsize=12)
    ax.set_ylabel(r"$\Phi\,[(h^{-1}\mathrm{Mpc})^{-3}\,\mathrm{dex}^{-1}]$",
                  fontsize=12)
    ax.set_title("Soft (0.5–2 keV) XLF prediction", fontsize=12)
    ax.set_ylim(1e-9, 1e-4)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    kcorr_label = f"K-corr mode: {model._kcorr_mode}"
    ax.text(0.03, 0.03, kcorr_label, transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, "fig_agn_ham_03_soft_xlf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Obscuration model: type fractions vs log10(L_X)
# ---------------------------------------------------------------------------

def make_fig04_obscuration(model: HamAGNModel) -> None:
    log10lx = np.linspace(41.0, 46.0, 200)
    lx_jnp  = jnp.array(log10lx)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors    = ["C0", "C1", "C2"]
    zs        = [0.1, 0.5, 1.0]

    for ax_idx, ax in enumerate(axes):
        for z, col in zip(zs, colors):
            f_obs = np.array(obscured_fraction(lx_jnp, z))
            f_ct  = np.array(_f_compton_thick(lx_jnp, z))
            f_t1  = np.clip(1.0 - f_obs, 0.0, 1.0)
            f_t2  = np.clip(f_obs - f_ct, 0.0, 1.0)
            f_ctc = np.clip(f_ct, 0.0, 1.0)

            if ax_idx == 0:
                ax.plot(log10lx, f_obs, color=col, lw=2, label=f"z={z} f_obs")
                ax.plot(log10lx, f_ct,  color=col, lw=1.5, ls="--", label=f"z={z} f_CT")
            else:
                ax.plot(log10lx, f_t1,  color=col, lw=2,   ls="-",  label=f"z={z} type-1")
                ax.plot(log10lx, f_t2,  color=col, lw=1.5, ls="--", label=f"z={z} type-2")
                ax.plot(log10lx, f_ctc, color=col, lw=1.0, ls=":",  label=f"z={z} CT")

        ax.set_xlabel(r"$\log_{10}(L_X^{2-10\,\mathrm{keV}}\,[\mathrm{erg/s}])$",
                      fontsize=12)
        ax.set_ylabel("Fraction", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Obscuration fractions (Comparat+2019 eqs 4–11)", fontsize=12)
    axes[1].set_title("Type fractions (type-1/2/CT)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, "fig_agn_ham_04_obscuration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 6 — Duty cycle: f_DC(>LX | M*) vs M* with Georgakakis+2017
# ---------------------------------------------------------------------------

def make_fig06_duty_cycle(model: HamAGNModel) -> None:
    """AGN duty cycle f(>LX | M*) vs M* at z = 0.25, 0.75, 1.25."""
    dx = np.log10(0.6777 ** 2)   # h shift for G17 literature masses
    panel_cfg = [
        (0.25, "z025", [41, 42, 43]),
        (0.75, "z075", [41, 42, 43, 44]),
        (1.25, "z125", [42, 43, 44]),
    ]
    thresh_col = {41: "black", 42: "red", 43: "blue", 44: "magenta"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (z, z_str, thresholds) in zip(axes, panel_cfg):
        _, log10mstar, lx_ham, _, f_dc, sigma_lx = _model_ingredients(model, z)

        for lx_t in thresholds:
            col  = thresh_col[lx_t]
            prob = 0.5 * erfc((lx_t - lx_ham) / (np.sqrt(2) * sigma_lx))
            ax.semilogy(log10mstar, f_dc * prob, color=col, lw=2,
                        label=rf"model $L_X>{lx_t}$")

        for lx_t in thresholds:
            col = thresh_col[lx_t]
            p   = _lit(f"duty_cycle_G17_{z_str}_LXhardgt{lx_t}.ascii")
            if p:
                x, y, y_up, y_lo = np.loadtxt(p, unpack=True, comments="#")
                ax.fill_between(x + dx, 10 ** y_lo, 10 ** y_up,
                                alpha=0.3, color=col)

        ax.set_xlabel(r"$\log_{10}(M_*\,[M_\odot/h])$", fontsize=11)
        ax.set_title(f"z = {z}", fontsize=11)
        ax.set_xlim(8.5, 12.5)
        ax.set_ylim(5e-5, 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel(r"$f_{\rm AGN}(M_*,\,>L_X)$", fontsize=11)
    fig.suptitle("AGN duty cycle (G17 shaded bands = literature)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, "fig_agn_ham_06_duty_cycle.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 7 — Specific accretion rate (LSAR) distribution with G17/A17
# ---------------------------------------------------------------------------

def make_fig07_lsar(model: HamAGNModel) -> None:
    """AGN specific accretion rate distribution at z = 0.25, 0.75, 1.25."""
    lsar_grid = np.arange(30.0, 38.0, 0.1)

    # (z, [(fname, color, is_log10_P), ...])
    panel_cfg = [
        (0.25, [("lsar_hist_G17_z025.ascii",            "grey",  True),
                ("lsar_hist_A17_010z050_095M100.ascii",  "green", False),
                ("lsar_hist_A17_010z050_100M105.ascii",  "red",   False)]),
        (0.75, [("lsar_hist_G17_z075.ascii",             "grey",  True),
                ("lsar_hist_A17_050z100_095M100.ascii",  "green", False),
                ("lsar_hist_A17_050z100_100M105.ascii",  "red",   False)]),
        (1.25, [("lsar_hist_G17_z125.ascii",             "grey",  True),
                ("lsar_hist_A17_100z150_095M100.ascii",  "green", False),
                ("lsar_hist_A17_100z150_100M105.ascii",  "red",   False)]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (z, lit_cfg) in zip(axes, panel_cfg):
        _, log10mstar, lx_ham, dndlogmh, f_dc, sigma_lx = _model_ingredients(model, z)

        # Gaussian-kernel-smoothed LSAR distribution
        lsar_mean = lx_ham - log10mstar
        p_lsar    = np.zeros(len(lsar_grid))
        for lsar_i, w in zip(lsar_mean, dndlogmh * f_dc):
            if w > 0:
                p_lsar += w * np.exp(-0.5 * ((lsar_grid - lsar_i) / sigma_lx) ** 2)
        norm = np.trapezoid(p_lsar, lsar_grid)
        if norm > 0:
            p_lsar /= norm
        ax.semilogy(lsar_grid, p_lsar, "k", lw=2.5, label="model", zorder=5)

        # Literature shading
        for fname, col, is_log in lit_cfg:
            p = _lit(fname)
            if p is None:
                continue
            x, y_min, y_max = np.loadtxt(p, unpack=True, comments="#")
            if is_log:
                # G17: values are log10(P); normalise after 10**
                mid = 10 ** (0.5 * (y_min + y_max))
                nrm = np.trapezoid(mid, x)
                y_lo = 10 ** y_min / nrm if nrm > 0 else 10 ** y_min
                y_hi = 10 ** y_max / nrm if nrm > 0 else 10 ** y_max
            else:
                # A17: values are linear P; normalise directly
                mid = 0.5 * (y_min + y_max)
                nrm = np.trapezoid(mid, x)
                y_lo = y_min / nrm if nrm > 0 else y_min
                y_hi = y_max / nrm if nrm > 0 else y_max
            ax.fill_between(x + 34, y_lo, y_hi, alpha=0.4, color=col)

        ax.set_xlabel(r"$\log_{10}(\lambda_{\rm SAR})\,[\rm erg\,s^{-1}\,M_\odot^{-1}]$",
                      fontsize=11)
        ax.set_title(f"z = {z}", fontsize=11)
        ax.set_xlim(30, 37)
        ax.set_ylim(1e-4, 4)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Normalized probability", fontsize=11)
    fig.suptitle(r"AGN specific accretion rate (G17 grey; A17 9.5<M$_*$<10 green; 10<M$_*$<10.5 red)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, "fig_agn_ham_07_lsar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 8 — AGN host galaxy SMF vs Bongiorno+2016
# ---------------------------------------------------------------------------

def make_fig08_hgsmf(model: HamAGNModel) -> None:
    """AGN host galaxy stellar mass function vs Bongiorno+2016."""
    panel_cfg = [
        (0.5,  "03_z_08"),
        (1.0,  "08_z_15"),
        (2.0,  "15_z_25"),
    ]
    lx_thresholds = [
        (43.0, "C2", "-",  r"$>10^{43}$"),
        (43.5, "C2", "--", r"$>10^{43.5}$"),
        (44.0, "C2", ":",  r"$>10^{44}$"),
    ]
    bo16_lx = [("430", "-"), ("435", "--"), ("440", ":"), ("445", "-.")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (z, z_str) in zip(axes, panel_cfg):
        log10mh, log10mstar, lx_ham, dndlogmh, f_dc, sigma_lx = _model_ingredients(model, z)

        # Galaxy SMF via SHMR Jacobian
        dlogms_dlogmh = np.gradient(log10mstar) / np.gradient(log10mh)
        with np.errstate(divide="ignore", invalid="ignore"):
            dndlogmstar = np.where(np.abs(dlogms_dlogmh) > 1e-10,
                                   dndlogmh / np.abs(dlogms_dlogmh), 0.0)
        idx = np.argsort(log10mstar)
        ms_s  = log10mstar[idx]
        smf_s = dndlogmstar[idx]
        ax.semilogy(ms_s, smf_s, "k", lw=2, ls="--", label="all galaxies")

        # AGN HGSMF for each LX threshold
        for lx_t, col, ls, lbl in lx_thresholds:
            prob    = 0.5 * erfc((lx_t - lx_ham) / (np.sqrt(2) * sigma_lx))
            agn_smf = dndlogmstar * f_dc * prob[idx]
            ax.semilogy(ms_s, agn_smf, color=col, ls=ls, lw=2, label=f"AGN {lbl}")

        # Bongiorno+2016 literature curves
        for lx_lbl, ls_lit in bo16_lx:
            p = _lit(f"AGN_HGSMF_BO16_{z_str}_LX_{lx_lbl}.ascii")
            if p:
                d = np.loadtxt(p, unpack=True, delimiter=",")
                mask = d[0] < 12
                ax.semilogy(d[0][mask], 10 ** d[1][mask],
                            ls=ls_lit, color="darkgreen", lw=1.5, alpha=0.8)

        ax.set_xlabel(r"$\log_{10}(M_*\,[M_\odot/h])$", fontsize=11)
        ax.set_title(f"z = {z}  (BO16 {z_str.replace('_z_', '<z<')})", fontsize=11)
        ax.set_xlim(9, 13)
        ax.set_ylim(1e-8, 1e-1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel(r"$\Phi\,[(h^{-1}\mathrm{Mpc})^{-3}\,\mathrm{dex}^{-1}]$", fontsize=11)
    fig.suptitle("AGN host galaxy SMF (dark green = Bongiorno+2016)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, "fig_agn_ham_08_hgsmf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(name)s: %(message)s")

    # Respect optional kcorr_path from CLI
    kcorr_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Building HamAGNModel (aird15) …")
    model_a15 = HamAGNModel(xlf="aird15", kcorr_path=kcorr_path)

    print("Building HamAGNModel (ueda14) …")
    model_u14 = HamAGNModel(xlf="ueda14", kcorr_path=kcorr_path)

    print("\nGenerating figures …")
    make_fig01_ham_mapping(model_a15)
    make_fig02_hard_xlf(model_a15)
    make_fig03_soft_xlf(model_a15)
    make_fig04_obscuration(model_a15)
    make_fig06_duty_cycle(model_a15)
    make_fig07_lsar(model_a15)
    make_fig08_hgsmf(model_a15)
    print("Done.")
