"""Generate AGN model diagnostic figures for the Galaxies documentation.

Produces four figures saved to ``docs/_images/``:

* ``fig_agn_01_shmr.png``
  Girelli+2020 SHMR: M_*/M_h ratio and log10 M_* vs log10 M_h at four
  redshifts (z = 0, 0.14, 0.5, 1.0).  Shaded band shows ±0.2 dex intrinsic
  scatter at z = 0.14.  Vertical ticks mark the pivot mass M_A(z).

* ``fig_agn_02_lx_mhalo.png``
  Mean soft X-ray AGN luminosity <L_X^{0.5-2 keV}> vs log10 M_h at four
  redshifts, including duty cycle and scatter boost.  Dashed lines show the
  pre-duty-cycle luminosity to visualise the f_DC(z) suppression.

* ``fig_agn_03_lx_logmmin.png``
  HOD-weighted mean AGN luminosity <L_X>_HOD vs log10 M_min at z = 0.14,
  0.5, 1.0 (Tinker+2008 HMF; sigma_logm = 0.25).

* ``fig_agn_04_xlf.png``
  Predicted soft X-ray AGN luminosity function (halo model) vs the
  Hasinger+2005 LDDE reference (arXiv:astro-ph/0506118) at z = 0.1, 0.5, 1.0.

Usage::

    cd /home/comparat/software/hod_mod
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.galaxies.plot_agn_model
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.connection.sham import (
    smhm_girelli20,
    _GIRELLI20_NO_SCATTER,
    _GIRELLI20_SCATTER,
)
from hod_mod.agn.xray import (
    XrayAGNModel,
    _DUTY_CYCLE_Z,
    _DUTY_CYCLE_LOG,
    _duty_cycle_at_z,
    _scatter_boost,
    _DEFAULT_SCATTER_LX,
)
from hod_mod.core.power_spectrum import LinearPowerSpectrum, rho_critical_0
from hod_mod.core.halo_mass_function import HaloMassFunction

_HERE    = Path(__file__).parent
_IMG_DIR = _HERE.parents[2] / "docs" / "_images"
_IMG_DIR.mkdir(parents=True, exist_ok=True)

_THETA  = LinearPowerSpectrum.default_cosmology()
_OM     = float(_THETA["Omega_m"])
_RHO_M  = rho_critical_0() * _OM

_LOG10M  = np.linspace(10.0, 16.0, 400)
_M_H     = 10.0 ** _LOG10M

_REDSHIFTS = [0.0, 0.14, 0.5, 1.0]
_COLORS    = ["C0", "C1", "C2", "C3"]


# ── Figure 1: Girelli+2020 SHMR ──────────────────────────────────────────────

def make_fig01_shmr():
    log10m = jnp.asarray(_LOG10M)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for z, col in zip(_REDSHIFTS, _COLORS):
        log10mstar_ns = np.asarray(smhm_girelli20(log10m, z, **_GIRELLI20_NO_SCATTER))
        ratio_ns = 10.0 ** (log10mstar_ns - _LOG10M)

        log10_MA = _GIRELLI20_NO_SCATTER["B"] + z * _GIRELLI20_NO_SCATTER["mu"]
        lbl = rf"$z={z:.2f}$  (Table 3)"

        ax1.semilogy(_M_H, ratio_ns, color=col, lw=2, label=lbl)
        ax2.plot(_LOG10M, log10mstar_ns, color=col, lw=2)

        # Pivot-mass tick on both panels
        for ax in (ax1, ax2):
            ax.axvline(10.0 ** log10_MA if ax is ax1 else log10_MA,
                       color=col, lw=0.8, ls=":", alpha=0.6)

    # Scatter variant at z=0.14 — solid line (slightly shifted)
    log10mstar_sc = np.asarray(smhm_girelli20(log10m, 0.14, **_GIRELLI20_SCATTER))
    ratio_sc = 10.0 ** (log10mstar_sc - _LOG10M)
    ax1.semilogy(_M_H, ratio_sc, color="C1", lw=1.5, ls="--",
                 label=r"$z=0.14$  (Table 4, $\sigma_*$=0.2 dex)")
    ax2.plot(_LOG10M, log10mstar_sc, color="C1", lw=1.5, ls="--")

    # ±0.2 dex scatter band at z=0.14 (no-scatter relation)
    log10mstar_14 = np.asarray(smhm_girelli20(log10m, 0.14, **_GIRELLI20_NO_SCATTER))
    ax2.fill_between(_LOG10M,
                     log10mstar_14 - 0.2, log10mstar_14 + 0.2,
                     color="C1", alpha=0.15, label=r"$\pm 0.2\,$dex scatter ($z=0.14$)")

    ax1.set_xscale("log")
    ax1.set_xlim(_M_H[0], _M_H[-1])
    ax1.set_ylim(1e-4, 2e-1)
    ax1.set_xlabel(r"$M_h\;[h^{-1}\,M_\odot]$", fontsize=12)
    ax1.set_ylabel(r"$M_*/M_h$", fontsize=12)
    ax1.set_title("Girelli+2020 SHMR — stellar fraction", fontsize=11)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, which="both", alpha=0.25)

    # Diagonal 1:1 reference on right panel
    ax2.plot([10, 16], [10, 16], "k--", lw=0.7, alpha=0.3, label=r"$M_*=M_h$ (unphysical)")
    ax2.set_xlim(10.5, 16)
    ax2.set_ylim(5, 14)
    ax2.set_xlabel(r"$\log_{10}(M_h\;[h^{-1}\,M_\odot])$", fontsize=12)
    ax2.set_ylabel(r"$\log_{10}(M_*\;[h^{-1}\,M_\odot])$", fontsize=12)
    ax2.set_title("Girelli+2020 SHMR — stellar mass", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, which="both", alpha=0.25)

    fig.text(
        0.5, 0.005,
        "Girelli et al. 2020, A&A 634, A135 (arXiv:2007.06220)  |  "
        "Dotted verticals: pivot mass log₁₀M_A = B + z·μ",
        ha="center", va="bottom", fontsize=7, color="0.45",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = _IMG_DIR / "fig_agn_01_shmr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 2: mean L_X vs M_h ─────────────────────────────────────────────────

def make_fig02_lx_mhalo():
    agn = XrayAGNModel()
    boost_dex = np.log10(float(_scatter_boost(_DEFAULT_SCATTER_LX)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for z, col in zip(_REDSHIFTS, _COLORS):
        lx  = agn.mean_agn_lx(_M_H, z)                 # with duty cycle
        fdc = float(_duty_cycle_at_z(z))
        lx_no_dc = lx / fdc                             # remove duty cycle

        ax.semilogy(_M_H, lx, color=col, lw=2,
                    label=rf"$z={z:.2f}$  ($f_{{DC}}={fdc:.3f}$)")
        ax.semilogy(_M_H, lx_no_dc, color=col, lw=1.2, ls="--", alpha=0.6)

    # Duty-cycle redshift nodes as a secondary annotation
    z_nodes = np.asarray(_DUTY_CYCLE_Z)
    fdc_nodes = 10.0 ** np.asarray(_DUTY_CYCLE_LOG)

    ax.set_xscale("log")
    ax.set_xlim(_M_H[0], _M_H[-1])
    ax.set_ylim(1e38, 1e46)
    ax.set_xlabel(r"$M_h\;[h^{-1}\,M_\odot]$", fontsize=12)
    ax.set_ylabel(r"$\langle L_X^{0.5-2\,\mathrm{keV}}\rangle\;[\mathrm{erg\,s}^{-1}]$",
                  fontsize=12)
    ax.set_title(
        rf"Mean soft X-ray AGN luminosity per halo  "
        rf"(scatter boost: $+{boost_dex:.2f}\,$dex)",
        fontsize=11,
    )
    ax.text(0.97, 0.06,
            "Solid: with duty cycle  |  Dashed: without",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)

    fig.text(
        0.5, 0.005,
        "Comparat et al. 2019, A&A 622, A12 (arXiv:1901.10866)  |  "
        "Girelli+2020 SHMR  |  scatter σ = 0.8 dex",
        ha="center", va="bottom", fontsize=7, color="0.45",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = _IMG_DIR / "fig_agn_02_lx_mhalo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 3: HOD-weighted <L_X> vs log10 M_min ──────────────────────────────

def make_fig03_lx_logmmin():
    from scipy.special import erfc as sp_erfc

    pklin = LinearPowerSpectrum()
    hmf   = HaloMassFunction(pklin.pk_linear, rho_mean=_RHO_M, model="tinker08")
    agn   = XrayAGNModel()

    sigma_logm = 0.25
    log10m_min_arr = np.linspace(11.5, 14.0, 80)

    # Fine mass grid for integration (M_h in Msun/h, no /h offset needed —
    # smhm_girelli20 takes log10(M_h / (Msun/h)))
    log10m_int = np.linspace(10.0, 16.0, 600)
    m_int      = 10.0 ** log10m_int
    dm         = np.diff(m_int)   # (599,)

    fig, ax = plt.subplots(figsize=(8, 5))

    for z, col in zip([0.14, 0.5, 1.0], ["C1", "C2", "C3"]):
        dndm = np.asarray(hmf.dndm(jnp.asarray(m_int), z, _THETA))
        lx   = agn.mean_agn_lx(m_int, z)

        lx_hod = np.empty(len(log10m_min_arr))
        for i, logMmin in enumerate(log10m_min_arr):
            nc     = 0.5 * sp_erfc((logMmin - log10m_int) / (np.sqrt(2.0) * sigma_logm))
            weight = dndm * nc
            # Trapezoid rule: integrate over M (not log M)
            w_mid  = 0.5 * (weight[:-1] + weight[1:])
            lx_mid = 0.5 * (lx[:-1]    + lx[1:])
            num    = np.sum(w_mid * lx_mid * dm)
            den    = np.sum(w_mid * dm)
            lx_hod[i] = num / den if den > 0 else np.nan

        ax.semilogy(log10m_min_arr, lx_hod, color=col, lw=2,
                    label=rf"$z={z:.2f}$")

    ax.set_xlim(11.5, 14.0)
    ax.set_ylim(1e38, 1e45)
    ax.set_xlabel(r"$\log_{10}(M_{\min}\;[h^{-1}\,M_\odot])$", fontsize=12)
    ax.set_ylabel(
        r"$\langle L_X^{0.5-2\,\mathrm{keV}}\rangle_{\rm HOD}\;[\mathrm{erg\,s}^{-1}]$",
        fontsize=12,
    )
    ax.set_title(
        r"HOD-weighted mean AGN luminosity vs $\log_{10}M_{\min}$"
        "\n"
        r"(Tinker+2008 HMF, $\sigma_{\log m}=0.25$)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)

    # Annotate the seven BGS stellar-mass samples from Comparat+2025
    bgs_logmmin = [12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.3]
    bgs_labels  = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    for logm, lbl in zip(bgs_logmmin, bgs_labels):
        ax.axvline(logm, color="0.6", lw=0.7, ls=":")
        ax.text(logm, 2e44, lbl, ha="center", va="bottom", fontsize=7, color="0.5")

    fig.text(
        0.5, 0.005,
        "Comparat et al. 2019 (arXiv:1901.10866)  |  "
        "Girelli+2020 SHMR  |  BGS sample logMmin from Comparat+2025 (arXiv:2503.19796)",
        ha="center", va="bottom", fontsize=7, color="0.45",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = _IMG_DIR / "fig_agn_03_lx_logmmin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 4: predicted soft XLF vs Hasinger+2005 ─────────────────────────────

def _hasinger05_ldde(log10_lx_arr: np.ndarray, z: float) -> np.ndarray:
    """Hasinger, Miyaji & Schmidt 2005 LDDE soft (0.5-2 keV) XLF.

    Parameters from Table 4 of A&A 441, 417 (arXiv:astro-ph/0506118).
    Cosmology: h=0.70, Ω_m=0.3, Ω_Λ=0.7.

    Returns Φ [Mpc⁻³ dex⁻¹] at the given redshift.
    """
    # Double power-law normalisation at z=0
    A        = 1.42e-5    # Mpc^{-3} dex^{-1}
    log10_Ls = 44.0       # erg/s  (break luminosity)
    gamma1   = 0.86       # faint-end slope
    gamma2   = 2.23       # bright-end slope

    # LDDE evolution parameters
    p1       = 3.97
    p2       = -1.5
    zc0      = 1.96
    alpha    = 0.21
    log10_La = 44.6       # erg/s  (luminosity above which z_c is fixed)

    L     = 10.0 ** log10_lx_arr
    Ls    = 10.0 ** log10_Ls
    La    = 10.0 ** log10_La

    # Local XLF (z=0)
    phi0 = A / ((L / Ls) ** gamma1 + (L / Ls) ** gamma2)

    # Luminosity-dependent peak redshift
    zc = np.where(L < La, zc0 * (L / La) ** alpha, zc0)

    # Evolution factor e_d(z, L)
    ed = np.where(
        z <= zc,
        (1.0 + z) ** p1,
        (1.0 + zc) ** p1 * ((1.0 + z) / (1.0 + zc)) ** p2,
    )

    return phi0 * ed


def make_fig04_xlf():
    from scipy.stats import norm as sp_norm

    pklin = LinearPowerSpectrum()
    hmf   = HaloMassFunction(pklin.pk_linear, rho_mean=_RHO_M, model="tinker08")
    agn   = XrayAGNModel()
    h     = float(_THETA["h"])

    # Mass integration grid
    log10m_int = np.linspace(10.0, 16.0, 800)
    m_int      = 10.0 ** log10m_int

    # Luminosity grid
    log10lx_arr = np.linspace(41.5, 46.0, 200)

    sigma = agn._scatter_lx  # 0.8 dex

    fig, ax = plt.subplots(figsize=(8, 6))

    for z, col in zip([0.1, 0.5, 1.0], ["C0", "C2", "C3"]):
        dndm   = np.asarray(hmf.dndm(jnp.asarray(m_int), z, _THETA))
        fdc    = float(_duty_cycle_at_z(z))

        # Median log10(L_X) per active AGN — strip out scatter boost and f_DC
        # mean_agn_log10lx returns log10(L_hard × h2s × boost × f_DC)
        log10_median = agn.mean_agn_log10lx(m_int, z) \
                       - np.log10(agn._boost) \
                       - np.log10(fdc)     # shape (NM,)

        # XLF via convolution: Φ(L) = ∫ dndM × f_DC × Gauss(L; median(M), σ) dM
        # Use Gaussian PDF: shape (N_lx, NM)
        diff   = log10lx_arr[:, None] - log10_median[None, :]   # (N_lx, NM)
        kernel = sp_norm.pdf(diff, scale=sigma)                  # (N_lx, NM)

        # Integrate over M with trapezoid rule
        integrand = kernel * (dndm * fdc)[None, :]               # (N_lx, NM)
        dm        = np.diff(m_int)
        xlf_h3    = np.sum(
            0.5 * (integrand[:, :-1] + integrand[:, 1:]) * dm[None, :], axis=1
        )  # (N_lx,) in h³ Mpc⁻³ dex⁻¹

        # Convert to Mpc⁻³ dex⁻¹ (divide by h³)
        xlf_pred = xlf_h3 / h ** 3

        # Hasinger+2005 reference
        xlf_has = _hasinger05_ldde(log10lx_arr, z)

        lbl = rf"$z={z:.1f}$"
        ax.semilogy(log10lx_arr, xlf_pred, color=col, lw=2, label=lbl + " (model)")
        ax.semilogy(log10lx_arr, xlf_has,  color=col, lw=1.5, ls="--",
                    label=lbl + " (Hasinger+05)")

    ax.set_xlim(41.5, 46.0)
    ax.set_ylim(1e-10, 1e-3)
    ax.set_xlabel(r"$\log_{10}(L_X^{0.5-2\,\mathrm{keV}}\;[\mathrm{erg\,s}^{-1}])$",
                  fontsize=12)
    ax.set_ylabel(r"$\Phi\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$", fontsize=12)
    ax.set_title("Soft X-ray AGN luminosity function: model vs Hasinger+2005 LDDE",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    fig.text(
        0.5, 0.005,
        "Hasinger, Miyaji & Schmidt 2005, A&A 441, 417 (arXiv:astro-ph/0506118)  |  "
        "Model: Tinker+2008 HMF + Girelli+2020 SHMR + Comparat+2019 LX–M*",
        ha="center", va="bottom", fontsize=7, color="0.45",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = _IMG_DIR / "fig_agn_04_xlf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def _ueda14_ldde(log10_lx_arr: np.ndarray, z: float) -> np.ndarray:
    """Ueda et al. 2014 LDDE total hard (2-10 keV) XLF.

    Parameters from Table 3 of arXiv:1402.7902.
    Cosmology: h=0.70, Ω_m=0.3, Ω_Λ=0.7.
    Counts ALL AGN (type 1 + type 2 + Compton-thick).

    Returns Φ [Mpc⁻³ dex⁻¹].
    """
    phi0   = 3.31e-6
    Ls     = 10.0 ** 43.97
    gamma1 = 0.96
    gamma2 = 2.71
    e1, e2 = 5.54, -0.36
    zc0    = 1.84
    alpha  = 0.335
    La     = 10.0 ** 44.61

    L   = 10.0 ** log10_lx_arr
    phi = phi0 / ((L / Ls) ** gamma1 + (L / Ls) ** gamma2)
    zc  = np.where(L < La, zc0 * (L / La) ** alpha, zc0)
    ed  = np.where(
        z <= zc,
        (1.0 + z) ** e1,
        (1.0 + zc) ** e1 * ((1.0 + z) / (1.0 + zc)) ** e2,
    )
    return phi * ed


def _hasinger_to_hard(log10_lx_hard: np.ndarray, z: float) -> np.ndarray:
    """Hasinger+2005 soft LDDE shifted to the hard band (type-1-only hard XLF).

    L_hard = L_soft / h2s  →  log10_L_soft = log10_L_hard + log10(h2s).
    Represents TYPE 1 (unobscured) AGN only; should bracket the model from above.
    """
    return _hasinger05_ldde(log10_lx_hard + np.log10(0.35), z)


def make_fig05_hard_xlf():
    from scipy.stats import norm as sp_norm

    pklin = LinearPowerSpectrum()
    hmf   = HaloMassFunction(pklin.pk_linear, rho_mean=_RHO_M, model="tinker08")
    agn   = XrayAGNModel()
    h     = float(_THETA["h"])

    log10m_int  = np.linspace(10.0, 16.0, 800)
    m_int       = 10.0 ** log10m_int
    log10lx_arr = np.linspace(41.5, 46.5, 200)
    sigma       = agn._scatter_lx  # 0.8 dex

    fig, ax = plt.subplots(figsize=(8, 6))

    for z, col in zip([0.1, 0.5, 1.0], ["C0", "C2", "C3"]):
        dndm = np.asarray(hmf.dndm(jnp.asarray(m_int), z, _THETA))
        fdc  = float(_duty_cycle_at_z(z))

        # Median log10(L_hard) per active AGN: undo scatter boost, duty cycle, band ratio
        log10_median_hard = (
            agn.mean_agn_log10lx(m_int, z)
            - np.log10(agn._boost)
            - np.log10(fdc)
            - np.log10(agn._h2s)
        )

        diff      = log10lx_arr[:, None] - log10_median_hard[None, :]
        kernel    = sp_norm.pdf(diff, scale=sigma)
        integrand = kernel * (dndm * fdc)[None, :]
        dm        = np.diff(m_int)
        xlf_h3    = np.sum(
            0.5 * (integrand[:, :-1] + integrand[:, 1:]) * dm[None, :], axis=1
        )
        xlf_pred = xlf_h3 / h ** 3

        lbl = rf"$z={z:.1f}$"
        ax.semilogy(log10lx_arr, xlf_pred,
                    color=col, lw=2,   label=lbl + " (model, type-1)")
        ax.semilogy(log10lx_arr, _ueda14_ldde(log10lx_arr, z),
                    color=col, lw=1.5, ls="--",  label=lbl + " (Ueda+14 total)")
        ax.semilogy(log10lx_arr, _hasinger_to_hard(log10lx_arr, z),
                    color=col, lw=1.0, ls=":",   label=lbl + " (Has+05→hard)")

    ax.set_xlim(41.5, 46.5)
    ax.set_ylim(1e-10, 1e-3)
    ax.set_xlabel(
        r"$\log_{10}(L_X^{2-10\,\mathrm{keV}}\;[\mathrm{erg\,s}^{-1}])$",
        fontsize=12,
    )
    ax.set_ylabel(r"$\Phi\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$", fontsize=12)
    ax.set_title(
        "Hard X-ray AGN luminosity function: model vs references", fontsize=11
    )
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    fig.text(
        0.5, 0.005,
        "Ueda+2014 (arXiv:1402.7902) total hard XLF  |  "
        "Has+05→hard: soft LDDE shifted by +log10(1/h2s)  |  "
        "Model: type-1 AGN only (calibrated to Hasinger+2005 soft)",
        ha="center", va="bottom", fontsize=7, color="0.45",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = _IMG_DIR / "fig_agn_05_hard_xlf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Generating AGN model figures …")
    make_fig01_shmr()
    make_fig02_lx_mhalo()
    make_fig03_lx_logmmin()
    make_fig04_xlf()
    make_fig05_hard_xlf()
    print("Done.")


if __name__ == "__main__":
    main()
