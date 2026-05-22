"""Showcase the GGA forward-model pipeline with 17 publication-quality figures.

Generates figures for all components of the forward model — from the linear
matter power spectrum to the weak-lensing excess surface mass density with
baryon and intrinsic-alignment corrections.

All figures use z = 0.14 (BGS effective redshift).  P_lin and P_nl figures
additionally show the Planck 2018 S8 ± 3σ uncertainty band.

Outputs (saved to ``--output`` directory, default ``results/showcase/``):

    fig01_power_spectrum.pdf        — P_lin(k) + ratio to fiducial, S8 ±3σ
    fig01b_nonlinear_power_spectrum — P_nl: Aletheia, CSST, CAMB HMcode-2020, to k=10
    fig02_hmf.pdf                   — dn/dM(M) and b(M) at z=0.14, + S8 ±3σ
    fig02a_hmf_models.pdf           — 6 analytic HMF models + ratio sub-panel
    fig02b_bias_models.pdf          — linear halo bias b(M) at z = 0, 0.14, 0.5, 1.0
    fig03_concentration.pdf         — c(M): 5 colossus models + σ₈ dependence
    fig04_halo_profiles.pdf         — NFW and Einasto Fourier transforms + ratio panel
    fig05_hod_occupation.pdf        — 3×3 panel: all 9 HOD/CSMF models
    fig05b_hod_redshift.pdf         — n_gal and b_eff vs z for fixed More+2015 HOD
    fig06_baryon_fraction.pdf       — sigmoid (no floor), + 1% floor, + double-sigmoid
    fig07_gas_concentration.pdf     — η(M) + c_gas(M) with σ₈ cosmology dependence
    fig08_wp.pdf                    — w_p(r_p): total, 1h, 2h
    fig09_delta_sigma.pdf           — ΔΣ(R): total, 1h, 2h
    fig10_cdm_baryon_split.pdf      — ΔΣ_CDM, ΔΣ_b, ΔΣ_total
    fig11_ia_delta_sigma.pdf        — ΔΣ(R) for A_IA = 0, 0.5, 1, 2
    fig12_wp_summary.pdf            — w_p: 1h/2h + off-centering
    fig13_ds_summary.pdf            — ΔΣ: 1h/2h + CDM/gas split + NLA IA

Usage
-----
::

    python scripts/utils/showcase_forward_model.py --output results/showcase/
    python scripts/utils/showcase_forward_model.py --show   # interactive

References
----------
Lewis & Bridle 2002, CAMB (`arXiv:astro-ph/9911177 <https://arxiv.org/abs/astro-ph/9911177>`_)
Tinker et al. 2008 (`arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_)
Diemer & Joyce 2019 (`arXiv:1809.07326 <https://arxiv.org/abs/1809.07326>`_)
Navarro, Frenk & White 1997 (`arXiv:astro-ph/9611107 <https://arxiv.org/abs/astro-ph/9611107>`_)
Einasto 1965; Klypin et al. 2016 (`arXiv:1711.01744 <https://arxiv.org/abs/1711.01744>`_)
Cooray & Sheth 2002 (`arXiv:astro-ph/0206508 <https://arxiv.org/abs/astro-ph/0206508>`_)
More et al. 2015 (`arXiv:1211.6211 <https://arxiv.org/abs/1211.6211>`_)
FLAMINGO (`arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_)
Veenema et al. 2026 (`arXiv:2603.13095 <https://arxiv.org/abs/2603.13095>`_)
IllustrisTNG baryonic effects (`arXiv:2409.01758 <https://arxiv.org/abs/2409.01758>`_)
Mead et al. 2015 (`arXiv:1611.08606 <https://arxiv.org/abs/1611.08606>`_)
Bridle & King 2007 (`arXiv:0705.0166 <https://arxiv.org/abs/0705.0166>`_)
Planck 2018 (`arXiv:1807.06209 <https://arxiv.org/abs/1807.06209>`_)
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# BGS effective redshift used throughout
_Z = 0.14

# Planck 2018 S8 constraint (TT+TE+EE+lowE, Table 2)
_S8_PLANCK = 0.832
_S8_SIGMA  = 0.013
_SIGMA8_FID = 0.811  # colossus fiducial

# ---------------------------------------------------------------------------
# Setup: build all model components once, share across figures
# ---------------------------------------------------------------------------

def _setup(theta_override: dict | None = None):
    """Construct the forward-model components and a fiducial Planck 2018 cosmology."""
    from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
    from hod_mod.cosmology.halo_mass_function import make_hmf
    from hod_mod.cosmology.halo_profiles import HaloProfile, nfw_uk, einasto_uk
    from hod_mod.galaxies.hod import MoreHODModel
    from hod_mod.galaxies.clustering import FullHaloModelPrediction
    from hod_mod.galaxies.baryon_fraction import BaryonFractionSigmoid
    from hod_mod.galaxies.intrinsic_alignment import NLAModel

    pk_lin = LinearPowerSpectrum()
    theta  = pk_lin.default_cosmology()
    if theta_override:
        theta.update(theta_override)

    hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)

    colossus_cosmo = {
        "flat":   True,
        "H0":     theta["h"] * 100.0,
        "Om0":    theta["Omega_m"],
        "Ob0":    theta["Omega_b"],
        "sigma8": _SIGMA8_FID,
        "ns":     theta["n_s"],
    }
    hp = HaloProfile(colossus_cosmo)

    # Default HOD (More+2015 parameters, arXiv:1211.6211 Table 1)
    hod = MoreHODModel(hmf, hmf.bias)
    hod_params = {
        "log10mmin":  11.5,
        "sigma_logm": 0.67,
        "log10m1":    12.5,
        "alpha":      1.0,
        "kappa":      1.13,
        "alpha_inc":  0.0,
        "log10m_inc": 11.5,
    }

    bf   = BaryonFractionSigmoid()
    pred = FullHaloModelPrediction(pk_lin, hod, hp, baryon_fraction=bf)
    nla  = NLAModel(pk_lin.pk_linear)

    # --- S8 ± 3σ variation (Planck 2018) -----------------------------------
    Omega_m = float(theta["Omega_m"])
    S8_lo = _S8_PLANCK - 3.0 * _S8_SIGMA   # 0.793
    S8_hi = _S8_PLANCK + 3.0 * _S8_SIGMA   # 0.871
    sigma8_lo = S8_lo * np.sqrt(0.3 / Omega_m)
    sigma8_hi = S8_hi * np.sqrt(0.3 / Omega_m)
    # P_lin ∝ A_s ∝ sigma8² → direct scaling, no extra CAMB call
    scale_lo = (sigma8_lo / _SIGMA8_FID) ** 2
    scale_hi = (sigma8_hi / _SIGMA8_FID) ** 2
    # For HMF: modified ln10As — triggers extra CAMB calls
    ln10As_fid = float(theta["ln10^{10}A_s"])
    ln10As_lo  = ln10As_fid + 2.0 * np.log10(sigma8_lo / _SIGMA8_FID)
    ln10As_hi  = ln10As_fid + 2.0 * np.log10(sigma8_hi / _SIGMA8_FID)
    theta_s8_lo = {**theta, "ln10^{10}A_s": ln10As_lo}
    theta_s8_hi = {**theta, "ln10^{10}A_s": ln10As_hi}

    return {
        "pk_lin":       pk_lin,
        "theta":        theta,
        "hmf":          hmf,
        "hp":           hp,
        "nfw_uk":       nfw_uk,
        "einasto_uk":   einasto_uk,
        "hod":          hod,
        "hod_params":   hod_params,
        "bf":           bf,
        "pred":         pred,
        "nla":          nla,
        "colossus_cosmo": colossus_cosmo,
        # S8 variation
        "s8_scale_lo":  scale_lo,
        "s8_scale_hi":  scale_hi,
        "theta_s8_lo":  theta_s8_lo,
        "theta_s8_hi":  theta_s8_hi,
        "S8_lo":        S8_lo,
        "S8_hi":        S8_hi,
    }


# ---------------------------------------------------------------------------
# Fig 01 — Linear matter power spectrum P_lin(k) with S8 ±3σ
# ---------------------------------------------------------------------------

def fig01_power_spectrum(ctx, output_dir, show=False):
    """P_lin(k) at z=0.14, with Planck 2018 S8 ±3σ variation + ratio panel.

    Two panels: top — P_lin(k); bottom — P_lin / P_lin^fid.

    Lewis & Bridle 2002 (CAMB `arXiv:astro-ph/9911177
    <https://arxiv.org/abs/astro-ph/9911177>`_).
    Planck 2018 S8 = 0.832 ± 0.013 (`arXiv:1807.06209
    <https://arxiv.org/abs/1807.06209>`_).
    """
    pk_lin, theta = ctx["pk_lin"], ctx["theta"]
    scale_lo = ctx["s8_scale_lo"]
    scale_hi = ctx["s8_scale_hi"]
    S8_lo, S8_hi = ctx["S8_lo"], ctx["S8_hi"]

    k   = np.logspace(-3, 1.5, 300)
    pk  = np.asarray(pk_lin.pk_linear(k, _Z, theta))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                   gridspec_kw={"height_ratios": [3, 1.5]},
                                   sharex=True)

    ax1.loglog(k, pk,            lw=2, color="#1f77b4", ls="-",
               label=rf"$z={_Z}$ (Planck 2018 fiducial)")
    ax1.loglog(k, pk * scale_hi, lw=1.5, color="#ff7f0e", ls="--",
               label=rf"S8 + 3σ = {S8_hi:.3f}")
    ax1.loglog(k, pk * scale_lo, lw=1.5, color="#d62728", ls=":",
               label=rf"S8 − 3σ = {S8_lo:.3f}")
    ax1.set_ylabel(r"$P_{\rm lin}(k)$ [$(h^{-1}\,{\rm Mpc})^3$]", fontsize=12)
    ax1.set_title(f"Linear matter power spectrum (CAMB, $z={_Z}$)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_xlim(1e-3, 30)

    ax2.semilogx(k, np.ones_like(k), lw=1, color="#1f77b4", ls="-")
    ax2.semilogx(k, np.full_like(k, scale_hi), lw=1.5, color="#ff7f0e", ls="--",
                 label=rf"S8 + 3σ")
    ax2.semilogx(k, np.full_like(k, scale_lo), lw=1.5, color="#d62728", ls=":",
                 label=rf"S8 − 3σ")
    ax2.axhline(1, color="k", lw=0.7, ls="--")
    ax2.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=12)
    ax2.set_ylabel(r"$P_{\rm lin}/P_{\rm lin}^{\rm fid}$", fontsize=11)
    ax2.set_ylim(0.85, 1.20)
    ax2.legend(fontsize=9, loc="upper right")

    fig.text(0.99, 0.01,
             "Lewis & Bridle 2002 (arXiv:astro-ph/9911177); Planck 2018 (arXiv:1807.06209)",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig01_power_spectrum.pdf", show)


# ---------------------------------------------------------------------------
# Fig 01b — Non-linear matter power spectrum P_nl(k) with S8 ±3σ
# ---------------------------------------------------------------------------

def fig01b_nonlinear_power_spectrum(ctx, output_dir, show=False):
    """P_nl(k) at z=0.14 from Aletheia, CSST CEmulator, and CAMB HMcode-2020.

    Two-panel vertical layout:
    top — P_nl + P_lin on a log-log scale, extended to k=10 h/Mpc (CSST range);
    bottom — non-linear boost factor P_nl/P_lin for all three emulators.

    CAMB HMcode-2020: `arXiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_.
    Aletheia: `arXiv:2511.13826 <https://arxiv.org/abs/2511.13826>`_.
    CEmulator: Chen et al. 2025 (CSST CEmulator v2.0).
    """
    from hod_mod.cosmology.nonlinear import NonLinearPowerSpectrum, HALOFITSpectrum
    pk_lin, theta = ctx["pk_lin"], ctx["theta"]

    # Extended k grid to k=10 h/Mpc (CSST and HALOFIT range)
    k_wide = np.logspace(-2, 1.0, 300)
    # Aletheia is only valid up to ~1.3 h/Mpc (2 Mpc^-1); extend cautiously
    k_ale  = np.logspace(-2, 0.1, 200)

    pkli_wide = np.asarray(pk_lin.pk_linear(k_wide, _Z, theta))

    # --- Aletheia (blue) ---
    pknl_ale = None
    try:
        pk_nl_ale = NonLinearPowerSpectrum(backend="aletheia")
        pknl_ale = np.asarray(pk_nl_ale.pk_nonlinear(k_ale, _Z, theta))
    except Exception as exc:
        warnings.warn(f"Aletheia unavailable: {exc}")

    # --- CSST CEmulator (orange) — extends to k=10 h/Mpc ---
    pknl_csst = None
    try:
        pk_nl_csst = NonLinearPowerSpectrum(backend="csst")
        pknl_csst = np.asarray(pk_nl_csst.pk_nonlinear(k_wide, _Z, theta))
    except Exception as exc:
        warnings.warn(f"CSST CEmulator unavailable: {exc}")

    # --- CAMB HMcode-2020 (green) ---
    try:
        pk_halofit = HALOFITSpectrum(halofit_version="mead2020")
        pknl_hf = np.asarray(pk_halofit.pk_nonlinear(k_wide, _Z, theta))
    except Exception as exc:
        warnings.warn(f"CAMB HMcode-2020 unavailable: {exc}")
        pknl_hf = None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                                   gridspec_kw={"height_ratios": [2.5, 1]},
                                   sharex=True)

    # Top — absolute power spectra
    ax1.loglog(k_wide, pkli_wide, lw=1.2, color="0.5", ls="-",
               label=r"$P_{\rm lin}$")
    if pknl_ale is not None:
        pkli_ale = np.asarray(pk_lin.pk_linear(k_ale, _Z, theta))
        ax1.loglog(k_ale, pknl_ale, lw=2, color="#1f77b4", ls="-",
                   label="Aletheia (arXiv:2511.13826)")
    if pknl_csst is not None:
        ax1.loglog(k_wide, pknl_csst, lw=2, color="#ff7f0e", ls="-",
                   label="CSST CEmulator")
    if pknl_hf is not None:
        ax1.loglog(k_wide, pknl_hf, lw=2, color="#2ca02c", ls="-",
                   label="CAMB HMcode-2020")
    ax1.set_ylabel(r"$P(k)$ [$(h^{-1}\,{\rm Mpc})^3$]", fontsize=12)
    ax1.set_title(f"Non-linear matter power spectrum ($z={_Z}$)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_xlim(k_wide[0], k_wide[-1])

    # Bottom — boost factor P_nl / P_lin
    ax2.axhline(1, color="k", lw=0.8, ls="--")
    if pknl_ale is not None:
        boost_ale = pknl_ale / np.maximum(pkli_ale, 1e-30)
        ax2.semilogx(k_ale, boost_ale, lw=2, color="#1f77b4", ls="-",
                     label="Aletheia")
    if pknl_csst is not None:
        boost_csst = pknl_csst / np.maximum(pkli_wide, 1e-30)
        ax2.semilogx(k_wide, boost_csst, lw=2, color="#ff7f0e", ls="-",
                     label="CSST CEmulator")
    if pknl_hf is not None:
        boost_hf = pknl_hf / np.maximum(pkli_wide, 1e-30)
        ax2.semilogx(k_wide, boost_hf, lw=2, color="#2ca02c", ls="-",
                     label="HMcode-2020")
    ax2.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=12)
    ax2.set_ylabel(r"$P_{\rm nl}/P_{\rm lin}$", fontsize=11)
    ax2.set_ylim(0.5, None)
    ax2.legend(fontsize=9, loc="upper left")

    fig.text(0.99, 0.01,
             "Aletheia (arXiv:2511.13826); CSST CEmulator; CAMB HMcode-2020 (arXiv:2009.01858)",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig01b_nonlinear_power_spectrum.pdf", show)


# ---------------------------------------------------------------------------
# Fig 02 — Halo mass function and bias, with S8 ±3σ
# ---------------------------------------------------------------------------

def fig02_hmf(ctx, output_dir, show=False):
    """dn/dM and b(M) at z=0.14, with Planck 2018 S8 ±3σ variation.

    .. math::

        \\frac{\\mathrm{d}n}{\\mathrm{d}M}(M,z)\\quad [h^4\\,M_\\odot^{-1}\\,{\\rm Mpc}^{-3}]

    Tinker et al. 2008 (`arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_).
    """
    hmf, theta = ctx["hmf"], ctx["theta"]
    theta_lo   = ctx["theta_s8_lo"]
    theta_hi   = ctx["theta_s8_hi"]
    S8_lo, S8_hi = ctx["S8_lo"], ctx["S8_hi"]

    m = jnp.logspace(11, 16, 200)

    dndm_fid = np.asarray(hmf.dndm(m, _Z, theta))
    bias_fid = np.asarray(hmf.bias(m, _Z, theta))

    try:
        dndm_lo = np.asarray(hmf.dndm(m, _Z, theta_lo))
        dndm_hi = np.asarray(hmf.dndm(m, _Z, theta_hi))
        bias_lo = np.asarray(hmf.bias(m, _Z, theta_lo))
        bias_hi = np.asarray(hmf.bias(m, _Z, theta_hi))
        has_s8 = True
    except Exception:
        has_s8 = False

    m_np = np.asarray(m)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.loglog(m_np, dndm_fid, lw=2, color="#1f77b4", label=f"$z={_Z}$ (fiducial)")
    if has_s8:
        ax1.loglog(m_np, dndm_hi, lw=1.5, color="#1f77b4", ls="--",
                   label=rf"S8 + 3σ = {S8_hi:.3f}")
        ax1.loglog(m_np, dndm_lo, lw=1.5, color="#1f77b4", ls=":",
                   label=rf"S8 − 3σ = {S8_lo:.3f}")

    ax1.set_xlabel(r"$M_h$ [$h^{-1}\,M_\odot$]", fontsize=12)
    ax1.set_ylabel(r"$\mathrm{d}n/\mathrm{d}M$ [$h^4\,M_\odot^{-1}\,{\rm Mpc}^{-3}$]",
                   fontsize=11)
    ax1.set_title(f"Halo mass function (Tinker+2008, $z={_Z}$)", fontsize=11)
    ax1.legend(fontsize=10)

    ax2.loglog(m_np, bias_fid, lw=2, color="#1f77b4", label=f"$z={_Z}$ (fiducial)")
    if has_s8:
        ax2.loglog(m_np, bias_hi, lw=1.5, color="#1f77b4", ls="--",
                   label=rf"S8 + 3σ = {S8_hi:.3f}")
        ax2.loglog(m_np, bias_lo, lw=1.5, color="#1f77b4", ls=":",
                   label=rf"S8 − 3σ = {S8_lo:.3f}")
    ax2.set_xlabel(r"$M_h$ [$h^{-1}\,M_\odot$]", fontsize=12)
    ax2.set_ylabel(r"$b(M)$", fontsize=12)
    ax2.set_title("Linear halo bias", fontsize=11)
    ax2.legend(fontsize=10)

    fig.text(0.99, 0.01,
             "Tinker et al. 2008 (arXiv:0803.2706); Planck 2018 (arXiv:1807.06209)",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig02_hmf.pdf", show)


# ---------------------------------------------------------------------------
# Fig 02a — HMF model comparison (6 analytic models)
# ---------------------------------------------------------------------------

def fig02a_hmf_models(ctx, output_dir, show=False):
    """Six analytic HMF models at z=0.14 + ratio sub-panel.

    Tinker+2008, Press+1974, Sheth+1999, Warren+2006, Bocquet+2016, Watson+2013.
    """
    from hod_mod.cosmology.halo_mass_function import make_hmf

    pk_lin, theta = ctx["pk_lin"], ctx["theta"]

    hmf_variants = {
        "tinker08":  make_hmf("tinker08",  pk_func=pk_lin.pk_linear),
        "press74":   make_hmf("press74",   pk_func=pk_lin.pk_linear),
        "sheth99":   make_hmf("sheth99",   pk_func=pk_lin.pk_linear),
        "warren06":  make_hmf("warren06",  pk_func=pk_lin.pk_linear),
        "bocquet16": make_hmf("bocquet16", pk_func=pk_lin.pk_linear),
        "watson13":  make_hmf("watson13",  pk_func=pk_lin.pk_linear),
    }
    style = {
        "tinker08":  ("#1f77b4", "-",   r"Tinker+2008 (fiducial)"),
        "press74":   ("0.5",    "--",  "Press+1974"),
        "sheth99":   ("#9467bd",":",   "Sheth+1999"),
        "warren06":  ("#ff7f0e","-.",  "Warren+2006"),
        "bocquet16": ("#2ca02c", "-",  "Bocquet+2016"),
        "watson13":  ("#d62728", "--", "Watson+2013"),
    }

    m = jnp.logspace(11, 16, 200)
    m_np = np.asarray(m)

    dndm = {}
    for name, hmf_i in hmf_variants.items():
        try:
            dndm[name] = np.asarray(hmf_i.dndm(m, _Z, theta))
        except Exception as exc:
            warnings.warn(f"HMF {name} failed: {exc}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                   gridspec_kw={"height_ratios": [2.5, 1]},
                                   sharex=True)

    ref = dndm.get("tinker08")
    for name, d in dndm.items():
        c, ls, label = style[name]
        ax1.loglog(m_np, d, lw=2 if name == "tinker08" else 1.5,
                   color=c, ls=ls, label=label)
        if ref is not None and name != "tinker08":
            ax2.semilogx(m_np, d / np.maximum(ref, 1e-50),
                         lw=1.5, color=c, ls=ls, label=label)

    ax1.set_ylabel(r"$\mathrm{d}n/\mathrm{d}M$ [$h^4\,M_\odot^{-1}\,{\rm Mpc}^{-3}$]",
                   fontsize=11)
    ax1.set_title(f"HMF model comparison ($z={_Z}$)", fontsize=11)
    ax1.legend(fontsize=9, ncol=2)

    ax2.axhline(1, color="#1f77b4", lw=1.2, ls="-", label="Tinker+2008")
    ax2.set_xlabel(r"$M_h$ [$h^{-1}\,M_\odot$]", fontsize=12)
    ax2.set_ylabel(r"$\frac{\mathrm{d}n/\mathrm{d}M}{\rm (Tinker+08)}$", fontsize=11)
    ax2.set_ylim(0, 2.5)
    ax2.legend(fontsize=8, ncol=2)

    fig.text(0.99, 0.01,
             "Tinker+2008 (arXiv:0803.2706); Press+1974; Sheth+1999; Watson+2013; Bocquet+2016",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig02a_hmf_models.pdf", show)


# ---------------------------------------------------------------------------
# Fig 02b — Halo bias redshift evolution
# ---------------------------------------------------------------------------

def fig02b_bias_models(ctx, output_dir, show=False):
    """Linear halo bias b(M) at z = 0, 0.14, 0.5, 1.0 (Tinker+2008/2010).

    Two panels: top — b(M, z) at four redshifts; bottom — b(M)/b(M,z=0.14).
    """
    hmf, theta = ctx["hmf"], ctx["theta"]
    m = jnp.logspace(11, 16, 200)
    m_np = np.asarray(m)

    redshifts = [0.0, _Z, 0.5, 1.0]
    colors    = ["#9467bd", "#1f77b4", "#ff7f0e", "#d62728"]

    bias_z = {}
    for z in redshifts:
        try:
            bias_z[z] = np.asarray(hmf.bias(m, float(z), theta))
        except Exception as exc:
            warnings.warn(f"bias(z={z}) failed: {exc}")

    ref = bias_z.get(_Z)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                   gridspec_kw={"height_ratios": [2.5, 1]},
                                   sharex=True)

    for z, color in zip(redshifts, colors):
        b = bias_z.get(z)
        if b is None:
            continue
        ls = "-" if z == _Z else "--"
        ax1.loglog(m_np, b, lw=2, color=color, ls=ls,
                   label=rf"$z={z}$")
        if ref is not None and z != _Z:
            ax2.semilogx(m_np, b / np.maximum(ref, 1e-10),
                         lw=1.5, color=color, ls=ls, label=rf"$z={z}$")

    ax1.set_ylabel(r"$b(M)$", fontsize=12)
    ax1.set_title("Linear halo bias — redshift evolution (Tinker+2010)", fontsize=11)
    ax1.legend(fontsize=10)

    ax2.axhline(1, color="#1f77b4", lw=1.2, ls="-", label=rf"$z={_Z}$")
    ax2.set_xlabel(r"$M_h$ [$h^{-1}\,M_\odot$]", fontsize=12)
    ax2.set_ylabel(rf"$b(M,z)/b(M,z={_Z})$", fontsize=11)
    ax2.legend(fontsize=9)

    fig.text(0.99, 0.01,
             "Tinker et al. 2010 (arXiv:1001.3162); Tinker et al. 2008 (arXiv:0803.2706)",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig02b_bias_models.pdf", show)


# ---------------------------------------------------------------------------
# Fig 03 — Concentration–mass relation
# ---------------------------------------------------------------------------

def fig03_concentration(ctx, output_dir, show=False):
    """c(M) at z=0.14: multiple colossus models + σ₈ cosmology dependence.

    Top panel — 5 concentration models at Planck 2018 cosmology.
    Bottom panel — σ₈ dependence for diemer19 (σ₈ = 0.75, 0.811, 0.85).

    Diemer & Joyce 2019 (`arXiv:1809.07326 <https://arxiv.org/abs/1809.07326>`_).
    """
    from hod_mod.cosmology.halo_profiles import HaloProfile

    theta        = ctx["theta"]
    colossus_fid = ctx["colossus_cosmo"]
    m = jnp.logspace(11, 16, 150)
    m_np = np.asarray(m)

    cm_models = ["diemer19", "duffy08", "dutton14", "prada12", "bullock01"]
    cm_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    cm_labels = ["Diemer+2019", "Duffy+2008", "Dutton+2014", "Prada+2012", "Bullock+2001"]

    # Top panel: multiple models at Planck 2018
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                                   gridspec_kw={"height_ratios": [2, 1.5]})
    for name, color, label in zip(cm_models, cm_colors, cm_labels):
        try:
            hp_i = HaloProfile(colossus_fid, cm_relation=name)
            try:
                c_i = np.asarray(hp_i.concentration(m, _Z))
            except TypeError:
                c_i = np.asarray(hp_i.concentration(m, _Z, theta))
            ax1.loglog(m_np, c_i, lw=2, color=color, label=label)
        except Exception as exc:
            warnings.warn(f"c(M) model {name} failed: {exc}")

    ax1.set_ylabel(r"$c_{200m}(M)$", fontsize=12)
    ax1.set_title(f"Concentration–mass relation: model comparison ($z={_Z}$)", fontsize=11)
    ax1.legend(fontsize=9)

    # Bottom panel: σ₈ dependence for diemer19
    sigma8_vals  = [0.75, _SIGMA8_FID, 0.85]
    sigma8_colors = ["#d62728", "#1f77b4", "#2ca02c"]
    sigma8_labels = [r"$\sigma_8=0.75$", rf"$\sigma_8={_SIGMA8_FID}$ (fiducial)", r"$\sigma_8=0.85$"]
    sigma8_ls    = [":", "-", "--"]

    for s8, color, label, ls in zip(sigma8_vals, sigma8_colors, sigma8_labels, sigma8_ls):
        try:
            cosmo_i = {**colossus_fid, "sigma8": s8}
            hp_i = HaloProfile(cosmo_i, cm_relation="diemer19")
            try:
                c_i = np.asarray(hp_i.concentration(m, _Z))
            except TypeError:
                c_i = np.asarray(hp_i.concentration(m, _Z, theta))
            ax2.loglog(m_np, c_i, lw=2, color=color, ls=ls, label=label)
        except Exception as exc:
            warnings.warn(f"σ₈={s8} concentration failed: {exc}")

    ax2.set_xlabel(r"$M_{200m}$ [$h^{-1}\,M_\odot$]", fontsize=12)
    ax2.set_ylabel(r"$c_{200m}(M)$  [Diemer+2019]", fontsize=11)
    ax2.set_title(r"$\sigma_8$ cosmology dependence", fontsize=11)
    ax2.legend(fontsize=9)

    fig.text(0.99, 0.01,
             "Diemer & Joyce 2019 (arXiv:1809.07326); Duffy+2008; Dutton+2014; Prada+2012",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig03_concentration.pdf", show)


# ---------------------------------------------------------------------------
# Fig 04 — NFW and Einasto halo profile Fourier transforms
# ---------------------------------------------------------------------------

def fig04_halo_profiles(ctx, output_dir, show=False):
    r"""NFW (solid) and Einasto α=0.18 (dashed) Fourier transforms for 4 masses.

    .. math::

        \tilde{u}_{\rm NFW}(k|M) = \frac{4\pi r_s^3}{M}
        \bigl[\cos(kr_s)\bigl(\mathrm{Ci}(c\,k r_s)
              - \mathrm{Ci}(k r_s)\bigr)
              + \sin(kr_s)\bigl(\mathrm{Si}(c\,k r_s)
              - \mathrm{Si}(k r_s)\bigr)
              - \frac{\sin(c\,k r_s)}{(1+c)\,k r_s}\bigr]

    .. math::

        \tilde{u}_{\rm Ein}(k|M) = \frac{
            \int_0^{c\,r_s} \rho_{\rm Ein}(r)\,j_0(kr)\,r^2\,\mathrm{d}r}{
            \int_0^{c\,r_s} \rho_{\rm Ein}(r)\,r^2\,\mathrm{d}r}

    with :math:`\rho_{\rm Ein}(r)=\rho_s\exp[-(2/\\alpha)((r/r_s)^\\alpha-1)]`,
    :math:`\\alpha=0.18` (Klypin et al. 2016 `arXiv:1711.01744
    <https://arxiv.org/abs/1711.01744>`_).

    Cooray & Sheth 2002 Eq. 11; NFW 1997; Einasto 1965.
    """
    nfw_uk    = ctx["nfw_uk"]
    einasto_uk = ctx["einasto_uk"]
    hp, theta  = ctx["hp"], ctx["theta"]

    k      = np.logspace(-2, 2, 200)
    masses = [1e12, 1e13, 1e14, 1e15]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = [r"$10^{12}\,M_\odot/h$", r"$10^{13}\,M_\odot/h$",
              r"$10^{14}\,M_\odot/h$", r"$10^{15}\,M_\odot/h$"]

    from matplotlib.lines import Line2D

    fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(8, 8),
                                   gridspec_kw={"height_ratios": [2, 1]},
                                   sharex=True)

    profile_data = []
    for M, color, label in zip(masses, colors, labels):
        m_arr = jnp.array([M])
        try:
            c_val = float(np.asarray(hp.concentration(m_arr, _Z))[0])
        except TypeError:
            c_val = float(np.asarray(hp.concentration(m_arr, _Z, theta))[0])

        delta, rho_ref = hp._mdef_delta_rho(_Z, theta)
        r_delta = (3.0 * M / (4.0 * np.pi * delta * rho_ref)) ** (1.0 / 3.0)
        r_s = r_delta / c_val

        uk_nfw = np.asarray(nfw_uk(k, np.array([r_s]), np.array([c_val])))[:, 0]
        ax.semilogx(k, uk_nfw, lw=2, color=color, label=label + " NFW")

        uk_ein = None
        try:
            uk_ein = np.asarray(einasto_uk(k, np.array([r_s]), np.array([c_val])))[:, 0]
            ax.semilogx(k, uk_ein, lw=1.5, color=color, ls="--",
                        label=label + " Einasto")
        except Exception:
            pass
        profile_data.append((color, uk_nfw, uk_ein))

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(r"$\tilde{u}(k|M)$", fontsize=12)
    ax.set_title(
        rf"Halo profile Fourier transforms (solid: NFW, dashed: Einasto $\alpha=0.18$, $z={_Z}$)",
        fontsize=10)
    handles, lab = ax.get_legend_handles_labels()
    ax.legend(handles[:4], [l.replace(" NFW", "") for l in lab[:4]],
              fontsize=9, title="Halo mass")
    ax.add_artist(ax.legend(
        handles=[Line2D([0], [0], color="k", lw=2),
                 Line2D([0], [0], color="k", lw=1.5, ls="--")],
        labels=["NFW", r"Einasto $\alpha=0.18$"],
        fontsize=9, loc="upper right",
    ))
    ax.set_ylim(-0.1, 1.05)

    # Bottom: Einasto / NFW ratio for each mass
    ax_r.axhline(1, color="k", lw=0.8, ls="--")
    for (color, uk_nfw, uk_ein), label in zip(profile_data, labels):
        if uk_ein is not None:
            ratio = uk_ein / np.where(np.abs(uk_nfw) > 1e-6, uk_nfw, 1e-6)
            ax_r.semilogx(k, ratio, lw=1.5, color=color, label=label)
    ax_r.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=12)
    ax_r.set_ylabel(r"$\tilde{u}_{\rm Ein}/\tilde{u}_{\rm NFW}$", fontsize=11)
    ax_r.set_ylim(0.5, 1.4)
    ax_r.legend(fontsize=8, ncol=2)

    fig.text(0.99, 0.01,
             "Cooray & Sheth 2002 (arXiv:astro-ph/0206508); Einasto 1965; Klypin+2016 (arXiv:1711.01744)",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig04_halo_profiles.pdf", show)


# ---------------------------------------------------------------------------
# Fig 05 — HOD occupation functions: 3×3 multi-panel, all 9 models
# ---------------------------------------------------------------------------

def fig05_hod_occupation(ctx, output_dir, show=False):
    r"""N_cen(M), N_sat(M), N_total(M) for all 9 HOD/CSMF models.

    3-row × 3-column layout:
      Row 1 (halo-mass models): Zheng+2007, Kravtsov+2004, More+2015
      Row 2 (SHMR threshold / bin): Zu & Mandelbaum 2015, van Uitert+2016, Zacharegkas+2025
      Row 3 (ICSMF / quenching):   Guo+2018, Guo+2019, Zu & Mandelbaum 2016 (quenching)

    For the ZM16 quenching panel the y-axis shows the red fraction
    f_red_cen (solid) and f_red_sat (dashed) rather than N_cen/N_sat.
    """
    from hod_mod.galaxies.hod import (
        HODModel, Kravtsov04HODModel, MoreHODModel,
        Guo18ICSMFModel, Guo19ICSMFModel,
        Zacharegkas25HODModel, VanUitert16CSMFModel, ZuMandelbaum15HODModel,
        f_red_cen_zu16, f_red_sat_zu16,
    )
    import jax

    hmf = ctx["hmf"]
    log10m = np.linspace(11.0, 16.0, 600)
    log10m_jnp = jnp.asarray(log10m)

    # --- shared halo-mass model params (Mmin=12.5 so curves fit) ----------
    _hm_base = dict(log10mmin=12.5, sigma_logm=0.4, log10m1=13.5, alpha=1.0)

    panel_configs = [
        # Row 1 — halo-mass
        {
            "label": "Zheng+2007",
            "ref":   "arXiv:astro-ph/0703457",
            "hod":   HODModel(hmf, hmf.bias),
            "params": {**_hm_base, "log10m0": 11.0},
        },
        {
            "label": "Kravtsov+2004",
            "ref":   "ApJ 609, 35 (2004)",
            "hod":   Kravtsov04HODModel(hmf, hmf.bias),
            "params": {**_hm_base, "log10m0": 11.0},
        },
        {
            "label": "More+2015",
            "ref":   "arXiv:1211.6211",
            "hod":   MoreHODModel(hmf, hmf.bias),
            "params": {**_hm_base, "kappa": 1.0, "alpha_inc": 0.0, "log10m_inc": 12.0},
        },
        # Row 2 — SHMR
        {
            "label": "Zu & Mandelbaum 2015",
            "ref":   "arXiv:1505.02781",
            "hod":   ZuMandelbaum15HODModel(hmf),
            "params": ZuMandelbaum15HODModel.default_params(),
        },
        {
            "label": "van Uitert+2016",
            "ref":   "arXiv:1601.06791",
            "hod":   VanUitert16CSMFModel(hmf),
            "params": VanUitert16CSMFModel.default_params(),
        },
        {
            "label": "Zacharegkas+2025",
            "ref":   "arXiv:2506.22367",
            "hod":   Zacharegkas25HODModel(hmf),
            "params": Zacharegkas25HODModel.default_params(),
        },
        # Row 3 — ICSMF / quenching
        {
            "label": "Guo+2018 ICSMF",
            "ref":   "arXiv:1707.01922",
            "hod":   Guo18ICSMFModel(hmf, hmf.bias),
            "params": Guo18ICSMFModel.default_params(),
        },
        {
            "label": "Guo+2019 ICSMF",
            "ref":   "arXiv:1811.10583",
            "hod":   Guo19ICSMFModel(hmf, hmf.bias),
            "params": Guo19ICSMFModel.default_params(),
        },
        {
            "label": "Zu & Mandelbaum 2016 (quenching)",
            "ref":   "arXiv:1509.06374",
            "hod":   None,  # special: plot f_red directly
            "params": {"lg_mqc_h": 12.5, "mu_c": 2.0,
                       "lg_mqs_h": 13.5, "mu_s": 1.5},
        },
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)

    for ax, cfg in zip(axes.flat, panel_configs):
        label  = cfg["label"]
        params = cfg["params"]
        hod    = cfg["hod"]

        if label.startswith("Zu & Mandelbaum 2016"):
            # Special panel: red fractions
            fred_c = np.asarray(f_red_cen_zu16(
                log10m_jnp, params["lg_mqc_h"], params["mu_c"]))
            fred_s = np.asarray(f_red_sat_zu16(
                log10m_jnp, params["lg_mqs_h"], params["mu_s"]))
            ax.plot(log10m, fred_c, lw=2, color="#1f77b4",
                    label=r"$f_{\rm red,cen}$")
            ax.plot(log10m, fred_s, lw=2, color="#ff7f0e", ls="--",
                    label=r"$f_{\rm red,sat}$")
            ax.set_ylim(0, 1.1)
            ax.set_ylabel(r"$f_{\rm red}(M_h)$", fontsize=10)
        else:
            try:
                with jax.disable_jit():
                    nc, ns = hod.nc_ns(log10m_jnp, params)
                nc = np.asarray(nc)
                ns = np.asarray(ns)
                nt = nc + ns
                ax.semilogy(log10m, np.maximum(nc, 1e-5), lw=2,
                            color="#1f77b4", label=r"$\langle N_c\rangle$")
                ax.semilogy(log10m, np.maximum(ns, 1e-5), lw=2,
                            color="#ff7f0e", ls="--",
                            label=r"$\langle N_s\rangle$")
                ax.semilogy(log10m, np.maximum(nt, 1e-5), lw=2,
                            color="#2ca02c", ls=":",
                            label=r"$\langle N\rangle$")
            except Exception as exc:
                ax.text(0.5, 0.5, f"Error:\n{exc}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="red",
                        wrap=True)
            ax.set_ylim(1e-4, 30)
            ax.set_ylabel(r"$\langle N\rangle(M_h)$", fontsize=10)

        ax.set_xlabel(r"$\log_{10}(M_h\,[h^{-1}M_\odot])$", fontsize=10)
        ax.set_xlim(11, 16)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(0.98, 0.97, cfg["ref"], transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color="grey")
        ax.legend(fontsize=8)

    fig.suptitle("HOD / CSMF occupation functions — all 9 models",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "fig05_hod_occupation.pdf", show)


# ---------------------------------------------------------------------------
# Fig 05b — HOD z-evolution: n_gal and b_eff vs z for fixed HOD params
# ---------------------------------------------------------------------------

def fig05b_hod_redshift(ctx, output_dir, show=False):
    """Galaxy number density n̄_g and effective bias b_eff vs z for fixed HOD.

    HOD parameters are held fixed (More+2015 fiducial).  The HMF dn/dM(M, z)
    and bias b(M, z) evolve with redshift — this figure shows the resulting
    redshift dependence of the HOD-integrated observables.
    """
    from hod_mod.galaxies.clustering import FullHaloModelPrediction

    hmf, theta    = ctx["hmf"], ctx["theta"]
    pred          = ctx["pred"]
    hod_params    = ctx["hod_params"]

    z_arr = np.linspace(0.0, 1.5, 40)
    n_gal_z = []
    b_eff_z = []

    for z in z_arr:
        try:
            ng = pred.n_gal(float(z), theta, hod_params)
            # b_eff from pk_tables: compute it via the internal tables
            tables = pred._pk_tables_full(float(z), theta, hod_params)
            b = tables["b_eff"]
            n_gal_z.append(ng)
            b_eff_z.append(b)
        except Exception as exc:
            n_gal_z.append(np.nan)
            b_eff_z.append(np.nan)

    n_gal_z = np.array(n_gal_z)
    b_eff_z = np.array(b_eff_z)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax1.semilogy(z_arr, n_gal_z, lw=2, color="#1f77b4")
    ax1.axvline(_Z, color="k", lw=0.8, ls="--", label=rf"$z={_Z}$ (BGS)")
    ax1.set_ylabel(r"$\bar{n}_g$ [$h^3\,{\rm Mpc}^{-3}$]", fontsize=12)
    ax1.set_title("HOD redshift evolution (fixed More+2015 params)", fontsize=11)
    ax1.legend(fontsize=9)

    ax2.plot(z_arr, b_eff_z, lw=2, color="#ff7f0e")
    ax2.axvline(_Z, color="k", lw=0.8, ls="--", label=rf"$z={_Z}$ (BGS)")
    ax2.set_xlabel(r"Redshift $z$", fontsize=12)
    ax2.set_ylabel(r"$b_{\rm eff}(z)$", fontsize=12)
    ax2.legend(fontsize=9)

    fig.text(0.99, 0.01,
             "Tinker+2008 (arXiv:0803.2706); More+2015 (arXiv:1211.6211)",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig05b_hod_redshift.pdf", show)


# ---------------------------------------------------------------------------
# Fig 06 — Baryon fraction sigmoid f_b(M)
# ---------------------------------------------------------------------------

def fig06_baryon_fraction(ctx, output_dir, show=False):
    r"""Baryon fraction models: sigmoid (no floor), sigmoid + 1% floor, and double-sigmoid upturn.

    Three curves:
    1. Pure sigmoid (f_b_min=0) — dashed grey.
    2. Sigmoid + 1% floor (default) — solid blue.
    3. Double-sigmoid valley with low-mass upturn — solid orange.

    Zhang et al. 2025 (`arXiv:2511.17313 <https://arxiv.org/abs/2511.17313>`_) provides
    observational motivation for residual gas in low-mass halos.
    """
    from hod_mod.galaxies.baryon_fraction import (
        BaryonFractionSigmoid, BaryonFractionUpturn,
    )

    theta = ctx["theta"]
    m = jnp.logspace(10, 16, 400)
    m_np = np.asarray(m)
    f_b_cosmic = float(theta["Omega_b"]) / float(theta["Omega_m"])

    bf_sig = BaryonFractionSigmoid()
    bf_up  = BaryonFractionUpturn()

    p_nofoor = {"log10_M_pivot": 13.5, "beta_b": 1.5, "f_b_min": 0.0}
    p_floor  = bf_sig.default_params()  # f_b_min=0.01
    p_upturn = bf_up.default_params()

    fb_no  = np.asarray(bf_sig(m, theta, p_nofoor)) / f_b_cosmic
    fb_fl  = np.asarray(bf_sig(m, theta, p_floor))  / f_b_cosmic
    fb_up  = np.asarray(bf_up(m, theta, p_upturn))  / f_b_cosmic

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(m_np, fb_no, lw=1.8, color="0.5", ls="--",
                label=r"Sigmoid, $f_b^{\min}=0$")
    ax.semilogx(m_np, fb_fl, lw=2,   color="#1f77b4", ls="-",
                label=r"Sigmoid + 1\% floor (default)")
    ax.semilogx(m_np, fb_up, lw=2,   color="#ff7f0e", ls="-",
                label="Double-sigmoid (valley + low-$M$ upturn)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label=r"$f_b^{\rm cosmic}$")
    ax.axhline(p_floor["f_b_min"] / f_b_cosmic, color="0.4", lw=0.8, ls=":",
               label=r"1% floor")

    ax.set_xlabel(r"$M_h$ [$h^{-1}\,M_\odot$]", fontsize=12)
    ax.set_ylabel(r"$f_b(M)\,/\,f_b^{\rm cosmic}$", fontsize=12)
    ax.set_title(r"Mass-dependent baryon fraction $f_b(M)$", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.2)
    ax.set_xlim(1e10, 1e16)
    ax.text(0.98, 0.02,
            "FLAMINGO (arXiv:2510.25419); Veenema+2026 (arXiv:2603.13095); Zhang+2025 (arXiv:2511.17313)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig06_baryon_fraction.pdf", show)


# ---------------------------------------------------------------------------
# Fig 07 — Gas concentration ratio η(M)
# ---------------------------------------------------------------------------

def fig07_gas_concentration(ctx, output_dir, show=False):
    r"""Gas concentration η(M) + c_gas(M) with σ₈ cosmology dependence.

    Top panel — η(M) = c_gas/c_DM for three η_min values.
    Bottom panel — c_gas(M) = η(M) × c_DM(M) for σ₈ = 0.75, 0.811, 0.85
    with fiducial η_min = 0.6.
    """
    from hod_mod.cosmology.halo_profiles import HaloProfile

    m = np.logspace(11, 16, 300)
    m_jnp = jnp.asarray(m)
    M_eta    = 10.0 ** 13.0
    beta_eta = 1.5
    colossus_fid = ctx["colossus_cosmo"]
    theta = ctx["theta"]

    eta_mins = [0.5, 0.6, 0.8]
    colors_eta = ["#d62728", "#ff7f0e", "#1f77b4"]
    labels_eta = [r"$\eta_{\min}=0.5$  (strong feedback)",
                  r"$\eta_{\min}=0.6$  (fiducial)",
                  r"$\eta_{\min}=0.8$  (mild feedback)"]

    eta_fid = 1.0 - (1.0 - 0.6) / (1.0 + (m / M_eta) ** beta_eta)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for eta_min, color, label in zip(eta_mins, colors_eta, labels_eta):
        eta_M = 1.0 - (1.0 - eta_min) / (1.0 + (m / M_eta) ** beta_eta)
        ax1.semilogx(m, eta_M, lw=2, color=color, label=label)
    ax1.axhline(1.0, color="k", lw=0.8, ls="--", label=r"$\eta=1$")
    ax1.set_ylabel(r"$\eta(M) = c_{\rm gas}/c_{\rm DM}$", fontsize=12)
    ax1.set_title(r"Gas concentration ratio $\eta(M)$", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.4, 1.05)

    # Bottom: c_gas(M) = η_fid(M) × c_DM(M) for 3 σ₈ values
    sigma8_vals   = [0.75, _SIGMA8_FID, 0.85]
    sigma8_colors = ["#d62728", "#1f77b4", "#2ca02c"]
    sigma8_ls     = [":", "-", "--"]
    sigma8_labels = [r"$\sigma_8=0.75$", rf"$\sigma_8={_SIGMA8_FID}$ (fid.)", r"$\sigma_8=0.85$"]

    for s8, color, ls, label in zip(sigma8_vals, sigma8_colors, sigma8_ls, sigma8_labels):
        try:
            cosmo_i = {**colossus_fid, "sigma8": s8}
            hp_i = HaloProfile(cosmo_i, cm_relation="diemer19")
            try:
                c_dm = np.asarray(hp_i.concentration(m_jnp, _Z))
            except TypeError:
                c_dm = np.asarray(hp_i.concentration(m_jnp, _Z, theta))
            c_gas = eta_fid * c_dm
            ax2.loglog(m, c_gas, lw=2, color=color, ls=ls, label=label)
        except Exception as exc:
            warnings.warn(f"c_gas σ₈={s8} failed: {exc}")

    ax2.set_xlabel(r"$M_h$ [$h^{-1}\,M_\odot$]", fontsize=12)
    ax2.set_ylabel(r"$c_{\rm gas}(M) = \eta(M)\,c_{\rm DM}(M)$", fontsize=12)
    ax2.set_title(r"Gas NFW concentration: $\sigma_8$ dependence ($\eta_{\min}=0.6$)", fontsize=11)
    ax2.set_xlim(1e11, 1e16)
    ax2.legend(fontsize=9)

    fig.text(0.99, 0.01,
             "arXiv:2409.01758 Table 2; Mead+2015 (arXiv:1611.08606); Diemer+2019 (arXiv:1809.07326)",
             ha="right", va="bottom", fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig07_gas_concentration.pdf", show)


# ---------------------------------------------------------------------------
# Fig 08 — Projected correlation function w_p(r_p)
# ---------------------------------------------------------------------------

def fig08_wp(ctx, output_dir, show=False):
    r"""Projected correlation function w_p(r_p) at z=0.14, with 1h/2h decomposition."""
    pred, theta = ctx["pred"], ctx["theta"]
    hod_params  = ctx["hod_params"]

    rp = jnp.logspace(-1, 1.5, 40)
    comps = pred.wp_components(rp, 60.0, _Z, theta, hod_params)
    rp_np = np.asarray(rp)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(rp_np, np.asarray(comps["total"]), lw=2, color="#1f77b4",
              label=rf"$w_p$ total (1h + 2h)")
    ax.loglog(rp_np, np.asarray(comps["1h"]), lw=2, color="#ff7f0e", ls="--",
              label=r"$w_p^{\rm 1h}$")
    ax.loglog(rp_np, np.asarray(comps["2h"]), lw=2, color="#2ca02c", ls=":",
              label=r"$w_p^{\rm 2h}$")
    ax.set_xlabel(r"$r_p$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_ylabel(r"$w_p(r_p)$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_title(rf"Projected correlation function $w_p(r_p)$, $z={_Z}$", fontsize=11)
    ax.legend(fontsize=10)
    ax.text(0.98, 0.02,
            "More et al. 2015 (arXiv:1211.6211) Eqs. 9–10",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig08_wp.pdf", show)


# ---------------------------------------------------------------------------
# Fig 09 — ΔΣ(R) without baryon correction
# ---------------------------------------------------------------------------

def fig09_delta_sigma(ctx, output_dir, show=False):
    r"""Excess surface mass density ΔΣ(R) at z=0.14, with 1h/2h decomposition."""
    pred, theta = ctx["pred"], ctx["theta"]
    hod_params  = ctx["hod_params"]

    R = jnp.logspace(-1, 1.2, 30)
    comps = pred.delta_sigma_components(R, _Z, theta, hod_params)
    R_np = np.asarray(R)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(R_np, np.asarray(comps["total"]), lw=2, color="#1f77b4",
              label=r"$\Delta\Sigma$ total (1h + 2h)")
    ax.loglog(R_np, np.asarray(comps["1h"]), lw=2, color="#ff7f0e", ls="--",
              label=r"$\Delta\Sigma^{\rm 1h}$")
    ax.loglog(R_np, np.asarray(comps["2h"]), lw=2, color="#2ca02c", ls=":",
              label=r"$\Delta\Sigma^{\rm 2h}$")
    ax.set_xlabel(r"$R$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]", fontsize=12)
    ax.set_title(rf"Excess surface mass density $\Delta\Sigma(R)$, $z={_Z}$", fontsize=11)
    ax.legend(fontsize=10)
    ax.text(0.98, 0.02,
            "More et al. 2015 (arXiv:1211.6211) Eq. 13",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig09_delta_sigma.pdf", show)


# ---------------------------------------------------------------------------
# Fig 10 — CDM/baryon split of ΔΣ
# ---------------------------------------------------------------------------

def fig10_cdm_baryon_split(ctx, output_dir, show=False):
    r"""ΔΣ_CDM(R), ΔΣ_b(R), ΔΣ_total(R) at z=0.14."""
    pred, theta = ctx["pred"], ctx["theta"]
    hod_params  = ctx["hod_params"]

    R  = jnp.logspace(-1, 1.2, 30)
    bp = {
        "log10_M_pivot": 13.5, "beta_b": 1.5,
        "log10_eta_min": -0.22,
        "log10_M_eta":   13.0,
    }
    result   = pred.delta_sigma_split(R, _Z, theta, hod_params, baryon_params=bp)
    R_np     = np.asarray(R)
    ds_cdm   = np.asarray(result["cdm"])
    ds_b     = np.asarray(result["b"])
    ds_total = np.asarray(result["total"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(R_np, ds_total, lw=2, color="k",
              label=r"$\Delta\Sigma_{\rm total}$")
    ax.loglog(R_np, ds_cdm,   lw=2, ls="--", color="#1f77b4",
              label=r"$\Delta\Sigma_{\rm CDM}$")
    ax.loglog(R_np, ds_b,     lw=2, ls=":",  color="#d62728",
              label=r"$\Delta\Sigma_b$ (gas, suppressed)")
    ax.set_xlabel(r"$R$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]", fontsize=12)
    ax.set_title(rf"CDM / baryon split of $\Delta\Sigma$, $z={_Z}$", fontsize=11)
    ax.legend(fontsize=10)
    ax.text(0.98, 0.02,
            "arXiv:2409.01758; arXiv:2510.25419; arXiv:2603.13095; Mead+2015 (arXiv:1611.08606)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig10_cdm_baryon_split.pdf", show)


# ---------------------------------------------------------------------------
# Fig 11 — IA contribution to ΔΣ
# ---------------------------------------------------------------------------

def fig11_ia_delta_sigma(ctx, output_dir, show=False):
    r"""ΔΣ(R) with NLA intrinsic-alignment correction at z=0.14."""
    pred, nla   = ctx["pred"], ctx["nla"]
    theta       = ctx["theta"]
    hod_params  = ctx["hod_params"]

    R    = jnp.logspace(-1, 1.2, 30)
    R_np = np.asarray(R)

    A_IA_values = [0.0, 0.5, 1.0, 2.0]
    colors      = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for A_IA, color in zip(A_IA_values, colors):
        ia_params = {"A_IA": A_IA, "eta_IA": 0.0} if A_IA > 0 else None
        ds = np.asarray(
            pred.delta_sigma(
                R, _Z, theta, hod_params,
                ia_model=nla if A_IA > 0 else None,
                ia_params=ia_params,
            )
        )
        ax.loglog(R_np, np.maximum(ds, 1e-5), lw=2, color=color,
                  label=rf"$A_{{\rm IA}}={A_IA}$")

    ax.set_xlabel(r"$R$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]", fontsize=12)
    ax.set_title(rf"$\Delta\Sigma$ with NLA IA correction, $z={_Z}$", fontsize=11)
    ax.legend(fontsize=10)
    ax.text(0.98, 0.02,
            "Bridle & King 2007 (arXiv:0705.0166); DESI KP6 (arXiv:2512.02954)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig11_ia_delta_sigma.pdf", show)


# ---------------------------------------------------------------------------
# Fig 12 — wp summary: all corrections overlaid
# ---------------------------------------------------------------------------

def fig12_wp_summary(ctx, output_dir, show=False):
    r"""w_p(r_p) summary: baseline 1h+2h, 1h alone, 2h alone, + off-centering.

    Shows the effect of off-centering (Johnston+2007 arXiv:0709.4193;
    More+2015 arXiv:1211.6211 §3.3) on w_p at z=0.14.
    """
    pred, theta = ctx["pred"], ctx["theta"]
    hod_params  = ctx["hod_params"]

    rp = jnp.logspace(-1, 1.5, 40)
    rp_np = np.asarray(rp)

    # Baseline (no off-centering)
    comps = pred.wp_components(rp, 60.0, _Z, theta, hod_params)

    # Off-centered: f_off = 0.15, sigma_off = 0.2 Mpc/h
    hod_off = {**hod_params, "f_off": 0.15, "sigma_off": 0.2}
    wp_off = np.asarray(pred.wp(rp, 60.0, _Z, theta, hod_off))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(rp_np, np.asarray(comps["total"]), lw=2.5, color="#1f77b4",
              label="Total 1h + 2h (baseline)")
    ax.loglog(rp_np, np.asarray(comps["1h"]),    lw=1.8, color="#ff7f0e", ls="--",
              label=r"1-halo only $w_p^{\rm 1h}$")
    ax.loglog(rp_np, np.asarray(comps["2h"]),    lw=1.8, color="#2ca02c", ls=":",
              label=r"2-halo only $w_p^{\rm 2h}$")
    ax.loglog(rp_np, wp_off,                     lw=2,   color="#9467bd", ls="-.",
              label=r"+ off-centering ($f_{\rm off}=0.15$, $\sigma=0.2\,h^{-1}$Mpc)")

    ax.set_xlabel(r"$r_p$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_ylabel(r"$w_p(r_p)$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_title(rf"$w_p$ summary — all corrections ($z={_Z}$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.98, 0.02,
            "More+2015 (arXiv:1211.6211); Johnston+2007 (arXiv:0709.4193)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig12_wp_summary.pdf", show)


# ---------------------------------------------------------------------------
# Fig 13 — ΔΣ summary: CDM/baryon split + 1h/2h + IA
# ---------------------------------------------------------------------------

def fig13_ds_summary(ctx, output_dir, show=False):
    r"""ΔΣ(R) summary: 1h/2h, CDM/baryon split, and NLA IA correction.

    Four curves:
    1. ΔΣ total (baseline, 1h + 2h).
    2. ΔΣ CDM contribution (1h + 2h, baryon-split).
    3. ΔΣ gas / baryon contribution.
    4. ΔΣ total + NLA IA (A_IA = 0.5).
    """
    pred, nla   = ctx["pred"], ctx["nla"]
    theta       = ctx["theta"]
    hod_params  = ctx["hod_params"]

    R  = jnp.logspace(-1, 1.2, 30)
    R_np = np.asarray(R)

    bp = {
        "log10_M_pivot": 13.5, "beta_b": 1.5, "f_b_min": 0.01,
        "log10_eta_min": -0.22, "log10_M_eta": 13.0,
    }

    # 1h/2h split
    comps = pred.delta_sigma_components(R, _Z, theta, hod_params)

    # CDM/baryon split
    split = pred.delta_sigma_split(R, _Z, theta, hod_params, baryon_params=bp)

    # + NLA IA
    ds_ia = np.asarray(
        pred.delta_sigma(R, _Z, theta, hod_params,
                         ia_model=nla, ia_params={"A_IA": 0.5, "eta_IA": 0.0})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(R_np, np.asarray(comps["total"]), lw=2.5, color="#1f77b4",
              label=r"$\Delta\Sigma$ total (1h + 2h)")
    ax.loglog(R_np, np.asarray(split["cdm"]),   lw=2,   color="#ff7f0e", ls="--",
              label=r"$\Delta\Sigma_{\rm CDM}$")
    ax.loglog(R_np, np.maximum(np.asarray(split["b"]), 1e-5),
              lw=2, color="#d62728", ls=":",
              label=r"$\Delta\Sigma_b$ (gas)")
    ax.loglog(R_np, np.maximum(ds_ia, 1e-5), lw=2, color="#9467bd", ls="-.",
              label=r"Total + NLA IA ($A_{\rm IA}=0.5$)")

    ax.set_xlabel(r"$R$ [$h^{-1}\,{\rm Mpc}$]", fontsize=12)
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]", fontsize=12)
    ax.set_title(rf"$\Delta\Sigma$ summary — all corrections ($z={_Z}$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.98, 0.02,
            "More+2015 (arXiv:1211.6211); arXiv:2409.01758; Bridle & King 2007 (arXiv:0705.0166)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey")
    plt.tight_layout()
    _save(fig, output_dir, "fig13_ds_summary.pdf", show)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig, output_dir, filename, show):
    if show:
        plt.show()
    else:
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        png_path = os.path.splitext(path)[0] + ".png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}  +  {os.path.basename(png_path)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", default="results/showcase/",
        help="Output directory for PDF/PNG figures.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display interactively instead of saving to files.",
    )
    args = parser.parse_args()

    if not args.show:
        os.makedirs(args.output, exist_ok=True)

    print("Building forward-model components …")
    ctx = _setup()
    print("  ✓ Cosmology, HMF, HOD, baryon fraction, NLA model ready.")
    print(f"\nGenerating 18 showcase figures → {args.output}")

    figures = [
        fig01_power_spectrum,
        fig01b_nonlinear_power_spectrum,
        fig02_hmf,
        fig02a_hmf_models,
        fig02b_bias_models,
        fig03_concentration,
        fig04_halo_profiles,
        fig05_hod_occupation,
        fig05b_hod_redshift,
        fig06_baryon_fraction,
        fig07_gas_concentration,
        fig08_wp,
        fig09_delta_sigma,
        fig10_cdm_baryon_split,
        fig11_ia_delta_sigma,
        fig12_wp_summary,
        fig13_ds_summary,
    ]

    for fn in figures:
        print(f"  [{fn.__name__}] …", end=" ", flush=True)
        try:
            fn(ctx, args.output, show=args.show)
            print("done")
        except Exception as exc:
            warnings.warn(f"\n  WARNING: {fn.__name__} failed: {exc}")


if __name__ == "__main__":
    main()
