r"""Fiducial parameters, labels and priors for the sensitivity forecast.

Fiducials are the on-disk campaign best fits:

* cosmology — Planck 2018 (``LinearPowerSpectrum.default_cosmology`` + σ8);
* HOD — ``bgs_zm15_joint/map_result.json`` (BGS M*>10 ZM15 MAP, 9 free);
* X-ray gas / AGN — ``xray_joint_bands/S1_bands_summary.json`` (S1 MAP).

If those JSON files are not found (e.g. ``$HOD_MOD_RESULTS`` unset), hard-coded
values from the plan are used so the module always imports.
"""

from __future__ import annotations

import json
import os

import numpy as np

from hod_mod.forecast.forward_jax import (
    PARAM_NAMES,
    _IDX,
    N_PARAM,
    TIER2_PROMOTED,
    TIER2_ZSLOPES,
    TIER2_EXTENSION,
    TIER3_EXTENSION,
)

# --- hard-coded fallbacks (values read from the MAP JSONs, see module docstring)
_FIDUCIAL_DEFAULT = {
    "Omega_m": 0.3100, "sigma8": 0.8111, "h": 0.6736, "n_s": 0.9649, "Omega_b": 0.0493,
    "lg_m1h": 11.67701, "lg_m0star": 10.47679, "beta": 0.734685,
    "delta": 0.661630, "gamma": 0.540446, "sigma_lnmstar": 1.024268,
    "eta": -0.314724, "fc": 0.300001, "bsat": 36.37306,
    "lx_norm": 45.1151, "lx_slope": 1.47861, "kt_norm": 1.12745,
    "kt_slope": 0.378256, "p2": 0.167427, "r_max": 3.732144,
    "log10DC": -0.9088, "beta_pressure": 0.0,
    "log10_M_pivot": 13.5, "log10_eta_min": -0.22,   # BaryonFractionSigmoid defaults
    # Powell AGN-XLF sector — powell_agn/powell_map_aird15.json (MAP fit values).
    "agn_mu_bh": 7.5184, "agn_al_bh": 0.8037, "agn_sig_bh": 0.2067,
    "agn_log10_lstar": -1.0120, "agn_delta1": 0.3123, "agn_delta2": 2.4761,
    "agn_log10_ferdf": -1.7243,
    # --- tier-2 promoted nuisances: fiducial == the former fixed constant, so
    # the fiducial prediction is bit-identical to the tier-1 model.
    "beta_sat": 0.9, "bcut": 0.86, "beta_cut": 0.41, "alpha_sat": 1.0,   # _FIXED_HOD
    "beta_b": 1.5, "log10_M_eta": 13.0, "beta_eta": 1.5,                 # _FIXED_BARYON
    "alpha_in_gas": 0.9, "alpha_tr_gas": 2.0,                            # gas emissivity slopes
    "p03_pressure": 1.627e-6, "c_dpm_pressure": 2.772, "alpha_in_pressure": 0.3,
    "alpha_tr_pressure": 1.3, "alpha_out_pressure": 4.1,                 # DPM Model 2 GNFW
    "agn_rho": 0.0, "agn_sig_mstar": 0.20,                               # Powell Model 2 internals
    # --- tier-2 z-evolution slopes (per ln(1+z), pivot z=0.3): fiducial 0 =
    # the tier-1 no-evolution model (self-similar E(z) scalings untouched).
    "lg_m1h_zs": 0.0, "lg_m0star_zs": 0.0, "sigma_lnmstar_zs": 0.0,
    "lx_zs": 0.0, "kt_zs": 0.0,
    "agn_log10_ferdf_zs": 0.0, "agn_log10_lstar_zs": 0.0,
    # --- tier-2 X-ray spectral sector (multi-band APEC layer).  Fiducials
    # reproduce the production conventions: isothermal halos (t_prof_slope=0),
    # Z=0.3 Z_sun (DPM normalisation, gas/metallicity.py), Γ=1.8 AGN power law
    # (fit_xray_joint_bands._GAMMA_AGN), ~30% obscured at NH=1e22.
    "t_prof_slope": 0.0, "z_gas_norm": 0.3, "z_gas_mslope": 0.0, "z_gas_zs": 0.0,
    "agn_gamma": 1.8, "agn_fabs": 0.3, "agn_mu_bh_zs": 0.0,
    # --- missing-physics extension (docs/missing_physics.rst).  Fiducials
    # preserve the tier-2 predictions: eps_sn = the former _EPS_SN constant;
    # (w0, wa, sum_mnu) = the ΛCDM/massless limit (the CPL growth ODE and the
    # ν suppression are exactly their tier-2 values there); the quenching and
    # FP/HI parameters only touch the new opt-in observables.
    "eps_sn": 0.1,
    "w0": -1.0, "wa": 0.0, "sum_mnu": 0.0,
    # ZM16 halo-quenching MAP (Zu & Mandelbaum 2016, the in-repo fiducials)
    "log10_Mq_cen": 11.78, "mu_q_cen": 0.41,
    "log10_Mq_sat": 12.19, "mu_q_sat": 0.24,
    "dlx_quenched": 0.0,
    # fundamental plane of BH activity (Merloni+03 fiducials, Gültekin+19 priors)
    "agn_xi_rx": 0.60, "agn_xi_rm": 0.78, "agn_b_r": 7.33, "agn_sig_r": 0.88,
    # M_HI(M_h) halo model (Villaescusa-Navarro+2018, z=0)
    "log10_M0_hi": 10.63, "log10_Mmin_hi": 12.3, "alpha_hi": 0.24,
    # --- wave 2: SN wind loading (η_0 = 0 ⇒ exactly the tier-2 η(M) sigmoid),
    # Speagle+14-like main sequence, quenched-HI deficit.
    "eta_w_norm": 0.0, "alpha_w": 1.0,
    "ssfr_ms_norm": -9.9, "ssfr_ms_slope": -0.25, "ssfr_ms_zs": 0.0,
    "dhi_quenched": 0.0,
    # --- wave 3: continuous sSFR distribution, [OII] calibration, radio-loud
    # jets (Best & Heckman-like) and the AGN IR bolometric correction.
    "sigma_ms": 0.30, "dssfr_q": -1.6,
    "loii_norm": 40.85,           # log10 L[OII]/SFR (Kennicutt-like)
    "f_loud0": 0.01, "beta_loud": 2.0, "b_jet": 41.0,
    "agn_bc_ir": -0.52,           # L_IR(6um) ≈ 0.3 L_bol
    # --- tier 3: multi-wavelength SED calibrations (radio/IR maps, band LFs).
    "l14_sfr": 28.20, "alpha_syn": 0.70,     # radio-FIR (Murphy+11) + synchrotron slope
    "lir_sfr": 43.41, "bir_color": 0.0,      # L_IR/SFR (Kennicutt-Murphy) + dust color tilt
    "ml_nir": 33.20, "ml_opt": 33.10,        # log10 nuLnu/M* (3.4um; r band, SF)
    "dopt_q": -0.20,                         # quiescent r-band offset [dex]
    "luv_norm": 42.65, "tau_uv_mslope": 0.0,  # UV/SFR incl. attenuation + its M* slope
    "lha_norm": 41.27,                       # log10 L_Halpha/SFR (Kennicutt)
    "agn_bc_uv": -0.62, "agn_bc_opt": -0.72,  # AGN bolometric corrections (1450A, 4400A)
    # --- wave 4: conditional galaxy morphology (early-type Weibull) + the
    # BH-bulge coupling (0 fiducial = the pure Powell chain).
    "log10_M_morph": 12.5, "beta_morph": 0.8,
    "f_morph_sat": 0.2,
    "mbh_bt_slope": 0.0,
    # --- tier 4: morphology observables (joint E/Q, sizes, IA).
    "rho_morph_q": 0.0,           # 0 = the wave-4 independence, exact
    "log10_f_size": -1.824,       # log10(R_e/R_200c), Kravtsov 2013 0.015
    "dsize_early": -0.20,         # early-type size offset [dex] (van der Wel)
    "f_size_zs": 0.0,
    "a_ia": 2.0,                  # NLA amplitude of early types
}

# Planck 2018 1σ priors (planck_prior.PLANCK18_SIGMAS) on the free cosmo params.
# Omega_b width is the BBN prior (Cooke+2018), tighter than Planck alone.
PLANCK_PRIOR_SIGMA = {"Omega_m": 0.0073, "sigma8": 0.0060,
                      "h": 0.0054, "n_s": 0.0042, "Omega_b": 0.0005}

# Weakly-informative *regularizing* priors (plausible parameter ranges).  They
# bound the near-flat / degenerate nuisance directions so the Fisher matrix is
# positive-definite and marginalised errors are well-defined and monotonic
# (adding data or fixing a parameter can then only tighten σ, never loosen it).
# Broad enough not to constrain the data-measured parameters.
BROAD_PRIOR_SIGMA = {
    "Omega_m": 0.3, "sigma8": 0.3,                      # ~uninformative on the targets
    # h, n_s, Omega_b are always externally calibrated (Planck/BBN); they are near-flat
    # directions for these low-z small-scale probes, so they carry their Planck/BBN
    # prior by default (differentiable and loosenable, but not spuriously free).
    "h": 0.0054, "n_s": 0.0042, "Omega_b": 0.0005,
    "lg_m1h": 2.0, "lg_m0star": 2.0, "beta": 5.0, "delta": 5.0, "gamma": 5.0,
    "sigma_lnmstar": 2.0, "eta": 5.0, "fc": 1.0, "bsat": 100.0,
    "lx_norm": 3.0, "lx_slope": 2.0, "kt_norm": 2.0, "kt_slope": 2.0,
    "p2": 0.5, "r_max": 3.0, "log10DC": 3.0, "beta_pressure": 2.0,
    "log10_M_pivot": 1.5, "log10_eta_min": 0.7,
    # Powell AGN-XLF sector — Gaussian priors from Powell 2022 / Ananna 2022.
    "agn_mu_bh": 0.30, "agn_al_bh": 0.24, "agn_sig_bh": 0.18,
    "agn_log10_lstar": 0.20, "agn_delta1": 0.15, "agn_delta2": 0.66,
    "agn_log10_ferdf": 1.0,
    # Tier-2 promoted nuisances — plausible-range widths (regularizing only).
    "beta_sat": 1.0, "bcut": 5.0, "beta_cut": 1.0, "alpha_sat": 0.5,
    "beta_b": 1.0, "log10_M_eta": 1.0, "beta_eta": 1.0,
    "alpha_in_gas": 0.5, "alpha_tr_gas": 1.0,
    "p03_pressure": 1.627e-6, "c_dpm_pressure": 1.4, "alpha_in_pressure": 0.3,
    "alpha_tr_pressure": 0.6, "alpha_out_pressure": 2.0,
    "agn_rho": 0.5, "agn_sig_mstar": 0.15,
    # Tier-2 z-evolution slopes — broad (an O(1) change per e-fold of 1+z).
    "lg_m1h_zs": 2.0, "lg_m0star_zs": 2.0, "sigma_lnmstar_zs": 2.0,
    "lx_zs": 3.0, "kt_zs": 2.0,
    "agn_log10_ferdf_zs": 3.0, "agn_log10_lstar_zs": 2.0,
    # Tier-2 X-ray spectral sector.
    "t_prof_slope": 0.5, "z_gas_norm": 0.3, "z_gas_mslope": 0.5, "z_gas_zs": 2.0,
    "agn_gamma": 0.3, "agn_fabs": 0.3, "agn_mu_bh_zs": 2.0,
    # Missing-physics extension.
    "eps_sn": 0.3,
    "w0": 1.0, "wa": 2.0, "sum_mnu": 0.25,
    "log10_Mq_cen": 1.0, "mu_q_cen": 1.0, "log10_Mq_sat": 1.0, "mu_q_sat": 1.0,
    "dlx_quenched": 1.0,
    "agn_xi_rx": 0.3, "agn_xi_rm": 0.3, "agn_b_r": 2.0, "agn_sig_r": 0.3,
    "log10_M0_hi": 1.0, "log10_Mmin_hi": 1.0, "alpha_hi": 0.3,
    # wave 2
    "eta_w_norm": 0.5, "alpha_w": 0.7,
    "ssfr_ms_norm": 0.5, "ssfr_ms_slope": 0.3, "ssfr_ms_zs": 2.0,
    "dhi_quenched": 1.0,
    # wave 3
    "sigma_ms": 0.15, "dssfr_q": 0.5, "loii_norm": 0.3,
    "f_loud0": 0.05, "beta_loud": 1.5, "b_jet": 2.0,
    "agn_bc_ir": 0.3,
    # tier 3
    "l14_sfr": 0.2, "alpha_syn": 0.15, "lir_sfr": 0.2, "bir_color": 0.5,
    "ml_nir": 0.3, "ml_opt": 0.3, "dopt_q": 0.3,
    "luv_norm": 0.3, "tau_uv_mslope": 0.3, "lha_norm": 0.2,
    "agn_bc_uv": 0.3, "agn_bc_opt": 0.3,
    # wave 4
    "log10_M_morph": 1.0, "beta_morph": 0.5,
    "f_morph_sat": 0.3, "mbh_bt_slope": 0.5,
    # tier 4
    "rho_morph_q": 0.3, "log10_f_size": 0.3, "dsize_early": 0.3,
    "f_size_zs": 2.0, "a_ia": 1.0,
}

PARAM_LATEX = {
    "Omega_m": r"$\Omega_m$", "sigma8": r"$\sigma_8$",
    "h": r"$h$", "n_s": r"$n_s$", "Omega_b": r"$\Omega_b$",
    "lg_m1h": r"$\lg M_{1}$", "lg_m0star": r"$\lg M_{0*}$",
    "beta": r"$\beta$", "delta": r"$\delta$", "gamma": r"$\gamma$",
    "sigma_lnmstar": r"$\sigma_{\ln M_*}$", "eta": r"$\eta$",
    "fc": r"$f_c$", "bsat": r"$b_{\rm sat}$",
    "lx_norm": r"$L_X^{\rm norm}$", "lx_slope": r"$L_X^{\rm slope}$",
    "kt_norm": r"$kT^{\rm norm}$", "kt_slope": r"$kT^{\rm slope}$",
    "p2": r"$p_2$", "r_max": r"$r_{\rm max}$",
    "log10DC": r"$\log_{10}{\rm DC}$", "beta_pressure": r"$\beta_P$",
    "log10_M_pivot": r"$\log_{10}M_{\rm pivot}$", "log10_eta_min": r"$\log_{10}\eta_{\min}$",
    "agn_mu_bh": r"$\mu_{\rm BH}$", "agn_al_bh": r"$\alpha_{\rm BH}$",
    "agn_sig_bh": r"$\sigma_{\rm BH}$", "agn_log10_lstar": r"$\log_{10}\lambda_*$",
    "agn_delta1": r"$\delta_1$", "agn_delta2": r"$\delta_2$",
    "agn_log10_ferdf": r"$\log_{10}f_{\rm ERDF}$",
    # Tier-2 promoted nuisances.
    "beta_sat": r"$\beta_{\rm sat}$", "bcut": r"$b_{\rm cut}$",
    "beta_cut": r"$\beta_{\rm cut}$", "alpha_sat": r"$\alpha_{\rm sat}$",
    "beta_b": r"$\beta_b$", "log10_M_eta": r"$\log_{10}M_\eta$",
    "beta_eta": r"$\beta_\eta$",
    "alpha_in_gas": r"$\alpha_{\rm in}$", "alpha_tr_gas": r"$\alpha_{\rm tr}$",
    "p03_pressure": r"$P_{0.3}$", "c_dpm_pressure": r"$c_{\rm DPM}$",
    "alpha_in_pressure": r"$\alpha_{\rm in}^P$", "alpha_tr_pressure": r"$\alpha_{\rm tr}^P$",
    "alpha_out_pressure": r"$\alpha_{\rm out}^P$",
    "agn_rho": r"$\rho_{\rm BH}$", "agn_sig_mstar": r"$\sigma_{M_*}^{\rm BH}$",
    # Tier-2 z-evolution slopes.
    "lg_m1h_zs": r"$\partial_z\lg M_1$", "lg_m0star_zs": r"$\partial_z\lg M_{0*}$",
    "sigma_lnmstar_zs": r"$\partial_z\sigma_{\ln M_*}$",
    "lx_zs": r"$\partial_z L_X^{\rm norm}$", "kt_zs": r"$\partial_z kT^{\rm norm}$",
    "agn_log10_ferdf_zs": r"$\partial_z\log_{10}f_{\rm ERDF}$",
    "agn_log10_lstar_zs": r"$\partial_z\log_{10}\lambda_*$",
    # Tier-2 X-ray spectral sector.
    "t_prof_slope": r"$\Gamma_T$", "z_gas_norm": r"$Z_{\rm gas}$",
    "z_gas_mslope": r"$\partial_M Z$", "z_gas_zs": r"$\partial_z Z$",
    "agn_gamma": r"$\Gamma_{\rm AGN}$", "agn_fabs": r"$f_{\rm abs}$",
    "agn_mu_bh_zs": r"$\partial_z\mu_{\rm BH}$",
    # Missing-physics extension.
    "eps_sn": r"$\varepsilon_{\rm SN}$",
    "w0": r"$w_0$", "wa": r"$w_a$", "sum_mnu": r"$\Sigma m_\nu$",
    "log10_Mq_cen": r"$\log_{10}M_{\rm q}^{\rm cen}$", "mu_q_cen": r"$\mu_{\rm q}^{\rm cen}$",
    "log10_Mq_sat": r"$\log_{10}M_{\rm q}^{\rm sat}$", "mu_q_sat": r"$\mu_{\rm q}^{\rm sat}$",
    "dlx_quenched": r"$\Delta L_X^{\rm Q}$",
    "agn_xi_rx": r"$\xi_{RX}$", "agn_xi_rm": r"$\xi_{RM}$",
    "agn_b_r": r"$b_R$", "agn_sig_r": r"$\sigma_R$",
    "log10_M0_hi": r"$\log_{10}M_0^{\rm HI}$",
    "log10_Mmin_hi": r"$\log_{10}M_{\min}^{\rm HI}$", "alpha_hi": r"$\alpha_{\rm HI}$",
    # wave 2
    "eta_w_norm": r"$\eta_{w,0}$", "alpha_w": r"$\alpha_w$",
    "ssfr_ms_norm": r"${\rm sSFR}_{\rm MS}$", "ssfr_ms_slope": r"$\partial_M{\rm sSFR}$",
    "ssfr_ms_zs": r"$\partial_z{\rm sSFR}$", "dhi_quenched": r"$\Delta M_{\rm HI}^{\rm Q}$",
    # wave 3
    "sigma_ms": r"$\sigma_{\rm MS}$", "dssfr_q": r"$\Delta_{\rm Q}$",
    "loii_norm": r"$\log_{10}L_{\rm [OII]}/{\rm SFR}$",
    "f_loud0": r"$f_{\rm loud,0}$", "beta_loud": r"$\beta_{\rm loud}$",
    "b_jet": r"$b_{\rm jet}$", "agn_bc_ir": r"${\rm BC}_{\rm IR}$",
    # tier 3
    "l14_sfr": r"$\log_{10}L_{1.4}/{\rm SFR}$", "alpha_syn": r"$\alpha_{\rm syn}$",
    "lir_sfr": r"$\log_{10}L_{\rm IR}/{\rm SFR}$", "bir_color": r"$b_{\rm IR}$",
    "ml_nir": r"$\Upsilon_{\rm NIR}$", "ml_opt": r"$\Upsilon_{\rm opt}$",
    "dopt_q": r"$\Delta_{\rm opt}^{\rm Q}$",
    "luv_norm": r"$\log_{10}L_{\rm UV}/{\rm SFR}$",
    "tau_uv_mslope": r"$\partial_M\tau_{\rm UV}$",
    "lha_norm": r"$\log_{10}L_{{\rm H}\alpha}/{\rm SFR}$",
    "agn_bc_uv": r"${\rm BC}_{\rm UV}$", "agn_bc_opt": r"${\rm BC}_{\rm opt}$",
    # wave 4
    "log10_M_morph": r"$\log_{10}M_{\rm morph}$",
    "beta_morph": r"$\beta_{\rm morph}$",
    "f_morph_sat": r"$f_{\rm morph}^{\rm sat}$",
    "mbh_bt_slope": r"$\partial_{B/T}M_{\rm BH}$",
    # tier 4
    "rho_morph_q": r"$\rho_{\rm morph,Q}$",
    "log10_f_size": r"$\log_{10}R_e/R_{200}$",
    "dsize_early": r"$\Delta_{\rm size}^{\rm E}$",
    "f_size_zs": r"$\partial_z R_e/R_{200}$",
    "a_ia": r"$A_{\rm IA}$",
}

# Parameter sectors for the tier-2 cosmology-vs-astrophysics decomposition
# (run_tier2_forecast): σ(sector) marginalised vs other-sectors-pinned, probe
# attribution, and information gain per sector.  Covers all of PARAM_NAMES.
SECTORS = {
    "cosmology": ["Omega_m", "sigma8", "h", "n_s", "Omega_b",
                  "w0", "wa", "sum_mnu"],
    "shmr": ["lg_m1h", "lg_m0star", "beta", "delta", "gamma", "sigma_lnmstar",
             "eta", "fc", "bsat", "beta_sat", "bcut", "beta_cut", "alpha_sat",
             "lg_m1h_zs", "lg_m0star_zs", "sigma_lnmstar_zs"],
    "gas": ["lx_norm", "lx_slope", "kt_norm", "kt_slope", "p2", "r_max",
            "alpha_in_gas", "alpha_tr_gas", "beta_pressure", "p03_pressure",
            "c_dpm_pressure", "alpha_in_pressure", "alpha_tr_pressure",
            "alpha_out_pressure", "lx_zs", "kt_zs",
            "t_prof_slope", "z_gas_norm", "z_gas_mslope", "z_gas_zs"],
    "baryon": ["log10_M_pivot", "log10_eta_min", "beta_b", "log10_M_eta",
               "beta_eta", "eps_sn", "eta_w_norm", "alpha_w"],
    "agn": ["agn_mu_bh", "agn_al_bh", "agn_sig_bh", "agn_log10_lstar",
            "agn_delta1", "agn_delta2", "agn_log10_ferdf", "agn_rho",
            "agn_sig_mstar", "agn_log10_ferdf_zs", "agn_log10_lstar_zs",
            "agn_gamma", "agn_fabs", "agn_mu_bh_zs",
            "agn_xi_rx", "agn_xi_rm", "agn_b_r", "agn_sig_r",
            "f_loud0", "beta_loud", "b_jet", "agn_bc_ir",
            "agn_bc_uv", "agn_bc_opt"],
    # SF/quiescent split (ZM16 halo quenching, the quenched-CGM offset, and
    # the star-forming main sequence)
    "sfq": ["log10_Mq_cen", "mu_q_cen", "log10_Mq_sat", "mu_q_sat",
            "dlx_quenched", "ssfr_ms_norm", "ssfr_ms_slope", "ssfr_ms_zs",
            "sigma_ms", "dssfr_q", "loii_norm",
            # tier 3: SED calibrations tied to SFR / M* (radio, IR, UV, opt, Halpha)
            "l14_sfr", "alpha_syn", "lir_sfr", "bir_color",
            "ml_nir", "ml_opt", "dopt_q",
            "luv_norm", "tau_uv_mslope", "lha_norm"],
    # cold gas / neutral hydrogen (VN18 halo model + quenched deficit)
    "coldgas": ["log10_M0_hi", "log10_Mmin_hi", "alpha_hi", "dhi_quenched"],
    # galaxy morphology (wave 4): early-type Weibull + the BH-bulge link
    "morphology": ["log10_M_morph", "beta_morph", "f_morph_sat",
                   "mbh_bt_slope",
                   # tier 4: joint E/Q correlation, sizes, IA
                   "rho_morph_q", "log10_f_size", "dsize_early",
                   "f_size_zs", "a_ia"],
    # retired: with agn_emission="powell" the duty cycle leaves the emissivity;
    # its only remaining consumer is the (off-by-default) energy-closure mode.
    "retired": ["log10DC"],
}


def _results_root() -> str:
    try:
        from hod_mod import paths
        return os.fspath(paths.results_root())
    except Exception:
        return os.environ.get("HOD_MOD_RESULTS", "")


_WARNED_NATIVE_BANDS = False


def _warn_native_band_summary(missing):
    """Warn once that the band summary cannot feed the forecast fiducial."""
    global _WARNED_NATIVE_BANDS
    if _WARNED_NATIVE_BANDS:
        return
    _WARNED_NATIVE_BANDS = True
    import warnings
    warnings.warn(
        "xray_joint_bands/S1_bands_summary.json does not carry "
        f"{missing} -- it is a native-DPM band fit, whose gas parameters are "
        "(log10_ne03, beta_n, log10_p03, beta_P). The forecast's gas sector is "
        "still the phenomenological power law, so the whole X-ray block is left "
        "at _FIDUCIAL_DEFAULT rather than mixing the two parametrisations. To "
        "use the native fit, project its MAP through "
        "hod_mod.fitting.dpm_priors.scaling_map and write the resulting "
        "(lx_norm, lx_slope, kt_norm, kt_slope) into the summary.",
        RuntimeWarning, stacklevel=3)


def load_fiducial() -> dict:
    """Return the fiducial parameter dict, preferring the on-disk MAP JSONs."""
    fid = dict(_FIDUCIAL_DEFAULT)
    root = _results_root()
    # HOD MAP
    try:
        with open(os.path.join(root, "bgs_zm15_joint", "map_result.json")) as fh:
            p = json.load(fh)["params"]
        for n in ("lg_m1h", "lg_m0star", "beta", "delta", "gamma",
                  "sigma_lnmstar", "eta", "fc", "bsat"):
            if n in p:
                fid[n] = float(p[n])
    except Exception:
        pass
    # X-ray MAP.
    #
    # The forecast's gas sector is still the phenomenological L_X-M / kT-M power
    # law (lx_norm, lx_slope, kt_norm, kt_slope; see forward_jax), but the band
    # fit was re-based onto native DPM parameters (log10_ne03, beta_n, log10_p03,
    # beta_P).  A native summary therefore satisfies none of the four amplitude
    # names while STILL matching p2/r_max/log10DC -- so a per-name loop would
    # quietly take the shape from the new fit and the amplitude from
    # _FIDUCIAL_DEFAULT, i.e. from a fit that no longer exists.  That mix is
    # worse than either source alone and is invisible downstream, so take the
    # gas block all-or-nothing.
    _XRAY_KEYS = ("lx_norm", "lx_slope", "kt_norm", "kt_slope", "p2", "r_max", "log10DC")
    try:
        with open(os.path.join(root, "xray_joint_bands", "S1_bands_summary.json")) as fh:
            summ = json.load(fh)
        m = summ["map"]
    except Exception:
        summ, m = None, None
    if m is not None:
        # A native-DPM summary carries the four scaling-relation values in
        # 'map_scaling', projected by fit_xray_joint_bands.scaling_from_native.
        # Prefer 'map' (an old phenomenological fit), then that projection.
        src = dict(m)
        proj = (summ or {}).get("map_scaling") or {}
        for n in ("lx_norm", "lx_slope", "kt_norm", "kt_slope"):
            if n not in src and n in proj:
                src[n] = proj[n]
        if all(n in src for n in _XRAY_KEYS):
            for n in _XRAY_KEYS:
                fid[n] = float(src[n])
        else:
            _warn_native_band_summary(sorted(set(_XRAY_KEYS) - set(src)))
    return fid


def fiducial_vector() -> np.ndarray:
    """Fiducial values as a flat vector in :data:`PARAM_NAMES` order."""
    fid = load_fiducial()
    return np.array([fid[n] for n in PARAM_NAMES], dtype=float)


def prior_sigma_vector() -> np.ndarray:
    """1σ Gaussian-prior widths per parameter (inf = no prior). Planck on Ω_m, σ8."""
    s = np.full(N_PARAM, np.inf)
    for n, val in PLANCK_PRIOR_SIGMA.items():
        s[_IDX[n]] = val
    return s


def regularizing_prior(add_planck: bool = False, fix: tuple = ()) -> np.ndarray:
    """Weakly-informative prior vector for stable marginalisation.

    Uses :data:`BROAD_PRIOR_SIGMA` for every parameter; applies the tight Planck
    prior on Ω_m, σ8 when ``add_planck``; and pins parameters named in ``fix`` to
    a delta-function (σ = 1e-4) — used to compare "baryons fixed" vs "marginalised".
    """
    s = np.array([BROAD_PRIOR_SIGMA[n] for n in PARAM_NAMES], dtype=float)
    if add_planck:
        for n, val in PLANCK_PRIOR_SIGMA.items():
            s[_IDX[n]] = val
    for n in fix:
        s[_IDX[n]] = 1e-4
    return s


def latex_labels() -> list[str]:
    return [PARAM_LATEX[n] for n in PARAM_NAMES]
