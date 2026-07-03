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
    "p0_pressure": 8.403, "c500_pressure": 1.177, "gamma_pressure": 0.3081,
    "alpha_pressure": 1.0510, "beta_out_pressure": 5.4905,               # A10 GNFW
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
    "p0_pressure": 4.0, "c500_pressure": 0.6, "gamma_pressure": 0.3,
    "alpha_pressure": 0.5, "beta_out_pressure": 2.0,
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
    "p0_pressure": r"$P_0$", "c500_pressure": r"$c_{500}$",
    "gamma_pressure": r"$\gamma_P$", "alpha_pressure": r"$\alpha_P$",
    "beta_out_pressure": r"$\beta_{P,\rm out}$",
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
            "alpha_in_gas", "alpha_tr_gas", "beta_pressure", "p0_pressure",
            "c500_pressure", "gamma_pressure", "alpha_pressure",
            "beta_out_pressure", "lx_zs", "kt_zs",
            "t_prof_slope", "z_gas_norm", "z_gas_mslope", "z_gas_zs"],
    "baryon": ["log10_M_pivot", "log10_eta_min", "beta_b", "log10_M_eta",
               "beta_eta", "eps_sn"],
    "agn": ["agn_mu_bh", "agn_al_bh", "agn_sig_bh", "agn_log10_lstar",
            "agn_delta1", "agn_delta2", "agn_log10_ferdf", "agn_rho",
            "agn_sig_mstar", "agn_log10_ferdf_zs", "agn_log10_lstar_zs",
            "agn_gamma", "agn_fabs", "agn_mu_bh_zs",
            "agn_xi_rx", "agn_xi_rm", "agn_b_r", "agn_sig_r"],
    # SF/quiescent split (ZM16 halo quenching + the quenched-CGM offset)
    "sfq": ["log10_Mq_cen", "mu_q_cen", "log10_Mq_sat", "mu_q_sat",
            "dlx_quenched"],
    # cold gas / neutral hydrogen (VN18 halo model)
    "coldgas": ["log10_M0_hi", "log10_Mmin_hi", "alpha_hi"],
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
    # X-ray MAP
    try:
        with open(os.path.join(root, "xray_joint_bands", "S1_bands_summary.json")) as fh:
            m = json.load(fh)["map"]
        for n in ("lx_norm", "lx_slope", "kt_norm", "kt_slope", "p2", "r_max", "log10DC"):
            if n in m:
                fid[n] = float(m[n])
    except Exception:
        pass
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
