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

from hod_mod.forecast.forward_jax import PARAM_NAMES, _IDX, N_PARAM

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
