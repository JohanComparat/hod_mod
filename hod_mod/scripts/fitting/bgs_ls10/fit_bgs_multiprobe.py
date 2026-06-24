"""Fit the More+2015 HOD model to BGS wp(rp) + ESD_HSC + ESD_DES data.

Reads the new joint summary-statistics HDF5 produced by ``sum_stat``, which
stores a 286-element data vector and its full jackknife covariance covering
SMF, WP, ESD_HSC, ESD_DES, ESD_KIDS, WTHETA, and KNN.  This script extracts
the [WP | ESD_HSC | ESD_DES] sub-data-vector with the corresponding
(N_wp + 2 N_esd) × (N_wp + 2 N_esd) joint covariance sub-block and fits with
:class:`~hod_mod.galaxies.hod.MoreHODModel` including four cosmological
parameters with tight Gaussian Planck 2018 priors.

HSC and DES observe the same lens sample, so the model predicts one ΔΣ(R)
curve that is compared against both; the joint covariance captures different
noise levels between the two source catalogues.

Cosmological parameters
-----------------------
h, Ω_m, n_s, ln10As are freed with Gaussian Planck 2018 priors (±3σ hard
bounds).  Ω_b is held fixed at the Planck mean; Ω_cdm = Ω_m − Ω_b is derived.

Physics flags
-------------
``--use-ia``
    Add intrinsic-alignment (IA) correction to ΔΣ using the non-linear
    alignment (NLA) model (Bridle & King 2007 `arXiv:0705.0166
    <https://arxiv.org/abs/0705.0166>`_).  Free parameter: ``A_IA``
    with a uniform prior (0, 5).  DESI KP6 finds A_IA ~ 0.3–1.5 for
    BGS-like lenses (`arXiv:2512.02954 <https://arxiv.org/abs/2512.02954>`_).
    η_IA is fixed to 0 (weak redshift dependence at z < 0.2).

``--use-baryon-fraction``
    Replace the constant-f_b post-hoc split with a physically motivated
    gas profile using a mass-dependent baryon fraction (FLAMINGO
    `arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_; closure-radius
    model `arXiv:2603.13095 <https://arxiv.org/abs/2603.13095>`_) and a
    gas NFW profile with reduced concentration c_gas = η(M) c_DM
    (IllustrisTNG `arXiv:2409.01758 <https://arxiv.org/abs/2409.01758>`_;
    Mead+2015 `arXiv:1611.08606 <https://arxiv.org/abs/1611.08606>`_).
    Free parameters: ``log10_M_pivot``, ``beta_b``, ``log10_eta_min``.

HOD priors
----------
log10mmin bound (11.0, 14.5) and Gaussian prior N(11.5, 0.5) motivated by
stellar-to-halo mass relation for M* > 10^10 M_sun
(`arXiv:2512.15960v3 <https://arxiv.org/abs/2512.15960>`_).
alpha_inc is fixed to 0 (BGS >95% complete at the stellar mass threshold).

Units
-----
- WP data in file: physical Mpc → ×h → Mpc/h for comparison with predictor
- ESD data in file: M_sun pc⁻² (invariant under h-rescaling)
- Predictor delta_sigma output: M_sun h pc⁻² → ÷h → M_sun pc⁻²

Inputs
------
- BGS_Mstar{XX}/LS10_VLIM_ANY_{XX}_Mstar_12.0_{zlo}_z_{zhi}_N_*_joint_*-sys-comb.h5

Outputs
-------
- results/bgs_multiprobe/mstar{XX}/map_result.json
- results/bgs_multiprobe/mstar{XX}/flatchain.npz
- results/bgs_multiprobe/mstar{XX}/multiprobe_bestfit.pdf  (with --plot)

Usage
-----
Single bin (full physics)::

    python scripts/fitting/bgs_ls10/fit_bgs_multiprobe.py \\
        --mstar 10.0 --map-only --use-ia --use-baryon-fraction

References
----------
More et al. 2015, ApJ 806, 2 (`arXiv:1211.6211 <https://arxiv.org/abs/1211.6211>`_)
Planck Collaboration 2020, A&A 641, A6 (`arXiv:1807.06209 <https://arxiv.org/abs/1807.06209>`_)
Bridle & King 2007, NJP 9, 444 (`arXiv:0705.0166 <https://arxiv.org/abs/0705.0166>`_)
FLAMINGO (`arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_)
Veenema et al. 2026 (`arXiv:2603.13095 <https://arxiv.org/abs/2603.13095>`_)
IllustrisTNG baryonic effects (`arXiv:2409.01758 <https://arxiv.org/abs/2409.01758>`_)
Mead et al. 2015 (`arXiv:1611.08606 <https://arxiv.org/abs/1611.08606>`_)
DESI KP6 IA (`arXiv:2512.02954 <https://arxiv.org/abs/2512.02954>`_)
BGS HOD priors (`arXiv:2512.15960 <https://arxiv.org/abs/2512.15960>`_)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

from hod_mod.cosmology.beyond_linear_bias import BeyondLinearBiasMead21
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies.hod import (
    HODModel, MoreHODModel, Kravtsov04HODModel,
    VanUitert16CSMFModel, ZuMandelbaum15HODModel, Zacharegkas25HODModel,
)
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.baryon_fraction import BaryonFractionSigmoid
from hod_mod.galaxies.intrinsic_alignment import NLAModel
from hod_mod.fitting.hod_wp import _assemble_hod_params
from hod_mod.fitting.planck_prior import (
    PLANCK18_MEANS,
    PLANCK18_SIGMAS,
    PLANCK18_3SIGMA,
    gaussian_log_prior,
)
from hod_mod.data_io.sum_stat_reader import SumStatReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUM_STAT_DIR = "/home/comparat/software/sum_stat/data"

# Cosmological parameters that are varied with a Planck Gaussian prior.
# Omega_b is held fixed; Omega_cdm is derived as Omega_m - Omega_b.
_COSMO_FREE = ["h", "Omega_m", "n_s", "ln10^{10}A_s"]

# Baryon fraction + gas concentration parameters consumed by _pk_tables_full.
# Separated from HOD params so _build_hod_params can strip them cleanly.
_BARYON_KEYS = ["log10_M_pivot", "beta_b", "log10_eta_min", "log10_M_eta"]

# NLA intrinsic alignment parameters.
_IA_KEYS = ["A_IA", "eta_IA"]

# Per-survey ESD amplitude calibration factors (shear/photo-z systematic).
# Applied as ΔΣ_model *= f_cal per survey; f_cal=1 is the unbiased limit.
_CALIB_KEYS = ["f_cal_hsc", "f_cal_des"]

# log10mmin_init set to SHMR-motivated values for each M* bin.
# Reference: arXiv:2512.15960v3 (BGS HOD) + Behroozi+2013 SHMR.
# Old values (12.1 for M*>10, etc.) were 0.5–1 dex too high and led to
# degenerate MAP solutions with log10mmin~10, alpha~2.
BGS_BINS = {
    9.0:  {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.08,  "z_eff": 0.065, "log10mmin_init": 11.2},
    9.5:  {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.12,  "z_eff": 0.085, "log10mmin_init": 11.5},
    10.0: {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.18,  "z_eff": 0.136, "log10mmin_init": 11.5},
    10.5: {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.25,  "z_eff": 0.150, "log10mmin_init": 12.0},
    11.0: {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.35,  "z_eff": 0.200, "log10mmin_init": 12.5},
    11.5: {"mstar_hi": 12.0, "z_min": 0.05, "z_max": 0.35,  "z_eff": 0.200, "log10mmin_init": 13.0},
}


# HOD model registry: name → {class, HOD-specific init, bounds, free param list}
# Baryon and IA parameters are added on top of these by _build_params() when the
# corresponding physics flags are set; they are common to all HOD models.
HOD_REGISTRY = {
    # --- Halo-mass threshold models (constructor: cls(hmf, hmf.bias)) ---
    "more2015": {
        "class":             MoreHODModel,
        "bias_arg":          True,
        "stellar_mass_model": False,
        "hod_init":   {
            "sigma_logm": 0.67, "log10m1_offset": 1.0,   # log10m1 = log10mmin + offset
            "alpha": 1.0, "kappa": 1.13,
            "alpha_inc": 0.0, "log10m_inc": 11.5,        # fixed (BGS ≥95% complete)
        },
        "hod_bounds": {
            "sigma_logm": (0.05, 1.5), "log10m1": (11.5, 15.0),
            "alpha": (0.5, 2.5), "kappa": (0.1, 5.0),
        },
        "hod_free":   ["log10mmin", "sigma_logm", "log10m1", "alpha", "kappa"],
    },
    "zheng2007": {
        "class":             HODModel,
        "bias_arg":          True,
        "stellar_mass_model": False,
        "hod_init":   {
            "sigma_logm": 0.67, "log10m1_offset": 1.0,
            "alpha": 1.0, "log10m0_offset": 0.0,         # log10m0 = log10mmin + offset
        },
        "hod_bounds": {
            "sigma_logm": (0.05, 1.5), "log10m1": (11.0, 15.0),
            "alpha": (0.5, 2.5), "log10m0": (10.0, 14.0),
        },
        "hod_free":   ["log10mmin", "sigma_logm", "log10m0", "log10m1", "alpha"],
    },
    "aum": {
        "class":             Kravtsov04HODModel,
        "bias_arg":          True,
        "stellar_mass_model": False,
        "hod_init":   {
            "sigma_logm": 0.67, "log10m1_offset": 1.0,
            "alpha": 1.0, "log10m0_offset": 0.0,
        },
        "hod_bounds": {
            "sigma_logm": (0.05, 1.5), "log10m1": (11.0, 15.0),
            "alpha": (0.5, 2.5), "log10m0": (10.0, 14.0),
        },
        "hod_free":   ["log10mmin", "sigma_logm", "log10m0", "log10m1", "alpha"],
    },
    # --- Stellar-mass SHMR models (constructor: cls(hmf) only) ---
    # log10m_star_lo/hi or log10m_star_thresh are fixed from BGS_BINS at runtime.
    "vanuitert16": {
        "class":             VanUitert16CSMFModel,
        "bias_arg":          False,
        "stellar_mass_model": True,
        "hod_init": {
            "log10m_h1":    11.5,
            "log10m_star0": 10.5,
            "beta1":         5.0,
            "log10_beta2":  -0.5,
            "sigma_c":       0.15,
            "alpha_s":      -1.1,
            "b0":            0.0,
            "b1":            1.5,
        },
        "hod_bounds": {
            "log10m_h1":    (10.5, 14.0),
            "log10m_star0": ( 9.0, 12.0),
            "beta1":        ( 0.5, 10.0),
            "log10_beta2":  (-2.0,  1.0),
            "sigma_c":      (0.05,  1.0),
            "alpha_s":      (-2.5,  0.0),
            "b0":           (-3.0,  3.0),
            "b1":           ( 0.0,  4.0),
        },
        "hod_free": ["log10m_h1", "log10m_star0", "beta1", "log10_beta2",
                     "sigma_c", "alpha_s", "b0", "b1"],
    },
    "zu_mandelbaum15": {
        "class":             ZuMandelbaum15HODModel,
        "bias_arg":          False,
        "stellar_mass_model": True,
        # Fixed ZM15 Table 2 values: delta, gamma, eta, fc, beta_sat, bcut, beta_cut
        "hod_init": {
            "lg_m1h":        12.10,
            "lg_m0star":     10.31,
            "beta":           0.33,
            "sigma_lnmstar":  0.50,
            "bsat":           8.98,
            "alpha_sat":      1.00,
            "delta": 0.42, "gamma": 1.21, "eta": -0.04, "fc": 0.86,
            "beta_sat": 0.90, "bcut": 0.86, "beta_cut": 0.41,
        },
        "hod_bounds": {
            "lg_m1h":       (11.0, 14.0),
            "lg_m0star":    ( 9.0, 12.0),
            "beta":          (0.1,  1.0),
            "sigma_lnmstar": (0.1,  1.5),
            "bsat":          (1.0, 30.0),
            "alpha_sat":     (0.5,  2.0),
        },
        "hod_free": ["lg_m1h", "lg_m0star", "beta", "sigma_lnmstar", "bsat", "alpha_sat"],
    },
    "zacharegkas25": {
        "class":             Zacharegkas25HODModel,
        "bias_arg":          False,
        "stellar_mass_model": True,
        # Fixed: gamma_shmr, delta_shmr, f_cen, beta_sat, beta_cut, f_sat
        "hod_init": {
            "log10m1_shmr":    11.506,
            "log10eps":        -1.632,
            "alpha_shmr":      -1.638,
            "sigma_logm_star":  0.30,
            "alpha_sat":        1.00,
            "kappa":            1.00,
            "B_sat":           10.00,
            "B_cut":            5.00,
            "gamma_shmr": 0.596, "delta_shmr": 3.810, "f_cen": 1.0,
            "beta_sat": 1.0, "beta_cut": 1.0, "f_sat": 1.0,
        },
        "hod_bounds": {
            "log10m1_shmr":    (10.0, 14.0),
            "log10eps":        (-4.0,  0.0),
            "alpha_shmr":      (-3.0,  0.0),
            "sigma_logm_star": (0.05,  1.0),
            "alpha_sat":        (0.5,  2.5),
            "kappa":            (0.1,  5.0),
            "B_sat":            (0.1, 50.0),
            "B_cut":            (0.1, 20.0),
        },
        "hod_free": ["log10m1_shmr", "log10eps", "alpha_shmr",
                     "sigma_logm_star", "alpha_sat", "kappa", "B_sat", "B_cut"],
    },
}


def _find_data_file(mstar_lo: float, info: dict, sum_stat_dir: str) -> str:
    """Locate the BGS joint HDF5 file for the given mass bin by glob."""
    mlo = f"{mstar_lo:.1f}"
    mhi = f"{info['mstar_hi']:.1f}"
    subdir = os.path.join(sum_stat_dir, f"BGS_Mstar{mlo}")
    pattern = os.path.join(
        subdir,
        f"LS10_VLIM_ANY_{mlo}_Mstar_{mhi}_*_joint_*-sys-comb.h5",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No joint BGS file found matching pattern:\n  {pattern}"
        )
    return matches[-1]  # take the most recent if multiple exist


# ---------------------------------------------------------------------------
# Parameter splitting: HOD vs cosmology
# ---------------------------------------------------------------------------

def _build_theta_cosmo(all_params: dict, cosmo_default: dict) -> dict:
    """Construct a theta_cosmo dict from the combined parameter dict.

    Omega_b is fixed to its Planck mean; Omega_cdm is derived from Omega_m.
    """
    tc = dict(cosmo_default)
    for k in _COSMO_FREE:
        if k in all_params:
            tc[k] = float(all_params[k])
    if "Omega_m" in all_params:
        Omega_b = cosmo_default.get("Omega_b", PLANCK18_MEANS["Omega_b"])
        tc["Omega_m"]  = float(all_params["Omega_m"])
        tc["Omega_cdm"] = tc["Omega_m"] - Omega_b
    return tc


def _build_hod_params(all_params: dict) -> dict:
    """Strip cosmological, baryon, IA, and calibration keys from a combined parameter dict."""
    skip = set(_COSMO_FREE) | set(_BARYON_KEYS) | set(_IA_KEYS) | set(_CALIB_KEYS)
    return {k: v for k, v in all_params.items() if k not in skip}


def _build_baryon_params(all_params: dict) -> dict | None:
    """Extract baryon/gas-concentration parameters from combined parameter dict."""
    bp = {k: all_params[k] for k in _BARYON_KEYS if k in all_params}
    return bp if bp else None


def _build_ia_params(all_params: dict) -> dict | None:
    """Extract IA parameters from combined parameter dict."""
    ia = {k: all_params[k] for k in _IA_KEYS if k in all_params}
    return ia if ia else None


# ---------------------------------------------------------------------------
# Log-probability
# ---------------------------------------------------------------------------

def _log_prob_multiprobe(
    theta_vec,
    free_params: list,
    fixed_params: dict,
    param_bounds: dict,
    param_prior_types: dict,
    param_prior_means: dict,
    param_prior_sigmas: dict,
    predictor: FullHaloModelPrediction,
    cosmo_default: dict,
    probes: list,
    rp_arrays: dict,
    dv_obs: np.ndarray,
    icov: np.ndarray,
    z: float,
    pi_max: float,
    h_file: float,
    ia_model=None,
    use_baryon_fraction: bool = False,
) -> float:
    """Log-posterior for any subset of [WP, ESD_HSC, ESD_DES].

    .. math::

        \\ln P(\\theta|d) = \\ln \\pi(\\theta)
          - \\frac{1}{2} \\mathbf{r}^T C^{-1} \\mathbf{r}

    The ESD model is:

    .. math::

        \\Delta\\Sigma^{\\rm model}(R) =
          \\Delta\\Sigma^{\\rm grav}(R) + \\Delta\\Sigma^{\\rm IA}(R)

    where the IA term uses the NLA model (Bridle & King 2007
    `arXiv:0705.0166 <https://arxiv.org/abs/0705.0166>`_) when
    ``ia_model`` is provided, and the gravitational term uses the
    mass-integrated CDM+gas split when ``use_baryon_fraction=True``
    (arXiv:2409.01758, Mead+2015 arXiv:1611.08606).

    Parameters
    ----------
    theta_vec : array_like, shape (n_free,)
    free_params, fixed_params, param_bounds : parameter specification
    param_prior_types, param_prior_means, param_prior_sigmas : priors
    predictor : FullHaloModelPrediction
    cosmo_default : dict — default cosmology (Planck 2018)
    probes : list of str — ordered subset of ``['wp', 'esd_hsc', 'esd_des']``
    rp_arrays : dict — ``{probe: rp_array [Mpc/h]}`` for each probe in probes
    dv_obs : concatenated observed data vector matching probes order
    icov   : inverse-covariance matching dv_obs
    z, pi_max, h_file : redshift, l.o.s. limit, h for ESD unit conversion
    ia_model : NLAModel or None — IA contribution added to ESD when set
    use_baryon_fraction : bool — use mass-integrated CDM+gas ESD split
    """
    # 1. Prior bounds + Gaussian contributions
    log_pi = 0.0
    for name, val in zip(free_params, theta_vec):
        lo, hi = param_bounds[name]
        if not (lo <= val <= hi):
            return -np.inf
        ptype = param_prior_types.get(name, "uniform")
        if ptype == "gaussian":
            log_pi += gaussian_log_prior(
                val,
                param_prior_means[name],
                param_prior_sigmas[name],
                lo,
                hi,
            )

    # 2. Build parameter dicts
    all_params  = dict(fixed_params)
    for name, val in zip(free_params, theta_vec):
        all_params[name] = float(val)
    theta_cosmo   = _build_theta_cosmo(all_params, cosmo_default)
    hod_params    = _build_hod_params(all_params)
    baryon_params = _build_baryon_params(all_params) if use_baryon_fraction else None
    ia_params     = _build_ia_params(all_params) if ia_model is not None else None

    # 3. Predict each probe — delta_sigma returns M_sun h/pc²; data is M_sun/pc²
    try:
        parts = []
        for probe in probes:
            rp = jnp.array(rp_arrays[probe])
            if probe == "wp":
                parts.append(np.asarray(
                    predictor.wp(rp, pi_max, z, theta_cosmo, hod_params)
                ))
            else:
                # Gravitational lensing ESD (with optional gas profile correction)
                if use_baryon_fraction and baryon_params:
                    ds_grav = np.asarray(
                        predictor.delta_sigma_split(
                            rp, z, theta_cosmo, hod_params,
                            baryon_params=baryon_params,
                        )["total"]
                    )
                else:
                    ds_grav = np.asarray(
                        predictor.delta_sigma(rp, z, theta_cosmo, hod_params)
                    )

                # NLA intrinsic alignment contribution (Bridle & King 2007 arXiv:0705.0166)
                if ia_model is not None and ia_params is not None:
                    ds_ia = np.asarray(
                        ia_model.delta_sigma_ia(rp, z, theta_cosmo, ia_params)
                    )
                    ds = (ds_grav + ds_ia) / h_file
                else:
                    ds = ds_grav / h_file

                # Per-survey amplitude calibration (shear/photo-z systematic).
                # f_cal=1 recovers the uncorrected model.  Key: f_cal_hsc or f_cal_des.
                cal_key = "f_cal_" + probe.split("_", 1)[1]   # esd_hsc → f_cal_hsc
                f_cal = float(all_params.get(cal_key, 1.0))
                parts.append(ds * f_cal)
        dv_pred = np.concatenate(parts)
    except Exception:
        return -np.inf

    # 4. Chi²
    residual = dv_pred - dv_obs
    chi2 = float(residual @ icov @ residual)
    return log_pi - 0.5 * chi2


# ---------------------------------------------------------------------------
# Cached linear power spectrum — avoids repeated CAMB calls during MCMC
# ---------------------------------------------------------------------------

class _CachedPkLinear:
    """Thin interpolation cache around LinearPowerSpectrum.

    On the first call for a given (z, Ω_m, ln10As, h) key, a reference P(k)
    is computed on a fixed k grid via CAMB and log-log interpolated on all
    subsequent calls.  This reduces per-sample cost from ~30 s to <1 ms once
    the cache warms up.
    """

    def __init__(self, pk_lin_obj, n_k: int = 512):
        self._base       = pk_lin_obj
        self._k_ref      = np.logspace(-4, 1.5, n_k)
        self._log_k_ref  = np.log(self._k_ref)
        self._cache: dict = {}

    def _key(self, z: float, theta: dict) -> tuple:
        return (
            round(float(z), 4),
            round(float(theta["Omega_m"]), 5),
            round(float(theta["ln10^{10}A_s"]), 4),
            round(float(theta.get("h", 0.6736)), 4),
        )

    def pk_linear(self, k, z: float, theta: dict):
        key = self._key(z, theta)
        if key not in self._cache:
            pk_ref = np.asarray(self._base.pk_linear(self._k_ref, float(z), theta))
            self._cache[key] = np.log(np.maximum(pk_ref, 1e-50))
        log_k = np.log(np.asarray(k, dtype=float))
        return jnp.asarray(np.exp(np.interp(log_k, self._log_k_ref, self._cache[key])))


# ---------------------------------------------------------------------------
# Fitter class
# ---------------------------------------------------------------------------

class MultiProbeFitter:
    """Fit More+2015 HOD to BGS [WP + ESD_HSC + ESD_DES] joint data vector.

    Parameters
    ----------
    data_file : str
        Path to the joint BGS HDF5 file.
    z : float
        Effective redshift of the sample.
    log10mmin_init : float
        Initial guess for log10(M_min).
    rp_min_wp, rp_max_wp : float
        Scale cuts for WP [Mpc/h].
    rp_min_hsc, rp_min_des, rp_max_esd : float
        Per-survey lower scale cuts and shared upper cut for ESD [Mpc/h].
        HSC probes smaller scales (default 0.02 Mpc/h); DES is noisier at
        small scales (default 0.1 Mpc/h).  The two surveys may therefore
        have different radial grids.
    esd_sn_min : float
        Minimum signal-to-noise ratio for ESD bins (applied per survey).
        Default 5.
    pi_max : float
        Line-of-sight integration limit [Mpc/h].
    n_walkers, n_steps, n_burnin : int
        MCMC configuration.
    output_dir : str
        Directory for output files.
    """

    def __init__(
        self,
        data_file: str,
        z: float,
        log10mmin_init: float,
        mstar_lo: float = 10.0,
        mstar_hi: float = 12.0,
        probes: tuple = ("wp", "esd_hsc", "esd_des"),
        rp_min_wp: float = 0.3,
        rp_max_wp: float = 50.0,
        rp_min_hsc: float = 0.3,
        rp_min_des: float = 0.3,
        rp_max_esd: float = 10.0,
        esd_sn_min: float = 5.0,
        pi_max: float = 100.0,
        use_full_cov: bool = False,
        use_ia: bool = False,
        use_baryon_fraction: bool = False,
        use_smf: bool = False,
        use_offcentering: bool = False,
        use_free_cosmo: bool = False,
        use_esd_calib: bool = False,
        use_sat_ext: bool = False,
        use_stellar_mass: bool = False,
        use_incompleteness: bool = False,
        use_bnl: bool = True,
        hod_model: str = "more2015",
        profile: str = "nfw",
        einasto_alpha: float = 0.18,
        n_walkers: int = 64,
        n_steps: int = 3000,
        n_burnin: int = 500,
        output_dir: str = "results/bgs_multiprobe/",
    ):
        if hod_model not in HOD_REGISTRY:
            raise ValueError(f"Unknown HOD model '{hod_model}'. Choose from {list(HOD_REGISTRY)}")
        self.hod_model           = hod_model
        self.mstar_lo            = mstar_lo
        self.mstar_hi            = mstar_hi
        self.z                   = z
        self.pi_max              = pi_max
        self.probes              = list(probes)
        self.use_full_cov        = use_full_cov
        self.use_ia              = use_ia
        self.use_baryon_fraction = use_baryon_fraction
        self.use_smf             = use_smf
        self.use_offcentering    = use_offcentering
        self.use_free_cosmo      = use_free_cosmo
        self.use_esd_calib       = use_esd_calib
        self.use_sat_ext         = use_sat_ext
        self.use_stellar_mass    = use_stellar_mass
        self.use_incompleteness  = use_incompleteness
        self.use_bnl             = use_bnl
        self.profile             = profile
        self.einasto_alpha       = einasto_alpha
        self.n_walkers           = n_walkers
        self.n_steps             = n_steps
        self.n_burnin            = n_burnin
        self.output_dir          = output_dir

        self._pk_lin = LinearPowerSpectrum()
        self.cosmo_default = self._pk_lin.default_cosmology()

        self._load_data(data_file, rp_min_wp, rp_max_wp, rp_min_hsc, rp_min_des, rp_max_esd, esd_sn_min)
        self._build_params(log10mmin_init)
        self._build_predictor()
        self._build_icov()

    # ------------------------------------------------------------------
    # Setup

    def _load_data(
        self,
        data_file: str,
        rp_min_wp: float,
        rp_max_wp: float,
        rp_min_hsc: float,
        rp_min_des: float,
        rp_max_esd: float,
        esd_sn_min: float,
    ):
        reader = SumStatReader.from_hdf5(data_file)
        jt = reader.joint_bgs(probes=("wp", "esd_hsc", "esd_des"))

        self.h_file = jt["h"]

        # joint_bgs always returns [wp(30) | esd_hsc(30) | esd_des(30)]
        n_full   = 30
        dv_full  = jt["data_vector"]
        cov_full = jt["cov"]
        rp_wp_f  = jt["rp_wp"]
        rp_esd_f = jt["rp_esd"]

        # Build per-probe masks (always computed; only selected probes are used)
        mask_wp = (rp_wp_f >= rp_min_wp) & (rp_wp_f <= rp_max_wp)

        ds_hsc  = dv_full[n_full : 2 * n_full]
        ds_des  = dv_full[2 * n_full :]
        err_hsc = np.sqrt(np.diag(cov_full[n_full:2*n_full, n_full:2*n_full]))
        err_des = np.sqrt(np.diag(cov_full[2*n_full:, 2*n_full:]))
        sn_hsc  = np.where(err_hsc > 0, ds_hsc / err_hsc, 0.0)
        sn_des  = np.where(err_des > 0, ds_des / err_des, 0.0)

        mask_hsc = (
            (rp_esd_f >= rp_min_hsc) & (rp_esd_f <= rp_max_esd)
            & (ds_hsc > 0) & (sn_hsc >= esd_sn_min)
        )
        mask_des = (
            (rp_esd_f >= rp_min_des) & (rp_esd_f <= rp_max_esd)
            & (ds_des > 0) & (sn_des >= esd_sn_min)
        )

        # Map probe name → (mask in full 90-vector, rp array)
        _probe_masks = {"wp": mask_wp, "esd_hsc": mask_hsc, "esd_des": mask_des}
        _probe_rp    = {"wp": rp_wp_f, "esd_hsc": rp_esd_f, "esd_des": rp_esd_f}
        # Offsets of each probe in the full 90-element joint_bgs vector
        _probe_offset = {"wp": 0, "esd_hsc": n_full, "esd_des": 2 * n_full}

        # Diagonal variances of the full 90-element joint_bgs vector
        var_full = np.diag(cov_full)

        # Build combined index array over only the selected probes
        idx_parts = []
        for probe in self.probes:
            offset = _probe_offset[probe]
            probe_idx = np.where(_probe_masks[probe])[0] + offset
            idx_parts.append(probe_idx)
        idx_all = np.concatenate(idx_parts)

        self.rp_wp  = rp_wp_f[mask_wp]
        self.rp_hsc = rp_esd_f[mask_hsc]
        self.rp_des = rp_esd_f[mask_des]
        self.n_wp   = int(mask_wp.sum())
        self.n_hsc  = int(mask_hsc.sum())
        self.n_des  = int(mask_des.sum())

        self.dv_obs = dv_full[idx_all]
        self._var   = var_full[idx_all]          # diagonal variances (always kept)
        if self.use_full_cov:
            self._cov = cov_full[np.ix_(idx_all, idx_all)]

        # rp_arrays dict used by log_prob
        self.rp_arrays = {
            "wp":      self.rp_wp,
            "esd_hsc": self.rp_hsc,
            "esd_des": self.rp_des,
        }

        # Full (unmasked) arrays kept for plotting — includes points outside scale cuts
        self.rp_wp_full  = rp_wp_f
        self.rp_esd_full = rp_esd_f
        wp_full  = dv_full[:n_full]
        wp_err_f = np.sqrt(var_full[:n_full])
        self.wp_full_obs  = wp_full
        self.wp_full_err  = wp_err_f
        self.ds_hsc_full_obs = ds_hsc
        self.ds_hsc_full_err = err_hsc
        self.ds_des_full_obs = ds_des
        self.ds_des_full_err = err_des

        # Jackknife subsamples for the full 90-element vector (for diagnostic overplot)
        if "subsamples" in jt:
            subs = jt["subsamples"]          # (n_jk, 90) in h-units
            self.n_jk = int(subs.shape[0])
            self.wp_subs_full  = subs[:, :n_full]            # (n_jk, 30)
            self.ds_hsc_subs_full = subs[:, n_full:2*n_full] # (n_jk, 30)
            self.ds_des_subs_full = subs[:, 2*n_full:]        # (n_jk, 30)
        else:
            self.n_jk = None
            self.wp_subs_full = self.ds_hsc_subs_full = self.ds_des_subs_full = None

        # Fit-range views for quick access (populated if probe selected)
        cursor = 0
        for probe in self.probes:
            n = {"wp": self.n_wp, "esd_hsc": self.n_hsc, "esd_des": self.n_des}[probe]
            sl = slice(cursor, cursor + n)
            err_sl = np.sqrt(self._var[sl])
            if probe == "wp":
                self.wp_obs, self.wp_err = self.dv_obs[sl], err_sl
            elif probe == "esd_hsc":
                self.ds_hsc_obs, self.ds_hsc_err = self.dv_obs[sl], err_sl
            elif probe == "esd_des":
                self.ds_des_obs, self.ds_des_err = self.dv_obs[sl], err_sl
            cursor += n

        # SMF n_gal constraint — integrate phi(log10M*) over the mass bin
        if self.use_smf:
            smf = reader.smf()
            log10m = smf["log10mstar"]
            phi    = smf["phi"]
            cov_smf = smf["cov"]
            dlog10m = float(np.median(np.diff(log10m)))
            mask_smf = (log10m >= self.mstar_lo) & (log10m <= self.mstar_hi)
            self.n_gal_smf = float(np.sum(phi[mask_smf]) * dlog10m)
            sub_cov = cov_smf[np.ix_(mask_smf, mask_smf)]
            self.sigma_ngal_smf = float(np.sqrt(np.sum(sub_cov)) * dlog10m)
        else:
            self.n_gal_smf = None
            self.sigma_ngal_smf = None

    def _build_params(self, log10mmin_init: float):
        cfg = HOD_REGISTRY[self.hod_model]

        if cfg.get("stellar_mass_model", False):
            # Stellar-mass-based models (VanUitert16, ZuMandelbaum15, Zacharegkas25).
            # log10mmin is not a free parameter; the SHMR maps stellar mass → halo mass.
            # Stellar-mass bin edges are fixed from the BGS sample definition.
            self.free_params  = list(cfg["hod_free"])
            self.param_init   = dict(cfg["hod_init"])   # free + fixed HOD params
            self.param_bounds = dict(cfg["hod_bounds"])

            if self.hod_model in ("vanuitert16", "zacharegkas25"):
                self.param_init["log10m_star_lo"] = self.mstar_lo
                self.param_init["log10m_star_hi"] = self.mstar_hi
            elif self.hod_model == "zu_mandelbaum15":
                self.param_init["log10m_star_thresh"] = self.mstar_lo

            self.param_prior_types  = {p: "uniform" for p in self.free_params}
            self.param_prior_means  = {}
            self.param_prior_sigmas = {}

        else:
            # Halo-mass threshold models (HODModel, Kravtsov04HODModel, MoreHODModel).
            # log10mmin is always free; log10m1 and log10m0 are initialised relative to
            # log10mmin so the starting point is physically sensible for any mass bin.
            m0 = log10mmin_init
            self.free_params = list(cfg["hod_free"])

            self.param_init = {
                "log10mmin":  m0,
                "sigma_logm": cfg["hod_init"].get("sigma_logm", 0.67),
                "alpha":      cfg["hod_init"].get("alpha", 1.0),
            }
            if "kappa" in cfg["hod_free"]:
                self.param_init["kappa"]      = cfg["hod_init"].get("kappa", 1.13)
                self.param_init["alpha_inc"]  = cfg["hod_init"].get("alpha_inc", 0.0)
                self.param_init["log10m_inc"] = cfg["hod_init"].get("log10m_inc", 11.5)
            if "log10m1" in cfg["hod_free"]:
                self.param_init["log10m1"] = m0 + cfg["hod_init"].get("log10m1_offset", 1.0)
            if "log10m0" in cfg["hod_free"]:
                self.param_init["log10m0"] = m0 + cfg["hod_init"].get("log10m0_offset", 0.0)

            self.param_bounds = {"log10mmin": (11.0, 14.5), **cfg["hod_bounds"]}
            self.param_prior_types  = {p: "uniform" for p in self.free_params}
            self.param_prior_means  = {}
            self.param_prior_sigmas = {}

            # Gaussian prior on log10mmin from SHMR for M* > 10^10 Msun
            # (arXiv:2512.15960v3 §4; Behroozi+2013 arXiv:1207.6105)
            self.param_prior_types["log10mmin"]  = "gaussian"
            self.param_prior_means["log10mmin"]  = 11.5
            self.param_prior_sigmas["log10mmin"] = 0.5

        # IA parameters (added when use_ia=True) — common to all HOD models
        if self.use_ia:
            self.free_params.append("A_IA")
            self.param_init["A_IA"]   = 0.3   # DESI KP6 best-fit arXiv:2512.02954
            self.param_bounds["A_IA"] = (0.0, 5.0)
            self.param_prior_types["A_IA"] = "uniform"
            self.param_init["eta_IA"] = 0.0   # fixed; weak z-dep at z<0.2

        # Baryon fraction + gas concentration parameters (added when use_baryon_fraction=True)
        if self.use_baryon_fraction:
            # log10_M_pivot, beta_b: sigmoid f_b(M) (FLAMINGO arXiv:2510.25419;
            #   closure-radius model arXiv:2603.13095)
            # log10_eta_min: gas concentration ratio η_min (arXiv:2409.01758 Table 2)
            # log10_M_eta: fixed break mass M_2=10^13.0 from arXiv:2409.01758
            self.free_params += ["log10_M_pivot", "beta_b", "log10_eta_min"]
            self.param_init.update({
                "log10_M_pivot": 13.5,
                "beta_b":        1.5,
                "log10_eta_min": -0.22,  # log10(0.6)
                "log10_M_eta":   13.0,   # fixed
            })
            self.param_bounds.update({
                "log10_M_pivot": (12.0, 15.0),
                "beta_b":        (0.5,  4.0),
                "log10_eta_min": (-0.8, 0.0),
            })
            for k in ["log10_M_pivot", "beta_b", "log10_eta_min"]:
                self.param_prior_types[k] = "uniform"

        # Off-centering parameters (added when use_offcentering=True)
        # f_off: fraction of off-centered centrals (Leauthaud+2012 finds ~0.4 for groups)
        # sigma_off: Rayleigh scale [Mpc/h] (Johnston+2007 arXiv:0709.4193)
        if self.use_offcentering:
            self.free_params += ["f_off", "sigma_off"]
            self.param_init.update({"f_off": 0.2, "sigma_off": 0.2})
            self.param_bounds.update({"f_off": (0.0, 1.0), "sigma_off": (0.01, 2.0)})
            for k in ["f_off", "sigma_off"]:
                self.param_prior_types[k] = "uniform"

        # Satellite profile extensions (passed through to clustering.py via hod_params).
        # A: b_sat_conc — satellite concentration bias c_sat = b_sat_conc × c_DM
        #    b_sat_conc > 1 → satellites more concentrated (tidal stripping retains
        #    orbits in inner halo); < 1 → disrupted, puffed outward.
        # B: f_cut — inner suppression [1 − exp(−r / (f_cut × r_vir))]
        #    models tidal disruption below the disruption radius r_cut = f_cut r_vir.
        # C: gamma_inner — power-law depletion (r/r_vir)^γ on top of NFW
        #    motivated by orbital energy redistribution (van den Bosch+2005).
        if self.use_sat_ext:
            self.free_params += ["b_sat_conc", "f_cut", "gamma_inner"]
            self.param_init.update({
                "b_sat_conc":   1.0,
                "f_cut":        0.0,
                "gamma_inner":  0.0,
            })
            self.param_bounds.update({
                "b_sat_conc":  (0.3, 3.0),
                "f_cut":       (0.0, 0.3),
                "gamma_inner": (0.0, 3.0),
            })
            for k in ["b_sat_conc", "f_cut", "gamma_inner"]:
                self.param_prior_types[k] = "uniform"

        # Incompleteness parameters (More+2015 Eq. 3): free alpha_inc and log10m_inc
        # when the sample may not be 100% complete above the stellar mass threshold.
        if self.use_incompleteness:
            self.free_params += ["alpha_inc", "log10m_inc"]
            self.param_init.update({"alpha_inc": 0.44, "log10m_inc": 13.57})
            self.param_bounds.update({"alpha_inc": (-2.0, 2.0), "log10m_inc": (10.0, 14.0)})
            for k in ["alpha_inc", "log10m_inc"]:
                self.param_prior_types[k] = "uniform"

        # Point-mass stellar contribution to ΔΣ: ΔΣ_*(R) = M_*_cen / (π R²)
        # log10_M_star_cen is the log10 mean stellar mass of centrals [M_sun/h]
        if self.use_stellar_mass:
            self.free_params.append("log10_M_star_cen")
            self.param_init["log10_M_star_cen"]        = 10.5
            self.param_bounds["log10_M_star_cen"]      = (8.0, 12.0)
            self.param_prior_types["log10_M_star_cen"] = "uniform"

        # Cosmology: either free (Planck Gaussian priors) or fixed to Planck mean.
        # When use_free_cosmo=True, h, Omega_m, n_s, ln10As are freed with
        # Gaussian priors from Planck 2018 TT,TE,EE+lowE (arXiv:1807.06209)
        # and hard bounds at ±3σ.  Omega_b remains fixed; Omega_cdm is derived.
        if self.use_free_cosmo:
            self.free_params += list(_COSMO_FREE)
            for cp in _COSMO_FREE:
                lo, hi = PLANCK18_3SIGMA[cp]
                self.param_init[cp]         = PLANCK18_MEANS[cp]
                self.param_bounds[cp]       = (lo, hi)
                self.param_prior_types[cp]  = "gaussian"
                self.param_prior_means[cp]  = PLANCK18_MEANS[cp]
                self.param_prior_sigmas[cp] = PLANCK18_SIGMAS[cp]
        else:
            for cp in _COSMO_FREE:
                self.param_init[cp] = PLANCK18_MEANS[cp]

        # Per-survey ESD amplitude calibration (shear/photo-z systematic).
        # f_cal ∈ [0.7, 1.3] with uniform prior; f_cal=1 is the unbiased limit.
        # Only added for ESD probes actually selected.
        if self.use_esd_calib:
            for probe in self.probes:
                if probe.startswith("esd_"):
                    cal_key = "f_cal_" + probe.split("_", 1)[1]
                    self.free_params.append(cal_key)
                    self.param_init[cal_key]        = 1.0
                    self.param_bounds[cal_key]      = (0.7, 1.3)
                    self.param_prior_types[cal_key] = "uniform"

    def _build_predictor(self):
        self._pk_cached = _CachedPkLinear(self._pk_lin)
        hmf = make_hmf("csst")
        cfg = HOD_REGISTRY[self.hod_model]
        if cfg.get("bias_arg", True):
            hod = cfg["class"](hmf, hmf.bias)
        else:
            hod = cfg["class"](hmf)
        self.hod = hod     # kept for galaxy_number_density() used by SMF constraint
        halo_profile = HaloProfile(self.cosmo_default)

        bf = BaryonFractionSigmoid() if self.use_baryon_fraction else None
        bnl = BeyondLinearBiasMead21() if self.use_bnl else None
        self.predictor = FullHaloModelPrediction(
            self._pk_cached, hod, halo_profile, baryon_fraction=bf,
            profile=self.profile, einasto_alpha=self.einasto_alpha,
            bnl_model=bnl,
        )

        # NLA model uses P_lin (correct per Bridle & King 2007 arXiv:0705.0166 §2;
        # "Non-Linear Alignment" refers to tidal physics, not P_nl)
        self.ia_model = NLAModel(self._pk_cached.pk_linear) if self.use_ia else None

    def _build_icov(self):
        # 1% systematic floor added in quadrature to diagonal variances.
        sys_var = (0.01 * self.dv_obs) ** 2
        if self.use_full_cov:
            # Full jackknife covariance + systematic on diagonal.
            # Warning: condition number is typically ~10^21; may be ill-conditioned.
            cov_reg = self._cov.copy()
            np.fill_diagonal(cov_reg, np.diag(cov_reg) + sys_var)
            self.icov = np.linalg.inv(cov_reg)
        else:
            # Diagonal-only: stable inverse regardless of condition number.
            self.icov = np.diag(1.0 / (self._var + sys_var))

    # ------------------------------------------------------------------
    # Fixed / free partitions

    @property
    def _fixed_params(self) -> dict:
        return {k: v for k, v in self.param_init.items() if k not in self.free_params}

    @property
    def _x0(self) -> np.ndarray:
        return np.array([self.param_init[p] for p in self.free_params])

    # ------------------------------------------------------------------
    # Core log-probability

    def _log_pi(self, theta_vec: np.ndarray) -> float:
        """Prior log-probability only (no data residuals)."""
        log_pi = 0.0
        for name, val in zip(self.free_params, theta_vec):
            lo, hi = self.param_bounds[name]
            if not (lo <= val <= hi):
                return -np.inf
            ptype = self.param_prior_types.get(name, "uniform")
            if ptype == "gaussian":
                log_pi += gaussian_log_prior(
                    val,
                    self.param_prior_means[name],
                    self.param_prior_sigmas[name],
                    lo,
                    hi,
                )
        return log_pi

    def _log_prob(self, theta_vec) -> float:
        log_p = _log_prob_multiprobe(
            theta_vec,
            self.free_params,
            self._fixed_params,
            self.param_bounds,
            self.param_prior_types,
            self.param_prior_means,
            self.param_prior_sigmas,
            self.predictor,
            self.cosmo_default,
            self.probes,
            self.rp_arrays,
            self.dv_obs,
            self.icov,
            self.z,
            self.pi_max,
            self.h_file,
            ia_model=self.ia_model,
            use_baryon_fraction=self.use_baryon_fraction,
        )
        if self.use_smf and np.isfinite(log_p):
            all_params = dict(self._fixed_params)
            for name, val in zip(self.free_params, theta_vec):
                all_params[name] = float(val)
            theta_cosmo = _build_theta_cosmo(all_params, self.cosmo_default)
            hod_params  = _build_hod_params(all_params)
            try:
                n_gal_model = float(self.hod.galaxy_number_density(self.z, theta_cosmo, hod_params))
                chi2_ngal = ((n_gal_model - self.n_gal_smf) / self.sigma_ngal_smf) ** 2
                log_p -= 0.5 * chi2_ngal
            except Exception:
                return -np.inf
        return log_p

    # ------------------------------------------------------------------
    # Prediction helpers

    def predict_wp(self, params: dict, rp: np.ndarray | None = None) -> np.ndarray:
        """Predict wp(rp) for given parameter dict.

        Parameters
        ----------
        rp : array, optional
            Radii in Mpc/h.  Defaults to the fit-range rp array.
        """
        if rp is None:
            rp = self.rp_wp
        tc = _build_theta_cosmo(params, self.cosmo_default)
        hp = _build_hod_params(params)
        return np.asarray(
            self.predictor.wp(jnp.array(rp), self.pi_max, self.z, tc, hp)
        )

    def predict_ds(self, params: dict, rp: np.ndarray) -> np.ndarray:
        """Predict ΔΣ(R) in M_sun/pc² at arbitrary radii rp [Mpc/h]."""
        tc = _build_theta_cosmo(params, self.cosmo_default)
        hp = _build_hod_params(params)
        h  = params.get("h", self.h_file)
        return np.asarray(
            self.predictor.delta_sigma(jnp.array(rp), self.z, tc, hp)
        ) / h

    def predict_ds_hsc(self, params: dict, rp: np.ndarray | None = None) -> np.ndarray:
        """Predict ΔΣ(R) at HSC radii in M_sun/pc²."""
        return self.predict_ds(params, self.rp_hsc if rp is None else rp)

    def predict_ds_des(self, params: dict, rp: np.ndarray | None = None) -> np.ndarray:
        """Predict ΔΣ(R) at DES radii in M_sun/pc²."""
        return self.predict_ds(params, self.rp_des if rp is None else rp)

    # ------------------------------------------------------------------
    # MAP estimation

    def map_fit(self) -> dict:
        """MAP estimate via Nelder-Mead.

        Returns
        -------
        dict with keys ``theta``, ``params``, ``chi2``, ``ndof``,
        ``success``, ``message``.
        """
        from scipy.optimize import minimize

        t0 = time.time()
        result = minimize(
            lambda x: -self._log_prob(x),
            self._x0,
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-4, "fatol": 1e-4, "disp": False},
        )
        elapsed = round(time.time() - t0, 1)
        best_theta = result.x
        best_params = dict(self._fixed_params)
        for name, val in zip(self.free_params, best_theta):
            best_params[name] = float(val)

        n_data = len(self.dv_obs)
        n_free = len(self.free_params)
        chi2_total = float(-2.0 * self._log_prob(best_theta))
        chi2_data  = chi2_total + 2.0 * float(self._log_pi(best_theta))
        if self.n_jk is not None:
            hartlap = (self.n_jk - n_data - 2) / (self.n_jk - 1)
        else:
            hartlap = None
        return {
            "theta":          best_theta,
            "params":         best_params,
            "chi2":           chi2_total,
            "chi2_data":      chi2_data,
            "ndof":           n_data - n_free,
            "n_jk":           self.n_jk,
            "hartlap_factor": hartlap,
            "success":        result.success,
            "message":        result.message,
            "elapsed":        elapsed,
        }

    # ------------------------------------------------------------------
    # MCMC

    def sample(
        self,
        initial_pos: np.ndarray | None = None,
        progress: bool = True,
    ):
        """Run emcee ensemble MCMC with per-step HDF5 checkpointing.

        The chain is written to ``chain.h5`` in ``output_dir`` after every
        step via :class:`emcee.backends.HDFBackend`.  If that file already
        exists and contains a compatible chain, sampling resumes from the
        last completed step — allowing the run to be safely interrupted and
        restarted on any machine.

        Burn-in and production are stored in separate backends so that the
        burn-in phase can also be resumed and the production chain remains
        clean.

        Parameters
        ----------
        initial_pos : array_like, shape (n_walkers, n_free), optional
            Starting positions.  Ignored when resuming from a checkpoint.
        progress : bool

        Returns
        -------
        emcee.EnsembleSampler  (production sampler)
        """
        import emcee

        os.makedirs(self.output_dir, exist_ok=True)

        n_free    = len(self.free_params)
        n_walkers = self.n_walkers

        burnin_h5  = os.path.join(self.output_dir, "chain_burnin.h5")
        prod_h5    = os.path.join(self.output_dir, "chain.h5")

        # ---- Burn-in phase ----
        bi_backend = emcee.backends.HDFBackend(burnin_h5)
        bi_done = bi_backend.iteration if os.path.exists(burnin_h5) else 0

        if bi_done < self.n_burnin:
            if bi_done == 0:
                # Fresh start: build initial positions
                if initial_pos is None:
                    x0  = self._x0
                    pos = np.zeros((n_walkers, n_free))
                    for i, name in enumerate(self.free_params):
                        lo, hi = self.param_bounds[name]
                        width = 0.03 * (hi - lo)
                        pos[:, i] = np.clip(
                            x0[i] + np.random.randn(n_walkers) * width, lo, hi
                        )
                    initial_pos = pos
                bi_backend.reset(n_walkers, n_free)
                print(f"Burn-in: {self.n_burnin} steps, {n_walkers} walkers …")
            else:
                initial_pos = None  # resume from last saved position
                remaining = self.n_burnin - bi_done
                print(f"Resuming burn-in from step {bi_done} ({remaining} steps remaining) …")

            bi_sampler = emcee.EnsembleSampler(
                n_walkers, n_free, self._log_prob, backend=bi_backend
            )
            bi_sampler.run_mcmc(initial_pos, self.n_burnin - bi_done, progress=progress)
            last_pos = bi_sampler.get_last_sample()
        else:
            print(f"Burn-in already complete ({bi_done} steps found).")
            last_pos = bi_backend.get_last_sample()

        # ---- Production phase ----
        prod_backend = emcee.backends.HDFBackend(prod_h5)
        prod_done = prod_backend.iteration if os.path.exists(prod_h5) else 0

        if prod_done >= self.n_steps:
            print(f"Production chain already complete ({prod_done} steps found).")
        else:
            if prod_done == 0:
                prod_backend.reset(n_walkers, n_free)
                start_pos = last_pos
                print(f"Sampling: {self.n_steps} steps …")
            else:
                start_pos = None  # resume
                remaining = self.n_steps - prod_done
                print(f"Resuming production from step {prod_done} ({remaining} steps remaining) …")

            prod_sampler = emcee.EnsembleSampler(
                n_walkers, n_free, self._log_prob, backend=prod_backend
            )
            prod_sampler.run_mcmc(start_pos, self.n_steps - prod_done, progress=progress)

        # Export flat chain as npz for downstream tools
        prod_backend = emcee.backends.HDFBackend(prod_h5, read_only=True)
        flatchain = prod_backend.get_chain(flat=True)
        out_npz = os.path.join(self.output_dir, "flatchain.npz")
        np.savez(out_npz, flatchain=flatchain, param_names=np.array(self.free_params))
        print(f"Chain saved → {out_npz}  ({len(flatchain)} samples)")

        acc = np.mean(prod_backend.accepted / prod_backend.iteration)
        print(f"Mean acceptance fraction: {acc:.3f}")

        return prod_sampler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mstar", type=float, default=10.0,
        choices=list(BGS_BINS.keys()),
        help="Lower stellar mass threshold log10(M*/M_sun).",
    )
    parser.add_argument("--sum-stat-dir", default=SUM_STAT_DIR)
    parser.add_argument(
        "--probes", default="wp,esd_hsc,esd_des",
        help="Comma-separated probes to fit: wp, esd_hsc, esd_des (default: all three).",
    )
    parser.add_argument("--rp-min-wp",  type=float, default=0.3)
    parser.add_argument("--rp-max-wp",  type=float, default=50.0)
    parser.add_argument("--rp-min-hsc", type=float, default=0.3,
                        help="Minimum ESD_HSC radius [Mpc/h] (default 0.3).")
    parser.add_argument("--rp-min-des", type=float, default=0.3,
                        help="Minimum ESD_DES radius [Mpc/h] (default 0.3).")
    parser.add_argument("--rp-max-esd", type=float, default=10.0)
    parser.add_argument("--esd-sn-min", type=float, default=5.0,
                        help="Minimum S/N per ESD bin (applied per survey independently).")
    parser.add_argument("--pi-max",      type=float, default=100.0)
    parser.add_argument("--use-full-cov", action="store_true",
                        help="Invert the full jackknife covariance (may be ill-conditioned)."
                             " Default: diagonal errors only.")
    parser.add_argument("--use-ia", action="store_true",
                        help="Add NLA intrinsic-alignment correction to ESD "
                             "(Bridle & King 2007 arXiv:0705.0166). "
                             "Adds free parameter A_IA with uniform prior (0, 5).")
    parser.add_argument("--use-baryon-fraction", action="store_true",
                        help="Use mass-integrated CDM+gas ESD split with reduced "
                             "gas concentration (arXiv:2409.01758, Mead+2015). "
                             "Adds free params log10_M_pivot, beta_b, log10_eta_min.")
    parser.add_argument("--use-smf", action="store_true",
                        help="Add n_gal from the SMF integral [mstar_lo, mstar_hi] "
                             "as an extra chi² constraint on the HOD normalisation.")
    parser.add_argument("--use-offcentering", action="store_true",
                        help="Add off-centering correction following Johnston+2007 "
                             "(arXiv:0709.4193). Adds free params f_off (fraction "
                             "off-centered, init 0.2) and sigma_off (Rayleigh scale "
                             "[Mpc/h], init 0.2).")
    parser.add_argument("--free-cosmo", action="store_true",
                        help="Free h, Omega_m, n_s, ln10As with Planck 2018 Gaussian "
                             "priors (arXiv:1807.06209) and ±3σ hard bounds. "
                             "Omega_b is held fixed; Omega_cdm = Omega_m − Omega_b.")
    parser.add_argument("--free-esd-calib", action="store_true",
                        help="Add per-survey ESD amplitude calibration factors "
                             "f_cal_hsc and/or f_cal_des with uniform prior [0.7, 1.3]. "
                             "Absorbs shear multiplicative bias and photo-z amplitude "
                             "errors (Mandelbaum+2018 arXiv:1710.00885).")
    parser.add_argument("--use-sat-ext", action="store_true",
                        help="Add satellite inner-profile extensions: "
                             "b_sat_conc (concentration bias, Ext A), "
                             "f_cut (inner suppression [1-exp(-r/r_cut)], Ext B), "
                             "gamma_inner (power-law depletion (r/r_vir)^γ, Ext C). "
                             "All three are freed simultaneously with uniform priors.")
    parser.add_argument("--use-stellar-mass", action="store_true",
                        help="Add point-mass stellar contribution to ΔΣ: "
                             "ΔΣ_*(R) = M_*_cen / (π R²). "
                             "Adds free parameter log10_M_star_cen with uniform prior (8, 12).")
    parser.add_argument("--use-incompleteness", action="store_true",
                        help="Free the More+2015 incompleteness parameters alpha_inc and "
                             "log10m_inc (fixed to 0 and 11.5 by default in BGS fits). "
                             "Use when the sample may not be fully complete.")
    parser.add_argument(
        "--hod-model", default="more2015",
        choices=list(HOD_REGISTRY), metavar="NAME",
        help="HOD model to use: " + ", ".join(HOD_REGISTRY) + " (default: more2015).",
    )
    parser.add_argument(
        "--profile", default="nfw", choices=["nfw", "einasto"],
        help="Halo profile for 1-halo Fourier transform: nfw (default) or einasto.",
    )
    parser.add_argument(
        "--einasto-alpha", type=float, default=0.18,
        help="Einasto shape parameter α (default 0.18; ignored for --profile nfw).",
    )
    parser.add_argument("--map-only",   action="store_true")
    parser.add_argument("--mcmc-only",  action="store_true")
    parser.add_argument("--n-walkers",  type=int, default=64)
    parser.add_argument("--n-steps",    type=int, default=3000)
    parser.add_argument("--n-burnin",   type=int, default=500)
    parser.add_argument("--plot",       action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(repo_root, "results", "bgs_multiprobe"),
    )
    args = parser.parse_args()

    info       = BGS_BINS[args.mstar]
    probes     = [p.strip() for p in args.probes.split(",")]
    mstar_str  = f"{args.mstar:.1f}"
    probe_tag  = "_".join(probes)
    # rp_min encoded as integer milliparsecs/h: 0.30→rp300, 0.10→rp100, 0.05→rp050
    rp_tag  = f"rp{int(round(args.rp_min_wp * 1000)):03d}"
    suffix  = "_".join(filter(None, [
        args.profile,                                       # always included
        rp_tag,                                             # always included
        "ia"      if args.use_ia              else "",
        "offcen"  if args.use_offcentering    else "",
        "bfrac"   if args.use_baryon_fraction else "",
        "stellar" if args.use_stellar_mass    else "",
        "inc"     if args.use_incompleteness  else "",
        "fcosmo"  if args.free_cosmo          else "",
        "fcalib"  if args.free_esd_calib      else "",
        "sext"    if args.use_sat_ext         else "",
    ]))
    dir_name = f"mstar{mstar_str}_{probe_tag}_{args.hod_model}_{suffix}"
    output_dir = os.path.join(args.output_dir, dir_name)

    data_file = _find_data_file(args.mstar, info, args.sum_stat_dir)

    fitter = MultiProbeFitter(
        data_file            = data_file,
        z                    = info["z_eff"],
        log10mmin_init       = info["log10mmin_init"],
        mstar_lo             = args.mstar,
        mstar_hi             = info["mstar_hi"],
        probes               = tuple(probes),
        rp_min_wp            = args.rp_min_wp,
        rp_max_wp            = args.rp_max_wp,
        rp_min_hsc           = args.rp_min_hsc,
        rp_min_des           = args.rp_min_des,
        rp_max_esd           = args.rp_max_esd,
        esd_sn_min           = args.esd_sn_min,
        pi_max               = args.pi_max,
        use_full_cov         = args.use_full_cov,
        use_ia               = args.use_ia,
        use_baryon_fraction  = args.use_baryon_fraction,
        use_smf              = args.use_smf,
        use_offcentering     = args.use_offcentering,
        use_free_cosmo       = args.free_cosmo,
        use_esd_calib        = args.free_esd_calib,
        use_sat_ext          = args.use_sat_ext,
        use_stellar_mass     = args.use_stellar_mass,
        use_incompleteness   = args.use_incompleteness,
        hod_model            = args.hod_model,
        profile              = args.profile,
        einasto_alpha        = args.einasto_alpha,
        n_walkers            = args.n_walkers,
        n_steps              = args.n_steps,
        n_burnin             = args.n_burnin,
        output_dir           = output_dir,
    )

    print(f"\nBGS multi-probe HOD fit  [{args.hod_model}]")
    print(f"  Stellar mass threshold:  log10(M*/M_sun) > {args.mstar}")
    print(f"  Redshift range:  z = {info['z_min']:.2f}–{info['z_max']:.2f}  (z_eff = {info['z_eff']:.3f})")
    print(f"  Data file:  {data_file}")
    print(f"  N_wp bins  (after scale cut): {fitter.n_wp}")
    print(f"  N_hsc bins (after scale cut): {fitter.n_hsc}  (rp > {args.rp_min_hsc} Mpc/h, S/N>{args.esd_sn_min})")
    print(f"  N_des bins (after scale cut): {fitter.n_des}  (rp > {args.rp_min_des} Mpc/h, S/N>{args.esd_sn_min})")
    print(f"  Total data points: {len(fitter.dv_obs)}")
    if fitter.use_smf:
        print(f"  SMF n_gal constraint: n_gal = {fitter.n_gal_smf:.4e} ± {fitter.sigma_ngal_smf:.4e} (h/Mpc)^3")
    print(f"  Free params ({len(fitter.free_params)}): {fitter.free_params}")

    os.makedirs(output_dir, exist_ok=True)

    if not args.mcmc_only:
        result = fitter.map_fit()
        print("\n=== MAP result ===")
        for name, val in zip(fitter.free_params, result["theta"]):
            print(f"  {name:30s} = {val:.5f}")
        print(f"  chi2/dof = {result['chi2']:.2f} / {result['ndof']}")
        print(f"  MAP elapsed: {result['elapsed']:.1f} s")
        print(f"  Optimizer: {result['message']}")

        out_json = os.path.join(output_dir, "map_result.json")
        with open(out_json, "w") as fh:
            json.dump(
                {
                    # --- fit quality ---
                    "params":   result["params"],
                    "chi2":     result["chi2"],
                    "ndof":     result["ndof"],
                    "elapsed":  result["elapsed"],
                    "success":  result["success"],
                    "message":  result["message"],
                    # --- sample / data ---
                    "mstar_lo":  args.mstar,
                    "mstar_hi":  info["mstar_hi"],
                    "z_min":     info["z_min"],
                    "z_max":     info["z_max"],
                    "z_eff":     info["z_eff"],
                    "data_file": data_file,
                    # --- probe selection and scale cuts ---
                    "probes":      args.probes,
                    "rp_min_wp":   args.rp_min_wp,
                    "rp_max_wp":   args.rp_max_wp,
                    "rp_min_hsc":  args.rp_min_hsc,
                    "rp_min_des":  args.rp_min_des,
                    "rp_max_esd":  args.rp_max_esd,
                    "esd_sn_min":  args.esd_sn_min,
                    "pi_max":      args.pi_max,
                    "n_wp":        fitter.n_wp,
                    "n_hsc":       fitter.n_hsc,
                    "n_des":       fitter.n_des,
                    # --- model choices ---
                    "hod_model":      args.hod_model,
                    "profile":        args.profile,
                    "einasto_alpha":  args.einasto_alpha,
                    # --- physics flags ---
                    "use_ia":              args.use_ia,
                    "use_baryon_fraction": args.use_baryon_fraction,
                    "use_smf":             args.use_smf,
                    "use_offcentering":    args.use_offcentering,
                    "use_full_cov":        args.use_full_cov,
                    "use_free_cosmo":      args.free_cosmo,
                    "use_esd_calib":       args.free_esd_calib,
                    "use_sat_ext":         args.use_sat_ext,
                    # --- MCMC settings (recorded even for MAP-only runs) ---
                    "n_walkers": args.n_walkers,
                    "n_steps":   args.n_steps,
                    "n_burnin":  args.n_burnin,
                    # --- free parameter names (for MCMC chain alignment) ---
                    "free_params": fitter.free_params,
                },
                fh,
                indent=2,
            )
        print(f"MAP result saved → {out_json}")

        if args.plot:
            import matplotlib.pyplot as plt

            p = result["params"]

            # Predictions over the full rp range (for model extrapolation below cut)
            wp_pred_full     = fitter.predict_wp(p, rp=fitter.rp_wp_full)
            ds_pred_full     = fitter.predict_ds(p, rp=fitter.rp_esd_full)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            kw_full  = dict(fmt="o", ms=3, color="0.6", ecolor="0.7", zorder=1)
            kw_fit   = dict(fmt="o", ms=5, color="k",   zorder=2)

            def _overplot_subsamples(ax, rp, subs, color="C3"):
                """Draw each jackknife subsample as a faint line (no markers)."""
                if subs is None:
                    return
                for row in subs:
                    mask = row > 0
                    if mask.any():
                        ax.plot(rp[mask], row[mask],
                                color=color, lw=0.3, alpha=0.15, zorder=0)

            def _plot_model_split(ax, rp_full, pred_full, rp_min, color, label):
                """Plot model solid above cut, dashed below cut."""
                below = rp_full < rp_min
                above = ~below
                if above.any():
                    ax.loglog(rp_full[above], pred_full[above],
                              color=color, lw=1.5, label=label)
                if below.any():
                    ax.loglog(rp_full[below], pred_full[below],
                              color=color, lw=1.5, ls="--", alpha=0.6)

            # --- WP ---
            ax = axes[0]
            _overplot_subsamples(ax, fitter.rp_wp_full, fitter.wp_subs_full)
            ax.errorbar(fitter.rp_wp_full, fitter.wp_full_obs, fitter.wp_full_err,
                        label="data (all scales)", **kw_full)
            ax.errorbar(fitter.rp_wp, fitter.wp_obs, fitter.wp_err,
                        label=f"BGS $M_* > 10^{{{args.mstar}}}$ (fit range)", **kw_fit)
            _plot_model_split(ax, fitter.rp_wp_full, wp_pred_full,
                              args.rp_min_wp, "C0", "More+2015 MAP")
            ax.axvline(args.rp_min_wp, ls=":", color="C0", lw=1, alpha=0.7)
            ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
            ax.set_ylabel(r"$w_p(r_p)$ [Mpc/$h$]")
            ax.legend(fontsize=7)
            ax.set_title("WP")

            # --- ESD HSC ---
            ax = axes[1]
            pos_hsc = fitter.ds_hsc_full_obs > 0
            _overplot_subsamples(ax, fitter.rp_esd_full, fitter.ds_hsc_subs_full)
            ax.errorbar(fitter.rp_esd_full[pos_hsc], fitter.ds_hsc_full_obs[pos_hsc],
                        fitter.ds_hsc_full_err[pos_hsc], label="HSC (all scales)", **kw_full)
            ax.errorbar(fitter.rp_hsc, fitter.ds_hsc_obs, fitter.ds_hsc_err,
                        label="HSC (fit range)", **kw_fit)
            _plot_model_split(ax, fitter.rp_esd_full[pos_hsc], ds_pred_full[pos_hsc],
                              args.rp_min_hsc, "C1", "More+2015 MAP")
            ax.axvline(args.rp_min_hsc, ls=":", color="C1", lw=1, alpha=0.7)
            ax.set_xlabel(r"$R$ [Mpc/$h$]")
            ax.set_ylabel(r"$\Delta\Sigma$ [$M_\odot\,{\rm pc}^{-2}$]")
            ax.legend(fontsize=7)
            ax.set_title("ESD — HSC")

            # --- ESD DES ---
            ax = axes[2]
            pos_des = fitter.ds_des_full_obs > 0
            _overplot_subsamples(ax, fitter.rp_esd_full, fitter.ds_des_subs_full)
            ax.errorbar(fitter.rp_esd_full[pos_des], fitter.ds_des_full_obs[pos_des],
                        fitter.ds_des_full_err[pos_des], label="DES (all scales)", **kw_full)
            ax.errorbar(fitter.rp_des, fitter.ds_des_obs, fitter.ds_des_err,
                        label="DES (fit range)", **kw_fit)
            _plot_model_split(ax, fitter.rp_esd_full[pos_des], ds_pred_full[pos_des],
                              args.rp_min_des, "C2", "More+2015 MAP")
            ax.axvline(args.rp_min_des, ls=":", color="C2", lw=1, alpha=0.7)
            ax.set_xlabel(r"$R$ [Mpc/$h$]")
            ax.set_ylabel(r"$\Delta\Sigma$ [$M_\odot\,{\rm pc}^{-2}$]")
            ax.legend(fontsize=7)
            ax.set_title("ESD — DES")

            fig.suptitle(
                rf"BGS/LS10  $\log M_* > {args.mstar}$"
                rf"  $z \in [{info['z_min']:.2f},{info['z_max']:.2f}]$"
                rf"  $\chi^2/\nu = {result['chi2']:.0f}/{result['ndof']}$"
            )
            plt.tight_layout()
            out_fig = os.path.join(output_dir, "multiprobe_bestfit.pdf")
            plt.savefig(out_fig)
            print(f"Figure saved → {out_fig}")

    if not args.map_only:
        initial_pos = None
        if not args.mcmc_only and "result" in dir():
            # Initialise walkers near MAP solution
            n_free = len(fitter.free_params)
            pos = np.zeros((args.n_walkers, n_free))
            for i, name in enumerate(fitter.free_params):
                lo, hi = fitter.param_bounds[name]
                width = 0.02 * (hi - lo)
                pos[:, i] = np.clip(
                    result["theta"][i] + np.random.randn(args.n_walkers) * width,
                    lo, hi,
                )
            initial_pos = pos

        sampler = fitter.sample(initial_pos=initial_pos, progress=True)
        flat    = sampler.get_chain(flat=True)
        acc     = np.mean(sampler.acceptance_fraction)
        print(f"\nMCMC acceptance fraction: {acc:.3f}")
        print(f"Chain shape: {flat.shape}")


if __name__ == "__main__":
    main()
