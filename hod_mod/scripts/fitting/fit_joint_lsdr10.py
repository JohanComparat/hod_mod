"""Joint fit of wp(rp) + ΔΣ(R) + w_θ(θ) for LSDR10 BGS galaxy samples.

Data sources
------------
wp + ESD
    ~/software/sum_stat/data/BGS_Mstar{10.0,10.5,11.0,11.5}/
    LS10_VLIM_ANY_*_N_*_joint_smf-wp-esd_hsc-esd_des-esd_kids-wtheta-knn-sys-comb.h5
    Read via SumStatReader.joint_bgs(probes=("wp", "esd_hsc")).

Galaxy × X-ray
    ~/data/zenodo/LSDR10_GALxEVT/Measurements_Xcorr_Stacks/XCORR/*_GALxEVT_wtheta.fits
    Same format as fit_comparat2025.py.

Samples with all three statistics (N matched between sum_stat and zenodo)
    S1  log10M*>10.0  z_mean=0.135  N=2,759,238
    S3  log10M*>10.5  z_mean=0.191  N=3,263,228
    S5  log10M*>11.0  z_mean=0.252  N=1,619,838
    S7  log10M*>11.5  z_mean=0.261  N=  120,882

Model
-----
HOD     : MoreHODModel (More+2015, arXiv:1407.1856)
Gas     : GasDensityDPM model 2 + 30-arcsec eROSITA PSF
AGN     : XrayAGNModel (Comparat+2019 abundance matching)

Free parameters
---------------
  log10mmin  central occ. threshold   [10.5, 14.5]
  sigma_logm central occ. scatter     [0.01, 1.5]
  log10m1    satellite mass scale     [11.0, 15.5]
  alpha      satellite slope          [0.5, 2.5]
  kappa      soft-cutoff factor       [0.1, 5.0]
  log10_A_gas  DPM gas amplitude      [-2, 12]
  log10_A_AGN  AGN amplitude          [-5, 15]

Likelihood
----------
  log L = log L_wp + log L_esd + log L_wtheta

  L_wp, L_esd : full jackknife covariance from sum_stat HDF5
  L_wtheta    : diagonal with systematic floor
                err_eff = sqrt(err_jk² + (f_sys × |wtheta|)²)

Usage::

    # MAP for all four samples
    python -m hod_mod.scripts.fitting.fit_joint_lsdr10 --sample all --mode map

    # MAP + MCMC for S1 only
    python -m hod_mod.scripts.fitting.fit_joint_lsdr10 --sample S1 --mode both

    # Custom scale cuts
    python -m hod_mod.scripts.fitting.fit_joint_lsdr10 \\
        --rp-min 0.3 --rp-max 30 --R-min 0.1 --theta-min 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.special import j0

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import GasDensityDPM
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.xray import XrayAGNModel
from hod_mod.data_io.sum_stat_reader import SumStatReader
from hod_mod.paths import results_root

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SUM_STAT_DIR = Path(os.path.expanduser("~/software/sum_stat/data"))
_ZENODO_DIR   = Path(os.path.expanduser(
    "~/data/zenodo/LSDR10_GALxEVT/Measurements_Xcorr_Stacks/XCORR"
))
_RESULTS_DIR  = results_root() / "fits" / "joint_lsdr10"
_SHAPE_CACHE  = _RESULTS_DIR / "shape_cache"

# ---------------------------------------------------------------------------
# Cosmology (Planck 2018)
# ---------------------------------------------------------------------------
_THETA_COSMO = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "flat": True,
    "H0":     _THETA_COSMO["h"] * 100.0,
    "Om0":    _THETA_COSMO["Omega_m"],
    "Ob0":    _THETA_COSMO["Omega_b"],
    "ns":     _THETA_COSMO["n_s"],
    "sigma8": 0.811,
}

# ---------------------------------------------------------------------------
# Sample definitions
# ---------------------------------------------------------------------------
SAMPLES = {
    "S1": dict(log10ms_min=10.00, zmax=0.18, zmean=0.135, N=2759238,
               sum_stat_dir="BGS_Mstar10.0"),
    "S3": dict(log10ms_min=10.50, zmax=0.26, zmean=0.191, N=3263228,
               sum_stat_dir="BGS_Mstar10.5"),
    "S5": dict(log10ms_min=11.00, zmax=0.35, zmean=0.252, N=1619838,
               sum_stat_dir="BGS_Mstar11.0"),
    "S7": dict(log10ms_min=11.50, zmax=0.35, zmean=0.261, N=120882,
               sum_stat_dir="BGS_Mstar11.5"),
}

# More+2015 HOD starting values per sample (from Comparat+2025 Table 3 as proxy).
# alpha_inc=0.0 disables the linear incompleteness correction (f_inc=1 everywhere),
# appropriate for volume-limited BGS samples.
_HOD_INIT = {
    "S1": dict(log10mmin=12.113, sigma_logm=0.666, log10m1=13.2, alpha=1.18, kappa=1.0,
               alpha_inc=0.0, log10m_inc=13.0),
    "S3": dict(log10mmin=12.362, sigma_logm=0.538, log10m1=13.4, alpha=1.16, kappa=1.0,
               alpha_inc=0.0, log10m_inc=13.0),
    "S5": dict(log10mmin=12.623, sigma_logm=0.296, log10m1=13.6, alpha=1.14, kappa=1.0,
               alpha_inc=0.0, log10m_inc=13.0),
    "S7": dict(log10mmin=12.880, sigma_logm=0.180, log10m1=13.8, alpha=1.10, kappa=1.0,
               alpha_inc=0.0, log10m_inc=13.0),
}

# Expected log10_A_gas initial guess from amplitude_vs_mstar analysis
_A_GAS_INIT = {"S1": 6.27, "S3": 6.34, "S5": 6.44, "S7": 6.59}

_N_ELL   = 80
_ELL_ARR = np.logspace(1.0, 4.3, _N_ELL)
_PSF_FWHM = 30.0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _sum_stat_path(label: str) -> Path:
    s = SAMPLES[label]
    d = _SUM_STAT_DIR / s["sum_stat_dir"]
    matches = sorted(d.glob(f"*_N_{s['N']:07d}_joint_smf-wp-esd*.h5"))
    if not matches:
        # Try zero-padded N with fewer digits
        N_str = f"{s['N']:07d}"
        matches = sorted(d.glob(f"*N_{N_str}*.h5"))
    if not matches:
        matches = sorted(d.glob("*.h5"))
    if not matches:
        raise FileNotFoundError(f"No HDF5 file found for sample {label} in {d}")
    return matches[0]


def _zenodo_path(label: str) -> Path:
    s = SAMPLES[label]
    N = f"{s['N']:07d}"
    matches = sorted(_ZENODO_DIR.glob(f"*_N_{N}_GALxEVT_wtheta.fits"))
    if not matches:
        raise FileNotFoundError(
            f"No zenodo wtheta FITS for sample {label} (N={N}) in {_ZENODO_DIR}"
        )
    return matches[0]


def load_wp_esd(label: str, esd_survey: str = "esd_hsc") -> dict:
    """Load wp + ESD from sum_stat HDF5.

    Returns a dict with keys:
        rp_wp      (Nwp,) [Mpc/h]
        wp         (Nwp,) [Mpc/h]
        rp_esd     (Nesd,) [Mpc/h]
        esd        (Nesd,) [Msun/pc²]
        cov_wp     (Nwp, Nwp)
        cov_esd    (Nesd, Nesd)
        h          float
    """
    path = _sum_stat_path(label)
    reader = SumStatReader.from_hdf5(str(path))
    jt = reader.joint_bgs(probes=("wp", esd_survey))
    sls = jt["slices_out"]

    dv  = jt["data_vector"]
    cov = jt["cov"]

    sl_wp  = sls["wp"]
    sl_esd = sls[esd_survey]

    return {
        "rp_wp":   jt.get("rp_wp",  np.ones(sl_wp.stop - sl_wp.start)),
        "wp":      dv[sl_wp],
        "cov_wp":  cov[sl_wp, sl_wp],
        "rp_esd":  jt.get("rp_esd", np.ones(sl_esd.stop - sl_esd.start)),
        "esd":     dv[sl_esd],
        "cov_esd": cov[sl_esd, sl_esd],
        "h":       jt["h"],
    }


def load_wtheta(label: str) -> dict:
    """Load galaxy × X-ray w_θ from zenodo FITS."""
    path = _zenodo_path(label)
    t = fits.open(path)[1].data
    theta_deg    = np.array(t["theta"],      dtype=float)
    wtheta       = np.array(t["wtheta"],     dtype=float)
    wtheta_err   = np.array(t["wtheta_err"], dtype=float)
    theta_arcsec = theta_deg * 3600.0
    theta_rad    = theta_deg * (np.pi / 180.0)
    conv_kpc     = np.array(t["convert_theta_to_kpc"], dtype=float) \
        if "convert_theta_to_kpc" in t.names else theta_arcsec * 0.0
    return {
        "theta_arcsec": theta_arcsec,
        "theta_rad":    theta_rad,
        "wtheta":       wtheta,
        "wtheta_err":   wtheta_err,
        "R_kpc":        conv_kpc * theta_arcsec,
    }


# ---------------------------------------------------------------------------
# n(z) builder
# ---------------------------------------------------------------------------

def _build_nz(label: str, n_pts: int = 5) -> tuple[np.ndarray, np.ndarray]:
    s  = SAMPLES[label]
    z  = s["zmean"]
    dz = min(0.02, s["zmax"] * 0.10)
    z_arr = np.linspace(max(0.01, z - 2.0 * dz), z + 2.0 * dz, n_pts)
    nz    = np.exp(-0.5 * ((z_arr - z) / dz) ** 2)
    return z_arr, nz / trapezoid(nz, z_arr)


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

class _Infrastructure:
    """One-time build of the halo model stack."""

    def __init__(self):
        print("Building halo model infrastructure (CAMB + HMF) ...", flush=True)
        t0 = time.time()
        pk_lin    = LinearPowerSpectrum()
        hmf       = make_hmf("csst")
        hp        = HaloProfile(_COLOSSUS, cm_relation="diemer19")
        hod       = MoreHODModel(hmf, hmf.bias)
        self.fhmp = FullHaloModelPrediction(pk_lin, hod, hp)
        self.dp   = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=200)
        self.agn  = XrayAGNModel()
        self.cross = HaloModelCrossSpectra(
            self.fhmp, density_profile=self.dp, agn_model=self.agn
        )
        print(f"  done in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# HOD parameter dict builder
# ---------------------------------------------------------------------------

def _hod_params(label: str, log10mmin=None, sigma_logm=None,
                log10m1=None, alpha=None, kappa=None) -> dict:
    base = dict(_HOD_INIT[label])
    if log10mmin  is not None: base["log10mmin"]  = log10mmin
    if sigma_logm is not None: base["sigma_logm"] = sigma_logm
    if log10m1    is not None: base["log10m1"]    = log10m1
    if alpha      is not None: base["alpha"]      = alpha
    if kappa      is not None: base["kappa"]      = kappa
    return base


# ---------------------------------------------------------------------------
# Shape cache for w_θ
# ---------------------------------------------------------------------------

def _shape_cache_key(label: str, hod_params: dict) -> str:
    hp_str = json.dumps(
        {k: round(float(v), 6) for k, v in sorted(hod_params.items())},
        sort_keys=True,
    )
    raw = f"{label}|{hp_str}|agn|joint"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _hankel(cl_arr: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    return np.array([
        trapezoid(_ELL_ARR * cl_arr * j0(_ELL_ARR * th) / (2.0 * np.pi), _ELL_ARR)
        for th in theta_rad
    ])


def _predict_wtheta_shapes(label: str, infra: _Infrastructure,
                            hod_params: dict,
                            use_disk_cache: bool = True) -> dict:
    """Return {"gas": wtheta_shape, "agn": wtheta_shape} — not multiplied by amplitude."""
    if use_disk_cache:
        key  = _shape_cache_key(label, hod_params)
        path = _SHAPE_CACHE / f"{label}_{key}.npz"
        if path.exists():
            print(f"  [{label}] wtheta shape loaded from cache ({path.name})", flush=True)
            d = np.load(path)
            return {"gas": d["gas"], "agn": d["agn"]}

    z_arr, nz_g = _build_nz(label)
    print(f"  [{label}] computing wtheta shapes (n_z={len(z_arr)}, n_ell={_N_ELL}) ...",
          flush=True)
    t0 = time.time()
    cl_comp = infra.cross.angular_cl_gX(
        _ELL_ARR, z_arr, nz_g, _THETA_COSMO, hod_params,
        psf_fwhm_arcsec=_PSF_FWHM,
        return_components=True,
        n_workers=1,   # serial — XLA memory allocator is not thread-safe
    )
    print(f"  [{label}] angular_cl_gX: {time.time()-t0:.1f}s", flush=True)

    theta_rad = load_wtheta(label)["theta_rad"]
    shapes = {
        "gas": _hankel(np.asarray(cl_comp["gas"], dtype=float), theta_rad),
        "agn": _hankel(np.asarray(cl_comp["agn"], dtype=float), theta_rad),
    }

    if use_disk_cache:
        _SHAPE_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **shapes)
        print(f"  [{label}] wtheta shape cached → {path.name}", flush=True)

    return shapes


# ---------------------------------------------------------------------------
# Scale masks
# ---------------------------------------------------------------------------

def _wp_mask(rp: np.ndarray, rp_min: float, rp_max: float) -> np.ndarray:
    return (rp >= rp_min) & (rp <= rp_max)


def _esd_mask(R: np.ndarray, R_min: float, R_max: float) -> np.ndarray:
    return (R >= R_min) & (R <= R_max)


def _theta_mask(theta_arcsec: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    return (theta_arcsec >= t_min) & (theta_arcsec <= t_max)


# ---------------------------------------------------------------------------
# Likelihood, prior, log-prob
# ---------------------------------------------------------------------------

def log_likelihood(params: np.ndarray, label: str, infra: _Infrastructure,
                   data_wp_esd: dict, data_wtheta: dict,
                   masks: dict, shape_cache: dict,
                   f_sys_wtheta: float = 0.05) -> float:
    """
    params = [log10mmin, sigma_logm, log10m1, alpha, kappa, log10_A_gas, log10_A_AGN]
    """
    log10mmin, sigma_logm, log10m1, alpha, kappa = params[:5]
    log10_A_gas, log10_A_AGN = params[5], params[6]

    hp = _hod_params(label,
                     log10mmin=log10mmin, sigma_logm=sigma_logm,
                     log10m1=log10m1, alpha=alpha, kappa=kappa)

    z_eff = SAMPLES[label]["zmean"]
    cache_key = (round(log10mmin, 4), round(sigma_logm, 4),
                 round(log10m1, 4), round(alpha, 4), round(kappa, 4))

    # -- wp prediction --
    wp_model = infra.fhmp.wp(
        data_wp_esd["rp_wp"], pi_max=100.0, z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=hp,
    )
    wp_model = np.asarray(wp_model, dtype=float)

    # -- ESD prediction --
    # Model returns [h Msun/pc²]; sum_stat data is in [Msun/pc²] → divide by h.
    h = float(data_wp_esd["h"])
    esd_model = infra.fhmp.delta_sigma(
        data_wp_esd["rp_esd"], z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=hp,
    )
    esd_model = np.asarray(esd_model, dtype=float) * h

    # -- wtheta prediction (uses cached shapes) --
    if cache_key not in shape_cache:
        shape_cache[cache_key] = _predict_wtheta_shapes(label, infra, hp)
    shapes = shape_cache[cache_key]
    A_gas, A_AGN = 10.0 ** log10_A_gas, 10.0 ** log10_A_AGN
    wtheta_model = A_gas * shapes["gas"] + A_AGN * shapes["agn"]

    # -- Gaussian log-likelihoods --
    mk_wp    = masks["wp"]
    mk_esd   = masks["esd"]
    mk_theta = masks["theta"]

    # wp: full covariance
    res_wp  = data_wp_esd["wp"][mk_wp] - wp_model[mk_wp]
    cov_wp  = data_wp_esd["cov_wp"][np.ix_(mk_wp, mk_wp)]
    try:
        L_wp = -0.5 * float(res_wp @ np.linalg.solve(cov_wp, res_wp))
    except np.linalg.LinAlgError:
        return -np.inf

    # ESD: diagonal covariance (jackknife matrix is ill-conditioned; diagonal is robust)
    res_esd  = data_wp_esd["esd"][mk_esd] - esd_model[mk_esd]
    var_esd  = np.diag(data_wp_esd["cov_esd"])[mk_esd]
    L_esd    = -0.5 * float(np.sum(res_esd**2 / var_esd))

    # wtheta: diagonal + systematic floor
    wd       = data_wtheta["wtheta"][mk_theta]
    err_jk   = data_wtheta["wtheta_err"][mk_theta]
    err_sys  = f_sys_wtheta * np.abs(wd)
    err      = np.sqrt(err_jk**2 + err_sys**2)
    err      = np.where(err > 1e-12 * np.abs(wd).max(), err, 1e-12 * np.abs(wd).max())
    res_wt   = wd - wtheta_model[mk_theta]
    L_wt     = -0.5 * np.sum((res_wt / err) ** 2)

    return L_wp + L_esd + L_wt


def log_prior(params: np.ndarray) -> float:
    log10mmin, sigma_logm, log10m1, alpha, kappa, log10_A_gas, log10_A_AGN = params
    if not (10.5 <= log10mmin  <= 14.5): return -np.inf
    if not (0.01 <= sigma_logm <= 1.5):  return -np.inf
    if not (11.0 <= log10m1    <= 15.5): return -np.inf
    if not (0.5  <= alpha      <= 2.5):  return -np.inf
    if not (0.1  <= kappa      <= 5.0):  return -np.inf
    if not (-2.0 <= log10_A_gas <= 12.0):return -np.inf
    if not (-5.0 <= log10_A_AGN <= 15.0):return -np.inf
    if log10m1 < log10mmin:              return -np.inf
    return 0.0


def log_prob(params, label, infra, data_wp_esd, data_wtheta,
             masks, shape_cache, f_sys_wtheta=0.05):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, label, infra, data_wp_esd, data_wtheta,
                        masks, shape_cache, f_sys_wtheta)
    return lp + ll


# ---------------------------------------------------------------------------
# chi2/dof per probe (for reporting)
# ---------------------------------------------------------------------------

def _chi2_per_probe(params, label, infra, data_wp_esd, data_wtheta,
                    masks, shape_cache, f_sys_wtheta=0.05):
    log10mmin, sigma_logm, log10m1, alpha, kappa = params[:5]
    log10_A_gas, log10_A_AGN = params[5], params[6]
    hp = _hod_params(label, log10mmin=log10mmin, sigma_logm=sigma_logm,
                     log10m1=log10m1, alpha=alpha, kappa=kappa)
    z_eff = SAMPLES[label]["zmean"]

    h = float(data_wp_esd["h"])
    wp_model  = np.asarray(infra.fhmp.wp(
        data_wp_esd["rp_wp"], pi_max=100.0, z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=hp), dtype=float)
    esd_model = np.asarray(infra.fhmp.delta_sigma(
        data_wp_esd["rp_esd"], z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=hp), dtype=float) * h

    cache_key = (round(log10mmin, 4), round(sigma_logm, 4),
                 round(log10m1, 4), round(alpha, 4), round(kappa, 4))
    shapes = shape_cache[cache_key]
    wtheta_model = 10.0**log10_A_gas * shapes["gas"] + 10.0**log10_A_AGN * shapes["agn"]

    mk_wp, mk_esd, mk_theta = masks["wp"], masks["esd"], masks["theta"]

    res_wp  = data_wp_esd["wp"][mk_wp] - wp_model[mk_wp]
    cov_wp  = data_wp_esd["cov_wp"][np.ix_(mk_wp, mk_wp)]
    chi2_wp = float(res_wp @ np.linalg.solve(cov_wp, res_wp))

    res_esd  = data_wp_esd["esd"][mk_esd] - esd_model[mk_esd]
    var_esd  = np.diag(data_wp_esd["cov_esd"])[mk_esd]
    chi2_esd = float(np.sum(res_esd**2 / var_esd))

    wd      = data_wtheta["wtheta"][mk_theta]
    err_jk  = data_wtheta["wtheta_err"][mk_theta]
    err_sys = f_sys_wtheta * np.abs(wd)
    err     = np.sqrt(err_jk**2 + err_sys**2)
    res_wt  = wd - wtheta_model[mk_theta]
    chi2_wt = float(np.sum((res_wt / err) ** 2))

    return {
        "chi2_wp":  chi2_wp,  "ndof_wp":  int(mk_wp.sum())  - 5,
        "chi2_esd": chi2_esd, "ndof_esd": int(mk_esd.sum()) - 5,
        "chi2_wt":  chi2_wt,  "ndof_wt":  int(mk_theta.sum()) - 2,
        "chi2_dof_wp":  chi2_wp  / max(int(mk_wp.sum())  - 5, 1),
        "chi2_dof_esd": chi2_esd / max(int(mk_esd.sum()) - 5, 1),
        "chi2_dof_wt":  chi2_wt  / max(int(mk_theta.sum()) - 2, 1),
    }


# ---------------------------------------------------------------------------
# MAP fit
# ---------------------------------------------------------------------------

def run_map(label: str, infra: _Infrastructure,
            rp_min: float = 0.3, rp_max: float = 30.0,
            R_min:  float = 0.1, R_max:  float = 30.0,
            theta_min: float = 8.0, theta_max: float = 300.0,
            esd_survey: str = "esd_hsc",
            f_sys_wtheta: float = 0.05) -> tuple[dict, dict]:

    data_wp_esd = load_wp_esd(label, esd_survey=esd_survey)
    data_wtheta = load_wtheta(label)

    masks = {
        "wp":    _wp_mask(data_wp_esd["rp_wp"],      rp_min,    rp_max),
        "esd":   _esd_mask(data_wp_esd["rp_esd"],    R_min,     R_max),
        "theta": _theta_mask(data_wtheta["theta_arcsec"], theta_min, theta_max),
    }

    n_pts = {k: int(v.sum()) for k, v in masks.items()}
    n_free = 7
    shape_cache: dict = {}

    # Initial guess: HOD from table + A_gas from previous X-ray-only fit
    hod0 = _HOD_INIT[label]
    x0 = np.array([
        hod0["log10mmin"], hod0["sigma_logm"],
        hod0["log10m1"],   hod0["alpha"], hod0["kappa"],
        _A_GAS_INIT.get(label, 6.3),
        _A_GAS_INIT.get(label, 6.3) - 1.0,
    ])
    bounds = [
        (10.5, 14.5), (0.01, 1.5), (11.0, 15.5), (0.5, 2.5), (0.1, 5.0),
        (-2.0, 12.0), (-5.0, 15.0),
    ]
    param_names = ["log10mmin", "sigma_logm", "log10m1", "alpha", "kappa",
                   "log10_A_gas", "log10_A_AGN"]

    def neg_log_prob(p):
        v = log_prob(p, label, infra, data_wp_esd, data_wtheta,
                     masks, shape_cache, f_sys_wtheta)
        return -v if np.isfinite(v) else 1e30

    print(f"  [{label}] MAP: x0={np.round(x0, 3)}  "
          f"n_pts=wp:{n_pts['wp']} esd:{n_pts['esd']} wtheta:{n_pts['theta']}",
          flush=True)
    t0 = time.time()
    res = minimize(neg_log_prob, x0, method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 2000})
    print(f"  [{label}] MAP done in {time.time()-t0:.1f}s  success={res.success}",
          flush=True)

    chi2_dict = _chi2_per_probe(res.x, label, infra, data_wp_esd, data_wtheta,
                                masks, shape_cache, f_sys_wtheta)
    chi2_total = chi2_dict["chi2_wp"] + chi2_dict["chi2_esd"] + chi2_dict["chi2_wt"]
    ndof_total = (chi2_dict["ndof_wp"] + chi2_dict["ndof_esd"]
                  + chi2_dict["ndof_wt"] + n_free)

    print(f"  [{label}] chi2/dof — wp:{chi2_dict['chi2_dof_wp']:.2f}  "
          f"esd:{chi2_dict['chi2_dof_esd']:.2f}  wtheta:{chi2_dict['chi2_dof_wt']:.2f}",
          flush=True)

    result = dict(
        label          = label,
        param_names    = param_names,
        params         = res.x.tolist(),
        **chi2_dict,
        chi2_total     = float(chi2_total),
        ndof_total     = int(ndof_total),
        chi2_dof_total = float(chi2_total / max(ndof_total, 1)),
        success        = bool(res.success),
        n_pts_wp       = n_pts["wp"],
        n_pts_esd      = n_pts["esd"],
        n_pts_wtheta   = n_pts["theta"],
        rp_min=rp_min, rp_max=rp_max,
        R_min=R_min,   R_max=R_max,
        theta_min_arcsec=theta_min, theta_max_arcsec=theta_max,
        esd_survey=esd_survey, f_sys_wtheta=f_sys_wtheta,
    )
    return result, shape_cache


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------

def run_mcmc(label: str, infra: _Infrastructure,
             map_result: dict, shape_cache: dict,
             n_walkers: int = 32, n_steps: int = 1000, n_burnin: int = 300,
             rp_min: float = 0.3, rp_max: float = 30.0,
             R_min: float = 0.1,  R_max: float = 30.0,
             theta_min: float = 8.0, theta_max: float = 300.0,
             esd_survey: str = "esd_hsc",
             f_sys_wtheta: float = 0.05) -> dict:
    import emcee

    data_wp_esd = load_wp_esd(label, esd_survey=esd_survey)
    data_wtheta = load_wtheta(label)
    masks = {
        "wp":    _wp_mask(data_wp_esd["rp_wp"],       rp_min, rp_max),
        "esd":   _esd_mask(data_wp_esd["rp_esd"],     R_min,  R_max),
        "theta": _theta_mask(data_wtheta["theta_arcsec"], theta_min, theta_max),
    }

    x_map  = np.array(map_result["params"])
    n_free = len(x_map)

    def lp(p):
        return log_prob(p, label, infra, data_wp_esd, data_wtheta,
                        masks, shape_cache, f_sys_wtheta)

    # Small scatter around MAP; HOD params wider, amplitude params narrower
    scales = np.array([0.05, 0.05, 0.05, 0.02, 0.05, 0.05, 0.05])
    pos = x_map[None, :] + scales[None, :] * np.random.randn(n_walkers, n_free)

    sampler = emcee.EnsembleSampler(n_walkers, n_free, lp)

    print(f"  [{label}] MCMC burn-in {n_burnin} steps ...", flush=True)
    pos, _, _ = sampler.run_mcmc(pos, n_burnin, progress=True)
    sampler.reset()

    print(f"  [{label}] MCMC production {n_steps} steps ...", flush=True)
    sampler.run_mcmc(pos, n_steps, progress=True)

    flat_chain = sampler.get_chain(flat=True)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = np.full(n_free, np.nan)

    medians = np.median(flat_chain, axis=0)
    lo      = np.percentile(flat_chain, 16, axis=0)
    hi      = np.percentile(flat_chain, 84, axis=0)

    return dict(
        label        = label,
        param_names  = map_result["param_names"],
        chain        = flat_chain,
        medians      = medians.tolist(),
        lo16         = lo.tolist(),
        hi84         = hi.tolist(),
        autocorr_tau = tau.tolist() if hasattr(tau, "tolist") else list(tau),
        n_walkers    = n_walkers,
        n_steps      = n_steps,
        n_burnin     = n_burnin,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bestfit(label: str, infra: _Infrastructure, params: list,
                 data_wp_esd: dict, data_wtheta: dict,
                 shape_cache: dict, out_path: Path,
                 chi2_dict: dict | None = None,
                 rp_min=0.3, rp_max=30.0, R_min=0.1, R_max=30.0,
                 theta_min=8.0, theta_max=300.0) -> None:
    """3-panel figure: wp | ESD | wtheta, each with residual sub-panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log10mmin, sigma_logm, log10m1, alpha, kappa = params[:5]
    log10_A_gas, log10_A_AGN = params[5], params[6]
    hp = _hod_params(label, log10mmin=log10mmin, sigma_logm=sigma_logm,
                     log10m1=log10m1, alpha=alpha, kappa=kappa)
    s     = SAMPLES[label]
    z_eff = s["zmean"]

    h = float(data_wp_esd["h"])
    wp_mod  = np.asarray(infra.fhmp.wp(
        data_wp_esd["rp_wp"], pi_max=100.0, z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=hp), dtype=float)
    esd_mod = np.asarray(infra.fhmp.delta_sigma(
        data_wp_esd["rp_esd"], z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=hp), dtype=float) / h

    cache_key = (round(log10mmin, 4), round(sigma_logm, 4),
                 round(log10m1, 4), round(alpha, 4), round(kappa, 4))
    shapes    = shape_cache[cache_key]
    A_gas, A_AGN = 10.0**log10_A_gas, 10.0**log10_A_AGN
    wt_gas  = A_gas * shapes["gas"]
    wt_agn  = A_AGN * shapes["agn"]
    wt_mod  = wt_gas + wt_agn

    mk_wp    = _wp_mask(data_wp_esd["rp_wp"],       rp_min, rp_max)
    mk_esd   = _esd_mask(data_wp_esd["rp_esd"],     R_min,  R_max)
    mk_theta = _theta_mask(data_wtheta["theta_arcsec"], theta_min, theta_max)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    title = (f"{label}: $\\log_{{10}} M_* > {s['log10ms_min']}$, "
             f"$z_{{\\rm mean}}={z_eff:.3f}$")
    fig.suptitle(title, fontsize=11)

    # --- wp ---
    ax, ax_r = axes[0, 0], axes[1, 0]
    rp = data_wp_esd["rp_wp"]
    wp = data_wp_esd["wp"]
    wp_err = np.sqrt(np.diag(data_wp_esd["cov_wp"]))
    ax.errorbar(rp[~mk_wp], wp[~mk_wp], yerr=wp_err[~mk_wp],
                fmt="o", ms=3, color="0.65")
    ax.errorbar(rp[mk_wp], wp[mk_wp], yerr=wp_err[mk_wp],
                fmt="o", ms=4, color="k", label="Data")
    ax.plot(rp, wp_mod, "-", lw=2, color="C0", label="Model")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$w_p(r_p)$ [Mpc/$h$]")
    if chi2_dict:
        ax.set_title(f"$\\chi^2/\\nu={chi2_dict['chi2_dof_wp']:.2f}$", fontsize=9)
    ax.legend(fontsize=8)
    ratio = wp / wp_mod
    ax_r.errorbar(rp[~mk_wp], ratio[~mk_wp], yerr=wp_err[~mk_wp]/wp_mod[~mk_wp],
                  fmt="o", ms=3, color="0.65")
    ax_r.errorbar(rp[mk_wp], ratio[mk_wp], yerr=wp_err[mk_wp]/wp_mod[mk_wp],
                  fmt="o", ms=4, color="k")
    ax_r.axhline(1.0, color="C0", lw=1)
    ax_r.axhline(1.1, color="gray", lw=0.7, ls="--")
    ax_r.axhline(0.9, color="gray", lw=0.7, ls="--")
    ax_r.set_ylim(0.3, 1.9); ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax_r.set_ylabel("Data/Model")

    # --- ESD ---
    ax, ax_r = axes[0, 1], axes[1, 1]
    R   = data_wp_esd["rp_esd"]
    esd = data_wp_esd["esd"]
    esd_err = np.sqrt(np.diag(data_wp_esd["cov_esd"]))
    ax.errorbar(R[~mk_esd], esd[~mk_esd], yerr=esd_err[~mk_esd],
                fmt="o", ms=3, color="0.65")
    ax.errorbar(R[mk_esd], esd[mk_esd], yerr=esd_err[mk_esd],
                fmt="o", ms=4, color="k", label="Data")
    ax.plot(R, esd_mod, "-", lw=2, color="C0", label="Model")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,{\rm pc}^{-2}$]")
    if chi2_dict:
        ax.set_title(f"$\\chi^2/\\nu={chi2_dict['chi2_dof_esd']:.2f}$", fontsize=9)
    ratio = esd / esd_mod
    ax_r.errorbar(R[~mk_esd], ratio[~mk_esd], yerr=esd_err[~mk_esd]/esd_mod[~mk_esd],
                  fmt="o", ms=3, color="0.65")
    ax_r.errorbar(R[mk_esd], ratio[mk_esd], yerr=esd_err[mk_esd]/esd_mod[mk_esd],
                  fmt="o", ms=4, color="k")
    ax_r.axhline(1.0, color="C0", lw=1)
    ax_r.axhline(1.1, color="gray", lw=0.7, ls="--")
    ax_r.axhline(0.9, color="gray", lw=0.7, ls="--")
    ax_r.set_ylim(0.3, 1.9); ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$R$ [Mpc/$h$]")
    ax_r.set_ylabel("Data/Model")

    # --- wtheta ---
    ax, ax_r = axes[0, 2], axes[1, 2]
    th = data_wtheta["theta_arcsec"]
    wt = data_wtheta["wtheta"]
    wt_err = data_wtheta["wtheta_err"]
    ax.errorbar(th[~mk_theta], wt[~mk_theta], yerr=wt_err[~mk_theta],
                fmt="o", ms=3, color="0.65")
    ax.errorbar(th[mk_theta], wt[mk_theta], yerr=wt_err[mk_theta],
                fmt="o", ms=4, color="k", label="Data")
    ax.plot(th, wt_mod, "-",  lw=2.0, color="C0", label="Gas + AGN")
    ax.plot(th, wt_gas, "--", lw=1.5, color="C2",
            label=f"Gas  $\\log A={log10_A_gas:.2f}$")
    ax.plot(th, wt_agn, ":",  lw=1.5, color="C1",
            label=f"AGN  $\\log A={log10_A_AGN:.2f}$")
    ax.axvline(30.0, ls="--", color="C3", lw=1, alpha=0.7, label="PSF FWHM")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$w_\theta(\theta)$")
    if chi2_dict:
        ax.set_title(f"$\\chi^2/\\nu={chi2_dict['chi2_dof_wt']:.2f}$", fontsize=9)
    ax.legend(fontsize=7)
    ratio = wt / wt_mod
    wt_err_eff = np.sqrt(wt_err**2 + (0.05 * np.abs(wt))**2)
    ax_r.errorbar(th[~mk_theta], ratio[~mk_theta],
                  yerr=wt_err_eff[~mk_theta]/wt_mod[~mk_theta],
                  fmt="o", ms=3, color="0.65")
    ax_r.errorbar(th[mk_theta], ratio[mk_theta],
                  yerr=wt_err_eff[mk_theta]/wt_mod[mk_theta],
                  fmt="o", ms=4, color="k")
    ax_r.axhline(1.0, color="C0", lw=1)
    ax_r.axhline(1.1, color="gray", lw=0.7, ls="--")
    ax_r.axhline(0.9, color="gray", lw=0.7, ls="--")
    ax_r.set_ylim(0.3, 1.9); ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$\theta$ [arcsec]")
    ax_r.set_ylabel("Data/Model")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{label}] bestfit figure → {out_path}")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_map(result: dict, out_dir: Path) -> None:
    label = result["label"]
    out = out_dir / f"{label}_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "chain"}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [{label}] MAP result → {out}")


def _save_chain(mcmc: dict, out_dir: Path) -> None:
    label = mcmc["label"]
    chain = mcmc.pop("chain")
    try:
        import h5py
        path = out_dir / f"{label}_chain.h5"
        with h5py.File(path, "w") as hf:
            hf.create_dataset("chain", data=chain, compression="gzip")
            for k, v in mcmc.items():
                hf.attrs[k] = json.dumps(v)
        print(f"  [{label}] MCMC chain → {path}")
    except ImportError:
        path = out_dir / f"{label}_chain.npy"
        np.save(path, chain)
        print(f"  [{label}] MCMC chain → {path}  (h5py unavailable)")
    summary = {k: v for k, v in mcmc.items()}
    with open(out_dir / f"{label}_mcmc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Joint MAP + MCMC fit: wp(rp) + ΔΣ(R) + galaxy×X-ray"
    )
    p.add_argument("--sample", nargs="+", default=["S1"],
                   help="Sample labels (S1 S3 S5 S7) or 'all'")
    p.add_argument("--mode", choices=["map", "mcmc", "both"], default="map")
    p.add_argument("--esd-survey", default="esd_hsc",
                   choices=["esd_hsc", "esd_des", "esd_kids"])
    p.add_argument("--rp-min",  type=float, default=0.3)
    p.add_argument("--rp-max",  type=float, default=30.0)
    p.add_argument("--R-min",   type=float, default=0.1)
    p.add_argument("--R-max",   type=float, default=30.0)
    p.add_argument("--theta-min", type=float, default=8.0)
    p.add_argument("--theta-max", type=float, default=300.0)
    p.add_argument("--n-walkers", type=int, default=32)
    p.add_argument("--n-steps",   type=int, default=1000)
    p.add_argument("--n-burnin",  type=int, default=300)
    p.add_argument("--f-sys",     type=float, default=0.05)
    p.add_argument("--no-plot",   action="store_true")
    p.add_argument("--out-dir",   default=None)
    return p.parse_args()


def main():
    args = _parse_args()

    labels = list(SAMPLES.keys()) if "all" in args.sample else args.sample
    for lb in labels:
        if lb not in SAMPLES:
            raise ValueError(f"Unknown sample '{lb}'. Choose from {list(SAMPLES)}")

    out_dir = Path(args.out_dir) if args.out_dir else _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    infra = _Infrastructure()
    all_map = []

    for label in labels:
        print(f"\n{'='*60}\n  Sample {label}  "
              f"(log10M*>{SAMPLES[label]['log10ms_min']}, z={SAMPLES[label]['zmean']:.3f})\n"
              f"{'='*60}", flush=True)

        if args.mode in ("map", "both"):
            t0 = time.time()
            map_res, shape_cache = run_map(
                label, infra,
                rp_min=args.rp_min, rp_max=args.rp_max,
                R_min=args.R_min,   R_max=args.R_max,
                theta_min=args.theta_min, theta_max=args.theta_max,
                esd_survey=args.esd_survey, f_sys_wtheta=args.f_sys,
            )
            print(f"  MAP wall-clock: {time.time()-t0:.1f}s", flush=True)
            _save_map(map_res, out_dir)
            all_map.append(map_res)

            if not args.no_plot:
                data_wp_esd = load_wp_esd(label, esd_survey=args.esd_survey)
                data_wtheta = load_wtheta(label)
                chi2_dict = _chi2_per_probe(
                    np.array(map_res["params"]), label, infra,
                    data_wp_esd, data_wtheta,
                    masks={
                        "wp":    _wp_mask(data_wp_esd["rp_wp"],       args.rp_min, args.rp_max),
                        "esd":   _esd_mask(data_wp_esd["rp_esd"],     args.R_min,  args.R_max),
                        "theta": _theta_mask(data_wtheta["theta_arcsec"],
                                             args.theta_min, args.theta_max),
                    },
                    shape_cache=shape_cache, f_sys_wtheta=args.f_sys,
                )
                plot_bestfit(
                    label, infra, map_res["params"],
                    data_wp_esd, data_wtheta, shape_cache,
                    out_dir / f"{label}_bestfit.pdf",
                    chi2_dict=chi2_dict,
                    rp_min=args.rp_min, rp_max=args.rp_max,
                    R_min=args.R_min,   R_max=args.R_max,
                    theta_min=args.theta_min, theta_max=args.theta_max,
                )
        else:
            # MCMC-only: run MAP first to seed walkers
            map_res, shape_cache = run_map(
                label, infra,
                rp_min=args.rp_min, rp_max=args.rp_max,
                R_min=args.R_min,   R_max=args.R_max,
                theta_min=args.theta_min, theta_max=args.theta_max,
                esd_survey=args.esd_survey, f_sys_wtheta=args.f_sys,
            )

        if args.mode in ("mcmc", "both"):
            mcmc_res = run_mcmc(
                label, infra, map_res, shape_cache,
                n_walkers=args.n_walkers, n_steps=args.n_steps,
                n_burnin=args.n_burnin,
                rp_min=args.rp_min, rp_max=args.rp_max,
                R_min=args.R_min,   R_max=args.R_max,
                theta_min=args.theta_min, theta_max=args.theta_max,
                esd_survey=args.esd_survey, f_sys_wtheta=args.f_sys,
            )
            _save_chain(mcmc_res, out_dir)

            if not args.no_plot:
                try:
                    import corner
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    chain = np.load(str(out_dir / f"{label}_chain.npy"), allow_pickle=True) \
                        if (out_dir / f"{label}_chain.npy").exists() else None
                    if chain is None:
                        try:
                            import h5py
                            with h5py.File(out_dir / f"{label}_chain.h5") as hf:
                                chain = hf["chain"][:]
                        except Exception:
                            pass
                    if chain is not None:
                        names_tex = [r"$\log_{10}M_{\min}$", r"$\sigma_{\log M}$",
                                     r"$\log_{10}M_1$", r"$\alpha$", r"$\kappa$",
                                     r"$\log_{10}A_{\rm gas}$", r"$\log_{10}A_{\rm AGN}$"]
                        fig = corner.corner(chain, labels=names_tex,
                                            quantiles=[0.16, 0.50, 0.84],
                                            show_titles=True)
                        fig.suptitle(f"{label} MCMC posterior", y=1.01)
                        cpath = out_dir / f"{label}_corner.pdf"
                        fig.savefig(cpath, dpi=120, bbox_inches="tight")
                        plt.close(fig)
                        print(f"  [{label}] corner → {cpath}")
                except ImportError:
                    pass

    # Summary table
    if all_map:
        print(f"\n{'='*80}")
        hdr = f"{'label':6s}  {'log10mmin':>11s}  {'log10m1':>9s}  {'alpha':>7s}"
        hdr += f"  {'logAgas':>9s}  {'logAAGN':>9s}  {'chi2wp':>8s}  {'chi2esd':>8s}  {'chi2wt':>8s}"
        print(hdr)
        print("-" * 80)
        for r in all_map:
            p = r["params"]
            print(f"{r['label']:6s}  {p[0]:>11.3f}  {p[2]:>9.3f}  {p[3]:>7.3f}"
                  f"  {p[5]:>9.3f}  {p[6]:>9.3f}"
                  f"  {r['chi2_dof_wp']:>8.2f}  {r['chi2_dof_esd']:>8.2f}"
                  f"  {r['chi2_dof_wt']:>8.2f}")
        print(f"{'='*80}")
    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
