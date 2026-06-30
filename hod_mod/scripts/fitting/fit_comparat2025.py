"""Joint fit of galaxy × X-ray w_θ(θ) + wp(rp) + SMF Phi(M*) to Comparat+2025 data.

Data sources:
    w_θ  : ~/data/zenodo/LSDR10_GALxEVT/  (zenodo record 15111974)
    wp   : ~/software/sum_stat/data/BGS_Mstar{10.0,…}/  (sum_stat HDF5)
    SMF  : same sum_stat HDF5 file ("smf" group), fit jointly with wp using
           the full smf+wp jackknife cross-covariance
           (``SumStatReader.joint_bgs(probes=("smf","wp"))``).

Model:
    HaloModelCrossSpectra with GasDensityDPM (model 2) + King PSF (θ_c = 8.64 arcsec)
    AGN: HamAGNModel (abundance-matching against Aird+2015 XLF, Comparat+2019)
    HOD: ZuMandelbaum15HODModel

Free parameters (9):
    log10_A_gas         DPM hot-gas amplitude
    beta_gas            DPM density mass-scaling slope (n_e ∝ M^beta_n); GAS.py target 0.20
    beta_pressure       DPM pressure mass-scaling slope (P ∝ M^beta_P); GAS.py target 0.80
    log10_A_AGN         AGN amplitude: fudge factor on the predicted HOD AGN
                        cross-power (--agn-model hod, default), or free King-PSF
                        amplitude (--agn-model ham)
    log10m_star_thresh  ZuMandelbaum15 stellar-mass threshold (≈ log10ms_min of sample)
    sigma_lnmstar       ZuMandelbaum15 SHMR log-normal scatter in M*
    lg_m1h              ZuMandelbaum15 SHMR characteristic halo mass scale
    alpha_sat           ZuMandelbaum15 satellite occupation slope
    fc                  ZuMandelbaum15 central completeness fraction (normalisation
                        knob freed so the fit can match the observed SMF Phi(M*)
                        without relying on log10m_star_thresh/lg_m1h alone)

Scale cuts:
    w_θ  : theta_min_arcsec (default 8) – theta_max_arcsec (default 300)
    wp   : rp > rp_min (default 0.02 Mpc/h)
    SMF  : all bins with phi > 0 kept (no scale cut)

Output (to results/fits/comparat2025/):
    <label>_map.json            MAP best-fit parameters + chi2/dof
    <label>_chain.h5            emcee chain (HDF5; falls back to .npy)
    <label>_mcmc_summary.json   MCMC medians and 68% credibles
    <label>_corner.pdf          posterior corner plot
    <label>_bestfit.pdf         data vs best-fit figure (w_θ + wp)

Usage::

    # MAP only, all 9 parameters free
    python -m hod_mod.scripts.fitting.fit_comparat2025 --sample S1

    # MAP + MCMC
    python -m hod_mod.scripts.fitting.fit_comparat2025 --sample S1 --mode both

    # Quick MAP, no plots, custom wp scale cut
    python -m hod_mod.scripts.fitting.fit_comparat2025 --sample S1 --mode map --no-plot --rp-min 0.1

References
----------
Comparat et al. 2025, arXiv:2503.19796  (galaxy × eROSITA X-ray)
Oppenheimer et al. 2025, arXiv:2505.14782  (DPM gas profile model)
Comparat et al. 2019, A&A 622, A12  (HAM AGN model)
Zu & Mandelbaum 2015, MNRAS 454, 1161  (iHOD SHMR)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.special import j0

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

# --------------------------------------------------------------------------
# hod_mod imports
# --------------------------------------------------------------------------
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf, _BACKENDS, _EMULATOR_BACKENDS
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import GasDensityDPM
from hod_mod.connection.hod import (
    ZuMandelbaum15HODModel, n_cen_thresh_zu15, n_sat_thresh_zu15,
)
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.ham import HamAGNModel
from hod_mod.agn.hod import HODAgnModel
from hod_mod.agn.xray import XrayAGNModel
from hod_mod.data_io.sum_stat_reader import SumStatReader
from hod_mod.paths import results_root

# Enable JAX XLA compilation cache. The angular_cl_gX JIT traces in Python
# (~200s) but the XLA compilation step is fast (<1s on CPU). Lower the
# threshold from 1.0s → 0s so even fast XLA compilations are persisted.
# This avoids re-tracing in every fresh Python process.
try:
    import jax
    _JAX_CACHE = Path(os.path.expanduser("~/.cache/jax_xla_comparat2025"))
    _JAX_CACHE.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_ZENODO_DIR = Path(os.path.expanduser(
    "~/data/zenodo/LSDR10_GALxEVT/Measurements_Xcorr_Stacks/XCORR"
))
_GAL_DIR = Path(os.path.expanduser(
    "~/data/zenodo/LSDR10_GALxEVT/Galaxy_samples/data_and_randoms"
))
_SUM_STAT_DIR = Path(os.path.expanduser("~/software/sum_stat/data"))
_RESULTS_DIR  = results_root() / "fits" / "comparat2025"

# sum_stat subdirectory per sample (only samples with BGS data)
_SUM_STAT_DIRS: dict[str, str] = {
    "S1": "BGS_Mstar10.0",
    "S3": "BGS_Mstar10.5",
    "S5": "BGS_Mstar11.0",
    "S7": "BGS_Mstar11.5",
}

# --------------------------------------------------------------------------
# Sample definitions (Comparat+2025 Table 1)
# --------------------------------------------------------------------------
SAMPLES = {
    "S1": dict(log10ms_min=10.00, zmax=0.18, zmean=0.135, N=2759238),
    "S2": dict(log10ms_min=10.25, zmax=0.22, zmean=0.162, N=3308841),
    "S3": dict(log10ms_min=10.50, zmax=0.26, zmean=0.191, N=3263228),
    "S4": dict(log10ms_min=10.75, zmax=0.31, zmean=0.226, N=2802710),
    "S5": dict(log10ms_min=11.00, zmax=0.35, zmean=0.252, N=1619838),
    "S6": dict(log10ms_min=11.25, zmax=0.35, zmean=0.255, N=541855),
    "S7": dict(log10ms_min=11.50, zmax=0.35, zmean=0.261, N=120882),
}

# ZuMandelbaum15HODModel initial parameters per sample.
# (log10m_star_thresh, sigma_lnmstar, lg_m1h, alpha_sat, fc)
# log10m_star_thresh = sample stellar-mass cut; other values: ZM15 defaults.
# fc (central completeness fraction) free since the SMF joint-fit was added;
# 0.86 is the ZM15 SDSS default (ZuMandelbaum15HODModel.default_params()).
_TABLE3 = {
    "S1": ( 9.56, 0.55, 10.00, 0.88, 0.86),   # MAP run6 best-fit (chi2/dof=23.25)
    "S2": (10.25, 0.50, 12.10, 1.00, 0.86),
    "S3": (10.50, 0.50, 12.10, 1.00, 0.86),
    "S4": (10.75, 0.50, 12.10, 1.00, 0.86),
    "S5": (11.00, 0.50, 12.10, 1.00, 0.86),
    "S6": (11.25, 0.50, 12.10, 1.00, 0.86),
    "S7": (11.50, 0.50, 12.10, 1.00, 0.86),
}

# Comparat+2025 (A&A 697, A173) Table 4 — surface-brightness-profile fit to the
# galaxy x event cross-correlation: alpha_SR with asymmetric 1-sigma errors
# (alpha_SR, +sigma_hi, -sigma_lo).
_ALPHA_SR_C25 = {
    "S1": (1.629, 0.091, 0.089),
    "S2": (1.573, 0.077, 0.070),
    "S3": (1.612, 0.068, 0.072),
    "S4": (1.654, 0.166, 0.104),
    "S5": (1.634, 0.066, 0.094),
    "S6": (1.713, 0.087, 0.143),
    "S7": (1.544, 0.256, 0.244),
}
_LOG10_L0_C25 = 44.7   # log10(L0/[erg/s]) at M500c = 1e15 Msun, z=0 (Comparat+2025 Eq. ?)


def _comparat25_lx_sr(m500c_msun: np.ndarray, z: float, alpha_sr: float) -> np.ndarray:
    """Comparat+2025 (Table 4) Lx-M500c relation.

    log10(Lx) = log10(L0) + alpha_SR * log10(M500c/1e15) + 2*log10(E(z))
    """
    from hod_mod.scripts.validate_gas_profiles import _ez
    log10_lx = (_LOG10_L0_C25 + alpha_sr * np.log10(m500c_msun / 1e15)
                + 2.0 * np.log10(_ez(z)))
    return 10.0 ** log10_lx

# --------------------------------------------------------------------------
# Cosmology (Planck 2018)
# --------------------------------------------------------------------------
_THETA_COSMO = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "H0":  _THETA_COSMO["h"] * 100.0,
    "Om0": _THETA_COSMO["Omega_m"],
    "Ob0": _THETA_COSMO["Omega_b"],
    "ns":  _THETA_COSMO["n_s"],
    "sigma8": 0.811,
}
_ASTROPY_COSMO = FlatLambdaCDM(
    H0=_COLOSSUS["H0"], Om0=_COLOSSUS["Om0"], Ob0=_COLOSSUS["Ob0"],
)

# --------------------------------------------------------------------------
# HOD helpers
# --------------------------------------------------------------------------

def _hod_params(
    label: str,
    log10m_star_thresh: float | None = None,
    sigma_lnmstar: float | None = None,
    lg_m1h: float | None = None,
    alpha_sat: float | None = None,
    fc: float | None = None,
) -> dict:
    """Build ZuMandelbaum15HODModel parameter dict, defaulting to _TABLE3 values.

    In fixed-ZM15 mode (``_FIT_CFG is not None``) all 13 ZM15 SHMR+satellite
    parameters are taken from the loaded MAP json and the per-sample
    ``log10m_star_thresh`` from ``_FIT_CFG["thresh"]``; the five ZM15 keyword
    arguments are ignored (they are no longer free parameters of the fit).
    """
    if _FIT_CFG is not None:
        base = ZuMandelbaum15HODModel.default_params()
        base.update(_FIT_CFG["zm15"])                       # all 13 fitted ZM15 params
        base["log10m_star_thresh"] = _FIT_CFG["thresh"][label]
        return base

    ms0, sig0, m1h0, a0, fc0 = _TABLE3[label]
    base = ZuMandelbaum15HODModel.default_params()
    base["log10m_star_thresh"] = ms0  if log10m_star_thresh is None else log10m_star_thresh
    base["sigma_lnmstar"]      = sig0 if sigma_lnmstar      is None else sigma_lnmstar
    base["lg_m1h"]             = m1h0 if lg_m1h             is None else lg_m1h
    base["alpha_sat"]          = a0   if alpha_sat           is None else alpha_sat
    base["fc"]                 = fc0  if fc                  is None else fc
    return base


# --------------------------------------------------------------------------
# Data loading — w_θ
# --------------------------------------------------------------------------

def _zenodo_fname(label: str) -> Path:
    s = SAMPLES[label]
    N = f"{s['N']:07d}"
    matches = sorted(_ZENODO_DIR.glob(f"*_N_{N}_GALxEVT_wtheta.fits"))
    if not matches:
        ms = f"{s['log10ms_min']:.1f}".rstrip("0").rstrip(".")
        zm = f"{s['zmax']:.2f}"
        return _ZENODO_DIR / f"LS10_VLIM_ANY_{ms}_Mstar_12.0_0.05_z_{zm}_N_{N}_GALxEVT_wtheta.fits"
    return matches[0]


def _galaxy_fname(label: str) -> Path:
    s = SAMPLES[label]
    N = f"{s['N']:07d}"
    matches = sorted(_GAL_DIR.glob(f"*_N_{N}_DATA.fits"))
    return matches[0] if matches else Path("__missing__")


def load_data(label: str) -> dict:
    """Load w_θ(θ) from zenodo FITS file."""
    path = _zenodo_fname(label)
    if not path.exists():
        raise FileNotFoundError(
            f"Zenodo data not found: {path}\n"
            f"Expected ~/data/zenodo/LSDR10_GALxEVT/ to be populated."
        )
    d = fits.open(path)[1].data
    return dict(
        theta_deg    = np.array(d["theta"],       dtype=float),
        theta_rad    = np.array(d["theta"],       dtype=float) * np.pi / 180.0,
        theta_arcsec = np.array(d["theta"],       dtype=float) * 3600.0,
        wtheta       = np.array(d["wtheta"],      dtype=float),
        wtheta_err   = np.array(d["wtheta_err"],  dtype=float),
        R_kpc        = np.array(d["theta"] * d["convert_theta_to_kpc"], dtype=float),
        # S^R_X (random x events) background surface brightness [erg kpc^-2 s^-1];
        # constant per sample.  Converts w(theta) -> physical: S_X = (1+w) * S^R_X
        # (DP83).  Used as the absolute normalisation when folding the true ECF.
        beckground   = np.array(d["beckground"],  dtype=float),
        R_to_kpc     = np.array(d["convert_theta_to_kpc"], dtype=float),
    )


# --------------------------------------------------------------------------
# Data loading — wp(rp)
# --------------------------------------------------------------------------

def _sum_stat_path(label: str) -> Path:
    """Locate sum_stat HDF5 file for the given sample."""
    if label not in _SUM_STAT_DIRS:
        raise FileNotFoundError(
            f"No sum_stat directory configured for sample {label}. "
            f"Available: {list(_SUM_STAT_DIRS)}"
        )
    d = _SUM_STAT_DIR / _SUM_STAT_DIRS[label]
    N_str = f"{SAMPLES[label]['N']:07d}"
    matches = sorted(d.glob(f"*_N_{N_str}_joint_smf-wp-esd*.h5"))
    if not matches:
        matches = sorted(d.glob("*.h5"))
    if not matches:
        raise FileNotFoundError(f"No HDF5 file found for sample {label} in {d}")
    return matches[0]


def load_wp_data(label: str, rp_min: float = 0.02) -> dict:
    """Load wp(rp) from sum_stat HDF5, applying rp > rp_min cut.

    Returns dict with keys: rp, wp, cov, icov, pi_max.
    """
    path = _sum_stat_path(label)
    reader = SumStatReader.from_hdf5(str(path))
    jt = reader.joint_bgs(probes=("wp",))
    rp  = np.asarray(jt["rp_wp"], dtype=float)
    wp  = np.asarray(jt["data_vector"], dtype=float)
    cov = np.asarray(jt["cov"], dtype=float)
    mask = rp > rp_min
    rp_m = rp[mask]
    wp_m = wp[mask]
    cov_m = cov[np.ix_(mask, mask)]
    try:
        icov = np.linalg.inv(cov_m)
    except np.linalg.LinAlgError:
        icov = np.diag(1.0 / np.maximum(np.diag(cov_m), 1e-30))
    pi_max = float(getattr(reader, "_pi_max", 100.0))
    return dict(rp=rp_m, wp=wp_m, cov=cov_m, icov=icov, pi_max=pi_max)


def _apply_wp_syst(data_wp: dict, f_sys: float) -> dict:
    """Return data_wp with icov inflated by a fractional systematic floor.

    Adds (f_sys * |wp|)^2 to the diagonal of the jackknife covariance, then
    re-inverts.  Called once at data-load time so the per-call likelihood is
    unchanged.  Returns the original dict unchanged when f_sys <= 0.
    """
    if f_sys <= 0.0:
        return data_wp
    syst_var = (f_sys * np.abs(data_wp["wp"])) ** 2
    cov_eff  = data_wp["cov"] + np.diag(syst_var)
    try:
        icov_eff = np.linalg.inv(cov_eff)
    except np.linalg.LinAlgError:
        icov_eff = np.diag(1.0 / np.maximum(np.diag(cov_eff), 1e-30))
    return {**data_wp, "icov": icov_eff}


def load_smf_data(label: str) -> dict:
    """Load the stellar-mass function Phi(M*) from sum_stat HDF5 (for plotting).

    The raw HDF5 ``phi`` is in physical Mpc^-3 dex^-1; the halo-model prediction
    (:func:`_predict_smf`) is in the **h-units** convention (Mpc/h)^-3 dex^-1
    (= h^3 Mpc^-3).  We therefore divide by h^3 here — exactly as
    :meth:`~hod_mod.data_io.sum_stat_reader.SumStatReader.smf` does — so the
    diagnostic panels compare model and data in the same units.  (Reading the raw
    phi without this factor made the plotted SMF/n_gal ~h^-3 ≈ 3.2x too high.)

    Returns dict with keys: log10mstar, phi, phi_err [(Mpc/h)^-3 dex^-1].
    """
    import h5py
    h3 = float(_THETA_COSMO["h"]) ** 3
    path = _sum_stat_path(label)
    with h5py.File(str(path), "r") as f:
        grp_name = list(f["smf"].keys())[0]
        g = f["smf"][grp_name]
        log10mstar = np.asarray(g["log10mstar_centres"], dtype=float)
        phi        = np.asarray(g["phi"],                dtype=float) / h3
        phi_err    = np.asarray(g["phi_err"],            dtype=float) / h3
    mask = phi > 0
    return dict(log10mstar=log10mstar[mask], phi=phi[mask], phi_err=phi_err[mask])


def load_smf_wp_data(label: str, rp_min: float = 0.02) -> dict:
    """Load the joint SMF Phi(M*) + wp(rp) data vector and covariance.

    Uses ``SumStatReader.joint_bgs(probes=("smf", "wp"))`` so the SMF-wp
    cross-covariance (both measured from the same galaxy catalog jackknife)
    is preserved, rather than fitting the two probes as if independent.

    Returns dict with keys: log10mstar, phi, rp, wp, cov, icov, pi_max,
    n_smf, n_wp.
    """
    path = _sum_stat_path(label)
    reader = SumStatReader.from_hdf5(str(path))
    jt = reader.joint_bgs(probes=("smf", "wp"))
    sl_smf = jt["slices_out"]["smf"]
    sl_wp  = jt["slices_out"]["wp"]

    log10mstar = np.asarray(reader.smf()["log10mstar"], dtype=float)
    phi_all    = np.asarray(jt["data_vector"][sl_smf], dtype=float)
    rp_all     = np.asarray(jt["rp_wp"], dtype=float)
    wp_all     = np.asarray(jt["data_vector"][sl_wp], dtype=float)

    smf_mask = phi_all > 0
    wp_mask  = rp_all > rp_min
    idx      = np.nonzero(np.concatenate([smf_mask, wp_mask]))[0]

    cov_m = np.asarray(jt["cov"], dtype=float)[np.ix_(idx, idx)]
    try:
        icov = np.linalg.inv(cov_m)
    except np.linalg.LinAlgError:
        icov = np.diag(1.0 / np.maximum(np.diag(cov_m), 1e-30))

    pi_max = float(getattr(reader, "_pi_max", 100.0))
    return dict(
        log10mstar=log10mstar[smf_mask], phi=phi_all[smf_mask],
        rp=rp_all[wp_mask], wp=wp_all[wp_mask],
        cov=cov_m, icov=icov, pi_max=pi_max,
        n_smf=int(smf_mask.sum()), n_wp=int(wp_mask.sum()),
    )


def _apply_smfwp_syst(data: dict, f_sys: float) -> dict:
    """Inflate the joint SMF+wp covariance diagonal by a fractional
    systematic floor on each data point, then re-invert.

    Mirrors ``_apply_wp_syst`` but operates on the concatenated
    [phi, wp] data vector / joint covariance from ``load_smf_wp_data``.
    """
    if f_sys <= 0.0:
        return data
    values   = np.concatenate([data["phi"], data["wp"]])
    syst_var = (f_sys * np.abs(values)) ** 2
    cov_eff  = data["cov"] + np.diag(syst_var)
    # DIAGONAL-only inverse: the full joint SMF+wp jackknife covariance is
    # near-singular (cond ~1e16; SMF variances ~1e-13 vs wp ~5e2), so its dense
    # inverse is numerically meaningless.  The likelihood uses only the diagonal
    # (number-density term + diagonal wp), so build a diagonal icov here.
    icov_eff = np.diag(1.0 / np.maximum(np.diag(cov_eff), 1e-30))
    return {**data, "cov": cov_eff, "icov": icov_eff}


def load_esd_data(label: str, survey: str) -> dict:
    """Load galaxy-galaxy lensing Delta_Sigma(rp) for one survey (diagnostic only).

    survey : one of "HSC", "DES", "KIDS".
    Returns dict with keys: rp [Mpc/h], delta_sigma, delta_sigma_err [Msun h/pc^2].
    """
    import h5py
    path = _sum_stat_path(label)
    suffix = f"_{survey.upper()}"
    with h5py.File(str(path), "r") as f:
        grp_name = next(k for k in f["esd"].keys() if k.endswith(suffix))
        g = f["esd"][grp_name]
        rp    = np.asarray(g["rp_centres"],    dtype=float)
        ds    = np.asarray(g["delta_sigma"],   dtype=float)
        cov   = np.asarray(g["cov"],           dtype=float)
    ds_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return dict(rp=rp, delta_sigma=ds, delta_sigma_err=ds_err)


# --------------------------------------------------------------------------
# n(z) builders
# --------------------------------------------------------------------------

def _build_nz_fast(label: str, n_pts: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Narrow Gaussian n(z) centred on zmean — fast approximation used in fitting."""
    s     = SAMPLES[label]
    z     = s["zmean"]
    dz    = min(0.02, s["zmax"] * 0.10)
    z_arr = np.linspace(max(0.01, z - 2.0 * dz), z + 2.0 * dz, n_pts)
    nz    = np.exp(-0.5 * ((z_arr - z) / dz) ** 2)
    return z_arr, nz / trapezoid(nz, z_arr)


def _build_nz_full(label: str, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Build n(z) histogram from the real galaxy catalog (validation only)."""
    s    = SAMPLES[label]
    path = _galaxy_fname(label)
    if not path.exists():
        return _build_nz_fast(label)
    d    = fits.open(path)[1].data
    z    = np.array(d["BEST_Z"], dtype=float)
    hist, edges = np.histogram(z, bins=n_bins, range=(0.02, s["zmax"] + 0.02), density=True)
    z_arr = 0.5 * (edges[:-1] + edges[1:])
    return z_arr, hist


# --------------------------------------------------------------------------
# Infrastructure (built once, shared across all samples)
# --------------------------------------------------------------------------

def _make_bnl():
    """Build the beyond-linear halo bias (Mead+2021), or None if unavailable.

    Applied consistently to the galaxy 2-halo terms via
    ``FullHaloModelPrediction(bnl_model=...)``.
    """
    try:
        from hod_mod.core.beyond_linear_bias import BeyondLinearBiasMead21
        return BeyondLinearBiasMead21()
    except Exception as exc:  # missing data table, etc.
        print(f"  WARNING: BeyondLinearBiasMead21 unavailable ({exc}); "
              f"falling back to linear (Tinker) 2-halo bias.", flush=True)
        return None


class _Infrastructure:
    """One-time build of the halo model stack (expensive CAMB + HAM precompute)."""

    def __init__(self, hmf_backend: str = "csst", agn_model: str = "hod",
                 agn_finc: float = 0.01, **hmf_kwargs):
        print(f"Building halo model infrastructure (CAMB + HMF[{hmf_backend}] + "
              f"AGN[{agn_model}]) ...", flush=True)
        t0 = time.time()
        pk_lin    = LinearPowerSpectrum()
        hmf       = make_hmf(hmf_backend, pk_func=pk_lin.pk_linear, **hmf_kwargs)
        hp        = HaloProfile(_COLOSSUS, cm_relation="diemer19")
        bnl       = _make_bnl()                       # beyond-linear halo bias (Mead+2021)
        hod       = ZuMandelbaum15HODModel(hmf, hmf.bias)
        self.fhmp = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
        self.pk_lin = pk_lin
        self.hp     = hp
        self.hmf    = hmf                             # shared HMF (e.g. CSST) reused by AGN models
        self.bnl    = bnl
        self.dp   = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=200)
        self.agn_model_choice = agn_model
        self.agn_finc         = float(agn_finc)
        self._agn_by_sample: dict = {}
        self._cooling_fn = None   # lazy full-DPM APEC cooling table (gas-build presets)
        if agn_model == "ham":
            # Shared HAM model (sample-independent). HAM table precomputed (~12 s).
            # Prefer the galaxy model's HMF for consistency, but some emulator
            # backends (e.g. CSST) lack the private ``_pk`` the HAM precompute
            # needs — fall back to HamAGNModel's own default (Tinker08) then.
            try:
                self.agn = HamAGNModel(pk_lin=pk_lin, hmf=hmf)
            except AttributeError:
                print(f"  WARNING: HamAGNModel incompatible with HMF[{hmf_backend}] "
                      f"(_pk missing); using its default Tinker08 HMF.", flush=True)
                self.agn = HamAGNModel(pk_lin=pk_lin)
        elif agn_model == "xray":
            # Shared parametric L_X(M*) X-ray AGN model (sample-independent).
            self.agn = XrayAGNModel()
        else:
            # HOD AGN models are sample-specific (z_mean / selection); built lazily
            # per sample in use_agn_for(). Start with no AGN attached.
            self.agn = None
        self.cross = HaloModelCrossSpectra(
            self.fhmp, density_profile=self.dp, agn_model=self.agn
        )
        self._ecf_fixed = None
        print(f"  done in {time.time() - t0:.1f}s", flush=True)

    def enable_ecf(self, sample: str):
        """Fold the true per-component eROSITA ECF (TM0 survey ARF+RMF) into the
        cross-power as **tabulated band averages**: a per-halo gas weight
        ``ECF_gas(kT(M))`` (kT from the Lovisari+2020 kT-M500c relation) and the
        AGN ECF_AGN(Γ=1.9).  Returns ``ecf_fixed`` (for K_abs).
        """
        import numpy as np
        from hod_mod.gas import load_ecf_tables
        from hod_mod.scripts import validate_gas_profiles as vgp
        gas_of_T, ecf_agn, ecf_fixed = load_ecf_tables(sample)
        z = float(SAMPLES[sample]["zmean"])

        def ecf_gas_of_mass(m200_h):                # m200 [Msun/h] -> ECF_gas(kT(M))
            m200_h = np.asarray(m200_h, dtype=float)
            r200 = vgp._r200(m200_h, z); c200 = vgp._c200_approx(m200_h)
            m500_h, _ = vgp.m200_to_m500c(m200_h, c200, r200, vgp._rho_crit_z(z))
            kT = vgp._lovisari20_kt(m500_h / vgp._H, z=z)
            return np.asarray(gas_of_T(kT), dtype=float)

        self.cross._ecf_gas_table = ecf_gas_of_mass
        self.cross._ecf_agn = ecf_agn
        self._ecf_fixed = ecf_fixed
        return ecf_fixed

    def use_agn_for(self, label: str) -> None:
        """Attach the AGN model appropriate for sample *label* to ``self.cross``.

        For ``agn_model='ham'`` this is the shared (sample-independent) HamAGNModel.
        For ``agn_model='hod'`` a per-sample :class:`HODAgnModel` (built once and
        cached) is attached, with the sample's z_mean/z_max and the configured
        ``agn_finc`` duty cycle.
        """
        if self.agn_model_choice in ("ham", "xray"):
            self.cross._agn = self.agn
            self.cross._agn_has_hod = False
            return
        if label not in self._agn_by_sample:
            s = SAMPLES[label]
            if self.agn_model_choice == "duty_cycle":
                # Lau+2025 model: ZuMandelbaum15 occupation (from the wp+ngal MAP
                # fit) × a free duty cycle, with the W_AGN(z) X-ray kernel
                # (Eq. A9).  The duty cycle is threaded per-call via
                # agn_kwargs={"log10DC": ...}, so the model holds only a default.
                from hod_mod.agn.duty_cycle import DutyCycleAGNModel
                print(f"  [{label}] building DutyCycleAGNModel "
                      f"(z_mean={s['zmean']:.3f}) ...", flush=True)
                self._agn_by_sample[label] = DutyCycleAGNModel(
                    sample=label, theta_cosmo=_THETA_COSMO, hmf=self.hmf,
                    log10DC=getattr(self, "dc_log10DC", -2.0),
                )
            else:
                print(f"  [{label}] building HODAgnModel (f_inc={self.agn_finc:g}, "
                      f"z_mean={s['zmean']:.3f}) ...", flush=True)
                self._agn_by_sample[label] = HODAgnModel(
                    pk_lin=self.pk_lin,
                    theta_cosmo=_THETA_COSMO,
                    hmf=self.hmf,                 # reuse the fit's (CSST) HMF for consistency
                    z_mean=s["zmean"],
                    z_max=s["zmax"],
                    hod_params={"f_inc": self.agn_finc},
                    xlf="aird15",
                )
        self.cross._agn = self._agn_by_sample[label]
        self.cross._agn_has_hod = True

    def _gas_variants(self, gas_build: dict):
        """Build the full DPM stack (density, pressure, metallicity, APEC cooling)
        with the requested ``gas_build`` overrides applied.

        Activating the full APEC emissivity path (vs the default density-only
        ``emissivity_uk``) is what makes the pressure / temperature / metallicity
        parameters meaningful; the overall normalisation difference is absorbed by
        the free ``log10_A_gas`` amplitude (the cross-spectra divides the full-APEC
        emissivity by Λ_ref to stay on the n_e²-scale and avoid float32 underflow —
        see ``HaloModelCrossSpectra._pk_tables_gX``).  The cooling table (~10 s) is
        built once and cached on the infrastructure.
        """
        from hod_mod.scripts.validate_gas_profiles import (
            _make_density_variant, _make_pressure_variant,
        )
        from hod_mod.gas import MetallicityProfileDPM, _gnfw_f_params
        from hod_mod.gas import ApecCoolingTable

        if self._cooling_fn is None:
            print("  building ApecCoolingTable (full-DPM emissivity) ...", flush=True)
            self._cooling_fn = ApecCoolingTable(emin=0.5, emax=2.0)

        dp = _make_density_variant(model=2, **gas_build.get("density", {}))
        pp = _make_pressure_variant(model=2, **gas_build.get("pressure", {}))
        mp = MetallicityProfileDPM()
        for k, v in gas_build.get("metal", {}).items():
            setattr(mp, f"_{k}", float(v))             # e.g. _Z_03
        x_ref = 0.3 * mp._C_DPM
        mp._Z0 = mp._Z_03 / float(_gnfw_f_params(x_ref, mp._ALPHA_IN,
                                                  mp._ALPHA_TR, mp._ALPHA_OUT))
        return dp, pp, mp, self._cooling_fn

    def use_agn_override(self, label: str, agn_build: dict) -> None:
        """Attach a per-call HODAgnModel rebuilt with occupation overrides.

        ``agn_build`` keys (log10mmin, sigma_logm, alpha, f_inc) override the
        More+2015 AGN-HOD; this triggers the ~10 s abundance-match precompute, so
        it is only used by the ``agn-occ`` preset.  Restore the base model with
        ``use_agn_for(label)`` afterwards.
        """
        s = SAMPLES[label]
        hodp = {"f_inc": self.agn_finc}
        hodp.update(agn_build)
        self.cross._agn = HODAgnModel(
            pk_lin=self.pk_lin, theta_cosmo=_THETA_COSMO, hmf=self.hmf,
            z_mean=s["zmean"], z_max=s["zmax"], hod_params=hodp, xlf="aird15",
        )
        self.cross._agn_has_hod = True


def _build_shared_components() -> dict:
    """Build the HMF-independent halo-model pieces (P_lin, halo profile, gas
    profile, HAM AGN model) once, so they can be reused across many HMF
    backends without repeating the ~12s AGN abundance-matching precompute.
    """
    print("Building shared halo-model components (CAMB + HAM) ...", flush=True)
    t0 = time.time()
    pk_lin = LinearPowerSpectrum()
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    dp     = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=200)
    bnl    = _make_bnl()
    agn    = HamAGNModel(pk_lin=pk_lin)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    return dict(pk_lin=pk_lin, hp=hp, dp=dp, agn=agn, bnl=bnl)


def _build_infra_for_hmf(hmf_backend: str, shared: dict, **hmf_kwargs) -> _Infrastructure:
    """Build a halo-model stack for one HMF backend, reusing ``shared`` components."""
    pk_lin, hp, dp, agn = shared["pk_lin"], shared["hp"], shared["dp"], shared["agn"]
    bnl  = shared.get("bnl")
    hmf  = make_hmf(hmf_backend, pk_func=pk_lin.pk_linear, **hmf_kwargs)
    hod  = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
    cross = HaloModelCrossSpectra(fhmp, density_profile=dp, agn_model=agn)

    infra = _Infrastructure.__new__(_Infrastructure)
    infra.pk_lin, infra.hp, infra.dp, infra.agn = pk_lin, hp, dp, agn
    infra.fhmp, infra.cross = fhmp, cross
    infra.hmf, infra.bnl = hmf, bnl
    # Multi-HMF comparison path uses the shared HamAGNModel (legacy PSF behaviour).
    infra.agn_model_choice = "ham"
    infra.agn_finc = 0.01
    infra._agn_by_sample = {}
    return infra


# --------------------------------------------------------------------------
# Model prediction + disk shape cache
# --------------------------------------------------------------------------

_N_ELL           = 160
_ELL             = np.logspace(1.0, 5.0, _N_ELL)   # ell_max = 100,000
_PSF_KING_THETA_C = 8.64   # arcsec, eROSITA TM CalDB on-axis fit
_SHAPE_CACHE_DIR  = _RESULTS_DIR / "shape_cache"

# GAS.py-calibrated starting values (from validate_gas_profiles.py).
# β_n=0.20 + β_P=0.80 → α_Lx=1.70, α_kT=0.60 matching Comparat+2025 scaling relations.
# DPM model-2 defaults: β_n=0.36, β_P=0.85.
_BETA_GAS_DEFAULT      = 0.25   # MAP run6 β_n (was 0.20)
_BETA_PRESSURE_DEFAULT = 0.86   # MAP run6 β_P (was 0.80)

# eROSITA CalDB PSF option (False → analytic King, True → TM1-7 mean from FITS)
_USE_CALDB_PSF = False
_CALDB_DIR = Path(
    "/home/comparat/data/erosita/caldb_221121v03/caldb/srv-0500-2000"
)


def _shape_cache_key(
    label: str,
    hod_params: dict,
    beta_gas: float,
    beta_pressure: float = _BETA_PRESSURE_DEFAULT,
    agn_model: str = "hod",
    agn_finc: float = 0.01,
) -> str:
    """Hash key encoding sample, HOD parameters, beta_gas/beta_pressure and the
    AGN configuration (model + duty cycle) for disk caching."""
    hp_str = json.dumps(
        {k: round(float(v), 6) for k, v in sorted(hod_params.items())},
        sort_keys=True,
    )
    psf_tag = "psf_caldb" if _USE_CALDB_PSF else "psf_pt"
    if agn_model == "hod":
        agn_tag = f"hod_finc{agn_finc:.4f}"
    else:
        agn_tag = f"ham_{psf_tag}"
    raw = f"{label}|{hp_str}|beta{beta_gas:.4f}|betaP{beta_pressure:.4f}|{agn_tag}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _hankel(cl_arr: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    """C_ℓ → w_θ(θ) via Hankel sum: w_θ(θ) = (1/2π) ∫ ℓ C_ℓ J₀(ℓθ) dℓ."""
    return np.array([
        trapezoid(_ELL * cl_arr * j0(_ELL * th) / (2.0 * np.pi), _ELL)
        for th in theta_rad
    ])


def _psf_caldb(theta_arcsec: np.ndarray) -> np.ndarray:
    """Mean TM1-7 eROSITA PSF, azimuthally averaged, interpolated at theta_arcsec."""
    from scipy.interpolate import interp1d

    profiles = []
    for tm in range(1, 8):
        fpath = _CALDB_DIR / f"tm{tm}_2dpsf_221121v03.fits"
        if not fpath.exists():
            continue
        with fits.open(fpath) as hdul:
            data = hdul[1].data.astype(float)   # (480, 480), 1 arcsec/pix
        cx = cy = 240.0
        ny, nx = data.shape
        y_arr, x_arr = np.mgrid[0:ny, 0:nx]
        r = np.sqrt((x_arr - cx) ** 2 + (y_arr - cy) ** 2)
        rbins = np.arange(0.5, 201.0, 1.0)
        rmid  = 0.5 * (rbins[:-1] + rbins[1:])
        prof  = np.array([
            data[(r >= r1) & (r < r2)].mean()
            if ((r >= r1) & (r < r2)).any() else 0.0
            for r1, r2 in zip(rbins[:-1], rbins[1:])
        ])
        prof /= prof[0]
        profiles.append(prof)
    if not profiles:
        raise FileNotFoundError(f"No CalDB PSF FITS files found in {_CALDB_DIR}")
    mean_prof = np.mean(profiles, axis=0)
    interp = interp1d(rmid, mean_prof, kind="linear",
                      bounds_error=False, fill_value=0.0)
    return interp(theta_arcsec)


def _psf_template(theta_arcsec: np.ndarray) -> np.ndarray:
    """AGN point-source template: normalized PSF profile at the given angles.

    AGN are unresolved → their angular cross-correlation template is the PSF
    itself.  Returns values in [0, 1] with PSF(0) = 1.
    """
    if _USE_CALDB_PSF:
        return _psf_caldb(theta_arcsec)
    tc, alpha = _PSF_KING_THETA_C, 1.5
    return (1.0 + (theta_arcsec / tc) ** 2) ** (-alpha)


def _predict_shape(
    label: str,
    infra: _Infrastructure,
    hod_params: dict,
    beta_gas: float = _BETA_GAS_DEFAULT,
    beta_pressure: float = _BETA_PRESSURE_DEFAULT,
    gas_build: dict | None = None,
    agn_cheap: dict | None = None,
    agn_build: dict | None = None,
    use_full_nz: bool = False,
    use_disk_cache: bool = True,
) -> dict:
    """Compute unnormalized w_θ shapes {"gas", "agn"} on the data theta grid.

    The model prediction is:  w_model = A_gas * shapes["gas"] + A_AGN * shapes["agn"]

    shapes["gas"] = Hankel( C_ℓ^gas × B_ℓ )  — gas Limber + PSF convolution (halo model)

    The AGN template depends on ``infra.agn_model_choice``:
      - ``"hod"`` (default): shapes["agn"] = Hankel( C_ℓ^{gX,agn} × B_ℓ ) — the
        occupation-weighted AGN cross-power from :class:`HODAgnModel`, PSF-convolved.
        ``log10_A_AGN`` becomes a *fudge factor* on this physically-predicted amplitude.
        Note: this carries a (small) 2-halo term in addition to the point-source 1-halo.
      - ``"ham"`` (legacy): shapes["agn"] = PSF(θ) directly — AGN as a pure free-amplitude
        point source (no halo-model AGN cross-power).
    King PSF (θ_c = 8.64 arcsec) and beta_gas tilt are applied here.
    Results are cached to .npz files keyed by HOD params + beta_gas + AGN config.
    """
    infra.use_agn_for(label)   # attach the per-sample AGN model to infra.cross
    if agn_build:
        infra.use_agn_override(label, agn_build)   # rebuild HODAgnModel occupation

    has_gas_override = bool(gas_build) and any(
        gas_build.get(p) for p in ("density", "pressure", "metal"))
    if has_gas_override or agn_cheap or agn_build:
        # Override presets are exploratory and not keyed in the disk cache.
        use_disk_cache = False

    if use_full_nz:
        z_arr, nz_g = _build_nz_full(label)
        use_disk_cache = False
    else:
        z_arr, nz_g = _build_nz_fast(label)

    _SHAPE_KEYS = {"gas", "agn", "gas_1h_cen", "gas_1h_sat", "gas_2h"}

    if use_disk_cache:
        key  = _shape_cache_key(label, hod_params, beta_gas, beta_pressure,
                                infra.agn_model_choice, infra.agn_finc)
        path = _SHAPE_CACHE_DIR / f"{label}_{key}.npz"
        if path.exists():
            d = np.load(path)
            if _SHAPE_KEYS <= set(d.files):
                cached = {k: d[k] for k in _SHAPE_KEYS}
                if all(np.isfinite(_arr).all() for _arr in cached.values()):
                    print(f"  [{label}] shape loaded from disk cache ({path.name})", flush=True)
                    return cached
            # Cache missing sub-components or has non-finite values — recompute
            print(f"  [{label}] stale/bad cache — deleting and recomputing", flush=True)
            path.unlink(missing_ok=True)

    # Swap in DPM profile variants (activating the full APEC emissivity path) for
    # the duration of the angular_cl_gX call, then restore the base profiles.
    _saved_gas = None
    if has_gas_override:
        _saved_gas = (infra.cross._dp, infra.cross._pp,
                      infra.cross._mp, infra.cross._cooling_fn)
        (infra.cross._dp, infra.cross._pp,
         infra.cross._mp, infra.cross._cooling_fn) = infra._gas_variants(gas_build)

    print(f"  [{label}] computing shape (n_z={len(z_arr)}, n_ell={_N_ELL}, "
          f"beta_gas={beta_gas:.3f}, beta_pressure={beta_pressure:.3f}"
          f"{', gas_build' if has_gas_override else ''}"
          f"{', agn_cheap' if agn_cheap else ''}"
          f"{', agn_build' if agn_build else ''}) ...", flush=True)
    t0 = time.time()
    try:
        cl_components = infra.cross.angular_cl_gX(
            _ELL, z_arr, nz_g, _THETA_COSMO, hod_params,
            psf_king_theta_c_arcsec=_PSF_KING_THETA_C,
            beta_gas=beta_gas,
            beta_pressure=beta_pressure,
            return_components=True,
            n_workers=1,   # serial: XLA memory allocator is not thread-safe
            agn_kwargs=(agn_cheap or None),
        )
    finally:
        if _saved_gas is not None:
            (infra.cross._dp, infra.cross._pp,
             infra.cross._mp, infra.cross._cooling_fn) = _saved_gas
        if agn_build:
            infra.use_agn_for(label)   # restore the base AGN model
    print(f"  [{label}] angular_cl_gX: {time.time()-t0:.1f}s", flush=True)

    # The full-APEC emissivity has a tiny k ≳ 150 h/Mpc tail that can still go
    # non-finite in float32 through the Limber/PSF stage (far beyond any fitted
    # angular scale).  Zero those C_ℓ bins so a single high-ℓ NaN does not poison
    # the Hankel transform of every θ.  (No-op for the density-only path.)
    n_bad_cl = 0
    for _ck, _cv in list(cl_components.items()):
        _a   = np.asarray(_cv, dtype=float)
        _bad = ~np.isfinite(_a)
        if _bad.any():
            n_bad_cl += int(_bad.sum())
            cl_components[_ck] = np.where(_bad, 0.0, _a)
    if n_bad_cl:
        print(f"  [{label}] sanitised {n_bad_cl} non-finite high-k C_ell bins",
              flush=True)

    data_d = load_data(label)
    theta  = data_d["theta_rad"]
    if infra.agn_model_choice in ("hod", "duty_cycle"):
        # Physically-predicted, occupation-weighted AGN cross-power (PSF-convolved
        # inside angular_cl_gX).  For "hod", log10_A_AGN is a fudge factor on this
        # amplitude; for "duty_cycle" the amplitude is the duty cycle, threaded via
        # agn_kwargs={"log10DC": ...} into the AGN emissivity (so the template
        # returned here is already DC-scaled).
        agn_shape = _hankel(np.asarray(cl_components["agn"], dtype=float), theta)
    else:
        # Legacy: AGN as a pure free-amplitude King PSF point source.
        agn_shape = _psf_template(data_d["theta_arcsec"])
    shapes = {
        "gas":        _hankel(np.asarray(cl_components["gas"],        dtype=float), theta),
        "gas_1h_cen": _hankel(np.asarray(cl_components["gas_1h_cen"], dtype=float), theta),
        "gas_1h_sat": _hankel(np.asarray(cl_components["gas_1h_sat"], dtype=float), theta),
        "gas_2h":     _hankel(np.asarray(cl_components["gas_2h"],     dtype=float), theta),
        "agn": agn_shape,
    }

    for _k, _arr in shapes.items():
        if not np.isfinite(_arr).all():
            n_bad = int(np.sum(~np.isfinite(_arr)))
            raise RuntimeError(
                f"[{label}] shapes['{_k}'] has {n_bad} non-finite values "
                f"(likely n_gal→0 at current HOD params). Shape not cached."
            )

    if use_disk_cache:
        _SHAPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, **shapes)   # key was computed above
        print(f"  [{label}] shape cached → {path.name}", flush=True)

    return shapes


_nc_thresh_vmap = jax.vmap(
    n_cen_thresh_zu15,
    in_axes=(None, 0, None, None, None, None, None, None, None, None),
)
_ns_thresh_vmap = jax.vmap(
    n_sat_thresh_zu15,
    in_axes=(None, 0, None, None, None, None, None, None, None, None,
              None, None, None, None, None),
)


def _predict_smf(
    infra: _Infrastructure,
    z: float,
    hod_params: dict,
    log10mstar_grid: np.ndarray,
) -> np.ndarray:
    """Model SMF Phi(M*) = -dN(>M*)/dlog10(M*) [Mpc^-3 dex^-1].

    Vectorized threshold scan: evaluates ``n_cen_thresh_zu15``/``n_sat_thresh_zu15``
    for the *whole* ``log10mstar_grid`` at once via ``jax.vmap`` (one batched
    XLA call) rather than looping in Python over ``infra.fhmp.n_gal()`` calls.
    This matters because, unlike the old diagnostic-only usage, this function
    is now called on every joint-likelihood evaluation (tens of thousands of
    times during MCMC); the per-call ``jax.disable_jit()`` overhead of
    ``n_gal()`` made the naive loop prohibitively slow at that call rate.

    ``log10mstar_grid`` (from sum_stat) is in the h-free convention
    log10(M*/Msun), so it must be shifted by +log10(h) before being used as
    ``log10m_star_thresh``.

    Units: the sum_stat SMF data ``phi`` is in **h^3 Mpc^-3 dex^-1**
    (= (Mpc/h)^-3 dex^-1; see ``SumStatReader.smf``), which is the *same*
    numeric convention as the halo model's native comoving density
    ``dn/dM`` [(Mpc/h)^-3 / (Msun/h)].  The cumulative density is therefore
    returned directly in (Mpc/h)^-3 with **no** h^3 conversion — multiplying by
    h^3 (as a previous version did) under-predicted the SMF by h^3 ~ 0.31.
    """
    h = float(_THETA_COSMO["h"])
    # In fixed-ZM15 mode the ZM15 params come from fit_bgs_zm15_joint, which feeds
    # the *physical* log10(M*/Msun) threshold straight to predictor.n_gal (no
    # h-shift); FullHaloModelPrediction.n_gal/wp/delta_sigma do the same.  Apply
    # the +log10(h) shift only in the legacy 9-param path (whose free threshold
    # absorbed it) so the SMF/n_gal diagnostic matches the clustering panels.
    log10h = 0.0 if _FIT_CFG is not None else np.log10(h)
    thresh_grid = jnp.asarray(log10mstar_grid, dtype=float) + log10h

    hod = infra.fhmp._hod
    cosmo_key = infra.fhmp._cosmo_cache_key(z, _THETA_COSMO)
    if cosmo_key not in infra.fhmp._static_cache:
        infra.fhmp._pk_tables_full(z, _THETA_COSMO, hod_params)
    sc = infra.fhmp._static_cache[cosmo_key]
    dndm_np = sc["dndm_np"]
    m_np    = sc["m_np"]
    log10m_grid = hod._log10m_grid

    p = hod_params
    nc = _nc_thresh_vmap(
        log10m_grid, thresh_grid,
        p["lg_m1h"], p["lg_m0star"], p["beta"], p["delta"], p["gamma"],
        p["sigma_lnmstar"], p["eta"], p["fc"],
    )
    ns = _ns_thresh_vmap(
        log10m_grid, thresh_grid,
        p["lg_m1h"], p["lg_m0star"], p["beta"], p["delta"], p["gamma"],
        p["sigma_lnmstar"], p["eta"], p["fc"],
        p["bsat"], p["beta_sat"], p["bcut"], p["beta_cut"], p["alpha_sat"],
    )
    n_tot = np.asarray(nc + ns)   # (n_thresh, n_mass)
    # n_cum in (Mpc/h)^-3 = h^3 Mpc^-3, matching the sum_stat SMF data units.
    n_cum = np.trapezoid(dndm_np[None, :] * n_tot, m_np, axis=1)
    return -np.gradient(n_cum, log10mstar_grid)


# --------------------------------------------------------------------------
# Likelihood and priors
# --------------------------------------------------------------------------

def _angular_mask(theta_arcsec: np.ndarray,
                  theta_min: float, theta_max: float) -> np.ndarray:
    return (theta_arcsec >= theta_min) & (theta_arcsec <= theta_max)


def log_likelihood(
    params: np.ndarray,
    label: str,
    infra: _Infrastructure,
    data_wt: dict,
    theta_mask: np.ndarray,
    data_wp: dict | None,
    shape_cache: dict,
    f_sys: float = 0.05,
) -> float:
    """Joint Gaussian log-likelihood over w_θ(θ), wp(rp), and the SMF Phi(M*).

    params = [log10_A_gas, beta_gas, beta_pressure, log10_A_AGN,
              log10m_star_thresh, sigma_lnmstar, lg_m1h, alpha_sat, fc]

    shape_cache : mutable dict keyed by (log10m_star_thresh, sigma_lnmstar, lg_m1h, alpha_sat,
                                          fc, beta_gas, beta_pressure).
    data_wp     : output of load_smf_wp_data(), or None to skip the wp+SMF term.
    """
    dec = None
    if _FIT_CFG is not None:
        # Fixed-ZM15 mode: only the X-ray/gas/AGN parameters are free; ZM15 is
        # held at the loaded MAP values.  data_wp is None here (run_map/run_mcmc
        # do not load it), so only w_θ enters the likelihood.
        dec = _decode_params(params)
        log10_A_gas, log10_A_AGN = dec["log10_A_gas"], dec["log10_A_AGN"]
        beta_gas, beta_pressure  = dec["beta_gas"], dec["beta_pressure"]
        hp = _hod_params(label)
    else:
        log10_A_gas, beta_gas, beta_pressure, log10_A_AGN = (
            params[0], params[1], params[2], params[3]
        )
        log10m_star_thresh, sigma_lnmstar, lg_m1h, alpha_sat, fc = (
            params[4], params[5], params[6], params[7], params[8]
        )
        hp = _hod_params(label,
                         log10m_star_thresh=log10m_star_thresh, sigma_lnmstar=sigma_lnmstar,
                         lg_m1h=lg_m1h, alpha_sat=alpha_sat, fc=fc)

    z_eff = SAMPLES[label]["zmean"]

    # Cache key derived from hp (the 5 ZM15 shape params) + betas — identical to
    # the legacy tuple in 9-param mode; in fixed mode it additionally carries the
    # gas-profile / AGN overrides so each distinct config gets its own shape.
    cache_key = (
        round(hp["log10m_star_thresh"], 4), round(hp["sigma_lnmstar"], 4),
        round(hp["lg_m1h"],             4), round(hp["alpha_sat"],     4),
        round(hp["fc"],                 4),
        round(beta_gas,                 4), round(beta_pressure,       4),
        _dec_shape_key(dec) if dec is not None else None,
    )
    if cache_key not in shape_cache:
        shape_cache[cache_key] = _predict_shape(
            label, infra, hp, beta_gas=beta_gas, beta_pressure=beta_pressure,
            gas_build=None if dec is None else dec["gas_build"],
            agn_cheap=None if dec is None else dec["agn_cheap"],
            agn_build=None if dec is None else dec["agn_build"],
        )

    shapes = shape_cache[cache_key]
    A_gas  = 10.0 ** log10_A_gas
    A_AGN  = 10.0 ** log10_A_AGN
    wm_all = A_gas * shapes["gas"] + A_AGN * shapes["agn"]

    # --- w_θ likelihood (diagonal + systematic floor) ---
    wd      = data_wt["wtheta"][theta_mask]
    err_jk  = data_wt["wtheta_err"][theta_mask]
    err_sys = f_sys * np.abs(wd)
    err     = np.sqrt(err_jk**2 + err_sys**2)
    err     = np.where(err > 1e-12 * np.abs(wd).max(), err, 1e-12 * np.abs(wd).max())
    resid_wt = wd - wm_all[theta_mask]
    ll_wt    = -0.5 * float(np.sum((resid_wt / err) ** 2))

    # --- number-density + wp likelihood (DIAGONAL covariance) ---
    # The binned SMF Phi(M*) is NOT fit directly: its jackknife covariance is
    # near-singular (cond ~1e16) and the iHOD over-predicts the high-mass tail
    # by >10x, which railed the galaxy params.  Instead we constrain the overall
    # number density n_gal = integral of the SMF, plus wp(rp) with diagonal
    # (systematic-floored) errors.  See _predict_smf for the h^3 unit fix.
    ll_smfwp = 0.0
    if data_wp is not None:
        x           = data_wp["log10mstar"]
        n_gal_data  = float(np.trapezoid(data_wp["phi"], x))               # (Mpc/h)^-3
        n_gal_model = float(np.trapezoid(_predict_smf(infra, z_eff, hp, x), x))
        sig_ngal    = max(f_sys, 1e-3) * abs(n_gal_data)   # jackknife n_gal error ~0 → use floor
        ll_ngal     = -0.5 * ((n_gal_data - n_gal_model) / sig_ngal) ** 2

        wp_model = np.asarray(
            infra.fhmp.wp(data_wp["rp"], pi_max=data_wp["pi_max"],
                          z=z_eff, theta_cosmo=_THETA_COSMO, hod_params=hp),
            dtype=float,
        )
        ns      = data_wp["n_smf"]
        wp_var  = np.diag(data_wp["cov"])[ns:]             # floored diagonal (see _apply_smfwp_syst)
        wp_var  = np.where(wp_var > 0, wp_var, (f_sys * np.abs(data_wp["wp"])) ** 2 + 1e-30)
        ll_wp   = -0.5 * float(np.sum((data_wp["wp"] - wp_model) ** 2 / wp_var))
        ll_smfwp = ll_ngal + ll_wp

    return ll_wt + ll_smfwp


def log_prior(params: np.ndarray) -> float:
    """Flat box priors.

    Legacy mode: all 9 parameters.  Fixed-ZM15 mode: only the free subset
    (bounds from ``_PARAM_REGISTRY``).
    """
    if _FIT_CFG is not None:
        for name, val in zip(_FIT_CFG["free"], params):
            lo, hi = _PARAM_REGISTRY[name][0], _PARAM_REGISTRY[name][1]
            if not (lo <= val <= hi):
                return -np.inf
        return 0.0

    log10_A_gas, beta_gas, beta_pressure, log10_A_AGN = (
        params[0], params[1], params[2], params[3]
    )
    log10m_star_thresh, sigma_lnmstar, lg_m1h, alpha_sat, fc = (
        params[4], params[5], params[6], params[7], params[8]
    )
    if not (-2.0  <= log10_A_gas        <= 12.0):  return -np.inf
    if not ( 0.0  <= beta_gas           <=  0.8):  return -np.inf
    if not ( 0.0  <= beta_pressure      <=  2.0):  return -np.inf
    if not (-5.0  <= log10_A_AGN        <= 15.0):  return -np.inf
    if not ( 9.0  <= log10m_star_thresh <= 12.0):  return -np.inf
    if not (0.01  <= sigma_lnmstar      <=  1.5):  return -np.inf
    if not ( 9.5  <= lg_m1h             <= 14.0):  return -np.inf
    if not ( 0.5  <= alpha_sat          <=  2.5):  return -np.inf
    if not (0.05  <= fc                 <=  1.0):  return -np.inf
    return 0.0


def log_prob(
    params: np.ndarray,
    label: str,
    infra: _Infrastructure,
    data_wt: dict,
    theta_mask: np.ndarray,
    data_wp: dict | None,
    shape_cache: dict,
    f_sys: float = 0.05,
) -> float:
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, label, infra,
                               data_wt, theta_mask, data_wp,
                               shape_cache, f_sys=f_sys)


# --------------------------------------------------------------------------
# MAP fit
# --------------------------------------------------------------------------

_PARAM_NAMES = [
    "log10_A_gas", "beta_gas", "beta_pressure", "log10_A_AGN",
    "log10m_star_thresh", "sigma_lnmstar", "lg_m1h", "alpha_sat", "fc",
]
_PARAM_BOUNDS = [
    (-2.0, 12.0),   # log10_A_gas
    ( 0.0,  0.8),   # beta_gas      (GAS.py target 0.20; physical range 0–0.8)
    ( 0.0,  2.0),   # beta_pressure (GAS.py target 0.80)
    (-5.0, 15.0),   # log10_A_AGN
    ( 9.0, 12.0),   # log10m_star_thresh  (stellar mass threshold ≈ sample log10ms_min ± 0.5)
    (0.01,  1.5),   # sigma_lnmstar       (ZM15 SHMR scatter)
    ( 9.5, 14.0),   # lg_m1h              (ZM15 SHMR characteristic halo mass)
    ( 0.5,  2.5),   # alpha_sat
    (0.05,  1.0),   # fc — central completeness fraction (ZM15 default 0.86);
                    # freed to fit the SMF normalisation jointly with wp/X-ray.
]

# --------------------------------------------------------------------------
# Fixed-ZM15 mode (--fix-zm15)
# --------------------------------------------------------------------------
# When enabled, the ZuMandelbaum15 stellar-halo connection is held fixed at the
# parameters of the dedicated wp+n_gal joint fit (results/bgs_zm15_joint_wp_ngal/
# map_result.json, 13 SHMR+satellite params) and only the X-ray/gas/AGN
# parameters are fit against the galaxy x X-ray cross-correlation w_θ(θ).  The
# legacy 9-parameter joint fit is unaffected when --fix-zm15 is absent
# (``_FIT_CFG is None``).
# --- Generalised free-parameter registry -------------------------------------
# name -> (lo, hi, default, kind).  ``kind`` says how the parameter enters the
# prediction:
#   "amp"                    log10 amplitude on the gas/AGN template (cheap)
#   "gas_tilt"               per-call mass-slope tilt in angular_cl_gX (cheap)
#   "gas_build:<prof>:<at>"  DPM profile attribute (density/pressure/metal);
#                            rebuilds the profile per likelihood call (expensive)
#   "agn_cheap:<kw>"         per-call kwarg to agn_emissivity_uk (ham; cheap)
#   "agn_build:<hodkey>"     AGN-HOD occupation param (hod); rebuilds the AGN
#                            abundance-match per call (expensive)
# log10-encoded profile normalisations (log10_ne_03, log10_P_03) are exponentiated
# in _decode_params before being handed to the DPM profile builders.
_LOG10_P03_DEF  = float(np.log10(115.0e-6))   # DPM model-2 pressure norm  [keV cm^-3]
_LOG10_NE03_DEF = float(np.log10(4.87e-5))    # DPM model-2 density norm   [cm^-3]

_PARAM_REGISTRY: dict[str, tuple] = {
    # amplitudes + cheap mass-slope tilts
    "log10_A_gas":        (-2.0, 12.0, 0.0,            "amp"),
    "beta_gas":           ( 0.0,  0.8, _BETA_GAS_DEFAULT,      "gas_tilt"),
    "beta_pressure":      ( 0.0,  2.0, _BETA_PRESSURE_DEFAULT, "gas_tilt"),
    "log10_A_AGN":        (-5.0, 15.0, 0.0,            "agn_amp"),
    # gas density profile (rebuild)
    "alpha_out_gas":      ( 1.5,  4.0, 2.7,            "gas_build:density:alpha_out"),
    "alpha_in_gas":       ( 0.0,  2.0, 1.0,            "gas_build:density:alpha_in"),
    "alpha_tr_gas":       ( 0.5,  3.0, 1.9,            "gas_build:density:alpha_tr"),
    "gamma_gas":          ( 0.0,  4.0, 2.0,            "gas_build:density:gamma"),
    "log10_ne_03":        (-6.0, -3.0, _LOG10_NE03_DEF, "gas_build:density:ne_03"),
    # gas pressure profile (rebuild)
    "alpha_out_pressure": ( 2.0,  6.0, 4.1,            "gas_build:pressure:alpha_out_12"),
    "alpha_in_pressure":  (-0.5,  1.0, 0.3,            "gas_build:pressure:alpha_in"),
    "alpha_tr_pressure":  ( 0.5,  3.0, 1.3,            "gas_build:pressure:alpha_tr"),
    "log10_P_03":         (-5.0, -2.0, _LOG10_P03_DEF, "gas_build:pressure:P_03"),
    # metallicity (rebuild)
    "Z_0":                (0.05,  1.0, 0.3,            "gas_build:metal:Z_03"),
    # AGN cheap luminosity overrides (HamAGNModel; per-call)
    "scatter_lx":         ( 0.3,  1.3, 0.8,            "agn_cheap:scatter_lx"),
    "log10_A_kcorr":      (-0.5,  0.0, 0.0,            "agn_cheap:log10_A_kcorr"),
    "log10_A_dc":         (-1.0,  0.0, 0.0,            "agn_cheap:log10_A_dc"),
    # AGN occupation (HODAgnModel; rebuild)
    "f_inc":              (1e-3,  0.5, 0.01,           "agn_build:f_inc"),
    "log10mmin_agn":      (11.0, 14.0, 12.5,           "agn_build:log10mmin"),
    "sigma_logm_agn":     ( 0.1,  1.5, 0.8,            "agn_build:sigma_logm"),
    "alpha_agn":          ( 0.2,  1.5, 0.8,            "agn_build:alpha"),
}

# log10-encoded profile normalisations exponentiated before the DPM builders.
_LOG_GAS_PARAMS = {"log10_ne_03", "log10_P_03"}

# The four "cheap" astro params seeded into _FIT_CFG["fixed_astro"] by run_map.
_ASTRO_PARAM_NAMES = ["log10_A_gas", "beta_gas", "beta_pressure", "log10_A_AGN"]

# Selectable free-parameter presets for --free-params (or pass an explicit list
# of registry parameter names).
_FREE_PRESETS = {
    # original tiers
    "all":  ["log10_A_gas", "beta_gas", "beta_pressure", "log10_A_AGN"],
    "amps": ["log10_A_gas", "log10_A_AGN"],
    "gas":  ["log10_A_gas", "beta_gas", "beta_pressure"],
    # gas tiers (AGN = amplitude only) — progressively richer DPM gas profile
    "gas-shape": ["log10_A_gas", "beta_gas", "beta_pressure",
                  "alpha_out_gas", "alpha_out_pressure", "log10_A_AGN"],
    "gas-temp":  ["log10_A_gas", "beta_gas", "beta_pressure",
                  "alpha_out_gas", "alpha_out_pressure",
                  "log10_P_03", "gamma_gas", "log10_A_AGN"],
    "gas-full":  ["log10_A_gas", "beta_gas", "beta_pressure",
                  "alpha_out_gas", "alpha_out_pressure",
                  "log10_P_03", "gamma_gas", "log10_ne_03",
                  "alpha_in_gas", "alpha_tr_gas",
                  "alpha_in_pressure", "alpha_tr_pressure", "Z_0", "log10_A_AGN"],
    # AGN tiers (gas = the 4 'all' params) — run under the matching --agn-model
    "agn-models": ["log10_A_gas", "beta_gas", "beta_pressure", "log10_A_AGN"],
    "agn-lum":    ["log10_A_gas", "beta_gas", "beta_pressure", "log10_A_AGN",
                   "scatter_lx", "log10_A_kcorr", "log10_A_dc"],
    "agn-occ":    ["log10_A_gas", "beta_gas", "beta_pressure", "log10_A_AGN",
                   "f_inc", "log10mmin_agn", "sigma_logm_agn", "alpha_agn"],
}

# Presets that require a specific --agn-model (validated in main()).
_PRESET_REQUIRES_AGN = {"agn-lum": "ham", "agn-occ": "hod"}

# Default ZM15 MAP json: results/bgs_zm15_joint_wp_ngal/map_result.json
_DEFAULT_ZM15_JSON = _RESULTS_DIR.parents[1] / "bgs_zm15_joint_wp_ngal" / "map_result.json"

# Populated by main() when --fix-zm15 is given; None ⇒ legacy 9-parameter fit.
#   {"zm15": {...13 ZM15 params...}, "free": [names...],
#    "fixed_astro": {name: value, ...}, "thresh": {label: log10m_star_thresh}}
_FIT_CFG: dict | None = None


def _decode_params(vec) -> dict:
    """Decode the free-parameter vector into a structured prediction config.

    Returns a dict::

        {"log10_A_gas", "log10_A_AGN", "beta_gas", "beta_pressure",
         "gas_build": {"density": {...}, "pressure": {...}, "metal": {...}},
         "agn_cheap": {...}, "agn_build": {...}}

    Free entries (order ``_FIT_CFG["free"]``) come from *vec*; non-free entries
    fall back to ``_FIT_CFG["fixed_astro"]`` then the registry default.  Only
    parameters that are actually **free** populate the gas_build/agn_cheap/
    agn_build sub-dicts, so amplitude-only presets never trigger a profile/AGN
    rebuild.
    """
    free  = _FIT_CFG["free"]
    fixed = _FIT_CFG.get("fixed_astro", {})
    vmap  = dict(zip(free, (float(v) for v in vec)))

    def get(name):
        if name in vmap:
            return vmap[name]
        if name in fixed:
            return float(fixed[name])
        return _PARAM_REGISTRY[name][2]

    cfg = {
        "log10_A_gas":   get("log10_A_gas"),
        "log10_A_AGN":   get("log10_A_AGN"),
        "beta_gas":      get("beta_gas"),
        "beta_pressure": get("beta_pressure"),
        "gas_build": {"density": {}, "pressure": {}, "metal": {}},
        "agn_cheap": {},
        "agn_build": {},
    }
    for name in free:
        kind = _PARAM_REGISTRY[name][3]
        val  = vmap[name]
        if kind.startswith("gas_build:"):
            _, prof, attr = kind.split(":")
            cfg["gas_build"][prof][attr] = 10.0 ** val if name in _LOG_GAS_PARAMS else val
        elif kind.startswith("agn_cheap:"):
            cfg["agn_cheap"][kind.split(":")[1]] = val
        elif kind.startswith("agn_build:"):
            cfg["agn_build"][kind.split(":")[1]] = val
    return cfg


def _dec_shape_key(dec: dict) -> tuple:
    """Hashable, rounded key for the gas-profile / AGN overrides of a decoded set.

    Used to extend the in-memory shape-cache key so each distinct gas_build /
    agn_cheap / agn_build configuration gets its own cached w_θ shape.
    """
    def _flat(d):
        return tuple(sorted((k, round(float(v), 6)) for k, v in d.items()))
    gb = dec["gas_build"]
    return (
        _flat(gb["density"]), _flat(gb["pressure"]), _flat(gb["metal"]),
        _flat(dec["agn_cheap"]), _flat(dec["agn_build"]),
    )


def _resolve_free_params(tokens: list[str]) -> list[str]:
    """Resolve a --free-params spec (preset name or explicit list) to param names.

    Returns the names in registry order.
    """
    if len(tokens) == 1 and tokens[0] in _FREE_PRESETS:
        names = list(_FREE_PRESETS[tokens[0]])
    else:
        bad = [t for t in tokens if t not in _PARAM_REGISTRY]
        if bad:
            raise ValueError(
                f"Unknown --free-params {bad}; choose a preset {list(_FREE_PRESETS)} "
                f"or an explicit subset of {list(_PARAM_REGISTRY)}."
            )
        names = list(tokens)
    return [n for n in _PARAM_REGISTRY if n in names]


def run_map(
    label: str,
    infra: _Infrastructure,
    theta_min: float = 8.0,
    theta_max: float = 300.0,
    rp_min: float = 0.02,
    f_sys: float = 0.05,
) -> tuple[dict, dict]:
    """Find MAP estimate for the 9-parameter joint fit (w_θ + wp + SMF).

    Returns (result_dict, shape_cache).
    """
    data_wt   = load_data(label)
    t_mask    = _angular_mask(data_wt["theta_arcsec"], theta_min, theta_max)
    n_pts_wt  = int(t_mask.sum())

    # smf+wp data (None if sample not in _SUM_STAT_DIRS).  In fixed-ZM15 mode the
    # galaxy side is held fixed, so wp+SMF do not enter the likelihood (w_θ only);
    # they are loaded only as diagnostic overlays by the plot functions.
    data_wp = None
    n_pts_smf = n_pts_wp = 0
    if _FIT_CFG is None and label in _SUM_STAT_DIRS:
        try:
            data_wp   = _apply_smfwp_syst(load_smf_wp_data(label, rp_min=rp_min), f_sys)
            n_pts_smf = 1            # SMF enters as a single integrated number-density point
            n_pts_wp  = data_wp["n_wp"]
        except FileNotFoundError as exc:
            print(f"  [{label}] WARNING: smf/wp data not found ({exc}); fitting w_θ only.", flush=True)

    n_free = len(_FIT_CFG["free"]) if _FIT_CFG is not None else len(_PARAM_NAMES)
    n_pts  = n_pts_wt + n_pts_smf + n_pts_wp
    ndof   = max(n_pts - n_free, 1)
    shape_cache: dict = {}

    # Initial HOD from Table 3
    ms0, sig0, m1h0, alpha0, fc0 = _TABLE3[label]

    # Initial gas amplitude: linear solve on w_θ with nominal shape
    hp0     = _hod_params(label)
    shapes0 = _predict_shape(
        label, infra, hp0,
        beta_gas=_BETA_GAS_DEFAULT, beta_pressure=_BETA_PRESSURE_DEFAULT,
    )
    ck0 = (
        round(hp0["log10m_star_thresh"], 4), round(hp0["sigma_lnmstar"], 4),
        round(hp0["lg_m1h"],             4), round(hp0["alpha_sat"],     4),
        round(hp0["fc"],                 4),
        round(_BETA_GAS_DEFAULT, 4), round(_BETA_PRESSURE_DEFAULT, 4),
    )
    shape_cache[ck0] = shapes0

    wd     = data_wt["wtheta"][t_mask]
    err_jk = data_wt["wtheta_err"][t_mask]
    err_jk = np.where(err_jk > 1e-12 * np.abs(wd).max(), err_jk, 1e-12 * np.abs(wd).max())
    # Use the same error definition as log_likelihood (jackknife + systematic floor)
    err    = np.sqrt(err_jk**2 + (f_sys * np.abs(wd))**2)
    wm0    = shapes0["gas"][t_mask]
    wm_agn = shapes0["agn"][t_mask]

    # Joint 2-parameter linear solve: A_gas*wm0 + A_AGN*wm_agn ≈ wd
    # Solves the 2×2 weighted normal equations by Cramer's rule.
    wt_inv = 1.0 / err**2
    sgg = float(np.sum(wm0**2       * wt_inv))
    sga = float(np.sum(wm0 * wm_agn * wt_inv))
    saa = float(np.sum(wm_agn**2    * wt_inv))
    sgd = float(np.sum(wm0    * wd  * wt_inv))
    sad = float(np.sum(wm_agn * wd  * wt_inv))
    det = sgg * saa - sga ** 2
    if abs(det) > 1e-40:
        A_lin     = (sgd * saa - sad * sga) / det
        A_agn_lin = (sad * sgg - sgd * sga) / det
    else:
        A_lin, A_agn_lin = 1.0, 1.0
    if not np.isfinite(A_lin)     or A_lin     <= 0: A_lin     = 1.0
    if not np.isfinite(A_agn_lin) or A_agn_lin <= 0: A_agn_lin = 1.0

    log10_A_gas0 = float(np.log10(max(A_lin,     1e-3)))
    log10_A_AGN0 = float(np.log10(max(A_agn_lin, 1e-5)))

    if _FIT_CFG is not None:
        # Seed every registry param at its default; override the two amplitudes
        # from the linear solve above.  Lock the non-free *cheap* astro params
        # (amplitudes + beta tilts) into _FIT_CFG["fixed_astro"] so _decode_params
        # can recover them; non-free gas-build/AGN params fall back to registry
        # defaults.  Done per label so multi-sample runs use each sample's seed.
        seed = {name: _PARAM_REGISTRY[name][2] for name in _PARAM_REGISTRY}
        seed["log10_A_gas"] = log10_A_gas0
        seed["log10_A_AGN"] = log10_A_AGN0
        for name in _ASTRO_PARAM_NAMES:
            if name not in _FIT_CFG["free"]:
                _FIT_CFG["fixed_astro"][name] = seed[name]
        x0          = np.array([seed[n] for n in _FIT_CFG["free"]])
        bounds      = [(_PARAM_REGISTRY[n][0], _PARAM_REGISTRY[n][1])
                       for n in _FIT_CFG["free"]]
        param_names = list(_FIT_CFG["free"])
    else:
        x0          = np.array([log10_A_gas0, _BETA_GAS_DEFAULT, _BETA_PRESSURE_DEFAULT,
                                log10_A_AGN0, ms0, sig0, m1h0, alpha0, fc0])
        bounds      = list(_PARAM_BOUNDS)
        param_names = list(_PARAM_NAMES)

    def neg_log_prob(p):
        try:
            v = log_prob(p, label, infra, data_wt, t_mask, data_wp, shape_cache, f_sys=f_sys)
            return -v if np.isfinite(v) else 1e30
        except Exception:
            return 1e30

    print(f"  [{label}] MAP: x0={np.round(x0, 3)}  "
          f"n_pts_wtheta={n_pts_wt}  n_pts_smf={n_pts_smf}  n_pts_wp={n_pts_wp}", flush=True)

    res = minimize(
        neg_log_prob, x0, method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-12, "gtol": 1e-7, "maxiter": 2000,
            "eps": 1e-3,   # finite-diff step ≫ cache rounding (1e-4) → enables HOD gradient
        },
    )

    chi2   = 2.0 * res.fun
    result = dict(
        label            = label,
        param_names      = param_names,
        params           = res.x.tolist(),
        agn_model        = infra.agn_model_choice,
        agn_finc         = infra.agn_finc,
        chi2             = float(chi2),
        ndof             = int(ndof),
        chi2_dof         = float(chi2 / ndof),
        log_prob         = float(-res.fun),
        success          = bool(res.success),
        n_pts_wtheta     = n_pts_wt,
        n_pts_smf        = n_pts_smf,
        n_pts_wp         = n_pts_wp,
        n_pts_fit        = n_pts,
        theta_min_arcsec = theta_min,
        theta_max_arcsec = theta_max,
        rp_min_hmpc      = rp_min,
        f_sys            = f_sys,
    )
    if _FIT_CFG is not None:
        result["fixed_zm15"]         = True
        result["free_params"]        = list(_FIT_CFG["free"])
        result["fixed_astro"]        = dict(_FIT_CFG["fixed_astro"])
        result["zm15_params"]        = dict(_FIT_CFG["zm15"])
        result["log10m_star_thresh"] = _FIT_CFG["thresh"][label]
    print(f"  [{label}] MAP done: params={np.round(res.x, 4)}  "
          f"chi2/dof={chi2/ndof:.2f}  success={res.success}", flush=True)
    return result, shape_cache


# --------------------------------------------------------------------------
# MCMC
# --------------------------------------------------------------------------

def run_mcmc(
    label: str,
    infra: _Infrastructure,
    map_result: dict,
    shape_cache: dict,
    n_walkers: int = 32,
    n_steps: int = 1000,
    n_burnin: int = 300,
    theta_min: float = 8.0,
    theta_max: float = 300.0,
    rp_min: float = 0.02,
    f_sys: float = 0.05,
) -> dict:
    """Run emcee ensemble sampler starting near MAP solution."""
    import emcee

    data_wt = load_data(label)
    t_mask  = _angular_mask(data_wt["theta_arcsec"], theta_min, theta_max)
    data_wp = None
    if _FIT_CFG is None and label in _SUM_STAT_DIRS:
        try:
            data_wp = _apply_smfwp_syst(load_smf_wp_data(label, rp_min=rp_min), f_sys)
        except FileNotFoundError:
            pass

    x_map  = np.array(map_result["params"])
    n_free = len(x_map)

    def lp(p):
        return log_prob(p, label, infra, data_wt, t_mask, data_wp, shape_cache,
                        f_sys=f_sys)

    # Walker initialisation scales: tighter for HOD, wider for amplitudes
    # Order: log10_A_gas, beta_gas, beta_pressure, log10_A_AGN, log10m_star_thresh, sigma_lnmstar, lg_m1h, alpha_sat, fc
    if _FIT_CFG is not None:
        scales = np.full(n_free, 0.05)          # free astro subset only
    else:
        scales = np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.10, 0.10, 0.05])
    pos = x_map[None, :] + scales[None, :] * np.random.randn(n_walkers, n_free)

    sampler = emcee.EnsembleSampler(n_walkers, n_free, lp)

    print(f"  [{label}] MCMC burn-in: {n_burnin} steps × {n_walkers} walkers ...", flush=True)
    pos, _, _ = sampler.run_mcmc(pos, n_burnin, progress=True)
    sampler.reset()

    print(f"  [{label}] MCMC production: {n_steps} steps ...", flush=True)
    sampler.run_mcmc(pos, n_steps, progress=True)

    flat_chain = sampler.get_chain(flat=True)   # (n_walkers*n_steps, n_free)

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
        agn_model    = infra.agn_model_choice,
        agn_finc     = infra.agn_finc,
        chain        = flat_chain,
        medians      = medians.tolist(),
        lo16         = lo.tolist(),
        hi84         = hi.tolist(),
        autocorr_tau = tau.tolist() if hasattr(tau, "tolist") else list(tau),
        n_walkers    = n_walkers,
        n_steps      = n_steps,
        n_burnin     = n_burnin,
    )


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

def _unpack_params(params: list) -> dict:
    """Extract named parameters into a dict the plot functions can consume.

    Legacy mode: the 9-element vector maps directly to ``_PARAM_NAMES``.
    Fixed-ZM15 mode: *params* is the free astro subset; the returned dict is
    completed with the fixed ZM15 params so the plot functions (which expect the
    full set of names) work unchanged.  ``log10m_star_thresh`` here is only
    informational / for in-memory cache keys — ``_hod_params(label)`` supplies
    the authoritative per-sample value (and the disk shape-cache is keyed on the
    full HOD dict), so a non-matching value at most triggers a cheap recompute.
    """
    if _FIT_CFG is not None:
        dec         = _decode_params(params)
        zm          = _FIT_CFG["zm15"]
        thresh_vals = list(_FIT_CFG["thresh"].values())
        return {
            "log10_A_gas": dec["log10_A_gas"], "beta_gas": dec["beta_gas"],
            "beta_pressure": dec["beta_pressure"], "log10_A_AGN": dec["log10_A_AGN"],
            "log10m_star_thresh": thresh_vals[0] if thresh_vals else float("nan"),
            "sigma_lnmstar": zm["sigma_lnmstar"], "lg_m1h": zm["lg_m1h"],
            "alpha_sat": zm["alpha_sat"], "fc": zm["fc"],
            # decoded gas/AGN overrides so plot functions can reproduce the
            # fitted shape via _predict_shape (gas_build / agn_cheap / agn_build).
            "_dec": dec,
        }
    names = _PARAM_NAMES
    return {n: params[i] for i, n in enumerate(names)}


def plot_bestfit(
    label: str,
    infra: _Infrastructure,
    params: list,
    shape_cache: dict,
    out_path: Path,
    chi2_dof: float | None = None,
    theta_min: float = 8.0,
    theta_max: float = 300.0,
    rp_min: float = 0.02,
    f_sys: float = 0.0,
) -> None:
    """Two-panel best-fit figure: w_θ(θ) residuals + wp(rp)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p       = _unpack_params(params)
    data_wt = load_data(label)
    s       = SAMPLES[label]

    hp = _hod_params(label,
                     log10m_star_thresh=p["log10m_star_thresh"],
                     sigma_lnmstar=p["sigma_lnmstar"],
                     lg_m1h=p["lg_m1h"], alpha_sat=p["alpha_sat"],
                     fc=p["fc"])
    dec = p.get("_dec")
    ck = (round(p["log10m_star_thresh"], 4), round(p["sigma_lnmstar"],  4),
          round(p["lg_m1h"],             4), round(p["alpha_sat"],      4),
          round(p["fc"],                 4),
          round(p["beta_gas"],           4), round(p["beta_pressure"],  4),
          _dec_shape_key(dec) if dec is not None else None)
    if ck not in shape_cache:
        shape_cache[ck] = _predict_shape(
            label, infra, hp,
            beta_gas=p["beta_gas"], beta_pressure=p["beta_pressure"],
            gas_build=None if dec is None else dec["gas_build"],
            agn_cheap=None if dec is None else dec["agn_cheap"],
            agn_build=None if dec is None else dec["agn_build"],
        )

    shapes  = shape_cache[ck]
    A_gas   = 10.0 ** p["log10_A_gas"]
    A_AGN   = 10.0 ** p["log10_A_AGN"]
    wm_gas         = A_gas * shapes["gas"]
    wm_gas_1h_cen  = A_gas * shapes.get("gas_1h_cen", shapes["gas"] * np.nan)
    wm_gas_1h_sat  = A_gas * shapes.get("gas_1h_sat", shapes["gas"] * np.nan)
    wm_gas_2h      = A_gas * shapes.get("gas_2h",     shapes["gas"] * np.nan)
    wm_agn  = A_AGN * shapes["agn"]
    wmodel  = wm_gas + wm_agn

    theta_as = data_wt["theta_arcsec"]
    R_kpc    = data_wt["R_kpc"]
    mask     = _angular_mask(theta_as, theta_min, theta_max)

    # --- figure layout ---
    has_wp = label in _SUM_STAT_DIRS
    n_rows = 4 if has_wp else 2
    hr     = [3, 1, 2, 1] if has_wp else [3, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 4.0 * n_rows),
                             gridspec_kw={"height_ratios": hr})

    # — w_θ main panel —
    ax = axes[0]
    ax.errorbar(theta_as[~mask], data_wt["wtheta"][~mask],
                yerr=data_wt["wtheta_err"][~mask],
                fmt="o", ms=3, color="0.65", zorder=1)
    ax.errorbar(theta_as[mask], data_wt["wtheta"][mask],
                yerr=data_wt["wtheta_err"][mask],
                fmt="o", ms=4, color="k", label="Data (fitted)", zorder=2)
    ax.plot(theta_as, wmodel, "-",  lw=2.0, color="C0",
            label="Model (gas + AGN)", zorder=4)
    ax.plot(theta_as, wm_gas, "--", lw=1.5, color="C2",
            label=f"Gas total  $\\beta_g={p['beta_gas']:.2f}$, "
                  f"$\\log_{{10}}A={p['log10_A_gas']:.2f}$", zorder=3)
    if np.isfinite(wm_gas_1h_cen).any():
        ax.plot(theta_as, wm_gas_1h_cen, "-.",  lw=1.0, color="C2",
                label="Gas 1h-cen", zorder=2)
        ax.plot(theta_as, wm_gas_1h_sat, ":",   lw=1.0, color="C3",
                label="Gas 1h-sat", zorder=2)
        ax.plot(theta_as, wm_gas_2h,     "--",  lw=1.0, color="C4",
                label="Gas 2h", zorder=2)
    _agn_tag = infra.agn_model_choice.upper()
    ax.plot(theta_as, wm_agn, ":",  lw=1.5, color="C1",
            label=f"AGN ({_agn_tag})  $\\log_{{10}}A={p['log10_A_AGN']:.2f}$",
            zorder=3)
    ax.axvline(_PSF_KING_THETA_C, ls=":", color="C3", lw=0.9, alpha=0.7,
               label=f"PSF $\\theta_c={_PSF_KING_THETA_C}$\"")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$w_\theta(\theta)$")
    ax.set_title(
        f"{label}: $\\log_{{10}} M_* > {s['log10ms_min']}$, "
        f"$z_{{\\rm mean}}={s['zmean']:.3f}$"
        + (f"\n$\\chi^2/\\nu = {chi2_dof:.2f}$" if chi2_dof else "")
    )
    ax.legend(fontsize=7)

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    tick_as = np.array([5, 10, 30, 100, 300])
    tick_as = tick_as[(tick_as >= theta_as.min()) & (tick_as <= theta_as.max())]
    conv    = R_kpc[0] / theta_as[0]
    ax2.set_xticks(tick_as)
    ax2.set_xticklabels([f"{v*conv:.0f}" for v in tick_as], fontsize=8)
    ax2.set_xlabel("R [kpc]", fontsize=9)

    # — w_θ residual panel —
    ax_res = axes[1]
    ratio  = data_wt["wtheta"] / wmodel
    err_r  = data_wt["wtheta_err"] / wmodel
    ax_res.errorbar(theta_as[~mask], ratio[~mask], yerr=err_r[~mask],
                    fmt="o", ms=3, color="0.65", zorder=1)
    ax_res.errorbar(theta_as[mask], ratio[mask], yerr=err_r[mask],
                    fmt="o", ms=4, color="k", zorder=2)
    ax_res.axhline(1.0, ls="-",  lw=1,   color="C0")
    ax_res.axhline(1.1, ls="--", lw=0.8, color="gray")
    ax_res.axhline(0.9, ls="--", lw=0.8, color="gray")
    ax_res.set_ylim(0.2, 2.0)
    ax_res.set_ylabel("Data / Model")
    ax_res.set_xscale("log")
    if not has_wp:
        ax_res.set_xlabel(r"$\theta$ [arcsec]")

    # — wp panel + residual —
    if has_wp:
        try:
            data_wp = _apply_wp_syst(load_wp_data(label, rp_min=rp_min), f_sys)
        except FileNotFoundError:
            data_wp = None
        ax_wp    = axes[2]
        ax_wp_r  = axes[3]
        if data_wp is not None:
            z_eff    = SAMPLES[label]["zmean"]
            wp_model = np.asarray(
                infra.fhmp.wp(data_wp["rp"], pi_max=data_wp["pi_max"],
                              z=z_eff, theta_cosmo=_THETA_COSMO, hod_params=hp),
                dtype=float,
            )
            err_wp = np.sqrt(1.0 / np.diag(data_wp["icov"]))
            ax_wp.errorbar(data_wp["rp"], data_wp["wp"], yerr=err_wp,
                           fmt="o", ms=4, color="k", label="Data (wp)", zorder=2)
            ax_wp.plot(data_wp["rp"], wp_model, "-", lw=2.0, color="C0",
                       label="Model", zorder=3)
            ax_wp.set_xscale("log"); ax_wp.set_yscale("log")
            ax_wp.axvline(rp_min, ls="--", color="gray", lw=0.8)
            ax_wp.set_ylabel(r"$w_p(r_p)$ [Mpc/$h$]")
            ax_wp.legend(fontsize=8)
            # wp residual
            ratio_wp  = data_wp["wp"] / wp_model
            err_r_wp  = err_wp / wp_model
            ax_wp_r.errorbar(data_wp["rp"], ratio_wp, yerr=err_r_wp,
                             fmt="o", ms=4, color="k", zorder=2)
            ax_wp_r.axhline(1.0, ls="-",  lw=1,   color="C0")
            ax_wp_r.axhline(1.1, ls="--", lw=0.8, color="gray")
            ax_wp_r.axhline(0.9, ls="--", lw=0.8, color="gray")
            ax_wp_r.set_xscale("log")
            ax_wp_r.set_ylim(0.5, 2.0)
            ax_wp_r.set_ylabel("Data / Model")
            ax_wp_r.axvline(rp_min, ls="--", color="gray", lw=0.8)
        ax_wp_r.set_xlabel(r"$r_p$ [Mpc/$h$]")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{label}] best-fit figure → {out_path}")


def _load_mcmc_lg_m1h(label: str, out_dir: Path) -> np.ndarray | None:
    """Posterior samples of ``lg_m1h`` from a saved MCMC chain, if one exists.

    Looks for ``{label}_chain.h5`` (preferred) or ``{label}_chain.npy`` next
    to the MAP json. Returns None if no chain has been run yet (--mode map) or
    if ``lg_m1h`` is not among the sampled parameters (fixed-ZM15 chains hold the
    ZM15 connection fixed, so they never sample it).
    """
    h5_path = out_dir / f"{label}_chain.h5"
    if h5_path.exists():
        import h5py
        with h5py.File(h5_path, "r") as f:
            chain = np.asarray(f["chain"])
            names = list(f.attrs["param_names"])
        return chain[:, names.index("lg_m1h")] if "lg_m1h" in names else None
    npy_path = out_dir / f"{label}_chain.npy"
    if npy_path.exists() and "lg_m1h" in _PARAM_NAMES:
        chain = np.load(npy_path)
        return chain[:, _PARAM_NAMES.index("lg_m1h")]
    return None


def _shmr_zu15(log10mh_grid: np.ndarray, hp: dict) -> np.ndarray:
    """Mean ZM15 SHMR log10(M*/[Msun/h]) at fixed log10(Mh/[Msun/h])."""
    from hod_mod.connection.hod import _mstar_from_mh_zu15
    return np.asarray(_mstar_from_mh_zu15(
        log10mh_grid, hp["lg_m1h"], hp["lg_m0star"], hp["beta"], hp["delta"], hp["gamma"],
    ))


def plot_diagnostics(
    label: str,
    infra: _Infrastructure,
    params: list,
    out_path: Path,
) -> None:
    """Diagnostic panels: SMF, n_gal, SHMR, gg-lensing, X-ray auto-power.

    1. Stellar-mass function Phi(M*): sum_stat data vs model prediction
       (fit jointly with w_theta + wp since the SMF term was added to
       ``log_likelihood``).
    2. Integrated galaxy number density: model vs sum_stat, by integrating
       Phi(M*) over the same stellar-mass range — now a fit residual (the
       SMF is part of the joint likelihood), so a large ratio here indicates
       a poor fit rather than an inherent HOD degeneracy.
    3. Stellar-to-halo mass relation: MAP (or MCMC posterior band, if a chain
       has been run) vs Moster+2013, Behroozi+2013, Girelli+2020.
    4. Galaxy-galaxy lensing Delta_Sigma(rp): HSC/DES/KIDS data vs model
       prediction (same best-fit HOD params as the wp/w_theta/SMF fit — NOT
       fit to the lensing data).
    5. X-ray auto-power spectrum C_ell^{XX}: model prediction only (no data).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hod_mod.connection import sham

    infra.use_agn_for(label)   # attach per-sample AGN before the direct angular_cl_XX call

    p     = _unpack_params(params)
    s     = SAMPLES[label]
    z_eff = s["zmean"]
    hp = _hod_params(label,
                     log10m_star_thresh=p["log10m_star_thresh"],
                     sigma_lnmstar=p["sigma_lnmstar"],
                     lg_m1h=p["lg_m1h"], alpha_sat=p["alpha_sat"],
                     fc=p["fc"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))

    # --- Panel 1: stellar mass function ---
    ax = axes[0, 0]
    try:
        smf_data = load_smf_data(label)
        ax.errorbar(smf_data["log10mstar"], smf_data["phi"], yerr=smf_data["phi_err"],
                    fmt="o", ms=4, color="k", label="sum_stat SMF", zorder=2)
        data_grid = smf_data["log10mstar"]
    except (FileNotFoundError, StopIteration):
        smf_data = None
        data_grid = np.linspace(s["log10ms_min"], s["log10ms_min"] + 2.0, 20)
    mgrid_fine = np.linspace(data_grid.min() - 0.2, data_grid.max() + 0.2, 60)
    phi_model_fine = _predict_smf(infra, z_eff, hp, mgrid_fine)
    ax.plot(mgrid_fine, phi_model_fine, "-", lw=2.0, color="C0", label="Model (HOD)", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel(r"$\Phi(M_*)$ [Mpc$^{-3}$ dex$^{-1}$]")
    ax.set_title(f"{label}: stellar mass function")
    ax.legend(fontsize=8)

    # --- Panel 2: integrated number density, model vs sum_stat ---
    # Integrate the model SMF over the SAME stellar-mass grid as the data so the
    # ratio is apples-to-apples (the widened mgrid_fine would inflate n_model).
    ax = axes[0, 1]
    n_model = float(trapezoid(phi_model_fine, mgrid_fine))   # fallback (no data)
    if smf_data is not None:
        n_data  = float(trapezoid(smf_data["phi"], smf_data["log10mstar"]))
        n_model = float(trapezoid(_predict_smf(infra, z_eff, hp, data_grid), data_grid))
        bars = ax.bar(["sum_stat", "Model (HOD)"], [n_data, n_model],
                      color=["k", "C0"], alpha=0.75)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\bar n_g = \int \Phi(M_*)\,d\log_{10}M_*$ [Mpc$^{-3}$]")
        ax.set_title(f"{label}: $\\bar n_g$ ratio = {n_model / n_data:.2f}")
        for bar, val in zip(bars, [n_data, n_model]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2e}",
                    ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No sum_stat SMF\n(n_gal comparison unavailable)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{label}: $\\bar n_g$ (model) = {n_model:.2e} Mpc$^{{-3}}$")
    ax.grid(True, alpha=0.2, axis="y")

    # --- Panel 3: stellar-to-halo mass relation ---
    ax = axes[0, 2]
    mh_grid = np.linspace(10.5, 15.0, 60)
    out_dir = out_path.parent
    chain_lg_m1h = _load_mcmc_lg_m1h(label, out_dir)
    if chain_lg_m1h is not None:
        lo, hi = np.percentile(chain_lg_m1h, [16, 84])
        ms_lo = _shmr_zu15(mh_grid, {**hp, "lg_m1h": lo})
        ms_hi = _shmr_zu15(mh_grid, {**hp, "lg_m1h": hi})
        ax.fill_between(mh_grid, np.minimum(ms_lo, ms_hi), np.maximum(ms_lo, ms_hi),
                        color="C3", alpha=0.25, label="ZM15 posterior (16-84%)")
        shmr_label = "ZM15 posterior median"
    else:
        shmr_label = "ZM15 (MAP)"
    ms_map = _shmr_zu15(mh_grid, hp)
    ax.plot(mh_grid, ms_map, "-", lw=2.2, color="C3", label=shmr_label, zorder=5)
    ax.plot(mh_grid, sham.smhm_moster13(mh_grid, z_eff), "--", lw=1.4,
            color="C0", label="Moster+2013")
    ax.plot(mh_grid, sham.smhm_behroozi13(mh_grid, z_eff), "-.", lw=1.4,
            color="C1", label="Behroozi+2013")
    ax.plot(mh_grid, sham.smhm_girelli20(mh_grid, z_eff), ":", lw=1.4,
            color="C2", label="Girelli+2020")
    ax.set_xlabel(r"$\log_{10}(M_h/[M_\odot\,h^{-1}])$")
    ax.set_ylabel(r"$\log_{10}(M_*/[M_\odot\,h^{-1}])$")
    ax.set_title(f"{label}: stellar-to-halo mass relation ($z={z_eff:.2f}$)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2)

    # --- Panel 4: galaxy-galaxy lensing ---
    ax = axes[1, 0]
    colors = {"HSC": "C0", "DES": "C1", "KIDS": "C2"}
    rp_for_model = None
    for survey, color in colors.items():
        try:
            esd = load_esd_data(label, survey)
        except (FileNotFoundError, StopIteration):
            continue
        ax.errorbar(esd["rp"], esd["delta_sigma"], yerr=esd["delta_sigma_err"],
                    fmt="o", ms=4, color=color, label=f"{survey} (data)", zorder=2)
        if rp_for_model is None:
            rp_for_model = esd["rp"]
    if rp_for_model is not None:
        ds_model = np.asarray(
            infra.fhmp.delta_sigma(np.asarray(rp_for_model), z_eff, _THETA_COSMO, hp),
            dtype=float,
        )
        ax.plot(rp_for_model, ds_model, "-", lw=2.0, color="k",
                label="Model (not fit)", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax.set_ylabel(r"$\Delta\Sigma(r_p)$ [$M_\odot\,h\,\mathrm{pc}^{-2}$]")
    ax.set_title(f"{label}: galaxy-galaxy lensing")
    ax.legend(fontsize=8)

    # --- Panel 5: X-ray auto-power spectrum ---
    ax = axes[1, 1]
    z_arr, nz_g = _build_nz_fast(label)
    cl_XX = infra.cross.angular_cl_XX(
        _ELL, z_arr, nz_g, _THETA_COSMO,
        beta_gas=p["beta_gas"], beta_pressure=p["beta_pressure"],
        psf_king_theta_c_arcsec=_PSF_KING_THETA_C,
        return_components=True,
        n_workers=1,
    )
    ax.plot(_ELL, cl_XX["total"],    "-",  lw=2.0, color="C0", label="Total")
    ax.plot(_ELL, cl_XX["gas_gas"],  "--", lw=1.2, color="C2", label="Gas $\\times$ Gas")
    ax.plot(_ELL, np.abs(cl_XX["cross"]), ":", lw=1.2, color="C3", label="|Gas $\\times$ AGN|")
    ax.plot(_ELL, cl_XX["agn_agn"],  "-.", lw=1.2, color="C1", label="AGN $\\times$ AGN")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{XX}$ (model, no data)")
    ax.set_title(f"{label}: X-ray auto-power (forward model)")
    ax.legend(fontsize=8)

    axes[1, 2].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{label}] diagnostics figure → {out_path}")


def plot_bestfit_hmf_compare(
    label: str,
    infra_by_hmf: dict,
    params: list,
    out_path: Path,
    chi2_dof: float | None = None,
    theta_min: float = 8.0,
    theta_max: float = 300.0,
    rp_min: float = 0.02,
    f_sys: float = 0.0,
) -> None:
    """w_θ(θ) + wp(rp) best-fit figure, one model line per halo mass function.

    All astrophysical parameters (gas, AGN, HOD) are fixed at the saved MAP
    values; only the HMF backend used inside ``infra.fhmp``/``infra.cross``
    is varied, to show how sensitive the Comparat+2025 model predictions are
    to the choice of halo mass function.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    p       = _unpack_params(params)
    data_wt = load_data(label)
    s       = SAMPLES[label]
    hp = _hod_params(label,
                     log10m_star_thresh=p["log10m_star_thresh"],
                     sigma_lnmstar=p["sigma_lnmstar"],
                     lg_m1h=p["lg_m1h"], alpha_sat=p["alpha_sat"],
                     fc=p["fc"])
    A_gas = 10.0 ** p["log10_A_gas"]
    A_AGN = 10.0 ** p["log10_A_AGN"]
    z_eff = s["zmean"]

    names  = list(infra_by_hmf)
    colors = cm.get_cmap("turbo")(np.linspace(0.0, 1.0, len(names)))

    has_wp = label in _SUM_STAT_DIRS
    n_rows = 4 if has_wp else 2
    hr     = [3, 1, 2, 1] if has_wp else [3, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 4.0 * n_rows),
                             gridspec_kw={"height_ratios": hr})

    theta_as = data_wt["theta_arcsec"]
    mask     = _angular_mask(theta_as, theta_min, theta_max)

    ax, ax_res = axes[0], axes[1]
    ax.errorbar(theta_as[~mask], data_wt["wtheta"][~mask],
                yerr=data_wt["wtheta_err"][~mask],
                fmt="o", ms=3, color="0.65", zorder=1)
    ax.errorbar(theta_as[mask], data_wt["wtheta"][mask],
                yerr=data_wt["wtheta_err"][mask],
                fmt="o", ms=4, color="k", label="Data (fitted)", zorder=2)
    ax_res.axhline(1.0, ls="-", lw=1, color="k")

    wp_data = None
    ax_wp = ax_wp_r = None
    if has_wp:
        ax_wp, ax_wp_r = axes[2], axes[3]
        try:
            wp_data = _apply_wp_syst(load_wp_data(label, rp_min=rp_min), f_sys)
        except FileNotFoundError:
            wp_data = None
        if wp_data is not None:
            err_wp = np.sqrt(1.0 / np.diag(wp_data["icov"]))
            ax_wp.errorbar(wp_data["rp"], wp_data["wp"], yerr=err_wp,
                           fmt="o", ms=4, color="k", label="Data (wp)", zorder=2)
            ax_wp_r.axhline(1.0, ls="-", lw=1, color="k")
            ax_wp.axvline(rp_min, ls="--", color="gray", lw=0.8)
            ax_wp_r.axvline(rp_min, ls="--", color="gray", lw=0.8)

    for name, color in zip(names, colors):
        infra  = infra_by_hmf[name]
        shapes = _predict_shape(label, infra, hp, beta_gas=p["beta_gas"],
                                beta_pressure=p["beta_pressure"], use_disk_cache=False)
        wmodel = A_gas * shapes["gas"] + A_AGN * shapes["agn"]
        ax.plot(theta_as, wmodel, "-", lw=1.3, color=color, label=name, zorder=3)
        ratio = data_wt["wtheta"] / wmodel
        ax_res.plot(theta_as, ratio, "-", lw=1.0, color=color, alpha=0.85, zorder=2)

        if has_wp and wp_data is not None:
            wp_model = np.asarray(
                infra.fhmp.wp(wp_data["rp"], pi_max=wp_data["pi_max"],
                              z=z_eff, theta_cosmo=_THETA_COSMO, hod_params=hp),
                dtype=float,
            )
            ax_wp.plot(wp_data["rp"], wp_model, "-", lw=1.3, color=color, zorder=3)
            ratio_wp = wp_data["wp"] / wp_model
            ax_wp_r.plot(wp_data["rp"], ratio_wp, "-", lw=1.0, color=color, alpha=0.85, zorder=2)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$w_\theta(\theta)$")
    ax.set_title(
        f"{label}: HMF comparison at fixed MAP astro params  "
        f"($\\log_{{10}} M_* > {s['log10ms_min']}$, $z_{{\\rm mean}}={s['zmean']:.3f}$)"
        + (f"\n$\\chi^2/\\nu$ (saved MAP fit) $= {chi2_dof:.2f}$" if chi2_dof else "")
    )
    ax.legend(fontsize=6, ncol=2)

    ax_res.set_ylim(0.2, 2.0)
    ax_res.set_ylabel("Data / Model")
    ax_res.set_xscale("log")
    if not has_wp:
        ax_res.set_xlabel(r"$\theta$ [arcsec]")
    elif wp_data is not None:
        ax_wp.set_xscale("log"); ax_wp.set_yscale("log")
        ax_wp.set_ylabel(r"$w_p(r_p)$ [Mpc/$h$]")
        ax_wp.legend(fontsize=6, ncol=2)
        ax_wp_r.set_xscale("log")
        ax_wp_r.set_ylim(0.5, 2.0)
        ax_wp_r.set_ylabel("Data / Model")
        ax_wp_r.set_xlabel(r"$r_p$ [Mpc/$h$]")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{label}] HMF-comparison best-fit figure → {out_path}")


def plot_diagnostics_hmf_compare(
    label: str,
    infra_by_hmf: dict,
    params: list,
    out_path: Path,
) -> None:
    """SMF, gg-lensing and X-ray auto-power diagnostics, one line per HMF backend.

    Same three panels as :func:`plot_diagnostics` but overlaying the model
    curve for every halo mass function in ``infra_by_hmf`` at fixed MAP
    astrophysical parameters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    p     = _unpack_params(params)
    s     = SAMPLES[label]
    z_eff = s["zmean"]
    hp = _hod_params(label,
                     log10m_star_thresh=p["log10m_star_thresh"],
                     sigma_lnmstar=p["sigma_lnmstar"],
                     lg_m1h=p["lg_m1h"], alpha_sat=p["alpha_sat"],
                     fc=p["fc"])

    names  = list(infra_by_hmf)
    colors = cm.get_cmap("turbo")(np.linspace(0.0, 1.0, len(names)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # --- Panel 1: stellar mass function ---
    ax = axes[0]
    try:
        smf_data = load_smf_data(label)
        ax.errorbar(smf_data["log10mstar"], smf_data["phi"], yerr=smf_data["phi_err"],
                    fmt="o", ms=4, color="k", label="sum_stat SMF", zorder=2)
        mgrid = smf_data["log10mstar"]
    except (FileNotFoundError, StopIteration):
        mgrid = np.linspace(s["log10ms_min"], s["log10ms_min"] + 2.0, 20)
    for name, color in zip(names, colors):
        phi_model = _predict_smf(infra_by_hmf[name], z_eff, hp, mgrid)
        ax.plot(mgrid, phi_model, "-", lw=1.3, color=color, label=name, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel(r"$\Phi(M_*)$ [Mpc$^{-3}$ dex$^{-1}$]")
    ax.set_title(f"{label}: stellar mass function (HMF comparison)")
    ax.legend(fontsize=5, ncol=2)

    # --- Panel 2: galaxy-galaxy lensing ---
    ax = axes[1]
    data_colors = {"HSC": "C0", "DES": "C1", "KIDS": "C2"}
    rp_for_model = None
    for survey, dcolor in data_colors.items():
        try:
            esd = load_esd_data(label, survey)
        except (FileNotFoundError, StopIteration):
            continue
        ax.errorbar(esd["rp"], esd["delta_sigma"], yerr=esd["delta_sigma_err"],
                    fmt="o", ms=4, color=dcolor, label=f"{survey} (data)", zorder=2)
        if rp_for_model is None:
            rp_for_model = esd["rp"]
    if rp_for_model is not None:
        for name, color in zip(names, colors):
            ds_model = np.asarray(
                infra_by_hmf[name].fhmp.delta_sigma(
                    np.asarray(rp_for_model), z_eff, _THETA_COSMO, hp),
                dtype=float,
            )
            ax.plot(rp_for_model, ds_model, "-", lw=1.3, color=color, label=name, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax.set_ylabel(r"$\Delta\Sigma(r_p)$ [$M_\odot\,h\,\mathrm{pc}^{-2}$]")
    ax.set_title(f"{label}: galaxy-galaxy lensing (HMF comparison)")
    ax.legend(fontsize=5, ncol=2)

    # --- Panel 3: X-ray auto-power spectrum ---
    ax = axes[2]
    z_arr, nz_g = _build_nz_fast(label)
    for name, color in zip(names, colors):
        cl_XX = infra_by_hmf[name].cross.angular_cl_XX(
            _ELL, z_arr, nz_g, _THETA_COSMO,
            beta_gas=p["beta_gas"], beta_pressure=p["beta_pressure"],
            psf_king_theta_c_arcsec=_PSF_KING_THETA_C,
            return_components=True,
            n_workers=1,
        )
        ax.plot(_ELL, cl_XX["total"], "-", lw=1.3, color=color, label=name, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{XX}$ (model, no data)")
    ax.set_title(f"{label}: X-ray auto-power (HMF comparison)")
    ax.legend(fontsize=5, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{label}] HMF-comparison diagnostics figure → {out_path}")


def plot_gas_diagnostics(
    label: str,
    infra: _Infrastructure,
    params: list,
    out_path: Path,
) -> None:
    """Gas scaling-relation + radial-profile diagnostics for the MAP best fit.

    Not part of the w_theta+wp fit likelihood — diagnostic only. Reuses the
    GAS.py amplitude-calibration algorithm from
    hod_mod/scripts/validate_gas_profiles.py (``_calibrate_ne03_P03``), fixing
    the density/pressure mass-scaling slopes to this sample's MAP
    ``beta_gas``/``beta_pressure`` and solving for the DPM amplitudes
    (``ne_03``, ``P_03``) that reproduce the GAS.py (Comparat+2025 Uchuu
    light-cone) Lx-M and kT-M normalisation at the pivot mass.

    Panels:
      (0,0) Lx-M500c   (0,1) kT-M500c   (0,2) Lx-kT   scaling relations,
            vs Lovisari+2020 (arXiv:2004.03401).
      (1,0-3) n_e(r), P_e(r), Z(r), T(r) radial profiles at
            M200 = 1e14 Msun/h, z = sample z_mean.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hod_mod.scripts import validate_gas_profiles as vgp

    p     = _unpack_params(params)
    s     = SAMPLES[label]
    z_eff = s["zmean"]
    beta_n, beta_p = p["beta_gas"], p["beta_pressure"]

    ne_cal, P03_cal = vgp._calibrate_ne03_P03(beta_n, beta_p, T_min=0.3, z=z_eff)
    dp_cal = vgp._make_density_variant(model=2, ne_03=ne_cal, beta=beta_n)
    pp_cal = vgp._make_pressure_variant(model=2, P_03=P03_cal, beta=beta_p)
    met    = vgp.MetallicityProfileDPM()

    # --- scaling relations vs M500c ---
    m_arr    = np.logspace(11.5, 15.2, 30)             # M200 [Msun/h]
    r_arr    = vgp._r200(m_arr, z_eff)
    c200_arr = vgp._c200_approx(m_arr)
    m500_arr, r500_arr = vgp.m200_to_m500c(m_arr, c200_arr, r_arr, vgp._rho_crit_z(z_eff))
    m500_msun = m500_arr / vgp._H

    Lx_cal = np.zeros(len(m_arr))
    kT_cal = np.zeros(len(m_arr))
    for j, (m, r2, r5) in enumerate(zip(m_arr, r_arr, r500_arr)):
        Lx_cal[j], kT_cal[j], _ = vgp._integrate_profile(
            m, r2, r5, z_eff, pp_cal, dp_cal, met, T_min=0.3)

    m_lit  = np.logspace(12.0, 15.5, 80)
    Lx_lit = vgp._lovisari20_lx(m_lit, z=z_eff)
    kT_lit = vgp._lovisari20_kt(m_lit, z=z_eff)

    # Individual / stacked observational data points (same as validate_gas_profiles.py)
    M_lo20, Lx_lo20, kT_lo20      = vgp._load_lovisari20_data()
    M_bu18, Lx_bu18, kT_bu18      = vgp._load_bulbul18()
    M_lo15, Lx_lo15, kT_lo15      = vgp._load_lovisari15()
    M_zh24, Lx_zh24, Lx_zh24_err  = vgp._load_zhang24()
    M_po24, Lx_po24_hi, Lx_po24_lo = vgp._load_popesso24()

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))

    alpha_sr, alpha_sr_hi, alpha_sr_lo = _ALPHA_SR_C25[label]
    Lx_c25      = _comparat25_lx_sr(m_lit, z_eff, alpha_sr)
    Lx_c25_plus = _comparat25_lx_sr(m_lit, z_eff, alpha_sr + alpha_sr_hi)
    Lx_c25_minus = _comparat25_lx_sr(m_lit, z_eff, alpha_sr - alpha_sr_lo)

    ax = axes[0, 0]
    ax.fill_between(M_po24, Lx_po24_lo, Lx_po24_hi,
                    alpha=0.15, color="purple", label="Popesso+2024")
    ax.scatter(M_lo20, Lx_lo20, s=12, color="gray", alpha=0.7,
               marker="o", label="Lovisari+2020 clusters", zorder=5)
    ax.scatter(M_bu18, Lx_bu18, s=14, color="steelblue", alpha=0.7,
               marker="s", label="Bulbul+2018 clusters", zorder=5)
    ax.errorbar(M_zh24, Lx_zh24, yerr=Lx_zh24_err, fmt="D",
                color="darkorange", ms=5, lw=1.2, label="Zhang+2024 CGM stacks", zorder=6)
    ax.fill_between(m_lit, np.minimum(Lx_c25_plus, Lx_c25_minus),
                    np.maximum(Lx_c25_plus, Lx_c25_minus),
                    alpha=0.2, color="darkviolet", zorder=4,
                    label=rf"Comparat+2025 ($\alpha_{{SR}}={alpha_sr:.3f}$, $1\sigma$)")
    ax.loglog(m_lit, Lx_c25, "-", color="darkviolet", lw=1.8, zorder=4)
    ax.loglog(m500_msun, Lx_cal, "C3-", lw=2.5,
              label=rf"MAP ($\beta_n={beta_n:.2f}$, $\beta_P={beta_p:.2f}$)", zorder=7)
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$L_X$ (0.5–2 keV) [erg s$^{-1}$]")
    ax.set_title(f"{label}: $L_X$–$M_{{500c}}$ (MAP)")
    # Tight y-range (was 1e38–1e47, 9 decades, which hid a ~0.5–1 dex model
    # cluster-Lx deficit visible in the Lx–kT panel) so the same deficit shows here.
    ax.set_xlim(1e11, 3e15); ax.set_ylim(1e40, 1e46)
    ax.legend(fontsize=6.5, ncol=2); ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    ax.loglog(m_lit, kT_lit, "k--", lw=1.5, label="Lovisari+2020 (fit)")
    ax.scatter(M_lo20, kT_lo20, s=12, color="gray", alpha=0.7,
               marker="o", label="Lovisari+2020", zorder=5)
    ax.scatter(M_bu18, kT_bu18, s=14, color="steelblue", alpha=0.7,
               marker="s", label="Bulbul+2018", zorder=5)
    ax.scatter(M_lo15, kT_lo15, s=14, color="green", alpha=0.7,
               marker="^", label="Lovisari+2015 groups", zorder=5)
    ax.loglog(m500_msun, kT_cal, "C3-", lw=2.5, label="MAP", zorder=7)
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$kT_{\rm ew}$ [keV]")
    ax.set_title(f"{label}: $kT$–$M_{{500c}}$ (MAP)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax = axes[0, 2]
    ax.scatter(kT_lo20, Lx_lo20, s=12, color="gray", alpha=0.7,
               marker="o", label="Lovisari+2020", zorder=5)
    ax.scatter(kT_bu18, Lx_bu18, s=14, color="steelblue", alpha=0.7,
               marker="s", label="Bulbul+2018", zorder=5)
    ax.loglog(kT_cal, Lx_cal, "C3-", lw=2.5, label="MAP", zorder=7)
    ax.set_xlabel(r"$kT_{\rm ew}$ [keV]")
    ax.set_ylabel(r"$L_X$ [erg s$^{-1}$]")
    ax.set_title(f"{label}: $L_X$–$kT$ (MAP)")
    ax.legend(fontsize=6.5); ax.grid(True, alpha=0.2)

    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0, 0.5,
        "GAS.py-calibrated DPM model 2\n"
        rf"$\beta_n={beta_n:.3f}$ (MAP beta_gas)" "\n"
        rf"$\beta_P={beta_p:.3f}$ (MAP beta_pressure)" "\n"
        rf"$n_{{e,0.3}}={ne_cal:.2e}$ cm$^{{-3}}$" "\n"
        rf"$P_{{0.3}}={P03_cal:.2e}$ keV cm$^{{-3}}$" "\n"
        "T > 0.3 keV X-ray selection cut\n"
        rf"$z={z_eff:.3f}$",
        fontsize=9, va="center", transform=axes[0, 3].transAxes,
    )

    # --- radial profiles at M200 = 1e11..1e15 Msun/h ---
    log10m_profs = [11, 12, 13, 14, 15]
    prof_colors  = plt.cm.viridis(np.linspace(0.05, 0.95, len(log10m_profs)))
    x_arr = np.logspace(-2, np.log10(4), 200)

    ax_ne, ax_pe, ax_z, ax_t = axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]
    for log10m, color in zip(log10m_profs, prof_colors):
        m_prof  = 10.0 ** log10m
        r2_prof = vgp._r200(m_prof, z_eff)
        r_arr_p = x_arr * r2_prof

        ne = dp_cal.density_3d(r_arr_p, m_prof, r2_prof, z_eff, vgp._OM)
        Pe = pp_cal._pressure_3d(r_arr_p, m_prof, r2_prof, z_eff, vgp._OM)
        Z  = met.metallicity_3d(r_arr_p, r2_prof)
        T  = vgp.temperature_from_profiles(Pe, ne)

        lbl = rf"$M_{{200}}=10^{{{log10m}}}\,M_\odot/h$"
        ax_ne.loglog(x_arr, ne, "-", lw=2, color=color, label=lbl)
        ax_pe.loglog(x_arr, Pe, "-", lw=2, color=color, label=lbl)
        ax_z.semilogx(x_arr, Z, "-", lw=2, color=color, label=lbl)
        ax_t.loglog(x_arr, T, "-", lw=2, color=color, label=lbl)

    ax_ne.set_xlabel(r"$r/R_{200}$"); ax_ne.set_ylabel(r"$n_e(r)$ [cm$^{-3}$]")
    ax_ne.set_title(f"{label}: $n_e(r)$ (MAP)")
    ax_ne.legend(fontsize=6.5); ax_ne.grid(True, alpha=0.2)

    ax_pe.set_xlabel(r"$r/R_{200}$"); ax_pe.set_ylabel(r"$P_e(r)$ [keV cm$^{-3}$]")
    ax_pe.set_title(f"{label}: $P_e(r)$ (MAP)")
    ax_pe.legend(fontsize=6.5); ax_pe.grid(True, alpha=0.2)

    ax_z.set_xlabel(r"$r/R_{200}$"); ax_z.set_ylabel(r"$Z(r)$ [$Z_\odot$]")
    ax_z.set_title(f"{label}: $Z(r)$")
    ax_z.legend(fontsize=6.5); ax_z.grid(True, alpha=0.2)

    ax_t.set_xlabel(r"$r/R_{200}$"); ax_t.set_ylabel(r"$T(r)=P_e/n_e$ [keV]")
    ax_t.set_title(f"{label}: $T(r)$ (MAP)")
    ax_t.legend(fontsize=6.5); ax_t.grid(True, alpha=0.2)

    fig.suptitle(
        f"{label}: gas scaling relations and radial profiles "
        "(MAP beta_gas/beta_pressure, amplitude-calibrated to GAS.py — not fit to X-ray scaling data)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{label}] gas diagnostics figure → {out_path}")


def plot_corner(label: str, mcmc_result: dict, out_path: Path) -> None:
    """Corner plot of MCMC posterior."""
    try:
        import corner
    except ImportError:
        print("  corner not installed — skipping corner plot")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chain = mcmc_result["chain"]
    _TEX = {
        "log10_A_gas":        r"$\log_{10} A_{\rm gas}$",
        "beta_gas":           r"$\beta_{\rm gas}$",
        "beta_pressure":      r"$\beta_{\rm P}$",
        "log10_A_AGN":        r"$\log_{10} A_{\rm AGN}$",
        "log10m_star_thresh": r"$\log_{10} M_{*,\rm th}$",
        "sigma_lnmstar":      r"$\sigma_{\ln M_*}$",
        "lg_m1h":             r"$\log_{10} M_{1h}$",
        "alpha_sat":          r"$\alpha_{\rm sat}$",
        "fc":                 r"$f_c$",
    }
    labels_tex = [_TEX.get(n, n) for n in mcmc_result["param_names"]]

    fig = corner.corner(
        chain,
        labels=labels_tex,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    fig.suptitle(f"{label}  MCMC posterior (HAM + King PSF + wp)", y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{label}] corner plot → {out_path}")


def plot_all_bestfit(
    all_results: list[dict],
    infra: _Infrastructure,
    shape_caches: dict,
    out_path: Path,
) -> None:
    """Overview panel for all fitted samples (w_θ only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n     = len(all_results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows),
                             sharex=False, sharey=False)
    axes_flat = np.array(axes).flatten()

    for ax, res in zip(axes_flat, all_results):
        label  = res["label"]
        data   = load_data(label)
        p      = _unpack_params(res["params"])

        hp  = _hod_params(label,
                          log10m_star_thresh=p["log10m_star_thresh"],
                          sigma_lnmstar=p["sigma_lnmstar"],
                          lg_m1h=p["lg_m1h"], alpha_sat=p["alpha_sat"],
                          fc=p["fc"])
        dec = p.get("_dec")
        ck  = (round(p["log10m_star_thresh"], 4), round(p["sigma_lnmstar"],  4),
               round(p["lg_m1h"],             4), round(p["alpha_sat"],      4),
               round(p["fc"],                 4),
               round(p["beta_gas"],           4), round(p["beta_pressure"],  4),
               _dec_shape_key(dec) if dec is not None else None)
        cache = shape_caches.get(label, {})
        if ck not in cache:
            cache[ck] = _predict_shape(
                label, infra, hp,
                beta_gas=p["beta_gas"], beta_pressure=p["beta_pressure"],
                gas_build=None if dec is None else dec["gas_build"],
                agn_cheap=None if dec is None else dec["agn_cheap"],
                agn_build=None if dec is None else dec["agn_build"],
            )

        shapes   = cache[ck]
        wm_gas   = 10.0 ** p["log10_A_gas"] * shapes["gas"]
        wm_agn   = 10.0 ** p["log10_A_AGN"] * shapes["agn"]
        wmodel   = wm_gas + wm_agn
        theta_as = data["theta_arcsec"]

        ax.errorbar(theta_as, data["wtheta"], yerr=data["wtheta_err"],
                    fmt="o", ms=3, color="k")
        ax.plot(theta_as, wmodel, "-",  lw=1.8, color="C0")
        ax.plot(theta_as, wm_gas, "--", lw=1.2, color="C2", alpha=0.8)
        ax.plot(theta_as, wm_agn, ":",  lw=1.2, color="C1", alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        s = SAMPLES[label]
        ax.set_title(
            f"{label} ($M_*>{s['log10ms_min']}$, $z={s['zmean']:.2f}$)\n"
            f"$\\beta_{{\\rm gas}}={p['beta_gas']:.2f}$, "
            f"$\\log_{{10}}A_{{\\rm gas}}={p['log10_A_gas']:.2f}$, "
            f"$\\chi^2/\\nu={res['chi2_dof']:.2f}$",
            fontsize=7,
        )
        ax.set_xlabel(r"$\theta$ [arcsec]", fontsize=8)
        if ax == axes_flat[0]:
            ax.set_ylabel(r"$w_\theta$", fontsize=9)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Comparat+2025: HAM AGN + King PSF + DPM-2 gas",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview figure → {out_path}")


# --------------------------------------------------------------------------
# Save / load helpers
# --------------------------------------------------------------------------

def _save_map(result: dict, out_dir: Path) -> None:
    label = result["label"]
    out   = out_dir / f"{label}_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "chain"}, f, indent=2)
    print(f"  [{label}] MAP result → {out}")


def _save_chain(mcmc_result: dict, out_dir: Path) -> None:
    label = mcmc_result["label"]
    chain = np.array(mcmc_result["chain"])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import h5py
        path = out_dir / f"{label}_chain.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("chain", data=chain)
            f.attrs["param_names"] = mcmc_result["param_names"]
            for k in ("medians", "lo16", "hi84", "n_walkers", "n_steps", "n_burnin"):
                f.attrs[k] = json.dumps(mcmc_result[k])
        print(f"  [{label}] MCMC chain → {path}")
    except ImportError:
        path = out_dir / f"{label}_chain.npy"
        np.save(path, chain)
        print(f"  [{label}] MCMC chain → {path}  (h5py not available)")

    summary = {k: v for k, v in mcmc_result.items() if k != "chain"}
    with open(out_dir / f"{label}_mcmc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Joint fit of Comparat+2025 w_θ(θ) + wp(rp) with 8 free parameters: "
            "HOD (4), gas amplitude + density slope + pressure slope, AGN HAM amplitude."
        )
    )
    p.add_argument("--sample", nargs="+", default=["S1"],
                   help="Sample labels (e.g. S1 S3) or 'all'")
    p.add_argument("--mode", choices=["map", "mcmc", "both"], default="map",
                   help="Fit mode: map, mcmc, or both (default: map)")
    p.add_argument("--theta-min", type=float, default=8.0,
                   help="Minimum angular scale to fit [arcsec] (default: 8)")
    p.add_argument("--theta-max", type=float, default=300.0,
                   help="Maximum angular scale to fit [arcsec] (default: 300)")
    p.add_argument("--rp-min", type=float, default=0.02,
                   help="Minimum projected radius for wp fit [Mpc/h] (default: 0.02)")
    p.add_argument("--n-walkers", type=int, default=32,
                   help="emcee: number of walkers (default: 32)")
    p.add_argument("--n-steps", type=int, default=1000,
                   help="emcee: production steps (default: 1000)")
    p.add_argument("--n-burnin", type=int, default=300,
                   help="emcee: burn-in steps (default: 300)")
    p.add_argument("--f-sys", type=float, default=0.05,
                   help="Fractional systematic floor for w_θ errors (default: 0.05)")
    p.add_argument("--add-syst", type=float, default=None, metavar="PCT",
                   help="Systematic floor as a percentage, e.g. 5 for 5%%. "
                        "Overrides --f-sys when specified.")
    p.add_argument("--agn-model", choices=["hod", "ham", "xray", "duty_cycle"], default="hod",
                   help="AGN component: 'hod' (HODAgnModel — physically-predicted "
                        "cross-power, log10_A_AGN is a fudge factor; default), "
                        "'ham' (free-amplitude King PSF, HamAGNModel), or "
                        "'xray' (parametric L_X(M*) point source, XrayAGNModel).")
    p.add_argument("--agn-finc", type=float, default=0.01,
                   help="AGN duty cycle f_inc for the HOD AGN model (default: 0.01). "
                        "Only used with --agn-model hod.")
    p.add_argument("--fix-zm15", nargs="?", const=str(_DEFAULT_ZM15_JSON),
                   default=None, metavar="PATH",
                   help="Hold the ZuMandelbaum15 stellar-halo connection fixed at the "
                        "parameters of a wp+n_gal joint-fit MAP json and fit ONLY the "
                        "X-ray/gas/AGN parameters against the w_θ cross-correlation "
                        "(w_θ-only likelihood; wp/SMF become fixed-prediction overlays). "
                        f"Default json: {_DEFAULT_ZM15_JSON}. Pass a path to override.")
    p.add_argument("--free-params", nargs="+", default=["all"], metavar="NAME",
                   help="Free X-ray/gas/AGN parameters in --fix-zm15 mode. A preset "
                        f"({'/'.join(_FREE_PRESETS)}) or an explicit list of registry "
                        "names. Gas tiers (gas-shape/gas-temp/gas-full) free DPM "
                        "profile params and rebuild the gas profiles per likelihood "
                        "eval (slower; activates the full APEC emissivity path). AGN "
                        "tiers: 'agn-models' (compare --agn-model hod|ham|xray, "
                        "amplitude only), 'agn-lum' (+scatter_lx/log10_A_kcorr/"
                        "log10_A_dc, needs --agn-model ham), 'agn-occ' (+f_inc and "
                        "More+2015 AGN-HOD, needs --agn-model hod). Note: in the "
                        "w_θ-only fit, normalisation/AGN-luminosity params are "
                        "degenerate with log10_A_gas/log10_A_AGN. Default: all.")
    p.add_argument("--zm15-thresh", type=float, default=None, metavar="LOG10MS",
                   help="Override the fixed ZM15 log10m_star_thresh in --fix-zm15 mode "
                        "(default: each sample's log10ms_min, e.g. 10.0 for S1).")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation")
    p.add_argument("--plot-only", action="store_true",
                   help="Load saved MAP JSON and regenerate best-fit plot without fitting.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: results/fits/comparat2025)")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete all .npz shape-cache files before fitting.")
    p.add_argument("--hmf", nargs="+", default=["csst"],
                   help="Halo mass function backend(s) (default: csst, the CSSTEMU "
                        "GP emulator used as the pipeline baseline). "
                        "Fitting (--mode map/mcmc) uses only the first value. "
                        "With --plot-only, pass several names (or 'all') to "
                        "overlay one model curve per HMF predictor, evaluated "
                        "at the saved MAP astrophysical parameters. Choices: "
                        f"{list(_BACKENDS) + list(_EMULATOR_BACKENDS)}. "
                        "Note: 'aemulusnu' is only calibrated for M >= 1e13 "
                        "Msun/h and is silently extrapolated below that, so "
                        "it is not recommended for low log10m_star_thresh "
                        "samples such as S1 (see AemulusNuHaloMassFunction "
                        "docstring in hod_mod.core.halo_mass_function).")
    return p.parse_args()


def main():
    args = _parse_args()
    f_sys = args.add_syst / 100.0 if args.add_syst is not None else args.f_sys

    labels = list(SAMPLES.keys()) if "all" in args.sample else args.sample
    for lb in labels:
        if lb not in SAMPLES:
            raise ValueError(f"Unknown sample '{lb}'. Choose from: {list(SAMPLES)}")

    # ---- Fixed-ZM15 mode setup ----
    if args.fix_zm15 is not None:
        global _FIT_CFG
        zm15_path = Path(args.fix_zm15)
        if not zm15_path.exists():
            raise FileNotFoundError(f"--fix-zm15 json not found: {zm15_path}")
        with open(zm15_path) as f:
            zm15_params = json.load(f)["params"]
        free   = _resolve_free_params(args.free_params)
        # Preset/model compatibility (agn-lum needs ham; agn-occ needs hod).
        if len(args.free_params) == 1:
            req = _PRESET_REQUIRES_AGN.get(args.free_params[0])
            if req is not None and args.agn_model != req:
                raise ValueError(
                    f"--free-params {args.free_params[0]} requires --agn-model {req} "
                    f"(got {args.agn_model})."
                )
        thresh = {
            lb: (args.zm15_thresh if args.zm15_thresh is not None
                 else SAMPLES[lb]["log10ms_min"])
            for lb in labels
        }
        _FIT_CFG = {"zm15": zm15_params, "free": free,
                    "fixed_astro": {}, "thresh": thresh}
        print(f"Fixed ZM15 from {zm15_path}\n"
              f"  free params ({len(free)}): {free}\n"
              f"  log10m_star_thresh: {thresh}", flush=True)

    _all_hmf = list(_BACKENDS) + list(_EMULATOR_BACKENDS)
    hmf_list = _all_hmf if "all" in args.hmf else args.hmf
    for m in hmf_list:
        if m not in _all_hmf:
            raise ValueError(f"Unknown HMF backend '{m}'. Choose from: {_all_hmf}")
    if not args.plot_only and len(hmf_list) > 1:
        raise ValueError("Multiple --hmf backends are only supported with --plot-only.")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.fix_zm15 is not None:
        # Separate directory so the existing 9-param results are not clobbered.
        out_dir = _RESULTS_DIR.parent / "comparat2025_fixedZM15"
    else:
        out_dir = _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_cache:
        cache_dir = _RESULTS_DIR / "shape_cache"
        removed = [f for ext in ("*.npz", "*.npy") for f in cache_dir.glob(ext)]
        for f in removed:
            f.unlink()
        if removed:
            print(f"  Cleared {len(removed)} shape-cache files from {cache_dir}")

    if args.plot_only:
        compare_hmf = len(hmf_list) > 1
        if compare_hmf:
            shared = _build_shared_components()
            infra_by_hmf = {}
            for m in hmf_list:
                print(f"  Building halo-model stack for HMF backend '{m}' ...", flush=True)
                try:
                    infra_by_hmf[m] = _build_infra_for_hmf(m, shared)
                except Exception as exc:
                    print(f"  WARNING: HMF backend '{m}' failed to build ({exc}); skipping.")
            if not infra_by_hmf:
                raise RuntimeError("No HMF backend could be built.")
        else:
            infra = _Infrastructure(hmf_backend=hmf_list[0],
                                    agn_model=args.agn_model, agn_finc=args.agn_finc)

        for label in labels:
            json_path = out_dir / f"{label}_map.json"
            if not json_path.exists():
                print(f"  [{label}] No MAP result at {json_path} — run MAP first.")
                continue
            with open(json_path) as f:
                saved = json.load(f)

            # Reconstruct the fixed-ZM15 config from the saved json so _unpack_params /
            # _hod_params can rebuild the full HOD without re-running the MAP fit.
            if _FIT_CFG is not None and saved.get("fixed_zm15"):
                _FIT_CFG["free"]        = saved.get("free_params", _FIT_CFG["free"])
                _FIT_CFG["fixed_astro"] = dict(saved.get("fixed_astro", {}))
                if "zm15_params" in saved:
                    _FIT_CFG["zm15"] = saved["zm15_params"]
                if "log10m_star_thresh" in saved:
                    _FIT_CFG["thresh"][label] = saved["log10m_star_thresh"]

            if compare_hmf:
                plot_bestfit_hmf_compare(
                    label, infra_by_hmf, saved["params"],
                    out_dir / f"{label}_bestfit_hmf_compare.pdf",
                    chi2_dof=saved.get("chi2_dof"),
                    theta_min=args.theta_min, theta_max=args.theta_max,
                    rp_min=args.rp_min, f_sys=f_sys,
                )
                plot_diagnostics_hmf_compare(
                    label, infra_by_hmf, saved["params"],
                    out_dir / f"{label}_diagnostics_hmf_compare.pdf",
                )
            else:
                plot_bestfit(
                    label, infra, saved["params"], {},
                    out_dir / f"{label}_bestfit.pdf",
                    chi2_dof=saved.get("chi2_dof"),
                    theta_min=args.theta_min, theta_max=args.theta_max,
                    rp_min=args.rp_min, f_sys=f_sys,
                )
                plot_diagnostics(
                    label, infra, saved["params"],
                    out_dir / f"{label}_diagnostics.pdf",
                )
                plot_gas_diagnostics(
                    label, infra, saved["params"],
                    out_dir / f"{label}_gas_diagnostics.pdf",
                )
        return

    infra = _Infrastructure(hmf_backend=hmf_list[0],
                            agn_model=args.agn_model, agn_finc=args.agn_finc)

    all_map_results  = []
    all_shape_caches = {}

    for label in labels:
        print(f"\n{'='*60}")
        print(f"  Sample {label}  (log10M*>{SAMPLES[label]['log10ms_min']}, "
              f"z_mean={SAMPLES[label]['zmean']:.3f})")
        print(f"{'='*60}")

        if args.mode in ("map", "both"):
            t0 = time.time()
            map_res, shape_cache = run_map(
                label, infra,
                theta_min=args.theta_min,
                theta_max=args.theta_max,
                rp_min=args.rp_min,
                f_sys=f_sys,
            )
            print(f"  MAP wall-clock: {time.time()-t0:.1f}s")
            _save_map(map_res, out_dir)
            all_map_results.append(map_res)
            all_shape_caches[label] = shape_cache

            if not args.no_plot:
                plot_bestfit(
                    label, infra, map_res["params"], shape_cache,
                    out_dir / f"{label}_bestfit.pdf",
                    chi2_dof=map_res["chi2_dof"],
                    theta_min=args.theta_min, theta_max=args.theta_max,
                    rp_min=args.rp_min, f_sys=f_sys,
                )
                # Full diagnostic figure suite (SMF, n_gal, SHMR, gg-lensing and
                # X-ray auto-power; gas scaling relations + radial profiles).  Each
                # is wrapped so a missing optional dataset cannot abort the others.
                for _plot_fn, _suffix in (
                    (plot_diagnostics,     "diagnostics"),
                    (plot_gas_diagnostics, "gas_diagnostics"),
                ):
                    try:
                        _plot_fn(label, infra, map_res["params"],
                                 out_dir / f"{label}_{_suffix}.pdf")
                    except Exception as _exc:
                        print(f"  [{label}] WARNING: {_suffix} figure failed ({_exc})",
                              flush=True)

        if args.mode in ("mcmc", "both"):
            if args.mode == "mcmc":
                t0 = time.time()
                map_res, shape_cache = run_map(
                    label, infra,
                    theta_min=args.theta_min, theta_max=args.theta_max,
                    rp_min=args.rp_min, f_sys=f_sys,
                )
                print(f"  MAP (seed) wall-clock: {time.time()-t0:.1f}s")
                all_map_results.append(map_res)
                all_shape_caches[label] = shape_cache

            t0 = time.time()
            mcmc_res = run_mcmc(
                label, infra, map_res, shape_cache,
                n_walkers=args.n_walkers,
                n_steps=args.n_steps,
                n_burnin=args.n_burnin,
                theta_min=args.theta_min,
                theta_max=args.theta_max,
                rp_min=args.rp_min,
                f_sys=f_sys,
            )
            print(f"  MCMC wall-clock: {time.time()-t0:.1f}s")
            print(f"  Medians: "
                  f"{dict(zip(mcmc_res['param_names'], np.round(mcmc_res['medians'], 4)))}")
            _save_chain(mcmc_res, out_dir)

            if not args.no_plot:
                plot_corner(label, mcmc_res, out_dir / f"{label}_corner.pdf")
                plot_bestfit(
                    label, infra, mcmc_res["medians"], shape_cache,
                    out_dir / f"{label}_bestfit_mcmc.pdf",
                    theta_min=args.theta_min, theta_max=args.theta_max,
                    rp_min=args.rp_min, f_sys=f_sys,
                )

    if not args.no_plot and all_map_results:
        plot_all_bestfit(
            all_map_results, infra, all_shape_caches,
            out_dir / "all_samples_bestfit.pdf",
        )

    print(f"\n{'='*80}")
    if _FIT_CFG is not None:
        print(f"Fixed-ZM15 fit (w_θ only) — free params: {_FIT_CFG['free']}")
        print(f"{'label':6s}  {'chi2/dof':>9s}  {'n_pts':>6s}  free-param values")
        print(f"{'-'*6}  {'-'*9}  {'-'*6}  {'-'*40}")
        for r in all_map_results:
            pv_str = ", ".join(f"{n}={v:.4f}"
                               for n, v in zip(r["param_names"], r["params"]))
            print(f"{r['label']:6s}  {r['chi2_dof']:>9.3f}  {r['n_pts_fit']:>6d}  {pv_str}")
    else:
        print(f"{'label':6s}  {'log10_A_gas':>11s}  {'beta_gas':>8s}  {'beta_P':>6s}  "
              f"{'log10_A_AGN':>11s}  {'chi2/dof':>9s}  {'n_pts':>6s}")
        print(f"{'-'*6}  {'-'*11}  {'-'*8}  {'-'*6}  {'-'*11}  {'-'*9}  {'-'*6}")
        for r in all_map_results:
            pv = r["params"]
            print(f"{r['label']:6s}  {pv[0]:>11.4f}  {pv[1]:>8.4f}  {pv[2]:>6.4f}  "
                  f"{pv[3]:>11.4f}  {r['chi2_dof']:>9.3f}  {r['n_pts_fit']:>6d}")
    print(f"{'='*80}")
    print(f"Results in: {out_dir}")


if __name__ == "__main__":
    main()
