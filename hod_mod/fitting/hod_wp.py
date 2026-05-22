"""HOD fitting to projected correlation function wp(rp).

Supports MAP estimation (scipy.optimize.minimize) and ensemble MCMC (emcee).
All configuration is read from an external YAML file so that different datasets
and HOD models can be fitted without modifying source code.

Typical usage::

    from hod_mod.fitting import load_config, WpFitter

    config  = load_config("configs/hod_fit_more2015_cmass.yml")
    fitter  = WpFitter(config)
    result  = fitter.map_fit()
    sampler = fitter.sample(initial_pos=result["theta"])

Input data formats
------------------
Two formats are supported for the ``data.format`` YAML key:

``csv`` (default, backward-compatible)
    Plain-text file with columns ``rp_hMpc``, ``wp_hMpc``, ``wp_err_hMpc``.
    Covariance is assumed diagonal.

``hdf5``
    HDF5 file produced by the ``sum_stat`` package.  The file is read by
    :class:`~hod_mod.data_io.sum_stat_reader.SumStatReader`; h-unit
    conversion is applied automatically using the embedded Hubble constant.
    The full covariance matrix from the file is used instead of a diagonal
    approximation.

``fits``
    FITS JK jackknife realisations; full covariance matrix is assembled
    automatically.  Set ``data.format: fits`` and ``fits.jk_dir``.

Cosmological priors
-------------------
Each free parameter may carry an optional ``prior_type`` field:

``uniform`` (default)
    Flat prior; ``bounds`` define hard [lo, hi] limits.

``gaussian``
    Gaussian prior; requires ``prior_mean`` and ``prior_sigma``.
    ``bounds`` still define hard clipping limits.

References
----------
More et al. 2015, ApJ 806, 2 (arXiv:1407.1856) — BOSS CMASS HOD constraints
Planck Collaboration 2020, A&A 641, A6 (arXiv:1807.06209) — Planck 2018 cosmology
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.hod import (
    HODModel,
    Kravtsov04HODModel,
    MoreHODModel,
    Guo18ICSMFModel,
    Guo19ICSMFModel,
    Zacharegkas25HODModel,
    VanUitert16CSMFModel,
    ZuMandelbaum15HODModel,
    Leauthaud12HODModel,
    Lange25HODModel,
)
from hod_mod.galaxies.clf import CLFModel

HOD_MODELS: dict = {
    "HODModel":                  HODModel,
    "Kravtsov04HODModel":        Kravtsov04HODModel,
    "MoreHODModel":              MoreHODModel,
    "Guo18ICSMFModel":           Guo18ICSMFModel,
    "Guo19ICSMFModel":           Guo19ICSMFModel,
    "Zacharegkas25HODModel":     Zacharegkas25HODModel,
    "VanUitert16CSMFModel":      VanUitert16CSMFModel,
    "ZuMandelbaum15HODModel":    ZuMandelbaum15HODModel,
    "Leauthaud12HODModel":       Leauthaud12HODModel,
    "Lange25HODModel":           Lange25HODModel,
    "CLFModel":                  CLFModel,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """Unified fitting configuration for wp, joint (wp+ΔΣ+n_g), and FITS JK fits.

    All paths are relative to the repository root (the directory containing
    the YAML file's parent ``configs/`` folder).

    For joint fits, set ``ds_file`` to a non-empty path.
    For FITS JK data, set ``data_format="fits"`` and ``jk_dir``.

    Parameters
    ----------
    data_format : str
        ``'csv'``, ``'hdf5'``, or ``'fits'``.
    param_prior_types : dict
        ``{name: 'uniform' | 'gaussian'}`` for each free parameter.
    ds_file : str or None
        Path to ΔΣ CSV file for joint fits.  *None* means wp-only.
    jk_dir : str or None
        Directory of FITS JK realisations.  *None* means no FITS loading.
    """
    data_file:          str
    rp_min:             float
    rp_max:             float
    hod_model:          str
    hmf_backend:        str
    z:                  float
    pi_max:             float
    free_params:        list
    param_bounds:       dict
    param_init:         dict
    method:             str   = "both"
    n_walkers:          int   = 32
    n_steps:            int   = 2000
    n_burnin:           int   = 500
    output_dir:         str   = "results/"
    repo_root:          str   = ""
    data_format:        str   = "csv"
    param_prior_types:  dict  = field(default_factory=dict)
    param_prior_means:  dict  = field(default_factory=dict)
    param_prior_sigmas: dict  = field(default_factory=dict)
    # Joint fit fields (None = wp-only)
    ds_file:            str | None  = None
    ds_rp_min:          float       = 0.1
    ds_rp_max:          float       = 20.0
    ng_obs:             float       = 3.0e-4
    ng_frac_err:        float       = 0.20
    # FITS JK fields (None = no FITS loading)
    jk_dir:             str | None  = None
    jk_pattern:         str         = "NSIDE_04"
    h_hubble:           float       = 0.6736
    cosmology:          dict | None = None   # None → use pk_lin.default_cosmology()
    use_free_cosmo:     bool        = False  # free Ω_m + S8 with Planck priors


# Backward-compatibility aliases
WpFitConfig    = FitConfig
JointFitConfig = FitConfig
WpFitFITSConfig = FitConfig


def _parse_params(params_cfg: dict) -> tuple[list, dict, dict, dict, dict, dict]:
    """Parse the ``parameters:`` section of a YAML config."""
    free_params        = []
    param_bounds       = {}
    param_init         = {}
    param_prior_types  = {}
    param_prior_means  = {}
    param_prior_sigmas = {}
    for name, spec in params_cfg.items():
        param_init[name] = float(spec["init"])
        if spec.get("free", False):
            free_params.append(name)
            lo, hi = spec["bounds"]
            param_bounds[name] = (float(lo), float(hi))
            prior_type = spec.get("prior_type", "uniform")
            param_prior_types[name] = prior_type
            if prior_type == "gaussian":
                param_prior_means[name]  = float(spec["prior_mean"])
                param_prior_sigmas[name] = float(spec["prior_sigma"])
    return (free_params, param_bounds, param_init,
            param_prior_types, param_prior_means, param_prior_sigmas)


def _sigma8_to_lnAs(cosmo: dict) -> float:
    """Return ln(10^{10} A_s) that reproduces cosmo['sigma8'] at z=0 via CAMB."""
    import camb
    h  = cosmo["h"]
    ob = cosmo["Omega_b"]
    oc = cosmo.get("Omega_cdm", cosmo["Omega_m"] - ob)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100.0 * h, ombh2=ob * h**2, omch2=oc * h**2)
    pars.InitPower.set_params(ns=cosmo["n_s"], As=1e-10)
    pars.set_matter_power(redshifts=[0.0], kmax=200.0)
    res = camb.get_results(pars)
    sigma8_unit = float(res.get_sigma8()[-1])
    As_target = (cosmo["sigma8"] / sigma8_unit) ** 2 * 1e-10
    import numpy as _np
    return float(_np.log(As_target * 1e10))


def load_config(path: str) -> FitConfig:
    """Load and validate a YAML fitting config (all variants).

    Handles wp-only, joint (wp+ΔΣ+n_g), and FITS JK data by detecting which
    optional YAML sections (``joint:``, ``fits:``) are present.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    FitConfig
    """
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    # Walk up from the config file until we find the directory that has
    # both 'configs/' and 'data/' as direct children (= repo root).
    repo_root = os.path.dirname(os.path.abspath(path))
    while not (os.path.isdir(os.path.join(repo_root, "configs")) and
               os.path.isdir(os.path.join(repo_root, "data"))):
        parent = os.path.dirname(repo_root)
        if parent == repo_root:  # reached filesystem root
            break
        repo_root = parent

    data_cfg   = raw.get("data", {})
    model_cfg  = raw["model"]
    fit_cfg    = raw.get("fitting", {})
    out_cfg    = raw.get("output", {})
    params_cfg = raw["parameters"]
    joint_cfg  = raw.get("joint", {})
    ds_only_cfg = raw.get("ds", {})   # DS-only section (no wp data required)
    fits_cfg   = raw.get("fits", {})
    cosmo_cfg  = raw.get("cosmology", None)
    cosmology  = {k: float(v) for k, v in cosmo_cfg.items()} if cosmo_cfg else None
    if cosmology is not None:
        if "Omega_cdm" not in cosmology and "Omega_m" in cosmology:
            cosmology["Omega_cdm"] = cosmology["Omega_m"] - cosmology["Omega_b"]
        if "sigma8" in cosmology and "ln10^{10}A_s" not in cosmology:
            cosmology["ln10^{10}A_s"] = _sigma8_to_lnAs(cosmology)

    (free_params, param_bounds, param_init,
     param_prior_types, param_prior_means, param_prior_sigmas) = _parse_params(params_cfg)

    data_fmt = data_cfg.get("format", "csv")
    # FITS section overrides data_format
    if fits_cfg:
        data_fmt = "fits"

    # Resolve data_file
    if fits_cfg:
        jk_dir_rel = fits_cfg.get("jk_dir", data_cfg.get("file", ""))
        jk_dir_abs = os.path.join(repo_root, jk_dir_rel)
        data_file  = jk_dir_abs
    else:
        data_file = os.path.join(repo_root, data_cfg.get("file", "")) if data_cfg.get("file") else ""

    # DS-file: from ``joint:`` section (joint fit) or ``ds:`` section (DS-only)
    ds_file = None
    if joint_cfg:
        ds_file_rel = joint_cfg.get("ds_file", "")
        ds_file = os.path.join(repo_root, ds_file_rel) if ds_file_rel else None
        ds_rp_min   = float(joint_cfg.get("ds_rp_min", 0.1))
        ds_rp_max   = float(joint_cfg.get("ds_rp_max", 20.0))
        ng_obs      = float(joint_cfg.get("ng_obs", 3.0e-4))
        ng_frac_err = float(joint_cfg.get("ng_frac_err", 0.20))
    elif ds_only_cfg:
        ds_file_rel = ds_only_cfg.get("file", "")
        ds_file = os.path.join(repo_root, ds_file_rel) if ds_file_rel else None
        ds_rp_min   = float(ds_only_cfg.get("rp_min", 0.1))
        ds_rp_max   = float(ds_only_cfg.get("rp_max", 20.0))
        ng_obs      = float(ds_only_cfg.get("ng_obs", 3.0e-4))
        ng_frac_err = float(ds_only_cfg.get("ng_frac_err", 0.20))
    else:
        ds_rp_min   = 0.1
        ds_rp_max   = 20.0
        ng_obs      = 3.0e-4
        ng_frac_err = 0.20

    return FitConfig(
        data_file          = data_file,
        rp_min             = float(data_cfg.get("rp_min", 0.1)),
        rp_max             = float(data_cfg.get("rp_max", 60.0)),
        hod_model          = model_cfg["hod_model"],
        hmf_backend        = model_cfg.get("hmf_backend", "tinker08"),
        z                  = float(model_cfg["z"]),
        pi_max             = float(model_cfg["pi_max"]),
        free_params        = free_params,
        param_bounds       = param_bounds,
        param_init         = param_init,
        method             = fit_cfg.get("method", "both"),
        n_walkers          = int(fit_cfg.get("n_walkers", 32)),
        n_steps            = int(fit_cfg.get("n_steps", 2000)),
        n_burnin           = int(fit_cfg.get("n_burnin", 500)),
        output_dir         = os.path.join(repo_root, out_cfg.get("dir", "results/")),
        repo_root          = repo_root,
        data_format        = data_fmt,
        param_prior_types  = param_prior_types,
        param_prior_means  = param_prior_means,
        param_prior_sigmas = param_prior_sigmas,
        ds_file            = ds_file,
        ds_rp_min          = ds_rp_min,
        ds_rp_max          = ds_rp_max,
        ng_obs             = ng_obs,
        ng_frac_err        = ng_frac_err,
        jk_dir             = data_file if fits_cfg else None,
        jk_pattern         = fits_cfg.get("jk_pattern", "NSIDE_04"),
        h_hubble           = float(fits_cfg.get("h", 0.6736)),
        cosmology          = cosmology,
        use_free_cosmo     = bool(fit_cfg.get("use_free_cosmo", False)),
    )


# Backward-compatibility aliases
load_joint_config = load_config
load_fits_config  = load_config


# ---------------------------------------------------------------------------
# Log-probability helpers (standalone, callable by emcee and scipy)
# ---------------------------------------------------------------------------

def _assemble_hod_params(theta_vec, free_params, fixed_params):
    """Combine a free-parameter vector with fixed-parameter dict."""
    p = dict(fixed_params)
    for name, val in zip(free_params, theta_vec):
        p[name] = float(val)
    return p


def log_prob_wp(
    theta_vec,
    free_params: list,
    fixed_params: dict,
    param_bounds: dict,
    predictor: FullHaloModelPrediction,
    rp_arr: np.ndarray,
    wp_obs: np.ndarray,
    icov_wp: np.ndarray,
    z: float,
    theta_cosmo: dict,
    pi_max: float,
) -> float:
    """Log-posterior for wp(rp) fitting.

    log P(theta | data) = -0.5 chi^2  (uniform prior, 0 inside bounds, -inf outside)

    Parameters
    ----------
    theta_vec : array_like, shape (n_free,)
    free_params : list of str
    fixed_params : dict
    param_bounds : dict — {name: (lo, hi)}
    predictor : FullHaloModelPrediction
    rp_arr, wp_obs, icov_wp : projected data arrays and inverse covariance
    z : float — effective redshift
    theta_cosmo : dict — cosmological parameters
    pi_max : float — l.o.s. integration limit [Mpc/h]
    """
    for name, val in zip(free_params, theta_vec):
        lo, hi = param_bounds[name]
        if not (lo <= val <= hi):
            return -np.inf
    hod_params = _assemble_hod_params(theta_vec, free_params, fixed_params)
    try:
        wp_pred = np.asarray(
            predictor.wp(jnp.array(rp_arr), pi_max, z, theta_cosmo, hod_params)
        )
    except Exception:
        return -np.inf
    residual = wp_pred - wp_obs
    return -0.5 * float(residual @ icov_wp @ residual)


def log_prob_joint(
    theta_vec,
    free_params: list,
    fixed_params: dict,
    param_bounds: dict,
    predictor: FullHaloModelPrediction,
    rp_arr: np.ndarray,
    wp_obs: np.ndarray,
    icov_wp: np.ndarray,
    R_arr: np.ndarray,
    ds_obs: np.ndarray,
    icov_ds: np.ndarray,
    ng_obs: float,
    ng_frac_err: float,
    z: float,
    theta_cosmo: dict,
    pi_max: float,
) -> float:
    r"""Log-posterior for joint wp + ΔΣ + n_g fitting.

    .. math::

        \log P(\theta|d) = -\frac{1}{2}\bigl[
          \chi^2_{w_p} + \chi^2_{\Delta\Sigma} + \chi^2_{n_g}
        \bigr]
    """
    for name, val in zip(free_params, theta_vec):
        lo, hi = param_bounds[name]
        if not (lo <= val <= hi):
            return -np.inf
    hod_params = _assemble_hod_params(theta_vec, free_params, fixed_params)
    try:
        wp_pred = np.asarray(
            predictor.wp(jnp.array(rp_arr), pi_max, z, theta_cosmo, hod_params)
        )
        ds_pred = np.asarray(
            predictor.delta_sigma(jnp.array(R_arr), z, theta_cosmo, hod_params)
        )
        ng_pred = predictor.n_gal(z, theta_cosmo, hod_params)
    except Exception:
        return -np.inf
    chi2_wp = float((wp_pred - wp_obs) @ icov_wp @ (wp_pred - wp_obs))
    chi2_ds = float((ds_pred - ds_obs) @ icov_ds @ (ds_pred - ds_obs))
    chi2_ng = float(((ng_pred - ng_obs) / (ng_frac_err * ng_obs)) ** 2)
    return -0.5 * (chi2_wp + chi2_ds + chi2_ng)


# ---------------------------------------------------------------------------
# Free-cosmology helpers
# ---------------------------------------------------------------------------


class _CachedPkLinear:
    """Interpolation cache around LinearPowerSpectrum for free-cosmo MCMC.

    On the first call for a given (z, Ω_m, ln10As, h) key a reference P(k)
    is computed on a fixed k grid via CAMB and log-log interpolated on all
    subsequent calls.  This reduces per-sample cost from ~30 s to <1 ms once
    the cache warms up.
    """

    def __init__(self, pk_lin_obj, n_k: int = 512):
        self._base      = pk_lin_obj
        self._k_ref     = np.logspace(-4, 1.5, n_k)
        self._log_k_ref = np.log(self._k_ref)
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
# WpFitter
# ---------------------------------------------------------------------------

class WpFitter:
    """Fit an HOD model to wp(rp) data.

    Supports ``data_format = "csv"``, ``"hdf5"``, or ``"fits"`` (FITS JK).
    For a joint wp+ΔΣ+n_g fit, use :class:`JointFitter` with a config that
    has ``ds_file`` set.  For a ΔΣ-only fit, use :class:`DeltaSigmaFitter`.

    Parameters
    ----------
    config : FitConfig
        Parsed fitting configuration (from :func:`load_config`).
    """

    def __init__(self, config: FitConfig):
        self._setup_common(config)
        self._load_data()
        self._build_predictor()
        self._build_icov()

    # ------------------------------------------------------------------
    # Common setup (reused by DeltaSigmaFitter)

    def _setup_common(self, config: FitConfig):
        self.config      = config
        self._pk_lin     = LinearPowerSpectrum()
        self.theta_cosmo = (
            config.cosmology if config.cosmology is not None
            else self._pk_lin.default_cosmology()
        )
        self._cov_wp     = None
        if config.use_free_cosmo:
            self._pk_lin = _CachedPkLinear(self._pk_lin)

    def _theta_cosmo_call(self, all_params: dict) -> dict:
        """Build a per-call theta_cosmo when use_free_cosmo is enabled.

        Extracts Omega_m and S8 from all_params, derives sigma8 and ln10As,
        and returns an updated cosmology dict for the predictor call.
        When use_free_cosmo is False, returns self.theta_cosmo unchanged.
        """
        if not self.config.use_free_cosmo:
            return self.theta_cosmo
        tc = dict(self.theta_cosmo)
        if "Omega_m" in all_params:
            Omega_m = float(all_params["Omega_m"])
            tc["Omega_m"]   = Omega_m
            tc["Omega_cdm"] = Omega_m - float(tc["Omega_b"])
        if "S8" in all_params:
            S8      = float(all_params["S8"])
            Omega_m = tc["Omega_m"]
            sigma8  = S8 * np.sqrt(0.3 / Omega_m)
            sigma8_fid = float(self.theta_cosmo.get("sigma8", 0.8111))
            ln10As_fid = float(self.theta_cosmo["ln10^{10}A_s"])
            # Fast update: sigma8 ∝ sqrt(As) → ln10As += 2*ln(sigma8_new/sigma8_fid)
            tc["ln10^{10}A_s"] = ln10As_fid + 2.0 * np.log(sigma8 / sigma8_fid)
        return tc

    # ------------------------------------------------------------------
    # Setup helpers

    def _load_data(self):
        fmt = self.config.data_format
        if fmt == "hdf5":
            self._load_data_hdf5()
        elif fmt == "fits":
            self._load_data_fits()
        else:
            self._load_data_csv()

    def _load_data_csv(self):
        import pandas as pd
        data = pd.read_csv(self.config.data_file, comment="#")
        mask = (data["rp_hMpc"] >= self.config.rp_min) & (data["rp_hMpc"] <= self.config.rp_max)
        self.rp_arr = data["rp_hMpc"][mask].to_numpy()
        self.wp_obs = data["wp_hMpc"][mask].to_numpy()
        self.wp_err = data["wp_err_hMpc"][mask].to_numpy()
        self._cov_wp = None

    def _load_data_hdf5(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        reader = SumStatReader.from_hdf5(self.config.data_file)
        d    = reader.wp()
        rp   = d["rp"]
        wp   = d["wp"]
        cov  = d["cov"]
        mask = (rp >= self.config.rp_min) & (rp <= self.config.rp_max)
        self.rp_arr  = rp[mask]
        self.wp_obs  = wp[mask]
        self.wp_err  = np.sqrt(np.diag(cov)[mask])
        self._cov_wp = cov[np.ix_(mask, mask)]

    def _load_data_fits(self):
        from hod_mod.data_io.wprp_fits import load_jk_wp_auto
        cfg = self.config
        rp_h, wp_h, cov_h = load_jk_wp_auto(
            directory=cfg.jk_dir,
            pattern=cfg.jk_pattern,
            h=cfg.h_hubble,
            rp_min=cfg.rp_min,
            rp_max=cfg.rp_max,
        )
        self.rp_arr  = rp_h
        self.wp_obs  = wp_h
        self.wp_err  = np.sqrt(np.diag(cov_h))
        self._cov_wp = cov_h

    def _build_predictor(self):
        hmf     = make_hmf(self.config.hmf_backend, pk_func=self._pk_lin.pk_linear)
        hod_cls = HOD_MODELS[self.config.hod_model]
        hod     = hod_cls(hmf) if hod_cls._SINGLE_ARG_INIT else hod_cls(hmf, hmf.bias)
        self.predictor = FullHaloModelPrediction(self._pk_lin, hod, HaloProfile(self.theta_cosmo))

    def _build_icov(self):
        if self._cov_wp is not None:
            cov = self._cov_wp
            reg = 0.01 * np.diag(np.diag(cov))
            self.icov_wp = np.linalg.inv(cov + reg)
        else:
            self.icov_wp = np.diag(1.0 / self.wp_err**2)

    # ------------------------------------------------------------------
    # Fixed / free partition

    @property
    def _fixed_params(self) -> dict:
        return {k: v for k, v in self.config.param_init.items()
                if k not in self.config.free_params}

    @property
    def _x0(self) -> np.ndarray:
        return np.array([self.config.param_init[p] for p in self.config.free_params])

    # ------------------------------------------------------------------
    # Log-probability

    def _prior_log_prob(self, theta_vec) -> float:
        """Log-prior: flat (uniform) or Gaussian, per parameter."""
        from hod_mod.fitting.planck_prior import gaussian_log_prior
        log_pi = 0.0
        for name, val in zip(self.config.free_params, theta_vec):
            lo, hi = self.config.param_bounds[name]
            if not (lo <= val <= hi):
                return -np.inf
            ptype = self.config.param_prior_types.get(name, "uniform")
            if ptype == "gaussian":
                mu    = self.config.param_prior_means[name]
                sigma = self.config.param_prior_sigmas[name]
                log_pi += gaussian_log_prior(val, mu, sigma, lo, hi)
        return log_pi

    def _log_prob(self, theta_vec) -> float:
        log_pi = self._prior_log_prob(theta_vec)
        if not np.isfinite(log_pi):
            return -np.inf
        hod_params  = _assemble_hod_params(
            theta_vec, self.config.free_params, self._fixed_params
        )
        theta_cosmo = self._theta_cosmo_call(hod_params)
        try:
            wp_pred = np.asarray(
                self.predictor.wp(
                    jnp.array(self.rp_arr), self.config.pi_max,
                    self.config.z, theta_cosmo, hod_params,
                )
            )
        except Exception:
            return -np.inf
        residual = wp_pred - self.wp_obs
        return log_pi - 0.5 * float(residual @ self.icov_wp @ residual)

    # ------------------------------------------------------------------
    # MAP estimation

    def map_fit(self) -> dict:
        """Maximum a-posteriori fit via Nelder-Mead.

        Returns
        -------
        dict
            Keys: ``theta``, ``params``, ``chi2``, ``ndof``, ``success``, ``message``.
        """
        from scipy.optimize import minimize
        result = minimize(
            lambda x: -self._log_prob(x),
            self._x0,
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-4, "fatol": 1e-4, "disp": False},
        )
        best_theta  = result.x
        best_params = _assemble_hod_params(best_theta, self.config.free_params, self._fixed_params)
        return {
            "theta":   best_theta,
            "params":  best_params,
            "chi2":    float(-2.0 * self._log_prob(best_theta)),
            "ndof":    len(self.rp_arr) - len(self.config.free_params),
            "success": result.success,
            "message": result.message,
        }

    # ------------------------------------------------------------------
    # Ensemble MCMC sampling

    def sample(self, initial_pos: np.ndarray | None = None, progress: bool = True):
        """Run emcee ensemble sampler.

        Parameters
        ----------
        initial_pos : array_like, shape (n_walkers, n_free), optional
        progress : bool — show tqdm progress bar

        Returns
        -------
        emcee.EnsembleSampler
        """
        import emcee
        n_free    = len(self.config.free_params)
        n_walkers = self.config.n_walkers
        if initial_pos is None:
            initial_pos = self._default_initial_pos(n_walkers, n_free)
        sampler = emcee.EnsembleSampler(n_walkers, n_free, self._log_prob)
        print(f"Burning in: {self.config.n_burnin} steps, {n_walkers} walkers …")
        sampler.run_mcmc(initial_pos, self.config.n_burnin, progress=progress)
        last_pos = sampler.get_last_sample()
        sampler.reset()
        print(f"Sampling: {self.config.n_steps} steps …")
        sampler.run_mcmc(last_pos, self.config.n_steps, progress=progress)
        os.makedirs(self.config.output_dir, exist_ok=True)
        out_path = os.path.join(self.config.output_dir, "flatchain.npz")
        np.savez(out_path, flatchain=sampler.get_chain(flat=True),
                 param_names=np.array(self.config.free_params))
        print(f"Chain saved → {out_path}")
        return sampler

    def _default_initial_pos(self, n_walkers: int, n_free: int) -> np.ndarray:
        x0  = self._x0
        pos = np.zeros((n_walkers, n_free))
        for i, name in enumerate(self.config.free_params):
            lo, hi  = self.config.param_bounds[name]
            width   = 0.05 * (hi - lo)
            pos[:, i] = np.clip(x0[i] + np.random.randn(n_walkers) * width, lo, hi)
        return pos

    # ------------------------------------------------------------------
    # Predictions

    def predict_wp(self, params: dict) -> np.ndarray:
        """Predicted wp(rp) [Mpc/h]."""
        return np.asarray(
            self.predictor.wp(
                jnp.array(self.rp_arr), self.config.pi_max,
                self.config.z, self.theta_cosmo, params,
            )
        )

    # ------------------------------------------------------------------
    # DS data loading (shared with JointFitter and DeltaSigmaFitter)

    def _load_ds_data(self):
        import pandas as pd
        data = pd.read_csv(self.config.ds_file, comment="#")
        mask = (
            (data["R_hMpc"] >= self.config.ds_rp_min)
            & (data["R_hMpc"] <= self.config.ds_rp_max)
        )
        self.R_arr  = data["R_hMpc"][mask].to_numpy()
        self.ds_obs = data["ds_Msun_h_pc2"][mask].to_numpy()
        self.ds_err = data["ds_err_Msun_h_pc2"][mask].to_numpy()

    def _build_icov_ds(self):
        self.icov_ds = np.diag(1.0 / self.ds_err**2)

    def chi2(self, params: dict) -> float:
        """χ² = (wp_pred - wp_obs)ᵀ icov (wp_pred - wp_obs)."""
        res = self.predict_wp(params) - self.wp_obs
        return float(res @ self.icov_wp @ res)


# ---------------------------------------------------------------------------
# JointFitter — wp + ΔΣ + n_g
# ---------------------------------------------------------------------------

class JointFitter(WpFitter):
    """Fit an HOD model to wp(rp) + ΔΣ(R) + n_g simultaneously.

    Implements the joint likelihood of More+2015 §3.1.

    Parameters
    ----------
    config : FitConfig
        Must have ``ds_file`` set to a non-empty path.
    """

    def __init__(self, config: FitConfig):
        super().__init__(config)
        self._load_ds_data()
        self._build_icov_ds()

    def _log_prob(self, theta_vec) -> float:
        log_pi = self._prior_log_prob(theta_vec)
        if not np.isfinite(log_pi):
            return -np.inf
        hod_params  = _assemble_hod_params(
            theta_vec, self.config.free_params, self._fixed_params
        )
        theta_cosmo = self._theta_cosmo_call(hod_params)
        try:
            wp_pred = np.asarray(
                self.predictor.wp(jnp.array(self.rp_arr), self.config.pi_max,
                                  self.config.z, theta_cosmo, hod_params)
            )
            ds_pred = np.asarray(
                self.predictor.delta_sigma(jnp.array(self.R_arr), self.config.z,
                                           theta_cosmo, hod_params)
            )
            ng_pred = self.predictor.n_gal(self.config.z, theta_cosmo, hod_params)
        except Exception:
            return -np.inf
        chi2_wp = float((wp_pred - self.wp_obs) @ self.icov_wp @ (wp_pred - self.wp_obs))
        chi2_ds = float((ds_pred - self.ds_obs) @ self.icov_ds @ (ds_pred - self.ds_obs))
        chi2_ng = float(
            ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
        )
        return log_pi - 0.5 * (chi2_wp + chi2_ds + chi2_ng)

    def predict_ds(self, params: dict) -> np.ndarray:
        """Predicted ΔΣ(R) [M_sun h pc⁻²]."""
        return np.asarray(
            self.predictor.delta_sigma(
                jnp.array(self.R_arr), self.config.z, self.theta_cosmo, params,
            )
        )

    def predict_ng(self, params: dict) -> float:
        """Predicted galaxy number density n̄_g [h³ Mpc⁻³]."""
        return self.predictor.n_gal(self.config.z, self.theta_cosmo, params)

    def chi2_joint(self, params: dict) -> dict:
        """Per-observable χ² contributions and total."""
        wp_pred = self.predict_wp(params)
        ds_pred = self.predict_ds(params)
        ng_pred = self.predict_ng(params)
        r_wp = wp_pred - self.wp_obs
        r_ds = ds_pred - self.ds_obs
        c_wp = float(r_wp @ self.icov_wp @ r_wp)
        c_ds = float(r_ds @ self.icov_ds @ r_ds)
        c_ng = float(
            ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
        )
        return {"chi2_wp": c_wp, "chi2_ds": c_ds, "chi2_ng": c_ng, "chi2_total": c_wp + c_ds + c_ng}


# ---------------------------------------------------------------------------
# DeltaSigmaFitter — ΔΣ(R) only + optional n_g constraint
# ---------------------------------------------------------------------------

class DeltaSigmaFitter(WpFitter):
    """Fit an HOD model to ΔΣ(R) data only (no wp constraint).

    Uses the galaxy–matter cross-correlation ΔΣ(R) and optionally the
    galaxy number density n_g as constraints.  The log-posterior is::

        log P(θ|d) = −½ (χ²_ΔΣ + χ²_n_g)

    Config requirements:
    - ``ds_file`` must point to a valid CSV with columns
      ``R_hMpc``, ``ds_Msun_h_pc2``, ``ds_err_Msun_h_pc2``.
    - ``ng_obs`` and ``ng_frac_err`` control the n_g constraint weight
      (defaults from :class:`FitConfig` apply if not set in YAML).
    - No ``data_file`` / ``data.file`` is required.

    Parameters
    ----------
    config : FitConfig
        Must have ``ds_file`` set.
    """

    def __init__(self, config: FitConfig):
        self._setup_common(config)
        self._build_predictor()
        self._load_ds_data()
        self._build_icov_ds()

    def _log_prob(self, theta_vec) -> float:
        log_pi = self._prior_log_prob(theta_vec)
        if not np.isfinite(log_pi):
            return -np.inf
        hod_params  = _assemble_hod_params(
            theta_vec, self.config.free_params, self._fixed_params
        )
        theta_cosmo = self._theta_cosmo_call(hod_params)
        try:
            ds_pred = np.asarray(
                self.predictor.delta_sigma(jnp.array(self.R_arr), self.config.z,
                                           theta_cosmo, hod_params)
            )
            ng_pred = self.predictor.n_gal(self.config.z, theta_cosmo, hod_params)
        except Exception:
            return -np.inf
        chi2_ds = float((ds_pred - self.ds_obs) @ self.icov_ds @ (ds_pred - self.ds_obs))
        chi2_ng = float(
            ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
        )
        return log_pi - 0.5 * (chi2_ds + chi2_ng)

    def map_fit(self) -> dict:
        """Maximum a-posteriori fit via Nelder-Mead (ΔΣ-only).

        Returns
        -------
        dict
            Keys: ``theta``, ``params``, ``chi2``, ``ndof``, ``success``, ``message``.
        """
        from scipy.optimize import minimize
        result = minimize(
            lambda x: -self._log_prob(x),
            self._x0,
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-4, "fatol": 1e-4, "disp": False},
        )
        best_theta  = result.x
        best_params = _assemble_hod_params(best_theta, self.config.free_params, self._fixed_params)
        return {
            "theta":   best_theta,
            "params":  best_params,
            "chi2":    float(-2.0 * self._log_prob(best_theta)),
            "ndof":    len(self.R_arr) - len(self.config.free_params),
            "success": result.success,
            "message": result.message,
        }

    def predict_ds(self, params: dict) -> np.ndarray:
        """Predicted ΔΣ(R) [M_sun h pc⁻²]."""
        return np.asarray(
            self.predictor.delta_sigma(
                jnp.array(self.R_arr), self.config.z, self.theta_cosmo, params,
            )
        )

    def chi2(self, params: dict) -> float:
        """χ² for ΔΣ-only (excludes n_g term)."""
        res = self.predict_ds(params) - self.ds_obs
        return float(res @ self.icov_ds @ res)


# Backward-compatibility alias (WpFitterFITS was eliminated; use WpFitter with data_format="fits")
WpFitterFITS = WpFitter
