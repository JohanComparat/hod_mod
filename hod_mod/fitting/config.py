"""Fit configuration dataclass, YAML loaders and data-vector readers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from hod_mod.paths import repo_root as _repo_root, results_root




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
    ds_format:          str         = "csv"   # "csv" or "bwpd"
    fit_ng:             bool        = False   # include chi2_ng in likelihood
    # FITS JK fields (None = no FITS loading)
    jk_dir:             str | None  = None
    jk_pattern:         str         = "NSIDE_04"
    h_hubble:           float       = 0.6736
    cosmology:          dict | None = None   # None → use pk_lin.default_cosmology()
    use_free_cosmo:     bool        = False  # free Ω_m + S8 with Planck priors
    use_bnl:            bool        = True   # beyond-linear halo bias (Mead & Verde 2021)


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

    # Code repository root: $HOD_MOD_REPO, else auto-detected from the package
    # location. Config files reference data/ relative to it.
    repo_root = str(_repo_root())

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
    fit_ng  = False  # default: never constrain n_g
    if joint_cfg:
        ds_file_rel = joint_cfg.get("ds_file", "")
        ds_file = os.path.join(repo_root, ds_file_rel) if ds_file_rel else None
        ds_rp_min   = float(joint_cfg.get("ds_rp_min", 0.1))
        ds_rp_max   = float(joint_cfg.get("ds_rp_max", 20.0))
        ng_obs      = float(joint_cfg.get("ng_obs", 3.0e-4))
        ng_frac_err = float(joint_cfg.get("ng_frac_err", 0.20))
        ds_fmt      = joint_cfg.get("ds_format", "csv")
        fit_ng      = bool(joint_cfg.get("fit_ng", False))
    elif ds_only_cfg:
        ds_file_rel = ds_only_cfg.get("file", "")
        ds_file = os.path.join(repo_root, ds_file_rel) if ds_file_rel else None
        ds_rp_min   = float(ds_only_cfg.get("rp_min", 0.1))
        ds_rp_max   = float(ds_only_cfg.get("rp_max", 20.0))
        ng_obs      = float(ds_only_cfg.get("ng_obs", 3.0e-4))
        ng_frac_err = float(ds_only_cfg.get("ng_frac_err", 0.20))
        ds_fmt      = ds_only_cfg.get("ds_format", "csv")
        fit_ng      = bool(ds_only_cfg.get("fit_ng", False))  # ESD-only default: no n_g
    else:
        ds_rp_min   = 0.1
        ds_rp_max   = 20.0
        ng_obs      = 3.0e-4
        ng_frac_err = 0.20
        ds_fmt      = "csv"

    return FitConfig(
        data_file          = data_file,
        rp_min             = float(data_cfg.get("rp_min", 0.1)),
        rp_max             = float(data_cfg.get("rp_max", 60.0)),
        hod_model          = model_cfg["hod_model"],
        hmf_backend        = model_cfg.get("hmf_backend", "csst"),
        z                  = float(model_cfg["z"]),
        pi_max             = float(model_cfg["pi_max"]),
        free_params        = free_params,
        param_bounds       = param_bounds,
        param_init         = param_init,
        method             = fit_cfg.get("method", "both"),
        n_walkers          = int(fit_cfg.get("n_walkers", 32)),
        n_steps            = int(fit_cfg.get("n_steps", 2000)),
        n_burnin           = int(fit_cfg.get("n_burnin", 500)),
        output_dir         = str(results_root() / out_cfg.get("dir", "").removeprefix("results/")),
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
        ds_format          = ds_fmt,
        fit_ng             = fit_ng,
        jk_dir             = data_file if fits_cfg else None,
        jk_pattern         = fits_cfg.get("jk_pattern", "NSIDE_04"),
        h_hubble           = float(fits_cfg.get("h", 0.6736)),
        cosmology          = cosmology,
        use_free_cosmo     = bool(fit_cfg.get("use_free_cosmo", False)),
        use_bnl            = bool(model_cfg.get("use_bnl", True)),
    )


# Backward-compatibility aliases
load_joint_config = load_config
load_fits_config  = load_config


# ---------------------------------------------------------------------------
# Figure-unit readers — return data as plotted on paper figures
# ---------------------------------------------------------------------------

def read_wp_bwpd_fig3(path: str, rp_min: float = 0.0, rp_max: float = 1e9):
    """Read a 3-column BWPD wp file and return quantities in figure-plot units.

    Expected column order (comma/whitespace-delimited, spaces allowed):
        rp_hMpc   rpwp_h2Mpc2   rpwp_upper_h2Mpc2

    Parameters
    ----------
    path : str
        Path to the CSV file.
    rp_min, rp_max : float
        Optional scale cuts in h^-1 Mpc (applied to rp_hMpc column).

    Returns
    -------
    rp : np.ndarray
        Projected separation r_p in h^-1 Mpc.
    rp_wp : np.ndarray
        r_p × w_p(r_p) in (h^-1 Mpc)^2  — the y-axis quantity in the figure.
    err : np.ndarray
        1-sigma uncertainty on r_p × w_p; computed as upper_bound − central.
    """
    import re
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [x for x in re.split(r"[,\s]+", line) if x]
            try:
                rows.append([float(x) for x in fields[:3]])
            except ValueError:
                continue
    d = np.array(rows)
    rp        = d[:, 0]
    rp_wp     = d[:, 1]
    rp_wp_up  = d[:, 2]
    mask = (rp >= rp_min) & (rp <= rp_max)
    return rp[mask], rp_wp[mask], (rp_wp_up - rp_wp)[mask]


def read_esd_bwpd_4col_fig3(path: str, R_min: float = 0.0, R_max: float = 1e9):
    """Read a 4-column BWPD ESD file and return ΔΣ in figure-plot units.

    Expected column order (comma/whitespace-delimited, spaces allowed):
        rp_hMpc   DS_hMsunpc2   DS_upper_hMsunpc2   DS_lower_hMsunpc2

    The ``DS_hMsunpc2`` column stores **ΔΣ** directly in h M_sun/pc^2,
    which is the y-axis quantity in More+2015 Figure 3 (right panels).
    Error bounds are asymmetric.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    R_min, R_max : float
        Optional scale cuts in h^-1 Mpc.

    Returns
    -------
    R : np.ndarray
        Projected radius R in h^-1 Mpc.
    DS : np.ndarray
        ΔΣ(R) in h M_sun/pc^2.
    err_hi : np.ndarray
        Upper 1-sigma error on ΔΣ  (upper_bound − central).
    err_lo : np.ndarray
        Lower 1-sigma error on ΔΣ  (central − lower_bound).
    """
    import re
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [x for x in re.split(r"[,\s]+", line) if x]
            try:
                rows.append([float(x) for x in fields[:4]])
            except ValueError:
                continue
    d = np.array(rows)
    R      = d[:, 0]
    DS     = d[:, 1]
    DS_up  = d[:, 2]
    DS_lo  = d[:, 3]
    mask = (R >= R_min) & (R <= R_max)
    return R[mask], DS[mask], (DS_up - DS)[mask], (DS - DS_lo)[mask]
