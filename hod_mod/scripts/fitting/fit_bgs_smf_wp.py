"""Joint fit of ZuMandelbaum+2015 iHOD to wp(rp) + SMF Phi(M*) for BGS Mstar-threshold samples.

Data source:
    ~/software/sum_stat/data/BGS_Mstar{9.0,10.0,10.5,11.0,11.5}/  (sum_stat HDF5)
    Both wp(rp) and SMF Phi(M*) are read from the same jackknife HDF5 file.

Model:
    ZuMandelbaum15HODModel (iHOD, Zu & Mandelbaum 2015, MNRAS 454, 1161)

Free parameters (14 — all ZM15 parameters):
    log10m_star_thresh  stellar-mass threshold (≈ sample cut)
    sigma_lnmstar       SHMR log-normal scatter in M*
    lg_m1h              SHMR characteristic halo mass
    lg_m0star           SHMR pivot stellar mass
    beta                SHMR power-law slope
    delta               SHMR Behroozi transition parameter
    gamma               SHMR Behroozi transition parameter
    eta                 SHMR scatter slope
    fc                  central completeness fraction
    bsat                satellite bsat normalisation
    beta_sat            satellite bsat slope
    bcut                satellite bcut normalisation
    beta_cut            satellite bcut slope
    alpha_sat           satellite occupation power-law slope

Likelihood:
    LL = -0.5 * Σ[(phi_data - phi_model)² / var_phi]
       - 0.5 * Σ[(wp_data  - wp_model )² / var_wp ]
    where var = jackknife_variance + (f_sys * |data|)²  (diagonal; no cross-terms).

Scale cuts:
    wp  : rp > rp_min (default 0.02 Mpc/h)
    SMF : all bins with phi > 0 kept

Output (to results/fits/bgs_smf_wp/):
    <label>_map.json            MAP parameters + chi2/dof
    <label>_chain.h5            emcee chain (HDF5; falls back to .npy)
    <label>_mcmc_summary.json   MCMC medians and 68% credibles
    <label>_corner.pdf          posterior corner plot (MCMC only)
    <label>_bestfit.pdf         wp(rp) + SMF best-fit figure

Samples:
    M9    BGS_Mstar9.0   (z_max=0.08, z_mean≈0.06)
    M10   BGS_Mstar10.0  (z_max=0.18, z_mean≈0.135)
    M10p5 BGS_Mstar10.5  (z_max=0.26, z_mean≈0.191)
    M11   BGS_Mstar11.0  (z_max=0.35, z_mean≈0.252)
    M11p5 BGS_Mstar11.5  (z_max=0.35, z_mean≈0.261)

Usage::

    # MAP fit for all samples
    for s in M9 M10 M10p5 M11 M11p5; do
        python -m hod_mod.scripts.fitting.fit_bgs_smf_wp --sample $s --mode map
    done

    # MAP + MCMC for one sample
    python -m hod_mod.scripts.fitting.fit_bgs_smf_wp --sample M10 --mode both

    # Custom scale cut, no plots
    python -m hod_mod.scripts.fitting.fit_bgs_smf_wp --sample M10 --mode map --rp-min 0.1 --no-plot

References
----------
Zu & Mandelbaum 2015, MNRAS 454, 1161  (iHOD)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import (
    ZuMandelbaum15HODModel, n_cen_thresh_zu15, n_sat_thresh_zu15,
)
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.data_io.sum_stat_reader import SumStatReader
from hod_mod.paths import results_root

try:
    _JAX_CACHE = Path(os.path.expanduser("~/.cache/jax_xla_bgs_smf_wp"))
    _JAX_CACHE.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_SUM_STAT_DIR = Path(os.path.expanduser("~/software/sum_stat/data"))
_RESULTS_DIR  = results_root() / "fits" / "bgs_smf_wp"

# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------
# z_mean: median redshift of volume-limited sample.
# z_max:  upper redshift limit encoded in the HDF5 filename.
# mstar_thresh: log10(M*/Msun) lower bound of the sample.
SAMPLES: dict[str, dict] = {
    "M9":    dict(mstar_thresh= 9.0, zmax=0.08, zmean=0.060, N=  523486, sum_stat_dir="BGS_Mstar9.0"),
    "M10":   dict(mstar_thresh=10.0, zmax=0.18, zmean=0.135, N=2759238, sum_stat_dir="BGS_Mstar10.0"),
    "M10p5": dict(mstar_thresh=10.5, zmax=0.26, zmean=0.191, N=3263228, sum_stat_dir="BGS_Mstar10.5"),
    "M11":   dict(mstar_thresh=11.0, zmax=0.35, zmean=0.252, N=1619838, sum_stat_dir="BGS_Mstar11.0"),
    "M11p5": dict(mstar_thresh=11.5, zmax=0.35, zmean=0.261, N= 120882, sum_stat_dir="BGS_Mstar11.5"),
}

# --------------------------------------------------------------------------
# Cosmology (Planck 2018)
# --------------------------------------------------------------------------
_THETA_COSMO = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "H0":     _THETA_COSMO["h"] * 100.0,
    "Om0":    _THETA_COSMO["Omega_m"],
    "Ob0":    _THETA_COSMO["Omega_b"],
    "ns":     _THETA_COSMO["n_s"],
    "sigma8": 0.811,
}

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
_PARAM_NAMES = [
    "log10m_star_thresh",   # (1)  stellar mass threshold
    "sigma_lnmstar",        # (2)  SHMR scatter
    "lg_m1h",               # (3)  characteristic halo mass
    "lg_m0star",            # (4)  pivot stellar mass
    "beta",                 # (5)  SHMR slope
    "delta",                # (6)  SHMR Behroozi transition
    "gamma",                # (7)  SHMR Behroozi transition
    "eta",                  # (8)  scatter slope
    "fc",                   # (9)  central completeness
    "bsat",                 # (10) satellite amplitude
    "beta_sat",             # (11) satellite slope
    "bcut",                 # (12) satellite cut amplitude
    "beta_cut",             # (13) satellite cut slope
    "alpha_sat",            # (14) satellite power-law slope
]

_PARAM_BOUNDS = [
    ( 8.0, 12.5),   # log10m_star_thresh
    (0.01,  2.0),   # sigma_lnmstar
    ( 9.0, 15.0),   # lg_m1h
    ( 8.0, 12.5),   # lg_m0star
    ( 0.0,  2.0),   # beta
    ( 0.0,  3.0),   # delta
    ( 0.1, 10.0),   # gamma
    (-1.5,  1.5),   # eta
    (0.05,  1.0),   # fc
    ( 0.1, 50.0),   # bsat
    ( 0.0,  2.0),   # beta_sat
    (0.01, 10.0),   # bcut
    ( 0.0,  2.0),   # beta_cut
    ( 0.3,  3.0),   # alpha_sat
]

assert len(_PARAM_NAMES) == len(_PARAM_BOUNDS) == 14


def _x0_for(label: str) -> np.ndarray:
    """Initial guess: ZM15 defaults with log10m_star_thresh set to sample cut."""
    d = ZuMandelbaum15HODModel.default_params()
    d["log10m_star_thresh"] = SAMPLES[label]["mstar_thresh"]
    return np.array([d[n] for n in _PARAM_NAMES])


def _params_to_hod(params: np.ndarray) -> dict:
    """Convert parameter vector to ZuMandelbaum15HODModel parameter dict."""
    return dict(zip(_PARAM_NAMES, params.tolist()))


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _sum_stat_path(label: str) -> Path:
    d = _SUM_STAT_DIR / SAMPLES[label]["sum_stat_dir"]
    N_str = f"{SAMPLES[label]['N']:07d}"
    matches = sorted(d.glob(f"*_N_{N_str}_joint_smf-wp-esd*.h5"))
    if not matches:
        matches = sorted(d.glob("*.h5"))
    if not matches:
        raise FileNotFoundError(f"No HDF5 file found for {label} in {d}")
    return matches[0]


def load_data(label: str, rp_min: float = 0.02, f_sys: float = 0.05) -> dict:
    """Load and prepare the joint SMF + wp data for fitting.

    Returns a dict with keys:
        log10mstar, phi, phi_var  — SMF bins [Mpc/h]^-3 dex^-1
        rp, wp, wp_var            — projected correlation [Mpc/h]
        pi_max                    — line-of-sight integration limit [Mpc/h]
        n_smf, n_wp               — number of data points in each probe
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

    # full joint covariance — diagonal only is used in the likelihood
    cov_full = np.asarray(jt["cov"], dtype=float)
    n_tot    = len(phi_all) + len(wp_all)
    # cov_full may span only kept probes; ensure shape matches
    n_smf_raw = len(phi_all)

    smf_mask = phi_all > 0
    wp_mask  = rp_all > rp_min

    phi = phi_all[smf_mask]
    rp  = rp_all[wp_mask]
    wp  = wp_all[wp_mask]

    # Extract diagonal variances from the joint covariance block
    # cov_full is (n_smf_raw + n_wp_raw, same) — build index arrays
    all_mask = np.concatenate([smf_mask, wp_mask])
    kept_idx = np.nonzero(all_mask)[0]
    jk_var   = np.diag(cov_full)[kept_idx]

    n_smf = int(smf_mask.sum())
    n_wp  = int(wp_mask.sum())

    # Separate jackknife variances per probe
    phi_jkvar = jk_var[:n_smf]
    wp_jkvar  = jk_var[n_smf:]

    # Systematic floor: var_eff = jk_var + (f_sys * |data|)^2
    phi_var = phi_jkvar + (f_sys * np.abs(phi)) ** 2
    wp_var  = wp_jkvar  + (f_sys * np.abs(wp))  ** 2
    # guard against exact zeros
    phi_var = np.where(phi_var > 0, phi_var, (f_sys * np.abs(phi).max() * 1e-3) ** 2 + 1e-60)
    wp_var  = np.where(wp_var  > 0, wp_var,  (f_sys * np.abs(wp ).max() * 1e-3) ** 2 + 1e-60)

    pi_max = float(getattr(reader, "_pi_max", 100.0))
    return dict(
        log10mstar=log10mstar[smf_mask], phi=phi, phi_var=phi_var,
        rp=rp, wp=wp, wp_var=wp_var,
        pi_max=pi_max,
        n_smf=n_smf, n_wp=n_wp,
    )


# --------------------------------------------------------------------------
# Infrastructure
# --------------------------------------------------------------------------

class _Infrastructure:
    """Halo-model stack built once and shared across samples."""

    def __init__(self, hmf_backend: str = "csst"):
        print(f"Building halo-model infrastructure (CAMB + HMF[{hmf_backend}]) ...", flush=True)
        t0 = time.time()
        try:
            from hod_mod.core.beyond_linear_bias import BeyondLinearBiasMead21
            bnl = BeyondLinearBiasMead21()
        except Exception as exc:
            print(f"  WARNING: BeyondLinearBiasMead21 unavailable ({exc}); using linear bias.", flush=True)
            bnl = None

        pk_lin    = LinearPowerSpectrum()
        hmf       = make_hmf(hmf_backend, pk_func=pk_lin.pk_linear)
        hp        = HaloProfile(_COLOSSUS, cm_relation="diemer19")
        hod       = ZuMandelbaum15HODModel(hmf, hmf.bias)
        self.fhmp = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
        print(f"  done in {time.time() - t0:.1f}s", flush=True)


# --------------------------------------------------------------------------
# SMF model prediction (vectorized)
# --------------------------------------------------------------------------

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
    """Model SMF Phi(M*) = -dN(>M*)/dlog10(M*) [(Mpc/h)^-3 dex^-1].

    Vectorised: evaluates n_cen + n_sat for the whole threshold grid in one
    batched JAX call.  Units match the sum_stat SMF convention (h^3 Mpc^-3 dex^-1
    = (Mpc/h)^-3 dex^-1).
    """
    h = float(_THETA_COSMO["h"])
    thresh_grid = jnp.asarray(log10mstar_grid, dtype=float) + np.log10(h)

    hod = infra.fhmp._hod
    cosmo_key = infra.fhmp._cosmo_cache_key(z, _THETA_COSMO)
    if cosmo_key not in infra.fhmp._static_cache:
        infra.fhmp._pk_tables_full(z, _THETA_COSMO, hod_params)
    sc      = infra.fhmp._static_cache[cosmo_key]
    dndm_np = sc["dndm_np"]
    m_np    = sc["m_np"]
    log10m_grid = hod._log10m_grid

    p  = hod_params
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
    n_cum = np.trapezoid(dndm_np[None, :] * n_tot, m_np, axis=1)
    return -np.gradient(n_cum, log10mstar_grid)


# --------------------------------------------------------------------------
# Likelihood, prior, log_prob
# --------------------------------------------------------------------------

def log_likelihood(
    params: np.ndarray,
    label: str,
    infra: _Infrastructure,
    data: dict,
) -> float:
    """Diagonal Gaussian log-likelihood over SMF Phi(M*) and wp(rp)."""
    hp   = _params_to_hod(params)
    z    = SAMPLES[label]["zmean"]

    phi_model = _predict_smf(infra, z, hp, data["log10mstar"])
    wp_model  = np.asarray(
        infra.fhmp.wp(
            data["rp"], pi_max=data["pi_max"],
            z=z, theta_cosmo=_THETA_COSMO, hod_params=hp,
        ),
        dtype=float,
    )

    ll_smf = -0.5 * float(np.sum((data["phi"] - phi_model) ** 2 / data["phi_var"]))
    ll_wp  = -0.5 * float(np.sum((data["wp"]  - wp_model)  ** 2 / data["wp_var"]))
    return ll_smf + ll_wp


def log_prior(params: np.ndarray) -> float:
    """Flat box priors on all 14 parameters."""
    for v, (lo, hi) in zip(params, _PARAM_BOUNDS):
        if not (lo <= v <= hi):
            return -np.inf
    return 0.0


def log_prob(
    params: np.ndarray,
    label: str,
    infra: _Infrastructure,
    data: dict,
) -> float:
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    try:
        ll = log_likelihood(params, label, infra, data)
        return lp + ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


# --------------------------------------------------------------------------
# MAP fit
# --------------------------------------------------------------------------

def run_map(
    label: str,
    infra: _Infrastructure,
    rp_min: float = 0.02,
    f_sys: float = 0.05,
) -> dict:
    """L-BFGS-B MAP fit for all 14 ZM15 parameters."""
    data  = load_data(label, rp_min=rp_min, f_sys=f_sys)
    x0    = _x0_for(label)
    n_pts = data["n_smf"] + data["n_wp"]
    ndof  = max(n_pts - len(_PARAM_NAMES), 1)

    print(f"  [{label}] MAP: x0={np.round(x0, 3)}", flush=True)
    print(f"  [{label}]      n_smf={data['n_smf']}  n_wp={data['n_wp']}  ndof={ndof}", flush=True)

    def neg_log_prob(p):
        try:
            v = log_prob(p, label, infra, data)
            return -v if np.isfinite(v) else 1e30
        except Exception:
            return 1e30

    res = minimize(
        neg_log_prob, x0, method="L-BFGS-B",
        bounds=_PARAM_BOUNDS,
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 3000, "eps": 1e-3},
    )

    chi2 = 2.0 * res.fun
    result = dict(
        label       = label,
        param_names = list(_PARAM_NAMES),
        params      = res.x.tolist(),
        chi2        = float(chi2),
        ndof        = int(ndof),
        chi2_dof    = float(chi2 / ndof),
        log_prob    = float(-res.fun),
        success     = bool(res.success),
        n_pts_smf   = data["n_smf"],
        n_pts_wp    = data["n_wp"],
        n_pts_fit   = n_pts,
        rp_min_hmpc = rp_min,
        f_sys       = f_sys,
    )
    print(
        f"  [{label}] MAP done: params={np.round(res.x, 4)}\n"
        f"  [{label}]           chi2/dof={chi2/ndof:.2f}  success={res.success}",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------
# MCMC
# --------------------------------------------------------------------------

def run_mcmc(
    label: str,
    infra: _Infrastructure,
    map_result: dict,
    n_walkers: int = 32,
    n_steps: int   = 1000,
    n_burnin: int  = 300,
    rp_min: float  = 0.02,
    f_sys: float   = 0.05,
) -> dict:
    """emcee ensemble sampler starting near MAP, for all 14 parameters."""
    import emcee

    data  = load_data(label, rp_min=rp_min, f_sys=f_sys)
    x_map = np.array(map_result["params"])
    n_dim = len(x_map)

    def lp(p):
        return log_prob(p, label, infra, data)

    # walker spread: tighter for smooth parameters, wider for amplitude-like ones
    # Order follows _PARAM_NAMES: thresh, sigma, m1h, m0star, beta, delta, gamma, eta, fc,
    #                              bsat, beta_sat, bcut, beta_cut, alpha_sat
    scales = np.array([0.10, 0.05, 0.10, 0.10,
                       0.05, 0.05, 0.10, 0.05,
                       0.05,
                       0.50, 0.05, 0.10, 0.05, 0.10])
    rng = np.random.default_rng(42)
    pos = x_map[None, :] + scales[None, :] * rng.standard_normal((n_walkers, n_dim))
    # clip walkers to prior bounds
    for i, (lo, hi) in enumerate(_PARAM_BOUNDS):
        pos[:, i] = np.clip(pos[:, i], lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lp)

    print(f"  [{label}] MCMC burn-in: {n_burnin} steps × {n_walkers} walkers ...", flush=True)
    pos, _, _ = sampler.run_mcmc(pos, n_burnin, progress=True)
    sampler.reset()

    print(f"  [{label}] MCMC production: {n_steps} steps ...", flush=True)
    sampler.run_mcmc(pos, n_steps, progress=True)

    flat_chain = sampler.get_chain(flat=True)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = np.full(n_dim, np.nan)

    medians = np.median(flat_chain, axis=0)
    lo16    = np.percentile(flat_chain, 16, axis=0)
    hi84    = np.percentile(flat_chain, 84, axis=0)

    return dict(
        label        = label,
        param_names  = list(_PARAM_NAMES),
        chain        = flat_chain,
        medians      = medians.tolist(),
        lo16         = lo16.tolist(),
        hi84         = hi84.tolist(),
        autocorr_tau = tau.tolist() if hasattr(tau, "tolist") else list(tau),
        n_walkers    = n_walkers,
        n_steps      = n_steps,
        n_burnin     = n_burnin,
    )


# --------------------------------------------------------------------------
# Save helpers
# --------------------------------------------------------------------------

def _save_map(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['label']}_map.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {path}", flush=True)
    return path


def _save_mcmc(mcmc: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = mcmc["label"]
    chain = mcmc.pop("chain")

    summary = {k: v for k, v in mcmc.items()}
    with open(out_dir / f"{label}_mcmc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {out_dir / f'{label}_mcmc_summary.json'}", flush=True)

    # Save chain
    try:
        import h5py
        chain_path = out_dir / f"{label}_chain.h5"
        with h5py.File(chain_path, "w") as f:
            f.create_dataset("chain", data=chain)
            f.attrs["param_names"] = json.dumps(mcmc["param_names"])
        print(f"  → {chain_path}", flush=True)
    except ImportError:
        chain_path = out_dir / f"{label}_chain.npy"
        np.save(chain_path, chain)
        print(f"  → {chain_path}", flush=True)
    mcmc["chain"] = chain


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_bestfit(
    label: str,
    infra: _Infrastructure,
    params: list,
    out_path: Path,
    rp_min: float = 0.02,
    f_sys: float  = 0.05,
    chi2_dof: float | None = None,
) -> None:
    """Two-panel best-fit: wp(rp) and SMF Phi(M*)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hp   = _params_to_hod(np.array(params))
    data = load_data(label, rp_min=rp_min, f_sys=0.0)  # f_sys=0 for plot
    z    = SAMPLES[label]["zmean"]
    s    = SAMPLES[label]

    phi_model = _predict_smf(infra, z, hp, data["log10mstar"])
    wp_model  = np.asarray(
        infra.fhmp.wp(
            data["rp"], pi_max=data["pi_max"],
            z=z, theta_cosmo=_THETA_COSMO, hod_params=hp,
        ),
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    chi_str = f"  $\\chi^2/\\nu={chi2_dof:.2f}$" if chi2_dof is not None else ""
    fig.suptitle(
        f"{label}: BGS M$_*>{s['mstar_thresh']}$, $z_{{\\rm mean}}={s['zmean']:.3f}$"
        f"{chi_str}", fontsize=11,
    )

    # --- wp ---
    ax, axr = axes[:, 0]
    phi_err = np.sqrt(data["phi_var"])
    wp_err  = np.sqrt(data["wp_var"])

    ax.errorbar(data["rp"], data["wp"], yerr=wp_err,
                fmt="o", ms=4, color="k", label="Data", zorder=2)
    ax.plot(data["rp"], wp_model, "-", lw=2, color="C0", label="Model", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$w_p(r_p)$ [$h^{-1}$Mpc]")
    ax.legend(fontsize=9); ax.set_title("Projected correlation function")

    pull_wp = (data["wp"] - wp_model) / wp_err
    axr.axhline(0, ls="--", color="0.5", lw=0.8)
    axr.plot(data["rp"], pull_wp, "o", ms=3, color="C0")
    axr.set_xscale("log")
    axr.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
    axr.set_ylabel("Pull")
    axr.set_ylim(-5, 5)

    # --- SMF ---
    ax2, axr2 = axes[:, 1]
    ax2.errorbar(data["log10mstar"], data["phi"], yerr=phi_err,
                 fmt="o", ms=4, color="k", label="Data", zorder=2)
    ax2.plot(data["log10mstar"], phi_model, "-", lw=2, color="C1", label="Model", zorder=3)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    ax2.set_ylabel(r"$\Phi$ [$(h^{-1}$Mpc)$^{-3}$dex$^{-1}$]")
    ax2.legend(fontsize=9); ax2.set_title("Stellar mass function")

    pull_smf = (data["phi"] - phi_model) / phi_err
    axr2.axhline(0, ls="--", color="0.5", lw=0.8)
    axr2.plot(data["log10mstar"], pull_smf, "o", ms=3, color="C1")
    axr2.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    axr2.set_ylabel("Pull")
    axr2.set_ylim(-5, 5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def plot_corner(label: str, mcmc: dict, out_path: Path) -> None:
    """Corner plot of the posterior chain."""
    try:
        import corner
    except ImportError:
        print("  corner not installed, skipping corner plot.", flush=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = corner.corner(
        mcmc["chain"],
        labels=_PARAM_NAMES,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 8},
    )
    fig.suptitle(label, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Fit ZuMandelbaum+2015 iHOD to BGS wp + SMF (all 14 params free)."
    )
    p.add_argument("--sample", required=True, choices=list(SAMPLES) + ["all"],
                   help="Sample label: M9, M10, M10p5, M11, M11p5, or 'all'")
    p.add_argument("--mode",   default="map", choices=["map", "mcmc", "both"],
                   help="map = MAP only; mcmc = MCMC from existing MAP; both = MAP + MCMC")
    p.add_argument("--rp-min",     type=float, default=0.02,
                   help="Minimum rp [Mpc/h] scale cut for wp(rp) (default: 0.02)")
    p.add_argument("--f-sys",      type=float, default=0.05,
                   help="Fractional systematic floor added to data variances (default: 0.05)")
    p.add_argument("--n-walkers",  type=int,   default=32)
    p.add_argument("--n-steps",    type=int,   default=1000)
    p.add_argument("--n-burnin",   type=int,   default=300)
    p.add_argument("--hmf-backend",default="csst",
                   help="HMF backend for FullHaloModelPrediction (default: csst)")
    p.add_argument("--no-plot",    action="store_true", help="Skip output figures")
    args = p.parse_args()

    labels = list(SAMPLES) if args.sample == "all" else [args.sample]

    infra = _Infrastructure(hmf_backend=args.hmf_backend)

    for label in labels:
        out_dir = _RESULTS_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n[{label}] Starting fit\n{'='*60}", flush=True)

        map_result = None
        if args.mode in ("map", "both"):
            t0 = time.time()
            map_result = run_map(label, infra, rp_min=args.rp_min, f_sys=args.f_sys)
            print(f"  [{label}] MAP wall-clock: {time.time()-t0:.0f}s", flush=True)
            _save_map(map_result, out_dir)
            if not args.no_plot:
                plot_bestfit(
                    label, infra, map_result["params"],
                    out_dir / f"{label}_bestfit.pdf",
                    rp_min=args.rp_min, f_sys=args.f_sys,
                    chi2_dof=map_result["chi2_dof"],
                )

        if args.mode in ("mcmc", "both"):
            if map_result is None:
                map_path = out_dir / f"{label}_map.json"
                if not map_path.exists():
                    print(
                        f"  [{label}] WARNING: MAP result not found at {map_path}, "
                        f"skipping MCMC.", flush=True,
                    )
                    continue
                with open(map_path) as fh:
                    map_result = json.load(fh)

            t0 = time.time()
            mcmc = run_mcmc(
                label, infra, map_result,
                n_walkers=args.n_walkers, n_steps=args.n_steps, n_burnin=args.n_burnin,
                rp_min=args.rp_min, f_sys=args.f_sys,
            )
            print(f"  [{label}] MCMC wall-clock: {time.time()-t0:.0f}s", flush=True)
            _save_mcmc(mcmc, out_dir)
            if not args.no_plot:
                plot_bestfit(
                    label, infra, mcmc["medians"],
                    out_dir / f"{label}_bestfit_mcmc.pdf",
                    rp_min=args.rp_min, f_sys=args.f_sys,
                )
                plot_corner(label, mcmc, out_dir / f"{label}_corner.pdf")


if __name__ == "__main__":
    main()
