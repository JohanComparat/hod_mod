#!/usr/bin/env python
"""Joint MAP + MCMC fit of the Zu & Mandelbaum (2015) iHOD model to the LS10 BGS
M*>10 stellar-mass-bin campaign measured by ``sum_stat``.

All stellar-mass bins are fit **simultaneously** with one shared set of thirteen
SHMR/HOD + satellite parameters.  Each bin contributes its number density, projected
clustering and (optionally) excess surface density from lensing surveys to a
single summed log-likelihood::

    log P(theta) = log_prior(theta)
                 + Sum_bins [ -0.5 * (chi2_ng + chi2_wp + Sum_surveys chi2_DS) ]

This is the global inverse-HOD fit of Zu & Mandelbaum 2015: one SHMR (with
scatter) predicts the galaxy content of every stellar-mass bin at once.

Data
----
The per-bin joint HDF5 files produced by::

    sum_stat/scripts/measure_joint_sumstat.py --survey bgs --mstar-binned \
        --mstar-edges 10.0,10.2,...,11.4 \
        --stats NBAR WP ESD_HSC ESD_DES ESD_KIDS --n-jk 100 ...

are read through
:class:`~hod_mod.data_io.sum_stat_reader.SumStatReader` — ``wp`` and
``esd_<survey>`` via :meth:`joint_bgs`, and the abundance via
:meth:`number_density`.

Units
-----
- ``wp``  : sum_stat stores physical Mpc; SumStatReader returns Mpc/h (x h).
            Predictor ``wp`` is native Mpc/h — compared directly.
- ``DS``  : sum_stat stores M_sun/pc^2 (h-invariant); predictor ``delta_sigma``
            returns M_sun h/pc^2 — divided by h before comparison.
- ``n_g`` : SumStatReader returns h^3 Mpc^-3; predictor ``n_gal`` is native.

Stellar-mass bins enter the predictor through ``log10m_star_thresh`` (bin lower
edge) and ``log10m_star_max`` (bin upper edge); the model returns the bin HOD.

Stellar masses are **physical** ``log10(M*/M_sun)`` throughout (the sum_stat
LePhare ``LPH_MASS_BEST``); no h-rescaling is applied to the mass axis.

Free parameters (13)
--------------------
lg_m1h        log10(M1 / [Msun/h])   SHMR characteristic halo mass
lg_m0star     log10(M*0 / Msun)      SHMR pivot stellar mass
beta          power-law slope of the SHMR
delta         Behroozi+10 transition parameter
gamma         Behroozi+10 transition parameter
sigma_lnmstar log-normal scatter in M* at fixed Mh
eta           slope of the scatter with Mh
fc            central-galaxy completeness fraction
bsat          satellite normalisation
beta_sat      satellite halo-mass slope
bcut          satellite cut-off normalisation
beta_cut      satellite cut-off slope
alpha_sat     satellite occupation power-law index

CLI options
-----------
--data-dir PATH
    Directory of per-bin sum_stat joint HDF5 files
    (default: ~/software/sum_stat/data/BGS_Mstar10_massbins)

--surveys [HSC [DES [KIDS]]]
    Lensing surveys to include.  Pass one or more of HSC, DES, KIDS.
    Omit entirely (``--surveys``) or pass an empty list to fit with
    wp + n_gal only (no lensing).  Default: HSC DES KIDS.

--mode {map,mcmc,both}
    map   — MAP optimisation only (Powell, saved to map_result.json).
    mcmc  — MCMC only; loads MAP starting point from map_result.json if present.
    both  — MAP then MCMC (default).

--plot-only
    Skip fitting; load existing map_result.json and regenerate plots.

--rp-min FLOAT    Minimum r_p for wp(rp) [Mpc/h] (default 0.1)
--rp-max FLOAT    Maximum r_p for wp(rp) [Mpc/h] (default 30)
--R-min  FLOAT    Minimum R for ΔΣ(R)   [Mpc/h] (default 0.1)
--R-max  FLOAT    Maximum R for ΔΣ(R)   [Mpc/h] (default 30)
--pi-max-mpc FLOAT   wp π_max in physical Mpc (default 100)
--z-eff FLOAT     Fallback redshift for bins without measured z_mean (default 0.13)
--hmf-backend STR HMF multiplicity function: tinker08, bocquet16, … (default tinker08)
--smf-file PATH   Observed SMF file (sum_stat joint *_smf_* HDF5) for comparison
                  only — NOT fitted. Default: auto-discover the widest-coverage
                  BGS_Mstar* file under --smf-data-dir.
--smf-data-dir PATH  Root searched for an observed SMF file when --smf-file is
                  omitted (default ~/software/sum_stat/data)
--ng-frac-err-floor FLOAT  Minimum fractional error on n_g (default 0.05)
--gaussian-prior  Add Gaussian prior from published ZM15 values (Table 2)
--n-walkers INT   emcee walkers (default 32)
--n-burnin  INT   emcee burn-in steps (default 500)
--n-steps   INT   emcee production steps (default 2000)
--out-dir   PATH  Output directory (default: results/bgs_zm15_joint)
--force-mcmc      Rerun MCMC even if chain.h5 / flatchain.npz already exist

Output files (in --out-dir)
---------------------------
map_result.json     MAP parameters, chi2, per-bin breakdown
chain.h5            emcee HDF5 backend (written incrementally; used for resume)
flatchain.npz       Flat MCMC chain (written once after all steps complete)
map_bestfit.pdf     Best-fit figure: wp + ΔΣ per survey for every bin
hod_occupation.pdf  N_cen / N_sat occupation curves at MAP
shmr.pdf            Stellar-to-halo mass relation vs literature
stellar_mass_function.pdf  Model SMF vs observed sum_stat SMF (NOT fitted)
satellite_fraction.pdf     Model satellite fraction f_sat(>M*)
zm15_montage.pdf    Combined 2×2 montage: wp, ΔΣ, SMF, SHMR (ZM15 layout)

Usage examples
--------------
    # Full fit: MAP + MCMC, all surveys
    python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint \\
        --data-dir ~/software/sum_stat/data/BGS_Mstar10_massbins \\
        --surveys HSC DES KIDS --mode both \\
        --n-walkers 32 --n-burnin 500 --n-steps 2000

    # wp + n_gal only (no lensing)
    python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint \\
        --data-dir ~/software/sum_stat/data/BGS_Mstar10_massbins \\
        --surveys --mode both --out-dir results/bgs_zm15_joint_wp

    # Resume interrupted MCMC
    python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint \\
        --mode mcmc --out-dir results/bgs_zm15_joint

    # Regenerate plots from existing MAP result
    python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint \\
        --plot-only --out-dir results/bgs_zm15_joint

References
----------
Zu & Mandelbaum 2015, MNRAS 454, 1161 (arXiv:1505.02781)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Zu & Mandelbaum 2015 parameter setup
# ---------------------------------------------------------------------------

# 13 free SHMR/HOD + satellite parameters with (lower, upper) bounds and starting
# value (ZM15 Table 2 published global iHOD values used as the optimiser init).
FREE_PARAMS: dict[str, tuple] = {
    "lg_m1h":        ((11.0, 14.0), 12.10),
    "lg_m0star":     ((9.0,  12.0), 10.31),
    "beta":          ((0.1,   1.5),  0.33),
    "delta":         ((0.1,   2.0),  0.42),
    "gamma":         ((0.5,   5.0),  1.21),
    "sigma_lnmstar": ((0.1,   1.5),  0.50),
    "eta":           ((-0.5,  0.2), -0.04),
    "fc":            ((0.3,   1.0),  0.86),
    "bsat":          ((1.0,  50.0),  8.98),
    "beta_sat":      ((0.0,   2.0),  0.90),
    "bcut":          ((0.01, 10.0),  0.86),
    "beta_cut":      ((0.0,   2.0),  0.41),
    "alpha_sat":     ((0.3,   3.0),  1.00),
}

# All parameters are now free; FIXED_PARAMS kept as empty dict for compatibility.
FIXED_PARAMS: dict[str, float] = {}

# Published global iHOD values (mean, sigma) for an optional Gaussian prior
PUBLISHED: dict[str, tuple] = {
    "lg_m1h":        (12.10, 0.17),
    "lg_m0star":     (10.31, 0.10),
    "beta":          (0.33,  0.21),
    "delta":         (0.42,  0.04),
    "gamma":         (1.21,  0.20),
    "sigma_lnmstar": (0.50,  0.04),
    "eta":           (-0.04, 0.02),
    "fc":            (0.86,  0.14),
    "bsat":          (8.98,  1.18),
    "beta_sat":      (0.90,  0.10),
    "bcut":          (0.86,  0.20),
    "beta_cut":      (0.41,  0.10),
    "alpha_sat":     (1.00,  0.20),
}

FREE_NAMES = list(FREE_PARAMS.keys())
_BIN_RE = re.compile(r"_Mbin_([0-9p.]+)_([0-9p.]+)_joint_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _regularised_icov(cov: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Inverse covariance with a small diagonal ridge (mirrors WpFitter)."""
    if cov is not None and cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] > 1:
        reg = 0.01 * np.diag(np.diag(cov))
        return np.linalg.inv(cov + reg)
    return np.diag(1.0 / np.asarray(err, dtype=float) ** 2)


def _parse_bin_edges(fname: str) -> tuple[float, float] | None:
    m = _BIN_RE.search(fname)
    if not m:
        return None
    lo, hi = (float(s.replace("p", ".")) for s in m.groups())
    return lo, hi


def load_bins(data_dir: str, surveys: list[str], rp_min: float, rp_max: float,
              R_min: float, R_max: float,
              ng_frac_err_floor: float, log=print) -> tuple[list[dict], float]:
    """Load every mass-bin HDF5 in *data_dir* into per-bin fit dictionaries."""
    from hod_mod.data_io.sum_stat_reader import SumStatReader

    paths = sorted(glob.glob(os.path.join(data_dir, "*_Mbin_*_joint_*.h5")))
    if not paths:
        raise FileNotFoundError(
            f"No '*_Mbin_*_joint_*.h5' files found in {data_dir}. "
            "Run the sum_stat --mstar-binned campaign first.")

    bins: list[dict] = []
    h_file = None
    for path in paths:
        edges = _parse_bin_edges(os.path.basename(path))
        if edges is None:
            log(f"  [skip] cannot parse bin edges: {os.path.basename(path)}")
            continue
        lo, hi = edges
        reader = SumStatReader.from_hdf5(path)
        h_file = reader.h() if h_file is None else h_file

        # wp(rp)
        jb_wp = reader.joint_bgs(probes=("wp",))
        rp    = np.asarray(jb_wp["rp_wp"])
        wp    = np.asarray(jb_wp["data_vector"])
        covwp = np.asarray(jb_wp["cov"])
        m_wp  = (rp >= rp_min) & (rp <= rp_max)
        rp, wp = rp[m_wp], wp[m_wp]
        icov_wp = _regularised_icov(covwp[np.ix_(m_wp, m_wp)], np.sqrt(np.diag(covwp))[m_wp])

        # ESD per survey (shared R grid; one model ΔΣ compared to each survey)
        slices = reader._cache.get("joint_bgs", {}).get("slices", {})
        surv_data: dict[str, tuple] = {}
        R_ref = None
        for s in surveys:
            probe = f"esd_{s.lower()}"
            if probe not in slices:
                continue
            jb = reader.joint_bgs(probes=(probe,))
            R   = np.asarray(jb["rp_esd"])
            ds  = np.asarray(jb["data_vector"])
            cov = np.asarray(jb["cov"])
            m_R = (R >= R_min) & (R <= R_max) & np.isfinite(ds)
            if m_R.sum() < 2 or not np.all(np.isfinite(ds[m_R])):
                log(f"  [{lo}-{hi}] {s}: insufficient finite ΔΣ points — dropped")
                continue
            R, ds = R[m_R], ds[m_R]
            icov_ds = _regularised_icov(cov[np.ix_(m_R, m_R)], np.sqrt(np.diag(cov))[m_R])
            surv_data[s] = (R, ds, icov_ds)
            R_ref = R if R_ref is None else R_ref

        # number density (single point)
        nd        = reader.number_density()
        n_obs     = float(nd["n"])
        n_err     = float(nd["n_err"])
        frac_err  = max(n_err / n_obs if n_obs > 0 else ng_frac_err_floor, ng_frac_err_floor)

        # Per-bin effective redshift, measured on the data by sum_stat (weighted
        # mean z written to the file).  Falls back to the global default later.
        z_bin = nd["attrs"].get("z_mean", None)
        try:
            z_bin = float(z_bin)
            if not np.isfinite(z_bin):
                z_bin = None
        except (TypeError, ValueError):
            z_bin = None

        bins.append({
            "label":    f"{lo:g}-{hi:g}",
            "thresh":   lo,            # physical log10(M*/M_sun), bin lower edge
            "max":      hi,            # physical log10(M*/M_sun), bin upper edge
            "rp":       rp,
            "wp_obs":   wp,
            "icov_wp":  icov_wp,
            "surveys":  surv_data,
            "n_obs":    n_obs,
            "n_frac":   frac_err,
            "z":        z_bin,
        })
        esd_str = (", ".join(f"{s}:{len(v[0])}" for s, v in surv_data.items())
                   if surv_data else "none")
        log(f"  [{lo:g}-{hi:g}]  z={'%.3f'%z_bin if z_bin is not None else 'n/a'}  "
            f"wp={len(rp):2d}pts  ESD={{{esd_str}}}  "
            f"n={n_obs:.3e} h^3Mpc^-3 (±{100*frac_err:.0f}%)")

    if not bins:
        raise RuntimeError("No usable mass bins were loaded.")
    return bins, float(h_file)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def build_predictor(hmf_backend: str):
    """Construct a ZuMandelbaum15HODModel FullHaloModelPrediction (mirrors WpFitter)."""
    from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
    from hod_mod.cosmology.halo_mass_function import make_hmf
    from hod_mod.cosmology.halo_profiles import HaloProfile
    from hod_mod.galaxies.clustering import FullHaloModelPrediction
    from hod_mod.galaxies.hod import ZuMandelbaum15HODModel

    pk          = LinearPowerSpectrum()
    theta_cosmo = pk.default_cosmology()
    hmf         = make_hmf(hmf_backend, pk_func=pk.pk_linear)
    hod         = ZuMandelbaum15HODModel(hmf)   # _SINGLE_ARG_INIT = True
    predictor   = FullHaloModelPrediction(pk, hod, HaloProfile(theta_cosmo))
    return predictor, theta_cosmo


# ---------------------------------------------------------------------------
# Joint likelihood across all bins (shared parameters)
# ---------------------------------------------------------------------------

class JointZM15:
    def __init__(self, bins, predictor, theta_cosmo, h, z, pi_max_h,
                 gaussian_prior=False):
        import jax.numpy as jnp
        self._jnp        = jnp
        self.bins        = bins
        self.predictor   = predictor
        self.theta_cosmo = theta_cosmo
        self.h           = h
        self.z           = z
        self.pi_max_h    = pi_max_h
        self.gauss       = gaussian_prior
        self.bounds      = [FREE_PARAMS[p][0] for p in FREE_NAMES]
        self.x0          = np.array([FREE_PARAMS[p][1] for p in FREE_NAMES])

    # -- prior ----------------------------------------------------------
    def log_prior(self, theta) -> float:
        lp = 0.0
        for val, (lo, hi) in zip(theta, self.bounds):
            if not (lo <= val <= hi):
                return -np.inf
        if self.gauss:
            for name, val in zip(FREE_NAMES, theta):
                mu, sig = PUBLISHED[name]
                lp += -0.5 * ((val - mu) / sig) ** 2
        return lp

    _PENALTY = 1e4   # finite chi2 penalty per probe when n_gal ~ 0

    # -- per-bin chi2 ---------------------------------------------------
    def _bin_chi2(self, theta, b) -> dict:
        jnp = self._jnp
        z   = b.get("z") or self.z          # per-bin effective redshift
        p = dict(zip(FREE_NAMES, theta))
        p["log10m_star_thresh"] = b["thresh"]
        p["log10m_star_max"]    = b["max"]

        n_surv = len(b["surveys"])
        _bad = {"wp": self._PENALTY, "ds": n_surv * self._PENALTY,
                "ng": self._PENALTY, "total": (2 + n_surv) * self._PENALTY}

        try:
            wp_pred = np.asarray(self.predictor.wp(
                jnp.array(b["rp"]), self.pi_max_h, z, self.theta_cosmo, p))
        except Exception:
            return _bad
        if not np.all(np.isfinite(wp_pred)):
            return _bad
        r = wp_pred - b["wp_obs"]
        chi2_wp = float(r @ b["icov_wp"] @ r)

        chi2_ds = 0.0
        for (R, ds_obs, icov_ds) in b["surveys"].values():
            try:
                ds_pred = np.asarray(self.predictor.delta_sigma(
                    jnp.array(R), z, self.theta_cosmo, p)) / self.h
            except Exception:
                chi2_ds += self._PENALTY
                continue
            if not np.all(np.isfinite(ds_pred)):
                chi2_ds += self._PENALTY
                continue
            rd = ds_pred - ds_obs
            chi2_ds += float(rd @ icov_ds @ rd)

        try:
            ng_pred = float(self.predictor.n_gal(z, self.theta_cosmo, p))
        except Exception:
            ng_pred = 0.0
        chi2_ng = ((ng_pred - b["n_obs"]) / (b["n_frac"] * b["n_obs"])) ** 2

        total = chi2_wp + chi2_ds + chi2_ng
        if not np.isfinite(total):
            return _bad
        return {"wp": chi2_wp, "ds": chi2_ds, "ng": chi2_ng, "total": total}

    def log_prob(self, theta) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        try:
            chi2 = sum(self._bin_chi2(theta, b)["total"] for b in self.bins)
        except Exception:
            return -np.inf
        if not np.isfinite(chi2):
            return -np.inf
        return lp - 0.5 * chi2

    # -- ndof -----------------------------------------------------------
    def n_data(self) -> int:
        n = 0
        for b in self.bins:
            n += len(b["rp"]) + 1               # wp + n_g
            n += sum(len(v[0]) for v in b["surveys"].values())
        return n

    # -- MAP ------------------------------------------------------------
    def _diagnose(self, theta) -> None:
        """Print per-bin chi2 or the exception — called when starting log-prob is -inf."""
        import traceback
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            for name, val, (lo, hi) in zip(FREE_NAMES, theta,
                                           [FREE_PARAMS[p][0] for p in FREE_NAMES]):
                if not (lo <= val <= hi):
                    print(f"  PRIOR FAIL: {name}={val:.4f} outside [{lo}, {hi}]")
            return
        for b in self.bins:
            try:
                c = self._bin_chi2(theta, b)
                print(f"  [{b['label']}] chi2_wp={c['wp']:.2f}  chi2_ds={c['ds']:.2f}"
                      f"  chi2_ng={c['ng']:.2f}  total={c['total']:.2f}")
            except Exception:
                print(f"  [{b['label']}] EXCEPTION:")
                traceback.print_exc()

    def map_fit(self) -> dict:
        from scipy.optimize import minimize
        lp0 = self.log_prob(self.x0)
        print(f"  start log-prob={lp0:.2f}  (chi2={-2*lp0:.1f})")
        if not np.isfinite(lp0):
            print("  Diagnosing starting point failure:")
            self._diagnose(self.x0)
            raise RuntimeError("Non-finite log-prob at the starting point.")
        res = minimize(lambda x: -self.log_prob(x), self.x0, method="Powell",
                       options={"maxiter": 50000, "xtol": 1e-5, "ftol": 1e-5,
                                "disp": True})
        theta = res.x
        params = dict(zip(FREE_NAMES, [float(v) for v in theta]))
        per_bin = {b["label"]: self._bin_chi2(theta, b) for b in self.bins}
        chi2_tot = sum(c["total"] for c in per_bin.values())
        ndof = self.n_data() - len(FREE_NAMES)
        return {
            "theta": [float(v) for v in theta],
            "free_params": FREE_NAMES,
            "params": params,
            "chi2": chi2_tot,
            "ndof": int(ndof),
            "chi2_per_dof": chi2_tot / max(ndof, 1),
            "chi2_per_bin": per_bin,
            "success": bool(res.success),
            "message": str(res.message),
        }

    # -- MCMC -----------------------------------------------------------
    def sample(self, out_dir, n_walkers, n_burnin, n_steps, x_start=None):
        import emcee
        ndim = len(FREE_NAMES)
        os.makedirs(out_dir, exist_ok=True)
        backend_path = os.path.join(out_dir, "chain.h5")

        # Burn-in and production are kept in ONE continuous chain on disk (no
        # mid-run reset), and emcee's HDFBackend flushes after *every* step.  A
        # job killed by the walltime can therefore be resumed exactly where it
        # stopped simply by re-submitting: ``backend.iteration`` tells us how
        # many steps survived, and we run only the remainder.  Burn-in is
        # discarded at read-out (below), never on disk.
        total   = int(n_burnin) + int(n_steps)
        backend = emcee.backends.HDFBackend(backend_path)
        already = backend.iteration if os.path.exists(backend_path) else 0

        sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_prob,
                                        backend=backend)

        if already >= total:
            print(f"  chain already complete ({already} >= {total} steps) — "
                  f"skipping sampling  ({backend_path})")
        elif already == 0:
            x0    = np.asarray(x_start if x_start is not None else self.x0, dtype=float)
            rng   = np.random.default_rng(42)
            scale = np.maximum(np.abs(x0) * 1e-2, 1e-3)
            p0    = x0 + rng.normal(0, scale, (n_walkers, ndim))
            for i, (lo, hi) in enumerate(self.bounds):
                p0[:, i] = np.clip(p0[:, i], lo + 1e-6, hi - 1e-6)
            print(f"  running {total} steps "
                  f"({n_burnin} burn-in + {n_steps} production), checkpointing "
                  f"every step to {backend_path} ...")
            sampler.run_mcmc(p0, total, progress=True)
        else:
            remaining = total - already
            print(f"  resuming from step {already}/{total} "
                  f"({remaining} steps left) ...")
            sampler.run_mcmc(None, remaining, progress=True)

        # Discard the burn-in only when reading the chain back out.
        n_done  = sampler.iteration
        discard = min(int(n_burnin), max(n_done - 1, 0))
        out = os.path.join(out_dir, "flatchain.npz")
        np.savez(out, flatchain=sampler.get_chain(discard=discard, flat=True),
                 param_names=np.array(FREE_NAMES))
        try:
            tau = sampler.get_autocorr_time(discard=discard, tol=0)
            print(f"  mean autocorr time: {np.nanmean(tau):.1f} steps")
        except Exception:
            pass
        print(f"  chain saved -> {out}  (HDF5 backend: {backend_path})")
        return sampler


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_map(bins, predictor, theta_cosmo, h, pi_max_h, map_result, out_dir,
             surveys):
    """Best-fit MAP figure: one column per mass bin, rows = wp / ESD per survey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    params   = map_result["params"]
    per_bin  = map_result.get("chi2_per_bin", {})
    n_bins   = len(bins)
    n_rows = 1 + len(surveys)   # wp row + one row per lensing survey

    fig, axes = plt.subplots(
        n_rows, n_bins,
        figsize=(3.0 * n_bins, 2.8 * n_rows),
        sharex="col",
        squeeze=False,
    )
    survey_colors = {"HSC": "C0", "DES": "C2", "KIDS": "C3"}

    for col, b in enumerate(bins):
        label = b["label"]
        z     = b.get("z") or 0.13
        p     = dict(params)
        p["log10m_star_thresh"] = b["thresh"]
        p["log10m_star_max"]    = b["max"]
        chi2_info = per_bin.get(label, {})

        # --- wp ---
        ax = axes[0, col]
        rp     = b["rp"]
        wp_obs = b["wp_obs"]
        wp_err = np.sqrt(np.diag(np.linalg.inv(b["icov_wp"])))
        try:
            wp_mod = np.asarray(predictor.wp(
                jnp.array(rp), pi_max_h, z, theta_cosmo, p))
        except Exception:
            wp_mod = np.full_like(rp, np.nan)

        ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=3, color="k",
                    zorder=2, label="data")
        ax.plot(rp, wp_mod, "-", lw=2, color="C0", zorder=3, label="MAP")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(
            f"$M_*\\in[{b['thresh']},{b['max']}]$\n"
            f"$\\chi^2_{{wp}}={chi2_info.get('wp', 0):.0f}$",
            fontsize=8,
        )
        if col == 0:
            ax.set_ylabel(r"$w_p$ [Mpc/$h$]")
        if col == n_bins - 1:
            ax.legend(fontsize=6)

        # --- ESD per survey ---
        for row, sname in enumerate(surveys, start=1):
            ax2 = axes[row, col]
            if sname not in b["surveys"]:
                ax2.text(0.5, 0.5, "no data", ha="center", va="center",
                         transform=ax2.transAxes, fontsize=8, color="0.5")
                if col == 0:
                    ax2.set_ylabel(fr"$\Delta\Sigma$ [{sname}]")
                continue
            R, ds_obs, icov_ds = b["surveys"][sname]
            ds_err = np.sqrt(np.diag(np.linalg.inv(icov_ds)))
            try:
                ds_mod = np.asarray(predictor.delta_sigma(
                    jnp.array(R), z, theta_cosmo, p)) / h
            except Exception:
                ds_mod = np.full_like(R, np.nan)
            color = survey_colors.get(sname, "C1")
            ax2.errorbar(R, ds_obs, yerr=ds_err, fmt="o", ms=3,
                         color="k", zorder=2)
            ax2.plot(R, ds_mod, "-", lw=2, color=color, zorder=3,
                     label=sname)
            ax2.set_xscale("log"); ax2.set_yscale("log")
            if col == 0:
                ylabel = fr"$\Delta\Sigma_{{\rm {sname}}}$ [$M_\odot\,h\,{{\rm pc}}^{{-2}}$]"
                ax2.set_ylabel(ylabel, fontsize=7)
            ax2.legend(fontsize=6, loc="lower left")

        axes[-1, col].set_xlabel(r"$r_p$ or $R$ [Mpc/$h$]", fontsize=8)

    chi2_tot = map_result.get("chi2", 0)
    ndof     = map_result.get("ndof", 1)
    fig.suptitle(
        f"ZM15 iHOD joint MAP — BGS mass bins\n"
        f"$\\chi^2/\\mathrm{{dof}} = {chi2_tot:.0f}/{ndof} = "
        f"{chi2_tot/max(ndof,1):.2f}$",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "map_bestfit.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_hod_shmr(bins, map_result, out_dir, z_eff=0.13):
    """Two extra figures at MAP: HOD occupation curves and SHMR vs literature.

    Parameters
    ----------
    bins       : list of bin dicts from load_bins
    map_result : dict from map_fit() / loaded JSON
    out_dir    : output directory
    z_eff      : effective redshift for literature SHMR curves
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from hod_mod.galaxies.hod import (
        _mstar_from_mh_zu15,
        n_cen_thresh_zu15,
        n_sat_thresh_zu15,
    )
    from hod_mod.galaxies.sham import smhm_moster13, smhm_behroozi13

    params = map_result["params"]
    log10mh = jnp.linspace(10.0, 15.5, 300)

    # ------------------------------------------------------------------ #
    # Figure 1 — HOD occupation: Ncen and Nsat vs Mh                     #
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))

    cmap = plt.get_cmap("plasma")
    n_bins = len(bins)
    colors = [cmap(i / max(n_bins - 1, 1)) for i in range(n_bins)]

    for b, col in zip(bins, colors):
        thresh = b["thresh"]
        nc = np.asarray(n_cen_thresh_zu15(
            log10mh,
            log10m_star_thresh=thresh,
            lg_m1h=params["lg_m1h"],
            lg_m0star=params["lg_m0star"],
            beta=params["beta"],
            delta=params["delta"],
            gamma=params["gamma"],
            sigma_lnmstar=params["sigma_lnmstar"],
            eta=params["eta"],
            fc=params["fc"],
        ))
        ns = np.asarray(n_sat_thresh_zu15(
            log10mh,
            log10m_star_thresh=thresh,
            lg_m1h=params["lg_m1h"],
            lg_m0star=params["lg_m0star"],
            beta=params["beta"],
            delta=params["delta"],
            gamma=params["gamma"],
            sigma_lnmstar=params["sigma_lnmstar"],
            eta=params["eta"],
            fc=params["fc"],
            bsat=params["bsat"],
            beta_sat=params["beta_sat"],
            bcut=params["bcut"],
            beta_cut=params["beta_cut"],
            alpha_sat=params["alpha_sat"],
        ))
        mh_np = np.asarray(log10mh)
        ax1.plot(mh_np, nc, "-",  color=col, lw=1.5,
                 label=fr"$M_*>{thresh}$" if b is bins[0] else f">{thresh}")
        ax1.plot(mh_np, ns, "--", color=col, lw=1.5)

    # dummy lines for legend
    ax1.plot([], [], "k-",  lw=1.5, label=r"$N_{\rm cen}$ (solid)")
    ax1.plot([], [], "k--", lw=1.5, label=r"$N_{\rm sat}$ (dashed)")

    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 30)
    ax1.set_xlim(10, 15.5)
    ax1.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot\,h^{-1}])$")
    ax1.set_ylabel(r"$\langle N \rangle (M_h)$")
    ax1.set_title("ZM15 iHOD MAP — occupation functions")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    fig1.tight_layout()
    out1 = os.path.join(out_dir, "hod_occupation.pdf")
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"  → {out1}")

    # ------------------------------------------------------------------ #
    # Figure 2 — SHMR: MAP vs ZM15 published vs Moster+13 vs Behroozi+13 #
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    mh_np = np.asarray(log10mh)

    # MAP result
    mstar_map = np.asarray(_mstar_from_mh_zu15(
        log10mh,
        lg_m1h=params["lg_m1h"],
        lg_m0star=params["lg_m0star"],
        beta=params["beta"],
        delta=params["delta"],
        gamma=params["gamma"],
    ))
    ax2.plot(mh_np, mstar_map, "-", color="C0", lw=2.5, label="ZM15 iHOD MAP (this work)")

    # ZM15 published reference
    mstar_zm15pub = np.asarray(_mstar_from_mh_zu15(
        log10mh,
        lg_m1h=PUBLISHED["lg_m1h"][0],
        lg_m0star=PUBLISHED["lg_m0star"][0],
        beta=PUBLISHED["beta"][0],
        delta=PUBLISHED["delta"][0],
        gamma=PUBLISHED["gamma"][0],
    ))
    ax2.plot(mh_np, mstar_zm15pub, "--", color="C0", lw=1.5,
             label="ZM15 published (Table 2)")

    # Moster+13 at z_eff
    mstar_mo13 = np.asarray(smhm_moster13(log10mh, z_eff))
    ax2.plot(mh_np, mstar_mo13, "-.", color="C1", lw=1.5,
             label=fr"Moster+13 ($z={z_eff:.2f}$)")

    # Behroozi+13 at z_eff
    mstar_be13 = np.asarray(smhm_behroozi13(log10mh, z_eff))
    ax2.plot(mh_np, mstar_be13, ":", color="C2", lw=1.5,
             label=fr"Behroozi+13 ($z={z_eff:.2f}$)")

    # Shade bin threshold regions
    for b, col in zip(bins, colors):
        ax2.axhline(b["thresh"], color=col, lw=0.6, ls="--", alpha=0.5)

    ax2.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot\,h^{-1}])$")
    ax2.set_ylabel(r"$\log_{10}(M_*\,/\,M_\odot)$")
    ax2.set_xlim(10, 15.5)
    ax2.set_ylim(7, 13)
    ax2.set_title(r"Stellar-to-halo mass relation")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    out2 = os.path.join(out_dir, "shmr.pdf")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"  → {out2}")


# ---------------------------------------------------------------------------
# Stellar mass function, satellite fraction, and combined montage
# ---------------------------------------------------------------------------

def _discover_smf_file(smf_data_dir: str) -> str | None:
    """Locate the widest-coverage observed SMF file (lowest M* threshold).

    Globs ``<smf_data_dir>/BGS_Mstar*/*joint_smf*.h5`` and returns the file whose
    threshold (parsed from the ``BGS_Mstar<thr>`` directory name) is lowest — the
    one spanning the largest stellar-mass range.  Returns ``None`` if none found.
    """
    paths = glob.glob(os.path.join(smf_data_dir, "BGS_Mstar*", "*joint_smf*.h5"))
    if not paths:
        return None

    def _thr(p: str) -> float:
        m = re.search(r"BGS_Mstar([0-9.]+)", p)
        return float(m.group(1)) if m else np.inf

    return min(paths, key=_thr)


def load_observed_smf(smf_file: str, z_fallback: float = 0.13) -> dict:
    """Read the observed stellar mass function from a sum_stat joint file.

    This SMF is **not** part of the joint fit — it is an independent observable
    used purely for comparison.  Returns physical ``log10(M*/M_sun)`` centres and
    ``Phi`` [h^3 Mpc^-3 dex^-1], with non-positive / non-finite bins dropped.
    """
    from hod_mod.data_io.sum_stat_reader import SumStatReader

    reader = SumStatReader.from_hdf5(smf_file)
    s = reader.smf()
    log10mstar = np.asarray(s["log10mstar"], dtype=float)
    phi        = np.asarray(s["phi"], dtype=float)
    phi_err_in = s.get("phi_err")
    good = np.isfinite(phi) & (phi > 0)

    phi_err = None
    if phi_err_in is not None:
        phi_err_in = np.asarray(phi_err_in, dtype=float)
        if phi_err_in.shape == phi.shape:
            phi_err = phi_err_in[good]

    attrs = s.get("attrs", {})
    z = attrs.get("z_mean", None)
    try:
        z = float(z)
        if not np.isfinite(z):
            raise ValueError
    except (TypeError, ValueError):
        try:
            z = 0.5 * (float(attrs.get("z_min")) + float(attrs.get("z_max")))
        except (TypeError, ValueError):
            z = z_fallback

    return {
        "log10mstar": log10mstar[good],
        "phi":        phi[good],
        "phi_err":    phi_err,
        "z":          float(z),
        "h":          float(reader.h()),
    }


def predict_smf(predictor, theta_cosmo, params, z, log10mstar_grid) -> np.ndarray:
    """Model stellar mass function ``Phi(M*)`` [h^3 Mpc^-3 dex^-1].

    Built from the predictor's cumulative number density: for each threshold
    ``M*_i``, ``N(>M*_i) = predictor.n_gal`` with ``log10m_star_thresh=M*_i`` and
    **no** ``log10m_star_max`` (so ``nc_ns`` returns the cumulative occupation).
    The SMF is the negative derivative ``Phi = -dN/dlog10M*``.  Native predictor
    units are already h^3 Mpc^-3, directly comparable to the observed SMF.
    """
    grid  = np.asarray(log10mstar_grid, dtype=float)
    n_cum = np.empty_like(grid)
    for i, mthr in enumerate(grid):
        p = dict(params)
        p["log10m_star_thresh"] = float(mthr)
        p.pop("log10m_star_max", None)
        try:
            n_cum[i] = float(predictor.n_gal(z, theta_cosmo, p))
        except Exception:
            n_cum[i] = np.nan
    phi = -np.gradient(n_cum, grid)        # N decreases with threshold
    phi[~(phi > 0)] = np.nan
    return phi


def _n_cen_n_sat(predictor, z, theta_cosmo, params) -> tuple[float, float]:
    """Central and satellite number densities [h^3 Mpc^-3] for *params*.

    Mirrors :meth:`FullHaloModelPrediction.n_gal` (clustering.py) but keeps the
    central/satellite occupation integrals separate.
    """
    import jax
    hod    = predictor._hod
    m_grid = np.asarray(hod._m_grid, dtype=float)
    dndm   = np.asarray(hod._hmf.dndm(hod._m_grid, float(z), theta_cosmo), dtype=float)
    with jax.disable_jit():
        nc, ns = hod.nc_ns(hod._log10m_grid, params)
    n_cen = float(np.trapezoid(dndm * np.asarray(nc, dtype=float), m_grid))
    n_sat = float(np.trapezoid(dndm * np.asarray(ns, dtype=float), m_grid))
    return n_cen, n_sat


def plot_smf(bins, predictor, theta_cosmo, h, map_result, obs_smf, out_dir):
    """SMF figure: model curve + per-bin model points vs observed (not fitted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params  = map_result["params"]
    z_model = obs_smf["z"] if obs_smf is not None else (bins[0].get("z") or 0.13)
    z_fit   = bins[0].get("z") or 0.13

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # observed (independent, not fitted)
    if obs_smf is not None:
        ax.errorbar(obs_smf["log10mstar"], obs_smf["phi"], yerr=obs_smf["phi_err"],
                    fmt="o", ms=4, color="k", zorder=3,
                    label=fr"observed (sum_stat, not fitted, $z={z_model:.2f}$)")

    # model continuous curve
    grid    = np.linspace(9.5, 12.0, 60)
    phi_mod = predict_smf(predictor, theta_cosmo, params, z_model, grid)
    ax.plot(grid, phi_mod, "-", color="C0", lw=2, zorder=2,
            label="ZM15 iHOD MAP (model)")

    # model per fit-bin points: n_gal(bin) / Δlog10M*
    mb, phib = [], []
    for b in bins:
        p = dict(params)
        p["log10m_star_thresh"] = b["thresh"]
        p["log10m_star_max"]    = b["max"]
        width = b["max"] - b["thresh"]
        try:
            ng = float(predictor.n_gal(z_model, theta_cosmo, p))
            if width > 0 and np.isfinite(ng):
                mb.append(0.5 * (b["thresh"] + b["max"]))
                phib.append(ng / width)
        except Exception:
            pass
    if mb:
        ax.plot(mb, phib, "s", color="C1", ms=6, zorder=4,
                label=r"model fit-bin $\bar n/\Delta\log M_*$")

    ax.set_yscale("log")
    ax.set_xlim(9.5, 12.0)
    ax.set_xlabel(r"$\log_{10}(M_*\,/\,M_\odot)$")
    ax.set_ylabel(r"$\Phi$ [$h^3\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}$]")
    ax.set_title(f"Stellar mass function — model vs observed (not fitted)\n"
                 fr"model at $z={z_model:.2f}$; fit bins at $z\approx{z_fit:.2f}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, "stellar_mass_function.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_satellite_fraction(predictor, theta_cosmo, map_result, out_dir, z_eff,
                            mstar_grid=None):
    """Model satellite fraction f_sat(>M*) vs stellar-mass threshold (ZM15-style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = map_result["params"]
    if mstar_grid is None:
        mstar_grid = np.linspace(9.5, 11.6, 30)
    fsat = np.full(mstar_grid.shape, np.nan)
    for i, mthr in enumerate(mstar_grid):
        p = dict(params)
        p["log10m_star_thresh"] = float(mthr)
        p.pop("log10m_star_max", None)
        try:
            nc, ns = _n_cen_n_sat(predictor, z_eff, theta_cosmo, p)
            tot = nc + ns
            fsat[i] = ns / tot if tot > 0 else np.nan
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(mstar_grid, fsat, "-", color="C3", lw=2)
    ax.set_xlabel(r"$\log_{10}(M_*\,/\,M_\odot)$ threshold")
    ax.set_ylabel(r"satellite fraction $f_{\rm sat}(>M_*)$")
    top = np.nanmax(fsat) if np.any(np.isfinite(fsat)) else 0.5
    ax.set_ylim(0, max(0.05, top * 1.15))
    ax.set_title(fr"ZM15 iHOD MAP — satellite fraction ($z={z_eff:.2f}$)")
    fig.tight_layout()
    out = os.path.join(out_dir, "satellite_fraction.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_montage(bins, predictor, theta_cosmo, h, pi_max_h, map_result, obs_smf,
                 out_dir, surveys, ref_survey="HSC"):
    """Combined 2×2 publication montage: wp, ΔΣ, SMF, SHMR (ZM15 layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from hod_mod.galaxies.hod import _mstar_from_mh_zu15

    params = map_result["params"]
    cmap   = plt.get_cmap("viridis")
    n_bins = len(bins)
    colors = [cmap(i / max(n_bins - 1, 1)) for i in range(n_bins)]

    # reference lensing survey for panel B
    ref = ref_survey if any(ref_survey in b["surveys"] for b in bins) else None
    if ref is None:
        for s in surveys:
            if any(s in b["surveys"] for b in bins):
                ref = s
                break

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- Panel A: wp(rp) all bins ---
    for b, col in zip(bins, colors):
        z = b.get("z") or 0.13
        p = dict(params)
        p["log10m_star_thresh"] = b["thresh"]
        p["log10m_star_max"]    = b["max"]
        rp     = b["rp"]
        wp_err = np.sqrt(np.diag(np.linalg.inv(b["icov_wp"])))
        try:
            wp_mod = np.asarray(predictor.wp(jnp.array(rp), pi_max_h, z, theta_cosmo, p))
        except Exception:
            wp_mod = np.full_like(rp, np.nan)
        axA.errorbar(rp, b["wp_obs"], yerr=wp_err, fmt="o", ms=3, color=col, alpha=0.6)
        axA.plot(rp, wp_mod, "-", color=col, lw=1.5, label=b["label"])
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel(r"$r_p$ [Mpc/$h$]"); axA.set_ylabel(r"$w_p$ [Mpc/$h$]")
    axA.set_title("Projected clustering")
    axA.legend(fontsize=6, ncol=2, title=r"$\log M_*$ bin")

    # --- Panel B: ΔΣ(R) all bins, reference survey ---
    if ref is not None:
        for b, col in zip(bins, colors):
            if ref not in b["surveys"]:
                continue
            z = b.get("z") or 0.13
            p = dict(params)
            p["log10m_star_thresh"] = b["thresh"]
            p["log10m_star_max"]    = b["max"]
            R, ds_obs, icov = b["surveys"][ref]
            ds_err = np.sqrt(np.diag(np.linalg.inv(icov)))
            try:
                ds_mod = np.asarray(predictor.delta_sigma(jnp.array(R), z, theta_cosmo, p)) / h
            except Exception:
                ds_mod = np.full_like(R, np.nan)
            axB.errorbar(R, ds_obs, yerr=ds_err, fmt="o", ms=3, color=col, alpha=0.6)
            axB.plot(R, ds_mod, "-", color=col, lw=1.5)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel(r"$R$ [Mpc/$h$]")
    axB.set_ylabel(r"$\Delta\Sigma$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
    axB.set_title(f"Galaxy-galaxy lensing ({ref or 'n/a'})")

    # --- Panel C: SMF model vs observed ---
    z_model = obs_smf["z"] if obs_smf is not None else (bins[0].get("z") or 0.13)
    if obs_smf is not None:
        axC.errorbar(obs_smf["log10mstar"], obs_smf["phi"], yerr=obs_smf["phi_err"],
                     fmt="o", ms=4, color="k", label="observed (not fitted)")
    grid = np.linspace(9.5, 12.0, 60)
    axC.plot(grid, predict_smf(predictor, theta_cosmo, params, z_model, grid),
             "-", color="C0", lw=2, label="model")
    axC.set_yscale("log"); axC.set_xlim(9.5, 12.0)
    axC.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    axC.set_ylabel(r"$\Phi$ [$h^3{\rm Mpc}^{-3}{\rm dex}^{-1}$]")
    axC.set_title("Stellar mass function")
    axC.legend(fontsize=8)

    # --- Panel D: SHMR ---
    log10mh = jnp.linspace(10.5, 15.0, 200)
    mh      = np.asarray(log10mh)
    mstar = np.asarray(_mstar_from_mh_zu15(
        log10mh, lg_m1h=params["lg_m1h"], lg_m0star=params["lg_m0star"],
        beta=params["beta"], delta=params["delta"], gamma=params["gamma"]))
    axD.plot(mh, mstar, "-", color="C0", lw=2, label="ZM15 iHOD MAP")
    mstar_pub = np.asarray(_mstar_from_mh_zu15(
        log10mh, lg_m1h=PUBLISHED["lg_m1h"][0], lg_m0star=PUBLISHED["lg_m0star"][0],
        beta=PUBLISHED["beta"][0], delta=PUBLISHED["delta"][0], gamma=PUBLISHED["gamma"][0]))
    axD.plot(mh, mstar_pub, "--", color="C0", lw=1.2, label="ZM15 published")
    axD.set_xlabel(r"$\log_{10}(M_h/[M_\odot h^{-1}])$")
    axD.set_ylabel(r"$\log_{10}(M_*/M_\odot)$")
    axD.set_title("Stellar-to-halo mass relation")
    axD.legend(fontsize=8)

    chi2 = map_result.get("chi2", 0); ndof = map_result.get("ndof", 1)
    fig.suptitle(
        f"ZM15 iHOD joint fit — BGS LS10   "
        f"$\\chi^2/\\mathrm{{dof}}={chi2:.0f}/{ndof}={chi2/max(ndof,1):.2f}$",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(out_dir, "zm15_montage.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_all(bins, predictor, theta_cosmo, h, pi_max_h, map_result, obs_smf,
             out_dir, surveys, z_eff):
    """Generate the full ZM15-style figure set from a MAP result."""
    plot_map(bins, predictor, theta_cosmo, h, pi_max_h, map_result, out_dir, surveys)
    plot_hod_shmr(bins, map_result, out_dir=out_dir, z_eff=z_eff)
    plot_smf(bins, predictor, theta_cosmo, h, map_result, obs_smf, out_dir)
    plot_satellite_fraction(predictor, theta_cosmo, map_result, out_dir, z_eff=z_eff)
    plot_montage(bins, predictor, theta_cosmo, h, pi_max_h, map_result, obs_smf,
                 out_dir, surveys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.expanduser(
        "~/software/sum_stat/data/BGS_Mstar10_massbins"),
        help="Directory of per-bin sum_stat joint HDF5 files")
    p.add_argument("--surveys", nargs="*", default=["HSC", "DES", "KIDS"],
                   help="Lensing surveys to include (HSC DES KIDS). "
                        "Pass --surveys with no arguments to fit wp+n_gal only.")
    p.add_argument("--mode", choices=["map", "mcmc", "both"], default="both")
    p.add_argument("--rp-min", type=float, default=0.1, help="wp r_p min [Mpc/h]")
    p.add_argument("--rp-max", type=float, default=30.0, help="wp r_p max [Mpc/h]")
    p.add_argument("--R-min",  type=float, default=0.1, help="ESD R min [Mpc/h]")
    p.add_argument("--R-max",  type=float, default=30.0, help="ESD R max [Mpc/h]")
    p.add_argument("--z-eff",  type=float, default=0.13,
                   help="Fallback effective redshift, used only for bins whose "
                        "file lacks a measured z_mean (default 0.13)")
    p.add_argument("--pi-max-mpc", type=float, default=100.0,
                   help="wp pi_max in physical Mpc (sum_stat value; converted to Mpc/h)")
    p.add_argument("--hmf-backend", default="tinker08")
    p.add_argument("--smf-file", default=None,
                   help="Observed SMF file (sum_stat joint *_smf_* HDF5) used only "
                        "for comparison (NOT fitted). Default: auto-discover the "
                        "widest-coverage BGS_Mstar* file under --smf-data-dir.")
    p.add_argument("--smf-data-dir", default=os.path.expanduser(
        "~/software/sum_stat/data"),
        help="Root searched for an observed SMF file when --smf-file is omitted")
    p.add_argument("--ng-frac-err-floor", type=float, default=0.05,
                   help="Minimum fractional error on n_g (default 0.05)")
    p.add_argument("--gaussian-prior", action="store_true",
                   help="Add a Gaussian prior from the published ZM15 values")
    p.add_argument("--n-walkers", type=int, default=32)
    p.add_argument("--n-burnin",  type=int, default=500)
    p.add_argument("--n-steps",   type=int, default=2000)
    p.add_argument("--out-dir", default=os.path.join(
        _REPO_ROOT, "results/bgs_zm15_joint"))
    p.add_argument("--force-mcmc", action="store_true",
                   help="Rerun MCMC even if a chain already exists")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip fitting; load existing MAP result and generate plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print("Loading mass-bin measurements ...")
    bins, h = load_bins(
        args.data_dir, args.surveys, args.rp_min, args.rp_max,
        args.R_min, args.R_max, args.ng_frac_err_floor)
    # Use the per-bin z_mean from the file; fall back to --z-eff if absent.
    n_missing = 0
    for b in bins:
        if b.get("z") is None:
            b["z"] = args.z_eff
            n_missing += 1
    if n_missing:
        print(f"  {n_missing}/{len(bins)} bins had no z_mean in file — "
              f"using --z-eff={args.z_eff} for those")
    print(f"Loaded {len(bins)} bins.  h={h:.4f}  "
          f"z_eff per bin: {[round(b['z'], 3) for b in bins]}")

    print("Building Zu & Mandelbaum 2015 predictor ...")
    predictor, theta_cosmo = build_predictor(args.hmf_backend)

    fitter = JointZM15(
        bins, predictor, theta_cosmo, h=h, z=args.z_eff,
        pi_max_h=args.pi_max_mpc * h, gaussian_prior=args.gaussian_prior)

    # Observed SMF (independent — NOT part of the fit), loaded for comparison.
    smf_file = args.smf_file or _discover_smf_file(args.smf_data_dir)
    obs_smf = None
    if smf_file and os.path.exists(smf_file):
        try:
            obs_smf = load_observed_smf(smf_file, z_fallback=args.z_eff)
            print(f"Observed SMF (not fitted): {os.path.basename(smf_file)}  "
                  f"({len(obs_smf['log10mstar'])} pts, z={obs_smf['z']:.3f})")
        except Exception as exc:
            print(f"  [warn] could not load observed SMF from {smf_file}: {exc}")
    else:
        print(f"  [warn] no observed SMF file found under {args.smf_data_dir}"
              f"/BGS_Mstar*/*joint_smf*.h5 — SMF panels will be model-only")

    map_json = os.path.join(args.out_dir, "map_result.json")
    map_result = None

    if args.plot_only:
        if not os.path.exists(map_json):
            raise FileNotFoundError(
                f"No MAP result found at {map_json}. Run --mode map first.")
        with open(map_json) as fh:
            map_result = json.load(fh)
        print(f"Loaded MAP result: chi2/dof={map_result['chi2_per_dof']:.3f}")
        plot_all(bins, predictor, theta_cosmo, h,
                 pi_max_h=args.pi_max_mpc * h, map_result=map_result,
                 obs_smf=obs_smf, out_dir=args.out_dir,
                 surveys=args.surveys, z_eff=args.z_eff)
        print(f"\nAll done in {(time.time() - t0) / 60:.1f} min.")
        return

    if args.mode in ("map", "both"):
        print("\n=== MAP optimisation (Powell) ===")
        map_result = fitter.map_fit()
        with open(map_json, "w") as fh:
            json.dump(map_result, fh, indent=2)
        print(f"\nchi2/ndof = {map_result['chi2']:.1f} / {map_result['ndof']} "
              f"= {map_result['chi2_per_dof']:.3f}")
        print(f"{'param':14s} {'MAP':>10s} {'published':>12s}")
        for name in FREE_NAMES:
            pub = PUBLISHED[name][0]
            print(f"{name:14s} {map_result['params'][name]:10.4f} {pub:12.4f}")
        print(f"MAP result -> {map_json}")
        plot_all(bins, predictor, theta_cosmo, h,
                 pi_max_h=args.pi_max_mpc * h, map_result=map_result,
                 obs_smf=obs_smf, out_dir=args.out_dir,
                 surveys=args.surveys, z_eff=args.z_eff)

    if args.mode in ("mcmc", "both"):
        chain = os.path.join(args.out_dir, "flatchain.npz")
        if os.path.exists(chain) and not args.force_mcmc:
            print(f"\n[skip] chain exists: {chain} (use --force-mcmc to rerun)")
        else:
            print("\n=== MCMC sampling (emcee) ===")
            if map_result is not None:
                x_start = np.array(map_result["theta"])
            elif os.path.exists(map_json):
                with open(map_json) as fh:
                    x_start = np.array(json.load(fh)["theta"])
                print(f"  loaded MAP starting point from {map_json}")
            else:
                x_start = None
            fitter.sample(args.out_dir, args.n_walkers, args.n_burnin,
                          args.n_steps, x_start=x_start)

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
