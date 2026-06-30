"""Joint MAP fit to wp(rp) + ΔΣ_HSC(R) + n_g for all four Lange+2025 DESI DR1 samples.

Fits BGS2, BGS3, LRG1, LRG2 simultaneously with:
  - Shared:     Omega_m, S8  (Planck 2018 Gaussian priors, applied once)
  - Per-sample: log10mmin, sigma_logm, log10m0, log10m1, alpha,
                f_Gamma, A_cen, A_sat  (8 HOD params × 4 samples)

Total:  2 + 4×8 = 34 free parameters
Data:   4×(11 wp + 6 DS + 1 ng) = 72 points
ndof:   72 − 34 = 38

The shared _CachedPkLinear evaluates CAMB once per unique (Omega_m, S8)
value and interpolates thereafter, keeping the joint likelihood fast.

Usage
-----
    python hod_mod/scripts/benchmarks/run_benchmark_hsc_wp_lange25.py [--plot] [--mcmc]
"""

import argparse
import json
import os
import sys

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, _REPO_ROOT)

from hod_mod.paths import results_root
from hod_mod.fitting import (
    JointFitter,
    _CachedPkLinear,
    _assemble_hod_params,
    load_config,
)
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.fitting import HOD_MODELS
from hod_mod.scripts.benchmarks.benchmark_plots import (
    _COL_DATA, _COL_MAP, _COL_PUB,
    _PARAM_LATEX,
    load_flatchain,
    mcmc_bands,
    add_bands,
    residual_panel,
)

# ---------------------------------------------------------------------------
# Sample definitions
# ---------------------------------------------------------------------------

SAMPLES = ["bgs2", "bgs3", "lrg1", "lrg2"]

SAMPLE_CONFIGS = {
    "bgs2": "configs/benchmarks/benchmark_lange2025_bgs2_bwpd_hsc.yml",
    "bgs3": "configs/benchmarks/benchmark_lange2025_bgs3_bwpd_hsc.yml",
    "lrg1": "configs/benchmarks/benchmark_lange2025_lrg1_bwpd_hsc.yml",
    "lrg2": "configs/benchmarks/benchmark_lange2025_lrg2_bwpd_hsc.yml",
}

# Per-sample published HOD parameters (from metadata.json galaxy_hod_parameters)
PUBLISHED_HOD = {
    "bgs2": {"log10mmin": 12.1,  "sigma_logm": 0.15, "log10m0": 11.07, "log10m1": 13.4,
             "alpha": 0.95, "f_Gamma": 1.0,  "A_cen":  1.0, "A_sat": -0.1},
    "bgs3": {"log10mmin": 12.5,  "sigma_logm": 0.1,  "log10m0": 12.6,  "log10m1": 13.8,
             "alpha": 0.9,  "f_Gamma": 1.0,  "A_cen":  1.0, "A_sat": -0.8},
    "lrg1": {"log10mmin": 12.7,  "sigma_logm": 0.1,  "log10m0": 12.4,  "log10m1": 14.2,
             "alpha": 1.2,  "f_Gamma": 0.8,  "A_cen": -1.0, "A_sat": -1.0},
    "lrg2": {"log10mmin": 12.7,  "sigma_logm": 0.1,  "log10m0": 12.4,  "log10m1": 14.1,
             "alpha": 1.2,  "f_Gamma": 0.8,  "A_cen": -1.0, "A_sat": -1.0},
}

# Published cosmological constraints (HSC-Y3 + wp joint analysis)
PUBLISHED_COSMO = {"Omega_m": 0.303, "S8": 0.793}

HOD_PARAM_NAMES = ["log10mmin", "sigma_logm", "log10m0", "log10m1",
                   "alpha", "f_Gamma", "A_cen", "A_sat"]

OUTPUT_DIR = os.path.join(results_root(),
                          "benchmarks/lange2025/all_samples_hsc_wp/")


# ---------------------------------------------------------------------------
# MultiSampleJointFitter
# ---------------------------------------------------------------------------

class MultiSampleJointFitter:
    """Joint fitter for wp + ΔΣ_HSC + n_g across four Lange+2025 DESI samples.

    Shared cosmological parameters (Omega_m, S8) enter the power spectrum
    computation via a single _CachedPkLinear instance to avoid redundant
    CAMB calls.  Per-sample HOD parameters are optimized independently.

    Free parameter vector layout
    ----------------------------
    [Omega_m, S8,
     log10mmin_bgs2, sigma_logm_bgs2, ..., A_sat_bgs2,   # 8 HOD params
     log10mmin_bgs3, ..., A_sat_bgs3,
     log10mmin_lrg1, ..., A_sat_lrg1,
     log10mmin_lrg2, ..., A_sat_lrg2]
    Total: 2 + 4×8 = 34 free params
    """

    # Planck 2018 priors for shared cosmological parameters
    _COSMO_PRIORS = {
        "Omega_m": (0.3153, 0.0073, 0.279, 0.352),   # (mean, sigma, lo, hi)
        "S8":      (0.832,  0.0114, 0.775, 0.889),
    }

    def __init__(self, sample_configs: dict[str, str]):
        """Build per-sample fitters sharing a single CachedPkLinear.

        Parameters
        ----------
        sample_configs : dict
            Mapping sample name → path to its benchmark YAML config.
        """
        # Shared, cached power-spectrum evaluator (one CAMB call per unique cosmo)
        pk_base = LinearPowerSpectrum()
        self._pk_shared = _CachedPkLinear(pk_base)

        # Per-sample JointFitter instances
        self.fitters: dict[str, JointFitter] = {}
        for name, cfg_path in sample_configs.items():
            cfg_abs = os.path.join(_REPO_ROOT, cfg_path)
            cfg     = load_config(cfg_abs)
            f       = JointFitter(cfg)
            # Replace per-instance pk_lin with the shared cache, then rebuild
            f._pk_lin = self._pk_shared
            f._build_predictor()
            self.fitters[name] = f

        # Baseline cosmology (all samples use the same Planck 2018 cosmo)
        first = next(iter(self.fitters.values()))
        self._theta_cosmo_base = dict(first.theta_cosmo)

        # Build combined free-parameter catalogue
        self._build_param_catalogue()

    # ------------------------------------------------------------------
    # Parameter structure

    def _build_param_catalogue(self):
        """Construct the combined parameter name list, bounds, and init vector."""
        names  = []
        bounds = {}
        x0     = {}

        # Shared cosmological parameters
        for pname, (mean, _, lo, hi) in self._COSMO_PRIORS.items():
            names.append(pname)
            bounds[pname] = (lo, hi)
            x0[pname]     = mean

        # Per-sample HOD parameters
        for sname, fitter in self.fitters.items():
            cfg = fitter.config
            for hname in HOD_PARAM_NAMES:
                key = f"{hname}_{sname}"
                names.append(key)
                if hname in cfg.param_bounds:
                    bounds[key] = cfg.param_bounds[hname]
                else:
                    lo_b, hi_b = cfg.param_bounds.get(hname, (0.01, 15.0))
                    bounds[key] = (lo_b, hi_b)
                x0[key] = cfg.param_init.get(hname, 0.0)

        self.free_params  = names
        self.param_bounds = bounds
        self._x0          = np.array([x0[p] for p in names])

    # ------------------------------------------------------------------
    # Cosmology helper

    def _build_theta_cosmo(self, Omega_m: float, S8: float) -> dict:
        """Convert (Omega_m, S8) → theta_cosmo dict for predictor calls."""
        tc = dict(self._theta_cosmo_base)
        tc["Omega_m"]   = Omega_m
        tc["Omega_cdm"] = Omega_m - float(tc["Omega_b"])
        sigma8     = S8 * np.sqrt(0.3 / Omega_m)
        sigma8_fid = float(self._theta_cosmo_base.get("sigma8", 0.8111))
        ln10As_fid = float(self._theta_cosmo_base["ln10^{10}A_s"])
        tc["ln10^{10}A_s"] = ln10As_fid + 2.0 * np.log(sigma8 / sigma8_fid)
        return tc

    # ------------------------------------------------------------------
    # Log-probability

    def _log_prob(self, theta_vec: np.ndarray) -> float:
        """Combined log-posterior: Planck priors (once) + sum of sample likelihoods."""
        params = dict(zip(self.free_params, theta_vec))

        # --- bounds check ---
        for p, val in params.items():
            lo, hi = self.param_bounds[p]
            if not (lo <= val <= hi):
                return -np.inf

        # --- Planck prior on shared cosmo (applied ONCE) ---
        log_pi = 0.0
        for pname, (mean, sigma, lo, hi) in self._COSMO_PRIORS.items():
            log_pi += -0.5 * ((params[pname] - mean) / sigma) ** 2

        Omega_m = params["Omega_m"]
        S8      = params["S8"]
        try:
            tc = self._build_theta_cosmo(Omega_m, S8)
        except Exception:
            return -np.inf

        # --- per-sample likelihood ---
        chi2_total = 0.0
        for sname, fitter in self.fitters.items():
            hod_p = {h: params[f"{h}_{sname}"] for h in HOD_PARAM_NAMES}
            try:
                wp_pred = np.asarray(fitter.predictor.wp(
                    jnp.array(fitter.rp_arr), fitter.config.pi_max,
                    fitter.config.z, tc, hod_p,
                ))
                ds_pred = np.asarray(fitter.predictor.delta_sigma(
                    jnp.array(fitter.R_arr), fitter.config.z, tc, hod_p,
                ))
                ng_pred = float(fitter.predictor.n_gal(fitter.config.z, tc, hod_p))
            except Exception:
                return -np.inf

            res_wp = wp_pred - fitter.wp_obs
            res_ds = ds_pred - fitter.ds_obs
            res_ng = (ng_pred - fitter.config.ng_obs) / (
                fitter.config.ng_frac_err * fitter.config.ng_obs
            )
            chi2_total += float(res_wp @ fitter.icov_wp @ res_wp)
            chi2_total += float(res_ds @ fitter.icov_ds @ res_ds)
            chi2_total += float(res_ng ** 2)

        return log_pi - 0.5 * chi2_total

    # ------------------------------------------------------------------
    # MAP fit

    def map_fit(self) -> dict:
        """Powell MAP optimisation of the combined 34-parameter likelihood.

        Returns
        -------
        dict
            Keys: ``params``, ``chi2``, ``ndof``, ``chi2_per_sample``,
            ``success``, ``message``.
        """
        print(f"Starting MAP fit: {len(self.free_params)} free params, "
              f"{self._n_data()} data points  (ndof={self._n_data()-len(self.free_params)})")
        result = minimize(
            lambda x: -self._log_prob(x),
            self._x0,
            method="Powell",
            options={"maxiter": 100000, "xtol": 1e-5, "ftol": 1e-5, "disp": False},
        )
        best_params = dict(zip(self.free_params, result.x))

        Omega_m = best_params["Omega_m"]
        S8      = best_params["S8"]
        tc      = self._build_theta_cosmo(Omega_m, S8)

        chi2_total   = 0.0
        chi2_samples = {}
        for sname, fitter in self.fitters.items():
            hod_p = {h: best_params[f"{h}_{sname}"] for h in HOD_PARAM_NAMES}
            wp_pred = np.asarray(fitter.predictor.wp(
                jnp.array(fitter.rp_arr), fitter.config.pi_max,
                fitter.config.z, tc, hod_p,
            ))
            ds_pred = np.asarray(fitter.predictor.delta_sigma(
                jnp.array(fitter.R_arr), fitter.config.z, tc, hod_p,
            ))
            ng_pred = float(fitter.predictor.n_gal(fitter.config.z, tc, hod_p))
            c_wp = float((wp_pred - fitter.wp_obs) @ fitter.icov_wp @ (wp_pred - fitter.wp_obs))
            c_ds = float((ds_pred - fitter.ds_obs) @ fitter.icov_ds @ (ds_pred - fitter.ds_obs))
            c_ng = float(((ng_pred - fitter.config.ng_obs) /
                          (fitter.config.ng_frac_err * fitter.config.ng_obs)) ** 2)
            chi2_samples[sname] = {"wp": c_wp, "ds": c_ds, "ng": c_ng,
                                   "total": c_wp + c_ds + c_ng}
            chi2_total += c_wp + c_ds + c_ng

        n_data = self._n_data()
        ndof   = n_data - len(self.free_params)
        return {
            "params":          best_params,
            "chi2":            chi2_total,
            "ndof":            ndof,
            "chi2_ndof":       chi2_total / ndof if ndof > 0 else float("nan"),
            "chi2_per_sample": chi2_samples,
            "success":         result.success,
            "message":         result.message,
        }

    def _n_data(self) -> int:
        """Total number of data points across all samples."""
        n = 0
        for f in self.fitters.values():
            n += len(f.rp_arr) + len(f.R_arr) + 1   # wp + DS + ng
        return n

    # ------------------------------------------------------------------
    # MCMC sampling

    _BURNIN_CHUNK = 50  # checkpoint burn-in every N steps

    def run_mcmc(self, map_params: dict, output_dir: str,
                 n_walkers: int = 80, n_burnin: int = 500,
                 n_steps: int = 2000, progress: bool = True):
        """Run emcee sampler with checkpoint/resume support.

        State files written to *output_dir*:

        ``chain.h5``
            emcee HDFBackend — production chain, appended every step.
            Resumable natively: if iteration < n_steps the sampler
            continues from the last stored position.
        ``burnin_pos.npz``
            Walker positions saved every _BURNIN_CHUNK steps during
            burn-in.  Deleted once production starts.
        ``flatchain.npz``
            Final flat chain + param_names written on completion.

        Parameters
        ----------
        map_params : dict
            MAP best-fit parameter dict (keys = ``self.free_params``).
        output_dir : str
            Directory for checkpoint and output files.
        n_walkers : int
            Number of emcee walkers (auto-raised to 2×n_params if needed).
        n_burnin : int
            Burn-in steps per walker (discarded).
        n_steps : int
            Production steps per walker.
        progress : bool
            Show tqdm progress bar.

        Returns
        -------
        emcee.backends.HDFBackend
        """
        import emcee

        n_free = len(self.free_params)
        if n_walkers < 2 * n_free:
            n_walkers = 2 * n_free
            print(f"n_walkers raised to {n_walkers} (must be ≥ 2×n_params={2*n_free})")

        os.makedirs(output_dir, exist_ok=True)
        chain_path     = os.path.join(output_dir, "chain.h5")
        burnin_path    = os.path.join(output_dir, "burnin_pos.npz")
        flatchain_path = os.path.join(output_dir, "flatchain.npz")

        backend = emcee.backends.HDFBackend(chain_path)

        # ------------------------------------------------------------------
        # Case 1: production chain already started — resume it
        # ------------------------------------------------------------------
        if os.path.exists(chain_path) and backend.iteration > 0:
            n_done = backend.iteration
            if n_done >= n_steps:
                print(f"Production chain already complete ({n_done}/{n_steps} steps).")
            else:
                remaining = n_steps - n_done
                print(f"Resuming production: {n_done}/{n_steps} steps done, "
                      f"{remaining} remaining…")
                sampler = emcee.EnsembleSampler(
                    n_walkers, n_free, self._log_prob, backend=backend)
                sampler.run_mcmc(None, remaining, progress=progress)
            np.savez(flatchain_path,
                     flatchain=backend.get_chain(flat=True),
                     param_names=np.array(self.free_params))
            print(f"Chain saved → {flatchain_path}")
            return backend

        # ------------------------------------------------------------------
        # Case 2: burn-in phase (fresh start or checkpoint resume)
        # ------------------------------------------------------------------
        theta0 = np.array([map_params[p] for p in self.free_params])
        rng    = np.random.default_rng(42)
        scale  = np.maximum(np.abs(theta0) * 1e-3, 1e-4)
        fresh_pos = theta0[None, :] + rng.normal(0, scale, (n_walkers, n_free))
        for wi in range(n_walkers):
            for pi, pname in enumerate(self.free_params):
                lo, hi = self.param_bounds[pname]
                fresh_pos[wi, pi] = np.clip(fresh_pos[wi, pi], lo, hi)

        if os.path.exists(burnin_path):
            ck = np.load(burnin_path)
            pos        = ck["pos"]
            steps_done = int(ck["steps_done"])
            print(f"Resuming burn-in: {steps_done}/{n_burnin} steps done…")
        else:
            pos        = fresh_pos
            steps_done = 0
            print(f"Burn-in: {n_burnin} steps, {n_walkers} walkers…")

        burnin_sampler = emcee.EnsembleSampler(n_walkers, n_free, self._log_prob)
        while steps_done < n_burnin:
            this_chunk = min(self._BURNIN_CHUNK, n_burnin - steps_done)
            burnin_sampler.run_mcmc(pos, this_chunk, progress=progress)
            pos         = burnin_sampler.get_last_sample().coords
            steps_done += this_chunk
            np.savez(burnin_path, pos=pos, steps_done=steps_done)
            burnin_sampler.reset()

        # ------------------------------------------------------------------
        # Production
        # ------------------------------------------------------------------
        os.remove(burnin_path)
        print(f"Burn-in complete. Production: {n_steps} steps…")
        sampler = emcee.EnsembleSampler(
            n_walkers, n_free, self._log_prob, backend=backend)
        sampler.run_mcmc(pos, n_steps, progress=progress)

        np.savez(flatchain_path,
                 flatchain=backend.get_chain(flat=True),
                 param_names=np.array(self.free_params))
        print(f"Chain saved → {flatchain_path}")
        return sampler

    # ------------------------------------------------------------------
    # Predictions for plotting

    def predict(self, best_params: dict, sample: str) -> dict:
        """Return wp, DS, ng predictions for a given sample at MAP params."""
        fitter  = self.fitters[sample]
        Omega_m = best_params["Omega_m"]
        S8      = best_params["S8"]
        tc      = self._build_theta_cosmo(Omega_m, S8)
        hod_p   = {h: best_params[f"{h}_{sample}"] for h in HOD_PARAM_NAMES}
        wp_pred = np.asarray(fitter.predictor.wp(
            jnp.array(fitter.rp_arr), fitter.config.pi_max,
            fitter.config.z, tc, hod_p,
        ))
        ds_pred = np.asarray(fitter.predictor.delta_sigma(
            jnp.array(fitter.R_arr), fitter.config.z, tc, hod_p,
        ))
        ng_pred = float(fitter.predictor.n_gal(fitter.config.z, tc, hod_p))
        return {"wp": wp_pred, "ds": ds_pred, "ng": ng_pred}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_results(result: dict):
    p       = result["params"]
    chi2    = result["chi2"]
    ndof    = result["ndof"]
    chi2_nd = result["chi2_ndof"]

    print(f"\n{'='*65}")
    print(f"Multi-sample joint MAP fit — Lange+2025 DESI DR1")
    print(f"{'='*65}")
    print(f"chi2 = {chi2:.2f}   ndof = {ndof}   chi2/ndof = {chi2_nd:.3f}")
    print(f"Optimizer: {'success' if result['success'] else 'FAILED'}  "
          f"({result['message']})")

    print(f"\n{'─'*65}")
    print(f"{'Parameter':<22}  {'Best-fit':>10}  {'Published':>10}")
    print(f"{'─'*65}")
    # Cosmological params
    for cp in ["Omega_m", "S8"]:
        pub = PUBLISHED_COSMO.get(cp, float("nan"))
        print(f"  {cp:<20}  {p[cp]:>10.4f}  {pub:>10.4f}")
    # Per-sample HOD params
    for sname in SAMPLES:
        print(f"\n  ── {sname.upper()} ──")
        for h in HOD_PARAM_NAMES:
            key = f"{h}_{sname}"
            pub = PUBLISHED_HOD[sname].get(h, float("nan"))
            print(f"  {key:<22}  {p[key]:>10.4f}  {pub:>10.4f}")

    print(f"\n{'─'*65}")
    print("Per-sample chi2 breakdown:")
    for sname, c in result["chi2_per_sample"].items():
        print(f"  {sname}: wp={c['wp']:.2f}  DS={c['ds']:.2f}  ng={c['ng']:.2f}"
              f"  total={c['total']:.2f}")


def _save_results(result: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "benchmark_result.json")
    with open(out_path, "w") as fh:
        json.dump({
            "chi2":            result["chi2"],
            "ndof":            result["ndof"],
            "chi2_ndof":       result["chi2_ndof"],
            "chi2_per_sample": result["chi2_per_sample"],
            "params":          result["params"],
            "published_cosmo": PUBLISHED_COSMO,
            "published_hod":   PUBLISHED_HOD,
            "success":         result["success"],
            "message":         result["message"],
        }, fh, indent=2)
    print(f"\nResult saved: {out_path}")


def _print_mcmc_summary(flatchain: np.ndarray, param_names: list[str],
                        map_params: dict):
    """Print per-parameter median ± 1σ vs MAP value."""
    print(f"\n{'─'*70}")
    print("MCMC marginals (median ± 1σ  vs  MAP):")
    print(f"{'─'*70}")
    print(f"{'Parameter':<26}  {'Median':>10}  {'−1σ':>8}  {'+1σ':>8}  {'MAP':>10}")
    print(f"{'─'*70}")
    for i, pname in enumerate(param_names):
        col  = flatchain[:, i]
        med  = np.median(col)
        lo   = med - np.percentile(col, 16)
        hi   = np.percentile(col, 84) - med
        mval = map_params.get(pname, float("nan"))
        print(f"  {pname:<24}  {med:>10.4f}  {lo:>8.4f}  {hi:>8.4f}  {mval:>10.4f}")


def _make_plots(multi_fitter: MultiSampleJointFitter, result: dict,
                output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    best_params  = result["params"]
    chi2_samples = result["chi2_per_sample"]
    colors       = {"bgs2": "C0", "bgs3": "C1", "lrg1": "C2", "lrg2": "C3"}

    # --- 4-panel wp figure ---
    fig_wp, axes_wp = plt.subplots(2, 4, figsize=(16, 8), sharex="col",
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig_wp.subplots_adjust(hspace=0.05, wspace=0.3)

    for ci, sname in enumerate(SAMPLES):
        fitter = multi_fitter.fitters[sname]
        preds  = multi_fitter.predict(best_params, sname)
        c      = colors[sname]
        rp     = fitter.rp_arr
        wp_o   = fitter.wp_obs
        wp_e   = np.sqrt(np.diag(np.linalg.inv(fitter.icov_wp)))
        wp_p   = preds["wp"]
        c_nd   = chi2_samples[sname]["wp"] / len(rp)

        ax0, ax1 = axes_wp[0, ci], axes_wp[1, ci]
        ax0.errorbar(rp, wp_o, yerr=wp_e, fmt="o", ms=4, color=_COL_DATA, label="Data")
        ax0.loglog(rp, wp_p, "-", color=c, lw=1.8,
                   label=f"MAP (χ²/dof={c_nd:.2f})")
        ax0.set_title(sname.upper(), fontsize=10)
        ax0.legend(fontsize=7)
        if ci == 0:
            ax0.set_ylabel(r"$w_p(r_p)$ [$h^{-1}$ Mpc]")
        residual_panel(ax1, rp, wp_o, wp_p, wp_e, fmt="o", color=c, ylabel=(ci == 0))
        ax1.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")

    chi2_nd_all = result["chi2_ndof"]
    fig_wp.suptitle(
        f"Lange+2025 joint fit — wp(rp) | χ²/ndof={chi2_nd_all:.3f} (ndof={result['ndof']})",
        fontsize=11,
    )
    fig_wp.savefig(os.path.join(output_dir, "joint_wp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_wp)

    # --- 4-panel DS figure ---
    fig_ds, axes_ds = plt.subplots(2, 4, figsize=(16, 8), sharex="col",
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig_ds.subplots_adjust(hspace=0.05, wspace=0.3)

    for ci, sname in enumerate(SAMPLES):
        fitter = multi_fitter.fitters[sname]
        preds  = multi_fitter.predict(best_params, sname)
        c      = colors[sname]
        R      = fitter.R_arr
        ds_o   = fitter.ds_obs
        ds_e   = np.sqrt(np.diag(np.linalg.inv(fitter.icov_ds)))
        ds_p   = preds["ds"]
        c_nd   = chi2_samples[sname]["ds"] / len(R)

        ax0, ax1 = axes_ds[0, ci], axes_ds[1, ci]
        ax0.errorbar(R, ds_o, yerr=ds_e, fmt="s", ms=4, color=_COL_DATA, label="Data")
        ax0.loglog(R, ds_p, "-", color=c, lw=1.8,
                   label=f"MAP (χ²/dof={c_nd:.2f})")
        ax0.set_title(sname.upper(), fontsize=10)
        ax0.legend(fontsize=7)
        if ci == 0:
            ax0.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
        residual_panel(ax1, R, ds_o, ds_p, ds_e, fmt="s", color=c, ylabel=(ci == 0))
        ax1.set_xlabel(r"$R$ [$h^{-1}$ Mpc]")

    fig_ds.suptitle(
        f"Lange+2025 joint fit — ΔΣ (HSC-Y3) | χ²/ndof={chi2_nd_all:.3f} (ndof={result['ndof']})",
        fontsize=11,
    )
    fig_ds.savefig(os.path.join(output_dir, "joint_ds_hsc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_ds)
    print(f"MAP plots saved to {output_dir}")

    # --- MCMC plots (only when chain exists) ---
    flatchain_path = os.path.join(output_dir, "flatchain.npz")
    chain_path     = os.path.join(output_dir, "chain.h5")
    if os.path.exists(flatchain_path):
        data      = np.load(flatchain_path, allow_pickle=True)
        flatchain = data["flatchain"]
        pnames    = list(data["param_names"])
        print("MCMC chain found — generating MCMC plots…")
        _make_mcmc_plots(flatchain, pnames, chain_path,
                         multi_fitter, result, output_dir, plt)
    else:
        print("No flatchain.npz found — skipping MCMC plots.")


def _make_mcmc_plots(flatchain: np.ndarray, param_names: list,
                     chain_path: str,
                     multi_fitter: MultiSampleJointFitter,
                     result: dict, output_dir: str, plt):
    """Generate MCMC diagnostic and posterior figures."""
    best_params = result["params"]
    colors      = {"bgs2": "C0", "bgs3": "C1", "lrg1": "C2", "lrg2": "C3"}

    # ── 1. Cosmology corner (Omega_m, S8) ────────────────────────────────────
    cosmo_params = ["Omega_m", "S8"]
    cosmo_idx    = [param_names.index(p) for p in cosmo_params]
    cosmo_chain  = flatchain[:, cosmo_idx]
    cosmo_map    = [best_params[p] for p in cosmo_params]
    cosmo_pub    = [PUBLISHED_COSMO[p] for p in cosmo_params]
    cosmo_labels = [r"$\Omega_m$", r"$S_8$"]

    try:
        import corner as corner_pkg
        fig_c = corner_pkg.corner(
            cosmo_chain,
            labels=cosmo_labels,
            truths=cosmo_map,
            truth_color=_COL_MAP,
            color=_COL_MAP,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 10},
        )
        axes_c = np.array(fig_c.axes).reshape(2, 2)
        for ai in range(2):
            axes_c[ai, ai].axvline(cosmo_pub[ai], color=_COL_PUB, ls="--", lw=1.2)
        axes_c[1, 0].axvline(cosmo_pub[0], color=_COL_PUB, ls="--", lw=1.2)
        axes_c[1, 0].axhline(cosmo_pub[1], color=_COL_PUB, ls="--", lw=1.2,
                              label="Published")
        axes_c[1, 0].legend(fontsize=7)
        fig_c.suptitle(f"MCMC cosmological parameters\n"
                       f"blue = MAP   green dashed = published", fontsize=9)
    except ImportError:
        fig_c, axes_c = plt.subplots(2, 2, figsize=(7, 7))
        axes_c[0, 1].axis("off")
        for ai, (col_idx, lbl, pub, mval) in enumerate(
                zip(cosmo_idx, cosmo_labels, cosmo_pub, cosmo_map)):
            col = flatchain[:, col_idx]
            axes_c[ai, ai].hist(col, bins=50, color=_COL_MAP, alpha=0.6, density=True)
            axes_c[ai, ai].axvline(mval, color=_COL_MAP, lw=1.5, label="MAP")
            axes_c[ai, ai].axvline(pub,  color=_COL_PUB, ls="--", lw=1.2, label="Published")
            axes_c[ai, ai].set_xlabel(lbl, fontsize=10)
            if ai == 0:
                axes_c[ai, ai].legend(fontsize=7)
        axes_c[1, 0].scatter(cosmo_chain[:, 0], cosmo_chain[:, 1],
                              s=0.5, alpha=0.1, color=_COL_MAP, rasterized=True)
        axes_c[1, 0].plot(*cosmo_map, "o", color=_COL_MAP, ms=5, label="MAP")
        axes_c[1, 0].axvline(cosmo_pub[0], color=_COL_PUB, ls="--", lw=1.0)
        axes_c[1, 0].axhline(cosmo_pub[1], color=_COL_PUB, ls="--", lw=1.0,
                              label="Published")
        axes_c[1, 0].set_xlabel(cosmo_labels[0], fontsize=10)
        axes_c[1, 0].set_ylabel(cosmo_labels[1], fontsize=10)
        axes_c[1, 0].legend(fontsize=7)
        fig_c.suptitle("MCMC cosmological parameters\n"
                        "blue = MAP   green dashed = published", fontsize=9)
        fig_c.tight_layout()

    fig_c.savefig(os.path.join(output_dir, "mcmc_cosmo_corner.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print("  mcmc_cosmo_corner.png")

    # ── 2. Trace plots ────────────────────────────────────────────────────────
    trace_params = ["Omega_m", "S8"] + [f"log10mmin_{s}" for s in SAMPLES]
    trace_params = [p for p in trace_params if p in param_names]
    trace_idx    = [param_names.index(p) for p in trace_params]

    try:
        import emcee
        backend = emcee.backends.HDFBackend(chain_path, read_only=True)
        chain   = backend.get_chain()          # (n_steps, n_walkers, n_params)

        n_panels = len(trace_params)
        fig_tr, axes_tr = plt.subplots(n_panels, 1,
                                        figsize=(10, 2 * n_panels), sharex=True)
        if n_panels == 1:
            axes_tr = [axes_tr]
        for ax, tidx, tname in zip(axes_tr, trace_idx, trace_params):
            ax.plot(chain[:, :, tidx], alpha=0.25, lw=0.4, color=_COL_MAP,
                    rasterized=True)
            ax.axhline(best_params[tname], color=_COL_MAP, lw=1.5, ls="--")
            lbl = _PARAM_LATEX.get(tname, tname)
            ax.set_ylabel(lbl, fontsize=8)
        axes_tr[-1].set_xlabel("Step")
        fig_tr.suptitle("MCMC traces  (dashed = MAP)", fontsize=10)
        fig_tr.tight_layout()
        fig_tr.savefig(os.path.join(output_dir, "mcmc_traces.png"),
                       dpi=120, bbox_inches="tight")
        plt.close(fig_tr)
        print("  mcmc_traces.png")
    except Exception as exc:
        print(f"  Trace plot skipped: {exc}")

    # ── 3. Posterior predictive wp + DS ───────────────────────────────────────
    N_DRAW   = min(200, len(flatchain))
    draw_idx = np.random.default_rng(1).choice(len(flatchain), N_DRAW, replace=False)

    draws: dict = {s: {"wp": [], "ds": []} for s in SAMPLES}
    print(f"  Computing posterior predictive ({N_DRAW} draws)…")
    for di in draw_idx:
        theta = dict(zip(param_names, flatchain[di]))
        try:
            tc = multi_fitter._build_theta_cosmo(theta["Omega_m"], theta["S8"])
        except Exception:
            continue
        for sname, fitter in multi_fitter.fitters.items():
            hod_p = {h: theta[f"{h}_{sname}"] for h in HOD_PARAM_NAMES}
            try:
                wp_p = np.asarray(fitter.predictor.wp(
                    jnp.array(fitter.rp_arr), fitter.config.pi_max,
                    fitter.config.z, tc, hod_p))
                ds_p = np.asarray(fitter.predictor.delta_sigma(
                    jnp.array(fitter.R_arr), fitter.config.z, tc, hod_p))
                draws[sname]["wp"].append(wp_p)
                draws[sname]["ds"].append(ds_p)
            except Exception:
                continue
    for sname in SAMPLES:
        for key in ("wp", "ds"):
            arr = draws[sname][key]
            draws[sname][key] = np.array(arr) if arr else None

    def _ppcheck_figure(stat: str, xlabel: str, ylabel: str, fname: str,
                        obs_key: str, err_key: str, fmt: str):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex="col",
                                  gridspec_kw={"height_ratios": [3, 1]})
        fig.subplots_adjust(hspace=0.05, wspace=0.3)
        for ci, sname in enumerate(SAMPLES):
            fitter  = multi_fitter.fitters[sname]
            c       = colors[sname]
            x       = fitter.rp_arr if stat == "wp" else fitter.R_arr
            obs     = fitter.wp_obs if stat == "wp" else fitter.ds_obs
            icov    = fitter.icov_wp if stat == "wp" else fitter.icov_ds
            err     = np.sqrt(np.diag(np.linalg.inv(icov)))
            pred_map = np.asarray(multi_fitter.predict(best_params, sname)[stat])

            ax0, ax1 = axes[0, ci], axes[1, ci]

            dd = draws[sname][stat]
            if dd is not None and len(dd) > 1:
                lo16  = np.percentile(dd, 16,  axis=0)
                med50 = np.percentile(dd, 50,  axis=0)
                hi84  = np.percentile(dd, 84,  axis=0)
                lo95  = np.percentile(dd,  2.5, axis=0)
                hi95  = np.percentile(dd, 97.5, axis=0)
                ax0.fill_between(x, lo95,  hi95,  color=c, alpha=0.12)
                ax0.fill_between(x, lo16,  hi84,  color=c, alpha=0.25, label="MCMC 68%")
                ax0.loglog(x, med50, "--", color=c, lw=1.5, alpha=0.85, label="MCMC median")
                ax1.fill_between(x, lo16 / pred_map - 1, hi84 / pred_map - 1,
                                  color=c, alpha=0.25)

            ax0.errorbar(x, obs, yerr=err, fmt=fmt, ms=4, color=_COL_DATA, zorder=3,
                         label="Data")
            ax0.loglog(x, pred_map, "-", color=c, lw=1.8, label="MAP")
            ax0.set_title(sname.upper(), fontsize=10)
            ax0.legend(fontsize=7)
            ax0.set_ylabel(ylabel if ci == 0 else "")

            ratio = obs / pred_map - 1
            ax1.axhline(0,    color=_COL_DATA, lw=0.8, ls="--")
            ax1.axhline( 0.1, color="grey", lw=0.5, ls=":")
            ax1.axhline(-0.1, color="grey", lw=0.5, ls=":")
            ax1.errorbar(x, ratio, yerr=err / pred_map, fmt=fmt, ms=4, color=_COL_DATA)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel("data/MAP−1" if ci == 0 else "")
            ax1.set_ylim(-0.6, 0.6)

        fig.suptitle(f"Lange+2025 — {stat} | MAP + MCMC 68% CI", fontsize=11)
        fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")

    _ppcheck_figure("wp", r"$r_p$ [$h^{-1}$ Mpc]",
                    r"$w_p$ [$h^{-1}$ Mpc]",
                    "mcmc_wp_posterior.png", "wp", "wp_err", "o")
    _ppcheck_figure("ds", r"$R$ [$h^{-1}$ Mpc]",
                    r"$\Delta\Sigma$ [$M_\odot\,h\,{\rm pc}^{-2}$]",
                    "mcmc_ds_posterior.png", "ds", "ds_err", "s")

    # ── 4. HOD marginal histograms (one figure per sample, 2×4 grid) ──────────
    for sname in SAMPLES:
        fig_hod, axes_hod = plt.subplots(2, 4, figsize=(14, 6))
        fig_hod.subplots_adjust(hspace=0.5, wspace=0.4)
        for ai, hname in enumerate(HOD_PARAM_NAMES):
            key = f"{hname}_{sname}"
            ax  = axes_hod[ai // 4, ai % 4]
            if key not in param_names:
                ax.axis("off")
                continue
            col = flatchain[:, param_names.index(key)]
            ax.hist(col, bins=50, color=colors[sname], alpha=0.6, density=True)
            ax.axvline(best_params[key], color=_COL_MAP, lw=1.5, label="MAP")
            ax.axvline(np.median(col),   color=_COL_DATA, lw=1.0, ls="--", label="Median")
            pub_val = PUBLISHED_HOD[sname].get(hname)
            if pub_val is not None:
                ax.axvline(pub_val, color=_COL_PUB, lw=1.0, ls=":", label="Published")
            ax.set_xlabel(_PARAM_LATEX.get(hname, hname), fontsize=8)
            ax.tick_params(labelsize=7)
            if ai == 0:
                ax.legend(fontsize=6)
        fig_hod.suptitle(f"MCMC HOD marginals — {sname.upper()}", fontsize=11)
        fname = f"mcmc_hod_marginals_{sname}.png"
        fig_hod.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig_hod)
        print(f"  {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true",
                    help="Generate wp and DS comparison figures.")
    ap.add_argument("--plot-only", action="store_true",
                    help="Reload saved result JSON and regenerate figures only (skip MAP/MCMC).")
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Directory for result JSON and figures.")
    ap.add_argument("--mcmc", action="store_true",
                    help="Run emcee MCMC after MAP (slow; 34 params).")
    ap.add_argument("--force-mcmc", action="store_true",
                    help="Rerun MCMC even if flatchain.npz already exists.")
    ap.add_argument("--n-walkers", type=int, default=80,
                    help="Number of emcee walkers (default: 80; must be ≥ 68).")
    ap.add_argument("--n-burnin", type=int, default=500,
                    help="Burn-in steps per walker (default: 500).")
    ap.add_argument("--n-steps", type=int, default=2000,
                    help="Production steps per walker (default: 2000).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building multi-sample joint fitter (BGS2, BGS3, LRG1, LRG2)…")
    multi = MultiSampleJointFitter(SAMPLE_CONFIGS)

    n_wp = sum(len(f.rp_arr) for f in multi.fitters.values())
    n_ds = sum(len(f.R_arr)  for f in multi.fitters.values())
    print(f"Data: {n_wp} wp bins + {n_ds} DS bins + 4 ng = {multi._n_data()} total")

    if args.plot_only:
        result_file = os.path.join(args.output_dir, "result.json")
        if not os.path.exists(result_file):
            print(f"ERROR: {result_file} not found — run without --plot-only first.")
            return 1
        import json
        with open(result_file) as fh:
            result = json.load(fh)
        print(f"  Loaded result from {result_file}")
        _make_plots(multi, result, args.output_dir)
        return 0

    result = multi.map_fit()
    _print_results(result)
    _save_results(result, args.output_dir)

    passes = result["chi2_ndof"] < 2.0
    print(f"\nBenchmark: {'PASS' if passes else 'FAIL'} "
          f"(chi2/ndof={result['chi2_ndof']:.3f})")

    if args.mcmc:
        chain_path     = os.path.join(args.output_dir, "chain.h5")
        burnin_path    = os.path.join(args.output_dir, "burnin_pos.npz")
        flatchain_path = os.path.join(args.output_dir, "flatchain.npz")

        if args.force_mcmc:
            for _f in [chain_path, burnin_path, flatchain_path]:
                if os.path.exists(_f):
                    os.remove(_f)
                    print(f"Removed {_f}")

        if os.path.exists(flatchain_path) and not args.force_mcmc:
            print(f"\nMCMC complete — loading {flatchain_path}")
            print("  Pass --force-mcmc to rerun from scratch.")
            data = np.load(flatchain_path, allow_pickle=True)
            _print_mcmc_summary(data["flatchain"],
                                list(data["param_names"]),
                                result["params"])
        else:
            print(f"\nRunning MCMC ({args.n_walkers} walkers, "
                  f"{args.n_burnin} burn-in + {args.n_steps} steps)…")
            multi.run_mcmc(
                map_params=result["params"],
                output_dir=args.output_dir,
                n_walkers=args.n_walkers,
                n_burnin=args.n_burnin,
                n_steps=args.n_steps,
            )
            data = np.load(flatchain_path, allow_pickle=True)
            _print_mcmc_summary(data["flatchain"],
                                list(data["param_names"]),
                                result["params"])

    if args.plot or args.plot_only:
        _make_plots(multi, result, args.output_dir)

    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
