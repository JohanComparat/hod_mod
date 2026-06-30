#!/usr/bin/env python
"""Joint MAP + MCMC fit of all Zu & Mandelbaum 2015 iHOD bins simultaneously.

All 7 stellar-mass bins (9.4-9.8 through 11.4-12.0) share a single set of 9
free SHMR/HOD parameters.  Per-bin fixed params (log10m_star_thresh, log10m_star_max,
z_eff) are wired into each individual fitter.

The combined log-probability sums contributions from all bins:
    log P(theta) = log_prior(theta) + Σ_i log L_i(theta | data_i)

Usage
-----
    # Run MAP (slow, ~2–3 h):
    python hod_mod/scripts/benchmarks/run_zm15_joint_all.py

    # Plot saved MAP result without re-optimising (~60 s for JAX compilation):
    python hod_mod/scripts/benchmarks/run_zm15_joint_all.py --plot

    # Run MCMC after MAP:
    python hod_mod/scripts/benchmarks/run_zm15_joint_all.py --mcmc

    # Force-rerun MCMC even if chain exists:
    python hod_mod/scripts/benchmarks/run_zm15_joint_all.py --mcmc --force-mcmc
"""

import argparse
import json
import os
import sys

import numpy as np
from hod_mod.paths import results_root

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

OUT_DIR = os.path.join(results_root(), "benchmarks/zumandelbaum2015_joint")

# Per-bin config files (all share the same 9 free parameters)
BIN_CONFIGS = [
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_9p4_9p8.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_9p8_10p2.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_10p2_10p6.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_10p6_11p0.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p0_11p2.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p2_11p4.yml",
    "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p4_12p0.yml",
]

# Published iHOD global parameters (Table 2 of ZM15)
PUBLISHED = {
    "lg_m1h":        (12.10, 0.17),
    "lg_m0star":     (10.31, 0.10),
    "beta":          (0.33,  0.21),
    "delta":         (0.42,  0.04),
    "gamma":         (1.21,  0.20),
    "sigma_lnmstar": (0.50,  0.04),
    "eta":           (-0.04, 0.02),
    "fc":            (0.86,  0.14),
    "bsat":          (8.98,  1.18),
}


# ---------------------------------------------------------------------------
# Combined fitter — wraps per-bin fitters into a single log-prob
# ---------------------------------------------------------------------------

class JointAllFitter:
    """Combines multiple per-bin fitters into a single log-probability.

    All fitters share the same free parameter names and bounds; only their
    fixed params (thresh, max, z_eff) differ.
    """

    def __init__(self, fitters):
        self.fitters = fitters
        # Borrow config from first fitter (all have identical free params)
        self.config = fitters[0].config
        self.free_params = self.config.free_params
        self.param_bounds = self.config.param_bounds

    @property
    def _x0(self) -> np.ndarray:
        return np.array([self.config.param_init[p] for p in self.free_params])

    def _log_prob(self, theta_vec) -> float:
        """Prior once + sum of data log-likelihoods from all bins."""
        # Check prior against the first fitter (prior is shared)
        lp = self.fitters[0]._prior_log_prob(theta_vec)
        if not np.isfinite(lp):
            return -np.inf
        for fitter in self.fitters:
            ll = fitter._log_likelihood_only(theta_vec)
            if not np.isfinite(ll):
                return -np.inf
            lp += ll
        return lp

    def map_fit(self) -> dict:
        from scipy.optimize import minimize

        x0 = self._x0
        lp0 = self._log_prob(x0)
        print(f"  log-prob at starting point: {lp0:.3f}  (chi2={-2*lp0:.1f})")
        if not np.isfinite(lp0):
            raise RuntimeError(
                "Starting point has non-finite log-probability. "
                "Check param_init values and model."
            )

        # Small simplex: perturb each param by 1% — stays well inside NaN-free zone
        n = len(x0)
        scale = np.maximum(np.abs(x0) * 0.01, 1e-3)
        simplex = np.vstack([x0, x0 + np.diag(scale)])  # shape (n+1, n)

        _iter = [0]

        def _callback(xk):
            _iter[0] += 1
            if _iter[0] % 50 == 0:
                lp = self._log_prob(xk)
                print(f"  iter {_iter[0]:4d}  chi2={-2*lp:.2f}")

        result = minimize(
            lambda x: -self._log_prob(x),
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": 5000,
                "xatol": 1e-4,
                "fatol": 1e-4,
                "adaptive": True,
                "disp": True,
                "initial_simplex": simplex,
            },
            callback=_callback,
        )
        theta = result.x
        params = _assemble_params(theta, self.free_params, self.fitters[0]._fixed_params)
        chi2, ndof = self._chi2_breakdown(theta)
        total_chi2 = sum(chi2.values())
        total_ndof  = sum(ndof.values())
        return {
            "theta":   list(theta),
            "params":  params,
            "chi2":    total_chi2,
            "chi2_per_bin": chi2,
            "ndof_per_bin": ndof,
            "ndof":    total_ndof,
            "success": bool(result.success),
            "message": result.message,
        }

    def _chi2_breakdown(self, theta):
        """Return per-bin chi2 and ndof dicts."""
        chi2, ndof = {}, {}
        for fitter in self.fitters:
            lbl = fitter._bin_label
            chi2[lbl] = fitter._chi2_at(theta)
            ndof[lbl] = fitter._ndof
        return chi2, ndof

    def sample(self, initial_pos=None, progress=True):
        import emcee
        n_free    = len(self.free_params)
        n_walkers = self.config.n_walkers
        if initial_pos is None:
            x0    = self._x0
            scale = np.maximum(np.abs(x0) * 1e-3, 1e-4)
            rng   = np.random.default_rng(42)
            initial_pos = x0 + rng.normal(0, scale, (n_walkers, n_free))
            for i, pname in enumerate(self.free_params):
                lo, hi = self.param_bounds[pname]
                initial_pos[:, i] = np.clip(initial_pos[:, i], lo, hi)

        sampler = emcee.EnsembleSampler(n_walkers, n_free, self._log_prob)
        print(f"  Burn-in: {self.config.n_burnin} steps …")
        sampler.run_mcmc(initial_pos, self.config.n_burnin, progress=progress)
        last = sampler.get_last_sample()
        sampler.reset()
        print(f"  Sampling: {self.config.n_steps} steps …")
        sampler.run_mcmc(last, self.config.n_steps, progress=progress)
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "flatchain.npz")
        np.savez(out, flatchain=sampler.get_chain(flat=True),
                 param_names=np.array(self.free_params))
        print(f"  Chain saved → {out}")
        return sampler


def _assemble_params(theta, free_params, fixed_params):
    p = dict(fixed_params)
    for name, val in zip(free_params, theta):
        p[name] = float(val)
    return p


# ---------------------------------------------------------------------------
# Patch per-bin fitters with helper methods
# ---------------------------------------------------------------------------

def _patch_fitter(fitter, bin_label):
    """Add _log_likelihood_only, _chi2_at, _ndof, _bin_label to a fitter."""
    from hod_mod.fitting import _assemble_hod_params
    import jax.numpy as jnp

    fitter._bin_label = bin_label

    def _ll_only(theta_vec):
        """Log-likelihood only (no prior); called by JointAllFitter."""
        fixed = fitter._fixed_params
        hod_params  = _assemble_hod_params(theta_vec, fitter.config.free_params, fixed)
        theta_cosmo = fitter._theta_cosmo_call(hod_params)
        try:
            wp_pred = np.asarray(fitter.predictor.wp(
                jnp.array(fitter.rp_arr), fitter.config.pi_max,
                fitter.config.z, theta_cosmo, hod_params))
        except Exception:
            return -np.inf
        res_wp = wp_pred - fitter.wp_obs
        ll = -0.5 * float(res_wp @ fitter.icov_wp @ res_wp)
        # ESD if available
        if hasattr(fitter, "R_arr"):
            try:
                ds_pred = np.asarray(fitter.predict_ds(hod_params))
            except Exception:
                return -np.inf
            res_ds = ds_pred - fitter.ds_obs
            ll += -0.5 * float(res_ds @ fitter.icov_ds @ res_ds)
        return ll

    def _chi2_at(theta_vec):
        ll = _ll_only(theta_vec)
        return -2.0 * ll if np.isfinite(ll) else np.nan

    has_ds = hasattr(fitter, "R_arr")
    n_data = len(fitter.rp_arr) + (len(fitter.R_arr) if has_ds else 0)
    n_free = len(fitter.config.free_params)
    fitter._ndof             = max(1, n_data - n_free)
    fitter._log_likelihood_only = _ll_only
    fitter._chi2_at             = _chi2_at
    return fitter


# ---------------------------------------------------------------------------
# Build per-bin fitters
# ---------------------------------------------------------------------------

def build_fitters():
    from hod_mod.fitting import load_config, WpFitter, JointFitter

    BIN_LABELS = [
        "9.4-9.8", "9.8-10.2", "10.2-10.6", "10.6-11.0",
        "11.0-11.2", "11.2-11.4", "11.4-12.0",
    ]

    fitters = []
    for cfg_rel, label in zip(BIN_CONFIGS, BIN_LABELS):
        cfg_path = os.path.join(_REPO_ROOT, cfg_rel)
        config   = load_config(cfg_path)
        has_wp   = bool(config.data_file and os.path.isfile(config.data_file))
        has_ds   = config.ds_file is not None
        if has_wp and has_ds:
            f = JointFitter(config)
        else:
            f = WpFitter(config)
        _patch_fitter(f, label)
        fitters.append(f)
        print(f"  [{label}] {'joint' if has_ds else 'wp-only':9s}  "
              f"wp={len(f.rp_arr):2d} pts"
              + (f"  ds={len(f.R_arr):2d} pts" if has_ds else ""))
    return fitters


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Published iHOD Table 2 parameters (for overlay on MAP figure)
_IHOD_PUB = {
    "lg_m1h": 12.10, "lg_m0star": 10.31, "beta": 0.33,
    "delta": 0.42, "gamma": 1.21, "sigma_lnmstar": 0.50,
    "eta": -0.04, "fc": 0.86, "bsat": 8.98,
    "beta_sat": 0.90, "bcut": 0.86, "beta_cut": 0.41, "alpha_sat": 1.00,
}

BIN_LABELS = [
    "9.4-9.8", "9.8-10.2", "10.2-10.6", "10.6-11.0",
    "11.0-11.2", "11.2-11.4", "11.4-12.0",
]
COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def _predict_wp_ds(fitter, params):
    """Return (rp, wp_pred, R, ds_pred|None) at params dict."""
    import jax.numpy as jnp
    theta_cosmo = fitter._theta_cosmo_call(params)
    wp_pred = np.asarray(fitter.predictor.wp(
        jnp.array(fitter.rp_arr), fitter.config.pi_max,
        fitter.config.z, theta_cosmo, params))
    ds_pred = None
    if hasattr(fitter, "R_arr"):
        ds_pred = np.asarray(fitter.predict_ds(params))
    return wp_pred, ds_pred


def plot_joint_map(fitters, map_params, chi2_per_bin):
    """Generate 2-row × 7-col figure: wp and ds for all bins at MAP + published."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fitters)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7),
                              gridspec_kw={"hspace": 0.05})

    for col, (fitter, label, color) in enumerate(zip(fitters, BIN_LABELS, COLORS)):
        # Build per-bin params from map_params (overrides free params only)
        params_map = dict(fitter._fixed_params)
        params_map.update(map_params)

        params_pub = dict(fitter._fixed_params)
        params_pub.update(_IHOD_PUB)

        try:
            wp_map, ds_map = _predict_wp_ds(fitter, params_map)
            wp_pub, ds_pub = _predict_wp_ds(fitter, params_pub)
        except Exception as exc:
            print(f"  [{label}] prediction failed: {exc}")
            continue

        rp     = np.array(fitter.rp_arr)
        wp_obs = np.array(fitter.wp_obs)
        wp_err = np.array(fitter.wp_err)
        chi2   = chi2_per_bin.get(label, float("nan"))

        # --- wp ---
        ax = axes[0, col]
        ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=3, color="k",
                    lw=0.7, capsize=2, zorder=3)
        ax.loglog(rp, wp_map, "-", color="C0", lw=2.0, label=f"MAP  χ²={chi2:.1f}")
        ax.loglog(rp, wp_pub, "--", color="C2", lw=1.5, label="published")
        ax.set_title(rf"$[{label}]$", fontsize=8)
        ax.legend(fontsize=5.5, loc="upper right")
        ax.set_xlim(0.04, 30)
        if col == 0:
            ax.set_ylabel(r"$w_p\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=9)
        else:
            ax.set_yticklabels([])

        # --- ds ---
        ax2 = axes[1, col]
        if hasattr(fitter, "R_arr") and ds_map is not None:
            R      = np.array(fitter.R_arr)
            ds_obs = np.array(fitter.ds_obs)
            ds_err = np.array(fitter.ds_err)
            ax2.errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=3, color="k",
                         lw=0.7, capsize=2, zorder=3)
            ax2.loglog(R, ds_map, "-", color="C0", lw=2.0)
            ax2.loglog(R, ds_pub, "--", color="C2", lw=1.5)
            ax2.set_xlim(0.04, 20)
            if col == 0:
                ax2.set_ylabel(r"$\Delta\Sigma\ [M_\odot\,h\,\mathrm{pc}^{-2}]$", fontsize=9)
            else:
                ax2.set_yticklabels([])
        else:
            ax2.text(0.5, 0.5, "ESD not used", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=8, color="gray")
            ax2.set_axis_off()
        ax2.set_xlabel(r"$r_p$ or $R\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=8)

    fig.suptitle(
        "Zu & Mandelbaum 2015 — joint MAP (blue) vs. published (green dashed)",
        fontsize=11, y=1.01,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "benchmark_joint_all_map.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_results(map_result, free_params):
    print(f"\nchi2/ndof = {map_result['chi2']:.3f} / {map_result['ndof']} "
          f"= {map_result['chi2']/map_result['ndof']:.3f}")
    print("\nPer-bin χ²:")
    for lbl, c2 in map_result["chi2_per_bin"].items():
        nd = map_result["ndof_per_bin"][lbl]
        print(f"  [{lbl}]  χ²={c2:.2f}  ndof={nd}  χ²/dof={c2/nd:.2f}")

    print(f"\n{'Parameter':20s}  {'MAP':>10s}  {'Published':>10s}  {'Δ/σ':>8s}")
    print("-" * 55)
    params = map_result["params"]
    for pname, (pub_val, pub_err) in PUBLISHED.items():
        bfit = params.get(pname, float("nan"))
        diff = (bfit - pub_val) / pub_err if pub_err > 0 else float("nan")
        print(f"{pname:20s}  {bfit:10.4f}  {pub_val:10.4f}  {diff:8.2f}σ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot",       action="store_true",
                        help="plot saved MAP result without re-optimising")
    parser.add_argument("--mcmc",       action="store_true", help="run MCMC after MAP")
    parser.add_argument("--force-mcmc", action="store_true", help="rerun MCMC even if chain exists")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    chain_file = os.path.join(OUT_DIR, "flatchain.npz")
    out_json   = os.path.join(OUT_DIR, "benchmark_result_joint_all.json")

    # ------------------------------------------------------------------
    # --plot: load saved result and generate figures, then exit
    if args.plot:
        if not os.path.exists(out_json):
            raise FileNotFoundError(
                f"No saved result found at {out_json}. Run without --plot first."
            )
        with open(out_json) as fh:
            saved = json.load(fh)
        map_params     = saved["params"]
        chi2_per_bin   = saved.get("chi2_per_bin", {})
        print(f"Loaded saved result: chi2/ndof = {saved['chi2']:.2f}/{saved['ndof']}")
        print("Building fitters (JAX compilation ~60 s) …")
        fitters = build_fitters()
        print("Generating figures …")
        out = plot_joint_map(fitters, map_params, chi2_per_bin)
        print(f"Done → {out}")
        return

    import os as _os
    print(f"PID={_os.getpid()}  Starting ZM15 joint fit …")
    print("Building per-bin fitters (JAX compilation ~60 s) …")
    fitters = build_fitters()
    print(f"\nBuilt {len(fitters)} fitters.")

    joint = JointAllFitter(fitters)

    # ------------------------------------------------------------------
    # MAP
    print("\nRunning joint MAP (Nelder-Mead, 9 params × 7 bins) …")
    map_result = joint.map_fit()
    _print_results(map_result, joint.free_params)

    passes = (map_result["chi2"] / map_result["ndof"]) < 2.0
    result = {
        "model":  "zumandelbaum2015_joint_all",
        "label":  "ZM15 joint all-bins MAP (ZuMandelbaum15HODModel)",
        "status": "pass" if passes else "fail",
        "chi2":         float(map_result["chi2"]),
        "ndof":         int(map_result["ndof"]),
        "chi2_ndof":    float(map_result["chi2"] / map_result["ndof"]),
        "chi2_per_bin": {k: float(v) for k, v in map_result["chi2_per_bin"].items()},
        "ndof_per_bin": {k: int(v)   for k, v in map_result["ndof_per_bin"].items()},
        "params":       {k: float(v) for k, v in map_result["params"].items()},
        "published_params": {k: list(v) for k, v in PUBLISHED.items()},
        "param_deviations_sigma": {
            pname: float((map_result["params"].get(pname, float("nan")) - pub_val) / pub_err)
            for pname, (pub_val, pub_err) in PUBLISHED.items()
        },
    }
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResult saved → {out_json}")
    print(f"Benchmark: {'PASSED' if passes else 'FAILED'}")

    # ------------------------------------------------------------------
    # MCMC (optional)
    if not args.mcmc:
        return

    if os.path.exists(chain_file) and not args.force_mcmc:
        print(f"\nMCMC chain already exists ({chain_file}). Pass --force-mcmc to rerun.")
        return

    print("\nInitialising MCMC walkers near MAP …")
    theta0 = np.asarray(map_result["theta"])
    scale  = np.maximum(np.abs(theta0) * 1e-3, 1e-4)
    rng    = np.random.default_rng(42)
    n_w    = joint.config.n_walkers
    pos    = theta0 + rng.normal(0, scale, (n_w, len(theta0)))
    for i, pname in enumerate(joint.free_params):
        lo, hi = joint.param_bounds[pname]
        pos[:, i] = np.clip(pos[:, i], lo, hi)
    joint.sample(initial_pos=pos, progress=True)


if __name__ == "__main__":
    main()
