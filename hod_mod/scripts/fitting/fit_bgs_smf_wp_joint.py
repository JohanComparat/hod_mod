"""Joint ZM15 iHOD fit to all 5 BGS Mstar-threshold samples simultaneously.

One shared set of 13 SHMR+satellite parameters (nothing fixed) is constrained
by the wp(rp) and SMF Phi(M*) of all five samples at once.  The per-sample
stellar-mass threshold (log10m_star_thresh) is set from each sample's known
selection cut — not a free parameter — analogous to how bin edges are handled
in bgs_ls10/fit_bgs_zm15_joint.py.

This is the iHOD philosophy from Zu & Mandelbaum (2015): one universal SHMR
(+ scatter + satellite occupation) predicts the galaxy content of every
stellar-mass threshold sample simultaneously.

Data (from ~/software/sum_stat/data/):
    BGS_Mstar9.0/    BGS_Mstar10.0/   BGS_Mstar10.5/
    BGS_Mstar11.0/   BGS_Mstar11.5/

Probes:
    wp(rp)     projected correlation function (diagonal jackknife variance)
    Phi(M*)    stellar mass function bins    (diagonal jackknife variance)

Free parameters (13, shared across all samples):
    lg_m1h, lg_m0star, beta, delta, gamma   SHMR shape
    sigma_lnmstar, eta                       SHMR scatter
    fc                                       central completeness
    bsat, beta_sat, bcut, beta_cut, alpha_sat  satellite occupation

Joint likelihood:
    log P(theta) = log_prior(theta)
                 + Sum_samples [ ll_smf(sample) + ll_wp(sample) ]
    where each term is a diagonal Gaussian with jackknife variance + f_sys floor.

Output (results/fits/bgs_smf_wp_joint/):
    joint_map.json            MAP params + chi2/ndof + per-sample breakdown
    joint_chain.h5            emcee flat chain (n_walkers×n_steps, 13)
    joint_mcmc_summary.json   medians and 68% credibles
    joint_corner.pdf          13-param corner plot
    joint_bestfit.pdf         10-panel figure: wp + SMF for each of 5 samples

Usage::

    python -m hod_mod.scripts.fitting.fit_bgs_smf_wp_joint --mode map
    python -m hod_mod.scripts.fitting.fit_bgs_smf_wp_joint --mode both
    python -m hod_mod.scripts.fitting.fit_bgs_smf_wp_joint --mode map --rp-min 0.1

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

import numpy as np
from scipy.optimize import minimize

from hod_mod.scripts.fitting.fit_bgs_smf_wp import (
    SAMPLES,
    _THETA_COSMO,
    _Infrastructure,
    _predict_smf,
    load_data,
)
from hod_mod.paths import results_root

_RESULTS_DIR = results_root() / "fits" / "bgs_smf_wp_joint"

_ALL_LABELS = list(SAMPLES)   # ["M9", "M10", "M10p5", "M11", "M11p5"]

# ---------------------------------------------------------------------------
# Shared free parameters — 13 total, nothing fixed
# ---------------------------------------------------------------------------

FREE_PARAMS: dict[str, tuple] = {       # name: (bounds, x0)
    "lg_m1h":        (( 9.0, 15.0), 12.10),
    "lg_m0star":     (( 8.0, 12.5), 10.31),
    "beta":          (( 0.0,  2.0),  0.33),
    "delta":         (( 0.0,  3.0),  0.42),
    "gamma":         (( 0.1, 10.0),  1.21),
    "sigma_lnmstar": ((0.01,  2.0),  0.50),
    "eta":           ((-1.5,  1.5), -0.04),
    "fc":            ((0.05,  1.0),  0.86),
    "bsat":          (( 0.1, 50.0),  8.98),
    "beta_sat":      (( 0.0,  2.0),  0.90),
    "bcut":          ((0.01, 10.0),  0.86),
    "beta_cut":      (( 0.0,  2.0),  0.41),
    "alpha_sat":     (( 0.3,  3.0),  1.00),
}

FREE_NAMES  = list(FREE_PARAMS.keys())
FREE_BOUNDS = [v[0] for v in FREE_PARAMS.values()]
FREE_X0     = np.array([v[1] for v in FREE_PARAMS.values()])

assert len(FREE_NAMES) == 13


def _params_to_hod(theta: np.ndarray, label: str) -> dict:
    """Build full ZuMandelbaum15HODModel param dict for one sample.

    Shared SHMR+satellite params come from ``theta``; ``log10m_star_thresh``
    is set to the sample's known stellar-mass cut.
    """
    hp = dict(zip(FREE_NAMES, theta.tolist()))
    hp["log10m_star_thresh"] = float(SAMPLES[label]["mstar_thresh"])
    return hp


# ---------------------------------------------------------------------------
# Joint model class
# ---------------------------------------------------------------------------

class JointZM15Thresh:
    """Joint likelihood over 5 BGS Mstar-threshold samples."""

    def __init__(
        self,
        samples_data: dict,   # {label: load_data() result}
        infra: _Infrastructure,
    ):
        self.samples_data = samples_data
        self.infra        = infra
        self.labels       = list(samples_data)

    # -- prior ---------------------------------------------------------------
    def log_prior(self, theta: np.ndarray) -> float:
        for val, (lo, hi) in zip(theta, FREE_BOUNDS):
            if not (lo <= val <= hi):
                return -np.inf
        return 0.0

    # -- per-sample log-likelihood -------------------------------------------
    def _sample_ll(self, theta: np.ndarray, label: str) -> float:
        hp   = _params_to_hod(theta, label)
        data = self.samples_data[label]
        z    = SAMPLES[label]["zmean"]

        phi_model = _predict_smf(self.infra, z, hp, data["log10mstar"])
        wp_model  = np.asarray(
            self.infra.fhmp.wp(
                data["rp"], pi_max=data["pi_max"],
                z=z, theta_cosmo=_THETA_COSMO, hod_params=hp,
            ),
            dtype=float,
        )

        ll_smf = -0.5 * float(np.sum((data["phi"] - phi_model) ** 2 / data["phi_var"]))
        ll_wp  = -0.5 * float(np.sum((data["wp"]  - wp_model)  ** 2 / data["wp_var"]))
        return ll_smf + ll_wp

    # -- full joint log-probability ------------------------------------------
    def log_prob(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        try:
            ll = sum(self._sample_ll(theta, lbl) for lbl in self.labels)
            return lp + ll if np.isfinite(ll) else -np.inf
        except Exception:
            return -np.inf

    def n_data(self) -> int:
        return sum(d["n_smf"] + d["n_wp"] for d in self.samples_data.values())

    def per_sample_chi2(self, theta: np.ndarray) -> dict:
        out = {}
        for lbl in self.labels:
            ll = self._sample_ll(theta, lbl)
            n  = self.samples_data[lbl]["n_smf"] + self.samples_data[lbl]["n_wp"]
            out[lbl] = {"chi2": float(-2 * ll), "n_pts": n}
        return out


# ---------------------------------------------------------------------------
# MAP fit
# ---------------------------------------------------------------------------

def run_map(
    infra: _Infrastructure,
    rp_min: float = 0.02,
    f_sys:  float = 0.05,
) -> dict:
    """L-BFGS-B MAP fit of 13 shared parameters across all 5 samples."""
    print("Loading data for all samples ...", flush=True)
    samples_data = {
        lbl: load_data(lbl, rp_min=rp_min, f_sys=f_sys)
        for lbl in _ALL_LABELS
    }
    for lbl, d in samples_data.items():
        print(f"  [{lbl}]  n_smf={d['n_smf']}  n_wp={d['n_wp']}", flush=True)

    model = JointZM15Thresh(samples_data, infra)
    n_pts = model.n_data()
    ndof  = max(n_pts - len(FREE_NAMES), 1)

    print(f"\nMAP: x0={np.round(FREE_X0, 3)}", flush=True)
    print(f"     n_pts_total={n_pts}  ndof={ndof}", flush=True)

    def neg_log_prob(t):
        v = model.log_prob(t)
        return -v if np.isfinite(v) else 1e30

    res = minimize(
        neg_log_prob, FREE_X0, method="L-BFGS-B",
        bounds=FREE_BOUNDS,
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 3000, "eps": 1e-3},
    )

    chi2    = 2.0 * res.fun
    per_bin = model.per_sample_chi2(res.x)

    print(
        f"\nMAP done: params={np.round(res.x, 4)}\n"
        f"          chi2={chi2:.2f}  ndof={ndof}  chi2/dof={chi2/ndof:.2f}"
        f"  success={res.success}",
        flush=True,
    )
    for lbl, info in per_bin.items():
        print(f"  [{lbl}]  chi2={info['chi2']:.2f}  n_pts={info['n_pts']}", flush=True)

    return dict(
        param_names   = FREE_NAMES,
        params        = res.x.tolist(),
        chi2          = float(chi2),
        ndof          = int(ndof),
        chi2_dof      = float(chi2 / ndof),
        log_prob      = float(-res.fun),
        success       = bool(res.success),
        n_pts_total   = n_pts,
        per_sample    = per_bin,
        rp_min_hmpc   = rp_min,
        f_sys         = f_sys,
    )


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------

def run_mcmc(
    infra: _Infrastructure,
    map_result: dict,
    n_walkers: int = 32,
    n_steps:   int = 1000,
    n_burnin:  int = 300,
    rp_min:    float = 0.02,
    f_sys:     float = 0.05,
) -> dict:
    """emcee ensemble sampler starting near MAP for 13 shared parameters."""
    import emcee

    samples_data = {
        lbl: load_data(lbl, rp_min=rp_min, f_sys=f_sys)
        for lbl in _ALL_LABELS
    }
    model = JointZM15Thresh(samples_data, infra)

    x_map = np.array(map_result["params"])
    n_dim = len(x_map)

    # Walker init scales — ordered as FREE_NAMES
    scales = np.array([
        0.10,  # lg_m1h
        0.10,  # lg_m0star
        0.05,  # beta
        0.05,  # delta
        0.10,  # gamma
        0.05,  # sigma_lnmstar
        0.05,  # eta
        0.05,  # fc
        0.50,  # bsat  (wide — poorly constrained individually)
        0.05,  # beta_sat
        0.10,  # bcut
        0.05,  # beta_cut
        0.10,  # alpha_sat
    ])

    rng = np.random.default_rng(42)
    pos = x_map[None, :] + scales[None, :] * rng.standard_normal((n_walkers, n_dim))
    for i, (lo, hi) in enumerate(FREE_BOUNDS):
        pos[:, i] = np.clip(pos[:, i], lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, model.log_prob)

    print(f"MCMC burn-in: {n_burnin} steps × {n_walkers} walkers ...", flush=True)
    pos, _, _ = sampler.run_mcmc(pos, n_burnin, progress=True)
    sampler.reset()

    print(f"MCMC production: {n_steps} steps ...", flush=True)
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
        param_names  = FREE_NAMES,
        chain        = flat_chain,
        medians      = medians.tolist(),
        lo16         = lo16.tolist(),
        hi84         = hi84.tolist(),
        autocorr_tau = tau.tolist() if hasattr(tau, "tolist") else list(tau),
        n_walkers    = n_walkers,
        n_steps      = n_steps,
        n_burnin     = n_burnin,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_map(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "joint_map.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {path}", flush=True)


def _save_mcmc(mcmc: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chain = mcmc.pop("chain")
    summary = {k: v for k, v in mcmc.items()}
    with open(out_dir / "joint_mcmc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {out_dir / 'joint_mcmc_summary.json'}", flush=True)

    try:
        import h5py
        chain_path = out_dir / "joint_chain.h5"
        with h5py.File(chain_path, "w") as f:
            f.create_dataset("chain", data=chain)
            f.attrs["param_names"] = json.dumps(mcmc["param_names"])
        print(f"  → {chain_path}", flush=True)
    except ImportError:
        chain_path = out_dir / "joint_chain.npy"
        np.save(chain_path, chain)
        print(f"  → {chain_path}", flush=True)
    mcmc["chain"] = chain


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bestfit(
    infra: _Infrastructure,
    params: list,
    out_path: Path,
    rp_min: float = 0.02,
    f_sys:  float = 0.05,
    per_sample_chi2: dict | None = None,
) -> None:
    """5-column figure: wp(rp) and SMF Phi(M*) for each sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.array(params)
    n_col = len(_ALL_LABELS)
    fig, axes = plt.subplots(
        4, n_col, figsize=(3.5 * n_col, 12),
        gridspec_kw={"height_ratios": [3, 1, 3, 1]},
    )

    for col, label in enumerate(_ALL_LABELS):
        hp   = _params_to_hod(theta, label)
        data = load_data(label, rp_min=rp_min, f_sys=0.0)
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

        wp_err  = np.sqrt(data["wp_var"])
        phi_err = np.sqrt(data["phi_var"])
        chi_str = ""
        if per_sample_chi2 and label in per_sample_chi2:
            info = per_sample_chi2[label]
            chi_str = f"  χ²={info['chi2']:.1f}/{info['n_pts']-13}"

        # --- wp main ---
        ax = axes[0, col]
        ax.errorbar(data["rp"], data["wp"], yerr=wp_err,
                    fmt="o", ms=3, color="k", zorder=2)
        ax.plot(data["rp"], wp_model, "-", lw=2, color="C0", zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(
            f"M$_*>{s['mstar_thresh']}$\n$z={s['zmean']:.3f}${chi_str}",
            fontsize=9,
        )
        if col == 0:
            ax.set_ylabel(r"$w_p$ [Mpc/$h$]")

        # --- wp residual ---
        axr = axes[1, col]
        pull_wp = (data["wp"] - wp_model) / wp_err
        axr.axhline(0, ls="--", color="0.5", lw=0.8)
        axr.plot(data["rp"], pull_wp, "o", ms=3, color="C0")
        axr.set_xscale("log")
        axr.set_ylim(-5, 5)
        axr.set_xlabel(r"$r_p$ [Mpc/$h$]", fontsize=8)
        if col == 0:
            axr.set_ylabel("Pull")

        # --- SMF main ---
        ax2 = axes[2, col]
        ax2.errorbar(data["log10mstar"], data["phi"], yerr=phi_err,
                     fmt="o", ms=3, color="k", zorder=2)
        ax2.plot(data["log10mstar"], phi_model, "-", lw=2, color="C1", zorder=3)
        ax2.set_yscale("log")
        if col == 0:
            ax2.set_ylabel(r"$\Phi$ [(Mpc/$h$)$^{-3}$dex$^{-1}$]")

        # --- SMF residual ---
        axr2 = axes[3, col]
        pull_smf = (data["phi"] - phi_model) / phi_err
        axr2.axhline(0, ls="--", color="0.5", lw=0.8)
        axr2.plot(data["log10mstar"], pull_smf, "o", ms=3, color="C1")
        axr2.set_ylim(-5, 5)
        axr2.set_xlabel(r"$\log_{10}M_*$", fontsize=8)
        if col == 0:
            axr2.set_ylabel("Pull")

    fig.suptitle("Joint ZM15 iHOD — BGS Mstar threshold samples", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def plot_corner(mcmc: dict, out_path: Path) -> None:
    try:
        import corner
    except ImportError:
        print("  corner not installed — skipping corner plot.", flush=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = corner.corner(
        mcmc["chain"],
        labels=FREE_NAMES,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 7},
        label_kwargs={"fontsize": 7},
    )
    fig.suptitle("Joint ZM15 iHOD — 13 shared parameters", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Joint ZM15 iHOD fit: 13 shared params, 5 BGS Mstar-threshold samples, "
            "wp + SMF."
        )
    )
    p.add_argument("--mode", default="map", choices=["map", "mcmc", "both"],
                   help="map = MAP only; mcmc = MCMC from existing MAP; both = MAP + MCMC")
    p.add_argument("--rp-min",    type=float, default=0.02,
                   help="Minimum rp [Mpc/h] for wp(rp) (default: 0.02)")
    p.add_argument("--f-sys",     type=float, default=0.05,
                   help="Fractional systematic floor on data variances (default: 0.05)")
    p.add_argument("--n-walkers", type=int,   default=32)
    p.add_argument("--n-steps",   type=int,   default=1000)
    p.add_argument("--n-burnin",  type=int,   default=300)
    p.add_argument("--hmf-backend", default="csst",
                   help="HMF backend (default: csst)")
    p.add_argument("--no-plot",   action="store_true", help="Skip output figures")
    args = p.parse_args()

    out_dir = _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    infra = _Infrastructure(hmf_backend=args.hmf_backend)

    map_result = None

    if args.mode in ("map", "both"):
        t0 = time.time()
        map_result = run_map(infra, rp_min=args.rp_min, f_sys=args.f_sys)
        print(f"MAP wall-clock: {time.time()-t0:.0f}s", flush=True)
        _save_map(map_result, out_dir)
        if not args.no_plot:
            plot_bestfit(
                infra, map_result["params"],
                out_dir / "joint_bestfit.pdf",
                rp_min=args.rp_min, f_sys=args.f_sys,
                per_sample_chi2=map_result.get("per_sample"),
            )

    if args.mode in ("mcmc", "both"):
        if map_result is None:
            map_path = out_dir / "joint_map.json"
            if not map_path.exists():
                raise FileNotFoundError(
                    f"MAP result not found at {map_path}. Run --mode map first."
                )
            with open(map_path) as fh:
                map_result = json.load(fh)

        t0 = time.time()
        mcmc = run_mcmc(
            infra, map_result,
            n_walkers=args.n_walkers, n_steps=args.n_steps, n_burnin=args.n_burnin,
            rp_min=args.rp_min, f_sys=args.f_sys,
        )
        print(f"MCMC wall-clock: {time.time()-t0:.0f}s", flush=True)
        _save_mcmc(mcmc, out_dir)
        if not args.no_plot:
            plot_bestfit(
                infra, mcmc["medians"],
                out_dir / "joint_bestfit_mcmc.pdf",
                rp_min=args.rp_min, f_sys=args.f_sys,
            )
            plot_corner(mcmc, out_dir / "joint_corner.pdf")


if __name__ == "__main__":
    main()
