"""MAP fits on BGS M10 w_p(rp) for linear vs non-linear halo models.

Compares three model configurations:
  1. MoreHODModel + linear 2-halo  (More+2015 prescription)
  2. MoreHODModel + HMcode nl 2-halo  (using P_nl for 2-halo term)
  3. Leauthaud+2012 HOD + HMcode nl 2-halo  (SHMR-based occupation)

All fitted to the BGS LS10 DR10 projected correlation function for the
stellar mass threshold log10(M*/M_sun) > 10.0, z = 0.05–0.18.

Usage:
    python scripts/fitting/bgs_ls10/fit_bgs_m10_nonlinear_models.py [--plot]
"""

import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.optimize as opt
import jax.numpy as jnp

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.cosmology.nonlinear import HALOFITSpectrum, CachedPkNonlinear
from hod_mod.galaxies.hod import MoreHODModel, Leauthaud12HODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.data_io.sum_stat_reader import SumStatReader

# ---------------------------------------------------------------------------
# Data + cosmology
# ---------------------------------------------------------------------------

_DATA_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "sum_stat", "data",
        "BGS_Mstar10.0",
        "LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_joint_smf-wp-esd_hsc-esd_des-esd_kids-wtheta-knn-sys-comb.h5",
    )
)

_THETA = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "flat": True,
    "H0":     _THETA["h"] * 100.0,
    "Om0":    _THETA["Omega_m"],
    "Ob0":    _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns":     _THETA["n_s"],
}

_Z_EFF    = 0.115
_PI_MAX   = 100.0
_RP_MIN   = 0.3
_RP_MAX   = 30.0


def load_data(data_file):
    reader = SumStatReader.from_hdf5(data_file)
    d      = reader.wp()
    rp     = np.asarray(d["rp"])
    wp     = np.asarray(d["wp"])
    cov    = np.asarray(d["cov"])
    mask   = (rp >= _RP_MIN) & (rp <= _RP_MAX)
    rp     = rp[mask]
    wp     = wp[mask]
    cov    = cov[np.ix_(mask, mask)]
    reg    = 0.01 * np.diag(np.diag(cov))
    icov   = np.linalg.inv(cov + reg)
    return rp, wp, icov


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def build_predictors():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    pk_nl  = CachedPkNonlinear(HALOFITSpectrum("mead2020"))

    hod_more = MoreHODModel(hmf, hmf.bias)
    hod_l12  = Leauthaud12HODModel(hmf, hmf.bias)

    pred_more_lin = FullHaloModelPrediction(pk_lin, hod_more, hp)
    pred_more_nl  = FullHaloModelPrediction(pk_lin, hod_more, hp, pk_nl=pk_nl, nl_2halo=True)
    pred_l12_nl   = FullHaloModelPrediction(pk_lin, hod_l12,  hp, pk_nl=pk_nl, nl_2halo=True)

    return pred_more_lin, pred_more_nl, pred_l12_nl


# ---------------------------------------------------------------------------
# MAP fitting
# ---------------------------------------------------------------------------

def chi2(pred, hod_params, rp_arr, wp_obs, icov):
    try:
        wp_pred = np.asarray(
            pred.wp(jnp.array(rp_arr), _PI_MAX, _Z_EFF, _THETA, hod_params)
        )
        if not np.all(np.isfinite(wp_pred)):
            return 1e10
        resid = wp_pred - wp_obs
        return float(resid @ icov @ resid)
    except Exception:
        return 1e10


# --- Model 1 & 2: MoreHODModel ---

_MORE_FREE    = ["log10mmin", "sigma_logm", "log10m1", "alpha", "kappa"]
_MORE_FIXED   = {"alpha_inc": 0.0, "log10m_inc": 12.0}
_MORE_INIT    = [12.0, 0.38, 13.1, 1.0, 1.0]
_MORE_BOUNDS  = [(10.5, 13.5), (0.05, 1.5), (11.5, 14.5), (0.5, 3.0), (0.1, 5.0)]


def _more_params(x):
    return dict(zip(_MORE_FREE, x), **_MORE_FIXED)


def map_more(pred, rp_arr, wp_obs, icov, label=""):
    def obj(x):
        # Bounds enforcement
        for v, (lo, hi) in zip(x, _MORE_BOUNDS):
            if not (lo <= v <= hi):
                return 1e10
        return chi2(pred, _more_params(x), rp_arr, wp_obs, icov)

    t0  = time.time()
    res = opt.minimize(obj, _MORE_INIT, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4})
    elapsed = time.time() - t0
    params  = _more_params(res.x)
    nbin    = len(rp_arr)
    ndof    = nbin - len(_MORE_FREE)
    c2dof   = res.fun / max(ndof, 1)
    print(f"\n{label}")
    print(f"  Converged: {res.success}  iterations: {res.nit}  time: {elapsed:.1f}s")
    print(f"  χ²/dof = {res.fun:.2f}/{ndof} = {c2dof:.3f}")
    for k in _MORE_FREE:
        print(f"    {k:20s} = {params[k]:.4f}")
    return {"params": params, "chi2": float(res.fun), "ndof": ndof, "success": res.success}


# --- Model 3: Leauthaud12HODModel ---

_L12_FREE   = ["log10m1", "sigma_logm", "log10m_sat", "log10m_cut", "alpha_sat"]
_L12_FIXED  = {
    "log10m_star0": 10.916, "beta": 0.457, "delta": 0.566,
    "gamma": 1.53, "log10m_star_thresh": 10.0,
}
_L12_INIT   = [12.5, 0.25, 13.0, 11.5, 1.0]
_L12_BOUNDS = [(11.5, 14.0), (0.05, 1.0), (12.0, 14.5), (10.5, 13.0), (0.3, 2.0)]


def _l12_params(x):
    return dict(zip(_L12_FREE, x), **_L12_FIXED)


def map_l12(pred, rp_arr, wp_obs, icov, label=""):
    def obj(x):
        for v, (lo, hi) in zip(x, _L12_BOUNDS):
            if not (lo <= v <= hi):
                return 1e10
        return chi2(pred, _l12_params(x), rp_arr, wp_obs, icov)

    t0  = time.time()
    res = opt.minimize(obj, _L12_INIT, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4})
    elapsed = time.time() - t0
    params  = _l12_params(res.x)
    nbin    = len(rp_arr)
    ndof    = nbin - len(_L12_FREE)
    c2dof   = res.fun / max(ndof, 1)
    print(f"\n{label}")
    print(f"  Converged: {res.success}  iterations: {res.nit}  time: {elapsed:.1f}s")
    print(f"  χ²/dof = {res.fun:.2f}/{ndof} = {c2dof:.3f}")
    for k in _L12_FREE:
        print(f"    {k:20s} = {params[k]:.4f}")
    return {"params": params, "chi2": float(res.fun), "ndof": ndof, "success": res.success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--output-json", type=str,
                        default="results/bgs_m10_nonlinear_models.json")
    args = parser.parse_args()

    print(f"Loading BGS M10 wp data from:\n  {_DATA_FILE}")
    rp, wp_obs, icov = load_data(_DATA_FILE)
    print(f"  N_rp bins in [{_RP_MIN},{_RP_MAX}] Mpc/h: {len(rp)}")
    print(f"  wp range: [{wp_obs.min():.1f}, {wp_obs.max():.1f}] Mpc/h")

    print("\nBuilding predictors…")
    pred_more_lin, pred_more_nl, pred_l12_nl = build_predictors()
    print("  Done.")

    results = {}
    results["more_lin"] = map_more(pred_more_lin, rp, wp_obs, icov,
                                   "Model 1: MoreHODModel + linear 2-halo")
    results["more_nl"]  = map_more(pred_more_nl,  rp, wp_obs, icov,
                                   "Model 2: MoreHODModel + HMcode nl 2-halo")
    results["l12_nl"]   = map_l12(pred_l12_nl,    rp, wp_obs, icov,
                                  "Model 3: Leauthaud+2012 HOD + HMcode nl 2-halo")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  {'Model':40s}  {'χ²/dof':>10}")
    for k, v in results.items():
        c2dof = v["chi2"] / max(v["ndof"], 1)
        print(f"  {k:40s}  {c2dof:10.3f}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to: {args.output_json}")

    if args.plot or args.save:
        _make_figure(rp, wp_obs, results, pred_more_lin, pred_more_nl, pred_l12_nl,
                     args.plot, args.save)

    return results


def _make_figure(rp, wp_obs, results, pred_more_lin, pred_more_nl, pred_l12_nl,
                 show, save_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(rp, wp_obs, fmt="ko", ms=5, label="BGS M10 data")

    rp_j = jnp.array(rp)
    for (key, res, pred, ls, col, lbl) in [
        ("more_lin", results["more_lin"], pred_more_lin, "-",  "#1f77b4",
         "More+2015 HOD, linear 2h"),
        ("more_nl",  results["more_nl"],  pred_more_nl,  "--", "#ff7f0e",
         "More+2015 HOD, HMcode nl 2h"),
        ("l12_nl",   results["l12_nl"],   pred_l12_nl,   "-.", "#2ca02c",
         "Leauthaud+2012 HOD, HMcode nl 2h"),
    ]:
        wp_pred = np.asarray(
            pred.wp(rp_j, _PI_MAX, _Z_EFF, _THETA, res["params"])
        )
        c2dof = res["chi2"] / max(res["ndof"], 1)
        ax.loglog(rp, wp_pred, ls=ls, color=col,
                  label=f"{lbl}  (χ²/dof={c2dof:.2f})")

    ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax.set_ylabel(r"$w_p(r_p)$ [Mpc/$h$]")
    ax.set_title("BGS M10 MAP fits — linear vs non-linear 2-halo")
    ax.legend(fontsize=9)
    ax.set_xlim(0.25, 35)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
