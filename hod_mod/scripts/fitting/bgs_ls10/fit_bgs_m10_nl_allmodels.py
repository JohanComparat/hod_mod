"""MAP fits on BGS M10 w_p(rp) — all HOD models × non-linear 2-halo backends.

Compares all nine HOD/ICSMF/CLF models with the non-linear 2-halo term enabled,
using two P_nl backends (HMcode-2020 and Aletheia) so the backend dependence of
each inferred HOD can be assessed.

Data:
    BGS LS10 DR10, stellar-mass threshold log10(M*/M_sun) > 10.0, z = 0.05-0.18

Usage:
    # both backends (default)
    python scripts/fitting/bgs_ls10/fit_bgs_m10_nl_allmodels.py \\
        --backend all \\
        --output-dir results/bgs_m10_nl_allmodels \\
        --save results/bgs_m10_nl_allmodels/fig_wp_survey.png

    # single backend or model subset
    python scripts/fitting/bgs_ls10/fit_bgs_m10_nl_allmodels.py --backend hmcode
    python scripts/fitting/bgs_ls10/fit_bgs_m10_nl_allmodels.py \\
        --backend hmcode --models MoreHODModel,HODModel
"""

import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.optimize as opt
import jax.numpy as jnp

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.paths import results_root
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.core.nonlinear import (
    CachedPkNonlinear, HALOFITSpectrum, NonLinearPowerSpectrum,
)
from hod_mod.connection.hod import (
    HODModel, MoreHODModel,
    Guo18ICSMFModel, Guo19ICSMFModel,
    Zacharegkas25HODModel, VanUitert16CSMFModel,
    ZuMandelbaum15HODModel, Leauthaud12HODModel,
)
from hod_mod.connection.clf import CLFModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.data_io.sum_stat_reader import SumStatReader

# ---------------------------------------------------------------------------
# Data file + shared constants
# ---------------------------------------------------------------------------

_DATA_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "sum_stat", "data",
        "BGS_Mstar10.0",
        "LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_"
        "joint_smf-wp-esd_hsc-esd_des-esd_kids-wtheta-knn-sys-comb.h5",
    )
)

_THETA = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "flat":   True,
    "H0":     _THETA["h"] * 100.0,
    "Om0":    _THETA["Omega_m"],
    "Ob0":    _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns":     _THETA["n_s"],
}

_Z_EFF  = 0.115
_PI_MAX = 100.0
_RP_MIN = 0.3
_RP_MAX = 30.0

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_HOD_CLASSES = {
    "HODModel":               HODModel,
    "MoreHODModel":           MoreHODModel,
    "Leauthaud12HODModel":    Leauthaud12HODModel,
    "Guo18ICSMFModel":        Guo18ICSMFModel,
    "Guo19ICSMFModel":        Guo19ICSMFModel,
    "Zacharegkas25HODModel":  Zacharegkas25HODModel,
    "VanUitert16CSMFModel":   VanUitert16CSMFModel,
    "ZuMandelbaum15HODModel": ZuMandelbaum15HODModel,
    "CLFModel":               CLFModel,
}

MODEL_SPECS = {

    "HODModel": {
        "free":   ["log10mmin", "sigma_logm", "log10m0", "log10m1", "alpha"],
        "fixed":  {},
        "init":   [11.5, 0.25, 11.3, 12.4, 1.0],
        "bounds": [(10.5, 13.5), (0.05, 1.5), (10.0, 13.0),
                   (11.5, 14.5), (0.5, 3.0)],
        "single_arg": False,
    },

    "MoreHODModel": {
        "free":   ["log10mmin", "sigma_logm", "log10m1", "alpha", "kappa"],
        "fixed":  {"alpha_inc": 0.0, "log10m_inc": 12.0},
        "init":   [12.0, 0.38, 13.1, 1.0, 1.0],
        "bounds": [(10.5, 13.5), (0.05, 1.5), (11.5, 14.5),
                   (0.5, 3.0), (0.1, 5.0)],
        "single_arg": False,
    },

    "Leauthaud12HODModel": {
        "free":   ["log10m1", "sigma_logm", "log10m_sat", "log10m_cut", "alpha_sat"],
        "fixed":  {"log10m_star0": 10.916, "beta": 0.457, "delta": 0.566,
                   "gamma": 1.53, "log10m_star_thresh": 10.0},
        "init":   [12.5, 0.25, 13.0, 11.5, 1.0],
        "bounds": [(11.5, 14.0), (0.05, 1.0), (12.0, 14.5),
                   (10.5, 13.0), (0.3, 2.0)],
        "single_arg": False,
    },

    "Guo18ICSMFModel": {
        "free":   ["log10m_star0", "log10m1_shmr", "alpha_shmr", "beta_shmr",
                   "log10m1_sat", "alpha_sat"],
        "fixed":  {"sigma_logm_star": 0.15, "f_cen": 1.0,
                   "log10m_star_min_cen": 10.0, "sigma_c_cen": 0.1,
                   "f_sat": 1.0, "log10m_star_min_sat": 9.8, "sigma_c_sat": 0.2},
        "init":   [10.7, 11.9, 0.3, 1.5, 13.0, 1.0],
        "bounds": [(10.0, 11.5), (11.0, 13.0), (0.1, 1.0), (0.5, 3.0),
                   (12.0, 14.5), (0.5, 2.0)],
        "single_arg": False,
    },

    "Guo19ICSMFModel": {
        "free":   ["log10m_star0", "log10m1_shmr", "alpha_shmr", "beta_shmr",
                   "log10m1_sat", "alpha_sat", "log10m_q"],
        "fixed":  {"sigma_logm_star": 0.15, "f_cen": 1.0,
                   "log10m_star_min_cen": 10.0, "sigma_c_cen": 0.1,
                   "f_sat": 1.0, "log10m_star_min_sat": 9.8, "sigma_c_sat": 0.2},
        "init":   [10.0, 11.5, 0.3, 1.5, 12.5, 1.0, 12.0],
        "bounds": [(9.5, 11.5), (10.5, 13.0), (0.1, 1.0), (0.5, 3.0),
                   (11.5, 14.5), (0.5, 2.0), (11.0, 13.5)],
        "single_arg": False,
    },

    "Zacharegkas25HODModel": {
        "free":   ["log10m1_shmr", "log10eps", "alpha_shmr",
                   "gamma_shmr", "delta_shmr", "B_sat"],
        "fixed":  {"log10m_star_lo": 10.0, "log10m_star_hi": 12.0,
                   "sigma_logm_star": 0.3, "f_cen": 1.0,
                   "alpha_sat": 1.0, "kappa": 1.0,
                   "beta_sat": 1.0, "B_cut": 5.0, "beta_cut": 1.0, "f_sat": 1.0},
        "init":   [11.5, -1.6, -1.6, 0.6, 3.8, 10.0],
        "bounds": [(10.5, 13.0), (-3.0, 0.0), (-3.0, 0.0),
                   (0.1, 2.0), (1.0, 8.0), (1.0, 30.0)],
        "single_arg": True,
    },

    "VanUitert16CSMFModel": {
        "free":   ["log10m_h1", "log10m_star0", "beta1", "log10_beta2",
                   "sigma_c", "alpha_s"],
        "fixed":  {"log10m_star_lo": 10.0, "log10m_star_hi": 12.0,
                   "b0": 0.0, "b1": 1.5},
        "init":   [11.5, 10.5, 5.0, -0.5, 0.15, -1.1],
        "bounds": [(10.5, 13.0), (10.0, 11.5), (1.0, 15.0), (-2.0, 1.0),
                   (0.05, 0.5), (-2.0, -0.1)],
        "single_arg": True,
    },

    "ZuMandelbaum15HODModel": {
        "free":   ["lg_m1h", "lg_m0star", "beta", "delta", "gamma",
                   "sigma_lnmstar", "fc", "bsat", "alpha_sat"],
        "fixed":  {"log10m_star_thresh": 10.0, "eta": -0.04,
                   "beta_sat": 0.90, "bcut": 0.86, "beta_cut": 0.41},
        "init":   [12.1, 10.3, 0.33, 0.42, 1.21, 0.50, 0.86, 9.0, 1.0],
        "bounds": [(11.0, 13.5), (9.5, 11.5), (0.1, 1.0), (0.1, 2.0),
                   (0.5, 3.0), (0.1, 1.5), (0.5, 1.0), (1.0, 20.0), (0.5, 2.5)],
        "single_arg": True,
    },

    "CLFModel": {
        "free":   ["log10m1", "log10l0", "alpha_cen", "beta_cen",
                   "sigma_c", "b_sat", "alpha_sat"],
        "fixed":  {"log10l_lim": 9.5},
        "init":   [11.0, 9.94, 2.95, 0.18, 0.15, 9.0, -1.15],
        "bounds": [(10.0, 13.0), (9.0, 11.0), (0.5, 6.0), (0.05, 1.5),
                   (0.05, 0.5), (1.0, 30.0), (-2.0, -0.1)],
        "single_arg": False,
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_file):
    reader = SumStatReader.from_hdf5(data_file)
    d    = reader.wp()
    rp   = np.asarray(d["rp"])
    wp   = np.asarray(d["wp"])
    cov  = np.asarray(d["cov"])
    mask = (rp >= _RP_MIN) & (rp <= _RP_MAX)
    rp   = rp[mask]
    wp   = wp[mask]
    cov  = cov[np.ix_(mask, mask)]
    reg  = 0.01 * np.diag(np.diag(cov))
    icov = np.linalg.inv(cov + reg)
    return rp, wp, icov

# ---------------------------------------------------------------------------
# Backend + HOD construction
# ---------------------------------------------------------------------------

def build_pk_nl(backend: str):
    if backend == "hmcode":
        return CachedPkNonlinear(HALOFITSpectrum("mead2020"))
    elif backend == "aletheia":
        return CachedPkNonlinear(NonLinearPowerSpectrum("aletheia"))
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def build_hod(name: str, hmf):
    cls  = _HOD_CLASSES[name]
    spec = MODEL_SPECS[name]
    return cls(hmf) if spec["single_arg"] else cls(hmf, hmf.bias)

# ---------------------------------------------------------------------------
# chi2 + MAP fit
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


def map_fit_model(pred, spec, rp_arr, wp_obs, icov, label=""):
    free, fixed = spec["free"], spec["fixed"]
    bounds, x0  = spec["bounds"], list(spec["init"])

    def obj(x):
        for v, (lo, hi) in zip(x, bounds):
            if not (lo <= v <= hi):
                return 1e10
        return chi2(pred, dict(zip(free, x), **fixed), rp_arr, wp_obs, icov)

    t0  = time.time()
    res = opt.minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4})
    elapsed = time.time() - t0
    params  = dict(zip(free, res.x), **fixed)
    ndof    = len(rp_arr) - len(free)
    c2dof   = res.fun / max(ndof, 1)
    print(f"\n{label}")
    print(f"  converged={res.success}  nit={res.nit}  time={elapsed:.0f}s")
    print(f"  chi2/dof = {res.fun:.2f}/{ndof} = {c2dof:.3f}")
    for k in free:
        print(f"    {k:25s} = {params[k]:.4f}")
    return {
        "params":  params,
        "chi2":    float(res.fun),
        "ndof":    ndof,
        "success": bool(res.success),
        "n_free":  len(free),
        "elapsed": round(elapsed, 1),
    }

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _make_figure(rp, wp_obs, all_results, backends, hmf, hp, pk_lin,
                 active_models, save_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available — skipping figure")
        return

    n_models = len(active_models)
    ncols    = 3
    nrows    = (n_models + ncols - 1) // ncols
    fig      = plt.figure(figsize=(5 * ncols, 4 * nrows))
    gs       = gridspec.GridSpec(nrows, ncols, figure=fig,
                                  hspace=0.45, wspace=0.35)

    colors   = {"hmcode": "#1f77b4", "aletheia": "#d62728"}
    ls_map   = {"hmcode": "-",       "aletheia": "--"}

    rp_j = jnp.array(rp)

    for idx, model_name in enumerate(active_models):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.errorbar(rp, wp_obs, fmt="ko", ms=3, lw=0.8,
                    label="data", zorder=5)

        for backend in backends:
            if model_name not in all_results.get(backend, {}):
                continue
            res    = all_results[backend][model_name]
            spec   = MODEL_SPECS[model_name]
            pk_nl  = build_pk_nl(backend)
            hod    = build_hod(model_name, hmf)
            pred   = FullHaloModelPrediction(pk_lin, hod, hp,
                                              pk_nl=pk_nl, nl_2halo=True)
            try:
                wp_pred = np.asarray(
                    pred.wp(rp_j, _PI_MAX, _Z_EFF, _THETA, res["params"])
                )
            except Exception:
                continue
            c2dof = res["chi2"] / max(res["ndof"], 1)
            ax.loglog(rp, wp_pred, ls_map[backend],
                      color=colors[backend], lw=1.5,
                      label=f"{backend}  χ²/dof={c2dof:.2f}")

        ax.set_xlabel(r"$r_p\,[h^{-1}\,\mathrm{Mpc}]$", fontsize=8)
        ax.set_ylabel(r"$w_p\,[h^{-1}\,\mathrm{Mpc}]$", fontsize=8)
        ax.set_title(model_name, fontsize=8)
        ax.set_xlim(0.25, 35)
        ax.legend(fontsize=6)

    fig.suptitle(r"BGS M10 MAP fits — non-linear 2-halo"
                 r"  ($\log M_*/M_\odot > 10$, $z_{\rm eff}=0.115$)",
                 fontsize=10, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MAP fits: all HOD models × non-linear 2-halo backends"
    )
    parser.add_argument("--backend", default="all",
                        choices=["hmcode", "aletheia", "all"],
                        help="P_nl backend(s) to run")
    parser.add_argument("--models", default=None,
                        help="Comma-separated subset of model names to run "
                             "(default: all 9)")
    parser.add_argument("--output-dir", default=str(results_root() / "bgs_m10_nl_allmodels"),
                        help="Directory for JSON and figure output")
    parser.add_argument("--save", default=None,
                        help="Path for comparison figure (PNG/PDF)")
    args = parser.parse_args()

    print(f"Loading BGS M10 wp data from:\n  {_DATA_FILE}")
    rp, wp_obs, icov = load_data(_DATA_FILE)
    print(f"  N_rp bins in [{_RP_MIN},{_RP_MAX}] Mpc/h: {len(rp)}")
    print(f"  wp range: [{wp_obs.min():.1f}, {wp_obs.max():.1f}] Mpc/h")

    active_models = (
        [m.strip() for m in args.models.split(",")]
        if args.models else list(MODEL_SPECS.keys())
    )
    for m in active_models:
        if m not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {m!r}.  "
                             f"Available: {list(MODEL_SPECS)}")

    backends = (["hmcode", "aletheia"] if args.backend == "all"
                else [args.backend])

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nBuilding shared pipeline (HMF + halo profile)…")
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("csst")
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    print("  Done.")

    all_results = {}

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Backend: {backend}")
        print(f"{'='*60}")
        pk_nl = build_pk_nl(backend)
        backend_results = {}

        for model_name in active_models:
            spec = MODEL_SPECS[model_name]
            hod  = build_hod(model_name, hmf)
            pred = FullHaloModelPrediction(pk_lin, hod, hp,
                                            pk_nl=pk_nl, nl_2halo=True)
            label  = f"{model_name} / {backend}"
            result = map_fit_model(pred, spec, rp, wp_obs, icov, label=label)
            backend_results[model_name] = result

        all_results[backend] = backend_results

        # Save immediately — fault-tolerant against long runs
        out_json = os.path.join(args.output_dir, f"{backend}_results.json")
        with open(out_json, "w") as fh:
            json.dump(backend_results, fh, indent=2)
        print(f"\n→ Saved {backend} results: {out_json}")

    # Summary table
    header_be = "".join(f"  {'chi2/dof ['+be+']':>22}" for be in backends)
    print("\n" + "=" * (42 + 24 * len(backends)))
    print(f"  {'Model':38s}{header_be}")
    for name in active_models:
        row = ""
        for be in backends:
            v     = all_results.get(be, {}).get(name, {})
            c2dof = v["chi2"] / max(v["ndof"], 1) if v else float("nan")
            row  += f"  {c2dof:>22.3f}"
        print(f"  {name:38s}{row}")

    if args.save:
        _make_figure(rp, wp_obs, all_results, backends,
                     hmf, hp, pk_lin, active_models, args.save)

    return all_results


if __name__ == "__main__":
    main()
