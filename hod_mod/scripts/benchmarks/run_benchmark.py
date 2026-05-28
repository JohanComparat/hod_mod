#!/usr/bin/env python
"""
Run a single HOD literature benchmark.

Usage
-----
    python hod_mod/scripts/benchmarks/run_benchmark.py --model more2015 [--mcmc] [--plot]

Each benchmark loads its config from configs/benchmarks/benchmark_{model}.yml,
fits the HOD model to the paper's published data vector using the paper's
cosmology, and compares the best-fit parameters against the published values.

A JSON result file is written to the configured output directory.
Pass --mcmc to run emcee sampling after the MAP fit (off by default).
"""

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Registry: model key → (config path, published params dict, published errors)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

BENCHMARK_REGISTRY = {
    "more2015": {
        "config": "configs/benchmarks/benchmark_more2015.yml",
        "label": "More+2015 BOSS CMASS (MoreHODModel)",
        "published_params": {
            "log10mmin": (13.03, 0.02),
            "sigma_logm": (0.38, 0.05),
            "log10m1":    (13.80, 0.05),
            "alpha":      (1.17, 0.10),
            "kappa":      (0.51, 0.20),
        },
        "published_chi2_ndof": 0.9,
        "data_status": "ready",
    },
    "kravtsov2004": {
        "config": "configs/benchmarks/benchmark_kravtsov2004.yml",
        "label": "Kravtsov+2004 (Kravtsov04HODModel, BOSS CMASS data)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zheng2007": {
        "config": "configs/benchmarks/benchmark_zheng2007.yml",
        "label": "Zheng+2007 SDSS M_r < -21 (HODModel)",
        "published_params": {
            "log10mmin":  (12.78, 0.10),
            "sigma_logm": (0.68, 0.15),
            "log10m0":    (11.92, 0.30),
            "log10m1":    (13.88, 0.08),
            "alpha":      (1.39, 0.15),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "leauthaud2012": {
        "config": "configs/benchmarks/benchmark_leauthaud2012.yml",
        "label": "Leauthaud+2012 COSMOS z2=[0.48,0.74] (Leauthaud12HODModel)",
        "published_params": {
            "log10m1":      (12.725, 0.032),
            "log10m_star0": (11.038, 0.019),
            "beta":         (0.466, 0.009),
            "delta":        (0.61, 0.13),
            "gamma":        (1.95, 0.25),
            "sigma_logm":   (0.249, 0.019),
        },
        "published_chi2_ndof": 1.6,
        "data_status": "NEEDS_DATA",
    },
    "vanutert2016": {
        "config": "configs/benchmarks/benchmark_vanutert2016.yml",
        "label": "van Uitert+2016 GAMA bin 2 (VanUitert16CSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "NEEDS_DATA",
    },
    "zumandelbaum2015": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015.yml",
        "label": "Zu & Mandelbaum 2015 SDSS (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h":        (12.10, 0.17),
            "lg_m0star":     (10.31, 0.10),
            "beta":          (0.33, 0.21),
            "delta":         (0.42, 0.04),
            "gamma":         (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04),
            "eta":           (-0.04, 0.02),
            "fc":            (0.86, 0.14),
            "bsat":          (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "guo2018": {
        "config": "configs/benchmarks/benchmark_guo2018.yml",
        "label": "Guo+2018 SDSS LOWZ (Guo18ICSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "guo2019": {
        "config": "configs/benchmarks/benchmark_guo2019.yml",
        "label": "Guo+2019 eBOSS ELG (Guo19ICSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zacharegkas2025": {
        "config": "configs/benchmarks/benchmark_zacharegkas2025.yml",
        "label": "Zacharegkas+2025 DES Y3 (Zacharegkas25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "NEEDS_DATA",
    },
    # -----------------------------------------------------------------------
    # Second benchmark tier: ΔΣ-only fits
    # -----------------------------------------------------------------------
    "leauthaud2012_ds": {
        "config": "configs/benchmarks/benchmark_leauthaud2012_ds.yml",
        "label": "Leauthaud+2012 COSMOS z2 — ΔΣ only (Leauthaud12HODModel)",
        "published_params": {
            "log10m1":      (12.725, 0.032),
            "log10m_star0": (11.038, 0.019),
            "beta":         (0.466, 0.009),
            "delta":        (0.61, 0.13),
            "gamma":        (1.95, 0.25),
            "sigma_logm":   (0.249, 0.019),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "vanutert2016_ds": {
        "config": "configs/benchmarks/benchmark_vanutert2016_ds.yml",
        "label": "van Uitert+2016 GAMA+KiDS bin M3 — ΔΣ only (VanUitert16CSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_ds": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_ds.yml",
        "label": "Zu & Mandelbaum 2015 SDSS — ΔΣ only (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h":        (12.10, 0.10),
            "lg_m0star":     (10.31, 0.05),
            "beta":          (0.33, 0.05),
            "delta":         (0.42, 0.05),
            "gamma":         (1.21, 0.10),
            "sigma_lnmstar": (0.50, 0.05),
            "eta":           (-0.04, 0.02),
            "fc":            (0.86, 0.05),
            "bsat":          (8.98, 1.00),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zacharegkas2025_ds": {
        "config": "configs/benchmarks/benchmark_zacharegkas2025_ds.yml",
        "label": "Zacharegkas+2025 DES Y3 bin 1 — ΔΣ only (Zacharegkas25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "NOT_APPLICABLE",
    },
    # -----------------------------------------------------------------------
    # DESI DR1 tracer benchmarks — Lange+2025 (arXiv:2512.15962)
    # Decorated HOD with effective assembly bias; free Omega_m + S8
    # Data: digitized from ar5iv PNG (Fig 3/4), ~20-30% accuracy; Zenodo 17831718 when published
    # -----------------------------------------------------------------------
    "lange2025_bgs2_des": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_des.yml",
        "label": "Lange+2025 DESI BGS2 joint wp+ESD — DES/KiDS (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs3_des": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs3_des.yml",
        "label": "Lange+2025 DESI BGS3 joint wp+ESD — DES/KiDS (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg1_des": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg1_des.yml",
        "label": "Lange+2025 DESI LRG1 joint wp+ESD — DES/KiDS (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_hsc.yml",
        "label": "Lange+2025 DESI BGS2 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs3_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs3_hsc.yml",
        "label": "Lange+2025 DESI BGS3 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg1_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg1_hsc.yml",
        "label": "Lange+2025 DESI LRG1 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg2_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg2_hsc.yml",
        "label": "Lange+2025 DESI LRG2 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_wp": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_wp.yml",
        "label": "Lange+2025 DESI BGS2 wp-only with free cosmo (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_ds_des": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_ds_des.yml",
        "label": "Lange+2025 DESI BGS2 ESD-only — DES/KiDS (Lange25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_ds_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_ds_hsc.yml",
        "label": "Lange+2025 DESI BGS2 ESD-only — HSC-Y3 (Lange25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
}


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model_key: str, mcmc: bool = False, plot: bool = False,
                  output_dir: str | None = None,
                  force_mcmc: bool = False) -> dict:
    """Run one benchmark and return the result dict.

    Parameters
    ----------
    model_key : str
        Key from BENCHMARK_REGISTRY (e.g. ``"more2015"``).
    mcmc : bool
        If True, run emcee sampling after MAP.
    plot : bool
        If True, save comparison figures.
    output_dir : str or None
        Override the output directory from the config.
    force_mcmc : bool
        If True, rerun MCMC even if flatchain.npz already exists.
    """
    from hod_mod.fitting import load_config, WpFitter, JointFitter, DeltaSigmaFitter

    entry = BENCHMARK_REGISTRY[model_key]
    label = entry["label"]
    published = entry["published_params"]
    pub_chi2 = entry["published_chi2_ndof"]
    status = entry["data_status"]

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"{'='*60}")

    if status not in ("ready",):
        print(f"SKIP — data not yet available (status={status})")
        # derive a directory guess for the README path
        key_base = model_key.replace("_ds", "").replace("_", "")
        print(f"  See data/{key_base}*/README_data.md for data extraction instructions.")
        return {"model": model_key, "status": "skipped", "reason": status}

    config_path = os.path.join(_REPO_ROOT, entry["config"])
    config = load_config(config_path)
    if output_dir is not None:
        config = _override_output(config, output_dir)

    os.makedirs(config.output_dir, exist_ok=True)

    # Choose fitter based on available data:
    #   ds_file + data_file → JointFitter  (third benchmark: wp + ΔΣ)
    #   ds_file only        → DeltaSigmaFitter (second benchmark: ΔΣ only)
    #   data_file only      → WpFitter     (first benchmark: wp only)
    has_wp = bool(config.data_file and os.path.isfile(config.data_file))
    has_ds = config.ds_file is not None
    if has_wp and has_ds:
        fitter = JointFitter(config)
        joint = True
    elif has_ds:
        fitter = DeltaSigmaFitter(config)
        joint = False
    else:
        fitter = WpFitter(config)
        joint = False

    # MAP fit
    map_result = fitter.map_fit()
    params = map_result["params"]
    chi2 = map_result["chi2"]
    ndof = map_result["ndof"]
    chi2_ndof = chi2 / ndof if ndof > 0 else float("nan")

    # Optional MCMC
    if mcmc:
        flatchain_path = os.path.join(config.output_dir, "flatchain.npz")
        if os.path.exists(flatchain_path) and not force_mcmc:
            print(f"MCMC already done — skipping ({flatchain_path}).")
            print("  Pass --force-mcmc to rerun.")
        else:
            print("Running MCMC (this may take several minutes)…")
            theta0 = np.asarray(map_result["theta"])
            n_walkers = fitter.config.n_walkers
            scale = np.maximum(np.abs(theta0) * 1e-3, 1e-4)
            rng = np.random.default_rng(42)
            initial_pos = theta0[None, :] + rng.normal(0, scale, (n_walkers, len(theta0)))
            fitter.sample(initial_pos=initial_pos)

    # Print comparison table
    _print_comparison(params, published, chi2_ndof, pub_chi2)

    # Optional plots
    ds_only = has_ds and not has_wp
    if plot:
        _make_plots(fitter, params, chi2_ndof, model_key, config.output_dir,
                    joint=joint, ds_only=ds_only)

    # Build result dict
    passes = chi2_ndof < 2.0 and not np.isnan(chi2_ndof)
    result = {
        "model": model_key,
        "label": label,
        "status": "pass" if passes else "fail",
        "chi2": float(chi2),
        "ndof": int(ndof),
        "chi2_ndof": float(chi2_ndof),
        "published_chi2_ndof": pub_chi2,
        "params": {k: float(v) for k, v in params.items()},
        "published_params": {k: list(v) for k, v in _normalize_published(published).items()},
        "param_deviations_sigma": _deviations(params, published),
    }

    out_file = os.path.join(config.output_dir, "benchmark_result.json")
    with open(out_file, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResult saved: {out_file}")
    print(f"Benchmark: {'PASSED' if passes else 'FAILED'}")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _override_output(config, output_dir):
    from dataclasses import replace
    return replace(config, output_dir=output_dir)


def _normalize_published(published: dict) -> dict:
    """Ensure every entry is a (value, error) tuple; plain floats get error=0."""
    return {
        k: v if isinstance(v, tuple) else (v, 0.0)
        for k, v in published.items()
    }


def _deviations(params: dict, published: dict) -> dict:
    published = _normalize_published(published)
    devs = {}
    for pname, (pub_val, pub_err) in published.items():
        bfit = params.get(pname, float("nan"))
        devs[pname] = float((bfit - pub_val) / pub_err) if pub_err > 0 else float("nan")
    return devs


def _print_comparison(params, published, chi2_ndof, pub_chi2):
    published = _normalize_published(published)
    print(f"\nchi2/ndof = {chi2_ndof:.3f}", end="")
    if pub_chi2 is not None:
        print(f"  (published: {pub_chi2:.2f})")
    else:
        print()
    if published:
        w = max(len(k) for k in published) + 2
        print(f"\n{'Parameter':{w}s}  {'Best-fit':>10s}  {'Published':>10s}  {'Δ/σ':>8s}")
        print("-" * (w + 34))
        for pname, (pub_val, pub_err) in published.items():
            bfit = params.get(pname, float("nan"))
            diff = (bfit - pub_val) / pub_err if pub_err > 0 else float("nan")
            diff_str = f"{diff:8.2f}σ" if not (diff != diff) else "      n/a"
            print(f"{pname:{w}s}  {bfit:10.4f}  {pub_val:10.4f}  {diff_str}")


def _make_plots(fitter, params, chi2_ndof, model_key, output_dir, joint, ds_only=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # DS-only benchmark: skip wp panel, jump straight to ΔΣ plot
    if ds_only and hasattr(fitter, "predict_ds"):
        R = np.array(fitter.R_arr)
        ds_obs = np.array(fitter.ds_obs)
        ds_err = np.sqrt(np.diag(np.linalg.inv(fitter.icov_ds))) if hasattr(fitter, "icov_ds") else np.ones_like(ds_obs)
        ds_pred = np.array(fitter.predict_ds(params))

        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
        axes2[0].errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=4, color="C1", label="Data")
        axes2[0].loglog(R, ds_pred, "-", color="C1", label=f"MAP (χ²/dof={chi2_ndof:.2f})")
        axes2[0].set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
        axes2[0].legend(fontsize=9)
        axes2[0].set_title(f"Benchmark: {model_key} — ΔΣ only", fontsize=10)

        ratio2 = ds_obs / ds_pred - 1
        ratio2_err = ds_err / ds_pred
        axes2[1].axhline(0, color="k", lw=0.8, ls="--")
        axes2[1].errorbar(R, ratio2, yerr=ratio2_err, fmt="s", ms=4, color="C1")
        axes2[1].set_xlabel(r"$R$ [$h^{-1}$ Mpc]")
        axes2[1].set_ylabel(r"data/model $-1$")
        axes2[1].set_ylim(-0.6, 0.6)

        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, f"benchmark_{model_key}_ds.png"), dpi=150)
        plt.close(fig2)
        return

    rp = np.array(fitter.rp_arr)
    wp_obs = np.array(fitter.wp_obs)
    wp_err = np.sqrt(np.diag(np.linalg.inv(fitter.icov_wp))) if hasattr(fitter, "icov_wp") else np.ones_like(wp_obs)
    wp_pred = np.array(fitter.predict_wp(params))

    # Compute published best-fit model if published_params are available
    entry = BENCHMARK_REGISTRY.get(model_key, {})
    published = entry.get("published_params", {})
    wp_pub = None
    if published:
        pub_params = dict(params)
        for pname, entry_val in _normalize_published(published).items():
            pub_params[pname] = entry_val[0]
        try:
            wp_pub = np.array(fitter.predict_wp(pub_params))
        except Exception:
            wp_pub = None

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=4, label="Data")
    axes[0].loglog(rp, wp_pred, "-", label=f"MAP (χ²/dof={chi2_ndof:.2f})")
    if wp_pub is not None:
        axes[0].loglog(rp, wp_pub, "--", color="C2", lw=1.5,
                       label="Published best-fit")
    axes[0].set_ylabel(r"$w_p(r_p)$ [$h^{-1}$ Mpc]")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Benchmark: {model_key}", fontsize=10)

    ratio = wp_obs / wp_pred - 1
    ratio_err = wp_err / wp_pred
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].errorbar(rp, ratio, yerr=ratio_err, fmt="o", ms=4)
    if wp_pub is not None:
        ratio_pub = wp_obs / wp_pub - 1
        axes[1].plot(rp, ratio_pub, "--", color="C2", lw=1.5)
    axes[1].axhline(0.1, color="gray", lw=0.5, ls=":")
    axes[1].axhline(-0.1, color="gray", lw=0.5, ls=":")
    axes[1].set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
    axes[1].set_ylabel(r"data/model $-1$")
    axes[1].set_ylim(-0.6, 0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"benchmark_{model_key}_wp.png"), dpi=150)
    plt.close(fig)

    if joint and hasattr(fitter, "predict_ds"):
        R = np.array(fitter.R_arr)
        ds_obs = np.array(fitter.ds_obs)
        ds_err = np.sqrt(np.diag(np.linalg.inv(fitter.icov_ds))) if hasattr(fitter, "icov_ds") else np.ones_like(ds_obs)
        ds_pred = np.array(fitter.predict_ds(params))

        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
        axes2[0].errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=4, color="C1", label="Data")
        axes2[0].loglog(R, ds_pred, "-", color="C1", label=f"MAP (χ²/dof={chi2_ndof:.2f})")
        axes2[0].set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
        axes2[0].legend(fontsize=9)
        axes2[0].set_title(f"Benchmark: {model_key} — ΔΣ", fontsize=10)

        ratio2 = ds_obs / ds_pred - 1
        ratio2_err = ds_err / ds_pred
        axes2[1].axhline(0, color="k", lw=0.8, ls="--")
        axes2[1].errorbar(R, ratio2, yerr=ratio2_err, fmt="s", ms=4, color="C1")
        axes2[1].set_xlabel(r"$R$ [$h^{-1}$ Mpc]")
        axes2[1].set_ylabel(r"data/model $-1$")
        axes2[1].set_ylim(-0.6, 0.6)

        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, f"benchmark_{model_key}_ds.png"), dpi=150)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run a single HOD literature benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(BENCHMARK_REGISTRY)}",
    )
    p.add_argument("--model", required=True, choices=list(BENCHMARK_REGISTRY),
                   help="Benchmark identifier")
    p.add_argument("--mcmc", action="store_true",
                   help="Run emcee MCMC after MAP (slow)")
    p.add_argument("--force-mcmc", action="store_true",
                   help="Rerun MCMC even if flatchain.npz already exists")
    p.add_argument("--plot", action="store_true",
                   help="Save comparison figures to output dir")
    p.add_argument("--output", default=None,
                   help="Override output directory from config")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_benchmark(args.model, mcmc=args.mcmc, plot=args.plot,
                           output_dir=args.output,
                           force_mcmc=args.force_mcmc)
    sys.exit(0 if result.get("status") in ("pass", "skipped") else 1)
