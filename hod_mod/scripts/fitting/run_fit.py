"""YAML-driven wrapper for BGS multi-probe HOD fitting.

Reads a config file from ``configs/fitting/`` and runs
:class:`~hod_mod.scripts.fitting.bgs_ls10.fit_bgs_multiprobe.MultiProbeFitter`
with the parameters specified in the YAML.

All physics flags and scale cuts are stored in the YAML so the fit is fully
reproducible from a single file.  Output is written to a sub-directory of
``output.base_dir`` whose name encodes the key settings.

Usage
-----
Full fit (MAP + MCMC)::

    python hod_mod/scripts/fitting/run_fit.py \\
        configs/fitting/BGS_LS10_Comparat2025_Mstar10_LargeScaleCuts.yml

MAP only (fast verification)::

    python hod_mod/scripts/fitting/run_fit.py \\
        configs/fitting/BGS_LS10_Comparat2025_Mstar10_LargeScaleCuts.yml \\
        --map-only

Override MCMC settings::

    python hod_mod/scripts/fitting/run_fit.py configs/fitting/....yml \\
        --n-walkers 128 --n-steps 5000

YAML schema
-----------
See ``configs/fitting/BGS_LS10_Comparat2025_Mstar10_LargeScaleCuts.yml`` for
a fully documented example.  Required top-level sections: ``data``, ``model``,
``fitting``, ``output``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import yaml

# Resolve the repo root so relative imports work regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hod_mod.paths import results_root
from hod_mod.scripts.fitting.bgs_ls10.fit_bgs_multiprobe import (
    MultiProbeFitter,
    BGS_BINS,
    SUM_STAT_DIR,
    _find_data_file,
)
from hod_mod.scripts.benchmarks.benchmark_plots import (
    _COL_DATA,
    _COL_MAP,
    load_flatchain  as _load_flatchain,
    mcmc_bands      as _mcmc_bands,
    add_bands       as _add_bands,
    residual_panel  as _residual_panel,
    plot_hod        as _plot_hod,
    plot_corner     as _plot_corner_fn,
)


def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _build_output_dir(cfg: dict, base_dir: str) -> str:
    """Construct an output directory name encoding the key fit settings."""
    d    = cfg["data"]
    m    = cfg["model"]
    fit  = cfg.get("fitting", {})

    mstar_str  = f"{d['mstar_lo']:.1f}"
    probes_str = "_".join(d.get("probes", ["wp", "esd_hsc"]))
    hod_model  = m.get("hod_model", "more2015")
    profile    = m.get("profile",   "nfw")
    rp_min_wp  = float(d.get("rp_min_wp", 0.3))
    rp_tag     = f"rp{int(round(rp_min_wp * 1000)):03d}"

    flags = "_".join(filter(None, [
        "fcosmo"  if m.get("use_free_cosmo",      False) else "",
        "ia"      if m.get("use_ia",              False) else "",
        "offcen"  if m.get("use_offcentering",    False) else "",
        "bfrac"   if m.get("use_baryon_fraction", False) else "",
        "stellar" if m.get("use_stellar_mass",    False) else "",
        "inc"     if m.get("use_incompleteness",  False) else "",
    ]))

    dir_name = f"mstar{mstar_str}_{probes_str}_{hod_model}_{profile}_{rp_tag}"
    if flags:
        dir_name = f"{dir_name}_{flags}"

    return os.path.join(base_dir, dir_name)


def _make_plots(fitter, params, chi2_ndof, output_dir, probes, mstar_lo, z_eff):
    """Generate per-probe and combined figures for a BGS multi-probe HOD fit."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    flatchain, fc_names = _load_flatchain(output_dir)
    tag        = f"bgs_mstar{mstar_lo:.1f}"
    title_base = rf"$M_* > 10^{{{mstar_lo:.1f}}}\ M_\odot$,  $z_{{\rm eff}}={z_eff:.3f}$"

    # ── probe metadata ──────────────────────────────────────────────────────
    _probe_cfg = {
        "wp": dict(
            get_rp   = lambda f: np.array(f.rp_wp),
            get_obs  = lambda f: np.array(f.wp_obs),
            get_err  = lambda f: np.array(f.wp_err),
            pred_fn  = fitter.predict_wp,
            fmt      = "o",
            xlabel   = r"$r_p$ [$h^{-1}$ Mpc]",
            ylabel   = r"$w_p(r_p)$ [$h^{-1}$ Mpc]",
            ylabel_c = r"$r_p\,w_p$ [$h^{-2}\ {\rm Mpc}^2$]",
            scaled   = True,
        ),
        "esd_hsc": dict(
            get_rp   = lambda f: np.array(f.rp_hsc),
            get_obs  = lambda f: np.array(f.ds_hsc_obs),
            get_err  = lambda f: np.array(f.ds_hsc_err),
            pred_fn  = fitter.predict_ds_hsc,
            fmt      = "s",
            xlabel   = r"$R$ [$h^{-1}$ Mpc]",
            ylabel   = r"$\Delta\Sigma(R)$ [$M_\odot\ {\rm pc}^{-2}$]",
            ylabel_c = r"$\Delta\Sigma$ HSC [$M_\odot\ {\rm pc}^{-2}$]",
            scaled   = False,
        ),
        "esd_des": dict(
            get_rp   = lambda f: np.array(f.rp_des),
            get_obs  = lambda f: np.array(f.ds_des_obs),
            get_err  = lambda f: np.array(f.ds_des_err),
            pred_fn  = fitter.predict_ds_des,
            fmt      = "^",
            xlabel   = r"$R$ [$h^{-1}$ Mpc]",
            ylabel   = r"$\Delta\Sigma(R)$ [$M_\odot\ {\rm pc}^{-2}$]",
            ylabel_c = r"$\Delta\Sigma$ DES [$M_\odot\ {\rm pc}^{-2}$]",
            scaled   = False,
        ),
    }

    # Assemble per-probe arrays
    probe_data = {}
    for probe in probes:
        if probe not in _probe_cfg:
            continue
        cfg_p   = _probe_cfg[probe]
        rp      = cfg_p["get_rp"](fitter)
        obs     = cfg_p["get_obs"](fitter)
        err     = cfg_p["get_err"](fitter)
        pred    = np.array(cfg_p["pred_fn"](params))
        bands   = (_mcmc_bands(cfg_p["pred_fn"], params, flatchain, fc_names)
                   if flatchain is not None else None)
        probe_data[probe] = dict(rp=rp, obs=obs, err=err, pred=pred, bands=bands, **cfg_p)

    # ── 1. Individual probe figures ─────────────────────────────────────────
    for probe_key, d in probe_data.items():
        rp, obs, err, pred, bands = d["rp"], d["obs"], d["err"], d["pred"], d["bands"]
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        _add_bands(axes[0], rp, bands, _COL_MAP)
        axes[0].errorbar(rp, obs, yerr=err, fmt=d["fmt"], ms=4,
                         color=_COL_DATA, zorder=5, label="Data")
        axes[0].loglog(rp, pred, "-", color=_COL_MAP, lw=1.8,
                       label=f"MAP (χ²/dof={chi2_ndof:.2f})")
        if bands is not None:
            axes[0].plot([], [], "--", color=_COL_MAP, lw=1.5,
                         label="MCMC median ± 68/95%")
        axes[0].set_ylabel(d["ylabel"])
        axes[0].legend(fontsize=9)
        axes[0].set_title(f"{title_base} — {probe_key}", fontsize=10)
        _residual_panel(axes[1], rp, obs, pred, err, bands=bands, fmt=d["fmt"])
        axes[1].set_xlabel(d["xlabel"])
        fig.tight_layout()
        out_png = os.path.join(output_dir, f"{probe_key}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Saved: {probe_key}.png")

    # ── 2. Combined figure ──────────────────────────────────────────────────
    if probe_data:
        n_cols  = len(probe_data)
        fig_c, axes_c = plt.subplots(
            2, n_cols, figsize=(6 * n_cols + 0.5, 8),
            sharex="col", squeeze=False,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05,
                         "wspace": 0.3 if n_cols > 1 else 0.0},
        )
        for col, (probe_key, d) in enumerate(probe_data.items()):
            rp, obs, err, pred, bands = d["rp"], d["obs"], d["err"], d["pred"], d["bands"]
            scale = rp if d["scaled"] else None
            yobs  = rp * obs  if scale is not None else obs
            ypred = rp * pred if scale is not None else pred
            yerr  = rp * err  if scale is not None else err
            _add_bands(axes_c[0, col], rp, bands, _COL_MAP, scale=scale)
            axes_c[0, col].errorbar(rp, yobs, yerr=yerr, fmt=d["fmt"], ms=4,
                                    color=_COL_DATA, zorder=5, label="Data")
            axes_c[0, col].loglog(rp, ypred, "-", color=_COL_MAP, lw=1.8,
                                  label=f"MAP (χ²/dof={chi2_ndof:.2f})")
            if bands is not None:
                axes_c[0, col].plot([], [], "--", color=_COL_MAP, lw=1.5,
                                    label="MCMC median ± 68/95%")
            axes_c[0, col].set_ylabel(d["ylabel_c"])
            axes_c[0, col].legend(fontsize=8)
            axes_c[0, col].set_title(f"{title_base} — {probe_key}", fontsize=9)
            _residual_panel(axes_c[1, col], rp, obs, pred, err,
                            bands=bands, fmt=d["fmt"])
            axes_c[1, col].set_xlabel(d["xlabel"])
        fig_c.savefig(os.path.join(output_dir, "combined.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_c)
        print("  Saved: combined.png")

    # ── 3. HOD figure ───────────────────────────────────────────────────────
    try:
        _plot_hod(fitter, params, None, tag, output_dir,
                  flatchain=flatchain, param_names=fc_names)
    except Exception as exc:
        print(f"  HOD plot skipped: {exc}")

    # ── 4. Corner plot (MCMC only) ──────────────────────────────────────────
    if flatchain is not None:
        fixed = {k: float(v) for k, v in params.items() if k not in fc_names}
        try:
            _plot_corner_fn(flatchain, fc_names, {}, tag, output_dir,
                            fixed_params=fixed or None)
        except Exception as exc:
            print(f"  Corner plot skipped: {exc}")


def run_from_config(cfg: dict, args: argparse.Namespace, sum_stat_dir: str) -> None:
    d   = cfg["data"]
    m   = cfg["model"]
    fit = cfg.get("fitting", {})
    out = cfg.get("output", {})

    mstar_lo = float(d["mstar_lo"])
    if mstar_lo not in BGS_BINS:
        raise ValueError(
            f"mstar_lo={mstar_lo} not in BGS_BINS. "
            f"Choose from {sorted(BGS_BINS.keys())}."
        )
    info = BGS_BINS[mstar_lo]

    probes = d.get("probes", ["wp", "esd_hsc"])
    if isinstance(probes, str):
        probes = [p.strip() for p in probes.split(",")]

    # Scale cuts — YAML values take precedence; CLI overrides if supplied
    rp_min_wp  = float(d.get("rp_min_wp",  0.3))
    rp_max_wp  = float(d.get("rp_max_wp",  50.0))
    rp_min_hsc = float(d.get("rp_min_hsc", rp_min_wp))
    rp_min_des = float(d.get("rp_min_des", rp_min_wp))
    rp_max_esd = float(d.get("rp_max_esd", 10.0))
    esd_sn_min = float(d.get("esd_sn_min", 5.0))
    pi_max     = float(d.get("pi_max",     100.0))

    # MCMC settings — CLI overrides YAML when provided
    n_walkers = args.n_walkers if args.n_walkers is not None else int(fit.get("n_walkers", 64))
    n_steps   = args.n_steps   if args.n_steps   is not None else int(fit.get("n_steps",   3000))
    n_burnin  = args.n_burnin  if args.n_burnin  is not None else int(fit.get("n_burnin",  500))

    base_dir   = os.path.join(results_root(), out.get("base_dir", "results/bgs_comparat2025").removeprefix("results/"))
    output_dir = _build_output_dir(cfg, base_dir)

    data_file = _find_data_file(mstar_lo, info, sum_stat_dir)

    fitter = MultiProbeFitter(
        data_file            = data_file,
        z                    = info["z_eff"],
        log10mmin_init       = info["log10mmin_init"],
        mstar_lo             = mstar_lo,
        mstar_hi             = float(d.get("mstar_hi", info["mstar_hi"])),
        probes               = tuple(probes),
        rp_min_wp            = rp_min_wp,
        rp_max_wp            = rp_max_wp,
        rp_min_hsc           = rp_min_hsc,
        rp_min_des           = rp_min_des,
        rp_max_esd           = rp_max_esd,
        esd_sn_min           = esd_sn_min,
        pi_max               = pi_max,
        use_full_cov         = bool(m.get("use_full_cov",        False)),
        use_ia               = bool(m.get("use_ia",              False)),
        use_baryon_fraction  = bool(m.get("use_baryon_fraction", False)),
        use_offcentering     = bool(m.get("use_offcentering",    False)),
        use_free_cosmo       = bool(m.get("use_free_cosmo",      False)),
        use_stellar_mass     = bool(m.get("use_stellar_mass",    False)),
        use_incompleteness   = bool(m.get("use_incompleteness",  False)),
        use_bnl              = bool(m.get("use_bnl",             True)),
        hod_model            = m.get("hod_model", "more2015"),
        profile              = m.get("profile",   "nfw"),
        n_walkers            = n_walkers,
        n_steps              = n_steps,
        n_burnin             = n_burnin,
        output_dir           = output_dir,
    )

    print(f"\nBGS multi-probe HOD fit [{m.get('hod_model', 'more2015')}]")
    print(f"  Config:  {args.config}")
    print(f"  M* > 10^{mstar_lo}  z = {info['z_min']:.2f}–{info['z_max']:.2f} (z_eff={info['z_eff']:.3f})")
    print(f"  Data:    {data_file}")
    print(f"  Probes:  {probes}")
    print(f"  wp scale cut:  rp > {rp_min_wp} Mpc/h   (max {rp_max_wp})")
    print(f"  HSC scale cut: rp > {rp_min_hsc} Mpc/h  (max {rp_max_esd})")
    print(f"  N_wp={fitter.n_wp}  N_hsc={fitter.n_hsc}")
    print(f"  Free params ({len(fitter.free_params)}): {fitter.free_params}")
    print(f"  Output:  {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    plot_only = getattr(args, "plot_only", False)
    if plot_only:
        out_json = os.path.join(output_dir, "map_result.json")
        if not os.path.exists(out_json):
            raise FileNotFoundError(
                f"No map_result.json found in {output_dir}. Run the fit first."
            )
        with open(out_json) as fh:
            saved = json.load(fh)
        print(f"\nPlot-only mode — loading {out_json}")
        _make_plots(fitter, saved["params"],
                    saved["chi2"] / saved["ndof"] if saved["ndof"] > 0 else float("nan"),
                    output_dir, probes, mstar_lo, info["z_eff"])
        return

    method = fit.get("method", "both") if not args.map_only else "map"

    if method in ("map", "both"):
        result = fitter.map_fit()
        print("\n=== MAP result ===")
        for name, val in zip(fitter.free_params, result["theta"]):
            print(f"  {name:30s} = {val:.5f}")
        chi2_data = result.get("chi2_data", None)
        hartlap   = result.get("hartlap_factor", None)
        n_jk      = result.get("n_jk", None)
        print(f"  chi2/dof       = {result['chi2']:.2f} / {result['ndof']}")
        if chi2_data is not None:
            print(f"  chi2_data/dof  = {chi2_data:.2f} / {result['ndof']}  (data only, no priors)")
        if hartlap is not None:
            print(f"  Hartlap factor = {hartlap:.4f}  (N_jk={n_jk}, N_bins={len(fitter.dv_obs)})")
        print(f"  Elapsed: {result['elapsed']:.1f} s  ({result['message']})")

        out_json = os.path.join(output_dir, "map_result.json")
        with open(out_json, "w") as fh:
            json.dump(
                {
                    "params":         result["params"],
                    "chi2":           result["chi2"],
                    "chi2_data":      chi2_data,
                    "ndof":           result["ndof"],
                    "n_jk":           n_jk,
                    "hartlap_factor": hartlap,
                    "elapsed":        result["elapsed"],
                    "success":        result["success"],
                    "message":        result["message"],
                    "config":         args.config,
                    "data_file":      data_file,
                    "mstar_lo":       mstar_lo,
                    "z_eff":          info["z_eff"],
                    "probes":         probes,
                    "rp_min_wp":      rp_min_wp,
                    "rp_min_hsc":     rp_min_hsc,
                    "rp_max_esd":     rp_max_esd,
                    "free_params":    fitter.free_params,
                    "physics": {k: bool(m.get(k, False)) for k in [
                        "use_bnl", "use_ia", "use_offcentering",
                        "use_baryon_fraction", "use_stellar_mass",
                        "use_incompleteness", "use_free_cosmo",
                    ]},
                },
                fh,
                indent=2,
            )
        print(f"MAP result saved → {out_json}")

    if method in ("mcmc", "both") and not args.map_only:
        chain_path = os.path.join(output_dir, "flatchain.npz")
        if method == "both":
            # seed MCMC walkers around the MAP solution
            for name, val in result["params"].items():
                if name in fitter.param_init:
                    fitter.param_init[name] = val
        fitter.sample()
        print(f"MCMC chain saved → {chain_path}")

    if method in ("map", "both"):
        print("\n=== Generating figures ===")
        _make_plots(fitter, result["params"],
                    result["chi2"] / result["ndof"] if result["ndof"] > 0 else float("nan"),
                    output_dir, probes, mstar_lo, info["z_eff"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to the YAML fitting config (e.g. configs/fitting/BGS_LS10_....yml).",
    )
    parser.add_argument("--map-only",  action="store_true",
                        help="Run MAP optimisation only, skip MCMC.")
    parser.add_argument("--sum-stat-dir", default=SUM_STAT_DIR,
                        help="Override default path to sum_stat data directory.")
    parser.add_argument("--n-walkers", type=int, default=None,
                        help="Number of MCMC walkers (overrides YAML).")
    parser.add_argument("--n-steps",   type=int, default=None,
                        help="Number of MCMC steps per walker (overrides YAML).")
    parser.add_argument("--n-burnin",  type=int, default=None,
                        help="Number of MCMC burn-in steps to discard (overrides YAML).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Load saved map_result.json and regenerate figures "
                             "without re-running the fit.")
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_REPO_ROOT, cfg_path)

    cfg = _load_yaml(cfg_path)
    args.config = cfg_path

    run_from_config(cfg, args, args.sum_stat_dir)


if __name__ == "__main__":
    main()
