"""Cross-prediction tension and SHMR comparison for BGS wp-only vs ESD-only MAP fits.

Usage
-----
From the repository root::

    python hod_mod/scripts/fitting/bgs_ls10/plot_tension_shmr.py \\
        results/bgs_comparat2025/<best_wp_dir>/map_result.json \\
        results/bgs_comparat2025/<best_esd_dir>/map_result.json

Outputs (written to results/bgs_comparat2025/tension_test/):
    tension_cross_prediction.png   — 4-panel cross-prediction figure
    tension_hod_params.png         — HOD parameter comparison bar plot
    shmr_girelli_comparison.png    — SHMR vs Girelli+2020 at z_eff
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax.numpy as jnp

from hod_mod.scripts.fitting.bgs_ls10.fit_bgs_multiprobe import (
    MultiProbeFitter,
    BGS_BINS,
    SUM_STAT_DIR,
    _find_data_file,
)
from hod_mod.connection.sham import smhm_girelli20
from hod_mod.paths import results_root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers

def _load_result(json_path: str) -> dict:
    with open(json_path) as fh:
        return json.load(fh)


def _build_fitter(result: dict, probes_override: list[str] | None = None) -> MultiProbeFitter:
    mstar_lo = float(result["mstar_lo"])
    info     = BGS_BINS[mstar_lo]
    probes   = probes_override if probes_override is not None else result["probes"]
    data_file = _find_data_file(mstar_lo, info, SUM_STAT_DIR)
    phys = result.get("physics", {})
    return MultiProbeFitter(
        data_file           = data_file,
        z                   = info["z_eff"],
        log10mmin_init      = info["log10mmin_init"],
        mstar_lo            = mstar_lo,
        mstar_hi            = float(result.get("mstar_hi", info["mstar_hi"])),
        probes              = tuple(probes),
        rp_min_wp           = float(result.get("rp_min_wp",   0.5)),
        rp_max_wp           = 50.0,
        rp_min_hsc          = float(result.get("rp_min_hsc",  1.5)),
        rp_min_des          = float(result.get("rp_min_des",  1.5)),
        rp_max_esd          = float(result.get("rp_max_esd",  10.0)),
        esd_sn_min          = 5.0,
        pi_max              = 100.0,
        use_ia              = bool(phys.get("use_ia", False)),
        use_baryon_fraction = bool(phys.get("use_baryon_fraction", False)),
        use_offcentering    = bool(phys.get("use_offcentering", False)),
        use_stellar_mass    = bool(phys.get("use_stellar_mass", False)),
        use_incompleteness  = bool(phys.get("use_incompleteness", False)),
        hod_model           = result.get("hod_model", "more2015"),
        profile             = result.get("profile", "nfw"),
        output_dir          = "/tmp/tension_fitter",
    )


def _chi2_per_bin(obs, pred, err):
    return ((pred - obs) / err) ** 2


# ---------------------------------------------------------------------------
# Figure 1: cross-prediction tension

def plot_cross_prediction(wp_result, esd_result, output_dir: str) -> None:
    params_wp  = wp_result["params"]
    params_esd = esd_result["params"]
    mstar_lo   = float(wp_result["mstar_lo"])
    info       = BGS_BINS[mstar_lo]
    z_eff      = info["z_eff"]

    # Build fitters with both probes loaded so we can cross-predict
    rp_min_wp  = float(wp_result.get("rp_min_wp",  0.5))
    rp_min_hsc = float(esd_result.get("rp_min_hsc", 1.5))

    fitter_joint = MultiProbeFitter(
        data_file           = _find_data_file(mstar_lo, info, SUM_STAT_DIR),
        z                   = z_eff,
        log10mmin_init      = info["log10mmin_init"],
        mstar_lo            = mstar_lo,
        mstar_hi            = info["mstar_hi"],
        probes              = ("wp", "esd_hsc"),
        rp_min_wp           = rp_min_wp,
        rp_max_wp           = 50.0,
        rp_min_hsc          = rp_min_hsc,
        rp_min_des          = rp_min_hsc,
        rp_max_esd          = 10.0,
        esd_sn_min          = 5.0,
        pi_max              = 100.0,
        use_ia              = bool(esd_result.get("physics", {}).get("use_ia", False)),
        use_baryon_fraction = False,
        use_offcentering    = bool(esd_result.get("physics", {}).get("use_offcentering", False)),
        use_stellar_mass    = bool(esd_result.get("physics", {}).get("use_stellar_mass", False)),
        use_incompleteness  = bool(wp_result.get("physics", {}).get("use_incompleteness", False)),
        hod_model           = wp_result.get("hod_model", "more2015"),
        profile             = "nfw",
        output_dir          = "/tmp/tension_fitter",
    )

    rp_wp  = np.array(fitter_joint.rp_wp)
    rp_hsc = np.array(fitter_joint.rp_hsc)
    rp_wp_full  = np.array(fitter_joint.rp_wp_full)
    rp_hsc_full = np.logspace(-2, 1, 60)

    # Predictions at fit-range radii
    wp_pred_wp   = fitter_joint.predict_wp(params_wp,  rp=rp_wp)
    wp_pred_esd  = fitter_joint.predict_wp(params_esd, rp=rp_wp)
    esd_pred_wp  = fitter_joint.predict_ds_hsc(params_wp,  rp=rp_hsc)
    esd_pred_esd = fitter_joint.predict_ds_hsc(params_esd, rp=rp_hsc)

    # Predictions on fine grid for smooth curves
    wp_curve_wp   = fitter_joint.predict_wp(params_wp,  rp=rp_wp_full)
    wp_curve_esd  = fitter_joint.predict_wp(params_esd, rp=rp_wp_full)
    esd_curve_wp  = fitter_joint.predict_ds_hsc(params_wp,  rp=rp_hsc_full)
    esd_curve_esd = fitter_joint.predict_ds_hsc(params_esd, rp=rp_hsc_full)

    wp_obs  = np.array(fitter_joint.wp_obs)
    wp_err  = np.array(fitter_joint.wp_err)
    esd_obs = np.array(fitter_joint.ds_hsc_obs)
    esd_err = np.array(fitter_joint.ds_hsc_err)

    chi2_wp_from_wp   = float(np.sum(_chi2_per_bin(wp_obs,  wp_pred_wp,   wp_err)))
    chi2_wp_from_esd  = float(np.sum(_chi2_per_bin(wp_obs,  wp_pred_esd,  wp_err)))
    chi2_esd_from_wp  = float(np.sum(_chi2_per_bin(esd_obs, esd_pred_wp,  esd_err)))
    chi2_esd_from_esd = float(np.sum(_chi2_per_bin(esd_obs, esd_pred_esd, esd_err)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        rf"Probe tension: BGS $M_* > 10^{{10}}\,M_\odot$,  $z_{{\rm eff}}={z_eff:.3f}$",
        fontsize=12,
    )

    # ── wp panel ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.errorbar(rp_wp, rp_wp * wp_obs, yerr=rp_wp * wp_err,
                fmt="o", color="k", ms=4, zorder=5, label="Data")
    ax.loglog(rp_wp_full, rp_wp_full * wp_curve_wp,
              "-",  color="C0", lw=1.8,
              label=rf"wp-only MAP  ($\chi^2={chi2_wp_from_wp:.1f}/{len(wp_obs)}$)")
    ax.loglog(rp_wp_full, rp_wp_full * wp_curve_esd,
              "--", color="C1", lw=1.8,
              label=rf"ESD-only MAP ($\chi^2={chi2_wp_from_esd:.1f}/{len(wp_obs)}$)")
    ax.axvline(rp_min_wp, ls=":", color="C0", lw=1, alpha=0.6)
    ax.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$r_p\,w_p$ [$h^{-2}\ {\rm Mpc}^2$]")
    ax.legend(fontsize=8)
    ax.set_title(r"$w_p(r_p)$", fontsize=10)

    # ── ESD panel ─────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.errorbar(rp_hsc, esd_obs, yerr=esd_err,
                fmt="s", color="k", ms=4, zorder=5, label="Data (HSC)")
    ax.loglog(rp_hsc_full, esd_curve_wp,
              "-",  color="C0", lw=1.8,
              label=rf"wp-only MAP  ($\chi^2={chi2_esd_from_wp:.1f}/{len(esd_obs)}$)")
    ax.loglog(rp_hsc_full, esd_curve_esd,
              "--", color="C1", lw=1.8,
              label=rf"ESD-only MAP ($\chi^2={chi2_esd_from_esd:.1f}/{len(esd_obs)}$)")
    ax.axvline(rp_min_hsc, ls=":", color="C1", lw=1, alpha=0.6)
    ax.set_xlabel(r"$R$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\ {\rm pc}^{-2}$]")
    ax.legend(fontsize=8)
    ax.set_title(r"$\Delta\Sigma(R)$ HSC", fontsize=10)

    # ── wp residuals ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.semilogx(rp_wp, (wp_pred_wp  - wp_obs) / wp_err, "o-",  color="C0", ms=4, lw=1.2,
                label="wp-only MAP")
    ax.semilogx(rp_wp, (wp_pred_esd - wp_obs) / wp_err, "s--", color="C1", ms=4, lw=1.2,
                label="ESD-only MAP")
    ax.axhline( 2, color="gray", lw=0.6, ls=":")
    ax.axhline(-2, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$({\rm pred} - {\rm data})/\sigma$")
    ax.legend(fontsize=8)
    ax.set_ylim(-6, 6)

    # ── ESD residuals ─────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.semilogx(rp_hsc, (esd_pred_wp  - esd_obs) / esd_err, "o-",  color="C0", ms=4, lw=1.2,
                label="wp-only MAP")
    ax.semilogx(rp_hsc, (esd_pred_esd - esd_obs) / esd_err, "s--", color="C1", ms=4, lw=1.2,
                label="ESD-only MAP")
    ax.axhline( 2, color="gray", lw=0.6, ls=":")
    ax.axhline(-2, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"$R$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$({\rm pred} - {\rm data})/\sigma$")
    ax.legend(fontsize=8)
    ax.set_ylim(-6, 6)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "tension_cross_prediction.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print tension summary
    print("\n=== Cross-prediction chi2 summary ===")
    print(f"  wp-only  MAP on wp  data: chi2 = {chi2_wp_from_wp:.1f}  / {len(wp_obs)} bins")
    print(f"  ESD-only MAP on wp  data: chi2 = {chi2_wp_from_esd:.1f}  / {len(wp_obs)} bins")
    print(f"  wp-only  MAP on ESD data: chi2 = {chi2_esd_from_wp:.1f} / {len(esd_obs)} bins")
    print(f"  ESD-only MAP on ESD data: chi2 = {chi2_esd_from_esd:.1f} / {len(esd_obs)} bins")


# ---------------------------------------------------------------------------
# Figure 2: HOD parameter comparison

def plot_hod_params(wp_result, esd_result, output_dir: str) -> None:
    params_wp  = wp_result["params"]
    params_esd = esd_result["params"]
    core_params = ["log10mmin", "sigma_logm", "log10m1", "alpha", "kappa"]
    present = [p for p in core_params if p in params_wp and p in params_esd]

    vals_wp  = np.array([params_wp[p]  for p in present])
    vals_esd = np.array([params_esd[p] for p in present])
    delta    = vals_wp - vals_esd

    # Normalise differences by the Gaussian prior sigma where it exists (log10mmin → 0.5)
    prior_sigma = {"log10mmin": 0.5}
    delta_norm = [
        delta[i] / prior_sigma.get(p, 1.0)
        for i, p in enumerate(present)
    ]

    x = np.arange(len(present))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2, vals_wp,  width, label="wp-only MAP",  color="C0", alpha=0.8)
    ax.bar(x + width / 2, vals_esd, width, label="ESD-only MAP", color="C1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Parameter value")
    ax.legend(fontsize=9)
    ax.set_title("HOD parameter values: wp-only vs ESD-only MAP", fontsize=10)

    ax = axes[1]
    colors = ["C2" if abs(d) < 1 else ("C3" if abs(d) < 2 else "C4") for d in delta_norm]
    ax.bar(x, delta_norm, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline( 1, color="gray", lw=0.7, ls="--")
    ax.axhline(-1, color="gray", lw=0.7, ls="--")
    ax.axhline( 2, color="gray", lw=0.5, ls=":")
    ax.axhline(-2, color="gray", lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$(p_{wp} - p_{ESD})$ / $\sigma_{\rm prior}$  [log10mmin]  else  $\Delta p$")
    ax.set_title("Parameter differences (wp-only − ESD-only)", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "tension_hod_params.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    print("\n=== HOD parameter comparison ===")
    print(f"  {'Param':20s} {'wp-only':>12s} {'ESD-only':>12s} {'diff':>10s}")
    for p, vw, ve in zip(present, vals_wp, vals_esd):
        print(f"  {p:20s} {vw:12.4f} {ve:12.4f} {vw-ve:+10.4f}")


# ---------------------------------------------------------------------------
# Figure 3: SHMR vs Girelli+2020

def plot_shmr_girelli(wp_result, esd_result, output_dir: str) -> None:
    z_eff   = float(wp_result["z_eff"])
    h       = wp_result["params"].get("h", 0.6736)
    mstar_lo = float(wp_result.get("mstar_lo", 10.0))

    log10mmin_wp  = float(wp_result["params"]["log10mmin"])
    log10mmin_esd = float(esd_result["params"]["log10mmin"])

    # Girelli+2020 SHMR curve at z_eff
    mhalo_grid = np.linspace(10.0, 15.5, 300)
    mstar_grid = np.array(smhm_girelli20(jnp.array(mhalo_grid), z_eff))

    # M★ threshold in M☉/h: mstar_lo is log10(M★/M☉), convert to log10(M★/(M☉/h))
    mstar_thresh_h = mstar_lo - np.log10(h)   # e.g. 10.0 - log10(0.6736) ≈ 10.17

    # Girelli-predicted M_min: find M_h where smhm_girelli20 = mstar_thresh_h
    idx_cross = np.argmin(np.abs(mstar_grid - mstar_thresh_h))
    m_min_girelli = mhalo_grid[idx_cross]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(mhalo_grid, mstar_grid, "-", color="C2", lw=2.2,
            label=r"Girelli+2020 SHMR at $z=%.3f$" % z_eff)

    ax.axhline(mstar_thresh_h, ls="--", color="k", lw=1.2,
               label=rf"$M_* = 10^{{{mstar_lo}}} M_\odot$ threshold (h-units: {mstar_thresh_h:.2f})")

    ax.axvline(m_min_girelli, ls=":", color="C2", lw=1.5,
               label=rf"Girelli $M_\min$ = {m_min_girelli:.2f}")

    ax.axvline(log10mmin_wp, ls="-", color="C0", lw=2.0, alpha=0.8,
               label=rf"wp-only MAP $\log_{{10}}M_\min$ = {log10mmin_wp:.2f}")

    ax.axvline(log10mmin_esd, ls="-", color="C1", lw=2.0, alpha=0.8,
               label=rf"ESD-only MAP $\log_{{10}}M_\min$ = {log10mmin_esd:.2f}")

    # Annotate offsets
    ax.annotate(
        rf"$\Delta\log_{{10}}M_\min$ (wp vs Girelli) = {log10mmin_wp - m_min_girelli:+.2f} dex",
        xy=(0.05, 0.20), xycoords="axes fraction", fontsize=9, color="C0",
    )
    ax.annotate(
        rf"$\Delta\log_{{10}}M_\min$ (ESD vs Girelli) = {log10mmin_esd - m_min_girelli:+.2f} dex",
        xy=(0.05, 0.13), xycoords="axes fraction", fontsize=9, color="C1",
    )
    ax.annotate(
        rf"$\Delta\log_{{10}}M_\min$ (wp vs ESD) = {log10mmin_wp - log10mmin_esd:+.2f} dex",
        xy=(0.05, 0.06), xycoords="axes fraction", fontsize=9, color="gray",
    )

    ax.set_xlabel(r"$\log_{10}(M_h / [M_\odot\,h^{-1}])$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(M_* / [M_\odot\,h^{-1}])$", fontsize=12)
    ax.set_title(
        rf"SHMR — BGS $M_*>10^{{{mstar_lo}}}\,M_\odot$ vs Girelli+2020, $z_{{eff}}={z_eff:.3f}$",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(10.0, 15.5)
    ax.set_ylim(7.0, 13.5)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "shmr_girelli_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    print("\n=== SHMR comparison ===")
    print(f"  Girelli predicted M_min at M*={mstar_lo}:  {m_min_girelli:.3f}  log10(M_h/[M_sun/h])")
    print(f"  wp-only  MAP log10mmin:                    {log10mmin_wp:.3f}  (offset {log10mmin_wp - m_min_girelli:+.3f} dex)")
    print(f"  ESD-only MAP log10mmin:                    {log10mmin_esd:.3f}  (offset {log10mmin_esd - m_min_girelli:+.3f} dex)")
    print(f"  wp vs ESD tension in log10mmin:            {log10mmin_wp - log10mmin_esd:+.3f} dex")


# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wp_json",  help="map_result.json from best wp-only run")
    parser.add_argument("esd_json", help="map_result.json from best ESD-only run")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/bgs_comparat2025/tension_test)")
    args = parser.parse_args()

    wp_result  = _load_result(args.wp_json)
    esd_result = _load_result(args.esd_json)

    if args.output_dir is None:
        output_dir = os.path.join(results_root(), "bgs_comparat2025", "tension_test")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Tension analysis ===")
    print(f"  wp-only  MAP: {args.wp_json}")
    print(f"  ESD-only MAP: {args.esd_json}")
    print(f"  Output dir:   {output_dir}")

    plot_cross_prediction(wp_result, esd_result, output_dir)
    plot_hod_params(wp_result, esd_result, output_dir)
    plot_shmr_girelli(wp_result, esd_result, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
