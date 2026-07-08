#!/usr/bin/env python
r"""Posterior figures for the finished BGS × LS10 Zu & Mandelbaum (2015) joint
``w_p`` + ``n_gal`` MCMC (no lensing).

Exploits the emcee chain produced by ``oarsub/fit_bgs_zm15_joint_mcmc.sh``
(:mod:`hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint` ``--mode mcmc
--surveys``) with hod_mod 0.2.1.  Reads ``flatchain.npz`` (64 000 × 13) and
``map_result.json`` from the results directory and writes a complete posterior
figure set into ``docs/_images/`` (prefix ``bgs_zm15_joint__``) for the docs
page :doc:`bgs_zm15_joint_mcmc`.

Figures
-------
``corner``              13-parameter posterior corner, MAP overlaid.
``wp_bins``            2×4 grid of ``w_p(r_p)`` per bin: data, MAP, 68/95 % bands
                       + residual strips.
``ngal``               ``n_gal`` per stellar-mass bin: data vs posterior-predictive.
``shmr``               stellar-to-halo mass relation posterior band vs literature.
``hod_occupation``     ``<N_cen>`` / ``<N_sat>`` (Mh) posterior bands, 3 bins.
``satellite_fraction`` ``f_sat(>M*)`` posterior band.
``smf``                model stellar mass function band vs observed (not fitted).
``constraints``        forest plot: this-work posterior vs ZM15 published (Table 2).

The wp/n_gal bands re-run the full halo model per draw (the expensive step); the
derived-quantity bands are analytic (fast).  See ``--n-draws*`` for budgets.

Usage::

    JAX_PLATFORMS=cpu HOD_MOD_SUMSTAT=/home/comparat/software/sum_stat/data \
      python -m hod_mod.scripts.fitting.bgs_ls10.plot_bgs_zm15_joint_posterior \
        --out-dir /home/comparat/data/hod_mod_results/bgs_zm15_joint_wp_ngal

References
----------
Zu & Mandelbaum 2015, MNRAS 454, 1161 (arXiv:1505.02781)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint import (  # noqa: E402
    FREE_NAMES,
    FREE_PARAMS,
    PUBLISHED,
    build_predictor,
    load_bins,
    _discover_smf_file,
    load_observed_smf,
)
from hod_mod.connection.hod import (  # noqa: E402
    _mstar_from_mh_zu15,
    n_cen_thresh_zu15,
    n_sat_thresh_zu15,
)
from hod_mod.connection.sham import smhm_behroozi13, smhm_moster13  # noqa: E402
from hod_mod.scripts.benchmarks.benchmark_plots import (  # noqa: E402
    _COL_DATA,
    _COL_MAP,
    _COL_PUB,
    add_bands,
    residual_panel,
)
from hod_mod.paths import results_root, sum_stat_root  # noqa: E402

_PFX = "bgs_zm15_joint__"
_DPI = 120

# LaTeX labels for the 13 ZM15 SHMR/HOD + satellite parameters (not in
# benchmark_plots._PARAM_LATEX, which targets the More+2015 HOD).
_LATEX: dict[str, str] = {
    "lg_m1h":        r"$\log_{10}M_1$",
    "lg_m0star":     r"$\log_{10}M_{*,0}$",
    "beta":          r"$\beta$",
    "delta":         r"$\delta$",
    "gamma":         r"$\gamma$",
    "sigma_lnmstar": r"$\sigma_{\ln M_*}$",
    "eta":           r"$\eta$",
    "fc":            r"$f_c$",
    "bsat":          r"$B_{\rm sat}$",
    "beta_sat":      r"$\beta_{\rm sat}$",
    "bcut":          r"$B_{\rm cut}$",
    "beta_cut":      r"$\beta_{\rm cut}$",
    "alpha_sat":     r"$\alpha_{\rm sat}$",
}

_COL_MED = "C0"    # posterior median / band
_COL_MAPLINE = "C1"  # MAP best-fit line/markers (distinct from published green)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _docs_images(override: str | None = None) -> str:
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    for _ in range(8):
        if os.path.isdir(os.path.join(root, "docs")):
            break
        root = os.path.dirname(root)
    d = os.path.join(root, "docs", "_images")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, _PFX + name + ".png")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path}")


def _bands(samples) -> dict | None:
    """68/95 % percentile bands from an array of draws (n_draw, nx) or (n_draw,)."""
    a = np.asarray(samples, dtype=float)
    if a.ndim == 1:
        a = a[np.isfinite(a)]
    else:
        a = a[np.all(np.isfinite(a), axis=1)]
    if len(a) < 10:
        return None
    return {
        "lo95": np.percentile(a,  2.5, axis=0),
        "lo68": np.percentile(a, 16.0, axis=0),
        "med":  np.percentile(a, 50.0, axis=0),
        "hi68": np.percentile(a, 84.0, axis=0),
        "hi95": np.percentile(a, 97.5, axis=0),
    }


def _subsample(n_total: int, n_draws: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n_draws, n_total), replace=False)


def _make_fast_occ(predictor, theta_cosmo, z: float):
    """Return ``n_cen_sat(params, thresh, mmax=None) -> (n_cen, n_sat)``.

    Mirrors :func:`fit_bgs_zm15_joint._n_cen_n_sat` but pre-computes ``dndm``
    once (it depends only on *z*), so a whole f_sat / SMF grid across many draws
    is fast.  Native units h^3 Mpc^-3.
    """
    hod = predictor._hod
    m_grid = np.asarray(hod._m_grid, dtype=float)
    log10m_grid = hod._log10m_grid
    dndm = np.asarray(hod._hmf.dndm(hod._m_grid, float(z), theta_cosmo), dtype=float)

    def n_cen_sat(params: dict, thresh: float, mmax: float | None = None):
        p = dict(params)
        p["log10m_star_thresh"] = float(thresh)
        if mmax is not None:
            p["log10m_star_max"] = float(mmax)
        else:
            p.pop("log10m_star_max", None)
        with jax.disable_jit():
            nc, ns = hod.nc_ns(log10m_grid, p)
        nc = np.asarray(nc, dtype=float)
        ns = np.asarray(ns, dtype=float)
        return (float(np.trapezoid(dndm * nc, m_grid)),
                float(np.trapezoid(dndm * ns, m_grid)))

    return n_cen_sat


# ---------------------------------------------------------------------------
# expensive per-draw pass: wp + n_gal for every bin
# ---------------------------------------------------------------------------

def compute_wp_ngal(predictor, theta_cosmo, bins, pi_max_h, flatchain, names,
                    n_draws: int):
    """Posterior-predictive ``w_p`` (per bin) and ``n_gal`` (per bin) draws.

    Returns ``(wp_bands, ng_samples, n_used)`` where ``wp_bands[j]`` is a band
    dict (or ``None``) for bin *j* and ``ng_samples[j]`` is a 1-D array of finite
    ``n_gal`` draws for bin *j*.
    """
    idx = _subsample(len(flatchain), n_draws)
    nb = len(bins)
    wp_draws = [[] for _ in range(nb)]
    ng_draws = [[] for _ in range(nb)]
    t0 = time.time()
    for k, i in enumerate(idx):
        pf = dict(zip(names, [float(v) for v in flatchain[i]]))
        for j, b in enumerate(bins):
            p = dict(pf)
            p["log10m_star_thresh"] = b["thresh"]
            if b.get("max") is not None:
                p["log10m_star_max"] = b["max"]
            try:
                wp = np.asarray(predictor.wp(
                    jnp.asarray(b["rp"]), pi_max_h, b["z"], theta_cosmo, p))
                ng = float(predictor.n_gal(b["z"], theta_cosmo, p))
            except Exception:
                wp = np.full(len(b["rp"]), np.nan)
                ng = np.nan
            wp_draws[j].append(wp)
            ng_draws[j].append(ng)
        if k == 0:
            print(f"    first draw (incl. JAX compile): {time.time() - t0:.0f}s")
        elif (k + 1) % 25 == 0:
            print(f"    wp/n_gal draw {k + 1}/{len(idx)}  "
                  f"({(time.time() - t0) / (k + 1):.1f}s/draw)")
    wp_bands = [_bands(np.asarray(wp_draws[j])) for j in range(nb)]
    ng_samples = [np.asarray(ng_draws[j])[np.isfinite(ng_draws[j])] for j in range(nb)]
    return wp_bands, ng_samples, len(idx)


def map_wp_ngal(predictor, theta_cosmo, bins, pi_max_h, map_params):
    wp, ng = [], []
    for b in bins:
        p = dict(map_params)
        p["log10m_star_thresh"] = b["thresh"]
        if b.get("max") is not None:
            p["log10m_star_max"] = b["max"]
        try:
            wp.append(np.asarray(predictor.wp(
                jnp.asarray(b["rp"]), pi_max_h, b["z"], theta_cosmo, p)))
            ng.append(float(predictor.n_gal(b["z"], theta_cosmo, p)))
        except Exception:
            wp.append(np.full(len(b["rp"]), np.nan))
            ng.append(np.nan)
    return wp, ng


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_corner(flatchain, names, map_params, out_dir):
    try:
        import corner
    except ImportError:
        print("  [skip] corner not installed")
        return
    labels = [_LATEX.get(n, n) for n in names]
    truths = [map_params.get(n) for n in names]
    thin = max(1, len(flatchain) // 20000)
    fig = corner.corner(
        flatchain[::thin], labels=labels, truths=truths,
        truth_color=_COL_MAPLINE, show_titles=True, title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 11},
        color=_COL_MED,
    )
    fig.suptitle(
        "ZM15 iHOD joint $w_p$ + $n_{\\rm gal}$ — BGS LS10 posterior\n"
        "(orange lines = MAP best fit)",
        fontsize=13, y=1.02)
    _save(fig, out_dir, "corner")


def fig_wp_bins(bins, wp_bands, map_wp, map_result, out_dir):
    nb = len(bins)
    ncol = 4
    per_bin = map_result.get("chi2_per_bin", {})
    fig, axes = plt.subplots(
        4, ncol, figsize=(3.2 * ncol, 9.5),
        gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.06, "wspace": 0.28},
        sharex="col")
    for j, b in enumerate(bins):
        half = j // ncol          # 0 -> top pair, 1 -> bottom pair
        col = j % ncol
        ax = axes[half * 2, col]
        axr = axes[half * 2 + 1, col]
        rp = b["rp"]
        wp_obs = b["wp_obs"]
        wp_err = np.sqrt(np.diag(np.linalg.inv(b["icov_wp"])))
        band = wp_bands[j]
        mp = map_wp[j]

        add_bands(ax, rp, band, _COL_MED)
        ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=3.5, color=_COL_DATA, zorder=5)
        ax.plot(rp, mp, "-", color=_COL_MAPLINE, lw=1.8, zorder=6)
        ax.set_xscale("log"); ax.set_yscale("log")
        c2 = per_bin.get(b["label"], {}).get("wp", np.nan)
        ax.set_title(rf"$M_*\in[{b['thresh']:g},{b['max']:g}]$"
                     + (f"\n$\\chi^2_{{w_p}}={c2:.1f}$" if np.isfinite(c2) else ""),
                     fontsize=8)
        if col == 0:
            ax.set_ylabel(r"$w_p$ [Mpc/$h$]")
        residual_panel(axr, rp, wp_obs, mp, wp_err, bands=band,
                       color=_COL_MAPLINE, ylabel=(col == 0))
        axr.set_xscale("log")
        if half == 1:
            axr.set_xlabel(r"$r_p$ [Mpc/$h$]", fontsize=9)

    # legend proxies
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=_COL_DATA, marker="o", ls="none", label="data"),
        Line2D([], [], color=_COL_MAPLINE, lw=1.8, label="MAP best fit"),
        Line2D([], [], color=_COL_MED, ls="--", lw=1.5, label="posterior median"),
        Patch(facecolor=_COL_MED, alpha=0.25, label="68 % band"),
        Patch(facecolor=_COL_MED, alpha=0.12, label="95 % band"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.005), frameon=False)
    chi2 = map_result.get("chi2", np.nan); ndof = map_result.get("ndof", 1)
    fig.suptitle(rf"Projected clustering $w_p(r_p)$ — posterior-predictive"
                 rf"   ($\chi^2/\mathrm{{dof}}={chi2:.0f}/{ndof}={chi2/max(ndof,1):.2f}$)",
                 fontsize=13, y=1.03)
    _save(fig, out_dir, "wp_bins")


def fig_ngal(bins, ng_samples, map_ng, out_dir):
    xc = np.array([0.5 * (b["thresh"] + b["max"]) for b in bins])
    n_obs = np.array([b["n_obs"] for b in bins])
    n_err = np.array([b["n_frac"] * b["n_obs"] for b in bins])
    med = np.array([np.percentile(s, 50) if len(s) else np.nan for s in ng_samples])
    lo68 = np.array([np.percentile(s, 16) if len(s) else np.nan for s in ng_samples])
    hi68 = np.array([np.percentile(s, 84) if len(s) else np.nan for s in ng_samples])
    lo95 = np.array([np.percentile(s, 2.5) if len(s) else np.nan for s in ng_samples])
    hi95 = np.array([np.percentile(s, 97.5) if len(s) else np.nan for s in ng_samples])

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.fill_between(xc, lo95, hi95, color=_COL_MED, alpha=0.12, label="95 % band")
    ax.fill_between(xc, lo68, hi68, color=_COL_MED, alpha=0.25, label="68 % band")
    ax.plot(xc, med, "--", color=_COL_MED, lw=1.6, label="posterior median")
    ax.plot(xc, map_ng, "s", color=_COL_MAPLINE, ms=6, label="MAP")
    ax.errorbar(xc, n_obs, yerr=n_err, fmt="o", ms=5, color=_COL_DATA,
                capsize=3, label="observed (fitted)")
    ax.set_yscale("log")
    ax.set_xlabel(r"stellar-mass bin centre $\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel(r"$n_{\rm gal}$ [$h^3\,{\rm Mpc}^{-3}$]")
    ax.set_title("Galaxy number density per stellar-mass bin")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    _save(fig, out_dir, "ngal")


def fig_shmr(flatchain, names, map_params, z_ref, out_dir, n_draws):
    log10mh = jnp.linspace(10.5, 15.0, 120)
    mh = np.asarray(log10mh)
    idx = _subsample(len(flatchain), n_draws, seed=1)
    keys = ("lg_m1h", "lg_m0star", "beta", "delta", "gamma")
    draws = []
    for i in idx:
        pf = dict(zip(names, flatchain[i]))
        draws.append(np.asarray(_mstar_from_mh_zu15(
            log10mh, **{k: float(pf[k]) for k in keys})))
    band = _bands(np.asarray(draws))

    fig, ax = plt.subplots(figsize=(7, 5.2))
    if band is not None:
        ax.fill_between(mh, band["lo95"], band["hi95"], color=_COL_MED, alpha=0.12)
        ax.fill_between(mh, band["lo68"], band["hi68"], color=_COL_MED, alpha=0.28)
        ax.plot(mh, band["med"], "--", color=_COL_MED, lw=1.8,
                label="this work — posterior median (68/95 %)")
    ax.plot(mh, np.asarray(_mstar_from_mh_zu15(
        log10mh, **{k: float(map_params[k]) for k in keys})),
        "-", color=_COL_MAPLINE, lw=2, label="this work — MAP")
    ax.plot(mh, np.asarray(_mstar_from_mh_zu15(
        log10mh, **{k: PUBLISHED[k][0] for k in keys})),
        ":", color=_COL_PUB, lw=1.8, label="ZM15 published (Table 2)")
    ax.plot(mh, np.asarray(smhm_moster13(log10mh, z_ref)), "-.", color="C4", lw=1.3,
            label=fr"Moster+13 ($z={z_ref:.2f}$)")
    ax.plot(mh, np.asarray(smhm_behroozi13(log10mh, z_ref)), "-.", color="C5", lw=1.3,
            label=fr"Behroozi+13 ($z={z_ref:.2f}$)")
    ax.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot\,h^{-1}])$")
    ax.set_ylabel(r"$\log_{10}(M_*\,/\,M_\odot)$")
    ax.set_xlim(10.5, 15.0); ax.set_ylim(8.0, 12.0)
    ax.set_title("Stellar-to-halo mass relation — posterior band vs literature")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, ls=":", alpha=0.3)
    _save(fig, out_dir, "shmr")


def fig_hod_occupation(bins, flatchain, names, map_params, out_dir, n_draws):
    log10mh = jnp.linspace(10.5, 15.5, 150)
    mh = np.asarray(log10mh)
    sel = [0, len(bins) // 2, len(bins) - 1]
    colors = ["C0", "C1", "C3"]
    idx = _subsample(len(flatchain), n_draws, seed=2)
    cen_keys = ("lg_m1h", "lg_m0star", "beta", "delta", "gamma",
                "sigma_lnmstar", "eta", "fc")
    sat_keys = cen_keys + ("bsat", "beta_sat", "bcut", "beta_cut", "alpha_sat")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for b, col in zip((bins[s] for s in sel), colors):
        thr = b["thresh"]
        nc_d, ns_d = [], []
        for i in idx:
            pf = dict(zip(names, flatchain[i]))
            nc_d.append(np.asarray(n_cen_thresh_zu15(
                log10mh, log10m_star_thresh=thr,
                **{k: float(pf[k]) for k in cen_keys})))
            ns_d.append(np.asarray(n_sat_thresh_zu15(
                log10mh, log10m_star_thresh=thr,
                **{k: float(pf[k]) for k in sat_keys})))
        bc = _bands(np.clip(np.asarray(nc_d), 1e-4, None))
        bs = _bands(np.clip(np.asarray(ns_d), 1e-4, None))
        if bc is not None:
            ax.fill_between(mh, bc["lo68"], bc["hi68"], color=col, alpha=0.2, lw=0)
            ax.plot(mh, bc["med"], "-", color=col, lw=1.6,
                    label=fr"$M_*>{thr:g}$")
        if bs is not None:
            ax.fill_between(mh, bs["lo68"], bs["hi68"], color=col, alpha=0.12, lw=0)
            ax.plot(mh, bs["med"], "--", color=col, lw=1.6)
    ax.plot([], [], "k-", lw=1.6, label=r"$\langle N_{\rm cen}\rangle$ (solid)")
    ax.plot([], [], "k--", lw=1.6, label=r"$\langle N_{\rm sat}\rangle$ (dashed)")
    ax.set_yscale("log")
    ax.set_xlim(10.5, 15.5); ax.set_ylim(1e-3, 40)
    ax.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot\,h^{-1}])$")
    ax.set_ylabel(r"$\langle N\rangle(M_h)$")
    ax.set_title("HOD occupation — posterior 68 % bands (3 representative bins)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    _save(fig, out_dir, "hod_occupation")


def fig_satellite_fraction(n_cen_sat, flatchain, names, map_params, out_dir, n_draws):
    grid = np.linspace(9.75, 11.6, 20)
    idx = _subsample(len(flatchain), n_draws, seed=3)
    draws = []
    for i in idx:
        pf = dict(zip(names, flatchain[i]))
        row = np.full(grid.shape, np.nan)
        for g, thr in enumerate(grid):
            nc, ns = n_cen_sat(pf, thr)
            tot = nc + ns
            row[g] = ns / tot if tot > 0 else np.nan
        draws.append(row)
    band = _bands(np.asarray(draws))
    fsat_map = np.full(grid.shape, np.nan)
    for g, thr in enumerate(grid):
        nc, ns = n_cen_sat(map_params, thr)
        tot = nc + ns
        fsat_map[g] = ns / tot if tot > 0 else np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    if band is not None:
        ax.fill_between(grid, band["lo95"], band["hi95"], color=_COL_MED, alpha=0.12)
        ax.fill_between(grid, band["lo68"], band["hi68"], color=_COL_MED, alpha=0.28)
        ax.plot(grid, band["med"], "--", color=_COL_MED, lw=1.8,
                label="posterior median (68/95 %)")
    ax.plot(grid, fsat_map, "-", color=_COL_MAPLINE, lw=2, label="MAP")
    ax.set_xlabel(r"$\log_{10}(M_*\,/\,M_\odot)$ threshold")
    ax.set_ylabel(r"satellite fraction $f_{\rm sat}(>M_*)$")
    top = np.nanmax(band["hi95"]) if band is not None else np.nanmax(fsat_map)
    ax.set_ylim(0, max(0.05, (top or 0.3) * 1.15))
    ax.set_title("Satellite fraction — posterior band")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    _save(fig, out_dir, "satellite_fraction")


def fig_smf(n_cen_sat, flatchain, names, map_params, obs_smf, z_ref, out_dir, n_draws):
    grid = np.linspace(9.6, 11.8, 25)

    def smf_of(pf):
        n_cum = np.array([sum(n_cen_sat(pf, m)) for m in grid])
        phi = -np.gradient(n_cum, grid)
        phi[~(phi > 0)] = np.nan
        return phi

    idx = _subsample(len(flatchain), n_draws, seed=4)
    draws = [smf_of(dict(zip(names, flatchain[i]))) for i in idx]
    band = _bands(np.asarray(draws))
    phi_map = smf_of(map_params)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if band is not None:
        ax.fill_between(grid, band["lo95"], band["hi95"], color=_COL_MED, alpha=0.12)
        ax.fill_between(grid, band["lo68"], band["hi68"], color=_COL_MED, alpha=0.28)
        ax.plot(grid, band["med"], "--", color=_COL_MED, lw=1.8,
                label="posterior median (68/95 %)")
    ax.plot(grid, phi_map, "-", color=_COL_MAPLINE, lw=2, label="MAP")
    if obs_smf is not None:
        ax.errorbar(obs_smf["log10mstar"], obs_smf["phi"], yerr=obs_smf.get("phi_err"),
                    fmt="o", ms=4, color=_COL_DATA,
                    label=fr"observed (not fitted, $z={obs_smf['z']:.2f}$)")
    ax.set_yscale("log")
    ax.set_xlim(9.6, 11.8)
    ax.set_xlabel(r"$\log_{10}(M_*\,/\,M_\odot)$")
    ax.set_ylabel(r"$\Phi$ [$h^3\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}$]")
    ax.set_title(fr"Stellar mass function — posterior band (model $z={z_ref:.2f}$)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    _save(fig, out_dir, "smf")


def fig_constraints(summary, names, out_dir):
    """Forest plot: this-work posterior vs ZM15 published, normalised to prior."""
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ys = np.arange(len(names))[::-1]   # first param on top
    first = True
    for n, y in zip(names, ys):
        lo, hi = FREE_PARAMS[n][0]
        span = hi - lo
        s = summary[n]
        tw_c = (s["med"] - lo) / span
        tw_e = np.array([[(s["med"] - s["p16"]) / span], [(s["p84"] - s["med"]) / span]])
        pub_m, pub_s = PUBLISHED[n]
        pub_c = (pub_m - lo) / span
        pub_e = pub_s / span
        ax.axhspan(y - 0.4, y + 0.4, color="0.93", zorder=0)
        ax.errorbar(tw_c, y, xerr=tw_e, fmt="o", color=_COL_MED, ms=6, capsize=3,
                    zorder=4, label="this work (median ±68 %)" if first else None)
        ax.errorbar(pub_c, y + 0.16, xerr=pub_e, fmt="s", color=_COL_PUB, ms=5,
                    capsize=3, zorder=3,
                    label="ZM15 published (Table 2)" if first else None)
        ax.text(1.03, y, fr"${s['med']:.3f}^{{+{s['p84']-s['med']:.3f}}}_{{-{s['med']-s['p16']:.3f}}}$",
                va="center", ha="left", fontsize=8, transform=ax.get_yaxis_transform())
        first = False
    ax.set_yticks(ys)
    ax.set_yticklabels([_LATEX.get(n, n) for n in names], fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("position within uniform prior range  (0 = lower bound, 1 = upper bound)")
    ax.set_title("Posterior constraints vs ZM15 published")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.axvline(1, color="0.6", lw=0.8, ls=":")
    ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2,
              frameon=False)
    fig.subplots_adjust(right=0.80)
    _save(fig, out_dir, "constraints")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(
        results_root(), "bgs_zm15_joint_wp_ngal"),
        help="Results dir holding flatchain.npz + map_result.json")
    ap.add_argument("--data-dir", default=str(sum_stat_root() / "BGS_Mstar10_massbins"),
                    help="Directory of per-bin sum_stat joint HDF5 files")
    ap.add_argument("--images-dir", default=None,
                    help="Where to write PNGs (default: docs/_images)")
    ap.add_argument("--rp-min", type=float, default=0.5, help="wp r_p min [Mpc/h] (fit value)")
    ap.add_argument("--rp-max", type=float, default=20.0, help="wp r_p max [Mpc/h] (fit value)")
    ap.add_argument("--pi-max-mpc", type=float, default=100.0)
    ap.add_argument("--hmf-backend", default="tinker08")
    ap.add_argument("--n-draws", type=int, default=200,
                    help="Draws for the expensive wp/n_gal bands")
    ap.add_argument("--n-draws-analytic", type=int, default=400,
                    help="Draws for the analytic SHMR / HOD bands")
    ap.add_argument("--n-draws-derived", type=int, default=150,
                    help="Draws for the f_sat / SMF bands (disable_jit occupation)")
    ap.add_argument("--figures", default="all",
                    help="Comma list to restrict which figures are made "
                         "(corner,wp,ngal,shmr,hod,fsat,smf,constraints) or 'all'")
    args = ap.parse_args()

    want = (set(f.strip() for f in args.figures.split(",")) if args.figures != "all"
            else {"corner", "wp", "ngal", "shmr", "hod", "fsat", "smf", "constraints"})

    out_dir = args.out_dir
    img_dir = _docs_images(args.images_dir)
    t0 = time.time()

    # -- load chain + MAP ---------------------------------------------------
    chain_path = os.path.join(out_dir, "flatchain.npz")
    d = np.load(chain_path, allow_pickle=True)
    flatchain = np.asarray(d["flatchain"], dtype=float)
    names = [str(n) for n in d["param_names"]]
    assert names == FREE_NAMES, f"param order mismatch: {names} vs {FREE_NAMES}"
    with open(os.path.join(out_dir, "map_result.json")) as fh:
        map_result = json.load(fh)
    map_params = map_result["params"]
    print(f"Chain: {flatchain.shape[0]} samples × {flatchain.shape[1]} params  ({chain_path})")

    # -- posterior summary table -------------------------------------------
    p16, p50, p84 = np.percentile(flatchain, [16, 50, 84], axis=0)
    summary = {n: {"med": float(p50[i]), "p16": float(p16[i]), "p84": float(p84[i]),
                   "map": float(map_params[n]),
                   "pub_mean": PUBLISHED[n][0], "pub_sig": PUBLISHED[n][1]}
               for i, n in enumerate(names)}
    with open(os.path.join(out_dir, "posterior_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n{'param':14s} {'median':>10s} {'-68%':>8s} {'+68%':>8s} "
          f"{'MAP':>10s} {'ZM15 pub':>10s}")
    for n in names:
        s = summary[n]
        print(f"{n:14s} {s['med']:10.4f} {s['med']-s['p16']:8.4f} {s['p84']-s['med']:8.4f} "
              f"{s['map']:10.4f} {s['pub_mean']:10.4f}")

    # -- corner + constraints (cheap, do first so failures surface early) ---
    if "corner" in want:
        print("\n[corner]")
        fig_corner(flatchain, names, map_params, img_dir)
    if "constraints" in want:
        print("[constraints]")
        fig_constraints(summary, names, img_dir)

    # -- build predictor + data --------------------------------------------
    need_model = want & {"wp", "ngal", "shmr", "hod", "fsat", "smf"}
    if need_model:
        print("\nLoading data + predictor ...")
        bins, h = load_bins(args.data_dir, surveys=[], rp_min=args.rp_min,
                            rp_max=args.rp_max, R_min=0.1, R_max=30.0,
                            ng_frac_err_floor=0.05, log=lambda *a, **k: None)
        for b in bins:
            if b.get("z") is None:
                b["z"] = 0.13
        predictor, theta_cosmo = build_predictor(args.hmf_backend)
        pi_max_h = args.pi_max_mpc * h
        z_ref = float(np.median([b["z"] for b in bins]))
        print(f"  {len(bins)} bins, h={h:.4f}, z_ref={z_ref:.3f}")

        # expensive pass: wp + n_gal
        wp_bands = ng_samples = map_wp = map_ng = None
        if want & {"wp", "ngal"}:
            print(f"\n[wp/n_gal]  {args.n_draws} posterior draws "
                  f"(~{args.n_draws * 7 / 60:.0f} min after compile) ...")
            wp_bands, ng_samples, n_used = compute_wp_ngal(
                predictor, theta_cosmo, bins, pi_max_h, flatchain, names, args.n_draws)
            map_wp, map_ng = map_wp_ngal(predictor, theta_cosmo, bins, pi_max_h, map_params)
            print(f"    used {n_used} draws")
        if "wp" in want:
            fig_wp_bins(bins, wp_bands, map_wp, map_result, img_dir)
        if "ngal" in want:
            fig_ngal(bins, ng_samples, map_ng, img_dir)

        if "shmr" in want:
            print("[shmr]")
            fig_shmr(flatchain, names, map_params, z_ref, img_dir, args.n_draws_analytic)
        if "hod" in want:
            print("[hod_occupation]")
            fig_hod_occupation(bins, flatchain, names, map_params, img_dir,
                               min(args.n_draws_analytic, 300))

        if want & {"fsat", "smf"}:
            n_cen_sat = _make_fast_occ(predictor, theta_cosmo, z_ref)
            if "fsat" in want:
                print("[satellite_fraction]")
                fig_satellite_fraction(n_cen_sat, flatchain, names, map_params,
                                       img_dir, args.n_draws_derived)
            if "smf" in want:
                print("[smf]")
                obs_smf = None
                try:
                    smf_file = _discover_smf_file(str(sum_stat_root()))
                    if smf_file:
                        obs_smf = load_observed_smf(smf_file, z_fallback=z_ref)
                except Exception as exc:
                    print(f"  [warn] observed SMF unavailable: {exc}")
                fig_smf(n_cen_sat, flatchain, names, map_params, obs_smf, z_ref,
                        img_dir, args.n_draws_derived)

    print(f"\nAll figures -> {img_dir}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
