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

    Returns ``(wp_bands, ng_samples, n_used, chi2_samples)`` where ``wp_bands[j]``
    is a band dict (or ``None``) for bin *j*, ``ng_samples[j]`` is a 1-D array of
    finite ``n_gal`` draws for bin *j*, and ``chi2_samples`` is the per-draw total
    :math:`\chi^2` (:math:`w_p` + :math:`n_{\rm gal}`, summed over bins) — computed
    here for free so the posterior :math:`\chi^2` distribution needs no second pass.
    """
    idx = _subsample(len(flatchain), n_draws)
    nb = len(bins)
    wp_draws = [[] for _ in range(nb)]
    ng_draws = [[] for _ in range(nb)]
    chi2_draws = []
    t0 = time.time()
    for k, i in enumerate(idx):
        pf = dict(zip(names, [float(v) for v in flatchain[i]]))
        c2 = 0.0
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
            if np.all(np.isfinite(wp)) and np.isfinite(ng):
                r = wp - b["wp_obs"]
                c2 += float(r @ b["icov_wp"] @ r)
                c2 += ((ng - b["n_obs"]) / (b["n_frac"] * b["n_obs"])) ** 2
            else:
                c2 = np.nan
        chi2_draws.append(c2)
        if k == 0:
            print(f"    first draw (incl. JAX compile): {time.time() - t0:.0f}s")
        elif (k + 1) % 25 == 0:
            print(f"    wp/n_gal draw {k + 1}/{len(idx)}  "
                  f"({(time.time() - t0) / (k + 1):.1f}s/draw)")
    wp_bands = [_bands(np.asarray(wp_draws[j])) for j in range(nb)]
    ng_samples = [np.asarray(ng_draws[j])[np.isfinite(ng_draws[j])] for j in range(nb)]
    return wp_bands, ng_samples, len(idx), np.asarray(chi2_draws, dtype=float)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_corner(flatchain, names, out_dir):
    try:
        import corner
    except ImportError:
        print("  [skip] corner not installed")
        return
    labels = [_LATEX.get(n, n) for n in names]
    thin = max(1, len(flatchain) // 20000)
    fig = corner.corner(
        flatchain[::thin], labels=labels,
        show_titles=True, title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 11},
        color=_COL_MED,
    )
    fig.suptitle(
        "ZM15 iHOD joint $w_p$ + $n_{\\rm gal}$ — BGS LS10 posterior",
        fontsize=13, y=1.02)
    _save(fig, out_dir, "corner")


def fig_wp_bins(bins, wp_bands, out_dir):
    ncol = 4
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

        add_bands(ax, rp, band, _COL_MED)
        ax.errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=3.5, color=_COL_DATA, zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(rf"$M_*\in[{b['thresh']:g},{b['max']:g}]$", fontsize=9)
        if col == 0:
            ax.set_ylabel(r"$w_p$ [Mpc/$h$]")
        ref = band["med"] if band is not None else wp_obs
        residual_panel(axr, rp, wp_obs, ref, wp_err, bands=band,
                       color=_COL_MED, ylabel=(col == 0))
        axr.set_xscale("log")
        if half == 1:
            axr.set_xlabel(r"$r_p$ [Mpc/$h$]", fontsize=9)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=_COL_DATA, marker="o", ls="none", label="data"),
        Line2D([], [], color=_COL_MED, ls="--", lw=1.5, label="posterior median"),
        Patch(facecolor=_COL_MED, alpha=0.25, label="68 % band"),
        Patch(facecolor=_COL_MED, alpha=0.12, label="95 % band"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.005), frameon=False)
    fig.suptitle(r"Projected clustering $w_p(r_p)$ — posterior-predictive "
                 r"(68/95 % credible bands); residual vs posterior median",
                 fontsize=13, y=1.03)
    _save(fig, out_dir, "wp_bins")


def fig_ngal(bins, ng_samples, out_dir):
    """Point-by-point observed vs posterior-predictive n_gal per bin (no lines)."""
    xc = np.array([0.5 * (b["thresh"] + b["max"]) for b in bins])
    n_obs = np.array([b["n_obs"] for b in bins])
    n_err = np.array([b["n_frac"] * b["n_obs"] for b in bins])
    med = np.array([np.percentile(s, 50) if len(s) else np.nan for s in ng_samples])
    lo68 = np.array([np.percentile(s, 16) if len(s) else np.nan for s in ng_samples])
    hi68 = np.array([np.percentile(s, 84) if len(s) else np.nan for s in ng_samples])
    lo95 = np.array([np.percentile(s, 2.5) if len(s) else np.nan for s in ng_samples])
    hi95 = np.array([np.percentile(s, 97.5) if len(s) else np.nan for s in ng_samples])

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    dx = 0.025   # small x-offset so the two points per bin don't overlap

    # top: absolute n_gal, one observed + one posterior point per bin, no lines
    ax.errorbar(xc - dx, n_obs, yerr=n_err, fmt="o", ms=6, color=_COL_DATA,
                capsize=3, ls="none", label="observed (fitted)", zorder=4)
    ax.errorbar(xc + dx, med, yerr=[med - lo95, hi95 - med], fmt="none",
                ecolor=_COL_MED, elinewidth=1.0, alpha=0.5, capsize=0, zorder=2)
    ax.errorbar(xc + dx, med, yerr=[med - lo68, hi68 - med], fmt="s", ms=6,
                color=_COL_MED, capsize=3, ls="none",
                label="posterior-predictive (68 / 95 %)", zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel(r"$n_{\rm gal}$ [$h^3\,{\rm Mpc}^{-3}$]")
    ax.set_title("Galaxy number density per stellar-mass bin — point-by-point")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    # bottom: fractional difference (model/obs - 1), point by point
    ratio = med / n_obs - 1.0
    rlo = lo68 / n_obs - 1.0
    rhi = hi68 / n_obs - 1.0
    axr.axhline(0, color=_COL_DATA, lw=0.8, ls="--")
    for frac in (0.05, -0.05):
        axr.axhline(frac, color="gray", lw=0.5, ls=":")
    axr.errorbar(xc, ratio, yerr=[ratio - rlo, rhi - ratio], fmt="s", ms=5,
                 color=_COL_MED, capsize=3, ls="none")
    axr.set_ylim(-0.12, 0.12)
    axr.set_ylabel(r"model/obs $-1$", fontsize=9)
    axr.set_xlabel(r"stellar-mass bin centre $\log_{10}(M_*/M_\odot)$")
    axr.grid(True, ls=":", alpha=0.3)
    _save(fig, out_dir, "ngal")


def fig_shmr(flatchain, names, z_ref, out_dir, n_draws):
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
        ax.plot(mh, band["med"], "-", color=_COL_MED, lw=2,
                label="this work — posterior median (68/95 %)")
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


def fig_hod_occupation(bins, flatchain, names, out_dir, n_draws):
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

    fig, ax = plt.subplots(figsize=(7, 5))
    if band is not None:
        ax.fill_between(grid, band["lo95"], band["hi95"], color=_COL_MED, alpha=0.12)
        ax.fill_between(grid, band["lo68"], band["hi68"], color=_COL_MED, alpha=0.28)
        ax.plot(grid, band["med"], "-", color=_COL_MED, lw=2,
                label="posterior median (68/95 %)")
    ax.set_xlabel(r"$\log_{10}(M_*\,/\,M_\odot)$ threshold")
    ax.set_ylabel(r"satellite fraction $f_{\rm sat}(>M_*)$")
    top = np.nanmax(band["hi95"]) if band is not None else 0.3
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

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if band is not None:
        ax.fill_between(grid, band["lo95"], band["hi95"], color=_COL_MED, alpha=0.12)
        ax.fill_between(grid, band["lo68"], band["hi68"], color=_COL_MED, alpha=0.28)
        ax.plot(grid, band["med"], "-", color=_COL_MED, lw=2,
                label="posterior median (68/95 %)")
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


def fig_scatter(flatchain, names, out_dir, n_draws):
    """Log-normal scatter sigma_lnM*(Mh) posterior band (ZM15 Eq. 20)."""
    from hod_mod.connection.hod import sigma_lnmstar_zu15
    log10mh = jnp.linspace(11.0, 15.0, 120)
    mh = np.asarray(log10mh)
    idx = _subsample(len(flatchain), n_draws, seed=10)
    keys = ("lg_m1h", "sigma_lnmstar", "eta")
    draws = [np.asarray(sigma_lnmstar_zu15(log10mh, **{k: float(dict(zip(names, flatchain[i]))[k])
                                                       for k in keys})) for i in idx]
    band = _bands(np.asarray(draws))
    fig, ax = plt.subplots(figsize=(7, 5))
    if band is not None:
        ax.fill_between(mh, band["lo95"], band["hi95"], color=_COL_MED, alpha=0.12)
        ax.fill_between(mh, band["lo68"], band["hi68"], color=_COL_MED, alpha=0.28)
        ax.plot(mh, band["med"], "-", color=_COL_MED, lw=2,
                label="this work — posterior median (68/95 %)")
    ax.plot(mh, np.asarray(sigma_lnmstar_zu15(
        log10mh, lg_m1h=PUBLISHED["lg_m1h"][0], sigma_lnmstar=PUBLISHED["sigma_lnmstar"][0],
        eta=PUBLISHED["eta"][0])), ":", color=_COL_PUB, lw=1.8,
        label="ZM15 published (Table 2)")
    ax.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot\,h^{-1}])$")
    ax.set_ylabel(r"scatter $\sigma_{\ln M_*}(M_h)$")
    ax.set_title("Log-normal scatter in stellar mass at fixed halo mass")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    _save(fig, out_dir, "scatter")


def fig_mhalo_mstar(predictor, theta_cosmo, bins, flatchain, names, out_dir, n_draws):
    """Inverse SHMR <log Mh | M*> with posterior band + per-bin mean halo mass.

    The band is the inverted central SHMR (halo mass hosting a central of stellar
    mass M*); the points are the posterior mean halo mass <M_h> of *all* galaxies
    (centrals + satellites) in each fitted bin, which lies above the central curve
    because of the satellite contribution.  (ZM15 Fig. 11 analogue.)
    """
    mstar_grid = np.linspace(9.8, 11.6, 60)
    log10mh = jnp.linspace(10.5, 15.0, 300)
    mh_np = np.asarray(log10mh)
    keys = ("lg_m1h", "lg_m0star", "beta", "delta", "gamma")

    # inverse-SHMR band (analytic)
    idx = _subsample(len(flatchain), n_draws, seed=11)
    inv = []
    for i in idx:
        pf = dict(zip(names, flatchain[i]))
        mstar_of_mh = np.asarray(_mstar_from_mh_zu15(log10mh, **{k: float(pf[k]) for k in keys}))
        inv.append(np.interp(mstar_grid, mstar_of_mh, mh_np,
                             left=np.nan, right=np.nan))
    band = _bands(np.asarray(inv))

    # per-bin mean halo mass <M_h> (centrals+satellites), cheap _integrate
    hod = predictor._hod
    m_grid = np.asarray(hod._m_grid, dtype=float)
    log10m_grid = hod._log10m_grid
    idx2 = _subsample(len(flatchain), min(n_draws, 150), seed=12)
    xb, mb, mlo, mhi = [], [], [], []
    for b in bins:
        dn = np.asarray(hod._hmf.dndm(hod._m_grid, float(b["z"]), theta_cosmo))
        meff = []
        for i in idx2:
            p = dict(zip(names, flatchain[i]))
            p["log10m_star_thresh"] = b["thresh"]
            if b.get("max") is not None:
                p["log10m_star_max"] = b["max"]
            with jax.disable_jit():
                nc, ns = hod.nc_ns(log10m_grid, p)
            nt = np.asarray(nc, float) + np.asarray(ns, float)
            n = np.trapezoid(dn * nt, m_grid)
            if n > 0:
                # occupation-weighted mean of log10 M_h (bias-consistent, unlike
                # the satellite-dominated linear mean)
                meff.append(np.trapezoid(dn * nt * np.log10(m_grid), m_grid) / n)
        if meff:
            q16, q50, q84 = np.percentile(meff, [16, 50, 84])
            xb.append(0.5 * (b["thresh"] + b["max"])); mb.append(q50)
            mlo.append(q50 - q16); mhi.append(q84 - q50)
            print(f"    [mhalo] {b['label']:>10s}  <log10 Mh> = "
                  f"{q50:.2f} +{q84 - q50:.2f} -{q50 - q16:.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if band is not None:
        ax.fill_between(mstar_grid, band["lo95"], band["hi95"], color=_COL_MED, alpha=0.12)
        ax.fill_between(mstar_grid, band["lo68"], band["hi68"], color=_COL_MED, alpha=0.28)
        ax.plot(mstar_grid, band["med"], "-", color=_COL_MED, lw=2,
                label=r"central $\log_{10}M_h(M_*)$ — inverse SHMR (68/95 %)")
    ax.errorbar(xb, mb, yerr=[mlo, mhi], fmt="s", ms=6, color=_COL_PUB, capsize=3,
                ls="none", label=r"per-bin $\langle\log_{10}M_h\rangle$ (cen+sat)")
    ax.set_xlabel(r"$\log_{10}(M_*\,/\,M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(M_h\,/\,[M_\odot\,h^{-1}])$")
    ax.set_title("Halo mass vs stellar mass — posterior")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.3)
    _save(fig, out_dir, "mhalo_mstar")


def fig_csmf_2d(flatchain, names, map_params, out_dir):
    """2D HOD / conditional stellar mass function at the posterior median (ZM15 Fig. 3).

    Colour = mean number of galaxies per dex in M* within haloes of mass M_h,
    Phi(M*|M_h) = -d<N(>M*|M_h)>/dlog10 M*.
    """
    med = {n: float(np.percentile(flatchain[:, i], 50)) for i, n in enumerate(names)}
    mh = np.linspace(11.0, 15.2, 90)
    ms = np.linspace(9.5, 11.9, 90)
    log10mh = jnp.asarray(mh)
    cen_keys = ("lg_m1h", "lg_m0star", "beta", "delta", "gamma",
                "sigma_lnmstar", "eta", "fc")
    sat_keys = cen_keys + ("bsat", "beta_sat", "bcut", "beta_cut", "alpha_sat")
    ncum = np.empty((len(ms), len(mh)))
    for j, mthr in enumerate(ms):
        nc = np.asarray(n_cen_thresh_zu15(log10mh, log10m_star_thresh=float(mthr),
                                          **{k: med[k] for k in cen_keys}))
        ns = np.asarray(n_sat_thresh_zu15(log10mh, log10m_star_thresh=float(mthr),
                                          **{k: med[k] for k in sat_keys}))
        ncum[j] = nc + ns
    phi = -np.gradient(ncum, ms, axis=0)          # per dex M*
    phi[phi <= 0] = np.nan

    # central ridge: M*(Mh) median SHMR
    mstar_ridge = np.asarray(_mstar_from_mh_zu15(
        log10mh, **{k: med[k] for k in ("lg_m1h", "lg_m0star", "beta", "delta", "gamma")}))

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    import matplotlib.colors as mcolors
    pcm = ax.pcolormesh(mh, ms, np.log10(phi), cmap="viridis", shading="auto",
                        norm=mcolors.Normalize(vmin=-3, vmax=1.2))
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(r"$\log_{10}\,\Phi(M_*\,|\,M_h)$  [dex$^{-1}$]")
    ax.plot(mh, mstar_ridge, "-", color="w", lw=2, alpha=0.85,
            label=r"central SHMR $M_*(M_h)$")
    ax.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot\,h^{-1}])$")
    ax.set_ylabel(r"$\log_{10}(M_*\,/\,M_\odot)$")
    ax.set_xlim(mh.min(), mh.max()); ax.set_ylim(ms.min(), ms.max())
    ax.set_title("2D HOD — conditional stellar mass function (posterior median)")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
    _save(fig, out_dir, "csmf_2d")


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
# quantitative posterior statistics  (writes posterior_stats.json)
# ---------------------------------------------------------------------------

def _pctl(a):
    """(median, -68, +68) tuple from a 1-D sample array."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    q16, q50, q84 = np.percentile(a, [16, 50, 84])
    return (float(q50), float(q50 - q16), float(q84 - q50))


def compute_posterior_stats(out_dir, flatchain, names, map_params,
                            predictor, theta_cosmo, bins, pi_max_h, h, z_ref,
                            chi2_samples=None, n_chi=150, n_derived=300,
                            n_analytic=800, n_fsat=300, n_walkers=32):
    """Derive quantitative posterior results and write ``posterior_stats.json``.

    Covers: MCMC convergence (autocorr time, N_eff, acceptance), the posterior
    :math:`\\chi^2` distribution, parameter correlations / prior-domination, per-bin
    effective mass / bias / satellite fraction, and SHMR-derived quantities
    (peak star-formation efficiency, halo mass at fixed :math:`M_*`), all with
    68 % credible intervals.
    """
    from hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint import JointZM15

    stats: dict = {}
    ndim = len(names)

    # ── convergence ───────────────────────────────────────────────────────
    # chain.h5 was written incrementally (store flag unset for read-back), so
    # recover the per-walker chain by reshaping the (already burn-in-discarded)
    # flatchain: emcee flattens (n_prod, n_walkers, ndim) step-major.
    conv = {}
    try:
        import emcee.autocorr
        n_prod = len(flatchain) // n_walkers
        chain3 = flatchain[:n_prod * n_walkers].reshape(n_prod, n_walkers, ndim)
        tau = emcee.autocorr.integrated_time(chain3, tol=0)
        conv = {
            "n_walkers": int(n_walkers), "n_production": int(n_prod),
            "n_samples": int(len(flatchain)),
            "tau_mean": float(np.nanmean(tau)), "tau_max": float(np.nanmax(tau)),
            "n_eff": float(len(flatchain) / np.nanmean(tau)),
            "n_indep_per_walker": float(n_prod / np.nanmean(tau)),
            "tau_per_param": {n: float(t) for n, t in zip(names, tau)},
        }
    except Exception as exc:
        conv = {"error": str(exc)}
    stats["convergence"] = conv

    # ── parameter correlations + prior-domination ─────────────────────────
    corr = np.corrcoef(flatchain, rowvar=False)
    pairs = []
    for i in range(ndim):
        for j in range(i + 1, ndim):
            pairs.append([names[i], names[j], float(corr[i, j])])
    pairs.sort(key=lambda x: -abs(x[2]))
    stats["correlations_top"] = pairs[:10]

    std = flatchain.std(axis=0)
    prior_dom = {}
    for i, n in enumerate(names):
        lo, hi = FREE_PARAMS[n][0]
        prior_std = (hi - lo) / np.sqrt(12.0)          # uniform-prior std
        # fraction of the prior 68 % width the posterior still occupies
        post68 = float(np.percentile(flatchain[:, i], 84)
                       - np.percentile(flatchain[:, i], 16))
        prior_dom[n] = {
            "post_std": float(std[i]),
            "std_ratio": float(std[i] / prior_std),     # →1 = prior-dominated
            "post68_over_prior_width": float(post68 / (hi - lo)),
        }
    stats["prior_domination"] = prior_dom

    # ── posterior chi2 distribution ───────────────────────────────────────
    # chi2_samples is supplied by compute_wp_ngal (computed for free during the
    # wp pass).  Only when running --stats standalone (no figures) do we pay a
    # dedicated wp pass here.
    hod = predictor._hod
    m_grid = np.asarray(hod._m_grid, dtype=float)
    log10m_grid = hod._log10m_grid
    ndof = 0
    for b in bins:
        ndof += len(b["rp"]) + 1
    ndof -= len(names)

    if chi2_samples is None:
        fit = JointZM15(bins, predictor, theta_cosmo, h=h, z=z_ref,
                        pi_max_h=pi_max_h, fit_esd=False)
        idx = _subsample(len(flatchain), n_chi, seed=7)
        chi2_samples = []
        for k, i in enumerate(idx):
            try:
                chi2_samples.append(float(sum(fit._bin_chi2(flatchain[i], b)["total"]
                                              for b in bins)))
            except Exception:
                chi2_samples.append(np.nan)
            if (k + 1) % 25 == 0:
                print(f"    chi2 draw {k + 1}/{len(idx)}")
        chi2_samples = np.asarray(chi2_samples, dtype=float)
    c2 = np.asarray(chi2_samples, dtype=float)
    c2 = c2[np.isfinite(c2)]
    stats["chi2_posterior"] = {
        "ndof": int(ndof), "n_draws": int(len(c2)),
        "chi2_med": float(np.percentile(c2, 50)),
        "chi2_lo68": float(np.percentile(c2, 16)),
        "chi2_hi68": float(np.percentile(c2, 84)),
        "chi2_per_dof_med": float(np.percentile(c2, 50) / ndof),
        "chi2_per_dof_68": [float(np.percentile(c2, 16) / ndof),
                            float(np.percentile(c2, 84) / ndof)],
    }

    # ── per-bin effective mass / bias / satellite fraction (cheap analytic) ─
    dndm_b = {b["label"]: np.asarray(hod._hmf.dndm(hod._m_grid, float(b["z"]), theta_cosmo))
              for b in bins}
    bias_b = {b["label"]: np.asarray(hod._bias(hod._m_grid, float(b["z"]), theta_cosmo))
              for b in bins}
    idx = _subsample(len(flatchain), n_derived, seed=7)
    perbin = {b["label"]: {"fsat": [], "meff": [], "beff": [], "ng": []} for b in bins}
    for i in idx:
        pf = dict(zip(names, [float(v) for v in flatchain[i]]))
        for b in bins:
            p = dict(pf); p["log10m_star_thresh"] = b["thresh"]
            if b.get("max") is not None:
                p["log10m_star_max"] = b["max"]
            try:
                with jax.disable_jit():
                    nc, ns = hod.nc_ns(log10m_grid, p)
                nc = np.asarray(nc, dtype=float); ns = np.asarray(ns, dtype=float)
                nt = nc + ns
                dn = dndm_b[b["label"]]
                n = float(np.trapezoid(dn * nt, m_grid))
                if n > 0:
                    perbin[b["label"]]["ng"].append(n)
                    perbin[b["label"]]["fsat"].append(float(np.trapezoid(dn * ns, m_grid) / n))
                    # occupation-weighted mean of log10 M_h (bias-consistent)
                    perbin[b["label"]]["meff"].append(
                        float(np.trapezoid(dn * nt * np.log10(m_grid), m_grid) / n))
                    perbin[b["label"]]["beff"].append(
                        float(np.trapezoid(dn * nt * bias_b[b["label"]], m_grid) / n))
            except Exception:
                pass
    stats["per_bin"] = {
        lbl: {"z": next(b["z"] for b in bins if b["label"] == lbl),
              "thresh": next(b["thresh"] for b in bins if b["label"] == lbl),
              "max": next(b["max"] for b in bins if b["label"] == lbl),
              "fsat": _pctl(v["fsat"]),
              "log10_meff": _pctl(v["meff"]) if v["meff"] else (np.nan,) * 3,
              "beff": _pctl(v["beff"])}
        for lbl, v in perbin.items()}

    # ── SHMR-derived quantities (analytic, many draws) ────────────────────
    mh_grid = jnp.linspace(10.5, 15.0, 300)
    mh_np = np.asarray(mh_grid)
    ia = _subsample(len(flatchain), n_analytic, seed=8)
    keys = ("lg_m1h", "lg_m0star", "beta", "delta", "gamma")
    m_peak, sfe_peak = [], []
    mh_at = {10.0: [], 10.5: [], 11.0: []}
    for i in ia:
        pf = dict(zip(names, flatchain[i]))
        mstar = np.asarray(_mstar_from_mh_zu15(mh_grid, **{k: float(pf[k]) for k in keys}))
        ratio = mstar - mh_np                     # log10(M*/Mh)
        kpk = int(np.nanargmax(ratio))
        m_peak.append(mh_np[kpk]); sfe_peak.append(10.0 ** ratio[kpk])
        for thr in mh_at:
            if mstar.min() < thr < mstar.max():
                mh_at[thr].append(float(np.interp(thr, mstar, mh_np)))
    stats["shmr_derived"] = {
        "log10_Mh_peak_SFE": _pctl(m_peak),
        "peak_SFE_MstarOverMh": _pctl(sfe_peak),
        "log10_Mh_at_Mstar": {f"{k:.1f}": _pctl(v) for k, v in mh_at.items()},
    }

    # ── cumulative satellite fraction at reference thresholds ─────────────
    n_cen_sat = _make_fast_occ(predictor, theta_cosmo, z_ref)
    ifs = _subsample(len(flatchain), n_fsat, seed=9)
    fsat_ref = {10.0: [], 10.5: [], 11.0: []}
    for i in ifs:
        pf = dict(zip(names, flatchain[i]))
        for thr in fsat_ref:
            nc, ns = n_cen_sat(pf, thr)
            tot = nc + ns
            if tot > 0:
                fsat_ref[thr].append(ns / tot)
    stats["fsat_cumulative"] = {f">{k:.1f}": _pctl(v) for k, v in fsat_ref.items()}
    stats["z_ref"] = float(z_ref)

    with open(os.path.join(out_dir, "posterior_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # ── human-readable summary ────────────────────────────────────────────
    print("\n================ POSTERIOR STATISTICS ================")
    c = stats["convergence"]
    if "error" not in c:
        print(f"convergence: tau_mean={c['tau_mean']:.1f}  tau_max={c['tau_max']:.1f}  "
              f"N_eff={c['n_eff']:.0f}  indep/walker={c['n_indep_per_walker']:.0f}"
              + (f"  accept={c['acceptance_mean']:.2f}" if 'acceptance_mean' in c else ""))
    cc = stats["chi2_posterior"]
    print(f"chi2 posterior: {cc['chi2_med']:.1f} (+{cc['chi2_hi68']-cc['chi2_med']:.1f}"
          f"/-{cc['chi2_med']-cc['chi2_lo68']:.1f})  chi2/dof={cc['chi2_per_dof_med']:.3f} "
          f"[{cc['chi2_per_dof_68'][0]:.3f},{cc['chi2_per_dof_68'][1]:.3f}]  ndof={cc['ndof']}")
    print("top correlations:", ", ".join(f"{a}-{b}:{r:+.2f}" for a, b, r in stats["correlations_top"][:5]))
    print(f"{'bin':11s} {'f_sat':>16s} {'log10 M_eff':>16s} {'b_eff':>16s}")
    for lbl, v in stats["per_bin"].items():
        f = v["fsat"]; m = v["log10_meff"]; b_ = v["beff"]
        print(f"{lbl:11s} {f[0]:.3f}+{f[2]:.3f}-{f[1]:.3f}   "
              f"{m[0]:.2f}+{m[2]:.2f}-{m[1]:.2f}    {b_[0]:.2f}+{b_[2]:.2f}-{b_[1]:.2f}")
    sd = stats["shmr_derived"]
    print(f"SHMR: log10 Mh(peak SFE)={sd['log10_Mh_peak_SFE'][0]:.2f}  "
          f"peak M*/Mh={sd['peak_SFE_MstarOverMh'][0]:.4f}")
    print("f_sat(>M*):", {k: round(v[0], 3) for k, v in stats["fsat_cumulative"].items()})
    print(f"stats -> {os.path.join(out_dir, 'posterior_stats.json')}")
    return stats


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
                         "(corner,wp,ngal,shmr,hod,fsat,smf,constraints), 'all', or 'none'")
    ap.add_argument("--stats", action="store_true",
                    help="Also compute quantitative posterior stats -> posterior_stats.json "
                         "(convergence, chi2 distribution, per-bin M_eff/b_eff/f_sat, SHMR)")
    ap.add_argument("--n-chi", type=int, default=150,
                    help="Draws for the posterior chi2 distribution (--stats)")
    ap.add_argument("--n-walkers", type=int, default=32,
                    help="Walkers used in the fit (for autocorr reshape; --stats)")
    args = ap.parse_args()

    _ALL = {"corner", "wp", "ngal", "shmr", "hod", "fsat", "smf", "constraints",
            "scatter", "mhalo", "csmf"}
    if args.figures in ("all", ""):
        want = set(_ALL)
    elif args.figures == "none":
        want = set()
    else:
        want = set(f.strip() for f in args.figures.split(","))

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

    # -- cheap analytic figures (no predictor) — do first to surface errors --
    if "corner" in want:
        print("\n[corner]")
        fig_corner(flatchain, names, img_dir)
    if "constraints" in want:
        print("[constraints]")
        fig_constraints(summary, names, img_dir)
    if "scatter" in want:
        print("[scatter]")
        fig_scatter(flatchain, names, img_dir, args.n_draws_analytic)
    if "csmf" in want:
        print("[csmf_2d]")
        fig_csmf_2d(flatchain, names, map_params, img_dir)

    # -- build predictor + data --------------------------------------------
    need_model = (want & {"wp", "ngal", "shmr", "hod", "fsat", "smf", "mhalo"}) or args.stats
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

        # expensive pass: wp + n_gal (also yields the per-draw chi2 for --stats)
        wp_bands = ng_samples = chi2_samples = None
        if want & {"wp", "ngal"} or args.stats:
            nd = args.n_draws
            print(f"\n[wp/n_gal]  {nd} posterior draws "
                  f"(~{nd * 7 / 60:.0f} min after compile) ...")
            wp_bands, ng_samples, n_used, chi2_samples = compute_wp_ngal(
                predictor, theta_cosmo, bins, pi_max_h, flatchain, names, nd)
            print(f"    used {n_used} draws")
        if "wp" in want:
            fig_wp_bins(bins, wp_bands, img_dir)
        if "ngal" in want:
            fig_ngal(bins, ng_samples, img_dir)

        if "shmr" in want:
            print("[shmr]")
            fig_shmr(flatchain, names, z_ref, img_dir, args.n_draws_analytic)
        if "hod" in want:
            print("[hod_occupation]")
            fig_hod_occupation(bins, flatchain, names, img_dir,
                               min(args.n_draws_analytic, 300))
        if "mhalo" in want:
            print("[mhalo_mstar]")
            fig_mhalo_mstar(predictor, theta_cosmo, bins, flatchain, names, img_dir,
                            args.n_draws_analytic)

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

        if args.stats:
            print("\n[stats]  convergence + chi2 distribution + derived quantities ...")
            compute_posterior_stats(out_dir, flatchain, names, map_params,
                                    predictor, theta_cosmo, bins, pi_max_h, h, z_ref,
                                    chi2_samples=chi2_samples, n_chi=args.n_chi,
                                    n_derived=args.n_draws_derived, n_fsat=args.n_draws_derived,
                                    n_walkers=args.n_walkers)

    print(f"\nAll figures -> {img_dir}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
