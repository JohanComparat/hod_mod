r"""Figures for the tier-2 forecast docs page (docs/tier2_forecast.rst).

Reads the ``tier2_forecast_<tag>.npz`` written by ``run_tier2_forecast`` and
produces the ``tier2_forecast__*.png`` suite (dpi=120), copying each into
``docs/_images/`` like the stage-4 script does.  Everything is recomputed from
the stored (d0, J, σ_noise, meta) by row masking — no model re-evaluation.

Usage::

    python -m hod_mod.scripts.forecasts.make_tier2_figures \
        <results>/tier2_forecast/tier2_forecast_nb6.npz [--compare nb1.npz]
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hod_mod.forecast.forward_jax import PARAM_NAMES, _IDX, TIER2_ZSLOPES  # noqa: E402
from hod_mod.forecast import params, fisher  # noqa: E402

_PFX = "tier2_forecast__"
_DPI = 120
# fixed sector colors (categorical, assigned by entity — matplotlib tab10)
_SEC_COLOR = {"cosmology": "C0", "shmr": "C1", "gas": "C2",
              "baryon": "C3", "agn": "C4", "retired": "0.6"}
EVOL_PARAMS = list(TIER2_ZSLOPES) + ["z_gas_zs", "agn_mu_bh_zs"]
SPECTRAL_PARAMS = ["t_prof_slope", "z_gas_norm", "z_gas_mslope", "z_gas_zs",
                   "agn_gamma", "agn_fabs", "kt_norm", "kt_slope"]


def _load(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = {k[5:]: d.pop(k) for k in list(d) if k.startswith("meta_")}
    d["rel"] = np.where(np.abs(d["d0"]) > 0,
                        d["sigma_noise"] / np.abs(d["d0"]), np.inf)
    d["names"] = [str(n) for n in d["param_names"]]
    return d


def _sigma_of(d, mask, prior=None):
    prior = d["prior"] if prior is None else prior
    m = mask & np.isfinite(d["rel"])
    F = fisher.fisher_matrix(d["d0"][m], d["J"][m], rel_err=d["rel"][m],
                             prior_sigma=prior)
    return fisher.constraints(F, scale=d["prior"])


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, _PFX + name + ".png")
    fig.savefig(p, dpi=_DPI)
    plt.close(fig)
    print("[fig]", p)
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            if os.path.isdir(os.path.join(root, "docs")):
                break
            root = os.path.dirname(root)
        dimg = os.path.join(root, "docs", "_images")
        if os.path.isdir(dimg):
            shutil.copyfile(p, os.path.join(dimg, os.path.basename(p)))
    except Exception as e:                                   # pragma: no cover
        print("  (docs copy skipped:", e, ")")


# ---------------------------------------------------------------------------

def fig_cell_grid(d, out_dir):
    """(z, M*) heatmaps: fiducial n_gal and total per-cell S/N."""
    meta = d["meta"]
    cells = meta["kind"] == "cell"
    zc = np.unique(meta["zeff"][cells])
    mlo = np.unique(meta["m_lo"][cells])
    ngal = np.full((len(mlo), len(zc)), np.nan)
    snr = np.full((len(mlo), len(zc)), np.nan)
    snr_row = np.where(np.isfinite(d["rel"]), 1.0 / d["rel"], 0.0) ** 2
    for i, m1 in enumerate(mlo):
        for j, z in enumerate(zc):
            sel = cells & (meta["m_lo"] == m1) & (meta["zeff"] == z)
            g = sel & (meta["obs"] == "n_gal")
            if g.any():
                ngal[i, j] = d["d0"][g][0]
            snr[i, j] = np.sqrt(snr_row[sel].sum())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, val, ttl in ((axes[0], np.log10(ngal), r"$\log_{10}\bar n_g$ [$h^3$Mpc$^{-3}$]"),
                         (axes[1], np.log10(snr), r"$\log_{10}$ total S/N per cell")):
        im = ax.imshow(val, origin="lower", aspect="auto", cmap="viridis",
                       extent=[zc[0] - 0.05, zc[-1] + 0.05, mlo[0], mlo[-1] + 0.2])
        ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$\log_{10} M_*$ bin")
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Tier-2 volume-limited cell grid", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir, "cell_grid")


def fig_noise_budget(d, out_dir):
    """Relative errors vs scale for a representative mid-z cell + shell."""
    meta = d["meta"]
    zc = np.unique(meta["zeff"][meta["kind"] == "cell"])
    z0 = zc[len(zc) // 2]
    m0 = np.nanmin(meta["m_lo"][meta["kind"] == "cell"])
    cell = (meta["zeff"] == z0) & (meta["m_lo"] == m0)
    shell = (meta["zeff"] == z0) & (meta["kind"] == "shell")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for name, sel, ax, xl in (("wp", cell, axes[0], r"$r_p$ [Mpc/$h$]"),
                              ("ds", cell, axes[0], None),
                              ("cl_gX", cell, axes[1], r"$\ell$"),
                              ("cl_gy", cell, axes[1], None),
                              ("cl_XX", shell, axes[1], None),
                              ("xlf", shell, axes[2], r"$\log_{10}L_X$")):
        s = sel & (meta["obs"] == name)
        if not s.any():
            continue
        x, r = meta["x"][s], d["rel"][s]
        ok = np.isfinite(r)
        ax.plot(x[ok], r[ok], "o-", ms=3, lw=1.2, label=name)
        if xl:
            ax.set_xlabel(xl)
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8)
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_ylabel(r"$\sigma/|d|$ (physical noise)")
    fig.suptitle(rf"Noise budget at $z\simeq{z0:.2f}$ "
                 rf"(cell $\log_{{10}}M_*>{m0:.1f}$ + shell)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "noise_budget")


def _ellipse(ax, cov2, x0, y0, color, label, nsig=1.0):
    from matplotlib.patches import Ellipse
    w, v = np.linalg.eigh(cov2)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    for k, alpha in ((1.52, 0.6), (2.48, 0.25)):    # 1σ, 2σ (2D)
        e = Ellipse((x0, y0), 2 * k * np.sqrt(w[-1]), 2 * k * np.sqrt(w[0]),
                    angle=ang, facecolor=color, alpha=alpha * 0.5,
                    edgecolor=color, lw=1.2,
                    label=label if k < 2 else None)
        ax.add_patch(e)


def fig_cosmo_constraints(d, out_dir):
    """Ω_m–σ8: everything-free vs astrophysics-pinned."""
    names = d["names"]
    iom, is8 = names.index("Omega_m"), names.index("sigma8")
    rm = d["rmin"][0]
    cov = d[f"cov_rmin{rm}"]
    cov_ap = d["cov_astro_pinned"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    x0, y0 = d["fid"][iom], d["fid"][is8]
    _ellipse(ax, cov[np.ix_([iom, is8], [iom, is8])], x0, y0, "C0",
             "all 61 params free")
    _ellipse(ax, cov_ap[np.ix_([iom, is8], [iom, is8])], x0, y0, "C1",
             "astrophysics pinned")
    ax.autoscale_view()
    ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$\sigma_8$")
    ax.legend(fontsize=9)
    ax.set_title(rf"Cosmology: marginalized vs astro-pinned ($R_\min={rm}$)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "cosmo_constraints")


def fig_astro_sectors(d, out_dir):
    """σ_post/σ_prior per parameter, grouped by sector, ± cosmology pinned."""
    names = d["names"]
    rm = d["rmin"][0]
    sig = d[f"sigma_rmin{rm}"]
    sig_cp = d["sigma_cosmo_pinned"]
    order = [n for sec in ("cosmology", "shmr", "gas", "baryon", "agn")
             for n in params.SECTORS[sec]]
    y = np.arange(len(order))
    ratio = np.array([sig[names.index(n)] / d["prior"][names.index(n)]
                      for n in order])
    ratio_cp = np.array([sig_cp[names.index(n)] / d["prior"][names.index(n)]
                         for n in order])
    col = [_SEC_COLOR[next(s for s, nn in params.SECTORS.items() if n in nn)]
           for n in order]
    fig, ax = plt.subplots(figsize=(7.0, 11.0))
    ax.barh(y, ratio, color=col, alpha=0.85, height=0.62,
            edgecolor="white", lw=0.5)
    ax.plot(ratio_cp, y, "k.", ms=5, label="cosmology pinned")
    ax.axvline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([params.PARAM_LATEX.get(n, n) for n in order], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{\rm posterior}/\sigma_{\rm prior}$")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=_SEC_COLOR[s])
               for s in ("cosmology", "shmr", "gas", "baryon", "agn")]
    ax.legend(handles + [plt.Line2D([], [], color="k", marker=".", ls="")],
              ["cosmology", "SHMR", "gas", "baryon", "AGN", "cosmo pinned"],
              fontsize=8, loc="lower right")
    ax.set_title("What the data teach per parameter (ratio < 1 = informative)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "astro_sectors")


def fig_zevolution(d, out_dir):
    """Evolution-slope constraints + the reconstructed lg M1(z) band."""
    names = d["names"]
    rm = d["rmin"][0]
    sig = d[f"sigma_rmin{rm}"]
    cov = d[f"cov_rmin{rm}"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    y = np.arange(len(EVOL_PARAMS))
    vals = [sig[names.index(n)] for n in EVOL_PARAMS]
    axes[0].barh(y, vals, color="C1", height=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([params.PARAM_LATEX.get(n, n) for n in EVOL_PARAMS],
                            fontsize=9)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\sigma$ (slope per $\ln(1+z)$)")
    axes[0].invert_yaxis()
    axes[0].set_title("Redshift-evolution constraints", fontsize=10)

    i0, i1 = names.index("lg_m1h"), names.index("lg_m1h_zs")
    z = np.linspace(0.05, 0.95, 60)
    x = np.log((1 + z) / 1.3)
    mu = d["fid"][i0] + d["fid"][i1] * x
    var = cov[i0, i0] + x ** 2 * cov[i1, i1] + 2 * x * cov[i0, i1]
    s = np.sqrt(var)
    axes[1].fill_between(z, mu - s, mu + s, color="C0", alpha=0.35,
                         label=r"$\pm1\sigma$")
    axes[1].plot(z, mu, color="C0", lw=1.5)
    axes[1].set_xlabel(r"$z$")
    axes[1].set_ylabel(r"$\lg M_1(z)$")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Reconstructed SHMR pivot-mass evolution", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "zevolution")


def fig_probe_attribution(d, out_dir):
    names = d["names"]
    labels = [str(x) for x in d["cum_labels"]]
    cum = d["cum_sigma"]
    iom, is8 = names.index("Omega_m"), names.index("sigma8")
    ir, ig = names.index("agn_rho"), names.index("t_prof_slope")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xpos = np.arange(len(labels))
    for i, lab, c in ((iom, r"$\sigma(\Omega_m)$", "C0"),
                      (is8, r"$\sigma(\sigma_8)$", "C1"),
                      (ir, r"$\sigma(\rho_{\rm BH})$", "C4"),
                      (ig, r"$\sigma(\Gamma_T)$", "C2")):
        ax.plot(xpos, cum[:, i], "o-", color=c, label=lab, lw=1.4, ms=4)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylabel(r"marginalized $1\sigma$")
    ax.legend(fontsize=9)
    ax.set_title("Cumulative probe build-up (all parameters free)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "probe_attribution")


def fig_degeneracies(d, out_dir):
    names = d["names"]
    rm = d["rmin"][0]
    corr = np.nan_to_num(np.asarray(
        d[f"cov_rmin{rm}"] /
        np.outer(np.sqrt(np.diag(d[f"cov_rmin{rm}"])),
                 np.sqrt(np.diag(d[f"cov_rmin{rm}"])))))
    pairs = fisher.top_degeneracies(corr, names, k=12)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    y = np.arange(len(pairs))
    vals = [c for _, _, _, c in pairs]
    cols = ["C3" if v > 0 else "C0" for v in vals]
    ax.barh(y, vals, color=cols, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n1} × {n2}" for _, n1, n2, _ in pairs], fontsize=8)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("posterior correlation")
    ax.invert_yaxis()
    ax.set_title("Strongest surviving degeneracies (61 params free)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "degeneracies")


def fig_agn_sector(d, out_dir):
    """Fiducial wp_agn per L_X bin + the AGN parameter constraints."""
    meta = d["meta"]
    names = d["names"]
    rm = d["rmin"][0]
    sig = d[f"sigma_rmin{rm}"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    agn_blocks = np.unique(meta["block"][meta["kind"] == "wp_agn"])
    b0 = agn_blocks[len(agn_blocks) // 2]
    sel = (meta["block"] == b0) & (meta["obs"] == "wp_agn")
    subs = np.unique(meta["sub"][sel])
    cmap = plt.get_cmap("viridis")
    for k in subs:
        s = sel & (meta["sub"] == k)
        ok = np.isfinite(d["rel"][s])
        axes[0].errorbar(meta["x"][s][ok], d["d0"][s][ok],
                         yerr=d["sigma_noise"][s][ok], fmt="o-", ms=3, lw=1.2,
                         color=cmap(0.15 + 0.7 * k / max(len(subs) - 1, 1)),
                         label=rf"$\log L_X$ bin {k}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$r_p$ [Mpc/$h$]")
    axes[0].set_ylabel(r"$w_{p,\rm AGN}$ [Mpc/$h$]")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"AGN clustering ({b0}) + physical errors", fontsize=10)

    agn_names = params.SECTORS["agn"]
    y = np.arange(len(agn_names))
    axes[1].barh(y, [sig[names.index(n)] / d["prior"][names.index(n)]
                     for n in agn_names], color="C4", height=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([params.PARAM_LATEX.get(n, n) for n in agn_names],
                            fontsize=9)
    axes[1].set_xscale("log")
    axes[1].axvline(1.0, color="0.4", lw=0.8, ls="--")
    axes[1].set_xlabel(r"$\sigma_{\rm post}/\sigma_{\rm prior}$")
    axes[1].invert_yaxis()
    axes[1].set_title("AGN sector constraints", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "agn_sector")


def fig_band_spectroscopy(d, out_dir, d_cmp=None):
    """What the X-ray energy bands buy for the spectral parameters."""
    names = d["names"]
    meta = d["meta"]
    rm = d["rmin"][0]
    keep = d[f"keep_rmin{rm}"].astype(bool)
    no_x = keep & ~np.isin(meta["obs"], ("cl_gX", "cl_XX"))
    sig_all = d[f"sigma_rmin{rm}"]
    _, sig_nox, _ = _sigma_of(d, no_x)
    y = np.arange(len(SPECTRAL_PARAMS))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    nb = int(d["n_bands"]) if "n_bands" in d else 0
    ax.barh(y - 0.18, [sig_all[names.index(n)] for n in SPECTRAL_PARAMS],
            color="C2", height=0.34, label=f"with {nb} band spectra")
    ax.barh(y + 0.18, [sig_nox[names.index(n)] for n in SPECTRAL_PARAMS],
            color="0.65", height=0.34, label="X-ray spectra removed")
    if d_cmp is not None:
        cn = d_cmp["names"]
        rmc = d_cmp["rmin"][0]
        sc = d_cmp[f"sigma_rmin{rmc}"]
        ax.plot([sc[cn.index(n)] for n in SPECTRAL_PARAMS], y, "k.",
                ms=6, label="broad band only (nb1 run)")
    ax.set_yticks(y)
    ax.set_yticklabels([params.PARAM_LATEX.get(n, n) for n in SPECTRAL_PARAMS],
                       fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"marginalized $1\sigma$")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title("Energy-band spectroscopy of the hot gas + AGN", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "band_spectroscopy")


def make_all(npz_path, out_dir, compare=None):
    d = _load(npz_path)
    d_cmp = _load(compare) if compare else None
    fig_cell_grid(d, out_dir)
    fig_noise_budget(d, out_dir)
    fig_cosmo_constraints(d, out_dir)
    fig_astro_sectors(d, out_dir)
    fig_zevolution(d, out_dir)
    fig_probe_attribution(d, out_dir)
    fig_degeneracies(d, out_dir)
    fig_agn_sector(d, out_dir)
    fig_band_spectroscopy(d, out_dir, d_cmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--compare", default=None,
                    help="second npz (e.g. the n_bands=1 run) to overlay")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    make_all(args.npz, args.out_dir or os.path.dirname(os.path.abspath(args.npz)),
             compare=args.compare)


if __name__ == "__main__":
    main()
