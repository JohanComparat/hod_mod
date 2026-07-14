"""Figures for the BGS S1 full-model joint fit (galaxies + hot gas + AGN).

Reads a fit output directory written by
:mod:`hod_mod.scripts.fitting.fit_bgs_full_joint` (``flatchain.npz`` +
``map_result.json``) and produces three figures, overlaying the **MAP** and the
**posterior-median** model on the data:

* ``<prefix>__corner.png``      — posterior corner of the free params, MAP overlaid.
* ``<prefix>__observables.png`` — wp, ESD (DES/HSC/KIDS), XLF (z=0.1, 0.4), AGN
  bias b(L_X) and the X-ray broad-band w(theta): data +/- err vs MAP + median.
* ``<prefix>__xray_bands.png``  — the 15 narrow 0.1-keV band w(theta) panels.

``prefix`` is ``bgs_full_joint_fixedzm15`` (default) or ``bgs_full_joint_allparams``
(``--free-zm15``).  Figures go into the run directory; ``--docs`` additionally
writes them into ``docs/_images/`` with the names the ``bgs_full_joint`` doc page
embeds.

Usage
-----
::

    HOD_MOD_DATA_DIR=/home/comparat/data/hod_mod_data JAX_PLATFORMS=cpu \\
        python -m hod_mod.scripts.fitting.plot_bgs_full_joint              # fixed-ZM15
    ... plot_bgs_full_joint --free-zm15                                    # all-params run
    ... plot_bgs_full_joint --out-dir /path/to/run --docs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hod_mod.paths import results_root
from hod_mod.scripts.fitting.fit_bgs_full_joint import _DEFAULT_OBS

# 15 narrow bands: 0.5-0.6 ... 1.9-2.0 keV (see fit_xray_joint_bands._BANDS).
_BAND_LABELS = [f"{lo / 1000:.1f}–{(lo + 100) / 1000:.1f} keV"
                for lo in range(500, 2000, 100)]

_MAP_C, _MED_C, _DATA_C = "tab:green", "tab:blue", "k"


def _load(out: Path):
    """Return (chain, names, theta_med, theta_map) from a fit output dir."""
    npz = np.load(out / "flatchain.npz")
    chain = np.asarray(npz["flatchain"], float)
    names = [str(n) for n in npz["param_names"]]
    theta_med = np.median(chain, axis=0)
    theta_map = None
    mj = out / "map_result.json"
    if mj.exists():
        theta_map = np.asarray(json.load(open(mj))["theta"], float)
    return chain, names, theta_med, theta_map


def fig_corner(chain, names, theta_map, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    fig = corner.corner(
        chain, labels=names,
        truths=(list(theta_map) if theta_map is not None else None),
        truth_color=_MAP_C, quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_fmt=".2f", title_kwargs=dict(fontsize=7), label_kwargs=dict(fontsize=7))
    fig.suptitle("BGS S1 full-joint posterior  (green = MAP)", fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _overlay(ax, x, y_map, y_med):
    if y_map is not None:
        ax.plot(x, y_map, color=_MAP_C, lw=1.6, label="MAP")
    ax.plot(x, y_med, color=_MED_C, lw=1.6, ls="--", label="posterior median")


def fig_observables(J, pred_map, pred_med, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []  # each: a function ax -> None

    if "wp" in J.obs:
        d = J.data_gal["wp"]
        def _wp(ax, d=d):
            ax.errorbar(d["rp"], d["wp"], yerr=d["err"], fmt="o", ms=3, color=_DATA_C, label="data")
            _overlay(ax, d["rp"], (pred_map["wp"] if pred_map else None), pred_med["wp"])
            ax.set(xscale="log", yscale="log", xlabel=r"$r_p$ [Mpc/$h$]", ylabel=r"$w_p$")
            ax.set_title("wp")
        panels.append(_wp)

    if "esd" in J.obs:
        for sv in J.data_gal["esd"]:
            d = J.data_gal["esd"][sv]
            def _esd(ax, sv=sv, d=d):
                ax.errorbar(d["rp"], d["ds"], yerr=d["err"], fmt="o", ms=3, color=_DATA_C, label="data")
                _overlay(ax, d["rp"], (pred_map["esd"][sv] if pred_map else None), pred_med["esd"][sv])
                ax.set(xscale="log", yscale="log", xlabel=r"$r_p$ [Mpc/$h$]",
                       ylabel=r"$\Delta\Sigma$ [$M_\odot h/\mathrm{pc}^2$]")
                ax.set_title(f"ESD {sv}")
            panels.append(_esd)

    if "xlf" in J.obs:
        for z in J.XLF_Z:
            d = J.data_agn["xlf"][z]
            def _xlf(ax, z=z, d=d):
                ax.errorbar(d["log10lx"], d["phi"], yerr=d["err"], fmt="o", ms=3, color=_DATA_C, label="data")
                _overlay(ax, d["log10lx"], (pred_map["xlf"][z] if pred_map else None), pred_med["xlf"][z])
                ax.set(yscale="log", xlabel=r"$\log_{10} L_X$ [erg/s, hard]", ylabel=r"$\phi$ [Mpc$^{-3}$]")
                ax.set_title(f"XLF  z={z}")
            panels.append(_xlf)

    if "agn_bias" in J.obs:
        d = J.data_agn["bias"]
        def _bias(ax, d=d):
            ax.errorbar(d["log10lx_soft"], d["bias"], yerr=d["bias_err"], fmt="o", ms=3,
                        color=_DATA_C, label="data")
            # z varies point-to-point, so plot markers (not a smooth curve)
            if pred_map is not None:
                ax.plot(d["log10lx_soft"], pred_map["agn_bias"], "s", color=_MAP_C, ms=5, label="MAP")
            ax.plot(d["log10lx_soft"], pred_med["agn_bias"], "^", color=_MED_C, ms=5, label="median")
            ax.set(xlabel=r"$\log_{10} L_X^{\rm soft}$ [erg/s]", ylabel="halo bias")
            ax.set_title("AGN bias  b(L_X)")
        panels.append(_bias)

    if "xray_broad" in J.obs and hasattr(J, "data_xray_broad"):
        d = J.data_xray_broad
        def _broad(ax, d=d):
            ax.errorbar(d["theta_as"], d["wtheta"], yerr=d["err"], fmt="o", ms=3, color=_DATA_C, label="data")
            _overlay(ax, d["theta_as"], (pred_map.get("xray_broad") if pred_map else None),
                     pred_med["xray_broad"])
            ax.set(xscale="log", xlabel=r"$\theta$ [arcsec]", ylabel=r"broad $w(\theta)$")
            ax.set_title("X-ray broad (0.5-2 keV)")
        panels.append(_broad)

    n = len(panels)
    ncol = min(3, n) or 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow), squeeze=False)
    for i, pf in enumerate(panels):
        pf(axes[i // ncol][i % ncol])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("BGS S1 full-joint: data vs MAP / posterior-median model", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_xray_bands(J, pred_map, pred_med, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xb_med = pred_med["xray_bands"]
    xb_map = pred_map.get("xray_bands") if pred_map else None
    th = xb_med["th_as"]
    mask = xb_med["mask"]
    wtheta, err = xb_med["wtheta"], xb_med["err"]
    nb = wtheta.shape[0]
    labels = (_BAND_LABELS + [f"band {i}" for i in range(nb)])[:nb]
    # focus on the fitted-theta span (the model grid runs far past the data)
    th_lo, th_hi = th[mask].min() * 0.6, th[mask].max() * 2.0

    ncol = 4
    nrow = int(np.ceil(nb / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow),
                             squeeze=False, sharex=True)
    for b in range(nb):
        ax = axes[b // ncol][b % ncol]
        ax.errorbar(th[mask], wtheta[b][mask], yerr=err[b][mask], fmt="o", ms=2.5,
                    color=_DATA_C, lw=0.8)
        if xb_map is not None:
            ax.plot(th, xb_map["total"][b], color=_MAP_C, lw=1.2)
        ax.plot(th, xb_med["total"][b], color=_MED_C, lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(th_lo, th_hi)
        ax.set_title(labels[b], fontsize=8)
    for j in range(nb, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("BGS S1 full-joint: narrow-band $w(\\theta)$  "
                 "(black = data, green = MAP, blue dashed = median)", fontsize=12)
    fig.supxlabel(r"$\theta$ [arcsec]")
    fig.supylabel(r"$w(\theta)$")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _print_chi2(J, label, theta):
    bd = J.chi2_breakdown(theta)
    tot = float(sum(bd.values()))
    dof = max(J.n_data() - J.ndim, 1)
    br = {k: round(x, 1) for k, x in bd.items()}
    print(f"[plot] {label:7s} chi2/dof = {tot:.1f}/{dof} = {tot / dof:.3f}  {br}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="S1")
    ap.add_argument("--free-zm15", action="store_true",
                    help="the all-params run (bgs_full_joint_allparams)")
    ap.add_argument("--observables", nargs="+", default=None,
                    help=f"observable set the fit used (default {_DEFAULT_OBS})")
    ap.add_argument("--out-dir", default=None,
                    help="fit output dir (default results_root()/bgs_full_joint_<tag>)")
    ap.add_argument("--docs", action="store_true",
                    help="also write figures into docs/_images/ with the doc-page names")
    args = ap.parse_args(argv)

    tag = "allparams" if args.free_zm15 else "fixedzm15"
    prefix = f"bgs_full_joint_{tag}"
    out = Path(args.out_dir) if args.out_dir else results_root() / prefix
    if not (out / "flatchain.npz").exists():
        raise SystemExit(f"[plot] no flatchain.npz in {out} — run fit_bgs_full_joint first "
                         f"(or pass --out-dir).")

    chain, names, theta_med, theta_map = _load(out)
    dests = [out]
    if args.docs:
        docs_img = Path(__file__).resolve().parents[3] / "docs" / "_images"
        docs_img.mkdir(parents=True, exist_ok=True)
        dests.append(docs_img)

    # Build the model so we can overlay predictions.  It must match the chain's
    # dimensionality; a stale/incompatible file is caught rather than crashing deep.
    from hod_mod.fitting.full_joint import JointFull
    obs = args.observables or _DEFAULT_OBS
    print(f"[plot] building JointFull ({tag}, {len(names)} params) ...", flush=True)
    J = JointFull(sample=args.sample, free_zm15=args.free_zm15, observables=obs, verbose=False)
    if theta_med.size != J.ndim:
        raise SystemExit(f"[plot] chain has {theta_med.size} params but the model has "
                         f"{J.ndim} (names={J.names}); wrong --observables/--free-zm15?")
    if theta_map is not None and theta_map.size != J.ndim:
        print(f"[plot] map_result.json has {theta_map.size} params != model {J.ndim}; "
              f"ignoring the MAP overlay", flush=True)
        theta_map = None

    pred_med = J.predict(theta_med)
    pred_map = J.predict(theta_map) if theta_map is not None else None

    for dd in dests:
        fig_corner(chain, names, theta_map, dd / f"{prefix}__corner.png")
        fig_observables(J, pred_map, pred_med, dd / f"{prefix}__observables.png")
        if {"xray_bands", "xray_broad"} & J.obs:
            fig_xray_bands(J, pred_map, pred_med, dd / f"{prefix}__xray_bands.png")

    if theta_map is not None:
        _print_chi2(J, "MAP", theta_map)
    _print_chi2(J, "median", theta_med)
    print(f"[plot] figures -> {'  '.join(str(d) for d in dests)}", flush=True)


if __name__ == "__main__":
    main()
