r"""Figures + posterior table for the benchmark MAP+MCMC fit documentation page.

Reads a run produced by ``hod_mod.scripts.fitting.fit_benchmark_observables``
(``<label>_map.json``, ``<label>_mcmc_summary.json``, ``<label>_chain.npy`` and
``<label>_data.npz``), rebuilds the forward model, propagates the MCMC chain
through it, and writes into ``docs/_images/`` (prefix ``<label>__``):

* ``<label>__observables.png`` — one panel per fitted observable: the data with
  error bars, the fiducial ("truth") model, the posterior-median model, and the
  **68 / 95 / 99.7 % (1/2/3σ)** posterior-predictive credible bands.
* ``<label>__corner.png`` — the free-parameter posterior with 1/2/3σ contours,
  the injected truth and the MAP overlaid.

and, next to the run outputs, the posterior table
``<label>__posterior.csv`` / ``.json`` (per parameter: truth, MAP, median,
±1σ, 95 %, 99.7 %).

Colours follow the project data-viz palette: a light chart surface, ink data
points, a single-hue blue sequential ramp for the nested credible bands, and an
orange truth marker.

Usage::

    JAX_ENABLE_X64=1 HOD_MOD_RESULTS=/home/comparat/data/hod_mod_results \
      python -m hod_mod.scripts.fitting.plot_benchmark_fit --label benchmark_map_mcmc
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.forecast.forward_jax import ForwardModel
from hod_mod.forecast import params
from hod_mod.fitting import benchmark_data as bd  # noqa: F401 (documents the source)
from hod_mod.fitting.jax_inference import MultiProbeGaussianLikelihood

# --- palette (project data-viz reference; light surface) --------------------
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e2"
MEDIAN = "#184f95"          # sequential blue 600 — posterior median line
BAND = ["#6da7ec", "#9ec5f4", "#cde2fb"]   # 1σ (darkest) → 3σ (lightest)
TRUTH = "#eb6834"           # orange — injected fiducial
MAPC = "#1baf7a"            # aqua — MAP point

# 1/2/3σ two-sided percentiles and the 2-D enclosed-mass contour levels
_PCT = {1: (15.865, 84.135), 2: (2.275, 97.725), 3: (0.135, 99.865)}
_LEVELS_2D = (0.393, 0.865, 0.989)

# per-observable presentation (label, x-axis, y-axis, scales)
_OBS = {
    "xlf": dict(title="AGN X-ray luminosity function",
                xlabel=r"$\log_{10}\,L_X\ \mathrm{[erg/s]}$",
                ylabel=r"$\phi\ \mathrm{[Mpc^{-3}\,dex^{-1}]}$",
                xlog=False, ylog=True),
    "n_gal": dict(title="Galaxy number density",
                  xlabel="sample", ylabel=r"$n_\mathrm{gal}\ \mathrm{[Mpc^{-3}]}$",
                  xlog=False, ylog=True),
    "wp_agn": dict(title="AGN projected clustering",
                   xlabel=r"$r_p\ \mathrm{[Mpc/h]}$",
                   ylabel=r"$w_p(r_p)$", xlog=True, ylog=True),
}


def _run_dir():
    return os.path.join(os.environ.get("HOD_MOD_RESULTS",
                        "/home/comparat/data/hod_mod_results"), "fits", "benchmark")


def _images_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", "docs", "_images"))


def load_run(run_dir, label):
    with open(os.path.join(run_dir, f"{label}_mcmc_summary.json")) as fh:
        cfg = json.load(fh)
    dat = np.load(os.path.join(run_dir, f"{label}_data.npz"), allow_pickle=True)
    chain = np.load(os.path.join(run_dir, f"{label}_chain.npy"))
    return cfg, dat, chain


def build_likelihood(cfg, dat):
    """Rebuild the forward model + likelihood so ``like.model`` is available."""
    which = [str(w) for w in dat["which"]]
    free = [str(f) for f in dat["free"]]
    fm = ForwardModel(z_eff=float(dat["z_eff"]),
                      n_k=int(dat["n_k"]), n_m=int(dat["n_m"]))
    data = np.asarray(dat["data"], float)
    icov = np.diag(1.0 / np.asarray(dat["y_err"], float) ** 2)
    theta0 = params.fiducial_vector()
    like = MultiProbeGaussianLikelihood.from_forward_model(
        fm, which, data, icov, free, theta0=theta0, prior="planck")
    return like, which, free


def posterior_predictive(like, chain):
    """Model vectors for every chain sample: ``(n_samples, n_data)``."""
    model = jax.jit(jax.vmap(lambda x: like.model(x)))
    return np.asarray(model(jnp.asarray(chain, dtype=float)))


def _bands(pred_cols):
    """median and (lo, hi) at 1/2/3σ for one observable's columns."""
    med = np.median(pred_cols, axis=0)
    out = {}
    for s, (lo, hi) in _PCT.items():
        out[s] = (np.percentile(pred_cols, lo, axis=0),
                  np.percentile(pred_cols, hi, axis=0))
    return med, out


def plot_observables(cfg, dat, pred, out_png):
    row_obs = np.asarray([str(o) for o in dat["row_obs"]])
    row_x = np.asarray(dat["row_x"], float)
    data = np.asarray(dat["data"], float)
    yerr = np.asarray(dat["y_err"], float)
    truth = np.asarray(dat["truth"], float)
    which = [str(w) for w in dat["which"]]

    n = len(which)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, obs in zip(axes, which):
        m = row_obs == obs
        x = row_x[m]
        d, e, t = data[m], yerr[m], truth[m]
        pred_cols = pred[:, m]
        spec = _OBS.get(obs, dict(title=obs, xlabel="x", ylabel=obs,
                                  xlog=False, ylog=False))
        # an observable may concatenate several sub-samples (e.g. wp_agn = L_X
        # bins × r_p): a new segment starts wherever the abscissa resets.
        breaks = [0] + list(np.where(np.diff(x) < 0)[0] + 1) + [x.size]
        segs = [slice(a, b) for a, b in zip(breaks[:-1], breaks[1:])]
        multi = len(segs) > 1
        first = True
        for sl in segs:
            xs = x[sl]
            order = np.argsort(xs)
            xs = xs[order]
            ds, es, ts = d[sl][order], e[sl][order], t[sl][order]
            med, bands = _bands(pred_cols[:, sl][:, order])
            lab = (lambda s: (f"{s}σ posterior" if first else None))
            if xs.size >= 3:
                for s, col in zip((3, 2, 1), (BAND[2], BAND[1], BAND[0])):
                    lo, hi = bands[s]
                    ax.fill_between(xs, lo, hi, color=col, lw=0,
                                    label=lab(s), zorder=1)
                ax.plot(xs, med, color=MEDIAN, lw=2, zorder=3,
                        label="posterior median" if first else None)
                ax.plot(xs, ts, color=MUTED, lw=1.1, ls="--", zorder=2,
                        label="fiducial (truth)" if first else None)
                ax.errorbar(xs, ds, yerr=es, fmt="o", ms=4.5, color=INK,
                            ecolor=MUTED, elinewidth=1, capsize=2, zorder=4,
                            label="data" if first else None)
            else:                       # 1–2 points (e.g. n_gal)
                px = np.arange(xs.size, dtype=float)
                for s, col in zip((3, 2, 1), (BAND[2], BAND[1], BAND[0])):
                    lo, hi = bands[s]
                    ax.errorbar(px + 0.12, med, yerr=[med - lo, hi - med],
                                fmt="none", ecolor=col, elinewidth=7 - 2 * s,
                                zorder=s, label=lab(s))
                ax.plot(px + 0.12, med, "s", color=MEDIAN, ms=7, zorder=5,
                        label="posterior median" if first else None)
                ax.errorbar(px, ds, yerr=es, fmt="o", ms=7, color=INK,
                            ecolor=MUTED, elinewidth=1.2, capsize=3, zorder=6,
                            label="data" if first else None)
                ax.set_xlim(-0.5, xs.size - 0.5)
                ax.set_xticks(px)
                ax.set_xticklabels([f"bin {i+1}" for i in range(xs.size)])
            first = False

        line_mode = any((sl.stop - sl.start) >= 3 for sl in segs)
        if spec["xlog"] and line_mode:
            ax.set_xscale("log")
        if spec["ylog"]:
            ax.set_yscale("log")
        if multi:
            ax.text(0.03, 0.03, f"{len(segs)} sub-samples", transform=ax.transAxes,
                    fontsize=8, color=MUTED, ha="left", va="bottom")
        ax.set_title(spec["title"], fontsize=11)
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Benchmark fit — data vs. posterior-predictive model "
                 "(1/2/3σ)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", out_png)


def plot_corner(cfg, chain, free, out_png):
    import corner
    truth = [cfg["x_true"][n] for n in free]
    map_vals = ([cfg["map"]["params"][n] for n in free]
                if "map" in cfg else None)
    labels = [_PLABEL.get(n, n) for n in free]
    fig = corner.corner(
        chain, labels=labels, truths=truth, truth_color=TRUTH,
        levels=_LEVELS_2D, color=MEDIAN, bins=32, smooth=1.0,
        show_titles=True, title_fmt=".3f", title_kwargs=dict(fontsize=9),
        label_kwargs=dict(fontsize=10),
        plot_datapoints=False, fill_contours=True,
        hist_kwargs=dict(color=MEDIAN))
    if map_vals is not None:
        axes = np.array(fig.axes).reshape(len(free), len(free))
        for i in range(len(free)):
            for j in range(i):
                axes[i, j].plot(map_vals[j], map_vals[i], "s",
                                color=MAPC, ms=6, zorder=6)
        for i in range(len(free)):
            axes[i, i].axvline(map_vals[i], color=MAPC, lw=1.3)
    fig.suptitle("Posterior — 1/2/3σ contours (orange = truth, "
                 "green = MAP)", fontsize=12, y=1.02)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", out_png)


_PLABEL = {
    "Omega_m": r"$\Omega_m$", "sigma8": r"$\sigma_8$",
    "lg_m1h": r"$\lg M_{1h}$", "lg_m0star": r"$\lg M_{0*}$",
    "agn_log10_lstar": r"$\log_{10}L_\star^{\rm AGN}$",
}


def posterior_table(cfg, chain, free, out_stem):
    truth = cfg["x_true"]
    map_p = cfg.get("map", {}).get("params", {})
    rows = []
    for j, n in enumerate(free):
        c = chain[:, j]
        med = float(np.median(c))
        p16, p84 = np.percentile(c, [15.865, 84.135])
        p025, p975 = np.percentile(c, [2.275, 97.725])
        p0015, p9985 = np.percentile(c, [0.135, 99.865])
        rows.append({
            "parameter": n, "truth": float(truth[n]),
            "map": float(map_p.get(n, np.nan)), "median": med,
            "minus_1sigma": med - float(p16), "plus_1sigma": float(p84) - med,
            "ci95_lo": float(p025), "ci95_hi": float(p975),
            "ci997_lo": float(p0015), "ci997_hi": float(p9985),
        })
    with open(out_stem + ".json", "w") as fh:
        json.dump(rows, fh, indent=2)
    cols = ["parameter", "truth", "map", "median", "minus_1sigma",
            "plus_1sigma", "ci95_lo", "ci95_hi", "ci997_lo", "ci997_hi"]
    with open(out_stem + ".csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    print("[table]", out_stem + ".csv")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="benchmark_map_mcmc")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--n-pred", type=int, default=400,
                    help="posterior draws propagated through the model for bands")
    args = ap.parse_args()

    run_dir = args.run_dir or _run_dir()
    img_dir = args.images_dir or _images_dir()
    os.makedirs(img_dir, exist_ok=True)

    cfg, dat, chain = load_run(run_dir, args.label)
    print(f"[load] {args.label}: chain {chain.shape}, "
          f"observables={list(dat['which'])}, free={list(dat['free'])}")
    like, which, free = build_likelihood(cfg, dat)
    # posterior-predictive bands only need a representative subsample of the
    # chain (evaluating the forward model over every draw is the costly step)
    rng = np.random.default_rng(0)
    n_pred = min(args.n_pred, chain.shape[0])
    sub = rng.choice(chain.shape[0], size=n_pred, replace=False)
    pred = posterior_predictive(like, chain[sub])
    print(f"[pred] forward model over {n_pred} posterior draws")

    plot_observables(cfg, dat, pred,
                     os.path.join(img_dir, f"{args.label}__observables.png"))
    plot_corner(cfg, chain, free,
                os.path.join(img_dir, f"{args.label}__corner.png"))
    posterior_table(cfg, chain, free,
                    os.path.join(run_dir, f"{args.label}__posterior"))


if __name__ == "__main__":
    main()
