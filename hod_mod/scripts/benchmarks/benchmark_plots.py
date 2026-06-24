"""Shared plotting helpers for HOD benchmark scripts.

All benchmark runner scripts import colour constants and figure helpers from
here so that every figure produced by the suite shares a consistent look.

Colour scheme
-------------
_COL_DATA  black  — data points and errorbars
_COL_MAP   C0     — MAP best-fit line and MCMC credible bands
_COL_PUB   C2     — published best-fit line and corner-plot truth markers

Line-style convention
---------------------
MAP prediction     solid  (lw=1.8)
MCMC median        dashed (lw=1.5, alpha=0.85)
Published best-fit dashed (lw=1.5)
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Unified colour scheme
# ---------------------------------------------------------------------------

_COL_DATA = "k"   # data points
_COL_MAP  = "C0"  # MAP prediction + MCMC bands
_COL_PUB  = "C2"  # published best-fit

# Per-sample colour cycle for multi-sample figures (Lange+2025 style)
_SAMPLE_COLORS = ["C0", "C1", "C3", "C4"]

_PARAM_LATEX: dict[str, str] = {
    "log10mmin":  r"$\log_{10}M_{\min}$",
    "sigma_logm": r"$\sigma_{\log M}$",
    "log10m0":    r"$\log_{10}M_0$",
    "log10m1":    r"$\log_{10}M_1$",
    "alpha":      r"$\alpha$",
    "kappa":      r"$\kappa$",
    "f_Gamma":    r"$f_\Gamma$",
    "A_cen":      r"$A_{\rm cen}$",
    "A_sat":      r"$A_{\rm sat}$",
    "S8":         r"$S_8$",
    "Omega_m":    r"$\Omega_m$",
    "sigma8":     r"$\sigma_8$",
    "p_off":      r"$p_{\rm off}$",
    "R_off":      r"$R_{\rm off}$",
}


# ---------------------------------------------------------------------------
# MCMC helpers
# ---------------------------------------------------------------------------

def load_flatchain(output_dir: str):
    """Return ``(flatchain, param_names)`` or ``(None, None)`` if not found."""
    path = os.path.join(output_dir, "flatchain.npz")
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return d["flatchain"], [str(n) for n in d["param_names"]]


def mcmc_bands(predict_fn, base_params: dict, flatchain, param_names, n_sub: int = 300):
    """Compute 68%/95% prediction bands from a random subsample of the flatchain.

    Parameters
    ----------
    predict_fn : callable ``(params_dict) -> array``
    base_params : full parameter dict (free + fixed); free params are overridden
    flatchain   : shape (N, n_free)
    param_names : list of free parameter names matching columns in flatchain
    n_sub       : number of random samples to draw

    Returns
    -------
    dict with keys ``lo95``, ``lo68``, ``med``, ``hi68``, ``hi95`` (1-D arrays),
    or ``None`` if fewer than 10 predictions succeed.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(flatchain), size=min(n_sub, len(flatchain)), replace=False)
    preds = []
    for i in idx:
        p = dict(base_params)
        for name, val in zip(param_names, flatchain[i]):
            p[name] = float(val)
        try:
            preds.append(np.asarray(predict_fn(p), dtype=float))
        except Exception:
            pass
    if len(preds) < 10:
        return None
    preds = np.array(preds)
    return {
        "lo95": np.percentile(preds,  2.5, axis=0),
        "lo68": np.percentile(preds, 16.0, axis=0),
        "med":  np.percentile(preds, 50.0, axis=0),
        "hi68": np.percentile(preds, 84.0, axis=0),
        "hi95": np.percentile(preds, 97.5, axis=0),
    }


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------

def add_bands(ax, x, bands: dict | None, color, alpha95=0.12, alpha68=0.25,
              scale=None):
    """Fill 95% (light) and 68% (darker) MCMC credible bands.

    The posterior median is drawn as a dashed line (visually distinct from
    the solid MAP line).  The caller is responsible for adding a legend entry
    to avoid duplicates.

    Parameters
    ----------
    scale : optional 1-D array multiplied into every band value
            (e.g. ``rp`` for an ``rp × wp`` panel).
    """
    if bands is None:
        return
    s = scale if scale is not None else np.ones(len(np.asarray(x)))
    ax.fill_between(x, s * bands["lo95"], s * bands["hi95"], color=color, alpha=alpha95)
    ax.fill_between(x, s * bands["lo68"], s * bands["hi68"], color=color, alpha=alpha68)
    ax.loglog(x, s * bands["med"], "--", color=color, lw=1.5, alpha=0.85)


def residual_panel(ax, x, obs, pred, err, pub=None, bands=None, fmt="o",
                   color=None, ylabel=True):
    """Populate a ``data/model − 1`` residual panel with consistent style.

    Parameters
    ----------
    color  : line/marker colour (defaults to ``_COL_MAP``).
    ylabel : if False, suppress the y-axis label (useful for non-leftmost
             panels in multi-column layouts).
    """
    c = color or _COL_MAP
    ratio     = obs / pred - 1
    ratio_err = err / pred
    ax.axhline(0,    color=_COL_DATA, lw=0.8, ls="--")
    ax.axhline( 0.1, color="gray", lw=0.5, ls=":")
    ax.axhline(-0.1, color="gray", lw=0.5, ls=":")
    ax.errorbar(x, ratio, yerr=ratio_err, fmt=fmt, ms=4, color=c)
    if pub is not None:
        ax.plot(x, obs / pub - 1, "--", color=_COL_PUB, lw=1.5)
    if bands is not None:
        ax.fill_between(x,
                        obs / bands["hi68"] - 1,
                        obs / bands["lo68"] - 1,
                        color=c, alpha=0.2)
    if ylabel:
        ax.set_ylabel("data/model − 1")
    ax.set_ylim(-0.6, 0.6)


# ---------------------------------------------------------------------------
# HOD figure
# ---------------------------------------------------------------------------

def plot_hod(fitter, params: dict, pub_params: dict | None,
             model_key: str, output_dir: str,
             flatchain=None, param_names: list[str] | None = None):
    """Save ``N_c``, ``N_s``, ``N_total`` vs halo mass.

    If *flatchain* and *param_names* are provided, the 16th/84th percentile
    posterior bands are drawn behind the MAP lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return
    try:
        hod    = fitter.predictor._hod
        m_np   = np.asarray(hod._m_grid, dtype=float)
        log10m = jnp.asarray(np.log10(m_np))
        with jax.disable_jit():
            nc_m, ns_m = [np.asarray(x) for x in hod.nc_ns(log10m, params)]
    except Exception:
        return
    nt_m = nc_m + ns_m

    fig, ax = plt.subplots(figsize=(6, 5))

    # MCMC posterior bands (16th / median / 84th percentile)
    if flatchain is not None and param_names is not None:
        fixed = fitter._fixed_params
        step  = max(1, len(flatchain) // 300)
        nc_samples, ns_samples = [], []
        with jax.disable_jit():
            for row in flatchain[::step]:
                hp = {**fixed, **dict(zip(param_names, row))}
                nc_i, ns_i = hod.nc_ns(log10m, hp)
                nc_samples.append(np.asarray(nc_i))
                ns_samples.append(np.asarray(ns_i))
        nc_arr = np.array(nc_samples)   # (n_samples, n_mass)
        ns_arr = np.array(ns_samples)
        nt_arr = nc_arr + ns_arr
        for arr, ls, label_base in [
            (nc_arr, "-",  r"$N_c$"),
            (ns_arr, "--", r"$N_s$"),
            (nt_arr, ":",  r"$N$"),
        ]:
            med = np.median(arr, axis=0)
            lo  = np.percentile(arr, 16, axis=0)
            hi  = np.percentile(arr, 84, axis=0)
            # clip zeros for log scale
            med = np.clip(med, 1e-4, None)
            lo  = np.clip(lo,  1e-4, None)
            hi  = np.clip(hi,  1e-4, None)
            ax.fill_between(m_np, lo, hi, color=_COL_MAP, alpha=0.18, lw=0)
            ax.loglog(m_np, med, ls, color=_COL_MAP, lw=1.4, alpha=0.7,
                      label=rf"$\langle {label_base[1:-1]} \rangle$ MCMC median")

    # MAP lines on top
    ax.loglog(m_np, np.clip(nc_m, 1e-4, None), "-",  color=_COL_MAP, lw=2,
              label=r"$\langle N_c \rangle$ MAP")
    ax.loglog(m_np, np.clip(ns_m, 1e-4, None), "--", color=_COL_MAP, lw=2,
              label=r"$\langle N_s \rangle$ MAP")
    ax.loglog(m_np, np.clip(nt_m, 1e-4, None), ":",  color=_COL_MAP, lw=2.5,
              label=r"$\langle N \rangle$ MAP")

    if pub_params:
        try:
            with jax.disable_jit():
                nc_p, ns_p = [np.asarray(x) for x in hod.nc_ns(log10m, pub_params)]
            nt_p = nc_p + ns_p
            ax.loglog(m_np, np.clip(nc_p, 1e-4, None), "-",  color=_COL_PUB, lw=1.5,
                      alpha=0.8, label=r"$\langle N_c \rangle$ published")
            ax.loglog(m_np, np.clip(ns_p, 1e-4, None), "--", color=_COL_PUB, lw=1.5,
                      alpha=0.8, label=r"$\langle N_s \rangle$ published")
            ax.loglog(m_np, np.clip(nt_p, 1e-4, None), ":",  color=_COL_PUB, lw=2.0,
                      alpha=0.8, label=r"$\langle N \rangle$ published")
        except Exception:
            pass

    ax.set_xlabel(r"$M_h$ [$h^{-1}M_\odot$]")
    ax.set_ylabel(r"$\langle N \rangle_M$")
    ax.set_xlim(1e11, 1e16)
    ax.set_ylim(5e-3, 50)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.set_title(f"HOD: {model_key}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"benchmark_{model_key}_hod.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: benchmark_{model_key}_hod.png")


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

def plot_corner(flatchain, param_names: list[str], published: dict,
                model_key: str, output_dir: str,
                normalize_fn=None, fixed_params: dict | None = None):
    """Save MCMC posterior corner plot.

    Tries getdist first (filled contours, your preferred style), then the
    ``corner`` package, then a plain matplotlib fallback.

    Parameters
    ----------
    normalize_fn : optional callable ``(dict) -> dict`` converting plain
                   floats to ``(value, error)`` tuples.
    fixed_params : optional dict of ``{name: value}`` for parameters that
                   were held fixed during sampling.  Shown as a text annotation
                   so the reader knows what was assumed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if normalize_fn is not None:
        published = normalize_fn(published)
    else:
        published = {k: v if isinstance(v, tuple) else (v, 0.0)
                     for k, v in published.items()}

    # plain-text labels for getdist (no surrounding $)
    names_gd  = param_names
    labels_gd = [_PARAM_LATEX.get(p, p).strip("$") for p in param_names]
    # full LaTeX labels for corner / matplotlib
    labels_lt = [_PARAM_LATEX.get(p, p) for p in param_names]
    truths    = [published[p][0] if p in published else None for p in param_names]

    # Build a human-readable fixed-param string for the annotation
    def _fixed_text(fp: dict | None) -> str:
        if not fp:
            return ""
        parts = [f"{_PARAM_LATEX.get(k, k)}={v:.3g}" for k, v in fp.items()]
        return "Fixed:  " + ",  ".join(parts)

    fixed_text = _fixed_text(fixed_params)

    # ── 1. getdist (preferred) ────────────────────────────────────────────────
    try:
        from getdist import MCSamples, plots as gd_plots

        ranges = {p: (float(flatchain[:, i].min()), float(flatchain[:, i].max()))
                  for i, p in enumerate(param_names)}
        samps = MCSamples(
            samples=flatchain,
            names=names_gd,
            labels=labels_gd,
            ranges=ranges,
            label="MCMC posterior",
        )
        g = gd_plots.get_subplot_plotter(subplot_size=2.2)
        g.triangle_plot(
            [samps],
            filled=True,
            contour_colors=[_COL_MAP],
            contour_lws=[2, 2, 2],
            title_limit=1,
        )
        # Mark published/truth values
        axes_gd = g.subplots
        for i, pname_i in enumerate(param_names):
            for j, pname_j in enumerate(param_names):
                if j > i or axes_gd[i, j] is None:
                    continue
                ax = axes_gd[i, j]
                if truths[j] is not None:
                    ax.axvline(truths[j], color=_COL_PUB, lw=1.5, ls="--")
                if i != j and truths[i] is not None:
                    ax.axhline(truths[i], color=_COL_PUB, lw=1.5, ls="--")
        fig = g.fig
        title = f"Posterior: {model_key}"
        if fixed_text:
            title += f"\n{fixed_text}"
        fig.suptitle(title, fontsize=10, y=1.01)

    # ── 2. corner package ─────────────────────────────────────────────────────
    except ImportError:
        try:
            import corner as corner_pkg
            fig = corner_pkg.corner(
                flatchain, labels=labels_lt, truths=truths,
                show_titles=True,
                title_kwargs={"fontsize": 9},
                label_kwargs={"fontsize": 10},
                truth_color=_COL_PUB,
                quantiles=[0.16, 0.5, 0.84],
                title_fmt=".3f",
                color=_COL_MAP,
            )
            title = f"Posterior: {model_key}"
            if fixed_text:
                title += f"\n{fixed_text}"
            fig.suptitle(title, fontsize=10, y=1.01)

        # ── 3. plain matplotlib fallback ──────────────────────────────────────
        except ImportError:
            n   = len(param_names)
            fig = plt.figure(figsize=(2.2 * n + 0.5, 2.2 * n + 0.5))
            axes = fig.subplots(n, n, sharex="col")
            if n == 1:
                axes = np.array([[axes]])
            for i in range(n):
                for j in range(n):
                    ax = axes[i, j]
                    if j > i:
                        ax.set_visible(False)
                        continue
                    if i == j:
                        ax.hist(flatchain[:, i], bins=50, density=True,
                                color=_COL_MAP, alpha=0.7)
                        if truths[i] is not None:
                            ax.axvline(truths[i], color=_COL_PUB, lw=1.5, ls="--")
                        q16, q50, q84 = np.percentile(flatchain[:, i], [16, 50, 84])
                        ax.set_title(
                            f"{labels_lt[i]}\n"
                            f"{q50:.3f}$^{{+{q84-q50:.3f}}}_{{-{q50-q16:.3f}}}$",
                            fontsize=8,
                        )
                        ax.tick_params(labelleft=False)
                    else:
                        ax.scatter(flatchain[::50, j], flatchain[::50, i],
                                   s=0.5, alpha=0.3, c=_COL_MAP, rasterized=True)
                        if truths[j] is not None:
                            ax.axvline(truths[j], color=_COL_PUB, lw=0.8, ls="--")
                        if truths[i] is not None:
                            ax.axhline(truths[i], color=_COL_PUB, lw=0.8, ls="--")
                        if j > 0:
                            ax.tick_params(labelleft=False)
                        else:
                            ax.set_ylabel(labels_lt[i], fontsize=10)
                    if i == n - 1:
                        ax.set_xlabel(labels_lt[j], fontsize=10)
                        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
                    else:
                        ax.tick_params(labelbottom=False)
            title = f"Posterior: {model_key}"
            if fixed_text:
                title += f"\n{fixed_text}"
            fig.suptitle(title, fontsize=10)
            fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig.savefig(os.path.join(output_dir, f"benchmark_{model_key}_corner.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: benchmark_{model_key}_corner.png")
