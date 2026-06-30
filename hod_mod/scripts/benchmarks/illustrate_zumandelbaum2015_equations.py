#!/usr/bin/env python
r"""Pedagogical illustration of the Zu & Mandelbaum (2015) iHOD equations.

This script walks through the chain of equations documented on the
``docs/_build/html/hod_zumandelbaum2015.html`` page and, for each one, produces a
figure that *shows* how the equation behaves and how it responds to its
parameters.  For the cosmology-dependent observables it also overlays the effect
of a **+2 % change in** :math:`\Omega_m` **and** :math:`\sigma_8`.

The 11 figures map one-to-one onto the equations on the docs page:

    01  SHMR forward direction         (ZM15 Eq. 19)  -- log10 M_h(M*)
    02  SHMR inversion                 (Step 2)       -- M*^c(M_h)
    03  Mass-dependent scatter         (ZM15 Eq. 20)  -- sigma_lnM*(M_h)
    04  Central occupation             (ZM15 Eq. 21)  -- <N_cen>(M_h)
    05  Satellite mass scales          (Step 5)       -- M_sat, M_cut(M_min)
    06  Satellite occupation           (ZM15 Eq. 22)  -- <N_sat>, <N_tot>(M_h)
    07  Stellar-mass-bin HOD           (Step 7)       -- threshold difference
    08  Effective bias                 (b_eff)        -- HOD-weighted bias integrand
    09  1-halo / 2-halo power spectra  (Step 8)       -- P_gg, P_gm
    10  Projected correlation function (w_p)          -- w_p(r_p)
    11  Excess surface mass density    (Delta Sigma)  -- Delta Sigma(R)
    12  gNFW gas shape function        (DPM Eq. 1)    -- f(x|alpha)
    13  Gas density/pressure/temp.     (DPM)          -- n_e(r), T(r)
    14  Metallicity + APEC cooling     (X-ray)        -- Z(r), Lambda(T,Z)
    15  X-ray emissivity + FT          (X-ray)        -- eps(r), X(k|M)
    16  Hard X-ray luminosity function (Aird+2015)    -- Phi(L_X,z)
    17  AGN obscuration + duty cycle   (Comparat+19)  -- f_obsc, f_CT, f_DC(z)
    18  AGN HOD occupation             (More+2015)    -- Nc/Ns^AGN(M)
    19  eROSITA King PSF + beam        (instrument)   -- PSF(theta), B_ell
    20  Emissivity FT cosmology sens.  (X-ray)        -- dlnX(k|M)/dln p

Figures 01-07 are *cosmology-independent* (the iHOD occupation depends only on
the HOD parameters), so each varies its own HOD parameters and additionally
shows the **analytic JAX gradient** of the curve with respect to one key
parameter -- a direct demonstration of differentiability via ``jax.grad`` /
``jax.vmap``.  Figures 08-11 and 20 are cosmology-dependent: the bottom panel
gives the **logarithmic sensitivity** :math:`d\ln O/d\ln p` to :math:`\Omega_m`,
:math:`\sigma_8` and :math:`h`.  Figures 12-19 are the X-ray / hot-gas / AGN
profiles, luminosity functions and instrument response, which are
cosmology-independent (their physical parameters are varied instead).

.. note::

   The linear matter power spectrum :math:`P_\mathrm{lin}(k)` is produced by
   CAMB, which is **not** JAX-traceable.  We therefore cannot back-propagate
   through the full ``w_p`` / ``Delta Sigma`` pipeline w.r.t. cosmology.  The
   :math:`\Omega_m`, :math:`\sigma_8` and :math:`h` sensitivities are computed
   by **central finite differences** (re-evaluating the prediction at
   :math:`p(1\pm\epsilon)`), while ``jax.grad`` is used on the pure-JAX
   occupation equations (figs 01-07) where the whole computation lives inside
   JAX.  Each cosmological parameter is varied **one at a time** (the others
   held fixed) so each curve isolates a single parameter.

Usage
-----
    # Generate all PNG figures into results/benchmarks/zumandelbaum2015_equations/
    python hod_mod/scripts/benchmarks/illustrate_zumandelbaum2015_equations.py

    # Convert this script (percent-cell format) into a tutorial notebook
    python hod_mod/scripts/benchmarks/illustrate_zumandelbaum2015_equations.py --to-notebook

Output
------
    results/benchmarks/zumandelbaum2015_equations/eq01_*.png ... eq20_*.png
    notebooks/zumandelbaum2015_equations.ipynb   (with --to-notebook)
"""

# %% [markdown]
# # Zu & Mandelbaum (2015) iHOD model — equation by equation
#
# This notebook reconstructs every equation of the **inverse HOD (iHOD)** model
# of Zu & Mandelbaum (2015) and shows, for each one, how it depends on its
# parameters.  For the cosmology-dependent observables ($b_\mathrm{eff}$, the
# power spectra, $w_p$ and $\Delta\Sigma$) the bottom panel gives the
# logarithmic sensitivity $d\ln O/d\ln p$ to the three cosmological parameters
# $\Omega_m$, $\sigma_8$ and $h$ (and the top panel overlays $+2\%$ variants).
#
# Everything reuses the existing `hod_mod` machinery — no physics is
# re-implemented here.  The pure occupation equations (Eqs. 19–22) are evaluated
# directly with their JAX primitives, which lets us show their analytic
# parameter sensitivity with `jax.grad`.  The full $w_p$ / $\Delta\Sigma$
# predictions run through CAMB (not JAX-differentiable), so their cosmology
# sensitivity is computed by central finite differences.
#
# The notebook then continues to the **X-ray extension** (Comparat+2025): the
# DPM hot-gas profiles, APEC X-ray emissivity, the AGN luminosity function,
# obscuration and duty cycle, the AGN HOD, the eROSITA PSF, and the galaxy ×
# X-ray power spectrum $P_{gX}$.

# %%
# ---------------------------------------------------------------------------
# Imports and environment set-up
# ---------------------------------------------------------------------------
import os
import sys

# Detect whether we are inside a Jupyter kernel; only force the non-interactive
# Agg backend when running as a plain script (so the notebook can use inline).
_IN_NOTEBOOK = ("ipykernel" in sys.modules) or ("IPython" in sys.modules)

import matplotlib
if not _IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make the repository importable whether run as a script or inside a notebook:
# walk up from the file (or cwd) until the directory holding the package is found.
def _find_repo_root() -> str:
    start = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in globals() else os.getcwd())
    d = start
    while d != os.path.dirname(d):
        if (os.path.isdir(os.path.join(d, "hod_mod"))
                and os.path.isdir(os.path.join(d, "configs"))):
            return d
        d = os.path.dirname(d)
    return start


_REPO_ROOT = _find_repo_root()
if os.path.isdir(os.path.join(_REPO_ROOT, "hod_mod")):
    sys.path.insert(0, _REPO_ROOT)

OUT_DIR = os.path.join(results_root(), "benchmarks/zumandelbaum2015_equations")


# ---------------------------------------------------------------------------
# Early CLI hook: build the notebook from this file, then exit.
# Must run *before* the heavy figure cells so `--to-notebook` is cheap.
# ---------------------------------------------------------------------------
def _build_notebook(src_path: str, out_path: str) -> None:
    """Convert this percent-cell script into a Jupyter notebook via nbformat.

    Cells are delimited by lines beginning with ``# %%``.  A marker line
    containing ``[markdown]`` starts a markdown cell whose body is the following
    ``#``-prefixed comment lines (the leading ``# `` is stripped); any other
    ``# %%`` starts a code cell.  Text before the first marker (shebang +
    module docstring) is dropped.  A leading ``%matplotlib inline`` code cell is
    injected so figures render inline in the notebook.
    """
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    with open(src_path) as fh:
        lines = fh.read().splitlines()

    cells = []
    cur_kind = None          # "code" | "markdown" | None (preamble)
    cur_body: list[str] = []

    def _flush():
        if cur_kind == "code":
            text = "\n".join(cur_body).strip("\n")
            if text.strip():
                cells.append(new_code_cell(text))
        elif cur_kind == "markdown":
            md = []
            for ln in cur_body:
                if ln.startswith("# "):
                    md.append(ln[2:])
                elif ln == "#":
                    md.append("")
                else:
                    md.append(ln)
            text = "\n".join(md).strip("\n")
            if text.strip():
                cells.append(new_markdown_cell(text))

    for ln in lines:
        if ln.startswith("# %%"):
            _flush()
            cur_kind = "markdown" if "[markdown]" in ln else "code"
            cur_body = []
        elif cur_kind is not None:
            cur_body.append(ln)
    _flush()

    nb = new_notebook(cells=[new_code_cell("%matplotlib inline")] + cells)
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"[notebook] wrote {out_path}  ({len(nb.cells)} cells)")


if __name__ == "__main__" and "--to-notebook" in sys.argv:
    _src = os.path.abspath(__file__)
    _out = os.path.join(_REPO_ROOT, "notebooks", "zumandelbaum2015_equations.ipynb")
    _build_notebook(_src, _out)
    raise SystemExit(0)

# %%
# ---------------------------------------------------------------------------
# Heavy imports (model machinery) and global model build.
# ---------------------------------------------------------------------------
import jax
import jax.numpy as jnp

from hod_mod.core import LinearPowerSpectrum, make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import (
    ZuMandelbaum15HODModel,
    _mh_from_mstar_zu15,
    _mstar_from_mh_zu15,
    sigma_lnmstar_zu15,
    n_cen_thresh_zu15,
    n_sat_thresh_zu15,
)
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.fitting import _sigma8_to_lnAs
from hod_mod.paths import results_root

plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.fontsize": 9})

# Representative effective redshift and SDSS line-of-sight integration limit.
Z_EFF = 0.10
PI_MAX = 60.0

# ZM15 best-fit iHOD parameters (Table 2; SDSS volume-limited threshold sample).
IHOD_PARAMS = ZuMandelbaum15HODModel.default_params()

# Colour convention used across the cosmology-overlay figures.
C_FID, C_OM, C_S8 = "k", "C3", "C0"


# %%
# ---------------------------------------------------------------------------
# Cosmology helpers: fiducial ZM15 cosmology + one-at-a-time perturbations and
# logarithmic-derivative (sensitivity) machinery for Omega_m, sigma_8 and h.
# ---------------------------------------------------------------------------
# Cosmological parameters whose sensitivity we probe, with display label + colour.
COSMO_PARAMS = (
    ("Omega_m", r"$\Omega_m$", "C3"),
    ("sigma8",  r"$\sigma_8$", "C0"),
    ("h",       r"$h$",        "C2"),
)


def fiducial_cosmology() -> dict:
    """ZM15 benchmark cosmology with a CAMB-calibrated ``ln10^{10}A_s``."""
    fid = {"Omega_m": 0.260, "Omega_b": 0.044, "h": 0.720,
           "n_s": 0.960, "sigma8": 0.770}
    fid["Omega_cdm"] = fid["Omega_m"] - fid["Omega_b"]
    fid["ln10^{10}A_s"] = _sigma8_to_lnAs(fid)
    return fid


def perturb_cosmo(fid: dict, param: str, factor: float) -> dict:
    r"""Return ``fid`` with a single cosmological parameter scaled by ``factor``.

    All other parameters are held fixed.  For ``sigma8`` we also shift
    ``ln10^{10}A_s`` by the exact :math:`A_s\propto\sigma_8^2` amount so the
    amplitude change is consistent however the pipeline anchors it; for
    ``Omega_m`` we keep ``Omega_cdm = Omega_m - Omega_b`` consistent.
    """
    th = dict(fid)
    if param == "Omega_m":
        th["Omega_m"] = fid["Omega_m"] * factor
        th["Omega_cdm"] = th["Omega_m"] - th["Omega_b"]
    elif param == "sigma8":
        th["sigma8"] = fid["sigma8"] * factor
        th["ln10^{10}A_s"] = fid["ln10^{10}A_s"] + 2.0 * np.log(factor)
    elif param == "h":
        th["h"] = fid["h"] * factor
    else:
        raise ValueError(f"unknown cosmological parameter {param!r}")
    return th


def log_sensitivity(obs_fn, fid: dict, eps: float = 0.02) -> dict:
    r"""Central finite-difference logarithmic derivative :math:`d\ln O/d\ln p`.

    ``obs_fn(theta_cosmo)`` returns the (positive) observable array.  For each
    parameter in :data:`COSMO_PARAMS` we evaluate it at :math:`p(1\pm\epsilon)`
    and form :math:`[\ln O_+ - \ln O_-]/\ln\frac{1+\epsilon}{1-\epsilon}`.
    Finite differences are used because CAMB (the linear :math:`P(k)`) is not
    JAX-traceable, so the full pipeline cannot be auto-differentiated.
    """
    denom = np.log((1.0 + eps) / (1.0 - eps))
    out = {}
    for name, _, _ in COSMO_PARAMS:
        o_plus = np.asarray(obs_fn(perturb_cosmo(fid, name, 1.0 + eps)))
        o_minus = np.asarray(obs_fn(perturb_cosmo(fid, name, 1.0 - eps)))
        # Mask non-positive samples (e.g. P_gX zero-crossings) so log is finite;
        # such points appear as gaps in the sensitivity curve rather than warnings.
        good = (o_plus > 0) & (o_minus > 0)
        d = np.full(np.shape(o_plus), np.nan)
        d[good] = (np.log(o_plus[good]) - np.log(o_minus[good])) / denom
        out[name] = d
    return out


def build_context() -> dict:
    """Build the power spectrum, HMF, HOD model and full prediction once."""
    fid = fiducial_cosmology()
    pk_lin = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hod = ZuMandelbaum15HODModel(hmf)
    # HaloProfile concentration is built once (cosmology enters per-call), so a
    # single predictor serves every perturbed cosmology.
    pred = FullHaloModelPrediction(pk_lin, hod, HaloProfile(fid))
    os.makedirs(OUT_DIR, exist_ok=True)
    return {"fid": fid, "pk_lin": pk_lin, "hmf": hmf, "hod": hod, "pred": pred}


def _save(fig, stem: str) -> None:
    """Save a figure to the output directory (skipped silently in notebooks)."""
    path = os.path.join(OUT_DIR, stem + ".png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    print(f"[fig] {path}")


def _plot_sensitivity(ax, x, sens: dict, logx: bool = True) -> None:
    """Draw the d ln O / d ln p curves for Omega_m, sigma_8, h on ``ax``."""
    for name, lab, c in COSMO_PARAMS:
        ax.plot(x, sens[name], c, lw=1.9, label=lab)
    if logx:
        ax.set_xscale("log")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.legend(ncol=3, fontsize=8, loc="best")


# Build the global model context (re-used by every figure cell below).
CTX = build_context()


# %% [markdown]
# ## 1 — SHMR forward direction (ZM15 Eq. 19)
#
# The stellar-to-halo mass relation is written as halo mass *as a function of*
# stellar mass (Behroozi et al. 2010 form):
#
# $$\log_{10} M_h(M_*) = \log_{10} M_1 + \beta\,\log_{10}\frac{M_*}{M_{*,0}}
#   + \frac{1}{\ln 10}\left[\frac{(M_*/M_{*,0})^\delta}{1+(M_*/M_{*,0})^{-\gamma}}
#   - \tfrac12\right].$$
#
# The five parameters $(\log M_1,\log M_{*,0},\beta,\delta,\gamma)$ set the
# pivot masses, the low-mass slope, the high-mass slope and the transition
# sharpness.  The lower panel is $\partial\log_{10}M_h/\partial\beta$ evaluated
# analytically with `jax.grad` — a check that the whole relation is
# differentiable.

# %%
def fig01_shmr_forward(ctx):
    p = IHOD_PARAMS
    lms = jnp.linspace(8.5, 11.8, 240)                       # log10(M*/[Msun/h])

    def mh(lms_, **kw):
        q = {**dict(lg_m1h=p["lg_m1h"], lg_m0star=p["lg_m0star"],
                    beta=p["beta"], delta=p["delta"], gamma=p["gamma"]), **kw}
        return _mh_from_mstar_zu15(lms_, q["lg_m1h"], q["lg_m0star"],
                                   q["beta"], q["delta"], q["gamma"])

    fig, (ax, axg) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.4],
                                               "hspace": 0.08})
    ax.plot(lms, mh(lms), C_FID, lw=2.4, label="fiducial")
    for b in (0.25, 0.40):
        ax.plot(lms, mh(lms, beta=b), "--", lw=1.4, label=fr"$\beta={b}$")
    for g in (0.8, 1.6):
        ax.plot(lms, mh(lms, gamma=g), ":", lw=1.4, label=fr"$\gamma={g}$")
    ax.axhline(p["lg_m1h"], color="grey", lw=0.8)
    ax.axvline(p["lg_m0star"], color="grey", lw=0.8)
    ax.set_ylabel(r"$\log_{10}(M_h\,/\,[M_\odot/h])$")
    ax.set_title("Eq. 19 — SHMR forward: halo mass from stellar mass")
    ax.legend(ncol=2)

    # Analytic JAX sensitivity d log10 M_h / d beta along the relation.
    dmh_dbeta = jax.vmap(
        lambda lm: jax.grad(lambda b: mh(lm, beta=b))(p["beta"]))(lms)
    axg.plot(lms, np.asarray(dmh_dbeta), "C2", lw=2)
    axg.set_ylabel(r"$\partial\log M_h/\partial\beta$")
    axg.set_xlabel(r"$\log_{10}(M_*\,/\,[M_\odot/h])$")
    _save(fig, "eq01_shmr_forward")
    return fig


fig01_shmr_forward(CTX)


# %% [markdown]
# ## 2 — SHMR inversion: mean stellar mass at fixed halo mass (Step 2)
#
# The iHOD needs $M_*^c(M_h)$ — the *inverse* of Eq. 19.  Eq. 19 has no
# closed form, so it is inverted with 60 JAX-compatible bisection iterations.
# We plot the inverse and confirm the round-trip
# $f(f^{-1}(M_h))=M_h$ to machine precision.

# %%
def fig02_shmr_inverse(ctx):
    p = IHOD_PARAMS
    lmh = jnp.linspace(10.5, 15.3, 240)
    lms_c = _mstar_from_mh_zu15(lmh, p["lg_m1h"], p["lg_m0star"],
                                p["beta"], p["delta"], p["gamma"])
    lmh_round = _mh_from_mstar_zu15(lms_c, p["lg_m1h"], p["lg_m0star"],
                                    p["beta"], p["delta"], p["gamma"])
    resid = np.asarray(lmh_round - lmh)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2],
                                               "hspace": 0.08})
    ax.plot(lmh, np.asarray(lms_c), C_FID, lw=2.4)
    ax.axvline(p["lg_m1h"], color="grey", lw=0.8, label=r"$M_1$")
    ax.set_ylabel(r"$\log_{10}(M_*^c\,/\,[M_\odot/h])$")
    ax.set_title(r"Step 2 — SHMR inversion: $M_*^c(M_h)=\mathrm{SHMR}^{-1}(M_h)$")
    ax.legend()

    axr.plot(lmh, resid, "C2", lw=1.6)
    axr.set_ylabel(r"round-trip $\Delta\log M_h$")
    axr.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot/h])$")
    axr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _save(fig, "eq02_shmr_inverse")
    return fig


fig02_shmr_inverse(CTX)


# %% [markdown]
# ## 3 — Mass-dependent scatter in $\ln M_*$ (ZM15 Eq. 20)
#
# $$\sigma_{\ln M_*}(M_h)=\begin{cases}\sigma_{\ln m_*} & M_h\le M_1\\
#   \sigma_{\ln m_*}+\eta\,\log_{10}(M_h/M_1) & M_h>M_1.\end{cases}$$
#
# The scatter is constant below the pivot $M_1$ and tilts with slope $\eta$
# above it ($\eta<0$ ⇒ tighter SHMR in clusters).  Lower panel:
# $\partial\sigma/\partial\eta$ via `jax.grad` (zero below $M_1$, growing above).

# %%
def fig03_scatter(ctx):
    p = IHOD_PARAMS
    lmh = jnp.linspace(10.5, 15.3, 240)

    def sig(lmh_=lmh, s0=p["sigma_lnmstar"], eta=p["eta"]):
        return sigma_lnmstar_zu15(lmh_, p["lg_m1h"], s0, eta)

    fig, (ax, axg) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.4],
                                               "hspace": 0.08})
    ax.plot(lmh, np.asarray(sig()), C_FID, lw=2.4, label="fiducial")
    for s0 in (0.35, 0.65):
        ax.plot(lmh, np.asarray(sig(s0=s0)), "--", lw=1.4,
                label=fr"$\sigma_0={s0}$")
    for eta in (-0.2, 0.1):
        ax.plot(lmh, np.asarray(sig(eta=eta)), ":", lw=1.4,
                label=fr"$\eta={eta}$")
    ax.axvline(p["lg_m1h"], color="grey", lw=0.8, label=r"$M_1$")
    ax.set_ylabel(r"$\sigma_{\ln M_*}(M_h)$")
    ax.set_title("Eq. 20 — mass-dependent log-normal scatter")
    ax.legend(ncol=2)

    dsig_deta = jax.vmap(
        lambda lm: jax.grad(lambda e: sig(lm, eta=e))(p["eta"]))(lmh)
    axg.plot(lmh, np.asarray(dsig_deta), "C2", lw=2)
    axg.set_ylabel(r"$\partial\sigma/\partial\eta$")
    axg.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot/h])$")
    _save(fig, "eq03_scatter")
    return fig


fig03_scatter(CTX)


# %% [markdown]
# ## 4 — Central occupation (ZM15 Eq. 21)
#
# $$\langle N_\mathrm{cen}\rangle(M_h)=\frac{f_c}{2}\,
#   \mathrm{erfc}\!\left[\frac{\ln M_{*,\mathrm{th}}-\ln M_*^c(M_h)}
#   {\sqrt2\,\sigma_{\ln M_*}(M_h)}\right].$$
#
# Centrals appear above $M_\mathrm{min}=\mathrm{SHMR}^{-1}(M_{*,\mathrm{th}})$,
# where the erfc argument vanishes and $\langle N_\mathrm{cen}\rangle=f_c/2$.
# We sweep the threshold $M_{*,\mathrm{th}}$ and the completeness $f_c$.

# %%
def fig04_central(ctx):
    p = IHOD_PARAMS
    lmh = jnp.linspace(10.5, 15.3, 240)

    def ncen(lmh_=lmh, thr=p["log10m_star_thresh"], fc=p["fc"]):
        return n_cen_thresh_zu15(lmh_, thr, p["lg_m1h"], p["lg_m0star"],
                                 p["beta"], p["delta"], p["gamma"],
                                 p["sigma_lnmstar"], p["eta"], fc)

    fig, (ax, axg) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.4],
                                               "hspace": 0.08})
    for thr, ls in [(9.8, "--"), (10.2, "-"), (10.8, "--")]:
        lw = 2.4 if thr == 10.2 else 1.4
        lmmin = float(_mh_from_mstar_zu15(jnp.array(thr), p["lg_m1h"],
                      p["lg_m0star"], p["beta"], p["delta"], p["gamma"]))
        ax.plot(lmh, np.asarray(ncen(thr=thr)), ls, lw=lw, color=C_FID if thr == 10.2 else None,
                label=fr"$\log M_{{*,\rm th}}={thr}$")
        ax.plot(lmmin, p["fc"] / 2, "o", ms=5, color="grey")
    ax.plot(lmh, np.asarray(ncen(fc=1.0)), ":", lw=1.4, color="C4", label=r"$f_c=1$")
    ax.axhline(p["fc"] / 2, color="grey", lw=0.8)
    ax.set_ylabel(r"$\langle N_\mathrm{cen}\rangle$")
    ax.set_title("Eq. 21 — central occupation (markers: $f_c/2$ at $M_\\mathrm{min}$)")
    ax.legend()

    dnc_dthr = jax.vmap(
        lambda lm: jax.grad(lambda t: ncen(lm, thr=t))(p["log10m_star_thresh"]))(lmh)
    axg.plot(lmh, np.asarray(dnc_dthr), "C2", lw=2)
    axg.set_ylabel(r"$\partial N_c/\partial\log M_{*,\rm th}$")
    axg.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot/h])$")
    _save(fig, "eq04_central")
    return fig


fig04_central(CTX)


# %% [markdown]
# ## 5 — Satellite mass scales (Step 5)
#
# $$M_\mathrm{sat}=B_\mathrm{sat}\Big(\tfrac{M_\mathrm{min}}{10^{12}}\Big)^{\beta_\mathrm{sat}}10^{12},
#   \qquad
#   M_\mathrm{cut}=B_\mathrm{cut}\Big(\tfrac{M_\mathrm{min}}{10^{12}}\Big)^{\beta_\mathrm{cut}}10^{12},$$
#
# with $M_\mathrm{min}=\mathrm{SHMR}^{-1}(M_{*,\mathrm{th}})$ (Eq. 19, forward).
# These two scales set where satellites turn on ($M_\mathrm{cut}$) and the mass
# per satellite ($M_\mathrm{sat}$).  We show both as a function of the threshold.

# %%
def fig05_satellite_scales(ctx):
    p = IHOD_PARAMS
    thr = np.linspace(9.5, 11.5, 120)
    lmmin = np.asarray(_mh_from_mstar_zu15(jnp.array(thr), p["lg_m1h"],
                       p["lg_m0star"], p["beta"], p["delta"], p["gamma"]))
    mnorm = 10.0 ** (lmmin - 12.0)

    def msat(bsat, beta_sat):
        return np.log10(bsat * mnorm ** beta_sat * 1e12)

    def mcut(bcut, beta_cut):
        return np.log10(bcut * mnorm ** beta_cut * 1e12)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(thr, lmmin, "k:", lw=1.6, label=r"$M_\mathrm{min}$")
    ax.plot(thr, msat(p["bsat"], p["beta_sat"]), C_S8, lw=2.4,
            label=r"$M_\mathrm{sat}$ (fid)")
    ax.plot(thr, msat(p["bsat"] * 1.5, p["beta_sat"]), C_S8, ls="--", lw=1.3,
            label=r"$1.5\,B_\mathrm{sat}$")
    ax.plot(thr, mcut(p["bcut"], p["beta_cut"]), C_OM, lw=2.4,
            label=r"$M_\mathrm{cut}$ (fid)")
    ax.plot(thr, mcut(p["bcut"], p["beta_cut"] * 1.8), C_OM, ls="--", lw=1.3,
            label=r"$1.8\,\beta_\mathrm{cut}$")
    ax.set_xlabel(r"$\log_{10}(M_{*,\mathrm{th}}\,/\,[M_\odot/h])$")
    ax.set_ylabel(r"$\log_{10}(M\,/\,[M_\odot/h])$")
    ax.set_title("Step 5 — satellite mass scales vs. threshold")
    ax.legend(ncol=2)
    _save(fig, "eq05_satellite_scales")
    return fig


fig05_satellite_scales(CTX)


# %% [markdown]
# ## 6 — Satellite occupation and total HOD (ZM15 Eq. 22)
#
# $$\langle N_\mathrm{sat}\rangle(M_h)=\langle N_\mathrm{cen}\rangle(M_h)\,
#   \Big(\tfrac{M_h}{M_\mathrm{sat}}\Big)^{\alpha_\mathrm{sat}}
#   \exp\!\Big(-\tfrac{M_\mathrm{cut}}{M_h}\Big).$$
#
# Satellites inherit the central prefactor (so they vanish in halos too light to
# host a central), rise as a power law of slope $\alpha_\mathrm{sat}$, and are
# truncated below $M_\mathrm{cut}$.  We plot $N_\mathrm{cen}$, $N_\mathrm{sat}$
# and the total $N_\mathrm{tot}$, sweeping $\alpha_\mathrm{sat}$.

# %%
def fig06_satellite(ctx):
    p = IHOD_PARAMS
    lmh = jnp.linspace(10.5, 15.3, 240)
    base = dict(lg_m1h=p["lg_m1h"], lg_m0star=p["lg_m0star"], beta=p["beta"],
                delta=p["delta"], gamma=p["gamma"],
                sigma_lnmstar=p["sigma_lnmstar"], eta=p["eta"], fc=p["fc"])

    def nsat(lmh_=lmh, alpha=p["alpha_sat"]):
        return n_sat_thresh_zu15(lmh_, p["log10m_star_thresh"], **base,
                                 bsat=p["bsat"], beta_sat=p["beta_sat"],
                                 bcut=p["bcut"], beta_cut=p["beta_cut"],
                                 alpha_sat=alpha)

    nc = n_cen_thresh_zu15(lmh, p["log10m_star_thresh"], **base)
    fig, (ax, axg) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.4],
                                               "hspace": 0.08})
    ax.plot(lmh, np.asarray(nc), "grey", lw=1.6, label=r"$N_\mathrm{cen}$")
    ax.plot(lmh, np.asarray(nsat()), C_S8, lw=2.4, label=r"$N_\mathrm{sat}$ (fid)")
    for a in (0.8, 1.3):
        ax.plot(lmh, np.asarray(nsat(alpha=a)), "--", lw=1.3,
                label=fr"$\alpha_\mathrm{{sat}}={a}$")
    ax.plot(lmh, np.asarray(nc + nsat()), C_FID, lw=2.0, label=r"$N_\mathrm{tot}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 5e1)
    ax.set_ylabel(r"$\langle N\rangle$")
    ax.set_title("Eq. 22 — satellite occupation and total HOD")
    ax.legend(ncol=2)

    dns_dalpha = jax.vmap(
        lambda lm: jax.grad(lambda a: nsat(lm, alpha=a))(p["alpha_sat"]))(lmh)
    axg.plot(lmh, np.asarray(dns_dalpha), "C2", lw=2)
    axg.set_ylabel(r"$\partial N_\mathrm{sat}/\partial\alpha$")
    axg.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot/h])$")
    _save(fig, "eq06_satellite")
    return fig


fig06_satellite(CTX)


# %% [markdown]
# ## 7 — Stellar-mass-bin HOD (Step 7)
#
# For a stellar-mass *bin* $[M_{*,\mathrm{lo}},M_{*,\mathrm{hi}})$ the HOD is the
# difference of two threshold HODs:
#
# $$\langle N(M_h\,|\,\mathrm{lo}\le M_*<\mathrm{hi})\rangle =
#   \langle N(M_h\,|\,M_{*,\mathrm{th}}=\mathrm{lo})\rangle -
#   \langle N(M_h\,|\,M_{*,\mathrm{th}}=\mathrm{hi})\rangle.$$
#
# Activated by passing `log10m_star_max` to `nc_ns`.

# %%
def fig07_bin_hod(ctx):
    hod = ctx["hod"]
    lmh = jnp.linspace(10.5, 15.3, 240)
    lo, hi = 10.2, 10.8

    p_lo = {**IHOD_PARAMS, "log10m_star_thresh": lo, "log10m_star_max": None}
    p_hi = {**IHOD_PARAMS, "log10m_star_thresh": hi, "log10m_star_max": None}
    p_bin = {**IHOD_PARAMS, "log10m_star_thresh": lo, "log10m_star_max": hi}

    nl = sum(np.asarray(x) for x in hod.nc_ns(lmh, p_lo))
    nh = sum(np.asarray(x) for x in hod.nc_ns(lmh, p_hi))
    nb_c, nb_s = hod.nc_ns(lmh, p_bin)
    nb = np.asarray(nb_c) + np.asarray(nb_s)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(lmh, nl, C_FID, lw=2.0, label=fr"$N(>{lo})$")
    ax.plot(lmh, nh, "grey", lw=2.0, ls="--", label=fr"$N(>{hi})$")
    ax.plot(lmh, nb, C_S8, lw=2.6, label=fr"bin $[{lo},{hi})$ = difference")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 5e1)
    ax.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot/h])$")
    ax.set_ylabel(r"$\langle N_\mathrm{tot}\rangle$")
    ax.set_title("Step 7 — stellar-mass-bin HOD by threshold subtraction")
    ax.legend()
    _save(fig, "eq07_bin_hod")
    return fig


fig07_bin_hod(CTX)


# %% [markdown]
# ## 8 — Effective bias (HOD-weighted)
#
# $$b_\mathrm{eff}(z)=\frac{\int\!\mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}\,
#   \langle N_\mathrm{tot}(M)\rangle\,b(M,z)}{\bar n_g}.$$
#
# This is the first *cosmology-dependent* quantity: both the halo mass function
# $\mathrm{d}n/\mathrm{d}M$ and the linear bias $b(M,z)$ shift with cosmology.
# The top panel shows the bias integrand for the fiducial cosmology and three
# $+2\%$ variants (quoting $b_\mathrm{eff}$); the **bottom panel** is the
# logarithmic sensitivity $d\ln(\mathrm{integrand})/d\ln p$ for
# $p\in\{\Omega_m,\sigma_8,h\}$ (central finite differences).

# %%
def fig08_effective_bias(ctx):
    hod, hmf, pred, fid = ctx["hod"], ctx["hmf"], ctx["pred"], ctx["fid"]
    lmh = np.linspace(11.5, 15.2, 200)
    m = 10.0 ** lmh
    nc, ns = hod.nc_ns(jnp.array(lmh), IHOD_PARAMS)
    ntot = np.asarray(nc) + np.asarray(ns)

    # Bias integrand dn/dM * N_tot * b(M), memoised per distinct cosmology so the
    # +2% overlays and the central-difference sensitivity reuse each CAMB call.
    _cache: dict = {}

    def integrand(th):
        key = (round(th["Omega_m"], 6), round(th["sigma8"], 6), round(th["h"], 6))
        if key not in _cache:
            with jax.disable_jit():
                dndm = np.asarray(hmf.dndm(m, Z_EFF, th))
                bias = np.asarray(hmf.bias(m, Z_EFF, th))
            _cache[key] = dndm * ntot * bias
        return _cache[key]

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.8, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.5],
                                               "hspace": 0.08})
    b_fid = float(pred._pk_tables_full(Z_EFF, fid, IHOD_PARAMS)["b_eff"])
    ax.plot(lmh, integrand(fid) * m, "k", lw=2.4,
            label=fr"fiducial: $b_\mathrm{{eff}}={b_fid:.3f}$")
    for name, lab, c in COSMO_PARAMS:
        th = perturb_cosmo(fid, name, 1.02)
        b = float(pred._pk_tables_full(Z_EFF, th, IHOD_PARAMS)["b_eff"])
        ax.plot(lmh, integrand(th) * m, c, lw=1.4, ls="--",
                label=fr"$+2\%\,${lab}: $b_\mathrm{{eff}}={b:.3f}$")
    ax.set_ylabel(r"$M\,\frac{dn}{dM}\,N_\mathrm{tot}\,b(M)$  (bias integrand)")
    ax.set_title("Effective bias — HOD-weighted integrand over halo mass")
    ax.legend()

    sens = log_sensitivity(integrand, fid)
    _plot_sensitivity(axr, lmh, sens, logx=False)
    axr.set_ylabel(r"$d\ln(\mathrm{integrand})/d\ln p$")
    axr.set_xlabel(r"$\log_{10}(M_h\,/\,[M_\odot/h])$")
    _save(fig, "eq08_effective_bias")
    return fig


fig08_effective_bias(CTX)


# %% [markdown]
# ## 9 — 1-halo / 2-halo power spectra (Step 8)
#
# $$P_{gg}^{1h}=\frac1{\bar n_g^2}\!\int\!\mathrm{d}M\frac{\mathrm{d}n}{\mathrm{d}M}
#   [N_s^2\tilde u^2+2N_cN_s\tilde u],\quad
#   P_{gg}^{2h}=b_\mathrm{eff}^2P_\mathrm{lin},$$
# $$P_{gm}^{1h}=\frac1{\bar n_g}\!\int\!\mathrm{d}M\frac{\mathrm{d}n}{\mathrm{d}M}
#   [N_c+N_s\tilde u]\tfrac{M}{\bar\rho_m}\tilde u,\quad
#   P_{gm}^{2h}=b_\mathrm{eff}P_\mathrm{lin}.$$
#
# The 1-halo term dominates at high $k$ (small scales), the 2-halo term at low
# $k$.  Cosmology enters through $P_\mathrm{lin}$, $b_\mathrm{eff}$ and
# $\mathrm{d}n/\mathrm{d}M$; the **bottom panel** is the logarithmic sensitivity
# $d\ln P_{gg}/d\ln p$ for $p\in\{\Omega_m,\sigma_8,h\}$. A pure amplitude
# parameter such as $\sigma_8$ gives a roughly flat $d\ln P/d\ln\sigma_8\approx2$
# ($P\propto\sigma_8^2$), while $\Omega_m$ and $h$ are scale-dependent.

# %%
def fig09_power_spectra(ctx):
    pred, fid = ctx["pred"], ctx["fid"]

    def pgg(th):
        return np.exp(np.asarray(
            pred._pk_tables_full(Z_EFF, th, IHOD_PARAMS)["log_pgg"]))

    t = pred._pk_tables_full(Z_EFF, fid, IHOD_PARAMS)
    k = np.exp(np.asarray(t["log_k"]))

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.8, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.5],
                                               "hspace": 0.08})
    ax.loglog(k, np.exp(np.asarray(t["log_pgg"])), "k", lw=2.4, label=r"$P_{gg}$ total")
    ax.loglog(k, np.exp(np.asarray(t["log_pgg_1h"])), "k", lw=1.1, ls="--", label=r"$P_{gg}^{1h}$")
    ax.loglog(k, np.exp(np.asarray(t["log_pgg_2h"])), "k", lw=1.1, ls=":", label=r"$P_{gg}^{2h}$")
    ax.loglog(k, np.exp(np.asarray(t["log_pgm"])), "C5", lw=1.8, label=r"$P_{gm}$ total")
    ax.set_ylabel(r"$P(k)\;[(\mathrm{Mpc}/h)^3]$")
    ax.set_ylim(1e0, 1e5)
    ax.set_title("Step 8 — galaxy auto / galaxy–matter power spectra")
    ax.legend(ncol=2)

    sens = log_sensitivity(pgg, fid)
    _plot_sensitivity(axr, k, sens, logx=True)
    axr.set_ylabel(r"$d\ln P_{gg}/d\ln p$")
    axr.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    _save(fig, "eq09_power_spectra")
    return fig


fig09_power_spectra(CTX)


# %% [markdown]
# ## 10 — Projected correlation function $w_p(r_p)$
#
# $$w_p(r_p)=2\int_0^{\pi_\mathrm{max}}\xi_{gg}\!\big(\sqrt{r_p^2+\pi^2}\big)\,
#   \mathrm{d}\pi,\qquad \pi_\mathrm{max}=60\,h^{-1}\mathrm{Mpc}.$$
#
# $\xi_{gg}$ comes from the Ogata (2005) $j_0$ Hankel transform of $P_{gg}$.
# The top panel overlays the fiducial $w_p$ with three $+2\%$ cosmology
# variants; the **bottom panel** is the logarithmic sensitivity
# $d\ln w_p/d\ln p$ for $p\in\{\Omega_m,\sigma_8,h\}$ (central finite
# differences) — note its distinct scale dependence per parameter.

# %%
def fig10_wp(ctx):
    pred, fid = ctx["pred"], ctx["fid"]
    rp = jnp.logspace(-1.3, 1.5, 22)
    rpn = np.asarray(rp)

    def wpf(th):
        return np.asarray(pred.wp(rp, PI_MAX, Z_EFF, th, IHOD_PARAMS))

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.6, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.5],
                                               "hspace": 0.08})
    ax.loglog(rpn, wpf(fid), "k", lw=2.4, label="fiducial")
    for name, lab, c in COSMO_PARAMS:
        ax.loglog(rpn, wpf(perturb_cosmo(fid, name, 1.02)), c, lw=1.3, ls="--",
                  label=fr"$+2\%\,${lab}")
    ax.set_ylabel(r"$w_p(r_p)\;[\mathrm{Mpc}/h]$")
    ax.set_title(r"$w_p(r_p)$ — projected correlation function")
    ax.legend()

    sens = log_sensitivity(wpf, fid)
    _plot_sensitivity(axr, rpn, sens, logx=True)
    axr.set_ylabel(r"$d\ln w_p/d\ln p$")
    axr.set_xlabel(r"$r_p\;[\mathrm{Mpc}/h]$")
    _save(fig, "eq10_wp")
    return fig


fig10_wp(CTX)


# %% [markdown]
# ## 11 — Excess surface mass density $\Delta\Sigma(R)$
#
# $$\Delta\Sigma(R)=\frac{2}{R^2}\int_0^R R'\,\Sigma_{gm}(R')\,\mathrm{d}R'
#   -\Sigma_{gm}(R)\quad[M_\odot\,h\,\mathrm{pc}^{-2}].$$
#
# The galaxy–galaxy lensing signal built from $P_{gm}$.  Because
# $\Sigma_{gm}\propto\bar\rho_m\propto\Omega_m$, $\Delta\Sigma$ is especially
# sensitive to $\Omega_m$ — visible in the **bottom panel**, the logarithmic
# sensitivity $d\ln\Delta\Sigma/d\ln p$ for $p\in\{\Omega_m,\sigma_8,h\}$
# (central finite differences).

# %%
def fig11_delta_sigma(ctx):
    pred, fid = ctx["pred"], ctx["fid"]
    R = jnp.logspace(-1.3, 1.4, 22)
    Rn = np.asarray(R)

    def dsf(th):
        return np.asarray(pred.delta_sigma(R, Z_EFF, th, IHOD_PARAMS))

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.6, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.5],
                                               "hspace": 0.08})
    ax.loglog(Rn, dsf(fid), "k", lw=2.4, label="fiducial")
    for name, lab, c in COSMO_PARAMS:
        ax.loglog(Rn, dsf(perturb_cosmo(fid, name, 1.02)), c, lw=1.3, ls="--",
                  label=fr"$+2\%\,${lab}")
    ax.set_ylabel(r"$\Delta\Sigma(R)\;[M_\odot\,h\,\mathrm{pc}^{-2}]$")
    ax.set_title(r"$\Delta\Sigma(R)$ — excess surface mass density")
    ax.legend()

    sens = log_sensitivity(dsf, fid)
    _plot_sensitivity(axr, Rn, sens, logx=True)
    axr.set_ylabel(r"$d\ln\Delta\Sigma/d\ln p$")
    axr.set_xlabel(r"$R\;[\mathrm{Mpc}/h]$")
    _save(fig, "eq11_delta_sigma")
    return fig


fig11_delta_sigma(CTX)


# %% [markdown]
# # X-ray, hot gas & AGN extension
#
# For the BGS × eROSITA analysis (Comparat et al. 2025) the iHOD is extended
# with a **hot-gas** component (DPM profiles + APEC cooling → soft X-ray
# emissivity) and an **AGN** component (X-ray luminosity function + obscuration,
# placed by abundance matching or an AGN HOD).  The figures below illustrate
# those equations.  The gas/AGN *profiles, luminosity functions and instrument
# response* are cosmology-independent (we vary their physical parameters); the
# **galaxy × X-ray power spectrum** $P_{gX}$ is the cosmology-dependent
# observable and carries the $\Omega_m/\sigma_8/h$ log-derivative panel.

# %%
# ---------------------------------------------------------------------------
# X-ray context: gas profiles + APEC cooling table.
# Built lazily (APEC init ~12 s) and reused by figs 12-20.
# ---------------------------------------------------------------------------
def _r200_interp(pred, fid):
    """Return a callable M200 [Msun/h] -> R200 [Mpc/h] from the predictor cache.

    Uses the same ``r_delta`` grid the cross-spectra pipeline feeds to the DPM
    gas profiles, so the radial scale is consistent with the actual model.
    """
    key = pred._cosmo_cache_key(Z_EFF, fid)
    if key not in pred._static_cache:
        pred._pk_tables_full(Z_EFF, fid, IHOD_PARAMS)
        key = pred._cosmo_cache_key(Z_EFF, fid)
    sc = pred._static_cache[key]
    lm = np.log10(np.asarray(sc["m_np"], dtype=float))
    rd = np.asarray(sc["r_delta"], dtype=float)
    return lambda M: float(np.interp(np.log10(M), lm, rd))


def build_xray_context(ctx) -> dict:
    from hod_mod.gas import (
        GasDensityDPM, PressureProfileDPM, MetallicityProfileDPM,
        ApecCoolingTable, temperature_from_profiles,
    )
    from hod_mod.connection.hod import MoreConstFincHODModel

    cool = ApecCoolingTable()                       # ~12 s one-off (APEC tables)
    dp = GasDensityDPM(model=2, n_gl=80)
    pp = PressureProfileDPM(model=2, n_gl=80)
    mp = MetallicityProfileDPM()
    return {
        "cool": cool, "dp": dp, "pp": pp, "mp": mp,
        "agn_hod": MoreConstFincHODModel(ctx["hmf"]),
        "r200_of": _r200_interp(ctx["pred"], ctx["fid"]),
        "T_of": temperature_from_profiles,
    }


XCTX = build_xray_context(CTX)


# %% [markdown]
# ## 12 — gNFW gas shape function (DPM Eq. 1)
#
# All three DPM profiles (density, pressure, metallicity) share a generalised
# NFW radial shape with an inner slope, a transition, and an outer slope:
#
# $$f(x\,|\,\boldsymbol{\alpha}) = x^{-\alpha_\mathrm{in}}
#   \left(1+x^{\alpha_\mathrm{tr}}\right)^{(\alpha_\mathrm{in}-\alpha_\mathrm{out})/\alpha_\mathrm{tr}},
#   \qquad x=r/R_s .$$
#
# Cosmology-independent — we vary the three slope parameters.

# %%
def fig12_gnfw_shape(ctx, xc):
    x = np.logspace(-2, 1.0, 240)

    def gnfw(ain, atr, aout):
        return x ** (-ain) * (1.0 + x ** atr) ** ((ain - aout) / atr)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.loglog(x, gnfw(1.0, 1.9, 2.7), "k", lw=2.4,
              label=r"DPM model 2 $(\alpha_\mathrm{in},\alpha_\mathrm{tr},\alpha_\mathrm{out})=(1.0,1.9,2.7)$")
    for ain in (0.4, 1.4):
        ax.loglog(x, gnfw(ain, 1.9, 2.7), "--", lw=1.3, label=fr"$\alpha_\mathrm{{in}}={ain}$")
    for aout in (2.0, 3.4):
        ax.loglog(x, gnfw(1.0, 1.9, aout), ":", lw=1.3, label=fr"$\alpha_\mathrm{{out}}={aout}$")
    ax.axvline(1.0, color="grey", lw=0.8)
    ax.set_xlabel(r"$x=r/R_s$")
    ax.set_ylabel(r"$f(x\,|\,\boldsymbol{\alpha})$")
    ax.set_ylim(1e-3, 1e3)
    ax.set_title("DPM gas profiles — gNFW shape function (Eq. 1)")
    ax.legend(fontsize=8)
    _save(fig, "eq12_gnfw_shape")
    return fig


fig12_gnfw_shape(CTX, XCTX)


# %% [markdown]
# ## 13 — Electron density, pressure and temperature profiles
#
# $$n_e(r,M,z)=n_{e,0.3}\,\frac{f(r/R_s)}{f(0.3R_{200}/R_s)}\,E(z)^\gamma\,
#   M_{12}^{\beta_\mathrm{gas}},\qquad
#   T(r,M,z)=\frac{P(r,M,z)}{n_e(r,M,z)}\ \ [\mathrm{keV}].$$
#
# The density and pressure share the gNFW shape with mass ($M_{12}^\beta$) and
# redshift ($E(z)^\gamma$) scalings; the temperature follows from the ideal gas
# law.  Curves are shown for three halo masses.

# %%
def fig13_gas_profiles(ctx, xc):
    dp, pp, T_of, r200_of = xc["dp"], xc["pp"], xc["T_of"], xc["r200_of"]
    om, z = ctx["fid"]["Omega_m"], Z_EFF
    r = np.logspace(-2, 0.6, 140)

    fig, (axn, axt) = plt.subplots(2, 1, figsize=(6.6, 7.0), sharex=True,
                                   gridspec_kw={"hspace": 0.08})
    for M, c in [(1e13, "C0"), (1e14, "C1"), (1e15, "C3")]:
        r200 = r200_of(M)
        ne = dp.density_3d(r, M, r200, z, om)
        P = pp._pressure_3d(r, M, r200, z, om)
        axn.loglog(r, ne, c, lw=2.0, label=fr"$M_{{200}}=10^{{{int(round(np.log10(M)))}}}\,M_\odot/h$")
        axt.semilogx(r, T_of(P, ne), c, lw=2.0)
    axn.set_ylabel(r"$n_e(r)\;[\mathrm{cm}^{-3}]$")
    axn.set_title("DPM gas density and temperature profiles")
    axn.legend()
    axt.set_ylabel(r"$T(r)=P/n_e\;[\mathrm{keV}]$")
    axt.set_xlabel(r"$r\;[\mathrm{Mpc}/h]$")
    _save(fig, "eq13_gas_profiles")
    return fig


fig13_gas_profiles(CTX, XCTX)


# %% [markdown]
# ## 14 — Metallicity profile and APEC cooling function
#
# $$Z(r)=Z_0\,f(r/R_s\,|\,\boldsymbol{\alpha}^Z),\qquad
#   \varepsilon=n_e^2\,\Lambda_{n_e^2}(T,Z).$$
#
# The 0.5–2 keV band-integrated APEC cooling function $\Lambda_{n_e^2}(T,Z)$
# (AtomDB, via soxs) sets the X-ray emissivity per $n_e^2$. Left: the gNFW
# metallicity profile; right: $\Lambda$ vs temperature for several metallicities.

# %%
def fig14_cooling(ctx, xc):
    mp, cool = xc["mp"], xc["cool"]
    fig, (az, al) = plt.subplots(1, 2, figsize=(11, 4.4))

    r = np.logspace(-2, 0.6, 140)
    az.semilogx(r, mp.metallicity_3d(r, 1.0), "C2", lw=2.2)
    az.set_xlabel(r"$r\;[\mathrm{Mpc}/h]$")
    az.set_ylabel(r"$Z(r)\;[Z_\odot]$")
    az.set_title("DPM metallicity profile $Z(r)$")

    T = np.logspace(np.log10(0.1), np.log10(15.0), 160)
    for Zval, c in [(0.1, "C0"), (0.3, "C1"), (1.0, "C3")]:
        lam = np.asarray(cool(T, np.full_like(T, Zval)))
        al.loglog(T, lam, c, lw=2.0, label=fr"$Z={Zval}\,Z_\odot$")
    al.set_xlabel(r"$T\;[\mathrm{keV}]$")
    al.set_ylabel(r"$\Lambda_{n_e^2}(T,Z)\;[\mathrm{erg\,cm^3\,s^{-1}}]$")
    al.set_title("APEC cooling function (0.5–2 keV)")
    al.legend()
    _save(fig, "eq14_cooling")
    return fig


fig14_cooling(CTX, XCTX)


# %% [markdown]
# ## 15 — X-ray emissivity profile and its Fourier transform
#
# $$\varepsilon(r,M,z)=n_e^2(r)\,\Lambda_{n_e^2}\!\bigl(T(r),Z(r)\bigr),\qquad
#   \tilde X(k|M,z)=4\pi\!\int_0^{r_\mathrm{max}}\!\varepsilon(r)\,j_0(kr)\,r^2\,\mathrm{d}r.$$
#
# $\tilde X(k|M,z)$ is the key quantity linking the 3D emissivity to the halo
# model.  Left: emissivity profiles; right: their spherical Fourier transforms.

# %%
def fig15_emissivity(ctx, xc):
    dp, pp, mp, cool, T_of, r200_of = (xc["dp"], xc["pp"], xc["mp"],
                                       xc["cool"], xc["T_of"], xc["r200_of"])
    fid = ctx["fid"]
    om, z = fid["Omega_m"], Z_EFF
    r = np.logspace(-2, 0.6, 140)

    fig, (ar, ak) = plt.subplots(1, 2, figsize=(11, 4.6))
    masses = [(1e13, "C0"), (1e14, "C1"), (1e15, "C3")]
    for M, c in masses:
        r200 = r200_of(M)
        ne = dp.density_3d(r, M, r200, z, om)
        P = pp._pressure_3d(r, M, r200, z, om)
        Zr = mp.metallicity_3d(r, r200)
        lam = np.asarray(cool(T_of(P, ne), Zr))
        ar.loglog(r, ne ** 2 * lam, c, lw=2.0,
                  label=fr"$10^{{{int(round(np.log10(M)))}}}\,M_\odot/h$")
    ar.set_xlabel(r"$r\;[\mathrm{Mpc}/h]$")
    ar.set_ylabel(r"$\varepsilon(r)=n_e^2\,\Lambda\;[\mathrm{erg\,s^{-1}cm^{-3}}]$")
    ar.set_title("X-ray emissivity profile")
    ar.legend()

    k = np.logspace(-1, 1.6, 50)
    m = np.array([M for M, _ in masses])
    r200 = np.array([r200_of(M) for M in m])
    X = dp.emissivity_full_uk(k, m, r200, z, fid, pp, mp, cool)  # (Nk, NM)
    for j, (M, c) in enumerate(masses):
        ak.loglog(k, np.abs(X[:, j]), c, lw=2.0,
                  label=fr"$10^{{{int(round(np.log10(M)))}}}\,M_\odot/h$")
    ak.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    ak.set_ylabel(r"$\tilde X(k|M,z)$")
    ak.set_title("Emissivity Fourier transform")
    ak.legend()
    _save(fig, "eq15_emissivity")
    return fig


fig15_emissivity(CTX, XCTX)


# %% [markdown]
# ## 16 — LADE hard X-ray luminosity function (Aird+2015)
#
# $$\Phi(L_X,z)=\frac{k(z)}{(L_X/L_s(z))^{\gamma_1}+(L_X/L_s(z))^{\gamma_2}},$$
#
# with luminosity-dependent density evolution $k(z)$, $L_s(z)$ and slopes
# $\gamma_1=0.48,\ \gamma_2=2.27$.  This XLF supplies the AGN abundance match.

# %%
def fig16_xlf(ctx, xc):
    from hod_mod.agn.ham import _aird15_lade_np
    lx = np.linspace(41.0, 46.0, 240)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for zz, c in [(0.1, "C0"), (0.5, "C1"), (1.0, "C2"), (2.0, "C3")]:
        ax.semilogy(lx, _aird15_lade_np(lx, zz), c, lw=2.0, label=fr"$z={zz}$")
    ax.set_xlabel(r"$\log_{10}(L_X^{2-10\,\mathrm{keV}}/\mathrm{erg\,s^{-1}})$")
    ax.set_ylabel(r"$\Phi(L_X,z)\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.set_ylim(1e-9, 1e-2)
    ax.set_title("LADE hard X-ray luminosity function (Aird+2015)")
    ax.legend()
    _save(fig, "eq16_xlf")
    return fig


fig16_xlf(CTX, XCTX)


# %% [markdown]
# ## 17 — AGN obscuration model and duty cycle
#
# The Comparat+2019 obscuration model gives the obscured ($\log N_H>22$) and
# Compton-thick ($\log N_H\geq24$) fractions as functions of $\log L_X$ and $z$;
# the duty cycle $f_\mathrm{DC}(z)$ sets the active fraction.  These feed the
# hard→soft K-correction and the mean soft luminosity.

# %%
def fig17_obscuration(ctx, xc):
    from hod_mod.agn.ham import (
        obscured_fraction, _f_compton_thick, _duty_cycle_interp)
    lx = np.linspace(41.0, 46.0, 240)
    fig, (ao, ad) = plt.subplots(1, 2, figsize=(11, 4.6))
    for zz, c in [(0.1, "C0"), (1.0, "C3")]:
        ao.plot(lx, np.asarray(obscured_fraction(jnp.asarray(lx), zz)), c, lw=2.2,
                label=fr"$f_\mathrm{{obsc}}\;(z={zz})$")
        ao.plot(lx, np.asarray(_f_compton_thick(jnp.asarray(lx), zz)), c, lw=1.3,
                ls="--", label=fr"$f_\mathrm{{CT}}\;(z={zz})$")
    ao.set_xlabel(r"$\log_{10}L_X$")
    ao.set_ylabel("fraction")
    ao.set_ylim(0, 1)
    ao.set_title("AGN obscured & Compton-thick fractions")
    ao.legend(fontsize=8)

    zz = np.linspace(0.0, 2.0, 120)
    ad.plot(zz, [_duty_cycle_interp(z) for z in zz], "C4", lw=2.2)
    ad.set_xlabel(r"$z$")
    ad.set_ylabel(r"$f_\mathrm{DC}(z)$")
    ad.set_title("AGN duty cycle")
    _save(fig, "eq17_obscuration")
    return fig


fig17_obscuration(CTX, XCTX)


# %% [markdown]
# ## 18 — AGN HOD occupation (More+2015, constant $f_\mathrm{inc}$)
#
# $$N_\mathrm{cen}^\mathrm{AGN}=f_\mathrm{inc}\,\tfrac12\,
#   \mathrm{erfc}\!\Big[\tfrac{\log M_\mathrm{min}-\log M}{\sigma_{\log M}}\Big],\quad
#   N_\mathrm{sat}^\mathrm{AGN}=N_\mathrm{cen}^\mathrm{AGN}
#   \Big(\tfrac{M-\kappa M_\mathrm{min}}{M_1}\Big)^{\alpha}.$$
#
# The mass-independent duty cycle $f_\mathrm{inc}$ scales the whole AGN
# occupation, which drives the occupation-weighted X-ray power spectra of the
# HOD-AGN branch.

# %%
def fig18_agn_hod(ctx, xc):
    agn = xc["agn_hod"]
    lmh = jnp.linspace(11.0, 15.3, 220)
    mh = 10.0 ** np.asarray(lmh)
    base = agn.default_params()
    nc, ns = agn.nc_ns(lmh, base)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.loglog(mh, np.asarray(nc), "grey", lw=1.8, label=r"$N_c^\mathrm{AGN}$ (fid)")
    ax.loglog(mh, np.asarray(ns), "C0", lw=2.2, label=r"$N_s^\mathrm{AGN}$ (fid)")
    ax.loglog(mh, np.asarray(nc) + np.asarray(ns), "k", lw=2.0, label=r"$N_\mathrm{tot}^\mathrm{AGN}$ (fid)")
    for finc, c in [(0.05, "C1"), (0.20, "C3")]:
        p = dict(base, f_inc=finc)
        nc2, ns2 = agn.nc_ns(lmh, p)
        ax.loglog(mh, np.asarray(nc2) + np.asarray(ns2), c, ls="--", lw=1.3,
                  label=fr"$N_\mathrm{{tot}}\,(f_\mathrm{{inc}}={finc})$")
    ax.set_xlabel(r"$M_h\;[M_\odot/h]$")
    ax.set_ylabel(r"$\langle N^\mathrm{AGN}\rangle$")
    ax.set_ylim(1e-4, 1e1)
    ax.set_title("AGN HOD occupation (More+2015, constant $f_\\mathrm{inc}$)")
    ax.legend(fontsize=8)
    _save(fig, "eq18_agn_hod")
    return fig


fig18_agn_hod(CTX, XCTX)


# %% [markdown]
# ## 19 — eROSITA King PSF and beam window
#
# AGN are unresolved point sources, so their angular template is the PSF; in
# $\ell$-space the gas $C_\ell^{gX}$ is multiplied by the beam $B_\ell$:
#
# $$\mathrm{PSF}_\mathrm{King}(\theta)=\Big[1+(\theta/\theta_c)^2\Big]^{-\alpha},
#   \qquad B_\ell=C\,(\ell\theta_c)^{\alpha-1/2}K_{\alpha-1/2}(\ell\theta_c),$$
#
# with $\theta_c=8.64''$, $\alpha=1.5$.  Cosmology-independent (instrument).

# %%
def fig19_psf(ctx, xc):
    from scipy.special import kv, gamma
    tc_arcsec, alpha = 8.64, 1.5
    nu = alpha - 0.5

    fig, (at, al) = plt.subplots(1, 2, figsize=(11, 4.4))
    theta = np.logspace(0.0, 2.7, 240)                       # arcsec
    psf = (1.0 + (theta / tc_arcsec) ** 2) ** (-alpha)
    at.loglog(theta, psf / psf[0], "C0", lw=2.2)
    at.axvline(tc_arcsec, color="grey", lw=0.8, label=r"$\theta_c=8.64''$")
    at.set_xlabel(r"$\theta\;[\mathrm{arcsec}]$")
    at.set_ylabel(r"$\mathrm{PSF}_\mathrm{King}(\theta)$ (norm.)")
    at.set_title("eROSITA King PSF")
    at.legend()

    ell = np.logspace(1, 5, 240)
    x = ell * (tc_arcsec * np.pi / 180.0 / 3600.0)            # ℓ θ_c (θ_c in rad)
    bl = x ** nu * kv(nu, x) / (2.0 ** (nu - 1.0) * gamma(nu))  # B_0 = 1 limit
    al.semilogx(ell, bl, "C3", lw=2.2)
    al.axhline(0.5, color="grey", lw=0.6, ls=":")
    al.set_xlabel(r"$\ell$")
    al.set_ylabel(r"$B_\ell$ (beam window)")
    al.set_title(r"PSF beam window $B_\ell$")
    _save(fig, "eq19_psf")
    return fig


fig19_psf(CTX, XCTX)


# %% [markdown]
# ## 20 — Cosmology sensitivity of the emissivity Fourier transform
#
# The emissivity FT $\tilde X(k|M,z)$ is the X-ray quantity that enters the
# galaxy × X-ray power spectrum
# $P_{gX}^{1h}\propto\int\!\frac{\mathrm{d}n}{\mathrm{d}M}[N_c+N_s\tilde u]\tilde X\,\mathrm{d}M$,
# so its cosmology dependence propagates into $P_{gX}$, $C_\ell^{gX}$ and
# $w_\theta$.  Unlike the raw $P_{gX}$ table (whose physical units underflow to
# zero in float64), $\tilde X(k|M)$ is finite and well-behaved.  It depends on
# $\Omega_m,\sigma_8,h$ through the halo concentration $c(M,z)$ and the
# $E(z)^\gamma$ density scaling.  Top: $\tilde X(k)$ at a fixed
# $M_{200}=10^{14}\,M_\odot/h$ for the fiducial cosmology and three $+2\%$
# variants; bottom: the logarithmic sensitivity $d\ln\tilde X/d\ln p$ (central
# finite differences, at fixed $R_{200}$).

# %%
def fig20_emissivity_sensitivity(ctx, xc):
    dp, pp, mp, cool, r200_of = (xc["dp"], xc["pp"], xc["mp"],
                                 xc["cool"], xc["r200_of"])
    fid = ctx["fid"]
    M = 1.0e14
    r200 = r200_of(M)
    k = np.logspace(-1.0, 1.6, 50)

    def Xk(th):
        return np.abs(dp.emissivity_full_uk(
            k, np.array([M]), np.array([r200]), Z_EFF, th, pp, mp, cool)[:, 0])

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.8, 6.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.08})
    ax.loglog(k, Xk(fid), "k", lw=2.4, label="fiducial")
    for name, lab, c in COSMO_PARAMS:
        ax.loglog(k, Xk(perturb_cosmo(fid, name, 1.02)), c, lw=1.3, ls="--",
                  label=fr"$+2\%\,${lab}")
    ax.set_ylabel(r"$\tilde X(k|M{=}10^{14})$")
    ax.set_title("Emissivity FT — cosmology sensitivity")
    ax.legend()

    sens = log_sensitivity(Xk, fid)
    _plot_sensitivity(axr, k, sens, logx=True)
    axr.set_ylabel(r"$d\ln\tilde X/d\ln p$")
    axr.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    _save(fig, "eq20_emissivity_sensitivity")
    return fig


fig20_emissivity_sensitivity(CTX, XCTX)


# %% [markdown]
# ## Summary
#
# We have illustrated the full ZM15 page.  Equations 1–11 are the iHOD core: the
# first seven are pure-JAX occupation equations (cosmology-independent, shown
# with analytic `jax.grad` parameter sensitivities), and 8–11 are the
# cosmology-dependent clustering/lensing observables with $\Omega_m/\sigma_8/h$
# log-derivative panels.  Equations 12–20 cover the X-ray extension: the DPM hot
# gas profiles, APEC emissivity and its Fourier transform $\tilde X(k|M)$, the
# AGN luminosity function, obscuration and duty cycle, the AGN HOD, the eROSITA
# PSF/beam, and the cosmology sensitivity of $\tilde X(k|M)$ — the ingredient the
# Limber integral and PSF convolution turn (via $P_{gX}$, $C_\ell^{gX}$) into the
# observed $w_\theta(\theta)$.  Throughout, cosmology sensitivities use central
# finite differences because CAMB sits outside the JAX graph.

# %%
if __name__ == "__main__":
    print(f"\nAll figures written to: {OUT_DIR}")
    print("To build the tutorial notebook, re-run with --to-notebook")
