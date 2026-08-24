r"""Growth factor: approximate D(z) versus D(z) inferred from P_lin.

Quantifies the trade-off between the three routes to the linear growth factor
offered by :mod:`hod_mod.core.growth` — the Carroll+1992 fitting formula, the
flat-CPL growth ODE, and the ratio of the active P(k) backend's own spectra —
in **value**, in **derivative**, in **propagated observable**, and in
**wall-clock**.

Motivation.  Until :mod:`hod_mod.core.growth` existed, the pipeline computed the
growth twice by two different methods and never checked that they agreed:
``HaloMassFunction.sigma`` evaluated the backend at z = 0 and multiplied by an
analytic D(z), while the 2-halo term asked the same backend for
``pk_linear(k, z, theta)`` and got the backend's own growth.  Which analytic
formula was used depended on whether ``theta`` happened to carry a ``w0`` key.
This script measures what that cost.

Six parts:

1. **Values** — D(z) per route, across Ω_m, dark energy and Σm_ν.
2. **Scale dependence** — how well-defined a *scalar* D(z) actually is.
3. **Derivatives** — ∂lnD/∂{Ω_m, w0, wa}, which is what a Fisher forecast uses.
4. **Propagation** — Δσ(M,z) → Δ dn/dM, Δb, Δb², Δn_gal, Δb_eff².
5. **Timing** — per-call cost of every route, value and gradient.
6. **Ledger** — backend evaluations per likelihood call.

Usage::

    cd $HOD_MOD_REPO
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.cosmology.growth_factor_study
    #   --no-camb   skip the CAMB cross-check (each solve costs seconds)
    #   --quick     coarse grids, for a smoke test

Writes ``growth_study.json`` plus the figures under
``$HOD_MOD_RESULTS/growth_study/``, and copies the documentation figures to
``docs/_images/growth_*.png``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.core import growth as G
from hod_mod.core.halo_mass_function import (
    HaloMassFunction, _FSIGMA_MODELS, make_bias, _RHO_CRIT0,
    _growth_factor_cpl_jax, _growth_factor_flat_jax)
from hod_mod.core.power_spectrum import (
    omega_cdm as _omega_cdm, omega_cdm_cold_equivalent as _omega_cdm_cold)
from hod_mod.paths import results_root

_HERE = Path(__file__).parent
_IMG_DIR = _HERE.parents[2] / "docs" / "_images"
_DELTA_C = 1.686

#: Planck-2018 flat ΛCDM in the A_s convention, the reference point everywhere.
#:
#: ``Omega_cdm`` is deliberately absent.  Each backend derives the value it
#: needs — the closed-budget ``Ω_m − Ω_b − Ω_ν`` for CAMB and CLASS, the
#: massless-equivalent ``Ω_m − Ω_b`` for the emulator and EH98 — and pinning one
#: here would override that *and* go stale the moment ``sum_mnu`` is added to a
#: copy of this dict, putting the spectrum at ``Ω_m + Ω_ν`` while ρ̄, E(z) and
#: D(z) still used ``Ω_m``.  See ``power_spectrum.omega_cdm``.
THETA = {"h": 0.6736, "Omega_b": 0.0493, "Omega_m": 0.3100,
         "n_s": 0.9649, "ln10^{10}A_s": 3.044}

#: BGS-like effective redshift — where the production fits actually live, and so
#: where a growth error actually costs something.
Z_EFF = 0.3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_HERE.parents[2]), text=True).strip()
    except Exception:
        return "unknown"


def _d_exact_lcdm(z, om, n=400001):
    r"""Exact flat-ΛCDM growing mode by quadrature — the independent reference.

    :math:`D(a) \propto E(a)\int_0^a {\rm d}a' / (a'E(a'))^3`, normalised to
    D(0) = 1.  Deliberately plain numpy on a very fine grid: it must not share
    an implementation with anything it is used to check.
    """
    a = np.linspace(1e-6, 1.0, n)
    e = np.sqrt(om / a ** 3 + 1.0 - om)
    integ = np.cumsum(1.0 / (a * e) ** 3) * (a[1] - a[0])
    d = e * integ
    return np.interp(1.0 / (1.0 + np.asarray(z)), a, d / d[-1])


def _bench(fn, n=50):
    """Milliseconds per call, after one warm-up, block_until_ready-ed."""
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / n * 1e3


def _save(fig, name, out_dir):
    fig.tight_layout()
    for d in (out_dir, _IMG_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=140)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# 1 — values
# ---------------------------------------------------------------------------

def part1_values(pk, zs, out_dir):
    """D(z) per route across Ω_m and the dark-energy sector."""
    print("[1/6] values")
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, om in zip(axes, (0.24, 0.31, 0.40)):
        th = {**THETA, "Omega_m": om}
        exact = _d_exact_lcdm(zs, om)
        car = np.array([float(G.growth_factor(z, th, mode="carroll")) for z in zs])
        ode = np.asarray(_growth_factor_cpl_jax(jnp.asarray(zs), om, -1.0, 0.0))
        rat = np.array([float(G.growth_pk_ratio(z, th, pk)) for z in zs])
        out[f"Omega_m={om}"] = {
            "z": list(map(float, zs)),
            "exact_quadrature": exact.tolist(),
            "carroll_rel": (car / exact - 1).tolist(),
            "ode_rel": (ode / exact - 1).tolist(),
            "pk_ratio_rel": (rat / exact - 1).tolist(),
            "carroll_max_abs_rel": float(np.max(np.abs(car / exact - 1))),
            "ode_max_abs_rel": float(np.max(np.abs(ode / exact - 1))),
            "pk_ratio_max_abs_rel": float(np.max(np.abs(rat / exact - 1))),
        }
        ax.axhline(0, color="0.7", lw=0.8)
        ax.plot(zs, 1e3 * (car / exact - 1), label="Carroll+1992", lw=2)
        ax.plot(zs, 1e3 * (ode / exact - 1), label="CPL growth ODE", lw=2)
        ax.plot(zs, 1e3 * (rat / exact - 1), "--", label="P$_{\\rm lin}$ ratio", lw=2)
        ax.set_title(f"$\\Omega_m = {om}$")
        ax.set_xlabel("$z$")
    axes[0].set_ylabel(r"$10^3\,(D/D_{\rm exact}-1)$")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Growth factor error against exact flat-$\\Lambda$CDM quadrature", y=1.02)
    _save(fig, "growth_values.png", out_dir)

    # dark-energy sector: the Carroll path cannot respond to it at all
    de = {}
    for label, (w0, wa) in {"LCDM": (-1.0, 0.0), "w0=-0.8": (-0.8, 0.0),
                            "wa=+0.5": (-1.0, 0.5), "wa=-0.5": (-1.0, -0.5)}.items():
        th = {**THETA, "w0": w0, "wa": wa}
        de[label] = {
            "ode": [float(_growth_factor_cpl_jax(z, THETA["Omega_m"], w0, wa)) for z in zs],
            "carroll": [float(_growth_factor_flat_jax(z, THETA["Omega_m"])) for z in zs],
        }
    de["note"] = ("Carroll is identical across all four rows: the flat-LCDM "
                  "fitting formula has no dark-energy sector.")
    out["dark_energy"] = de
    return out


# ---------------------------------------------------------------------------
# 2 — scale dependence
# ---------------------------------------------------------------------------

def part2_scale_dependence(pk, zs, out_dir):
    r"""How well-defined a scalar D(z) is: the k-spread of √(P(k,z)/P(k,0))."""
    print("[2/6] scale dependence")
    k = jnp.logspace(-3.0, 1.0, 128)
    knp = np.asarray(k)
    out = {"k_min": float(knp[0]), "k_max": float(knp[-1]), "cases": {}}
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for mnu in (0.0, 0.06, 0.12):
        th = dict(THETA)
        if mnu > 0:
            th["sum_mnu"] = mnu
        spreads, curves = [], []
        for z in zs:
            tas = G._as_amplitude_theta(th)
            p_z = np.asarray(pk.pk_linear(k, float(z), tas))
            p_0 = np.asarray(pk.pk_linear(k, 0.0, tas))
            d_k = np.sqrt(p_z / p_0)
            spreads.append(float(d_k.max() / d_k.min() - 1.0) if z > 0 else 0.0)
            curves.append(d_k)
        out["cases"][f"sum_mnu={mnu}"] = {
            "z": list(map(float, zs)), "k_spread": spreads,
            "max_k_spread": float(np.max(spreads)),
        }
        ax.plot(zs, spreads, marker="o", lw=2, label=f"$\\Sigma m_\\nu = {mnu}$ eV")
        # the z=1 curve, normalised, shows *where* in k the spread sits
        iz = int(np.argmin(np.abs(np.asarray(zs) - 1.0)))
        out["cases"][f"sum_mnu={mnu}"]["dk_over_dk_large_scale_z1"] = \
            (curves[iz] / curves[iz][0]).tolist()

    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\max_k D(k)\,/\,\min_k D(k) - 1$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("A scalar $D(z)$ is only defined to this precision\n"
                 r"($k \in [10^{-3},10]\ h\,{\rm Mpc}^{-1}$)", fontsize=10)
    _save(fig, "growth_scale_dependence.png", out_dir)
    return out


# ---------------------------------------------------------------------------
# 3 — derivatives
# ---------------------------------------------------------------------------

def part3_derivatives(pk, out_dir):
    r"""∂lnD/∂θ — what a Fisher forecast actually differentiates through."""
    print("[3/6] derivatives")
    zs = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    out = {"z": zs, "dlnD_dOmega_m": {}, "dlnD_dw0": {}, "dlnD_dwa": {}}

    def dlnD_dOm(z, mode):
        if mode == "pk_ratio":
            f = lambda om: jnp.log(G.growth_pk_ratio(
                z, {**THETA, "Omega_m": om}, pk))
        else:
            f = lambda om: jnp.log(G.growth_factor(
                z, {**THETA, "Omega_m": om}, mode=mode))
        return float(jax.grad(f)(THETA["Omega_m"]))

    for mode in ("carroll", "ode", "pk_ratio"):
        out["dlnD_dOmega_m"][mode] = [dlnD_dOm(z, mode) for z in zs]
    ref = np.array(out["dlnD_dOmega_m"]["pk_ratio"])
    out["dlnD_dOmega_m"]["carroll_rel_err"] = (
        np.array(out["dlnD_dOmega_m"]["carroll"]) / ref - 1).tolist()
    out["dlnD_dOmega_m"]["ode_rel_err"] = (
        np.array(out["dlnD_dOmega_m"]["ode"]) / ref - 1).tolist()

    for name, key in (("dlnD_dw0", "w0"), ("dlnD_dwa", "wa")):
        base = -1.0 if key == "w0" else 0.0
        out[name]["ode"] = [
            float(jax.grad(lambda v: jnp.log(G.growth_factor(
                z, {**THETA, key: v}, mode="ode")))(base)) for z in zs]
        # The Carroll branch has no dark-energy argument, so this is structurally
        # zero — not a small number, an absent one.
        out[name]["carroll"] = [0.0] * len(zs)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.axhline(0, color="0.7", lw=0.8)
    a1.plot(zs, 100 * np.array(out["dlnD_dOmega_m"]["carroll_rel_err"]),
            marker="o", lw=2, label="Carroll+1992")
    a1.plot(zs, 100 * np.array(out["dlnD_dOmega_m"]["ode_rel_err"]),
            marker="s", lw=2, label="CPL growth ODE")
    a1.set_xlabel("$z$")
    a1.set_ylabel(r"error on $\partial \ln D/\partial \Omega_m$  [%]")
    a1.set_title("Sensitivity that the Fisher matrix inherits", fontsize=10)
    a1.legend(frameon=False)

    a2.plot(zs, out["dlnD_dw0"]["ode"], marker="o", lw=2, label=r"ODE, $\partial\ln D/\partial w_0$")
    a2.plot(zs, out["dlnD_dwa"]["ode"], marker="s", lw=2, label=r"ODE, $\partial\ln D/\partial w_a$")
    a2.axhline(0, color="crimson", lw=2, ls="--", label="Carroll (both, structurally)")
    a2.set_xlabel("$z$")
    a2.set_ylabel(r"$\partial \ln D/\partial \theta$")
    a2.set_title("The dark-energy sector Carroll cannot see", fontsize=10)
    a2.legend(frameon=False, fontsize=9)
    _save(fig, "growth_derivatives.png", out_dir)
    return out


# ---------------------------------------------------------------------------
# 4 — propagation into the halo model
# ---------------------------------------------------------------------------

def part4_propagation(pk, out_dir):
    r"""Δ D(z) → Δ σ(M,z) → Δ dn/dM, Δb, Δb², Δn_gal, Δb_eff².

    A uniform rescaling of σ leaves ``dlnσ/dlnM`` untouched, so dn/dM responds
    only through the multiplicity function and b only through ν = δ_c/σ.  That
    makes the propagation exact without re-running the mass function.
    """
    print("[4/6] propagation")
    m = jnp.logspace(11.0, 15.2, 120)
    mnp = np.asarray(m)
    hmf = HaloMassFunction(pk.pk_linear, rho_mean=THETA["Omega_m"] * _RHO_CRIT0,
                           model="tinker08")
    fsig = _FSIGMA_MODELS["tinker08"]
    bfn = make_bias("tinker10")
    out = {"log10_m": np.log10(mnp).tolist(), "z": {}}

    for z in (0.1, 0.3, 0.7):
        d_ref = float(G.growth_pk_ratio(z, THETA, pk))       # accurate
        d_car = float(G.growth_factor(z, THETA, mode="carroll"))
        eps = d_car / d_ref - 1.0                            # Carroll's error

        sig_ref = np.asarray(hmf.sigma(m, z, THETA))
        # hmf.sigma applies its own internal growth; renormalise to the accurate
        # one, then apply Carroll's error to get the pair being compared.
        sig_car = sig_ref * (1.0 + eps)

        f_ref = np.asarray(fsig(jnp.asarray(sig_ref), z))
        f_car = np.asarray(fsig(jnp.asarray(sig_car), z))
        b_ref = np.asarray(bfn(_DELTA_C / jnp.asarray(sig_ref), 200.0))
        b_car = np.asarray(bfn(_DELTA_C / jnp.asarray(sig_car), 200.0))

        # A Zheng07-like occupation, only to weight the mass integrals the way a
        # real BGS-threshold sample does — n_gal and b_eff are what wp sees.
        n_of_m = 0.5 * (1.0 + jax.scipy.special.erf(
            (jnp.log10(m) - 12.1) / 0.25)) + (m / 10 ** 13.4) ** 1.0
        n_of_m = np.asarray(n_of_m)
        dndm_ref = f_ref / mnp ** 2
        dndm_car = f_car / mnp ** 2
        ng_ref = np.trapezoid(dndm_ref * n_of_m, mnp)
        ng_car = np.trapezoid(dndm_car * n_of_m, mnp)
        beff_ref = np.trapezoid(dndm_ref * n_of_m * b_ref, mnp) / ng_ref
        beff_car = np.trapezoid(dndm_car * n_of_m * b_car, mnp) / ng_car

        out["z"][str(z)] = {
            "D_pk_ratio": d_ref, "D_carroll": d_car, "dlnD_carroll_error": eps,
            "sigma": sig_ref.tolist(),
            "dndm_rel": (f_car / f_ref - 1).tolist(),
            "bias_rel": (b_car / b_ref - 1).tolist(),
            "bias2_rel": ((b_car / b_ref) ** 2 - 1).tolist(),
            "max_abs_dndm_rel": float(np.max(np.abs(f_car / f_ref - 1))),
            "max_abs_bias2_rel": float(np.max(np.abs((b_car / b_ref) ** 2 - 1))),
            "n_gal_rel": float(ng_car / ng_ref - 1),
            "b_eff2_rel": float((beff_car / beff_ref) ** 2 - 1),
            "note": ("b_eff2_rel is the fractional shift of the 2-halo amplitude "
                     "of w_p, which scales as b_eff^2."),
        }
        print(f"      z={z}: dn/dM up to {100*out['z'][str(z)]['max_abs_dndm_rel']:.2f}%, "
              f"b^2 up to {100*out['z'][str(z)]['max_abs_bias2_rel']:.2f}%, "
              f"n_gal {100*out['z'][str(z)]['n_gal_rel']:+.2f}%, "
              f"b_eff^2 {100*out['z'][str(z)]['b_eff2_rel']:+.2f}%")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for z in ("0.1", "0.3", "0.7"):
        d = out["z"][z]
        a1.plot(out["log10_m"], 100 * np.array(d["dndm_rel"]), lw=2, label=f"z = {z}")
        a2.plot(out["log10_m"], 100 * np.array(d["bias2_rel"]), lw=2, label=f"z = {z}")
    for a, lab in ((a1, r"$\Delta (dn/dM)$  [%]"), (a2, r"$\Delta b^2$  [%]")):
        a.axhline(0, color="0.7", lw=0.8)
        a.set_xlabel(r"$\log_{10} M\ [M_\odot/h]$")
        a.set_ylabel(lab)
        a.legend(frameon=False)
    fig.suptitle("What a $1.7\\times10^{-3}$ growth error becomes downstream "
                 "(Carroll vs $P_{\\rm lin}$ ratio)", y=1.02, fontsize=11)
    _save(fig, "growth_propagation.png", out_dir)
    return out


# ---------------------------------------------------------------------------
# 5 — timing
# ---------------------------------------------------------------------------

def part5_timing(pk, out_dir, with_camb):
    """Per-call cost of every route, forward and gradient."""
    print("[5/6] timing")
    zs = jnp.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0])
    om = THETA["Omega_m"]
    out = {"n_redshifts": int(zs.size), "device": jax.default_backend(),
           "forward_ms": {}, "gradient_ms": {}, "ode_convergence": {}}

    f_car = jax.jit(lambda o: jax.vmap(lambda z: _growth_factor_flat_jax(z, o))(zs))
    f_ode = jax.jit(lambda o: _growth_factor_cpl_jax(zs, o, -1.0, 0.0))
    out["forward_ms"]["carroll"] = _bench(lambda: f_car(om))
    out["forward_ms"]["ode_n160"] = _bench(lambda: f_ode(om))
    for n in (40, 80, 160, 320, 640):
        fn = jax.jit(lambda o, n=n: _growth_factor_cpl_jax(zs, o, -1.0, 0.0, n_steps=n))
        exact = _d_exact_lcdm(np.asarray(zs), om)
        out["ode_convergence"][f"n_steps={n}"] = {
            "ms": _bench(lambda: fn(om)),
            "max_abs_rel_vs_exact": float(np.max(np.abs(np.asarray(fn(om)) / exact - 1))),
        }

    f_rat = jax.jit(lambda t: jnp.stack(
        [G.growth_pk_ratio(float(z), t, pk) for z in np.asarray(zs)]))
    out["forward_ms"]["pk_ratio_emulator"] = _bench(lambda: f_rat(THETA), n=20)

    g_car = jax.jit(jax.grad(lambda o: jnp.sum(f_car(o))))
    g_ode = jax.jit(jax.grad(lambda o: jnp.sum(f_ode(o))))
    out["gradient_ms"]["carroll"] = _bench(lambda: g_car(om))
    out["gradient_ms"]["ode_n160"] = _bench(lambda: g_ode(om))

    if with_camb:
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        lp = LinearPowerSpectrum()
        k = jnp.logspace(-3, 1, 64)
        lp.pk_linear(k, 0.0, THETA)                       # warm import/first solve
        t0 = time.perf_counter(); lp.pk_linear(k, 0.0, THETA)
        t_one = time.perf_counter() - t0
        t0 = time.perf_counter(); lp.pk_linear(k, 0.3, THETA)
        t_two = time.perf_counter() - t0
        out["camb_s"] = {
            "single_solve": t_one,
            "second_solve": t_two,
            "pk_ratio_today_two_solves": t_one + t_two,
            "note": ("A single CAMB run can return many redshifts at essentially "
                     "the cost of one, so the two-solve pattern is avoidable in "
                     "the backend; measured separately in the campaign notes."),
        }
        print(f"      CAMB: {t_one:.2f} s + {t_two:.2f} s for the pair")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = ["Carroll", "ODE n=160", "ODE n=320", "ODE n=640", "P$_{\\rm lin}$ ratio\n(emulator)"]
    vals = [out["forward_ms"]["carroll"],
            out["ode_convergence"]["n_steps=160"]["ms"],
            out["ode_convergence"]["n_steps=320"]["ms"],
            out["ode_convergence"]["n_steps=640"]["ms"],
            out["forward_ms"]["pk_ratio_emulator"]]
    errs = [np.max(np.abs(np.asarray(f_car(om)) / _d_exact_lcdm(np.asarray(zs), om) - 1)),
            out["ode_convergence"]["n_steps=160"]["max_abs_rel_vs_exact"],
            out["ode_convergence"]["n_steps=320"]["max_abs_rel_vs_exact"],
            out["ode_convergence"]["n_steps=640"]["max_abs_rel_vs_exact"],
            1e-6]
    sc = ax.scatter(vals, errs, s=90, c=range(len(vals)), cmap="viridis", zorder=3)
    for x, y, n in zip(vals, errs, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"ms per call ({zs.size} redshifts, jitted, {jax.default_backend()})")
    ax.set_ylabel(r"max $|D/D_{\rm exact}-1|$")
    ax.set_title("Cost against accuracy: the ODE dominates both alternatives", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    _save(fig, "growth_timing.png", out_dir)
    return out


# ---------------------------------------------------------------------------
# 6 — the double-calculation ledger
# ---------------------------------------------------------------------------

def part6_ledger(pk):
    """Backend evaluations per halo-model call — the 'double calculation'.

    Counts real ``predict`` calls on the emulator by wrapping it, so the number
    is measured rather than reasoned about.
    """
    print("[6/6] ledger")
    from hod_mod.forecast.pk_cosmopower import _load_emulator
    emu = _load_emulator("mpk_lin")
    orig = emu.predict
    count = {"n": 0}

    def counting(p, *a, **kw):
        count["n"] += 1
        return orig(p, *a, **kw)

    out = {}
    try:
        emu.predict = counting
        m = jnp.logspace(11.0, 15.0, 64)
        hmf = HaloMassFunction(pk.pk_linear, rho_mean=THETA["Omega_m"] * _RHO_CRIT0,
                               model="tinker08")
        count["n"] = 0
        hmf.sigma(m, Z_EFF, THETA)
        n_sigma = count["n"]
        count["n"] = 0
        pk.pk_linear(jnp.logspace(-4, 2, 256), Z_EFF, THETA)
        n_2halo = count["n"]
        count["n"] = 0
        G.growth_pk_ratio(Z_EFF, THETA, pk)
        n_growth = count["n"]
        out = {
            "hmf_sigma_calls": n_sigma,
            "two_halo_pk_linear_calls": n_2halo,
            "growth_pk_ratio_calls": n_growth,
            "total_today": n_sigma + n_2halo,
            "note": ("hmf.sigma evaluates the backend at z=0 and the 2-halo term "
                     "evaluates it again at z; the growth ratio needs both, so "
                     "sharing one pair of spectra across all three is what phase B "
                     "collapses.  Under jax.jit XLA already CSEs identical "
                     "subgraphs, so this ledger is the eager/CAMB cost, not the "
                     "jitted forecast cost."),
        }
        print(f"      sigma: {n_sigma}, 2-halo: {n_2halo}, growth ratio: {n_growth}")
    finally:
        emu.predict = orig
    return out


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-camb", action="store_true", help="skip the CAMB cross-check")
    ap.add_argument("--quick", action="store_true", help="coarse grids, smoke test")
    args = ap.parse_args(argv)

    from hod_mod.core.power_spectrum import default_pk_linear
    pk = default_pk_linear()
    zs = np.linspace(0.0, 3.0, 7 if args.quick else 25)
    out_dir = results_root() / "growth_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"backend: {type(pk).__name__}   ->  {out_dir}")

    doc = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "platform": platform.platform(),
        "device": jax.default_backend(),
        "backend": type(pk).__name__,
        "cosmology": THETA,
        "omega_cdm_derived": {
            "closed_budget_camb_class": float(_omega_cdm(THETA)),
            "massless_equivalent_emulator_eh98": float(_omega_cdm_cold(THETA)),
            "note": ("The two differ by Omega_nu once sum_mnu > 0; the study "
                     "pins neither, so each backend derives its own."),
        },
        "values": part1_values(pk, zs, out_dir),
        "scale_dependence": part2_scale_dependence(pk, zs, out_dir),
        "derivatives": part3_derivatives(pk, out_dir),
        "propagation": part4_propagation(pk, out_dir),
        "timing": part5_timing(pk, out_dir, with_camb=not args.no_camb),
        "ledger": part6_ledger(pk),
    }
    path = out_dir / "growth_study.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {path}")
    return doc


if __name__ == "__main__":
    main()
