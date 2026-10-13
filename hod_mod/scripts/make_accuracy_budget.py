"""Regenerate ``benchmarks/accuracy.json`` -- the per-quantity accuracy budget.

Following the practice of CCL (Chisari et al. 2019), every predicted quantity
carries a *quoted numerical accuracy*, established by comparison against an
independent implementation and recorded in a machine-readable file rather than
in prose. Tables in the documentation and the paper are generated from it, so
they cannot drift away from what the code actually does.

Each entry records the quantity, the reference it was compared against, the
measured deviation, the tolerance the test suite enforces, and enough context to
reproduce it.

Run::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.make_accuracy_budget

Entries whose reference package is not installed are recorded with
``"status": "skipped"`` rather than omitted, so a gap in the budget is visible.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.core.power_spectrum import (
    LinearPowerSpectrum as _LPS)

_default_cosmology = _LPS.default_cosmology

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "benchmarks", "accuracy.json"))

Z = 0.2
# The package default, with the budget closed: Omega_cdm = Om - Ob - Onu.
# It already carries ln10^{10}A_s = 3.044; a sigma8 used to be injected on top,
# which made this THETA carry two amplitudes and take the retired branch.
THETA = dict(_default_cosmology())


def _entry(quantity, reference, deviation, tolerance, metric, note,
           status="ok", deviation_median=None):
    """One line of the budget.

    ``deviation`` is the worst case, which is what a tolerance is set on.
    ``deviation_median`` is recorded too where it is meaningful: a single bad
    point and a uniform offset of the same size are very different situations,
    and the max alone cannot tell them apart.
    """
    return {"quantity": quantity, "reference": reference,
            "deviation": deviation, "deviation_median": deviation_median,
            "tolerance": tolerance, "metric": metric,
            "note": note, "status": status}


def _dev(a, b):
    """(max, median) of |a/b - 1|."""
    r = np.abs(np.asarray(a) / np.asarray(b) - 1.0)
    return float(np.max(r)), float(np.median(r))


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.join(_HERE, "..", ".."), text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
def check_distances():
    """chi(z) and E(z) against astropy."""
    try:
        from astropy.cosmology import FlatLambdaCDM
    except ImportError:
        return [_entry("E(z)", "astropy", None, 1e-5, "max |ratio-1|",
                       "astropy not installed", "skipped")]
    from hod_mod.observables.clustering import _hubble_E, _comoving_dist_h
    ap = FlatLambdaCDM(H0=100 * THETA["h"], Om0=THETA["Omega_m"], Tcmb0=0)
    z = np.linspace(0.01, 3.0, 40)
    dE = float(np.max(np.abs(
        np.asarray(_hubble_E(jnp.array(z), THETA["Omega_m"])) / ap.efunc(z) - 1)))
    dchi = float(np.max(np.abs(
        np.asarray(_comoving_dist_h(z, THETA))
        / (ap.comoving_distance(z).value * THETA["h"]) - 1)))
    return [
        _entry("E(z)", "astropy FlatLambdaCDM", dE, 1e-5, "max |ratio-1|",
               "z in [0.01, 3]; Tcmb0=0 to match the radiation-free convention"),
        _entry("chi(z)", "astropy FlatLambdaCDM", dchi, 1e-3, "max |ratio-1|",
               "set by the trapezoidal quadrature, not by a modelling choice"),
    ]


def check_concentration_and_bias():
    """c(M,z) and b(M,z) against COLOSSUS."""
    try:
        from colossus.cosmology import cosmology as cc
        from colossus.halo import concentration as ccon
        from colossus.lss import bias as cbias
    except ImportError:
        return [_entry("c(M,z) Duffy08", "COLOSSUS", None, 1e-3,
                       "max |ratio-1|", "colossus not installed", "skipped")]
    from hod_mod.core.concentration import ConcentrationModel
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.core.power_spectrum import eisenstein_hu_pk
    cc.setCosmology("planck18")
    m = np.logspace(11, 15.5, 60)
    cm = ConcentrationModel(model="duffy08", mdef="200m")
    d_c = float(np.max(np.abs(
        np.asarray(cm.concentration(jnp.array(m), Z, THETA))
        / ccon.concentration(m, "200m", Z, model="duffy08") - 1)))
    hmf = make_hmf("tinker08", pk_func=lambda k, z, t: eisenstein_hu_pk(k, t))
    d_b = float(np.max(np.abs(
        np.asarray(hmf.bias(jnp.array(m), Z, THETA))
        / cbias.haloBias(m, Z, mdef="200m", model="tinker10") - 1)))
    return [
        _entry("c(M,z) Duffy08", "COLOSSUS", d_c, 1e-3, "max |ratio-1|",
               "two implementations of one closed-form fit"),
        _entry("b(M,z) Tinker10", "COLOSSUS", d_b, 0.05, "max |ratio-1|",
               "folds in the P(k) backend: COLOSSUS uses its own sigma(M)"),
    ]


def check_power_spectrum():
    """Linear P(k): every backend against the CLASS reference, plus the nu table."""
    out = []
    from hod_mod.core.power_spectrum import LinearPowerSpectrum, eisenstein_hu_pk
    k = np.logspace(-3, 1, 200)
    kj = jnp.array(k)
    i0 = int(np.argmin(abs(k - 0.05)))
    nrm = lambda p: np.asarray(p) / np.asarray(p)[i0]

    # CLASS is the reference: an independent Boltzmann solver, and the only
    # backend here that treats massive neutrinos exactly.  When classy is not
    # installed the reference falls back to CAMB -- recorded in each row's
    # "reference" field, so which solver was used is never implicit.
    cl = None
    try:
        from hod_mod.core.power_spectrum import ClassPkLinear
        cl = ClassPkLinear()
        ref = nrm(cl.pk_linear(kj, 0.0, THETA))
        ref_name = "CLASS"
    except Exception:
        ref = nrm(LinearPowerSpectrum().pk_linear(kj, 0.0, THETA))
        ref_name = "CAMB"

    if ref_name == "CLASS":
        d = float(np.max(np.abs(nrm(LinearPowerSpectrum().pk_linear(kj, 0.0, THETA)) / ref - 1)))
        out.append(_entry("P_lin(k) shape, CAMB", "CLASS", d, 0.02, "max |ratio-1|",
                          "two Boltzmann solvers at matched N_eff and "
                          "Sigma m_nu; bounds the reference's own error"))
    else:
        out.append(_entry("P_lin(k) shape, CAMB", "CLASS", None, 0.02,
                          "max |ratio-1|", "unavailable: classy not installed",
                          "skipped"))

    try:
        from hod_mod.forecast.pk_cosmopower import CosmoPowerJaxPkLinear
        d = float(np.max(np.abs(nrm(CosmoPowerJaxPkLinear().pk_shape(kj, THETA)) / ref - 1)))
        out.append(_entry("P_lin(k) shape, CosmoPower-JAX", ref_name, d, 0.05,
                          "max |ratio-1|",
                          "normalised at k=0.05 h/Mpc; the emulator is trained "
                          "massless and is corrected by the CLASS nu table"))
    except Exception as e:
        out.append(_entry("P_lin(k) shape, CosmoPower-JAX", ref_name, None, 0.05,
                          "max |ratio-1|", f"unavailable: {type(e).__name__}", "skipped"))

    d = float(np.max(np.abs(nrm(eisenstein_hu_pk(kj, THETA)) / ref - 1)))
    out.append(_entry("P_lin(k) shape, EH98", ref_name, d, 0.06, "max |ratio-1|",
                      "oscillating acoustic-feature error of a fitting formula"))

    # The distilled cold/total table against a live CLASS solve: this is what
    # sigma(M) and the mass function are built on.
    if cl is not None:
        try:
            from hod_mod.core.nu_ratio import suppression_cb, suppression_m
            th = dict(THETA, sum_mnu=0.30)      # amplify the effect ~5x
            live = (np.asarray(cl.pk_linear_cb(kj, 0.0, th))
                    / np.asarray(cl.pk_linear(kj, 0.0, th)))
            tab = (np.asarray(suppression_cb(kj, th))
                   / np.asarray(suppression_m(kj, th)))
            d = float(np.max(np.abs(tab / live - 1)))
            out.append(_entry("P_cb/P_m table", "CLASS", d, 0.005, "max |ratio-1|",
                              "shipped class_nu_ratio.npz vs a live solve at "
                              "Sigma m_nu = 0.30 eV"))
        except Exception as e:
            out.append(_entry("P_cb/P_m table", "CLASS", None, 0.005,
                              "max |ratio-1|", f"unavailable: {type(e).__name__}",
                              "skipped"))
        cl.clear_cache()
    else:
        out.append(_entry("P_cb/P_m table", "CLASS", None, 0.005, "max |ratio-1|",
                          "unavailable: classy not installed", "skipped"))
    return out


def check_bias_consistency():
    """Mass--bias consistency for every (f(sigma), b) pairing shipped."""
    from hod_mod.core.halo_mass_function import (
        _FSIGMA_MODELS, bias_consistency, matched_bias_for)
    out = []
    for fs in sorted(_FSIGMA_MODELS):
        matched = matched_bias_for(fs)
        for bias in sorted({matched, "tinker10"} - {None}):
            r = bias_consistency(fs, bias)
            normalisable = 0.8 < r["mass_norm"] < 1.4
            out.append(_entry(
                f"<b> consistency, {fs} + {bias}",
                "peak-background split (analytic)",
                r["deviation"], 0.05 if (bias == matched and normalisable) else None,
                "|<b> - 1|",
                f"mass_norm = {r['mass_norm']:.3f}"
                + ("" if normalisable else "; not normalisable, <b> not meaningful")
                + ("; PBS-matched pair" if bias == matched else "; ad hoc pairing"),
                "ok" if normalisable else "informational",
            ))
    return out


def check_mass_translation():
    """Round-trip exactness of the NFW mass-definition translation."""
    from hod_mod.core.mass_definitions import translate_mass
    m = jnp.array([1e12, 1e13, 1e14, 1e15])
    c = jnp.array([8.0, 6.0, 5.0, 4.0])
    worst = 0.0
    for other in ("200c", "500c", "vir", "2500c"):
        m1, _, c1 = translate_mass(m, c, "200m", other, Z, THETA)
        m2, _, _ = translate_mass(m1, c1, other, "200m", Z, THETA)
        worst = max(worst, float(np.max(np.abs(np.asarray(m2) / np.asarray(m) - 1))))
    return [_entry("mass translation round trip", "self (NFW, analytic)",
                   worst, 1e-4, "max |ratio-1|",
                   "200m -> {200c, 500c, vir, 2500c} -> 200m")]


def check_aum():
    """w_p and Delta Sigma against the AUM C++ halo model."""
    sys.path.insert(0, "/home/comparat/software/previous/aum/install/"
                       "lib/python3.11/site-packages")
    try:
        import hod as h_aum
    except ImportError:
        return [_entry("w_p(r_p)", "AUM (C++)", None, 0.15, "max |ratio-1|",
                       "AUM not importable", "skipped")]
    import math
    import camb
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.core.halo_profiles import HaloProfile
    from hod_mod.connection.hod import AUMHODModel
    from hod_mod.observables.clustering import FullHaloModelPrediction

    th = dict(THETA); th.update(w0=-1.0, wa=0.0)
    hp_par = {"log10mmin": 13.0, "sigma_logm": 0.5, "log10m0": 10.0,
              "log10m1": 14.0, "alpha": 1.0}
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * th["h"], ombh2=th["Omega_b"] * th["h"] ** 2,
                       omch2=th["Omega_cdm"] * th["h"] ** 2)
    pars.InitPower.set_params(ns=th["n_s"], As=np.exp(th["ln10^{10}A_s"]) * 1e-10)
    pars.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model="ppf")
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)
    s8 = float(camb.get_results(pars).get_sigma8_0())

    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    pred = FullHaloModelPrediction(
        pk, AUMHODModel(hmf, hmf.bias),
        HaloProfile({"flat": True, "H0": th["h"] * 100, "Om0": th["Omega_m"],
                     "Ob0": th["Omega_b"], "sigma8": s8, "ns": th["n_s"]},
                    cm_relation="diemer19"), profile="nfw")
    p = h_aum.cosmo()
    p.Om0, p.Omk, p.w0, p.wa = th["Omega_m"], 0.0, -1.0, 0.0
    p.Omb, p.hval, p.th, p.s8, p.nspec = th["Omega_b"], th["h"], 2.7255, s8, th["n_s"]
    p.ximax, p.cfac = math.log10(8.0), 1.0
    q = h_aum.hodpars()
    q.Mmin, q.siglogM, q.Msat = 13.0, 0.5, 14.0
    q.alpsat, q.Mcut, q.csbycdm, q.fac = 1.0, 10.0, 1.0, 1.0
    aum = h_aum.hod(p, q)

    def carr(a):
        c_ = h_aum.doubleArray(len(a))
        for i, v in enumerate(a):
            c_[i] = float(v)
        return c_

    rp = np.logspace(np.log10(0.5), np.log10(20.0), 16)
    out_c = h_aum.doubleArray(len(rp))
    aum.Wp(0.1, len(rp), carr(rp), out_c, 60.0)
    wp_aum = np.array([out_c[i] for i in range(len(rp))])
    wp_hm = np.asarray(pred.wp(jnp.asarray(rp), 60.0, 0.1, th, hp_par))

    R = np.logspace(np.log10(0.3), 1.0, 14)
    out_c = h_aum.doubleArray(len(R))
    aum.ESD(0.1, len(R), carr(R), out_c, len(R) + 8)
    ds_aum = np.array([out_c[i] for i in range(len(R))])
    ds_hm = np.asarray(pred.delta_sigma(jnp.asarray(R), 0.1, th, hp_par))

    wp_max, wp_med = _dev(wp_hm, wp_aum)
    ds_max, ds_med = _dev(ds_hm, ds_aum)
    return [
        _entry("w_p(r_p)", "AUM (C++)", wp_max, 0.15, "max |ratio-1|",
               "matched configuration: identical occupation and sigma8; residual is "
               "the P(k) shape at large r_p and c(M) below 1 Mpc/h",
               deviation_median=wp_med),
        _entry("Delta Sigma(R)", "AUM (C++)", ds_max, 0.15, "max |ratio-1|",
               "same configuration as w_p", deviation_median=ds_med),
    ]


def check_growth():
    """D(z)/D(0): every route against the ones that can arbitrate it.

    Four independent references, because no single one settles it:

    * **exact flat-LCDM quadrature** -- ``D(a) ~ E(a) int da/(aE)^3``, plain
      numpy on a fine grid.  Arbitrates the analytic routes in LCDM, and shares
      no implementation with what it checks.
    * **the backend P(k) ratio** -- what the 2-halo term actually grows by.  The
      ODE agreeing with it is what makes the two legs of the halo model
      consistent, which is the whole point of the dispatcher.
    * **the k-spread of that ratio** -- recorded as informational because it is
      not an error at all: with massive neutrinos the growth genuinely *is*
      scale-dependent, and the number says how well a *scalar* D(z) is even
      defined.  At Sigma m_nu = 0.06 it is 3.4e-3, an order of magnitude above
      the ODE's own error -- so above that mass the choice of scalar route stops
      being the dominant question.
    * **Carroll+1992** -- carried as informational, not because anything should
      use it, but so the budget records what the pre-0.6 default cost.
    """
    from hod_mod.core import growth as G
    from hod_mod.core.halo_mass_function import _growth_factor_cpl_jax
    from hod_mod.core.power_spectrum import default_pk_linear

    zs = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    om = float(THETA["Omega_m"])

    # exact flat-LCDM growing mode, by quadrature
    a = np.linspace(1e-6, 1.0, 400001)
    e = np.sqrt(om / a ** 3 + 1.0 - om)
    d = e * (np.cumsum(1.0 / (a * e) ** 3) * (a[1] - a[0]))
    exact = np.interp(1.0 / (1.0 + zs), a, d / d[-1])

    th_lcdm = {kk: vv for kk, vv in THETA.items() if kk != "sum_mnu"}
    ode = np.asarray(_growth_factor_cpl_jax(jnp.asarray(zs), om, -1.0, 0.0))
    car = np.array([float(G.growth_factor(z, th_lcdm, mode="carroll")) for z in zs])
    d_ode_max, d_ode_med = _dev(ode, exact)
    d_car_max, d_car_med = _dev(car, exact)

    out = [
        _entry("D(z), growth ODE", "exact flat-LCDM quadrature",
               d_ode_max, 3e-4, "max |ratio-1|",
               "z in [0.1, 3], Omega_m = 0.31; RK4 on 160 ln-a steps",
               deviation_median=d_ode_med),
        _entry("D(z), Carroll+1992", "exact flat-LCDM quadrature",
               d_car_max, None, "max |ratio-1|",
               "informational: the pre-0.6 default, kept only for reproducing "
               "old results.  The error is Omega_m-dependent (3.9e-3 at 0.24, "
               "5.5e-4 at 0.40), so a free-cosmology fit cannot absorb it",
               deviation_median=d_car_med),
    ]

    try:
        pk = default_pk_linear()
    except Exception as e:
        out.append(_entry("D(z), growth ODE", "backend P(k) ratio", None, 1e-3,
                          "max |ratio-1|", f"backend unavailable: {e}", "skipped"))
        return out

    rat = np.array([float(G.growth_pk_ratio(z, th_lcdm, pk)) for z in zs])
    r_max, r_med = _dev(ode, rat)
    out.append(_entry(f"D(z), growth ODE", f"P(k) ratio, {type(pk).__name__}",
                      r_max, 1e-3, "max |ratio-1|",
                      "the consistency that matters: sigma(M) and the 2-halo "
                      "P_lin must grow by the same number",
                      deviation_median=r_med))

    # how well-defined is a scalar D(z)?  informational by nature.
    k = jnp.logspace(-3.0, 1.0, 128)
    for mnu in (0.0, 0.06):
        th = dict(th_lcdm)
        if mnu > 0:
            th["sum_mnu"] = mnu
        tas = G._as_amplitude_theta(th)
        spread = 0.0
        for z in (1.0, 3.0):
            dk = np.sqrt(np.asarray(pk.pk_linear(k, float(z), tas))
                         / np.asarray(pk.pk_linear(k, 0.0, tas)))
            spread = max(spread, float(dk.max() / dk.min() - 1.0))
        out.append(_entry(f"D(z) scale-dependence, Sigma m_nu = {mnu} eV",
                          "its own k-spread", spread, None,
                          "max_k D / min_k D - 1",
                          "informational: not an error but a statement that a "
                          "scalar D(z) is only defined to this precision, over "
                          "k in [1e-3, 10] h/Mpc at z = 1 and 3"))

    # sigma(M,z) separability -- exact only in the massless limit now that the
    # cold-to-total ratio carries a redshift.
    from hod_mod.core.halo_mass_function import HaloMassFunction, _RHO_CRIT0
    hmf = HaloMassFunction(pk.pk_linear, rho_mean=om * _RHO_CRIT0, model="tinker08")
    m = jnp.logspace(11.0, 15.0, 48)
    s0 = np.asarray(hmf.sigma(m, 0.0, th_lcdm))
    s1 = np.asarray(hmf.sigma(m, 0.7, th_lcdm))
    r = s1 / s0
    out.append(_entry("sigma(M,z)/sigma(M,0) mass-independence, massless",
                      "its own mass-spread", float(r.max() / r.min() - 1.0),
                      1e-6, "max_M r / min_M r - 1",
                      "the separability sigma(M,z) = sigma(M,0) D(z) that lets a "
                      "multi-z caller hoist the k-integral.  The tolerance is "
                      "float32 round-off, not physics: this script does not "
                      "enable x64, so ~1.2e-7 is its floor for any exact "
                      "identity.  In float64 the spread is 2e-16 -- see "
                      "tests/test_growth.py.  Exact only while sum_mnu is "
                      "absent: the cold-to-total ratio S_cb(k,z) carries a "
                      "redshift, so with massive neutrinos this separability is "
                      "broken by design"))
    return out


def main():
    entries = []
    for fn in (check_distances, check_concentration_and_bias, check_power_spectrum,
               check_growth, check_bias_consistency, check_mass_translation,
               check_aum):
        try:
            entries.extend(fn())
        except Exception as e:                     # a broken check must not hide the rest
            entries.append(_entry(fn.__name__, "-", None, None, "-",
                                  f"{type(e).__name__}: {e}", "error"))
            print(f"  {fn.__name__}: FAILED {type(e).__name__}: {e}", flush=True)

    failures = [e for e in entries
                if e["status"] == "ok" and e["tolerance"] is not None
                and e["deviation"] is not None
                and e["deviation"] > e["tolerance"]]

    budget = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "platform": jax.default_backend(),
        "redshift": Z,
        "cosmology": {k: v for k, v in THETA.items()},
        "n_entries": len(entries),
        "n_out_of_tolerance": len(failures),
        "entries": entries,
    }
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w") as fh:
        json.dump(budget, fh, indent=2)

    for e in entries:
        dev = "-" if e["deviation"] is None else f"{e['deviation']:.3e}"
        med = e.get("deviation_median")
        dev = dev if med is None else f"{dev} (med {med:.1e})"
        tol = "-" if e["tolerance"] is None else f"{e['tolerance']:.1e}"
        flag = "  <-- OUT OF TOLERANCE" if e in failures else ""
        print(f"  {e['quantity']:48s} vs {e['reference']:28s} "
              f"{dev:>22s} / {tol:>7s}  [{e['status']}]{flag}")
    print(f"\nwrote {_OUT}  ({len(entries)} entries, "
          f"{len(failures)} out of tolerance)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
