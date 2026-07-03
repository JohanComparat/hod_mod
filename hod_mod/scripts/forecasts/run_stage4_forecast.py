r"""Stage-IV multi-survey Fisher forecast for the gas--galaxy--AGN pipeline.

Unlike :mod:`hod_mod.scripts.forecasts.run_sensitivity_study` (which assigns a
single *constant* relative error to every data point to expose degeneracy
structure), this script assigns **realistic, per-observable, scale-dependent
Gaussian errors** representative of the end-of-survey Stage-IV combination of
maps, and propagates them to :math:`\sigma(\Omega_m,\sigma_8,S_8)` and the
:math:`(\Omega_m,\sigma_8)` figure of merit **as a function of the scale cut**.

Survey → observable mapping (2026 HDR outlook, Comparat; "by 2030" combination):

* ``wp``    projected galaxy clustering — DESI + 4MOST (spectroscopic).
* ``ds``    galaxy--galaxy lensing (ΔΣ) — LSST/Euclid/Roman shapes × DESI/LSST lenses.
* ``cl_kk`` cosmic shear — LSST + Euclid + Roman.
* ``cl_gy`` galaxy × thermal SZ — DESI/LSST × ACT / SPT / Planck.
* ``cl_gX`` galaxy × soft X-ray (0.5--2 keV) — DESI/LSST × eROSITA.
* ``cl_XX`` soft X-ray auto — eROSITA.
* ``xlf``   AGN X-ray luminosity function — eROSITA AGN.

Error model (all values documented in ``_STAGE4`` and the docstring of
:func:`per_bin_relerr`; areas from the HDR survey table, so f_sky is grounded;
the noise-to-signal ratios are effective per-bin values calibrated to the current
measurement S/N and standard shape/shot/instrument budgets):

* angular spectra: Gaussian (Knox) ``ΔC/C = sqrt(2 / N_modes) · (1 + N_ℓ/C_ℓ)``
  with ``N_modes(ℓ) = (2ℓ+1)·ℓ·Δlnℓ·f_sky`` and ``N_ℓ/C_ℓ = rN·(ℓ/100)^aN``;
* projected wp/ΔΣ: ``sqrt( (f_CV/√f_sky)² + (rN·(1 Mpc/h / r_p)^aN)² )``;
* the XLF: a constant Poisson-like per-dex error ``rN``.

Usage::

    JAX_PLATFORMS=cpu HOD_MOD_RESULTS=/path python -m \
      hod_mod.scripts.forecasts.run_stage4_forecast --rmin 0.1 0.3 0.5 1 2.5 5 10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES, OBSERVABLES, _IDX  # noqa: E402
from hod_mod.forecast import params, fisher  # noqa: E402

# --- Stage-IV per-observable error model ----------------------------------
# (survey label, f_sky, rN [noise-to-signal at the pivot], aN [growth exponent])
#  f_sky from the HDR survey-area table; rN/aN are effective per-bin values for
#  the *end-of-survey* Stage-IV depth (higher source/tracer densities and lower
#  shape/shot noise than Stage-III, so the noise-to-signal at the pivot and its
#  small-scale growth are markedly milder than a Stage-III forecast).
_STAGE4 = {
    "wp":    ("DESI + 4MOST (spectroscopic clustering)",             0.35, 0.015, 0.5),
    "ds":    ("LSST/Euclid/Roman shapes × DESI/LSST lenses (GGL)",   0.25, 0.03, 0.6),
    "cl_kk": ("LSST + Euclid + Roman (cosmic shear)",                0.40, 0.05, 1.1),
    "cl_gy": ("DESI/LSST × ACT/SPT/Planck (tSZ)",                    0.30, 0.25, 0.9),
    "cl_gX": ("DESI/LSST × eROSITA soft X-ray (0.5–2 keV)",          0.35, 0.30, 1.0),
    "cl_XX": ("eROSITA soft X-ray auto",                             0.35, 0.70, 1.2),
    "cl_kCMB":       ("ACT/SPT/Planck CMB lensing auto",             0.40, 0.25, 0.9),
    "cl_gkCMB":      ("DESI/LSST × CMB lensing",                     0.35, 0.15, 0.7),
    "cl_shear_kCMB": ("LSST shear × CMB lensing",                    0.35, 0.20, 0.9),
    "xlf":   ("eROSITA AGN X-ray luminosity function",              0.35, 0.05, 0.0),
    "n_gal": ("galaxy number density",                              0.35, 0.01, 0.0),
    "smf":   ("stellar mass function",                              0.35, 0.03, 0.0),
}
_ELL_PIVOT = 100.0
_RP_PIVOT = 1.0            # Mpc/h
_F_CV_PROJ = 0.003        # projected large-scale cosmic-variance floor at f_sky=1
                          # (a few×1e-3: Stage-IV volumes contain a huge number of
                          #  large-scale modes, so the sample-variance floor per
                          #  r_p bin is well below 1%)
_ANGULAR = ("cl_gX", "cl_gy", "cl_XX", "cl_kk", "cl_kCMB", "cl_gkCMB", "cl_shear_kCMB")
_PROJECTED = ("wp", "ds")


def per_bin_relerr(model, row_obs, row_x):
    """Return the per-row fractional error σ_i/d_i under the Stage-IV model."""
    row_obs = np.asarray(row_obs)
    row_x = np.asarray(row_x, dtype=float)
    dlnl = float(np.mean(np.diff(np.log(np.asarray(model.ell)))))
    out = np.ones_like(row_x)
    for o in OBSERVABLES:
        m = row_obs == o
        if not m.any():
            continue
        _, fsky, rN, aN = _STAGE4[o]
        x = row_x[m]
        if o in _ANGULAR:
            nmodes = (2.0 * x + 1.0) * x * dlnl * fsky
            cv = np.sqrt(2.0 / nmodes)
            nts = rN * (x / _ELL_PIVOT) ** aN
            out[m] = cv * (1.0 + nts)
        elif o in _PROJECTED:
            floor = _F_CV_PROJ / np.sqrt(fsky)
            noise = rN * (_RP_PIVOT / x) ** aN
            out[m] = np.sqrt(floor ** 2 + noise ** 2)
        else:                      # xlf: constant Poisson-like per-dex error
            out[m] = rN * np.ones_like(x)
    return np.clip(out, 1e-4, 2.0)


def _s8_sigma(cov, fid):
    """σ(S8) with S8 = σ8 (Ω_m/0.3)^0.5, via linear error propagation."""
    iom, is8 = _IDX["Omega_m"], _IDX["sigma8"]
    Om, s8 = fid[iom], fid[is8]
    g = np.zeros(cov.shape[0])
    g[iom] = s8 * 0.5 * (Om / 0.3) ** (-0.5) / 0.3
    g[is8] = (Om / 0.3) ** 0.5
    var = float(g @ cov @ g)
    return np.sqrt(max(var, 0.0)), s8 * (Om / 0.3) ** 0.5


def _constraints(d0, J, relerr, mask, prior, cov_full=None):
    if cov_full is not None:
        F = fisher.fisher_matrix(d0[mask], J[mask], prior_sigma=prior,
                                 cov=cov_full[np.ix_(mask, mask)])
    else:
        F = fisher.fisher_matrix(d0[mask], J[mask], relerr[mask], prior)
    return fisher.constraints(F, ridge=1e-10)


def _out_dir():
    try:
        from hod_mod import paths
        d = os.fspath(paths.results_root() / "stage4_forecast")
    except Exception:
        d = os.path.join(os.environ.get("HOD_MOD_RESULTS", "."), "stage4_forecast")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rmin", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 1.0, 2.5, 5.0, 10.0])
    ap.add_argument("--n-k", type=int, default=256)
    ap.add_argument("--n-m", type=int, default=256)
    ap.add_argument("--agn-pinned", action="store_true",
                    help="externally pin the 7 Powell AGN params (unlocks the XLF's cosmology).")
    ap.add_argument("--free-tier2", action="store_true",
                    help="free the tier-2 extension parameters (formerly-fixed "
                         "nuisances + z-slopes) instead of pinning them")
    ap.add_argument("--covariance", choices=["diagonal", "gaussian"], default="diagonal",
                    help="diagonal per-bin errors, or the analytic Gaussian covariance "
                         "with lensing-triplet cross-observable correlations.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = _out_dir()
    names = PARAM_NAMES
    fid = params.fiducial_vector()
    iom, is8 = _IDX["Omega_m"], _IDX["sigma8"]

    print(f"[setup] Stage-IV forecast, {len(names)} params, {len(OBSERVABLES)} observables")
    model = ForwardModel(z_eff=0.2, n_k=args.n_k, n_m=args.n_m)
    f_full, row_obs, row_x = model.full_data_vector_fn(OBSERVABLES)
    print("[jacobian] full jax.jacfwd ...", flush=True)
    d0, J = fisher.jacobian(f_full, fid)
    d0, J = np.asarray(d0), np.asarray(J)
    relerr = per_bin_relerr(model, row_obs, row_x)
    row_obs = np.asarray(row_obs)
    print(f"[jacobian] {d0.size} rows; finite={np.all(np.isfinite(J))}")

    cov_full = None
    if args.covariance == "gaussian":
        from hod_mod.forecast import covariance
        cov_full = covariance.gaussian_covariance(model, fid, d0, row_obs, row_x, relerr)
        print("[covariance] analytic Gaussian (lensing-triplet cross-terms) built")

    # priors: LSS-only (regularizing) and LSS+Planck (tight Ω_m, σ8).  The
    # tier-2 extension parameters are pinned by default so this Stage-IV study
    # keeps its fixed-nuisance meaning (free them via run_tier2_forecast).
    fix_agn = ("agn_mu_bh", "agn_al_bh", "agn_sig_bh", "agn_log10_lstar",
               "agn_delta1", "agn_delta2", "agn_log10_ferdf") if args.agn_pinned else ()
    fix_t2 = () if args.free_tier2 else tuple(params.TIER2_EXTENSION)
    fix = tuple(fix_agn) + fix_t2
    prior_lss = params.regularizing_prior(add_planck=False, fix=fix)
    prior_cmb = params.regularizing_prior(add_planck=True, fix=fix)

    # cumulative probe build-up, grouped by data type:
    # clustering (wp, ds) → abundances (n_gal, smf) → lensing/CMB-lensing sector
    # → X-ray and SZ (gas + AGN).
    order = ["wp", "ds", "n_gal", "smf",
             "cl_gkCMB", "cl_shear_kCMB", "cl_kk", "cl_kCMB",
             "cl_gX", "cl_XX", "cl_gy", "xlf"]

    results = {}
    print("\n[forecast] σ per scale cut (Stage-IV combination, all 7 maps):")
    hdr = f"{'Rmin':>6} | {'σ(Ωm)':>9} {'σ(σ8)':>9} {'σ(S8)':>9} {'FoM':>9} | {'+Planck σ(Ωm)':>13} {'σ(σ8)':>9} {'σ(S8)':>9}"
    print(hdr)
    for rmin in args.rmin:
        smask = model.scale_cut_mask(row_obs, row_x, rmin)
        rec = {}
        for tag, prior in (("lss", prior_lss), ("cmb", prior_cmb)):
            cov, sig, _ = _constraints(d0, J, relerr, smask, prior, cov_full)
            s8sig, _ = _s8_sigma(cov, fid)
            rec[tag] = dict(sigma=sig, cov=cov, s8=s8sig,
                            fom=fisher.figure_of_merit(cov, iom, is8))
        # cumulative probe attribution (LSS-only)
        cum = {}
        for i in range(1, len(order) + 1):
            subset = order[:i]
            mm = smask & np.isin(row_obs, subset)
            cov, sig, _ = _constraints(d0, J, relerr, mm, prior_lss, cov_full)
            cum["+".join(subset)] = dict(
                sOm=float(sig[iom]), sS8=float(_s8_sigma(cov, fid)[0]),
                fom=float(fisher.figure_of_merit(cov, iom, is8)))
        results[rmin] = dict(lss=rec["lss"], cmb=rec["cmb"], cumulative=cum)
        L, C = rec["lss"], rec["cmb"]
        print(f"{rmin:6.2f} | {L['sigma'][iom]:9.3e} {L['sigma'][is8]:9.3e} {L['s8']:9.3e} "
              f"{L['fom']:9.3g} | {C['sigma'][iom]:13.3e} {C['sigma'][is8]:9.3e} {C['s8']:9.3e}",
              flush=True)

    _save(out, args, names, fid, d0, J, relerr, row_obs, row_x, results, order)
    if not args.no_plots:
        _make_plots(out, model, fid, d0, J, relerr, row_obs, row_x, results, args, order)
    print(f"[done] outputs in {out}")


def _save(out, args, names, fid, d0, J, relerr, row_obs, row_x, results, order):
    tab = {"rmin": args.rmin, "agn_pinned": bool(args.agn_pinned),
           "stage4_error_model": {o: dict(survey=_STAGE4[o][0], f_sky=_STAGE4[o][1],
                                          rN=_STAGE4[o][2], aN=_STAGE4[o][3]) for o in OBSERVABLES},
           "results": {f"{rm}": {
               "lss": {"sigma_Omega_m": float(results[rm]["lss"]["sigma"][_IDX["Omega_m"]]),
                       "sigma_sigma8": float(results[rm]["lss"]["sigma"][_IDX["sigma8"]]),
                       "sigma_S8": float(results[rm]["lss"]["s8"]),
                       "FoM": float(results[rm]["lss"]["fom"])},
               "lss_plus_planck": {"sigma_Omega_m": float(results[rm]["cmb"]["sigma"][_IDX["Omega_m"]]),
                                   "sigma_sigma8": float(results[rm]["cmb"]["sigma"][_IDX["sigma8"]]),
                                   "sigma_S8": float(results[rm]["cmb"]["s8"]),
                                   "FoM": float(results[rm]["cmb"]["fom"])},
               "cumulative": results[rm]["cumulative"]} for rm in args.rmin}}
    with open(os.path.join(out, "stage4_forecast.json"), "w") as fh:
        json.dump(tab, fh, indent=2)
    np.savez(os.path.join(out, "stage4_forecast.npz"),
             param_names=np.array(names), fiducial=fid, d0=d0, jacobian=J,
             relerr=relerr, row_obs=row_obs, row_x=row_x, rmin=np.array(args.rmin))


def _make_plots(out, model, fid, d0, J, relerr, row_obs, row_x, results, args, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmins = args.rmin
    iom, is8 = _IDX["Omega_m"], _IDX["sigma8"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # panel 1: the assumed per-bin fractional errors per observable
    ax = axes[0]
    lab = {"wp": r"$w_p$", "ds": r"$\Delta\Sigma$", "cl_kk": r"$C_\ell^{\kappa\kappa}$",
           "cl_gy": r"$C_\ell^{gy}$", "cl_gX": r"$C_\ell^{gX}$", "cl_XX": r"$C_\ell^{XX}$",
           "cl_kCMB": r"$C_\ell^{\kappa_c\kappa_c}$", "cl_gkCMB": r"$C_\ell^{g\kappa_c}$",
           "cl_shear_kCMB": r"$C_\ell^{\kappa\kappa_c}$", "xlf": r"$\Phi(L_X)$",
           "n_gal": r"$n_{\rm gal}$", "smf": r"$\Phi(M_*)$"}
    for o in OBSERVABLES:
        m = row_obs == o
        xx = row_x[m]; yy = relerr[m] * 100.0
        s = np.argsort(xx)
        ax.plot(xx[s], yy[s], "o-", ms=3, label=lab[o])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc/h] / $\ell$ / $\log_{10}L_X$")
    ax.set_ylabel("assumed per-bin error [%]")
    ax.set_title("Stage-IV per-bin Gaussian errors")
    ax.legend(fontsize=8, ncol=2)

    # panel 2: sigma(Om), sigma(s8), sigma(S8) vs scale cut (LSS-only + Planck)
    ax = axes[1]
    sOm = [results[rm]["lss"]["sigma"][iom] for rm in rmins]
    ss8 = [results[rm]["lss"]["sigma"][is8] for rm in rmins]
    sS8 = [results[rm]["lss"]["s8"] for rm in rmins]
    sOm_c = [results[rm]["cmb"]["sigma"][iom] for rm in rmins]
    ax.loglog(rmins, sOm, "o-", color="C0", label=r"$\sigma(\Omega_m)$ (LSS)")
    ax.loglog(rmins, ss8, "s-", color="C1", label=r"$\sigma(\sigma_8)$ (LSS)")
    ax.loglog(rmins, sS8, "^-", color="C2", label=r"$\sigma(S_8)$ (LSS)")
    ax.loglog(rmins, sOm_c, "o--", color="C0", alpha=0.6, label=r"$\sigma(\Omega_m)$ (+Planck)")
    ax.invert_xaxis()
    ax.set_xlabel(r"scale cut $R_{\min}$ [Mpc/h]")
    ax.set_ylabel(r"marginalised $1\sigma$")
    ax.set_title("Stage-IV constraint vs scale cut")
    ax.legend(fontsize=8)

    # panel 3: cumulative probe build-up of sigma(S8) at the smallest scale cut
    ax = axes[2]
    rm0 = min(rmins)
    cum = results[rm0]["cumulative"]
    keys = ["+".join(order[:i]) for i in range(1, len(order) + 1)]
    xlabels = ["+" + order[i - 1] if i > 1 else order[0] for i in range(1, len(order) + 1)]
    sS8c = [cum[k]["sS8"] for k in keys]
    sOmc = [cum[k]["sOm"] for k in keys]
    xpos = np.arange(len(keys))
    ax.plot(xpos, np.array(sS8c) * 1e3, "^-", color="C2", label=r"$\sigma(S_8)\times10^3$")
    ax.plot(xpos, np.array(sOmc) * 1e3, "o-", color="C0", label=r"$\sigma(\Omega_m)\times10^3$")
    ax.set_yscale("log")
    ax.set_xticks(xpos); ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("cumulative maps added")
    ax.set_ylabel(r"$1\sigma\times10^3$")
    ax.set_title(rf"Probe build-up at $R_{{\min}}={rm0}$ Mpc/h (LSS-only)")
    ax.legend(fontsize=9)

    tag = " (AGN sector pinned)" if args.agn_pinned else ""
    fig.suptitle("Stage-IV multi-survey forecast: gas–galaxy–AGN pipeline" + tag, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, "stage4_forecast.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)
    # also drop a copy into docs/_images for the doc page
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        root = here
        for _ in range(6):
            if os.path.isdir(os.path.join(root, "docs")):
                break
            root = os.path.dirname(root)
        dimg = os.path.join(root, "docs", "_images")
        if os.path.isdir(dimg):
            import shutil
            shutil.copyfile(p, os.path.join(dimg, "stage4_forecast.png"))
            print("[fig] copied to", os.path.join(dimg, "stage4_forecast.png"))
    except Exception as e:
        print("  (docs copy skipped:", e, ")")


if __name__ == "__main__":
    main()
