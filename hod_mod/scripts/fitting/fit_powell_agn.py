"""Fit the analytic Powell 2022 AGN-halo model (:class:`hod_mod.agn.powell.PowellAGNModel`).

Constrains the M_BH-M_* relation (norm, slope, scatter), the ERDF (log10 lambda*,
delta1, delta2), the ERDF normalisation / active fraction (log10_ferdf) and the
M_BH-M_halo correlation rho against:
  * the X-ray luminosity function (abundance) — Aird+2015 or Ueda+2014 hard XLF;
  * a clustering constraint — the AGN effective bias / median host halo mass
    (Powell Table 1: median host ~10^12, average ~10^13.3 Msun/h).

MAP (Nelder-Mead) then emcee, with informative Gaussian priors centred on the
Powell 2022 (M_BH-M_*) and Ananna 2022 (ERDF) values.  Results -> $HOD_MOD_RESULTS.

Usage:
    HOD_MOD_RESULTS=<...> JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.fit_powell_agn
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

from hod_mod import paths
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.agn.powell import PowellAGNModel
from hod_mod.agn.ham import _aird15_lade_np, _ueda14_ldde_np

_OUT_DIR = os.fspath(paths.results_root() / "powell_agn")
_PARAMS = ["mu_bh", "al_bh", "sig_bh", "log10_lstar", "delta1", "delta2", "log10_ferdf", "rho"]
# informative priors (Powell 2022 Model 1 M_BH-M_*; Ananna 2022 ERDF); ferdf/rho wide
_PRIOR_MU  = np.array([7.76, 0.67, 0.33, np.log10(0.13), 0.30, 3.70, -2.0, 0.0])
_PRIOR_SIG = np.array([0.30, 0.24, 0.18, 0.20, 0.15, 0.66, 1.0, np.inf])
_BOUNDS = np.array([[6.5, 9.0], [0.0, 1.5], [0.05, 1.0], [-1.5, 0.3],
                    [0.05, 1.0], [1.5, 6.0], [-4.0, 0.0], [0.0, 0.95]])
# clustering constraint (Powell Table 1): median host log10 M_halo [Msun/h]
_HOST_MU, _HOST_SIG = 12.1, 0.4


def _xlf_data(which="aird15", z=0.135):
    lx = np.arange(42.0, 45.51, 0.5)
    f = {"aird15": _aird15_lade_np, "ueda14": _ueda14_ldde_np}[which]
    phi = f(lx, z) * 0.70 ** 3            # Aird/Ueda h=0.70 -> Mpc^-3 dex^-1
    return lx, phi, 0.15 * np.ones_like(lx)   # 0.15 dex assumed error


def _apply(M, p, h):
    M.set_params(mbh_m=(p[0], p[1], p[2]), erdf=(p[3], p[4], p[5]),
                 rho=p[7], log10_ferdf=p[6])


def _neg_log_prob(p, M, h, lx_d, logphi_d, sig_d):
    for v, (lo, hi) in zip(p, _BOUNDS):
        if not (lo <= v <= hi):
            return 1e30
    _apply(M, p, h)
    grid, phi = M.xlf(band="hard")
    logphi = np.log10(np.maximum(np.interp(lx_d, grid, phi * h ** 3), 1e-40))
    chi2 = np.sum(((logphi - logphi_d) / sig_d) ** 2)
    chi2 += ((M.median_host_logmhalo() - _HOST_MU) / _HOST_SIG) ** 2
    fin = np.isfinite(_PRIOR_SIG)
    chi2 += np.sum(((np.asarray(p)[fin] - _PRIOR_MU[fin]) / _PRIOR_SIG[fin]) ** 2)
    return 0.5 * float(chi2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlf", default="aird15", choices=["aird15", "ueda14"])
    ap.add_argument("--z", type=float, default=0.135)
    ap.add_argument("--mcmc", action="store_true")
    ap.add_argument("--nwalkers", type=int, default=48)
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--nburn", type=int, default=1000)
    args = ap.parse_args(argv)
    os.makedirs(_OUT_DIR, exist_ok=True)

    pk = LinearPowerSpectrum(); theta = LinearPowerSpectrum.default_cosmology()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear); h = float(theta["h"])
    M = PowellAGNModel(theta, hmf, z_mean=args.z, log10lx_min=42.0)
    lx_d, phi_d, sig_d = _xlf_data(args.xlf, args.z)
    logphi_d = np.log10(np.maximum(phi_d, 1e-40))

    def nlp(p):
        return _neg_log_prob(p, M, h, lx_d, logphi_d, sig_d)

    print(f"Fitting Powell AGN to {args.xlf} XLF (z={args.z}) + host-mass prior ...", flush=True)
    best = None
    for jit in (0.0, 0.15, -0.15):
        o = minimize(nlp, _PRIOR_MU + jit * np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.5, 0.5, 0.0]),
                     method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
        if best is None or o.fun < best.fun:
            best = o
    mp = best.x
    _apply(M, mp, h)
    ndof = max(len(lx_d) + 1 - len(_PARAMS), 1)
    out = dict(zip(_PARAMS, [float(v) for v in mp]))
    out["chi2"] = 2 * best.fun; out["median_host_log10M"] = M.median_host_logmhalo()
    out["agn_bias"] = M.agn_bias(); out["active_frac"] = float(10 ** mp[6])
    with open(os.path.join(_OUT_DIR, f"powell_map_{args.xlf}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== POWELL AGN MAP ===")
    for k in _PARAMS:
        print(f"  {k:12s} = {out[k]:+.4f}")
    print(f"  median host log10M = {out['median_host_log10M']:.2f}  bias={out['agn_bias']:.2f}  "
          f"active_frac={out['active_frac']:.3f}  chi2={out['chi2']:.1f}", flush=True)

    if not args.mcmc:
        return out
    import emcee
    ndim = len(_PARAMS); nw = args.nwalkers
    rng = np.random.default_rng(0)
    p0 = mp + 1e-3 * rng.standard_normal((nw, ndim)) * np.ptp(_BOUNDS, axis=1)
    p0 = np.clip(p0, _BOUNDS[:, 0] + 1e-6, _BOUNDS[:, 1] - 1e-6)
    sampler = emcee.EnsembleSampler(nw, ndim, lambda p: -nlp(p) if nlp(p) < 1e29 else -np.inf)
    print(f"\nMCMC {nw}x{args.nsteps} ...", flush=True)
    sampler.run_mcmc(p0, args.nsteps, progress=False)
    flat = sampler.get_chain(discard=args.nburn, flat=True)
    np.savez(os.path.join(_OUT_DIR, f"powell_chain_{args.xlf}.npz"), flatchain=flat, params=_PARAMS)
    pct = np.percentile(flat, [16, 50, 84], axis=0)
    print(f"acceptance={np.mean(sampler.acceptance_fraction):.2f}; posterior (median +hi -lo):")
    for i, k in enumerate(_PARAMS):
        print(f"  {k:12s} = {pct[1,i]:.3f} +{pct[2,i]-pct[1,i]:.3f} -{pct[1,i]-pct[0,i]:.3f}", flush=True)
    with open(os.path.join(_OUT_DIR, f"powell_summary_{args.xlf}.json"), "w") as fh:
        json.dump(dict(map=out, posterior={k: dict(median=float(pct[1, i]), lo=float(pct[1, i]-pct[0, i]),
                       hi=float(pct[2, i]-pct[1, i])) for i, k in enumerate(_PARAMS)}), fh, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import corner
        corner.corner(flat, labels=_PARAMS, truths=list(mp)).savefig(
            os.path.join(_OUT_DIR, f"powell_corner_{args.xlf}.png"), dpi=110)
    except Exception as e:
        print(f"  (corner skipped: {e})", flush=True)
    return out


if __name__ == "__main__":
    main()
