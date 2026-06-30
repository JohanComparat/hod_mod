"""Joint full-APEC galaxy×X-ray w(theta) fit across the LS10-BGS samples S1-S7.

Unlike the per-sample density-only baseline (``fit_agn_duty_cycle_baseline``),
this fits ALL samples SIMULTANEOUSLY with ONE set of shared *physical* gas/AGN
scaling-relation parameters.  The samples (different stellar-mass thresholds)
probe complementary halo masses through their FIXED ZuMandelbaum15 HOD, so a
single physical gas scaling relation must reproduce every sample's w(theta) at
once.  The X-ray emission uses the full APEC cooling Λ(T(r),Z(r)) — i.e. w(theta)
is temperature-dependent — via ``emissivity_full_uk`` (the n_e²·Λ FT).

Speed: the expensive piece is the spherical-Bessel FT of the emissivity
(``_profile_uk_gl``, ~1.3 s/z after the einsum+sinc rewrite).  It depends only on
the gas PROFILE SHAPE (p2, r_max) and z, so we cache the raw FT X̃(k|M)/Λ_ref on a
(p2, r_max) grid per sample (``emissivity_xuk_per_z``) and feed it back through
``angular_cl_gX(x_uk_override=...)`` — the cheap β_gas/β_pressure tilts, ECF
weight, mass integral, Limber and Hankel still run, so the β grid is built without
re-doing the FT.  This builds the per-sample gas-shape grid in ~2 min.

Shared free parameters (Phase A, broad 0.5-2 keV band):
  log10_ne_03  gas central electron density [cm^-3]  (physical, no free norm)
  beta_gas     gas mass slope  n_e² ∝ M^(2β)         (= L_X-M scaling)
  p2           gas outer slope (profile shape)
  r_max        gas truncation radius / r200
  log10DC      AGN duty cycle (host fraction), in [0.001, 0.5]
The pressure (→ temperature) normalisation P_03 is tied self-similarly to ne_03
(β_P = β_gas + 2/3, P_03 calibrated per β_gas) in Phase A; freeing it is Phase B,
where the 15 narrow energy bands constrain kT directly.

Usage:
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.fit_xray_joint \
        --samples S1 S2 S3 S4 S5 S6 S7 --map-only
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import GasDensityDPM
from hod_mod.gas.cooling import ApecCoolingTable
from hod_mod.gas.metallicity import MetallicityProfileDPM
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.duty_cycle import DutyCycleAGNModel
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts.fitting import fit_agn_duty_cycle_baseline as B
from hod_mod.scripts.validate_gas_profiles import (
    _make_density_variant, _make_pressure_variant, _calibrate_ne03_P03,
)

_OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "results", "xray_joint")
)

# ---- emulator grid (gas emission profile shape) ----
# Each (p2, r_max, beta) cell builds the full-APEC emissivity FT EXPLICITLY (no
# post-FT beta tilt approximation, which only captures n_e² and not the T=P/n_e
# shift of Λ(T) for the full-APEC path).  The override is used only for the exact
# (n_e,0.3/n_e_fid)² rescale.  Grid kept modest so the ~14 s/cell stays ~45 min/7.
_ALPHA_PROF = 0.9
# Grid shifted toward the data-preferred region (S1 diagnostic wanted low p2,
# high beta, large r_max).
_P2_GRID   = np.array([0.3, 1.0, 2.4])                  # gas outer slope
_RMAX_GRID = np.array([3.0, 4.0, 5.0])                  # r_max / r200
_BETA_GRID = np.array([0.9, 1.5, 2.1])                  # gas mass slope (L_X-M)

_THETA_MIN, _THETA_MAX = 8.0, 300.0

# Physical priors (shared with the baseline).
_NE03_FID = B._NE03_FID
_NORM_LO, _NORM_HI = B._NORM_LO, B._NORM_HI        # n_e/n_e_fid in [0.1, 10]
_C_OBS_FIXED = B._C_OBS_FIXED
_LOG10DC_LO, _LOG10DC_HI = B._LOG10DC_LO, B._LOG10DC_HI
_SRX_S1 = B._SRX_S1

_PARAMS = ["log10_ne_03", "beta_gas", "p2", "r_max", "log10DC"]
_BOUNDS = np.array([
    [np.log10(_NE03_FID) + np.log10(_NORM_LO), np.log10(_NE03_FID) + np.log10(_NORM_HI)],
    [_BETA_GRID[0], _BETA_GRID[-1]],
    [_P2_GRID[0],   _P2_GRID[-1]],
    [_RMAX_GRID[0], _RMAX_GRID[-1]],
    [_LOG10DC_LO,   _LOG10DC_HI],
])


def _c_obs_total(sample):
    """AGN observed→data conversion for a sample (∝1/S^R_X, anchored on S1, same
    convention as the gas ``B._c_total``)."""
    srx = float(F.load_data(sample)["beckground"][0])
    return 10.0 ** _C_OBS_FIXED * _SRX_S1 / srx


def _make_full_gas(p2, r_max, beta_gas):
    """Full-APEC gas profiles (density+pressure+metallicity) at the fiducial
    density and the self-similar pressure calibration for this beta_gas."""
    beta_P = beta_gas + 2.0 / 3.0
    ne_03, P_03 = _calibrate_ne03_P03(beta_gas, beta_P, T_min=0.3, z=0.135)
    dp = _make_density_variant(model=2, ne_03=ne_03, beta=beta_gas,
                               alpha_in=_ALPHA_PROF, alpha_tr=2.0,
                               alpha_out=_ALPHA_PROF + 2.0 * float(p2))
    dp._r_max_factor = float(r_max)
    pp = _make_pressure_variant(model=2, P_03=P_03, beta=beta_P)
    return dp, pp, beta_P, ne_03


def _precompute(sample, hmf_backend, f_sys):
    """Build (or load) the per-sample full-APEC gas-shape grid + AGN template.

    gas_grid[i,j,k] = w_gas(theta) for (p2_i, r_max_j, beta_k) at the FIDUCIAL
    density (density_norm=1).  Built with the X_uk emulator: one FT per
    (p2, r_max) (``emissivity_xuk_per_z``), the beta axis via cheap
    ``x_uk_override`` re-evaluations.
    """
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, f"{sample}_emulator_fullapec.npz")
    data = F.load_data(sample)
    th_as = data["theta_arcsec"]; th_rad = data["theta_rad"]; wdata = data["wtheta"]
    err = np.sqrt(data["wtheta_err"] ** 2 + (f_sys * np.abs(wdata)) ** 2)
    mask = (th_as >= _THETA_MIN) & (th_as <= _THETA_MAX)

    if os.path.exists(cache):
        d = np.load(cache)
        # only reuse if the cached grid axes match the current globals
        if (d["p2_grid"].shape == _P2_GRID.shape and np.allclose(d["p2_grid"], _P2_GRID)
                and np.allclose(d["rmax_grid"], _RMAX_GRID)
                and np.allclose(d["beta_grid"], _BETA_GRID)):
            return d["gas_grid"], d["agn_dc1"], th_as, th_rad, wdata, err, mask, data
        print(f"  [{sample}] cached grid axes changed -> rebuilding", flush=True)

    th = F._THETA_COSMO
    pk = LinearPowerSpectrum()
    hmf = make_hmf(hmf_backend, pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"],
                Ob0=th["Omega_b"], sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    hod_params = B._build_hod_params(sample)
    z_arr, nz = F._build_nz_fast(sample)
    cool = ApecCoolingTable(emin=0.5, emax=2.0)
    mp = MetallicityProfileDPM()

    # AGN template (duty-cycle) at DC=1 — gas-independent.
    agn = DutyCycleAGNModel(sample=sample, theta_cosmo=th, hmf=hmf, log10DC=0.0)
    cross_a = HaloModelCrossSpectra(fhmp, density_profile=GasDensityDPM(model=2),
                                    agn_model=agn)
    comp = cross_a.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                 psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                 return_components=True, agn_kwargs={"log10DC": 0.0},
                                 n_workers=1)
    agn_dc1 = F._hankel(np.asarray(comp["agn"], dtype=float), th_rad)

    nth = th_as.size
    gas_grid = np.zeros((len(_P2_GRID), len(_RMAX_GRID), len(_BETA_GRID), nth))
    ncell = _P2_GRID.size * _RMAX_GRID.size * _BETA_GRID.size
    t0 = time.time(); done = 0
    for i, p2 in enumerate(_P2_GRID):
        for j, rmax in enumerate(_RMAX_GRID):
            for k, bg in enumerate(_BETA_GRID):
                # full-APEC profiles at THIS (p2, r_max, beta_gas); pressure tied
                # self-similarly (beta_P = beta_gas + 2/3), (n_e,0.3, P_03)
                # calibrated together so kT is physical.
                dp, pp, beta_P, ne_cal = _make_full_gas(p2, rmax, float(bg))
                cross = HaloModelCrossSpectra(fhmp, density_profile=dp)
                cross._dp = dp; cross._pp = pp; cross._mp = mp; cross._cooling_fn = cool
                # explicit FT at this cell, rescaled to the fiducial density so the
                # grid is at density_norm = 1 (w_gas SHAPE is ne-independent; the
                # (n_e/n_e_fid)² amplitude is applied analytically at fit time).
                xuk = cross.emissivity_xuk_per_z(z_arr, th, hod_params)
                xuk_fid = [x * (_NE03_FID / ne_cal) ** 2 for x in xuk]
                c = cross.angular_cl_gX(
                    F._ELL, z_arr, nz, th, hod_params,
                    psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                    x_uk_override=xuk_fid, return_components=True, n_workers=1)
                gas_grid[i, j, k] = F._hankel(np.asarray(c["gas"], dtype=float), th_rad)
                done += 1
            print(f"  [{sample}] p2={p2:.2f} r_max={rmax:.1f} "
                  f"({done}/{ncell} cells) [{time.time()-t0:.0f}s]", flush=True)

    np.savez(cache, gas_grid=gas_grid, agn_dc1=agn_dc1,
             p2_grid=_P2_GRID, rmax_grid=_RMAX_GRID, beta_grid=_BETA_GRID)
    print(f"[{sample}] full-APEC emulator built in {time.time()-t0:.0f}s -> {cache}",
          flush=True)
    return gas_grid, agn_dc1, th_as, th_rad, wdata, err, mask, data


def _anchor_c_total_S1(S):
    """Empirical full-APEC gas model->data anchor on S1.

    ``B._c_total`` was calibrated for the DENSITY-ONLY emissivity; the full-APEC
    ``emissivity_full_uk/Λ_ref`` grid is ~1e5× smaller AND the gap is
    temperature/shape-dependent (not a constant), so the density-only anchor rails
    the fit.  Re-anchor exactly as the baseline did for its -7.257 number: scan the
    gas-shape grid, solve the unconstrained best (A_gas, A_AGN) on S1, and DEFINE
    that A_gas as ``c_total(S1)`` (i.e. density_norm = 1 at the S1 best fit).  Other
    samples keep the data-side 1/S^R_X scaling.  Returns (c_total_S1, best_shape).
    """
    from scipy.optimize import lsq_linear
    interp, agn_dc1, wdata, err, mask = (S["interp"], S["agn_dc1"], S["wdata"],
                                         S["err"], S["mask"])
    w = 1.0 / err[mask]
    best = (np.inf, None, None)
    for p2 in _P2_GRID:
        for rmax in _RMAX_GRID:
            for bg in _BETA_GRID:
                gas = interp([[p2, rmax, bg]])[0]
                A = np.column_stack([gas[mask] * w, agn_dc1[mask] * w])
                res = lsq_linear(A, wdata[mask] * w, bounds=([0, 0], [np.inf, np.inf]),
                                 method="bvls")
                chi2 = float(np.sum((A @ res.x - wdata[mask] * w) ** 2))
                if chi2 < best[0]:
                    best = (chi2, float(res.x[0]), (float(p2), float(rmax), float(bg)))
    return best[1], best[2], best[0]


def _model_sample(p, interp, agn_dc1, c_total, c_obs_total):
    """w_model(theta) for one sample given the SHARED physical params p."""
    log10_ne_03, beta, p2, r_max, log10DC = p
    gas_shape = interp([[p2, r_max, beta]])[0]
    a_gas = c_total * (10.0 ** (log10_ne_03 - np.log10(_NE03_FID))) ** 2
    a_agn = 10.0 ** log10DC * c_obs_total
    return a_gas * gas_shape + a_agn * agn_dc1


def _chi2_sample(p, S):
    wm = _model_sample(p, S["interp"], S["agn_dc1"], S["c_total"], S["c_obs_total"])
    r = (wm - S["wdata"])[S["mask"]] / S["err"][S["mask"]]
    return float(np.sum(r ** 2))


def _neg_log_prob(p, samples):
    for v, (lo, hi) in zip(p, _BOUNDS):
        if not (lo <= v <= hi):
            return 1e30
    return 0.5 * sum(_chi2_sample(p, S) for S in samples.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", nargs="+", default=list(F.SAMPLES))
    ap.add_argument("--hmf", default="tinker08")
    ap.add_argument("--f-sys", type=float, default=0.05)
    ap.add_argument("--map-only", action="store_true")
    args = ap.parse_args(argv)

    samples = {}
    for s in args.samples:
        gas_grid, agn_dc1, th_as, th_rad, wdata, err, mask, data = _precompute(
            s, args.hmf, args.f_sys)
        interp = RegularGridInterpolator((_P2_GRID, _RMAX_GRID, _BETA_GRID), gas_grid,
                                         method="linear", bounds_error=False,
                                         fill_value=None)
        samples[s] = dict(interp=interp, agn_dc1=agn_dc1, th_as=th_as, wdata=wdata,
                          err=err, mask=mask, c_obs_total=_c_obs_total(s),
                          n_pts=int(mask.sum()), srx=float(data["beckground"][0]))
        print(f"[{s}] grid ready, n_pts={int(mask.sum())}", flush=True)

    # ---- empirical full-APEC gas anchor on S1 (density_norm=1 at the S1 best fit) ----
    anchor_sample = "S1" if "S1" in samples else args.samples[0]
    c_total_S1, best_shape, chi2_S1 = _anchor_c_total_S1(samples[anchor_sample])
    srx_anchor = samples[anchor_sample]["srx"]
    print(f"\nFull-APEC anchor on {anchor_sample}: c_total={c_total_S1:.3e} "
          f"(best shape p2/r_max/beta={best_shape}, unconstrained chi2={chi2_S1:.1f}); "
          f"density-only B._c_total={B._c_total(anchor_sample):.3e}", flush=True)
    for s, S in samples.items():
        # data-side 1/S^R_X scaling from the anchor sample (model norm is anchored)
        S["c_total"] = c_total_S1 * srx_anchor / S["srx"]

    n_tot = sum(S["n_pts"] for S in samples.values())
    print(f"\nJoint MAP over {len(samples)} samples, {n_tot} data points, "
          f"{len(_PARAMS)} shared params ...", flush=True)

    def nlp(p):
        return _neg_log_prob(p, samples)

    # multi-start Nelder-Mead over the 5 shared physical params
    starts = [
        [np.log10(_NE03_FID),        0.9, 2.0, 3.0, -2.0],
        [np.log10(_NE03_FID) + 0.3,  0.6, 1.5, 3.5, -1.5],
        [np.log10(_NE03_FID) - 0.3,  1.6, 2.4, 3.0, -2.5],
        [np.log10(_NE03_FID),        1.2, 3.0, 5.0, -1.0],
        [np.log10(_NE03_FID) + 0.5,  0.4, 1.5, 2.5, -2.0],
    ]
    best = None
    for q0 in starts:
        o = minimize(nlp, np.array(q0), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
        if best is None or o.fun < best.fun:
            best = o
        print(f"  start {np.round(q0,2)} -> chi2={2*o.fun:.1f}", flush=True)
    map_p = best.x
    chi2 = 2.0 * best.fun
    ndof = max(n_tot - len(_PARAMS), 1)
    map_dict = dict(zip(_PARAMS, [float(v) for v in map_p]))
    map_dict["density_norm"] = float(10.0 ** (map_p[0] - np.log10(_NE03_FID)))
    map_dict["chi2"] = chi2; map_dict["ndof"] = ndof; map_dict["chi2_per_dof"] = chi2 / ndof
    per_sample = {s: _chi2_sample(map_p, S) for s, S in samples.items()}
    map_dict["chi2_per_sample"] = {s: float(v) for s, v in per_sample.items()}

    os.makedirs(_OUT_DIR, exist_ok=True)
    out = os.path.join(_OUT_DIR, "joint_map.json")
    with open(out, "w") as fh:
        json.dump(map_dict, fh, indent=2)
    print("\n=== JOINT MAP ===")
    for k in _PARAMS:
        print(f"  {k:14s} = {map_dict[k]:+.4f}")
    print(f"  density_norm   = {map_dict['density_norm']:.3f}")
    print(f"  chi2/dof       = {chi2:.1f}/{ndof} = {chi2/ndof:.3f}")
    for s, v in per_sample.items():
        print(f"    {s}: chi2={v:.1f} ({samples[s]['n_pts']} pts)")
    print(f"\nSaved -> {out}", flush=True)
    return map_dict


if __name__ == "__main__":
    main()
