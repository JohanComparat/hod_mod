"""Baseline cross-correlation model: MAP + MCMC.

Baseline model (galaxy x soft-X-ray cross-correlation, S1)
----------------------------------------------------------
- galaxy: ZuMandelbaum15 occupation, FIXED at the wp+ngal MAP fit.
- AGN: duty-cycle AGN (:class:`DutyCycleAGNModel`, correlated, WITH the high-mass
  cutoff). Amplitude = duty cycle ``log10DC``.
- gas: GasDensityDPM with a free emission profile shape (Comparat 2025 Eq. 8:
  ``alpha_in=0.9, alpha_tr=2, alpha_out=0.9+2*p2``) and free mass slope / extent.

Free parameters (5):
    log10_A_gas, beta_gas, p2 (gas outer slope), r_max (r_max_over_r200), log10DC

    w_model(theta) = 10^log10_A_gas * gas_shape(p2, r_max, beta_gas)
                   + 10^(log10DC + C_obs) * agn_shape_dc1

Physical gas density (no free normalisation)
--------------------------------------------
With the true eROSITA TM0 ECF + cooling function + data background folded in
(``K_abs``, ``calibrate_kabs.py``), the fitted gas amplitude ``log10_A_gas`` is
reported as a PHYSICAL central density ``log10_ne_03_physical`` (and
``density_norm`` vs the X-ray-scaling-calibrated DPM fiducial): the gas leg
predicts observed counts from first principles, so the amplitude is the gas
density, not a free instrument-absorbing fudge.  The gas X-ray ECF is ~constant
over the fitted scales (hot-halo dominated), so this is a clean rescale and the
fit itself is unchanged.

Emulator
--------
Each likelihood needs an ``angular_cl_gX`` (~seconds), so an MCMC of ~3x10^5
evaluations is run through a fast emulator: the gas ``w(theta)`` shape is
precomputed on a grid over (p2, r_max, beta_gas) and trilinearly interpolated;
the AGN shape (DC=1, cutoff) is precomputed once.  The two amplitudes scale the
templates analytically, so each MAP/MCMC step is microseconds.  The flat priors
are bounded to the emulator grid.  ``C_obs`` (the AGN flux->observed-map
conversion, a w(theta)-only-undeterminable zero-point) is re-anchored so the
data-preferred AGN amplitude corresponds to the physical ``log10DC ~ -2``.

Run with:
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.fit_agn_duty_cycle_baseline --sample S1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, lsq_linear

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import GasDensityDPM
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.duty_cycle import load_zm15_map_params, DutyCycleAGNModel
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts.validate_gas_profiles import _make_density_variant

_OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "results", "agn_duty_cycle", "baseline")
)

# ---- emulator grid (gas emission profile shape) ----
_ALPHA_PROF = 0.9
_P2_GRID = np.array([0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.0])      # gas outer slope (Eq. 8)
_RMAX_GRID = np.array([2.0, 3.5, 5.0])                        # r_max_over_r200
_BETA_GRID = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])        # mass slope (widened)

_THETA_MIN, _THETA_MAX = 8.0, 300.0

# Free-parameter order + flat-prior bounds (bounded to the emulator grid).
# The AGN free parameter is the *observed* amplitude log10_A_AGN (robust; no
# fragile per-run C_obs anchoring).  The physical duty cycle is reported as
# log10DC = log10_A_AGN - _C_OBS_FIXED, with _C_OBS_FIXED the fixed (anchored)
# flux->observed-map conversion -- the single number a w(theta)-only fit cannot
# determine.  It is set so the data-preferred amplitude (~10^8.1) maps to the
# physical log10DC ~ -2 (AGN-host fraction of Comparat+2023/2025).
_C_OBS_FIXED = 10.1

# Physical duty-cycle prior: the AGN-host fraction is between 0.1% and 50%, so
# log10DC in [log10(0.001), log10(0.5)] = [-3.0, -0.301].  Since log10DC =
# log10_A_AGN - _C_OBS_FIXED, this bounds the fitted log10_A_AGN to
# [_C_OBS_FIXED - 3.0, _C_OBS_FIXED - 0.301] = [7.1, 9.799].
_LOG10DC_LO = float(np.log10(0.001))   # -3.0
_LOG10DC_HI = float(np.log10(0.5))     # -0.301

# Fiducial DPM model-2 central density [cm^-3].  With the true eROSITA TM0 ECF +
# cooling function + data background folded in (K_abs, scripts/fitting/
# calibrate_kabs.py), the fitted gas amplitude maps to a PHYSICAL central density
# (no free normalisation): log10_ne_03 = log10(ne_03^fid) + 0.5*(log10_A_gas -
# log10_A_gas^fid), where log10_A_gas^fid = log10(K_abs/ratio_mean) is the
# amplitude the fiducial-density gas produces.  The gas ECF is ~constant over the
# fitted scales (gas X-ray dominated by hot halos), so this is a clean rescale.
_NE03_FID = 4.87e-5

# The fit is parametrised by PHYSICAL quantities (no free normalisation factors):
#   - the gas central density  log10_ne_03  (replaces log10_A_gas)
#   - the AGN duty cycle        log10DC      (replaces log10_A_AGN)
# The model->data gas conversion C_total (= the amplitude the fiducial-density gas
# produces) is computed from first principles: anchored on S1 (where the fiducial
# DPM density reproduces the data, log10_A_gas^fid = -7.257) and scaled by the
# background S^R_X (C_total ∝ 1/S^R_X; the cooling Λ, the eROSITA TM0 ECF, and the
# model geometry are sample-independent).  Then A_gas = C_total*(ne_03/ne_03_fid)^2
# and A_AGN = 10^(log10DC + C_obs), so the linear amplitudes are still solved fast.
_LOG10_AGAS_FID_S1 = -7.257        # S1 gas amplitude at the fiducial density
_NORM_LO, _NORM_HI = 0.1, 10.0     # physical prior: n_e/n_e_fid in [0.1, 10]
_SRX_S1 = 8.6605e36                # S1 background S^R_X [erg kpc^-2 s^-1]


def _c_total(sample):
    """Universal model->data gas conversion for a sample: the amplitude the
    fiducial-density gas produces, = 10^(-7.257) * S^R_X(S1)/S^R_X(sample)."""
    srx = float(F.load_data(sample)["beckground"][0])
    return 10.0 ** _LOG10_AGAS_FID_S1 * _SRX_S1 / srx


def _log10_ne03(a_gas, c_total):
    """Physical gas central density from the solved gas amplitude + C_total."""
    return float(np.log10(_NE03_FID) + 0.5 * np.log10(max(a_gas, 1e-300) / c_total))


_PARAMS = ["log10_ne_03", "beta_gas", "p2", "r_max", "log10DC"]
_BOUNDS = np.array([
    [np.log10(_NE03_FID) + np.log10(_NORM_LO),     # log10_ne_03 (physical density,
     np.log10(_NE03_FID) + np.log10(_NORM_HI)],    #   n_e/n_e_fid in [0.1, 10])
    [_BETA_GRID[0], _BETA_GRID[-1]],   # beta_gas (gas mass slope)
    [_P2_GRID[0], _P2_GRID[-1]],       # p2 (gas outer slope)
    [_RMAX_GRID[0], _RMAX_GRID[-1]],   # r_max
    [_LOG10DC_LO, _LOG10DC_HI],        # log10DC (duty cycle, in [0.001, 0.5])
])


def _make_gas(p2, r_max):
    dp = _make_density_variant(model=2, alpha_in=_ALPHA_PROF, alpha_tr=2.0,
                               alpha_out=_ALPHA_PROF + 2.0 * float(p2))
    dp._r_max_factor = float(r_max)
    return dp


def _build_hod_params(sample):
    base = ZuMandelbaum15HODModel.default_params()
    base.update(load_zm15_map_params())
    base["log10m_star_thresh"] = float(F.SAMPLES[sample]["log10ms_min"])
    return base


def _reanchor_cobs(gas_grid, agn_dc1, wdata, err, mask):
    """Re-anchor the AGN flux->observed conversion C_obs at a gas profile where
    the duty-cycle AGN is *engaged* (p2=2.1, r_max=3.5, beta=0.9 — NOT the
    steepest gas, where the AGN is degenerate with the gas and fits to ~0).
    Fit (A_gas, A_AGN) there and set C_obs so log10DC = -2 maps to that
    data-preferred amplitude.
    """
    i0 = int(np.argmin(np.abs(_P2_GRID - 2.1)))
    j0 = int(np.argmin(np.abs(_RMAX_GRID - 3.5)))
    k0 = int(np.argmin(np.abs(_BETA_GRID - 0.9)))
    gas0 = gas_grid[i0, j0, k0]
    w = 1.0 / err[mask]
    A = np.column_stack([gas0[mask] * w, agn_dc1[mask] * w])
    res = lsq_linear(A, wdata[mask] * w, bounds=([0.0, 0.0], [np.inf, np.inf]),
                     method="bvls")
    a_agn = max(float(res.x[1]), 1e-300)
    return float(np.log10(a_agn)) + 2.0    # log10DC = log10(A_AGN) - C_obs = -2


def _precompute(sample, hmf_backend, f_sys):
    """Build (or load) the emulator: gas-shape grid + AGN template + C_obs."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, f"{sample}_emulator.npz")
    data = F.load_data(sample)
    th_as = data["theta_arcsec"]; th_rad = data["theta_rad"]
    wdata = data["wtheta"]
    err = np.sqrt(data["wtheta_err"] ** 2 + (f_sys * np.abs(wdata)) ** 2)
    mask = (th_as >= _THETA_MIN) & (th_as <= _THETA_MAX)

    if os.path.exists(cache):
        d = np.load(cache)
        # AGN free parameter is the observed amplitude log10_A_AGN, so the model
        # uses no internal conversion (c_obs = 0); the physical duty cycle is
        # reported afterwards via the fixed _C_OBS_FIXED.
        return (d["gas_grid"], d["agn_dc1"], 0.0,
                th_as, th_rad, wdata, err, mask, data)

    th = F._THETA_COSMO
    pk = LinearPowerSpectrum()
    hmf = make_hmf(hmf_backend, pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"],
                Ob0=th["Omega_b"], sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    hod_params = _build_hod_params(sample)
    z_arr, nz = F._build_nz_fast(sample)

    # AGN template (duty-cycle, cutoff) at DC=1 — gas-independent.
    agn = DutyCycleAGNModel(sample=sample, theta_cosmo=th, hmf=hmf, log10DC=0.0)
    cross_a = HaloModelCrossSpectra(fhmp, density_profile=GasDensityDPM(model=2),
                                    agn_model=agn)
    comp = cross_a.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                 psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                 return_components=True, agn_kwargs={"log10DC": 0.0},
                                 n_workers=1)
    agn_dc1 = F._hankel(np.asarray(comp["agn"], dtype=float), th_rad)

    # gas-shape grid over (p2, r_max, beta_gas)
    nth = th_as.size
    gas_grid = np.zeros((len(_P2_GRID), len(_RMAX_GRID), len(_BETA_GRID), nth))
    t0 = time.time()
    for i, p2 in enumerate(_P2_GRID):
        for j, rmax in enumerate(_RMAX_GRID):
            cross = HaloModelCrossSpectra(fhmp, density_profile=_make_gas(p2, rmax))
            for k, bg in enumerate(_BETA_GRID):
                c = cross.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                        psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                        beta_gas=float(bg), return_components=True,
                                        n_workers=1)
                gas_grid[i, j, k] = F._hankel(np.asarray(c["gas"], dtype=float), th_rad)
            print(f"  emulator p2={p2:.2f} r_max={rmax:.1f} "
                  f"({i*len(_RMAX_GRID)+j+1}/{len(_P2_GRID)*len(_RMAX_GRID)}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    c_obs = 0.0   # model uses observed AGN amplitude directly (see _C_OBS_FIXED)
    np.savez(cache, gas_grid=gas_grid, agn_dc1=agn_dc1, c_obs=c_obs,
             p2_grid=_P2_GRID, rmax_grid=_RMAX_GRID, beta_grid=_BETA_GRID)
    print(f"Emulator built in {time.time()-t0:.0f}s; C_obs={c_obs:.3f}. "
          f"Cached -> {cache}", flush=True)
    return gas_grid, agn_dc1, c_obs, th_as, th_rad, wdata, err, mask, data


def _model(p, interp, agn_dc1, c_total):
    """w_model(theta) for the PHYSICAL parameter vector
    p = [log10_ne_03, beta, p2, r_max, log10DC].  The gas amplitude is the physical
    density^2 times the model->data conversion C_total; the AGN amplitude is the
    duty cycle (A_AGN = 10^(log10DC + C_obs))."""
    log10_ne_03, beta, p2, r_max, log10DC = p
    gas_shape = interp([[p2, r_max, beta]])[0]              # (nth,)
    a_gas = c_total * (10.0 ** (log10_ne_03 - np.log10(_NE03_FID))) ** 2
    a_agn = 10.0 ** (log10DC + _C_OBS_FIXED)
    return a_gas * gas_shape + a_agn * agn_dc1


def _neg_log_prob(p, interp, agn_dc1, c_total, wdata, err, mask):
    for v, (lo, hi) in zip(p, _BOUNDS):
        if not (lo <= v <= hi):
            return 1e30
    wm = _model(p, interp, agn_dc1, c_total)
    return 0.5 * float(np.sum(((wm - wdata)[mask] / err[mask]) ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1", choices=list(F.SAMPLES))
    ap.add_argument("--hmf", default="tinker08")
    ap.add_argument("--f-sys", type=float, default=0.05)
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=5000)
    ap.add_argument("--nburn", type=int, default=1000)
    ap.add_argument("--map-only", action="store_true")
    args = ap.parse_args(argv)

    (gas_grid, agn_dc1, c_obs, th_as, th_rad, wdata, err, mask, data) = _precompute(
        args.sample, args.hmf, args.f_sys)
    interp = RegularGridInterpolator((_P2_GRID, _RMAX_GRID, _BETA_GRID), gas_grid,
                                     method="linear", bounds_error=False,
                                     fill_value=None)
    n_pts = int(mask.sum())
    c_total = _c_total(args.sample)              # model->data gas conversion (∝1/S^R_X)

    def nlp(p):
        return _neg_log_prob(p, interp, agn_dc1, c_total, wdata, err, mask)

    # ---- MAP: solve the two linear amplitudes analytically (bounded to the
    #          PHYSICAL density / duty-cycle priors), optimize only the 3 nonlinear
    #          shape params (beta, p2, r_max), multi-start. ----
    def _solve_amp(beta, p2, r_max):
        gas = interp([[p2, r_max, beta]])[0]
        w = 1.0 / err[mask]
        A = np.column_stack([gas[mask] * w, agn_dc1[mask] * w])
        lo_gas = c_total * _NORM_LO ** 2          # n_e/n_e_fid in [0.1, 10]
        hi_gas = c_total * _NORM_HI ** 2
        lo_agn = 10.0 ** (_C_OBS_FIXED + _LOG10DC_LO)   # duty cycle in [0.001, 0.5]
        hi_agn = 10.0 ** (_C_OBS_FIXED + _LOG10DC_HI)
        res = lsq_linear(A, wdata[mask] * w, bounds=([lo_gas, lo_agn], [hi_gas, hi_agn]),
                         method="bvls")
        chi2 = float(np.sum((A @ res.x - wdata[mask] * w) ** 2))
        return float(res.x[0]), float(res.x[1]), chi2

    def nlp3(q):
        beta, p2, r_max = q
        if not (_BOUNDS[1, 0] <= beta <= _BOUNDS[1, 1]
                and _BOUNDS[2, 0] <= p2 <= _BOUNDS[2, 1]
                and _BOUNDS[3, 0] <= r_max <= _BOUNDS[3, 1]):
            return 1e30
        return 0.5 * _solve_amp(beta, p2, r_max)[2]

    print("Running MAP (analytic amplitudes, multi-start) ...", flush=True)
    best3 = None
    for q0 in [(0.9, 2.0, 3.0), (0.6, 1.5, 3.0), (1.6, 2.4, 3.5),
               (0.3, 1.5, 5.0), (1.2, 3.0, 3.0), (2.0, 2.0, 3.0)]:
        o = minimize(nlp3, np.array(q0), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
        if best3 is None or o.fun < best3.fun:
            best3 = o
    beta_m, p2_m, rmax_m = best3.x
    a_gas, a_agn, chi2_map = _solve_amp(beta_m, p2_m, rmax_m)
    log10_ne_03 = _log10_ne03(a_gas, c_total)
    log10DC = float(np.log10(max(a_agn, 1e-300)) - _C_OBS_FIXED)
    map_p = np.array([log10_ne_03, beta_m, p2_m, rmax_m, log10DC])
    ndof = max(n_pts - len(_PARAMS), 1)
    map_dict = dict(zip(_PARAMS, [float(v) for v in map_p]))
    map_dict["density_norm"] = float(10.0 ** (log10_ne_03 - np.log10(_NE03_FID)))
    map_out = dict(sample=args.sample, model="duty_cycle+cutoff gas+AGN baseline",
                   high_mass_cutoff="fixed 14.0-14.3 (no AGN in clusters)",
                   free_params=_PARAMS, c_obs_fixed=_C_OBS_FIXED, c_total=c_total,
                   S_R_X=float(data["beckground"][0]), map=map_dict,
                   chi2=chi2_map, n_points=n_pts, ndof=ndof,
                   chi2_per_dof=chi2_map / ndof,
                   theta_min=_THETA_MIN, theta_max=_THETA_MAX, f_sys=args.f_sys)
    with open(os.path.join(_OUT_DIR, f"{args.sample}_baseline_map.json"), "w") as fh:
        json.dump(map_out, fh, indent=2)
    print(f"MAP: chi2/dof = {chi2_map:.1f}/{ndof} = {chi2_map/ndof:.3f}; "
          + ", ".join(f"{k}={v:.3f}" for k, v in map_dict.items()), flush=True)

    if args.map_only:
        return

    # ---- MCMC ----
    import emcee
    ndim = len(_PARAMS)
    nw = args.nwalkers
    rng = np.random.default_rng(42)
    p0 = map_p + 1e-3 * rng.standard_normal((nw, ndim)) * np.ptp(_BOUNDS, axis=1)
    p0 = np.clip(p0, _BOUNDS[:, 0] + 1e-6, _BOUNDS[:, 1] - 1e-6)

    def logp(p):
        v = nlp(p)
        return -v if v < 1e29 else -np.inf

    sampler = emcee.EnsembleSampler(nw, ndim, logp)
    print(f"Running MCMC: {nw} walkers x {args.nsteps} steps ...", flush=True)
    t0 = time.time()
    sampler.run_mcmc(p0, args.nsteps, progress=False)
    print(f"MCMC done in {time.time()-t0:.0f}s. "
          f"mean acceptance = {np.mean(sampler.acceptance_fraction):.2f}", flush=True)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        print(f"  autocorr times: {np.round(tau,0)}", flush=True)
    except Exception:
        tau = None

    chain_full = sampler.get_chain()            # (nsteps, nwalkers, ndim)
    lp_full = sampler.get_log_prob()            # (nsteps, nwalkers)
    flat = sampler.get_chain(discard=args.nburn, flat=True)
    logprob = sampler.get_log_prob(discard=args.nburn, flat=True)
    np.savez(os.path.join(_OUT_DIR, f"{args.sample}_baseline_chain.npz"),
             flatchain=flat, log_prob=logprob, chain=chain_full, lp=lp_full,
             params=_PARAMS, c_obs=c_obs, nburn=args.nburn)

    # posterior summary (median + 16/84) for the physical params; add density_norm
    pct = np.percentile(flat, [16, 50, 84], axis=0)
    summary = {p: dict(median=float(pct[1, i]),
                       lo=float(pct[1, i] - pct[0, i]),
                       hi=float(pct[2, i] - pct[1, i]))
               for i, p in enumerate(_PARAMS)}
    dn = 10.0 ** (flat[:, 0] - np.log10(_NE03_FID))     # density_norm = n_e/n_e_fid
    dnp = np.percentile(dn, [16, 50, 84])
    summary["density_norm"] = dict(median=float(dnp[1]),
                                   lo=float(dnp[1] - dnp[0]),
                                   hi=float(dnp[2] - dnp[1]))
    with open(os.path.join(_OUT_DIR, f"{args.sample}_baseline_summary.json"), "w") as fh:
        json.dump(dict(sample=args.sample, c_obs=c_obs, n_points=n_pts, ndof=ndof,
                       map=map_dict, chi2_per_dof_map=chi2_map / ndof,
                       acceptance=float(np.mean(sampler.acceptance_fraction)),
                       posterior=summary), fh, indent=2)
    print("Posterior (median +hi -lo):", flush=True)
    for i, p in enumerate(_PARAMS):
        print(f"  {p:12s} = {pct[1,i]:.3f}  +{pct[2,i]-pct[1,i]:.3f} "
              f"-{pct[1,i]-pct[0,i]:.3f}", flush=True)

    _figures(args.sample, flat, chain_full, lp_full, args.nburn, map_p,
             interp, agn_dc1, c_obs, th_as, th_rad, wdata, err, mask, data)
    print(f"Saved MAP/chain/summary/figures to {_OUT_DIR}", flush=True)


def _components(p, interp, agn_dc1, c_obs):
    gas = 10.0 ** p[0] * interp([[p[2], p[3], p[1]]])[0]
    agn = 10.0 ** (p[4] + c_obs) * agn_dc1
    return gas, agn, gas + agn


def _decomp_fig(path, title, th_as, wdata, wderr, err, gas, agn, total, mask,
                band=None, total_label="gas + AGN (MAP)"):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, axr):
        a.axvspan(th_as.min(), _THETA_MIN, color="0.93", zorder=0)
    if band is not None:
        ax.fill_between(th_as, band[0], band[1], color="C3", alpha=0.25,
                        label="model 16-84%")
    ax.errorbar(th_as, wdata, yerr=wderr, fmt="ko", ms=3, label="data (S1)", zorder=6)
    ax.plot(th_as, gas, "C0-", label="gas")
    ax.plot(th_as, agn, "C1--", label="AGN duty-cycle")
    ax.plot(th_as, total, "C3-", lw=2, label=total_label)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(title); ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), visible=False)
    # residual panel: pull (data - model)/sigma
    pull = (wdata - total) / err
    axr.axhspan(-1, 1, color="0.88"); axr.axhline(0, color="0.5", lw=0.8)
    axr.plot(th_as[mask], pull[mask], "ko", ms=3)
    axr.plot(th_as[~mask], pull[~mask], "o", color="0.6", ms=3, alpha=0.6)
    axr.set_xscale("log"); axr.set_xlabel(r"$\theta$ [arcsec]")
    axr.set_ylabel(r"$(d-m)/\sigma$"); axr.set_ylim(-5, 5)
    chi2 = float(np.sum(pull[mask] ** 2)); ndof = max(int(mask.sum()) - len(_PARAMS), 1)
    axr.text(0.02, 0.85, fr"$\chi^2/{{\rm dof}}={chi2/ndof:.2f}$",
             transform=axr.transAxes, fontsize=8, va="top")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _figures(sample, flat, chain, lp, nburn, map_p, interp, agn_dc1, c_obs,
             th_as, th_rad, wdata, err, mask, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner

    def out(n):
        return os.path.join(_OUT_DIR, f"{sample}_baseline_{n}.png")

    # (1) corner plot
    fig = corner.corner(flat, labels=_PARAMS, truths=list(map_p),
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt=".3f", title_kwargs=dict(fontsize=8))
    fig.suptitle(f"{sample}: baseline posterior  (duty-cycle AGN + gas)",
                 fontsize=11)
    fig.savefig(out("corner"), dpi=110); plt.close(fig)

    # (2) walker trace plots (convergence)
    ns, nw, nd = chain.shape
    fig, axs = plt.subplots(nd + 1, 1, figsize=(9, 1.5 * (nd + 1)), sharex=True)
    for i in range(nd):
        axs[i].plot(chain[:, :, i], color="k", alpha=0.12, lw=0.5)
        axs[i].set_ylabel(_PARAMS[i], fontsize=8)
        axs[i].axvline(nburn, color="C3", ls=":")
    axs[-1].plot(lp, color="k", alpha=0.12, lw=0.5)
    axs[-1].set_ylabel("log prob", fontsize=8); axs[-1].axvline(nburn, color="C3", ls=":")
    axs[-1].set_xlabel("step")
    axs[0].set_title(f"{sample}: MCMC traces ({nw} walkers; red = burn-in {nburn})",
                     fontsize=10)
    fig.tight_layout(); fig.savefig(out("trace"), dpi=110); plt.close(fig)

    # (3) MAP best-fit + residual panel
    gas, agn, tot = _components(map_p, interp, agn_dc1, c_obs)
    _decomp_fig(out("bestfit"), f"{sample}: baseline MAP fit (duty-cycle AGN + gas)",
                th_as, wdata, data["wtheta_err"], err, gas, agn, tot, mask,
                total_label="gas + AGN (MAP)")

    # (4) posterior-predictive: median + 16-84 band + residual panel
    idx = np.random.default_rng(0).choice(len(flat), size=min(400, len(flat)),
                                          replace=False)
    mods = np.array([_components(flat[i], interp, agn_dc1, c_obs)[2] for i in idx])
    band = np.percentile(mods, [16, 84], axis=0)
    med = np.percentile(mods, 50, axis=0)
    _decomp_fig(out("posterior_predictive"),
                f"{sample}: baseline posterior (median + 16-84%)",
                th_as, wdata, data["wtheta_err"], err, gas, agn, med, mask,
                band=band, total_label="posterior median")


if __name__ == "__main__":
    main()
