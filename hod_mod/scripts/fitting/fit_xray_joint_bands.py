"""Phase B — energy-band (temperature-resolved) joint galaxy×X-ray w(theta) fit.

Extends the broad-band joint fit (``fit_xray_joint``) to the 15 narrow energy bands
(0.5-0.6 ... 1.9-2.0 keV).  The gas emissivity ``ε_b = n_e²·Λ_b(T,Z)`` is
band-dependent through the per-band APEC cooling ``Λ_b``, so the **band RATIOS
constrain the gas temperature kT** — the freedom the broad-band fit lacked (where
S6/S7, the cluster-mass samples, failed even individually).  All samples are fit
JOINTLY with ONE shared set of physical scaling-relation parameters; the new free
parameter vs Phase A is ``kT_norm`` (the temperature-normalisation of the kT-M
relation), constrained by the band ratios.

The band data (reconstructed + validated by ``reconstruct_band_wtheta``) are read
from ``$HOD_MOD_DATA_DIR/xray_bands/<basename>/<band>.fits`` (env-var data link via
``hod_mod.paths.data_path``), falling back to the in-repo ``hod_mod/data``.

Speed: the 15 bands share n_e/T/Z and the FT geometry, so each cell's emissivity FT
is built ONCE for all bands (``emissivity_full_uk_bands`` →
``emissivity_xuk_bands_per_z``); the per-band Limber+Hankel reuse the cached HOD
weights via ``x_uk_override``.

Usage (after the band-data move is complete):
    HOD_MOD_DATA_DIR=<data root> JAX_PLATFORMS=cpu python -m \
        hod_mod.scripts.fitting.fit_xray_joint_bands --samples S1 S2 S3 S4 S5 S6 S7 --map-only
    # quick smoke test:
    ... fit_xray_joint_bands --samples S1 --grid-tiny --map-only
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, lsq_linear

from hod_mod import paths
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
from hod_mod.scripts.fitting import fit_xray_joint as J
from hod_mod.scripts.validate_gas_profiles import (
    _make_density_variant, _make_pressure_variant, _calibrate_ne03_P03,
)

_OUT_DIR = os.fspath(paths.results_root() / "xray_joint_bands")

# 15 bands (folder names in eV) + their keV edges
_BANDS = [f"{lo:04d}_E_{lo+100:04d}" for lo in range(500, 2000, 100)]
_BAND_EDGES = [(lo / 1000.0, (lo + 100) / 1000.0) for lo in range(500, 2000, 100)]
_NB = len(_BANDS)

# emulator grid: gas shape (p2, r_max, beta_gas) + the NEW temperature axis kT_norm
_ALPHA_PROF = 0.9
_P2_GRID   = np.array([0.3, 1.0, 2.4])
_RMAX_GRID = np.array([3.0, 4.0, 5.0])
_BETA_GRID = np.array([0.9, 1.5, 2.1])
_KT_GRID   = np.array([0.5, 1.0, 2.0])       # multiplies the self-similar kT

_THETA_MIN, _THETA_MAX = 8.0, 300.0
_NE03_FID = B._NE03_FID
_NORM_LO, _NORM_HI = B._NORM_LO, B._NORM_HI
_LOG10DC_LO, _LOG10DC_HI = B._LOG10DC_LO, B._LOG10DC_HI
_GAMMA_AGN = 1.8                              # AGN photon index for the band split

_PARAMS = ["log10_ne_03", "kT_norm", "beta_gas", "p2", "r_max", "log10DC"]


def _grids(tiny=False):
    if tiny:
        return (np.array([0.3, 2.4]), np.array([3.0, 5.0]),
                np.array([0.9, 2.1]), np.array([0.5, 2.0]))
    return _P2_GRID, _RMAX_GRID, _BETA_GRID, _KT_GRID


# --- band data + cooling + AGN spectral split ------------------------------

def _basename(label):
    return F._zenodo_fname(label).name.replace("_GALxEVT_wtheta.fits", "")


def load_band_data(label):
    """Per-band reconstructed w_b(theta) for a sample.

    Returns dict with theta_arcsec/theta_rad (deg→) and (Nb, Ntheta) wtheta/err.
    Reads ``$HOD_MOD_DATA_DIR/xray_bands/<basename>/<band>.fits``.
    """
    base = _basename(label)
    root = paths.data_path("xray_bands", base)
    w = np.zeros((_NB, 0)); e = np.zeros((_NB, 0)); th_deg = None
    rows = []
    for band in _BANDS:
        fp = os.fspath(root / (band + ".fits"))
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"missing band file: {fp}\n"
                                    f"set $HOD_MOD_DATA_DIR to the moved data root.")
        t = Table.read(fp)
        if th_deg is None:
            th_deg = np.asarray(t["theta_mid"], float)
        rows.append((np.asarray(t["wtheta"], float), np.asarray(t["wtheta_err"], float)))
    w = np.vstack([r[0] for r in rows]); e = np.vstack([r[1] for r in rows])
    return dict(theta_deg=th_deg, theta_arcsec=th_deg * 3600.0,
                theta_rad=th_deg * np.pi / 180.0, wtheta=w, wtheta_err=e)


_COOLING_CACHE = None
def _band_cooling():
    """15 per-band ApecCoolingTable instances (built once, ~150 s)."""
    global _COOLING_CACHE
    if _COOLING_CACHE is None:
        _COOLING_CACHE = [ApecCoolingTable(emin=lo, emax=hi) for lo, hi in _BAND_EDGES]
    return _COOLING_CACHE


def _agn_band_fractions(gamma=_GAMMA_AGN):
    """Energy-flux fraction of a Γ power-law AGN spectrum in each band:
    f_b = ∫_b E^{1-Γ}dE / ∫_{0.5}^{2.0} E^{1-Γ}dE  (Σ f_b = 1).
    First-pass AGN spectral split; a full spectrum×eROSITA-response-per-band AGN
    is a later refinement."""
    p = 2.0 - gamma   # exponent of the antiderivative E^{2-Γ}/(2-Γ); energy flux ∝ E^{1-Γ}
    def _integ(lo, hi):
        if abs(p) < 1e-9:
            return np.log(hi / lo)
        return (hi ** p - lo ** p) / p
    tot = _integ(0.5, 2.0)
    return np.array([_integ(lo, hi) / tot for lo, hi in _BAND_EDGES])


def _make_full_gas_kT(p2, r_max, beta_gas, kT_norm):
    """Full-APEC gas profiles at (p2, r_max, beta_gas) with the temperature scaled
    by kT_norm (P_03 ×= kT_norm so T = P/n_e ∝ kT_norm); density at the calibrated
    n_e (rescaled to the fiducial later)."""
    beta_P = beta_gas + 2.0 / 3.0
    ne_cal, P_cal = _calibrate_ne03_P03(beta_gas, beta_P, T_min=0.3, z=0.135)
    dp = _make_density_variant(model=2, ne_03=ne_cal, beta=beta_gas,
                               alpha_in=_ALPHA_PROF, alpha_tr=2.0,
                               alpha_out=_ALPHA_PROF + 2.0 * float(p2))
    dp._r_max_factor = float(r_max)
    pp = _make_pressure_variant(model=2, P_03=P_cal * float(kT_norm), beta=beta_P)
    return dp, pp, ne_cal


# --- emulator precompute ----------------------------------------------------

def _precompute(sample, hmf_backend, tiny):
    """Build (or load) the per-sample band emulator: gas w_b(theta) over
    (p2, r_max, beta_gas, kT_norm) for all 15 bands at density_norm=1, plus the
    broad-band AGN template (DC=1)."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    suff = "_tiny" if tiny else ""
    cache = os.path.join(_OUT_DIR, f"{sample}_bands_emulator{suff}.npz")
    p2g, rg, bg_, ktg = _grids(tiny)

    bd = load_band_data(sample)
    th_as = bd["theta_arcsec"]; th_rad = bd["theta_rad"]
    mask = (th_as >= _THETA_MIN) & (th_as <= _THETA_MAX)

    if os.path.exists(cache):
        d = np.load(cache)
        if (np.array_equal(d["p2_grid"], p2g) and np.array_equal(d["rmax_grid"], rg)
                and np.array_equal(d["beta_grid"], bg_) and np.array_equal(d["kt_grid"], ktg)):
            return d["gas_grid"], d["agn_dc1"], bd, mask
        print(f"  [{sample}] cached band grid axes changed -> rebuilding", flush=True)

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
    cool = _band_cooling()
    mp = MetallicityProfileDPM()

    # broad-band AGN template (DC=1) — split into bands at fit time via f_b
    agn = DutyCycleAGNModel(sample=sample, theta_cosmo=th, hmf=hmf, log10DC=0.0)
    cross_a = HaloModelCrossSpectra(fhmp, density_profile=GasDensityDPM(model=2),
                                    agn_model=agn)
    comp = cross_a.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                 psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                 return_components=True, agn_kwargs={"log10DC": 0.0},
                                 n_workers=1)
    agn_dc1 = F._hankel(np.asarray(comp["agn"], float), th_rad)

    nth = th_as.size
    gas_grid = np.zeros((p2g.size, rg.size, bg_.size, ktg.size, _NB, nth))
    ncell = p2g.size * rg.size * bg_.size * ktg.size
    t0 = time.time(); done = 0
    for i, p2 in enumerate(p2g):
        for j, rmax in enumerate(rg):
            for kk, beta in enumerate(bg_):
                for l, kt in enumerate(ktg):
                    dp, pp, ne_cal = _make_full_gas_kT(p2, rmax, float(beta), float(kt))
                    cross = HaloModelCrossSpectra(fhmp, density_profile=dp)
                    cross._dp = dp; cross._pp = pp; cross._mp = mp
                    # ONE batched FT for all 15 bands, per z
                    xukb = cross.emissivity_xuk_bands_per_z(z_arr, th, hod_params, cool)
                    scale = (_NE03_FID / ne_cal) ** 2
                    for b in range(_NB):
                        ov = [scale * xukb[iz][b] for iz in range(len(z_arr))]
                        c = cross.angular_cl_gX(
                            F._ELL, z_arr, nz, th, hod_params,
                            psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                            x_uk_override=ov, return_components=True, n_workers=1)
                        gas_grid[i, j, kk, l, b] = F._hankel(
                            np.asarray(c["gas"], float), th_rad)
                    done += 1
            print(f"  [{sample}] p2={p2:.2f} r_max={rmax:.1f} "
                  f"({done}/{ncell} cells) [{time.time()-t0:.0f}s]", flush=True)

    np.savez(cache, gas_grid=gas_grid, agn_dc1=agn_dc1,
             p2_grid=p2g, rmax_grid=rg, beta_grid=bg_, kt_grid=ktg)
    print(f"[{sample}] band emulator built in {time.time()-t0:.0f}s -> {cache}", flush=True)
    return gas_grid, agn_dc1, bd, mask


# --- model + objective ------------------------------------------------------

def _model_bands(p, S):
    """(Nb, Ntheta) model w_b(theta) for shared params p on sample dict S."""
    log10_ne_03, kT_norm, beta, p2, r_max, log10DC = p
    gas = S["interp"]([[p2, r_max, beta, kT_norm]])[0].reshape(_NB, -1)   # (Nb, Nth)
    a_gas = S["c_total"] * (10.0 ** (log10_ne_03 - np.log10(_NE03_FID))) ** 2
    a_agn = 10.0 ** log10DC * S["c_obs_total"]
    return a_gas * gas + a_agn * (S["fb"][:, None] * S["agn_dc1"][None, :])


def _chi2_sample(p, S):
    wm = _model_bands(p, S)
    r = (wm - S["wtheta"])[:, S["mask"]] / S["err"][:, S["mask"]]
    return float(np.sum(r ** 2))


def _bounds():
    return np.array([
        [np.log10(_NE03_FID) + np.log10(_NORM_LO), np.log10(_NE03_FID) + np.log10(_NORM_HI)],
        [_KT_GRID[0], _KT_GRID[-1]],
        [_BETA_GRID[0], _BETA_GRID[-1]],
        [_P2_GRID[0], _P2_GRID[-1]],
        [_RMAX_GRID[0], _RMAX_GRID[-1]],
        [_LOG10DC_LO, _LOG10DC_HI],
    ])


def _anchor_c_total_S1(S):
    """Empirical full-APEC gas anchor on S1 using the BAND-SUMMED amplitude (free
    A_gas, A_AGN at the best shape, kT), defining that A_gas as c_total(S1)."""
    bnds = _bounds()
    w = 1.0 / S["err"][:, S["mask"]].ravel()
    best = (np.inf, None)
    p2g, rg, bg_, ktg = S["axes"]
    for p2 in p2g:
        for rmax in rg:
            for beta in bg_:
                for kt in ktg:
                    gas = S["interp"]([[p2, rmax, beta, kt]])[0].reshape(_NB, -1)
                    g = gas[:, S["mask"]].ravel() * w
                    a = (S["fb"][:, None] * S["agn_dc1"][None, :])[:, S["mask"]].ravel() * w
                    A = np.column_stack([g, a])
                    res = lsq_linear(A, S["wtheta"][:, S["mask"]].ravel() * w,
                                     bounds=([0, 0], [np.inf, np.inf]), method="bvls")
                    chi2 = float(np.sum((A @ res.x - S["wtheta"][:, S["mask"]].ravel() * w) ** 2))
                    if chi2 < best[0]:
                        best = (chi2, float(res.x[0]))
    return best[1], best[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", nargs="+", default=list(F.SAMPLES))
    ap.add_argument("--hmf", default="tinker08")
    ap.add_argument("--f-sys", type=float, default=0.05)
    ap.add_argument("--grid-tiny", action="store_true", help="2^4 grid smoke test")
    ap.add_argument("--map-only", action="store_true")
    args = ap.parse_args(argv)

    p2g, rg, bg_, ktg = _grids(args.grid_tiny)
    fb = _agn_band_fractions()
    samples = {}
    for s in args.samples:
        gas_grid, agn_dc1, bd, mask = _precompute(s, args.hmf, args.grid_tiny)
        err = np.sqrt(bd["wtheta_err"] ** 2 + (args.f_sys * np.abs(bd["wtheta"])) ** 2)
        # interp over (p2,r_max,beta,kT); values flattened over (band,theta)
        vals = gas_grid.reshape(p2g.size, rg.size, bg_.size, ktg.size, -1)
        interp = RegularGridInterpolator((p2g, rg, bg_, ktg), vals,
                                         method="linear", bounds_error=False, fill_value=None)
        samples[s] = dict(interp=interp, agn_dc1=agn_dc1, wtheta=bd["wtheta"], err=err,
                          mask=mask, fb=fb, c_obs_total=J._c_obs_total(s),
                          srx=float(F.load_data(s)["beckground"][0]),
                          axes=(p2g, rg, bg_, ktg),
                          n_pts=int(mask.sum()) * _NB)
        print(f"[{s}] band grid ready, n_pts={samples[s]['n_pts']}", flush=True)

    anchor_sample = "S1" if "S1" in samples else args.samples[0]
    c_total_S1, chi2_S1 = _anchor_c_total_S1(samples[anchor_sample])
    srx_anchor = samples[anchor_sample]["srx"]
    print(f"\nBand anchor on {anchor_sample}: c_total={c_total_S1:.3e} "
          f"(unconstrained band-summed chi2={chi2_S1:.1f})", flush=True)
    for s, S in samples.items():
        S["c_total"] = c_total_S1 * srx_anchor / S["srx"]

    n_tot = sum(S["n_pts"] for S in samples.values())
    print(f"\nJoint BAND MAP over {len(samples)} samples × {_NB} bands, {n_tot} pts, "
          f"{len(_PARAMS)} shared params ...", flush=True)
    bnds = _bounds()

    def nlp(p):
        for v, (lo, hi) in zip(p, bnds):
            if not (lo <= v <= hi):
                return 1e30
        return 0.5 * sum(_chi2_sample(p, S) for S in samples.values())

    starts = [
        [np.log10(_NE03_FID),       1.0, 1.5, 1.0, 4.0, -1.8],
        [np.log10(_NE03_FID) + 0.3, 0.5, 0.9, 0.3, 5.0, -1.5],
        [np.log10(_NE03_FID) - 0.3, 2.0, 2.1, 2.4, 3.0, -2.2],
        [np.log10(_NE03_FID),       1.5, 1.2, 1.0, 4.0, -1.0],
    ]
    best = None
    for q0 in starts:
        o = minimize(nlp, np.array(q0), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=6000))
        if best is None or o.fun < best.fun:
            best = o
        print(f"  start {np.round(q0,2)} -> chi2={2*o.fun:.1f}", flush=True)

    map_p = best.x; chi2 = 2.0 * best.fun
    ndof = max(n_tot - len(_PARAMS), 1)
    out = dict(zip(_PARAMS, [float(v) for v in map_p]))
    out["density_norm"] = float(10.0 ** (map_p[0] - np.log10(_NE03_FID)))
    out["chi2"] = chi2; out["ndof"] = ndof; out["chi2_per_dof"] = chi2 / ndof
    out["chi2_per_sample"] = {s: float(_chi2_sample(map_p, S)) for s, S in samples.items()}
    os.makedirs(_OUT_DIR, exist_ok=True)
    outf = os.path.join(_OUT_DIR, "joint_bands_map.json")
    with open(outf, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== JOINT BAND MAP ===")
    for k in _PARAMS:
        print(f"  {k:14s} = {out[k]:+.4f}")
    print(f"  density_norm   = {out['density_norm']:.3f}")
    print(f"  chi2/dof       = {chi2:.1f}/{ndof} = {chi2/ndof:.3f}")
    for s, v in out["chi2_per_sample"].items():
        print(f"    {s}: chi2={v:.1f} ({samples[s]['n_pts']} pts)")
    print(f"\nSaved -> {outf}", flush=True)
    return out


if __name__ == "__main__":
    main()
