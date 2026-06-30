"""MAP fit of the galaxy x AGN+gas X-ray cross-correlation w(theta) with the
duty-cycle AGN model (agn_duty_cycle.DutyCycleAGNModel), for LS10-BGS S1.

Free parameters (all fit jointly):
    log10_A_gas, beta_gas, beta_pressure, log10DC

- ``log10_A_gas``                gas X-ray emissivity amplitude
- ``beta_gas``, ``beta_pressure``   DPM gas mass-slope tilts
- ``log10DC``                    AGN duty cycle (the only AGN parameter), bounded
                                 to the physical range [-4, -1]

The model is ``w = A_gas * gas_shape + 10^(log10DC + C_obs) * agn_shape_dc1``,
where ``agn_shape_dc1`` is the duty-cycle AGN cross-power at DC=1 and
``C_obs = _LOG10_AGN_OBS_CONV`` is the fixed AGN flux -> observed-map conversion.

Degeneracy note
---------------
The eRASS1 data w(theta) cross-correlates galaxies with the **background-
normalised** X-ray count-rate map (Lau+2025 Fig. 1, cts/s/arcsec^2), so it is
dimensionless.  A w(theta)-only fit cannot separate the duty cycle from the
absolute flux -> observed-map conversion (eROSITA energy-conversion factor /
mean background): both are linear amplitudes on the AGN cross-power.  We fix that
conversion (``_LOG10_AGN_OBS_CONV``, anchored so the data-preferred amplitude
maps to the physical log10DC ~ -2, the AGN-host fraction of Comparat+2023/2025)
and fit log10DC over its physical range.  Only log10DC's zero-point depends on
the anchor; the chi^2 does not.

Run with:
    python -m hod_mod.scripts.fitting.fit_agn_duty_cycle_cross --sample S1 --gas-model 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize, lsq_linear

from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.agn.duty_cycle import load_zm15_map_params, w_agn_path_for
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.paths import results_root

_OUT_DIR = os.path.normpath(
    os.path.join(results_root(), "agn_duty_cycle")
)

# Angular range of the data-validated fixed-ZM15 fit (<8" is inside the eROSITA
# PSF; >300" is noise-dominated).
_THETA_MIN_ARCSEC, _THETA_MAX_ARCSEC = 8.0, 300.0

# Physical duty-cycle bounds and gas-beta bounds.
_LOG10DC_LO, _LOG10DC_HI = -4.0, -1.0
_BETA_GAS_BOUNDS = (0.0, 1.0)
_BETA_P_BOUNDS = (0.0, 2.0)

# Fixed AGN flux -> observed-map conversion (log10).  Anchored so the
# data-preferred AGN cross-power amplitude (S1) corresponds to the physically-
# expected duty cycle log10DC = -2 (the AGN-host fraction of Comparat+2023/2025).
# This is the single number a w(theta)-only analysis cannot derive on its own;
# only the zero-point of log10DC depends on it, not the chi^2.
_LOG10_AGN_OBS_CONV = 9.61


def _build_galaxy_hod_params(label: str) -> dict:
    """Fixed ZuMandelbaum15 galaxy occupation from the wp+ngal MAP fit."""
    zm15 = load_zm15_map_params()
    base = ZuMandelbaum15HODModel.default_params()
    base.update(zm15)
    base["log10m_star_thresh"] = float(F.SAMPLES[label]["log10ms_min"])
    return base


def _solve_amplitudes(gas, agn_dc1, wdata, err):
    """Bounded linear least squares for (A_gas, A_AGN_obs).

    Model: w = A_gas*gas + A_AGN_obs*agn_dc1, with A_gas >= 0 and
    A_AGN_obs = 10^(log10DC + C_obs) bounded so log10DC in [_LOG10DC_LO,
    _LOG10DC_HI].  Returns (A_gas, A_AGN_obs, chi2).
    """
    w = 1.0 / np.asarray(err, dtype=float)
    A = np.column_stack([np.asarray(gas) * w, np.asarray(agn_dc1) * w])
    b = np.asarray(wdata, dtype=float) * w
    lo = 10.0 ** (_LOG10_AGN_OBS_CONV + _LOG10DC_LO)
    hi = 10.0 ** (_LOG10_AGN_OBS_CONV + _LOG10DC_HI)
    res = lsq_linear(A, b, bounds=([0.0, lo], [np.inf, hi]), method="bvls")
    a_gas, a_agn = float(res.x[0]), float(res.x[1])
    resid = A @ res.x - b
    return a_gas, a_agn, float(resid @ resid)


def fit_map(sample: str = "S1", gas_model: int = 2, f_sys: float = 0.05,
            hmf_backend: str = "tinker08",
            theta_min_arcsec: float = _THETA_MIN_ARCSEC,
            theta_max_arcsec: float = _THETA_MAX_ARCSEC,
            out_dir: str | None = None) -> dict:
    """Joint MAP fit of {log10_A_gas, beta_gas, beta_pressure, log10DC}."""
    out_dir = out_dir if out_dir is not None else _OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if gas_model != 2:
        print(f"  NOTE: gas_model={gas_model} requested; _Infrastructure uses "
              f"GasDensityDPM(model=2). Rebuild infra.dp to change.", flush=True)

    if not os.path.exists(w_agn_path_for(sample)):
        from hod_mod.agn.duty_cycle import compute_w_agn_kernel
        compute_w_agn_kernel(sample=sample)

    print(f"Building infrastructure (agn_model=duty_cycle, HMF={hmf_backend}) ...", flush=True)
    infra = F._Infrastructure(hmf_backend=hmf_backend, agn_model="duty_cycle")

    hod_params = _build_galaxy_hod_params(sample)
    data = F.load_data(sample)
    th_as = data["theta_arcsec"]
    wdata = data["wtheta"]
    err = np.sqrt(data["wtheta_err"] ** 2 + (f_sys * np.abs(wdata)) ** 2)
    mask = (th_as >= theta_min_arcsec) & (th_as <= theta_max_arcsec)
    wdata_m, err_m = wdata[mask], err[mask]

    _cache: dict = {}

    def _templates(beta_gas, beta_pressure):
        key = (round(beta_gas, 5), round(beta_pressure, 5))
        if key not in _cache:
            shapes = F._predict_shape(
                sample, infra, hod_params,
                beta_gas=beta_gas, beta_pressure=beta_pressure,
                agn_cheap={"log10DC": 0.0}, use_disk_cache=False,
            )
            _cache[key] = (np.asarray(shapes["gas"], dtype=float)[mask],
                           np.asarray(shapes["agn"], dtype=float)[mask])
        return _cache[key]

    def _objective(beta):
        bg = float(np.clip(beta[0], *_BETA_GAS_BOUNDS))
        bp = float(np.clip(beta[1], *_BETA_P_BOUNDS))
        gas, agn_dc1 = _templates(bg, bp)
        _, _, chi2 = _solve_amplitudes(gas, agn_dc1, wdata_m, err_m)
        return chi2

    x0 = np.array([F._BETA_GAS_DEFAULT, F._BETA_PRESSURE_DEFAULT])
    print(f"Starting MAP over (beta_gas, beta_pressure) from {x0}; "
          f"(A_gas, log10DC) solved analytically each step ...", flush=True)
    t0 = time.time()
    opt = minimize(_objective, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-3, maxiter=200))
    bg = float(np.clip(opt.x[0], *_BETA_GAS_BOUNDS))
    bp = float(np.clip(opt.x[1], *_BETA_P_BOUNDS))
    gas, agn_dc1 = _templates(bg, bp)
    a_gas, a_agn, chi2 = _solve_amplitudes(gas, agn_dc1, wdata_m, err_m)
    dt = time.time() - t0

    log10_A_gas = float(np.log10(max(a_gas, 1e-300)))
    log10DC = float(np.log10(max(a_agn, 1e-300))) - _LOG10_AGN_OBS_CONV
    n_pts = int(mask.sum())
    n_free = 4
    ndof = max(n_pts - n_free, 1)
    result = {
        "sample": sample,
        "agn_model": "duty_cycle",
        "gas_model": gas_model,
        "hmf_backend": hmf_backend,
        "free_params": ["log10_A_gas", "beta_gas", "beta_pressure", "log10DC"],
        "params": {
            "log10_A_gas": log10_A_gas,
            "beta_gas": bg,
            "beta_pressure": bp,
            "log10DC": log10DC,
        },
        "log10_AGN_obs_conv": _LOG10_AGN_OBS_CONV,
        "log10DC_bounds": [_LOG10DC_LO, _LOG10DC_HI],
        "galaxy_hod": "ZuMandelbaum15 (fixed, bgs_zm15_joint_wp_ngal/map_result.json)",
        "log10m_star_thresh": hod_params["log10m_star_thresh"],
        "theta_min_arcsec": theta_min_arcsec,
        "theta_max_arcsec": theta_max_arcsec,
        "chi2": chi2,
        "n_points": n_pts,
        "n_free": n_free,
        "ndof": ndof,
        "chi2_per_dof": chi2 / ndof,
        "f_sys": f_sys,
        "wall_time_s": dt,
        "success": bool(opt.success),
        "message": str(opt.message),
        "log10DC_at_bound": bool(log10DC <= _LOG10DC_LO + 1e-3
                                 or log10DC >= _LOG10DC_HI - 1e-3),
    }
    out_path = os.path.join(out_dir, f"{sample}_duty_cycle_map.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nMAP done in {dt:.1f}s (chi2/dof = {chi2:.2f}/{ndof} = {chi2/ndof:.3f}; "
          f"log10_A_gas={log10_A_gas:.3f}, beta_gas={bg:.3f}, "
          f"beta_pressure={bp:.3f}, log10DC={log10DC:.3f})", flush=True)
    print(f"Saved {out_path}", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1", choices=list(F.SAMPLES))
    ap.add_argument("--gas-model", type=int, default=2, choices=[1, 2, 3],
                    help="DPM gas density model (default 2; current infra uses 2).")
    ap.add_argument("--mode", default="map", choices=["map"],
                    help="Only MAP is implemented in this lean driver.")
    ap.add_argument("--hmf", default="tinker08",
                    help="HMF backend (default tinker08; 'csst' for the emulator).")
    ap.add_argument("--f-sys", type=float, default=0.05,
                    help="Fractional systematic error floor (default 0.05).")
    ap.add_argument("--theta-min-arcsec", type=float, default=_THETA_MIN_ARCSEC)
    ap.add_argument("--theta-max-arcsec", type=float, default=_THETA_MAX_ARCSEC)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    fit_map(sample=args.sample, gas_model=args.gas_model, f_sys=args.f_sys,
            hmf_backend=args.hmf, theta_min_arcsec=args.theta_min_arcsec,
            theta_max_arcsec=args.theta_max_arcsec, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
