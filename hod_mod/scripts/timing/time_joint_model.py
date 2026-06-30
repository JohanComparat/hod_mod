"""Wall-clock timing benchmark for wp, ESD, and galaxy×X-ray predictions.

Measures the time for each component of the joint WPRP + ESD + galaxy×X-ray
model, both in serial and parallel (threaded z-loop) mode.  Results are stored
as JSON so they can be included in the documentation.

Usage::

    python -m hod_mod.scripts.timing.time_joint_model
    python -m hod_mod.scripts.timing.time_joint_model --sample S1 --n-repeat 3

Output
------
results/timing/timing_joint_model.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import j0

# ---------------------------------------------------------------------------
# hod_mod imports
# ---------------------------------------------------------------------------
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import GasDensityDPM
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.xray import XrayAGNModel
from hod_mod.paths import results_root

# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------
_COLOSSUS = {
    "flat": True, "H0": 67.74, "Om0": 0.3089,
    "Ob0": 0.0486, "sigma8": 0.811, "ns": 0.9667,
}
_THETA_COSMO = {
    "h": 0.6774, "Omega_m": 0.3089, "Omega_b": 0.0486,
    "n_s": 0.9667, "sigma8": 0.811, "w0": -1.0, "wa": 0.0,
}

# ---------------------------------------------------------------------------
# Sample definitions (same as fit_comparat2025.py)
# ---------------------------------------------------------------------------
SAMPLES = {
    "S1": dict(log10ms_min=10.00, zmax=0.18, zmean=0.135, N=2759238),
    "S3": dict(log10ms_min=10.50, zmax=0.26, zmean=0.191, N=3263228),
    "S5": dict(log10ms_min=11.00, zmax=0.35, zmean=0.252, N=1619838),
    "S7": dict(log10ms_min=11.50, zmax=0.35, zmean=0.261, N=120882),
}

# More+2015 HOD defaults
_MORE_HOD = {
    "log10mmin": 12.2, "sigma_logm": 0.45,
    "log10m1": 13.4,   "alpha": 1.05, "kappa": 1.0,
}

_N_ELL   = 80
_ELL_ARR = np.logspace(1.0, 4.3, _N_ELL)
_PSF_FWHM = 30.0

_RESULTS_DIR = results_root() / "timing"


def _build_nz(label: str, n_pts: int = 5) -> tuple[np.ndarray, np.ndarray]:
    s  = SAMPLES[label]
    z  = s["zmean"]
    dz = min(0.02, s["zmax"] * 0.10)
    z_arr = np.linspace(max(0.01, z - 2.0 * dz), z + 2.0 * dz, n_pts)
    nz    = np.exp(-0.5 * ((z_arr - z) / dz) ** 2)
    return z_arr, nz / trapezoid(nz, z_arr)


def _rp_arr():
    return np.logspace(-1, 1.8, 20)    # 20 bins 0.1–63 Mpc/h


def _R_arr():
    return np.logspace(-1, 1.5, 16)    # 16 bins 0.1–32 Mpc/h


def _time_call(fn, *args, n_repeat: int = 1, **kwargs) -> tuple[float, object]:
    """Run fn(*args, **kwargs) n_repeat times; return (mean_wall_s, last_result)."""
    result = None
    t_total = 0.0
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t_total += time.perf_counter() - t0
    return t_total / n_repeat, result


def run_timing(label: str = "S1", n_repeat: int = 1) -> dict:
    s      = SAMPLES[label]
    z_eff  = s["zmean"]
    z_arr, nz_g = _build_nz(label)
    rp = _rp_arr()
    R  = _R_arr()

    results: dict = {
        "sample": label,
        "log10ms_min": s["log10ms_min"],
        "z_mean": z_eff,
        "n_z": len(z_arr),
        "n_ell": _N_ELL,
        "n_rp": len(rp),
        "n_R": len(R),
        "n_repeat": n_repeat,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }

    # ------------------------------------------------------------------
    # Phase A: Infrastructure build
    # ------------------------------------------------------------------
    print("Phase A: infrastructure build ...", flush=True)
    t0 = time.perf_counter()
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("csst")
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    fhmp   = FullHaloModelPrediction(pk_lin, hod, hp)
    dp     = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=200)
    agn    = XrayAGNModel()
    cross  = HaloModelCrossSpectra(fhmp, density_profile=dp, agn_model=agn)
    t_infra = time.perf_counter() - t0
    results["infra_build_s"] = round(t_infra, 2)
    print(f"  done: {t_infra:.1f} s", flush=True)

    # ------------------------------------------------------------------
    # Phase B: wp (JAX JIT warmup on first call)
    # ------------------------------------------------------------------
    print("Phase B: wp(rp) warmup ...", flush=True)
    _ = fhmp.wp(rp, pi_max=100.0, z=z_eff, theta_cosmo=_THETA_COSMO,
                hod_params=_MORE_HOD)
    print("  (JIT warmup done)", flush=True)

    t_wp, _ = _time_call(
        fhmp.wp, rp, pi_max=100.0, z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=_MORE_HOD,
        n_repeat=n_repeat,
    )
    results["wp_s"] = round(t_wp, 3)
    print(f"  wp: {t_wp:.3f} s  (mean of {n_repeat})", flush=True)

    # ------------------------------------------------------------------
    # Phase C: ESD (delta_sigma)
    # ------------------------------------------------------------------
    print("Phase C: delta_sigma(R) ...", flush=True)
    _ = fhmp.delta_sigma(R, z=z_eff, theta_cosmo=_THETA_COSMO,
                         hod_params=_MORE_HOD)
    t_esd, _ = _time_call(
        fhmp.delta_sigma, R, z=z_eff,
        theta_cosmo=_THETA_COSMO, hod_params=_MORE_HOD,
        n_repeat=n_repeat,
    )
    results["esd_s"] = round(t_esd, 3)
    print(f"  ESD: {t_esd:.3f} s  (mean of {n_repeat})", flush=True)

    # ------------------------------------------------------------------
    # Phase D: wp + ESD combined (as in one joint likelihood evaluation)
    # ------------------------------------------------------------------
    def _wp_esd():
        fhmp.wp(rp, pi_max=100.0, z=z_eff,
                theta_cosmo=_THETA_COSMO, hod_params=_MORE_HOD)
        fhmp.delta_sigma(R, z=z_eff,
                         theta_cosmo=_THETA_COSMO, hod_params=_MORE_HOD)

    t_wpesd, _ = _time_call(_wp_esd, n_repeat=n_repeat)
    results["wp_esd_s"] = round(t_wpesd, 3)
    print(f"  wp + ESD: {t_wpesd:.3f} s  (mean of {n_repeat})", flush=True)

    # ------------------------------------------------------------------
    # Phase E: angular_cl_gX — serial (n_workers=1)
    # ------------------------------------------------------------------
    print(f"Phase E: angular_cl_gX SERIAL  (n_z={len(z_arr)}, n_ell={_N_ELL}) ...",
          flush=True)
    t_clgX_serial, _ = _time_call(
        cross.angular_cl_gX,
        _ELL_ARR, z_arr, nz_g, _THETA_COSMO, _MORE_HOD,
        psf_fwhm_arcsec=_PSF_FWHM,
        return_components=True,
        n_workers=1,
        n_repeat=1,   # always 1 — this takes ~900 s
    )
    results["angular_cl_gX_serial_s"] = round(t_clgX_serial, 1)
    print(f"  serial: {t_clgX_serial:.1f} s", flush=True)

    # ------------------------------------------------------------------
    # Phase E': angular_cl_gX — parallel (all CPUs)
    # ------------------------------------------------------------------
    ncpu = os.cpu_count()
    print(f"Phase E': angular_cl_gX PARALLEL  ({ncpu} workers) ...", flush=True)
    t_clgX_par, cl_components = _time_call(
        cross.angular_cl_gX,
        _ELL_ARR, z_arr, nz_g, _THETA_COSMO, _MORE_HOD,
        psf_fwhm_arcsec=_PSF_FWHM,
        return_components=True,
        n_workers=-1,
        n_repeat=1,
    )
    results["angular_cl_gX_parallel_s"] = round(t_clgX_par, 1)
    speedup = t_clgX_serial / t_clgX_par if t_clgX_par > 0 else float("nan")
    results["speedup_parallel"] = round(speedup, 2)
    print(f"  parallel: {t_clgX_par:.1f} s  (×{speedup:.1f} speedup)", flush=True)

    # ------------------------------------------------------------------
    # Phase F: full joint evaluation (serial)
    # ------------------------------------------------------------------
    results["joint_serial_s"]   = round(t_wpesd + t_clgX_serial, 1)
    results["joint_parallel_s"] = round(t_wpesd + t_clgX_par,    1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"  Timing summary — sample {label}")
    print("=" * 55)
    rows = [
        ("Infrastructure build",     results["infra_build_s"],          "s"),
        ("wp(rp)  [More+2015 HOD]",  results["wp_s"],                   "s"),
        ("ΔΣ(R)   [More+2015 HOD]",  results["esd_s"],                  "s"),
        ("wp + ΔΣ combined",         results["wp_esd_s"],               "s"),
        ("angular_cl_gX — serial",   results["angular_cl_gX_serial_s"], "s"),
        ("angular_cl_gX — parallel", results["angular_cl_gX_parallel_s"],"s"),
        ("Full joint — serial",      results["joint_serial_s"],          "s"),
        ("Full joint — parallel",    results["joint_parallel_s"],        "s"),
    ]
    for name, val, unit in rows:
        print(f"  {name:<35} {val:>8.1f} {unit}")
    print("=" * 55)

    return results


def main():
    p = argparse.ArgumentParser(description="Time joint model components")
    p.add_argument("--sample", default="S1",
                   help="Sample label (default: S1)")
    p.add_argument("--n-repeat", type=int, default=1,
                   help="Repeat wp/ESD calls n times and report mean (default: 1)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: results/timing/)")
    args = p.parse_args()

    results = run_timing(label=args.sample, n_repeat=args.n_repeat)

    out_dir = Path(args.out_dir) if args.out_dir else _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timing_joint_model.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
