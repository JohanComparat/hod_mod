"""Calibrate K_abs: the fixed gas-amplitude zero-point that replaces the free
log10_A_gas once the true per-halo eROSITA ECF is folded in.

The S1 duty-cycle baseline (density-only gas) fit gave log10_A_gas = -7.26 for the
NO-ECF gas shape.  With the ECF weight ECF_gas(kT(M)) folded into the gas
emissivity, the gas shape changes; K_abs makes the ECF-folded gas at fiducial
density reproduce the same physical w(theta):

    K_abs * gas_shape_ECF(theta) = 10^(-7.26) * gas_shape_noECF(theta)
    => K_abs = 10^(-7.26) * < gas_shape_noECF / gas_shape_ECF >   (theta in [8,300]")

Also reports the implied physical chain K_abs = Lambda_ref * U_geom / (S^R_X * ECF_fixed)
-> U_geom, and the per-theta spread (the small ECF shape change = the physical
mass-weighting correction).
"""
from __future__ import annotations

import numpy as np

from hod_mod.gas import load_ecf_tables
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts.fitting import fit_agn_duty_cycle_baseline as B
from hod_mod.scripts import validate_gas_profiles as vgp

_LOG10_A_GAS_BASELINE = -7.257       # S1 duty-cycle baseline MAP (density-only gas)
_NE03_FID = 4.87e-5
_LAMBDA_REF = None                   # filled below
# baseline gas profile params
_P2, _RMAX, _BETA = 1.214, 2.297, 1.298


def _ecf_gas_of_mass(sample, z):
    gas_of_T, ecf_agn, ecf_fixed = load_ecf_tables(sample)

    def f(m200_h):
        m200_h = np.asarray(m200_h, float)
        r200 = vgp._r200(m200_h, z); c200 = vgp._c200_approx(m200_h)
        m500_h, _ = vgp.m200_to_m500c(m200_h, c200, r200, vgp._rho_crit_z(z))
        kT = vgp._lovisari20_kt(m500_h / vgp._H, z=z)
        return np.asarray(gas_of_T(kT), float)
    return f, ecf_agn, ecf_fixed


def main():
    sample = "S1"
    z = float(F.SAMPLES[sample]["zmean"])
    th = F._THETA_COSMO
    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"],
                Ob0=th["Omega_b"], sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    hod_params = B._build_hod_params(sample)
    z_arr, nz = F._build_nz_fast(sample)
    data = F.load_data(sample)
    th_as = data["theta_arcsec"]; th_rad = data["theta_rad"]
    mask = (th_as >= 8.0) & (th_as <= 300.0)

    ecf_gas, ecf_agn, ecf_fixed = _ecf_gas_of_mass(sample, z)

    def gas_shape(with_ecf):
        cross = HaloModelCrossSpectra(fhmp, density_profile=B._make_gas(_P2, _RMAX),
                                      ecf_gas_table=(ecf_gas if with_ecf else None))
        c = cross.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                beta_gas=_BETA, return_components=True, n_workers=1)
        return F._hankel(np.asarray(c["gas"], dtype=float), th_rad)

    print("computing gas shape (no ECF) ...", flush=True)
    g0 = gas_shape(False)
    print("computing gas shape (ECF) ...", flush=True)
    g1 = gas_shape(True)

    ratio = g0[mask] / g1[mask]
    K_abs = 10.0 ** _LOG10_A_GAS_BASELINE * np.mean(ratio)
    print(f"\nratio gas_noECF/gas_ECF over [8,300]\": "
          f"mean={np.mean(ratio):.4e} std/mean={np.std(ratio)/np.mean(ratio):.3f} "
          f"(spread = physical ECF mass-weighting shape change)")
    print(f"K_abs = 10^({_LOG10_A_GAS_BASELINE}) * mean(ratio) = {K_abs:.4e}")
    print(f"  -> log10(K_abs) = {np.log10(K_abs):.3f}")

    # physical chain: K_abs = Lambda_ref * U_geom / (S^R_X * ECF_fixed)
    srx = float(data.get("beckground", [np.nan])[0]) if "beckground" in data else np.nan
    from hod_mod.gas import ApecCoolingTable
    cool = ApecCoolingTable(emin=0.5, emax=2.0)
    lam_ref = float(cool(np.array([1.0]), np.array([0.3]))[0])
    if np.isfinite(srx):
        U_geom = K_abs * srx * ecf_fixed / lam_ref
        print(f"\nphysical chain: Lambda_ref={lam_ref:.3e}, S^R_X={srx:.3e}, "
              f"ECF_fixed={ecf_fixed:.3e}")
        print(f"  -> U_geom = K_abs*S^R_X*ECF_fixed/Lambda_ref = {U_geom:.4e}")
    np.savez(B._OUT_DIR + "/S1_kabs.npz", K_abs=K_abs, log10_A_gas_base=_LOG10_A_GAS_BASELINE,
             ratio_mean=float(np.mean(ratio)), lam_ref=lam_ref, ecf_fixed=ecf_fixed,
             ecf_agn=ecf_agn)
    print(f"\nsaved K_abs -> {B._OUT_DIR}/S1_kabs.npz")


if __name__ == "__main__":
    main()
