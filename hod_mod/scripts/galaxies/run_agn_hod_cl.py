"""Demo driver for the HOD-based AGN model (HODAgnModel).

Builds the full halo-model chain for an LS10-BGS sample, runs the modified
abundance matching, and reports the AGN occupation diagnostics plus the X-ray
angular auto/cross power spectra.

Usage
-----
    python -m hod_mod.scripts.galaxies.run_agn_hod_cl --sample S1
    python -m hod_mod.scripts.galaxies.run_agn_hod_cl --sample S5
"""

from __future__ import annotations

import argparse

import numpy as np

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import GasDensityDPM
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.hod import HODAgnModel, BGS_SAMPLES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1", choices=list(BGS_SAMPLES),
                    help="LS10-BGS sample (default S1)")
    ap.add_argument("--area-deg2", type=float, default=None,
                    help="Override survey solid angle [deg^2]")
    ap.add_argument("--n-z", type=int, default=7,
                    help="n(z) sampling points for the Limber projection")
    ap.add_argument("--n-ell", type=int, default=10,
                    help="Number of multipoles")
    ap.add_argument("--skip-cl", action="store_true",
                    help="Only report the abundance-match diagnostics (fast)")
    args = ap.parse_args()

    cfg = BGS_SAMPLES[args.sample]
    z = cfg["z_mean"]
    theta = LinearPowerSpectrum.default_cosmology()
    colossus = dict(flat=True, H0=theta["h"] * 100.0, Om0=theta["Omega_m"],
                    Ob0=theta["Omega_b"], sigma8=0.811, ns=theta["n_s"])

    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    hp = HaloProfile(colossus, cm_relation="diemer19")
    hod = MoreHODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    dp = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80)

    agn_kw = dict(pk_lin=pk, theta_cosmo=theta, z_mean=z,
                  z_max=cfg["z_max"])
    if args.area_deg2 is not None:
        agn_kw["sky_area_deg2"] = args.area_deg2
    agn = HODAgnModel(**agn_kw)

    cx = HaloModelCrossSpectra(fhmp, density_profile=dp, agn_model=agn)

    print(f"\n=== HODAgnModel — LS10-BGS {args.sample} (z_mean={z:.3f}) ===")
    print(f"  AGN HOD params           : {agn._hod_params}")
    print(f"  mean observed log10 LX   : {agn._mean_log10lx:.3f}  (0.5-2 keV, erg/s)")
    print(f"  mean observed FX         : {agn.mean_observed_fx():.3e} erg/s/cm^2")
    print(f"  n_AGN host (f_inc applied): {agn._n_agn_host:.3e} (Mpc/h)^-3")
    print(f"  n_LX selected             : {agn._n_lx_selected:.3e} (Mpc/h)^-3")
    print(f"  XLF bins kept by r-cut    : {agn._frac_selected:.3f}")
    print(f"  host fraction clamped     : {agn._frac_clamped:.3f}")
    print(f"  selected soft LX range    : [{agn._lx_soft_floor:.2f}, {agn._lx_soft_ceil:.2f}]")
    print(f"  sample volume             : {agn._volume_h3:.3e} (Mpc/h)^3")
    print(f"  N_AGN (count)             : {agn._n_agn_count:.3e}")

    if args.skip_cl:
        return

    # Angular power spectra.  The Limber projection integrates over redshift,
    # so n(z) needs several points spanning the sample's z range — a single
    # delta-function point integrates to zero.  Use a Gaussian around z_mean.
    # The per-redshift gas-profile Fourier transform dominates the cost, so
    # keep n_z / n_ell modest for the demo (increase for production).
    ell = np.logspace(1.5, 3.0, args.n_ell)
    z_arr = np.linspace(max(1e-3, z - 0.05), z + 0.05, args.n_z)
    nz_g = np.exp(-0.5 * ((z_arr - z) / 0.025) ** 2)
    nz_g /= np.trapezoid(nz_g, z_arr)
    hod_params = MoreHODModel.default_params()

    cl_gX = cx.angular_cl_gX(ell, z_arr, nz_g, theta, hod_params,
                             return_components=True, n_workers=1)
    cl_XX = cx.angular_cl_XX(ell, z_arr, nz_g, theta, return_components=True,
                             n_workers=1)

    def _show(name, res):
        d = res if isinstance(res, dict) else {"total": res}
        print(f"\n  {name} components (ell={ell[0]:.0f} … {ell[-1]:.0f}):")
        for key, val in d.items():
            val = np.asarray(val)
            print(f"    {key:16s}: min={np.nanmin(val):.3e} max={np.nanmax(val):.3e}")

    _show("C_ell^gX", cl_gX)
    _show("C_ell^XX", cl_XX)
    print()


if __name__ == "__main__":
    main()
