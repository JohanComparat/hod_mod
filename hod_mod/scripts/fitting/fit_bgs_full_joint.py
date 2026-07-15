"""Driver for the BGS S1 full-model joint fit (galaxies + hot gas + AGN).

Step 2 (fixed ZM15 at the mass-bin posterior median):

    HOD_MOD_DATA_DIR=/home/comparat/data/hod_mod_data JAX_PLATFORMS=cpu \
      python -m hod_mod.scripts.fitting.fit_bgs_full_joint --mode map

Step 3 (all parameters free, ZM15 anchored by Gaussian priors from the mass-bin posterior):

    ... fit_bgs_full_joint --free-zm15 --mode both

MAP runs locally; MCMC (``--mode mcmc``) uses a resumable emcee HDF backend suitable for
dahu (see oarsub/fit_bgs_full_joint_*.sh).  Default observable set is the Step-1 decision:
n_gal + wp + ESD(>2 Mpc/h) + X-ray broad + 15 bands + XLF(z=0.1,0.4) + AGN bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hod_mod.paths import results_root
from hod_mod.fitting.full_joint import JointFull

_DEFAULT_OBS = ["ngal", "wp", "esd", "xray_broad", "xray_bands", "xlf", "agn_bias"]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default="S1")
    p.add_argument("--free-zm15", action="store_true",
                   help="Step 3: free the 13 ZM15 params (Gaussian priors from the mass-bin posterior).")
    p.add_argument("--observables", nargs="+", default=_DEFAULT_OBS,
                   help=f"subset of {_DEFAULT_OBS}")
    p.add_argument("--mode", choices=["map", "mcmc", "both"], default="map")
    p.add_argument("--rp-min-wp", type=float, default=0.5)
    p.add_argument("--rp-min-esd", type=float, default=2.0)
    p.add_argument("--esd-surveys", nargs="+", default=["DES", "HSC", "KIDS"],
                   help="which ESD lensing surveys to fit (default all three)")
    p.add_argument("--esd-rp-max", type=float, default=float("inf"),
                   help="upper rp cut [Mpc/h] on the ESD data (default no cut)")
    p.add_argument("--xlf-z", nargs="+", type=float, default=[0.1, 0.4],
                   help="XLF redshift slices to fit (default 0.1 0.4)")
    p.add_argument("--xlf-lx-min", type=float, default=float("-inf"),
                   help="lower log10 L_X cut on the XLF data (default no cut)")
    p.add_argument("--agn-bias-refs", nargs="+", default=None,
                   help="restrict the AGN halo-bias compilation to these references "
                        "(e.g. Comparat23 Krumpe15); default keeps all")
    p.add_argument("--kt-prior-sig", nargs=2, type=float, default=None,
                   metavar=("SIG_NORM", "SIG_SLOPE"),
                   help="tighten the kt_norm / kt_slope Gaussian prior widths (default "
                        "0.2 0.15) to keep kT-M near the cluster scaling, e.g. 0.10 0.06")
    p.add_argument("--f-sys", type=float, default=0.05)
    p.add_argument("--hmf", default="tinker08")
    p.add_argument("--n-walkers", type=int, default=48)
    p.add_argument("--n-burnin", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else (
        results_root() / ("bgs_full_joint_allparams" if a.free_zm15
                          else "bgs_full_joint_fixedzm15"))
    out.mkdir(parents=True, exist_ok=True)

    # Persist the exact data selection so the plotter can rebuild an identical model.
    json.dump(dict(sample=a.sample, free_zm15=a.free_zm15, observables=list(a.observables),
                   rp_min_wp=a.rp_min_wp, rp_min_esd=a.rp_min_esd,
                   esd_surveys=list(a.esd_surveys), esd_rp_max=a.esd_rp_max,
                   xlf_z=list(a.xlf_z), xlf_lx_min=a.xlf_lx_min,
                   agn_bias_refs=(list(a.agn_bias_refs) if a.agn_bias_refs else None),
                   kt_prior_sig=(list(a.kt_prior_sig) if a.kt_prior_sig else None),
                   f_sys=a.f_sys, hmf=a.hmf),
              open(out / "fit_config.json", "w"), indent=2)

    J = JointFull(sample=a.sample, free_zm15=a.free_zm15, observables=a.observables,
                  rp_min_wp=a.rp_min_wp, rp_min_esd=a.rp_min_esd, f_sys=a.f_sys,
                  hmf_backend=a.hmf, esd_surveys=a.esd_surveys, esd_rp_max=a.esd_rp_max,
                  xlf_z=a.xlf_z, xlf_lx_min=a.xlf_lx_min, agn_bias_refs=a.agn_bias_refs,
                  kt_prior_sig=a.kt_prior_sig)
    print(f"[fit] ndim={J.ndim}  n_data={J.n_data()}  free_zm15={a.free_zm15}", flush=True)
    print(f"[fit] free params: {J.names}", flush=True)

    x_start = None
    map_json = out / "map_result.json"

    if a.mode in ("map", "both"):
        print("[fit] running MAP (local) ...", flush=True)
        res = J.map_fit()
        json.dump(res, open(map_json, "w"), indent=2)
        print(f"[fit] MAP chi2/dof = {res['chi2']:.1f}/{res['ndof']} = {res['chi2_per_dof']:.3f}",
              flush=True)
        print(f"[fit]   breakdown: {({k: round(x, 1) for k, x in res['breakdown'].items()})}",
              flush=True)
        for n, x in res["params"].items():
            print(f"[fit]     {n:20s} = {x:+.4f}", flush=True)
        x_start = np.array(res["theta"])

    if a.mode in ("mcmc", "both"):
        if x_start is None and map_json.exists():
            x_start = np.array(json.load(open(map_json))["theta"])
        print(f"[fit] running MCMC: {a.n_walkers} walkers, {a.n_burnin}+{a.n_steps} steps "
              f"-> {out} ...", flush=True)
        r = J.sample(out, n_walkers=a.n_walkers, n_burnin=a.n_burnin, n_steps=a.n_steps,
                     x_start=x_start)
        print(f"[fit] MCMC done; acceptance={r['acceptance']:.2f}", flush=True)

    print(f"[fit] outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
