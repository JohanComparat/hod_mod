#!/usr/bin/env python
"""Joint MAP + MCMC fit of the Zu & Mandelbaum (2015) iHOD model to several LS10
BGS stellar-mass-**threshold** samples measured by ``sum_stat``.

This is the threshold-sample sibling of
:mod:`hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint` (which fits stellar-mass
*bins*).  Every volume-limited threshold sample (``M* > thr``) is fit
**simultaneously** with one shared set of thirteen SHMR/HOD + satellite parameters.
Each sample contributes its projected clustering ``wp(rp)`` and a
**stellar-mass-function** constraint to a single summed log-likelihood::

    log P(theta) = log_prior(theta)
                 + Sum_samples [ -0.5 * (chi2_wp + chi2_n + [chi2_DS if --fit-esd]) ]

The "stellar-mass-function" term is the **integrated number density** above the
threshold, ``n(>thr) = integral phi(M*) dlog10M*``, compared to the model cumulative
``n_gal``.  The binned SMF ``phi(M*)`` itself is **not** fit point-by-point: its
jackknife covariance is near-singular (see
``fit_comparat2025.py`` for the same reasoning), so we constrain its integral.

Threshold vs bin
----------------
Each sample sets ``log10m_star_thresh`` (the threshold) and leaves
``log10m_star_max`` **unset**, so :meth:`ZuMandelbaum15HODModel.nc_ns` and
``n_gal`` return the cumulative occupation ``N(M* > thresh)`` rather than a bin HOD.

Lensing (ESD)
-------------
Galaxy-galaxy lensing ΔΣ is **optional**.  Pass ``--surveys HSC DES KIDS`` to load
it (it then appears in the best-fit figures); add ``--fit-esd`` to additionally
include it in the likelihood.  By default no ESD is loaded or fit — only
``wp + n(>thr)``.

Data
----
The per-sample joint HDF5 files produced by ``sum_stat``::

    ~/software/sum_stat/data/BGS_Mstar<thr>/LS10_VLIM_ANY_<thr>_Mstar_12.0_..._joint_..._comb.h5

are read through :class:`~hod_mod.data_io.sum_stat_reader.SumStatReader`:
``wp`` and ``esd_<survey>`` via :meth:`joint_bgs`, and the SMF via :meth:`smf`.

Units (identical to the bin script)
-----------------------------------
- ``wp``  : SumStatReader returns Mpc/h; predictor ``wp`` is native Mpc/h.
- ``DS``  : sum_stat stores M_sun/pc^2 (h-invariant); predictor ``delta_sigma``
            returns M_sun h/pc^2 — divided by h before comparison.
- ``n_g`` : SumStatReader returns h^3 Mpc^-3; predictor ``n_gal`` is native.
- Stellar masses are **physical** log10(M*/M_sun) throughout.

Output files (in --out-dir, default results/bgs_zm15_thresh_joint)
-----------------------------------------------------------------
map_result.json, chain.h5, flatchain.npz, and the same ZM15 figure set as the bin
script (map_bestfit / hod_occupation / shmr / stellar_mass_function /
satellite_fraction / zm15_montage pdfs).

Usage examples
--------------
    # wp + n(>thr) only (no lensing), MAP + MCMC
    python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_thresh_joint \\
        --data-dir ~/software/sum_stat/data --mode both

    # load ESD for plotting only (not fit)
    python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_thresh_joint \\
        --surveys HSC DES KIDS --mode map

    # also fit ESD
    python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_thresh_joint \\
        --surveys HSC DES KIDS --fit-esd --mode both

References
----------
Zu & Mandelbaum 2015, MNRAS 454, 1161 (arXiv:1505.02781)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse everything that is identical to the bin fit: parameter setup, the
# regularised inverse-covariance helper, the predictor builder, the joint
# likelihood class (now threshold-aware via ``b["max"] is None`` + ``fit_esd``),
# and the full plotting suite.
from hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint import (  # noqa: E402
    FREE_NAMES,
    PUBLISHED,
    JointZM15,
    _discover_smf_file,
    _regularised_icov,
    build_predictor,
    load_observed_smf,
    plot_all,
)

# LS10_VLIM_ANY_<thr>_Mstar_12.0_...   →  thr is the stellar-mass threshold.
_THRESH_RE = re.compile(r"VLIM_ANY_([0-9p.]+)_Mstar_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_threshold(fname: str) -> float | None:
    m = _THRESH_RE.search(fname)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def load_thresholds(data_dir: str, surveys: list[str], rp_min: float, rp_max: float,
                    R_min: float, R_max: float,
                    ng_frac_err_floor: float, log=print) -> tuple[list[dict], float]:
    """Load every threshold-sample HDF5 under *data_dir* into fit dictionaries.

    Globs ``<data_dir>/BGS_Mstar*/LS10_VLIM_ANY_*_joint_*comb.h5``.  Each sample
    contributes ``wp(rp)``, an integrated number density ``n(>thr)`` from its SMF,
    and (if *surveys* requested) per-survey ESD for plotting / optional fitting.
    """
    from hod_mod.data_io.sum_stat_reader import SumStatReader

    # Threshold-sample files carry the SMF in their joint vector and are named
    # ``..._joint_smf-...``; the mass-BIN campaign files are ``..._joint_nbar-...``
    # (and have no smf slice), so the ``smf`` glob excludes them cleanly even when
    # both live under the same data root (e.g. the BGS_Mstar10_massbins dir).
    paths = sorted(glob.glob(os.path.join(
        data_dir, "BGS_Mstar*", "LS10_VLIM_ANY_*_joint_smf*comb.h5")))
    if not paths:
        raise FileNotFoundError(
            f"No 'BGS_Mstar*/LS10_VLIM_ANY_*_joint_smf*comb.h5' files found under "
            f"{data_dir}. Point --data-dir at the parent of the BGS_Mstar* dirs.")

    samples: list[dict] = []
    h_file = None
    seen: set[float] = set()
    for path in paths:
        thr = _parse_threshold(os.path.basename(path))
        if thr is None:
            log(f"  [skip] cannot parse threshold: {os.path.basename(path)}")
            continue
        if thr in seen:
            log(f"  [skip] duplicate threshold {thr:g}: {os.path.basename(path)}")
            continue
        seen.add(thr)

        reader = SumStatReader.from_hdf5(path)
        h_file = reader.h() if h_file is None else h_file

        # wp(rp) from the joint covariance block
        jb_wp = reader.joint_bgs(probes=("wp",))
        rp    = np.asarray(jb_wp["rp_wp"])
        wp    = np.asarray(jb_wp["data_vector"])
        covwp = np.asarray(jb_wp["cov"])
        m_wp  = (rp >= rp_min) & (rp <= rp_max)
        rp, wp = rp[m_wp], wp[m_wp]
        icov_wp = _regularised_icov(covwp[np.ix_(m_wp, m_wp)], np.sqrt(np.diag(covwp))[m_wp])

        # ESD per survey (loaded for plotting; fit only when --fit-esd)
        slices = reader._cache.get("joint_bgs", {}).get("slices", {})
        surv_data: dict[str, tuple] = {}
        for s in surveys:
            probe = f"esd_{s.lower()}"
            if probe not in slices:
                continue
            jb = reader.joint_bgs(probes=(probe,))
            R   = np.asarray(jb["rp_esd"])
            ds  = np.asarray(jb["data_vector"])
            cov = np.asarray(jb["cov"])
            m_R = (R >= R_min) & (R <= R_max) & np.isfinite(ds)
            if m_R.sum() < 2 or not np.all(np.isfinite(ds[m_R])):
                log(f"  [>{thr:g}] {s}: insufficient finite ΔΣ points — dropped")
                continue
            R, ds = R[m_R], ds[m_R]
            icov_ds = _regularised_icov(cov[np.ix_(m_R, m_R)], np.sqrt(np.diag(cov))[m_R])
            surv_data[s] = (R, ds, icov_ds)

        # Stellar mass function → integrated number density n(>thr).
        smf  = reader.smf()
        mst  = np.asarray(smf["log10mstar"], dtype=float)
        phi  = np.asarray(smf["phi"], dtype=float)
        phie = np.asarray(smf.get("phi_err"), dtype=float) if smf.get("phi_err") is not None else None
        good = np.isfinite(phi) & (phi > 0)
        m_n  = good & (mst >= thr)        # only complete bins above the threshold
        if m_n.sum() < 2:
            log(f"  [>{thr:g}] SMF: <2 usable bins above threshold — dropped")
            continue
        n_obs = float(np.trapezoid(phi[m_n], mst[m_n]))
        if not (np.isfinite(n_obs) and n_obs > 0):
            log(f"  [>{thr:g}] SMF integral non-finite/zero — dropped")
            continue
        # Jackknife n_gal error is ~0 (cumulative); use the fractional floor.
        n_frac = float(ng_frac_err_floor)

        # Per-sample effective redshift from the joint-covariance z_min/z_max.
        attrs = jb_wp.get("attrs", {})
        z_sample = None
        try:
            zlo = float(attrs.get("z_min"))
            zhi = float(attrs.get("z_max"))
            z_sample = 0.5 * (zlo + zhi)
            if not np.isfinite(z_sample):
                z_sample = None
        except (TypeError, ValueError):
            z_sample = None

        samples.append({
            "label":    f">{thr:g}",
            "thresh":   thr,            # physical log10(M*/M_sun) threshold
            "max":      None,           # threshold sample → cumulative HOD
            "rp":       rp,
            "wp_obs":   wp,
            "icov_wp":  icov_wp,
            "surveys":  surv_data,
            "n_obs":    n_obs,
            "n_frac":   n_frac,
            "smf_obs":  {"log10mstar": mst[good], "phi": phi[good],
                         "phi_err": (phie[good] if phie is not None else None)},
            "z":        z_sample,
        })
        esd_str = (", ".join(f"{s}:{len(v[0])}" for s, v in surv_data.items())
                   if surv_data else "none")
        log(f"  [>{thr:g}]  z={'%.3f'%z_sample if z_sample is not None else 'n/a'}  "
            f"wp={len(rp):2d}pts  ESD={{{esd_str}}}  "
            f"n(>{thr:g})={n_obs:.3e} h^3Mpc^-3 (±{100*n_frac:.0f}%)")

    if not samples:
        raise RuntimeError("No usable threshold samples were loaded.")
    return samples, float(h_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.expanduser("~/software/sum_stat/data"),
                   help="Parent directory of the BGS_Mstar*/ threshold-sample dirs")
    p.add_argument("--surveys", nargs="*", default=[],
                   help="Lensing surveys to LOAD (HSC DES KIDS). Loaded for plotting; "
                        "use --fit-esd to also include them in the likelihood. "
                        "Default: none.")
    p.add_argument("--fit-esd", action="store_true",
                   help="Include the loaded ESD surveys in the likelihood "
                        "(default: ESD is plot-only)")
    p.add_argument("--mode", choices=["map", "mcmc", "both"], default="both")
    p.add_argument("--rp-min", type=float, default=0.1, help="wp r_p min [Mpc/h]")
    p.add_argument("--rp-max", type=float, default=30.0, help="wp r_p max [Mpc/h]")
    p.add_argument("--R-min",  type=float, default=0.1, help="ESD R min [Mpc/h]")
    p.add_argument("--R-max",  type=float, default=30.0, help="ESD R max [Mpc/h]")
    p.add_argument("--z-eff",  type=float, default=0.13,
                   help="Fallback effective redshift for samples lacking z_min/z_max")
    p.add_argument("--pi-max-mpc", type=float, default=100.0,
                   help="wp pi_max in physical Mpc (sum_stat value; converted to Mpc/h)")
    p.add_argument("--hmf-backend", default="tinker08")
    p.add_argument("--smf-file", default=None,
                   help="Observed SMF file for the comparison curve (NOT fitted). "
                        "Default: auto-discover the widest-coverage BGS_Mstar* file.")
    p.add_argument("--smf-data-dir", default=None,
                   help="Root searched for the comparison SMF file (default: --data-dir)")
    p.add_argument("--ng-frac-err-floor", type=float, default=0.05,
                   help="Fractional error on the integrated n(>thr) (default 0.05)")
    p.add_argument("--gaussian-prior", action="store_true",
                   help="Add a Gaussian prior from the published ZM15 values")
    p.add_argument("--n-walkers", type=int, default=32)
    p.add_argument("--n-burnin",  type=int, default=500)
    p.add_argument("--n-steps",   type=int, default=2000)
    from hod_mod.paths import results_root
    p.add_argument("--out-dir", default=os.path.join(
        results_root(), "bgs_zm15_thresh_joint"))
    p.add_argument("--force-mcmc", action="store_true",
                   help="Rerun MCMC even if a chain already exists")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip fitting; load existing MAP result and generate plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print("Loading threshold-sample measurements ...")
    samples, h = load_thresholds(
        args.data_dir, args.surveys, args.rp_min, args.rp_max,
        args.R_min, args.R_max, args.ng_frac_err_floor)
    n_missing = 0
    for b in samples:
        if b.get("z") is None:
            b["z"] = args.z_eff
            n_missing += 1
    if n_missing:
        print(f"  {n_missing}/{len(samples)} samples had no z_min/z_max — "
              f"using --z-eff={args.z_eff} for those")
    print(f"Loaded {len(samples)} threshold samples.  h={h:.4f}  "
          f"z per sample: {[round(b['z'], 3) for b in samples]}")
    if args.surveys and not args.fit_esd:
        print(f"  ESD surveys {args.surveys} loaded for PLOTTING only "
              f"(pass --fit-esd to include them in the fit)")

    print("Building Zu & Mandelbaum 2015 predictor ...")
    predictor, theta_cosmo = build_predictor(args.hmf_backend)

    fitter = JointZM15(
        samples, predictor, theta_cosmo, h=h, z=args.z_eff,
        pi_max_h=args.pi_max_mpc * h, gaussian_prior=args.gaussian_prior,
        fit_esd=args.fit_esd)

    # Observed SMF (comparison curve only — NOT part of the fit).
    smf_root = args.smf_data_dir or args.data_dir
    smf_file = args.smf_file or _discover_smf_file(smf_root)
    obs_smf = None
    if smf_file and os.path.exists(smf_file):
        try:
            obs_smf = load_observed_smf(smf_file, z_fallback=args.z_eff)
            print(f"Comparison SMF (not fitted): {os.path.basename(smf_file)}  "
                  f"({len(obs_smf['log10mstar'])} pts, z={obs_smf['z']:.3f})")
        except Exception as exc:
            print(f"  [warn] could not load comparison SMF from {smf_file}: {exc}")
    else:
        print(f"  [warn] no comparison SMF file found under {smf_root}"
              f"/BGS_Mstar*/*joint_smf*.h5 — SMF curve will be model-only")

    map_json = os.path.join(args.out_dir, "map_result.json")
    map_result = None

    if args.plot_only:
        if not os.path.exists(map_json):
            raise FileNotFoundError(
                f"No MAP result found at {map_json}. Run --mode map first.")
        with open(map_json) as fh:
            map_result = json.load(fh)
        print(f"Loaded MAP result: chi2/dof={map_result['chi2_per_dof']:.3f}")
        plot_all(samples, predictor, theta_cosmo, h,
                 pi_max_h=args.pi_max_mpc * h, map_result=map_result,
                 obs_smf=obs_smf, out_dir=args.out_dir,
                 surveys=args.surveys, z_eff=args.z_eff)
        print(f"\nAll done in {(time.time() - t0) / 60:.1f} min.")
        return

    if args.mode in ("map", "both"):
        print("\n=== MAP optimisation (Powell) ===")
        map_result = fitter.map_fit()
        with open(map_json, "w") as fh:
            json.dump(map_result, fh, indent=2)
        print(f"\nchi2/ndof = {map_result['chi2']:.1f} / {map_result['ndof']} "
              f"= {map_result['chi2_per_dof']:.3f}")
        print(f"{'param':14s} {'MAP':>10s} {'published':>12s}")
        for name in FREE_NAMES:
            pub = PUBLISHED[name][0]
            print(f"{name:14s} {map_result['params'][name]:10.4f} {pub:12.4f}")
        print(f"MAP result -> {map_json}")
        plot_all(samples, predictor, theta_cosmo, h,
                 pi_max_h=args.pi_max_mpc * h, map_result=map_result,
                 obs_smf=obs_smf, out_dir=args.out_dir,
                 surveys=args.surveys, z_eff=args.z_eff)

    if args.mode in ("mcmc", "both"):
        chain = os.path.join(args.out_dir, "flatchain.npz")
        if os.path.exists(chain) and not args.force_mcmc:
            print(f"\n[skip] chain exists: {chain} (use --force-mcmc to rerun)")
        else:
            print("\n=== MCMC sampling (emcee) ===")
            if map_result is not None:
                x_start = np.array(map_result["theta"])
            elif os.path.exists(map_json):
                with open(map_json) as fh:
                    x_start = np.array(json.load(fh)["theta"])
                print(f"  loaded MAP starting point from {map_json}")
            else:
                x_start = None
            fitter.sample(args.out_dir, args.n_walkers, args.n_burnin,
                          args.n_steps, x_start=x_start)

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
