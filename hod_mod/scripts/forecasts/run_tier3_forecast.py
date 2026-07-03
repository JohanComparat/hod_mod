r"""Tier-3 Fisher forecast: 102 parameters, multi-wavelength maps, z < 2, M* > 10⁹.

Extends the tier-2 production run (``run_tier2_forecast``) with the tier-3
observable families of :class:`hod_mod.forecast.tier3.Tier3Forecast`:

* the coarse exploratory grid — Δz = 0.2 shells over 0 < z < 2 × 0.2-dex
  volume-limited M* bins over 9.0 ≤ log10 M* ≤ 11.6, SF/Q split ON, with the
  two-tier (wide + deep-field) spectroscopic completeness model;
* SKA-like radio and WISE/SPHEREx-like IR intensity maps: galaxy crosses per
  cell, autos + AGN crosses per shell;
* galaxy band LFs (UV/opt/NIR/Hα) and AGN UV/optical LFs per shell;
* the wide-M* SFRD(z) blocks and the four extras — tSZ auto, 21 cm auto,
  X-ray cluster counts, AGN galaxy–galaxy lensing.

Usage::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier3_forecast \
        --jobs 8
    # quick end-to-end check (2×2 cells, tiny grids, ~4 min):
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier3_forecast --smoke
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from hod_mod.forecast.forward_jax import PARAM_NAMES, _IDX, TIER3_EXTENSION  # noqa: E402
from hod_mod.forecast import params, fisher  # noqa: E402
from hod_mod.forecast.tier3 import Tier3Forecast  # noqa: E402
from hod_mod.scripts.forecasts.run_tier2_forecast import (  # noqa: E402
    _s8_sigma, _sig, _bits_per_sector, EVOL_PARAMS)

# cumulative probe attribution, tier-2 order + the tier-3 families
PROBE_GROUPS = [
    ("galaxy grid", ("wp", "ds", "n_gal", "ssfr", "sfrd", "oiilf")),
    ("+lensing", ("cl_kk", "cl_kCMB", "cl_shear_kCMB", "cl_gkCMB")),
    ("+X-ray/tSZ", ("cl_gX", "cl_XX", "cl_gy")),
    ("+XLF(z)", ("xlf",)),
    ("+wp_agn", ("wp_agn",)),
    ("+radio LF", ("rlf",)),
    ("+IR AGN", ("ilf",)),
    ("+HI", ("himf", "cl_gHI")),
    ("+radio/IR maps", ("cl_gR", "cl_RR", "cl_aR", "cl_gI", "cl_II",
                        "cl_aI", "cl_ag")),
    ("+band LFs", ("uvlf", "optlf", "nirlf", "half", "qlf_uv", "qlf_opt")),
    ("+tSZ/HI autos", ("cl_yy", "cl_HIHI")),
    ("+clusters", ("ncl",)),
    ("+AGN lensing", ("ds_agn",)),
]


def _out_dir():
    try:
        from hod_mod import paths
        d = os.fspath(paths.results_root() / "tier3_forecast")
    except Exception:
        d = os.path.join(os.environ.get("HOD_MOD_RESULTS", "."), "tier3_forecast")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rmin", type=float, nargs="+", default=[0.1, 0.5, 2.5])
    ap.add_argument("--n-bands", type=int, default=6, choices=[1, 6, 15],
                    help="X-ray energy bands over 0.5-2 keV")
    ap.add_argument("--n-k", type=int, default=96)
    ap.add_argument("--n-gl", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for the block precompute")
    ap.add_argument("--cache-dir", default=None,
                    help="per-block Jacobian cache (default <out>/cache)")
    ap.add_argument("--smoke", action="store_true",
                    help="2x2 cells, tiny grids: fast end-to-end check")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--no-bandlfs", action="store_true")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = _out_dir()
    cache = args.cache_dir or os.path.join(out, "cache")
    tag = f"nb{args.n_bands}" + ("_smoke" if args.smoke else "")

    common = dict(include_maps=not args.no_maps,
                  include_bandlfs=not args.no_bandlfs,
                  include_extras=not args.no_extras)
    if args.smoke:
        t3 = Tier3Forecast(
            z_edges=[0.2, 0.4, 0.6], mstar_edges=[9.4, 9.6, 10.6],
            n_bands=[(0.5, 1.0), (1.0, 2.0)], n_shear_bins=2,
            agn_lx_bins=[(42.0, 42.5), (42.5, 43.0)], agn_z_centers=(0.3,),
            n_k=48, n_m=48, n_gl=16, n_z=3, cell_n_m=96,
            rp_wp=np.logspace(-1, 1.4, 6), rp_ds=np.logspace(-1, 1.2, 5),
            ell=np.logspace(1.0, 3.3, 6), rp_wp_agn=np.logspace(0.1, 1.4, 4),
            **common)
    else:
        t3 = Tier3Forecast(n_bands=args.n_bands, n_k=args.n_k,
                           n_gl=args.n_gl, **common)

    fid = t3.fiducial()
    names = PARAM_NAMES
    n_cells = sum(1 for b in t3.blocks if b.kind == "cell")
    print(f"[setup] tier-3: {len(names)} parameters "
          f"({len(TIER3_EXTENSION)} new), {n_cells} (z,M*) cell blocks, "
          f"{len(t3.bands)} X-ray bands, {len(t3.ska.bands)} radio + "
          f"{len(t3.irmap.bands)} IR map bands, {len(t3.blocks)} blocks")
    if t3.skipped_cells:
        print(f"[setup] {len(t3.skipped_cells)} cells skipped "
              f"(incomplete in both spectro tiers):")
        for z1, z2, m1, m2 in t3.skipped_cells:
            print(f"        z=[{z1:.1f},{z2:.1f}) logM*=[{m1:.1f},{m2:.1f})")

    if args.jobs > 1:
        missing = t3.precompute_blocks(fid, cache, jobs=args.jobs)
        print(f"[precompute] {len(missing)} blocks computed with "
              f"{args.jobs} workers")
    d0, J, meta = t3.data_and_jacobian(fid, cache_dir=cache)
    print(f"[jacobian] {d0.size} rows x {J.shape[1]} params; "
          f"finite={np.all(np.isfinite(J))}")

    sig_noise = t3.noise_sigma(fid, d0, meta)
    rel = np.where(np.abs(d0) > 0, sig_noise / np.abs(d0), np.inf)
    finite = np.isfinite(rel)
    print(f"[noise] median rel err = {np.median(rel[finite]):.3g}; "
          f"{np.sum(~finite)} rows at sigma=inf (completeness)")

    prior = params.regularizing_prior(fix=("log10DC",))
    scale = params.regularizing_prior()          # conditioning scale (finite)

    results = {}
    for rmin in args.rmin:
        keep = t3.scale_cut_mask(meta, rmin) & finite
        cov, sig, corr = _sig(d0, J, keep, rel, prior, scale)

        astro = [n for n in names if n not in params.SECTORS["cosmology"]]
        prior_ap = params.regularizing_prior(fix=tuple(astro))
        cov_ap, sig_ap, _ = _sig(d0, J, keep, rel, prior_ap, scale)

        prior_cp = params.regularizing_prior(
            fix=tuple(params.SECTORS["cosmology"]) + ("log10DC",))
        cov_cp, sig_cp, _ = _sig(d0, J, keep, rel, prior_cp, scale)

        cum, got = {}, []
        for label, obs_group in PROBE_GROUPS:
            got += list(obs_group)
            m = keep & np.isin(meta["obs"], got)
            _, s_c, _ = _sig(d0, J, m, rel, prior, scale)
            cum[label] = s_c

        r = dict(rmin=rmin, sigma=sig, cov=cov, corr=corr, keep=keep,
                 sigma_astro_pinned=sig_ap, cov_astro_pinned=cov_ap,
                 sigma_cosmo_pinned=sig_cp,
                 s8=_s8_sigma(cov, fid), s8_astro_pinned=_s8_sigma(cov_ap, fid),
                 bits=_bits_per_sector(cov, prior),
                 cumulative=cum, n_rows=int(keep.sum()))
        results[rmin] = r
        print(f"\n[rmin={rmin}] rows={keep.sum()}")
        print("  cosmology (marginalized over astro vs astro-pinned):")
        for n in params.SECTORS["cosmology"]:
            i = _IDX[n]
            print(f"    sigma({n:8s}) = {sig[i]:9.2e}  vs pinned {sig_ap[i]:9.2e}"
                  f"   (degradation x{sig[i] / sig_ap[i]:6.1f})")
        print(f"    sigma(S8      ) = {r['s8']:9.2e}  vs pinned "
              f"{r['s8_astro_pinned']:9.2e}")
        print("  information gained [bits]: " +
              ", ".join(f"{k}={v:.1f}" for k, v in r["bits"].items()))

    # ------- summary + npz ------------------------------------------------
    rm0 = args.rmin[0]
    r0 = results[rm0]
    lines = [f"TIER-3 FORECAST SUMMARY ({tag})",
             f"params={len(names)}  rows={d0.size}  cells={n_cells}  "
             f"bands={len(t3.bands)}  shear_bins={t3.n_shear_bins}  "
             f"skipped_cells={len(t3.skipped_cells)}", ""]
    lines.append(f"--- marginalized 1-sigma at rmin={rm0} Mpc/h "
                 f"(vs cosmo-pinned for astro; vs astro-pinned for cosmo) ---")
    for n in names:
        i = _IDX[n]
        ref = (r0["sigma_astro_pinned"][i]
               if n in params.SECTORS["cosmology"] else r0["sigma_cosmo_pinned"][i])
        sec = next(s for s, nn in params.SECTORS.items() if n in nn)
        lines.append(f"{n:20s} [{sec:9s}] sigma={r0['sigma'][i]:9.3e}  "
                     f"pinned-ref={ref:9.3e}  prior={prior[i]:8.2g}")
    lines.append("")
    lines.append("--- tier-3 SED calibrations ---")
    for n in TIER3_EXTENSION:
        i = _IDX[n]
        lines.append(f"{n:20s} sigma={r0['sigma'][i]:9.3e}  (prior {prior[i]:.2g})")
    lines.append("")
    lines.append("--- evolution parameters ---")
    for n in EVOL_PARAMS:
        i = _IDX[n]
        lines.append(f"{n:20s} sigma={r0['sigma'][i]:9.3e}  (prior {prior[i]:.2g})")
    lines.append("")
    lines.append("--- cumulative probe attribution: sigma(Omega_m), sigma(sigma8) ---")
    for label, s_c in r0["cumulative"].items():
        lines.append(f"{label:16s} sigma(Om)={s_c[_IDX['Omega_m']]:9.3e}  "
                     f"sigma(s8)={s_c[_IDX['sigma8']]:9.3e}")
    lines.append("")
    lines.append("--- top degeneracies ---")
    for a, n1, n2, c in fisher.top_degeneracies(r0["corr"], names, k=10):
        lines.append(f"corr({n1}, {n2}) = {c:+.3f}")
    lines.append("")
    for d in fisher.principal_directions(r0["cov"], names, sigma_fid=fid, k=3):
        comp = " ".join(f"{v:+.2f}{n}" for n, v in d["components"])
        lines.append(f"worst direction (var {d['variance']:.2e}): {comp}")
    if t3.skipped_cells:
        lines.append("")
        lines.append(f"--- {len(t3.skipped_cells)} cells skipped (two-tier "
                     f"completeness) ---")
        for z1, z2, m1, m2 in t3.skipped_cells:
            lines.append(f"  z=[{z1:.1f},{z2:.1f}) logM*=[{m1:.1f},{m2:.1f})")
    if t3.completeness_flags:
        lines.append("")
        lines.append(f"--- completeness: {len(t3.completeness_flags)} rows "
                     f"dropped (below the survey limits) ---")
        for lab, o, xv in t3.completeness_flags:
            lines.append(f"  {lab} {o} x={xv:.2f}")
    summary = "\n".join(lines)
    with open(os.path.join(out, f"SUMMARY_{tag}.txt"), "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)

    np.savez_compressed(
        os.path.join(out, f"tier3_forecast_{tag}.npz"),
        param_names=np.array(names), fid=fid, d0=d0, J=J,
        sigma_noise=sig_noise, prior=prior,
        **{f"meta_{k}": v for k, v in meta.items()},
        **{f"sigma_rmin{rm}": results[rm]["sigma"] for rm in args.rmin},
        **{f"cov_rmin{rm}": results[rm]["cov"] for rm in args.rmin},
        **{f"keep_rmin{rm}": results[rm]["keep"] for rm in args.rmin},
        n_bands=len(t3.bands),
        sigma_astro_pinned=r0["sigma_astro_pinned"],
        cov_astro_pinned=r0["cov_astro_pinned"],
        sigma_cosmo_pinned=r0["sigma_cosmo_pinned"],
        cum_labels=np.array([k for k in r0["cumulative"]]),
        cum_sigma=np.stack([v for v in r0["cumulative"].values()]),
        rmin=np.asarray(args.rmin))
    js = {str(rm): {"sigma": {n: float(results[rm]["sigma"][_IDX[n]]) for n in names},
                    "s8": results[rm]["s8"],
                    "s8_astro_pinned": results[rm]["s8_astro_pinned"],
                    "bits": results[rm]["bits"], "n_rows": results[rm]["n_rows"]}
          for rm in args.rmin}
    with open(os.path.join(out, f"tier3_forecast_{tag}.json"), "w") as fh:
        json.dump(js, fh, indent=1)
    print(f"[save] {out}/tier3_forecast_{tag}.npz + .json + SUMMARY_{tag}.txt")

    if not args.no_plots:
        from hod_mod.scripts.forecasts.make_tier2_figures import make_all
        make_all(os.path.join(out, f"tier3_forecast_{tag}.npz"), out)


if __name__ == "__main__":
    main()
