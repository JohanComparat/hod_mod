r"""Tier-2 Fisher forecast: ALL 61 parameters free, (z, M*) grid, physical noise.

The tier-2 study answers one question for an optimistic Stage-IV data scenario:
**how much cosmology and how much astrophysics does the combined data teach us
when nothing is fixed?**  Compared with the tier-1 studies
(``run_sensitivity_study`` / ``run_stage4_forecast``) it

* frees every parameter — the 31-entry tier-1 vector plus the 16 promoted
  nuisance shapes, 7 redshift-evolution slopes and 7 X-ray spectral parameters
  (61 total; only the retired ``log10DC`` is pinned, since the Powell AGN
  emission replaces the duty-cycle surrogate);
* tiles the galaxy sample over the (z, M*) plane: Δz = 0.1 shells (z < 1) ×
  0.2-dex volume-limited M* bins (M* > 1e10) = 80 cells sharing ONE parameter
  vector (redshift evolution is explicit, not per-bin);
* adds the z-resolved soft-band AGN XLF and the projected AGN clustering
  ``wp_agn`` in 0.5-dex L_X bins (complete above 1e42 erg/s), the tomographic
  Euclid+LSST shear (30 arcmin⁻², f_sky = 0.5, 5 source bins), and the
  multi-band APEC X-ray layer for a hypothetical Athena all-sky survey
  (F_lim = 2e-16: exactly the depth making L_X > 1e42 complete to z = 1);
* replaces the effective (rN, aN) error recipe with physical noise —
  pair counts, shape noise, CXB photon statistics, Poisson XLF counts.

Usage::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier2_forecast \
        --rmin 0.1 0.5 2.5 --n-bands 6
    # quick end-to-end check (2×2 cells, tiny grids, ~2 min):
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier2_forecast --smoke
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from hod_mod.forecast.forward_jax import PARAM_NAMES, _IDX, TIER2_ZSLOPES  # noqa: E402
from hod_mod.forecast import params, fisher  # noqa: E402
from hod_mod.forecast.tier2 import Tier2Forecast  # noqa: E402

# probe groups for the cumulative attribution (galaxy grid → lensing →
# X-ray/tSZ crosses → AGN abundance → AGN clustering)
PROBE_GROUPS = [
    ("galaxy grid", ("wp", "ds", "n_gal", "ssfr")),
    ("+lensing", ("cl_kk", "cl_kCMB", "cl_shear_kCMB", "cl_gkCMB")),
    ("+X-ray/tSZ", ("cl_gX", "cl_XX", "cl_gy")),
    ("+XLF(z)", ("xlf",)),
    ("+wp_agn", ("wp_agn",)),
    ("+radio LF", ("rlf",)),
    ("+HI", ("himf", "cl_gHI")),
]
EVOL_PARAMS = list(TIER2_ZSLOPES) + ["z_gas_zs", "agn_mu_bh_zs"]


def _out_dir():
    try:
        from hod_mod import paths
        d = os.fspath(paths.results_root() / "tier2_forecast")
    except Exception:
        d = os.path.join(os.environ.get("HOD_MOD_RESULTS", "."), "tier2_forecast")
    os.makedirs(d, exist_ok=True)
    return d


def _s8_sigma(cov, fid):
    """σ(S8), S8 = σ8 (Ω_m/0.3)^0.5, by linear propagation."""
    iom, is8 = _IDX["Omega_m"], _IDX["sigma8"]
    om, s8 = fid[iom], fid[is8]
    g = np.zeros(len(fid))
    g[is8] = (om / 0.3) ** 0.5
    g[iom] = 0.5 * s8 * (om / 0.3) ** 0.5 / om
    return float(np.sqrt(g @ cov @ g))


def _sig(d0, J, mask, rel, prior, scale):
    F = fisher.fisher_matrix(d0[mask], J[mask], rel_err=rel[mask],
                             prior_sigma=prior)
    return fisher.constraints(F, scale=scale)


def _bits_per_sector(cov, prior):
    """Information gain per sector: ½ log2 det(C_prior)/det(C_post) [bits]."""
    out = {}
    for sec, names in params.SECTORS.items():
        if sec == "retired":
            continue
        idx = [_IDX[n] for n in names]
        cp = np.diag(np.asarray(prior)[idx] ** 2)
        cs = cov[np.ix_(idx, idx)]
        s, ldp = np.linalg.slogdet(cp)
        s2, lds = np.linalg.slogdet(cs)
        out[sec] = float(0.5 * (ldp - lds) / np.log(2.0)) if s > 0 and s2 > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rmin", type=float, nargs="+", default=[0.1, 0.5, 2.5])
    ap.add_argument("--n-bands", type=int, default=6, choices=[1, 6, 15],
                    help="X-ray energy bands over 0.5-2 keV")
    ap.add_argument("--n-k", type=int, default=96)
    ap.add_argument("--n-m", type=int, default=128)
    ap.add_argument("--n-gl", type=int, default=48)
    ap.add_argument("--cache-dir", default=None,
                    help="per-block Jacobian cache (default <out>/cache)")
    ap.add_argument("--smoke", action="store_true",
                    help="2x2 cells, tiny grids: fast end-to-end check")
    ap.add_argument("--free-log10dc", action="store_true",
                    help="do not pin the retired duty-cycle parameter")
    ap.add_argument("--split-sfq", action="store_true",
                    help="split every (z, M*) cell into SF and quiescent samples")
    ap.add_argument("--include-radio", action="store_true",
                    help="add the fundamental-plane radio LF per shell")
    ap.add_argument("--include-hi", action="store_true",
                    help="add the HIMF (low-z shells) and 21cm×galaxy crosses")
    ap.add_argument("--include-ssfr", action="store_true",
                    help="add the main-sequence mean sSFR datum per cell")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = _out_dir()
    cache = args.cache_dir or os.path.join(out, "cache")
    tag = f"nb{args.n_bands}" + ("_smoke" if args.smoke else "")

    if args.smoke:
        t2 = Tier2Forecast(
            z_edges=[0.1, 0.2, 0.3], mstar_edges=[10.0, 10.4, 10.8],
            n_bands=[(0.5, 1.0), (1.0, 2.0)], n_shear_bins=2,
            agn_lx_bins=[(42.0, 42.5), (42.5, 43.0)], agn_z_centers=(0.15,),
            n_k=48, n_m=48, n_gl=16, n_z=3,
            rp_wp=np.logspace(-1, 1.4, 6), rp_ds=np.logspace(-1, 1.2, 5),
            ell=np.logspace(1.0, 3.3, 6), rp_wp_agn=np.logspace(0.1, 1.4, 4))
    else:
        t2 = Tier2Forecast(n_bands=args.n_bands, n_k=args.n_k, n_m=args.n_m,
                           n_gl=args.n_gl, split_sfq=args.split_sfq,
                           include_radio=args.include_radio,
                           include_hi=args.include_hi,
                           include_ssfr=args.include_ssfr)

    fid = t2.fiducial()
    names = PARAM_NAMES
    n_cells = sum(1 for b in t2.blocks if b.kind == "cell")
    print(f"[setup] tier-2: {len(names)} parameters, {n_cells} (z,M*) cells, "
          f"{len(t2.bands)} X-ray bands, {t2.n_shear_bins} shear bins, "
          f"{len(t2.blocks)} blocks")
    print(f"[athena] F_lim={t2.athena.f_lim:.1e} erg/s/cm² -> "
          f"t·A_eff={t2.athena.exposure_area:.2e} s·cm²; "
          f"L_lim(z=1)={t2.athena.l_lim(1.0, fid[_IDX['h']], fid[_IDX['Omega_m']]):.2e} erg/s")

    d0, J, meta = t2.data_and_jacobian(fid, cache_dir=cache)
    print(f"[jacobian] {d0.size} rows x {J.shape[1]} params; "
          f"finite={np.all(np.isfinite(J))}")

    sig_noise = t2.noise_sigma(fid, d0, meta)
    rel = np.where(np.abs(d0) > 0, sig_noise / np.abs(d0), np.inf)
    finite = np.isfinite(rel)
    print(f"[noise] median rel err = {np.median(rel[finite]):.3g}; "
          f"{np.sum(~finite)} rows at sigma=inf (completeness)")

    fix = () if args.free_log10dc else ("log10DC",)
    prior = params.regularizing_prior(fix=fix)
    scale = params.regularizing_prior()          # conditioning scale (finite)

    results = {}
    for rmin in args.rmin:
        keep = t2.scale_cut_mask(meta, rmin) & finite
        cov, sig, corr = _sig(d0, J, keep, rel, prior, scale)

        # cosmology: marginalized over ALL astrophysics vs astro-pinned
        astro = [n for n in names if n not in params.SECTORS["cosmology"]]
        prior_ap = params.regularizing_prior(fix=tuple(astro))
        cov_ap, sig_ap, _ = _sig(d0, J, keep, rel, prior_ap, scale)

        # per-sector astro: marginalized vs cosmology externally pinned
        prior_cp = params.regularizing_prior(
            fix=tuple(params.SECTORS["cosmology"]) + fix)
        cov_cp, sig_cp, _ = _sig(d0, J, keep, rel, prior_cp, scale)

        # cumulative probe attribution
        cum = {}
        got = []
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
                 cumulative={k: v for k, v in cum.items()},
                 n_rows=int(keep.sum()))
        results[rmin] = r
        print(f"\n[rmin={rmin}] rows={keep.sum()}")
        print(f"  cosmology (marginalized over 55 astro params vs astro-pinned):")
        for n in params.SECTORS["cosmology"]:
            i = _IDX[n]
            print(f"    sigma({n:8s}) = {sig[i]:9.2e}  vs pinned {sig_ap[i]:9.2e}"
                  f"   (degradation x{sig[i] / sig_ap[i]:6.1f})")
        print(f"    sigma(S8      ) = {r['s8']:9.2e}  vs pinned {r['s8_astro_pinned']:9.2e}")
        print(f"  information gained [bits]: " +
              ", ".join(f"{k}={v:.1f}" for k, v in r["bits"].items()))

    # ------- summary + npz ------------------------------------------------
    rm0 = args.rmin[0]
    r0 = results[rm0]
    lines = [f"TIER-2 FORECAST SUMMARY ({tag})",
             f"params={len(names)}  rows={d0.size}  cells={n_cells}  "
             f"bands={len(t2.bands)}  shear_bins={t2.n_shear_bins}", ""]
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
    lines.append("--- evolution parameters (the new tier-2 science) ---")
    for n in EVOL_PARAMS:
        i = _IDX[n]
        lines.append(f"{n:20s} sigma={r0['sigma'][i]:9.3e}  (prior {prior[i]:.2g})")
    lines.append("")
    lines.append("--- cumulative probe attribution: sigma(Omega_m), sigma(sigma8) ---")
    for label, s_c in r0["cumulative"].items():
        lines.append(f"{label:14s} sigma(Om)={s_c[_IDX['Omega_m']]:9.3e}  "
                     f"sigma(s8)={s_c[_IDX['sigma8']]:9.3e}")
    lines.append("")
    lines.append("--- top degeneracies ---")
    for a, n1, n2, c in fisher.top_degeneracies(r0["corr"], names, k=10):
        lines.append(f"corr({n1}, {n2}) = {c:+.3f}")
    lines.append("")
    for d in fisher.principal_directions(r0["cov"], names, sigma_fid=fid, k=3):
        comp = " ".join(f"{v:+.2f}{n}" for n, v in d["components"])
        lines.append(f"worst direction (var {d['variance']:.2e}): {comp}")
    if t2.completeness_flags:
        lines.append("")
        lines.append(f"--- completeness: {len(t2.completeness_flags)} rows "
                     f"dropped (below L_lim) ---")
        for lab, o, xv in t2.completeness_flags:
            lines.append(f"  {lab} {o} log10Lx={xv:.2f}")
    summary = "\n".join(lines)
    with open(os.path.join(out, f"SUMMARY_{tag}.txt"), "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)

    np.savez_compressed(
        os.path.join(out, f"tier2_forecast_{tag}.npz"),
        param_names=np.array(names), fid=fid, d0=d0, J=J,
        sigma_noise=sig_noise, prior=prior,
        **{f"meta_{k}": v for k, v in meta.items()},
        **{f"sigma_rmin{rm}": results[rm]["sigma"] for rm in args.rmin},
        **{f"cov_rmin{rm}": results[rm]["cov"] for rm in args.rmin},
        **{f"keep_rmin{rm}": results[rm]["keep"] for rm in args.rmin},
        n_bands=len(t2.bands),
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
    with open(os.path.join(out, f"tier2_forecast_{tag}.json"), "w") as fh:
        json.dump(js, fh, indent=1)
    print(f"[save] {out}/tier2_forecast_{tag}.npz + .json + SUMMARY_{tag}.txt")

    if not args.no_plots:
        from hod_mod.scripts.forecasts.make_tier2_figures import make_all
        make_all(os.path.join(out, f"tier2_forecast_{tag}.npz"), out)


if __name__ == "__main__":
    main()
