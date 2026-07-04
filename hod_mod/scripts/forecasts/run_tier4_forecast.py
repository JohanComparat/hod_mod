r"""Tier-4 Fisher forecast: 111 parameters, the morphology observables.

Extends the tier-3 production run (``run_tier3_forecast``) with the
morphology measurements of :class:`hod_mod.forecast.tier4.Tier4Forecast`:
the per-cell early-type fraction, joint early∩quenched fraction, mean galaxy
size (Kravtsov R_e–R_200c) and galaxy–IA cross w_g+; the per-shell AGN-host
early fraction; and early/late-split (w_p, ΔΣ, n̄_g) blocks to z = 1.2.

Usage::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier4_forecast \
        --jobs 6
    # quick end-to-end check (2×2 cells, tiny grids):
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier4_forecast --smoke
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from hod_mod.forecast.forward_jax import (  # noqa: E402
    PARAM_NAMES, _IDX, TIER4_MORPHOLOGY, WAVE4_MORPHOLOGY)
from hod_mod.forecast import params, fisher  # noqa: E402
from hod_mod.forecast.tier4 import Tier4Forecast  # noqa: E402
from hod_mod.scripts.forecasts.run_tier2_forecast import (  # noqa: E402
    _s8_sigma, _sig, _bits_per_sector, EVOL_PARAMS)
from hod_mod.scripts.forecasts.run_tier3_forecast import (  # noqa: E402
    PROBE_GROUPS as TIER3_PROBE_GROUPS)

# tier-3 order + the morphology families; "+morph-split wp/dS" selects by
# block KIND (its observable names are the generic wp/ds/n_gal)
PROBE_GROUPS = list(TIER3_PROBE_GROUPS) + [
    ("+f_early(z,M*)", ("f_early",)),
    ("+E∩Q joint", ("f_early_q",)),
    ("+sizes", ("size",)),
    ("+AGN hosts", ("f_early_agn",)),
    ("+morph-split wp/dS", ("__kind__morph_cell",)),
    ("+IA w_g+", ("wgp",)),
]


def _out_dir():
    try:
        from hod_mod import paths
        d = os.fspath(paths.results_root() / "tier4_forecast")
    except Exception:
        d = os.path.join(os.environ.get("HOD_MOD_RESULTS", "."), "tier4_forecast")
    os.makedirs(d, exist_ok=True)
    return d


def _group_mask(meta, obs_group):
    """Row mask for a probe group; ``__kind__<k>`` entries select by block
    kind, plain entries by observable name OUTSIDE special kinds."""
    kinds = [g[len("__kind__"):] for g in obs_group if g.startswith("__kind__")]
    names = [g for g in obs_group if not g.startswith("__kind__")]
    m = np.isin(meta["obs"], names) & (meta["kind"] != "morph_cell")
    for k in kinds:
        m |= meta["kind"] == k
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rmin", type=float, nargs="+", default=[0.1, 0.5, 2.5])
    ap.add_argument("--n-bands", type=int, default=6, choices=[1, 6, 15])
    ap.add_argument("--n-k", type=int, default=96)
    ap.add_argument("--n-gl", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-morph-split", action="store_true")
    ap.add_argument("--no-ia", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = _out_dir()
    cache = args.cache_dir or os.path.join(out, "cache")
    tag = f"nb{args.n_bands}" + ("_smoke" if args.smoke else "")

    common = dict(include_morph_split=not args.no_morph_split,
                  include_ia=not args.no_ia)
    if args.smoke:
        t4 = Tier4Forecast(
            z_edges=[0.2, 0.4, 0.6], mstar_edges=[9.4, 9.6, 10.6],
            n_bands=[(0.5, 1.0), (1.0, 2.0)], n_shear_bins=2,
            agn_lx_bins=[(42.0, 42.5), (42.5, 43.0)], agn_z_centers=(0.3,),
            n_k=48, n_m=48, n_gl=16, n_z=3, cell_n_m=96,
            rp_wp=np.logspace(-1, 1.4, 6), rp_ds=np.logspace(-1, 1.2, 5),
            ell=np.logspace(1.0, 3.3, 6), rp_wp_agn=np.logspace(0.1, 1.4, 4),
            **common)
    else:
        t4 = Tier4Forecast(n_bands=args.n_bands, n_k=args.n_k,
                           n_gl=args.n_gl, **common)

    fid = t4.fiducial()
    names = PARAM_NAMES
    n_cells = sum(1 for b in t4.blocks if b.kind == "cell")
    n_morph = sum(1 for b in t4.blocks if b.kind == "morph_cell")
    print(f"[setup] tier-4: {len(names)} parameters "
          f"({len(WAVE4_MORPHOLOGY) + len(TIER4_MORPHOLOGY)} morphology), "
          f"{n_cells} cell + {n_morph} morph-split blocks, "
          f"{len(t4.blocks)} blocks total")

    if args.jobs > 1:
        missing = t4.precompute_blocks(fid, cache, jobs=args.jobs)
        print(f"[precompute] {len(missing)} blocks with {args.jobs} workers")
    d0, J, meta = t4.data_and_jacobian(fid, cache_dir=cache)
    print(f"[jacobian] {d0.size} rows x {J.shape[1]} params; "
          f"finite={np.all(np.isfinite(J))}")

    sig_noise = t4.noise_sigma(fid, d0, meta)
    rel = np.where(np.abs(d0) > 0, sig_noise / np.abs(d0), np.inf)
    finite = np.isfinite(rel)
    print(f"[noise] median rel err = {np.median(rel[finite]):.3g}; "
          f"{np.sum(~finite)} rows at sigma=inf")

    prior = params.regularizing_prior(fix=("log10DC",))
    scale = params.regularizing_prior()

    results = {}
    for rmin in args.rmin:
        keep = t4.scale_cut_mask(meta, rmin) & finite
        cov, sig, corr = _sig(d0, J, keep, rel, prior, scale)
        astro = [n for n in names if n not in params.SECTORS["cosmology"]]
        prior_ap = params.regularizing_prior(fix=tuple(astro))
        cov_ap, sig_ap, _ = _sig(d0, J, keep, rel, prior_ap, scale)
        prior_cp = params.regularizing_prior(
            fix=tuple(params.SECTORS["cosmology"]) + ("log10DC",))
        cov_cp, sig_cp, _ = _sig(d0, J, keep, rel, prior_cp, scale)

        cum, mask_cum = {}, np.zeros(d0.size, dtype=bool)
        for label, obs_group in PROBE_GROUPS:
            mask_cum |= _group_mask(meta, obs_group)
            _, s_c, _ = _sig(d0, J, keep & mask_cum, rel, prior, scale)
            cum[label] = s_c

        r = dict(rmin=rmin, sigma=sig, cov=cov, corr=corr, keep=keep,
                 sigma_astro_pinned=sig_ap, cov_astro_pinned=cov_ap,
                 sigma_cosmo_pinned=sig_cp,
                 s8=_s8_sigma(cov, fid), s8_astro_pinned=_s8_sigma(cov_ap, fid),
                 bits=_bits_per_sector(cov, prior),
                 cumulative=cum, n_rows=int(keep.sum()))
        results[rmin] = r
        print(f"\n[rmin={rmin}] rows={keep.sum()}")
        for n in params.SECTORS["cosmology"]:
            i = _IDX[n]
            print(f"    sigma({n:8s}) = {sig[i]:9.2e}  vs pinned {sig_ap[i]:9.2e}"
                  f"   (x{sig[i] / sig_ap[i]:6.1f})")
        print("  bits: " + ", ".join(f"{k}={v:.1f}" for k, v in r["bits"].items()))

    rm0 = args.rmin[0]
    r0 = results[rm0]
    lines = [f"TIER-4 FORECAST SUMMARY ({tag})",
             f"params={len(names)}  rows={d0.size}  cells={n_cells}  "
             f"morph_cells={n_morph}", ""]
    lines.append(f"--- marginalized 1-sigma at rmin={rm0} Mpc/h ---")
    for n in names:
        i = _IDX[n]
        ref = (r0["sigma_astro_pinned"][i]
               if n in params.SECTORS["cosmology"] else r0["sigma_cosmo_pinned"][i])
        sec = next(s for s, nn in params.SECTORS.items() if n in nn)
        lines.append(f"{n:20s} [{sec:10s}] sigma={r0['sigma'][i]:9.3e}  "
                     f"pinned-ref={ref:9.3e}  prior={prior[i]:8.2g}")
    lines.append("")
    lines.append("--- morphology sector (waves 4 + tier 4) ---")
    for n in list(WAVE4_MORPHOLOGY) + list(TIER4_MORPHOLOGY):
        i = _IDX[n]
        lines.append(f"{n:20s} sigma={r0['sigma'][i]:9.3e}  (prior {prior[i]:.2g})")
    lines.append("")
    lines.append("--- evolution parameters ---")
    for n in EVOL_PARAMS + ["f_size_zs"]:
        i = _IDX[n]
        lines.append(f"{n:20s} sigma={r0['sigma'][i]:9.3e}  (prior {prior[i]:.2g})")
    lines.append("")
    lines.append("--- cumulative probe attribution ---")
    for label, s_c in r0["cumulative"].items():
        lines.append(f"{label:20s} sigma(Om)={s_c[_IDX['Omega_m']]:9.3e}  "
                     f"sigma(s8)={s_c[_IDX['sigma8']]:9.3e}")
    lines.append("")
    lines.append("--- top degeneracies ---")
    for a, n1, n2, c in fisher.top_degeneracies(r0["corr"], names, k=10):
        lines.append(f"corr({n1}, {n2}) = {c:+.3f}")
    if t4.completeness_flags:
        lines.append("")
        lines.append(f"--- {len(t4.completeness_flags)} completeness flags ---")
        for lab, o, xv in t4.completeness_flags:
            lines.append(f"  {lab} {o} x={xv}")
    summary = "\n".join(lines)
    with open(os.path.join(out, f"SUMMARY_{tag}.txt"), "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)

    np.savez_compressed(
        os.path.join(out, f"tier4_forecast_{tag}.npz"),
        param_names=np.array(names), fid=fid, d0=d0, J=J,
        sigma_noise=sig_noise, prior=prior,
        **{f"meta_{k}": v for k, v in meta.items()},
        **{f"sigma_rmin{rm}": results[rm]["sigma"] for rm in args.rmin},
        **{f"cov_rmin{rm}": results[rm]["cov"] for rm in args.rmin},
        **{f"keep_rmin{rm}": results[rm]["keep"] for rm in args.rmin},
        n_bands=len(t4.bands),
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
    with open(os.path.join(out, f"tier4_forecast_{tag}.json"), "w") as fh:
        json.dump(js, fh, indent=1)
    print(f"[save] {out}/tier4_forecast_{tag}.npz + .json + SUMMARY_{tag}.txt")

    if not args.no_plots:
        from hod_mod.scripts.forecasts.make_tier2_figures import make_all
        make_all(os.path.join(out, f"tier4_forecast_{tag}.npz"), out)


if __name__ == "__main__":
    main()
