r"""Complete sensitivity / Fisher-forecast study for the ZM15 + X-ray pipeline.

Fixes a fiducial model (Planck18 cosmology, BGS M*>10 ZM15 MAP, X-ray S1 band
MAP), makes clean predictions of the five data vectors — w_p, ΔΣ, C_ℓ^{gX},
C_ℓ^{gy}, C_ℓ^{XX} — assigns a constant relative statistical error
(0.1 / 1 / 5 %), and propagates it to parameter constraints via a JAX-autodiff
Fisher matrix, for scale cuts R > {0.1, 0.5, 2.5, 10} Mpc/h.  It then reports the
degeneracies (correlation matrix + eigen-directions) and attributes the
constraining power by probe and by scale.

Because the whole model is differentiable (EH98 linear P(k), not CAMB), the
Jacobian ∂d/∂θ is computed once with ``jax.jacfwd``; every (error, scale-cut,
probe) combination is then a fast linear-algebra derivation via row masking.

Usage::

    JAX_PLATFORMS=cpu HOD_MOD_RESULTS=/path/to/results \
      python -m hod_mod.scripts.forecasts.run_sensitivity_study \
      --rel-err 0.001 0.01 0.05 --rmin 0.1 0.5 2.5 10 --add-planck-prior
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)   # X-ray amplitudes need >float32 range

import jax.numpy as jnp  # noqa: E402

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES, OBSERVABLES  # noqa: E402
from hod_mod.forecast import params, fisher  # noqa: E402


def _out_dir():
    try:
        from hod_mod import paths
        d = os.fspath(paths.results_root() / "sensitivity_fisher")
    except Exception:
        d = os.path.join(os.environ.get("HOD_MOD_RESULTS", "."), "sensitivity_fisher")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rel-err", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    ap.add_argument("--rmin", type=float, nargs="+", default=[0.1, 0.5, 2.5, 10.0])
    ap.add_argument("--observables", nargs="+", default=OBSERVABLES)
    ap.add_argument("--add-planck-prior", action="store_true")
    ap.add_argument("--z-eff", type=float, default=0.2)
    ap.add_argument("--n-k", type=int, default=256)
    ap.add_argument("--n-m", type=int, default=256)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = _out_dir()
    names = PARAM_NAMES
    fid = params.fiducial_vector()
    # Weakly-informative regularizing prior (bounds flat nuisance directions →
    # well-defined marginalised errors); tight Planck prior on Ω_m, σ8 optional.
    prior = params.regularizing_prior(add_planck=args.add_planck_prior)

    print(f"[setup] {len(names)} parameters, observables={args.observables}, z_eff={args.z_eff}")
    model = ForwardModel(z_eff=args.z_eff, n_k=args.n_k, n_m=args.n_m)

    # --- one full Jacobian over the full (unmasked) data vector ---------
    f_full, row_obs, row_x = model.full_data_vector_fn(args.observables)
    print("[jacobian] computing full jax.jacfwd (one pass) ...")
    d0, J = fisher.jacobian(f_full, fid)
    d0 = np.asarray(d0); J = np.asarray(J)
    print(f"[jacobian] data vector = {d0.size} rows, params = {J.shape[1]}; "
          f"finite={np.all(np.isfinite(J))}")

    # clean prediction (fiducial data vectors on full grids)
    pred = {o: np.asarray(v) for o, v in model.predict(jnp.asarray(fid), args.observables).items()}
    grids = {o: np.asarray(model.grid_of(o)) for o in args.observables}

    results = {}          # (rel_err, rmin) -> dict
    for rmin in args.rmin:
        smask = model.scale_cut_mask(row_obs, row_x, rmin)
        for re in args.rel_err:
            F = fisher.fisher_matrix(d0[smask], J[smask], re, prior)
            cov, sig, corr = fisher.constraints(F, ridge=1e-10)
            evals = np.linalg.eigvalsh(F)
            degen = fisher.top_degeneracies(corr, names, k=8)
            eig = fisher.principal_directions(cov, names, sigma_fid=fid, k=3)
            probe = fisher.probe_decomposition(d0, J, row_obs, smask,
                                               args.observables, re, prior)
            results[(re, rmin)] = dict(
                sigma=sig, cov=cov, corr=corr, F=F,
                fisher_min_eval=float(evals.min()), n_data=int(smask.sum()),
                degeneracies=degen, eigen=eig, probe=probe,
            )
            print(f"[fisher] rel_err={re:6.3f} Rmin={rmin:5.2f} "
                  f"Ndata={int(smask.sum()):3d} sig(Om)={sig[0]:.3e} "
                  f"sig(s8)={sig[1]:.3e} Fmin_eval={evals.min():.1e}")

    _save(out, args, names, fid, pred, grids, d0, J, row_obs, row_x, results)
    _write_summary(out, args, names, fid, results)
    if not args.no_plots:
        _make_plots(out, args, names, fid, pred, grids, results)
    print(f"[done] outputs written to {out}")


def _save(out, args, names, fid, pred, grids, d0, J, row_obs, row_x, results):
    np.savez(
        os.path.join(out, "fisher_study.npz"),
        param_names=np.array(names), fiducial=fid,
        rel_err=np.array(args.rel_err), rmin=np.array(args.rmin),
        row_obs=row_obs, row_x=row_x, d0=d0, jacobian=J,
        **{f"pred__{o}": pred[o] for o in pred},
        **{f"grid__{o}": grids[o] for o in grids},
        **{f"sigma__{re}__{rm}": results[(re, rm)]["sigma"]
           for (re, rm) in results},
        **{f"cov__{re}__{rm}": results[(re, rm)]["cov"]
           for (re, rm) in results},
        **{f"corr__{re}__{rm}": results[(re, rm)]["corr"]
           for (re, rm) in results},
    )


def _write_summary(out, args, names, fid, results):
    lines = ["# Sensitivity / Fisher forecast summary", ""]
    lines.append(f"Parameters ({len(names)}): " + ", ".join(names))
    lines.append(f"Fiducial: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, fid)))
    lines.append("")
    for rmin in args.rmin:
        for re in args.rel_err:
            R = results[(re, rmin)]
            lines.append(f"## rel_err={re}  Rmin={rmin} Mpc/h  (Ndata={R['n_data']}, "
                         f"Fisher min eigval={R['fisher_min_eval']:.2e})")
            lines.append("  1σ marginalised constraints:")
            for n, s, fv in zip(names, R["sigma"], fid):
                unc = (not np.isfinite(s))
                frac = (s / abs(fv)) if (fv != 0 and np.isfinite(s)) else np.nan
                fs = f"{frac:8.2%}" if np.isfinite(frac) else "     n/a"
                flag = "  <-- unconstrained" if unc else ""
                lines.append(f"    {n:15s} = {fv:12.5g} ± {s:11.4g}  ({fs}){flag}")
            lines.append("  Top degeneracies (|corr|):")
            for c, a, b, cc in R["degeneracies"]:
                lines.append(f"    {a:15s} x {b:15s}  corr={cc:+.3f}")
            lines.append("  Worst-constrained directions (θ/θ_fid eigen-basis):")
            for e in R["eigen"]:
                comp = ", ".join(f"{nm}:{v:+.2f}" for nm, v in e["components"])
                lines.append(f"    sqrt(var)={np.sqrt(e['variance']):.3g}: {comp}")
            lines.append("")
    # probe attribution at the tightest error, per Rmin
    re0 = min(args.rel_err)
    lines.append(f"## Probe attribution (rel_err={re0}) — σ(θ) from each probe alone")
    for rmin in args.rmin:
        lines.append(f"  Rmin={rmin} Mpc/h:")
        single = results[(re0, rmin)]["probe"]["single"]
        for probe, sig in single.items():
            key = ", ".join(f"{n}:{s:.2g}" for n, s in
                            zip(names[:2] + ["log10DC", "p2"],
                                [sig[0], sig[1], sig[PARAM_NAMES.index("log10DC")],
                                 sig[PARAM_NAMES.index("p2")]]))
            lines.append(f"    {probe:6s}: {key}")
    with open(os.path.join(out, "SUMMARY.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _make_plots(out, args, names, fid, pred, grids, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (0) clean prediction data vectors
    fig, axes = plt.subplots(1, len(pred), figsize=(3.2 * len(pred), 3.0))
    if len(pred) == 1:
        axes = [axes]
    for ax, o in zip(axes, args.observables):
        ax.loglog(grids[o], np.abs(pred[o]), "o-", ms=3)
        ax.set_title(o); ax.set_xlabel("r_p [Mpc/h]" if o in ("wp", "ds") else r"$\ell$")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_predictions.pdf")); plt.close(fig)

    # (1) sigma vs Rmin for key params, one line per rel_err
    key_params = ["Omega_m", "sigma8", "log10_M_pivot", "log10DC", "p2", "bsat"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, kp in zip(axes.ravel(), key_params):
        j = names.index(kp)
        for re in args.rel_err:
            ys = [results[(re, rm)]["sigma"][j] for rm in args.rmin]
            ax.loglog(args.rmin, ys, "o-", label=f"{re*100:g}%")
        ax.set_title(params.PARAM_LATEX[kp]); ax.set_xlabel(r"$R_{\min}$ [Mpc/h]")
        ax.set_ylabel(r"$\sigma$")
    axes.ravel()[0].legend(title="rel. err", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_sigma_vs_scale.pdf")); plt.close(fig)

    # (2) correlation matrix at the reference config
    re_ref = args.rel_err[len(args.rel_err) // 2]
    rm_ref = min(args.rmin)
    corr = results[(re_ref, rm_ref)]["corr"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"parameter correlation (rel_err={re_ref}, Rmin={rm_ref})")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_correlation.pdf")); plt.close(fig)

    # (3) probe attribution: sigma(sigma8) and sigma(log10DC) per probe vs Rmin
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kp in zip(axes, ["sigma8", "log10DC"]):
        j = names.index(kp)
        for probe in args.observables:
            ys = []
            for rm in args.rmin:
                sd = results[(min(args.rel_err), rm)]["probe"]["single"]
                ys.append(sd[probe][j] if probe in sd else np.nan)
            ax.loglog(args.rmin, ys, "o-", label=probe)
        # combined
        ys = [results[(min(args.rel_err), rm)]["sigma"][j] for rm in args.rmin]
        ax.loglog(args.rmin, ys, "k--", lw=2, label="combined")
        ax.set_title(f"σ({params.PARAM_LATEX[kp]}) per probe")
        ax.set_xlabel(r"$R_{\min}$ [Mpc/h]")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_probe_attribution.pdf")); plt.close(fig)


if __name__ == "__main__":
    main()
