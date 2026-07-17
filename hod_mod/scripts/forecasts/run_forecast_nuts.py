r"""Full posterior for the forecast forward model via Hamiltonian Monte Carlo.

This is the payoff of the differentiable stack: instead of the Gaussian/linear
Fisher approximation (:mod:`hod_mod.forecast.fisher`), it samples the *actual*
posterior with blackjax NUTS (Betancourt 2017, arXiv:1701.02434), which needs
exact gradients of the log-posterior — supplied by ``jax.grad`` straight through
:class:`~hod_mod.forecast.forward_jax.ForwardModel`.

The cosmology backend defaults to the **CosmoPower-JAX emulator**
(``pk_correction="cosmopower"``, see :mod:`hod_mod.forecast.pk_cosmopower`),
which is CAMB-accurate to <0.1% *and* differentiable — so the posterior is
trustworthy, not merely fast.  EH98 (``none`` / ``camb_linear``) remains
selectable for comparison.

Nothing here reimplements inference: the likelihood
(:class:`~hod_mod.fitting.jax_inference.MultiProbeGaussianLikelihood`), the
gradient MAP (``run_map_jax``) and the sampler (``run_nuts``) are reused as-is.

With ``--compare-fisher`` (default) the run also reports the Fisher σ from the
same Jacobian machinery next to the NUTS σ; in the Gaussian regime they should
agree, which validates both the emulator swap and the sampler.

Observables default to the projected/abundance set (``wp``/``smf``): the angular
spectra do a Limber projection that unrolls into the NUTS trajectory
``while_loop`` and inflates the compile ~10× (see ``run_nuts``'s docstring).

Usage::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu HOD_MOD_RESULTS=/path/to/results \
      python -m hod_mod.scripts.forecasts.run_forecast_nuts \
      --which wp smf --free cosmo --n-warmup 400 --n-samples 800
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)   # gradients need >float32

import jax.numpy as jnp  # noqa: E402

from hod_mod.forecast.forward_jax import ForwardModel  # noqa: E402
from hod_mod.forecast import params, fisher  # noqa: E402
from hod_mod.fitting.jax_inference import (  # noqa: E402
    MultiProbeGaussianLikelihood,
    run_map_jax,
    run_nuts,
)

# Free-parameter presets.  "cosmo" is the natural POC set: the 5 cosmology
# parameters the emulator makes differentiable.  "cosmo_hod" adds the ZM15
# occupation core, which is where the interesting HOD/cosmology degeneracies live.
_FREE_PRESETS = {
    "cosmo": ["Omega_m", "sigma8", "h", "n_s", "Omega_b"],
    "cosmo_hod": ["Omega_m", "sigma8", "h", "n_s", "Omega_b",
                  "lg_m1h", "lg_m0star", "beta", "delta", "gamma"],
    "hod": ["lg_m1h", "lg_m0star", "beta", "delta", "gamma", "sigma_lnmstar"],
}


def _out_dir():
    try:
        from hod_mod import paths
        d = os.fspath(paths.results_root() / "forecast_nuts")
    except Exception:                                        # pragma: no cover
        d = "results/forecast_nuts"
    os.makedirs(d, exist_ok=True)
    return d


def _diagnostics(samples):
    """split-R̂ and ESS for a single chain, via blackjax's diagnostics.

    R̂ compares the two halves of the chain (the standard single-chain split
    diagnostic); ESS accounts for autocorrelation.  Both want a leading chain
    axis, so the halves are stacked as two pseudo-chains.
    """
    try:
        from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
    except ImportError:                                      # pragma: no cover
        return None, None
    n = samples.shape[0] // 2
    if n < 2:
        return None, None
    split = jnp.stack([samples[:n], samples[n:2 * n]])       # (2, n, n_free)
    rhat = np.asarray(potential_scale_reduction(split, chain_axis=0, sample_axis=1))
    ess = np.asarray(effective_sample_size(split, chain_axis=0, sample_axis=1))
    return rhat, ess


def _fisher_sigma(like, x0, rel_err):
    """Fisher σ over the *free* parameters, from the same autodiff Jacobian.

    Differentiates the reduced map ``x -> f(theta0[free]=x)`` so the Jacobian has
    n_free columns rather than the full 111 (forward-mode jacfwd costs one tangent
    per column; the full vector through the emulator is needlessly large here).
    """
    d0, J = fisher.jacobian(lambda x: like.model(x), jnp.asarray(x0))
    prior_sigma = np.where(like._prior_prec > 0,
                           1.0 / np.sqrt(np.where(like._prior_prec > 0,
                                                  like._prior_prec, 1.0)),
                           np.inf)
    F = fisher.fisher_matrix(d0, J, rel_err=rel_err, prior_sigma=prior_sigma)
    _cov, sigma, _corr = fisher.constraints(F)
    return sigma


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", nargs="+", default=["wp", "smf"],
                   help="observables to include (default: wp smf)")
    p.add_argument("--free", default="cosmo",
                   help="preset name (%s) or explicit parameter names"
                        % "/".join(_FREE_PRESETS))
    p.add_argument("--free-names", nargs="+", default=None,
                   help="explicit free parameter names (overrides --free)")
    p.add_argument("--pk", default="cosmopower",
                   choices=["cosmopower", "none", "camb_linear"],
                   help="linear P(k) backend (default: cosmopower emulator)")
    p.add_argument("--rel-err", type=float, default=0.05,
                   help="fractional error per data bin (default: 0.05)")
    p.add_argument("--n-warmup", type=int, default=400)
    p.add_argument("--n-samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-k", type=int, default=256)
    p.add_argument("--n-m", type=int, default=256)
    p.add_argument("--z-eff", type=float, default=0.25)
    p.add_argument("--no-map", action="store_true",
                   help="start NUTS at the truth instead of the gradient MAP")
    p.add_argument("--compare-fisher", dest="compare_fisher",
                   action="store_true", default=True)
    p.add_argument("--no-compare-fisher", dest="compare_fisher",
                   action="store_false")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    free_names = args.free_names or _FREE_PRESETS.get(args.free)
    if free_names is None:
        raise SystemExit("--free must be one of %s, or pass --free-names"
                         % list(_FREE_PRESETS))
    out = args.out_dir or _out_dir()

    print(f"[forecast-nuts] P(k) backend : {args.pk}")
    print(f"[forecast-nuts] observables  : {args.which}")
    print(f"[forecast-nuts] free params  : {free_names}")

    model = ForwardModel(z_eff=args.z_eff, n_k=args.n_k, n_m=args.n_m,
                         pk_correction=args.pk)

    # synthetic data = model(fiducial) + Gaussian noise, diagonal rel_err errors
    like, x_true = MultiProbeGaussianLikelihood.synthetic(
        model, args.which, free_names, rel_err=args.rel_err,
        prior="planck", seed=args.seed)

    x0 = np.asarray(x_true, dtype=float)
    if not args.no_map:
        print("[forecast-nuts] gradient MAP (L-BFGS-B) ...")
        map_res = run_map_jax(like, x0)
        x0 = np.asarray(map_res["x"], dtype=float)
        print(f"[forecast-nuts]   MAP chi2 = {like.chi2(jnp.asarray(x0)):.2f}")

    print(f"[forecast-nuts] NUTS: {args.n_warmup} warmup + {args.n_samples} samples ...")
    res = run_nuts(like, x0, n_warmup=args.n_warmup, n_samples=args.n_samples,
                   seed=args.seed)
    samples = res["samples"]
    rhat, ess = _diagnostics(jnp.asarray(samples))

    fisher_sigma = None
    if args.compare_fisher:
        print("[forecast-nuts] Fisher cross-check ...")
        fisher_sigma = _fisher_sigma(like, x_true, args.rel_err)

    # ---- report -------------------------------------------------------
    print(f"\n[forecast-nuts] accept_rate={res['accept_rate']:.3f} "
          f"step_size={res['step_size']:.3g}")
    hdr = f"{'param':>14s} {'truth':>10s} {'NUTS mean':>11s} {'NUTS sig':>10s}"
    if fisher_sigma is not None:
        hdr += f" {'Fisher sig':>11s} {'ratio':>7s}"
    hdr += f" {'Rhat':>6s} {'ESS':>7s}"
    print(hdr)
    rows = {}
    for i, name in enumerate(free_names):
        line = (f"{name:>14s} {x_true[i]:10.4f} {res['mean'][i]:11.4f} "
                f"{res['std'][i]:10.4f}")
        row = {"truth": float(x_true[i]), "nuts_mean": float(res["mean"][i]),
               "nuts_sigma": float(res["std"][i])}
        if fisher_sigma is not None:
            ratio = res["std"][i] / fisher_sigma[i] if fisher_sigma[i] > 0 else np.nan
            line += f" {fisher_sigma[i]:11.4f} {ratio:7.2f}"
            row["fisher_sigma"] = float(fisher_sigma[i])
            row["nuts_over_fisher"] = float(ratio)
        if rhat is not None:
            line += f" {rhat[i]:6.3f} {ess[i]:7.0f}"
            row["rhat"] = float(rhat[i])
            row["ess"] = float(ess[i])
        print(line)
        rows[name] = row

    tag = f"{args.pk}_{'-'.join(args.which)}_{args.free}"
    np.savez(os.path.join(out, f"nuts_{tag}_chain.npz"),
             samples=samples, free_names=np.array(free_names), x_true=x_true)
    summary = {"pk_backend": args.pk, "which": args.which,
               "free_names": free_names, "rel_err": args.rel_err,
               "n_warmup": args.n_warmup, "n_samples": args.n_samples,
               "accept_rate": float(res["accept_rate"]),
               "step_size": float(res["step_size"]), "params": rows}
    with open(os.path.join(out, f"nuts_{tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[forecast-nuts] wrote {out}/nuts_{tag}_{{chain.npz,summary.json}}")
    return summary


if __name__ == "__main__":
    main()
