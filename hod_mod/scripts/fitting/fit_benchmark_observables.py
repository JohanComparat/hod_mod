r"""MAP + MCMC on the benchmark observables — the multi-probe forecast fit.

Wires the published-measurement tree
(``$HOD_MOD_DATA_DIR/benchmark_observables/``, built by
``hod_mod.scripts.data.make_benchmark_observables``) into the differentiable
inference stack of :mod:`hod_mod.fitting.jax_inference`.  This is the concrete
answer to *"prepare a MAP+MCMC run on these observations interfacing the model
and the data"*: it builds the 12-observable forecast
:class:`~hod_mod.forecast.forward_jax.ForwardModel`, loads the benchmark data
vector aligned to the model's own rows via
:func:`hod_mod.fitting.benchmark_data.load_forecast_vector`, and drives a
gradient MAP (``run_map_jax``) followed by blackjax NUTS (``run_nuts``) under a
Planck prior on the cosmological parameters.

Two data modes for the ``simulated`` benchmark entries (forward-model fiducial +
Stage-IV forecast noise, dumped from the tier-2/tier-3 grid):

* ``--data-mode recover`` (default) — the honest use of the stand-ins: take the
  **forecast noise** ``y_err`` from the tree, generate self-consistent data
  ``= model(θ_fiducial) + N(0, y_err)``, and check the pipeline recovers the
  fiducial within the forecast errors.  A single forward-model configuration
  cannot reproduce the *values* of the multi-cell tier-2 dump, but its published
  noise model is exactly what a Stage-IV MAP+MCMC should be exercised against.
* ``--data-mode benchmark`` — fit the tree's ``y`` values as-is; the resulting
  χ² is a cross-configuration goodness-of-fit of this single-config model against
  the forecast stand-ins (expect a large χ²/dof — see the module note).

Usage::

    JAX_ENABLE_X64=1 python -m hod_mod.scripts.fitting.fit_benchmark_observables \
        --which xlf n_gal wp_agn cl_kk --free Omega_m sigma8 --mode both

    # list what the tree offers
    python -m hod_mod.scripts.fitting.fit_benchmark_observables --list
"""

from __future__ import annotations

import argparse
import json
import os

# x64 is required for reliable gradients; set before importing jax.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.forecast.forward_jax import ForwardModel, OBSERVABLES
from hod_mod.forecast import params
from hod_mod.fitting import benchmark_data as bd
from hod_mod.fitting.jax_inference import (
    MultiProbeGaussianLikelihood, run_map_jax, run_nuts)


# L-BFGS-B needs a box on the stiff Stage-IV χ² surface (huge gradients when a
# probe is precise); physical priors double as optimiser bounds.
PARAM_BOUNDS = {
    "Omega_m": (0.1, 0.5), "sigma8": (0.5, 1.1), "h": (0.5, 0.9),
    "Omega_b": (0.02, 0.08), "n_s": (0.8, 1.1),
    # SHMR / AGN nuisance parameters (wide physical boxes; note several AGN
    # params are dimensionless offsets, not raw log-luminosities)
    "lg_m1h": (10.0, 14.0), "lg_m0star": (9.0, 12.0),
    "beta": (0.0, 2.0), "sigma_lnmstar": (0.1, 0.6),
    "agn_log10_lstar": (-4.0, 2.0), "agn_delta1": (0.0, 1.5),
    "agn_delta2": (1.5, 3.5), "lx_norm": (-2.0, 2.0),
}


def _bounds_for(free):
    return [PARAM_BOUNDS.get(n, (None, None)) for n in free]


def _data_root():
    env = os.environ.get("HOD_MOD_DATA_DIR")
    if env and os.path.isdir(os.path.join(env, "benchmark_observables")):
        return os.path.join(env, "benchmark_observables")
    return "/home/comparat/data/hod_mod_data/benchmark_observables"


def _results_root():
    return os.environ.get("HOD_MOD_RESULTS", "/home/comparat/data/hod_mod_results")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="benchmark_observables tree root")
    ap.add_argument("--which", nargs="+", default=["xlf", "n_gal", "wp_agn", "cl_kk"],
                    help="observables to include (must have simulated entries)")
    ap.add_argument("--free", nargs="+", default=["Omega_m", "sigma8"],
                    help="free parameters (subset of PARAM_NAMES)")
    ap.add_argument("--z-eff", type=float, default=0.2)
    ap.add_argument("--data-mode", choices=["recover", "benchmark"],
                    default="recover")
    ap.add_argument("--mode", choices=["map", "mcmc", "both"], default="both")
    ap.add_argument("--sampler", choices=["emcee", "nuts"], default="emcee",
                    help="MCMC backend; emcee is gradient-free and avoids the "
                         "NUTS trajectory compile that is intractable here")
    ap.add_argument("--n-walkers", type=int, default=32,
                    help="emcee ensemble size (>= 2*ndim+2)")
    ap.add_argument("--n-k", type=int, default=128)
    ap.add_argument("--n-m", type=int, default=128)
    ap.add_argument("--n-warmup", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--target-accept", type=float, default=0.8,
                    help="NUTS target acceptance; lower (~0.6) shortens the "
                         "Hamiltonian trajectories on a stiff Stage-IV posterior")
    ap.add_argument("--err-scale", type=float, default=1.0,
                    help="multiply the benchmark forecast errors (the tiny "
                         "Stage-IV σ give a razor-sharp posterior that NUTS "
                         "samples slowly; ~10 gives a well-sampled posterior)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-offset", type=float, default=0.02,
                    help="fractional perturbation of the MAP start from fiducial")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true",
                    help="print the simulated observables in the tree and exit")
    args = ap.parse_args()

    root = args.root or _data_root()
    if args.list:
        avail = bd.available_observables(root, "simulated")
        print(f"{len(avail)} simulated observables in {root}:")
        for o in sorted(avail):
            print(f"  {o:16s} {os.path.relpath(avail[o], root)}")
        return

    # observables present in the tree (drop and warn on the rest)
    avail = bd.available_observables(root, "simulated")
    which = [o for o in args.which if o in avail]
    dropped = [o for o in args.which if o not in avail]
    if dropped:
        print(f"[warn] no simulated entry for {dropped}; using {which}")
    if not which:
        raise SystemExit("no requested observable has a simulated entry; "
                         "run with --list to see what is available")

    label = args.label or ("bench_" + "_".join(which) + f"_z{args.z_eff}")
    out_dir = args.out or os.path.join(_results_root(), "fits", "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    # --- forward model + benchmark data vector, aligned row-for-row ----------
    print(f"[setup] ForwardModel(z_eff={args.z_eff}, n_k={args.n_k}, n_m={args.n_m}); "
          f"observables={which}; free={args.free}")
    fm = ForwardModel(z_eff=args.z_eff, n_k=args.n_k, n_m=args.n_m)
    f, row_obs, row_x = fm.full_data_vector_fn(which)
    bench = bd.load_forecast_vector(root, which, row_obs, row_x, z_eff=args.z_eff)
    # The raw Stage-IV forecast σ give a razor-sharp posterior that NUTS samples
    # very slowly; --err-scale inflates them to a chosen (e.g. current-survey)
    # precision for a well-sampled illustrative posterior.
    sigma = np.asarray(bench["y_err"], dtype=float) * float(args.err_scale)
    icov = np.diag(1.0 / sigma ** 2)

    theta0 = params.fiducial_vector()
    truth = np.asarray(f(jnp.asarray(theta0)), dtype=float)
    if args.data_mode == "recover":
        rng = np.random.default_rng(args.seed)
        data = truth + sigma * rng.standard_normal(truth.shape)
    else:
        data = np.asarray(bench["data"], dtype=float)

    like = MultiProbeGaussianLikelihood.from_forward_model(
        fm, which, data, icov, args.free, theta0=theta0, prior="planck")

    n_data = data.size
    n_free = len(args.free)
    dof = max(n_data - n_free, 1)
    x_true = theta0[[params.PARAM_NAMES.index(n) for n in args.free]]
    print(f"[data] {n_data} points from {list(bench['sources'])}; "
          f"mode={args.data_mode}; dof={dof}")

    result = {"label": label, "which": which, "free": args.free,
              "z_eff": args.z_eff, "data_mode": args.data_mode,
              "n_data": n_data, "dof": dof, "n_k": args.n_k, "n_m": args.n_m,
              "err_scale": args.err_scale,
              "x_true": {n: float(v) for n, v in zip(args.free, x_true)},
              "sources": {k: os.path.relpath(v, root)
                          for k, v in bench["sources"].items()}}

    # dump the exact fitted data vector so the plotting script needs no recompute
    np.savez(os.path.join(out_dir, f"{label}_data.npz"),
             row_obs=np.asarray(row_obs), row_x=np.asarray(row_x, dtype=float),
             data=data, y_err=sigma, truth=truth,
             which=np.asarray(which), free=np.asarray(args.free),
             z_eff=args.z_eff, n_k=args.n_k, n_m=args.n_m)

    # --- MAP -----------------------------------------------------------------
    if args.mode in ("map", "both"):
        x0 = x_true * (1.0 + args.start_offset)
        map_res = run_map_jax(like, x0, bounds=_bounds_for(args.free))
        map_res["chi2_per_dof"] = map_res["chi2"] / dof
        result["map"] = {k: map_res[k] for k in
                         ("params", "chi2", "chi2_per_dof", "logprob",
                          "success", "message", "nfev")}
        print(f"[map] chi2/dof = {map_res['chi2']:.1f}/{dof} = "
              f"{map_res['chi2_per_dof']:.3f}  success={map_res['success']}")
        for n in args.free:
            print(f"      {n:10s} MAP={map_res['params'][n]:+.4f}  "
                  f"true={result['x_true'][n]:+.4f}")
        x_start_mcmc = map_res["x"]
        # persist the MAP result immediately — NUTS can be slow / interrupted
        map_path = os.path.join(out_dir, f"{label}_map.json")
        with open(map_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[write] {map_path}")
    else:
        x_start_mcmc = x_true

    # --- MCMC ----------------------------------------------------------------
    if args.mode in ("mcmc", "both"):
        if args.sampler == "nuts":
            samples, accept = _sample_nuts(like, x_start_mcmc, args)
        else:
            samples, accept = _sample_emcee(like, x_start_mcmc, args)
        mean, std = samples.mean(axis=0), samples.std(axis=0)
        result["mcmc"] = {
            "sampler": args.sampler,
            "params": {n: float(m) for n, m in zip(args.free, mean)},
            "std": {n: float(s) for n, s in zip(args.free, std)},
            "accept_rate": float(accept), "n_samples": int(samples.shape[0])}
        np.save(os.path.join(out_dir, f"{label}_chain.npy"), samples)
        print(f"[mcmc] {args.sampler}: accept_rate={accept:.2f}, "
              f"{samples.shape[0]} samples")
        for n, m, s in zip(args.free, mean, std):
            print(f"      {n:10s} = {m:+.4f} ± {s:.4f}  "
                  f"(true {result['x_true'][n]:+.4f})")
        mcmc_path = os.path.join(out_dir, f"{label}_mcmc_summary.json")
        with open(mcmc_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[write] {mcmc_path}")
    print("[done]")


def _sample_nuts(like, x0, args):
    out = run_nuts(like, x0, n_warmup=args.n_warmup, n_samples=args.n_samples,
                   seed=args.seed, target_acceptance=args.target_accept)
    return out["samples"], out["accept_rate"]


def _sample_emcee(like, x0, args):
    """Affine-invariant ensemble sampling on the jitted log-posterior.

    Gradient-free, so it avoids the NUTS trajectory ``while_loop`` that makes the
    blackjax compile intractable on this halo model.  ``vectorize=True`` evaluates
    the whole walker ensemble in one ``jax.vmap`` call per step.
    """
    import emcee
    logp = jax.jit(jax.vmap(lambda x: like.logprob(x)))

    def log_prob(theta):                       # theta: (nwalkers, ndim)
        return np.asarray(logp(jnp.asarray(theta)), dtype=float)

    ndim = len(args.free)
    nwalkers = max(2 * ndim + 2, args.n_walkers)
    rng = np.random.default_rng(args.seed)
    # start in a small ball around the MAP, scaled per-parameter
    scale = np.maximum(np.abs(np.asarray(x0)) * 1e-3, 1e-4)
    p0 = np.asarray(x0) + scale * rng.standard_normal((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)
    n_steps = args.n_warmup + args.n_samples
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=args.n_warmup, flat=True)
    accept = float(np.mean(sampler.acceptance_fraction))
    return chain, accept


if __name__ == "__main__":
    main()
