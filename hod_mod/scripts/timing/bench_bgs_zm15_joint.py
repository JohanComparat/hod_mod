"""CPU-vs-GPU benchmark for the ZM15 joint BGS fit.

Benchmarks the compute core of
``hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint`` — i.e. one
``JointZM15.log_prob(theta)`` evaluation (per mass bin: wp via Ogata j0 Hankel,
n_gal, and ΔΣ per requested lensing survey).  The backend (CPU or GPU) is chosen
by the ``JAX_PLATFORMS`` environment variable, which JAX reads once at import —
so a fair comparison runs the *same* workload in two separate processes.

Usage
-----
Single backend (whatever ``JAX_PLATFORMS`` currently is)::

    JAX_PLATFORMS=cpu  python -m hod_mod.scripts.timing.bench_bgs_zm15_joint
    JAX_PLATFORMS=cuda python -m hod_mod.scripts.timing.bench_bgs_zm15_joint

Both backends + comparison table (re-execs itself per backend)::

    python -m hod_mod.scripts.timing.bench_bgs_zm15_joint --both

Notes
-----
- ``log_prob`` returns a Python float, so ``float(...)`` forces a host-device
  sync each call (JAX is async) — required for correct GPU timing.
- The first call includes JIT compilation; it is reported separately from the
  steady-state per-eval time.
"""

from __future__ import annotations

import argparse
import os
import sys

from hod_mod.paths import sum_stat_root


# ---------------------------------------------------------------------------
# Single-backend benchmark (heavy imports happen inside, after JAX_PLATFORMS
# is already fixed by the parent shell / --both orchestrator).
# ---------------------------------------------------------------------------

def _run_single(args) -> int:
    import time
    import numpy as np
    import jax

    from hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint import (
        build_predictor, load_bins, JointZM15, FREE_NAMES,
    )

    backend = jax.default_backend()
    try:
        dev = str(jax.devices()[0])
    except Exception:
        dev = "?"
    print(f"[backend] JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '(unset)')}  "
          f"default_backend={backend}  device={dev}", flush=True)

    # -- infra build (CAMB + HMF + data load + fitter) -----------------------
    t0 = time.perf_counter()
    bins, h = load_bins(args.data_dir, args.surveys, args.rp_min, args.rp_max,
                        args.R_min, args.R_max, args.ng_frac_err_floor, log=lambda *a: None)
    for b in bins:
        if b.get("z") is None:
            b["z"] = args.z_eff
    if args.n_bins is not None:
        bins = bins[:args.n_bins]
    predictor, theta_cosmo = build_predictor(args.hmf_backend)
    fitter = JointZM15(bins, predictor, theta_cosmo, h=h, z=args.z_eff,
                       pi_max_h=args.pi_max_mpc * h, gaussian_prior=False)
    t_build = time.perf_counter() - t0
    print(f"[build]   infra build (CAMB+HMF+{len(bins)} bins): {t_build:.2f} s "
          f"(surveys={args.surveys or 'none'})", flush=True)

    x0 = np.asarray(fitter.x0, dtype=float)
    lo = np.array([b[0] for b in fitter.bounds])
    hi = np.array([b[1] for b in fitter.bounds])

    # -- compile (first call) ------------------------------------------------
    t0 = time.perf_counter()
    val0 = float(fitter.log_prob(x0))
    t_compile = time.perf_counter() - t0
    print(f"[compile] first log_prob (incl. JIT): {t_compile:.2f} s  "
          f"(log_prob(x0)={val0:.4f})", flush=True)

    # -- steady-state per-eval ----------------------------------------------
    rng = np.random.default_rng(0)
    thetas = x0[None, :] * (1.0 + args.eps * rng.standard_normal((args.n_eval, x0.size)))
    thetas = np.clip(thetas, lo + 1e-9, hi - 1e-9)   # keep inside bounds (avoid -inf fast path)

    dts = np.empty(args.n_eval)
    finite = 0
    for i in range(args.n_eval):
        t0 = time.perf_counter()
        v = float(fitter.log_prob(thetas[i]))
        dts[i] = time.perf_counter() - t0
        finite += int(np.isfinite(v))
    per_ms = dts * 1e3
    print(f"[eval]    n={args.n_eval}  finite={finite}/{args.n_eval}  "
          f"per-eval: mean={per_ms.mean():.1f} ms  std={per_ms.std():.1f}  "
          f"min={per_ms.min():.1f}  max={per_ms.max():.1f}", flush=True)

    # parseable lines for the --both orchestrator
    print(f"RESULT BACKEND={backend} BUILD_S={t_build:.4f} "
          f"COMPILE_S={t_compile:.4f} PEREVAL_MS={per_ms.mean():.4f}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# --both orchestrator: run this module once per backend, parse, compare.
# ---------------------------------------------------------------------------

def _child_argv(args) -> list[str]:
    argv = [
        "--data-dir", args.data_dir,
        "--n-eval", str(args.n_eval),
        "--eps", str(args.eps),
        "--hmf-backend", args.hmf_backend,
        "--rp-min", str(args.rp_min), "--rp-max", str(args.rp_max),
        "--R-min", str(args.R_min), "--R-max", str(args.R_max),
        "--z-eff", str(args.z_eff), "--pi-max-mpc", str(args.pi_max_mpc),
        "--ng-frac-err-floor", str(args.ng_frac_err_floor),
    ]
    if args.n_bins is not None:
        argv += ["--n-bins", str(args.n_bins)]
    argv += ["--surveys", *args.surveys]   # nargs="*"; empty => wp+n_gal only
    return argv


def _run_both(args) -> int:
    import re
    import subprocess

    rows = {}
    for plat in ("cpu", "cuda"):
        print(f"\n{'='*60}\n  Running backend: {plat}\n{'='*60}", flush=True)
        env = {**os.environ, "JAX_PLATFORMS": plat}
        cmd = [sys.executable, "-m", "hod_mod.scripts.timing.bench_bgs_zm15_joint",
               *_child_argv(args)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        out = proc.stdout
        # echo the child's human-readable lines
        for line in out.splitlines():
            if line.startswith("[") or line.startswith("RESULT"):
                print("   " + line, flush=True)
        m = re.search(r"RESULT BACKEND=(\S+) BUILD_S=(\S+) COMPILE_S=(\S+) PEREVAL_MS=(\S+)", out)
        if not m:
            print(f"   !! backend {plat} produced no RESULT line; stderr tail:", flush=True)
            print("   " + "\n   ".join(proc.stderr.splitlines()[-8:]), flush=True)
            continue
        rows[plat] = dict(backend=m.group(1), build_s=float(m.group(2)),
                          compile_s=float(m.group(3)), pereval_ms=float(m.group(4)))

    # -- comparison table ----------------------------------------------------
    print(f"\n{'='*60}\n  CPU vs GPU — JointZM15.log_prob  (surveys={args.surveys or 'none'}, "
          f"bins={args.n_bins or 'all'})\n{'='*60}")
    hdr = f"{'metric':<22}{'CPU':>14}{'GPU':>14}"
    print(hdr); print("-" * len(hdr))
    if "cpu" in rows and "cuda" in rows:
        c, g = rows["cpu"], rows["cuda"]
        print(f"{'build (s)':<22}{c['build_s']:>14.2f}{g['build_s']:>14.2f}")
        print(f"{'compile 1st call (s)':<22}{c['compile_s']:>14.2f}{g['compile_s']:>14.2f}")
        print(f"{'per-eval (ms)':<22}{c['pereval_ms']:>14.1f}{g['pereval_ms']:>14.1f}")
        spd = c["pereval_ms"] / g["pereval_ms"] if g["pereval_ms"] > 0 else float("nan")
        faster = "GPU" if spd > 1 else "CPU"
        print("-" * len(hdr))
        print(f"per-eval speedup (CPU/GPU) = {spd:.2f}×  →  {faster} is faster per eval")
        # extrapolation to representative full-fit eval counts
        print("\nExtrapolated steady-state wall time (compile excluded):")
        print(f"{'eval count':<22}{'CPU':>14}{'GPU':>14}")
        for n in (1_000, 10_000, 32 * 2_000):
            tc = c["pereval_ms"] * n / 1e3
            tg = g["pereval_ms"] * n / 1e3
            tag = f"{n:,}"
            print(f"{tag:<22}{tc/60:>12.1f} m{tg/60:>12.1f} m")
        print("\n(MAP=Powell, data-dependent count; MCMC default = 32 walkers × "
              "(500+2000) steps = 80,000 evals.)")
    else:
        print("Incomplete results — see per-backend output above.")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--both", action="store_true",
                   help="Run both CPU and GPU (subprocess per backend) and compare.")
    p.add_argument("--data-dir",
                   default=str(sum_stat_root() / "BGS_Mstar10_massbins"))
    p.add_argument("--surveys", nargs="*", default=[],
                   help="Lensing surveys (HSC DES KIDS). Default none = wp+n_gal only.")
    p.add_argument("--n-eval", type=int, default=30, help="Steady-state eval count.")
    p.add_argument("--n-bins", type=int, default=None, help="Use only the first N mass bins.")
    p.add_argument("--eps", type=float, default=1e-3, help="Relative theta jitter.")
    p.add_argument("--hmf-backend", default="tinker08")
    p.add_argument("--rp-min", type=float, default=0.1)
    p.add_argument("--rp-max", type=float, default=30.0)
    p.add_argument("--R-min", type=float, default=0.1)
    p.add_argument("--R-max", type=float, default=30.0)
    p.add_argument("--z-eff", type=float, default=0.13)
    p.add_argument("--pi-max-mpc", type=float, default=100.0)
    p.add_argument("--ng-frac-err-floor", type=float, default=0.05)
    args = p.parse_args()

    return _run_both(args) if args.both else _run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
