#!/usr/bin/env python
"""
Run all HOD literature benchmarks and print a summary table.

Usage
-----
    python hod_mod/scripts/benchmarks/run_all_benchmarks.py [--mcmc] [--plot]

Iterates over all entries in BENCHMARK_REGISTRY, skipping those whose data
files are not yet populated (status='NEEDS_DATA').
"""

import argparse
import sys

from run_benchmark import BENCHMARK_REGISTRY, run_benchmark


def run_all(mcmc: bool = False, plot: bool = False) -> list[dict]:
    results = []
    for model_key in BENCHMARK_REGISTRY:
        try:
            r = run_benchmark(model_key, mcmc=mcmc, plot=plot)
        except Exception as exc:
            print(f"\nERROR running {model_key}: {exc}")
            r = {"model": model_key, "status": "error", "chi2_ndof": float("nan")}
        results.append(r)
    return results


def _print_summary(results: list[dict]) -> None:
    col_w = 28
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    header = f"{'Model':{col_w}s}  {'chi2/dof':>9s}  {'status':>8s}  {'max|Δ/σ|':>10s}"
    print(header)
    print("-" * 70)

    n_pass = n_fail = n_skip = n_err = 0
    for r in results:
        name = BENCHMARK_REGISTRY[r["model"]]["label"].split("(")[0].strip()
        status = r.get("status", "error")
        chi2 = r.get("chi2_ndof", float("nan"))
        devs = r.get("param_deviations_sigma", {})
        max_dev = max((abs(v) for v in devs.values() if v == v), default=float("nan"))

        chi2_str = f"{chi2:.2f}" if chi2 == chi2 else "---"
        dev_str = f"{max_dev:.1f}σ" if max_dev == max_dev else "N/A"

        if status == "pass":
            mark = "✓"
            n_pass += 1
        elif status == "fail":
            mark = "✗"
            n_fail += 1
        elif status == "skipped":
            mark = "SKIP"
            n_skip += 1
        else:
            mark = "ERR"
            n_err += 1

        print(f"{name:{col_w}s}  {chi2_str:>9s}  {mark:>8s}  {dev_str:>10s}")

    print("-" * 70)
    print(f"Total: {n_pass} passed, {n_fail} failed, {n_skip} skipped, {n_err} errors")
    print(f"{'='*70}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Run all HOD benchmarks.")
    p.add_argument("--mcmc", action="store_true", help="Run MCMC after each MAP fit")
    p.add_argument("--plot", action="store_true", help="Save comparison figures")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_all(mcmc=args.mcmc, plot=args.plot)
    _print_summary(results)
    n_fail = sum(1 for r in results if r.get("status") == "fail")
    sys.exit(1 if n_fail > 0 else 0)
