"""Calibrate HamAGNModel's {scatter_lx, log10_A_kcorr, log10_A_dc} against Comparat+2025.

`audit_agn_lx_comparat2025.py` found that `HamAGNModel`'s HOD-weighted mean
AGN luminosity prediction is below Comparat et al. 2025's (A&A 697, A173)
independently-published "deduced LX" (Table 4) for several BGS samples. This
script fits the three free parameters added to `HamAGNModel` in this same
change (`mean_agn_log10lx`'s `scatter_lx`, `log10_A_kcorr`, `log10_A_dc` —
see agn_ham.py) against those published values directly, decoupled from any
w(theta) MAP-fit staleness (HOD weighting uses `_TABLE3` defaults via
`_hod_params()`, not `<label>_map.json` overrides).

Important caveat surfaced while building this: sample S1's stellar-mass
threshold (9.56, the lowest of the 7) pulls the HOD-N_cen weighting down to
halo masses (~1e10 Msun/h) where the abundance match against the Aird+2015
XLF has no finite solution (the cumulative XLF integral diverges as the
luminosity floor -> 0; verified directly, see agn_ham.py's
`_LOG10_LX_MIN_PHYSICAL`). Most of S1's weight therefore sits at that
deliberate physical floor by construction -- the 3 calibration parameters
here cannot "fix" S1 the way they can for an unsaturated sample, since they
act multiplicatively on a floor-clamped value rather than restoring the
mass-dependence that was clamped away. `floor_fraction` is reported for
every sample so this can't be missed; samples with `floor_fraction > 0.5`
are excluded from the primary fit and reported separately.

Usage::

    python -m hod_mod.scripts.fitting.calibrate_ham_agn_lx
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from hod_mod.scripts.fitting.fit_comparat2025 import SAMPLES, _RESULTS_DIR
from hod_mod.scripts.fitting.audit_agn_lx_comparat2025 import (
    _PAPER_DEDUCED_LX_POINTSOURCE,
    build_precomputed,
    floor_fraction,
    ham_predicted_lx_fast,
    implied_lx_from_fit,
    precompute_hod_weights,
)

_BOUNDS = [(0.3, 1.3), (-0.5, 0.5), (-0.5, 1.0)]   # scatter_lx, log10_A_kcorr, log10_A_dc
_X0     = [0.8, 0.0, 0.0]
_FLOOR_FRACTION_EXCLUDE = 0.5


def _self_check(precomputed: dict) -> None:
    """Default kwargs must reproduce the unmodified mean_agn_lx output exactly."""
    agn = precomputed["agn"]
    m_arr = np.array([1e11, 1e12, 1e13])
    lx_default = agn.mean_agn_lx(m_arr, 0.2)
    lx_explicit = agn.mean_agn_lx(m_arr, 0.2, scatter_lx=None, log10_A_kcorr=0.0, log10_A_dc=0.0)
    assert np.array_equal(lx_default, lx_explicit), (
        "Default kwargs changed mean_agn_lx's behavior -- regression in the "
        "scatter_lx/log10_A_kcorr/log10_A_dc additions."
    )
    print("Self-check passed: default kwargs reproduce unmodified mean_agn_lx output.\n")


def _objective(x: np.ndarray, samples: list[str], agn, hod_weights: dict) -> float:
    scatter_lx, log10_A_kcorr, log10_A_dc = x
    resid2 = 0.0
    for label in samples:
        lx_pred = ham_predicted_lx_fast(
            hod_weights[label], agn,
            scatter_lx=scatter_lx, log10_A_kcorr=log10_A_kcorr, log10_A_dc=log10_A_dc,
        )
        lx_paper = _PAPER_DEDUCED_LX_POINTSOURCE[label]
        resid2 += (np.log10(lx_pred) - np.log10(lx_paper)) ** 2
    # Soft regularization toward literature defaults -- with only 7 (or fewer)
    # data points and 3 parameters, scatter_lx in particular is highly
    # leveraged (boost grows as exp(c*scatter_lx^2)) and can otherwise run
    # away to the box bound chasing residuals the other 2 parameters could
    # explain just as well.
    resid2 += ((scatter_lx - 0.8) / 0.3) ** 2
    resid2 += (log10_A_kcorr / 0.3) ** 2
    resid2 += (log10_A_dc / 0.4) ** 2
    return resid2


def _run_fit(samples: list[str], agn, hod_weights: dict) -> dict:
    res = minimize(
        _objective, _X0, args=(samples, agn, hod_weights),
        method="L-BFGS-B", bounds=_BOUNDS,
    )
    scatter_lx, log10_A_kcorr, log10_A_dc = res.x
    pinned = [
        name for name, val, (lo, hi) in zip(
            ["scatter_lx", "log10_A_kcorr", "log10_A_dc"], res.x, _BOUNDS,
        )
        if abs(val - lo) < 1e-3 or abs(val - hi) < 1e-3
    ]
    return dict(
        samples=samples,
        scatter_lx=float(scatter_lx),
        log10_A_kcorr=float(log10_A_kcorr),
        log10_A_dc=float(log10_A_dc),
        objective=float(res.fun),
        success=bool(res.success),
        pinned_at_bound=pinned,
    )


def main() -> None:
    print("Building HMF + HOD + HamAGNModel (one-time, ~tens of seconds) ...", flush=True)
    precomputed = build_precomputed()
    agn = precomputed["agn"]
    _self_check(precomputed)

    print("Precomputing per-sample HOD weights (one-time per sample, _TABLE3 defaults) ...", flush=True)
    hod_weights = precompute_hod_weights(precomputed, use_map_overrides=False)

    print(f"{'sample':6s} {'floor_frac':>10s} {'LX_raw':>11s} {'LX_paper':>11s} {'ratio':>7s}", flush=True)
    floor_fracs = {}
    for label in SAMPLES:
        ff = floor_fraction(label, precomputed, use_map_overrides=False)
        floor_fracs[label] = ff
        lx_raw = ham_predicted_lx_fast(hod_weights[label], agn)
        lx_paper = _PAPER_DEDUCED_LX_POINTSOURCE[label]
        print(f"{label:6s} {ff:10.3f} {lx_raw:11.3e} {lx_paper:11.3e} {lx_raw/lx_paper:7.2f}", flush=True)

    clean_samples = [l for l in SAMPLES if floor_fracs[l] <= _FLOOR_FRACTION_EXCLUDE]
    excluded = [l for l in SAMPLES if l not in clean_samples]
    if excluded:
        print(f"\nExcluding from calibration (floor_fraction > {_FLOOR_FRACTION_EXCLUDE}): {excluded}", flush=True)

    print(f"\nFitting {{scatter_lx, log10_A_kcorr, log10_A_dc}} against Table 4 "
          f"over {clean_samples} ...", flush=True)
    fit_clean = _run_fit(clean_samples, agn, hod_weights)
    fit_all = _run_fit(list(SAMPLES), agn, hod_weights)

    for tag, fit in [("excluding floor-saturated samples", fit_clean), ("all 7 samples", fit_all)]:
        print(f"\n--- Calibration ({tag}) ---", flush=True)
        print(f"  scatter_lx     = {fit['scatter_lx']:.3f} dex")
        print(f"  log10_A_kcorr  = {fit['log10_A_kcorr']:+.3f}")
        print(f"  log10_A_dc     = {fit['log10_A_dc']:+.3f}")
        if fit["pinned_at_bound"]:
            print(f"  WARNING: pinned at bound: {fit['pinned_at_bound']} "
                  f"-- these 3 parameters cannot fully explain the discrepancy.")

    print(f"\n{'sample':6s} {'floor_frac':>10s} {'LX_raw':>11s} {'LX_calib':>11s} "
          f"{'LX_paper':>11s} {'ratio_calib':>11s} {'LX_fit_implied':>14s}", flush=True)
    rows = []
    for label in SAMPLES:
        lx_raw = ham_predicted_lx_fast(hod_weights[label], agn)
        lx_calib = ham_predicted_lx_fast(
            hod_weights[label], agn,
            scatter_lx=fit_clean["scatter_lx"],
            log10_A_kcorr=fit_clean["log10_A_kcorr"],
            log10_A_dc=fit_clean["log10_A_dc"],
        )
        lx_paper = _PAPER_DEDUCED_LX_POINTSOURCE[label]
        try:
            lx_fit_implied, _ = implied_lx_from_fit(label)
        except FileNotFoundError:
            lx_fit_implied = None
        print(
            f"{label:6s} {floor_fracs[label]:10.3f} {lx_raw:11.3e} {lx_calib:11.3e} "
            f"{lx_paper:11.3e} {lx_calib/lx_paper:11.2f} "
            f"{'n/a' if lx_fit_implied is None else f'{lx_fit_implied:.3e}':>14s}"
        )
        rows.append(dict(
            sample=label,
            floor_fraction=floor_fracs[label],
            L_X_HAM_raw_erg_s=lx_raw,
            L_X_HAM_calibrated_erg_s=lx_calib,
            L_X_paper_deduced_erg_s=lx_paper,
            ratio_calibrated_to_paper=lx_calib / lx_paper,
            L_X_fit_implied_erg_s=lx_fit_implied,
        ))

    out_path = _RESULTS_DIR / "ham_agn_calibration.json"
    with open(out_path, "w") as f:
        json.dump(dict(
            fit_excluding_floor_saturated=fit_clean,
            fit_all_samples=fit_all,
            per_sample=rows,
        ), f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
