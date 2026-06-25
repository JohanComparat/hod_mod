"""Audit: is the AGN amplitude `log10_A_AGN` consistent with the HAM/SHAM L_X prediction?

`fit_comparat2025.py` fits `log10_A_AGN` as a free linear amplitude on a
normalized PSF template against the dimensionless w(theta) data — it has no
built-in connection to `HamAGNModel.mean_agn_lx()` (the HAM/SHAM-predicted
mean AGN soft-band luminosity, in erg/s). This script builds that connection
explicitly, using the background-subtraction technique from Comparat+2025
(A&A 697, A173, arXiv:2503.19796):

    S_X^G(R) = (1 + w(R)) * S_R_X                              (their Eq. 3)

Applied to just the AGN/point-source component of the model
(`A_AGN * PSF(theta)`), this converts the fitted amplitude into a physical
excess surface-brightness profile, which is integrated over area to give the
mean AGN luminosity implied by the fit. That number is compared to:

    (a) HamAGNModel's HOD-(central-galaxy-)weighted predicted mean L_X, and
    (b) Comparat+2025 Table 4's independently-deduced point-source L_X.

Caveat: Table 2's S_R_X values calibrate the Davis-Peebles stacking estimator
(their Eq. 3), while `hod_mod`'s `wtheta` data is the Landy-Szalay estimator
(their Eq. 2). The paper shows the two agree to ~5-10% over 20-500 kpc
(proper) and diverge outside that range (their Fig. 4), so this is the right
available tool but carries that systematic.

Usage::

    python -m hod_mod.scripts.fitting.audit_agn_lx_comparat2025
    python -m hod_mod.scripts.fitting.audit_agn_lx_comparat2025 --sample S1

References
----------
Comparat et al. 2025, A&A 697, A173 (arXiv:2503.19796) — Table 2 (S_R_X),
Table 4 (deduced point-source L_X), Eq. 3 (background-subtraction estimator).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.integrate import trapezoid

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.galaxies.hod import ZuMandelbaum15HODModel
from hod_mod.galaxies.agn_ham import HamAGNModel, _LOG10_LX_MIN_PHYSICAL

from hod_mod.scripts.fitting.fit_comparat2025 import (
    SAMPLES,
    _hod_params,
    _psf_template,
    _THETA_COSMO,
    load_data,
    _RESULTS_DIR,
)

# Comparat+2025 Table 2 — background surface brightness, tabulated as
# S_R_X * (1+zbar)^-2 [erg kpc^-2 s^-1], keyed by log10(M*_0 / Msun).
_BACKGROUND_SX_TABLE: dict[float, float] = {
    10.00: 6.859e36,
    10.25: 6.799e36,
    10.50: 6.772e36,
    10.75: 6.712e36,
    11.00: 6.693e36,
    11.25: 6.706e36,
    11.50: 6.727e36,
}

# Comparat+2025 Table 4 — deduced point-source (AGN+XRB) luminosity [erg/s].
_PAPER_DEDUCED_LX_POINTSOURCE: dict[str, float] = {
    "S1": 3.59e40, "S2": 4.54e40, "S3": 6.20e40, "S4": 8.63e40,
    "S5": 7.60e40, "S6": 7.43e40, "S7": 5.53e40,
}


def _background_sx(label: str) -> float:
    """S_R_X at the sample's mean redshift [erg kpc^-2 s^-1]."""
    log10ms = SAMPLES[label]["log10ms_min"]
    zbar = SAMPLES[label]["zmean"]
    return _BACKGROUND_SX_TABLE[log10ms] * (1.0 + zbar) ** 2


def _kpc_per_arcsec(label: str) -> float:
    """Physical-separation-per-arcsecond factor at the sample's z, from the data file."""
    d = load_data(label)
    return float(d["R_kpc"][0] / d["theta_arcsec"][0])


def _load_best_fit(label: str) -> dict:
    """Flatten a ``<label>_map.json`` file's ``param_names``/``params`` lists into a dict."""
    with open(_RESULTS_DIR / f"{label}_map.json") as f:
        raw = json.load(f)
    return dict(zip(raw["param_names"], raw["params"]))


def implied_lx_from_fit(label: str) -> tuple[float, float]:
    """Mean AGN luminosity [erg/s] implied by the fitted log10_A_AGN.

    Returns (L_X_implied, log10_A_AGN).
    """
    best = _load_best_fit(label)
    log10_A_AGN = float(best["log10_A_AGN"])
    A_AGN = 10.0 ** log10_A_AGN

    theta_arcsec = np.logspace(-2, np.log10(3600.0), 4000)
    psf = _psf_template(theta_arcsec)
    R_kpc = theta_arcsec * _kpc_per_arcsec(label)

    delta_sx = A_AGN * psf * _background_sx(label)        # erg kpc^-2 s^-1
    L_X = trapezoid(delta_sx * 2.0 * np.pi * R_kpc, R_kpc)  # erg/s
    return float(L_X), log10_A_AGN


def build_precomputed() -> dict:
    """Build the (expensive, ~12s) HMF + HOD + HamAGNModel components once.

    Reuse the returned dict across samples and, when calibrating free
    parameters, across every optimizer iteration — none of
    ``scatter_lx``/``log10_A_kcorr``/``log10_A_dc`` touch this precompute.
    """
    pk_lin = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    agn = HamAGNModel(pk_lin=pk_lin)
    return dict(hmf=hmf, hod=hod, agn=agn)


def _hod_weighted_inputs(
    label: str, precomputed: dict, use_map_overrides: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """(m_arr [Msun/h], HOD-N_cen-weight, zbar) for the halo-mass integral.

    ``use_map_overrides=True`` (default, used by the audit functions below)
    prefers each sample's fitted HOD shape from ``<label>_map.json`` when
    available, for consistency with that specific w(theta) fit. Set False
    (used by ``calibrate_ham_agn_lx.py``) to always use ``_TABLE3`` defaults
    instead, decoupling the HOD weighting from any MAP-fit staleness.
    """
    hod_overrides = {}
    map_path = _RESULTS_DIR / f"{label}_map.json"
    if use_map_overrides and map_path.exists():
        best = _load_best_fit(label)
        for k in ("log10m_star_thresh", "sigma_lnmstar", "lg_m1h", "alpha_sat"):
            if k in best:
                hod_overrides[k] = best[k]
    hod_params = _hod_params(label, **hod_overrides)

    hmf, hod = precomputed["hmf"], precomputed["hod"]
    zbar = SAMPLES[label]["zmean"]
    log10m_arr = np.linspace(10.0, 16.0, 240)
    m_arr = 10.0 ** log10m_arr

    nc, _ns = hod.nc_ns(log10m_arr, hod_params)
    nc = np.asarray(nc, dtype=float)
    dndm = np.asarray(hmf.dndm(m_arr, zbar, _THETA_COSMO), dtype=float)
    return m_arr, dndm * nc, zbar


def precompute_hod_weights(
    precomputed: dict, samples: list[str] | None = None, use_map_overrides: bool = True,
) -> dict:
    """Precompute ``{label: (m_arr, weight, zbar)}`` once.

    This is the per-sample-but-not-per-parameter-trial cost of
    :func:`ham_predicted_lx` (it doesn't depend on
    ``scatter_lx``/``log10_A_kcorr``/``log10_A_dc``). Call once and reuse via
    :func:`ham_predicted_lx_fast` across every optimizer iteration when
    calibrating — recomputing it per-iteration (as the simple
    :func:`ham_predicted_lx` does) is correct but far too slow inside a fit.
    See :func:`_hod_weighted_inputs` for ``use_map_overrides``.
    """
    samples = list(SAMPLES) if samples is None else samples
    return {
        label: _hod_weighted_inputs(label, precomputed, use_map_overrides)
        for label in samples
    }


def ham_predicted_lx_fast(
    hod_weight: tuple[np.ndarray, np.ndarray, float],
    agn: HamAGNModel,
    *,
    scatter_lx: float | None = None,
    log10_A_kcorr: float = 0.0,
    log10_A_dc: float = 0.0,
) -> float:
    """Like :func:`ham_predicted_lx`, given a precomputed ``(m_arr, weight, zbar)``."""
    m_arr, weight, zbar = hod_weight
    lx_per_halo = agn.mean_agn_lx(
        m_arr, zbar,
        scatter_lx=scatter_lx, log10_A_kcorr=log10_A_kcorr, log10_A_dc=log10_A_dc,
    )
    return float(trapezoid(weight * lx_per_halo, m_arr) / trapezoid(weight, m_arr))


def ham_predicted_lx(
    label: str,
    precomputed: dict | None = None,
    *,
    scatter_lx: float | None = None,
    log10_A_kcorr: float = 0.0,
    log10_A_dc: float = 0.0,
) -> float:
    """HOD-(central-galaxy-)weighted mean AGN soft-band L_X [erg/s] from HamAGNModel.

    One-shot convenience wrapper. For repeated evaluation at many
    ``{scatter_lx, log10_A_kcorr, log10_A_dc}`` trial values (e.g. inside an
    optimizer), call :func:`precompute_hod_weights` once and use
    :func:`ham_predicted_lx_fast` instead — this function redoes the
    (expensive) HOD/HMF weighting on every call.
    """
    if precomputed is None:
        precomputed = build_precomputed()
    hod_weight = _hod_weighted_inputs(label, precomputed)
    return ham_predicted_lx_fast(
        hod_weight, precomputed["agn"],
        scatter_lx=scatter_lx, log10_A_kcorr=log10_A_kcorr, log10_A_dc=log10_A_dc,
    )


def floor_fraction(
    label: str, precomputed: dict | None = None, use_map_overrides: bool = True,
) -> float:
    """Fraction of the HOD-N_cen-weighted halo population sitting at HamAGNModel's
    physical luminosity floor (``_LOG10_LX_MIN_PHYSICAL``) -- diagnoses whether a
    sample's mean-LX prediction is dominated by the floor clamp (see agn_ham.py)
    rather than a genuine abundance-matched value. See :func:`_hod_weighted_inputs`
    for ``use_map_overrides``.
    """
    if precomputed is None:
        precomputed = build_precomputed()
    m_arr, weight, zbar = _hod_weighted_inputs(label, precomputed, use_map_overrides)
    lx_hard = precomputed["agn"].ham_log10lx_hard(np.log10(m_arr), zbar)
    at_floor = lx_hard <= _LOG10_LX_MIN_PHYSICAL + 1e-3
    return float(trapezoid(weight * at_floor, m_arr) / trapezoid(weight, m_arr))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", nargs="+", default=list(SAMPLES.keys()))
    args = parser.parse_args()

    precomputed = build_precomputed()
    rows = []
    for label in args.sample:
        map_path = _RESULTS_DIR / f"{label}_map.json"
        if not map_path.exists():
            print(f"[{label}] no MAP result at {map_path}, skipping")
            continue

        lx_fit, log10_a_agn = implied_lx_from_fit(label)
        lx_ham = ham_predicted_lx(label, precomputed)
        ff = floor_fraction(label, precomputed)
        lx_paper = _PAPER_DEDUCED_LX_POINTSOURCE.get(label)

        row = dict(
            sample=label,
            log10_A_AGN=log10_a_agn,
            L_X_fit_implied_erg_s=lx_fit,
            L_X_HAM_predicted_erg_s=lx_ham,
            floor_fraction=ff,
            ratio_fit_to_HAM=lx_fit / lx_ham,
            L_X_paper_deduced_erg_s=lx_paper,
            ratio_fit_to_paper=(lx_fit / lx_paper) if lx_paper else None,
        )
        rows.append(row)
        print(
            f"[{label}] log10_A_AGN={log10_a_agn:+.3f}  "
            f"L_X_fit={lx_fit:.3e} erg/s  L_X_HAM={lx_ham:.3e} erg/s  "
            f"floor_fraction={ff:.3f}  "
            f"ratio(fit/HAM)={row['ratio_fit_to_HAM']:.2f}  "
            f"L_X_paper={lx_paper:.3e} erg/s  ratio(fit/paper)={row['ratio_fit_to_paper']:.2f}"
        )

    out_path = _RESULTS_DIR / "agn_lx_audit.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
