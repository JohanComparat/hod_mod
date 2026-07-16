"""Export a frozen Battaglia+2012 tSZ reference for the hod_mod GODMAX cross-check.

This script produces the reference file consumed by
``hod_mod/scripts/validate_godmax.py`` and ``tests/test_godmax_comparison.py``.
It is meant to be run **once**, outside the hod_mod test environment, and its
output committed (or hosted) so the cross-check runs deterministically with no
GODMAX dependency in CI — the same pattern hod_mod uses for its CAMB pins and
benchmark-JSON references.

Two sources, selected with ``--source``:

* ``godmax`` (default) — imports the real GODMAX code
  (https://github.com/shivampcosmo/GODMAX, Pandey+2024 arXiv:2401.18072).
  The Battaglia+2012 electron pressure comes from GODMAX's ``Battaglia_12_16``
  (``src/get_B12_profile.py``); the angular spectra C_ℓ^{yy}/C_ℓ^{κy} come from
  GODMAX's ``get_power_BCMP`` (``src/get_power_spectra_jit.py``).  Point your
  ``PYTHONPATH`` at the GODMAX ``src/`` directory before running, and align the
  ``_POWER_CONFIG`` block with the notebook you reproduce
  (``notebooks/ACTxDES/run_test_sampling_datavector.ipynb``).

* ``independent`` — a self-contained NumPy re-implementation of the *same* B12
  analytic profile plus an independent Gauss-Legendre spherical Fourier
  transform (no GODMAX, no hod_mod internals).  This lets the profile + ỹ(k|M)
  machinery be cross-checked against a second, independently-coded transform
  without the GODMAX stack.  It does **not** produce C_ℓ (which needs a halo
  model); use ``godmax`` for the full stack.

The **shared** convention (both sources, matching hod_mod's B12 profile):
electron pressure P_e = f_e · P_th with f_e = (2+2 X_H)/(3+5 X_H); ỹ(k|M) in
(Mpc/h)² via the σ_T/(m_e c²) prefactor; radii in comoving Mpc/h.

Usage
-----
    # in a checkout with GODMAX src on PYTHONPATH:
    python scripts/godmax/export_godmax_b12_reference.py --source godmax \
        --out hod_mod/data/godmax/godmax_b12_reference.npz

    # anywhere (NumPy only), profile + ỹ only:
    python scripts/godmax/export_godmax_b12_reference.py --source independent \
        --out hod_mod/data/godmax/independent_b12_reference.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Matched configuration — hod_mod's validate/test must use the SAME values.
# ---------------------------------------------------------------------------
CONFIG = {
    # cosmology (Planck-18-like; hod_mod's LinearPowerSpectrum.default_cosmology)
    "h":        0.6774,
    "Omega_m":  0.3089,
    "Omega_b":  0.0486,
    "sigma8":   0.8159,
    "n_s":      0.9667,
    # B12 electron-fraction / truncation (must match PressureProfileBattaglia12)
    "x_h":               0.76,
    "r_max_over_r200c":  4.0,
    # halo-model choices hod_mod should mirror for the C_ℓ comparison
    "mdef":       "200c",
    "hmf_model":  "tinker08",
    "bias_model": "tinker10",
    "conc_model": "diemer19",
    # profile grid (physical M200c in Msun, dimensionless x = r/R200c)
    "prof_z":            [0.0, 0.25, 0.55, 1.0],
    "prof_log10M_Msun":  [13.0, 13.5, 14.0, 14.5, 15.0],
    "prof_x":            np.logspace(-2.0, np.log10(4.0), 40).tolist(),
    # ỹ(k|M) grid (k in h/Mpc, M200c in Msun/h to match the hod_mod cache)
    "uk_z":              [0.25, 0.55],
    "uk_log10k":         np.linspace(-2.0, 1.0, 40).tolist(),
    "uk_log10M_Msunh":   [13.0, 13.5, 14.0, 14.5, 15.0],
    # angular-spectrum grid + shear source n(z)
    "ell":               np.unique(np.geomspace(80.0, 12000.0, 24).round()).tolist(),
    "cl_z":              np.linspace(0.02, 3.5, 30).tolist(),
    "nz_source_z":       np.linspace(0.02, 3.5, 30).tolist(),
    "nz_source":         np.exp(-0.5 * ((np.linspace(0.02, 3.5, 30) - 0.62) / 0.28) ** 2).tolist(),
}

# GODMAX get_power_BCMP config — ALIGN with the ACTxDES notebook you reproduce.
_POWER_CONFIG_NOTE = (
    "Fill sim_params_dict/halo_params_dict/analysis_dict/other_params_dict from "
    "notebooks/ACTxDES/run_test_sampling_datavector.ipynb, pointing the pressure "
    "profile at get_B12_profile (Battaglia_12_16) rather than the BCM profile so "
    "the C_ℓ use the SAME shared B12 pressure as the profile/ỹ blocks."
)

# Physical constants (independent copy — do not import hod_mod)
_SIGMA_T_OVER_ME_C2 = 6.6524e-25 / 511.0      # cm²/keV
_MPC_CM             = 3.0857e24               # cm per Mpc
_RHO_CRIT0          = 2.775e11                # (Msun/h)/(Mpc/h)³
_G_MSUN2_MPC4_KEV   = 1.8168e-30              # keV/cm³  (G·M_sun²/Mpc⁴)

# Battaglia+2012 Table 1 — AGN feedback, Δ=200
_B12 = {
    "P0":   {"A0": 18.1,  "am":  0.154,   "az": -0.758},
    "xc":   {"A0": 0.497, "am": -0.00865, "az":  0.731},
    "beta": {"A0": 4.35,  "am":  0.0393,  "az":  0.415},
    "gamma": -0.3, "alpha": 1.0,
}


def _r200c_comoving_mpch(m200c_msunh, z, omega_m):
    """Comoving R200c [Mpc/h] from M200c [Msun/h] using comoving ρ_cr(z)."""
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    rho_cr_com = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3
    return (3.0 * m200c_msunh / (4.0 * np.pi * 200.0 * rho_cr_com)) ** (1.0 / 3.0)


def _b12_pe_electron(x, m200c_msunh, z, cfg):
    """Independent NumPy B12 electron pressure P_e(x=r/R200c|M200c,z) [keV/cm³]."""
    h, om, ob, xh = cfg["h"], cfg["Omega_m"], cfg["Omega_b"], cfg["x_h"]
    f_e = (2.0 + 2.0 * xh) / (3.0 + 5.0 * xh)
    f_b = ob / om
    m_phys = m200c_msunh / h
    ez2 = om * (1.0 + z) ** 3 + (1.0 - om)
    rho_cr_phys = _RHO_CRIT0 * h ** 2 * ez2                      # Msun/Mpc³ physical
    r200c_com = _r200c_comoving_mpch(m200c_msunh, z, om)
    r200c_phys = r200c_com / h / (1.0 + z)                       # Mpc physical
    p200 = _G_MSUN2_MPC4_KEV * m_phys * 200.0 * rho_cr_phys * f_b / (2.0 * r200c_phys)

    def _scale(p):
        return p["A0"] * (m_phys / 1.0e14) ** p["am"] * (1.0 + z) ** p["az"]

    p0, xc, beta = _scale(_B12["P0"]), _scale(_B12["xc"]), _scale(_B12["beta"])
    xr = np.asarray(x) / xc
    shape = xr ** _B12["gamma"] * (1.0 + xr ** _B12["alpha"]) ** (-beta)
    return f_e * p200 * p0 * shape


def _ytilde_gl(k_hMpc, m200c_msunh, z, cfg, n_gl=256):
    """Independent GL spherical FT ỹ(k|M) [(Mpc/h)²] for one (M,z).

    Deliberately a plain-NumPy leggauss loop — independent of hod_mod's
    einsum/sinc implementation — so agreement validates the transform, not a
    shared code path.
    """
    h, om = cfg["h"], cfg["Omega_m"]
    r200c = _r200c_comoving_mpch(m200c_msunh, z, om)             # comoving Mpc/h
    r_max = cfg["r_max_over_r200c"] * r200c
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    r = 0.5 * r_max * (xg + 1.0)                                 # (n_gl,) Mpc/h
    pe = _b12_pe_electron(r / r200c, m200c_msunh, z, cfg)        # keV/cm³
    a = 0.5 * r_max * wg * pe * r ** 2
    kr = np.outer(k_hMpc, r)                                     # (Nk, n_gl)
    j0 = np.sinc(kr / np.pi)
    raw = 4.0 * np.pi * (j0 * a[None, :]).sum(axis=1)           # (Nk,)
    return _SIGMA_T_OVER_ME_C2 * (_MPC_CM / h) * raw            # (Mpc/h)²


# ---------------------------------------------------------------------------
# Profile + ỹ blocks (shared by both sources; C_ℓ only for --source godmax)
# ---------------------------------------------------------------------------

def _build_profile_block(cfg, pe_fn):
    z = np.asarray(cfg["prof_z"], float)
    m = 10.0 ** np.asarray(cfg["prof_log10M_Msun"], float)      # physical Msun
    x = np.asarray(cfg["prof_x"], float)
    pe = np.empty((z.size, m.size, x.size))
    for i, zi in enumerate(z):
        for j, mj in enumerate(m):
            pe[i, j] = pe_fn(x, mj * cfg["h"], zi, cfg)         # pe_fn wants Msun/h
    return {"prof_z": z, "prof_m200c_Msun": m, "prof_x": x, "prof_Pe": pe}


def _build_uk_block(cfg, uk_fn):
    z = np.asarray(cfg["uk_z"], float)
    k = 10.0 ** np.asarray(cfg["uk_log10k"], float)             # h/Mpc
    m = 10.0 ** np.asarray(cfg["uk_log10M_Msunh"], float)       # Msun/h
    yt = np.empty((z.size, k.size, m.size))
    for i, zi in enumerate(z):
        for j, mj in enumerate(m):
            yt[i, :, j] = uk_fn(k, mj, zi, cfg)
    return {"uk_z": z, "uk_k": k, "uk_m200c_h": m, "uk_ytilde": yt}


def _build_cl_block_godmax(cfg):
    """C_ℓ^{yy} and C_ℓ^{κy} from GODMAX's get_power_BCMP.

    This is the section to complete against your ACTxDES notebook — see
    ``_POWER_CONFIG_NOTE``.  It must return arrays aligned with ``cfg['ell']``.
    """
    from get_power_spectra_jit import get_power_BCMP  # noqa: F401  (GODMAX src)

    raise NotImplementedError(
        "Wire get_power_BCMP with the ACTxDES config dicts, projecting the "
        "shared Battaglia_12_16 pressure. " + _POWER_CONFIG_NOTE
    )


def _pe_godmax(x, m200c_msunh, z, cfg):
    """GODMAX Battaglia_12_16 electron pressure at x=r/R200c [keV/cm³]."""
    from get_B12_profile import Battaglia_12_16  # noqa: F401 (GODMAX src)

    # GODMAX takes physical M200c [Msun] and z; get_Pth(r) returns thermal
    # pressure in keV/cm³ — apply the shared electron factor f_e here so the
    # export matches hod_mod's PressureProfileBattaglia12 convention.
    f_e = (2.0 + 2.0 * cfg["x_h"]) / (3.0 + 5.0 * cfg["x_h"])
    m_phys = m200c_msunh / cfg["h"]
    r200c_phys = _r200c_comoving_mpch(m200c_msunh, z, cfg["Omega_m"]) / cfg["h"] / (1.0 + z)
    bat = Battaglia_12_16(M=m_phys, z=z)            # default AGN-feedback params
    r_phys = np.asarray(x) * r200c_phys            # Mpc physical
    p_th = np.asarray([float(bat.get_Pth(ri)) for ri in np.atleast_1d(r_phys)])
    return f_e * p_th.reshape(np.shape(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["godmax", "independent"], default="godmax")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with-cl", action="store_true",
                    help="godmax only: also build C_ℓ (needs the notebook config).")
    args = ap.parse_args()

    cfg = CONFIG
    if args.source == "godmax":
        prof = _build_profile_block(cfg, _pe_godmax)
        uk   = _build_uk_block(cfg, lambda k, m, z, c: _ytilde_gl(k, m, z, c))
    else:
        prof = _build_profile_block(cfg, _b12_pe_electron)
        uk   = _build_uk_block(cfg, _ytilde_gl)

    blocks = {**prof, **uk,
              "nz_source_z": np.asarray(cfg["nz_source_z"], float),
              "nz_source":   np.asarray(cfg["nz_source"], float),
              "ell":         np.asarray(cfg["ell"], float),
              "cl_z":        np.asarray(cfg["cl_z"], float),
              "source":      np.array(args.source)}

    if args.source == "godmax" and args.with_cl:
        blocks.update(_build_cl_block_godmax(cfg))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **blocks)
    manifest = args.out.with_name(args.out.stem + "_manifest.json")
    with open(manifest, "w") as fh:
        json.dump({"source": args.source,
                   "config": {k: v for k, v in cfg.items()},
                   "has_cl": bool(args.source == "godmax" and args.with_cl),
                   "note": _POWER_CONFIG_NOTE}, fh, indent=2, default=list)
    print(f"wrote {args.out}  (source={args.source}, keys={sorted(blocks)})")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
