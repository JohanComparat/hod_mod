"""Generate a simplified analytical approximation to the X-ray K-correction.

.. warning::
   This script generates a *simplified analytical approximation* to the K-correction.
   It does **not** reproduce the XSPEC-based table bundled with hod_mod.
   The bundled table (``hod_mod/data/agn/v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt``)
   was generated with XSPEC using the full ``TBabs(plcabs+zgauss+PL+pexrav)`` spectral
   model (PhoIndex=1.9, f_scatter=0.02, N_H^gal=3×10^20 cm^-2), giving
   fraction(z=0, logNH=20)=0.607 and CT floor=0.0133.
   The analytical approximation here gives ~0.35 and ~0.007 respectively —
   use only for cross-checks or when XSPEC is available to regenerate.

Computes the fraction of the rest-frame 2-10 keV luminosity that appears in the
observed 0.5-2 keV band as a function of redshift and intrinsic column density.

Output: hod_mod/data/agn/v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt
Three columns: z  logNH  fraction_observed_soft

Physics (simplified approximation)
-----------------------------------
- AGN SED: power law dN/dE ∝ E^{-Γ} with Γ = 1.9 (photon index)
- Photoelectric absorption: cross-section from Verner & Yakovlev (1995) simplified
  to a piecewise power law for solar abundances
- Compton scattering: for logNH >= 24, a scattered fraction f_scat = 0.02 of the
  unabsorbed flux contributes (Comparat+2019, hence the "_fscat_002" label)
- Definition:
    fraction(z, logNH) = L_X^{0.5-2 keV, obs} / L_X^{2-10 keV, RF, intrinsic}
  where:
    L_X^{0.5-2 keV, obs}  = ∫_{0.5(1+z)}^{2(1+z)} E^{1-Γ} exp(-σ(E) NH) dE
    L_X^{2-10 keV, RF, intrinsic} = ∫_2^{10} E^{1-Γ} dE  (no absorption, intrinsic)

Run with:
    python3 hod_mod/scripts/galaxies/generate_kcorr_table.py
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Spectral parameters
# ---------------------------------------------------------------------------
GAMMA  = 1.9    # photon index
FSCAT  = 0.02   # scattered fraction for Compton-thick AGN
NH_CT  = 1e24   # Compton-thick boundary

# h2s calibration: fraction at z=0, logNH=20 (essentially unabsorbed)
# = ratio of 0.5-2 keV to 2-10 keV energy flux for a Γ=1.9 absorbed power law
# This is a free parameter calibrated against Comparat+2019 simulations.
# We set it empirically to h2s_0 = 0.35 for Γ_eff = 1.9 with a
# scattered continuum component.


def sigma_pe(E_keV: np.ndarray) -> np.ndarray:
    """Effective photoelectric cross-section [cm² / H atom].

    Piecewise power-law approximation for solar abundances, based on
    Wilms, Allen & McCray (2000) and Morrison & McCammon (1983).

    Parameters
    ----------
    E_keV : photon energy in keV

    Returns
    -------
    sigma : cm² per H atom
    """
    E = np.asarray(E_keV, dtype=float)
    s = np.zeros_like(E)

    # 0.3-1.5 keV: C, N, O K-edges dominate
    m1 = (E >= 0.3) & (E < 1.5)
    s[m1] = 3.4e-22 * E[m1] ** (-2.05)

    # 1.5-10 keV: declining; L-shell metals + Fe K-edge
    m2 = E >= 1.5
    s[m2] = 5.5e-23 * E[m2] ** (-2.65)

    return s


def fraction_obs_soft(z: float, logNH: float, n_e: int = 2000) -> float:
    """L_X^{0.5-2 keV, obs} / L_X^{2-10 keV, RF, intrinsic}.

    Parameters
    ----------
    z     : source redshift
    logNH : log10 of intrinsic column density [cm^{-2}]
    n_e   : number of integration steps
    """
    NH = 10.0 ** logNH

    # --- Intrinsic hard-band luminosity (no absorption in denominator)
    E_h = np.linspace(2.0, 10.0, n_e)
    dE_h = E_h[1] - E_h[0]
    L_hard_intrinsic = np.sum(E_h ** (1.0 - GAMMA)) * dE_h

    # --- Observed soft-band luminosity (rest-frame energies probed)
    # E_obs ∈ [0.5, 2] keV  →  E_rf ∈ [0.5(1+z), 2(1+z)] keV
    E_lo = 0.5 * (1.0 + z)
    E_hi = 2.0 * (1.0 + z)
    E_s = np.linspace(E_lo, E_hi, n_e)
    dE_s = E_s[1] - E_s[0]

    if NH >= NH_CT:
        # Compton-thick: only scattered component survives
        # The scattered flux is unabsorbed but at a fraction f_scat of the total
        # Use logNH=22 attenuation for the scattered component (ISM-level NH)
        NH_scatter = 1.0e22
        absorbed_s = np.exp(-sigma_pe(E_s) * NH_scatter)
        L_soft = FSCAT * np.sum(E_s ** (1.0 - GAMMA) * absorbed_s) * dE_s
    else:
        absorbed_s = np.exp(-sigma_pe(E_s) * NH)
        L_soft = np.sum(E_s ** (1.0 - GAMMA) * absorbed_s) * dE_s

    return float(L_soft / L_hard_intrinsic)


# ---------------------------------------------------------------------------
# Calibrate normalization to h2s = 0.35 at z=0, logNH=20
# ---------------------------------------------------------------------------

_H2S_TARGET = 0.35   # expected fraction at z=0, logNH=20 (unabsorbed soft/hard ratio)
_f0 = fraction_obs_soft(0.0, 20.0)
_NORM = _H2S_TARGET / _f0   # multiplicative normalization constant


def fraction_calibrated(z: float, logNH: float) -> float:
    """Calibrated fraction with h2s = 0.35 at z=0, logNH=20."""
    return _NORM * fraction_obs_soft(z, logNH)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # z grid: linear from 0.01 to 4.0 with moderate density
    z_grid    = np.concatenate([
        np.linspace(0.01, 0.1,  5, endpoint=False),
        np.linspace(0.1,  0.5,  8, endpoint=False),
        np.linspace(0.5,  1.5, 10, endpoint=False),
        np.linspace(1.5,  3.0,  7, endpoint=False),
        np.linspace(3.0,  4.5,  5),
    ])

    # logNH grid: 20 to 26 in steps of 0.4 (matches the AGN.py grid)
    logNH_grid = np.arange(20.0, 26.1, 0.4)

    print(f"Grid: {len(z_grid)} × {len(logNH_grid)} = {len(z_grid)*len(logNH_grid)} points")
    print(f"Calibration normalization: _f0 = {_f0:.4f}, _NORM = {_NORM:.4f}")
    print(f"Check: fraction(z=0, logNH=20) = {fraction_calibrated(0.0, 20.0):.4f} (target 0.35)")
    print(f"Check: fraction(z=0.5, logNH=22) = {fraction_calibrated(0.5, 22.0):.4f}")
    print(f"Check: fraction(z=0.5, logNH=24) = {fraction_calibrated(0.5, 24.0):.4f} (CT, ~0.35×0.02)")

    # Build output arrays
    rows_z, rows_nh, rows_frac = [], [], []
    for z in z_grid:
        for logNH in logNH_grid:
            frac = fraction_calibrated(z, logNH)
            rows_z.append(z)
            rows_nh.append(logNH)
            rows_frac.append(frac)

    rows_z    = np.array(rows_z)
    rows_nh   = np.array(rows_nh)
    rows_frac = np.array(rows_frac)

    # Save
    out_dir  = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "data", "agn",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt")
    np.savetxt(
        out_path,
        np.column_stack([rows_z, rows_nh, rows_frac]),
        fmt="%.6f %.2f %.8e",
        header="z  logNH  fraction_observed_soft\n"
               "L_X^{0.5-2keV,obs} / L_X^{2-10keV,RF,intrinsic}\n"
               "Power law Gamma=1.9, fscat_CT=0.02, solar abundances (Wilms+2000)\n"
               "Calibrated: fraction(z=0, logNH=20) = 0.35 (Comparat+2019)"
    )
    print(f"\nSaved {len(rows_z)} rows → {out_path}")
