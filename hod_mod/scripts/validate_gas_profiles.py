"""Validation plots for all gas profiles and scaling relations.

Produces four multi-panel figures comparing profile shapes across models,
mass-scaling power laws, parameter sensitivity, and predicted Lx/kT/Y_SZ
scaling relations against Lovisari+2020 (arXiv:2004.03401).

Figures produced
----------------
1. ``gas_01_radial_profiles.pdf``
   P_e, n_e, Z, T radial profiles at M=1e14 Msun/h, z=0.1.
   Compares A10 pressure to DPM models 1,2,3.
2. ``gas_02_mass_scaling.pdf``
   Profile amplitude at r=0.3 R200 vs M200 for each model.
   Also shows the APEC cooling function Λ(T,Z) for varying metallicity.
3. ``gas_03_parameter_sensitivity.pdf``
   How key DPM parameters (ne_03, beta, alpha_out) shift the density and
   pressure profiles.
4. ``gas_04_scaling_relations.pdf``
   Predicted Lx, kT_ew, Y×D_A², and Lx-kT vs M500c at z=0.1.
   Overlays Lovisari+2020 power-law fits (soft band 0.5–2 keV).

Usage
-----
    cd /path/to/hod_mod
    python -m hod_mod.scripts.validate_gas_profiles

Note on pressure units
----------------------
The DPM P_03 values (409, 115, 71) are stored in the units published by
Oppenheimer+2025.  The temperature panel T = P/n_e and the Y_SZ comparison
against A10 serve as a unit sanity check.  If Y_SZ from DPM and A10 disagree
by more than ~1 dex, a unit conversion factor (eV→keV or similar) should be
applied to PressureProfileDPM's P_03 on construction.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.cosmology import (
    PressureProfileA10,
    PressureProfileDPM,
    GasDensityDPM,
    MetallicityProfileDPM,
    temperature_from_profiles,
    xray_cooling_function,
    ApecCoolingTable,
    m200_to_m500c,
)
from hod_mod.cosmology.gas_profiles import (
    _RHO_CRIT0,
    _MPC_CM,
    _SIGMA_T_OVER_ME_C2,
    _gnfw_f_params,
)
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum

_HERE    = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

_DATA_SR  = Path("/home/comparat/data/st_mod_data/data/validation/validation_GAS/scaling_relations")
_DATA_POP = Path("/home/comparat/data/st_mod_data/data/benchmark/popesso2024")

_THETA   = LinearPowerSpectrum.default_cosmology()
_H       = float(_THETA["h"])
_OM      = float(_THETA["Omega_m"])

# Latest fit_comparat2025 S1 MAP best-fit (results/fits/comparat2025/S1_map.json,
# chi2/dof=3.84): joint w_theta+wp fit of DPM-2 gas + HAM AGN + ZuMandelbaum15 HOD.
# Supersedes the earlier ad-hoc GAS.py-target calibration (beta_n=0.20, beta_P=0.80).
_FIT_BETA_GAS      = 0.2437   # fit_comparat2025 S1 MAP beta_gas
_FIT_BETA_PRESSURE = 0.8932   # fit_comparat2025 S1 MAP beta_pressure

_MODEL_COLORS  = {1: "C1", 2: "C2", 3: "C3"}
_MODEL_LS      = {1: "-", 2: "--", 3: ":"}
_MODEL_LABELS  = {
    1: r"DPM m.1 ($\beta^n=0$)",
    2: r"DPM m.2 ($\beta^n=0.36$)",
    3: r"DPM m.3 (steep inner)",
}
_MODEL_LABELS_P = {
    1: r"DPM m.1 ($\beta^P=2/3$)",
    2: r"DPM m.2 ($\beta^P=0.85$)",
    3: r"DPM m.3 ($\beta^P=0.92$)",
}

print("Building APEC cooling table (takes ~10 s) ...")
_APEC = ApecCoolingTable(emin=0.5, emax=2.0, n_T=40, n_Z=10)
print("  APEC table ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ez(z):
    return float(np.sqrt(_OM * (1 + z)**3 + (1 - _OM)))


def _rho_crit_z(z):
    """Critical density [(Msun/h)/(Mpc/h)³] at redshift z."""
    return _RHO_CRIT0 * _ez(z)**2 / (1 + z)**3


def _r200(m200_h, z=0.0):
    """R200 [Mpc/h] from M200 [Msun/h] (matter mean density convention)."""
    return (m200_h / (4 / 3 * np.pi * 200 * _RHO_CRIT0 * _OM)) ** (1 / 3)


def _c200_approx(m200_h):
    """Approximate c200 following Dutton+2014 trend (for M200 in Msun/h)."""
    return 9.0 * (m200_h / 1e12) ** (-0.13)


def _make_density_variant(model: int = 2, **overrides) -> GasDensityDPM:
    """Clone GasDensityDPM(model) with one or more parameter overrides.

    Attributes recognised: ne_03, beta, gamma, alpha_in, alpha_tr, alpha_out.
    Renormalises _ne0 after any change to ne_03 or alpha parameters.
    """
    dp = GasDensityDPM(model=model)
    for k, v in overrides.items():
        setattr(dp, f"_{k}", float(v))
    x_ref = 0.3 * dp._C_DPM
    dp._f_xref = dp._gnfw_f(x_ref)
    dp._ne0    = dp._ne_03 / dp._f_xref
    return dp


def _make_pressure_variant(model: int = 2, **overrides) -> PressureProfileDPM:
    """Clone PressureProfileDPM(model) with parameter overrides.

    Attributes recognised: P_03, beta, gamma, alpha_in, alpha_tr, alpha_out_12.
    """
    pp = PressureProfileDPM(model=model)
    for k, v in overrides.items():
        setattr(pp, f"_{k}", float(v))
    x_ref = 0.3 * pp._C_DPM
    f_ref = _gnfw_f_params(x_ref, pp._alpha_in, pp._alpha_tr, pp._alpha_out_12)
    pp._P0 = pp._P_03 / float(f_ref)
    return pp


# Literature power-law fits — Lovisari+2020 arXiv:2004.03401 Table 3
# Soft-band 0.5–2 keV, mass and luminosity in h-free (Msun, erg/s) units at z_ref.

def _lovisari20_lx(m500c_msun, z=0.1):
    """Lx [erg/s] soft band fitted to Lovisari+2020 Table A1 data (E^1 scaling, slope 1.472)."""
    ez = np.sqrt(0.31 * (1 + z)**3 + 0.69)
    log10_lx = 44.293 + 1.472 * np.log10(m500c_msun / 3e14) + np.log10(ez)
    return 10**log10_lx


def _lovisari20_kt(m500c_msun, z=0.1):
    """kT [keV] from Lovisari+2020 Table 3 power law."""
    ez = np.sqrt(0.31 * (1 + z)**3 + 0.69)
    return 5.0 * (ez**(2 / 3) * m500c_msun / 3e14)**0.60


def _comparat25_lx(m500c_msun):
    """Lx [erg/s] 0.5–2 keV from Comparat+2025 (arXiv:2503.19796) cross-corr fit."""
    return 10**(1.612 * (np.log10(m500c_msun) - 15) + 44.7)


def _gas_py_lx(m500c_msun, z=0.1):
    """GAS.py LX–M relation from GAS.populate_cat (Comparat+2025 Uchuu light cone).

    log10(LX / E(z)^2) = 44.7 + 1.61 * (log10 M500c - 15),  sigma_LX = 0.3 dex.
    Returns (mean, +1sigma, -1sigma) in erg/s.
    """
    ez2 = 0.31 * (1.0 + z)**3 + 0.69
    log10_lx = 44.7 + 1.61 * (np.log10(m500c_msun) - 15.0) + np.log10(ez2)
    return 10**log10_lx, 10**(log10_lx + 0.3), 10**(log10_lx - 0.3)


def _gas_py_kt(m500c_msun, z=0.1):
    """GAS.py kT–M relation from GAS.populate_cat (Comparat+2025 Uchuu light cone).

    log10(kT / E(z)^{2/3}) = 0.6 * log10 M500c - 8,  sigma_kT = 0.2 dex.
    Returns (mean, +1sigma, -1sigma) in keV.
    """
    ez = np.sqrt(0.31 * (1.0 + z)**3 + 0.69)
    log10_kt = 0.6 * np.log10(m500c_msun) - 8.0 + (2.0 / 3.0) * np.log10(ez)
    return 10**log10_kt, 10**(log10_kt + 0.2), 10**(log10_kt - 0.2)


def _load_lovisari20_data():
    d = np.loadtxt(_DATA_SR / "lovisari_2020_tableA1.ascii", comments="#")
    return d[:, 2] * 1e14, d[:, 6] * 1e44, d[:, 4]    # M500 [Msun], Lx [erg/s], kT [keV]


def _load_bulbul18():
    d = np.loadtxt(_DATA_SR / "bulbul_2018_table1_2.ascii", comments="#")
    return d[:, 13] * 1e14, d[:, 3] * 1e44, d[:, 5]   # M500 [Msun], LXcin [erg/s], TXcin [keV]


def _load_lovisari15():
    d = np.loadtxt(_DATA_SR / "lovisari_2015_table2.ascii", comments="#")
    return d[:, 5] * 1e13, d[:, 17] * 1e44, d[:, 0]   # M500 [Msun] groups, LXxmm [erg/s], kT [keV]


def _load_zhang24():
    d = np.loadtxt(_DATA_SR / "zhang_2024_HaloMass.ascii", comments="#")
    M_mid  = 10**((d[:, 2] + d[:, 3]) / 2)             # geometric mean of M500c log-bins [Msun]
    Lx     = d[:, 12] * 10**d[:, 14]                   # LX_CGM = value × 10^exp [erg/s]
    Lx_err = d[:, 13] * 10**d[:, 14]
    return M_mid, Lx, Lx_err


def _load_popesso24():
    d = np.loadtxt(_DATA_POP / "SR_fig6_left.ascii", comments="#")
    return 10**d[:, 0], 10**d[:, 1], 10**d[:, 2]      # M500c [Msun], Lx_high, Lx_low [erg/s]


def _calibrate_ne03_P03(beta_n, beta_P, T_min=0.3, z=0.1, n_bisect=22, n_outer=3):
    """Joint calibration: ratio-update algorithm to match GAS.py Lx AND kT at M_pivot.

    Key insight: fix ratio r = P_03/ne_03 so T(r, M) is INDEPENDENT of ne_03.
    Then Lx ∝ ne_03² × Λ(T) is strictly monotone in ne_03 → safe to bisect.
    After bisection, update r ← r × (kT_target / kT_ew) and repeat.
    Converges in 2 outer iterations because kT_ew ∝ r exactly (Λ(T)T/Λ(T) = T ∝ r).

    Returns (ne_03_cal [cm⁻³], P_03_cal [keV/cm³]).
    """
    m_pivot  = 4e14 * _H
    r_pivot  = _r200(m_pivot, z)
    c2_pivot = _c200_approx(m_pivot)
    m500_p, r500_p = m200_to_m500c(
        np.array([m_pivot]), np.array([c2_pivot]),
        np.array([r_pivot]), _rho_crit_z(z))
    M_piv_msun = float(m500_p[0]) / _H

    Lx_target = _gas_py_lx(M_piv_msun, z=z)[0]
    kT_target = _gas_py_kt(M_piv_msun, z=z)[0]

    met = MetallicityProfileDPM()
    # Initial ratio from DPM model-2 defaults (P_03/ne_03 = 2.36 keV at M_ref)
    ratio = 115.0e-6 / 4.87e-5   # keV (= local T at r=0.3×R200 of M_ref=10^12 Msun/h)

    ne_03, P_03 = 4.87e-5, 115.0e-6   # will be overwritten each outer iteration

    for outer in range(n_outer):
        # Bisect ne_03 for Lx with P_03 = ratio × ne_03.
        # T = P/ne = ratio × M12^Δβ is constant → Lx ∝ ne_03² is monotone.
        ne_lo, ne_hi = 1e-9, 1e-2
        for _ in range(n_bisect):
            ne_mid = np.sqrt(ne_lo * ne_hi)
            dp = _make_density_variant(model=2, ne_03=ne_mid, beta=beta_n)
            pp = _make_pressure_variant(model=2, P_03=ratio * ne_mid, beta=beta_P)
            lx, _, _ = _integrate_profile(
                m_pivot, r_pivot, float(r500_p[0]), z, pp, dp, met, T_min=T_min)
            if lx > Lx_target:
                ne_hi = ne_mid
            else:
                ne_lo = ne_mid
        ne_03 = np.sqrt(ne_lo * ne_hi)
        P_03  = ratio * ne_03

        # Compute actual kT_ew and update ratio (kT_ew ∝ ratio exactly)
        dp = _make_density_variant(model=2, ne_03=ne_03, beta=beta_n)
        pp = _make_pressure_variant(model=2, P_03=P_03, beta=beta_P)
        lx_chk, kt_chk, _ = _integrate_profile(
            m_pivot, r_pivot, float(r500_p[0]), z, pp, dp, met, T_min=T_min)
        print(f"  outer {outer+1}: ne_03={ne_03:.3e}, P_03={P_03:.3e}, "
              f"Lx={lx_chk:.2e}/{Lx_target:.2e}, kT={kt_chk:.2f}/{kT_target:.2f} keV")
        ratio *= kT_target / kt_chk   # exact correction for next iteration

    print(f"  Calibrated: ne_03={ne_03:.3e} cm⁻³, P_03={P_03:.3e} keV/cm³"
          f"  (defaults: 4.87e-05, 1.15e-04)")
    return ne_03, P_03


# ---------------------------------------------------------------------------
# Figure 1 — radial profiles
# ---------------------------------------------------------------------------

def fig_radial_profiles():
    """2×2 panel: P_e, n_e, Z, T vs r/R200 at M=1e14 Msun/h, z=0.1."""
    print("Figure 1: radial profiles ...")
    z  = 0.1
    m  = 1e14
    r2 = _r200(m, z)

    x_arr = np.logspace(-2, np.log10(4), 300)
    r_arr = x_arr * r2

    # A10 pressure — needs m500c/r500c
    c200_ref = 5.0
    m500, r500 = m200_to_m500c(
        np.array([m]), np.array([c200_ref]), np.array([r2]), _rho_crit_z(z)
    )
    m500 = float(m500[0]); r500 = float(r500[0])
    a10  = PressureProfileA10()
    Pe_a10 = a10._p3d(r_arr / r500, m500, z, _H, _OM)

    dp_models = {i: GasDensityDPM(model=i)    for i in (1, 2, 3)}
    pp_models = {i: PressureProfileDPM(model=i) for i in (1, 2, 3)}
    met = MetallicityProfileDPM()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (0,0) Pressure
    ax = axes[0, 0]
    ax.loglog(x_arr, Pe_a10, "C0-", lw=2, label="A10 (arXiv:0910.1234)")
    for i in (1, 2, 3):
        Pe = pp_models[i]._pressure_3d(r_arr, m, r2, z, _OM)
        ax.loglog(x_arr, Pe, color=_MODEL_COLORS[i], ls=_MODEL_LS[i],
                  lw=2, label=_MODEL_LABELS_P[i])
    ax.axvline(1.0, ls=":", color="gray", alpha=0.5, label=r"$r=R_{200}$")
    ax.axvline(r500 / r2, ls="--", color="gray", alpha=0.4, label=r"$r=R_{500c}$")
    ax.set_xlabel(r"$r/R_{200}$")
    ax.set_ylabel(r"$P_e(r)$ [P$_{0.3}$ units]")
    ax.set_title(r"Electron pressure ($M_{200}=10^{14}\,M_\odot/h$, $z=0.1$)")
    ax.legend(fontsize=7.5)
    ax.set_xlim(0.02, 4)
    ax.grid(True, alpha=0.2)

    # (0,1) Electron density
    ax = axes[0, 1]
    for i in (1, 2, 3):
        ne = dp_models[i].density_3d(r_arr, m, r2, z, _OM)
        ax.loglog(x_arr, ne, color=_MODEL_COLORS[i], ls=_MODEL_LS[i],
                  lw=2, label=_MODEL_LABELS[i])
    ax.axvline(1.0, ls=":", color="gray", alpha=0.5)
    ax.axvline(r500 / r2, ls="--", color="gray", alpha=0.4)
    ax.set_xlabel(r"$r/R_{200}$")
    ax.set_ylabel(r"$n_e(r)$ [cm$^{-3}$]")
    ax.set_title(r"Electron density ($M_{200}=10^{14}\,M_\odot/h$, $z=0.1$)")
    ax.legend(fontsize=7.5)
    ax.set_xlim(0.02, 4)
    ax.grid(True, alpha=0.2)

    # (1,0) Metallicity — no mass/z dependence
    ax = axes[1, 0]
    Z = met.metallicity_3d(r_arr, r2)
    ax.semilogx(x_arr, Z, "k-", lw=2, label=r"$Z_{0.3}=0.3\,Z_\odot$")
    ax.axvline(1.0, ls=":", color="gray", alpha=0.5)
    ax.axvline(r500 / r2, ls="--", color="gray", alpha=0.4)
    ax.set_xlabel(r"$r/R_{200}$")
    ax.set_ylabel(r"$Z(r)$ [$Z_\odot$]")
    ax.set_title("Metallicity (MetallicityProfileDPM, no mass/z dependence)")
    ax.legend(fontsize=9)
    ax.set_xlim(0.02, 4)
    ax.grid(True, alpha=0.2)

    # (1,1) Temperature T = Pe / ne from matched DPM models
    ax = axes[1, 1]
    for i in (1, 2, 3):
        Pe = pp_models[i]._pressure_3d(r_arr, m, r2, z, _OM)
        ne = dp_models[i].density_3d(r_arr, m, r2, z, _OM)
        T  = temperature_from_profiles(Pe, ne)
        ax.loglog(x_arr, T, color=_MODEL_COLORS[i], ls=_MODEL_LS[i],
                  lw=2, label=rf"DPM m.{i}")
    ax.axvline(1.0, ls=":", color="gray", alpha=0.5)
    ax.axvline(r500 / r2, ls="--", color="gray", alpha=0.4)
    ax.set_xlabel(r"$r/R_{200}$")
    ax.set_ylabel(r"$T(r) = P_e/n_e$ [P$_{0.3}$/cm$^{-3}$]")
    ax.set_title(r"Temperature ($M_{200}=10^{14}\,M_\odot/h$, $z=0.1$)")
    ax.legend(fontsize=9)
    ax.set_xlim(0.02, 4)
    ax.grid(True, alpha=0.2)

    fig.suptitle(r"Gas profiles: radial shapes at $M_{200}=10^{14}\,M_\odot/h$, $z=0.1$",
                 fontsize=12)
    fig.tight_layout()
    out = _FIG_DIR / "gas_01_radial_profiles.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — mass scaling
# ---------------------------------------------------------------------------

def fig_mass_scaling():
    """2×2 panel: profile amplitude at r=0.3 R200 vs M200; cooling function."""
    print("Figure 2: mass scaling ...")
    z     = 0.1
    m_arr = np.logspace(11, 15, 40)
    r_arr = _r200(m_arr, z)
    r_ref = 0.3 * r_arr   # evaluation point

    dp_models = {i: GasDensityDPM(model=i)     for i in (1, 2, 3)}
    pp_models = {i: PressureProfileDPM(model=i) for i in (1, 2, 3)}

    # A10 needs m500c at each mass
    c200_arr = _c200_approx(m_arr)
    m500_arr, r500_arr = m200_to_m500c(m_arr, c200_arr, r_arr, _rho_crit_z(z))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (0,0) Pressure at r=0.3 R200 vs M200 — A10 + DPM
    ax = axes[0, 0]
    a10 = PressureProfileA10()
    Pe_a10 = np.array([
        float(a10._p3d(np.array([r_ref[i] / r500_arr[i]]),
                       float(m500_arr[i]), z, _H, _OM)[0])
        for i in range(len(m_arr))
    ])
    ax.loglog(m_arr, Pe_a10, "C0-", lw=2, label="A10")
    for i in (1, 2, 3):
        Pe_ref = np.array([
            float(pp_models[i]._pressure_3d(
                np.array([r_ref[j]]), m_arr[j], r_arr[j], z, _OM)[0])
            for j in range(len(m_arr))
        ])
        ax.loglog(m_arr, Pe_ref, color=_MODEL_COLORS[i], ls=_MODEL_LS[i],
                  lw=2, label=_MODEL_LABELS_P[i])
    ax.set_xlabel(r"$M_{200}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$P_e(0.3\,R_{200})$ [P$_{0.3}$ units]")
    ax.set_title(r"Pressure amplitude at $r=0.3\,R_{200}$, $z=0.1$")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2)

    # (0,1) Electron density at r=0.3 R200 vs M200
    ax = axes[0, 1]
    for i in (1, 2, 3):
        ne_ref = np.array([
            float(dp_models[i].density_3d(
                np.array([r_ref[j]]), m_arr[j], r_arr[j], z, _OM)[0])
            for j in range(len(m_arr))
        ])
        ax.loglog(m_arr, ne_ref, color=_MODEL_COLORS[i], ls=_MODEL_LS[i],
                  lw=2, label=_MODEL_LABELS[i])
        # power-law guide
        plaw = ne_ref[20] * (m_arr / m_arr[20]) ** dp_models[i]._beta
        ax.loglog(m_arr, plaw, color=_MODEL_COLORS[i], lw=0.8, alpha=0.4)
    ax.set_xlabel(r"$M_{200}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$n_e(0.3\,R_{200})$ [cm$^{-3}$]")
    ax.set_title(r"Density amplitude at $r=0.3\,R_{200}$, $z=0.1$")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2)

    # (1,0) Temperature at r=0.3 R200 vs M200
    ax = axes[1, 0]
    for i in (1, 2, 3):
        T_ref = np.array([
            float(temperature_from_profiles(
                pp_models[i]._pressure_3d(np.array([r_ref[j]]), m_arr[j], r_arr[j], z, _OM),
                dp_models[i].density_3d(np.array([r_ref[j]]), m_arr[j], r_arr[j], z, _OM)
            )[0])
            for j in range(len(m_arr))
        ])
        ax.loglog(m_arr, T_ref, color=_MODEL_COLORS[i], ls=_MODEL_LS[i],
                  lw=2, label=rf"DPM m.{i}")
    ax.set_xlabel(r"$M_{200}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$T(0.3\,R_{200}) = P_e/n_e$ [P$_{0.3}$/cm$^{-3}$]")
    ax.set_title(r"Temperature amplitude at $r=0.3\,R_{200}$, $z=0.1$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (1,1) Cooling function Λ(T,Z) — APEC table, varying metallicity
    ax = axes[1, 1]
    T_grid = np.logspace(-1.5, 1.5, 200)
    for Z_val, ls, color in [(0.1, ":", "C3"), (0.3, "--", "C2"), (1.0, "-", "C0")]:
        Lambda = _APEC(T_grid, np.full_like(T_grid, Z_val))
        ax.loglog(T_grid, Lambda, ls=ls, lw=2, color=color,
                  label=rf"$Z={Z_val}\,Z_\odot$ (APEC 0.5–2 keV)")
    ax.set_xlabel(r"$T$ [keV]")
    ax.set_ylabel(r"$\Lambda(T, Z)$ [erg cm$^3$ s$^{-1}$]")
    ax.set_title(r"X-ray cooling function (APEC table, soxs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle(r"Gas profile amplitudes at $r=0.3\,R_{200}$, $z=0.1$", fontsize=12)
    fig.tight_layout()
    out = _FIG_DIR / "gas_02_mass_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — parameter sensitivity
# ---------------------------------------------------------------------------

def fig_parameter_sensitivity():
    """2×3 panel: effect of ne_03/beta/alpha_out on density and pressure profiles."""
    print("Figure 3: parameter sensitivity ...")
    z   = 0.1
    m   = 1e13
    r2  = _r200(m, z)
    x_arr = np.logspace(-2, np.log10(4), 300)
    r_arr = x_arr * r2

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # ---- Row 0: GasDensityDPM ----

    # (0,0) vary ne_03 — amplitude only
    ax = axes[0, 0]
    ne_03_ref = GasDensityDPM(model=2)._ne_03
    for fac, ls, lw in [(0.3, "--", 1.5), (1.0, "-", 2.5), (3.0, ":", 1.5)]:
        dp = _make_density_variant(model=2, ne_03=ne_03_ref * fac)
        ne = dp.density_3d(r_arr, m, r2, z, _OM)
        ax.loglog(x_arr, ne, "C2", ls=ls, lw=lw,
                  label=rf"$n_{{e,0.3}}\times{fac}$")
    ax.axvline(1.0, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel(r"$r/R_{200}$"); ax.set_ylabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_title(r"Density: vary $n_{e,0.3}$ (model 2)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xlim(0.02, 4)

    # (0,1) vary beta (mass-scaling slope) — show at 3 masses
    ax = axes[0, 1]
    m_arr_3 = np.array([1e12, 1e13, 1e14])
    r_arr_3 = _r200(m_arr_3, z)
    for beta, ls, color in [(0.0, "--", "C1"), (0.36, "-", "C2"), (0.7, ":", "C3")]:
        dp = _make_density_variant(model=2, beta=beta)
        ne_ref = np.array([
            float(dp.density_3d(np.array([0.3 * r_arr_3[j]]),
                                m_arr_3[j], r_arr_3[j], z, _OM)[0])
            for j in range(3)
        ])
        ax.loglog(m_arr_3, ne_ref, color=color, ls=ls, lw=2, marker="o",
                  label=rf"$\beta={beta}$")
    ax.set_xlabel(r"$M_{200}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$n_e(0.3\,R_{200})$ [cm$^{-3}$]")
    ax.set_title(r"Density: vary $\beta$ (mass scaling)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (0,2) vary alpha_out — outer slope
    ax = axes[0, 2]
    for alpha_out, ls, lw in [(2.0, "--", 1.5), (2.7, "-", 2.5), (3.5, ":", 1.5)]:
        dp = _make_density_variant(model=2, alpha_out=alpha_out)
        ne = dp.density_3d(r_arr, m, r2, z, _OM)
        ax.loglog(x_arr, ne, "C2", ls=ls, lw=lw,
                  label=rf"$\alpha_{{\rm out}}={alpha_out}$")
    ax.axvline(1.0, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel(r"$r/R_{200}$"); ax.set_ylabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_title(r"Density: vary $\alpha_{\rm out}$ (outer slope)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xlim(0.02, 4)

    # ---- Row 1: PressureProfileDPM ----

    # (1,0) vary P_03 — amplitude only
    ax = axes[1, 0]
    P_03_ref = PressureProfileDPM(model=2)._P_03
    for fac, ls, lw in [(0.3, "--", 1.5), (1.0, "-", 2.5), (3.0, ":", 1.5)]:
        pp = _make_pressure_variant(model=2, P_03=P_03_ref * fac)
        Pe = pp._pressure_3d(r_arr, m, r2, z, _OM)
        ax.loglog(x_arr, Pe, "C4", ls=ls, lw=lw,
                  label=rf"$P_{{0.3}}\times{fac}$")
    ax.axvline(1.0, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel(r"$r/R_{200}$"); ax.set_ylabel(r"$P_e$ [P$_{0.3}$ units]")
    ax.set_title(r"Pressure: vary $P_{0.3}$ (model 2)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xlim(0.02, 4)

    # (1,1) vary beta — show at 3 masses
    ax = axes[1, 1]
    for beta, ls, color in [(0.50, "--", "C1"), (0.85, "-", "C2"), (1.10, ":", "C3")]:
        pp = _make_pressure_variant(model=2, beta=beta)
        Pe_ref = np.array([
            float(pp._pressure_3d(np.array([0.3 * r_arr_3[j]]),
                                  m_arr_3[j], r_arr_3[j], z, _OM)[0])
            for j in range(3)
        ])
        ax.loglog(m_arr_3, Pe_ref, color=color, ls=ls, lw=2, marker="o",
                  label=rf"$\beta={beta}$")
    ax.set_xlabel(r"$M_{200}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$P_e(0.3\,R_{200})$ [P$_{0.3}$ units]")
    ax.set_title(r"Pressure: vary $\beta$ (mass scaling)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (1,2) vary alpha_out_12 — outer slope
    ax = axes[1, 2]
    for alpha_out, ls, lw in [(3.5, "--", 1.5), (4.1, "-", 2.5), (5.0, ":", 1.5)]:
        pp = _make_pressure_variant(model=2, alpha_out_12=alpha_out)
        Pe = pp._pressure_3d(r_arr, m, r2, z, _OM)
        ax.loglog(x_arr, Pe, "C4", ls=ls, lw=lw,
                  label=rf"$\alpha_{{\rm out}}={alpha_out}$")
    ax.axvline(1.0, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel(r"$r/R_{200}$"); ax.set_ylabel(r"$P_e$ [P$_{0.3}$ units]")
    ax.set_title(r"Pressure: vary $\alpha_{\rm out}$ (outer slope)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xlim(0.02, 4)

    fig.suptitle("DPM parameter sensitivity (model 2 base)",  fontsize=12)
    fig.tight_layout()
    out = _FIG_DIR / "gas_03_parameter_sensitivity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — scaling relations
# ---------------------------------------------------------------------------

def _integrate_profile(m200, r200, r500, z, pp, dp, met, n_r=250, T_min=None):
    """Integrate gas profiles from 0.01*r200 to r500.

    Returns (Lx [erg/s], kT_ew [keV], Y_DA2 [Mpc²]).
    Lx and kT use T = P/ne with APEC cooling table (0.5–2 keV, soxs).

    T_min [keV]: if set, only gas with T > T_min contributes to Lx and kT_ew.
    This approximates the X-ray selection: 0.5–2 keV band detects gas T > ~0.1–0.3 keV.
    DPM ne includes all ionized phases (warm CGM + hot ICM); T_min selects only the
    X-ray-emitting hot phase, removing the ~100–1000× overestimate at low masses.
    """
    r_lo = 0.01 * r200
    r_hi = min(r500, 3.0 * r200)
    r_h  = np.linspace(r_lo, r_hi, n_r)           # Mpc/h
    r_cm = r_h * (_MPC_CM / _H)                    # physical cm

    ne = dp.density_3d(r_h, m200, r200, z, _OM)
    Pe = pp._pressure_3d(r_h, m200, r200, z, _OM)
    Z  = met.metallicity_3d(r_h, r200)

    T      = temperature_from_profiles(Pe, ne)     # keV
    Lambda = _APEC(T, Z)

    xray_w = np.where(T > T_min, ne**2, 0.0) if T_min is not None else ne**2
    r2c    = r_cm**2

    _trapz = np.trapezoid
    Lx    = 4 * np.pi * float(_trapz(xray_w * Lambda * r2c, r_cm))
    denom = float(_trapz(xray_w * r2c, r_cm))
    kT_ew = (float(_trapz(xray_w * T * r2c, r_cm)) / denom) if denom > 0 else 0.0
    Y_DA2 = (_SIGMA_T_OVER_ME_C2
             * 4 * np.pi
             * float(_trapz(Pe * r2c, r_cm))
             / _MPC_CM**2)                          # Mpc²

    return Lx, kT_ew, Y_DA2


def fig_scaling_relations():
    """2×2 panel: Lx, kT, Y×D_A², Lx–kT vs M500c.

    Panel (0,0) shows:
    - DPM model 2 integrating all ionized gas (warm CGM + hot ICM) — grey, thin
    - DPM model 2 with T_min=0.1 keV and T_min=0.3 keV cuts (X-ray selection)
    - Individual cluster data: Lovisari+2020, Bulbul+2018, Lovisari+2015 groups
    - Stacked/band data: Zhang+2024 CGM, Popesso+2024
    - GAS.py (Comparat+2025 Uchuu LC): analytical LX/kT–M scaling relations + scatter bands

    Why the DPM Lx is so high: GasDensityDPM ne is calibrated to *total* electron
    density across all ionized phases. X-ray observations (0.5–2 keV) detect only
    gas with T > ~0.1–0.3 keV. Applying T_min removes the warm CGM contribution
    and brings DPM into the observational ballpark without changing model parameters.
    """
    print("Figure 4: scaling relations (this may take ~60 s) ...")
    z        = 0.1
    m_arr    = np.logspace(11.5, 15.2, 35)      # Msun/h
    r_arr    = _r200(m_arr, z)
    c200_arr = _c200_approx(m_arr)
    m500_arr, r500_arr = m200_to_m500c(m_arr, c200_arr, r_arr, _rho_crit_z(z))
    m500_msun = m500_arr / _H                    # h-free Msun
    met = MetallicityProfileDPM()

    dp_models = {i: GasDensityDPM(model=i)     for i in (1, 2, 3)}
    pp_models = {i: PressureProfileDPM(model=i) for i in (1, 2, 3)}
    a10 = PressureProfileA10()

    # Calibrated model: beta_n, beta_P fixed to the fit_comparat2025 S1 MAP best-fit;
    # ne_03, P_03 amplitude found by bisection to match GAS.py Lx AND kT targets.
    print("  Running ne_03, P_03 calibration ...")
    _BN_CAL, _BP_CAL = _FIT_BETA_GAS, _FIT_BETA_PRESSURE
    alpha_lx_cal = 1.0 + 1.5 * _BN_CAL + 0.5 * _BP_CAL
    alpha_kt_cal = _BP_CAL - _BN_CAL
    ne_cal, P03_cal = _calibrate_ne03_P03(_BN_CAL, _BP_CAL, T_min=0.3, z=z)
    dp_cal  = _make_density_variant(model=2, ne_03=ne_cal, beta=_BN_CAL)
    pp_cal  = _make_pressure_variant(model=2, P_03=P03_cal, beta=_BP_CAL)

    # Model 2: no cut, T>0.3; calibrated model T>0.3
    Lx_nocut  = np.zeros(len(m_arr))
    Lx_t03    = np.zeros(len(m_arr))
    Lx_cal    = np.zeros(len(m_arr))
    kT_nocut  = np.zeros(len(m_arr))
    kT_t03    = np.zeros(len(m_arr))
    kT_cal    = np.zeros(len(m_arr))
    Y    = {i: np.zeros(len(m_arr)) for i in (1, 2, 3)}
    Y_a10 = np.zeros(len(m_arr))

    pp2 = pp_models[2]; dp2 = dp_models[2]
    for j, (m, r2, r5, c2) in enumerate(
            zip(m_arr, r_arr, r500_arr, c200_arr)):
        lx, kt, _ = _integrate_profile(m, r2, r5, z, pp2, dp2, met)
        Lx_nocut[j] = lx; kT_nocut[j] = kt
        lx, kt, _ = _integrate_profile(m, r2, r5, z, pp2, dp2, met, T_min=0.3)
        Lx_t03[j] = lx; kT_t03[j] = kt
        lx, kt, _ = _integrate_profile(m, r2, r5, z, pp_cal, dp_cal, met, T_min=0.3)
        Lx_cal[j] = lx; kT_cal[j] = kt

        for i in (1, 2, 3):
            _, _, y = _integrate_profile(m, r2, r5, z,
                                         pp_models[i], dp_models[i], met)
            Y[i][j] = y
        r_h  = np.linspace(0.01 * r2, min(r5, 3 * r2), 250)
        r_cm = r_h * (_MPC_CM / _H)
        m5s  = float(m500_arr[j]); r5s = float(r5)
        Pe_a10 = a10._p3d(r_h / r5s, m5s, z, _H, _OM)
        Y_a10[j] = (_SIGMA_T_OVER_ME_C2 * 4 * np.pi
                    * float(np.trapezoid(Pe_a10 * r_cm**2, r_cm)) / _MPC_CM**2)

    # Literature power laws
    m_lit  = np.logspace(12, 15.5, 100)
    Lx_lit = _lovisari20_lx(m_lit, z=z)
    kT_lit = _lovisari20_kt(m_lit, z=z)
    Lx_c25, Lx_c25_hi, Lx_c25_lo = _gas_py_lx(m_lit, z=z)
    kT_c25, kT_c25_hi, kT_c25_lo = _gas_py_kt(m_lit, z=z)

    # Individual data points
    M_lo20, Lx_lo20, kT_lo20 = _load_lovisari20_data()
    M_bu18, Lx_bu18, kT_bu18 = _load_bulbul18()
    M_lo15, Lx_lo15, kT_lo15 = _load_lovisari15()
    M_zh24, Lx_zh24, Lx_zh24_err = _load_zhang24()
    M_po24, Lx_po24_hi, Lx_po24_lo = _load_popesso24()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── (0,0) Lx vs M500c ──────────────────────────────────────────────────
    ax = axes[0, 0]
    # Shaded literature bands
    ax.fill_between(m_lit, Lx_lit / 10**0.2, Lx_lit * 10**0.2,
                    alpha=0.12, color="gray")
    ax.fill_between(M_po24, Lx_po24_lo, Lx_po24_hi,
                    alpha=0.15, color="purple", label="Popesso+2024")
    # Literature power laws
    ax.loglog(m_lit, Lx_lit, "k--", lw=1.5, label="Lovisari+2020")
    ax.fill_between(m_lit, Lx_c25_lo, Lx_c25_hi, alpha=0.15, color="red",
                    label=r"GAS.py $\pm1\sigma$ (0.3 dex)")
    ax.loglog(m_lit, Lx_c25, "r-.", lw=1.5,
              label=r"GAS.py: $\log L_X/E^2=44.7+1.61(\log M_{500c}-15)$")
    # Individual / stacked points
    ax.scatter(M_lo20, Lx_lo20, s=12, color="gray",   alpha=0.7,
               marker="o", label="Lovisari+2020 clusters", zorder=5)
    ax.scatter(M_bu18, Lx_bu18, s=14, color="steelblue", alpha=0.7,
               marker="s", label="Bulbul+2018 clusters", zorder=5)
    ax.scatter(M_lo15, Lx_lo15, s=14, color="green",  alpha=0.7,
               marker="^", label="Lovisari+2015 groups", zorder=5)
    ax.errorbar(M_zh24, Lx_zh24, yerr=Lx_zh24_err, fmt="D",
                color="darkorange", ms=5, lw=1.2, label="Zhang+2024 CGM stacks", zorder=6)
    # DPM model 2 curves + calibrated model
    ax.loglog(m500_msun, Lx_nocut, color="0.6",  ls="-",  lw=1.2,
              label="DPM m.2 all gas", zorder=3)
    ax.loglog(m500_msun, Lx_t03,   color="C2",   ls="--", lw=1.8,
              label=r"DPM m.2, $T>0.3$ keV", zorder=4)
    ax.loglog(m500_msun, Lx_cal,   color="C3",   ls="-",  lw=2.5,
              label=rf"Calibrated ($\beta_n={_BN_CAL:.2f}$, $\alpha_{{L_X}}={alpha_lx_cal:.2f}$)", zorder=5)
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$L_X$ (0.5–2 keV) [erg s$^{-1}$]")
    ax.set_title(r"$L_X$–$M_{500c}$ at $z=0.1$ (integrated to $R_{500c}$)")
    ax.set_xlim(1e11, 3e15); ax.set_ylim(1e38, 1e47)
    ax.legend(fontsize=6.5, ncol=2); ax.grid(True, alpha=0.2)

    # ── (0,1) kT vs M500c ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.fill_between(m_lit, kT_lit / 10**0.15, kT_lit * 10**0.15,
                    alpha=0.12, color="gray")
    ax.loglog(m_lit, kT_lit, "k--", lw=1.5, label="Lovisari+2020")
    ax.scatter(M_lo20, kT_lo20, s=12, color="gray",    alpha=0.7,
               marker="o", label="Lovisari+2020", zorder=5)
    ax.scatter(M_bu18, kT_bu18, s=14, color="steelblue", alpha=0.7,
               marker="s", label="Bulbul+2018",   zorder=5)
    ax.scatter(M_lo15, kT_lo15, s=14, color="green",   alpha=0.7,
               marker="^", label="Lovisari+2015 groups", zorder=5)
    ax.loglog(m500_msun, kT_nocut, color="0.6", ls="-",  lw=1.2,
              label="DPM m.2 all gas")
    ax.loglog(m500_msun, kT_t03,   color="C2",  ls="--", lw=1.8,
              label=r"DPM m.2, $T>0.3$ keV")
    ax.loglog(m500_msun, kT_cal,   color="C3",  ls="-",  lw=2.5,
              label=rf"Calibrated ($\alpha_{{kT}}={alpha_kt_cal:.2f}$)", zorder=5)
    ax.fill_between(m_lit, kT_c25_lo, kT_c25_hi, alpha=0.15, color="red")
    ax.loglog(m_lit, kT_c25, "r-.", lw=1.5,
              label=r"GAS.py: $\log kT/E^{2/3}=0.6\log M_{500c}-8$")
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$kT_{\rm ew}$ [keV]")
    ax.set_title(r"Emission-weighted $T$ vs $M_{500c}$ at $z=0.1$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # ── (1,0) Y×D_A² vs M500c ──────────────────────────────────────────────
    ax = axes[1, 0]
    ax.loglog(m500_msun, Y_a10, "C0-", lw=2, label="A10 pressure")
    for i in (1, 2, 3):
        ax.loglog(m500_msun, Y[i], color=_MODEL_COLORS[i], ls=_MODEL_LS[i],
                  lw=2, label=_MODEL_LABELS_P[i])
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$Y \cdot D_A^2$ [Mpc$^2$]")
    ax.set_title(r"tSZ signal $Y\cdot D_A^2$ vs $M_{500c}$ at $z=0.1$")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.2)

    # ── (1,1) Lx vs kT ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.loglog(kT_lit, Lx_lit, "k--", lw=1.5, label="Lovisari+2020")
    ax.loglog(kT_c25, Lx_c25, "r-.", lw=1.5, label="GAS.py")
    ax.scatter(kT_lo20, Lx_lo20, s=12, color="gray",    alpha=0.7,
               marker="o", label="Lovisari+2020", zorder=5)
    ax.scatter(kT_bu18, Lx_bu18, s=14, color="steelblue", alpha=0.7,
               marker="s", label="Bulbul+2018",   zorder=5)
    ax.scatter(kT_lo15, Lx_lo15, s=14, color="green",   alpha=0.7,
               marker="^", label="Lovisari+2015 groups", zorder=5)
    ax.loglog(kT_nocut, Lx_nocut, color="0.6", ls="-",  lw=1.2,
              label="DPM m.2 all gas")
    ax.loglog(kT_t03,   Lx_t03,   color="C2",  ls="--", lw=1.8,
              label=r"DPM m.2, $T>0.3$ keV")
    ax.loglog(kT_cal,   Lx_cal,   color="C3",  ls="-",  lw=2.5,
              label=rf"Calibrated ($\beta_n={_BN_CAL:.2f}$)", zorder=5)
    ax.set_xlabel(r"$kT_{\rm ew}$ [keV]")
    ax.set_ylabel(r"$L_X$ (0.5–2 keV) [erg s$^{-1}$]")
    ax.set_title(r"$L_X$–$kT$ at $z=0.1$")
    ax.legend(fontsize=6.5); ax.grid(True, alpha=0.2)

    fig.suptitle(
        r"Gas scaling relations at $z=0.1$ — literature + DPM + calibrated model"
        "\n"
        rf"Calibrated ($\beta_n,\beta_P$ = fit_comparat2025 S1 MAP): "
        rf"$\beta_n={_BN_CAL:.3f}$, $\beta_P={_BP_CAL:.3f}$, "
        rf"$n_{{e,0.3}}={ne_cal:.1e}$ cm$^{{-3}}$, $T>0.3$ keV"
        "\n"
        rf"DPM m.2 defaults give $\alpha_{{L_X}}=1.97$, $\alpha_{{kT}}=0.49$; "
        rf"fitted slopes give $\alpha_{{L_X}}={alpha_lx_cal:.2f}$, $\alpha_{{kT}}={alpha_kt_cal:.2f}$",
        fontsize=10)
    fig.tight_layout()
    out = _FIG_DIR / "gas_04_scaling_relations.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 — pressure unit correction diagnostic
# ---------------------------------------------------------------------------

# Unit correction already applied in gas_profiles.py _PARAMS (P_03 ×1e-6).
# This figure re-runs the integration with the corrected profiles to confirm.
_P_DPM_CORR = 1.0   # no additional correction needed post-fix


def _integrate_profile_pcorr(m200, r200, r500, z, pp, dp, met,
                              p_corr=1.0, n_r=250):
    """Like _integrate_profile but applies p_corr to the pressure."""
    r_lo = 0.01 * r200
    r_hi = min(r500, 3.0 * r200)
    r_h  = np.linspace(r_lo, r_hi, n_r)
    r_cm = r_h * (_MPC_CM / _H)

    ne = dp.density_3d(r_h, m200, r200, z, _OM)
    Pe = pp._pressure_3d(r_h, m200, r200, z, _OM) * p_corr
    Z  = met.metallicity_3d(r_h, r200)

    T      = temperature_from_profiles(Pe, ne)
    Lambda = _APEC(T, Z)

    ne2 = ne**2
    r2c = r_cm**2
    _trapz = np.trapezoid

    Lx    = 4 * np.pi * float(_trapz(ne2 * Lambda * r2c, r_cm))
    kT_ew = (float(_trapz(ne2 * T * r2c, r_cm))
             / float(_trapz(ne2 * r2c, r_cm)))
    Y_DA2 = (_SIGMA_T_OVER_ME_C2 * 4 * np.pi
             * float(_trapz(Pe * r2c, r_cm)) / _MPC_CM**2)
    return Lx, kT_ew, Y_DA2


def fig_pressure_unit_correction():
    """4-panel post-fix comparison: DPM model 2 vs A10 vs Lovisari+2020.

    Confirms the ×1e-6 unit correction (meV cm⁻³ → keV cm⁻³) applied to
    PressureProfileDPM._PARAMS P_03 values.  Both "raw" and "×1e-6" curves
    now overlap because the fix is already in gas_profiles.py.
    The residual DPM/A10 Y_SZ ratio (~10–30) is physical: DPM integrates
    all ionized gas (including warm CGM) while A10 is calibrated to hot
    X-ray ICM in hydrostatic equilibrium.
    """
    print("Figure 5: pressure unit-correction diagnostic ...")
    z     = 0.1
    m_arr = np.logspace(11.5, 15, 25)
    r_arr = _r200(m_arr, z)
    c200_arr = _c200_approx(m_arr)
    m500_arr, r500_arr = m200_to_m500c(m_arr, c200_arr, r_arr, _rho_crit_z(z))
    m500_msun = m500_arr / _H
    met  = MetallicityProfileDPM()
    dp2  = GasDensityDPM(model=2)
    pp2  = PressureProfileDPM(model=2)
    a10  = PressureProfileA10()

    Lx_raw  = np.zeros(len(m_arr))
    Lx_cor  = np.zeros(len(m_arr))
    kT_raw  = np.zeros(len(m_arr))
    kT_cor  = np.zeros(len(m_arr))
    Y_raw   = np.zeros(len(m_arr))
    Y_cor   = np.zeros(len(m_arr))
    Y_a10   = np.zeros(len(m_arr))

    _trapz = np.trapezoid
    for j, (m, r2, r5, c2) in enumerate(zip(m_arr, r_arr, r500_arr, c200_arr)):
        Lx_raw[j], kT_raw[j], Y_raw[j] = _integrate_profile_pcorr(
            m, r2, r5, z, pp2, dp2, met, p_corr=1.0)
        Lx_cor[j], kT_cor[j], Y_cor[j] = _integrate_profile_pcorr(
            m, r2, r5, z, pp2, dp2, met, p_corr=_P_DPM_CORR)
        # A10 Y_SZ
        m5s = float(m200_to_m500c(
            np.array([m]), np.array([c2]), np.array([r2]), _rho_crit_z(z))[0][0])
        r5s = float(m200_to_m500c(
            np.array([m]), np.array([c2]), np.array([r2]), _rho_crit_z(z))[1][0])
        r_h  = np.linspace(0.01 * r2, min(r5s, 3 * r2), 250)
        r_cm = r_h * (_MPC_CM / _H)
        Pe_a10_arr = a10._p3d(r_h / r5s, m5s, z, _H, _OM)
        Y_a10[j] = (_SIGMA_T_OVER_ME_C2 * 4 * np.pi
                    * float(_trapz(Pe_a10_arr * r_cm**2, r_cm)) / _MPC_CM**2)

    m_lit = np.logspace(12.5, 15.5, 80)
    Lx_lit = _lovisari20_lx(m_lit, z=z)
    kT_lit = _lovisari20_kt(m_lit, z=z)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) Y_SZ comparison — key diagnostic
    ax = axes[0, 0]
    ax.loglog(m500_msun, Y_a10,  "C0-",  lw=2.5, label="A10 pressure (reference)")
    ax.loglog(m500_msun, Y_raw,  "C2--", lw=2,   label=r"DPM m.2 raw ($P_{0.3}$ as stored)")
    ax.loglog(m500_msun, Y_cor,  "C2-",  lw=2.5, label=r"DPM m.2 × $10^{-6}$ (μeV→keV)")
    ax.axhspan(1e-5, 1e-2, alpha=0.06, color="C0", label="Planck cluster range (ref.)")
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$Y\cdot D_A^2$ [Mpc$^2$]")
    ax.set_title(r"tSZ $Y\cdot D_A^2$: unit correction diagnostic")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (0,1) kT diagnostic
    ax = axes[0, 1]
    ax.loglog(m500_msun, kT_raw, "C2--", lw=2,   label=r"DPM m.2 raw $T=P/n_e$")
    ax.loglog(m500_msun, kT_cor, "C2-",  lw=2.5, label=r"DPM m.2 × $10^{-6}$ [keV]")
    ax.loglog(m_lit,     kT_lit, "k--",  lw=1.5, label="Lovisari+2020")
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$kT_{\rm ew}$ [keV]")
    ax.set_title(r"Emission-weighted $T$: unit correction")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (1,0) Lx diagnostic
    ax = axes[1, 0]
    ax.fill_between(m_lit, Lx_lit / 10**0.2, Lx_lit * 10**0.2,
                    alpha=0.15, color="gray")
    ax.loglog(m_lit,     Lx_lit, "k--",  lw=1.5, label="Lovisari+2020")
    ax.loglog(m500_msun, Lx_raw, "C2--", lw=2,   label="DPM m.2 raw")
    ax.loglog(m500_msun, Lx_cor, "C2-",  lw=2.5, label=r"DPM m.2 × $10^{-6}$")
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$L_X$ [erg s$^{-1}$]")
    ax.set_title(r"$L_X$–$M$: unit correction")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (1,1) Ratio DPM/A10 for Y_SZ — quantify the offset
    ax = axes[1, 1]
    ratio_raw = Y_raw / Y_a10
    ratio_cor = Y_cor / Y_a10
    ax.loglog(m500_msun, ratio_raw, "C2--", lw=2,   label="DPM raw / A10")
    ax.loglog(m500_msun, ratio_cor, "C2-",  lw=2.5, label=r"DPM × $10^{-6}$ / A10")
    ax.axhline(1.0, ls=":", color="gray", label="Ratio = 1")
    ax.axhline(1e-6, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$Y_{\rm DPM} / Y_{\rm A10}$")
    ax.set_title(r"Y$_{\rm SZ}$ ratio: DPM / A10 (measures unit offset)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.suptitle(
        r"Post-fix: DPM pressure (meV cm$^{-3}$ → keV cm$^{-3}$, ×$10^{-6}$ applied in gas\_profiles.py)"
        "\n"
        r"Residual DPM/A10 Y$_{\rm SZ}$ ratio $\approx10$–30 is physical (DPM = total gas, A10 = hot ICM)",
        fontsize=11)
    fig.tight_layout()
    out = _FIG_DIR / "gas_05_pressure_unit_correction.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 6 — X-ray calibration: slope analysis + calibrated DPM model
# ---------------------------------------------------------------------------

def fig_xray_calibration():
    """2×2 panel: slope diagnostics and calibrated DPM model vs X-ray data.

    The DPM model 2 (β_n=0.36, β_P=0.85) predicts:
      α_Lx = 1 + 1.5×β_n + 0.5×β_P = 1.97  (too steep; observed: 1.70)
      α_kT = β_P - β_n = 0.49              (too flat;  observed: 0.60)

    Calibrated model: β_n, β_P fixed to the fit_comparat2025 S1 MAP best-fit
    (results/fits/comparat2025/S1_map.json, chi2/dof=3.84), giving
      α_Lx = 1 + 1.5×β_n + 0.5×β_P,  α_kT = β_P - β_n  (computed below).

    ne_03 is found by bisection to match Lovisari+2020 at M500c≈4×10^14 Msun.
    """
    print("Figure 6: X-ray calibration (this may take ~90 s) ...")
    z = 0.1
    BN_CAL = _FIT_BETA_GAS       # fit_comparat2025 S1 MAP beta_gas
    BP_CAL = _FIT_BETA_PRESSURE  # fit_comparat2025 S1 MAP beta_pressure
    ALPHA_LX_CAL = 1.0 + 1.5 * BN_CAL + 0.5 * BP_CAL
    ALPHA_KT_CAL = BP_CAL - BN_CAL

    print("  Running bisection for ne_03, P_03 ...")
    ne_cal, P03_cal = _calibrate_ne03_P03(BN_CAL, BP_CAL, T_min=0.3, z=z)

    # Mass array for model curves
    m_arr    = np.logspace(11.5, 15.2, 35)
    r_arr    = _r200(m_arr, z)
    c200_arr = _c200_approx(m_arr)
    m500_arr, r500_arr = m200_to_m500c(m_arr, c200_arr, r_arr, _rho_crit_z(z))
    m500_msun = m500_arr / _H

    met    = MetallicityProfileDPM()
    dp2    = GasDensityDPM(model=2)
    pp2    = PressureProfileDPM(model=2)
    dp_cal = _make_density_variant(model=2, ne_03=ne_cal, beta=BN_CAL)
    pp_cal = _make_pressure_variant(model=2, P_03=P03_cal, beta=BP_CAL)

    Lx_nocut = np.zeros(len(m_arr))
    Lx_t03   = np.zeros(len(m_arr))
    Lx_cal   = np.zeros(len(m_arr))
    kT_nocut = np.zeros(len(m_arr))
    kT_t03   = np.zeros(len(m_arr))
    kT_cal   = np.zeros(len(m_arr))

    for j, (m, r2, r5) in enumerate(zip(m_arr, r_arr, r500_arr)):
        lx, kt, _ = _integrate_profile(m, r2, r5, z, pp2, dp2, met)
        Lx_nocut[j] = lx; kT_nocut[j] = kt
        lx, kt, _ = _integrate_profile(m, r2, r5, z, pp2, dp2, met, T_min=0.3)
        Lx_t03[j] = lx; kT_t03[j] = kt
        lx, kt, _ = _integrate_profile(m, r2, r5, z, pp_cal, dp_cal, met, T_min=0.3)
        Lx_cal[j] = lx; kT_cal[j] = kt

    # Literature
    m_lit  = np.logspace(12, 15.5, 100)
    Lx_lit = _lovisari20_lx(m_lit, z=z)
    kT_lit = _lovisari20_kt(m_lit, z=z)
    Lx_c25 = _comparat25_lx(m_lit)
    M_lo20, Lx_lo20, kT_lo20 = _load_lovisari20_data()
    M_bu18, Lx_bu18, kT_bu18 = _load_bulbul18()
    M_lo15, Lx_lo15, kT_lo15 = _load_lovisari15()
    M_zh24, Lx_zh24, Lx_zh24_err = _load_zhang24()
    M_po24, Lx_po24_hi, Lx_po24_lo = _load_popesso24()

    Lx_lo20_interp = _lovisari20_lx(m500_msun, z=z)
    ratio_nocut = np.log10(np.clip(Lx_nocut / Lx_lo20_interp, 1e-10, None))
    ratio_t03   = np.log10(np.clip(Lx_t03   / Lx_lo20_interp, 1e-10, None))
    ratio_cal   = np.log10(np.clip(Lx_cal   / Lx_lo20_interp, 1e-10, None))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── (0,0) Lx–M ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.fill_between(m_lit, Lx_lit / 10**0.2, Lx_lit * 10**0.2,
                    alpha=0.12, color="gray")
    ax.fill_between(M_po24, Lx_po24_lo, Lx_po24_hi,
                    alpha=0.15, color="purple", label="Popesso+2024")
    ax.loglog(m_lit, Lx_lit, "k--", lw=1.5, label="Lovisari+2020")
    ax.loglog(m_lit, Lx_c25, "r-.", lw=1.5, label="Comparat+2025")
    ax.scatter(M_lo20, Lx_lo20, s=10, color="gray",      alpha=0.7,
               marker="o", label="Lo+20", zorder=5)
    ax.scatter(M_bu18, Lx_bu18, s=12, color="steelblue", alpha=0.7,
               marker="s", label="Bu+18", zorder=5)
    ax.scatter(M_lo15, Lx_lo15, s=12, color="green",     alpha=0.7,
               marker="^", label="Lo+15 groups", zorder=5)
    ax.errorbar(M_zh24, Lx_zh24, yerr=Lx_zh24_err, fmt="D",
                color="darkorange", ms=5, lw=1.2, label="Zhang+24 CGM", zorder=6)
    ax.loglog(m500_msun, Lx_nocut, color="0.6", ls="-",  lw=1.2,
              label=r"DPM m.2 all gas ($\alpha_{L_X}\approx1.97$)")
    ax.loglog(m500_msun, Lx_t03,   color="0.6", ls="--", lw=1.2,
              label=r"DPM m.2, $T>0.3$ keV")
    ax.loglog(m500_msun, Lx_cal,   color="C3",  ls="-",  lw=2.5,
              label=rf"Calibrated ($\beta_n={BN_CAL:.2f}$, $\alpha_{{L_X}}={ALPHA_LX_CAL:.2f}$)",
              zorder=4)
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$L_X$ (0.5–2 keV) [erg s$^{-1}$]")
    ax.set_title(r"$L_X$–$M_{500c}$: DPM m.2 vs calibrated")
    ax.set_xlim(1e11, 3e15); ax.set_ylim(1e38, 1e47)
    ax.legend(fontsize=6.5, ncol=2); ax.grid(True, alpha=0.2)

    # ── (0,1) kT–M ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.fill_between(m_lit, kT_lit / 10**0.15, kT_lit * 10**0.15,
                    alpha=0.12, color="gray")
    ax.loglog(m_lit, kT_lit, "k--", lw=1.5, label="Lovisari+2020")
    ax.scatter(M_lo20, kT_lo20, s=10, color="gray",      alpha=0.7,
               marker="o", label="Lo+20", zorder=5)
    ax.scatter(M_bu18, kT_bu18, s=12, color="steelblue", alpha=0.7,
               marker="s", label="Bu+18", zorder=5)
    ax.scatter(M_lo15, kT_lo15, s=12, color="green",     alpha=0.7,
               marker="^", label="Lo+15 groups", zorder=5)
    ax.loglog(m500_msun, kT_nocut, color="0.6", ls="-",  lw=1.2,
              label=r"DPM m.2 all gas ($\alpha_{kT}\approx0.49$)")
    ax.loglog(m500_msun, kT_t03,   color="0.6", ls="--", lw=1.2,
              label=r"DPM m.2, $T>0.3$ keV")
    ax.loglog(m500_msun, kT_cal,   color="C3",  ls="-",  lw=2.5,
              label=rf"Calibrated ($\alpha_{{kT}}={BP_CAL-BN_CAL:.2f}$)",
              zorder=4)
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$kT_{\rm ew}$ [keV]")
    ax.set_title(r"$kT$–$M_{500c}$: DPM m.2 vs calibrated")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # ── (1,0) Analytical α_Lx vs β_n ──────────────────────────────────────
    ax = axes[1, 0]
    bn_arr   = np.linspace(0.0, 0.65, 200)
    bp_offset = ALPHA_KT_CAL              # beta_P - beta_n, fixed at fit_comparat2025 value
    intercept = 1.0 + 0.5 * bp_offset
    al_arr   = intercept + 2.0 * bn_arr   # = 1 + 1.5*beta_n + 0.5*(beta_n+bp_offset)
    ax.plot(bn_arr, al_arr, "C0-", lw=2,
            label=rf"$\alpha_{{L_X}} = {intercept:.2f} + 2\beta_n$  "
                  rf"($\beta_P=\beta_n+{bp_offset:.2f}$)")
    ax.axhline(ALPHA_LX_CAL, ls="--", color="k",   lw=1.5,
               label=rf"Target (fit) $\alpha_{{L_X}}={ALPHA_LX_CAL:.2f}$")
    ax.axhline(1.97, ls=":",  color="0.5", lw=1.2,
               label=r"DPM m.2: $\alpha_{L_X}=1.97$")
    ax.plot(0.36, 1.97, "ro", ms=9, zorder=5,
            label=r"DPM m.2 ($\beta_n=0.36$)")
    ax.plot(BN_CAL, ALPHA_LX_CAL, "g*", ms=14, zorder=5,
            label=rf"Calibrated ($\beta_n={BN_CAL:.2f}$)")
    ax2 = ax.twinx()
    ax2.plot(bn_arr, bn_arr + bp_offset, "C1--", lw=1.5,
             label=rf"$\alpha_{{kT}} = \beta_n + {bp_offset:.2f}$ (right axis)")
    ax2.axhline(bp_offset, ls="--", color="C1", alpha=0.4, lw=1)
    ax2.set_ylabel(r"$\alpha_{kT}$", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(0.0, 1.4)
    ax.set_xlabel(r"$\beta_n$ (density mass-slope)")
    ax.set_ylabel(r"$\alpha_{L_X}$")
    ax.set_title(rf"Analytical: $L_X \propto M^{{1+1.5\beta_n+0.5\beta_P}}$  "
                 rf"($\beta_P=\beta_n+{bp_offset:.2f}$)")
    ax.set_ylim(1.1, 2.7); ax.set_xlim(0.0, 0.65)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (1,1) Log-ratio ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.semilogx(m500_msun, ratio_nocut, color="0.6", ls="-",  lw=1.5,
                label="DPM m.2 all gas")
    ax.semilogx(m500_msun, ratio_t03,   color="0.6", ls="--", lw=1.5,
                label=r"DPM m.2, $T>0.3$ keV")
    ax.semilogx(m500_msun, ratio_cal,   color="C3",  ls="-",  lw=2.5,
                label=rf"Calibrated ($\beta_n={BN_CAL:.2f}$, $T>0.3$ keV)")
    ax.axhline(0, ls="--", color="k", lw=1.5, label="Perfect match")
    ax.axhspan(-0.5, 0.5, alpha=0.07, color="green", label="±0.5 dex")
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]")
    ax.set_ylabel(r"$\log_{10}(L_{X,{\rm model}} / L_{X,{\rm Lo20}})$")
    ax.set_title(r"Normalization ratio vs Lovisari+2020")
    ax.set_ylim(-3, 7); ax.legend(fontsize=7.5); ax.grid(True, alpha=0.2)

    fig.suptitle(
        rf"DPM X-ray calibration ($\beta_n,\beta_P$ = fit_comparat2025 S1 MAP): "
        rf"$\beta_n={BN_CAL:.3f}$, $\beta_P={BP_CAL:.3f}$, "
        rf"$n_{{e,0.3}}={ne_cal:.2e}$ cm$^{{-3}}$"
        "\n"
        rf"Slope fix: $\alpha_{{L_X}}: 1.97\to{ALPHA_LX_CAL:.2f}$;  "
        rf"$\alpha_{{kT}}: 0.49\to{ALPHA_KT_CAL:.2f}$",
        fontsize=11)
    fig.tight_layout()
    out = _FIG_DIR / "gas_06_xray_calibration.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Output directory: {_FIG_DIR}")
    fig_radial_profiles()
    fig_mass_scaling()
    fig_parameter_sensitivity()
    fig_scaling_relations()
    fig_pressure_unit_correction()
    fig_xray_calibration()
    print("Done.")


if __name__ == "__main__":
    main()
