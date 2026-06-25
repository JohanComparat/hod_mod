"""Direct model prediction for BGS LS10 S1 (M* > 10^10 Msun) galaxy × X-ray.

Uses:
  - GAS.py-calibrated DPM model-2  (ne_03 = 1.26e-5 cm⁻³, beta_n = 0.20)
  - ZuMandelbaum15HODModel iHOD with log10m_star_thresh = 10.0, alpha_sat from MAP
  - HAM AGN model  (Comparat+2019)

Pipeline steps and diagnostic figures produced
----------------------------------------------
Fig 1  HOD occupation        N_c(M), N_s(M), dn/dM × N_tot(M), b_eff integrand
Fig 2  Gas density profile   ne(r, M) radial profiles + ne0 vs M power law
Fig 3  Emissivity FT         X̃(k, M) at z_eff — k-dependence and mass scaling
Fig 4  Halo integrands       which masses dominate 1h cen, 1h sat, 2h at z_eff
Fig 5  P_gX(k)               3D power spectrum at z_eff: 1h cen/sat, 2h, AGN
Fig 6  C_ell                 angular power spectrum: raw and PSF-convolved
Fig 7  w_theta(theta)        final observable, all components vs data

Usage::

    python -m hod_mod.scripts.direct_prediction_gal_gas_agn

Output: results/fits/comparat2025/direct_prediction_S1_fig*.pdf
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import trapezoid
from scipy.special import j0

from astropy.io import fits

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.cosmology import GasDensityDPM, ApecCoolingTable
from hod_mod.cosmology.gas_profiles import PressureProfileDPM
from hod_mod.galaxies.hod import ZuMandelbaum15HODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.cross_spectra import HaloModelCrossSpectra
from hod_mod.galaxies.agn_ham import HamAGNModel
from hod_mod.cosmology.distances import comoving_distance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ZENODO_DIR  = Path.home() / "data/zenodo/LSDR10_GALxEVT/Measurements_Xcorr_Stacks/XCORR"
_RESULTS_DIR = Path(__file__).parents[2] / "results" / "fits" / "comparat2025"

# ---------------------------------------------------------------------------
# Cosmology (Planck 2018)
# ---------------------------------------------------------------------------
_THETA_COSMO = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "H0":     _THETA_COSMO["h"] * 100.0,
    "Om0":    _THETA_COSMO["Omega_m"],
    "Ob0":    _THETA_COSMO["Omega_b"],
    "ns":     _THETA_COSMO["n_s"],
    "sigma8": 0.811,
}
_H = float(_THETA_COSMO["h"])
_APEC_TABLE = ApecCoolingTable(emin=0.5, emax=2.0, n_T=40, n_Z=10)

# ---------------------------------------------------------------------------
# Calibrated DPM parameters (validate_gas_profiles._calibrate_ne03_P03)
# beta_n=0.20, beta_P=0.80 → alpha_Lx = 1.70, alpha_kT = 0.60 (GAS.py targets)
# ---------------------------------------------------------------------------
_NE_03_CAL  = 1.260e-5   # cm⁻³   density amplitude at r=0.3R200
_P_03_CAL   = 1.627e-6   # keV/cm³ pressure amplitude at r=0.3R200
_BETA_N_CAL = 0.20       # density mass-scaling slope  n_e ∝ M^beta_n
_BETA_P_CAL = 0.80       # pressure mass-scaling slope  P ∝ M^beta_P

# ---------------------------------------------------------------------------
# S1 parameters
# ---------------------------------------------------------------------------
_S1_ZMEAN = 0.135
_S1_ZMAX  = 0.18
_S1_N     = 2_759_238

# Angular ell grid
_N_ELL = 160
_ELL   = np.logspace(1.0, 5.0, _N_ELL)   # ell_max = 100,000

# PSF: King profile (eROSITA TM CalDB on-axis)
_PSF_THETA_C = 8.64   # arcsec


# ---------------------------------------------------------------------------
# Helper: clone GasDensityDPM with overridden parameters
# ---------------------------------------------------------------------------

def _make_density_variant(model: int = 2, ne_03: float | None = None,
                          beta: float | None = None) -> GasDensityDPM:
    dp = GasDensityDPM(model=model)
    if ne_03 is not None:
        dp._ne_03 = float(ne_03)
    if beta is not None:
        dp._beta = float(beta)
    x_ref      = 0.3 * dp._C_DPM
    dp._f_xref = dp._gnfw_f(x_ref)
    dp._ne0    = dp._ne_03 / dp._f_xref
    return dp


def _make_pressure_variant(model: int = 2, P_03: float | None = None,
                           beta: float | None = None) -> PressureProfileDPM:
    from hod_mod.cosmology.gas_profiles import _gnfw_f_params
    pp = PressureProfileDPM(model=model)
    if P_03 is not None:
        pp._P_03 = float(P_03)
    if beta is not None:
        pp._beta = float(beta)
    x_ref = 0.3 * pp._C_DPM
    f_ref = _gnfw_f_params(x_ref, pp._alpha_in, pp._alpha_tr, pp._alpha_out_12)
    pp._P0 = pp._P_03 / float(f_ref)
    return pp


# ---------------------------------------------------------------------------
# n(z): narrow Gaussian centred on z_mean
# ---------------------------------------------------------------------------

def _build_nz(z_mean: float, z_max: float, n_pts: int = 7) -> tuple[np.ndarray, np.ndarray]:
    dz    = min(0.02, z_max * 0.10)
    z_arr = np.linspace(max(0.01, z_mean - 2.0 * dz), z_mean + 2.0 * dz, n_pts)
    nz    = np.exp(-0.5 * ((z_arr - z_mean) / dz) ** 2)
    return z_arr, nz / trapezoid(nz, z_arr)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _load_data() -> dict | None:
    N_str   = f"{_S1_N:07d}"
    matches = sorted(_ZENODO_DIR.glob(f"*_N_{N_str}_GALxEVT_wtheta.fits"))
    if not matches:
        print(f"  [data] S1 w_theta not found in {_ZENODO_DIR}")
        return None
    d = fits.open(matches[0])[1].data
    return dict(
        theta_arcsec = np.array(d["theta"],      dtype=float) * 3600.0,
        theta_rad    = np.array(d["theta"],      dtype=float) * np.pi / 180.0,
        wtheta       = np.array(d["wtheta"],     dtype=float),
        wtheta_err   = np.array(d["wtheta_err"], dtype=float),
    )


# ---------------------------------------------------------------------------
# PSF window in ell space (King profile, numerical FT)
# ---------------------------------------------------------------------------

def _psf_king_window(ell: np.ndarray, theta_c_arcsec: float = _PSF_THETA_C,
                     alpha: float = 1.5) -> np.ndarray:
    th_max = 800.0 * np.pi / (180.0 * 3600.0)   # rad
    th    = np.linspace(0.0, th_max, 3000)
    th_as = th * 180.0 * 3600.0 / np.pi
    psf   = (1.0 + (th_as / theta_c_arcsec) ** 2) ** (-alpha)
    B = np.array([
        2.0 * np.pi * trapezoid(psf * j0(ell_i * th) * th, th)
        for ell_i in ell
    ])
    return np.clip(B / B[0], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Core: per-z 3D power spectrum with 1h cen/sat decomposition
# ---------------------------------------------------------------------------

def _pk_decomposed(cross: HaloModelCrossSpectra, z: float,
                   hod_params: dict) -> dict[str, np.ndarray]:
    """P_{g,X}(k) at redshift z, decomposed into 1h cen, 1h sat, 2h, AGN."""
    sc = cross._get_static_cache(z, _THETA_COSMO, hod_params)
    nc_np, ns_np, n_gal, b_eff = cross._get_hod_weights(z, _THETA_COSMO, hod_params, sc)

    m_np   = sc["m_np"]
    dndm   = sc["dndm_np"]
    bias   = sc["bias_np"]
    pk_lin = sc["pk_lin"]
    uk     = sc["uk"]
    k_np   = sc["k_np"]

    X_uk   = cross._density_uk_cached(z, _THETA_COSMO, sc, emissivity=True)  # (Nk, NM)

    m_j    = jnp.asarray(m_np)
    X_j    = jnp.asarray(X_uk)
    nc_j   = jnp.asarray(nc_np)
    ns_j   = jnp.asarray(ns_np)
    dndm_j = jnp.asarray(dndm)
    bias_j = jnp.asarray(bias)
    pk_j   = jnp.asarray(pk_lin)
    uk_j   = jnp.asarray(uk)

    # 1-halo: centrals (no DM profile convolution — centrals sit at halo centre)
    P_cen = jnp.trapezoid(dndm_j[None, :] * nc_j[None, :] * X_j, m_j, axis=1) / n_gal
    # 1-halo: satellites (convolved with NFW profile u(k,M))
    P_sat = jnp.trapezoid(dndm_j[None, :] * ns_j[None, :] * uk_j * X_j, m_j, axis=1) / n_gal
    P_1h  = P_cen + P_sat

    # 2-halo
    I_X   = jnp.trapezoid(dndm_j[None, :] * bias_j[None, :] * X_j, m_j, axis=1)
    P_2h  = b_eff * pk_j * I_X
    P_gas = P_1h + P_2h

    # AGN (HAM model)
    if cross._agn is not None:
        X_agn_j  = jnp.asarray(cross._agn.agn_emissivity_uk(k_np, m_np, z, _THETA_COSMO))
        f_sat_a  = float(cross._agn._f_sat_agn)
        gw_agn   = nc_j[None, :] + f_sat_a * ns_j[None, :] * uk_j
        P_agn_1h = jnp.trapezoid(dndm_j[None, :] * gw_agn * X_agn_j, m_j, axis=1) / n_gal
        I_agn    = jnp.trapezoid(dndm_j[None, :] * bias_j[None, :] * X_agn_j, m_j, axis=1)
        P_agn_r  = P_agn_1h + b_eff * pk_j * I_agn
        mpc_cm_h  = 3.0857e24 / _H
        lambda_ref = float(_APEC_TABLE(np.array([1.0]), np.array([0.3]))[0])
        P_agn     = P_agn_r * (1e43 / (lambda_ref * mpc_cm_h ** 3))
    else:
        P_agn = jnp.zeros_like(P_gas)

    P_total = P_gas + P_agn

    return dict(
        k=k_np,
        nc=nc_np, ns=ns_np, n_gal=n_gal, b_eff=b_eff,
        m=m_np, dndm=dndm, bias=bias, uk=np.asarray(uk),
        X_uk=np.asarray(X_uk),
        P_cen=np.asarray(P_cen), P_sat=np.asarray(P_sat),
        P_1h=np.asarray(P_1h),   P_2h=np.asarray(P_2h),
        P_gas=np.asarray(P_gas), P_agn=np.asarray(P_agn),
        P_total=np.asarray(P_total),
    )


# ---------------------------------------------------------------------------
# Limber integration → C_ell for all components
# ---------------------------------------------------------------------------

def _limber_all(cross: HaloModelCrossSpectra, z_arr: np.ndarray,
                nz_g: np.ndarray, hod_params: dict) -> tuple[dict, list]:
    h, omega_m = float(_THETA_COSMO["h"]), float(_THETA_COSMO["Omega_m"])
    chi_z  = np.array([
        float(np.asarray(comoving_distance(float(zi), h, omega_m)).ravel()[0]) * h
        for zi in z_arr
    ])
    dndchi = nz_g / trapezoid(nz_g, chi_z)

    tables = [_pk_decomposed(cross, zi, hod_params) for zi in z_arr]
    log_k  = np.log(tables[0]["k"])
    keys   = ["P_cen", "P_sat", "P_1h", "P_2h", "P_gas", "P_agn", "P_total"]

    def _cl(key):
        logP = np.array([np.log(np.maximum(t[key], 1e-60)) for t in tables])
        cl   = np.zeros(len(_ELL))
        for i, ell in enumerate(_ELL):
            k_lim    = (ell + 0.5) / chi_z
            logk_lim = np.log(np.maximum(k_lim, 1e-4))
            P_at_k   = np.array([np.exp(np.interp(logk_lim[j], log_k, logP[j]))
                                  for j in range(len(z_arr))])
            cl[i]    = trapezoid(dndchi * P_at_k / chi_z ** 2, chi_z)
        return cl

    cl_dict = {}
    for key in keys:
        print(f"    Limber {key} ...", flush=True)
        cl_dict[key] = _cl(key)
    return cl_dict, tables


# ---------------------------------------------------------------------------
# Hankel transform C_ell → w_theta
# ---------------------------------------------------------------------------

def _hankel(cl: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    return np.array([
        trapezoid(_ELL * cl * j0(_ELL * th) / (2.0 * np.pi), _ELL)
        for th in theta_rad
    ])


# ===========================================================================
# Diagnostic figure functions
# ===========================================================================

def _fig_hod(m_np, nc, ns, dndm, bias, n_gal, b_eff, z, out_path):
    """Fig 1: HOD occupation and HMF integrands."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    log10m = np.log10(m_np)

    # panel 0: N_c, N_s, N_tot
    ax = axes[0]
    ax.semilogy(log10m, nc,      color="C0", lw=2, label=r"$N_c(M)$")
    ax.semilogy(log10m, ns,      color="C1", lw=2, label=r"$N_s(M)$")
    ax.semilogy(log10m, nc + ns, color="k",  lw=1.5, ls="--", label=r"$N_{tot}$")
    ax.set_xlabel(r"$\log_{10}(M_h/[M_\odot/h])$")
    ax.set_ylabel("Mean occupation")
    ax.set_title(f"HOD occupation  ($z={z:.3f}$)")
    ax.legend(fontsize=9)
    ax.set_ylim(1e-4, None)

    # panel 1: dn/dM × N(M) — which halos dominate n_gal
    ax = axes[1]
    nt = nc + ns
    integrand = dndm * nt
    ax.fill_between(log10m, integrand, alpha=0.3, color="C0")
    ax.semilogy(log10m, dndm * nc, color="C0", lw=1.5, label=r"$\frac{dn}{dM}N_c$")
    ax.semilogy(log10m, dndm * ns, color="C1", lw=1.5, label=r"$\frac{dn}{dM}N_s$")
    ax.semilogy(log10m, integrand,  color="k",  lw=1, ls="--",
                label=rf"total (n_gal={n_gal:.2e} $h^3$/Mpc$^3$)")
    ax.set_xlabel(r"$\log_{10}(M_h/[M_\odot/h])$")
    ax.set_ylabel(r"$\frac{dn}{dM}\,N(M)$  [$h^4$ Mpc$^{-3}$ $M_\odot^{-1}$]")
    ax.set_title("n_gal integrand")
    ax.legend(fontsize=8)

    # panel 2: dn/dM × b(M) × N(M) — b_eff integrand
    ax = axes[2]
    ax.semilogy(log10m, dndm * bias * nt / n_gal, color="C2", lw=2,
                label=rf"$b_{{eff}}={b_eff:.2f}$")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(r"$\log_{10}(M_h/[M_\odot/h])$")
    ax.set_ylabel(r"$\frac{dn}{dM}\,b(M)\,N_{tot}/\bar{n}$")
    ax.set_title("Effective bias integrand")
    ax.legend(fontsize=9)

    fig.suptitle("Fig 1 — HOD occupation  (ZuMandelbaum15 iHOD, $M_*>10^{10}\\,M_\\odot$)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def _fig_density_profile(dp_cal, dp_default, z, theta_cosmo, out_path):
    """Fig 2: DPM gas density profiles ne(r, M) and ne_0 vs M."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    masses_msun = [1e12, 1e13, 1e14]   # Msun/h
    colors      = ["C0", "C1", "C2"]
    r_frac      = np.logspace(-2, 0.5, 200)   # r/r200
    omega_m     = float(theta_cosmo["Omega_m"])

    rho_crit0 = 2.775e11 * _H**2   # Msun/h / (Mpc/h)^3
    ez2 = omega_m * (1 + z)**3 + (1 - omega_m)
    rho_crit_z = rho_crit0 * ez2
    r200_arr   = ((3 * np.array(masses_msun) / (4 * np.pi * 200 * rho_crit_z)) ** (1/3))

    # panel 0: ne(r, M) calibrated vs default
    ax = axes[0]
    for M_msun, r200, color in zip(masses_msun, r200_arr, colors):
        r_phys = r_frac * r200      # Mpc/h
        ne_cal = dp_cal.density_3d(r_phys, M_msun, r200, z, omega_m)
        ne_def = dp_default.density_3d(r_phys, M_msun, r200, z, omega_m)
        label = rf"$M={M_msun:.0e}\,M_\odot/h$"
        ax.loglog(r_frac, ne_cal, color=color, lw=2,   label=label + " cal")
        ax.loglog(r_frac, ne_def, color=color, lw=1.2, ls="--", alpha=0.6)

    ax.axvline(0.3, color="grey", ls=":", lw=0.8, label=r"$r=0.3\,R_{200}$")
    ax.set_xlabel(r"$r/R_{200}$")
    ax.set_ylabel(r"$n_e(r)$ [cm$^{-3}$]")
    ax.set_title("DPM ne profile  (solid=cal, dashed=default)")
    ax.legend(fontsize=7.5)

    # panel 1: ne at r=0.3*R200 vs M
    ax = axes[1]
    m_arr  = np.logspace(11, 15, 100)
    r200_m = ((3 * m_arr / (4 * np.pi * 200 * rho_crit_z)) ** (1/3))
    r_ref_m = 0.3 * r200_m   # Mpc/h

    ne0_cal = dp_cal.density_3d(r_ref_m, m_arr, r200_m, z, omega_m)
    ne0_def = dp_default.density_3d(r_ref_m, m_arr, r200_m, z, omega_m)
    ax.loglog(m_arr, ne0_cal, color="C0", lw=2,   label=rf"Calibrated ($\beta_n={_BETA_N_CAL}$)")
    ax.loglog(m_arr, ne0_def, color="C1", lw=1.5, ls="--", label="Default (β=0.36)")
    m12_ref = np.array([1e12, 1e14, 1e15])
    ax.loglog(m12_ref, 1.260e-5 * (m12_ref / 1e12)**_BETA_N_CAL,
              "k:", lw=0.8, label=rf"$\propto M^{{{_BETA_N_CAL}}}$")
    ax.set_xlabel(r"$M_{200c}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$n_e(0.3\,R_{200})$ [cm$^{-3}$]")
    ax.set_title(r"$n_e$ at $r=0.3\,R_{200}$ vs halo mass")
    ax.legend(fontsize=8)

    # panel 2: emissivity = ne^2 at r=0.3R200 vs M
    ax = axes[2]
    ax.loglog(m_arr, ne0_cal**2, color="C0", lw=2,
              label=rf"Calibrated ($\propto M^{{{2*_BETA_N_CAL:.2f}}}$)")
    ax.loglog(m_arr, ne0_def**2, color="C1", lw=1.5, ls="--",
              label="Default (β=0.36, $\\propto M^{0.72}$)")
    ax.set_xlabel(r"$M_{200c}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$n_e^2(0.3\,R_{200})$ [cm$^{-6}$]")
    ax.set_title(r"Local emissivity $n_e^2$ vs mass  (∝ M$^{2\beta_n}$)")
    ax.legend(fontsize=8)

    fig.suptitle(rf"Fig 2 — DPM gas density profile  ($z={z}$, cal: $n_{{e,0.3}}={_NE_03_CAL:.2e}$, $\beta_n={_BETA_N_CAL}$)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def _fig_emissivity_ft(pk_z, out_path):
    """Fig 3: Emissivity FT X̃(k, M) at z_eff."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    k   = pk_z["k"]
    m   = pk_z["m"]
    Xuk = pk_z["X_uk"]   # (Nk, NM)

    # panel 0: X̃(k) at fixed masses
    ax = axes[0]
    for idx_m, color in [(0, "C3"), (len(m)//4, "C1"), (len(m)//2, "C0"),
                          (3*len(m)//4, "C2"), (-1, "k")]:
        ax.loglog(k, Xuk[:, idx_m], color=color,
                  label=rf"$M={m[idx_m]:.1e}\,M_\odot/h$")
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$\tilde{X}(k|M)$ [(Mpc/$h$)$^3$ cm$^{-6}$]")
    ax.set_title(r"Emissivity FT $\tilde{X}(k|M)$ at fixed masses")
    ax.legend(fontsize=8)

    # panel 1: X̃(k=k_1h, M) vs M — shows mass scaling
    ax = axes[1]
    k_ref_idx = np.argmin(np.abs(k - 1.0))   # k ≈ 1 h/Mpc
    ax.loglog(m, Xuk[k_ref_idx, :], color="C0", lw=2,
              label=rf"$k\approx{k[k_ref_idx]:.2f}\,h$/Mpc")
    k_ref_idx2 = np.argmin(np.abs(k - 0.1))
    ax.loglog(m, Xuk[k_ref_idx2, :], color="C1", lw=2,
              label=rf"$k\approx{k[k_ref_idx2]:.3f}\,h$/Mpc")
    # overlay power law
    m12_ref = np.logspace(11, 15, 50)
    # X ∝ M^(1+2*beta_n) * R200^3 ∝ M^(1+2*beta_n+1) = M^(2.4)
    ax.loglog(m12_ref,
              Xuk[k_ref_idx, len(m)//2] * (m12_ref / m[len(m)//2]) ** (1 + 2 * _BETA_N_CAL),
              "k:", lw=0.8, label=rf"$\propto M^{{1+2\beta_n}}={1+2*_BETA_N_CAL:.2f}$")
    ax.set_xlabel(r"$M_{200c}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$\tilde{X}(k|M)$")
    ax.set_title(r"Mass scaling of $\tilde{X}$ at fixed $k$")
    ax.legend(fontsize=8)

    fig.suptitle(r"Fig 3 — Emissivity Fourier transform $\tilde{X}(k|M)$ at $z_{\rm eff}$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def _fig_integrands(pk_z, out_path):
    """Fig 4: Halo model integrands — which masses dominate each term."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    m    = pk_z["m"]
    dndm = pk_z["dndm"]
    bias = pk_z["bias"]
    nc   = pk_z["nc"]
    ns   = pk_z["ns"]
    uk   = pk_z["uk"]     # (Nk, NM)
    Xuk  = pk_z["X_uk"]  # (Nk, NM)
    k    = pk_z["k"]
    n_gal= pk_z["n_gal"]
    b_eff= pk_z["b_eff"]
    log10m = np.log10(m)

    k_idx = np.argmin(np.abs(k - 1.0))   # k ≈ 1 h/Mpc

    # panel 0: 1h cen integrand
    ax = axes[0]
    I_cen  = dndm * nc        * Xuk[k_idx, :] / n_gal
    I_sat  = dndm * ns * uk[k_idx, :] * Xuk[k_idx, :] / n_gal
    ax.semilogy(log10m, np.abs(I_cen), color="C2", lw=2, label="1h cen")
    ax.semilogy(log10m, np.abs(I_sat), color="C3", lw=2, label="1h sat")
    ax.semilogy(log10m, np.abs(I_cen + I_sat), color="k", lw=1.2, ls="--",
                label="1h total")
    ax.set_xlabel(r"$\log_{10}(M_h)$")
    ax.set_ylabel(r"$|\mathrm{d}P_{1h}/\mathrm{d}\ln M|$")
    ax.set_title(rf"1-halo integrands  ($k={k[k_idx]:.1f}\,h$/Mpc)")
    ax.legend(fontsize=9)

    # panel 1: 2h integrand
    ax = axes[1]
    I_2h = dndm * bias * Xuk[k_idx, :]
    ax.semilogy(log10m, np.abs(I_2h), color="C0", lw=2, label="2h gas")
    ax.fill_between(log10m, np.abs(I_2h), alpha=0.2, color="C0")
    ax.set_xlabel(r"$\log_{10}(M_h)$")
    ax.set_ylabel(r"$\frac{dn}{dM}b(M)\tilde{X}$")
    ax.set_title(r"2-halo integrand $I_X(k)$")
    ax.legend(fontsize=9)

    # panel 2: cumulative mass contribution to P_gX (1h cen)
    ax = axes[2]
    cum_cen = np.cumsum(np.abs(I_cen)) / (np.sum(np.abs(I_cen)) + 1e-60)
    cum_sat = np.cumsum(np.abs(I_sat)) / (np.sum(np.abs(I_sat)) + 1e-60)
    ax.plot(log10m, cum_cen, color="C2", lw=2, label="1h cen cumulative")
    ax.plot(log10m, cum_sat, color="C3", lw=2, label="1h sat cumulative")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlabel(r"$\log_{10}(M_h)$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Cumulative contribution (1h)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    fig.suptitle(rf"Fig 4 — Halo model integrands at $k\approx1\,h$/Mpc, $z={_S1_ZMEAN}$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def _fig_pgX(pk_z, out_path):
    """Fig 5: P_gX(k) at z_eff with all components."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    k = pk_z["k"]

    styles = dict(
        P_total=(dict(color="k",  lw=2.2, ls="-",  label="Total"),),
        P_gas  =(dict(color="C0", lw=1.8, ls="-",  label="Gas (1h+2h)"),),
        P_1h   =(dict(color="C0", lw=1.2, ls="--", label="Gas 1h"),),
        P_cen  =(dict(color="C2", lw=1.0, ls="-.", label="Gas 1h cen"),),
        P_sat  =(dict(color="C3", lw=1.0, ls=":",  label="Gas 1h sat"),),
        P_2h   =(dict(color="C0", lw=1.2, ls=":",  alpha=0.7, label="Gas 2h"),),
        P_agn  =(dict(color="C1", lw=1.5, ls="-",  label="AGN (HAM)"),),
    )

    for key, (sty,) in styles.items():
        P = pk_z[key]
        pos = P > 0
        if pos.any():
            axes[0].loglog(k[pos], P[pos], **sty)

    axes[0].set_xlabel(r"$k$ [$h$/Mpc]")
    axes[0].set_ylabel(r"$P_{g,X}(k)$")
    axes[0].set_title(rf"$P_{{g,X}}(k)$ at $z={_S1_ZMEAN}$")
    axes[0].legend(fontsize=8)

    # right panel: ratio to total
    P_tot = pk_z["P_total"]
    pos_t = P_tot > 0
    for key, (sty,) in styles.items():
        if key == "P_total":
            continue
        P   = pk_z[key]
        rat = np.where(pos_t, P / P_tot, np.nan)
        axes[1].semilogx(k, rat, **sty)
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel(r"$k$ [$h$/Mpc]")
    axes[1].set_ylabel("Fraction of total P_{g,X}")
    axes[1].set_title("Component fractions")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-0.1, 1.5)

    fig.suptitle(rf"Fig 5 — 3D power spectrum $P_{{g,X}}(k)$  (DPM cal, $z={_S1_ZMEAN}$)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def _fig_cl(cl_dict, B_ell, out_path):
    """Fig 6: Angular power spectra C_ell before and after PSF."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    styles = {
        "P_total": dict(color="k",  lw=2.2, ls="-",  label="Total"),
        "P_gas":   dict(color="C0", lw=1.8, ls="-",  label="Gas"),
        "P_1h":    dict(color="C0", lw=1.2, ls="--", label="Gas 1h"),
        "P_2h":    dict(color="C0", lw=1.2, ls=":",  alpha=0.7, label="Gas 2h"),
        "P_agn":   dict(color="C1", lw=1.5, ls="-",  label="AGN"),
    }

    ax = axes[0]
    for key, sty in styles.items():
        if key not in cl_dict:
            continue
        cl = cl_dict[key]
        pos = cl > 0
        if pos.any():
            ax.loglog(_ELL[pos], cl[pos], **sty)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{g,X}$")
    ax.set_title("Before PSF")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.loglog(_ELL, B_ell, color="C4", lw=1.5, ls="--", label=rf"PSF King $\theta_c={_PSF_THETA_C}^{{''}}$")
    for key, sty in styles.items():
        if key not in cl_dict:
            continue
        cl_psf = cl_dict[key] * B_ell
        pos = cl_psf > 0
        if pos.any():
            ax2.loglog(_ELL[pos], cl_psf[pos], **sty)
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$C_\ell^{g,X} \times B_\ell$")
    ax2.set_title("After PSF convolution")
    ax2.legend(fontsize=8)

    fig.suptitle(r"Fig 6 — Angular cross-power $C_\ell^{g,X}$  (Limber)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


def _fig_wtheta(wt_dict, theta_as, data, out_path):
    """Fig 7: w_theta decomposed + data comparison."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    styles = {
        "P_total": dict(color="k",  lw=2.2, ls="-",  label="Total (gas + AGN)"),
        "P_gas":   dict(color="C0", lw=1.8, ls="-",  label=rf"Gas, $\beta_n={_BETA_N_CAL}$"),
        "P_1h":    dict(color="C0", lw=1.2, ls="--", label="Gas 1h"),
        "P_cen":   dict(color="C2", lw=1.0, ls="-.", label=r"Gas 1h cen ($G_c\times X$)"),
        "P_sat":   dict(color="C3", lw=1.0, ls=":",  label=r"Gas 1h sat ($G_s\times X$)"),
        "P_2h":    dict(color="C0", lw=1.2, ls=":",  alpha=0.7, label="Gas 2h"),
        "P_agn":   dict(color="C1", lw=1.5, ls="-",  label="AGN (HAM)"),
    }

    ax = axes[0]
    for key, sty in styles.items():
        if key not in wt_dict:
            continue
        wt  = wt_dict[key]
        pos = wt > 0
        if pos.any():
            ax.plot(theta_as[pos], wt[pos], **sty)

    if data is not None:
        ax.errorbar(data["theta_arcsec"], data["wtheta"], yerr=data["wtheta_err"],
                    fmt="o", ms=4, color="k", zorder=5, label="S1 data")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ [arcsec]")
    ax.set_ylabel(r"$w_\theta(\theta)$")
    ax.set_title(
        r"S1 BGS LS10  ($M_*>10^{10}\,M_\odot$,  $\bar{z}=0.135$)"
        "\n"
        rf"DPM cal: $n_{{e,0.3}}={_NE_03_CAL:.2e}$, $\beta_n={_BETA_N_CAL}$"
    )
    ax.legend(fontsize=7.5, loc="upper right")
    ax.axvline(_PSF_THETA_C, ls=":", color="grey", lw=0.8, alpha=0.5,
               label=rf"PSF $\theta_c={_PSF_THETA_C}^{{''}}$")

    # right panel: ratio to data (if available)
    ax2 = axes[1]
    if data is not None:
        theta_data = data["theta_arcsec"]
        wobs       = data["wtheta"]
        werr       = data["wtheta_err"]
        for key, sty in styles.items():
            if key not in wt_dict:
                continue
            wmod = np.interp(theta_data, theta_as, wt_dict[key])
            ratio = np.where(np.abs(wobs) > 1e-12, wmod / wobs, np.nan)
            ax2.semilogx(theta_data, ratio, **sty)
        ax2.fill_between(theta_data,
                         1 - werr / np.abs(wobs),
                         1 + werr / np.abs(wobs),
                         alpha=0.15, color="k", label="Data ±1σ")
        ax2.axhline(1.0, color="k", lw=1.0, ls="--")
        ax2.set_ylim(-0.5, 5.0)
        ax2.set_ylabel("model / data")
    else:
        total = wt_dict["P_total"]
        pos_t = total > 0
        for key, sty in styles.items():
            if key == "P_total" or key not in wt_dict:
                continue
            rat = np.where(pos_t, wt_dict[key] / total, np.nan)
            ax2.semilogx(theta_as, rat, **sty)
        ax2.axhline(1.0, color="k", lw=0.8, ls="--")
        ax2.set_ylim(-0.1, 1.5)
        ax2.set_ylabel("fraction of total")
    ax2.set_xlabel(r"$\theta$ [arcsec]")
    ax2.set_title("Model / Data ratio")
    ax2.legend(fontsize=7.5)

    fig.suptitle(r"Fig 7 — $w_\theta(\theta)$ prediction vs data",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}", flush=True)


# ===========================================================================
# Main
# ===========================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _RESULTS_DIR / "direct_prediction_S1"

    # ------------------------------------------------------------------
    # Step 1: Build infrastructure
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Building infrastructure (CAMB + HMF + HAM) ...", flush=True)
    t0 = time.time()
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp   = FullHaloModelPrediction(pk_lin, hod, hp)
    agn    = HamAGNModel(pk_lin=pk_lin)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # HOD parameter set
    # ZuMandelbaum15HODModel reads SHMR keys + alpha_sat; NOT log10mmin etc.
    # ------------------------------------------------------------------
    hod_params = ZuMandelbaum15HODModel.default_params()
    hod_params["log10m_star_thresh"] = 10.0    # S1: M* > 10^10 Msun
    hod_params["alpha_sat"]          = 1.184   # only effective HOD DOF from MAP

    # ------------------------------------------------------------------
    # Step 2: Calibrated DPM profiles (joint ne_03 + P_03 calibration)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Step 2: Calibrated DPM  ne_03={_NE_03_CAL:.2e}  P_03={_P_03_CAL:.2e}  "
          f"beta_n={_BETA_N_CAL}  beta_P={_BETA_P_CAL}", flush=True)
    dp_cal     = _make_density_variant(model=2, ne_03=_NE_03_CAL, beta=_BETA_N_CAL)
    pp_cal     = _make_pressure_variant(model=2, P_03=_P_03_CAL,  beta=_BETA_P_CAL)
    dp_default = _make_density_variant(model=2)   # for comparison in Fig 2
    cross      = HaloModelCrossSpectra(fhmp, density_profile=dp_cal,
                                       pressure_profile=pp_cal, agn_model=agn)

    # ------------------------------------------------------------------
    # Step 3: n(z) for S1
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Build n(z) ...", flush=True)
    z_arr, nz_g = _build_nz(_S1_ZMEAN, _S1_ZMAX, n_pts=7)
    print(f"  z_arr = {np.round(z_arr, 3)}", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Compute P_gX at z_eff (for diagnostic figures 1–5)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Step 4: P_gX at z_eff = {_S1_ZMEAN} ...", flush=True)
    pk_z = _pk_decomposed(cross, _S1_ZMEAN, hod_params)
    print(f"  n_gal = {pk_z['n_gal']:.3e}  b_eff = {pk_z['b_eff']:.3f}", flush=True)

    # Fig 1 — HOD occupation
    _fig_hod(pk_z["m"], pk_z["nc"], pk_z["ns"], pk_z["dndm"], pk_z["bias"],
             pk_z["n_gal"], pk_z["b_eff"], _S1_ZMEAN,
             f"{prefix}_fig1_hod.pdf")

    # Fig 2 — gas density profile
    _fig_density_profile(dp_cal, dp_default, _S1_ZMEAN, _THETA_COSMO,
                         f"{prefix}_fig2_density.pdf")

    # Fig 3 — emissivity FT
    _fig_emissivity_ft(pk_z, f"{prefix}_fig3_emissivity_ft.pdf")

    # Fig 4 — halo integrands
    _fig_integrands(pk_z, f"{prefix}_fig4_integrands.pdf")

    # Fig 5 — P_gX(k)
    _fig_pgX(pk_z, f"{prefix}_fig5_pgX.pdf")

    # ------------------------------------------------------------------
    # Step 5: angular_cl_gX (JAX-vectorized Limber, ~10s vs 213s)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 5: angular_cl_gX (JAX-vectorized Limber) ...", flush=True)
    t0 = time.time()
    cl_components = cross.angular_cl_gX(
        _ELL, z_arr, nz_g, _THETA_COSMO, hod_params,
        return_components=True,   # {"gas", "agn"}
        n_workers=1,              # serial: XLA memory allocator is not thread-safe
    )
    cl_gas_raw   = np.asarray(cl_components["gas"])
    cl_agn_raw   = np.asarray(cl_components["agn"])
    cl_total_raw = cl_gas_raw + cl_agn_raw
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # PSF window (for Fig 6 and Hankel transform)
    print("  computing King PSF window B_ell ...", flush=True)
    B_ell = _psf_king_window(_ELL, _PSF_THETA_C)

    # Build C_ell dict (raw, pre-PSF) for Fig 6
    cl_dict = {
        "P_gas":   cl_gas_raw,
        "P_agn":   cl_agn_raw,
        "P_total": cl_total_raw,
    }

    # ------------------------------------------------------------------
    # Step 6: PSF convolution + Hankel transform → w_theta
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 6: Hankel transform C_ell × B_ell → w_theta ...", flush=True)

    data      = _load_data()
    theta_rad = (data["theta_rad"] if data is not None
                 else np.logspace(np.log10(5.0), np.log10(400.0), 50)
                      * np.pi / (180.0 * 3600.0))
    theta_as  = theta_rad * 180.0 * 3600.0 / np.pi

    wt_dict = {key: _hankel(cl_dict[key] * B_ell, theta_rad) for key in cl_dict}

    # Fig 6 — C_ell
    _fig_cl(cl_dict, B_ell, f"{prefix}_fig6_cl.pdf")

    # Fig 7 — w_theta
    _fig_wtheta(wt_dict, theta_as, data, f"{prefix}_fig7_wtheta.pdf")

    # ------------------------------------------------------------------
    # Step 7: Summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Summary at theta = 30 arcsec:")
    idx = np.argmin(np.abs(theta_as - 30.0))
    total30 = wt_dict["P_total"][idx]
    for key in ["P_gas", "P_agn", "P_total"]:
        v = wt_dict[key][idx]
        frac = v / total30 if total30 > 0 else float("nan")
        print(f"  {key:12s}  {v:.3e}   ({frac*100:.1f}%)")

    if data is not None:
        obs30 = np.interp(30.0, data["theta_arcsec"], data["wtheta"])
        print(f"  {'data':12s}  {obs30:.3e}   (model/data = {total30/obs30:.2f})")

    print("\nAll figures written to:", _RESULTS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
