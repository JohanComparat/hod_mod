"""Validation figures for galaxy × tSZ and galaxy × soft X-ray cross-correlations.

Produces seven panels:

1. A10 pressure profile P_e(r/R₅₀₀) for 3 halo masses at z=0.3
2. Pressure profile FT ỹ(k|M) vs k for same 3 masses
3. P_{g,y}(k) decomposition: 1h + 2h + total
4. P_{g,X}(k) decomposition: 1h + 2h + total  (DPM Model 2)
5. Projected Σ_y(r_p)
6. Angular C_ℓ^{g,y} vs ℓ
7. Projected w_{g,X}(r_p)

References
----------
Arnaud+2010 : arXiv:0910.1234 (A10 pressure profile)
Oppenheimer+2025 : arXiv:2505.14782 (DPM density profile)
Amodeo+2021 : arXiv:2009.05557 (galaxy × tSZ stacking benchmark)
Comparat+2025 : arXiv:2503.19796 (galaxy × soft X-ray benchmark)

Usage
-----
    cd /home/comparat/software/hod_mod
    python -m hod_mod.scripts.validate_sz_xray
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.cosmology import PressureProfileA10, GasDensityDPM
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.cosmology.gas_profiles import _RHO_CRIT0
from hod_mod.galaxies.hod import MoreHODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.cross_spectra import HaloModelCrossSpectra

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Common cosmology / HOD setup
# ---------------------------------------------------------------------------
_THETA = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "flat":   True,
    "H0":     _THETA["h"] * 100.0,
    "Om0":    _THETA["Omega_m"],
    "Ob0":    _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns":     _THETA["n_s"],
}
_Z          = 0.3
_HOD_PARAMS = MoreHODModel.default_params()
_H          = float(_THETA["h"])
_OM         = float(_THETA["Omega_m"])

_MASS_LABELS = [r"$10^{13}\,M_\odot/h$", r"$10^{14}\,M_\odot/h$", r"$10^{15}\,M_\odot/h$"]
_MASS_ARR    = np.array([1e13, 1e14, 1e15])
_R200_ARR    = (_MASS_ARR / (4.0 / 3.0 * np.pi * 200.0 * _RHO_CRIT0 * _OM))**(1.0 / 3.0)
_C200_ARR    = np.array([8.0, 5.0, 3.5])
_COLORS      = ["C0", "C1", "C2"]


def _build_prediction():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    fhmp   = FullHaloModelPrediction(pk_lin, hod, hp)
    pp     = PressureProfileA10(r_max_over_r500c=5.0, n_gl=150)
    dp     = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=150)
    cross  = HaloModelCrossSpectra(fhmp, pressure_profile=pp, density_profile=dp)
    return fhmp, cross


# ---------------------------------------------------------------------------
# Figure 1: A10 pressure profile P_e(r/R₅₀₀)
# ---------------------------------------------------------------------------

def fig_pressure_profile():
    print("Figure 1: Arnaud+2010 pressure profile ...")
    pp   = PressureProfileA10()
    x    = np.logspace(-1.5, 0.8, 200)   # r / R₅₀₀

    ez2        = _OM * (1.0 + _Z)**3 + (1.0 - _OM)
    rho_crit_z = _RHO_CRIT0 * ez2 / (1.0 + _Z)**3

    from hod_mod.cosmology.gas_profiles import m200_to_m500c
    m500_arr, r500_arr = m200_to_m500c(_MASS_ARR, _C200_ARR, _R200_ARR, rho_crit_z)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (m200, r200, m500, r500) in enumerate(zip(_MASS_ARR, _R200_ARR, m500_arr, r500_arr)):
        pe = pp._p3d(x, float(m500), _Z, _H, _OM)
        ax.loglog(x, pe, color=_COLORS[i], label=_MASS_LABELS[i])

    ax.set_xlabel(r"$r / R_{500c}$")
    ax.set_ylabel(r"$P_e$ [keV cm$^{-3}$]")
    ax.set_title("Arnaud+2010 pressure profile (arXiv:0910.1234 Table 1)")
    ax.legend()
    ax.set_xlim(0.03, 6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "sz_01_pressure_profile.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Pressure FT ỹ(k|M)
# ---------------------------------------------------------------------------

def fig_pressure_uk():
    print("Figure 2: Pressure profile Fourier transform ỹ(k|M) ...")
    pp   = PressureProfileA10(r_max_over_r500c=5.0, n_gl=150)
    k    = np.logspace(-2, 1, 80)

    ez2        = _OM * (1.0 + _Z)**3 + (1.0 - _OM)
    rho_crit_z = _RHO_CRIT0 * ez2 / (1.0 + _Z)**3
    from hod_mod.cosmology.gas_profiles import m200_to_m500c
    m500_arr, r500_arr = m200_to_m500c(_MASS_ARR, _C200_ARR, _R200_ARR, rho_crit_z)

    uk = pp.pressure_uk(k, _MASS_ARR, _R200_ARR, _C200_ARR, _Z, _THETA)   # (Nk, NM)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(3):
        ax.loglog(k, uk[:, i], color=_COLORS[i], label=_MASS_LABELS[i])

    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$\tilde{y}(k|M)$ [$({\rm Mpc}/h)^2$]")
    ax.set_title(r"A10 pressure profile FT $\tilde{y}(k|M,z=0.3)$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "sz_02_pressure_uk.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: P_{g,y}(k) decomposition
# ---------------------------------------------------------------------------

def fig_pgy_decomposition(cross):
    print("Figure 3: P_{g,y}(k) decomposition ...")
    tables = cross._pk_tables_gy(_Z, _THETA, _HOD_PARAMS)
    k      = np.exp(np.array(tables["log_k"]))
    pgy    = np.exp(np.array(tables["log_pgy"]))
    pgy_1h = np.exp(np.array(tables["log_pgy_1h"]))
    pgy_2h = np.exp(np.array(tables["log_pgy_2h"]))
    pmy    = np.exp(np.array(tables["log_pmy"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(k, pgy,    "k-",  lw=2,  label=r"$P_{g,y}$ total")
    ax.loglog(k, pgy_1h, "b--", lw=1.5, label=r"$P_{g,y}^{\rm 1h}$")
    ax.loglog(k, pgy_2h, "r:",  lw=1.5, label=r"$P_{g,y}^{\rm 2h}$")
    ax.loglog(k, pmy,    "g-.", lw=1.5, label=r"$P_{m,y}$ total")

    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$P(k)$ [$({\rm Mpc}/h)^2$]")
    ax.set_title(rf"Galaxy $\times$ tSZ power spectrum ($z={_Z}$, A10 pressure)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-2, 20)
    fig.tight_layout()
    out = _FIG_DIR / "sz_03_pgy_decomposition.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: P_{g,X}(k) decomposition
# ---------------------------------------------------------------------------

def fig_pgX_decomposition(cross):
    print("Figure 4: P_{g,X}(k) decomposition ...")
    tables = cross._pk_tables_gX(_Z, _THETA, _HOD_PARAMS)
    k      = np.exp(np.array(tables["log_k"]))
    pgX    = np.exp(np.array(tables["log_pgX"]))
    pgX_1h = np.exp(np.array(tables["log_pgX_1h"]))
    pgX_2h = np.exp(np.array(tables["log_pgX_2h"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(k, pgX,    "k-",  lw=2,  label=r"$P_{g,X}$ total")
    ax.loglog(k, pgX_1h, "b--", lw=1.5, label=r"$P_{g,X}^{\rm 1h}$")
    ax.loglog(k, pgX_2h, "r:",  lw=1.5, label=r"$P_{g,X}^{\rm 2h}$")

    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$P(k)$ [$({\rm Mpc}/h)^3\,{\rm cm}^{-6}$]")
    ax.set_title(rf"Galaxy $\times$ X-ray power spectrum ($z={_Z}$, DPM Model 2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-2, 20)
    fig.tight_layout()
    out = _FIG_DIR / "sz_04_pgX_decomposition.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Projected Σ_y(r_p)
# ---------------------------------------------------------------------------

def fig_projected_gy(cross):
    print("Figure 5: Projected Σ_y(r_p) ...")
    rp     = np.logspace(-1, 1.5, 24)
    sigma_y = cross.projected_gy(rp, _Z, _THETA, _HOD_PARAMS)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(rp, np.maximum(sigma_y, 1e-30), "k-", lw=2, label=r"$\Sigma_y(r_p)$")
    ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax.set_ylabel(r"$\Sigma_y(r_p)$ [dimensionless Compton-$y$]")
    ax.set_title(rf"Projected galaxy $\times$ tSZ ($z={_Z}$, A10 pressure)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "sz_05_projected_gy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: Angular C_ℓ^{g,y}
# ---------------------------------------------------------------------------

def fig_angular_cl_gy(cross):
    print("Figure 6: Angular C_ℓ^{g,y} ...")
    ell   = np.logspace(1, 4, 20)
    z_arr = np.linspace(0.20, 0.50, 12)
    nz_g  = np.exp(-0.5 * ((z_arr - 0.30) / 0.05)**2)
    cl_gy = cross.angular_cl_gy(ell, z_arr, nz_g, _THETA, _HOD_PARAMS)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ell, np.abs(cl_gy), "k-", lw=2)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{g,y}$ [$({\rm Mpc}/h)^2$]")
    ax.set_title(r"Angular $C_\ell^{g,y}$ (Limber, $z\sim 0.3$ CMASS-like)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "sz_06_cl_gy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 7: Projected w_{g,X}(r_p)
# ---------------------------------------------------------------------------

def fig_projected_gX(cross):
    print("Figure 7: Projected w_{g,X}(r_p) ...")
    rp  = np.logspace(-1, 1.5, 24)
    wgX = cross.projected_gX(rp, _Z, _THETA, _HOD_PARAMS)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(rp, np.maximum(wgX, 1e-50), "k-", lw=2, label=r"$w_{g,X}(r_p)$")
    ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax.set_ylabel(r"$w_{g,X}(r_p)$ [(Mpc/$h$) cm$^{-6}$]")
    ax.set_title(rf"Projected galaxy $\times$ X-ray ($z={_Z}$, DPM Model 2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "sz_07_projected_gX.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building FullHaloModelPrediction + HaloModelCrossSpectra ...")
    fhmp, cross = _build_prediction()

    fig_pressure_profile()
    fig_pressure_uk()
    fig_pgy_decomposition(cross)
    fig_pgX_decomposition(cross)
    fig_projected_gy(cross)
    fig_angular_cl_gy(cross)
    fig_projected_gX(cross)

    print(f"\nAll figures saved to {_FIG_DIR}/")
