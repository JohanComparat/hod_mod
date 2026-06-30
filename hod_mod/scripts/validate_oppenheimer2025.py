"""Validation figures for the DPM electron density profile.

Reproduces and verifies the key results from:

    Oppenheimer B.D. et al. 2025, arXiv:2505.14782
    "DPMhalo: parametric gas profiles for the diffuse gas around galaxies"

Figures produced
----------------
1. ``dpm_01_profiles_3models.pdf``
   Electron density n_e(r/R200) for the 3 DPM models at fixed mass and z=0.
2. ``dpm_02_mass_scaling.pdf``
   n_e(0.3 R200) vs M200 for all 3 models — verifies the beta mass scaling.
3. ``dpm_03_redshift_scaling.pdf``
   n_e(0.3 R200) vs z for all 3 models at M200=10^13 Msun/h
   — verifies the E(z)^gamma redshift scaling.
4. ``dpm_04_density_uk.pdf``
   Fourier transform ñ_e(k|M) for all 3 models at z=0.3.
5. ``dpm_05_emissivity_uk.pdf``
   Emissivity FT X̃(k|M) = FT[n_e²] for all 3 models.

Normalization check:
   n_e(r=0.3 R200 | M200=10^12 Msun/h, z=0) = ne_03 for each model.

Usage
-----
    cd /home/comparat/software/hod_mod
    python -m hod_mod.scripts.validate_oppenheimer2025
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.gas import GasDensityDPM
from hod_mod.gas import _RHO_CRIT0
from hod_mod.core.power_spectrum import LinearPowerSpectrum

_HERE    = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

_THETA  = LinearPowerSpectrum.default_cosmology()
_H      = float(_THETA["h"])
_OM     = float(_THETA["Omega_m"])
_COLORS = ["C0", "C1", "C2"]

_MODEL_LABELS = ["DPM Model 1", "DPM Model 2", "DPM Model 3"]

# Three reference masses
_MASSES   = np.array([1e12, 1e13, 1e14])
_R200_ARR = (_MASSES / (4 / 3 * np.pi * 200 * _RHO_CRIT0 * _OM)) ** (1 / 3)


def _r200(m, z=0.0):
    return (m / (4 / 3 * np.pi * 200 * _RHO_CRIT0 * _OM)) ** (1 / 3)


# ---------------------------------------------------------------------------
# Figure 1 — profile shapes for 3 models
# ---------------------------------------------------------------------------

def fig_profiles_3models():
    print("Figure 1: DPM n_e(r/R200) for 3 models ...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    m200   = 1e13
    r200   = _r200(m200)
    x_arr  = np.logspace(-1.8, 0.3, 200)
    r_arr  = x_arr * r200

    for idx, (model_id, ax) in enumerate(zip([1, 2, 3], axes)):
        dp = GasDensityDPM(model=model_id, n_gl=100)
        ne = dp.density_3d(r_arr, m200, r200, 0.0, _OM)
        ax.loglog(x_arr, ne, color=_COLORS[idx], lw=2)
        ax.axvline(0.3, ls="--", color="k", alpha=0.5, label="$r=0.3R_{200}$ (norm pt)")
        ax.set_xlabel(r"$r/R_{200}$")
        ax.set_title(f"DPM Model {model_id}\n"
                     rf"$n_{{e,03}}={dp._ne_03:.2e}$, $\beta={dp._beta}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$n_e(r)$ [cm$^{-3}$]")
    fig.suptitle(r"DPM electron density profiles (Oppenheimer+2025, arXiv:2505.14782)"
                 rf"   $M_{{200}}=10^{{13}}\,M_\odot/h$, $z=0$")
    fig.tight_layout()
    out = _FIG_DIR / "dpm_01_profiles_3models.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — mass scaling n_e(0.3R200) vs M200
# ---------------------------------------------------------------------------

def fig_mass_scaling():
    print("Figure 2: Mass scaling n_e(0.3 R200) vs M200 ...")
    m_arr  = np.logspace(11, 15, 30)
    r_arr  = _r200(m_arr)
    r_ref  = 0.3 * r_arr

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, model_id in enumerate([1, 2, 3]):
        dp = GasDensityDPM(model=model_id, n_gl=100)
        ne_ref = np.array([
            float(dp.density_3d(np.array([r_ref[i]]), m_arr[i], r_arr[i], 0.0, _OM)[0])
            for i in range(len(m_arr))
        ])
        ax.loglog(m_arr / 1e12, ne_ref, color=_COLORS[idx], lw=2,
                  label=f"Model {model_id} ($\\beta={dp._beta}$)")
        # Overlay power-law reference: n_e ∝ M^beta
        if dp._beta > 0:
            plaw = ne_ref[10] * (m_arr / m_arr[10]) ** dp._beta
            ax.loglog(m_arr / 1e12, plaw, color=_COLORS[idx], ls="--", lw=1, alpha=0.5)

    ax.set_xlabel(r"$M_{200}$ [$10^{12}\,M_\odot/h$]")
    ax.set_ylabel(r"$n_e(0.3\,R_{200})$ [cm$^{-3}$]")
    ax.set_title(r"DPM mass scaling $n_e \propto M_{200}^\beta$ at $z=0$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "dpm_02_mass_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — redshift scaling
# ---------------------------------------------------------------------------

def fig_redshift_scaling():
    print("Figure 3: Redshift scaling n_e(0.3 R200) vs z ...")
    z_arr = np.linspace(0.0, 1.5, 30)
    m200  = 1e13
    r200  = _r200(m200)
    r_ref = 0.3 * r200

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, model_id in enumerate([1, 2, 3]):
        dp = GasDensityDPM(model=model_id, n_gl=100)
        ne_z = np.array([
            float(dp.density_3d(np.array([r_ref]), m200, r200, float(z), _OM)[0])
            for z in z_arr
        ])
        ax.semilogy(z_arr, ne_z, color=_COLORS[idx], lw=2,
                    label=f"Model {model_id}")

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$n_e(0.3\,R_{200})$ [cm$^{-3}$]")
    ax.set_title(rf"DPM redshift scaling, $M_{{200}}=10^{{13}}\,M_\odot/h$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "dpm_03_redshift_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — density Fourier transform
# ---------------------------------------------------------------------------

def fig_density_uk():
    print("Figure 4: DPM density FT ñ_e(k|M) ...")
    k_arr = np.logspace(-2, 1.5, 60)
    z     = 0.3
    m_arr = np.array([1e12, 1e13, 1e14])
    r_arr = _r200(m_arr)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for idx, (model_id, ax) in enumerate(zip([1, 2, 3], axes)):
        dp = GasDensityDPM(model=model_id, n_gl=100)
        uk = dp.density_uk(k_arr, m_arr, r_arr, z, _THETA)   # (Nk, NM)
        m_labels = [r"$10^{12}$", r"$10^{13}$", r"$10^{14}$"]
        for j in range(3):
            ax.loglog(k_arr, uk[:, j], lw=1.8, color=f"C{j}",
                      label=rf"$M_{{200}}=10^{{{12+j}}}\,M_\odot/h$")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_title(f"Model {model_id}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$\tilde{n}_e(k|M)$ [$({\rm Mpc}/h)^3\,{\rm cm}^{-3}$]")
    fig.suptitle(r"DPM density FT $\tilde{n}_e(k|M)$, $z=0.3$")
    fig.tight_layout()
    out = _FIG_DIR / "dpm_04_density_uk.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 — emissivity Fourier transform
# ---------------------------------------------------------------------------

def fig_emissivity_uk():
    print("Figure 5: DPM emissivity FT X̃(k|M) = FT[n_e²] ...")
    k_arr = np.logspace(-2, 1.5, 60)
    z     = 0.3
    m_arr = np.array([1e12, 1e13, 1e14])
    r_arr = _r200(m_arr)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for idx, (model_id, ax) in enumerate(zip([1, 2, 3], axes)):
        dp = GasDensityDPM(model=model_id, n_gl=100)
        uk = dp.emissivity_uk(k_arr, m_arr, r_arr, z, _THETA)
        for j in range(3):
            ax.loglog(k_arr, uk[:, j], lw=1.8, color=f"C{j}",
                      label=rf"$M_{{200}}=10^{{{12+j}}}\,M_\odot/h$")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_title(f"Model {model_id}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$\tilde{\varepsilon}(k|M)$ [$({\rm Mpc}/h)^3\,{\rm cm}^{-6}$]")
    fig.suptitle(r"DPM emissivity FT $\tilde{\varepsilon}(k|M) = \widetilde{n_e^2}(k|M)$, $z=0.3$")
    fig.tight_layout()
    out = _FIG_DIR / "dpm_05_emissivity_uk.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Normalization check
# ---------------------------------------------------------------------------

def normalization_check():
    print("\n--- DPM normalization check ---")
    print("  n_e(r=0.3 R200 | M200=10^12 Msun/h, z=0) should equal ne_03:")
    m_ref   = 1e12
    r200_ref = _r200(m_ref)
    r_ref   = 0.3 * r200_ref

    all_ok = True
    for model_id in [1, 2, 3]:
        dp    = GasDensityDPM(model=model_id)
        ne    = float(dp.density_3d(np.array([r_ref]), m_ref, r200_ref, 0.0, _OM)[0])
        ref   = dp._ne_03
        rel   = abs(ne - ref) / ref
        ok    = rel < 1e-5
        print(f"  Model {model_id}: ne={ne:.4e}, ne_03={ref:.4e}, "
              f"rel_err={rel:.2e}  [{'OK' if ok else 'FAIL'}]")
        if not ok:
            all_ok = False
    print("All normalizations verified." if all_ok else "WARNING: normalization mismatch!")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Validating Oppenheimer+2025 DPM profile (arXiv:2505.14782) ===\n")
    ok = normalization_check()
    print()
    fig_profiles_3models()
    fig_mass_scaling()
    fig_redshift_scaling()
    fig_density_uk()
    fig_emissivity_uk()
    print(f"\nAll figures saved to {_FIG_DIR}/")
    if ok:
        print("PASS: DPM normalization verified for all 3 models.")
    else:
        print("FAIL: normalization mismatch — check GasDensityDPM implementation.")
