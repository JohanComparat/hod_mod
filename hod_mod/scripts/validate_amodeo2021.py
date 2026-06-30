"""Model predictions for the Amodeo+2021 galaxy × tSZ stacking benchmark.

Reference
---------
Amodeo S. et al. 2021, Phys. Rev. D 103, 063514
arXiv:2009.05557
ACT × BOSS: Thermal and Kinematic Sunyaev-Zel'dovich effect

Measurement
-----------
Stacked tSZ Compton-y and kSZ temperature profiles around BOSS CMASS and LOWZ
galaxies from ACT DR4 maps.  This script reproduces the **tSZ** model; kSZ
requires peculiar velocity fields and is not currently implemented in hod_mod.

The observable is the stacked Compton-y profile as a function of projected
separation r_p [Mpc/h]:

    Sigma_y(r_p) = (sigma_T / m_e c^2) * 2 * integral P_e(sqrt(r_p^2 + pi^2)) d_pi

computed via the halo model P_{g,y}(k) cross-spectrum and Abel projection.

Data
----
Data points are NOT included in hod_mod — this script shows the model prediction
for the CMASS and LOWZ HOD and labels the expected amplitude and scale.

To reproduce Amodeo+2021 Fig. 4, obtain the data from:
    https://github.com/EmmanuelSchaan/ThumbStack
  or digitize the published figure.

Samples
-------
- CMASS: 0.43 < z < 0.70, z_eff ≈ 0.55, ~750 000 galaxies
  HOD: More+2015 with log10Mmin≈13.03, sigma_logm≈0.38, log10M1≈13.80
- LOWZ: 0.16 < z < 0.36, z_eff ≈ 0.27, ~200 000 galaxies
  HOD: similar parametrization at lower mass

Figures produced
----------------
1. ``amo21_01_sigma_y_cmass.pdf``   Sigma_y(r_p) for CMASS HOD at z=0.55
2. ``amo21_02_sigma_y_lowz.pdf``    Sigma_y(r_p) for LOWZ HOD at z=0.27
3. ``amo21_03_pgy_decomposition.pdf`` P_{g,y}(k) 1h+2h decomposition

Usage
-----
    cd $HOD_MOD_REPO
    python -m hod_mod.scripts.validate_amodeo2021
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.gas import PressureProfileA10
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra

_HERE    = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Cosmology — Planck 2018 (used in Amodeo+2021 analysis)
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

# ---------------------------------------------------------------------------
# BOSS CMASS and LOWZ HOD parameters
# Taken from More+2015 Table 2 (arXiv:1407.1856), fiducial cosmology row
# ---------------------------------------------------------------------------
_CMASS_Z   = 0.55
_CMASS_HOD = MoreHODModel.default_params()
_CMASS_HOD.update({
    "log10mmin": 13.03,
    "sigma_logm": 0.38,
    "log10m1":    13.80,
    "alpha":      1.17,
    "kappa":      0.51,
})

_LOWZ_Z    = 0.27
_LOWZ_HOD  = MoreHODModel.default_params()
_LOWZ_HOD.update({
    "log10mmin": 13.20,
    "sigma_logm": 0.40,
    "log10m1":    14.00,
    "alpha":      1.15,
    "kappa":      0.50,
})

# Projected separation grid: Amodeo+2021 uses theta * D_A ≈ 0.1-30 Mpc/h
_RP_ARR = np.logspace(-1.5, 1.5, 30)   # [Mpc/h]


def _build_cross(pp=None):
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    fhmp   = FullHaloModelPrediction(pk_lin, hod, hp)
    pp     = PressureProfileA10(r_max_over_r500c=5.0, n_gl=200) if pp is None else pp
    return HaloModelCrossSpectra(fhmp, pressure_profile=pp), fhmp


# ---------------------------------------------------------------------------
# Figure 1 — Sigma_y(r_p) for CMASS
# ---------------------------------------------------------------------------

def fig_sigma_y_cmass(cross):
    print("Figure 1: Sigma_y(r_p) for BOSS CMASS ...")
    sigma_y = cross.projected_gy(_RP_ARR, _CMASS_Z, _THETA, _CMASS_HOD)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(_RP_ARR, np.maximum(sigma_y, 1e-30), "C0-", lw=2,
              label=r"$\Sigma_y(r_p)$ — CMASS HOD (More+2015)")

    ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax.set_ylabel(r"$\Sigma_y(r_p)$ [dimensionless Compton-$y$]")
    ax.set_title(
        "Amodeo+2021 (arXiv:2009.05557) — model tSZ stack, CMASS\n"
        rf"$z_{{eff}}={_CMASS_Z}$, A10 pressure, More+2015 HOD"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate expected data range from Amodeo+2021
    ax.axvspan(0.1, 5.0, alpha=0.06, color="gray",
               label="Amodeo+2021 measurement range (~0.1-5 Mpc/h)")
    ax.text(0.95, 0.95,
            "Data: github.com/EmmanuelSchaan/ThumbStack",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="gray")
    fig.tight_layout()
    out = _FIG_DIR / "amo21_01_sigma_y_cmass.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Sigma_y(r_p) for LOWZ
# ---------------------------------------------------------------------------

def fig_sigma_y_lowz(cross):
    print("Figure 2: Sigma_y(r_p) for BOSS LOWZ ...")
    sigma_y = cross.projected_gy(_RP_ARR, _LOWZ_Z, _THETA, _LOWZ_HOD)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(_RP_ARR, np.maximum(sigma_y, 1e-30), "C1-", lw=2,
              label=r"$\Sigma_y(r_p)$ — LOWZ HOD")
    ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
    ax.set_ylabel(r"$\Sigma_y(r_p)$ [dimensionless Compton-$y$]")
    ax.set_title(
        "Amodeo+2021 (arXiv:2009.05557) — model tSZ stack, LOWZ\n"
        rf"$z_{{eff}}={_LOWZ_Z}$, A10 pressure, More+2015-like HOD"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "amo21_02_sigma_y_lowz.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — P_{g,y}(k) decomposition
# ---------------------------------------------------------------------------

def fig_pgy_decomposition(cross):
    print("Figure 3: P_{g,y}(k) 1h+2h decomposition ...")
    tables = cross._pk_tables_gy(_CMASS_Z, _THETA, _CMASS_HOD)
    k      = np.exp(np.array(tables["log_k"]))
    pgy    = np.exp(np.array(tables["log_pgy"]))
    pgy_1h = np.exp(np.array(tables["log_pgy_1h"]))
    pgy_2h = np.exp(np.array(tables["log_pgy_2h"]))
    pmy    = np.exp(np.array(tables["log_pmy"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(k, pgy,    "k-",  lw=2,   label=r"$P_{g,y}$ total")
    ax.loglog(k, pgy_1h, "C0--", lw=1.5, label=r"$P_{g,y}^{\rm 1h}$")
    ax.loglog(k, pgy_2h, "C1:",  lw=1.5, label=r"$P_{g,y}^{\rm 2h}$")
    ax.loglog(k, pmy,    "C2-.", lw=1.5, label=r"$P_{m,y}$ total")
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$P(k)$ [$({\rm Mpc}/h)^2$]")
    ax.set_title(
        rf"$P_{{g,y}}(k)$: CMASS HOD $\times$ A10 pressure, $z={_CMASS_Z}$"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "amo21_03_pgy_decomposition.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def print_model_summary(cross):
    tables = cross._pk_tables_gy(_CMASS_Z, _THETA, _CMASS_HOD)
    print(f"\n--- CMASS halo model summary ---")
    print(f"  n_gal = {tables['n_gal']:.4e}  (Mpc/h)^-3")
    print(f"  b_eff = {tables['b_eff']:.4f}")
    tables_l = cross._pk_tables_gy(_LOWZ_Z, _THETA, _LOWZ_HOD)
    print(f"\n--- LOWZ halo model summary ---")
    print(f"  n_gal = {tables_l['n_gal']:.4e}  (Mpc/h)^-3")
    print(f"  b_eff = {tables_l['b_eff']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Model predictions for Amodeo+2021 (arXiv:2009.05557) ===")
    print("    ACT DR4 × BOSS CMASS/LOWZ stacked tSZ\n")
    print("Building HaloModelCrossSpectra ...")
    cross, _ = _build_cross()
    print_model_summary(cross)
    print()
    fig_sigma_y_cmass(cross)
    fig_sigma_y_lowz(cross)
    fig_pgy_decomposition(cross)
    print(f"\nAll figures saved to {_FIG_DIR}/")
    print(
        "\nNOTE: Data (Amodeo+2021 Fig. 4) is not included in hod_mod.\n"
        "      Obtain from: https://github.com/EmmanuelSchaan/ThumbStack\n"
        "      Then load and overplot in the figures above."
    )
