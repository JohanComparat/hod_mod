"""Model predictions for the Pandey+2025 lensing × tSZ benchmark.

Reference
---------
Pandey S. et al. 2025, arXiv:2506.07432
"Measurement of the lensing × thermal Sunyaev-Zel'dovich effect cross-correlation
with DES Year 3 and ACT DR6" (21-sigma detection)

Measurement
-----------
Angular cross-power spectrum C_ell^{gamma,y} between DES Year 3 weak-lensing
shear and ACT DR6 Compton-y maps over 0 < z < 2.

This script computes the equivalent P_{g,y}(k) and C_ell^{g,y} for a
DES-like lens sample using the halo model with the A10 pressure profile.

Note: The full lensing × tSZ signal C_ell^{kappa,y} requires folding in the
lensing efficiency kernel W_kappa(chi); this script models C_ell^{g,y} for the
galaxy-overdensity × tSZ component which has the same P(k) kernel.

Samples (approximate, from Pandey+2025 §2)
-------------------------------------------
- DES Y3 source galaxies: n(z) peaks at z~0.5-1.0
- Lens redshifts: 0.2 < z < 0.9

For the HOD we use a BOSS CMASS-like model at z~0.5.

Figures produced
----------------
1. ``pand25_01_cl_gy.pdf``
   C_ell^{g,y} via Limber approximation for a DES-like galaxy n(z).
2. ``pand25_02_cl_gy_decomposition.pdf``
   C_ell^{g,y} 1h + 2h decomposition.
3. ``pand25_03_pgy_vs_k.pdf``
   P_{g,y}(k) at three representative redshifts z=0.3, 0.5, 0.7.

Data
----
Data are not included in hod_mod.  The DES Y3 × ACT DR6 data vector is
available from the DES DR1/DR3 data release pages or the ACT DR6 GitHub:
    https://github.com/ACTCollaboration

Usage
-----
    cd /home/comparat/software/hod_mod
    python -m hod_mod.scripts.validate_pandey2025
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.cosmology import PressureProfileA10
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies.hod import MoreHODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.cross_spectra import HaloModelCrossSpectra

_HERE    = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Cosmology — Planck 2018
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
# HOD — BOSS CMASS-like for DES Y3 lens redshift range
# ---------------------------------------------------------------------------
_HOD_PARAMS = MoreHODModel.default_params()
_HOD_PARAMS.update({
    "log10mmin": 13.03,
    "sigma_logm": 0.38,
    "log10m1":    13.80,
    "alpha":      1.17,
    "kappa":      0.51,
})

# ---------------------------------------------------------------------------
# Galaxy n(z) kernel: DES Y3-like, double Gaussian
# ---------------------------------------------------------------------------
_Z_ARR  = np.linspace(0.1, 1.2, 40)
_NZ_G   = (np.exp(-0.5 * ((_Z_ARR - 0.40) / 0.08)**2)
           + 0.5 * np.exp(-0.5 * ((_Z_ARR - 0.70) / 0.10)**2))
_NZ_G  /= np.trapezoid(_NZ_G, _Z_ARR)

# Multipole array — Pandey+2025 uses ell ~ 100-5000
_ELL = np.logspace(2, 3.7, 24)


def _build_cross():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    fhmp   = FullHaloModelPrediction(pk_lin, hod, hp)
    pp     = PressureProfileA10(r_max_over_r500c=5.0, n_gl=200)
    return HaloModelCrossSpectra(fhmp, pressure_profile=pp)


# ---------------------------------------------------------------------------
# Figure 1 — C_ell^{g,y}
# ---------------------------------------------------------------------------

def fig_cl_gy(cross):
    print("Figure 1: C_ell^{g,y} via Limber ...")
    cl_gy = cross.angular_cl_gy(_ELL, _Z_ARR, _NZ_G, _THETA, _HOD_PARAMS)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(_ELL, np.abs(cl_gy) * _ELL * (_ELL + 1) / (2 * np.pi),
              "C0-", lw=2, label=r"$\ell(\ell+1)C_\ell^{g,y}/(2\pi)$")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{g,y}/(2\pi)$ [$({\rm Mpc}/h)^2$]")
    ax.set_title(
        "Pandey+2025 (arXiv:2506.07432) model\n"
        r"DES Y3-like $n(z)$ $\times$ A10 pressure, More+2015 HOD"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.05,
            "Data: ACT DR6 GitHub / DES DR3 data release",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="gray")
    fig.tight_layout()
    out = _FIG_DIR / "pand25_01_cl_gy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — 1h + 2h decomposition at z=0.5
# ---------------------------------------------------------------------------

def fig_cl_gy_decomposition(cross):
    print("Figure 2: C_ell^{g,y} 1h/2h decomposition at z=0.5 ...")
    z      = 0.5
    z_arr  = np.linspace(0.4, 0.6, 12)
    nz_g   = np.exp(-0.5 * ((z_arr - z) / 0.04)**2)

    tables = cross._pk_tables_gy(z, _THETA, _HOD_PARAMS)
    k_tab  = tables["log_k"]
    lpgy   = tables["log_pgy"]
    lpgy1h = tables["log_pgy_1h"]
    lpgy2h = tables["log_pgy_2h"]

    from hod_mod.cosmology.distances import comoving_distance
    h      = float(_THETA["h"])
    om     = float(_THETA["Omega_m"])
    chi_arr = np.array([
        float(np.asarray(comoving_distance(float(zi), h, om)).ravel()[0]) * h
        for zi in z_arr
    ])
    dndchi = nz_g / np.trapezoid(nz_g, chi_arr)

    def _limber_cl(log_k_tab, log_p_tab):
        cl = np.zeros(len(_ELL))
        for i, l in enumerate(_ELL):
            k_lim = (l + 0.5) / chi_arr
            p_lim = np.exp(np.interp(
                np.log(np.maximum(k_lim, 1e-4)),
                log_k_tab, log_p_tab,
            ))
            cl[i] = np.trapezoid(dndchi * p_lim / chi_arr**2, chi_arr)
        return cl

    cl_tot  = _limber_cl(k_tab, lpgy)
    cl_1h   = _limber_cl(k_tab, lpgy1h)
    cl_2h   = _limber_cl(k_tab, lpgy2h)

    ell_fac = _ELL * (_ELL + 1) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(_ELL, np.abs(cl_tot) * ell_fac, "k-",  lw=2, label="total")
    ax.loglog(_ELL, np.abs(cl_1h)  * ell_fac, "C0--", lw=1.5, label="1-halo")
    ax.loglog(_ELL, np.abs(cl_2h)  * ell_fac, "C1:",  lw=1.5, label="2-halo")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{g,y}/(2\pi)$")
    ax.set_title(rf"$C_\ell^{{g,y}}$ 1h/2h, narrow $n(z)$ at $z={z}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "pand25_02_cl_gy_decomposition.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — P_{g,y}(k) at 3 redshifts
# ---------------------------------------------------------------------------

def fig_pgy_vs_z(cross):
    print("Figure 3: P_{g,y}(k) at z=0.3, 0.5, 0.7 ...")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ["C0", "C1", "C2"]
    for z, col in zip([0.3, 0.5, 0.7], colors):
        tables = cross._pk_tables_gy(z, _THETA, _HOD_PARAMS)
        k   = np.exp(np.asarray(tables["log_k"]))
        pgy = np.exp(np.asarray(tables["log_pgy"]))
        ax.loglog(k, pgy, color=col, lw=1.8, label=rf"$z={z}$")

    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$P_{g,y}(k)$ [$({\rm Mpc}/h)^2$]")
    ax.set_title(
        r"$P_{g,y}(k)$ at DES Y3 lens redshifts" "\n"
        "A10 pressure × More+2015 HOD"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-2, 20)
    fig.tight_layout()
    out = _FIG_DIR / "pand25_03_pgy_vs_k.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Model predictions for Pandey+2025 (arXiv:2506.07432) ===")
    print("    DES Y3 × ACT DR6 lensing × tSZ\n")
    print("Building HaloModelCrossSpectra ...")
    cross = _build_cross()
    print()
    fig_cl_gy(cross)
    fig_cl_gy_decomposition(cross)
    fig_pgy_vs_z(cross)
    print(f"\nAll figures saved to {_FIG_DIR}/")
    print(
        "\nNOTE: Measured C_ell^{gamma,y} data are not included in hod_mod.\n"
        "      Obtain the DES Y3 × ACT DR6 data vector from the ACT DR6 data release\n"
        "      or the DES DR3 supplementary materials."
    )
