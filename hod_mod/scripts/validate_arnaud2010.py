"""Validation figures for the Arnaud+2010 universal pressure profile.

Reproduces and verifies the key results from:

    Arnaud M., Pratt G.W., Piffaretti R. et al. 2010, A&A 517, A92
    arXiv:0910.1234

Figures produced
----------------
1. ``a10_01_profile_shape.pdf``
   Dimensionless shape function p(x) vs x = r/R500c (compare to A10 Fig. 5).
2. ``a10_02_pressure_profile.pdf``
   Physical pressure P_e(r/R500c) [keV/cm³] for 3 masses at z=0 and z=0.5
   (compare to A10 Fig. 9).
3. ``a10_03_mass_scaling.pdf``
   Total SZ signal Y_SZ ∝ M500c^{5/3+α_p} self-similar scaling check.
4. ``a10_04_pressure_uk.pdf``
   Fourier transform ỹ(k|M) vs k for 3 masses at z=0.3
   (decreasing from flat plateau to k^{-2} tail).

Parameter check (A10 Table 1):
   P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905, alpha_p=0.12

Usage
-----
    cd /home/comparat/software/hod_mod
    python -m hod_mod.scripts.validate_arnaud2010
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.cosmology import PressureProfileA10
from hod_mod.cosmology.gas_profiles import _RHO_CRIT0, m200_to_m500c
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum

_HERE    = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

_THETA  = LinearPowerSpectrum.default_cosmology()
_H      = float(_THETA["h"])
_OM     = float(_THETA["Omega_m"])
_COLORS = ["C0", "C1", "C2"]

# A10 Table 1 reference parameters
_A10_TABLE1 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510,
                   beta=5.4905, alpha_p=0.12)

# Three reference masses in Msun/h
_MASSES     = np.array([1e13, 1e14, 1e15])
_MLABELS    = [r"$10^{13}\,M_\odot/h$", r"$10^{14}\,M_\odot/h$", r"$10^{15}\,M_\odot/h$"]


def _rho_crit_z(z):
    ez2 = _OM * (1 + z)**3 + (1 - _OM)
    return _RHO_CRIT0 * ez2 / (1 + z)**3


def _m200_r200_c200_m500_r500(masses, z):
    r200 = (masses / (4 / 3 * np.pi * 200 * _RHO_CRIT0 * _OM)) ** (1 / 3)
    c200 = np.array([8.0, 5.0, 3.5])
    m500, r500 = m200_to_m500c(masses, c200, r200, _rho_crit_z(z))
    return r200, c200, m500, r500


# ---------------------------------------------------------------------------
# Figure 1 — dimensionless shape p(x)
# ---------------------------------------------------------------------------

def fig_shape():
    print("Figure 1: A10 dimensionless shape p(x) ...")
    P0     = _A10_TABLE1["P0"]
    c500   = _A10_TABLE1["c500"]
    gamma  = _A10_TABLE1["gamma"]
    alpha  = _A10_TABLE1["alpha"]
    beta   = _A10_TABLE1["beta"]
    x      = np.logspace(-1.5, 1.0, 300)
    cx     = c500 * x
    p      = P0 / (cx**gamma * (1 + cx**alpha)**((beta - gamma) / alpha))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(x, p, "k-", lw=2, label=rf"A10 Eq. 11 ($P_0={P0}$, $c_{{500}}={c500}$)")
    ax.axvline(1.0, ls="--", color="gray", alpha=0.5, label=r"$x=1$ ($r=R_{500c}$)")
    ax.set_xlabel(r"$x = r/R_{500c}$")
    ax.set_ylabel(r"$p(x)$ [dimensionless shape]")
    ax.set_title("Arnaud+2010 universal pressure shape (arXiv:0910.1234, Table 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.03, 10)
    fig.tight_layout()
    out = _FIG_DIR / "a10_01_profile_shape.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — physical pressure P_e(r/R500)
# ---------------------------------------------------------------------------

def fig_pressure_profile():
    print("Figure 2: Physical pressure P_e(r/R500c) ...")
    pp    = PressureProfileA10(r_max_over_r500c=5.0, n_gl=150)
    x_arr = np.logspace(-1.5, 0.8, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for iz, (z, ax) in enumerate([(0.0, axes[0]), (0.5, axes[1])]):
        _, c200, m500, r500 = _m200_r200_c200_m500_r500(_MASSES, z)
        for i, (m200, m5, c) in enumerate(zip(_MASSES, m500, c200)):
            pe = pp._p3d(x_arr, float(m5), z, _H, _OM)
            ax.loglog(x_arr, pe, color=_COLORS[i], lw=1.8, label=_MLABELS[i])
        ax.set_xlabel(r"$r/R_{500c}$")
        ax.set_title(rf"$z={z}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.03, 6)

    axes[0].set_ylabel(r"$P_e$ [keV cm$^{-3}$]")
    fig.suptitle("Arnaud+2010 physical pressure profile (arXiv:0910.1234, Eq. 11)")
    fig.tight_layout()
    out = _FIG_DIR / "a10_02_pressure_profile.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — mass scaling Y_SZ ∝ M^{5/3+α_p}
# ---------------------------------------------------------------------------

def fig_mass_scaling():
    print("Figure 3: Self-similar mass scaling Y_SZ vs M500c ...")
    pp      = PressureProfileA10(r_max_over_r500c=5.0, n_gl=150)
    z       = 0.0
    n_mass  = 20
    m200_arr = np.logspace(12.5, 15.5, n_mass)
    r200_arr = (m200_arr / (4 / 3 * np.pi * 200 * _RHO_CRIT0 * _OM)) ** (1 / 3)
    c200_arr = np.full(n_mass, 5.0)
    m500_arr, r500_arr = m200_to_m500c(m200_arr, c200_arr, r200_arr, _rho_crit_z(z))

    # Compute Y_SZ = (sigma_T/m_e c^2) * 4pi * int P_e r^2 dr (dimensionless * volume)
    # Here we use ỹ(k→0) as a proxy for the volume integral of y
    _SIGMA_T_OVER_ME_C2 = 6.6524e-25 / 511.0   # cm²/keV
    _MPC_CM = 3.0857e24
    Y_arr = np.zeros(n_mass)
    for i in range(n_mass):
        r_arr = np.linspace(1e-4 * r500_arr[i], 5 * r500_arr[i], 500)
        pe    = pp._p3d(r_arr / r500_arr[i], float(m500_arr[i]), z, _H, _OM)
        Y_arr[i] = _SIGMA_T_OVER_ME_C2 * (_MPC_CM / _H) * np.trapezoid(4 * np.pi * pe * r_arr**2, r_arr)

    # Self-similar slope: Y_SZ ∝ M^{5/3+alpha_p}
    alpha_p = _A10_TABLE1["alpha_p"]
    slope   = 5 / 3 + alpha_p  # = 1.797
    m_piv   = 3e14 / _H
    Y_piv   = np.interp(np.log(m_piv), np.log(m500_arr), np.log(Y_arr))
    Y_selfsim = np.exp(Y_piv) * (m500_arr / m_piv) ** slope

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(m500_arr * _H / 1e14, Y_arr, "C0-", lw=2,
              label=r"$Y_{\rm SZ}$ (numerical integral)")
    ax.loglog(m500_arr * _H / 1e14, Y_selfsim, "k--", lw=1.5,
              label=rf"self-similar $\propto M^{{{slope:.3f}}}$")
    ax.set_xlabel(r"$M_{500c}$ [$10^{14}\,M_\odot$]")
    ax.set_ylabel(r"$Y_{\rm SZ}$ [arbitrary units]")
    ax.set_title(r"A10 self-similar scaling $Y_{\rm SZ} \propto M_{500c}^{5/3+\alpha_p}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "a10_03_mass_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Fourier transform ỹ(k|M)
# ---------------------------------------------------------------------------

def fig_pressure_uk():
    print("Figure 4: Pressure FT ỹ(k|M) ...")
    pp    = PressureProfileA10(r_max_over_r500c=5.0, n_gl=150)
    z     = 0.3
    k_arr = np.logspace(-2, 1.5, 80)
    r200, c200, m500, r500 = _m200_r200_c200_m500_r500(_MASSES, z)
    uk    = pp.pressure_uk(k_arr, _MASSES, r200, c200, z, _THETA)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(3):
        ax.loglog(k_arr, uk[:, i], color=_COLORS[i], lw=1.8, label=_MLABELS[i])

    # Annotate the plateau region (k << 1/R500) and the fall-off
    ax.axvline(1 / float(r500[1]), ls=":", color=_COLORS[1], alpha=0.6,
               label=rf"$1/R_{{500c}}$ (M=10$^{{14}}$)")
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$\tilde{y}(k|M)$ [$({\rm Mpc}/h)^2$]")
    ax.set_title(r"A10 pressure Fourier transform $\tilde{y}(k|M)$, $z=0.3$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIG_DIR / "a10_04_pressure_uk.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Parameter check
# ---------------------------------------------------------------------------

def parameter_check():
    pp = PressureProfileA10()
    print("\n--- A10 parameter check (vs Table 1 of arXiv:0910.1234) ---")
    checks = [
        ("P0",      pp._P0,      _A10_TABLE1["P0"]),
        ("c500",    pp._c500,    _A10_TABLE1["c500"]),
        ("gamma",   pp._gamma,   _A10_TABLE1["gamma"]),
        ("alpha",   pp._alpha,   _A10_TABLE1["alpha"]),
        ("beta",    pp._beta,    _A10_TABLE1["beta"]),
        ("alpha_p", pp._alpha_p, _A10_TABLE1["alpha_p"]),
    ]
    all_ok = True
    for name, impl, ref in checks:
        ok = abs(impl - ref) < 1e-9
        status = "OK" if ok else "MISMATCH"
        print(f"  {name:10s}: implemented={impl:.6g}, reference={ref:.6g}  [{status}]")
        if not ok:
            all_ok = False
    print("All parameters match." if all_ok else "WARNING: parameter mismatch detected!")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Validating Arnaud+2010 (arXiv:0910.1234) ===\n")
    ok = parameter_check()
    print()
    fig_shape()
    fig_pressure_profile()
    fig_mass_scaling()
    fig_pressure_uk()
    print(f"\nAll figures saved to {_FIG_DIR}/")
    if ok:
        print("PASS: all A10 parameters verified against Table 1.")
    else:
        print("FAIL: parameter mismatch — check PressureProfileA10 implementation.")
