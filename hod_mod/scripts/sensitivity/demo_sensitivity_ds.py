"""Cosmological parameter sensitivity of ΔΣ(R) via numerical finite differences.

Computes the logarithmic response function

    S_θ(R) = d ln ΔΣ(R) / d ln θ
           = (θ / ΔΣ(R)) × ∂ΔΣ(R)/∂θ

for each cosmological parameter θ ∈ {Ω_m, σ₈, h, n_s} using ±1% finite
differences through the full 1-halo + 2-halo HOD+HMF+P(k) pipeline.

Backend: Eisenstein & Hu (1998) physical power spectrum
(:func:`hod_mod.cosmology.power_spectrum.eisenstein_hu_pk_phys`) for P(k),
NFW 1-halo term with Diemer+19 concentration–mass relation.  The 1-halo term
dominates at R ≲ 0.3 Mpc/h and makes ΔΣ positive everywhere (unlike the
2-halo-only approximation which gives unphysical negative values at R ≲ 0.04 Mpc/h).

Usage
-----
    python scripts/cosmology/demo_sensitivity_ds.py
    python scripts/cosmology/demo_sensitivity_ds.py --output results/showcase/fig_sensitivity_ds

Outputs
-------
    <output>.png / <output>.pdf   Two-panel figure: ΔΣ (top) + S_θ (bottom)
    Numerical table printed to stdout (RST format for docs inclusion)

References
----------
Eisenstein & Hu 1998, ApJ 496, 605 (arXiv:astro-ph/9709066)
Zheng et al. 2007, ApJ 667, 760 (arXiv:astro-ph/0612166)
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

jax.config.update("jax_enable_x64", True)


from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum, eisenstein_hu_pk_phys
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies.hod import HODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.scripts.cosmology._map_params import load_map_params


# ---------------------------------------------------------------------------
# EH98 backend wrapper
# ---------------------------------------------------------------------------

class _EH98PhysBackend:
    """Wrap eisenstein_hu_pk_phys so FullHaloModelPrediction can call .pk_linear."""
    def pk_linear(self, k, z, theta):  # z unused; growth baked into amplitude
        return eisenstein_hu_pk_phys(k, theta)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_Z = 0.136   # BGS M10 z_eff
_CHI_MAX = 200.0
_N_CHI = 256


def _setup():
    pk_obj = _EH98PhysBackend()
    hmf    = make_hmf("tinker08", pk_func=pk_obj.pk_linear)
    theta, hod_all = load_map_params("zheng2007")
    sigma8 = float(0.811 * np.exp(0.5 * (float(theta["ln10^{10}A_s"]) - 3.044) * np.log(10)))
    colossus_cosmo = {
        "flat": True, "H0": float(theta["h"]) * 100.0,
        "Om0": float(theta["Omega_m"]), "Ob0": float(theta["Omega_b"]),
        "sigma8": sigma8, "ns": float(theta["n_s"]),
    }
    hp   = HaloProfile(colossus_cosmo, cm_relation="diemer19")
    hod  = HODModel(hmf=hmf, halo_bias=hmf.bias)
    pred = FullHaloModelPrediction(pk_obj, hod, hp, k_max=200.0, n_k=1024)
    p    = {k: v for k, v in hod_all.items() if k in set(HODModel.default_params().keys())}
    return pred, theta, p


# ---------------------------------------------------------------------------
# Compute sensitivities
# ---------------------------------------------------------------------------

_DELTA = 0.01   # relative step for finite differences


def compute_sensitivity(pred, theta, p, R, z=_Z,
                        chi_max=_CHI_MAX, n_chi=_N_CHI):
    """Return fiducial ΔΣ and logarithmic response dict (1h+2h, finite differences).

    Parameters
    ----------
    pred : FullHaloModelPrediction
    theta : dict  — cosmological parameter dict
    p : dict      — HOD parameter dict
    R : array  — projected radii [Mpc/h]

    Returns
    -------
    ds_fid : jnp.ndarray, shape (nR,)
    S : dict mapping param name → jnp.ndarray, shape (nR,)
        Logarithmic response S_θ(R) = d ln ΔΣ / d ln θ
    """
    ds_fid = np.array(pred.delta_sigma(R, z=z, theta_cosmo=theta, hod_params=p,
                                        chi_max=chi_max, n_chi=n_chi))

    _PARAMS = [
        ("Omega_m",       r"$\Omega_m$"),
        ("ln10^{10}A_s",  r"$\sigma_8$"),
        ("h",             r"$h$"),
        ("n_s",           r"$n_s$"),
    ]
    S = {}
    for key, label in _PARAMS:
        th_p = dict(theta); th_p[key] = theta[key] * (1.0 + _DELTA)
        th_m = dict(theta); th_m[key] = theta[key] * (1.0 - _DELTA)
        if key == "Omega_m":
            th_p["Omega_cdm"] = th_p["Omega_m"] - theta["Omega_b"]
            th_m["Omega_cdm"] = th_m["Omega_m"] - theta["Omega_b"]
        ds_p = np.array(pred.delta_sigma(R, z=z, theta_cosmo=th_p, hod_params=p,
                                          chi_max=chi_max, n_chi=n_chi))
        ds_m = np.array(pred.delta_sigma(R, z=z, theta_cosmo=th_m, hod_params=p,
                                          chi_max=chi_max, n_chi=n_chi))
        S[label] = (ds_p - ds_m) / (2.0 * _DELTA * ds_fid)

    S[r"$\sigma_8$"] *= 2.0   # ln10As → σ₈ Jacobian: σ₈ ∝ A_s^½
    return jnp.array(ds_fid), {k: jnp.array(v) for k, v in S.items()}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_COLORS = {
    r"$\Omega_m$": "#1f77b4",
    r"$\sigma_8$": "#d62728",
    r"$h$":        "#2ca02c",
    r"$n_s$":      "#ff7f0e",
}
_LS = {
    r"$\Omega_m$": "-",
    r"$\sigma_8$": "--",
    r"$h$":        "-.",
    r"$n_s$":      ":",
}


def make_figure(R, ds_fid, S, output=None):
    """Two-panel figure: ΔΣ (top) and S_θ (bottom)."""
    fig, (ax_ds, ax_s) = plt.subplots(
        2, 1, figsize=(7, 6),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 1.4]},
    )

    # ---- upper: fiducial ΔΣ ----
    # ΔΣ may be negative at R < ~0.02 Mpc/h (tabulation artifact); use symlog
    ax_ds.semilogx(R, ds_fid, color="k", lw=2)
    ax_ds.set_yscale("symlog", linthresh=0.1)
    ax_ds.axhline(0, color="gray", lw=0.6, ls=":")
    ax_ds.set_ylabel(r"$\Delta\Sigma(R)\;[M_\odot\,h\,{\rm pc}^{-2}]$")
    ax_ds.set_title(
        rf"BGS M10 MAP · Zheng+07 HOD · $z={_Z}$  (EH98 phys, 1h+2h More+2015)",
        fontsize=9,
    )
    ax_ds.grid(which="both", ls=":", alpha=0.4)

    # ---- lower: logarithmic response ----
    for name, s_arr in S.items():
        ax_s.semilogx(
            R, np.array(s_arr),
            color=_COLORS[name], ls=_LS[name], lw=2,
            label=name,
        )

    ax_s.set_yscale("symlog", linthresh=0.5)
    ax_s.axhline(0, color="gray", lw=0.8, ls="--")
    ax_s.set_xlabel(r"$R\;[h^{-1}{\rm Mpc}]$")
    ax_s.set_ylabel(
        r"$\mathcal{S}_\theta(R) = \mathrm{d}\ln\Delta\Sigma\,/\,\mathrm{d}\ln\theta$"
    )
    ax_s.legend(
        title=r"1% in $\theta$ $\Rightarrow$ $\mathcal{S}_\theta$% in $\Delta\Sigma$",
        fontsize=9, title_fontsize=8,
        loc="upper right",
    )
    ax_s.grid(which="both", ls=":", alpha=0.4)
    ax_s.set_xlim(float(R[0]) * 0.9, float(R[-1]) * 1.1)

    plt.tight_layout()

    if output:
        for ext in ("pdf", "png"):
            path = f"{output}.{ext}"
            fig.savefig(path, dpi=150)
            print(f"Saved → {path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Table (RST list-table format)
# ---------------------------------------------------------------------------

_TABLE_R = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]   # Mpc/h


def print_table(R, ds_fid, S):
    """Print a reStructuredText list-table with sensitivity values."""
    R_arr  = np.array(R)
    ds_arr = np.array(ds_fid)
    s_om   = np.array(S[r"$\Omega_m$"])
    s_s8   = np.array(S[r"$\sigma_8$"])
    s_h    = np.array(S[r"$h$"])
    s_ns   = np.array(S[r"$n_s$"])

    print()
    print(".. list-table:: "
          r"Logarithmic response :math:`\mathcal{S}_\theta(R)` "
          "(Planck 2018 / Zheng+07 HOD / z=0.14)")
    print("   :header-rows: 1")
    print("   :widths: 12 14 14 14 14 14")
    print()
    print("   * - :math:`R` [:math:`h^{-1}` Mpc]")
    print(r"     - :math:`\Delta\Sigma` [:math:`M_\odot h\,{\rm pc}^{-2}`]")
    print(r"     - :math:`\mathcal{S}_{\Omega_m}`")
    print(r"     - :math:`\mathcal{S}_{\sigma_8}`")
    print(r"     - :math:`\mathcal{S}_{h}`")
    print(r"     - :math:`\mathcal{S}_{n_s}`")

    for R_val in _TABLE_R:
        idx = int(np.argmin(np.abs(R_arr - R_val)))
        print(f"   * - {R_val:g}")
        print(f"     - {ds_arr[idx]:.2f}")
        print(f"     - {s_om[idx]:.3f}")
        print(f"     - {s_s8[idx]:.3f}")
        print(f"     - {s_h[idx]:.3f}")
        print(f"     - {s_ns[idx]:.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=None,
                        help="Output path prefix (no extension); "
                             "saves .png and .pdf.  Default: show interactively.")
    parser.add_argument("--z", type=float, default=_Z)
    parser.add_argument("--chi-max", type=float, default=_CHI_MAX)
    parser.add_argument("--n-chi", type=int, default=_N_CHI)
    parser.add_argument("--n-R", type=int, default=40)
    args = parser.parse_args()

    print(f"Setting up HOD pipeline (EH98 backend, z={args.z})...")
    pred, theta, p = _setup()

    R = jnp.logspace(-2, jnp.log10(30.0), args.n_R)

    print("Computing fiducial ΔΣ and sensitivities (finite differences, 9 forward passes)...")
    ds_fid, S = compute_sensitivity(pred, theta, p, R, z=args.z,
                                    chi_max=args.chi_max, n_chi=args.n_chi)

    print("\n--- Logarithmic response at selected R ---")
    print(f"{'R':>8}  {'dS':>12}  {'S_Om':>8}  {'S_s8':>8}  {'S_h':>8}  {'S_ns':>8}")
    R_arr  = np.array(R)
    ds_arr = np.array(ds_fid)
    k_om = r"$\Omega_m$"
    k_s8 = r"$\sigma_8$"
    k_h  = r"$h$"
    k_ns = r"$n_s$"
    for R_val in _TABLE_R:
        idx = int(np.argmin(np.abs(R_arr - R_val)))
        print(
            f"{R_val:8g}  {ds_arr[idx]:12.2f}"
            f"  {float(S[k_om][idx]):8.3f}"
            f"  {float(S[k_s8][idx]):8.3f}"
            f"  {float(S[k_h][idx]):8.3f}"
            f"  {float(S[k_ns][idx]):8.3f}"
        )

    print("\n--- RST table (for docs/autodiff_ds_sensitivity.rst) ---")
    print_table(R, ds_fid, S)

    make_figure(R, ds_fid, S, output=args.output)


if __name__ == "__main__":
    main()
