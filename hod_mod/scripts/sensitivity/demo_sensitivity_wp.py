"""Cosmological parameter sensitivity of w_p(r_p) via numerical finite differences.

Computes the logarithmic response function

    S_θ(r_p) = d ln w_p(r_p) / d ln θ
             = (θ / w_p(r_p)) × ∂w_p(r_p)/∂θ

for each cosmological parameter θ ∈ {Ω_m, σ₈, h, n_s} using ±1% finite
differences through the full 1-halo + 2-halo HOD+HMF+P(k) pipeline.

Backend: Eisenstein & Hu (1998) physical power spectrum
(:func:`hod_mod.cosmology.power_spectrum.eisenstein_hu_pk_phys`) for P(k),
NFW 1-halo term with Diemer+19 concentration–mass relation.  The 1-halo term
dominates at r_p ≲ 1 Mpc/h and cannot be ignored for the sensitivity study.

Usage
-----
    python scripts/cosmology/demo_sensitivity_wp.py
    python scripts/cosmology/demo_sensitivity_wp.py --output results/showcase/fig_sensitivity_wp

Outputs
-------
    <output>.png / <output>.pdf   Two-panel figure: w_p (top) + S_θ (bottom)
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
    def pk_linear(self, k, z, theta):  # noqa: D102  (z unused; growth in amplitude)
        return eisenstein_hu_pk_phys(k, theta)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_Z = 0.136   # BGS M10 z_eff


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


def compute_sensitivity(pred, theta, p, rp, z=_Z, pi_max=60.0):
    """Return fiducial w_p and logarithmic response dict (1h+2h, finite differences).

    Parameters
    ----------
    pred : FullHaloModelPrediction
    theta : dict  — cosmological parameter dict
    p : dict      — HOD parameter dict
    rp : array  — projected separations [Mpc/h]

    Returns
    -------
    wp_fid : jnp.ndarray, shape (n_rp,)
    S : dict mapping param name → jnp.ndarray
        Logarithmic response S_θ(r_p) = d ln w_p / d ln θ
    """
    wp_fid = np.array(pred.wp(rp, pi_max=pi_max, z=z, theta_cosmo=theta, hod_params=p))

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
        wp_p = np.array(pred.wp(rp, pi_max=pi_max, z=z, theta_cosmo=th_p, hod_params=p))
        wp_m = np.array(pred.wp(rp, pi_max=pi_max, z=z, theta_cosmo=th_m, hod_params=p))
        S[label] = (wp_p - wp_m) / (2.0 * _DELTA * wp_fid)

    S[r"$\sigma_8$"] *= 2.0   # ln10As → σ₈ Jacobian: σ₈ ∝ A_s^½
    return jnp.array(wp_fid), {k: jnp.array(v) for k, v in S.items()}


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


def make_figure(rp, wp_fid, S, output=None):
    """Two-panel figure: w_p (top) and S_θ (bottom)."""
    fig, (ax_wp, ax_s) = plt.subplots(
        2, 1, figsize=(7, 6),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 1.4]},
    )

    # ---- upper: fiducial w_p ----
    ax_wp.loglog(rp, wp_fid, color="k", lw=2)
    ax_wp.set_ylabel(r"$w_p(r_p)\;[{\rm Mpc}/h]$")
    ax_wp.set_title(
        rf"BGS M10 MAP · Zheng+07 HOD · $z={_Z}$  (EH98 phys, 1h+2h More+2015)",
        fontsize=9,
    )
    ax_wp.grid(which="both", ls=":", alpha=0.4)
    ax_wp.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    # ---- lower: logarithmic response ----
    for name, s_arr in S.items():
        ax_s.semilogx(
            rp, np.array(s_arr),
            color=_COLORS[name], ls=_LS[name], lw=2,
            label=name,
        )

    ax_s.axhline(0, color="gray", lw=0.8, ls="--")
    ax_s.set_xlabel(r"$r_p\;[h^{-1}{\rm Mpc}]$")
    ax_s.set_ylabel(
        r"$\mathcal{S}_\theta(r_p) = \mathrm{d}\ln w_p\,/\,\mathrm{d}\ln\theta$"
    )
    ax_s.legend(
        title=r"1% in $\theta$ $\Rightarrow$ $\mathcal{S}_\theta$% in $w_p$",
        fontsize=9, title_fontsize=8,
        loc="upper right",
    )
    ax_s.grid(which="both", ls=":", alpha=0.4)
    ax_s.set_xlim(float(rp[0]) * 0.9, float(rp[-1]) * 1.1)

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

_TABLE_RP = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0]   # Mpc/h


def print_table(rp, wp_fid, S):
    """Print a reStructuredText list-table with sensitivity values."""
    rp_arr = np.array(rp)
    wp_arr = np.array(wp_fid)
    s_om   = np.array(S[r"$\Omega_m$"])
    s_s8   = np.array(S[r"$\sigma_8$"])
    s_h    = np.array(S[r"$h$"])
    s_ns   = np.array(S[r"$n_s$"])

    print()
    print(".. list-table:: "
          r"Logarithmic response :math:`\mathcal{S}_\theta(r_p)` "
          "(Planck 2018 / Zheng+07 HOD / z=0.14)")
    print("   :header-rows: 1")
    print("   :widths: 12 14 14 14 14 14")
    print()
    print("   * - :math:`r_p` [:math:`h^{-1}` Mpc]")
    print(r"     - :math:`w_p` [:math:`h^{-1}` Mpc]")
    print(r"     - :math:`\mathcal{S}_{\Omega_m}`")
    print(r"     - :math:`\mathcal{S}_{\sigma_8}`")
    print(r"     - :math:`\mathcal{S}_{h}`")
    print(r"     - :math:`\mathcal{S}_{n_s}`")

    for rp_val in _TABLE_RP:
        idx = int(np.argmin(np.abs(rp_arr - rp_val)))
        rp_str = f"{rp_val:g}"
        print(f"   * - {rp_str}")
        print(f"     - {wp_arr[idx]:.2f}")
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
    parser.add_argument("--pi-max", type=float, default=100.0)
    parser.add_argument("--n-rp", type=int, default=40)
    args = parser.parse_args()

    print(f"Setting up HOD pipeline (EH98 backend, z={args.z})...")
    pred, theta, p = _setup()

    rp = jnp.logspace(-2, jnp.log10(60.0), args.n_rp)

    print("Computing fiducial w_p and sensitivities (finite differences, 9 forward passes)...")
    wp_fid, S = compute_sensitivity(pred, theta, p, rp, z=args.z, pi_max=args.pi_max)

    print("\n--- Logarithmic response at selected r_p ---")
    print(f"{'rp':>8}  {'wp':>10}  {'S_Om':>8}  {'S_s8':>8}  {'S_h':>8}  {'S_ns':>8}")
    rp_arr = np.array(rp)
    wp_arr = np.array(wp_fid)
    k_om = r"$\Omega_m$"
    k_s8 = r"$\sigma_8$"
    k_h  = r"$h$"
    k_ns = r"$n_s$"
    for rp_val in _TABLE_RP:
        idx = int(np.argmin(np.abs(rp_arr - rp_val)))
        print(
            f"{rp_val:8g}  {wp_arr[idx]:10.2f}"
            f"  {float(S[k_om][idx]):8.3f}"
            f"  {float(S[k_s8][idx]):8.3f}"
            f"  {float(S[k_h][idx]):8.3f}"
            f"  {float(S[k_ns][idx]):8.3f}"
        )

    print("\n--- RST table (for docs/autodiff_sensitivity.rst) ---")
    print_table(rp, wp_fid, S)

    make_figure(rp, wp_fid, S, output=args.output)


if __name__ == "__main__":
    main()
