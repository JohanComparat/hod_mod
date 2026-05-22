"""Sensitivity of ΔΣ(R) to all cosmological and HOD parameters, all models.

For each HOD model computes the logarithmic response

    S_θ(R) = d ln ΔΣ / d ln θ = (θ / ΔΣ) × ∂ΔΣ/∂θ

for every cosmological parameter θ ∈ {Ω_m, σ₈, h, n_s} and every HOD model
parameter, using ``jax.jacfwd`` end-to-end through the EH98 physical backend.

Degeneracy between parameters θ_i and θ_j is quantified by the Pearson
correlation of (S_θ_i(R), S_θ_j(R)) across all R scales.  Parameters
with |corr| > 0.95 are indistinguishable from ΔΣ(R) alone.

HOD models covered
------------------
  Zheng+2007, Kravtsov+2004, More+2015,
  Zacharegkas+2025, van Uitert+2016, Zu & Mandelbaum 2015

Parameters used
---------------
  Best-fit MAP values from the BGS M10 multi-probe fit (rp > 0.3 Mpc/h,
  LS10 DR10, z_eff = 0.136).  Results are in
  ``results/bgs_multiprobe/mstar10.0_wp_{model}_nfw_rp300/map_result.json``.

Usage
-----
    python scripts/cosmology/demo_sensitivity_ds_all_hod.py
    python scripts/cosmology/demo_sensitivity_ds_all_hod.py \\
        --output results/showcase/fig_sensitivity_ds

Outputs
-------
    <output>_{model_name}.{png,pdf}   Per-model two-panel figure
    <output>_cosmo_comparison.{png,pdf}  Cosmological-parameter response across models
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

jax.config.update("jax_enable_x64", True)


from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum, eisenstein_hu_pk_phys
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.hod import (
    HODModel, Kravtsov04HODModel, MoreHODModel,
    Guo18ICSMFModel, Guo19ICSMFModel,
    Zacharegkas25HODModel, VanUitert16CSMFModel,
    ZuMandelbaum15HODModel, Leauthaud12HODModel,
)
from hod_mod.scripts.cosmology._map_params import load_map_params

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_Z = 0.136   # BGS M10 z_eff
_CHI_MAX = 200.0
_N_CHI = 256
_N_R = 40
_RS = np.logspace(-2, np.log10(30.0), _N_R)

_TABLE_R = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

_COSMO_DISPLAY = {
    "Omega_m":        r"$\Omega_m$",
    "ln10^{10}A_s":   r"$\sigma_8$",
    "h":              r"$h$",
    "n_s":            r"$n_s$",
    "Omega_b":        r"$\Omega_b$",
    "Omega_cdm":      r"$\Omega_c$",
}

_HOD_DISPLAY = {
    "log10mmin":       r"$\log M_{\min}$",
    "sigma_logm":      r"$\sigma_{\log M}$",
    "log10m0":         r"$\log M_0$",
    "log10m1":         r"$\log M_1$",
    "alpha":           r"$\alpha$",
    "kappa":           r"$\kappa$",
    "alpha_inc":       r"$\alpha_{\rm inc}$",
    "log10m_inc":      r"$\log M_{\rm inc}$",
    "log10m_star0":    r"$\log M_{\star 0}$",
    "log10m1_shmr":    r"$\log M_{1,\rm sh}$",
    "alpha_shmr":      r"$\alpha_{\rm sh}$",
    "beta_shmr":       r"$\beta_{\rm sh}$",
    "sigma_logm_star": r"$\sigma_{\log M_\star}$",
    "f_cen":           r"$f_{\rm cen}$",
    "log10m_star_min_cen": r"$\log M_{\star,\rm cen}^{\rm min}$",
    "sigma_c_cen":     r"$\sigma_{c,\rm cen}$",
    "f_sat":           r"$f_{\rm sat}$",
    "log10m_star_min_sat": r"$\log M_{\star,\rm sat}^{\rm min}$",
    "sigma_c_sat":     r"$\sigma_{c,\rm sat}$",
    "log10m1_sat":     r"$\log M_{1,\rm sat}$",
    "alpha_sat":       r"$\alpha_{\rm sat}$",
    "log10m_q":        r"$\log M_q$",
    "log10m_star_lo":  r"$\log M_{\star}^{\rm lo}$",
    "log10m_star_hi":  r"$\log M_{\star}^{\rm hi}$",
    "log10eps":        r"$\log\epsilon$",
    "gamma_shmr":      r"$\gamma_{\rm sh}$",
    "delta_shmr":      r"$\delta_{\rm sh}$",
    "B_sat":           r"$B_{\rm sat}$",
    "beta_sat":        r"$\beta_{\rm sat}$",
    "B_cut":           r"$B_{\rm cut}$",
    "beta_cut":        r"$\beta_{\rm cut}$",
    "log10m_h1":       r"$\log M_{h1}$",
    "beta1":           r"$\beta_1$",
    "log10_beta2":     r"$\log\beta_2$",
    "sigma_c":         r"$\sigma_c$",
    "alpha_s":         r"$\alpha_s$",
    "b0":              r"$b_0$",
    "b1":              r"$b_1$",
    "log10m_star_thresh": r"$\log M_{\star,\rm thr}$",
    "lg_m1h":          r"$\log M_{1h}$",
    "lg_m0star":       r"$\log M_0^\star$",
    "beta":            r"$\beta$",
    "delta":           r"$\delta$",
    "gamma":           r"$\gamma$",
    "sigma_lnmstar":   r"$\sigma_{\ln M_\star}$",
    "eta":             r"$\eta$",
    "fc":              r"$f_c$",
    "bsat":            r"$b_{\rm sat}$",
    "bcut":            r"$b_{\rm cut}$",
}

_COSMO_COLORS = {
    "Omega_m":      "#1f77b4",
    "ln10^{10}A_s": "#d62728",
    "h":            "#2ca02c",
    "n_s":          "#ff7f0e",
}
_COSMO_LS = {
    "Omega_m": "-", "ln10^{10}A_s": "--", "h": "-.", "n_s": ":"
}


# ---------------------------------------------------------------------------
# Backend with z-dependent growth factor (Carroll+1992)
# ---------------------------------------------------------------------------

def _growth_jax(z, om):
    """D(z)/D(0) for flat ΛCDM via Carroll+1992 approximation."""
    ol = 1.0 - om
    a = 1.0 / (1.0 + z)
    om_z = om / (om + ol * a ** 3)
    def _g(omx):
        olx = 1.0 - omx
        return 5.0 / 2.0 * omx / (omx ** (4.0 / 7.0) - olx + (1.0 + omx / 2.0) * (1.0 + olx / 70.0))
    return a * _g(om_z) / _g(om)


class _EH98PhysBackend:
    """EH98 physical P(k,z) with z-dependent growth factor."""
    def pk_linear(self, k, z, theta):
        pk_z0 = eisenstein_hu_pk_phys(k, theta)
        D_z = _growth_jax(float(z), theta["Omega_m"])
        return pk_z0 * D_z ** 2


# ---------------------------------------------------------------------------
# HOD model registry
# ---------------------------------------------------------------------------

def _build_models(hmf):
    return [
        ("Zheng+07",         "zheng07",       HODModel,              {"hmf": hmf, "halo_bias": hmf.bias}, "zheng2007"),
        ("Kravtsov+04",      "kravtsov04",    Kravtsov04HODModel,    {"hmf": hmf, "halo_bias": hmf.bias}, "kravtsov2004"),
        ("More+15",          "more15",        MoreHODModel,          {"hmf": hmf, "halo_bias": hmf.bias}, "more2015"),
        ("Guo+18",           "guo18",         Guo18ICSMFModel,       {"hmf": hmf, "halo_bias": hmf.bias}, "guo2018"),
        ("Guo+19",           "guo19",         Guo19ICSMFModel,       {"hmf": hmf, "halo_bias": hmf.bias}, "guo2019"),
        ("Zacharegkas+25",   "zacharegkas25", Zacharegkas25HODModel, {"hmf": hmf},                        "zacharegkas25"),
        ("van Uitert+16",    "vanutert16",    VanUitert16CSMFModel,  {"hmf": hmf},                        "vanuitert16"),
        ("Zu & Mandel. +15", "zm15",          ZuMandelbaum15HODModel,{"hmf": hmf},                        "zu_mandelbaum15"),
        ("Leauthaud+12",     "leauthaud12",   Leauthaud12HODModel,   {"hmf": hmf, "halo_bias": hmf.bias}, "leauthaud12"),
    ]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

_COSMO_KEYS = ["Omega_m", "ln10^{10}A_s", "h", "n_s"]
_DELTA = 0.01   # relative step for finite differences


def compute_all_sensitivities(pred, theta, p, R, z, chi_max, n_chi):
    """Return ds_fid and dict of S_θ(R) for cosmo + HOD params (finite differences)."""
    ds_fid = np.array(pred.delta_sigma(R, z=z, theta_cosmo=theta, hod_params=p,
                                        chi_max=chi_max, n_chi=n_chi))
    S = {}

    # Cosmological parameters
    for key in _COSMO_KEYS:
        if key not in theta:
            continue
        th_p = dict(theta); th_p[key] = theta[key] * (1.0 + _DELTA)
        th_m = dict(theta); th_m[key] = theta[key] * (1.0 - _DELTA)
        if key == "Omega_m":
            th_p["Omega_cdm"] = th_p["Omega_m"] - theta["Omega_b"]
            th_m["Omega_cdm"] = th_m["Omega_m"] - theta["Omega_b"]
        ds_p = np.array(pred.delta_sigma(R, z=z, theta_cosmo=th_p, hod_params=p,
                                          chi_max=chi_max, n_chi=n_chi))
        ds_m = np.array(pred.delta_sigma(R, z=z, theta_cosmo=th_m, hod_params=p,
                                          chi_max=chi_max, n_chi=n_chi))
        s = (ds_p - ds_m) / (2.0 * _DELTA * ds_fid)
        if key == "ln10^{10}A_s":
            s = s * 2.0   # ln10As → σ₈: σ₈ ∝ A_s^½
        S[key] = s

    # HOD parameters
    for key, val in p.items():
        step = max(abs(val) * _DELTA, 1e-6)
        p_p = dict(p); p_p[key] = val + step
        p_m = dict(p); p_m[key] = val - step
        ds_p = np.array(pred.delta_sigma(R, z=z, theta_cosmo=theta, hod_params=p_p,
                                          chi_max=chi_max, n_chi=n_chi))
        ds_m = np.array(pred.delta_sigma(R, z=z, theta_cosmo=theta, hod_params=p_m,
                                          chi_max=chi_max, n_chi=n_chi))
        if abs(val) > 1e-10:
            S[key] = val * (ds_p - ds_m) / (2.0 * step * ds_fid)
        else:
            S[key] = np.zeros_like(ds_fid)

    return jnp.array(ds_fid), {k: jnp.array(v) for k, v in S.items()}


# ---------------------------------------------------------------------------
# Correlation matrix (degeneracy)
# ---------------------------------------------------------------------------

def correlation_matrix(S):
    """Pearson correlation of S_θ(R) vectors → (n_params×n_params) array."""
    keys = list(S.keys())
    n = len(keys)
    mat = np.array([np.asarray(S[k]) for k in keys])
    mat_c = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(mat_c, axis=1, keepdims=True)
    norms = np.where(norms < 1e-20, 1.0, norms)
    mat_n = mat_c / norms
    C = mat_n @ mat_n.T
    return keys, C


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _param_label(key):
    if key in _COSMO_DISPLAY:
        return _COSMO_DISPLAY[key]
    if key in _HOD_DISPLAY:
        return _HOD_DISPLAY[key]
    return key


def make_per_model_figure(name, R, ds_fid, S, output_prefix=None):
    """Two-panel: response curves + correlation heatmap."""
    keys, C = correlation_matrix(S)
    n = len(keys)
    labels = [_param_label(k) for k in keys]

    cosmo_in_S = [k for k in keys if k in _COSMO_KEYS]
    hod_in_S   = [k for k in keys if k not in _COSMO_KEYS]
    hod_cmap   = plt.cm.tab20
    hod_color  = {k: hod_cmap((i % 20) / 20) for i, k in enumerate(hod_in_S)}

    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1], wspace=0.35)
    ax_s = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    # ---- response curves ----
    for k in cosmo_in_S:
        lbl = _COSMO_DISPLAY.get(k, k)
        col = _COSMO_COLORS[k]
        ls  = _COSMO_LS[k]
        ax_s.semilogx(R, np.array(S[k]), color=col, ls=ls, lw=2.5, label=lbl, zorder=5)

    for k in hod_in_S:
        col = hod_color[k]
        ax_s.semilogx(R, np.array(S[k]), color=col, lw=1.2,
                      label=_param_label(k), alpha=0.85)

    ax_s.set_yscale("symlog", linthresh=0.5)
    ax_s.axhline(0, color="gray", lw=0.7, ls="--")
    ax_s.set_xlabel(r"$R\;[h^{-1}{\rm Mpc}]$")
    ax_s.set_ylabel(r"$\mathcal{S}_\theta = \mathrm{d}\ln\Delta\Sigma/\mathrm{d}\ln\theta$")
    ax_s.set_title(f"{name}", fontsize=11, fontweight="bold")
    ax_s.set_xlim(R[0] * 0.9, R[-1] * 1.1)
    ax_s.legend(fontsize=6.5, ncol=2, loc="lower left",
                framealpha=0.85, handlelength=1.5)
    ax_s.grid(which="both", ls=":", alpha=0.35)

    # ---- correlation heatmap ----
    divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax_c.imshow(C, cmap="RdBu_r", norm=divnorm, aspect="auto")
    ax_c.set_xticks(range(n))
    ax_c.set_yticks(range(n))
    ax_c.set_xticklabels(labels, rotation=90, fontsize=7)
    ax_c.set_yticklabels(labels, fontsize=7)
    ax_c.set_title("Pearson corr of\n"
                   r"$\mathcal{S}_i(R)$ vs $\mathcal{S}_j(R)$",
                   fontsize=8)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = C[i, j]
            if abs(v) > 0.95:
                marker = "★" if abs(v) > 0.99 else "◆"
                ax_c.text(j, i, marker, ha="center", va="center",
                          fontsize=8, color="k")

    plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04,
                 label=r"$r(\mathcal{S}_i,\mathcal{S}_j)$")

    fig.suptitle(
        rf"$R\in[0.01,30]\;h^{{-1}}$Mpc · $z={_Z}$ · EH98 phys, 1h+2h (More+2015)",
        fontsize=8, y=1.01,
    )
    plt.tight_layout()

    if output_prefix:
        for ext in ("pdf", "png"):
            fig.savefig(f"{output_prefix}.{ext}", dpi=150, bbox_inches="tight")
    return fig


def make_cosmo_comparison_figure(results, output_prefix=None):
    """2×2 panel: each cosmological parameter response across all HOD models."""
    params  = ["Omega_m", "ln10^{10}A_s", "h", "n_s"]
    display = [r"$\Omega_m$", r"$\sigma_8$", r"$h$", r"$n_s$"]
    ls_cycle = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2)), (0,(1,1)), (0,(5,1,1,1))]

    cmap = plt.cm.tab10
    model_colors = {name: cmap(i / 10) for i, (name, *_) in enumerate(results)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.28)

    for ax, pk, disp in zip(axes.flat, params, display):
        for i, (name, R, _, S, _) in enumerate(results):
            if pk not in S:
                continue
            ax.semilogx(R, np.array(S[pk]),
                        color=model_colors[name],
                        ls=ls_cycle[i % len(ls_cycle)],
                        lw=1.8, label=name)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_title(disp, fontsize=12)
        ax.grid(which="both", ls=":", alpha=0.35)
        ax.set_xlim(_RMIN * 0.9, _RMAX * 1.1)

    for ax in axes[1]:
        ax.set_xlabel(r"$R\;[h^{-1}{\rm Mpc}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mathcal{S}_\theta$")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        r"Cosmological parameter response $\Delta\Sigma$ across all HOD models"
        "\n"
        rf"(EH98 backend, $z={_Z}$)",
        fontsize=10,
    )
    plt.tight_layout()

    if output_prefix:
        for ext in ("pdf", "png"):
            fig.savefig(f"{output_prefix}.{ext}", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def print_summary_table(name, R, S):
    """Print stdout table of key S_θ values at selected R."""
    R_arr = np.asarray(R)
    cosmo_keys_present = [k for k in _COSMO_KEYS if k in S]
    disp_cosmo = [_COSMO_DISPLAY.get(k, k) for k in cosmo_keys_present]
    header = f"{'R':>7}  " + "  ".join(f"{d:>10}" for d in disp_cosmo)
    print(f"\n  {name}")
    print("  " + "-" * len(header))
    print("  " + header)
    for R_val in _TABLE_R:
        idx = int(np.argmin(np.abs(R_arr - R_val)))
        row = f"{R_val:7g}  "
        row += "  ".join(f"{float(S[k][idx]):>+10.3f}" for k in cosmo_keys_present)
        print("  " + row)


def print_degeneracy_summary(name, S):
    """Print strongly degenerate parameter pairs (|corr| > 0.90)."""
    keys, C = correlation_matrix(S)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if abs(C[i, j]) >= 0.90:
                pairs.append((abs(C[i, j]), C[i, j], keys[i], keys[j]))
    pairs.sort(reverse=True)
    if pairs:
        print(f"\n  {name} — degenerate pairs (|r| ≥ 0.90):")
        for absv, v, a, b in pairs[:10]:
            la = _param_label(a)
            lb = _param_label(b)
            print(f"    r = {v:+.3f}  {la:30s}  ↔  {lb}")
    else:
        print(f"\n  {name} — no strongly degenerate pairs (|r| < 0.90)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_RMIN = _RMAX = None


def main():
    global _RMIN, _RMAX

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=None,
                        help="Path prefix for output figures (no extension). "
                             "Per-model figures appended with _{name}.{ext}.")
    parser.add_argument("--z",       type=float, default=_Z)
    parser.add_argument("--chi-max", type=float, default=_CHI_MAX)
    parser.add_argument("--n-chi",   type=int,   default=_N_CHI)
    parser.add_argument("--n-R",     type=int,   default=_N_R)
    args = parser.parse_args()

    R = jnp.array(np.logspace(-2, np.log10(30.0), args.n_R))
    _RMIN = float(R[0])
    _RMAX = float(R[-1])

    print("Building EH98 backend and HMF...")
    pk_obj = _EH98PhysBackend()
    hmf    = make_hmf("tinker08", pk_func=pk_obj.pk_linear)

    models = _build_models(hmf)

    results = []
    for display_name, short_name, cls, kw, map_key in models:
        print(f"\n{'='*55}")
        print(f"  {display_name}  ({cls.__name__})")
        print(f"{'='*55}")
        try:
            theta, hod_all = load_map_params(map_key)
            accepted = set(cls.default_params().keys())
            p = {k: v for k, v in hod_all.items() if k in accepted}
            print(f"  Parameters ({len(p)}): {list(p.keys())}")
            sigma8 = float(0.811 * np.exp(0.5 * (float(theta["ln10^{10}A_s"]) - 3.044) * np.log(10)))
            colossus_cosmo = {
                "flat": True, "H0": float(theta["h"]) * 100.0,
                "Om0": float(theta["Omega_m"]), "Ob0": float(theta["Omega_b"]),
                "sigma8": sigma8, "ns": float(theta["n_s"]),
            }
            hp   = HaloProfile(colossus_cosmo, cm_relation="diemer19")
            hod  = cls(**kw)
            pred = FullHaloModelPrediction(pk_obj, hod, hp, k_max=200.0, n_k=1024)

            ds_fid, S = compute_all_sensitivities(
                pred, theta, p, R, z=args.z,
                chi_max=args.chi_max, n_chi=args.n_chi
            )
            print(f"  ΔΣ_fid range: [{float(ds_fid.min()):.1f}, {float(ds_fid.max()):.1f}] M_sun h pc^-2")

            print_summary_table(display_name, R, S)
            print_degeneracy_summary(display_name, S)

            results.append((display_name, R, ds_fid, S, short_name))
        except Exception as exc:
            import traceback
            print(f"  [SKIPPED]: {exc}")
            traceback.print_exc()

    print(f"\n{'='*55}")
    print("Generating figures...")
    for display_name, R_r, ds_fid, S, short_name in results:
        prefix = f"{args.output}_{short_name}" if args.output else None
        fig = make_per_model_figure(display_name, R_r, ds_fid, S, output_prefix=prefix)
        if prefix:
            print(f"  Saved → {prefix}.{{png,pdf}}")
        else:
            plt.show()
        plt.close(fig)

    comp_prefix = f"{args.output}_cosmo_comparison" if args.output else None
    fig_comp = make_cosmo_comparison_figure(results, output_prefix=comp_prefix)
    if comp_prefix:
        print(f"  Saved → {comp_prefix}.{{png,pdf}}")
    else:
        plt.show()
    plt.close(fig_comp)

    print("\nDone.")


if __name__ == "__main__":
    main()
