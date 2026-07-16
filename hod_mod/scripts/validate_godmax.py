"""Independent SZ cross-check of hod_mod against GODMAX (shared Battaglia+2012).

Reference
---------
Pandey S. et al. 2024, arXiv:2401.18072 — GODMAX: Gas thermODynamics and
Matter distribution using jAX (https://github.com/shivampcosmo/GODMAX).
Battaglia N. et al. 2012, ApJ 758, 74, arXiv:1109.3711 — the AGN-feedback
generalized-NFW electron pressure profile, used here as the *shared* profile so
both codes model the identical gas physics and the comparison isolates the
projection machinery (HMF/bias integrals, Ogata-Hankel + Limber).

What is compared
----------------
Against a frozen reference produced by
``scripts/godmax/export_godmax_b12_reference.py`` (``--source godmax`` for the
true GODMAX numbers, ``--source independent`` for a self-contained NumPy B12 +
independent quadrature reference), at matched cosmology / mass-z grids:

1. 3D electron pressure  P_e(r|M,z)        [must agree ~sub-percent]
2. Fourier y-profile     ỹ(k|M,z)          [tests the spherical FT + units]
3. tSZ auto              C_ℓ^{yy}           [tests the 1h+2h mass integral + Limber]
4. shear × tSZ           C_ℓ^{κy}           [GODMAX headline; adds the W_κ kernel]

(3)–(4) are only present when the reference was built with ``--source godmax
--with-cl``; otherwise the script compares the profile + ỹ blocks and plots the
hod_mod C_ℓ prediction alone.

Figures produced (docs/_images/, double-underscore convention)
--------------------------------------------------------------
- ``godmax__profile.png``   P_e(x) overlay + ratio, several (M,z)
- ``godmax__uk.png``        ỹ(k|M) overlay + ratio
- ``godmax__cl_yy.png``     C_ℓ^{yy} overlay (+ ratio if reference has it)
- ``godmax__cl_ky.png``     C_ℓ^{κy} overlay (+ ratio if reference has it)

Usage
-----
    python -m hod_mod.scripts.validate_godmax \
        --ref hod_mod/data/godmax/independent_b12_reference.npz
    python -m hod_mod.scripts.validate_godmax --backend camb   # closer c(M) match
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.gas import PressureProfileBattaglia12
from hod_mod.gas.conversions import _RHO_CRIT0
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra

_HERE      = Path(__file__).parent
_IMAGES    = _HERE.parents[1] / "docs" / "_images"
_DEFAULT_REF = _HERE.parents[1] / "hod_mod" / "data" / "godmax" / "independent_b12_reference.npz"

# project data-viz palette (see scripts/fitting/plot_benchmark_fit.py)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"
REF_C, HOD_C = "#eb6834", "#184f95"          # reference (orange) vs hod_mod (blue)

# CMASS-like galaxy HOD (only seeds the halo cache for the C_ℓ blocks)
_HODP = MoreHODModel.default_params()
_HODP.update({"log10mmin": 13.03, "sigma_logm": 0.38, "log10m1": 13.80,
              "alpha": 1.17, "kappa": 0.51})


def _r200c_comoving(m200c_msunh, z, omega_m):
    """Comoving R200c [Mpc/h] (same convention as mdef='200c')."""
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    rho = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3
    return (3.0 * m200c_msunh / (4.0 * np.pi * 200.0 * rho)) ** (1.0 / 3.0)


def _build_cross(cfg, backend):
    """Return (HaloModelCrossSpectra, theta_cosmo) for the B12 pressure profile."""
    theta = {"Omega_m": cfg["Omega_m"], "Omega_b": cfg["Omega_b"], "h": cfg["h"],
             "n_s": cfg["n_s"], "sigma8": cfg["sigma8"]}
    pp = PressureProfileBattaglia12(r_max_over_r200c=cfg["r_max_over_r200c"],
                                    n_gl=200, x_h=cfg["x_h"])
    if backend == "eh98":
        from hod_mod.observables.clustering import make_differentiable_prediction
        pred = make_differentiable_prediction("more15", cm_model="dutton14",
                                              mdef="200c", hmf_model=cfg["hmf_model"])
    else:  # camb — closer to GODMAX's diemer19 c(M)
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        from hod_mod.core.halo_mass_function import make_hmf
        from hod_mod.core.halo_profiles import HaloProfile
        from hod_mod.observables.clustering import FullHaloModelPrediction
        theta = LinearPowerSpectrum.default_cosmology()
        theta.setdefault("Omega_b", cfg["Omega_b"])
        col = {"flat": True, "H0": theta["h"] * 100, "Om0": theta["Omega_m"],
               "Ob0": theta["Omega_b"], "sigma8": cfg["sigma8"], "ns": theta["n_s"]}
        pk = LinearPowerSpectrum()
        hmf = make_hmf(cfg["hmf_model"], pk_func=pk.pk_linear)
        hp = HaloProfile(col, cm_relation=cfg["conc_model"], mdef="200c")
        pred = FullHaloModelPrediction(pk, MoreHODModel(hmf, hmf.bias), hp)
    return HaloModelCrossSpectra(pred, pressure_profile=pp), theta


def _stats(hod, ref):
    r = np.asarray(hod) / np.asarray(ref) - 1.0
    return float(np.max(np.abs(r))), float(np.median(np.abs(r)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_profile(ref, cross, theta):
    pp = cross._pp
    z, m, x, Pe = ref["prof_z"], ref["prof_m200c_Msun"], ref["prof_x"], ref["prof_Pe"]
    om, h = theta["Omega_m"], theta["h"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    zi = int(np.argmin(np.abs(z - 0.55))) if len(z) > 1 else 0
    import jax.numpy as jnp
    mx, md = 0.0, 0.0
    for j, mphys in enumerate(m):
        mh = mphys * h
        r200 = _r200c_comoving(mh, float(z[zi]), om)
        hodPe = np.asarray(pp._p3d(jnp.asarray(x), jnp.asarray(mh), float(z[zi]),
                                   h, om, theta["Omega_b"] / om, jnp.asarray(r200)))
        c = plt.cm.viridis(j / max(len(m) - 1, 1))
        a.loglog(x, Pe[zi, j], color=c, lw=2.4, alpha=0.5)
        a.loglog(x, hodPe, color=c, lw=1.1, ls="--")
        b.semilogx(x, hodPe / Pe[zi, j] - 1.0, color=c, lw=1.2,
                   label=r"$\log_{10}M=%.1f$" % np.log10(mphys))
        u, v = _stats(hodPe, Pe[zi, j]); mx, md = max(mx, u), max(md, v)
    a.set_ylabel(r"$P_e(x)\ [\mathrm{keV\,cm^{-3}}]$"); a.set_title(
        f"B12 electron pressure — reference (solid) vs hod_mod (dashed), z={z[zi]:.2f}")
    b.axhline(0, color=MUTED, lw=0.7); b.set_ylim(-0.02, 0.02)
    b.set_xlabel(r"$x = r/R_{200c}$"); b.set_ylabel("hod_mod/ref − 1")
    b.legend(fontsize=7, ncol=2)
    for ax in (a, b): ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout(); _save(fig, "godmax__profile")
    return mx, md


def fig_uk(ref, cross, theta):
    pp = cross._pp
    z, k, m, yt = ref["uk_z"], ref["uk_k"], ref["uk_m200c_h"], ref["uk_ytilde"]
    om, h = theta["Omega_m"], theta["h"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    zi = 0
    r200 = np.array([_r200c_comoving(mm, float(z[zi]), om) for mm in m])
    hod = np.asarray(pp.pressure_uk(k, m, r200, np.full_like(m, 5.0), float(z[zi]), theta))
    mx, md = 0.0, 0.0
    for j, mm in enumerate(m):
        c = plt.cm.viridis(j / max(len(m) - 1, 1))
        a.loglog(k, yt[zi, :, j], color=c, lw=2.4, alpha=0.5)
        a.loglog(k, hod[:, j], color=c, lw=1.1, ls="--",
                 label=r"$\log_{10}M=%.1f$" % np.log10(mm))
        b.semilogx(k, hod[:, j] / yt[zi, :, j] - 1.0, color=c, lw=1.2)
        u, v = _stats(hod[:, j], yt[zi, :, j]); mx, md = max(mx, u), max(md, v)
    a.set_ylabel(r"$\tilde{y}(k|M)\ [(\mathrm{Mpc}/h)^2]$"); a.legend(fontsize=7, ncol=2)
    a.set_title(f"B12 y-profile FT — reference (solid) vs hod_mod (dashed), z={z[zi]:.2f}")
    b.axhline(0, color=MUTED, lw=0.7); b.set_ylim(-0.02, 0.02)
    b.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); b.set_ylabel("hod_mod/ref − 1")
    for ax in (a, b): ax.grid(True, color=GRID, lw=0.5)
    fig.tight_layout(); _save(fig, "godmax__uk")
    return mx, md


def fig_cl(ref, cross, theta, kind):
    """kind in {'yy','ky'}: overlay hod_mod C_ℓ (and reference if present).

    The LOS z-grid is a fixed 24-point quadrature (converged for the tSZ Limber
    integral) — decoupled from the reference's own grid so figure generation
    stays tractable regardless of how finely GODMAX sampled z.
    """
    ell = ref["ell"]; z = np.linspace(0.02, 3.5, 24)
    has_ref = f"cl_{kind}" in ref
    if kind == "yy":
        hod = np.asarray(cross.angular_cl_yy(ell, z, theta, _HODP))
        ylab = r"$\ell^2 C_\ell^{yy}/2\pi$"
    else:
        nz = np.interp(z, ref["nz_source_z"], ref["nz_source"])   # onto the local z-grid
        hod = np.asarray(cross.angular_cl_ky(ell, z, nz, theta, _HODP))
        ylab = r"$\ell^2 C_\ell^{\kappa y}/2\pi$"
    dl_hod = ell ** 2 * hod / (2 * np.pi)
    fig, axes = plt.subplots(2 if has_ref else 1, 1, figsize=(6.4, 6.2 if has_ref else 4.2),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]} if has_ref else None)
    a = axes[0, 0]
    a.loglog(ell, dl_hod, color=HOD_C, lw=1.6, marker="o", ms=3.5, label="hod_mod")
    mx = md = None
    if has_ref:
        dl_ref = ell ** 2 * ref[f"cl_{kind}"] / (2 * np.pi)
        a.loglog(ell, dl_ref, color=REF_C, lw=2.4, alpha=0.6, label="GODMAX")
        b = axes[1, 0]; b.semilogx(ell, hod / ref[f"cl_{kind}"] - 1.0, color=INK, lw=1.3)
        b.axhline(0, color=MUTED, lw=0.7); b.set_ylim(-0.2, 0.2)
        b.set_xlabel(r"$\ell$"); b.set_ylabel("hod_mod/ref − 1"); b.grid(True, color=GRID, lw=0.5)
        mx, md = _stats(hod, ref[f"cl_{kind}"])
    else:
        a.set_xlabel(r"$\ell$")
    a.set_ylabel(ylab); a.legend(fontsize=9)
    a.set_title(f"C_ℓ^{{{kind}}} — B12 shared profile" + ("" if has_ref else "  (hod_mod prediction only)"))
    a.grid(True, color=GRID, lw=0.5)
    fig.tight_layout(); _save(fig, f"godmax__cl_{kind}")
    return mx, md


def _save(fig, name):
    _IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(_IMAGES / f"{name}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, default=_DEFAULT_REF)
    ap.add_argument("--backend", choices=["eh98", "camb"], default="eh98")
    args = ap.parse_args()

    if not args.ref.exists():
        raise SystemExit(f"reference not found: {args.ref}\n"
                         "Generate it with scripts/godmax/export_godmax_b12_reference.py")
    ref = np.load(args.ref, allow_pickle=True)
    manifest = args.ref.with_name(args.ref.stem + "_manifest.json")
    if manifest.exists():
        cfg = json.loads(manifest.read_text())["config"]
    else:  # fall back to the export-script defaults for cosmology + model choices
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_godmax_export",
            _HERE.parents[1] / "scripts" / "godmax" / "export_godmax_b12_reference.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cfg = mod.CONFIG
    src = str(ref["source"]) if "source" in ref else "?"

    cross, theta = _build_cross(cfg, args.backend)
    print(f"reference source = {src}   backend = {args.backend}")
    print("=" * 60)
    pmx, pmd = fig_profile(ref, cross, theta)
    print(f"P_e       max |Δ| = {pmx:.2e}   median = {pmd:.2e}")
    umx, umd = fig_uk(ref, cross, theta)
    print(f"ytilde    max |Δ| = {umx:.2e}   median = {umd:.2e}")
    for kind in ("yy", "ky"):
        mx, md = fig_cl(ref, cross, theta, kind)
        if mx is not None:
            print(f"C_l^{kind:3s}  max |Δ| = {mx:.2e}   median = {md:.2e}")
        else:
            print(f"C_l^{kind:3s}  (no reference in npz — plotted hod_mod prediction only)")
    print("=" * 60)
    print(f"figures written to {_IMAGES}")


if __name__ == "__main__":
    main()
