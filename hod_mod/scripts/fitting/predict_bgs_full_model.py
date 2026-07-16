"""Step 0 — predict *every* BGS LS10 M*>10 (S1) summary statistic with the full model.

The galaxy sector is held at the **posterior median** of the BGS mass-bin ZM15
wp+n_gal MCMC (``$HOD_MOD_RESULTS/bgs_zm15_joint_wp_ngal/posterior_summary.json``);
the hot gas (calibrated DPM) and AGN (PowellAGNModel, default Powell/Ananna centres)
sectors run at their default parameters.  For each observable we overlay the default
full-model prediction on its measurement and write one figure.  Nothing is fit here —
this is the diagnostic that tells us where the default model lands, so the
observable-inclusion decision (Step 1) is made from evidence.

Observables
-----------
Galaxy vector (sum_stat, threshold M*>10):  SMF, wp(rp), ESD ΔΣ (DES/HSC/KIDS),
    galaxy w(θ) auto, kNN (data only — the model has no kNN predictor).
Gas:   galaxy × soft-X-ray cross w(θ) broad 0.5–2 keV (Comparat 2025) and the
    15 narrow 0.1-keV bands (full-APEC energy-band path).
AGN:   XLF Φ(L_X) at z=0.1, 0.4 (Roster 2026) and halo bias b(L_X) (literature compilation).

Run::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.predict_bgs_full_model

Output: $HOD_MOD_RESULTS/bgs_full_joint/predictions/*.png  (paths printed at the end).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import h5py

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.powell import PowellAGNModel
from hod_mod.gas.metallicity import MetallicityProfileDPM
from hod_mod.paths import data_root, results_root

# Reused helpers -------------------------------------------------------------
from hod_mod.scripts.direct_prediction_gal_gas_agn import (
    _make_density_variant, _make_pressure_variant, _psf_king_window, _hankel,
    _build_nz, _NE_03_CAL, _P_03_CAL, _BETA_N_CAL, _BETA_P_CAL, _COLOSSUS,
    _ELL, _PSF_THETA_C,
)
from hod_mod.scripts.fitting.fit_comparat2025 import (
    _THETA_COSMO, SAMPLES, load_data as _load_xray_broad,
    load_smf_data, load_wp_data, load_esd_data, _sum_stat_path,
)
from hod_mod.scripts.fitting.fit_xray_joint_bands import (
    load_band_data, _band_cooling, _agn_band_fractions, _BAND_EDGES,
)

_LABEL = "S1"
_Z = SAMPLES[_LABEL]["zmean"]          # 0.135
_ZMAX = SAMPLES[_LABEL]["zmax"]        # 0.18
_H = float(_THETA_COSMO["h"])
_OUT = results_root() / "bgs_full_joint" / "predictions"


# ---------------------------------------------------------------------------
# Data loaders (new)
# ---------------------------------------------------------------------------

def load_zm15_median() -> dict:
    """13-param ZM15 vector from the mass-bin wp+n_gal posterior median."""
    p = results_root() / "bgs_zm15_joint_wp_ngal" / "posterior_summary.json"
    d = json.load(open(p))
    return {k: float(v["med"]) for k, v in d.items()}


def load_roster26_xlf(z: float) -> tuple[np.ndarray, np.ndarray]:
    """Roster 2026 hard-band (2–10 keV) XLF: (log10 L_X, Φ [Mpc^-3 dex^-1])."""
    tag = "z0p1" if abs(z - 0.1) < 0.05 else "z0p4"
    p = data_root() / "erosita" / "observations" / "XLF_Roster26" / f"roster26_{tag}.csv"
    arr = np.genfromtxt(p, delimiter=",", comments="#")
    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr[:, 0], arr[:, 1]


def load_agn_bias_compilation() -> dict:
    """AGN large-scale halo bias vs soft (0.5–2 keV) luminosity compilation."""
    p = data_root() / "erosita" / "observations" / "AGN_bias_halo_compilation.txt"
    arr = np.genfromtxt(p, delimiter=",", comments="#", usecols=(0, 1, 2, 3))
    arr = np.atleast_2d(arr)
    arr = arr[np.isfinite(arr).all(axis=1)]
    return dict(z=arr[:, 0], log10lx_soft=arr[:, 1], bias=arr[:, 2], bias_err=arr[:, 3])


def _load_gal_wtheta() -> dict | None:
    """Galaxy angular auto-correlation w(θ) from sum_stat (twopcf/wtheta_*)."""
    with h5py.File(str(_sum_stat_path(_LABEL)), "r") as f:
        keys = [k for k in f["twopcf"].keys() if k.startswith("wtheta")]
        if not keys:
            return None
        g = f["twopcf"][keys[0]]
        theta = np.asarray(g["sep_centres"], float)
        xi = np.asarray(g["xi"], float)
        err = np.sqrt(np.maximum(np.diag(np.asarray(g["cov"], float)), 0.0))
    return dict(theta_deg=theta, wtheta=xi, wtheta_err=err)


def _load_knn() -> dict | None:
    with h5py.File(str(_sum_stat_path(_LABEL)), "r") as f:
        if "knn" not in f:
            return None
        g = f["knn"][list(f["knn"].keys())[0]]
        return dict(F_k=np.asarray(g["F_k"], float),
                    k_values=np.asarray(g["k_values"], float),
                    r=np.asarray(g["r_centres"], float))


# ---------------------------------------------------------------------------
# Full-model assembly
# ---------------------------------------------------------------------------

def build_full_model():
    """Return (fhmp, cross, powell_at_z, hmf, hod_params).  powell_at_z(z) caches
    a PowellAGNModel per redshift (default Powell/Ananna centres)."""
    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    hp = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)

    dp_cal = _make_density_variant(model=2, ne_03=_NE_03_CAL, beta=_BETA_N_CAL)
    pp_cal = _make_pressure_variant(model=2, P_03=_P_03_CAL, beta=_BETA_P_CAL)
    mp = MetallicityProfileDPM()

    _pw_cache: dict[float, PowellAGNModel] = {}

    def powell_at_z(z: float) -> PowellAGNModel:
        zk = round(float(z), 3)
        if zk not in _pw_cache:
            _pw_cache[zk] = PowellAGNModel(_THETA_COSMO, hmf, z_mean=zk)
        return _pw_cache[zk]

    cross = HaloModelCrossSpectra(
        fhmp, density_profile=dp_cal, pressure_profile=pp_cal,
        metallicity_profile=mp, agn_model=powell_at_z(_Z),
    )

    med = load_zm15_median()
    hod_params = ZuMandelbaum15HODModel.default_params()
    hod_params.update(med)
    hod_params["log10m_star_thresh"] = 10.0
    return fhmp, cross, powell_at_z, hmf, hod_params


# ---------------------------------------------------------------------------
# Per-observable prediction + figure.  Each returns (path, ratio_str).
# ---------------------------------------------------------------------------

def _ratio_str(model, data, mask=None):
    m = np.asarray(model, float); d = np.asarray(data, float)
    ok = np.isfinite(m) & np.isfinite(d) & (np.abs(d) > 0)
    if mask is not None:
        ok &= mask
    if not ok.any():
        return "n/a"
    return f"median model/data = {np.median(m[ok] / d[ok]):.2f}"


def fig_smf(fhmp, hod_params, out):
    import matplotlib.pyplot as plt
    data = load_smf_data(_LABEL)
    grid = data["log10mstar"]
    # Phi(M*) = -dN(>M*)/dlog10M*  from the ZM15 threshold cumulative HOD.
    hp = dict(hod_params)
    n_cum = np.empty_like(grid)
    for i, ms in enumerate(grid):
        hp["log10m_star_thresh"] = float(ms)
        n_cum[i] = fhmp.n_gal(_Z, _THETA_COSMO, hp)      # (Mpc/h)^-3
    phi_model = -np.gradient(n_cum, grid)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(grid, data["phi"], yerr=data["phi_err"], fmt="o", ms=4, color="k", label="BGS S1")
    ax.plot(grid, phi_model, "-", color="C0", lw=2, label="ZM15 median (full model)")
    ax.set_yscale("log"); ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel(r"$\Phi\ [(\mathrm{Mpc}/h)^{-3}\,\mathrm{dex}^{-1}]$")
    ax.set_title("Stellar mass function"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return _ratio_str(phi_model, data["phi"])


def fig_wp(fhmp, hod_params, out):
    import matplotlib.pyplot as plt
    d = load_wp_data(_LABEL, rp_min=0.05)
    wp_m = np.asarray(fhmp.wp(d["rp"], pi_max=d["pi_max"], z=_Z,
                              theta_cosmo=_THETA_COSMO, hod_params=hod_params), float)
    err = np.sqrt(np.maximum(np.diag(d["cov"]), 0.0))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(d["rp"], d["wp"], yerr=err, fmt="o", ms=4, color="k", label="BGS S1")
    ax.plot(d["rp"], wp_m, "-", color="C0", lw=2, label="ZM15 median")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p\ [\mathrm{Mpc}/h]$"); ax.set_ylabel(r"$w_p(r_p)\ [\mathrm{Mpc}/h]$")
    ax.set_title("Projected clustering"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return _ratio_str(wp_m, d["wp"])


def fig_esd(fhmp, hod_params, survey, out):
    import matplotlib.pyplot as plt
    d = load_esd_data(_LABEL, survey)
    rp = d["rp"]; mask = rp > 0.03                      # avoid the inner-boundary artefact
    ds_m = np.asarray(fhmp.delta_sigma(rp[mask], z=_Z, theta_cosmo=_THETA_COSMO,
                                       hod_params=hod_params), float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(rp, d["delta_sigma"], yerr=d["delta_sigma_err"], fmt="o", ms=4, color="k",
                label=f"BGS S1 × {survey}")
    ax.plot(rp[mask], ds_m, "-", color="C0", lw=2, label="ZM15 median")
    ax.set_xscale("log"); ax.set_xlabel(r"$r_p\ [\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$\Delta\Sigma\ [M_\odot h/\mathrm{pc}^2]$")
    ax.set_title(f"Galaxy–galaxy lensing ({survey})"); ax.legend()
    ax.axhline(0, color="grey", lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return _ratio_str(ds_m, d["delta_sigma"][mask])


def fig_wtheta_gal(fhmp, hod_params, out):
    import matplotlib.pyplot as plt
    d = _load_gal_wtheta()
    if d is None:
        return "no data"
    wt_m = np.asarray(fhmp.w_theta(d["theta_deg"], _Z, _THETA_COSMO, hod_params), float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(d["theta_deg"], d["wtheta"], yerr=d["wtheta_err"], fmt="o", ms=4, color="k",
                label="BGS S1")
    ax.plot(d["theta_deg"], wt_m, "-", color="C0", lw=2, label="ZM15 median")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\theta\ [\deg]$"); ax.set_ylabel(r"$w(\theta)$")
    ax.set_title("Galaxy angular auto-correlation"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return _ratio_str(wt_m, d["wtheta"])


def fig_knn(out):
    import matplotlib.pyplot as plt
    d = _load_knn()
    if d is None:
        return "no data"
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, k in enumerate(d["k_values"]):
        ax.plot(d["r"], d["F_k"][i], "-o", ms=3, label=f"k={int(k)}")
    ax.set_xscale("log"); ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$"); ax.set_ylabel(r"$\mathrm{CDF}_k$")
    ax.set_title("kNN CDFs  (DATA ONLY — no model predictor)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return "data only"


def _cross_wtheta(cross, hod_params, z_arr, nz_g, theta_rad, B_ell, x_uk_override=None):
    cl = cross.angular_cl_gX(_ELL, z_arr, nz_g, _THETA_COSMO, hod_params,
                             return_components=True, n_workers=1,
                             x_uk_override=x_uk_override)
    w_gas = _hankel(np.asarray(cl["gas"]) * B_ell, theta_rad)
    w_agn = _hankel(np.asarray(cl["agn"]) * B_ell, theta_rad)
    return w_gas, w_agn


def fig_xray_broad(cross, hod_params, z_arr, nz_g, B_ell, out):
    import matplotlib.pyplot as plt
    d = _load_xray_broad(_LABEL)
    w_gas, w_agn = _cross_wtheta(cross, hod_params, z_arr, nz_g, d["theta_rad"], B_ell)
    w_tot = w_gas + w_agn
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.errorbar(d["theta_arcsec"], d["wtheta"], yerr=d["wtheta_err"], fmt="o", ms=4,
                color="k", label="Comparat 2025")
    ax.plot(d["theta_arcsec"], w_tot, "-", color="k", lw=2, label="total (gas+AGN)")
    ax.plot(d["theta_arcsec"], w_gas, "--", color="C0", lw=1.5, label="gas")
    ax.plot(d["theta_arcsec"], w_agn, "-.", color="C1", lw=1.5, label="AGN (Powell)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\theta\ [\mathrm{arcsec}]$"); ax.set_ylabel(r"$w_\theta$ (galaxy × X-ray)")
    ax.set_title("Galaxy × soft-X-ray cross, broad 0.5–2 keV\n"
                 "(absolute amplitude set by ECF — calibrated in the fit)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return _ratio_str(w_tot, d["wtheta"])


def fig_xray_narrow(cross, hod_params, z_arr, nz_g, B_ell, out):
    import matplotlib.pyplot as plt
    bd = load_band_data(_LABEL)
    cooling_bands, _ = _band_cooling()
    fb = _agn_band_fractions()
    Xb_per_z = cross.emissivity_xuk_bands_per_z(z_arr, _THETA_COSMO, hod_params, cooling_bands)
    nb = len(_BAND_EDGES)
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex=True)
    ratios = []
    for b in range(nb):
        ax = axes.flat[b]
        xov = [Xb_per_z[iz][b] for iz in range(len(z_arr))]
        w_gas, w_agn = _cross_wtheta(cross, hod_params, z_arr, nz_g, bd["theta_rad"], B_ell,
                                     x_uk_override=xov)
        w_tot = w_gas + fb[b] * w_agn
        lo, hi = _BAND_EDGES[b]
        ax.errorbar(bd["theta_arcsec"], bd["wtheta"][b], yerr=bd["wtheta_err"][b],
                    fmt="o", ms=2.5, color="k")
        ax.plot(bd["theta_arcsec"], w_tot, "-", color="C0", lw=1.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{lo:.1f}–{hi:.1f} keV", fontsize=8)
        ratios.append(_ratio_str(w_tot, bd["wtheta"][b]))
        print(f"    band {b:2d} {lo:.1f}-{hi:.1f} keV  {ratios[-1]}", flush=True)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\theta\ [\mathrm{arcsec}]$")
    fig.suptitle("Galaxy × X-ray narrow 0.1-keV bands  (full-APEC; points=data, line=model)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return "; ".join(r for r in ratios[:1]) + " (band 0)"


def fig_xlf(powell_at_z, out_z01, out_z04, hmf):
    import matplotlib.pyplot as plt
    outs = {0.1: out_z01, 0.4: out_z04}
    ratio = {}
    for z, out in outs.items():
        lx_d, phi_d = load_roster26_xlf(z)
        pw = powell_at_z(z)
        grid, phi = pw.xlf(band="hard")
        phi_phys = np.asarray(phi) * _H ** 3          # (Mpc/h)^-3 -> Mpc^-3
        phi_at_d = np.interp(lx_d, grid, phi_phys, left=np.nan, right=np.nan)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(lx_d, phi_d, yerr=0.10 * phi_d, fmt="o", ms=5, color="k",
                    label=f"Roster 2026 (z={z})")
        ax.plot(grid, phi_phys, "-", color="C1", lw=2, label="Powell (default)")
        ax.set_yscale("log"); ax.set_xlim(41.5, 45.5)
        ax.set_ylim(max(1e-9, np.nanmin(phi_d) * 0.2), np.nanmax(phi_d) * 5)
        ax.set_xlabel(r"$\log_{10}(L_X^{2-10\,\mathrm{keV}}/[\mathrm{erg/s}])$")
        ax.set_ylabel(r"$\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        ax.set_title(f"AGN XLF  (z={z})"); ax.legend()
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        ratio[z] = _ratio_str(phi_at_d, phi_d)
    return f"z=0.1: {ratio[0.1]} | z=0.4: {ratio[0.4]}"


def fig_agn_bias(powell_at_z, out):
    import matplotlib.pyplot as plt
    d = load_agn_bias_compilation()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.errorbar(d["log10lx_soft"], d["bias"], yerr=d["bias_err"], fmt="o", ms=5, color="k",
                label="compilation")
    # model b(L_X) per unique redshift
    lx_grid = np.linspace(42.0, 45.5, 40)
    b_pred = np.empty_like(d["bias"])
    for zu in np.unique(np.round(d["z"], 3)):
        pw = powell_at_z(float(zu))
        ax.plot(lx_grid, pw.agn_bias_of_lx(lx_grid, band="soft"), "-", lw=1.5, alpha=0.8,
                label=f"Powell z={zu:.2f}")
        sel = np.round(d["z"], 3) == zu
        b_pred[sel] = pw.agn_bias_of_lx(d["log10lx_soft"][sel], band="soft")
    ax.set_xlabel(r"$\log_{10}(L_X^{0.5-2\,\mathrm{keV}}/[\mathrm{erg/s}])$")
    ax.set_ylabel("large-scale halo bias"); ax.set_title("AGN halo bias vs luminosity")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return _ratio_str(b_pred, d["bias"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description="Step 0 full-model prediction")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="run only these observables (e.g. xray_cross_narrow_montage xlf). "
                         "Default: all.")
    args = ap.parse_args(argv)
    sel = set(args.only) if args.only else None
    _OUT.mkdir(parents=True, exist_ok=True)

    print("Building full model (CAMB + HMF + ZM15 + DPM gas + Powell AGN) ...", flush=True)
    t0 = time.time()
    fhmp, cross, powell_at_z, hmf, hod_params = build_full_model()
    z_arr, nz_g = _build_nz(_Z, _ZMAX, n_pts=7)
    B_ell = _psf_king_window(_ELL, _PSF_THETA_C)
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    tasks = [
        ("smf",              lambda p: fig_smf(fhmp, hod_params, p)),
        ("wp",               lambda p: fig_wp(fhmp, hod_params, p)),
        ("esd_des",          lambda p: fig_esd(fhmp, hod_params, "DES", p)),
        ("esd_hsc",          lambda p: fig_esd(fhmp, hod_params, "HSC", p)),
        ("esd_kids",         lambda p: fig_esd(fhmp, hod_params, "KIDS", p)),
        ("wtheta_gal_auto",  lambda p: fig_wtheta_gal(fhmp, hod_params, p)),
        ("knn",              lambda p: fig_knn(p)),
        ("xray_cross_broad", lambda p: fig_xray_broad(cross, hod_params, z_arr, nz_g, B_ell, p)),
        ("xray_cross_narrow_montage",
                             lambda p: fig_xray_narrow(cross, hod_params, z_arr, nz_g, B_ell, p)),
        ("agn_bias_lx",      lambda p: fig_agn_bias(powell_at_z, p)),
    ]

    if sel is not None:
        tasks = [(n, f) for (n, f) in tasks if n in sel]

    results = []
    for name, fn in tasks:
        path = _OUT / f"{name}.png"
        t = time.time()
        try:
            info = fn(path)
            results.append((name, str(path), info, time.time() - t))
            print(f"  [ok] {name:26s} {time.time()-t:5.1f}s  {info}", flush=True)
        except Exception as e:
            results.append((name, "FAILED", f"{type(e).__name__}: {e}", time.time() - t))
            print(f"  [ERR] {name:26s} {e}", flush=True)
            traceback.print_exc()

    # XLF writes two files
    if sel is None or "xlf" in sel:
        try:
            t = time.time()
            info = fig_xlf(powell_at_z, _OUT / "xlf_z0p1.png", _OUT / "xlf_z0p4.png", hmf)
            results.append(("xlf_z0p1 / xlf_z0p4", str(_OUT / "xlf_z0p*.png"), info, time.time() - t))
            print(f"  [ok] {'xlf':26s} {time.time()-t:5.1f}s  {info}", flush=True)
        except Exception as e:
            results.append(("xlf", "FAILED", f"{type(e).__name__}: {e}", 0.0))
            print(f"  [ERR] xlf  {e}", flush=True)
            traceback.print_exc()

    print("\n" + "=" * 78)
    print("STEP 0 PREDICTION INDEX  (dir: %s)" % _OUT)
    print("=" * 78)
    for name, path, info, dt in results:
        print(f"  {name:28s}  {info}")
    print("\nAll figures in:", _OUT)


if __name__ == "__main__":
    main()
