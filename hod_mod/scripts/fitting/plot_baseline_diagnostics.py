"""Diagnostic figures for the duty-cycle-AGN + gas baseline best fit (S1).

Produces the same two diagnostic figures shown for the other models:
  <sample>_baseline_diagnostics.png      -- SMF, n_gal, SHMR, gg-lensing, X-ray auto
  <sample>_baseline_gas_diagnostics.png  -- Lx-M, kT-M, Lx-kT scaling relations + profiles

The gas physical density is obtained the right way: given the MAP mass slope
``beta_gas``, the DPM central density ``n_e,0.3`` is **calibrated to the X-ray
L_X-M scaling relation** (Comparat+2025) -- not read off the (degenerate) w(theta)
amplitude.  The text panel reports n_e,0.3 and the density normalisation relative
to the fiducial DPM model-2 value, which answers "what is the physical gas
density and is the A_gas normalisation ~1".

Run with:
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.plot_baseline_diagnostics --sample S1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.agn.duty_cycle import load_zm15_map_params
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts import validate_gas_profiles as vgp

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "results", "agn_duty_cycle", "baseline"))
_ALPHA_PROF = 0.9
_BETA_PRESSURE = 0.85           # DPM pressure model-2 mass slope (not fit here)
_NE03_FID = 4.87e-5             # DPM model-2 fiducial central density [cm^-3]


def _hod_params(sample):
    base = ZuMandelbaum15HODModel.default_params()
    base.update(load_zm15_map_params())          # full 13-param ZM15 (fixed)
    base["log10m_star_thresh"] = float(F.SAMPLES[sample]["log10ms_min"])
    return base


def _gas_profile(beta_gas, p2, r_max, ne_03):
    dp = vgp._make_density_variant(model=2, ne_03=ne_03, beta=beta_gas,
                                   alpha_out=_ALPHA_PROF + 2.0 * p2)
    dp._r_max_factor = float(r_max)
    return dp


def plot_gas_diagnostics(sample, mp, out_path):
    from hod_mod.scripts import validate_gas_profiles as v
    beta_gas, p2, r_max = mp["beta_gas"], mp["p2"], mp["r_max"]
    z_eff = F.SAMPLES[sample]["zmean"]
    # pressure mass slope is NOT constrained by w(theta); fix it to self-similar
    # T propto M^(2/3) (beta_P = beta_n + 2/3) so kT(M) is physical and the
    # L_X-M calibration does not collapse below T_min.
    beta_pressure = beta_gas + 2.0 / 3.0

    # physical density calibrated to the X-ray scaling relation (given beta_gas)
    ne_cal, P03_cal = v._calibrate_ne03_P03(beta_gas, beta_pressure, T_min=0.3, z=z_eff)
    dp = _gas_profile(beta_gas, p2, r_max, ne_cal)
    pp = v._make_pressure_variant(model=2, P_03=P03_cal, beta=beta_pressure)
    met = v.MetallicityProfileDPM()

    m_arr = np.logspace(11.5, 15.2, 30)
    r_arr = v._r200(m_arr, z_eff); c_arr = v._c200_approx(m_arr)
    m500_arr, r500_arr = v.m200_to_m500c(m_arr, c_arr, r_arr, v._rho_crit_z(z_eff))
    m500_msun = m500_arr / v._H
    Lx = np.zeros(len(m_arr)); kT = np.zeros(len(m_arr))
    for j, (m, r2, r5) in enumerate(zip(m_arr, r_arr, r500_arr)):
        Lx[j], kT[j], _ = v._integrate_profile(m, r2, r5, z_eff, pp, dp, met, T_min=0.3)

    m_lit = np.logspace(12.0, 15.5, 80)
    M_lo20, Lx_lo20, kT_lo20 = v._load_lovisari20_data()
    M_bu18, Lx_bu18, kT_bu18 = v._load_bulbul18()

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    ax = axes[0, 0]
    ax.scatter(M_lo20, Lx_lo20, s=12, color="gray", alpha=0.7, label="Lovisari+2020")
    ax.scatter(M_bu18, Lx_bu18, s=14, color="steelblue", alpha=0.7, label="Bulbul+2018")
    ax.loglog(m_lit, v._lovisari20_lx(m_lit, z=z_eff), "k--", lw=1.4, label="Lovisari+2020 fit")
    ax.loglog(m500_msun, Lx, "C3-", lw=2.5, label=fr"baseline ($\beta_n$={beta_gas:.2f})")
    ax.set_xlim(1e11, 3e15); ax.set_ylim(1e40, 1e46)
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]"); ax.set_ylabel(r"$L_X$ (0.5-2 keV) [erg/s]")
    ax.set_title(f"{sample}: $L_X$-$M_{{500c}}$"); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.loglog(m_lit, v._lovisari20_kt(m_lit, z=z_eff), "k--", lw=1.4, label="Lovisari+2020 fit")
    ax.scatter(M_lo20, kT_lo20, s=12, color="gray", alpha=0.7, label="Lovisari+2020")
    ax.loglog(m500_msun, kT, "C3-", lw=2.5, label="baseline")
    ax.set_xlabel(r"$M_{500c}$ [$M_\odot$]"); ax.set_ylabel(r"$kT_{\rm ew}$ [keV]")
    ax.set_title(f"{sample}: $kT$-$M_{{500c}}$"); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    ax = axes[0, 2]
    ax.scatter(kT_lo20, Lx_lo20, s=12, color="gray", alpha=0.7, label="Lovisari+2020")
    ax.loglog(kT, Lx, "C3-", lw=2.5, label="baseline")
    ax.set_xlabel(r"$kT_{\rm ew}$ [keV]"); ax.set_ylabel(r"$L_X$ [erg/s]")
    ax.set_title(f"{sample}: $L_X$-$kT$"); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # --- physical gas density: the fit is parametrised directly in log10_ne_03
    # (no free gas normalisation; the model->data conversion C_total = true eROSITA
    # TM0 ECF + APEC cooling + data background S^R_X is built in).
    ne03_wth = float(mp.get("log10_ne_03", np.nan))
    dens_norm = float(mp.get("density_norm", np.nan))
    log10DC = float(mp.get("log10DC", mp.get("log10DC_derived", np.nan)))
    axes[0, 3].axis("off")
    txt = (
        "Baseline gas (DPM model 2 shape, Eq. 8)\n"
        rf"$\beta_n={beta_gas:.3f}$ (MAP density slope)" "\n"
        rf"$\beta_P=\beta_n+2/3={beta_pressure:.3f}$ (self-similar)" "\n"
        rf"$p_2={p2:.3f}\Rightarrow\alpha_{{\rm out}}={_ALPHA_PROF+2*p2:.2f}$" "\n"
        rf"$r_{{\rm max}}={r_max:.2f}\,R_{{200}}$,  $z={z_eff:.3f}$" "\n\n"
        "Cluster $L_X$-$M$ calibrated density:\n"
        rf"$n_{{e,0.3}}={ne_cal:.2e}$ cm$^{{-3}}$,  $P_{{0.3}}={P03_cal:.2e}$" "\n\n"
        "Physical fit parameters (no free norm.):\n"
        rf"  $\log_{{10}}n_{{e,0.3}}={ne03_wth:+.2f}$ (fid ${np.log10(_NE03_FID):+.2f}$)" "\n"
        rf"  density norm $n_e/n_e^{{\rm fid}}={dens_norm:.2f}$" "\n"
        rf"  duty cycle $\log_{{10}}DC={log10DC:+.2f}$")
    axes[0, 3].text(-0.05, 0.5, txt, fontsize=8.5, va="center",
                    transform=axes[0, 3].transAxes)

    x_arr = np.logspace(-2, np.log10(4), 200)
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    ax_ne, ax_pe, ax_z, ax_t = axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]
    for log10m, c in zip([11, 12, 13, 14, 15], cols):
        m_p = 10.0 ** log10m; r2 = v._r200(m_p, z_eff); rr = x_arr * r2
        ne = dp.density_3d(rr, m_p, r2, z_eff, v._OM)
        Pe = pp._pressure_3d(rr, m_p, r2, z_eff, v._OM)
        Z = met.metallicity_3d(rr, r2); T = v.temperature_from_profiles(Pe, ne)
        lbl = rf"$10^{{{log10m}}}$"
        ax_ne.loglog(x_arr, ne, color=c, lw=2, label=lbl)
        ax_pe.loglog(x_arr, Pe, color=c, lw=2, label=lbl)
        ax_z.semilogx(x_arr, Z, color=c, lw=2, label=lbl)
        ax_t.loglog(x_arr, T, color=c, lw=2, label=lbl)
    for a, yl, ti in [(ax_ne, r"$n_e(r)$ [cm$^{-3}$]", "$n_e(r)$"),
                      (ax_pe, r"$P_e(r)$ [keV cm$^{-3}$]", "$P_e(r)$"),
                      (ax_z, r"$Z(r)$ [$Z_\odot$]", "$Z(r)$"),
                      (ax_t, r"$T(r)$ [keV]", "$T(r)$")]:
        a.set_xlabel(r"$r/R_{200}$"); a.set_ylabel(yl); a.set_title(f"{sample}: {ti}")
        a.legend(fontsize=6.5, title=r"$M_{200}[M_\odot/h]$"); a.grid(alpha=0.2)

    fig.suptitle(f"{sample}: baseline gas scaling relations + radial profiles "
                 "(density calibrated to $L_X$-$M$)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  gas diagnostics -> {out_path}  (n_e,0.3={ne_cal:.2e}, norm={dens_norm:.2f})",
          flush=True)
    return ne_cal, dens_norm


def plot_diagnostics(sample, mp, hod_params, out_path):
    from hod_mod.connection import sham
    # infrastructure (fhmp with ZM15 + duty-cycle AGN), inject the baseline gas
    infra = F._Infrastructure(hmf_backend="tinker08", agn_model="duty_cycle")
    z_eff = F.SAMPLES[sample]["zmean"]
    beta_pressure = mp["beta_gas"] + 2.0 / 3.0          # self-similar T propto M^2/3
    ne_cal, _ = vgp._calibrate_ne03_P03(mp["beta_gas"], beta_pressure, T_min=0.3, z=z_eff)
    infra.cross._dp = _gas_profile(mp["beta_gas"], mp["p2"], mp["r_max"], ne_cal)
    infra.use_agn_for(sample)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))

    # SMF
    ax = axes[0, 0]
    try:
        smf = F.load_smf_data(sample)
        ax.errorbar(smf["log10mstar"], smf["phi"], yerr=smf["phi_err"], fmt="o",
                    ms=4, color="k", label="sum_stat")
        grid = smf["log10mstar"]
    except Exception:
        smf = None; grid = np.linspace(F.SAMPLES[sample]["log10ms_min"],
                                       F.SAMPLES[sample]["log10ms_min"] + 2, 20)
    mg = np.linspace(grid.min() - 0.2, grid.max() + 0.2, 60)
    phi = F._predict_smf(infra, z_eff, hod_params, mg)
    ax.plot(mg, phi, "C0-", lw=2, label="model (ZM15)")
    ax.set_yscale("log"); ax.set_xlabel(r"$\log_{10}M_*$")
    ax.set_ylabel(r"$\Phi(M_*)$ [Mpc$^{-3}$dex$^{-1}$]")
    ax.set_title(f"{sample}: stellar mass function"); ax.legend(fontsize=8)

    # n_gal
    ax = axes[0, 1]
    if smf is not None:
        n_d = float(trapezoid(smf["phi"], smf["log10mstar"]))
        n_m = float(trapezoid(F._predict_smf(infra, z_eff, hod_params, grid), grid))
        ax.bar(["sum_stat", "model"], [n_d, n_m], color=["k", "C0"], alpha=0.75)
        ax.set_yscale("log"); ax.set_title(f"{sample}: $\\bar n_g$ ratio={n_m/n_d:.2f}")
        ax.set_ylabel(r"$\bar n_g$ [Mpc$^{-3}$]")
    else:
        ax.axis("off")

    # SHMR
    ax = axes[0, 2]
    mh = np.linspace(10.5, 15.0, 60)
    ax.plot(mh, F._shmr_zu15(mh, hod_params), "C3-", lw=2.2, label="ZM15 (fixed)")
    ax.plot(mh, sham.smhm_moster13(mh, z_eff), "--", lw=1.4, color="C0", label="Moster+2013")
    ax.plot(mh, sham.smhm_behroozi13(mh, z_eff), "-.", lw=1.4, color="C1", label="Behroozi+2013")
    ax.plot(mh, sham.smhm_girelli20(mh, z_eff), ":", lw=1.4, color="C2", label="Girelli+2020")
    ax.set_xlabel(r"$\log_{10}M_h$"); ax.set_ylabel(r"$\log_{10}M_*$")
    ax.set_title(f"{sample}: SHMR ($z={z_eff:.2f}$)"); ax.legend(fontsize=7.5); ax.grid(alpha=0.2)

    # gg-lensing
    ax = axes[1, 0]
    rp = None
    for survey, color in {"HSC": "C0", "DES": "C1", "KIDS": "C2"}.items():
        try:
            esd = F.load_esd_data(sample, survey)
        except Exception:
            continue
        ax.errorbar(esd["rp"], esd["delta_sigma"], yerr=esd["delta_sigma_err"],
                    fmt="o", ms=4, color=color, label=f"{survey}")
        rp = rp if rp is not None else esd["rp"]
    if rp is not None:
        ds = np.asarray(infra.fhmp.delta_sigma(np.asarray(rp), z_eff, F._THETA_COSMO,
                                               hod_params), dtype=float)
        ax.plot(rp, ds, "k-", lw=2, label="model (not fit)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc/$h$]"); ax.set_ylabel(r"$\Delta\Sigma$ [$M_\odot h\,{\rm pc}^{-2}$]")
    ax.set_title(f"{sample}: galaxy-galaxy lensing"); ax.legend(fontsize=8)

    # X-ray auto-power
    ax = axes[1, 1]
    z_arr, nz = F._build_nz_fast(sample)
    cl = infra.cross.angular_cl_XX(F._ELL, z_arr, nz, F._THETA_COSMO,
                                   beta_gas=mp["beta_gas"], beta_pressure=beta_pressure,
                                   psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                   return_components=True, n_workers=1)
    ax.plot(F._ELL, cl["total"], "C0-", lw=2, label="total")
    ax.plot(F._ELL, cl["gas_gas"], "C2--", lw=1.2, label="gas$\\times$gas")
    ax.plot(F._ELL, np.abs(cl["cross"]), "C3:", lw=1.2, label="|gas$\\times$AGN|")
    ax.plot(F._ELL, cl["agn_agn"], "C1-.", lw=1.2, label="AGN$\\times$AGN")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$C_\ell^{XX}$ (model)")
    ax.set_title(f"{sample}: X-ray auto-power"); ax.legend(fontsize=8)

    axes[1, 2].axis("off")
    fig.suptitle(f"{sample}: baseline diagnostics (ZM15 HOD fixed; gas+AGN best fit)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  diagnostics -> {out_path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1")
    args = ap.parse_args(argv)
    mp = json.load(open(os.path.join(_BASE, f"{args.sample}_baseline_map.json")))["map"]
    hod_params = _hod_params(args.sample)
    ne_cal, dens_norm = plot_gas_diagnostics(
        args.sample, mp, os.path.join(_BASE, f"{args.sample}_baseline_gas_diagnostics.png"))
    plot_diagnostics(args.sample, mp, hod_params,
                     os.path.join(_BASE, f"{args.sample}_baseline_diagnostics.png"))
    print(f"\nPhysical parameters ({args.sample} baseline, no free normalisation):"
          f"\n  log10_ne_03 = {mp.get('log10_ne_03', float('nan')):+.3f}  "
          f"(density norm n_e/n_e_fid = {mp.get('density_norm', float('nan')):.2f})"
          f"\n  beta_gas = {mp['beta_gas']:.3f}, p2 = {mp['p2']:.3f}, "
          f"r_max = {mp['r_max']:.2f}"
          f"\n  log10DC = {mp.get('log10DC', mp.get('log10DC_derived', float('nan'))):+.3f}"
          f"  (duty cycle DC = {10**mp.get('log10DC', mp.get('log10DC_derived', -99)):.4f})",
          flush=True)


if __name__ == "__main__":
    main()
