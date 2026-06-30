"""Verification / illustration figures for the duty-cycle AGN model
(:class:`hod_mod.agn.duty_cycle.DutyCycleAGNModel`), following the
Lau et al. 2025 (ApJ 983, 8) Appendix-A formalism.

Figures (saved to ``results/agn_duty_cycle/``, referenced from the docs):
  fig_dc_01_xlf.png         — Aird+2015 LADE hard XLF Phi(L_X, z)
  fig_dc_02_selection.png   — k_eff(L,z) and observed soft flux S_X(L) (no r-band cut)
  fig_dc_03_kernel.png      — W_AGN(z), n_AGN(z), mean_SX(z) + integrand (Eq. A9)
  fig_dc_04_occupation.png  — ZM15 occupation N_c, N_s × duty cycle 10^log10DC
  fig_dc_05_wtheta.png      — galaxy×AGN-emission w(theta) vs log10DC + gas + data

Run with:
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.galaxies.plot_agn_duty_cycle_model
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.galaxies.plot_agn_duty_cycle_model --no-cross
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.agn.duty_cycle import (
    DutyCycleAGNModel, compute_w_agn_kernel, w_agn_path_for,
)
from hod_mod.paths import results_root

_OUT_DIR = os.path.normpath(
    os.path.join(results_root(), "agn_duty_cycle")
)
_THETA = LinearPowerSpectrum.default_cosmology()


def _kernel(sample: str) -> dict:
    path = w_agn_path_for(sample)
    if not os.path.exists(path):
        compute_w_agn_kernel(sample=sample)
    with h5py.File(path, "r") as f:
        d = {k: np.asarray(f[k]) for k in f.keys()}
        d["_attrs"] = dict(f.attrs)
    return d


def fig_xlf(k, sample, out):
    z_grid, lx, phi = k["z_grid"], k["log10LX_hard"], k["phi_dex"]
    zsel = [z_grid[1], z_grid[len(z_grid)//3], z_grid[2*len(z_grid)//3], z_grid[-2]]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for z in zsel:
        i = int(np.argmin(np.abs(z_grid - z)))
        ax.plot(lx, phi[i], label=f"z = {z_grid[i]:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(L_X^{\rm hard}\,/\,{\rm erg\,s^{-1}})$")
    ax.set_ylabel(r"$\Phi_{\rm AGN}\ [(\rm Mpc/\it h)^{-3}\,{\rm dex}^{-1}]$")
    ax.set_title("Aird+2015 LADE hard XLF (Eq. A4–A6)")
    ax.set_ylim(1e-10, 1e-2)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_selection(k, sample, out):
    z_grid, lx = k["z_grid"], k["log10LX_hard"]
    iz = int(np.argmin(np.abs(z_grid - float(k["_attrs"]["z_mean"]))))
    keff, sx, sel = k["k_eff"][iz], k["S_X"][iz], k["selection_mask"][iz]
    f_lo = float(k["_attrs"].get("flux_lo", 1e-20))
    f_hi = float(k["_attrs"].get("flux_hi", 1e-10))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(lx, keff); axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\log_{10} L_X^{\rm hard}$")
    axs[0].set_ylabel(r"$k_{\rm eff}$ (obsc.-weighted soft/hard)")
    axs[0].set_title("Hard $\\to$ soft K-correction")
    axs[1].plot(lx, np.log10(sx))
    axs[1].axhspan(np.log10(f_lo), np.log10(f_hi), color="C2", alpha=0.15,
                   label="flux range (all events)")
    axs[1].set_xlabel(r"$\log_{10} L_X^{\rm hard}$")
    axs[1].set_ylabel(r"$\log_{10}(S_X\,/\,{\rm erg\,s^{-1}cm^{-2}})$")
    axs[1].set_title(f"Observed soft flux at z={z_grid[iz]:.2f} (no r-band cut)")
    axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_kernel(k, sample, out):
    z_grid = k["z_grid"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].plot(z_grid, k["W_AGN"], "C0-o", ms=3)
    axs[0].set_xlabel("z"); axs[0].set_ylabel(r"$W_{\rm AGN}(z)$ [(Mpc/$h$)$^{-3}$ erg/s/cm$^2$]")
    axs[0].set_title("Kernel $W_{\\rm AGN}(z)$ (Eq. A9)")
    axs[1].plot(z_grid, k["n_AGN"], "C1-o", ms=3)
    axs[1].set_xlabel("z"); axs[1].set_ylabel(r"$n_{\rm AGN}(z)$ [(Mpc/$h$)$^{-3}$]")
    axs[1].set_title("Selected AGN number density")
    axs[2].plot(z_grid, k["mean_SX"], "C2-o", ms=3)
    axs[2].set_xlabel("z"); axs[2].set_ylabel(r"$\langle S_X\rangle(z)$ [erg/s/cm$^2$]")
    axs[2].set_title(r"Mean selected flux $W_{\rm AGN}/n_{\rm AGN}$")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_occupation(agn, out):
    from hod_mod.agn.duty_cycle import DutyCycleAGNModel
    log10m = np.linspace(11, 15.3, 220)
    nc, ns = agn.nc_ns_agn(log10m)                     # AGN occ (ZM15 x cutoff)
    # raw ZM15 (no high-mass cutoff) for reference
    agn_raw = DutyCycleAGNModel(sample=agn._sample, theta_cosmo=agn._theta_cosmo,
                                apply_high_mass_cutoff=False,
                                w_agn_path=agn._w_agn_path)
    nc0, ns0 = agn_raw.nc_ns_agn(log10m)
    lo, hi = agn._log10m_cut_lo, agn._log10m_cut_hi

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axvspan(lo, hi, color="0.9", zorder=0)
    ax.plot(log10m, nc0, "0.6", lw=1, ls="-")
    ax.plot(log10m, ns0, "0.6", lw=1, ls="--", label="ZM15 (no cutoff)")
    ax.plot(log10m, nc, "k-", label=r"$N_c^{\rm AGN}$ (ZM15 $\times$ cutoff)")
    ax.plot(log10m, ns, "k--", label=r"$N_s^{\rm AGN}$ (ZM15 $\times$ cutoff)")
    for ldc, c in zip([-1.0, -2.0, -3.0], ["C0", "C1", "C2"]):
        ax.plot(log10m, 10.0 ** ldc * (nc + ns), c,
                label=fr"$DC(N_c+N_s),\ \log_{{10}}DC={ldc:.0f}$")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1e3)
    ax.set_xlabel(r"$\log_{10}(M_h\,/\,M_\odot h^{-1})$")
    ax.set_ylabel("occupation")
    ax.set_title("AGN occupation = duty cycle $\\times$ ZM15, high-mass cutoff (S1)")
    ax.text(0.5 * (lo + hi), 2e-6, "AGN high-mass\ncutoff", ha="center",
            fontsize=6.5, color="0.45")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_wtheta(agn, sample, out):
    """galaxy×AGN-emission w(theta) vs duty cycle, with gas and the data points."""
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.core.halo_profiles import HaloProfile
    from hod_mod.gas import GasDensityDPM
    from hod_mod.connection.hod import ZuMandelbaum15HODModel
    from hod_mod.observables.clustering import FullHaloModelPrediction
    from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
    from hod_mod.scripts.fitting import fit_comparat2025 as F

    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=_THETA["h"]*100, Om0=_THETA["Omega_m"],
                Ob0=_THETA["Omega_b"], sigma8=0.811, ns=_THETA["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    dp = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=120)
    cross = HaloModelCrossSpectra(fhmp, density_profile=dp, agn_model=agn)

    import json
    # Gas parameters from the data-validated fixed-ZM15 fit ("previous calculation").
    wf = os.path.normpath(os.path.join(
        _OUT_DIR, "..", "fits", "comparat2025_fixedZM15", f"{sample}_map.json"))
    log10_A_gas, beta_gas, beta_pressure = 3.871, 0.25, 0.86
    if os.path.exists(wf):
        d = json.load(open(wf))
        pm = dict(zip(d["param_names"], d["params"]))
        log10_A_gas = float(pm["log10_A_gas"])
        beta_gas = float(pm["beta_gas"])
        beta_pressure = float(pm["beta_pressure"])

    z_arr, nz = F._build_nz_fast(sample)
    comp = cross.angular_cl_gX(
        F._ELL, z_arr, nz, _THETA, agn.zm15_hod_params,
        psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
        beta_gas=beta_gas, beta_pressure=beta_pressure,
        return_components=True, agn_kwargs={"log10DC": 0.0}, n_workers=1,
    )
    data = F.load_data(sample)
    th = data["theta_rad"]; th_as = data["theta_arcsec"]
    w_agn_dc1 = F._hankel(np.asarray(comp["agn"], dtype=float), th)      # AGN at DC=1
    w_gas = (10.0 ** log10_A_gas) * F._hankel(np.asarray(comp["gas"], dtype=float), th)

    # The w(theta)-only data cannot separate the duty cycle from the absolute AGN
    # flux -> observed-map conversion (eROSITA ECF / background normalisation), so
    # we anchor that conversion to the data: the constant C_obs is set so that
    # log10DC = -2 (the physically-expected AGN-host fraction) reproduces the AGN
    # excess that the data leave after the fixed gas model.  log10DC = -4..-1 then
    # bracket the prediction around it.
    mask = (th_as >= 8.0) & (th_as <= 300.0)
    err = np.sqrt(data["wtheta_err"] ** 2 + (0.05 * np.abs(data["wtheta"])) ** 2)
    resid = (data["wtheta"] - w_gas)[mask]
    g = w_agn_dc1[mask]; iv = 1.0 / err[mask] ** 2
    a_anchor = max(np.sum(iv * resid * g) / np.sum(iv * g * g), 1e-300)
    C_obs = a_anchor / 10.0 ** (-2.0)   # DC = 1e-2 reproduces the data residual

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axvspan(th_as.min(), 8.0, color="0.92", zorder=0)
    ax.errorbar(th_as, data["wtheta"], yerr=data["wtheta_err"], fmt="ko", ms=3,
                label="data (S1)", zorder=6)
    ax.plot(th_as, w_gas, "C3-", lw=2,
            label=fr"gas (fixed, $\log_{{10}}A_g$={log10_A_gas:.2f})")
    for ldc, c in zip([-4, -3, -2, -1], ["C0", "C1", "C2", "C4"]):
        w_agn = C_obs * 10.0 ** ldc * w_agn_dc1
        ax.plot(th_as, w_gas + w_agn, c + "-",
                label=fr"gas + AGN, $\log_{{10}}DC={ldc:d}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ [arcsec]"); ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(f"{sample}: fixed gas + duty-cycle AGN prediction")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.04, r"shaded: $\theta<8''$ (inside PSF, excluded from fit)",
            transform=ax.transAxes, fontsize=7, color="0.4")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1")
    ap.add_argument("--no-cross", action="store_true",
                    help="Skip the (slow) cross-correlation w(theta) figure.")
    args = ap.parse_args(argv)

    os.makedirs(_OUT_DIR, exist_ok=True)
    k = _kernel(args.sample)
    fig_xlf(k, args.sample, os.path.join(_OUT_DIR, "fig_dc_01_xlf.png"))
    fig_selection(k, args.sample, os.path.join(_OUT_DIR, "fig_dc_02_selection.png"))
    fig_kernel(k, args.sample, os.path.join(_OUT_DIR, "fig_dc_03_kernel.png"))

    agn = DutyCycleAGNModel(sample=args.sample, theta_cosmo=_THETA, log10DC=-2.0)
    fig_occupation(agn, os.path.join(_OUT_DIR, "fig_dc_04_occupation.png"))
    if not args.no_cross:
        fig_wtheta(agn, args.sample, os.path.join(_OUT_DIR, "fig_dc_05_wtheta.png"))
    print(f"Figures written to {_OUT_DIR}")


if __name__ == "__main__":
    main()
