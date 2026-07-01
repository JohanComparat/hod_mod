"""Cross-sample diagnostic figures for the broad-band full-APEC galaxy×X-ray fit.

Post-processes the per-sample MCMC artifacts (``<s>_bb_summary.json``,
``<s>_bb_chain.npz``) and the cached emulator grids (``<s>_emulator_fullapec.npz``)
plus the measured band data, producing SUMMARY figures (no model re-runs):

  1. params_vs_mass.png    physical params + χ²/dof vs stellar-mass threshold
  2. decomp_overview.png   4-panel data vs gas/AGN/total (at the MAP)
  3. band_energy.png       measured w_b(θ) energy dependence per sample
  4. band_ratios.png       soft/hard band ratio vs θ (gas-temperature tracer)
  5. band_validation.png   Σ(15 bands) vs the broad-band w(θ)

Usage:
    HOD_MOD_RESULTS=<results root> python -m hod_mod.scripts.fitting.make_xray_diagnostics
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts.fitting import fit_xray_joint as J
from hod_mod.scripts.fitting import fit_xray_joint_bands as JB

_SAMPLES = ["S1", "S2", "S3", "S4"]
_OUT = os.path.join(J._OUT_DIR, "diagnostics")
_PARAMS = J._PARAMS
_TMIN, _TMAX = 8.0, 300.0


def _load(sample):
    """Reconstruct the model pieces for a sample from the cached artifacts."""
    r = J._OUT_DIR
    summ = json.load(open(os.path.join(r, f"{sample}_bb_summary.json")))
    ch = np.load(os.path.join(r, f"{sample}_bb_chain.npz"))
    em = np.load(os.path.join(r, f"{sample}_emulator_fullapec.npz"))
    interp = RegularGridInterpolator((em["p2_grid"], em["rmax_grid"], em["beta_grid"]),
                                     em["gas_grid"], method="linear",
                                     bounds_error=False, fill_value=None)
    d = F.load_data(sample)
    S = dict(interp=interp, agn_dc1=em["agn_dc1"], c_total=float(ch["c_total"]),
             c_obs_total=J._c_obs_total(sample), th_as=d["theta_arcsec"],
             wdata=d["wtheta"], err=np.sqrt(d["wtheta_err"] ** 2 + (0.05 * np.abs(d["wtheta"])) ** 2),
             mask=(d["theta_arcsec"] >= _TMIN) & (d["theta_arcsec"] <= _TMAX),
             map=summ["map"], post=summ["posterior"], flat=ch["flatchain"],
             log10ms=F.SAMPLES[sample]["log10ms_min"])
    return S


def _model(mp, S):
    ln, beta, p2, rmax, dc = [mp[k] for k in _PARAMS]
    gas = S["c_total"] * (10.0 ** (ln - np.log10(J._NE03_FID))) ** 2 * \
        S["interp"]([[p2, rmax, beta]])[0]
    agn = 10.0 ** dc * S["c_obs_total"] * S["agn_dc1"]
    return gas, agn


def fig_params_vs_mass(SS):
    keys = ["density_norm", "beta_gas", "p2", "r_max", "log10DC"]
    labels = [r"$n_e/n_{e,\rm fid}$", r"$\beta_{\rm gas}$ (L$_X$-M)", r"$p_2$ (outer slope)",
              r"$r_{\max}/r_{200}$", r"$\log_{10}$DC"]
    ms = [SS[s]["log10ms"] for s in _SAMPLES]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
    for a, key, lab in zip(axs.ravel(), keys, labels):
        med = [SS[s]["post"][key]["median"] for s in _SAMPLES]
        lo = [SS[s]["post"][key]["lo"] for s in _SAMPLES]
        hi = [SS[s]["post"][key]["hi"] for s in _SAMPLES]
        a.errorbar(ms, med, yerr=[lo, hi], fmt="o-", capsize=3, color="C0")
        for x, s in zip(ms, _SAMPLES):
            a.annotate(s, (x, SS[s]["post"][key]["median"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
        a.set_xlabel(r"$\log_{10}(M_\star^{\rm thr}/M_\odot)$"); a.set_ylabel(lab, fontsize=9)
    a = axs.ravel()[5]
    chi2 = [SS[s]["map"]["chi2_per_dof"] for s in _SAMPLES]
    a.plot(ms, chi2, "s-", color="C3"); a.axhline(1.0, color="grey", ls=":")
    a.set_xlabel(r"$\log_{10}(M_\star^{\rm thr}/M_\odot)$"); a.set_ylabel(r"MAP $\chi^2$/dof", fontsize=9)
    fig.suptitle("Broad-band (0.5-2 keV) full-APEC fit: physical parameters vs sample", fontsize=11)
    fig.tight_layout(); _save(fig, "params_vs_mass")


def fig_decomp_overview(SS):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for a, s in zip(axs.ravel(), _SAMPLES):
        S = SS[s]; th = S["th_as"]; m = S["mask"]
        gas, agn = _model(S["map"], S); tot = gas + agn
        a.errorbar(th[m], S["wdata"][m], yerr=S["err"][m], fmt="o", ms=3, color="k", label="data")
        a.plot(th[m], np.abs(tot[m]), "C0-", label="total")
        a.plot(th[m], np.abs(gas[m]), "C1--", label="gas")
        a.plot(th[m], np.abs(agn[m]), "C2:", label="AGN")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title(f"{s}  (χ²/dof={S['map']['chi2_per_dof']:.2f}, "
                    f"log₁₀DC={S['map']['log10DC']:.2f})", fontsize=9)
        a.set_xlabel(r"$\theta$ [arcsec]"); a.set_ylabel(r"$|w(\theta)|$"); a.legend(fontsize=7)
    fig.suptitle("Broad-band decomposition: data vs gas / AGN / total (MAP)", fontsize=11)
    fig.tight_layout(); _save(fig, "decomp_overview")


def _band_load(sample):
    bd = JB.load_band_data(sample)
    th = bd["theta_arcsec"]; m = (th >= _TMIN) & (th <= _TMAX)
    return th, m, bd["wtheta"], bd["wtheta_err"]


def fig_band_energy(SS):
    ecen = np.array([0.5 * (lo + hi) for lo, hi in JB._BAND_EDGES])
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for a, s in zip(axs.ravel(), _SAMPLES):
        th, m, w, e = _band_load(s)
        # w vs energy at a few angular scales
        for th0, c in [(10.0, "C0"), (30.0, "C1"), (100.0, "C2")]:
            j = int(np.argmin(np.abs(th - th0)))
            a.plot(ecen, w[:, j], "o-", color=c, ms=3, label=fr"$\theta$≈{th[j]:.0f}″")
        a.set_xlabel("band energy [keV]"); a.set_ylabel(r"$w_b(\theta)$")
        a.set_title(f"{s}: energy dependence", fontsize=9); a.legend(fontsize=7)
        a.axhline(0, color="grey", lw=0.5)
    fig.suptitle("Measured band w(θ): X-ray energy dependence (the temperature signal)", fontsize=11)
    fig.tight_layout(); _save(fig, "band_energy")


def fig_band_ratios(SS):
    # soft = sum(0.5-0.9 keV) = bands 0-3; hard = sum(1.5-2.0) = bands 10-14
    fig, a = plt.subplots(figsize=(7.5, 5.5))
    for s, c in zip(_SAMPLES, ["C0", "C1", "C2", "C3"]):
        th, m, w, e = _band_load(s)
        soft = np.nansum(w[0:4], axis=0); hard = np.nansum(w[10:15], axis=0)
        good = m & np.isfinite(soft) & np.isfinite(hard) & (hard > 0)
        a.plot(th[good], soft[good] / hard[good], "o-", ms=3, color=c, label=s)
    a.set_xscale("log"); a.set_xlabel(r"$\theta$ [arcsec]")
    a.set_ylabel("soft(0.5-0.9) / hard(1.5-2.0) band ratio")
    a.set_title("Band ratio vs θ — higher = cooler gas (Phase-B temperature tracer)", fontsize=10)
    a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "band_ratios")


def fig_band_validation(SS):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for a, s in zip(axs.ravel(), _SAMPLES):
        th, m, w, e = _band_load(s)
        bandsum = np.nansum(w, axis=0)
        d = F.load_data(s)
        a.plot(d["theta_arcsec"], d["wtheta"], "k-", label="broad-band (zenodo)")
        a.plot(th, bandsum, "C1o", ms=3, label="Σ(15 bands)")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(_TMIN, _TMAX)
        a.set_title(f"{s}: band-sum vs broad-band", fontsize=9)
        a.set_xlabel(r"$\theta$ [arcsec]"); a.set_ylabel(r"$w(\theta)$"); a.legend(fontsize=7)
    fig.suptitle("Reconstruction check: Σ(15 energy bands) vs the zenodo broad band", fontsize=11)
    fig.tight_layout(); _save(fig, "band_validation")


def fig_band_params_vs_mass():
    """Phase-B params vs stellar mass — only if the band-fit summaries exist."""
    r = JB._OUT_DIR
    rows = {}
    for s in _SAMPLES:
        f = os.path.join(r, f"{s}_bands_summary.json")
        if os.path.isfile(f):
            rows[s] = json.load(open(f))
    if len(rows) < 2:
        print("  (band-fit summaries not found - skipping Phase-B params figure; "
              "run fit_xray_joint_bands --mcmc first)", flush=True)
        return
    ss = list(rows)
    ms = [F.SAMPLES[s]["log10ms_min"] for s in ss]
    keys = JB._PARAMS
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
    for a, key in zip(axs.ravel(), keys):
        med = [rows[s]["posterior"][key]["median"] for s in ss]
        lo = [rows[s]["posterior"][key]["lo"] for s in ss]
        hi = [rows[s]["posterior"][key]["hi"] for s in ss]
        a.errorbar(ms, med, yerr=[lo, hi], fmt="o-", capsize=3, color="C2")
        a.set_xlabel(r"$\log_{10}(M_\star^{\rm thr}/M_\odot)$"); a.set_ylabel(key, fontsize=9)
    fig.suptitle("Energy-band (Phase B) fit: physical parameters vs sample", fontsize=11)
    fig.tight_layout(); _save(fig, "bands_params_vs_mass")


def fig_phaseA_vs_B():
    """Compare the broad-band (Phase A) and energy-band (Phase B) posteriors for the
    parameters they share, per sample, with the grid bounds marked (rail check)."""
    keys = ["beta_gas", "p2", "r_max", "log10DC"]
    bounds = {"beta_gas": (J._BOUNDS[1, 0], J._BOUNDS[1, 1]),
              "p2": (J._BOUNDS[2, 0], J._BOUNDS[2, 1]),
              "r_max": (J._BOUNDS[3, 0], J._BOUNDS[3, 1]),
              "log10DC": (J._BOUNDS[4, 0], J._BOUNDS[4, 1])}
    A, Bd = {}, {}
    for s in _SAMPLES:
        fa = os.path.join(J._OUT_DIR, f"{s}_bb_summary.json")
        fb = os.path.join(JB._OUT_DIR, f"{s}_bands_summary.json")
        if os.path.isfile(fa):
            A[s] = json.load(open(fa))
        if os.path.isfile(fb):
            Bd[s] = json.load(open(fb))
    if not A or not Bd:
        print("  (need both Phase-A and Phase-B summaries - skipping comparison)", flush=True)
        return
    ss = [s for s in _SAMPLES if s in A and s in Bd]
    ms = [F.SAMPLES[s]["log10ms_min"] for s in ss]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for a, key in zip(axs.ravel(), keys):
        for lab, D, c, dx in [("Phase A (broad)", A, "C0", -0.01), ("Phase B (bands)", Bd, "C2", 0.01)]:
            med = [D[s]["posterior"][key]["median"] for s in ss]
            lo = [D[s]["posterior"][key]["lo"] for s in ss]
            hi = [D[s]["posterior"][key]["hi"] for s in ss]
            a.errorbar(np.array(ms) + dx, med, yerr=[lo, hi], fmt="o-", capsize=3,
                       color=c, label=lab)
        for b in bounds[key]:
            a.axhline(b, color="grey", ls=":", lw=0.8)
        a.set_xlabel(r"$\log_{10}(M_\star^{\rm thr}/M_\odot)$"); a.set_ylabel(key, fontsize=9)
        a.legend(fontsize=7)
    fig.suptitle("Phase A (broad-band) vs Phase B (energy-band) posteriors "
                 "(dotted = grid bounds)", fontsize=11)
    fig.tight_layout(); _save(fig, "phaseA_vs_B")


def _save(fig, name):
    os.makedirs(_OUT, exist_ok=True)
    p = os.path.join(_OUT, name + ".png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  saved {p}", flush=True)


def main():
    print(f"Loading MCMC + emulator artifacts from {J._OUT_DIR} ...", flush=True)
    SS = {s: _load(s) for s in _SAMPLES}
    fig_params_vs_mass(SS)
    fig_decomp_overview(SS)
    fig_band_energy(SS)
    fig_band_ratios(SS)
    fig_band_validation(SS)
    fig_band_params_vs_mass()
    fig_phaseA_vs_B()
    print(f"Done -> {_OUT}", flush=True)


if __name__ == "__main__":
    main()
