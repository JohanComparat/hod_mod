"""Best-fit w(theta) decomposition + cross-sample parameter summary for the
duty-cycle baseline (physical parametrisation: log10_ne_03, log10DC).

For each sample it reconstructs the MAP model from the cached emulator + the
physical params and plots data vs (gas, AGN, total) with a pull residual.  The
last panel summarises the physical parameters vs stellar-mass threshold.

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.plot_baseline_bestfit
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
from hod_mod.scripts.fitting import fit_agn_duty_cycle_baseline as B

_OUT = B._OUT_DIR
_SAMPLES = ["S1", "S3", "S5", "S7"]


def _model_components(sample):
    mp = json.load(open(os.path.join(_OUT, f"{sample}_baseline_map.json")))["map"]
    d = np.load(os.path.join(_OUT, f"{sample}_emulator.npz"))
    interp = RegularGridInterpolator((B._P2_GRID, B._RMAX_GRID, B._BETA_GRID),
                                     d["gas_grid"], method="linear",
                                     bounds_error=False, fill_value=None)
    data = F.load_data(sample)
    th = data["theta_arcsec"]
    gas_shape = interp([[mp["p2"], mp["r_max"], mp["beta_gas"]]])[0]
    c_total = B._c_total(sample)
    a_gas = c_total * (10.0 ** (mp["log10_ne_03"] - np.log10(B._NE03_FID))) ** 2
    a_agn = 10.0 ** (mp["log10DC"] + B._C_OBS_FIXED)
    w_gas = a_gas * gas_shape
    w_agn = a_agn * d["agn_dc1"]
    err = np.sqrt(data["wtheta_err"] ** 2 + (0.05 * np.abs(data["wtheta"])) ** 2)
    return dict(th=th, w=data["wtheta"], err=err, gas=w_gas, agn=w_agn,
                total=w_gas + w_agn, mp=mp, chi2=json.load(
                    open(os.path.join(_OUT, f"{sample}_baseline_map.json")))["chi2_per_dof"])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    rows = {}
    for ax, sample in zip(axes.flat[:4], _SAMPLES):
        c = _model_components(sample)
        rows[sample] = c
        m = (c["th"] >= 8.0) & (c["th"] <= 300.0)
        ax.errorbar(c["th"][m], c["w"][m], yerr=c["err"][m], fmt="o", ms=3.5,
                    color="k", label="data", zorder=5)
        ax.loglog(c["th"][m], np.abs(c["gas"][m]), "C2--", lw=1.5, label="gas")
        ax.loglog(c["th"][m], np.abs(c["agn"][m]), "C1:", lw=1.5, label="AGN")
        ax.loglog(c["th"][m], np.abs(c["total"][m]), "C3-", lw=2, label="total")
        mp = c["mp"]
        ax.set_title(f"{sample} (M*>{F.SAMPLES[sample]['log10ms_min']:.2f}, "
                     f"z={F.SAMPLES[sample]['zmean']:.2f})  "
                     fr"$\chi^2/\nu$={c['chi2']:.2f}", fontsize=10)
        ax.text(0.03, 0.04,
                f"norm={mp['density_norm']:.1f}  "
                f"$\\beta$={mp['beta_gas']:.2f}  $p_2$={mp['p2']:.2f}\n"
                f"$r_{{max}}$={mp['r_max']:.2f}  "
                f"log10DC={mp['log10DC']:+.2f} (DC={10**mp['log10DC']:.3f})",
                transform=ax.transAxes, fontsize=7.5, va="bottom")
        ax.set_xlabel(r"$\theta$ [arcsec]"); ax.set_ylabel(r"$w(\theta)$")
        ax.legend(fontsize=7, loc="upper right")

    # cross-sample parameter summary
    ms = [F.SAMPLES[s]["log10ms_min"] for s in _SAMPLES]
    ax = axes.flat[4]
    ax.plot(ms, [rows[s]["mp"]["density_norm"] for s in _SAMPLES], "o-", color="C2",
            label=r"density norm $n_e/n_e^{\rm fid}$")
    ax.plot(ms, [rows[s]["mp"]["beta_gas"] for s in _SAMPLES], "s-", color="C0",
            label=r"$\beta_{\rm gas}$")
    ax.plot(ms, [rows[s]["mp"]["p2"] for s in _SAMPLES], "^-", color="C4", label=r"$p_2$")
    ax.axhline(10, ls=":", color="C2", lw=0.8); ax.axhline(0.6, ls=":", color="C4", lw=0.8)
    ax.set_xlabel(r"$\log_{10}M_*^{\rm min}$"); ax.set_ylabel("gas parameter")
    ax.set_title("Gas params vs stellar-mass threshold"); ax.legend(fontsize=8)
    ax.text(0.5, 0.5, "dotted = prior edges\n(density railed at 10,\np2 railed at 0.6)",
            transform=ax.transAxes, fontsize=7.5, color="0.4", ha="center")

    ax = axes.flat[5]
    dc = [rows[s]["mp"]["log10DC"] for s in _SAMPLES]
    ax.plot(ms, dc, "o-", color="C1", label=r"$\log_{10}DC$")
    ax.axhspan(-3, -0.301, color="0.92", zorder=0)
    ax.set_xlabel(r"$\log_{10}M_*^{\rm min}$"); ax.set_ylabel(r"$\log_{10}DC$")
    ax.set_title("Duty cycle vs stellar-mass threshold (band = prior)")
    ax.legend(fontsize=8); ax.set_ylim(-3.1, -0.2)

    fig.suptitle("Duty-cycle baseline best fit (physical parameters) — S1/S3/S5/S7",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(_OUT, "baseline_bestfit_allsamples.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved ->", out)
    print("\nsample  M*    norm   beta   p2    rmax   log10DC  DC     chi2/dof")
    for s in _SAMPLES:
        mp = rows[s]["mp"]
        print(f"{s}   {F.SAMPLES[s]['log10ms_min']:.2f}  {mp['density_norm']:5.2f}  "
              f"{mp['beta_gas']:.2f}  {mp['p2']:.2f}  {mp['r_max']:.2f}  "
              f"{mp['log10DC']:+.2f}  {10**mp['log10DC']:.3f}  {rows[s]['chi2']:.2f}")


if __name__ == "__main__":
    main()
