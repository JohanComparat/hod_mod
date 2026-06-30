"""Explore the duty-cycle AGN high-mass cutoff *scale* for the cross-correlation.

The duty-cycle AGN occupation carries a smooth high-mass cutoff (no X-ray AGN
above a transition mass M_cut).  Here we **free the cutoff scale** and see what
the galaxy x soft-X-ray cross-correlation prefers: for each transition mass
M_50 (the 50%-occupation point, with a fixed 0.3-dex taper width) we recompute
the AGN cross-power, fit the two amplitudes (gas + AGN) to the data with the gas
profile held at the exploration-best shape, and record the chi2 and the AGN's
contribution.

Low M_cut  -> AGN confined to low-mass (galaxy-scale) halos -> small-scale only.
High M_cut -> AGN includes cluster halos -> spans all scales (no cutoff limit).

A free AGN amplitude is used (no C_obs anchoring), so the AGN can be absorbed by
the gas if the two are degenerate on a given scale.

Run with:
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.galaxies.explore_cutoff_scale --sample S1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.duty_cycle import load_zm15_map_params, DutyCycleAGNModel
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts.validate_gas_profiles import _make_density_variant

_OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "results", "agn_duty_cycle", "cutoff_scale_exploration")
)

# Fixed gas profile at the best FULL-range gas (gas-shape exploration, correlated
# AGN): Comparat Eq.8 p2=2.0, beta=0.9 -- where the duty-cycle AGN is engaged.
_GAS_P2, _GAS_RMAX, _GAS_BETA = 2.0, 3.0, 0.9
_ALPHA_PROF = 0.9

# Cutoff transition mass (50%-occupation point); taper width 0.3 dex.
# 17.0 == effectively no cutoff.
_MCUT_GRID = [12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 17.0]
_CUT_WIDTH = 0.301

_THETA_MIN, _THETA_MAX = 8.0, 300.0
_THETA_MED = 30.0


def _make_gas(p2, r_max):
    dp = _make_density_variant(model=2, alpha_in=_ALPHA_PROF, alpha_tr=2.0,
                               alpha_out=_ALPHA_PROF + 2.0 * p2)
    dp._r_max_factor = float(r_max)
    return dp


def _hod_params(sample):
    base = ZuMandelbaum15HODModel.default_params()
    base.update(load_zm15_map_params())
    base["log10m_star_thresh"] = float(F.SAMPLES[sample]["log10ms_min"])
    return base


def _fit_two(gas, agn, wdata, err, mask):
    w = 1.0 / err[mask]
    A = np.column_stack([gas[mask] * w, agn[mask] * w])
    res = lsq_linear(A, wdata[mask] * w, bounds=([0.0, 0.0], [np.inf, np.inf]),
                     method="bvls")
    a_gas, a_agn = float(res.x[0]), float(res.x[1])
    model = a_gas * gas + a_agn * agn
    chi2 = float(np.sum(((model - wdata)[mask] / err[mask]) ** 2))
    return a_gas, a_agn, chi2, model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1", choices=list(F.SAMPLES))
    ap.add_argument("--hmf", default="tinker08")
    ap.add_argument("--f-sys", type=float, default=0.05)
    args = ap.parse_args(argv)
    os.makedirs(_OUT_DIR, exist_ok=True)

    th = F._THETA_COSMO
    pk = LinearPowerSpectrum()
    hmf = make_hmf(args.hmf, pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"],
                Ob0=th["Omega_b"], sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    hod_params = _hod_params(args.sample)

    data = F.load_data(args.sample)
    th_as = data["theta_arcsec"]; th_rad = data["theta_rad"]
    wdata = data["wtheta"]
    err = np.sqrt(data["wtheta_err"] ** 2 + (args.f_sys * np.abs(wdata)) ** 2)
    mask = (th_as >= _THETA_MIN) & (th_as <= _THETA_MAX)
    mask_med = mask & (th_as >= _THETA_MED)
    mask_sml = mask & (th_as < _THETA_MED)
    n_all = int(mask.sum())
    z_arr, nz = F._build_nz_fast(args.sample)

    agn = DutyCycleAGNModel(sample=args.sample, theta_cosmo=th, hmf=hmf, log10DC=0.0)
    cross = HaloModelCrossSpectra(fhmp, density_profile=_make_gas(_GAS_P2, _GAS_RMAX),
                                  agn_model=agn)

    results = []
    best = dict(chi2=np.inf)
    agn_curves = []      # fitted AGN contribution per M_cut
    gas_curve = None     # gas contribution (same for all M_cut)
    t0 = time.time()
    for mcut in _MCUT_GRID:
        agn._log10m_cut_lo = mcut - _CUT_WIDTH / 2.0
        agn._log10m_cut_hi = mcut + _CUT_WIDTH / 2.0
        comp = cross.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                   psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                   beta_gas=_GAS_BETA, return_components=True,
                                   agn_kwargs={"log10DC": 0.0}, n_workers=1)
        gas = F._hankel(np.asarray(comp["gas"], dtype=float), th_rad)
        agn_s = F._hankel(np.asarray(comp["agn"], dtype=float), th_rad)
        a_gas, a_agn, chi2, model = _fit_two(gas, agn_s, wdata, err, mask)
        # AGN fractional contribution to the model on small / medium scales
        agn_w = a_agn * agn_s; tot = np.maximum(model, 1e-300)
        f_agn_sml = float(np.mean((agn_w / tot)[mask_sml]))
        f_agn_med = float(np.mean((agn_w / tot)[mask_med]))
        rec = dict(M_cut=mcut, log10_A_gas=float(np.log10(max(a_gas, 1e-300))),
                   log10_A_agn=float(np.log10(max(a_agn, 1e-300))),
                   chi2=chi2, chi2_per_dof=chi2 / max(n_all - 2, 1),
                   f_agn_small=f_agn_sml, f_agn_med=f_agn_med)
        results.append(rec)
        agn_curves.append(a_agn * agn_s)
        gas_curve = a_gas * gas
        if chi2 < best["chi2"]:
            best = dict(rec, gas=gas, agn_s=agn_s, a_gas=a_gas, a_agn=a_agn, model=model)
        print(f"  M_cut={mcut:.2f}: chi2/dof={chi2/max(n_all-2,1):.2f} "
              f"log10_A_agn={rec['log10_A_agn']:.2f} "
              f"f_agn(small)={f_agn_sml:.2f} f_agn(med)={f_agn_med:.2f}", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. Best chi2/dof at M_cut={best['M_cut']} "
          f"= {best['chi2']/max(n_all-2,1):.2f}", flush=True)

    out = dict(sample=args.sample, gas=dict(p2=_GAS_P2, r_max=_GAS_RMAX, beta=_GAS_BETA),
               cut_width=_CUT_WIDTH, theta_min=_THETA_MIN, theta_max=_THETA_MAX,
               theta_med=_THETA_MED, free_amplitude_AGN=True, results=results,
               best={k: v for k, v in best.items()
                     if k not in ("gas", "agn_s", "model")})
    with open(os.path.join(_OUT_DIR, "cutoff_scale.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- figure ----
    M = np.array([r["M_cut"] for r in results])
    chi2 = np.array([r["chi2_per_dof"] for r in results])
    aagn = np.array([r["log10_A_agn"] for r in results])
    fsml = np.array([r["f_agn_small"] for r in results])
    fmed = np.array([r["f_agn_med"] for r in results])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    axs[0].plot(M, chi2, "ko-")
    axs[0].axvline(14.15, color="0.6", ls=":", label="baseline cutoff (14.15)")
    axs[0].set_xlabel(r"cutoff scale $\log_{10}M_{\rm cut}$ (50% point)")
    axs[0].set_ylabel(r"$\chi^2/{\rm dof}$ ($\theta\in[8,300]''$)")
    axs[0].set_title("Goodness of fit vs cutoff scale"); axs[0].legend(fontsize=8)

    axs[1].plot(M, fsml, "C0-o", label=r"small ($\theta<30''$)")
    axs[1].plot(M, fmed, "C1-o", label=r"medium/large ($\theta>30''$)")
    axs[1].axvline(14.15, color="0.6", ls=":")
    axs[1].set_xlabel(r"$\log_{10}M_{\rm cut}$"); axs[1].set_ylabel("AGN fraction of model")
    axs[1].set_title("AGN contribution vs cutoff scale"); axs[1].legend(fontsize=8)

    axs[2].axvspan(th_as.min(), _THETA_MED, color="0.93", zorder=0)
    axs[2].errorbar(th_as, wdata, yerr=data["wtheta_err"], fmt="ko", ms=3, label="data")
    axs[2].plot(th_as, best["a_gas"] * best["gas"], "C0-", label="gas")
    axs[2].plot(th_as, best["a_agn"] * best["agn_s"], "C1--",
                label=fr"AGN ($M_{{\rm cut}}$={best['M_cut']})")
    axs[2].plot(th_as, best["model"], "C3-", lw=2, label="gas + AGN")
    axs[2].set_xscale("log"); axs[2].set_yscale("log")
    axs[2].set_xlabel(r"$\theta$ [arcsec]"); axs[2].set_ylabel(r"$w(\theta)$")
    axs[2].set_title(f"best decomposition ($M_{{\\rm cut}}$={best['M_cut']})")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT_DIR, "cutoff_scale.png"), dpi=120)
    plt.close(fig)

    # ---- summary figure: AGN prediction (fitted) vs cutoff scale ----
    import matplotlib.cm as cm
    fig2, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.axvspan(th_as.min(), _THETA_MED, color="0.95", zorder=0)
    ax.errorbar(th_as, wdata, yerr=data["wtheta_err"], fmt=".", color="0.55",
                ms=4, alpha=0.6, label="data", zorder=1)
    ax.plot(th_as, gas_curve, "k-", lw=1.8, label="gas (fixed: $p_2$=2.0, $\\beta$=0.9)",
            zorder=2)
    Mc = np.array(_MCUT_GRID, dtype=float)
    norm = plt.Normalize(Mc.min(), Mc.max())
    for mc, agn in zip(_MCUT_GRID, agn_curves):
        lbl = "AGN (no cutoff)" if mc >= 17.0 else None
        ax.plot(th_as, agn, "-", color=cm.viridis(norm(mc)), lw=1.8, zorder=3)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis); sm.set_array([])
    cb = fig2.colorbar(sm, ax=ax)
    cb.set_label(r"cutoff scale $\log_{10}M_{\rm cut}$  (17 = no cutoff)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r"$\theta$ [arcsec]"); ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(f"{args.sample}: duty-cycle AGN prediction vs high-mass cutoff $M_{{\\rm cut}}$")
    ax.text(0.03, 0.06, "lower $M_{\\rm cut}$ removes the AGN cluster\n"
            "(medium/large) tail; small scales unchanged",
            transform=ax.transAxes, fontsize=7.5, color="0.35")
    ax.legend(fontsize=8, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(os.path.join(_OUT_DIR, "cutoff_scale_agn_predictions.png"), dpi=120)
    plt.close(fig2)
    print(f"Saved cutoff_scale.[json|png] + cutoff_scale_agn_predictions.png "
          f"to {_OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
