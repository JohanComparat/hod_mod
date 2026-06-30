"""Explore gas-emission profile shapes for the galaxy x X-ray cross-correlation.

Motivation
----------
Following Comparat et al. 2025 (A&A 697, A173), the medium scales (80 kpc - 2
Mpc) of the galaxy x soft-X-ray cross-correlation are dominated by the **hot gas**
of large halos seen from satellite galaxies, with a radial profile that differs
from NFW.  This script scans a set of gas **emission** profile shapes and finds
which best reproduces the cross-correlation on medium and large scales.

The AGN/XRB component is selectable with ``--agn``:
- ``psf``        : an unresolved point source (eROSITA King PSF, small scales
                   only) -- the *uncorrelated* point-source model. The gas
                   profile then shapes the medium/large signal on its own.
- ``correlated`` : the *fully-correlated* duty-cycle AGN cross-power
                   (:class:`~hod_mod.agn.duty_cycle.DutyCycleAGNModel`),
                   which traces the galaxy occupation (1-halo + 2-halo) and so
                   spans all scales -- partly degenerate with the gas.

Gas profile (Comparat 2025, Eq. 8)
----------------------------------
    p_e(r) ∝ x^(-alpha_prof) (1 + x^2)^(-p2),   x = (r/r200m) * c200m

maps onto the package gNFW density profile
:class:`~hod_mod.gas.GasDensityDPM` with
``alpha_in = alpha_prof``, ``alpha_tr = 2``, ``alpha_out = alpha_prof + 2*p2``.
Comparat 2025 use ``(alpha_prof, p2) = (0.9, 1.6)``.  The emission is n_e^2, so
the surface-brightness outer slope is set by ``p2`` and the mass-slope
``beta_gas`` weights the halo masses that dominate medium/large scales.

Run with:
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.galaxies.explore_gas_profiles --sample S1 --agn psf
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.galaxies.explore_gas_profiles --sample S1 --agn correlated
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
                 "results", "agn_duty_cycle", "gas_shape_exploration")
)

_ALPHA_PROF = 0.9                       # Comparat 2025 inner slope
_P2_GRID = [0.6, 0.9, 1.2, 1.6, 2.0, 2.5]   # outer slope (Comparat uses 1.6)
_BETA_GRID = [0.0, 0.3, 0.6, 0.9]           # mass slope (n_e ~ M^beta)

_THETA_MIN, _THETA_MAX = 8.0, 300.0
_THETA_MED = 30.0   # medium/large scales: theta > 30" (~30 kpc proper at z~0.13)


def _king_psf(theta_arcsec, theta_c=8.64, alpha=1.5):
    """Unresolved point-source (AGN/XRB) template: eROSITA King PSF."""
    return (1.0 + (np.asarray(theta_arcsec) / theta_c) ** 2) ** (-alpha)


def _make_gas(alpha_prof, p2, r_max=3.0):
    """GasDensityDPM emulating Comparat 2025 Eq. 8 with the given outer slope."""
    dp = _make_density_variant(model=2, alpha_in=alpha_prof, alpha_tr=2.0,
                               alpha_out=alpha_prof + 2.0 * p2)
    dp._r_max_factor = float(r_max)
    return dp


def _build_hod_params(sample):
    base = ZuMandelbaum15HODModel.default_params()
    base.update(load_zm15_map_params())
    base["log10m_star_thresh"] = float(F.SAMPLES[sample]["log10ms_min"])
    return base


def _fit_two(gas_shape, agn_shape, wdata, err, mask):
    """Non-negative LSQ for (A_gas, A_agn); return amplitudes + chi2 on mask."""
    w = 1.0 / err[mask]
    A = np.column_stack([gas_shape[mask] * w, agn_shape[mask] * w])
    b = wdata[mask] * w
    res = lsq_linear(A, b, bounds=([0.0, 0.0], [np.inf, np.inf]), method="bvls")
    a_gas, a_agn = float(res.x[0]), float(res.x[1])
    model = a_gas * gas_shape + a_agn * agn_shape
    chi2 = float(np.sum(((model[mask] - wdata[mask]) / err[mask]) ** 2))
    return a_gas, a_agn, chi2, model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1", choices=list(F.SAMPLES))
    ap.add_argument("--agn", default="psf", choices=["psf", "correlated"],
                    help="AGN component: 'psf' (point source) or 'correlated' "
                         "(duty-cycle AGN cross-power).")
    ap.add_argument("--hmf", default="tinker08")
    ap.add_argument("--f-sys", type=float, default=0.05)
    args = ap.parse_args(argv)

    os.makedirs(_OUT_DIR, exist_ok=True)
    tag = args.agn
    agn_label = ("AGN/XRB (King PSF, point source)" if tag == "psf"
                 else "AGN (duty-cycle, fully correlated)")

    # ---- infrastructure (galaxy ZM15 HOD fixed) ----
    pk = LinearPowerSpectrum()
    hmf = make_hmf(args.hmf, pk_func=pk.pk_linear)
    th = F._THETA_COSMO
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"],
                Ob0=th["Omega_b"], sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    hod_params = _build_hod_params(args.sample)

    agn_model = None
    agn_kwargs = None
    if tag == "correlated":
        agn_model = DutyCycleAGNModel(sample=args.sample, theta_cosmo=th, hmf=hmf,
                                      log10DC=0.0)
        agn_kwargs = {"log10DC": 0.0}   # AGN template at DC=1; amplitude is fit

    data = F.load_data(args.sample)
    th_as = data["theta_arcsec"]; th_rad = data["theta_rad"]
    wdata = data["wtheta"]
    err = np.sqrt(data["wtheta_err"] ** 2 + (args.f_sys * np.abs(wdata)) ** 2)
    mask_all = (th_as >= _THETA_MIN) & (th_as <= _THETA_MAX)
    mask_med = mask_all & (th_as >= _THETA_MED)
    psf = _king_psf(th_as)

    z_arr, nz = F._build_nz_fast(args.sample)
    n_med = int(mask_med.sum())

    results = []
    chi2_full_grid = np.full((len(_P2_GRID), len(_BETA_GRID)), np.nan)
    chi2_med_grid = np.full((len(_P2_GRID), len(_BETA_GRID)), np.nan)
    best = dict(chi2_med=np.inf)

    t0 = time.time()
    for ip2, p2 in enumerate(_P2_GRID):
        dp = _make_gas(_ALPHA_PROF, p2)
        cross = HaloModelCrossSpectra(fhmp, density_profile=dp, agn_model=agn_model)
        for ib, bg in enumerate(_BETA_GRID):
            comp = cross.angular_cl_gX(
                F._ELL, z_arr, nz, th, hod_params,
                psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                beta_gas=bg, return_components=True, n_workers=1,
                agn_kwargs=agn_kwargs,
            )
            gas_shape = F._hankel(np.asarray(comp["gas"], dtype=float), th_rad)
            if tag == "psf":
                agn_shape = psf
            else:
                agn_shape = F._hankel(np.asarray(comp["agn"], dtype=float), th_rad)
            if not np.all(np.isfinite(gas_shape)) or np.all(gas_shape == 0):
                continue
            a_gas, a_agn, chi2_full, model = _fit_two(gas_shape, agn_shape, wdata, err, mask_all)
            _, _, chi2_med, _ = _fit_two(gas_shape, agn_shape, wdata, err, mask_med)
            chi2_full_grid[ip2, ib] = chi2_full
            chi2_med_grid[ip2, ib] = chi2_med
            rec = dict(alpha_prof=_ALPHA_PROF, p2=p2,
                       alpha_out=_ALPHA_PROF + 2.0 * p2, beta_gas=bg,
                       log10_A_gas=float(np.log10(max(a_gas, 1e-300))),
                       log10_A_agn=float(np.log10(max(a_agn, 1e-300))),
                       chi2_full=chi2_full, chi2_med=chi2_med,
                       n_pts_full=int(mask_all.sum()), n_pts_med=n_med)
            results.append(rec)
            if chi2_med < best["chi2_med"]:
                best = dict(rec, gas_shape=gas_shape, agn_shape=agn_shape,
                            model=model, a_gas=a_gas, a_agn=a_agn)
            print(f"  [{tag}] p2={p2:.2f} (alpha_out={_ALPHA_PROF+2*p2:.2f}) "
                  f"beta={bg:.2f}: chi2_full/dof={chi2_full/(mask_all.sum()-2):.2f} "
                  f"chi2_med/dof={chi2_med/max(n_med-2,1):.2f}", flush=True)

    dt = time.time() - t0
    print(f"\n[{tag}] done in {dt:.1f}s. Best (medium/large): p2={best['p2']}, "
          f"alpha_out={best['alpha_out']:.2f}, beta_gas={best['beta_gas']}, "
          f"chi2_med/dof={best['chi2_med']/max(n_med-2,1):.2f}", flush=True)

    out = dict(sample=args.sample, agn=tag, alpha_prof=_ALPHA_PROF,
               p2_grid=_P2_GRID, beta_grid=_BETA_GRID,
               theta_min=_THETA_MIN, theta_max=_THETA_MAX, theta_med=_THETA_MED,
               agn_component=agn_label, results=results,
               best={k: v for k, v in best.items()
                     if k not in ("gas_shape", "agn_shape", "model")})
    with open(os.path.join(_OUT_DIR, f"gas_exploration_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    _make_figure(out, best, chi2_med_grid, n_med, th_as, data, wdata, tag, agn_label)
    _write_advice(out, best, chi2_med_grid, n_med, tag, agn_label)
    print(f"Saved gas_exploration_{tag}.[json|png] + ADVICE_{tag}.md to {_OUT_DIR}",
          flush=True)


def _make_figure(out, best, chi2_med_grid, n_med, th_as, data, wdata, tag, agn_label):
    p2g, bg = out["p2_grid"], out["beta_grid"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0))
    im = axs[0].imshow(chi2_med_grid / max(n_med - 2, 1), origin="lower",
                       aspect="auto", cmap="viridis_r")
    axs[0].set_xticks(range(len(bg))); axs[0].set_xticklabels([f"{b:.1f}" for b in bg])
    axs[0].set_yticks(range(len(p2g))); axs[0].set_yticklabels([f"{p:.1f}" for p in p2g])
    axs[0].set_xlabel(r"$\beta_{\rm gas}$ (mass slope)")
    axs[0].set_ylabel(r"$p_2$ (outer slope, Eq. 8)")
    axs[0].set_title(rf"med/large $\chi^2/{{\rm dof}}$ ($\theta>30''$) — AGN: {tag}")
    fig.colorbar(im, ax=axs[0])
    axs[0].plot(bg.index(best["beta_gas"]), p2g.index(best["p2"]), "r*", ms=16)

    axs[1].errorbar(th_as, wdata, yerr=data["wtheta_err"], fmt="ko", ms=3,
                    label="data", zorder=6)
    axs[1].axvspan(th_as.min(), out["theta_med"], color="0.93", zorder=0)
    axs[1].plot(th_as, best["a_gas"] * best["gas_shape"], "C0-",
                label=fr"gas ($p_2$={best['p2']}, $\beta$={best['beta_gas']})")
    axs[1].plot(th_as, best["a_agn"] * best["agn_shape"], "C1--", label=agn_label)
    axs[1].plot(th_as, best["model"], "C3-", lw=2, label="gas + AGN")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel(r"$\theta$ [arcsec]"); axs[1].set_ylabel(r"$w(\theta)$")
    axs[1].set_title(f"{out['sample']}: best decomposition (AGN: {tag})")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT_DIR, f"gas_exploration_{tag}.png"), dpi=120)
    plt.close(fig)


def _write_advice(out, best, chi2_med_grid, n_med, tag, agn_label):
    p2g, bg = out["p2_grid"], out["beta_grid"]
    ip = p2g.index(best["p2"]); jb = bg.index(best["beta_gas"])
    col = chi2_med_grid[:, jb] / max(n_med - 2, 1)
    row = chi2_med_grid[ip, :] / max(n_med - 2, 1)
    spread_p2 = float(np.nanmax(col) - np.nanmin(col))
    spread_beta = float(np.nanmax(row) - np.nanmin(row))
    edge = (best["p2"] == p2g[-1] and best["beta_gas"] == bg[-1])
    knobs = sorted(
        [("beta_gas (mass slope, n_e ~ M^beta): weights the halo masses that "
          "dominate medium/large scales", spread_beta),
         ("alpha_out (outer slope = Comparat p2): large-scale fall-off / "
          "surface-brightness beta; the data prefer steeper-than-NFW", spread_p2)],
        key=lambda t: -t[1])
    lines = [
        f"# Gas-profile exploration — advice ({out['sample']}, AGN: {tag})\n",
        f"AGN component: **{agn_label}**.  The gas emission profile (Comparat "
        "2025 Eq. 8) is scanned over outer slope `p2` x mass slope `beta_gas`.\n",
        f"**Best medium/large fit:** `p2 = {best['p2']}` "
        f"(alpha_out = {best['alpha_out']:.2f}), `beta_gas = {best['beta_gas']}` "
        f"-> chi2_med/dof = {best['chi2_med']/max(n_med-2,1):.2f}.\n",
        "## Sensitivity (medium/large chi2/dof spread across the grid)\n",
        f"- mass slope `beta_gas` : Delta(chi2/dof) = {spread_beta:.2f}",
        f"- outer slope `p2` : Delta(chi2/dof) = {spread_p2:.2f}\n",
        ("**Grid edge:** the best (p2, beta_gas) is at the maximum of both axes "
         "-> optimum at/beyond the grid; widen it.\n" if edge else ""),
        "## Recommended free parameters (ordered by measured sensitivity)\n",
        f"1. **{knobs[0][0]}** (Delta chi2/dof = {knobs[0][1]:.2f}) — free first.",
        f"2. **{knobs[1][0]}** (Delta chi2/dof = {knobs[1][1]:.2f}).",
        "3. **concentration / scale radius (`c200m`, `r_max_over_r200`)** — "
        "shifts the medium-scale transition.",
        "4. Keep **`alpha_in`** fixed (~0.9) — only affects small scales.\n",
    ]
    if tag == "correlated":
        lines += [
            "## Caveat (fully-correlated AGN)\n",
            "The duty-cycle AGN cross-power traces the galaxy occupation "
            "(1-halo + 2-halo) and so has a similar theta-shape to the gas on "
            "medium/large scales. It can therefore *absorb* part of the "
            "medium/large signal, weakening the gas-profile constraint (compare "
            "the chi2 spreads with the point-source run, `ADVICE_psf.md`). To "
            "isolate the gas radial profile, the point-source AGN is cleaner.",
        ]
    with open(os.path.join(_OUT_DIR, f"ADVICE_{tag}.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
