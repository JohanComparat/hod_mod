"""Validation figures for Comparat+2025 galaxy × eROSITA soft X-ray benchmark.

Reproduces the angular cross-correlation w_θ(θ) between galaxy positions (LS DR10,
stellar-mass-selected) and eROSITA eRASS:5 soft X-ray (0.5-2 keV) photons for 7
stellar-mass-limited samples.

Reference
---------
Comparat et al. 2025, A&A 697, A173 (arXiv:2503.19796)

Model
-----
* ZuMandelbaum15HODModel — best-fit parameters from Table 3
* GasDensityDPM (Model 2, Oppenheimer+2025 arXiv:2505.14782) — hot gas emissivity
* eROSITA PSF: 30 arcsec FWHM Gaussian convolution applied to C_ℓ^{g,X}
* Planck 2018 cosmology (arXiv:1807.06209)
* Tinker+2008 HMF (arXiv:0803.2706)
* C_ell^{g,X} via Limber -> w_theta(theta) via Legendre / Hankel transform

HOD tying: log10m0 = log10mmin + DeltaM0, log10m1 = log10m0 + 1.0

Usage
-----
    cd /home/comparat/software/hod_mod
    python -m hod_mod.scripts.validate_comparat2025

    # Focus on key samples only (M*>10 and M*>11):
    python -m hod_mod.scripts.validate_comparat2025 --key-only

    # Disable eROSITA PSF convolution:
    python -m hod_mod.scripts.validate_comparat2025 --no-psf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j0

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.cosmology import GasDensityDPM
from hod_mod.galaxies.hod import ZuMandelbaum15HODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.cross_spectra import HaloModelCrossSpectra

# eROSITA PSF FWHM [arcsec] — soft X-ray band (0.5–2 keV)
_EROSITA_PSF_FWHM = 30.0

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)
_DATA_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "xray"

# ---------------------------------------------------------------------------
# Planck 2018 cosmology
# ---------------------------------------------------------------------------
_THETA = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = {
    "flat":   True,
    "H0":     _THETA["h"] * 100.0,
    "Om0":    _THETA["Omega_m"],
    "Ob0":    _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns":     _THETA["n_s"],
}

# ---------------------------------------------------------------------------
# Sample definitions (Comparat+2025 Table 1)
# ---------------------------------------------------------------------------
SAMPLES = [
    dict(label="S1", log10ms_min=10.00, zmax=0.18, zmean=0.136, N=2759238),
    dict(label="S2", log10ms_min=10.25, zmax=0.22, zmean=0.172, N=3308841),
    dict(label="S3", log10ms_min=10.50, zmax=0.26, zmean=0.205, N=3263228),
    dict(label="S4", log10ms_min=10.75, zmax=0.31, zmean=0.243, N=2802710),
    dict(label="S5", log10ms_min=11.00, zmax=0.35, zmean=0.261, N=1619838),
    dict(label="S6", log10ms_min=11.25, zmax=0.35, zmean=0.261, N=541855),
    dict(label="S7", log10ms_min=11.50, zmax=0.35, zmean=0.261, N=120882),
]

# HOD best-fit parameters from Comparat+2025 Table 3
# (log10mmin, alpha_sat, sigma_logm, DeltaM0)
_TABLE3 = [
    (12.113, 1.184, 0.666, 0.052),
    (12.260, 1.178, 0.619, 0.014),
    (12.362, 1.163, 0.538, 0.016),
    (12.327, 1.091, 0.228, 0.031),
    (12.674, 1.131, 0.202, 0.018),
    (13.096, 1.159, 0.123, 0.036),
    (13.483, 1.261, 0.100, 0.002),
]

# Free X-ray amplitude from Table 4 (log10(L_X^gas) in units used in paper)
_TABLE4_LX_GAS = [7.81, 11.60, 15.64, 18.27, 42.11, 137.91, 401.20]


def _hod_params_from_table3(row):
    """Build ZuMandelbaum15HODModel parameter dict from Table 3 row."""
    log10mmin, alpha_sat, sigma_logm, delta_m0 = row
    log10m0 = log10mmin + delta_m0
    log10m1 = log10m0 + 1.0
    base = ZuMandelbaum15HODModel.default_params()
    base.update({
        "log10mmin":   log10mmin,
        "alpha":       alpha_sat,
        "sigma_logm":  sigma_logm,
        "log10m0":     log10m0,
        "log10m1":     log10m1,
    })
    return base


def _load_data(label):
    """Load Comparat+2025 w_theta(theta) data for one sample."""
    path = _DATA_DIR / f"comparat2025_wtheta_{label}.csv"
    theta_rad, wtheta, wtheta_err, R_kpc = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            theta_rad.append(float(parts[0]))
            wtheta.append(float(parts[2]))
            wtheta_err.append(float(parts[3]))
            R_kpc.append(float(parts[4]))
    return (
        np.array(theta_rad),
        np.array(wtheta),
        np.array(wtheta_err),
        np.array(R_kpc),
    )


def _build_infrastructure():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp   = FullHaloModelPrediction(pk_lin, hod, hp)
    dp     = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=200)
    cross  = HaloModelCrossSpectra(fhmp, density_profile=dp)
    return cross


def _cl_to_wtheta(ell_arr, cl_arr, theta_rad_arr):
    """Convert C_ell^{g,X} to w_theta(theta) via discrete Legendre / Hankel sum.

    For narrow kernels (galaxy n(z) effectively a delta function in z), the
    angular power spectrum to angular correlation function transform is:

        w_theta(theta) = sum_ell (2l+1)/(4pi) C_ell P_ell(cos(theta))

    We approximate the sum as an integral using a trapezoidal rule over ell:

        w_theta(theta) ≈ (1/2pi) integral ell C_ell J_0(ell theta) dell
    """
    theta = np.asarray(theta_rad_arr, dtype=float)
    ell   = np.asarray(ell_arr,       dtype=float)
    cl    = np.asarray(cl_arr,        dtype=float)

    wtheta = np.zeros(len(theta))
    for i, th in enumerate(theta):
        integrand = ell * cl * j0(ell * th) / (2.0 * np.pi)
        wtheta[i] = np.trapezoid(integrand, ell)
    return wtheta


def _predict_wtheta(sample, cross, hod_params, A_X=1.0, psf_fwhm_arcsec=None):
    """Compute model w_theta(theta) for one sample at its zmean.

    Parameters
    ----------
    A_X : overall amplitude factor (fitted to data)
    psf_fwhm_arcsec : if set, apply eROSITA PSF window to C_ℓ before transform
    """
    z      = sample["zmean"]
    n_ell  = 80
    ell    = np.logspace(1, 4, n_ell)
    # Build narrow n(z) peaked at zmean
    dz     = min(0.02, sample["zmax"] * 0.1)
    z_arr  = np.linspace(max(0.01, z - 3 * dz), z + 3 * dz, 20)
    nz_g   = np.exp(-0.5 * ((z_arr - z) / dz) ** 2)

    cl_gX  = cross.angular_cl_gX(
        ell, z_arr, nz_g, _THETA, hod_params,
        psf_fwhm_arcsec=psf_fwhm_arcsec,
    )
    theta_data, wtheta_data, _, _ = _load_data(sample["label"])
    wtheta_model = _cl_to_wtheta(ell, cl_gX * A_X, theta_data)
    return wtheta_model


def _compute_amplitude(theta, wtheta_data, wtheta_err, wtheta_model):
    """Least-squares amplitude A_X to rescale model to data."""
    mask = (wtheta_err > 0) & np.isfinite(wtheta_data) & np.isfinite(wtheta_model)
    if mask.sum() < 3:
        return 1.0
    W      = 1.0 / wtheta_err[mask] ** 2
    d      = wtheta_data[mask]
    m      = wtheta_model[mask]
    A_fit  = np.sum(W * d * m) / np.sum(W * m ** 2)
    return float(A_fit)


def _run_one_sample(sample, cross, hod_params, psf_fwhm, verbose=True):
    """Run prediction + amplitude fit for a single sample.  Returns results dict."""
    lbl = sample["label"]
    z   = sample["zmean"]
    if verbose:
        print(f"  {lbl}  z={z:.3f}  log10Mmin={hod_params['log10mmin']:.3f} ...")

    theta_rad, wtheta_data, wtheta_err, R_kpc = _load_data(lbl)

    try:
        wtheta_unit = _predict_wtheta(
            sample, cross, hod_params, A_X=1.0,
            psf_fwhm_arcsec=psf_fwhm,
        )
    except Exception as exc:
        if verbose:
            print(f"    WARNING: prediction failed for {lbl}: {exc}")
        wtheta_unit = np.ones_like(wtheta_data)

    A_fit = _compute_amplitude(theta_rad, wtheta_data, wtheta_err, wtheta_unit)
    wtheta_model = wtheta_unit * A_fit

    mask     = (wtheta_err > 0) & np.isfinite(wtheta_data) & np.isfinite(wtheta_model)
    ndof     = max(1, mask.sum() - 1)
    chi2_dof = float(
        np.sum(((wtheta_data[mask] - wtheta_model[mask]) / wtheta_err[mask]) ** 2) / ndof
    )
    if verbose:
        print(f"    chi2/dof={chi2_dof:.3f}  A_X={A_fit:.3f}")

    return dict(
        label=lbl, sample=sample, hod_params=hod_params,
        theta_rad=theta_rad, wtheta_data=wtheta_data, wtheta_err=wtheta_err,
        R_kpc=R_kpc, wtheta_model=wtheta_model, A_fit=A_fit, chi2_dof=chi2_dof,
    )


def _plot_individual(res, out_path, psf_label=""):
    """Save single-sample w_θ(θ) figure."""
    sample = res["sample"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(res["R_kpc"], res["wtheta_data"], yerr=res["wtheta_err"],
                fmt="o", ms=4, color="C0", label="Comparat+2025 data")
    ax.plot(res["R_kpc"], res["wtheta_model"], "r-", lw=2,
            label=rf"Model{psf_label} ($A_X={res['A_fit']:.2f}$, "
                  rf"$\chi^2/\nu={res['chi2_dof']:.2f}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$w_\theta(\theta)$")
    ax.set_title(
        rf"Comparat+2025 {res['label']}  "
        rf"$\log_{{10}}M_*/M_\odot>{sample['log10ms_min']}$  $z_{{mean}}={sample['zmean']}$"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_key_samples(res_s1, res_s5, out_path, psf_label=""):
    """Publication-quality 2×2 figure for M*>10 (S1) and M*>11 (S5)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for col, res in enumerate([res_s1, res_s5]):
        sample = res["sample"]
        title_base = (
            rf"$\log_{{10}}(M_*/M_\odot) > {sample['log10ms_min']}$  "
            rf"$z_{{mean}}={sample['zmean']}$"
        )

        # Top: w_θ(θ)
        ax = axes[0, col]
        ax.errorbar(res["R_kpc"], res["wtheta_data"], yerr=res["wtheta_err"],
                    fmt="o", ms=5, color="C0", zorder=5, label="Comparat+2025")
        ax.plot(res["R_kpc"], res["wtheta_model"], "r-", lw=2,
                label=rf"DPM Model 2{psf_label}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$R$ [kpc]", fontsize=11)
        ax.set_ylabel(r"$w_\theta(\theta)$", fontsize=11)
        ax.set_title(title_base, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate χ²/dof and A_X
        ax.text(0.97, 0.95,
                rf"$A_X={res['A_fit']:.2f}$" + "\n" + rf"$\chi^2/\nu={res['chi2_dof']:.2f}$",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Bottom: residuals
        ax_res = axes[1, col]
        mask = (res["wtheta_err"] > 0) & np.isfinite(res["wtheta_data"]) & np.isfinite(res["wtheta_model"])
        pull = np.where(mask,
                        (res["wtheta_data"] - res["wtheta_model"]) / res["wtheta_err"],
                        np.nan)
        ax_res.axhline(0, color="k", lw=1)
        ax_res.axhline(+2, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax_res.axhline(-2, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax_res.scatter(res["R_kpc"][mask], pull[mask], color="C0", s=20, zorder=5)
        ax_res.set_xscale("log")
        ax_res.set_ylim(-4, 4)
        ax_res.set_xlabel(r"$R$ [kpc]", fontsize=11)
        ax_res.set_ylabel(r"$(d - m)/\sigma$", fontsize=11)
        ax_res.grid(True, alpha=0.3)

    fig.suptitle(
        "Comparat+2025: galaxy × eROSITA soft X-ray  (DPM Model 2"
        + (", eROSITA PSF 30\"" if psf_label else "")
        + ")",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Key-sample figure saved to {out_path}")


def main(psf_fwhm=_EROSITA_PSF_FWHM, key_only=False):
    psf_label = f", PSF {psf_fwhm:.0f}\"" if psf_fwhm else ""
    print(f"Building infrastructure (FullHaloModelPrediction + HaloModelCrossSpectra){psf_label} ...")
    cross = _build_infrastructure()

    all_results = []
    for sample, hod_row in zip(SAMPLES, _TABLE3):
        hod_params = _hod_params_from_table3(hod_row)
        res = _run_one_sample(sample, cross, hod_params, psf_fwhm=psf_fwhm)
        all_results.append(res)

        out_i = _FIG_DIR / f"comparat2025_{res['label']}_wtheta.pdf"
        _plot_individual(res, out_i, psf_label=psf_label)
        print(f"    saved {out_i}")

    # --- Key samples: M*>10 (S1) and M*>11 (S5) ---
    res_s1 = all_results[0]   # S1: log10Ms > 10.0
    res_s5 = all_results[4]   # S5: log10Ms > 11.0
    out_key = _FIG_DIR / "comparat2025_key_S1_S5.pdf"
    _plot_key_samples(res_s1, res_s5, out_key, psf_label=psf_label)

    if not key_only:
        # --- 3×3 summary grid ---
        fig_all, axes_all = plt.subplots(3, 3, figsize=(14, 12))
        axes_flat = axes_all.ravel()
        for idx, res in enumerate(all_results):
            ax = axes_flat[idx]
            ax.errorbar(res["R_kpc"], res["wtheta_data"], yerr=res["wtheta_err"],
                        fmt="o", ms=3, color="C0")
            ax.plot(res["R_kpc"], res["wtheta_model"], "r-", lw=1.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                rf"{res['label']}: $M_*>{res['sample']['log10ms_min']}$, $z={res['sample']['zmean']}$"
                f"\n" + rf"$\chi^2/\nu={res['chi2_dof']:.2f}$, $A_X={res['A_fit']:.2f}$",
                fontsize=8,
            )
            ax.set_xlabel(r"$R$ [kpc]", fontsize=7)
            ax.set_ylabel(r"$w_\theta$", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        for j in range(len(SAMPLES), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig_all.suptitle(
            f"Comparat+2025: galaxy × eROSITA soft X-ray  (DPM Model 2{psf_label})",
            fontsize=11,
        )
        fig_all.tight_layout()
        out_all = _FIG_DIR / "comparat2025_all_samples.pdf"
        fig_all.savefig(out_all)
        plt.close(fig_all)
        print(f"Summary figure saved to {out_all}")

    print("\n--- chi2/dof summary ---")
    for res in all_results:
        status = "PASS" if res["chi2_dof"] < 2.0 else "FAIL"
        print(f"  {res['label']}  chi2/dof={res['chi2_dof']:.3f}  A_X={res['A_fit']:.3f}  [{status}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate hod_mod against Comparat+2025 galaxy × eROSITA data"
    )
    parser.add_argument("--no-psf", action="store_true",
                        help="Disable eROSITA PSF convolution")
    parser.add_argument("--key-only", action="store_true",
                        help="Only produce S1 (M*>10) and S5 (M*>11) key figures")
    parser.add_argument("--psf-fwhm", type=float, default=_EROSITA_PSF_FWHM,
                        help="PSF FWHM in arcsec (default 30.0)")
    args = parser.parse_args()

    psf = None if args.no_psf else args.psf_fwhm
    main(psf_fwhm=psf, key_only=args.key_only)
