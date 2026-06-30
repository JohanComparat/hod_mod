"""Validate the analytic King-profile PSF model against eROSITA TM CalDB data.

Produces a three-panel figure:
  Panel A — radial PSF profiles: TM1–TM7 (individual + mean) vs King fit vs Gaussian
  Panel B — fractional residuals (TM mean − King) / TM mean
  Panel C — PSF window B_ℓ: Gaussian vs analytic King vs tabulated (showing truncation ringing)

Output: results/psf/erosita_psf_king_fit.png

Usage::

    python -m hod_mod.scripts.galaxies.plot_erosita_psf

The TM CalDB files are read from::

    /home/comparat/data/erosita/caldb_221121v03/caldb/srv-0500-2000/tm{1..7}_2dpsf_221121v03.fits
"""

from __future__ import annotations

import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import j0 as _j0
from hod_mod.paths import results_root

_CALDB = pathlib.Path(
    "/home/comparat/data/erosita/caldb_221121v03/caldb/srv-0500-2000"
)
_TM_IDS = [1, 2, 3, 4, 5, 6, 7]
_OUT_DIR = results_root() / "psf"
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


# ---------------------------------------------------------------------------
# PSF loading and radial profile extraction
# ---------------------------------------------------------------------------

def _load_tm_radial_profile(tm_id: int, rmax_arcsec: float = 200.0) -> tuple:
    """Load TM PSF image and return azimuthally-averaged radial profile.

    Returns
    -------
    rmid : arcsec bin centres
    profile : PSF values (normalized to peak at r≈0)
    """
    from astropy.io import fits

    fpath = _CALDB / f"tm{tm_id}_2dpsf_221121v03.fits"
    with fits.open(fpath) as hdul:
        data = hdul[1].data.astype(float)  # (480, 480), normalized, 1"/pix

    cx = cy = 240.0  # centre pixel (0-indexed)
    ny, nx = data.shape
    y_arr, x_arr = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((x_arr - cx) ** 2 + (y_arr - cy) ** 2)  # arcsec (1 pix = 1 arcsec)

    rbins = np.arange(0.5, rmax_arcsec + 1.0, 1.0)
    rmid = 0.5 * (rbins[:-1] + rbins[1:])
    profile = np.array([
        data[(r >= r1) & (r < r2)].mean() if ((r >= r1) & (r < r2)).any() else 0.0
        for r1, r2 in zip(rbins[:-1], rbins[1:])
    ])
    profile /= profile[0]  # normalize to 1 at r≈0
    return rmid, profile


# ---------------------------------------------------------------------------
# King profile model and Hankel B_ℓ
# ---------------------------------------------------------------------------

def _king(r, theta_c, alpha):
    return (1.0 + (r / theta_c) ** 2) ** (-alpha)


def _b_ell_king_analytic(ell, theta_c_arcsec, alpha=1.5):
    """Analytic King-profile PSF window in harmonic space."""
    tc_rad = theta_c_arcsec * _ARCSEC_TO_RAD
    x = ell * tc_rad
    if abs(alpha - 1.5) < 1e-6:
        return np.exp(-x)
    import math
    from scipy.special import kv as _kv
    nu = alpha - 1.0
    norm0 = 2.0 ** (nu - 1.0) * math.gamma(nu)
    prefac = 2.0 ** (2.0 - alpha) / math.gamma(nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bx = x ** nu * _kv(nu, x)
    Bx = np.where(x == 0.0, norm0, Bx)
    return prefac * Bx / norm0


def _b_ell_gaussian(ell, fwhm_arcsec):
    sigma_rad = fwhm_arcsec * _ARCSEC_TO_RAD / 2.355
    return np.exp(-0.5 * ell ** 2 * sigma_rad ** 2)


def _b_ell_tabulated(rmid, profile, ell):
    """Numerical Hankel transform of tabulated 1-D radial profile.

    B_ℓ = 2π ∫ PSF(θ) J₀(ℓ θ) θ dθ  (θ in radians)
    Normalized to B_0 = 1.
    """
    theta_rad = rmid * _ARCSEC_TO_RAD
    # Weight for azimuthal integration: 2π θ dθ
    dtheta = np.gradient(theta_rad)
    B = np.array([
        np.trapezoid(profile * _j0(ll * theta_rad) * theta_rad * 2.0 * np.pi, theta_rad)
        for ll in ell
    ])
    B /= B[0] if B[0] != 0 else 1.0
    return B


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all TM profiles ────────────────────────────────────────────
    profiles = {}
    for tm in _TM_IDS:
        fpath = _CALDB / f"tm{tm}_2dpsf_221121v03.fits"
        if not fpath.exists():
            print(f"  Warning: {fpath} not found, skipping TM{tm}")
            continue
        rmid, prof = _load_tm_radial_profile(tm)
        profiles[tm] = (rmid, prof)
        print(f"  Loaded TM{tm}")

    rmid_ref = profiles[_TM_IDS[0]][0]
    all_profs = np.array([profiles[t][1] for t in sorted(profiles)])
    mean_prof = all_profs.mean(axis=0)

    # ── 2. Fit King profile to TM mean ────────────────────────────────────
    fit_mask = (rmid_ref > 1.5) & (rmid_ref < 180.0) & (mean_prof > 0)
    popt, _ = curve_fit(
        _king, rmid_ref[fit_mask], mean_prof[fit_mask],
        p0=[8.0, 1.5], bounds=([0.5, 0.8], [50.0, 6.0]),
    )
    tc_fit, alpha_fit = popt
    king_fit = _king(rmid_ref, tc_fit, alpha_fit)

    # FWHM: (1 + (FWHM/2/tc)^2)^(-alpha) = 0.5  →  FWHM/2 = tc*sqrt(2^(1/alpha)-1)
    fwhm_king = 2.0 * tc_fit * np.sqrt(2.0 ** (1.0 / alpha_fit) - 1.0)
    gauss_fwhm = fwhm_king
    gauss_profile = np.exp(-0.5 * (rmid_ref / (gauss_fwhm / 2.355)) ** 2)

    print(f"\nKing fit:  θ_c = {tc_fit:.2f}\", α = {alpha_fit:.3f}, FWHM = {fwhm_king:.1f}\"")
    residuals_frac = (mean_prof - king_fit) / np.where(mean_prof > 0, mean_prof, np.nan)

    # ── 3. B_ℓ comparison ─────────────────────────────────────────────────
    ell = np.logspace(1, 4.5, 300)
    Bk = _b_ell_king_analytic(ell, tc_fit, alpha_fit)
    Bg = _b_ell_gaussian(ell, gauss_fwhm)
    Btab = _b_ell_tabulated(rmid_ref, mean_prof, ell)

    # ── 4. Figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_tm = plt.cm.tab10(np.linspace(0, 0.7, len(profiles)))

    # Panel A: real-space profiles
    ax = axes[0]
    for (tm, (rm, pr)), col in zip(sorted(profiles.items()), colors_tm):
        ax.loglog(rm, pr, color=col, lw=0.8, alpha=0.6, label=f"TM{tm}")
    ax.loglog(rmid_ref, mean_prof, "k-",  lw=2.0, label="TM mean")
    ax.loglog(rmid_ref, king_fit,  "r-",  lw=2.0,
              label=rf"King fit  $\theta_c={tc_fit:.1f}''$, $\alpha={alpha_fit:.2f}$")
    ax.loglog(rmid_ref, gauss_profile, "b--", lw=1.5,
              label=rf"Gaussian  FWHM$={gauss_fwhm:.1f}''$")
    ax.set_xlabel(r"$\theta$ [arcsec]")
    ax.set_ylabel("PSF (normalized)")
    ax.set_title("eROSITA TM CalDB PSF radial profile")
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-5, 2)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # Panel B: fractional residuals
    ax = axes[1]
    ax.axhline(0,    color="k", lw=1.0)
    ax.axhline(+0.1, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-0.1, color="k", lw=0.8, ls="--", alpha=0.5, label=r"$\pm10\%$")
    for (tm, (rm, pr)), col in zip(sorted(profiles.items()), colors_tm):
        res = (pr - _king(rm, tc_fit, alpha_fit)) / np.where(pr > 1e-8, pr, np.nan)
        ax.semilogx(rm, res, color=col, lw=0.8, alpha=0.6)
    ax.semilogx(rmid_ref, residuals_frac, "k-", lw=2.0, label="TM mean residual")
    ax.set_xlabel(r"$\theta$ [arcsec]")
    ax.set_ylabel(r"(TM $-$ King) / TM")
    ax.set_title("Fractional residuals: TM mean vs King fit")
    ax.set_xlim(1, 200)
    ax.set_ylim(-0.5, 0.5)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Panel C: B_ℓ comparison
    ax = axes[2]
    ax.loglog(ell, Bk,   "r-",  lw=2.0, label=rf"King analytic ($\alpha={alpha_fit:.2f}$)")
    ax.loglog(ell, Bg,   "b--", lw=1.5, label=rf"Gaussian FWHM$={gauss_fwhm:.1f}''$")
    ax.loglog(ell, np.abs(Btab), "k:", lw=1.5, label="Tabulated (truncated, |B_ℓ|)")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$B_\ell$")
    ax.set_title(r"PSF window $B_\ell$ in harmonic space")
    ax.set_xlim(10, 3e4)
    ax.set_ylim(1e-4, 1.5)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"eROSITA PSF: King-profile analytic model vs TM CalDB (srv 0.5–2 keV)\n"
        f"Fitted:  $\\theta_c = {tc_fit:.2f}''$,  $\\alpha = {alpha_fit:.3f}$,  "
        f"FWHM $= {fwhm_king:.1f}''$",
        fontsize=11,
    )
    fig.tight_layout()
    out = _OUT_DIR / "erosita_psf_king_fit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    make_figure()
