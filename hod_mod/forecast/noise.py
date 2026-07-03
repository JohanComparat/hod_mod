r"""Physical survey-noise models for the tier-2 forecast.

Unlike the tier-1 effective ``(rN, aN)`` recipe (``run_stage4_forecast``), every
noise term here is derived from explicit survey specifications:

* :class:`ShearSurvey` — Euclid+LSST: shape noise σ_e²/n̄ per tomographic bin.
* :class:`CMBLensingSurvey` — S4-like flat N_L^{κκ}.
* :class:`AthenaAllSky` — hypothetical all-sky X-ray survey pinned by the
  completeness argument: F_lim(0.5–2 keV) = 2e-16 erg/s/cm² is exactly the
  depth that makes an L_X > 1e42 erg/s AGN sample complete to z = 1.  Photon
  (CXB) noise for the band C_ℓ, PSF beam, and the L_lim(z) completeness check.
* :class:`SpectroSurvey` — DESI/4MOST-like: pair-count + cosmic-variance errors
  for the per-cell w_p and ΔΣ, Poisson errors for the abundances/XLF.

All functions are numpy and evaluated once at the fiducial — consistent with
``fisher.fisher_matrix`` treating the covariance as parameter-independent.
Model-unit conversions (the forecast X-ray field carries arbitrary but
self-consistent units) are handled by expressing every noise as a
noise-to-signal ratio against fiducial model quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from hod_mod.core.distances import comoving_distance

_SR_PER_ARCMIN2 = (np.pi / (180.0 * 60.0)) ** 2
_MPC_CM = 3.0856775814913673e24
# c²/(4πG) in M⊙/Mpc; Σ_crit,com [h M⊙/pc²] = 1.6624e6·χs/(χl(χs−χl)(1+zl)),
# with all χ comoving in Mpc/h (the h's cancel in this combination).
_SIGCRIT_COEF = 1.6624e6


def chi_of(z, h, Om):
    """Comoving distance χ(z) [Mpc/h] (numpy scalar/array)."""
    z_arr = jnp.atleast_1d(jnp.asarray(z, dtype=float))
    # cast to float64: without jax_enable_x64 the JAX result is float32, which
    # overflows downstream (e.g. d_L² in cm²)
    out = np.asarray(comoving_distance(z_arr, h, Om), dtype=np.float64) * h
    return out[0] if np.ndim(z) == 0 else out


def shell_volume(z1, z2, h, Om, f_sky):
    """Comoving volume of the z-shell [(Mpc/h)³]."""
    c1, c2 = chi_of(z1, h, Om), chi_of(z2, h, Om)
    return f_sky * (4.0 * np.pi / 3.0) * (c2 ** 3 - c1 ** 3)


def n_modes(ell, f_sky):
    """Gaussian mode count per log-spaced ℓ bin (the stage4 convention)."""
    ell = np.asarray(ell, dtype=float)
    dlnl = np.log(ell[1] / ell[0]) if ell.size > 1 else 1.0
    return (2.0 * ell + 1.0) * ell * dlnl * f_sky


def knox_auto(ell, cl, noise_cl, f_sky):
    """Absolute Gaussian σ on an auto spectrum: √(2/N_modes)·(C+N)."""
    return np.sqrt(2.0 / n_modes(ell, f_sky)) * (np.asarray(cl) + noise_cl)


def knox_cross(ell, cl, cl_a, noise_a, cl_b, noise_b, f_sky):
    """Absolute Gaussian σ on a cross spectrum A×B."""
    var = (np.asarray(cl) ** 2
           + (np.asarray(cl_a) + noise_a) * (np.asarray(cl_b) + noise_b))
    return np.sqrt(var / n_modes(ell, f_sky))


# ---------------------------------------------------------------------------
# Surveys
# ---------------------------------------------------------------------------

@dataclass
class ShearSurvey:
    """Euclid+LSST combined shear: 30 sources/arcmin², f_sky = 0.5."""
    n_eff: float = 30.0          # total effective sources [arcmin⁻²]
    sigma_e: float = 0.26        # per-component shape dispersion
    f_sky: float = 0.5

    def n_bin_sr(self, n_bins=1):
        """Source density per tomographic bin [sr⁻¹] (equal-number split)."""
        return (self.n_eff / n_bins) / _SR_PER_ARCMIN2

    def noise_cl(self, n_bins=1):
        """Shape-noise power N_κ = σ_e²/n̄ per bin [sr]."""
        return self.sigma_e ** 2 / self.n_bin_sr(n_bins)


@dataclass
class CMBLensingSurvey:
    """S4-like CMB lensing: flat N_L^{κκ} (a good approximation at L < 3000)."""
    f_sky: float = 0.4
    n0: float = 7.0e-9           # N_L^{κκ} [dimensionless κ² sr]


@dataclass
class AthenaAllSky:
    """Hypothetical Athena all-sky survey (completeness-pinned depth).

    ``f_lim`` = 2e-16 erg/s/cm² (0.5–2 keV) makes L_X > 1e42 complete to z = 1
    (:meth:`l_lim`); the implied all-sky depth t·A_eff = n_det·ē/F_lim ≈ 8e7
    s·cm² is the stated optimistic premise.  The 5" HEW PSF matters through
    source detection/confusion (the XLF), not the ℓ ≤ 3000 band C_ℓ (beam ≈ 1).
    """
    f_lim: float = 2.0e-16       # point-source flux limit, 0.5–2 keV [erg/s/cm²]
    psf_hew: float = 5.0         # PSF half-energy width ["]
    f_sky: float = 0.65          # |b| > 10° extragalactic sky
    n_det_lim: float = 10.0      # counts for a detection at f_lim
    e_mean: float = 1.63e-9      # mean CXB photon energy 0.5–2 keV [erg] (Γ=1.4)
    i_cxb: float = 2.6e-8        # total 0.5–2 keV CXB intensity [erg/s/cm²/sr]
    gamma_cxb: float = 1.4       # CXB effective photon index (band partition)
    shell_cxb_frac: float = 0.05 # CXB energy-flux share of one Δz = 0.1 shell

    @property
    def exposure_area(self):
        """Effective survey depth t·A_eff [s·cm²] implied by the flux limit."""
        return self.n_det_lim * self.e_mean / self.f_lim

    def band_flux_fractions(self, bands):
        """CXB energy-flux fraction per band (Γ_cxb power law), Σ over 0.5–2 = 1."""
        p = 2.0 - self.gamma_cxb
        lo = np.array([b[0] for b in bands]); hi = np.array([b[1] for b in bands])
        return (hi ** p - lo ** p) / (2.0 ** p - 0.5 ** p)

    def photon_density(self, bands=None):
        """CXB photon surface density [sr⁻¹], total or per band."""
        n_tot = self.i_cxb * self.exposure_area / self.e_mean
        if bands is None:
            return n_tot
        p = 1.0 - self.gamma_cxb          # photon-flux weighting ∝ E^{-Γ}
        lo = np.array([b[0] for b in bands]); hi = np.array([b[1] for b in bands])
        frac = (hi ** p - lo ** p) / (2.0 ** p - 0.5 ** p)
        return n_tot * frac

    def beam(self, ell):
        """Gaussian beam b_ℓ for the PSF (HEW = 2.355 σ)."""
        sig = (self.psf_hew / 2.355) * np.pi / (180.0 * 3600.0)
        return np.exp(-0.5 * np.asarray(ell) ** 2 * sig ** 2)

    def l_lim(self, z, h, Om):
        """Faintest detectable L_X(0.5–2) [erg/s] at redshift z (completeness)."""
        d_l = (1.0 + z) * chi_of(z, h, Om) / h * _MPC_CM      # [cm]
        return 4.0 * np.pi * d_l ** 2 * self.f_lim

    def noise_cl_model(self, ibar_model, bands=None):
        """White photon-noise power in MODEL units, N_X = Ī_model²/(n_γ·s_x²).

        ``ibar_model`` is the fiducial mean intensity of the modeled shell field
        in the forecast's own X-ray units; ``s_x = shell_cxb_frac`` anchors that
        field to its physical share of the CXB, whose photon statistics set the
        map noise (per band when ``bands`` is given).
        """
        n_g = self.photon_density(bands)
        return np.asarray(ibar_model) ** 2 / (n_g * self.shell_cxb_frac ** 2)


@dataclass
class SpectroSurvey:
    """DESI/4MOST-like spectroscopy over the shear footprint."""
    f_sky: float = 0.5
    f_cv0: float = 0.005         # relative cosmic variance × [V/(Gpc/h)³]^{-1/2}
    pi_max: float = 100.0        # w_p projection depth [Mpc/h]
    ssfr_err: float = 0.05       # absolute σ on the mean MS log10 sSFR per cell [dex]

    def cv_rel(self, volume):
        """Relative cosmic-variance floor for a cell of ``volume`` (Mpc/h)³."""
        return self.f_cv0 / np.sqrt(volume / 1.0e9)


@dataclass
class RadioSurvey:
    """LOFAR/LoTSS-like radio continuum survey with redshift counterparts.

    ``nulnu_lim`` is the νL_ν detection threshold expressed at the fundamental
    plane's 5 GHz reference (a 144 MHz flux limit of ~0.8 mJy scaled with a
    ν^{-0.7} synchrotron spectrum) — the radio analogue of the Athena F_lim,
    driving the L_lim(z) completeness of the radio luminosity function.
    """
    f_sky: float = 0.13          # LoTSS-wide with photo/spec-z counterparts
    nulnu_lim: float = 3.0e-22   # νL_ν(5 GHz)-equivalent limit [erg/s/cm²]

    def l_lim(self, z, h, Om):
        """Faintest detectable 5 GHz νL_ν [erg/s] at redshift z."""
        d_l = (1.0 + z) * chi_of(z, h, Om) / h * _MPC_CM      # [cm]
        return 4.0 * np.pi * d_l ** 2 * self.nulnu_lim


@dataclass
class HISurvey:
    """Blind HI survey + 21 cm intensity mapping (ALFALFA/MIGHTEE → SKA1 era).

    The HIMF uses Poisson counts with an M_HI detection limit
    M_lim(z) = 2.36×10⁵ d_L² S_int (the standard 21 cm mass–flux relation);
    the 21 cm × galaxy cross uses the calibrated effective (rN, aN) recipe
    (the tSZ precedent) — a CHIME/MeerKLASS-era noise-to-signal.
    """
    f_sky: float = 0.16          # ALFALFA-like footprint for the HIMF
    s_int_lim: float = 0.6       # integrated-flux limit [Jy km/s]
    z_himf: float = 0.06         # depth of the LOCAL blind-HIMF volume
    f_sky_im: float = 0.1        # 21 cm IM × galaxies overlap
    rn_im: float = 2.0           # IM noise-to-signal at ℓ = 100
    an_im: float = 0.5           # and its ℓ growth exponent

    def mhi_lim(self, z, h, Om):
        """Faintest detectable M_HI [Msun/h] at redshift z (2.36e5 d_L² S)."""
        d_l = (1.0 + z) * chi_of(z, h, Om) / h                # [Mpc]
        return 2.36e5 * max(d_l, 1e-3) ** 2 * self.s_int_lim * h


# ---------------------------------------------------------------------------
# Per-observable noise
# ---------------------------------------------------------------------------

def _bin_edges(x):
    """Geometric bin edges around a log-spaced abscissa grid."""
    x = np.asarray(x, dtype=float)
    mid = np.sqrt(x[:-1] * x[1:])
    lo = np.concatenate([[x[0] ** 2 / mid[0]], mid])
    hi = np.concatenate([mid, [x[-1] ** 2 / mid[-1]]])
    return lo, hi


def wp_pair_sigma(rp, wp, ngal, volume, survey: SpectroSurvey):
    """Absolute σ on w_p(r_p) [Mpc/h]: pair-count Poisson + cosmic variance.

    N_pair = ½ n̄² V · π(r₂²−r₁²) · 2π_max per r_p bin;
    σ_shot = 2π_max (1 + w_p/2π_max) / √N_pair.
    """
    rp = np.asarray(rp, dtype=float); wp = np.asarray(wp, dtype=float)
    lo, hi = _bin_edges(rp)
    n_pair = 0.5 * ngal ** 2 * volume * np.pi * (hi ** 2 - lo ** 2) * 2.0 * survey.pi_max
    shot = 2.0 * survey.pi_max * (1.0 + wp / (2.0 * survey.pi_max)) / np.sqrt(n_pair)
    return np.sqrt(shot ** 2 + (survey.cv_rel(volume) * wp) ** 2)


def delta_sigma_noise(rp, ds, z_l, ngal, volume, h, Om,
                      zs_grid, nz_src, shear: ShearSurvey,
                      spectro: SpectroSurvey, dz_buffer=0.1):
    """Absolute σ on ΔΣ(r_p) [h M⊙/pc²]: lensing shape noise (+ CV floor).

    σ_shape = σ_e ⟨Σ_crit⁻¹⟩⁻¹ / √(n_src^eff · A_ann · N_lens), with the source
    density and ⟨Σ_crit⁻¹⟩ restricted to z_s > z_l + dz_buffer — high-z lens
    cells become noise-dominated automatically as the background empties.
    """
    rp = np.asarray(rp, dtype=float); ds = np.asarray(ds, dtype=float)
    zs = np.asarray(zs_grid, dtype=float); nz = np.asarray(nz_src, dtype=float)
    chi_l = chi_of(z_l, h, Om)
    chi_s = chi_of(zs, h, Om)
    behind = zs > (z_l + dz_buffer)
    frac_behind = np.trapezoid(nz * behind, zs)                # source fraction
    if frac_behind < 1e-6:
        return np.full(rp.shape, np.inf)
    inv_sc = np.where(behind & (chi_s > chi_l),
                      (chi_l * np.clip(chi_s - chi_l, 0.0, None) * (1.0 + z_l))
                      / (_SIGCRIT_COEF * np.clip(chi_s, 1e-9, None)), 0.0)
    mean_inv_sc = np.trapezoid(nz * inv_sc, zs) / frac_behind  # [pc²/(h M⊙)]
    n_src = shear.n_eff * frac_behind                          # [arcmin⁻²]
    lo, hi = _bin_edges(rp)
    rad2arcmin = 180.0 * 60.0 / np.pi
    a_ann = np.pi * ((hi / chi_l * rad2arcmin) ** 2 - (lo / chi_l * rad2arcmin) ** 2)
    n_lens = ngal * volume
    shape = shear.sigma_e / mean_inv_sc / np.sqrt(n_src * a_ann * n_lens)
    return np.sqrt(shape ** 2 + (spectro.cv_rel(volume) * ds) ** 2)


def band_mean_photon_energy(bands, gamma_cxb=1.4):
    """Mean photon energy per band [erg] for an E^{-Γ} photon spectrum."""
    out = []
    for lo, hi in bands:
        e = np.linspace(float(lo), float(hi), 256)
        w = e ** (-float(gamma_cxb))
        out.append(np.trapezoid(w * e, e) / np.trapezoid(w, e))
    return np.asarray(out) * 1.602e-9


def athena_noise_cl_model(ath: AthenaAllSky, bands, z_eff, chi1, chi2, h):
    """White photon-noise power for a shell's band C_ℓ^XX in MODEL units, (Nb,).

    The forecast's X-ray amplitudes are (L/10^45 erg/s) comoving emissivity
    densities, so the model→physical intensity conversion per unit comoving
    depth is conv = 10^45/(4π(1+z)⁴ (Mpc/h → cm)²).  Anchoring the shell field
    with a thin-shell window of depth Δχ = χ₂−χ₁ gives the (signal-independent)
    white-noise level

    .. math:: N^{\\rm model}_b = \\frac{I_{\\rm CXB,b}\\,\\bar e_b}
              {t A_{\\rm eff}\\; {\\rm conv}^2\\, \\Delta\\chi} .

    Divide by ``ath.beam(ell)**2`` for the beam-deconvolved noise (≈1 here).
    """
    conv = 1.0e45 / (4.0 * np.pi * (1.0 + z_eff) ** 4 * (_MPC_CM / h) ** 2)
    dchi = chi2 - chi1
    i_b = ath.i_cxb * ath.band_flux_fractions(bands)
    e_b = band_mean_photon_energy(bands, ath.gamma_cxb)
    return i_b * e_b / (ath.exposure_area * conv ** 2 * dchi)


def n2d_of(ngal, chi1, chi2):
    """Projected galaxy density [sr⁻¹] of a shell: n̄_g·(χ₂³−χ₁³)/3."""
    return ngal * (chi2 ** 3 - chi1 ** 3) / 3.0


def poisson_relerr(density, volume):
    """Relative Poisson error 1/√N for a count density × volume."""
    n = np.asarray(density, dtype=float) * volume
    return 1.0 / np.sqrt(np.maximum(n, 1e-300))


def xlf_relerr(phi, volume, dloglx=0.5):
    """Relative Poisson σ per XLF point: N = Φ·V·Δlog L_X."""
    return poisson_relerr(np.asarray(phi, dtype=float) * dloglx, volume)
