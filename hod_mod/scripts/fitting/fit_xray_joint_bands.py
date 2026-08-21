"""Phase B v3 — energy-band gas fit on the NATIVE DPM gas parameters.

The gas is parametrised by the DPM (Oppenheimer+2025) density and pressure profiles
themselves, replacing the free LX–M / kT–M scaling relations of v2:

* n_e(r,M,z) = n_e,0.3 · f_n(x) · E(z)^{γ_n} · M_12^{β_n}
* P (r,M,z) = P_0.3   · f_P(x) · E(z)^{γ_P} · M_12^{β_P}
* T (r,M,z) = P/n_e   — the X-ray temperature IS the DPM temperature

Free params ``[log10_ne03, beta_n, log10_p03, beta_P, p2, r_max, log10DC, z_metal,
agn_gamma]``.  The point of the re-base: **the same four gas parameters drive the
X-ray bands and the tSZ Σ_y** (:mod:`hod_mod.fitting.sz_transfer` integrates the very
same P), so a joint X-ray×SZ fit constrains one gas model rather than two disjoint ones.
``--sz`` activates that SZ leg: the Das et al. 2023 stacked Compton-y profiles
(``data/das_2023/``) enter the per-sample likelihood through a precomputed
:func:`~hod_mod.fitting.sz_transfer.build_sz_transfer` kernel, rescaled analytically
in ``(P_0.3, β_P)`` at MCMC speed.

Forward model: the band-b luminosity of a halo is

    L_X,b(M) = n_e,0.3² M_12^{2β_n} E^{2γ_n} · V_shape(M|p2,r_max) · J_b(T_0(M), Z)

where ``J_b`` (:mod:`hod_mod.fitting.dpm_bands`) integrates Λ_b over DPM's **radial**
temperature profile.  Because the radial shape of T is fixed by the DPM slopes, this
factorises exactly, so J_b is tabulated once per (p2, r_max) and interpolated — the
weight stays a cheap analytic array, folded through the PRECOMPUTED per-mass transfer
``G(θ,M | p2,r_max)`` (Limber + Hankel of the normalised n_e²-shape FT with the
halo-model mass kernel; reproduces a direct ``angular_cl_gX`` to ~5e-5).  The eROSITA
per-band response weight ``A_b`` multiplies gas and AGN; the empirical S1 anchor sets
the absolute amplitude, which is why the native re-base does not need the
n_e,0.3 → erg/s → counts chain re-derived.  No per-eval FT ⇒ fast MCMC.

.. warning::

   This is **not** the v2 isothermal model reparametrised.  Integrating Λ_b over the
   radial T profile differs from a single Λ_b(kT_ew) by ~14%, so fitted gas parameters
   shift relative to the v2 baseline.  That difference is physics, not a regression.
   Requires float64 (``JAX_ENABLE_X64=1``): the emission integral carries r_cm² ~ 1e49.

Usage:
    HOD_MOD_DATA_DIR=/home/comparat/data HOD_MOD_RESULTS=/home/comparat/data/hod_mod_results \
      JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.fit_xray_joint_bands \
      --samples S1 S2 S3 S4 --mcmc

    # X-ray x SZ joint fit (adds the Das et al. 2023 Sigma_y leg on S4/S5):
    ... fit_xray_joint_bands --samples S1 S4 S5 --sz --mcmc
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, lsq_linear

from hod_mod import paths
from hod_mod.core.power_spectrum import LinearPowerSpectrum, default_pk_linear
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.core.distances import hubble_e, comoving_distance
from hod_mod.gas import GasDensityDPM, PressureProfileDPM
from hod_mod.gas.cooling import ApecCoolingTable
from hod_mod.gas.conversions import m200_to_m500c
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra, psf_king_window_ell
from hod_mod.agn.duty_cycle import DutyCycleAGNModel
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts.fitting import fit_agn_duty_cycle_baseline as B
from hod_mod.scripts.fitting import fit_xray_joint as J
from hod_mod.scripts.validate_gas_profiles import _make_density_variant, _rho_crit_z
from hod_mod.gas.conversions import _MPC_CM
from hod_mod.fitting.dpm_bands import (
    build_j_table, t0_of_mass, emission_measure_factor, shape_integral, v_shape_of_mass)
from hod_mod.fitting.sz_transfer import build_sz_transfer, predict_sigma_y

_OUT_DIR = os.fspath(paths.results_root() / "xray_joint_bands")

_BANDS = [f"{lo:04d}_E_{lo+100:04d}" for lo in range(500, 2000, 100)]
_BAND_EDGES = [(lo / 1000.0, (lo + 100) / 1000.0) for lo in range(500, 2000, 100)]
_NB = len(_BANDS)
_GAMMA_AGN = 1.8

# profile-shape grid (the scaling relations are analytic, NOT gridded)
_ALPHA_PROF = 0.9
_P2_GRID   = np.array([0.1, 0.6, 1.2, 2.4])   # gas outer slope
_RMAX_GRID = np.array([3.0, 4.0, 5.0])         # r_max / r200
_Z_METAL   = 0.3                                # representative gas metallicity [Z_sun]

# Scaling-relation pivots.  kt_norm is PIVOTED at M500c=10^14 (near the emission-
# weighted mass) — NOT at M500c=1 — so kt_norm ≡ log10(kT/E^{2/3}) at 10^14 keV and
# does NOT trade off with kt_slope (the M500c=1 pivot amplified the slope ~10x at the
# data mass and drove kT to unphysical values).  GAS.py at 10^14: 0.6·14−8 = +0.4.
_KT_PIVOT = 14.0
_LX_PIVOT = 15.0

# DPM scale-radius convention (R_s = R_200/c_DPM), shared with GasDensityDPM /
# PressureProfileDPM.  Native-DPM fiducials from the joint Lx+kT calibration in
# validate_gas_profiles._calibrate_ne03_P03 (beta_n=0.20, beta_P=0.80 -> the
# GAS.py alpha_Lx=1.70 / alpha_kT=0.60 targets), also used by
# hod_mod.scripts.direct_prediction_gal_gas_agn.
_C_DPM_REF   = 2.772
_NE03_FID    = 1.260e-5     # cm^-3      -> log10 = -4.899
_P03_FID     = 1.627e-6     # keV cm^-3  -> log10 = -5.789
_BETA_N_FID  = 0.20
_BETA_P_FID  = 0.80


def kT_of_M(log10_m500c, ez, kt_slope, kt_norm):
    """kT(M500c) [keV].  kt_norm = log10(kT/E(z)^{2/3}) at M500c = 10^_KT_PIVOT."""
    return 10.0 ** (kt_slope * (np.asarray(log10_m500c) - _KT_PIVOT) + kt_norm) * ez ** (2.0 / 3.0)


def LX_of_M(log10_m500c, ez, lx_norm, lx_slope, boost=1.0):
    """LX_0.5-2(M500c) [erg/s].  lx_norm at M500c = 10^_LX_PIVOT."""
    return 10.0 ** (lx_norm + lx_slope * (np.asarray(log10_m500c) - _LX_PIVOT)) * ez ** 2 * boost


# 9 free params, NATIVE DPM gas sector (Oppenheimer+2025) + shape + AGN + metallicity.
#
# The former phenomenological LX-M / kT-M power laws (lx_norm, lx_slope, kt_norm,
# kt_slope) are REPLACED by the DPM density and pressure normalisations/slopes:
#
#   n_e(r,M,z) = n_e,0.3 f_n(x) E(z)^{gamma_n} M_12^{beta_n}
#   P (r,M,z) = P_0.3   f_P(x) E(z)^{gamma_P} M_12^{beta_P}
#   T (r,M,z) = P/n_e                       <- the X-ray temperature IS the DPM one
#
# so the SAME four parameters drive the X-ray bands (through n_e^2 and T = P/n_e)
# and the tSZ Sigma_y (through P; see hod_mod.fitting.sz_transfer).  Vary once,
# both observables move -- this is the X-ray <-> SZ coupling of the joint fit.
#
# NOTE this is *not* the old isothermal model reparametrised: the DPM temperature
# has a radial profile, and integrating Lambda_b over it differs from a single
# Lambda_b(kT_ew) by ~14% (see hod_mod.fitting.dpm_bands).  Fitted gas parameters
# are expected to shift relative to the pre-DPM baseline.
_PARAMS   = ["log10_ne03", "beta_n", "log10_p03", "beta_P", "p2", "r_max", "log10DC",
             "z_metal", "agn_gamma"]
_BOOST_LX  = float(np.exp(0.5 * (np.log(10.0) * 0.3) ** 2))   # log-normal mean boost
_LOG10DC_LO, _LOG10DC_HI = B._LOG10DC_LO, B._LOG10DC_HI
_Z_FID = 0.3                                                  # fiducial gas metallicity [Z_sun]

# --- native-DPM gas prior, back-propagated from the GAS.py scaling relations ------
#
# The informative priors are physical statements about the R500c-integrated LX-M and
# kT-M relations.  Re-basing onto the native DPM parameters does not destroy that
# information, so rather than invent priors on (log10_ne03, beta_n, log10_p03,
# beta_P) we map the scaling-relation prior through the model with JAX autodiff:
#
#   theta_s = f(theta_n),  J = df/dtheta_n,  Sigma_n = J^-1 Sigma_s J^-T
#
# The result is a FULL-COVARIANCE Gaussian: the induced correlations are 0.71-0.95
# (e.g. rho(log10_ne03, beta_n) = -0.95, rho(log10_ne03, log10_p03) = +0.84 -- the
# latter is the T = P/n_e ratio constraint forcing the two normalisations to move
# together).  A diagonal prior cannot represent any of that.  See
# :mod:`hod_mod.fitting.dpm_priors`.
_PRIOR_Z = 0.135          # reference z (the relations are E(z)-normalised -> weak dep.)
_GAS_PRIOR_WIDEN = 1.5    # inflate the induced sigmas (Sigma scales as widen^2)
_GAS_PRIOR = None         # dict(mu=(4,), icov=(4,4)), set by _apply_candidate


def _induced_gas_prior(widen=None):
    """(mu, cov) on [log10_ne03, beta_n, log10_p03, beta_P] from the GAS.py priors."""
    widen = _GAS_PRIOR_WIDEN if widen is None else float(widen)
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, "dpm_induced_prior.npz")
    mu_s = np.array([44.7, 1.61, 0.4, 0.6])
    sig_s = np.array([0.3, 0.3, 0.2, 0.15])
    if os.path.exists(cache):
        d = np.load(cache)
        # Validate everything the mapping depends on, not just z: the cache stores
        # mu_s/sig_s precisely so a change to the GAS.py priors cannot be silently
        # ignored.  (Before this, editing mu_s or sig_s below reused a stale prior.)
        if (float(d["z"]) == _PRIOR_Z
                and "mu_s" in d.files and np.array_equal(d["mu_s"], mu_s)
                and "sig_s" in d.files and np.array_equal(d["sig_s"], sig_s)):
            return d["mu"], d["cov"] * widen ** 2
        print("  induced gas prior cache is stale (z/mu_s/sig_s changed) -> rebuild",
              flush=True)
    import jax
    jax.config.update("jax_enable_x64", True)   # emission integral carries r_cm^2 ~ 1e49
    from hod_mod.fitting.dpm_priors import ScalingCtx, induced_gaussian_prior
    from hod_mod.scripts.validate_gas_profiles import _r200, _c200_approx, _ez

    z = _PRIOR_Z
    h = float(F._THETA_COSMO["h"])
    m200 = np.geomspace(3e13, 2e15, 12) * h          # the group->cluster range the priors describe
    r200 = np.array([_r200(m, z) for m in m200])
    c200 = np.array([_c200_approx(m) for m in m200])
    m500c, r500c = m200_to_m500c(m200, c200, r200, _rho_crit_z(z))
    shape = dict(a_in_n=1.0, a_tr_n=1.9, a_out_n=2.7, gamma_n=2.0,       # DPM model 2 density
                 a_in_p=0.3, a_tr_p=1.3, a_out_p=4.1, gamma_p=8.0 / 3.0)  # DPM model 2 pressure
    _, cool_broad = _band_cooling()
    ctx = ScalingCtx(m200, r200, np.asarray(m500c), np.asarray(r500c), cool_broad,
                     shape, z, float(_ez(z)), h, z_metal=_Z_FID, t_min=_T_MIN_XRAY)
    theta0 = np.array([np.log10(_NE03_FID), _BETA_N_FID, np.log10(_P03_FID), _BETA_P_FID])
    # GAS.py scaling-relation priors (the v2 informative priors, unchanged).
    # mu_s / sig_s are defined at the top of this function so the cache check above
    # can compare against the values actually in force.
    mu_n, cov_n, J, f_at = induced_gaussian_prior(ctx, mu_s, sig_s, theta0)
    np.savez(cache, mu=mu_n, cov=cov_n, J=J, f_at=f_at, z=z, mu_s=mu_s, sig_s=sig_s)
    print(f"  induced native-DPM gas prior -> {cache}", flush=True)
    return mu_n, cov_n * widen ** 2


# --- candidate machinery: per-candidate Gaussian priors + bounds (8-vectors) ------
# Each candidate isolates one hypothesis for the "gas runs hot" result and writes to
# its own subfolder.  sig=inf ⇒ flat prior (bounds only); a tight sig ⇒ effectively
# fixed.  Set by _apply_candidate() into module globals used by _log_prior/_weight.
_CANDIDATES = ["baseline", "agn_fixed", "free_metal", "flat_kt"]
_MU8 = _SIG8 = _BND8 = None


def _apply_candidate(cand, dc_fix=-1.8):
    """Configure the priors/bounds for a candidate; returns the output subdir name."""
    global _MU8, _SIG8, _BND8, _GAS_PRIOR
    inf = np.inf
    # The 4 native-DPM gas params carry the FULL-COVARIANCE prior back-propagated
    # from the GAS.py scaling relations (_induced_gas_prior); their entries in the
    # diagonal sig-vector are therefore inf, and _log_prior adds the covariant term.
    gas_widen = 8.0 if cand == "flat_kt" else _GAS_PRIOR_WIDEN
    mu_n, cov_n = _induced_gas_prior(widen=gas_widen)
    _GAS_PRIOR = dict(mu=np.asarray(mu_n, float), icov=np.linalg.inv(np.asarray(cov_n, float)))

    # 9th param agn_gamma: AGN/continuum photon index, free with a Gaussian prior
    # around 1.8 so the continuum (not hot gas) can absorb a flat band spectrum.
    mu  = np.array([mu_n[0], mu_n[1], mu_n[2], mu_n[3],
                    0.0, 0.0, 0.0, _Z_FID, _GAMMA_AGN])
    sig = np.array([inf, inf, inf, inf, inf, inf, inf, 0.005, 0.3])  # gas: covariant; DC flat
    # log10_p03 upper bound is deliberately loose: the X-ray-only fit rails a
    # tighter bound (the known "gas runs hot" pull), and a bound doing the
    # constraining is not a fit — let the induced prior and the SZ leg decide.
    bnd = np.array([[-7.0, -3.0], [-0.4, 1.2], [-8.0, -2.5], [0.0, 1.8],
                    [_P2_GRID[0], _P2_GRID[-1]], [_RMAX_GRID[0], _RMAX_GRID[-1]],
                    [_LOG10DC_LO, _LOG10DC_HI], [0.05, 1.0], [1.2, 2.6]])
    if cand == "agn_fixed":          # fix AGN duty cycle to the Phase-A broad-band value
        mu[6] = dc_fix; sig[6] = 0.02; bnd[6] = [dc_fix - 0.1, dc_fix + 0.1]
    elif cand == "free_metal":       # free the gas metallicity (flat in [0.05, 1.0])
        sig[7] = inf
    elif cand == "flat_kt":          # relax the gas prior -> data-driven T = P/n_e
        pass                         # handled by gas_widen above
    elif cand != "baseline":
        raise ValueError(f"unknown candidate {cand}; choose from {_CANDIDATES}")
    _MU8, _SIG8, _BND8 = mu, sig, bnd
    return "baseline" if cand == "baseline" else cand


# --- band data + cooling + AGN spectral split + eROSITA response -------------

def _basename(label):
    return F._zenodo_fname(label).name.replace("_GALxEVT_wtheta.fits", "")


def load_band_data(label):
    """Per-band reconstructed w_b(theta): (Nb, Ntheta) wtheta/err + theta grid."""
    root = paths.data_path("xray_bands", _basename(label))
    rows = []; th_deg = None
    for band in _BANDS:
        fp = os.fspath(root / (band + ".fits"))
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"missing band file: {fp}\n"
                                    f"set $HOD_MOD_DATA_DIR to the data root.")
        t = Table.read(fp)
        if th_deg is None:
            th_deg = np.asarray(t["theta_mid"], float)
        rows.append((np.asarray(t["wtheta"], float), np.asarray(t["wtheta_err"], float)))
    return dict(theta_deg=th_deg, theta_arcsec=th_deg * 3600.0,
                theta_rad=th_deg * np.pi / 180.0,
                wtheta=np.vstack([r[0] for r in rows]),
                wtheta_err=np.vstack([r[1] for r in rows]))


_COOLING_CACHE = None
def _band_cooling():
    """15 per-band + 1 broad (0.5-2 keV) ApecCoolingTable (built once)."""
    global _COOLING_CACHE
    if _COOLING_CACHE is None:
        bands = [ApecCoolingTable(emin=lo, emax=hi) for lo, hi in _BAND_EDGES]
        broad = ApecCoolingTable(emin=0.5, emax=2.0)
        _COOLING_CACHE = (bands, broad)
    return _COOLING_CACHE


def _agn_band_fractions(gamma=_GAMMA_AGN):
    """Energy-flux fraction of a Γ power-law AGN in each band (Σ f_b = 1)."""
    p = 2.0 - gamma
    def _integ(lo, hi):
        return np.log(hi / lo) if abs(p) < 1e-9 else (hi ** p - lo ** p) / p
    tot = _integ(0.5, 2.0)
    return np.array([_integ(lo, hi) / tot for lo, hi in _BAND_EDGES])


_ARF_CACHE = None
def _arf_band_weights():
    """Per-band eROSITA effective-response weight A_b = <ARF·g_inband>_band (mean 1)."""
    global _ARF_CACHE
    if _ARF_CACHE is None:
        from hod_mod.gas import erosita_response as ER
        d = np.load(ER._DEFAULT_NPZ)
        emid = 0.5 * (d["energ_lo"] + d["energ_hi"])
        resp = d["arf_comb"] * d["g_inband"]
        A = np.array([np.mean(resp[(emid >= lo) & (emid < hi)]) for lo, hi in _BAND_EDGES])
        _ARF_CACHE = A / A.mean()
    return _ARF_CACHE


# --- per-mass transfer G(theta, M) (the validated core) ---------------------

def _shape_ft(dp, sc, z, theta_cosmo):
    """Normalised n_e²-shape FT Ŝ(k,M) = emissivity_uk(k,M)/emissivity_uk(k→0,M);
    cancels the density amplitude AND mass slope (those are now LX(M))."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        em = np.asarray(dp.emissivity_uk(sc["k_np"], sc["m_np"], sc["r_delta"],
                                         float(z), theta_cosmo))
    return em / em[0:1, :]


def _build_transfer(cross, hod, theta_cosmo, z_arr, nz, th_rad, ell, dp):
    """G(θ,M) (Ntheta, NM): w_gas(θ) = weight(M) @ G.  Linear-interp Limber + Hankel
    of the halo-model mass kernel × Ŝ, with the trapezoid dM measure folded in."""
    h = float(theta_cosmo["h"]); om = float(theta_cosmo["Omega_m"])
    chi_z = np.array([float(np.asarray(comoving_distance(float(zi), h, om)).ravel()[0]) * h
                      for zi in z_arr])
    dndchi = np.asarray(nz, float) / np.trapezoid(np.asarray(nz, float), chi_z)
    ell = np.asarray(ell, float)
    k_lim = (ell[:, None] + 0.5) / chi_z[None, :]           # (Nell, Nz)
    per_z = []
    m_np = None
    for iz, zi in enumerate(z_arr):
        sc = cross._get_static_cache(float(zi), theta_cosmo, hod)
        k_np = np.asarray(sc["k_np"], float); m_np = np.asarray(sc["m_np"], float)
        dndm = np.asarray(sc["dndm_np"], float); bias = np.asarray(sc["bias_np"], float)
        pk = np.asarray(sc["pk_lin"], float); uk = np.asarray(sc["uk"], float)
        nc, ns, n_gal, b_eff = cross._get_hod_weights(float(zi), theta_cosmo, hod, sc)
        nc = np.asarray(nc, float); ns = np.asarray(ns, float)
        S = _shape_ft(dp, sc, zi, theta_cosmo)             # (Nk, NM)
        Pk = (dndm[None, :] * (nc[None, :] + ns[None, :] * uk) * S / n_gal
              + dndm[None, :] * bias[None, :] * S * (b_eff * pk[:, None]))   # (Nk, NM)
        klz = k_lim[:, iz]
        Pint = np.stack([np.interp(klz, k_np, Pk[:, j]) for j in range(Pk.shape[1])], axis=1)
        per_z.append(dndchi[iz] * Pint / chi_z[iz] ** 2)   # (Nell, NM)
    Cl_M = np.trapezoid(np.stack(per_z, 0), chi_z, axis=0)  # (Nell, NM)
    Cl_M = Cl_M * np.asarray(psf_king_window_ell(ell, F._PSF_KING_THETA_C, 1.5))[:, None]
    # vectorised Hankel: G[θ,M] = Σ_ℓ (ℓ w_ℓ / 2π) j0(ℓθ) Cl_M[ℓ,M]
    from scipy.special import j0 as _j0
    wl = np.empty_like(ell)
    wl[1:-1] = (ell[2:] - ell[:-2]) / 2.0; wl[0] = (ell[1] - ell[0]) / 2.0
    wl[-1] = (ell[-1] - ell[-2]) / 2.0
    J0w = (ell * wl / (2.0 * np.pi))[None, :] * _j0(ell[None, :] * np.asarray(th_rad)[:, None])
    G = J0w @ Cl_M                                          # (Ntheta, NM)
    wtrap = np.empty_like(m_np)
    wtrap[1:-1] = (m_np[2:] - m_np[:-2]) / 2.0; wtrap[0] = (m_np[1] - m_np[0]) / 2.0
    wtrap[-1] = (m_np[-1] - m_np[-2]) / 2.0
    return G * wtrap[None, :], m_np


def _precompute(sample, hmf_backend):
    """Build (or load) the per-sample transfer grid G(θ,M | p2,r_max) + m500c(M) +
    E(z) + AGN template + band data."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, f"{sample}_transfer.npz")
    bd = load_band_data(sample)
    th_as = bd["theta_arcsec"]; th_rad = bd["theta_rad"]
    mask = (th_as >= 8.0) & (th_as <= 300.0)
    if os.path.exists(cache):
        d = np.load(cache)
        # "m200"/"r200" were added with the native-DPM band weight (dpm_bands): a
        # cache written before that lacks them and must be rebuilt.
        if (np.array_equal(d["p2_grid"], _P2_GRID) and np.array_equal(d["rmax_grid"], _RMAX_GRID)
                and d["G_grid"].shape[-2] == th_as.size
                and "m200" in d.files and "r200" in d.files):
            return (d["G_grid"], d["log10_m500c"], float(d["ez"]), d["agn_dc1"], bd, mask,
                    d["m200"], d["r200"])
        print(f"  [{sample}] cached transfer stale -> rebuild", flush=True)

    th = F._THETA_COSMO
    # default_pk_linear(), not LinearPowerSpectrum(): full_joint.py drives this via
    # XB._precompute() while building its galaxy sector from default_pk_linear(), so
    # hard-coding CAMB here would mix two different P(k) inside one likelihood.
    pk = default_pk_linear(); hmf = make_hmf(hmf_backend, pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"], Ob0=th["Omega_b"],
                sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    hod_params = B._build_hod_params(sample)
    z_arr, nz = F._build_nz_fast(sample)
    zmean = float(F.SAMPLES[sample]["zmean"])
    cross = HaloModelCrossSpectra(fhmp, density_profile=GasDensityDPM(model=2))

    # AGN template (DC=1)
    agn = DutyCycleAGNModel(sample=sample, theta_cosmo=th, hmf=hmf, log10DC=0.0)
    cross_a = HaloModelCrossSpectra(fhmp, density_profile=GasDensityDPM(model=2), agn_model=agn)
    comp = cross_a.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                 psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                 return_components=True, agn_kwargs={"log10DC": 0.0}, n_workers=1)
    agn_dc1 = F._hankel(np.asarray(comp["agn"], float), th_rad)

    # m500c(M) at zmean + E(zmean) (relation inputs; constant-per-halo)
    sc0 = cross._get_static_cache(zmean, th, hod_params)
    m_np = np.asarray(sc0["m_np"], float); c_np = np.asarray(sc0["c_np"], float)
    r_delta = np.asarray(sc0["r_delta"], float)
    m500c_h, _ = m200_to_m500c(m_np, c_np, r_delta, _rho_crit_z(zmean))
    log10_m500c = np.log10(np.asarray(m500c_h, float) / th["h"])   # Msun (physical)
    ez = float(hubble_e(zmean, th["Omega_m"]))

    G_grid = np.zeros((_P2_GRID.size, _RMAX_GRID.size, th_as.size, m_np.size))
    t0 = time.time()
    for i, p2 in enumerate(_P2_GRID):
        for j, rmax in enumerate(_RMAX_GRID):
            dp = _make_density_variant(model=2, ne_03=1e-4, beta=0.5, alpha_in=_ALPHA_PROF,
                                       alpha_tr=2.0, alpha_out=_ALPHA_PROF + 2.0 * float(p2))
            dp._r_max_factor = float(rmax)
            G_grid[i, j], _ = _build_transfer(cross, hod_params, th, z_arr, nz, th_rad, F._ELL, dp)
        print(f"  [{sample}] transfer p2={p2:.2f} ({(i+1)*_RMAX_GRID.size}/"
              f"{_P2_GRID.size*_RMAX_GRID.size}) [{time.time()-t0:.0f}s]", flush=True)
    # m200/r200 are the native-DPM inputs (M_12 and R_s = R_200/c_DPM); the DPM
    # profiles are M200-based, unlike the M500c-based scaling relations they replace.
    np.savez(cache, G_grid=G_grid, log10_m500c=log10_m500c, ez=ez, agn_dc1=agn_dc1,
             p2_grid=_P2_GRID, rmax_grid=_RMAX_GRID, m200=m_np, r200=r_delta)
    print(f"[{sample}] transfer built in {time.time()-t0:.0f}s -> {cache}", flush=True)
    return G_grid, log10_m500c, ez, agn_dc1, bd, mask, m_np, r_delta


# --- tSZ Sigma_y leg (opt-in --sz): Das et al. 2023 stacked Compton-y --------
#
# The digitized profiles (data/das_2023/, r in units of R200, y scaled by 1e8)
# are per stellar-mass BIN, while the fit samples are stellar-mass THRESHOLD
# samples; the mapping below pairs each bin with the threshold sample sharing
# its lower edge (the steep SMF makes the bin dominate the threshold counts).
# fig_B1 [10.9, 11.2] straddles the S4/S5 thresholds and stays unmapped.
#
# Model side: predict_sigma_y rescales a precomputed kernel analytically in
# (P_0.3, beta_P) — the SAME two parameters that set T = P/n_e in the X-ray
# bands, which is the entire point of the joint fit.  The pressure radial
# SHAPE is the fixed DPM model-2 form (p2/r_max in this fit deform only the
# density); the SZ leg therefore constrains amplitude and mass slope only.
#
# Beam and averaging conventions (Das, Chiang & Mathur 2023, ApJ 951, 125):
# the fitted/plotted profiles (their Figs. 5 and B1) come from the CIB- and
# Galactic-dust-corrected M20 ACT+Planck y-map, whose effective Gaussian beam
# is FWHM = 2.4' (the pre-deprojection map is 1.6'; their Sec. 2.1).  Each
# data point is the MEAN y over a circular annulus whose full width equals the
# beam FWHM (their Eq. 4; that is what the wide r_up/r_low columns are), so
# the model must be annulus-averaged, not evaluated at r_mid — at the
# innermost annulus (0.1-0.85 R200) a point evaluation biases ~2x high.
#
# Stated approximations (all well inside the ~40% Sigma_y errors):
#  * the data's per-galaxy r/R200 stacking (their method iii) is compared
#    against the model at a single occupation-weighted effective R200 —
#    the same median-theta200 approximation their own model uses (Eq. 6);
#  * the kernel is built at the sample's zmean, not the Das et al. stack's;
#  * their fit includes a free zero-point offset y_zp (map systematic) that
#    our halo-model prediction has no counterpart for; the digitized points
#    are the measurements, so any residual y_zp is absorbed nowhere.
_SZ_BEAM_FWHM_ARCMIN = 2.4   # CIB-deprojected map (Das+2023 Sec 2.1); 1.6' = pre-deprojection
_SZ_N_ANNULUS = 16           # radial nodes per annulus for the area-weighted mean
_SZ_DATA_FILES = {
    "S4": "fig_B1_top_107M110_compton_y_profile.txt",   # log10 M* in [10.7, 11.0]
    "S5": "fig_5_bottom_left_compton_y_profile.txt",    # log10 M* in [11.0, 11.3]
}


def _load_sz_data(sample):
    """(x = r/R200, x_lo, x_hi, y, sigma_y) from the digitized Das et al. 2023 profile.

    Columns: r_mid, r_up, r_low, y1e8_up, y1e8_mid, y1e8_low (comma-separated).
    ``x_lo/x_hi`` are the annulus edges (width = the 2.4' beam FWHM in R200
    units — NOT an uncertainty).  Returns ``None`` when the sample has no
    mapped profile.
    """
    fn = _SZ_DATA_FILES.get(sample)
    if fn is None:
        return None
    fp = paths.repo_root() / "data" / "das_2023" / fn
    if not fp.is_file():
        raise FileNotFoundError(f"missing SZ profile: {fp}")
    arr = np.loadtxt(fp, delimiter=",")
    x = arr[:, 0]
    x_hi = arr[:, 1]
    x_lo = arr[:, 2]
    y = arr[:, 4] * 1e-8
    sig = 0.5 * (arr[:, 3] - arr[:, 5]) * 1e-8    # (up - low)/2, symmetrised
    return x, x_lo, x_hi, y, sig


def _annulus_average_matrix(x_lo, x_hi, n_nodes=_SZ_N_ANNULUS):
    """(W, x_nodes): area-weighted annulus-mean operator.

    ``W @ f(x_nodes)`` is the mean of ``f`` over each annulus
    [x_lo_i, x_hi_i] with the 2D area measure 2πx dx (trapezoid weights on
    ``n_nodes`` radial nodes per annulus), matching how Das et al. 2023
    average y in beam-width annuli (their Eq. 4a).
    """
    x_lo = np.asarray(x_lo, float); x_hi = np.asarray(x_hi, float)
    n_ann = x_lo.size
    nodes = np.empty((n_ann, n_nodes))
    W = np.zeros((n_ann, n_ann * n_nodes))
    for i in range(n_ann):
        r = np.linspace(x_lo[i], x_hi[i], n_nodes)
        w = np.empty(n_nodes)
        w[1:-1] = (r[2:] - r[:-2]) / 2.0
        w[0] = (r[1] - r[0]) / 2.0
        w[-1] = (r[-1] - r[-2]) / 2.0
        w = w * r                       # 2πr dr, the 2π cancels in the mean
        nodes[i] = r
        W[i, i * n_nodes:(i + 1) * n_nodes] = w / w.sum()
    return W, nodes.ravel()


def _precompute_sz(sample, hmf_backend, beam_fwhm_arcmin=_SZ_BEAM_FWHM_ARCMIN):
    """Build (or load) the per-sample Sigma_y transfer kernel + data vector.

    Mirrors the X-ray ``_precompute`` cache pattern.  The kernel G_sz(r_p, M)
    is exact at any (P_0.3, beta_P) via :func:`sz_amplitude` rescaling from the
    reference the pressure profile was built with (see
    :mod:`hod_mod.fitting.sz_transfer`), so it is built ONCE per sample.
    """
    data = _load_sz_data(sample)
    if data is None:
        return None
    x, x_lo, x_hi, y, err = data
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, f"{sample}_sz_transfer.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if ("x_lo" in d.files and np.array_equal(d["x"], x)
                and np.array_equal(d["x_lo"], x_lo)
                and int(d["n_annulus"]) == _SZ_N_ANNULUS
                and float(d["beam"]) == float(beam_fwhm_arcmin)):
            return dict(G=d["G"], m200=d["m200"], x=x, y=y, err=err,
                        p03_ref=float(d["p03_ref"]), beta_ref=float(d["beta_ref"]),
                        r200_eff=float(d["r200_eff"]), n_pts=int(y.size))
        print(f"  [{sample}] cached SZ kernel stale -> rebuild", flush=True)

    from hod_mod.gas import PressureProfileDPM
    th = F._THETA_COSMO
    # same P(k) routing as the X-ray transfer (_precompute): one likelihood,
    # one linear P(k)
    pk = default_pk_linear(); hmf = make_hmf(hmf_backend, pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"], Ob0=th["Omega_b"],
                sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    pp = PressureProfileDPM(model=2, r_max_over_r200=3.0, n_gl=60)
    cross = HaloModelCrossSpectra(fhmp, pressure_profile=pp)
    hod_params = B._build_hod_params(sample)
    zmean = float(F.SAMPLES[sample]["zmean"])

    # occupation-weighted effective R200 [Mpc/h comoving] to map the data's
    # x = r/R200 onto the kernel's r_p grid (frame-independent: the (1+z)
    # factor cancels in the ratio)
    sc = cross._get_static_cache(zmean, th, hod_params)
    nc, ns, n_gal, b_eff = cross._get_hod_weights(zmean, th, hod_params, sc)
    m_np = np.asarray(sc["m_np"], float)
    w = np.asarray(sc["dndm_np"], float) * np.asarray(nc, float)
    r200_eff = float(np.trapezoid(w * np.asarray(sc["r_delta"], float), m_np)
                     / np.trapezoid(w, m_np))

    # kernel on the fine annulus nodes, then fold the annulus-mean operator W
    # in: predict_sigma_y(W @ G_fine, ...) IS the annulus-averaged prediction
    # (both are linear in the per-mass amplitudes).
    W, x_nodes = _annulus_average_matrix(x_lo, x_hi)
    rp = x_nodes * r200_eff

    t0 = time.time()
    G_fine, m200 = build_sz_transfer(cross, zmean, th, hod_params, rp,
                                     beam_fwhm_arcmin=float(beam_fwhm_arcmin))
    G = W @ G_fine
    np.savez(cache, G=G, m200=m200, x=x, x_lo=x_lo, x_hi=x_hi,
             n_annulus=_SZ_N_ANNULUS, beam=float(beam_fwhm_arcmin),
             p03_ref=float(pp._P_03), beta_ref=float(pp._beta), r200_eff=r200_eff)
    print(f"[{sample}] SZ kernel built in {time.time()-t0:.0f}s "
          f"(R200_eff={r200_eff:.3f} Mpc/h) -> {cache}", flush=True)
    return dict(G=G, m200=m200, x=x, y=y, err=err, p03_ref=float(pp._P_03),
                beta_ref=float(pp._beta), r200_eff=r200_eff, n_pts=int(y.size))


# --- tSZ leg, sum_stat source (DEFAULT): the BGS_SZ stacked Sigma_y ----------
#
# Preferred over the digitized Das+2023 profiles above, on three counts:
#
#  * COVERAGE — the measurement exists for every fit sample (S1..S7), not just
#    the two whose Das+2023 M* BIN happens to line up with a threshold sample;
#  * NO BIN->THRESHOLD MAPPING — it is measured on the SAME threshold samples
#    this fit uses (matched by N), so the "bin dominates the threshold counts"
#    approximation documented above simply does not arise;
#  * NO R200_EFF MAPPING — r_p is already comoving Mpc/h, so the kernel is built
#    directly on the data grid; the occupation-weighted effective-R200 step (and
#    its median-theta200 approximation) drops out entirely.
#
# It also carries a full covariance rather than symmetrised digitized error bars,
# which matters: neighbouring r_p bins of a beam-smoothed stack are strongly
# correlated, so a diagonal chi2 would over-count the information.
#
# Remaining approximation: the model is annulus-averaged over the r_p bin edges
# (as for Das+2023), and the kernel is built at the sample's zmean rather than
# integrated over n(z).
_SZ_SOURCES = ("sumstat", "das23")
_SZ_SUMSTAT_SUBDIR = "BGS_SZ"


def _sz_sumstat_path(sample):
    """BGS_SZ file for a fit sample; the galaxy count N is a unique key."""
    n = int(F.SAMPLES[sample]["N"])
    root = paths.sum_stat_root() / _SZ_SUMSTAT_SUBDIR
    if not root.is_dir():
        return None
    cand = sorted(root.glob(f"*_N_{n:07d}_*sz*.h5"))
    return cand[0] if cand else None


def _load_sz_sumstat(sample):
    """(r_p [Mpc/h comoving], Sigma_y, cov, bin edges, beam, z_eff) or None."""
    fp = _sz_sumstat_path(sample)
    if fp is None:
        return None
    from hod_mod.data_io.sum_stat_reader import SumStatReader
    d = SumStatReader.from_hdf5(os.fspath(fp)).sz()
    edges = np.asarray(d["bin_edges"], float)
    return dict(rp=np.asarray(d["rp"], float), y=np.asarray(d["sigma_y"], float),
                cov=np.asarray(d["cov"], float), rp_lo=edges[:-1], rp_hi=edges[1:],
                beam=float(d["beam_fwhm_arcmin"]), z_eff=float(d["z_eff"]))


def _precompute_sz_sumstat(sample, hmf_backend):
    """Sigma_y kernel + data vector for the sum_stat BGS_SZ measurement."""
    data = _load_sz_sumstat(sample)
    if data is None:
        return None
    rp, y, cov = data["rp"], data["y"], data["cov"]
    rp_lo, rp_hi, beam = data["rp_lo"], data["rp_hi"], data["beam"]
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, f"{sample}_sz_sumstat_transfer.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if (np.array_equal(d["rp"], rp) and int(d["n_annulus"]) == _SZ_N_ANNULUS
                and float(d["beam"]) == beam):
            return dict(G=d["G"], m200=d["m200"], y=y, cov=cov,
                        icov=np.linalg.inv(cov), err=np.sqrt(np.diag(cov)),
                        p03_ref=float(d["p03_ref"]), beta_ref=float(d["beta_ref"]),
                        n_pts=int(y.size))
        print(f"  [{sample}] cached sum_stat SZ kernel stale -> rebuild", flush=True)

    from hod_mod.gas import PressureProfileDPM
    th = F._THETA_COSMO
    pk = default_pk_linear(); hmf = make_hmf(hmf_backend, pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"], Ob0=th["Omega_b"],
                sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    pp = PressureProfileDPM(model=2, r_max_over_r200=3.0, n_gl=60)
    cross = HaloModelCrossSpectra(fhmp, pressure_profile=pp)
    hod_params = B._build_hod_params(sample)
    zmean = float(F.SAMPLES[sample]["zmean"])

    # r_p is already comoving Mpc/h -> the kernel is built straight on the data
    # grid; annulus-average over the bin edges (both linear in G, so W folds in).
    W, rp_nodes = _annulus_average_matrix(rp_lo, rp_hi)
    t0 = time.time()
    G_fine, m200 = build_sz_transfer(cross, zmean, th, hod_params, rp_nodes,
                                     beam_fwhm_arcmin=beam)
    G = W @ G_fine
    np.savez(cache, G=G, m200=m200, rp=rp, n_annulus=_SZ_N_ANNULUS, beam=beam,
             p03_ref=float(pp._P_03), beta_ref=float(pp._beta))
    print(f"[{sample}] sum_stat SZ kernel built in {time.time()-t0:.0f}s "
          f"({y.size} bins, beam={beam:.2f}') -> {cache}", flush=True)
    return dict(G=G, m200=m200, y=y, cov=cov, icov=np.linalg.inv(cov),
                err=np.sqrt(np.diag(cov)), p03_ref=float(pp._P_03),
                beta_ref=float(pp._beta), n_pts=int(y.size))


def _sigma_y_model(p, sz):
    """Sigma_y(r_p) at the native-DPM params ``p`` from the precomputed kernel."""
    return predict_sigma_y(sz["G"], sz["m200"], 10.0 ** p[2], p[3],
                           sz["p03_ref"], sz["beta_ref"])


# --- native-DPM J_b(T_0, Z) grid over the (p2, r_max) shape axes -------------

# T_0 axis must resolve the sharp edge that the T_min selection puts in J(T_0):
# a coarse grid smears it and costs several % at low mass (see dpm_bands docs).
_LT0_GRID = np.linspace(-1.6, 1.6, 481)          # log10 T_0 [keV]
_ZMET_GRID = np.array([0.05, 0.15, 0.3, 0.5, 0.75, 1.0])   # spans the z_metal bound
_T_MIN_XRAY = 0.3        # keV: 0.5-2 keV selects the hot phase (validate_gas_profiles)


def _j_grid(sample):
    """Build (or load) J_b(T_0, Z | p2, r_max): (Np2, Nrmax, Nb+1, NT, NZ) + V-shape grid.

    The density shape (alpha_out = _ALPHA_PROF + 2 p2) and the truncation r_max both
    enter f_n and g(x) = f_P/f_n, so J must be tabulated per (p2, r_max) node — the
    same nodes the transfer grid G uses, so the two interpolate consistently.
    """
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, "dpm_j_grid.npz")
    cool_bands, cool_broad = _band_cooling()
    all_cool = list(cool_bands) + [cool_broad]
    if os.path.exists(cache):
        d = np.load(cache)
        if (np.array_equal(d["p2_grid"], _P2_GRID) and np.array_equal(d["rmax_grid"], _RMAX_GRID)
                and np.array_equal(d["lt0"], _LT0_GRID) and np.array_equal(d["zmet"], _ZMET_GRID)
                and d["J"].shape[2] == len(all_cool)):
            return d["J"], d["V"]
        print("  dpm J grid stale -> rebuild", flush=True)

    pp = PressureProfileDPM(model=2)      # fixed DPM pressure shape -> g(x)
    J = np.zeros((_P2_GRID.size, _RMAX_GRID.size, len(all_cool),
                  _LT0_GRID.size, _ZMET_GRID.size))
    V = np.zeros((_P2_GRID.size, _RMAX_GRID.size))
    t0 = time.time()
    for i, p2 in enumerate(_P2_GRID):
        for j, rmax in enumerate(_RMAX_GRID):
            dp = _make_density_variant(model=2, ne_03=1e-4, beta=0.5, alpha_in=_ALPHA_PROF,
                                       alpha_tr=2.0, alpha_out=_ALPHA_PROF + 2.0 * float(p2))
            dp._r_max_factor = float(rmax)
            tab = build_j_table(dp, pp, all_cool, z_grid=_ZMET_GRID,
                                log10_t0_grid=_LT0_GRID, x_lo=1e-2, n_x=400,
                                t_min=_T_MIN_XRAY)
            J[i, j] = tab._tab
            V[i, j] = shape_integral(dp, x_lo=1e-2, n_x=400)
        print(f"  dpm J grid p2={p2:.2f} ({(i+1)*_RMAX_GRID.size}/"
              f"{_P2_GRID.size*_RMAX_GRID.size}) [{time.time()-t0:.0f}s]", flush=True)
    np.savez(cache, J=J, V=V, p2_grid=_P2_GRID, rmax_grid=_RMAX_GRID,
             lt0=_LT0_GRID, zmet=_ZMET_GRID)
    print(f"dpm J grid built in {time.time()-t0:.0f}s -> {cache}", flush=True)
    return J, V


def _make_j_interp(J, V):
    """(J_interp(p2, rmax, T_0[NM], Z) -> (Nb+1, NM), V_interp(p2, rmax) -> float)."""
    axes = (_P2_GRID, _RMAX_GRID, _LT0_GRID, _ZMET_GRID)
    itps = [RegularGridInterpolator(axes, J[:, :, b], method="linear",
                                    bounds_error=False, fill_value=None)
            for b in range(J.shape[2])]
    v_itp = RegularGridInterpolator((_P2_GRID, _RMAX_GRID), V, method="linear",
                                    bounds_error=False, fill_value=None)

    def j_interp(p2, rmax, t0, z_metal):
        lt = np.log10(np.clip(np.asarray(t0, float), 10 ** _LT0_GRID[0], 10 ** _LT0_GRID[-1]))
        n = lt.size
        pts = np.column_stack([np.full(n, float(p2)), np.full(n, float(rmax)), lt,
                               np.full(n, float(np.clip(z_metal, _ZMET_GRID[0],
                                                        _ZMET_GRID[-1])))])
        return np.stack([itp(pts) for itp in itps])       # (Nb+1, NM)

    return j_interp, (lambda p2, rmax: float(v_itp([[float(p2), float(rmax)]])[0]))


# --- model + priors ---------------------------------------------------------

def _weight_bands(p, S):
    """Native-DPM radial band weight — (Nb, NM) [erg/s per halo, pre-anchor].

    .. math::
        L_{X,b}(M) = n_{e,0.3}^2 M_{12}^{2\\beta_n} E^{2\\gamma_n}
                     \\; V_{\\rm shape}(M|p_2,r_{\\max}) \\; J_b(T_0(M), Z)

    with :math:`T_0(M) = (P_{0.3}/n_{e,0.3}) M_{12}^{\\beta_P-\\beta_n} E^{\\gamma_P-\\gamma_n}`.
    ``J_b`` integrates :math:`\\Lambda_b` over DPM's **radial** temperature profile
    (exact factorisation, see :mod:`hod_mod.fitting.dpm_bands`), so this returns the
    band luminosity directly — there is no Λ_b/Λ_broad isothermal split any more.
    """
    log10_ne03, beta_n, log10_p03, beta_P = p[0], p[1], p[2], p[3]
    p2, rmax = p[4], p[5]
    ne03, p03 = 10.0 ** log10_ne03, 10.0 ** log10_p03
    m200, ez = S["m200"], S["ez"]
    t0 = np.clip(t0_of_mass(m200, p03, ne03, beta_P, beta_n, ez), S["kT_lo"], S["kT_hi"])
    em = emission_measure_factor(m200, ne03, beta_n, ez)             # (NM,)
    v = v_shape_of_mass(S["r200"], _C_DPM_REF, S["V_interp"](p2, rmax), _MPC_CM, S["h"])
    jb = S["J_interp"](p2, rmax, t0, float(np.clip(p[7], 0.05, 3.0)))  # (Nb+1, NM)
    return (em * v)[None, :] * jb[:_NB]


def _components_bands(p, S):
    """ARF-weighted gas and AGN (Nb, Ntheta)."""
    G = S["G_interp"]([[p[4], p[5]]])[0].reshape(S["nth"], -1)   # (Ntheta, NM)
    gas = S["c_total"] * (_weight_bands(p, S) @ G.T)             # (Nb, Ntheta)
    fb = _agn_band_fractions(gamma=p[8]) if len(p) > 8 else S["fb"]   # free AGN photon index
    agn = 10.0 ** p[6] * S["c_obs_total"] * (fb[:, None] * S["agn_dc1"][None, :])
    return S["arf"][:, None] * gas, S["arf"][:, None] * agn


def _model_bands(p, S):
    g, a = _components_bands(p, S)
    return g + a


def _chi2_sample(p, S):
    r = (_model_bands(p, S) - S["wtheta"])[:, S["mask"]] / S["err"][:, S["mask"]]
    chi2 = float(np.sum(r ** 2))
    sz = S.get("sz")
    if sz is not None:
        # the tSZ leg: same (P_0.3, beta_P) as T = P/n_e in the bands above
        r_sz = _sigma_y_model(p, sz) - sz["y"]
        icov = sz.get("icov")
        if icov is not None:
            # neighbouring r_p bins of a beam-smoothed stack are strongly
            # correlated; a diagonal chi2 would over-count the SZ information
            chi2 += float(r_sz @ icov @ r_sz)
        else:
            chi2 += float(np.sum((r_sz / sz["err"]) ** 2))
    return chi2


def _bounds():
    return _BND8


def _log_prior(p):
    # p may be shorter than _PARAMS (agn_gamma optional -- _components_bands falls
    # back to the fixed band fractions), so slice the prior vectors to len(p) rather
    # than assuming they align.  Indexing p with a full-length boolean mask raised
    # IndexError whenever agn_gamma was omitted.
    p = np.asarray(p, float)
    n = p.size
    for v, (lo, hi) in zip(p, _BND8):
        if not (lo <= v <= hi):
            return -np.inf
    lp = 0.0
    if _GAS_PRIOR is not None:
        # full-covariance native-DPM gas prior (back-propagated from LX-M / kT-M);
        # its strong correlations are physical -- see _induced_gas_prior.
        d = p[:4] - _GAS_PRIOR["mu"]
        lp -= 0.5 * float(d @ _GAS_PRIOR["icov"] @ d)
    sig, mu = _SIG8[:n], _MU8[:n]
    fin = np.isfinite(sig)
    d = (p[fin] - mu[fin]) / sig[fin]
    return lp - 0.5 * float(np.sum(d ** 2))


def _neg_log_prob(p, samples):
    lp = _log_prior(p)
    if not np.isfinite(lp):
        return 1e30
    return -lp + 0.5 * sum(_chi2_sample(p, S) for S in samples.values())


def _anchor(samples, anchor_sample):
    """Empirical gas amplitude anchor on the anchor sample: at the prior-centre
    native-DPM params + mid shape, free (A_gas, A_AGN) lsq over the band data -> the
    A_gas that reproduces it defines c_total.

    This is what lets the native re-base skip re-deriving the absolute
    n_e,0.3 -> erg/s -> counts chain: c_total absorbs any constant offset, so only
    the *mass scalings* of L_X and T = P/n_e need to be right."""
    S = samples[anchor_sample]
    pc = np.array([_MU8[0], _MU8[1], _MU8[2], _MU8[3], 0.6, 4.0, -1.5, _Z_FID, _GAMMA_AGN])
    St = dict(S); St["c_total"] = 1.0
    gas1, _ = _components_bands(pc, St)                # c_total=1
    agn1 = St["arf"][:, None] * St["c_obs_total"] * (St["fb"][:, None] * St["agn_dc1"][None, :])
    m = S["mask"]; w = 1.0 / S["err"][:, m].ravel()
    A = np.column_stack([gas1[:, m].ravel() * w, agn1[:, m].ravel() * w])
    res = lsq_linear(A, S["wtheta"][:, m].ravel() * w, bounds=([0, 0], [np.inf, np.inf]),
                     method="bvls")
    chi2 = float(np.sum((A @ res.x - S["wtheta"][:, m].ravel() * w) ** 2))
    return float(res.x[0]), chi2


# --- figures ----------------------------------------------------------------

def _figures_bands(tag, samples, flat, chain_full, lp_full, nburn, map_p, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ecen = np.array([0.5 * (lo + hi) for lo, hi in _BAND_EDGES])
    np_ = len(_PARAMS)
    fig, axs = plt.subplots(np_ + 1, 1, figsize=(8, 1.5 * (np_ + 1)), sharex=True)
    for i, p in enumerate(_PARAMS):
        axs[i].plot(chain_full[:, :, i], color="C0", alpha=0.12, lw=0.5)
        axs[i].set_ylabel(p, fontsize=8); axs[i].axvline(nburn, color="C3", ls=":")
    axs[-1].plot(lp_full, color="C0", alpha=0.12, lw=0.5)
    axs[-1].set_ylabel("log prob", fontsize=8); axs[-1].axvline(nburn, color="C3", ls=":")
    axs[0].set_title(f"{tag}: band-fit traces (red=burn-in {nburn})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{tag}_bands_traces.png"), dpi=110)
    plt.close(fig)
    try:
        import corner
        fig = corner.corner(flat, labels=_PARAMS, truths=list(map_p))
        fig.savefig(os.path.join(out_dir, f"{tag}_bands_corner.png"), dpi=110); plt.close(fig)
    except Exception as e:
        print(f"  (corner skipped: {e})", flush=True)
    for s, S in samples.items():
        th = S["th_as"]; wd = S["wtheta"]; err = S["err"]
        gas_m, agn_m = _components_bands(map_p, S); tot = gas_m + agn_m
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for th0, c in [(10.0, "C0"), (40.0, "C1")]:
            k = int(np.argmin(np.abs(th - th0)))
            a1.errorbar(ecen, wd[:, k], yerr=err[:, k], fmt="o", ms=3, color=c,
                        label=fr"data $\theta$≈{th[k]:.0f}″")
            a1.plot(ecen, tot[:, k], "-", color=c)
        a1.set_xlabel("band energy [keV]"); a1.set_ylabel(r"$w_b(\theta)$")
        a1.set_title(f"{s}: band spectrum (pts=data, line=model)", fontsize=9)
        a1.legend(fontsize=7); a1.axhline(0, color="grey", lw=0.5)
        m = S["mask"]
        sd = np.nansum(wd[0:4], 0); hd = np.nansum(wd[10:15], 0)
        sm = np.nansum(tot[0:4], 0); hm = np.nansum(tot[10:15], 0)
        good = m & np.isfinite(sd) & np.isfinite(hd) & (hd > 0)
        a2.plot(th[good], (sd / hd)[good], "ko", ms=3, label="data")
        a2.plot(th[good], (sm / hm)[good], "C0-", label="model")
        a2.set_xscale("log"); a2.set_xlabel(r"$\theta$ [arcsec]")
        a2.set_ylabel("soft(0.5-0.9)/hard(1.5-2.0)")
        a2.set_title(f"{s}: band ratio (temperature)", fontsize=9); a2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{s}_bands_spectrum.png"), dpi=110)
        plt.close(fig)
        sz = S.get("sz")
        if sz is not None:
            # posterior-median Sigma_y band from the flat chain (kernel is exact
            # in (P_0.3, beta_P), so this is a cheap matrix product per draw)
            sub = flat[:: max(1, flat.shape[0] // 400)]
            ys = np.array([_sigma_y_model(q, sz) for q in sub])
            lo, med, hi = np.percentile(ys, [16, 50, 84], axis=0)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.errorbar(sz["x"], sz["y"], yerr=sz["err"], fmt="ko", ms=4,
                        label="Das et al. 2023")
            ax.plot(sz["x"], _sigma_y_model(map_p, sz), "C0-", label="MAP")
            ax.fill_between(sz["x"], lo, hi, color="C0", alpha=0.25,
                            label="posterior 16-84%")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$r / R_{200}$"); ax.set_ylabel(r"$\Sigma_y$")
            ax.set_title(f"{s}: stacked Compton-y (same $P_{{0.3}}, \\beta_P$ "
                         "as the X-ray bands)", fontsize=9)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{s}_sz_profile.png"), dpi=110)
            plt.close(fig)


# --- main -------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", nargs="+", default=list(F.SAMPLES))
    ap.add_argument("--hmf", default="tinker08")
    ap.add_argument("--f-sys", type=float, default=0.05)
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--mcmc", action="store_true")
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=8000)
    ap.add_argument("--nburn", type=int, default=2000)
    ap.add_argument("--candidate", default="baseline", choices=_CANDIDATES,
                    help="hot-gas hypothesis to test; writes to its own subfolder")
    ap.add_argument("--sz", action="store_true",
                    help="add the tSZ Sigma_y leg, coupled to the X-ray bands "
                         "through the shared (P_0.3, beta_P)")
    ap.add_argument("--sz-source", default="sumstat", choices=_SZ_SOURCES,
                    help="Sigma_y measurement: 'sumstat' = the BGS_SZ stack for "
                         "ALL samples, measured on these same threshold samples, "
                         "with a full covariance (default); 'das23' = the two "
                         "digitized Das+2023 M*-bin profiles")
    ap.add_argument("--sz-beam-arcmin", type=float, default=_SZ_BEAM_FWHM_ARCMIN,
                    help="das23 only: Gaussian beam FWHM of the y-map the profiles "
                         "were measured on (default %(default)s; MUST match the map). "
                         "The sumstat source reads its own beam from the file.")
    args = ap.parse_args(argv)
    tag = "_".join(args.samples)

    # candidate config: agn_fixed pins log10DC to the Phase-A broad-band value
    dc_fix = -1.8
    if args.candidate == "agn_fixed":
        af = "S1" if "S1" in args.samples else args.samples[0]
        pa = os.path.join(J._OUT_DIR, f"{af}_bb_summary.json")
        if os.path.isfile(pa):
            dc_fix = float(json.load(open(pa))["posterior"]["log10DC"]["median"])
    subdir = _apply_candidate(args.candidate, dc_fix)
    out_dir = _OUT_DIR if subdir == "baseline" else os.path.join(_OUT_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Candidate '{args.candidate}' -> {out_dir}"
          + (f"  (dc_fix={dc_fix:.2f})" if args.candidate == "agn_fixed" else ""), flush=True)

    cool_bands, cool_broad = _band_cooling()
    fb = _agn_band_fractions(); arf = _arf_band_weights()
    kT_lo, kT_hi = 0.09, 19.0
    # Native-DPM J_b(T_0, Z | p2, r_max): shape-independent of the sample, built once.
    j_grid, v_grid = _j_grid(args.samples[0])
    j_interp, v_interp = _make_j_interp(j_grid, v_grid)
    samples = {}
    for s in args.samples:
        G_grid, log10_m500c, ez, agn_dc1, bd, mask, m200, r200 = _precompute(s, args.hmf)
        err = np.sqrt(bd["wtheta_err"] ** 2 + (args.f_sys * np.abs(bd["wtheta"])) ** 2)
        nth = bd["theta_arcsec"].size
        G_interp = RegularGridInterpolator(
            (_P2_GRID, _RMAX_GRID), G_grid.reshape(_P2_GRID.size, _RMAX_GRID.size, -1),
            method="linear", bounds_error=False, fill_value=None)
        samples[s] = dict(G_interp=G_interp, nth=nth, log10_m500c=log10_m500c, ez=ez,
                          agn_dc1=agn_dc1, wtheta=bd["wtheta"], err=err, mask=mask,
                          th_as=bd["theta_arcsec"], fb=fb, arf=arf,
                          cool_bands=cool_bands, cool_broad=cool_broad, kT_lo=kT_lo, kT_hi=kT_hi,
                          c_obs_total=J._c_obs_total(s), srx=float(F.load_data(s)["beckground"][0]),
                          n_pts=int(mask.sum()) * _NB,
                          # native-DPM inputs (M200-based, unlike the M500c relations)
                          m200=m200, r200=r200, h=float(F._THETA_COSMO["h"]),
                          J_interp=j_interp, V_interp=v_interp)
        print(f"[{s}] transfer ready, n_pts={samples[s]['n_pts']}", flush=True)

    if args.sz:
        n_sz = 0
        for s in args.samples:
            if args.sz_source == "sumstat":
                sz = _precompute_sz_sumstat(s, args.hmf)
                miss = "no BGS_SZ measurement in $HOD_MOD_SUMSTAT"
            else:
                sz = _precompute_sz(s, args.hmf, beam_fwhm_arcmin=args.sz_beam_arcmin)
                miss = f"no mapped das_2023 profile (mapped: {sorted(_SZ_DATA_FILES)})"
            if sz is not None:
                samples[s]["sz"] = sz
                samples[s]["n_pts"] += sz["n_pts"]
                n_sz += 1
                extra = (f", R200_eff={sz['r200_eff']:.3f} Mpc/h"
                         if "r200_eff" in sz else ", full covariance")
                print(f"[{s}] SZ leg ready ({args.sz_source}): {sz['n_pts']} "
                      f"Sigma_y pts{extra}", flush=True)
            else:
                print(f"[{s}] SZ leg: {miss} — X-ray only", flush=True)
        if n_sz == 0:
            print(f"WARNING: --sz given but no requested sample has a "
                  f"{args.sz_source} Sigma_y measurement", flush=True)

    anchor_sample = "S1" if "S1" in samples else args.samples[0]
    samples[anchor_sample]["c_total"] = 1.0
    c_total_S1, chi2a = _anchor(samples, anchor_sample)
    srx_a = samples[anchor_sample]["srx"]
    print(f"\nAnchor on {anchor_sample}: c_total={c_total_S1:.3e} (unconstrained band chi2={chi2a:.1f})",
          flush=True)
    for s, S in samples.items():
        S["c_total"] = c_total_S1 * srx_a / S["srx"]

    n_tot = sum(S["n_pts"] for S in samples.values())
    print(f"\nBand MAP over {len(samples)} samples × {_NB} bands, {n_tot} pts, "
          f"{len(_PARAMS)} shared params (native DPM gas: n_e,0.3/beta_n, P_0.3/beta_P) ...",
          flush=True)

    def nlp(p):
        return _neg_log_prob(p, samples)

    # 9-vec starts, matching _PARAMS (…, log10DC, z_metal, agn_gamma); log10DC seeded
    # at the candidate's bound mid.  NOTE these were 8-vectors while _PARAMS/_MU8 were
    # 9 ever since agn_gamma was promoted, so agn_gamma was silently dropped by the
    # zip() into `out` and _log_prior raised IndexError — the standalone MAP path was
    # dead.  Keep these aligned with _PARAMS.
    dc0 = float(np.clip(-1.5, _BND8[6, 0], _BND8[6, 1]))
    # starts centred on the induced gas prior (_MU8[:4]); the spread walks along the
    # prior's strongly-correlated directions rather than across them.
    g = _MU8[:4]
    starts = [   # native DPM: log10 n_e,0.3 [cm^-3], beta_n, log10 P_0.3 [keV cm^-3], beta_P
        [g[0], g[1], g[2], g[3], 0.6, 4.0, dc0, _Z_FID, _GAMMA_AGN],
        [g[0] + 0.3, g[1] - 0.1, g[2] + 0.3, g[3] - 0.1, 0.3, 3.5, dc0 + 0.4, _Z_FID, _GAMMA_AGN],
        [g[0] - 0.3, g[1] + 0.1, g[2] - 0.3, g[3] + 0.1, 1.2, 4.5, dc0 - 0.4, 0.5, _GAMMA_AGN],
        [g[0], g[1], g[2], g[3], 0.6, 4.0, dc0, 0.15, _GAMMA_AGN],
    ]
    best = None
    for q0 in starts:
        o = minimize(nlp, np.array(q0), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=6000))
        if best is None or o.fun < best.fun:
            best = o
        print(f"  start {np.round(q0,2)} -> chi2={2*o.fun:.1f}", flush=True)
    map_p = best.x
    chi2 = 2.0 * sum(_chi2_sample(map_p, S) for S in samples.values())
    ndof = max(n_tot - len(_PARAMS), 1)
    out = dict(zip(_PARAMS, [float(v) for v in map_p]))
    out["chi2"] = chi2; out["ndof"] = ndof; out["chi2_per_dof"] = chi2 / ndof
    out["chi2_per_sample"] = {s: float(_chi2_sample(map_p, S)) for s, S in samples.items()}
    if args.sz:
        out["sz"] = {s: dict(n_pts=S["sz"]["n_pts"], beam_arcmin=args.sz_beam_arcmin,
                             r200_eff=S["sz"]["r200_eff"],
                             chi2=float(np.sum(((_sigma_y_model(map_p, S["sz"])
                                                 - S["sz"]["y"]) / S["sz"]["err"]) ** 2)))
                     for s, S in samples.items() if "sz" in S}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"joint_bands_map_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== BAND MAP (free LX-M, kT-M) ===")
    for k in _PARAMS:
        print(f"  {k:10s} = {out[k]:+.4f}")
    print(f"  chi2/dof   = {chi2:.1f}/{ndof} = {chi2/ndof:.3f}")
    for s, v in out["chi2_per_sample"].items():
        print(f"    {s}: chi2={v:.1f} ({samples[s]['n_pts']} pts)", flush=True)

    if args.map_only or not args.mcmc:
        return out

    import emcee
    from hod_mod.fitting.mcmc_resume import revive_ensemble
    ndim = len(_PARAMS); nw = args.nwalkers
    rng = np.random.default_rng(42)

    # Checkpoint to an HDF backend and resume from it.  Without this the whole run
    # was lost to any walltime kill or preemption -- and this fit pays ~3 h of
    # precompute (dpm_j_grid + per-sample transfer + SZ kernels) before the first
    # step, inside a 24 h job.  Same failure mode fit_joint_lsdr10 documents as
    # having destroyed two attempts in 2026-07.
    backend_path = os.path.join(out_dir, f"{tag}_bands_backend.h5")
    backend = emcee.backends.HDFBackend(backend_path)
    try:
        already = backend.iteration
    except (AttributeError, OSError, KeyError):
        already = 0
    if already and backend.shape != (nw, ndim):
        print(f"  existing backend is {backend.shape}, need {(nw, ndim)} -> fresh chain",
              flush=True)
        already = 0
    def _lnp(p):
        # nlp() was being called TWICE per proposal (once for the test, once for the
        # value), which doubled the cost of every band MCMC for no benefit --
        # measured at 2.04 nlp calls per walker-step.
        v = nlp(p)
        return -v if v < 1e29 else -np.inf

    sampler = emcee.EnsembleSampler(nw, ndim, _lnp, backend=backend)
    t0 = time.time()
    if already >= args.nsteps:
        print(f"\nBAND MCMC: chain already complete ({already} >= {args.nsteps} steps) "
              f"-> {backend_path}", flush=True)
    elif already == 0:
        p0 = map_p + 1e-3 * rng.standard_normal((nw, ndim)) * np.ptp(_bounds(), axis=1)
        p0 = np.clip(p0, _bounds()[:, 0] + 1e-6, _bounds()[:, 1] - 1e-6)
        backend.reset(nw, ndim)
        print(f"\nBAND MCMC: {nw} walkers × {args.nsteps} steps ({tag}), "
              f"checkpointing to {backend_path} ...", flush=True)
        sampler.run_mcmc(p0, args.nsteps, progress=False)
    else:
        print(f"\nBAND MCMC: resuming {tag} from step {already}/{args.nsteps} "
              f"({args.nsteps - already} left) -> {backend_path}", flush=True)
        _start = revive_ensemble(backend, _bounds()[:, 0], _bounds()[:, 1], label=tag)
        sampler.run_mcmc(_start, args.nsteps - already, progress=False,
                         skip_initial_state_check=True)
    acc = float(np.mean(backend.accepted / max(backend.iteration, 1)))
    print(f"MCMC done in {time.time()-t0:.0f}s; acceptance={acc:.2f}", flush=True)
    flat = sampler.get_chain(discard=args.nburn, flat=True)
    np.savez(os.path.join(out_dir, f"{tag}_bands_chain.npz"), flatchain=flat,
             log_prob=sampler.get_log_prob(discard=args.nburn, flat=True),
             chain=sampler.get_chain(), lp=sampler.get_log_prob(), params=_PARAMS,
             nburn=args.nburn, c_total={s: S["c_total"] for s, S in samples.items()})
    pct = np.percentile(flat, [16, 50, 84], axis=0)
    post = {p: dict(median=float(pct[1, i]), lo=float(pct[1, i] - pct[0, i]),
                    hi=float(pct[2, i] - pct[1, i])) for i, p in enumerate(_PARAMS)}
    with open(os.path.join(out_dir, f"{tag}_bands_summary.json"), "w") as fh:
        json.dump(dict(samples=args.samples, map=out,
                       acceptance=acc, posterior=post), fh, indent=2)
    print("Posterior (median +hi -lo):", flush=True)
    for i, p in enumerate(_PARAMS):
        print(f"  {p:10s} = {pct[1,i]:.3f} +{pct[2,i]-pct[1,i]:.3f} -{pct[1,i]-pct[0,i]:.3f}",
              flush=True)
    _figures_bands(tag, samples, flat, sampler.get_chain(), sampler.get_log_prob(),
                   args.nburn, map_p, out_dir)
    print(f"Saved chain/summary/figures -> {out_dir}", flush=True)
    return out


if __name__ == "__main__":
    main()
