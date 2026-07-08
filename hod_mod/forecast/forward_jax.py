r"""Pure-JAX forward model for the ZM15 + X-ray gas & AGN Fisher forecast.

This module re-expresses the *observable assembly* (the 1-halo/2-halo mass
integrals and the projection / Limber transforms) in ``jnp`` so that the five
data vectors are differentiable end-to-end w.r.t. every parameter — including
cosmology, because the linear P(k) comes from the analytic Eisenstein & Hu
(1998) transfer function (see :mod:`hod_mod.forecast.pk_eisenstein_hu`).

It **reuses** the repository's JAX primitives rather than re-deriving physics:

* :class:`~hod_mod.core.halo_mass_function.HaloMassFunction` for σ(M,z), dn/dM,
  b(M) — fed the EH98 *shape* spectrum, with σ8 normalisation done inside it;
* :func:`~hod_mod.core.halo_profiles.nfw_uk_jax` and
  :func:`~hod_mod.core.halo_profiles.concentration_dutton14_jax` (200c);
* the ZM15 occupation functions ``n_cen_thresh_zu15`` / ``n_sat_thresh_zu15``;
* the Ogata j₀ Hankel transform :func:`~hod_mod.observables.clustering._pk_to_xi`;
* the JAX comoving distance :func:`~hod_mod.core.distances.comoving_distance`.

The clustering (w_p) and lensing (ΔΣ) legs are an *exact* port of
``FullHaloModelPrediction`` (standard 1h+2h, no baryon split / BNL / off-centre)
and are validated to sub-percent against it.  The X-ray gas, X-ray auto and tSZ
legs use analytic, differentiable surrogates (a gNFW n_e² emissivity shape with
an analytic L_X(M) amplitude, and an A10 GNFW pressure shape) — the
sensitivity-relevant parameterisation — since the production emissivity /
pressure FTs are numpy Gauss–Legendre and the full-APEC path is not JAX-traceable.

Parameter vector: see :data:`PARAM_NAMES` (single source of truth for names,
ordering and count).  The first 31 entries are the tier-1 vector (cosmology,
ZM15 HOD, X-ray gas, duty cycle/pressure, baryon feedback, Powell AGN-XLF);
the tier-2 extension (:data:`TIER2_EXTENSION`) appends the formerly-fixed
nuisance shapes so that "nothing is fixed" studies can free them, while tier-1
scripts pin them to their fiducials (σ=1e-4) to keep their published meaning.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.core.halo_mass_function import (
    HaloMassFunction,
    _growth_factor_flat_jax,
    _RHO_CRIT0,
    growth_factor,
)
from hod_mod.core.halo_profiles import nfw_uk_jax, concentration_dutton14_jax
from hod_mod.connection.hod.zumandelbaum15 import (
    _mh_from_mstar_zu15,
    _mstar_from_mh_zu15,
    sigma_lnmstar_zu15,
    f_red_cen_zu16,
    f_red_sat_zu16,
)
from hod_mod.connection.morphology import f_early_cen, f_early_sat
from jax import custom_jvp
from jax.scipy.special import erf, erfc
from hod_mod.observables.clustering import _pk_to_xi
from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid, make_baryon_fraction
from hod_mod.core.distances import comoving_distance
from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear


# ---------------------------------------------------------------------------
# Parameter vector layout
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "Omega_m", "sigma8", "h", "n_s", "Omega_b",
    "lg_m1h", "lg_m0star", "beta", "delta", "gamma",
    "sigma_lnmstar", "eta", "fc", "bsat",
    "lx_norm", "lx_slope", "kt_norm", "kt_slope", "p2", "r_max",
    "log10DC", "beta_pressure",
    "log10_M_pivot", "log10_eta_min",   # shared hot-gas / baryon-feedback sector
    # Powell 2022 AGN-halo sector: M_BH–M_* relation + universal ERDF that
    # forward-model the AGN X-ray luminosity function (the "xlf" observable).
    "agn_mu_bh", "agn_al_bh", "agn_sig_bh",
    "agn_log10_lstar", "agn_delta1", "agn_delta2", "agn_log10_ferdf",
    # ---- tier-2 promoted nuisances (fiducial == the former fixed constant, so
    # fiducial predictions are unchanged; tier-1 scripts pin them by default) ----
    "beta_sat", "bcut", "beta_cut", "alpha_sat",          # satellite HOD shape
    "beta_b", "log10_M_eta", "beta_eta",                  # baryon-sector shape
    "alpha_in_gas", "alpha_tr_gas",                       # gas emissivity slopes
    "p0_pressure", "c500_pressure", "gamma_pressure",     # A10 GNFW pressure
    "alpha_pressure", "beta_out_pressure",                # ("beta_pressure" is the tilt)
    "agn_rho", "agn_sig_mstar",                           # AGN-XLF internals (Powell Model 2)
    # ---- tier-2 redshift-evolution slopes: additive on the base parameter per
    # ln[(1+z_eff)/(1+z_pivot)] (see ForwardModel._theta_eff); fiducial 0 ----
    "lg_m1h_zs", "lg_m0star_zs", "sigma_lnmstar_zs",      # SHMR evolution
    "lx_zs", "kt_zs",                                     # departures from self-similar E(z)
    "agn_log10_ferdf_zs", "agn_log10_lstar_zs",           # AGN sector evolution
    # ---- tier-2 X-ray spectral sector (multi-band APEC layer) ----
    "t_prof_slope",                                       # radial T profile tilt (0 = isothermal)
    "z_gas_norm", "z_gas_mslope", "z_gas_zs",             # ICM metallicity [Z_sun] + slopes
    "agn_gamma", "agn_fabs",                              # AGN photon index + obscured fraction
    "agn_mu_bh_zs",                                       # M_BH–M* zero-point evolution
    # ---- missing-physics extension (docs/missing_physics.rst) ----
    "eps_sn",                                             # SN feedback coupling (energy closure)
    "w0", "wa", "sum_mnu",                                # beyond-ΛCDM: CPL dark energy + Σm_ν [eV]
    "log10_Mq_cen", "mu_q_cen",                           # ZM16 halo quenching (centrals)
    "log10_Mq_sat", "mu_q_sat",                           # ZM16 halo quenching (satellites)
    "dlx_quenched",                                       # L_X–M offset of quenched centrals [dex]
    "agn_xi_rx", "agn_xi_rm", "agn_b_r", "agn_sig_r",     # fundamental plane of BH activity
    "log10_M0_hi", "log10_Mmin_hi", "alpha_hi",           # M_HI(M_h) halo model (VN18)
    # ---- missing-physics wave 2 ----
    "eta_w_norm", "alpha_w",                              # SN wind mass loading (Muratov+15 form)
    "ssfr_ms_norm", "ssfr_ms_slope", "ssfr_ms_zs",        # star-forming main sequence (Speagle+14)
    "dhi_quenched",                                       # HI deficit of quenched centrals [dex]
    # ---- missing-physics wave 3 ----
    "sigma_ms", "dssfr_q",                                # MS scatter + quenched sSFR offset [dex]
    "loii_norm",                                          # log10 L[OII]/SFR calibration (Kennicutt-like)
    "f_loud0", "beta_loud", "b_jet",                      # radio-loud jet population (HERG/LERG)
    "agn_bc_ir",                                          # log10 L_IR(6um)/L_bol bolometric correction
    # ---- tier 3: multi-wavelength SED calibrations ----
    "l14_sfr", "alpha_syn",                               # radio-FIR calibration + synchrotron index
    "lir_sfr", "bir_color",                               # L_IR/SFR + dust color tilt across IR bands
    "ml_nir", "ml_opt", "dopt_q",                         # stellar M/L (3.4um; r SF) + quiescent offset
    "luv_norm", "tau_uv_mslope",                          # UV/SFR (attenuated) + attenuation M* slope
    "lha_norm",                                           # log10 L_Halpha/SFR (Kennicutt)
    "agn_bc_uv", "agn_bc_opt",                            # AGN bolometric corrections (1450A, 4400A)
    # ---- missing-physics wave 4: galaxy morphology ----
    "log10_M_morph", "beta_morph",                        # early-type Weibull transition (centrals)
    "f_morph_sat",                                        # satellite early-type boost
    "mbh_bt_slope",                                       # M_BH-bulge coupling (0 = pure Powell chain)
    # ---- tier 4: morphology observables ----
    "rho_morph_q",                                        # morphology-quenching correlation at fixed M_h
    "log10_f_size", "dsize_early",                        # R_e/R_200c (Kravtsov13) + early offset [dex]
    "f_size_zs",                                          # size-ratio evolution slope (via _Z_EVOL)
    "a_ia",                                               # NLA IA amplitude carried by early types
]
_IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
N_PARAM = len(PARAM_NAMES)

# Tier-2 extension bookkeeping: names appended beyond the 31-entry tier-1 vector.
TIER2_PROMOTED = PARAM_NAMES[31:47]
TIER2_ZSLOPES = PARAM_NAMES[47:54]
TIER2_SPECTRAL = PARAM_NAMES[54:61]
# Missing-physics extension (docs/missing_physics.rst): SN coupling, beyond-ΛCDM
# cosmology, SF/quiescent quenching, the AGN fundamental plane, the HI sector,
# and (wave 2) SN wind loading, the star-forming main sequence and the
# quenched-HI deficit.  Frozen slices: later extensions append AFTER [90:].
MISSING_PHYSICS = list(PARAM_NAMES[61:90])
TIER2_EXTENSION = list(PARAM_NAMES[31:90])
# Tier-3 extension: multi-wavelength SED calibrations (radio/IR maps, band LFs).
TIER3_EXTENSION = list(PARAM_NAMES[90:102])
# Missing-physics wave 4: conditional galaxy morphology + the BH-bulge link.
WAVE4_MORPHOLOGY = list(PARAM_NAMES[102:106])
# Tier-4 morphology observables: joint E/Q fractions, sizes, AGN hosts, IA.
TIER4_MORPHOLOGY = list(PARAM_NAMES[106:])

# base parameter → its ln(1+z) evolution slope (applied in ForwardModel._theta_eff)
_Z_EVOL = {"lg_m1h": "lg_m1h_zs", "lg_m0star": "lg_m0star_zs",
           "sigma_lnmstar": "sigma_lnmstar_zs", "lx_norm": "lx_zs",
           "kt_norm": "kt_zs", "agn_log10_ferdf": "agn_log10_ferdf_zs",
           "agn_log10_lstar": "agn_log10_lstar_zs", "agn_mu_bh": "agn_mu_bh_zs",
           "ssfr_ms_norm": "ssfr_ms_zs", "log10_f_size": "f_size_zs"}
# (z_gas_zs acts on log10 Z inside _gas_log10Z — the base z_gas_norm is linear)

OBSERVABLES = ["wp", "ds", "cl_gX", "cl_gy", "cl_XX", "cl_kk",
               "cl_kCMB", "cl_gkCMB", "cl_shear_kCMB", "xlf", "n_gal", "smf"]

_C_OVER_H0 = 2997.92   # c/H0 [Mpc/h]  (H0 = 100 h km/s/Mpc)

# Fixed (non-varied) cosmology + HOD nuisance held at Planck18 / ZM15-MAP values.
# w0/wa/sum_mnu fallbacks are the ΛCDM/massless limits (the missing-physics
# extension promotes them into the vector with these same fiducials).
_FIXED_COSMO = {"h": 0.6736, "Omega_b": 0.0493, "n_s": 0.9649,
                "w0": -1.0, "wa": 0.0, "sum_mnu": 0.0}
_FIXED_HOD = {
    "log10m_star_thresh": 10.0,   # BGS M* > 10 threshold sample
    "beta_sat": 0.9, "bcut": 0.86, "beta_cut": 0.41, "alpha_sat": 1.0,
}

# gNFW emissivity shape constants (fit_xray_joint_bands convention: the outer
# slope is α_out = α_in + 2·p2 with α_in fixed, α_tr fixed).
_ALPHA_IN_GAS = 0.9
_ALPHA_TR_GAS = 2.0
_LX_PIVOT = 15.0   # log10 M500c pivot for L_X–M
_KT_PIVOT = 14.0   # log10 M500c pivot for kT–M
_MPC_CM = 3.0856775814913673e24

# Baryon-feedback sector held-fixed nuisances (BaryonFractionSigmoid defaults +
# the η(M) gas-concentration break); only log10_M_pivot and log10_eta_min vary.
_FIXED_BARYON = {"beta_b": 1.5, "log10_M_eta": 13.0, "beta_eta": 1.5, "f_b_min": 0.01}
_BARYON_FRACTION = BaryonFractionSigmoid()
# Fixed extra shape of the double-sigmoid "upturn" f_b(M) (BaryonFractionUpturn):
# a low-mass upturn added on top of the same group-scale sigmoid whose pivot is the
# free `log10_M_pivot` (reused as M_hi).  Selected via ForwardModel(baryon_model="upturn").
_UPTURN = {"f_b_lo_amp": 0.05, "log10_M_lo": 11.5, "beta_lo": 2.0}

# First-order feedback energetics (optional "energy-closure" mode).  The missing
# baryons Δf_b·M are displaced against the halo binding energy ~ M v200²; the
# available energy is a coupling fraction of the BH rest mass (M_BH c²) plus a
# fraction of the SN energy budget (∝ M*).  Requiring balance turns the baryon
# fraction into a *prediction* from the (X-ray-constrained) AGN + stellar sector.
_C2_KMS2 = 8.98755e10     # c² [(km/s)²]
_G_KMS2 = 4.30091e-9      # G [Mpc (km/s)² / Msun]  → v200²=G M/r200
_E_SN_PER_MSUN = 5.0e5    # SN energy per Msun of stars [(km/s)²]  (~1e49 erg/Msun)
_EPS_SN = 0.1             # SN coupling efficiency (fixed, first order)
# AGN energy tied to the *measured* X-ray luminosity + duty cycle: the same
# log10DC that sets the X-ray amplitude in C_gX/C_XX also sets the feedback
# energy, so X-ray data pins ε_couple → predicts f_b → de-contaminates lensing.
_K_BOL = 20.0                       # X-ray(0.5-2 keV)→bolometric correction
_T_HUBBLE_S = 4.35e17               # Hubble time [s]
_ERG_PER_MSUN_KMS2 = 1.989e43       # 1 Msun (km/s)² in erg
_LX_ON_NORM, _LX_ON_SLOPE = 43.0, 1.0   # L_X^on = 10^43 (M/1e13)^1 erg/s (when active)

# Powell 2022 AGN-halo XLF sector (see hod_mod.agn.powell for the validated
# standalone model).  L_bol = 1.26e38·M_BH·λ [erg/s]; hard L_X = L_bol / 20.
_POWELL_LBOL_COEF = 1.26e38
_POWELL_KBOL_HARD = 20.0
_POWELL_LOGK = float(np.log10(_POWELL_LBOL_COEF / _POWELL_KBOL_HARD))  # log10 L_X/(M_BH·λ)
_SIG_MSTAR_XLF = 0.20   # fixed log10 M* scatter entering the (shift-invariant) M_BH–M_halo width
_SIG_MHI = 0.35         # fixed lognormal scatter of M_HI at fixed M_h [dex]
# missing-physics wave 3 constants
_SIG_SSFR_Q = 0.5       # width of the quenched log sSFR lognormal [dex]
_SIG_JET = 0.7          # scatter of the radio-loud jet luminosity at fixed M_BH [dex]
_SIG_OII = 0.2          # extra [OII]-calibration scatter beyond the MS width [dex]
_POWELL_LOGBOL = float(np.log10(_POWELL_LBOL_COEF))   # log10 L_bol/(M_BH·λ)
# ---- tier 3: band-LF calibration scatters + cluster selection --------
_SIG_UV = 0.2           # UV/SFR calibration scatter [dex]
_SIG_HA = 0.2           # Halpha/SFR calibration scatter [dex] (== _SIG_OII)
_SIG_OPT = 0.2          # optical M/L scatter [dex]
_SIG_NIR = 0.2          # NIR M/L scatter [dex]
_SIG_LXCL = 0.25        # cluster L_X selection scatter at fixed M_h [dex]
# IR band templates (anchors at 3.4/4.9/12 um, log-interpolated per band):
# fraction of L_IR emerging per band (dust), stellar-continuum weight
# relative to the 3.4 um M/L, and the AGN-torus band fraction.
_IR_TPL_LAM = np.array([3.4, 4.9, 12.0])
_IR_TPL_DUST = np.array([0.02, 0.05, 0.15])
_IR_TPL_STAR = np.array([1.0, 0.6, 0.1])
_IR_TPL_TORUS = np.array([0.15, 0.30, 0.50])
# hard(2-10 keV) → soft(0.5-2 keV) energy-flux ratio for a Γ≈1.8 power law
# (hod_mod.agn.ham convention; becomes Γ-dependent in the multi-band layer).
from hod_mod.agn.ham import _HARD_TO_SOFT_RATIO as _K_H2S_FID
_LOG10_K_H2S = float(np.log10(_K_H2S_FID))
# fiducial AGN photon index and the Γ=1.8 soft/hard power-law flux ratio used to
# calibrate the differentiable k_h2s(Γ) so that k_h2s(1.8) == _K_H2S_FID exactly.
_GAMMA_AGN_FID = 1.8
_P_FID = 2.0 - _GAMMA_AGN_FID
_R_H2S_FID = float((2.0 ** _P_FID - 0.5 ** _P_FID) / (10.0 ** _P_FID - 2.0 ** _P_FID))
_NH_ABS = 1.0e22   # obscured-AGN absorption column [cm⁻²] for the t_b template


def _bilinear_logtable(tab, lt_grid, lz_grid, lt, lz):
    r"""10**bilinear interpolation of a log10 Λ(log10 T, log10 Z) table.

    ``tab`` (nT, nZ); ``lt``/``lz`` broadcastable query arrays, clipped to the
    grid (the clip freezes derivatives outside — only the >60 keV tail of the
    most massive halos, negligible emission weight).  Differentiable in lt/lz.
    """
    lt, lz = jnp.broadcast_arrays(jnp.asarray(lt), jnp.asarray(lz))
    lt = jnp.clip(lt, lt_grid[0], lt_grid[-1])
    lz = jnp.clip(lz, lz_grid[0], lz_grid[-1])
    it = jnp.clip(jnp.searchsorted(lt_grid, lt) - 1, 0, lt_grid.size - 2)
    iz = jnp.clip(jnp.searchsorted(lz_grid, lz) - 1, 0, lz_grid.size - 2)
    ft = (lt - lt_grid[it]) / (lt_grid[it + 1] - lt_grid[it])
    fz = (lz - lz_grid[iz]) / (lz_grid[iz + 1] - lz_grid[iz])
    v = (tab[it, iz] * (1.0 - ft) * (1.0 - fz) + tab[it + 1, iz] * ft * (1.0 - fz)
         + tab[it, iz + 1] * (1.0 - ft) * fz + tab[it + 1, iz + 1] * ft * fz)
    return 10.0 ** v


# ---------------------------------------------------------------------------
# JAX Gauss–Legendre radial Fourier transform of a spherical profile
# ---------------------------------------------------------------------------

def _gl_nodes(n_gl: int):
    x, w = np.polynomial.legendre.leggauss(n_gl)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)   # on [0,1]


def _profile_uk_normalized(k, r_max, f_nodes, gx, gw):
    r"""Normalised FT û(k|M) = ∫ f(r) j₀(kr) r² dr / ∫ f(r) r² dr, → 1 as k→0.

    Parameters
    ----------
    k : (Nk,)              wavenumbers [h/Mpc]
    r_max : (NM,)          truncation radius per halo [Mpc/h]
    f_nodes : (NM, Ngl)    profile values at the GL radial nodes
    gx, gw : (Ngl,)        GL nodes/weights on [0, 1]
    """
    r_nodes = r_max[:, None] * gx[None, :]          # (NM, Ngl)
    dr_w = r_max[:, None] * gw[None, :]             # (NM, Ngl)  (∫ dr weight)
    base = f_nodes * r_nodes ** 2 * dr_w            # (NM, Ngl)
    den = jnp.sum(base, axis=1)                     # (NM,)
    kr = k[:, None, None] * r_nodes[None, :, :]     # (Nk, NM, Ngl)
    j0 = jnp.where(kr < 1e-7, 1.0, jnp.sin(kr) / jnp.where(kr < 1e-7, 1.0, kr))
    num = jnp.sum(base[None, :, :] * j0, axis=2)    # (Nk, NM)
    return num / den[None, :]                       # (Nk, NM)


def _gnfw_sq(x, alpha_in, alpha_tr, alpha_out):
    r"""[f_gNFW(x)]² for the n_e² emissivity shape (f from DPM Eq. 1)."""
    xs = jnp.maximum(x, 1e-8)
    f = xs ** (-alpha_in) * (1.0 + xs ** alpha_tr) ** ((alpha_in - alpha_out) / alpha_tr)
    return f ** 2


def _gnfw(x, alpha_in, alpha_tr, alpha_out):
    r"""gNFW shape f(x) (density; DPM Eq. 1) — used for the gas mass profile."""
    xs = jnp.maximum(x, 1e-8)
    return xs ** (-alpha_in) * (1.0 + xs ** alpha_tr) ** ((alpha_in - alpha_out) / alpha_tr)


def _gnfw_pressure(x, alpha_in, alpha_tr, alpha_out):
    r"""A10 GNFW pressure shape p(x) (Arnaud+2010 universal profile)."""
    xs = jnp.maximum(x, 1e-8)
    return xs ** (-alpha_in) * (1.0 + xs ** alpha_tr) ** ((alpha_in - alpha_out) / alpha_tr)


# A10 universal GNFW pressure slopes (Arnaud+2010).
_A10 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)


def _c_dk15(sigma, n_eff):
    r"""Diemer & Kravtsov 2015 c200c(ν, n_eff), Diemer & Joyce 2019 median
    parameters — a pure-jnp mirror of
    :func:`hod_mod.core.concentration.c_diemer15` (cross-validated in tests).
    Cosmology enters through BOTH inputs: ν = δ_c/σ(M, z) and the local P(k)
    slope n_eff, so the concentration finally responds to σ8, n_s, h and (via
    growth) to w0/wa/Σm_ν.
    """
    phi0, phi1 = 6.58, 1.27
    eta0, eta1 = 7.28, 1.56
    alpha, beta = 1.08, 1.77
    nu = 1.686 / sigma
    x = nu / (eta0 + n_eff * eta1)
    c_min = phi0 + n_eff * phi1
    return 0.5 * c_min * (x ** (-alpha) + x ** beta)     # DK15 Eq. 9 / DJ19 Eq. 30

_LN10 = float(np.log(10.0))


# ---------------------------------------------------------------------------
# Differentiable inverse SHMR M*(M_h) — value from the production bisection,
# gradient from the smooth forward SHMR via the implicit function theorem.
# (jacfwd through the raw lax.fori_loop / jnp.where bisection gives a WRONG
# gradient; this custom_jvp fixes it while reusing the same value.)
# ---------------------------------------------------------------------------

@custom_jvp
def _inv_shmr(log10mh, m1, m0, be, de, ga):
    return _mstar_from_mh_zu15(log10mh, m1, m0, be, de, ga)


@_inv_shmr.defjvp
def _inv_shmr_jvp(primals, tangents):
    log10mh, m1, m0, be, de, ga = primals
    dmh, dm1, dm0, dbe, dde, dga = tangents
    y = _mstar_from_mh_zu15(log10mh, m1, m0, be, de, ga)          # M*(M_h) value
    # F(y, p) = _mh_from_mstar_zu15(y, p) − log10mh = 0  ⇒  dy = (dmh − dF_p)/F_y
    F_y = jax.grad(lambda yy: jnp.sum(_mh_from_mstar_zu15(yy, m1, m0, be, de, ga)))(y)
    _, dF_p = jax.jvp(lambda a, b, c, d, e: _mh_from_mstar_zu15(y, a, b, c, d, e),
                      (m1, m0, be, de, ga), (dm1, dm0, dbe, dde, dga))
    dy = (dmh - dF_p) / F_y
    return y, dy


def _n_cen(log10m_h, thr, m1, m0, be, de, ga, sig0, eta, fc):
    """ZM15 central occupation (Eq. 21) with the differentiable inverse SHMR."""
    log10m_star_c = _inv_shmr(log10m_h, m1, m0, be, de, ga)
    sigma = sigma_lnmstar_zu15(log10m_h, m1, sig0, eta)
    arg = (thr - log10m_star_c) * _LN10 / (jnp.sqrt(2.0) * sigma)
    return (fc / 2.0) * erfc(arg)


def _n_sat(log10m_h, thr, m1, m0, be, de, ga, sig0, eta, fc,
           bsat, beta_sat, bcut, beta_cut, alpha_sat):
    """ZM15 satellite occupation (Eq. 22); M_min uses the smooth forward SHMR."""
    log10m_min = _mh_from_mstar_zu15(thr, m1, m0, be, de, ga)
    m_min_norm = jnp.power(10.0, log10m_min - 12.0)
    msat = bsat * jnp.power(m_min_norm, beta_sat) * 1e12
    mcut = bcut * jnp.power(m_min_norm, beta_cut) * 1e12
    nc = _n_cen(log10m_h, thr, m1, m0, be, de, ga, sig0, eta, fc)
    m_h = jnp.power(10.0, log10m_h)
    return nc * jnp.power(m_h / msat, alpha_sat) * jnp.exp(-mcut / m_h)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

class ForwardModel:
    """Differentiable forward model producing the five forecast data vectors.

    Parameters
    ----------
    z_eff : float
        Effective redshift of the (single) representative sample.
    n_k, n_m : int
        Wavenumber and mass grid sizes.
    log10m_min : float
        Lower edge of the halo-mass grid [log10 Msun/h].  The 10.0 default is
        bit-identical to tiers 1-2; low-mass samples (M* down to 10^9) need
        8.5 together with a larger ``n_m``.
    n_gl : int
        Gauss–Legendre nodes for the gas/pressure radial FT.
    rp_wp, rp_ds : array
        Projected-radius grids [Mpc/h] for w_p and ΔΣ.
    ell : array
        Multipole grid for the angular spectra.
    nz_sig : float
        Gaussian n(z) width (top-hat-ish window) around ``z_eff`` for Limber.
    """

    def __init__(
        self,
        z_eff: float = 0.2,
        n_k: int = 256,
        n_m: int = 256,
        n_gl: int = 96,
        rp_wp=None,
        rp_ds=None,
        ell=None,
        nz_sig: float = 0.05,
        n_z: int = 11,
        galaxy_nz: tuple = None,
        z_src_mean: float = 0.8,
        z_src_sig: float = 0.30,
        n_z_shear: int = 12,
        n_shear_bins: int = 1,
        energy_closure: bool = False,
        log10m_star_thresh: float = None,
        log10m_star_bin: tuple = None,
        baryon_model: str = "sigmoid",
        z_pivot_evol: float = 0.3,
        agn_emission: str = "surrogate",
        agn_lx_bins=None,
        rp_wp_agn=None,
        xray_bands=None,
        xlf_band: str = "hard",
        nh_abs: float = _NH_ABS,
        cm_relation: str = "dutton14",
        sfq: str = None,
        ssfr_cut: float = None,
        morph: str = None,
        loglr_rlf=None,
        logmhi_himf=None,
        logloii_oiilf=None,
        loglir_ilf=None,
        pk_correction: str = "none",
        log10m_min: float = 10.0,
        radio_map_bands=None,
        ir_map_bands=None,
        logluv_uvlf=None,
        loglopt_optlf=None,
        loglnir_nirlf=None,
        loglha_half=None,
        logluv_qlf=None,
        loglopt_qlf=None,
        logl_ncl: float = None,
    ):
        self.z_eff = float(z_eff)
        # tier-2 z-evolution: static per-model lever arm ln[(1+z_eff)/(1+z_p)]
        self.z_pivot_evol = float(z_pivot_evol)
        self._x_evol = float(np.log((1.0 + self.z_eff) / (1.0 + self.z_pivot_evol)))
        # tier-2 volume-limited M*-BIN sample [lo, hi): occupation = N(>lo) − N(>hi)
        if log10m_star_bin is not None:
            if log10m_star_thresh is not None:
                raise ValueError("log10m_star_bin and log10m_star_thresh are mutually exclusive")
            lo, hi = (float(log10m_star_bin[0]), float(log10m_star_bin[1]))
            if not hi > lo:
                raise ValueError("log10m_star_bin must satisfy hi > lo")
            log10m_star_thresh, self._thr_hi = lo, hi
        else:
            self._thr_hi = None
        # per-bin stellar-mass threshold override (tomography); default = _FIXED_HOD
        self._thr = (float(log10m_star_thresh) if log10m_star_thresh is not None
                     else _FIXED_HOD["log10m_star_thresh"])
        # When True, the baryon fraction is *derived* from the feedback energy
        # budget (the `log10_M_pivot` slot is reinterpreted as log10 ε_AGN, the
        # AGN mechanical coupling efficiency) rather than the free sigmoid.
        self.energy_closure = bool(energy_closure)
        # f_b(M) shape: "sigmoid" (default, monotonic) or "upturn"/"double_sigmoid"
        # (adds a fixed low-mass upturn; the free log10_M_pivot becomes the group
        # pivot M_hi so the parameter vector is unchanged for a clean comparison).
        self.baryon_model = str(baryon_model).lower()
        self._fb_model = make_baryon_fraction(self.baryon_model)
        # linear P(k): the EH98 shape, optionally corrected by the linearized
        # CAMB ratio table (missing-physics wave 2; spectrum + first
        # derivatives CAMB-accurate near the fiducial)
        self.pk_correction = str(pk_correction).lower()
        if self.pk_correction == "camb_linear":
            from hod_mod.forecast.pk_camb_ratio import load as _load_ratio
            self._pk = EisensteinHu98PkLinear(camb_ratio=_load_ratio())
        elif self.pk_correction == "none":
            self._pk = EisensteinHu98PkLinear()
        else:
            raise ValueError("pk_correction must be 'none' or 'camb_linear'")
        self._hmf = HaloMassFunction(self._pk.as_hmf_pk_func(), model="tinker08", Delta=200.0)

        self.k = jnp.logspace(-4.0, jnp.log10(200.0), n_k)
        self.log_k = jnp.log(self.k)
        # Mass-grid floor: 10.0 covers M* >= 10^10 samples; low-mass (tier-3)
        # cells need ~8.5 so that M_h(M* = 10^9) ~ 2.4e10 sits well inside the
        # grid.  The inverse-SHMR bracket bounds the usable floor at ~8.5.
        if not 8.4 <= float(log10m_min) <= 12.0:
            raise ValueError("log10m_min must lie in [8.4, 12.0]")
        self.m = jnp.logspace(float(log10m_min), 16.0, n_m)
        self.log10m = jnp.log10(self.m)
        self.gx, self.gw = _gl_nodes(n_gl)

        self.rp_wp = jnp.asarray(rp_wp if rp_wp is not None else np.logspace(-1.3, 1.7, 24))
        self.rp_ds = jnp.asarray(rp_ds if rp_ds is not None else np.logspace(-1.3, 1.3, 20))
        self.ell = jnp.asarray(ell if ell is not None else np.logspace(1.0, 3.5, 24))

        # n(z) window for the Limber projection: a real survey dN/dz via
        # ``galaxy_nz=(z_grid, nz)`` (for fitting measured angular spectra), else
        # a synthetic narrow Gaussian at (z_eff, nz_sig) for forecasting.
        if galaxy_nz is not None:
            zg = np.asarray(galaxy_nz[0], dtype=float)
            nz = np.asarray(galaxy_nz[1], dtype=float)
        else:
            zc = self.z_eff
            zg = np.linspace(max(1e-3, zc - 4 * nz_sig), zc + 4 * nz_sig, n_z)
            nz = np.exp(-0.5 * ((zg - zc) / nz_sig) ** 2)
        self.z_grid = jnp.asarray(zg)
        self.nz = jnp.asarray(nz / np.trapezoid(nz, zg))
        self.pi_max = 100.0

        # Cosmic-shear source distribution + lens-distance integration grid.
        # n_shear_bins == 1 (tier-1): a single Gaussian n_s(z) bin.
        # n_shear_bins > 1 (tier-2): a Smail-type overall distribution
        # n(z) ∝ z² exp[−(z/z0)^1.5] with ⟨z⟩ = z_src_mean, split into
        # equal-number tomographic bins by photo-z (σ_z = 0.05(1+z)) selection.
        self.n_shear_bins = int(n_shear_bins)
        if self.n_shear_bins > 1 and n_z_shear < 3 * self.n_shear_bins:
            raise ValueError(f"n_z_shear={n_z_shear} cannot resolve "
                             f"{self.n_shear_bins} tomographic bins; "
                             f"use n_z_shear >= {3 * self.n_shear_bins}")
        if self.n_shear_bins == 1:
            zs = np.linspace(0.03, max(2.0, z_src_mean + 3 * z_src_sig), n_z_shear)
            ns = np.exp(-0.5 * ((zs - z_src_mean) / z_src_sig) ** 2)
            self.z_shear = jnp.asarray(zs)
            self.nz_src = jnp.asarray(ns / np.trapezoid(ns, zs))
            self.nz_src_bins = [self.nz_src]
        else:
            import math
            from scipy.special import erf as _np_erf
            z0 = z_src_mean * math.gamma(2.0) / math.gamma(8.0 / 3.0)   # ⟨z⟩ = z0·Γ(8/3)/Γ(2)
            zs_max = max(2.0, 3.3 * z_src_mean)
            zf = np.linspace(0.01, zs_max, 400)                         # fine grid: bin edges
            nf = zf ** 2 * np.exp(-((zf / z0) ** 1.5))
            cdf = np.cumsum(nf); cdf = cdf / cdf[-1]
            q = np.interp(np.linspace(0.0, 1.0, self.n_shear_bins + 1)[1:-1], cdf, zf)
            edges = np.concatenate([[0.0], q, [zs_max + 1.0]])
            zs = np.linspace(0.03, zs_max, n_z_shear)
            n_true = zs ** 2 * np.exp(-((zs / z0) ** 1.5))
            sig_z = 0.05 * (1.0 + zs)                                   # photo-z scatter
            self.z_shear = jnp.asarray(zs)
            self.nz_src = jnp.asarray(n_true / np.trapezoid(n_true, zs))
            self.nz_src_bins = []
            for i in range(self.n_shear_bins):
                w = 0.5 * (_np_erf((edges[i + 1] - zs) / (np.sqrt(2.0) * sig_z))
                           - _np_erf((edges[i] - zs) / (np.sqrt(2.0) * sig_z)))
                nb = n_true * w
                self.nz_src_bins.append(jnp.asarray(nb / np.trapezoid(nb, zs)))
        # unique source-bin pairs (i ≤ j) — the cl_kk tomographic spectra order
        self.shear_pairs = [(i, j) for i in range(self.n_shear_bins)
                            for j in range(i, self.n_shear_bins)]

        # Powell-XLF grids: Eddington-ratio (log λ) integration grid over the
        # Ananna 2022 range, and the log10 L_X [erg/s] abscissa of the observable.
        loglam = np.linspace(-3.0, 1.5, 120)
        self.loglam = jnp.asarray(loglam)
        self.dlam = float(loglam[1] - loglam[0])
        self.loglx_xlf = jnp.asarray(np.linspace(42.0, 45.0, 7))

        # AGN sector mode: "surrogate" keeps the tier-1 L∝M point-source term in
        # C_gX/C_XX; "powell" replaces it with the Powell-chain mean emission so
        # the X-ray spectra constrain the same AGN parameters as the XLF.
        self.agn_emission = str(agn_emission).lower()
        if self.agn_emission not in ("surrogate", "powell"):
            raise ValueError("agn_emission must be 'surrogate' or 'powell'")
        # wp_agn: projected clustering of complete L_X-selected AGN samples.
        # Bins are SOFT-band (0.5-2 keV) log10 L_X edges (the survey selection);
        # the Powell kernel is hard-band, so edges shift by log10(k_h2s).
        bins = agn_lx_bins if agn_lx_bins is not None else \
            [(42.0, 42.5), (42.5, 43.0), (43.0, 43.5), (43.5, 44.0)]
        self.agn_lx_bins = [(float(a), float(b)) for a, b in bins]
        self.rp_wp_agn = jnp.asarray(rp_wp_agn if rp_wp_agn is not None
                                     else np.logspace(0.0, 1.5, 8))

        # ---- missing-physics extension (docs/missing_physics.rst) -------
        # concentration–mass relation: "dutton14" (tier-1/2 default, fixed
        # Planck13 fit) or "diemer15" (cosmology-dependent c(ν, n_eff)).
        self.cm_relation = str(cm_relation).lower()
        if self.cm_relation not in ("dutton14", "diemer15"):
            raise ValueError("cm_relation must be 'dutton14' or 'diemer15'")
        # SF/quiescent sample split: None (all galaxies, tier-1/2 default),
        # "sf" or "q" — ZM16 Weibull quenching weights on the occupations.
        self.sfq = None if sfq is None else str(sfq).lower()
        if self.sfq not in (None, "sf", "q"):
            raise ValueError("sfq must be None, 'sf' or 'q'")
        # wave 4: morphology sample split — None (all), "early" or "late";
        # Weibull early-type weights on the occupations, composable with the
        # SF/Q split (EARLY + LATE ≡ unsplit exactly, like SF + Q).
        self.morph = None if morph is None else str(morph).lower()
        if self.morph not in (None, "early", "late"):
            raise ValueError("morph must be None, 'early' or 'late'")
        # radio LF abscissa (5 GHz νLν [erg/s], fundamental-plane observable)
        self.loglr_rlf = jnp.asarray(loglr_rlf if loglr_rlf is not None
                                     else np.linspace(36.0, 42.0, 7))
        # HI mass-function abscissa (log10 M_HI [Msun/h])
        self.logmhi_himf = jnp.asarray(logmhi_himf if logmhi_himf is not None
                                       else np.linspace(8.5, 11.0, 7))
        # wave 3: sSFR-threshold selection (log10 sSFR [yr⁻¹]; e.g. −10.5 for
        # an ELG-like star-forming cut) — composes with the sfq split
        self.ssfr_cut = None if ssfr_cut is None else float(ssfr_cut)
        # [OII] LF and AGN IR LF abscissas [erg/s]
        self.logloii_oiilf = jnp.asarray(logloii_oiilf if logloii_oiilf is not None
                                         else np.linspace(40.5, 43.5, 7))
        self.loglir_ilf = jnp.asarray(loglir_ilf if loglir_ilf is not None
                                      else np.linspace(42.5, 46.0, 7))

        # ---- tier 3: radio / IR intensity-map bands + band-LF abscissas --
        # radio bands in GHz (SKA-like), IR bands in um (WISE/SPHEREx-like);
        # None keeps the map observables off (opt-in, like the LF grids).
        self.radio_map_bands = (None if radio_map_bands is None
                                else tuple(float(b) for b in radio_map_bands))
        self.ir_map_bands = (None if ir_map_bands is None
                             else tuple(float(b) for b in ir_map_bands))
        if self.ir_map_bands is not None:
            llam = np.log10(np.asarray(self.ir_map_bands))
            ltpl = np.log10(_IR_TPL_LAM)
            self._ir_cdust = np.interp(llam, ltpl, _IR_TPL_DUST)
            self._ir_cstar = np.interp(llam, ltpl, _IR_TPL_STAR)
            self._ir_ctorus = np.interp(llam, ltpl, _IR_TPL_TORUS)
            # dust color tilt lever arm, anchored at 4.9 um (bir_color inert there)
            self._ir_dlognu = np.log10(4.9 / np.asarray(self.ir_map_bands))
        # galaxy band-LF abscissas [log10 nuLnu erg/s] + AGN UV/opt LFs
        self.logluv_uvlf = jnp.asarray(logluv_uvlf if logluv_uvlf is not None
                                       else np.linspace(41.5, 44.5, 7))
        self.loglopt_optlf = jnp.asarray(loglopt_optlf if loglopt_optlf is not None
                                         else np.linspace(41.5, 44.5, 7))
        self.loglnir_nirlf = jnp.asarray(loglnir_nirlf if loglnir_nirlf is not None
                                         else np.linspace(42.0, 45.0, 7))
        self.loglha_half = jnp.asarray(loglha_half if loglha_half is not None
                                       else np.linspace(40.3, 43.3, 7))
        self.logluv_qlf = jnp.asarray(logluv_qlf if logluv_qlf is not None
                                      else np.linspace(43.0, 46.5, 7))
        self.loglopt_qlf = jnp.asarray(loglopt_qlf if loglopt_qlf is not None
                                       else np.linspace(43.0, 46.5, 7))
        # X-ray cluster-count selection: log10 L_X limit [erg/s] at this z
        self.logl_ncl = None if logl_ncl is None else float(logl_ncl)

        # ---- tier-2 multi-band APEC X-ray layer -------------------------
        # xray_bands=None keeps the tier-1 broad-band model; a list of (emin,
        # emax) keV edges switches cl_gX/cl_XX to per-band stacks built from
        # APEC Λ_b(T, Z) tables (distilled once via hod_mod.forecast.apec_bands).
        self.xlf_band = str(xlf_band).lower()
        if self.xlf_band not in ("hard", "soft"):
            raise ValueError("xlf_band must be 'hard' or 'soft'")
        from hod_mod.forecast import apec_bands as _AB
        # broad soft-band transmission for the obscured-AGN XLF / point-source
        # absorption (MM83 template at NH=nh_abs; no soxs needed).
        self._t_soft = float(_AB.band_transmission([_AB.BROAD_BAND], nh=nh_abs,
                                                   gamma=_GAMMA_AGN_FID)[0])
        self.xray_bands = None
        if xray_bands is not None:
            self.xray_bands = [(float(a), float(b)) for a, b in xray_bands]
            tabs = _AB.band_tables(self.xray_bands)
            self._apec_lt = jnp.asarray(tabs["lt"])
            self._apec_lz = jnp.asarray(tabs["lz"])
            self._apec_tables = [jnp.asarray(t) for t in tabs["tables"][:-1]]
            self._apec_broad = jnp.asarray(tabs["tables"][-1])
            self._band_edges = np.asarray(self.xray_bands, dtype=float)
            self._t_bands = jnp.asarray(_AB.band_transmission(
                self.xray_bands, nh=nh_abs, gamma=_GAMMA_AGN_FID))

        # Stellar-mass-function abscissa (log10 M* [Msun/h]) for the SMF observable.
        self.logmstar_smf = jnp.asarray(np.linspace(10.0, 11.6, 9))

        # CMB lensing: last-scattering redshift + a line-of-sight integration grid
        # for the κ_CMB auto spectrum (the kernel peaks at z~2; a grid to z~6
        # captures the bulk — the high-z tail is negligible for ℓ<3000).
        self.z_star = 1089.0
        self.z_cmb = jnp.asarray(np.geomspace(0.03, 6.0, 14))

    # ---- tier-2 redshift-evolution mapping ---------------------------
    def _theta_eff(self, theta):
        r"""Effective parameter vector at this model's ``z_eff``.

        For each (base, slope) pair in :data:`_Z_EVOL`,

        .. math:: \theta_{\rm eff}[{\rm base}] = \theta[{\rm base}]
                  + \theta[{\rm slope}]\,\ln\frac{1+z_{\rm eff}}{1+z_{\rm pivot}}

        At the fiducial (all slopes 0) this is the identity, so tier-1 results
        are unchanged; ``jax.jacfwd`` through it yields exactly
        ∂d/∂slope = ln[(1+z)/(1+z_p)]·∂d/∂base by the chain rule.  One shared
        global vector thereby drives every (z, M*) cell of the tier-2 grid.
        """
        theta = jnp.asarray(theta)
        for base, sl in _Z_EVOL.items():
            theta = theta.at[_IDX[base]].add(self._x_evol * theta[_IDX[sl]])
        return theta

    # ---- cosmology / HOD dicts from the flat vector ------------------
    def _cosmo(self, theta):
        # Read from the parameter vector when the name is free, else the frozen
        # Planck18 value — so h, n_s, Omega_b become differentiable simply by being
        # present in PARAM_NAMES.
        def g(name):
            return theta[_IDX[name]] if name in _IDX else _FIXED_COSMO[name]
        Om = theta[_IDX["Omega_m"]]
        Ob = g("Omega_b")
        return {
            "h": g("h"),
            "Omega_b": Ob,
            "Omega_m": Om,
            "Omega_cdm": Om - Ob,
            "n_s": g("n_s"),
            "sigma8": theta[_IDX["sigma8"]],
            # beyond-ΛCDM sector (missing-physics extension).  Their presence
            # as dict KEYS statically routes growth to the CPL ODE and P(k) to
            # the ν-suppressed shape; the ΛCDM/massless fiducials keep both
            # paths numerically at their tier-2 values.
            "w0": g("w0"),
            "wa": g("wa"),
            "sum_mnu": g("sum_mnu"),
        }

    @staticmethod
    def _e2z(c, z):
        """E²(z) for flat CPL dark energy (reduces to ΛCDM at w0=−1, wa=0)."""
        Om = c["Omega_m"]
        fde = (1.0 + z) ** (3.0 * (1.0 + c["w0"] + c["wa"])) \
            * jnp.exp(-3.0 * c["wa"] * z / (1.0 + z))
        return Om * (1.0 + z) ** 3 + (1.0 - Om) * fde

    def _hod(self, theta):
        d = {"log10m_star_thresh": self._thr}
        # tier-2: the satellite-shape nuisances (beta_sat, bcut, beta_cut,
        # alpha_sat) are now vector entries with fiducials == the old _FIXED_HOD.
        for n in ("lg_m1h", "lg_m0star", "beta", "delta", "gamma",
                  "sigma_lnmstar", "eta", "fc", "bsat",
                  "beta_sat", "bcut", "beta_cut", "alpha_sat"):
            d[n] = theta[_IDX[n]]
        return d

    def _fb_eta(self, theta):
        r"""Shared hot-gas sector on the mass grid: (f_b(M), η(M)).

        f_b(M) is the hot-gas mass fraction (BaryonFractionSigmoid, feedback
        expels gas from M < M_pivot); η(M) is the gas-concentration ratio
        c_gas/c_DM (gas puffed out towards η_min at group scales).  Both drive
        the ΔΣ baryon split AND the X-ray/tSZ amplitudes & extent.
        """
        c = self._cosmo(theta)
        if self.energy_closure:
            fb = self._fb_energy(theta, c)
        elif self.baryon_model in ("upturn", "double_sigmoid", "valley"):
            # double sigmoid: free log10_M_pivot is the group pivot M_hi; the
            # low-mass upturn (f_b_lo_amp below M_lo) is held at _UPTURN defaults.
            pars = {"f_b_min": _FIXED_BARYON["f_b_min"],
                    "log10_M_hi": theta[_IDX["log10_M_pivot"]],
                    "beta_hi": theta[_IDX["beta_b"]],
                    "f_b_lo_amp": _UPTURN["f_b_lo_amp"],
                    "log10_M_lo": _UPTURN["log10_M_lo"],
                    "beta_lo": _UPTURN["beta_lo"]}
            fb = self._fb_model(self.m, c, pars)                        # (NM,)
        else:
            pars = {"log10_M_pivot": theta[_IDX["log10_M_pivot"]],
                    "beta_b": theta[_IDX["beta_b"]], "f_b_min": _FIXED_BARYON["f_b_min"]}
            fb = self._fb_model(self.m, c, pars)                        # (NM,)
        eta_min = 10.0 ** theta[_IDX["log10_eta_min"]]
        M_eta = 10.0 ** theta[_IDX["log10_M_eta"]]
        eta = 1.0 - (1.0 - eta_min) / (1.0 + (self.m / M_eta) ** theta[_IDX["beta_eta"]])
        # SN wind mass loading (missing-physics wave 2, Muratov+15 form):
        # η_w = η_0 (V_c/200 km/s)^{−α_w} additionally puffs out the low-mass
        # hot gas, η_eff = η_sigmoid/(1 + η_w).  At the fiducial η_0 = 0 this
        # is exactly the tier-2 sigmoid (bit-identical); α_w interpolates the
        # momentum-driven (1) to energy-driven (2) scalings.
        eta_w0 = theta[_IDX["eta_w_norm"]]
        z = self.z_eff
        ez2 = self._e2z(c, z)
        rho_crit_com = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3
        r200 = (3.0 * self.m / (4.0 * jnp.pi * 200.0 * rho_crit_com)) ** (1.0 / 3.0)
        v_c = jnp.sqrt(_G_KMS2 * self.m / r200)                        # [km/s]
        eta_w = eta_w0 * (v_c / 200.0) ** (-theta[_IDX["alpha_w"]])
        eta = eta / (1.0 + eta_w)
        return fb, eta

    def _fb_energy(self, theta, c):
        r"""First-order energy-regulated baryon fraction f_b(M).

        Balances the energy to displace the missing baryons,
        :math:`\Delta f_b\,M\,v_{200}^2`, against the available feedback energy.
        The **AGN channel is tied to the measured X-ray sector**: E_AGN ∝
        ε_couple · 10^{log10DC} · L_X^{on}(M), where ``log10DC`` is the *same*
        duty cycle that sets the X-ray amplitude in C_gX / C_XX — so the X-ray
        data pins ε_couple·DC and thereby *predicts* f_b (and the lensing
        suppression).  The **stellar/SN channel** (∝ M_*(M) from the ZM15 SHMR)
        dominates at low halo mass.  ε_couple is the free parameter (the
        ``log10_M_pivot`` slot, reinterpreted).  Differentiable throughout.
        """
        Om, Ob = c["Omega_m"], c["Omega_b"]
        fbc = Ob / Om
        z = self.z_eff
        ez2 = self._e2z(c, z)
        rho_crit_com = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3
        r200 = (3.0 * self.m / (4.0 * jnp.pi * 200.0 * rho_crit_com)) ** (1.0 / 3.0)
        v200sq = _G_KMS2 * self.m / r200                               # (km/s)²

        # AGN channel — tied to the measured X-ray luminosity × duty cycle
        eps_couple = 10.0 ** theta[_IDX["log10_M_pivot"]]             # slot reused
        dc = 10.0 ** theta[_IDX["log10DC"]]                          # shared with X-ray
        L_on = 10.0 ** _LX_ON_NORM * (self.m / 1.0e13) ** _LX_ON_SLOPE  # erg/s
        E_agn = (eps_couple * dc * _K_BOL * L_on * _T_HUBBLE_S
                 / _ERG_PER_MSUN_KMS2)                                 # Msun (km/s)²

        # stellar/SN channel — low-mass, ∝ M_*(M) from the ZM15 SHMR
        hp = self._hod(theta)
        log10_mstar = _inv_shmr(self.log10m, hp["lg_m1h"], hp["lg_m0star"],
                                hp["beta"], hp["delta"], hp["gamma"])
        E_sn = theta[_IDX["eps_sn"]] * 10.0 ** log10_mstar * _E_SN_PER_MSUN

        expelled = jnp.minimum(fbc - _FIXED_BARYON["f_b_min"],
                               (E_agn + E_sn) / (self.m * v200sq))
        return fbc - jnp.maximum(expelled, 0.0)

    def _neff_eh98(self, c):
        r"""Effective spectral slope n_eff = dln P/dln k at k_R = κ·2π/R(M), (NM,).

        The Diemer & Kravtsov (2015) input alongside the peak height, with the
        Diemer & Joyce (2019) calibration scale κ = 0.42 (their Table 2 median
        c200c model — the same convention the _c_dk15 parameters belong to).
        Computed differentiably from the (ν-suppressed) EH98 shape on a
        uniform-in-ln k grid (jnp.gradient with scalar spacing).
        """
        kappa = 0.42
        rho_m = _RHO_CRIT0 * c["Omega_m"]
        R = (3.0 * self.m / (4.0 * jnp.pi * rho_m)) ** (1.0 / 3.0)
        kg = jnp.logspace(-3.0, 2.0, 256)
        lnk = jnp.log(kg)
        lnp = jnp.log(jnp.maximum(self._pk.pk_shape(kg, c), 1e-30))
        dlnp = jnp.gradient(lnp, float(lnk[1] - lnk[0]))
        return jnp.interp(jnp.log(kappa * 2.0 * jnp.pi / R), lnk, dlnp)

    # ---- shared halo-model quantities on the mass grid ---------------
    def _halo_common(self, theta, z):
        c = self._cosmo(theta)
        Om = c["Omega_m"]
        # 200c halo radius (comoving h-units) and NFW FT
        ez2 = self._e2z(c, z)
        rho_crit_com = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3
        if self.cm_relation == "diemer15":
            # cosmology-dependent c(ν, n_eff) — Diemer & Kravtsov 2015 with
            # Diemer & Joyce 2019 parameters (validated against
            # hod_mod.core.concentration.c_diemer15); responds to σ8/n_s/h and,
            # through the growth factor, to the beyond-ΛCDM sector.
            sig = self._hmf.sigma(self.m, float(z), c)
            conc = _c_dk15(sig, self._neff_eh98(c))
        else:
            conc = concentration_dutton14_jax(self.m, float(z))       # (NM,)
        r_delta = (3.0 * self.m / (4.0 * jnp.pi * 200.0 * rho_crit_com)) ** (1.0 / 3.0)
        r_s = r_delta / conc
        uk = nfw_uk_jax(self.k, r_s, conc)                             # (Nk, NM)

        dndm = self._hmf.dndm(self.m, float(z), c)                     # (NM,)
        bias = self._hmf.bias(self.m, float(z), c)                     # (NM,)

        nc, ns = self._occ_sample(theta)
        nt = nc + ns
        n_gal = jnp.trapezoid(dndm * nt, self.m)
        b_eff = jnp.trapezoid(dndm * nt * bias, self.m) / n_gal
        pk_lin = self._pk.pk_linear(self.k, float(z), c)               # (Nk,)
        rho_m = _RHO_CRIT0 * Om

        # Shared hot-gas sector: f_b(M), η(M), and the reduced-concentration gas
        # NFW FT used for the ΔΣ baryon split (Mead+2015; arXiv:2409.01758).
        fb, eta = self._fb_eta(theta)
        c_gas = conc * eta
        uk_gas = nfw_uk_jax(self.k, r_delta / c_gas, c_gas)            # (Nk, NM)
        return dict(c=c, conc=conc, r_delta=r_delta, r_s=r_s, uk=uk,
                    dndm=dndm, bias=bias, nc=nc, ns=ns, nt=nt,
                    n_gal=n_gal, b_eff=b_eff, pk_lin=pk_lin, rho_m=rho_m, z=z,
                    fb=fb, eta=eta, uk_gas=uk_gas)

    def _gas_density_uk(self, theta, H):
        """Normalised FT of the extended hot-gas *mass* profile (gNFW, →1 at k→0).

        Same shape/extent as the X-ray/tSZ gas (outer slope α_in+2·p2, scale
        r_s/η, truncated at r_max·r200), so the gas that suppresses ΔΣ is the
        *same* hot gas the cross-statistics see.  (Nk, NM)."""
        p2 = theta[_IDX["p2"]]
        a_in, a_tr = theta[_IDX["alpha_in_gas"]], theta[_IDX["alpha_tr_gas"]]
        alpha_out = a_in + 2.0 * p2
        r_max = theta[_IDX["r_max"]] * H["r_delta"]
        r_s = H["r_s"] / H["eta"]
        fn = _gnfw(r_max[:, None] * self.gx[None, :] / r_s[:, None],
                   a_in, a_tr, alpha_out)
        return _profile_uk_normalized(self.k, r_max, fn, self.gx, self.gw)

    # ---- P(k) tables -------------------------------------------------
    def _pk_gg_gm(self, H, theta):
        """1h+2h galaxy auto and galaxy-matter (More+2015 Eqs. 9, 13).

        The galaxy–matter 1-halo term carries the **baryonic-feedback** split:
        a mass fraction f_b(M) of the matter is redistributed from the dark-matter
        NFW into an extended gas NFW (concentration c·η(M)), suppressing ΔΣ at
        small R (clustering/P_gg is unchanged — satellites trace the DM).
        """
        dndm, uk, nc, ns = H["dndm"], H["uk"], H["nc"], H["ns"]
        m, n_gal, b_eff, pk_lin = self.m, H["n_gal"], H["b_eff"], H["pk_lin"]
        integ_gg = dndm[None, :] * (ns[None, :] ** 2 * uk ** 2
                                    + 2.0 * nc[None, :] * ns[None, :] * uk)
        P_gg_1h = jnp.trapezoid(integ_gg, m, axis=1) / n_gal ** 2
        m_over_rho = m / H["rho_m"]
        gal = nc[None, :] + ns[None, :] * uk                          # galaxy FT weight
        # baryons: mass fraction f_b(M) follows the extended hot-gas profile
        # (the same gas the X-ray/tSZ see), the rest stays in the DM NFW.
        fb = H["fb"]
        uk_gas = self._gas_density_uk(theta, H)
        u_matter = (1.0 - fb[None, :]) * uk + fb[None, :] * uk_gas      # CDM + gas
        integ_gm = dndm[None, :] * gal * m_over_rho[None, :] * u_matter
        P_gm_1h = jnp.trapezoid(integ_gm, m, axis=1) / n_gal
        P_gg = P_gg_1h + b_eff ** 2 * pk_lin
        P_gm = P_gm_1h + b_eff * pk_lin
        return P_gg, P_gm

    def _emissivity_uk(self, theta, H):
        """Normalised n_e² gNFW emissivity FT × analytic L_X(M) × band weight.

        Returns X̃(k|M) [arbitrary but self-consistent X-ray units], (Nk, NM).
        """
        p2 = theta[_IDX["p2"]]
        r_max_fac = theta[_IDX["r_max"]]
        a_in, a_tr = theta[_IDX["alpha_in_gas"]], theta[_IDX["alpha_tr_gas"]]
        alpha_out = a_in + 2.0 * p2
        r_max = r_max_fac * H["r_delta"]                                # (NM,)
        # gas extent shared with the ΔΣ split: puffier gas (η<1) → larger r_s
        r_s = H["r_s"] / H["eta"]                                       # (NM,)

        def f_nodes(r):                                                 # r: (NM, Ngl)
            return _gnfw_sq(r / r_s[:, None], a_in, a_tr, alpha_out)

        u_shape = _profile_uk_normalized(self.k, r_max, f_nodes(r_max[:, None] * self.gx[None, :]),
                                         self.gx, self.gw)              # (Nk, NM)
        lx, kT = self._lx_kt_of(theta, H["z"])
        w_band = jnp.maximum(kT, 1e-3) ** 0.25                          # weak band-response weight
        # n_e² emissivity ∝ (gas density)² ∝ f_b(M)²  → shared baryon sector
        amp = lx * w_band * H["fb"] ** 2                               # (NM,), O(1)
        return u_shape * amp[None, :]

    def _lx_kt_of(self, theta, z):
        r"""Analytic L_X(M500c) and kT(M500c) scaling relations, (NM,) each.

        The absolute X-ray amplitude is arbitrary for a relative-error Fisher
        (log-derivs), so lx_norm is referenced to 45 to keep values O(1) and
        float-safe while d ln X / d lx_norm = ln10 is preserved exactly.
        Self-similar E(z)² / E(z)^{2/3} scalings are hardcoded; the free lx_zs /
        kt_zs slopes (via _theta_eff) parameterize departures from them.
        """
        ez = jnp.sqrt(self._e2z(self._cosmo(theta), z))
        log10_m500c = self.log10m + jnp.log10(0.72)                     # M500c ≈ 0.72 M200 (comoving)
        lx = 10.0 ** (theta[_IDX["lx_norm"]] - 45.0
                      + theta[_IDX["lx_slope"]] * (log10_m500c - _LX_PIVOT)) * ez ** 2
        kT = 10.0 ** (theta[_IDX["kt_slope"]] * (log10_m500c - _KT_PIVOT)
                      + theta[_IDX["kt_norm"]]) * ez ** (2.0 / 3.0)
        # SF/quiescent split (missing-physics): quenched centrals carry an
        # L_X–M offset at fixed halo mass — the Zhang+2025 eROSITA CGM signal.
        if self.sfq == "q":
            lx = lx * 10.0 ** theta[_IDX["dlx_quenched"]]
        return lx, kT

    def _gas_log10Z(self, theta):
        r"""ICM metallicity log10 Z(M) [Z_sun], (NM,).

        log10 Z = log10 z_gas_norm + z_gas_mslope·(log10 M500c − 14)
                  + z_gas_zs·ln[(1+z_eff)/(1+z_p)]  (evolution applied here, in
        log space, because the base z_gas_norm parameter is linear).
        """
        log10_m500c = self.log10m + jnp.log10(0.72)
        return (jnp.log10(theta[_IDX["z_gas_norm"]])
                + theta[_IDX["z_gas_mslope"]] * (log10_m500c - 14.0)
                + theta[_IDX["z_gas_zs"]] * self._x_evol)

    def _emissivity_uk_bands(self, theta, H):
        r"""Per-band gas emissivity FTs [X_b(k|M)], each (Nk, NM), APEC-weighted.

        .. math:: \varepsilon_b(r|M) \propto f_{\rm gNFW}(x)^2\,
                  \Lambda_b\!\big(T(r|M),\,Z(M)\big),\qquad
                  T(r|M) = kT(M)\,[f_{\rm gNFW}(x)/f_{\rm gNFW}(1)]^{\,\rm t\_prof\_slope}

        The total amplitude is lx(M)·f_b(M)² — lx_norm keeps setting the 0.5–2
        keV luminosity — partitioned by the emission-weighted band fractions
        w_b = ∫ε_b r²dr / ∫ε_broad r²dr.  Because Σ_b Λ_b(T) = Λ_broad(T)
        pointwise (APEC band additivity), Σ_b X_b equals the broad-band
        prediction exactly, even with a temperature profile.
        """
        p2 = theta[_IDX["p2"]]
        a_in, a_tr = theta[_IDX["alpha_in_gas"]], theta[_IDX["alpha_tr_gas"]]
        alpha_out = a_in + 2.0 * p2
        r_max = theta[_IDX["r_max"]] * H["r_delta"]                     # (NM,)
        r_s = H["r_s"] / H["eta"]                                       # (NM,)
        r_nodes = r_max[:, None] * self.gx[None, :]                     # (NM, Ngl)
        x_nodes = r_nodes / r_s[:, None]
        f_gas = _gnfw(x_nodes, a_in, a_tr, alpha_out)                   # (NM, Ngl)
        f2 = f_gas ** 2
        # radial temperature: polytropic-like tilt on the density shape,
        # anchored at x = 1 (r = r_s) so kT(M) keeps its scaling-relation sense
        lx, kT = self._lx_kt_of(theta, H["z"])
        f_ref = _gnfw(jnp.asarray(1.0), a_in, a_tr, alpha_out)
        lt_nodes = (jnp.log10(kT)[:, None]
                    + theta[_IDX["t_prof_slope"]] * jnp.log10(f_gas / f_ref))
        lz = self._gas_log10Z(theta)[:, None]                           # (NM, 1)
        w_r = r_nodes ** 2 * (r_max[:, None] * self.gw[None, :])        # ∫ r² dr weights
        amp = lx * H["fb"] ** 2                                         # (NM,)
        lams = [_bilinear_logtable(tab, self._apec_lt, self._apec_lz,
                                   lt_nodes, lz)                        # (NM, Ngl)
                for tab in self._apec_tables]
        # broad-band weighting = Σ_b of the interpolated band Λ's, so the
        # amplitude partition Σ_b w_b = 1 is exact by construction (using the
        # broad table's own interpolant would leak O(interp) amplitude).
        den = jnp.sum(f2 * sum(lams) * w_r, axis=1)                     # (NM,)
        outs = []
        for lam_b in lams:
            eps_b = f2 * lam_b
            u_b = _profile_uk_normalized(self.k, r_max, eps_b, self.gx, self.gw)
            w_b = jnp.sum(eps_b * w_r, axis=1) / den                    # (NM,)
            outs.append(u_b * (amp * w_b)[None, :])
        return outs

    def _pressure_uk(self, theta, H):
        """Normalised A10 GNFW pressure FT × P500(M) × mass tilt, (Nk, NM)."""
        r_max = 3.0 * H["r_delta"]
        r_s = (H["r_delta"] / theta[_IDX["c500_pressure"]]) / H["eta"]  # shared gas extent

        def f_nodes(r):
            return _gnfw_pressure(r / r_s[:, None], theta[_IDX["gamma_pressure"]],
                                  theta[_IDX["alpha_pressure"]], theta[_IDX["beta_out_pressure"]])

        u_shape = _profile_uk_normalized(self.k, r_max, f_nodes(r_max[:, None] * self.gx[None, :]),
                                         self.gx, self.gw)
        ez = jnp.sqrt(self._e2z(self._cosmo(theta), H["z"]))
        log10_m500c = self.log10m + jnp.log10(0.72)
        beta_p = 2.0 / 3.0 + 0.12 + theta[_IDX["beta_pressure"]]        # A10 self-similar + tilt
        # thermal energy ∝ gas mass ∝ f_b(M)  → shared baryon sector.  The GNFW
        # amplitude enters as the ratio p0_pressure/P0_fid so the fiducial
        # prediction is unchanged while ∂lnC_gy/∂P0 = 1/P0 is exact.
        P500 = (10.0 ** (beta_p * (log10_m500c - 14.5)) * ez ** (8.0 / 3.0) * H["fb"]
                * (theta[_IDX["p0_pressure"]] / _A10["P0"]))
        return u_shape * P500[None, :]

    # ---- AGN spectral helpers (Γ power law + obscured fraction) ------
    def _k_h2s(self, theta):
        """Differentiable hard→soft flux ratio k_h2s(Γ), == _K_H2S_FID at Γ=1.8."""
        p = 2.0 - theta[_IDX["agn_gamma"]]
        r = (2.0 ** p - 0.5 ** p) / (10.0 ** p - 2.0 ** p)
        return _K_H2S_FID * r / _R_H2S_FID

    def _agn_band_fractions(self, theta):
        """Energy-flux fraction of a Γ power law in each band, normalised over
        the 0.5–2 keV soft band (so bands spanning it sum to 1), (Nb,)."""
        p = 2.0 - theta[_IDX["agn_gamma"]]
        lo, hi = self._band_edges[:, 0], self._band_edges[:, 1]
        return (hi ** p - lo ** p) / (2.0 ** p - 0.5 ** p)

    def _abs_survival_soft(self, theta):
        """Mean broad-soft-band flux survival of the AGN population:
        (1−f_abs) unabsorbed + f_abs transmitted through NH (MM83 template)."""
        fabs = theta[_IDX["agn_fabs"]]
        return (1.0 - fabs) + fabs * self._t_soft

    def _abs_survival_bands(self, theta):
        """Per-band AGN flux survival, (Nb,)."""
        fabs = theta[_IDX["agn_fabs"]]
        return (1.0 - fabs) + fabs * self._t_bands

    # ---- X-ray / tSZ power spectra -----------------------------------
    def _pk_tracer_field(self, theta, H, wc, ws, X, Xa, n_tr, b_tr):
        """tracer × field 1h+2h: tracer occupation wc (central) + ws (satellite,
        NFW-distributed), mean density n_tr, effective bias b_tr; extended
        field FT X (Nk, NM) + flat (point-source) term Xa (NM,)."""
        dndm, uk = H["dndm"], H["uk"]
        m, pk_lin, bias = self.m, H["pk_lin"], H["bias"]
        gal = wc[None, :] + ws[None, :] * uk
        P_1h = jnp.trapezoid(dndm[None, :] * gal * X, m, axis=1) / n_tr
        I_X = jnp.trapezoid(dndm[None, :] * bias[None, :] * X, m, axis=1)
        P_gas = P_1h + b_tr * pk_lin * I_X
        P_agn = jnp.trapezoid(dndm[None, :] * gal * Xa[None, :], m, axis=1) / n_tr \
            + b_tr * pk_lin * jnp.trapezoid(dndm[None, :] * bias[None, :] * Xa[None, :], m, axis=1)
        return P_gas + P_agn

    def _pk_gX_of(self, theta, H, X, Xa):
        """galaxy × X-ray 1h+2h for one gas FT X (Nk, NM) + flat AGN term Xa (NM,)."""
        return self._pk_tracer_field(theta, H, H["nc"], H["ns"], X, Xa,
                                     H["n_gal"], H["b_eff"])

    def _pk_gX(self, theta, H):
        """galaxy × X-ray (gas + AGN), 1h+2h; (Nk,) or (Nb, Nk) in band mode."""
        if self.xray_bands is None:
            return self._pk_gX_of(theta, H, self._emissivity_uk(theta, H),
                                  self._agn_point_source(theta))
        Xbs = self._emissivity_uk_bands(theta, H)
        A = self._agn_emissivity_amp(theta)                             # (NM,) unabsorbed soft
        wb = self._agn_band_fractions(theta) * self._abs_survival_bands(theta)
        return jnp.stack([self._pk_gX_of(theta, H, Xbs[b], A * wb[b])
                          for b in range(len(Xbs))])

    def _pk_gy(self, theta, H):
        Y = self._pressure_uk(theta, H)
        dndm, uk, nc, ns = H["dndm"], H["uk"], H["nc"], H["ns"]
        m, n_gal, b_eff, pk_lin, bias = self.m, H["n_gal"], H["b_eff"], H["pk_lin"], H["bias"]
        gal = nc[None, :] + ns[None, :] * uk
        P_1h = jnp.trapezoid(dndm[None, :] * gal * Y, m, axis=1) / n_gal
        I_Y = jnp.trapezoid(dndm[None, :] * bias[None, :] * Y, m, axis=1)
        return P_1h + b_eff * pk_lin * I_Y

    def _pk_XX_of(self, theta, H, X, Xa):
        """X-ray auto 1h+2h for one gas FT X (Nk, NM) + flat AGN term Xa (NM,)."""
        dndm, bias = H["dndm"], H["bias"]
        Xtot = X + Xa[None, :]                                          # gas + AGN per halo
        P_1h = jnp.trapezoid(dndm[None, :] * Xtot ** 2, self.m, axis=1)
        I = jnp.trapezoid(dndm[None, :] * bias[None, :] * Xtot, self.m, axis=1)
        return P_1h + H["pk_lin"] * I ** 2

    def _pk_XX(self, theta, H):
        """X-ray auto (gas + AGN); (Nk,) or (Nb, Nk) band autos in band mode."""
        if self.xray_bands is None:
            return self._pk_XX_of(theta, H, self._emissivity_uk(theta, H),
                                  self._agn_point_source(theta))
        Xbs = self._emissivity_uk_bands(theta, H)
        A = self._agn_emissivity_amp(theta)
        wb = self._agn_band_fractions(theta) * self._abs_survival_bands(theta)
        return jnp.stack([self._pk_XX_of(theta, H, Xbs[b], A * wb[b])
                          for b in range(len(Xbs))])

    def _pk_mm(self, theta, H):
        """Halo-model matter power P_mm(k): 1-halo (baryon-split profile) + P_lin.

        The 2-halo term is taken as the linear P(k) (matter is unbiased on large
        scales by mass conservation); the 1-halo term carries the baryonic-feedback
        suppression, because a fraction f_b(M) of the mass sits in the extended
        gas profile, lowering u(k|M) at high k.  This is the cosmic-shear
        systematic the X-ray/tSZ statistics calibrate.
        """
        dndm, uk, m = H["dndm"], H["uk"], self.m
        m_over_rho = m / H["rho_m"]
        u_m = (1.0 - H["fb"][None, :]) * uk + H["fb"][None, :] * self._gas_density_uk(theta, H)
        P_1h = jnp.trapezoid(dndm[None, :] * (m_over_rho[None, :] ** 2) * u_m ** 2, m, axis=1)
        return P_1h + H["pk_lin"]

    def _lensing_kernel(self, theta, nz_src=None):
        r"""Weak-lensing efficiency W_κ(χ) on the shear grid, + χ(z) [Mpc/h].

        .. math::
            W_\kappa(\chi) = \tfrac32 \Omega_m (H_0/c)^2 \frac{\chi}{a}
                \int_\chi^{\chi_H}\! d\chi'\, n_s(\chi')\,\frac{\chi'-\chi}{\chi'}

        ``nz_src`` defaults to the overall source n(z); tomographic callers pass
        one of ``self.nz_src_bins``.
        """
        c = self._cosmo(theta)
        h, Om = c["h"], c["Omega_m"]
        z = self.z_shear
        chi = comoving_distance(z, h, Om, c["w0"], c["wa"]) * h        # (Nz,) Mpc/h
        a = 1.0 / (1.0 + z)
        # source distribution per unit χ (n_s(z) dz = n_s(χ) dχ), renormalised
        dz_dchi = jnp.gradient(z) / jnp.gradient(chi)
        ns_chi = (self.nz_src if nz_src is None else nz_src) * dz_dchi
        ns_chi = ns_chi / jnp.trapezoid(ns_chi, chi)
        # lensing efficiency g(χ_i) = ∫_{χ_i} n_s(χ') (χ'-χ_i)/χ' dχ'
        dchi = jnp.gradient(chi)
        kern = jnp.clip(chi[None, :] - chi[:, None], 0.0, None) / chi[None, :]   # (Ni, Nj)
        g = jnp.sum(ns_chi[None, :] * kern * dchi[None, :], axis=1)     # (Ni,)
        W = 1.5 * Om / _C_OVER_H0 ** 2 * (chi / a) * g                 # (Nz,) [h/Mpc]
        return chi, W, z

    def _pmm_logstack(self, theta, z_arr):
        """log P_mm(k) at each z in ``z_arr`` → (Nz, Nk), for lensing Limber integrals."""
        return jnp.stack([jnp.log(jnp.maximum(self._pk_mm(theta, self._halo_common(theta, float(z))), 1e-30))
                          for z in np.asarray(z_arr)])

    def _limber_kernels(self, chi, W1, W2, logP_stack):
        """C_ℓ = ∫ dχ W1(χ)W2(χ)/χ² P(k=(ℓ+½)/χ, z(χ)) for two lensing-type kernels."""
        klim = jnp.log(jnp.maximum((self.ell[:, None] + 0.5) / chi[None, :], 1e-4))

        def _interp(lkq, lpt):
            return jnp.exp(jnp.interp(lkq, self.log_k, lpt))
        Pz = jax.vmap(jax.vmap(_interp, in_axes=(0, 0)), in_axes=(0, None))(klim, logP_stack)
        return jnp.trapezoid(W1[None, :] * W2[None, :] / chi[None, :] ** 2 * Pz, chi, axis=1)

    def _cosmic_shear(self, theta, stack_shear=None):
        """Convergence power spectra C_ℓ^{κκ} via the lensing Limber integral.

        Single-bin mode returns one spectrum (Nℓ,); tomographic mode returns
        the stacked unique pairs {C_ij, i ≤ j} in ``self.shear_pairs`` order,
        (n_pairs·Nℓ,).
        """
        if stack_shear is None:
            stack_shear = self._pmm_logstack(theta, self.z_shear)
        if self.n_shear_bins == 1:
            chi, W, _ = self._lensing_kernel(theta)
            return self._limber_kernels(chi, W, W, stack_shear)
        chi = None
        Ws = []
        for nz in self.nz_src_bins:
            chi, W, _ = self._lensing_kernel(theta, nz)
            Ws.append(W)
        return jnp.concatenate([self._limber_kernels(chi, Ws[i], Ws[j], stack_shear)
                                for (i, j) in self.shear_pairs])

    def _cmb_kernel(self, theta, chi, z):
        r"""CMB-lensing efficiency W_{κ_CMB}(χ), single source plane at z*≈1089.

        .. math::
            W_{\kappa_{\rm CMB}}(\chi) = \tfrac32 \Omega_m (H_0/c)^2
                \frac{\chi}{a}\,\frac{\chi_*-\chi}{\chi_*}
        """
        c = self._cosmo(theta)
        h, Om = c["h"], c["Omega_m"]
        chi_star = comoving_distance(jnp.array([self.z_star]), h, Om,
                                     c["w0"], c["wa"])[0] * h
        return (1.5 * Om / _C_OVER_H0 ** 2 * chi * (1.0 + z)
                * jnp.clip(chi_star - chi, 0.0, None) / chi_star)

    def _cl_gkCMB(self, theta, Hs_zgrid):
        """Galaxy × CMB-lensing C_ℓ: galaxy window × κ_CMB kernel, via P_gm."""
        c = self._cosmo(theta)
        chi = comoving_distance(self.z_grid, c["h"], c["Omega_m"],
                                c["w0"], c["wa"]) * c["h"]
        dndchi = self.nz / jnp.trapezoid(self.nz, chi)
        Wc = self._cmb_kernel(theta, chi, self.z_grid)
        stack = jnp.stack([jnp.log(jnp.maximum(self._pk_gg_gm(H, theta)[1], 1e-30)) for H in Hs_zgrid])
        return self._limber_kernels(chi, dndchi, Wc, stack)

    def _cl_shear_kCMB(self, theta, stack_shear):
        """Cosmic-shear × CMB-lensing C_ℓ (both lensing kernels, matter power).

        Tomographic mode returns one spectrum per source bin, concatenated.
        """
        if self.n_shear_bins == 1:
            chi, Wk, z = self._lensing_kernel(theta)
            Wc = self._cmb_kernel(theta, chi, z)
            return self._limber_kernels(chi, Wk, Wc, stack_shear)
        outs = []
        for nz in self.nz_src_bins:
            chi, Wk, z = self._lensing_kernel(theta, nz)
            Wc = self._cmb_kernel(theta, chi, z)
            outs.append(self._limber_kernels(chi, Wk, Wc, stack_shear))
        return jnp.concatenate(outs)

    def _cl_kCMB(self, theta):
        """CMB-lensing auto C_ℓ^{κκ_CMB} over the high-z LOS grid."""
        c = self._cosmo(theta)
        chi = comoving_distance(self.z_cmb, c["h"], c["Omega_m"],
                                c["w0"], c["wa"]) * c["h"]
        Wc = self._cmb_kernel(theta, chi, self.z_cmb)
        stack = self._pmm_logstack(theta, self.z_cmb)
        return self._limber_kernels(chi, Wc, Wc, stack)

    # ---- projections / Limber ---------------------------------------
    def _wp(self, P_gg, rp=None):
        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_tab = _pk_to_xi(r_tab, self.log_k, jnp.log(jnp.maximum(P_gg, 1e-20)))
        pi_grid = jnp.linspace(0.0, self.pi_max, 512)

        def _one(rp_i):
            rr = jnp.sqrt(rp_i ** 2 + pi_grid ** 2)
            return 2.0 * jnp.trapezoid(jnp.interp(rr, r_tab, xi_tab), pi_grid)
        return jax.vmap(_one)(self.rp_wp if rp is None else rp)

    def _delta_sigma(self, P_gm, theta):
        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_tab = _pk_to_xi(r_tab, self.log_k, jnp.log(jnp.maximum(P_gm, 1e-20)))
        R_tab = jnp.logspace(-2, 2.0, 256)
        chi_log = jnp.logspace(-2, jnp.log10(300.0), 256)
        chi_lin = jnp.linspace(1.0, 300.0, 256)
        chi = jnp.sort(jnp.concatenate([chi_log, chi_lin]))

        def _wp_one(R_i):
            rr = jnp.sqrt(R_i ** 2 + chi ** 2)
            return 2.0 * jnp.trapezoid(jnp.interp(rr, r_tab, xi_tab), chi)
        wp_gm = jax.vmap(_wp_one)(R_tab)
        integ = R_tab * wp_gm
        dR = jnp.diff(R_tab)
        mid = 0.5 * (integ[:-1] + integ[1:])
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mid * dR)])
        sig_bar = 2.0 * cum / R_tab ** 2
        rho_m = _RHO_CRIT0 * self._cosmo(theta)["Omega_m"]
        ds_tab = (sig_bar - wp_gm) * rho_m * 1e-12
        return jnp.interp(self.rp_ds, R_tab, ds_tab)

    def _limber_from_stack(self, theta, logP_stack):
        """C_ℓ = ∫ dχ/χ² W_g(χ) P(k=(ℓ+½)/χ, z(χ)) from a precomputed (Nz,Nk) stack."""
        c = self._cosmo(theta)
        h, Om = c["h"], c["Omega_m"]
        chi = comoving_distance(self.z_grid, h, Om, c["w0"], c["wa"]) * h   # (Nz,) Mpc/h
        dndchi = self.nz / jnp.trapezoid(self.nz, chi)
        klim = jnp.log(jnp.maximum((self.ell[:, None] + 0.5) / chi[None, :], 1e-4))

        def _interp(lkq, lpt):
            return jnp.exp(jnp.interp(lkq, self.log_k, lpt))
        Pz = jax.vmap(jax.vmap(_interp, in_axes=(0, 0)), in_axes=(0, None))(klim, logP_stack)
        integrand = dndchi[None, :] * Pz / chi[None, :] ** 2
        return jnp.trapezoid(integrand, chi, axis=1)                   # (Nell,)

    def _xlf(self, theta):
        r"""Powell 2022 AGN X-ray luminosity function Φ(log L_X) [(Mpc/h)⁻³ dex⁻¹].

        Forward-models the SMBH luminosity per halo — the ZM15 SHMR gives
        ⟨log M_*⟩(M_h); a free M_BH–M_* relation gives ⟨log M_BH⟩; a universal
        Eddington-ratio distribution (Ananna 2022 broken power law) sets the
        accretion; ``L_bol = 1.26e38·M_BH·λ`` — integrated over the HMF
        ``dn/dM_h``.  The XLF **amplitude/shape therefore carry the cosmology
        through dn/dM** (as the cluster mass function does), which is the new
        handle it adds to (Ω_m, σ_8); its shape simultaneously pins the AGN
        sector, breaking the AGN↔cosmology degeneracy of the cross-spectra.

        A shift-invariant kernel makes it cheap and exactly differentiable:
        P(log L_X | M_h) is the ERDF convolved with a Gaussian of (halo-independent)
        width σ_lm = √(α_BH²σ_M*² + σ_BH²), evaluated at
        ``log L_X − log k − ⟨log M_BH⟩(M_h)``.  Matches the validated numpy
        :class:`hod_mod.agn.powell.PowellAGNModel` (see that module + its tests).
        """
        if self.xlf_band == "hard":
            return self._xlf_at(theta, self.loglx_xlf)
        # tier-2 soft-band observed XLF: intrinsic hard abscissa shifted by
        # k_h2s(Γ), with the obscured fraction f_abs dimmed by the NH
        # transmission (a two-component mixture — analytic and cheap).
        shift = jnp.log10(self._k_h2s(theta))            # l_soft = l_hard + shift
        labs = float(np.log10(self._t_soft))             # < 0
        fabs = theta[_IDX["agn_fabs"]]
        l_hard = self.loglx_xlf - shift
        return ((1.0 - fabs) * self._xlf_at(theta, l_hard)
                + fabs * self._xlf_at(theta, l_hard - labs))

    def _xlf_at(self, theta, loglx_hard):
        """Φ(log L_X^hard) at the given hard-band abscissas (the tier-1 kernel)."""
        c = self._cosmo(theta)
        z = self.z_eff
        dndm = self._hmf.dndm(self.m, float(z), c)                      # (NM,)
        dndlogm = dndm * self.m * _LN10                                 # (NM,) per dex(M_h)

        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        # K(logL_X | M_h) = Σ_λ ERDF(λ) dλ · N(logL_X − logk − ⟨logM_BH⟩ − λ ; σ_lm)
        t = loglx_hard[:, None, None] - _POWELL_LOGK - mean_bh[None, :, None]
        gk = jnp.exp(-0.5 * ((t - self.loglam[None, None, :]) / sig_lm) ** 2) \
            / (jnp.sqrt(2.0 * jnp.pi) * sig_lm)                        # (Nlx, NM, Nlam)
        K = jnp.sum(erdf[None, None, :] * gk, axis=2) * self.dlam      # (Nlx, NM)  pdf in dex(L_X)
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]                 # active fraction
        return jnp.trapezoid(dndlogm[None, :] * K * ferdf, self.log10m, axis=1)  # (Nlx,)

    def _rlf(self, theta):
        r"""Radio luminosity function Φ(log L_R) [(Mpc/h)⁻³ dex⁻¹] via the
        fundamental plane of black-hole activity (missing-physics extension).

        Per (M_h, λ) cell the FP maps the Powell chain's X-ray luminosity and
        black-hole mass to 5 GHz νL_ν:

        .. math::

            \log L_R = \xi_{RX}\log L_X + \xi_{RM}\log M_{\rm BH} + b_R
            = \xi_{RX}(\log k + \lambda) + (\xi_{RX}+\xi_{RM})\log M_{\rm BH} + b_R,

        so the M_BH scatter σ_lm enters with coefficient (ξ_RX + ξ_RM) — it is
        the SAME deviate in L_X and M_BH — and the FP scatter σ_R adds in
        quadrature.  Exact identity (tested): at (ξ_RX, ξ_RM, b_R, σ_R) =
        (1, 0, 0, 0) the rlf equals the hard-band xlf on the same abscissas.
        """
        c = self._cosmo(theta)
        dndm = self._hmf.dndm(self.m, float(self.z_eff), c)
        dndlogm = dndm * self.m * _LN10

        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        xi_rx = theta[_IDX["agn_xi_rx"]]
        xi_rm = theta[_IDX["agn_xi_rm"]]
        sig_r = jnp.sqrt((xi_rx + xi_rm) ** 2 * sig_lm ** 2
                         + theta[_IDX["agn_sig_r"]] ** 2)
        mu0 = (xi_rx * _POWELL_LOGK + (xi_rx + xi_rm) * mean_bh
               + theta[_IDX["agn_b_r"]])                                # (NM,)
        t = self.loglr_rlf[:, None, None] - mu0[None, :, None] \
            - xi_rx * self.loglam[None, None, :]                        # (Nlr, NM, Nlam)
        gk = jnp.exp(-0.5 * (t / sig_r) ** 2) / (jnp.sqrt(2.0 * jnp.pi) * sig_r)
        K = jnp.sum(erdf[None, None, :] * gk, axis=2) * self.dlam       # (Nlr, NM)
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]
        rlf_fp = jnp.trapezoid(dndlogm[None, :] * K * ferdf, self.log10m, axis=1)
        # radio-loud jet population (wave 3, HERG/LERG): NOT tied to the
        # radiatively-efficient ERDF-active fraction — a loud fraction
        # f_loud(M_BH) of ALL central black holes with a jet luminosity
        # lognormal around b_jet + (log M_BH − 8).  f_loud0 = 0 removes it.
        f_loud = jnp.minimum(theta[_IDX["f_loud0"]]
                             * 10.0 ** (theta[_IDX["beta_loud"]]
                                        * (mean_bh - 8.0)), 1.0)        # (NM,)
        sig_jet = jnp.sqrt(sig_lm ** 2 + _SIG_JET ** 2)
        t_j = self.loglr_rlf[:, None] - theta[_IDX["b_jet"]] \
            - (mean_bh[None, :] - 8.0)
        gk_j = jnp.exp(-0.5 * (t_j / sig_jet) ** 2) \
            / (jnp.sqrt(2.0 * jnp.pi) * sig_jet)                        # (Nlr, NM)
        rlf_jet = jnp.trapezoid(dndlogm[None, :] * f_loud[None, :] * gk_j,
                                self.log10m, axis=1)
        return rlf_fp + rlf_jet

    # ---- cold gas / neutral hydrogen (missing-physics extension) ------
    def _mhi(self, theta):
        r"""M_HI(M_h) [Msun/h] halo model (Villaescusa-Navarro et al. 2018):

        .. math::

            M_{\rm HI}(M_h) = M_0\,(M_h/M_{\rm min})^{\alpha}\,
            \exp\!\left[-(M_{\rm min}/M_h)^{0.35}\right]
        """
        m0 = 10.0 ** theta[_IDX["log10_M0_hi"]]
        mmin = 10.0 ** theta[_IDX["log10_Mmin_hi"]]
        return (m0 * (self.m / mmin) ** theta[_IDX["alpha_hi"]]
                * jnp.exp(-((mmin / self.m) ** 0.35)))                  # (NM,)

    def _himf(self, theta, H):
        r"""HI mass function Φ(log M_HI) [(Mpc/h)⁻³ dex⁻¹]: the M_HI(M_h)
        relation with a fixed 0.35 dex lognormal scatter (the conditional-
        scatter pattern of the SMF/XLF kernels)."""
        dndlogm = H["dndm"] * self.m * _LN10
        t = self.logmhi_himf[:, None] - jnp.log10(self._mhi(theta))[None, :]
        gk = jnp.exp(-0.5 * (t / _SIG_MHI) ** 2) / (jnp.sqrt(2.0 * jnp.pi) * _SIG_MHI)
        return jnp.trapezoid(dndlogm[None, :] * gk, self.log10m, axis=1)

    def _pk_gHI(self, theta, H):
        """galaxy × HI power spectrum: the C_ℓ^{gX} machinery with the HI mass
        (NFW-distributed, 10^10-referenced model units) as the tracer field.

        In quenched mode the cross probes the HI around quenched centrals — the
        ``dhi_quenched`` deficit rescales M_HI (0 dex fiducial; the
        NeutralUniverseMachine phenomenology)."""
        return self._pk_gX_of(theta, H, self._hi_field(theta, H),
                              jnp.zeros_like(self.m))

    # ---- star-forming main sequence (missing-physics wave 2) -----------
    def _ssfr(self, theta):
        r"""Mean main-sequence log10 sSFR [yr⁻¹] of THIS cell's M* sample, (1,).

        Speagle+2014-style linear main sequence with the tier-2 evolution
        mechanism (``ssfr_ms_zs`` acts on the normalisation via _theta_eff):

        .. math:: \langle\log_{10}{\rm sSFR}\rangle = {\rm norm}
                  + {\rm slope}\,(\log_{10}M_*^{\rm cell} - 10.5)

        The datum per (z, M*) cell directly constrains the MS normalisation,
        slope and evolution (COSMOS/Euclid main-sequence measurements).
        """
        return jnp.array([self._mu_ms(theta)])

    def _sfrd(self, theta, H):
        r"""SFR density of THIS cell's sample [M_⊙/yr (Mpc/h)⁻³], (1,).

        Occupation-weighted over the SF and quenched members with their
        lognormal mean SFRs, SFR = sSFR · M_*; the cosmic ρ_SFR(z)
        (Madau–Dickinson) is the sum over the cell column at each z.
        """
        mstar_c = (0.5 * (self._thr + self._thr_hi) if self._thr_hi is not None
                   else self._thr + 0.25)
        mu = self._mu_ms(theta)
        sfr_ms = 10.0 ** (mu + mstar_c) \
            * jnp.exp(0.5 * (theta[_IDX["sigma_ms"]] * _LN10) ** 2)
        sfr_q = 10.0 ** (mu + theta[_IDX["dssfr_q"]] + mstar_c) \
            * jnp.exp(0.5 * (_SIG_SSFR_Q * _LN10) ** 2)
        nc, ns = self._occ_base(theta)
        fq_c, fq_s, s_ms, s_q = self._sfq_weights(theta)
        n_sf = jnp.trapezoid(H["dndm"] * (nc * (1.0 - fq_c) + ns * (1.0 - fq_s))
                             * s_ms, self.m)
        n_q = jnp.trapezoid(H["dndm"] * (nc * fq_c + ns * fq_s) * s_q, self.m)
        if self.sfq == "sf":
            return jnp.array([n_sf * sfr_ms])
        if self.sfq == "q":
            return jnp.array([n_q * sfr_q])
        return jnp.array([n_sf * sfr_ms + n_q * sfr_q])

    def _oiilf(self, theta):
        r"""[OII] luminosity function Φ(log L_[OII]) [(Mpc/h)⁻³ dex⁻¹].

        Kennicutt-like calibration on the star-forming main sequence:
        log L_[OII] = loii_norm + log SFR with
        log SFR(M_h) = μ_MS(M_*(M_h)) + log M_*(M_h) through the ZM15 SHMR;
        the kernel width combines the MS scatter, the [OII]-calibration
        scatter and the (1+slope)-propagated M_*|M_h scatter.  Centrals-only
        v1 (the AGN-chain convention).  Data: the z-resolved [OII] LFs.
        """
        c = self._cosmo(theta)
        dndm = self._hmf.dndm(self.m, float(self.z_eff), c)
        dndlogm = dndm * self.m * _LN10
        hp = self._hod(theta)
        log10ms = _inv_shmr(self.log10m, hp["lg_m1h"], hp["lg_m0star"],
                            hp["beta"], hp["delta"], hp["gamma"])
        mu_ssfr = (theta[_IDX["ssfr_ms_norm"]]
                   + theta[_IDX["ssfr_ms_slope"]] * (log10ms - 10.5))
        mu_l = theta[_IDX["loii_norm"]] + mu_ssfr + log10ms               # (NM,)
        sig_star = sigma_lnmstar_zu15(self.log10m, hp["lg_m1h"],
                                      hp["sigma_lnmstar"], hp["eta"]) / _LN10
        sig_l = jnp.sqrt(theta[_IDX["sigma_ms"]] ** 2 + _SIG_OII ** 2
                         + (1.0 + theta[_IDX["ssfr_ms_slope"]]) ** 2
                         * sig_star ** 2)                                 # (NM,)
        fq_c = f_red_cen_zu16(self.log10m, theta[_IDX["log10_Mq_cen"]],
                              theta[_IDX["mu_q_cen"]])
        t = self.logloii_oiilf[:, None] - mu_l[None, :]
        gk = jnp.exp(-0.5 * (t / sig_l[None, :]) ** 2) \
            / (jnp.sqrt(2.0 * jnp.pi) * sig_l[None, :])                   # (NL, NM)
        return jnp.trapezoid(dndlogm[None, :] * (1.0 - fq_c)[None, :] * gk,
                             self.log10m, axis=1)

    def _ilf(self, theta):
        r"""AGN infrared (6 μm) luminosity function [(Mpc/h)⁻³ dex⁻¹].

        L_IR = 10^{agn_bc_ir} · L_bol on the Powell chain — obscuration-robust
        by construction (no f_abs suppression), so the IR LF cross-checks the
        obscured fraction the soft-X-ray XLF is dimmed by.
        """
        c = self._cosmo(theta)
        dndm = self._hmf.dndm(self.m, float(self.z_eff), c)
        dndlogm = dndm * self.m * _LN10
        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        t = self.loglir_ilf[:, None, None] - _POWELL_LOGBOL \
            - theta[_IDX["agn_bc_ir"]] - mean_bh[None, :, None]
        gk = jnp.exp(-0.5 * ((t - self.loglam[None, None, :]) / sig_lm) ** 2) \
            / (jnp.sqrt(2.0 * jnp.pi) * sig_lm)
        K = jnp.sum(erdf[None, None, :] * gk, axis=2) * self.dlam
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]
        return jnp.trapezoid(dndlogm[None, :] * K * ferdf, self.log10m, axis=1)

    # ---- tier 3: SFR moments + radio/IR per-halo fields ----------------
    def _sfr_moments(self, theta):
        r"""(μ_SFR, σ_SFR, f_Q^cen, ⟨log M*⟩, σ_M*) of the CENTRAL per halo.

        μ_SFR = μ_MS(M*) + log M* through the ZM15 SHMR; the width propagates
        the MS scatter and the (1+slope)-scaled M*|M_h scatter (the _oiilf
        algebra, shared by every SFR-calibrated band).
        """
        hp = self._hod(theta)
        log10ms = _inv_shmr(self.log10m, hp["lg_m1h"], hp["lg_m0star"],
                            hp["beta"], hp["delta"], hp["gamma"])        # (NM,)
        mu_ssfr = (theta[_IDX["ssfr_ms_norm"]]
                   + theta[_IDX["ssfr_ms_slope"]] * (log10ms - 10.5))
        sig_star = sigma_lnmstar_zu15(self.log10m, hp["lg_m1h"],
                                      hp["sigma_lnmstar"], hp["eta"]) / _LN10
        sig_lsfr = jnp.sqrt(theta[_IDX["sigma_ms"]] ** 2
                            + (1.0 + theta[_IDX["ssfr_ms_slope"]]) ** 2
                            * sig_star ** 2)                             # (NM,)
        fq_c = f_red_cen_zu16(self.log10m, theta[_IDX["log10_Mq_cen"]],
                              theta[_IDX["mu_q_cen"]])
        return mu_ssfr + log10ms, sig_lsfr, fq_c, log10ms, sig_star

    def _mean_sfr(self, theta):
        """Star-forming-central mean SFR per halo [Msun/yr], (NM,): the full
        lognormal mean weighted by the SF fraction (quenched centrals carry
        negligible SFR-driven emission)."""
        mu_lsfr, sig_lsfr, fq_c, _, _ = self._sfr_moments(theta)
        return ((1.0 - fq_c) * 10.0 ** mu_lsfr
                * jnp.exp(0.5 * (sig_lsfr * _LN10) ** 2))

    def _radio_fields(self, theta):
        r"""Central νL_ν per halo in each radio map band [10³⁸ erg/s], list of (NM,).

        Three source populations: SF synchrotron on the radio–FIR calibration
        (``l14_sfr``, spectral index ``alpha_syn``, anchored at 1.4 GHz),
        fundamental-plane cores (the ``_rlf`` FP moments, flat spectrum), and
        radio-loud jets (5 GHz anchored, ``alpha_syn``-scaled) — the same
        populations the rlf observable counts, here as a mean intensity field.
        """
        sfr = self._mean_sfr(theta)
        l_sf14 = 10.0 ** (theta[_IDX["l14_sfr"]] + float(np.log10(1.4e9))
                          - 38.0) * sfr                                  # (NM,)
        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        xi_rx = theta[_IDX["agn_xi_rx"]]
        xi_rm = theta[_IDX["agn_xi_rm"]]
        sig_r = jnp.sqrt((xi_rx + xi_rm) ** 2 * sig_lm ** 2
                         + theta[_IDX["agn_sig_r"]] ** 2)
        mu0 = (xi_rx * _POWELL_LOGK + (xi_rx + xi_rm) * mean_bh
               + theta[_IDX["agn_b_r"]])
        lam_mean = jnp.sum(erdf * 10.0 ** (xi_rx * self.loglam)) * self.dlam
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]
        l_core = (ferdf * 10.0 ** (mu0 - 38.0) * lam_mean
                  * jnp.exp(0.5 * (sig_r * _LN10) ** 2))                 # (NM,)
        f_loud = jnp.minimum(theta[_IDX["f_loud0"]]
                             * 10.0 ** (theta[_IDX["beta_loud"]]
                                        * (mean_bh - 8.0)), 1.0)
        sig_jet = jnp.sqrt(sig_lm ** 2 + _SIG_JET ** 2)
        l_jet = (f_loud * 10.0 ** (theta[_IDX["b_jet"]] + mean_bh - 8.0 - 38.0)
                 * jnp.exp(0.5 * (sig_jet * _LN10) ** 2))                # (NM,)
        a = 1.0 - theta[_IDX["alpha_syn"]]
        return [l_sf14 * (nu / 1.4) ** a + l_core + l_jet * (nu / 5.0) ** a
                for nu in self.radio_map_bands]

    def _ir_fields(self, theta):
        r"""Central νL_ν per halo in each IR map band [10⁴⁴ erg/s], list of (NM,).

        Dust re-emission (∝ SFR via ``lir_sfr``, color-tilted by ``bir_color``
        around the 4.9 μm anchor), stellar continuum (∝ M* via ``ml_nir``), and
        the AGN torus (the ``_ilf`` chain, NO f_abs — IR is obscuration-robust).
        """
        sfr = self._mean_sfr(theta)
        _, _, _, log10ms, sig_star = self._sfr_moments(theta)
        l_star34 = (10.0 ** (theta[_IDX["ml_nir"]] - 44.0 + log10ms)
                    * jnp.exp(0.5 * (sig_star * _LN10) ** 2))            # (NM,)
        l_dust = 10.0 ** (theta[_IDX["lir_sfr"]] - 44.0) * sfr           # (NM,)
        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        lam_mean = jnp.sum(erdf * 10.0 ** self.loglam) * self.dlam
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]
        l_tor = (ferdf * 10.0 ** (_POWELL_LOGBOL + theta[_IDX["agn_bc_ir"]]
                                  + mean_bh - 44.0) * lam_mean
                 * jnp.exp(0.5 * (sig_lm * _LN10) ** 2))                 # (NM,)
        bir = theta[_IDX["bir_color"]]
        return [(float(self._ir_cdust[i]) * 10.0 ** (bir * float(self._ir_dlognu[i]))
                 * l_dust
                 + float(self._ir_cstar[i]) * l_star34
                 + float(self._ir_ctorus[i]) * l_tor)
                for i in range(len(self.ir_map_bands))]

    def _hi_field(self, theta, H):
        """HI mass field M_HI(M_h)·u_NFW in 10^10-referenced model units,
        (Nk, NM) — shared by the g×HI cross and the 21 cm auto."""
        mhi = self._mhi(theta)
        if self.sfq == "q":
            mhi = mhi * 10.0 ** theta[_IDX["dhi_quenched"]]
        return (mhi / 1.0e10)[None, :] * H["uk"]

    # ---- tier 3: AGN tracer crosses + cluster counts -------------------
    def _agn_tracer(self, theta, H, l1, l2):
        """(N_AGN(M), n_agn, b_agn) for one observed-soft L_X bin — the
        wp_agn ingredients, shared by every AGN cross-statistic."""
        n_a = self._agn_occupation_obs(theta, l1, l2)                    # (NM,)
        n_agn = jnp.trapezoid(H["dndm"] * n_a, self.m)
        b_agn = jnp.trapezoid(H["dndm"] * H["bias"] * n_a, self.m) / n_agn
        return n_a, n_agn, b_agn

    def _pk_ag(self, theta, H, n_a, n_agn, b_agn):
        """AGN × galaxy number cross: central-AGN × satellite-galaxy 1-halo
        + b_a b_g 2-halo.  Central–central pairs are the AGN host itself
        (the AGN is the central's nucleus) and are excluded as self-pairs."""
        P_1h = jnp.trapezoid(H["dndm"][None, :] * (n_a * H["ns"])[None, :]
                             * H["uk"], self.m, axis=1) / (n_agn * H["n_gal"])
        return P_1h + b_agn * H["b_eff"] * H["pk_lin"]

    def _pk_am(self, theta, H, n_a, n_agn, b_agn):
        """AGN × matter (the ΔΣ_AGN kernel): the central-AGN tracer against
        the baryon-split matter field of ``_pk_gg_gm``."""
        m_over_rho = self.m / H["rho_m"]
        uk_gas = self._gas_density_uk(theta, H)
        u_matter = (1.0 - H["fb"][None, :]) * H["uk"] + H["fb"][None, :] * uk_gas
        integ = (H["dndm"][None, :] * n_a[None, :] * m_over_rho[None, :]
                 * u_matter)
        P_1h = jnp.trapezoid(integ, self.m, axis=1) / n_agn
        return P_1h + b_agn * H["pk_lin"]

    def _ncl(self, theta, H):
        r"""X-ray cluster count density [h³ Mpc⁻³] above the survey L_X limit:

        .. math:: n_{\rm cl} = \int dM\,\frac{dn}{dM}\,
            \tfrac12\,{\rm erfc}\frac{\log L_{\rm lim} - \log L_X(M)}
                                     {\sqrt2\,\sigma_{L_X}}

        with the free L_X–M relation of the gas sector (``_lx_kt_of``) and a
        fixed 0.25 dex selection scatter — the classic cluster-count probe,
        cosmology through dn/dM and astrophysics through the shared scaling.
        """
        lx, _ = self._lx_kt_of(theta, H["z"])
        loglx = jnp.log10(lx) + 45.0                                     # erg/s
        sel = 0.5 * erfc((self.logl_ncl - loglx) / (jnp.sqrt(2.0) * _SIG_LXCL))
        return jnp.array([jnp.trapezoid(H["dndm"] * sel, self.m)])

    # ---- tier 3: galaxy band LFs + AGN UV/opt LFs -----------------------
    def _lf_lognormal(self, theta, grid, mu_l, sig_l, weight):
        """Lognormal LF kernel Φ(log L) = ∫ dlog M dn/dlog M · w(M) ·
        N(log L − μ(M); σ(M)) — the shared ``_oiilf`` body; μ, σ, w are (NM,)."""
        c = self._cosmo(theta)
        dndm = self._hmf.dndm(self.m, float(self.z_eff), c)
        dndlogm = dndm * self.m * _LN10
        t = grid[:, None] - mu_l[None, :]
        gk = jnp.exp(-0.5 * (t / sig_l[None, :]) ** 2) \
            / (jnp.sqrt(2.0 * jnp.pi) * sig_l[None, :])                  # (NL, NM)
        return jnp.trapezoid(dndlogm[None, :] * weight[None, :] * gk,
                             self.log10m, axis=1)

    def _uvlf(self, theta):
        """UV (1500 Å) LF of star-forming centrals: μ = luv_norm + log SFR with
        an M*-dependent attenuation tilt (``tau_uv_mslope``)."""
        mu_lsfr, sig_lsfr, fq_c, log10ms, _ = self._sfr_moments(theta)
        mu = (theta[_IDX["luv_norm"]] + mu_lsfr
              - theta[_IDX["tau_uv_mslope"]] * (log10ms - 10.5))
        sig = jnp.sqrt(sig_lsfr ** 2 + _SIG_UV ** 2)
        return self._lf_lognormal(theta, self.logluv_uvlf, mu, sig, 1.0 - fq_c)

    def _half(self, theta):
        """Hα LF (Kennicutt calibration ``lha_norm``) — the ``_oiilf`` clone
        at a different line normalisation (identical at lha_norm = loii_norm)."""
        mu_lsfr, sig_lsfr, fq_c, _, _ = self._sfr_moments(theta)
        mu = theta[_IDX["lha_norm"]] + mu_lsfr
        sig = jnp.sqrt(sig_lsfr ** 2 + _SIG_HA ** 2)
        return self._lf_lognormal(theta, self.loglha_half, mu, sig, 1.0 - fq_c)

    def _nirlf(self, theta):
        """NIR (3.4 μm) LF of ALL centrals: νL_ν = Υ_NIR · M* — the stellar-
        continuum leg, direct SHMR probe free of the SFR chain."""
        _, _, _, log10ms, sig_star = self._sfr_moments(theta)
        mu = theta[_IDX["ml_nir"]] + log10ms
        sig = jnp.sqrt(sig_star ** 2 + _SIG_NIR ** 2)
        return self._lf_lognormal(theta, self.loglnir_nirlf, mu, sig,
                                  jnp.ones_like(log10ms))

    def _optlf(self, theta):
        """Optical (r-band) LF: SF/Q mixture — quiescent centrals are offset
        by ``dopt_q`` dex (at dopt_q = 0 the mixture collapses exactly to a
        single all-central lognormal, the additivity invariant)."""
        _, _, fq_c, log10ms, sig_star = self._sfr_moments(theta)
        sig = jnp.sqrt(sig_star ** 2 + _SIG_OPT ** 2)
        mu_sf = theta[_IDX["ml_opt"]] + log10ms
        mu_q = mu_sf + theta[_IDX["dopt_q"]]
        return (self._lf_lognormal(theta, self.loglopt_optlf, mu_sf, sig,
                                   1.0 - fq_c)
                + self._lf_lognormal(theta, self.loglopt_optlf, mu_q, sig, fq_c))

    def _qlf(self, theta, grid, bc):
        """Type-1 AGN band LF: the ``_ilf`` kernel at bolometric correction
        ``bc``, times (1 − f_abs) — UV/optical sees only unobscured AGN
        (completing the cross-band obscuration system: UV/opt ∝ 1−f_abs,
        IR ∝ 1, soft X-ray = the NH transmission mixture)."""
        c = self._cosmo(theta)
        dndm = self._hmf.dndm(self.m, float(self.z_eff), c)
        dndlogm = dndm * self.m * _LN10
        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        t = grid[:, None, None] - _POWELL_LOGBOL - bc - mean_bh[None, :, None]
        gk = jnp.exp(-0.5 * ((t - self.loglam[None, None, :]) / sig_lm) ** 2) \
            / (jnp.sqrt(2.0 * jnp.pi) * sig_lm)
        K = jnp.sum(erdf[None, None, :] * gk, axis=2) * self.dlam
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]
        phi = jnp.trapezoid(dndlogm[None, :] * K * ferdf, self.log10m, axis=1)
        return (1.0 - theta[_IDX["agn_fabs"]]) * phi

    # ---- tier 3: map-spectra dispatch ----------------------------------
    def _predict_maps(self, theta, Hs, names):
        """Radio/IR map crosses+autos, AGN crosses, tSZ + 21 cm autos.

        Stacked orderings match ``grid_of``: band-major for cl_gR/cl_gI and
        cl_RR/cl_II; L_X-bin-major (band inner) for cl_aR/cl_aI; L_X-bin-major
        for cl_ag.  Point-source fields are theta-only, computed once.
        """
        out = {}
        z2d = jnp.zeros((1, self.m.size))
        R = (self._radio_fields(theta)
             if any(n in names for n in ("cl_gR", "cl_RR", "cl_aR")) else None)
        I = (self._ir_fields(theta)
             if any(n in names for n in ("cl_gI", "cl_II", "cl_aI")) else None)

        def limber(fn):
            stack = jnp.stack([jnp.log(jnp.maximum(fn(H), 1e-30)) for H in Hs])
            return self._limber_from_stack(theta, stack)

        if "cl_gR" in names:
            out["cl_gR"] = jnp.concatenate(
                [limber(lambda H, Xa=Xa: self._pk_gX_of(theta, H, z2d, Xa))
                 for Xa in R])
        if "cl_gI" in names:
            out["cl_gI"] = jnp.concatenate(
                [limber(lambda H, Xa=Xa: self._pk_gX_of(theta, H, z2d, Xa))
                 for Xa in I])
        if "cl_RR" in names:
            out["cl_RR"] = jnp.concatenate(
                [limber(lambda H, Xa=Xa: self._pk_XX_of(theta, H, z2d, Xa))
                 for Xa in R])
        if "cl_II" in names:
            out["cl_II"] = jnp.concatenate(
                [limber(lambda H, Xa=Xa: self._pk_XX_of(theta, H, z2d, Xa))
                 for Xa in I])
        if any(n in names for n in ("cl_aR", "cl_aI", "cl_ag")):
            # AGN tracer ingredients per (L_X bin, z) — reused by all crosses
            tr = [[self._agn_tracer(theta, H, l1, l2) for H in Hs]
                  for (l1, l2) in self.agn_lx_bins]

            def limber_a(i, fn):
                stack = jnp.stack([jnp.log(jnp.maximum(
                    fn(Hs[j], *tr[i][j]), 1e-30)) for j in range(len(Hs))])
                return self._limber_from_stack(theta, stack)

            if "cl_aR" in names:
                out["cl_aR"] = jnp.concatenate(
                    [limber_a(i, lambda H, n_a, n, b, Xa=Xa:
                              self._pk_tracer_field(theta, H, n_a,
                                                    jnp.zeros_like(n_a),
                                                    z2d, Xa, n, b))
                     for i in range(len(self.agn_lx_bins)) for Xa in R])
            if "cl_aI" in names:
                out["cl_aI"] = jnp.concatenate(
                    [limber_a(i, lambda H, n_a, n, b, Xa=Xa:
                              self._pk_tracer_field(theta, H, n_a,
                                                    jnp.zeros_like(n_a),
                                                    z2d, Xa, n, b))
                     for i in range(len(self.agn_lx_bins)) for Xa in I])
            if "cl_ag" in names:
                out["cl_ag"] = jnp.concatenate(
                    [limber_a(i, lambda H, n_a, n, b:
                              self._pk_ag(theta, H, n_a, n, b))
                     for i in range(len(self.agn_lx_bins))])
        if "cl_yy" in names:
            out["cl_yy"] = limber(
                lambda H: self._pk_XX_of(theta, H, self._pressure_uk(theta, H),
                                         jnp.zeros_like(self.m)))
        if "cl_HIHI" in names:
            out["cl_HIHI"] = limber(
                lambda H: self._pk_XX_of(theta, H, self._hi_field(theta, H),
                                         jnp.zeros_like(self.m)))
        return out

    # ---- Powell AGN chain: shared kernel, occupation, emission, clustering --
    def _agn_kernel_parts(self, theta):
        r"""Shared pieces of the Powell chain: (⟨log M_BH⟩(M_h), σ_lm, ERDF).

        ⟨log M_BH⟩ comes from the ZM15 SHMR ⟨log M_*⟩(M_h) through the free
        M_BH–M_* relation; σ_lm is the halo-independent lognormal width
        √(α_BH²σ_M*²(1−ρ) + σ_BH²) (Powell Model 2: ρ>0 aligns part of the M*
        scatter with M_halo, shrinking the M_BH|M_halo width; unclipped — the
        Fisher is local around the fiducial ρ=0); ERDF is the Ananna 2022
        broken power law normalised to ∫ dlog10λ = 1 on the loglam grid.
        """
        hp = self._hod(theta)
        log10ms = _inv_shmr(self.log10m, hp["lg_m1h"], hp["lg_m0star"],
                            hp["beta"], hp["delta"], hp["gamma"])       # (NM,)
        al_bh = theta[_IDX["agn_al_bh"]]
        mean_bh = theta[_IDX["agn_mu_bh"]] + al_bh * (log10ms - 11.0)   # (NM,)
        # wave 4: BH-bulge coupling — M_BH follows (B/T · M*)-like scaling
        # (Yang+2019) with B/T proxied by the mean early-type fraction of the
        # halo; the 1e-4 floor keeps the log finite at the low-mass end.  At
        # the mbh_bt_slope = 0 fiducial the Powell chain is EXACTLY unchanged,
        # while ∂/∂mbh_bt_slope routes the morphology parameters into the
        # XLF/rlf/ilf — morphology becomes testable through the AGN sector.
        fe_c = f_early_cen(self.log10m, theta[_IDX["log10_M_morph"]],
                           theta[_IDX["beta_morph"]])
        mean_bh = mean_bh + theta[_IDX["mbh_bt_slope"]] \
            * jnp.log10(fe_c + 1.0e-4)
        sig_lm = jnp.sqrt(al_bh ** 2 * theta[_IDX["agn_sig_mstar"]] ** 2
                          * (1.0 - theta[_IDX["agn_rho"]])
                          + theta[_IDX["agn_sig_bh"]] ** 2)
        x = 10.0 ** (self.loglam - theta[_IDX["agn_log10_lstar"]])
        erdf = 1.0 / (x ** theta[_IDX["agn_delta1"]] + x ** theta[_IDX["agn_delta2"]])
        erdf = erdf / (jnp.sum(erdf) * self.dlam)                       # (Nlam,)
        return mean_bh, sig_lm, erdf

    def _agn_occupation(self, theta, l1, l2):
        r"""⟨N_AGN(l1 < log10 L_X^hard < l2 | M_h)⟩ on the mass grid, (NM,).

        Analytic Gaussian-CDF integral of P(log L_X|M_h) = ERDF ⊛ N(σ_lm) over
        the HARD-band bin [l1, l2) — exact and cheap (no L_X quadrature):

        .. math:: N = f_{\rm ERDF} \sum_\lambda {\rm ERDF}(\lambda)\Delta\lambda\;
            \tfrac12\left[{\rm erf}\tfrac{l_2-\mu}{\sqrt2\sigma}
                        - {\rm erf}\tfrac{l_1-\mu}{\sqrt2\sigma}\right],
            \quad \mu = \log_{10}k + \langle\log M_{\rm BH}\rangle + \lambda.

        Centrals only (every galaxy hosts one SMBH; satellites are a documented
        refinement, matching powell.py), clipped to ≤ 1 like the numpy model.
        """
        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        mu = _POWELL_LOGK + mean_bh[:, None] + self.loglam[None, :]     # (NM, Nlam)
        s = jnp.sqrt(2.0) * sig_lm
        cdf = 0.5 * (erf((l2 - mu) / s) - erf((l1 - mu) / s))           # (NM, Nlam)
        n = (10.0 ** theta[_IDX["agn_log10_ferdf"]]
             * jnp.sum(erdf[None, :] * cdf, axis=1) * self.dlam)        # (NM,)
        return jnp.minimum(n, 1.0)

    def _agn_emissivity_amp(self, theta):
        r"""Powell-mode mean AGN emission per halo, (NM,), in the 10^45-referenced
        model units of the gas leg (SOFT band, full lognormal mean — the X-ray
        map sees all AGN emission, no L_min cut):

        .. math:: \langle L_X\rangle(M) = f_{\rm ERDF}\,k_{\rm h2s}\,
            10^{\log k + \langle\log M_{\rm BH}\rangle}\,
            e^{(\sigma_{lm}\ln 10)^2/2}\;\textstyle\sum_\lambda
            {\rm ERDF}(\lambda)\,10^\lambda\,\Delta\lambda .
        """
        mean_bh, sig_lm, erdf = self._agn_kernel_parts(theta)
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]
        lam_mean = jnp.sum(erdf * 10.0 ** self.loglam) * self.dlam
        lognorm = jnp.exp(0.5 * (sig_lm * _LN10) ** 2)
        lx_hard = 10.0 ** (_POWELL_LOGK + mean_bh - 45.0) * lam_mean * lognorm
        return ferdf * self._k_h2s(theta) * lx_hard                     # (NM,)

    def _agn_point_source(self, theta):
        """AGN point-source emission per halo for C_gX/C_XX, (NM,), model units.

        "powell": the Powell-chain mean emission (shares the XLF parameters),
        including the obscured-fraction flux survival in the soft band.
        "surrogate": the tier-1 duty-cycle-scaled L∝M term (log10DC-driven).
        """
        if self.agn_emission == "powell":
            return self._agn_emissivity_amp(theta) * self._abs_survival_soft(theta)
        dc = 10.0 ** theta[_IDX["log10DC"]]
        return dc * (self.m / 1e13) * 1.0e-1

    def _wp_agn(self, theta, H):
        r"""Projected AGN clustering w_p(r_p) per L_X bin, concatenated, 2-halo.

        A Bernoulli central occupation has no self-pairs, so the 1-halo term
        vanishes exactly: P_agn(k) = b_agn² P_lin(k) with

        .. math:: b_{\rm agn} = \frac{\int dM\,\frac{dn}{dM}\,b(M)\,N_{\rm AGN}(M)}
                                    {\int dM\,\frac{dn}{dM}\,N_{\rm AGN}(M)} .

        The abundance (denominator) is what the XLF measures; the bias per
        (L_X, z) bin is the new information — it breaks agn_rho, which shrinks
        σ_lm and steepens the L_X-selection in halo mass at fixed XLF.
        """
        outs = []
        for (l1, l2) in self.agn_lx_bins:
            n = self._agn_occupation_obs(theta, l1, l2)                 # (NM,)
            n_agn = jnp.trapezoid(H["dndm"] * n, self.m)
            b_agn = jnp.trapezoid(H["dndm"] * H["bias"] * n, self.m) / n_agn
            outs.append(self._wp(b_agn ** 2 * H["pk_lin"], rp=self.rp_wp_agn))
        return jnp.concatenate(outs)

    def _agn_occupation_obs(self, theta, l1_soft, l2_soft):
        r"""AGN occupation for an OBSERVED soft-band L_X bin [l1, l2), (NM,).

        The complete X-ray selection is by observed soft flux: unabsorbed AGN
        enter at their intrinsic soft luminosity (hard + log10 k_h2s(Γ));
        obscured AGN (fraction f_abs) are dimmed by the NH transmission t_soft,
        so their intrinsic bin shifts up by −log10 t_soft.
        """
        shift = jnp.log10(self._k_h2s(theta))            # soft = hard + shift
        labs = float(np.log10(self._t_soft))             # < 0, static template
        fabs = theta[_IDX["agn_fabs"]]
        n_un = self._agn_occupation(theta, l1_soft - shift, l2_soft - shift)
        n_ab = self._agn_occupation(theta, l1_soft - shift - labs, l2_soft - shift - labs)
        return (1.0 - fabs) * n_un + fabs * n_ab

    # ---- abundance observables: n_gal and the stellar-mass function --
    def _occ_above(self, theta, thr):
        """(N_c, N_s) occupation above a stellar-mass threshold ``thr`` on the grid."""
        hp = self._hod(theta)
        nc = _n_cen(self.log10m, thr, hp["lg_m1h"], hp["lg_m0star"], hp["beta"],
                    hp["delta"], hp["gamma"], hp["sigma_lnmstar"], hp["eta"], hp["fc"])
        ns = _n_sat(self.log10m, thr, hp["lg_m1h"], hp["lg_m0star"], hp["beta"],
                    hp["delta"], hp["gamma"], hp["sigma_lnmstar"], hp["eta"], hp["fc"],
                    hp["bsat"], hp["beta_sat"], hp["bcut"], hp["beta_cut"], hp["alpha_sat"])
        return nc, ns

    def _occ_sample(self, theta):
        """(N_c, N_s) occupation of THIS model's sample: threshold or M*-bin.

        Bin mode ([lo, hi)) is the exact difference of two threshold
        occupations — each with its own M_min(thr)-derived msat/mcut — and both
        differences are ≥ 0 because N_cen and N_sat decrease with the threshold.

        SF/quiescent mode (``sfq``) weights the occupations by the ZM16 Weibull
        quenched fractions f_Q(M_h); by construction the "sf" and "q" samples
        sum EXACTLY to the unsplit sample (the regression invariant).
        """
        if self.morph is not None and self.sfq is not None:
            # tier 4: the 4-way (early/late × SF/Q) partition uses the JOINT
            # fractions with the rho_morph_q correlation — at rho = 0 this is
            # the wave-4 independent product exactly.  The four weights
            # (E∩Q, E∩SF, L∩Q, L∩SF) partition unity by construction.
            nc, ns = self._occ_base(theta)
            fq_c, fq_s, s_ms, s_q = self._sfq_weights(theta)
            fe_c, fe_s = self._morph_fractions(theta)
            feq_c, feq_s = self._morph_q_joint(theta)
            if self.morph == "early" and self.sfq == "q":
                wc, ws = feq_c * s_q, feq_s * s_q
            elif self.morph == "early":                     # early ∩ SF
                wc, ws = (fe_c - feq_c) * s_ms, (fe_s - feq_s) * s_ms
            elif self.sfq == "q":                           # late ∩ Q
                wc, ws = (fq_c - feq_c) * s_q, (fq_s - feq_s) * s_q
            else:                                           # late ∩ SF
                wc = (1.0 - fe_c - fq_c + feq_c) * s_ms
                ws = (1.0 - fe_s - fq_s + feq_s) * s_ms
            return nc * wc, ns * ws
        nc, ns = self._occ_sfq(theta)
        if self.morph is not None:
            # marginal early/late weights (no SF/Q selection on this sample)
            fe_c, fe_s = self._morph_fractions(theta)
            if self.morph == "early":
                nc, ns = nc * fe_c, ns * fe_s
            else:
                nc, ns = nc * (1.0 - fe_c), ns * (1.0 - fe_s)
        return nc, ns

    def _occ_sfq(self, theta):
        """Sample occupation after the SF/Q and sSFR selections, BEFORE the
        morphology split (the sample the ``f_early`` observable is measured on)."""
        nc, ns = self._occ_base(theta)
        if self.sfq is not None or self.ssfr_cut is not None:
            fq_c, fq_s, s_ms, s_q = self._sfq_weights(theta)
            if self.sfq == "q":
                nc, ns = nc * fq_c * s_q, ns * fq_s * s_q
            elif self.sfq == "sf":
                nc, ns = nc * (1.0 - fq_c) * s_ms, ns * (1.0 - fq_s) * s_ms
            else:               # pure sSFR-threshold selection (ELG-like)
                nc = nc * ((1.0 - fq_c) * s_ms + fq_c * s_q)
                ns = ns * ((1.0 - fq_s) * s_ms + fq_s * s_q)
        return nc, ns

    def _occ_base(self, theta):
        """Bin/threshold occupation BEFORE any SF/Q or sSFR selection."""
        nc, ns = self._occ_above(theta, self._thr)
        if self._thr_hi is not None:
            nc_hi, ns_hi = self._occ_above(theta, self._thr_hi)
            nc, ns = nc - nc_hi, ns - ns_hi
        return nc, ns

    def _mu_ms(self, theta):
        """Main-sequence mean log10 sSFR of this cell's M* sample (scalar)."""
        mstar_c = (0.5 * (self._thr + self._thr_hi) if self._thr_hi is not None
                   else self._thr + 0.25)
        return (theta[_IDX["ssfr_ms_norm"]]
                + theta[_IDX["ssfr_ms_slope"]] * (mstar_c - 10.5))

    def _morph_fractions(self, theta):
        """(f_early^cen, f_early^sat) Weibull early-type fractions, (NM,) each
        (missing-physics wave 4)."""
        fe_c = f_early_cen(self.log10m, theta[_IDX["log10_M_morph"]],
                           theta[_IDX["beta_morph"]])
        fe_s = f_early_sat(self.log10m, theta[_IDX["log10_M_morph"]],
                           theta[_IDX["beta_morph"]],
                           theta[_IDX["f_morph_sat"]])
        return fe_c, fe_s

    def _morph_q_joint(self, theta):
        r"""(f_{E∩Q}^cen, f_{E∩Q}^sat) joint early∩quenched fractions (tier 4):

        .. math:: f_{E\cap Q} = f_E f_Q + \rho_{\rm morph,Q}
            \sqrt{f_E(1-f_E)\,f_Q(1-f_Q)}

        — the bounded-correlation form; at the ρ = 0 fiducial this is the
        wave-4 independent product exactly.  The floor inside the square root
        keeps the Jacobian finite where the Weibulls saturate to exactly 0/1.
        """
        fe_c, fe_s = self._morph_fractions(theta)
        fq_c, fq_s, _, _ = self._sfq_weights(theta)
        rho = theta[_IDX["rho_morph_q"]]
        v_c = jnp.sqrt(jnp.maximum(fe_c * (1.0 - fe_c)
                                   * fq_c * (1.0 - fq_c), 1e-24))
        v_s = jnp.sqrt(jnp.maximum(fe_s * (1.0 - fe_s)
                                   * fq_s * (1.0 - fq_s), 1e-24))
        return fe_c * fq_c + rho * v_c, fe_s * fq_s + rho * v_s

    def _f_early(self, theta, H):
        r"""Mean early-type fraction of THIS cell's sample, (1,) — the cheap
        morphology observable of the roadmap (Euclid-VIS-like
        :math:`f_{\rm early}(M_*, z)` alongside the cell grid):

        .. math:: f_{\rm early} = \frac{\int dM\,\frac{dn}{dM}\,
            (N_c f_{e,c} + N_s f_{e,s})}{\int dM\,\frac{dn}{dM}\,(N_c + N_s)}

        Measured on the (possibly SF/Q-selected) sample BEFORE any morphology
        split — the tier integration uses morph=None cells.  For SF/Q-selected
        samples the CONDITIONAL early fractions are used (tier 4):
        f_E|Q = f_{E∩Q}/f_Q and f_E|SF = (f_E − f_{E∩Q})/(1 − f_Q) — at the
        ρ_morph,Q = 0 fiducial both reduce to the marginal f_E, so the split
        cells' f_early data measure the morphology–quenching correlation.
        """
        nc, ns = self._occ_sfq(theta)
        fe_c, fe_s = self._morph_fractions(theta)
        if self.sfq in ("sf", "q"):
            fq_c, fq_s, _, _ = self._sfq_weights(theta)
            feq_c, feq_s = self._morph_q_joint(theta)
            if self.sfq == "q":
                fe_c = feq_c / jnp.maximum(fq_c, 1e-12)
                fe_s = feq_s / jnp.maximum(fq_s, 1e-12)
            else:
                fe_c = (fe_c - feq_c) / jnp.maximum(1.0 - fq_c, 1e-12)
                fe_s = (fe_s - feq_s) / jnp.maximum(1.0 - fq_s, 1e-12)
        num = jnp.trapezoid(H["dndm"] * (nc * fe_c + ns * fe_s), self.m)
        den = jnp.trapezoid(H["dndm"] * (nc + ns), self.m)
        return jnp.array([num / den])

    # ---- tier 4: morphology observables --------------------------------
    def _f_early_q(self, theta, H):
        r"""Joint early∩quenched fraction of the cell's BASE sample, (1,) —
        the red-spiral / blue-elliptical census (Galaxy Zoo × SDSS), the
        direct measurement of ``rho_morph_q``:

        .. math:: f_{E\cap Q} = \frac{\int dM\,\frac{dn}{dM}\,
            (N_c f_{E\cap Q,c} + N_s f_{E\cap Q,s})}
            {\int dM\,\frac{dn}{dM}\,(N_c + N_s)}
        """
        nc, ns = self._occ_base(theta)
        feq_c, feq_s = self._morph_q_joint(theta)
        num = jnp.trapezoid(H["dndm"] * (nc * feq_c + ns * feq_s), self.m)
        den = jnp.trapezoid(H["dndm"] * (nc + ns), self.m)
        return jnp.array([num / den])

    def _size_mean(self, theta, H):
        r"""Mean galaxy size of the cell's centrals, ⟨log10 R_e⟩, (1,):

        .. math:: \langle\log_{10} R_e\rangle = \log_{10} f_{\rm size}
            + \langle\log_{10} R_{200c}\rangle_{N_c}
            + \Delta_{\rm size}^{E}\, f_{\rm early}^{\rm cell}

        — the Kravtsov (2013) R_e ≈ 0.015 R_200c relation with a free
        normalisation (evolving via ``f_size_zs``) and the van-der-Wel-like
        early-type offset.  R_200c is the comoving ``r_delta`` of the halo
        grid [Mpc/h]; the constant unit offset to physical kpc is absorbed by
        ``log10_f_size``.  Centrals only (satellite sizes do not follow the
        host halo radius); the cosmology enters through
        R_200c ∝ (M/ρ_crit)^{1/3}.
        """
        nc, _ = self._occ_sample(theta)
        w = jnp.trapezoid(H["dndm"] * nc, self.m)
        mean_lr = jnp.trapezoid(H["dndm"] * nc
                                * jnp.log10(H["r_delta"]), self.m) / w
        return (theta[_IDX["log10_f_size"]] + mean_lr
                + theta[_IDX["dsize_early"]] * self._f_early(theta, H))

    def _f_early_agn(self, theta, H):
        r"""Early-type fraction among the shell's AGN hosts, (1,) — the
        bulge-dominance of X-ray-selected AGN (Kocevski-style), the direct
        probe of the BH–bulge coupling: with ``mbh_bt_slope`` > 0 the AGN
        occupation shifts toward early-type-rich haloes."""
        n_a = self._agn_occupation_obs(theta, 42.0, 44.0)
        fe_c, _ = self._morph_fractions(theta)
        num = jnp.trapezoid(H["dndm"] * n_a * fe_c, self.m)
        den = jnp.trapezoid(H["dndm"] * n_a, self.m)
        return jnp.array([num / den])

    def _wgp(self, theta, H):
        r"""Galaxy–intrinsic-alignment cross w_{g+}(r_p) of the cell, NLA
        with the alignment carried by the early-type fraction (tier 4):

        .. math:: w_{g+}(r_p) = a_{\rm IA}\, f_{\rm early}^{\rm cell}\,
            \frac{C_1\rho_{\rm crit}\Omega_m}{D(z)}\;
            \Delta\Sigma\text{-transform}\big[b_g P_{\rm lin}\big]

        with C₁ρ_crit = 0.0134 (the NLA convention) and D(z) normalised to
        D(0) = 1 through the growth dispatcher (CPL-aware).  Reuses the
        ``_delta_sigma`` J₂-type transform verbatim.  Data: KiDS-1000 /
        DESI direct IA — the amplitude is driven by morphology, so w_{g+}
        self-calibrates the shear IA systematic through this sector.
        """
        c = self._cosmo(theta)
        d_z = growth_factor(self.z_eff, c)           # already D(z)/D(0)
        prefac = (theta[_IDX["a_ia"]] * self._f_early(theta, H)[0]
                  * 0.0134 * c["Omega_m"] / d_z)
        return self._delta_sigma(prefac * H["b_eff"] * H["pk_lin"], theta)

    def _sfq_weights(self, theta):
        r"""(f_Q^cen, f_Q^sat, S_MS, S_Q): the ZM16 quenched fractions and the
        sSFR-cut survival of each population (wave 3).

        With the double-lognormal p(log sSFR | M*) = f_Q N(μ_MS + Δ_Q, σ_Q)
        + (1−f_Q) N(μ_MS, σ_MS), the fraction above a threshold is the
        Gaussian survival ½ erfc[(cut − μ)/√2σ] per component.  S = 1 when no
        cut is set, so the SF+Q ≡ unsplit invariant is untouched.
        """
        fq_c = f_red_cen_zu16(self.log10m, theta[_IDX["log10_Mq_cen"]],
                              theta[_IDX["mu_q_cen"]])
        fq_s = f_red_sat_zu16(self.log10m, theta[_IDX["log10_Mq_sat"]],
                              theta[_IDX["mu_q_sat"]])
        if self.ssfr_cut is None:
            return fq_c, fq_s, 1.0, 1.0
        mu = self._mu_ms(theta)
        s_ms = 0.5 * erfc((self.ssfr_cut - mu)
                          / (jnp.sqrt(2.0) * theta[_IDX["sigma_ms"]]))
        s_q = 0.5 * erfc((self.ssfr_cut - mu - theta[_IDX["dssfr_q"]])
                         / (jnp.sqrt(2.0) * _SIG_SSFR_Q))
        return fq_c, fq_s, s_ms, s_q

    def _n_gal(self, theta, H):
        """Comoving galaxy number density n̄_g [h³ Mpc⁻³] of the sample.

        Threshold mode: n̄(>M*_thr); bin mode: the binned count density (which
        IS the stellar-mass-function datum of the cell).  n̄_g ∝ f_c, so this
        observable is what breaks the otherwise-degenerate f_c.
        """
        nc, ns = self._occ_sample(theta)
        return jnp.array([jnp.trapezoid(H["dndm"] * (nc + ns), self.m)])

    def _smf(self, theta, H):
        r"""Stellar-mass function Φ(M*) = −d n̄(>M*)/d log10 M*  [h³ Mpc⁻³ dex⁻¹].

        Vmaps the ZM15 threshold occupation over the M* grid, integrates against
        dn/dM, and differentiates the cumulative number density — the same
        estimator as the production ``_predict_smf``.  Constrains the SHMR shape.
        """
        def cum(thr):
            nc, ns = self._occ_above(theta, thr)
            return jnp.trapezoid(H["dndm"] * (nc + ns), self.m)
        nbar = jax.vmap(cum)(self.logmstar_smf)
        return -jnp.gradient(nbar, self.logmstar_smf)

    # ---- public: full prediction and data vector --------------------
    def predict(self, theta, which=None):
        """Return a dict of observable arrays on the full grids."""
        theta = self._theta_eff(theta)      # tier-2 z-evolution (identity at fiducial)
        which = which or OBSERVABLES
        out = {}
        if any(o in which for o in ("wp", "ds", "n_gal", "smf", "wp_agn",
                                    "himf", "sfrd", "ds_agn", "ncl",
                                    "f_early", "f_early_q", "size",
                                    "f_early_agn", "wgp")):
            H0 = self._halo_common(theta, self.z_eff)
            if "wp" in which or "ds" in which:
                P_gg, P_gm = self._pk_gg_gm(H0, theta)
                if "wp" in which:
                    out["wp"] = self._wp(P_gg)
                if "ds" in which:
                    out["ds"] = self._delta_sigma(P_gm, theta)
            if "n_gal" in which:
                out["n_gal"] = self._n_gal(theta, H0)
            if "smf" in which:
                out["smf"] = self._smf(theta, H0)
            if "wp_agn" in which:
                out["wp_agn"] = self._wp_agn(theta, H0)
            if "himf" in which:
                out["himf"] = self._himf(theta, H0)
            if "sfrd" in which:
                out["sfrd"] = self._sfrd(theta, H0)
            if "ds_agn" in which:
                parts = []
                for (l1, l2) in self.agn_lx_bins:
                    n_a, n_agn, b_agn = self._agn_tracer(theta, H0, l1, l2)
                    P_am = self._pk_am(theta, H0, n_a, n_agn, b_agn)
                    parts.append(self._delta_sigma(P_am, theta))
                out["ds_agn"] = jnp.concatenate(parts)
            if "ncl" in which:
                out["ncl"] = self._ncl(theta, H0)
            if "f_early" in which:
                out["f_early"] = self._f_early(theta, H0)
            if "f_early_q" in which:
                out["f_early_q"] = self._f_early_q(theta, H0)
            if "size" in which:
                out["size"] = self._size_mean(theta, H0)
            if "f_early_agn" in which:
                out["f_early_agn"] = self._f_early_agn(theta, H0)
            if "wgp" in which:
                out["wgp"] = self._wgp(theta, H0)
        # Galaxy-window angular spectra (gas cross + galaxy×CMB-lensing) share the
        # per-z halo quantities on the galaxy n(z) grid.
        ang = [o for o in ("cl_gX", "cl_gy", "cl_XX", "cl_gHI") if o in which]
        t3maps = [o for o in ("cl_gR", "cl_gI", "cl_RR", "cl_II", "cl_aR",
                              "cl_aI", "cl_ag", "cl_yy", "cl_HIHI")
                  if o in which]
        if ang or t3maps or "cl_gkCMB" in which:
            Hs = [self._halo_common(theta, float(z)) for z in np.asarray(self.z_grid)]
            fns = {"cl_gX": self._pk_gX, "cl_gy": self._pk_gy, "cl_XX": self._pk_XX,
                   "cl_gHI": self._pk_gHI}
            for o in ang:
                if self.xray_bands is not None and o in ("cl_gX", "cl_XX"):
                    # band mode: P is (Nb, Nk) per z → one Limber per band,
                    # concatenated band-major (matches grid_of's tiled ℓ).
                    Ps = jnp.stack([fns[o](theta, H) for H in Hs])      # (Nz, Nb, Nk)
                    out[o] = jnp.concatenate([
                        self._limber_from_stack(
                            theta, jnp.log(jnp.maximum(Ps[:, b, :], 1e-30)))
                        for b in range(len(self.xray_bands))])
                else:
                    stack = jnp.stack([jnp.log(jnp.maximum(fns[o](theta, H), 1e-30)) for H in Hs])
                    out[o] = self._limber_from_stack(theta, stack)
            if t3maps:
                out.update(self._predict_maps(theta, Hs, t3maps))
            if "cl_gkCMB" in which:
                out["cl_gkCMB"] = self._cl_gkCMB(theta, Hs)
        # Matter-power lensing spectra (cosmic shear + shear×CMB-lensing) share the
        # P_mm stack on the shear source grid.
        if "cl_kk" in which or "cl_shear_kCMB" in which:
            stack_shear = self._pmm_logstack(theta, self.z_shear)
            if "cl_kk" in which:
                out["cl_kk"] = self._cosmic_shear(theta, stack_shear)
            if "cl_shear_kCMB" in which:
                out["cl_shear_kCMB"] = self._cl_shear_kCMB(theta, stack_shear)
        if "cl_kCMB" in which:
            out["cl_kCMB"] = self._cl_kCMB(theta)
        if "xlf" in which:
            out["xlf"] = self._xlf(theta)
        if "rlf" in which:
            out["rlf"] = self._rlf(theta)
        if "ssfr" in which:
            out["ssfr"] = self._ssfr(theta)
        if "oiilf" in which:
            out["oiilf"] = self._oiilf(theta)
        if "ilf" in which:
            out["ilf"] = self._ilf(theta)
        # tier 3: galaxy band LFs + AGN UV/opt LFs
        if "uvlf" in which:
            out["uvlf"] = self._uvlf(theta)
        if "optlf" in which:
            out["optlf"] = self._optlf(theta)
        if "nirlf" in which:
            out["nirlf"] = self._nirlf(theta)
        if "half" in which:
            out["half"] = self._half(theta)
        if "qlf_uv" in which:
            out["qlf_uv"] = self._qlf(theta, self.logluv_qlf,
                                      theta[_IDX["agn_bc_uv"]])
        if "qlf_opt" in which:
            out["qlf_opt"] = self._qlf(theta, self.loglopt_qlf,
                                       theta[_IDX["agn_bc_opt"]])
        return out

    def cl_gg_fiducial(self, theta):
        """Galaxy angular auto C_ℓ over the cell window (fiducial-only helper
        for the tier-2 noise module — the galaxy-side Knox term of C_gX etc.)."""
        theta = self._theta_eff(jnp.asarray(theta))
        Hs = [self._halo_common(theta, float(z)) for z in np.asarray(self.z_grid)]
        stack = jnp.stack([jnp.log(jnp.maximum(self._pk_gg_gm(H, theta)[0], 1e-30))
                           for H in Hs])
        return self._limber_from_stack(theta, stack)

    def cl_aa_fiducial(self, theta, l1, l2):
        """AGN angular auto C_ℓ over the window for one observed-soft L_X bin,
        plus the 3-D AGN density (fiducial-only helper for the tier-3 AGN-cross
        Knox noise).  2-halo b²P_lin — Bernoulli centrals have no 1-halo
        self-pairs (the wp_agn convention); the caller adds the shot term."""
        theta = self._theta_eff(jnp.asarray(theta))
        Hs = [self._halo_common(theta, float(z)) for z in np.asarray(self.z_grid)]
        stack, n_agn0 = [], None
        for H in Hs:
            _, n_agn, b_agn = self._agn_tracer(theta, H, float(l1), float(l2))
            stack.append(jnp.log(jnp.maximum(b_agn ** 2 * H["pk_lin"], 1e-30)))
            if abs(float(H["z"]) - self.z_eff) < 1e-9:
                n_agn0 = n_agn
        if n_agn0 is None:
            H0 = self._halo_common(theta, self.z_eff)
            _, n_agn0, _ = self._agn_tracer(theta, H0, float(l1), float(l2))
        cl = self._limber_from_stack(theta, jnp.stack(stack))
        return cl, float(n_agn0)

    def grid_of(self, name):
        """Return the abscissa grid (r_p, ℓ, or log10 L_X) for observable ``name``."""
        if name.startswith("cl_"):
            if name in ("cl_gX", "cl_XX") and self.xray_bands is not None:
                return jnp.asarray(np.tile(np.asarray(self.ell), len(self.xray_bands)))
            if name == "cl_kk" and self.n_shear_bins > 1:
                return jnp.asarray(np.tile(np.asarray(self.ell), len(self.shear_pairs)))
            if name == "cl_shear_kCMB" and self.n_shear_bins > 1:
                return jnp.asarray(np.tile(np.asarray(self.ell), self.n_shear_bins))
            # tier-3 stacked map spectra (orderings match _predict_maps)
            reps = {"cl_gR": lambda: len(self.radio_map_bands),
                    "cl_RR": lambda: len(self.radio_map_bands),
                    "cl_gI": lambda: len(self.ir_map_bands),
                    "cl_II": lambda: len(self.ir_map_bands),
                    "cl_aR": lambda: len(self.agn_lx_bins) * len(self.radio_map_bands),
                    "cl_aI": lambda: len(self.agn_lx_bins) * len(self.ir_map_bands),
                    "cl_ag": lambda: len(self.agn_lx_bins)}
            if name in reps:
                return jnp.asarray(np.tile(np.asarray(self.ell), reps[name]()))
            return self.ell
        return {"wp": self.rp_wp, "ds": self.rp_ds,
                "xlf": self.loglx_xlf, "n_gal": jnp.array([0.0]),
                "smf": self.logmstar_smf, "ssfr": jnp.array([0.0]),
                "sfrd": jnp.array([0.0]), "ncl": jnp.array([0.0]),
                "f_early": jnp.array([0.0]),
                "f_early_q": jnp.array([0.0]), "size": jnp.array([0.0]),
                "f_early_agn": jnp.array([0.0]), "wgp": self.rp_ds,
                "rlf": self.loglr_rlf, "himf": self.logmhi_himf,
                "oiilf": self.logloii_oiilf, "ilf": self.loglir_ilf,
                "uvlf": self.logluv_uvlf, "optlf": self.loglopt_optlf,
                "nirlf": self.loglnir_nirlf, "half": self.loglha_half,
                "qlf_uv": self.logluv_qlf, "qlf_opt": self.loglopt_qlf,
                "wp_agn": jnp.asarray(np.tile(np.asarray(self.rp_wp_agn),
                                              len(self.agn_lx_bins))),
                "ds_agn": jnp.asarray(np.tile(np.asarray(self.rp_ds),
                                              len(self.agn_lx_bins)))}[name]

    def full_data_vector_fn(self, which=None):
        """Return ``f(theta) -> concat of all observables (full grids)`` + row metadata.

        Metadata is ``(row_obs, row_x)``: for each row, the observable name and
        its abscissa (r_p [Mpc/h] for wp/ds, ℓ for the angular spectra).  A single
        ``jax.jacfwd`` of ``f`` then supports every (probe, scale-cut) selection by
        row masking — no need to recompute the Jacobian per configuration.
        """
        which = list(which or OBSERVABLES)
        row_obs, row_x = [], []
        for name in which:
            g = np.asarray(self.grid_of(name))
            row_obs += [name] * g.size
            row_x += list(g)

        def f(theta):
            pred = self.predict(theta, which)
            return jnp.concatenate([pred[name] for name in which])
        return f, np.array(row_obs), np.array(row_x, dtype=float)

    def scale_cut_mask(self, row_obs, row_x, rmin):
        """Boolean row mask for a scale cut: r_p>rmin (projected), ℓ<χ(z_eff)/rmin (angular).

        The ``xlf`` abundance observable has no length scale, so it is always kept.
        """
        chi_eff = float(np.asarray(comoving_distance(
            jnp.asarray([self.z_eff]), _FIXED_COSMO["h"], 0.31)).ravel()[0]) * _FIXED_COSMO["h"]
        ell_max = chi_eff / float(rmin)
        row_obs = np.asarray(row_obs)
        proj = np.isin(row_obs, ["wp", "ds", "wp_agn", "ds_agn", "wgp"])
        ang = np.array([str(o).startswith("cl_") for o in row_obs])
        keep = np.ones(len(row_obs), dtype=bool)          # abundance (xlf/n_gal/smf): always kept
        keep = np.where(proj, row_x > float(rmin), keep)
        keep = np.where(ang, row_x < ell_max, keep)
        return keep

    def data_vector_fn(self, which, rmin):
        """Return ``f(theta) -> concatenated masked data vector`` for jacfwd.

        The scale-cut selection is baked in as static integer indices, so the
        returned function has a fixed output shape (required by ``jax.jacfwd``).
        Projected observables keep r_p > rmin; angular spectra keep
        ℓ < χ(z_eff)/rmin.
        """
        which = list(which)
        chi_eff = float(np.asarray(comoving_distance(
            jnp.asarray([self.z_eff]), _FIXED_COSMO["h"], 0.31)).ravel()[0]) * _FIXED_COSMO["h"]
        ell_max = chi_eff / float(rmin)
        idx = {}
        for name in which:
            g = np.asarray(self.grid_of(name))
            if name in ("wp", "ds", "wp_agn", "ds_agn", "wgp"):
                sel = np.where(g > float(rmin))[0]
            elif name in ("xlf", "n_gal", "smf", "rlf", "himf", "ssfr",
                          "sfrd", "oiilf", "ilf", "ncl", "uvlf", "optlf",
                          "nirlf", "half", "qlf_uv", "qlf_opt", "f_early",
                          "f_early_q", "size", "f_early_agn"):
                sel = np.arange(g.size)            # abundance: no scale cut
            else:
                sel = np.where(g < ell_max)[0]
            idx[name] = jnp.asarray(sel, dtype=int)

        def f(theta):
            pred = self.predict(theta, which)
            parts = [pred[name][idx[name]] for name in which if idx[name].size > 0]
            return jnp.concatenate(parts) if parts else jnp.zeros(0)
        return f, idx
