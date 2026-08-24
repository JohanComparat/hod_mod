r"""Fiducial parameters, labels and priors for the sensitivity forecast.

Fiducials are the on-disk campaign best fits:

* cosmology — Planck 2018 (``LinearPowerSpectrum.default_cosmology`` + σ8);
* HOD — ``bgs_zm15_joint/map_result.json`` (BGS M*>10 ZM15 MAP, 9 free);
* X-ray gas / AGN — ``xray_joint_bands/S1_bands_summary.json`` (S1 MAP).

If those JSON files are not found (e.g. ``$HOD_MOD_RESULTS`` unset), hard-coded
values from the plan are used so the module always imports.
"""

from __future__ import annotations

import json
import os

import numpy as np

from hod_mod.forecast.forward_jax import (
    PARAM_NAMES,
    _IDX,
    N_PARAM,
    TIER2_PROMOTED,
    TIER2_ZSLOPES,
    TIER2_EXTENSION,
    TIER3_EXTENSION,
)

# --- hard-coded fallbacks (values read from the MAP JSONs, see module docstring)
_FIDUCIAL_DEFAULT = {
    "Omega_m": 0.3100, "sigma8": 0.8111, "h": 0.6736, "n_s": 0.9649, "Omega_b": 0.0493,
    "lg_m1h": 11.67701, "lg_m0star": 10.47679, "beta": 0.734685,
    "delta": 0.661630, "gamma": 0.540446, "sigma_lnmstar": 1.024268,
    "eta": -0.314724, "fc": 0.300001, "bsat": 36.37306,
    "lx_norm": 45.1151, "lx_slope": 1.47861, "kt_norm": 1.12745,
    "kt_slope": 0.378256, "p2": 0.167427, "r_max": 3.732144,
    "log10DC": -0.9088, "beta_pressure": 0.0,
    "log10_M_pivot": 13.5, "log10_eta_min": -0.22,   # BaryonFractionSigmoid defaults
    # Powell AGN-XLF sector — powell_agn/powell_map_aird15.json (MAP fit values).
    "agn_mu_bh": 7.5184, "agn_al_bh": 0.8037, "agn_sig_bh": 0.2067,
    "agn_log10_lstar": -1.0120, "agn_delta1": 0.3123, "agn_delta2": 2.4761,
    "agn_log10_ferdf": -1.7243,
    # --- tier-2 promoted nuisances: fiducial == the former fixed constant, so
    # the fiducial prediction is bit-identical to the tier-1 model.
    "beta_sat": 0.9, "bcut": 0.86, "beta_cut": 0.41, "alpha_sat": 1.0,   # _FIXED_HOD
    "beta_b": 1.5, "log10_M_eta": 13.0, "beta_eta": 1.5,                 # _FIXED_BARYON
    "alpha_in_gas": 0.9, "alpha_tr_gas": 2.0,                            # gas emissivity slopes
    "p03_pressure": 1.627e-6, "c_dpm_pressure": 2.772, "alpha_in_pressure": 0.3,
    "alpha_tr_pressure": 1.3, "alpha_out_pressure": 4.1,                 # DPM Model 2 GNFW
    "agn_rho": 0.0, "agn_sig_mstar": 0.20,                               # Powell Model 2 internals
    # --- tier-2 z-evolution slopes (per ln(1+z), pivot z=0.3): fiducial 0 =
    # the tier-1 no-evolution model (self-similar E(z) scalings untouched).
    "lg_m1h_zs": 0.0, "lg_m0star_zs": 0.0, "sigma_lnmstar_zs": 0.0,
    "lx_zs": 0.0, "kt_zs": 0.0,
    "agn_log10_ferdf_zs": 0.0, "agn_log10_lstar_zs": 0.0,
    # --- tier-2 X-ray spectral sector (multi-band APEC layer).  Fiducials
    # reproduce the production conventions: isothermal halos (t_prof_slope=0),
    # Z=0.3 Z_sun (DPM normalisation, gas/metallicity.py), Γ=1.8 AGN power law
    # (fit_xray_joint_bands._GAMMA_AGN), ~30% obscured at NH=1e22.
    "t_prof_slope": 0.0, "z_gas_norm": 0.3, "z_gas_mslope": 0.0, "z_gas_zs": 0.0,
    "agn_gamma": 1.8, "agn_fabs": 0.3, "agn_mu_bh_zs": 0.0,
    # --- missing-physics extension (docs/missing_physics.rst).  eps_sn is the
    # former _EPS_SN constant and (w0, wa) are the ΛCDM limit, so the CPL growth
    # ODE reproduces the tier-2 prediction exactly.  sum_mnu is the minimal
    # normal hierarchy, matching LinearPowerSpectrum.default_cosmology(); the
    # quenching and FP/HI parameters only touch the new opt-in observables.
    "eps_sn": 0.1,
    "w0": -1.0, "wa": 0.0, "sum_mnu": 0.06,
    # ZM16 halo-quenching MAP (Zu & Mandelbaum 2016, the in-repo fiducials)
    "log10_Mq_cen": 11.78, "mu_q_cen": 0.41,
    "log10_Mq_sat": 12.19, "mu_q_sat": 0.24,
    "dlx_quenched": 0.0,
    # fundamental plane of BH activity (Merloni+03 fiducials, Gültekin+19 priors)
    "agn_xi_rx": 0.60, "agn_xi_rm": 0.78, "agn_b_r": 7.33, "agn_sig_r": 0.88,
    # M_HI(M_h) halo model (Villaescusa-Navarro+2018, z=0)
    "log10_M0_hi": 10.63, "log10_Mmin_hi": 12.3, "alpha_hi": 0.24,
    # --- wave 2: SN wind loading (η_0 = 0 ⇒ exactly the tier-2 η(M) sigmoid),
    # Speagle+14-like main sequence, quenched-HI deficit.
    "eta_w_norm": 0.0, "alpha_w": 1.0,
    "ssfr_ms_norm": -9.9, "ssfr_ms_slope": -0.25, "ssfr_ms_zs": 0.0,
    "dhi_quenched": 0.0,
    # --- wave 3: continuous sSFR distribution, [OII] calibration, radio-loud
    # jets (Best & Heckman-like) and the AGN IR bolometric correction.
    "sigma_ms": 0.30, "dssfr_q": -1.6,
    "loii_norm": 40.85,           # log10 L[OII]/SFR (Kennicutt-like)
    "f_loud0": 0.01, "beta_loud": 2.0, "b_jet": 41.0,
    "agn_bc_ir": -0.52,           # L_IR(6um) ≈ 0.3 L_bol
    # --- tier 3: multi-wavelength SED calibrations (radio/IR maps, band LFs).
    "l14_sfr": 28.20, "alpha_syn": 0.70,     # radio-FIR (Murphy+11) + synchrotron slope
    "lir_sfr": 43.41, "bir_color": 0.0,      # L_IR/SFR (Kennicutt-Murphy) + dust color tilt
    "ml_nir": 33.20, "ml_opt": 33.10,        # log10 nuLnu/M* (3.4um; r band, SF)
    "dopt_q": -0.20,                         # quiescent r-band offset [dex]
    "luv_norm": 42.65, "tau_uv_mslope": 0.0,  # UV/SFR incl. attenuation + its M* slope
    "lha_norm": 41.27,                       # log10 L_Halpha/SFR (Kennicutt)
    "agn_bc_uv": -0.62, "agn_bc_opt": -0.72,  # AGN bolometric corrections (1450A, 4400A)
    # --- wave 4: conditional galaxy morphology (early-type Weibull) + the
    # BH-bulge coupling (0 fiducial = the pure Powell chain).
    "log10_M_morph": 12.5, "beta_morph": 0.8,
    "f_morph_sat": 0.2,
    "mbh_bt_slope": 0.0,
    # --- tier 4: morphology observables (joint E/Q, sizes, IA).
    "rho_morph_q": 0.0,           # 0 = the wave-4 independence, exact
    "log10_f_size": -1.824,       # log10(R_e/R_200c), Kravtsov 2013 0.015
    "dsize_early": -0.20,         # early-type size offset [dex] (van der Wel)
    "f_size_zs": 0.0,
    "a_ia": 2.0,                  # NLA amplitude of early types
}

# ---------------------------------------------------------------------------
# Strict parameter bounds
# ---------------------------------------------------------------------------
# A hard (lo, hi) for every entry of PARAM_NAMES: outside it the value is not
# merely improbable but wrong -- a negative density, a fraction above one, a
# width of zero that the code divides by, or a point outside the calibration
# range of the emulator that consumes it.  Gaussian prior widths cannot express
# this: BROAD_PRIOR_SIGMA puts ~40% of its mass on sum_mnu < 0.
#
# Each entry is tagged with what kind of statement it is, because the
# distinction matters when someone wants to relax one:
#
#   physical      a law forbids the outside (a density or mass cannot be
#                 negative; Sigma m_nu >= 0)
#   definitional  a fraction lies in [0,1]; a width is strictly positive; an
#                 ordering holds
#   validity      the calibration or training range of the model that consumes
#                 it (the CosmoPower TRAINING_BOX, the APEC table edges, the
#                 grid the distilled neutrino ratios were measured on)
#   prior         a defensible but conventional range, used only where none of
#                 the above applies -- these are the ones to argue with
#
# The bounds are deliberately the widest defensible choice. A bound that
# excludes a value the literature reports, or that this package itself fits to,
# is a bug: tests/test_param_bounds.py checks the shipped benchmark best-fits
# against them for exactly that reason.
#
# Sentinels for genuinely one-sided quantities.
_INF = 1e30

PARAM_BOUNDS = {
    # -- tier 1: cosmology, occupation, gas, AGN, survey nuisance --------------
    # validity: Omega_m>0 is the hard wall (EH98 omh2**-0.419 at
    # power_spectrum.py:512 and the Carroll growth om**(4/7) at
    # halo_mass_function.py:116 both NaN at Omega_m<=0), but the ...
    'Omega_m': (0.1, 0.6),
    # prior: sigma8 is an rms amplitude so it must be >0 --
    # halo_mass_function.py:1126 uses rescale2=sigma8^2/s2_8 and therefore
    # silently accepts sigma8<0 as |sigma8| -- and measured ...
    'sigma8': (0.2, 1.6),
    # validity: Exactly TRAINING_BOX['h'] in forecast/pk_cosmopower.py:69-76,
    # the box of the default runtime backend, outside which the network
    # extrapolates and the 0.141%-vs-CLASS ...
    'h': (0.6, 0.82),
    # validity: TRAINING_BOX['n_s'] in forecast/pk_cosmopower.py:69-76; the
    # fiducial 0.9649 is central, and this range keeps n_eff = dlnP/dlnk far
    # from the c_diemer15 pole at ...
    'n_s': (0.84, 1.1),
    # validity: TRAINING_BOX omega_b=Omega_b h^2 in [0.01875,0.02625] over h
    # in [0.64,0.82] maps to Omega_b in [0.0279,0.0641] (measured); this also
    # enforces both hard walls by ...
    'Omega_b': (0.028, 0.064),
    # validity: Below 11 the hard-coded logspace(10,16,512) halo grid
    # truncates the occupation (at lg_m1h=10, n_gal loses 13% and b_eff shifts
    # 6% against a 1e8 floor; at 11.0 the two ...
    'lg_m1h': (10.0, 15.5),
    # prior: Reused verbatim from scripts/fitting/fit_bgs_smf_wp.py:150-165;
    # the ZM15 Eq.19 pivot stellar mass must sit strictly inside the [4,13]
    # bisection bracket and inside any ...
    'lg_m0star': (8.0, 11.5),
    # definitional: beta = d log10 M_h / d log10 M* must be strictly positive
    # or ZM15 Eq.19 is non-monotone and the bisection at
    # zumandelbaum15.py:81-84 -- which has no bracket check and no ...
    'beta': (0.05, 2.0),
    # validity: MEASURED: at the fiducial (lg_m1h, lg_m0star, beta, gamma) the
    # inverse-SHMR bisection pins at the bracket top -- M*(1e16)=13.0000
    # exactly -- for delta<=0.32 and is clean ...
    'delta': (0.1, 3.0),
    # prior: gamma>0 is required for the (M*/M*0)^{-gamma} denominator of ZM15
    # Eq.19 to act as a low-mass turnover -- gamma<=0 reverses its role and
    # drives the jnp.maximum(m,1e-30) ...
    'gamma': (0.05, 10.0),
    # definitional: A lognormal width must be >0: it divides the erfc argument
    # at zumandelbaum15.py:155, and sigma<=0 flips that argument's sign and
    # inverts N_cen entirely; 2.0 is the ...
    'sigma_lnmstar': (0.01, 2.0),
    # prior: MEASURED AND ALARMING: sigma_lnMstar(M_h)=sigma_0+eta*(log10M_h-
    # lg_m1h) must stay >0 across the whole [10,16] grid, i.e. eta >
    # -sigma_0/(16-lg_m1h) = -0.237 at the ...
    'eta': (-0.5, 0.5),
    # definitional: A central-occupation ceiling is a probability: fc>1 gives
    # N_cen>1 and fc<0 gives negative occupation (zumandelbaum15.py:156,
    # unclipped, verified), while fc=0 zeroes ...
    'fc': (0.01, 1.0),
    # physical: M_sat = bsat*1e12*(M_min/1e12)^{beta_sat} is a mass, so bsat>0
    # strictly -- bsat=0 gives N_sat=inf at every mass and bsat=-1 gives
    # finite NEGATIVE satellite counts with ...
    'bsat': (0.1, 200.0),
    # prior: log10 of the 0.5-2 keV luminosity in erg/s at the pivot
    # M500c=1e15 with E(z)=1 (forward_jax.py:982, where the -45 is only a
    # float-safety offset restored downstream) ...
    'lx_norm': (41.0, 48.0),
    # prior: d log10 L_X / d log10 M500c: the self-similar soft-band value is
    # 1.5-1.8 and measured slopes span 1.3-2.2 with the fiducial at 1.479,
    # while the floor is kept above zero ...
    'lx_slope': (0.5, 3.0),
    # validity: log10 kT [keV] at M500c=1e14 must land inside the distilled
    # APEC grid T in [0.05,60] keV (apec_bands._T_MIN/_T_MAX), outside which
    # _bilinear_logtable ...
    'kt_norm': (-1.3, 1.778),
    # definitional: kT must not decrease with halo mass -- kt_slope<=0 makes
    # clusters cooler than groups and inverts the whole APEC band partition --
    # and the self-similar value is 2/3 with ...
    'kt_slope': (0.0, 1.5),
    # definitional: alpha_out = alpha_in_gas + 2*p2 (forward_jax.py:911-914),
    # so p2>0 is exactly the requirement that the gas emissivity fall off
    # faster outside than inside; p2<=0 gives a ...
    'p2': (0.01, 4.0),
    # prior: Truncation radius in units of R_200c: r_max<=0 zeroes the
    # normalising denominator in _profile_uk_normalized
    # (forward_jax.py:277-292) and returns NaN, r_max<1 would ...
    'r_max': (1.0, 6.0),
    # definitional: A duty cycle is a fraction so DC=10^{log10DC}<=1, and
    # nothing enforces it (agn/duty_cycle.py:393) -- log10DC>0 means more than
    # 100% of halos host an active nucleus; the ...
    'log10DC': (-4.0, 0.0),
    # definitional: beta_P = 0.85 (DPM Model 2) + beta_pressure
    # (forward_jax.py:1070) and the tilt must keep beta_P in [0,2.0]: beta_P<0
    # makes electron pressure DECREASE with halo mass, and ...
    'beta_pressure': (-0.85, 1.15),
    # prior: log10 of the halo mass [M_sun/h] at which the baryon-fraction
    # sigmoid reaches half f_b^cosmic, reused verbatim from
    # fit_bgs_multiprobe.py:804 (FLAMINGO ...
    'log10_M_pivot': (-4.0, 15.0),
    # prior: eta_min=c_gas/c_DM is positive by construction (it enters as
    # 10^x), but the NFW normalisation ln(1+c)-c/(1+c) is a complete O(c)
    # cancellation, so log10_eta_min below ...
    'log10_eta_min': (-2.0, 0.3),
    # prior: log10 M_BH [M_sun] at the pivot M*=1e11, reused from
    # scripts/fitting/fit_powell_agn.py:37-38 (the widest shipped box;
    # full_joint.py:58-65 uses the tighter (7.0,8.5)) ...
    'agn_mu_bh': (6.5, 9.0),
    # definitional: d log M_BH / d log M* must be >=0 or black holes shrink as
    # their hosts grow, and it also enters the kernel width only as
    # alpha_BH^2*sigma_M*^2*(1-rho) ...
    'agn_al_bh': (0.0, 2.0),
    # definitional: The intrinsic scatter of log10 M_BH in dex must be >0
    # because it is the only term in sigma_lm^2 =
    # alpha_BH^2*sigma_M*^2*(1-rho) + sigma_BH^2 guaranteed positive, so it
    # ...
    'agn_sig_bh': (0.02, 1.5),
    # validity: The ERDF break lambda* must sit strictly inside the tabulated
    # loglam = linspace(-3.0, 1.5, 120) grid (forward_jax.py:581) or the
    # broken power law 1/(x^{d1}+x^{d2}) ...
    'agn_log10_lstar': (-2.8, 0.5),
    # prior: Faint-end ERDF slope in 1/(x^{delta1}+x^{delta2})
    # (forward_jax.py:1865): delta1<0 makes the ERDF turn OVER toward low
    # Eddington ratio, contradicting every measured ERDF ...
    'agn_delta1': (0.0, 1.5),
    # definitional: Bright-end ERDF slope, with the floor set equal to the
    # agn_delta1 ceiling so that delta1 < delta2 is enforced by construction
    # rather than by an unexpressible ordering ...
    'agn_delta2': (1.5, 6.0),
    # definitional: f_ERDF is the AGN active fraction, so 10^{log10_ferdf}<=1
    # -- the code multiplies it in completely unclipped (forward_jax.py:1376,
    # :1887 and every band LF), and log10>0 ...
    'agn_log10_ferdf': (-5.0, 0.0),
    # -- tier 2: promoted ------------------------------------------------------
    # prior: Power-law index of M_sat on M_min, M_sat proportional to
    # (M_min/1e12)^{beta_sat}: negative values invert the sense so that richer
    # central samples get a SMALLER satellite ...
    'beta_sat': (0.0, 2.0),
    # physical: M_cut = bcut*1e12*(M_min/1e12)^{beta_cut} is a mass, so bcut>0
    # strictly -- bcut<0 turns exp(-M_cut/M_h) at zumandelbaum15.py:215 into
    # exp(+|M_cut|/M_h), an exponentially ...
    'bcut': (0.01, 20.0),
    # prior: Power-law index of M_cut on M_min; same argument and the same
    # shipped (0.0,2.0) box as beta_sat (fit_bgs_smf_wp.py,
    # fit_bgs_zm15_joint.py), with the fiducial 0.41 well ...
    'beta_cut': (0.0, 2.0),
    # definitional: N_sat proportional to (M_h/M_sat)^{alpha_sat} must
    # increase with halo mass: alpha_sat<=0 inverts the satellite occupation,
    # and in the shared base.py:55-56 form the ...
    'alpha_sat': (0.05, 3.0),
    # definitional: Sharpness of the baryon-fraction sigmoid f_b =
    # f_b^cosmic/[1+(M_pivot/M)^{beta_b}] (baryon_fraction.py:101): beta_b>0
    # IS the feedback-expels-at-low-mass sense, and ...
    'beta_b': (0.1, 5.0),
    # prior: log10 of the break halo mass [M_sun/h] of the gas-concentration
    # sigmoid eta(M) (clustering.py:1051): it must sit inside the fixed
    # [10,16] halo grid with room on both ...
    'log10_M_eta': (11.0, 15.0),
    # definitional: Sharpness of the eta(M) gas-concentration transition, eta
    # = 1-(1-eta_min)/[1+(M/M_eta)^{beta_eta}] (clustering.py:1051):
    # beta_eta>0 is required for eta to rise from ...
    'beta_eta': (0.1, 5.0),
    # physical: MEASURED: the X-ray emissivity goes as n_e^2 proportional to
    # x^{-2*alpha_in}, so the inner integral int x^{2-2*alpha_in} dx diverges
    # for alpha_in>=1.5 and the answer ...
    'alpha_in_gas': (0.0, 1.3),
    # physical: alpha_tr is the divisor of the gNFW exponent (alpha_in-
    # alpha_out)/alpha_tr (forward_jax.py:300, gas/density.py:127), so
    # alpha_tr=0 is a literal division by zero and ...
    'alpha_tr_gas': (0.1, 5.0),
    # physical: An electron pressure in keV cm^-3 is strictly positive, and it
    # enters cl_gy as a pure linear amplitude p03_pressure/P03_fid
    # (forward_jax.py:1076), so a negative value ...
    'p03_pressure': (1e-08, 0.0031623),
    # physical: A concentration is strictly positive and here it is a divisor
    # -- R_s = (R_200c/c_DPM)/eta at forward_jax.py:1061 -- so c<=0 gives a
    # negative or infinite scale radius and ...
    'c_dpm_pressure': (0.3, 20.0),
    # physical: MEASURED: the Compton-y integral is linear in P, int
    # x^{2-alpha_in} dx, which diverges at the inner boundary for alpha_in>=3
    # -- the 200- and 800-node Gauss-Legendre ...
    'alpha_in_pressure': (-0.7, 2.5),
    # physical: Same divisor as alpha_tr_gas: the GNFW exponent is (alpha_in-
    # alpha_out)/alpha_tr (forward_jax.py:306-309, _gnfw_pressure), so
    # alpha_tr=0 divides by zero and alpha_tr<0 ...
    'alpha_tr_pressure': (0.1, 5.0),
    # definitional: alpha_out>alpha_in is required for a decaying pressure
    # profile and no box can express an ordering, so the floor is set equal to
    # the alpha_in_pressure ceiling (2.5) ...
    'alpha_out_pressure': (2.0, 8.0),
    # definitional: rho enters only through sigma_lm^2 =
    # alpha_BH^2*sigma_M*^2*(1-rho) + sigma_BH^2 (forward_jax.py:1861-1863)
    # and is completely UNCLIPPED in the forecast, so rho>1 makes ...
    'agn_rho': (-1.0, 0.99),
    # definitional: Lognormal width of log10 M* at fixed halo mass in dex; it
    # enters only squared and multiplied by (1-rho) (forward_jax.py:1861), so
    # 0 is admissible because agn_sig_bh ...
    'agn_sig_mstar': (0.0, 1.0),
    # -- tier 2: redshift slopes -----------------------------------------------
    # prior: Additive dex-per-ln(1+z) slope on lg_m1h via _theta_eff
    # (forward_jax.py:713); Behroozi+19/Moster+18 give d lg M1/d ln(1+z) ~
    # 0.6, so +-2 is >3x the strongest reported ...
    'lg_m1h_zs': (-2.0, 2.0),
    # prior: Additive dex-per-ln(1+z) slope on the ZM15 stellar pivot
    # lg_m0star (_Z_EVOL, forward_jax.py:154); +-2 moves the pivot by at most
    # +-1.6 dex at the tier-3 grid edge ...
    'lg_m0star_zs': (-2.0, 2.0),
    # prior: A Gaussian width cannot be negative and sigma_lnmstar+eta*dlgM
    # already flips N_cen's erfc sign when it crosses zero
    # (zumandelbaum15.py:119,155); with fid sigma_lnmstar = ...
    'sigma_lnmstar_zs': (-1.25, 1.25),
    # prior: Dex of L_X per ln(1+z) ON TOP of the hard-coded self-similar
    # E(z)^2 (_lx_kt_of); +-2 is a factor 10^1.6 = 40 departure from self-
    # similarity between z=0.3 and z=1.9 ...
    'lx_zs': (-2.0, 2.0),
    # validity: APEC validity: the band tables are tabulated on log10 T in
    # [-1.301, 1.778] (kT in [0.05, 60] keV, apec_bands.py:31) and
    # _bilinear_logtable CLIPS outside, freezing the ...
    'kt_zs': (-1.0, 0.81),
    # prior: f_ERDF is an active FRACTION so log10 f_ERDF <= 0; with fid
    # agn_log10_ferdf = -1.7243 the slope reaching zero at the tier-3 edge is
    # exactly 1.7243/0.8023 = 2.149, so hi ...
    'agn_log10_ferdf_zs': (-3.0, 2.0),
    # validity: The ERDF break must stay on the integration grid loglam =
    # linspace(-3.0, 1.5, 120) (forward_jax.py:581); with fid agn_log10_lstar
    # = -1.012 the grid edges are reached at ...
    'agn_log10_lstar_zs': (-2.0, 2.0),
    # -- tier 2: X-ray spectral ------------------------------------------------
    # prior: Polytropic-like exponent on the gas density shape, T(r|M) =
    # kT(M)[f_gNFW(x)/f_gNFW(1)]^Gamma_T anchored at x=1; Gamma_T = gamma_poly
    # - 1, so the ...
    't_prof_slope': (-1.0, 1.0),
    # validity: This is the LINEAR metallicity in Z_sun fed straight to
    # jnp.log10 (_gas_log10Z, forward_jax.py:1002), and the shipped APEC
    # tables are built on Z in [0.05, 3.0] Z_sun ...
    'z_gas_norm': (0.05, 3.0),
    # prior: d log10 Z / d log10 M500c about the 1e14 pivot (_gas_log10Z); ICM
    # abundance is observed essentially mass-independent (|slope| < 0.3), and
    # +-1 is >3x that — note the APEC ...
    'z_gas_mslope': (-1.0, 1.0),
    # validity: d log10 Z / d ln(1+z), added in log space inside _gas_log10Z
    # (not routed through _theta_eff); the APEC table headroom around
    # log10(0.3) = -0.523 is -0.778/+1.000 dex ...
    'z_gas_zs': (-0.96, 1.24),
    # prior: Reused verbatim from the existing expert bound in
    # scripts/fitting/fit_xray_joint_bands.py:272 (_BND8 index 8), matching
    # the observed AGN photon-index distribution Gamma ...
    'agn_gamma': (1.0, 3.0),
    # definitional: A population fraction obscured at N_H = 1e22, used as both
    # f_abs and (1-f_abs) (_abs_survival_soft/_bands, and the QLF factor at
    # forward_jax.py:1759); f_abs = 1 is ...
    'agn_fabs': (0.0, 0.95),
    # prior: Dex of the M_BH-M* zero point per ln(1+z) (_Z_EVOL pair with
    # agn_mu_bh); measured offsets are ~0.5 dex out to z=2 (slope ~0.6), and
    # +-2 moves fid mu_BH = 7.518 by at ...
    'agn_mu_bh_zs': (-2.0, 2.0),
    # -- missing physics, waves 1 and 2 ----------------------------------------
    # definitional: A coupling EFFICIENCY of the SN energy budget: E_SN =
    # eps_sn * M*(M_h) * 5e5 (km/s)^2 (forward_jax.py:838), so it cannot
    # exceed the budget it is a fraction of, nor be ...
    'eps_sn': (0.0, 1.0),
    # prior: Standard DESI/Planck CPL prior box for w(a) = w0 + wa(1-a),
    # consumed by _e2z (forward_jax.py:749), comoving_distance and the CPL
    # growth ODE; E^2(z) stays positive ...
    'w0': (-3.0, 1.0),
    # prior: Standard DESI/Planck CPL prior box, same consumers as w0; the
    # only pathological corner is w0 near +1 with wa near +2, where w0+wa > 0
    # makes dark energy dominate at a = ...
    'wa': (-3.0, 2.0),
    # validity: Lower bound 0 is PHYSICAL (a mass sum cannot be negative; the
    # f_nu = 0 row of the table is exactly one so the massless limit is bit-
    # identical). Upper bound 0.5 eV is the ...
    'sum_mnu': (0.0, 0.4),
    # validity: log10 of the ZM16 central-quenching Weibull mass in M_sun/h
    # (f_red_cen_zu16, zumandelbaum15.py:356); it must sit inside the forecast
    # halo-mass grid logspace(log10m_min ...
    'log10_Mq_cen': (9.0, 15.0),
    # definitional: Weibull shape parameter of f_Q = 1 - exp[-(M_h/M_q)^mu]:
    # mu <= 0 inverts the relation so the MOST massive halos become unquenched
    # (verified: mu = -0.2 gives a ...
    'mu_q_cen': (0.0, 5.0),
    # validity: Same argument as log10_Mq_cen for the satellite Weibull
    # (f_red_sat_zu16): outside the forecast halo-mass grid the quenched
    # fraction saturates and the parameter stops ...
    'log10_Mq_sat': (9.0, 15.0),
    # definitional: Weibull shape parameter of satellite quenching, same
    # inversion at mu <= 0 as mu_q_cen. LOUD FLAG: fid 0.24 sits only 0.24
    # sigma of BROAD_PRIOR_SIGMA = 1.0 above a hard ...
    'mu_q_sat': (0.0, 5.0),
    # prior: Additive dex offset on log10 L_X of quenched centrals, lx *=
    # 10^dlx_quenched applied only when sfq='q' (forward_jax.py:989); a pure
    # positive amplitude so nothing breaks ...
    'dlx_quenched': (-2.0, 2.0),
    # prior: Coefficient of log L_X in the Merloni+2003 fundamental plane log
    # L_R = xi_RX log L_X + xi_RM log M_BH + b_R (forward_jax.py:1401-1406);
    # measurements cluster at 0.60 +- ...
    'agn_xi_rx': (0.0, 1.5),
    # prior: Coefficient of log M_BH in the same fundamental plane, measured
    # 0.78 +- 0.11 (Merloni+2003); [0, 1.5] is >6 sigma wide and keeps the
    # plane's sense (radio brightens with ...
    'agn_xi_rm': (0.0, 1.5),
    # prior: Fundamental-plane zero point, log10 of the 5 GHz nuLnu in erg/s
    # extrapolated to log L_X = 0 and log M_BH = 0, measured 7.33 +- 4.05
    # (Merloni+2003, i.e. this box is ...
    'agn_b_r': (0.0, 14.0),
    # definitional: Intrinsic FP scatter in dex, entering as sig_r =
    # sqrt((xi_RX+xi_RM)^2 sig_lm^2 + sigma_R^2) which then DIVIDES the
    # Gaussian kernel (forward_jax.py:1404,1408): it is a ...
    'agn_sig_r': (0.02, 3.0),
    # physical: log10 of the HI mass normalisation in M_sun/h for M_HI = M0
    # (M_h/M_min)^alpha exp[-(M_min/M_h)^0.35] (Villaescusa-Navarro+2018,
    # forward_jax.py:1438-1441); the upper wall ...
    'log10_M0_hi': (8.0, 11.9),
    # validity: log10 of the HI cut-off halo mass, simultaneously the power-
    # law pivot and the exponential suppression scale of the VN18 relation; it
    # must sit inside the forecast mass ...
    'log10_Mmin_hi': (9.0, 14.0),
    # physical: Power-law index of M_HI(M_h) above M_min; the upper wall is
    # the cosmic baryon budget at the top of the mass grid — with fid
    # log10_M0_hi = 10.63 and log10_Mmin_hi = 12.3 ...
    'alpha_hi': (0.0, 1.2),
    # physical: SN wind mass-loading normalisation at V_c = 200 km/s: eta_eff
    # = eta(M)/(1 + eta_w) with eta_w = eta_w0 (V_c/200)^{-alpha_w}
    # (forward_jax.py:796-803). A mass-loading ...
    'eta_w_norm': (0.0, 3.0),
    # prior: Power-law index of the wind loading in halo circular velocity,
    # eta_w ∝ V_c^{-alpha_w}; 1 = momentum-driven, 2 = energy-driven, and
    # Muratov+2015 report up to 3.2 at low ...
    'alpha_w': (0.0, 3.5),
    # prior: log10 of the main-sequence specific SFR in yr^-1 at log10 M* =
    # 10.5 (_mu_ms, forward_jax.py:2030); the upper wall is the shortest
    # plausible star-formation timescale ...
    'ssfr_ms_norm': (-13.0, -7.0),
    # prior: d log10 sSFR / d log10 M* of the main sequence; reported log SFR-
    # log M* slopes span 0.4-1.2, i.e. sSFR slopes -0.6 to +0.2, so this box
    # is ~2.5x wider on each side. It ...
    'ssfr_ms_slope': (-1.5, 0.5),
    # prior: Dex of main-sequence normalisation per ln(1+z) (_Z_EVOL pair with
    # ssfr_ms_norm); sSFR ∝ (1+z)^2.75 corresponds to slope 2.75/ln10 = 1.19,
    # so +-3 is ~2.5x the measured ...
    'ssfr_ms_zs': (-3.0, 3.0),
    # prior: Dex offset of M_HI for quenched centrals, M_HI *= 10^dhi_quenched
    # applied only when sfq='q' (the NeutralUniverseMachine deficit
    # phenomenology); the asymmetry is physical ...
    'dhi_quenched': (-3.0, 1.0),
    # definitional: Lognormal scatter of log sSFR on the main sequence in dex:
    # a Gaussian width, and it DIVIDES in the sSFR-cut survival S_MS = 0.5
    # erfc[(cut-mu)/(sqrt2 sigma_MS)] ...
    'sigma_ms': (0.02, 1.5),
    # definitional: Offset in dex of the quenched population's log sSFR BELOW
    # the main sequence, mu_Q = mu_MS + Delta_Q (forward_jax.py:1489, 2183);
    # Delta_Q >= 0 would put the 'quenched' ...
    'dssfr_q': (-4.0, 0.0),
    # prior: log10 of L_[OII] per unit SFR in erg/s per (M_sun/yr); the
    # fiducial 40.85 is exactly the Kennicutt 1998 calibration (SFR = 1.4e-41
    # L_[OII]), metallicity/dust variants of ...
    'loii_norm': (39.0, 42.5),
    # definitional: Radio-loud (HERG/LERG) fraction of central black holes at
    # M_BH = 1e8 — a probability, and the code only guards the UPPER end via
    # jnp.minimum(f_loud0 * 10^(beta_loud ...
    'f_loud0': (0.0, 1.0),
    # prior: d log10 f_loud / d log10 M_BH; the radio-loud fraction rising
    # steeply with BH mass is universal (Best+2005 find f_loud ∝ M_BH^2.5, the
    # fiducial 2.0), and a negative ...
    'beta_loud': (0.0, 5.0),
    # prior: log10 of the jet 5 GHz nuLnu in erg/s at M_BH = 1e8
    # (forward_jax.py:1421, 1610); observed jetted AGN span ~10^36 (quiescent,
    # Sgr A*-like) to ~10^45 (the brightest FR ...
    'b_jet': (35.0, 45.0),
    # definitional: log10 of the 6 micron-to-bolometric ratio
    # nuLnu(6um)/L_bol: a monochromatic nuLnu cannot exceed the bolometric
    # luminosity it is an integral part of, so the ratio is <= 1 ...
    'agn_bc_ir': (-3.0, 0.0),
    # -- tier 3: radio, IR and extras ------------------------------------------
    # prior: log10 L_nu(1.4 GHz)/SFR feeding _radio_fields
    # (forward_jax.py:1593, l_sf14 = 10**(l14_sfr + log10(1.4e9) - 38)*SFR);
    # fiducial 28.20 reproduces Murphy+2011 (1/6.35e-29 = ...
    'l14_sfr': (26.0, 30.5),
    # physical: Synchrotron index in L_nu ~ nu^-alpha (forward_jax.py:1612, a
    # = 1 - alpha_syn applied as (nu/1.4)^a and (nu/5)^a); the lower edge is
    # the synchrotron self-absorption ...
    'alpha_syn': (-2.5, 3.0),
    # physical: log10 L_IR/SFR [erg/s per Msun/yr] in _ir_fields
    # (forward_jax.py:1627, l_dust = 10**(lir_sfr - 44)*SFR); the ceiling is
    # the energy budget - with 100% dust reprocessing ...
    'lir_sfr': (41.0, 43.7),
    # prior: Dust-SED tilt exponent multiplying the band dust fraction by
    # (4.9um/lambda)^b_IR (forward_jax.py:1634-1636), inert at the 4.9um
    # anchor; over the 3.4-12um WISE/SPHEREx ...
    'bir_color': (-4.0, 4.0),
    # prior: log10 nuLnu(3.4um) per Msun/h (forward_jax.py:1625/1727), the
    # stellar M/L behind the nirlf and the IR-map stellar continuum; the hard
    # ceiling is the stellar bolometric ...
    'ml_nir': (31.0, 35.5),
    # prior: log10 nuLnu(r)/M* of star-forming centrals (forward_jax.py:1738);
    # same stellar-bolometric ceiling argument as ml_nir, +/-2.3 dex about the
    # 33.10 fiducial. Note the optlf ...
    'ml_opt': (31.0, 35.5),
    # prior: Quiescent r-band M/L offset in dex (forward_jax.py:1739, mu_q =
    # mu_sf + dopt_q); observed SF-vs-quiescent r-band M/L differs by ~0.3-0.5
    # dex, so +/-2 dex is ~4x the ...
    'dopt_q': (-2.0, 2.0),
    # physical: log10 nuLnu(1500A)/SFR including mean attenuation
    # (forward_jax.py:1710); the ceiling is the UNATTENUATED stellar output -
    # Kennicutt 1998 gives 43.155 (verified) and ...
    'luv_norm': (39.0, 43.3),
    # prior: UV attenuation slope, subtracted as -tau*(log10 M* - 10.5)
    # (forward_jax.py:1711); observed attenuation-mass relations give 0.2-0.8
    # dex per dex, so +/-2 is ~3x the ...
    'tau_uv_mslope': (-2.0, 2.0),
    # physical: log10 L_Halpha/SFR (forward_jax.py:1719); the ceiling is
    # Case-B recombination with zero dust and zero LyC escape, which is
    # exactly the Kennicutt calibration (41.10 ...
    'lha_norm': (39.0, 41.8),
    # definitional: log10 of the band fraction nuLnu(1450A)/L_bol
    # (forward_jax.py:2313); a single-band monochromatic luminosity cannot
    # reach the bolometric total for any physically broad ...
    'agn_bc_uv': (-2.5, -0.05),
    # definitional: log10 nuLnu(4400A)/L_bol (forward_jax.py:2316); identical
    # band-fraction argument to agn_bc_uv - the ratio is definitionally below
    # 1 so the log is strictly negative ...
    'agn_bc_opt': (-2.5, -0.05),
    # -- wave 4: morphology ----------------------------------------------------
    # validity: Early-type Weibull transition mass, log10(M_morph/[Msun/h])
    # (morphology.py:52). The forecast halo grid is logspace(log10m_min, 16)
    # with log10m_min = 8.5 for the tier-3/4 ...
    'log10_M_morph': (8.5, 14.0),
    # definitional: Weibull shape of f_early = 1 - exp[-(M/M_morph)^beta]
    # (morphology.py:52). beta > 0 is required for a monotone-RISING early-
    # type fraction: verified at beta = -0.5 the ...
    'beta_morph': (0.0, 10.0),
    # definitional: Satellite early-type boost, f_E,s = f_E,c + f_sat(1 -
    # f_E,c) (morphology.py:67). A probability: verified f_sat = -0.3 gives
    # f_E,s = -0.287 at 1e10 (negative early-type ...
    'f_morph_sat': (0.0, 1.0),
    # prior: BH-bulge coupling in dex of M_BH per dex of the f_early bulge
    # proxy (forward_jax.py:1859, mean_bh += slope*log10(f_E,c + 1e-4)); M_BH
    # proportional to M_bulge corresponds ...
    'mbh_bt_slope': (-2.0, 2.0),
    # -- tier 4: morphology observables ----------------------------------------
    # definitional: Correlation of two Bernoulli indicators in f_EQ = f_E f_Q
    # + rho*sqrt(f_E(1-f_E)f_Q(1-f_Q)) (forward_jax.py:2059-2064), so |rho| <=
    # 1 by definition. CRITICAL FLAG: this ...
    'rho_morph_q': (-1.0, 1.0),
    # definitional: log10(R_e/R_200c) (forward_jax.py:2131). The upper bound
    # is definitional: a central galaxy's effective radius cannot exceed its
    # own halo radius, so the log of the ratio ...
    'log10_f_size': (-4.0, 0.0),
    # prior: Early-type size offset in dex, entered as <log10 R_e> +=
    # dsize_early*f_early^cell (forward_jax.py:2132); van der Wel+2014-style
    # early-vs-late size differences are <= 0.5 ...
    'dsize_early': (-1.5, 1.5),
    # prior: Evolution slope of log10_f_size per ln(1+z), applied additively
    # via _theta_eff/_Z_EVOL (forward_jax.py:159). Bounded by the induced-
    # bound rule: over the tier-3/4 cell ...
    'f_size_zs': (-2.5, 2.2),
    # prior: NLA intrinsic-alignment amplitude carried by the early types,
    # A_eff = a_ia*f_early^cell (forward_jax.py:2161); because it is divided
    # out by f_early < 1 it must be ...
    'a_ia': (-10.0, 10.0),
}


# Planck 2018 1σ priors (planck_prior.PLANCK18_SIGMAS) on the free cosmo params.
# Omega_b width is the BBN prior (Cooke+2018), tighter than Planck alone.
PLANCK_PRIOR_SIGMA = {"Omega_m": 0.0073, "sigma8": 0.0060,
                      "h": 0.0054, "n_s": 0.0042, "Omega_b": 0.0005}

# Weakly-informative *regularizing* priors (plausible parameter ranges).  They
# bound the near-flat / degenerate nuisance directions so the Fisher matrix is
# positive-definite and marginalised errors are well-defined and monotonic
# (adding data or fixing a parameter can then only tighten σ, never loosen it).
# Broad enough not to constrain the data-measured parameters.
BROAD_PRIOR_SIGMA = {
    "Omega_m": 0.3, "sigma8": 0.3,                      # ~uninformative on the targets
    # h, n_s, Omega_b are always externally calibrated (Planck/BBN); they are near-flat
    # directions for these low-z small-scale probes, so they carry their Planck/BBN
    # prior by default (differentiable and loosenable, but not spuriously free).
    "h": 0.0054, "n_s": 0.0042, "Omega_b": 0.0005,
    "lg_m1h": 2.0, "lg_m0star": 2.0, "beta": 5.0, "delta": 5.0, "gamma": 5.0,
    "sigma_lnmstar": 2.0, "eta": 5.0, "fc": 1.0, "bsat": 100.0,
    "lx_norm": 3.0, "lx_slope": 2.0, "kt_norm": 2.0, "kt_slope": 2.0,
    "p2": 0.5, "r_max": 3.0, "log10DC": 3.0, "beta_pressure": 2.0,
    "log10_M_pivot": 1.5, "log10_eta_min": 0.7,
    # Powell AGN-XLF sector — Gaussian priors from Powell 2022 / Ananna 2022.
    "agn_mu_bh": 0.30, "agn_al_bh": 0.24, "agn_sig_bh": 0.18,
    "agn_log10_lstar": 0.20, "agn_delta1": 0.15, "agn_delta2": 0.66,
    "agn_log10_ferdf": 1.0,
    # Tier-2 promoted nuisances — plausible-range widths (regularizing only).
    "beta_sat": 1.0, "bcut": 5.0, "beta_cut": 1.0, "alpha_sat": 0.5,
    "beta_b": 1.0, "log10_M_eta": 1.0, "beta_eta": 1.0,
    "alpha_in_gas": 0.5, "alpha_tr_gas": 1.0,
    "p03_pressure": 1.627e-6, "c_dpm_pressure": 1.4, "alpha_in_pressure": 0.3,
    "alpha_tr_pressure": 0.6, "alpha_out_pressure": 2.0,
    "agn_rho": 0.5, "agn_sig_mstar": 0.15,
    # Tier-2 z-evolution slopes — broad (an O(1) change per e-fold of 1+z).
    "lg_m1h_zs": 2.0, "lg_m0star_zs": 2.0, "sigma_lnmstar_zs": 2.0,
    "lx_zs": 3.0, "kt_zs": 2.0,
    "agn_log10_ferdf_zs": 3.0, "agn_log10_lstar_zs": 2.0,
    # Tier-2 X-ray spectral sector.
    "t_prof_slope": 0.5, "z_gas_norm": 0.3, "z_gas_mslope": 0.5, "z_gas_zs": 2.0,
    "agn_gamma": 0.3, "agn_fabs": 0.3, "agn_mu_bh_zs": 2.0,
    # Missing-physics extension.
    "eps_sn": 0.3,
    "w0": 1.0, "wa": 2.0, "sum_mnu": 0.25,
    "log10_Mq_cen": 1.0, "mu_q_cen": 1.0, "log10_Mq_sat": 1.0, "mu_q_sat": 1.0,
    "dlx_quenched": 1.0,
    "agn_xi_rx": 0.3, "agn_xi_rm": 0.3, "agn_b_r": 2.0, "agn_sig_r": 0.3,
    "log10_M0_hi": 1.0, "log10_Mmin_hi": 1.0, "alpha_hi": 0.3,
    # wave 2
    "eta_w_norm": 0.5, "alpha_w": 0.7,
    "ssfr_ms_norm": 0.5, "ssfr_ms_slope": 0.3, "ssfr_ms_zs": 2.0,
    "dhi_quenched": 1.0,
    # wave 3
    "sigma_ms": 0.15, "dssfr_q": 0.5, "loii_norm": 0.3,
    "f_loud0": 0.05, "beta_loud": 1.5, "b_jet": 2.0,
    "agn_bc_ir": 0.3,
    # tier 3
    "l14_sfr": 0.2, "alpha_syn": 0.15, "lir_sfr": 0.2, "bir_color": 0.5,
    "ml_nir": 0.3, "ml_opt": 0.3, "dopt_q": 0.3,
    "luv_norm": 0.3, "tau_uv_mslope": 0.3, "lha_norm": 0.2,
    "agn_bc_uv": 0.3, "agn_bc_opt": 0.3,
    # wave 4
    "log10_M_morph": 1.0, "beta_morph": 0.5,
    "f_morph_sat": 0.3, "mbh_bt_slope": 0.5,
    # tier 4
    "rho_morph_q": 0.3, "log10_f_size": 0.3, "dsize_early": 0.3,
    "f_size_zs": 2.0, "a_ia": 1.0,
}

PARAM_LATEX = {
    "Omega_m": r"$\Omega_m$", "sigma8": r"$\sigma_8$",
    "h": r"$h$", "n_s": r"$n_s$", "Omega_b": r"$\Omega_b$",
    "lg_m1h": r"$\lg M_{1}$", "lg_m0star": r"$\lg M_{0*}$",
    "beta": r"$\beta$", "delta": r"$\delta$", "gamma": r"$\gamma$",
    "sigma_lnmstar": r"$\sigma_{\ln M_*}$", "eta": r"$\eta$",
    "fc": r"$f_c$", "bsat": r"$b_{\rm sat}$",
    "lx_norm": r"$L_X^{\rm norm}$", "lx_slope": r"$L_X^{\rm slope}$",
    "kt_norm": r"$kT^{\rm norm}$", "kt_slope": r"$kT^{\rm slope}$",
    "p2": r"$p_2$", "r_max": r"$r_{\rm max}$",
    "log10DC": r"$\log_{10}{\rm DC}$", "beta_pressure": r"$\beta_P$",
    "log10_M_pivot": r"$\log_{10}M_{\rm pivot}$", "log10_eta_min": r"$\log_{10}\eta_{\min}$",
    "agn_mu_bh": r"$\mu_{\rm BH}$", "agn_al_bh": r"$\alpha_{\rm BH}$",
    "agn_sig_bh": r"$\sigma_{\rm BH}$", "agn_log10_lstar": r"$\log_{10}\lambda_*$",
    "agn_delta1": r"$\delta_1$", "agn_delta2": r"$\delta_2$",
    "agn_log10_ferdf": r"$\log_{10}f_{\rm ERDF}$",
    # Tier-2 promoted nuisances.
    "beta_sat": r"$\beta_{\rm sat}$", "bcut": r"$b_{\rm cut}$",
    "beta_cut": r"$\beta_{\rm cut}$", "alpha_sat": r"$\alpha_{\rm sat}$",
    "beta_b": r"$\beta_b$", "log10_M_eta": r"$\log_{10}M_\eta$",
    "beta_eta": r"$\beta_\eta$",
    "alpha_in_gas": r"$\alpha_{\rm in}$", "alpha_tr_gas": r"$\alpha_{\rm tr}$",
    "p03_pressure": r"$P_{0.3}$", "c_dpm_pressure": r"$c_{\rm DPM}$",
    "alpha_in_pressure": r"$\alpha_{\rm in}^P$", "alpha_tr_pressure": r"$\alpha_{\rm tr}^P$",
    "alpha_out_pressure": r"$\alpha_{\rm out}^P$",
    "agn_rho": r"$\rho_{\rm BH}$", "agn_sig_mstar": r"$\sigma_{M_*}^{\rm BH}$",
    # Tier-2 z-evolution slopes.
    "lg_m1h_zs": r"$\partial_z\lg M_1$", "lg_m0star_zs": r"$\partial_z\lg M_{0*}$",
    "sigma_lnmstar_zs": r"$\partial_z\sigma_{\ln M_*}$",
    "lx_zs": r"$\partial_z L_X^{\rm norm}$", "kt_zs": r"$\partial_z kT^{\rm norm}$",
    "agn_log10_ferdf_zs": r"$\partial_z\log_{10}f_{\rm ERDF}$",
    "agn_log10_lstar_zs": r"$\partial_z\log_{10}\lambda_*$",
    # Tier-2 X-ray spectral sector.
    "t_prof_slope": r"$\Gamma_T$", "z_gas_norm": r"$Z_{\rm gas}$",
    "z_gas_mslope": r"$\partial_M Z$", "z_gas_zs": r"$\partial_z Z$",
    "agn_gamma": r"$\Gamma_{\rm AGN}$", "agn_fabs": r"$f_{\rm abs}$",
    "agn_mu_bh_zs": r"$\partial_z\mu_{\rm BH}$",
    # Missing-physics extension.
    "eps_sn": r"$\varepsilon_{\rm SN}$",
    "w0": r"$w_0$", "wa": r"$w_a$", "sum_mnu": r"$\Sigma m_\nu$",
    "log10_Mq_cen": r"$\log_{10}M_{\rm q}^{\rm cen}$", "mu_q_cen": r"$\mu_{\rm q}^{\rm cen}$",
    "log10_Mq_sat": r"$\log_{10}M_{\rm q}^{\rm sat}$", "mu_q_sat": r"$\mu_{\rm q}^{\rm sat}$",
    "dlx_quenched": r"$\Delta L_X^{\rm Q}$",
    "agn_xi_rx": r"$\xi_{RX}$", "agn_xi_rm": r"$\xi_{RM}$",
    "agn_b_r": r"$b_R$", "agn_sig_r": r"$\sigma_R$",
    "log10_M0_hi": r"$\log_{10}M_0^{\rm HI}$",
    "log10_Mmin_hi": r"$\log_{10}M_{\min}^{\rm HI}$", "alpha_hi": r"$\alpha_{\rm HI}$",
    # wave 2
    "eta_w_norm": r"$\eta_{w,0}$", "alpha_w": r"$\alpha_w$",
    "ssfr_ms_norm": r"${\rm sSFR}_{\rm MS}$", "ssfr_ms_slope": r"$\partial_M{\rm sSFR}$",
    "ssfr_ms_zs": r"$\partial_z{\rm sSFR}$", "dhi_quenched": r"$\Delta M_{\rm HI}^{\rm Q}$",
    # wave 3
    "sigma_ms": r"$\sigma_{\rm MS}$", "dssfr_q": r"$\Delta_{\rm Q}$",
    "loii_norm": r"$\log_{10}L_{\rm [OII]}/{\rm SFR}$",
    "f_loud0": r"$f_{\rm loud,0}$", "beta_loud": r"$\beta_{\rm loud}$",
    "b_jet": r"$b_{\rm jet}$", "agn_bc_ir": r"${\rm BC}_{\rm IR}$",
    # tier 3
    "l14_sfr": r"$\log_{10}L_{1.4}/{\rm SFR}$", "alpha_syn": r"$\alpha_{\rm syn}$",
    "lir_sfr": r"$\log_{10}L_{\rm IR}/{\rm SFR}$", "bir_color": r"$b_{\rm IR}$",
    "ml_nir": r"$\Upsilon_{\rm NIR}$", "ml_opt": r"$\Upsilon_{\rm opt}$",
    "dopt_q": r"$\Delta_{\rm opt}^{\rm Q}$",
    "luv_norm": r"$\log_{10}L_{\rm UV}/{\rm SFR}$",
    "tau_uv_mslope": r"$\partial_M\tau_{\rm UV}$",
    "lha_norm": r"$\log_{10}L_{{\rm H}\alpha}/{\rm SFR}$",
    "agn_bc_uv": r"${\rm BC}_{\rm UV}$", "agn_bc_opt": r"${\rm BC}_{\rm opt}$",
    # wave 4
    "log10_M_morph": r"$\log_{10}M_{\rm morph}$",
    "beta_morph": r"$\beta_{\rm morph}$",
    "f_morph_sat": r"$f_{\rm morph}^{\rm sat}$",
    "mbh_bt_slope": r"$\partial_{B/T}M_{\rm BH}$",
    # tier 4
    "rho_morph_q": r"$\rho_{\rm morph,Q}$",
    "log10_f_size": r"$\log_{10}R_e/R_{200}$",
    "dsize_early": r"$\Delta_{\rm size}^{\rm E}$",
    "f_size_zs": r"$\partial_z R_e/R_{200}$",
    "a_ia": r"$A_{\rm IA}$",
}

# Parameter sectors for the tier-2 cosmology-vs-astrophysics decomposition
# (run_tier2_forecast): σ(sector) marginalised vs other-sectors-pinned, probe
# attribution, and information gain per sector.  Covers all of PARAM_NAMES.
SECTORS = {
    "cosmology": ["Omega_m", "sigma8", "h", "n_s", "Omega_b",
                  "w0", "wa", "sum_mnu"],
    "shmr": ["lg_m1h", "lg_m0star", "beta", "delta", "gamma", "sigma_lnmstar",
             "eta", "fc", "bsat", "beta_sat", "bcut", "beta_cut", "alpha_sat",
             "lg_m1h_zs", "lg_m0star_zs", "sigma_lnmstar_zs"],
    "gas": ["lx_norm", "lx_slope", "kt_norm", "kt_slope", "p2", "r_max",
            "alpha_in_gas", "alpha_tr_gas", "beta_pressure", "p03_pressure",
            "c_dpm_pressure", "alpha_in_pressure", "alpha_tr_pressure",
            "alpha_out_pressure", "lx_zs", "kt_zs",
            "t_prof_slope", "z_gas_norm", "z_gas_mslope", "z_gas_zs"],
    "baryon": ["log10_M_pivot", "log10_eta_min", "beta_b", "log10_M_eta",
               "beta_eta", "eps_sn", "eta_w_norm", "alpha_w"],
    "agn": ["agn_mu_bh", "agn_al_bh", "agn_sig_bh", "agn_log10_lstar",
            "agn_delta1", "agn_delta2", "agn_log10_ferdf", "agn_rho",
            "agn_sig_mstar", "agn_log10_ferdf_zs", "agn_log10_lstar_zs",
            "agn_gamma", "agn_fabs", "agn_mu_bh_zs",
            "agn_xi_rx", "agn_xi_rm", "agn_b_r", "agn_sig_r",
            "f_loud0", "beta_loud", "b_jet", "agn_bc_ir",
            "agn_bc_uv", "agn_bc_opt"],
    # SF/quiescent split (ZM16 halo quenching, the quenched-CGM offset, and
    # the star-forming main sequence)
    "sfq": ["log10_Mq_cen", "mu_q_cen", "log10_Mq_sat", "mu_q_sat",
            "dlx_quenched", "ssfr_ms_norm", "ssfr_ms_slope", "ssfr_ms_zs",
            "sigma_ms", "dssfr_q", "loii_norm",
            # tier 3: SED calibrations tied to SFR / M* (radio, IR, UV, opt, Halpha)
            "l14_sfr", "alpha_syn", "lir_sfr", "bir_color",
            "ml_nir", "ml_opt", "dopt_q",
            "luv_norm", "tau_uv_mslope", "lha_norm"],
    # cold gas / neutral hydrogen (VN18 halo model + quenched deficit)
    "coldgas": ["log10_M0_hi", "log10_Mmin_hi", "alpha_hi", "dhi_quenched"],
    # galaxy morphology (wave 4): early-type Weibull + the BH-bulge link
    "morphology": ["log10_M_morph", "beta_morph", "f_morph_sat",
                   "mbh_bt_slope",
                   # tier 4: joint E/Q correlation, sizes, IA
                   "rho_morph_q", "log10_f_size", "dsize_early",
                   "f_size_zs", "a_ia"],
    # retired: with agn_emission="powell" the duty cycle leaves the emissivity;
    # its only remaining consumer is the (off-by-default) energy-closure mode.
    "retired": ["log10DC"],
}


def _results_root() -> str:
    try:
        from hod_mod import paths
        return os.fspath(paths.results_root())
    except Exception:
        return os.environ.get("HOD_MOD_RESULTS", "")


_WARNED_NATIVE_BANDS = False


def _warn_native_band_summary(missing):
    """Warn once that the band summary cannot feed the forecast fiducial."""
    global _WARNED_NATIVE_BANDS
    if _WARNED_NATIVE_BANDS:
        return
    _WARNED_NATIVE_BANDS = True
    import warnings
    warnings.warn(
        "xray_joint_bands/S1_bands_summary.json does not carry "
        f"{missing} -- it is a native-DPM band fit, whose gas parameters are "
        "(log10_ne03, beta_n, log10_p03, beta_P). The forecast's gas sector is "
        "still the phenomenological power law, so the whole X-ray block is left "
        "at _FIDUCIAL_DEFAULT rather than mixing the two parametrisations. To "
        "use the native fit, project its MAP through "
        "hod_mod.fitting.dpm_priors.scaling_map and write the resulting "
        "(lx_norm, lx_slope, kt_norm, kt_slope) into the summary.",
        RuntimeWarning, stacklevel=3)


def load_fiducial() -> dict:
    """Return the fiducial parameter dict, preferring the on-disk MAP JSONs."""
    fid = dict(_FIDUCIAL_DEFAULT)
    root = _results_root()
    # HOD MAP
    try:
        with open(os.path.join(root, "bgs_zm15_joint", "map_result.json")) as fh:
            p = json.load(fh)["params"]
        for n in ("lg_m1h", "lg_m0star", "beta", "delta", "gamma",
                  "sigma_lnmstar", "eta", "fc", "bsat"):
            if n in p:
                fid[n] = float(p[n])
    except Exception:
        pass
    # X-ray MAP.
    #
    # The forecast's gas sector is still the phenomenological L_X-M / kT-M power
    # law (lx_norm, lx_slope, kt_norm, kt_slope; see forward_jax), but the band
    # fit was re-based onto native DPM parameters (log10_ne03, beta_n, log10_p03,
    # beta_P).  A native summary therefore satisfies none of the four amplitude
    # names while STILL matching p2/r_max/log10DC -- so a per-name loop would
    # quietly take the shape from the new fit and the amplitude from
    # _FIDUCIAL_DEFAULT, i.e. from a fit that no longer exists.  That mix is
    # worse than either source alone and is invisible downstream, so take the
    # gas block all-or-nothing.
    _XRAY_KEYS = ("lx_norm", "lx_slope", "kt_norm", "kt_slope", "p2", "r_max", "log10DC")
    try:
        with open(os.path.join(root, "xray_joint_bands", "S1_bands_summary.json")) as fh:
            summ = json.load(fh)
        m = summ["map"]
    except Exception:
        summ, m = None, None
    if m is not None:
        # A native-DPM summary carries the four scaling-relation values in
        # 'map_scaling', projected by fit_xray_joint_bands.scaling_from_native.
        # Prefer 'map' (an old phenomenological fit), then that projection.
        src = dict(m)
        proj = (summ or {}).get("map_scaling") or {}
        for n in ("lx_norm", "lx_slope", "kt_norm", "kt_slope"):
            if n not in src and n in proj:
                src[n] = proj[n]
        if all(n in src for n in _XRAY_KEYS):
            for n in _XRAY_KEYS:
                fid[n] = float(src[n])
        else:
            _warn_native_band_summary(sorted(set(_XRAY_KEYS) - set(src)))
    return fid


def fiducial_vector() -> np.ndarray:
    """Fiducial values as a flat vector in :data:`PARAM_NAMES` order."""
    fid = load_fiducial()
    return np.array([fid[n] for n in PARAM_NAMES], dtype=float)


def prior_sigma_vector() -> np.ndarray:
    """1σ Gaussian-prior widths per parameter (inf = no prior). Planck on Ω_m, σ8."""
    s = np.full(N_PARAM, np.inf)
    for n, val in PLANCK_PRIOR_SIGMA.items():
        s[_IDX[n]] = val
    return s


# ---------------------------------------------------------------------------
# Bounds: vectors, validation, enforcement
# ---------------------------------------------------------------------------

#: Fiducials that deliberately sit ON a bound, with the reason.  These are
#: modelling choices, not table errors, but they have a consequence worth
#: knowing: a Fisher derivative at such a point is one-sided, and an MCMC
#: walker initialised there must be nudged inward.
BOUNDARY_FIDUCIALS = {
    "eta_w_norm": "the wave-2 wind sector is off by default, and a mass-loading "
                  "normalisation cannot be negative, so the fiducial 0.0 is the "
                  "physical wall itself",
}


def bounds_vector() -> np.ndarray:
    """Strict bounds as an ``(N_PARAM, 2)`` array in :data:`PARAM_NAMES` order."""
    return np.array([PARAM_BOUNDS[n] for n in PARAM_NAMES], dtype=float)


def bounds_pairs() -> list:
    """Bounds as a list of ``(lo, hi)`` tuples, the form ``scipy.optimize`` takes.

    ``None`` replaces a sentinel, which is what the SciPy minimisers expect for
    a genuinely one-sided parameter.
    """
    out = []
    for n in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[n]
        out.append((None if lo <= -_INF else lo, None if hi >= _INF else hi))
    return out


def bounds_violations(theta, atol: float = 0.0) -> list:
    """Names and values of parameters outside their bounds; empty when clean.

    Parameters
    ----------
    theta : array_like or dict
        A full parameter vector, or a dict keyed by parameter name (in which
        case only the keys present are checked).
    atol : float
        Tolerance, for callers that only care about meaningful excursions.
    """
    if isinstance(theta, dict):
        items = [(n, float(v)) for n, v in theta.items() if n in PARAM_BOUNDS]
    else:
        v = np.asarray(theta, dtype=float)
        if v.shape[-1] != N_PARAM:
            raise ValueError(
                f"expected {N_PARAM} parameters, got {v.shape[-1]}")
        items = list(zip(PARAM_NAMES, v))
    bad = []
    for n, val in items:
        lo, hi = PARAM_BOUNDS[n]
        if not np.isfinite(val) or val < lo - atol or val > hi + atol:
            bad.append((n, val, lo, hi))
    return bad


def clip_to_bounds(theta) -> np.ndarray:
    """Clip a parameter vector into its bounds. Does not warn -- callers decide."""
    b = bounds_vector()
    return np.clip(np.asarray(theta, dtype=float), b[:, 0], b[:, 1])


def log_prior_bounds(theta) -> float:
    """``0.0`` inside every bound, ``-inf`` outside.

    The piece a Gaussian precision vector cannot express. Add it to any
    log-posterior that samples the forecast vector; without it a sampler will
    happily walk to a negative neutrino mass or a fraction above one, and the
    forward model will return a number rather than complain.
    """
    v = np.asarray(theta, dtype=float)
    b = bounds_vector()
    inside = np.all(np.isfinite(v)) and np.all(v >= b[:, 0]) and np.all(v <= b[:, 1])
    return 0.0 if inside else -np.inf


def perturb_inside_bounds(theta, scale, rng, max_tries: int = 20) -> np.ndarray:
    """Gaussian perturbation of ``theta`` that lands strictly inside the bounds.

    Used to seed walkers. Resamples rather than clipping, because a clipped
    ensemble piles walkers exactly on the wall and an affine-invariant sampler
    never recovers from a degenerate initial simplex.
    """
    v = np.asarray(theta, dtype=float)
    b = bounds_vector()
    out = np.array(v)
    for i in range(len(v)):
        lo, hi = b[i]
        s = float(np.atleast_1d(scale)[i % np.size(scale)])
        for _ in range(max_tries):
            x = v[i] + s * rng.standard_normal()
            if lo < x < hi:
                out[i] = x
                break
        else:                       # pathological: fall back to the interior
            out[i] = np.clip(v[i] + 1e-3 * (hi - lo), lo, hi)
    return out


def regularizing_prior(add_planck: bool = False, fix: tuple = ()) -> np.ndarray:
    """Weakly-informative prior vector for stable marginalisation.

    Uses :data:`BROAD_PRIOR_SIGMA` for every parameter; applies the tight Planck
    prior on Ω_m, σ8 when ``add_planck``; and pins parameters named in ``fix`` to
    a delta-function (σ = 1e-4) — used to compare "baryons fixed" vs "marginalised".
    """
    s = np.array([BROAD_PRIOR_SIGMA[n] for n in PARAM_NAMES], dtype=float)
    if add_planck:
        for n, val in PLANCK_PRIOR_SIGMA.items():
            s[_IDX[n]] = val
    for n in fix:
        s[_IDX[n]] = 1e-4
    return s


def latex_labels() -> list[str]:
    return [PARAM_LATEX[n] for n in PARAM_NAMES]
