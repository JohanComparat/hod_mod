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

Parameter vector (see :data:`PARAM_NAMES`), 19 entries::

    Omega_m sigma8 | lg_m1h lg_m0star beta delta gamma sigma_lnmstar eta fc bsat
                   | lx_norm lx_slope kt_norm kt_slope p2 r_max | log10DC beta_pressure
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
)
from hod_mod.core.halo_profiles import nfw_uk_jax, concentration_dutton14_jax
from hod_mod.connection.hod.zumandelbaum15 import (
    _mh_from_mstar_zu15,
    _mstar_from_mh_zu15,
    sigma_lnmstar_zu15,
)
from jax import custom_jvp
from jax.scipy.special import erfc
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
]
_IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
N_PARAM = len(PARAM_NAMES)

OBSERVABLES = ["wp", "ds", "cl_gX", "cl_gy", "cl_XX", "cl_kk",
               "cl_kCMB", "cl_gkCMB", "cl_shear_kCMB", "xlf", "n_gal", "smf"]

_C_OVER_H0 = 2997.92   # c/H0 [Mpc/h]  (H0 = 100 h km/s/Mpc)

# Fixed (non-varied) cosmology + HOD nuisance held at Planck18 / ZM15-MAP values.
_FIXED_COSMO = {"h": 0.6736, "Omega_b": 0.0493, "n_s": 0.9649}
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
        z_src_mean: float = 0.8,
        z_src_sig: float = 0.30,
        n_z_shear: int = 12,
        energy_closure: bool = False,
        log10m_star_thresh: float = None,
        baryon_model: str = "sigmoid",
    ):
        self.z_eff = float(z_eff)
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
        self._pk = EisensteinHu98PkLinear()
        self._hmf = HaloMassFunction(self._pk.as_hmf_pk_func(), model="tinker08", Delta=200.0)

        self.k = jnp.logspace(-4.0, jnp.log10(200.0), n_k)
        self.log_k = jnp.log(self.k)
        self.m = jnp.logspace(10.0, 16.0, n_m)
        self.log10m = jnp.log10(self.m)
        self.gx, self.gw = _gl_nodes(n_gl)

        self.rp_wp = jnp.asarray(rp_wp if rp_wp is not None else np.logspace(-1.3, 1.7, 24))
        self.rp_ds = jnp.asarray(rp_ds if rp_ds is not None else np.logspace(-1.3, 1.3, 20))
        self.ell = jnp.asarray(ell if ell is not None else np.logspace(1.0, 3.5, 24))

        # narrow n(z) window for the Limber projection
        zc = self.z_eff
        zg = np.linspace(max(1e-3, zc - 4 * nz_sig), zc + 4 * nz_sig, n_z)
        nz = np.exp(-0.5 * ((zg - zc) / nz_sig) ** 2)
        self.z_grid = jnp.asarray(zg)
        self.nz = jnp.asarray(nz / np.trapezoid(nz, zg))
        self.pi_max = 100.0

        # Cosmic-shear source distribution + lens-distance integration grid.
        zs = np.linspace(0.03, max(2.0, z_src_mean + 3 * z_src_sig), n_z_shear)
        ns = np.exp(-0.5 * ((zs - z_src_mean) / z_src_sig) ** 2)
        self.z_shear = jnp.asarray(zs)
        self.nz_src = jnp.asarray(ns / np.trapezoid(ns, zs))

        # Powell-XLF grids: Eddington-ratio (log λ) integration grid over the
        # Ananna 2022 range, and the log10 L_X [erg/s] abscissa of the observable.
        loglam = np.linspace(-3.0, 1.5, 120)
        self.loglam = jnp.asarray(loglam)
        self.dlam = float(loglam[1] - loglam[0])
        self.loglx_xlf = jnp.asarray(np.linspace(42.0, 45.0, 7))

        # Stellar-mass-function abscissa (log10 M* [Msun/h]) for the SMF observable.
        self.logmstar_smf = jnp.asarray(np.linspace(10.0, 11.6, 9))

        # CMB lensing: last-scattering redshift + a line-of-sight integration grid
        # for the κ_CMB auto spectrum (the kernel peaks at z~2; a grid to z~6
        # captures the bulk — the high-z tail is negligible for ℓ<3000).
        self.z_star = 1089.0
        self.z_cmb = jnp.asarray(np.geomspace(0.03, 6.0, 14))

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
        }

    def _hod(self, theta):
        d = dict(_FIXED_HOD)
        d["log10m_star_thresh"] = self._thr
        for n in ("lg_m1h", "lg_m0star", "beta", "delta", "gamma",
                  "sigma_lnmstar", "eta", "fc", "bsat"):
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
                    "beta_hi": _FIXED_BARYON["beta_b"],
                    "f_b_lo_amp": _UPTURN["f_b_lo_amp"],
                    "log10_M_lo": _UPTURN["log10_M_lo"],
                    "beta_lo": _UPTURN["beta_lo"]}
            fb = self._fb_model(self.m, c, pars)                        # (NM,)
        else:
            pars = {"log10_M_pivot": theta[_IDX["log10_M_pivot"]],
                    "beta_b": _FIXED_BARYON["beta_b"], "f_b_min": _FIXED_BARYON["f_b_min"]}
            fb = self._fb_model(self.m, c, pars)                        # (NM,)
        eta_min = 10.0 ** theta[_IDX["log10_eta_min"]]
        M_eta = 10.0 ** _FIXED_BARYON["log10_M_eta"]
        eta = 1.0 - (1.0 - eta_min) / (1.0 + (self.m / M_eta) ** _FIXED_BARYON["beta_eta"])
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
        ez2 = Om * (1.0 + z) ** 3 + (1.0 - Om)
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
        E_sn = _EPS_SN * 10.0 ** log10_mstar * _E_SN_PER_MSUN

        expelled = jnp.minimum(fbc - _FIXED_BARYON["f_b_min"],
                               (E_agn + E_sn) / (self.m * v200sq))
        return fbc - jnp.maximum(expelled, 0.0)

    # ---- shared halo-model quantities on the mass grid ---------------
    def _halo_common(self, theta, z):
        c = self._cosmo(theta)
        Om = c["Omega_m"]
        # 200c halo radius (comoving h-units) and NFW FT
        ez2 = Om * (1.0 + z) ** 3 + (1.0 - Om)
        rho_crit_com = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3
        conc = concentration_dutton14_jax(self.m, float(z))            # (NM,)
        r_delta = (3.0 * self.m / (4.0 * jnp.pi * 200.0 * rho_crit_com)) ** (1.0 / 3.0)
        r_s = r_delta / conc
        uk = nfw_uk_jax(self.k, r_s, conc)                             # (Nk, NM)

        dndm = self._hmf.dndm(self.m, float(z), c)                     # (NM,)
        bias = self._hmf.bias(self.m, float(z), c)                     # (NM,)

        hp = self._hod(theta)
        thr = hp["log10m_star_thresh"]
        nc = _n_cen(self.log10m, thr, hp["lg_m1h"], hp["lg_m0star"], hp["beta"],
                    hp["delta"], hp["gamma"], hp["sigma_lnmstar"], hp["eta"], hp["fc"])
        ns = _n_sat(self.log10m, thr, hp["lg_m1h"], hp["lg_m0star"], hp["beta"],
                    hp["delta"], hp["gamma"], hp["sigma_lnmstar"], hp["eta"], hp["fc"],
                    hp["bsat"], hp["beta_sat"], hp["bcut"], hp["beta_cut"], hp["alpha_sat"])
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
        alpha_out = _ALPHA_IN_GAS + 2.0 * p2
        r_max = theta[_IDX["r_max"]] * H["r_delta"]
        r_s = H["r_s"] / H["eta"]
        fn = _gnfw(r_max[:, None] * self.gx[None, :] / r_s[:, None],
                   _ALPHA_IN_GAS, _ALPHA_TR_GAS, alpha_out)
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
        alpha_out = _ALPHA_IN_GAS + 2.0 * p2
        r_max = r_max_fac * H["r_delta"]                                # (NM,)
        # gas extent shared with the ΔΣ split: puffier gas (η<1) → larger r_s
        r_s = H["r_s"] / H["eta"]                                       # (NM,)

        def f_nodes(r):                                                 # r: (NM, Ngl)
            return _gnfw_sq(r / r_s[:, None], _ALPHA_IN_GAS, _ALPHA_TR_GAS, alpha_out)

        u_shape = _profile_uk_normalized(self.k, r_max, f_nodes(r_max[:, None] * self.gx[None, :]),
                                         self.gx, self.gw)              # (Nk, NM)
        # analytic L_X(M500c) amplitude and weak kT band weight.  The absolute
        # X-ray amplitude is arbitrary for a relative-error Fisher (log-derivs),
        # so lx_norm is referenced to 45 to keep values O(1) and float-safe while
        # its derivative (d ln X / d lx_norm = ln10) is preserved exactly.
        Om = self._cosmo(theta)["Omega_m"]
        ez = jnp.sqrt(Om * (1.0 + H["z"]) ** 3 + (1.0 - Om))
        log10_m500c = self.log10m + jnp.log10(0.72)                     # M500c ≈ 0.72 M200 (comoving)
        lx = 10.0 ** (theta[_IDX["lx_norm"]] - 45.0
                      + theta[_IDX["lx_slope"]] * (log10_m500c - _LX_PIVOT)) * ez ** 2
        kT = 10.0 ** (theta[_IDX["kt_slope"]] * (log10_m500c - _KT_PIVOT)
                      + theta[_IDX["kt_norm"]]) * ez ** (2.0 / 3.0)
        w_band = jnp.maximum(kT, 1e-3) ** 0.25                          # weak band-response weight
        # n_e² emissivity ∝ (gas density)² ∝ f_b(M)²  → shared baryon sector
        amp = lx * w_band * H["fb"] ** 2                               # (NM,), O(1)
        return u_shape * amp[None, :]

    def _pressure_uk(self, theta, H):
        """Normalised A10 GNFW pressure FT × P500(M) × mass tilt, (Nk, NM)."""
        r_max = 3.0 * H["r_delta"]
        r_s = (H["r_delta"] / _A10["c500"]) / H["eta"]                  # shared gas extent

        def f_nodes(r):
            return _gnfw_pressure(r / r_s[:, None], _A10["gamma"], _A10["alpha"], _A10["beta"])

        u_shape = _profile_uk_normalized(self.k, r_max, f_nodes(r_max[:, None] * self.gx[None, :]),
                                         self.gx, self.gw)
        Om = self._cosmo(theta)["Omega_m"]
        ez = jnp.sqrt(Om * (1.0 + H["z"]) ** 3 + (1.0 - Om))
        log10_m500c = self.log10m + jnp.log10(0.72)
        beta_p = 2.0 / 3.0 + 0.12 + theta[_IDX["beta_pressure"]]        # A10 self-similar + tilt
        # thermal energy ∝ gas mass ∝ f_b(M)  → shared baryon sector
        P500 = 10.0 ** (beta_p * (log10_m500c - 14.5)) * ez ** (8.0 / 3.0) * H["fb"]
        return u_shape * P500[None, :]

    def _pk_gX(self, theta, H):
        """galaxy × X-ray (gas + AGN), 1h+2h."""
        X = self._emissivity_uk(theta, H)                              # (Nk, NM)
        dndm, uk, nc, ns = H["dndm"], H["uk"], H["nc"], H["ns"]
        m, n_gal, b_eff, pk_lin, bias = self.m, H["n_gal"], H["b_eff"], H["pk_lin"], H["bias"]
        gal = nc[None, :] + ns[None, :] * uk
        P_1h = jnp.trapezoid(dndm[None, :] * gal * X, m, axis=1) / n_gal
        I_X = jnp.trapezoid(dndm[None, :] * bias[None, :] * X, m, axis=1)
        P_gas = P_1h + b_eff * pk_lin * I_X
        # AGN point-source term scaled by duty cycle; L_AGN(M) ∝ M (flat in k)
        dc = 10.0 ** theta[_IDX["log10DC"]]
        l_agn = (self.m / 1e13) * 1.0e-1                               # O(1) surrogate amplitude
        Xa = dc * l_agn                                               # (NM,)
        P_agn = jnp.trapezoid(dndm[None, :] * gal * Xa[None, :], m, axis=1) / n_gal \
            + b_eff * pk_lin * jnp.trapezoid(dndm[None, :] * bias[None, :] * Xa[None, :], m, axis=1)
        return P_gas + P_agn

    def _pk_gy(self, theta, H):
        Y = self._pressure_uk(theta, H)
        dndm, uk, nc, ns = H["dndm"], H["uk"], H["nc"], H["ns"]
        m, n_gal, b_eff, pk_lin, bias = self.m, H["n_gal"], H["b_eff"], H["pk_lin"], H["bias"]
        gal = nc[None, :] + ns[None, :] * uk
        P_1h = jnp.trapezoid(dndm[None, :] * gal * Y, m, axis=1) / n_gal
        I_Y = jnp.trapezoid(dndm[None, :] * bias[None, :] * Y, m, axis=1)
        return P_1h + b_eff * pk_lin * I_Y

    def _pk_XX(self, theta, H):
        X = self._emissivity_uk(theta, H)
        dndm, uk, nc, ns = H["dndm"], H["uk"], H["nc"], H["ns"]
        m, b_eff, pk_lin, bias = self.m, H["b_eff"], H["pk_lin"], H["bias"]
        dc = 10.0 ** theta[_IDX["log10DC"]]
        l_agn = (self.m / 1e13) * 1.0e-1
        Xtot = X + dc * l_agn[None, :]                                 # gas + AGN emission per halo
        P_1h = jnp.trapezoid(dndm[None, :] * Xtot ** 2, m, axis=1)
        I = jnp.trapezoid(dndm[None, :] * bias[None, :] * Xtot, m, axis=1)
        P_2h = pk_lin * I ** 2
        return P_1h + P_2h

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

    def _lensing_kernel(self, theta):
        r"""Weak-lensing efficiency W_κ(χ) on the shear grid, + χ(z) [Mpc/h].

        .. math::
            W_\kappa(\chi) = \tfrac32 \Omega_m (H_0/c)^2 \frac{\chi}{a}
                \int_\chi^{\chi_H}\! d\chi'\, n_s(\chi')\,\frac{\chi'-\chi}{\chi'}
        """
        c = self._cosmo(theta)
        h, Om = c["h"], c["Omega_m"]
        z = self.z_shear
        chi = comoving_distance(z, h, Om) * h                          # (Nz,) Mpc/h
        a = 1.0 / (1.0 + z)
        # source distribution per unit χ (n_s(z) dz = n_s(χ) dχ), renormalised
        dz_dchi = jnp.gradient(z) / jnp.gradient(chi)
        ns_chi = self.nz_src * dz_dchi
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
        """Convergence auto power spectrum C_ℓ^{κκ} via the lensing Limber integral."""
        chi, W, _ = self._lensing_kernel(theta)
        if stack_shear is None:
            stack_shear = self._pmm_logstack(theta, self.z_shear)
        return self._limber_kernels(chi, W, W, stack_shear)

    def _cmb_kernel(self, theta, chi, z):
        r"""CMB-lensing efficiency W_{κ_CMB}(χ), single source plane at z*≈1089.

        .. math::
            W_{\kappa_{\rm CMB}}(\chi) = \tfrac32 \Omega_m (H_0/c)^2
                \frac{\chi}{a}\,\frac{\chi_*-\chi}{\chi_*}
        """
        c = self._cosmo(theta)
        h, Om = c["h"], c["Omega_m"]
        chi_star = comoving_distance(jnp.array([self.z_star]), h, Om)[0] * h
        return (1.5 * Om / _C_OVER_H0 ** 2 * chi * (1.0 + z)
                * jnp.clip(chi_star - chi, 0.0, None) / chi_star)

    def _cl_gkCMB(self, theta, Hs_zgrid):
        """Galaxy × CMB-lensing C_ℓ: galaxy window × κ_CMB kernel, via P_gm."""
        c = self._cosmo(theta)
        chi = comoving_distance(self.z_grid, c["h"], c["Omega_m"]) * c["h"]
        dndchi = self.nz / jnp.trapezoid(self.nz, chi)
        Wc = self._cmb_kernel(theta, chi, self.z_grid)
        stack = jnp.stack([jnp.log(jnp.maximum(self._pk_gg_gm(H, theta)[1], 1e-30)) for H in Hs_zgrid])
        return self._limber_kernels(chi, dndchi, Wc, stack)

    def _cl_shear_kCMB(self, theta, stack_shear):
        """Cosmic-shear × CMB-lensing C_ℓ (both lensing kernels, matter power)."""
        chi, Wk, z = self._lensing_kernel(theta)
        Wc = self._cmb_kernel(theta, chi, z)
        return self._limber_kernels(chi, Wk, Wc, stack_shear)

    def _cl_kCMB(self, theta):
        """CMB-lensing auto C_ℓ^{κκ_CMB} over the high-z LOS grid."""
        c = self._cosmo(theta)
        chi = comoving_distance(self.z_cmb, c["h"], c["Omega_m"]) * c["h"]
        Wc = self._cmb_kernel(theta, chi, self.z_cmb)
        stack = self._pmm_logstack(theta, self.z_cmb)
        return self._limber_kernels(chi, Wc, Wc, stack)

    # ---- projections / Limber ---------------------------------------
    def _wp(self, P_gg):
        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_tab = _pk_to_xi(r_tab, self.log_k, jnp.log(jnp.maximum(P_gg, 1e-20)))
        pi_grid = jnp.linspace(0.0, self.pi_max, 512)

        def _one(rp_i):
            rr = jnp.sqrt(rp_i ** 2 + pi_grid ** 2)
            return 2.0 * jnp.trapezoid(jnp.interp(rr, r_tab, xi_tab), pi_grid)
        return jax.vmap(_one)(self.rp_wp)

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
        chi = comoving_distance(self.z_grid, h, Om) * h                # (Nz,) Mpc/h
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
        c = self._cosmo(theta)
        z = self.z_eff
        dndm = self._hmf.dndm(self.m, float(z), c)                      # (NM,)
        dndlogm = dndm * self.m * _LN10                                 # (NM,) per dex(M_h)

        hp = self._hod(theta)
        log10ms = _inv_shmr(self.log10m, hp["lg_m1h"], hp["lg_m0star"],
                            hp["beta"], hp["delta"], hp["gamma"])       # (NM,) ⟨log M_*⟩(M_h)
        mu_bh, al_bh = theta[_IDX["agn_mu_bh"]], theta[_IDX["agn_al_bh"]]
        sig_bh = theta[_IDX["agn_sig_bh"]]
        mean_bh = mu_bh + al_bh * (log10ms - 11.0)                      # (NM,) ⟨log M_BH⟩(M_h)
        sig_lm = jnp.sqrt(al_bh ** 2 * _SIG_MSTAR_XLF ** 2 + sig_bh ** 2)

        lstar = theta[_IDX["agn_log10_lstar"]]
        d1, d2 = theta[_IDX["agn_delta1"]], theta[_IDX["agn_delta2"]]
        x = 10.0 ** (self.loglam - lstar)
        erdf = 1.0 / (x ** d1 + x ** d2)                               # (Nlam,)
        erdf = erdf / (jnp.sum(erdf) * self.dlam)                       # ∫ dlogλ = 1

        # K(logL_X | M_h) = Σ_λ ERDF(λ) dλ · N(logL_X − logk − ⟨logM_BH⟩ − λ ; σ_lm)
        t = self.loglx_xlf[:, None, None] - _POWELL_LOGK - mean_bh[None, :, None]
        gk = jnp.exp(-0.5 * ((t - self.loglam[None, None, :]) / sig_lm) ** 2) \
            / (jnp.sqrt(2.0 * jnp.pi) * sig_lm)                        # (Nlx, NM, Nlam)
        K = jnp.sum(erdf[None, None, :] * gk, axis=2) * self.dlam      # (Nlx, NM)  pdf in dex(L_X)
        ferdf = 10.0 ** theta[_IDX["agn_log10_ferdf"]]                 # active fraction
        return jnp.trapezoid(dndlogm[None, :] * K * ferdf, self.log10m, axis=1)  # (Nlx,)

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

    def _n_gal(self, theta, H):
        """Comoving galaxy number density n̄_g [h³ Mpc⁻³] above the sample threshold.

        n̄_g ∝ f_c, so this observable is what breaks the otherwise-degenerate f_c.
        """
        nc, ns = self._occ_above(theta, self._hod(theta)["log10m_star_thresh"])
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
        which = which or OBSERVABLES
        out = {}
        if any(o in which for o in ("wp", "ds", "n_gal", "smf")):
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
        # Galaxy-window angular spectra (gas cross + galaxy×CMB-lensing) share the
        # per-z halo quantities on the galaxy n(z) grid.
        ang = [o for o in ("cl_gX", "cl_gy", "cl_XX") if o in which]
        if ang or "cl_gkCMB" in which:
            Hs = [self._halo_common(theta, float(z)) for z in np.asarray(self.z_grid)]
            fns = {"cl_gX": self._pk_gX, "cl_gy": self._pk_gy, "cl_XX": self._pk_XX}
            for o in ang:
                stack = jnp.stack([jnp.log(jnp.maximum(fns[o](theta, H), 1e-30)) for H in Hs])
                out[o] = self._limber_from_stack(theta, stack)
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
        return out

    def grid_of(self, name):
        """Return the abscissa grid (r_p, ℓ, or log10 L_X) for observable ``name``."""
        if name.startswith("cl_"):
            return self.ell
        return {"wp": self.rp_wp, "ds": self.rp_ds,
                "xlf": self.loglx_xlf, "n_gal": jnp.array([0.0]),
                "smf": self.logmstar_smf}[name]

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
        proj = np.isin(row_obs, ["wp", "ds"])
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
            if name in ("wp", "ds"):
                sel = np.where(g > float(rmin))[0]
            elif name in ("xlf", "n_gal", "smf"):
                sel = np.arange(g.size)            # abundance: no scale cut
            else:
                sel = np.where(g < ell_max)[0]
            idx[name] = jnp.asarray(sel, dtype=int)

        def f(theta):
            pred = self.predict(theta, which)
            parts = [pred[name][idx[name]] for name in which if idx[name].size > 0]
            return jnp.concatenate(parts) if parts else jnp.zeros(0)
        return f, idx
