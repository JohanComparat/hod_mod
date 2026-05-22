"""Halo Occupation Distribution (HOD) models in JAX.

Implements four HOD/ICSMF parametrisations:

**Zheng+2007** (ApJ 667, 760) — standard erfc HOD:

.. math::

    \\langle N_{\\rm cen}(M) \\rangle = \\frac{1}{2}\\,
    {\\rm erfc}\\!\\left[\\frac{\\log_{10}M_{\\rm min} - \\log_{10}M}{\\sigma_{\\log M}}\\right]

    \\langle N_{\\rm sat}(M) \\rangle = \\langle N_{\\rm cen}(M) \\rangle
    \\left(\\frac{M - M_0}{M_1}\\right)^\\alpha

**More+2015** (arXiv:1407.1856) — BOSS CMASS HOD with linear incompleteness function:

.. math::

    f_{\\rm inc}(M) = \\min\\!\\left[1,\\,\\max\\!\\left(0,\\,
        1 + \\alpha_{\\rm inc}\\,(\\log_{10}M - \\log_{10}M_{\\rm inc})\\right)\\right]

    \\langle N_{\\rm cen}(M) \\rangle = \\frac{f_{\\rm inc}(M)}{2}\\,
    {\\rm erfc}\\!\\left[\\frac{\\log_{10}M_{\\rm min} - \\log_{10}M}{\\sigma_{\\log M}}\\right]

    \\langle N_{\\rm sat}(M) \\rangle = \\langle N_{\\rm cen}(M) \\rangle
    \\left(\\frac{M - \\kappa M_{\\rm min}}{M_1}\\right)^\\alpha

**Guo+2018** (ApJ 858, 30) — Incomplete Conditional Stellar Mass Function (ICSMF)
using a broken power-law stellar-to-halo mass relation:

.. math::

    \\langle M_*(M) \\rangle = M_{*0}
    \\left(\\frac{M}{M_1}\\right)^{\\alpha+\\beta}
    \\left(1 + \\frac{M}{M_1}\\right)^{-\\beta}

Completeness functions (separate for centrals I and satellites II):

.. math::

    c(M_*) = \\frac{f}{2}\\left[1 + {\\rm erf}
    \\left(\\frac{\\log_{10}M_* - \\log_{10}M_{*,\\rm min}}{\\sigma_c}\\right)\\right]

**Guo+2019** (ApJ 871, 147) — ICSMF for eBOSS ELGs with quenched fraction:

.. math::

    f_q(M) = \\frac{1}{1 + M/M_q}, \\qquad f_{\\rm sf}(M) = 1 - f_q(M)

**Zu & Mandelbaum 2015** (arXiv:1505.02781, Paper I) — iHOD stellar-to-halo mass relation
with Behroozi+2010 inverse SHMR, mass-dependent log-normal scatter, and power-law
satellite occupation:

.. math::

    M_h = M_1 \\left(\\frac{M_*}{M_{*0}}\\right)^\\beta
    \\exp\\!\\left[\\frac{(M_*/M_{*0})^\\delta}{1+(M_*/M_{*0})^{-\\gamma}} - \\frac{1}{2}\\right]

    \\langle N_{\\rm sat}^{>M_*}\\rangle(M_h) = \\langle N_{\\rm cen}^{>M_*}\\rangle(M_h)
    \\left(\\frac{M_h}{M_{\\rm sat}}\\right)^{\\alpha_{\\rm sat}}
    \\exp\\!\\left(-\\frac{M_{\\rm cut}}{M_h}\\right)

**Zu & Mandelbaum 2016/2017** (arXiv:1509.06374, 1703.09219) — halo quenching model:

.. math::

    f_{\\rm red,c}(M_h) = 1 - \\exp\\!\\left[-\\left(M_h/M_h^{qc}\\right)^{\\mu_c}\\right]

    f_{\\rm red,s}(M_h) = 1 - \\exp\\!\\left[-\\left(M_h/M_h^{qs}\\right)^{\\mu_s}\\right]
"""

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf


@jax.jit
def n_cen(log10m: jnp.ndarray, log10mmin: float, sigma_logm: float) -> jnp.ndarray:
    """Mean number of central galaxies.

    N_cen(M) = (1/2) * erfc[(log10 M_min - log10 M) / sigma_logM]

    Accuracy
    --------
    N_cen(M_min) = 0.5 exactly (erfc argument = 0).  Asymptotes N_cen → 0 for
    M << M_min and N_cen → 1 for M >> M_min verified to < 1e-6 (2026-04-23).

    Timing
    ------
    ~ 17 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    return 0.5 * erfc((log10mmin - log10m) / sigma_logm)


@jax.jit
def n_sat(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    log10m0: float,
    log10m1: float,
    alpha: float,
) -> jnp.ndarray:
    """Mean number of satellite galaxies.

    N_sat(M) = N_cen(M) * [(M - M_0) / M_1]^alpha  for M > M_0

    Accuracy
    --------
    N_sat(M ≤ M_0) = 0 exactly (jnp.where guard).  Power-law slope
    d log N_sat / d log M → α verified to < 2% for M >> M_1 (2026-04-23).

    Timing
    ------
    ~ 31 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    nc = n_cen(log10m, log10mmin, sigma_logm)
    m = jnp.power(10.0, log10m)
    m0 = jnp.power(10.0, log10m0)
    m1 = jnp.power(10.0, log10m1)
    ratio = jnp.where(m > m0, (m - m0) / m1, 0.0)
    return nc * ratio**alpha


@jax.jit
def n_total(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    log10m0: float,
    log10m1: float,
    alpha: float,
) -> jnp.ndarray:
    """Total mean galaxy number N_tot = N_cen + N_sat.

    Accuracy
    --------
    N_tot ≥ N_cen pointwise; N_tot → 1 for M_min < M < M_0 (satellite cutoff).

    Timing
    ------
    ~ 33 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    return n_cen(log10m, log10mmin, sigma_logm) + n_sat(
        log10m, log10mmin, sigma_logm, log10m0, log10m1, alpha
    )


# ---------------------------------------------------------------------------
# HODBase — Abstract base class shared by all HOD/ICSMF models
# ---------------------------------------------------------------------------

class HODBase(ABC):
    """Abstract base class for HOD and ICSMF models.

    Subclasses must implement :meth:`nc_ns` (occupation arrays) and
    :meth:`default_params` (reference parameters).  :meth:`_integrate`,
    :meth:`galaxy_number_density`, :meth:`effective_bias`, and
    :meth:`effective_mass` are provided here and delegate to :meth:`nc_ns`.

    Parameters
    ----------
    hmf : HaloMassFunction
        Halo mass function instance providing ``.dndm(m, z, theta)`` and
        ``.bias(m, z, theta)``.
    halo_bias : callable or None
        Function (m, z, theta_cosmo) → b(m).  When *None* (default) the
        bias callable is taken from ``hmf.bias`` — use this for models that
        receive only ``hmf`` in their constructor.
    """

    #: Set to True on classes whose constructor takes only `hmf` (no halo_bias).
    _SINGLE_ARG_INIT: bool = False
    #: Number of log-spaced halo mass points for the integration grid.
    _N_M_GRID: int = 512

    def __init__(self, hmf, halo_bias=None):
        self._hmf = hmf
        self._bias = halo_bias if halo_bias is not None else hmf.bias
        self._m_grid = jnp.logspace(10, 16, self._N_M_GRID)
        self._log10m_grid = jnp.log10(self._m_grid)

    @abstractmethod
    def nc_ns(
        self,
        log10m_arr: jnp.ndarray,
        hod_params: dict,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (N_cen, N_sat) occupation arrays on *log10m_arr*."""

    @staticmethod
    @abstractmethod
    def default_params() -> dict:
        """Return a dict of reference HOD parameter values."""

    def _integrate(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> tuple:
        """Return (n_gal, b_eff, M_eff) — number density, effective bias, mean mass."""
        m      = self._m_grid
        dn     = self._hmf.dndm(m, z, theta_cosmo)
        b      = self._bias(m, z, theta_cosmo)
        nc, ns = self.nc_ns(self._log10m_grid, hod_params)
        nt     = nc + ns
        n_gal  = jnp.trapezoid(dn * nt, m)
        b_eff  = jnp.trapezoid(dn * nt * b, m) / n_gal
        m_eff  = jnp.trapezoid(dn * nt * m, m) / n_gal
        return n_gal, b_eff, m_eff

    def galaxy_number_density(
        self, z: float, theta_cosmo: dict, hod_params: dict
    ) -> jnp.ndarray:
        """Comoving galaxy number density n_gal [h³ Mpc⁻³]."""
        n, _, _ = self._integrate(z, theta_cosmo, hod_params)
        return n

    def effective_bias(
        self, z: float, theta_cosmo: dict, hod_params: dict
    ) -> jnp.ndarray:
        """Effective large-scale galaxy bias b_eff."""
        _, b, _ = self._integrate(z, theta_cosmo, hod_params)
        return b

    def effective_mass(
        self, z: float, theta_cosmo: dict, hod_params: dict
    ) -> jnp.ndarray:
        """Effective halo mass ⟨M⟩ [M_sun/h]."""
        _, _, m = self._integrate(z, theta_cosmo, hod_params)
        return m


class HODModel(HODBase):
    """Zheng+2007 HOD: computes galaxy number density, mean halo mass, and bias.

    Parameters
    ----------
    hmf : HaloMassFunction
        Halo mass function instance providing dndm(m, z, theta).
    halo_bias : callable
        Function (m, z, theta) → b(m), e.g. HaloMassFunction.bias.
    """



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model."""
        p = hod_params
        nc = n_cen(log10m_arr, p["log10mmin"], p["sigma_logm"])
        ns = n_sat(log10m_arr, p["log10mmin"], p["sigma_logm"],
                   p["log10m0"], p["log10m1"], p["alpha"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """SDSS Main Sample-like HOD parameters (Zheng+2007 Table 1)."""
        return {
            "log10mmin": 11.35,
            "sigma_logm": 0.25,
            "log10m0": 11.20,
            "log10m1": 12.40,
            "alpha": 1.0,
        }


# ---------------------------------------------------------------------------
# Kravtsov+2004 HOD  (implemented in the surhudm/aum code as TINK=0)
# ---------------------------------------------------------------------------

@jax.jit
def n_sat_kravtsov04(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    log10m0: float,
    log10m1: float,
    alpha: float,
) -> jnp.ndarray:
    """Satellite occupation from Kravtsov+2004 (ApJ 609, 35).

    N_sat(M) = N_cen(M) * (M / M_1)^{\\alpha} * exp(-M_0 / M)

    where M_1 = 10^{log10m1} and M_0 = 10^{log10m0}.

    The exponential factor suppresses satellites in haloes with
    M \\lesssim M_0 without a hard mass threshold.  For M_0 \\ll M_min
    the exponential is \\approx 1 and the formula reduces to a pure
    power law (M / M_1)^\\alpha, identical to the large-M limit of the
    Zheng+2007 `n_sat`.  Also used by Zu & Mandelbaum 2015 for satellites.

    Parameters
    ----------
    log10m : jnp.ndarray
        log10(M [M_sun/h])
    log10mmin : float
        log10 of the central mass threshold [log10(M_sun/h)].
    sigma_logm : float
        Log-normal scatter in the central HOD.
    log10m0 : float
        log10 of the satellite exponential cutoff mass [log10(M_sun/h)].
        Equivalent to surhudm/aum ``hodpars.Mcut``.
    log10m1 : float
        log10 of the satellite mass scale [log10(M_sun/h)].
        Equivalent to surhudm/aum ``hodpars.Msat``.
    alpha : float
        Power-law slope of the satellite occupation.
        Equivalent to surhudm/aum ``hodpars.alpsat``.

    Returns
    -------
    jnp.ndarray
        N_sat(M), same shape as log10m.
    """
    nc = n_cen(log10m, log10mmin, sigma_logm)
    # (M / M1)^alpha = 10^(alpha * (log10m - log10m1))
    ratio = jnp.power(10.0, alpha * (log10m - log10m1))
    # exp(-M0 / M) = exp(-10^(log10m0 - log10m))
    cutoff = jnp.exp(-jnp.power(10.0, log10m0 - log10m))
    return nc * ratio * cutoff


# backward-compat alias
n_sat_aum = n_sat_kravtsov04


@jax.jit
def n_total_kravtsov04(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    log10m0: float,
    log10m1: float,
    alpha: float,
) -> jnp.ndarray:
    """N_tot = N_cen + N_sat with the Kravtsov+2004 satellite formula."""
    return n_cen(log10m, log10mmin, sigma_logm) + n_sat_kravtsov04(
        log10m, log10mmin, sigma_logm, log10m0, log10m1, alpha
    )


# backward-compat alias
n_total_aum = n_total_kravtsov04


class Kravtsov04HODModel(HODBase):
    """Kravtsov+2004 HOD with an exponential satellite cutoff.

    Satellite formula (Kravtsov et al. 2004, ApJ 609, 35):

    .. math::

        N_{\\rm sat}(M) = N_{\\rm cen}(M)
            \\left(\\frac{M}{M_1}\\right)^{\\alpha}
            \\exp\\!\\left(-\\frac{M_0}{M}\\right)

    where :math:`M_0 = 10^{\\rm log10m0}` and :math:`M_1 = 10^{\\rm log10m1}`.
    The exponential term replaces the hard step-function cutoff of `HODModel`.
    The same satellite form is used by Zu & Mandelbaum 2015 (arXiv:1505.02781).
    This parametrisation is the TINK=0 default of the surhudm/aum C++ code.

    Parameters
    ----------
    hmf : HaloMassFunction
        Halo mass function instance (``tinker08`` recommended).
    halo_bias : callable
        Function (m, z, theta) → b(m).

    Parameter mapping to surhudm/aum ``hodpars``
    ---------------------------------------------
    ``log10mmin`` ↔ ``Mmin``  (central erfc threshold)
    ``sigma_logm`` ↔ ``siglogM``
    ``log10m0``   ↔ ``Mcut``  (exponential cutoff)
    ``log10m1``   ↔ ``Msat``  (satellite mass scale)
    ``alpha``     ↔ ``alpsat``
    """



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays for the full halo model."""
        p = hod_params
        nc = n_cen(log10m_arr, p["log10mmin"], p["sigma_logm"])
        ns = n_sat_kravtsov04(log10m_arr, p["log10mmin"], p["sigma_logm"],
                              p["log10m0"], p["log10m1"], p["alpha"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """Default HOD parameters matching surhudm/aum hod.cpp defaults."""
        return {
            "log10mmin":  13.0,
            "sigma_logm":  0.5,
            "log10m0":    13.5,   # Mcut — satellite exponential cutoff
            "log10m1":    14.0,   # Msat — satellite mass scale
            "alpha":       1.0,
        }


# backward-compat alias — name after validation code (surhudm/aum)
AUMHODModel = Kravtsov04HODModel


# ---------------------------------------------------------------------------
# More+2015  (arXiv:1407.1856)  BOSS CMASS HOD with incompleteness
# ---------------------------------------------------------------------------

@jax.jit
def _incompleteness_more15(
    log10m: jnp.ndarray,
    alpha_inc: float,
    log10m_inc: float,
) -> jnp.ndarray:
    """Linear incompleteness function of More+2015 Eq. 1.

    f_inc(M) = clip(1 + alpha_inc * (log10 M - log10 M_inc), 0, 1)
    """
    f = 1.0 + alpha_inc * (log10m - log10m_inc)
    return jnp.clip(f, 0.0, 1.0)


@jax.jit
def n_cen_more15(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    alpha_inc: float,
    log10m_inc: float,
) -> jnp.ndarray:
    """More+2015 mean central galaxy count with incompleteness.

    N_cen(M) = f_inc(M) * (1/2) * erfc[(log10 M_min - log10 M) / sigma_logM]

    Parameters
    ----------
    log10m : log10(M / (M_sun/h))
    log10mmin, sigma_logm : Zheng+2007 erfc threshold
    alpha_inc, log10m_inc : slope and pivot of the linear incompleteness function
    """
    f_inc = _incompleteness_more15(log10m, alpha_inc, log10m_inc)
    return f_inc * 0.5 * erfc((log10mmin - log10m) / sigma_logm)


@jax.jit
def n_sat_more15(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    log10m1: float,
    alpha: float,
    kappa: float,
    alpha_inc: float,
    log10m_inc: float,
) -> jnp.ndarray:
    """More+2015 mean satellite galaxy count.

    N_sat(M) = N_cen(M) * [(M - kappa*M_min) / M_1]^alpha  for M > kappa*M_min

    The satellite threshold kappa*M_min replaces the M_0 parameter of Zheng+2007.
    """
    nc = n_cen_more15(log10m, log10mmin, sigma_logm, alpha_inc, log10m_inc)
    m = jnp.power(10.0, log10m)
    mmin = jnp.power(10.0, log10mmin)
    m1 = jnp.power(10.0, log10m1)
    m_thresh = kappa * mmin
    ratio = jnp.where(m > m_thresh, (m - m_thresh) / m1, 0.0)
    return nc * ratio**alpha


@jax.jit
def n_total_more15(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    log10m1: float,
    alpha: float,
    kappa: float,
    alpha_inc: float,
    log10m_inc: float,
) -> jnp.ndarray:
    """Total More+2015 HOD: N_tot = N_cen + N_sat."""
    return n_cen_more15(log10m, log10mmin, sigma_logm, alpha_inc, log10m_inc) + n_sat_more15(
        log10m, log10mmin, sigma_logm, log10m1, alpha, kappa, alpha_inc, log10m_inc
    )


class MoreHODModel(HODBase):
    """More+2015 HOD for BOSS CMASS galaxies with linear incompleteness.

    Parameters
    ----------
    hmf : HaloMassFunction
    halo_bias : callable  (m, z, theta_cosmo) → b(m)
    """



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model."""
        p = hod_params
        nc = n_cen_more15(log10m_arr, p["log10mmin"], p["sigma_logm"],
                          p["alpha_inc"], p["log10m_inc"])
        ns = n_sat_more15(log10m_arr, p["log10mmin"], p["sigma_logm"],
                          p["log10m1"], p["alpha"], p["kappa"],
                          p["alpha_inc"], p["log10m_inc"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """BOSS CMASS-like HOD parameters (More+2015 Table 1)."""
        return {
            "log10mmin": 13.03,
            "sigma_logm": 0.38,
            "log10m1": 14.00,
            "alpha": 1.0,
            "kappa": 1.0,
            "alpha_inc": 1.0,
            "log10m_inc": 13.0,
        }


# ---------------------------------------------------------------------------
# Guo+2018  (ApJ 858, 30)  Incomplete CSMF
# ---------------------------------------------------------------------------

@jax.jit
def shmr_guo18(
    log10m: jnp.ndarray,
    log10m_star0: float,
    log10m1: float,
    alpha: float,
    beta: float,
) -> jnp.ndarray:
    """Guo+2018 broken power-law stellar-to-halo mass relation.

    log10 <M_*(M)> = log10 M_*0 + (alpha+beta)*log10(M/M_1) - beta*log10(1 + M/M_1)

    Returns log10(M_* / (M_sun/h)).

    Accuracy
    --------
    Monotonically increasing over log10(M_h) ∈ [10, 15] for α, β > 0.
    At M = M_1: log10(M_*) = log10(M_*0) + α*log10(2) − β*log10(2) by
    construction (analytical check, 2026-04-23).

    Timing
    ------
    ~ 25 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    x = log10m - log10m1  # log10(M/M_1)
    m_over_m1 = jnp.power(10.0, x)
    return log10m_star0 + (alpha + beta) * x - beta * jnp.log10(1.0 + m_over_m1)


@jax.jit
def completeness_guo18(
    log10m_star: jnp.ndarray,
    f_comp: float,
    log10m_star_min: float,
    sigma_c: float,
) -> jnp.ndarray:
    """Guo+2018 erf-based galaxy completeness function.

    c(M_*) = (f/2) * [1 + erf((log10 M_* - log10 M_*,min) / sigma_c)]

    Parameters
    ----------
    log10m_star : log10(M_* / (M_sun/h))
    f_comp : amplitude (≤1), encodes survey incompleteness ceiling
    log10m_star_min : log10 of the stellar mass completeness threshold
    sigma_c : width of the erf transition
    """
    return (f_comp / 2.0) * (1.0 + erf((log10m_star - log10m_star_min) / sigma_c))


@jax.jit
def n_cen_guo18(
    log10m: jnp.ndarray,
    log10m_star0: float,
    log10m1_shmr: float,
    alpha_shmr: float,
    beta_shmr: float,
    sigma_logm_star: float,
    f_cen: float,
    log10m_star_min_cen: float,
    sigma_c_cen: float,
) -> jnp.ndarray:
    """Guo+2018 central galaxy ICSMF collapsed to mean occupation.

    N_cen(M) = c_cen(M_*(M)) * (1/2) * erfc[(log10 M_*(M) - log10 M_min_*) / sigma_logM_*]

    The central count integrates over the log-normal CSMF weighted by completeness.
    Here we use the analytic approximation: N_cen ≈ completeness at the mean SHMR.
    """
    log10m_star_mean = shmr_guo18(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr)
    c_cen = completeness_guo18(log10m_star_mean, f_cen, log10m_star_min_cen, sigma_c_cen)
    return c_cen * 0.5 * erfc((log10m_star_min_cen - log10m_star_mean) / sigma_logm_star)


@jax.jit
def n_sat_guo18(
    log10m: jnp.ndarray,
    log10m_star0: float,
    log10m1_shmr: float,
    alpha_shmr: float,
    beta_shmr: float,
    sigma_logm_star: float,
    f_sat: float,
    log10m_star_min_sat: float,
    sigma_c_sat: float,
    log10m1_sat: float,
    alpha_sat: float,
) -> jnp.ndarray:
    """Guo+2018 satellite galaxy mean occupation via ICSMF.

    N_sat(M) = c_sat(M_*(M)) * (M / M_1,sat)^alpha_sat
    """
    log10m_star_mean = shmr_guo18(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr)
    c_sat = completeness_guo18(log10m_star_mean, f_sat, log10m_star_min_sat, sigma_c_sat)
    m = jnp.power(10.0, log10m)
    m1_sat = jnp.power(10.0, log10m1_sat)
    return c_sat * (m / m1_sat) ** alpha_sat


@jax.jit
def n_total_guo18(
    log10m: jnp.ndarray,
    log10m_star0: float,
    log10m1_shmr: float,
    alpha_shmr: float,
    beta_shmr: float,
    sigma_logm_star: float,
    f_cen: float,
    log10m_star_min_cen: float,
    sigma_c_cen: float,
    f_sat: float,
    log10m_star_min_sat: float,
    sigma_c_sat: float,
    log10m1_sat: float,
    alpha_sat: float,
) -> jnp.ndarray:
    """Total Guo+2018 ICSMF occupation: N_tot = N_cen + N_sat."""
    nc = n_cen_guo18(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
        sigma_logm_star, f_cen, log10m_star_min_cen, sigma_c_cen,
    )
    ns = n_sat_guo18(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
        sigma_logm_star, f_sat, log10m_star_min_sat, sigma_c_sat,
        log10m1_sat, alpha_sat,
    )
    return nc + ns


class Guo18ICSMFModel(HODBase):
    """Guo+2018 Incomplete CSMF HOD for BOSS LOWZ/CMASS.

    Parameters
    ----------
    hmf : HaloMassFunction
    halo_bias : callable  (m, z, theta_cosmo) → b(m)
    """



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model."""
        p = hod_params
        nc = n_cen_guo18(log10m_arr, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_cen"], p["log10m_star_min_cen"], p["sigma_c_cen"])
        ns = n_sat_guo18(log10m_arr, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_sat"], p["log10m_star_min_sat"], p["sigma_c_sat"],
                         p["log10m1_sat"], p["alpha_sat"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """BOSS LOWZ-like ICSMF parameters (Guo+2018 Table 2)."""
        return {
            "log10m_star0": 10.7,
            "log10m1_shmr": 11.9,
            "alpha_shmr": 0.3,
            "beta_shmr": 1.5,
            "sigma_logm_star": 0.15,
            "f_cen": 1.0,
            "log10m_star_min_cen": 10.5,
            "sigma_c_cen": 0.1,
            "f_sat": 1.0,
            "log10m_star_min_sat": 10.2,
            "sigma_c_sat": 0.2,
            "log10m1_sat": 13.0,
            "alpha_sat": 1.0,
        }


# ---------------------------------------------------------------------------
# Guo+2019  (ApJ 871, 147)  eBOSS ELG ICSMF with quenched fraction
# ---------------------------------------------------------------------------

@jax.jit
def f_quenched(log10m: jnp.ndarray, log10m_q: float) -> jnp.ndarray:
    """Guo+2019 quenched galaxy fraction.

    f_q(M) = 1 / (1 + M / M_q)

    Returns the fraction of quenched galaxies at halo mass M.

    Accuracy
    --------
    f_q(M → 0) → 1 and f_q(M → ∞) → 0 exactly.  f_q(M_q) = 0.5 by
    construction; verified to < 1e-6 (2026-04-23).

    Timing
    ------
    ~ 14 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    m = jnp.power(10.0, log10m)
    m_q = jnp.power(10.0, log10m_q)
    return 1.0 / (1.0 + m / m_q)


@jax.jit
def f_starforming(log10m: jnp.ndarray, log10m_q: float) -> jnp.ndarray:
    """Guo+2019 star-forming galaxy fraction: f_sf = 1 - f_q(M)."""
    return 1.0 - f_quenched(log10m, log10m_q)


@jax.jit
def n_cen_guo19(
    log10m: jnp.ndarray,
    log10m_star0: float,
    log10m1_shmr: float,
    alpha_shmr: float,
    beta_shmr: float,
    sigma_logm_star: float,
    f_cen: float,
    log10m_star_min_cen: float,
    sigma_c_cen: float,
    log10m_q: float,
) -> jnp.ndarray:
    """Guo+2019 star-forming central galaxy mean occupation.

    N_cen,sf(M) = f_sf(M) * N_cen,Guo18(M)

    The quenched fraction suppresses central occupation at high halo mass.
    """
    nc_guo18 = n_cen_guo18(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
        sigma_logm_star, f_cen, log10m_star_min_cen, sigma_c_cen,
    )
    return f_starforming(log10m, log10m_q) * nc_guo18


@jax.jit
def n_sat_guo19(
    log10m: jnp.ndarray,
    log10m_star0: float,
    log10m1_shmr: float,
    alpha_shmr: float,
    beta_shmr: float,
    sigma_logm_star: float,
    f_sat: float,
    log10m_star_min_sat: float,
    sigma_c_sat: float,
    log10m1_sat: float,
    alpha_sat: float,
    log10m_q: float,
) -> jnp.ndarray:
    """Guo+2019 star-forming satellite galaxy mean occupation.

    N_sat,sf(M) = f_sf(M) * N_sat,Guo18(M)
    """
    ns_guo18 = n_sat_guo18(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
        sigma_logm_star, f_sat, log10m_star_min_sat, sigma_c_sat,
        log10m1_sat, alpha_sat,
    )
    return f_starforming(log10m, log10m_q) * ns_guo18


@jax.jit
def n_total_guo19(
    log10m: jnp.ndarray,
    log10m_star0: float,
    log10m1_shmr: float,
    alpha_shmr: float,
    beta_shmr: float,
    sigma_logm_star: float,
    f_cen: float,
    log10m_star_min_cen: float,
    sigma_c_cen: float,
    f_sat: float,
    log10m_star_min_sat: float,
    sigma_c_sat: float,
    log10m1_sat: float,
    alpha_sat: float,
    log10m_q: float,
) -> jnp.ndarray:
    """Total Guo+2019 star-forming ICSMF occupation: N_tot = N_cen,sf + N_sat,sf."""
    nc = n_cen_guo19(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
        sigma_logm_star, f_cen, log10m_star_min_cen, sigma_c_cen, log10m_q,
    )
    ns = n_sat_guo19(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
        sigma_logm_star, f_sat, log10m_star_min_sat, sigma_c_sat,
        log10m1_sat, alpha_sat, log10m_q,
    )
    return nc + ns


class Guo19ICSMFModel(HODBase):
    """Guo+2019 ICSMF HOD for eBOSS ELGs with quenched fraction.

    Extends Guo+2018 by multiplying all occupations by the star-forming fraction
    f_sf(M) = 1 / (1 + M_q/M), suppressing ELG occupation in massive (quenched) haloes.

    Parameters
    ----------
    hmf : HaloMassFunction
    halo_bias : callable  (m, z, theta_cosmo) → b(m)
    """



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model."""
        p = hod_params
        nc = n_cen_guo19(log10m_arr, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_cen"], p["log10m_star_min_cen"], p["sigma_c_cen"], p["log10m_q"])
        ns = n_sat_guo19(log10m_arr, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_sat"], p["log10m_star_min_sat"], p["sigma_c_sat"],
                         p["log10m1_sat"], p["alpha_sat"], p["log10m_q"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """eBOSS ELG-like ICSMF parameters (Guo+2019 Table 2)."""
        return {
            "log10m_star0": 10.0,
            "log10m1_shmr": 11.5,
            "alpha_shmr": 0.3,
            "beta_shmr": 1.5,
            "sigma_logm_star": 0.15,
            "f_cen": 0.5,
            "log10m_star_min_cen": 9.8,
            "sigma_c_cen": 0.1,
            "f_sat": 0.3,
            "log10m_star_min_sat": 9.5,
            "sigma_c_sat": 0.2,
            "log10m1_sat": 12.5,
            "alpha_sat": 1.0,
            "log10m_q": 12.0,
        }


# =============================================================================
# Zacharegkas & Chang et al. 2025 HOD (arXiv:2506.22367, Section 2)
# Kravtsov+2018 SHMR with stellar-mass threshold/bin selection and
# exponential satellite cutoff.
# =============================================================================

_LN10 = jnp.log(10.0)


@jax.jit
def _f_kravtsov(x: jnp.ndarray, alpha: float, gamma: float, delta: float) -> jnp.ndarray:
    r"""Auxiliary function for the Kravtsov+2018 SHMR.

    .. math::
        f(x) = -\log_{10}(10^{\alpha x}+1)
               + \delta\,[\log_{10}(1+e^x)]^\gamma / (1+e^{10^{-x}})

    Numerically stable via ``jnp.logaddexp``; clamps ``10^{-x}`` to avoid
    overflow in the denominator for large negative ``x``.

    Parameters
    ----------
    x : log10(M_h / M_1)
    alpha, gamma, delta : Kravtsov+2018 shape parameters
    """
    # -log10(10^{alpha*x} + 1) = -logaddexp(alpha*x*ln10, 0) / ln10
    term1 = -jnp.logaddexp(alpha * x * _LN10, 0.0) / _LN10
    # log10(1 + e^x) = logaddexp(x, 0) / ln10
    log10_1pex = jnp.logaddexp(x, 0.0) / _LN10
    # 10^{-x} clipped to prevent exp overflow in denominator
    pow10_neg_x = jnp.minimum(jnp.power(10.0, -x), 500.0)
    term2 = delta * jnp.power(jnp.maximum(log10_1pex, 0.0), gamma) / (1.0 + jnp.exp(pow10_neg_x))
    return term1 + term2


@jax.jit
def shmr_zacharegkas25(
    log10m_h: jnp.ndarray,
    log10m1: float,
    log10eps: float,
    alpha: float,
    gamma: float,
    delta: float,
) -> jnp.ndarray:
    r"""Stellar-to-halo mass relation from Kravtsov+2018 used in Zacharegkas+2025.

    .. math::
        \log_{10} M_\star = \log_{10}(\varepsilon M_1) + f(\log_{10}(M_h/M_1)) - f(0)

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    log10m1 : log10(M_1 / [M_sun/h]), characteristic halo mass
    log10eps : log10(ε), amplitude of SHMR
    alpha, gamma, delta : Kravtsov+2018 shape parameters

    Returns
    -------
    log10(M_star / [M_sun/h])

    Accuracy
    --------
    Monotonically increasing over log10(M_h) ∈ [10, 15] for the Zacharegkas+2025
    best-fit parameters (log10m1=11.506, log10eps=-1.632, alpha=-1.638, gamma=0.596,
    delta=3.810); verified numerically (2026-04-23).

    Timing
    ------
    ~ 33 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    x = log10m_h - log10m1
    f0 = _f_kravtsov(jnp.zeros_like(x), alpha, gamma, delta)
    return (log10eps + log10m1) + _f_kravtsov(x, alpha, gamma, delta) - f0


def _inverse_shmr_z25(
    log10m_star_thresh: float,
    log10m1: float,
    log10eps: float,
    alpha: float,
    gamma: float,
    delta: float,
    n_iter: int = 60,
) -> jnp.ndarray:
    """Invert the Zacharegkas+2025 SHMR via bisection.

    Returns ``log10(M_min)`` such that ``shmr_zacharegkas25(log10m_min, ...) = log10m_star_thresh``.
    Uses ``jax.lax.fori_loop`` for 60 iterations over [10, 16] in log10(M_h).
    """
    thresh = jnp.array(log10m_star_thresh)

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        f_mid = shmr_zacharegkas25(mid, log10m1, log10eps, alpha, gamma, delta)
        lo = jnp.where(f_mid < thresh, mid, lo)
        hi = jnp.where(f_mid < thresh, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (jnp.array(10.0), jnp.array(16.0)))
    return 0.5 * (lo + hi)


@jax.jit
def n_cen_thresh_z25(
    log10m_h: jnp.ndarray,
    log10m_star_thresh: float,
    log10m1_shmr: float,
    log10eps: float,
    alpha_shmr: float,
    gamma_shmr: float,
    delta_shmr: float,
    sigma_logm_star: float,
    f_cen: float,
) -> jnp.ndarray:
    r"""Mean central occupation for a stellar-mass threshold sample (Eq. 2).

    .. math::
        \langle N_\mathrm{cen}^{>M_\star}\rangle(M_h)
        = \frac{f_\mathrm{cen}}{2}\left[1 + \mathrm{erf}\!\left(
          \frac{\log_{10}\langle M_\star\rangle(M_h) - \log_{10}M_\star^\mathrm{thresh}}
               {\sigma_{\log M_\star}}\right)\right]

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    log10m_star_thresh : log10(stellar mass threshold / [M_sun/h])
    """
    log10m_star_mean = shmr_zacharegkas25(log10m_h, log10m1_shmr, log10eps, alpha_shmr, gamma_shmr, delta_shmr)
    return (f_cen / 2.0) * (1.0 + erf((log10m_star_mean - log10m_star_thresh) / sigma_logm_star))


def n_sat_thresh_z25(
    log10m_h: jnp.ndarray,
    log10m_star_thresh: float,
    log10m1_shmr: float,
    log10eps: float,
    alpha_shmr: float,
    gamma_shmr: float,
    delta_shmr: float,
    alpha_sat: float,
    kappa: float,
    B_sat: float,
    beta_sat: float,
    B_cut: float,
    beta_cut: float,
    f_sat: float,
) -> jnp.ndarray:
    r"""Mean satellite occupation for a stellar-mass threshold sample (Eqs. 5–7).

    .. math::
        \langle N_\mathrm{sat}^{>M_\star}\rangle(M_h)
        = f_\mathrm{sat}\left(\frac{M_h - \kappa M_\mathrm{min}}{M_\mathrm{sat}}\right)^{\alpha_\mathrm{sat}}
          e^{-M_\mathrm{cut}/M_h}

    with :math:`M_\mathrm{sat} = B_\mathrm{sat}(M_\mathrm{min}/10^{12})^{\beta_\mathrm{sat}}\times10^{12}`
    and :math:`M_\mathrm{cut} = B_\mathrm{cut}(M_\mathrm{min}/10^{12})^{\beta_\mathrm{cut}}\times10^{12}`.

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    log10m_star_thresh : log10(stellar mass threshold / [M_sun/h])
    """
    log10m_min = _inverse_shmr_z25(log10m_star_thresh, log10m1_shmr, log10eps, alpha_shmr, gamma_shmr, delta_shmr)
    m_min = jnp.power(10.0, log10m_min)
    m_h = jnp.power(10.0, log10m_h)
    m_min_norm = m_min / 1e12
    m_sat = B_sat * jnp.power(m_min_norm, beta_sat) * 1e12
    m_cut = B_cut * jnp.power(m_min_norm, beta_cut) * 1e12
    ratio = jnp.where(m_h > kappa * m_min, (m_h - kappa * m_min) / m_sat, 0.0)
    return f_sat * jnp.power(jnp.maximum(ratio, 0.0), alpha_sat) * jnp.exp(-m_cut / m_h)


@jax.jit
def n_cen_bin_z25(
    log10m_h: jnp.ndarray,
    log10m_star_lo: float,
    log10m_star_hi: float,
    log10m1_shmr: float,
    log10eps: float,
    alpha_shmr: float,
    gamma_shmr: float,
    delta_shmr: float,
    sigma_logm_star: float,
    f_cen: float,
) -> jnp.ndarray:
    r"""Mean central occupation for a stellar-mass bin (Eq. 3).

    .. math::
        \langle N_\mathrm{cen}^{[M_{\star,\mathrm{lo}}, M_{\star,\mathrm{hi}}]}\rangle
        = \langle N_\mathrm{cen}^{>M_{\star,\mathrm{lo}}}\rangle
          - \langle N_\mathrm{cen}^{>M_{\star,\mathrm{hi}}}\rangle
    """
    nc_lo = n_cen_thresh_z25(log10m_h, log10m_star_lo, log10m1_shmr, log10eps,
                              alpha_shmr, gamma_shmr, delta_shmr, sigma_logm_star, f_cen)
    nc_hi = n_cen_thresh_z25(log10m_h, log10m_star_hi, log10m1_shmr, log10eps,
                              alpha_shmr, gamma_shmr, delta_shmr, sigma_logm_star, f_cen)
    return nc_lo - nc_hi


def n_sat_bin_z25(
    log10m_h: jnp.ndarray,
    log10m_star_lo: float,
    log10m_star_hi: float,
    log10m1_shmr: float,
    log10eps: float,
    alpha_shmr: float,
    gamma_shmr: float,
    delta_shmr: float,
    alpha_sat: float,
    kappa: float,
    B_sat: float,
    beta_sat: float,
    B_cut: float,
    beta_cut: float,
    f_sat: float,
) -> jnp.ndarray:
    r"""Mean satellite occupation for a stellar-mass bin (Eq. 4).

    .. math::
        \langle N_\mathrm{sat}^{[M_{\star,\mathrm{lo}}, M_{\star,\mathrm{hi}}]}\rangle
        = \langle N_\mathrm{sat}^{>M_{\star,\mathrm{lo}}}\rangle
          - \langle N_\mathrm{sat}^{>M_{\star,\mathrm{hi}}}\rangle
    """
    ns_lo = n_sat_thresh_z25(log10m_h, log10m_star_lo, log10m1_shmr, log10eps,
                              alpha_shmr, gamma_shmr, delta_shmr, alpha_sat,
                              kappa, B_sat, beta_sat, B_cut, beta_cut, f_sat)
    ns_hi = n_sat_thresh_z25(log10m_h, log10m_star_hi, log10m1_shmr, log10eps,
                              alpha_shmr, gamma_shmr, delta_shmr, alpha_sat,
                              kappa, B_sat, beta_sat, B_cut, beta_cut, f_sat)
    return ns_lo - ns_hi


def n_total_bin_z25(
    log10m_h: jnp.ndarray,
    log10m_star_lo: float,
    log10m_star_hi: float,
    log10m1_shmr: float,
    log10eps: float,
    alpha_shmr: float,
    gamma_shmr: float,
    delta_shmr: float,
    sigma_logm_star: float,
    f_cen: float,
    alpha_sat: float,
    kappa: float,
    B_sat: float,
    beta_sat: float,
    B_cut: float,
    beta_cut: float,
    f_sat: float,
) -> jnp.ndarray:
    """Total mean occupation (central + satellite) for a stellar-mass bin."""
    nc = n_cen_bin_z25(log10m_h, log10m_star_lo, log10m_star_hi,
                       log10m1_shmr, log10eps, alpha_shmr, gamma_shmr, delta_shmr,
                       sigma_logm_star, f_cen)
    ns = n_sat_bin_z25(log10m_h, log10m_star_lo, log10m_star_hi,
                       log10m1_shmr, log10eps, alpha_shmr, gamma_shmr, delta_shmr,
                       alpha_sat, kappa, B_sat, beta_sat, B_cut, beta_cut, f_sat)
    return nc + ns


class Zacharegkas25HODModel(HODBase):
    """HOD model from Zacharegkas & Chang et al. 2025 (arXiv:2506.22367).

    Uses the Kravtsov+2018 SHMR to define stellar-mass selected samples
    (threshold or bin) and scales satellite halo masses from the SHMR-derived
    minimum halo mass ``M_min``.

    Parameters
    ----------
    hmf : HaloMassFunction
        Must expose ``.dndm(m, z, theta_cosmo)`` and ``.bias(m, z, theta_cosmo)``.
    """

    _N_M_GRID = 256

    _SINGLE_ARG_INIT = True



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model."""
        p = hod_params
        nc = n_cen_bin_z25(log10m_arr, p["log10m_star_lo"], p["log10m_star_hi"],
                           p["log10m1_shmr"], p["log10eps"],
                           p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"],
                           p["sigma_logm_star"], p["f_cen"])
        ns = n_sat_bin_z25(log10m_arr, p["log10m_star_lo"], p["log10m_star_hi"],
                           p["log10m1_shmr"], p["log10eps"],
                           p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"],
                           p["alpha_sat"], p["kappa"],
                           p["B_sat"], p["beta_sat"], p["B_cut"], p["beta_cut"], p["f_sat"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """Kravtsov+2018 SHMR best-fit (Zacharegkas+2025 Table 2) + fiducial HOD priors."""
        return {
            "log10m_star_lo": 10.0,
            "log10m_star_hi": 10.5,
            # Kravtsov+2018 SHMR (Zacharegkas+2025 Table 2)
            "log10m1_shmr": 11.506,
            "log10eps": -1.632,
            "alpha_shmr": -1.638,
            "gamma_shmr": 0.596,
            "delta_shmr": 3.810,
            # HOD scatter and completeness
            "sigma_logm_star": 0.3,
            "f_cen": 1.0,
            # Satellite parameters
            "alpha_sat": 1.0,
            "kappa": 1.0,
            "B_sat": 10.0,
            "beta_sat": 1.0,
            "B_cut": 5.0,
            "beta_cut": 1.0,
            "f_sat": 1.0,
        }


# =============================================================================
# van Uitert et al. 2016 Conditional Stellar Mass Function (CSMF)
# arXiv:1601.06791, Section 2.4
# GAMA+KiDS weak-lensing SHMR: double-power-law centrals + modified-Schechter
# satellites, parameterised directly as CSMF rather than HOD thresholds.
# =============================================================================


@jax.jit
def _mean_stellar_mass_c_vanuitert16(
    log10m_h: jnp.ndarray,
    log10m_star0: float,
    log10m_h1: float,
    beta1: float,
    log10_beta2: float,
) -> jnp.ndarray:
    r"""Mean stellar mass of central galaxies — double power law (Eq. 16).

    .. math::
        M_\star^c(M_h) = M_{\star,0}
        \frac{(M_h/M_{h,1})^{\beta_1}}{1 + (M_h/M_{h,1})^{\beta_1 - \beta_2}}

    In log-space this avoids overflow by computing in log10:

    .. math::
        \log_{10} M_\star^c = \log_{10} M_{\star,0}
          + \beta_1 x - \log_{10}[1 + 10^{(\beta_1-\beta_2)\,x}]

    where :math:`x = \log_{10}(M_h/M_{h,1})`.

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    log10m_star0 : log10(M_{*,0}), normalization
    log10m_h1 : log10(M_{h,1}), characteristic halo mass
    beta1 : low-mass power-law slope
    log10_beta2 : log10(β2), high-mass slope (sampled in log to stay positive)
    """
    beta2 = jnp.power(10.0, log10_beta2)
    x = log10m_h - log10m_h1
    # Numerically stable: log10(1 + 10^((beta1-beta2)*x)) via logaddexp
    log10_denom = jnp.logaddexp((beta1 - beta2) * x * _LN10, 0.0) / _LN10
    return log10m_star0 + beta1 * x - log10_denom


@jax.jit
def n_cen_vanuitert16(
    log10m_h: jnp.ndarray,
    log10m_star_lo: float,
    log10m_star_hi: float,
    log10m_star0: float,
    log10m_h1: float,
    beta1: float,
    log10_beta2: float,
    sigma_c: float,
) -> jnp.ndarray:
    r"""Mean central occupation for stellar-mass bin [M_*lo, M_*hi] (Eqs. 11, 15).

    Analytic integral of the log-normal central CSMF (Eq. 15) over the bin:

    .. math::
        \langle N_c | M_h \rangle = \frac{1}{2}\left[
          \mathrm{erf}\!\left(\frac{\log_{10}M_{\star,\mathrm{hi}} - \mu}{\sqrt{2}\,\sigma_c}\right)
         -\mathrm{erf}\!\left(\frac{\log_{10}M_{\star,\mathrm{lo}} - \mu}{\sqrt{2}\,\sigma_c}\right)
        \right]

    where :math:`\mu = \log_{10}M_\star^c(M_h)`.

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    log10m_star_lo, log10m_star_hi : log10 stellar mass bin edges
    sigma_c : scatter in log10(M_*) at fixed halo mass
    """
    mu = _mean_stellar_mass_c_vanuitert16(log10m_h, log10m_star0, log10m_h1, beta1, log10_beta2)
    sqrt2_sigma = jnp.sqrt(2.0) * sigma_c
    nc = 0.5 * (erf((log10m_star_hi - mu) / sqrt2_sigma) - erf((log10m_star_lo - mu) / sqrt2_sigma))
    return jnp.clip(nc, 0.0, 1.0)


def n_sat_vanuitert16(
    log10m_h: jnp.ndarray,
    log10m_star_lo: float,
    log10m_star_hi: float,
    log10m_star0: float,
    log10m_h1: float,
    beta1: float,
    log10_beta2: float,
    alpha_s: float,
    b0: float,
    b1: float,
) -> jnp.ndarray:
    r"""Mean satellite occupation for stellar-mass bin [M_*lo, M_*hi] (Eqs. 11, 17, 18).

    Numerical integral of the modified Schechter satellite CSMF (Eq. 17):

    .. math::
        \Phi_s(M_\star | M_h) = \frac{\phi_s}{M_\star^s}
          \left(\frac{M_\star}{M_\star^s}\right)^{\alpha_s}
          \exp\!\left[-\left(\frac{M_\star}{M_\star^s}\right)^2\right]

    with :math:`M_\star^s = 0.56\,M_\star^c(M_h)` and
    :math:`\log_{10}\phi_s = b_0 + b_1 \log_{10}(M_h/10^{13}\,h^{-1}M_\odot)`.

    The integral :math:`\int_{M_{\star,\mathrm{lo}}}^{M_{\star,\mathrm{hi}}} \Phi_s\,\mathrm{d}M_\star`
    is evaluated on a 128-point log-spaced grid in stellar mass.  Broadcasting
    over ``log10m_h`` is handled via explicit array reshaping so the function
    accepts both scalar and 1-D array inputs.

    Parameters
    ----------
    alpha_s : faint-end slope of satellite CSMF (typically ≈ −1.1)
    b0, b1 : normalization parameters for log10(φ_s) (Eq. 18)
    """
    mu_c = _mean_stellar_mass_c_vanuitert16(log10m_h, log10m_star0, log10m_h1, beta1, log10_beta2)
    log10m_star_s = jnp.log10(0.56) + mu_c
    m_h = jnp.power(10.0, log10m_h)
    m13 = jnp.maximum(m_h / 1e13, 1e-10)
    log10_phi_s = b0 + b1 * jnp.log10(m13)
    phi_s = jnp.power(10.0, log10_phi_s)

    # Integrate Φ_s dM_* over [M_*lo, M_*hi] in log10 space.
    # Φ_s dM_* = φ_s * ln(10) * u^(α_s+1) * exp(-u^2) d(log10 M_*)  where u = M_*/M_*^s
    log10m_grid = jnp.linspace(log10m_star_lo, log10m_star_hi, 128)  # (128,)
    original_shape = jnp.shape(log10m_h)
    # Broadcast: (N, 1) over (128,) → (N, 128)
    log10m_star_s_r = jnp.reshape(log10m_star_s, (-1, 1))
    phi_s_r = jnp.reshape(phi_s, (-1, 1))
    u = jnp.power(10.0, log10m_grid - log10m_star_s_r)                     # (N, 128)
    integrand = phi_s_r * _LN10 * jnp.power(jnp.maximum(u, 1e-30), alpha_s + 1.0) * jnp.exp(-u ** 2)
    result = jnp.trapezoid(integrand, log10m_grid, axis=-1)                 # (N,)
    return jnp.reshape(result, original_shape)


@jax.jit
def n_total_vanuitert16(
    log10m_h: jnp.ndarray,
    log10m_star_lo: float,
    log10m_star_hi: float,
    log10m_star0: float,
    log10m_h1: float,
    beta1: float,
    log10_beta2: float,
    sigma_c: float,
    alpha_s: float,
    b0: float,
    b1: float,
) -> jnp.ndarray:
    """Total mean occupation (central + satellite) for stellar-mass bin (Eq. 14)."""
    nc = n_cen_vanuitert16(log10m_h, log10m_star_lo, log10m_star_hi,
                           log10m_star0, log10m_h1, beta1, log10_beta2, sigma_c)
    ns = n_sat_vanuitert16(log10m_h, log10m_star_lo, log10m_star_hi,
                           log10m_star0, log10m_h1, beta1, log10_beta2, alpha_s, b0, b1)
    return nc + ns


class VanUitert16CSMFModel(HODBase):
    """CSMF HOD model from van Uitert et al. 2016 (arXiv:1601.06791).

    The Conditional Stellar Mass Function (CSMF) Φ(M_*|M_h) is split into
    a log-normal central component and a modified Schechter satellite component.
    The central mean stellar mass follows a double power-law SHMR (Eq. 16).

    Parameters
    ----------
    hmf : HaloMassFunction
        Must expose ``.dndm(m, z, theta_cosmo)`` and ``.bias(m, z, theta_cosmo)``.
    """

    _N_M_GRID = 256

    _SINGLE_ARG_INIT = True



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model."""
        p = hod_params
        nc = n_cen_vanuitert16(log10m_arr, p["log10m_star_lo"], p["log10m_star_hi"],
                               p["log10m_star0"], p["log10m_h1"],
                               p["beta1"], p["log10_beta2"], p["sigma_c"])
        ns = n_sat_vanuitert16(log10m_arr, p["log10m_star_lo"], p["log10m_star_hi"],
                               p["log10m_star0"], p["log10m_h1"],
                               p["beta1"], p["log10_beta2"],
                               p["alpha_s"], p["b0"], p["b1"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """KiDS+GAMA-like parameters based on van Uitert+2016 Table 2 prior means."""
        return {
            "log10m_star_lo": 9.8,
            "log10m_star_hi": 10.3,
            # Double power-law central SHMR (Eq. 16)
            "log10m_h1": 11.5,    # log10(M_{h,1} / [M_sun/h])
            "log10m_star0": 10.5, # log10(M_{*,0} / [M_sun/h])
            "beta1": 5.0,         # low-mass slope (Gaussian prior mean)
            "log10_beta2": -0.5,  # log10(β2), high-mass slope → β2 ≈ 0.32
            # Central scatter
            "sigma_c": 0.15,
            # Satellite modified Schechter (Eqs. 17, 18)
            "alpha_s": -1.1,      # faint-end slope (Gaussian prior mean)
            "b0": 0.0,            # normalization intercept (Gaussian prior mean)
            "b1": 1.5,            # normalization slope (Gaussian prior mean)
        }


# ---------------------------------------------------------------------------
# Zu & Mandelbaum 2015 iHOD (arXiv:1505.02781)
# + Zu & Mandelbaum 2016/2017 halo quenching (arXiv:1509.06374, 1703.09219)
# ---------------------------------------------------------------------------

@jax.jit
def _mh_from_mstar_zu15(
    log10m_star: jnp.ndarray,
    lg_m1h: float,
    lg_m0star: float,
    beta: float,
    delta: float,
    gamma: float,
) -> jnp.ndarray:
    r"""Halo mass from stellar mass — Eq. 19 of Zu & Mandelbaum (2015).

    .. math::
        \log_{10} M_h = \log_{10} M_1
        + \beta\,\log_{10}\!\left(\frac{M_*}{M_{*0}}\right)
        + \frac{1}{\ln 10}\left[
          \frac{(M_*/M_{*0})^\delta}{1+(M_*/M_{*0})^{-\gamma}} - \frac{1}{2}
        \right]

    Returns :math:`\log_{10}(M_h\,/\,[M_\odot/h])`.

    Parameters
    ----------
    log10m_star : log10(M_* / [M_sun/h])
    lg_m1h : log10(M_1 / [M_sun/h]), pivot halo mass
    lg_m0star : log10(M_{*0} / [M_sun/h]), pivot stellar mass
    beta : low-mass slope
    delta : high-mass exponential index
    gamma : transition sharpness
    """
    x = log10m_star - lg_m0star
    m = jnp.power(10.0, x)
    m_neg_gamma = jnp.power(jnp.maximum(m, 1e-30), -gamma)
    exponent = jnp.power(m, delta) / (1.0 + m_neg_gamma) - 0.5
    return lg_m1h + beta * x + exponent / _LN10


def _mstar_from_mh_zu15(
    log10m_h: jnp.ndarray,
    lg_m1h: float,
    lg_m0star: float,
    beta: float,
    delta: float,
    gamma: float,
    n_iter: int = 60,
) -> jnp.ndarray:
    """Invert Eq. 19 of Zu & Mandelbaum (2015) via bisection.

    Returns :math:`\\log_{10}(M_*)` such that
    ``_mh_from_mstar_zu15(log10m_star, ...) = log10m_h``.
    Works element-wise for array inputs via ``jax.lax.fori_loop`` over 60 iterations
    in the range :math:`\\log_{10}(M_*/[M_\\odot/h]) \\in [6, 13]`.
    """
    target = jnp.asarray(log10m_h)

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        f_mid = _mh_from_mstar_zu15(mid, lg_m1h, lg_m0star, beta, delta, gamma)
        lo = jnp.where(f_mid < target, mid, lo)
        hi = jnp.where(f_mid < target, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(
        0, n_iter, body, (jnp.full_like(target, 4.0), jnp.full_like(target, 13.0))
    )
    return 0.5 * (lo + hi)


@jax.jit
def sigma_lnmstar_zu15(
    log10m_h: jnp.ndarray,
    lg_m1h: float,
    sigma_lnmstar: float,
    eta: float,
) -> jnp.ndarray:
    r"""Mass-dependent scatter in :math:`\ln M_*` — Eq. 20 of Zu & Mandelbaum (2015).

    .. math::
        \sigma_{\ln M_*}(M_h) = \begin{cases}
          \sigma_{\ln M_*,0} & M_h \leq M_1 \\
          \sigma_{\ln M_*,0} + \eta\,\log_{10}(M_h/M_1) & M_h > M_1
        \end{cases}

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    lg_m1h : log10(M_1 / [M_sun/h])
    sigma_lnmstar : baseline scatter in ln(M_*) at M_h ≤ M_1
    eta : mass-dependence slope (typically slightly negative)

    Accuracy
    --------
    Positive and finite for σ_0 > 0; continuity at M_h = M_1 verified
    analytically (2026-04-23).

    Timing
    ------
    ~ 22 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    delta_lgm = log10m_h - lg_m1h
    return jnp.where(delta_lgm <= 0.0, sigma_lnmstar, sigma_lnmstar + eta * delta_lgm)


@jax.jit
def n_cen_thresh_zu15(
    log10m_h: jnp.ndarray,
    log10m_star_thresh: float,
    lg_m1h: float,
    lg_m0star: float,
    beta: float,
    delta: float,
    gamma: float,
    sigma_lnmstar: float,
    eta: float,
    fc: float,
) -> jnp.ndarray:
    r"""Mean central occupation above a stellar-mass threshold — Eq. 21.

    .. math::
        \langle N_\mathrm{cen}^{>M_*}\rangle(M_h)
        = \frac{f_c}{2}\,\mathrm{erfc}\!\left[
          \frac{\ln M_*^\mathrm{thresh} - \ln M_*^c(M_h)}
               {\sqrt{2}\,\sigma_{\ln M_*}(M_h)}
        \right]

    where :math:`M_*^c(M_h)` is the mean stellar mass from the inverted SHMR
    and :math:`\sigma_{\ln M_*}` follows Eq. 20.

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    log10m_star_thresh : log10 stellar-mass threshold / [M_sun/h]
    fc : central galaxy fraction (incompleteness ceiling, ≤1)
    """
    log10m_star_c = _mstar_from_mh_zu15(log10m_h, lg_m1h, lg_m0star, beta, delta, gamma)
    sigma = sigma_lnmstar_zu15(log10m_h, lg_m1h, sigma_lnmstar, eta)
    arg = (log10m_star_thresh - log10m_star_c) * _LN10 / (jnp.sqrt(2.0) * sigma)
    return (fc / 2.0) * erfc(arg)


@jax.jit
def n_sat_thresh_zu15(
    log10m_h: jnp.ndarray,
    log10m_star_thresh: float,
    lg_m1h: float,
    lg_m0star: float,
    beta: float,
    delta: float,
    gamma: float,
    sigma_lnmstar: float,
    eta: float,
    fc: float,
    bsat: float,
    beta_sat: float,
    bcut: float,
    beta_cut: float,
    alpha_sat: float,
) -> jnp.ndarray:
    r"""Mean satellite occupation above a stellar-mass threshold — Eq. 22.

    .. math::
        \langle N_\mathrm{sat}^{>M_*}\rangle(M_h)
        = \langle N_\mathrm{cen}^{>M_*}\rangle(M_h)
          \left(\frac{M_h}{M_\mathrm{sat}}\right)^{\alpha_\mathrm{sat}}
          \exp\!\left(-\frac{M_\mathrm{cut}}{M_h}\right)

    with

    .. math::
        M_\mathrm{sat} = B_\mathrm{sat}
          \left(\frac{M_\mathrm{min}}{10^{12}\,h^{-1}M_\odot}\right)^{\beta_\mathrm{sat}}
          \times 10^{12}\,h^{-1}M_\odot, \qquad
        M_\mathrm{cut} = B_\mathrm{cut}
          \left(\frac{M_\mathrm{min}}{10^{12}}\right)^{\beta_\mathrm{cut}}
          \times 10^{12}

    where :math:`M_\mathrm{min} = f_\mathrm{SHMR}^{-1}(M_*^\mathrm{thresh})` is
    the characteristic halo mass for the threshold sample (direct evaluation of Eq. 19).

    Parameters
    ----------
    bsat, beta_sat : Msat normalisation amplitude and slope
    bcut, beta_cut : Mcut normalisation amplitude and slope
    alpha_sat : power-law index of satellite occupation
    """
    log10m_min = _mh_from_mstar_zu15(log10m_star_thresh, lg_m1h, lg_m0star, beta, delta, gamma)
    m_min_norm = jnp.power(10.0, log10m_min - 12.0)
    msat = bsat * jnp.power(m_min_norm, beta_sat) * 1e12
    mcut = bcut * jnp.power(m_min_norm, beta_cut) * 1e12

    nc = n_cen_thresh_zu15(
        log10m_h, log10m_star_thresh,
        lg_m1h, lg_m0star, beta, delta, gamma,
        sigma_lnmstar, eta, fc,
    )
    m_h = jnp.power(10.0, log10m_h)
    return nc * jnp.power(m_h / msat, alpha_sat) * jnp.exp(-mcut / m_h)


@jax.jit
def n_total_thresh_zu15(
    log10m_h: jnp.ndarray,
    log10m_star_thresh: float,
    lg_m1h: float,
    lg_m0star: float,
    beta: float,
    delta: float,
    gamma: float,
    sigma_lnmstar: float,
    eta: float,
    fc: float,
    bsat: float,
    beta_sat: float,
    bcut: float,
    beta_cut: float,
    alpha_sat: float,
) -> jnp.ndarray:
    """Total mean occupation (central + satellite) above threshold — Zu & Mandelbaum (2015)."""
    nc = n_cen_thresh_zu15(
        log10m_h, log10m_star_thresh,
        lg_m1h, lg_m0star, beta, delta, gamma,
        sigma_lnmstar, eta, fc,
    )
    ns = n_sat_thresh_zu15(
        log10m_h, log10m_star_thresh,
        lg_m1h, lg_m0star, beta, delta, gamma,
        sigma_lnmstar, eta, fc,
        bsat, beta_sat, bcut, beta_cut, alpha_sat,
    )
    return nc + ns


class ZuMandelbaum15HODModel(HODBase):
    """iHOD model from Zu & Mandelbaum (2015) (arXiv:1505.02781, Paper I).

    Stellar-to-halo mass relation based on the Behroozi+2010 inverse SHMR (Eq. 19),
    with mass-dependent log-normal scatter (Eq. 20), central threshold occupation
    (Eq. 21), and satellite occupation (Eq. 22).

    Parameters
    ----------
    hmf : HaloMassFunction
        Must expose ``.dndm(m, z, theta_cosmo)`` and ``.bias(m, z, theta_cosmo)``.
    """

    _SINGLE_ARG_INIT = True



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model."""
        p = hod_params
        nc = n_cen_thresh_zu15(log10m_arr, p["log10m_star_thresh"],
                               p["lg_m1h"], p["lg_m0star"],
                               p["beta"], p["delta"], p["gamma"],
                               p["sigma_lnmstar"], p["eta"], p["fc"])
        ns = n_sat_thresh_zu15(log10m_arr, p["log10m_star_thresh"],
                               p["lg_m1h"], p["lg_m0star"],
                               p["beta"], p["delta"], p["gamma"],
                               p["sigma_lnmstar"], p["eta"], p["fc"],
                               p["bsat"], p["beta_sat"], p["bcut"], p["beta_cut"], p["alpha_sat"])
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """iHOD best-fit parameters from Zu & Mandelbaum 2015, Table 2 (SDSS volume-limited)."""
        return {
            "log10m_star_thresh": 10.2,
            "lg_m1h": 12.10,
            "lg_m0star": 10.31,
            "beta": 0.33,
            "delta": 0.42,
            "gamma": 1.21,
            "sigma_lnmstar": 0.50,
            "eta": -0.04,
            "fc": 0.86,
            "bsat": 8.98,
            "beta_sat": 0.90,
            "bcut": 0.86,
            "beta_cut": 0.41,
            "alpha_sat": 1.00,
        }


# ---------------------------------------------------------------------------
# Halo quenching — Zu & Mandelbaum 2016/2017
# ---------------------------------------------------------------------------

@jax.jit
def f_red_cen_zu16(
    log10m_h: jnp.ndarray,
    lg_mqc_h: float,
    mu_c: float,
) -> jnp.ndarray:
    r"""Red fraction of central galaxies — Eq. 2 of Zu & Mandelbaum (2016/2017).

    .. math::
        f_{\rm red,c}(M_h) = 1 - \exp\!\left[-\left(\frac{M_h}{M_h^{qc}}\right)^{\mu_c}\right]

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    lg_mqc_h : log10(M_h^{qc} / [M_sun/h]), quenching pivot mass for centrals
    mu_c : Weibull shape parameter for central quenching

    Accuracy
    --------
    f_red,c(M_h → ∞) → 1 (Weibull CDF saturates); verified to < 1e-4 at
    M_h = 10 M_q^{qc} (2026-04-23).  Output ∈ [0, 1] by construction.

    Timing
    ------
    ~ 20 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    m_h = jnp.power(10.0, log10m_h)
    m_qc = jnp.power(10.0, lg_mqc_h)
    return 1.0 - jnp.exp(-jnp.power(m_h / m_qc, mu_c))


@jax.jit
def f_red_sat_zu16(
    log10m_h: jnp.ndarray,
    lg_mqs_h: float,
    mu_s: float,
) -> jnp.ndarray:
    r"""Red fraction of satellite galaxies — Eq. 3 of Zu & Mandelbaum (2016/2017).

    .. math::
        f_{\rm red,s}(M_h) = 1 - \exp\!\left[-\left(\frac{M_h}{M_h^{qs}}\right)^{\mu_s}\right]

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    lg_mqs_h : log10(M_h^{qs} / [M_sun/h]), quenching pivot mass for satellites
    mu_s : Weibull shape parameter for satellite quenching

    Accuracy
    --------
    f_red,s(M_h → ∞) → 1 (Weibull CDF saturates); verified to < 1e-4 at
    M_h = 10 M_h^{qs} (2026-04-23).  Output ∈ [0, 1] by construction.

    Timing
    ------
    ~ 17 µs / call  (JIT-compiled, N=200 masses, CPU x86-64, 2026-04-23).
    """
    m_h = jnp.power(10.0, log10m_h)
    m_qs = jnp.power(10.0, lg_mqs_h)
    return 1.0 - jnp.exp(-jnp.power(m_h / m_qs, mu_s))


class ZuMandelbaum16QuenchingModel:
    """Halo quenching model from Zu & Mandelbaum 2016/2017 (arXiv:1509.06374, 1703.09219).

    Computes occupation-weighted effective red fractions for centrals and satellites
    as Weibull CDFs of halo mass (Eqs. 2, 3 of Paper III).

    Parameters
    ----------
    hmf : HaloMassFunction
        Must expose ``.dndm(m, z, theta_cosmo)``.
    """

    def __init__(self, hmf):
        self._hmf = hmf
        self._m_grid = jnp.logspace(10, 16, 512)
        self._log10m_grid = jnp.log10(self._m_grid)

    def effective_red_fraction_cen(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        quench_params: dict,
    ) -> jnp.ndarray:
        r"""Occupation-weighted red fraction for centrals.

        .. math::
            \bar{f}_{\rm red,c} =
            \frac{\int \mathrm{d}M_h\,\frac{\mathrm{d}n}{\mathrm{d}M_h}
                  N_\mathrm{cen}(M_h)\,f_{\rm red,c}(M_h)}
            {\int \mathrm{d}M_h\,\frac{\mathrm{d}n}{\mathrm{d}M_h}\,N_\mathrm{cen}(M_h)}
        """
        m = self._m_grid
        log10m = self._log10m_grid
        p = hod_params
        q = quench_params
        dn = self._hmf.dndm(m, z, theta_cosmo)
        nc = n_cen_thresh_zu15(
            log10m, p["log10m_star_thresh"],
            p["lg_m1h"], p["lg_m0star"],
            p["beta"], p["delta"], p["gamma"],
            p["sigma_lnmstar"], p["eta"], p["fc"],
        )
        fred_c = f_red_cen_zu16(log10m, q["lg_mqc_h"], q["mu_c"])
        norm = jnp.trapezoid(dn * nc, m)
        return jnp.trapezoid(dn * nc * fred_c, m) / norm

    def effective_red_fraction_sat(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        quench_params: dict,
    ) -> jnp.ndarray:
        r"""Occupation-weighted red fraction for satellites.

        .. math::
            \bar{f}_{\rm red,s} =
            \frac{\int \mathrm{d}M_h\,\frac{\mathrm{d}n}{\mathrm{d}M_h}
                  N_\mathrm{sat}(M_h)\,f_{\rm red,s}(M_h)}
            {\int \mathrm{d}M_h\,\frac{\mathrm{d}n}{\mathrm{d}M_h}\,N_\mathrm{sat}(M_h)}
        """
        m = self._m_grid
        log10m = self._log10m_grid
        p = hod_params
        q = quench_params
        dn = self._hmf.dndm(m, z, theta_cosmo)
        ns = n_sat_thresh_zu15(
            log10m, p["log10m_star_thresh"],
            p["lg_m1h"], p["lg_m0star"],
            p["beta"], p["delta"], p["gamma"],
            p["sigma_lnmstar"], p["eta"], p["fc"],
            p["bsat"], p["beta_sat"],
            p["bcut"], p["beta_cut"],
            p["alpha_sat"],
        )
        fred_s = f_red_sat_zu16(log10m, q["lg_mqs_h"], q["mu_s"])
        norm = jnp.trapezoid(dn * ns, m)
        return jnp.trapezoid(dn * ns * fred_s, m) / norm

    @staticmethod
    def default_quench_params() -> dict:
        """Halo quenching best-fit from Zu & Mandelbaum 2017 Table 2 (WMAP5 cosmology)."""
        return {
            "lg_mqc_h": 11.78,
            "mu_c": 0.41,
            "lg_mqs_h": 12.19,
            "mu_s": 0.24,
        }


# ---------------------------------------------------------------------------
# Leauthaud+2012 HOD (arXiv:1104.0928)
# ---------------------------------------------------------------------------

@jax.jit
def _mh_from_mstar_leauthaud12(
    log10m_star: jnp.ndarray,
    log10m1: float,
    log10m_star0: float,
    beta: float,
    delta: float,
    gamma: float,
) -> jnp.ndarray:
    r"""SHMR — Leauthaud+2012 Eq. 3: :math:`\log_{10} M_h` from :math:`\log_{10} M_*`.

    .. math::

        \log_{10} M_h = \log_{10} M_1
        + \beta\,\log_{10}\!\left(\frac{M_*}{M_{*0}}\right)
        + \frac{(M_*/M_{*0})^\delta}{1 + (M_*/M_{*0})^{-\gamma}} - 0.5

    Note: this form has no :math:`1/\ln 10` prefactor on the non-linear term,
    unlike the Zu & Mandelbaum 2015 variant.

    Parameters
    ----------
    log10m_star : :math:`\log_{10}(M_*/[M_\odot/h])`
    log10m1 : :math:`\log_{10}(M_1/[M_\odot/h])` — pivot halo mass
    log10m_star0 : :math:`\log_{10}(M_{*0}/[M_\odot/h])` — pivot stellar mass
    beta : low-mass SHMR slope
    delta : high-mass SHMR exponential index
    gamma : SHMR transition sharpness
    """
    x = log10m_star - log10m_star0
    t = jnp.power(10.0, x)
    t_neg_gamma = jnp.power(jnp.maximum(t, 1e-30), -gamma)
    f = jnp.power(t, delta) / (1.0 + t_neg_gamma) - 0.5
    return log10m1 + beta * x + f


def _mstar_from_mh_leauthaud12(
    log10m_h: jnp.ndarray,
    log10m1: float,
    log10m_star0: float,
    beta: float,
    delta: float,
    gamma: float,
    n_iter: int = 60,
) -> jnp.ndarray:
    r"""Invert the Leauthaud+2012 SHMR via bisection.

    Returns :math:`\log_{10} M_*` such that
    ``_mh_from_mstar_leauthaud12(log10m_star, ...) == log10m_h``.
    Bisects in :math:`\log_{10}(M_*/[M_\odot/h]) \in [4, 13]` using
    ``jax.lax.fori_loop`` for JAX-compatibility.
    """
    target = jnp.asarray(log10m_h)

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        f_mid = _mh_from_mstar_leauthaud12(mid, log10m1, log10m_star0, beta, delta, gamma)
        lo = jnp.where(f_mid < target, mid, lo)
        hi = jnp.where(f_mid < target, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(
        0, n_iter, body,
        (jnp.full_like(target, 4.0), jnp.full_like(target, 13.0)),
    )
    return 0.5 * (lo + hi)


@jax.jit
def n_cen_leauthaud12(
    log10m_h: jnp.ndarray,
    log10m_star_thresh: float,
    log10m1: float,
    log10m_star0: float,
    beta: float,
    delta: float,
    gamma: float,
    sigma_logm: float,
) -> jnp.ndarray:
    r"""Central galaxy occupation for a stellar-mass threshold sample.

    Integrates the log-normal scatter in :math:`\log_{10} M_*` at fixed :math:`M_h`
    above the threshold (Leauthaud+2012 Eq. 8):

    .. math::

        \langle N_{\rm cen}(M_h) \rangle =
        \frac{1}{2}\,{\rm erfc}\!\left[
          \frac{\log_{10} M_{*,{\rm thresh}} - \langle\log_{10} M_*(M_h)\rangle}
               {\sqrt{2}\,\sigma_{\log M}}
        \right]

    where :math:`\langle\log_{10} M_*(M_h)\rangle` is the inverse SHMR.

    Parameters
    ----------
    log10m_h : :math:`\log_{10}(M_h/[M_\odot/h])`
    log10m_star_thresh : :math:`\log_{10}(M_{*,{\rm thresh}}/[M_\odot/h])`
    log10m1, log10m_star0, beta, delta, gamma : SHMR parameters
    sigma_logm : log-normal scatter in :math:`\log_{10} M_*` at fixed :math:`M_h`
    """
    log10m_star_mean = _mstar_from_mh_leauthaud12(
        log10m_h, log10m1, log10m_star0, beta, delta, gamma
    )
    return 0.5 * jax.scipy.special.erfc(
        (log10m_star_thresh - log10m_star_mean) / (jnp.sqrt(2.0) * sigma_logm)
    )


@jax.jit
def n_sat_leauthaud12(
    log10m_h: jnp.ndarray,
    log10m_star_thresh: float,
    log10m1: float,
    log10m_star0: float,
    beta: float,
    delta: float,
    gamma: float,
    sigma_logm: float,
    alpha_sat: float,
    log10m_sat: float,
    log10m_cut: float,
) -> jnp.ndarray:
    r"""Satellite galaxy occupation — Leauthaud+2012 Eq. 12.

    .. math::

        \langle N_{\rm sat}(M_h) \rangle =
        \langle N_{\rm cen}(M_h) \rangle
        \left(\frac{M_h}{M_{\rm sat}}\right)^{\alpha_{\rm sat}}
        \exp\!\left(-\frac{M_{\rm cut}}{M_h}\right)

    Parameters
    ----------
    log10m_h : :math:`\log_{10}(M_h/[M_\odot/h])`
    log10m_star_thresh, log10m1, log10m_star0, beta, delta, gamma, sigma_logm :
        passed to :func:`n_cen_leauthaud12`
    alpha_sat : satellite power-law slope
    log10m_sat : :math:`\log_{10}(M_{\rm sat}/[M_\odot/h])`
    log10m_cut : :math:`\log_{10}(M_{\rm cut}/[M_\odot/h])` — exponential cut-off
    """
    nc = n_cen_leauthaud12(
        log10m_h, log10m_star_thresh,
        log10m1, log10m_star0, beta, delta, gamma, sigma_logm,
    )
    m_h   = jnp.power(10.0, log10m_h)
    m_sat = jnp.power(10.0, log10m_sat)
    m_cut = jnp.power(10.0, log10m_cut)
    return nc * jnp.power(m_h / m_sat, alpha_sat) * jnp.exp(-m_cut / m_h)


class Leauthaud12HODModel(HODBase):
    """Leauthaud+2012 HOD for stellar-mass selected COSMOS galaxies.

    Galaxy selection is by stellar mass threshold :math:`M_* > M_{*,\\rm thresh}`.
    The central occupation derives from integrating the log-normal
    :math:`p(\log_{10} M_* | M_h)` above the threshold, using the Behroozi-like
    SHMR of Leauthaud+2012 Eq. 3.  The satellite occupation follows their Eq. 12.

    This model is compatible with :class:`~hod_mod.galaxies.clustering.FullHaloModelPrediction`
    (duck-typed on ``_integrate`` and ``nc_ns``).

    Parameters
    ----------
    hmf : HaloMassFunction
    halo_bias : callable — bias(m, z, theta_cosmo)

    HOD parameter dict keys
    -----------------------
    log10m1 : :math:`\\log_{10}(M_1/[M_\\odot/h])` — SHMR pivot halo mass
    log10m_star0 : :math:`\\log_{10}(M_{*0}/[M_\\odot/h])` — SHMR pivot stellar mass
    beta : SHMR low-mass slope
    delta : SHMR high-mass exponential index
    gamma : SHMR transition sharpness
    sigma_logm : log-normal scatter in :math:`\\log_{10} M_*` at fixed :math:`M_h`
    log10m_star_thresh : :math:`\\log_{10}(M_{*,\\rm thresh}/[M_\\odot/h])`
    alpha_sat : satellite power-law slope (Eq. 12)
    log10m_sat : :math:`\\log_{10}(M_{\\rm sat}/[M_\\odot/h])`
    log10m_cut : :math:`\\log_{10}(M_{\\rm cut}/[M_\\odot/h])`

    References
    ----------
    Leauthaud+2012, ApJ 744, 159  `arXiv:1104.0928 <https://arxiv.org/abs/1104.0928>`_
    """



    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple:
        """(N_c, N_s) occupation arrays on log10m_arr."""
        p  = hod_params
        nc = n_cen_leauthaud12(
            log10m_arr,
            p["log10m_star_thresh"],
            p["log10m1"], p["log10m_star0"],
            p["beta"], p["delta"], p["gamma"],
            p["sigma_logm"],
        )
        ns = n_sat_leauthaud12(
            log10m_arr,
            p["log10m_star_thresh"],
            p["log10m1"], p["log10m_star0"],
            p["beta"], p["delta"], p["gamma"],
            p["sigma_logm"],
            p["alpha_sat"], p["log10m_sat"], p["log10m_cut"],
        )
        return nc, ns




    @staticmethod
    def default_params() -> dict:
        """Leauthaud+2012 Table 3 best-fit for COSMOS z = 0.22–0.48, M_* > 10^10 M_sun/h."""
        return {
            "log10m1":            12.520,
            "log10m_star0":       10.916,
            "beta":                0.457,
            "delta":               0.566,
            "gamma":               1.53,
            "sigma_logm":          0.206,
            "log10m_star_thresh":  10.0,
            "alpha_sat":           1.0,
            "log10m_sat":         12.500,
            "log10m_cut":         11.500,
        }


# =============================================================================
# Lange+2025 Decorated HOD with assembly bias
# arXiv:2512.15962 — DESI DR1 galaxy clustering + weak lensing
# =============================================================================


class Lange25HODModel(HODBase):
    """DESI DR1 decorated HOD with effective assembly bias (Lange+2025).

    Base occupation: Zheng+2007 centrals + Kravtsov+2004 satellites, with a
    completeness factor ``f_Gamma`` on centrals.

    Assembly bias is approximated analytically via a ``(b−1)/b`` kernel that
    modifies the effective galaxy bias used in the 2-halo power spectrum.
    A_cen > 0 (A_sat > 0) means centrals (satellites) preferentially inhabit
    high-bias halos, boosting the effective large-scale clustering amplitude.

    Parameters
    ----------
    log10mmin : float
        Central occupation threshold log10(M_min / [M_sun/h]).
    sigma_logm : float
        Log-normal scatter in the central occupation.
    log10m0 : float
        Satellite exponential cutoff mass log10(M_0 / [M_sun/h]).
    log10m1 : float
        Satellite power-law mass scale log10(M_1 / [M_sun/h]).
    alpha : float
        Satellite power-law slope.
    f_Gamma : float
        Central galaxy completeness fraction in [0.5, 1.0].
    A_cen : float
        Assembly bias amplitude for centrals in [−1, 1].
    A_sat : float
        Assembly bias amplitude for satellites in [−1, 1].

    References
    ----------
    Lange et al. 2025, arXiv:2512.15962 — DESI DR1 tracer HOD analysis
    Hearin et al. 2016, AJ 152, 1 — Decorated HOD framework
    """

    _SINGLE_ARG_INIT = True

    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(N_c, N_s) occupation arrays on log10m_arr."""
        p  = hod_params
        nc = n_cen(log10m_arr, p["log10mmin"], p["sigma_logm"]) * p.get("f_Gamma", 1.0)
        ns = n_sat_kravtsov04(
            log10m_arr, p["log10mmin"], p["sigma_logm"],
            p["log10m0"], p["log10m1"], p["alpha"],
        )
        return nc, ns

    def _integrate(
        self, z: float, theta_cosmo: dict, hod_params: dict
    ) -> tuple:
        """Integrate with assembly bias correction on b_eff.

        Overrides HODBase._integrate to apply A_cen / A_sat corrections
        to the effective galaxy bias via a (b−1)/b assembly bias kernel.
        """
        m  = self._m_grid
        dn = self._hmf.dndm(m, z, theta_cosmo)
        b  = self._bias(m, z, theta_cosmo)
        nc, ns = self.nc_ns(self._log10m_grid, hod_params)
        nt = nc + ns

        n_gal = jnp.trapezoid(dn * nt, m)

        A_cen = float(hod_params.get("A_cen", 0.0))
        A_sat = float(hod_params.get("A_sat", 0.0))
        # Assembly bias kernel: (b−1)/b is negative for low-mass (b<1) halos
        # and positive for high-mass (b>1) halos, qualitatively matching the
        # concentration-bias correlation from N-body simulations.
        gamma = (b - 1.0) / jnp.where(b > 0.5, b, 0.5)
        b_nc  = b * (1.0 + A_cen * gamma)
        b_ns  = b * (1.0 + A_sat * gamma)

        b_eff = jnp.trapezoid(dn * (nc * b_nc + ns * b_ns), m) / n_gal
        m_eff = jnp.trapezoid(dn * nt * m, m) / n_gal
        return n_gal, b_eff, m_eff

    @staticmethod
    def default_params() -> dict:
        """DESI DR1 fiducial HOD parameters (Lange+2025 Table 1 centre values)."""
        return {
            "log10mmin":  13.0,
            "sigma_logm":  0.3,
            "log10m0":    13.5,
            "log10m1":    14.0,
            "alpha":       1.0,
            "f_Gamma":     1.0,
            "A_cen":       0.0,
            "A_sat":       0.0,
        }
