"""More et al. 2015 HOD (mass-dependent and constant incompleteness).

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10




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
# More+2015 occupation with a CONSTANT duty cycle f_inc (mass-independent).
#
# Unlike :func:`_incompleteness_more15` — which is a *mass-dependent* linear
# ramp clip(1 + alpha_inc*(log10 M - log10 M_inc), 0, 1) — here ``f_inc`` is a
# single scalar in [0, 1] that multiplies the whole occupation uniformly. This
# is the "duty cycle" interpretation used by the HOD-based AGN model
# (:class:`~hod_mod.agn.hod.HODAgnModel`): a constant fraction f_inc of
# the host halos carry an active AGN at any time, with no halo-mass dependence.
# ---------------------------------------------------------------------------

@jax.jit
def n_cen_more15_const_finc(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    f_inc: float,
) -> jnp.ndarray:
    """More+2015 central count with a constant duty cycle.

    N_cen(M) = f_inc * (1/2) * erfc[(log10 M_min - log10 M) / sigma_logM]

    ``f_inc`` is a mass-independent scalar in [0, 1] (the duty cycle).
    """
    return f_inc * 0.5 * erfc((log10mmin - log10m) / sigma_logm)


@jax.jit
def n_sat_more15_const_finc(
    log10m: jnp.ndarray,
    log10mmin: float,
    sigma_logm: float,
    log10m1: float,
    alpha: float,
    kappa: float,
    f_inc: float,
) -> jnp.ndarray:
    """More+2015 satellite count with a constant duty cycle.

    N_sat(M) = N_cen(M) * [(M - kappa*M_min) / M_1]^alpha  for M > kappa*M_min

    The duty cycle enters through N_cen, so f_inc multiplies the full
    central+satellite occupation uniformly.
    """
    nc = n_cen_more15_const_finc(log10m, log10mmin, sigma_logm, f_inc)
    m = jnp.power(10.0, log10m)
    mmin = jnp.power(10.0, log10mmin)
    m1 = jnp.power(10.0, log10m1)
    m_thresh = kappa * mmin
    ratio = jnp.where(m > m_thresh, (m - m_thresh) / m1, 0.0)
    return nc * ratio**alpha


class MoreConstFincHODModel(HODBase):
    """More+2015 5-parameter HOD with a constant duty cycle ``f_inc``.

    Identical functional form to :class:`MoreHODModel` but with a
    mass-independent scalar duty cycle (``f_inc``) replacing the mass-dependent
    incompleteness ``(alpha_inc, log10m_inc)``.  Used to populate halos with
    AGN in :class:`~hod_mod.agn.hod.HODAgnModel`.

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
        nc = n_cen_more15_const_finc(log10m_arr, p["log10mmin"], p["sigma_logm"],
                                     p["f_inc"])
        ns = n_sat_more15_const_finc(log10m_arr, p["log10mmin"], p["sigma_logm"],
                                     p["log10m1"], p["alpha"], p["kappa"],
                                     p["f_inc"])
        return nc, ns

    @staticmethod
    def default_params() -> dict:
        """Default AGN-HOD parameters (constant duty cycle).

        log10m1 defaults to log10mmin + 1.5.
        """
        return {
            "log10mmin": 12.5,
            "sigma_logm": 0.8,
            "log10m1": 14.0,
            "alpha": 0.8,
            "kappa": 0.3,
            "f_inc": 0.1,
        }
