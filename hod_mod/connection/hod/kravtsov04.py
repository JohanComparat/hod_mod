"""Kravtsov et al. 2004 HOD (and AUM alias).

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10




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
