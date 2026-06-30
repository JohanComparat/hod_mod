"""Leauthaud et al. 2012 HOD.

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10




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
    r"""Leauthaud+2012 HOD for stellar-mass selected COSMOS galaxies.

    Galaxy selection is by stellar mass threshold :math:`M_* > M_{*,\\rm thresh}`.
    The central occupation derives from integrating the log-normal
    :math:`p(\log_{10} M_* | M_h)` above the threshold, using the Behroozi-like
    SHMR of Leauthaud+2012 Eq. 3.  The satellite occupation follows their Eq. 12.

    This model is compatible with :class:`~hod_mod.observables.clustering.FullHaloModelPrediction`
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
