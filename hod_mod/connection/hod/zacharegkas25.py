"""Zacharegkas et al. 2025 HOD.

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10




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
