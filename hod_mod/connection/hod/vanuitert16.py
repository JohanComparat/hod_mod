"""van Uitert et al. 2016 conditional SMF model.

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10




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
