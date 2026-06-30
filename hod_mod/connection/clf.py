"""Conditional Luminosity Function (CLF) halo model — Cacciato+2009/2013.

Parameterizes galaxy occupation by luminosity threshold using a log-normal CLF
for centrals and a Schechter CLF for satellites, both anchored to an SHMR.
Integrating over the CLF above a luminosity threshold yields N_cen(M) and
N_sat(M) suitable for the standard 1h+2h halo model integrals.

References
----------
Cacciato+2009  `arXiv:0807.4932 <https://arxiv.org/abs/0807.4932>`_
Cacciato+2013  `arXiv:1303.5445 <https://arxiv.org/abs/1303.5445>`_
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from .hod import HODBase


_LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# SHMR: central luminosity as a function of halo mass
# ---------------------------------------------------------------------------

@jax.jit
def log10_lc(
    log10m: jnp.ndarray,
    log10l0: float,
    log10m1: float,
    alpha_cen: float,
    beta_cen: float,
) -> jnp.ndarray:
    r"""Mean central luminosity vs halo mass — Cacciato+2009 Eq. 3.

    .. math::

        \log_{10} L_c(M) = \log_{10} L_0
        + \alpha_{\rm cen}\,\log_{10}\!\left(\frac{M}{M_1}\right)
        + (\beta_{\rm cen} - \alpha_{\rm cen})
          \,\log_{10}\!\left(1 + \frac{M}{M_1}\right)

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10l0 : :math:`\log_{10}(L_0/[L_\odot\,h^{-2}])` — central luminosity at M = M_1
    log10m1 : :math:`\log_{10}(M_1/[M_\odot/h])` — pivot halo mass
    alpha_cen : faint-end SHMR slope
    beta_cen : bright-end SHMR slope
    """
    x = log10m - log10m1   # log10(M / M_1)
    t = jnp.power(10.0, x)
    return log10l0 + alpha_cen * x + (beta_cen - alpha_cen) * jnp.log10(1.0 + t)


# ---------------------------------------------------------------------------
# Central CLF occupation
# ---------------------------------------------------------------------------

@jax.jit
def clf_central_mean(
    log10m: jnp.ndarray,
    log10l_lim: float,
    log10l0: float,
    log10m1: float,
    alpha_cen: float,
    beta_cen: float,
    sigma_c: float,
) -> jnp.ndarray:
    r"""Mean number of central galaxies above luminosity threshold.

    .. math::

        \langle N_{\rm cen}(M) \rangle =
        \frac{1}{2}\,{\rm erfc}\!\left[
          \frac{\log_{10} L_{\rm lim} - \log_{10} L_c(M)}
               {\sqrt{2}\,\sigma_c}
        \right]

    where :math:`L_c(M)` is the mean central luminosity from :func:`log10_lc`.

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10l_lim : :math:`\log_{10}` luminosity threshold
    log10l0, log10m1, alpha_cen, beta_cen : SHMR parameters (see :func:`log10_lc`)
    sigma_c : scatter in :math:`\log_{10} L_c` at fixed :math:`M` (log-normal width)
    """
    lc = log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen)
    return 0.5 * jax.scipy.special.erfc(
        (log10l_lim - lc) / (jnp.sqrt(2.0) * sigma_c)
    )


# ---------------------------------------------------------------------------
# Satellite CLF occupation
# ---------------------------------------------------------------------------

def _phi_s_star(
    log10m: jnp.ndarray,
    log10m1: float,
    sigma_c: float,
    b_sat: float,
) -> jnp.ndarray:
    r"""Satellite CLF normalization :math:`\Phi_s^*(M)` — Cacciato+2009 Eq. 5.

    .. math::

        \Phi_s^*(M) = \frac{b_{\rm sat}}{\sqrt{2\pi}\,\sigma_c}
                      \cdot \frac{M}{M_1}

    The normalization is proportional to the peak of the central log-normal
    CLF :math:`\Phi_c^* = 1/(\sqrt{2\pi}\,\sigma_c)` times the satellite
    amplitude :math:`b_{\rm sat}` and a power-law in halo mass.

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10m1 : :math:`\log_{10}(M_1/[M_\odot/h])`
    sigma_c : central log-normal width
    b_sat : satellite normalization amplitude
    """
    m_over_m1 = jnp.power(10.0, log10m - log10m1)
    return (b_sat / (jnp.sqrt(2.0 * jnp.pi) * sigma_c)) * m_over_m1


def _upper_gamma(a: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    r"""Unregularized upper incomplete gamma :math:`\Gamma(a, x) = \int_x^\infty t^{a-1} e^{-t} dt`.

    Handles :math:`a \in (-1, 0)` via the recurrence
    :math:`\Gamma(a, x) = (\Gamma(a+1, x) - x^a e^{-x})/a`, so the full
    range :math:`a \in (-1, \infty) \setminus \{0\}` is supported.

    Parameters
    ----------
    a : :math:`a = \alpha_s + 1`  (first argument of upper incomplete gamma)
    x : :math:`x = L_{\rm lim}/L_s^*(M)` (lower integration limit)
    """
    # --- branch for a > 0: direct regularized formula ---
    a_pos  = jnp.maximum(a, 1e-10)          # guard against a=0 pole in direct path
    g_direct = (
        jax.scipy.special.gamma(a_pos)
        * jax.scipy.special.gammaincc(a_pos, jnp.maximum(x, 0.0))
    )

    # --- branch for a in (-1, 0]: one-step recurrence ---
    # Γ(a, x) = (Γ(a+1, x) - x^a * exp(-x)) / a  with a+1 ∈ (0, 1]
    a1      = a + 1.0                       # a1 ∈ (0, 1] when a ∈ (-1, 0]
    a1_pos  = jnp.maximum(a1, 1e-10)
    g_a1    = (
        jax.scipy.special.gamma(a1_pos)
        * jax.scipy.special.gammaincc(a1_pos, jnp.maximum(x, 0.0))
    )
    x_safe  = jnp.maximum(x, 1e-30)         # x^a with a<0 → 0^a=inf; clamp x
    denom   = jnp.where(jnp.abs(a) > 1e-10, a, jnp.ones_like(a))
    g_recur = (g_a1 - jnp.power(x_safe, a) * jnp.exp(-x)) / denom

    return jnp.where(a > 0.0, g_direct, g_recur)


@jax.jit
def clf_satellite_mean(
    log10m: jnp.ndarray,
    log10l_lim: float,
    log10l0: float,
    log10m1: float,
    alpha_cen: float,
    beta_cen: float,
    sigma_c: float,
    alpha_sat: float,
    b_sat: float,
) -> jnp.ndarray:
    r"""Mean number of satellite galaxies above luminosity threshold.

    The satellite CLF is a **modified** Schechter function (Cacciato+2009 Eq. 36,
    per :math:`dL/L`):

    .. math::

        \Phi_s(L|M)\,\frac{dL}{L} =
        \Phi_s^*(M)\left(\frac{L}{L_c(M)}\right)^{\!\alpha_s}
        \exp\!\left[-\left(\frac{L}{L_c(M)}\right)^{\!2}\right]\frac{dL}{L}

    where :math:`L_s^*(M) = L_c(M)` is the mean central luminosity from
    :func:`log10_lc`.  Integrating over :math:`L > L_{\rm lim}` by substituting
    :math:`t = (L/L_c)^2` gives:

    .. math::

        \langle N_{\rm sat}(M) \rangle =
        \frac{\Phi_s^*(M)}{2}\,
        \Gamma\!\left(\frac{\alpha_s+1}{2},\,
                      \left(\frac{L_{\rm lim}}{L_c(M)}\right)^{\!2}\right)

    where :math:`\Gamma(a, x) = \int_x^\infty t^{a-1} e^{-t}\,dt` (unregularized).
    Values :math:`\alpha_s \in (-2, 0)` give :math:`a \in (-0.5, 0.5)`, fully
    covered by the one-step recurrence in :func:`_upper_gamma`.

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10l_lim : log10 luminosity threshold
    log10l0, log10m1, alpha_cen, beta_cen, sigma_c : SHMR/central CLF parameters
    alpha_sat : modified-Schechter faint-end slope :math:`\alpha_s`
    b_sat : satellite normalization amplitude
    """
    lc      = log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen)
    x       = jnp.power(10.0, log10l_lim - lc)       # L_lim / L_c(M)
    phi_s   = _phi_s_star(log10m, log10m1, sigma_c, b_sat)
    a       = (alpha_sat + 1.0) / 2.0                # (α_s+1)/2 for modified Schechter
    g_a     = _upper_gamma(a, x * x) / 2.0           # Γ((α_s+1)/2, x²) / 2
    return phi_s * jnp.maximum(g_a, 0.0)


# ---------------------------------------------------------------------------
# Cacciato+2013 quadratic satellite normalization
# ---------------------------------------------------------------------------

@jax.jit
def phi_s_star_cacciato13(
    log10m: jnp.ndarray,
    log10m1: float,
    b0: float,
    b1: float,
    b2: float,
) -> jnp.ndarray:
    r"""Satellite CLF normalization :math:`\Phi_s^*(M)` — Cacciato+2013 generalization.

    Replaces the single-parameter :func:`_phi_s_star` with a quadratic polynomial
    in :math:`x = \log_{10}(M/M_1)`:

    .. math::

        \log_{10}\Phi_s^*(M) = b_0 + b_1\,x + b_2\,x^2,
        \qquad x = \log_{10}(M/M_1)

    Best-fit values from Cacciato+2013 Table 1 (SDSS, all luminosities):
    :math:`b_0 = -1.17`, :math:`b_1 = 1.53`, :math:`b_2 = -0.217`.

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10m1 : pivot halo mass :math:`\log_{10}(M_1/[M_\odot/h])`
    b0, b1, b2 : polynomial coefficients
    """
    x = log10m - log10m1
    return jnp.power(10.0, b0 + b1 * x + b2 * x * x)


@jax.jit
def clf_satellite_mean_13(
    log10m: jnp.ndarray,
    log10l_lim: float,
    log10l0: float,
    log10m1: float,
    alpha_cen: float,
    beta_cen: float,
    alpha_sat: float,
    b0: float,
    b1: float,
    b2: float,
) -> jnp.ndarray:
    r"""Mean number of satellite galaxies — Cacciato+2013 quadratic normalization.

    Same modified Schechter integral as :func:`clf_satellite_mean` but uses the
    three-parameter quadratic satellite normalization :func:`phi_s_star_cacciato13`
    instead of the single-parameter :func:`_phi_s_star`.

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10l_lim : log10 luminosity threshold
    log10l0, log10m1, alpha_cen, beta_cen : SHMR/central CLF parameters
    alpha_sat : modified-Schechter faint-end slope :math:`\alpha_s`
    b0, b1, b2 : quadratic normalization coefficients (see :func:`phi_s_star_cacciato13`)
    """
    lc    = log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen)
    x     = jnp.power(10.0, log10l_lim - lc)
    phi_s = phi_s_star_cacciato13(log10m, log10m1, b0, b1, b2)
    a     = (alpha_sat + 1.0) / 2.0
    g_a   = _upper_gamma(a, x * x) / 2.0
    return phi_s * jnp.maximum(g_a, 0.0)


# ---------------------------------------------------------------------------
# van den Bosch+2013 helper functions (quadratic polynomial normalization)
# ---------------------------------------------------------------------------

def phi_sat_cacciato09(
    log10m: jnp.ndarray,
    b_0: float,
    b_1: float,
    b_2: float,
) -> jnp.ndarray:
    r"""Satellite CLF normalization phi_s*(M) — Cacciato+2009 Eq. 36.

    .. math::

        \phi_s^*(M) = 10^{b_0 + b_1\,(\log_{10} M - 12)
                                + b_2\,(\log_{10} M - 12)^2}

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    b_0, b_1, b_2 : quadratic polynomial coefficients in :math:`\log_{10} M - 12`
    """
    x = log10m - 12.0
    return jnp.power(10.0, b_0 + b_1 * x + b_2 * x**2)


@jax.jit
def alpha_sat_cacciato09(
    log10m: jnp.ndarray,
    a_1: float,
    a_2: float,
    log_m_2: float,
) -> jnp.ndarray:
    r"""Mass-dependent satellite CLF slope — Cacciato+2009.

    .. math::

        \alpha_s(M) = -2 + a_1\left[1 - \frac{2}{\pi}
                      \arctan\!\left(a_2\,(\log_{10} M - \log_{10} M_2)\right)
                      \right]

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    a_1 : amplitude of the slope transition
    a_2 : width of the transition
    log_m_2 : pivot mass :math:`\log_{10}(M_2/[M_\odot/h])`
    """
    return -2.0 + a_1 * (1.0 - 2.0 / jnp.pi * jnp.arctan(a_2 * (log10m - log_m_2)))


@jax.jit
def l_s_star_log10(
    log10m: jnp.ndarray,
    log10l0: float,
    log10m1: float,
    alpha_cen: float,
    beta_cen: float,
) -> jnp.ndarray:
    r"""Satellite CLF cutoff luminosity log10 L_s*(M) — Cacciato+2009 Eq. 38.

    .. math::

        L_s^*(M) = 0.562\,L_c(M)

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10l0, log10m1, alpha_cen, beta_cen : SHMR parameters (see :func:`log10_lc`)
    """
    return log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen) + jnp.log10(0.562)


@jax.jit
def clf_satellite_mean_vdB13(
    log10m: jnp.ndarray,
    log10l_lim: float,
    log10l0: float,
    log10m1: float,
    alpha_cen: float,
    beta_cen: float,
    b_0: float,
    b_1: float,
    b_2: float,
    alpha_sat: float,
) -> jnp.ndarray:
    r"""Mean satellite occupation — van den Bosch, More & Cacciato+2013 simplified form.

    Same CLF shape as :func:`clf_satellite_mean` but with a constant
    (mass-independent) satellite slope alpha_s and delta = 1:

    .. math::

        \langle N_{\rm sat}(M) \rangle =
        \frac{\phi_s^*(M)}{2}
        \,\Gamma\!\left(\frac{\alpha_s+1}{2},\,
                        \left(\frac{L_{\rm lim}}{L_s^*(M)}\right)^2\right)

    Parameters
    ----------
    log10m : :math:`\log_{10}(M/[M_\odot/h])`
    log10l_lim : luminosity threshold in :math:`\log_{10}` units
    log10l0, log10m1, alpha_cen, beta_cen : SHMR parameters
    b_0, b_1, b_2 : satellite normalization polynomial (Eq. 36)
    alpha_sat : constant satellite slope (scalar)
    """
    ls_log10 = l_s_star_log10(log10m, log10l0, log10m1, alpha_cen, beta_cen)
    phi_s    = phi_sat_cacciato09(log10m, b_0, b_1, b_2)
    x_sq     = jnp.power(10.0, 2.0 * (log10l_lim - ls_log10))
    a_half   = (alpha_sat + 1.0) / 2.0
    g        = _upper_gamma(jnp.full_like(x_sq, a_half), x_sq)
    return phi_s * 0.5 * jnp.maximum(g, 0.0)


# ---------------------------------------------------------------------------
# CLF model class
# ---------------------------------------------------------------------------

class CLFModel:
    r"""Cacciato+2009/2013 Conditional Luminosity Function occupation model.

    Parameterizes galaxy luminosity using a log-normal CLF for centrals and a
    Schechter CLF for satellites, both anchored to a two-slope SHMR:

    .. math::

        \log_{10} L_c(M) = \log_{10} L_0
        + \alpha_{\rm cen}\,\log_{10}\!\left(\frac{M}{M_1}\right)
        + (\beta_{\rm cen} - \alpha_{\rm cen})
          \log_{10}\!\left(1 + \frac{M}{M_1}\right)

    Integrating each CLF above a luminosity threshold :math:`L_{\rm lim}` yields
    :math:`N_{\rm cen}(M)` and :math:`N_{\rm sat}(M)`.  These occupy the same duck-typed
    interface as all HOD classes and can be passed directly to
    :class:`~hod_mod.observables.clustering.FullHaloModelPrediction`.

    Parameters
    ----------
    hmf : HaloMassFunction
    halo_bias : callable — ``bias(m, z, theta_cosmo)``

    CLF parameter dict keys
    -----------------------
    log10m1 : :math:`\log_{10}(M_1/[M_\odot/h])` — pivot halo mass at :math:`L_0`
    log10l0 : :math:`\log_{10}(L_0/[L_\odot\,h^{-2}])` — central luminosity at :math:`M_1`
    alpha_cen : faint-end SHMR slope
    beta_cen : bright-end SHMR slope
    sigma_c : scatter in :math:`\log_{10} L_c` at fixed :math:`M`
    alpha_sat : modified-Schechter faint-end slope for satellite CLF
    b_sat : satellite normalization amplitude (single-parameter 2009 form)
    log10l_lim : :math:`\log_{10}(L_{\rm lim}/[L_\odot\,h^{-2}])` — luminosity threshold

    References
    ----------
    Cacciato+2009  `arXiv:0807.4932 <https://arxiv.org/abs/0807.4932>`_
    Cacciato+2013  `arXiv:1303.5445 <https://arxiv.org/abs/1303.5445>`_
    """

    def __init__(self, hmf, halo_bias):
        self._hmf          = hmf
        self._bias         = halo_bias
        self._m_grid       = jnp.logspace(10, 16, 512)
        self._log10m_grid  = jnp.log10(self._m_grid)

    def _integrate(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> tuple:
        """Return (n_gal, b_eff, M_eff) for a luminosity-threshold CLF sample."""
        m      = self._m_grid
        log10m = self._log10m_grid
        dn     = self._hmf.dndm(m, z, theta_cosmo)
        b      = self._bias(m, z, theta_cosmo)
        nc, ns = self.nc_ns(log10m, hod_params)
        nt     = nc + ns
        n_gal  = jnp.trapezoid(dn * nt, m)
        b_eff  = jnp.trapezoid(dn * nt * b, m) / n_gal
        m_eff  = jnp.trapezoid(dn * nt * m, m) / n_gal
        return n_gal, b_eff, m_eff

    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple:
        r"""(N_c, N_s) CLF occupation arrays on log10m_arr.

        Parameters
        ----------
        log10m_arr : :math:`\log_{10}(M/[M_\odot/h])`, shape (NM,)
        hod_params : CLF parameter dict (see class docstring)

        Returns
        -------
        (N_c, N_s) : tuple of jnp.ndarray, shape (NM,)
        """
        p  = hod_params
        nc = clf_central_mean(
            log10m_arr,
            p["log10l_lim"],
            p["log10l0"], p["log10m1"],
            p["alpha_cen"], p["beta_cen"],
            p["sigma_c"],
        )
        ns = clf_satellite_mean(
            log10m_arr,
            p["log10l_lim"],
            p["log10l0"], p["log10m1"],
            p["alpha_cen"], p["beta_cen"],
            p["sigma_c"],
            p["alpha_sat"], p["b_sat"],
        )
        return nc, ns

    def galaxy_number_density(self, z, theta_cosmo, hod_params) -> jnp.ndarray:
        """n_gal [h³ Mpc⁻³]."""
        n, _, _ = self._integrate(z, theta_cosmo, hod_params)
        return n

    def effective_bias(self, z, theta_cosmo, hod_params) -> jnp.ndarray:
        """Effective large-scale galaxy bias b_eff."""
        _, b, _ = self._integrate(z, theta_cosmo, hod_params)
        return b

    def effective_mass(self, z, theta_cosmo, hod_params) -> jnp.ndarray:
        """Effective halo mass ⟨M⟩ [M_sun/h]."""
        _, _, m = self._integrate(z, theta_cosmo, hod_params)
        return m

    @staticmethod
    def default_params() -> dict:
        """Cacciato+2009 Table 1 fiducial CLF parameters (L* galaxies, SDSS r-band)."""
        return {
            "log10m1":    11.0,
            "log10l0":    9.94,
            "alpha_cen":  2.95,
            "beta_cen":   0.18,
            "sigma_c":    0.15,
            "alpha_sat":  -1.15,
            "b_sat":      9.0,
            "log10l_lim": 9.5,
        }


class CLFModel13(CLFModel):
    r"""Cacciato+2013 CLF model — quadratic satellite normalization.

    Extends :class:`CLFModel` by replacing the single-parameter satellite
    normalization :math:`\Phi_s^*(M) \propto M/M_1` with the three-parameter
    quadratic polynomial of Cacciato+2013:

    .. math::

        \log_{10}\Phi_s^*(M) = b_0 + b_1\,x + b_2\,x^2,
        \quad x = \log_{10}(M/M_1)

    All other equations (central CLF, SHMR, modified-Schechter satellite integral)
    are inherited from :class:`CLFModel` unchanged.

    CLF parameter dict keys
    -----------------------
    log10m1 : pivot halo mass
    log10l0 : central luminosity at :math:`M_1`
    alpha_cen : faint-end SHMR slope
    beta_cen : bright-end SHMR slope
    sigma_c : central log-normal scatter
    alpha_sat : modified-Schechter faint-end slope
    b0, b1, b2 : quadratic satellite normalization coefficients
    log10l_lim : luminosity threshold

    References
    ----------
    Cacciato+2013  `arXiv:1303.5445 <https://arxiv.org/abs/1303.5445>`_
    van den Bosch+2013  `arXiv:1204.1326 <https://arxiv.org/abs/1204.1326>`_
    """

    def nc_ns(
        self, log10m_arr: jnp.ndarray, hod_params: dict
    ) -> tuple:
        r"""(N_c, N_s) CLF occupation arrays using Cacciato+2013 satellite normalization.

        Parameters
        ----------
        log10m_arr : :math:`\log_{10}(M/[M_\odot/h])`, shape (NM,)
        hod_params : CLF parameter dict (see class docstring)

        Returns
        -------
        (N_c, N_s) : tuple of jnp.ndarray, shape (NM,)
        """
        p  = hod_params
        nc = clf_central_mean(
            log10m_arr,
            p["log10l_lim"],
            p["log10l0"], p["log10m1"],
            p["alpha_cen"], p["beta_cen"],
            p["sigma_c"],
        )
        ns = clf_satellite_mean_13(
            log10m_arr,
            p["log10l_lim"],
            p["log10l0"], p["log10m1"],
            p["alpha_cen"], p["beta_cen"],
            p["alpha_sat"],
            p["b0"], p["b1"], p["b2"],
        )
        return nc, ns

    @staticmethod
    def default_params() -> dict:
        """Cacciato+2013 Table 1 best-fit CLF parameters (SDSS, all luminosities)."""
        return {
            "log10m1":    11.24,
            "log10l0":    9.95,
            "alpha_cen":  3.18,
            "beta_cen":   0.245,
            "sigma_c":    0.157,
            "alpha_sat":  -1.18,
            "b0":         -1.17,
            "b1":          1.53,
            "b2":         -0.217,
            "log10l_lim":  9.5,
        }


class VanDenBosch13CLFModel(HODBase):
    r"""Simplified CLF model — van den Bosch, More & Cacciato+2013.

    Same central CLF as :class:`CLFModel` (Cacciato+2009) but with a constant
    (mass-independent) satellite slope alpha_s and delta = 1.
    This parametrization is used in:

    * **van den Bosch, More & Cacciato+2013** (MNRAS 430, 725) — theoretical framework
    * **Cacciato+2013** (MNRAS 430, 767) — SDSS application
    * **Cacciato & van Uitert+2014** (MNRAS 437, 377) — RCS2+SDSS application

    CLF parameter dict keys
    -----------------------
    log10m1, log10l0, alpha_cen, beta_cen, sigma_c : central CLF (same as CLFModel)
    b_0, b_1, b_2 : satellite normalization polynomial (Eq. 36 of Cacciato+2009)
    alpha_sat : constant satellite slope alpha_s
    log10l_lim : luminosity threshold

    References
    ----------
    van den Bosch+2013  `MNRAS 430, 725 <https://arxiv.org/abs/1207.0503>`_
    Cacciato+2013  `MNRAS 430, 767 <https://arxiv.org/abs/1207.0503>`_
    Cacciato & van Uitert+2014  `MNRAS 437, 377 <https://arxiv.org/abs/1307.6070>`_
    """

    def nc_ns(self, log10m_arr: jnp.ndarray, hod_params: dict) -> tuple:
        r"""(N_c, N_s) CLF occupation arrays on log10m_arr.

        Parameters
        ----------
        log10m_arr : log10(M / [M_sun/h]), shape (NM,)
        hod_params : CLF parameter dict (see class docstring)

        Returns
        -------
        (N_c, N_s) : tuple of jnp.ndarray, shape (NM,)
        """
        p  = hod_params
        nc = clf_central_mean(
            log10m_arr,
            p["log10l_lim"],
            p["log10l0"], p["log10m1"],
            p["alpha_cen"], p["beta_cen"],
            p["sigma_c"],
        )
        ns = clf_satellite_mean_vdB13(
            log10m_arr,
            p["log10l_lim"],
            p["log10l0"], p["log10m1"],
            p["alpha_cen"], p["beta_cen"],
            p["b_0"], p["b_1"], p["b_2"],
            p["alpha_sat"],
        )
        return nc, ns

    @staticmethod
    def default_params() -> dict:
        """van den Bosch+2013 L250 mock fiducial CLF parameters (Table 1)."""
        return {
            "log10m1":    10.9,
            "log10l0":     9.9,
            "alpha_cen":   5.0,
            "beta_cen":    0.24,
            "sigma_c":     0.16,
            "b_0":        -1.2,
            "b_1":         1.4,
            "b_2":        -0.17,
            "alpha_sat":  -1.3,
            "log10l_lim":  9.5,
        }

    @staticmethod
    def default_params_cacciato13() -> dict:
        """Cacciato+2013 SDSS WMAP7 best-fit CLF parameters (Table 5)."""
        return {
            "log10m1":    11.24,
            "log10l0":     9.94,
            "alpha_cen":   3.273,
            "beta_cen":    0.255,
            "sigma_c":     0.157,
            "b_0":        -0.766,
            "b_1":         1.008,
            "b_2":        -0.094,
            "alpha_sat":  -1.18,
            "log10l_lim":  9.5,
        }

    @staticmethod
    def default_params_cacciato14() -> dict:
        """Cacciato & van Uitert+2014 RCS2+SDSS best-fit (MNRAS 437, 377)."""
        return {
            "log10m1":    11.07,
            "log10l0":     9.935,
            "alpha_cen":   3.18,
            "beta_cen":    0.255,
            "sigma_c":     0.146,
            "b_0":        -0.766,
            "b_1":         1.008,
            "b_2":        -0.094,
            "alpha_sat":  -1.18,
            "log10l_lim":  9.5,
        }


class BASILISKCLFModel(VanDenBosch13CLFModel):
    r"""Flexible CLF with mass-dependent shape parameters — BASILISK III.

    Extends :class:`VanDenBosch13CLFModel` by allowing the central scatter
    :math:`\sigma_c`, the satellite faint-end slope :math:`\alpha_s`, and the
    satellite cutoff luminosity offset :math:`\Delta_s \equiv \log_{10}(L_s^*/L_c)`
    to vary linearly (and optionally quadratically) with halo mass:

    .. math::

        f(M) = f_{13} + f_{p1}\,\log_{10}\!\left(\frac{M}{10^{13}}\right)
               + f_{p2}\,\left[\log_{10}\!\left(\frac{M}{10^{13}}\right)\right]^2

    Setting all slope parameters to zero and ``delta_13 = log10(0.562) ≈ -0.25``
    reproduces :class:`VanDenBosch13CLFModel` exactly (BASILISK III Model A).

    Six models from Mitra & van den Bosch+2025 (arXiv:2510.08421, Table 1):

    * **Model A** — all constant, ``delta_13`` fixed at ``log10(0.562)``.  9 CLF params.
    * **Model B** — mass-dependent :math:`\sigma_c`.  10 params (``sigma_p1 ≠ 0``).
    * **Model C** — mass-dependent :math:`\alpha_s`.  11 params (``alpha_p1 ≠ 0``).
    * **Model D** — :math:`\Delta_s` free constant.  12 params (``delta_13`` free).
    * **Model E** — mass-dependent :math:`\Delta_s`.  13 params (``delta_p1 ≠ 0``).
    * **Model F** — all three quadratic.  16 params.

    CLF parameter dict keys
    -----------------------
    log10m1, log10l0, alpha_cen, beta_cen : SHMR (same as VanDenBosch13CLFModel)
    b_0, b_1, b_2 : satellite normalization polynomial
    log10l_lim : luminosity threshold
    sigma_13 : :math:`\sigma_c` at :math:`M = 10^{13}\,M_\odot/h`
    sigma_p1 : linear slope of :math:`\sigma_c(M)` (0 → constant)
    sigma_p2 : quadratic slope (0 → linear)
    alpha_13 : :math:`\alpha_s` at :math:`M = 10^{13}\,M_\odot/h`
    alpha_p1 : linear slope of :math:`\alpha_s(M)` (0 → constant)
    alpha_p2 : quadratic slope (0 → linear)
    delta_13 : :math:`\Delta_s = \log_{10}(L_s^*/L_c)` at :math:`M = 10^{13}`;
               default ``log10(0.562) ≈ -0.2503`` reproduces VanDenBosch13CLFModel
    delta_p1 : linear slope of :math:`\Delta_s(M)` (0 → constant)
    delta_p2 : quadratic slope (0 → linear)

    References
    ----------
    Mitra & van den Bosch+2025  `arXiv:2510.08421 <https://arxiv.org/abs/2510.08421>`_
    """

    def nc_ns(self, log10m_arr: jnp.ndarray, hod_params: dict) -> tuple:
        r"""(N_c, N_s) with mass-dependent :math:`\sigma_c`, :math:`\alpha_s`, :math:`\Delta_s`.

        All three shape parameters may vary linearly (or quadratically) with
        :math:`\log_{10}(M/10^{13})`.  Because :func:`_upper_gamma` is fully
        vectorised via :func:`jnp.where`, no ``vmap`` is needed.

        Parameters
        ----------
        log10m_arr : log10(M / [M_sun/h]), shape (NM,)
        hod_params : CLF parameter dict (see class docstring)

        Returns
        -------
        (N_c, N_s) : tuple of jnp.ndarray, shape (NM,)
        """
        p = hod_params
        log_m13 = log10m_arr - 13.0

        sigma_c = (p["sigma_13"]
                   + p.get("sigma_p1", 0.0) * log_m13
                   + p.get("sigma_p2", 0.0) * log_m13 ** 2)

        alpha_s = (p["alpha_13"]
                   + p.get("alpha_p1", 0.0) * log_m13
                   + p.get("alpha_p2", 0.0) * log_m13 ** 2)

        delta_s = (p["delta_13"]
                   + p.get("delta_p1", 0.0) * log_m13
                   + p.get("delta_p2", 0.0) * log_m13 ** 2)

        nc = clf_central_mean(
            log10m_arr, p["log10l_lim"],
            p["log10l0"], p["log10m1"],
            p["alpha_cen"], p["beta_cen"],
            sigma_c,
        )

        log10_ls = (log10_lc(log10m_arr, p["log10l0"], p["log10m1"],
                             p["alpha_cen"], p["beta_cen"])
                    + delta_s)
        x  = jnp.power(10.0, p["log10l_lim"] - log10_ls)
        a  = (alpha_s + 1.0) / 2.0
        g  = _upper_gamma(a, x * x) / 2.0
        ns = phi_sat_cacciato09(log10m_arr, p["b_0"], p["b_1"], p["b_2"]) * g

        return nc, ns

    @staticmethod
    def default_params() -> dict:
        """BASILISK III Model A — van den Bosch+2013 L250 fiducial with BASILISK parametrization."""
        return {
            "log10m1":    10.9,
            "log10l0":     9.9,
            "alpha_cen":   5.0,
            "beta_cen":    0.24,
            "b_0":        -1.2,
            "b_1":         1.4,
            "b_2":        -0.17,
            "log10l_lim":  9.5,
            "sigma_13":    0.16,
            "sigma_p1":    0.0,
            "sigma_p2":    0.0,
            "alpha_13":   -1.3,
            "alpha_p1":    0.0,
            "alpha_p2":    0.0,
            "delta_13":   float(np.log10(0.562)),   # = log10(0.562) ≈ -0.2503
            "delta_p1":    0.0,
            "delta_p2":    0.0,
        }
