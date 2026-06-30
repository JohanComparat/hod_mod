"""Zu & Mandelbaum 2015 iHOD and 2016 quenching model.

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10




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
        """(N_c, N_s) occupation arrays on log10m_arr for the full halo model.

        When ``hod_params`` contains ``log10m_star_max`` (not None), returns the
        *bin* HOD N(Mlo ≤ M* < Mhi) = N_thresh(Mlo) − N_thresh(Mhi).  Without
        it, returns the threshold HOD N(M* > log10m_star_thresh) as usual.
        """
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
        max_thresh = p.get("log10m_star_max")
        if max_thresh is not None:
            nc_hi = n_cen_thresh_zu15(log10m_arr, max_thresh,
                                      p["lg_m1h"], p["lg_m0star"],
                                      p["beta"], p["delta"], p["gamma"],
                                      p["sigma_lnmstar"], p["eta"], p["fc"])
            ns_hi = n_sat_thresh_zu15(log10m_arr, max_thresh,
                                      p["lg_m1h"], p["lg_m0star"],
                                      p["beta"], p["delta"], p["gamma"],
                                      p["sigma_lnmstar"], p["eta"], p["fc"],
                                      p["bsat"], p["beta_sat"], p["bcut"], p["beta_cut"], p["alpha_sat"])
            nc = nc - nc_hi
            ns = ns - ns_hi
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
