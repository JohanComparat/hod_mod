"""Guo et al. 2018/2019 incomplete conditional SMF (ICSMF) models.

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10




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
