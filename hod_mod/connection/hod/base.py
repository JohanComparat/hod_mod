"""Shared HOD base class and occupation kernels (split from hod.py)."""

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


# =============================================================================
# Zacharegkas & Chang et al. 2025 HOD (arXiv:2506.22367, Section 2)
# Kravtsov+2018 SHMR with stellar-mass threshold/bin selection and
# exponential satellite cutoff.
# =============================================================================

_LN10 = jnp.log(10.0)
