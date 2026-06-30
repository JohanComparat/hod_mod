"""Lange et al. 2025 assembly-bias HOD.

Split from the original ``hod.py``; see :mod:`hod_mod.connection.hod`.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erf

from .base import HODBase, HODModel, n_cen, n_sat, n_total, _LN10
from .kravtsov04 import n_sat_kravtsov04




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
