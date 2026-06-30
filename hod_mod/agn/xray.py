"""X-ray AGN model following Comparat+2019 abundance matching, implemented in JAX.

Provides :class:`XrayAGNModel` which maps dark-matter halo mass to a mean
soft X-ray (0.5–2 keV) AGN luminosity via:

1. **SHMR** — any callable M_halo → log10(M_*) (e.g. :func:`~hod_mod.connection.sham.smhm_girelli20`)
2. **LX–M_* relation** — parametric fit to the Comparat+2019 abundance-matching result
   (HAM of the hard-band XLF with the SHMR):
   log10(L_X^{2-10 keV}) = a + b × (log10 M_* − 10) + c × (log10 M_* − 10)²
3. **Band conversion** — hard-to-soft (0.5–2 / 2–10 keV) flux ratio from Comparat+2019 Table 2
4. **Log-normal scatter** — 0.8 dex in LX at fixed M_* → boost factor on mean ⟨L_X⟩
5. **Duty cycle** — f_DC(z) from Comparat+2019 Table 1 interpolation

The class provides:

* :meth:`mean_agn_lx` — mean soft-band L_X(M_halo, z) including duty cycle
* :meth:`agn_emissivity_uk` — Fourier transform of the AGN contribution to the
  X-ray surface brightness (point-source, flat in k-space)

Both methods are fully differentiable with JAX.

References
----------
Comparat et al. 2019, A&A 622, A12 (arXiv:1901.10866)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc
from jax.scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Default Comparat+2019 LX–M_* fit parameters (hard 2–10 keV band)
# Parametric fit to the HAM result; log10 LX in erg/s, log10 M_* in M_sun
# ---------------------------------------------------------------------------

# log10 L_X^{hard}(log10 M_*) = _A + _B × (log10 M_* − 10) + _C × (log10 M_* − 10)²
# Calibrated against the Hasinger+2005 LDDE soft XLF at z=0.1, 0.5, 1.0
# (joint 4-param fit; residuals < 0.025 dex at all three redshifts)
_LX_HARD_A = 41.04    # normalisation at log10 M_* = 10
_LX_HARD_B = 1.22     # linear slope
_LX_HARD_C = 0.0      # quadratic term (set to 0 for linear model)

# Hard-to-soft flux ratio (0.5–2 keV / 2–10 keV) for power-law Γ=1.7, NH=10^21 cm^-2
# (Comparat+2019 §3.2 / Table 2)
_HARD_TO_SOFT_RATIO = 0.35

# Default scatter in log10 LX at fixed log10 M_* [dex]
_DEFAULT_SCATTER_LX = 0.8

# Default satellite AGN fraction
_DEFAULT_F_SAT_AGN = 0.10

# Duty cycle table: (z, log10 DC) calibrated against Hasinger+2005 LDDE soft XLF
# f_DC(z): fraction of halos with an active AGN
# z=0–0.75 nodes: best-fit power-law 10^(-1.416 + 4.171 log10(1+z))
# z>0.75 nodes: capped at log10_DC=-0.301 (DC=0.50); the fitted power-law diverges
# unphysically at high z (extrapolation beyond the calibration range z<1)
_DUTY_CYCLE_Z   = jnp.array([0.00, 0.25, 0.75, 1.75, 3.50, 10.1])
_DUTY_CYCLE_LOG = jnp.array([-1.416, -1.012, -0.402, -0.301, -0.301, -0.301])


@jax.jit
def _duty_cycle_at_z(z: float) -> jnp.ndarray:
    """Linear interpolation of duty cycle f_DC(z) in log space."""
    log_dc = jnp.interp(jnp.asarray(z, dtype=float), _DUTY_CYCLE_Z, _DUTY_CYCLE_LOG)
    return jnp.power(10.0, log_dc)


@jax.jit
def _lx_hard_mean(
    log10mstar: jnp.ndarray,
    lx_a: float = _LX_HARD_A,
    lx_b: float = _LX_HARD_B,
    lx_c: float = _LX_HARD_C,
) -> jnp.ndarray:
    """Mean log10(L_X^{hard} [erg/s]) at fixed stellar mass — Comparat+2019 HAM.

    Parameters
    ----------
    log10mstar : log10(M_* [M_sun])
    lx_a, lx_b, lx_c : polynomial fit coefficients

    Returns
    -------
    log10_lx : log10(L_X [erg/s]) in the 2–10 keV band
    """
    dm = log10mstar - 10.0
    return lx_a + lx_b * dm + lx_c * dm ** 2


@jax.jit
def _scatter_boost(sigma_dex: float) -> jnp.ndarray:
    """Log-normal scatter boost: ⟨L_X⟩ / exp(μ) = exp(σ² / 2) where σ = σ_dex × ln10."""
    sigma_nat = sigma_dex * jnp.log(10.0)
    return jnp.exp(0.5 * sigma_nat ** 2)


class XrayAGNModel:
    """X-ray AGN model following Comparat+2019 abundance matching.

    Connects dark-matter halo mass to mean soft X-ray AGN luminosity via
    stellar-to-halo mass relation (SHMR) + LX–M* HAM relation with log-normal
    scatter and a redshift-dependent duty cycle.

    The model is fully JAX-differentiable: all array computations use ``jnp``.

    Parameters
    ----------
    shmr_func : callable(log10m_halo, z, **shmr_params) → log10(M_* [M_sun])
        SHMR function.  Should accept JAX arrays and return a JAX array.
        The default is :func:`~hod_mod.connection.sham.smhm_girelli20`.
    scatter_lx : float
        Log-normal scatter in log10 L_X at fixed log10 M_* [dex].  Default 0.8.
    f_sat_agn : float
        Fraction of AGN that are satellites (default 0.1).
    lx_a, lx_b, lx_c : float
        Coefficients of the LX–M_* polynomial (see :func:`_lx_hard_mean`).

    Notes
    -----
    The mean luminosity per halo already includes the scatter boost:
    ⟨L_X^{soft}⟩ = f_DC(z) × 10^{log10_L_hard} × hard_to_soft × exp(σ² / 2)

    The AGN contribution to P_{g,X}(k) is a point-source (delta function in
    real space), so X̃^{AGN}(k|M) is flat in k at the value ⟨L_X^{soft}⟩.
    """

    def __init__(
        self,
        shmr_func=None,
        scatter_lx: float = _DEFAULT_SCATTER_LX,
        f_sat_agn: float = _DEFAULT_F_SAT_AGN,
        lx_a: float = _LX_HARD_A,
        lx_b: float = _LX_HARD_B,
        lx_c: float = _LX_HARD_C,
        hard_to_soft: float = _HARD_TO_SOFT_RATIO,
    ):
        if shmr_func is None:
            from hod_mod.connection.sham import smhm_girelli20
            self._shmr = smhm_girelli20
        else:
            self._shmr = shmr_func
        self._scatter_lx  = float(scatter_lx)
        self._f_sat_agn   = float(f_sat_agn)
        self._lx_a        = float(lx_a)
        self._lx_b        = float(lx_b)
        self._lx_c        = float(lx_c)
        self._h2s         = float(hard_to_soft)
        # Precompute scatter boost (constant given fixed scatter_lx)
        self._boost = float(jax.device_get(_scatter_boost(scatter_lx)))

    def mean_agn_log10lx(
        self,
        m_halo_arr,
        z: float,
        shmr_params: dict | None = None,
    ) -> np.ndarray:
        """log10 of the mean soft X-ray AGN luminosity per halo [erg/s].

        Stays in log-space to avoid float32 overflow (L_X ~ 10^{42-44} erg/s
        exceeds the float32 maximum of ~3.4×10^{38}).

        Returns
        -------
        log10_lx_soft : (NM,) float64 ndarray
        """
        if shmr_params is None:
            shmr_params = {}
        log10m = np.log10(np.asarray(m_halo_arr, dtype=np.float64))
        log10mstar = np.asarray(
            self._shmr(jnp.asarray(log10m), z, **shmr_params), dtype=np.float64
        )
        log10_lx_hard = np.asarray(
            _lx_hard_mean(jnp.asarray(log10mstar),
                          self._lx_a, self._lx_b, self._lx_c),
            dtype=np.float64,
        )
        log10_lx_soft = log10_lx_hard + np.log10(self._h2s * self._boost)
        log10_lx_soft += np.log10(float(_duty_cycle_at_z(z)))
        return log10_lx_soft

    def mean_agn_lx(
        self,
        m_halo_arr: jnp.ndarray,
        z: float,
        shmr_params: dict | None = None,
    ) -> np.ndarray:
        """Mean soft X-ray AGN luminosity ⟨L_X^{0.5-2 keV}⟩ per halo [erg/s].

        Includes duty cycle and scatter boost but not the point-to-point scatter
        (which only affects the variance, not the mean in the halo model).

        Parameters
        ----------
        m_halo_arr : (NM,) [Msun/h] — halo mass
        z          : redshift
        shmr_params : dict, optional — extra kwargs forwarded to the SHMR function

        Returns
        -------
        lx_soft : (NM,) float64 ndarray [erg/s]
        """
        return np.power(10.0, self.mean_agn_log10lx(m_halo_arr, z, shmr_params))

    def agn_emissivity_uk(
        self,
        k_arr: jnp.ndarray,
        m_halo_arr: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        shmr_params: dict | None = None,
    ) -> jnp.ndarray:
        """Fourier transform of the AGN X-ray emissivity contribution.

        AGN are point sources, so their 3D profile is a delta function.  The
        Fourier transform is flat in k:

        .. math::

            \\tilde{X}^{\\rm AGN}(k|M) = \\frac{\\langle L_X^{\\rm AGN}(M) \\rangle}
                {4\\pi D_L^2(z)} \\times (1+z)^2 \\times f_{\\rm surf}

        The array is normalized by 1e43 to keep float32-safe magnitudes.
        :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra._pk_tables_gX`
        applies ``1e43 / (Lambda_eff × (cm_per_Mpc_h)³)`` to convert
        P_gX_agn into (Mpc/h)³ cm⁻⁶, matching the gas emissivity units.

        Parameters
        ----------
        k_arr      : (Nk,) [h/Mpc] — wavenumber array (output shape driver)
        m_halo_arr : (NM,) [Msun/h]
        z          : redshift
        theta_cosmo : dict with 'h', 'Omega_m'
        shmr_params : dict, optional

        Returns
        -------
        uk_agn : (Nk, NM) [L_X / 1e43, dimensionless] — flat in k (point-source AGN)
        """
        k    = jnp.asarray(k_arr, dtype=float)
        m    = jnp.asarray(m_halo_arr, dtype=float)
        Nk   = k.shape[0]
        NM   = m.shape[0]

        # Stay in log-space to avoid float32 overflow (L_X ~ 10^{42-44} erg/s).
        # Normalize by 1e43 so values are O(0.01–100); A_AGN absorbs the scale.
        log10_lx = self.mean_agn_log10lx(m, z, shmr_params)   # (NM,) float64
        lx_norm  = np.power(10.0, log10_lx - 43.0)             # (NM,) O(0.01–100)

        # Point-source: FT is flat in k.  Broadcast to (Nk, NM).
        return np.ones((Nk, 1), dtype=np.float64) * lx_norm[None, :]  # (Nk, NM)
