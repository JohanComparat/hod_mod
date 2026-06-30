"""HAM-based X-ray AGN model following Comparat+2019.

Implements :class:`HamAGNModel` which abundance-matches galaxy stellar masses
(from the Zu & Mandelbaum 2015 SHMR + halo mass function) to hard X-ray
luminosities drawn from the Aird+2015 LADE or Ueda+2014 LDDE XLF.

Pipeline
--------
1. **SHMR** — Zu & Mandelbaum (2015) Eq. 19 maps M_halo → M_* for centrals.
2. **HAM** — Cumulative n_gal(>M_*) × f_DC = n_AGN(>L_X) is matched to the
   Aird+2015 or Ueda+2014 XLF to assign L_X^{hard} to each halo.
   Precomputed at init time over a 2D grid (z, log10 M_halo).
3. **Obscuration model** — Comparat+2019 eqs 4–11 assign type fractions
   (Compton-thick, type-2, type-1) as a function of L_X^{hard} and z.
4. **K-correction** — Tabulated fraction_observed(z, logNH) from
   ``v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt`` converts the
   hard-band luminosity to an effective soft-band (0.5–2 keV) signal.
   Falls back to simplified f_unobs × h2s when the table is unavailable.

The hard XLF is reproduced by construction; the soft XLF is predicted.

The class exposes the same interface as :class:`~hod_mod.agn.xray.XrayAGNModel`
so it plugs into :class:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra`
without modification.

References
----------
Comparat et al. 2019, A&A 622, A12 (arXiv:1901.10866)
Aird et al. 2015, ApJ 815, 66 (arXiv:1503.01120) — LADE hard XLF
Ueda et al. 2014, ApJ 786, 104 (arXiv:1402.7902) — LDDE total hard XLF
Zu & Mandelbaum 2015, MNRAS 454, 1161 (arXiv:1505.02781) — iHOD SHMR
"""

from __future__ import annotations

import logging
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interp1d

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants (kept consistent with agn.py)
# ---------------------------------------------------------------------------

_HARD_TO_SOFT_RATIO = 0.607  # h2s: 0.5-2 keV / 2-10 keV for unobscured AGN (XSPEC, z=0 logNH=20)
_DEFAULT_SCATTER_LX = 0.8    # dex scatter in log10(L_X) at fixed M_halo
_DEFAULT_F_SAT_AGN  = 0.10   # fraction of satellites hosting AGN

# Minimum hard-band luminosity treated as an "active AGN" for the abundance
# match (log10(L_X^hard) / erg/s). This is a PHYSICAL assumption, not a
# numerical-resolution choice: the Aird+2015 LADE faint-end slope (~L^-0.48)
# makes the cumulative XLF integral diverge as the floor -> 0, so for
# samples whose stellar-mass threshold pulls in very low-mass halos (e.g.
# Comparat+2025 BGS sample S1, log10m_star_thresh~9.6), the abundance match
# against the full halo population down to ~1e10 Msun/h has NO finite
# solution -- no grid, however wide, converges. 40.0 dex (1e40 erg/s) is
# chosen as a round value near the conventional low-luminosity-AGN/XRB
# boundary (below this, X-ray emission is not reliably distinguishable from
# host-galaxy X-ray binary populations). For low-threshold samples, a large
# fraction of the HOD-weighted population will be clamped at this floor by
# construction -- that is an intentional modeling choice, not a hidden
# artifact, and should be reported (see `floor_fraction` diagnostics in
# audit_agn_lx_comparat2025.py / calibrate_ham_agn_lx.py).
_LOG10_LX_MIN_PHYSICAL = 40.0

# Duty cycle table — fraction of halos with an active AGN, from Comparat+2019
_DUTY_CYCLE_Z   = np.array([0.00, 0.25, 0.75, 1.75, 3.50, 10.1])
_DUTY_CYCLE_LOG = np.array([-1.416, -1.012, -0.402, -0.301, -0.301, -0.301])

# Default ZuMandelbaum+2015 SHMR parameters (Table 2, SDSS volume-limited)
_ZU15_DEFAULT_SHMR = {
    "lg_m1h":    12.10,
    "lg_m0star": 10.31,
    "beta":       0.33,
    "delta":      0.42,
    "gamma":      1.21,
}

# h=0.70 used by both XLF references (Aird+2015 and Ueda+2014)
_H_XLF = 0.70


# ---------------------------------------------------------------------------
# XLF functions (numpy, used in precomputation only)
# ---------------------------------------------------------------------------

def _aird15_lade_np(log10_lx_arr: np.ndarray, z: float) -> np.ndarray:
    """Aird+2015 LADE hard (2–10 keV) XLF — Comparat+2019 eqs 2–3.

    Parameters from the parametric fit in arXiv:1901.10866 (Comparat+2019)
    as implemented in ``AGN_setup.py`` of the st_mod pipeline.

    Returns Φ [Mpc^{-3} dex^{-1}] at h=0.70.
    """
    kz = 10.0 ** (-4.03 - 0.19 * (1.0 + z))
    Ls = 10.0 ** (
        44.84
        - np.log10(
            ((1.0 + 2.0) / (1.0 + z)) ** 3.87
            + ((1.0 + 2.0) / (1.0 + z)) ** (-2.12)
        )
    )
    L = 10.0 ** log10_lx_arr
    return kz / ((L / Ls) ** 0.48 + (L / Ls) ** 2.27)


def _ueda14_ldde_np(log10_lx_arr: np.ndarray, z: float) -> np.ndarray:
    """Ueda+2014 LDDE total hard (2–10 keV) XLF — Table 3 of arXiv:1402.7902.

    Counts all AGN (type-1 + type-2 + Compton-thick).
    Returns Φ [Mpc^{-3} dex^{-1}] at h=0.70.
    """
    phi0, Ls, g1, g2 = 3.31e-6, 10.0 ** 43.97, 0.96, 2.71
    e1, e2, zc0, alpha, La = 5.54, -0.36, 1.84, 0.335, 10.0 ** 44.61
    L = 10.0 ** log10_lx_arr
    phi = phi0 / ((L / Ls) ** g1 + (L / Ls) ** g2)
    zc = np.where(L < La, zc0 * (L / La) ** alpha, zc0)
    ed = np.where(
        z <= zc,
        (1.0 + z) ** e1,
        (1.0 + zc) ** e1 * ((1.0 + z) / (1.0 + zc)) ** e2,
    )
    return phi * ed


_XLF_FUNCS = {"aird15": _aird15_lade_np, "ueda14": _ueda14_ldde_np}


# ---------------------------------------------------------------------------
# Obscuration model (JAX-native, Comparat+2019 eqs 4–11)
# All functions take log10(L_X / erg/s) and redshift z as arguments.
# ---------------------------------------------------------------------------

def _thick_ll(z):
    """Compton-thick luminosity threshold as function of z."""
    return 41.5 + jnp.arctan(jnp.asarray(z, dtype=float) * 5.0) * 1.5


def _f_compton_thick(log10_lx: jnp.ndarray, z: float) -> jnp.ndarray:
    """Compton-thick fraction (logNH >= 24) — Comparat+2019 eq. 4."""
    return 0.30 * (0.5 + 0.5 * erf((_thick_ll(z) - log10_lx) / 0.25))


def _f_obsc_faint(log10_lx: jnp.ndarray, z: float) -> jnp.ndarray:
    """f_2: high obscured fraction at faint luminosities (Comparat+2019 eq. 6)."""
    return 0.9 * jnp.sqrt(41.0 / jnp.maximum(log10_lx, 1.0))


def _f_obsc_bright(log10_lx: jnp.ndarray, z: float) -> jnp.ndarray:
    """f_1: obscured fraction at bright luminosities (Comparat+2019 eq. 5)."""
    return (
        _f_compton_thick(log10_lx, z)
        + 0.01
        + erf(jnp.asarray(z, dtype=float) / 4.0) * 0.3
    )


def _ll_transition(z: float) -> float:
    """Luminosity transition between obscuration regimes."""
    return 43.2 + float(erf(jnp.asarray(z, dtype=float))) * 1.2


def obscured_fraction(log10_lx: jnp.ndarray, z: float) -> jnp.ndarray:
    """Total obscured fraction logNH > 22 (type-2 + CT) — Comparat+2019 eq. 11.

    Returns values in [0, 1].  All operations use ``jnp`` (JIT-compatible).
    """
    f1 = _f_obsc_bright(log10_lx, z)
    f2 = _f_obsc_faint(log10_lx, z)
    ll = _ll_transition(z)
    blend = 0.5 + 0.5 * erf((ll - log10_lx) / 0.6)
    return jnp.clip(f1 + (f2 - f1) * blend, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Helper: duty-cycle interpolation
# ---------------------------------------------------------------------------

def _duty_cycle_interp(z: float) -> float:
    """f_DC(z) from the Comparat+2019 table (log-linear interpolation)."""
    log_dc = float(np.interp(z, _DUTY_CYCLE_Z, _DUTY_CYCLE_LOG))
    return 10.0 ** log_dc


# ---------------------------------------------------------------------------
# K-correction helpers (module-level, shared by HamAGNModel and HODAgnModel)
# ---------------------------------------------------------------------------

_KCORR_TABLE_FILENAME = "v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt"


def setup_kcorrection(kcorr_path: str | None = None):
    """Resolve the K-correction table and build the interpolator.

    Resolution order: *kcorr_path* argument → package ``data/agn/`` →
    ``$GIT_STMOD_DATA`` → simplified fallback.

    Returns
    -------
    (interp, mode) : (LinearNDInterpolator or None, {'table', 'simplified'})
        ``interp`` maps ``(z, logNH) → fraction_observed``; ``mode`` is
        ``'table'`` when the table loaded successfully, else ``'simplified'``.
    """
    _PKG_TABLE = os.path.join(
        os.path.dirname(__file__), "..", "data", "agn", _KCORR_TABLE_FILENAME
    )
    candidates = []
    if kcorr_path is not None:
        candidates.append(kcorr_path)
    candidates.append(os.path.normpath(_PKG_TABLE))
    env_base = os.environ.get("GIT_STMOD_DATA", "")
    if env_base:
        candidates.append(
            os.path.join(
                env_base,
                "data", "models", "model_AGN", "xray_k_correction",
                _KCORR_TABLE_FILENAME,
            )
        )

    resolved = None
    for p in candidates:
        if os.path.isfile(p):
            resolved = p
            break

    if resolved is None:
        _LOG.warning(
            "K-correction table not found (tried %s). "
            "Using simplified k_eff = f_unobs × h2s = %.2f.",
            candidates or ["<no path given>"],
            _HARD_TO_SOFT_RATIO,
        )
        return None, "simplified"

    try:
        z_kc, lognh_kc, frac_kc = np.loadtxt(resolved, unpack=True)
        interp = LinearNDInterpolator(
            np.column_stack([z_kc, lognh_kc]),
            frac_kc,
        )
        _LOG.info("K-correction table loaded from %s.", resolved)
        return interp, "table"
    except Exception as exc:
        _LOG.warning(
            "Failed to load K-correction table (%s): %s. "
            "Falling back to simplified K-correction.",
            resolved, exc,
        )
        return None, "simplified"


def kcorr_at(kcorr_interp, kcorr_mode: str, z: float, lognh: float) -> float:
    """K-correction fraction at (z, logNH).  Returns h2s if table unavailable."""
    if kcorr_mode == "table" and kcorr_interp is not None:
        val = float(kcorr_interp([[z, lognh]])[0])
        # LinearNDInterpolator returns nan outside convex hull — fall back to h2s
        if np.isnan(val):
            return _HARD_TO_SOFT_RATIO if lognh < 22 else 0.0
        return val
    return _HARD_TO_SOFT_RATIO if lognh < 22 else 0.0


def mean_k_eff(kcorr_interp, kcorr_mode: str, log10_lx_hard: np.ndarray, z: float) -> np.ndarray:
    """Effective soft K-correction averaged over the N_H type distribution."""
    lx_jnp = jnp.asarray(log10_lx_hard)
    f_obs = np.array(obscured_fraction(lx_jnp, z))         # type-2 + CT fraction
    f_ct  = np.array(_f_compton_thick(lx_jnp, z))          # CT fraction only

    if kcorr_mode == "table":
        # Evaluate at representative logNH for each type
        # Type-1 (logNH~20), Type-2 (logNH~23), CT (logNH~25)
        k1 = kcorr_at(kcorr_interp, kcorr_mode, z, 20.0)
        k2 = kcorr_at(kcorr_interp, kcorr_mode, z, 23.0)
        k3 = kcorr_at(kcorr_interp, kcorr_mode, z, 25.0)
        f_type1 = np.clip(1.0 - f_obs, 0.0, 1.0)
        f_type2 = np.clip(f_obs - f_ct, 0.0, 1.0)
        f_ct_c  = np.clip(f_ct, 0.0, 1.0)
        return k1 * f_type1 + k2 * f_type2 + k3 * f_ct_c
    else:
        return (1.0 - f_obs) * _HARD_TO_SOFT_RATIO


# ---------------------------------------------------------------------------
# HamAGNModel
# ---------------------------------------------------------------------------

class HamAGNModel:
    """Comparat+2019 HAM AGN model.

    Abundance-matches galaxy stellar masses (from the Zu & Mandelbaum 2015 SHMR)
    to hard X-ray luminosities from the Aird+2015 or Ueda+2014 XLF.  The hard
    XLF is reproduced by construction; the soft XLF follows from the obscuration
    model and K-corrections.

    Parameters
    ----------
    pk_lin : LinearPowerSpectrum, optional
        Linear power spectrum instance.  Used to build the HMF for the HAM
        precomputation.  If *None*, a default Planck-2018 instance is created.
    theta_cosmo : dict, optional
        Cosmology dictionary (same format as :meth:`LinearPowerSpectrum.default_cosmology`).
        Default: Planck 2018.
    zu15_shmr_params : dict, optional
        Kwargs for :func:`_mstar_from_mh_zu15`.  Default: Zu & Mandelbaum (2015)
        Table 2 best-fit values for the SDSS volume-limited sample.
    scatter_lx : float
        Log-normal scatter in log10(L_X) at fixed M_halo [dex].  Default 0.8.
    f_sat_agn : float
        Fraction of satellite galaxies hosting AGN.  Default 0.10.
    duty_cycle : float or None
        Fixed duty cycle f_DC.  If *None* (default), f_DC(z) is interpolated
        from the Comparat+2019 table.
    xlf : {'aird15', 'ueda14'}
        Hard XLF reference.  ``'aird15'`` (default): Aird+2015 LADE
        (Comparat+2019 parametric form).  ``'ueda14'``: Ueda+2014 LDDE.
    kcorr_path : str or None
        Path to the K-correction table
        ``v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt``.
        Resolution order:

        1. *kcorr_path* argument
        2. ``$GIT_STMOD_DATA/data/models/model_AGN/xray_k_correction/v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt``
        3. Simplified fallback: ``k_eff = f_unobs × h2s`` (with a warning).
    """

    def __init__(
        self,
        pk_lin=None,
        theta_cosmo: dict | None = None,
        zu15_shmr_params: dict | None = None,
        scatter_lx: float = _DEFAULT_SCATTER_LX,
        f_sat_agn: float = _DEFAULT_F_SAT_AGN,
        duty_cycle: float | None = None,
        xlf: str = "aird15",
        kcorr_path: str | None = None,
        hmf=None,
    ):
        # -- Validate XLF choice
        if xlf not in _XLF_FUNCS:
            raise ValueError(f"xlf must be 'aird15' or 'ueda14', got '{xlf}'")
        self._xlf_name = xlf
        self._xlf_func = _XLF_FUNCS[xlf]

        # -- Cosmology
        if theta_cosmo is None:
            from hod_mod.core.power_spectrum import LinearPowerSpectrum
            theta_cosmo = LinearPowerSpectrum.default_cosmology()
        self._theta_cosmo = theta_cosmo

        # -- Power spectrum / HMF
        if pk_lin is None:
            from hod_mod.core.power_spectrum import LinearPowerSpectrum
            pk_lin = LinearPowerSpectrum()
        if hmf is not None:
            self._hmf = hmf
        else:
            from hod_mod.core.halo_mass_function import make_hmf
            self._hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)

        # -- SHMR parameters
        self._zu15_params = dict(_ZU15_DEFAULT_SHMR)
        if zu15_shmr_params is not None:
            self._zu15_params.update(zu15_shmr_params)

        # -- Scalar model parameters
        self._scatter_lx = float(scatter_lx)
        self._f_sat_agn  = float(f_sat_agn)
        self._duty_cycle = float(duty_cycle) if duty_cycle is not None else None
        # Scatter boost: ⟨L_X⟩ / median(L_X) = exp(σ²_nat / 2)
        self._boost      = float(np.exp(0.5 * (scatter_lx * np.log(10.0)) ** 2))

        # -- K-correction table
        self._kcorr_interp = None
        self._kcorr_mode   = "simplified"
        self._setup_kcorrection(kcorr_path)

        # -- Precompute HAM table
        _LOG.info("HamAGNModel: precomputing HAM table (xlf=%s) …", xlf)
        self._z_grid, self._log10mh_grid, self._ham_table = self._precompute_ham()
        self._ham_interp = RegularGridInterpolator(
            (self._z_grid, self._log10mh_grid),
            self._ham_table,
            method="linear",
            bounds_error=False,
            fill_value=None,   # extrapolate at boundaries
        )
        _LOG.info("HamAGNModel: precomputation done (table shape %s).", self._ham_table.shape)

    # ------------------------------------------------------------------
    # K-correction setup
    # ------------------------------------------------------------------

    def _setup_kcorrection(self, kcorr_path: str | None) -> None:
        """Resolve the K-correction table path and build the interpolator."""
        self._kcorr_interp, self._kcorr_mode = setup_kcorrection(kcorr_path)

    def _kcorr_at(self, z: float, lognh: float) -> float:
        """K-correction fraction at (z, logNH).  Returns h2s if table unavailable."""
        return kcorr_at(self._kcorr_interp, self._kcorr_mode, z, lognh)

    # ------------------------------------------------------------------
    # HAM precomputation
    # ------------------------------------------------------------------

    def _precompute_ham(
        self,
        z_grid: np.ndarray | None = None,
        log10mh_grid: np.ndarray | None = None,
        log10lx_grid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the 2D HAM mapping (z, log10 M_h) → log10(L_X^{hard}).

        ``log10lx_grid``'s lower bound defaults to ``_LOG10_LX_MIN_PHYSICAL``
        (40.0 dex) — a deliberate physical floor on the abundance match, not
        a numerical-resolution choice. The Aird+2015 faint-end slope makes
        the cumulative XLF integral diverge as the luminosity floor -> 0, so
        for low stellar-mass-threshold samples (pulling in halos down to
        ~1e10 Msun/h) there is no finite abundance-matched luminosity to
        find; widening the grid further does not converge, it just slides
        the clamp fainter (verified: cumulative N(>L) grows roughly as
        L^-0.48 with no sign of leveling off down to 26 dex). Halos whose
        abundance-matched luminosity would fall below the floor are clamped
        there instead.

        Returns
        -------
        z_grid : (n_z,)
        log10mh_grid : (n_mh,)
        ham_table : (n_z, n_mh) — log10(L_X^hard / erg/s)
        """
        from hod_mod.core.halo_mass_function import (
            _growth_factor_flat, _RHO_CRIT0,
        )

        if z_grid is None:
            z_grid = np.linspace(0.05, 2.0, 20)
        if log10mh_grid is None:
            log10mh_grid = np.linspace(10.0, 15.5, 400)
        if log10lx_grid is None:
            log10lx_grid = np.linspace(_LOG10_LX_MIN_PHYSICAL, 46.5, 500)

        # Unit conversion: XLF is at h=0.70, HMF at h=theta_cosmo['h']
        h_planck = float(self._theta_cosmo.get("h", 0.6736))
        h_factor = (_H_XLF / h_planck) ** 3   # Mpc^-3 → (Mpc/h)^-3

        theta  = self._theta_cosmo
        dlx    = np.gradient(log10lx_grid)    # dex step on LX grid
        dlogmh = np.gradient(log10mh_grid)    # dex step on Mh grid (uniform)
        mh_arr = 10.0 ** log10mh_grid
        dlnm   = 0.01

        # ------------------------------------------------------------------
        # Precompute sigma(M, z=0) ONCE — avoids calling the power spectrum
        # 60× (20 z-values × 3 finite-difference sigma calls).
        # sigma(M, z) = sigma(M, 0) × D(z)/D(0) for the same cosmology.
        # ------------------------------------------------------------------
        omega_m  = float(theta["Omega_m"])
        rho_mean = omega_m * float(_RHO_CRIT0)

        pk_z0 = self._hmf._pk(self._hmf._k_int, 0.0, theta)
        s2_arr = np.array(self._hmf._sigma2_z0(
            jnp.array(mh_arr), pk_z0, jnp.array(rho_mean)
        ))
        s2_hi = np.array(self._hmf._sigma2_z0(
            jnp.array(mh_arr * np.exp(dlnm)), pk_z0, jnp.array(rho_mean)
        ))
        s2_lo = np.array(self._hmf._sigma2_z0(
            jnp.array(mh_arr * np.exp(-dlnm)), pk_z0, jnp.array(rho_mean)
        ))
        # sigma8 rescaling (only if theta has sigma8 key)
        if "sigma8" in theta:
            R8   = 8.0
            M8   = (4.0 / 3.0) * np.pi * R8**3 * rho_mean
            s2_8 = float(self._hmf._sigma2_z0(
                jnp.array([M8]), pk_z0, jnp.array(rho_mean)
            )[0])
            rescale2 = theta["sigma8"] ** 2 / s2_8
            s2_arr *= rescale2
            s2_hi  *= rescale2
            s2_lo  *= rescale2

        sig0_arr = np.sqrt(s2_arr)
        sig0_hi  = np.sqrt(s2_hi)
        sig0_lo  = np.sqrt(s2_lo)
        # Pre-compute dlns/dlnm at z=0 (growth factor cancels in ratio)
        dlns_dlnm = (np.log(sig0_hi) - np.log(sig0_lo)) / (2.0 * dlnm)

        ham_table = np.zeros((len(z_grid), len(log10mh_grid)), dtype=np.float64)

        for i_z, z in enumerate(z_grid):
            # --- Cumulative HMF: n(> M_h) in (Mpc/h)^{-3}
            # Scale sigma by growth factor; dlns/dlnm is growth-factor-independent
            growth   = _growth_factor_flat(z, omega_m)
            sig      = sig0_arr * growth
            fsig     = np.array(self._hmf._fsigma_fn(jnp.array(sig), z))
            dndm_arr = fsig * (rho_mean / mh_arr ** 2) * np.abs(dlns_dlnm)
            dndlogmh = dndm_arr * mh_arr * np.log(10.0)    # (Mpc/h)^{-3} dex^{-1}
            n_halo_cumul = np.cumsum(dndlogmh[::-1] * dlogmh[::-1])[::-1]

            # --- Cumulative XLF: n_AGN(> L_X) in (Mpc/h)^{-3}
            xlf_val = self._xlf_func(log10lx_grid, z)       # Mpc^{-3} dex^{-1} at h=0.70
            xlf_h3  = xlf_val * h_factor                    # → (Mpc/h)^{-3} dex^{-1}
            n_agn_cumul = np.cumsum(xlf_h3[::-1] * dlx[::-1])[::-1]

            # --- Duty cycle at this redshift
            if self._duty_cycle is not None:
                f_dc = self._duty_cycle
            else:
                f_dc = _duty_cycle_interp(z)

            # --- HAM inversion: for each n_halo, find L_X s.t. f_DC × n_halo = n_AGN
            # n_agn_cumul is monotonically decreasing → reverse for interp1d
            # interp1d(x_increasing, y_decreasing_in_lx) → L_X from target density
            n_agn_rev = n_agn_cumul[::-1]    # now increasing (from ~0 to max)
            lx_rev    = log10lx_grid[::-1]   # now decreasing (46.5 → 41.0)

            lx_from_nagn = interp1d(
                n_agn_rev,
                lx_rev,
                kind="linear",
                bounds_error=False,
                fill_value=(lx_rev[0], lx_rev[-1]),  # (bright end, faint end)
            )

            target = n_halo_cumul * f_dc
            ham_table[i_z, :] = lx_from_nagn(target)

        return z_grid, log10mh_grid, ham_table

    # ------------------------------------------------------------------
    # Core public methods (same interface as XrayAGNModel)
    # ------------------------------------------------------------------

    def ham_log10lx_hard(
        self,
        log10_mh: np.ndarray,
        z: float,
    ) -> np.ndarray:
        """Raw HAM hard-band luminosity at given halo masses.

        No scatter, no obscuration, no duty-cycle correction.

        Parameters
        ----------
        log10_mh : log10(M_h / [M_sun/h]) array
        z : redshift

        Returns
        -------
        log10_lx_hard : log10(L_X^{2-10 keV} / erg/s)
        """
        log10_mh = np.asarray(log10_mh, dtype=np.float64)
        pts = np.column_stack([
            np.full(log10_mh.shape, float(z)),
            log10_mh,
        ])
        return self._ham_interp(pts)

    def _mean_k_eff(self, log10_lx_hard: np.ndarray, z: float) -> np.ndarray:
        """Effective soft K-correction averaged over the N_H type distribution."""
        return mean_k_eff(self._kcorr_interp, self._kcorr_mode, log10_lx_hard, z)

    def mean_agn_log10lx(
        self,
        m_halo_arr,
        z: float,
        shmr_params: dict | None = None,
        *,
        scatter_lx: float | None = None,
        log10_A_kcorr: float = 0.0,
        log10_A_dc: float = 0.0,
    ) -> np.ndarray:
        """log10 of the mean soft X-ray AGN luminosity per halo [erg/s].

        Includes: HAM hard luminosity + scatter boost + K-correction
        (obscuration-weighted) + duty cycle.

        Parameters
        ----------
        m_halo_arr : (NM,) array [M_sun/h]
        z : redshift
        shmr_params : ignored (retained for interface compatibility)
        scatter_lx : optional override of the constructor's ``scatter_lx``
            (dex), used to recompute the scatter boost on the fly. Cheap:
            unlike ``duty_cycle``, scatter never feeds into the precomputed
            HAM table, so this does not retrigger the ~12s precompute.
        log10_A_kcorr : log10 multiplicative rescaling of the effective
            K-correction ``k_eff``, clamped so the result never exceeds 1
            (``k_eff`` is a flux fraction). Default 0.0 = no change.
        log10_A_dc : log10 multiplicative rescaling of the duty cycle used
            in this population-averaging step ONLY, clamped at 1. This
            deliberately does *not* alter the duty cycle baked into the
            abundance-matching table (``ham_log10lx_hard`` / HAM precompute)
            — re-deriving that self-consistently would require rebuilding
            the table per trial value. Default 0.0 = no change. See
            :meth:`_precompute_ham` for the two distinct roles duty cycle
            plays in this model.

        Returns
        -------
        log10_lx_soft : (NM,) float64 ndarray [erg/s]
        """
        m_halo_arr = np.asarray(m_halo_arr, dtype=np.float64)
        log10mh = np.log10(m_halo_arr)

        # 1. HAM hard luminosity (from precomputed table)
        log10_lx_hard = self.ham_log10lx_hard(log10mh, z)

        # 2. Effective soft K-correction (obscuration-weighted)
        k_eff = self._mean_k_eff(log10_lx_hard, z)
        if log10_A_kcorr != 0.0:
            k_eff = np.minimum(k_eff * 10.0 ** log10_A_kcorr, 1.0)

        # 3. Duty cycle
        if self._duty_cycle is not None:
            f_dc = self._duty_cycle
        else:
            f_dc = _duty_cycle_interp(z)
        if log10_A_dc != 0.0:
            f_dc = min(f_dc * 10.0 ** log10_A_dc, 1.0)

        # 4. Scatter boost (optionally overridden)
        if scatter_lx is not None:
            boost = float(np.exp(0.5 * (scatter_lx * np.log(10.0)) ** 2))
        else:
            boost = self._boost

        # 5. Combine in log space (scatter boost × K-correction × duty cycle)
        # boost = exp(σ²_nat / 2) accounts for log-normal ⟨L⟩ > median(L)
        log10_k_eff_fdc = np.log10(np.maximum(k_eff * f_dc * boost, 1e-300))
        return log10_lx_hard + log10_k_eff_fdc

    def mean_agn_lx(
        self,
        m_halo_arr,
        z: float,
        shmr_params: dict | None = None,
        *,
        scatter_lx: float | None = None,
        log10_A_kcorr: float = 0.0,
        log10_A_dc: float = 0.0,
    ) -> np.ndarray:
        """Mean soft X-ray AGN luminosity per halo [erg/s]."""
        return np.power(10.0, self.mean_agn_log10lx(
            m_halo_arr, z, shmr_params,
            scatter_lx=scatter_lx, log10_A_kcorr=log10_A_kcorr, log10_A_dc=log10_A_dc,
        ))

    def agn_emissivity_uk(
        self,
        k_arr,
        m_halo_arr,
        z: float,
        theta_cosmo: dict,
        shmr_params: dict | None = None,
        *,
        scatter_lx: float | None = None,
        log10_A_kcorr: float = 0.0,
        log10_A_dc: float = 0.0,
    ) -> np.ndarray:
        """Fourier transform of the AGN X-ray emissivity (point-source, flat in k).

        Same interface and units as
        :meth:`~hod_mod.agn.xray.XrayAGNModel.agn_emissivity_uk`.

        Returns
        -------
        uk_agn : (Nk, NM) float64 ndarray [L_X / 1e43, dimensionless]
        """
        k   = np.asarray(k_arr, dtype=np.float64)
        m   = np.asarray(m_halo_arr, dtype=np.float64)
        Nk  = k.shape[0]

        log10_lx = self.mean_agn_log10lx(
            m, z, shmr_params,
            scatter_lx=scatter_lx, log10_A_kcorr=log10_A_kcorr, log10_A_dc=log10_A_dc,
        )    # (NM,)
        lx_norm  = np.power(10.0, log10_lx - 43.0)             # O(0.01–100)

        # Point-source: flat in k → broadcast to (Nk, NM)
        return np.ones((Nk, 1), dtype=np.float64) * lx_norm[None, :]
