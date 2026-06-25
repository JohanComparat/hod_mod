"""Halo model cross-power spectra between galaxies and gas fields.

Provides :class:`HaloModelCrossSpectra` which wraps a
:class:`~hod_mod.galaxies.clustering.FullHaloModelPrediction` instance and reuses
its static cache (HMF, bias, DM profile FT, linear power spectrum) to compute:

* **P_{g,y}(k)** — galaxy × Compton-y (tSZ) cross-power, in (Mpc/h)².
* **P_{m,y}(k)** — matter × Compton-y cross-power (for lensing × tSZ), in (Mpc/h)².
* **P_{g,X}(k)** — galaxy × X-ray emissivity cross-power, in (Mpc/h)³ cm⁻⁶.

Projected observables:

* :meth:`projected_gy`  — Σ_y(r_p) [dimensionless Compton-y] via Abel projection.
* :meth:`projected_gX`  — w_{g,X}(r_p) via Abel projection.
* :meth:`angular_cl_gy` — C_ℓ^{g,y} via Limber approximation.
* :meth:`angular_cl_gX` — C_ℓ^{g,X} via Limber approximation.

References
----------
Galaxy × tSZ formalism:
  Pandey+2025, arXiv:2506.07432 — DES Year 3 shear × ACT DR6 tSZ
  Amodeo+2021, arXiv:2009.05557 — ACT × BOSS CMASS/LOWZ stacked tSZ

Galaxy × soft X-ray:
  Comparat+2025, arXiv:2503.19796, A&A 697 A173

Pressure profile:
  Arnaud+2010, arXiv:0910.1234 — A10 generalized NFW

Density profile:
  Oppenheimer+2025, arXiv:2505.14782 — DPM
"""

from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.cosmology.gas_profiles import (
    PressureProfileA10,
    PressureProfileDPM,
    GasDensityDPM,
    _RHO_CRIT0,
)

# Mpc → cm conversion (for unit conversion in projected observables if needed)
_MPC_CM = 3.0857e24
_ARCSEC_TO_RAD = float(jnp.pi) / (180.0 * 3600.0)


def _safe_log(P, floor: float = 1e-60):
    """``log(max(P, floor))`` with non-finite ``P`` floored instead of propagated.

    The full-APEC gas emissivity has a tiny high-k tail (k ≳ 150 h/Mpc) that can
    underflow float32 to NaN/inf inside the halo-model integrals.  Those k are far
    beyond any fitted angular scale, so we floor them to ``floor`` (≈ zero power)
    rather than let a single NaN poison the whole Limber/Hankel transform.
    """
    P = jnp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.log(jnp.maximum(P, floor))


def psf_window_ell(ell_arr: np.ndarray, fwhm_arcsec: float = 30.0) -> np.ndarray:
    """Gaussian PSF window function B_ℓ = exp(−ℓ² σ² / 2).

    For a Gaussian PSF with FWHM = ``fwhm_arcsec`` arcseconds, the angular
    power spectrum of a PSF-convolved map is

    .. math::

        C_\\ell^{\\rm obs} = C_\\ell^{\\rm true} \\times B_\\ell

    where the galaxy field is not convolved (only the X-ray / y field).
    For the X-ray *auto*-power (if needed), multiply by B_ℓ².

    Parameters
    ----------
    ell_arr     : (Nell,) angular multipoles
    fwhm_arcsec : PSF FWHM [arcsec], default 30 (eROSITA soft X-ray)

    Returns
    -------
    B_ell : (Nell,) dimensionless window function in [0, 1]
    """
    sigma_rad = fwhm_arcsec * _ARCSEC_TO_RAD / 2.355   # Gaussian σ [rad]
    ell = jnp.asarray(ell_arr)
    return jnp.exp(-0.5 * ell ** 2 * sigma_rad ** 2)


def psf_king_profile(
    theta_arcsec: np.ndarray,
    theta_c_arcsec: float = 8.64,
    alpha: float = 1.5,
) -> np.ndarray:
    """King-profile PSF in real (angular) space, normalized to 1 at θ=0.

    PSF(θ) = (1 + (θ / θ_c)²)^{−α}

    Parameters
    ----------
    theta_arcsec   : angular separations [arcsec]
    theta_c_arcsec : King core radius [arcsec].
        Fitted to eROSITA TM CalDB (0.5–2 keV, on-axis): 8.64".
    alpha          : King slope (α > 1). Default 1.5 matches eROSITA TM fit.

    Returns
    -------
    PSF : (N,) array, in [0, 1]
    """
    x = np.asarray(theta_arcsec, dtype=float) / theta_c_arcsec
    return (1.0 + x ** 2) ** (-float(alpha))


def psf_king_window_ell(
    ell_arr: np.ndarray,
    theta_c_arcsec: float = 8.64,
    alpha: float = 1.5,
) -> np.ndarray:
    """King-profile PSF window B_ℓ — analytic Hankel (Fourier-Bessel) transform.

    Real-space PSF: PSF(θ) ∝ (1 + (θ/θ_c)²)^{−α}

    Analytic Hankel transform (normalized to B_0 = 1):

    .. math::

        B_\\ell = \\frac{2^{2-\\alpha}}{\\Gamma(\\alpha-1)}
                  (\\ell\\,\\theta_c)^{\\alpha-1}\\,K_{\\alpha-1}(\\ell\\,\\theta_c)

    where :math:`K_\\nu` is the modified Bessel function of the second kind.

    Special case α = 3/2 → :math:`B_\\ell = \\exp(-\\ell\\,\\theta_c)`,
    a pure exponential that is fully JAX-native.

    Parameters
    ----------
    ell_arr        : (Nell,) angular multipoles
    theta_c_arcsec : King core radius [arcsec].
        Fitted to eROSITA TM CalDB (0.5–2 keV, on-axis): 8.64".
        For a survey-averaged 30"-FWHM effective PSF use ~19.6".
    alpha          : King slope (α > 1). Default 1.5 matches eROSITA TM fit.

    Returns
    -------
    B_ell : (Nell,) dimensionless window function in [0, 1]
    """
    theta_c_rad = float(theta_c_arcsec) * _ARCSEC_TO_RAD
    ell = np.asarray(ell_arr, dtype=float)
    x = ell * theta_c_rad

    if abs(alpha - 1.5) < 1e-6:
        # K_{1/2}(x) = sqrt(π/2x) exp(-x)  →  x^{1/2} K_{1/2}(x) = sqrt(π/2) exp(-x)
        # prefac = 2^{1/2} / Γ(1/2) = sqrt(2)/sqrt(π)  →  B_ℓ = exp(-x)
        return jnp.asarray(np.exp(-x))

    import math
    from scipy.special import kv as _kv

    nu = alpha - 1.0
    norm0 = 2.0 ** (nu - 1.0) * math.gamma(nu)   # limit of x^nu K_nu(x) as x→0
    prefac = 2.0 ** (2.0 - alpha) / math.gamma(nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bx = x ** nu * _kv(nu, x)
    Bx = np.where(x == 0.0, norm0, Bx)
    return jnp.asarray(prefac * Bx / norm0)


# ---------------------------------------------------------------------------
# Internal helper: Ogata-style Hankel/Abel projection (reuse P_{g,y} → Σ_y)
# ---------------------------------------------------------------------------

def _pk_to_wp(rp_arr: np.ndarray, log_k: np.ndarray, log_pk: np.ndarray) -> np.ndarray:
    """Project P_{cross}(k) to w_{cross}(r_p) via two-step Abel projection.

    Uses the same two-step approach as
    :func:`hod_mod.galaxies.clustering._delta_sigma_from_pgm`:

    1. Compute the 3D correlation via Ogata j₀ Hankel transform:

    .. math::

        \\xi(r) = \\frac{1}{2\\pi^2}
        \\int_0^\\infty k^2\\,P(k)\\,\\frac{\\sin(kr)}{kr}\\,\\mathrm{d}k

    2. Abel-project along the line of sight:

    .. math::

        w(r_p) = 2 \\int_0^{\\pi_{\\max}}
        \\xi\\!\\left(\\sqrt{r_p^2 + \\pi^2}\\right)\\,\\mathrm{d}\\pi

    This avoids the rapidly oscillating J₀(k rp) at large rp that causes
    ringing with direct trapezoidal Hankel integration.

    Parameters
    ----------
    rp_arr : (NR,) [Mpc/h]
    log_k  : (Nk,) log(k [h/Mpc])
    log_pk : (Nk,) log(P(k)) — any units consistent across P and output

    Returns
    -------
    wp : (NR,) in units [P(k) units × (h/Mpc)²  ÷  (Mpc/h)  = P(k)/volume × length]
         For P in (Mpc/h)², wp is dimensionless (Compton-y).
         For P in (Mpc/h)³ cm⁻⁶, wp is in (Mpc/h) cm⁻⁶.
    """
    from hod_mod.galaxies.clustering import _pk_to_xi

    rp   = np.asarray(rp_arr, dtype=float)
    lk   = jnp.asarray(log_k)
    lpk  = jnp.asarray(log_pk)

    # Step 1: 3D correlation function on a dense r grid
    pi_max = 300.0
    n_pi   = 512
    # Build log-linear hybrid chi grid (same pattern as _delta_sigma_from_pgm)
    pi_log  = np.logspace(-2, np.log10(pi_max), n_pi // 2)
    pi_lin  = np.linspace(1.0, pi_max, n_pi // 2)
    pi_grid = np.sort(np.unique(np.concatenate([pi_log, pi_lin])))

    # Evaluate ξ on all needed 3D separations at once
    r_2d   = np.outer(rp, np.ones(len(pi_grid)))    # (NR, Npi)
    pi_2d  = np.outer(np.ones(len(rp)), pi_grid)    # (NR, Npi)
    r_3d   = np.sqrt(r_2d**2 + pi_2d**2)             # (NR, Npi)

    r_flat = jnp.asarray(r_3d.ravel())
    xi_flat = _pk_to_xi(r_flat, lk, lpk)
    xi_2d   = xi_flat.reshape(len(rp), len(pi_grid))   # (NR, Npi)

    # Step 2: LOS integration for each rp
    wp = 2.0 * jnp.trapezoid(xi_2d, jnp.asarray(pi_grid), axis=1)   # (NR,)
    return np.asarray(wp)


# ---------------------------------------------------------------------------
# HaloModelCrossSpectra
# ---------------------------------------------------------------------------

class HaloModelCrossSpectra:
    """Halo model galaxy × gas cross-power spectra and projected observables.

    Wraps a :class:`~hod_mod.galaxies.clustering.FullHaloModelPrediction`
    and reuses its static cache (HMF, bias, DM profile FT, linear P(k)),
    adding gas-profile Fourier transforms for the y-field (tSZ) and
    X-ray emissivity field.

    Parameters
    ----------
    fhmp : FullHaloModelPrediction
        Already-instantiated prediction object whose static cache is reused.
    pressure_profile : PressureProfileA10, optional
        Electron pressure profile for tSZ. If ``None``, tSZ methods raise.
    density_profile : GasDensityDPM, optional
        Electron density profile for X-ray. If ``None``, X-ray methods raise.
    """

    def __init__(
        self,
        fhmp,
        pressure_profile: PressureProfileA10 | PressureProfileDPM | None = None,
        density_profile: GasDensityDPM | None = None,
        metallicity_profile=None,
        cooling_function=None,
        agn_model=None,
    ):
        """
        Parameters
        ----------
        fhmp : FullHaloModelPrediction
        pressure_profile : PressureProfileA10 | PressureProfileDPM | None
            Electron pressure for tSZ.  Both A10 and DPM profiles are supported
            via the shared ``pressure_uk(k, m200, r200, c200, z, theta_cosmo)``
            interface.
        density_profile : GasDensityDPM | None
            Electron density for X-ray.
        metallicity_profile : MetallicityProfileDPM | None
            Gas metallicity for cooling-function-weighted emissivity.  Required
            for the full APEC emissivity path (``_pk_tables_gX`` always uses
            ``emissivity_full_uk`` when all three profiles are provided).
        cooling_function : ApecCoolingTable | None
            Precomputed APEC band-integrated cooling function Λ(T, Z).
            When provided together with ``pressure_profile`` (DPM) and
            ``metallicity_profile``, the full per-quadrature-point emissivity
            ε(r) = n_e²(r) Λ(T(r), Z(r)) is evaluated.  Instantiate once
            and reuse across fits.
        agn_model : HamAGNModel | XrayAGNModel | None
            Optional AGN contribution to the X-ray cross-power.  When provided,
            an AGN point-source term is added in ``_pk_tables_gX``.
            ``HamAGNModel`` (Aird+2015 LADE) is preferred over ``XrayAGNModel``.
        """
        self._fhmp       = fhmp
        self._pp         = pressure_profile
        self._dp         = density_profile
        self._mp         = metallicity_profile
        self._cooling_fn = cooling_function
        self._agn        = agn_model
        # An AGN model that exposes its own occupation (HODAgnModel) drives a
        # dedicated, occupation-weighted X-ray branch in _pk_tables_gX /
        # _pk_tables_XX_hod.  Parametric / HAM AGN models (no nc_ns_agn) use the
        # legacy galaxy-HOD-weighted point-source path.
        self._agn_has_hod = agn_model is not None and hasattr(agn_model, "nc_ns_agn")
        # Separate cache for gas profile FTs (keyed by (model_id, z, cosmo_key))
        self._gas_cache: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosmo_key(self, theta_cosmo: dict) -> tuple:
        """Stable hashable key for a cosmology dict."""
        return tuple(sorted((k, float(v)) for k, v in theta_cosmo.items()))

    def _get_static_cache(self, z: float, theta_cosmo: dict, hod_params: dict) -> dict:
        """Trigger FullHaloModelPrediction static cache population and return it."""
        from hod_mod.galaxies.clustering import FullHaloModelPrediction
        _ = self._fhmp._pk_tables_full(z, theta_cosmo, hod_params)
        cosmo_key = FullHaloModelPrediction._cosmo_cache_key(z, theta_cosmo)
        return self._fhmp._static_cache[cosmo_key]

    def _get_hod_weights(
        self, z: float, theta_cosmo: dict, hod_params: dict, sc: dict
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return (nc_np, ns_np, n_gal, b_eff) using the static-cache mass grid."""
        import jax
        with jax.disable_jit():
            nc_arr, ns_arr = self._fhmp._hod.nc_ns(
                self._fhmp._hod._log10m_grid, hod_params
            )
        nc_np  = np.asarray(nc_arr, dtype=float)
        ns_np  = np.asarray(ns_arr, dtype=float)
        nt_np  = nc_np + ns_np
        dndm   = sc["dndm_np"]
        bias   = sc["bias_np"]
        m_np   = sc["m_np"]
        n_gal  = float(np.trapezoid(dndm * nt_np, m_np))
        b_eff  = float(np.trapezoid(dndm * nt_np * bias, m_np) / n_gal)
        return nc_np, ns_np, n_gal, b_eff

    def _pressure_uk_cached(
        self, z: float, theta_cosmo: dict, sc: dict
    ) -> np.ndarray:
        """ỹ(k|M,z) from PressureProfileA10, with caching. (Nk, NM) [(Mpc/h)²]."""
        if self._pp is None:
            raise RuntimeError("No pressure_profile provided to HaloModelCrossSpectra")
        gas_key = ("pressure", id(self._pp), z, self._cosmo_key(theta_cosmo))
        if gas_key not in self._gas_cache:
            self._gas_cache[gas_key] = self._pp.pressure_uk(
                k_arr     = sc["k_np"],
                m200_arr  = sc["m_np"],
                r200_arr  = sc["r_delta"],
                c200_arr  = sc["c_np"],
                z         = z,
                theta_cosmo = theta_cosmo,
            )
        return self._gas_cache[gas_key]

    def _density_uk_cached(
        self, z: float, theta_cosmo: dict, sc: dict, emissivity: bool = False
    ) -> np.ndarray:
        """ñ_e(k|M) or X̃(k|M) from GasDensityDPM, with caching. (Nk, NM)."""
        if self._dp is None:
            raise RuntimeError("No density_profile provided to HaloModelCrossSpectra")
        kind    = "emissivity" if emissivity else "density"
        gas_key = (kind, id(self._dp), z, self._cosmo_key(theta_cosmo))
        if gas_key not in self._gas_cache:
            if emissivity:
                result = self._dp.emissivity_uk(
                    k_arr     = sc["k_np"],
                    m200_arr  = sc["m_np"],
                    r200_arr  = sc["r_delta"],
                    z         = z,
                    theta_cosmo = theta_cosmo,
                )
            else:
                result = self._dp.density_uk(
                    k_arr     = sc["k_np"],
                    m200_arr  = sc["m_np"],
                    r200_arr  = sc["r_delta"],
                    z         = z,
                    theta_cosmo = theta_cosmo,
                )
            self._gas_cache[gas_key] = result
        return self._gas_cache[gas_key]

    # ------------------------------------------------------------------
    # Power spectrum tables
    # ------------------------------------------------------------------

    def _pk_tables_gy(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> dict:
        """Compute P_{g,y}(k) and P_{m,y}(k) via the halo model.

        1-halo galaxy × y term:

        .. math::

            P_{g,y}^{\\rm 1h}(k) = \\frac{1}{n_g}
            \\int \\frac{\\mathrm{d}n}{\\mathrm{d}M}
            \\left[N_c(M) + N_s(M)\\,\\tilde{u}_s(k,M)\\right]
            \\tilde{y}(k|M,z) \\, \\mathrm{d}M

        2-halo galaxy × y term:

        .. math::

            P_{g,y}^{\\rm 2h}(k) = b_{\\rm eff}\\,P_{\\rm lin}(k)
            \\int \\frac{\\mathrm{d}n}{\\mathrm{d}M}\\, b(M)\\,\\tilde{y}(k|M,z)
            \\, \\mathrm{d}M

        where ỹ(k|M,z) is from :class:`~hod_mod.cosmology.gas_profiles.PressureProfileA10`
        and has units (Mpc/h)².

        Parameters
        ----------
        z, theta_cosmo, hod_params : as for FullHaloModelPrediction

        Returns
        -------
        dict with keys:
          log_k, log_pgy, log_pgy_1h, log_pgy_2h, log_pmy, n_gal, b_eff
        """
        sc = self._get_static_cache(z, theta_cosmo, hod_params)
        nc_np, ns_np, n_gal, b_eff = self._get_hod_weights(
            z, theta_cosmo, hod_params, sc
        )
        m_np   = sc["m_np"]
        dndm   = sc["dndm_np"]
        bias   = sc["bias_np"]
        pk_lin = sc["pk_lin"]
        uk     = sc["uk"]      # DM profile FT, (Nk, NM), for satellite occupation
        rho_m  = sc["rho_m"]

        y_uk = self._pressure_uk_cached(z, theta_cosmo, sc)   # (Nk, NM)

        m_jnp    = jnp.asarray(m_np)
        dndm_j   = jnp.asarray(dndm)
        bias_j   = jnp.asarray(bias)
        pk_lin_j = jnp.asarray(pk_lin)
        uk_j     = jnp.asarray(uk)
        y_uk_j   = jnp.asarray(y_uk)
        nc_j     = jnp.asarray(nc_np)
        ns_j     = jnp.asarray(ns_np)

        # 1-halo g×y: integral over (N_c + N_s ũ_s) × ỹ
        gal_weights_1h = nc_j[None, :] + ns_j[None, :] * uk_j   # (Nk, NM)
        integrand_pgy_1h = dndm_j[None, :] * gal_weights_1h * y_uk_j
        P_gy_1h = jnp.trapezoid(integrand_pgy_1h, m_jnp, axis=1) / n_gal   # (Nk,)

        # 2-halo g×y
        I_y = jnp.trapezoid(dndm_j[None, :] * bias_j[None, :] * y_uk_j, m_jnp, axis=1)
        P_gy_2h = b_eff * pk_lin_j * I_y

        P_gy = P_gy_1h + P_gy_2h

        # Matter × y
        m_over_rho = m_jnp / rho_m
        integrand_pmy_1h = dndm_j[None, :] * m_over_rho[None, :] * uk_j * y_uk_j
        P_my_1h = jnp.trapezoid(integrand_pmy_1h, m_jnp, axis=1)

        I_m = jnp.trapezoid(dndm_j[None, :] * bias_j[None, :] * m_over_rho[None, :] * uk_j, m_jnp, axis=1)
        P_my_2h = pk_lin_j * I_m * I_y

        P_my = P_my_1h + P_my_2h

        log_k = jnp.log(jnp.asarray(sc["k_np"]))
        return {
            "log_k":       log_k,
            "log_pgy":     jnp.log(jnp.maximum(P_gy,    1e-30)),
            "log_pgy_1h":  jnp.log(jnp.maximum(P_gy_1h, 1e-30)),
            "log_pgy_2h":  jnp.log(jnp.maximum(P_gy_2h, 1e-30)),
            "log_pmy":     jnp.log(jnp.maximum(P_my,    1e-30)),
            "n_gal":       n_gal,
            "b_eff":       b_eff,
        }

    def _pk_tables_gX(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        beta_gas: float | None = None,
        beta_pressure: float | None = None,
        agn_kwargs: dict | None = None,
    ) -> dict:
        """Compute P_{g,X}(k) via the halo model.

        Same structure as :meth:`_pk_tables_gy` but with the X-ray emissivity
        profile FT X̃(k|M) from :class:`~hod_mod.cosmology.gas_profiles.GasDensityDPM`.

        When ``density_profile``, ``pressure_profile`` (DPM), ``metallicity_profile``,
        and ``cooling_function`` are all set on this instance, uses
        :meth:`GasDensityDPM.emissivity_full_uk` with the full per-quadrature-point
        APEC cooling function Λ(T(r), Z(r)).  Otherwise falls back to the
        deprecated plain ``emissivity_uk`` (no T/Z weighting).

        Returns
        -------
        dict with keys:
          log_k, log_pgX, log_pgX_1h, log_pgX_2h, log_pgX_gas, log_pgX_agn, n_gal, b_eff
        """
        sc = self._get_static_cache(z, theta_cosmo, hod_params)
        nc_np, ns_np, n_gal, b_eff = self._get_hod_weights(
            z, theta_cosmo, hod_params, sc
        )
        m_np   = sc["m_np"]
        dndm   = sc["dndm_np"]
        bias   = sc["bias_np"]
        pk_lin = sc["pk_lin"]
        uk     = sc["uk"]

        _has_full = (
            self._dp is not None
            and isinstance(self._pp, PressureProfileDPM)
            and self._mp is not None
            and self._cooling_fn is not None
        )
        # Λ_ref: cooling-function value at (1 keV, 0.3 Z⊙) when the APEC table is
        # set, else the legacy power-law reference.  Used to put the full-APEC gas
        # emissivity AND the AGN luminosity on the same n_e²-scale convention.
        if self._cooling_fn is not None:
            lambda_ref = float(self._cooling_fn(np.array([1.0]), np.array([0.3]))[0])
        else:
            lambda_ref = 3e-23   # legacy power-law default

        if _has_full:
            X_uk = self._dp.emissivity_full_uk(
                k_arr               = sc["k_np"],
                m200_arr            = sc["m_np"],
                r200_arr            = sc["r_delta"],
                z                   = z,
                theta_cosmo         = theta_cosmo,
                pressure_profile    = self._pp,
                metallicity_profile = self._mp,
                cooling_fn          = self._cooling_fn,
            )
            # emissivity_full_uk carries Λ(T,Z) (~1e-24), so it is ~Λ_ref smaller
            # than the n_e²-scale emissivity_uk and underflows float32 in the
            # integrals below.  Divide by Λ_ref to land in the same
            # (Mpc/h)³ cm⁻⁶ convention as emissivity_uk / the AGN term; the
            # Λ(T,Z)/Λ_ref ratio keeps the temperature/metallicity dependence and
            # the overall amplitude is absorbed by the free A_gas.
            X_uk = np.asarray(X_uk) / lambda_ref
        else:
            X_uk = self._density_uk_cached(z, theta_cosmo, sc, emissivity=True)   # (Nk, NM)

        m_jnp    = jnp.asarray(m_np)
        dndm_j   = jnp.asarray(dndm)
        bias_j   = jnp.asarray(bias)
        pk_lin_j = jnp.asarray(pk_lin)
        uk_j     = jnp.asarray(uk)
        X_uk_j   = jnp.asarray(X_uk)
        nc_j     = jnp.asarray(nc_np)
        ns_j     = jnp.asarray(ns_np)

        # Optional mass-slope tilt for the gas emissivity: n_e² ∝ M^(2β), so
        # shifting β by Δβ multiplies X̃(k|M) by (M/1e12)^(2Δβ).  Pure JAX op.
        M12_j = m_jnp / 1.0e12  # (NM,) — used by both tilts below
        if beta_gas is not None and self._dp is not None:
            delta_beta = float(beta_gas) - float(self._dp._beta)
            X_uk_j = X_uk_j * M12_j[None, :] ** (2.0 * delta_beta)  # (Nk, NM)

        # Pressure slope tilt: Λ(T) ∝ T^0.5 ∝ (P/n_e)^0.5 adds a P^0.5 factor.
        # Shifting β_P by Δβ_P multiplies X̃ by (M/1e12)^(0.5 Δβ_P).
        # Reference β_P is DPM model-2 construction value (0.85).
        if beta_pressure is not None:
            _BETA_P_REF = 0.85   # DPM model-2 PressureProfileDPM._PARAMS[2]["beta"]
            delta_beta_P = float(beta_pressure) - _BETA_P_REF
            X_uk_j = X_uk_j * M12_j[None, :] ** (0.5 * delta_beta_P)

        integrand_pgX_1h_cen = dndm_j[None, :] * nc_j[None, :] * X_uk_j
        P_gX_1h_cen = jnp.trapezoid(integrand_pgX_1h_cen, m_jnp, axis=1) / n_gal
        integrand_pgX_1h_sat = dndm_j[None, :] * ns_j[None, :] * uk_j * X_uk_j
        P_gX_1h_sat = jnp.trapezoid(integrand_pgX_1h_sat, m_jnp, axis=1) / n_gal
        P_gX_1h = P_gX_1h_cen + P_gX_1h_sat

        I_X = jnp.trapezoid(dndm_j[None, :] * bias_j[None, :] * X_uk_j, m_jnp, axis=1)
        P_gX_2h = b_eff * pk_lin_j * I_X

        P_gX_gas = P_gX_1h + P_gX_2h

        # AGN contribution: point source (flat in k)
        if self._agn is not None:
            X_uk_agn_j = jnp.asarray(
                self._agn.agn_emissivity_uk(sc["k_np"], m_np, z, theta_cosmo,
                                            **(agn_kwargs or {}))
            )  # (Nk, NM), in L_X/1e43 [dimensionless normalized luminosity]

            if self._agn_has_hod:
                # Independent AGN occupation (HODAgnModel): centrals at the halo
                # centre, satellites following the NFW profile uk_j.  X_uk_agn_j
                # is the luminosity *per occupied AGN*, so multiply by the AGN
                # occupation here.  The 1-halo normalization keeps 1/n_gal (it
                # belongs to the galaxy leg of this g×X cross-power).
                nc_agn, ns_agn = self._agn.nc_ns_agn(np.log10(m_np))
                nc_agn_j = jnp.asarray(nc_agn)
                ns_agn_j = jnp.asarray(ns_agn)
                agn_occ_1h = nc_agn_j[None, :] + ns_agn_j[None, :] * uk_j
                agn_occ_2h = nc_agn_j[None, :] + ns_agn_j[None, :] * uk_j
            else:
                # Legacy point source: weight by the galaxy HOD, with only a
                # fraction f_sat_agn of satellites hosting AGN.
                f_sat_agn = float(self._agn._f_sat_agn)
                agn_occ_1h = nc_j[None, :] + f_sat_agn * ns_j[None, :] * uk_j
                agn_occ_2h = jnp.ones_like(uk_j)

            P_gX_agn_1h = jnp.trapezoid(
                dndm_j[None, :] * agn_occ_1h * X_uk_agn_j, m_jnp, axis=1
            ) / n_gal
            I_X_agn = jnp.trapezoid(
                dndm_j[None, :] * bias_j[None, :] * agn_occ_2h * X_uk_agn_j, m_jnp, axis=1
            )
            P_gX_agn_raw = P_gX_agn_1h + b_eff * pk_lin_j * I_X_agn

            # Convert L_X/1e43 → same units as gas emissivity FT so P_gX_agn is
            # dimensionally consistent with P_gX_gas.
            #   X̃_agn = L_X / (Λ_ref [erg cm³/s] × (cm per Mpc/h)³)
            # where Λ_ref is a reference cooling-function value at T=1 keV, Z=0.3 Z☉.
            # For the APEC table path, we use the actual table value; for the
            # deprecated plain-emissivity path we use the old power-law reference.
            h_val = float(theta_cosmo.get("h", 0.6736))
            mpc_cm_h = _MPC_CM / h_val          # cm per (Mpc/h)
            agn_conv = 1e43 / (lambda_ref * mpc_cm_h ** 3)   # lambda_ref hoisted above
            P_gX_agn = P_gX_agn_raw * agn_conv
        else:
            P_gX_agn = jnp.zeros_like(P_gX_gas)

        P_gX = P_gX_gas + P_gX_agn

        log_k = jnp.log(jnp.asarray(sc["k_np"]))
        return {
            "log_k":          log_k,
            "log_pgX":        _safe_log(P_gX),
            "log_pgX_1h":     _safe_log(P_gX_1h),
            "log_pgX_1h_cen": _safe_log(P_gX_1h_cen),
            "log_pgX_1h_sat": _safe_log(P_gX_1h_sat),
            "log_pgX_2h":     _safe_log(P_gX_2h),
            "log_pgX_gas":    _safe_log(P_gX_gas),
            "log_pgX_agn":    _safe_log(P_gX_agn),
            "n_gal":          n_gal,
            "b_eff":          b_eff,
        }

    def _pk_tables_XX(
        self,
        z: float,
        theta_cosmo: dict,
        beta_gas: float | None = None,
        beta_pressure: float | None = None,
    ) -> dict:
        """Compute P_{X,X}(k) — the X-ray emissivity auto-power spectrum.

        Unlike :meth:`_pk_tables_gX`, no galaxy HOD weighting is applied.
        The 1-halo term expands the squared total emissivity, exposing the
        gas–AGN cross-term that is absent in P_{g,X}:

        .. math::

            P^{1h}_{XX}(k) = \\int \\frac{dn}{dM}
            \\left[\\tilde{X}_{\\rm gas}^2 + 2\\tilde{X}_{\\rm gas}\\tilde{X}_{\\rm agn}
            + \\tilde{X}_{\\rm agn}^2\\right] dM

        Returns
        -------
        dict with keys:
          log_k, log_pXX, log_pXX_gas_gas, log_pXX_cross, log_pXX_agn_agn, log_pXX_2h
        """
        # P_{X,X} has no galaxy weighting, but populating the static cosmology
        # cache requires *some* valid hod_params dict (it is discarded here —
        # the cache key depends only on z, theta_cosmo).
        sc = self._get_static_cache(z, theta_cosmo, hod_params=self._fhmp._hod.default_params())
        m_np   = sc["m_np"]
        dndm   = sc["dndm_np"]
        bias   = sc["bias_np"]
        pk_lin = sc["pk_lin"]

        _has_full = (
            self._dp is not None
            and isinstance(self._pp, PressureProfileDPM)
            and self._mp is not None
            and self._cooling_fn is not None
        )
        # Λ_ref puts the full-APEC gas emissivity and the AGN luminosity on the
        # same n_e²-scale convention (see _pk_tables_gX).
        if self._cooling_fn is not None:
            lambda_ref = float(self._cooling_fn(np.array([1.0]), np.array([0.3]))[0])
        else:
            lambda_ref = 3e-23

        if _has_full:
            X_uk = self._dp.emissivity_full_uk(
                k_arr               = sc["k_np"],
                m200_arr            = sc["m_np"],
                r200_arr            = sc["r_delta"],
                z                   = z,
                theta_cosmo         = theta_cosmo,
                pressure_profile    = self._pp,
                metallicity_profile = self._mp,
                cooling_fn          = self._cooling_fn,
            )
            # Normalise to the n_e²-scale so the squared gas×gas term does not
            # underflow float32 (see _pk_tables_gX); retains Λ(T,Z)/Λ_ref.
            X_uk = np.asarray(X_uk) / lambda_ref
        else:
            X_uk = self._density_uk_cached(z, theta_cosmo, sc, emissivity=True)

        m_jnp    = jnp.asarray(m_np)
        dndm_j   = jnp.asarray(dndm)
        bias_j   = jnp.asarray(bias)
        pk_lin_j = jnp.asarray(pk_lin)
        X_uk_j   = jnp.asarray(X_uk)   # (Nk, NM) gas emissivity FT

        # Same mass-slope tilts as in _pk_tables_gX
        M12_j = m_jnp / 1.0e12
        if beta_gas is not None and self._dp is not None:
            delta_beta = float(beta_gas) - float(self._dp._beta)
            X_uk_j = X_uk_j * M12_j[None, :] ** (2.0 * delta_beta)
        if beta_pressure is not None:
            _BETA_P_REF = 0.85
            delta_beta_P = float(beta_pressure) - _BETA_P_REF
            X_uk_j = X_uk_j * M12_j[None, :] ** (0.5 * delta_beta_P)

        # AGN emissivity — convert to same physical units as gas before squaring
        if self._agn is not None:
            X_uk_agn_raw = jnp.asarray(
                self._agn.agn_emissivity_uk(sc["k_np"], m_np, z, theta_cosmo)
            )  # (Nk, NM), in L_X/1e43
            h_val = float(theta_cosmo.get("h", 0.6736))
            mpc_cm_h = _MPC_CM / h_val
            agn_conv = 1e43 / (lambda_ref * mpc_cm_h ** 3)   # lambda_ref hoisted above
            X_uk_agn_j = X_uk_agn_raw * agn_conv   # same units as X_uk_j (per object if HOD)
        else:
            X_uk_agn_j = jnp.zeros_like(X_uk_j)

        # 1-halo terms.  For an HOD AGN model, X_uk_agn_j is the luminosity *per
        # occupied AGN* and the point-source structure is set by the occupation:
        # central at the halo centre (flat) + N_sat satellites on the NFW
        # profile uk_j.  The auto/cross terms then mirror the galaxy HOD power
        # spectrum (Lau et al. 2024, App. A), luminosity-weighted by X_uk_agn_j.
        if self._agn_has_hod:
            uk_j = jnp.asarray(sc["uk"])                          # (Nk, NM) NFW FT
            nc_agn, ns_agn = self._agn.nc_ns_agn(np.log10(m_np))
            nc_agn_j = jnp.asarray(nc_agn)[None, :]
            ns_agn_j = jnp.asarray(ns_agn)[None, :]
            # cen-sat + sat-sat pair structure (no central self-pair)
            agn_pair = 2.0 * nc_agn_j * ns_agn_j * uk_j + ns_agn_j ** 2 * uk_j ** 2
            P_XX_1h_agn_agn = jnp.trapezoid(
                dndm_j[None, :] * agn_pair * X_uk_agn_j ** 2, m_jnp, axis=1
            )
            agn_occ = nc_agn_j + ns_agn_j * uk_j                  # cen + sat (for cross/2h)
            P_XX_1h_cross = 2.0 * jnp.trapezoid(
                dndm_j[None, :] * X_uk_j * agn_occ * X_uk_agn_j, m_jnp, axis=1
            )
            I_agn = jnp.trapezoid(
                dndm_j[None, :] * bias_j[None, :] * agn_occ * X_uk_agn_j, m_jnp, axis=1
            )
        else:
            P_XX_1h_agn_agn = jnp.trapezoid(dndm_j[None, :] * X_uk_agn_j ** 2,          m_jnp, axis=1)
            P_XX_1h_cross   = 2.0 * jnp.trapezoid(dndm_j[None, :] * X_uk_j * X_uk_agn_j, m_jnp, axis=1)
            I_agn = jnp.trapezoid(dndm_j[None, :] * bias_j[None, :] * X_uk_agn_j,    m_jnp, axis=1)

        P_XX_1h_gas_gas = jnp.trapezoid(dndm_j[None, :] * X_uk_j ** 2,              m_jnp, axis=1)
        P_XX_1h = P_XX_1h_gas_gas + P_XX_1h_cross + P_XX_1h_agn_agn

        # 2-halo terms
        I_gas = jnp.trapezoid(dndm_j[None, :] * bias_j[None, :] * X_uk_j,        m_jnp, axis=1)
        P_XX_2h_gas_gas = pk_lin_j * I_gas ** 2
        P_XX_2h_cross   = 2.0 * pk_lin_j * I_gas * I_agn
        P_XX_2h_agn_agn = pk_lin_j * I_agn ** 2
        P_XX_2h = P_XX_2h_gas_gas + P_XX_2h_cross + P_XX_2h_agn_agn

        P_XX = P_XX_1h + P_XX_2h

        log_k = jnp.log(jnp.asarray(sc["k_np"]))
        return {
            "log_k":            log_k,
            "log_pXX":          _safe_log(P_XX,              1e-120),
            "log_pXX_gas_gas":  _safe_log(P_XX_1h_gas_gas + P_XX_2h_gas_gas, 1e-120),
            "log_pXX_cross":    _safe_log(P_XX_1h_cross   + P_XX_2h_cross,   1e-120),
            "log_pXX_agn_agn":  _safe_log(P_XX_1h_agn_agn + P_XX_2h_agn_agn, 1e-120),
            "log_pXX_2h":       _safe_log(P_XX_2h,          1e-120),
            "log_pXX_1h":       _safe_log(P_XX_1h,          1e-120),
        }

    # ------------------------------------------------------------------
    # Projected observables
    # ------------------------------------------------------------------

    def projected_gy(
        self,
        rp_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> np.ndarray:
        """Projected galaxy × y signal Σ_y(r_p) [dimensionless Compton-y].

        Computes the Abel projection of P_{g,y}(k):

        .. math::

            \\Sigma_y(r_p) = \\frac{1}{2\\pi^2}
            \\int_0^\\infty k\\,P_{g,y}(k)\\,J_0(k r_p)\\,dk

        Parameters
        ----------
        rp_arr : (NR,) projected separations [Mpc/h]
        z, theta_cosmo, hod_params : as for ``_pk_tables_gy``

        Returns
        -------
        sigma_y : (NR,) [dimensionless]
        """
        tables = self._pk_tables_gy(z, theta_cosmo, hod_params)
        return _pk_to_wp(
            np.asarray(rp_arr),
            tables["log_k"],
            tables["log_pgy"],
        )

    def projected_gX(
        self,
        rp_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> np.ndarray:
        """Projected galaxy × X-ray emissivity w_{g,X}(r_p).

        Same Abel projection as :meth:`projected_gy` but for P_{g,X}(k).
        Units: (Mpc/h)³ cm⁻⁶ × (h/Mpc)² = (Mpc/h) cm⁻⁶.
        Multiply by the effective cooling function Λ_eff [erg cm³ s⁻¹] to
        compare to surface-brightness data.

        Parameters
        ----------
        rp_arr : (NR,) [Mpc/h]
        z, theta_cosmo, hod_params : as for ``_pk_tables_gX``

        Returns
        -------
        wgX : (NR,) [(Mpc/h) cm⁻⁶]
        """
        tables = self._pk_tables_gX(z, theta_cosmo, hod_params)
        return _pk_to_wp(
            np.asarray(rp_arr),
            tables["log_k"],
            tables["log_pgX"],
        )

    def angular_cl_gy(
        self,
        ell_arr: np.ndarray,
        z_arr: np.ndarray,
        nz_g: np.ndarray,
        theta_cosmo: dict,
        hod_params: dict,
        psf_fwhm_arcsec: float | None = None,
        psf_king_theta_c_arcsec: float | None = None,
        psf_king_alpha: float = 1.5,
    ) -> np.ndarray:
        """Angular cross-power spectrum C_ℓ^{g,y} via the Limber approximation.

        Under Limber (Limber 1953; LoVerde & Afshordi 2008):

        .. math::

            C_\\ell^{g,y} = \\int_0^{\\chi_{\\max}} \\frac{\\mathrm{d}\\chi}{\\chi^2}
            W_g(\\chi)\\,P_{g,y}\\!\\left(k=\\frac{\\ell+\\tfrac{1}{2}}{\\chi},
            z(\\chi)\\right)

        where :math:`W_g(\\chi) = \\mathrm{d}N_g/\\mathrm{d}\\chi` (normalized).
        The y-field window is unity (already a LOS integral).

        Parameters
        ----------
        ell_arr : (Nell,) angular multipoles
        z_arr : (Nz,) redshift array for n(z) [must bracket the galaxy distribution]
        nz_g : (Nz,) dN/dz of the galaxy sample (will be normalized internally)
        theta_cosmo : cosmological parameters
        hod_params : HOD parameters
        psf_fwhm_arcsec : float | None
            If given, multiply C_ℓ by the Gaussian PSF window B_ℓ (single field).
        psf_king_theta_c_arcsec : float | None
            If given, multiply C_ℓ by the analytic King-profile PSF window
            B_ℓ = exp(−ℓ θ_c) for α=3/2, or the general Bessel-K form.
            Cannot be used together with ``psf_fwhm_arcsec``.
        psf_king_alpha : float
            King slope for the analytic PSF window. Default 1.5.

        Returns
        -------
        cl_gy : (Nell,) [(Mpc/h)²] (dimensionless after h-unit cancellation with χ²)
        """
        from hod_mod.cosmology.distances import comoving_distance

        z_arr  = np.asarray(z_arr,  dtype=float)
        nz_g   = np.asarray(nz_g,   dtype=float)
        ell    = jnp.asarray(ell_arr)

        h       = float(theta_cosmo["h"])
        omega_m = float(theta_cosmo["Omega_m"])
        chi_z = np.array([
            float(np.asarray(comoving_distance(float(zi), h, omega_m)).ravel()[0]) * h
            for zi in z_arr
        ])

        dndchi_j = jnp.asarray(nz_g) / jnp.trapezoid(jnp.asarray(nz_g), jnp.asarray(chi_z))
        chi_z_j  = jnp.asarray(chi_z)

        raw_gy = [self._pk_tables_gy(zi, theta_cosmo, hod_params) for zi in z_arr]
        log_k_ref_gy = np.asarray(raw_gy[0]["log_k"])
        log_pgy_stack = jnp.stack([jnp.asarray(t["log_pgy"]) for t in raw_gy])  # (Nz, Nk)
        log_k_j_gy   = jnp.asarray(log_k_ref_gy)

        ell_j    = jnp.asarray(ell_arr, dtype=float)
        k_lim_gy = jnp.log(jnp.maximum((ell_j[:, None] + 0.5) / chi_z_j[None, :], 1e-4))

        def _interp_one_gy(lkq, lpt):
            return jnp.exp(jnp.interp(lkq, log_k_j_gy, lpt))

        _interp_z_gy   = jax.vmap(_interp_one_gy, in_axes=(0, 0))
        _interp_ellz_gy = jax.vmap(_interp_z_gy,  in_axes=(0, None))

        pgy_mat   = _interp_ellz_gy(k_lim_gy, log_pgy_stack)                       # (Nell, Nz)
        integrand = dndchi_j[None, :] * pgy_mat / chi_z_j[None, :] ** 2
        cl_gy     = jnp.trapezoid(integrand, chi_z_j, axis=1)

        if psf_fwhm_arcsec is not None and psf_king_theta_c_arcsec is not None:
            raise ValueError("Specify at most one of psf_fwhm_arcsec or psf_king_theta_c_arcsec.")
        if psf_fwhm_arcsec is not None:
            cl_gy = cl_gy * psf_window_ell(ell_j, psf_fwhm_arcsec)
        elif psf_king_theta_c_arcsec is not None:
            cl_gy = cl_gy * psf_king_window_ell(ell_j, psf_king_theta_c_arcsec, psf_king_alpha)
        return cl_gy

    def angular_cl_gX(
        self,
        ell_arr: np.ndarray,
        z_arr: np.ndarray,
        nz_g: np.ndarray,
        theta_cosmo: dict,
        hod_params: dict,
        psf_fwhm_arcsec: float | None = None,
        psf_king_theta_c_arcsec: float | None = None,
        psf_king_alpha: float = 1.5,
        return_components: bool = False,
        n_workers: int = -1,
        beta_gas: float | None = None,
        beta_pressure: float | None = None,
        agn_kwargs: dict | None = None,
    ) -> "np.ndarray | dict":
        """Angular cross-power spectrum C_ℓ^{g,X} via the Limber approximation.

        Identical structure to :meth:`angular_cl_gy` but for the X-ray emissivity
        field (DPM Model).  The returned spectrum has units of the emissivity
        power spectrum [(Mpc/h)³ cm⁻⁶] divided by χ² [(Mpc/h)²], giving
        [(Mpc/h) cm⁻⁶].

        .. math::

            C_\\ell^{g,X} = \\int_0^{\\chi_{\\max}} \\frac{\\mathrm{d}\\chi}{\\chi^2}
            W_g(\\chi)\\,P_{g,X}\\!\\left(k=\\frac{\\ell+\\tfrac{1}{2}}{\\chi},
            z(\\chi)\\right)

        Parameters
        ----------
        ell_arr : (Nell,) angular multipoles
        z_arr : (Nz,) redshift array bracketing the galaxy distribution
        nz_g : (Nz,) dN/dz of the galaxy sample (normalized internally)
        theta_cosmo : cosmological parameters
        hod_params : HOD parameters
        psf_fwhm_arcsec : float | None
            eROSITA PSF FWHM [arcsec].  If given, multiply C_ℓ by the Gaussian
            PSF window B_ℓ = exp(−ℓ²σ²/2) (single-field convolution).
            Use 30.0 for the eROSITA soft X-ray PSF.
        psf_king_theta_c_arcsec : float | None
            King core radius [arcsec] for the analytic PSF window.  If given,
            multiply C_ℓ by B_ℓ = exp(−ℓ θ_c) (α=3/2) or the general
            Bessel-K form.  Fitted to eROSITA TM CalDB on-axis: 8.64".
            Cannot be used together with ``psf_fwhm_arcsec``.
        psf_king_alpha : float
            King slope for the analytic PSF window. Default 1.5.
        return_components : bool
            If True return a dict ``{"total", "gas", "agn"}`` instead of the
            total array.
        n_workers : int
            Number of threads for parallel evaluation of ``_pk_tables_gX`` at
            each redshift.  -1 (default) uses ``os.cpu_count()``.  Set to 1 to
            disable parallelism.  The z-points are independent, so thread-based
            parallelism is safe because JAX releases the GIL during computation.

        Returns
        -------
        cl_gX : (Nell,) [(Mpc/h) cm⁻⁶]  or dict when return_components=True
        """
        from hod_mod.cosmology.distances import comoving_distance

        z_arr  = np.asarray(z_arr,  dtype=float)
        nz_g   = np.asarray(nz_g,   dtype=float)
        ell    = jnp.asarray(ell_arr)

        h       = float(theta_cosmo["h"])
        omega_m = float(theta_cosmo["Omega_m"])
        chi_z = np.array([
            float(np.asarray(comoving_distance(float(zi), h, omega_m)).ravel()[0]) * h
            for zi in z_arr
        ])

        dndchi_j = jnp.asarray(nz_g) / jnp.trapezoid(jnp.asarray(nz_g), jnp.asarray(chi_z))
        chi_z_j  = jnp.asarray(chi_z)

        # ------------------------------------------------------------------ #
        # Step 1: build P_{g,X}(k) tables at each redshift.                  #
        # Each z-point is independent → parallelize with threads.            #
        # JAX releases the GIL during compute, so ThreadPoolExecutor is safe.#
        # ------------------------------------------------------------------ #
        nz = len(z_arr)
        _nw = os.cpu_count() if n_workers == -1 else n_workers

        def _tables_at_z(zi):
            return self._pk_tables_gX(
                zi, theta_cosmo, hod_params,
                beta_gas=beta_gas,
                beta_pressure=beta_pressure,
                agn_kwargs=agn_kwargs,
            )

        if _nw == 1 or nz == 1:
            raw_tables = [_tables_at_z(zi) for zi in z_arr]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(_nw, nz)) as pool:
                raw_tables = list(pool.map(_tables_at_z, z_arr))

        # Stack per-component log-P tables into (Nz, Nk) arrays for fast
        # vectorized Limber integration below.
        log_k_ref = np.asarray(raw_tables[0]["log_k"])   # (Nk,) — same grid for all z
        Nk = len(log_k_ref)
        log_pgX_stack = {
            comp: jnp.stack([jnp.asarray(raw_tables[i][key]) for i in range(nz)])
            for comp, key in (
                ("total",       "log_pgX"),
                ("gas",         "log_pgX_gas"),
                ("gas_1h_cen",  "log_pgX_1h_cen"),
                ("gas_1h_sat",  "log_pgX_1h_sat"),
                ("gas_2h",      "log_pgX_2h"),
                ("agn",         "log_pgX_agn"),
            )
        }  # each entry shape (Nz, Nk)
        log_k_j = jnp.asarray(log_k_ref)   # (Nk,)

        # ------------------------------------------------------------------ #
        # Step 2: Limber integral — vectorized over ℓ.                       #
        #                                                                     #
        # For each ℓ:  k_Limber(z) = (ℓ + 0.5) / χ(z)                      #
        # Then interpolate log P(k_Limber, z) for each z, sum over z.        #
        #                                                                     #
        # Build k_limber for ALL ℓ at once: shape (Nell, Nz).               #
        # Use jax.vmap over ℓ to call 1-D interp at each (ℓ, z) pair.       #
        # ------------------------------------------------------------------ #
        ell_j     = jnp.asarray(ell_arr, dtype=float)                # (Nell,)
        k_limber  = (ell_j[:, None] + 0.5) / chi_z_j[None, :]        # (Nell, Nz)
        log_klim  = jnp.log(jnp.maximum(k_limber, 1e-4))             # (Nell, Nz)

        def _interp_one(log_k_query, log_p_table):
            """Interpolate log P at a single (ℓ, z) pair."""
            return jnp.exp(jnp.interp(log_k_query, log_k_j, log_p_table))

        # vmap over z-axis, then over ℓ-axis
        _interp_z   = jax.vmap(_interp_one, in_axes=(0, 0))   # over Nz
        _interp_ellz = jax.vmap(_interp_z,  in_axes=(0, None)) # over Nell

        def _limber_integral(component_key):
            # log_pgX_stack[component_key] : (Nz, Nk)
            # _interp_ellz maps (Nell, Nz) × (Nz, Nk) → (Nell, Nz)
            pk_mat = _interp_ellz(log_klim, log_pgX_stack[component_key])  # (Nell, Nz)
            integrand = dndchi_j[None, :] * pk_mat / chi_z_j[None, :] ** 2  # (Nell, Nz)
            return jnp.trapezoid(integrand, chi_z_j, axis=1)               # (Nell,)

        cl_gas         = _limber_integral("gas")
        cl_gas_1h_cen  = _limber_integral("gas_1h_cen")
        cl_gas_1h_sat  = _limber_integral("gas_1h_sat")
        cl_gas_2h      = _limber_integral("gas_2h")
        cl_agn         = _limber_integral("agn")
        cl_gX          = cl_gas + cl_agn

        if psf_fwhm_arcsec is not None and psf_king_theta_c_arcsec is not None:
            raise ValueError("Specify at most one of psf_fwhm_arcsec or psf_king_theta_c_arcsec.")
        if psf_fwhm_arcsec is not None:
            psf = psf_window_ell(ell, psf_fwhm_arcsec)
            cl_gas        = cl_gas        * psf
            cl_gas_1h_cen = cl_gas_1h_cen * psf
            cl_gas_1h_sat = cl_gas_1h_sat * psf
            cl_gas_2h     = cl_gas_2h     * psf
            cl_agn        = cl_agn        * psf
            cl_gX         = cl_gX         * psf
        elif psf_king_theta_c_arcsec is not None:
            psf = psf_king_window_ell(ell, psf_king_theta_c_arcsec, psf_king_alpha)
            cl_gas        = cl_gas        * psf
            cl_gas_1h_cen = cl_gas_1h_cen * psf
            cl_gas_1h_sat = cl_gas_1h_sat * psf
            cl_gas_2h     = cl_gas_2h     * psf
            cl_agn        = cl_agn        * psf
            cl_gX         = cl_gX         * psf

        if return_components:
            return {
                "total":      np.asarray(cl_gX),
                "gas":        np.asarray(cl_gas),
                "gas_1h_cen": np.asarray(cl_gas_1h_cen),
                "gas_1h_sat": np.asarray(cl_gas_1h_sat),
                "gas_2h":     np.asarray(cl_gas_2h),
                "agn":        np.asarray(cl_agn),
            }
        return cl_gX

    def angular_cl_XX(
        self,
        ell_arr: np.ndarray,
        z_arr: np.ndarray,
        nz_X: np.ndarray,
        theta_cosmo: dict,
        psf_fwhm_arcsec: float | None = None,
        psf_king_theta_c_arcsec: float | None = None,
        psf_king_alpha: float = 1.5,
        return_components: bool = False,
        n_workers: int = -1,
        beta_gas: float | None = None,
        beta_pressure: float | None = None,
    ) -> "np.ndarray | dict":
        """Angular auto-power spectrum C_ℓ^{X,X} of the total X-ray emission.

        Includes the 1-halo and 2-halo gas–AGN cross-terms that vanish in
        :meth:`angular_cl_gX` (which is exact and linear in X). See
        :meth:`_pk_tables_XX` for the underlying P_{X,X}(k) decomposition.

        .. math::

            C_\\ell^{X,X} = \\int_0^{\\chi_{\\max}} \\frac{\\mathrm{d}\\chi}{\\chi^2}
            W_X(\\chi)^2\\,P_{X,X}\\!\\left(k=\\frac{\\ell+\\tfrac{1}{2}}{\\chi},
            z(\\chi)\\right)

        Parameters
        ----------
        ell_arr : (Nell,) angular multipoles
        z_arr : (Nz,) redshift array bracketing the X-ray source distribution
        nz_X : (Nz,) X-ray window function (e.g. emissivity-weighted dV/dz,
            or a matched source dN/dz). Normalized internally like ``nz_g``
            in :meth:`angular_cl_gX`, but appears **squared** in the Limber
            integral since both legs of the correlation are the X-ray field.
        return_components : bool
            If True return ``{"total", "gas_gas", "cross", "agn_agn"}``.

        Returns
        -------
        cl_XX : (Nell,) or dict when return_components=True
        """
        from hod_mod.cosmology.distances import comoving_distance

        z_arr = np.asarray(z_arr, dtype=float)
        nz_X  = np.asarray(nz_X,  dtype=float)
        ell   = jnp.asarray(ell_arr)

        h       = float(theta_cosmo["h"])
        omega_m = float(theta_cosmo["Omega_m"])
        chi_z = np.array([
            float(np.asarray(comoving_distance(float(zi), h, omega_m)).ravel()[0]) * h
            for zi in z_arr
        ])

        dndchi_j = jnp.asarray(nz_X) / jnp.trapezoid(jnp.asarray(nz_X), jnp.asarray(chi_z))
        chi_z_j  = jnp.asarray(chi_z)

        nz  = len(z_arr)
        _nw = os.cpu_count() if n_workers == -1 else n_workers

        def _tables_at_z(zi):
            return self._pk_tables_XX(
                zi, theta_cosmo, beta_gas=beta_gas, beta_pressure=beta_pressure,
            )

        if _nw == 1 or nz == 1:
            raw_tables = [_tables_at_z(zi) for zi in z_arr]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(_nw, nz)) as pool:
                raw_tables = list(pool.map(_tables_at_z, z_arr))

        log_k_ref = np.asarray(raw_tables[0]["log_k"])
        log_pXX_stack = {
            comp: jnp.stack([jnp.asarray(raw_tables[i][key]) for i in range(nz)])
            for comp, key in (
                ("total",    "log_pXX"),
                ("gas_gas",  "log_pXX_gas_gas"),
                ("cross",    "log_pXX_cross"),
                ("agn_agn",  "log_pXX_agn_agn"),
            )
        }
        log_k_j = jnp.asarray(log_k_ref)

        ell_j    = jnp.asarray(ell_arr, dtype=float)
        k_limber = (ell_j[:, None] + 0.5) / chi_z_j[None, :]
        log_klim = jnp.log(jnp.maximum(k_limber, 1e-4))

        def _interp_one(log_k_query, log_p_table):
            return jnp.exp(jnp.interp(log_k_query, log_k_j, log_p_table))

        _interp_z    = jax.vmap(_interp_one, in_axes=(0, 0))
        _interp_ellz = jax.vmap(_interp_z,  in_axes=(0, None))

        def _limber_integral(component_key):
            pk_mat = _interp_ellz(log_klim, log_pXX_stack[component_key])
            # W_X(χ) appears twice (both legs of the auto-correlation).
            integrand = dndchi_j[None, :] ** 2 * pk_mat / chi_z_j[None, :] ** 2
            return jnp.trapezoid(integrand, chi_z_j, axis=1)

        cl_gas_gas = _limber_integral("gas_gas")
        cl_cross   = _limber_integral("cross")
        cl_agn_agn = _limber_integral("agn_agn")
        cl_XX      = cl_gas_gas + cl_cross + cl_agn_agn

        if psf_fwhm_arcsec is not None and psf_king_theta_c_arcsec is not None:
            raise ValueError("Specify at most one of psf_fwhm_arcsec or psf_king_theta_c_arcsec.")
        if psf_fwhm_arcsec is not None:
            psf = psf_window_ell(ell, psf_fwhm_arcsec)
        elif psf_king_theta_c_arcsec is not None:
            psf = psf_king_window_ell(ell, psf_king_theta_c_arcsec, psf_king_alpha)
        else:
            psf = None
        if psf is not None:
            cl_gas_gas = cl_gas_gas * psf
            cl_cross   = cl_cross   * psf
            cl_agn_agn = cl_agn_agn * psf
            cl_XX      = cl_XX      * psf

        if return_components:
            return {
                "total":    np.asarray(cl_XX),
                "gas_gas":  np.asarray(cl_gas_gas),
                "cross":    np.asarray(cl_cross),
                "agn_agn":  np.asarray(cl_agn_agn),
            }
        return cl_XX
