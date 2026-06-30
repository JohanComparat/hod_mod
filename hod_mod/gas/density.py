"""Hot-gas density profile (Oppenheimer+2025 DPM)."""
from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
from hod_mod.core.concentration import c_diemer15
from hod_mod.core.concentration import _neff_eisenstein_hu
from hod_mod.core.halo_mass_function import HaloMassFunction
from .conversions import _eh_pk_3arg, _profile_uk_gl, _profile_uk_gl_bands
from .cooling import temperature_from_profiles

if TYPE_CHECKING:  # annotation-only refs (avoid runtime circular imports)
    from .pressure import PressureProfileDPM
    from .metallicity import MetallicityProfileDPM
    from .cooling import ApecCoolingTable


# ---------------------------------------------------------------------------
# DPM electron density profile  (soft X-ray)
# ---------------------------------------------------------------------------

class GasDensityDPM:
    """DPM electron density profile for X-ray emissivity (Oppenheimer+2025).

    Reference: Oppenheimer et al. 2025, arXiv:2505.14782, Table 1.

    The profile uses a generalized NFW (gNFW) shape function:

    .. math::

        f(x|\\alpha) = x^{-\\alpha_{\\rm in}}
            \\left(1 + x^{\\alpha_{\\rm tr}}\\right)^{(\\alpha_{\\rm in}
            - \\alpha_{\\rm out})/\\alpha_{\\rm tr}}

    where :math:`x = r/R_s` and :math:`R_s = R_{200}/c_{\\rm DPM}` with
    :math:`c_{\\rm DPM} = 2.772` (Table 1 of arXiv:2505.14782).

    The electron density is:

    .. math::

        n_e(r, M_{200}, z) = n_{e0}\\,f(x|\\alpha^{n_e})
            \\,E(z)^{\\gamma^{n_e}}\\,M_{12}^{\\beta^{n_e}}

    where :math:`M_{12} = M_{200}/(10^{12}\\,M_\\odot)`,
    :math:`E(z) = H(z)/H_0`, and :math:`n_{e0}` is normalised so that
    :math:`n_e(0.3 R_{200}, 10^{12}\\,M_\\odot, z=0) = n_{e,0.3}`.

    Three calibrated models are provided (Table 1 of arXiv:2505.14782):

    * Model 1 — self-similar (β=0)
    * Model 2 — cluster-reduced slope (β=0.36)
    * Model 3 — slope-changing outer profile

    Parameters
    ----------
    model : int (1, 2, or 3), default 2
    r_max_over_r200 : float (default 3.0)
    n_gl : int (default 200)
    """

    # DPM scale radius: R_s = R₂₀₀ / c_DPM  (Table 1 of arXiv:2505.14782)
    _C_DPM = 2.772

    # Model parameters from Table 1 of arXiv:2505.14782
    # ne_03: n_e at r=0.3 R₂₀₀, M=10^12 M☉, z=0  [cm⁻³]
    # alpha_in, alpha_tr, alpha_out: gNFW shape exponents
    # beta: mass scaling exponent
    # gamma: redshift scaling exponent
    _PARAMS = {
        1: dict(ne_03=5.86e-4, alpha_in=1.0, alpha_tr=1.9, alpha_out=2.7, beta=0.00, gamma=2.0),
        2: dict(ne_03=4.87e-5, alpha_in=1.0, alpha_tr=1.9, alpha_out=2.7, beta=0.36, gamma=2.0),
        3: dict(ne_03=4.87e-5, alpha_in=0.4, alpha_tr=0.45, alpha_out=0.5, beta=0.36, gamma=2.0),
    }

    def __init__(
        self,
        model: int = 2,
        r_max_over_r200: float = 3.0,
        n_gl: int = 200,
        sigma_scatter: float = 0.0,
        concentration_model: str = "diemer19",
    ):
        """
        Parameters
        ----------
        sigma_scatter : float
            Log-normal scatter σ in the profile amplitude (dex).  Adds a boost
            factor exp((σ ln 10)²) to the emissivity FT (⟨n_e²⟩ / ⟨n_e⟩²  for
            log-normal; DPM Eq. 6).  Default 0 = no scatter.
        concentration_model : str
            ``"diemer19"`` (default) — mass-dependent c(M, z) from Diemer & Joyce 2019
            via ``c_diemer15``.  ``"fixed"`` — original constant ``_C_DPM = 2.772``.
        """
        if model not in self._PARAMS:
            raise ValueError(f"model must be 1, 2, or 3; got {model}")
        if concentration_model not in ("diemer19", "fixed"):
            raise ValueError(f"concentration_model must be 'diemer19' or 'fixed'; got {concentration_model!r}")
        self._model = model
        self._r_max_factor = float(r_max_over_r200)
        self._n_gl = int(n_gl)
        p = self._PARAMS[model]
        self._ne_03    = p["ne_03"]
        self._alpha_in = p["alpha_in"]
        self._alpha_tr = p["alpha_tr"]
        self._alpha_out= p["alpha_out"]
        self._beta     = p["beta"]
        self._gamma    = p["gamma"]
        self._concentration_model = concentration_model
        if concentration_model == "diemer19":
            self._hmf = HaloMassFunction(_eh_pk_3arg, model="tinker08")
        # Normalisation at fixed c_DPM (kept for reference / "fixed" mode)
        self._f_xref = self._gnfw_f(0.3 * self._C_DPM)
        self._ne0 = self._ne_03 / self._f_xref   # [cm⁻³]
        # Log-normal scatter boost (DPM Eq. 6): ⟨n_e²⟩ = ⟨n_e⟩² × exp(σ²)
        # where σ = sigma_scatter × ln(10)
        self._scatter_boost = float(np.exp((float(sigma_scatter) * np.log(10.0)) ** 2))

    def _gnfw_f(self, x: np.ndarray) -> np.ndarray:
        """gNFW shape function f(x|α) — Eq. 1 of arXiv:2505.14782."""
        x = np.asarray(x, dtype=float)
        # Guard against x ≤ 0 (inner boundary)
        x_safe = np.where(x > 1e-8, x, 1e-8)
        return (x_safe**(-self._alpha_in)
                * (1.0 + x_safe**self._alpha_tr)
                ** ((self._alpha_in - self._alpha_out) / self._alpha_tr))

    def _concentration(
        self,
        m200_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
    ) -> np.ndarray:
        """Per-halo c₂₀₀c array from Diemer & Joyce 2019, or fixed _C_DPM.

        Parameters
        ----------
        m200_arr : (NM,) [Msun/h]
        z : redshift
        theta_cosmo : dict with at least 'Omega_m', 'h', 'sigma8', 'n_s', 'Omega_b'

        Returns
        -------
        c_arr : (NM,) dimensionless concentration c₂₀₀c
        """
        if self._concentration_model == "fixed":
            return np.full(len(m200_arr), self._C_DPM)
        m_jax   = jnp.asarray(m200_arr, dtype=float)
        sigma   = self._hmf.sigma(m_jax, float(z), theta_cosmo)
        n_eff   = _neff_eisenstein_hu(m_jax, theta_cosmo)
        omega_m = float(theta_cosmo["Omega_m"])
        return np.asarray(c_diemer15(m_jax, sigma, n_eff, omega_m, float(z)))

    def density_3d(
        self,
        r: np.ndarray,
        m200: float,
        r200: float,
        z: float,
        omega_m: float,
        c200c: float | None = None,
    ) -> np.ndarray:
        """Electron number density n_e(r|M₂₀₀, z) [cm⁻³].

        Parameters
        ----------
        r : radii [Mpc/h]
        m200 : M₂₀₀ [Msun/h]
        r200 : R₂₀₀ [Mpc/h]
        z : redshift
        omega_m : matter fraction Ω_m
        c200c : concentration c₂₀₀c for this halo (pre-computed by the caller).
            If None, falls back to the fixed class constant ``_C_DPM``.
        """
        c   = float(c200c) if c200c is not None else self._C_DPM
        r_s = r200 / c
        x   = np.asarray(r, dtype=float) / r_s
        ne0 = self._ne_03 / self._gnfw_f(0.3 * c)   # per-halo normalization
        M12 = m200 / 1.0e12
        ez  = np.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m))
        return ne0 * self._gnfw_f(x) * ez**self._gamma * M12**self._beta

    def _ne_grid(self, r_nodes, m_c, r_c, c_c, ez):
        """Vectorised n_e(r|M) on the (NM, n_gl) quadrature grid [cm⁻³].

        Same formula as :meth:`density_3d` but broadcasts over the mass axis —
        ``density_3d`` casts ``c200c`` to a Python float, which would collapse
        the mass dimension.  ``m_c``, ``r_c``, ``c_c`` are (NM, 1); ``r_nodes``
        is (NM, n_gl); ``ez`` is a scalar.
        """
        x   = r_nodes / (r_c / c_c)                         # (NM, n_gl)
        ne0 = self._ne_03 / self._gnfw_f(0.3 * c_c)         # (NM, 1)
        return ne0 * self._gnfw_f(x) * ez**self._gamma * (m_c / 1.0e12)**self._beta

    def density_uk(
        self,
        k_arr: np.ndarray,
        m200_arr: np.ndarray,
        r200_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
    ) -> np.ndarray:
        """FT of the electron density: ñ_e(k|M) = 4π ∫ n_e(r) j₀(kr) r² dr.

        Output units: (Mpc/h)³ cm⁻³.  Multiply by ``(Mpc_cm/h)³`` to convert
        to dimensionless (but this is done at the power-spectrum level when
        needed).

        Parameters
        ----------
        k_arr : (Nk,) [h/Mpc]
        m200_arr : (NM,) [Msun/h]
        r200_arr : (NM,) [Mpc/h]
        z : redshift
        theta_cosmo : dict with 'Omega_m'

        Returns
        -------
        uk : (Nk, NM) [(Mpc/h)³ cm⁻³]
        """
        omega_m = float(theta_cosmo["Omega_m"])
        m200    = np.asarray(m200_arr, dtype=float)
        r200    = np.asarray(r200_arr, dtype=float)
        k       = np.asarray(k_arr,    dtype=float)
        NM      = len(m200)
        c_arr   = np.asarray(self._concentration(m200, z, theta_cosmo), dtype=float)

        m_c = m200[:, None]; r_c = r200[:, None]; c_c = c_arr[:, None]   # (NM, 1)
        ez  = np.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m))

        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            """n_e(r, M) for all halos on the quadrature grid (vectorised)."""
            return self._ne_grid(r_nodes, m_c, r_c, c_c, ez)

        r_max = self._r_max_factor * r200   # (NM,)
        return _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)   # (Nk, NM) [(Mpc/h)³ cm⁻³]

    def emissivity_uk(
        self,
        k_arr: np.ndarray,
        m200_arr: np.ndarray,
        r200_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
    ) -> np.ndarray:
        """FT of n_e²(r) without cooling function weighting.

        .. deprecated::
            Use :meth:`emissivity_full_uk` with :class:`PressureProfileDPM`,
            :class:`MetallicityProfileDPM`, and :class:`ApecCoolingTable`.

        Output units: (Mpc/h)³ cm⁻⁶.
        """
        import warnings
        warnings.warn(
            "emissivity_uk is deprecated — use emissivity_full_uk with all three "
            "DPM profiles (PressureProfileDPM, MetallicityProfileDPM) and "
            "an ApecCoolingTable.",
            DeprecationWarning, stacklevel=2,
        )
        omega_m = float(theta_cosmo["Omega_m"])
        m200    = np.asarray(m200_arr, dtype=float)
        r200    = np.asarray(r200_arr, dtype=float)
        k       = np.asarray(k_arr,    dtype=float)
        NM      = len(m200)
        c_arr   = np.asarray(self._concentration(m200, z, theta_cosmo), dtype=float)

        m_c = m200[:, None]; r_c = r200[:, None]; c_c = c_arr[:, None]   # (NM, 1)
        ez  = np.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m))

        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            ne = self._ne_grid(r_nodes, m_c, r_c, c_c, ez)
            return ne**2 * self._scatter_boost

        r_max = self._r_max_factor * r200
        return _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)

    def emissivity_full_uk(
        self,
        k_arr: np.ndarray,
        m200_arr: np.ndarray,
        r200_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
        pressure_profile: "PressureProfileDPM",
        metallicity_profile: "MetallicityProfileDPM",
        cooling_fn: "ApecCoolingTable",
    ) -> np.ndarray:
        """FT of n_e²(r) × Λ_APEC(T(r), Z(r)) — X-ray surface brightness emissivity.

        Evaluates the full temperature- and metallicity-dependent APEC cooling
        function at each quadrature node:

        .. math::

            \\varepsilon(r) = n_e^2(r) \\times \\Lambda_{n_e^2}(T(r), Z(r))

        where :math:`T(r) = P_{\\rm DPM}(r) / n_e(r)` [keV] (ideal gas law),
        :math:`Z(r)` comes from :class:`MetallicityProfileDPM` [Z_sun], and
        :math:`\\Lambda_{n_e^2}` is the band-integrated APEC emissivity from
        :class:`ApecCoolingTable` (``0.83 × Λ_{\\rm APEC}`` converting
        :math:`n_e n_H \\to n_e^2`).

        Parameters
        ----------
        pressure_profile : PressureProfileDPM
        metallicity_profile : MetallicityProfileDPM
        cooling_fn : ApecCoolingTable
            Precomputed APEC cooling table.  Instantiate once and reuse.

        Returns
        -------
        uk : (Nk, NM) [erg cm³ s⁻¹ × (Mpc/h)³ cm⁻⁶]
        """
        omega_m = float(theta_cosmo["Omega_m"])
        m200    = np.asarray(m200_arr, dtype=float)
        r200    = np.asarray(r200_arr, dtype=float)
        k       = np.asarray(k_arr,    dtype=float)
        NM      = len(m200)
        c_arr   = np.asarray(self._concentration(m200, z, theta_cosmo), dtype=float)

        m_c = m200[:, None]; r_c = r200[:, None]; c_c = c_arr[:, None]   # (NM, 1)
        ez  = np.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m))

        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            # Vectorised over mass — pressure/metallicity already broadcast
            # (they use the fixed C_DPM concentration), density via _ne_grid.
            ne  = self._ne_grid(r_nodes, m_c, r_c, c_c, ez)               # (NM, n_gl)
            P   = pressure_profile._pressure_3d(r_nodes, m_c, r_c, z, omega_m)
            T   = temperature_from_profiles(P, ne)                        # [keV]
            Z   = metallicity_profile.metallicity_3d(r_nodes, r_c)        # [Z_sun]
            lam = cooling_fn(T, Z)
            return ne**2 * self._scatter_boost * lam

        r_max = self._r_max_factor * r200
        return _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)

    def emissivity_full_uk_bands(
        self,
        k_arr: np.ndarray,
        m200_arr: np.ndarray,
        r200_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
        pressure_profile: "PressureProfileDPM",
        metallicity_profile: "MetallicityProfileDPM",
        cooling_fns: "list",
    ) -> np.ndarray:
        """Multi-band version of :meth:`emissivity_full_uk`.

        Computes the emissivity FT ``X̃_b(k|M)`` for a LIST of energy-band cooling
        tables in ONE batched spherical-Bessel FT.  The bands share n_e(r,M),
        T(r,M), Z(r,M) and the j₀ geometry; only ``Λ_b`` differs, so this is ≈ the
        cost of a single :meth:`emissivity_full_uk` (the per-band table eval is
        cheap).  Used by the energy-band (temperature-resolved) joint fit.

        Parameters
        ----------
        cooling_fns : list of ApecCoolingTable
            One per band (e.g. 15 × ``ApecCoolingTable(emin, emax)``).

        Returns
        -------
        uk : (Nb, Nk, NM) [erg cm³ s⁻¹ × (Mpc/h)³ cm⁻⁶]
        """
        omega_m = float(theta_cosmo["Omega_m"])
        m200    = np.asarray(m200_arr, dtype=float)
        r200    = np.asarray(r200_arr, dtype=float)
        k       = np.asarray(k_arr,    dtype=float)
        c_arr   = np.asarray(self._concentration(m200, z, theta_cosmo), dtype=float)

        m_c = m200[:, None]; r_c = r200[:, None]; c_c = c_arr[:, None]   # (NM, 1)
        ez  = np.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m))

        def _integrand_bands(r_nodes: np.ndarray) -> np.ndarray:
            # n_e, T, Z built ONCE; each band just re-weights by its Λ_b(T,Z).
            ne  = self._ne_grid(r_nodes, m_c, r_c, c_c, ez)               # (NM, n_gl)
            P   = pressure_profile._pressure_3d(r_nodes, m_c, r_c, z, omega_m)
            T   = temperature_from_profiles(P, ne)                        # [keV]
            Z   = metallicity_profile.metallicity_3d(r_nodes, r_c)        # [Z_sun]
            ne2 = ne**2 * self._scatter_boost
            return np.stack([ne2 * cf(T, Z) for cf in cooling_fns])       # (Nb, NM, n_gl)

        r_max = self._r_max_factor * r200
        return _profile_uk_gl_bands(k, r_max, _integrand_bands, n_gl=self._n_gl)
