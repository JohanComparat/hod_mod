"""Gas (electron pressure and density) halo profiles and their Fourier transforms.

Provides two profile models:

* :class:`PressureProfileA10` — Arnaud, Pratt, Piffaretti et al. 2010, A&A 517, A92
  (arXiv:0910.1234), Table 1.  Generalized NFW electron pressure profile
  parameterised by M₅₀₀c and R₅₀₀c.  Used for the thermal Sunyaev-Zel'dovich (tSZ)
  effect via the Compton-y cross-power spectrum.

* :class:`GasDensityDPM` — Oppenheimer et al. 2025, arXiv:2505.14782, Table 1.
  Descriptive Parametric Model (DPM) electron density profile parameterised by
  M₂₀₀ and R₂₀₀.  Used to model the soft X-ray emissivity (∝ n_e²).

Both classes provide:
  * ``profile_3d(r_arr, M, r_delta, ...)`` — radial profile values
  * ``profile_uk(k_arr, m_arr, r_delta_arr, c_arr, z, theta_cosmo)``
    — Fourier-space window function, shape (Nk, NM), via Gauss-Legendre quadrature

Unit conventions
----------------
* Lengths in Mpc/h (comoving h-units throughout).
* Masses in Msun/h.
* Electron pressure P_e in keV cm⁻³ (physical).
* Electron number density n_e in cm⁻³ (physical).
* ``pressure_uk`` output in (Mpc/h)², consistent with a 3D galaxy × y cross-power
  spectrum P_{gy}(k) in (Mpc/h)² (one lower dimension than P_gg [(Mpc/h)³]).
  Derivation: σ_T/(m_e c²) [cm²/keV] × (Mpc_cm/h) [cm/(Mpc/h)] × P_e [keV/cm³] × r²dr [(Mpc/h)³]
  → (Mpc/h)².
* ``density_uk`` output in (Mpc/h)³ cm⁻³ (FT of n_e; for X-ray, square and multiply
  by Λ_eff to get the emissivity FT).

References
----------
Arnaud+2010  arXiv:0910.1234
Oppenheimer+2025  arXiv:2505.14782
"""

import numpy as np
import jax.numpy as jnp
from scipy.optimize import brentq

from hod_mod.cosmology.concentration import c_diemer15, _neff_eisenstein_hu
from hod_mod.cosmology.halo_mass_function import HaloMassFunction
from hod_mod.cosmology.power_spectrum import eisenstein_hu_pk


def _eh_pk_3arg(k, z, theta):
    """Wrap eisenstein_hu_pk to match HaloMassFunction's (k, z, theta) signature."""
    return eisenstein_hu_pk(k, theta)

# ---------------------------------------------------------------------------
# Physical constants (CGS / keV / Mpc units)
# ---------------------------------------------------------------------------

_SIGMA_T_CM2       = 6.6524e-25    # cm²  (Thomson cross-section)
_ME_C2_KEV         = 511.0          # keV  (electron rest-mass energy)
_MPC_CM            = 3.0857e24      # cm per Mpc
_RHO_CRIT0         = 2.775e11       # (Msun/h) (Mpc/h)⁻³  (h-independent)

# Conversion for pressure_uk: gives (Mpc/h)² when P_e in keV/cm³, r in Mpc/h
# conv(h) = (σ_T/m_e c²) [cm²/keV] × (Mpc_cm/h) [cm/(Mpc/h)]
#         = σ_T/(m_e c²) × Mpc_cm / h   [units: cm³/(keV × Mpc/h)]
# Then:  conv × P_e [keV/cm³] × r²dr [(Mpc/h)³] = (Mpc/h)²
_SIGMA_T_OVER_ME_C2 = _SIGMA_T_CM2 / _ME_C2_KEV   # cm²/keV
_KB_KEV             = 8.617333e-8                   # keV/K (Boltzmann constant)


# ---------------------------------------------------------------------------
# Helper: M₂₀₀ (any Δ) → M₅₀₀c, R₅₀₀c  via NFW bisection
# ---------------------------------------------------------------------------

def _nfw_g(x: np.ndarray) -> np.ndarray:
    """NFW shape function g(x) = ln(1+x) - x/(1+x)."""
    return np.log1p(x) - x / (1.0 + x)


def _gnfw_f_params(
    x: np.ndarray,
    alpha_in: float,
    alpha_tr: float,
    alpha_out: float,
) -> np.ndarray:
    """gNFW shape function with explicit (possibly per-halo) α parameters.

    f(x|α) = x^{-α_in} (1 + x^{α_tr})^{(α_in - α_out)/α_tr}

    This version accepts scalar α values (unlike the instance method which uses
    self._alpha_* fixed at construction).  Used by :class:`PressureProfileDPM`
    which has a mass-dependent outer slope (DPM Eq. 5).

    Parameters
    ----------
    x : dimensionless radius r / R_s
    alpha_in, alpha_tr, alpha_out : gNFW shape exponents
    """
    x = np.asarray(x, dtype=float)
    x_safe = np.where(x > 1e-8, x, 1e-8)
    return (
        x_safe ** (-alpha_in)
        * (1.0 + x_safe ** alpha_tr)
        ** ((alpha_in - alpha_out) / alpha_tr)
    )


def m200_to_m500c(
    m200_arr: np.ndarray,
    c200_arr: np.ndarray,
    r200_arr: np.ndarray,
    rho_crit_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert M₂₀₀ to M₅₀₀c and R₅₀₀c using NFW enclosed-mass bisection.

    Given that the NFW halo profile has an enclosed mass:

    .. math::

        M(<r) = M_{200} \\frac{g(c_{200} r/r_{200})}{g(c_{200})}

    we solve for :math:`r_{500c}` such that:

    .. math::

        M(<r_{500c}) = \\frac{4\\pi}{3} \\times 500 \\times \\rho_{\\rm crit}(z) \\times r_{500c}^3

    Parameters
    ----------
    m200_arr : (NM,) [Msun/h]
    c200_arr : (NM,) NFW concentration at Δ=200 (any ρ_ref)
    r200_arr : (NM,) [Mpc/h]
    rho_crit_z : float [(Msun/h)/(Mpc/h)³] comoving critical density at z

    Returns
    -------
    m500c : (NM,) [Msun/h]
    r500c : (NM,) [Mpc/h]
    """
    m200 = np.asarray(m200_arr, dtype=float)
    c200 = np.asarray(c200_arr, dtype=float)
    r200 = np.asarray(r200_arr, dtype=float)
    g200 = _nfw_g(c200)
    NM   = len(m200)

    m500c = np.empty(NM)
    r500c = np.empty(NM)

    for i in range(NM):
        def _f(x, _i=i):
            """x = r/r200;  f(x) = 0 at r = r500c."""
            r = x * r200[_i]
            m_enc = m200[_i] * _nfw_g(c200[_i] * x) / g200[_i]
            m_sph = (4.0 / 3.0) * np.pi * 500.0 * rho_crit_z * r**3
            return m_enc - m_sph

        # Bracket: for typical concentrations, r500c/r200 ∈ [0.25, 0.85]
        x_r = brentq(_f, 0.10, 0.99, xtol=1e-8, maxiter=100)
        r500c[i] = x_r * r200[i]
        m500c[i] = (4.0 / 3.0) * np.pi * 500.0 * rho_crit_z * r500c[i]**3

    return m500c, r500c


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature helper for 3D profile Fourier transforms
# ---------------------------------------------------------------------------

def _profile_uk_gl(
    k_arr: np.ndarray,
    r_max_arr: np.ndarray,
    integrand_fn,
    n_gl: int = 200,
) -> np.ndarray:
    """Gauss-Legendre quadrature for :math:`\\int_0^{r_{\\max}} f(r)\\,j_0(kr)\\,dr`.

    Computes:

    .. math::

        I(k, M) = 4\\pi \\int_0^{r_{\\max}(M)} f(r, M) \\frac{\\sin(kr)}{kr} r^2 \\mathrm{d}r

    for all (k, M) simultaneously using vectorised GL quadrature.

    Parameters
    ----------
    k_arr : (Nk,)
    r_max_arr : (NM,) integration upper limit [same length unit as r]
    integrand_fn : callable(r_nodes) → (NM, n_gl)
        Evaluates the profile f(r, M) on the quadrature grid.
        ``r_nodes`` has shape (NM, n_gl) in the same length units as r_max_arr.
    n_gl : number of GL nodes (default 200)

    Returns
    -------
    uk : (Nk, NM) = 4π ∫ f(r) j₀(kr) r² dr
    """
    k      = np.asarray(k_arr,     dtype=float)
    r_max  = np.asarray(r_max_arr, dtype=float)
    NM     = len(r_max)
    Nk     = len(k)

    x_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)

    # Map GL nodes from [-1,1] to [0, r_max[i]]
    r_nodes = np.outer(r_max / 2.0, x_gl + 1.0)   # (NM, n_gl) [same units as r_max]
    w_eff   = np.outer(r_max / 2.0, w_gl)           # (NM, n_gl)

    # Profile values at quadrature nodes: shape (NM, n_gl)
    f_vals = integrand_fn(r_nodes)   # (NM, n_gl)

    # j₀(K) = sin(K)/K,  K = k × r
    K  = k[:, None, None] * r_nodes[None, :, :]   # (Nk, NM, n_gl)
    j0 = np.where(K < 1e-6, 1.0 - K**2 / 6.0, np.sin(K) / K)   # (Nk, NM, n_gl)

    # Weighted sum over GL nodes: (Nk, NM)
    result = np.sum(
        w_eff[None, :, :] * f_vals[None, :, :] * r_nodes[None, :, :]**2 * j0,
        axis=2,
    )
    return 4.0 * np.pi * result   # (Nk, NM)


# ---------------------------------------------------------------------------
# Arnaud+2010 electron pressure profile  (tSZ)
# ---------------------------------------------------------------------------

class PressureProfileA10:
    """Arnaud+2010 generalized NFW electron pressure profile for tSZ.

    Reference: Arnaud, Pratt, Piffaretti et al. 2010, A&A 517, A92
    (arXiv:0910.1234), Eq. 11 and Table 1.

    The "universal pressure profile" is:

    .. math::

        P_e(r|M_{500c}, z) = 1.65 \\times 10^{-3}\\,h_{70}^2\\,E(z)^{8/3}
            \\left[\\frac{M_{500c}}{3 \\times 10^{14}\\,h_{70}^{-1}\\,M_\\odot}
            \\right]^{2/3 + \\alpha_p}
            p(r/R_{500c}) \\quad [\\text{keV cm}^{-3}]

    with shape function:

    .. math::

        p(x) = \\frac{P_0}{(c_{500}\\,x)^\\gamma
            \\left[1 + (c_{500}\\,x)^\\alpha\\right]^{(\\beta-\\gamma)/\\alpha}}

    Universal parameters from Table 1 of arXiv:0910.1234:
    P₀=8.403, c₅₀₀=1.177, γ=0.3081, α=1.0510, β=5.4905, α_p=0.12.

    Parameters
    ----------
    r_max_over_r500c : float
        Integration truncation radius as a multiple of R₅₀₀c (default 6).
    n_gl : int
        Gauss-Legendre quadrature nodes (default 200).
    """

    # Universal parameters — Arnaud+2010, Table 1
    _P0      = 8.403
    _c500    = 1.177
    _gamma   = 0.3081
    _alpha   = 1.0510
    _beta    = 5.4905
    _alpha_p = 0.12

    def __init__(self, r_max_over_r500c: float = 6.0, n_gl: int = 200):
        self._r_max_factor = float(r_max_over_r500c)
        self._n_gl = int(n_gl)

    def _p3d(
        self,
        r_over_r500: np.ndarray,
        m500c: float,
        z: float,
        h: float,
        omega_m: float,
    ) -> np.ndarray:
        """P_e(r/R₅₀₀c | M₅₀₀c, z) in keV cm⁻³ (Arnaud+2010 Eq. 11).

        Parameters
        ----------
        r_over_r500 : dimensionless radii x = r/R₅₀₀c
        m500c : M₅₀₀c [Msun/h]
        z, h, omega_m : redshift, Hubble parameter, matter fraction
        """
        h70  = h / 0.7
        ez   = np.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m))
        x    = r_over_r500
        pnorm = (1.65e-3 * h70**2 * ez**(8.0 / 3.0)
                 * (m500c / (3.0e14 * h70))**(2.0 / 3.0 + self._alpha_p))
        shape = (
            self._P0
            / ((self._c500 * x)**self._gamma
               * (1.0 + (self._c500 * x)**self._alpha)
               ** ((self._beta - self._gamma) / self._alpha))
        )
        return pnorm * shape

    def pressure_uk(
        self,
        k_arr: np.ndarray,
        m200_arr: np.ndarray,
        r200_arr: np.ndarray,
        c200_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
    ) -> np.ndarray:
        """Pressure-profile Fourier transform ỹ(k|M) in (Mpc/h)².

        Defined as:

        .. math::

            \\tilde{y}(k|M,z) = \\frac{\\sigma_T}{m_e c^2}
                \\frac{\\mathrm{Mpc\\_cm}}{h}
                \\times 4\\pi \\int_0^{r_{\\max}} P_e(r|M,z)\\,
                \\frac{\\sin(kr)}{kr}\\,r^2\\,\\mathrm{d}r

        with ``r`` in Mpc/h and ``P_e`` in keV cm⁻³.  The prefactor
        (σ_T/m_e c²)×(Mpc_cm/h) has units cm³/(keV·Mpc/h) so that

        .. math::

            [\\tilde{y}] = \\frac{\\mathrm{cm}^3}{\\mathrm{keV}\\cdot(\\mathrm{Mpc}/h)}
                \\times \\frac{\\mathrm{keV}}{\\mathrm{cm}^3}
                \\times (\\mathrm{Mpc}/h)^3 = (\\mathrm{Mpc}/h)^2

        The 3D galaxy×y cross-power P_{gy}(k) then has units (Mpc/h)², and
        the projected Σ_y(r_p) = (1/π) ∫ P_{gy}(k) J₀(k r_p) k dk is
        dimensionless (Compton-y parameter).

        Parameters
        ----------
        k_arr : (Nk,) [h/Mpc]
        m200_arr : (NM,) [Msun/h]
        r200_arr : (NM,) [Mpc/h]
        c200_arr : (NM,) concentration at the overdensity stored in the static cache
        z : redshift
        theta_cosmo : dict with keys 'h', 'Omega_m'

        Returns
        -------
        uk : (Nk, NM) [(Mpc/h)²]
        """
        h       = float(theta_cosmo["h"])
        omega_m = float(theta_cosmo["Omega_m"])

        m200 = np.asarray(m200_arr, dtype=float)
        r200 = np.asarray(r200_arr, dtype=float)
        c200 = np.asarray(c200_arr, dtype=float)
        k    = np.asarray(k_arr,    dtype=float)
        NM   = len(m200)

        # Comoving critical density at z — required for M₂₀₀→M₅₀₀c conversion
        ez2          = omega_m * (1.0 + z)**3 + (1.0 - omega_m)
        rho_crit_z   = _RHO_CRIT0 * ez2 / (1.0 + z)**3

        # M₂₀₀ → M₅₀₀c, R₅₀₀c (NFW bisection, ~0.02 s for NM=200)
        m500c, r500c = m200_to_m500c(m200, c200, r200, rho_crit_z)

        # For each halo, build a closure capturing its M₅₀₀c and R₅₀₀c
        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            """P_e(r, M) for all halos on the quadrature grid.

            Args:
                r_nodes : (NM, n_gl) [Mpc/h]
            Returns:
                P_e : (NM, n_gl) [keV/cm³]
            """
            out = np.empty_like(r_nodes)
            for i in range(NM):
                out[i] = self._p3d(
                    r_nodes[i] / r500c[i],
                    m500c[i], z, h, omega_m,
                )
            return out

        r_max = self._r_max_factor * r500c   # (NM,) [Mpc/h]
        raw   = _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)   # (Nk, NM) [keV/cm³ × (Mpc/h)³]

        # Unit conversion → (Mpc/h)²:
        # conv = (σ_T/m_e c²) [cm²/keV] × (Mpc_cm/h) [cm/(Mpc/h)]
        conv = _SIGMA_T_OVER_ME_C2 * (_MPC_CM / h)
        return conv * raw   # (Nk, NM) [(Mpc/h)²]


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
        c_arr   = self._concentration(m200, z, theta_cosmo)

        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            """n_e(r, M) for all halos on the quadrature grid."""
            out = np.empty_like(r_nodes)
            for i in range(NM):
                out[i] = self.density_3d(r_nodes[i], m200[i], r200[i], z, omega_m,
                                         c200c=c_arr[i])
            return out

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
        c_arr   = self._concentration(m200, z, theta_cosmo)

        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            out = np.empty_like(r_nodes)
            for i in range(NM):
                ne = self.density_3d(r_nodes[i], m200[i], r200[i], z, omega_m,
                                     c200c=c_arr[i])
                out[i] = ne**2 * self._scatter_boost
            return out

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
        c_arr   = self._concentration(m200, z, theta_cosmo)

        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            out = np.empty_like(r_nodes)
            for i in range(NM):
                ne  = self.density_3d(r_nodes[i], m200[i], r200[i], z, omega_m,
                                      c200c=c_arr[i])
                P   = pressure_profile._pressure_3d(r_nodes[i], m200[i], r200[i],
                                                    z, omega_m)
                T   = temperature_from_profiles(P, ne)                    # [keV]
                Z   = metallicity_profile.metallicity_3d(r_nodes[i], r200[i])  # [Z_sun]
                lam = cooling_fn(T, Z)
                out[i] = ne**2 * self._scatter_boost * lam
            return out

        r_max = self._r_max_factor * r200
        return _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)


# ---------------------------------------------------------------------------
# Module-level helpers (X-ray temperature and cooling function)
# ---------------------------------------------------------------------------

def temperature_from_profiles(
    pressure: np.ndarray,
    n_electron: np.ndarray,
) -> np.ndarray:
    """Gas temperature T = P / (n_e k_B) [keV].

    Used internally by :meth:`GasDensityDPM.emissivity_full_uk` to build the
    temperature map from DPM pressure and density profiles (Section 3.1.1 of
    arXiv:2505.14782).

    Parameters
    ----------
    pressure   : P_e [keV cm⁻³] — from :class:`PressureProfileDPM`
    n_electron : n_e [cm⁻³]     — from :class:`GasDensityDPM`

    Returns
    -------
    T : [keV]
    """
    ne_safe = np.where(np.asarray(n_electron) > 1e-40, n_electron, 1e-40)
    return np.asarray(pressure) / ne_safe


def temperature_from_dpm(
    pressure_profile,
    density_profile,
    r: np.ndarray,
    m200: float,
    r200: float,
    z: float,
    theta_cosmo: dict,
) -> np.ndarray:
    """Gas temperature T(r, M, z) [keV] from DPM pressure and density profiles.

    Uses the ideal gas law T = P_e / (n_e k_B).  Convenience wrapper that
    evaluates both DPM profiles at the same radii.

    Parameters
    ----------
    pressure_profile  : PressureProfileDPM instance
    density_profile   : GasDensityDPM instance
    r     : radii [Mpc/h]
    m200  : M₂₀₀ [Msun/h]
    r200  : R₂₀₀ [Mpc/h]
    z     : redshift
    theta_cosmo : dict with key 'Omega_m'

    Returns
    -------
    T : (Nr,) [keV]
    """
    omega_m = float(theta_cosmo["Omega_m"])
    P  = pressure_profile._pressure_3d(r, m200, r200, z, omega_m)
    ne = density_profile.density_3d(r, m200, r200, z, omega_m)
    return temperature_from_profiles(P, ne)


def xray_cooling_function(
    T_keV: np.ndarray,
    Z_solar: np.ndarray,
    alpha_T: float = 0.5,
    alpha_Z: float = 1.0,
    Lambda_0: float = 3e-23,
) -> np.ndarray:
    """Simplified power-law cooling function Λ(T, Z) [erg cm³ s⁻¹].

    .. deprecated::
        Use :class:`ApecCoolingTable` instead, which evaluates the full
        APEC plasma code tables (soxs) over the specified energy band.

    Models the 0.5–2 keV soft X-ray volume emissivity coefficient:

    .. math::

        \\Lambda(T, Z) = \\Lambda_0
            \\left(\\frac{T}{1\\,\\text{keV}}\\right)^{\\alpha_T}
            \\left(\\frac{Z}{0.3\\,Z_\\odot}\\right)^{\\alpha_Z}
    """
    import warnings
    warnings.warn(
        "xray_cooling_function is deprecated — use ApecCoolingTable instead.",
        DeprecationWarning, stacklevel=2,
    )
    T = np.asarray(T_keV,   dtype=float)
    Z = np.asarray(Z_solar, dtype=float)
    T_safe = np.where(T > 1e-6, T, 1e-6)
    Z_safe = np.where(Z > 1e-6, Z, 1e-6)
    return Lambda_0 * (T_safe / 1.0) ** alpha_T * (Z_safe / 0.3) ** alpha_Z


class ApecCoolingTable:
    """Band-integrated APEC cooling function Λ(T, Z) precomputed as a 2D table.

    Uses :class:`soxs.ApecGenerator` to compute the X-ray emission spectrum from
    the AtomDB APEC CIE plasma model for a grid of temperatures and metallicities,
    integrates each spectrum over the requested energy band, and stores the result
    as a 2D log-log interpolator for fast evaluation.

    The output follows the APEC normalization convention:

    .. math::

        \\varepsilon(r) = n_e(r)\\,n_H(r)\\,\\Lambda(T(r), Z(r))

    with :math:`n_H \\approx 0.83\\,n_e` (fully ionized solar-abundance plasma).
    The stored table gives :math:`\\Lambda_{n_e^2}(T, Z) = 0.83\\,\\Lambda_{\\rm APEC}`,
    so that :math:`\\varepsilon = n_e^2\\,\\Lambda_{n_e^2}`.

    Parameters
    ----------
    emin, emax : float
        Energy band edges [keV].  Default: 0.5–2.0 (soft X-ray).
    n_T : int
        Number of temperature grid points (log-spaced).  Default: 60.
    T_min, T_max : float
        Temperature range [keV].  Default: 0.08–20.
    n_Z : int
        Number of metallicity grid points (log-spaced).  Default: 15.
    Z_min, Z_max : float
        Metallicity range [Z_sun].  Default: 0.05–3.0.
    apec_vers : str
        APEC version string (default from soxs config, currently "3.1.3").
    nbins : int
        Number of spectral bins used for the band integration.  Default: 1000.

    Notes
    -----
    Requires ``soxs`` (``pip install pyxsim soxs``) and the APEC spectral tables,
    which are downloaded once via ``soxs.download_spectrum_tables("apec")``.

    Initialisation takes a few seconds (60×15 = 900 APEC evaluations).
    Cache the instance across calls.
    """

    _NH_RATIO = 0.83   # n_H / n_e for fully ionized solar-abundance plasma

    def __init__(
        self,
        emin: float = 0.5,
        emax: float = 2.0,
        n_T: int = 60,
        T_min: float = 0.08,
        T_max: float = 20.0,
        n_Z: int = 15,
        Z_min: float = 0.05,
        Z_max: float = 3.0,
        apec_vers: str = None,
        nbins: int = 1000,
    ):
        try:
            import soxs
            from scipy.interpolate import RegularGridInterpolator
        except ImportError as e:
            raise ImportError(
                "soxs not installed — run: pip install pyxsim soxs"
            ) from e

        self._emin = emin
        self._emax = emax

        T_grid = np.logspace(np.log10(T_min), np.log10(T_max), n_T)   # [keV]
        Z_grid = np.logspace(np.log10(Z_min), np.log10(Z_max), n_Z)   # [Z_sun]

        kwargs = dict(broadening=False)
        if apec_vers is not None:
            kwargs["apec_vers"] = apec_vers
        apec = soxs.ApecGenerator(emin, emax, nbins, **kwargs)

        # Precompute Λ_{n_e²}(T_i, Z_j)  [erg cm³ s⁻¹]
        # APEC convention: norm = 1e-14 × EM / (4π D_A²)
        # For norm=1, D_A=1 cm, z=0: EM = 4π × 1e14 cm⁻⁵
        # flux F [erg/s/cm²] = Λ_APEC × EM / (4π D_A²) = Λ_APEC × 1e14
        # → Λ_APEC = F.value / 1e14   [erg cm³/s w.r.t. n_e n_H]
        # → Λ_{n_e²} = 0.83 × Λ_APEC
        table = np.empty((n_T, n_Z))
        for i, kT in enumerate(T_grid):
            for j, Z in enumerate(Z_grid):
                spec = apec.get_spectrum(kT, Z, redshift=0.0, norm=1.0)
                table[i, j] = float(spec.total_energy_flux.value) / 1e14 * self._NH_RATIO

        self._T_grid = T_grid
        self._Z_grid = Z_grid
        self._table  = table
        self._interp = RegularGridInterpolator(
            (np.log10(T_grid), np.log10(Z_grid)),
            np.log10(np.maximum(table, 1e-60)),
            method="linear",
            bounds_error=False,
            fill_value=None,   # linear extrapolation beyond grid edges
        )

    def __call__(self, T_keV: np.ndarray, Z_solar: np.ndarray) -> np.ndarray:
        """Λ_{n_e²}(T, Z) [erg cm³ s⁻¹] by 2D log-log interpolation.

        Parameters
        ----------
        T_keV   : temperature [keV]
        Z_solar : metallicity [Z_sun]

        Returns
        -------
        Lambda : same shape as T_keV  [erg cm³ s⁻¹]
        """
        T = np.asarray(T_keV,   dtype=float)
        Z = np.asarray(Z_solar, dtype=float)
        pts = np.column_stack([
            np.log10(np.maximum(T.ravel(), 1e-6)),
            np.log10(np.maximum(Z.ravel(), 1e-6)),
        ])
        return 10.0 ** self._interp(pts).reshape(T.shape)


# ---------------------------------------------------------------------------
# DPM pressure profile  (tSZ)
# ---------------------------------------------------------------------------

class PressureProfileDPM:
    """DPM electron pressure profile for tSZ (Oppenheimer+2025, arXiv:2505.14782).

    Reference: Table 1 of arXiv:2505.14782 — 3 calibrated models for
    the generalized NFW pressure profile.

    The profile uses the same gNFW shape as :class:`GasDensityDPM` (Eq. 1),
    with the addition of a *mass-dependent outer slope* (Eq. 5):

    .. math::

        \\alpha_{\\rm out}(M) = \\alpha_{\\rm out,12}
            + \\alpha_{\\rm out,var} \\log_{10}(M_{200} / 10^{12}\\,M_\\odot/h)

    The pressure profile is (Eq. 2):

    .. math::

        P(r, M, z) = P_0 \\, f(r/R_s \\mid \\alpha(M)) \\, E(z)^{\\gamma^P}
            \\, M_{12}^{\\beta^P}

    normalised so that :math:`P(0.3 R_{200}, 10^{12}\\,M_\\odot/h, z=0) = P_{0.3}`.

    The ``pressure_uk`` method uses the same unit convention as
    :class:`PressureProfileA10` and outputs in (Mpc/h)².

    Parameters from Table 1 (DPM paper arXiv:2505.14782), converted to keV cm⁻³:

    +---------+----------+----------+----------+
    | Param   | Model 1  | Model 2  | Model 3  |
    +=========+==========+==========+==========+
    | P_0.3   | 4.09e-4  | 1.15e-4  | 7.10e-5  |
    +---------+----------+----------+----------+
    | α_in^P  | 0.3      | 0.3      | −0.6     |
    +---------+----------+----------+----------+
    | α_tr^P  | 1.3      | 1.3      | 0.2      |
    +---------+----------+----------+----------+
    | α_out^P | 4.1      | 4.1      | 2.0      |
    +---------+----------+----------+----------+
    | β^P     | 2/3      | 0.85     | 0.92     |
    +---------+----------+----------+----------+
    | γ^P     | 8/3      | 8/3      | 8/3      |
    +---------+----------+----------+----------+

    .. note::

        The paper (arXiv:2505.14782 Table 1) lists P_0.3 as 409, 115, 71 in
        meV cm⁻³.  The values stored here have been converted to keV cm⁻³
        (factor 10⁻⁶) so that ``pressure_uk`` and ``_pressure_3d`` return
        physically correct units.  Sanity check: T = P_0.3 / ne_0.3 gives
        0.70, 2.36, 1.46 keV for models 1–3 at M=10¹² M☉/h, z=0 — consistent
        with observed group/cluster temperatures at those masses.

    Parameters
    ----------
    model : int (1, 2, or 3), default 2
    r_max_over_r200 : float (default 3.0)
    n_gl : int (default 200)
    """

    _C_DPM = 2.772  # same scale-radius convention as GasDensityDPM

    # Table 1 of arXiv:2505.14782 — P_03 converted from meV cm⁻³ → keV cm⁻³ (×1e-6)
    _PARAMS = {
        1: dict(P_03=409.0e-6,  alpha_in=0.3,  alpha_tr=1.3, alpha_out=4.1, alpha_out_var=0.0, beta=2.0/3.0, gamma=8.0/3.0),
        2: dict(P_03=115.0e-6,  alpha_in=0.3,  alpha_tr=1.3, alpha_out=4.1, alpha_out_var=0.0, beta=0.85,    gamma=8.0/3.0),
        3: dict(P_03=71.0e-6,   alpha_in=-0.6, alpha_tr=0.2, alpha_out=2.0, alpha_out_var=0.0, beta=0.92,    gamma=8.0/3.0),
    }

    def __init__(self, model: int = 2, r_max_over_r200: float = 3.0, n_gl: int = 200):
        if model not in self._PARAMS:
            raise ValueError(f"model must be 1, 2, or 3; got {model}")
        self._model = model
        self._r_max_factor = float(r_max_over_r200)
        self._n_gl = int(n_gl)
        p = self._PARAMS[model]
        self._P_03         = p["P_03"]
        self._alpha_in     = p["alpha_in"]
        self._alpha_tr     = p["alpha_tr"]
        self._alpha_out_12 = p["alpha_out"]      # at M = 10^12 M_sun/h
        self._alpha_out_var = p["alpha_out_var"] # mass-dependent variation (Eq. 5)
        self._beta         = p["beta"]
        self._gamma        = p["gamma"]
        # Normalisation constant: P0 = P_03 / f(0.3 * c_DPM | alpha at M_12=1)
        x_ref = 0.3 * self._C_DPM
        f_ref = _gnfw_f_params(x_ref, self._alpha_in, self._alpha_tr, self._alpha_out_12)
        self._P0 = self._P_03 / float(f_ref)   # units of P_03

    def _pressure_3d(
        self,
        r: np.ndarray,
        m200: float,
        r200: float,
        z: float,
        omega_m: float,
    ) -> np.ndarray:
        """P(r | M₂₀₀, z) in the same units as P_0.3 (keV cm⁻³).

        DPM Eq. 2 with mass-dependent outer slope (Eq. 5).

        Parameters
        ----------
        r      : radii [Mpc/h]
        m200   : M₂₀₀ [Msun/h]
        r200   : R₂₀₀ [Mpc/h]
        z      : redshift
        omega_m: matter fraction Ω_m
        """
        r_s  = r200 / self._C_DPM
        x    = np.asarray(r, dtype=float) / r_s
        M12  = m200 / 1.0e12                          # in h-units
        ez   = np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
        # Mass-dependent outer slope (Eq. 5)
        alpha_out_eff = self._alpha_out_12 + self._alpha_out_var * np.log10(np.maximum(M12, 1e-10))
        f = _gnfw_f_params(x, self._alpha_in, self._alpha_tr, alpha_out_eff)
        return self._P0 * f * ez ** self._gamma * M12 ** self._beta

    def pressure_uk(
        self,
        k_arr: np.ndarray,
        m200_arr: np.ndarray,
        r200_arr: np.ndarray,
        z: float,
        theta_cosmo: dict,
    ) -> np.ndarray:
        """DPM pressure-profile Fourier transform ỹ(k|M) in (Mpc/h)².

        Same interface and unit convention as
        :meth:`PressureProfileA10.pressure_uk`.  The tSZ Compton-y prefactor
        σ_T/(m_e c²) × (Mpc_cm/h) is applied assuming P_0.3 is in keV cm⁻³.

        Parameters
        ----------
        k_arr    : (Nk,) [h/Mpc]
        m200_arr : (NM,) [Msun/h]
        r200_arr : (NM,) [Mpc/h]
        z        : redshift
        theta_cosmo : dict with keys 'h', 'Omega_m'

        Returns
        -------
        uk : (Nk, NM) [(Mpc/h)²]
        """
        h       = float(theta_cosmo["h"])
        omega_m = float(theta_cosmo["Omega_m"])
        m200    = np.asarray(m200_arr, dtype=float)
        r200    = np.asarray(r200_arr, dtype=float)
        k       = np.asarray(k_arr,    dtype=float)
        NM      = len(m200)

        def _integrand(r_nodes: np.ndarray) -> np.ndarray:
            out = np.empty_like(r_nodes)
            for i in range(NM):
                out[i] = self._pressure_3d(r_nodes[i], m200[i], r200[i], z, omega_m)
            return out

        r_max = self._r_max_factor * r200
        raw   = _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)  # (Nk, NM)

        conv = _SIGMA_T_OVER_ME_C2 * (_MPC_CM / h)   # cm³/(keV·Mpc/h)
        return conv * raw   # (Nk, NM) [(Mpc/h)²]


# ---------------------------------------------------------------------------
# DPM metallicity profile
# ---------------------------------------------------------------------------

class MetallicityProfileDPM:
    """DPM gas metallicity profile (Oppenheimer+2025, arXiv:2505.14782, Eq. 4).

    All three DPM models share the same metallicity profile (Table 1):

    .. math::

        Z(r, M, z) = Z_0 \\, f(r/R_s \\mid \\alpha^Z)

    with :math:`\\alpha_{\\rm in}^Z = 0`, :math:`\\alpha_{\\rm tr}^Z = 0.5`,
    :math:`\\alpha_{\\rm out}^Z = 0.7`, :math:`\\beta^Z = 0`, :math:`\\gamma^Z = 0`
    (no mass or redshift dependence).  The normalisation is
    :math:`Z(0.3 R_{200}) = 0.3\\,Z_\\odot`.

    The same :data:`_C_DPM` = 2.772 scale radius convention is used.

    This profile is used by :meth:`GasDensityDPM.emissivity_full_uk` to
    evaluate the metallicity-dependent X-ray cooling function Λ(T, Z).
    """

    _C_DPM     = 2.772
    _Z_03      = 0.3   # [Z_sun] at r=0.3 R_200 (all models, no M or z dependence)
    _ALPHA_IN  = 0.0
    _ALPHA_TR  = 0.5
    _ALPHA_OUT = 0.7

    def __init__(self):
        x_ref = 0.3 * self._C_DPM
        f_ref = _gnfw_f_params(x_ref, self._ALPHA_IN, self._ALPHA_TR, self._ALPHA_OUT)
        self._Z0 = self._Z_03 / float(f_ref)   # [Z_sun]

    def metallicity_3d(self, r: np.ndarray, r200: float) -> np.ndarray:
        """Gas metallicity Z(r) [Z_sun].

        No mass or redshift dependence (β^Z = γ^Z = 0).

        Parameters
        ----------
        r    : radii [Mpc/h]
        r200 : R₂₀₀ [Mpc/h]
        """
        r_s = r200 / self._C_DPM
        x   = np.asarray(r, dtype=float) / r_s
        return self._Z0 * _gnfw_f_params(x, self._ALPHA_IN, self._ALPHA_TR, self._ALPHA_OUT)
