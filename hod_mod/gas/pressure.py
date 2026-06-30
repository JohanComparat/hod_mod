"""Electron-pressure profiles (Arnaud+2010, DPM) for the tSZ Compton-y signal."""
import numpy as np
from .conversions import _MPC_CM, _RHO_CRIT0, _SIGMA_T_OVER_ME_C2, _gnfw_f_params, _profile_uk_gl, m200_to_m500c


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
