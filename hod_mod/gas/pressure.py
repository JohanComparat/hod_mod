"""Electron-pressure profiles (Arnaud+2010, DPM, Battaglia+2012) for the tSZ Compton-y signal."""
import numpy as np
import jax.numpy as jnp
from .conversions import (
    _MPC_CM,
    _RHO_CRIT0,
    _SIGMA_T_OVER_ME_C2,
    _G_MSUN2_MPC4_KEV,
    _gnfw_f_params,
    _profile_uk_gl,
    m200_to_m500c,
)


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
        r_over_r500: jnp.ndarray,
        m500c,
        z: float,
        h: float,
        omega_m: float,
    ) -> jnp.ndarray:
        """P_e(r/R₅₀₀c | M₅₀₀c, z) in keV cm⁻³ (Arnaud+2010 Eq. 11).

        Parameters
        ----------
        r_over_r500 : dimensionless radii x = r/R₅₀₀c; broadcasts against m500c
        m500c : M₅₀₀c [Msun/h], scalar or array broadcastable against x
        z, h, omega_m : redshift, Hubble parameter, matter fraction
        """
        h70  = h / 0.7
        ez   = jnp.sqrt(omega_m * (1.0 + z)**3 + (1.0 - omega_m))
        x    = jnp.asarray(r_over_r500)
        pnorm = (1.65e-3 * h70**2 * ez**(8.0 / 3.0)
                 * (jnp.asarray(m500c) / (3.0e14 * h70))**(2.0 / 3.0 + self._alpha_p))
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
        # Kept traceable (jnp, no float()) so the tSZ cross-power is
        # differentiable w.r.t. cosmology on the eh98 backend; concrete inputs
        # (CAMB) pass through jnp.asarray unchanged.
        h       = theta_cosmo["h"]
        omega_m = theta_cosmo["Omega_m"]

        m200 = jnp.asarray(m200_arr, dtype=float)
        r200 = jnp.asarray(r200_arr, dtype=float)
        c200 = jnp.asarray(c200_arr, dtype=float)
        k    = jnp.asarray(k_arr,    dtype=float)

        # Comoving critical density at z — required for M₂₀₀→M₅₀₀c conversion
        ez2          = omega_m * (1.0 + z)**3 + (1.0 - omega_m)
        rho_crit_z   = _RHO_CRIT0 * ez2 / (1.0 + z)**3

        # M₂₀₀ → M₅₀₀c, R₅₀₀c (NFW bisection, ~0.02 s for NM=200)
        m500c, r500c = m200_to_m500c(m200, c200, r200, rho_crit_z)

        def _integrand(r_nodes) -> jnp.ndarray:
            """P_e(r, M) for all halos on the quadrature grid, broadcast
            over the mass axis (no per-halo loop).

            Args:
                r_nodes : (NM, n_gl) [Mpc/h]
            Returns:
                P_e : (NM, n_gl) [keV/cm³]
            """
            return self._p3d(
                jnp.asarray(r_nodes) / r500c[:, None],
                m500c[:, None], z, h, omega_m,
            )

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
        r: jnp.ndarray,
        m200,
        r200,
        z: float,
        omega_m: float,
    ) -> jnp.ndarray:
        """P(r | M₂₀₀, z) in the same units as P_0.3 (keV cm⁻³).

        DPM Eq. 2 with mass-dependent outer slope (Eq. 5).

        Parameters
        ----------
        r      : radii [Mpc/h]; broadcasts against m200/r200
        m200   : M₂₀₀ [Msun/h], scalar or array broadcastable against r
        r200   : R₂₀₀ [Mpc/h], scalar or array broadcastable against r
        z      : redshift
        omega_m: matter fraction Ω_m
        """
        r_s  = jnp.asarray(r200) / self._C_DPM
        x    = jnp.asarray(r) / r_s
        M12  = jnp.asarray(m200) / 1.0e12             # in h-units
        ez   = jnp.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
        # Mass-dependent outer slope (Eq. 5)
        alpha_out_eff = self._alpha_out_12 + self._alpha_out_var * jnp.log10(jnp.maximum(M12, 1e-10))
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

        def _integrand(r_nodes) -> jnp.ndarray:
            # broadcast over the mass axis (no per-halo loop)
            return self._pressure_3d(
                jnp.asarray(r_nodes), m200[:, None], r200[:, None], z, omega_m)

        r_max = self._r_max_factor * r200
        raw   = _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)  # (Nk, NM)

        conv = _SIGMA_T_OVER_ME_C2 * (_MPC_CM / h)   # cm³/(keV·Mpc/h)
        return conv * raw   # (Nk, NM) [(Mpc/h)²]


# ---------------------------------------------------------------------------
# Battaglia+2012 pressure profile  (tSZ) — shared profile for the GODMAX check
# ---------------------------------------------------------------------------

class PressureProfileBattaglia12:
    """Battaglia+2012 generalized-NFW electron pressure profile for tSZ.

    Reference: Battaglia, Bond, Pfrommer & Sijacki 2012, ApJ 758, 74
    (arXiv:1109.3711), Table 1 "AGN feedback, Δ=200".  This is the analytic
    pressure model used by **GODMAX** (Pandey+2024, arXiv:2401.18072), so it
    serves as the *shared, apples-to-apples* profile for the independent SZ
    cross-check: driving both codes through the identical B12 profile isolates
    hod_mod's projection machinery (HMF/bias integrals, Ogata-Hankel + Limber)
    from the gas-physics model.

    The **thermal** pressure is a generalized NFW in ``x = r / R_{200c}``:

    .. math::

        P_{\\rm th}(x|M_{200c}, z) = P_{200c}\\, P_0
            \\left(\\frac{x}{x_c}\\right)^{\\gamma}
            \\left[1 + \\left(\\frac{x}{x_c}\\right)^{\\alpha}\\right]^{-\\beta}

    with the self-similar (Kaiser) amplitude

    .. math::

        P_{200c} = \\frac{G\\,M_{200c}\\,200\\,\\rho_{\\rm cr}(z)\\,f_b}{2\\,R_{200c}}

    (physical :math:`M_{200c}`, :math:`\\rho_{\\rm cr}(z)`, :math:`R_{200c}`;
    :math:`f_b = \\Omega_b/\\Omega_m`).  Each of :math:`\\{P_0, x_c, \\beta\\}`
    scales with mass and redshift as

    .. math::

        A(M, z) = A_0 \\left(\\frac{M_{200c}}{10^{14}\\,M_\\odot}\\right)^{\\alpha_m}
                  (1+z)^{\\alpha_z}

    while :math:`\\gamma = -0.3` and :math:`\\alpha = 1.0` are fixed.

    The **electron** pressure returned by :meth:`_p3d` is
    :math:`P_e = f_e\\,P_{\\rm th}` with
    :math:`f_e = (2 + 2 X_H)/(3 + 5 X_H) \\approx 0.518` for hydrogen mass
    fraction :math:`X_H = 0.76`, so that the shared :meth:`pressure_uk` applies
    only the Compton-y prefactor :math:`\\sigma_T/(m_e c^2)` — identical to
    :class:`PressureProfileA10`.

    Fiducial parameters — Battaglia+2012 Table 1 (AGN feedback, Δ=200):

    +--------+---------+-----------+-----------+
    | Param  | A0      | α_m       | α_z       |
    +========+=========+===========+===========+
    | P0     | 18.1    | 0.154     | −0.758    |
    +--------+---------+-----------+-----------+
    | xc     | 0.497   | −0.00865  | 0.731     |
    +--------+---------+-----------+-----------+
    | β      | 4.35    | 0.0393    | 0.415     |
    +--------+---------+-----------+-----------+

    Unlike :class:`PressureProfileA10` (which converts M₂₀₀→M₅₀₀c internally),
    B12 is natively an M₂₀₀c profile, so the halo cache must be built with
    ``mdef='200c'`` — then ``m200_arr``/``r200_arr`` are the M₂₀₀c/R₂₀₀c triple.

    Parameters
    ----------
    r_max_over_r200c : float
        Integration truncation radius as a multiple of R₂₀₀c (default 4).
        Must match the GODMAX export configuration for the cross-check.
    n_gl : int
        Gauss-Legendre quadrature nodes (default 200).
    x_h : float
        Hydrogen mass fraction for the thermal→electron pressure factor
        ``f_e = (2 + 2 x_h)/(3 + 5 x_h)`` (default 0.76).
    """

    # Battaglia+2012 Table 1 — AGN feedback, Δ=200 (dict: A0, alpha_m, alpha_z)
    _P0   = {"A0": 18.1,  "am":  0.154,    "az": -0.758}
    _XC   = {"A0": 0.497, "am": -0.00865,  "az":  0.731}
    _BETA = {"A0": 4.35,  "am":  0.0393,   "az":  0.415}
    _gamma = -0.3
    _alpha = 1.0

    def __init__(self, r_max_over_r200c: float = 4.0, n_gl: int = 200, x_h: float = 0.76):
        self._r_max_factor = float(r_max_over_r200c)
        self._n_gl = int(n_gl)
        self._x_h = float(x_h)
        self._f_e = (2.0 + 2.0 * self._x_h) / (3.0 + 5.0 * self._x_h)

    @staticmethod
    def _scale(params: dict, m200c_phys, z) -> jnp.ndarray:
        """A(M,z) = A0·(M/1e14)^α_m·(1+z)^α_z, with M in physical Msun."""
        return (params["A0"]
                * (jnp.asarray(m200c_phys) / 1.0e14) ** params["am"]
                * (1.0 + z) ** params["az"])

    def _p3d(
        self,
        r_over_r200: jnp.ndarray,
        m200c,
        z: float,
        h: float,
        omega_m: float,
        f_b: float,
        r200c: jnp.ndarray,
    ) -> jnp.ndarray:
        """P_e(x = r/R₂₀₀c | M₂₀₀c, z) in keV cm⁻³ (Battaglia+2012, ×f_e).

        Parameters
        ----------
        r_over_r200 : dimensionless x = r/R₂₀₀c; broadcasts against m200c
        m200c : M₂₀₀c [Msun/h], scalar or array broadcastable against x
        z, h, omega_m : redshift, Hubble parameter, matter fraction
        f_b : baryon fraction Ω_b/Ω_m
        r200c : comoving R₂₀₀c [Mpc/h] (same convention as the halo cache)
        """
        # Physical quantities for the Kaiser amplitude P_200c.
        m_phys      = jnp.asarray(m200c) / h                       # Msun
        ez2         = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
        rho_cr_phys = _RHO_CRIT0 * h ** 2 * ez2                    # Msun/Mpc³ (physical, z)
        r200c_phys  = jnp.asarray(r200c) / h / (1.0 + z)          # Mpc (physical)
        p200        = (_G_MSUN2_MPC4_KEV * m_phys * 200.0 * rho_cr_phys * f_b
                       / (2.0 * r200c_phys))                       # keV/cm³

        # Mass/redshift-scaled shape parameters (M in physical Msun).
        p0   = self._scale(self._P0,   m_phys, z)
        xc   = self._scale(self._XC,   m_phys, z)
        beta = self._scale(self._BETA, m_phys, z)

        xr    = jnp.asarray(r_over_r200) / xc
        shape = xr ** self._gamma * (1.0 + xr ** self._alpha) ** (-beta)
        return self._f_e * p200 * p0 * shape

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

        Same interface and unit convention as
        :meth:`PressureProfileA10.pressure_uk` (the ``c200_arr`` argument is
        accepted for signature compatibility but unused — B12 needs no
        concentration).  Requires ``theta_cosmo`` to carry ``'Omega_b'`` for the
        baryon fraction f_b = Ω_b/Ω_m.

        Parameters
        ----------
        k_arr : (Nk,) [h/Mpc]
        m200_arr : (NM,) M₂₀₀c [Msun/h]  (build the cache with mdef='200c')
        r200_arr : (NM,) R₂₀₀c [Mpc/h], comoving
        c200_arr : (NM,) unused
        z : redshift
        theta_cosmo : dict with keys 'h', 'Omega_m', 'Omega_b'

        Returns
        -------
        uk : (Nk, NM) [(Mpc/h)²]
        """
        # Kept traceable (jnp, no float()) so the tSZ cross-power stays
        # differentiable w.r.t. cosmology on the eh98 backend.
        h       = theta_cosmo["h"]
        omega_m = theta_cosmo["Omega_m"]
        omega_b = theta_cosmo["Omega_b"]
        f_b     = omega_b / omega_m

        m200 = jnp.asarray(m200_arr, dtype=float)
        r200 = jnp.asarray(r200_arr, dtype=float)
        k    = jnp.asarray(k_arr,    dtype=float)

        def _integrand(r_nodes) -> jnp.ndarray:
            """P_e(r, M) on the quadrature grid, broadcast over the mass axis."""
            return self._p3d(
                jnp.asarray(r_nodes) / r200[:, None],
                m200[:, None], z, h, omega_m, f_b, r200[:, None],
            )

        r_max = self._r_max_factor * r200          # (NM,) [Mpc/h]
        raw   = _profile_uk_gl(k, r_max, _integrand, n_gl=self._n_gl)

        conv = _SIGMA_T_OVER_ME_C2 * (_MPC_CM / h)   # cm³/(keV·Mpc/h)
        return conv * raw   # (Nk, NM) [(Mpc/h)²]
