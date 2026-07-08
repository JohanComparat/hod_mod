"""JAX-native concentration–mass relations.

Implements analytic c(M, z) models that are differentiable through the
mass array.  All functions work in h-units (masses in M_sun/h).

Available models
----------------

+------------------+----------+------+---------+-------------------------------------+
| function         | mdef     | cosm.| needs σ | Reference                           |
+==================+==========+======+=========+=====================================+
| ``c_duffy08``    | any      | WMAP5|  no     | Duffy et al. 2008, MNRAS 390 L64    |
| ``c_dutton14``   | 200c,vir | P13  |  no     | Dutton & Macciò 2014, MNRAS 441 3359|
| ``c_klypin16``   | 200c,vir | P13  |  no     | Klypin et al. 2016, MNRAS 457 4340  |
| ``c_bhattacharya13`` | any  | WMAP7|  yes    | Bhattacharya+2013, ApJ 766 32       |
| ``c_diemer15``   | 200c     | any  |  yes    | Diemer & Kravtsov 2015, ApJ 799 108 |
+------------------+----------+------+---------+-------------------------------------+

Notes
-----
- All power-law models (Duffy, Dutton, Klypin) are `@jax.jit`-compiled and
  fully differentiable via JAX auto-diff.
- Models that require the RMS density fluctuation σ(M, z) (Bhattacharya, Diemer)
  accept a pre-computed ``sigma`` array so they remain JAX-traceable.
- Diemer+2019 (``diemer19`` in colossus) requires a 3-D lookup table and is not
  implemented here; use ``HaloProfile`` (which wraps colossus) for that model.

References
----------
Duffy et al. 2008, MNRAS 390 L64 (arXiv:0804.2486)
Dutton & Macciò 2014, MNRAS 441 3359 (arXiv:1402.7073)
Bhattacharya et al. 2013, ApJ 766 32 (arXiv:1112.5020)
Klypin et al. 2016, MNRAS 457 4340 (arXiv:1412.0028)
Diemer & Kravtsov 2015, ApJ 799 108 (arXiv:1407.4605)
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .power_spectrum import rho_critical_0, eisenstein_hu_pk
from .halo_mass_function import _growth_factor_flat_jax

_RHO_CRIT0 = rho_critical_0()  # (Msun/h)/(Mpc/h)³


# ---------------------------------------------------------------------------
# Duffy et al. 2008 — power law c(M, z)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1, 2))
def c_duffy08(
    m_h: jnp.ndarray,
    z: float,
    mdef: str = "200m",
) -> jnp.ndarray:
    r"""Concentration–mass relation of Duffy et al. 2008 (WMAP5).

    .. math::

        c(M, z) = A \left(\frac{M}{2 \times 10^{12}\,h^{-1}M_\odot}\right)^B
                    (1 + z)^C

    Parameters for each mass definition (Table 1 of Duffy+2008):

    +---------+--------+--------+--------+
    | mdef    |   A    |   B    |   C    |
    +=========+========+========+========+
    | 200c    |  5.71  | −0.084 | −0.47  |
    | vir     |  7.85  | −0.081 | −0.71  |
    | 200m    | 10.14  | −0.081 | −1.01  |
    +---------+--------+--------+--------+

    Parameters
    ----------
    m_h : jnp.ndarray
        Halo mass [M_sun/h].
    z : float
        Redshift (static in JIT).
    mdef : str
        Mass definition: ``'200c'``, ``'vir'``, or ``'200m'`` (static in JIT).

    Returns
    -------
    c : jnp.ndarray
        Dimensionless concentration, same shape as ``m_h``.

    Notes
    -----
    Calibrated on WMAP5.  Valid for
    :math:`10^{11} < M < 10^{15}\ M_\odot/h` and :math:`0 < z < 2`.
    """
    if mdef == "200c":
        A, B, C = 5.71, -0.084, -0.47
    elif mdef == "vir":
        A, B, C = 7.85, -0.081, -0.71
    elif mdef == "200m":
        A, B, C = 10.14, -0.081, -1.01
    else:
        raise ValueError(f"mdef must be '200c', 'vir', or '200m', got {mdef!r}")
    return A * (m_h / 2.0e12) ** B * (1.0 + z) ** C


# ---------------------------------------------------------------------------
# Dutton & Macciò 2014 — power law c(M, z) in log-space (Planck13)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1, 2))
def c_dutton14(
    m_h: jnp.ndarray,
    z: float,
    mdef: str = "200c",
) -> jnp.ndarray:
    r"""Concentration–mass relation of Dutton & Macciò 2014 (Planck13).

    .. math::

        \log_{10} c(M, z) = a(z) + b(z)\,\log_{10}\!\left(\frac{M}{10^{12}\,h^{-1}M_\odot}\right)

    with redshift-dependent coefficients from Table 2 of Dutton+2014:

    For ``mdef = '200c'``:

    .. math::

        a(z) &= 0.520 + (0.905 - 0.520)\,e^{-0.617\,z^{1.21}} \\
        b(z) &= -0.101 + 0.026\,z

    For ``mdef = 'vir'``:

    .. math::

        a(z) &= 0.537 + (1.025 - 0.537)\,e^{-0.718\,z^{1.08}} \\
        b(z) &= -0.097 + 0.024\,z

    Parameters
    ----------
    m_h : jnp.ndarray
        Halo mass [M_sun/h].
    z : float
        Redshift (static in JIT).
    mdef : str
        Mass definition: ``'200c'`` or ``'vir'`` (static in JIT).

    Returns
    -------
    c : jnp.ndarray
        Dimensionless concentration, same shape as ``m_h``.

    Notes
    -----
    Calibrated on Planck13.  Valid for :math:`M > 10^{10}\ M_\odot/h`,
    :math:`0 < z < 5`.
    """
    if mdef == "200c":
        a = 0.520 + (0.905 - 0.520) * jnp.exp(-0.617 * z**1.21)
        b = -0.101 + 0.026 * z
    elif mdef == "vir":
        a = 0.537 + (1.025 - 0.537) * jnp.exp(-0.718 * z**1.08)
        b = -0.097 + 0.024 * z
    else:
        raise ValueError(f"mdef must be '200c' or 'vir', got {mdef!r}")
    return jnp.power(10.0, a + b * jnp.log10(m_h / 1.0e12))


# ---------------------------------------------------------------------------
# Klypin et al. 2016 — mass-based power law (Planck13)
# ---------------------------------------------------------------------------

# Tabulated parameters from Table 2 of Klypin+2016 for 200c, planck13 cosmology.
# Columns: (z, C0, gamma, M0_in_units_of_1e12)
_KLYPIN16_200C_PLANCK13 = np.array([
    [0.00,  7.40, 0.120, 5.5e5],
    [0.35,  6.25, 0.117, 1e5],
    [0.50,  5.65, 0.115, 2e4],
    [1.00,  4.30, 0.110, 900.0],
    [1.44,  3.53, 0.095, 300.0],
    [2.15,  2.70, 0.085, 42.0],
    [2.50,  2.42, 0.080, 17.0],
    [2.90,  2.20, 0.080, 8.5],
    [4.10,  1.92, 0.080, 2.0],
    [5.40,  1.65, 0.080, 0.3],
], dtype=float)

_KLYPIN16_VIR_PLANCK13 = np.array([
    [0.00,  9.75, 0.110, 5e5],
    [0.35,  7.25, 0.107, 2.2e4],
    [0.50,  6.50, 0.105, 1e4],
    [1.00,  4.75, 0.100, 1000.0],
    [1.44,  3.80, 0.095, 210.0],
    [2.15,  3.00, 0.085, 43.0],
    [2.50,  2.65, 0.080, 18.0],
    [2.90,  2.42, 0.080, 9.0],
    [4.10,  2.10, 0.080, 1.9],
    [5.40,  1.86, 0.080, 0.42],
], dtype=float)


def c_klypin16(
    m_h: jnp.ndarray,
    z: float,
    mdef: str = "200c",
) -> jnp.ndarray:
    r"""Concentration–mass relation of Klypin et al. 2016 (Planck13).

    Mass-based fitting function (Eq. 14 of Klypin+2016):

    .. math::

        c(M, z) = C_0(z)\left(\frac{M}{10^{12}\,h^{-1}M_\odot}\right)^{-\gamma(z)}
                  \left[1 + \left(\frac{M}{M_0(z)}\right)^{0.4}\right]

    with redshift-interpolated parameters from Table 2 of Klypin+2016.
    This function implements the Planck13 cosmology fit.

    Parameters
    ----------
    m_h : jnp.ndarray
        Halo mass [M_sun/h].
    z : float
        Redshift (static).  Must be within the tabulated range [0, 5.4].
    mdef : str
        Mass definition: ``'200c'`` or ``'vir'`` (static).

    Returns
    -------
    c : jnp.ndarray
        Dimensionless concentration, same shape as ``m_h``.

    Notes
    -----
    Calibrated on Planck13 (MultiDark Planck simulation).
    Valid for :math:`M > 10^{10}\ M_\odot/h`, :math:`0 \leq z \leq 5.4`.
    Parameters are linearly interpolated between the tabulated redshift bins.
    """
    if mdef == "200c":
        tab = _KLYPIN16_200C_PLANCK13
    elif mdef == "vir":
        tab = _KLYPIN16_VIR_PLANCK13
    else:
        raise ValueError(f"mdef must be '200c' or 'vir', got {mdef!r}")

    z_tab = jnp.asarray(tab[:, 0])
    C0    = jnp.interp(z, z_tab, jnp.asarray(tab[:, 1]))
    gamma = jnp.interp(z, z_tab, jnp.asarray(tab[:, 2]))
    M0    = jnp.interp(z, z_tab, jnp.asarray(tab[:, 3])) * 1.0e12

    return C0 * (m_h / 1.0e12) ** (-gamma) * (1.0 + (m_h / M0) ** 0.4)


# ---------------------------------------------------------------------------
# Bhattacharya et al. 2013 — c–ν relation (WMAP7)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(3, 4))
def c_bhattacharya13(
    m_h: jnp.ndarray,
    sigma: jnp.ndarray,
    omega_m: float,
    z: float,
    mdef: str = "200c",
) -> jnp.ndarray:
    r"""Concentration–mass relation of Bhattacharya et al. 2013 (WMAP7).

    .. math::

        c(M, z) = K\,D(z)^{\alpha}\,\nu(M, z)^{\beta},
        \qquad \nu = \frac{\delta_c}{\sigma(M, z)},
        \quad \delta_c = 1.686

    where :math:`D(z) = D(z)/D(0)` is the linear growth factor (flat ΛCDM)
    and the parameters :math:`(K, \alpha, \beta)` depend on ``mdef``
    (Table 2 of Bhattacharya+2013):

    +---------+------+-------+-------+
    | mdef    |  K   |  α    |  β    |
    +=========+======+=======+=======+
    | 200c    | 5.9  |  0.54 | −0.35 |
    | vir     | 7.7  |  0.90 | −0.29 |
    | 200m    | 9.0  |  1.15 | −0.29 |
    +---------+------+-------+-------+

    Parameters
    ----------
    m_h : jnp.ndarray
        Halo mass [M_sun/h].
    sigma : jnp.ndarray
        RMS linear density fluctuation σ(M, z) at the requested redshift,
        same shape as ``m_h``.  Compute via ``HaloMassFunction.sigma``.
    omega_m : float
        Total matter density parameter Ω_m (static in JIT).
    z : float
        Redshift (static in JIT).
    mdef : str
        Mass definition: ``'200c'``, ``'vir'``, or ``'200m'`` (static in JIT).

    Returns
    -------
    c : jnp.ndarray
        Dimensionless concentration, same shape as ``m_h``.

    Notes
    -----
    Calibrated on WMAP7.  Valid for
    :math:`2 \times 10^{12} < M < 2 \times 10^{15}\ M_\odot/h` and
    :math:`0 < z < 2`.
    """
    if mdef == "200c":
        K, alpha, beta = 5.9, 0.54, -0.35
    elif mdef == "vir":
        K, alpha, beta = 7.7, 0.90, -0.29
    elif mdef == "200m":
        K, alpha, beta = 9.0, 1.15, -0.29
    else:
        raise ValueError(f"mdef must be '200c', 'vir', or '200m', got {mdef!r}")

    D = _growth_factor_flat_jax(z, omega_m)
    nu = 1.686 / sigma
    return K * D**alpha * nu**beta


# ---------------------------------------------------------------------------
# Diemer & Kravtsov 2015 — universal c–ν–n model (any cosmology)
# ---------------------------------------------------------------------------

def _neff_eisenstein_hu(m_h: jnp.ndarray, theta: dict,
                        kappa: float = 0.42) -> jnp.ndarray:
    r"""Effective power-spectrum slope n = d ln P_lin / d ln k at scale k_R(M).

    Following Diemer & Kravtsov 2015 (Eq. 7): the relevant scale is

    .. math::

        k_R = \kappa\,\frac{2\pi}{R(M)}, \quad
        R(M) = \left(\frac{3M}{4\pi\bar{\rho}_m}\right)^{1/3}

    with κ = 0.42 — the Diemer & Joyce 2019 calibration the *median*
    ``c_diemer15`` parameters belong to (κ was previously hard-coded to 1,
    which biases n_eff and hence c high; fixed 2026-07).

    The slope is computed by tabulating P_EH on a dense log-spaced grid and
    differentiating numerically.  Calling ``eisenstein_hu_pk`` with a
    single-element array would always return 1.0 (normalisation artefact), so
    we always evaluate on a 500-point grid and interpolate.

    Parameters
    ----------
    m_h : jnp.ndarray, shape (NM,)
    theta : dict  Cosmological parameters.
    kappa : float  Scale calibration k_R = κ·2π/R (0.42 = DJ19 median).

    Returns
    -------
    n_eff : jnp.ndarray, shape (NM,), effective slope (typically −3 to 0)
    """
    rho_m = _RHO_CRIT0 * theta["Omega_m"]
    m = jnp.asarray(m_h)
    R = (3.0 * m / (4.0 * jnp.pi * rho_m)) ** (1.0 / 3.0)     # Mpc/h
    k_R = kappa * 2.0 * jnp.pi / R                             # h/Mpc

    k_grid = np.logspace(-3, 2, 500)                           # static grid
    log_k_np = np.log(k_grid)                                  # uniform log spacing
    dlk = float(log_k_np[1] - log_k_np[0])   # concrete spacing (static grid → jit-safe)
    pk_grid = eisenstein_hu_pk(jnp.asarray(k_grid), theta)
    log_k = jnp.asarray(log_k_np)
    log_pk = jnp.log(jnp.maximum(pk_grid, 1e-30))
    d_log_pk = jnp.gradient(log_pk, dlk)
    return jnp.interp(jnp.log(k_R), log_k, d_log_pk)


# omega_m (arg 3) is dynamic — it is unused in the body (cosmology enters via
# sigma/n_eff), so keeping it non-static lets a traced Ω_m pass through
# harmlessly (needed for the differentiable gas-density concentration).
@partial(jax.jit, static_argnums=(4, 5))
def c_diemer15(
    m_h: jnp.ndarray,
    sigma: jnp.ndarray,
    n_eff: jnp.ndarray,
    omega_m: float,
    z: float,
    statistic: str = "median",
) -> jnp.ndarray:
    r"""Concentration for the Diemer & Kravtsov 2015 universal c–ν–n model.

    This model predicts :math:`c_{200c}` from the peak height ν and the
    local slope n of the linear power spectrum (Eq. 1 of Diemer+2015 with
    updated parameters from Diemer & Joyce 2019):

    .. math::

        c_{200c}(\nu, n) =
        (\phi_0 + n\,\phi_1)\,\left(\frac{\nu}{\eta_0 + n\,\eta_1}\right)^{-\alpha}
        \left[1 + \left(\frac{\nu}{\eta_0 + n\,\eta_1}\right)^{\beta}\right]

    Updated (Diemer & Joyce 2019) median parameters:
    :math:`\phi_0=6.58,\ \phi_1=1.27,\ \eta_0=7.28,\ \eta_1=1.56,
    \ \alpha=1.08,\ \beta=1.77`.

    Parameters
    ----------
    m_h : jnp.ndarray
        Halo mass [M_sun/h].
    sigma : jnp.ndarray
        RMS fluctuation σ(M, z) at the target redshift, same shape as ``m_h``.
    n_eff : jnp.ndarray
        Effective spectral slope n = d ln P / d ln k at scale k_R(M),
        same shape as ``m_h``.  Compute via ``neff_eisenstein_hu``.
    omega_m : float
        Total matter density Ω_m (static in JIT).
    z : float
        Redshift (static in JIT; unused here, kept for interface consistency).
    statistic : str
        ``'median'`` (default) or ``'mean'``.  Static in JIT.

    Returns
    -------
    c200c : jnp.ndarray
        Concentration parameter :math:`c_{200c}`, same shape as ``m_h``.

    Notes
    -----
    Always returns :math:`c_{200c}`.  This model is cosmology-independent
    in the sense that it works for any input σ(M, z) and n_eff computed
    from the corresponding power spectrum.
    """
    if statistic == "median":
        phi0, phi1 = 6.58, 1.27
        eta0, eta1 = 7.28, 1.56
        alpha, beta = 1.08, 1.77
    elif statistic == "mean":
        phi0, phi1 = 6.66, 1.37
        eta0, eta1 = 5.41, 1.06
        alpha, beta = 1.22, 1.22
    else:
        raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")

    # DK15 Eq. 9 / DJ19 Eq. 30: c = (c_min/2)[x^{-α} + x^{β}], x = ν/ν_min.
    # (Previously implemented as c_min·x^{-α}(1 + x^{β}) — missing the ½ and
    # with a β−α second exponent — which biases c high by ~2×; fixed 2026-07,
    # anchored against COLOSSUS in tests/test_missing_physics.py.)
    nu = 1.686 / sigma
    x = nu / (eta0 + n_eff * eta1)
    c_min = phi0 + n_eff * phi1
    return 0.5 * c_min * (x ** (-alpha) + x ** beta)


# ---------------------------------------------------------------------------
# ConcentrationModel — unified interface
# ---------------------------------------------------------------------------

class ConcentrationModel:
    """Unified c(M, z) interface for all JAX-native concentration models.

    Wraps all five analytic models behind a single ``.concentration()`` method.
    For models requiring σ(M, z) (Bhattacharya+2013, Diemer+2015), an HMF
    object must be supplied at construction time.

    Parameters
    ----------
    model : str
        One of ``'duffy08'``, ``'dutton14'``, ``'klypin16'``,
        ``'bhattacharya13'``, ``'diemer15'``.
    mdef : str
        Mass definition, e.g. ``'200c'``, ``'200m'``, ``'vir'``.
    hmf : HaloMassFunction or None
        Required for ``'bhattacharya13'`` and ``'diemer15'`` (provides σ(M, z)).
    statistic : str
        ``'median'`` or ``'mean'`` (only used by ``'diemer15'``).

    Examples
    --------
    Pure power-law (no HMF needed):

    >>> cm = ConcentrationModel('dutton14', mdef='200c')
    >>> c = cm.concentration(m_h, z=0.5, theta=theta)

    Peak-height model (requires HMF):

    >>> cm = ConcentrationModel('diemer15', mdef='200c', hmf=hmf)
    >>> c = cm.concentration(m_h, z=0.5, theta=theta)
    """

    _SIGMA_MODELS = ("bhattacharya13", "diemer15")

    def __init__(
        self,
        model: str = "dutton14",
        mdef: str = "200c",
        hmf=None,
        statistic: str = "median",
    ):
        if model not in ("duffy08", "dutton14", "klypin16", "bhattacharya13", "diemer15"):
            raise ValueError(f"Unknown model {model!r}")
        if model in self._SIGMA_MODELS and hmf is None:
            raise ValueError(f"model='{model}' requires an HMF object (hmf=)")
        self.model = model
        self.mdef = mdef
        self._hmf = hmf
        self.statistic = statistic

    def concentration(
        self,
        m_h: jnp.ndarray,
        z: float,
        theta: dict,
    ) -> jnp.ndarray:
        """Concentration c(M, z).

        Parameters
        ----------
        m_h : jnp.ndarray
            Halo masses [M_sun/h].
        z : float
            Redshift.
        theta : dict
            Cosmological parameter dict (needs at least ``'Omega_m'``).

        Returns
        -------
        c : jnp.ndarray
            Dimensionless concentration, same shape as ``m_h``.
        """
        if self.model == "duffy08":
            return c_duffy08(m_h, z, self.mdef)
        if self.model == "dutton14":
            return c_dutton14(m_h, z, self.mdef)
        if self.model == "klypin16":
            return c_klypin16(m_h, z, self.mdef)
        omega_m = float(theta["Omega_m"])
        sigma = self._hmf.sigma(m_h, float(z), theta)
        if self.model == "bhattacharya13":
            return c_bhattacharya13(m_h, sigma, omega_m, float(z), self.mdef)
        # diemer15
        n_eff = _neff_eisenstein_hu(m_h, theta)
        return c_diemer15(m_h, sigma, n_eff, omega_m, float(z), self.statistic)

    def _mdef_delta_rho(self, z: float, theta_cosmo: dict) -> tuple:
        """(delta, rho_ref) for this model's SO mass definition [comoving h-units].

        Delegates to :func:`hod_mod.core.halo_profiles.mdef_delta_rho` so a
        ``ConcentrationModel`` is drop-in for ``HaloProfile`` in the halo-model
        radius calculation of :class:`~hod_mod.observables.clustering.
        FullHaloModelPrediction` (both the CAMB and ``eh98_jax`` backends).
        ``theta_cosmo['Omega_m']`` may be a traced JAX value.
        """
        from hod_mod.core.halo_profiles import mdef_delta_rho
        return mdef_delta_rho(self.mdef, z, theta_cosmo)
