"""Native-DPM radial band weights for the X-ray joint fit (option *b-radial*).

Re-bases the X-ray band model from the phenomenological ``(lx_norm, lx_slope,
kt_norm, kt_slope)`` scaling relations onto the **native DPM gas parameters**

.. math::

    \\{n_{e,0.3},\\; \\beta_n,\\; P_{0.3},\\; \\beta_P\\},

so that the X-ray bands and the tSZ :math:`\\Sigma_y` (see
:mod:`hod_mod.fitting.sz_transfer`) are driven by **one** DPM gas model:
:math:`P_{0.3}, \\beta_P` set the electron pressure that the SZ integrates *and*
the X-ray temperature :math:`T = P/n_e`; :math:`n_{e,0.3}, \\beta_n` set the
density that the X-ray emission measure integrates *and* the same temperature.
Vary once, both observables move.

Why this is both exact and fast
-------------------------------
The DPM temperature is :math:`T(r,M) = P(r,M)/n_e(r,M)`.  Writing the two gNFW
shapes as :math:`f_P` and :math:`f_n`,

.. math::

    T(r, M) = T_0(M)\\, g(x), \\qquad
    g(x) \\equiv \\frac{f_P(x)/f_n(x)}{f_P(x_{\\rm ref})/f_n(x_{\\rm ref})},

.. math::

    T_0(M) = \\frac{P_{0.3}}{n_{e,0.3}}\\; M_{12}^{\\,\\beta_P-\\beta_n}\\;
             E(z)^{\\gamma_P-\\gamma_n} .

The radial shape :math:`g(x)` is fixed by the DPM slopes — it does **not** depend
on any of the four native parameters.  Only the normalisation :math:`T_0` does.
Hence the band emission integral factorises *exactly*:

.. math::

    L_{X,b}(M) = n_{e,0.3}^2\\, M_{12}^{2\\beta_n}\\, E(z)^{2\\gamma_n}\\,
                 V_{\\rm shape}(M)\\; J_b\\!\\big(T_0(M), Z\\big),

.. math::

    J_b(T_0, Z) \\equiv
    \\frac{\\int f_n(x)^2\\, \\Lambda_b\\!\\big(T_0 g(x), Z\\big)\\, x^2\\,dx}
         {\\int f_n(x)^2\\, x^2\\,dx} ,

a function of only :math:`(T_0, Z)` for a given profile shape — so it is
tabulated once and interpolated at MCMC speed.

This keeps DPM's genuine **radial** temperature profile.  A single
:math:`\\Lambda(T_{\\rm ew})` rescaling (the isothermal shortcut used by the
previous phenomenological band model) is wrong at the ~14% level, because
:math:`\\Lambda` is non-linear in :math:`T` — measured against
``validate_gas_profiles._integrate_profile``.  Consequently this model does
**not** reproduce the old isothermal baseline; that shift is the physics, not a
regression.

The ``T_min`` X-ray-selection cut folds into :math:`J_b` exactly, because the
mask :math:`T_0 g(x) > T_{\\rm min}` depends only on :math:`T_0` and :math:`x`.

.. note::

   Requires float64 (``JAX_ENABLE_X64=1``).  The emission integral carries
   :math:`r_{\\rm cm}^2 \\sim 10^{49}`, which silently overflows float32 to ``inf``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["t_shape_g", "t0_of_mass", "build_j_table", "JTable",
           "emission_measure_factor", "shape_integral", "v_shape_of_mass",
           "lx_kt_of_mass"]

_M_PIVOT = 1.0e12          # DPM M_12 pivot [Msun/h]
_X_REF_FRAC = 0.3          # the DPM reference radius, r = 0.3 R_200


def _gnfw(x, a_in, a_tr, a_out):
    """gNFW shape f(x|α) (DPM Eq. 1) — mirrors hod_mod.gas._gnfw_f_params."""
    xs = np.maximum(np.asarray(x, float), 1e-12)
    return xs ** (-a_in) * (1.0 + xs ** a_tr) ** ((a_in - a_out) / a_tr)


def t_shape_g(x: np.ndarray, dp, pp) -> np.ndarray:
    """Fixed radial temperature shape ``g(x) = [f_P/f_n](x) / [f_P/f_n](x_ref)``.

    Independent of ``(n_e,0.3, β_n, P_0.3, β_P)`` — that independence is what makes
    :func:`build_j_table` a function of ``T_0`` alone.

    Parameters
    ----------
    x : radii in units of the scale radius R_s = R_200/c_DPM
    dp : GasDensityDPM (supplies f_n slopes)
    pp : PressureProfileDPM (supplies f_P slopes)
    """
    f_n = _gnfw(x, dp._alpha_in, dp._alpha_tr, dp._alpha_out)
    f_p = _gnfw(x, pp._alpha_in, pp._alpha_tr, pp._alpha_out_12)
    x_ref = _X_REF_FRAC * dp._C_DPM
    ratio_ref = (_gnfw(x_ref, pp._alpha_in, pp._alpha_tr, pp._alpha_out_12)
                 / _gnfw(x_ref, dp._alpha_in, dp._alpha_tr, dp._alpha_out))
    return (f_p / f_n) / ratio_ref


def t0_of_mass(m200, p03: float, ne03: float, beta_p: float, beta_n: float,
               ez: float, gamma_p: float = 8.0 / 3.0, gamma_n: float = 2.0) -> np.ndarray:
    """DPM temperature normalisation ``T_0(M)`` [keV] at the reference radius.

    ``T_0 = (P_0.3/n_e,0.3) · M_12^(β_P−β_n) · E(z)^(γ_P−γ_n)``.  Exact: verified
    against ``_integrate_profile`` to 1e-15 (the emission-weighted kT is strictly
    proportional to the pressure/density ratio at fixed shape).
    """
    m12 = np.asarray(m200, float) / _M_PIVOT
    return (float(p03) / float(ne03)) * m12 ** (float(beta_p) - float(beta_n)) \
        * float(ez) ** (gamma_p - gamma_n)


def emission_measure_factor(m200, ne03: float, beta_n: float, ez: float,
                            gamma_n: float = 2.0) -> np.ndarray:
    """``n_e,0.3² · M_12^(2β_n) · E(z)^(2γ_n)`` — the density part of L_X.

    The remaining geometric factor ``V_shape(M) ∝ R_200³`` and all absolute unit
    conversions are absorbed by the empirical ``c_total`` anchor of the band fit,
    so only the mass-scaling matters here.
    """
    m12 = np.asarray(m200, float) / _M_PIVOT
    return float(ne03) ** 2 * m12 ** (2.0 * float(beta_n)) * float(ez) ** (2.0 * gamma_n)


def shape_integral(dp, *, x_lo=1e-2, x_hi=None, n_x=400) -> float:
    """``∫ f_n(x)² x² dx / f_n(x_ref)²`` — the dimensionless part of V_shape.

    Depends only on the density shape (i.e. on ``p2`` via α_out) and the
    truncation ``r_max``; the mass enters separately through R_s³ (see
    :func:`v_shape_of_mass`).
    """
    if x_hi is None:
        x_hi = float(dp._r_max_factor) * dp._C_DPM
    x = np.linspace(float(x_lo), float(x_hi), int(n_x))
    f2 = _gnfw(x, dp._alpha_in, dp._alpha_tr, dp._alpha_out) ** 2
    f_ref = _gnfw(_X_REF_FRAC * dp._C_DPM, dp._alpha_in, dp._alpha_tr, dp._alpha_out)
    return float(np.trapezoid(f2 * x ** 2, x)) / float(f_ref) ** 2


def v_shape_of_mass(r200, c_dpm: float, shape_int: float, mpc_cm: float, h: float):
    """``V_shape(M) = 4π R_s³ · shape_integral`` in cm³, with R_s = R_200/c_DPM.

    Carries the whole mass dependence of the emission volume (∝ R_200³ ∝ M).  Its
    absolute scale is degenerate with the band fit's empirical ``c_total`` anchor,
    but it is kept physical so L_X stays interpretable in erg/s.
    """
    r_s_cm = (np.asarray(r200, float) / float(c_dpm)) * (float(mpc_cm) / float(h))
    return 4.0 * np.pi * r_s_cm ** 3 * float(shape_int)


def lx_kt_of_mass(m200, r200, r500c, dp, pp, cooling, *, ne03, beta_n, p03, beta_P,
                  ez, h, mpc_cm, z_metal=0.3, t_min=None, n_x=200):
    """:math:`R_{500c}`-integrated ``L_X`` [erg/s] and emission-weighted ``kT`` [keV].

    The band model above returns the luminosity integrated over the WHOLE profile
    (out to ``r_max R_200``), because that is what the angular cross-correlation
    sees.  The literature scaling relations — and the priors in
    :mod:`hod_mod.fitting.dpm_priors` — are instead defined inside
    :math:`R_{500c}`, so a plot comparing the fit against Lovisari/Bulbul must use
    this quantity, not ``_weight_bands``.

    Same exact factorisation as :func:`build_j_table`: :math:`T = T_0(M) g(x)` with
    a radial shape *g* that none of the four native parameters touch, so only the
    mass-dependent upper limit needs a per-mass quadrature.  Integration limits
    (:math:`0.01 R_{200}` to :math:`R_{500c}`) and the :math:`n_e^2` weighting match
    ``validate_gas_profiles._integrate_profile``, which is the definition the
    literature relations use.

    Kept in numpy end-to-end: the emission integral carries :math:`r_{\rm cm}^2
    \sim 10^{49}`, which overflows float32, so routing it through jnp would need
    ``JAX_ENABLE_X64=1``.  Only ``cooling`` is jnp, and its output is O(1e-23).

    Parameters
    ----------
    m200, r200, r500c : (NM,) [Msun/h, Mpc/h, Mpc/h]
    dp, pp : GasDensityDPM / PressureProfileDPM — supply the fixed f_n, g(x) shapes
    cooling : ApecCoolingTable for the band of interest (0.5-2 keV for L_X)
    ne03, beta_n, p03, beta_P : the four native DPM gas parameters
    ez, h, mpc_cm : E(z), little h, and the Mpc->cm conversion
    z_metal : gas metallicity [Z_sun]; t_min : X-ray selection cut [keV] or None

    Returns
    -------
    (lx, kt) : each (NM,)
    """
    m200 = np.asarray(m200, float)
    r200 = np.asarray(r200, float)
    r500c = np.asarray(r500c, float)
    c = dp._C_DPM

    x_lo = 0.01 * c                                  # r = 0.01 R_200, in R_s units
    x_hi = c * r500c / r200                          # r = R_500c
    x = x_lo + (x_hi - x_lo)[:, None] * np.linspace(0.0, 1.0, int(n_x))[None, :]

    f_n = _gnfw(x, dp._alpha_in, dp._alpha_tr, dp._alpha_out)
    f_ref = _gnfw(_X_REF_FRAC * c, dp._alpha_in, dp._alpha_tr, dp._alpha_out)
    t0 = t0_of_mass(m200, p03, ne03, beta_P, beta_n, ez)
    T = t0[:, None] * t_shape_g(x, dp, pp)                       # (NM, n_x) [keV]

    w = (f_n / f_ref) ** 2 * x ** 2
    if t_min is not None:
        w = np.where(T > float(t_min), w, 0.0)

    lam = np.asarray(cooling(T, np.full_like(T, float(z_metal))), float)
    em = emission_measure_factor(m200, ne03, beta_n, ez)
    r_s_cm = (r200 / c) * (float(mpc_cm) / float(h))
    lx = 4.0 * np.pi * r_s_cm ** 3 * em * np.trapezoid(w * lam, x, axis=1)
    denom = np.trapezoid(w, x, axis=1)
    kt = np.trapezoid(w * T, x, axis=1) / np.maximum(denom, 1e-300)
    return lx, kt


class JTable:
    """Interpolable ``J_b(T_0, Z)`` for a list of band cooling functions.

    Built once per profile shape; evaluated per MCMC step.
    """

    def __init__(self, log10_t0_grid, z_grid, tables):
        self._lt = np.asarray(log10_t0_grid, float)
        self._z = np.asarray(z_grid, float)
        self._tab = np.asarray(tables, float)      # (Nb, NT, NZ)

    @property
    def n_band(self) -> int:
        return self._tab.shape[0]

    def __call__(self, t0, z_metal) -> np.ndarray:
        """(Nb, NM) J_b at per-mass ``t0`` [keV] and scalar ``z_metal`` [Z_sun]."""
        from scipy.interpolate import RegularGridInterpolator
        lt = np.log10(np.clip(np.asarray(t0, float), 10 ** self._lt[0], 10 ** self._lt[-1]))
        zz = np.full_like(lt, float(np.clip(z_metal, self._z[0], self._z[-1])))
        pts = np.column_stack([lt, zz])
        out = np.empty((self._tab.shape[0], lt.size), float)
        for b in range(self._tab.shape[0]):
            itp = RegularGridInterpolator((self._lt, self._z), self._tab[b],
                                          method="linear", bounds_error=False,
                                          fill_value=None)
            out[b] = itp(pts)
        return out


def build_j_table(dp, pp, cool_bands, *, z_grid, log10_t0_grid,
                  x_lo=0.02, x_hi=None, n_x=400, t_min=None) -> JTable:
    """Tabulate ``J_b(T_0, Z)`` = shape-weighted radial band emissivity.

    .. math::

        J_b(T_0, Z) = \\frac{\\int f_n(x)^2 \\Lambda_b(T_0 g(x), Z) x^2 dx}
                           {\\int f_n(x)^2 x^2 dx}

    Parameters
    ----------
    dp, pp : GasDensityDPM / PressureProfileDPM — supply the fixed f_n, g(x) shapes
    cool_bands : sequence of ApecCoolingTable — Λ_b(T, Z), one per band
    z_grid, log10_t0_grid : tabulation axes
    x_lo, x_hi, n_x : radial quadrature in units of R_s (x_hi defaults to r_max·c_DPM)
    t_min : float | None — X-ray selection cut [keV]; folds in exactly (see module docs)
    """
    if x_hi is None:
        x_hi = float(dp._r_max_factor) * dp._C_DPM
    x = np.linspace(float(x_lo), float(x_hi), int(n_x))
    f_n2 = _gnfw(x, dp._alpha_in, dp._alpha_tr, dp._alpha_out) ** 2
    g = t_shape_g(x, dp, pp)
    w = f_n2 * x ** 2
    denom = float(np.trapezoid(w, x))

    lt = np.asarray(log10_t0_grid, float)
    zs = np.asarray(z_grid, float)
    nt, nz, nx = lt.size, zs.size, x.size

    # T(x | T_0) = T_0 g(x) on the full (T_0, Z, x) grid.  Evaluating each cooling
    # table ONCE on the flattened grid (rather than per (T_0, Z) node) is ~4 orders
    # of magnitude fewer Python-level calls — the difference between seconds and
    # hours for a 12-node shape grid.
    t_r = (10.0 ** lt)[:, None] * g[None, :]                       # (NT, nx)
    t_full = np.broadcast_to(t_r[:, None, :], (nt, nz, nx))
    z_full = np.broadcast_to(zs[None, :, None], (nt, nz, nx))
    mask = None if t_min is None else (t_r > float(t_min))[:, None, :]

    out = np.empty((len(cool_bands), nt, nz), float)
    for b, cb in enumerate(cool_bands):
        lam = np.asarray(cb(t_full.ravel(), z_full.ravel()), float).reshape(nt, nz, nx)
        integ = w[None, None, :] * lam
        if mask is not None:
            integ = np.where(mask, integ, 0.0)
        out[b] = np.trapezoid(integ, x, axis=2) / denom
    return JTable(lt, zs, out)
