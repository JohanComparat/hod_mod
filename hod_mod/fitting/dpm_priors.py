"""Back-propagate the L_X–M / kT–M scaling-relation priors onto the native DPM
profile parameters, carrying the full covariance.

Motivation
----------
The band fit's informative priors are physical statements about **scaling relations**
measured inside :math:`R_{500c}` (Comparat+2025 GAS.py):

.. math::

    \\log_{10}\\!\\big(L_X / E(z)^2\\big) &= {\\rm lx\\_norm}
        + {\\rm lx\\_slope}\\,(\\log_{10} M_{500c} - 15) \\\\
    \\log_{10}\\!\\big(kT / E(z)^{2/3}\\big) &= {\\rm kt\\_norm}
        + {\\rm kt\\_slope}\\,(\\log_{10} M_{500c} - 14)

Re-basing the fit onto the native DPM parameters
:math:`\\theta_n = (\\log_{10} n_{e,0.3},\\, \\beta_n,\\, \\log_{10} P_{0.3},\\, \\beta_P)`
does **not** invalidate that information — it just expresses it in different
coordinates.  Rather than invent fresh priors on :math:`\\theta_n`, this module maps
the scaling-relation prior through the model:

.. math::

    \\theta_s = f(\\theta_n), \\qquad
    J = \\partial f/\\partial \\theta_n \\;\\;(\\text{by JAX autodiff}),

.. math::

    \\theta_n^\\star : f(\\theta_n^\\star) = \\mu_s
    \\quad\\text{(Newton, using } J),
    \\qquad
    \\Sigma_n = J^{-1}\\, \\Sigma_s\\, J^{-\\mathsf{T}} .

The induced prior is a **full-covariance** Gaussian on :math:`\\theta_n`.  The
off-diagonal structure is the point: the naive guess "n_e sets L_X, P sets kT" is
only true to first order.  In truth

* :math:`kT = P/n_e` exactly, so ``kt_norm`` traces :math:`\\log_{10}(P_{0.3}/n_{e,0.3})`
  — it constrains a *ratio*, i.e. an anti-correlated combination — and
  ``kt_slope`` traces :math:`\\beta_P - \\beta_n`;
* :math:`L_X \\propto n_{e,0.3}^2 M^{2\\beta_n}\\,V_{\\rm shape}(M)\\,\\Lambda(T(M))`, so
  ``lx_slope`` picks up :math:`2\\beta_n + 1` (the :math:`V_{\\rm shape}\\propto R_{200}^3`
  volume term) **plus** a cooling-function cross-term in :math:`\\beta_P-\\beta_n` —
  L_X partly traces the pressure too.

A diagonal prior on :math:`\\theta_n` cannot represent either effect; this one does.

Differentiability
-----------------
The whole map is pure ``jnp``: the DPM profiles are analytic, and
:class:`~hod_mod.gas.cooling.ApecCoolingTable` is a jnp log-log interpolator built to
be differentiable.  Requires ``JAX_ENABLE_X64=1`` (the emission integral carries
:math:`r_{\\rm cm}^2 \\sim 10^{49}`, which overflows float32).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.gas.conversions import _MPC_CM

__all__ = ["ScalingCtx", "scaling_map", "induced_gaussian_prior"]

_X_REF_FRAC = 0.3
_C_DPM = 2.772


def _gnfw(x, a_in, a_tr, a_out):
    xs = jnp.maximum(x, 1e-12)
    return xs ** (-a_in) * (1.0 + xs ** a_tr) ** ((a_in - a_out) / a_tr)


class ScalingCtx:
    """Fixed (non-differentiated) context for :func:`scaling_map`.

    Parameters
    ----------
    m200 : (NM,) halo masses [Msun/h] spanning the group→cluster range the priors
        describe.
    r200, m500c, r500c : (NM,) matching radii/masses [Mpc/h, Msun/h] — computed once
        outside the map (they do not depend on the gas parameters).
    cooling : ApecCoolingTable for the broad 0.5–2 keV band (jnp, differentiable)
    shape : dict of the fixed gNFW slopes (density + pressure) and gamma exponents
    """

    def __init__(self, m200, r200, m500c, r500c, cooling, shape, z, ez, h,
                 z_metal=0.3, t_min=0.3, n_r=220, lx_pivot=15.0, kt_pivot=14.0):
        self.m200 = jnp.asarray(m200, float)
        self.r200 = jnp.asarray(r200, float)
        self.log10_m500c = jnp.log10(jnp.asarray(m500c, float) / float(h))  # physical Msun
        self.r500c = jnp.asarray(r500c, float)
        self.cooling = cooling
        self.shape = shape
        self.z, self.ez, self.h = float(z), float(ez), float(h)
        self.z_metal, self.t_min, self.n_r = float(z_metal), float(t_min), int(n_r)
        self.lx_pivot, self.kt_pivot = float(lx_pivot), float(kt_pivot)


def _profiles(theta_n, ctx):
    """n_e(r,M), T(r,M) on a per-mass radial grid out to R_500c.  Pure jnp."""
    log10_ne03, beta_n, log10_p03, beta_P = theta_n
    ne03, p03 = 10.0 ** log10_ne03, 10.0 ** log10_p03
    s = ctx.shape

    # radial grid per mass: 0.01 R200 -> R500c (the scaling relations are R500c-integrated)
    frac = jnp.linspace(0.0, 1.0, ctx.n_r)[None, :]
    r_lo = 0.01 * ctx.r200[:, None]
    r_hi = ctx.r500c[:, None]
    r = r_lo + (r_hi - r_lo) * frac                      # (NM, n_r) [Mpc/h]

    rs = (ctx.r200 / _C_DPM)[:, None]
    x = r / rs
    m12 = (ctx.m200 / 1.0e12)[:, None]

    f_n = _gnfw(x, s["a_in_n"], s["a_tr_n"], s["a_out_n"])
    f_p = _gnfw(x, s["a_in_p"], s["a_tr_p"], s["a_out_p"])
    x_ref = _X_REF_FRAC * _C_DPM
    fn_ref = _gnfw(jnp.asarray(x_ref), s["a_in_n"], s["a_tr_n"], s["a_out_n"])
    fp_ref = _gnfw(jnp.asarray(x_ref), s["a_in_p"], s["a_tr_p"], s["a_out_p"])

    ne = (ne03 / fn_ref) * f_n * ctx.ez ** s["gamma_n"] * m12 ** beta_n
    pe = (p03 / fp_ref) * f_p * ctx.ez ** s["gamma_p"] * m12 ** beta_P
    T = pe / jnp.maximum(ne, 1e-30)                      # keV  (k_B T = P/n_e)
    return r, ne, T


def scaling_map(theta_n, ctx):
    """``theta_n -> [lx_norm, lx_slope, kt_norm, kt_slope]``.  Differentiable.

    L_X and kT_ew are integrated inside R_500c exactly as the scaling relations are
    defined, then a log-log linear fit against log10 M_500c gives (norm, slope) at
    the same pivots the priors use.
    """
    r, ne, T = _profiles(theta_n, ctx)
    r_cm = r * (_MPC_CM / ctx.h)
    lam = ctx.cooling(T, jnp.full_like(T, ctx.z_metal))          # (NM, n_r)

    # X-ray selection: only gas above T_min contributes (smooth sigmoid instead of a
    # hard step so the map stays differentiable; width 0.02 dex ~ the hard cut).
    w_sel = jax.nn.sigmoid((jnp.log10(jnp.maximum(T, 1e-12))
                            - jnp.log10(ctx.t_min)) / 0.02)
    em = ne ** 2 * w_sel
    r2 = r_cm ** 2

    lx = 4.0 * jnp.pi * jnp.trapezoid(em * lam * r2, r_cm, axis=1)        # (NM,) erg/s
    denom = jnp.trapezoid(em * r2, r_cm, axis=1)
    kt = jnp.trapezoid(em * T * r2, r_cm, axis=1) / jnp.maximum(denom, 1e-300)

    # log-log fits vs log10 M500c at the prior pivots
    lm = ctx.log10_m500c
    y_lx = jnp.log10(jnp.maximum(lx, 1e-30)) - 2.0 * jnp.log10(ctx.ez)
    y_kt = jnp.log10(jnp.maximum(kt, 1e-30)) - (2.0 / 3.0) * jnp.log10(ctx.ez)

    def _fit(y, pivot):
        xc = lm - pivot
        A = jnp.stack([jnp.ones_like(xc), xc], axis=1)          # (NM, 2)
        coef = jnp.linalg.lstsq(A, y, rcond=None)[0]
        return coef[0], coef[1]                                 # norm at pivot, slope

    lx_norm, lx_slope = _fit(y_lx, ctx.lx_pivot)
    kt_norm, kt_slope = _fit(y_kt, ctx.kt_pivot)
    return jnp.array([lx_norm, lx_slope, kt_norm, kt_slope])


def induced_gaussian_prior(ctx, mu_s, sigma_s, theta0, n_newton=12, tol=1e-10):
    """Map a Gaussian prior on the scaling relations onto the native DPM params.

    Parameters
    ----------
    ctx : ScalingCtx
    mu_s : (4,) prior means [lx_norm, lx_slope, kt_norm, kt_slope]
    sigma_s : (4,) prior sigmas (diagonal in scaling-relation space)
    theta0 : (4,) starting guess for the native params
    """
    f = lambda t: scaling_map(t, ctx)
    jac = jax.jacfwd(f)
    mu_s = jnp.asarray(mu_s, float)

    # Newton solve f(theta*) = mu_s  (the prior CENTRE in native coordinates)
    t = jnp.asarray(theta0, float)
    for _ in range(n_newton):
        resid = f(t) - mu_s
        if float(jnp.max(jnp.abs(resid))) < tol:
            break
        J = jac(t)
        t = t - jnp.linalg.solve(J, resid)

    J = jac(t)
    Jinv = jnp.linalg.inv(J)
    Sigma_s = jnp.diag(jnp.asarray(sigma_s, float) ** 2)
    Sigma_n = Jinv @ Sigma_s @ Jinv.T
    Sigma_n = 0.5 * (Sigma_n + Sigma_n.T)          # symmetrise against round-off
    return (np.asarray(t), np.asarray(Sigma_n), np.asarray(J),
            np.asarray(f(t)))
