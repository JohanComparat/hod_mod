"""Truncated NFW and Hernquist lensing profiles (pure JAX).

Analytic 3D densities, enclosed masses, projected surface densities Σ(R),
mean surface densities Σ̄(<R), excess surface densities ΔΣ(R) and normalized
Fourier windows û(k) for three profile families that complement the infinite
NFW of :mod:`hod_mod.core.halo_profiles`:

* **Sharply truncated NFW** (Takada & Jain 2003) — NFW density cut at
  r_t = c_t·r_s (``tnfw_*``).
* **Smoothly truncated NFW** (Baltz, Marshall & Oguri 2009, n=2) —
  ρ_NFW(r) × [r_t²/(r² + r_t²)]² with τ = r_t/r_s (``bmo_*``).
* **Hernquist** (1990) — ρ ∝ 1/[x(1+x)³] (``hernquist_*``).

Ported from the ``halo_lensing`` reference implementation (companion code of
Oguri et al. 2026, PASJ 78, 416, arXiv:2512.13954,
https://github.com/massarin/halo_lensing) with two deliberate changes:

* pure JAX throughout (no colossus/scipy at call time), with autodiff-safe
  piecewise branches (double-``where`` on inputs, series windows at x = 1);
* the sharply truncated profile is normalized to the **exact projection of
  the truncated NFW density**: Σ = 4 ρ_s r_s f(x, c_t).  The reference's
  real-space ``tj_*`` functions are a factor m_nfw(c) larger than this
  (inconsistent with their own Fourier window ``y_tj``, which integrates to
  the halo mass); verified against direct numerical Abel projection.

Units follow the house convention: radii comoving [Mpc/h], masses [Msun/h],
Σ [Msun h/Mpc²], ρ [Msun h²/Mpc³].

Float32 note: ``bmo_mean_sigma`` and the x ≈ 1 seams lose relative accuracy
in float32 for x = R/r_s ≲ 0.05 (log-term cancellations inherited from the
closed forms).  Enable ``jax_enable_x64`` in entry-point scripts (house
pattern) when sub-percent accuracy at small radii matters.

References
----------
Takada & Jain 2003, MNRAS 340, 580 (arXiv:astro-ph/0209167)
Baltz, Marshall & Oguri 2009, JCAP 01, 015 (arXiv:0705.0682)
Hernquist 1990, ApJ 356, 359
Oguri et al. 2026, PASJ 78, 416 (arXiv:2512.13954)
Wright & Brainerd 2000, ApJ 534, 34 — NFW comparison limits
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import exp1, expi

from .halo_profiles import _si_jax, _ci_jax, nfw_uk_jax, mdef_delta_rho

__all__ = [
    "tnfw_rho", "tnfw_mass", "tnfw_sigma", "tnfw_mean_sigma",
    "tnfw_delta_sigma", "tnfw_uk",
    "bmo_rho", "bmo_mass", "bmo_mass_total", "bmo_sigma", "bmo_mean_sigma",
    "bmo_delta_sigma", "bmo_uk_jax",
    "hernquist_rho", "hernquist_mass", "hernquist_sigma",
    "hernquist_mean_sigma", "hernquist_delta_sigma", "HERNQUIST_RB_RE",
    "nfw_params_from_mass",
]

# x = 1 branch handling.  TJ kernels: quadratic interpolation window of
# half-width _TJ_W with closed-form anchors at δ = ±_TJ_D (c-dependent series
# avoided).  Hernquist/BMO kernels: analytic quadratic series inside
# |x−1| < _X1_EPS, derived symbolically (sympy, 2026-07-07):
#   F(1+δ)                    = 1 − 2δ/3 + 7δ²/15 − 12δ³/35
#   [(2+x²)F−3]/(x²−1)²       = 4/15 − 16δ/35 + 8δ²/15    (Hernquist Σ)
#   2(1−F)/(x²−1)             = 2/3 − 4δ/5 + 26δ²/35      (Hernquist Σ̄)
#   (1−F)/(x²−1)              = 1/3 − 2δ/5 + 13δ²/35      (BMO f1)
_TJ_W = 1.0e-2
_TJ_D = 2.0e-2
_X1_EPS = 1.0e-2


def _m_nfw(x):
    """Dimensionless NFW enclosed mass m(x) = ln(1+x) − x/(1+x)."""
    return jnp.log1p(x) - x / (1.0 + x)


# ---------------------------------------------------------------------------
# Takada & Jain 2003 — sharply truncated NFW (cut at r = c_t · r_s)
# ---------------------------------------------------------------------------

def _quad_window(x, f_m, f_0, f_p):
    """Quadratic through (1−D, f_m), (1, f_0), (1+D, f_p) evaluated at x.

    Used to bridge the |x−1| ≤ W window where the closed forms cancel
    catastrophically (values like 1/δ, gradients like 1/δ²).  The anchors
    sit at δ = ±D in the well-conditioned region, so the interpolant (and
    its gradient) is float32-accurate; interpolation error is O(f‴ D³)
    ≈ 1e-6 for these kernels.
    """
    dlt = x - 1.0
    return (f_0 + (f_p - f_m) / (2.0 * _TJ_D) * dlt
            + (f_p - 2.0 * f_0 + f_m) / (2.0 * _TJ_D**2) * dlt**2)


def _tj_sigma_lo(x, c):
    """TJ Σ kernel, x < 1 branch: −s/((1−x²)(1+c)) + arccosh(u)/(1−x²)^{3/2}.

    arccosh(u) is computed from d = u−1 = (1−x)(c−x)/(x(1+c)) (exact
    product — no cancellation near x = 1)."""
    s = jnp.sqrt(c * c - x**2)
    one_m_x2 = 1.0 - x**2
    d = (1.0 - x) * (c - x) / (x * (1.0 + c))
    acosh_u = jnp.log1p(d + jnp.sqrt(d * (d + 2.0)))
    return -s / (one_m_x2 * (1.0 + c)) + acosh_u / (one_m_x2 * jnp.sqrt(one_m_x2))


def _tj_sigma_hi(x, c):
    """TJ Σ kernel, 1 < x < c branch: s/((x²−1)(1+c)) − arccos(u)/(x²−1)^{3/2}.

    arccos(u) = 2 arcsin(√(d/2)) with d = 1−u — stable at both ends."""
    s = jnp.sqrt(jnp.maximum(c * c - x**2, 1e-30))
    x2_m_one = x**2 - 1.0
    d = (x - 1.0) * (c - x) / (x * (1.0 + c))
    acos_u = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(0.5 * d, 0.0, 1.0)))
    return s / (x2_m_one * (1.0 + c)) - acos_u / (x2_m_one * jnp.sqrt(x2_m_one))


def _tj_sigma_dl(x, c):
    """Dimensionless TJ surface density f(x, c):  Σ = 4 ρ_s r_s f.

    Branches: x < 1 (arccosh), 1 < x < c (arccos), x ≥ c (zero), with a
    quadratic interpolation window |x−1| ≤ 0.01 (see :func:`_quad_window`).
    Requires c > 1 + 0.02 (truncation outside the scale radius).
    """
    lo = x < 1.0 - _TJ_W
    hi = (x > 1.0 + _TJ_W) & (x < c)
    inside = x < c

    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 0.5 * (1.0 + c))

    # window anchors (scalar in x, closed forms at δ = ±D and the exact x=1 limit)
    f_m = _tj_sigma_lo(1.0 - _TJ_D, c)
    f_p = _tj_sigma_hi(1.0 + _TJ_D, c)
    f_0 = jnp.sqrt(c * c - 1.0) * (1.0 + 1.0 / (1.0 + c)) / (3.0 * (1.0 + c))

    f = jnp.where(lo, _tj_sigma_lo(x_lo, c),
                  jnp.where(hi, _tj_sigma_hi(x_hi, c),
                            _quad_window(x, f_m, f_0, f_p)))
    return 0.5 * jnp.where(inside, f, 0.0)


def _tj_bsigma_lo(x, c):
    """TJ Σ̄ kernel, x < 1 branch — cancellation-free rewrite of
    (s−c)/(x²(1+c)) + ln(x(1+c)/(c+s))/x² + arccosh(u)/(x²√(1−x²)):
    the ~ln(x)/x² pieces are paired into a log1p of an O(x²) argument, so
    small projected radii stay accurate (the reference form loses all
    float32 precision below x ≈ 3e-3)."""
    s = jnp.sqrt(c * c - x**2)
    q = jnp.sqrt(1.0 - x**2)
    d = (1.0 - x) * (c - x) / (x * (1.0 + c))
    acosh_u = jnp.log1p(d + jnp.sqrt(d * (d + 2.0)))
    return (-1.0 / ((s + c) * (1.0 + c))
            + jnp.log1p(x**2 * (1.0 - s / (1.0 + q)) / (c + s)) / x**2
            + acosh_u / (q * (1.0 + q)))


def _tj_bsigma_hi(x, c):
    """TJ Σ̄ kernel, 1 < x < c branch (reference form; no small-x issue)."""
    s = jnp.sqrt(jnp.maximum(c * c - x**2, 1e-30))
    q = jnp.sqrt(x**2 - 1.0)
    d = (x - 1.0) * (c - x) / (x * (1.0 + c))
    acos_u = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(0.5 * d, 0.0, 1.0)))
    return (-1.0 / ((s + c) * (1.0 + c))
            + jnp.log(x * (1.0 + c) / (c + s)) / x**2
            + acos_u / (x**2 * q))


def _tj_bsigma_dl(x, c):
    """Dimensionless TJ mean surface density g(x, c):  Σ̄(<R) = 4 ρ_s r_s g."""
    lo = x < 1.0 - _TJ_W
    hi = (x > 1.0 + _TJ_W) & (x < c)
    out = x >= c

    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 0.5 * (1.0 + c))
    x_out = jnp.where(x > 0, x, 1.0)

    g_m = _tj_bsigma_lo(1.0 - _TJ_D, c)
    g_p = _tj_bsigma_hi(1.0 + _TJ_D, c)
    sc1 = jnp.sqrt(c * c - 1.0)
    g_0 = (2.0 * sc1 - c) / (1.0 + c) + jnp.log((1.0 + c) / (c + sc1))

    # x ≥ c: all truncated mass enclosed — Σ̄ = M_t/(πR²)
    g_out = _m_nfw(c) / x_out**2

    return jnp.where(out, g_out,
                     jnp.where(lo, _tj_bsigma_lo(x_lo, c),
                               jnp.where(hi, _tj_bsigma_hi(x_hi, c),
                                         _quad_window(x, g_m, g_0, g_p))))


@jax.jit
def tnfw_rho(r: jnp.ndarray, rho_s: float, r_s: float, c_t: float) -> jnp.ndarray:
    """Sharply truncated NFW 3D density [Msun h²/Mpc³]: NFW for r < c_t·r_s, else 0."""
    x = r / r_s
    x_safe = jnp.where(x > 0, x, 1.0)
    rho = rho_s / (x_safe * (1.0 + x_safe) ** 2)
    return jnp.where(x < c_t, rho, 0.0)


@jax.jit
def tnfw_mass(r: jnp.ndarray, rho_s: float, r_s: float, c_t: float) -> jnp.ndarray:
    """Enclosed mass M(<r) [Msun/h] of the sharply truncated NFW profile.

    Total (truncated) mass is 4π ρ_s r_s³ m_nfw(c_t), reached at r = c_t·r_s.
    """
    x = jnp.minimum(r / r_s, c_t)
    return 4.0 * jnp.pi * rho_s * r_s**3 * _m_nfw(x)


@jax.jit
def tnfw_sigma(R: jnp.ndarray, rho_s: float, r_s: float, c_t: float) -> jnp.ndarray:
    """Projected surface density Σ(R) [Msun h/Mpc²] of the truncated NFW.

    Takada & Jain 2003 closed form, normalized to the exact projection of
    the truncated density (see module docstring re the reference code's
    extra m_nfw(c) factor).  Σ ≡ 0 for R ≥ c_t·r_s.
    """
    return 4.0 * rho_s * r_s * _tj_sigma_dl(R / r_s, c_t)


@jax.jit
def tnfw_mean_sigma(R: jnp.ndarray, rho_s: float, r_s: float, c_t: float) -> jnp.ndarray:
    """Mean surface density Σ̄(<R) [Msun h/Mpc²] of the truncated NFW.

    Mass conservation: π R² Σ̄ → 4π ρ_s r_s³ m_nfw(c_t) for R ≥ c_t·r_s.
    """
    return 4.0 * rho_s * r_s * _tj_bsigma_dl(R / r_s, c_t)


@jax.jit
def tnfw_delta_sigma(R: jnp.ndarray, rho_s: float, r_s: float, c_t: float) -> jnp.ndarray:
    """Excess surface density ΔΣ(R) = Σ̄(<R) − Σ(R) [Msun h/Mpc²]."""
    return tnfw_mean_sigma(R, rho_s, r_s, c_t) - tnfw_sigma(R, rho_s, r_s, c_t)


#: Normalized Fourier window of the sharply truncated NFW.  The analytic NFW
#: û(k) of Cooray & Sheth 2002 Eq. 11 *is* the Takada–Jain window y_TJ(k·r_s, c)
#: (the integral is cut at r = c·r_s), so this is an alias, with truncation
#: concentration c_t passed as the third argument.
tnfw_uk = nfw_uk_jax


# ---------------------------------------------------------------------------
# Baltz, Marshall & Oguri 2009 — smoothly truncated NFW (n = 2)
# ---------------------------------------------------------------------------

def _bmo_ff(x):
    """(f1, F) with F(x) the NFW projection kernel and f1 = (F−1)/(1−x²).

    F(x) = arctanh(√(1−x²))/√(1−x²) for x < 1, arctan(√(x²−1))/√(x²−1)
    for x > 1; quadratic series inside |x−1| < 1e-2 (1/δ² cancellation).
    """
    lo = x < 1.0 - _X1_EPS
    hi = x > 1.0 + _X1_EPS
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 2.0)

    q_lo = jnp.sqrt(1.0 - x_lo**2)
    f1_lo = (2.0 * jnp.arctanh(jnp.sqrt((1.0 - x_lo) / (1.0 + x_lo))) / q_lo
             - 1.0) / (1.0 - x_lo**2)
    q_hi = jnp.sqrt(x_hi**2 - 1.0)
    f1_hi = (1.0 - 2.0 * jnp.arctan(jnp.sqrt((x_hi - 1.0) / (x_hi + 1.0))) / q_hi
             ) / (x_hi**2 - 1.0)
    dlt = x - 1.0
    f1_mid = 1.0 / 3.0 - 2.0 * dlt / 5.0 + 13.0 * dlt**2 / 35.0

    f1 = jnp.where(lo, f1_lo, jnp.where(hi, f1_hi, f1_mid))
    f2 = f1 * (1.0 - x**2) + 1.0   # = F(x)
    return f1, f2


def _bmo_L(x, tau):
    """L(x, τ) = ln[x/(√(x²+τ²)+τ)] (BMO09 App. A)."""
    x_safe = jnp.where(x > 0, x, 1.0)
    return jnp.log(x_safe / (jnp.sqrt(x_safe**2 + tau**2) + tau))


def _bmo_sigma_dl(x, tau):
    """Dimensionless BMO surface density (BMO09 Eq. A.28): Σ = 4 ρ_s r_s f."""
    ff1, ff2 = _bmo_ff(x)
    t2 = tau * tau
    tx2 = t2 + x**2
    pre = t2**2 / (4.0 * (t2 + 1.0) ** 3)
    return pre * (
        2.0 * (t2 + 1.0) * ff1
        + 8.0 * ff2
        + (t2**2 - 1.0) / (t2 * tx2)
        - jnp.pi * (4.0 * tx2 + t2 + 1.0) / (tx2 * jnp.sqrt(tx2))
        + (t2 * (t2**2 - 1.0) + tx2 * (3.0 * t2**2 - 6.0 * t2 - 1.0))
        * _bmo_L(x, tau) / (tau**3 * tx2 * jnp.sqrt(tx2))
    )


def _bmo_bsigma_dl(x, tau):
    """Dimensionless BMO mean surface density: Σ̄(<R) = 4 ρ_s r_s g.

    Float32 caution: the closed form cancels ~ln(x)/x² terms; accuracy
    degrades below x ≈ 0.05 in float32 (exact in float64).
    """
    ff1, ff2 = _bmo_ff(x)
    t2 = tau * tau
    x_safe = jnp.where(x > 0, x, 1.0)
    tx2 = t2 + x_safe**2
    pre = t2**2 / (2.0 * (t2 + 1.0) ** 3 * x_safe**2)
    return pre * (
        2.0 * (t2 + 4.0 * x_safe**2 - 3.0) * ff2
        + (jnp.pi * (3.0 * t2 - 1.0)
           + 2.0 * tau * (t2 - 3.0) * jnp.log(tau)) / tau
        + (
            -(tau**3) * jnp.pi * (4.0 * x_safe**2 + 3.0 * t2 - 1.0)
            + (2.0 * t2**2 * (t2 - 3.0)
               + x_safe**2 * (3.0 * t2**2 - 6.0 * t2 - 1.0))
            * _bmo_L(x_safe, tau)
        ) / (tau**3 * jnp.sqrt(tx2))
    )


def _m_bmo_dl(x, tau):
    """Dimensionless BMO enclosed mass: M(<r) = 4π ρ_s r_s³ m(x, τ)."""
    t2 = tau * tau
    pre = t2 / (2.0 * (t2 + 1.0) ** 3 * (1.0 + x) * (t2 + x**2))
    x_safe = jnp.where(x > 0, x, 1.0)
    return pre * (
        (t2 + 1.0) * x * (x * (x + 1.0)
                          - t2 * (x - 1.0) * (2.0 + 3.0 * x) - 2.0 * t2**2)
        + tau * (x + 1.0) * (t2 + x**2)
        * (2.0 * (3.0 * t2 - 1.0) * jnp.arctan(x_safe / tau)
           + tau * (t2 - 3.0)
           * jnp.log(t2 * (1.0 + x_safe) ** 2 / (t2 + x_safe**2)))
    )


def _m_bmo_tot_dl(tau):
    """Dimensionless BMO total mass: M_tot = 4π ρ_s r_s³ m_tot(τ)."""
    t2 = tau * tau
    return t2 / (2.0 * (t2 + 1.0) ** 3) * (
        (3.0 * t2 - 1.0) * (jnp.pi * tau - t2 - 1.0)
        + 2.0 * t2 * (t2 - 3.0) * jnp.log(tau)
    )


@jax.jit
def bmo_rho(r: jnp.ndarray, rho_s: float, r_s: float, tau: float) -> jnp.ndarray:
    """BMO 3D density [Msun h²/Mpc³]: ρ_NFW(r) × [τ²/(τ² + x²)]², x = r/r_s."""
    x = r / r_s
    x_safe = jnp.where(x > 0, x, 1.0)
    rho_nfw = rho_s / (x_safe * (1.0 + x_safe) ** 2)
    return rho_nfw * (tau**2 / (tau**2 + x_safe**2)) ** 2


@jax.jit
def bmo_mass(r: jnp.ndarray, rho_s: float, r_s: float, tau: float) -> jnp.ndarray:
    """Enclosed mass M(<r) [Msun/h] of the BMO profile (BMO09 Eq. A.2)."""
    return 4.0 * jnp.pi * rho_s * r_s**3 * _m_bmo_dl(r / r_s, tau)


@jax.jit
def bmo_mass_total(rho_s: float, r_s: float, tau: float) -> jnp.ndarray:
    """Total (finite) BMO mass [Msun/h]: 4π ρ_s r_s³ m_tot(τ).

    For (ρ_s, r_s) matched to an NFW halo of mass M_Δ = 4π ρ_s r_s³ m_nfw(c),
    the ratio M_tot/M_Δ = m_tot(τ)/m_nfw(c) (≈ 1.34 for τ = 2.5c, c = 6).
    """
    return 4.0 * jnp.pi * rho_s * r_s**3 * _m_bmo_tot_dl(tau)


@jax.jit
def bmo_sigma(R: jnp.ndarray, rho_s: float, r_s: float, tau: float) -> jnp.ndarray:
    """Projected surface density Σ(R) [Msun h/Mpc²] of the BMO profile."""
    return 4.0 * rho_s * r_s * _bmo_sigma_dl(R / r_s, tau)


@jax.jit
def bmo_mean_sigma(R: jnp.ndarray, rho_s: float, r_s: float, tau: float) -> jnp.ndarray:
    """Mean surface density Σ̄(<R) [Msun h/Mpc²] of the BMO profile."""
    return 4.0 * rho_s * r_s * _bmo_bsigma_dl(R / r_s, tau)


@jax.jit
def bmo_delta_sigma(R: jnp.ndarray, rho_s: float, r_s: float, tau: float) -> jnp.ndarray:
    """Excess surface density ΔΣ(R) = Σ̄(<R) − Σ(R) [Msun h/Mpc²]."""
    return bmo_mean_sigma(R, rho_s, r_s, tau) - bmo_sigma(R, rho_s, r_s, tau)


# --- hyperbolic sine/cosine integral combinations for the BMO Fourier window

def _pq_hyperbolic(x):
    """P(x) = sinh(x)Chi(x) − cosh(x)Shi(x) and Q(x) = cosh(x)Chi(x) − sinh(x)Shi(x).

    Computed cancellation-free via Ei = Shi + Chi, E1 = Shi − Chi:
        P = −[eˣE1(x) + e⁻ˣEi(x)]/2,   Q = [e⁻ˣEi(x) − eˣE1(x)]/2
    with the scaled asymptotic series for x ≥ 30 (seam agreement ~5e-8):
        P ≈ −(1/x + 2/x³ + 24/x⁵ + 720/x⁷),  Q ≈ 1/x² + 6/x⁴ + 120/x⁶ + 5040/x⁸.
    Valid for x > 0.
    """
    x_safe = jnp.where(x > 0, x, 1.0)
    small = x_safe < 30.0
    x_sm = jnp.where(small, x_safe, 1.0)

    e1s = jnp.exp(x_sm) * exp1(x_sm)     # eˣ E1(x)
    eis = jnp.exp(-x_sm) * expi(x_sm)    # e⁻ˣ Ei(x)
    p_sm = -0.5 * (e1s + eis)
    q_sm = 0.5 * (eis - e1s)

    xi = 1.0 / x_safe
    xi2 = xi * xi
    p_lg = -xi * (1.0 + xi2 * (2.0 + xi2 * (24.0 + xi2 * 720.0)))
    q_lg = xi2 * (1.0 + xi2 * (6.0 + xi2 * (120.0 + xi2 * 5040.0)))

    return jnp.where(small, p_sm, p_lg), jnp.where(small, q_sm, q_lg)


@jax.jit
def bmo_uk_jax(
    k_arr: jnp.ndarray,
    r_s_arr: jnp.ndarray,
    tau_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Normalized BMO Fourier window û(k) (autodiff-compatible).

    Analytic Fourier transform of the BMO (n=2) density (Oguri et al. 2026
    App.; reference ``y_bmo``), renormalized to the **total** BMO mass so
    û(k→0) = 1, matching the halo-model convention of :func:`nfw_uk_jax`
    (the reference normalizes by m_nfw(c), i.e. by M_Δ instead).

    Parameters
    ----------
    k_arr : shape (Nk,), wavenumbers [h/Mpc]
    r_s_arr : shape (NM,), scale radii [Mpc/h]
    tau_arr : shape (NM,), truncation ratios τ = r_t/r_s

    Returns
    -------
    uk : shape (Nk, NM)

    Float32 note: log-term cancellations grow toward small k·r_s; guarded
    below k·r_s = 1e-4 (→ 1).  Use x64 for sub-0.1% accuracy at k·r_s < 0.01.
    """
    k = jnp.asarray(k_arr).reshape(-1, 1)      # (Nk, 1)
    r_s = jnp.asarray(r_s_arr).reshape(1, -1)  # (1, NM)
    tau = jnp.asarray(tau_arr).reshape(1, -1)  # (1, NM)

    K = k * r_s
    # Safe-input threshold matches the output guard below so gradients never
    # flow through the cancellation-prone small-K expression.
    K_safe = jnp.where(K >= 1e-4, K, 1.0)
    t2 = tau * tau

    si = _si_jax(K_safe)
    ci = _ci_jax(K_safe)
    p, q = _pq_hyperbolic(tau * K_safe)

    sK, cK = jnp.sin(K_safe), jnp.cos(K_safe)
    f2 = (
        2.0 * (3.0 * t2**2 - 6.0 * t2 - 1.0) * p
        - 2.0 * tau * (t2**2 - 1.0) * K_safe * q
        - 2.0 * t2 * jnp.pi * jnp.exp(-tau * K_safe) * ((t2 + 1.0) * K_safe + 4.0 * tau)
        + 2.0 * tau**3 * (jnp.pi - 2.0 * si) * (4.0 * cK + (t2 + 1.0) * K_safe * sK)
        + 4.0 * tau**3 * ci * (4.0 * sK - (t2 + 1.0) * K_safe * cK)
    )
    uk = tau / (4.0 * _m_bmo_tot_dl(tau) * (1.0 + t2) ** 3 * K_safe) * f2

    return jnp.where(K < 1e-4, 1.0, uk)


# ---------------------------------------------------------------------------
# Hernquist 1990
# ---------------------------------------------------------------------------

#: Scale radius over effective (projected half-mass) radius: r_b = 0.551 r_e
#: (Hernquist 1990, Eq. 38).
HERNQUIST_RB_RE = 0.551


def _hern_sigma_dl(x):
    """Dimensionless Hernquist Σ kernel (Hernquist 1990 Eq. 32):
    Σ = M/(2π r_b²) f(x)."""
    lo = x < 1.0 - _X1_EPS
    hi = x > 1.0 + _X1_EPS
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 2.0)

    a_lo = jnp.sqrt(1.0 - x_lo**2)
    f_lo = ((2.0 + x_lo**2) * jnp.arctanh(a_lo) / a_lo - 3.0) / (x_lo**2 - 1.0) ** 2
    a_hi = jnp.sqrt(x_hi**2 - 1.0)
    f_hi = ((2.0 + x_hi**2) * jnp.arctan(a_hi) / a_hi - 3.0) / (x_hi**2 - 1.0) ** 2
    dlt = x - 1.0
    f_mid = 4.0 / 15.0 - 16.0 * dlt / 35.0 + 8.0 * dlt**2 / 15.0

    return jnp.where(lo, f_lo, jnp.where(hi, f_hi, f_mid))


def _hern_bsigma_dl(x):
    """Dimensionless Hernquist Σ̄ kernel: Σ̄(<R) = M/(2π r_b²) g(x);
    g → 2/x² as x → ∞ (total mass M)."""
    lo = x < 1.0 - _X1_EPS
    hi = x > 1.0 + _X1_EPS
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 2.0)

    a_lo = jnp.sqrt(1.0 - x_lo**2)
    g_lo = 2.0 * (1.0 - jnp.arctanh(a_lo) / a_lo) / (x_lo**2 - 1.0)
    a_hi = jnp.sqrt(x_hi**2 - 1.0)
    g_hi = 2.0 * (1.0 - jnp.arctan(a_hi) / a_hi) / (x_hi**2 - 1.0)
    dlt = x - 1.0
    g_mid = 2.0 / 3.0 - 4.0 * dlt / 5.0 + 26.0 * dlt**2 / 35.0

    return jnp.where(lo, g_lo, jnp.where(hi, g_hi, g_mid))


@jax.jit
def hernquist_rho(r: jnp.ndarray, m_tot: float, r_b: float) -> jnp.ndarray:
    """Hernquist 3D density ρ(r) = M r_b / (2π r (r + r_b)³) [Msun h²/Mpc³]."""
    r_safe = jnp.where(r > 0, r, r_b)
    return m_tot * r_b / (2.0 * jnp.pi * r_safe * (r_safe + r_b) ** 3)


@jax.jit
def hernquist_mass(r: jnp.ndarray, m_tot: float, r_b: float) -> jnp.ndarray:
    """Hernquist enclosed mass M(<r) = M x²/(1+x)², x = r/r_b [Msun/h]."""
    x = r / r_b
    return m_tot * x**2 / (1.0 + x) ** 2


@jax.jit
def hernquist_sigma(R: jnp.ndarray, m_tot: float, r_b: float) -> jnp.ndarray:
    """Hernquist projected surface density Σ(R) [Msun h/Mpc²].

    r_b is the Hernquist scale radius; for a measured effective radius use
    r_b = HERNQUIST_RB_RE × r_e.
    """
    return m_tot / (2.0 * jnp.pi * r_b**2) * _hern_sigma_dl(R / r_b)


@jax.jit
def hernquist_mean_sigma(R: jnp.ndarray, m_tot: float, r_b: float) -> jnp.ndarray:
    """Hernquist mean surface density Σ̄(<R) [Msun h/Mpc²]; π R² Σ̄ → M."""
    return m_tot / (2.0 * jnp.pi * r_b**2) * _hern_bsigma_dl(R / r_b)


@jax.jit
def hernquist_delta_sigma(R: jnp.ndarray, m_tot: float, r_b: float) -> jnp.ndarray:
    """Hernquist excess surface density ΔΣ(R) = Σ̄(<R) − Σ(R) [Msun h/Mpc²]."""
    return hernquist_mean_sigma(R, m_tot, r_b) - hernquist_sigma(R, m_tot, r_b)


# ---------------------------------------------------------------------------
# Mass-definition plumbing (colossus-free)
# ---------------------------------------------------------------------------

def nfw_params_from_mass(
    m_h: jnp.ndarray,
    c: jnp.ndarray,
    z: float,
    theta_cosmo: dict,
    mdef: str = "200c",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(ρ_s, r_s, r_Δ) from (M_Δ, c) — functional twin of
    :meth:`HaloProfile.rho_s_and_rs`, plus the overdensity radius.

    Uses :func:`hod_mod.core.halo_profiles.mdef_delta_rho` for the
    (delta, rho_ref) of ``mdef`` ('200m', '200c', 'vir'), all in comoving
    h-units.  Differentiable w.r.t. ``m_h`` and ``c``.
    """
    delta, rho_ref = mdef_delta_rho(mdef, float(z), theta_cosmo)
    r_delta = (3.0 * m_h / (4.0 * jnp.pi * delta * rho_ref)) ** (1.0 / 3.0)
    r_s = r_delta / c
    rho_s = m_h / (4.0 * jnp.pi * r_s**3 * _m_nfw(c))
    return rho_s, r_s, r_delta
