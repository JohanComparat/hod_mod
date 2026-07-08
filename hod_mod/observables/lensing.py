"""Halo weak- and strong-lensing observables (pure JAX).

Builds cluster/galaxy lensing predictions from the analytic profiles of
:mod:`hod_mod.core.lensing_profiles` (+ the infinite NFW of
:mod:`hod_mod.core.halo_profiles`):

* **critical surface density** Σ_crit(z_l, z_s) and κ/κ̄/γ_t conversions;
* **mis-centering** — fixed-offset and Gaussian (Rayleigh) centering-error
  models, evaluated by real-space azimuthal averaging of the analytic Σ
  kernels (Gauss–Legendre; exact at R ≪ R_off, no Hankel/fftlog needed);
* **2-halo term** — Tinker10 halo bias × linear EH98 matter correlation,
  projected with the same hybrid line-of-sight grid as
  :meth:`hod_mod.observables.clustering.FullHaloModelPrediction.delta_sigma`;
* **strong lensing** (axisymmetric) — deflection α(θ) = θ κ̄(θ), Einstein
  radius (κ̄ = 1) via differentiable bisection + implicit-function Newton
  polish, magnification μ⁻¹ = (1−κ̄)(1+κ̄−2κ), tangential/radial critical
  curves, and composite lenses via bare κ/κ̄ callables;
* :class:`ClusterLensingPrediction` — the convenience pipeline binding a
  profile family, c(M, z) relation and mass definition.

The weak-lensing feature set mirrors the ``halo_lensing`` reference code of
Oguri et al. 2026 (PASJ 78, 416, arXiv:2512.13954); the strong-lensing block
is new.  All surface densities are **comoving** [Msun h/Mpc²] (multiply by
1e-12 for Msun h/pc²); radii are comoving [Mpc/h]; Σ_crit defaults to the
matching comoving convention so κ = Σ/Σ_crit is consistent.

References
----------
Oguri et al. 2026, PASJ 78, 416 (arXiv:2512.13954) — weak-lensing model
Wright & Brainerd 2000; Takada & Jain 2003; Baltz, Marshall & Oguri 2009
Bartelmann & Schneider 2001, Phys. Rep. 340, 291 — lensing formulae
Tinker et al. 2010, ApJ 724, 878 — halo bias
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.core.distances import comoving_distance
from hod_mod.core.power_spectrum import eisenstein_hu_pk_phys, rho_critical_0
from hod_mod.core.halo_mass_function import growth_factor, make_hmf
from hod_mod.core.concentration import ConcentrationModel
from hod_mod.core.halo_profiles import (
    nfw_sigma, nfw_mean_sigma, nfw_delta_sigma,
)
from hod_mod.core.lensing_profiles import (
    nfw_params_from_mass,
    tnfw_sigma, tnfw_mean_sigma, tnfw_delta_sigma,
    bmo_sigma, bmo_mean_sigma, bmo_delta_sigma,
)
from .clustering import _pk_to_xi

__all__ = [
    "sigma_crit", "inv_sigma_crit",
    "offset_sigma", "offset_sigma_gaussian", "mean_sigma_from_tab",
    "sigma_2h", "delta_sigma_2h",
    "solve_einstein_radius", "radial_critical_radius", "magnification",
    "tangential_shear",
    "ClusterLensingPrediction",
]

_RHO_CRIT0 = rho_critical_0()          # (Msun/h)/(Mpc/h)³
#: c²/(4πG) in Msun/Mpc (pinned to the astropy value used by the reference:
#: 1e6 · c²/(4πG)/(M_sun/pc), 2026-07-07).
_SIGMA_CRIT_C = 1.6629165401756012e18
_RAD2ARCSEC = 180.0 * 3600.0 / np.pi   # 206264.806...

# Gauss–Legendre rules, memoised at module level (numpy at setup time only —
# same convention as the Si/Ci tables in core.halo_profiles).
_GL_CACHE: dict[tuple, tuple] = {}


def _gl_rule(n: int, a: float, b: float):
    """Cached Gauss–Legendre nodes/weights on [a, b] as jnp arrays."""
    key = (n, a, b)
    rule = _GL_CACHE.get(key)
    if rule is None:
        x, w = np.polynomial.legendre.leggauss(n)
        x = 0.5 * (b - a) * (x + 1.0) + a
        w = 0.5 * (b - a) * w
        rule = (jnp.asarray(x), jnp.asarray(w))
        _GL_CACHE[key] = rule
    return rule


# ---------------------------------------------------------------------------
# Critical surface density
# ---------------------------------------------------------------------------

def sigma_crit(z_l, z_s, theta_cosmo: dict, comoving: bool = True):
    """Critical surface mass density Σ_crit [Msun h/Mpc²].

    .. math::

        \\Sigma_{\\rm crit} = \\frac{c^2}{4\\pi G}
        \\frac{D_s}{D_l D_{ls}}
        \\quad(\\times (1+z_l)^{-2}\\ \\text{if comoving})

    Flat geometry from :func:`hod_mod.core.distances.comoving_distance`
    (D_A(z1, z2) = (χ2 − χ1)/(1 + z2)).  With ``comoving=True`` (default)
    the result matches the comoving Σ of the profile modules, so
    κ = Σ/Σ_crit needs no further (1+z) factors.

    Parameters
    ----------
    z_l, z_s : scalar or 1d — lens and source redshifts (broadcast).
    theta_cosmo : dict — needs 'h', 'Omega_m' (optional 'w0', 'wa').
    comoving : bool — comoving (default) or proper/physical convention.

    Returns
    -------
    Σ_crit, shape broadcast(z_l, z_s); +inf where z_s ≤ z_l.
    """
    h = theta_cosmo["h"]
    om = theta_cosmo["Omega_m"]
    w0 = theta_cosmo.get("w0", -1.0)
    wa = theta_cosmo.get("wa", 0.0)

    z_l = jnp.asarray(z_l)
    z_s = jnp.asarray(z_s)
    chi_l = jnp.reshape(comoving_distance(z_l, h, om, w0, wa), jnp.shape(z_l))
    chi_s = jnp.reshape(comoving_distance(z_s, h, om, w0, wa), jnp.shape(z_s))

    d_l = chi_l / (1.0 + z_l)
    d_s = chi_s / (1.0 + z_s)
    d_ls = (chi_s - chi_l) / (1.0 + z_s)

    valid = chi_s > chi_l
    d_ls_safe = jnp.where(valid, d_ls, 1.0)
    # distances are in Mpc (no h): Σ[Msun h/Mpc²] = (C/h)·D_s/(D_l·D_ls)
    sc = _SIGMA_CRIT_C / h * d_s / (d_l * d_ls_safe)
    if comoving:
        sc = sc / (1.0 + z_l) ** 2
    return jnp.where(valid, sc, jnp.inf)


def inv_sigma_crit(z_l, z_s, theta_cosmo: dict, comoving: bool = True):
    """1/Σ_crit [Mpc²/(Msun h)]; exactly 0 where z_s ≤ z_l (unlensed)."""
    sc = sigma_crit(z_l, z_s, theta_cosmo, comoving)
    return jnp.where(jnp.isfinite(sc), 1.0 / sc, 0.0)


# ---------------------------------------------------------------------------
# Mis-centering (real-space azimuthal averaging)
# ---------------------------------------------------------------------------

def offset_sigma(R, r_off: float, sigma_fn, n_phi: int = 64):
    """Surface density around a center offset by r_off [Mpc/h].

    .. math::

        \\Sigma_{\\rm off}(R; R_{\\rm off}) = \\frac{1}{\\pi} \\int_0^\\pi
        \\Sigma\\!\\left(\\sqrt{R^2 + R_{\\rm off}^2
        + 2 R R_{\\rm off}\\cos\\phi}\\right) d\\phi

    evaluated with an ``n_phi``-node Gauss–Legendre rule on the analytic
    kernel — exact at R ≪ R_off and R ≫ R_off; the integrable log feature
    at R ≈ R_off is resolved to ~0.1% with the default 64 nodes.

    Parameters
    ----------
    R : (NR,) projected radii [Mpc/h]
    r_off : offset [Mpc/h]
    sigma_fn : callable, (N,) radii → (N,) Σ (any profile closure)
    """
    phi, w = _gl_rule(n_phi, 0.0, np.pi)
    R = jnp.atleast_1d(R)
    r_eval = jnp.sqrt(R[:, None] ** 2 + r_off**2
                      + 2.0 * R[:, None] * r_off * jnp.cos(phi)[None, :])
    sig = sigma_fn(r_eval.reshape(-1)).reshape(r_eval.shape)
    return jnp.sum(sig * w[None, :], axis=-1) / jnp.pi


def offset_sigma_gaussian(R, sigma_off: float, sigma_fn,
                          n_phi: int = 64, n_off: int = 48,
                          u_max: float = 5.0):
    """Mis-centered Σ averaged over a Gaussian (2D) centering-error PDF.

    The offset distribution is Rayleigh, P(R_off) = (R_off/σ²)
    exp(−R_off²/2σ²) (a 2D Gaussian of width ``sigma_off`` per axis),
    integrated with an ``n_off``-node Gauss–Legendre rule on
    [0, u_max·σ_off] and renormalized for the truncated tail.

    Equivalent to the reference pipeline's Fourier-space
    exp(−k²σ_off²/2) factor, evaluated in real space.
    """
    phi, w_phi = _gl_rule(n_phi, 0.0, np.pi)
    u, w_u = _gl_rule(n_off, 0.0, u_max)

    R = jnp.atleast_1d(R)
    r_off = sigma_off * u                                   # (Noff,)
    r_eval = jnp.sqrt(R[:, None, None] ** 2 + r_off[None, :, None] ** 2
                      + 2.0 * R[:, None, None] * r_off[None, :, None]
                      * jnp.cos(phi)[None, None, :])        # (NR, Noff, Nphi)
    sig = sigma_fn(r_eval.reshape(-1)).reshape(r_eval.shape)
    sig_off = jnp.sum(sig * w_phi[None, None, :], axis=-1) / jnp.pi  # (NR, Noff)

    p_u = u * jnp.exp(-0.5 * u**2)                          # Rayleigh in u
    norm = 1.0 - jnp.exp(-0.5 * u_max**2)
    return jnp.sum(sig_off * (w_u * p_u)[None, :], axis=-1) / norm


def mean_sigma_from_tab(R, R_tab, sigma_tab):
    """Σ̄(<R) from a tabulated Σ(R_tab) by cumulative trapezoid.

    Includes the inner-disk closure term Σ(R₀)·R₀²/2 (Σ assumed flat inside
    the first node — exact for mis-centered/2-halo profiles, a ~0.1%
    underestimate of the log-divergent centered-NFW core when
    R₀ = 1e-3 Mpc/h).

    Parameters
    ----------
    R : output radii [Mpc/h]
    R_tab : (Nt,) increasing tabulation radii
    sigma_tab : (Nt,) Σ values on R_tab
    """
    integrand = R_tab * sigma_tab
    dR = jnp.diff(R_tab)
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mid * dR)])
    cum = cum + 0.5 * sigma_tab[0] * R_tab[0] ** 2   # inner-disk closure
    sbar_tab = 2.0 * cum / R_tab**2
    return jnp.interp(jnp.asarray(R), R_tab, sbar_tab)


# ---------------------------------------------------------------------------
# 2-halo term
# ---------------------------------------------------------------------------

def _pk_lin_tab(theta_cosmo: dict, n_k: int = 512):
    """(log k, log P_lin(k, z=0)) table [(Mpc/h)³], σ8-rescaled if requested.

    Uses :func:`eisenstein_hu_pk_phys` (amplitude from ln10^{10}A_s); if
    ``theta_cosmo`` carries 'sigma8', the amplitude is rescaled so the
    top-hat σ(8 Mpc/h) matches it (same convention as
    :meth:`HaloMassFunction.sigma`).
    """
    k = jnp.logspace(-4, 3, n_k)
    pk = eisenstein_hu_pk_phys(k, theta_cosmo)
    if "sigma8" in theta_cosmo:
        x = k * 8.0
        w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
        s2_8 = jnp.trapezoid(pk * w**2 * k**2, k) / (2.0 * jnp.pi**2)
        pk = pk * theta_cosmo["sigma8"] ** 2 / s2_8
    return jnp.log(k), jnp.log(pk)


def sigma_2h(R, m_h, z: float, theta_cosmo: dict, hmf,
             chi_max: float = 300.0, n_chi: int = 512):
    """2-halo surface density Σ_2h(R) [Msun h/Mpc²].

    .. math::

        \\Sigma_{2h}(R) = \\bar{\\rho}_m\\, b(M, z)\\, D^2(z)
        \\int_{-\\chi_{max}}^{\\chi_{max}}
        \\xi_{\\rm lin}\\!\\left(\\sqrt{R^2 + \\chi^2}, z{=}0\\right) d\\chi

    with Tinker10 bias from ``hmf.bias`` and ξ_lin from the Ogata-j₀
    transform of the EH98 spectrum (:func:`_pk_lin_tab`).  Reuses the
    log-linear hybrid χ grid validated in
    ``FullHaloModelPrediction._delta_sigma_from_pgm``.
    """
    log_k, log_pk = _pk_lin_tab(theta_cosmo)
    r_tab = jnp.logspace(-2, 2.5, 512)
    xi_tab = _pk_to_xi(r_tab, log_k, log_pk)

    growth = growth_factor(z, theta_cosmo)
    bias = jnp.squeeze(hmf.bias(jnp.atleast_1d(m_h), z, theta_cosmo))
    rho_m = theta_cosmo["Omega_m"] * _RHO_CRIT0

    chi_log = jnp.logspace(-2, jnp.log10(float(chi_max)), n_chi // 2)
    chi_lin = jnp.linspace(1.0, float(chi_max), n_chi // 2)
    chi_grid = jnp.sort(jnp.concatenate([chi_log, chi_lin]))

    def _one(R_i):
        r_grid = jnp.sqrt(R_i**2 + chi_grid**2)
        xi_i = jnp.interp(r_grid, r_tab, xi_tab)
        return 2.0 * jnp.trapezoid(xi_i, chi_grid)

    wp = jax.vmap(_one)(jnp.atleast_1d(R))
    return rho_m * bias * growth**2 * wp


def delta_sigma_2h(R, m_h, z: float, theta_cosmo: dict, hmf,
                   chi_max: float = 300.0, n_chi: int = 512,
                   n_R_tab: int = 256):
    """2-halo excess surface density ΔΣ_2h(R) = Σ̄_2h(<R) − Σ_2h(R)
    [Msun h/Mpc²]."""
    R_tab = jnp.logspace(-3, 2, n_R_tab)
    sig_tab = sigma_2h(R_tab, m_h, z, theta_cosmo, hmf, chi_max, n_chi)
    sbar = mean_sigma_from_tab(R_tab, R_tab, sig_tab)
    ds_tab = sbar - sig_tab
    return jnp.interp(jnp.asarray(R), R_tab, ds_tab)


# ---------------------------------------------------------------------------
# Strong lensing (axisymmetric)
# ---------------------------------------------------------------------------

def magnification(kappa, kappa_bar):
    """Magnification μ = 1/[(1 − κ̄)(1 + κ̄ − 2κ)] of an axisymmetric lens.

    The tangential eigenvalue (1 − κ̄) vanishes on the Einstein ring, the
    radial one (1 + κ̄ − 2κ) on the radial critical curve.
    """
    return 1.0 / ((1.0 - kappa_bar) * (1.0 + kappa_bar - 2.0 * kappa))


def tangential_shear(kappa, kappa_bar):
    """Tangential shear γ_t = κ̄ − κ (axisymmetric lens)."""
    return kappa_bar - kappa


def solve_einstein_radius(kappa_bar_fn, log10_r_lo: float = -4.0,
                          log10_r_hi: float = 1.5, n_iter: int = 80):
    """Einstein radius R_E [Mpc/h]: the root of κ̄(R_E) = 1.

    κ̄ is monotone decreasing, so a fixed ``n_iter``-step bisection in
    log10(R) (pattern of ``gas.conversions._m200_to_m500c_jax``) brackets
    the root; one Newton step through ``stop_gradient`` then makes the
    result differentiable with the exact implicit-function-theorem
    gradient dR_E/dp = −(∂κ̄/∂p)/(∂κ̄/∂R) w.r.t. any traced parameter of
    ``kappa_bar_fn``.

    Parameters
    ----------
    kappa_bar_fn : callable, scalar R [Mpc/h] → scalar κ̄(R).
        Pass a sum of component κ̄'s for a composite lens.

    Returns
    -------
    R_E [Mpc/h] (scalar); NaN when κ̄ < 1 everywhere in the bracket
    (no strong lensing).
    """
    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        above = kappa_bar_fn(10.0**mid) > 1.0
        return (jnp.where(above, mid, lo), jnp.where(above, hi, mid))

    lo, hi = jax.lax.fori_loop(0, n_iter, body,
                               (jnp.asarray(log10_r_lo, dtype=jnp.result_type(float)),
                                jnp.asarray(log10_r_hi, dtype=jnp.result_type(float))))
    r0 = jax.lax.stop_gradient(10.0 ** (0.5 * (lo + hi)))
    kb, dkb = jax.value_and_grad(kappa_bar_fn)(r0)
    r_e = r0 - (kb - 1.0) / dkb
    solvable = kappa_bar_fn(10.0**log10_r_lo) > 1.0
    return jnp.where(solvable, r_e, jnp.nan)


def radial_critical_radius(kappa_fn, kappa_bar_fn, log10_r_lo: float = -4.0,
                           log10_r_hi: float = 1.0, n_grid: int = 512,
                           n_iter: int = 60):
    """Radial critical radius [Mpc/h]: root of 1 + κ̄(R) − 2κ(R) = 0.

    The radial eigenvalue is negative in the deep interior of a
    super-critical lens and positive outside; the first sign change on an
    ``n_grid``-point log grid is bracketed, bisected ``n_iter`` times, and
    polished with the same stop-gradient Newton step as
    :func:`solve_einstein_radius`.  Returns NaN when no sign change exists
    (sub-critical lens).
    """
    def d_fn(r):
        return 1.0 + kappa_bar_fn(r) - 2.0 * kappa_fn(r)

    t_grid = jnp.linspace(log10_r_lo, log10_r_hi, n_grid)
    d_grid = jax.vmap(lambda t: d_fn(10.0**t))(t_grid)
    flip = (d_grid[:-1] < 0.0) & (d_grid[1:] >= 0.0)
    found = jnp.any(flip)
    idx = jnp.argmax(flip)

    lo0 = t_grid[idx]
    hi0 = t_grid[idx + 1]

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        below = d_fn(10.0**mid) < 0.0
        return (jnp.where(below, mid, lo), jnp.where(below, hi, mid))

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo0, hi0))
    r0 = jax.lax.stop_gradient(10.0 ** (0.5 * (lo + hi)))
    d0, dd0 = jax.value_and_grad(d_fn)(r0)
    r_rad = r0 - d0 / dd0
    return jnp.where(found, r_rad, jnp.nan)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class ClusterLensingPrediction:
    """Weak- and strong-lensing predictions for a single halo.

    Binds a profile family, a JAX-native concentration–mass relation and a
    mass definition; all methods are differentiable w.r.t. mass and
    cosmology.  Mirrors the model of the ``halo_lensing`` reference code
    (Oguri et al. 2026): total ΔΣ = f_cen·ΔΣ_1h + (1−f_cen)·ΔΣ_off + ΔΣ_2h.

    Parameters
    ----------
    profile : 'bmo' (default) | 'tnfw' | 'nfw'
        1-halo profile family.  'bmo' smoothly truncates at
        r_t = tau_v·r_Δ (τ = tau_v·c); 'tnfw' sharply truncates at the
        same radius (reference TJ convention: ``tau_v=1``, truncation at
        the overdensity radius); 'nfw' is the infinite profile.
    mdef : '200m' | '200c' | 'vir'
        Mass definition (Bryan & Norman 1998 for 'vir').
    cm_relation : str
        Any JAX-native :class:`ConcentrationModel` key ('duffy08',
        'dutton14', 'klypin16', 'bhattacharya13', 'diemer15').  The
        reference's colossus-only 'diemer19' is intentionally unavailable.
    tau_v : float
        Truncation radius in units of r_Δ (default 2.5, reference value).
    hmf : HaloMassFunction or None
        Used for the 2-halo bias (and σ(M)-based c(M) relations); default
        builds ``make_hmf('tinker08')`` on the EH98 spectrum.
    n_phi, n_off : int
        Gauss–Legendre nodes for the mis-centering integrals.

    Notes
    -----
    Surface densities are comoving [Msun h/Mpc²]; radii comoving [Mpc/h];
    angles in arcsec.  Methods take scalar ``m_h``/``z``/``z_s`` (vmap for
    grids).  Enable ``jax_enable_x64`` in entry-point scripts for
    sub-percent accuracy at R/r_s < 0.05.
    """

    _R_TAB = jnp.logspace(-3, 2, 256)   # internal Σ̄ tabulation grid [Mpc/h]

    def __init__(self, profile: str = "bmo", mdef: str = "vir",
                 cm_relation: str = "duffy08", tau_v: float = 2.5,
                 hmf=None, n_phi: int = 64, n_off: int = 48):
        if profile not in ("nfw", "tnfw", "bmo"):
            raise ValueError(f"Unknown profile '{profile}' (nfw|tnfw|bmo)")
        self.profile = profile
        self.mdef = mdef
        self.tau_v = float(tau_v)
        self.n_phi = int(n_phi)
        self.n_off = int(n_off)
        if hmf is None:
            hmf = make_hmf("tinker08",
                           pk_func=lambda k, z, theta: eisenstein_hu_pk_phys(k, theta))
        self.hmf = hmf
        needs_sigma = cm_relation in ("bhattacharya13", "diemer15")
        self._cm = ConcentrationModel(cm_relation, mdef=mdef,
                                      hmf=hmf if needs_sigma else None)

    # -- profile plumbing ---------------------------------------------------

    def concentration(self, m_h, z: float, theta_cosmo: dict):
        """c(M, z) from the chosen relation."""
        return self._cm.concentration(m_h, z, theta_cosmo)

    def _params(self, m_h, z: float, theta_cosmo: dict):
        """(ρ_s, r_s, shape) for the bound profile family."""
        c = self.concentration(jnp.asarray(m_h), z, theta_cosmo)
        rho_s, r_s, _ = nfw_params_from_mass(m_h, c, z, theta_cosmo, self.mdef)
        shape = self.tau_v * c   # τ for bmo, c_t for tnfw
        return rho_s, r_s, shape

    def _sigma_fns(self, m_h, z: float, theta_cosmo: dict):
        """(Σ, Σ̄, ΔΣ) closures over 1d radius arrays for the bound profile."""
        rho_s, r_s, shape = self._params(m_h, z, theta_cosmo)
        if self.profile == "nfw":
            return (lambda r: nfw_sigma(r, rho_s, r_s),
                    lambda r: nfw_mean_sigma(r, rho_s, r_s),
                    lambda r: nfw_delta_sigma(r, rho_s, r_s))
        if self.profile == "tnfw":
            return (lambda r: tnfw_sigma(r, rho_s, r_s, shape),
                    lambda r: tnfw_mean_sigma(r, rho_s, r_s, shape),
                    lambda r: tnfw_delta_sigma(r, rho_s, r_s, shape))
        return (lambda r: bmo_sigma(r, rho_s, r_s, shape),
                lambda r: bmo_mean_sigma(r, rho_s, r_s, shape),
                lambda r: bmo_delta_sigma(r, rho_s, r_s, shape))

    # -- weak lensing: 1-halo -------------------------------------------------

    def sigma_1h(self, R, m_h, z: float, theta_cosmo: dict):
        """Centered 1-halo Σ(R) [Msun h/Mpc²]."""
        sig, _, _ = self._sigma_fns(m_h, z, theta_cosmo)
        return sig(jnp.atleast_1d(R))

    def mean_sigma_1h(self, R, m_h, z: float, theta_cosmo: dict):
        """Centered 1-halo Σ̄(<R) [Msun h/Mpc²]."""
        _, sbar, _ = self._sigma_fns(m_h, z, theta_cosmo)
        return sbar(jnp.atleast_1d(R))

    def delta_sigma_1h(self, R, m_h, z: float, theta_cosmo: dict):
        """Centered 1-halo ΔΣ(R) [Msun h/Mpc²] (analytic)."""
        _, _, ds = self._sigma_fns(m_h, z, theta_cosmo)
        return ds(jnp.atleast_1d(R))

    # -- weak lensing: mis-centered 1-halo -----------------------------------

    def _sigma_off_fn(self, m_h, z, theta_cosmo, r_off, sigma_off):
        sig, _, _ = self._sigma_fns(m_h, z, theta_cosmo)
        if (r_off is None) == (sigma_off is None):
            raise ValueError("give exactly one of r_off= or sigma_off=")
        if r_off is not None:
            return lambda r: offset_sigma(r, r_off, sig, self.n_phi)
        return lambda r: offset_sigma_gaussian(r, sigma_off, sig,
                                               self.n_phi, self.n_off)

    def sigma_off(self, R, m_h, z: float, theta_cosmo: dict, *,
                  r_off=None, sigma_off=None):
        """Mis-centered 1-halo Σ(R) [Msun h/Mpc²].

        Give either ``r_off`` (fixed offset [Mpc/h]) or ``sigma_off``
        (Gaussian centering-error width [Mpc/h]).
        """
        fn = self._sigma_off_fn(m_h, z, theta_cosmo, r_off, sigma_off)
        return fn(jnp.atleast_1d(R))

    def delta_sigma_off(self, R, m_h, z: float, theta_cosmo: dict, *,
                        r_off=None, sigma_off=None):
        """Mis-centered 1-halo ΔΣ(R) [Msun h/Mpc²] (via internal Σ̄ table)."""
        fn = self._sigma_off_fn(m_h, z, theta_cosmo, r_off, sigma_off)
        sig_tab = fn(self._R_TAB)
        sbar = mean_sigma_from_tab(R, self._R_TAB, sig_tab)
        return sbar - fn(jnp.atleast_1d(R))

    # -- weak lensing: 2-halo -------------------------------------------------

    def sigma_2h(self, R, m_h, z: float, theta_cosmo: dict):
        """2-halo Σ(R) [Msun h/Mpc²] (Tinker10 bias × linear ξ)."""
        return sigma_2h(R, m_h, z, theta_cosmo, self.hmf)

    def delta_sigma_2h(self, R, m_h, z: float, theta_cosmo: dict):
        """2-halo ΔΣ(R) [Msun h/Mpc²]."""
        return delta_sigma_2h(R, m_h, z, theta_cosmo, self.hmf)

    def delta_sigma_total(self, R, m_h, z: float, theta_cosmo: dict,
                          f_cen: float = 1.0, sigma_off: float = 0.1,
                          two_halo: bool = True):
        """Total ΔΣ = f_cen ΔΣ_1h + (1−f_cen) ΔΣ_off(Gaussian) + ΔΣ_2h.

        The reference model (Oguri et al. 2026): a fraction f_cen of halos
        is perfectly centered, the rest mis-centered with a Gaussian
        (Rayleigh-modulus) PDF of width ``sigma_off`` [Mpc/h].
        """
        ds = f_cen * self.delta_sigma_1h(R, m_h, z, theta_cosmo)
        if f_cen < 1.0:
            ds = ds + (1.0 - f_cen) * self.delta_sigma_off(
                R, m_h, z, theta_cosmo, sigma_off=sigma_off)
        if two_halo:
            ds = ds + self.delta_sigma_2h(R, m_h, z, theta_cosmo)
        return ds

    # -- convergence and shear ------------------------------------------------

    def kappa(self, R, m_h, z: float, z_s, theta_cosmo: dict,
              two_halo: bool = False):
        """Convergence κ(R) = Σ/Σ_crit (comoving convention throughout)."""
        sig = self.sigma_1h(R, m_h, z, theta_cosmo)
        if two_halo:
            sig = sig + self.sigma_2h(R, m_h, z, theta_cosmo)
        return sig * inv_sigma_crit(z, z_s, theta_cosmo)

    def mean_kappa(self, R, m_h, z: float, z_s, theta_cosmo: dict):
        """Mean convergence κ̄(<R) = Σ̄/Σ_crit."""
        return (self.mean_sigma_1h(R, m_h, z, theta_cosmo)
                * inv_sigma_crit(z, z_s, theta_cosmo))

    def gamma_t(self, R, m_h, z: float, z_s, theta_cosmo: dict,
                f_cen: float = 1.0, sigma_off: float = 0.1,
                two_halo: bool = True):
        """Tangential shear γ_t(R) = ΔΣ_total/Σ_crit."""
        ds = self.delta_sigma_total(R, m_h, z, theta_cosmo,
                                    f_cen, sigma_off, two_halo)
        return ds * inv_sigma_crit(z, z_s, theta_cosmo)

    # -- strong lensing ---------------------------------------------------------

    def _kappa_fns(self, m_h, z: float, z_s, theta_cosmo: dict):
        """Scalar (κ(R), κ̄(R)) closures for the strong-lensing solvers."""
        sig, sbar, _ = self._sigma_fns(m_h, z, theta_cosmo)
        isc = jnp.squeeze(inv_sigma_crit(z, z_s, theta_cosmo))
        kap = lambda r: jnp.squeeze(sig(jnp.atleast_1d(r))) * isc
        kbar = lambda r: jnp.squeeze(sbar(jnp.atleast_1d(r))) * isc
        return kap, kbar

    def arcsec_per_mpc(self, z: float, theta_cosmo: dict):
        """Angular scale: arcsec per comoving Mpc/h at the lens redshift."""
        h = theta_cosmo["h"]
        om = theta_cosmo["Omega_m"]
        chi_l = jnp.squeeze(comoving_distance(jnp.asarray(z), h, om,
                                              theta_cosmo.get("w0", -1.0),
                                              theta_cosmo.get("wa", 0.0)))
        return _RAD2ARCSEC / (chi_l * h)

    def einstein_radius(self, m_h, z: float, z_s, theta_cosmo: dict):
        """Einstein radius: (R_E [Mpc/h comoving], θ_E [arcsec]).

        NaN when the halo is sub-critical (κ̄ < 1 everywhere).
        """
        _, kbar = self._kappa_fns(m_h, z, z_s, theta_cosmo)
        r_e = solve_einstein_radius(kbar)
        return r_e, r_e * self.arcsec_per_mpc(z, theta_cosmo)

    def deflection(self, R, m_h, z: float, z_s, theta_cosmo: dict):
        """Reduced deflection angle α(θ) = θ κ̄(θ) [arcsec] at θ = R/χ_l."""
        kbar = self.mean_kappa(R, m_h, z, z_s, theta_cosmo)
        theta_arcsec = jnp.atleast_1d(R) * self.arcsec_per_mpc(z, theta_cosmo)
        return theta_arcsec * kbar

    def magnification(self, R, m_h, z: float, z_s, theta_cosmo: dict):
        """Magnification μ(R) of an axisymmetric lens (signed)."""
        kap = self.kappa(R, m_h, z, z_s, theta_cosmo)
        kbar = self.mean_kappa(R, m_h, z, z_s, theta_cosmo)
        return magnification(kap, kbar)

    def critical_curves(self, m_h, z: float, z_s, theta_cosmo: dict):
        """(R_tangential, R_radial) [Mpc/h] — Einstein ring and radial
        critical curve; NaN where the lens is sub-critical."""
        kap, kbar = self._kappa_fns(m_h, z, z_s, theta_cosmo)
        r_t = solve_einstein_radius(kbar)
        r_r = radial_critical_radius(kap, kbar)
        return r_t, r_r
