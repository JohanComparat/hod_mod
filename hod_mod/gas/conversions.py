"""Shared constants, gNFW helpers, m200->m500c, GL profile FT."""
import numpy as np
import jax
import jax.numpy as jnp
from hod_mod.core.power_spectrum import eisenstein_hu_pk


def _eh_pk_3arg(k, z, theta):
    """Wrap eisenstein_hu_pk to match HaloMassFunction's (k, z, theta) signature."""
    return eisenstein_hu_pk(k, theta)


_LEGGAUSS_CACHE: dict[int, tuple] = {}


def _leggauss_cached(n_gl: int):
    """Gauss-Legendre nodes/weights, memoised (the leggauss call is not free and
    ``_profile_uk_gl`` is invoked per-z / per-mass-grid build)."""
    rule = _LEGGAUSS_CACHE.get(n_gl)
    if rule is None:
        rule = np.polynomial.legendre.leggauss(n_gl)
        _LEGGAUSS_CACHE[n_gl] = rule
    return rule

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
    m200 = jnp.asarray(m200_arr, dtype=float)
    c200 = jnp.asarray(c200_arr, dtype=float)
    r200 = jnp.asarray(r200_arr, dtype=float)
    return _m200_to_m500c_jax(m200, c200, r200, float(rho_crit_z))


@jax.jit
def _m200_to_m500c_jax(m200, c200, r200, rho_crit_z):
    """Vectorised JAX NFW-bisection for M₂₀₀→M₅₀₀c (replaces the per-halo brentq loop).

    Solves ``f(x) = M(<x·r200) − (4π/3)·500·ρ_crit·(x·r200)³ = 0`` for ``x = r/r200``.
    ``f`` decreases monotonically from + to − on ``x ∈ [0.10, 0.99]``, so a fixed
    60-step bisection (interval < 0.9/2⁶⁰ ≪ the former xtol=1e-8) brackets the root
    over all haloes at once — jittable and differentiable, no Python loop.
    """
    g_nfw = lambda x: jnp.log1p(x) - x / (1.0 + x)   # noqa: E731
    g200 = g_nfw(c200)
    coef = (4.0 / 3.0) * jnp.pi * 500.0 * rho_crit_z

    def f(x):
        m_enc = m200 * g_nfw(c200 * x) / g200
        m_sph = coef * (x * r200) ** 3
        return m_enc - m_sph

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        right = f(mid) > 0.0            # f decreasing → root lies to the right
        return (jnp.where(right, mid, lo), jnp.where(right, hi, mid))

    lo = jnp.full_like(m200, 0.10)
    hi = jnp.full_like(m200, 0.99)
    lo, hi = jax.lax.fori_loop(0, 60, body, (lo, hi))
    r500c = 0.5 * (lo + hi) * r200
    m500c = coef * r500c ** 3
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

    x_gl, w_gl = _leggauss_cached(n_gl)

    # Map GL nodes from [-1,1] to [0, r_max[i]]
    half    = r_max / 2.0
    r_nodes = np.outer(half, x_gl + 1.0)            # (NM, n_gl) [same units as r_max]

    # Profile values at quadrature nodes, pre-folded with the GL weight r²·w:
    #   A[m,g] = (r_max/2)·w_g · f(r_mg) · r_mg²
    A = (half[:, None] * w_gl[None, :]) * integrand_fn(r_nodes) * r_nodes**2   # (NM, n_gl)

    # j₀(K) = sin(K)/K = sinc(K/π) (branchless; np.sinc handles K→0 → 1).
    # einsum contracts the GL axis WITHOUT materialising the (Nk, NM, n_gl)
    # weighted-product cube — only the j₀ cube is built (the giant product
    # temporary is what made the dense np.sum memory-bound at NM=n_k=512).
    K  = k[:, None, None] * r_nodes[None, :, :]     # (Nk, NM, n_gl)
    j0 = np.sinc(K / np.pi)                          # (Nk, NM, n_gl)
    result = np.einsum('kmg,mg->km', j0, A, optimize=True)   # (Nk, NM)
    return 4.0 * np.pi * result   # (Nk, NM)


def _profile_uk_gl_bands(
    k_arr: np.ndarray,
    r_max_arr: np.ndarray,
    integrand_bands_fn,
    n_gl: int = 200,
) -> np.ndarray:
    """Batched :func:`_profile_uk_gl` over an energy-band axis.

    Identical quadrature, but the integrand returns ``(Nb, NM, n_gl)`` (one slice
    per band) and the ``j₀`` cube — the only expensive piece — is built ONCE and
    reused across bands.  Used for the 15 narrow X-ray bands, whose emissivities
    ``ε_b = n_e²·Λ_b(T,Z)`` share the same n_e/T/Z profiles and FT geometry, so all
    bands cost ≈ a single FT instead of Nb of them.

    Parameters
    ----------
    k_arr : (Nk,)
    r_max_arr : (NM,)
    integrand_bands_fn : callable(r_nodes) → (Nb, NM, n_gl)
    n_gl : number of GL nodes

    Returns
    -------
    uk : (Nb, Nk, NM) = 4π ∫ f_b(r) j₀(kr) r² dr
    """
    k      = np.asarray(k_arr,     dtype=float)
    r_max  = np.asarray(r_max_arr, dtype=float)

    x_gl, w_gl = _leggauss_cached(n_gl)
    half    = r_max / 2.0
    r_nodes = np.outer(half, x_gl + 1.0)                       # (NM, n_gl)

    f_bands = np.asarray(integrand_bands_fn(r_nodes))          # (Nb, NM, n_gl)
    A = (half[:, None] * w_gl[None, :]) * f_bands * r_nodes[None, :, :]**2   # (Nb, NM, n_gl)

    K  = k[:, None, None] * r_nodes[None, :, :]                # (Nk, NM, n_gl)
    j0 = np.sinc(K / np.pi)                                    # (Nk, NM, n_gl) — built once
    result = np.einsum('kmg,bmg->bkm', j0, A, optimize=True)   # (Nb, Nk, NM)
    return 4.0 * np.pi * result
