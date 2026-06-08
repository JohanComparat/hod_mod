"""Galaxy two-point statistics and weak lensing observables from HOD models.

Four independent prediction paths:
- **ξ_gg(r)** — 3D correlation function via the Ogata (2005) j₀ Hankel transform
- **wp(rp)**  — projected correlation function, 2 ∫₀^{π_max} ξ(√(rp²+π²)) dπ
- **w(θ)**   — angular clustering via the Limber approximation (Coupon+2015 App. B)
- **ΔΣ(R)**  — excess projected surface mass density from the g-m cross spectrum

All four use the large-scale (2-halo) approximation with the **linear** matter power
spectrum following More et al. (2015):

    P_gg(k) = b_eff² P_lin(k),  P_gm(k) = b_eff P_lin(k)

Using P_lin (not P_nl) for the 2-halo term avoids double-counting non-linear
clustering at the 1h/2h transition; see More+2015 Section 3.1 and
van den Bosch+2013 for the full Q(k|M1,M2,z) treatment.

``HODClusteringPrediction`` accepts any of the six HOD/CSMF model objects defined in
``hod_mod.galaxies.hod`` (duck-typed on ``._integrate``).

The ``projected_correlation_function`` function measures wp from a galaxy catalogue
using corrfunc (pair counting), and ``HODProjectedCorrelation`` is retained for
backward compatibility.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from hod_mod.cosmology.power_spectrum import rho_critical_0


# ---------------------------------------------------------------------------
# Ogata (2005) j₀ Hankel transform: P(k) → ξ(r)
#
# Implements the double-exponential (tanh-sinh) quadrature of Ogata 2005
# (JSIAM Letters 5, 547) for the integral
#
#   ξ(r) = (1/2π²) ∫₀^∞ k² P(k) sin(kr)/(kr) dk
#         = (h/2π²r³) Σ_n [π sin(x_n) ψ'(hn) x_n] P(x_n/r)
#
# where x_n = π n tanh(π/2 sinh(hn)) are the DE quadrature nodes.
# ---------------------------------------------------------------------------

def _ogata_table(N: int = 512, h: float = 0.005):
    """Compute Ogata 2005 DE quadrature nodes x_n and combined weights.

    Returns (x, w) where w_n = π sin(x_n) ψ'(hn) x_n so that
    ξ(r) = h * Σ w_n P(x_n/r) / (2π²r³).
    """
    n = np.arange(1, N + 1, dtype=np.float64)
    t = h * n
    s = np.pi * np.sinh(t)
    x = np.pi * t * np.tanh(s / 2)   # nodes x_n = π ψ(hn), ψ(u)=u tanh(πsinh(u)/2)
    denom = 1.0 + np.cosh(s)
    dpsi = np.where(denom > 1e-100,
                    (np.pi * t * np.cosh(t) + np.sinh(s)) / denom, 0.0)
    w = np.pi * np.sin(x) * dpsi * x
    return x, w


_OG_N = 512
_OG_H = 0.005
_OG_X_NP, _OG_W_NP = _ogata_table(_OG_N, _OG_H)
_OG_X = jnp.array(_OG_X_NP)
_OG_W = jnp.array(_OG_W_NP)


@jax.jit
def _pk_to_xi(
    r: jnp.ndarray,
    log_k: jnp.ndarray,
    log_pk: jnp.ndarray,
) -> jnp.ndarray:
    """Ogata 2005 j₀ Hankel transform: P(k) → ξ(r).

    .. math::
        \\xi(r) = \\frac{1}{2\\pi^2} \\int_0^\\infty k^2 P(k)
                  \\frac{\\sin(kr)}{kr}\\,\\mathrm{d}k

    P(k) is supplied as a log-log interpolation table (log_k, log_pk).
    Values outside the supplied k range are clamped to the boundary (nearest
    P(k) value), so accuracy degrades for r ≲ 0.1 Mpc/h if P(k) is only
    tabulated to k ≲ 100 h/Mpc.

    Parameters
    ----------
    r : [Mpc/h], shape (Nr,)
    log_k : log(k [h/Mpc]) nodes, shape (Nk,), strictly increasing
    log_pk : log(P(k) [(Mpc/h)³]) values, shape (Nk,)

    Returns
    -------
    xi : ξ(r), unitless, shape (Nr,)

    Accuracy
    --------
    Power-law P(k) ∝ k^n → ξ(r) ∝ r^(-n-3) slope recovered to < 5% for
    n ∈ [-2, -0.5], r ∈ [0.1, 10] Mpc/h (512-node Ogata table, 2026-04-23).
    Absolute accuracy degrades for r ≲ 0.1 Mpc/h if P(k) is not tabulated
    to k ≳ 100 h/Mpc.

    Timing
    ------
    ~ 590 µs / call  (JIT-compiled, N=50 radii, CPU x86-64, 2026-04-23).
    """
    def _one(r_i):
        log_k_n = jnp.log(_OG_X) - jnp.log(r_i)
        log_pk_n = jnp.interp(log_k_n, log_k, log_pk,
                               left=log_pk[0], right=log_pk[-1])
        return _OG_H * jnp.sum(_OG_W * jnp.exp(log_pk_n)) / (2.0 * jnp.pi**2 * r_i**3)

    return jax.vmap(_one)(r)


# ---------------------------------------------------------------------------
# Mean comoving matter density at z=0  [in (Msun/h) / (Mpc/h)³]
# ρ̄_m = Ω_m × ρ_crit,0,  where ρ_crit,0 = 2.775e11 (Msun/h)/(Mpc/h)³
# ---------------------------------------------------------------------------

_RHO_CRIT0 = rho_critical_0()  # (Msun/h)/(Mpc/h)³


def _rho_m(theta_cosmo: dict):
    """Mean matter density ρ̄_m [(Msun/h)/(Mpc/h)³] from cosmology dict."""
    return _RHO_CRIT0 * theta_cosmo["Omega_m"]


# ---------------------------------------------------------------------------
# Flat ΛCDM geometry helpers (numpy-level, used at the non-JAX boundary)
# ---------------------------------------------------------------------------

def _hubble_E(z: np.ndarray, Omega_m: float) -> np.ndarray:
    """Dimensionless Hubble function E(z) = H(z)/H0 for flat ΛCDM.

    .. math::
        E(z) = \\sqrt{\\Omega_m(1+z)^3 + (1-\\Omega_m)}
    """
    return np.sqrt(Omega_m * (1.0 + z) ** 3 + (1.0 - Omega_m))


def _comoving_dist_h(z_arr: np.ndarray, theta_cosmo: dict) -> np.ndarray:
    """Comoving distance χ(z) [Mpc/h] for flat ΛCDM via trapezoidal quadrature.

    .. math::
        \\chi(z) = \\frac{c}{H_0}\\int_0^z \\frac{\\mathrm{d}z'}{E(z')}
                 = 2997.92\\int_0^z \\frac{\\mathrm{d}z'}{E(z')}\\;[\\mathrm{Mpc}/h]

    Parameters
    ----------
    z_arr : redshifts at which χ is evaluated, shape (Nz,)
    theta_cosmo : cosmology dict (needs ``'Omega_m'``)

    Returns
    -------
    chi : [Mpc/h], shape (Nz,)
    """
    Omega_m = float(theta_cosmo["Omega_m"])
    c_over_H0 = 2997.92  # Mpc/h  (c / [100 km/s/Mpc])
    z_arr = np.asarray(z_arr, dtype=float)
    n_fine = max(2000, 20 * len(z_arr))
    z_fine = np.linspace(0.0, float(z_arr.max()), n_fine)
    inv_E = 1.0 / _hubble_E(z_fine, Omega_m)
    chi_fine = np.concatenate([
        [0.0],
        np.cumsum(c_over_H0 * 0.5 * (inv_E[:-1] + inv_E[1:]) * np.diff(z_fine)),
    ])
    return np.interp(z_arr, z_fine, chi_fine)


# ---------------------------------------------------------------------------
# HODClusteringPrediction — generic predictor for all HOD models
# ---------------------------------------------------------------------------

class HODClusteringPrediction:
    """Compute ξ_gg(r), wp(rp), ΔΣ(R) for any of the six HOD/CSMF models.

    Uses the 2-halo approximation with the **linear** matter power spectrum
    following More et al. (2015) Section 3.1:

    .. math::

        P_{gg}(k) = b_{\\rm eff}^2\\, P_{\\rm lin}(k), \\quad
        P_{gm}(k) = b_{\\rm eff}\\, P_{\\rm lin}(k)

    Using P_lin for the 2-halo term is the correct More+2015 prescription;
    mixing P_nl with a 1-halo term double-counts non-linear power at the
    transition scale.

    ``hod`` must expose ``._integrate(z, theta_cosmo, hod_params)`` returning
    ``(n_gal, b_eff, m_eff)`` — satisfied by all six model classes in
    ``hod_mod.galaxies.hod``.

    Parameters
    ----------
    pk_lin : LinearPowerSpectrum
        Linear P(k) backend (CAMB).
    hod : HOD model object
        Any of HODModel, MoreHODModel, Guo18ICSMFModel, Guo19ICSMFModel,
        Zacharegkas25HODModel, VanUitert16CSMFModel.
    k_min, k_max : float [h/Mpc]
        Wavenumber range for the P(k) tabulation used in transforms.
    n_k : int
        Number of k points in the tabulation grid.
    """

    def __init__(
        self,
        pk_lin,
        hod,
        k_min: float = 1e-4,
        k_max: float = 20.0,
        n_k: int = 512,
    ):
        """
        Parameters
        ----------
        pk_lin : LinearPowerSpectrum
            Linear P(k) backend (CAMB).
        hod : HOD model object
        k_min, k_max : float [h/Mpc]
            Wavenumber range for the P(k) tabulation.  Must span at least
            k=[1e-4, 200] h/Mpc for reliable predictions at r > 0.02 Mpc/h;
            at r = 0.02 Mpc/h the Ogata quadrature first node accesses
            k ~ π/r ≈ 157 h/Mpc, so clamping at k_max = 20 h/Mpc zeroes
            the 1-halo term.  With k_max = 200 h/Mpc the error is < 1%.
        n_k : int
            Number of k grid points.
        """
        self._pk_lin = pk_lin
        self._hod = hod
        self._k = jnp.logspace(np.log10(k_min), np.log10(k_max), n_k)

    # ------------------------------------------------------------------
    # Internal: tabulate P_gg and P_gm on self._k
    # ------------------------------------------------------------------

    def _pk_tables(self, z: float, theta_cosmo: dict, hod_params: dict):
        """Return (log_k, log_P_gg, log_P_gm) on the internal k grid.

        ``_integrate`` on HOD models is JIT-compiled and calls CAMB via
        ``dndm``.  CAMB requires concrete Python floats, which are only
        available outside a JIT trace.  ``jax.disable_jit()`` forces all
        JIT-decorated callees to run eagerly so ``float(theta["h"])`` works.
        """
        _, b_eff, _ = self._hod._integrate(float(z), theta_cosmo, hod_params)
        k = self._k
        pk_lin = self._pk_lin.pk_linear(np.asarray(k), float(z), theta_cosmo)
        log_k = jnp.log(k)
        log_pgg = jnp.log(jnp.maximum(b_eff**2 * jnp.asarray(pk_lin), 1e-20))
        log_pgm = jnp.log(jnp.maximum(b_eff * jnp.asarray(pk_lin), 1e-20))
        return log_k, log_pgg, log_pgm

    # ------------------------------------------------------------------
    # ξ_gg(r)
    # ------------------------------------------------------------------

    def xi_3d(
        self,
        r: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> jnp.ndarray:
        """3D galaxy–galaxy correlation function ξ_gg(r).

        .. math::

            \\xi_{gg}(r) = \\frac{1}{2\\pi^2}\\int_0^\\infty
            k^2\\,b_{\\rm eff}^2\\,P_{\\rm lin}(k)\\,j_0(kr)\\,\\mathrm{d}k

        Parameters
        ----------
        r : [Mpc/h], shape (Nr,)

        Returns
        -------
        xi : ξ_gg(r), unitless, shape (Nr,)
        """
        log_k, log_pgg, _ = self._pk_tables(z, theta_cosmo, hod_params)
        return _pk_to_xi(jnp.asarray(r), log_k, log_pgg)

    # ------------------------------------------------------------------
    # wp(rp)
    # ------------------------------------------------------------------

    def wp(
        self,
        rp: jnp.ndarray,
        pi_max: float,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        n_pi: int = 512,
    ) -> jnp.ndarray:
        """Projected correlation function wp(rp) [Mpc/h].

        .. math::

            w_p(r_p) = 2\\int_0^{\\pi_{\\rm max}}
            \\xi_{gg}\\!\\left(\\sqrt{r_p^2 + \\pi^2}\\right)\\mathrm{d}\\pi

        Parameters
        ----------
        rp : projected separations [Mpc/h], shape (Nrp,)
        pi_max : line-of-sight integration limit [Mpc/h]
        n_pi : number of π integration steps

        Returns
        -------
        wp : [Mpc/h], shape (Nrp,)
        """
        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_tab = self.xi_3d(r_tab, z, theta_cosmo, hod_params)
        pi_grid = jnp.linspace(0.0, float(pi_max), n_pi)

        def _one(rp_i):
            r_grid = jnp.sqrt(rp_i**2 + pi_grid**2)
            xi_i = jnp.interp(r_grid, r_tab, xi_tab)
            return 2.0 * jnp.trapezoid(xi_i, pi_grid)

        return jax.vmap(_one)(jnp.asarray(rp))

    # ------------------------------------------------------------------
    # ΔΣ(R)
    # ------------------------------------------------------------------

    def delta_sigma(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        chi_max: float = 300.0,
        n_chi: int = 512,
        n_R_tab: int = 256,
        ia_model=None,
        ia_params: dict = None,
    ) -> jnp.ndarray:
        """Excess projected surface mass density ΔΣ(R) [Msun h pc⁻²].

        .. math::

            \\Delta\\Sigma(R) = \\bar{\\rho}_m\\left[
              \\frac{2}{R^2}\\int_0^R R'\\,w_p^{gm}(R')\\,\\mathrm{d}R'
              - w_p^{gm}(R)\\right] \\times 10^{-12}
            + \\Delta\\Sigma_{\\rm IA}(R)

        where :math:`P_{gm}(k) = b_{\\rm eff}\\,P_{\\rm lin}(k)` (2-halo only).
        The optional IA term :math:`\\Delta\\Sigma_{\\rm IA}` is computed by
        ``ia_model`` (e.g. :class:`NLAModel`); it is negative for :math:`A_{\\rm IA}>0`.

        Parameters
        ----------
        R : projected radii [Mpc/h], shape (NR,)
        chi_max : LOS integration limit [Mpc/h]
        n_chi : number of LOS integration steps
        n_R_tab : number of points in the internal R tabulation
        ia_model : NLAModel or TATTModel, optional
            Intrinsic alignment contribution to add to ΔΣ.
        ia_params : dict, optional
            Parameters for ``ia_model`` (e.g. ``{'A_IA': 0.5, 'eta_IA': 0.0}``).

        Returns
        -------
        ds : ΔΣ(R) [Msun h pc⁻²], shape (NR,)
        """
        log_k, _, log_pgm = self._pk_tables(z, theta_cosmo, hod_params)

        # ξ_gm(r) on a log-spaced r table
        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_gm_tab = _pk_to_xi(r_tab, log_k, log_pgm)

        # wp_gm(R) on a dense log-spaced R table
        R_tab = jnp.logspace(-2, 2.0, n_R_tab)
        # Log-linear hybrid chi grid (fixed size — avoids np.unique shape ambiguity)
        chi_log = jnp.logspace(-2, jnp.log10(float(chi_max)), n_chi // 2)
        chi_lin = jnp.linspace(1.0, float(chi_max), n_chi // 2)
        chi_grid = jnp.sort(jnp.concatenate([chi_log, chi_lin]))

        def _wp_gm_one(R_i):
            r_grid = jnp.sqrt(R_i**2 + chi_grid**2)
            xi_i = jnp.interp(r_grid, r_tab, xi_gm_tab)
            return 2.0 * jnp.trapezoid(xi_i, chi_grid)

        wp_gm_tab = jax.vmap(_wp_gm_one)(R_tab)   # (n_R_tab,)

        # Σ̄(<R) = (2/R²) ∫₀^R R' wp_gm(R') dR'  via cumulative trapezoid
        integrand = R_tab * wp_gm_tab
        dR = jnp.diff(R_tab)
        mid_vals = 0.5 * (integrand[:-1] + integrand[1:])
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mid_vals * dR)])
        sigma_bar_tab = 2.0 * cum / R_tab**2

        ds_tab = (sigma_bar_tab - wp_gm_tab) * _rho_m(theta_cosmo) * 1e-12
        ds = jnp.interp(jnp.asarray(R), R_tab, ds_tab)

        if ia_model is not None and ia_params is not None:
            ds = ds + ia_model.delta_sigma_ia(
                R, z, theta_cosmo, ia_params,
                chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
            )
        return ds

    # ------------------------------------------------------------------
    # w(θ)  — angular clustering (Limber approximation)
    # ------------------------------------------------------------------

    def w_theta(
        self,
        theta_deg: np.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        n_z: tuple = None,
        pi_max_h: float = 300.0,
        n_pi: int = 512,
        n_z_steps: int = 64,
    ) -> jnp.ndarray:
        """Angular two-point correlation function w(θ) via the Limber approximation.

        Projects ξ_gg(r) evaluated at redshift ``z`` onto the galaxy
        redshift distribution n(z), following Coupon et al. (2015) Appendix B:

        .. math::

            w(\\theta) = \\int \\mathrm{d}z\\,\\frac{H(z)}{c}\\,
                        \\left[\\frac{n(z)}{N}\\right]^2\\,
                        w_p\\!\\left(\\chi(z)\\,\\theta\\right)

        with :math:`N = \\int n(z)\\,\\mathrm{d}z`,

        .. math::

            w_p(r_p) = 2\\int_0^{\\pi_{\\rm max}}
            \\xi_{gg}\\!\\left(\\sqrt{r_p^2 + \\pi^2}\\right)\\mathrm{d}\\pi

        and :math:`\\chi(z)` the comoving distance [Mpc/h].

        The galaxy power spectrum :math:`P_{gg}(k) = b_{\\rm eff}^2 P_{\\rm lin}(k)`
        is evaluated at the single effective redshift ``z``; this is a good
        approximation for a narrow n(z) of width :math:`\\Delta z \\lesssim 0.2`.

        The integral constraint is not applied here; callers should subtract
        ``IC = Σ_i w(θ_i) RR_i / Σ_i RR_i`` computed from the modelled w(θ)
        and the survey random-pair counts, as in Coupon+2015 Section 3.2.

        Parameters
        ----------
        theta_deg : angular separations [degrees], shape (Nθ,)
        z : effective redshift for ξ_gg evaluation
        theta_cosmo : cosmology parameter dict
        hod_params : HOD parameter dict
        n_z : ``(z_arr, nz_arr)`` — galaxy redshift distribution; ``nz_arr``
            need not be normalised.  If ``None`` a Gaussian of width
            σ = 0.05 centred on ``z`` is used.
        pi_max_h : line-of-sight integration limit for wp [Mpc/h]
        n_pi : number of π quadrature points
        n_z_steps : number of z quadrature points for the Limber integral

        Returns
        -------
        w_th : w(θ) unitless, shape (Nθ,)
        """
        # --- redshift distribution ------------------------------------------
        if n_z is None:
            sigma_z = 0.05
            z_lo = max(0.005, float(z) - 4.0 * sigma_z)
            z_hi = float(z) + 4.0 * sigma_z
            z_grid = np.linspace(z_lo, z_hi, n_z_steps)
            nz_grid = np.exp(-0.5 * ((z_grid - float(z)) / sigma_z) ** 2)
        else:
            z_in = np.asarray(n_z[0], dtype=float)
            nz_in = np.asarray(n_z[1], dtype=float)
            z_grid = np.linspace(float(z_in.min()), float(z_in.max()), n_z_steps)
            nz_grid = jnp.interp(jnp.asarray(z_grid), jnp.asarray(z_in), jnp.asarray(nz_in), left=0.0, right=0.0)
            nz_grid = jnp.maximum(nz_grid, 0.0)

        N = jnp.trapezoid(jnp.asarray(nz_grid), jnp.asarray(z_grid))
        pz = nz_grid / N  # normalised redshift distribution

        # --- flat ΛCDM geometry ---------------------------------------------
        chi_z = _comoving_dist_h(z_grid, theta_cosmo)  # [Mpc/h]
        Ez = _hubble_E(z_grid, float(theta_cosmo["Omega_m"]))
        Hz_over_c = Ez / 2997.92  # H(z)/c  [h/Mpc]

        # Limber weight: H(z)/c × p(z)²
        limber_wt = Hz_over_c * pz ** 2  # (n_z_steps,)

        # --- tabulate ξ_gg(r) at z_eff -------------------------------------
        log_k, log_pgg, _ = self._pk_tables(z, theta_cosmo, hod_params)
        r_max_tab = float(np.hypot(chi_z.max() * np.deg2rad(np.asarray(theta_deg).max()),
                                   pi_max_h)) * 1.2
        r_tab = jnp.logspace(-2, np.log10(max(r_max_tab, 10.0)), 1024)
        xi_tab = _pk_to_xi(r_tab, log_k, log_pgg)

        # --- projected correlation function wp(rp) -------------------------
        pi_grid = jnp.linspace(0.0, float(pi_max_h), n_pi)

        def _wp(rp_i):
            r_grid = jnp.sqrt(rp_i ** 2 + pi_grid ** 2)
            xi_i = jnp.interp(r_grid, r_tab, xi_tab,
                               left=xi_tab[0], right=jnp.zeros(()))
            return 2.0 * jnp.trapezoid(xi_i, pi_grid)

        # --- Limber integral ------------------------------------------------
        # rp[n_theta, n_z] = chi(z) × theta_rad
        theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=float))  # (Nθ,)
        rp_all = jnp.asarray(
            np.outer(theta_rad, chi_z)
        )                                         # (Nθ, n_z_steps)
        wp_flat = jax.vmap(_wp)(rp_all.reshape(-1))   # (Nθ × n_z_steps,)
        wp_all = wp_flat.reshape(len(theta_rad), n_z_steps)  # (Nθ, n_z_steps)

        integrand = jnp.asarray(limber_wt) * wp_all   # broadcast over θ
        return jnp.trapezoid(integrand, jnp.asarray(z_grid), axis=1)  # (Nθ,)


# ---------------------------------------------------------------------------
# FullHaloModelPrediction — 1-halo + 2-halo model (More+2015)
# ---------------------------------------------------------------------------

class FullHaloModelPrediction:
    """Galaxy power spectrum via the full 1-halo + 2-halo halo model (More+2015).

    Implements the decomposition following More et al. (2015) exactly:

    .. math::

        P_{gg}(k) = P_{gg}^{\\rm 1h}(k) + b_{\\rm eff}^2\\,P_{\\rm lin}(k)

        P_{gm}(k) = P_{gm}^{\\rm 1h}(k) + b_{\\rm eff}\\,P_{\\rm lin}(k)

    The 2-halo term uses P_lin (not P_nl) to avoid double-counting non-linear
    clustering power at the 1h/2h transition scale (More+2015 Section 3.1).

    The 1-halo terms follow More et al. (2015) Eqs. 9 and 13 with the chosen
    radial profile Fourier transform û(k|M) for both the satellite galaxy
    distribution and the halo matter profile.  Poisson satellite statistics
    are assumed throughout (⟨N_s(N_s-1)⟩ = ⟨N_s⟩²).

    Two profile choices are available:

    * ``'nfw'`` — analytic NFW Fourier transform (Cooray & Sheth 2002 Eq. 11)
      via ``halo_profiles.nfw_uk``.  Fast; uses scipy.special.sici.
    * ``'einasto'`` — Einasto (1965) Fourier transform computed by
      Gauss-Legendre quadrature via ``halo_profiles.einasto_uk``.
      Shape parameter ``einasto_alpha`` controls the inner profile slope
      (default 0.18, close to NFW for cluster-mass halos).

    The HOD model must expose a ``nc_ns(log10m_arr, hod_params)`` method
    returning ``(N_c, N_s)`` arrays (all seven HOD classes in
    ``hod_mod.galaxies.hod`` satisfy this).

    Parameters
    ----------
    pk_lin : LinearPowerSpectrum
        Linear P(k) backend (CAMB), used for the 2-halo term (unless ``nl_2halo``
        is set).
    hod : HOD model object
        Any HOD/CSMF class from ``hod_mod.galaxies.hod``.
    halo_profile : HaloProfile or ConcentrationModel
        Provides ``concentration(m, z)`` (and optionally ``theta``) for the
        scale-radius calculation at 200× mean density.
    profile : str
        Radial profile for the 1-halo Fourier transform: ``'nfw'`` (default)
        or ``'einasto'``.
    einasto_alpha : float
        Einasto shape parameter α (default 0.18).  Ignored for ``'nfw'``.
    k_min, k_max : float [h/Mpc]
        Wavenumber range for the P(k) tabulation.
    n_k : int
        Number of k points.
    pk_nl : object, optional
        Non-linear power spectrum backend exposing
        ``pk_nonlinear(k, z, theta) -> array``.  Pass any of
        :class:`~hod_mod.cosmology.nonlinear.NonLinearPowerSpectrum`,
        :class:`~hod_mod.cosmology.nonlinear.HALOFITSpectrum`, or
        :class:`~hod_mod.cosmology.nonlinear.CachedPkNonlinear`.
        Ignored unless ``nl_2halo=True``.
    nl_2halo : bool
        If ``True`` and ``pk_nl`` is provided, replace the linear 2-halo
        :math:`P_{\\rm lin}(k)` with the non-linear :math:`P_{\\rm nl}(k)`.
        Follows Cacciato+2009/2013, Leauthaud+2012, Wibking+2019.
    """

    def __init__(
        self,
        pk_lin,
        hod,
        halo_profile,
        profile: str = "nfw",
        einasto_alpha: float = 0.18,
        k_min: float = 1e-4,
        k_max: float = 200.0,
        n_k: int = 1024,
        baryon_fraction=None,
        pk_nl=None,
        nl_2halo: bool = False,
        bnl_model=None,
    ):
        if profile not in ("nfw", "einasto"):
            raise ValueError(f"profile must be 'nfw' or 'einasto', got {profile!r}")
        self._pk_lin = pk_lin
        self._hod = hod
        self._halo_profile = halo_profile
        self._profile = profile
        self._einasto_alpha = float(einasto_alpha)
        self._k = jnp.logspace(np.log10(k_min), np.log10(k_max), n_k)
        self._baryon_fraction = baryon_fraction
        self._pk_nl = pk_nl
        self._nl_2halo = nl_2halo and (pk_nl is not None)
        self._bnl_model = bnl_model
        # Cache for HOD-parameter-independent quantities (pk_lin, uk, dndm, c).
        # Keyed by _cosmo_cache_key() so re-evaluated only when cosmology changes.
        # Rounded keys prevent per-step cache misses during free-cosmo MAP fitting.
        self._static_cache: dict = {}

    @staticmethod
    def _cosmo_cache_key(z: float, theta_cosmo: dict) -> tuple:
        """Rounded cache key for theta_cosmo.

        Rounding prevents per-step cache misses when free-cosmo MAP fitting
        explores cosmologies that differ only in the last floating-point digits.
        Precision is ~10× finer than typical Nelder-Mead / Powell step sizes.
        """
        return (
            round(float(z), 4),
            round(float(theta_cosmo.get("Omega_m",      0.3153)), 4),
            round(float(theta_cosmo.get("ln10^{10}A_s", 3.044)),  3),
            round(float(theta_cosmo.get("h",             0.6736)), 4),
            round(float(theta_cosmo.get("n_s",           0.9649)), 4),
            round(float(theta_cosmo.get("Omega_b",       0.0493)), 5),
        )

    # ------------------------------------------------------------------
    # Internal: tabulate P_gg and P_gm with full 1h+2h decomposition
    # ------------------------------------------------------------------

    def _pk_tables_full(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        baryon_params: dict = None,
    ) -> dict:
        """Return tabulated log-power spectra using 1h + 2h decomposition.

        When ``baryon_params`` is supplied and a ``baryon_fraction`` model was
        passed to ``__init__``, the galaxy–matter 1-halo integral is split into
        a CDM contribution (dark matter NFW profile) and a gas contribution
        (NFW with reduced concentration):

        .. math::

            P_{gm}^{\\rm 1h}(k) = \\frac{1}{\\bar{n}_g}
            \\int\\!\\mathrm{d}M\\,n(M)
            \\bigl[N_c + N_s\\,\\tilde{u}_{\\rm DM}(k|M)\\bigr]
            \\frac{M}{\\bar{\\rho}_m}
            \\bigl[(1-f_b(M))\\,\\tilde{u}_{\\rm DM}(k|M)
                  + f_b(M)\\,\\tilde{u}_{\\rm gas}(k|M)\\bigr]

        where :math:`\\tilde{u}_{\\rm DM}` uses the DM concentration and
        :math:`\\tilde{u}_{\\rm gas}` uses a reduced concentration
        :math:`c_{\\rm gas}(M) = c_{\\rm DM}(M)\\,\\eta(M)`.

        The gas concentration ratio is parametrised as (arXiv:2409.01758,
        IllustrisTNG z=0 Table 4; Mead+2015 arXiv:1611.08606 §2.3):

        .. math::

            \\eta(M) = 1 - \\frac{1-\\eta_{\\min}}{1+(M/M_\\eta)^{\\beta_\\eta}}

        with default :math:`\\eta_{\\min}=0.6`, :math:`M_\\eta=10^{13}\\,M_\\odot/h`,
        :math:`\\beta_\\eta=1.5` calibrated to IllustrisTNG group-scale halos
        (arXiv:2409.01758 Table 2).  The sigmoid :math:`f_b(M)` is motivated by
        the FLAMINGO f_gas measurements (arXiv:2510.25419) and the closure-radius
        model (arXiv:2603.13095).

        Without baryon_params the standard 1-halo formula is used:

        .. math::

            P_{gm}^{\\rm 1h}(k) = \\frac{1}{\\bar{n}_g}
            \\int\\!\\mathrm{d}M\\,n(M)
            \\bigl[N_c + N_s\\,\\tilde{u}(k|M)\\bigr]
            \\frac{M}{\\bar{\\rho}_m}\\,\\tilde{u}(k|M)

        The galaxy auto-spectrum is unchanged in both cases (satellites trace DM):

        .. math::

            P_{gg}^{\\rm 1h}(k) = \\frac{1}{\\bar{n}_g^2}
            \\int\\!\\mathrm{d}M\\,n(M)\\left[
              N_s^2\\,\\tilde{u}^2 + 2N_c N_s\\,\\tilde{u}
            \\right]

        (More+2015 arXiv:1211.6211 Eqs. 9, 13; Cooray & Sheth 2002 Eq. 11).

        **Off-centering correction** (Johnston+2007 arXiv:0709.4193, More+2015 §3.3):
        If ``hod_params`` contains ``f_off`` and ``sigma_off``, a fraction ``f_off`` of
        central galaxies is assumed off-centered with a Rayleigh-distributed 2D projected
        offset of scale ``sigma_off`` [Mpc/h].  Their Fourier-space contribution is
        damped by the Rayleigh transform:

        .. math::

            W_{\\rm off}(k) = \\exp\\!\\bigl(-k^2\\,\\sigma_{\\rm off}^2/2\\bigr)

        so :math:`N_c \\to N_c^{\\rm eff}(k) = N_c[(1-f_{\\rm off}) + f_{\\rm off}\\,W_{\\rm off}(k)]`
        in both :math:`P_{gm}^{\\rm 1h}` and :math:`P_{gg}^{\\rm 1h}`.
        Setting ``f_off = 0`` (default) recovers the standard formula exactly.

        Returns
        -------
        dict with keys:

        * ``log_k``        — log wavenumber array
        * ``log_pgg``      — log P_gg(k)
        * ``log_pgm``      — log P_gm(k) (total, CDM+baryon when split)
        * ``log_pgm_cdm``  — log P_gm_cdm(k), only set when baryon split is active
        * ``log_pgm_b``    — log P_gm_b(k), only set when baryon split is active
        * ``n_gal``        — galaxy number density [h³ Mpc⁻³]
        * ``b_eff``        — HOD-weighted effective bias
        """
        # ---- HOD-parameter-independent tables (cached by z + cosmology) ----
        cosmo_key = self._cosmo_cache_key(z, theta_cosmo)
        if cosmo_key not in self._static_cache:
            m_grid = self._hod._m_grid
            with jax.disable_jit():
                dndm_arr = self._hod._hmf.dndm(m_grid, float(z), theta_cosmo)
                bias_arr = self._hod._bias(m_grid, float(z), theta_cosmo)
            m_np = np.asarray(m_grid, dtype=float)
            rho_m = _rho_m(theta_cosmo)

            # concentration: HaloProfile takes (m, z); ConcentrationModel takes (m, z, theta)
            try:
                c_np = np.asarray(
                    self._halo_profile.concentration(m_grid, float(z), theta_cosmo),
                    dtype=float,
                )
            except TypeError:
                c_np = np.asarray(
                    self._halo_profile.concentration(m_grid, float(z)),
                    dtype=float,
                )
            # Halo radius consistent with the mdef stored in halo_profile
            delta, rho_ref = self._halo_profile._mdef_delta_rho(float(z), theta_cosmo)
            r_delta = (3.0 * m_np / (4.0 * np.pi * delta * rho_ref)) ** (1.0 / 3.0)
            r_s_np = r_delta / c_np

            k_np = np.asarray(self._k, dtype=float)
            if self._profile == "nfw":
                from hod_mod.cosmology.halo_profiles import nfw_uk as _profile_uk
                uk = np.asarray(_profile_uk(k_np, r_s_np, c_np))
            else:
                from hod_mod.cosmology.halo_profiles import einasto_uk as _profile_uk
                uk = np.asarray(_profile_uk(k_np, r_s_np, c_np, alpha=self._einasto_alpha))

            pk_lin_np = np.asarray(
                self._pk_lin.pk_linear(k_np, float(z), theta_cosmo), dtype=float
            )
            pk_nl_np = None
            if self._nl_2halo:
                pk_nl_np = np.asarray(
                    self._pk_nl.pk_nonlinear(k_np, float(z), theta_cosmo), dtype=float
                )

            # Peak heights for BNL interpolation: nu = delta_c / sigma(M, z)
            with jax.disable_jit():
                sig_arr = self._hod._hmf.sigma(m_grid, float(z), theta_cosmo)
            nu_np = 1.686 / np.asarray(sig_arr, dtype=float)

            # Pre-compute the (Nk, NM, NM) beta^NL matrix once per cosmology
            bnl_matrix = None
            if self._bnl_model is not None:
                bnl_matrix = self._bnl_model.beta_nl_matrix(k_np, nu_np)

            self._static_cache[cosmo_key] = {
                "m_np":      m_np,
                "dndm_np":   np.asarray(dndm_arr, dtype=float),
                "bias_np":   np.asarray(bias_arr, dtype=float),
                "rho_m":     rho_m,
                "uk":        uk,
                "pk_lin":    pk_lin_np,
                "pk_nl":     pk_nl_np,
                "k_np":      k_np,
                "nu_np":     nu_np,
                "bnl_matrix": bnl_matrix,
                # store concentration and halo radius for gas profile computation
                "c_np":      c_np,
                "r_delta":   r_delta,
            }

        sc = self._static_cache[cosmo_key]
        m_np     = sc["m_np"]
        dndm_np  = sc["dndm_np"]
        bias_np  = sc["bias_np"]
        rho_m    = sc["rho_m"]
        uk       = sc["uk"]
        pk_lin   = sc["pk_lin"]
        k_np     = sc["k_np"]

        # Satellite-profile extensions A/B/C: b_sat_conc, f_cut, gamma_inner.
        # These are HOD-level free parameters that modify only the satellite FT
        # (uk_sat); the DM matter profile (uk) is always the standard NFW/Einasto.
        b_sat_conc  = float(hod_params.get("b_sat_conc",   1.0))
        f_cut       = float(hod_params.get("f_cut",         0.0))
        gamma_inner = float(hod_params.get("gamma_inner",   0.0))
        if f_cut > 0.0 or gamma_inner > 0.0:
            from hod_mod.cosmology.halo_profiles import satellite_nfw_uk
            uk_sat = np.asarray(satellite_nfw_uk(
                k_np,
                sc["r_delta"] / sc["c_np"],   # DM scale radius
                sc["c_np"],                    # DM concentration
                sc["r_delta"],                 # halo radius
                b_sat_conc=b_sat_conc, f_cut=f_cut, gamma=gamma_inner,
            ))
        elif b_sat_conc != 1.0:
            from hod_mod.cosmology.halo_profiles import nfw_uk
            c_sat   = b_sat_conc * sc["c_np"]
            r_s_sat = sc["r_delta"] / c_sat
            uk_sat = np.asarray(nfw_uk(k_np, r_s_sat, c_sat))
        else:
            uk_sat = uk

        # HOD occupation on the mass grid (fast, pure JAX)
        with jax.disable_jit():
            nc_arr, ns_arr = self._hod.nc_ns(self._hod._log10m_grid, hod_params)
        nc_np = np.asarray(nc_arr, dtype=float)
        ns_np = np.asarray(ns_arr, dtype=float)

        # n_gal and b_eff from cached dndm + bias, no CAMB call needed
        nt_np = nc_np + ns_np
        n_gal = float(np.trapezoid(dndm_np * nt_np, m_np))

        # Assembly bias correction (Hearin+2016 decorated HOD kernel):
        # A_cen/A_sat modify b_eff via (b-1)/b kernel matching Lange25HODModel._integrate().
        # Without this fix, A_cen=0.5 → b_eff shift is silently ignored in 2h term.
        A_cen_pk = float(hod_params.get("A_cen", 0.0))
        A_sat_pk = float(hod_params.get("A_sat", 0.0))
        if A_cen_pk != 0.0 or A_sat_pk != 0.0:
            gamma_pk = (bias_np - 1.0) / np.where(bias_np > 0.5, bias_np, 0.5)
            b_nc_pk  = bias_np * (1.0 + A_cen_pk * gamma_pk)
            b_ns_pk  = bias_np * (1.0 + A_sat_pk * gamma_pk)
            b_eff = float(
                np.trapezoid(dndm_np * (nc_np * b_nc_pk + ns_np * b_ns_pk), m_np) / n_gal
            )
        else:
            b_eff = float(np.trapezoid(dndm_np * nt_np * bias_np, m_np) / n_gal)

        # Off-centering: H_c(k,M) = N_c × [(1-p_off) + p_off × exp(-k² σ_off²/2)]
        # Two conventions supported:
        #   R_off  (dimensionless): width = R_off × r_s(M)  — More+2015 Eq. 9, mass-dependent
        #   sigma_off (Mpc/h):     width = sigma_off (fixed) — Johnston+2007, legacy
        p_off     = float(hod_params.get("p_off", hod_params.get("f_off", 0.0)))
        R_off     = float(hod_params.get("R_off",     0.0))   # dimensionless
        sigma_off = float(hod_params.get("sigma_off", 0.0))   # fixed Mpc/h (legacy)

        if p_off > 0.0 and R_off > 0.0:
            # More+2015 Eq. 9: mass-dependent width R_off × r_s(M)
            r_s_m = sc["r_delta"] / sc["c_np"]                      # (NM,)
            W_off = np.exp(
                -k_np[:, None]**2 * (R_off * r_s_m[None, :])**2 / 2.0
            )                                                         # (Nk, NM)
            nc_eff = nc_np[None, :] * ((1.0 - p_off) + p_off * W_off)  # (Nk, NM)
        elif p_off > 0.0 and sigma_off > 0.0:
            # Legacy fixed-width (Johnston+2007)
            W_off  = np.exp(-k_np**2 * sigma_off**2 / 2.0)          # (Nk,)
            nc_eff = nc_np[None, :] * ((1.0 - p_off) + p_off * W_off[:, None])  # (Nk, NM)
        else:
            nc_eff = nc_np[None, :]                                  # (1, NM) — no off-centering

        # 1-halo galaxy-galaxy (More+2015 arXiv:1211.6211 Eq. 9, Poisson satellites)
        # uk_sat: satellite spatial FT (may differ from DM profile via Extensions A/B/C)
        integrand_pgg = dndm_np[None, :] * (
            ns_np[None, :]**2 * uk_sat**2
            + 2.0 * nc_eff * ns_np[None, :] * uk_sat
        )
        P_gg_1h = np.trapezoid(integrand_pgg, m_np, axis=1) / n_gal**2   # (Nk,)

        # 1-halo galaxy-matter (More+2015 arXiv:1211.6211 Eq. 13)
        m_over_rho = m_np / rho_m                                # (NM,)

        log_pgm_cdm = None
        log_pgm_b   = None
        P_gm_cdm    = None
        P_gm_b      = None

        pk_2h = sc["pk_nl"] if self._nl_2halo else pk_lin

        if self._baryon_fraction is not None and baryon_params is not None:
            # Gas profile: NFW with mass-dependent concentration ratio η(M).
            # η(M) transitions from η_min (AGN-feedback-dominated groups) to 1
            # (clusters that re-accrete baryons), motivated by IllustrisTNG
            # broken-power-law c_hydro/c_DMO fit (arXiv:2409.01758 Table 2,
            # M_2 = 10^13.03 Msun, α_2 = -0.032) and Mead+2015 arXiv:1611.08606.
            # Closure-radius model (arXiv:2603.13095) provides the first-principles
            # motivation: baryons expelled beyond r_cl contribute nothing inside R_200.
            eta_min  = 10.0 ** float(baryon_params.get("log10_eta_min", np.log10(0.6)))
            M_eta    = 10.0 ** float(baryon_params.get("log10_M_eta",   13.0))
            beta_eta = float(baryon_params.get("beta_eta", 1.5))
            eta_M    = 1.0 - (1.0 - eta_min) / (1.0 + (m_np / M_eta) ** beta_eta)  # (NM,)

            c_gas_np   = sc["c_np"] * eta_M                    # (NM,)
            r_s_gas_np = sc["r_delta"] / c_gas_np              # (NM,)

            # Always NFW for the gas component (Mead+2015 arXiv:1611.08606 §2.3)
            from hod_mod.cosmology.halo_profiles import nfw_uk
            uk_gas = np.asarray(nfw_uk(k_np, r_s_gas_np, c_gas_np))    # (Nk, NM)

            # Mass-dependent baryon fraction f_b(M) (FLAMINGO arXiv:2510.25419;
            # closure-radius model arXiv:2603.13095)
            m_grid = self._hod._m_grid
            fb_np = np.asarray(
                self._baryon_fraction(m_grid, theta_cosmo, baryon_params), dtype=float
            )                                                             # (NM,)

            # Galaxy profile weight: N_c_eff + N_s × ũ_sat
            nt_arr = nc_eff + ns_np[None, :] * uk_sat                   # (Nk, NM)

            # CDM contribution: (1 - f_b) × ũ_DM for the matter Fourier transform
            integrand_pgm_cdm = (
                dndm_np[None, :] * nt_arr * m_over_rho[None, :]
                * uk * (1.0 - fb_np[None, :])
            )
            # Gas contribution: f_b × ũ_gas for the matter Fourier transform
            integrand_pgm_b = (
                dndm_np[None, :] * nt_arr * m_over_rho[None, :]
                * uk_gas * fb_np[None, :]
            )
            P_gm_cdm_1h = np.trapezoid(integrand_pgm_cdm, m_np, axis=1) / n_gal
            P_gm_b_1h   = np.trapezoid(integrand_pgm_b,   m_np, axis=1) / n_gal
            P_gm_1h     = P_gm_cdm_1h + P_gm_b_1h

            P_gm_cdm = P_gm_cdm_1h + b_eff * pk_2h   # updated below if BNL active
            P_gm_b   = P_gm_b_1h
        else:
            # Standard 1-halo galaxy-matter without baryon split
            integrand_pgm = dndm_np[None, :] * (
                nc_eff + ns_np[None, :] * uk_sat
            ) * m_over_rho[None, :] * uk
            P_gm_1h = np.trapezoid(integrand_pgm, m_np, axis=1) / n_gal

        P_gg_2h = b_eff**2 * pk_2h
        P_gm_2h = b_eff * pk_2h

        # Beyond-linear bias correction (Mead & Verde 2021, arXiv:2011.08858)
        if self._bnl_model is not None:
            bnl_matrix = sc["bnl_matrix"]   # (Nk, NM, NM), pre-computed per cosmology
            # Trapezoid integration weights for the mass grid
            _dm = np.zeros_like(m_np)
            _dm[:-1] += np.diff(m_np) / 2.0
            _dm[1:]  += np.diff(m_np) / 2.0
            # Galaxy-side weight: dndm * N_tot * b / n_gal * dm
            w_g = dndm_np * nt_np * bias_np / n_gal * _dm        # (NM,)
            # Matter-side weight: dndm * M * b / rho_m * dm
            w_m = dndm_np * m_np * bias_np / rho_m * _dm         # (NM,)
            delta_gg = self._bnl_model.correction_2h_gg(
                k_np, sc["nu_np"], w_g, uk, beta_matrix=bnl_matrix
            )
            delta_gm = self._bnl_model.correction_2h_gm(
                k_np, sc["nu_np"], w_g, w_m, uk, uk, beta_matrix=bnl_matrix
            )
            P_gg_2h = P_gg_2h + pk_2h * delta_gg
            P_gm_2h = P_gm_2h + pk_2h * delta_gm
            if P_gm_cdm is not None:
                P_gm_cdm = P_gm_cdm + pk_2h * delta_gm

        if P_gm_cdm is not None:
            log_pgm_cdm = jnp.log(jnp.maximum(jnp.asarray(P_gm_cdm), 1e-20))
            log_pgm_b   = jnp.log(jnp.maximum(jnp.asarray(P_gm_b),   1e-20))

        P_gg = P_gg_1h + P_gg_2h
        P_gm = P_gm_1h + P_gm_2h

        log_k       = jnp.log(self._k)
        log_pgg     = jnp.log(jnp.maximum(jnp.asarray(P_gg),     1e-20))
        log_pgg_1h  = jnp.log(jnp.maximum(jnp.asarray(P_gg_1h),  1e-20))
        log_pgg_2h  = jnp.log(jnp.maximum(jnp.asarray(P_gg_2h),  1e-20))
        log_pgm     = jnp.log(jnp.maximum(jnp.asarray(P_gm),     1e-20))
        log_pgm_1h  = jnp.log(jnp.maximum(jnp.asarray(P_gm_1h),  1e-20))
        log_pgm_2h  = jnp.log(jnp.maximum(jnp.asarray(P_gm_2h),  1e-20))
        return {
            "log_k":       log_k,
            "log_pgg":     log_pgg,
            "log_pgg_1h":  log_pgg_1h,
            "log_pgg_2h":  log_pgg_2h,
            "log_pgm":     log_pgm,
            "log_pgm_1h":  log_pgm_1h,
            "log_pgm_2h":  log_pgm_2h,
            "log_pgm_cdm": log_pgm_cdm,
            "log_pgm_b":   log_pgm_b,
            "n_gal":       n_gal,
            "b_eff":       b_eff,
        }

    # ------------------------------------------------------------------
    # Internal helper: ΔΣ from a pre-tabulated log P_gm
    # ------------------------------------------------------------------

    def _delta_sigma_from_pgm(
        self,
        R: jnp.ndarray,
        log_k: jnp.ndarray,
        log_pgm: jnp.ndarray,
        theta_cosmo: dict,
        chi_max: float = 300.0,
        n_chi: int = 512,
        n_R_tab: int = 256,
    ) -> jnp.ndarray:
        """Compute ΔΣ(R) from a tabulated log P_gm(k).

        Shared implementation used by :meth:`delta_sigma` and
        :meth:`delta_sigma_split`.  The log-linear hybrid chi grid avoids the
        ~10× overestimate that arises when ``chi_step >> R_min`` with a purely
        linear grid (see :meth:`delta_sigma`).

        .. math::

            \\Sigma(R) = \\bar{\\rho}_m
            \\int_{-\\chi_{\\max}}^{+\\chi_{\\max}}
            \\xi_{gm}\\!\\left(\\sqrt{R^2+\\chi^2}\\right)\\mathrm{d}\\chi

            \\bar{\\Sigma}(R) = \\frac{2}{R^2}
            \\int_0^R R'\\,\\Sigma(R')\\,\\mathrm{d}R'

            \\Delta\\Sigma(R) = \\bar{\\Sigma}(R) - \\Sigma(R)

        where :math:`\\xi_{gm}` is the galaxy–matter cross-correlation obtained
        by Fourier-transforming :math:`P_{gm}(k)` (Cooray & Sheth 2002
        `arXiv:astro-ph/0206508 <https://arxiv.org/abs/astro-ph/0206508>`_ Eq. 11).

        Parameters
        ----------
        R : projected radii [Mpc/h], shape (NR,)
        log_k : log wavenumber tabulation
        log_pgm : log P_gm(k) tabulation (same length as log_k)
        theta_cosmo : cosmological parameter dict
        chi_max : LOS integration limit [Mpc/h]
        n_chi : number of LOS integration steps
        n_R_tab : internal R tabulation points

        Returns
        -------
        jnp.ndarray, shape (NR,) — ΔΣ [Msun h pc⁻²]
        """
        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_gm_tab = _pk_to_xi(r_tab, log_k, log_pgm)

        R_tab = jnp.logspace(-2, 2.0, n_R_tab)
        # Log-linear hybrid chi grid (fixed size — avoids np.unique shape ambiguity)
        chi_log = jnp.logspace(-2, jnp.log10(float(chi_max)), n_chi // 2)
        chi_lin = jnp.linspace(1.0, float(chi_max), n_chi // 2)
        chi_grid = jnp.sort(jnp.concatenate([chi_log, chi_lin]))

        def _wp_gm_one(R_i):
            r_grid = jnp.sqrt(R_i**2 + chi_grid**2)
            xi_i = jnp.interp(r_grid, r_tab, xi_gm_tab)
            return 2.0 * jnp.trapezoid(xi_i, chi_grid)

        wp_gm_tab = jax.vmap(_wp_gm_one)(R_tab)

        integrand = R_tab * wp_gm_tab
        dR = jnp.diff(R_tab)
        mid_vals = 0.5 * (integrand[:-1] + integrand[1:])
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mid_vals * dR)])
        sigma_bar_tab = 2.0 * cum / R_tab**2

        ds_tab = (sigma_bar_tab - wp_gm_tab) * _rho_m(theta_cosmo) * 1e-12
        return jnp.interp(jnp.asarray(R), R_tab, ds_tab)

    # ------------------------------------------------------------------
    # Public interface — same signatures as HODClusteringPrediction
    # ------------------------------------------------------------------

    def xi_3d(
        self,
        r: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> jnp.ndarray:
        """3D galaxy–galaxy correlation function ξ_gg(r) [Mpc/h]⁻¹.

        Parameters
        ----------
        r : [Mpc/h], shape (Nr,)
        """
        tables = self._pk_tables_full(z, theta_cosmo, hod_params)
        return _pk_to_xi(jnp.asarray(r), tables["log_k"], tables["log_pgg"])

    def wp(
        self,
        rp: jnp.ndarray,
        pi_max: float,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        n_pi: int = 512,
    ) -> jnp.ndarray:
        """Projected correlation function wp(rp) [Mpc/h].

        Parameters
        ----------
        rp : projected separations [Mpc/h], shape (Nrp,)
        pi_max : line-of-sight integration limit [Mpc/h]
        """
        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_tab = self.xi_3d(r_tab, z, theta_cosmo, hod_params)
        pi_grid = jnp.linspace(0.0, float(pi_max), n_pi)

        def _one(rp_i):
            r_grid = jnp.sqrt(rp_i**2 + pi_grid**2)
            xi_i = jnp.interp(r_grid, r_tab, xi_tab)
            return 2.0 * jnp.trapezoid(xi_i, pi_grid)

        return jax.vmap(_one)(jnp.asarray(rp))

    def delta_sigma(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        chi_max: float = 300.0,
        n_chi: int = 512,
        n_R_tab: int = 256,
        ia_model=None,
        ia_params: dict = None,
    ) -> jnp.ndarray:
        """Excess projected surface mass density ΔΣ(R) [Msun h pc⁻²].

        Includes the full 1-halo + 2-halo decomposition plus an optional
        intrinsic alignment (IA) contribution.

        Parameters
        ----------
        R : projected radii [Mpc/h], shape (NR,)
        chi_max : LOS integration limit [Mpc/h]
        n_chi : number of LOS integration steps
        n_R_tab : number of points in the internal R tabulation
        ia_model : NLAModel or TATTModel, optional
        ia_params : dict, optional — parameters for ``ia_model``
        """
        tables = self._pk_tables_full(z, theta_cosmo, hod_params)
        ds = self._delta_sigma_from_pgm(
            R, tables["log_k"], tables["log_pgm"], theta_cosmo,
            chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
        )

        if ia_model is not None and ia_params is not None:
            extra_kwargs = {}
            if hasattr(ia_model, "_nla"):  # TATTModel needs gravitational ΔΣ and b_eff
                extra_kwargs["b_eff"] = tables["b_eff"]
                extra_kwargs["ds_gm"] = ds
            ds = ds + ia_model.delta_sigma_ia(
                R, z, theta_cosmo, ia_params,
                chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
                **extra_kwargs,
            )
        return ds

    def delta_sigma_split(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        baryon_params: dict = None,
        chi_max: float = 300.0,
        n_chi: int = 512,
        n_R_tab: int = 256,
    ) -> dict:
        """Split ΔΣ into CDM and baryon contributions.

        When a ``baryon_fraction`` model was passed to ``__init__`` **and**
        ``baryon_params`` is provided, the split is computed from mass-integrated
        1-halo P_gm integrals with separate Fourier transforms for dark matter
        (NFW with concentration :math:`c_{\\rm DM}`) and gas (NFW with reduced
        concentration :math:`c_{\\rm gas} = \\eta(M)\\,c_{\\rm DM}`):

        .. math::

            \\Delta\\Sigma_{\\rm CDM}(R) = \\Delta\\Sigma\\!\\left[
              P_{gm}^{\\rm CDM}(k)\\right]

            \\Delta\\Sigma_b(R) = \\Delta\\Sigma\\!\\left[
              P_{gm}^{\\rm b}(k)\\right]

            \\Delta\\Sigma_{\\rm total}(R) = \\Delta\\Sigma_{\\rm CDM}(R)
              + \\Delta\\Sigma_b(R)

        where :math:`P_{gm}^{\\rm CDM}` and :math:`P_{gm}^{\\rm b}` are the
        mass-integrated CDM and gas contributions from :meth:`_pk_tables_full`
        (arXiv:2409.01758, Mead+2015 arXiv:1611.08606, arXiv:2603.13095).

        Without ``baryon_fraction`` / ``baryon_params``, the constant cosmic
        baryon fraction :math:`f_b = \\Omega_b/\\Omega_m` is applied as a
        post-hoc scalar split of the total :math:`\\Delta\\Sigma`.

        Parameters
        ----------
        R : jnp.ndarray — projected radii [Mpc/h]
        z, theta_cosmo, hod_params : same as :meth:`delta_sigma`
        baryon_params : dict, optional
            Parameters forwarded to ``_pk_tables_full`` for the gas profile and
            mass-dependent baryon fraction.  Must include at minimum
            ``log10_M_pivot``, ``beta_b``, ``log10_eta_min``, ``log10_M_eta``
            (see :class:`~hod_mod.galaxies.baryon_fraction.BaryonFractionSigmoid`).

        Returns
        -------
        dict with keys ``'cdm'``, ``'b'``, ``'total'`` — each an array of
        shape (NR,) [Msun h pc⁻²].
        """
        tables = self._pk_tables_full(
            z, theta_cosmo, hod_params, baryon_params=baryon_params
        )

        ds_total = self._delta_sigma_from_pgm(
            R, tables["log_k"], tables["log_pgm"], theta_cosmo,
            chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
        )

        if tables["log_pgm_cdm"] is not None and tables["log_pgm_b"] is not None:
            # Use mass-integrated CDM and gas tables from _pk_tables_full
            ds_cdm = self._delta_sigma_from_pgm(
                R, tables["log_k"], tables["log_pgm_cdm"], theta_cosmo,
                chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
            )
            ds_b = self._delta_sigma_from_pgm(
                R, tables["log_k"], tables["log_pgm_b"], theta_cosmo,
                chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
            )
        else:
            # Fallback: post-hoc scalar split using constant cosmic f_b
            f_b_cosmic = float(theta_cosmo["Omega_b"]) / float(theta_cosmo["Omega_m"])
            ds_cdm = (1.0 - f_b_cosmic) * ds_total
            ds_b   = f_b_cosmic * ds_total

        return {
            "cdm":   ds_cdm,
            "b":     ds_b,
            "total": ds_total,
        }

    def wp_components(
        self,
        rp: jnp.ndarray,
        pi_max: float,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        n_pi: int = 512,
    ) -> dict:
        """Projected correlation function split into 1-halo and 2-halo terms.

        Parameters
        ----------
        rp : projected separations [Mpc/h], shape (Nrp,)
        pi_max : line-of-sight integration limit [Mpc/h]

        Returns
        -------
        dict with keys ``'1h'``, ``'2h'``, ``'total'`` — each shape (Nrp,) [Mpc/h].
        """
        tables = self._pk_tables_full(z, theta_cosmo, hod_params)
        r_tab = jnp.logspace(-2, 2.5, 512)
        pi_grid = jnp.linspace(0.0, float(pi_max), n_pi)

        def _wp_from_log_pk(log_pk):
            xi_tab = _pk_to_xi(r_tab, tables["log_k"], log_pk)

            def _one(rp_i):
                r_grid = jnp.sqrt(rp_i**2 + pi_grid**2)
                return 2.0 * jnp.trapezoid(jnp.interp(r_grid, r_tab, xi_tab), pi_grid)

            return jax.vmap(_one)(jnp.asarray(rp))

        return {
            "1h":    _wp_from_log_pk(tables["log_pgg_1h"]),
            "2h":    _wp_from_log_pk(tables["log_pgg_2h"]),
            "total": _wp_from_log_pk(tables["log_pgg"]),
        }

    def delta_sigma_components(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        chi_max: float = 300.0,
        n_chi: int = 512,
        n_R_tab: int = 256,
    ) -> dict:
        """Excess surface mass density split into 1-halo and 2-halo terms.

        Parameters
        ----------
        R : projected radii [Mpc/h], shape (NR,)

        Returns
        -------
        dict with keys ``'1h'``, ``'2h'``, ``'total'`` — each shape (NR,)
        [Msun h pc⁻²].
        """
        tables = self._pk_tables_full(z, theta_cosmo, hod_params)
        return {
            "1h": self._delta_sigma_from_pgm(
                R, tables["log_k"], tables["log_pgm_1h"], theta_cosmo,
                chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
            ),
            "2h": self._delta_sigma_from_pgm(
                R, tables["log_k"], tables["log_pgm_2h"], theta_cosmo,
                chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
            ),
            "total": self._delta_sigma_from_pgm(
                R, tables["log_k"], tables["log_pgm"], theta_cosmo,
                chi_max=chi_max, n_chi=n_chi, n_R_tab=n_R_tab,
            ),
        }

    def n_gal(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> float:
        """Galaxy number density n̄_g [h³ Mpc⁻³].

        .. math::

            \\bar{n}_g(z) = \\int n(M,z)\\,\\langle N \\rangle_M\\,\\mathrm{d}M

        More+2015 Eq. 12.
        """
        cosmo_key = self._cosmo_cache_key(z, theta_cosmo)
        if cosmo_key not in self._static_cache:
            self._pk_tables_full(z, theta_cosmo, hod_params)
        sc = self._static_cache[cosmo_key]
        with jax.disable_jit():
            nc_arr, ns_arr = self._hod.nc_ns(self._hod._log10m_grid, hod_params)
        return float(
            np.trapezoid(
                sc["dndm_np"] * (np.asarray(nc_arr) + np.asarray(ns_arr)),
                sc["m_np"],
            )
        )

    def w_theta(
        self,
        theta_deg: np.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        n_z: tuple = None,
        pi_max_h: float = 300.0,
        n_pi: int = 512,
        n_z_steps: int = 64,
    ) -> jnp.ndarray:
        """Angular two-point correlation function w(θ) via the Limber approximation.

        Identical to ``HODClusteringPrediction.w_theta`` but uses the full
        1h + 2h power spectrum.

        Parameters
        ----------
        theta_deg : [degrees], shape (Nθ,)
        """
        if n_z is None:
            sigma_z = 0.05
            z_lo = max(0.005, float(z) - 4.0 * sigma_z)
            z_hi = float(z) + 4.0 * sigma_z
            z_grid = np.linspace(z_lo, z_hi, n_z_steps)
            nz_grid = np.exp(-0.5 * ((z_grid - float(z)) / sigma_z) ** 2)
        else:
            z_in = np.asarray(n_z[0], dtype=float)
            nz_in = np.asarray(n_z[1], dtype=float)
            z_grid = np.linspace(float(z_in.min()), float(z_in.max()), n_z_steps)
            nz_grid = jnp.interp(jnp.asarray(z_grid), jnp.asarray(z_in), jnp.asarray(nz_in), left=0.0, right=0.0)
            nz_grid = jnp.maximum(nz_grid, 0.0)

        N = jnp.trapezoid(jnp.asarray(nz_grid), jnp.asarray(z_grid))
        pz = nz_grid / N

        chi_z = _comoving_dist_h(z_grid, theta_cosmo)
        Ez = _hubble_E(z_grid, float(theta_cosmo["Omega_m"]))
        Hz_over_c = Ez / 2997.92
        limber_wt = Hz_over_c * pz ** 2

        tables = self._pk_tables_full(z, theta_cosmo, hod_params)
        r_max_tab = float(np.hypot(chi_z.max() * np.deg2rad(np.asarray(theta_deg).max()),
                                   pi_max_h)) * 1.2
        r_tab = jnp.logspace(-2, np.log10(max(r_max_tab, 10.0)), 1024)
        xi_tab = _pk_to_xi(r_tab, tables["log_k"], tables["log_pgg"])

        pi_grid = jnp.linspace(0.0, float(pi_max_h), n_pi)

        def _wp(rp_i):
            r_grid = jnp.sqrt(rp_i ** 2 + pi_grid ** 2)
            xi_i = jnp.interp(r_grid, r_tab, xi_tab,
                               left=xi_tab[0], right=jnp.zeros(()))
            return 2.0 * jnp.trapezoid(xi_i, pi_grid)

        theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=float))
        rp_all = jnp.asarray(np.outer(theta_rad, chi_z))
        wp_flat = jax.vmap(_wp)(rp_all.reshape(-1))
        wp_all = wp_flat.reshape(len(theta_rad), n_z_steps)

        integrand = jnp.asarray(limber_wt) * wp_all
        return jnp.trapezoid(integrand, jnp.asarray(z_grid), axis=1)


# ---------------------------------------------------------------------------
# Measured wp from a galaxy catalogue (corrfunc)
# ---------------------------------------------------------------------------

def projected_correlation_function(
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    rp_bins: np.ndarray,
    pi_max: float = 80.0,
    n_threads: int = 4,
) -> np.ndarray:
    """Measured wp(rp) from a galaxy catalogue using corrfunc.

    Parameters
    ----------
    ra, dec : degrees
    redshift : spectroscopic redshift
    rp_bins : projected separation bin edges [Mpc/h]
    pi_max : line-of-sight integration limit [Mpc/h]
    n_threads : number of OpenMP threads for corrfunc

    Returns
    -------
    wp : ndarray, shape (len(rp_bins)-1,)  [Mpc/h]
    """
    try:
        from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
    except ImportError as e:
        raise ImportError("corrfunc not installed — pip install corrfunc") from e

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=67.36, Om0=0.3100)
    chi = cosmo.comoving_distance(redshift).value

    cz = (redshift * 3e5).astype(np.float64)
    ra_d = ra.astype(np.float64)
    dec_d = dec.astype(np.float64)

    n = len(ra)
    pi_bins = np.arange(0, pi_max + 1.0, 1.0)
    results = DDrppi_mocks(
        1, 2, n_threads, pi_max, rp_bins,
        ra_d, dec_d, cz,
        is_comoving_dist=False,
    )
    dd = results["npairs"].reshape(len(rp_bins) - 1, len(pi_bins) - 1)
    rp_c = 0.5 * (rp_bins[:-1] + rp_bins[1:])
    vol = np.pi * (rp_bins[1:] ** 2 - rp_bins[:-1] ** 2) * 2.0 * pi_max
    rr = n * (n - 1) / 2 * vol / np.max(chi) ** 3 / (4.0 / 3.0 * np.pi)
    wp = 2.0 * np.sum(dd / rr[:, None] - 1.0, axis=1)
    return wp


# ---------------------------------------------------------------------------
# HODProjectedCorrelation — legacy class (Zheng+2007 only)
# ---------------------------------------------------------------------------

class HODProjectedCorrelation:
    """Predicted wp(rp) from a Zheng+2007 HOD via the halo model.

    Kept for backward compatibility.  New code should use
    ``HODClusteringPrediction`` which supports all six HOD models and also
    provides ξ_gg(r) and ΔΣ(R).

    Parameters
    ----------
    hmf : HaloMassFunction
    hod : HODModel  (Zheng+2007)
    pk_nl : NonLinearPowerSpectrum
    """

    def __init__(self, hmf, hod, pk_nl):
        self._hmf = hmf
        self._hod = hod
        self._pk_nl = pk_nl
        self._m_grid = jnp.logspace(10, 16, 256)
        self._log10m_grid = jnp.log10(self._m_grid)

    def power_spectrum_gg(
        self,
        k: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> jnp.ndarray:
        """Galaxy power spectrum P_gg(k) = b_eff² P_nl(k)."""
        from .hod import n_total

        m = self._m_grid
        log10m = self._log10m_grid
        dn = self._hmf.dndm(m, z, theta_cosmo)
        b = self._hmf.bias(m, z, theta_cosmo)
        nt = n_total(
            log10m,
            hod_params["log10mmin"],
            hod_params["sigma_logm"],
            hod_params["log10m0"],
            hod_params["log10m1"],
            hod_params["alpha"],
        )
        n_gal = jnp.trapezoid(dn * nt, m)
        b_eff = jnp.trapezoid(dn * nt * b, m) / n_gal
        # pk_nonlinear is a numpy-based backend — must stay outside any jit boundary
        pk_nl = self._pk_nl.pk_nonlinear(np.asarray(k), z, theta_cosmo)
        return b_eff**2 * pk_nl

    def wp(
        self,
        rp: jnp.ndarray,
        pi_max: float,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> jnp.ndarray:
        """Projected correlation function wp(rp) [Mpc/h]."""
        k = jnp.logspace(-3, 2, 512)
        pgg = self.power_spectrum_gg(k, z, theta_cosmo, hod_params)

        def _xi_at_r(r_i):
            integrand = k**2 * pgg * jnp.sinc(k * r_i / jnp.pi) / (2.0 * jnp.pi**2)
            return jnp.trapezoid(integrand, k)

        pi_grid = jnp.linspace(0.0, pi_max, 256)

        def _wp_at_rp(rp_i):
            r_grid = jnp.sqrt(rp_i**2 + pi_grid**2)
            xi_grid = jax.vmap(_xi_at_r)(r_grid)
            return 2.0 * jnp.trapezoid(xi_grid, pi_grid)

        return jax.vmap(_wp_at_rp)(rp)


# ---------------------------------------------------------------------------
# NonLinearHaloModelPrediction — unified forward model
# ---------------------------------------------------------------------------

_HOD_MODEL_MAP = {
    "zheng07":      "HODModel",
    "kravtsov04":   "Kravtsov04HODModel",
    "more15":       "MoreHODModel",
    "guo18":        "Guo18ICSMFModel",
    "guo19":        "Guo19ICSMFModel",
    "vanuitert16":  "VanUitert16CSMFModel",
    "zumand15":     "ZuMandelbaum15HODModel",
    "zacharegkas25": "Zacharegkas25HODModel",
    "leauthaud12":  "Leauthaud12HODModel",
    "clf_cacciato09": "CLFModel",
}


class NonLinearHaloModelPrediction:
    """Unified forward model for :math:`w_p(r_p)` and :math:`\\Delta\\Sigma(R)`.

    Assembles a non-linear (or linear) 2-halo term with any HOD/CLF occupation
    model and an NFW or Einasto 1-halo term.  All assembly is done from string
    flags; the internal state is a :class:`FullHaloModelPrediction`.

    The public API (``wp``, ``delta_sigma``) is identical to
    :class:`FullHaloModelPrediction`.

    Parameters
    ----------
    pk_lin : LinearPowerSpectrum
        Linear P(k) backend (CAMB) used for both the 2-halo term (when
        ``pk_nl_backend=None``) and internally for the HMF.
    hmf : HaloMassFunction
        Must expose ``.dndm`` and ``.bias``.  Use ``backend='tinker08'`` for
        full JAX autodiff through the HMF integrals.
    halo_profile : HaloProfile
        Provides the concentration–mass relation for the 1-halo Fourier transform.
    hod_model : str
        Occupation model key.  One of:

        ``'zheng07'``, ``'kravtsov04'``, ``'more15'``, ``'guo18'``, ``'guo19'``,
        ``'vanuitert16'``, ``'zumand15'``, ``'zacharegkas25'``,
        ``'leauthaud12'``, ``'clf_cacciato09'``.
    pk_nl_backend : {``'aletheia'``, ``'hmcode'``, ``None``}
        Non-linear P(k) backend for the 2-halo term.

        * ``'aletheia'`` — Sanchez+2025 emulator (JAX-native; supports autodiff
          through :meth:`~hod_mod.cosmology.nonlinear.NonLinearPowerSpectrum.pk_nonlinear_jax`).
        * ``'hmcode'`` — CAMB HMcode-2020 (Mead+2021, arXiv:2009.01858);
          not differentiable w.r.t. cosmological parameters.
        * ``None`` — linear 2-halo term (More+2015 prescription).
    profile : ``'nfw'`` or ``'einasto'``
    baryon_fraction : optional BaryonFractionModel
    k_min, k_max : float [h/Mpc]
    n_k : int

    Examples
    --------
    Non-linear 2-halo with Aletheia (HOD + CLF interchangeable):

    >>> from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
    >>> from hod_mod.cosmology.halo_mass_function import make_hmf
    >>> from hod_mod.cosmology.halo_profiles import HaloProfile
    >>> pk_lin = LinearPowerSpectrum()
    >>> hmf = make_hmf('tinker08', pk_func=pk_lin.pk_linear)
    >>> hp  = HaloProfile({'flat': True, 'H0': 67.36, 'Om0': 0.31,
    ...                    'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.965})
    >>> model = NonLinearHaloModelPrediction(
    ...     pk_lin, hmf, hp, hod_model='more15', pk_nl_backend='aletheia'
    ... )
    >>> theta = LinearPowerSpectrum.default_cosmology()
    >>> hod_p = model.default_hod_params()
    >>> rp    = jnp.logspace(-1, 1.5, 20)
    >>> wp    = model.wp(rp, pi_max=60., z=0.3, theta_cosmo=theta, hod_params=hod_p)
    """

    def __init__(
        self,
        pk_lin,
        hmf,
        halo_profile,
        hod_model: str = "more15",
        pk_nl_backend=None,
        profile: str = "nfw",
        baryon_fraction=None,
        k_min: float = 1e-4,
        k_max: float = 200.0,
        n_k: int = 1024,
    ):
        valid_hod = set(_HOD_MODEL_MAP.keys())
        if hod_model not in valid_hod:
            raise ValueError(f"hod_model must be one of {sorted(valid_hod)!r}")

        valid_nl = {"aletheia", "hmcode", None}
        if pk_nl_backend not in valid_nl:
            raise ValueError(
                f"pk_nl_backend must be 'aletheia', 'hmcode', or None; "
                f"got {pk_nl_backend!r}"
            )

        # Build HOD/CLF object
        hod_obj = self._build_hod(hod_model, hmf)
        self._hod_model_key = hod_model

        # Build non-linear P(k) backend
        pk_nl = None
        if pk_nl_backend is not None:
            from hod_mod.cosmology.nonlinear import (
                NonLinearPowerSpectrum, HALOFITSpectrum, CachedPkNonlinear,
            )
            if pk_nl_backend == "aletheia":
                pk_nl = CachedPkNonlinear(NonLinearPowerSpectrum("aletheia"))
            else:  # "hmcode"
                pk_nl = CachedPkNonlinear(HALOFITSpectrum("mead2020"))

        self._pred = FullHaloModelPrediction(
            pk_lin, hod_obj, halo_profile,
            profile=profile,
            k_min=k_min, k_max=k_max, n_k=n_k,
            baryon_fraction=baryon_fraction,
            pk_nl=pk_nl,
            nl_2halo=(pk_nl is not None),
        )

    @staticmethod
    def _build_hod(hod_model_key: str, hmf):
        """Instantiate the HOD/CLF object for the given key."""
        if hod_model_key == "clf_cacciato09":
            from hod_mod.galaxies.clf import CLFModel
            return CLFModel(hmf, hmf.bias)

        if hod_model_key == "clf_cacciato13":
            from hod_mod.galaxies.clf import CLFModel13
            return CLFModel13(hmf, hmf.bias)

        if hod_model_key == "leauthaud12":
            from hod_mod.galaxies.hod import Leauthaud12HODModel
            return Leauthaud12HODModel(hmf, hmf.bias)

        from hod_mod.galaxies import hod as _hod_module
        cls_name = _HOD_MODEL_MAP[hod_model_key]
        cls = getattr(_hod_module, cls_name)
        # Most HOD classes take (hmf, halo_bias); Zacharegkas25 only takes (hmf,)
        try:
            return cls(hmf, hmf.bias)
        except TypeError:
            return cls(hmf)

    def default_hod_params(self) -> dict:
        """Return default parameters for the configured HOD/CLF model."""
        return self._pred._hod.default_params()

    # ------------------------------------------------------------------
    # Public interface — delegates to FullHaloModelPrediction
    # ------------------------------------------------------------------

    def wp(
        self,
        rp: jnp.ndarray,
        pi_max: float,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        baryon_params: dict = None,
    ) -> jnp.ndarray:
        """Projected correlation function :math:`w_p(r_p)` [Mpc/h].

        Parameters
        ----------
        rp : projected radii [Mpc/h]
        pi_max : LOS integration limit [Mpc/h]
        z : redshift
        theta_cosmo : cosmological parameter dict
        hod_params : HOD/CLF parameter dict
        baryon_params : optional baryon fraction parameters
        """
        return self._pred.wp(rp, pi_max, z, theta_cosmo, hod_params, baryon_params)

    def delta_sigma(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        baryon_params: dict = None,
    ) -> jnp.ndarray:
        r"""Excess projected surface mass density :math:`\Delta\Sigma(R)` [M_sun h pc⁻²].

        Parameters
        ----------
        R : projected radii [Mpc/h]
        z : redshift
        theta_cosmo : cosmological parameter dict
        hod_params : HOD/CLF parameter dict
        baryon_params : optional baryon fraction parameters
        """
        return self._pred.delta_sigma(R, z, theta_cosmo, hod_params, baryon_params)

    def xi_3d(
        self,
        r: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
    ) -> jnp.ndarray:
        """3D galaxy–galaxy correlation function :math:`\\xi_{gg}(r)`.

        Parameters
        ----------
        r : comoving separation [Mpc/h]
        z : redshift
        theta_cosmo : cosmological parameter dict
        hod_params : HOD/CLF parameter dict
        """
        return self._pred.xi_3d(r, z, theta_cosmo, hod_params)
