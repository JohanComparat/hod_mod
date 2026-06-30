"""Intrinsic alignment models for galaxy-galaxy lensing (ΔΣ).

Implements the galaxy-intrinsic (GI) cross-correlation contribution to the
excess surface mass density ΔΣ.  Two models are provided:

* **NLA** — Non-Linear Alignment (Bridle & King 2007, arXiv:0705.0166):
  the simplest widely-used model; amplitude A_IA with power-law redshift
  evolution (1+z)^η_IA.

* **TATT** — Tidal Alignment and Tidal Torquing (Blazek+2019, arXiv:1708.09247):
  extends NLA with a density-weighting bias b_TA.  Tidal torquing (A_2) is
  reserved for future cosmic-shear C_ell extensions.

Both models compute ΔΣ_IA by applying the alignment amplitude to the
nonlinear matter–matter cross term and subtracting from the gravitational
signal.  The sign convention follows standard weak-lensing analyses:

    ΔΣ_observed = ΔΣ_grav + ΔΣ_IA,  ΔΣ_IA < 0 for A_IA > 0

References
----------
Bridle & King 2007, NJPh 9, 444 (arXiv:0705.0166) — NLA model
Blazek et al. 2019, JCAP 08, 010 (arXiv:1708.09247) — TATT model
Hirata & Seljak 2004, PRD 70, 063526 — C₁ normalisation
Joachimi et al. 2015, SSRv 193, 1 (arXiv:1504.05456) — IA review
"""

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.observables.clustering import _pk_to_xi, _rho_m


# C₁ × ρ_crit0 [dimensionless] — Hirata & Seljak 2004 / Brown+2002 calibration.
# Full expression: C₁ = 5×10⁻¹⁴ Msun⁻¹ h² Mpc³,
# ρ_crit0 = 2.775×10¹¹ Msun h² Mpc⁻³ → C₁ ρ_crit0 = 0.0134.
C1_RHO_CRIT0: float = 0.0134


def _linear_growth_factor(z: float, omega_m: float) -> float:
    """Linear growth factor D(z)/D(0), flat ΛCDM (Carroll+1992 fitting formula).

    Reuses ``_growth_factor_flat_jax`` from the HMF module; returns a Python
    float so it can be used outside JAX-traced code.
    """
    from hod_mod.core.halo_mass_function import _growth_factor_flat_jax
    return float(_growth_factor_flat_jax(float(z), float(omega_m)))


def _ia_amplitude(z: float, theta_cosmo: dict, A_ia: float, eta_ia: float) -> float:
    r"""NLA amplitude factor F_IA(z).

    .. math::

        F_{\rm IA}(z) = A_{\rm IA}\,C_1\,\bar{\rho}_m\,(1+z)^{\eta_{\rm IA}}
                        / D^2(z)

    where :math:`\bar{\rho}_m = \Omega_m \rho_{\rm crit,0}` [Msun h² Mpc⁻³]
    and :math:`D(z)` is the linear growth factor normalised to :math:`D(0)=1`.

    Parameters
    ----------
    z : float
    theta_cosmo : dict — needs ``Omega_m``
    A_ia : float — dimensionless IA amplitude
    eta_ia : float — redshift-evolution exponent

    Returns
    -------
    F_IA : float [dimensionless] — sign convention: positive F_IA suppresses ΔΣ.
    """
    D_z = _linear_growth_factor(float(z), float(theta_cosmo["Omega_m"]))
    rho_bar_m = float(theta_cosmo["Omega_m"]) * 2.775e11   # Msun h^2 Mpc^-3
    # C1 * rho_bar_m in (Msun h Mpc^-3) * (Mpc^3 h^-3) units cancel → dimensionless
    c1_rho = C1_RHO_CRIT0 * float(theta_cosmo["Omega_m"])
    return float(A_ia) * c1_rho * (1.0 + z) ** float(eta_ia) / D_z ** 2


class NLAModel:
    r"""Non-Linear Alignment (NLA) contribution to galaxy-galaxy lensing ΔΣ.

    The galaxy-intrinsic (GI) power spectrum in the NLA model is:

    .. math::

        P_{\rm GI}(k,z) = -F_{\rm IA}(z)\,P_{\rm nl}(k,z)

    giving a ΔΣ contribution:

    .. math::

        \Delta\Sigma_{\rm IA}(R) = -F_{\rm IA}(z)\,\Delta\Sigma_{\rm nl}(R)

    where :math:`\Delta\Sigma_{\rm nl}` is computed from :math:`P_{\rm nl}(k,z)`
    using the same Ogata Hankel-transform pipeline as the gravitational signal.

    Parameters
    ----------
    pk_nl : NonLinearPowerSpectrum or callable ``(k, z, theta) -> pk``
        Nonlinear matter power spectrum (Aletheia backend recommended).
    k_min, k_max : float [h/Mpc]
        Wavenumber range matching the clustering grid.
    n_k : int
        Number of k grid points.

    References
    ----------
    Bridle & King 2007, NJPh 9, 444 (arXiv:0705.0166)
    """

    def __init__(self, pk_nl, k_min: float = 1e-4, k_max: float = 20.0, n_k: int = 512):
        self._pk_nl_func = (
            pk_nl.pk_nonlinear if hasattr(pk_nl, "pk_nonlinear") else pk_nl
        )
        self._k = jnp.logspace(np.log10(k_min), np.log10(k_max), n_k)

    def _ds_nl(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        chi_max: float,
        n_chi: int,
        n_R_tab: int,
    ) -> jnp.ndarray:
        """ΔΣ from the nonlinear matter auto-spectrum [Msun h pc⁻²]."""
        k_np = np.asarray(self._k, dtype=float)
        pk_nl_np = np.asarray(
            self._pk_nl_func(k_np, float(z), theta_cosmo), dtype=float
        )
        log_k = jnp.log(self._k)
        log_pnl = jnp.log(jnp.maximum(jnp.asarray(pk_nl_np), 1e-50))

        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_nl_tab = _pk_to_xi(r_tab, log_k, log_pnl)

        R_tab = jnp.logspace(-2, 2.0, n_R_tab)
        _chi_log = np.logspace(-2, np.log10(float(chi_max)), n_chi // 2)
        _chi_lin = np.linspace(1.0, float(chi_max), n_chi // 2)
        chi_grid = jnp.asarray(np.unique(np.concatenate([_chi_log, _chi_lin])))

        def _wp_one(R_i):
            r_grid = jnp.sqrt(R_i ** 2 + chi_grid ** 2)
            xi_i = jnp.interp(r_grid, r_tab, xi_nl_tab)
            return 2.0 * jnp.trapezoid(xi_i, chi_grid)

        wp_nl_tab = jax.vmap(_wp_one)(R_tab)

        integrand = R_tab * wp_nl_tab
        dR = jnp.diff(R_tab)
        mid_vals = 0.5 * (integrand[:-1] + integrand[1:])
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mid_vals * dR)])
        sigma_bar_tab = 2.0 * cum / R_tab ** 2

        ds_tab = (sigma_bar_tab - wp_nl_tab) * _rho_m(theta_cosmo) * 1e-12
        return jnp.interp(jnp.asarray(R), R_tab, ds_tab)

    def delta_sigma_ia(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        ia_params: dict,
        chi_max: float = 300.0,
        n_chi: int = 512,
        n_R_tab: int = 256,
    ) -> jnp.ndarray:
        r"""NLA intrinsic alignment contribution to ΔΣ(R) [Msun h pc⁻²].

        .. math::

            \Delta\Sigma_{\rm IA}(R) = -F_{\rm IA}(z)\,\Delta\Sigma_{\rm nl}(R)

        where :math:`F_{\rm IA}` is the amplitude factor from :func:`_ia_amplitude`.
        The result is negative (suppresses the gravitational signal) for
        :math:`A_{\rm IA} > 0`.

        Parameters
        ----------
        R : jnp.ndarray — projected radii [Mpc/h]
        z : float
        theta_cosmo : dict — cosmological parameters
        ia_params : dict — ``A_IA``, ``eta_IA``
        chi_max, n_chi, n_R_tab : integration controls
        """
        F_IA = _ia_amplitude(
            float(z), theta_cosmo,
            float(ia_params["A_IA"]), float(ia_params["eta_IA"]),
        )
        ds_nl = self._ds_nl(R, z, theta_cosmo, chi_max, n_chi, n_R_tab)
        return -F_IA * ds_nl

    @staticmethod
    def default_params() -> dict:
        """Null IA (no alignment) as default."""
        return {"A_IA": 0.0, "eta_IA": 0.0}


class TATTModel:
    r"""Tidal Alignment and Tidal Torquing (TATT) contribution to ΔΣ.

    Extends NLA with a density-weighting bias b_TA (Blazek+2019):

    .. math::

        \Delta\Sigma_{\rm IA}^{\rm TATT}(R)
          = -F_{\rm IA}^{a}(z)\,\Delta\Sigma_{\rm nl}(R)
            - F_{\rm IA}^{b}(z)\,\Delta\Sigma_{\rm gm}(R)

    where :math:`F_{\rm IA}^{a}` uses amplitude ``a_TA`` (tidal alignment term)
    and :math:`F_{\rm IA}^{b}` uses amplitude ``b_TA`` (density-weighting term
    approximated via the galaxy-matter cross-spectrum scaled by ``b_eff``).

    When ``b_TA = 0``, TATT reduces exactly to NLA with :math:`A_{\rm IA} = a_{\rm TA}`.
    When ``a_TA = A_IA`` and ``b_TA = 0`` the result matches :class:`NLAModel`.

    The tidal-torquing amplitude ``A_2`` is reserved for cosmic-shear C_ell
    extensions and is not implemented here.

    Parameters
    ----------
    pk_nl : NonLinearPowerSpectrum or callable
        Nonlinear P(k) for the tidal-alignment (a_TA) term.
    k_min, k_max : float [h/Mpc]
    n_k : int

    References
    ----------
    Blazek et al. 2019, JCAP 08, 010 (arXiv:1708.09247)
    """

    def __init__(self, pk_nl, k_min: float = 1e-4, k_max: float = 20.0, n_k: int = 512):
        self._nla = NLAModel(pk_nl, k_min=k_min, k_max=k_max, n_k=n_k)

    def delta_sigma_ia(
        self,
        R: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        ia_params: dict,
        ds_gm: jnp.ndarray = None,
        b_eff: float = 1.0,
        chi_max: float = 300.0,
        n_chi: int = 512,
        n_R_tab: int = 256,
    ) -> jnp.ndarray:
        r"""TATT intrinsic alignment contribution to ΔΣ(R) [Msun h pc⁻²].

        .. math::

            \Delta\Sigma_{\rm IA}^{\rm TATT}
              = -F_a\,\Delta\Sigma_{\rm nl}
                - F_b\,\Delta\Sigma_{\rm gm}

        where :math:`F_a = F_{\rm IA}(a_{\rm TA},\eta_{\rm TA})` and
        :math:`F_b = F_{\rm IA}(b_{\rm TA},\eta_{\rm TA}) / b_{\rm eff}`.

        Parameters
        ----------
        R : jnp.ndarray — projected radii [Mpc/h]
        z : float
        theta_cosmo : dict — cosmological parameters
        ia_params : dict — ``a_TA``, ``b_TA``, ``eta_TA``
        ds_gm : jnp.ndarray, optional
            Galaxy-matter ΔΣ on the same R grid [Msun h pc⁻²].  Provide this
            from the clustering predictor to avoid recomputing it.  If None
            the density-weighting (b_TA) term is set to zero.
        b_eff : float
            Effective galaxy bias (used to normalise the b_TA term).
        chi_max, n_chi, n_R_tab : integration controls
        """
        a_TA = float(ia_params.get("a_TA", 0.0))
        b_TA = float(ia_params.get("b_TA", 0.0))
        eta_TA = float(ia_params.get("eta_TA", 0.0))

        # Tidal-alignment term (same structure as NLA)
        F_a = _ia_amplitude(float(z), theta_cosmo, a_TA, eta_TA)
        ds_nl = self._nla._ds_nl(R, z, theta_cosmo, chi_max, n_chi, n_R_tab)
        ds_ia = -F_a * ds_nl

        # Density-weighting term (approximated via galaxy-matter cross-spectrum)
        if b_TA != 0.0 and ds_gm is not None:
            F_b = _ia_amplitude(float(z), theta_cosmo, b_TA / max(b_eff, 1e-10), eta_TA)
            ds_ia = ds_ia - F_b * ds_gm

        return ds_ia

    @staticmethod
    def default_params() -> dict:
        """Null TATT (no alignment) as default."""
        return {"a_TA": 0.0, "b_TA": 0.0, "eta_TA": 0.0}
