"""Cluster-galaxy projected cross-correlation wp^{CG}(rp).

Implements the halo-model prediction for the cross-correlation between a
population of galaxy clusters (acting as tracers of massive halos) and a
background galaxy sample described by an HOD.

Power spectrum decomposition (Comparat & Macias-Perez 2025, Eq. 1-3):

.. math::

    P_{cg}(k) = P_{cg}^{\\rm 1h}(k) + P_{cg}^{\\rm 2h}(k)

**1-halo term** — galaxies physically residing in cluster halos:

.. math::

    P_{cg}^{\\rm 1h}(k) = \\frac{1}{\\bar{n}_C\\,\\bar{n}_G}
    \\int\\!\\mathrm{d}M\\,n(M)\\,N_C(M)
    \\bigl[\\langle N_{\\rm cen}\\rangle_M
           + \\langle N_{\\rm sat}\\rangle_M\\,\\tilde{u}(k|M)\\bigr]

where :math:`N_C(M) = \\Theta(M - M_{\\rm min,C})` is a step function at the
cluster mass threshold (cluster treated as a point mass at halo centre so
:math:`\\tilde{u}_C(k|M) = 1`).

**2-halo term** — large-scale bias coupling:

.. math::

    P_{cg}^{\\rm 2h}(k) = b_C\\,b_{G,{\\rm eff}}\\,P_{\\rm lin}(k)

**Projected cross-correlation**:

.. math::

    w_p^{CG}(r_p) = 2\\int_0^{\\pi_{\\rm max}} \\xi_{cg}(\\sqrt{r_p^2+\\pi^2})\\,\\mathrm{d}\\pi

where :math:`\\xi_{cg}(r)` is obtained from :math:`P_{cg}(k)` via the Ogata
(2005) :math:`j_0` Hankel transform (same quadrature as ``FullHaloModelPrediction``).

``ClusterGalaxyCrossCorrelation`` reuses the static cache of
``FullHaloModelPrediction`` (halo mass function, bias, and NFW/Einasto Fourier
transforms tabulated on the same mass and wavenumber grids) so that a joint
galaxy auto + cluster-galaxy cross fit incurs no redundant HMF evaluations.

Usage example::

    from hod_mod.galaxies.clustering import FullHaloModelPrediction
    from hod_mod.galaxies.cross_clustering import ClusterGalaxyCrossCorrelation

    full = FullHaloModelPrediction(pk_lin, hod, halo_profile, profile='nfw')
    cross = ClusterGalaxyCrossCorrelation(full)

    wp_cg = cross.wp(
        rp, pi_max=100., z=0.16,
        theta_cosmo=theta, hod_params=p,
        b_cluster=4.5, log10_m_min_cluster=13.5,
    )
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .clustering import _pk_to_xi, _rho_m


class ClusterGalaxyCrossCorrelation:
    """Cluster-galaxy cross-correlation wp^{CG}(rp) via the 1h + 2h halo model.

    Parameters
    ----------
    full_halo_model : FullHaloModelPrediction
        Pre-built galaxy auto-correlation predictor.  Its static cache (HMF,
        bias, halo profiles) is reused for the cluster-galaxy terms.
    """

    def __init__(self, full_halo_model):
        self._full = full_halo_model

    # ------------------------------------------------------------------
    # Internal: tabulate P_cg with 1h + 2h decomposition
    # ------------------------------------------------------------------

    def _pk_table_cg(
        self,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        b_cluster: float,
        log10_m_min_cluster: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (log_k, log_P_cg).

        Triggers ``FullHaloModelPrediction._pk_tables_full`` to populate the
        static cache if not already done for this (z, cosmology) pair.

        Parameters
        ----------
        b_cluster : float
            Effective large-scale bias of the cluster sample.
        log10_m_min_cluster : float
            log10 of the minimum cluster halo mass [M_sun/h].
        """
        # Fill cache via the galaxy auto predictor
        self._full._pk_tables_full(z, theta_cosmo, hod_params)

        cosmo_key = (float(z), tuple(sorted(theta_cosmo.items())))
        sc = self._full._static_cache[cosmo_key]

        m_np    = sc["m_np"]
        dndm_np = sc["dndm_np"]
        bias_np = sc["bias_np"]
        uk      = sc["uk"]          # (Nk, NM) — NFW/Einasto Fourier transform table
        pk_lin  = sc["pk_lin"]      # (Nk,)
        k_np    = sc["k_np"]        # (Nk,)

        # HOD occupation
        with jax.disable_jit():
            nc_arr, ns_arr = self._full._hod.nc_ns(
                self._full._hod._log10m_grid, hod_params
            )
        nc_np = np.asarray(nc_arr, dtype=float)
        ns_np = np.asarray(ns_arr, dtype=float)

        # Galaxy number density and effective bias
        nt_np = nc_np + ns_np
        n_gal = float(np.trapezoid(dndm_np * nt_np, m_np))
        b_gal = float(np.trapezoid(dndm_np * nt_np * bias_np, m_np) / n_gal)

        # Cluster step-function occupation: N_C(M) = Θ(M - M_min_C)
        m_min_cluster = 10.0 ** float(log10_m_min_cluster)
        N_C = (m_np >= m_min_cluster).astype(float)

        n_cluster = float(np.trapezoid(dndm_np * N_C, m_np))
        if n_cluster <= 0.0:
            raise ValueError(
                f"log10_m_min_cluster={log10_m_min_cluster} yields zero cluster count."
            )

        # 1-halo cross-power: clusters at halo centres (u_C = 1)
        integrand_cg_1h = dndm_np[None, :] * N_C[None, :] * (
            nc_np[None, :] + ns_np[None, :] * uk
        )
        P_cg_1h = np.trapezoid(integrand_cg_1h, m_np, axis=1) / (n_cluster * n_gal)

        # 2-halo: b_C × b_G_eff × P_lin
        P_cg_2h = float(b_cluster) * b_gal * pk_lin

        P_cg = P_cg_1h + P_cg_2h

        log_k = jnp.log(jnp.asarray(k_np))
        log_pcg = jnp.log(jnp.maximum(jnp.asarray(P_cg), 1e-20))
        return log_k, log_pcg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def xi_3d(
        self,
        r: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        b_cluster: float,
        log10_m_min_cluster: float,
    ) -> jnp.ndarray:
        """3D cluster-galaxy cross-correlation function ξ_cg(r) [Mpc/h]⁻¹.

        Parameters
        ----------
        r : [Mpc/h], shape (Nr,)
        b_cluster : float
            Effective bias of the cluster sample.
        log10_m_min_cluster : float
            log10(M_min_C / [M_sun/h]).
        """
        log_k, log_pcg = self._pk_table_cg(
            z, theta_cosmo, hod_params, b_cluster, log10_m_min_cluster
        )
        return _pk_to_xi(jnp.asarray(r), log_k, log_pcg)

    def wp(
        self,
        rp: jnp.ndarray,
        pi_max: float,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        b_cluster: float,
        log10_m_min_cluster: float,
        n_pi: int = 512,
    ) -> jnp.ndarray:
        """Projected cluster-galaxy cross-correlation wp^{CG}(rp) [Mpc/h].

        .. math::

            w_p^{CG}(r_p) = 2\\int_0^{\\pi_{\\rm max}}
                \\xi_{cg}(\\sqrt{r_p^2+\\pi^2})\\,\\mathrm{d}\\pi

        Parameters
        ----------
        rp : [Mpc/h], shape (Nrp,)
            Projected separation bin centres.
        pi_max : float [Mpc/h]
            Line-of-sight integration limit.
        z : float
            Effective redshift.
        theta_cosmo : dict
            Cosmological parameter dict.
        hod_params : dict
            HOD parameter dict (same keys as the HOD model used in
            ``FullHaloModelPrediction``).
        b_cluster : float
            Effective large-scale bias of the cluster population.
        log10_m_min_cluster : float
            log10(M_min,C / [M_sun/h]) — minimum halo mass hosting a cluster.
        n_pi : int
            Number of line-of-sight grid points for the π integration.

        Returns
        -------
        wp_cg : [Mpc/h], shape (Nrp,)
        """
        log_k, log_pcg = self._pk_table_cg(
            z, theta_cosmo, hod_params, b_cluster, log10_m_min_cluster
        )

        r_tab = jnp.logspace(-2, 2.5, 512)
        xi_tab = _pk_to_xi(r_tab, log_k, log_pcg)
        pi_grid = jnp.linspace(0.0, float(pi_max), n_pi)

        def _one(rp_i):
            r_grid = jnp.sqrt(rp_i**2 + pi_grid**2)
            xi_i = jnp.interp(r_grid, r_tab, xi_tab)
            return 2.0 * jnp.trapezoid(xi_i, pi_grid)

        return jax.vmap(_one)(jnp.asarray(rp))

    def wp_bias_ratio(
        self,
        rp: jnp.ndarray,
        wp_gg: jnp.ndarray,
        z: float,
        theta_cosmo: dict,
        hod_params: dict,
        b_cluster: float,
        log10_m_min_cluster: float,
        n_pi: int = 512,
    ) -> jnp.ndarray:
        """Cross-correlation amplitude relative to galaxy auto-correlation.

        At 2-halo scales: wp^{CG}(rp) ≈ (b_C / b_G) × wp^{GG}(rp).

        This method computes the full model ratio for diagnostics.

        Parameters
        ----------
        wp_gg : [Mpc/h], shape (Nrp,)
            Galaxy auto-correlation wp^{GG}(rp) at the same rp values.

        Returns
        -------
        ratio : shape (Nrp,)  — dimensionless
        """
        wp_cg = self.wp(
            rp, pi_max=100.0, z=z,
            theta_cosmo=theta_cosmo, hod_params=hod_params,
            b_cluster=b_cluster, log10_m_min_cluster=log10_m_min_cluster,
            n_pi=n_pi,
        )
        return wp_cg / jnp.maximum(wp_gg, 1e-10)
