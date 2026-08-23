"""Tests for ClusterGalaxyCrossCorrelation."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import numpy as np
import jax.numpy as jnp

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_clustering import ClusterGalaxyCrossCorrelation


_THETA = LinearPowerSpectrum.default_cosmology()

_COLOSSUS = {
    "flat": True,
    "H0": _THETA["h"] * 100.0,
    "Om0": _THETA["Omega_m"],
    "Ob0": _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns": _THETA["n_s"],
}

_HOD_PARAMS = MoreHODModel.default_params()

_B_CLUSTER = 4.5
_LOG10_MMIN_CLUSTER = 13.5


@pytest.fixture(scope="module")
def cross_pred():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    full   = FullHaloModelPrediction(pk_lin, hod, hp)
    return ClusterGalaxyCrossCorrelation(full)


class TestClusterGalaxyCrossCorrelation:
    _RP = jnp.logspace(-1, 1.5, 10)
    _R  = jnp.logspace(-1, 1.5, 10)
    _Z  = 0.16

    def test_xi_3d_shape(self, cross_pred):
        xi = cross_pred.xi_3d(
            self._R, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert xi.shape == (10,)

    def test_xi_3d_positive(self, cross_pred):
        xi = cross_pred.xi_3d(
            self._R, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert jnp.all(xi > 0)

    def test_xi_3d_finite(self, cross_pred):
        xi = cross_pred.xi_3d(
            self._R, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert jnp.all(jnp.isfinite(xi))

    def test_wp_shape(self, cross_pred):
        wp = cross_pred.wp(
            self._RP, pi_max=60.0, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert wp.shape == (10,)

    def test_wp_positive(self, cross_pred):
        wp = cross_pred.wp(
            self._RP, pi_max=60.0, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert jnp.all(wp > 0)

    def test_wp_finite(self, cross_pred):
        wp = cross_pred.wp(
            self._RP, pi_max=60.0, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert jnp.all(jnp.isfinite(wp))

    def test_wp_higher_bias_gives_higher_wp(self, cross_pred):
        """Larger b_cluster should increase the 2-halo term → larger wp."""
        wp_lo = cross_pred.wp(
            self._RP, pi_max=60.0, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=3.0, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        wp_hi = cross_pred.wp(
            self._RP, pi_max=60.0, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=6.0, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        # On large scales (2-halo dominated) higher bias → higher amplitude
        assert float(wp_hi[-1]) > float(wp_lo[-1])

    def test_wp_bias_ratio_shape(self, cross_pred):
        wp_gg = cross_pred._full.wp(
            self._RP, 60.0, self._Z, _THETA, _HOD_PARAMS
        )
        ratio = cross_pred.wp_bias_ratio(
            self._RP, wp_gg, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert ratio.shape == (10,)

    def test_wp_bias_ratio_positive(self, cross_pred):
        wp_gg = cross_pred._full.wp(
            self._RP, 60.0, self._Z, _THETA, _HOD_PARAMS
        )
        ratio = cross_pred.wp_bias_ratio(
            self._RP, wp_gg, z=self._Z,
            theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
            b_cluster=_B_CLUSTER, log10_m_min_cluster=_LOG10_MMIN_CLUSTER,
        )
        assert jnp.all(ratio > 0)

    def test_zero_cluster_count_raises(self, cross_pred):
        """log10_m_min_cluster above all halo masses → zero cluster count."""
        with pytest.raises(ValueError, match="zero cluster count"):
            cross_pred.wp(
                self._RP, pi_max=60.0, z=self._Z,
                theta_cosmo=_THETA, hod_params=_HOD_PARAMS,
                b_cluster=_B_CLUSTER, log10_m_min_cluster=17.0,
            )


# ---------------------------------------------------------------------------
# Float64 numpy oracle for the P_cg assembly (Wave-1 jnp-port regression)
# ---------------------------------------------------------------------------

class TestPcgNumpyOracle:
    def test_pcg_matches_float64_reference(self, cross_pred):
        import jax

        z = 0.16
        log_k, log_pcg = cross_pred._pk_table_cg(
            z, _THETA, _HOD_PARAMS, _B_CLUSTER, _LOG10_MMIN_CLUSTER)

        full = cross_pred._full
        sc = full._static_cache[full._cosmo_cache_key(z, _THETA)]
        with jax.disable_jit():
            nc, ns = full._hod.nc_ns(full._hod._log10m_grid, _HOD_PARAMS)
        nc = np.asarray(nc, dtype=float)
        ns = np.asarray(ns, dtype=float)
        m, dndm, bias = sc["m_np"], sc["dndm_np"], sc["bias_np"]
        uk = np.asarray(sc["uk"], dtype=float)

        nt = nc + ns
        n_gal = np.trapezoid(dndm * nt, m)
        b_gal = np.trapezoid(dndm * nt * bias, m) / n_gal
        N_C = (m >= 10.0 ** _LOG10_MMIN_CLUSTER).astype(float)
        n_cluster = np.trapezoid(dndm * N_C, m)
        # satellites only: the cluster×central coincident pair is a delta at
        # r = 0 and is excluded from the tabulated spectrum (see
        # cross_clustering module docstring)
        p1h = np.trapezoid(
            dndm[None, :] * N_C[None, :] * ns[None, :] * uk,
            m, axis=1) / (n_cluster * n_gal)
        p2h = _B_CLUSTER * b_gal * np.asarray(sc["pk_lin"], dtype=float)
        p_ref = np.maximum(p1h + p2h, 1e-20)

        # the returned table is float32, so 2e-5 is the meaningful floor;
        # an axis/transpose bug in the mass integrals would be O(1)
        np.testing.assert_allclose(
            np.exp(np.asarray(log_pcg, dtype=float)), p_ref, rtol=2e-5)
        np.testing.assert_allclose(
            np.asarray(log_k, dtype=float), np.log(sc["k_np"]), rtol=1e-6)

    def test_empty_cluster_sample_raises(self, cross_pred):
        with pytest.raises(ValueError, match="zero cluster count"):
            cross_pred._pk_table_cg(0.16, _THETA, _HOD_PARAMS,
                                    _B_CLUSTER, log10_m_min_cluster=17.0)
