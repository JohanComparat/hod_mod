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
