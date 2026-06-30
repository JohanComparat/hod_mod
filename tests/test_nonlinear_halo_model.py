"""Tests for non-linear power spectrum backends and FullHaloModelPrediction nl_2halo."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.core.nonlinear import HALOFITSpectrum, CachedPkNonlinear
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction


_THETA = LinearPowerSpectrum.default_cosmology()

_COLOSSUS = {
    "flat": True,
    "H0": _THETA["h"] * 100.0,
    "Om0": _THETA["Omega_m"],
    "Ob0": _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns": _THETA["n_s"],
}


@pytest.fixture(scope="module")
def shared_objects():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    return pk_lin, hmf, hp, hod


class TestHALOFITSpectrum:
    def test_pk_nonlinear_returns_array(self):
        hf = HALOFITSpectrum("mead2020")
        k = np.logspace(-2, 1, 20)
        pk = np.asarray(hf.pk_nonlinear(k, 0.3, _THETA))
        assert pk.shape == (20,)
        assert np.all(pk > 0)

    def test_pk_nonlinear_exceeds_linear_at_high_k(self):
        hf = HALOFITSpectrum("mead2020")
        pk_lin = LinearPowerSpectrum()
        k = np.array([0.5, 1.0, 2.0])
        pnl = np.asarray(hf.pk_nonlinear(k, 0.3, _THETA))
        plin = np.asarray(pk_lin.pk_linear(k, 0.3, _THETA))
        # At k > 0.3 h/Mpc the non-linear power should exceed linear
        assert np.all(pnl > plin), f"P_nl={pnl}, P_lin={plin}"

    def test_variants_finite(self):
        for version in ("mead2020", "takahashi", "original"):
            hf = HALOFITSpectrum(version)
            k = np.logspace(-2, 1, 10)
            pk = np.asarray(hf.pk_nonlinear(k, 0.1, _THETA))
            assert np.all(np.isfinite(pk)), f"version={version}: non-finite P_nl"


class TestCachedPkNonlinear:
    def test_cache_returns_same_result(self):
        cached = CachedPkNonlinear(HALOFITSpectrum("mead2020"))
        k = np.logspace(-2, 1, 30)
        pk1 = np.asarray(cached.pk_nonlinear(k, 0.2, _THETA))
        pk2 = np.asarray(cached.pk_nonlinear(k, 0.2, _THETA))
        np.testing.assert_array_equal(pk1, pk2)

    def test_cache_key_changes_with_z(self):
        cached = CachedPkNonlinear(HALOFITSpectrum("mead2020"))
        k = np.logspace(-2, 1, 30)
        pk_z01 = np.asarray(cached.pk_nonlinear(k, 0.1, _THETA))
        pk_z05 = np.asarray(cached.pk_nonlinear(k, 0.5, _THETA))
        assert not np.allclose(pk_z01, pk_z05)

    def test_cache_interpolation_close_to_direct(self):
        hf = HALOFITSpectrum("mead2020")
        cached = CachedPkNonlinear(hf, n_k=2048)
        k = np.logspace(-2, 0.5, 15)
        pk_direct = np.asarray(hf.pk_nonlinear(k, 0.3, _THETA))
        pk_cached = np.asarray(cached.pk_nonlinear(k, 0.3, _THETA))
        # interpolation should agree to 1% at the grid points
        np.testing.assert_allclose(pk_cached, pk_direct, rtol=0.01)


class TestFullHaloModelNL2Halo:
    def test_nl_2halo_flag_changes_wp(self, shared_objects):
        pk_lin, hmf, hp, hod = shared_objects
        pk_nl = CachedPkNonlinear(HALOFITSpectrum("mead2020"))
        hod_p = MoreHODModel.default_params()

        pred_lin = FullHaloModelPrediction(pk_lin, hod, hp)
        pred_nl  = FullHaloModelPrediction(pk_lin, hod, hp, pk_nl=pk_nl, nl_2halo=True)

        rp = jnp.logspace(-1, 1.5, 10)
        wp_lin = pred_lin.wp(rp, 60.0, 0.3, _THETA, hod_p)
        wp_nl  = pred_nl.wp(rp, 60.0, 0.3, _THETA, hod_p)
        # With non-linear 2-halo the predictions differ (at small rp dominated by 1h,
        # at large rp by 2h)
        assert not jnp.allclose(wp_lin, wp_nl, rtol=1e-6), "nl_2halo made no difference"

    def test_1halo_unchanged_by_nl_flag(self, shared_objects):
        pk_lin, hmf, hp, hod = shared_objects
        pk_nl = CachedPkNonlinear(HALOFITSpectrum("mead2020"))
        hod_p = MoreHODModel.default_params()

        pred_lin = FullHaloModelPrediction(pk_lin, hod, hp)
        pred_nl  = FullHaloModelPrediction(pk_lin, hod, hp, pk_nl=pk_nl, nl_2halo=True)

        rp = jnp.logspace(-1, 1.5, 10)
        with jax.disable_jit():
            tables_lin = pred_lin._pk_tables_full(0.3, _THETA, hod_p)
            tables_nl  = pred_nl._pk_tables_full(0.3, _THETA, hod_p)

        # The 1-halo terms in log space must be identical
        np.testing.assert_allclose(
            np.asarray(tables_lin["log_pgg_1h"]),
            np.asarray(tables_nl["log_pgg_1h"]),
            rtol=1e-10,
        )

    def test_nl_2halo_false_with_pk_nl_provided_uses_linear(self, shared_objects):
        pk_lin, hmf, hp, hod = shared_objects
        pk_nl = CachedPkNonlinear(HALOFITSpectrum("mead2020"))
        hod_p = MoreHODModel.default_params()

        # nl_2halo=False even though pk_nl is provided → should behave like linear
        pred_default = FullHaloModelPrediction(pk_lin, hod, hp)
        pred_silent  = FullHaloModelPrediction(pk_lin, hod, hp, pk_nl=pk_nl, nl_2halo=False)

        rp = jnp.logspace(-1, 1.5, 10)
        wp_default = pred_default.wp(rp, 60.0, 0.3, _THETA, hod_p)
        wp_silent  = pred_silent.wp(rp, 60.0, 0.3, _THETA, hod_p)
        np.testing.assert_allclose(
            np.asarray(wp_default), np.asarray(wp_silent), rtol=1e-8
        )
