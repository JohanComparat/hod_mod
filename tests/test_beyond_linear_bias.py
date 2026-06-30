"""Tests for BeyondLinearBiasMead21 (Mead & Verde 2021, arXiv:2011.08858)."""

import numpy as np
import pytest

from hod_mod.core import BeyondLinearBiasMead21
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_THETA = LinearPowerSpectrum.default_cosmology()

_COLOSSUS = {
    "flat": True,
    "H0": _THETA["h"] * 100.0,
    "Om0": _THETA["Omega_m"],
    "Ob0": _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns": _THETA["n_s"],
}


# ---------------------------------------------------------------------------
# BeyondLinearBiasMead21 data loading and interpolation
# ---------------------------------------------------------------------------

class TestBeyondLinearBiasMead21:

    @pytest.fixture(scope="class")
    def bnl(self):
        return BeyondLinearBiasMead21()

    def test_data_loads(self, bnl):
        assert bnl._nu_ref.shape == (8,)
        assert bnl._k_ref.shape == (25,)
        assert bnl._beta_nl_grid.shape == (25, 8, 8)

    def test_nu_ref_increasing(self, bnl):
        assert np.all(np.diff(bnl._nu_ref) > 0)

    def test_k_ref_range(self, bnl):
        assert bnl._k_ref[0] < 0.01        # starts below 0.01 h/Mpc
        assert bnl._k_ref[-1] > 0.5        # extends to ~0.74 h/Mpc

    def test_symmetry(self, bnl):
        """beta^NL(k, nu1, nu2) == beta^NL(k, nu2, nu1) from the data."""
        k = np.array([0.1, 0.3])
        b12 = bnl.beta_nl(k, bnl._nu_ref, bnl._nu_ref)   # (Nk, 8, 8)
        assert np.allclose(b12, b12.transpose(0, 2, 1), atol=1e-12)

    def test_beta_nl_shape(self, bnl):
        k = np.logspace(-2, -0.2, 10)
        nu1 = np.array([1.0, 2.0, 3.0])
        nu2 = np.array([1.5, 2.5])
        result = bnl.beta_nl(k, nu1, nu2)
        assert result.shape == (10, 3, 2)

    def test_outside_k_range_is_zero(self, bnl):
        """k outside tabulated range → beta^NL = 0 (fill_value)."""
        k_small = np.array([1e-4])   # below k_ref[0]
        k_large = np.array([5.0])    # above k_ref[-1]
        nu = bnl._nu_ref

        assert np.allclose(bnl.beta_nl(k_small, nu, nu), 0.0, atol=1e-12)
        assert np.allclose(bnl.beta_nl(k_large, nu, nu), 0.0, atol=1e-12)

    def test_beta_nl_matrix_shape(self, bnl):
        k = np.logspace(-2, -0.2, 20)
        mat = bnl.beta_nl_matrix(k)
        assert mat.shape == (20, 8, 8)

    def test_beta_nl_matrix_matches_beta_nl(self, bnl):
        k = np.array([0.05, 0.2, 0.5])
        mat = bnl.beta_nl_matrix(k)
        direct = bnl.beta_nl(k, bnl._nu_ref, bnl._nu_ref)
        np.testing.assert_allclose(mat, direct, rtol=1e-12)

    def test_correction_shape(self, bnl):
        k_arr = np.logspace(-2, 0, 30)
        nu_arr = np.linspace(0.9, 3.5, 60)
        weights = np.ones(60) / 60.0
        uk = np.ones((30, 60))
        delta_gg = bnl.correction_2h_gg(k_arr, nu_arr, weights, uk)
        assert delta_gg.shape == (30,)

    def test_correction_gm_shape(self, bnl):
        k_arr = np.logspace(-2, 0, 20)
        nu_arr = np.linspace(1.0, 3.0, 50)
        w_g = np.ones(50) / 50.0
        w_m = np.ones(50) / 50.0
        uk = np.ones((20, 50))
        delta_gm = bnl.correction_2h_gm(k_arr, nu_arr, w_g, w_m, uk, uk)
        assert delta_gm.shape == (20,)

    def test_correction_zero_outside_k_range(self, bnl):
        """No correction below and above the tabulated k range."""
        k_outside = np.array([1e-5, 3.0])
        nu_arr = np.linspace(1.0, 3.0, 40)
        weights = np.ones(40) / 40.0
        uk = np.ones((2, 40))
        delta = bnl.correction_2h_gg(k_outside, nu_arr, weights, uk)
        np.testing.assert_allclose(delta, 0.0, atol=1e-12)

    def test_cached_beta_matrix_gives_same_result(self, bnl):
        """Passing pre-computed beta_matrix gives identical result."""
        k_arr = np.logspace(-1.5, -0.2, 15)
        nu_arr = np.linspace(1.0, 3.5, 50)
        weights = np.ones(50) / 50.0
        uk = np.ones((15, 50))

        beta_mat = bnl.beta_nl_matrix(k_arr)
        d1 = bnl.correction_2h_gg(k_arr, nu_arr, weights, uk, beta_matrix=beta_mat)
        d2 = bnl.correction_2h_gg(k_arr, nu_arr, weights, uk, beta_matrix=None)
        np.testing.assert_allclose(d1, d2, rtol=1e-10)


# ---------------------------------------------------------------------------
# Integration with FullHaloModelPrediction
# ---------------------------------------------------------------------------

class TestBNLInFullHaloModel:
    """Verify BNL correction integrates correctly with FullHaloModelPrediction."""

    @pytest.fixture(scope="class")
    def setup(self):
        pk_lin = LinearPowerSpectrum()
        hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        hp = HaloProfile(_COLOSSUS, cm_relation="diemer19")
        hod = MoreHODModel(hmf, hmf.bias)
        bnl = BeyondLinearBiasMead21()
        return pk_lin, hod, hp, bnl

    def test_bnl_pred_instantiates(self, setup):
        pk_lin, hod, hp, bnl = setup
        pred = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
        assert pred._bnl_model is bnl

    def test_without_bnl_instantiates(self, setup):
        pk_lin, hod, hp, bnl = setup
        pred = FullHaloModelPrediction(pk_lin, hod, hp)
        assert pred._bnl_model is None

    def test_bnl_tables_finite(self, setup):
        pk_lin, hod, hp, bnl = setup
        pred = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
        hod_params = MoreHODModel.default_params()
        tables = pred._pk_tables_full(0.3, _THETA, hod_params)
        assert np.all(np.isfinite(np.array(tables["log_pgg"])))
        assert np.all(np.isfinite(np.array(tables["log_pgm"])))

    def test_bnl_modifies_2h_term(self, setup):
        """BNL correction should change the 2-halo term at intermediate k."""
        pk_lin, hod, hp, bnl = setup
        pred_lin = FullHaloModelPrediction(pk_lin, hod, hp)
        pred_bnl = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
        hod_params = MoreHODModel.default_params()

        t_lin = pred_lin._pk_tables_full(0.0, _THETA, hod_params)
        t_bnl = pred_bnl._pk_tables_full(0.0, _THETA, hod_params)

        pgg_lin = np.exp(np.array(t_lin["log_pgg_2h"]))
        pgg_bnl = np.exp(np.array(t_bnl["log_pgg_2h"]))

        # At least some k values should differ
        assert not np.allclose(pgg_lin, pgg_bnl, rtol=1e-6)

    def test_bnl_consistent_with_no_bnl_at_large_scales(self, setup):
        """At very large scales (k < 6e-3), BNL correction is zero → spectra agree."""
        pk_lin, hod, hp, bnl = setup
        pred_lin = FullHaloModelPrediction(
            pk_lin, hod, hp, k_min=1e-4, k_max=5e-3, n_k=10
        )
        pred_bnl = FullHaloModelPrediction(
            pk_lin, hod, hp, k_min=1e-4, k_max=5e-3, n_k=10, bnl_model=bnl
        )
        hod_params = MoreHODModel.default_params()
        t_lin = pred_lin._pk_tables_full(0.0, _THETA, hod_params)
        t_bnl = pred_bnl._pk_tables_full(0.0, _THETA, hod_params)

        pgg_lin = np.exp(np.array(t_lin["log_pgg"]))
        pgg_bnl = np.exp(np.array(t_bnl["log_pgg"]))
        np.testing.assert_allclose(pgg_lin, pgg_bnl, rtol=1e-6)

    def test_nu_np_cached(self, setup):
        """Static cache should include nu_np after first call."""
        pk_lin, hod, hp, bnl = setup
        pred = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
        hod_params = MoreHODModel.default_params()
        pred._pk_tables_full(0.0, _THETA, hod_params)

        key = pred._cosmo_cache_key(0.0, _THETA)
        assert "nu_np" in pred._static_cache[key]
        nu = pred._static_cache[key]["nu_np"]
        assert nu.shape == hod._m_grid.shape
        assert np.all(nu > 0)

    def test_bnl_matrix_cached(self, setup):
        """bnl_matrix in static cache should be (Nk, 8, 8)."""
        pk_lin, hod, hp, bnl = setup
        pred = FullHaloModelPrediction(pk_lin, hod, hp, n_k=64, bnl_model=bnl)
        hod_params = MoreHODModel.default_params()
        pred._pk_tables_full(0.0, _THETA, hod_params)

        key = pred._cosmo_cache_key(0.0, _THETA)
        mat = pred._static_cache[key]["bnl_matrix"]
        assert mat.shape == (64, 8, 8)

    def test_wp_shape_with_bnl(self, setup):
        """wp() should return correct shape when BNL is active."""
        pk_lin, hod, hp, bnl = setup
        pred = FullHaloModelPrediction(pk_lin, hod, hp, bnl_model=bnl)
        rp = np.logspace(-1, 1.5, 8)
        wp = pred.wp(rp, 60.0, 0.3, _THETA, MoreHODModel.default_params())
        assert wp.shape == (8,)
        assert np.all(np.isfinite(np.array(wp)))
        assert np.all(np.array(wp) > 0)
