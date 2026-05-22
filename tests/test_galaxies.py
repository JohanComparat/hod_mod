"""Tests for galaxy HOD and SHAM modules."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.galaxies.hod import (
    HODBase,
    n_cen, n_sat, n_total, HODModel,
    _incompleteness_more15, n_cen_more15, n_sat_more15, n_total_more15,
    shmr_guo18, completeness_guo18, n_cen_guo18, n_sat_guo18, n_total_guo18,
    f_quenched, f_starforming, n_cen_guo19,
    shmr_zacharegkas25, n_cen_thresh_z25, n_sat_thresh_z25, n_cen_bin_z25,
    _mean_stellar_mass_c_vanuitert16, n_cen_vanuitert16, n_sat_vanuitert16,
    MoreHODModel, Guo18ICSMFModel, Guo19ICSMFModel,
    Zacharegkas25HODModel, VanUitert16CSMFModel,
    _mh_from_mstar_zu15, _mstar_from_mh_zu15, sigma_lnmstar_zu15,
    n_cen_thresh_zu15, n_sat_thresh_zu15, n_total_thresh_zu15,
    f_red_cen_zu16, f_red_sat_zu16,
    ZuMandelbaum15HODModel, ZuMandelbaum16QuenchingModel,
)
from hod_mod.galaxies.clustering import (
    _ogata_table, _pk_to_xi, HODClusteringPrediction,
    _hubble_E, _comoving_dist_h,
)
from hod_mod.galaxies.sham import (
    smhm_moster13, smhm_behroozi13, smhm_girelli20,
    SHAMModel, _GIRELLI20_NO_SCATTER, _GIRELLI20_SCATTER,
)
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum


_HOD_PARAMS = {
    "log10mmin": 11.35,
    "sigma_logm": 0.25,
    "log10m0": 11.20,
    "log10m1": 12.40,
    "alpha": 1.0,
}

_LOG10M = jnp.linspace(10.0, 15.0, 100)


class TestHOD:
    def test_ncen_range(self):
        nc = n_cen(_LOG10M, _HOD_PARAMS["log10mmin"], _HOD_PARAMS["sigma_logm"])
        assert jnp.all(nc >= 0.0)
        assert jnp.all(nc <= 1.0)

    def test_ncen_asymptotes(self):
        """N_cen → 0 for M << M_min and → 1 for M >> M_min."""
        nc_lo = n_cen(jnp.array([8.0]), 11.35, 0.25)
        nc_hi = n_cen(jnp.array([15.0]), 11.35, 0.25)
        assert float(nc_lo[0]) < 0.01
        assert float(nc_hi[0]) > 0.99

    def test_nsat_zero_below_m0(self):
        nc = n_sat(jnp.array([10.5]), 11.35, 0.25, 11.20, 12.40, 1.0)
        assert float(nc[0]) == pytest.approx(0.0, abs=1e-6)

    def test_ntot_ge_ncen(self):
        nt = n_total(
            _LOG10M,
            _HOD_PARAMS["log10mmin"],
            _HOD_PARAMS["sigma_logm"],
            _HOD_PARAMS["log10m0"],
            _HOD_PARAMS["log10m1"],
            _HOD_PARAMS["alpha"],
        )
        nc = n_cen(_LOG10M, _HOD_PARAMS["log10mmin"], _HOD_PARAMS["sigma_logm"])
        assert jnp.all(nt >= nc - 1e-7)

    def test_jit_ncen(self):
        f = jax.jit(n_cen)
        nc1 = f(_LOG10M, 11.35, 0.25)
        nc2 = n_cen(_LOG10M, 11.35, 0.25)
        np.testing.assert_allclose(nc1, nc2, rtol=1e-5)

    def test_grad_ncen(self):
        """N_cen should be differentiable w.r.t. log10mmin."""
        grad_fn = jax.grad(lambda lmmin: n_cen(jnp.array([12.0]), lmmin, 0.25)[0])
        g = grad_fn(11.35)
        assert jnp.isfinite(g)


class TestSHAM:
    def test_moster13_shape(self):
        log10m = jnp.linspace(10, 15, 50)
        log10ms = smhm_moster13(log10m, z=0.5)
        assert log10ms.shape == (50,)

    def test_moster13_increasing(self):
        log10m = jnp.linspace(10, 15, 50)
        log10ms = smhm_moster13(log10m, z=0.5)
        assert jnp.all(jnp.diff(log10ms) > 0)

    def test_moster13_lt_halo(self):
        """Stellar mass must not exceed halo mass."""
        log10m = jnp.linspace(10, 15, 50)
        log10ms = smhm_moster13(log10m, z=0.0)
        assert jnp.all(log10ms < log10m)

    def test_behroozi13_shape(self):
        log10m = jnp.linspace(10, 15, 50)
        log10ms = smhm_behroozi13(log10m, z=0.5)
        assert log10ms.shape == (50,)

    def test_sham_model_sample(self):
        model = SHAMModel(parametrisation="moster13", scatter_dex=0.2)
        log10m = jnp.linspace(11, 14, 20)
        key = jax.random.PRNGKey(42)
        samples = model.sample(log10m, z=0.5, key=key)
        assert samples.shape == (20,)
        assert jnp.all(jnp.isfinite(samples))

    def test_sham_model_jit(self):
        model = SHAMModel()
        log10m = jnp.linspace(11, 14, 20)
        ms1 = jax.jit(model.log10mstar)(log10m, 0.5)
        ms2 = model.log10mstar(log10m, 0.5)
        np.testing.assert_allclose(ms1, ms2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Shared fixtures for HOD model integration tests
# ---------------------------------------------------------------------------

_THETA = {
    "h": 0.6736, "Omega_m": 0.3111, "Omega_b": 0.0493,
    "Omega_cdm": 0.2607, "n_s": 0.9649, "ln10^{10}A_s": 3.044,
}
_LOG10M_FULL = jnp.linspace(10.0, 15.0, 80)


# ---------------------------------------------------------------------------
# More+2015 standalone functions
# ---------------------------------------------------------------------------

class TestMoreHOD:
    """More+2015 incompleteness HOD standalone functions."""

    _P = MoreHODModel.default_params()

    def test_incompleteness_range(self):
        f = _incompleteness_more15(_LOG10M_FULL, self._P["alpha_inc"], self._P["log10m_inc"])
        assert jnp.all(f >= 0.0) and jnp.all(f <= 1.0)

    def test_incompleteness_shape(self):
        f = _incompleteness_more15(_LOG10M_FULL, 2.0, 12.5)
        assert f.shape == _LOG10M_FULL.shape

    def test_ncen_range(self):
        nc = n_cen_more15(_LOG10M_FULL, self._P["log10mmin"], self._P["sigma_logm"],
                          self._P["alpha_inc"], self._P["log10m_inc"])
        assert jnp.all(nc >= 0.0) and jnp.all(nc <= 1.0)

    def test_nsat_positive(self):
        log10m_high = jnp.linspace(14.0, 16.0, 20)
        ns = n_sat_more15(log10m_high, self._P["log10mmin"], self._P["sigma_logm"],
                          self._P["log10m1"], self._P["alpha"], self._P["kappa"],
                          self._P["alpha_inc"], self._P["log10m_inc"])
        assert jnp.all(ns >= 0.0)

    def test_ntotal_ge_ncen(self):
        p = self._P
        nt = n_total_more15(_LOG10M_FULL, p["log10mmin"], p["sigma_logm"],
                            p["log10m1"], p["alpha"], p["kappa"],
                            p["alpha_inc"], p["log10m_inc"])
        nc = n_cen_more15(_LOG10M_FULL, p["log10mmin"], p["sigma_logm"],
                          p["alpha_inc"], p["log10m_inc"])
        assert jnp.all(nt >= nc - 1e-7)

    def test_jit_ncen_more15(self):
        f = jax.jit(n_cen_more15)
        nc1 = f(_LOG10M_FULL, 13.03, 0.38, 1.0, 13.0)
        nc2 = n_cen_more15(_LOG10M_FULL, 13.03, 0.38, 1.0, 13.0)
        np.testing.assert_allclose(nc1, nc2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Guo+2018 ICSMF standalone functions
# ---------------------------------------------------------------------------

class TestGuo18HOD:
    """Guo+2018 ICSMF standalone functions."""

    _P = Guo18ICSMFModel.default_params()

    def test_shmr_shape(self):
        ms = shmr_guo18(_LOG10M_FULL, self._P["log10m_star0"], self._P["log10m1_shmr"],
                        self._P["alpha_shmr"], self._P["beta_shmr"])
        assert ms.shape == _LOG10M_FULL.shape

    def test_shmr_increasing(self):
        ms = shmr_guo18(_LOG10M_FULL, self._P["log10m_star0"], self._P["log10m1_shmr"],
                        self._P["alpha_shmr"], self._P["beta_shmr"])
        assert jnp.all(jnp.diff(ms) > 0)

    def test_completeness_range(self):
        ms_star = jnp.linspace(9.0, 12.0, 40)
        c = completeness_guo18(ms_star, f_comp=1.0, log10m_star_min=10.5, sigma_c=0.1)
        assert jnp.all(c >= 0.0) and jnp.all(c <= 1.0)

    def test_ncen_range(self):
        p = self._P
        nc = n_cen_guo18(_LOG10M_FULL, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_cen"], p["log10m_star_min_cen"], p["sigma_c_cen"])
        assert jnp.all(nc >= 0.0) and jnp.all(nc <= 1.0)

    def test_nsat_positive(self):
        p = self._P
        ns = n_sat_guo18(_LOG10M_FULL, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_sat"], p["log10m_star_min_sat"], p["sigma_c_sat"],
                         p["log10m1_sat"], p["alpha_sat"])
        assert jnp.all(ns >= 0.0)

    def test_ntotal_eq_ncen_plus_nsat(self):
        p = self._P
        nc = n_cen_guo18(_LOG10M_FULL, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_cen"], p["log10m_star_min_cen"], p["sigma_c_cen"])
        ns = n_sat_guo18(_LOG10M_FULL, p["log10m_star0"], p["log10m1_shmr"],
                         p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                         p["f_sat"], p["log10m_star_min_sat"], p["sigma_c_sat"],
                         p["log10m1_sat"], p["alpha_sat"])
        nt = n_total_guo18(_LOG10M_FULL, p["log10m_star0"], p["log10m1_shmr"],
                           p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                           p["f_cen"], p["log10m_star_min_cen"], p["sigma_c_cen"],
                           p["f_sat"], p["log10m_star_min_sat"], p["sigma_c_sat"],
                           p["log10m1_sat"], p["alpha_sat"])
        np.testing.assert_allclose(nt, nc + ns, rtol=1e-5)


# ---------------------------------------------------------------------------
# Guo+2019 eBOSS ELG functions
# ---------------------------------------------------------------------------

class TestGuo19HOD:
    """Guo+2019 star-forming fraction and occupation functions."""

    _P = Guo19ICSMFModel.default_params()

    def test_fquenched_range(self):
        fq = f_quenched(_LOG10M_FULL, log10m_q=12.0)
        assert jnp.all(fq >= 0.0) and jnp.all(fq <= 1.0)

    def test_fquenched_decreasing_with_mass(self):
        fq = f_quenched(_LOG10M_FULL, log10m_q=12.0)
        assert float(fq[0]) > float(fq[-1])

    def test_fsf_complement(self):
        fq = f_quenched(_LOG10M_FULL, log10m_q=12.0)
        fsf = f_starforming(_LOG10M_FULL, log10m_q=12.0)
        np.testing.assert_allclose(fq + fsf, jnp.ones_like(fq), rtol=1e-6)

    def test_ncen_guo19_le_guo18(self):
        """Guo+2019 central occ. ≤ Guo+2018 because of the star-forming fraction."""
        p = self._P
        nc18 = n_cen_guo18(_LOG10M_FULL, p["log10m_star0"], p["log10m1_shmr"],
                           p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                           p["f_cen"], p["log10m_star_min_cen"], p["sigma_c_cen"])
        nc19 = n_cen_guo19(_LOG10M_FULL, p["log10m_star0"], p["log10m1_shmr"],
                           p["alpha_shmr"], p["beta_shmr"], p["sigma_logm_star"],
                           p["f_cen"], p["log10m_star_min_cen"], p["sigma_c_cen"],
                           p["log10m_q"])
        assert jnp.all(nc19 <= nc18 + 1e-7)


# ---------------------------------------------------------------------------
# Zacharegkas+2025 Kravtsov SHMR and occupation functions
# ---------------------------------------------------------------------------

class TestZacharegkas25HOD:
    """Zacharegkas+2025 SHMR and occupation functions."""

    _P = Zacharegkas25HODModel.default_params()

    def test_shmr_shape(self):
        p = self._P
        ms = shmr_zacharegkas25(_LOG10M_FULL, p["log10m1_shmr"], p["log10eps"],
                                p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"])
        assert ms.shape == _LOG10M_FULL.shape

    def test_shmr_finite(self):
        p = self._P
        ms = shmr_zacharegkas25(_LOG10M_FULL, p["log10m1_shmr"], p["log10eps"],
                                p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"])
        assert jnp.all(jnp.isfinite(ms))

    def test_shmr_increasing(self):
        p = self._P
        ms = shmr_zacharegkas25(_LOG10M_FULL, p["log10m1_shmr"], p["log10eps"],
                                p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"])
        assert jnp.all(jnp.diff(ms) > 0)

    def test_ncen_thresh_range(self):
        p = self._P
        nc = n_cen_thresh_z25(_LOG10M_FULL, p["log10m_star_lo"],
                              p["log10m1_shmr"], p["log10eps"],
                              p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"],
                              p["sigma_logm_star"], p["f_cen"])
        assert jnp.all(nc >= 0.0) and jnp.all(nc <= 1.0)

    def test_ncen_bin_le_thresh(self):
        """Bin occupation ≤ threshold occupation by definition."""
        p = self._P
        nc_thresh = n_cen_thresh_z25(_LOG10M_FULL, p["log10m_star_lo"],
                                     p["log10m1_shmr"], p["log10eps"],
                                     p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"],
                                     p["sigma_logm_star"], p["f_cen"])
        nc_bin = n_cen_bin_z25(_LOG10M_FULL, p["log10m_star_lo"], p["log10m_star_hi"],
                               p["log10m1_shmr"], p["log10eps"],
                               p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"],
                               p["sigma_logm_star"], p["f_cen"])
        assert jnp.all(nc_bin <= nc_thresh + 1e-7)

    def test_nsat_thresh_positive(self):
        p = self._P
        log10m_high = jnp.linspace(13.0, 16.0, 20)
        ns = n_sat_thresh_z25(log10m_high, p["log10m_star_lo"],
                              p["log10m1_shmr"], p["log10eps"],
                              p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"],
                              p["alpha_sat"], p["kappa"],
                              p["B_sat"], p["beta_sat"],
                              p["B_cut"], p["beta_cut"], p["f_sat"])
        assert jnp.all(ns >= 0.0)


# ---------------------------------------------------------------------------
# van Uitert+2016 CSMF standalone functions
# ---------------------------------------------------------------------------

class TestVanUitert16HOD:
    """van Uitert+2016 CSMF standalone functions."""

    _P = VanUitert16CSMFModel.default_params()

    def test_mean_stellar_mass_shape(self):
        p = self._P
        mu = _mean_stellar_mass_c_vanuitert16(_LOG10M_FULL, p["log10m_star0"],
                                              p["log10m_h1"], p["beta1"], p["log10_beta2"])
        assert mu.shape == _LOG10M_FULL.shape

    def test_mean_stellar_mass_increasing(self):
        p = self._P
        mu = _mean_stellar_mass_c_vanuitert16(_LOG10M_FULL, p["log10m_star0"],
                                              p["log10m_h1"], p["beta1"], p["log10_beta2"])
        assert jnp.all(jnp.diff(mu) > 0)

    def test_ncen_range(self):
        p = self._P
        nc = n_cen_vanuitert16(_LOG10M_FULL, p["log10m_star_lo"], p["log10m_star_hi"],
                               p["log10m_star0"], p["log10m_h1"],
                               p["beta1"], p["log10_beta2"], p["sigma_c"])
        assert jnp.all(nc >= 0.0) and jnp.all(nc <= 1.0)

    def test_nsat_positive(self):
        p = self._P
        ns = n_sat_vanuitert16(_LOG10M_FULL, p["log10m_star_lo"], p["log10m_star_hi"],
                               p["log10m_star0"], p["log10m_h1"],
                               p["beta1"], p["log10_beta2"],
                               p["alpha_s"], p["b0"], p["b1"])
        assert jnp.all(ns >= 0.0)


# ---------------------------------------------------------------------------
# HODBase ABC tests
# ---------------------------------------------------------------------------

class TestHODBase:
    @pytest.fixture(scope="class")
    def _hmf_bias(self):
        pk_lin = LinearPowerSpectrum()
        hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        return hmf, hmf.bias

    def test_hodmodel_is_hodbase_instance(self, _hmf_bias):
        hmf, bias = _hmf_bias
        hod = HODModel(hmf, bias)
        assert isinstance(hod, HODBase)

    def test_hodbase_cannot_be_instantiated_directly(self, _hmf_bias):
        hmf, bias = _hmf_bias
        with pytest.raises(TypeError):
            HODBase(hmf, bias)

    def test_single_arg_init_flag(self):
        from hod_mod.galaxies.hod import ZuMandelbaum15HODModel, Zacharegkas25HODModel
        assert ZuMandelbaum15HODModel._SINGLE_ARG_INIT is True
        assert Zacharegkas25HODModel._SINGLE_ARG_INIT is True
        assert HODModel._SINGLE_ARG_INIT is False


# ---------------------------------------------------------------------------
# All six HOD models: ._integrate() integration test
# ---------------------------------------------------------------------------

class TestHODModelsIntegrate:
    """Integration tests for ._integrate() on all six HOD/CSMF model classes.

    Uses jax.disable_jit() so CAMB (called via dndm→sigma) stays concrete.
    Uses LinearPowerSpectrum (CAMB) for a realistic HMF.
    """

    @pytest.fixture(scope="class")
    def hmf_and_theta(self):
        pk_lin = LinearPowerSpectrum()
        theta = pk_lin.default_cosmology()
        hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        return hmf, theta

    @staticmethod
    def _run(hod, theta, params, z=0.5):
        with jax.disable_jit():
            n, b, m = hod._integrate(float(z), theta, params)
        return float(n), float(b), float(m)

    def test_zheng07(self, hmf_and_theta):
        hmf, theta = hmf_and_theta
        hod = HODModel(hmf, hmf.bias)
        n, b, m = self._run(hod, theta, HODModel.default_params())
        assert n > 0 and np.isfinite(n)
        assert b > 0 and np.isfinite(b)
        assert 1e10 < m < 1e16

    def test_more15(self, hmf_and_theta):
        hmf, theta = hmf_and_theta
        hod = MoreHODModel(hmf, hmf.bias)
        n, b, m = self._run(hod, theta, MoreHODModel.default_params())
        assert n > 0 and np.isfinite(n)
        assert b > 0 and np.isfinite(b)

    def test_guo18(self, hmf_and_theta):
        hmf, theta = hmf_and_theta
        hod = Guo18ICSMFModel(hmf, hmf.bias)
        n, b, m = self._run(hod, theta, Guo18ICSMFModel.default_params())
        assert n > 0 and np.isfinite(n)
        assert b > 0 and np.isfinite(b)

    def test_guo19(self, hmf_and_theta):
        hmf, theta = hmf_and_theta
        hod = Guo19ICSMFModel(hmf, hmf.bias)
        n, b, m = self._run(hod, theta, Guo19ICSMFModel.default_params())
        assert n > 0 and np.isfinite(n)
        assert b > 0 and np.isfinite(b)

    def test_zacharegkas25(self, hmf_and_theta):
        hmf, theta = hmf_and_theta
        hod = Zacharegkas25HODModel(hmf)
        n, b, m = self._run(hod, theta, Zacharegkas25HODModel.default_params())
        assert n > 0 and np.isfinite(n)
        assert b > 0 and np.isfinite(b)

    def test_vanuitert16(self, hmf_and_theta):
        hmf, theta = hmf_and_theta
        hod = VanUitert16CSMFModel(hmf)
        n, b, m = self._run(hod, theta, VanUitert16CSMFModel.default_params())
        assert n > 0 and np.isfinite(n)
        assert b > 0 and np.isfinite(b)

    def test_bias_gt_one_for_massive(self, hmf_and_theta):
        """BOSS CMASS-like More+2015 HOD should give b_eff > 1.5."""
        hmf, theta = hmf_and_theta
        hod = MoreHODModel(hmf, hmf.bias)
        _, b, _ = self._run(hod, theta, MoreHODModel.default_params())
        assert b > 1.5

    def test_bias_increases_with_mmin(self, hmf_and_theta):
        """Higher M_min → more massive halos → higher b_eff."""
        hmf, theta = hmf_and_theta
        hod = HODModel(hmf, hmf.bias)
        p_lo = dict(HODModel.default_params(), log10mmin=11.0)
        p_hi = dict(HODModel.default_params(), log10mmin=13.0)
        _, b_lo, _ = self._run(hod, theta, p_lo)
        _, b_hi, _ = self._run(hod, theta, p_hi)
        assert b_hi > b_lo


# ---------------------------------------------------------------------------
# Ogata (2005) Hankel transform and _pk_to_xi
# ---------------------------------------------------------------------------

class TestOgataHankel:
    """Ogata 2005 quadrature node table and _pk_to_xi."""

    def test_table_shapes(self):
        x, w = _ogata_table(128, 0.01)
        assert x.shape == (128,) and w.shape == (128,)

    def test_nodes_positive(self):
        x, _ = _ogata_table(128, 0.01)
        assert np.all(x > 0)

    def test_nodes_increasing(self):
        x, _ = _ogata_table(256, 0.005)
        assert np.all(np.diff(x) > 0)

    def test_default_table_size(self):
        x, w = _ogata_table()
        assert len(x) == 512

    def test_pk_to_xi_shape(self):
        r = jnp.logspace(-1, 1, 20)
        k = jnp.logspace(-2, 1, 200)
        pk = 3000.0 * (k / 0.1) ** (-2.0)
        xi = _pk_to_xi(r, jnp.log(k), jnp.log(pk))
        assert xi.shape == (20,)

    def test_pk_to_xi_finite(self):
        r = jnp.logspace(-1, 1, 20)
        k = jnp.logspace(-2, 1, 200)
        pk = 3000.0 * (k / 0.1) ** (-2.0)
        xi = _pk_to_xi(r, jnp.log(k), jnp.log(pk))
        assert jnp.all(jnp.isfinite(xi))

    def test_pk_to_xi_positive(self):
        """Power-law P(k) ∝ k^{-2} gives positive ξ(r)."""
        r = jnp.logspace(0, 1.5, 20)
        k = jnp.logspace(-2, 1, 256)
        pk = 5000.0 * (k / 0.1) ** (-2.0)
        xi = _pk_to_xi(r, jnp.log(k), jnp.log(pk))
        assert jnp.all(xi > 0)

    def test_pk_to_xi_decreasing(self):
        r = jnp.logspace(0, 1.5, 20)
        k = jnp.logspace(-2, 1, 256)
        pk = 5000.0 * (k / 0.1) ** (-2.0)
        xi = _pk_to_xi(r, jnp.log(k), jnp.log(pk))
        assert jnp.all(jnp.diff(xi) < 0)

    def test_pk_to_xi_jit(self):
        r = jnp.logspace(0, 1, 10)
        k = jnp.logspace(-2, 1, 128)
        pk = 1000.0 * (k / 0.1) ** (-2.0)
        xi1 = _pk_to_xi(r, jnp.log(k), jnp.log(pk))
        xi2 = jax.jit(_pk_to_xi)(r, jnp.log(k), jnp.log(pk))
        np.testing.assert_allclose(xi1, xi2, rtol=1e-5)

    def test_pk_to_xi_accuracy(self):
        # P(k) = A / k^2 gives xi(r) = A / (4 pi r) analytically.
        # Power-law decay of xi avoids float32 cancellation issues that
        # afflict exponentially-small test values at large r.
        A = 50.0
        k = jnp.logspace(-4, 2.5, 1024)
        pk = A / k ** 2
        r = jnp.array([1.0, 5.0, 10.0])
        xi = _pk_to_xi(r, jnp.log(k), jnp.log(pk))
        xi_analytic = A / (4.0 * jnp.pi * r)
        np.testing.assert_allclose(xi, xi_analytic, rtol=0.02)


# ---------------------------------------------------------------------------
# HODClusteringPrediction with a mock nonlinear P(k)
# ---------------------------------------------------------------------------

class TestHODClusteringPrediction:
    """HODClusteringPrediction.xi_3d / wp / delta_sigma with a mock P_lin."""

    @pytest.fixture(scope="class")
    def setup(self):
        class _MockLinPk:
            def pk_linear(self, k, z, theta):
                k_arr = jnp.asarray(k)
                return 3000.0 * (k_arr / 0.1) ** (-2.0)

        pk_lin = LinearPowerSpectrum()
        theta = pk_lin.default_cosmology()
        hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        hod = HODModel(hmf, hmf.bias)
        pred = HODClusteringPrediction(_MockLinPk(), hod)
        params = HODModel.default_params()
        return pred, theta, params

    def test_xi_shape(self, setup):
        pred, theta, params = setup
        r = jnp.logspace(0, 1.5, 15)
        xi = pred.xi_3d(r, 0.5, theta, params)
        assert xi.shape == (15,)

    def test_xi_finite(self, setup):
        pred, theta, params = setup
        r = jnp.logspace(0, 1.5, 15)
        xi = pred.xi_3d(r, 0.5, theta, params)
        assert jnp.all(jnp.isfinite(xi))

    def test_xi_positive(self, setup):
        pred, theta, params = setup
        r = jnp.logspace(0, 1.5, 15)
        xi = pred.xi_3d(r, 0.5, theta, params)
        assert jnp.all(xi > 0)

    def test_wp_shape(self, setup):
        pred, theta, params = setup
        rp = jnp.logspace(0, 1.5, 12)
        wp = pred.wp(rp, 100.0, 0.5, theta, params)
        assert wp.shape == (12,)

    def test_wp_positive(self, setup):
        pred, theta, params = setup
        rp = jnp.logspace(0, 1.5, 12)
        wp = pred.wp(rp, 100.0, 0.5, theta, params)
        assert jnp.all(wp > 0)

    def test_wp_finite(self, setup):
        pred, theta, params = setup
        rp = jnp.logspace(0, 1.5, 12)
        wp = pred.wp(rp, 100.0, 0.5, theta, params)
        assert jnp.all(jnp.isfinite(wp))

    def test_delta_sigma_shape(self, setup):
        pred, theta, params = setup
        R = jnp.logspace(0, 1.3, 10)
        ds = pred.delta_sigma(R, 0.5, theta, params)
        assert ds.shape == (10,)

    def test_delta_sigma_finite(self, setup):
        pred, theta, params = setup
        R = jnp.logspace(0, 1.3, 10)
        ds = pred.delta_sigma(R, 0.5, theta, params)
        assert jnp.all(jnp.isfinite(ds))

    def test_wp_gt_xi_at_same_scale(self, setup):
        """wp(rp) > ξ(r=rp) since wp integrates ξ over LOS."""
        pred, theta, params = setup
        rp = jnp.array([1.0, 5.0, 10.0])
        xi = pred.xi_3d(rp, 0.5, theta, params)
        wp = pred.wp(rp, 100.0, 0.5, theta, params)
        assert jnp.all(wp > xi)

    def test_w_theta_shape(self, setup):
        pred, theta, params = setup
        theta_deg = np.logspace(-2, 0.3, 10)
        wt = pred.w_theta(theta_deg, 0.8, theta, params)
        assert wt.shape == (10,)

    def test_w_theta_finite(self, setup):
        pred, theta, params = setup
        theta_deg = np.logspace(-2, 0.3, 10)
        wt = pred.w_theta(theta_deg, 0.8, theta, params)
        assert jnp.all(jnp.isfinite(wt))

    def test_w_theta_positive(self, setup):
        pred, theta, params = setup
        theta_deg = np.logspace(-2, 0.3, 10)
        wt = pred.w_theta(theta_deg, 0.8, theta, params)
        assert jnp.all(wt > 0)

    def test_w_theta_decreasing(self, setup):
        """w(θ) should decrease with increasing angular separation."""
        pred, theta, params = setup
        theta_deg = np.logspace(-2, 0.3, 10)
        wt = pred.w_theta(theta_deg, 0.8, theta, params)
        assert jnp.all(jnp.diff(wt) < 0)

    def test_w_theta_with_nz(self, setup):
        """w(θ) computed with an explicit n(z) should be finite and positive."""
        pred, theta, params = setup
        z_arr = np.linspace(0.5, 1.0, 40)
        # Top-hat n(z)
        nz_arr = np.ones_like(z_arr)
        theta_deg = np.logspace(-2, 0.0, 8)
        wt = pred.w_theta(theta_deg, 0.75, theta, params, n_z=(z_arr, nz_arr))
        assert wt.shape == (8,)
        assert jnp.all(jnp.isfinite(wt))
        assert jnp.all(wt > 0)

    def test_w_theta_nz_default_vs_explicit(self, setup):
        """Default Gaussian n(z) and explicit Gaussian n(z) should agree."""
        pred, theta, params = setup
        z0 = 0.8
        sigma_z = 0.05
        z_arr = np.linspace(z0 - 4 * sigma_z, z0 + 4 * sigma_z, 64)
        nz_arr = np.exp(-0.5 * ((z_arr - z0) / sigma_z) ** 2)
        theta_deg = np.array([0.1, 0.3, 1.0])
        wt_default = pred.w_theta(theta_deg, z0, theta, params, n_z=None, n_z_steps=64)
        wt_explicit = pred.w_theta(theta_deg, z0, theta, params, n_z=(z_arr, nz_arr), n_z_steps=64)
        np.testing.assert_allclose(wt_default, wt_explicit, rtol=0.02)


# ---------------------------------------------------------------------------
# Flat ΛCDM geometry helpers
# ---------------------------------------------------------------------------

class TestFlatLCDMGeometry:
    """Unit tests for _hubble_E and _comoving_dist_h."""

    _OMEGA_M = 0.3111
    _THETA = {"Omega_m": _OMEGA_M}

    def test_hubble_E_at_z0(self):
        assert _hubble_E(np.array([0.0]), self._OMEGA_M)[0] == pytest.approx(1.0, rel=1e-6)

    def test_hubble_E_increasing(self):
        z = np.linspace(0.0, 3.0, 30)
        E = _hubble_E(z, self._OMEGA_M)
        assert np.all(np.diff(E) > 0)

    def test_hubble_E_shape(self):
        z = np.linspace(0.0, 2.0, 50)
        E = _hubble_E(z, self._OMEGA_M)
        assert E.shape == (50,)

    def test_comoving_dist_zero_at_z0(self):
        chi = _comoving_dist_h(np.array([0.0]), self._THETA)
        assert chi[0] == pytest.approx(0.0, abs=1.0)

    def test_comoving_dist_increasing(self):
        z = np.linspace(0.01, 3.0, 40)
        chi = _comoving_dist_h(z, self._THETA)
        assert np.all(np.diff(chi) > 0)

    def test_comoving_dist_planck_z1(self):
        """χ(z=1) ≈ 2298 Mpc/h for Planck 2018 (Ω_m=0.3111, flat, h-units).

        Analytic: c/H0 × ∫₀¹ dz/E(z) = 2997.92 × 0.766 ≈ 2298 Mpc/h.
        """
        theta = {"Omega_m": 0.3111}
        chi = _comoving_dist_h(np.array([1.0]), theta)
        assert chi[0] == pytest.approx(2298.0, rel=0.02)


# ---------------------------------------------------------------------------
# Zu & Mandelbaum 2015 iHOD
# ---------------------------------------------------------------------------

_ZU15_SHMR = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42, gamma=1.21)
_ZU15_LOG10M = jnp.linspace(10.0, 15.0, 80)
_ZU15_LOG10MSTAR = jnp.linspace(7.0, 12.5, 80)


class TestMhFromMstarZu15:
    def test_output_shape(self):
        out = _mh_from_mstar_zu15(_ZU15_LOG10MSTAR, **_ZU15_SHMR)
        assert out.shape == _ZU15_LOG10MSTAR.shape

    def test_monotonically_increasing(self):
        out = _mh_from_mstar_zu15(_ZU15_LOG10MSTAR, **_ZU15_SHMR)
        assert jnp.all(jnp.diff(out) > 0)

    def test_pivot_point(self):
        """At M_* = M_{*0}, Eq. 19 gives M_h ≈ M_1 * exp(-0.5) / ln(10)."""
        lg_mh = _mh_from_mstar_zu15(
            jnp.array([10.31]), **_ZU15_SHMR
        )
        # m = 1, beta*0 = 0, exponent = 1/(1+1) - 0.5 = 0
        assert float(lg_mh[0]) == pytest.approx(12.10, abs=0.01)

    def test_finite(self):
        out = _mh_from_mstar_zu15(_ZU15_LOG10MSTAR, **_ZU15_SHMR)
        assert jnp.all(jnp.isfinite(out))


class TestMstarFromMhZu15:
    def test_roundtrip(self):
        """Bisection inversion: M_h -> M_* -> M_h recovers original M_h."""
        log10m_star_c = _mstar_from_mh_zu15(_ZU15_LOG10M, **_ZU15_SHMR)
        log10m_h_recovered = _mh_from_mstar_zu15(log10m_star_c, **_ZU15_SHMR)
        np.testing.assert_allclose(
            np.array(log10m_h_recovered), np.array(_ZU15_LOG10M), atol=1e-3
        )

    def test_output_shape(self):
        out = _mstar_from_mh_zu15(_ZU15_LOG10M, **_ZU15_SHMR)
        assert out.shape == _ZU15_LOG10M.shape

    def test_monotonically_increasing(self):
        out = _mstar_from_mh_zu15(_ZU15_LOG10M, **_ZU15_SHMR)
        assert jnp.all(jnp.diff(out) >= 0)


class TestSigmaLnMstarZu15:
    def test_below_pivot_constant(self):
        """For M_h < M_1, scatter equals the baseline value."""
        log10m_h = jnp.array([10.0, 11.0, 11.5])
        sigma = sigma_lnmstar_zu15(log10m_h, lg_m1h=12.10, sigma_lnmstar=0.50, eta=-0.04)
        np.testing.assert_allclose(np.array(sigma), 0.50, atol=1e-6)

    def test_above_pivot_decreasing(self):
        """For M_h > M_1 with eta < 0, scatter decreases above the pivot."""
        log10m_h = jnp.array([12.10, 13.0, 14.0])
        sigma = sigma_lnmstar_zu15(log10m_h, lg_m1h=12.10, sigma_lnmstar=0.50, eta=-0.04)
        assert float(sigma[0]) == pytest.approx(0.50, abs=1e-6)
        assert float(sigma[1]) < float(sigma[0])

    def test_finite(self):
        sigma = sigma_lnmstar_zu15(_ZU15_LOG10M, lg_m1h=12.10, sigma_lnmstar=0.50, eta=-0.04)
        assert jnp.all(jnp.isfinite(sigma))


class TestNCenThreshZu15:
    _P = dict(**_ZU15_SHMR, sigma_lnmstar=0.50, eta=-0.04, fc=0.86)
    _THRESH = 10.2

    def test_range(self):
        nc = n_cen_thresh_zu15(_ZU15_LOG10M, self._THRESH, **self._P)
        assert jnp.all(nc >= 0.0)
        assert jnp.all(nc <= self._P["fc"])

    def test_increases_with_mh(self):
        nc = n_cen_thresh_zu15(_ZU15_LOG10M, self._THRESH, **self._P)
        assert jnp.all(jnp.diff(nc) >= 0.0)

    def test_ceiling_at_fc(self):
        """Very massive halos should host centrals with probability ≈ fc."""
        nc = n_cen_thresh_zu15(jnp.array([16.0]), self._THRESH, **self._P)
        assert float(nc[0]) == pytest.approx(self._P["fc"], abs=0.01)

    def test_finite(self):
        nc = n_cen_thresh_zu15(_ZU15_LOG10M, self._THRESH, **self._P)
        assert jnp.all(jnp.isfinite(nc))

    def test_jit(self):
        nc = n_cen_thresh_zu15(_ZU15_LOG10M, self._THRESH, **self._P)
        assert nc.shape == _ZU15_LOG10M.shape


class TestNSatThreshZu15:
    _P = dict(
        **_ZU15_SHMR,
        sigma_lnmstar=0.50, eta=-0.04, fc=0.86,
        bsat=8.98, beta_sat=0.90, bcut=0.86, beta_cut=0.41, alpha_sat=1.00,
    )
    _THRESH = 10.2

    def test_non_negative(self):
        ns = n_sat_thresh_zu15(_ZU15_LOG10M, self._THRESH, **self._P)
        assert jnp.all(ns >= 0.0)

    def test_finite(self):
        ns = n_sat_thresh_zu15(_ZU15_LOG10M, self._THRESH, **self._P)
        assert jnp.all(jnp.isfinite(ns))

    def test_sat_le_cen_at_low_mass(self):
        """Satellite occupation must stay below central at low halo mass."""
        log10m_low = jnp.array([11.0, 11.5, 12.0])
        nc = n_cen_thresh_zu15(log10m_low, self._THRESH,
                               lg_m1h=self._P["lg_m1h"], lg_m0star=self._P["lg_m0star"],
                               beta=self._P["beta"], delta=self._P["delta"],
                               gamma=self._P["gamma"], sigma_lnmstar=self._P["sigma_lnmstar"],
                               eta=self._P["eta"], fc=self._P["fc"])
        ns = n_sat_thresh_zu15(log10m_low, self._THRESH, **self._P)
        assert jnp.all(ns <= nc + 1e-8)

    def test_total_ge_cen(self):
        nc = n_cen_thresh_zu15(_ZU15_LOG10M, self._THRESH,
                               lg_m1h=self._P["lg_m1h"], lg_m0star=self._P["lg_m0star"],
                               beta=self._P["beta"], delta=self._P["delta"],
                               gamma=self._P["gamma"], sigma_lnmstar=self._P["sigma_lnmstar"],
                               eta=self._P["eta"], fc=self._P["fc"])
        nt = n_total_thresh_zu15(_ZU15_LOG10M, self._THRESH, **self._P)
        assert jnp.all(nt >= nc - 1e-8)


class TestZuMandelbaum15HODModel:
    @pytest.fixture(scope="class")
    def setup(self):
        from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
        from hod_mod.cosmology.halo_mass_function import make_hmf
        pk = LinearPowerSpectrum()
        theta = LinearPowerSpectrum.default_cosmology()
        hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
        model = ZuMandelbaum15HODModel(hmf)
        params = ZuMandelbaum15HODModel.default_params()
        return model, params, theta

    def test_number_density_positive(self, setup):
        model, params, theta = setup
        n = model.galaxy_number_density(0.1, theta, params)
        assert float(n) > 0.0

    def test_number_density_finite(self, setup):
        model, params, theta = setup
        n = model.galaxy_number_density(0.1, theta, params)
        assert jnp.isfinite(n)

    def test_effective_bias_gt1(self, setup):
        model, params, theta = setup
        b = model.effective_bias(0.1, theta, params)
        assert float(b) > 1.0

    def test_effective_mass_finite(self, setup):
        model, params, theta = setup
        m = model.effective_mass(0.1, theta, params)
        assert jnp.isfinite(m)

    def test_effective_mass_range(self, setup):
        model, params, theta = setup
        m = model.effective_mass(0.1, theta, params)
        assert 1e10 < float(m) < 1e16

    def test_default_params_keys(self):
        p = ZuMandelbaum15HODModel.default_params()
        required = {"log10m_star_thresh", "lg_m1h", "lg_m0star", "beta", "delta",
                    "gamma", "sigma_lnmstar", "eta", "fc",
                    "bsat", "beta_sat", "bcut", "beta_cut", "alpha_sat"}
        assert required.issubset(p.keys())


# ---------------------------------------------------------------------------
# Zu & Mandelbaum 2016/2017 halo quenching
# ---------------------------------------------------------------------------

_ZU16_LOG10M = jnp.linspace(10.0, 15.0, 80)
_ZU16_QUENCH = dict(lg_mqc_h=11.78, mu_c=0.41, lg_mqs_h=12.19, mu_s=0.24)


class TestFRedCenZu16:
    def test_range(self):
        fred = f_red_cen_zu16(_ZU16_LOG10M, **{k: v for k, v in _ZU16_QUENCH.items()
                                               if k in ("lg_mqc_h", "mu_c")})
        assert jnp.all(fred >= 0.0)
        assert jnp.all(fred <= 1.0)

    def test_increasing(self):
        fred = f_red_cen_zu16(_ZU16_LOG10M, lg_mqc_h=11.78, mu_c=0.41)
        assert jnp.all(jnp.diff(fred) >= 0.0)

    def test_half_at_pivot(self):
        """f_red_cen = 1 - exp(-1) ≈ 0.632 at M_h = M_h^{qc}."""
        fred = f_red_cen_zu16(jnp.array([11.78]), lg_mqc_h=11.78, mu_c=1.0)
        assert float(fred[0]) == pytest.approx(1.0 - np.exp(-1.0), abs=1e-5)

    def test_finite(self):
        fred = f_red_cen_zu16(_ZU16_LOG10M, lg_mqc_h=11.78, mu_c=0.41)
        assert jnp.all(jnp.isfinite(fred))


class TestFRedSatZu16:
    def test_range(self):
        fred = f_red_sat_zu16(_ZU16_LOG10M, lg_mqs_h=12.19, mu_s=0.24)
        assert jnp.all(fred >= 0.0)
        assert jnp.all(fred <= 1.0)

    def test_increasing(self):
        fred = f_red_sat_zu16(_ZU16_LOG10M, lg_mqs_h=12.19, mu_s=0.24)
        assert jnp.all(jnp.diff(fred) >= 0.0)

    def test_finite(self):
        fred = f_red_sat_zu16(_ZU16_LOG10M, lg_mqs_h=12.19, mu_s=0.24)
        assert jnp.all(jnp.isfinite(fred))


class TestZuMandelbaum16QuenchingModel:
    @pytest.fixture(scope="class")
    def setup(self):
        from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
        from hod_mod.cosmology.halo_mass_function import make_hmf
        pk = LinearPowerSpectrum()
        theta = LinearPowerSpectrum.default_cosmology()
        hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
        qmodel = ZuMandelbaum16QuenchingModel(hmf)
        hod_params = ZuMandelbaum15HODModel.default_params()
        quench_params = ZuMandelbaum16QuenchingModel.default_quench_params()
        return qmodel, hod_params, quench_params, theta

    def test_fred_cen_range(self, setup):
        qmodel, hp, qp, theta = setup
        f = qmodel.effective_red_fraction_cen(0.1, theta, hp, qp)
        assert 0.0 < float(f) < 1.0

    def test_fred_sat_range(self, setup):
        qmodel, hp, qp, theta = setup
        f = qmodel.effective_red_fraction_sat(0.1, theta, hp, qp)
        assert 0.0 < float(f) < 1.0

    def test_fred_cen_finite(self, setup):
        qmodel, hp, qp, theta = setup
        f = qmodel.effective_red_fraction_cen(0.1, theta, hp, qp)
        assert jnp.isfinite(f)

    def test_fred_sat_finite(self, setup):
        qmodel, hp, qp, theta = setup
        f = qmodel.effective_red_fraction_sat(0.1, theta, hp, qp)
        assert jnp.isfinite(f)

    def test_default_quench_params_keys(self):
        p = ZuMandelbaum16QuenchingModel.default_quench_params()
        assert set(p.keys()) == {"lg_mqc_h", "mu_c", "lg_mqs_h", "mu_s"}


_SHAM_LOG10M = jnp.linspace(10.0, 15.0, 60)


class TestSmhmGirelli20:
    def test_output_shape(self):
        out = smhm_girelli20(_SHAM_LOG10M, 0.0)
        assert out.shape == _SHAM_LOG10M.shape

    def test_finite_z0(self):
        out = smhm_girelli20(_SHAM_LOG10M, 0.0)
        assert jnp.all(jnp.isfinite(out))

    def test_finite_z1(self):
        out = smhm_girelli20(_SHAM_LOG10M, 1.0)
        assert jnp.all(jnp.isfinite(out))

    def test_mstar_less_than_mhalo(self):
        out = smhm_girelli20(_SHAM_LOG10M, 0.0)
        assert jnp.all(out < _SHAM_LOG10M)

    def test_monotonically_increasing(self):
        out = smhm_girelli20(_SHAM_LOG10M, 0.0)
        assert jnp.all(jnp.diff(out) >= 0.0)

    def test_peak_near_1e12(self):
        m = jnp.linspace(10.0, 14.0, 500)
        mstar = smhm_girelli20(m, 0.0)
        efficiency = mstar - m
        peak_idx = jnp.argmax(efficiency)
        peak_m = float(m[peak_idx])
        assert 11.5 < peak_m < 12.5

    def test_scatter_params_different(self):
        out_ns = smhm_girelli20(_SHAM_LOG10M, 0.3, **_GIRELLI20_NO_SCATTER)
        out_s = smhm_girelli20(_SHAM_LOG10M, 0.3, **_GIRELLI20_SCATTER)
        assert not jnp.allclose(out_ns, out_s)

    def test_redshift_evolution(self):
        out_z0 = smhm_girelli20(_SHAM_LOG10M, 0.0)
        out_z2 = smhm_girelli20(_SHAM_LOG10M, 2.0)
        assert not jnp.allclose(out_z0, out_z2)

    def test_jit(self):
        out = jax.jit(smhm_girelli20)(_SHAM_LOG10M, 0.5)
        assert out.shape == _SHAM_LOG10M.shape


class TestSHAMModelGirelli20:
    def test_init(self):
        m = SHAMModel(parametrisation="girelli20", scatter_dex=0.2)
        assert m.parametrisation == "girelli20"

    def test_invalid_param(self):
        with pytest.raises(ValueError):
            SHAMModel(parametrisation="unknown")

    def test_log10mstar_shape(self):
        m = SHAMModel(parametrisation="girelli20")
        out = m.log10mstar(_SHAM_LOG10M, 0.0)
        assert out.shape == _SHAM_LOG10M.shape

    def test_log10mstar_finite(self):
        m = SHAMModel(parametrisation="girelli20")
        out = m.log10mstar(_SHAM_LOG10M, 0.0)
        assert jnp.all(jnp.isfinite(out))

    def test_sample_shape(self):
        m = SHAMModel(parametrisation="girelli20")
        key = jax.random.PRNGKey(42)
        out = m.sample(_SHAM_LOG10M, 0.0, key)
        assert out.shape == _SHAM_LOG10M.shape

    def test_sample_scatter_applied(self):
        m = SHAMModel(parametrisation="girelli20", scatter_dex=0.3)
        key = jax.random.PRNGKey(0)
        mu = m.log10mstar(_SHAM_LOG10M, 0.0)
        samples = m.sample(_SHAM_LOG10M, 0.0, key)
        residuals = samples - mu
        assert float(jnp.std(residuals)) > 0.1

    def test_all_parametrisations_run(self):
        for p in ("moster13", "behroozi13", "girelli20"):
            m = SHAMModel(parametrisation=p)
            out = m.log10mstar(_SHAM_LOG10M, 0.3)
            assert jnp.all(jnp.isfinite(out))
