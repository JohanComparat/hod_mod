"""Tests for the CLF models (Cacciato+2009 and van den Bosch+2013)."""

import pytest
import numpy as np
import jax.numpy as jnp

from hod_mod.connection.clf import (
    log10_lc,
    clf_central_mean,
    phi_sat_cacciato09,
    alpha_sat_cacciato09,
    l_s_star_log10,
    clf_satellite_mean,
    clf_satellite_mean_vdB13,
    CLFModel,
    VanDenBosch13CLFModel,
    BASILISKCLFModel,
)
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf


_THETA = LinearPowerSpectrum.default_cosmology()

_CLF09_PARAMS = CLFModel.default_params()
_VDB13_PARAMS = VanDenBosch13CLFModel.default_params()

_PHI_SAT_PARAMS   = {"b_0": -0.766, "b_1": 1.008, "b_2": -0.094}
_ALPHA_SAT_PARAMS = {"a_1":  0.501, "a_2":   2.49, "log_m_2": 14.0}

_LOG10M = jnp.linspace(10.0, 15.0, 200)


# ---------------------------------------------------------------------------
# Standalone function tests — Cacciato+2009 helpers
# ---------------------------------------------------------------------------

class TestCLF09Helpers:
    def test_log10_lc_increasing(self):
        lc = log10_lc(
            _LOG10M,
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
        )
        assert jnp.all(jnp.diff(lc) > 0), "L_c(M) should be monotonically increasing"

    def test_phi_sat_nonnegative(self):
        phi = phi_sat_cacciato09(_LOG10M, _PHI_SAT_PARAMS["b_0"],
                                 _PHI_SAT_PARAMS["b_1"], _PHI_SAT_PARAMS["b_2"])
        assert jnp.all(phi >= 0.0)
        assert jnp.all(jnp.isfinite(phi))

    def test_phi_sat_at_pivot(self):
        # At log10(M) = 12, phi_sat = 10^b_0
        val = phi_sat_cacciato09(jnp.array([12.0]), -0.766, 1.008, -0.094)
        expected = 10.0 ** (-0.766)
        assert abs(float(val[0]) - expected) < 1e-6

    def test_alpha_sat_range(self):
        # With default params: alpha_s -> -2+2*a_1 ~ -0.998 at low M, -> -2 at high M
        a_low  = float(alpha_sat_cacciato09(jnp.array(10.0),
                        _ALPHA_SAT_PARAMS["a_1"], _ALPHA_SAT_PARAMS["a_2"], _ALPHA_SAT_PARAMS["log_m_2"]))
        a_high = float(alpha_sat_cacciato09(jnp.array(16.0),
                        _ALPHA_SAT_PARAMS["a_1"], _ALPHA_SAT_PARAMS["a_2"], _ALPHA_SAT_PARAMS["log_m_2"]))
        assert a_low > a_high, "alpha_s should decrease with halo mass"
        assert -1.1 < a_low < -0.9, f"Low-mass alpha_s = {a_low}, expected ~-0.998"
        assert -2.1 < a_high < -1.9, f"High-mass alpha_s = {a_high}, expected ~-2.0"

    def test_l_s_star_ratio(self):
        lc   = log10_lc(_LOG10M, _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
                        _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"])
        ls   = l_s_star_log10(_LOG10M, _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
                               _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"])
        ratio = jnp.power(10.0, ls - lc)
        assert jnp.allclose(ratio, 0.562, atol=1e-5), "L_s* = 0.562 × L_c"


# ---------------------------------------------------------------------------
# Central CLF
# ---------------------------------------------------------------------------

class TestCLFCentral:
    def test_central_range(self):
        nc = clf_central_mean(
            _LOG10M,
            _CLF09_PARAMS["log10l_lim"],
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
            _CLF09_PARAMS["sigma_c"],
        )
        assert jnp.all(nc >= 0.0)
        assert jnp.all(nc <= 1.0)

    def test_central_monotone(self):
        nc = clf_central_mean(
            _LOG10M,
            _CLF09_PARAMS["log10l_lim"],
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
            _CLF09_PARAMS["sigma_c"],
        )
        assert jnp.all(jnp.diff(nc) >= -1e-6)


# ---------------------------------------------------------------------------
# Satellite CLF — Cacciato+2009
# ---------------------------------------------------------------------------

class TestCLF09Satellite:
    def test_nonnegative(self):
        ns = clf_satellite_mean(
            _LOG10M, _CLF09_PARAMS["log10l_lim"],
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
            _CLF09_PARAMS["sigma_c"], _CLF09_PARAMS["alpha_sat"], _CLF09_PARAMS["b_sat"],
        )
        assert jnp.all(ns >= 0.0), "N_sat must be non-negative"

    def test_finite(self):
        ns = clf_satellite_mean(
            _LOG10M, _CLF09_PARAMS["log10l_lim"],
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
            _CLF09_PARAMS["sigma_c"], _CLF09_PARAMS["alpha_sat"], _CLF09_PARAMS["b_sat"],
        )
        assert jnp.all(jnp.isfinite(ns)), "N_sat should be finite everywhere"

    def test_increasing_with_mass(self):
        ns = clf_satellite_mean(
            _LOG10M, _CLF09_PARAMS["log10l_lim"],
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
            _CLF09_PARAMS["sigma_c"], _CLF09_PARAMS["alpha_sat"], _CLF09_PARAMS["b_sat"],
        )
        ns_hi = float(ns[len(ns)//2:].mean())
        ns_lo = float(ns[:len(ns)//4].mean())
        assert ns_hi > ns_lo, "N_sat should be larger at higher masses on average"

    def test_higher_threshold_fewer_sats(self):
        p_lo = dict(_CLF09_PARAMS, log10l_lim=9.0)
        p_hi = dict(_CLF09_PARAMS, log10l_lim=10.5)
        ns_lo = clf_satellite_mean(
            _LOG10M, p_lo["log10l_lim"],
            p_lo["log10l0"], p_lo["log10m1"],
            p_lo["alpha_cen"], p_lo["beta_cen"],
            p_lo["sigma_c"], p_lo["alpha_sat"], p_lo["b_sat"],
        )
        ns_hi = clf_satellite_mean(
            _LOG10M, p_hi["log10l_lim"],
            p_hi["log10l0"], p_hi["log10m1"],
            p_hi["alpha_cen"], p_hi["beta_cen"],
            p_hi["sigma_c"], p_hi["alpha_sat"], p_hi["b_sat"],
        )
        assert float(ns_lo.sum()) > float(ns_hi.sum()), \
            "Higher L threshold should yield fewer satellites"

    def test_b_sat_increases_satellites(self):
        ns_lo = clf_satellite_mean(
            _LOG10M, _CLF09_PARAMS["log10l_lim"],
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
            _CLF09_PARAMS["sigma_c"], _CLF09_PARAMS["alpha_sat"], b_sat=0.0,
        )
        ns_hi = clf_satellite_mean(
            _LOG10M, _CLF09_PARAMS["log10l_lim"],
            _CLF09_PARAMS["log10l0"], _CLF09_PARAMS["log10m1"],
            _CLF09_PARAMS["alpha_cen"], _CLF09_PARAMS["beta_cen"],
            _CLF09_PARAMS["sigma_c"], _CLF09_PARAMS["alpha_sat"], b_sat=1.0,
        )
        assert float(ns_hi.sum()) > float(ns_lo.sum()), \
            "Larger b_sat should yield more satellites"


# ---------------------------------------------------------------------------
# Satellite CLF — van den Bosch+2013 simplified
# ---------------------------------------------------------------------------

class TestVdB13Satellite:
    def test_nonnegative(self):
        ns = clf_satellite_mean_vdB13(
            _LOG10M, _VDB13_PARAMS["log10l_lim"],
            _VDB13_PARAMS["log10l0"], _VDB13_PARAMS["log10m1"],
            _VDB13_PARAMS["alpha_cen"], _VDB13_PARAMS["beta_cen"],
            _VDB13_PARAMS["b_0"], _VDB13_PARAMS["b_1"], _VDB13_PARAMS["b_2"],
            _VDB13_PARAMS["alpha_sat"],
        )
        assert jnp.all(ns >= 0.0)
        assert jnp.all(jnp.isfinite(ns))

    def test_constant_alpha_sat_effect(self):
        ns_m1 = clf_satellite_mean_vdB13(
            _LOG10M, _VDB13_PARAMS["log10l_lim"],
            _VDB13_PARAMS["log10l0"], _VDB13_PARAMS["log10m1"],
            _VDB13_PARAMS["alpha_cen"], _VDB13_PARAMS["beta_cen"],
            _VDB13_PARAMS["b_0"], _VDB13_PARAMS["b_1"], _VDB13_PARAMS["b_2"],
            alpha_sat=-1.3,
        )
        ns_m2 = clf_satellite_mean_vdB13(
            _LOG10M, _VDB13_PARAMS["log10l_lim"],
            _VDB13_PARAMS["log10l0"], _VDB13_PARAMS["log10m1"],
            _VDB13_PARAMS["alpha_cen"], _VDB13_PARAMS["beta_cen"],
            _VDB13_PARAMS["b_0"], _VDB13_PARAMS["b_1"], _VDB13_PARAMS["b_2"],
            alpha_sat=-1.18,
        )
        assert not jnp.allclose(ns_m1, ns_m2), "Different alpha_sat should change N_sat"


# ---------------------------------------------------------------------------
# CLFModel class (Cacciato+2009)
# ---------------------------------------------------------------------------

class TestCLFModel:
    @pytest.fixture(scope="class")
    def clf(self):
        pk_lin = LinearPowerSpectrum()
        hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        return CLFModel(hmf, hmf.bias)

    def test_nc_ns_shapes(self, clf):
        nc, ns = clf.nc_ns(_LOG10M, _CLF09_PARAMS)
        assert nc.shape == _LOG10M.shape
        assert ns.shape == _LOG10M.shape

    def test_nc_ns_nonnegative(self, clf):
        nc, ns = clf.nc_ns(_LOG10M, _CLF09_PARAMS)
        assert jnp.all(nc >= 0.0)
        assert jnp.all(ns >= 0.0)

    def test_integrate_positive(self, clf):
        n_gal, b_eff, m_eff = clf._integrate(0.3, _THETA, _CLF09_PARAMS)
        assert float(n_gal) > 0.0,  f"n_gal={n_gal} should be positive"
        assert float(b_eff) > 0.5,  f"b_eff={b_eff} should be > 0.5"
        assert float(m_eff) > 1e11, f"m_eff={m_eff} should be > 10^11 Msun/h"

    def test_galaxy_number_density(self, clf):
        n = clf.galaxy_number_density(0.3, _THETA, _CLF09_PARAMS)
        assert float(n) > 0.0

    def test_effective_bias(self, clf):
        b = clf.effective_bias(0.3, _THETA, _CLF09_PARAMS)
        assert 0.5 < float(b) < 10.0

    def test_higher_threshold_fewer_galaxies(self, clf):
        p_lo = dict(_CLF09_PARAMS, log10l_lim=9.0)
        p_hi = dict(_CLF09_PARAMS, log10l_lim=10.5)
        n_lo = float(clf.galaxy_number_density(0.3, _THETA, p_lo))
        n_hi = float(clf.galaxy_number_density(0.3, _THETA, p_hi))
        assert n_lo > n_hi, "Higher L threshold should give fewer galaxies"

    def test_delta_optional(self, clf):
        # delta_1/delta_2 should be optional (CLFModel.nc_ns uses .get with default 0)
        p_no_delta = {k: v for k, v in _CLF09_PARAMS.items()
                      if k not in ("delta_1", "delta_2")}
        nc, ns = clf.nc_ns(_LOG10M, p_no_delta)
        assert jnp.all(jnp.isfinite(ns))


# ---------------------------------------------------------------------------
# VanDenBosch13CLFModel class
# ---------------------------------------------------------------------------

class TestVanDenBosch13CLFModel:
    @pytest.fixture(scope="class")
    def clf(self):
        pk_lin = LinearPowerSpectrum()
        hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        return VanDenBosch13CLFModel(hmf, hmf.bias)

    def test_nc_ns_shapes(self, clf):
        nc, ns = clf.nc_ns(_LOG10M, _VDB13_PARAMS)
        assert nc.shape == _LOG10M.shape
        assert ns.shape == _LOG10M.shape

    def test_nc_ns_nonnegative(self, clf):
        nc, ns = clf.nc_ns(_LOG10M, _VDB13_PARAMS)
        assert jnp.all(nc >= 0.0)
        assert jnp.all(ns >= 0.0)

    def test_integrate_positive(self, clf):
        n_gal, b_eff, m_eff = clf._integrate(0.3, _THETA, _VDB13_PARAMS)
        assert float(n_gal) > 0.0
        assert float(b_eff) > 0.5
        assert float(m_eff) > 1e11

    def test_higher_threshold_fewer_galaxies(self, clf):
        p_lo = dict(_VDB13_PARAMS, log10l_lim=9.0)
        p_hi = dict(_VDB13_PARAMS, log10l_lim=10.5)
        n_lo = float(clf.galaxy_number_density(0.3, _THETA, p_lo))
        n_hi = float(clf.galaxy_number_density(0.3, _THETA, p_hi))
        assert n_lo > n_hi

    def test_default_params_cacciato13(self):
        p = VanDenBosch13CLFModel.default_params_cacciato13()
        assert abs(p["alpha_sat"] - (-1.18)) < 1e-6
        assert abs(p["sigma_c"] - 0.157) < 1e-6

    def test_default_params_cacciato14(self):
        p = VanDenBosch13CLFModel.default_params_cacciato14()
        assert abs(p["alpha_cen"] - 3.18) < 1e-6
        assert abs(p["sigma_c"] - 0.146) < 1e-6

    def test_cacciato13_params_integrate(self, clf):
        p = VanDenBosch13CLFModel.default_params_cacciato13()
        n_gal, b_eff, _ = clf._integrate(0.1, _THETA, p)
        assert float(n_gal) > 0.0
        assert 0.5 < float(b_eff) < 10.0

    def test_cacciato14_params_integrate(self, clf):
        p = VanDenBosch13CLFModel.default_params_cacciato14()
        n_gal, b_eff, _ = clf._integrate(0.3, _THETA, p)
        assert float(n_gal) > 0.0
        assert 0.5 < float(b_eff) < 10.0


# ---------------------------------------------------------------------------
# BASILISK III flexible CLF model
# ---------------------------------------------------------------------------

_LOG10M = jnp.linspace(10.0, 15.0, 100)


@pytest.fixture(scope="module")
def basilisk_clf():
    pk_lin = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    return BASILISKCLFModel(hmf, hmf.bias)


_BSLK_PARAMS = BASILISKCLFModel.default_params()


class TestBASILISKCLFModel:
    def test_nc_ns_shapes(self, basilisk_clf):
        nc, ns = basilisk_clf.nc_ns(_LOG10M, _BSLK_PARAMS)
        assert nc.shape == _LOG10M.shape
        assert ns.shape == _LOG10M.shape

    def test_nc_ns_nonnegative(self, basilisk_clf):
        nc, ns = basilisk_clf.nc_ns(_LOG10M, _BSLK_PARAMS)
        assert jnp.all(nc >= 0.0)
        assert jnp.all(ns >= 0.0)
        assert jnp.all(jnp.isfinite(nc))
        assert jnp.all(jnp.isfinite(ns))

    def test_model_a_matches_vdB13(self, basilisk_clf):
        # With all slope params = 0 and delta_13 = -0.25, must reproduce VdB13
        vdb_clf = VanDenBosch13CLFModel(basilisk_clf._hmf, basilisk_clf._bias)
        vdb_p = VanDenBosch13CLFModel.default_params()
        nc_vdb, ns_vdb = vdb_clf.nc_ns(_LOG10M, vdb_p)

        bslk_p = dict(_BSLK_PARAMS)   # Model A defaults: all slopes = 0
        nc_b, ns_b = basilisk_clf.nc_ns(_LOG10M, bslk_p)

        assert jnp.allclose(nc_vdb, nc_b, atol=1e-5), "N_c must match VdB13 (Model A)"
        assert jnp.allclose(ns_vdb, ns_b, rtol=1e-3), "N_s must match VdB13 (Model A) to 0.1%"

    def test_sigma_p1_effect(self, basilisk_clf):
        p_flat  = dict(_BSLK_PARAMS, sigma_p1=0.0)
        p_slop  = dict(_BSLK_PARAMS, sigma_p1=0.1)
        nc_flat, _ = basilisk_clf.nc_ns(_LOG10M, p_flat)
        nc_slop, _ = basilisk_clf.nc_ns(_LOG10M, p_slop)
        assert not jnp.allclose(nc_flat, nc_slop), "sigma_p1 must change N_c"

    def test_alpha_p1_effect(self, basilisk_clf):
        p_flat  = dict(_BSLK_PARAMS, alpha_p1=0.0)
        p_slop  = dict(_BSLK_PARAMS, alpha_p1=0.2)
        _, ns_flat = basilisk_clf.nc_ns(_LOG10M, p_flat)
        _, ns_slop = basilisk_clf.nc_ns(_LOG10M, p_slop)
        assert not jnp.allclose(ns_flat, ns_slop), "alpha_p1 must change N_s"

    def test_delta_p1_effect(self, basilisk_clf):
        p_flat  = dict(_BSLK_PARAMS, delta_p1=0.0)
        p_slop  = dict(_BSLK_PARAMS, delta_p1=0.1)
        _, ns_flat = basilisk_clf.nc_ns(_LOG10M, p_flat)
        _, ns_slop = basilisk_clf.nc_ns(_LOG10M, p_slop)
        assert not jnp.allclose(ns_flat, ns_slop), "delta_p1 must change N_s"

    def test_free_delta_increases_nsat(self, basilisk_clf):
        p_std  = dict(_BSLK_PARAMS, delta_13=-0.25)
        p_high = dict(_BSLK_PARAMS, delta_13=0.0)
        _, ns_std  = basilisk_clf.nc_ns(_LOG10M, p_std)
        _, ns_high = basilisk_clf.nc_ns(_LOG10M, p_high)
        assert jnp.all(ns_high >= ns_std), "Larger delta_13 must increase N_sat"

    def test_integrate_positive(self, basilisk_clf):
        n_gal, b_eff, m_eff = basilisk_clf._integrate(0.1, _THETA, _BSLK_PARAMS)
        assert float(n_gal) > 0.0
        assert float(b_eff) > 0.0
        assert float(m_eff) > 0.0
        assert jnp.isfinite(n_gal)
        assert jnp.isfinite(b_eff)

    def test_higher_threshold_fewer_galaxies(self, basilisk_clf):
        p_lo = dict(_BSLK_PARAMS, log10l_lim=9.5)
        p_hi = dict(_BSLK_PARAMS, log10l_lim=10.0)
        n_lo, _, _ = basilisk_clf._integrate(0.1, _THETA, p_lo)
        n_hi, _, _ = basilisk_clf._integrate(0.1, _THETA, p_hi)
        assert float(n_hi) < float(n_lo), "Higher threshold must give fewer galaxies"
