"""Tests for Leauthaud+2012 HOD model."""

import pytest
import numpy as np
import jax.numpy as jnp

from hod_mod.galaxies.hod import (
    Leauthaud12HODModel,
    _mh_from_mstar_leauthaud12,
    _mstar_from_mh_leauthaud12,
    n_cen_leauthaud12,
    n_sat_leauthaud12,
)
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf


_THETA = LinearPowerSpectrum.default_cosmology()

_P = Leauthaud12HODModel.default_params()

_LOG10M = jnp.linspace(10.0, 15.0, 200)


class TestSHMR:
    def test_shmr_monotone(self):
        log10m_star = jnp.linspace(8.0, 12.0, 100)
        log10m_h = _mh_from_mstar_leauthaud12(
            log10m_star, _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
        )
        assert jnp.all(jnp.diff(log10m_h) > 0), "SHMR must be monotonically increasing"

    def test_shmr_inversion_roundtrip(self):
        log10m_h_in = jnp.linspace(11.0, 15.0, 50)
        log10m_star = _mstar_from_mh_leauthaud12(
            log10m_h_in, _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
        )
        log10m_h_out = _mh_from_mstar_leauthaud12(
            log10m_star, _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
        )
        np.testing.assert_allclose(
            np.asarray(log10m_h_in), np.asarray(log10m_h_out), atol=0.01,
            err_msg="SHMR inversion roundtrip should recover log10 M_h to 0.01 dex"
        )


class TestLeauthaud12Occupation:
    def test_ncen_range(self):
        nc = n_cen_leauthaud12(
            _LOG10M, _P["log10m_star_thresh"],
            _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
            _P["sigma_logm"],
        )
        assert jnp.all(nc >= 0.0)
        assert jnp.all(nc <= 1.0)

    def test_ncen_low_mass_near_zero(self):
        log10m_low = jnp.array([10.0, 10.5])
        nc = n_cen_leauthaud12(
            log10m_low, _P["log10m_star_thresh"],
            _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
            _P["sigma_logm"],
        )
        assert jnp.all(nc < 0.1), f"N_cen at M=10^10 should be near 0, got {nc}"

    def test_ncen_high_mass_near_one(self):
        log10m_high = jnp.array([15.0])
        nc = n_cen_leauthaud12(
            log10m_high, _P["log10m_star_thresh"],
            _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
            _P["sigma_logm"],
        )
        assert jnp.all(nc > 0.9), f"N_cen at M=10^15 should be near 1, got {nc}"

    def test_nsat_nonnegative(self):
        ns = n_sat_leauthaud12(
            _LOG10M, _P["log10m_star_thresh"],
            _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
            _P["sigma_logm"],
            _P["alpha_sat"], _P["log10m_sat"], _P["log10m_cut"],
        )
        assert jnp.all(ns >= 0.0)

    def test_nsat_finite(self):
        ns = n_sat_leauthaud12(
            _LOG10M, _P["log10m_star_thresh"],
            _P["log10m1"], _P["log10m_star0"],
            _P["beta"], _P["delta"], _P["gamma"],
            _P["sigma_logm"],
            _P["alpha_sat"], _P["log10m_sat"], _P["log10m_cut"],
        )
        assert jnp.all(jnp.isfinite(ns)), "N_sat must be finite everywhere"


class TestLeauthaud12HODModel:
    @pytest.fixture(scope="class")
    def hod(self):
        pk_lin = LinearPowerSpectrum()
        hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        return Leauthaud12HODModel(hmf, hmf.bias)

    def test_nc_ns_shapes(self, hod):
        nc, ns = hod.nc_ns(_LOG10M, _P)
        assert nc.shape == _LOG10M.shape
        assert ns.shape == _LOG10M.shape

    def test_integrate_positive(self, hod):
        n_gal, b_eff, m_eff = hod._integrate(0.3, _THETA, _P)
        assert float(n_gal) > 0.0
        assert float(b_eff) > 0.5
        assert float(m_eff) > 1e11

    def test_higher_threshold_fewer_galaxies(self, hod):
        p_lo = dict(_P, log10m_star_thresh=9.5)
        p_hi = dict(_P, log10m_star_thresh=11.5)
        n_lo = float(hod.galaxy_number_density(0.3, _THETA, p_lo))
        n_hi = float(hod.galaxy_number_density(0.3, _THETA, p_hi))
        assert n_lo > n_hi, "Higher M_star_thresh should give fewer galaxies"

    def test_default_params_keys(self):
        keys = set(_P.keys())
        required = {
            "log10m1", "log10m_star0", "beta", "delta", "gamma",
            "sigma_logm", "log10m_star_thresh",
            "alpha_sat", "log10m_sat", "log10m_cut",
        }
        assert required.issubset(keys)
