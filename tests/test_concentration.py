"""Tests for JAX-native concentration–mass relations."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.cosmology.concentration import (
    c_duffy08,
    c_dutton14,
    c_klypin16,
    c_bhattacharya13,
    c_diemer15,
    _neff_eisenstein_hu,
    ConcentrationModel,
)
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf


_PLANCK18 = {
    "h": 0.6736,
    "Omega_m": 0.3111,
    "Omega_b": 0.0493,
    "Omega_cdm": 0.2607,
    "n_s": 0.9649,
    "ln10^{10}A_s": 3.044,
}

_M = jnp.logspace(11.0, 15.0, 40)


# ---------------------------------------------------------------------------
# c_duffy08
# ---------------------------------------------------------------------------

class TestDuffy08:
    @pytest.mark.parametrize("mdef", ["200c", "vir", "200m"])
    def test_shape(self, mdef):
        c = c_duffy08(_M, z=0.3, mdef=mdef)
        assert c.shape == (40,)

    @pytest.mark.parametrize("mdef", ["200c", "vir", "200m"])
    def test_positive(self, mdef):
        c = c_duffy08(_M, z=0.3, mdef=mdef)
        assert jnp.all(c > 0)

    @pytest.mark.parametrize("mdef", ["200c", "vir", "200m"])
    def test_decreasing_with_mass(self, mdef):
        """Concentration should decrease with halo mass."""
        c = c_duffy08(_M, z=0.3, mdef=mdef)
        assert jnp.all(jnp.diff(c) < 0)

    def test_decreasing_with_redshift(self):
        """Concentration should decrease with redshift."""
        c_z0 = c_duffy08(_M, z=0.0, mdef="200c")
        c_z1 = c_duffy08(_M, z=1.0, mdef="200c")
        assert jnp.all(c_z0 > c_z1)

    def test_invalid_mdef_raises(self):
        with pytest.raises(ValueError, match="mdef"):
            c_duffy08(_M, z=0.0, mdef="999c")

    def test_jit(self):
        c1 = jax.jit(c_duffy08)(_M, z=0.3, mdef="200c")
        c2 = c_duffy08(_M, z=0.3, mdef="200c")
        np.testing.assert_allclose(c1, c2, rtol=1e-5)

    def test_finite(self):
        c = c_duffy08(_M, z=0.5, mdef="200c")
        assert jnp.all(jnp.isfinite(c))


# ---------------------------------------------------------------------------
# c_dutton14
# ---------------------------------------------------------------------------

class TestDutton14:
    @pytest.mark.parametrize("mdef", ["200c", "vir"])
    def test_shape(self, mdef):
        c = c_dutton14(_M, z=0.3, mdef=mdef)
        assert c.shape == (40,)

    @pytest.mark.parametrize("mdef", ["200c", "vir"])
    def test_positive(self, mdef):
        c = c_dutton14(_M, z=0.3, mdef=mdef)
        assert jnp.all(c > 0)

    @pytest.mark.parametrize("mdef", ["200c", "vir"])
    def test_decreasing_with_mass(self, mdef):
        c = c_dutton14(_M, z=0.3, mdef=mdef)
        assert jnp.all(jnp.diff(c) < 0)

    def test_decreasing_with_redshift(self):
        c_z0 = c_dutton14(_M, z=0.0, mdef="200c")
        c_z1 = c_dutton14(_M, z=1.0, mdef="200c")
        assert jnp.all(c_z0 > c_z1)

    def test_invalid_mdef_raises(self):
        with pytest.raises(ValueError, match="mdef"):
            c_dutton14(_M, z=0.0, mdef="200m")

    def test_jit(self):
        c1 = jax.jit(c_dutton14)(_M, z=0.5, mdef="200c")
        c2 = c_dutton14(_M, z=0.5, mdef="200c")
        np.testing.assert_allclose(c1, c2, rtol=1e-5)


# ---------------------------------------------------------------------------
# c_klypin16
# ---------------------------------------------------------------------------

class TestKlypin16:
    @pytest.mark.parametrize("mdef", ["200c", "vir"])
    def test_shape(self, mdef):
        c = c_klypin16(_M, z=0.5, mdef=mdef)
        assert c.shape == (40,)

    @pytest.mark.parametrize("mdef", ["200c", "vir"])
    def test_positive(self, mdef):
        c = c_klypin16(_M, z=0.5, mdef=mdef)
        assert jnp.all(c > 0)

    def test_redshift_interpolation(self):
        """Parameters are interpolated between tabulated z values."""
        c_z035 = c_klypin16(_M, z=0.35, mdef="200c")
        c_z04  = c_klypin16(_M, z=0.40, mdef="200c")
        c_z05  = c_klypin16(_M, z=0.50, mdef="200c")
        assert jnp.all(c_z035 > c_z04)
        assert jnp.all(c_z04 > c_z05)

    def test_multiple_redshifts(self):
        for z in [0.0, 0.35, 1.0, 2.15, 4.0]:
            c = c_klypin16(_M, z=z, mdef="200c")
            assert jnp.all(c > 0)

    def test_invalid_mdef_raises(self):
        with pytest.raises(ValueError, match="mdef"):
            c_klypin16(_M, z=0.0, mdef="200m")

    def test_finite(self):
        c = c_klypin16(_M, z=1.0, mdef="200c")
        assert jnp.all(jnp.isfinite(c))


# ---------------------------------------------------------------------------
# c_bhattacharya13
# ---------------------------------------------------------------------------

class TestBhattacharya13:
    @pytest.fixture(scope="class")
    def sigma(self):
        """Synthetic σ(M) array, monotonically decreasing (as in reality)."""
        return jnp.logspace(0.3, -0.3, 40)

    @pytest.mark.parametrize("mdef", ["200c", "vir", "200m"])
    def test_shape(self, sigma, mdef):
        c = c_bhattacharya13(_M, sigma, omega_m=0.311, z=0.3, mdef=mdef)
        assert c.shape == (40,)

    @pytest.mark.parametrize("mdef", ["200c", "vir", "200m"])
    def test_positive(self, sigma, mdef):
        c = c_bhattacharya13(_M, sigma, omega_m=0.311, z=0.3, mdef=mdef)
        assert jnp.all(c > 0)

    @pytest.mark.parametrize("mdef", ["200c", "vir", "200m"])
    def test_finite(self, sigma, mdef):
        c = c_bhattacharya13(_M, sigma, omega_m=0.311, z=0.3, mdef=mdef)
        assert jnp.all(jnp.isfinite(c))

    def test_invalid_mdef_raises(self, sigma):
        with pytest.raises(ValueError, match="mdef"):
            c_bhattacharya13(_M, sigma, omega_m=0.311, z=0.0, mdef="bad")

    def test_jit(self, sigma):
        fn = jax.jit(c_bhattacharya13, static_argnums=(2, 3, 4))
        c1 = fn(_M, sigma, 0.311, 0.3, "200c")
        c2 = c_bhattacharya13(_M, sigma, 0.311, 0.3, "200c")
        np.testing.assert_allclose(c1, c2, rtol=1e-5)


# ---------------------------------------------------------------------------
# c_diemer15
# ---------------------------------------------------------------------------

class TestDiemer15:
    @pytest.fixture(scope="class")
    def sigma_neff(self):
        sigma = jnp.logspace(0.3, -0.3, 40)
        n_eff = jnp.linspace(-2.5, -1.5, 40)
        return sigma, n_eff

    @pytest.mark.parametrize("stat", ["median", "mean"])
    def test_shape(self, sigma_neff, stat):
        sigma, n_eff = sigma_neff
        c = c_diemer15(_M, sigma, n_eff, omega_m=0.311, z=0.3, statistic=stat)
        assert c.shape == (40,)

    @pytest.mark.parametrize("stat", ["median", "mean"])
    def test_positive(self, sigma_neff, stat):
        sigma, n_eff = sigma_neff
        c = c_diemer15(_M, sigma, n_eff, omega_m=0.311, z=0.3, statistic=stat)
        assert jnp.all(c > 0)

    @pytest.mark.parametrize("stat", ["median", "mean"])
    def test_finite(self, sigma_neff, stat):
        sigma, n_eff = sigma_neff
        c = c_diemer15(_M, sigma, n_eff, omega_m=0.311, z=0.3, statistic=stat)
        assert jnp.all(jnp.isfinite(c))

    def test_median_ne_mean(self, sigma_neff):
        sigma, n_eff = sigma_neff
        c_med  = c_diemer15(_M, sigma, n_eff, 0.311, 0.3, "median")
        c_mean = c_diemer15(_M, sigma, n_eff, 0.311, 0.3, "mean")
        assert not jnp.allclose(c_med, c_mean)

    def test_invalid_statistic_raises(self, sigma_neff):
        sigma, n_eff = sigma_neff
        with pytest.raises(ValueError, match="statistic"):
            c_diemer15(_M, sigma, n_eff, 0.311, 0.3, "mode")


# ---------------------------------------------------------------------------
# _neff_eisenstein_hu helper
# ---------------------------------------------------------------------------

class TestNeffEisensteinHu:
    def test_shape(self):
        n = _neff_eisenstein_hu(_M, _PLANCK18)
        assert n.shape == (40,)

    def test_range(self):
        """n_eff should be in (-4, 0) for typical halo masses."""
        n = _neff_eisenstein_hu(_M, _PLANCK18)
        assert jnp.all(n > -4.0)
        assert jnp.all(n < 0.0)

    def test_finite(self):
        n = _neff_eisenstein_hu(_M, _PLANCK18)
        assert jnp.all(jnp.isfinite(n))


# ---------------------------------------------------------------------------
# ConcentrationModel unified interface
# ---------------------------------------------------------------------------

class TestConcentrationModel:
    @pytest.fixture(scope="class")
    def hmf(self):
        pk_lin = LinearPowerSpectrum()
        return make_hmf("tinker08", pk_func=pk_lin.pk_linear)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ConcentrationModel(model="bogus")

    def test_sigma_model_without_hmf_raises(self):
        with pytest.raises(ValueError, match="requires an HMF"):
            ConcentrationModel(model="diemer15")

    def test_sigma_model_without_hmf_raises_bhat(self):
        with pytest.raises(ValueError, match="requires an HMF"):
            ConcentrationModel(model="bhattacharya13")

    @pytest.mark.parametrize("model", ["duffy08", "dutton14", "klypin16"])
    def test_power_law_models(self, model):
        cm = ConcentrationModel(model=model, mdef="200c")
        c = cm.concentration(_M, z=0.3, theta=_PLANCK18)
        assert c.shape == (40,)
        assert jnp.all(c > 0)

    def test_bhattacharya13_with_hmf(self, hmf):
        cm = ConcentrationModel(model="bhattacharya13", mdef="200c", hmf=hmf)
        c = cm.concentration(_M, z=0.3, theta=_PLANCK18)
        assert c.shape == (40,)
        assert jnp.all(c > 0)

    def test_diemer15_with_hmf(self, hmf):
        cm = ConcentrationModel(model="diemer15", mdef="200c", hmf=hmf)
        c = cm.concentration(_M, z=0.3, theta=_PLANCK18)
        assert c.shape == (40,)
        assert jnp.all(c > 0)

    def test_diemer15_mean_statistic(self, hmf):
        cm_med  = ConcentrationModel(model="diemer15", hmf=hmf, statistic="median")
        cm_mean = ConcentrationModel(model="diemer15", hmf=hmf, statistic="mean")
        c_med  = cm_med.concentration(_M, z=0.3, theta=_PLANCK18)
        c_mean = cm_mean.concentration(_M, z=0.3, theta=_PLANCK18)
        assert not jnp.allclose(c_med, c_mean)
