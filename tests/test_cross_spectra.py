"""Tests for HaloModelCrossSpectra (galaxy × tSZ and galaxy × soft X-ray)."""

import numpy as np
import pytest

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.cosmology.gas_profiles import PressureProfileA10, GasDensityDPM
from hod_mod.galaxies.hod import MoreHODModel
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.cross_spectra import HaloModelCrossSpectra

_THETA = LinearPowerSpectrum.default_cosmology()
_Z     = 0.3

_COLOSSUS = {
    "flat": True,
    "H0": _THETA["h"] * 100.0,
    "Om0": _THETA["Omega_m"],
    "Ob0": _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns": _THETA["n_s"],
}


# ---------------------------------------------------------------------------
# Shared fixture: FullHaloModelPrediction + HaloModelCrossSpectra
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fhmp():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    return FullHaloModelPrediction(pk_lin, hod, hp)


@pytest.fixture(scope="module")
def cross_gy(fhmp):
    pp = PressureProfileA10(r_max_over_r500c=4.0, n_gl=80)
    return HaloModelCrossSpectra(fhmp, pressure_profile=pp)


@pytest.fixture(scope="module")
def cross_gX(fhmp):
    dp = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80)
    return HaloModelCrossSpectra(fhmp, density_profile=dp)


@pytest.fixture(scope="module")
def hod_params():
    return MoreHODModel.default_params()


# ---------------------------------------------------------------------------
# TestGalaxyTSZSpectrum
# ---------------------------------------------------------------------------

class TestGalaxyTSZSpectrum:

    def test_pk_tables_gy_keys(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        for key in ["log_k", "log_pgy", "log_pgy_1h", "log_pgy_2h", "log_pmy", "n_gal", "b_eff"]:
            assert key in tables, f"Missing key: {key}"

    def test_pgy_finite(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.isfinite(tables["log_pgy"]))

    def test_pgy_1h_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.exp(tables["log_pgy_1h"]) > 0)

    def test_pgy_2h_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.exp(tables["log_pgy_2h"]) > 0)

    def test_pgy_total_geq_components(self, cross_gy, hod_params):
        """P_gy ≥ max(P_gy_1h, P_gy_2h) at every k."""
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        pgy    = np.exp(np.array(tables["log_pgy"]))
        pgy_1h = np.exp(np.array(tables["log_pgy_1h"]))
        pgy_2h = np.exp(np.array(tables["log_pgy_2h"]))
        assert np.all(pgy >= pgy_1h - 1e-10 * pgy)
        assert np.all(pgy >= pgy_2h - 1e-10 * pgy)

    def test_pgy_2h_dominates_at_low_k(self, cross_gy, hod_params):
        """At k=0.01 h/Mpc the 2-halo term should dominate."""
        tables  = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        k       = np.exp(np.array(tables["log_k"]))
        idx_lo  = np.argmin(np.abs(k - 0.01))
        pgy_1h  = float(np.exp(tables["log_pgy_1h"][idx_lo]))
        pgy_2h  = float(np.exp(tables["log_pgy_2h"][idx_lo]))
        assert pgy_2h > pgy_1h

    def test_pmy_finite(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.isfinite(tables["log_pmy"]))

    def test_ngal_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert tables["n_gal"] > 0

    def test_beff_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert 0.5 < tables["b_eff"] < 10.0

    def test_gas_cache_populated(self, cross_gy, hod_params):
        """After one call, gas profile FT should be cached."""
        _ = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert len(cross_gy._gas_cache) > 0

    def test_no_pressure_profile_raises(self, fhmp, hod_params):
        cross_no_pp = HaloModelCrossSpectra(fhmp)  # no pressure_profile
        with pytest.raises(RuntimeError, match="No pressure_profile"):
            cross_no_pp._pk_tables_gy(_Z, _THETA, hod_params)


# ---------------------------------------------------------------------------
# TestGalaxyXRaySpectrum
# ---------------------------------------------------------------------------

class TestGalaxyXRaySpectrum:

    def test_pk_tables_gX_keys(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        for key in ["log_k", "log_pgX", "log_pgX_1h", "log_pgX_2h", "n_gal", "b_eff"]:
            assert key in tables

    def test_pgX_finite(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        assert np.all(np.isfinite(tables["log_pgX"]))

    def test_pgX_1h_2h_positive(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        assert np.all(np.exp(tables["log_pgX_1h"]) > 0)
        assert np.all(np.exp(tables["log_pgX_2h"]) > 0)

    def test_pgX_total_geq_components(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        pgX    = np.exp(np.array(tables["log_pgX"]))
        pgX_1h = np.exp(np.array(tables["log_pgX_1h"]))
        pgX_2h = np.exp(np.array(tables["log_pgX_2h"]))
        assert np.all(pgX >= pgX_1h - 1e-10 * pgX)
        assert np.all(pgX >= pgX_2h - 1e-10 * pgX)

    def test_no_density_profile_raises(self, fhmp, hod_params):
        cross_no_dp = HaloModelCrossSpectra(fhmp)
        with pytest.raises(RuntimeError, match="No density_profile"):
            cross_no_dp._pk_tables_gX(_Z, _THETA, hod_params)


# ---------------------------------------------------------------------------
# TestProjectedSignal
# ---------------------------------------------------------------------------

class TestProjectedSignalGY:

    def test_projected_gy_shape(self, cross_gy, hod_params):
        rp = np.logspace(-1, 1.5, 10)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert sigma_y.shape == (10,)

    def test_projected_gy_positive(self, cross_gy, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert np.all(sigma_y > 0)

    def test_projected_gy_finite(self, cross_gy, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert np.all(np.isfinite(sigma_y))

    def test_projected_gy_decreasing(self, cross_gy, hod_params):
        """Σ_y(r_p) should decrease monotonically with r_p at r_p > 0.3 Mpc/h."""
        rp = np.logspace(-0.5, 1.3, 12)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert np.all(np.diff(sigma_y) < 0), "projected tSZ signal should be monotonically decreasing"


class TestProjectedSignalGX:

    def test_projected_gX_shape(self, cross_gX, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        wgx = cross_gX.projected_gX(rp, _Z, _THETA, hod_params)
        assert wgx.shape == (8,)

    def test_projected_gX_positive(self, cross_gX, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        wgx = cross_gX.projected_gX(rp, _Z, _THETA, hod_params)
        assert np.all(wgx > 0)

    def test_projected_gX_decreasing(self, cross_gX, hod_params):
        rp = np.logspace(-0.5, 1.3, 10)
        wgx = cross_gX.projected_gX(rp, _Z, _THETA, hod_params)
        assert np.all(np.diff(wgx) < 0)


# ---------------------------------------------------------------------------
# TestAngularPowerSpectrum
# ---------------------------------------------------------------------------

class TestAngularPowerSpectrumGY:

    def test_cl_gy_shape(self, cross_gy, hod_params):
        ell  = np.logspace(1, 4, 10)
        z_arr = np.linspace(0.2, 0.5, 8)
        nz_g  = np.exp(-0.5 * ((z_arr - 0.3) / 0.05)**2)
        cl_gy = cross_gy.angular_cl_gy(ell, z_arr, nz_g, _THETA, hod_params)
        assert cl_gy.shape == (10,)

    def test_cl_gy_positive(self, cross_gy, hod_params):
        ell  = np.logspace(1, 4, 8)
        z_arr = np.linspace(0.2, 0.5, 8)
        nz_g  = np.exp(-0.5 * ((z_arr - 0.3) / 0.05)**2)
        cl_gy = cross_gy.angular_cl_gy(ell, z_arr, nz_g, _THETA, hod_params)
        assert np.all(cl_gy > 0)

    def test_cl_gy_finite(self, cross_gy, hod_params):
        ell  = np.logspace(2, 3.5, 6)
        z_arr = np.linspace(0.25, 0.45, 6)
        nz_g  = np.exp(-0.5 * ((z_arr - 0.35) / 0.04)**2)
        cl_gy = cross_gy.angular_cl_gy(ell, z_arr, nz_g, _THETA, hod_params)
        assert np.all(np.isfinite(cl_gy))
