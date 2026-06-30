"""Tests for the HOD-based AGN model (HODAgnModel) and its wiring into the
X-ray cross/auto power spectra."""

import numpy as np
import pytest

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import GasDensityDPM
from hod_mod.connection.hod import (
    MoreHODModel,
    MoreConstFincHODModel,
    n_sat_more15_const_finc,
)
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
from hod_mod.agn.xray import XrayAGNModel
from hod_mod.agn.hod import HODAgnModel, BGS_SAMPLES

_THETA = LinearPowerSpectrum.default_cosmology()
_COLOSSUS = dict(
    flat=True, H0=_THETA["h"] * 100.0, Om0=_THETA["Omega_m"],
    Ob0=_THETA["Omega_b"], sigma8=0.811, ns=_THETA["n_s"],
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pk_lin():
    return LinearPowerSpectrum()


@pytest.fixture(scope="module")
def fhmp(pk_lin):
    hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod = MoreHODModel(hmf, hmf.bias)
    return FullHaloModelPrediction(pk_lin, hod, hp)


@pytest.fixture(scope="module")
def agn(pk_lin):
    cfg = BGS_SAMPLES["S1"]
    return HODAgnModel(pk_lin=pk_lin, theta_cosmo=_THETA,
                       z_mean=cfg["z_mean"], z_max=cfg["z_max"])


@pytest.fixture(scope="module")
def cross_hod(fhmp, agn):
    dp = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80)
    return HaloModelCrossSpectra(fhmp, density_profile=dp, agn_model=agn)


@pytest.fixture(scope="module")
def hod_params():
    return MoreHODModel.default_params()


# ---------------------------------------------------------------------------
# AGN occupation
# ---------------------------------------------------------------------------

class TestAgnOccupation:

    def test_ncen_bounded_by_finc(self, agn):
        log10m = np.linspace(10, 16, 200)
        nc, ns = agn.nc_ns_agn(log10m)
        finc = agn._hod_params["f_inc"]
        assert np.all(nc <= finc + 1e-6)   # float32 jax tolerance
        assert np.all(nc >= 0.0)

    def test_ncen_approaches_finc_at_high_mass(self, agn):
        nc, _ = agn.nc_ns_agn(np.array([15.5]))
        assert nc[0] == pytest.approx(agn._hod_params["f_inc"], rel=1e-3)

    def test_nsat_zero_below_threshold(self, agn):
        p = agn._hod_params
        log10m_thresh = np.log10(p["kappa"] * 10.0 ** p["log10mmin"])
        ns = float(n_sat_more15_const_finc(
            log10m_thresh - 0.01, p["log10mmin"], p["sigma_logm"],
            p["log10m1"], p["alpha"], p["kappa"], p["f_inc"]))
        assert ns == 0.0

    def test_model_default_params(self):
        p = MoreConstFincHODModel.default_params()
        assert p["log10m1"] == pytest.approx(p["log10mmin"] + 1.5)


# ---------------------------------------------------------------------------
# Interface parity
# ---------------------------------------------------------------------------

class TestInterface:

    def test_emissivity_shape_and_flat_in_k(self, agn):
        k = np.logspace(-3, 1, 32)
        m = np.logspace(11, 15, 40)
        uk = agn.agn_emissivity_uk(k, m, 0.135, _THETA)
        assert uk.shape == (32, 40)
        assert np.all(np.isfinite(uk))
        # flat in k: every row identical
        assert np.allclose(uk, uk[0][None, :])

    def test_mean_lx_monotonic_in_mass(self, agn):
        m = np.logspace(11, 15, 50)
        lx = agn.mean_agn_log10lx(m, 0.135)
        assert np.all(np.diff(lx) >= -1e-9)

    def test_mean_agn_lx_consistency(self, agn):
        m = np.logspace(12, 14, 10)
        assert np.allclose(agn.mean_agn_lx(m, 0.135),
                           10.0 ** agn.mean_agn_log10lx(m, 0.135))


# ---------------------------------------------------------------------------
# Abundance matching / selection
# ---------------------------------------------------------------------------

class TestAbundanceMatch:

    def test_selection_edges_in_rmag_range(self, agn):
        # The selected soft-L_X floor/ceil should correspond to r_mag inside
        # the [16, 19.5] window (by construction of the selection).
        a, b = agn._alpha_ox_coeffs
        for log10lx in (agn._lx_soft_floor, agn._lx_soft_ceil):
            fx = 10.0 ** log10lx / (4.0 * np.pi * agn._dl_cm ** 2)
            r = a + b * np.log10(fx)
            assert agn._r_mag_range[0] - 0.1 <= r <= agn._r_mag_range[1] + 0.1

    def test_mean_observed_positive(self, agn):
        assert agn.mean_observed_lx() > 0
        assert agn.mean_observed_fx() > 0

    def test_mean_observed_flux_in_selection(self, agn):
        a, b = agn._alpha_ox_coeffs
        r = a + b * np.log10(agn.mean_observed_fx())
        assert agn._r_mag_range[0] <= r <= agn._r_mag_range[1]


# ---------------------------------------------------------------------------
# Cross / auto power spectra wiring
# ---------------------------------------------------------------------------

class TestCrossSpectra:

    def test_detects_hod_agn(self, cross_hod):
        assert cross_hod._agn_has_hod is True

    def test_gX_total_geq_gas(self, cross_hod, hod_params):
        t = cross_hod._pk_tables_gX(0.135, _THETA, hod_params)
        pgX = np.exp(np.array(t["log_pgX"]))
        pgas = np.exp(np.array(t["log_pgX_gas"]))
        pagn = np.exp(np.array(t["log_pgX_agn"]))
        assert np.all(np.isfinite(pgX))
        assert np.all(pgX >= pgas - 1e-10 * pgX)
        assert np.all(pagn > 0)

    def test_XX_decomposition(self, cross_hod):
        t = cross_hod._pk_tables_XX(0.135, _THETA)
        pXX = np.exp(np.array(t["log_pXX"]))
        gg = np.exp(np.array(t["log_pXX_gas_gas"]))
        cr = np.exp(np.array(t["log_pXX_cross"]))
        aa = np.exp(np.array(t["log_pXX_agn_agn"]))
        assert np.allclose(pXX, gg + cr + aa, rtol=1e-5)

    def test_finc_zero_kills_agn_term(self, cross_hod, agn, hod_params):
        # With f_inc -> 0 the AGN occupation vanishes, so the AGN cross term
        # must drop far below the f_inc=0.1 value.
        t1 = cross_hod._pk_tables_gX(0.135, _THETA, hod_params)
        pagn1 = np.exp(np.array(t1["log_pgX_agn"]))
        saved = agn._hod_params
        try:
            agn._hod_params = dict(saved, f_inc=0.0)
            t0 = cross_hod._pk_tables_gX(0.135, _THETA, hod_params)
            pagn0 = np.exp(np.array(t0["log_pgX_agn"]))
        finally:
            agn._hod_params = saved
        assert np.max(pagn0) < 1e-6 * np.max(pagn1)


# ---------------------------------------------------------------------------
# Regression: a non-HOD AGN model uses the legacy point-source path
# ---------------------------------------------------------------------------

class TestLegacyRegression:

    def test_non_hod_agn_uses_legacy_path(self, fhmp, hod_params):
        dp = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80)
        cx = HaloModelCrossSpectra(fhmp, density_profile=dp,
                                   agn_model=XrayAGNModel())
        assert cx._agn_has_hod is False
        t = cx._pk_tables_gX(0.135, _THETA, hod_params)
        assert np.all(np.isfinite(np.array(t["log_pgX"])))
        assert np.all(np.exp(np.array(t["log_pgX_agn"])) > 0)
        tXX = cx._pk_tables_XX(0.135, _THETA)
        assert np.all(np.isfinite(np.array(tXX["log_pXX"])))
