"""Functional tests for the Kravtsov+2004 HOD (connection/hod/kravtsov04.py).

Previously only import-smoke-tested via test_public_api; the AUM alias is
compared against the aum C++ code in test_aum_comparison, but the class and
standalone functions had no direct functional coverage.
"""

import numpy as np
import pytest
import jax.numpy as jnp

_LOG10M = jnp.linspace(11.0, 16.0, 300)


class TestNSatKravtsov04:
    def test_power_law_slope_above_cutoff(self):
        from hod_mod.connection.hod.kravtsov04 import n_sat_kravtsov04

        # far above M_min and M_0 the occupation is a pure power law:
        # dlog10 N_sat / dlog10 M -> alpha  (residual exp(-M0/M) bend at
        # lm=14.5 with log10m0=10 is ~1e-4 in slope)
        p = dict(log10mmin=12.0, sigma_logm=0.3, log10m0=10.0,
                 log10m1=13.0, alpha=1.1)
        lm = jnp.linspace(14.5, 16.0, 50)
        ns = n_sat_kravtsov04(lm, **p)
        slope = np.diff(np.log10(np.asarray(ns))) / np.diff(np.asarray(lm))
        np.testing.assert_allclose(slope, p["alpha"], rtol=1e-3)

    def test_amplitude_at_m1(self):
        from hod_mod.connection.hod.kravtsov04 import n_sat_kravtsov04

        # at M = M_1 >> M_min, M_0: N_sat = N_cen * 1^alpha * exp(-M0/M1) ~ 1
        ns = n_sat_kravtsov04(jnp.array([15.0]), 12.0, 0.3, 12.0, 15.0, 1.0)
        expected = np.exp(-10.0 ** (12.0 - 15.0))
        assert float(ns[0]) == pytest.approx(expected, rel=1e-4)

    def test_exponential_cutoff_suppresses_low_mass(self):
        from hod_mod.connection.hod.kravtsov04 import n_sat_kravtsov04

        base = dict(log10mmin=11.0, sigma_logm=0.2, log10m1=13.0, alpha=1.0)
        ns_soft = n_sat_kravtsov04(_LOG10M, log10m0=11.0, **base)
        ns_hard = n_sat_kravtsov04(_LOG10M, log10m0=13.5, **base)
        # a higher M_0 suppresses satellites everywhere, most strongly at low M
        assert jnp.all(ns_hard <= ns_soft)
        i_lo = int(jnp.argmin(jnp.abs(_LOG10M - 12.0)))
        assert float(ns_hard[i_lo] / ns_soft[i_lo]) < 1e-10
        # and leaves the high-mass end nearly untouched
        assert float(ns_hard[-1] / ns_soft[-1]) == pytest.approx(
            np.exp(-10.0 ** (13.5 - 16.0)) / np.exp(-10.0 ** (11.0 - 16.0)),
            rel=1e-6,
        )

    def test_aum_alias(self):
        from hod_mod.connection.hod.kravtsov04 import (
            n_sat_aum, n_sat_kravtsov04, n_total_aum, n_total_kravtsov04)

        assert n_sat_aum is n_sat_kravtsov04
        assert n_total_aum is n_total_kravtsov04


class TestNTotalKravtsov04:
    def test_sum_of_cen_and_sat(self):
        from hod_mod.connection.hod.base import n_cen
        from hod_mod.connection.hod.kravtsov04 import (
            n_sat_kravtsov04, n_total_kravtsov04)

        args = (12.5, 0.4, 12.8, 13.8, 1.05)
        nt = n_total_kravtsov04(_LOG10M, *args)
        nc = n_cen(_LOG10M, *args[:2])
        ns = n_sat_kravtsov04(_LOG10M, *args)
        np.testing.assert_allclose(np.asarray(nt), np.asarray(nc + ns), rtol=1e-6)


class TestKravtsov04HODModel:
    def test_nc_ns_and_integrate(self, hmf, planck_cosmo):
        from hod_mod.connection.hod.kravtsov04 import Kravtsov04HODModel

        model = Kravtsov04HODModel(hmf, hmf.bias)
        p = Kravtsov04HODModel.default_params()
        nc, ns = model.nc_ns(_LOG10M, p)
        assert nc.shape == _LOG10M.shape and ns.shape == _LOG10M.shape
        assert jnp.all((nc >= 0.0) & (nc <= 1.0))
        assert jnp.all(ns >= 0.0)
        # centrals saturate to 1 well above the threshold
        assert float(nc[-1]) == pytest.approx(1.0, abs=1e-6)

        n_gal, b_eff, m_eff = model._integrate(0.3, planck_cosmo, p)
        assert float(n_gal) > 0.0
        assert 0.5 < float(b_eff) < 5.0
        assert float(m_eff) > 10.0 ** p["log10mmin"]

    def test_lower_threshold_more_galaxies(self, hmf, planck_cosmo):
        from hod_mod.connection.hod.kravtsov04 import Kravtsov04HODModel

        model = Kravtsov04HODModel(hmf, hmf.bias)
        p = Kravtsov04HODModel.default_params()
        n_hi, _, _ = model._integrate(0.3, planck_cosmo, p)
        n_lo, _, _ = model._integrate(
            0.3, planck_cosmo, dict(p, log10mmin=12.0, log10m1=13.0))
        assert float(n_lo) > float(n_hi)

    def test_class_alias(self):
        from hod_mod.connection.hod.kravtsov04 import (
            AUMHODModel, Kravtsov04HODModel)

        assert AUMHODModel is Kravtsov04HODModel
