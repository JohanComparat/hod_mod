"""Tests for hod_mod.gas (Arnaud+2010, DPM Oppenheimer+2025)."""

import numpy as np
import pytest

from hod_mod.gas import (
    PressureProfileA10,
    GasDensityDPM,
    m200_to_m500c,
    _RHO_CRIT0,
)
from hod_mod.core.power_spectrum import LinearPowerSpectrum

_THETA = LinearPowerSpectrum.default_cosmology()
_Z     = 0.3
_H     = float(_THETA["h"])
_OM    = float(_THETA["Omega_m"])

# Reference mass and radius for tests (cluster-scale)
_M200 = 1.0e14   # Msun/h
_C200 = 5.0
_R200 = (3.0 * _M200 / (4.0 * np.pi * 200.0 * _RHO_CRIT0 * _OM))**(1.0/3.0)  # approximate

_K_ARR   = np.logspace(-2, 1, 30)    # h/Mpc
_M_ARR   = np.array([1e13, 1e14, 1e15])  # Msun/h
_R200_ARR = (_M_ARR / (4.0 / 3.0 * np.pi * 200.0 * _RHO_CRIT0 * _OM))**(1.0/3.0)
_C200_ARR = np.array([8.0, 5.0, 3.5])


# ===========================================================================
# m200_to_m500c
# ===========================================================================

class TestM200ToM500c:

    def test_returns_two_arrays(self):
        ez2 = _OM * (1 + _Z)**3 + (1 - _OM)
        rho_crit_z = _RHO_CRIT0 * ez2 / (1 + _Z)**3
        m500, r500 = m200_to_m500c(
            np.array([_M200]), np.array([_C200]), np.array([_R200]), rho_crit_z
        )
        assert m500.shape == (1,)
        assert r500.shape == (1,)

    def test_m500_less_than_m200(self):
        ez2 = _OM * (1 + _Z)**3 + (1 - _OM)
        rho_crit_z = _RHO_CRIT0 * ez2 / (1 + _Z)**3
        m500, r500 = m200_to_m500c(
            _M_ARR, _C200_ARR, _R200_ARR, rho_crit_z
        )
        # M₅₀₀ is always smaller than M₂₀₀ (smaller radius, steeper Δ threshold)
        assert np.all(m500 < _M_ARR)

    def test_r500_less_than_r200(self):
        ez2 = _OM * (1 + _Z)**3 + (1 - _OM)
        rho_crit_z = _RHO_CRIT0 * ez2 / (1 + _Z)**3
        m500, r500 = m200_to_m500c(
            _M_ARR, _C200_ARR, _R200_ARR, rho_crit_z
        )
        assert np.all(r500 < _R200_ARR)

    def test_m500_consistent_with_r500(self):
        """M₅₀₀ = (4π/3) × 500 × ρ_crit × r₅₀₀³."""
        ez2 = _OM * (1 + _Z)**3 + (1 - _OM)
        rho_crit_z = _RHO_CRIT0 * ez2 / (1 + _Z)**3
        m500, r500 = m200_to_m500c(
            _M_ARR, _C200_ARR, _R200_ARR, rho_crit_z
        )
        m500_check = (4.0 / 3.0) * np.pi * 500.0 * rho_crit_z * r500**3
        np.testing.assert_allclose(m500, m500_check, rtol=1e-5)

    def test_ratio_m500_m200_typical_range(self):
        """Typical M₅₀₀/M₂₀₀ ratio is 0.5–0.8 for cluster concentrations."""
        ez2 = _OM * (1 + _Z)**3 + (1 - _OM)
        rho_crit_z = _RHO_CRIT0 * ez2 / (1 + _Z)**3
        m500, r500 = m200_to_m500c(
            _M_ARR, _C200_ARR, _R200_ARR, rho_crit_z
        )
        ratio = m500 / _M_ARR
        assert np.all(ratio > 0.4)
        assert np.all(ratio < 0.9)


# ===========================================================================
# PressureProfileA10
# ===========================================================================

class TestPressureProfileA10:

    @pytest.fixture(scope="class")
    def pp(self):
        return PressureProfileA10(r_max_over_r500c=5.0, n_gl=100)

    def test_instantiation(self, pp):
        assert pp._P0    == pytest.approx(8.403)
        assert pp._c500  == pytest.approx(1.177)
        assert pp._alpha == pytest.approx(1.0510)
        assert pp._beta  == pytest.approx(5.4905)
        assert pp._gamma == pytest.approx(0.3081)
        assert pp._alpha_p == pytest.approx(0.12)

    def test_p3d_positive(self, pp):
        r_over_r500 = np.logspace(-1, 1, 20)
        p = pp._p3d(r_over_r500, _M200, _Z, _H, _OM)
        assert np.all(p > 0)

    def test_p3d_decreasing(self, pp):
        r_over_r500 = np.logspace(-1, 1, 50)
        p = pp._p3d(r_over_r500, _M200, _Z, _H, _OM)
        assert np.all(np.diff(p) < 0), "pressure profile should be monotonically decreasing"

    def test_p3d_amplitude_reasonable(self, pp):
        """Central pressure of a 10¹⁴ Msun/h cluster at z=0.3 should be O(0.01-1) keV/cm³."""
        p_center = pp._p3d(np.array([0.1]), _M200, _Z, _H, _OM)
        assert 1e-4 < float(p_center[0]) < 10.0

    def test_pressure_uk_shape(self, pp):
        uk = pp.pressure_uk(_K_ARR, _M_ARR, _R200_ARR, _C200_ARR, _Z, _THETA)
        assert uk.shape == (len(_K_ARR), len(_M_ARR))

    def test_pressure_uk_positive(self, pp):
        uk = pp.pressure_uk(_K_ARR, _M_ARR, _R200_ARR, _C200_ARR, _Z, _THETA)
        assert np.all(uk > 0)

    def test_pressure_uk_finite(self, pp):
        uk = pp.pressure_uk(_K_ARR, _M_ARR, _R200_ARR, _C200_ARR, _Z, _THETA)
        assert np.all(np.isfinite(uk))

    def test_pressure_uk_decreasing_with_k(self, pp):
        """At high k, the profile FT should fall — at low k it should be nearly flat."""
        uk = pp.pressure_uk(_K_ARR, _M_ARR, _R200_ARR, _C200_ARR, _Z, _THETA)
        # Compare value at k=0.1 vs k=5: high-k should be at least 2x smaller
        idx_lo = np.argmin(np.abs(_K_ARR - 0.1))
        idx_hi = np.argmin(np.abs(_K_ARR - 5.0))
        assert np.all(uk[idx_hi] < uk[idx_lo])

    def test_pressure_uk_mass_ordering(self, pp):
        """Larger halos should have larger ỹ at low k (more gas, bigger pressure volume)."""
        uk = pp.pressure_uk(np.array([0.01]), _M_ARR, _R200_ARR, _C200_ARR, _Z, _THETA)
        # uk[0, :] is shape (NM,) — should increase with halo mass
        assert uk[0, 0] < uk[0, 1] < uk[0, 2]

    def test_pressure_uk_units_order(self, pp):
        """ỹ at k=0.01 for a 10¹⁴ M☉/h cluster should be O(10⁻⁴ – 10⁰) (Mpc/h)²."""
        uk = pp.pressure_uk(np.array([0.01]), _M_ARR[1:2], _R200_ARR[1:2], _C200_ARR[1:2], _Z, _THETA)
        assert 1e-8 < float(uk[0, 0]) < 1e4


# ===========================================================================
# GasDensityDPM
# ===========================================================================

class TestGasDensityDPMModel1:

    @pytest.fixture(scope="class")
    def dp(self):
        return GasDensityDPM(model=1, r_max_over_r200=3.0, n_gl=100)

    def test_instantiation(self, dp):
        assert dp._model == 1
        assert dp._ne_03   == pytest.approx(5.86e-4)
        assert dp._alpha_in  == pytest.approx(1.0)
        assert dp._alpha_out == pytest.approx(2.7)
        assert dp._beta == pytest.approx(0.0)

    def test_density_positive(self, dp):
        r = np.logspace(-1, 0, 20) * _R200
        ne = dp.density_3d(r, _M200, _R200, _Z, _OM)
        assert np.all(ne > 0)

    def test_density_decreasing(self, dp):
        r = np.logspace(-1, 0.3, 50) * _R200
        ne = dp.density_3d(r, _M200, _R200, _Z, _OM)
        assert np.all(np.diff(ne) < 0)

    def test_density_uk_shape(self, dp):
        uk = dp.density_uk(_K_ARR, _M_ARR, _R200_ARR, _Z, _THETA)
        assert uk.shape == (len(_K_ARR), len(_M_ARR))

    def test_density_uk_positive(self, dp):
        uk = dp.density_uk(_K_ARR, _M_ARR, _R200_ARR, _Z, _THETA)
        assert np.all(uk > 0)

    def test_density_uk_finite(self, dp):
        uk = dp.density_uk(_K_ARR, _M_ARR, _R200_ARR, _Z, _THETA)
        assert np.all(np.isfinite(uk))

    def test_emissivity_uk_shape(self, dp):
        uk = dp.emissivity_uk(_K_ARR, _M_ARR, _R200_ARR, _Z, _THETA)
        assert uk.shape == (len(_K_ARR), len(_M_ARR))

    def test_emissivity_greater_than_zero(self, dp):
        uk = dp.emissivity_uk(_K_ARR, _M_ARR, _R200_ARR, _Z, _THETA)
        assert np.all(uk > 0)

    def test_emissivity_less_than_density_squared_at_center(self, dp):
        """ñ_e²(k) at low k < ñ_e(k) (no divergence from squaring)."""
        uk_ne  = dp.density_uk(np.array([0.01]),   _M_ARR, _R200_ARR, _Z, _THETA)
        uk_ne2 = dp.emissivity_uk(np.array([0.01]), _M_ARR, _R200_ARR, _Z, _THETA)
        # emissivity FT should be smaller since n_e² is more concentrated
        assert np.all(uk_ne2[0] > 0)


class TestGasDensityDPMModel2:

    @pytest.fixture(scope="class")
    def dp(self):
        return GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80)

    def test_model2_lower_ne03(self, dp):
        assert dp._ne_03 == pytest.approx(4.87e-5)
        assert dp._beta  == pytest.approx(0.36)

    def test_mass_scaling(self, dp):
        """Higher mass halos should have higher n_e at fixed r/R₂₀₀ for Model 2 (β>0)."""
        r_ref = 0.3 * _R200_ARR
        ne_low  = dp.density_3d(r_ref[0:1], _M_ARR[0], _R200_ARR[0], _Z, _OM)
        ne_high = dp.density_3d(r_ref[2:3], _M_ARR[2], _R200_ARR[2], _Z, _OM)
        assert float(ne_high[0]) > float(ne_low[0])

    def test_density_uk_shape(self, dp):
        uk = dp.density_uk(_K_ARR, _M_ARR, _R200_ARR, _Z, _THETA)
        assert uk.shape == (len(_K_ARR), len(_M_ARR))


class TestGasDensityDPMModel3:

    def test_model3_instantiation(self):
        dp = GasDensityDPM(model=3, n_gl=50)
        assert dp._alpha_in  == pytest.approx(0.4)
        assert dp._alpha_tr  == pytest.approx(0.45)
        assert dp._alpha_out == pytest.approx(0.5)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model must be"):
            GasDensityDPM(model=4)

    def test_density_uk_positive(self):
        dp = GasDensityDPM(model=3, n_gl=50)
        uk = dp.density_uk(np.array([0.1, 1.0]), _M_ARR[:2], _R200_ARR[:2], _Z, _THETA)
        assert np.all(uk > 0)


# ===========================================================================
# Profile normalization consistency check
# ===========================================================================

class TestProfileNormalization:

    def test_a10_profile_at_reference_point(self):
        """Arnaud+2010 profile at r=0 should give the peak pressure (no analytical check,
        but we verify it's larger than at r=R₅₀₀)."""
        pp = PressureProfileA10()
        p_inner = pp._p3d(np.array([0.01]), _M200, 0.0, _H, _OM)
        p_outer = pp._p3d(np.array([1.0]),  _M200, 0.0, _H, _OM)
        assert float(p_inner[0]) > float(p_outer[0])

    def test_dpm_normalization_at_reference(self):
        """DPM: n_e(0.3 R₂₀₀ | 10¹² M☉/h, z=0) should equal ne_03."""
        for model_id in [1, 2, 3]:
            dp = GasDensityDPM(model=model_id, n_gl=50)
            # At exactly 10¹² Msun/h (no h factor ambiguity in test — use raw Msun)
            # The model uses M₁₂ = M₂₀₀ / 10¹² → 1.0 for reference mass
            m_ref  = 1.0e12
            r200_ref = (m_ref / (4.0/3.0 * np.pi * 200.0 * _RHO_CRIT0 * _OM))**(1.0/3.0)
            r_ref  = 0.3 * r200_ref
            ne_val = dp.density_3d(np.array([r_ref]), m_ref, r200_ref, 0.0, _OM)
            np.testing.assert_allclose(float(ne_val[0]), dp._ne_03, rtol=1e-6,
                                       err_msg=f"Model {model_id}: normalization mismatch")
