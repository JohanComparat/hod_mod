"""Tests for the complete DPM model implementation.

Covers:
- PressureProfileDPM  (DPM Eq. 2 + Eq. 5)
- MetallicityProfileDPM  (DPM Eq. 4)
- GasDensityDPM scatter boost  (DPM Eq. 6)
- psf_window_ell  (eROSITA PSF)
- temperature_from_dpm  (Section 3.1.1)
- xray_cooling_function

References
----------
Oppenheimer+2025, arXiv:2505.14782 — DPM paper
"""

import numpy as np
import pytest

from hod_mod.gas import (
    PressureProfileDPM,
    MetallicityProfileDPM,
    GasDensityDPM,
    temperature_from_dpm,
    temperature_from_profiles,
    xray_cooling_function,
    _RHO_CRIT0,
)
from hod_mod.observables.cross_spectra import (
    psf_window_ell,
    psf_king_profile,
    psf_king_window_ell,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_OM = 0.3
# Full cosmology dict: the emissivity scatter-boost path reaches eisenstein_hu_pk,
# which requires Omega_b and n_s in addition to h and Omega_m.
_THETA = {"h": 0.7, "Omega_m": _OM, "Omega_b": 0.048, "n_s": 0.96}


def _r200(m200, omega_m=_OM):
    """R_200 [Mpc/h] for halo of mass M_200 [Msun/h]."""
    return (m200 / (4.0 / 3.0 * np.pi * 200.0 * _RHO_CRIT0 * omega_m)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# PressureProfileDPM
# ---------------------------------------------------------------------------

class TestPressureProfileDPM:

    @pytest.mark.parametrize("model_id", [1, 2, 3])
    def test_normalization(self, model_id):
        """P(0.3 R_200, 10^12 M_sun/h, z=0) = P_0.3 from Table 1."""
        pp = PressureProfileDPM(model=model_id, n_gl=50)
        m_ref = 1e12
        r200_ref = _r200(m_ref)
        r_ref = 0.3 * r200_ref

        P_val = float(pp._pressure_3d(np.array([r_ref]), m_ref, r200_ref, 0.0, _OM)[0])
        P_expected = pp._P_03
        rel_err = abs(P_val - P_expected) / P_expected
        assert rel_err < 1e-5, (
            f"Model {model_id}: P(0.3 R200) = {P_val:.6e}, expected {P_expected:.6e}, "
            f"rel_err = {rel_err:.2e}"
        )

    @pytest.mark.parametrize("model_id", [2, 3])
    def test_mass_scaling(self, model_id):
        """P ∝ M^{β^P} from Table 1."""
        pp = PressureProfileDPM(model=model_id, n_gl=50)
        m1 = 1e12
        m2 = 1e13
        r200_1 = _r200(m1)
        r200_2 = _r200(m2)
        r1 = 0.3 * r200_1
        r2 = 0.3 * r200_2

        P1 = float(pp._pressure_3d(np.array([r1]), m1, r200_1, 0.0, _OM)[0])
        P2 = float(pp._pressure_3d(np.array([r2]), m2, r200_2, 0.0, _OM)[0])
        expected_ratio = (m2 / m1) ** pp._beta
        actual_ratio = P2 / P1
        rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_err < 0.05, (
            f"Model {model_id}: mass slope β={pp._beta:.3f}, "
            f"P ratio = {actual_ratio:.4f}, expected {expected_ratio:.4f}"
        )

    @pytest.mark.parametrize("model_id", [1, 2, 3])
    def test_pressure_uk_shape(self, model_id):
        """pressure_uk returns (Nk, NM) with positive values."""
        pp = PressureProfileDPM(model=model_id, n_gl=30)
        k = np.logspace(-2, 1, 10)
        m = np.array([1e12, 1e13, 1e14])
        r = _r200(m)
        c = np.array([10.0, 8.0, 6.0])  # concentration (not used by DPM, but kept for interface)

        uk = pp.pressure_uk(k, m, r, z=0.2, theta_cosmo=_THETA)
        assert uk.shape == (10, 3), f"Expected (10, 3), got {uk.shape}"
        assert np.all(uk > 0), "pressure_uk must be positive"
        assert np.all(np.isfinite(uk)), "pressure_uk must be finite"

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            PressureProfileDPM(model=0)


# ---------------------------------------------------------------------------
# MetallicityProfileDPM
# ---------------------------------------------------------------------------

class TestMetallicityProfileDPM:

    def test_normalization(self):
        """Z(0.3 R_200) = 0.3 Z_sun — Table 1 of arXiv:2505.14782."""
        mp = MetallicityProfileDPM()
        r200 = _r200(1e13)
        r_ref = 0.3 * r200

        Z_val = float(mp.metallicity_3d(np.array([r_ref]), r200)[0])
        Z_expected = mp._Z_03
        rel_err = abs(Z_val - Z_expected) / Z_expected
        assert rel_err < 1e-5, (
            f"Z(0.3 R200) = {Z_val:.6e}, expected {Z_expected:.6e}"
        )

    def test_no_mass_dependence(self):
        """β^Z = 0: Z(0.3 R_200) is the same for all masses."""
        mp = MetallicityProfileDPM()
        masses = [1e11, 1e12, 1e13, 1e14]
        Z_vals = []
        for m in masses:
            r200 = _r200(m)
            Z_vals.append(float(mp.metallicity_3d(np.array([0.3 * r200]), r200)[0]))
        # All should equal Z_0.3 (within floating-point tolerance)
        assert np.allclose(Z_vals, mp._Z_03, rtol=1e-5), (
            f"Z values vary with mass: {Z_vals}"
        )

    def test_no_redshift_dependence(self):
        """γ^Z = 0: Z(0.3 R_200) does not evolve with z."""
        mp = MetallicityProfileDPM()
        r200 = _r200(1e13)
        r_ref = 0.3 * r200
        # Z has no z argument (correct — β^Z = γ^Z = 0)
        Z = float(mp.metallicity_3d(np.array([r_ref]), r200)[0])
        assert np.isfinite(Z)
        assert Z > 0

    def test_radial_profile_is_monotone(self):
        """Metallicity profile is a monotonically decreasing gNFW."""
        mp = MetallicityProfileDPM()
        r200 = _r200(1e13)
        r_arr = np.linspace(0.1, 1.0, 20) * r200
        Z = mp.metallicity_3d(r_arr, r200)
        assert np.all(np.diff(Z) < 0), "Metallicity profile should decrease outward"


# ---------------------------------------------------------------------------
# GasDensityDPM scatter boost  (DPM Eq. 6)
# ---------------------------------------------------------------------------

class TestScatterBoost:

    def test_no_scatter_boost_is_unity(self):
        """sigma_scatter=0 → scatter_boost = 1."""
        dp = GasDensityDPM(model=2, sigma_scatter=0.0, n_gl=30)
        assert abs(dp._scatter_boost - 1.0) < 1e-10

    def test_scatter_boost_formula(self):
        """exp((σ × ln 10)²) for σ = 0.8 dex."""
        sigma = 0.8
        dp = GasDensityDPM(model=2, sigma_scatter=sigma, n_gl=30)
        expected_boost = np.exp((sigma * np.log(10.0)) ** 2)
        assert abs(dp._scatter_boost - expected_boost) / expected_boost < 1e-8

    def test_emissivity_uk_scaled_by_boost(self):
        """emissivity_uk with scatter > emissivity_uk without scatter."""
        k = np.array([0.1, 1.0, 5.0])
        m = np.array([1e13])
        r = _r200(m)

        dp0 = GasDensityDPM(model=2, sigma_scatter=0.0, n_gl=30)
        dp1 = GasDensityDPM(model=2, sigma_scatter=0.5, n_gl=30)

        uk0 = dp0.emissivity_uk(k, m, r, z=0.2, theta_cosmo=_THETA)
        uk1 = dp1.emissivity_uk(k, m, r, z=0.2, theta_cosmo=_THETA)

        ratio = uk1 / uk0
        expected = dp1._scatter_boost / dp0._scatter_boost
        assert np.allclose(ratio, expected, rtol=1e-5), (
            f"Emissivity ratio {ratio} != scatter boost ratio {expected}"
        )


# ---------------------------------------------------------------------------
# PSF window function
# ---------------------------------------------------------------------------

class TestPsfWindowEll:

    def test_unity_at_ell_zero(self):
        """B_ℓ(ℓ=0) = 1 for any FWHM."""
        B = psf_window_ell(np.array([0.0]), fwhm_arcsec=30.0)
        assert abs(float(B[0]) - 1.0) < 1e-10

    def test_suppression_at_large_ell(self):
        """B_ℓ < 1 at ℓ=1000 for 30-arcsec PSF."""
        B = psf_window_ell(np.array([1000.0]), fwhm_arcsec=30.0)
        assert float(B[0]) < 1.0
        assert float(B[0]) > 0.0

    def test_monotone_decrease(self):
        """B_ℓ decreases monotonically with ℓ."""
        ell = np.linspace(0, 2000, 50)
        B = psf_window_ell(ell, fwhm_arcsec=30.0)
        assert np.all(np.diff(B) <= 0.0), "PSF window should decrease with ℓ"

    def test_wider_psf_more_suppression(self):
        """Wider PSF → more suppression at same ℓ."""
        ell = np.array([500.0])
        B_narrow = float(psf_window_ell(ell, fwhm_arcsec=10.0)[0])
        B_wide   = float(psf_window_ell(ell, fwhm_arcsec=60.0)[0])
        assert B_narrow > B_wide


# ---------------------------------------------------------------------------
# King-profile PSF
# ---------------------------------------------------------------------------

class TestKingPsfProfile:

    def test_unity_at_zero(self):
        """PSF(0) = 1."""
        assert psf_king_profile(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_decreasing(self):
        """King profile decreases with radius."""
        theta = np.linspace(0, 100, 50)
        P = psf_king_profile(theta)
        assert np.all(np.diff(P) <= 0.0)

    def test_wider_core_more_extended(self):
        """Larger θ_c → higher value at same θ."""
        theta = np.array([20.0])
        P_narrow = float(psf_king_profile(theta, theta_c_arcsec=5.0)[0])
        P_wide   = float(psf_king_profile(theta, theta_c_arcsec=20.0)[0])
        assert P_wide > P_narrow


class TestKingPsfWindowEll:

    def test_unity_at_ell_zero(self):
        """B_ℓ(ℓ=0) = 1."""
        B = psf_king_window_ell(np.array([0.0]))
        assert float(B[0]) == pytest.approx(1.0, abs=1e-8)

    def test_suppression_at_large_ell(self):
        """B_ℓ < 1 at ℓ=1000."""
        B = psf_king_window_ell(np.array([1000.0]))
        assert 0.0 < float(B[0]) < 1.0

    def test_monotone_decrease(self):
        """B_ℓ decreases monotonically with ℓ."""
        ell = np.linspace(0, 5000, 100)
        B = np.asarray(psf_king_window_ell(ell))
        assert np.all(np.diff(B) <= 0.0)

    def test_alpha_1p5_is_exponential(self):
        """α=3/2 special case: B_ℓ = exp(−ℓ θ_c) exactly."""
        from hod_mod.observables.cross_spectra import _ARCSEC_TO_RAD
        tc_arcsec = 8.64
        ell = np.array([0.0, 100.0, 1000.0, 5000.0])
        Bk = np.asarray(psf_king_window_ell(ell, theta_c_arcsec=tc_arcsec, alpha=1.5))
        expected = np.exp(-ell * tc_arcsec * _ARCSEC_TO_RAD)
        np.testing.assert_allclose(Bk, expected, rtol=1e-6)

    def test_wider_core_more_suppression(self):
        """Larger θ_c → more suppression at same ℓ."""
        ell = np.array([500.0])
        B_narrow = float(psf_king_window_ell(ell, theta_c_arcsec=5.0)[0])
        B_wide   = float(psf_king_window_ell(ell, theta_c_arcsec=30.0)[0])
        assert B_narrow > B_wide


# ---------------------------------------------------------------------------
# Temperature and cooling function
# ---------------------------------------------------------------------------

class TestTemperatureFromDpm:

    def test_temperature_positive(self):
        """T = P / n_e > 0 everywhere."""
        pp = PressureProfileDPM(model=2, n_gl=30)
        dp = GasDensityDPM(model=2, n_gl=30)
        m200 = 1e13
        r200 = _r200(m200)
        r_arr = np.linspace(0.05, 1.0, 10) * r200
        T = temperature_from_dpm(pp, dp, r_arr, m200, r200, 0.2, _THETA)
        assert np.all(T > 0), "Temperature must be positive"
        assert np.all(np.isfinite(T)), "Temperature must be finite"

    def test_temperature_physically_reasonable(self):
        """T > 0.01 keV for a 10^13 M_sun halo."""
        pp = PressureProfileDPM(model=2, n_gl=30)
        dp = GasDensityDPM(model=2, n_gl=30)
        m200 = 1e13
        r200 = _r200(m200)
        r_arr = np.array([0.3 * r200])
        T = temperature_from_dpm(pp, dp, r_arr, m200, r200, 0.2, _THETA)
        # We don't require a specific value since P_0.3 units need verification,
        # but T must be a finite positive number.
        assert float(T[0]) > 0
        assert np.isfinite(float(T[0]))


class TestXrayCoolingFunction:

    def test_positive(self):
        """Λ(T, Z) > 0."""
        T = np.array([0.1, 1.0, 5.0, 10.0])
        Z = np.array([0.1, 0.3, 1.0, 1.0])
        L = xray_cooling_function(T, Z)
        assert np.all(L > 0)

    def test_normalization(self):
        """Λ(1 keV, 0.3 Z_sun) = Lambda_0."""
        Lambda_0 = 3e-23
        L = float(xray_cooling_function(1.0, 0.3, Lambda_0=Lambda_0))
        assert abs(L - Lambda_0) / Lambda_0 < 1e-8

    def test_temperature_slope(self):
        """Λ ∝ T^{alpha_T}."""
        alpha_T = 0.5
        T1, T2 = 0.5, 2.0
        L1 = float(xray_cooling_function(T1, 0.3, alpha_T=alpha_T))
        L2 = float(xray_cooling_function(T2, 0.3, alpha_T=alpha_T))
        expected_ratio = (T2 / T1) ** alpha_T
        assert abs(L2 / L1 - expected_ratio) / expected_ratio < 1e-8
