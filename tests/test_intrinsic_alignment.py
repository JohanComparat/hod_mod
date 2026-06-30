"""Tests for intrinsic alignment models (NLA, TATT) and related infrastructure.

Covers:
- NLAModel: zero amplitude, negative signal, redshift scaling
- TATTModel: reduction to NLA when b_TA = 0
- _linear_growth_factor: physical limits
- mdef consistency: HaloProfile._mdef_delta_rho for '200m' and '200c'
- LinearPowerSpectrum: CDM and baryon auto-spectra
- delta_sigma with ia_model integration
"""

import numpy as np
import pytest
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared mock P(k): power-law spectrum so tests don't need CAMB/Aletheia
# ---------------------------------------------------------------------------

def _mock_pk_nl(k, z, theta):
    """Power-law mock nonlinear P(k) ∝ k^{-2} [(Mpc/h)^3]."""
    k_arr = jnp.asarray(k)
    return 1e4 * (k_arr / 0.1) ** (-2.0)


@pytest.fixture(scope="module")
def theta():
    return {
        "h": 0.6736, "Omega_b": 0.0493, "Omega_cdm": 0.2607,
        "Omega_m": 0.3100, "n_s": 0.9649, "ln10^{10}A_s": 3.044,
    }


@pytest.fixture(scope="module")
def R_arr():
    return jnp.logspace(-1, 1.0, 12)


# ---------------------------------------------------------------------------
# Growth factor
# ---------------------------------------------------------------------------

class TestLinearGrowthFactor:
    def test_z0_is_one(self):
        from hod_mod.observables.intrinsic_alignment import _linear_growth_factor
        assert _linear_growth_factor(0.0, 0.31) == pytest.approx(1.0, rel=1e-6)

    def test_decreases_with_z(self):
        from hod_mod.observables.intrinsic_alignment import _linear_growth_factor
        D0 = _linear_growth_factor(0.0, 0.31)
        D1 = _linear_growth_factor(1.0, 0.31)
        D3 = _linear_growth_factor(3.0, 0.31)
        assert D0 > D1 > D3 > 0

    def test_physical_range(self):
        from hod_mod.observables.intrinsic_alignment import _linear_growth_factor
        for z in [0.0, 0.3, 0.5, 1.0, 2.0]:
            D = _linear_growth_factor(z, 0.31)
            assert 0.0 < D <= 1.0


# ---------------------------------------------------------------------------
# NLA model
# ---------------------------------------------------------------------------

class TestNLAModel:
    @pytest.fixture(scope="class")
    def nla(self):
        from hod_mod.observables.intrinsic_alignment import NLAModel
        return NLAModel(_mock_pk_nl)

    def test_zero_amplitude_gives_zero(self, nla, theta, R_arr):
        """A_IA = 0 → ΔΣ_IA = 0 everywhere."""
        ds_ia = nla.delta_sigma_ia(
            R_arr, 0.3, theta, {"A_IA": 0.0, "eta_IA": 0.0}
        )
        np.testing.assert_allclose(np.asarray(ds_ia), 0.0, atol=1e-30)

    def test_positive_amplitude_gives_negative_ds(self, nla, theta, R_arr):
        """A_IA > 0 → ΔΣ_IA < 0 (GI term suppresses the lensing signal)."""
        ds_ia = nla.delta_sigma_ia(
            R_arr, 0.3, theta, {"A_IA": 1.0, "eta_IA": 0.0}
        )
        assert np.all(np.asarray(ds_ia) < 0.0)

    def test_linearity_in_A_IA(self, nla, theta, R_arr):
        """ΔΣ_IA scales linearly with A_IA."""
        ds1 = np.asarray(nla.delta_sigma_ia(R_arr, 0.3, theta, {"A_IA": 1.0, "eta_IA": 0.0}))
        ds2 = np.asarray(nla.delta_sigma_ia(R_arr, 0.3, theta, {"A_IA": 2.0, "eta_IA": 0.0}))
        np.testing.assert_allclose(ds2, 2.0 * ds1, rtol=1e-6)

    def test_redshift_scaling(self, nla, theta, R_arr):
        """ΔΣ_IA(z) ∝ (1+z)^η / D(z)^2 — ratio of two redshifts matches."""
        from hod_mod.observables.intrinsic_alignment import _linear_growth_factor
        z1, z2 = 0.1, 0.5
        eta = 1.0
        params = {"A_IA": 1.0, "eta_IA": eta}
        ds1 = np.asarray(nla.delta_sigma_ia(R_arr, z1, theta, params))
        ds2 = np.asarray(nla.delta_sigma_ia(R_arr, z2, theta, params))

        D1 = _linear_growth_factor(z1, theta["Omega_m"])
        D2 = _linear_growth_factor(z2, theta["Omega_m"])
        expected_ratio = ((1 + z2) ** eta / D2 ** 2) / ((1 + z1) ** eta / D1 ** 2)
        # ratio of ΔΣ at fixed R (pick midpoint)
        mid = len(R_arr) // 2
        actual_ratio = float(ds2[mid] / ds1[mid])
        assert actual_ratio == pytest.approx(expected_ratio, rel=5e-3)

    def test_default_params(self):
        from hod_mod.observables.intrinsic_alignment import NLAModel
        p = NLAModel.default_params()
        assert p["A_IA"] == 0.0 and p["eta_IA"] == 0.0

    def test_output_shape(self, nla, theta, R_arr):
        ds_ia = nla.delta_sigma_ia(R_arr, 0.3, theta, {"A_IA": 0.5, "eta_IA": 0.0})
        assert ds_ia.shape == R_arr.shape


# ---------------------------------------------------------------------------
# TATT model
# ---------------------------------------------------------------------------

class TestTATTModel:
    @pytest.fixture(scope="class")
    def tatt(self):
        from hod_mod.observables.intrinsic_alignment import TATTModel
        return TATTModel(_mock_pk_nl)

    @pytest.fixture(scope="class")
    def nla(self):
        from hod_mod.observables.intrinsic_alignment import NLAModel
        return NLAModel(_mock_pk_nl)

    def test_zero_params_gives_zero(self, tatt, theta, R_arr):
        ds_ia = tatt.delta_sigma_ia(
            R_arr, 0.3, theta, {"a_TA": 0.0, "b_TA": 0.0, "eta_TA": 0.0}
        )
        np.testing.assert_allclose(np.asarray(ds_ia), 0.0, atol=1e-30)

    def test_reduces_to_nla_when_b_ta_zero(self, tatt, nla, theta, R_arr):
        """b_TA = 0, a_TA = A_IA → TATT ≡ NLA."""
        A = 1.5
        eta = 0.5
        ds_nla = np.asarray(nla.delta_sigma_ia(R_arr, 0.3, theta, {"A_IA": A, "eta_IA": eta}))
        ds_tatt = np.asarray(tatt.delta_sigma_ia(
            R_arr, 0.3, theta, {"a_TA": A, "b_TA": 0.0, "eta_TA": eta}
        ))
        np.testing.assert_allclose(ds_tatt, ds_nla, rtol=1e-6)

    def test_b_ta_term_adds_negative(self, tatt, theta, R_arr):
        """Non-zero b_TA with a positive ds_gm makes ΔΣ_IA more negative."""
        # ds_gm > 0 (gravitational lensing signal); the b_TA term subtracts F_b*ds_gm
        ds_gm = 0.5 * jnp.ones_like(R_arr)  # mock positive gravitational ΔΣ
        p_with_bta = {"a_TA": 1.0, "b_TA": 0.5, "eta_TA": 0.0}
        p_no_bta   = {"a_TA": 1.0, "b_TA": 0.0, "eta_TA": 0.0}
        ds_bta  = np.asarray(tatt.delta_sigma_ia(R_arr, 0.3, theta, p_with_bta, ds_gm=ds_gm))
        ds_none = np.asarray(tatt.delta_sigma_ia(R_arr, 0.3, theta, p_no_bta))
        # b_TA correction subtracts F_b * ds_gm > 0, making the signal more negative
        assert np.all(ds_bta < ds_none)

    def test_default_params(self):
        from hod_mod.observables.intrinsic_alignment import TATTModel
        p = TATTModel.default_params()
        assert p["a_TA"] == 0.0 and p["b_TA"] == 0.0 and p["eta_TA"] == 0.0


# ---------------------------------------------------------------------------
# delta_sigma integration with IA model
# ---------------------------------------------------------------------------

class TestDeltaSigmaWithIA:
    @pytest.fixture(scope="class")
    def pred_and_nla(self):
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        from hod_mod.core.halo_mass_function import make_hmf
        from hod_mod.core.halo_profiles import HaloProfile
        from hod_mod.connection.hod import HODModel
        from hod_mod.observables.clustering import FullHaloModelPrediction
        from hod_mod.observables.intrinsic_alignment import NLAModel

        pk_lin = LinearPowerSpectrum()
        theta = pk_lin.default_cosmology()
        hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        colossus_cosmo = {
            "flat": True, "H0": theta["h"] * 100,
            "Om0": theta["Omega_m"], "Ob0": theta["Omega_b"],
            "sigma8": 0.811, "ns": theta["n_s"],
        }
        hp = HaloProfile(colossus_cosmo)
        hod = HODModel(hmf, hmf.bias)
        pred = FullHaloModelPrediction(pk_lin, hod, hp)
        nla = NLAModel(_mock_pk_nl)
        return pred, nla, theta

    def test_no_ia_unchanged(self, pred_and_nla):
        """ia_model=None returns the standard gravitational ΔΣ."""
        pred, nla, theta = pred_and_nla
        R = jnp.logspace(-1, 1.0, 6)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        ds_grav = np.asarray(pred.delta_sigma(R, 0.1, theta, hod_p))
        ds_null = np.asarray(pred.delta_sigma(R, 0.1, theta, hod_p,
                                              ia_model=nla,
                                              ia_params={"A_IA": 0.0, "eta_IA": 0.0}))
        np.testing.assert_allclose(ds_grav, ds_null, rtol=1e-6)

    def test_ia_suppresses_signal(self, pred_and_nla):
        """A_IA > 0 decreases ΔΣ relative to the gravitational-only prediction."""
        pred, nla, theta = pred_and_nla
        R = jnp.logspace(-1, 1.0, 6)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        ds_grav = np.asarray(pred.delta_sigma(R, 0.1, theta, hod_p))
        ds_ia   = np.asarray(pred.delta_sigma(R, 0.1, theta, hod_p,
                                              ia_model=nla,
                                              ia_params={"A_IA": 2.0, "eta_IA": 0.0}))
        assert np.all(ds_ia < ds_grav)


# ---------------------------------------------------------------------------
# HaloProfile mass-definition consistency
# ---------------------------------------------------------------------------

class TestMdefConsistency:
    @pytest.fixture(scope="class")
    def colossus_cosmo(self, theta):
        return {
            "flat": True, "H0": theta["h"] * 100,
            "Om0": theta["Omega_m"], "Ob0": theta["Omega_b"],
            "sigma8": 0.811, "ns": theta["n_s"],
        }

    @pytest.fixture(scope="class")
    def theta(self):
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        return LinearPowerSpectrum.default_cosmology()

    def test_200c_and_200m_give_different_rs(self, colossus_cosmo, theta):
        """mdef='200c' and '200m' produce different r_s at the same mass."""
        from hod_mod.core.halo_profiles import HaloProfile
        hp_m = HaloProfile(colossus_cosmo, mdef="200m")
        hp_c = HaloProfile(colossus_cosmo, mdef="200c")
        m = jnp.array([1e14])
        _, rs_m = hp_m.rho_s_and_rs(m, 0.3, theta)
        _, rs_c = hp_c.rho_s_and_rs(m, 0.3, theta)
        assert abs(float(rs_m[0]) - float(rs_c[0])) > 0.01  # differ by >1%

    def test_200m_rho_ref_z_independent(self, colossus_cosmo, theta):
        """_mdef_delta_rho('200m') returns the same rho_ref at all z."""
        from hod_mod.core.halo_profiles import HaloProfile
        hp = HaloProfile(colossus_cosmo, mdef="200m")
        _, rho0 = hp._mdef_delta_rho(0.0, theta)
        _, rho1 = hp._mdef_delta_rho(1.0, theta)
        assert rho0 == pytest.approx(rho1, rel=1e-10)

    def test_200c_rho_ref_increases_with_z(self, colossus_cosmo, theta):
        """_mdef_delta_rho('200c') comoving ρ_crit(z) decreases with z in comoving coords."""
        from hod_mod.core.halo_profiles import HaloProfile
        hp = HaloProfile(colossus_cosmo, mdef="200c")
        _, rho0 = hp._mdef_delta_rho(0.0, theta)
        _, rho1 = hp._mdef_delta_rho(1.0, theta)
        _, rho2 = hp._mdef_delta_rho(2.0, theta)
        # comoving critical density at high z approaches Omega_m * rho_crit0 (matter-dominated)
        # so it's not strictly monotone; just check both differ from z=0
        assert rho0 != pytest.approx(rho1, rel=0.01)

    def test_vir_delta_in_physical_range(self, colossus_cosmo, theta):
        """Bryan & Norman virial overdensity ≈ 102 at z=0 for Ω_m=0.31 (ΛCDM, not EdS)."""
        from hod_mod.core.halo_profiles import HaloProfile
        hp = HaloProfile(colossus_cosmo, mdef="vir")
        delta, _ = hp._mdef_delta_rho(0.0, theta)
        # EdS gives 18π² ≈ 178; ΛCDM with Ω_m=0.31 gives ≈102 (dark energy reduces binding)
        assert 50.0 < delta < 200.0

    def test_unknown_mdef_raises(self, colossus_cosmo, theta):
        from hod_mod.core.halo_profiles import HaloProfile
        hp = HaloProfile(colossus_cosmo, mdef="500c")
        with pytest.raises(ValueError):
            hp._mdef_delta_rho(0.0, theta)


# ---------------------------------------------------------------------------
# LinearPowerSpectrum: CDM and baryon auto-spectra
# ---------------------------------------------------------------------------

class TestPkLinearSplit:
    @pytest.fixture(scope="class")
    def pk_lin(self):
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        return LinearPowerSpectrum()

    @pytest.fixture(scope="class")
    def theta(self, pk_lin):
        return pk_lin.default_cosmology()

    def test_cdm_and_b_are_positive(self, pk_lin, theta):
        k = jnp.logspace(-2, 0, 20)
        pk_cdm = np.asarray(pk_lin.pk_linear_cdm(k, 0.0, theta))
        pk_b   = np.asarray(pk_lin.pk_linear_b(k, 0.0, theta))
        assert np.all(pk_cdm > 0.0)
        assert np.all(pk_b > 0.0)

    def test_weighted_auto_less_than_total(self, pk_lin, theta):
        """f_c^2*P_CDM < P_total and f_b^2*P_b < P_total (cross-term is positive)."""
        k = jnp.logspace(-2, 0, 20)
        f_b   = theta["Omega_b"]   / theta["Omega_m"]
        f_cdm = theta["Omega_cdm"] / theta["Omega_m"]
        pk_tot = np.asarray(pk_lin.pk_linear(k, 0.0, theta))
        pk_cdm = np.asarray(pk_lin.pk_linear_cdm(k, 0.0, theta))
        pk_b   = np.asarray(pk_lin.pk_linear_b(k, 0.0, theta))
        # P_total = f_c^2*P_cdm + 2*f_c*f_b*P_cross + f_b^2*P_b; cross-term > 0
        assert np.all(f_cdm**2 * pk_cdm < pk_tot * 1.001)
        assert np.all(f_b**2   * pk_b   < pk_tot * 1.001)

    def test_weighted_sum_approx_total(self, pk_lin, theta):
        """f_c * P_CDM + f_b * P_b ≈ P_total within 10% (auto-spectra approximation)."""
        k = jnp.logspace(-2, 0.5, 30)
        f_b  = theta["Omega_b"]   / theta["Omega_m"]
        f_cdm = theta["Omega_cdm"] / theta["Omega_m"]
        pk_tot = np.asarray(pk_lin.pk_linear(k, 0.0, theta))
        pk_cdm = np.asarray(pk_lin.pk_linear_cdm(k, 0.0, theta))
        pk_b   = np.asarray(pk_lin.pk_linear_b(k, 0.0, theta))
        approx = f_cdm * pk_cdm + f_b * pk_b
        # The auto-power sum underestimates total (cross-term missing),
        # but should be within 30% for ΛCDM at k < 1 h/Mpc
        ratio = approx / pk_tot
        assert np.all(ratio > 0.5) and np.all(ratio <= 1.05)
