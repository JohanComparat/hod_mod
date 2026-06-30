"""Tests for mass-dependent baryon fraction models.

Verifies the physical limits, JAX-compatibility, and the delta_sigma_split
interface of FullHaloModelPrediction.
"""

import numpy as np
import pytest
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def theta():
    return {
        "h": 0.6736, "Omega_b": 0.0493, "Omega_cdm": 0.2607,
        "Omega_m": 0.3100, "n_s": 0.9649, "ln10^{10}A_s": 3.044,
    }


@pytest.fixture(scope="module")
def m_grid():
    return jnp.logspace(10, 16, 200)


# ---------------------------------------------------------------------------
# BaryonFractionSigmoid
# ---------------------------------------------------------------------------

class TestBaryonFractionSigmoid:
    def test_cluster_limit(self, theta, m_grid):
        """M >> M_pivot → f_b(M) ≈ f_b^cosmic."""
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
        model = BaryonFractionSigmoid()
        params = {"log10_M_pivot": 13.5, "beta_b": 2.0}
        f_b_cosmic = theta["Omega_b"] / theta["Omega_m"]
        m_cluster = jnp.array([1e16])  # >> M_pivot = 3×10^13
        fb = model(m_cluster, theta, params)
        assert float(fb[0]) == pytest.approx(f_b_cosmic, rel=1e-3)

    def test_low_mass_limit(self, theta):
        """M << M_pivot → f_b(M) ≈ 0."""
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
        model = BaryonFractionSigmoid()
        params = {"log10_M_pivot": 13.5, "beta_b": 2.0}
        m_dwarf = jnp.array([1e9])  # << M_pivot
        fb = model(m_dwarf, theta, params)
        assert float(fb[0]) < 0.01 * theta["Omega_b"] / theta["Omega_m"]

    def test_cdm_plus_baryon_sums_to_one(self, theta, m_grid):
        """(1 - f_b(M)) + f_b(M) = 1 exactly."""
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
        model = BaryonFractionSigmoid()
        params = BaryonFractionSigmoid.default_params()
        fb = model(m_grid, theta, params)
        fc = 1.0 - fb
        np.testing.assert_allclose(np.asarray(fb + fc), np.ones(len(m_grid)), atol=1e-12)

    def test_monotonically_increasing(self, theta, m_grid):
        """f_b(M) increases with mass (more massive halos retain more gas)."""
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
        model = BaryonFractionSigmoid()
        params = BaryonFractionSigmoid.default_params()
        fb = np.asarray(model(m_grid, theta, params))
        assert np.all(np.diff(fb) >= -1e-15)

    def test_bounded_by_cosmic(self, theta, m_grid):
        """f_b(M) ≤ f_b^cosmic at all masses."""
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
        model = BaryonFractionSigmoid()
        params = BaryonFractionSigmoid.default_params()
        fb = np.asarray(model(m_grid, theta, params))
        f_b_cosmic = theta["Omega_b"] / theta["Omega_m"]
        assert np.all(fb <= f_b_cosmic + 1e-10)

    def test_default_params_shape(self):
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
        p = BaryonFractionSigmoid.default_params()
        assert "log10_M_pivot" in p and "beta_b" in p


# ---------------------------------------------------------------------------
# BaryonFractionPowerLaw
# ---------------------------------------------------------------------------

class TestBaryonFractionPowerLaw:
    def test_zero_slope_is_flat(self, theta, m_grid):
        """alpha_b = 0 → f_b(M) = f_b^cosmic at all masses (when M > M_ref)."""
        from hod_mod.observables.baryon_fraction import BaryonFractionPowerLaw
        model = BaryonFractionPowerLaw()
        params = {"log10_M_ref": 14.0, "alpha_b": 0.0}
        fb = model(m_grid, theta, params)
        f_b_cosmic = theta["Omega_b"] / theta["Omega_m"]
        np.testing.assert_allclose(np.asarray(fb), f_b_cosmic, rtol=1e-6)

    def test_clipped_at_cosmic(self, theta, m_grid):
        """f_b(M) never exceeds f_b^cosmic regardless of alpha_b."""
        from hod_mod.observables.baryon_fraction import BaryonFractionPowerLaw
        model = BaryonFractionPowerLaw()
        params = {"log10_M_ref": 10.0, "alpha_b": 5.0}  # large slope → clip
        fb = np.asarray(model(m_grid, theta, params))
        f_b_cosmic = theta["Omega_b"] / theta["Omega_m"]
        assert np.all(fb <= f_b_cosmic + 1e-10)

    def test_non_negative(self, theta, m_grid):
        """f_b(M) ≥ 0 at all masses."""
        from hod_mod.observables.baryon_fraction import BaryonFractionPowerLaw
        model = BaryonFractionPowerLaw()
        params = BaryonFractionPowerLaw.default_params()
        fb = np.asarray(model(m_grid, theta, params))
        assert np.all(fb >= 0.0)

    def test_default_params_shape(self):
        from hod_mod.observables.baryon_fraction import BaryonFractionPowerLaw
        p = BaryonFractionPowerLaw.default_params()
        assert "log10_M_ref" in p and "alpha_b" in p


# ---------------------------------------------------------------------------
# make_baryon_fraction factory
# ---------------------------------------------------------------------------

class TestMakeBaryonFraction:
    def test_sigmoid(self):
        from hod_mod.observables.baryon_fraction import (
            make_baryon_fraction, BaryonFractionSigmoid,
        )
        assert isinstance(make_baryon_fraction("sigmoid"), BaryonFractionSigmoid)

    def test_powerlaw(self):
        from hod_mod.observables.baryon_fraction import (
            make_baryon_fraction, BaryonFractionPowerLaw,
        )
        assert isinstance(make_baryon_fraction("powerlaw"), BaryonFractionPowerLaw)

    def test_invalid_raises(self):
        from hod_mod.observables.baryon_fraction import make_baryon_fraction
        with pytest.raises(ValueError):
            make_baryon_fraction("unknown_model")


# ---------------------------------------------------------------------------
# delta_sigma_split: constant f_b (no baryon_fraction model)
# ---------------------------------------------------------------------------

class TestDeltaSigmaSplitConstant:
    @pytest.fixture(scope="class")
    def pred_and_theta(self):
        """Lightweight FullHaloModelPrediction using Tinker08 HMF."""
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        from hod_mod.core.halo_mass_function import make_hmf
        from hod_mod.core.halo_profiles import HaloProfile
        from hod_mod.connection.hod import HODModel
        from hod_mod.observables.clustering import FullHaloModelPrediction

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
        return pred, theta

    def test_split_sums_to_total(self, pred_and_theta):
        """ds_cdm + ds_b == ds_total to float precision."""
        pred, theta = pred_and_theta
        R = jnp.logspace(-1, 1.0, 8)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        z = 0.1
        result = pred.delta_sigma_split(R, z, theta, hod_p)
        np.testing.assert_allclose(
            np.asarray(result["cdm"] + result["b"]),
            np.asarray(result["total"]),
            rtol=1e-6,
        )

    def test_cdm_fraction_matches_omega(self, pred_and_theta):
        """With constant f_b, ds_cdm / ds_total = Omega_cdm / Omega_m."""
        pred, theta = pred_and_theta
        R = jnp.logspace(-1, 1.0, 8)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        result = pred.delta_sigma_split(R, 0.1, theta, hod_p)
        expected_fcdm = theta["Omega_cdm"] / theta["Omega_m"]
        ratio = np.asarray(result["cdm"] / result["total"])
        np.testing.assert_allclose(ratio, expected_fcdm, rtol=1e-6)


# ---------------------------------------------------------------------------
# delta_sigma_split: mass-dependent f_b(M)
# ---------------------------------------------------------------------------

class TestDeltaSigmaSplitMassDependent:
    @pytest.fixture(scope="class")
    def pred_and_theta(self):
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        from hod_mod.core.halo_mass_function import make_hmf
        from hod_mod.core.halo_profiles import HaloProfile
        from hod_mod.connection.hod import HODModel
        from hod_mod.observables.clustering import FullHaloModelPrediction
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid

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
        bf = BaryonFractionSigmoid()
        pred = FullHaloModelPrediction(pk_lin, hod, hp, baryon_fraction=bf)
        return pred, theta

    def test_split_sums_to_total(self, pred_and_theta):
        """With mass-dependent f_b, split still sums to total."""
        pred, theta = pred_and_theta
        R = jnp.logspace(-1, 1.0, 6)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        bf_p = {"log10_M_pivot": 13.5, "beta_b": 1.5}
        result = pred.delta_sigma_split(R, 0.1, theta, hod_p, baryon_params=bf_p)
        np.testing.assert_allclose(
            np.asarray(result["cdm"] + result["b"]),
            np.asarray(result["total"]),
            rtol=1e-4,  # mass-integrated split uses separate Hankel transforms
        )

    def test_baryon_less_than_cosmic(self, pred_and_theta):
        """Mass-dep f_b at group scale < constant Omega_b/Omega_m fraction."""
        pred, theta = pred_and_theta
        R = jnp.logspace(-1, 1.0, 6)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        bf_p = {"log10_M_pivot": 14.0, "beta_b": 2.0}
        result = pred.delta_sigma_split(R, 0.1, theta, hod_p, baryon_params=bf_p)
        f_b_cosmic = theta["Omega_b"] / theta["Omega_m"]
        ds_total = np.asarray(result["total"])
        ds_b = np.asarray(result["b"])
        # baryon fraction at the effective halo mass (group scale) < cosmic
        ratio = ds_b / ds_total
        assert float(ratio.mean()) < f_b_cosmic


# ---------------------------------------------------------------------------
# delta_sigma_split: gas concentration profile (arXiv:2409.01758 + Mead+2015)
# ---------------------------------------------------------------------------

class TestGasProfileSuppression:
    """Tests for the mass-integrated CDM/gas split in _pk_tables_full.

    Physics: the gas profile uses c_gas = eta(M) * c_DM where
    eta(M) = 1 - (1-eta_min) / (1 + (M/M_eta)^beta_eta)
    (arXiv:2409.01758 Table 2; Mead+2015 arXiv:1611.08606 §2.3).
    A reduced concentration (eta_min < 1) spreads the gas profile,
    suppressing the 1-halo ΔΣ at small R compared to a pure DM profile.
    """

    @pytest.fixture(scope="class")
    def pred_and_theta(self):
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        from hod_mod.core.halo_mass_function import make_hmf
        from hod_mod.core.halo_profiles import HaloProfile
        from hod_mod.connection.hod import HODModel
        from hod_mod.observables.clustering import FullHaloModelPrediction
        from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid

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
        bf = BaryonFractionSigmoid()
        pred = FullHaloModelPrediction(pk_lin, hod, hp, baryon_fraction=bf)
        return pred, theta

    def test_gas_less_concentrated_suppresses_ds(self, pred_and_theta):
        """Reduced gas concentration (eta_min < 1) lowers ΔΣ at small R.

        With eta_min = 0.5 (stronger AGN feedback than arXiv:2409.01758 fiducial
        0.6), the gas profile is more extended → 1-halo ΔΣ suppressed relative
        to eta_min = 1 (c_gas = c_DM, no suppression).
        """
        pred, theta = pred_and_theta
        R = jnp.logspace(-1, 0.5, 6)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        # No gas suppression: eta_min = 1 → c_gas = c_DM
        bp_nodiff = {
            "log10_M_pivot": 13.5, "beta_b": 1.5,
            "log10_eta_min": 0.0,    # 10^0 = 1.0 → c_gas = c_DM
            "log10_M_eta":   13.0,
        }
        # Strong gas suppression: eta_min = 0.5 < fiducial 0.6
        bp_suppress = {
            "log10_M_pivot": 13.5, "beta_b": 1.5,
            "log10_eta_min": -0.30,  # 10^-0.30 ≈ 0.5
            "log10_M_eta":   13.0,
        }
        ds_nodiff   = np.asarray(pred.delta_sigma_split(
            R, 0.1, theta, hod_p, baryon_params=bp_nodiff)["total"])
        ds_suppress = np.asarray(pred.delta_sigma_split(
            R, 0.1, theta, hod_p, baryon_params=bp_suppress)["total"])
        # At small R the 1-halo term dominates; more extended gas → lower ΔΣ
        assert np.all(ds_suppress <= ds_nodiff * 1.001)

    def test_eta_zero_recovers_no_suppression(self, pred_and_theta):
        """eta_min = 1 (c_gas = c_DM) gives same total as delta_sigma().

        When ũ_gas = ũ_DM, the mass-integrated split is identical to the
        standard 1-halo integral regardless of f_b(M):
        (1-f_b)*ũ_DM + f_b*ũ_DM = ũ_DM.
        """
        pred, theta = pred_and_theta
        R = jnp.logspace(-1, 0.5, 6)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        # eta_min = 1: gas profile = DM profile → total unchanged by f_b split
        bp_nodiff = {
            "log10_M_pivot": 13.5, "beta_b": 1.5,
            "log10_eta_min": 0.0,
            "log10_M_eta":   13.0,
        }
        ds_split = np.asarray(pred.delta_sigma_split(
            R, 0.1, theta, hod_p, baryon_params=bp_nodiff)["total"])
        ds_plain = np.asarray(pred.delta_sigma(R, 0.1, theta, hod_p))
        np.testing.assert_allclose(ds_split, ds_plain, rtol=1e-4)

    def test_split_sums_to_total_with_gas_profile(self, pred_and_theta):
        """ds_cdm + ds_b == ds_total with full gas profile parameters.

        Linearity of the Fourier + LOS integrals guarantees this exactly:
        P_gm = P_gm_cdm + P_gm_b → ΔΣ_total = ΔΣ_CDM + ΔΣ_b.
        (arXiv:2409.01758; Mead+2015 arXiv:1611.08606)
        """
        pred, theta = pred_and_theta
        R = jnp.logspace(-1, 0.5, 6)
        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.3,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
        }
        bp = {
            "log10_M_pivot": 13.5, "beta_b": 1.5,
            "log10_eta_min": -0.22,  # fiducial arXiv:2409.01758 group scale
            "log10_M_eta":   13.0,
        }
        result = pred.delta_sigma_split(R, 0.1, theta, hod_p, baryon_params=bp)
        np.testing.assert_allclose(
            np.asarray(result["cdm"] + result["b"]),
            np.asarray(result["total"]),
            rtol=1e-4,
        )
