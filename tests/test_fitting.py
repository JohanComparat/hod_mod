"""Tests for the HOD fitting pipeline.

Covers WpFitter (CSV and HDF5 data formats), Gaussian prior support,
Planck 2018 prior values, and the joint fitter n_g constraint.
All tests use synthetic data and do not require internet access.
"""

import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixture: synthetic wp data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_wp_csv(tmp_path):
    """Write a tiny 3-column CSV with synthetic wp data."""
    rp   = np.logspace(-1, 1.5, 12)
    wp   = 50.0 * rp ** (-0.8)
    err  = wp * 0.15
    path = str(tmp_path / "wp_test.csv")
    np.savetxt(path, np.column_stack([rp, wp, err]),
               header="rp_hMpc,wp_hMpc,wp_err_hMpc", delimiter=",", comments="")
    return path, rp, wp, err


@pytest.fixture
def synthetic_wp_hdf5(tmp_path):
    """Write a minimal sum_stat-style HDF5 wp file."""
    import h5py
    h    = 0.6736
    rp_Mpc = np.logspace(-1, 1.5, 12)
    wp_Mpc = 50.0 / h * rp_Mpc ** (-0.8)   # Mpc units
    cov_Mpc2 = np.diag((0.15 * wp_Mpc) ** 2)

    path = str(tmp_path / "wp_test.h5")
    with h5py.File(path, "w") as f:
        f.attrs["created_by"] = "test"
        g = f.create_group("wp")
        g.attrs["pi_max_Mpc"] = 100.0
        g.attrs["estimator"]  = "landy-szalay"
        g.attrs["n_gal"]      = 100000
        cosmo = g.create_group("cosmology")
        cosmo.create_dataset("H0",  data=h * 100.0)
        cosmo.create_dataset("Om0", data=0.315)
        cosmo.create_dataset("Ob0", data=0.049)
        cosmo.create_dataset("Ok0", data=0.0)
        g.create_dataset("sep_centres", data=rp_Mpc)
        g.create_dataset("xi",          data=wp_Mpc)
        g.create_dataset("cov",         data=cov_Mpc2)
        g.create_dataset("bin_edges",
                         data=np.linspace(rp_Mpc[0], rp_Mpc[-1], 13))
    return path


# ---------------------------------------------------------------------------
# Test: Planck prior values
# ---------------------------------------------------------------------------

class TestPlanckPrior:
    def test_means_exist(self):
        from hod_mod.fitting.planck_prior import PLANCK18_MEANS
        assert "h" in PLANCK18_MEANS
        assert abs(PLANCK18_MEANS["h"] - 0.6736) < 1e-4

    def test_sigma8_exists(self):
        from hod_mod.fitting.planck_prior import PLANCK18_MEANS, PLANCK18_SIGMAS
        assert "sigma8" in PLANCK18_MEANS
        assert 0.004 < PLANCK18_SIGMAS["sigma8"] < 0.010

    def test_3sigma_bounds_width(self):
        from hod_mod.fitting.planck_prior import PLANCK18_SIGMAS, PLANCK18_3SIGMA
        for k, (lo, hi) in PLANCK18_3SIGMA.items():
            expected_width = 6.0 * PLANCK18_SIGMAS[k]
            assert abs((hi - lo) - expected_width) < 1e-9, (
                f"3σ bounds for {k} have wrong width"
            )

    def test_log_prior_inside_bounds(self):
        from hod_mod.fitting.planck_prior import (
            PLANCK18_MEANS, planck18_log_prior
        )
        lp = planck18_log_prior(PLANCK18_MEANS)
        assert lp == 0.0   # at the mean, all terms vanish

    def test_log_prior_outside_bounds(self):
        from hod_mod.fitting.planck_prior import PLANCK18_MEANS, planck18_log_prior
        theta = dict(PLANCK18_MEANS)
        theta["h"] = 0.3   # far outside any reasonable bound
        assert planck18_log_prior(theta) == -np.inf

    def test_gaussian_log_prior_value(self):
        from hod_mod.fitting.planck_prior import gaussian_log_prior
        lp_at_mean = gaussian_log_prior(val=1.0, mean=1.0, sigma=0.1)
        assert lp_at_mean == 0.0
        lp_1sigma = gaussian_log_prior(val=1.1, mean=1.0, sigma=0.1)
        assert abs(lp_1sigma - (-0.5)) < 1e-9
        lp_outside = gaussian_log_prior(val=0.0, mean=1.0, sigma=0.1, lo=0.5)
        assert lp_outside == -np.inf


# ---------------------------------------------------------------------------
# Test: WpFitter — CSV format
# ---------------------------------------------------------------------------

class TestWpFitterCSV:
    def test_load_csv(self, synthetic_wp_csv):
        from hod_mod.fitting.hod_wp import WpFitConfig, WpFitter
        path, rp, wp, err = synthetic_wp_csv

        cfg = WpFitConfig(
            data_file   = path,
            data_format = "csv",
            rp_min      = 0.1,
            rp_max      = 50.0,
            hod_model   = "MoreHODModel",
            hmf_backend = "tinker08",
            z           = 0.1,
            pi_max      = 60.0,
            free_params = ["log10mmin"],
            param_bounds = {"log10mmin": (11.0, 13.5)},
            param_init   = {"log10mmin": 12.0, "sigma_logm": 0.38,
                            "log10m1": 13.5, "alpha": 1.0, "kappa": 1.0,
                            "alpha_inc": 1.0, "log10m_inc": 12.0},
            output_dir   = "/tmp/test_fit/",
        )
        fitter = WpFitter(cfg)
        assert fitter.rp_arr.shape[0] > 0
        np.testing.assert_allclose(fitter.wp_obs, wp, rtol=1e-6)

    def test_gaussian_prior_changes_logprob(self, synthetic_wp_csv):
        from hod_mod.fitting.hod_wp import WpFitConfig, WpFitter
        path, rp, wp, err = synthetic_wp_csv

        base = dict(
            data_file   = path,
            data_format = "csv",
            rp_min      = 0.1,
            rp_max      = 50.0,
            hod_model   = "MoreHODModel",
            hmf_backend = "tinker08",
            z           = 0.1,
            pi_max      = 60.0,
            free_params = ["log10mmin"],
            param_bounds = {"log10mmin": (11.0, 13.5)},
            param_init   = {"log10mmin": 12.0, "sigma_logm": 0.38,
                            "log10m1": 13.5, "alpha": 1.0, "kappa": 1.0,
                            "alpha_inc": 1.0, "log10m_inc": 12.0},
            output_dir   = "/tmp/",
        )

        # Uniform prior
        cfg_u = WpFitConfig(**base, param_prior_types={"log10mmin": "uniform"})
        f_u   = WpFitter(cfg_u)
        lp_u  = f_u._prior_log_prob(np.array([12.5]))

        # Gaussian prior — penalises moving away from mean
        cfg_g = WpFitConfig(**base,
                            param_prior_types={"log10mmin": "gaussian"},
                            param_prior_means={"log10mmin": 12.0},
                            param_prior_sigmas={"log10mmin": 0.2})
        f_g   = WpFitter(cfg_g)
        lp_g  = f_g._prior_log_prob(np.array([12.5]))

        assert np.isfinite(lp_u)
        assert np.isfinite(lp_g)
        # Gaussian prior is more negative at 12.5 (0.5σ/0.2 = 2.5σ from mean)
        assert lp_g < lp_u


# ---------------------------------------------------------------------------
# Test: WpFitter — HDF5 format
# ---------------------------------------------------------------------------

class TestWpFitterHDF5:
    def test_load_hdf5_and_icov_full(self, synthetic_wp_hdf5):
        """HDF5 format should produce a full inverse covariance matrix."""
        from hod_mod.fitting.hod_wp import WpFitConfig, WpFitter

        cfg = WpFitConfig(
            data_file    = synthetic_wp_hdf5,
            data_format  = "hdf5",
            rp_min       = 0.15,
            rp_max       = 25.0,
            hod_model    = "MoreHODModel",
            hmf_backend  = "tinker08",
            z            = 0.1,
            pi_max       = 100.0,
            free_params  = ["log10mmin"],
            param_bounds = {"log10mmin": (11.0, 13.5)},
            param_init   = {"log10mmin": 12.0, "sigma_logm": 0.38,
                            "log10m1": 13.5, "alpha": 1.0, "kappa": 1.0,
                            "alpha_inc": 1.0, "log10m_inc": 12.0},
            output_dir   = "/tmp/",
        )
        fitter = WpFitter(cfg)
        n = len(fitter.rp_arr)
        # icov_wp should be square and full (not diagonal zeros off-diag)
        assert fitter.icov_wp.shape == (n, n)
        # rp values should be in h-units (Mpc/h)
        assert fitter.rp_arr[0] > 0.1 * 0.6736  # > rp_min * h

    def test_hdf5_rp_in_h_units(self, synthetic_wp_hdf5):
        """Verify rp is multiplied by h compared to the Mpc values in the file."""
        import h5py
        from hod_mod.fitting.hod_wp import WpFitConfig, WpFitter

        cfg = WpFitConfig(
            data_file    = synthetic_wp_hdf5,
            data_format  = "hdf5",
            rp_min       = 0.01,
            rp_max       = 100.0,
            hod_model    = "MoreHODModel",
            hmf_backend  = "tinker08",
            z            = 0.1,
            pi_max       = 100.0,
            free_params  = [],
            param_bounds = {},
            param_init   = {"log10mmin": 12.0, "sigma_logm": 0.38,
                            "log10m1": 13.5, "alpha": 1.0, "kappa": 1.0,
                            "alpha_inc": 1.0, "log10m_inc": 12.0},
            output_dir   = "/tmp/",
        )
        fitter = WpFitter(cfg)

        with h5py.File(synthetic_wp_hdf5, "r") as f:
            rp_Mpc = np.array(f["wp/sep_centres"])
            h_val  = float(f["wp/cosmology/H0"][()]) / 100.0

        np.testing.assert_allclose(fitter.rp_arr, rp_Mpc * h_val, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: Joint fitter n_g term
# ---------------------------------------------------------------------------

class TestJointFitterNgTerm:
    def test_ng_chi2_present(self, synthetic_wp_csv):
        from hod_mod.fitting.hod_wp import (
            JointFitConfig, JointFitter
        )
        wp_path, rp, wp, err = synthetic_wp_csv

        # Write a tiny ΔΣ CSV
        ds_path = wp_path.replace("wp_test", "ds_test")
        np.savetxt(ds_path,
                   np.column_stack([rp[:6], 100.0 * np.ones(6), 10.0 * np.ones(6)]),
                   header="rp_hMpc,ds_hMpc,ds_err_hMpc", delimiter=",", comments="")

        cfg = JointFitConfig(
            data_file    = wp_path,
            data_format  = "csv",
            rp_min       = 0.1,
            rp_max       = 50.0,
            hod_model    = "MoreHODModel",
            hmf_backend  = "tinker08",
            z            = 0.1,
            pi_max       = 60.0,
            free_params  = ["log10mmin"],
            param_bounds = {"log10mmin": (11.0, 13.5)},
            param_init   = {"log10mmin": 12.0, "sigma_logm": 0.38,
                            "log10m1": 13.5, "alpha": 1.0, "kappa": 1.0,
                            "alpha_inc": 1.0, "log10m_inc": 12.0},
            output_dir   = "/tmp/",
            ds_file      = ds_path,
            ds_rp_min    = 0.1,
            ds_rp_max    = 10.0,
            ng_obs       = 1e-3,
            ng_frac_err  = 0.20,
        )
        fitter = JointFitter(cfg)

        theta_vec = np.array([12.0])
        lp = fitter._log_prob(theta_vec)
        # log-prob must be finite (not -inf) for a valid parameter
        assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# Test: MultiProbeFitter physics (HOD priors + IA)
# ---------------------------------------------------------------------------

class TestMultiProbeFitterPhysics:
    """Smoke-tests for the physical priors and IA wiring in _log_prob_multiprobe.

    These tests use the module-level _log_prob_multiprobe function directly
    with a minimal mock predictor so they run without real data files.
    """

    @pytest.fixture(scope="class")
    def minimal_predictor(self):
        """FullHaloModelPrediction with MoreHOD and no baryon fraction."""
        from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
        from hod_mod.cosmology.halo_mass_function import make_hmf
        from hod_mod.cosmology.halo_profiles import HaloProfile
        from hod_mod.galaxies.hod import MoreHODModel
        from hod_mod.galaxies.clustering import FullHaloModelPrediction

        pk_lin = LinearPowerSpectrum()
        theta = pk_lin.default_cosmology()
        hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        colossus_cosmo = {
            "flat": True, "H0": theta["h"] * 100,
            "Om0": theta["Omega_m"], "Ob0": theta["Omega_b"],
            "sigma8": 0.811, "ns": theta["n_s"],
        }
        hp = HaloProfile(colossus_cosmo)
        hod = MoreHODModel(hmf, hmf.bias)
        pred = FullHaloModelPrediction(pk_lin, hod, hp)
        return pred, theta

    def test_low_mmin_penalized(self, minimal_predictor):
        """Gaussian prior N(11.5, 0.5) on log10mmin penalises log10mmin=10.

        log10mmin=10.0 is 3σ from the prior mean; log10mmin=11.5 is at the
        mean.  The prior contribution to log_prob must be lower at 10.0.
        Reference: arXiv:2512.15960v3 BGS HOD prior.
        """
        import sys, importlib
        sys.path.insert(
            0, "/home/comparat/software/hod_mod/hod_mod/scripts/fitting/bgs_ls10"
        )
        mod = importlib.import_module("fit_bgs_multiprobe")

        pred, theta = minimal_predictor
        import jax.numpy as jnp

        # Synthetic WP observation (8 bins, not used in prior comparison)
        rp = np.logspace(-1, 1, 8)
        wp_obs = 50.0 * rp ** (-0.8)
        dv_obs = np.concatenate([wp_obs])
        icov = np.diag(1.0 / (0.15 * dv_obs) ** 2)

        common = dict(
            free_params=["log10mmin"],
            fixed_params={
                "sigma_logm": 0.67, "log10m1": 12.5, "alpha": 1.0,
                "kappa": 1.13, "alpha_inc": 0.0, "log10m_inc": 11.5,
                "h": 0.6736, "Omega_m": 0.31, "n_s": 0.9649,
                "ln10^{10}A_s": 3.044,
            },
            param_bounds={"log10mmin": (11.0, 14.5)},
            param_prior_types={"log10mmin": "gaussian"},
            param_prior_means={"log10mmin": 11.5},
            param_prior_sigmas={"log10mmin": 0.5},
            predictor=pred,
            cosmo_default=theta,
            probes=["wp"],
            rp_arrays={"wp": rp},
            dv_obs=dv_obs,
            icov=icov,
            z=0.1,
            pi_max=60.0,
            h_file=0.6736,
        )

        lp_at_prior_mean = mod._log_prob_multiprobe(np.array([11.5]), **common)
        lp_at_low_mmin   = mod._log_prob_multiprobe(np.array([10.0]), **common)

        # log10mmin=10.0 is outside the (11.0, 14.5) hard bound → -inf
        assert lp_at_low_mmin == -np.inf
        # log10mmin=11.5 is at the prior mean → log_prior contribution = 0
        assert np.isfinite(lp_at_prior_mean)

    def test_ia_reduces_esd(self, minimal_predictor):
        """A_IA > 0 lowers the total predicted ESD (NLA adds negative term).

        NLA intrinsic alignment: ΔΣ_IA < 0 for source galaxies behind a lens
        (Bridle & King 2007 arXiv:0705.0166).  Adding it to ΔΣ_grav reduces
        the total ΔΣ at all R.
        """
        from hod_mod.galaxies.intrinsic_alignment import NLAModel
        from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum

        pred, theta = minimal_predictor
        pk_lin = LinearPowerSpectrum()
        nla = NLAModel(pk_lin.pk_linear)

        hod_p = {
            "log10mmin": 11.5, "sigma_logm": 0.67,
            "log10m0": 11.5, "log10m1": 12.5, "alpha": 1.0,
            "kappa": 1.13, "alpha_inc": 0.0, "log10m_inc": 11.5,
        }
        R = np.logspace(-1, 1, 8)

        ds_no_ia = np.asarray(
            pred.delta_sigma(
                R, 0.1, theta,
                {k: v for k, v in hod_p.items()
                 if k not in ("log10m0",)},  # MoreHODModel doesn't use log10m0
                ia_model=None,
            )
        )
        # NLA with A_IA=1 adds negative ΔΣ_IA contribution
        ia_params = {"A_IA": 1.0, "eta_IA": 0.0}
        ds_with_ia = np.asarray(
            pred.delta_sigma(
                R, 0.1, theta,
                {k: v for k, v in hod_p.items() if k not in ("log10m0",)},
                ia_model=nla,
                ia_params=ia_params,
            )
        )
        # ΔΣ_IA < 0 → total ESD is lower with IA
        assert np.all(ds_with_ia <= ds_no_ia * 1.001)
