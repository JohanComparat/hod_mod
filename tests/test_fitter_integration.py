"""Integration coverage for the MAP/MCMC fitters (fitting.fitters).

These run real CAMB + halo-model evaluations and a small optimisation / MCMC, so
they are marked ``slow``. They exercise WpFitter.map_fit/sample, plus the ΔΣ-only
and joint fitters, which the unit tests only constructed but never ran.
"""
import os

import numpy as np
import pytest

from hod_mod.fitting import WpFitConfig, FitConfig, WpFitter, DeltaSigmaFitter, JointFitter
from hod_mod.connection.hod import MoreHODModel

pytestmark = pytest.mark.slow

_PARAM_INIT = MoreHODModel.default_params()
_BWPD_WP = "data/lange2025_desi_dr1/BGS2/wp_bgs2_bwpd.csv"


@pytest.fixture(scope="module")
def wp_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("fit") / "wp.csv"
    rp  = np.logspace(-1, 1.4, 12)
    wp  = 60.0 * rp ** (-0.85)
    err = 0.15 * wp
    np.savetxt(str(p), np.column_stack([rp, wp, err]),
               header="rp_hMpc,wp_hMpc,wp_err_hMpc", delimiter=",", comments="")
    return str(p)


@pytest.fixture(scope="module")
def ds_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("fit") / "ds.csv"
    R   = np.logspace(-1, 1.3, 10)
    ds  = 30.0 * R ** (-0.9)
    err = 0.20 * ds
    np.savetxt(str(p), np.column_stack([R, ds, err]),
               header="R_hMpc,ds_Msun_h_pc2,ds_err_Msun_h_pc2", delimiter=",", comments="")
    return str(p)


def _wp_cfg(wp_csv, **kw):
    base = dict(
        data_file=wp_csv, data_format="csv", rp_min=0.1, rp_max=40.0,
        hod_model="MoreHODModel", hmf_backend="tinker08", z=0.15, pi_max=60.0,
        free_params=["log10mmin"], param_bounds={"log10mmin": (11.0, 13.5)},
        param_init=dict(_PARAM_INIT), output_dir="/tmp/hodmod_fit_test/",
    )
    base.update(kw)
    return WpFitConfig(**base)


def test_wpfitter_map_fit(wp_csv):
    fitter = WpFitter(_wp_cfg(wp_csv))
    res = fitter.map_fit()
    assert set(res) >= {"theta", "params", "chi2", "ndof", "success"}
    assert np.isfinite(res["chi2"]) and res["chi2"] >= 0.0
    assert 11.0 <= float(res["theta"][0]) <= 13.5      # stayed in bounds
    # predict_wp at the best fit is finite and positive
    wp_pred = np.asarray(fitter.predict_wp(res["params"]))
    assert wp_pred.shape == fitter.rp_arr.shape and np.all(np.isfinite(wp_pred)) and np.all(wp_pred > 0)


def test_wpfitter_sample(wp_csv):
    fitter = WpFitter(_wp_cfg(wp_csv, n_walkers=8, n_steps=4, n_burnin=2))
    sampler = fitter.sample(progress=False)
    chain = sampler.get_chain()
    assert chain.shape[1] == 8 and chain.shape[2] == 1     # (steps, walkers, n_free)
    assert np.all(np.isfinite(sampler.get_log_prob()))


def test_deltasigma_map_fit(ds_csv):
    cfg = FitConfig(
        data_file="", data_format="csv", rp_min=0.1, rp_max=40.0,
        hod_model="MoreHODModel", hmf_backend="tinker08", z=0.15, pi_max=60.0,
        free_params=["log10mmin"], param_bounds={"log10mmin": (11.0, 13.5)},
        param_init=dict(_PARAM_INIT), output_dir="/tmp/hodmod_fit_test/",
        ds_file=ds_csv, ds_format="csv", ds_rp_min=0.1, ds_rp_max=40.0,
    )
    fitter = DeltaSigmaFitter(cfg)
    res = fitter.map_fit()
    assert np.isfinite(res["chi2"]) and res["chi2"] >= 0.0
    ds_pred = np.asarray(fitter.predict_ds(res["params"]))
    assert np.all(np.isfinite(ds_pred))


@pytest.mark.skipif(not os.path.exists(_BWPD_WP), reason="digitized bwpd data absent")
def test_wpfitter_bwpd_loader():
    """WpFitter with data_format='bwpd' reads a manually-digitized wp file."""
    cfg = _wp_cfg(_BWPD_WP, data_format="bwpd", z=0.2)
    fitter = WpFitter(cfg)
    assert fitter.rp_arr.shape[0] > 0
    assert np.all(np.isfinite(fitter.wp_obs)) and np.all(fitter.wp_obs > 0)


def test_joint_map_fit(wp_csv, ds_csv):
    cfg = FitConfig(
        data_file=wp_csv, data_format="csv", rp_min=0.1, rp_max=40.0,
        hod_model="MoreHODModel", hmf_backend="tinker08", z=0.15, pi_max=60.0,
        free_params=["log10mmin"], param_bounds={"log10mmin": (11.0, 13.5)},
        param_init=dict(_PARAM_INIT), output_dir="/tmp/hodmod_fit_test/",
        ds_file=ds_csv, ds_format="csv", ds_rp_min=0.1, ds_rp_max=40.0,
        ng_obs=3.0e-4, ng_frac_err=0.2, fit_ng=False,
    )
    fitter = JointFitter(cfg)
    res = fitter.map_fit()
    assert np.isfinite(res["chi2"]) and res["chi2"] >= 0.0
