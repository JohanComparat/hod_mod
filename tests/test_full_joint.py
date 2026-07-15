"""Tests for the BGS full-model joint fit (hod_mod.fitting.full_joint).

Heavy imports (CAMB/HMF/APEC) are done inside the test bodies so collection stays
fast.  Data-dependent tests skip cleanly when the observation files are absent.
"""

import numpy as np
import pytest


def _have_data():
    from hod_mod.paths import data_root
    return (data_root() / "erosita" / "observations" / "XLF_Roster26" /
            "roster26_z0p1.csv").exists()


def test_load_roster26_xlf():
    if not _have_data():
        pytest.skip("Roster26 XLF data not present")
    from hod_mod.fitting.full_joint import load_roster26_xlf
    for z in (0.1, 0.4):
        lx, phi = load_roster26_xlf(z)
        assert lx.ndim == phi.ndim == 1 and lx.size == phi.size >= 10
        assert np.all(np.isfinite(lx)) and np.all(phi > 0)
        assert 39.0 < lx.min() < lx.max() < 46.0        # hard-band log10 L_X range


def test_load_agn_bias_compilation():
    if not _have_data():
        pytest.skip("AGN bias compilation not present")
    from hod_mod.fitting.full_joint import load_agn_bias_compilation
    d = load_agn_bias_compilation()
    for k in ("z", "log10lx_soft", "bias", "bias_err"):
        assert k in d and d[k].ndim == 1 and d[k].size >= 5
    assert np.all(d["bias"] > 0) and np.all(d["bias_err"] > 0)
    assert np.all((d["z"] > 0) & (d["z"] < 1.0))


@pytest.mark.slow
def test_agn_bias_of_lx_monotone_and_finite():
    """b(L_X) is finite, non-decreasing in L_X, and soft-band ≈ hard-band shifted."""
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.agn.powell import PowellAGNModel
    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    pw = PowellAGNModel(pk.default_cosmology(), hmf, z_mean=0.3)
    lx_soft = np.linspace(42.5, 45.0, 12)
    b = pw.agn_bias_of_lx(lx_soft, band="soft")
    assert np.all(np.isfinite(b))
    assert np.all(np.diff(b) >= -1e-6)                  # non-decreasing with luminosity
    assert np.all((b > 0.5) & (b < 4.0))               # physical bias range
    # A soft-band AGN at L_soft has hard luminosity L_soft − log10(k_h2s), so the
    # soft query equals the hard query shifted onto the same internal (hard) grid point.
    b_hard = pw.agn_bias_of_lx(lx_soft - np.log10(pw.k_h2s), band="hard")
    assert np.allclose(b, b_hard, atol=1e-6)


@pytest.mark.slow
@pytest.mark.x64
def test_joint_logprob_finite_and_map_smoke():
    """A cheap sub-vector (wp + n_gal + XLF) builds, gives finite log_prob, and a
    short MAP improves on the fiducial."""
    if not _have_data():
        pytest.skip("observation data not present")
    from hod_mod.fitting.full_joint import JointFull
    J = JointFull(free_zm15=False, observables=("ngal", "wp", "xlf"), verbose=False)
    assert J.ndim == 5                                  # 5 Powell AGN params
    lp0 = J.log_prob(J.x0)
    assert np.isfinite(lp0)
    bd = J.chi2_breakdown(J.x0)
    assert bd["galaxy"] >= 0 and bd["agn"] > 0 and bd["xray"] == 0.0
    res = J.map_fit(maxiter=400)
    # MAP must not be worse than the fiducial (XLF over-predicts at fiducial -> AGN
    # params should pull it down)
    assert -2.0 * J.log_prob(res["theta"]) <= -2.0 * lp0 + 1e-6


@pytest.mark.slow
def test_sample_smoke(tmp_path):
    """sample() runs a tiny chain, writes chain.h5, and a second call resumes it.

    Regression guard: __init__ used to bind ``self.sample`` (the sample NAME),
    which shadowed the ``sample()`` MCMC method -> ``'str' object is not callable``
    the moment a chain was launched.  This exercises both the fresh and resume
    branches of ``JointFull.sample``.
    """
    if not _have_data():
        pytest.skip("observation data not present")
    from hod_mod.fitting.full_joint import JointFull
    J = JointFull(free_zm15=False, observables=("ngal", "wp", "xlf"), verbose=False)
    r = J.sample(tmp_path, n_walkers=12, n_burnin=2, n_steps=3)
    assert (tmp_path / "chain.h5").exists()
    assert np.isfinite(r["acceptance"])
    assert r["flat"].shape[1] == J.ndim
    # resume: a second call continues the SAME backend and runs the remaining steps
    r2 = J.sample(tmp_path, n_walkers=12, n_burnin=2, n_steps=6)
    assert np.isfinite(r2["acceptance"])
    # predict() + the plotter figure builders run end-to-end (feed the doc-page figures)
    from hod_mod.scripts.fitting import plot_bgs_full_joint as P
    theta_med = np.median(r2["flat"], axis=0)
    pred = J.predict(theta_med)
    assert pred["wp"].shape == J.data_gal["wp"]["wp"].shape and np.all(np.isfinite(pred["wp"]))
    assert all(np.all(np.isfinite(v)) for v in pred["xlf"].values())
    comp = J.predict_components(theta_med)
    assert set(comp["wp"]) >= {"1h", "2h", "total"} and np.all(np.isfinite(comp["wp"]["1h"]))
    P.fig_corner(r2["flat"], J.names, theta_med, tmp_path / "corner.png")
    P.fig_observables(J, None, pred, comp, tmp_path / "obs.png")
    assert (tmp_path / "corner.png").exists() and (tmp_path / "obs.png").exists()
    # a stale/incompatible seed (wrong length) must be ignored, not broadcast-crash
    # (regression for the 15-vs-14 map_result.json mismatch)
    d2 = tmp_path / "mismatch"
    r3 = J.sample(d2, n_walkers=12, n_burnin=2, n_steps=3, x_start=np.zeros(J.ndim + 1))
    assert (d2 / "chain.h5").exists() and np.isfinite(r3["acceptance"])
