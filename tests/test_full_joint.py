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


def test_gas_figure_reads_only_live_band_params():
    """The gas figure's parameter names must stay in step with the band model.

    The native-DPM re-base swapped the four gas parameters (lx_norm/lx_slope/
    kt_norm/kt_slope -> log10_ne03/beta_n/log10_p03/beta_P) and the plotter kept
    reading the old ones, which only surfaced as a KeyError once a chain was
    plotted.  Pin the two lists together instead.
    """
    from hod_mod.scripts.fitting import plot_bgs_full_joint as P
    from hod_mod.scripts.fitting import fit_xray_joint_bands as XB
    assert set(P._GAS_KEYS) <= set(XB._PARAMS), (
        f"plotter reads {sorted(set(P._GAS_KEYS) - set(XB._PARAMS))}, which the band "
        f"model no longer fits ({XB._PARAMS})")
    with pytest.raises(KeyError, match="native-DPM"):
        P._gas({"lx_norm": 44.7, "lx_slope": 1.6})


def test_revive_restores_rank_of_a_collapsed_ensemble():
    """A resumed ensemble that emcee rejects must come back with its rank restored.

    Regression guard for the failure that froze ``bgs_full_joint_allparams_v0.3``
    at 1061/4000: a long stretch-move run collapsed the walkers onto a
    lower-dimensional affine subspace, emcee's ``walkers_independent`` check
    (cond > 1e8) refused the restart, and every resubmission died before its
    first step.  ``skip_initial_state_check=True`` alone is not a fix -- the
    stretch move cannot leave the affine hull of the walkers -- so the guard
    asserts the ensemble is genuinely independent again, not merely accepted.
    """
    from emcee.ensemble import walkers_independent
    from hod_mod.fitting.mcmc_resume import revive_ensemble

    nw, nd = 64, 28
    rng = np.random.default_rng(0)
    # a rank-15-of-28 ensemble, i.e. the shape the real chain collapsed into
    basis = rng.standard_normal((15, nd))
    coords = rng.standard_normal((nw, 15)) @ basis * 0.01
    assert not walkers_independent(coords)

    class _Backend:                       # minimal stand-in for an emcee backend
        def __init__(self, c): self._c = c
        def get_last_sample(self): return type("S", (), {"coords": self._c})()

    lo, hi = np.full(nd, -50.0), np.full(nd, 50.0)
    revived = revive_ensemble(_Backend(coords), lo, hi)
    assert revived is not None and revived.shape == coords.shape
    assert walkers_independent(revived), "re-scatter did not restore rank"
    # and it must stay in the neighbourhood: no walker moved by more than a few
    # times the per-parameter spread it was re-scattered with
    assert np.max(np.abs(revived - coords)) < 1.0

    # a healthy ensemble is left alone (None => emcee continues from its state)
    assert revive_ensemble(_Backend(rng.standard_normal((nw, nd))), lo, hi) is None


@pytest.mark.slow
def test_gas_sector_seed_and_prior_are_live():
    """The gas sector must be seeded off its bounds and actually carry its prior.

    Three regressions this pins, all of which shipped in the v0.3/v0.31 campaigns
    because every existing JointFull test used observables=("ngal","wp","xlf") and
    therefore never entered the X-ray branch at all:

    * the seed was ``mu_s`` (the SCALING-RELATION prior centre) used as if it were
      ``mu_n``, which clipped three of four gas params onto a bound and pinned the
      MAP there (v0.3: log10_ne03 = -3.0002 against a bound of -3.0);
    * the full-covariance induced gas prior was never added to ``log_prior`` -- the
      diagonal ``pri_sig`` is inf for those four by design -- so the gas sector ran
      on bounds alone and reached beta_P - beta_n = -0.41, an INVERTED kT-M relation;
    * ``--kt-prior-sig`` wrote onto indices 2/3, which after the re-base are
      log10_p03 and beta_P, not kt_norm/kt_slope.
    """
    if not _have_data():
        pytest.skip("observation data not present")
    from hod_mod.fitting.full_joint import JointFull
    from hod_mod.scripts.fitting import fit_xray_joint_bands as XB

    J = JointFull(free_zm15=False, observables=("ngal", "wp", "xray_bands"), verbose=False)

    # (1) the four gas slots are where we think they are -- so an index-based
    #     prior edit can never silently retarget another parameter again
    assert J._gas_idx is not None
    for k, i in enumerate(J._gas_idx):
        assert J.names[i] == XB._PARAMS[k]

    # (2) seeded at the induced-prior centre, strictly inside the bounds
    for k, i in enumerate(J._gas_idx):
        assert J.lo[i] + 1e-3 < J.x0[i] < J.hi[i] - 1e-3, f"{J.names[i]} seeded on a bound"
        assert abs(J.x0[i] - XB._MU8[k]) < 1e-6

    # (3) log_prior actually responds to the gas parameters.  Step 3 sigma along
    #     the STIFFEST direction of the induced covariance; before the fix this
    #     difference was exactly 0.0 for any gas displacement whatsoever.
    icov = XB._GAS_PRIOR["icov"]
    w, V = np.linalg.eigh(icov)
    stiff = V[:, -1] / np.sqrt(w[-1])            # 1 sigma along the tightest direction
    theta = J.x0.copy()
    base = J.log_prior(theta)
    theta[J._gas_idx] = theta[J._gas_idx] + 3.0 * stiff
    shifted = J.log_prior(theta)
    assert np.isfinite(base) and np.isfinite(shifted)
    assert base - shifted > 1.0, "gas sector carries no prior (bounds-only)"
    assert abs((base - shifted) - 4.5) < 0.5, "3 sigma should cost ~4.5 in log-prior"
