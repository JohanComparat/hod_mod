"""Benchmark-observables loader + the MAP/MCMC harness wiring.

Unit tests for `hod_mod.fitting.benchmark_data` (pure-numpy: alignment,
interpolation, block selection, observed-sample reads) on a synthetic tree, plus
a guarded gradient-MAP recovery test that drives the forecast ForwardModel
against the loader's forecast-noise model — the end-to-end check that the JSON
tree snaps into `MultiProbeGaussianLikelihood.from_forward_model`.
"""

import json
import os

import numpy as np
import pytest

from hod_mod.fitting import benchmark_data as bd


# --------------------------------------------------------------------------
# synthetic tree helpers
# --------------------------------------------------------------------------

def _write(root, wl, tr, ref, obs, data, provenance="simulated"):
    d = os.path.join(root, wl, tr)
    os.makedirs(d, exist_ok=True)
    body = {"schema_version": 1, "wavelength": wl, "tracer": tr,
            "observable": obs,
            "reference": {"key": ref, "citation": f"{ref} et al.",
                          "arxiv": "https://arxiv.org/abs/0000.00000"},
            "provenance": {"type": provenance,
                           "needs_operator_extraction": provenance != "observed"},
            "data": data}
    with open(os.path.join(d, f"{ref}__{obs}__{provenance[:3]}.json"
                           if provenance != "observed"
                           else f"{ref}__{obs}.json"), "w") as fh:
        json.dump(body, fh)


@pytest.fixture
def synth_tree(tmp_path):
    root = str(tmp_path / "benchmark_observables")
    # one abundance observable "xlf" over two z-blocks (nearest-z selection)
    x = list(np.linspace(42.0, 45.0, 7))
    y_lo = list(10.0 ** (-3 - 0.5 * (np.array(x) - 42)))
    y_hi = list(10.0 ** (-3.3 - 0.5 * (np.array(x) - 42)))
    _write(root, "xray", "agn", "Aird2015", "xlf",
           {"x": x + x, "y": y_lo + y_hi,
            "y_err": list(0.1 * np.array(y_lo)) + list(0.1 * np.array(y_hi)),
            "block": ["z0.1"] * 7 + ["z0.9"] * 7,
            "zeff": [0.1] * 7 + [0.9] * 7})
    # a projected observable "wp_agn"
    rp = list(np.logspace(-1, 1.3, 6))
    _write(root, "xray", "agn", "Comparat2023", "wp_agn",
           {"x": rp, "y": list(50.0 / np.array(rp)),
            "y_err": list(5.0 / np.array(rp)),
            "block": ["b0"] * 6, "zeff": [0.2] * 6})
    # an observed wp sample (rp, wp, wp_err columns)
    _write(root, "optical", "galaxies", "Zehavi2005", "wp",
           {"rp_hMpc": rp, "wp": list(80.0 / np.array(rp)),
            "wp_err": list(8.0 / np.array(rp))}, provenance="observed")
    return root


# --------------------------------------------------------------------------
# unit tests (no JAX)
# --------------------------------------------------------------------------

def test_interp_onto_is_exact_on_grid():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 10.0 ** np.array([1.0, 2.0, 1.5, 0.5])
    np.testing.assert_allclose(bd._interp_onto(x, x, y), y, rtol=1e-12)
    # clamps beyond the range (no extrapolation blow-up)
    assert bd._interp_onto(np.array([0.0]), x, y)[0] == pytest.approx(y[0])
    assert bd._interp_onto(np.array([9.0]), x, y)[0] == pytest.approx(y[-1])


def test_available_observables(synth_tree):
    assert set(bd.available_observables(synth_tree, "simulated")) == {"xlf", "wp_agn"}
    assert set(bd.available_observables(synth_tree, "observed")) == {"wp"}


def test_block_selection_picks_nearest_zeff(synth_tree):
    # a "model" that asks for xlf on its own 5-point grid at z_eff=0.15 -> z0.1 block
    row_obs = np.array(["xlf"] * 5)
    row_x = np.linspace(42.2, 44.8, 5)
    out = bd.load_forecast_vector(synth_tree, ["xlf"], row_obs, row_x, z_eff=0.15)
    # z0.1 block normalisation is 10**-3 at x=42; z0.9 is 10**-3.3 -> check we got z0.1
    expected_at_42 = 10.0 ** (-3 - 0.5 * (42.2 - 42))
    assert out["data"][0] == pytest.approx(expected_at_42, rel=0.05)
    assert out["data"].shape == (5,)
    assert np.all(np.isfinite(out["data"])) and np.all(out["y_err"] > 0)


def test_forecast_vector_alignment_and_icov(synth_tree):
    row_obs = np.array(["xlf"] * 4 + ["wp_agn"] * 4)
    row_x = np.concatenate([np.linspace(42.5, 44.5, 4),
                            np.logspace(-0.5, 1.0, 4)])
    out = bd.load_forecast_vector(synth_tree, ["xlf", "wp_agn"],
                                  row_obs, row_x, z_eff=0.2)
    assert out["data"].shape == (8,) and out["icov"].shape == (8, 8)
    # diagonal icov == 1/y_err^2
    np.testing.assert_allclose(np.diag(out["icov"]), 1.0 / out["y_err"] ** 2)
    assert set(out["sources"]) == {"xlf", "wp_agn"}


def test_missing_observable_raises(synth_tree):
    with pytest.raises(KeyError):
        bd.load_forecast_vector(synth_tree, ["does_not_exist"],
                                np.array(["does_not_exist"]), np.array([1.0]))


def test_load_observed_sample(synth_tree):
    x, y, e, meta = bd.load_observed_sample(synth_tree, "Zehavi2005", "wp")
    assert x.shape == y.shape == e.shape and x.size == 6
    assert np.all(np.isfinite(y)) and meta["columns"][0] == "rp_hMpc"


# --------------------------------------------------------------------------
# guarded integration test: real tree + gradient MAP recovery
# --------------------------------------------------------------------------

def _real_tree():
    env = os.environ.get("HOD_MOD_DATA_DIR")
    for cand in ([os.path.join(env, "benchmark_observables")] if env else []) + \
            ["/home/comparat/data/hod_mod_data/benchmark_observables"]:
        if os.path.isdir(cand):
            return cand
    return None


@pytest.mark.slow
@pytest.mark.x64
def test_forecast_map_recovers_fiducial_from_benchmark_noise():
    import jax
    if not jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=1")
    root = _real_tree()
    if root is None:
        pytest.skip("benchmark_observables tree not available")
    import jax.numpy as jnp
    from hod_mod.forecast.forward_jax import ForwardModel
    from hod_mod.forecast import params
    from hod_mod.fitting.jax_inference import (
        MultiProbeGaussianLikelihood, run_map_jax)

    which = ["xlf", "wp_agn"]           # abundance + projected: cheap, no Limber
    fm = ForwardModel(z_eff=0.2, n_k=96, n_m=96)
    f, row_obs, row_x = fm.full_data_vector_fn(which)
    bench = bd.load_forecast_vector(root, which, row_obs, row_x, z_eff=0.2)
    sigma = np.asarray(bench["y_err"], float)

    theta0 = params.fiducial_vector()
    truth = np.asarray(f(jnp.asarray(theta0)), float)
    rng = np.random.default_rng(0)
    data = truth + sigma * rng.standard_normal(truth.shape)
    icov = np.diag(1.0 / sigma ** 2)
    free = ["Omega_m", "sigma8"]
    like = MultiProbeGaussianLikelihood.from_forward_model(
        fm, which, data, icov, free, theta0=theta0, prior="planck")

    x_true = theta0[[params.PARAM_NAMES.index(n) for n in free]]
    res = run_map_jax(like, x_true * np.array([1.02, 0.98]),
                      bounds=[(0.1, 0.5), (0.5, 1.1)])
    assert res["success"]
    assert res["chi2"] / max(data.size - len(free), 1) < 3.0
    np.testing.assert_allclose(res["x"], x_true, rtol=0.1)
