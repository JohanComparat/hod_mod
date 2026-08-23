"""Batched evaluation over HOD parameter sets.

The correctness requirement is that ``vmap`` reproduces the Python loop it
replaces; the performance claim is measured in ``benchmark.py``, not here.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from hod_mod.observables.batched import (
    map_over_hod, batched_wp, batched_delta_sigma,
)
from hod_mod.connection.hod import ZuMandelbaum15HODModel

THETA = {"Omega_m": 0.31, "Omega_b": 0.0493, "h": 0.6736,
         "n_s": 0.9649, "sigma8": 0.8111}
THRESH = [10.0, 10.5, 11.0]


@pytest.fixture(scope="module")
def pred():
    from hod_mod.observables.clustering import make_differentiable_prediction
    return make_differentiable_prediction("zumand15")


@pytest.fixture(scope="module")
def base():
    return ZuMandelbaum15HODModel.default_params()


class TestArgumentChecking:
    def test_rejects_empty_varying(self, base):
        with pytest.raises(ValueError, match="nothing to batch"):
            map_over_hod(lambda p: 0.0, base, {})

    def test_rejects_unknown_key(self, base):
        """A batched key the model never reads is silently ineffective; catch it."""
        with pytest.raises(KeyError, match="absent from base_params"):
            map_over_hod(lambda p: 0.0, base, {"not_a_param": [1.0, 2.0]})

    def test_rejects_ragged_batch(self, base):
        with pytest.raises(ValueError, match="same length"):
            map_over_hod(lambda p: 0.0, base,
                         {"log10m_star_thresh": [1.0, 2.0], "fc": [1.0]})

    def test_scalar_is_promoted_to_length_one(self, base):
        out = map_over_hod(lambda p: p["fc"] * 2.0, base, {"fc": 0.5})
        assert np.asarray(out).shape == (1,)


class TestCorrectness:
    def test_matches_the_loop_for_wp(self, pred, base):
        rp = jnp.logspace(-1, 1.3, 12)
        batched = np.asarray(
            batched_wp(pred, rp, 60.0, 0.2, THETA, base,
                       {"log10m_star_thresh": THRESH})
        )
        loop = np.array([
            np.asarray(pred.wp(rp, 60.0, 0.2, THETA, {**base, "log10m_star_thresh": t}))
            for t in THRESH
        ])
        assert batched.shape == loop.shape
        assert np.allclose(batched, loop, rtol=1e-4)

    def test_matches_the_loop_for_delta_sigma(self, pred, base):
        R = jnp.logspace(-1, 1.0, 10)
        batched = np.asarray(
            batched_delta_sigma(pred, R, 0.2, THETA, base,
                                {"log10m_star_thresh": THRESH})
        )
        loop = np.array([
            np.asarray(pred.delta_sigma(R, 0.2, THETA, {**base, "log10m_star_thresh": t}))
            for t in THRESH
        ])
        assert np.allclose(batched, loop, rtol=1e-4)

    def test_multiple_varying_parameters(self, pred, base):
        rp = jnp.logspace(-1, 1.0, 8)
        varying = {"log10m_star_thresh": [10.0, 10.5], "fc": [0.8, 0.9]}
        batched = np.asarray(batched_wp(pred, rp, 60.0, 0.2, THETA, base, varying))
        loop = np.array([
            np.asarray(pred.wp(rp, 60.0, 0.2, THETA,
                               {**base, "log10m_star_thresh": t, "fc": f}))
            for t, f in zip(varying["log10m_star_thresh"], varying["fc"])
        ])
        assert np.allclose(batched, loop, rtol=1e-4)

    def test_ordering_is_preserved(self, pred, base):
        """Row i must correspond to varying[i], not to some permutation."""
        rp = jnp.logspace(-1, 1.0, 8)
        fwd = np.asarray(batched_wp(pred, rp, 60.0, 0.2, THETA, base,
                                    {"log10m_star_thresh": THRESH}))
        rev = np.asarray(batched_wp(pred, rp, 60.0, 0.2, THETA, base,
                                    {"log10m_star_thresh": THRESH[::-1]}))
        assert np.allclose(fwd, rev[::-1], rtol=1e-5)

    def test_higher_threshold_clusters_more(self, pred, base):
        """A physical sanity check on the batch, not just self-consistency."""
        rp = jnp.logspace(0.0, 1.0, 6)
        out = np.asarray(batched_wp(pred, rp, 60.0, 0.2, THETA, base,
                                    {"log10m_star_thresh": [10.0, 11.0]}))
        assert np.all(out[1] > out[0]), "more massive sample should be more clustered"


class TestJax:
    def test_batched_result_is_differentiable(self, pred, base):
        """The batch must not break the gradient path it exists to serve."""
        rp = jnp.logspace(-1, 1.0, 6)

        def total(fc):
            return jnp.sum(batched_wp(pred, rp, 60.0, 0.2, THETA,
                                      {**base, "fc": fc},
                                      {"log10m_star_thresh": THRESH}))
        g = float(jax.grad(total)(0.86))
        assert np.isfinite(g) and g != 0.0

    def test_jittable(self, pred, base):
        rp = jnp.logspace(-1, 1.0, 6)
        f = jax.jit(lambda t: batched_wp(pred, rp, 60.0, 0.2, THETA, base,
                                         {"log10m_star_thresh": t}))
        assert np.all(np.isfinite(np.asarray(f(jnp.array(THRESH)))))
