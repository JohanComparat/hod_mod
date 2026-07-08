"""Wave-3 differentiable multi-probe inference (fitting/jax_inference.py).

Validates the gradient MAP + blackjax NUTS on the forecast ForwardModel, over a
joint data vector spanning galaxy clustering, galaxy-galaxy lensing, tSZ, X-ray,
cosmic shear and AGN — the differentiable multi-probe fit that Waves 1-2 unlocked.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

_WHICH = ["wp", "ds", "cl_gy", "cl_gX", "cl_kk", "xlf"]
_FREE = ["Omega_m", "sigma8", "lg_m1h", "lg_m0star"]


def _small_model():
    from hod_mod.forecast.forward_jax import ForwardModel
    return ForwardModel(z_eff=0.2, n_k=128, n_m=128, n_gl=48, n_z=5,
                        rp_wp=np.logspace(-1, 1.3, 6),
                        rp_ds=np.logspace(-1, 1.3, 6),
                        ell=np.logspace(1, 3, 6))


class TestLikelihoodConstruction:
    def test_synthetic_shapes_and_packing(self):
        from hod_mod.fitting.jax_inference import MultiProbeGaussianLikelihood

        fm = _small_model()
        like, x_true = MultiProbeGaussianLikelihood.synthetic(
            fm, _WHICH, _FREE, rel_err=0.05, seed=0)
        assert like._data.shape[0] > 0
        assert like._icov.shape == (like._data.shape[0],) * 2
        assert like.free_names == _FREE
        assert x_true.shape == (len(_FREE),)
        # unpack puts the free slice back into a full 111-vector, rest fixed
        full = like.unpack(jnp.asarray(x_true))
        from hod_mod.forecast import params
        assert full.shape == (params.N_PARAM,)
        np.testing.assert_allclose(np.asarray(full)[list(like._free_idx)], x_true)

    def test_planck_prior_penalises_cosmology_only(self):
        from hod_mod.fitting.jax_inference import MultiProbeGaussianLikelihood

        fm = _small_model()
        like, x_true = MultiProbeGaussianLikelihood.synthetic(
            fm, _WHICH, _FREE, rel_err=0.05, seed=0, prior="planck")
        # moving Omega_m off fiducial costs prior; moving lg_m1h (flat) does not
        xc = jnp.asarray(x_true).at[0].add(0.02)     # Omega_m
        xh = jnp.asarray(x_true).at[2].add(0.5)      # lg_m1h (no prior)
        assert float(like.log_prior(xc)) < float(like.log_prior(jnp.asarray(x_true)))
        assert float(like.log_prior(xh)) == pytest.approx(
            float(like.log_prior(jnp.asarray(x_true))), abs=1e-9)


@pytest.mark.x64
class TestGradientAndMap:
    def test_value_and_grad_vs_fd(self):
        if not jax.config.jax_enable_x64:
            pytest.skip("requires JAX_ENABLE_X64=1")
        from hod_mod.fitting.jax_inference import MultiProbeGaussianLikelihood

        fm = _small_model()
        like, x_true = MultiProbeGaussianLikelihood.synthetic(
            fm, _WHICH, _FREE, rel_err=0.05, seed=1)
        x0 = np.asarray(x_true) * np.array([1.03, 0.97, 1.01, 0.99])
        _, g = jax.value_and_grad(like.neg_logprob)(jnp.asarray(x0))
        eps = 1e-4
        fd = np.array([
            (float(like.neg_logprob(jnp.asarray(x0).at[i].add(eps)))
             - float(like.neg_logprob(jnp.asarray(x0).at[i].add(-eps)))) / (2 * eps)
            for i in range(len(x0))])
        np.testing.assert_allclose(np.asarray(g), fd, rtol=1e-4)

    def test_map_recovers_injected_parameters(self):
        if not jax.config.jax_enable_x64:
            pytest.skip("requires JAX_ENABLE_X64=1")
        from hod_mod.fitting.jax_inference import (
            MultiProbeGaussianLikelihood, run_map_jax)

        fm = _small_model()
        like, x_true = MultiProbeGaussianLikelihood.synthetic(
            fm, _WHICH, _FREE, rel_err=0.05, seed=1)
        x0 = np.asarray(x_true) * np.array([1.05, 0.95, 1.02, 0.98])
        res = run_map_jax(like, x0)
        assert res["success"]
        ndof = like._data.shape[0] - len(_FREE)
        assert res["chi2"] / ndof < 2.0
        # recovered within 5% (noise + EH98 surrogate level)
        np.testing.assert_allclose(res["x"], x_true, rtol=0.05)


@pytest.mark.slow
@pytest.mark.x64
class TestNuts:
    def test_nuts_posterior_mean_near_truth(self):
        if not jax.config.jax_enable_x64:
            pytest.skip("requires JAX_ENABLE_X64=1")
        pytest.importorskip("blackjax")
        from hod_mod.forecast.forward_jax import ForwardModel
        from hod_mod.fitting.jax_inference import (
            MultiProbeGaussianLikelihood, run_map_jax, run_nuts)

        # Projected + abundance probes for a tractable NUTS run: the Limber
        # angular spectra (cl_*) inflate the NUTS trajectory compile ~10x, so
        # sampling over them is impractical (MAP handles them fine — see
        # TestGradientAndMap on the full 6-probe vector). This still exercises a
        # genuine multi-probe posterior: clustering + g-g lensing + AGN XLF.
        fm = ForwardModel(z_eff=0.2, n_k=96, n_m=96, n_gl=32, n_z=3,
                          rp_wp=np.logspace(-1, 1.3, 6),
                          rp_ds=np.logspace(-1, 1.3, 6))
        which = ["wp", "ds", "xlf"]
        like, x_true = MultiProbeGaussianLikelihood.synthetic(
            fm, which, _FREE, rel_err=0.05, seed=1)
        res = run_map_jax(like, np.asarray(x_true) * 1.02)
        out = run_nuts(like, res["x"], n_warmup=100, n_samples=200, seed=0)
        assert out["samples"].shape == (200, len(_FREE))
        assert 0.5 < out["accept_rate"] <= 1.0
        # posterior mean within ~2 sigma of the injected truth
        z = np.abs(out["mean"] - x_true) / np.maximum(out["std"], 1e-6)
        assert np.max(z) < 3.0
