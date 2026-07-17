r"""Fast tests for the JAX forecast forward model
(:mod:`hod_mod.forecast.forward_jax`).

Two layers, both cheap:

* the differentiable *primitives* (GL nodes, gNFW shapes, the normalised profile
  FT, the ZM15 inverse SHMR and occupation), tested in isolation on small arrays;
* a tiny-grid :class:`ForwardModel` **smoke test** that every observable in
  ``OBSERVABLES`` is produced finite with the right shape, and that the
  active-fraction identity ``∂lnΦ/∂log10 f_ERDF = ln 10`` holds for the XLF.

The heavy production-match / finite-difference validation lives in
``hod_mod/forecast/tests/test_forward_matches_production.py``; this module is the
fast gate that runs in the default ``tests/`` suite.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

# ForwardModel builds the default CosmoPower P(k) emulator unconditionally;
# without the package these tests would error at fixture time, not skip.
pytest.importorskip("cosmopower_jax")

from hod_mod.forecast.forward_jax import (  # noqa: E402
    ForwardModel, OBSERVABLES, PARAM_NAMES, _IDX,
    _gl_nodes, _gnfw, _gnfw_sq, _gnfw_pressure, _profile_uk_normalized,
    _inv_shmr, _n_cen, _n_sat,
)
from hod_mod.connection.hod.zumandelbaum15 import (  # noqa: E402
    _mstar_from_mh_zu15, _mh_from_mstar_zu15,
)
from hod_mod.forecast import params  # noqa: E402

FID = jnp.asarray(params.fiducial_vector())

# tiny grids keep the smoke test to a couple of seconds
_TINY = dict(n_k=48, n_m=48, n_gl=16, n_z=3, n_z_shear=3)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def test_gl_nodes_on_unit_interval():
    x, w = _gl_nodes(16)
    x, w = np.asarray(x), np.asarray(w)
    assert x.shape == (16,) and w.shape == (16,)
    assert np.all((x >= 0) & (x <= 1))
    assert np.sum(w) == pytest.approx(1.0, rel=1e-10)       # ∫_0^1 dx = 1


def test_gnfw_shapes_positive_and_squared_consistent():
    x = jnp.logspace(-2, 1, 40)
    f = np.asarray(_gnfw(x, 0.9, 2.0, 3.0))
    f2 = np.asarray(_gnfw_sq(x, 0.9, 2.0, 3.0))
    fp = np.asarray(_gnfw_pressure(x, 0.31, 1.05, 5.49))
    assert np.all(f > 0) and np.all(fp > 0)
    np.testing.assert_allclose(f2, f ** 2, rtol=1e-10)      # n_e² shape = f²
    # steeper outer slope → faster fall-off at large x
    f_steep = np.asarray(_gnfw(x, 0.9, 2.0, 4.0))
    assert f_steep[-1] < f[-1]


def test_profile_ft_normalised_to_unity_at_k0_and_decays():
    gx, gw = _gl_nodes(24)
    r_max = jnp.array([1.0, 2.0])                            # two halos
    f_nodes = jnp.ones((2, 24))                              # flat profile
    k = jnp.array([1e-6, 1.0, 5.0])
    uk = np.asarray(_profile_uk_normalized(k, r_max, f_nodes, gx, gw))
    assert uk.shape == (3, 2)
    np.testing.assert_allclose(uk[0], 1.0, atol=1e-6)        # û(k→0) = 1
    assert np.all(uk[1] < 1.0) and np.all(uk[2] < uk[1])     # decays with k


def test_inv_shmr_matches_forward_and_round_trips():
    log10mh = jnp.linspace(11.0, 15.0, 20)
    args = (11.7, 10.5, 0.73, 0.66, 0.54)
    ms = _inv_shmr(log10mh, *args)
    np.testing.assert_allclose(np.asarray(ms),
                               np.asarray(_mstar_from_mh_zu15(log10mh, *args)), rtol=1e-10)
    # round trip: M_h(M_*(M_h)) = M_h
    back = _mh_from_mstar_zu15(ms, *args)
    np.testing.assert_allclose(np.asarray(back), np.asarray(log10mh), atol=1e-3)


def test_inv_shmr_gradient_is_finite():
    # the custom_jvp is the whole reason this wrapper exists — check it traces
    g = jax.grad(lambda lm: jnp.sum(_inv_shmr(lm, 11.7, 10.5, 0.73, 0.66, 0.54)))
    val = np.asarray(g(jnp.linspace(12.0, 14.0, 5)))
    assert np.all(np.isfinite(val)) and np.all(val > 0)      # M_* increases with M_h


def test_occupation_centrals_bounded_and_monotonic():
    log10mh = jnp.linspace(10.0, 15.0, 60)
    fc = 0.3
    nc = np.asarray(_n_cen(log10mh, 10.0, 11.7, 10.5, 0.73, 0.66, 0.54, 1.0, -0.3, fc))
    assert np.all(nc >= 0) and np.all(nc <= fc + 1e-9)      # 0 ≤ N_cen ≤ f_c
    assert nc[-1] > nc[0]                                    # rises with halo mass


def test_occupation_satellites_nonnegative_and_grow():
    log10mh = jnp.linspace(11.0, 15.0, 60)
    ns = np.asarray(_n_sat(log10mh, 10.0, 11.7, 10.5, 0.73, 0.66, 0.54, 1.0, -0.3,
                           0.3, 36.0, 0.9, 0.86, 0.41, 1.0))
    assert np.all(ns >= 0)
    assert ns[-1] > ns[len(ns) // 2]                        # more satellites in big halos


# --------------------------------------------------------------------------
# tiny-grid ForwardModel smoke test
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model():
    return ForwardModel(**_TINY)


def test_predict_all_observables_finite(tiny_model):
    out = tiny_model.predict(FID)
    assert set(out) == set(OBSERVABLES)
    for name, arr in out.items():
        a = np.asarray(arr)
        assert np.all(np.isfinite(a)), f"{name} has non-finite entries"
    # abundances / spectra are physically non-negative
    for name in ("wp", "n_gal", "smf", "xlf", "cl_kk", "cl_XX"):
        assert np.all(np.asarray(out[name]) >= 0), f"{name} negative"
    # n_gal is a single positive number
    assert np.asarray(out["n_gal"]).ravel()[0] > 0


def test_grids_match_observable_lengths(tiny_model):
    out = tiny_model.predict(FID)
    for name in OBSERVABLES:
        assert np.asarray(out[name]).shape == np.asarray(tiny_model.grid_of(name)).shape


def test_xlf_active_fraction_identity(tiny_model):
    # ∂lnΦ/∂log10 f_ERDF = ln 10 exactly (f_ERDF is a pure amplitude on the XLF)
    j = _IDX["agn_log10_ferdf"]

    def logphi_mean(f):
        th = FID.at[j].set(f)
        return jnp.mean(jnp.log(tiny_model._xlf(th)))       # mean → per-bin identity

    dlog = float(jax.grad(logphi_mean)(FID[j]))
    assert dlog == pytest.approx(np.log(10.0), rel=1e-4)


def test_scale_cut_mask_selects_rows(tiny_model):
    f, row_obs, row_x = tiny_model.full_data_vector_fn()
    keep = tiny_model.scale_cut_mask(row_obs, row_x, rmin=1.0)
    assert keep.shape == row_obs.shape and keep.dtype == bool
    # projected wp/ds rows honour the r_p > rmin cut; abundances always kept
    assert keep[row_obs == "n_gal"].all()
    wp_rows = row_obs == "wp"
    if wp_rows.any():
        assert np.all(keep[wp_rows] == (row_x[wp_rows] > 1.0))
