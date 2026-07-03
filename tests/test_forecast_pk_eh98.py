r"""Unit tests for the differentiable EH98 linear power spectrum
(:mod:`hod_mod.forecast.pk_eisenstein_hu`).

Checks the σ8 normalisation (the whole reason this wrapper exists), the growth
scaling, positivity, and the ``as_hmf_pk_func`` shape callable.  EH98 is analytic
so these are fast and do not need CAMB.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hod_mod.forecast.pk_eisenstein_hu import (  # noqa: E402
    EisensteinHu98PkLinear, _sigma2_tophat, _K_INT,
)

_THETA = {"h": 0.6736, "Omega_m": 0.3100, "Omega_b": 0.0493,
          "n_s": 0.9649, "sigma8": 0.8111}


def test_pk_shape_positive():
    pk = EisensteinHu98PkLinear()
    k = jnp.logspace(-3, 1, 64)
    shape = np.asarray(pk.pk_shape(k, _THETA))
    assert shape.shape == (64,)
    assert np.all(shape > 0)


def test_pk_linear_recovers_sigma8_at_z0():
    pk = EisensteinHu98PkLinear()
    p_lin = pk.pk_linear(_K_INT, 0.0, _THETA)
    sigma8_recovered = float(jnp.sqrt(_sigma2_tophat(p_lin, _K_INT, 8.0)))
    assert abs(sigma8_recovered - _THETA["sigma8"]) < 1e-3


def test_pk_linear_growth_suppresses_high_z():
    pk = EisensteinHu98PkLinear()
    k = jnp.logspace(-2, 0, 32)
    p0 = np.asarray(pk.pk_linear(k, 0.0, _THETA))
    p1 = np.asarray(pk.pk_linear(k, 1.0, _THETA))
    assert np.all(p1 < p0)                                   # D(z=1) < D(0)
    # growth is a scale-independent multiplicative factor at fixed z
    ratio = p1 / p0
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)


def test_sigma8_scales_quadratically_with_amplitude():
    pk = EisensteinHu98PkLinear()
    th2 = dict(_THETA, sigma8=2 * _THETA["sigma8"])
    p1 = np.asarray(pk.pk_linear(_K_INT, 0.0, _THETA))
    p2 = np.asarray(pk.pk_linear(_K_INT, 0.0, th2))
    np.testing.assert_allclose(p2 / p1, 4.0, rtol=1e-6)      # P ∝ sigma8²


def test_as_hmf_pk_func_returns_shape():
    pk = EisensteinHu98PkLinear()
    fn = pk.as_hmf_pk_func()
    k = jnp.logspace(-3, 1, 16)
    np.testing.assert_allclose(np.asarray(fn(k, 0.0, _THETA)),
                               np.asarray(pk.pk_shape(k, _THETA)), rtol=1e-12)
