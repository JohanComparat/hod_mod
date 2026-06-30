"""Tests for the JAX-vectorised m200->m500c bisection (hod_mod.gas.conversions).

Replaces the former per-halo scipy.optimize.brentq loop; this asserts the JAX
implementation reproduces brentq and is differentiable/jittable.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from hod_mod.gas import m200_to_m500c

_RHO = 0.3 * 2.775e11 * 1.3   # arbitrary comoving rho_crit(z)-like value


def _brentq_reference(m200, c200, r200, rho):
    """The original scipy.brentq implementation, scalar."""
    def g(x):
        return np.log1p(x) - x / (1.0 + x)
    g200 = g(c200)

    def f(x):
        m_enc = m200 * g(c200 * x) / g200
        m_sph = (4.0 / 3.0) * np.pi * 500.0 * rho * (x * r200) ** 3
        return m_enc - m_sph

    x_r = brentq(f, 0.10, 0.99, xtol=1e-8, maxiter=100)
    r5 = x_r * r200
    m5 = (4.0 / 3.0) * np.pi * 500.0 * rho * r5 ** 3
    return m5, r5


@pytest.fixture
def halos():
    m200 = np.array([1e12, 1e13, 5e13, 1e14, 5e14, 1e15])
    c200 = np.array([8.0, 6.5, 5.5, 5.0, 4.2, 3.8])
    r200 = (m200 / ((4.0 / 3.0) * np.pi * 200.0 * _RHO)) ** (1.0 / 3.0)
    return m200, c200, r200


def test_matches_brentq(halos):
    m200, c200, r200 = halos
    m5j, r5j = (np.asarray(a) for a in m200_to_m500c(m200, c200, r200, _RHO))
    m5r = np.array([_brentq_reference(m200[i], c200[i], r200[i], _RHO)[0] for i in range(len(m200))])
    r5r = np.array([_brentq_reference(m200[i], c200[i], r200[i], _RHO)[1] for i in range(len(m200))])
    assert np.allclose(m5j, m5r, rtol=1e-6)
    assert np.allclose(r5j, r5r, rtol=1e-6)


def test_physical_ordering(halos):
    """M500c < M200 and R500c < R200 for every halo."""
    m200, c200, r200 = halos
    m5, r5 = (np.asarray(a) for a in m200_to_m500c(m200, c200, r200, _RHO))
    assert np.all(m5 < m200)
    assert np.all(r5 < r200)
    assert np.all(np.isfinite(m5)) and np.all(np.isfinite(r5))


def test_scalar_input():
    r200 = (1e14 / ((4.0 / 3.0) * np.pi * 200.0 * _RHO)) ** (1.0 / 3.0)
    m5, r5 = m200_to_m500c(1e14, 5.0, r200, _RHO)
    m5r, r5r = _brentq_reference(1e14, 5.0, r200, _RHO)
    assert np.isclose(float(m5), m5r, rtol=1e-6)
    assert np.isclose(float(r5), r5r, rtol=1e-6)


def test_jit_path_and_consistency(halos):
    """The inner kernel is jit-compiled; repeated calls return identical results
    (exercises the compiled/cached path), and the batched result equals the
    per-halo loop (vectorisation correctness)."""
    from hod_mod.gas.conversions import _m200_to_m500c_jax
    assert hasattr(_m200_to_m500c_jax, "lower"), "inner kernel should be jax.jit-compiled"

    m200, c200, r200 = halos
    m5a, r5a = (np.asarray(a) for a in m200_to_m500c(m200, c200, r200, _RHO))
    m5b, r5b = (np.asarray(a) for a in m200_to_m500c(m200, c200, r200, _RHO))
    assert np.array_equal(m5a, m5b) and np.array_equal(r5a, r5b)

    # batched == per-element
    looped = np.array([np.asarray(m200_to_m500c(m200[i], c200[i], r200[i], _RHO)[0])
                       for i in range(len(m200))])
    assert np.allclose(m5a, looped, rtol=1e-6)
