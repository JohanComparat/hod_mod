"""Direct unit coverage for the gas.conversions helper functions."""
import numpy as np
import pytest

from hod_mod.gas.conversions import (
    _nfw_g,
    _gnfw_f_params,
    _eh_pk_3arg,
    _leggauss_cached,
    _profile_uk_gl,
)

_THETA = {"h": 0.6766, "Omega_m": 0.3111, "Omega_b": 0.0490, "n_s": 0.9665}


def test_nfw_g_known_values():
    # g(x) = ln(1+x) - x/(1+x);  g(0)=0,  g(1)=ln2 - 1/2
    assert _nfw_g(np.array([0.0]))[0] == pytest.approx(0.0)
    assert _nfw_g(np.array([1.0]))[0] == pytest.approx(np.log(2.0) - 0.5)
    g = _nfw_g(np.logspace(-2, 2, 50))
    assert np.all(np.diff(g) > 0)               # monotonically increasing


def test_gnfw_f_params_positive_and_decreasing():
    x = np.logspace(-2, 1.5, 40)
    f = _gnfw_f_params(x, alpha_in=1.0, alpha_tr=1.9, alpha_out=2.7)
    assert np.all(np.isfinite(f)) and np.all(f > 0)
    assert f[-1] < f[0]                          # outer profile falls off


def test_eh_pk_3arg_wraps_eisenstein_hu():
    from hod_mod.core.power_spectrum import eisenstein_hu_pk
    k = np.logspace(-3, 1, 50)
    pk_3 = np.asarray(_eh_pk_3arg(k, 0.5, _THETA))       # z is ignored
    pk_2 = np.asarray(eisenstein_hu_pk(k, _THETA))
    assert np.allclose(pk_3, pk_2)
    assert np.all(np.isfinite(pk_3)) and np.all(pk_3 > 0)


def test_leggauss_cached_is_memoised():
    a = _leggauss_cached(64)
    b = _leggauss_cached(64)
    assert a is b                                 # same cached tuple object
    nodes, weights = a
    assert len(nodes) == 64 and len(weights) == 64
    assert _leggauss_cached(32) is not a          # different n -> different entry


def test_profile_uk_gl_zero_k_volume_limit():
    """For a constant profile f(r)=1, û(k→0) = 4π ∫₀^rmax r² dr = 4π rmax³/3."""
    k = np.array([1e-6, 0.5, 2.0])
    r_max = np.array([2.0, 3.0])                  # NM = 2 haloes
    uk = _profile_uk_gl(k, r_max, lambda r: np.ones_like(r), n_gl=128)
    assert uk.shape == (3, 2)
    expected_volume = 4.0 * np.pi * r_max ** 3 / 3.0
    assert np.allclose(uk[0], expected_volume, rtol=1e-3)     # k ≈ 0 row
    assert np.all(np.abs(uk[2]) < np.abs(uk[0]))             # damped at high k
