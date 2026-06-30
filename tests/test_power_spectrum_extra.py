"""Coverage for the analytic (CAMB-free) power-spectrum helpers."""
import numpy as np
import jax.numpy as jnp
import pytest

from hod_mod.core.power_spectrum import (
    eisenstein_hu_pk,
    eisenstein_hu_pk_nowiggle,
)

_THETA = {"h": 0.6766, "Omega_m": 0.3111, "Omega_b": 0.0490, "n_s": 0.9665}
_K = jnp.asarray(np.logspace(-3, 1, 200))


def test_nowiggle_finite_positive():
    pk = np.asarray(eisenstein_hu_pk_nowiggle(_K, _THETA))
    assert pk.shape == _K.shape
    assert np.all(np.isfinite(pk)) and np.all(pk > 0)


def test_nowiggle_normalised_at_pivot():
    """EH no-wiggle P(k) is normalised to 1 at k = 0.05 h/Mpc by construction."""
    pivot = jnp.asarray([0.05])
    assert float(eisenstein_hu_pk_nowiggle(pivot, _THETA)[0]) == pytest.approx(1.0, rel=1e-3)


def test_nowiggle_has_no_bao_wiggles():
    """The no-wiggle spectrum is smooth where the full EH spectrum oscillates
    (BAO scale ~0.05-0.3 h/Mpc): its second log-derivative has smaller variance."""
    k = jnp.asarray(np.logspace(-1.5, -0.3, 120))
    lp_full = np.log(np.asarray(eisenstein_hu_pk(k, _THETA)))
    lp_nw   = np.log(np.asarray(eisenstein_hu_pk_nowiggle(k, _THETA)))
    lk = np.log(np.asarray(k))
    curv_full = np.diff(lp_full, 2) / np.diff(lk)[:-1] ** 2
    curv_nw   = np.diff(lp_nw,   2) / np.diff(lk)[:-1] ** 2
    assert np.var(curv_nw) < np.var(curv_full)


def test_nowiggle_large_scale_slope_ns():
    """On very large scales P(k) ∝ k^{n_s}."""
    k = jnp.asarray(np.logspace(-4, -3, 20))
    lp = np.log(np.asarray(eisenstein_hu_pk_nowiggle(k, _THETA)))
    slope = np.polyfit(np.log(np.asarray(k)), lp, 1)[0]
    assert slope == pytest.approx(_THETA["n_s"], abs=0.05)


@pytest.mark.slow
def test_csst_linear_power_spectrum():
    """CSST CEmulator linear P(k) backend. Skips if CEmulator isn't importable or
    needs the simps->simpson patch in this environment."""
    pytest.importorskip("CEmulator")
    from hod_mod.core.power_spectrum import CsstLinearPowerSpectrum, LinearPowerSpectrum
    try:
        csst = CsstLinearPowerSpectrum()
        theta = LinearPowerSpectrum.default_cosmology()
        pk = np.asarray(csst.pk_linear(np.logspace(-2.5, 0.0, 30), 0.0, theta))
    except Exception as e:                       # missing patch / emulator data
        pytest.skip(f"CEmulator unavailable: {type(e).__name__}: {e}")
    assert pk.shape == (30,)
    assert np.all(np.isfinite(pk)) and np.all(pk > 0)
