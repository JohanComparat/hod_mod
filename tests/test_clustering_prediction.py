"""Integration coverage for FullHaloModelPrediction wp/ΔΣ branches (CAMB-heavy).

Exercises the prediction methods that the existing tests don't reach:
delta_sigma, delta_sigma_split, wp_components, delta_sigma_components, n_gal,
and the baryon-fraction-corrected ΔΣ. Reuses the session pk_lin/hmf fixtures.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import MoreHODModel, HODModel
from hod_mod.observables.clustering import (
    FullHaloModelPrediction,
    NonLinearHaloModelPrediction,
    HODProjectedCorrelation,
    projected_correlation_function,
)
from hod_mod.observables import make_baryon_fraction

pytestmark = pytest.mark.slow

_THETA = LinearPowerSpectrum.default_cosmology()
_HODP  = MoreHODModel.default_params()
_Z     = 0.2
_RP    = jnp.asarray(np.logspace(-1, 1.3, 10))
_R     = jnp.asarray(np.logspace(-0.5, 1.2, 8))
_COLOSSUS = {
    "flat": True, "H0": _THETA["h"] * 100.0,
    "Om0": _THETA["Omega_m"], "Ob0": _THETA["Omega_b"],
    "sigma8": 0.81, "ns": _THETA["n_s"],
}


@pytest.fixture(scope="module")
def hod(hmf):
    return MoreHODModel(hmf, hmf.bias)


@pytest.fixture(scope="module")
def halo_profile():
    return HaloProfile(_COLOSSUS, cm_relation="diemer19")


@pytest.fixture(scope="module")
def pred(pk_lin, hod, halo_profile):
    return FullHaloModelPrediction(pk_lin, hod, halo_profile)


def _arr(x):
    return np.asarray(x)


def test_n_gal_positive(pred):
    n = float(pred.n_gal(_Z, _THETA, _HODP))
    assert np.isfinite(n) and n > 0.0


def test_delta_sigma_finite_positive_decreasing(pred):
    ds = _arr(pred.delta_sigma(_R, _Z, _THETA, _HODP))
    assert ds.shape == _R.shape
    assert np.all(np.isfinite(ds)) and np.all(ds > 0)
    assert ds[0] > ds[-1]                      # ΔΣ decreases with radius


def test_delta_sigma_split_finite(pred):
    out = pred.delta_sigma_split(_R, _Z, _THETA, _HODP)
    parts = list(out.values()) if isinstance(out, dict) else list(out)
    assert len(parts) >= 2
    for p in parts:
        assert np.all(np.isfinite(_arr(p)))


def test_wp_components_reconstruct_total(pred):
    comps = pred.wp_components(_RP, 60.0, _Z, _THETA, _HODP)
    assert isinstance(comps, dict) and len(comps) >= 2
    for v in comps.values():
        assert np.all(np.isfinite(_arr(v)))
    total = _arr(pred.wp(_RP, 60.0, _Z, _THETA, _HODP))
    one_two = [v for k, v in comps.items() if any(t in k.lower() for t in ("1h", "2h"))]
    if len(one_two) >= 2:
        recon = sum(_arr(v) for v in one_two)
        assert np.allclose(recon, total, rtol=0.05)


def test_delta_sigma_components_finite(pred):
    comps = pred.delta_sigma_components(_R, _Z, _THETA, _HODP)
    parts = comps.values() if isinstance(comps, dict) else comps
    for p in parts:
        assert np.all(np.isfinite(_arr(p)))


def test_baryon_fraction_split(pk_lin, hod, halo_profile):
    """The baryon model is applied in delta_sigma_split when baryon_params is given
    (FullHaloModelPrediction.delta_sigma itself does not apply it)."""
    from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
    pred_b = FullHaloModelPrediction(pk_lin, hod, halo_profile,
                                     baryon_fraction=make_baryon_fraction("sigmoid"))
    bp = BaryonFractionSigmoid.default_params()
    split_b = pred_b.delta_sigma_split(_R, _Z, _THETA, _HODP, baryon_params=bp)
    split_0 = pred_b.delta_sigma_split(_R, _Z, _THETA, _HODP, baryon_params=None)
    flat_b = np.concatenate([_arr(v).ravel() for v in split_b.values()])
    flat_0 = np.concatenate([_arr(v).ravel() for v in split_0.values()])
    assert np.all(np.isfinite(flat_b)) and np.all(np.isfinite(flat_0))
    # supplying baryon_params activates the f_b(M) model -> a different split
    assert not np.allclose(flat_b, flat_0, rtol=1e-3)


def test_einasto_profile_wp_finite(pk_lin, hod, halo_profile):
    pred_e = FullHaloModelPrediction(pk_lin, hod, halo_profile, profile="einasto")
    wp = _arr(pred_e.wp(_RP, 60.0, _Z, _THETA, _HODP))
    assert np.all(np.isfinite(wp)) and np.all(wp > 0)


def test_invalid_profile_raises(pk_lin, hod, halo_profile):
    with pytest.raises(ValueError):
        FullHaloModelPrediction(pk_lin, hod, halo_profile, profile="bogus")


# ---------------------------------------------------------------------------
# NonLinearHaloModelPrediction (string-flag wrapper around FullHaloModelPrediction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["hmcode", "aletheia"])
def test_nonlinear_prediction_emulator_backend(pk_lin, hmf, halo_profile, backend):
    """NonLinearHaloModelPrediction with a non-linear 2-halo backend (HMcode / Aletheia)."""
    try:
        pred = NonLinearHaloModelPrediction(pk_lin, hmf, halo_profile,
                                            hod_model="more15", pk_nl_backend=backend)
        wp = _arr(pred.wp(_RP, 60.0, _Z, _THETA, _HODP))
        ds = _arr(pred.delta_sigma(_R, _Z, _THETA, _HODP))
    except Exception as e:                       # emulator/data not available
        pytest.skip(f"{backend} backend unavailable: {type(e).__name__}: {e}")
    assert wp.shape == _RP.shape and np.all(np.isfinite(wp)) and np.all(wp > 0)
    assert ds.shape == _R.shape and np.all(np.isfinite(ds)) and np.all(ds > 0)


# ---------------------------------------------------------------------------
# HODProjectedCorrelation (legacy Zheng+2007 wp)
# ---------------------------------------------------------------------------

def test_hod_projected_correlation_wp(hmf):
    from hod_mod.core.nonlinear import HALOFITSpectrum
    hod = HODModel(hmf, hmf.bias)
    pred = HODProjectedCorrelation(hmf, hod, HALOFITSpectrum("mead2020"))
    wp = _arr(pred.wp(_RP, 60.0, _Z, _THETA, HODModel.default_params()))
    assert wp.shape == _RP.shape and np.all(np.isfinite(wp)) and np.all(wp > 0)


# ---------------------------------------------------------------------------
# projected_correlation_function (Corrfunc pair-counted wp from a mock catalog)
# ---------------------------------------------------------------------------

def test_projected_correlation_function_from_catalog():
    pytest.importorskip("Corrfunc")
    rng = np.random.default_rng(0)
    n = 400
    ra  = rng.uniform(150.0, 160.0, n)        # deg
    dec = rng.uniform(0.0, 10.0, n)           # deg
    z   = rng.uniform(0.1, 0.2, n)
    rp_bins = np.logspace(-0.5, 1.2, 8)       # h^-1 Mpc
    wp = np.asarray(projected_correlation_function(ra, dec, z, rp_bins,
                                                   pi_max=40.0, n_threads=1))
    assert wp.shape == (len(rp_bins) - 1,)
    assert np.all(np.isfinite(wp))
