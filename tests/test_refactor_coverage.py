"""Targeted coverage for refactored/under-tested code paths (all CAMB-free).

* baryon-fraction PowerLaw/Upturn models + the make_baryon_fraction dispatch,
* the Lange+2025 assembly-bias ``_integrate`` override (the (b-1)/b kernel),
* the eROSITA instrument-response loaders (shipped DR1 npz tables),
* the ``python -m hod_mod`` entry point.
"""
import subprocess
import sys

import numpy as np
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Baryon-fraction models (only Sigmoid was previously exercised)
# ---------------------------------------------------------------------------

_THETA = {"Omega_b": 0.048, "Omega_m": 0.31}
_M = jnp.asarray(np.logspace(12.0, 15.0, 30))


@pytest.mark.parametrize("name", ["sigmoid", "powerlaw", "upturn"])
def test_make_baryon_fraction_dispatch(name):
    from hod_mod.observables.baryon_fraction import make_baryon_fraction
    model = make_baryon_fraction(name)
    fb = np.asarray(model(_M, _THETA, model.default_params()))
    f_b_cosmic = _THETA["Omega_b"] / _THETA["Omega_m"]
    assert fb.shape == _M.shape
    assert np.all(np.isfinite(fb))
    assert np.all(fb >= 0.0) and np.all(fb <= f_b_cosmic + 1e-9)


def test_make_baryon_fraction_invalid_raises():
    from hod_mod.observables.baryon_fraction import make_baryon_fraction
    with pytest.raises(ValueError):
        make_baryon_fraction("not-a-model")


def test_powerlaw_increases_with_mass():
    from hod_mod.observables.baryon_fraction import BaryonFractionPowerLaw
    m = BaryonFractionPowerLaw()
    fb = np.asarray(m(_M, _THETA, m.default_params()))
    assert fb[-1] >= fb[0]


def test_upturn_default_params_keys():
    from hod_mod.observables.baryon_fraction import BaryonFractionUpturn
    p = BaryonFractionUpturn.default_params()
    assert isinstance(p, dict) and len(p) > 0


# ---------------------------------------------------------------------------
# Lange+2025 assembly-bias _integrate (mock HMF -> no CAMB)
# ---------------------------------------------------------------------------

class _MockHMF:
    """Minimal HMF stand-in: positive dndm and a bias spanning b<1 and b>1."""
    def dndm(self, m, z, theta):
        return jnp.asarray(m) ** -2.0

    def bias(self, m, z, theta):
        return 0.4 + (jnp.asarray(m) / 3e13) ** 0.35


@pytest.fixture
def lange_model():
    from hod_mod.connection.hod import Lange25HODModel
    return Lange25HODModel(_MockHMF())


def test_lange25_integrate_finite(lange_model):
    params = lange_model.default_params()
    n_gal, b_eff, m_eff = lange_model._integrate(0.25, {"Omega_m": 0.3}, params)
    for v in (n_gal, b_eff, m_eff):
        assert np.isfinite(float(v)) and float(v) > 0.0


def test_lange25_assembly_bias_shifts_beff(lange_model):
    """A_cen > 0 must raise b_eff via the (b-1)/b kernel (the REPORT_strategy fix)."""
    p0 = dict(lange_model.default_params(), A_cen=0.0, A_sat=0.0)
    p1 = dict(lange_model.default_params(), A_cen=0.5, A_sat=0.0)
    _, b_eff_0, _ = lange_model._integrate(0.25, {"Omega_m": 0.3}, p0)
    _, b_eff_1, _ = lange_model._integrate(0.25, {"Omega_m": 0.3}, p1)
    assert float(b_eff_1) > float(b_eff_0)


# ---------------------------------------------------------------------------
# eROSITA instrument response (shipped DR1 npz tables, no CAMB)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_load_ecf_tables(sample):
    from hod_mod.gas import load_ecf_tables
    interp, lo, hi = load_ecf_tables(sample)          # (callable, float, float)
    assert callable(interp)
    assert np.isfinite(lo) and np.isfinite(hi)
    val = float(interp(1.0))                          # exercise the interpolator at kT=1 keV
    assert np.isfinite(val)


def test_erosita_response_construction():
    from hod_mod.gas import ErositaResponse
    r = ErositaResponse()
    assert np.all(np.isfinite(np.asarray(r.arf)))
    assert np.all(np.isfinite(np.asarray(r.e_obs)))
    assert np.asarray(r.arf).shape == np.asarray(r.e_obs).shape
    assert callable(r.ecf_apec) and callable(r.ecf_apec_table)


def test_erosita_transmission_and_powerlaw_ecf():
    """tbabs transmission (soxs or fallback) ∈ [0,1]; the absorbed power-law ECF
    (AGN leg, no soxs needed) is finite and positive, and decreases with nH."""
    from hod_mod.gas import ErositaResponse
    r = ErositaResponse()
    T = np.asarray(r._transmission(0.03))
    assert np.all(np.isfinite(T)) and T.min() >= 0.0 and T.max() <= 1.0 + 1e-9
    ecf = float(r.ecf_powerlaw(photon_index=1.9, z=0.0))
    assert np.isfinite(ecf) and ecf > 0.0
    ecf_absorbed = float(r.ecf_powerlaw(photon_index=1.9, z=0.0, nH=0.30))
    assert ecf_absorbed < ecf            # more absorption -> fewer counts per flux


@pytest.mark.slow
def test_erosita_apec_ecf():
    pytest.importorskip("soxs")
    from hod_mod.gas import ErositaResponse
    r = ErositaResponse()
    ecf = float(r.ecf_apec(kT=1.0, Z=0.3, z=0.05))
    assert np.isfinite(ecf) and ecf > 0.0


@pytest.mark.slow
def test_apec_cooling_table():
    """Band-integrated APEC cooling Λ(T,Z): build the soxs table and look it up.
    Skips if soxs or the APEC spectral tables are unavailable in the environment."""
    pytest.importorskip("soxs")
    from hod_mod.gas import ApecCoolingTable
    try:
        cool = ApecCoolingTable()                       # ~12 s soxs table build
    except Exception as e:                              # APEC data not downloaded/configured
        pytest.skip(f"APEC table unavailable: {type(e).__name__}")
    T = np.array([0.5, 1.0, 3.0, 8.0])
    Z = np.full_like(T, 0.3)
    Lam = np.asarray(cool(T, Z))
    assert Lam.shape == T.shape
    assert np.all(np.isfinite(Lam)) and np.all(Lam > 0)
    assert Lam[0] != Lam[-1]                            # T-dependence is non-trivial


# ---------------------------------------------------------------------------
# fitting.config bwpd data-vector reader
# ---------------------------------------------------------------------------

_WP_BWPD = "data/lange2025_desi_dr1/BGS2/wp_bgs2_bwpd.csv"


@pytest.mark.skipif(not __import__("os").path.exists(_WP_BWPD),
                    reason="digitized bwpd data not present")
def test_read_wp_bwpd_fig3():
    from hod_mod.fitting.config import read_wp_bwpd_fig3
    rp, wp, err = read_wp_bwpd_fig3(_WP_BWPD)
    for a in (rp, wp, err):
        a = np.asarray(a)
        assert a.ndim == 1 and a.size > 0 and np.all(np.isfinite(a))
    assert np.all(np.asarray(rp) > 0.0)

    # rp_min/rp_max filtering trims the radial range
    rp_cut, _, _ = read_wp_bwpd_fig3(_WP_BWPD, rp_min=1.0, rp_max=20.0)
    rp_cut = np.asarray(rp_cut)
    assert rp_cut.size <= np.asarray(rp).size
    if rp_cut.size:
        assert rp_cut.min() >= 1.0 and rp_cut.max() <= 20.0


# ---------------------------------------------------------------------------
# `python -m hod_mod` entry point
# ---------------------------------------------------------------------------

def test_python_m_hod_mod_help():
    res = subprocess.run([sys.executable, "-m", "hod_mod"],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0
    assert "hod-mod" in res.stdout
