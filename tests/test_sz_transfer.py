"""The Σ_y transfer kernel must reproduce projected_gy and rescale analytically.

Guards the MCMC-fast path of the coupled X-ray × SZ joint fit: if
:func:`hod_mod.fitting.sz_transfer.build_sz_transfer` ever drifts from
``HaloModelCrossSpectra.projected_gy``, the SZ leg of the likelihood is wrong.
"""

import numpy as np
import pytest

# ~3e-6 is the intrinsic project-then-sum vs sum-then-project difference from the
# log-space interpolation inside _pk_to_wp (identical in float32/float64 — not a
# precision effect).  It is ~5 orders of magnitude below the Σ_y errors.
_RTOL = 1e-5

_RP = np.logspace(-0.4, 1.2, 8)
_Z = 0.135
_BEAM = 1.6          # ACT DR6 NILC


@pytest.fixture(scope="module")
def cross_and_params():
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.core.halo_profiles import HaloProfile
    from hod_mod.connection.hod import MoreHODModel
    from hod_mod.observables.clustering import FullHaloModelPrediction
    from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
    from hod_mod.gas import PressureProfileDPM

    theta = LinearPowerSpectrum.default_cosmology()
    colo = dict(flat=True, H0=theta["h"] * 100.0, Om0=theta["Omega_m"],
                Ob0=theta["Omega_b"], sigma8=0.811, ns=theta["n_s"])
    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = MoreHODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    pp = PressureProfileDPM(model=2, r_max_over_r200=3.0, n_gl=60)
    cross = HaloModelCrossSpectra(fhmp, pressure_profile=pp)
    return cross, theta, MoreHODModel.default_params(), pp


@pytest.mark.slow
def test_transfer_reproduces_projected_gy(cross_and_params):
    from hod_mod.fitting.sz_transfer import build_sz_transfer, predict_sigma_y
    cross, theta, hod_params, pp = cross_and_params
    direct = np.asarray(cross.projected_gy(_RP, _Z, theta, hod_params,
                                           beam_fwhm_arcmin=_BEAM))
    G, m200 = build_sz_transfer(cross, _Z, theta, hod_params, _RP,
                                beam_fwhm_arcmin=_BEAM)
    emu = predict_sigma_y(G, m200, pp._P_03, pp._beta, pp._P_03, pp._beta)
    assert np.all(np.isfinite(emu))
    np.testing.assert_allclose(emu, direct, rtol=_RTOL)


@pytest.mark.slow
def test_p03_betap_rescaling_is_analytic(cross_and_params):
    """Rescaling the kernel must match a fresh projected_gy at new (P_0.3, β_P).

    This is what lets the joint fit vary the native DPM pressure params at MCMC
    speed — and it is the coupling to the X-ray leg (T = P/n_e).
    """
    from hod_mod.fitting.sz_transfer import build_sz_transfer, predict_sigma_y
    from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
    from hod_mod.gas import PressureProfileDPM
    cross, theta, hod_params, pp = cross_and_params
    G, m200 = build_sz_transfer(cross, _Z, theta, hod_params, _RP,
                                beam_fwhm_arcmin=_BEAM)

    p03_new, beta_new = 1.4 * pp._P_03, pp._beta + 0.15
    pp2 = PressureProfileDPM(model=2, r_max_over_r200=3.0, n_gl=60)
    pp2._P_03 = p03_new
    pp2._beta = beta_new
    from hod_mod.gas import _gnfw_f_params
    pp2._P0 = pp2._P_03 / float(_gnfw_f_params(0.3 * pp2._C_DPM, pp2._alpha_in,
                                               pp2._alpha_tr, pp2._alpha_out_12))
    cross2 = HaloModelCrossSpectra(cross._fhmp, pressure_profile=pp2)
    direct2 = np.asarray(cross2.projected_gy(_RP, _Z, theta, hod_params,
                                             beam_fwhm_arcmin=_BEAM))
    emu2 = predict_sigma_y(G, m200, p03_new, beta_new, pp._P_03, pp._beta)
    np.testing.assert_allclose(emu2, direct2, rtol=_RTOL)


def test_amplitude_is_identity_at_reference():
    """A(M) == 1 when the params equal the kernel's reference (cheap, no model)."""
    from hod_mod.fitting.sz_transfer import sz_amplitude
    m = np.array([1e12, 1e13, 1e14, 1e15])
    a = sz_amplitude(m, p03=1.627e-6, beta_p=0.8, p03_ref=1.627e-6, beta_p_ref=0.8)
    np.testing.assert_allclose(a, np.ones_like(m), rtol=0, atol=0)


def test_amplitude_scales_linearly_in_p03():
    """Σ_y ∝ P_0.3 exactly — the tSZ amplitude identity."""
    from hod_mod.fitting.sz_transfer import sz_amplitude
    m = np.array([1e13, 1e14])
    a1 = sz_amplitude(m, 1.627e-6, 0.8, 1.627e-6, 0.8)
    a2 = sz_amplitude(m, 2.0 * 1.627e-6, 0.8, 1.627e-6, 0.8)
    np.testing.assert_allclose(a2, 2.0 * a1, rtol=1e-12)


# --- the SZ leg of fit_xray_joint_bands (data loader + chi2 plumbing) ---------

def test_das2023_loader_units_and_shape():
    """Every mapped das_2023 profile loads with sane units and ordering."""
    from hod_mod.scripts.fitting.fit_xray_joint_bands import (
        _SZ_DATA_FILES, _load_sz_data)
    for sample in _SZ_DATA_FILES:
        x, x_lo, x_hi, y, err = _load_sz_data(sample)
        assert x.shape == x_lo.shape == x_hi.shape == y.shape == err.shape
        assert x.size >= 5
        assert np.all(np.diff(x) > 0), "r/R200 must be increasing"
        assert np.all(x > 0) and np.all(x < 20), "r is in R200 units"
        # annulus edges bracket the mid radius (beam-width annuli, Das+23 Eq. 4)
        assert np.all(x_lo < x) and np.all(x < x_hi)
        # y1e8 columns were rescaled by 1e-8 -> stacked y is ~1e-8..1e-6
        assert np.all(y > 1e-10) and np.all(y < 1e-5)
        assert np.all(err > 0)
        # the digitized (up, low) envelope must bracket mid: err < y at these S/N
        assert np.all(err < y)


def test_annulus_average_matrix_is_exact_on_powers():
    """W must reproduce the analytic area-weighted mean of r^0 and r^1."""
    from hod_mod.scripts.fitting.fit_xray_joint_bands import (
        _annulus_average_matrix)
    lo = np.array([0.1, 1.0]); hi = np.array([0.9, 2.0])
    W, nodes = _annulus_average_matrix(lo, hi, n_nodes=64)
    # constant: mean is exactly 1
    np.testing.assert_allclose(W @ np.ones_like(nodes), 1.0, rtol=1e-12)
    # f(r) = r: analytic 2D-area-weighted mean = (2/3)(hi^3-lo^3)/(hi^2-lo^2)
    expect = 2.0 / 3.0 * (hi**3 - lo**3) / (hi**2 - lo**2)
    np.testing.assert_allclose(W @ nodes, expect, rtol=1e-4)


def test_unmapped_sample_returns_none():
    from hod_mod.scripts.fitting.fit_xray_joint_bands import _load_sz_data
    assert _load_sz_data("S1") is None


def test_sigma_y_model_uses_native_params():
    """_sigma_y_model must consume p[2] (log10_p03) / p[3] (beta_P) exactly."""
    from hod_mod.scripts.fitting.fit_xray_joint_bands import _sigma_y_model
    rng = np.random.default_rng(0)
    m200 = np.geomspace(1e12, 1e15, 30)
    sz = dict(G=rng.random((6, 30)), m200=m200, p03_ref=1.627e-6, beta_ref=0.8)
    p = np.zeros(9); p[2] = np.log10(1.627e-6); p[3] = 0.8
    base = _sigma_y_model(p, sz)
    # doubling P_0.3 doubles Sigma_y (exact linearity)
    p2 = p.copy(); p2[2] = np.log10(2 * 1.627e-6)
    np.testing.assert_allclose(_sigma_y_model(p2, sz), 2.0 * base, rtol=1e-12)
    # beta_P tilts the mass weighting -> changes the profile
    p3 = p.copy(); p3[3] = 1.0
    assert not np.allclose(_sigma_y_model(p3, sz), base)
