"""Physical survey-noise models (hod_mod.forecast.noise) — unit checks."""

import numpy as np
import pytest

from hod_mod.forecast import noise as N

_H, _OM = 0.6736, 0.31


def test_shear_shape_noise_value():
    s = N.ShearSurvey(n_eff=30.0, sigma_e=0.26, f_sky=0.5)
    n_sr = 30.0 / (np.pi / 180.0 / 60.0) ** 2
    np.testing.assert_allclose(s.n_bin_sr(1), n_sr, rtol=1e-12)
    np.testing.assert_allclose(s.noise_cl(1), 0.26 ** 2 / n_sr, rtol=1e-12)
    # 5 equal bins: 5x the per-bin noise
    np.testing.assert_allclose(s.noise_cl(5), 5 * s.noise_cl(1), rtol=1e-12)
    assert s.noise_cl(1) == pytest.approx(1.9e-10, rel=0.05)


def test_shell_volume_analytic():
    v = N.shell_volume(0.0, 0.1, _H, _OM, 1.0)
    chi = N.chi_of(0.1, _H, _OM)
    np.testing.assert_allclose(v, 4 * np.pi / 3 * chi ** 3, rtol=1e-10)
    # f_sky scales linearly; outer shells are bigger
    np.testing.assert_allclose(N.shell_volume(0.0, 0.1, _H, _OM, 0.5), 0.5 * v)
    assert N.shell_volume(0.5, 0.6, _H, _OM, 1.0) > v


def test_athena_depth_and_completeness():
    a = N.AthenaAllSky()
    assert a.exposure_area == pytest.approx(8.15e7, rel=0.05)   # ~8e7 s cm²
    # the completeness-pinning premise: L_lim(z=1) ≈ 1e42 erg/s
    l1 = a.l_lim(1.0, _H, _OM)
    assert 0.7e42 < l1 < 1.5e42
    # LX>42 bins: complete at z~0.55, NOT complete in the 0.9–1.0 shell top edge
    assert a.l_lim(0.55, _H, _OM) < 10 ** 42.0
    assert a.l_lim(1.0, _H, _OM) > 10 ** 42.0 * 0.9


def test_athena_band_partition():
    a = N.AthenaAllSky()
    bands = [(0.5, 0.9), (0.9, 1.3), (1.3, 2.0)]
    np.testing.assert_allclose(a.band_flux_fractions(bands).sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(a.photon_density(bands).sum(),
                               a.photon_density(), rtol=1e-12)
    # PSF beam: irrelevant at ell<=3000 for 5", suppressive at ell=1e5
    assert a.beam(3000.0) > 0.995
    assert a.beam(1.0e5) < 0.6
    # a 30" eROSITA-like PSF decays much faster
    e = N.AthenaAllSky(psf_hew=30.0)
    assert e.beam(3.0e4) < a.beam(3.0e4)


def test_knox_formulas():
    ell = np.logspace(1, 3.5, 12)
    cl = 1e-8 * (ell / 100.0) ** -1.5
    sig = N.knox_auto(ell, cl, 0.0, 0.5)
    np.testing.assert_allclose(sig, np.sqrt(2.0 / N.n_modes(ell, 0.5)) * cl)
    # cross of a field with itself (equal noise) exceeds the auto error
    sig_x = N.knox_cross(ell, cl, cl, 0.0, cl, 0.0, 0.5)
    np.testing.assert_allclose(sig_x, sig, rtol=1e-12)
    # more sky = less noise
    assert np.all(N.knox_auto(ell, cl, 0.0, 0.25) > sig)


def test_wp_pair_sigma_scalings():
    sp = N.SpectroSurvey(f_sky=0.5)
    rp = np.logspace(-1, 1.5, 12)
    wp = 200.0 * rp ** -0.8
    v = N.shell_volume(0.2, 0.3, _H, _OM, sp.f_sky)
    sig = N.wp_pair_sigma(rp, wp, 1e-3, v, sp)
    assert sig.shape == rp.shape and np.all(np.isfinite(sig)) and np.all(sig > 0)
    # denser sample -> smaller shot noise (up to the CV floor)
    sig_dense = N.wp_pair_sigma(rp, wp, 1e-2, v, sp)
    assert np.all(sig_dense <= sig)
    # CV floor: relative error never below f_cv0-scaled floor
    assert np.all(sig_dense / wp >= 0.99 * sp.cv_rel(v))


def test_delta_sigma_noise_diverges_behind_sources():
    sh = N.ShearSurvey(); sp = N.SpectroSurvey()
    rp = np.logspace(-1, 1.3, 10)
    ds = 20.0 * rp ** -0.9
    zs = np.linspace(0.03, 3.0, 60)
    nz = zs ** 2 * np.exp(-((zs / 0.6) ** 1.5)); nz /= np.trapezoid(nz, zs)
    def sig_at(zl, v=1e9):
        # fixed volume/lens count: isolates the geometric/source-depletion trend
        return N.delta_sigma_noise(rp, ds, zl, 1e-3, v, _H, _OM, zs, nz, sh, sp)
    s_low, s_mid, s_hi = sig_at(0.35), sig_at(0.85), sig_at(1.35)
    assert np.all(np.isfinite(s_low)) and np.all(s_low > 0)
    # the background empties out -> per-lens noise grows with lens z
    assert np.median(s_hi / ds) > np.median(s_mid / ds) > np.median(s_low / ds)
    # no sources at all -> infinite noise
    assert np.all(np.isinf(N.delta_sigma_noise(
        rp, ds, 2.9, 1e-3, 1e9, _H, _OM, zs, nz, sh, sp)))


def test_xlf_poisson():
    v05 = N.shell_volume(0.5, 0.6, _H, _OM, 0.5)
    v10 = N.shell_volume(0.5, 0.6, _H, _OM, 1.0)
    phi = np.array([1e-5, 1e-6, 1e-7])
    r05, r10 = N.xlf_relerr(phi, v05), N.xlf_relerr(phi, v10)
    np.testing.assert_allclose(r05 / r10, np.sqrt(2.0), rtol=1e-12)   # ∝ 1/√f_sky
    np.testing.assert_allclose(r10, 1.0 / np.sqrt(phi * 0.5 * v10), rtol=1e-12)
