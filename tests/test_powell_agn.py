"""Tests for the analytic Powell 2022 AGN-halo model (hod_mod.agn.powell)."""
import numpy as np
import pytest

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.agn.powell import PowellAGNModel, erdf_dloglambda, _LBOL_COEF, _KBOL_HARD


@pytest.fixture(scope="module")
def model():
    pk = LinearPowerSpectrum()
    theta = LinearPowerSpectrum.default_cosmology()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    return PowellAGNModel(theta, hmf, z_mean=0.135, log10lx_min=42.0)


def test_erdf_broken_powerlaw():
    loglam = np.linspace(-3, 1.5, 200)
    ls = np.log10(0.13)
    e = erdf_dloglambda(loglam, ls, 0.30, 3.70)
    assert np.all(np.isfinite(e)) and np.all(e > 0)
    # Ananna ERDF (delta1=0.3>0) rises toward low lambda and steepens above
    # lambda* (delta2=3.7): so dN/dloglambda declines and the break is at lambda*.
    assert e[np.argmin(np.abs(loglam + 1))] > e[np.argmin(np.abs(loglam - 1))]
    # high-lambda log-slope ~ -delta2 (steep)
    hi = loglam > ls + 0.5
    slope = np.polyfit(loglam[hi], np.log10(e[hi]), 1)[0]
    assert -4.2 < slope < -3.0


def test_xlf_finite_positive_declining(model):
    grid, phi = model.xlf(band="hard")
    assert np.all(np.isfinite(phi)) and np.all(phi >= 0)
    # XLF declines toward high L_X
    i42, i44 = np.argmin(np.abs(grid - 42)), np.argmin(np.abs(grid - 44))
    assert phi[i44] < phi[i42]


def test_convolution_matches_montecarlo(model):
    """Analytic P(L_X|M_halo) reproduces a direct Monte-Carlo draw."""
    M = model
    im = np.argmin(np.abs(M.log10m - 13.0))
    lms = float(M._log10ms[im])
    rng = np.random.default_rng(1)
    N = 300000
    ms = lms + rng.normal(0, M.sigma_ms, N)
    mbh = M.mu_bh + M.al_bh * (ms - 11.0) + rng.normal(0, M.sig_bh, N)
    e = erdf_dloglambda(M.loglam, M.log10_lstar, M.delta1, M.delta2); e /= e.sum()
    lam = rng.choice(M.loglam, size=N, p=e) + rng.uniform(-M.dlam / 2, M.dlam / 2, N)
    loglx = np.log10(_LBOL_COEF / _KBOL_HARD) + mbh + lam
    p = M._p_lx_given_m()[im]            # normalised shape pdf
    hist, _ = np.histogram(loglx, bins=np.append(M.loglx - M.dlx / 2, M.loglx[-1] + M.dlx / 2),
                           density=True)
    sel = (M.loglx > 41) & (M.loglx < 45) & (hist > 1e-2)
    assert np.median(np.abs(p[sel] - hist[sel]) / hist[sel]) < 0.06


def test_ferdf_scales_xlf_linearly(model):
    model.set_params(log10_ferdf=-2.0)
    _, phi1 = model.xlf(band="hard")
    model.set_params(log10_ferdf=-1.0)
    _, phi2 = model.xlf(band="hard")
    model.set_params(log10_ferdf=-2.0)      # restore
    assert np.allclose(phi2 / np.maximum(phi1, 1e-300), 10.0, rtol=1e-6)


def test_occupation_monotonic_and_bounded(model):
    nc, ns = model.nc_ns_agn(np.array([11.0, 12.0, 13.0, 14.0]))
    assert np.all(np.isfinite(nc)) and np.all((nc >= 0) & (nc <= 1))
    assert nc[3] > nc[0]                     # more massive halos host more AGN


def test_bias_and_hostmass_sane(model):
    b = model.agn_bias()
    mh = model.median_host_logmhalo()
    assert 0.3 < b < 5.0
    assert 10.5 < mh < 14.5


def test_emissivity_shape(model):
    k = np.logspace(-2, 1, 20); m = np.logspace(11, 15, 30)
    em = model.agn_emissivity_uk(k, m, 0.135, {})
    assert em.shape == (20, 30) and np.all(np.isfinite(em))
    # point source: flat in k
    assert np.allclose(em[0], em[-1])
