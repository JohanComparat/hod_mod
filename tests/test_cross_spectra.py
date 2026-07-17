"""Tests for HaloModelCrossSpectra (galaxy × tSZ and galaxy × soft X-ray)."""

import numpy as np
import pytest

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.gas import PressureProfileA10, GasDensityDPM
from hod_mod.connection.hod import MoreHODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra

_THETA = LinearPowerSpectrum.default_cosmology()
_Z     = 0.3

_COLOSSUS = {
    "flat": True,
    "H0": _THETA["h"] * 100.0,
    "Om0": _THETA["Omega_m"],
    "Ob0": _THETA["Omega_b"],
    "sigma8": 0.811,
    "ns": _THETA["n_s"],
}


# ---------------------------------------------------------------------------
# Shared fixture: FullHaloModelPrediction + HaloModelCrossSpectra
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fhmp():
    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hp     = HaloProfile(_COLOSSUS, cm_relation="diemer19")
    hod    = MoreHODModel(hmf, hmf.bias)
    return FullHaloModelPrediction(pk_lin, hod, hp)


@pytest.fixture(scope="module")
def cross_gy(fhmp):
    pp = PressureProfileA10(r_max_over_r500c=4.0, n_gl=80)
    return HaloModelCrossSpectra(fhmp, pressure_profile=pp)


@pytest.fixture(scope="module")
def cross_gX(fhmp):
    dp = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80)
    return HaloModelCrossSpectra(fhmp, density_profile=dp)


@pytest.fixture(scope="module")
def hod_params():
    return MoreHODModel.default_params()


# ---------------------------------------------------------------------------
# TestGalaxyTSZSpectrum
# ---------------------------------------------------------------------------

class TestGalaxyTSZSpectrum:

    def test_pk_tables_gy_keys(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        for key in ["log_k", "log_pgy", "log_pgy_1h", "log_pgy_2h", "log_pmy", "n_gal", "b_eff"]:
            assert key in tables, f"Missing key: {key}"

    def test_pgy_finite(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.isfinite(tables["log_pgy"]))

    def test_pgy_1h_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.exp(tables["log_pgy_1h"]) > 0)

    def test_pgy_2h_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.exp(tables["log_pgy_2h"]) > 0)

    def test_pgy_total_geq_components(self, cross_gy, hod_params):
        """P_gy ≥ max(P_gy_1h, P_gy_2h) at every k."""
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        pgy    = np.exp(np.array(tables["log_pgy"]))
        pgy_1h = np.exp(np.array(tables["log_pgy_1h"]))
        pgy_2h = np.exp(np.array(tables["log_pgy_2h"]))
        assert np.all(pgy >= pgy_1h - 1e-10 * pgy)
        assert np.all(pgy >= pgy_2h - 1e-10 * pgy)

    def test_pgy_2h_dominates_at_low_k(self, cross_gy, hod_params):
        """At k=0.01 h/Mpc the 2-halo term should dominate."""
        tables  = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        k       = np.exp(np.array(tables["log_k"]))
        idx_lo  = np.argmin(np.abs(k - 0.01))
        pgy_1h  = float(np.exp(tables["log_pgy_1h"][idx_lo]))
        pgy_2h  = float(np.exp(tables["log_pgy_2h"][idx_lo]))
        assert pgy_2h > pgy_1h

    def test_pmy_finite(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert np.all(np.isfinite(tables["log_pmy"]))

    def test_ngal_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert tables["n_gal"] > 0

    def test_beff_positive(self, cross_gy, hod_params):
        tables = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert 0.5 < tables["b_eff"] < 10.0

    def test_gas_cache_populated(self, cross_gy, hod_params):
        """After one call, gas profile FT should be cached."""
        _ = cross_gy._pk_tables_gy(_Z, _THETA, hod_params)
        assert len(cross_gy._gas_cache) > 0

    def test_no_pressure_profile_raises(self, fhmp, hod_params):
        cross_no_pp = HaloModelCrossSpectra(fhmp)  # no pressure_profile
        with pytest.raises(RuntimeError, match="No pressure_profile"):
            cross_no_pp._pk_tables_gy(_Z, _THETA, hod_params)


# ---------------------------------------------------------------------------
# TestGalaxyXRaySpectrum
# ---------------------------------------------------------------------------

class TestGalaxyXRaySpectrum:

    def test_pk_tables_gX_keys(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        for key in ["log_k", "log_pgX", "log_pgX_1h", "log_pgX_2h", "n_gal", "b_eff"]:
            assert key in tables

    def test_pgX_finite(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        assert np.all(np.isfinite(tables["log_pgX"]))

    def test_pgX_1h_2h_positive(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        assert np.all(np.exp(tables["log_pgX_1h"]) > 0)
        assert np.all(np.exp(tables["log_pgX_2h"]) > 0)

    def test_pgX_total_geq_components(self, cross_gX, hod_params):
        tables = cross_gX._pk_tables_gX(_Z, _THETA, hod_params)
        pgX    = np.exp(np.array(tables["log_pgX"]))
        pgX_1h = np.exp(np.array(tables["log_pgX_1h"]))
        pgX_2h = np.exp(np.array(tables["log_pgX_2h"]))
        assert np.all(pgX >= pgX_1h - 1e-10 * pgX)
        assert np.all(pgX >= pgX_2h - 1e-10 * pgX)

    def test_no_density_profile_raises(self, fhmp, hod_params):
        cross_no_dp = HaloModelCrossSpectra(fhmp)
        with pytest.raises(RuntimeError, match="No density_profile"):
            cross_no_dp._pk_tables_gX(_Z, _THETA, hod_params)


# ---------------------------------------------------------------------------
# TestProjectedSignal
# ---------------------------------------------------------------------------

class TestProjectedSignalGY:

    def test_projected_gy_shape(self, cross_gy, hod_params):
        rp = np.logspace(-1, 1.5, 10)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert sigma_y.shape == (10,)

    def test_projected_gy_positive(self, cross_gy, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert np.all(sigma_y > 0)

    def test_projected_gy_finite(self, cross_gy, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert np.all(np.isfinite(sigma_y))

    def test_projected_gy_decreasing(self, cross_gy, hod_params):
        """Σ_y(r_p) should decrease monotonically with r_p at r_p > 0.3 Mpc/h."""
        rp = np.logspace(-0.5, 1.3, 12)
        sigma_y = cross_gy.projected_gy(rp, _Z, _THETA, hod_params)
        assert np.all(np.diff(sigma_y) < 0), "projected tSZ signal should be monotonically decreasing"


class TestProjectedSignalGX:

    def test_projected_gX_shape(self, cross_gX, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        wgx = cross_gX.projected_gX(rp, _Z, _THETA, hod_params)
        assert wgx.shape == (8,)

    def test_projected_gX_positive(self, cross_gX, hod_params):
        rp = np.logspace(-1, 1.5, 8)
        wgx = cross_gX.projected_gX(rp, _Z, _THETA, hod_params)
        assert np.all(wgx > 0)

    def test_projected_gX_decreasing(self, cross_gX, hod_params):
        rp = np.logspace(-0.5, 1.3, 10)
        wgx = cross_gX.projected_gX(rp, _Z, _THETA, hod_params)
        assert np.all(np.diff(wgx) < 0)


# ---------------------------------------------------------------------------
# TestAngularPowerSpectrum
# ---------------------------------------------------------------------------

class TestAngularPowerSpectrumGY:

    def test_cl_gy_shape(self, cross_gy, hod_params):
        ell  = np.logspace(1, 4, 10)
        z_arr = np.linspace(0.2, 0.5, 8)
        nz_g  = np.exp(-0.5 * ((z_arr - 0.3) / 0.05)**2)
        cl_gy = cross_gy.angular_cl_gy(ell, z_arr, nz_g, _THETA, hod_params)
        assert cl_gy.shape == (10,)

    def test_cl_gy_positive(self, cross_gy, hod_params):
        ell  = np.logspace(1, 4, 8)
        z_arr = np.linspace(0.2, 0.5, 8)
        nz_g  = np.exp(-0.5 * ((z_arr - 0.3) / 0.05)**2)
        cl_gy = cross_gy.angular_cl_gy(ell, z_arr, nz_g, _THETA, hod_params)
        assert np.all(cl_gy > 0)

    def test_cl_gy_finite(self, cross_gy, hod_params):
        ell  = np.logspace(2, 3.5, 6)
        z_arr = np.linspace(0.25, 0.45, 6)
        nz_g  = np.exp(-0.5 * ((z_arr - 0.35) / 0.04)**2)
        cl_gy = cross_gy.angular_cl_gy(ell, z_arr, nz_g, _THETA, hod_params)
        assert np.all(np.isfinite(cl_gy))


@pytest.mark.slow
class TestAngularPowerSpectrumGX:
    """Galaxy × soft-X-ray angular cross-spectrum (Limber). Serial by default;
    the threaded path (n_workers>1) must agree after the warm-up fix, and the
    result must be finite (regression for the float32 _safe_log floor bug)."""

    _ELL = np.logspace(2, 3.5, 5)
    _Z   = np.linspace(0.25, 0.45, 4)
    _NZ  = np.exp(-0.5 * ((np.linspace(0.25, 0.45, 4) - 0.35) / 0.05) ** 2)

    def test_cl_gX_finite_positive(self, cross_gX, hod_params):
        cl = cross_gX.angular_cl_gX(self._ELL, self._Z, self._NZ, _THETA, hod_params)
        assert cl.shape == (5,)
        assert np.all(np.isfinite(cl)) and np.all(cl > 0)

    def test_cl_gX_serial_equals_threaded(self, cross_gX, hod_params):
        serial   = np.asarray(cross_gX.angular_cl_gX(self._ELL, self._Z, self._NZ,
                                                     _THETA, hod_params, n_workers=1))
        threaded = np.asarray(cross_gX.angular_cl_gX(self._ELL, self._Z, self._NZ,
                                                     _THETA, hod_params, n_workers=2))
        assert np.all(np.isfinite(threaded))
        assert np.allclose(serial, threaded, rtol=1e-6)


class TestAngularPowerSpectrumGXFullApec:
    """FULL-APEC C_ℓ^{gX}: density + pressure + metallicity + cooling.

    Regression for the 2026-07 campaign NaN: activating the temperature-
    dependent Λ(T,Z) emissivity made the whole C_ℓ^{gX} non-finite (float32
    underflow of Λ~1e-24), which silently pinned the Family-C gas presets to
    the density-only path.  Guarded since by the float32-safe ``safe_log``
    floor and the Λ_ref renormalisation in ``_pk_tables_gX`` — this test is
    what would have caught it end-to-end.
    """

    _ELL = np.logspace(2, 3.5, 5)
    _Z   = np.linspace(0.25, 0.45, 4)
    _NZ  = np.exp(-0.5 * ((np.linspace(0.25, 0.45, 4) - 0.35) / 0.05) ** 2)

    @pytest.fixture(scope="class")
    def cross_gX_full(self, fhmp):
        from hod_mod.gas import PressureProfileDPM
        from hod_mod.gas.metallicity import MetallicityProfileDPM
        from hod_mod.gas.cooling import ApecCoolingTable
        soxs = pytest.importorskip("soxs")  # noqa: F841 — APEC table needs it
        return HaloModelCrossSpectra(
            fhmp,
            density_profile=GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80),
            pressure_profile=PressureProfileDPM(model=2, r_max_over_r200=3.0, n_gl=60),
            metallicity_profile=MetallicityProfileDPM(),
            cooling_function=ApecCoolingTable(emin=0.5, emax=2.0, n_T=20, n_Z=5),
        )

    @pytest.mark.slow
    def test_cl_gX_full_apec_finite_positive(self, cross_gX_full, hod_params):
        cl = np.asarray(cross_gX_full.angular_cl_gX(
            self._ELL, self._Z, self._NZ, _THETA, hod_params, n_workers=1))
        assert cl.shape == (5,)
        assert np.all(np.isfinite(cl)) and np.all(cl > 0)

    def test_cl_gX_psf_suppresses_small_scales(self, cross_gX, hod_params):
        raw = np.asarray(cross_gX.angular_cl_gX(self._ELL, self._Z, self._NZ, _THETA, hod_params))
        psf = np.asarray(cross_gX.angular_cl_gX(self._ELL, self._Z, self._NZ, _THETA, hod_params,
                                                psf_fwhm_arcsec=30.0))
        assert np.all(psf <= raw + 1e-30) and psf[-1] < raw[-1]


@pytest.mark.slow
class TestAngularPowerSpectrumXX:
    """X-ray × X-ray angular auto-spectrum (no galaxy occupation)."""

    def test_cl_XX_finite_positive(self, cross_gX):
        ell  = np.logspace(1, 3.5, 6)
        z    = np.linspace(0.2, 0.5, 5)
        nz_X = np.exp(-0.5 * ((z - 0.3) / 0.06) ** 2)
        cl_XX = cross_gX.angular_cl_XX(ell, z, nz_X, _THETA)
        assert cl_XX.shape == (6,)
        assert np.all(np.isfinite(cl_XX)) and np.all(cl_XX > 0)


# ---------------------------------------------------------------------------
# Float64 numpy oracle for _get_hod_weights (Wave-1 jnp-port regression)
# ---------------------------------------------------------------------------

class TestHodWeightsNumpyOracle:
    def test_ngal_beff_match_float64_reference(self, cross_gy, hod_params):
        import jax

        sc = cross_gy._get_static_cache(_Z, _THETA, hod_params)
        nc_np, ns_np, n_gal, b_eff = cross_gy._get_hod_weights(
            _Z, _THETA, hod_params, sc)

        with jax.disable_jit():
            nc_ref, ns_ref = cross_gy._fhmp._hod.nc_ns(
                cross_gy._fhmp._hod._log10m_grid, hod_params)
        nc_ref = np.asarray(nc_ref, dtype=float)
        ns_ref = np.asarray(ns_ref, dtype=float)
        nt = nc_ref + ns_ref
        n_ref = float(np.trapezoid(sc["dndm_np"] * nt, sc["m_np"]))
        b_ref = float(np.trapezoid(sc["dndm_np"] * nt * sc["bias_np"],
                                   sc["m_np"]) / n_ref)

        # float32-safe tolerance so the pin survives the jnp.trapezoid port;
        # an axis/weighting bug would miss by orders of magnitude
        assert n_gal == pytest.approx(n_ref, rel=2e-5)
        assert b_eff == pytest.approx(b_ref, rel=2e-5)
        np.testing.assert_allclose(nc_np, nc_ref, rtol=2e-5)
        np.testing.assert_allclose(ns_np, ns_ref, rtol=2e-5)
        # physical sanity for the More15 defaults at z=0.3
        assert 1e-6 < n_ref < 1e-1
        assert 1.0 < b_ref < 3.0


# ---------------------------------------------------------------------------
# TestBeamAndCAP
# ---------------------------------------------------------------------------

class TestProjectedGyBeam:
    """The beam on Sigma_y(r_p), so the model matches a beam-convolved measurement."""

    _RP = np.logspace(-1, 1.0, 10)

    def test_beam_none_is_unchanged(self, cross_gy, hod_params):
        """beam=None must reproduce the pre-beam result exactly — the regression guard."""
        a = np.asarray(cross_gy.projected_gy(self._RP, _Z, _THETA, hod_params))
        b = np.asarray(cross_gy.projected_gy(self._RP, _Z, _THETA, hod_params,
                                             beam_fwhm_arcmin=None))
        assert np.array_equal(a, b)

    def test_beam_suppresses_small_scales_only(self, cross_gy, hod_params):
        """A beam smooths the inner profile and leaves the outer profile alone."""
        raw = np.asarray(cross_gy.projected_gy(self._RP, _Z, _THETA, hod_params))
        bea = np.asarray(cross_gy.projected_gy(self._RP, _Z, _THETA, hod_params,
                                               beam_fwhm_arcmin=1.6))
        assert bea[0] < raw[0], "beam must suppress the innermost point"
        assert np.isclose(bea[-1], raw[-1], rtol=2e-2), "beam must not touch large scales"

    def test_bigger_beam_suppresses_more(self, cross_gy, hod_params):
        rp = np.array([0.15, 0.3])
        s = [np.asarray(cross_gy.projected_gy(rp, _Z, _THETA, hod_params,
                                              beam_fwhm_arcmin=f))[0]
             for f in (0.5, 1.6, 5.0)]
        assert s[0] > s[1] > s[2]

    def test_beam_matches_direct_transverse_convolution(self):
        """The beam is applied as an isotropic B(|k|chi) on the 3D power before an Abel
        projection, yet it must equal a purely *transverse* convolution.

        Projection-slice makes this exact: the line-of-sight integral samples the 3D power at
        k_z=0, where |k| = k_perp, and convolving along pi then integrating over pi is a no-op.
        This pins that claim against a direct 2D Hankel transform of P*B -- it is the
        justification for not building a separate transform, so if it ever fails, the beam is
        silently smearing along the line of sight.

        Tested where the beam actually acts.  At large r_p both suppression factors tend to 1
        and their ratio just amplifies _pk_to_wp's own numerical noise (Ogata quadrature,
        float32, finite pi_max), which is ~2% there -- that would test the transform, not the
        beam, so those bins get a separate, looser no-op check.
        """
        from scipy.special import j0

        from hod_mod.core.numerics import safe_log
        from hod_mod.observables.cross_spectra import _pk_to_wp, psf_window_ell

        k = np.logspace(-4, 3, 4000)
        pk = 1e2 * k ** (-1.5) / (1.0 + (k / 5.0) ** 4)   # stand-in P_gy(k)
        chi = 900.0                                        # Mpc/h
        rp = np.logspace(-1, 0.7, 7)

        for fwhm_arcmin in (1.6, 5.0):
            b_k = np.asarray(psf_window_ell(k * chi, fwhm_arcmin * 60.0))
            # mirror the production path exactly: log(P) + safe_log(B).  B underflows to 0 at
            # high k, so a raw np.log would put -inf into the interpolation table.
            got = np.asarray(_pk_to_wp(rp, np.log(k), np.log(pk) + np.asarray(safe_log(b_k))))
            ref = np.asarray(_pk_to_wp(rp, np.log(k), np.log(pk)))
            # the truth: a direct transverse-only 2D Hankel.  Ratios cancel the prefactor and
            # the two-step's finite pi_max.
            d_b = np.array([np.trapezoid(k * pk * b_k * j0(k * r), k) for r in rp])
            d_r = np.array([np.trapezoid(k * pk * j0(k * r), k) for r in rp])
            got_ratio, want_ratio = got / ref, d_b / d_r

            active = np.abs(1.0 - want_ratio) > 0.02
            assert active.sum() >= 2, "test is vacuous unless the beam suppresses something"
            assert np.allclose(got_ratio[active], want_ratio[active], atol=0.01), (
                f"FWHM {fwhm_arcmin}': suppression {got_ratio[active]} != transverse "
                f"{want_ratio[active]}"
            )
            assert np.allclose(got_ratio[~active], 1.0, atol=0.03), (
                f"FWHM {fwhm_arcmin}': beam changed r_p bins it should not have: "
                f"{got_ratio[~active]}"
            )

    def test_nz_average_brackets_the_single_z_result(self, cross_gy, hod_params):
        """An n(z)-weighted profile must lie between the single-z profiles it averages."""
        rp = np.array([0.3, 1.0])
        z_arr = np.array([0.2, 0.4])
        nz = np.array([0.5, 0.5])
        avg = np.asarray(cross_gy.projected_gy_nz(rp, z_arr, nz, _THETA, hod_params,
                                                  beam_fwhm_arcmin=1.6))
        lo = np.asarray(cross_gy.projected_gy(rp, 0.2, _THETA, hod_params, beam_fwhm_arcmin=1.6))
        hi = np.asarray(cross_gy.projected_gy(rp, 0.4, _THETA, hod_params, beam_fwhm_arcmin=1.6))
        assert np.allclose(avg, 0.5 * (lo + hi), rtol=1e-6)

    def test_nz_delta_function_equals_single_z(self, cross_gy, hod_params):
        rp = np.array([0.5, 2.0])
        avg = np.asarray(cross_gy.projected_gy_nz(rp, np.array([0.1, _Z, 0.5]),
                                                  np.array([0.0, 1.0, 0.0]),
                                                  _THETA, hod_params))
        one = np.asarray(cross_gy.projected_gy(rp, _Z, _THETA, hod_params))
        assert np.allclose(avg, one, rtol=1e-10)

    def test_nz_rejects_bad_weights(self, cross_gy, hod_params):
        rp = np.array([1.0])
        with pytest.raises(ValueError):
            cross_gy.projected_gy_nz(rp, np.array([0.2, 0.3]), np.array([0.0, 0.0]),
                                     _THETA, hod_params)
        with pytest.raises(ValueError):
            cross_gy.projected_gy_nz(rp, np.array([0.2, 0.3]), np.array([1.0]),
                                     _THETA, hod_params)


class TestCapFilter:
    """Compensated aperture photometry on the model side."""

    def test_flat_background_cancels(self):
        """The defining property: equal-area disk minus ring kills a constant.

        The cancellation is exact analytically (Y = pi c theta^2, so 2Y(td) - Y(sqrt2 td) = 0).
        What survives is only the O(dtheta^2) error from linearly interpolating a quadratic
        cumulative at off-grid theta_d -- ~1e-6 relative here, and it falls as dtheta^2 (see
        test_flat_background_residual_is_grid_error).  That is five orders of magnitude below
        the ~3% the beam itself moves the inner bins.
        """
        from hod_mod.observables.cross_spectra import cap_filter

        th = np.linspace(0.0, 20.0, 4001)
        flat = np.full_like(th, 3.7e-6)
        for td in (1.0, 2.7, 5.0):
            scale = 2.0 * np.pi * td ** 2 * 3.7e-6      # the disk's integrated value
            assert abs(cap_filter(flat, th, td)) < 1e-4 * scale, (
                f"a flat background must cancel at theta_d={td}"
            )

    def test_flat_background_residual_is_grid_error(self):
        """Pin *why* the flat background does not cancel to machine precision.

        If the residual ever stops scaling as dtheta^2, it is no longer interpolation error and
        the compensation itself is broken -- which a fixed tolerance would not catch.
        """
        from hod_mod.observables.cross_spectra import cap_filter

        c, td = 3.7e-6, 1.0
        res = []
        for n in (4001, 40001):
            th = np.linspace(0.0, 20.0, n)
            res.append(abs(cap_filter(np.full_like(th, c), th, td)))
        # 10x finer grid -> ~100x smaller error (allow a wide band; it is an asymptotic rate)
        assert res[0] / res[1] > 20.0, f"residual scaling {res[0] / res[1]:.1f}x is not O(dtheta^2)"

    def test_matches_analytic_gaussian(self):
        """Against a Gaussian profile with an analytic 2*Y(R) - Y(sqrt2 R)."""
        from hod_mod.observables.cross_spectra import cap_filter

        s, amp = 1.5, 2e-5
        th = np.linspace(0.0, 40.0, 20001)
        prof = amp * np.exp(-0.5 * (th / s) ** 2)
        y_an = lambda t: 2.0 * np.pi * amp * s ** 2 * (1.0 - np.exp(-0.5 * (t / s) ** 2))  # noqa: E731
        for td in (0.5, 1.5, 4.0):
            expect = 2.0 * y_an(td) - y_an(np.sqrt(2.0) * td)
            assert np.isclose(cap_filter(prof, th, td), expect, rtol=1e-4)

    def test_vector_and_scalar_theta_d(self):
        from hod_mod.observables.cross_spectra import cap_filter

        th = np.linspace(0.0, 20.0, 2001)
        prof = np.exp(-th)
        vec = cap_filter(prof, th, np.array([1.0, 2.0]))
        assert vec.shape == (2,)
        assert np.isclose(vec[0], cap_filter(prof, th, 1.0))
        assert np.ndim(cap_filter(prof, th, 1.0)) == 0

    def test_rejects_too_short_theta_grid(self):
        """sqrt(2)*theta_d must be covered, or the ring is silently truncated and T_AP is
        biased high with nothing to indicate it."""
        from hod_mod.observables.cross_spectra import cap_filter

        th = np.linspace(0.0, 5.0, 501)
        with pytest.raises(ValueError, match="sqrt"):
            cap_filter(np.ones_like(th), th, 4.0)

    def test_rejects_unsorted_theta(self):
        from hod_mod.observables.cross_spectra import cap_filter

        th = np.array([0.0, 2.0, 1.0, 3.0])
        with pytest.raises(ValueError, match="increasing"):
            cap_filter(np.ones_like(th), th, 1.0)


class TestSigmaYTheta:
    def test_angular_grid_matches_manual_conversion(self, cross_gy, hod_params):
        from hod_mod.core.distances import comoving_distance

        th = np.array([1.0, 3.0, 6.0])
        got = np.asarray(cross_gy.sigma_y_theta(th, _Z, _THETA, hod_params))
        chi = float(comoving_distance(np.array([_Z]), _THETA["h"], _THETA["Omega_m"])[0]) \
            * _THETA["h"]
        rp = np.deg2rad(th / 60.0) * chi
        want = np.asarray(cross_gy.projected_gy(rp, _Z, _THETA, hod_params))
        assert np.allclose(got, want, rtol=1e-10)

    def test_cap_of_model_is_finite_and_rises(self, cross_gy, hod_params):
        """The CAP of a real model profile: positive, and rising with aperture as the disk
        captures more of the (positive) pressure signal."""
        from hod_mod.observables.cross_spectra import cap_filter

        th = np.linspace(1e-3, 12.0, 600)
        prof = np.asarray(cross_gy.sigma_y_theta(th, _Z, _THETA, hod_params,
                                                 beam_fwhm_arcmin=1.6))
        tap = cap_filter(prof, th, np.array([1.0, 2.0, 4.0]))
        assert np.all(np.isfinite(tap))
        assert np.all(tap > 0)
        assert tap[0] < tap[1] < tap[2]


class TestBeamAgainstIndependentRealSpaceImplementation:
    """Cross-check the beam and CAP against the tuto_stage stacking notebook.

    The notebook (`tuto_stage/notebooks/sz_actxdesi_bgs_measurement.ipynb`) convolves in *real*
    angular space -- an explicit 2D integral over a Gaussian kernel -- and filters with a
    cumulative-integral CAP.  hod_mod convolves in *harmonic* space by multiplying the power
    spectrum by B_ell.  Two different implementations of the same physics, so agreement is a
    real test of both.  These are ports of the notebook's `beam_smear` and `ap_of`.
    """

    @staticmethod
    def _notebook_beam_smear(prof, th, fwhm_arcmin):
        """Verbatim port of the notebook's real-space 2D angular convolution."""
        sig = np.radians(fwhm_arcmin / 60) / 2.355
        phi = np.linspace(0, 2 * np.pi, 241)
        out = np.empty_like(th)

        def kern(d):
            return np.exp(-(d ** 2) / (2 * sig ** 2)) / (2 * np.pi * sig ** 2)

        for i, t in enumerate(th):
            d = np.sqrt(np.clip(t * t + th * th - 2 * t * th * np.cos(phi[:, None]), 0, None))
            out[i] = np.trapezoid(th * prof * np.trapezoid(kern(d), phi, axis=0), th)
        return out

    @staticmethod
    def _notebook_ap_of(prof, th, R):
        """Verbatim port of the notebook's CAP (`ap_of`)."""
        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import interp1d

        y_cum = 2 * np.pi * cumulative_trapezoid(th * prof, th, initial=0)
        f = interp1d(th, y_cum, bounds_error=False, fill_value=(0.0, y_cum[-1]))
        return 2 * f(R) - f(np.sqrt(2) * R)

    def test_cap_matches_notebook_ap_of(self):
        """hod_mod's cap_filter vs the notebook's ap_of on the same profile."""
        from hod_mod.observables.cross_spectra import cap_filter

        th = np.linspace(1e-4, 30.0, 6000)
        prof = 1e-5 * (1.0 + (th / 1.2) ** 2) ** (-1.0)
        for rad in (1.0, 2.7, 5.0):
            got = cap_filter(prof, th, rad)
            want = float(self._notebook_ap_of(prof, th, rad))
            assert np.isclose(got, want, rtol=1e-3), f"R={rad}: {got} vs notebook {want}"

    def test_harmonic_beam_matches_notebook_real_space_beam(self):
        """The beam, done two completely different ways.

        hod_mod multiplies P(k) by B(k*chi) and Abel-projects; the notebook convolves the
        projected profile with a Gaussian in real angular space.  Compared through the CAP,
        which is what a stacking analysis actually reports.
        """
        from hod_mod.core.numerics import safe_log
        from hod_mod.observables.cross_spectra import _pk_to_wp, cap_filter, psf_window_ell

        k = np.logspace(-4, 3, 4000)
        pk = 1e2 * k ** (-1.5) / (1.0 + (k / 5.0) ** 4)
        chi = 900.0                      # Mpc/h
        fwhm = 2.0                       # arcmin

        # angular grid, and the matching comoving r_p
        th = np.linspace(1e-3, 25.0, 700)
        rp = np.deg2rad(th / 60.0) * chi

        b_k = np.asarray(psf_window_ell(k * chi, fwhm * 60.0))
        harmonic = np.asarray(_pk_to_wp(rp, np.log(k), np.log(pk) + np.asarray(safe_log(b_k))))
        raw = np.asarray(_pk_to_wp(rp, np.log(k), np.log(pk)))
        real_space = self._notebook_beam_smear(raw, np.radians(th / 60.0), fwhm)

        for rad in (2.0, 4.0):
            a = cap_filter(harmonic, th, rad)
            b = float(self._notebook_ap_of(real_space, th, rad))
            assert np.isclose(a, b, rtol=0.05), (
                f"R={rad}': harmonic-space beam T_AP={a:.4e} vs notebook real-space "
                f"T_AP={b:.4e} -- the two beam implementations disagree"
            )


class TestPkToWpAccuracyEnvelope:
    """Characterise _pk_to_wp against an exact analytic transform pair.

    Not a test of the beam -- a test of the projection the beam rides on, added because the
    beam's *ratio* (beamed/unbeamed) is only as trustworthy as the transform underneath it.

    The pair is exact::

        P(k) = (2 pi a^2)^{3/2} exp(-k^2 a^2 / 2)
        xi(r) = exp(-r^2 / 2a^2)
        w(rp) = sqrt(2 pi) a exp(-rp^2 / 2a^2)

    Until 2026-07 the underlying rule was not an Ogata quadrature at all but a trapezoid
    truncated at k*r ~ 8 (a factor 1/h missing from the nodes), and it returned the wrong *sign*
    at a=1, r=4 where xi = 3.4e-4.  These tests now assert the transform is correct there; they
    are the regression guard that should have existed before.
    """

    @staticmethod
    def _analytic(a, r):
        return np.sqrt(2.0 * np.pi) * a * np.exp(-(r ** 2) / (2.0 * a ** 2))

    @staticmethod
    def _table(a):
        k = np.logspace(-4, 3, 6000)
        p = (2.0 * np.pi * a ** 2) ** 1.5 * np.exp(-0.5 * k ** 2 * a ** 2)
        return np.log(k), np.log(np.maximum(p, 1e-300))

    def test_accurate_where_the_signal_is_healthy(self):
        """Where xi(r) is not tiny, the projection is good to ~1%."""
        from hod_mod.observables.cross_spectra import _pk_to_wp

        lk, lp = self._table(3.0)
        r = np.array([0.5, 1.0, 2.0, 4.0])
        got = np.asarray(_pk_to_wp(r, lk, lp))
        assert np.allclose(got, self._analytic(3.0, r), rtol=0.02)

    def test_accurate_in_the_tiny_xi_regime(self):
        """The regime the predecessor got wrong: a=1, where xi falls to 3.4e-4 by r=4.

        The truncated-trapezoid rule returned -2.5e-3 here -- the wrong sign and 7.6x the
        magnitude -- because its nodes only reached k*r ~ 8, so it never saw the power that
        cancels the integrand. A correct j0 transform reproduces the analytic value.
        """
        from hod_mod.observables.cross_spectra import _pk_to_wp

        lk, lp = self._table(1.0)
        for r_i in (0.5, 1.0, 2.0, 4.0):
            r = np.array([r_i])
            got = float(np.asarray(_pk_to_wp(r, lk, lp))[0])
            exact = float(self._analytic(1.0, r)[0])
            assert np.isclose(got, exact, rtol=0.05), (
                f"a=1, r={r_i}: got {got:.4e}, exact {exact:.4e}"
            )

    def test_reaches_far_enough_in_k(self):
        """The nodes must span k*r out to ~1600, not ~8.

        This is the bug in one line: `x = pi*(h*n)*tanh(...)` instead of `pi*n*tanh(...)` made
        the rule a trapezoid truncated at pi*h*N = 8.04, and the compensating `h` prefactor in
        _pk_to_xi hid it in the normalisation. Everything else here is downstream of that.
        """
        from hod_mod.observables.clustering import _OG_X

        x_max = float(np.asarray(_OG_X).max())
        assert x_max > 1000.0, (
            f"Ogata nodes reach only k*r = {x_max:.2f}; the DE rule needs ~(pi/h)*psi(hN) "
            f"~ 1608. A truncated node range silently biases every xi(r)."
        )
