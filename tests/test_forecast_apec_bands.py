"""Tests for hod_mod.forecast.apec_bands — band-integrated APEC tables,
the MM83 ISM photoelectric cross-section, and per-band AGN transmission.

The DEFAULT_BANDS table is shipped as a cached npz under
``hod_mod/data/apec_bands/``, so everything here runs without soxs except
the explicitly-marked build test.
"""

import numpy as np
import pytest


class TestMM83Sigma:
    def test_positive_and_decreasing_within_segments(self):
        from hod_mod.forecast.apec_bands import _MM83, mm83_sigma

        # sigma > 0 everywhere in the fit range
        e = np.geomspace(0.03, 10.0, 400)
        sig = mm83_sigma(e)
        assert np.all(sig > 0)
        assert np.all(np.isfinite(sig))
        # inside each polynomial segment the E^-3 prefactor dominates:
        # sigma is strictly decreasing between absorption edges
        edges = np.append(_MM83[:, 0], 10.0)
        for lo, hi in zip(edges[:-1], edges[1:]):
            ee = np.linspace(lo * 1.001, hi * 0.999, 32)
            assert np.all(np.diff(mm83_sigma(ee)) < 0), f"segment {lo}-{hi}"

    def test_spot_value_1kev(self):
        from hod_mod.forecast.apec_bands import mm83_sigma

        # MM83 Table 2, 0.867-1.303 keV row: (120.6 + 169.3 - 47.7)e-24 cm^2
        assert mm83_sigma(1.0) == pytest.approx(2.422e-22, rel=1e-6)
        # soft X-rays are far more absorbed than hard
        assert mm83_sigma(0.5) > 100 * mm83_sigma(8.0)

    def test_scalar_and_array_inputs_agree(self):
        from hod_mod.forecast.apec_bands import mm83_sigma

        e = np.array([0.3, 1.0, 2.0, 5.0])
        vec = mm83_sigma(e)
        for ei, vi in zip(e, vec):
            assert mm83_sigma(ei) == pytest.approx(vi, rel=1e-14)


class TestBandTransmission:
    def test_in_unit_interval_and_ordered_in_energy(self):
        from hod_mod.forecast.apec_bands import DEFAULT_BANDS, band_transmission

        t = band_transmission(DEFAULT_BANDS, nh=1e22)
        assert t.shape == (len(DEFAULT_BANDS),)
        assert np.all(t > 0) and np.all(t <= 1.0)
        # harder bands are more transparent
        assert np.all(np.diff(t) > 0)

    def test_unity_limit_and_nh_monotonicity(self):
        from hod_mod.forecast.apec_bands import DEFAULT_BANDS, band_transmission

        t0 = band_transmission(DEFAULT_BANDS, nh=0.0)
        np.testing.assert_allclose(t0, 1.0, rtol=1e-12)
        t20 = band_transmission(DEFAULT_BANDS, nh=1e20)
        t22 = band_transmission(DEFAULT_BANDS, nh=1e22)
        t23 = band_transmission(DEFAULT_BANDS, nh=1e23)
        assert np.all(t20 > t22) and np.all(t22 > t23)

    def test_gamma_dependence_is_second_order(self):
        from hod_mod.forecast.apec_bands import DEFAULT_BANDS, band_transmission

        # docstring contract: intra-band Gamma dependence is weak, which is
        # why the template is frozen at the fiducial Gamma
        t_soft = band_transmission(DEFAULT_BANDS, nh=1e22, gamma=1.4)
        t_hard = band_transmission(DEFAULT_BANDS, nh=1e22, gamma=2.2)
        # the heavily-absorbed softest band moves ~6% relative but only
        # ~1e-4 absolute; "second order" is an absolute-scale statement
        np.testing.assert_allclose(t_soft, t_hard, rtol=0.10)
        assert np.max(np.abs(t_soft - t_hard)) < 0.01


class TestBandTables:
    def test_cached_default_bands_shapes_and_finiteness(self):
        from hod_mod.forecast.apec_bands import (
            BROAD_BAND, DEFAULT_BANDS, _N_T, _N_Z, band_tables)

        d = band_tables(DEFAULT_BANDS)
        n_bands = len(DEFAULT_BANDS)
        assert d["lt"].shape == (_N_T,)
        assert d["lz"].shape == (_N_Z,)
        assert d["tables"].shape == (n_bands + 1, _N_T, _N_Z)
        assert d["edges"].shape == (n_bands + 1, 2)
        assert np.all(np.isfinite(d["tables"]))
        # broad band is last
        np.testing.assert_allclose(d["edges"][-1], BROAD_BAND)
        # grids are strictly increasing (log10 T, log10 Z)
        assert np.all(np.diff(d["lt"]) > 0)
        assert np.all(np.diff(d["lz"]) > 0)

    def test_narrow_bands_tile_broad_band(self):
        from hod_mod.forecast.apec_bands import DEFAULT_BANDS, band_tables

        # DEFAULT_BANDS tile 0.5-2.0 keV contiguously, so the linear
        # emissivities must sum to the broad band at every (T, Z) node
        d = band_tables(DEFAULT_BANDS)
        lin = 10.0 ** d["tables"]
        np.testing.assert_allclose(lin[:-1].sum(axis=0), lin[-1], rtol=0.01)

    def test_broad_band_rises_through_threshold(self):
        from hod_mod.forecast.apec_bands import DEFAULT_BANDS, band_tables

        # for kT well below the 0.5 keV band floor, the band emissivity
        # climbs steeply with T (exponential cutoff moving into the band)
        d = band_tables(DEFAULT_BANDS)
        broad = d["tables"][-1]
        i_rise = 10.0 ** d["lt"] < 0.3
        dlam = np.diff(broad[i_rise, :], axis=0)
        assert np.all(dlam > 0)

    @pytest.mark.slow
    def test_build_and_cache_roundtrip(self, tmp_path, monkeypatch):
        pytest.importorskip("soxs")
        from hod_mod.forecast import apec_bands as ab

        # tiny un-cached configuration: exercises the soxs build branch and
        # the npz cache write/read without touching the repo data dir
        monkeypatch.setattr(ab, "_DATA_DIR", str(tmp_path))
        bands = [(0.5, 2.0)]
        kw = dict(n_T=4, T_min=0.5, T_max=8.0, n_Z=3, Z_min=0.1, Z_max=1.0)
        built = ab.band_tables(bands, **kw)
        assert built["tables"].shape == (2, 4, 3)
        assert len(list(tmp_path.glob("apec_bands_*.npz"))) == 1
        # second call must hit the cache and reproduce bit-identical arrays
        cached = ab.band_tables(bands, **kw)
        for k in ("lt", "lz", "tables", "edges"):
            np.testing.assert_array_equal(built[k], cached[k])
