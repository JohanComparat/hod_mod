"""Functional coverage for thin/indirectly-tested modules.

Covers: hod_mod.paths, gas.metallicity, gas.erosita_response,
forecast.pk_camb_ratio, core.halo_model.HaloModelPowerSpectrum,
connection.hod.lange25 (assembly-bias path), more15 const-f_inc occupation,
vanuitert16 standalone functions, and the fitting/clustering HOD dispatch
tables.
"""

import numpy as np
import pytest
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# hod_mod.paths
# ---------------------------------------------------------------------------

class TestPaths:
    def test_repo_root_default_and_override(self, monkeypatch, tmp_path):
        from hod_mod import paths

        monkeypatch.delenv("HOD_MOD_REPO", raising=False)
        root = paths.repo_root()
        assert (root / "hod_mod" / "paths.py").exists()
        monkeypatch.setenv("HOD_MOD_REPO", str(tmp_path))
        assert paths.repo_root() == tmp_path

    def test_results_root_and_path(self, monkeypatch, tmp_path):
        from hod_mod import paths

        monkeypatch.setenv("HOD_MOD_RESULTS", str(tmp_path / "res"))
        assert paths.results_root() == tmp_path / "res"
        p = paths.results_path("runA", "chain.npz")
        assert p == tmp_path / "res" / "runA" / "chain.npz"
        assert p.parent.is_dir()          # mkdir=True default
        p2 = paths.results_path("runB", "x.npz", mkdir=False)
        assert not p2.parent.exists()

    def test_results_root_platformdirs_fallback(self, monkeypatch):
        from hod_mod import paths

        monkeypatch.delenv("HOD_MOD_RESULTS", raising=False)
        root = paths.results_root()
        assert root.name == "results"
        # never inside the repository
        assert not str(root).startswith(str(paths.repo_root()))

    def test_data_root_and_path(self, monkeypatch, tmp_path):
        from hod_mod import paths

        monkeypatch.delenv("HOD_MOD_DATA_DIR", raising=False)
        assert paths.data_root().name == "data"
        assert (paths.data_root() / "__init__.py").exists()   # in-repo default
        monkeypatch.setenv("HOD_MOD_DATA_DIR", str(tmp_path))
        assert paths.data_root() == tmp_path
        p = paths.data_path("xray_bands", "f1.h5")
        assert p == tmp_path / "xray_bands" / "f1.h5"
        assert not p.parent.exists()      # mkdir=False default (reader)
        paths.data_path("xray_bands", "f1.h5", mkdir=True)
        assert p.parent.is_dir()

    def test_sum_stat_and_cache_roots(self, monkeypatch, tmp_path):
        from hod_mod import paths

        monkeypatch.setenv("HOD_MOD_SUMSTAT", str(tmp_path / "ss"))
        assert paths.sum_stat_root() == tmp_path / "ss"
        monkeypatch.delenv("HOD_MOD_SUMSTAT", raising=False)
        assert paths.sum_stat_root().name == "data"

        monkeypatch.setenv("HOD_MOD_CACHE", str(tmp_path / "cache"))
        assert paths.cache_root() == tmp_path / "cache"
        monkeypatch.delenv("HOD_MOD_CACHE", raising=False)
        assert "hod_mod" in str(paths.cache_root())


# ---------------------------------------------------------------------------
# gas.metallicity — float64 pins double as the Wave-1 jnp-port oracle
# ---------------------------------------------------------------------------

class TestMetallicityProfileDPM:
    def test_normalisation_at_03_r200(self):
        from hod_mod.gas.metallicity import MetallicityProfileDPM

        m = MetallicityProfileDPM()
        for r200 in (0.5, 1.0, 2.3):
            z = m.metallicity_3d(np.array([0.3 * r200]), r200)
            assert float(z[0]) == pytest.approx(0.3, rel=2e-5)

    def test_pinned_profile_values(self):
        from hod_mod.gas.metallicity import MetallicityProfileDPM

        # float64 reference computed with the numpy implementation
        # (2026-07-07); the jnp port must reproduce these to float32 accuracy
        m = MetallicityProfileDPM()
        r = np.array([0.03, 0.1, 0.3, 0.5, 1.0])
        expected = np.array([0.52133886, 0.41115399, 0.30000000,
                             0.25009085, 0.18845971])
        np.testing.assert_allclose(
            np.asarray(m.metallicity_3d(r, 1.0)), expected, rtol=2e-5)

    def test_monotone_decreasing_and_scale_invariant(self):
        from hod_mod.gas.metallicity import MetallicityProfileDPM

        m = MetallicityProfileDPM()
        r = np.geomspace(0.01, 3.0, 64)
        z1 = np.asarray(m.metallicity_3d(r, 1.0))
        assert np.all(np.diff(z1) < 0)
        assert np.all(z1 > 0)
        # profile depends only on r/r200
        z2 = np.asarray(m.metallicity_3d(2.0 * r, 2.0))
        np.testing.assert_allclose(z1, z2, rtol=1e-12)


# ---------------------------------------------------------------------------
# gas.erosita_response
# ---------------------------------------------------------------------------

class TestErositaResponse:
    @pytest.fixture(scope="class")
    def resp(self):
        from hod_mod.gas.erosita_response import ErositaResponse
        return ErositaResponse()

    def test_response_arrays_consistent(self, resp):
        n = len(resp.e_obs)
        assert resp.arf.shape == (n,) and resp.g.shape == (n,)
        assert np.all(resp.de_obs > 0)
        assert np.all((resp.g >= 0) & (resp.g <= 1))
        assert resp.band == (0.5, 2.0)

    def test_powerlaw_ecf_magnitude(self, resp):
        # the shipped default is the TM0 survey-averaged response; its
        # inverse ECF for the standard AGN power law lands near 1.7e-12
        # (the 1.05e-12 in the module docstring is the TM1-7 artifact)
        ecf = resp.ecf_powerlaw(1.9, 0.0, 0.03)
        assert 5e-13 < 1.0 / ecf < 5e-12

    def test_absorption_orderings(self, resp):
        e_abs = resp.ecf_powerlaw(1.9, 0.0, 0.03, absorb=True)
        e_noabs = resp.ecf_powerlaw(1.9, 0.0, 0.03, absorb=False)
        e_thick = resp.ecf_powerlaw(1.9, 0.0, 1.0, absorb=True)
        assert e_noabs > e_abs > e_thick > 0

    def test_transmission_cached_and_bounded(self, resp):
        t1 = resp._transmission(0.03)
        t2 = resp._transmission(0.03)
        assert t1 is t2                       # cache hit
        assert np.all((t1 >= 0.0) & (t1 <= 1.0))
        # harder photons see less absorption
        assert t1[np.argmin(np.abs(resp.e_obs - 1.8))] > \
            t1[np.argmin(np.abs(resp.e_obs - 0.6))]

    def test_redshift_moves_ecf(self, resp):
        assert resp.ecf_powerlaw(1.9, 0.0) != resp.ecf_powerlaw(1.9, 0.5)

    def test_load_ecf_tables(self):
        from hod_mod.gas.erosita_response import _ECF_TABLE_DIR, load_ecf_tables
        import os

        gas, ecf_agn, ecf_fixed = load_ecf_tables("S1")
        d = np.load(os.path.join(_ECF_TABLE_DIR, "ecf_tables_S1.npz"))
        kT, eg = d["kT_grid"], d["ecf_gas"]
        # exact at the table nodes
        np.testing.assert_allclose(np.asarray(gas(kT)), eg, rtol=1e-12)
        # flat extrapolation beyond both ends
        assert float(gas(1e-4)) == pytest.approx(float(eg[0]))
        assert float(gas(1e3)) == pytest.approx(float(eg[-1]))
        assert ecf_agn > 0 and ecf_fixed > 0


# ---------------------------------------------------------------------------
# forecast.pk_camb_ratio
# ---------------------------------------------------------------------------

class TestPkCambRatio:
    def test_load_table_structure(self):
        from hod_mod.forecast.pk_camb_ratio import load

        tab = load()
        n_k = tab["lnk"].shape[0]
        assert tab["lnr0"].shape == (n_k,)
        assert tab["dlnr"].shape == (len(tab["names"]), n_k)
        assert set(tab["names"]) == {"h", "Omega_b", "Omega_m", "n_s", "sum_mnu"}
        assert tab["fid"].shape == (len(tab["names"]),)
        # the fiducial correction is genuinely non-trivial (BAO wiggles)
        assert float(jnp.max(jnp.abs(tab["lnr0"]))) > 0.005

    def test_load_missing_file_raises(self, tmp_path):
        from hod_mod.forecast.pk_camb_ratio import load

        with pytest.raises(FileNotFoundError, match="build it once"):
            load(str(tmp_path / "nope.npz"))

    def test_apply_ratio_trivial_table_is_identity(self):
        from hod_mod.forecast.pk_camb_ratio import apply_ratio

        k = jnp.geomspace(1e-3, 10.0, 50)
        pk = k ** -1.5
        tab = dict(lnk=jnp.log(k), lnr0=jnp.zeros_like(k),
                   dlnr=jnp.zeros((0, k.shape[0])), names=[],
                   fid=jnp.zeros((0,)))
        out = apply_ratio(pk, k, {}, tab)
        np.testing.assert_allclose(np.asarray(out), np.asarray(pk), rtol=1e-6)

    def test_apply_ratio_fiducial_and_derivative_row(self):
        from hod_mod.forecast.pk_camb_ratio import apply_ratio, load

        tab = load()
        k = jnp.geomspace(1e-3, 10.0, 200)
        pk = jnp.ones_like(k)
        theta_fid = {n: float(tab["fid"][i]) for i, n in enumerate(tab["names"])}
        out_fid = apply_ratio(pk, k, theta_fid, tab)
        expected = jnp.exp(jnp.interp(jnp.log(k), tab["lnk"], tab["lnr0"]))
        np.testing.assert_allclose(np.asarray(out_fid), np.asarray(expected),
                                   rtol=1e-6)
        # moving h off-fiducial engages the derivative row
        theta_h = dict(theta_fid, h=theta_fid["h"] + 0.02)
        out_h = apply_ratio(pk, k, theta_h, tab)
        assert not np.allclose(np.asarray(out_h), np.asarray(out_fid))


# ---------------------------------------------------------------------------
# core.halo_model.HaloModelPowerSpectrum
# ---------------------------------------------------------------------------

class TestHaloModelPowerSpectrum:
    @pytest.fixture(scope="class")
    def hm(self, hmf, halo_profile, pk_lin):
        from hod_mod.core.halo_model import HaloModelPowerSpectrum
        return HaloModelPowerSpectrum(hmf, halo_profile, pk_lin, n_m=60)

    def test_pk_mm_is_sum_of_terms(self, hm, planck_cosmo):
        k = np.geomspace(1e-2, 10.0, 20)
        p1 = np.asarray(hm.pk_1h_mm(k, 0.3, planck_cosmo))
        p2 = np.asarray(hm.pk_2h_mm(k, 0.3, planck_cosmo))
        pt = np.asarray(hm.pk_mm(k, 0.3, planck_cosmo))
        assert np.all(p1 > 0) and np.all(p2 > 0)
        np.testing.assert_allclose(pt, p1 + p2, rtol=1e-6)

    def test_2halo_tracks_linear_on_large_scales(self, hm, pk_lin, planck_cosmo):
        k = np.array([1e-3, 3e-3])
        p2 = np.asarray(hm.pk_2h_mm(k, 0.0, planck_cosmo))
        plin = np.asarray(pk_lin.pk_linear(jnp.asarray(k), 0.0, planck_cosmo))
        # finite mass range truncates the bias integral; expect O(1)
        ratio = p2 / plin
        assert np.all((ratio > 0.4) & (ratio < 1.5))

    def test_1halo_dominates_small_scales(self, hm, planck_cosmo):
        k = np.array([20.0])
        p1 = float(hm.pk_1h_mm(k, 0.3, planck_cosmo)[0])
        p2 = float(hm.pk_2h_mm(k, 0.3, planck_cosmo)[0])
        assert p1 > p2


# ---------------------------------------------------------------------------
# connection.hod.lange25 — assembly-bias branch
# ---------------------------------------------------------------------------

class TestLange25AssemblyBias:
    def test_both_constructor_forms(self, hmf):
        from hod_mod.connection.hod.lange25 import Lange25HODModel

        m1 = Lange25HODModel(hmf)                 # _SINGLE_ARG_INIT path
        m2 = Lange25HODModel(hmf, hmf.bias)
        p = Lange25HODModel.default_params()
        lm = jnp.linspace(11.0, 16.0, 100)
        for a, b in zip(m1.nc_ns(lm, p), m2.nc_ns(lm, p)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_f_gamma_scales_centrals_only(self, hmf):
        from hod_mod.connection.hod.lange25 import Lange25HODModel

        model = Lange25HODModel(hmf)
        p = Lange25HODModel.default_params()
        lm = jnp.linspace(11.0, 16.0, 100)
        nc1, ns1 = model.nc_ns(lm, dict(p, f_Gamma=1.0))
        nc7, ns7 = model.nc_ns(lm, dict(p, f_Gamma=0.7))
        np.testing.assert_allclose(np.asarray(nc7), 0.7 * np.asarray(nc1),
                                   rtol=1e-6)
        np.testing.assert_allclose(np.asarray(ns7), np.asarray(ns1), rtol=1e-6)

    def test_assembly_bias_moves_beff_not_ngal(self, hmf, planck_cosmo):
        from hod_mod.connection.hod.lange25 import Lange25HODModel

        model = Lange25HODModel(hmf)
        p = Lange25HODModel.default_params()
        n0, b0, _ = model._integrate(0.3, planck_cosmo, p)
        np_, bp, _ = model._integrate(0.3, planck_cosmo, dict(p, A_cen=0.5))
        nm, bm, _ = model._integrate(0.3, planck_cosmo, dict(p, A_cen=-0.5))
        # a (b-1)/b kernel with galaxies in b>1 halos: +A_cen raises b_eff
        assert float(bp) > float(b0) > float(bm)
        assert float(np_) == pytest.approx(float(n0), rel=1e-12)
        assert float(nm) == pytest.approx(float(n0), rel=1e-12)
        # satellites carry the same kernel
        _, bs, _ = model._integrate(0.3, planck_cosmo, dict(p, A_sat=0.5))
        assert float(bs) > float(b0)


# ---------------------------------------------------------------------------
# connection.hod.more15 — constant-f_inc occupation
# ---------------------------------------------------------------------------

class TestMore15ConstFinc:
    def test_ncen_is_finc_times_erfc(self):
        from hod_mod.connection.hod.base import n_cen
        from hod_mod.connection.hod.more15 import n_cen_more15_const_finc

        lm = jnp.linspace(11.0, 15.0, 200)
        base = np.asarray(n_cen(lm, 12.5, 0.4))
        for f_inc in (1.0, 0.3, 0.05):
            got = np.asarray(n_cen_more15_const_finc(lm, 12.5, 0.4, f_inc))
            np.testing.assert_allclose(got, f_inc * base, rtol=1e-6)

    def test_model_occupation_capped_by_finc(self, hmf):
        from hod_mod.connection.hod.more15 import MoreConstFincHODModel

        model = MoreConstFincHODModel(hmf, hmf.bias)
        p = dict(MoreConstFincHODModel.default_params(), f_inc=0.1)
        lm = jnp.linspace(11.0, 16.0, 200)
        nc, ns = model.nc_ns(lm, p)
        assert float(jnp.max(nc)) <= 0.1 + 1e-7
        assert jnp.all(ns >= 0.0)


# ---------------------------------------------------------------------------
# connection.hod.vanuitert16 — standalone occupation functions
# ---------------------------------------------------------------------------

class TestVanUitert16Functions:
    def _params(self):
        from hod_mod.connection.hod.vanuitert16 import VanUitert16CSMFModel
        return VanUitert16CSMFModel.default_params()

    def test_ncen_bounded_and_bin_additive(self):
        from hod_mod.connection.hod.vanuitert16 import n_cen_vanuitert16

        p = self._params()
        lm = jnp.linspace(10.5, 15.5, 200)
        args = (p["log10m_star0"], p["log10m_h1"], p["beta1"],
                p["log10_beta2"], p["sigma_c"])
        nc = n_cen_vanuitert16(lm, 9.8, 10.3, *args)
        assert jnp.all((nc >= 0.0) & (nc <= 1.0))
        # adjacent stellar-mass bins add up to the union bin
        nc_lo = n_cen_vanuitert16(lm, 9.8, 10.0, *args)
        nc_hi = n_cen_vanuitert16(lm, 10.0, 10.3, *args)
        np.testing.assert_allclose(np.asarray(nc_lo + nc_hi), np.asarray(nc),
                                   rtol=1e-5, atol=1e-7)

    def test_nsat_shapes_and_scaling(self):
        from hod_mod.connection.hod.vanuitert16 import n_sat_vanuitert16

        p = self._params()
        args = (9.8, 10.3, p["log10m_star0"], p["log10m_h1"], p["beta1"],
                p["log10_beta2"], p["alpha_s"], p["b0"], p["b1"])
        # scalar in, scalar-shaped out
        s = n_sat_vanuitert16(jnp.asarray(14.0), *args)
        assert jnp.shape(s) == ()
        lm = jnp.linspace(12.0, 15.5, 50)
        ns = n_sat_vanuitert16(lm, *args)
        assert ns.shape == lm.shape
        assert jnp.all(ns >= 0.0)
        # phi_s grows as 10^(b1 log10(M/1e13)): satellites increase with M
        assert float(ns[-1]) > float(ns[0])
        assert float(s) == pytest.approx(
            float(ns[jnp.argmin(jnp.abs(lm - 14.0))]), rel=1e-3)

    def test_ntotal_is_sum(self):
        from hod_mod.connection.hod.vanuitert16 import (
            n_cen_vanuitert16, n_sat_vanuitert16, n_total_vanuitert16)

        p = self._params()
        lm = jnp.linspace(11.0, 15.5, 100)
        common = (9.8, 10.3, p["log10m_star0"], p["log10m_h1"], p["beta1"],
                  p["log10_beta2"])
        nt = n_total_vanuitert16(lm, *common, p["sigma_c"], p["alpha_s"],
                                 p["b0"], p["b1"])
        nc = n_cen_vanuitert16(lm, *common, p["sigma_c"])
        ns = n_sat_vanuitert16(lm, *common, p["alpha_s"], p["b0"], p["b1"])
        # n_total is one fused jit; float32 reordering vs the separate
        # calls leaves ~4e-6 relative differences
        np.testing.assert_allclose(np.asarray(nt), np.asarray(nc + ns),
                                   rtol=1e-5)


# ---------------------------------------------------------------------------
# HOD dispatch tables — every registered model constructs and occupies
# ---------------------------------------------------------------------------

class TestHodDispatchTables:
    def _instantiate(self, cls, hmf):
        try:
            return cls(hmf, hmf.bias)
        except TypeError:
            return cls(hmf)

    def test_every_fitting_model_constructs_and_occupies(self, hmf):
        from hod_mod.fitting.models import HOD_MODELS

        lm = jnp.linspace(11.0, 15.5, 64)
        for name, cls in HOD_MODELS.items():
            model = self._instantiate(cls, hmf)
            p = cls.default_params()
            nc, ns = model.nc_ns(lm, p)
            assert jnp.all(jnp.isfinite(nc)), name
            assert jnp.all(jnp.isfinite(ns)), name
            assert jnp.all(nc >= 0.0) and jnp.all(ns >= 0.0), name

    def test_every_clustering_key_dispatches(self, hmf):
        from hod_mod.observables.clustering import (
            _HOD_MODEL_MAP, NonLinearHaloModelPrediction)

        keys = list(_HOD_MODEL_MAP) + ["clf_cacciato13"]
        for key in keys:
            obj = NonLinearHaloModelPrediction._build_hod(key, hmf)
            assert hasattr(obj, "nc_ns"), key
            assert isinstance(obj.default_params(), dict), key

    def test_clf_cacciato13_is_a_valid_public_key(self):
        # regression: the key was dispatchable in _build_hod but missing
        # from _HOD_MODEL_MAP, so the public constructor rejected it
        from hod_mod.observables.clustering import _HOD_MODEL_MAP

        assert "clf_cacciato13" in _HOD_MODEL_MAP
