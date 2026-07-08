"""Wave-1 regression: FullHaloModelPrediction jnp table assembly vs float64 numpy.

The per-call P(k)-table assembly in ``_pk_tables_full`` was converted from
np.trapezoid (float64) to jnp.trapezoid (float32).  These tests re-assemble
the same tables in float64 numpy directly from the static cache and pin the
jnp results to them at float32-limited tolerance.  An axis/transpose or
weighting bug in the conversion would miss by orders of magnitude.

Marked slow: builds the CAMB static cache once (module-scoped predictor).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.slow

_Z = 0.4
_RTOL = 3e-5      # float32 assembly + float32 log/exp round trip


@pytest.fixture(scope="module")
def pred(pk_lin, hmf, halo_profile):
    from hod_mod.connection.hod import MoreHODModel
    from hod_mod.observables.clustering import FullHaloModelPrediction

    hod = MoreHODModel(hmf, hmf.bias)
    return FullHaloModelPrediction(pk_lin, hod, halo_profile)


@pytest.fixture(scope="module")
def hod_params():
    from hod_mod.connection.hod import MoreHODModel
    return MoreHODModel.default_params()


def _occupation_f64(pred, hod_params):
    with jax.disable_jit():
        nc, ns = pred._hod.nc_ns(pred._hod._log10m_grid, hod_params)
    return np.asarray(nc, dtype=float), np.asarray(ns, dtype=float)


def _reference_tables(pred, sc, hod_params):
    """Float64 numpy re-assembly of the standard (no-extension) branch."""
    m, dndm, bias = sc["m_np"], sc["dndm_np"], sc["bias_np"]
    uk = np.asarray(sc["uk"], dtype=float)
    pk_lin = sc["pk_lin"]
    nc, ns = _occupation_f64(pred, hod_params)
    nt = nc + ns
    n_gal = np.trapezoid(dndm * nt, m)
    b_eff = np.trapezoid(dndm * nt * bias, m) / n_gal
    p_gg_1h = np.trapezoid(
        dndm[None, :] * (ns[None, :] ** 2 * uk ** 2
                         + 2.0 * nc[None, :] * ns[None, :] * uk),
        m, axis=1) / n_gal ** 2
    m_over_rho = m / sc["rho_m"]
    p_gm_1h = np.trapezoid(
        dndm[None, :] * (nc[None, :] + ns[None, :] * uk)
        * m_over_rho[None, :] * uk,
        m, axis=1) / n_gal
    return dict(n_gal=n_gal, b_eff=b_eff,
                p_gg=np.maximum(p_gg_1h + b_eff ** 2 * pk_lin, 1e-20),
                p_gm=np.maximum(p_gm_1h + b_eff * pk_lin, 1e-20))


class TestPkTablesFloat64Oracle:
    def test_standard_branch(self, pred, hod_params, planck_cosmo):
        tables = pred._pk_tables_full(_Z, planck_cosmo, hod_params)
        sc = pred._static_cache[pred._cosmo_cache_key(_Z, planck_cosmo)]
        ref = _reference_tables(pred, sc, hod_params)

        assert tables["n_gal"] == pytest.approx(ref["n_gal"], rel=_RTOL)
        assert tables["b_eff"] == pytest.approx(ref["b_eff"], rel=_RTOL)
        np.testing.assert_allclose(
            np.exp(np.asarray(tables["log_pgg"], dtype=float)), ref["p_gg"],
            rtol=_RTOL)
        np.testing.assert_allclose(
            np.exp(np.asarray(tables["log_pgm"], dtype=float)), ref["p_gm"],
            rtol=_RTOL)

    def test_off_centering_branch(self, pred, hod_params, planck_cosmo):
        p = dict(hod_params, p_off=0.3, R_off=2.0)
        tables = pred._pk_tables_full(_Z, planck_cosmo, p)
        sc = pred._static_cache[pred._cosmo_cache_key(_Z, planck_cosmo)]

        m, dndm = sc["m_np"], sc["dndm_np"]
        k = sc["k_np"]
        uk = np.asarray(sc["uk"], dtype=float)
        nc, ns = _occupation_f64(pred, p)
        n_gal = np.trapezoid(dndm * (nc + ns), m)
        r_s_m = sc["r_delta"] / sc["c_np"]
        w_off = np.exp(-k[:, None] ** 2 * (2.0 * r_s_m[None, :]) ** 2 / 2.0)
        nc_eff = nc[None, :] * (0.7 + 0.3 * w_off)
        p_gg_1h = np.trapezoid(
            dndm[None, :] * (ns[None, :] ** 2 * uk ** 2
                             + 2.0 * nc_eff * ns[None, :] * uk),
            m, axis=1) / n_gal ** 2
        got_1h = np.exp(np.asarray(tables["log_pgg_1h"], dtype=float))
        np.testing.assert_allclose(got_1h, np.maximum(p_gg_1h, 1e-20),
                                   rtol=_RTOL)
        # off-centering must suppress the 1h term at high k
        tables0 = pred._pk_tables_full(_Z, planck_cosmo, hod_params)
        assert (got_1h[-1]
                < np.exp(float(tables0["log_pgg_1h"][-1])))

    def test_assembly_bias_branch(self, pred, hod_params, planck_cosmo):
        p = dict(hod_params, A_cen=0.4, A_sat=-0.2)
        tables = pred._pk_tables_full(_Z, planck_cosmo, p)
        sc = pred._static_cache[pred._cosmo_cache_key(_Z, planck_cosmo)]

        m, dndm, bias = sc["m_np"], sc["dndm_np"], sc["bias_np"]
        nc, ns = _occupation_f64(pred, p)
        n_gal = np.trapezoid(dndm * (nc + ns), m)
        gam = (bias - 1.0) / np.where(bias > 0.5, bias, 0.5)
        b_ref = np.trapezoid(
            dndm * (nc * bias * (1.0 + 0.4 * gam)
                    + ns * bias * (1.0 - 0.2 * gam)), m) / n_gal
        assert tables["b_eff"] == pytest.approx(b_ref, rel=_RTOL)
        # and it must differ from the undecorated bias
        tables0 = pred._pk_tables_full(_Z, planck_cosmo, hod_params)
        assert abs(tables["b_eff"] - tables0["b_eff"]) > 1e-3

    def test_satellite_extension_branch_active(self, pred, hod_params,
                                               planck_cosmo):
        # Extensions route through satellite_nfw_uk (own oracle in
        # test_cosmology); here just verify the branch changes P_gg sanely
        p = dict(hod_params, b_sat_conc=1.5, f_cut=0.1)
        t_ext = pred._pk_tables_full(_Z, planck_cosmo, p)
        t_std = pred._pk_tables_full(_Z, planck_cosmo, hod_params)
        pgg_ext = np.exp(np.asarray(t_ext["log_pgg"], dtype=float))
        pgg_std = np.exp(np.asarray(t_std["log_pgg"], dtype=float))
        assert np.all(np.isfinite(pgg_ext))
        assert not np.allclose(pgg_ext, pgg_std, rtol=1e-3)
        # 2h term (b_eff) is untouched by the satellite profile
        assert t_ext["b_eff"] == pytest.approx(t_std["b_eff"], rel=1e-6)

    def test_n_gal_method_matches_tables(self, pred, hod_params, planck_cosmo):
        tables = pred._pk_tables_full(_Z, planck_cosmo, hod_params)
        assert pred.n_gal(_Z, planck_cosmo, hod_params) == pytest.approx(
            tables["n_gal"], rel=1e-6)


class TestObservablesStillPhysical:
    def test_wp_and_delta_sigma(self, pred, hod_params, planck_cosmo):
        rp = jnp.asarray([0.5, 2.0, 10.0])
        wp = np.asarray(pred.wp(rp, 100.0, _Z, planck_cosmo, hod_params))
        assert np.all(np.isfinite(wp)) and np.all(wp > 0)
        assert wp[0] > wp[1] > wp[2]
        ds = np.asarray(pred.delta_sigma(rp, _Z, planck_cosmo, hod_params))
        assert np.all(np.isfinite(ds)) and np.all(ds > 0)
        assert ds[0] > ds[2]


class TestComovingDistance:
    def test_matches_float64_reference(self, planck_cosmo):
        from hod_mod.observables.clustering import _comoving_dist_h

        z = np.array([0.1, 0.4, 1.0, 2.0])
        got = np.asarray(_comoving_dist_h(z, planck_cosmo), dtype=float)
        om = float(planck_cosmo["Omega_m"])
        z_f = np.linspace(0.0, 2.0, 2000 * 10)
        inv_e = 1.0 / np.sqrt(om * (1 + z_f) ** 3 + 1 - om)
        chi_f = np.concatenate([
            [0.0],
            np.cumsum(2997.92 * 0.5 * (inv_e[:-1] + inv_e[1:]) * np.diff(z_f)),
        ])
        ref = np.interp(z, z_f, chi_f)
        # float32 cumsum over 2000 nodes: ~1e-5 relative
        np.testing.assert_allclose(got, ref, rtol=5e-5)
