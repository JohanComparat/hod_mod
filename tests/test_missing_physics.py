"""Missing-physics extension (docs/missing_physics.rst): phase gates A–E.

A: cosmology-dependent Diemer & Kravtsov c(M) wiring + eps_sn promotion
B: beyond-ΛCDM — CPL growth ODE, w0/wa/sum_mnu, ν suppression, geometry
C: SF/quiescent split (ZM16 Weibull) + the quenched L_X offset
D: fundamental-plane radio AGN (rlf observable)
E: HI sector (M_HI(M_h), himf, cl_gHI)
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_TINY = dict(n_k=48, n_m=48, n_gl=16, n_z=3, n_z_shear=3)
_Z = 0.35


@pytest.fixture(scope="module")
def fid():
    from hod_mod.forecast import params
    return jnp.asarray(params.fiducial_vector())


@pytest.fixture(scope="module")
def model(fid):
    from hod_mod.forecast.forward_jax import ForwardModel
    return ForwardModel(z_eff=_Z, **_TINY)


# ---------------------------------------------------------------- Phase A --

def test_eps_sn_promotion(fid):
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    from hod_mod.forecast import params
    assert params.load_fiducial()["eps_sn"] == pytest.approx(0.1)
    # in energy-closure mode the log10_M_pivot slot is log10 ε_AGN: use a
    # physical coupling so the energy branch of the min() is active and the
    # SN channel actually flows into f_b(M)
    th = np.asarray(fid).copy()
    th[_IDX["log10_M_pivot"]] = -2.0
    th = jnp.asarray(th)
    m_ec = ForwardModel(z_eff=_Z, energy_closure=True, **_TINY)
    m_no = ForwardModel(z_eff=_Z, energy_closure=False, **_TINY)
    for m, active in ((m_ec, True), (m_no, False)):
        f = lambda t: m.predict(t, ["ds"])["ds"]
        col = np.asarray(jax.jacfwd(f)(th))[:, _IDX["eps_sn"]]
        assert (np.abs(col).max() > 0) == active


def test_dk15_matches_core_implementation(model, fid):
    """The inline _c_dk15 mirrors core.concentration.c_diemer15 exactly."""
    from hod_mod.forecast.forward_jax import _c_dk15
    from hod_mod.core.concentration import c_diemer15
    sigma = jnp.linspace(0.4, 4.0, 20)
    neff = jnp.linspace(-2.6, -1.2, 20)
    ours = np.asarray(_c_dk15(sigma, neff))
    ref = np.asarray(c_diemer15(jnp.ones(20) * 1e13, sigma, neff, 0.31, 0.0))
    np.testing.assert_allclose(ours, ref, rtol=1e-12)


def test_dk15_concentration_is_cosmology_dependent(fid):
    """diemer15 concentration responds to sigma8; dutton14 does not."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX

    def conc_of(cm):
        m = ForwardModel(z_eff=_Z, cm_relation=cm, **_TINY)
        def f(t):
            return m._halo_common(m._theta_eff(t), _Z)["conc"]
        return np.asarray(jax.jacfwd(f)(fid))[:, _IDX["sigma8"]]

    assert np.abs(conc_of("diemer15")).max() > 1e-3
    assert np.abs(conc_of("dutton14")).max() == 0.0
    # COLOSSUS DK19-median anchors at z=0 (generous EH98-shape tolerance):
    # c200c(1e12) ≈ 9.5, c200c(1e15) ≈ 5.0
    m0 = ForwardModel(z_eff=0.0, cm_relation="diemer15", **_TINY)
    c = np.asarray(m0._halo_common(m0._theta_eff(fid), 0.0)["conc"])
    lm = np.asarray(m0.log10m)
    c12 = float(np.interp(12.0, lm, c))
    c15 = float(np.interp(15.0, lm, c))
    assert 7.0 < c12 < 12.0 and 3.5 < c15 < 6.5
    assert np.all(c > 2.0) and np.all(c < 20.0)


# ---------------------------------------------------------------- Phase B --

def test_growth_ode_lcdm_matches_carroll_and_exact():
    from hod_mod.core.halo_mass_function import (
        _growth_factor_cpl_jax, _growth_factor_flat_jax)
    z = np.linspace(0.0, 3.0, 13)
    om = 0.31
    ode = np.asarray(_growth_factor_cpl_jax(z, om))
    car = np.asarray(jax.vmap(lambda zz: _growth_factor_flat_jax(zz, om))(z))
    assert np.max(np.abs(ode / car - 1.0)) < 5e-3       # Carroll is ~0.1% itself
    # exact flat-ΛCDM growing mode: D ∝ E(a) ∫ da / (a E)³
    a = np.linspace(1e-4, 1.0, 20001)
    e = np.sqrt(om / a ** 3 + 1.0 - om)
    integ = np.cumsum(1.0 / (a * e) ** 3) * (a[1] - a[0])
    d_exact = e * integ
    d_exact /= d_exact[-1]
    for zz in (0.5, 1.0, 2.0):
        d_ref = np.interp(1.0 / (1.0 + zz), a, d_exact)
        assert abs(float(_growth_factor_cpl_jax(zz, om)) / d_ref - 1.0) < 2e-3


def test_growth_responds_to_dark_energy():
    from hod_mod.core.halo_mass_function import _growth_factor_cpl_jax
    d_lcdm = float(_growth_factor_cpl_jax(1.0, 0.31, -1.0, 0.0))
    d_quint = float(_growth_factor_cpl_jax(1.0, 0.31, -0.8, 0.0))
    # w0 > −1: dark energy dominates earlier → less recent growth → the
    # normalised D(z)/D(0) sits HIGHER at z > 0
    assert d_quint > d_lcdm
    # differentiable in w0/wa
    g = jax.grad(lambda w: _growth_factor_cpl_jax(1.0, 0.31, w, 0.0))(-1.0)
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_nu_suppression_shape(model, fid):
    """Σm_ν = 0 is exactly massless; m_ν > 0 suppresses only small scales."""
    from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear
    pk = EisensteinHu98PkLinear()
    k = jnp.logspace(-3, 1, 128)
    c0 = {"Omega_m": 0.31, "Omega_b": 0.049, "Omega_cdm": 0.261, "h": 0.6736,
          "n_s": 0.9649, "sigma8": 0.8111, "w0": -1.0, "wa": 0.0, "sum_mnu": 0.0}
    c_nu = dict(c0, sum_mnu=0.3)
    c_no_key = {kk: v for kk, v in c0.items() if kk != "sum_mnu"}
    p0 = np.asarray(pk.pk_linear(k, 0.0, c0))
    p_nokey = np.asarray(pk.pk_linear(k, 0.0, c_no_key))
    np.testing.assert_array_equal(p0, p_nokey)          # mnu=0 == no-ν path
    p_nu = np.asarray(pk.pk_linear(k, 0.0, c_nu))
    r = p_nu / p0
    assert r[0] > 0.999                                 # large scales safe
    # small scales suppressed relative to large scales (σ8-anchored shape)
    assert r[-1] < r[0] - 0.01


def test_beyond_lcdm_params_are_plumbed(model, fid):
    """w0/wa/sum_mnu move the observables through growth AND geometry."""
    from hod_mod.forecast.forward_jax import _IDX
    f, row_obs, _ = model.full_data_vector_fn(["wp", "cl_kk"])
    J = np.asarray(jax.jacfwd(f)(fid))
    d0 = np.asarray(f(fid))
    assert np.all(np.isfinite(J))
    for n in ("w0", "wa", "sum_mnu"):
        col = J[:, _IDX[n]]
        assert np.abs(col).max() > 1e-8 * np.abs(d0).max(), n


# ---------------------------------------------------------------- Phase C --

def test_sfq_split_sums_to_unsplit(fid):
    from hod_mod.forecast.forward_jax import ForwardModel
    kw = dict(z_eff=_Z, **_TINY)
    m_all = ForwardModel(**kw)
    m_sf = ForwardModel(sfq="sf", **kw)
    m_q = ForwardModel(sfq="q", **kw)
    th = m_all._theta_eff(fid)
    nc_a, ns_a = m_all._occ_sample(th)
    nc_s, ns_s = m_sf._occ_sample(th)
    nc_q, ns_q = m_q._occ_sample(th)
    np.testing.assert_allclose(np.asarray(nc_s + nc_q), np.asarray(nc_a), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ns_s + ns_q), np.asarray(ns_a), rtol=1e-12)
    # abundances partition too
    n_a = float(m_all.predict(fid, ["n_gal"])["n_gal"][0])
    n_s = float(m_sf.predict(fid, ["n_gal"])["n_gal"][0])
    n_q = float(m_q.predict(fid, ["n_gal"])["n_gal"][0])
    np.testing.assert_allclose(n_s + n_q, n_a, rtol=1e-12)
    # quenched galaxies live in more massive halos → higher effective bias
    H_s = m_sf._halo_common(th, _Z)
    H_q = m_q._halo_common(th, _Z)
    assert float(H_q["b_eff"]) > float(H_s["b_eff"])


def test_dlx_quenched_only_touches_quenched_gas(fid):
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    kw = dict(z_eff=_Z, **_TINY)
    for sfq, active in ((None, False), ("sf", False), ("q", True)):
        m = ForwardModel(sfq=sfq, **kw)
        f = lambda t: m.predict(t, ["cl_gX"])["cl_gX"]
        col = np.asarray(jax.jacfwd(f)(fid))[:, _IDX["dlx_quenched"]]
        assert (np.abs(col).max() > 0) == active, sfq


# ---------------------------------------------------------------- Phase D --

def test_rlf_identity_with_xlf(fid):
    """(ξ_RX, ξ_RM, b_R, σ_R) = (1, 0, 0, 0) collapses the FP onto L_X."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    grid = np.linspace(42.0, 45.0, 7)
    m = ForwardModel(z_eff=_Z, loglr_rlf=grid, **_TINY)
    th = np.asarray(fid).copy()
    th[_IDX["agn_xi_rx"]] = 1.0
    th[_IDX["agn_xi_rm"]] = 0.0
    th[_IDX["agn_b_r"]] = 0.0
    th[_IDX["agn_sig_r"]] = 0.0
    th = jnp.asarray(th)
    rlf = np.asarray(m._rlf(m._theta_eff(th)))
    xlf = np.asarray(m._xlf_at(m._theta_eff(th), jnp.asarray(grid)))
    np.testing.assert_allclose(rlf, xlf, rtol=1e-10)


def test_rlf_amplitude_identity_and_plumbing(model, fid):
    from hod_mod.forecast.forward_jax import _IDX
    f = lambda t: model.predict(t, ["rlf"])["rlf"]
    d0 = np.asarray(f(fid))
    J = np.asarray(jax.jacfwd(f)(fid))
    assert np.all(d0 > 0) and np.all(np.isfinite(J))
    # Φ_R ∝ 10^ferdf exactly
    np.testing.assert_allclose(J[:, _IDX["agn_log10_ferdf"]],
                               np.log(10.0) * d0, rtol=1e-8)
    for n in ("agn_xi_rx", "agn_xi_rm", "agn_b_r", "agn_sig_r"):
        assert np.abs(J[:, _IDX[n]]).max() > 0, n


# ---------------------------------------------------------------- Phase E --

def test_hi_sector(model, fid):
    from hod_mod.forecast.forward_jax import _IDX
    # cl_gHI is linear in M0: dlnC/dlog10_M0 = ln10 exactly
    f = lambda t: model.predict(t, ["cl_gHI"])["cl_gHI"]
    d0 = np.asarray(f(fid))
    J = np.asarray(jax.jacfwd(f)(fid))
    np.testing.assert_allclose(J[:, _IDX["log10_M0_hi"]], np.log(10.0) * d0,
                               rtol=1e-8)
    # himf: finite, positive, responds to the slope and the cutoff mass
    g = lambda t: model.predict(t, ["himf"])["himf"]
    h0 = np.asarray(g(fid))
    Jh = np.asarray(jax.jacfwd(g)(fid))
    assert np.all(h0 > 0) and np.all(np.isfinite(Jh))
    for n in ("alpha_hi", "log10_Mmin_hi"):
        assert np.abs(Jh[:, _IDX[n]]).max() > 0, n
    # M_HI(M) is monotonic over the halo grid at the fiducial
    mhi = np.asarray(model._mhi(model._theta_eff(fid)))
    assert np.all(np.diff(mhi) > 0)


def test_tier2_split_sfq_assembly(fid, tmp_path):
    """split_sfq doubles the cell blocks; SF+Q abundances match the unsplit run."""
    from hod_mod.forecast.tier2 import Tier2Forecast
    kw = dict(z_edges=[0.2, 0.3], mstar_edges=[10.0, 10.4],
              n_bands=[(0.5, 2.0)], n_shear_bins=2,
              agn_lx_bins=[(42.0, 43.0)], agn_z_centers=(0.25,),
              n_k=32, n_m=32, n_gl=12, n_z=3,
              rp_wp=np.logspace(-1, 1.3, 4), rp_ds=np.logspace(-1, 1.2, 4),
              ell=np.logspace(1.0, 3.0, 4), rp_wp_agn=np.logspace(0.1, 1.3, 3))
    t0 = Tier2Forecast(split_sfq=False, **kw)
    t2 = Tier2Forecast(split_sfq=True, **kw)
    n_cells = lambda t: sum(1 for b in t.blocks if b.kind == "cell")
    assert n_cells(t2) == 2 * n_cells(t0)
    # the shell block must be UNSPLIT gas even in split mode
    shell = next(b for b in t2.blocks if b.kind == "shell")
    assert shell.model.sfq is None

    fidv = t0.fiducial()
    d0a, _, ma = t0.data_and_jacobian(fidv, cache_dir=str(tmp_path), verbose=False)
    d0s, _, ms = t2.data_and_jacobian(fidv, cache_dir=str(tmp_path), verbose=False)
    ng_a = d0a[(ma["kind"] == "cell") & (ma["obs"] == "n_gal")]
    ng_sf = d0s[(ms["obs"] == "n_gal")
                & np.char.endswith(ms["block"].astype(str), "_sf")]
    ng_q = d0s[(ms["obs"] == "n_gal")
               & np.char.endswith(ms["block"].astype(str), "_q")]
    np.testing.assert_allclose(ng_sf + ng_q, ng_a, rtol=1e-10)
    # noise assembly runs on the split configuration
    sig = t2.noise_sigma(fidv, d0s, ms, verbose=False)
    assert np.all(sig > 0)


def test_vector_and_sectors_extended():
    from hod_mod.forecast.forward_jax import PARAM_NAMES, N_PARAM, MISSING_PHYSICS
    from hod_mod.forecast.params import SECTORS
    assert N_PARAM == 77 and len(MISSING_PHYSICS) == 16
    flat = [n for sec in SECTORS.values() for n in sec]
    assert sorted(flat) == sorted(PARAM_NAMES)
