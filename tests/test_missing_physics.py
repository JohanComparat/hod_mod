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
    """(ξ_RX, ξ_RM, b_R, σ_R) = (1, 0, 0, 0), jets off, collapses onto L_X."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    grid = np.linspace(42.0, 45.0, 7)
    m = ForwardModel(z_eff=_Z, loglr_rlf=grid, **_TINY)
    th = np.asarray(fid).copy()
    th[_IDX["agn_xi_rx"]] = 1.0
    th[_IDX["agn_xi_rm"]] = 0.0
    th[_IDX["agn_b_r"]] = 0.0
    th[_IDX["agn_sig_r"]] = 0.0
    th[_IDX["f_loud0"]] = 0.0            # wave 3: remove the jet component
    th = jnp.asarray(th)
    rlf = np.asarray(m._rlf(m._theta_eff(th)))
    xlf = np.asarray(m._xlf_at(m._theta_eff(th), jnp.asarray(grid)))
    np.testing.assert_allclose(rlf, xlf, rtol=1e-10)


def test_rlf_amplitude_identity_and_plumbing(model, fid):
    from hod_mod.forecast.forward_jax import _IDX
    f = lambda t: model.predict(t, ["rlf"])["rlf"]
    # with jets off, Φ_R ∝ 10^ferdf exactly (the FP part is ERDF-tied)
    th0 = np.asarray(fid).copy()
    th0[_IDX["f_loud0"]] = 0.0
    th0 = jnp.asarray(th0)
    d0 = np.asarray(f(th0))
    J = np.asarray(jax.jacfwd(f)(th0))
    assert np.all(d0 > 0) and np.all(np.isfinite(J))
    np.testing.assert_allclose(J[:, _IDX["agn_log10_ferdf"]],
                               np.log(10.0) * d0, rtol=1e-8)
    for n in ("agn_xi_rx", "agn_xi_rm", "agn_b_r", "agn_sig_r"):
        assert np.abs(J[:, _IDX[n]]).max() > 0, n
    # wave 3: the jet population ADDs radio sources and is NOT ERDF-tied —
    # d ln Φ / d ferdf < ln10 once jets are on
    d1 = np.asarray(f(fid))
    assert np.all(d1 >= d0)
    J1 = np.asarray(jax.jacfwd(f)(fid))
    dln = J1[:, _IDX["agn_log10_ferdf"]] / d1
    assert dln.max() < np.log(10.0) + 1e-9 and dln.min() < np.log(10.0) - 1e-4
    for n in ("f_loud0", "beta_loud", "b_jet"):
        assert np.abs(J1[:, _IDX[n]]).max() > 0, n


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
    from hod_mod.forecast.forward_jax import (
        PARAM_NAMES, N_PARAM, MISSING_PHYSICS, TIER3_EXTENSION,
        WAVE4_MORPHOLOGY, TIER4_MORPHOLOGY)
    from hod_mod.forecast.params import SECTORS
    assert len(MISSING_PHYSICS) == 29 and MISSING_PHYSICS[0] == "eps_sn"
    assert N_PARAM == (90 + len(TIER3_EXTENSION) + len(WAVE4_MORPHOLOGY)
                       + len(TIER4_MORPHOLOGY))
    flat = [n for sec in SECTORS.values() for n in sec]
    assert sorted(flat) == sorted(PARAM_NAMES)


# ---------------------------------------------------------------- wave 3 --

def test_ssfr_cut_selection(fid):
    """sSFR-threshold samples: monotone, composable with the SF/Q split."""
    from hod_mod.forecast.forward_jax import ForwardModel
    kw = dict(z_eff=_Z, **_TINY)
    n_of = lambda m: float(m.predict(fid, ["n_gal"])["n_gal"][0])
    m_all = ForwardModel(**kw)
    m_cut = ForwardModel(ssfr_cut=-10.5, **kw)
    m_deep = ForwardModel(ssfr_cut=-30.0, **kw)
    assert n_of(m_cut) < n_of(m_all)
    np.testing.assert_allclose(n_of(m_deep), n_of(m_all), rtol=1e-10)
    # additivity survives the cut: sf-cut + q-cut == mixture-cut
    n_sf = n_of(ForwardModel(sfq="sf", ssfr_cut=-10.5, **kw))
    n_q = n_of(ForwardModel(sfq="q", ssfr_cut=-10.5, **kw))
    np.testing.assert_allclose(n_sf + n_q, n_of(m_cut), rtol=1e-10)
    # a star-forming cut keeps mostly SF galaxies
    assert n_sf > 10.0 * n_q


def test_sfrd_observable(fid):
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    kw = dict(z_eff=_Z, **_TINY)
    m_all = ForwardModel(**kw)
    v = lambda m: float(m.predict(fid, ["sfrd"])["sfrd"][0])
    assert v(m_all) > 0
    # SF + Q partition the SFR density exactly
    v_sf = v(ForwardModel(sfq="sf", **kw))
    v_q = v(ForwardModel(sfq="q", **kw))
    np.testing.assert_allclose(v_sf + v_q, v(m_all), rtol=1e-10)
    assert v_sf > v_q                     # SF galaxies dominate the SFRD
    # ρ_SFR ∝ 10^{sSFR_MS} exactly (both populations scale with μ_MS)
    f = lambda t: m_all.predict(t, ["sfrd"])["sfrd"]
    J = np.asarray(jax.jacfwd(f)(fid))
    d0 = np.asarray(f(fid))
    np.testing.assert_allclose(J[:, _IDX["ssfr_ms_norm"]], np.log(10.0) * d0,
                               rtol=1e-8)


def test_oiilf_observable(model, fid):
    from hod_mod.forecast.forward_jax import _IDX
    f = lambda t: model.predict(t, ["oiilf"])["oiilf"]
    d0 = np.asarray(f(fid))
    J = np.asarray(jax.jacfwd(f)(fid))
    assert np.all(d0 > 0) and np.all(np.isfinite(J))
    # loii_norm and ssfr_ms_norm shift the same abscissa: identical columns
    np.testing.assert_allclose(J[:, _IDX["loii_norm"]],
                               J[:, _IDX["ssfr_ms_norm"]], rtol=1e-8)
    # quenching removes [OII] emitters
    assert np.abs(J[:, _IDX["log10_Mq_cen"]]).max() > 0


def test_ilf_observable(model, fid):
    from hod_mod.forecast.forward_jax import _IDX
    f = lambda t: model.predict(t, ["ilf"])["ilf"]
    d0 = np.asarray(f(fid))
    J = np.asarray(jax.jacfwd(f)(fid))
    assert np.all(d0 > 0) and np.all(np.isfinite(J))
    # ERDF-tied: Φ_IR ∝ 10^ferdf exactly
    np.testing.assert_allclose(J[:, _IDX["agn_log10_ferdf"]],
                               np.log(10.0) * d0, rtol=1e-8)
    # the bolometric correction moves it; the obscured fraction does NOT
    # (IR is obscuration-robust by construction — the cross-band test)
    assert np.abs(J[:, _IDX["agn_bc_ir"]]).max() > 0
    assert np.abs(J[:, _IDX["agn_fabs"]]).max() == 0.0


# ---------------------------------------------------------------- wave 2 --

def test_wind_loading(model, fid):
    """η_w0 = 0 is exactly the tier-2 sigmoid; winds puff low-mass gas most."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX, _RHO_CRIT0
    import jax.numpy as jnp
    th = model._theta_eff(fid)
    _, eta0 = model._fb_eta(th)
    # manual tier-2 sigmoid (no wind term)
    eta_min = 10.0 ** float(th[_IDX["log10_eta_min"]])
    m_eta = 10.0 ** float(th[_IDX["log10_M_eta"]])
    beta = float(th[_IDX["beta_eta"]])
    sig = 1.0 - (1.0 - eta_min) / (1.0 + (np.asarray(model.m) / m_eta) ** beta)
    np.testing.assert_allclose(np.asarray(eta0), sig, rtol=1e-12)
    # switch winds on: eta drops, more at low mass (low V_c)
    tw = np.asarray(fid).copy()
    tw[_IDX["eta_w_norm"]] = 0.5
    _, eta_w = model._fb_eta(model._theta_eff(jnp.asarray(tw)))
    ratio = np.asarray(eta_w) / np.asarray(eta0)
    assert np.all(ratio < 1.0) and ratio[0] < ratio[-1]
    # the coupling is live at the fiducial (∂η/∂η_w0 ≠ 0)
    g = jax.jacfwd(lambda t: model._fb_eta(model._theta_eff(t))[1])(fid)
    assert np.abs(np.asarray(g)[:, _IDX["eta_w_norm"]]).max() > 0


def test_ssfr_observable(model, fid):
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    v = float(model.predict(fid, ["ssfr"])["ssfr"][0])
    norm = float(fid[_IDX["ssfr_ms_norm"]])
    slope = float(fid[_IDX["ssfr_ms_slope"]])
    mstar_c = model._thr + 0.25
    np.testing.assert_allclose(v, norm + slope * (mstar_c - 10.5), rtol=1e-12)
    # evolution slope acts through _theta_eff with the standard lever arm
    f = lambda t: model.predict(t, ["ssfr"])["ssfr"]
    J = np.asarray(jax.jacfwd(f)(fid))
    np.testing.assert_allclose(J[0, _IDX["ssfr_ms_zs"]], model._x_evol,
                               rtol=1e-12)


def test_dhi_quenched_only_touches_quenched_hi(fid):
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    kw = dict(z_eff=_Z, **_TINY)
    for sfq, active in ((None, False), ("sf", False), ("q", True)):
        m = ForwardModel(sfq=sfq, **kw)
        f = lambda t: m.predict(t, ["cl_gHI"])["cl_gHI"]
        col = np.asarray(jax.jacfwd(f)(fid))[:, _IDX["dhi_quenched"]]
        assert (np.abs(col).max() > 0) == active, sfq


def test_camb_ratio_correction(fid):
    """The linearized CAMB ratio: bounded, differentiable, and it moves the
    P(k) shape derivatives the EH98 form gets slightly wrong."""
    from hod_mod.forecast.pk_camb_ratio import load
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    tab = load()
    assert np.abs(np.asarray(tab["lnr0"])).max() < 0.1        # EH98 is ~4% off
    assert set(tab["names"]) == {"h", "Omega_b", "Omega_m", "n_s", "sum_mnu"}

    kw = dict(z_eff=_Z, **_TINY)
    m0 = ForwardModel(pk_correction="none", **kw)
    mc = ForwardModel(pk_correction="camb_linear", **kw)
    wp0 = np.asarray(m0.predict(fid, ["wp"])["wp"])
    wpc = np.asarray(mc.predict(fid, ["wp"])["wp"])
    assert np.all(np.isfinite(wpc))
    dev = np.abs(wpc / wp0 - 1.0)
    # the largest rp bin (10^1.7 ~ 50 Mpc/h) sits in the BAO regime where EH98 is
    # ~4% off in P(k); projected, the correction there reaches ~0.10
    assert 0.0 < dev.max() < 0.12          # a small, nonzero shape correction
    # the correction changes the n_s response (the derivative row is live)
    for m, tag in ((m0, "none"), (mc, "camb")):
        J = np.asarray(jax.jacfwd(lambda t: m.predict(t, ["wp"])["wp"])(fid))
        assert np.all(np.isfinite(J)), tag
    Jn = np.asarray(jax.jacfwd(lambda t: mc.predict(t, ["wp"])["wp"])(fid))
    J0 = np.asarray(jax.jacfwd(lambda t: m0.predict(t, ["wp"])["wp"])(fid))
    i = _IDX["n_s"]
    assert np.abs(Jn[:, i] - J0[:, i]).max() > 0


def test_tier2_wave2_observables(fid, tmp_path):
    """Radio LF, HIMF, 21cm cross and sSFR enter the tier-2 assembly + noise."""
    from hod_mod.forecast.tier2 import Tier2Forecast
    t2 = Tier2Forecast(
        z_edges=[0.2, 0.3], mstar_edges=[10.0, 10.4],
        n_bands=[(0.5, 2.0)], n_shear_bins=2,
        agn_lx_bins=[(42.0, 43.0)], agn_z_centers=(0.25,),
        include_radio=True, include_hi=True, include_ssfr=True,
        include_ir=True,
        n_k=32, n_m=32, n_gl=12, n_z=3,
        rp_wp=np.logspace(-1, 1.3, 4), rp_ds=np.logspace(-1, 1.2, 4),
        ell=np.logspace(1.0, 3.0, 4), rp_wp_agn=np.logspace(0.1, 1.3, 3))
    cell = next(b for b in t2.blocks if b.kind == "cell")
    shell = next(b for b in t2.blocks if b.label.endswith("_shell"))
    assert "cl_gHI" in cell.which and "ssfr" in cell.which
    assert "sfrd" in cell.which
    assert "rlf" in shell.which and "himf" not in shell.which
    assert "oiilf" in shell.which and "ilf" in shell.which
    hi_local = next(b for b in t2.blocks if b.label == "hi_local")
    assert hi_local.which == ("himf",) and hi_local.z_hi <= 0.1
    fidv = t2.fiducial()
    d0, J, meta = t2.data_and_jacobian(fidv, cache_dir=str(tmp_path),
                                       verbose=False)
    assert np.all(np.isfinite(d0)) and np.all(np.isfinite(J))
    sig = t2.noise_sigma(fidv, d0, meta, verbose=False)
    for o in ("rlf", "himf", "cl_gHI", "ssfr", "sfrd", "oiilf", "ilf"):
        s = sig[meta["obs"] == o]
        assert s.size > 0 and np.all(s > 0), o
        # at least part of each new observable carries finite noise
        assert np.isfinite(s).any(), o


# ---------------------------------------------------------------- wave 4 --

def test_wave4_layout():
    from hod_mod.forecast.forward_jax import (
        PARAM_NAMES, N_PARAM, TIER3_EXTENSION, WAVE4_MORPHOLOGY,
        TIER4_MORPHOLOGY)
    from hod_mod.forecast.params import SECTORS
    assert N_PARAM == 106 + len(TIER4_MORPHOLOGY)
    assert list(TIER3_EXTENSION) == list(PARAM_NAMES[90:102])
    assert list(WAVE4_MORPHOLOGY) == list(PARAM_NAMES[102:106])
    assert WAVE4_MORPHOLOGY == ["log10_M_morph", "beta_morph",
                                "f_morph_sat", "mbh_bt_slope"]
    assert set(SECTORS["morphology"]) == set(WAVE4_MORPHOLOGY
                                             + TIER4_MORPHOLOGY)


def test_wave4_f_early_weibull():
    """f_early_cen: monotone in M_h, 0/1 limits; satellite boost keeps [0,1]
    and reduces to the central fraction at f_morph_sat = 0."""
    from hod_mod.connection.morphology import f_early_cen, f_early_sat
    lm = jnp.linspace(10.0, 16.0, 60)
    fc = np.asarray(f_early_cen(lm, 12.5, 0.8))
    d = np.diff(fc)
    assert np.all(d >= 0)                    # saturates to exactly 1 (underflow)
    assert np.all(d[np.asarray(lm)[:-1] < 13.5] > 0)   # strict in the transition
    assert fc[0] < 1e-2 and fc[-1] > 1 - 1e-6
    fs = np.asarray(f_early_sat(lm, 12.5, 0.8, 0.2))
    assert np.all(fs >= fc) and np.all(fs <= 1.0)
    np.testing.assert_allclose(np.asarray(f_early_sat(lm, 12.5, 0.8, 0.0)),
                               fc, rtol=1e-12)


def test_wave4_early_late_additivity(fid):
    """EARLY + LATE ≡ unsplit (n_gal), and the 4-way SF/Q × early/late
    partition sums exactly to the unsplit sample."""
    from hod_mod.forecast.forward_jax import ForwardModel
    kw = dict(z_eff=_Z, **_TINY)
    n_of = lambda m: float(m.predict(fid, ["n_gal"])["n_gal"][0])
    n_all = n_of(ForwardModel(**kw))
    n_e = n_of(ForwardModel(morph="early", **kw))
    n_l = n_of(ForwardModel(morph="late", **kw))
    np.testing.assert_allclose(n_e + n_l, n_all, rtol=1e-12)
    tot = sum(n_of(ForwardModel(sfq=sv, morph=mo, **kw))
              for sv in ("sf", "q") for mo in ("early", "late"))
    np.testing.assert_allclose(tot, n_all, rtol=1e-12)


def test_wave4_f_early_observable(fid):
    """Per-cell early-type fraction: bounded, mass-trend positive, and the
    expected parameter responses (more massive pivot -> later types)."""
    import jax
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    kw = dict(z_eff=_Z, **_TINY)
    m_lo = ForwardModel(log10m_star_bin=(10.0, 10.4), **kw)
    m_hi = ForwardModel(log10m_star_bin=(11.0, 11.4), **kw)
    f_lo = float(m_lo.predict(fid, ["f_early"])["f_early"][0])
    f_hi = float(m_hi.predict(fid, ["f_early"])["f_early"][0])
    assert 0.0 < f_lo < f_hi < 1.0
    g = jax.jacfwd(lambda t: m_lo.predict(t, ["f_early"])["f_early"])(fid)
    assert float(g[0, _IDX["log10_M_morph"]]) < 0
    assert float(g[0, _IDX["f_morph_sat"]]) > 0
    assert float(g[0, _IDX["mbh_bt_slope"]]) == 0.0    # AGN-only parameter


def test_wave4_bh_bulge_coupling(fid):
    """mbh_bt_slope = 0 leaves the Powell chain exactly (zero morphology
    response of the XLF at the fiducial); off-fiducial the coupling routes
    log10_M_morph into the XLF — morphology testable through the AGN sector."""
    import jax
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    m = ForwardModel(z_eff=_Z, **_TINY)
    g0 = jax.jacfwd(lambda t: m.predict(t, ["xlf"])["xlf"])(fid)
    assert np.abs(np.asarray(g0)[:, _IDX["mbh_bt_slope"]]).max() > 0
    assert np.allclose(np.asarray(g0)[:, _IDX["log10_M_morph"]], 0.0)
    th = fid.at[_IDX["mbh_bt_slope"]].set(0.3)
    g1 = jax.jacfwd(lambda t: m.predict(t, ["xlf"])["xlf"])(th)
    assert np.abs(np.asarray(g1)[:, _IDX["log10_M_morph"]]).max() > 0


def test_wave4_tier2_integration(fid, tmp_path):
    """include_morph adds the f_early datum per cell with finite binomial +
    floor noise; the tier-2 assembly runs end-to-end."""
    from hod_mod.forecast.tier2 import Tier2Forecast
    t2 = Tier2Forecast(
        z_edges=[0.2, 0.3], mstar_edges=[10.0, 10.4],
        n_bands=[(0.5, 2.0)], n_shear_bins=2,
        agn_lx_bins=[(42.0, 43.0)], agn_z_centers=(0.25,),
        include_morph=True,
        n_k=32, n_m=32, n_gl=12, n_z=3,
        rp_wp=np.logspace(-1, 1.3, 4), rp_ds=np.logspace(-1, 1.2, 4),
        ell=np.logspace(1.0, 3.0, 4), rp_wp_agn=np.logspace(0.1, 1.3, 3))
    cell = next(b for b in t2.blocks if b.kind == "cell")
    assert "f_early" in cell.which
    fidv = t2.fiducial()
    d0, J, meta = t2.data_and_jacobian(fidv, cache_dir=str(tmp_path),
                                       verbose=False)
    assert np.all(np.isfinite(d0)) and np.all(np.isfinite(J))
    sig = t2.noise_sigma(fidv, d0, meta, verbose=False)
    s = meta["obs"] == "f_early"
    assert s.any() and np.all(np.isfinite(sig[s])) and np.all(sig[s] > 0)
    # the calibration floor dominates for a big cell
    assert np.all(sig[s] >= t2.spectro.fmorph_err)
