"""Tier-3 forecast extension: multi-wavelength maps, band LFs, extras.

Phase gates (docs/missing_physics_implementation.rst conventions):

A: 102-parameter layout, frozen slices, _pk_tracer_field bit-identity,
   the log10m_min mass-grid floor
B: radio/IR intensity-map fields + AGN crosses + tSZ/21cm autos + cluster
   counts + AGN lensing — exact chain-rule / band-scaling identities
C: galaxy band LFs (UV/opt/NIR/Halpha) + AGN UV/opt LFs — clone, shift and
   mixture-additivity identities
"""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_TINY = dict(n_k=48, n_m=64, n_gl=24, n_z=5)
_Z = 0.5


@pytest.fixture(scope="module")
def fid():
    from hod_mod.forecast import params
    return jnp.asarray(params.fiducial_vector())


@pytest.fixture(scope="module")
def model(fid):
    from hod_mod.forecast.forward_jax import ForwardModel
    return ForwardModel(z_eff=_Z, radio_map_bands=(0.95, 1.4, 3.0),
                        ir_map_bands=(3.4, 4.9, 12.0),
                        agn_lx_bins=[(42.0, 43.0)], logl_ncl=44.0,
                        agn_emission="powell", **_TINY)


# ------------------------------------------------------------- phase A --

def test_vector_layout_tier3():
    from hod_mod.forecast.forward_jax import (
        PARAM_NAMES, N_PARAM, MISSING_PHYSICS, TIER2_EXTENSION,
        TIER3_EXTENSION)
    from hod_mod.forecast.params import SECTORS
    from hod_mod.forecast.forward_jax import (WAVE4_MORPHOLOGY,
                                                TIER4_MORPHOLOGY)
    assert N_PARAM == 102 + len(WAVE4_MORPHOLOGY) + len(TIER4_MORPHOLOGY)
    assert list(MISSING_PHYSICS) == list(PARAM_NAMES[61:90])
    assert list(TIER2_EXTENSION) == list(PARAM_NAMES[31:90])
    assert list(TIER3_EXTENSION) == list(PARAM_NAMES[90:102])
    assert TIER3_EXTENSION[0] == "l14_sfr" and len(TIER3_EXTENSION) == 12
    flat = [n for sec in SECTORS.values() for n in sec]
    assert sorted(flat) == sorted(PARAM_NAMES)


def test_tier3_fiducials_priors_latex_complete():
    from hod_mod.forecast.forward_jax import TIER3_EXTENSION
    from hod_mod.forecast import params
    for n in TIER3_EXTENSION:
        assert n in params._FIDUCIAL_DEFAULT
        assert n in params.BROAD_PRIOR_SIGMA
        assert n in params.PARAM_LATEX


def test_pk_tracer_field_bit_identity(fid):
    """_pk_gX_of must delegate to _pk_tracer_field bit-identically, at an
    off-fiducial theta."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    m = ForwardModel(z_eff=_Z, **_TINY)
    th = fid.at[_IDX["lx_norm"]].add(0.3).at[_IDX["Omega_m"]].add(0.02)
    H = m._halo_common(th, m.z_eff)
    X = m._emissivity_uk(th, H)
    Xa = m._agn_point_source(th)
    new = m._pk_gX_of(th, H, X, Xa)
    gal = H["nc"][None, :] + H["ns"][None, :] * H["uk"]
    P1 = jnp.trapezoid(H["dndm"][None, :] * gal * X, m.m, axis=1) / H["n_gal"]
    IX = jnp.trapezoid(H["dndm"][None, :] * H["bias"][None, :] * X, m.m, axis=1)
    Pa = (jnp.trapezoid(H["dndm"][None, :] * gal * Xa[None, :], m.m, axis=1)
          / H["n_gal"]
          + H["b_eff"] * H["pk_lin"]
          * jnp.trapezoid(H["dndm"][None, :] * H["bias"][None, :] * Xa[None, :],
                          m.m, axis=1))
    old = P1 + H["b_eff"] * H["pk_lin"] * IX + Pa
    assert np.array_equal(np.asarray(new), np.asarray(old))


def test_log10m_min_default_bit_identical():
    from hod_mod.forecast.forward_jax import ForwardModel
    m0 = ForwardModel(z_eff=_Z, **_TINY)
    m1 = ForwardModel(z_eff=_Z, log10m_min=10.0, **_TINY)
    assert np.array_equal(np.asarray(m0.m), np.asarray(m1.m))
    with pytest.raises(ValueError):
        ForwardModel(z_eff=_Z, log10m_min=8.0, **_TINY)


def test_low_mass_cell_grid(fid):
    """A (9.0, 9.2) M* cell needs the 8.5 floor: occupations must not be
    truncated (edge occupation tiny vs peak) and n_gal must converge in n_m."""
    from hod_mod.forecast.forward_jax import ForwardModel
    kw = dict(z_eff=0.3, log10m_star_bin=(9.0, 9.2), log10m_min=8.5,
              n_k=48, n_gl=24, n_z=5)
    m = ForwardModel(n_m=256, **kw)
    nc, ns = m._occ_sample(fid)
    tot = np.asarray(nc + ns)
    assert tot[0] < 1e-6 * tot.max()
    n1 = float(m.predict(fid, ["n_gal"])["n_gal"][0])
    n2 = float(ForwardModel(n_m=512, **kw).predict(fid, ["n_gal"])["n_gal"][0])
    assert n1 > 0 and abs(n1 / n2 - 1.0) < 1e-3


# ------------------------------------------------------------- phase B --

def test_radio_band_scaling(model, fid):
    """SF-only (cores+jets off): cl_gR bands scale exactly as
    (nu/1.4)^(1-alpha_syn), the auto as its square, and the l14_sfr
    log-derivatives are ln10 (cross) and 2 ln10 (auto)."""
    from hod_mod.forecast.forward_jax import _IDX
    th = fid.at[_IDX["f_loud0"]].set(0.0) \
            .at[_IDX["agn_log10_ferdf"]].set(-99.0)
    a = 1.0 - float(fid[_IDX["alpha_syn"]])
    cl = np.asarray(model.predict(th, ["cl_gR"])["cl_gR"]).reshape(3, -1)
    for i, nu in enumerate((0.95, 1.4, 3.0)):
        np.testing.assert_allclose(cl[i] / cl[1], (nu / 1.4) ** a, rtol=1e-8)
    clRR = np.asarray(model.predict(th, ["cl_RR"])["cl_RR"]).reshape(3, -1)
    np.testing.assert_allclose(clRR[0] / clRR[1], (0.95 / 1.4) ** (2 * a),
                               rtol=1e-8)
    f = lambda t: jnp.log(jnp.concatenate(
        [model.predict(t, ["cl_gR"])["cl_gR"],
         model.predict(t, ["cl_RR"])["cl_RR"]]))
    g = jax.jacfwd(f)(th)[:, _IDX["l14_sfr"]]
    n = cl.size
    np.testing.assert_allclose(np.asarray(g[:n]), np.log(10.0), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(g[n:]), 2 * np.log(10.0), rtol=1e-6)


def test_alpha_syn_anchor(model, fid):
    """alpha_syn is exactly inert at the 1.4 GHz anchor band (SF-only) and
    active in the others."""
    from hod_mod.forecast.forward_jax import _IDX
    th = fid.at[_IDX["f_loud0"]].set(0.0) \
            .at[_IDX["agn_log10_ferdf"]].set(-99.0)
    g = jax.jacfwd(lambda t: model.predict(t, ["cl_gR"])["cl_gR"])(th)
    g = np.asarray(g[:, _IDX["alpha_syn"]]).reshape(3, -1)
    assert np.allclose(g[1], 0.0, atol=1e-12)
    assert np.all(np.abs(g[0]) > 0) and np.all(np.abs(g[2]) > 0)


def test_ir_field_sector_responses(model, fid):
    """cl_gI responds to lir_sfr (dust), ml_nir (stellar) and agn_bc_ir
    (torus); bir_color is exactly inert at the 4.9 um anchor band."""
    from hod_mod.forecast.forward_jax import _IDX
    g = jax.jacfwd(lambda t: model.predict(t, ["cl_gI"])["cl_gI"])(fid)
    g = np.asarray(g)
    for p in ("lir_sfr", "ml_nir", "agn_bc_ir"):
        assert np.abs(g[:, _IDX[p]]).max() > 0, p
    gb = g[:, _IDX["bir_color"]].reshape(3, -1)
    assert np.allclose(gb[1], 0.0, atol=1e-12)      # 4.9 um anchor
    assert np.abs(gb[0]).max() > 0 and np.abs(gb[2]).max() > 0


def test_agn_cross_spectra(model, fid):
    """cl_aR/cl_aI/cl_ag: finite, positive, and responsive to both the AGN
    occupation (agn_log10_ferdf) and the field calibrations."""
    from hod_mod.forecast.forward_jax import _IDX
    names = ["cl_aR", "cl_aI", "cl_ag"]
    out = model.predict(fid, names)
    for n in names:
        v = np.asarray(out[n])
        assert v.shape == np.asarray(model.grid_of(n)).shape
        assert np.all(np.isfinite(v)) and np.all(v > 0)
    g = jax.jacfwd(lambda t: model.predict(t, ["cl_ag"])["cl_ag"])(fid)
    assert np.abs(np.asarray(g)[:, _IDX["agn_log10_ferdf"]]).max() > 0


def test_cl_yy_p0_identity(model, fid):
    """d ln C_yy / d p03_pressure = 2/P_0.3 exactly (the amplitude enters the
    pressure squared)."""
    from hod_mod.forecast.forward_jax import _IDX
    g = jax.jacfwd(lambda t: jnp.log(model.predict(t, ["cl_yy"])["cl_yy"]))(fid)
    np.testing.assert_allclose(np.asarray(g[:, _IDX["p03_pressure"]]),
                               2.0 / float(fid[_IDX["p03_pressure"]]), rtol=1e-6)


def test_cl_hihi_m0_identity(model, fid):
    """d ln C_HIHI / d log10_M0_hi = 2 ln10 exactly (M_HI ∝ M0 squared)."""
    from hod_mod.forecast.forward_jax import _IDX
    g = jax.jacfwd(lambda t: jnp.log(model.predict(t, ["cl_HIHI"])["cl_HIHI"]))(fid)
    np.testing.assert_allclose(np.asarray(g[:, _IDX["log10_M0_hi"]]),
                               2 * np.log(10.0), rtol=1e-6)


def test_ncl_responses(fid):
    """Cluster counts respond to the L_X–M relation AND the cosmology
    (through dn/dM) — the classic cluster-count degeneracy pairing."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    m = ForwardModel(z_eff=_Z, logl_ncl=44.0, **_TINY)
    n0 = float(m.predict(fid, ["ncl"])["ncl"][0])
    assert 1e-9 < n0 < 1e-3
    g = jax.jacfwd(lambda t: jnp.log(m.predict(t, ["ncl"])["ncl"]))(fid)
    assert abs(float(g[0, _IDX["lx_norm"]])) > 0.1
    assert abs(float(g[0, _IDX["Omega_m"]])) > 0.1


def test_ds_agn(model, fid):
    """AGN galaxy–galaxy lensing: finite, positive at small R, responsive to
    agn_rho (the L_X-selection ↔ halo-mass link it is designed to break)."""
    from hod_mod.forecast.forward_jax import _IDX
    v = np.asarray(model.predict(fid, ["ds_agn"])["ds_agn"])
    assert v.shape == np.asarray(model.grid_of("ds_agn")).shape
    assert np.all(np.isfinite(v)) and v[0] > 0
    g = jax.jacfwd(lambda t: model.predict(t, ["ds_agn"])["ds_agn"])(fid)
    assert np.abs(np.asarray(g)[:, _IDX["agn_rho"]]).max() > 0


def test_hi_field_shared(model, fid):
    """cl_gHI and cl_HIHI share _hi_field: the cross keeps its C ∝ M0
    identity through the refactor."""
    from hod_mod.forecast.forward_jax import _IDX
    g = jax.jacfwd(lambda t: jnp.log(model.predict(t, ["cl_gHI"])["cl_gHI"]))(fid)
    np.testing.assert_allclose(np.asarray(g[:, _IDX["log10_M0_hi"]]),
                               np.log(10.0), rtol=1e-6)


# ------------------------------------------------------------- phase C --

def test_half_is_oiilf_clone(fid):
    """half(lha_norm = loii_norm) == oiilf on matched grids, exactly."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    m = ForwardModel(z_eff=_Z, loglha_half=np.linspace(40.5, 43.5, 7), **_TINY)
    th = fid.at[_IDX["lha_norm"]].set(float(fid[_IDX["loii_norm"]]))
    np.testing.assert_allclose(np.asarray(m.predict(th, ["half"])["half"]),
                               np.asarray(m.predict(th, ["oiilf"])["oiilf"]),
                               rtol=1e-12)


def test_qlf_uv_is_ilf_at_matched_bc(fid):
    """qlf_uv(bc_uv = bc_ir, f_abs = 0) == ilf on matched grids; and
    d qlf/d f_abs = −qlf/(1−f_abs) exactly (type-1 visibility)."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    m = ForwardModel(z_eff=_Z, logluv_qlf=np.linspace(42.5, 46.0, 7), **_TINY)
    th = fid.at[_IDX["agn_bc_uv"]].set(float(fid[_IDX["agn_bc_ir"]])) \
            .at[_IDX["agn_fabs"]].set(0.0)
    q = np.asarray(m.predict(th, ["qlf_uv"])["qlf_uv"])
    np.testing.assert_allclose(q, np.asarray(m.predict(th, ["ilf"])["ilf"]),
                               rtol=1e-12)
    g = jax.jacfwd(lambda t: m.predict(t, ["qlf_uv"])["qlf_uv"])(th)
    np.testing.assert_allclose(np.asarray(g[:, _IDX["agn_fabs"]]), -q, rtol=1e-8)


def test_optlf_mixture_additivity(model, fid):
    """At dopt_q = 0 the SF/Q optical mixture collapses exactly to a single
    all-central lognormal."""
    from hod_mod.forecast.forward_jax import _IDX, _SIG_OPT
    th = fid.at[_IDX["dopt_q"]].set(0.0)
    lf_mix = np.asarray(model.predict(th, ["optlf"])["optlf"])
    te = model._theta_eff(th)
    _, _, _, log10ms, sig_star = model._sfr_moments(te)
    sig = jnp.sqrt(sig_star ** 2 + _SIG_OPT ** 2)
    lf_one = np.asarray(model._lf_lognormal(
        te, model.loglopt_optlf, th[_IDX["ml_opt"]] + log10ms, sig,
        jnp.ones_like(log10ms)))
    np.testing.assert_allclose(lf_mix, lf_one, rtol=1e-12)


def test_nirlf_shift_invariance(fid):
    """nirlf on grid+delta with ml_nir+delta equals nirlf on the base grid
    at the base ml_nir (pure shift kernel)."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    d = 0.17
    base = np.linspace(42.0, 45.0, 7)
    m0 = ForwardModel(z_eff=_Z, loglnir_nirlf=base, **_TINY)
    m1 = ForwardModel(z_eff=_Z, loglnir_nirlf=base + d, **_TINY)
    np.testing.assert_allclose(
        np.asarray(m1.predict(fid.at[_IDX["ml_nir"]].add(d), ["nirlf"])["nirlf"]),
        np.asarray(m0.predict(fid, ["nirlf"])["nirlf"]), rtol=1e-12)


def test_uvlf_attenuation_slope(model, fid):
    """tau_uv_mslope tilts the UV LF: inert only if the sample were a delta
    at 10^10.5 — on the full HMF it must reshape the counts (bright vs faint
    derivative signs differ)."""
    from hod_mod.forecast.forward_jax import _IDX
    g = jax.jacfwd(lambda t: model.predict(t, ["uvlf"])["uvlf"])(fid)
    g = np.asarray(g[:, _IDX["tau_uv_mslope"]])
    assert np.abs(g).max() > 0
    assert np.sign(g[0]) != np.sign(g[-1])


def test_band_lf_grids_and_positivity(model, fid):
    names = ["uvlf", "optlf", "nirlf", "half", "qlf_uv", "qlf_opt"]
    out = model.predict(fid, names)
    for n in names:
        v = np.asarray(out[n])
        assert v.shape == np.asarray(model.grid_of(n)).shape
        assert np.all(np.isfinite(v)) and np.all(v > 0), n


def test_scale_cut_ds_agn_and_abundance(model):
    """ds_agn rows obey the projected r_p cut; ncl/band-LF rows are kept."""
    which = ["ds_agn", "ncl", "uvlf"]
    f, row_obs, row_x = model.full_data_vector_fn(which)
    keep = model.scale_cut_mask(row_obs, row_x, rmin=1.0)
    ds = row_obs == "ds_agn"
    assert np.all(keep[ds] == (row_x[ds] > 1.0))
    assert np.all(keep[~ds])


def test_cl_aa_fiducial(model, fid):
    cl, n_agn = model.cl_aa_fiducial(fid, 42.0, 43.0)
    assert np.all(np.isfinite(np.asarray(cl))) and np.all(np.asarray(cl) > 0)
    assert 0 < n_agn < 1e-2


# ------------------------------------------------------------ assembly --

def _tiny_tier3(**over):
    from hod_mod.forecast.tier3 import Tier3Forecast
    kw = dict(z_edges=[0.2, 0.4], mstar_edges=[9.4, 10.6],
              agn_z_centers=(0.3,), agn_lx_bins=[(42.0, 43.0)],
              n_bands=1, n_shear_bins=2, n_k=32, n_m=48, n_gl=12, n_z=3,
              cell_n_m=64,
              rp_wp=np.logspace(-1, 1.4, 5), rp_ds=np.logspace(-1, 1.2, 4),
              ell=np.logspace(1.0, 3.3, 5), rp_wp_agn=np.logspace(0.1, 1.4, 3))
    kw.update(over)
    return Tier3Forecast(**kw)


def test_tier3_assembly_smoke(tmp_path):
    """Tiny end-to-end: finite d0/J, positive noise, every tier-3 family
    present, cache round-trip identical."""
    t3 = _tiny_tier3()
    fid = t3.fiducial()
    cache = str(tmp_path / "cache")
    d0, J, meta = t3.data_and_jacobian(fid, cache_dir=cache, verbose=False)
    assert np.all(np.isfinite(d0)) and np.all(np.isfinite(J))
    sig = t3.noise_sigma(fid, d0, meta, verbose=False)
    assert np.all(sig[np.isfinite(sig)] > 0)
    obs = set(meta["obs"])
    for o in ("cl_gR", "cl_gI", "cl_RR", "cl_II", "cl_aR", "cl_aI", "cl_ag",
              "cl_yy", "cl_HIHI", "ncl", "ds_agn", "uvlf", "optlf", "nirlf",
              "half", "qlf_uv", "qlf_opt", "sfrd"):
        assert o in obs, o
    d0b, Jb, _ = t3.data_and_jacobian(fid, cache_dir=cache, verbose=False)
    assert np.array_equal(d0, d0b) and np.array_equal(J, Jb)


def test_tier3_two_tier_completeness():
    """Cells fall back wide → deep → skipped by the M*_lim(z) model."""
    from hod_mod.forecast import noise
    t3 = _tiny_tier3(z_edges=[0.2, 0.4, 0.8], mstar_edges=[9.2, 9.4])
    # z_hi=0.4: wide lim = 9.0+0.4 > 9.2 -> deep; z_hi=0.8: lim 9.8 -> deep
    cells = [b for b in t3.blocks if b.kind == "cell"]
    assert all(b.spectro is t3.spectro_deep for b in cells)
    assert not t3.skipped_cells
    # a deep tier that cannot reach 9.2 either -> the cells are skipped
    t3b = _tiny_tier3(z_edges=[0.2, 0.4, 0.8], mstar_edges=[9.2, 9.4],
                      spectro_deep=noise.SpectroSurvey(
                          f_sky=0.004, mstar_lim0=9.3, mstar_lim_slope=0.0))
    assert len(t3b.skipped_cells) == 2
    assert not any(b.kind == "cell" for b in t3b.blocks)


_PAR_EQ_SER = r"""
import os, sys, tempfile
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from hod_mod.forecast.tier3 import Tier3Forecast

t3 = Tier3Forecast(z_edges=[0.2, 0.4], mstar_edges=[9.4, 10.6],
                   agn_z_centers=(0.3,), agn_lx_bins=[(42.0, 43.0)],
                   n_bands=1, n_shear_bins=2, n_k=32, n_m=48, n_gl=12,
                   n_z=3, cell_n_m=64,
                   rp_wp=np.logspace(-1, 1.4, 5),
                   rp_ds=np.logspace(-1, 1.2, 4),
                   ell=np.logspace(1.0, 3.3, 5),
                   rp_wp_agn=np.logspace(0.1, 1.4, 3))
fid = np.asarray(t3.fiducial())
with tempfile.TemporaryDirectory() as td:
    cs, cp = os.path.join(td, "s"), os.path.join(td, "p")
    d0s, Js, _ = t3.data_and_jacobian(fid, cache_dir=cs, verbose=False)
    missing = t3.precompute_blocks(fid, cp, jobs=2, verbose=False)
    assert len(missing) == len(t3.blocks), (len(missing), len(t3.blocks))
    d0p, Jp, _ = t3.data_and_jacobian(fid, cache_dir=cp, verbose=False)
    assert np.array_equal(d0s, d0p), "d0 parallel != serial"
    assert np.array_equal(Js, Jp), "J parallel != serial"
print("PARALLEL==SERIAL OK")
"""


def test_tier3_parallel_equals_serial():
    """precompute_blocks(jobs=2) fills a cache whose assembled (d0, J) is
    BYTE-identical to the serial computation.  Runs in a fresh interpreter:
    the invariant requires a uniformly-x64 parent (the production driver
    pattern), while the pytest parent carries float32 module constants from
    conftest imports that predate the x64 switch."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", _PAR_EQ_SER],
                       capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, JAX_PLATFORMS="cpu"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PARALLEL==SERIAL OK" in r.stdout
