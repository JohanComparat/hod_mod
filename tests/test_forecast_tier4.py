"""Tier-4 morphology observables: layout, exact identities, assembly.

Gates (the missing-physics conventions):

- 111-parameter layout, frozen WAVE4_MORPHOLOGY / TIER4_MORPHOLOGY slices;
- joint early∩quenched fractions: product identity at rho = 0, 4-way
  partition additivity at any rho, conditional f_early ordering;
- size: exact unit response to log10_f_size + the _Z_EVOL chain rule;
- w_g+: exact 1/a_ia log-derivative + strict proportionality to the cell's
  early-type fraction;
- f_early_agn: the BH-bulge coupling direction;
- Tier4Forecast end-to-end smoke + parallel == serial (fresh interpreter).
"""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ForwardModel builds the default CosmoPower P(k) emulator unconditionally;
# without the package these tests would error at fixture time, not skip.
pytest.importorskip("cosmopower_jax")

_TINY = dict(n_k=48, n_m=64, n_gl=24, n_z=3)
_Z = 0.4
_BIN = (10.0, 10.4)


@pytest.fixture(scope="module")
def fid():
    from hod_mod.forecast import params
    return jnp.asarray(params.fiducial_vector())


@pytest.fixture(scope="module")
def model():
    from hod_mod.forecast.forward_jax import ForwardModel
    return ForwardModel(z_eff=_Z, log10m_star_bin=_BIN, **_TINY)


def test_tier4_layout():
    from hod_mod.forecast.forward_jax import (
        PARAM_NAMES, N_PARAM, WAVE4_MORPHOLOGY, TIER4_MORPHOLOGY, _Z_EVOL)
    from hod_mod.forecast.params import SECTORS
    assert N_PARAM == 111
    assert list(WAVE4_MORPHOLOGY) == list(PARAM_NAMES[102:106])
    assert list(TIER4_MORPHOLOGY) == list(PARAM_NAMES[106:])
    assert TIER4_MORPHOLOGY == ["rho_morph_q", "log10_f_size", "dsize_early",
                                "f_size_zs", "a_ia"]
    assert _Z_EVOL["log10_f_size"] == "f_size_zs"
    assert set(SECTORS["morphology"]) == set(WAVE4_MORPHOLOGY
                                             + TIER4_MORPHOLOGY)
    flat = [n for sec in SECTORS.values() for n in sec]
    assert sorted(flat) == sorted(PARAM_NAMES)


def test_joint_fraction_identities(model, fid):
    """rho = 0: joint == product exactly; the analytic rho derivative."""
    from hod_mod.forecast.forward_jax import _IDX
    te = model._theta_eff(fid)
    feq_c, feq_s = model._morph_q_joint(te)
    fe_c, fe_s = model._morph_fractions(te)
    fq_c, fq_s, _, _ = model._sfq_weights(te)
    np.testing.assert_allclose(np.asarray(feq_c), np.asarray(fe_c * fq_c),
                               rtol=1e-14)
    np.testing.assert_allclose(np.asarray(feq_s), np.asarray(fe_s * fq_s),
                               rtol=1e-14)
    g = jax.jacfwd(lambda t: model._morph_q_joint(t)[0])(te)
    v = np.sqrt(np.maximum(np.asarray(fe_c * (1 - fe_c) * fq_c * (1 - fq_c)),
                           1e-24))
    np.testing.assert_allclose(np.asarray(g[:, _IDX["rho_morph_q"]]), v,
                               rtol=1e-10)


def test_four_way_partition_any_rho(fid):
    """(E,SF)+(E,Q)+(L,SF)+(L,Q) == unsplit n_gal at rho = 0 AND rho != 0."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    kw = dict(z_eff=_Z, log10m_star_bin=_BIN, **_TINY)
    n_of = lambda m, t: float(m.predict(t, ["n_gal"])["n_gal"][0])
    n_all = n_of(ForwardModel(**kw), fid)
    for rho in (0.0, 0.25, -0.15):
        th = fid.at[_IDX["rho_morph_q"]].set(rho)
        tot = sum(n_of(ForwardModel(sfq=sv, morph=mo, **kw), th)
                  for sv in ("sf", "q") for mo in ("early", "late"))
        np.testing.assert_allclose(tot, n_all, rtol=1e-12)


def test_conditional_f_early_ordering(fid):
    """rho > 0 ⇒ the Q sample is earlier-typed than the SF sample."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX
    kw = dict(z_eff=_Z, log10m_star_bin=_BIN, **_TINY)
    th = fid.at[_IDX["rho_morph_q"]].set(0.2)
    f_q = float(ForwardModel(sfq="q", **kw).predict(th, ["f_early"])["f_early"][0])
    f_sf = float(ForwardModel(sfq="sf", **kw).predict(th, ["f_early"])["f_early"][0])
    assert f_q > f_sf


def test_f_early_q_observable(model, fid):
    """Joint fraction: bounded by the marginals, positive rho response."""
    from hod_mod.forecast.forward_jax import _IDX
    out = model.predict(fid, ["f_early", "f_early_q"])
    fe, feq = float(out["f_early"][0]), float(out["f_early_q"][0])
    assert 0.0 < feq < fe < 1.0
    g = jax.jacfwd(lambda t: model.predict(t, ["f_early_q"])["f_early_q"])(fid)
    assert float(g[0, _IDX["rho_morph_q"]]) > 0


def test_size_identities(model, fid):
    """∂size/∂log10_f_size = 1 exact; the f_size_zs chain rule; a
    nonzero cosmology response through R_200c."""
    from hod_mod.forecast.forward_jax import _IDX
    g = jax.jacfwd(lambda t: model.predict(t, ["size"])["size"])(fid)
    np.testing.assert_allclose(float(g[0, _IDX["log10_f_size"]]), 1.0,
                               rtol=1e-12)
    np.testing.assert_allclose(float(g[0, _IDX["f_size_zs"]]),
                               model._x_evol, rtol=1e-10)
    assert (abs(float(g[0, _IDX["Omega_m"]])) > 0
            or abs(float(g[0, _IDX["h"]])) > 0)
    assert float(g[0, _IDX["dsize_early"]]) > 0    # = f_early > 0


def test_size_mass_trend(fid):
    """Higher-M* cells live in bigger haloes -> larger mean sizes."""
    from hod_mod.forecast.forward_jax import ForwardModel
    kw = dict(z_eff=_Z, **_TINY)
    s_lo = float(ForwardModel(log10m_star_bin=(10.0, 10.4), **kw)
                 .predict(fid, ["size"])["size"][0])
    s_hi = float(ForwardModel(log10m_star_bin=(11.0, 11.4), **kw)
                 .predict(fid, ["size"])["size"][0])
    assert s_hi > s_lo


def test_wgp_identities(model, fid):
    """∂ln w_g+/∂a_ia = 1/a_ia exactly; w_g+ strictly ∝ the cell's early
    fraction (morphology-parameter changes rescale w_g+ and f_early by the
    same factor)."""
    from hod_mod.forecast.forward_jax import _IDX
    a0 = float(fid[_IDX["a_ia"]])
    g = jax.jacfwd(lambda t: jnp.log(jnp.abs(
        model.predict(t, ["wgp"])["wgp"])))(fid)
    np.testing.assert_allclose(np.asarray(g[:, _IDX["a_ia"]]), 1.0 / a0,
                               rtol=1e-10)
    th = fid.at[_IDX["log10_M_morph"]].add(-0.4)
    for t1, t2 in ((fid, th),):
        w1 = np.asarray(model.predict(t1, ["wgp"])["wgp"])
        w2 = np.asarray(model.predict(t2, ["wgp"])["wgp"])
        f1 = float(model.predict(t1, ["f_early"])["f_early"][0])
        f2 = float(model.predict(t2, ["f_early"])["f_early"][0])
        np.testing.assert_allclose(w2 / w1, f2 / f1, rtol=1e-10)


def test_f_early_agn_coupling(model, fid):
    """AGN hosts are earlier-typed than average; mbh_bt_slope > 0 pushes the
    AGN occupation into early-type haloes."""
    from hod_mod.forecast.forward_jax import _IDX
    out = model.predict(fid, ["f_early_agn"])
    fea = float(out["f_early_agn"][0])
    assert 0.0 < fea < 1.0
    g = jax.jacfwd(lambda t: model.predict(t, ["f_early_agn"])["f_early_agn"])(fid)
    assert float(g[0, _IDX["mbh_bt_slope"]]) > 0
    assert abs(float(g[0, _IDX["log10_M_morph"]])) > 0


def test_wgp_scale_cut(model):
    which = ["wgp", "size"]
    f, row_obs, row_x = model.full_data_vector_fn(which)
    keep = model.scale_cut_mask(row_obs, row_x, rmin=1.0)
    w = row_obs == "wgp"
    assert np.all(keep[w] == (row_x[w] > 1.0))
    assert np.all(keep[~w])


# ------------------------------------------------------------ assembly --

def _tiny_tier4(**over):
    from hod_mod.forecast.tier4 import Tier4Forecast
    kw = dict(z_edges=[0.2, 0.4], mstar_edges=[9.4, 10.6],
              agn_z_centers=(0.3,), agn_lx_bins=[(42.0, 43.0)],
              n_bands=1, n_shear_bins=2, n_k=32, n_m=48, n_gl=12, n_z=3,
              cell_n_m=64,
              rp_wp=np.logspace(-1, 1.4, 5), rp_ds=np.logspace(-1, 1.2, 4),
              ell=np.logspace(1.0, 3.3, 5), rp_wp_agn=np.logspace(0.1, 1.4, 3))
    kw.update(over)
    return Tier4Forecast(**kw)


def test_tier4_assembly_smoke(tmp_path):
    """Tiny end-to-end: every tier-4 family present with finite d0/J and
    positive noise; morph_cell blocks built for wide-complete cells only."""
    t4 = _tiny_tier4()
    fid = t4.fiducial()
    cache = str(tmp_path / "cache")
    d0, J, meta = t4.data_and_jacobian(fid, cache_dir=cache, verbose=False)
    assert np.all(np.isfinite(d0)) and np.all(np.isfinite(J))
    sig = t4.noise_sigma(fid, d0, meta, verbose=False)
    assert np.all(sig[np.isfinite(sig)] > 0)
    obs = set(meta["obs"])
    for o in ("f_early", "f_early_q", "size", "wgp", "f_early_agn"):
        assert o in obs, o
        s = (meta["obs"] == o) & np.isfinite(sig)
        assert s.any(), o
    kinds = set(meta["kind"])
    assert "morph_cell" in kinds
    mc = meta["kind"] == "morph_cell"
    assert np.all(np.isin(meta["obs"][mc], ["wp", "ds", "n_gal"]))
    assert np.isfinite(sig[mc]).all()
    d0b, Jb, _ = t4.data_and_jacobian(fid, cache_dir=cache, verbose=False)
    assert np.array_equal(d0, d0b) and np.array_equal(Jb, J)


def test_tier4_morph_split_additivity(tmp_path):
    """EARLY + LATE morph_cell n_gal equals the unsplit cell total."""
    t4 = _tiny_tier4()
    fid = t4.fiducial()
    d0, J, meta = t4.data_and_jacobian(fid, cache_dir=str(tmp_path),
                                       verbose=False)
    mc = (meta["kind"] == "morph_cell") & (meta["obs"] == "n_gal")
    n_split = d0[mc].sum()
    cells = ((meta["kind"] == "cell") & (meta["obs"] == "n_gal")
             & (meta["z_lo"] == meta["z_lo"][mc][0]))
    n_cells = d0[cells].sum()          # SF + Q sub-cells sum to the unsplit
    np.testing.assert_allclose(n_split, n_cells, rtol=1e-10)


_PAR_EQ_SER_T4 = r"""
import os, tempfile
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from hod_mod.forecast.tier4 import Tier4Forecast

t4 = Tier4Forecast(z_edges=[0.2, 0.4], mstar_edges=[9.4, 10.6],
                   agn_z_centers=(0.3,), agn_lx_bins=[(42.0, 43.0)],
                   n_bands=1, n_shear_bins=2, n_k=32, n_m=48, n_gl=12,
                   n_z=3, cell_n_m=64,
                   rp_wp=np.logspace(-1, 1.4, 5),
                   rp_ds=np.logspace(-1, 1.2, 4),
                   ell=np.logspace(1.0, 3.3, 5),
                   rp_wp_agn=np.logspace(0.1, 1.4, 3))
fid = np.asarray(t4.fiducial())
with tempfile.TemporaryDirectory() as td:
    cs, cp = os.path.join(td, "s"), os.path.join(td, "p")
    d0s, Js, _ = t4.data_and_jacobian(fid, cache_dir=cs, verbose=False)
    t4.precompute_blocks(fid, cp, jobs=2, verbose=False)
    d0p, Jp, _ = t4.data_and_jacobian(fid, cache_dir=cp, verbose=False)
    assert np.array_equal(d0s, d0p) and np.array_equal(Js, Jp)
print("T4 PARALLEL==SERIAL OK")
"""


def test_tier4_parallel_equals_serial():
    """Byte-identity through the batched pools, tier-4 blocks included
    (fresh interpreter — the pytest parent carries pre-x64 f32 constants)."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", _PAR_EQ_SER_T4],
                       capture_output=True, text=True, timeout=2400,
                       env=dict(os.environ, JAX_PLATFORMS="cpu"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "T4 PARALLEL==SERIAL OK" in r.stdout


# ------------------------------------------------------ coverage gaps --

def test_wgp_noise_unit():
    """wgp_noise: shape, positivity, source-density scaling, CV floor."""
    from hod_mod.forecast import noise
    rp = np.logspace(-1, 1.2, 6)
    wgp = 0.05 * (rp / 1.0) ** -0.8
    sh = noise.ShearSurvey()
    sp = noise.SpectroSurvey()
    sig = noise.wgp_noise(rp, wgp, 0.4, 1e-3, 1e8, 0.6736, 0.31, sh, sp)
    assert sig.shape == rp.shape and np.all(np.isfinite(sig)) and np.all(sig > 0)
    # doubling the source density lowers the shape-noise term
    sh2 = noise.ShearSurvey(n_eff=60.0)
    sig2 = noise.wgp_noise(rp, wgp, 0.4, 1e-3, 1e8, 0.6736, 0.31, sh2, sp)
    assert np.all(sig2 < sig)
    # a huge signal activates the cosmic-variance floor
    sig_big = noise.wgp_noise(rp, 1e4 * wgp, 0.4, 1e-3, 1e8, 0.6736, 0.31,
                              sh, sp)
    assert np.all(sig_big >= sp.cv_rel(1e8) * 1e4 * wgp * 0.999)


def test_driver_group_mask():
    """The __kind__ sentinel selects morph_cell rows; named groups exclude
    them (no double counting in the cumulative attribution)."""
    from hod_mod.scripts.forecasts.run_tier4_forecast import _group_mask
    meta = {"obs": np.array(["wp", "wp", "size", "wgp"]),
            "kind": np.array(["cell", "morph_cell", "cell", "cell"])}
    m_named = _group_mask(meta, ("wp",))
    np.testing.assert_array_equal(m_named, [True, False, False, False])
    m_kind = _group_mask(meta, ("__kind__morph_cell",))
    np.testing.assert_array_equal(m_kind, [False, True, False, False])
    m_both = _group_mask(meta, ("size", "__kind__morph_cell"))
    np.testing.assert_array_equal(m_both, [False, True, True, False])


def test_tier4_block_composition():
    """Observable placement rules, ctor-only (no Jacobians): f_early_q on
    ONE variant per (z, M*) cell; size/wgp on every cell; f_early_agn on
    main shells but not hi_local; morph_cell count follows the wide-tier
    completeness gate."""
    t4 = _tiny_tier4(z_edges=[0.2, 0.4, 0.6], mstar_edges=[9.4, 9.6, 10.6])
    cells = [b for b in t4.blocks if b.kind == "cell"]
    n_zm = len({(b.z_lo, b.m_lo) for b in cells})
    n_feq = sum(1 for b in cells if "f_early_q" in b.which)
    assert n_feq == n_zm                       # one copy per (z, M*) cell
    assert all("size" in b.which and "wgp" in b.which and
               "f_early" in b.which for b in cells)
    shells = [b for b in t4.blocks if b.kind == "shell"]
    for b in shells:
        assert ("f_early_agn" in b.which) == (b.label != "hi_local")
    # wide-tier gate: z_hi=0.4 -> lim 9.4 (both bins pass); z_hi=0.6 ->
    # lim 9.6 (only the 9.6 bin passes) => (2+1) bins x 2 morphs = 6
    mc = [b for b in t4.blocks if b.kind == "morph_cell"]
    assert len(mc) == 6
    assert all(set(b.which) == {"wp", "ds", "n_gal"} for b in mc)
    assert all(b.spectro is t4.spectro for b in mc)
