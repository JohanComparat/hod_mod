"""Tier-2 forecast extensions: promoted nuisances + z-evolution slopes.

Gates for the tier-2 param-vector growth (31 -> 47 promoted -> 54 with z-slopes):

* fiducials of promoted params equal the former module constants, so the
  fiducial prediction is unchanged (bit-identity is covered by the production
  regression suite in hod_mod/forecast/tests/);
* every promoted parameter is actually plumbed (nonzero derivative in the
  observable that should see it);
* the z-evolution mapping _theta_eff is the identity at the fiducial and obeys
  the chain rule d(data)/d(slope) = ln[(1+z)/(1+z_p)] * d(data)/d(base).
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


_TINY = dict(n_k=48, n_m=48, n_gl=16, n_z=3, n_z_shear=3)

# observable that must respond to each promoted parameter
_PLUMBING = {
    "beta_sat": "wp", "bcut": "wp", "beta_cut": "wp", "alpha_sat": "wp",
    "beta_b": "ds", "log10_M_eta": "ds", "beta_eta": "ds",
    "alpha_in_gas": "cl_gX", "alpha_tr_gas": "cl_gX",
    "p03_pressure": "cl_gy", "c_dpm_pressure": "cl_gy", "alpha_in_pressure": "cl_gy",
    "alpha_tr_pressure": "cl_gy", "alpha_out_pressure": "cl_gy",
    "agn_rho": "xlf", "agn_sig_mstar": "xlf",
}
_OBS = ("wp", "ds", "cl_gX", "cl_gy", "xlf")


@pytest.fixture(scope="module")
def model():
    from hod_mod.forecast.forward_jax import ForwardModel
    return ForwardModel(z_eff=0.4, **_TINY)      # z_eff != pivot -> x_evol != 0


@pytest.fixture(scope="module")
def fid():
    from hod_mod.forecast import params
    return jnp.asarray(params.fiducial_vector())


@pytest.fixture(scope="module")
def jac(model, fid):
    """(d0, J, slices) of the concatenated (wp, ds, cl_gX, cl_gy, xlf) vector."""
    f, row_obs, _ = model.full_data_vector_fn(list(_OBS))
    d0 = np.asarray(f(fid))
    J = np.asarray(jax.jacfwd(f)(fid))
    row_obs = np.asarray(row_obs)
    return d0, J, {o: row_obs == o for o in _OBS}


def test_vector_layout():
    from hod_mod.forecast.forward_jax import (
        PARAM_NAMES, N_PARAM, TIER2_PROMOTED, TIER2_ZSLOPES, TIER2_EXTENSION)
    assert PARAM_NAMES[0] == "Omega_m" and PARAM_NAMES[30] == "agn_log10_ferdf"
    assert len(TIER2_PROMOTED) == 16 and TIER2_PROMOTED[0] == "beta_sat"
    assert len(TIER2_ZSLOPES) == 7
    assert len(TIER2_EXTENSION) == 59
    assert list(TIER2_EXTENSION) == list(PARAM_NAMES[31:90])


def test_promoted_fiducials_equal_constants():
    from hod_mod.forecast import params
    from hod_mod.forecast import forward_jax as FJ
    fid = params.load_fiducial()
    expected = {
        "beta_sat": FJ._FIXED_HOD["beta_sat"], "bcut": FJ._FIXED_HOD["bcut"],
        "beta_cut": FJ._FIXED_HOD["beta_cut"], "alpha_sat": FJ._FIXED_HOD["alpha_sat"],
        "beta_b": FJ._FIXED_BARYON["beta_b"],
        "log10_M_eta": FJ._FIXED_BARYON["log10_M_eta"],
        "beta_eta": FJ._FIXED_BARYON["beta_eta"],
        "alpha_in_gas": FJ._ALPHA_IN_GAS, "alpha_tr_gas": FJ._ALPHA_TR_GAS,
        "p03_pressure": FJ._DPM_P2["P03"], "c_dpm_pressure": FJ._DPM_P2["c_dpm"],
        "alpha_in_pressure": FJ._DPM_P2["alpha_in"], "alpha_tr_pressure": FJ._DPM_P2["alpha_tr"],
        "alpha_out_pressure": FJ._DPM_P2["alpha_out"],
        "agn_rho": 0.0, "agn_sig_mstar": FJ._SIG_MSTAR_XLF,
    }
    for n, v in expected.items():
        assert fid[n] == pytest.approx(v, rel=0, abs=0), n
    # z-slopes are identically zero at the fiducial (no-evolution tier-1 model)
    from hod_mod.forecast.forward_jax import TIER2_ZSLOPES
    for n in TIER2_ZSLOPES:
        assert fid[n] == 0.0, n


def test_promoted_params_are_plumbed(jac):
    from hod_mod.forecast.forward_jax import _IDX
    d0, J, sel = jac
    for name, obs in _PLUMBING.items():
        col = J[sel[obs], _IDX[name]]
        ref = np.abs(d0[sel[obs]]).max()
        assert np.abs(col).max() > 1e-8 * ref, f"{name} not plumbed into {obs}"


def test_p03_pressure_is_pure_tsz_amplitude(jac, fid):
    """C_gy ∝ P_0.3 exactly: dlnC/dlnP_0.3 = 1; and P_0.3 touches nothing else.

    In the forecast the DPM pressure params drive only the tSZ leg — the X-ray
    temperature is still the phenomenological kT–M relation (kt_norm/kt_slope),
    so the pressure amplitude remains a pure-tSZ parameter.
    """
    from hod_mod.forecast.forward_jax import _IDX
    d0, J, sel = jac
    i = _IDX["p03_pressure"]
    gy = sel["cl_gy"]
    np.testing.assert_allclose(J[gy, i] * float(fid[i]), d0[gy], rtol=1e-8)
    for obs in ("wp", "ds", "cl_gX", "xlf"):
        assert np.abs(J[sel[obs], i]).max() == 0.0, f"p03_pressure leaks into {obs}"


def test_theta_eff_identity_at_fiducial(model, fid):
    np.testing.assert_array_equal(np.asarray(model._theta_eff(fid)), np.asarray(fid))


def test_theta_eff_chain_rule(jac, model):
    """d(data)/d(slope) == ln[(1+z)/(1+z_p)] * d(data)/d(base), exactly."""
    from hod_mod.forecast.forward_jax import _IDX, _Z_EVOL
    _, J, _ = jac
    x = model._x_evol
    assert x != 0.0
    for base, sl in _Z_EVOL.items():
        np.testing.assert_allclose(J[:, _IDX[sl]], x * J[:, _IDX[base]],
                                   rtol=1e-10, atol=1e-30, err_msg=f"{base}<->{sl}")


def test_sectors_cover_param_names():
    from hod_mod.forecast.forward_jax import PARAM_NAMES
    from hod_mod.forecast.params import SECTORS
    flat = [n for sec in SECTORS.values() for n in sec]
    assert sorted(flat) == sorted(PARAM_NAMES)


_BANDS3 = [(0.5, 0.9), (0.9, 1.3), (1.3, 2.0)]


@pytest.fixture(scope="module")
def band_models():
    from hod_mod.forecast.forward_jax import ForwardModel
    kw = dict(z_eff=0.4, agn_emission="powell", xlf_band="soft", **_TINY)
    mb = ForwardModel(xray_bands=_BANDS3, **kw)
    m1 = ForwardModel(xray_bands=[(0.5, 2.0)], **kw)
    return mb, m1


def test_band_additivity(band_models, fid):
    """Σ_b X-ray band spectra == the single-broad-band prediction.

    Exact at the P_gX(k) level (the galaxy×X-ray field is linear in the X-ray
    emission and APEC bands partition Λ_broad pointwise, even with a T profile);
    the C_ℓ level inherits a small log-interpolation nonlinearity.
    """
    mb, m1 = band_models
    H = mb._halo_common(fid, 0.4)
    # exact amplitude partition: Σ_b w_b = 1 by construction, so at k→0
    # (û→1) the band stack sums to the broad amplitude
    X3 = mb._emissivity_uk_bands(fid, H)
    X1 = m1._emissivity_uk_bands(fid, H)
    np.testing.assert_allclose(np.asarray(sum(X3))[0], np.asarray(X1[0])[0],
                               rtol=1e-6)
    # k-resolved: the per-band log-bilinear interpolants differ from the
    # broad interpolant between table nodes — sub-percent, mass-weighted tiny
    P3 = np.asarray(mb._pk_gX(fid, H))
    P1 = np.asarray(m1._pk_gX(fid, H))
    np.testing.assert_allclose(P3.sum(axis=0), P1[0], rtol=1e-4)
    cl3 = np.asarray(mb.predict(fid, ["cl_gX"])["cl_gX"]).reshape(len(_BANDS3), -1)
    cl1 = np.asarray(m1.predict(fid, ["cl_gX"])["cl_gX"])
    np.testing.assert_allclose(cl3.sum(axis=0), cl1, rtol=5e-3)


def test_spectral_params_plumbed_into_bands(band_models, fid):
    from hod_mod.forecast.forward_jax import _IDX
    mb, _ = band_models
    f = lambda t: mb.predict(t, ["cl_gX"])["cl_gX"]
    J = np.asarray(jax.jacfwd(f)(fid))
    d0 = np.asarray(f(fid))
    for n in ("t_prof_slope", "z_gas_norm", "z_gas_mslope", "z_gas_zs",
              "agn_gamma", "agn_fabs", "kt_norm"):
        assert np.abs(J[:, _IDX[n]]).max() > 1e-8 * np.abs(d0).max(), n
    # the temperature tilt must act through band RATIOS, not the total:
    # its derivative changes sign across bands (soft up, hard down or v.v.)
    nb, nell = len(_BANDS3), len(np.asarray(mb.ell))
    col = J[:, _IDX["t_prof_slope"]].reshape(nb, nell)
    assert (col.mean(axis=1).min() < 0.0) and (col.mean(axis=1).max() > 0.0)


def test_soft_xlf_shift_identity(fid):
    """At f_abs=0 and Γ=1.8, Φ_soft(l) == Φ_hard(l − log10 k_h2s) exactly."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX, _LOG10_K_H2S
    m = ForwardModel(z_eff=0.4, xlf_band="soft", **_TINY)
    th = np.asarray(fid).copy()
    th[_IDX["agn_fabs"]] = 0.0
    th = jnp.asarray(th)
    soft = np.asarray(m._xlf(th))
    hard = np.asarray(m._xlf_at(th, m.loglx_xlf - _LOG10_K_H2S))
    np.testing.assert_allclose(soft, hard, rtol=1e-12)
    # obscuration suppresses the observed bright end (shifts AGN to lower L_obs)
    dim = np.asarray(m._xlf(fid))
    assert dim[-1] < soft[-1]


def test_agn_band_fractions_sum_to_one(band_models, fid):
    mb, _ = band_models
    fb = np.asarray(mb._agn_band_fractions(fid))
    np.testing.assert_allclose(fb.sum(), 1.0, rtol=1e-12)
    assert np.all(fb > 0)


def test_mstar_bin_occupation_identities(fid):
    """Bin occupations are >= 0 and partition the threshold difference exactly."""
    from hod_mod.forecast.forward_jax import ForwardModel
    edges = [10.0, 10.4, 10.8]
    kw = dict(z_eff=0.25, **_TINY)
    bins = [ForwardModel(log10m_star_bin=(lo, hi), **kw)
            for lo, hi in zip(edges[:-1], edges[1:])]
    thr_lo = ForwardModel(log10m_star_thresh=edges[0], **kw)
    thr_hi = ForwardModel(log10m_star_thresh=edges[-1], **kw)

    for m in bins:
        nc, ns = m._occ_sample(fid)
        nc, ns = np.asarray(nc), np.asarray(ns)
        # non-negative up to float roundoff of the difference construction
        # (the ZM15 occupation itself underflows to ~0 at the top of the mass
        # grid, so the subtraction leaves ~1e-24 absolute noise there)
        assert np.all(nc >= -1e-12 * nc.max()) and np.all(ns >= -1e-12 * ns.max())

    def ngal(m):
        return float(m.predict(fid, ["n_gal"])["n_gal"][0])
    # sum of bin count densities == n(>lo_min) − n(>hi_max)  (exact algebra)
    np.testing.assert_allclose(sum(ngal(m) for m in bins),
                               ngal(thr_lo) - ngal(thr_hi), rtol=1e-12)

    # a bin sample still produces a sane clustering signal
    wp = np.asarray(bins[0].predict(fid, ["wp"])["wp"])
    assert np.all(np.isfinite(wp)) and np.all(wp > 0.0)


def test_mstar_bin_ctor_validation():
    from hod_mod.forecast.forward_jax import ForwardModel
    with pytest.raises(ValueError):
        ForwardModel(log10m_star_bin=(10.0, 10.4), log10m_star_thresh=10.0, **_TINY)
    with pytest.raises(ValueError):
        ForwardModel(log10m_star_bin=(10.4, 10.0), **_TINY)


def test_tomographic_shear(fid):
    """N=5 source bins: 15 pair spectra, normalized bins, sane auto ordering."""
    from hod_mod.forecast.forward_jax import ForwardModel
    kw = dict(_TINY, n_z_shear=16)          # the z grid must resolve the bins
    m = ForwardModel(z_eff=0.3, n_shear_bins=5, z_src_mean=0.9, **kw)
    assert len(m.shear_pairs) == 15
    with pytest.raises(ValueError):
        ForwardModel(z_eff=0.3, n_shear_bins=5, **_TINY)   # n_z_shear=3 too coarse
    zs = np.asarray(m.z_shear)
    for nz in m.nz_src_bins:
        np.testing.assert_allclose(np.trapezoid(np.asarray(nz), zs), 1.0, rtol=1e-10)
    # bin mean redshifts are ordered (equal-number quantile split)
    zbar = [float(np.trapezoid(np.asarray(nz) * zs, zs)) for nz in m.nz_src_bins]
    assert all(a < b for a, b in zip(zbar[:-1], zbar[1:]))

    out = m.predict(fid, ["cl_kk", "cl_shear_kCMB"])
    nell = len(np.asarray(m.ell))
    kk = np.asarray(out["cl_kk"]).reshape(15, nell)
    skc = np.asarray(out["cl_shear_kCMB"]).reshape(5, nell)
    assert np.all(np.isfinite(kk)) and np.all(kk > 0)
    assert np.all(np.isfinite(skc)) and np.all(skc > 0)
    # deeper source bins lens more: auto spectra increase with bin index
    autos = {p: k for p, k in zip(m.shear_pairs, kk) if p[0] == p[1]}
    a0, a4 = autos[(0, 0)], autos[(4, 4)]
    assert np.all(a4 > a0)
    # grids are tiled consistently
    assert len(np.asarray(m.grid_of("cl_kk"))) == 15 * nell
    assert len(np.asarray(m.grid_of("cl_shear_kCMB"))) == 5 * nell
    # Cauchy–Schwarz: C_ij² ≤ C_ii C_jj
    for (i, j), c in zip(m.shear_pairs, kk):
        assert np.all(c ** 2 <= autos[(i, i)] * autos[(j, j)] * (1 + 1e-12))


def test_single_bin_shear_unchanged(model, fid):
    """n_shear_bins=1 keeps the tier-1 single-spectrum path bit-identical."""
    from hod_mod.forecast.forward_jax import ForwardModel
    m1 = ForwardModel(z_eff=0.4, n_shear_bins=1, **_TINY)
    a = np.asarray(model.predict(fid, ["cl_kk"])["cl_kk"])
    b = np.asarray(m1.predict(fid, ["cl_kk"])["cl_kk"])
    np.testing.assert_array_equal(a, b)
    assert len(a) == len(np.asarray(m1.ell))


def test_tier2_forecast_smoke(fid, tmp_path):
    """2×2-cell Tier2Forecast: assembly, Jacobian, noise, Fisher end-to-end."""
    from hod_mod.forecast.tier2 import Tier2Forecast
    from hod_mod.forecast import fisher, params

    t2 = Tier2Forecast(
        z_edges=[0.1, 0.2, 0.3], mstar_edges=[10.0, 10.4, 10.8],
        n_bands=[(0.5, 1.0), (1.0, 2.0)], n_shear_bins=2,
        agn_lx_bins=[(42.0, 42.5), (42.5, 43.0)], agn_z_centers=(0.15,),
        n_k=48, n_m=48, n_gl=16, n_z=3,
        rp_wp=np.logspace(-1, 1.4, 6), rp_ds=np.logspace(-1, 1.2, 5),
        ell=np.logspace(1.0, 3.3, 6), rp_wp_agn=np.logspace(0.1, 1.4, 4))
    # 4 cells + 2 shells + global + 1 wp_agn block
    kinds = [b.kind for b in t2.blocks]
    assert kinds.count("cell") == 4 and kinds.count("shell") == 2
    assert kinds.count("global") == 1 and kinds.count("wp_agn") == 1

    fid61 = t2.fiducial()
    d0, J, meta = t2.data_and_jacobian(fid61, cache_dir=str(tmp_path),
                                       verbose=False)
    n_expected = (4 * (6 + 5 + 2 * 6 + 6 + 6 + 1)      # cells
                  + 2 * (7 + 2 * 6)                    # shells: xlf + band XX
                  + (3 * 6 + 6 + 2 * 6)                # global: pairs+kCMB+cross
                  + 2 * 4)                             # wp_agn
    assert d0.size == n_expected and J.shape == (n_expected, len(fid61))
    assert np.all(np.isfinite(d0)) and np.all(np.isfinite(J))
    # cache round-trip is identical
    d0b, Jb, _ = t2.data_and_jacobian(fid61, cache_dir=str(tmp_path),
                                      verbose=False)
    np.testing.assert_array_equal(d0, d0b)
    np.testing.assert_array_equal(J, Jb)

    sig = t2.noise_sigma(fid61, d0, meta, verbose=False)
    assert sig.shape == d0.shape and np.all(sig > 0)
    finite = np.isfinite(sig)
    assert finite.mean() > 0.8          # only completeness-flagged rows are inf

    keep = t2.scale_cut_mask(meta, 0.5) & finite
    rel = sig[keep] / np.abs(d0[keep])
    F = fisher.fisher_matrix(d0[keep], J[keep], rel_err=rel,
                             prior_sigma=t2.prior())
    cov, sigma, _ = fisher.constraints(F)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)
    # the data must inform Omega_m beyond its broad prior
    from hod_mod.forecast.forward_jax import _IDX
    assert sigma[_IDX["Omega_m"]] < params.BROAD_PRIOR_SIGMA["Omega_m"]


def test_zslope_inert_at_pivot(fid):
    """At z_eff == z_pivot the lever arm vanishes: slopes have zero effect."""
    from hod_mod.forecast.forward_jax import ForwardModel, _IDX, TIER2_ZSLOPES
    m = ForwardModel(z_eff=0.3, z_pivot_evol=0.3, **_TINY)
    assert m._x_evol == 0.0
    th = np.asarray(fid).copy()
    for n in TIER2_ZSLOPES:
        th[_IDX[n]] = 0.7
    base = np.asarray(m.predict(fid, ["n_gal"])["n_gal"])
    moved = np.asarray(m.predict(jnp.asarray(th), ["n_gal"])["n_gal"])
    np.testing.assert_array_equal(base, moved)
