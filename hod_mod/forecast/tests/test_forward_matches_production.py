r"""Validation of the JAX forecast forward model.

Three checks:

1. w_p(r_p) and ΔΣ(R) from :class:`ForwardModel` reproduce the production
   :class:`~hod_mod.observables.clustering.FullHaloModelPrediction` (built with
   the *same* EH98 P(k) and Dutton14/200c concentration) to a few percent — this
   validates the ported 1h+2h halo-model assembly and the projections.
2. The JAX normalised gNFW² emissivity FT matches a direct numpy quadrature.
3. The autodiff Jacobian agrees with central finite differences for a few
   representative parameters — this validates ``jax.jacfwd`` through the model.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES, _IDX  # noqa: E402
from hod_mod.forecast import params  # noqa: E402


FID = params.fiducial_vector()


def _production_predictor(model):
    """Build a FullHaloModelPrediction with the SAME cosmology backend + c(M)
    (no baryons — the DM-only limit the forecast reduces to when f_b → 0)."""
    from hod_mod.core.halo_mass_function import HaloMassFunction
    from hod_mod.core.halo_profiles import HaloProfile
    from hod_mod.connection.hod.zumandelbaum15 import ZuMandelbaum15HODModel
    from hod_mod.observables.clustering import FullHaloModelPrediction
    from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear

    pk = EisensteinHu98PkLinear()
    hmf = HaloMassFunction(pk.as_hmf_pk_func(), model="tinker08", Delta=200.0)
    hod = ZuMandelbaum15HODModel(hmf)
    hp = HaloProfile({}, cm_relation="dutton14", mdef="200c")
    return FullHaloModelPrediction(pk, hod, hp, n_k=1024)


def _cosmo_hod_dicts(model, theta):
    c = {k: (float(v) if not hasattr(v, "shape") else float(v))
         for k, v in model._cosmo(jnp.asarray(theta)).items()}
    h = {k: (float(v) if np.ndim(v) == 0 else v)
         for k, v in model._hod(jnp.asarray(theta)).items()}
    return c, h


@pytest.mark.slow
def test_wp_ds_match_production():
    m = ForwardModel(n_k=1024, n_m=512, n_gl=64,
                     rp_wp=np.logspace(-1.0, 1.5, 12),
                     rp_ds=np.logspace(-1.0, 1.3, 12))
    th = jnp.asarray(FID)
    fhmp = _production_predictor(m)
    c, hod = _cosmo_hod_dicts(m, FID)

    # ΔΣ carries the extended-gas baryon split; validate its DM limit (f_b → 0)
    # against the production no-baryon ΔΣ by pinning the baryon fraction to zero.
    import hod_mod.forecast.forward_jax as FJ
    th_nobar = FID.copy(); th_nobar[_IDX["log10_M_pivot"]] = 16.0   # f_b → f_b_min
    saved = FJ._FIXED_BARYON["f_b_min"]
    try:
        FJ._FIXED_BARYON["f_b_min"] = 0.0                          # f_b → 0 exactly
        ds_jax = np.asarray(m.predict(jnp.asarray(th_nobar), ["ds"])["ds"])
    finally:
        FJ._FIXED_BARYON["f_b_min"] = saved
    wp_jax = np.asarray(m.predict(th, ["wp"])["wp"])
    wp_prod = np.asarray(fhmp.wp(m.rp_wp, m.pi_max, m.z_eff, c, hod))
    ds_prod = np.asarray(fhmp.delta_sigma(m.rp_ds, m.z_eff, c, hod))

    r_wp = np.abs(wp_jax / wp_prod - 1.0)
    r_ds = np.abs(ds_jax / ds_prod - 1.0)
    print("wp max rel diff", r_wp.max(), "ds (DM limit) max rel diff", r_ds.max())
    assert np.median(r_wp) < 0.03
    assert np.median(r_ds) < 0.03


def test_emissivity_ft_matches_numpy():
    """JAX normalised gNFW² FT vs a fine numpy reference for one halo."""
    from hod_mod.forecast.forward_jax import _profile_uk_normalized, _gnfw_sq, _gl_nodes
    k = jnp.logspace(-2, 1.5, 30)
    r_s = jnp.array([0.2]); r_max = jnp.array([1.5])
    a_in, a_tr, a_out = 0.9, 2.0, 1.234
    gx, gw = _gl_nodes(96)
    fn = _gnfw_sq(r_max[:, None] * gx[None, :] / r_s[:, None], a_in, a_tr, a_out)
    u_jax = np.asarray(_profile_uk_normalized(k, r_max, fn, gx, gw))[:, 0]

    # numpy reference
    rr = np.linspace(1e-4, 1.5, 40000)
    x = rr / 0.2
    f = (np.maximum(x, 1e-8) ** (-a_in) * (1 + np.maximum(x, 1e-8) ** a_tr)
         ** ((a_in - a_out) / a_tr)) ** 2
    kk = np.asarray(k)
    num = np.array([np.trapz(f * np.sinc(ki * rr / np.pi) * rr ** 2, rr) for ki in kk]) \
        if hasattr(np, "trapz") else \
        np.array([np.trapezoid(f * np.sinc(ki * rr / np.pi) * rr ** 2, rr) for ki in kk])
    den = np.trapezoid(f * rr ** 2, rr)
    u_ref = num / den
    rel = np.abs(u_jax / u_ref - 1.0)
    print("emissivity FT max rel diff", rel.max())
    assert rel.max() < 0.02


def test_jacobian_matches_finite_difference():
    """Autodiff dwp/dθ vs central finite differences for σ8, Ω_m, lg_m1h."""
    m = ForwardModel(n_k=256, n_m=256, n_gl=48, rp_wp=np.logspace(-0.5, 1.2, 6))
    th = jnp.asarray(FID)

    def wp_of(theta):
        return m.predict(theta, ["wp"])["wp"]

    J = np.asarray(jax.jacfwd(wp_of)(th))            # (Nrp, Nparam)
    # These params have a non-negligible wp derivative (fc cancels analytically:
    # wp is independent of fc, so its FD is ~0 and the ratio metric is undefined).
    for name in ("sigma8", "Omega_m", "h", "n_s", "Omega_b",
                 "lg_m1h", "beta", "gamma", "sigma_lnmstar", "bsat"):
        i = _IDX[name]
        eps = 1e-4 * max(abs(FID[i]), 1.0)   # small step: large steps cross a mild
        tp = FID.copy(); tp[i] += eps         # non-smoothness in the SHMR bisection

        tm = FID.copy(); tm[i] -= eps
        fd = (np.asarray(wp_of(jnp.asarray(tp))) - np.asarray(wp_of(jnp.asarray(tm)))) / (2 * eps)
        ad = J[:, i]
        rel = np.abs(ad - fd) / (np.abs(fd) + 1e-30)
        print(name, "median rel diff (AD vs FD)", np.median(rel))
        assert np.median(rel) < 0.02


def test_baryon_jacobian_matches_fd():
    """Autodiff d(ΔΣ)/dθ vs FD for the shared baryon-sector parameters."""
    m = ForwardModel(n_k=256, n_m=256, n_gl=48, rp_ds=np.logspace(-1.0, 1.0, 6))
    th = jnp.asarray(FID)

    def ds_of(theta):
        return m.predict(theta, ["ds"])["ds"]

    J = np.asarray(jax.jacfwd(ds_of)(th))
    for name in ("log10_M_pivot", "log10_eta_min", "sigma8"):
        i = _IDX[name]
        eps = 1e-4 * max(abs(FID[i]), 1.0)
        tp = FID.copy(); tp[i] += eps
        tm = FID.copy(); tm[i] -= eps
        fd = (np.asarray(ds_of(jnp.asarray(tp))) - np.asarray(ds_of(jnp.asarray(tm)))) / (2 * eps)
        ad = J[:, i]
        rel = np.abs(ad - fd) / (np.abs(fd) + 1e-30)
        print(name, "median rel diff dΔΣ (AD vs FD)", np.median(rel))
        assert np.median(rel) < 0.02


def test_cosmic_shear_finite_and_differentiable():
    """C_ℓ^{κκ} is finite, has a sensible amplitude, and its autodiff matches FD."""
    m = ForwardModel(n_k=128, n_m=128, n_z_shear=8)
    th = jnp.asarray(FID)
    ckk = np.asarray(m.predict(th, ["cl_kk"])["cl_kk"])
    ell = np.asarray(m.ell)
    assert np.all(np.isfinite(ckk)) and np.all(ckk > 0)
    peak = np.max(ell ** 2 * ckk / (2 * np.pi))     # ℓ²C/2π peaks ~1e-5–1e-3
    assert 1e-6 < peak < 1e-2, peak

    def kk(theta):
        return m.predict(theta, ["cl_kk"])["cl_kk"]
    J = np.asarray(jax.jacfwd(kk)(th))
    for name in ("sigma8", "Omega_m", "log10_M_pivot"):
        i = _IDX[name]
        eps = 1e-4 * max(abs(FID[i]), 1.0)
        tp = FID.copy(); tp[i] += eps
        tm = FID.copy(); tm[i] -= eps
        fd = (np.asarray(kk(jnp.asarray(tp))) - np.asarray(kk(jnp.asarray(tm)))) / (2 * eps)
        rel = np.abs(J[:, i] - fd) / (np.abs(fd) + 1e-40)
        print(name, "median rel diff dC_kk (AD vs FD)", np.median(rel))
        assert np.median(rel) < 0.02


def test_cmb_lensing_observables():
    """κ_CMB auto + galaxy×κ_CMB + shear×κ_CMB are finite, positive, differentiable."""
    m = ForwardModel(n_k=128, n_m=128, n_z_shear=8)
    th = jnp.asarray(FID)
    for o in ("cl_kCMB", "cl_gkCMB", "cl_shear_kCMB"):
        v = np.asarray(m.predict(th, [o])[o])
        assert np.all(np.isfinite(v)) and np.all(v > 0), o
    # galaxy×CMB-lensing must respond to cosmology; check autodiff vs FD
    J = np.asarray(jax.jacfwd(lambda t: m.predict(t, ["cl_gkCMB"])["cl_gkCMB"])(th))
    for name in ("sigma8", "Omega_m"):
        i = _IDX[name]; eps = 1e-4 * max(abs(FID[i]), 1.0)
        tp = FID.copy(); tp[i] += eps; tm = FID.copy(); tm[i] -= eps
        fd = (np.asarray(m.predict(jnp.asarray(tp), ["cl_gkCMB"])["cl_gkCMB"])
              - np.asarray(m.predict(jnp.asarray(tm), ["cl_gkCMB"])["cl_gkCMB"])) / (2 * eps)
        assert np.median(np.abs(J[:, i] - fd) / (np.abs(fd) + 1e-40)) < 0.02


def test_energy_closure_baryon_fraction():
    """Energy-regulated f_b is finite, ∈[f_b_min, f_b_cosmic], rises with mass."""
    m = ForwardModel(n_k=64, n_m=128, energy_closure=True)
    th = FID.copy(); th[_IDX["log10_M_pivot"]] = -2.0    # log10 ε_AGN
    c = {k: float(v) for k, v in m._cosmo(jnp.asarray(th)).items()}
    fb = np.asarray(m._fb_energy(jnp.asarray(th), m._cosmo(jnp.asarray(th))))
    fbc = c["Omega_b"] / c["Omega_m"]
    assert np.all(np.isfinite(fb))
    assert fb.min() >= 0.0 and fb.max() <= fbc + 1e-6
    # f_b increases towards cluster mass (baryons re-accreted in deep potentials)
    assert fb[-1] > fb[m.m.shape[0] // 2]


def test_abundance_observables():
    """n_gal matches production HODBase._integrate; SMF is finite, positive, differentiable."""
    from hod_mod.core.halo_mass_function import HaloMassFunction
    from hod_mod.connection.hod.zumandelbaum15 import ZuMandelbaum15HODModel
    from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear
    m = ForwardModel(n_k=128, n_m=256)
    th = jnp.asarray(FID)
    pk = EisensteinHu98PkLinear()
    hmf = HaloMassFunction(pk.as_hmf_pk_func(), model="tinker08", Delta=200.0)
    hod = ZuMandelbaum15HODModel(hmf)
    c = {k: float(v) for k, v in m._cosmo(th).items()}
    hp = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in m._hod(th).items()}
    ng_prod = float(hod.galaxy_number_density(m.z_eff, c, hp))
    ng_mine = float(np.asarray(m.predict(th, ["n_gal"])["n_gal"])[0])
    print("n_gal reldiff vs production", abs(ng_mine / ng_prod - 1))
    assert abs(ng_mine / ng_prod - 1) < 0.02

    smf = np.asarray(m.predict(th, ["smf"])["smf"])
    assert np.all(np.isfinite(smf)) and np.all(smf > 0)
    # SMF autodiff vs FD w.r.t. an SHMR parameter it must respond to
    J = np.asarray(jax.jacfwd(lambda t: m.predict(t, ["smf"])["smf"])(th))
    i = _IDX["lg_m0star"]; eps = 1e-4 * abs(FID[i])
    tp = FID.copy(); tp[i] += eps; tm = FID.copy(); tm[i] -= eps
    fd = (np.asarray(m.predict(jnp.asarray(tp), ["smf"])["smf"])
          - np.asarray(m.predict(jnp.asarray(tm), ["smf"])["smf"])) / (2 * eps)
    ad = J[:, i]
    assert np.median(np.abs(ad - fd) / (np.abs(fd) + 1e-30)) < 0.02


def test_tomography_shared_cosmology():
    """Multi-bin tomography: shared cosmology + per-bin HOD, correct stacked Jacobian."""
    from hod_mod.forecast.tomography import TomographicForecast
    tf = TomographicForecast([("A", 0.15, 10.0), ("B", 0.25, 10.6)], n_k=64, n_m=96)
    assert len(tf.global_names) == 22 + 9 * 2          # 22 shared + 9 HOD per bin
    fg = tf.fiducial()
    f, rb, ro, rx = tf.data_vector_fn(["wp", "n_gal"])
    J = np.asarray(jax.jacfwd(f)(jnp.asarray(fg)))
    assert np.all(np.isfinite(J)) and J.shape[1] == len(tf.global_names)
    # a shared cosmology param perturbs BOTH bins; a bin-A HOD param only bin A
    gi = {n: i for i, n in enumerate(tf.global_names)}
    rowsA = rb == "A"; rowsB = rb == "B"
    assert np.any(J[rowsA, gi["sigma8"]] != 0) and np.any(J[rowsB, gi["sigma8"]] != 0)
    jA = gi["lg_m1h__A"]
    assert np.any(J[rowsA, jA] != 0) and np.allclose(J[rowsB, jA], 0.0)


def test_gas_density_ft_matches_numpy():
    """JAX normalised gNFW *density* FT (used for the ΔΣ gas component) vs numpy."""
    from hod_mod.forecast.forward_jax import _profile_uk_normalized, _gnfw, _gl_nodes
    k = jnp.logspace(-2, 1.5, 30)
    r_s = jnp.array([0.15]); r_max = jnp.array([2.0])
    a_in, a_tr, a_out = 0.9, 2.0, 1.234
    gx, gw = _gl_nodes(96)
    fn = _gnfw(r_max[:, None] * gx[None, :] / r_s[:, None], a_in, a_tr, a_out)
    u_jax = np.asarray(_profile_uk_normalized(k, r_max, fn, gx, gw))[:, 0]
    rr = np.linspace(1e-4, 2.0, 40000)
    x = rr / 0.15
    f = (np.maximum(x, 1e-8) ** (-a_in) * (1 + np.maximum(x, 1e-8) ** a_tr)
         ** ((a_in - a_out) / a_tr))
    num = np.array([np.trapezoid(f * np.sinc(ki * rr / np.pi) * rr ** 2, rr) for ki in np.asarray(k)])
    u_ref = num / np.trapezoid(f * rr ** 2, rr)
    rel = np.abs(u_jax / u_ref - 1.0)
    print("gas density FT max rel diff", rel.max())
    assert rel.max() < 0.02


def test_xlf_finite_and_jacobian_matches_fd():
    """Powell XLF observable is finite/declining, its autodiff matches FD, and the
    active-fraction response is the exact flat amplitude ∂lnΦ/∂log10_ferdf = ln10."""
    m = ForwardModel(n_k=128, n_m=256)
    th = jnp.asarray(FID)
    phi = np.asarray(m.predict(th, ["xlf"])["xlf"])
    lx = np.asarray(m.grid_of("xlf"))
    assert np.all(np.isfinite(phi)) and np.all(phi > 0)
    assert phi[-1] < phi[0]                              # declines toward high L_X

    def xlf(theta):
        return m.predict(theta, ["xlf"])["xlf"]
    J = np.asarray(jax.jacfwd(xlf)(th))                  # (Nlx, Nparam)
    for name in ("Omega_m", "sigma8", "agn_mu_bh", "agn_log10_ferdf", "agn_delta2"):
        i = _IDX[name]
        eps = 1e-4 * max(abs(FID[i]), 1.0)
        tp = FID.copy(); tp[i] += eps
        tm = FID.copy(); tm[i] -= eps
        fd = (np.asarray(xlf(jnp.asarray(tp))) - np.asarray(xlf(jnp.asarray(tm)))) / (2 * eps)
        rel = np.abs(J[:, i] - fd) / (np.abs(fd) + 1e-40)
        print(name, "median rel diff dPhi (AD vs FD)", np.median(rel))
        assert np.median(rel) < 0.02
    # f_ERDF is a pure multiplicative amplitude: ∂lnΦ/∂log10_ferdf = ln10 exactly.
    dln_ferdf = J[:, _IDX["agn_log10_ferdf"]] / phi
    assert np.allclose(dln_ferdf, np.log(10.0), rtol=1e-6)


if __name__ == "__main__":
    test_xlf_finite_and_jacobian_matches_fd()
    test_emissivity_ft_matches_numpy()
    test_gas_density_ft_matches_numpy()
    test_cosmic_shear_finite_and_differentiable()
    test_energy_closure_baryon_fraction()
    test_jacobian_matches_finite_difference()
    test_baryon_jacobian_matches_fd()
    test_wp_ds_match_production()
    print("all validation checks passed")
