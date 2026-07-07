"""Wave-2 differentiable production backend (``pk_backend='eh98_jax'``).

Validates the opt-in JAX forward model on FullHaloModelPrediction:
observables are finite/physical and jit-stable, agree with the CAMB backend
to a few percent (matched c(M)), and are jax.jacfwd/grad-able end-to-end
against central finite differences.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

_THETA_EH = {"Omega_m": 0.31, "Omega_b": 0.0493, "h": 0.6736,
             "n_s": 0.9649, "sigma8": 0.8111}
_RP = jnp.logspace(-1.0, 1.3, 8)


def _hod():
    from hod_mod.connection.hod import MoreHODModel
    return MoreHODModel.default_params()


# ---------------------------------------------------------------------------
# Observables: finite, physical, jit-stable
# ---------------------------------------------------------------------------

class TestEh98Observables:
    @pytest.fixture(scope="class")
    def pred(self):
        from hod_mod.observables import make_differentiable_prediction
        return make_differentiable_prediction("more15")

    def test_wp_physical(self, pred):
        wp = pred.wp(_RP, 100.0, 0.2, _THETA_EH, _hod())
        assert wp.shape == _RP.shape
        assert jnp.all(jnp.isfinite(wp)) and jnp.all(wp > 0)
        assert jnp.all(jnp.diff(wp) < 0)          # wp decreases with rp

    def test_delta_sigma_physical(self, pred):
        ds = pred.delta_sigma(_RP, 0.2, _THETA_EH, _hod())
        assert jnp.all(jnp.isfinite(ds)) and jnp.all(ds > 0)
        assert float(ds[0]) > float(ds[-1])

    def test_xi_positive_on_1halo_scales(self, pred):
        r = jnp.logspace(-1.0, 0.5, 6)
        xi = pred.xi_3d(r, 0.2, _THETA_EH, _hod())
        assert jnp.all(jnp.isfinite(xi)) and jnp.all(xi > 0)

    def test_jit_matches_eager(self, pred):
        hod = _hod()
        eager = pred.wp(_RP, 100.0, 0.2, _THETA_EH, hod)
        jitted = jax.jit(lambda th, h: pred.wp(_RP, 100.0, 0.2, th, h))(_THETA_EH, hod)
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-5)

    def test_n_gal_traceable(self, pred):
        # occupation-weighted number density is a traced scalar in this backend
        tables = pred._pk_tables_full(0.2, _THETA_EH, _hod())
        assert float(tables["n_gal"]) > 0
        assert 0.5 < float(tables["b_eff"]) < 5.0

    def test_hod_model_variants_build(self):
        from hod_mod.observables import make_differentiable_prediction
        from hod_mod.connection.hod import HODModel, ZuMandelbaum15HODModel
        for name, cls in (("zheng07", HODModel),
                          ("zumand15", ZuMandelbaum15HODModel)):
            p = make_differentiable_prediction(name)
            wp = p.wp(_RP, 100.0, 0.2, _THETA_EH, cls.default_params())
            assert jnp.all(jnp.isfinite(wp)) and jnp.all(wp > 0), name


# ---------------------------------------------------------------------------
# Accuracy vs the CAMB backend (matched c(M): isolates the EH98 P(k) shape)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEh98VsCamb:
    def test_wp_and_ds_within_few_percent(self, pk_lin, hmf):
        from hod_mod.observables import (
            make_differentiable_prediction, FullHaloModelPrediction)
        from hod_mod.core.concentration import ConcentrationModel
        from hod_mod.connection.hod import MoreHODModel

        theta_camb = pk_lin.default_cosmology()
        cm = ConcentrationModel("dutton14", mdef="200c")
        camb = FullHaloModelPrediction(
            pk_lin, MoreHODModel(hmf, hmf.bias), cm, profile="nfw")
        eh = make_differentiable_prediction("more15", cm_model="dutton14",
                                            mdef="200c")
        hod = MoreHODModel.default_params()

        wp_c = np.asarray(camb.wp(_RP, 100.0, 0.2, theta_camb, hod))
        wp_e = np.asarray(eh.wp(_RP, 100.0, 0.2, _THETA_EH, hod))
        # EH98 shape + sigma8 anchor reproduces CAMB clustering to a few percent
        assert np.max(np.abs(wp_e / wp_c - 1.0)) < 0.05

        ds_c = np.asarray(camb.delta_sigma(_RP, 0.2, theta_camb, hod))
        ds_e = np.asarray(eh.delta_sigma(_RP, 0.2, _THETA_EH, hod))
        assert np.max(np.abs(ds_e / ds_c - 1.0)) < 0.06


# ---------------------------------------------------------------------------
# Gradients: jacfwd vs central finite differences (needs x64)
# ---------------------------------------------------------------------------

@pytest.mark.x64
class TestEh98Gradients:
    @pytest.fixture(scope="class")
    def pred(self):
        from hod_mod.observables import make_differentiable_prediction
        return make_differentiable_prediction("more15")

    def _check(self, f, x0, rtol=1e-2):
        g = jax.jacfwd(f)(x0)
        eps = 1e-4 * (abs(x0) if x0 != 0 else 1.0)
        fd = (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps)
        assert jnp.all(jnp.isfinite(g))
        np.testing.assert_allclose(np.asarray(g), np.asarray(fd), rtol=rtol)
        return g

    def test_wp_grad_cosmology(self, pred):
        if not jax.config.jax_enable_x64:
            pytest.skip("requires JAX_ENABLE_X64=1")
        hod = _hod()
        g_s8 = self._check(
            lambda s8: pred.wp(_RP, 100.0, 0.2, dict(_THETA_EH, sigma8=s8), hod),
            0.8111)
        assert jnp.all(g_s8 > 0)          # more amplitude -> more clustering
        self._check(
            lambda om: pred.wp(_RP, 100.0, 0.2, dict(_THETA_EH, Omega_m=om), hod),
            0.31)

    def test_wp_grad_hod(self, pred):
        if not jax.config.jax_enable_x64:
            pytest.skip("requires JAX_ENABLE_X64=1")
        hod = _hod()
        self._check(
            lambda mm: pred.wp(_RP, 100.0, 0.2, _THETA_EH, dict(hod, log10mmin=mm)),
            float(hod["log10mmin"]))
        self._check(
            lambda a: pred.wp(_RP, 100.0, 0.2, _THETA_EH, dict(hod, alpha=a)),
            float(hod["alpha"]))

    def test_delta_sigma_grad(self, pred):
        if not jax.config.jax_enable_x64:
            pytest.skip("requires JAX_ENABLE_X64=1")
        self._check(
            lambda s8: pred.delta_sigma(_RP, 0.2, dict(_THETA_EH, sigma8=s8), _hod()),
            0.8111)

    def test_extension_params_carry_gradient(self, pred):
        if not jax.config.jax_enable_x64:
            pytest.skip("requires JAX_ENABLE_X64=1")
        hod = _hod()
        # assembly bias A_cen is identity at 0 but must carry a finite gradient
        g = jax.jacfwd(
            lambda a: pred.wp(_RP, 100.0, 0.2, _THETA_EH, dict(hod, A_cen=a)))(0.0)
        assert jnp.all(jnp.isfinite(g)) and jnp.any(jnp.abs(g) > 0)


# ---------------------------------------------------------------------------
# Guard rails: mis-wiring is rejected at __init__ / on call
# ---------------------------------------------------------------------------

class TestEh98Guards:
    def _eh_hmf_cm(self):
        from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear
        from hod_mod.core.halo_mass_function import make_hmf
        from hod_mod.core.concentration import ConcentrationModel
        eh = EisensteinHu98PkLinear()
        hmf = make_hmf("tinker08", pk_func=eh.as_hmf_pk_func())
        cm = ConcentrationModel("dutton14", mdef="200c")
        return eh, hmf, cm

    def test_rejects_camb_pk_lin(self, pk_lin, hmf):
        from hod_mod.observables import FullHaloModelPrediction
        from hod_mod.core.concentration import ConcentrationModel
        from hod_mod.connection.hod import MoreHODModel
        cm = ConcentrationModel("dutton14", mdef="200c")
        with pytest.raises(TypeError, match="EisensteinHu98PkLinear"):
            FullHaloModelPrediction(pk_lin, MoreHODModel(hmf, hmf.bias), cm,
                                    pk_backend="eh98_jax")

    def test_rejects_halo_profile(self, halo_profile):
        from hod_mod.observables import FullHaloModelPrediction
        from hod_mod.connection.hod import MoreHODModel
        eh, hmf, _ = self._eh_hmf_cm()
        with pytest.raises(TypeError, match="ConcentrationModel"):
            FullHaloModelPrediction(eh, MoreHODModel(hmf, hmf.bias), halo_profile,
                                    pk_backend="eh98_jax")

    def test_rejects_untraceable_cm_model(self):
        from hod_mod.observables import FullHaloModelPrediction
        from hod_mod.core.concentration import ConcentrationModel
        from hod_mod.connection.hod import MoreHODModel
        eh, hmf, _ = self._eh_hmf_cm()
        cm = ConcentrationModel("diemer15", mdef="200c", hmf=hmf)
        with pytest.raises(ValueError, match="c\\(M\\) model"):
            FullHaloModelPrediction(eh, MoreHODModel(hmf, hmf.bias), cm,
                                    pk_backend="eh98_jax")

    def test_rejects_einasto_and_bnl(self):
        from hod_mod.observables import FullHaloModelPrediction
        from hod_mod.connection.hod import MoreHODModel
        eh, hmf, cm = self._eh_hmf_cm()
        with pytest.raises(ValueError, match="profile='nfw'"):
            FullHaloModelPrediction(eh, MoreHODModel(hmf, hmf.bias), cm,
                                    profile="einasto", pk_backend="eh98_jax")
        with pytest.raises(ValueError, match="baryon split, BNL"):
            FullHaloModelPrediction(eh, MoreHODModel(hmf, hmf.bias), cm,
                                    bnl_model=object(), pk_backend="eh98_jax")

    def test_rejects_bad_backend_name(self):
        from hod_mod.observables import FullHaloModelPrediction
        eh, hmf, cm = self._eh_hmf_cm()
        from hod_mod.connection.hod import MoreHODModel
        with pytest.raises(ValueError, match="pk_backend"):
            FullHaloModelPrediction(eh, MoreHODModel(hmf, hmf.bias), cm,
                                    pk_backend="nonsense")

    def test_rejects_unsupported_hod_keys_on_call(self):
        from hod_mod.observables import make_differentiable_prediction
        pred = make_differentiable_prediction("more15")
        with pytest.raises(ValueError, match="f_cut"):
            pred.wp(_RP, 100.0, 0.2, _THETA_EH, dict(_hod(), f_cut=0.1))

    def test_rejects_baryon_params_on_call(self):
        from hod_mod.observables import make_differentiable_prediction
        pred = make_differentiable_prediction("more15")
        with pytest.raises(ValueError, match="baryon_params"):
            pred._pk_tables_full(0.2, _THETA_EH, _hod(),
                                 baryon_params={"log10_M_eta": 13.0})


# ---------------------------------------------------------------------------
# ConcentrationModel._mdef_delta_rho (new; makes it drop-in for HaloProfile)
# ---------------------------------------------------------------------------

class TestConcentrationModelMdef:
    def test_matches_halo_profile_and_supports_camb_path(self, pk_lin, hmf):
        from hod_mod.core.concentration import ConcentrationModel
        from hod_mod.core.halo_profiles import mdef_delta_rho
        from hod_mod.observables import FullHaloModelPrediction
        from hod_mod.connection.hod import MoreHODModel

        theta = pk_lin.default_cosmology()
        for mdef in ("200m", "200c", "vir"):
            cm = ConcentrationModel("dutton14" if mdef != "200m" else "duffy08",
                                    mdef=mdef)
            d1, r1 = cm._mdef_delta_rho(0.3, theta)
            d2, r2 = mdef_delta_rho(mdef, 0.3, theta)
            assert float(d1) == pytest.approx(float(d2))
            assert float(r1) == pytest.approx(float(r2))

        # a ConcentrationModel is now usable with the CAMB backend too
        cm = ConcentrationModel("dutton14", mdef="200c")
        camb = FullHaloModelPrediction(pk_lin, MoreHODModel(hmf, hmf.bias), cm)
        wp = camb.wp(_RP, 100.0, 0.2, theta, MoreHODModel.default_params())
        assert jnp.all(jnp.isfinite(wp)) and jnp.all(wp > 0)
