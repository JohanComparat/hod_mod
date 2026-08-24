"""Growth-factor dispatcher (:mod:`hod_mod.core.growth`).

Gates the three routes to D(z)/D(0) — the Carroll+1992 fitting formula, the
flat-CPL growth ODE, and the ratio of the active P(k) backend's own spectra —
on the properties the halo model depends on:

* mode selection is explicit (argument, then ``$HOD_MOD_GROWTH``, then default);
* every route is normalised so that D(0) = 1 *exactly*;
* the ODE reproduces the exact flat-LCDM quadrature, and the backend ratio
  reproduces the ODE, across the Omega_m range a free-cosmology fit explores;
* a backend with no redshift dependence of its own falls back to the ODE
  loudly and exactly once, rather than silently returning D(z) = 1;
* every route stays differentiable, which is what the Fisher forecast needs;
* the cold/total distinction is exact (not merely close) at Sigma m_nu = 0.
"""

import os
import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_THETA = {"h": 0.6736, "Omega_b": 0.0493, "Omega_cdm": 0.2607, "Omega_m": 0.3100,
          "n_s": 0.9649, "ln10^{10}A_s": 3.044}
_THETA_S8 = {"h": 0.6736, "Omega_b": 0.0493, "Omega_m": 0.3100,
             "n_s": 0.9649, "sigma8": 0.8111}
_ZS = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])


def _d_exact(z, om, n=200001):
    """Exact flat-LCDM growing mode by quadrature: D ∝ E(a) ∫ da/(aE)³."""
    a = np.linspace(1e-6, 1.0, n)
    e = np.sqrt(om / a ** 3 + 1.0 - om)
    d = e * (np.cumsum(1.0 / (a * e) ** 3) * (a[1] - a[0]))
    return np.interp(1.0 / (1.0 + np.asarray(z)), a, d / d[-1])


@pytest.fixture(scope="module")
def cosmopower():
    pytest.importorskip("cosmopower_jax")
    from hod_mod.forecast.pk_cosmopower import CosmoPowerJaxPkLinear
    return CosmoPowerJaxPkLinear()


# ---------------------------------------------------------------- selection --

class TestModeSelection:
    def test_default_is_pk_ratio(self, monkeypatch):
        from hod_mod.core import growth
        monkeypatch.delenv("HOD_MOD_GROWTH", raising=False)
        assert growth.resolve_mode() == "pk_ratio"
        assert growth.DEFAULT_MODE in growth.GROWTH_MODES

    def test_env_var_overrides_default(self, monkeypatch):
        from hod_mod.core import growth
        monkeypatch.setenv("HOD_MOD_GROWTH", "carroll")
        assert growth.resolve_mode() == "carroll"

    def test_argument_overrides_env_var(self, monkeypatch):
        from hod_mod.core import growth
        monkeypatch.setenv("HOD_MOD_GROWTH", "carroll")
        assert growth.resolve_mode("ode") == "ode"

    def test_unknown_mode_raises(self):
        from hod_mod.core import growth
        with pytest.raises(ValueError, match="growth mode"):
            growth.resolve_mode("hubble")

    def test_unknown_variant_raises(self):
        from hod_mod.core import growth
        with pytest.raises(ValueError, match="variant"):
            growth.growth_factor(0.3, _THETA, mode="ode", variant="warm")


# ---------------------------------------------------------------- accuracy --

class TestAccuracy:
    def test_d_of_zero_is_exactly_one(self, cosmopower):
        """Not approximately: a normalisation that drifts would bias sigma8."""
        from hod_mod.core import growth
        for mode in growth.GROWTH_MODES:
            d0 = float(growth.growth_factor(0.0, _THETA, pk=cosmopower, mode=mode))
            assert d0 == 1.0, mode

    def test_ode_matches_exact_quadrature(self):
        """The ODE is the accurate analytic route; hold it to 3e-4."""
        from hod_mod.core import growth
        for om in (0.20, 0.24, 0.31, 0.40, 0.45):
            th = {**_THETA, "Omega_m": om}
            d = np.array([float(growth.growth_factor(z, th, mode="ode")) for z in _ZS])
            assert np.max(np.abs(d / _d_exact(_ZS, om) - 1.0)) < 3e-4

    def test_carroll_is_worse_and_omega_m_dependent(self):
        """Documents *why* Carroll is not the default: its error tracks Omega_m."""
        from hod_mod.core import growth
        errs = {}
        for om in (0.24, 0.31, 0.40):
            th = {**_THETA, "Omega_m": om}
            d = np.array([float(growth.growth_factor(z, th, mode="carroll")) for z in _ZS])
            errs[om] = np.max(np.abs(d / _d_exact(_ZS, om) - 1.0))
        assert errs[0.24] > 3e-3          # far outside the ODE's 3e-4
        assert errs[0.24] > 5 * errs[0.40]  # and strongly Omega_m dependent

    def test_pk_ratio_matches_ode(self, cosmopower):
        """The two accurate routes agree, so either can anchor the other."""
        from hod_mod.core import growth
        for z in (0.1, 0.3, 0.7, 1.0, 2.0):
            r = float(growth.growth_pk_ratio(z, _THETA, cosmopower))
            o = float(growth.growth_factor(z, _THETA, mode="ode"))
            assert abs(r / o - 1.0) < 1e-3

    def test_monotonic_and_bounded(self, cosmopower):
        from hod_mod.core import growth
        for mode in growth.GROWTH_MODES:
            d = np.array([float(growth.growth_factor(z, _THETA, pk=cosmopower, mode=mode))
                          for z in _ZS])
            assert np.all(np.diff(d) < 0.0), mode      # growth decreases with z
            assert np.all((d > 0.0) & (d <= 1.0)), mode


# ---------------------------------------------------------------- fallback --

class TestFallback:
    def test_eh98_falls_back_to_ode_and_warns_once(self):
        """A z-independent backend must never be read as a perfect D(z) = 1."""
        from hod_mod.core import growth
        from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear
        eh = EisensteinHu98PkLinear()
        assert growth.backend_has_native_z(eh) is False
        growth._WARNED.discard(f"pk_ratio-no-z:{type(eh).__name__}")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            d1 = float(growth.growth_factor(0.3, _THETA_S8, pk=eh))
            d2 = float(growth.growth_factor(0.3, _THETA_S8, pk=eh))
        assert len(rec) == 1 and "EisensteinHu98PkLinear" in str(rec[0].message)
        assert d1 == d2 == pytest.approx(
            float(growth.growth_factor(0.3, _THETA_S8, mode="ode")), abs=1e-15)

    def test_missing_backend_falls_back(self):
        from hod_mod.core import growth
        growth._WARNED.discard("pk_ratio-no-backend")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            d = float(growth.growth_factor(0.3, _THETA, pk=None, mode="pk_ratio"))
        assert len(rec) == 1
        assert d == pytest.approx(float(growth.growth_factor(0.3, _THETA, mode="ode")))

    def test_unknown_backend_is_not_assumed_z_aware(self):
        """Refusing to guess is what keeps a z-ignoring backend from looking exact."""
        from hod_mod.core import growth

        class MysteryPk:
            def pk_linear(self, k, z, theta):
                return jnp.ones_like(jnp.asarray(k))

        growth._WARNED.discard("unknown-backend:MysteryPk")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert growth.backend_has_native_z(MysteryPk()) is False
        assert len(rec) == 1 and "known-backend table" in str(rec[0].message)

    def test_every_shipped_backend_declares_itself(self):
        """No backend may be classified by the class-name fallback table.

        The table exists for third-party objects.  A shipped backend that relies
        on it is one rename away from being silently treated as not z-aware,
        which downgrades it to the ODE without any error.
        """
        from hod_mod.core.power_spectrum import (
            LinearPowerSpectrum, ClassPkLinear, CsstLinearPowerSpectrum)
        from hod_mod.forecast.pk_cosmopower import CosmoPowerJaxPkLinear
        from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear
        expected = {LinearPowerSpectrum: True, ClassPkLinear: True,
                    CsstLinearPowerSpectrum: True, CosmoPowerJaxPkLinear: True,
                    EisensteinHu98PkLinear: False}
        for cls, want in expected.items():
            assert "has_native_z" in vars(cls), f"{cls.__name__} not declared"
            assert vars(cls)["has_native_z"] is want, cls.__name__

    def test_cache_wrapper_answers_for_what_it_wraps(self):
        """_CachedPkLinear must not read as an unknown backend."""
        from hod_mod.core.growth import backend_has_native_z
        from hod_mod.fitting.fitters import _CachedPkLinear
        from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear

        class FakeCamb:
            has_native_z = True

            def pk_linear(self, k, z, theta):
                return jnp.ones_like(jnp.asarray(k))

        with warnings.catch_warnings():
            warnings.simplefilter("error")        # any warning here is a failure
            assert backend_has_native_z(_CachedPkLinear(FakeCamb())) is True
            assert backend_has_native_z(
                _CachedPkLinear(EisensteinHu98PkLinear())) is False

    def test_pk_ratio_refuses_a_non_native_backend(self):
        """The public entry point must not KeyError its way to a wrong answer."""
        from hod_mod.core.growth import growth_pk_ratio
        from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear
        with pytest.raises(ValueError, match="has_native_z=False"):
            growth_pk_ratio(0.3, _THETA, EisensteinHu98PkLinear())

    def test_declaration_wins_over_the_fallback_table(self):
        """A declared attribute must beat the class-name table, not tie with it.

        Written as a class *in* the table whose declaration disagrees with it --
        the only construction that actually exercises the precedence.  Setting an
        instance attribute on a backend that already declares one (the earlier
        form of this test) exercises Python's attribute lookup, not ours.
        """
        from hod_mod.core import growth

        class LinearPowerSpectrum:               # shadows the real name on purpose
            has_native_z = False                 # ... and disagrees with the table

        assert growth._NATIVE_Z["LinearPowerSpectrum"] is True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert growth.backend_has_native_z(LinearPowerSpectrum()) is False

    def test_fallback_table_still_serves_undeclared_objects(self):
        """The table is a courtesy for duck-typed third parties; keep it working."""
        from hod_mod.core import growth

        class ClassPkLinear:                     # no declaration at all
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error")       # a table hit must not warn
            assert growth.backend_has_native_z(ClassPkLinear()) is True


# ------------------------------------------------------- amplitude handling --

class TestAmplitudeConvention:
    def test_sigma8_theta_does_not_recurse(self, cosmopower):
        """sigma8-convention pk_linear calls growth_factor; the ratio must not."""
        from hod_mod.core import growth
        d = float(growth.growth_factor(0.3, _THETA_S8, pk=cosmopower))
        assert 0.8 < d < 0.9

    def test_ratio_is_amplitude_independent(self, cosmopower):
        """A_s cancels in the ratio analytically -- but only to ~3e-7 in practice.

        P(k) is linear in A_s, so D(z) read as a ratio should not depend on it at
        all.  The emulator is a neural network with ln10^{10}A_s as an *input*,
        and its learned mapping is only approximately linear in that input, so a
        0.14 shift in ln10^{10}A_s moves the recovered D(z) by ~2.5e-7.  That is
        far below every tolerance in this module and below the emulator's own
        ~4e-4 fidelity, but it is not zero, and pinning it here means a future
        emulator swap that degrades it will be noticed.
        """
        from hod_mod.core import growth
        a = float(growth.growth_pk_ratio(0.5, _THETA, cosmopower))
        b = float(growth.growth_pk_ratio(
            0.5, {**_THETA, "ln10^{10}A_s": 2.9}, cosmopower))
        assert abs(a / b - 1.0) < 1e-6

    def test_never_synthesises_omega_cdm(self):
        """The neutrino budget must stay closed: no invented Omega_cdm.

        There is no single correct value to invent.  CAMB and CLASS want the
        closed-budget Omega_m - Omega_b - Omega_nu; the massless-trained
        emulator and EH98 want Omega_m - Omega_b, with the neutrinos taken back
        out by the nu_ratio table.  Both derive it themselves, so an injected
        key can only *override* the right choice -- which for the two Boltzmann
        backends means silently running at Omega_m + Omega_nu, and for CLASS
        also poisoning its solver cache key with that value.
        """
        from hod_mod.core.growth import _as_amplitude_theta
        th = {"h": 0.6736, "Omega_b": 0.0493, "Omega_m": 0.3100,
              "n_s": 0.9649, "sigma8": 0.8111, "sum_mnu": 0.06}
        out = _as_amplitude_theta(th)
        assert "Omega_cdm" not in out
        assert "sigma8" not in out                    # the recursion guard
        assert out["sum_mnu"] == 0.06                 # the nu sector survives

    def test_explicit_omega_cdm_is_passed_through(self):
        """A caller who sets it keeps it -- the guard must not strip the key."""
        from hod_mod.core.growth import _as_amplitude_theta
        th = {"h": 0.6736, "Omega_b": 0.0493, "Omega_m": 0.3100,
              "Omega_cdm": 0.2593, "n_s": 0.9649, "sum_mnu": 0.06}
        assert _as_amplitude_theta(th)["Omega_cdm"] == 0.2593

    def test_budget_closes_for_the_production_default(self):
        """default_cosmology() + the helper must reproduce Omega_m exactly."""
        from hod_mod.core.power_spectrum import (
            LinearPowerSpectrum, omega_cdm, omega_nu)
        th = LinearPowerSpectrum.default_cosmology()
        assert (float(th["Omega_b"]) + float(omega_cdm(th)) + float(omega_nu(th))
                == pytest.approx(float(th["Omega_m"]), rel=1e-12))

    def test_sigma2_tophat_matches_the_forecast_copy(self):
        """core.growth duplicates this helper to avoid a core->forecast cycle."""
        from hod_mod.core.growth import _sigma2_tophat as core_impl, _K_INT
        from hod_mod.forecast.pk_eisenstein_hu import _sigma2_tophat as fc_impl
        pk = _K_INT ** (-1.5)
        assert float(core_impl(pk, _K_INT, 8.0)) == pytest.approx(
            float(fc_impl(pk, _K_INT, 8.0)), rel=1e-12)


# ------------------------------------------------------------- cold / total --

class TestColdVariant:
    def test_massless_cold_equals_total_exactly(self, cosmopower):
        """At Sigma m_nu = 0 the cold and total fields are the same field."""
        from hod_mod.core import growth
        for z in (0.3, 1.0):
            t = float(growth.growth_factor(z, _THETA, pk=cosmopower, variant="total"))
            c = float(growth.growth_factor(z, _THETA, pk=cosmopower, variant="cold"))
            assert c == t

    def test_analytic_modes_warn_on_cold_with_massive_nu(self):
        """Carroll and the ODE have no neutrino sector; say so rather than lie."""
        from hod_mod.core import growth
        th = {**_THETA, "sum_mnu": 0.06}
        growth._WARNED.discard("cold-analytic:ode")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            growth.growth_factor(0.3, th, mode="ode", variant="cold")
        assert len(rec) == 1 and "no neutrino sector" in str(rec[0].message)


# ------------------------------------------------------------ differentiable --

@pytest.mark.x64
class TestDifferentiability:
    def test_grad_omega_m_finite_in_every_mode(self, cosmopower):
        from hod_mod.core import growth
        for mode in ("carroll", "ode"):
            g = float(jax.grad(lambda om: growth.growth_factor(
                0.5, {**_THETA, "Omega_m": om}, mode=mode))(0.31))
            assert np.isfinite(g) and g < 0.0, mode
        g = float(jax.grad(lambda om: growth.growth_pk_ratio(
            0.5, {**_THETA, "Omega_m": om, "Omega_cdm": om - _THETA["Omega_b"]},
            cosmopower))(0.31))
        assert np.isfinite(g) and g < 0.0

    def test_carroll_derivative_is_biased_against_the_ode(self):
        """The headline forecast result: Carroll's dlnD/dOmega_m is several % off."""
        from hod_mod.core import growth
        f = lambda om, mode: jnp.log(growth.growth_factor(
            0.3, {**_THETA, "Omega_m": om}, mode=mode))
        g_car = float(jax.grad(lambda om: f(om, "carroll"))(0.31))
        g_ode = float(jax.grad(lambda om: f(om, "ode"))(0.31))
        assert abs(g_car / g_ode - 1.0) > 0.03

    def test_dark_energy_gradient_exists_only_in_the_ode(self):
        """Carroll's dlnD/dw0 is structurally absent, not merely small."""
        from hod_mod.core import growth
        g = float(jax.grad(lambda w: growth.growth_factor(
            1.0, {**_THETA, "w0": w}, mode="ode"))(-1.0))
        assert np.isfinite(g) and abs(g) > 1e-3
        d_a = float(growth.growth_factor(1.0, {**_THETA, "w0": -0.8}, mode="carroll"))
        d_b = float(growth.growth_factor(1.0, {**_THETA, "w0": -1.2}, mode="carroll"))
        assert d_a == d_b

    def test_jit_and_vmap_safe(self):
        from hod_mod.core import growth
        f = jax.jit(lambda om: growth.growth_factor(0.4, {**_THETA, "Omega_m": om},
                                                    mode="ode"))
        v = jax.vmap(f)(jnp.array([0.28, 0.31, 0.34]))
        assert v.shape == (3,) and np.all(np.isfinite(np.asarray(v)))
        assert float(v[0]) > float(v[2])       # lower Omega_m grows less by z=0.4


# ------------------------------------------------------------ regression pin --

@pytest.mark.x64
class TestGrowthRegressionPin:
    """Hard numeric pins on sigma/dn/dM/b at z > 0.

    Why this class exists: before it, nothing in the suite could see a change of
    growth model.  Every sigma/dndm/bias assertion in ``tests/`` is at z = 0,
    where D(z) = 1 *exactly* in all three modes, so the routes are
    indistinguishable there.  The tests that do reach z > 0
    (``test_lensing_observables``, ``test_forward_matches_production``) rebuild
    their own reference through the same dispatcher, so a uniform change cancels
    on both sides.  The 1-3 % shift in dn/dM that flipping Carroll -> ODE
    produces sailed through the entire suite.

    These values were generated on the CosmoPower backend, x64, Planck-2018 A_s
    convention, with ``sigma()`` receiving the backend object so the growth comes
    from ``mode="pk_ratio"`` -- the backend's own spectra, the same ones the
    2-halo term grows by.  They are *not* independent truth: they are a tripwire.
    If a change moves them, that is not automatically a bug, but it must be a
    change someone intended and recorded in the CHANGELOG.

    They have already earned their keep once.  Generated first with ``sigma()``
    falling back to the growth ODE, they shifted by 2.3e-5 the moment the backend
    object was threaded through -- which is exactly the residual inconsistency
    between the two legs of the halo model that the pk_ratio default exists to
    remove, and which nothing else in the suite could see.
    """

    _M = [1e12, 1e13, 1e14, 1e15]
    _SIGMA = {
        0.3: [1.8101941606, 1.2511042916, 0.7975052615, 0.4573672308],
        1.0: [1.2899214493, 0.8915210844, 0.5682921562, 0.3259141003],
    }
    _DNDM = {
        0.3: [4.152010e-15, 4.861440e-17, 3.298352e-19, 1.836308e-22],
        1.0: [3.745595e-15, 3.096525e-17, 7.347933e-20, 1.020236e-24],
    }
    _BIAS = {
        0.3: [0.9077416000, 1.3363946082, 2.6542657235, 7.8428940966],
        1.0: [1.2862069830, 2.1960513997, 5.0231123892, 16.3133891797],
    }
    #: Massive neutrinos: sigma is NOT sigma(M,0) x D(z) here, because
    #: _cold_factor carries a redshift.  Pinned so that separability cannot be
    #: "restored" as an optimisation without the tripwire firing.
    _SIGMA_MNU06_Z1 = [1.2750066534, 0.8816937357, 0.5624142789, 0.3228242645]

    @pytest.fixture(scope="class")
    def hmf(self, cosmopower):
        from hod_mod.core.halo_mass_function import HaloMassFunction, _RHO_CRIT0
        return HaloMassFunction(cosmopower.pk_linear,
                                rho_mean=0.31 * _RHO_CRIT0, model="tinker08")

    @pytest.mark.parametrize("z", [0.3, 1.0])
    def test_sigma_dndm_bias_pinned(self, hmf, z):
        m = jnp.array(self._M)
        np.testing.assert_allclose(np.asarray(hmf.sigma(m, z, _THETA)),
                                   self._SIGMA[z], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(hmf.dndm(m, z, _THETA)),
                                   self._DNDM[z], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(hmf.bias(m, z, _THETA)),
                                   self._BIAS[z], rtol=1e-6)

    def test_massive_nu_sigma_pinned(self, hmf):
        th = {**_THETA, "sum_mnu": 0.06}
        np.testing.assert_allclose(np.asarray(hmf.sigma(jnp.array(self._M), 1.0, th)),
                                   self._SIGMA_MNU06_Z1, rtol=1e-6)

    def test_carroll_would_move_dndm_by_percent(self, hmf, monkeypatch):
        """The pin has teeth: the old default is percent-level away, not noise."""
        m = jnp.array(self._M)
        monkeypatch.setenv("HOD_MOD_GROWTH", "carroll")
        d_car = np.asarray(hmf.dndm(m, 0.3, _THETA))
        shift = np.max(np.abs(d_car / np.array(self._DNDM[0.3]) - 1.0))
        assert shift > 0.01, (
            f"switching to Carroll moved dn/dM by only {shift:.2%}; if that is "
            "now below a percent the pinned values or the dispatcher changed")

    def test_separability_is_exact_only_when_massless(self, hmf):
        """sigma(M,z)/sigma(M,0) is one scalar massless, and mass-dependent not.

        The claim under test is *mass-independence* of the ratio -- that is what
        licenses a multi-z caller to compute the k-integral once and scale it.
        Deliberately compared against nothing external: the ratio's own spread
        across mass is the whole question, so bringing a D(z) into it would only
        fold in whichever growth route that D came from.  (An earlier version of
        this test did exactly that and measured the 1.5e-5 gap between the ODE
        and pk_ratio instead of measuring separability at all.)

        Massive neutrinos break it because ``_cold_factor`` returns
        S_cb(k,z)/S_m(k,0), and that tilts the spectrum *shape* with redshift,
        not merely its amplitude.
        """
        m = jnp.logspace(11.5, 15.0, 24)
        for th, exact in ((_THETA, True), ({**_THETA, "sum_mnu": 0.06}, False)):
            r = (np.asarray(hmf.sigma(m, 1.0, th))
                 / np.asarray(hmf.sigma(m, 0.0, th)))
            spread = float(r.max() / r.min() - 1.0)
            if exact:
                assert spread < 1e-12, f"massless separability broken: {spread:.2e}"
            else:
                assert spread > 1e-6, (
                    f"massive-nu ratio is mass-independent to {spread:.2e}; if "
                    "_cold_factor stopped carrying z, the multi-z hoisting "
                    "optimisation is back on the table and this test needs "
                    "re-thinking")


class TestNoTracerLeakFromDeferredImport:
    """The module-level grids must be concrete before any jit traces them.

    ``core.growth`` is imported lazily (``halo_mass_function.growth_factor``
    defers it, so that ``core.power_spectrum`` can import ``growth`` in turn).
    If that first import happens *inside* a ``jax.jit`` trace, every ``jnp``
    expression at the module's top level is staged into that trace, and the
    resulting constants escape as leaked tracers the next time they are used.

    That is not hypothetical: at 6b6ccc4 ``growth.py`` declared
    ``_K_INT = jnp.logspace(-4.0, 3.0, 512)`` at module scope, and
    ``ConcentrationModel("bhattacharya13")`` -- whose ``c_bhattacharya13`` is
    jitted and reaches ``growth_factor`` through ``HaloMassFunction.sigma`` --
    died with ``UnexpectedTracerError``, naming that exact line as the origin.
    Three tests in ``tests/test_concentration.py`` failed and nothing else did,
    because it takes a jitted caller *and* a first import to trigger it.

    The grids now live in ``core.amplitude``, which ``core.halo_mass_function``
    imports at module scope, so they are built at package-import time.  These
    assertions pin that: a traced constant is not an ``ndarray``, and the jitted
    concentration path is the caller that actually caught it.
    """

    def test_grids_are_concrete_not_traced(self):
        import jax
        from hod_mod.core import amplitude, growth
        for name, arr in (("amplitude.K_INT", amplitude.K_INT),
                          ("growth._K_INT", growth._K_INT)):
            assert not isinstance(arr, jax.core.Tracer), f"{name} is a tracer"
            assert hasattr(arr, "shape") and arr.shape == (512,)

    def test_jitted_concentration_reaches_growth_without_leaking(self):
        """The exact call that failed at 6b6ccc4."""
        from hod_mod.core.concentration import ConcentrationModel
        from hod_mod.core.halo_mass_function import HaloMassFunction
        from hod_mod.core.power_spectrum import LinearPowerSpectrum

        pk = LinearPowerSpectrum()
        hmf = HaloMassFunction(pk.pk_linear, model="tinker08")
        cm = ConcentrationModel("bhattacharya13", hmf=hmf)
        c = cm.concentration(jnp.logspace(12, 15, 4), z=0.3, theta=_THETA)
        c = np.asarray(c)
        assert np.all(np.isfinite(c)) and np.all(c > 0)
        assert np.all(np.diff(c) < 0), "c(M) must fall with mass"
