"""Massive neutrinos: the density budget, the canonical key, and the cold field.

Three things are guarded here.

*The budget closes.*  ``Omega_b + Omega_cdm + Omega_nu == Omega_m`` exactly.
Before Sigma m_nu became a first-class parameter the default dictionary carried
``Omega_cdm = 0.2607`` with ``Omega_b + Omega_cdm == Omega_m`` -- i.e. no room
for neutrinos -- while CAMB silently added its own 0.06 eV on top, so the
reference spectrum ran at Omega_m ~ 0.3114 against a package that used 0.3100
everywhere else.

*There is one key.*  ``sum_mnu`` is canonical.  ``mnu`` used to be read by four
emulator paths and written by none, so CSST ran at 0.06 eV and Aemulus-nu at
0.0 eV whatever the user asked for.

*The massless limit is untouched.*  Every correction here is exactly the
identity at Sigma m_nu = 0, so runs that do not ask for neutrinos are
bit-identical to what they were.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax

from hod_mod.core.power_spectrum import (
    LinearPowerSpectrum, NU_DENOM_EV, mnu_sum, omega_nu, omega_cb, omega_cdm)

classy = pytest.importorskip  # re-exported for the skipif below


def _has_classy():
    try:
        import classy  # noqa: F401
        return True
    except ImportError:
        return False


needs_classy = pytest.mark.skipif(not _has_classy(), reason="classy not installed")


# ---------------------------------------------------------------- budget --

class TestDensityBudget:
    def test_default_cosmology_closes(self):
        th = LinearPowerSpectrum.default_cosmology()
        assert th["sum_mnu"] == pytest.approx(0.06)
        total = th["Omega_b"] + th["Omega_cdm"] + omega_nu(th)
        assert total == pytest.approx(th["Omega_m"], abs=1e-12)

    def test_omega_nu_closure_relation(self):
        th = LinearPowerSpectrum.default_cosmology()
        assert omega_nu(th) == pytest.approx(
            th["sum_mnu"] / (NU_DENOM_EV * th["h"] ** 2), rel=1e-12)
        assert omega_cb(th) == pytest.approx(th["Omega_m"] - omega_nu(th), rel=1e-12)

    def test_omega_cdm_derives_when_absent(self):
        th = dict(LinearPowerSpectrum.default_cosmology())
        explicit = th.pop("Omega_cdm")
        assert omega_cdm(th) == pytest.approx(explicit, rel=1e-12)

    def test_camb_internal_budget_matches_the_dict(self):
        """CAMB's own Omega_m must equal the dictionary's, not exceed it."""
        camb = pytest.importorskip("camb")
        lp = LinearPowerSpectrum()
        th = lp.default_cosmology()
        pars = lp._camb_results(0.0, th).Params
        om = (pars.omnuh2 + pars.ombh2 + pars.omch2) / th["h"] ** 2
        assert om == pytest.approx(th["Omega_m"], abs=1e-4)
        assert pars.omnuh2 > 0.0                       # the mass reached CAMB

    def test_container_budget_closes(self):
        from hod_mod.core.containers import Cosmology
        for mnu in (0.0, 0.06, 0.3):
            c = Cosmology.planck18(sum_mnu=mnu)
            t = c.to_theta()
            assert "sum_mnu" in t                      # to_theta used to drop it
            assert t["sum_mnu"] == pytest.approx(mnu)
            assert (t["Omega_b"] + t["Omega_cdm"]
                    + mnu / (NU_DENOM_EV * t["h"] ** 2)) == pytest.approx(
                        t["Omega_m"], abs=1e-12)

    def test_forecast_cosmo_budget_closes(self):
        from hod_mod.forecast.forward_jax import ForwardModel
        from hod_mod.forecast import params as P
        c = ForwardModel()._cosmo(P.fiducial_vector())
        total = float(c["Omega_b"] + c["Omega_cdm"]
                      + c["sum_mnu"] / (NU_DENOM_EV * c["h"] ** 2))
        assert total == pytest.approx(float(c["Omega_m"]), abs=1e-12)
        assert float(c["sum_mnu"]) == pytest.approx(0.06)


# ------------------------------------------------------------- one key ----

class TestCanonicalKey:
    def test_sum_mnu_is_read(self):
        assert mnu_sum({"sum_mnu": 0.12}) == 0.12

    def test_missing_key_is_massless(self):
        assert mnu_sum({"h": 0.7}) == 0.0

    def test_mnu_is_a_deprecated_alias(self):
        import hod_mod.core.power_spectrum as ps
        ps._WARNED_MNU_ALIAS = False
        with pytest.warns(DeprecationWarning):
            assert mnu_sum({"mnu": 0.09}) == 0.09

    def test_sum_mnu_wins_over_the_alias(self):
        assert mnu_sum({"sum_mnu": 0.06, "mnu": 0.3}) == 0.06


# ------------------------------------------------------- the cold field ---

class TestColdSector:
    @pytest.fixture(scope="class")
    def hmf(self):
        from hod_mod.core.halo_mass_function import make_hmf
        from hod_mod.core.power_spectrum import default_pk_linear
        pk = default_pk_linear("cosmopower")
        return make_hmf("tinker08", pk_func=pk.as_hmf_pk_func())

    @pytest.fixture(scope="class")
    def theta(self):
        return dict(LinearPowerSpectrum.default_cosmology(), sigma8=0.8111)

    def test_massless_is_bit_identical_to_no_key(self, hmf, theta):
        m = jnp.logspace(11, 15, 12)
        no_key = {k: v for k, v in theta.items() if k != "sum_mnu"}
        a = np.asarray(hmf.dndm(m, 0.0, no_key))
        b = np.asarray(hmf.dndm(m, 0.0, dict(theta, sum_mnu=0.0)))
        np.testing.assert_array_equal(a, b)

    def test_rho_cold_is_below_rho_matter(self, hmf, theta):
        from hod_mod.core.power_spectrum import rho_critical_0
        rho_m = theta["Omega_m"] * rho_critical_0()
        assert float(hmf._rho_cold(theta)) < rho_m
        assert float(hmf._rho_cold(theta)) == pytest.approx(
            omega_cb(theta) * rho_critical_0(), rel=1e-12)

    def test_cold_spectrum_exceeds_total_at_fixed_sigma8(self, hmf, theta):
        """P_cb is less suppressed than P_m, so sigma_cb > sigma_m."""
        m = jnp.logspace(12, 15, 8)
        no_key = {k: v for k, v in theta.items() if k != "sum_mnu"}
        s_nu = np.asarray(hmf.sigma(m, 0.0, theta))
        s_0 = np.asarray(hmf.sigma(m, 0.0, no_key))
        assert np.all(s_nu > s_0)
        assert np.all(s_nu / s_0 < 1.01)          # a sub-percent effect at 0.06 eV

    def test_hmf_responds_to_the_mass(self, hmf, theta):
        m = jnp.logspace(11, 15, 12)
        light = np.asarray(hmf.dndm(m, 0.0, dict(theta, sum_mnu=0.0)))
        heavy = np.asarray(hmf.dndm(m, 0.0, dict(theta, sum_mnu=0.30)))
        assert np.abs(heavy / light - 1).max() > 0.02

    def test_dndm_differentiable_in_sum_mnu(self, hmf, theta):
        g = jax.grad(lambda mn: jnp.log(
            hmf.dndm(jnp.array([1e14]), 0.0, dict(theta, sum_mnu=mn))[0]))(0.06)
        assert np.isfinite(float(g)) and float(g) != 0.0


# ------------------------------------------------- the distilled table ----

class TestNuRatioTable:
    def test_massless_row_is_exactly_one(self):
        from hod_mod.core.nu_ratio import suppression_m, suppression_cb
        th = dict(LinearPowerSpectrum.default_cosmology(), sum_mnu=0.0)
        k = jnp.logspace(-3, 1, 32)
        np.testing.assert_array_equal(np.asarray(suppression_m(k, th)), 1.0)
        np.testing.assert_array_equal(np.asarray(suppression_cb(k, th)), 1.0)

    def test_asymptote_is_near_minus_eight_f_nu(self):
        """The classic -8 f_nu small-scale limit, to within the ~20% the
        first-order result is good to."""
        from hod_mod.core.nu_ratio import suppression_m
        th = dict(LinearPowerSpectrum.default_cosmology(), sum_mnu=0.30)
        f_nu = omega_nu(th) / th["Omega_m"]
        s = float(suppression_m(jnp.array([5.0]), th)[0])
        assert -8.0 * f_nu * 1.25 < (s - 1.0) < -8.0 * f_nu * 0.8

    def test_transition_is_not_saturated_at_large_scales(self):
        """The regression for the retuned suppression.

        The closed tanh form this replaced was centred on k_nr ~ 0.0014 h/Mpc
        and was already >90% saturated by k = 0.01, where CLASS puts the
        suppression at about a third of its asymptote.
        """
        from hod_mod.core.nu_ratio import suppression_m
        th = dict(LinearPowerSpectrum.default_cosmology(), sum_mnu=0.30)
        s_lo = float(suppression_m(jnp.array([0.01]), th)[0]) - 1.0
        s_hi = float(suppression_m(jnp.array([1.0]), th)[0]) - 1.0
        assert s_lo / s_hi < 0.45           # tanh form gave ~0.82 here

    def test_cold_is_less_suppressed_than_total(self):
        from hod_mod.core.nu_ratio import suppression_m, suppression_cb
        th = dict(LinearPowerSpectrum.default_cosmology(), sum_mnu=0.30)
        k = jnp.logspace(-2, 1, 24)
        assert np.all(np.asarray(suppression_cb(k, th))
                      > np.asarray(suppression_m(k, th)))

    def test_table_survives_being_first_touched_inside_a_trace(self):
        """Regression: the cache must not hold JAX arrays.

        ``load()`` used to stash ``jnp`` arrays. If the first call happened
        inside a ``jax.jit`` trace -- which is exactly what happens when the
        emulator is jitted before anything else touches the table -- the cached
        arrays carried that trace with them, and the next transformation died
        with ``UnexpectedTracerError``. Ordering matters here: jit first, then
        grad.
        """
        import hod_mod.core.nu_ratio as nr
        from hod_mod.forecast.pk_cosmopower import CosmoPowerJaxPkLinear
        nr._TABLE = None                       # force a cold load inside the trace
        th = dict(LinearPowerSpectrum.default_cosmology(), sigma8=0.8111)
        k = jnp.logspace(-3, 1, 64)
        cp = CosmoPowerJaxPkLinear()
        jitted = jax.jit(lambda t: cp.pk_shape(k, t))
        assert np.isfinite(float(jnp.sum(jitted(th))))
        g = jax.jit(jax.grad(
            lambda ns: jnp.sum(cp.pk_shape(k, {**th, "n_s": ns}))))
        assert np.isfinite(float(g(0.9649)))
        assert not isinstance(nr.load()["lnr_m"], jnp.ndarray), (
            "the table cache must hold numpy, not JAX arrays")

    def test_differentiable_in_sum_mnu(self):
        from hod_mod.core.nu_ratio import suppression_m
        th = LinearPowerSpectrum.default_cosmology()
        k = jnp.logspace(-3, 1, 16)
        g = jax.grad(lambda mn: jnp.sum(
            suppression_m(k, dict(th, sum_mnu=mn))))(0.06)
        assert np.isfinite(float(g)) and float(g) < 0.0

    @needs_classy
    def test_table_matches_a_live_class_solve(self):
        from hod_mod.core.power_spectrum import ClassPkLinear
        from hod_mod.core.nu_ratio import suppression_m, suppression_cb
        cl = ClassPkLinear()
        th = LinearPowerSpectrum.default_cosmology()
        k = jnp.logspace(-3, 1, 60)
        try:
            th0 = dict(th, sum_mnu=0.0)
            th3 = dict(th, sum_mnu=0.30)
            th0.pop("Omega_cdm"); th3.pop("Omega_cdm")
            p0 = np.asarray(cl.pk_linear(k, 0.0, th0))
            live_m = np.asarray(cl.pk_linear(k, 0.0, th3)) / p0
            live_cb = np.asarray(cl.pk_linear_cb(k, 0.0, th3)) / p0
        finally:
            cl.clear_cache()
        assert np.abs(np.asarray(suppression_m(k, th3)) / live_m - 1).max() < 5e-3
        assert np.abs(np.asarray(suppression_cb(k, th3)) / live_cb - 1).max() < 5e-3


# ------------------------------------------------------- CLASS backend ----

@needs_classy
class TestClassBackend:
    @pytest.fixture(scope="class")
    def cl(self):
        from hod_mod.core.power_spectrum import ClassPkLinear
        c = ClassPkLinear()
        yield c
        c.clear_cache()

    def test_selected_by_name(self):
        from hod_mod.core.power_spectrum import default_pk_linear, ClassPkLinear
        assert isinstance(default_pk_linear("class"), ClassPkLinear)

    def test_unknown_backend_raises(self):
        from hod_mod.core.power_spectrum import default_pk_linear
        with pytest.raises(ValueError, match="cosmopower"):
            default_pk_linear("nope")

    def test_agrees_with_camb_in_shape(self, cl):
        """Two independent Boltzmann solvers at matched N_eff and Sigma m_nu."""
        pytest.importorskip("camb")
        th = cl.default_cosmology()
        k = jnp.logspace(-3, 1, 200)
        i0 = int(np.argmin(np.abs(np.asarray(k) - 0.05)))
        nrm = lambda p: np.asarray(p) / np.asarray(p)[i0]
        r = nrm(cl.pk_linear(k, 0.0, th)) / nrm(LinearPowerSpectrum().pk_linear(k, 0.0, th))
        assert np.abs(r - 1).max() < 0.01

    def test_cold_equals_total_when_massless(self, cl):
        th = dict(cl.default_cosmology(), sum_mnu=0.0)
        th.pop("Omega_cdm")
        k = jnp.logspace(-3, 1, 32)
        np.testing.assert_allclose(np.asarray(cl.pk_linear_cb(k, 0.0, th)),
                                   np.asarray(cl.pk_linear(k, 0.0, th)), rtol=0)

    def test_cold_exceeds_total_when_massive(self, cl):
        th = dict(cl.default_cosmology(), sum_mnu=0.30)
        th.pop("Omega_cdm")
        k = jnp.logspace(-1, 1, 24)
        assert np.all(np.asarray(cl.pk_linear_cb(k, 0.0, th))
                      > np.asarray(cl.pk_linear(k, 0.0, th)))

    def test_cache_reuses_one_solve(self, cl):
        """A second redshift must not pay for another CLASS solve."""
        import time
        th = cl.default_cosmology()
        k = jnp.logspace(-3, 1, 64)
        cl.clear_cache()
        t0 = time.perf_counter(); cl.pk_linear(k, 0.0, th); first = time.perf_counter() - t0
        t0 = time.perf_counter(); cl.pk_linear(k, 1.0, th); second = time.perf_counter() - t0
        assert second < first / 10.0


# ------------------------------------------------- the ratio-table bug ----

def test_camb_ratio_nu_row_is_not_the_eh98_cancellation():
    """Regression for the sum_mnu row of the CAMB/EH98 ratio table.

    ``build()`` used to hand CAMB a ``"mnu"`` key that ``LinearPowerSpectrum``
    never read, so both finite-difference steps ran at the same neutrino mass
    and the row came out as exactly minus the EH98 suppression -- cancelling
    the Sigma m_nu response of the shape instead of correcting it.  The
    signature was ``max|dlnR/dSigma m_nu| = 8 f_nu / Sigma m_nu``.
    """
    from hod_mod.forecast.pk_camb_ratio import load
    tab = load()
    names = [str(n) for n in np.asarray(tab["names"])]
    row = np.asarray(tab["dlnr"])[names.index("sum_mnu")]
    th = LinearPowerSpectrum.default_cosmology()
    cancellation = 8.0 / NU_DENOM_EV / (th["Omega_m"] * th["h"] ** 2)
    assert np.abs(row).max() < 0.25 * cancellation
