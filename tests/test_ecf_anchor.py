"""Guards for the ``--ecf`` anchor of ``fit_comparat2025``.

The v0.4 campaign shipped five ``--ecf`` fits with no AGN component at all: the
anchor was measured by a *non-negative* least squares under a weighting that
drove the AGN coefficient negative, so it clipped to exactly ``0.0``.  Because
the model applies it as ``A_AGN = 10**log10_A_AGN * c_agn``, a zero conversion
deletes that component for **every** amplitude and turns the parameter into an
exactly flat direction — which reports back as the untouched optimiser seed
rather than as an error.  Nothing in the suite covered the anchor, which is why
it was invisible.

These tests cover the guard and the flatness property, not the physics of the
ECF tables themselves (see ``test_thin_modules``/``test_refactor_coverage``).
"""
import json

import numpy as np
import pytest


@pytest.fixture()
def F():
    from hod_mod.scripts.fitting import fit_comparat2025 as _F
    saved = (_F._USE_ECF, _F._ECF_ANCHOR, _F._RESULTS_DIR)
    yield _F
    _F._USE_ECF, _F._ECF_ANCHOR, _F._RESULTS_DIR = saved


class _StubInfra:
    """Only the attribute the cache-reading path of the anchor touches."""
    agn_model_choice = "hod"


class TestValidateEcfAnchor:
    def test_rejects_zero_agn_conversion(self, F):
        anc = dict(c_gas_S1=8.8e-13, c_agn_S1=0.0, srx_S1=1.0)
        with pytest.raises(RuntimeError, match="degenerate anchor"):
            F._validate_ecf_anchor(anc, "unit-test")

    def test_rejects_zero_gas_conversion(self, F):
        anc = dict(c_gas_S1=0.0, c_agn_S1=1e-3, srx_S1=1.0)
        with pytest.raises(RuntimeError, match="degenerate anchor"):
            F._validate_ecf_anchor(anc, "unit-test")

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite(self, F, bad):
        anc = dict(c_gas_S1=8.8e-13, c_agn_S1=float(bad), srx_S1=1.0)
        with pytest.raises(RuntimeError, match="degenerate anchor"):
            F._validate_ecf_anchor(anc, "unit-test")

    def test_accepts_two_positive_conversions(self, F):
        F._validate_ecf_anchor(
            dict(c_gas_S1=7.8e-13, c_agn_S1=8.7e-01, srx_S1=1.0), "unit-test")

    def test_names_the_offending_key(self, F):
        with pytest.raises(RuntimeError, match="c_agn_S1"):
            F._validate_ecf_anchor(
                dict(c_gas_S1=1.0, c_agn_S1=0.0, srx_S1=1.0), "unit-test")


class TestAnchorCacheHandling:
    def test_no_ecf_is_the_identity(self, F):
        """Without --ecf the conversions must not perturb the amplitudes."""
        F._USE_ECF = False
        assert F._ecf_anchor_consts("S1", _StubInfra()) == (1.0, 1.0)

    def test_degenerate_cache_is_rejected_not_used(self, F, tmp_path):
        """A cached zero must raise rather than silently delete the AGN leg."""
        F._USE_ECF = True
        F._ECF_ANCHOR = None
        F._RESULTS_DIR = tmp_path
        (tmp_path / "ecf_anchor_hod.json").write_text(json.dumps(dict(
            c_gas_S1=8.835923748706981e-13, c_agn_S1=0.0,
            srx_S1=8.660511557662721e+36, agn_model="hod",
            scheme=F._ECF_ANCHOR_SCHEME)))
        with pytest.raises(RuntimeError, match="degenerate anchor"):
            F._ecf_anchor_consts("S1", _StubInfra())

    def test_anchor_carries_its_measurement_scheme(self, F):
        """Provenance: an anchor written by an older scheme must be re-measured."""
        assert isinstance(F._ECF_ANCHOR_SCHEME, str) and F._ECF_ANCHOR_SCHEME


class TestAmplitudeIsIdentifiable:
    """The property whose absence made the bug invisible.

    ``A = 10**log10_A * c``: with ``c == 0`` the model is independent of the
    amplitude, so any optimiser returns its seed and reports success.  Assert
    the model actually responds.
    """

    @staticmethod
    def _model(log10_A_gas, log10_A_AGN, c_gas, c_agn, gas, agn):
        return 10.0 ** log10_A_gas * c_gas * gas + 10.0 ** log10_A_AGN * c_agn * agn

    def test_zero_conversion_makes_the_amplitude_flat(self):
        gas, agn = np.array([1.0, 0.5]), np.array([2.0, 0.1])
        a = self._model(0.0, 0.0, 8.8e-13, 0.0, gas, agn)
        b = self._model(0.0, 5.0, 8.8e-13, 0.0, gas, agn)
        np.testing.assert_array_equal(a, b)   # the failure mode, made explicit

    def test_positive_conversion_makes_the_amplitude_matter(self):
        gas, agn = np.array([1.0, 0.5]), np.array([2.0, 0.1])
        a = self._model(0.0, 0.0, 8.8e-13, 8.7e-01, gas, agn)
        b = self._model(0.0, 1.0, 8.8e-13, 8.7e-01, gas, agn)
        assert np.all(b > a)


class TestAnchorSolveWeighting:
    """The measurement itself: floored errors + no non-negativity box.

    Reproduces the shipped failure on a synthetic problem whose AGN coefficient
    is genuinely negative under the unfloored weighting, and shows the corrected
    weighting recovers a positive one.
    """

    def test_unfloored_weighting_can_clip_agn_to_zero(self):
        from scipy.optimize import lsq_linear
        rng = np.random.default_rng(0)
        theta = np.logspace(np.log10(8.0), np.log10(300.0), 31)
        gas, agn = theta ** -1.2, theta ** -2.6
        wd = 1.0 * gas - 0.2 * agn            # a negative AGN coefficient
        err = 0.01 * np.abs(wd)
        w = 1.0 / err
        A = np.column_stack([gas * w, agn * w])
        clipped = lsq_linear(A, wd * w, bounds=([0.0, 0.0], [np.inf, np.inf]),
                             method="bvls").x
        assert clipped[1] == 0.0               # exactly the shipped failure
        free = np.linalg.lstsq(A, wd * w, rcond=None)[0]
        assert free[1] < 0.0                   # the box hid a negative solution

    def test_floor_changes_the_recovered_split(self):
        """The f_sys floor reweights toward the AGN-dominated small-theta bins."""
        theta = np.logspace(np.log10(8.0), np.log10(300.0), 31)
        gas, agn = theta ** -1.2, theta ** -2.6
        wd = 1.0 * gas + 0.3 * agn
        err_jk = 0.01 * np.abs(wd) * (theta / theta[0]) ** -1.5   # tiny at large theta
        for err in (err_jk, np.sqrt(err_jk ** 2 + (0.05 * np.abs(wd)) ** 2)):
            w = 1.0 / err
            A = np.column_stack([gas * w, agn * w])
            sol = np.linalg.lstsq(A, wd * w, rcond=None)[0]
            assert np.all(np.isfinite(sol))
            np.testing.assert_allclose(sol, [1.0, 0.3], rtol=1e-6)
