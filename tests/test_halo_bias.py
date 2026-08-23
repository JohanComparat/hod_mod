"""Halo bias models and the mass--bias consistency integral.

The peak-background split ties a multiplicity function to a bias relation. This
module checks that the pairings shipped by ``hod_mod`` respect that tie, and --
more usefully -- records how badly the *ad hoc* pairings violate it, so a user
combining an arbitrary ``f(sigma)`` with an arbitrary ``b(nu)`` can see what it
costs.

The diagnostic is ``bias_consistency``, evaluated in peak height. See its
docstring for why mass space is the wrong domain for this test.
"""
import numpy as np
import pytest
import jax.numpy as jnp

from hod_mod.core.halo_mass_function import (
    _BIAS_MODELS, _FSIGMA_MODELS, bias_consistency, make_bias,
    matched_bias_for, HaloMassFunction,
)
from hod_mod.core.power_spectrum import LinearPowerSpectrum, eisenstein_hu_pk

# A multiplicity function whose mass normalisation is far from unity is not
# normalisable over the integration range; <b> is then not a meaningful
# statement about the bias model, so the assertions skip it. The value is still
# recorded by scripts/make_accuracy_budget.py.
_NORM_OK = (0.8, 1.4)


class TestBiasModels:
    _NU = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])

    @pytest.mark.parametrize("name", sorted(_BIAS_MODELS))
    def test_positive_and_finite(self, name):
        b = np.asarray(make_bias(name)(self._NU, 200.0))
        assert np.all(np.isfinite(b)), f"{name} produced non-finite bias"
        assert np.all(b > 0), f"{name} produced non-positive bias"

    @pytest.mark.parametrize("name", sorted(_BIAS_MODELS))
    def test_monotonic_above_the_turnover(self, name):
        """Rarer peaks are more biased, for nu above each fit's turnover.

        Every one of these functional forms has a shallow minimum at low nu and
        rises again below it -- an artefact of extrapolating a fit calibrated on
        resolved haloes down towards nu -> 0, not physics. The turnover sits at
        nu = 0.18 (Tinker10), 0.23 (Sheth99), 0.36 (Sheth01) and 0.55
        (Bhattacharya11), so monotonicity is asserted above 0.6 and the
        turnover location is guarded separately below.
        """
        nu = jnp.linspace(0.6, 6.0, 200)
        b = np.asarray(make_bias(name)(nu, 200.0))
        assert np.all(np.diff(b) > 0), f"{name} is not monotonic above nu = 0.6"

    @pytest.mark.parametrize("name", sorted(_BIAS_MODELS))
    def test_turnover_is_below_the_asserted_range(self, name):
        """The low-nu minimum must stay below 0.6, or the test above is vacuous."""
        nu = jnp.linspace(0.05, 2.0, 400)
        b = np.asarray(make_bias(name)(nu, 200.0))
        nu_min = float(nu[int(np.argmin(b))])
        assert nu_min < 0.6, (
            f"{name} turns over at nu = {nu_min:.3f}, inside the range where "
            "monotonicity is asserted"
        )

    @pytest.mark.parametrize("name", sorted(_BIAS_MODELS))
    def test_crosses_unity_near_nu_one(self, name):
        """b(nu) passes through ~1 at the characteristic mass, nu = 1."""
        b1 = float(make_bias(name)(jnp.array([1.0]), 200.0)[0])
        assert 0.7 < b1 < 1.3, f"{name}: b(nu=1) = {b1:.3f}, expected near unity"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="bias model must be one of"):
            make_bias("not_a_model")


class TestConsistencyIntegral:
    """int f(sigma) b(nu) dln nu / int f(sigma) dln nu should be 1."""

    def test_reference_pair_is_exact(self):
        """Sheth-Tormen with its own PBS bias is the calibration point.

        If this drifts, the integral itself has broken -- not a model.
        """
        r = bias_consistency("sheth99", "sheth99")
        assert 0.9 < r["mass_norm"] < 1.05
        assert r["deviation"] < 0.02, f"<b> = {r['mean_bias']:.4f}"

    @pytest.mark.parametrize(
        "fsigma", [m for m in sorted(_FSIGMA_MODELS) if matched_bias_for(m)]
    )
    def test_matched_pairs_are_consistent(self, fsigma):
        bias = matched_bias_for(fsigma)
        r = bias_consistency(fsigma, bias)
        if not (_NORM_OK[0] < r["mass_norm"] < _NORM_OK[1]):
            pytest.skip(
                f"{fsigma} is not normalisable over the range "
                f"(mass_norm = {r['mass_norm']:.3f}); <b> is not meaningful"
            )
        assert r["deviation"] < 0.05, (
            f"{fsigma}+{bias}: <b> = {r['mean_bias']:.4f}, "
            f"deviation {r['deviation']:.4f} exceeds 5%"
        )

    @pytest.mark.parametrize(
        "fsigma", [m for m in sorted(_FSIGMA_MODELS)
                   if matched_bias_for(m) and matched_bias_for(m) != "tinker10"]
    )
    def test_matched_beats_mismatched(self, fsigma):
        """The PBS partner must be closer to unity than the generic default.

        This is the regression guard on ``HaloMassFunction`` defaulting to the
        matched bias: if a future edit makes the matched pairing worse, the
        default is no longer justified.
        """
        matched = bias_consistency(fsigma, matched_bias_for(fsigma))
        generic = bias_consistency(fsigma, "tinker10")
        if not (_NORM_OK[0] < matched["mass_norm"] < _NORM_OK[1]):
            pytest.skip(f"{fsigma} is not normalisable over the range")
        assert matched["deviation"] < generic["deviation"], (
            f"{fsigma}: matched bias {matched_bias_for(fsigma)} gives "
            f"{matched['deviation']:.4f}, generic tinker10 gives "
            f"{generic['deviation']:.4f} -- the default is not justified"
        )

    def test_unknown_fsigma_raises(self):
        with pytest.raises(ValueError, match="unknown multiplicity model"):
            bias_consistency("not_a_model", "tinker10")


class TestHaloMassFunctionWiring:
    @staticmethod
    def _hmf(**kw):
        return HaloMassFunction(lambda k, z, t: eisenstein_hu_pk(k, t), **kw)

    def test_default_is_the_matched_partner(self):
        assert self._hmf(model="sheth99").bias_model == "sheth99"
        assert self._hmf(model="bhattacharya11").bias_model == "bhattacharya11"

    def test_default_falls_back_to_tinker10(self):
        """A model with no implemented partner keeps the historical default."""
        assert matched_bias_for("despali16") is None
        assert self._hmf(model="despali16").bias_model == "tinker10"

    def test_tinker08_default_is_unchanged(self):
        """The library default pairing must not have moved."""
        assert self._hmf(model="tinker08").bias_model == "tinker10"

    def test_explicit_choice_overrides(self):
        assert self._hmf(model="sheth99", bias_model="tinker10").bias_model == "tinker10"

    def test_bias_values_track_the_choice(self):
        theta = LinearPowerSpectrum.default_cosmology()
        theta["sigma8"] = 0.8111
        m = jnp.logspace(12, 15, 12)
        b_st = np.asarray(self._hmf(model="sheth99", bias_model="sheth99").bias(m, 0.2, theta))
        b_tk = np.asarray(self._hmf(model="sheth99", bias_model="tinker10").bias(m, 0.2, theta))
        assert np.all(np.isfinite(b_st)) and np.all(np.isfinite(b_tk))
        assert not np.allclose(b_st, b_tk), "bias_model had no effect"
