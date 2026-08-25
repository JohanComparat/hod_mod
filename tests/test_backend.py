"""The flavour registry, and the promises it refuses to make.

``Backend`` replaces a ``pk_backend: str`` constructor flag that six call sites
re-derived independently with ``getattr(fhmp, "_pk_backend", "camb") ==
"eh98_jax"``.  A string flag cannot be validated; an object can, and these are
the checks that matter:

* ``ACCURATE`` must keep the numerical settings that were hard-coded before the
  registry existed, or selecting it would silently move every number;
* a backend that claims ``traced=True`` must actually be differentiable in all
  of its parts — the failure mode otherwise is a gradient that succeeds and is
  quietly missing a term.
"""
import os

import pytest

from hod_mod.core import backend as B


class TestRegistry:
    def test_names_are_unique_and_self_describing(self):
        assert set(B.BACKENDS) == {"accurate", "fast", "reference"}
        for name, b in B.BACKENDS.items():
            assert b.name == name

    def test_frozen(self):
        with pytest.raises(Exception):
            B.ACCURATE.name = "nope"

    def test_resolve_precedence(self, monkeypatch):
        monkeypatch.delenv(B.ENV_VAR, raising=False)
        assert B.resolve_backend().name == "accurate"
        monkeypatch.setenv(B.ENV_VAR, "fast")
        assert B.resolve_backend().name == "fast"
        # An explicit argument beats the environment.
        assert B.resolve_backend("accurate").name == "accurate"
        assert B.resolve_backend(B.FAST) is B.FAST

    def test_resolve_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown backend"):
            B.resolve_backend("halofit-ish")


class TestAccurateReproducesThePreRegistryDefaults:
    """These constants were hard-coded; selecting ACCURATE must change nothing."""

    def test_k_grid_matches_full_halo_model_prediction(self):
        import inspect
        from hod_mod.observables.clustering import FullHaloModelPrediction
        sig = inspect.signature(FullHaloModelPrediction.__init__).parameters
        assert B.ACCURATE.n_k == sig["n_k"].default
        assert B.ACCURATE.k_min == sig["k_min"].default
        assert B.ACCURATE.k_max == sig["k_max"].default

    def test_mass_grid_matches_hod_base(self):
        from hod_mod.connection.hod.base import HODBase
        assert B.ACCURATE.n_m == HODBase._N_M_GRID

    def test_ogata_nodes_match_the_hankel_table(self):
        from hod_mod.observables import clustering
        assert B.ACCURATE.n_ogata == clustering._OG_N

    def test_gas_quadrature_matches_gas_density(self):
        import inspect
        from hod_mod.gas.density import GasDensityDPM
        sig = inspect.signature(GasDensityDPM.__init__).parameters
        assert B.ACCURATE.n_gl == sig["n_gl"].default

    def test_delta_sigma_los_matches(self):
        import inspect
        from hod_mod.observables.clustering import FullHaloModelPrediction
        sig = inspect.signature(FullHaloModelPrediction.delta_sigma).parameters
        assert B.ACCURATE.chi_max == sig["chi_max"].default
        assert B.ACCURATE.n_chi == sig["n_chi"].default
        assert B.ACCURATE.n_r_tab == sig["n_R_tab"].default


class TestTracedIsAPromiseThatIsChecked:
    def test_fast_is_traced_and_accurate_is_not(self):
        assert B.FAST.traced and not B.ACCURATE.traced and not B.REFERENCE.traced

    @pytest.mark.parametrize("override", [
        {"cm_model": "diemer19"},   # colossus: numpy + global state
        {"pk": "camb"},             # Fortran
        {"pk": "class"},            # C
        {"gas_ft": "gl"},           # numpy Gauss-Legendre
    ])
    def test_traced_backend_refuses_a_non_differentiable_part(self, override):
        with pytest.raises(ValueError, match="traced"):
            B.FAST.with_(**override)

    def test_marking_accurate_as_traced_is_refused(self):
        """The dangerous direction: a gradient that runs but omits dP/dtheta."""
        with pytest.raises(ValueError, match="traced"):
            B.ACCURATE.with_(traced=True)


class TestGridValidation:
    @pytest.mark.parametrize("override", [
        {"k_min": 0.0}, {"k_max": 1e-5}, {"m_min": 0.0}, {"m_max": 1.0},
    ])
    def test_empty_ranges_are_refused(self, override):
        with pytest.raises(ValueError, match="empty"):
            B.ACCURATE.with_(**override)

    @pytest.mark.parametrize("field", ["n_k", "n_m", "n_chi", "n_pi",
                                       "n_r_tab", "n_ogata", "n_gl"])
    def test_degenerate_resolutions_are_refused(self, field):
        with pytest.raises(ValueError, match=field):
            B.ACCURATE.with_(**{field: 1})
