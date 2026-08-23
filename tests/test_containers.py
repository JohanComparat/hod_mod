"""Pytree parameter containers.

The point of these is not convenience. A dict of parameters lets a ``float()``
cast silently drop a value out of the computation graph, and when the dropped
parameter is not consumed by the branch taken, ``jax.grad`` returns exactly zero
with no error. These tests assert that the container makes that outcome
structurally impossible: params are leaves and are traced, config lives in the
treedef and cannot be differentiated at all.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from hod_mod.core.containers import Cosmology, ParameterContainer


class TestPytreeProtocol:
    def test_flattens_params_only(self):
        c = Cosmology.planck18()
        leaves = jax.tree_util.tree_leaves(c)
        assert len(leaves) == len(Cosmology._PARAMS)

    def test_config_is_not_a_leaf(self):
        """A static config entry must never appear as a differentiable leaf."""
        c = Cosmology.planck18(transfer="eh98")
        leaves = jax.tree_util.tree_leaves(c)
        assert "eh98" not in leaves
        assert c.transfer == "eh98"

    def test_round_trips_through_flatten(self):
        c = Cosmology.planck18(transfer="camb")
        flat, treedef = jax.tree_util.tree_flatten(c)
        back = jax.tree_util.tree_unflatten(treedef, flat)
        assert back.Omega_m == c.Omega_m
        assert back.transfer == "camb"

    def test_survives_jit(self):
        c = Cosmology.planck18()
        out = jax.jit(lambda cc: cc.Omega_m * cc.sigma8)(c)
        assert float(out) == pytest.approx(0.3100 * 0.8111, rel=1e-6)

    def test_vmap_over_a_batch(self):
        """A container built inside vmap batches over its traced leaves."""
        oms = jnp.linspace(0.28, 0.34, 5)
        f = jax.vmap(lambda om: Cosmology.planck18(Omega_m=om).Omega_m ** 2)
        assert np.allclose(np.asarray(f(oms)), np.asarray(oms) ** 2)


class TestGradients:
    def test_grad_returns_a_container(self):
        c = Cosmology.planck18()
        g = jax.grad(lambda cc: cc.Omega_m ** 2 + 3.0 * cc.sigma8)(c)
        assert isinstance(g, Cosmology)
        assert float(g.Omega_m) == pytest.approx(2 * 0.3100, rel=1e-5)
        assert float(g.sigma8) == pytest.approx(3.0, rel=1e-5)

    def test_unused_param_has_zero_grad_but_is_still_a_leaf(self):
        """Zero here is the *correct* answer, and it is distinguishable.

        The failure this class prevents is a zero that means "left the graph".
        A parameter that is genuinely unused is still a leaf, so the zero can be
        told apart from a missing one by inspecting the tree.
        """
        c = Cosmology.planck18()
        g = jax.grad(lambda cc: cc.Omega_m ** 2)(c)
        assert float(g.sigma8) == 0.0
        assert "sigma8" in [k for k in Cosmology._PARAMS]

    def test_cannot_differentiate_config(self):
        """Config is not a leaf, so grad cannot even be asked for it."""
        c = Cosmology.planck18()
        g = jax.grad(lambda cc: cc.Omega_m ** 2)(c)
        with pytest.raises(AttributeError):
            _ = g.not_a_parameter

    def test_gradient_matches_finite_difference(self):
        def f(om):
            return Cosmology.planck18(Omega_m=om).Omega_m ** 3
        ad = float(jax.grad(f)(0.31))
        fd = (f(0.3101) - f(0.3099)) / 0.0002
        assert ad == pytest.approx(fd, rel=1e-4)


class TestConstruction:
    def test_rejects_unknown_parameter(self):
        with pytest.raises(TypeError):
            Cosmology.planck18(bogus=1.0)

    def test_omega_cdm_defaults_consistently(self):
        c = Cosmology.planck18()
        assert float(c.Omega_cdm) == pytest.approx(0.3100 - 0.0493, rel=1e-9)

    def test_explicit_omega_cdm_is_kept(self):
        c = Cosmology.planck18(Omega_cdm=0.25)
        assert float(c.Omega_cdm) == 0.25

    def test_to_theta_uses_legacy_key_spellings(self):
        """Interop: the rest of the package indexes theta['ln10^{10}A_s']."""
        t = Cosmology.planck18().to_theta()
        assert "ln10^{10}A_s" in t
        assert t["Omega_m"] == 0.3100

    def test_from_dict_ignores_extra_keys(self):
        c = Cosmology.from_dict({"Omega_m": 0.3, "sigma8": 0.8, "h": 0.7,
                                 "n_s": 0.96, "Omega_b": 0.05, "junk": 1})
        assert float(c.Omega_m) == 0.3

    def test_replace_is_a_copy(self):
        c = Cosmology.planck18()
        c2 = c.replace(sigma8=0.75)
        assert float(c.sigma8) == 0.8111 and float(c2.sigma8) == 0.75

    def test_missing_required_parameter_raises(self):
        class Tiny(ParameterContainer):
            _PARAMS = ("a", "b")
        with pytest.raises(TypeError, match="missing required"):
            Tiny(a=1.0)


class TestInteropWithTheExistingStack:
    def test_theta_drives_the_mass_function(self):
        from hod_mod.core.halo_mass_function import HaloMassFunction
        from hod_mod.core.power_spectrum import eisenstein_hu_pk
        hmf = HaloMassFunction(lambda k, z, t: eisenstein_hu_pk(k, t))
        theta = Cosmology.planck18().to_theta()
        theta["sigma8"] = 0.8111
        out = np.asarray(hmf.dndm(jnp.logspace(12, 15, 8), 0.2, theta))
        assert np.all(np.isfinite(out)) and np.all(out > 0)

    def test_theta_drives_mass_translation(self):
        from hod_mod.core.mass_definitions import translate_mass
        theta = Cosmology.planck18().to_theta()
        m, r, c = translate_mass(jnp.array([1e14]), jnp.array([5.0]),
                                 "200m", "500c", 0.2, theta)
        assert float(m[0]) < 1e14 and float(c[0]) < 5.0
