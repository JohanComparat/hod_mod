"""Spherical-overdensity mass definitions and NFW translation between them."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from hod_mod.core.mass_definitions import (
    MassDef, translate_mass, parse_mass_def, rho_reference,
)

THETA = {"Omega_m": 0.3100}
Z = 0.2
M = jnp.array([1e12, 1e13, 1e14, 1e15])
C = jnp.array([8.0, 6.0, 5.0, 4.0])


class TestMassDef:
    @pytest.mark.parametrize("name,delta,rho", [
        ("200m", 200.0, "matter"), ("200c", 200.0, "critical"),
        ("500c", 500.0, "critical"), ("2500c", 2500.0, "critical"),
        ("vir", "vir", "critical"),
    ])
    def test_parse(self, name, delta, rho):
        d = MassDef.from_string(name)
        assert d.delta == delta and d.rho_type == rho

    @pytest.mark.parametrize("bad", ["200", "abc", "200x", ""])
    def test_parse_rejects_junk(self, bad):
        with pytest.raises(ValueError):
            MassDef.from_string(bad)

    def test_rejects_bad_rho_type(self):
        with pytest.raises(ValueError, match="rho_type"):
            MassDef(200, "dark_energy")

    def test_rejects_nonpositive_delta(self):
        with pytest.raises(ValueError, match="positive"):
            MassDef(-200, "matter")

    def test_equality_and_repr(self):
        assert MassDef.from_string("500c") == MassDef(500, "critical")
        assert MassDef.from_string("200m") != MassDef(200, "critical")
        assert repr(MassDef(500, "critical")) == "MassDef(500c)"
        assert repr(MassDef("vir")) == "MassDef(vir)"

    def test_200m_reference_is_redshift_independent(self):
        """In comoving h-units the mean-density reference does not evolve."""
        d = MassDef(200, "matter")
        _, r0 = d.delta_rho(0.0, THETA)
        _, r2 = d.delta_rho(2.0, THETA)
        assert float(r0) == pytest.approx(float(r2), rel=1e-12)

    def test_virial_matches_bryan_norman(self):
        delta, _ = rho_reference("vir", Z, THETA)
        om = THETA["Omega_m"]
        ez2 = om * (1 + Z) ** 3 + (1 - om)
        x = om * (1 + Z) ** 3 / ez2 - 1.0
        assert float(delta) == pytest.approx(18 * np.pi**2 + 82 * x - 39 * x**2, rel=1e-10)

    def test_virial_tends_to_18pi2_in_eds(self):
        delta, _ = rho_reference("vir", 0.0, {"Omega_m": 1.0})
        assert float(delta) == pytest.approx(18 * np.pi**2, rel=1e-10)


class TestTranslate:
    def test_identity_is_exact(self):
        m, r, c = translate_mass(M, C, "200m", "200m", Z, THETA)
        assert np.allclose(np.asarray(m), np.asarray(M), rtol=0, atol=0)
        assert np.allclose(np.asarray(c), np.asarray(C), rtol=0, atol=0)

    @pytest.mark.parametrize("other", ["200c", "500c", "vir", "2500c"])
    def test_round_trip(self, other):
        m1, _, c1 = translate_mass(M, C, "200m", other, Z, THETA)
        m2, _, c2 = translate_mass(m1, c1, other, "200m", Z, THETA)
        assert np.allclose(np.asarray(m2), np.asarray(M), rtol=1e-4)
        assert np.allclose(np.asarray(c2), np.asarray(C), rtol=1e-4)

    def test_mass_ordering(self):
        """Higher overdensity encloses less mass: M_2500c < M_500c < M_200c < M_200m."""
        m200m = np.asarray(M)
        m200c = np.asarray(translate_mass(M, C, "200m", "200c", Z, THETA)[0])
        m500c = np.asarray(translate_mass(M, C, "200m", "500c", Z, THETA)[0])
        m2500 = np.asarray(translate_mass(M, C, "200m", "2500c", Z, THETA)[0])
        assert np.all(m2500 < m500c) and np.all(m500c < m200c) and np.all(m200c < m200m)

    def test_virial_between_200c_and_200m(self):
        """Delta_vir ~ 100-180 relative to critical, so M_vir exceeds M_200c."""
        mvir = np.asarray(translate_mass(M, C, "200m", "vir", Z, THETA)[0])
        m200c = np.asarray(translate_mass(M, C, "200m", "200c", Z, THETA)[0])
        assert np.all(mvir > m200c)

    def test_concentration_shrinks_with_overdensity(self):
        """r_s is fixed by the profile, so c = r_delta/r_s falls as delta rises."""
        c200c = np.asarray(translate_mass(M, C, "200m", "200c", Z, THETA)[2])
        c500c = np.asarray(translate_mass(M, C, "200m", "500c", Z, THETA)[2])
        assert np.all(c500c < c200c) and np.all(c200c < np.asarray(C))

    def test_scale_radius_is_invariant(self):
        """r_s = r_delta / c is a property of the profile, not of the definition."""
        _, r_in, _ = translate_mass(M, C, "200m", "200m", Z, THETA)
        rs_in = np.asarray(r_in) / np.asarray(C)
        for other in ("200c", "500c", "vir"):
            _, r_o, c_o = translate_mass(M, C, "200m", other, Z, THETA)
            rs_o = np.asarray(r_o) / np.asarray(c_o)
            assert np.allclose(rs_o, rs_in, rtol=1e-4), other

    def test_matches_the_legacy_gas_converter(self):
        """Cross-check against gas.conversions.m200_to_m500c, which predates this."""
        from hod_mod.gas.conversions import m200_to_m500c
        from hod_mod.core.mass_definitions import _RHO_CRIT0
        om = THETA["Omega_m"]
        ez2 = om * (1 + Z) ** 3 + (1 - om)
        rho_crit_comoving = _RHO_CRIT0 * ez2 / (1 + Z) ** 3
        r200c = MassDef(200, "critical").radius(M, Z, THETA)
        m500_legacy, r500_legacy = m200_to_m500c(M, C, r200c, rho_crit_comoving)
        m500_new, r500_new, _ = translate_mass(M, C, "200c", "500c", Z, THETA)
        assert np.allclose(np.asarray(m500_new), np.asarray(m500_legacy), rtol=1e-4)
        assert np.allclose(np.asarray(r500_new), np.asarray(r500_legacy), rtol=1e-4)


class TestJax:
    def test_jittable(self):
        f = jax.jit(lambda m, c: translate_mass(m, c, "200m", "500c", Z, THETA)[0])
        assert np.all(np.isfinite(np.asarray(f(M, C))))

    def test_differentiable_wrt_mass(self):
        g = jax.grad(lambda m: jnp.sum(translate_mass(m, C, "200m", "500c", Z, THETA)[0]))
        d = np.asarray(g(M))
        assert np.all(np.isfinite(d)) and np.all(d > 0), "dM500c/dM200m must be positive"

    def test_differentiable_wrt_omega_m(self):
        """Gradients must flow through the cosmology, not be silently zero."""
        def f(om):
            return jnp.sum(translate_mass(M, C, "200m", "500c", Z, {"Omega_m": om})[0])
        d = float(jax.grad(f)(0.31))
        fd = (float(f(0.311)) - float(f(0.309))) / 0.002
        assert np.isfinite(d) and d != 0.0
        assert d == pytest.approx(fd, rel=1e-3)
