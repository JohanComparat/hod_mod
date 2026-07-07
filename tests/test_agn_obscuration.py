"""Tests for agn.ham.obscured_fraction (Comparat+2019 eq. 11).

Kept out of test_agn_ham.py because that module is marked slow (HamAGNModel
table builds); obscured_fraction is a cheap standalone function.
"""

import numpy as np
import pytest
import jax.numpy as jnp


class TestObscuredFraction:
    def test_bounded_and_finite(self):
        from hod_mod.agn import obscured_fraction

        lx = jnp.linspace(40.0, 47.0, 200)
        for z in (0.0, 0.5, 1.0, 2.0, 4.0):
            f = obscured_fraction(lx, z)
            assert f.shape == lx.shape
            assert jnp.all(jnp.isfinite(f))
            assert jnp.all((f >= 0.0) & (f <= 1.0))

    def test_faint_end_is_highly_obscured(self):
        from hod_mod.agn import obscured_fraction

        # faint limit: f -> f_2 = 0.9 sqrt(41/log10Lx)
        f = obscured_fraction(jnp.array([41.0]), 0.0)
        assert float(f[0]) == pytest.approx(0.9, abs=0.02)

    def test_bright_agn_less_obscured_than_faint(self):
        from hod_mod.agn import obscured_fraction

        for z in (0.0, 1.0, 2.0):
            f_faint = float(obscured_fraction(jnp.array([42.0]), z)[0])
            f_bright = float(obscured_fraction(jnp.array([46.0]), z)[0])
            assert f_bright < f_faint

    def test_decreasing_through_transition(self):
        from hod_mod.agn import obscured_fraction

        # the erf blend makes f monotonically decreasing across the
        # luminosity transition ll(z) ~ 43-44
        lx = jnp.linspace(42.0, 46.0, 100)
        f = np.asarray(obscured_fraction(lx, 0.5))
        assert np.all(np.diff(f) <= 1e-6)

    def test_obscuration_rises_with_redshift_at_bright_end(self):
        from hod_mod.agn import obscured_fraction

        # bright-end f_1 carries the erf(z/4)*0.3 evolution term
        lx = jnp.array([45.5])
        f0 = float(obscured_fraction(lx, 0.0)[0])
        f2 = float(obscured_fraction(lx, 2.0)[0])
        assert f2 > f0

    def test_matches_hand_computed_blend(self):
        from hod_mod.agn.ham import (
            _f_obsc_bright, _f_obsc_faint, _ll_transition, obscured_fraction)

        lx = jnp.array([42.0, 43.5, 45.0])
        z = 0.8
        f1 = _f_obsc_bright(lx, z)
        f2 = _f_obsc_faint(lx, z)
        blend = 0.5 + 0.5 * jax_erf((_ll_transition(z) - lx) / 0.6)
        expected = jnp.clip(f1 + (f2 - f1) * blend, 0.0, 1.0)
        np.testing.assert_allclose(
            np.asarray(obscured_fraction(lx, z)), np.asarray(expected), rtol=1e-6)


def jax_erf(x):
    from jax.scipy.special import erf
    return erf(x)
