"""Tests for hod_mod.core.numerics.safe_log (shared float32-safe log floor)."""

import numpy as np
import pytest
import jax.numpy as jnp


class TestSafeLog:
    def test_identity_above_floor(self):
        from hod_mod.core.numerics import safe_log

        p = jnp.asarray([1e-10, 1.0, 1e5])
        np.testing.assert_allclose(np.asarray(safe_log(p)),
                                   np.log(np.asarray(p)), rtol=1e-6)

    def test_floor_applied(self):
        from hod_mod.core.numerics import safe_log

        out = safe_log(jnp.asarray([0.0, 1e-45]), 1e-30)
        np.testing.assert_allclose(np.asarray(out), np.log(1e-30), rtol=1e-6)
        out20 = safe_log(jnp.asarray([0.0]), 1e-20)
        assert float(out20[0]) == pytest.approx(np.log(1e-20), rel=1e-6)

    def test_nonfinite_floored_not_propagated(self):
        from hod_mod.core.numerics import safe_log

        p = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf, 1.0])
        out = np.asarray(safe_log(p))
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out[:3], np.log(1e-30), rtol=1e-6)
        assert out[3] == pytest.approx(0.0, abs=1e-6)

    def test_floor_is_float32_representable(self):
        from hod_mod.core.numerics import safe_log

        # the whole point of the 1e-30 default: it must survive float32,
        # otherwise an all-zero field gives -inf and jnp.interp NaNs
        assert np.float32(1e-30) > 0.0
        out = safe_log(jnp.zeros(4, dtype=jnp.float32))
        assert np.all(np.isfinite(np.asarray(out)))

    def test_cross_spectra_alias(self):
        from hod_mod.core.numerics import safe_log
        from hod_mod.observables.cross_spectra import _safe_log

        assert _safe_log is safe_log
