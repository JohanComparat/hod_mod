"""``baryon_params`` must reach the power spectrum, and cannot be passed by position.

``NonLinearHaloModelPrediction.wp`` / ``.delta_sigma`` document a
``baryon_params`` argument and forwarded it *positionally* to
``FullHaloModelPrediction``, whose signatures had no such parameter.  The dict
therefore landed in ``n_pi`` (``wp``) and ``chi_max`` (``delta_sigma``) —
numerical tuning arguments.  Nothing raised on the ``wp`` path, so the
advertised baryon split silently did nothing while reporting a normal result.

Two guards, because they fail at different times:

* a **static** one on the signatures, which costs no runtime and catches the
  whole class — every optional argument of these methods is keyword-only, so a
  dict can never again slide into a float slot;
* a **numeric** one that the split actually moves ΔΣ and leaves w_p alone
  (galaxies are a discrete tracer; satellites trace the dark matter, so P_gg
  carries no baryons while P_gm does).
"""
import ast
import inspect
import pathlib

import numpy as np
import pytest

from hod_mod.observables.clustering import (
    FullHaloModelPrediction,
    NonLinearHaloModelPrediction,
)

ROOT = pathlib.Path(__file__).resolve().parents[1] / "hod_mod"


# ---------------------------------------------------------------- static ----
@pytest.mark.parametrize("method", ["wp", "delta_sigma", "xi_3d"])
def test_baryon_params_is_keyword_only(method):
    """A dict must not be assignable to a numerical slot by position."""
    params = inspect.signature(getattr(FullHaloModelPrediction, method)).parameters
    assert "baryon_params" in params, f"{method} does not accept baryon_params"
    assert params["baryon_params"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("method", ["wp", "delta_sigma"])
def test_every_optional_argument_is_keyword_only(method):
    """No optional argument may be positional — that is what let the bug in."""
    positional = [
        p for p in inspect.signature(getattr(FullHaloModelPrediction, method))
        .parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is not inspect.Parameter.empty
    ]
    assert not positional, (
        f"{method} has positional-with-default arguments "
        f"{[p.name for p in positional]}; a caller passing an extra positional "
        "argument would silently overwrite one of them"
    )


def test_delegation_passes_baryon_params_by_keyword():
    """``NonLinearHaloModelPrediction`` must not forward it positionally."""
    tree = ast.parse((ROOT / "observables" / "clustering.py").read_text())
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "NonLinearHaloModelPrediction"
    )
    checked = 0
    for call in (n for n in ast.walk(cls) if isinstance(n, ast.Call)):
        if not (isinstance(call.func, ast.Attribute)
                and call.func.attr in {"wp", "delta_sigma", "xi_3d"}):
            continue
        passes_positionally = any(
            isinstance(a, ast.Name) and a.id == "baryon_params" for a in call.args
        )
        assert not passes_positionally, (
            f"{call.func.attr} forwards baryon_params positionally at line "
            f"{call.lineno}; it lands in a numerical argument"
        )
        checked += 1
    assert checked >= 2, "the delegating calls were not found — test is vacuous"


# --------------------------------------------------------------- numeric ----
@pytest.mark.slow
class TestBaryonSplitReachesPgm:
    @staticmethod
    def _build(pk_lin, hmf):
        from hod_mod.connection.hod import MoreHODModel
        from hod_mod.core.halo_profiles import HaloProfile
        from hod_mod.observables import make_baryon_fraction

        theta = pk_lin.default_cosmology()
        hod = MoreHODModel(hmf, hmf.bias)
        hp = HaloProfile(
            {"flat": True, "H0": theta["h"] * 100.0, "Om0": theta["Omega_m"],
             "Ob0": theta["Omega_b"], "ns": theta["n_s"]},
            cm_relation="diemer19",
        )
        bf = make_baryon_fraction("sigmoid")
        pred = FullHaloModelPrediction(pk_lin, hod, hp, baryon_fraction=bf)
        return pred, theta, MoreHODModel.default_params(), bf.default_params()

    def test_delta_sigma_responds(self, pk_lin, hmf):
        pred, theta, hod_p, bar_p = self._build(pk_lin, hmf)
        R = np.logspace(-0.5, 1.2, 6)
        without = np.asarray(pred.delta_sigma(R, 0.2, theta, hod_p))
        with_b = np.asarray(pred.delta_sigma(R, 0.2, theta, hod_p,
                                             baryon_params=bar_p))
        assert np.all(np.isfinite(with_b))
        assert np.max(np.abs(with_b / without - 1.0)) > 1e-5, (
            "baryon_params did not reach P_gm — the split is inert again"
        )

    def test_wp_is_unchanged(self, pk_lin, hmf):
        """P_gg carries no baryons: satellites trace the dark matter."""
        pred, theta, hod_p, bar_p = self._build(pk_lin, hmf)
        rp = np.logspace(-1.0, 1.3, 6)
        without = np.asarray(pred.wp(rp, 60.0, 0.2, theta, hod_p))
        with_b = np.asarray(pred.wp(rp, 60.0, 0.2, theta, hod_p,
                                    baryon_params=bar_p))
        np.testing.assert_allclose(with_b, without, rtol=0, atol=0)

    def test_positional_is_refused(self, pk_lin, hmf):
        pred, theta, hod_p, bar_p = self._build(pk_lin, hmf)
        with pytest.raises(TypeError, match="positional"):
            pred.delta_sigma(np.logspace(-0.5, 1.2, 6), 0.2, theta, hod_p, bar_p)
