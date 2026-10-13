"""Strict bounds on every forecast parameter.

A Gaussian prior width cannot say "never negative". With ``sigma = 0.25`` eV
about a ``0.06`` eV fiducial, roughly 40% of the prior mass on ``sum_mnu``
sits below zero, and the forward model returns a number there rather than
complaining -- which is how a marginalised neutrino constraint ends up
including unphysical territory. ``PARAM_BOUNDS`` is the piece the precision
vector cannot express.

Two failure modes are guarded here, and they pull in opposite directions:

*A missing or wrong bound.* Every parameter must have one, every fiducial must
sit strictly inside it, and the numerical hazards the bounds exist to prevent
(a width of zero that is divided by, a fraction above one) must actually be
excluded.

*A bound that is too tight.* A bound that excludes a value the literature
reports, or that this package itself fits to, is worse than no bound: it
silently truncates a posterior. The shipped benchmark best-fits are checked
against the table for exactly that reason.
"""
import json
import os

import numpy as np
import pytest

from hod_mod.forecast.forward_jax import PARAM_NAMES, N_PARAM, _IDX
from hod_mod.forecast import params as P

_KINDS = {"physical", "definitional", "validity", "prior"}


# ------------------------------------------------------------- coverage --

class TestCoverage:
    def test_every_parameter_has_a_bound(self):
        missing = [n for n in PARAM_NAMES if n not in P.PARAM_BOUNDS]
        assert not missing, (
            f"{len(missing)} parameters have no bound: {missing}. A new "
            "parameter must declare one; that is the point of this test.")

    def test_no_bounds_for_parameters_that_do_not_exist(self):
        extra = sorted(set(P.PARAM_BOUNDS) - set(PARAM_NAMES))
        assert not extra, f"bounds for unknown parameters: {extra}"

    def test_bounds_are_ordered_and_finite_or_sentinel(self):
        for n, (lo, hi) in P.PARAM_BOUNDS.items():
            assert lo < hi, f"{n}: lo={lo} is not below hi={hi}"
            assert np.isfinite(lo) or lo <= -P._INF, f"{n}: lo is a bare inf"
            assert np.isfinite(hi) or hi >= P._INF, f"{n}: hi is a bare inf"

    def test_vector_matches_registry_order(self):
        b = P.bounds_vector()
        assert b.shape == (N_PARAM, 2)
        for n in PARAM_NAMES:
            assert tuple(b[_IDX[n]]) == pytest.approx(P.PARAM_BOUNDS[n])


# ------------------------------------------------------------- fiducial --

def _margin_fraction(lo, hi, v):
    """How far the fiducial sits from the nearer bound, as a fraction of width.

    Measured in log10 for a strictly positive parameter spanning decades:
    ``p03_pressure`` runs from 1e-8 to 3e-3, where a linear margin makes a
    perfectly central fiducial look pinned against the lower bound.
    """
    lo_f = max(lo, -1e6)
    hi_f = min(hi, 1e6)
    if lo_f > 0 and hi_f / lo_f >= 100.0 and v > 0:
        lo_f, hi_f, v = np.log10(lo_f), np.log10(hi_f), np.log10(v)
    width = hi_f - lo_f
    if not np.isfinite(width) or width <= 0:
        return None
    return min(v - lo_f, hi_f - v) / width


class TestFiducial:
    def test_fiducial_is_inside_every_bound(self):
        assert P.bounds_violations(P.fiducial_vector()) == []

    def test_fiducial_has_room_to_move(self):
        """A fiducial on a bound gives a one-sided derivative and a degenerate
        walker start. Known cases are declared in BOUNDARY_FIDUCIALS."""
        fid = P.load_fiducial()
        pinned = []
        for n in PARAM_NAMES:
            m = _margin_fraction(*P.PARAM_BOUNDS[n], float(fid[n]))
            if m is not None and m < 1e-3:
                pinned.append(n)
        unexpected = sorted(set(pinned) - set(P.BOUNDARY_FIDUCIALS))
        assert not unexpected, (
            f"fiducials sitting on a bound without being declared in "
            f"BOUNDARY_FIDUCIALS: {unexpected}")

    def test_declared_boundary_fiducials_really_are_on_a_bound(self):
        """The exemption list must not outlive the thing it exempts."""
        fid = P.load_fiducial()
        for n in P.BOUNDARY_FIDUCIALS:
            assert n in P.PARAM_BOUNDS, n
            m = _margin_fraction(*P.PARAM_BOUNDS[n], float(fid[n]))
            assert m is not None and m < 1e-3, (
                f"{n} is declared a boundary fiducial but sits well inside "
                "its bounds; drop it from BOUNDARY_FIDUCIALS")


# ------------------------------------------------------------ semantics --

class TestPhysicalContent:
    def test_neutrino_mass_cannot_be_negative(self):
        lo, hi = P.PARAM_BOUNDS["sum_mnu"]
        assert lo == 0.0
        assert 0.06 < hi <= 0.5, "upper bound should stay on the distilled grid"

    def test_neutrino_bound_stays_within_the_distilled_table(self):
        """Above the table's f_nu grid the ratios extrapolate flat, which would
        silently understate the suppression."""
        from hod_mod.core import nu_ratio
        tab = nu_ratio.load()
        f_max = float(np.asarray(tab["f_nu"]).max())
        fid = P.load_fiducial()
        h, om = float(fid["h"]), float(fid["Omega_m"])
        f_at_bound = P.PARAM_BOUNDS["sum_mnu"][1] / (nu_ratio.NU_DENOM_EV
                                                     if hasattr(nu_ratio, "NU_DENOM_EV")
                                                     else 93.14) / h ** 2 / om
        assert f_at_bound <= f_max, (
            f"sum_mnu upper bound implies f_nu={f_at_bound:.4f}, past the "
            f"table's {f_max:.4f}")

    @pytest.mark.parametrize("name", [
        "sigma_ms", "agn_sig_r",
    ])
    def test_widths_are_strictly_positive(self, name):
        """These are divided by; a closed bound at zero admits the pole."""
        lo, _ = P.PARAM_BOUNDS[name]
        assert lo > 0.0, f"{name} is a width that the model divides by"

    def test_fractions_do_not_exceed_one(self):
        for n in ("f_loud0", "agn_fabs", "f_morph_sat"):
            if n not in P.PARAM_BOUNDS:
                continue
            lo, hi = P.PARAM_BOUNDS[n]
            assert lo >= 0.0 and hi <= 1.0, f"{n} is a fraction: {(lo, hi)}"


# ------------------------------------------------------ not-too-tight ----

#: Result files whose keys collide with a forecast parameter name while meaning
#: something else. Each entry is (path fragment, parameter, why). These are not
#: bounds failures; they are the reason a name match is not a parameter match.
_LAYOUT_COLLISIONS = [
    ("xray_joint_bands", "kt_norm",
     "fit_xray_joint_bands pivots kt_norm at M500c = 1e14 (module docstring); "
     "the _v2_pivotM1 results predate that and are log10(kT) at M500c = 1, "
     "which is why they read about -8"),
    ("agn_duty_cycle", "beta_pressure",
     "the forecast parameter is a TILT added to the DPM Model-2 slope "
     "beta_P = 0.85 (forward_jax.py:1070); the duty-cycle script fits the "
     "absolute beta_P, and 2.0 is that script's own ceiling, i.e. railed"),
    ("agn_duty_cycle", "beta_gas",
     "same tilt-versus-absolute collision as beta_pressure"),
]


def _is_known_collision(relpath, name):
    return any(frag in relpath and name == param
               for frag, param, _ in _LAYOUT_COLLISIONS)


class TestNotTooTight:
    """A bound that excludes something the package fits to is a bug."""

    def test_bounds_admit_the_shipped_benchmark_best_fits(self):
        from hod_mod.paths import results_root
        roots = [os.path.normpath(os.path.join(os.path.dirname(__file__),
                                               os.pardir, "results")),
                 os.fspath(results_root())]
        roots = [r for r in dict.fromkeys(roots) if os.path.isdir(r)]
        if not roots:
            pytest.skip("no results tree available in this environment")
        offenders = []
        walked = 0
        for root in roots:
          for dirpath, _, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(dirpath, fn)) as fh:
                        doc = json.load(fh)
                except Exception:
                    continue
                if not isinstance(doc, dict):
                    continue
                walked += 1
                for block in ("best_fit", "map_params", "params", "median",
                              "map", "best_fit_params"):
                    vals = doc.get(block)
                    if not isinstance(vals, dict):
                        continue
                    for n, v in vals.items():
                        if n not in P.PARAM_BOUNDS or not isinstance(v, (int, float)):
                            continue
                        lo, hi = P.PARAM_BOUNDS[n]
                        rel = os.path.relpath(os.path.join(dirpath, fn), root)
                        if not (lo <= float(v) <= hi) and not _is_known_collision(rel, n):
                            offenders.append((rel, n, float(v), lo, hi))
        if walked == 0:
            pytest.skip("no result JSONs found to check against")
        assert not offenders, (
            "bounds exclude values this package itself fitted:\n"
            + "\n".join(f"  {f}: {n}={v} outside [{lo}, {hi}]"
                        for f, n, v, lo, hi in offenders))

    def test_planck_prior_ranges_fit_inside_their_bounds(self):
        """Every parameter carrying a tight external prior must have room for
        it: a bound inside 5 sigma of Planck would truncate the prior."""
        fid = P.load_fiducial()
        for n, sig in P.PLANCK_PRIOR_SIGMA.items():
            lo, hi = P.PARAM_BOUNDS[n]
            v = float(fid[n])
            assert v - 5 * sig > lo or lo <= -P._INF, f"{n}: bound cuts Planck 5σ low"
            assert v + 5 * sig < hi or hi >= P._INF, f"{n}: bound cuts Planck 5σ high"


# ------------------------------------------------------- enforcement -----

class TestEnforcement:
    def test_log_prior_bounds_rejects_outside(self):
        v = P.fiducial_vector()
        assert P.log_prior_bounds(v) == 0.0
        bad = v.copy()
        bad[_IDX["sum_mnu"]] = -1e-6
        assert P.log_prior_bounds(bad) == -np.inf

    def test_clip_lands_inside(self):
        v = P.fiducial_vector().copy()
        v[_IDX["sum_mnu"]] = -5.0
        v[_IDX["sigma8"]] = 99.0
        assert P.bounds_violations(P.clip_to_bounds(v)) == []

    def test_violations_reports_name_and_value(self):
        v = P.fiducial_vector().copy()
        v[_IDX["sum_mnu"]] = -0.2
        (name, val, lo, hi), = P.bounds_violations(v)
        assert name == "sum_mnu" and val == pytest.approx(-0.2) and lo == 0.0

    def test_nonfinite_is_a_violation(self):
        v = P.fiducial_vector().copy()
        v[_IDX["sigma8"]] = np.nan
        assert [n for n, *_ in P.bounds_violations(v)] == ["sigma8"]

    def test_perturbation_stays_inside(self):
        rng = np.random.default_rng(7)
        v = P.fiducial_vector()
        for _ in range(20):
            assert P.bounds_violations(P.perturb_inside_bounds(v, 0.05, rng)) == []

    def test_scipy_pairs_use_none_for_open_sides(self):
        pairs = P.bounds_pairs()
        assert len(pairs) == N_PARAM
        for (lo, hi), n in zip(pairs, PARAM_NAMES):
            blo, bhi = P.PARAM_BOUNDS[n]
            assert (lo is None) == (blo <= -P._INF)
            assert (hi is None) == (bhi >= P._INF)


class TestLikelihoodEnforcement:
    @pytest.fixture(scope="class")
    def like(self):
        jnp = pytest.importorskip("jax.numpy")
        from hod_mod.fitting.jax_inference import MultiProbeGaussianLikelihood
        from hod_mod.forecast.forward_jax import ForwardModel
        fm = ForwardModel()
        f, _, _ = fm.full_data_vector_fn(["wp"])
        x = P.fiducial_vector()
        d0 = np.asarray(f(x))
        free = [_IDX["sum_mnu"], _IDX["sigma8"]]
        return MultiProbeGaussianLikelihood(
            f, d0, np.eye(len(d0)), free, x,
            prior_mean=x[free], prior_sigma=[0.25, 0.3])

    def test_bounds_reach_the_sampler_target(self, like):
        import jax.numpy as jnp
        x = P.fiducial_vector()[[_IDX["sum_mnu"], _IDX["sigma8"]]]
        assert np.isfinite(float(like.log_prior(jnp.asarray(x))))
        bad = x.copy(); bad[0] = -1e-3
        assert float(like.log_prior(jnp.asarray(bad))) == -np.inf

    def test_gradient_inside_is_unaffected(self, like):
        """The bound must not perturb the gradient anywhere inside the box."""
        import jax, jax.numpy as jnp
        x = jnp.asarray(P.fiducial_vector()[[_IDX["sum_mnu"], _IDX["sigma8"]]])
        g = np.asarray(jax.grad(like.log_prior)(x))
        assert np.all(np.isfinite(g))

    def test_pairs_are_exact_not_float32(self, like):
        assert like.bounds_pairs()[0] == (0.0, P.PARAM_BOUNDS["sum_mnu"][1])

    def test_opt_out_leaves_it_unbounded(self):
        from hod_mod.fitting.jax_inference import MultiProbeGaussianLikelihood
        from hod_mod.forecast.forward_jax import ForwardModel
        fm = ForwardModel()
        f, _, _ = fm.full_data_vector_fn(["wp"])
        x = P.fiducial_vector()
        d0 = np.asarray(f(x))
        like = MultiProbeGaussianLikelihood(
            f, d0, np.eye(len(d0)), [_IDX["sum_mnu"]], x, enforce_bounds=False)
        assert like.bounds_pairs() == [(None, None)]
