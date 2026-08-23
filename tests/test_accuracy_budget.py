"""The accuracy budget must exist, be current, and be within its own tolerances.

``benchmarks/accuracy.json`` is the machine-readable statement of how accurate
each predicted quantity is, established against an independent implementation.
Tables in the documentation and the paper are generated from it. These tests
guard the file itself; regenerating it is
``python -m hod_mod.scripts.make_accuracy_budget``.
"""
import json
import os

import pytest

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_BUDGET = os.path.join(_ROOT, "benchmarks", "accuracy.json")

_REQUIRED_KEYS = {"quantity", "reference", "deviation", "tolerance",
                  "metric", "note", "status"}

# Quantities that must be present. A silent disappearance -- a check that stops
# running because its reference package vanished from the environment -- is the
# failure mode this guards against.
_MUST_COVER = [
    "E(z)", "chi(z)", "c(M,z) Duffy08", "b(M,z) Tinker10",
    "P_lin(k) shape, EH98", "mass translation round trip",
]


@pytest.fixture(scope="module")
def budget():
    if not os.path.exists(_BUDGET):
        pytest.fail(
            f"{_BUDGET} is missing; regenerate with "
            "`python -m hod_mod.scripts.make_accuracy_budget`"
        )
    with open(_BUDGET) as fh:
        return json.load(fh)


class TestStructure:
    def test_has_provenance(self, budget):
        for key in ("generated", "git_sha", "python", "jax", "platform", "cosmology"):
            assert key in budget, f"budget is missing provenance field '{key}'"

    def test_entries_are_well_formed(self, budget):
        assert budget["entries"], "budget contains no entries"
        for e in budget["entries"]:
            missing = _REQUIRED_KEYS - set(e)
            assert not missing, f"entry {e.get('quantity')} is missing {missing}"

    def test_count_matches(self, budget):
        assert budget["n_entries"] == len(budget["entries"])

    @pytest.mark.parametrize("quantity", _MUST_COVER)
    def test_required_quantity_is_covered(self, budget, quantity):
        names = [e["quantity"] for e in budget["entries"]]
        assert quantity in names, (
            f"'{quantity}' is absent from the accuracy budget; either the check "
            "was removed or its reference package is not installed"
        )


class TestTolerances:
    def test_nothing_is_out_of_tolerance(self, budget):
        bad = [e for e in budget["entries"]
               if e["status"] == "ok" and e["tolerance"] is not None
               and e["deviation"] is not None
               and e["deviation"] > e["tolerance"]]
        assert not bad, "out of tolerance: " + "; ".join(
            f"{e['quantity']} vs {e['reference']}: "
            f"{e['deviation']:.3e} > {e['tolerance']:.1e}" for e in bad
        )

    def test_recorded_failure_count_is_consistent(self, budget):
        bad = [e for e in budget["entries"]
               if e["status"] == "ok" and e["tolerance"] is not None
               and e["deviation"] is not None
               and e["deviation"] > e["tolerance"]]
        assert budget["n_out_of_tolerance"] == len(bad)

    def test_skipped_entries_are_visible_not_dropped(self, budget):
        """A skipped check keeps its row, with a reason. Gaps must be visible."""
        for e in budget["entries"]:
            if e["status"] == "skipped":
                assert e["deviation"] is None
                assert e["note"], f"{e['quantity']} skipped without a reason"

    def test_no_check_errored(self, budget):
        errored = [e for e in budget["entries"] if e["status"] == "error"]
        assert not errored, "checks raised: " + "; ".join(
            f"{e['quantity']}: {e['note']}" for e in errored
        )
