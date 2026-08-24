"""Every ``revive_ensemble`` call site must pass attributes that exist.

``revive_ensemble`` runs only on the RESUME branch of a sampler — the branch
taken when a chain already has steps on disk.  No test starts a real MCMC, kills
it and restarts it, so that branch is executed for the first time on the cluster,
hours into a job, where an ``AttributeError`` costs a walltime slot and looks
like a scheduler problem.

That is exactly what happened: ``JointZM15.sample`` passed ``self.lo``/``self.hi``
to a class that only has ``self.bounds``, so *every* resume of
``bgs_zm15_joint_wp_ngal`` and ``bgs_zm15_thresh_joint`` died on its first
statement.  Both fits therefore only ever advanced during a fresh run and lost
all progress at the first walltime kill — across three campaigns.

This checks the wiring statically, which is enough to catch that whole class of
bug and costs no runtime.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "hod_mod"


def _self_attrs_assigned(class_node: ast.ClassDef) -> set[str]:
    """Names assigned as ``self.<name> = ...`` anywhere in the class body."""
    found: set[str] = set()
    for node in ast.walk(class_node):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                found.add(t.attr)
    return found


def _call_sites():
    """(file, lineno, attr, class_node) for each ``self.X`` passed to revive_ensemble."""
    for path in sorted(ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                      # pragma: no cover
            continue
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "revive_ensemble"):
                continue
            # walk up to the enclosing class, if any
            cls, cur = None, node
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, ast.ClassDef):
                    cls = cur
                    break
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if (isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                        and arg.value.id == "self"):
                    yield path, node.lineno, arg.attr, cls


def test_revive_ensemble_call_sites_exist():
    """A ``self.<attr>`` handed to revive_ensemble must be assigned by its class."""
    sites = list(_call_sites())
    assert sites, "no revive_ensemble(self.…) call sites found — did the API move?"
    broken = []
    for path, lineno, attr, cls in sites:
        if cls is None:
            broken.append(f"{path.name}:{lineno} self.{attr} outside any class")
            continue
        if attr not in _self_attrs_assigned(cls):
            broken.append(f"{path.name}:{lineno} {cls.name} has no self.{attr}")
    assert not broken, (
        "revive_ensemble would raise AttributeError on the resume branch:\n  "
        + "\n  ".join(broken)
    )


def test_jointzm15_exposes_bounds_not_lo_hi():
    """Pin the specific contract the regression depended on."""
    src = (ROOT / "scripts/fitting/bgs_ls10/fit_bgs_zm15_joint.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "JointZM15")
    attrs = _self_attrs_assigned(cls)
    assert "bounds" in attrs, "JointZM15 lost self.bounds"
    assert "lo" not in attrs and "hi" not in attrs, (
        "JointZM15 now defines lo/hi — simplify the resume call site to use them"
    )
