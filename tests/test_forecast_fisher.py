r"""Unit tests for the Fisher-matrix machinery (:mod:`hod_mod.forecast.fisher`).

These exercise the pure-numpy linear algebra (matrix assembly, pseudo-inverse
with flat-direction dropping, figure of merit, degeneracy ranking, principal
directions, masking and probe decomposition) on small synthetic Jacobians, plus
one trivial ``jax.jacfwd`` check of :func:`fisher.jacobian`.  No heavy forward
model is built, so the suite is fast and free of the JAX/coverage interaction of
the full forecast tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from hod_mod.forecast import fisher


# ---------------------------------------------------------------------------
# A tiny linear model  d = J @ theta  with known Jacobian J (constant).
# ---------------------------------------------------------------------------
@pytest.fixture
def linear_model():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(12, 3))
    theta0 = np.array([1.0, 2.0, 0.5])
    d0 = J @ theta0
    return d0, J, theta0


def test_fisher_matrix_diagonal_matches_manual(linear_model):
    d0, J, _ = linear_model
    rel = 0.05
    F = fisher.fisher_matrix(d0, J, rel_err=rel)
    inv_var = 1.0 / (rel * d0) ** 2
    F_manual = (J.T * inv_var) @ J
    assert F.shape == (3, 3)
    np.testing.assert_allclose(F, F_manual, rtol=1e-12)
    np.testing.assert_allclose(F, F.T, rtol=1e-12)          # symmetric


def test_fisher_matrix_zero_data_gives_infinite_variance():
    # a data point equal to 0 has infinite (rel-error) sigma → no information
    d0 = np.array([0.0, 1.0])
    J = np.array([[1.0, 0.0], [0.0, 1.0]])
    F = fisher.fisher_matrix(d0, J, rel_err=0.1)
    assert F[0, 0] == 0.0                                    # first param unconstrained
    assert F[1, 1] > 0.0


def test_fisher_matrix_prior_adds_diagonal(linear_model):
    d0, J, _ = linear_model
    F0 = fisher.fisher_matrix(d0, J, rel_err=0.1)
    ps = np.array([0.5, np.inf, 2.0])                        # prior on params 0 and 2
    F1 = fisher.fisher_matrix(d0, J, rel_err=0.1, prior_sigma=ps)
    add = np.diag([1 / 0.5 ** 2, 0.0, 1 / 2.0 ** 2])
    np.testing.assert_allclose(F1 - F0, add, atol=1e-12)


def test_fisher_matrix_covariance_mode_matches_diagonal(linear_model):
    d0, J, _ = linear_model
    sig = np.abs(0.1 * d0)
    cov = np.diag(sig ** 2)
    F_cov = fisher.fisher_matrix(d0, J, cov=cov)
    F_diag = fisher.fisher_matrix(d0, J, rel_err=0.1)
    np.testing.assert_allclose(F_cov, F_diag, rtol=1e-8)


def test_fisher_matrix_covariance_mode_offdiagonal():
    # a correlated 2x2 covariance inverts correctly through the scaled path
    J = np.eye(2)
    d0 = np.array([1.0, 1.0])
    cov = np.array([[1.0, 0.4], [0.4, 2.0]])
    F = fisher.fisher_matrix(d0, J, cov=cov)
    np.testing.assert_allclose(F, np.linalg.inv(cov), rtol=1e-8)


def test_constraints_recovers_sigma_and_identity_corr():
    s = np.array([0.1, 0.3, 1.0])
    F = np.diag(1.0 / s ** 2)
    cov, sigma, corr = fisher.constraints(F)
    np.testing.assert_allclose(sigma, s, rtol=1e-8)
    np.testing.assert_allclose(cov, np.diag(s ** 2), rtol=1e-8)
    np.testing.assert_allclose(corr, np.eye(3), atol=1e-8)


def test_constraints_flat_direction_is_unconstrained():
    # param 2 has zero derivative → a flat direction → sigma = inf, dropped
    J = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    d0 = np.ones(3)
    F = fisher.fisher_matrix(d0, J, rel_err=0.1)
    cov, sigma, corr = fisher.constraints(F)
    assert np.isinf(sigma[2])
    assert np.all(np.isfinite(sigma[:2]))
    # the flat direction contributes nothing to the correlation block
    assert corr[2, 0] == 0.0 and corr[2, 1] == 0.0


def test_constraints_correlation_bounded():
    F = np.array([[4.0, 1.5, 0.0], [1.5, 3.0, 0.5], [0.0, 0.5, 2.0]])
    _, _, corr = fisher.constraints(F)
    assert np.all(np.abs(corr) <= 1.0 + 1e-9)
    np.testing.assert_allclose(np.diag(corr), np.ones(3), atol=1e-8)


def test_figure_of_merit():
    cov = np.diag([0.04, 0.09, 1.0])
    fom = fisher.figure_of_merit(cov, 0, 1)
    assert fom == pytest.approx(1.0 / np.sqrt(0.04 * 0.09))
    # correlated block: FoM uses the full 2x2 determinant
    cov2 = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert fisher.figure_of_merit(cov2, 0, 1) == pytest.approx(1.0 / np.sqrt(0.75))


def test_figure_of_merit_singular_returns_inf():
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])                 # det = 0
    assert np.isinf(fisher.figure_of_merit(cov, 0, 1))


def test_top_degeneracies_orders_by_abs_correlation():
    names = ["a", "b", "c"]
    corr = np.array([[1.0, 0.9, -0.2], [0.9, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    pairs = fisher.top_degeneracies(corr, names, k=2)
    assert len(pairs) == 2
    assert pairs[0][1:3] == ("a", "b")                       # strongest pair first
    assert pairs[0][0] == pytest.approx(0.9)
    # sorted descending by |corr|
    assert pairs[0][0] >= pairs[1][0]


def test_principal_directions_identifies_worst_constrained():
    cov = np.diag([1.0, 0.01, 0.04])                         # param 0 worst
    dirs = fisher.principal_directions(cov, ["a", "b", "c"], k=1)
    assert len(dirs) == 1
    assert dirs[0]["variance"] == pytest.approx(1.0)
    top_component = dirs[0]["components"][0][0]
    assert top_component == "a"


def test_principal_directions_normalised_basis():
    cov = np.diag([1.0, 1.0])
    dirs = fisher.principal_directions(cov, ["a", "b"], sigma_fid=np.array([2.0, 1.0]), k=2)
    # scaling by sigma_fid changes the relative variances (a becomes 1/4)
    variances = sorted(d["variance"] for d in dirs)
    assert variances[0] == pytest.approx(0.25)
    assert variances[1] == pytest.approx(1.0)


def test_masked_constraints_matches_direct(linear_model):
    d0, J, _ = linear_model
    mask = np.array([True] * 6 + [False] * 6)
    _, sig_masked, _ = fisher.masked_constraints(d0, J, mask, rel_err=0.1)
    F = fisher.fisher_matrix(d0[mask], J[mask], rel_err=0.1)
    _, sig_direct, _ = fisher.constraints(F, ridge=1e-9)
    np.testing.assert_allclose(sig_masked, sig_direct, rtol=1e-8)


def test_probe_decomposition_structure():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(8, 2))
    d0 = np.abs(rng.normal(size=8)) + 1.0
    row_obs = np.array(["wp"] * 4 + ["ds"] * 4)
    scale_mask = np.ones(8, dtype=bool)
    out = fisher.probe_decomposition(
        d0, J, row_obs, scale_mask, ["wp", "ds"], rel_err=0.1,
        prior_sigma=np.array([1.0, 1.0]))
    assert set(out["single"]) == {"wp", "ds"}
    assert "wp" in out["cumulative"]
    assert "wp+ds" in out["cumulative"]
    # adding a second probe cannot loosen the constraint (with fixed priors)
    s_wp = out["cumulative"]["wp"]
    s_all = out["cumulative"]["wp+ds"]
    assert np.all(s_all <= s_wp + 1e-9)


def test_probe_decomposition_skips_empty_probe():
    J = np.ones((4, 2))
    d0 = np.ones(4)
    row_obs = np.array(["wp", "wp", "wp", "wp"])
    out = fisher.probe_decomposition(
        d0, J, row_obs, np.ones(4, bool), ["wp", "ds"], 0.1, np.array([1.0, 1.0]))
    assert "ds" not in out["single"]                         # no ds rows → skipped
    assert "wp" in out["single"]


def test_jacobian_of_linear_function():
    import jax.numpy as jnp
    A = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])

    def f(theta):
        return A @ theta

    theta0 = np.array([0.3, -0.7])
    d0, Jac = fisher.jacobian(f, theta0)
    np.testing.assert_allclose(d0, np.asarray(A) @ theta0, rtol=1e-6)
    np.testing.assert_allclose(Jac, np.asarray(A), rtol=1e-6)
