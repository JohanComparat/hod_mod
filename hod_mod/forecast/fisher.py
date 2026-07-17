r"""Fisher-matrix machinery for the sensitivity forecast.

With a constant *relative* error ``f`` on every data point, the data covariance
is ``C = diag((f · d_i)²)`` and the Fisher matrix is

.. math::

    F_{ab} = \sum_i \frac{1}{(f\,d_i)^2}
             \frac{\partial d_i}{\partial\theta_a}
             \frac{\partial d_i}{\partial\theta_b}
           = \frac{1}{f^2}\sum_i
             \frac{\partial\ln d_i}{\partial\theta_a}
             \frac{\partial\ln d_i}{\partial\theta_b},

so the **degeneracy directions are independent of ``f``** and only the overall
constraint size scales as ``f``.  The Jacobian ``∂d/∂θ`` is obtained by a single
``jax.jacfwd`` — exact autodiff through the whole (EH98-based) forward model.

Gaussian parameter priors (e.g. Planck on Ω_m, σ8) add ``diag(1/σ_prior²)`` to
``F`` before inversion.
"""

from __future__ import annotations

import numpy as np
import jax


def jacobian(data_vector_fn, theta0):
    """Return (d0, J) with d0 = f(θ0) and J = ∂f/∂θ at θ0 via jax.jacfwd."""
    import jax.numpy as jnp
    theta0 = jnp.asarray(theta0, dtype=float)
    d0 = np.asarray(data_vector_fn(theta0))
    J = np.asarray(jax.jacfwd(data_vector_fn)(theta0))     # (Ndata, Nparam)
    return d0, J


def fisher_matrix(d0, J, rel_err=None, prior_sigma=None, cov=None):
    r"""Fisher matrix (+ Gaussian priors).

    Diagonal mode (``rel_err`` given, ``cov`` None): ``F = Jᵀ diag(1/(rel_err·d)²) J``.
    Full-covariance mode (``cov`` given): ``F = Jᵀ C⁻¹ J`` — used for the analytic
    Gaussian covariance with cross-observable/cross-bin correlations.  The
    covariance is held fixed at the fiducial (only the mean is differentiated).
    """
    d0 = np.asarray(d0, dtype=float)
    J = np.asarray(J, dtype=float)
    if cov is not None:
        # invert in correlation-scaled space (the data vector spans many orders of
        # magnitude across observables, so a direct inverse is ill-conditioned)
        cov = np.asarray(cov, dtype=float)
        dsc = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        R = cov / np.outer(dsc, dsc)
        Cinv = np.linalg.inv(R) / np.outer(dsc, dsc)
        F = J.T @ Cinv @ J
    else:
        sigma = np.abs(rel_err * d0)
        inv_var = 1.0 / np.where(sigma > 0, sigma ** 2, np.inf)
        F = (J.T * inv_var) @ J                             # (Np, Np)
    if prior_sigma is not None:
        ps = np.asarray(prior_sigma, dtype=float)
        F = F + np.diag(np.where(np.isfinite(ps), 1.0 / ps ** 2, 0.0))
    return F


def constraints(F, ridge=0.0, rcond=1e-10, scale=None):
    """Invert F → (cov, sigma, corr) via eigenvalue-floored pseudo-inversion.

    Symmetrises F and drops eigen-directions with eigenvalue below
    ``rcond × max_eigenvalue`` (unconstrained/flat directions, e.g. the
    analytically degenerate ``fc``), which would otherwise make the plain
    inverse blow up.  Those directions get an effectively infinite variance;
    the well-constrained sub-space is unaffected.

    ``scale`` (optional, e.g. the broad-prior widths): eigendecompose the
    dimensionless ``S F S`` (S = diag(scale)) instead of ``F`` — parameters
    spanning many decades of natural units (b_sat ~ 100 vs Ω_b ~ 5e-4) then
    contribute comparably to the eigenvalue floor.  Recommended for the
    full tier-2 parameter vector (``PARAM_NAMES``).
    """
    F = 0.5 * (np.asarray(F, dtype=float) + np.asarray(F, dtype=float).T)
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
        cov_s, sigma_s, corr = constraints(F * np.outer(s, s), ridge=ridge,
                                           rcond=rcond)
        return cov_s * np.outer(s, s), sigma_s * s, corr
    if ridge > 0:
        F = F + ridge * np.diag(np.abs(np.diag(F)) + 1e-300)
    w, V = np.linalg.eigh(F)
    wmax = np.max(w) if w.size else 1.0
    floor = rcond * wmax
    inv_w = np.where(w > floor, 1.0 / w, 0.0)     # drop flat directions
    cov = (V * inv_w) @ V.T
    var = np.clip(np.diag(cov), 0.0, np.inf)
    # parameters living entirely in a dropped direction → unconstrained (inf)
    unconstrained = np.isclose(var, 0.0) & (np.abs(np.diag(F)) < floor)
    sigma = np.where(unconstrained, np.inf, np.sqrt(var))
    denom = np.outer(sigma, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(np.isfinite(denom) & (denom > 0), cov / denom, 0.0)
    return cov, sigma, corr


def figure_of_merit(cov, i, j):
    """FoM = 1/sqrt(det(cov[[i,j],[i,j]])) for a 2-parameter sub-block."""
    sub = cov[np.ix_([i, j], [i, j])]
    det = np.linalg.det(sub)
    return float(1.0 / np.sqrt(det)) if det > 0 else np.inf


def top_degeneracies(corr, names, k=6):
    """Return the k parameter pairs with the largest ``|correlation|``."""
    n = corr.shape[0]
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            pairs.append((abs(corr[a, b]), names[a], names[b], corr[a, b]))
    pairs.sort(reverse=True)
    return pairs[:k]


def principal_directions(cov, names, sigma_fid=None, k=3):
    """Eigen-decompose the covariance → worst/best-constrained combinations.

    Returns the ``k`` largest-variance eigenvectors (worst-constrained
    directions).  If ``sigma_fid`` (fiducial values, for normalisation) is
    given, eigenvectors are reported in the dimensionless θ/θ_fid basis.
    """
    C = np.asarray(cov, dtype=float)
    if sigma_fid is not None:
        s = np.asarray(sigma_fid, dtype=float)
        s = np.where(np.abs(s) > 0, np.abs(s), 1.0)
        C = C / np.outer(s, s)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    out = []
    for idx in order[:k]:
        vec = V[:, idx]
        contrib = sorted(zip(np.abs(vec), names, vec), reverse=True)[:4]
        out.append({"variance": float(w[idx]),
                    "components": [(nm, float(v)) for _, nm, v in contrib]})
    return out


def masked_constraints(d0, J, row_mask, rel_err, prior_sigma=None, ridge=1e-9):
    """Fisher constraints from a row subset of a precomputed full (d0, J)."""
    m = np.asarray(row_mask, dtype=bool)
    F = fisher_matrix(np.asarray(d0)[m], np.asarray(J)[m], rel_err, prior_sigma)
    return constraints(F, ridge=ridge)


def probe_decomposition(d0, J, row_obs, scale_mask, which_all, rel_err, prior_sigma):
    """σ(θ) per probe and cumulatively, from a single precomputed full (d0, J).

    ``row_obs`` labels each row's observable; ``scale_mask`` is the scale-cut row
    mask.  Returns {"single": {probe: sigma}, "cumulative": {label: sigma}}.
    """
    row_obs = np.asarray(row_obs)
    single = {}
    for probe in which_all:
        m = scale_mask & (row_obs == probe)
        if m.sum() == 0:
            continue
        _, sig, _ = masked_constraints(d0, J, m, rel_err, prior_sigma)
        single[probe] = sig
    cumulative = {}
    for i in range(1, len(which_all) + 1):
        subset = which_all[:i]
        m = scale_mask & np.isin(row_obs, subset)
        _, sig, _ = masked_constraints(d0, J, m, rel_err, prior_sigma)
        cumulative["+".join(subset)] = sig
    return {"single": single, "cumulative": cumulative}
