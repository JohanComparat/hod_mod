r"""Analytic Gaussian covariance for the Fisher forecast.

The forecast is a Gaussian-likelihood Fisher analysis with the covariance held
**fixed at the fiducial** (only the mean/data-vector is differentiated), so the
field spectra entering the covariance are computed once, not through ``jacfwd``.

This module upgrades the previous *diagonal* (per-bin fractional-error) model by
adding **cross-observable** correlations.  We keep the validated per-bin diagonal
(σ_i from :func:`run_stage4_forecast.per_bin_relerr`, i.e. Knox mode-counting +
shot/shape noise) and inject off-diagonal terms via the Gaussian correlation
coefficient

.. math::

    \rho_{ij}(\ell) = \frac{\tilde C^{ac}\tilde C^{bd}+\tilde C^{ad}\tilde C^{bc}}
        {\sqrt{[\tilde C^{aa}\tilde C^{bb}+(\tilde C^{ab})^2]
               [\tilde C^{cc}\tilde C^{dd}+(\tilde C^{cd})^2]}},

for observables :math:`O_i=C^{ab}`, :math:`O_j=C^{cd}`.  The survey-area /
mode-count prefactor cancels in ρ, so ``Cov_ij = ρ_ij σ_i σ_j`` cleanly combines
the correlation *shape* with the validated per-bin errors.  ρ is diagonal in ℓ
(Gaussian), so cross-observable terms couple only matching ℓ-bins.

**v1 scope:** ρ is computed exactly for the **lensing triplet**
``{cl_kk, cl_shear_kCMB, cl_kCMB}`` — three probes of the *same* matter field,
hence the most strongly correlated, and whose required field spectra
(:math:`C^{\kappa\kappa}, C^{\kappa\kappa_c}, C^{\kappa_c\kappa_c}`) are all
already forecast observables.  Other cross-observable blocks (gas/galaxy sectors)
default to zero correlation and are the documented next increment; the framework
below extends to them once the extra field spectra are provided.
"""

from __future__ import annotations

import numpy as np

# nominal reconstruction/shape noise for the lensing fields (Stage-IV-like),
# as a fraction of the fiducial auto-signal at the ℓ-pivot — only used to soften
# ρ towards 0 at noise-dominated ℓ; ρ is insensitive to the exact values.
_NOISE_FRAC = {"kk": 0.3, "kckc": 1.0}   # shear vs CMB-lensing auto noise-to-signal


def _field_pair(obs_name):
    """Map an angular observable to its two fields."""
    return {"cl_kk": ("k", "k"), "cl_kCMB": ("kc", "kc"),
            "cl_shear_kCMB": ("k", "kc"), "cl_gX": ("g", "X"), "cl_gy": ("g", "y"),
            "cl_XX": ("X", "X"), "cl_gkCMB": ("g", "kc")}.get(obs_name)


def _lensing_rho(pred, ell):
    """ρ_ij(ℓ) among the lensing triplet from the fiducial field spectra."""
    Ckk = np.asarray(pred["cl_kk"]);  Ckckc = np.asarray(pred["cl_kCMB"])
    Ckkc = np.asarray(pred["cl_shear_kCMB"])
    # total (signal+noise) autos; cross carries no extra noise
    S = {("k", "k"): Ckk * (1.0 + _NOISE_FRAC["kk"]),
         ("kc", "kc"): Ckckc * (1.0 + _NOISE_FRAC["kckc"]),
         ("k", "kc"): Ckkc, ("kc", "k"): Ckkc}

    def C(a, b):
        return S[(a, b)]

    def cov(ab, cd):
        a, b = ab; c, d = cd
        return C(a, c) * C(b, d) + C(a, d) * C(b, c)

    names = ["cl_kk", "cl_shear_kCMB", "cl_kCMB"]
    fp = {n: _field_pair(n) for n in names}
    rho = {}
    for i in names:
        for j in names:
            num = cov(fp[i], fp[j])
            den = np.sqrt(cov(fp[i], fp[i]) * cov(fp[j], fp[j]))
            rho[(i, j)] = np.where(den > 0, num / den, 0.0)
    return rho, names


def gaussian_covariance(model, theta, d0, row_obs, row_x, rel_err):
    r"""Full data-vector covariance: per-bin diagonal + lensing-triplet cross-terms.

    Parameters
    ----------
    model : ForwardModel
    theta : fiducial parameter vector
    d0 : fiducial data vector (full, unmasked), shape (N,)
    row_obs, row_x : per-row observable name and abscissa (from ``full_data_vector_fn``)
    rel_err : per-row fractional error σ_i/d_i (e.g. from ``per_bin_relerr``)

    Returns
    -------
    C : (N, N) covariance matrix (symmetric, positive-definite).
    """
    d0 = np.asarray(d0, dtype=float)
    sig = np.abs(np.asarray(rel_err, dtype=float) * d0)
    C = np.diag(sig ** 2)                                  # validated per-bin diagonal

    row_obs = np.asarray(row_obs)
    # fiducial field spectra for the lensing triplet (on model.ell)
    import jax.numpy as jnp
    pred = {o: np.asarray(model.predict(jnp.asarray(theta), [o])[o])
            for o in ("cl_kk", "cl_shear_kCMB", "cl_kCMB")}
    rho, names = _lensing_rho(pred, np.asarray(model.ell))

    # row indices of each lensing-triplet observable, matched by ℓ (same grid/order)
    idx = {n: np.where(row_obs == n)[0] for n in names}
    for i in names:
        for j in names:
            if i == j:
                continue
            ri, rj = idx[i], idx[j]
            if ri.size == 0 or rj.size == 0:
                continue
            r = rho[(i, j)]                                # (Nell,)
            # same ℓ-bin only (both live on model.ell in identical order)
            n = min(ri.size, rj.size)
            C[ri[:n], rj[:n]] = r[:n] * sig[ri[:n]] * sig[rj[:n]]
    # symmetrise for numerical safety
    C = 0.5 * (C + C.T)
    return C
