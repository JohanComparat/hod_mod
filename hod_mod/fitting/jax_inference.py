r"""Differentiable multi-probe Gaussian inference on the forecast forward model.

Wave 3.  The forecast :class:`hod_mod.forecast.forward_jax.ForwardModel` already
computes every probe (galaxy clustering ``wp``, galaxy-galaxy lensing ``ds``,
tSZ ``cl_gy``, X-ray ``cl_gX``/``cl_XX``, cosmic shear ``cl_kk``, AGN ``xlf``,
stellar-mass function ``smf``, CMB lensing, ...) as one ``jax.jacfwd``-able call
in the σ8-native EH98 parameterisation.  This module wraps that into a Gaussian
likelihood whose ``logprob`` is ``jax.value_and_grad``-able over a chosen set of
free parameters, and drives it with a gradient MAP optimiser and blackjax NUTS.

The whole point is that the gradient flows: unlike the CAMB-based
:mod:`hod_mod.fitting.fitters`, nothing here casts to numpy or ``float`` inside
the objective.  Run under ``JAX_ENABLE_X64=1`` for reliable gradients.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.forecast import params as _fparams


class MultiProbeGaussianLikelihood:
    r"""Gaussian log-likelihood over a concatenated multi-probe data vector.

    .. math::

        \ln\mathcal{L}(\theta) = -\tfrac12\,[f(\theta)-d]^\top C^{-1}[f(\theta)-d]
                                 -\tfrac12\sum_i \frac{(\theta_i-\mu_i)^2}{\sigma_i^2}

    where :math:`f(\theta)` is the forward-model data vector (a subset of the 111
    forecast parameters is free; the rest are held at ``theta0``), :math:`d` /
    :math:`C^{-1}` the data and inverse covariance, and the second term a diagonal
    Gaussian parameter prior (Planck on cosmology by default; flat elsewhere).

    Construct via :meth:`from_forward_model` (real/synthetic data) rather than the
    raw ``__init__``.
    """

    def __init__(self, model_fn, data, icov, free_idx, theta0,
                 prior_mean=None, prior_sigma=None):
        self._f = model_fn                                   # f(theta_full)->vec
        self._data = jnp.asarray(data)
        self._icov = jnp.asarray(icov)
        self._free_idx = jnp.asarray(np.asarray(free_idx), dtype=int)
        self._theta0 = jnp.asarray(theta0)
        n_free = self._free_idx.shape[0]
        mu = np.zeros(n_free) if prior_mean is None else np.asarray(prior_mean, float)
        sig = np.full(n_free, np.inf) if prior_sigma is None \
            else np.asarray(prior_sigma, float)
        self._prior_mean = jnp.asarray(mu)
        # precision 1/sigma^2; sigma=inf -> 0 (flat), so the prior term vanishes
        self._prior_prec = jnp.asarray(np.where(np.isfinite(sig) & (sig > 0),
                                                1.0 / np.where(sig > 0, sig, 1.0) ** 2,
                                                0.0))

    # -- parameter packing ------------------------------------------------
    def unpack(self, x):
        """Insert the free vector ``x`` into the fixed full parameter vector."""
        return self._theta0.at[self._free_idx].set(x)

    def model(self, x):
        """Forward-model data vector at free parameters ``x``."""
        return self._f(self.unpack(x))

    # -- log-probability terms -------------------------------------------
    def log_prior(self, x):
        d = x - self._prior_mean
        return -0.5 * jnp.sum(self._prior_prec * d * d)

    def log_likelihood(self, x):
        r = self.model(x) - self._data
        return -0.5 * (r @ (self._icov @ r))

    def logprob(self, x):
        """Log posterior density (up to a constant) — the NUTS target."""
        return self.log_likelihood(x) + self.log_prior(x)

    def neg_logprob(self, x):
        """``-logprob`` — the MAP objective to minimise."""
        return -self.logprob(x)

    def chi2(self, x):
        """Data :math:`\\chi^2` (prior excluded), for reporting/goodness-of-fit."""
        r = self.model(x) - self._data
        return float(r @ (self._icov @ r))

    @property
    def free_names(self):
        return [_fparams.PARAM_NAMES[int(i)] for i in np.asarray(self._free_idx)]

    def params_dict(self, x):
        """Map a free vector to a ``{name: value}`` dict."""
        return {n: float(v) for n, v in zip(self.free_names, np.asarray(x))}

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_forward_model(cls, forward_model, which, data, icov, free_names,
                           rmin=None, theta0=None, prior="planck"):
        """Build a likelihood from a :class:`ForwardModel` and measured data.

        Parameters
        ----------
        forward_model : ForwardModel
        which : list[str]
            Observables to include (e.g. ``["wp","ds","cl_gy","cl_gX","cl_kk","xlf"]``).
        data, icov : array
            Concatenated data vector and inverse covariance matching the
            forward-model output for ``which`` (and ``rmin`` scale cut if given).
        free_names : list[str]
            Free parameter names (subset of ``PARAM_NAMES``); order sets the
            free-vector layout.
        rmin : float, optional
            Scale cut passed to ``ForwardModel.data_vector_fn`` (fixed output
            shape); when ``None`` the full grids are used.
        theta0 : array, optional
            Full 111-vector for the fixed parameters (default: fiducial).
        prior : {"planck", None} or (mean, sigma)
            ``"planck"`` applies the tight Planck Ω_m/σ8 prior (flat elsewhere);
            ``None`` is flat; or pass ``(mean, sigma)`` arrays over the free params.
        """
        if rmin is not None:
            f, _ = forward_model.data_vector_fn(list(which), rmin)
        else:
            f, _, _ = forward_model.full_data_vector_fn(list(which))
        theta0 = _fparams.fiducial_vector() if theta0 is None else np.asarray(theta0)
        free_idx = [_fparams._IDX[n] for n in free_names]

        prior_mean = prior_sigma = None
        if prior == "planck":
            sig_full = _fparams.prior_sigma_vector()          # inf except Ω_m, σ8
            prior_mean = theta0[free_idx]
            prior_sigma = sig_full[free_idx]
        elif isinstance(prior, (tuple, list)):
            prior_mean, prior_sigma = np.asarray(prior[0]), np.asarray(prior[1])
        return cls(f, data, icov, free_idx, theta0, prior_mean, prior_sigma)

    @classmethod
    def synthetic(cls, forward_model, which, free_names, theta_true=None,
                  rel_err=0.05, rmin=None, prior="planck", seed=0, cov="diag"):
        """A likelihood with data = model(θ_true) + Gaussian noise (for tests/demos).

        Returns ``(likelihood, x_true)`` where ``x_true`` is the free-vector slice
        of ``theta_true``.  ``rel_err`` sets a diagonal fractional error per bin.
        """
        theta0 = _fparams.fiducial_vector() if theta_true is None else np.asarray(theta_true)
        if rmin is not None:
            f, _ = forward_model.data_vector_fn(list(which), rmin)
        else:
            f, _, _ = forward_model.full_data_vector_fn(list(which))
        truth = np.asarray(f(jnp.asarray(theta0)), dtype=float)
        sigma = np.abs(truth) * float(rel_err) + 1e-30
        rng = np.random.default_rng(seed)
        data = truth + sigma * rng.standard_normal(truth.shape)
        icov = np.diag(1.0 / sigma ** 2)
        free_idx = [_fparams._IDX[n] for n in free_names]
        prior_mean = prior_sigma = None
        if prior == "planck":
            sig_full = _fparams.prior_sigma_vector()
            prior_mean = theta0[free_idx]
            prior_sigma = sig_full[free_idx]
        like = cls(f, data, icov, free_idx, theta0, prior_mean, prior_sigma)
        return like, np.asarray(theta0)[free_idx]


# ---------------------------------------------------------------------------
# Gradient MAP
# ---------------------------------------------------------------------------

def run_map_jax(likelihood, x0, bounds=None, method="L-BFGS-B", maxiter=2000):
    """Gradient-based MAP via ``scipy.optimize.minimize`` with the JAX gradient.

    Uses ``jax.value_and_grad(likelihood.neg_logprob)`` as a jitted ``jac=True``
    objective — a genuine gradient descent, unlike the gradient-free Powell in
    :mod:`hod_mod.fitting.fitters`.

    Returns a dict: ``x`` (free MAP vector), ``params`` (name→value), ``logprob``,
    ``chi2``, ``success``, ``message``, ``nfev``.
    """
    from scipy.optimize import minimize

    vg = jax.jit(jax.value_and_grad(likelihood.neg_logprob))

    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(fun, np.asarray(x0, dtype=np.float64), jac=True,
                   method=method, bounds=bounds,
                   options={"maxiter": maxiter})
    return {
        "x": res.x,
        "params": likelihood.params_dict(res.x),
        "logprob": float(likelihood.logprob(jnp.asarray(res.x))),
        "chi2": likelihood.chi2(jnp.asarray(res.x)),
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev),
    }


# ---------------------------------------------------------------------------
# Hamiltonian Monte Carlo (blackjax NUTS)
# ---------------------------------------------------------------------------

def run_nuts(likelihood, x0, n_warmup=500, n_samples=1000, seed=0,
             target_acceptance=0.8):
    """Sample the posterior with blackjax NUTS (window-adaptation warmup).

    Because ``likelihood.logprob`` is fully differentiable, NUTS gets exact
    gradients — the payoff of the whole JAX port for posterior inference.
    ``blackjax`` is an optional dependency (``pip install blackjax``).

    Performance note: the forecast angular spectra (``cl_gy``/``cl_gX``/``cl_kk``
    etc.) do a Limber projection over the redshift grid, which unrolls into the
    NUTS trajectory ``while_loop`` and inflates the compile ~10×.  MAP
    (:func:`run_map_jax`) handles the full multi-probe vector cheaply; for NUTS
    prefer projected/abundance observables (``wp``/``ds``/``xlf``/``smf``), a
    small ``n_z``, or fewer samples when angular spectra are included.

    Returns a dict: ``samples`` (n_samples × n_free), ``mean``, ``std``,
    ``params`` (name→mean), ``accept_rate``, ``step_size``.
    """
    try:
        import blackjax
    except ImportError as e:                                    # pragma: no cover
        raise ImportError(
            "run_nuts needs blackjax — `pip install blackjax`.") from e

    logdensity = likelihood.logprob
    rng = jax.random.PRNGKey(seed)
    rng, warmup_key, sample_key = jax.random.split(rng, 3)

    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity, target_acceptance_rate=target_acceptance)
    (state, parameters), _ = warmup.run(
        warmup_key, jnp.asarray(x0, dtype=float), num_steps=n_warmup)

    kernel = blackjax.nuts(logdensity, **parameters)

    @jax.jit
    def one_step(carry, key):
        st, _info = kernel.step(key, carry)
        return st, (st.position, _info.acceptance_rate)

    keys = jax.random.split(sample_key, n_samples)
    _, (positions, accept) = jax.lax.scan(one_step, state, keys)

    samples = np.asarray(positions)
    return {
        "samples": samples,
        "mean": samples.mean(axis=0),
        "std": samples.std(axis=0),
        "params": {n: float(m) for n, m in
                   zip(likelihood.free_names, samples.mean(axis=0))},
        "accept_rate": float(np.asarray(accept).mean()),
        "step_size": float(parameters.get("step_size", np.nan)),
    }
