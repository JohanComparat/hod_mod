r"""Vectorised evaluation of an observable across a batch of HOD parameters.

A multi-bin fit evaluates the same observable once per stellar-mass bin, and the
natural way to write that is a Python loop. On a JAX backend it is the wrong
way. Each iteration dispatches its own sequence of kernels and ends in a device
synchronisation, so the loop pays the launch overhead N times over and never
presents the hardware with more than one bin's worth of work.

:func:`map_over_hod` replaces the loop with a single ``jax.vmap``. The bins are
evaluated as one batched computation: kernels are launched once, arrays are N
times larger, and there is one synchronisation instead of N.

This matters for two separate reasons.

On CPU it is a straightforward win from removing dispatch overhead --- measured
at 5.6x for eight bins of :math:`w_{\rm p}`.

On GPU it is the difference between using the device and not. The per-stage
benchmarks of this package put the CPU/GPU crossover at roughly
:math:`10^4` elements of genuine arithmetic: below that, a kernel launch costs
more than the arithmetic it dispatches, and every small stage runs faster on
CPU. Batching eight bins multiplies the working set by eight and moves several
stages across that threshold at once.

Requirements
------------
The prediction must be JAX-traceable with respect to the batched parameters,
which means a predictor built by
:func:`~hod_mod.observables.clustering.make_differentiable_prediction`
(``pk_backend='eh98_jax'``). The CAMB backend cannot be traced and will raise.
"""
from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

__all__ = ["map_over_hod", "batched_wp", "batched_delta_sigma"]


def map_over_hod(
    predict: Callable[[dict], jnp.ndarray],
    base_params: dict,
    varying: dict[str, Sequence[float]],
) -> jnp.ndarray:
    r"""Evaluate ``predict`` over a batch of HOD parameter sets.

    Parameters
    ----------
    predict : callable
        ``predict(hod_params) -> array``. Must be JAX-traceable in the entries
        of ``hod_params`` named in ``varying``.
    base_params : dict
        Parameters shared by every member of the batch.
    varying : dict of str -> array
        The parameters that differ between bins. Every value must have the same
        leading length ``N``; that becomes the batch dimension.

    Returns
    -------
    array of shape ``(N,) + predict(...).shape``

    Examples
    --------
    >>> out = map_over_hod(
    ...     lambda p: pred.wp(rp, 60.0, 0.2, theta, p),
    ...     ZuMandelbaum15HODModel.default_params(),
    ...     {"log10m_star_thresh": [10.0, 10.5, 11.0]},
    ... )
    >>> out.shape
    (3, len(rp))

    Notes
    -----
    Parameters are stacked into one array before mapping so that a single
    ``vmap`` covers an arbitrary number of varying entries, and so that the
    batch dimension is unambiguous even when only one parameter varies.
    """
    if not varying:
        raise ValueError("`varying` is empty: nothing to batch over")

    keys = list(varying)
    cols = [jnp.atleast_1d(jnp.asarray(varying[k], dtype=float)) for k in keys]
    lengths = {c.shape[0] for c in cols}
    if len(lengths) != 1:
        raise ValueError(
            "every entry of `varying` must have the same length; got "
            + ", ".join(f"{k}: {c.shape[0]}" for k, c in zip(keys, cols))
        )
    overlap = set(keys) - set(base_params)
    if overlap:
        # Not fatal, but almost always a typo: a batched key the model will not read.
        raise KeyError(
            f"varying parameter(s) {sorted(overlap)} are absent from base_params; "
            "check the spelling against the model's default_params()"
        )

    stacked = jnp.stack(cols, axis=-1)          # (N, n_keys)

    def _one(row):
        params = dict(base_params)
        for i, k in enumerate(keys):
            params[k] = row[i]
        return predict(params)

    return jax.vmap(_one)(stacked)


def batched_wp(pred, rp, pi_max: float, z: float, theta_cosmo: dict,
               base_params: dict, varying: dict) -> jnp.ndarray:
    """:math:`w_{\\rm p}(r_{\\rm p})` for a batch of HOD parameter sets.

    Thin wrapper over :func:`map_over_hod`; see there for the arguments.
    """
    return map_over_hod(
        lambda p: pred.wp(jnp.asarray(rp), pi_max, z, theta_cosmo, p),
        base_params, varying,
    )


def batched_delta_sigma(pred, R, z: float, theta_cosmo: dict,
                        base_params: dict, varying: dict) -> jnp.ndarray:
    """:math:`\\Delta\\Sigma(R)` for a batch of HOD parameter sets."""
    return map_over_hod(
        lambda p: pred.delta_sigma(jnp.asarray(R), z, theta_cosmo, p),
        base_params, varying,
    )
