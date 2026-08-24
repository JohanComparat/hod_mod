r"""Derived cosmological quantities and their Fisher error propagation.

:math:`\sigma_8` and :math:`S_8` are not parameters of this package -- the
amplitude is ``ln10^{10}A_s`` and nothing else.  They are still the quantities a
reader wants a forecast quoted in, so they are computed *from* the parameter
vector here, and their forecast errors are propagated through the exact
gradient rather than a hand-written one.

Why this module exists at all
-----------------------------
Before the retirement, ``sigma8`` sat at index 1 of ``PARAM_NAMES``, and
``run_tier2_forecast`` and ``run_stage4_forecast`` each carried a small
``_s8_sigma`` that read it out and propagated

.. math::

    \frac{\partial S_8}{\partial\Omega_m}
        = \tfrac12\,\sigma_8\,(\Omega_m/0.3)^{1/2}/\Omega_m ,
    \qquad
    \frac{\partial S_8}{\partial\sigma_8} = (\Omega_m/0.3)^{1/2}.

**Those were correct in that basis** and it is worth saying so, because the
number changes here and it would be easy to read that as a bug fix.  With
:math:`\sigma_8` an independent Fisher coordinate, :math:`S_8` is an exact
function of two coordinates and the partial at fixed :math:`\sigma_8` *is* the
whole Jacobian row -- the :math:`\Omega_m`--:math:`\sigma_8` correlation lives in
``cov``, not in the gradient.  At Planck-2018 that partial is **1.336**.

Once the amplitude is :math:`A_s`, :math:`\sigma_8` is no longer a coordinate.
It acquires a genuine :math:`\Omega_m` dependence -- through :math:`T(k)` and
through :math:`D^2(0)/\Omega_m^2` in the primordial normalisation -- so the
*total* derivative is what must be propagated, and it is **3.334**.  Not a
correction to the old number: a different derivative, because the chart changed.

Consequently :math:`\sigma(S_8)` from this module is not expected to reproduce
the previously published value exactly, while :math:`\sigma(\sigma_8)` is (a
pure reparameterisation of the same constraint).  A large move in
:math:`\sigma(\sigma_8)` means something is wrong here, not that a bug was fixed.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from hod_mod.core.amplitude import S8_PIVOT_OMEGA_M, sigma8_from_pk
from hod_mod.forecast.forward_jax import PARAM_NAMES, _IDX

__all__ = ["sigma8_of", "s8_of", "sigma_of", "cov_of", "fom_of",
           "s8_sigma", "sigma8_sigma"]


@functools.lru_cache(maxsize=1)
def _default_model():
    """A throwaway :class:`ForwardModel` used only for its cosmology and P(k).

    :math:`\sigma_8` depends on ``theta`` only through ``_cosmo`` and ``_pk``,
    neither of which knows about the mass grid, the redshift shell or the
    stellar-mass cell -- so the small grids here cost nothing and change
    nothing.  Callers that already have a model should pass it, both to save the
    construction and to be sure the backend matches (a ``camb_ratio``-corrected
    EH98 and a bare one give slightly different σ8).  The tiered drivers build
    many per-cell models and have no canonical one, which is why this exists.
    """
    from hod_mod.forecast.forward_jax import ForwardModel
    return ForwardModel(n_k=128, n_m=64, n_gl=16)


def sigma8_of(theta, model=None):
    r""":math:`\sigma_8` for parameter vector ``theta``, on ``model``'s backend.

    Goes through the model's ``_pk`` and ``_cosmo``, so it picks up whichever
    backend the forecast was built with (EH98 or CosmoPower) and any
    ``camb_ratio`` correction, rather than re-deriving a spectrum that could
    drift from the one the forecast actually differentiates.

    Differentiable in every entry of ``theta``.  Evaluated at ``z = 0``
    directly -- see :mod:`hod_mod.core.amplitude` for why a growth call must not
    appear here.
    """
    from hod_mod.core.amplitude import K_INT
    model = _default_model() if model is None else model
    c = model._cosmo(jnp.asarray(theta))
    return sigma8_from_pk(model._pk.pk_linear(K_INT, 0.0, c), K_INT)


def s8_of(theta, model=None):
    r""":math:`S_8 = \sigma_8\sqrt{\Omega_m/0.3}` for parameter vector ``theta``."""
    om = jnp.asarray(theta)[_IDX["Omega_m"]]
    return sigma8_of(theta, model) * jnp.sqrt(om / S8_PIVOT_OMEGA_M)


def sigma_of(cov, fid, fn):
    r"""Forecast :math:`\sigma` of a derived scalar, and its fiducial value.

    Linear propagation :math:`\sigma^2 = g^\top C\, g` with
    :math:`g = \nabla_\theta f` from ``jax.grad`` -- exact to first order in the
    same sense the Fisher matrix itself is.

    Parameters
    ----------
    cov : (N, N) array
        Parameter covariance, in ``PARAM_NAMES`` order.
    fid : (N,) array
        Fiducial parameter vector.
    fn : callable
        ``theta -> scalar``, e.g. ``lambda t: s8_of(model, t)``.

    Returns
    -------
    (sigma, value) : tuple of float
    """
    fid = jnp.asarray(fid)
    g = np.asarray(jax.grad(lambda t: jnp.asarray(fn(t)).reshape(()))(fid))
    var = float(g @ np.asarray(cov) @ g)
    return float(np.sqrt(max(var, 0.0))), float(fn(fid))


def cov_of(cov, fid, fns):
    r"""Covariance of several derived scalars: :math:`J C J^\top`.

    What a confidence ellipse in derived coordinates needs -- e.g. the
    :math:`(\Omega_m, \sigma_8)` plane, which is no longer a 2x2 slice of the
    parameter covariance now that :math:`\sigma_8` is derived.

    Parameters
    ----------
    fns : sequence of callables
        Each ``theta -> scalar``.

    Returns
    -------
    (m, m) ndarray
    """
    fid = jnp.asarray(fid)
    J = np.stack([np.asarray(jax.grad(lambda t, f=f: jnp.asarray(f(t)).reshape(()))(fid))
                  for f in fns])
    return J @ np.asarray(cov) @ J.T


def fom_of(cov, fid, fns):
    r"""Figure of merit :math:`1/\sqrt{\det}` for a pair of derived scalars.

    The counterpart of :func:`hod_mod.forecast.fisher.figure_of_merit`, which
    takes positional indices into the parameter covariance and so cannot express
    a plane one of whose axes is derived.
    """
    sub = cov_of(cov, fid, fns)
    det = float(np.linalg.det(sub))
    return float(1.0 / np.sqrt(det)) if det > 0 else float("inf")


def sigma8_sigma(cov, fid, model=None):
    r"""``(sigma(sigma_8), sigma_8)`` -- the common case, spelled once."""
    return sigma_of(cov, fid, lambda t: sigma8_of(t, model))


def s8_sigma(cov, fid, model=None):
    r"""``(sigma(S_8), S_8)`` -- the common case, spelled once.

    Drop-in for the ``_s8_sigma`` that ``run_tier2_forecast`` and
    ``run_stage4_forecast`` each carried, which cannot work now that
    ``_IDX["sigma8"]`` does not exist.  See the module docstring for why the
    number it returns is not the number those produced.
    """
    return sigma_of(cov, fid, lambda t: s8_of(t, model))
