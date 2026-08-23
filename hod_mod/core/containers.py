r"""JAX pytree containers for parameter sets.

A plain ``dict`` of cosmological parameters works, but it makes one failure mode
easy to write and impossible to see: a value extracted with ``float()`` inside a
traced function silently leaves the computation graph. If the parameter is used,
JAX raises a ``ConcretizationTypeError`` -- loud, and therefore harmless. If the
parameter is *not* used by the branch being taken, the derivative with respect
to it is returned as exactly zero and nothing complains. A Fisher matrix built
on such a branch reports infinite confidence rather than failing.

:class:`ParameterContainer` makes the distinction explicit. Values registered as
``params`` are pytree leaves and are traced; values registered as ``config`` are
static metadata, live in the treedef, and are compared by identity when JAX
decides whether to recompile. Attempting to differentiate with respect to a
config entry is then a structural impossibility rather than a silent zero.

The design follows the container pattern of ``jax_cosmo``
(Campagne et al. 2023, OJAp 6, 15).

Examples
--------
>>> c = Cosmology(Omega_m=0.31, sigma8=0.81, h=0.6736, n_s=0.9649, Omega_b=0.0493)
>>> jax.grad(lambda c: c.Omega_m ** 2)(c).Omega_m      # 0.62
>>> c.to_dict()["Omega_m"]                             # interop with dict APIs
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["ParameterContainer", "Cosmology"]


class ParameterContainer:
    """Base class for pytree-registered parameter sets.

    Subclasses declare ``_PARAMS`` (traced, differentiable) and optionally
    ``_CONFIG`` (static). Both are exposed as attributes.
    """

    _PARAMS: tuple[str, ...] = ()
    _CONFIG: tuple[str, ...] = ()

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(self._PARAMS) - set(self._CONFIG)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected parameter(s) "
                f"{sorted(unknown)}; known parameters are {list(self._PARAMS)} "
                f"and config keys {list(self._CONFIG)}"
            )
        missing = set(self._PARAMS) - set(kwargs)
        if missing:
            raise TypeError(
                f"{type(self).__name__} is missing required parameter(s) "
                f"{sorted(missing)}"
            )
        self._p = {k: kwargs[k] for k in self._PARAMS}
        self._c = {k: kwargs.get(k) for k in self._CONFIG}

    # -- attribute access ----------------------------------------------
    def __getattr__(self, name: str):
        # __getattr__ only fires when normal lookup fails, so _p/_c are safe
        for store in ("_p", "_c"):
            d = object.__getattribute__(self, store)
            if name in d:
                return d[name]
        raise AttributeError(
            f"{type(self).__name__!r} has no parameter {name!r}"
        )

    # -- pytree protocol -----------------------------------------------
    def tree_flatten(self):
        """Traced leaves are the params; config travels in the static treedef."""
        children = tuple(self._p[k] for k in self._PARAMS)
        aux = tuple(self._c[k] for k in self._CONFIG)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj._p = dict(zip(cls._PARAMS, children))
        obj._c = dict(zip(cls._CONFIG, aux))
        return obj

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        jax.tree_util.register_pytree_node(
            cls, lambda o: o.tree_flatten(), cls.tree_unflatten
        )

    # -- interop --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain dict, for the many APIs that still take one."""
        return {**self._p, **{k: v for k, v in self._c.items() if v is not None}}

    @classmethod
    def from_dict(cls, d: dict):
        """Build from a dict, ignoring keys this container does not declare."""
        known = set(cls._PARAMS) | set(cls._CONFIG)
        return cls(**{k: v for k, v in d.items() if k in known})

    def replace(self, **kwargs):
        """Return a copy with some parameters changed."""
        return type(self)(**{**self.to_dict(), **kwargs})

    def __repr__(self) -> str:
        inner = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in self._p.items()
        )
        return f"{type(self).__name__}({inner})"


class Cosmology(ParameterContainer):
    """Cosmological parameters, differentiable with respect to every entry.

    The parameter set matches the dictionary the rest of the package already
    uses, so :meth:`to_dict` is a drop-in for existing call sites.

    Parameters
    ----------
    Omega_m, sigma8, h, n_s, Omega_b : float
        Required.
    Omega_cdm, ln10_10_As, w0, wa, sum_mnu : float, optional
        Optional; defaults are the flat-LCDM, massless-neutrino values.
    transfer, mass_def : str, optional
        Static configuration. Not differentiable *by construction* -- that is
        the point of separating them.
    """

    _PARAMS = ("Omega_m", "sigma8", "h", "n_s", "Omega_b",
               "Omega_cdm", "ln10_10_As", "w0", "wa", "sum_mnu")
    _CONFIG = ("transfer", "mass_def")

    def __init__(self, Omega_m, sigma8, h, n_s, Omega_b,
                 Omega_cdm=None, ln10_10_As=3.044, w0=-1.0, wa=0.0,
                 sum_mnu=0.0, transfer="cosmopower", mass_def="200m"):
        if Omega_cdm is None:
            Omega_cdm = Omega_m - Omega_b
        super().__init__(
            Omega_m=Omega_m, sigma8=sigma8, h=h, n_s=n_s, Omega_b=Omega_b,
            Omega_cdm=Omega_cdm, ln10_10_As=ln10_10_As, w0=w0, wa=wa,
            sum_mnu=sum_mnu, transfer=transfer, mass_def=mass_def,
        )

    # -- the dict spelling the rest of the package expects ---------------
    def to_theta(self) -> dict:
        """The legacy ``theta_cosmo`` dict, with its original key spellings."""
        return {
            "Omega_m": self.Omega_m, "sigma8": self.sigma8, "h": self.h,
            "n_s": self.n_s, "Omega_b": self.Omega_b,
            "Omega_cdm": self.Omega_cdm,
            "ln10^{10}A_s": self.ln10_10_As,
            "w0": self.w0, "wa": self.wa,
        }

    @classmethod
    def planck18(cls, **overrides):
        """Planck 2018 TT,TE,EE+lowE best fit."""
        base = dict(Omega_m=0.3100, sigma8=0.8111, h=0.6736,
                    n_s=0.9649, Omega_b=0.0493, ln10_10_As=3.044)
        return cls(**{**base, **overrides})
