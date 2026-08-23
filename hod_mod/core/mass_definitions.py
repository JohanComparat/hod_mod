r"""Spherical-overdensity mass definitions and translation between them.

A halo mass is meaningless without the convention that produced it. This module
holds the convention -- :class:`MassDef` -- and the NFW-based translation
between conventions, :func:`translate_mass`.

Both are needed because published relations are calibrated in different
conventions and the pipeline mixes them: the CSST emulator returns
``RockstarM200m`` masses, the Diemer-Joyce concentration is defined at
:math:`\Delta = 200` critical, and the Arnaud pressure profile is a function of
:math:`M_{500c}`. Converting between them by hand at each call site is how
inconsistencies get in.

The translation assumes an NFW profile, which is the standard assumption and the
dominant systematic in the result: the enclosed mass

.. math::

    M(<r) = M_\Delta \frac{g(c_\Delta r/r_\Delta)}{g(c_\Delta)},
    \qquad g(y) = \ln(1+y) - \frac{y}{1+y}

is solved for the radius enclosing the target overdensity. The solve is a
fixed-step bisection rather than a root finder with a tolerance, so it is
jittable and differentiable.

Notes
-----
``gas.conversions.m200_to_m500c`` predates this module and is kept as a thin
wrapper for backwards compatibility.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .power_spectrum import rho_critical_0

_RHO_CRIT0 = rho_critical_0()

__all__ = ["MassDef", "translate_mass", "parse_mass_def", "rho_reference"]


class MassDef:
    """A spherical-overdensity mass definition.

    Parameters
    ----------
    delta : float or ``"vir"``
        Overdensity contrast. The string ``"vir"`` selects the
        Bryan & Norman (1998) virial contrast, evaluated with respect to the
        critical density.
    rho_type : {"matter", "critical"}
        The density the contrast is measured against. Ignored for ``"vir"``,
        which is always critical.

    Examples
    --------
    >>> MassDef(200, "matter")          # 200m, the hod_mod default
    >>> MassDef(500, "critical")        # 500c, the X-ray/SZ convention
    >>> MassDef("vir", "critical")      # virial
    """

    def __init__(self, delta, rho_type: str = "matter"):
        if isinstance(delta, str):
            if delta != "vir":
                raise ValueError(f"delta must be a number or 'vir', got '{delta}'")
            rho_type = "critical"
        elif delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if rho_type not in ("matter", "critical"):
            raise ValueError(
                f"rho_type must be 'matter' or 'critical', got '{rho_type}'"
            )
        self.delta = delta if isinstance(delta, str) else float(delta)
        self.rho_type = rho_type

    # -- construction ---------------------------------------------------
    @classmethod
    def from_string(cls, name: str) -> "MassDef":
        """Parse the shorthand used across the codebase: ``200m``, ``500c``, ``vir``."""
        n = name.strip().lower()
        if n in ("vir", "virial"):
            return cls("vir")
        if n.endswith("m"):
            return cls(float(n[:-1]), "matter")
        if n.endswith("c"):
            return cls(float(n[:-1]), "critical")
        raise ValueError(
            f"cannot parse mass definition '{name}'; expected e.g. '200m', '500c', 'vir'"
        )

    def __repr__(self) -> str:
        if self.delta == "vir":
            return "MassDef(vir)"
        return f"MassDef({self.delta:g}{'m' if self.rho_type == 'matter' else 'c'})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, MassDef)
                and self.delta == other.delta
                and self.rho_type == other.rho_type)

    # -- the physics ----------------------------------------------------
    def delta_rho(self, z: float, theta_cosmo: dict) -> tuple:
        """Return ``(delta, rho_ref)`` with ``rho_ref`` comoving [(Msun/h)/(Mpc/h)^3].

        ``rho_ref`` is comoving so that ``r_delta`` from :meth:`radius` is a
        comoving radius, matching the convention of Sec. 2.1 of the paper.
        ``theta_cosmo['Omega_m']`` may be a traced JAX value.
        """
        omega_m = theta_cosmo["Omega_m"]
        ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
        rho_crit_comoving = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3

        if self.delta == "vir":
            omega_m_z = omega_m * (1.0 + z) ** 3 / ez2
            x = omega_m_z - 1.0
            delta = 18.0 * jnp.pi ** 2 + 82.0 * x - 39.0 * x ** 2
            return delta, rho_crit_comoving
        if self.rho_type == "matter":
            return self.delta, omega_m * _RHO_CRIT0
        return self.delta, rho_crit_comoving

    def radius(self, m, z: float, theta_cosmo: dict):
        r"""Comoving halo radius, :math:`r_\Delta = [3M/(4\pi\Delta\rho_{\rm ref})]^{1/3}`."""
        delta, rho_ref = self.delta_rho(z, theta_cosmo)
        return (3.0 * jnp.asarray(m) / (4.0 * jnp.pi * delta * rho_ref)) ** (1.0 / 3.0)


def parse_mass_def(mdef) -> MassDef:
    """Accept a :class:`MassDef` or its string shorthand."""
    return mdef if isinstance(mdef, MassDef) else MassDef.from_string(mdef)


def rho_reference(mdef, z: float, theta_cosmo: dict) -> tuple:
    """``(delta, rho_ref)`` for *mdef*; convenience wrapper."""
    return parse_mass_def(mdef).delta_rho(z, theta_cosmo)


def _g_nfw(y):
    return jnp.log1p(y) - y / (1.0 + y)


def _dg_nfw(y):
    """g'(y) for g(y) = ln(1+y) - y/(1+y)."""
    return y / (1.0 + y) ** 2


@partial(jax.jit, static_argnums=(5,))
def _solve_radius_ratio(c_in, delta_in, rho_in, delta_out, rho_out, n_steps: int = 60):
    r"""Solve for :math:`x = r_{\rm out}/r_{\rm in}`, with a correct derivative.

    The condition is that the NFW mass enclosed within :math:`x\,r_{\rm in}`
    equals the target overdensity times the enclosed volume, which reduces to

    .. math::

        F(x) \equiv \frac{g(c\,x)}{g(c)}
                    - \frac{\Delta_{\rm out}\rho_{\rm out}}
                           {\Delta_{\rm in}\rho_{\rm in}}\,x^3 = 0 .

    :math:`F` decreases monotonically, so a fixed 60-step bisection on
    :math:`x\in[10^{-3},20]` brackets the root without data-dependent control
    flow.

    **Bisection alone is not differentiable.** Its output depends on the inputs
    only through the sign of :math:`F` at each midpoint, and a boolean carries
    no derivative: ``jax.grad`` through a bare bisection returns exactly zero,
    silently. The root is therefore refined by a single Newton step taken from
    a ``stop_gradient``-ed bracket,

    .. math::

        x = x_0 - F(x_0)/F'(x_0), \qquad x_0 = {\rm sg}[x_{\rm bisect}],

    which leaves the *value* unchanged to machine precision --- :math:`F(x_0)`
    is already ~1e-16 --- while giving the *derivative* the implicit-function
    result :math:`{\rm d}x/{\rm d}\theta = -(\partial F/\partial\theta)/
    (\partial F/\partial x)`. This is the standard implicit-differentiation
    trick for an iterative solve, and it is why gradients with respect to
    :math:`\Omega_{\rm m}` flow through a mass translation at all.
    """
    ratio = (delta_out * rho_out) / (delta_in * rho_in)
    gc = _g_nfw(c_in)

    def F(x):
        return _g_nfw(c_in * x) / gc - ratio * x ** 3

    def dF(x):
        return c_in * _dg_nfw(c_in * x) / gc - 3.0 * ratio * x ** 2

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        go_right = F(mid) > 0.0          # F decreasing: root is to the right
        return (jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid))

    shape = jnp.broadcast_shapes(jnp.shape(c_in), jnp.shape(ratio))
    lo = jnp.full(shape, 1e-3)
    hi = jnp.full(shape, 20.0)
    lo, hi = jax.lax.fori_loop(0, n_steps, body, (lo, hi))
    x0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    return x0 - F(x0) / dF(x0)


def translate_mass(m_in, c_in, mdef_in, mdef_out, z: float, theta_cosmo: dict):
    r"""Translate a halo mass between spherical-overdensity definitions.

    Assumes an NFW profile of concentration ``c_in`` *in the input definition*.

    Parameters
    ----------
    m_in : array_like
        Halo mass in the input definition [Msun/h].
    c_in : array_like
        NFW concentration :math:`r_{\rm in}/r_s`, i.e. defined with respect to
        ``mdef_in``. Broadcast against ``m_in``.
    mdef_in, mdef_out : :class:`MassDef` or str
        Input and output definitions, e.g. ``"200m"`` and ``"500c"``.
    z : float
    theta_cosmo : dict
        Needs ``Omega_m``; may be traced.

    Returns
    -------
    m_out, r_out, c_out
        Mass [Msun/h], comoving radius [Mpc/h] and concentration
        :math:`r_{\rm out}/r_s` in the output definition.

    Notes
    -----
    Identical definitions short-circuit and return the input unchanged, so
    round-tripping is exact rather than accurate-to-a-tolerance.
    """
    din = parse_mass_def(mdef_in)
    dout = parse_mass_def(mdef_out)

    m_in = jnp.asarray(m_in, dtype=float)
    c_in = jnp.asarray(c_in, dtype=float)
    r_in = din.radius(m_in, z, theta_cosmo)

    if din == dout:
        return m_in, r_in, c_in

    delta_in, rho_in = din.delta_rho(z, theta_cosmo)
    delta_out, rho_out = dout.delta_rho(z, theta_cosmo)

    x = _solve_radius_ratio(c_in, delta_in, rho_in, delta_out, rho_out)
    r_out = x * r_in
    m_out = m_in * _g_nfw(c_in * x) / _g_nfw(c_in)
    c_out = c_in * x
    return m_out, r_out, c_out
