r"""Conditional galaxy morphology — the early-type fraction of the halo.

Missing-physics wave 4 (docs/missing_physics.rst, "Galaxy morphology"):
a Weibull conditional early-type fraction mirroring the Zu & Mandelbaum
(2016/2017) halo-quenching pattern of
:func:`hod_mod.connection.hod.zumandelbaum15.f_red_cen_zu16`,

.. math::

    f_{\rm early,c}(M_h) = 1 - \exp\!\left[-\left(\frac{M_h}{M_{\rm morph}}
    \right)^{\beta_{\rm morph}}\right],

with a satellite boost toward early types (environmental transformation),

.. math::

    f_{\rm early,s}(M_h) = f_{\rm early,c}(M_h)
    + f_{\rm morph,sat}\,[1 - f_{\rm early,c}(M_h)] .

Both are ∈ [0, 1] by construction for :math:`f_{\rm morph,sat} \in [0, 1]`,
and EARLY + LATE sums exactly to the unsplit occupation (the SF/Q-split
invariant).  The mean :math:`f_{\rm early,c}` also serves as the bulge-to-
total proxy that couples morphology to the black-hole sector
(:math:`M_{\rm BH} \propto (B/T\,M_*)`-like; Yang et al. 2019) inside the
forecast's Powell chain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def f_early_cen(
    log10m_h: jnp.ndarray,
    lg_m_morph: float,
    beta_morph: float,
) -> jnp.ndarray:
    r"""Early-type fraction of central galaxies (Weibull in halo mass).

    Parameters
    ----------
    log10m_h : log10(M_h / [M_sun/h])
    lg_m_morph : log10(M_morph / [M_sun/h]), morphological transition mass
    beta_morph : Weibull shape (transition sharpness)

    Output ∈ [0, 1] by construction; → 1 for M_h ≫ M_morph.
    """
    m_h = jnp.power(10.0, log10m_h)
    m_morph = jnp.power(10.0, lg_m_morph)
    return 1.0 - jnp.exp(-jnp.power(m_h / m_morph, beta_morph))


@jax.jit
def f_early_sat(
    log10m_h: jnp.ndarray,
    lg_m_morph: float,
    beta_morph: float,
    f_morph_sat: float,
) -> jnp.ndarray:
    r"""Early-type fraction of satellite galaxies: the central Weibull with
    an environmental boost toward early types,
    f_s = f_c + f_morph_sat (1 − f_c) — ∈ [0, 1] for f_morph_sat ∈ [0, 1],
    and identical to the central fraction at f_morph_sat = 0."""
    fc = f_early_cen(log10m_h, lg_m_morph, beta_morph)
    return fc + f_morph_sat * (1.0 - fc)
