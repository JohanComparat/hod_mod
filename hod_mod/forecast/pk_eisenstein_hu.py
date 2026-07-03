r"""Differentiable Eisenstein & Hu (1998) linear power spectrum.

This wraps the repository's pure-JAX EH98 transfer function
(:func:`hod_mod.core.power_spectrum.eisenstein_hu_pk`) into a
``LinearPowerSpectrum``-compatible object whose amplitude is set directly by
``sigma8`` (rather than ``A_s``) and whose redshift dependence uses the
Carroll+1992 growth factor.  Because EH98 is analytic, the whole thing is
JAX-traceable and differentiable w.r.t. the cosmological parameters — which is
exactly what the Fisher forecast needs (CAMB is not JAX-traceable).

Two entry points are exposed on :class:`EisensteinHu98PkLinear`:

* ``pk_shape(k, theta)`` — the EH98 *shape* spectrum (normalised to
  ``P(0.05 h/Mpc) = 1``).  Feed this to :class:`HaloMassFunction` as its
  ``pk_func``: the HMF then rescales σ(M) so that σ(8 Mpc/h) = ``theta['sigma8']``
  and applies the growth factor internally (see
  :meth:`hod_mod.core.halo_mass_function.HaloMassFunction.sigma`).
* ``pk_linear(k, z, theta)`` — the σ8-normalised **physical** P(k, z)
  [(Mpc/h)³] used for the 2-halo term.  The σ8 normalisation is derived from the
  same shape spectrum via a top-hat σ(R=8) integral, so it is identical to the
  normalisation the HMF applies — the two paths stay consistent by construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hod_mod.core.power_spectrum import eisenstein_hu_pk
from hod_mod.core.halo_mass_function import growth_factor


# Wavenumber grid used for the σ(R=8) normalisation integral.  Matches the
# HaloMassFunction default (logspace(-4, 3, 512)) so the σ8 rescaling derived
# here is numerically identical to the one applied inside HaloMassFunction.sigma.
_K_INT = jnp.logspace(-4.0, 3.0, 512)


def _sigma2_tophat(pk: jnp.ndarray, k: jnp.ndarray, R: float) -> jnp.ndarray:
    r"""σ²(R) = (1/2π²) ∫ P(k) W²(kR) k² dk with a real-space top-hat W."""
    x = k * R
    w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x ** 3
    return jnp.trapezoid(pk * w ** 2 * k ** 2, k) / (2.0 * jnp.pi ** 2)


def _nu_suppression(k: jnp.ndarray, theta: dict) -> jnp.ndarray:
    r"""Massive-neutrino small-scale suppression of the linear P(k) shape.

    First-order, differentiable form: the asymptotic ΔP/P ≃ −8 f_ν
    (f_ν = Ω_ν/Ω_m, Ω_ν h² = Σm_ν/93.14 eV) switched on smoothly across the
    free-streaming scale with a tanh in ln k around
    k_nr ≈ 0.018 √(Ω_m m_ν,i/1 eV) h/Mpc (three degenerate species).
    Exactly 1 at Σm_ν = 0 (the massless fiducial is bit-identical); the small
    mass floor inside √ keeps the jacfwd column finite at 0.  A distilled CAMB
    ratio table is the documented accuracy upgrade (docs/missing_physics.rst).
    """
    mnu = theta["sum_mnu"]
    f_nu = (mnu / 93.14) / (theta["Omega_m"] * theta["h"] ** 2)
    k_nr = 0.018 * jnp.sqrt(theta["Omega_m"] * (mnu / 3.0 + 1e-6))
    step = 0.5 * (1.0 + jnp.tanh(jnp.log(k / k_nr) / 1.5))
    return 1.0 - 8.0 * f_nu * step


class EisensteinHu98PkLinear:
    """σ8-parameterised, JAX-differentiable EH98 linear power spectrum.

    Parameters
    ----------
    R8 : float
        Top-hat radius for the σ8 normalisation [Mpc/h] (8 by convention).
    """

    def __init__(self, R8: float = 8.0):
        self._R8 = float(R8)

    # -- shape spectrum, for the HMF -----------------------------------
    def pk_shape(self, k: jnp.ndarray, theta: dict) -> jnp.ndarray:
        """EH98 shape spectrum (normalised to P(0.05 h/Mpc)=1).

        When ``theta`` carries ``sum_mnu`` (a static dict-membership branch),
        the massive-neutrino suppression multiplies the shape — consistently
        here, in the σ8 anchor and in the HMF path, so σ8 keeps its measured-
        amplitude meaning and only the *shape* responds to Σm_ν.
        """
        k = jnp.asarray(k)
        pk = eisenstein_hu_pk(k, theta)
        if "sum_mnu" in theta:
            pk = pk * _nu_suppression(k, theta)
        return pk

    def _sigma8_shape2(self, theta: dict) -> jnp.ndarray:
        """σ²(R8) of the shape spectrum — the denominator of the σ8 rescale."""
        return _sigma2_tophat(self.pk_shape(_K_INT, theta), _K_INT, self._R8)

    # -- physical spectrum, for the 2-halo term ------------------------
    def pk_linear(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        r"""σ8-normalised physical P(k, z) [(Mpc/h)³].

        .. math::

            P(k, z) = P_\mathrm{shape}(k)\,
                      \frac{\sigma_8^2}{\sigma_{8,\mathrm{shape}}^2}\,
                      \left[\frac{D(z)}{D(0)}\right]^2
        """
        k = jnp.asarray(k)
        pk_shape = self.pk_shape(k, theta)
        norm = theta["sigma8"] ** 2 / self._sigma8_shape2(theta)
        growth = growth_factor(float(z), theta)   # D(z)/D(0); CPL ODE if w0/wa present
        return pk_shape * norm * growth ** 2

    # -- convenience: a plain callable for make_hmf(pk_func=...) --------
    def as_hmf_pk_func(self):
        """Return a ``(k, z, theta) -> P_shape`` callable for the HMF."""
        def _pk_func(k, z, theta):
            return self.pk_shape(k, theta)
        return _pk_func
