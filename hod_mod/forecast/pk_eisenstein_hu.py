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


def _nu_suppression(k: jnp.ndarray, theta: dict, z=0.0) -> jnp.ndarray:
    r"""Massive-neutrino suppression of the linear P(k) shape.

    :math:`P_{\rm m}(k,z;\Sigma m_\nu)/P_{\rm m}(k,z;0)`, read from the
    CLASS-distilled table of :mod:`hod_mod.core.nu_ratio` and interpolated in
    :math:`(\ln k, z, f_\nu)`, so it stays differentiable with respect to
    :math:`\Sigma m_\nu`.  Exactly 1 at :math:`\Sigma m_\nu = 0`, so the
    massless fiducial is bit-identical to an uncorrected spectrum.

    This replaces the first-order closed form
    :math:`1 - 8f_\nu\cdot\tfrac12[1+\tanh(\ln(k/k_{\rm nr})/1.5)]` that
    stood here.  Its :math:`-8f_\nu` asymptote was sound, but centring the
    transition on the non-relativistic scale :math:`k_{\rm nr}` saturated it
    about 1.5 decades of :math:`k` too early: against CLASS it over-suppressed
    by ~2 percentage points at 0.06 eV and ~11 at 0.3 eV, right across the
    scales the mass function and the 2-halo term use.
    """
    from hod_mod.core.nu_ratio import suppression_m
    return suppression_m(k, theta, z)


def _nu_cold_ratio(k: jnp.ndarray, theta: dict, z=0.0) -> jnp.ndarray:
    r"""Cold-spectrum ratio :math:`P_{\rm cb}(k,z)/P_{\rm m}(k,z;0)`.

    What :math:`\sigma(M)` and the mass function need: neutrinos count in
    :math:`\Omega_{\rm m}` but do not cluster into halos.
    """
    from hod_mod.core.nu_ratio import suppression_cb
    return suppression_cb(k, theta, z)


class EisensteinHu98PkLinear:
    """σ8-parameterised, JAX-differentiable EH98 linear power spectrum.

    Parameters
    ----------
    R8 : float
        Top-hat radius for the σ8 normalisation [Mpc/h] (8 by convention).
    camb_ratio : dict or None
        Linearized CAMB/EH98 shape-ratio table
        (:func:`hod_mod.forecast.pk_camb_ratio.load`) — multiplies the shape
        everywhere (HMF path, σ8 anchor, 2-halo) so the spectrum and its first
        derivatives become CAMB-accurate near the fiducial.
    """

    #: EH98 is an analytic *shape*: its ``pk_linear`` gets every bit of its
    #: redshift dependence from the growth factor it multiplies in.  Asking it
    #: for a growth ratio would be circular, so the dispatcher must not.
    has_native_z = False

    def __init__(self, R8: float = 8.0, camb_ratio: dict = None):
        self._R8 = float(R8)
        self._camb_ratio = camb_ratio

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
        if self._camb_ratio is not None:
            from hod_mod.forecast.pk_camb_ratio import apply_ratio
            pk = apply_ratio(pk, k, theta, self._camb_ratio)
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
        # mode="ode" is required, not preferred: EH98 has no redshift
        # dependence of its own, so this line *is* the growth model.  See the
        # matching note in pk_cosmopower.pk_linear.
        growth = growth_factor(float(z), theta, mode="ode")
        return pk_shape * norm * growth ** 2

    # -- convenience: a plain callable for make_hmf(pk_func=...) --------
    def as_hmf_pk_func(self):
        """Return a ``(k, z, theta) -> P_shape`` callable for the HMF."""
        def _pk_func(k, z, theta):
            return self.pk_shape(k, theta)
        return _pk_func
