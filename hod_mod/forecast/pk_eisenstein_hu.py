r"""Differentiable Eisenstein & Hu (1998) linear power spectrum.

This wraps the repository's pure-JAX EH98 transfer function into a
``LinearPowerSpectrum``-compatible object.  Because EH98 is analytic, the whole
thing is JAX-traceable and differentiable w.r.t. the cosmological parameters --
which is exactly what the Fisher forecast needs (CAMB is not JAX-traceable).

The amplitude is ``ln10^{10}A_s``, the package's single amplitude parameter, via
:func:`~hod_mod.core.power_spectrum.eisenstein_hu_pk_phys`.  It used to be
``sigma8``, and this module was the reason the package carried two amplitude
conventions at once: ``pk_linear`` here required a ``sigma8`` and had no A_s
branch at all, while CAMB, CLASS and the CSST emulator required ``ln10^{10}A_s``
and silently *ignored* a ``sigma8``.  Which physics ran was decided by which
key a YAML file happened to contain.

Three entry points are exposed on :class:`EisensteinHu98PkLinear`:

* ``pk_linear(k, z, theta)`` -- physical :math:`P(k, z)` [(Mpc/h)³] for the
  2-halo term, ``P(k,0)`` times the growth ODE squared.
* ``as_hmf_pk_func()`` -- a ``(k, z, theta)`` callable returning the physical
  ``P(k, 0)`` for :class:`~hod_mod.core.halo_mass_function.HaloMassFunction`,
  which applies its own growth.  This used to return a *shape*, with the HMF
  supplying the amplitude from ``theta['sigma8']``; that rescale is gone, so the
  amplitude arrives here instead.  The two paths therefore still share one
  amplitude by construction -- it is just no longer a second parameter.
* ``pk_shape(k, theta)`` -- the amplitude-free shape, kept only for the two
  callers that normalise it away regardless (the CAMB/EH98 ratio table and the
  accuracy budget).  Not on the σ(M) path any more.

σ8 for a given ``theta`` is a *derived* quantity:
:func:`hod_mod.core.amplitude.sigma8_from_theta`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hod_mod.core.amplitude import K_INT, require_as_amplitude, sigma2_tophat
from hod_mod.core.power_spectrum import eisenstein_hu_pk, eisenstein_hu_pk_phys
from hod_mod.core.halo_mass_function import growth_factor


# The σ(R=8) grid and quadrature now come from hod_mod.core.amplitude, the one
# place they are defined.  These module-level aliases are load-bearing: they are
# imported by name from pk_cosmopower.py, tests/test_forecast_pk_eh98.py and
# hod_mod/forecast/tests/test_pk_cosmopower.py.
_K_INT = K_INT
_sigma2_tophat = sigma2_tophat


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
    """A_s-parameterised, JAX-differentiable EH98 linear power spectrum.

    Parameters
    ----------
    camb_ratio : dict or None
        Linearized CAMB/EH98 shape-ratio table
        (:func:`hod_mod.forecast.pk_camb_ratio.load`) — multiplies the spectrum
        on every path (HMF and 2-halo alike) so it and its first derivatives
        become CAMB-accurate near the fiducial.
    """

    #: EH98 is an analytic *shape*: its ``pk_linear`` gets every bit of its
    #: redshift dependence from the growth factor it multiplies in.  Asking it
    #: for a growth ratio would be circular, so the dispatcher must not.
    has_native_z = False

    def __init__(self, camb_ratio: dict = None):
        self._camb_ratio = camb_ratio

    # -- shape spectrum, amplitude-free ---------------------------------
    def pk_shape(self, k: jnp.ndarray, theta: dict) -> jnp.ndarray:
        """EH98 shape spectrum (normalised to P(0.05 h/Mpc)=1).

        Amplitude-free by construction.  **Off the sigma(M) path** -- that is
        :meth:`as_hmf_pk_func`, which carries the real amplitude.  What is left
        for a shape is the two callers that normalise it away anyway: the
        CAMB/EH98 ratio table and the accuracy budget.

        When ``theta`` carries ``sum_mnu`` (a static dict-membership branch),
        the massive-neutrino suppression multiplies the shape.
        """
        k = jnp.asarray(k)
        pk = eisenstein_hu_pk(k, theta)
        if "sum_mnu" in theta:
            pk = pk * _nu_suppression(k, theta)
        if self._camb_ratio is not None:
            from hod_mod.forecast.pk_camb_ratio import apply_ratio
            pk = apply_ratio(pk, k, theta, self._camb_ratio)
        return pk

    # -- physical spectrum ----------------------------------------------
    def _pk_physical(self, k: jnp.ndarray, theta: dict) -> jnp.ndarray:
        r"""EH98 :math:`P(k, z{=}0)` in :math:`(\mathrm{Mpc}/h)^3`.

        The amplitude comes from ``ln10^{10}A_s`` via
        :func:`~hod_mod.core.power_spectrum.eisenstein_hu_pk_phys`, which shares
        its transfer function with :func:`eisenstein_hu_pk` exactly and adds the
        primordial normalisation and :math:`D^2(z{=}0)`.

        This used to be ``pk_shape * sigma8**2 / sigma8_shape**2``.  Two things
        were wrong with that beyond the retired key: ``eisenstein_hu_pk``
        normalises itself by ``P(0.05)`` interpolated **on the caller's k grid**,
        so the shape's normalisation depended on which grid it was handed, while
        the denominator was always evaluated on ``_K_INT`` -- the amplitude was
        off by the ratio of the two pivots whenever the grids differed.  Going
        through ``_phys`` is grid-independent by construction.
        """
        k = jnp.asarray(k)
        pk = eisenstein_hu_pk_phys(k, theta)
        if "sum_mnu" in theta:
            pk = pk * _nu_suppression(k, theta)
        if self._camb_ratio is not None:
            from hod_mod.forecast.pk_camb_ratio import apply_ratio
            pk = apply_ratio(pk, k, theta, self._camb_ratio)
        return pk

    def pk_linear(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        r"""Physical :math:`P(k, z)` [(Mpc/h)³], amplitude from ``ln10^{10}A_s``.

        .. math::

            P(k, z) = P(k, 0)\,\left[\frac{D(z)}{D(0)}\right]^2
        """
        require_as_amplitude(theta, "EisensteinHu98PkLinear.pk_linear")
        # mode="ode" is required, not preferred: EH98 has no redshift
        # dependence of its own, so this line *is* the growth model.  See the
        # matching note in pk_cosmopower.pk_linear.
        growth = growth_factor(float(z), theta, mode="ode")
        return self._pk_physical(k, theta) * growth ** 2

    # -- convenience: a plain callable for make_hmf(pk_func=...) --------
    def as_hmf_pk_func(self):
        """Return a ``(k, z, theta) -> P(k, 0)`` callable for the HMF.

        The z=0 **physical** spectrum, not a shape.  ``HaloMassFunction.sigma``
        used to be handed a shape and rescale it to ``theta['sigma8']``; with
        that rescale gone the amplitude has to arrive here instead.  ``z`` is
        ignored because the HMF applies its own growth -- it always calls with
        ``0.0`` -- so this deliberately does not pay for the growth ODE.
        """
        def _pk_func(k, z, theta):
            require_as_amplitude(theta, "EisensteinHu98PkLinear.as_hmf_pk_func")
            return self._pk_physical(k, theta)
        return _pk_func
