r"""The single amplitude parameter, and the quantities derived from it.

``ln10^{10}A_s`` is the **only** amplitude parameter in this package.  It is
what the Boltzmann solvers and the emulator actually take, it is what the
primordial spectrum is actually parameterised by, and it is the one number a
fit or a forecast varies.  :math:`\sigma_8` and :math:`S_8` are *outputs*: you
compute them from a spectrum, a posteriori, with the functions here.

Why one parameter and not two
-----------------------------
The package used to carry both conventions at once.  ``sigma8`` in ``theta``
selected a different branch in three different files, and the branch it
selected was not the same physics: the A_s branch asks the backend for
:math:`P(k, z)` and gets the backend's own growth, while the sigma8 branch
builds :math:`P_{\rm shape}(k)\times\sigma_8^2/\sigma_{8,\rm shape}^2 \times
D^2(z)` and gets the growth ODE.  Which one ran was decided by whether a YAML
file happened to contain a key.

There is also a plain double-counting hazard: a ``theta`` carrying *both*
``ln10^{10}A_s`` and ``sigma8`` is over-determined, and every consumer resolved
the contradiction differently.  In the free-cosmology fits that was not
academic -- the sampled ``S8`` updated only ``ln10^{10}A_s`` while the backend
and the mass function both read a frozen ``sigma8``, so the sampled parameter
moved nothing at all.  :func:`require_as_amplitude` makes that state
unrepresentable rather than merely discouraged.

Derived, not sampled
--------------------
:func:`sigma8_from_theta` evaluates the backend **at z = 0 directly**.  It does
not call the growth dispatcher.  That is deliberate and load-bearing: a
:math:`\sigma_8` written as ``sigma8_from_pk(pk_shape) * growth_factor(0, ...)``
carries a factor that is structurally :math:`D(0)/D(0) = 1`, so its *value* is
right and its *derivative* is a lie -- ``jax.grad`` propagates through both legs
of the ratio and picks up whatever the growth model's numerator and denominator
do not exactly cancel.  Asking the backend for ``pk_linear(k, 0.0, theta)`` has
no such term.

Measured at Planck-2018 (:math:`\Omega_m = 0.31`) on the CosmoPower backend
(2026-08-24, CPU x86-64, float64), against the analytic derivative
:math:`\partial\sigma_8/\partial\ln10^{10}A_s = \sigma_8/2`:

:math:`\sigma_8`:
    0.8148909951
:math:`\partial\sigma_8/\partial\ln10^{10}A_s`:
    0.4074519672 (autodiff) against 0.4074454976 (analytic)
:math:`\partial\sigma_8/\partial\Omega_m`:
    1.964975125
:math:`\partial S_8/\partial\Omega_m`:
    3.333522673

That last row is the one to keep.  :math:`S_8` depends on :math:`\Omega_m`
twice -- once through the explicit :math:`(\Omega_m/0.3)^{1/2}` and once through
:math:`\sigma_8(\Omega_m)` itself, because at fixed :math:`A_s` a larger
:math:`\Omega_m` grows more structure.  A caller who wants
:math:`\partial S_8/\partial\Omega_m` as a *total* derivative must let autodiff
find both terms; see :func:`s8_from_theta` for when that is and is not what you
want.

Dependencies
------------
``jax.numpy`` and nothing else.  ``hod_mod.core`` must never import
``hod_mod.forecast`` (the reverse edge exists), and this module is imported by
both, so it is kept free of every package-internal import -- including
``hod_mod.core`` ones -- so that no future edit can close a cycle through it.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "AMPLITUDE_KEY",
    "RETIRED_AMPLITUDE_KEYS",
    "K_INT",
    "R8",
    "S8_PIVOT_OMEGA_M",
    "sigma2_tophat",
    "sigma8_from_pk",
    "sigma8_from_theta",
    "s8_from_theta",
    "require_as_amplitude",
]

#: The one amplitude parameter.  Spelled exactly as CAMB, CLASS and
#: CosmoPower-JAX spell it, braces included, so no translation layer exists to
#: get wrong.
AMPLITUDE_KEY = "ln10^{10}A_s"

#: Keys that are no longer parameters.  Present in a ``theta`` they mean the
#: caller is on the old convention, and :func:`require_as_amplitude` refuses.
RETIRED_AMPLITUDE_KEYS = ("sigma8", "S8")

#: Wavenumber grid for the :math:`\sigma(R{=}8)` integral [h/Mpc].  Identical to
#: ``HaloMassFunction``'s default ``logspace(-4, 3, 512)``, so a sigma8 read off
#: this grid is the same number sigma(M) is built from.  Verified elementwise
#: equal to ``jnp.logspace(-4, 3, 512)``, ``core.growth._K_INT`` and
#: ``forecast.pk_eisenstein_hu._K_INT``.  It is *not*
#: ``CsstHaloMassFunction._k_int``, which is ``logspace(-4, 2, 512)`` because the
#: CSST emulator stops at k = 100 h/Mpc -- pass that grid explicitly there.
K_INT = jnp.logspace(-4.0, 3.0, 512)

#: Top-hat radius that defines sigma8 [Mpc/h].
R8 = 8.0

#: Pivot in :math:`S_8 = \sigma_8\sqrt{\Omega_m/0.3}`.
S8_PIVOT_OMEGA_M = 0.3


def sigma2_tophat(pk, k, R=R8):
    r"""σ²(R) = (1/2π²) ∫ P(k) W²(kR) k² dk, real-space top-hat W.

    The canonical copy.  Five independent transcriptions of these three lines
    existed before this module; they agreed, but nothing made them agree, and
    the sigma8 a backend *normalises to* and the sigma8 the mass function *reads
    back* have to be the same integral on the same grid or the two legs of the
    halo model drift apart silently.

    Parameters
    ----------
    pk : array_like, shape (nk,)
        P(k) on ``k``, any units; σ² comes out in the units of ``pk·k³``.
    k : array_like, shape (nk,)
        Wavenumbers [h/Mpc], increasing.  ``jnp.trapezoid`` integrates on this
        grid as given, so the grid is part of the definition -- pass
        :data:`K_INT` unless you have a reason not to.
    R : float or traced scalar
        Top-hat radius [Mpc/h].  Traced ``R`` is supported and is what the
        mass-space callers need: ``jax.vmap(lambda r: sigma2_tophat(pk, k, r))``
        over the Lagrangian radii of a mass grid.

    Notes
    -----
    No small-``x`` guard, matching every copy this replaces.  ``W`` is written as
    ``3(sin x - x cos x)/x³``, which is 0/0 at ``x = 0``.  Below ``x ~ 1e-3`` the
    subtraction cancels catastrophically: in float32 it returns exactly ``0.0``
    at the ``x = k_min·R ~ 3e-6`` the lightest halo on the default grid reaches,
    and ``1.023`` at ``x = 8e-4`` (float64 is correct to 1.7e-5 at 3e-6).  That
    is pre-existing in all five copies and harmless -- the integrand carries
    ``P(k) k²``, which at ``k = 1e-4 h/Mpc`` is ~14 orders below the peak, so a
    zeroed window there costs nothing.  A guard is *not* added here because it
    would change the traced graph of every sigma(M) in the package; if one is
    ever wanted it belongs in a separate commit with its own sigma(M) regression.
    """
    x = k * R
    w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x ** 3
    return jnp.trapezoid(pk * w ** 2 * k ** 2, k) / (2.0 * jnp.pi ** 2)


def sigma8_from_pk(pk, k):
    r"""sigma8 from a physically normalised z=0 spectrum.

    ``pk`` must be :math:`P(k, z{=}0)` in :math:`(\mathrm{Mpc}/h)^3` -- a real
    amplitude, not a shape.  Feeding a shape spectrum here returns a number with
    units of the shape's arbitrary normalisation and no meaning at all; that is
    exactly the mistake the shape/physical split used to invite, and it is why
    :func:`sigma8_from_theta` calls ``pk_linear`` and never ``pk_shape``.
    """
    return jnp.sqrt(sigma2_tophat(pk, k, R8))


def _pk_linear_at_z0(theta: dict, pk_lin, k):
    """``P(k, z=0)`` from either a backend object or a bare ``(k, z, theta)``.

    Backends are passed as objects (``pk_lin.pk_linear``), but
    ``HaloMassFunction`` stores a plain closure (``self._pk``) and the forecast
    hands around ``as_hmf_pk_func()`` results, so both call shapes reach here.
    """
    fn = getattr(pk_lin, "pk_linear", None)
    if fn is not None and isinstance(pk_lin, type):
        # getattr on the *class* yields an unbound function, which would fail
        # later with a baffling "missing 1 required positional argument".
        raise TypeError(
            "sigma8_from_theta wants a P(k) backend *instance*, not the class "
            f"{pk_lin.__name__}; call {pk_lin.__name__}() first.")
    if fn is None:
        if not callable(pk_lin):
            raise TypeError(
                "sigma8_from_theta needs a P(k) backend with a .pk_linear(k, z, "
                "theta) method, or a plain callable of that signature; got "
                f"{type(pk_lin).__name__}, which is neither.")
        fn = pk_lin
    return fn(k, 0.0, theta)


def sigma8_from_theta(theta: dict, pk_lin, k=None):
    r"""sigma8 for ``theta``, read off ``pk_lin``'s own z=0 spectrum.

    A **derived** quantity.  ``theta`` must carry :data:`AMPLITUDE_KEY` and must
    not carry a retired key (:func:`require_as_amplitude` enforces both), so a
    caller that still thinks sigma8 is an input gets an error naming the fix
    rather than a plausible number computed from a stale convention.

    Differentiable in every entry of ``theta`` the backend is differentiable in.
    Evaluated at ``z = 0.0`` literally: see the module docstring for why a growth
    call belongs nowhere near this function.

    Parameters
    ----------
    theta : dict
        Cosmology in the A_s convention.
    pk_lin : object or callable
        Backend with ``pk_linear(k, z, theta)``, or that callable directly.
    k : array_like, optional
        Wavenumber grid [h/Mpc].  Defaults to :data:`K_INT`.  Override only for a
        backend whose support does not cover it -- the CSST emulator stops at
        ``k = 100 h/Mpc``, and ``CsstHaloMassFunction`` uses
        ``logspace(-4, 2, 512)`` for exactly that reason.
    """
    k = K_INT if k is None else jnp.asarray(k)
    require_as_amplitude(theta, "sigma8_from_theta")
    return sigma8_from_pk(_pk_linear_at_z0(theta, pk_lin, k), k)


def s8_from_theta(theta: dict, pk_lin, k=None):
    r""":math:`S_8 = \sigma_8\sqrt{\Omega_m/0.3}`, derived the same way.

    Both of :math:`S_8`'s dependencies on :math:`\Omega_m` are live here -- the
    explicit root and the one inside :math:`\sigma_8` -- which is what a
    posterior-processing step wants, and what a Fisher propagation wants once
    :math:`\sigma_8` is no longer a coordinate.

    Note this is *not* the Jacobian row the Fisher scripts used before the
    retirement.  ``run_tier2_forecast._s8_sigma`` and
    ``run_stage4_forecast._s8_sigma`` propagated a covariance in a basis where
    ``sigma8`` was an independent coordinate, so their
    :math:`\partial S_8/\partial\Omega_m` was correctly the *partial* at fixed
    :math:`\sigma_8` (1.336 at Planck-2018) and the correlation lived in ``cov``.
    They were right in that basis; they stopped working only because
    ``_IDX["sigma8"]`` ceased to exist.  Rewritten against ``ln10^{10}A_s`` they
    need this function's *total* derivative (3.334).
    """
    s8 = sigma8_from_theta(theta, pk_lin, k=k)
    return s8 * jnp.sqrt(theta["Omega_m"] / S8_PIVOT_OMEGA_M)


def require_as_amplitude(theta: dict, where: str):
    r"""Assert ``theta`` is in the A_s convention; return :math:`\ln10^{10}A_s`.

    The tripwire for the sigma8 retirement.  Placed at every entry point that
    turns a ``theta`` into an amplitude, so a stale caller fails loudly at the
    boundary instead of quietly taking a branch that no longer exists.

    Two failures, two different **exception types**, because they need two
    different fixes and a caller may reasonably want to catch one and not the
    other:

    * :exc:`ValueError` -- a retired key is present.  The caller is on the old
      convention and must be migrated (config, script, or notebook).
    * :exc:`KeyError` -- the amplitude key is missing.  The caller built a
      ``theta`` by hand and forgot it.

    ``ValueError`` for the first is also the readable choice: ``KeyError``
    reprs its argument, which turns a multi-line explanation into one escaped
    string.

    Safe under ``jit``: dictionary membership is decided on the key set, which is
    static, so this raises at trace time and never becomes a traced predicate.

    Parameters
    ----------
    theta : dict
    where : str
        The call site, named the way a reader would grep for it, e.g.
        ``"HaloMassFunction.sigma"`` or ``"CosmoPowerJaxPkLinear.pk_linear"``.
    """
    stray = [key for key in RETIRED_AMPLITUDE_KEYS if key in theta]
    if stray:
        raise ValueError(
            f"{where}: {', '.join(repr(s) for s in stray)} is no longer a "
            f"parameter -- {AMPLITUDE_KEY!r} is the only amplitude in this "
            "package, and sigma8/S8 are derived from the spectrum a posteriori "
            "by hod_mod.core.amplitude.sigma8_from_theta / s8_from_theta.  A "
            "theta carrying both is over-determined, and which one won used to "
            "depend on which file you were in.  Drop the retired key; if you "
            "need the old amplitude, convert it once, up front, by solving "
            f"sigma8_from_theta for {AMPLITUDE_KEY!r}.")
    if AMPLITUDE_KEY not in theta:
        raise KeyError(
            f"{where}: needs the amplitude {AMPLITUDE_KEY!r}; theta has "
            f"{sorted(theta)}.  Planck-2018 central value is 3.044.")
    return theta[AMPLITUDE_KEY]
