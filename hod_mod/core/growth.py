r"""Linear growth factor :math:`D(z)/D(0)` — one dispatcher, three routes.

The halo model needs :math:`D(z)` in two very different places, and until this
module existed it got them from two different formulas:

* :meth:`~hod_mod.core.halo_mass_function.HaloMassFunction.sigma` evaluates the
  P(k) backend **at z = 0** and multiplies by an analytic growth factor;
* the 2-halo term asks the *same backend* for ``pk_linear(k, z, theta)``, which
  for CAMB, CLASS and the CosmoPower A_s branch carries the backend's own,
  exact growth.

So one leg of the model grew with a fitting formula and the other with a
Boltzmann solver — and the pipeline paid for both spectra.  This module makes
the choice explicit and puts the accurate route first.

Modes
-----
``"pk_ratio"`` (default)
    :math:`D(z)` is whatever the active P(k) backend says, read off as a
    :math:`\sigma(R{=}8)`-weighted ratio of its own spectra (below).  The HMF
    leg and the 2-halo leg then agree *by construction*, with no residual.
``"ode"``
    The flat-CPL linear growth ODE
    (:func:`~hod_mod.core.halo_mass_function._growth_factor_cpl_jax`).  Used
    for :math:`\Lambda`\ CDM too — there is no key-sniffing branch here.
    Reproduces CAMB and CosmoPower to :math:`\le 5\times10^{-5}` and is the
    declared fallback for backends with no redshift dependence of their own.
``"carroll"``
    The Carroll+1992 fitting formula.  Retained to reproduce results produced
    before this module and to drive the accuracy study; **not** accurate enough
    to be a default (see *Accuracy* below).

Selection is ``mode`` argument → ``$HOD_MOD_GROWTH`` → ``"pk_ratio"``, mirroring
the ``backend`` / ``$HOD_MOD_PK_BACKEND`` pattern of
:func:`~hod_mod.core.power_spectrum.default_pk_linear`.

Why a σ(R=8) weight and not a single k
--------------------------------------
With massive neutrinos the growth is genuinely scale-dependent, so
:math:`\sqrt{P(k,z)/P(k,0)}` is not one number: across
:math:`k \in [10^{-3}, 10]\,h\,{\rm Mpc}^{-1}` it spreads by
:math:`2.5\times10^{-5}` at :math:`z=0.3` and :math:`6.5\times10^{-4}` at
:math:`z=3`.  Collapsing it with the same top-hat :math:`\sigma(R{=}8)` integral
that :math:`\sigma(M)` and the σ8 normalisation already use projects that
k-dependence out *the way the consumer consumes it*, rather than by picking a
privileged wavenumber.

Cold versus total
-----------------
Neutrinos count in :math:`\Omega_m` but do not cluster into halos, so the growth
that drives :math:`\sigma(M)`, :math:`dn/dM` and :math:`b(M)` is the **cold**
(CDM + baryon) growth :math:`D_{\rm cb}(z)`, while the 2-halo term, the NLA
intrinsic-alignment amplitude and :math:`c(M,z)` live on the total-matter field
and want :math:`D_{\rm m}(z)`.  Select with ``variant="cold"`` / ``"total"``.
At :math:`\Sigma m_\nu = 0` the two are identical and the cold path is exact.

Accuracy
--------
Measured against the CAMB and CosmoPower P(k) ratios at Planck-2018,
:math:`z \in [0, 3]` (2026-08-24, CPU x86-64):

===========================  ===================  ==================
route                        :math:`\Omega_m`     max :math:`|D/D_{\rm ref}-1|`
===========================  ===================  ==================
Carroll+1992                 0.31                 1.8e-3
Carroll+1992                 0.24                 3.9e-3
Carroll+1992                 0.40                 0.6e-4
CPL ODE (``n_steps=160``)    0.24 – 0.40          2.1e-4
===========================  ===================  ==================

The value error is not the whole story: :math:`d\ln n/d\ln\sigma` runs 10 → 24
across the mass grid, so Carroll's :math:`1.7\times10^{-3}` becomes **1–4 % in
dn/dM** and **0.2–0.7 % in** :math:`b^2`.  Worse for forecasting, Carroll's
:math:`\partial\ln D/\partial\Omega_m` is **4–7 % wrong** and its
:math:`\partial\ln D/\partial w_0` is **identically zero**.

Timing
------
:math:`D(z)` at 8 redshifts, jitted, CPU x86-64 (2026-08-24):

=========================================  ===========
route                                      per call
=========================================  ===========
Carroll (vmap)                             0.012 ms
CPL ODE, ``n_steps=160`` (one solve)       0.063 ms
CPL ODE, ``n_steps=320``                   0.120 ms
CosmoPower P-ratio (9 network passes)      1.963 ms
=========================================  ===========

Reverse-mode ``grad`` through the ODE ``lax.scan`` costs 1.5 ms against 0.012 ms
for Carroll — the one place the ODE is genuinely expensive.  ``n_steps=320``
buys nothing over 160: both already sit at the emulator's own noise floor.
"""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp

#: Valid ``mode`` values, in decreasing order of fidelity.
GROWTH_MODES = ("pk_ratio", "ode", "carroll")

#: Mode used when neither the ``mode`` argument nor ``$HOD_MOD_GROWTH`` is set.
DEFAULT_MODE = "pk_ratio"

#: Wavenumber grid for the σ(R=8) projection.  Matches ``HaloMassFunction``'s
#: default ``logspace(-4, 3, 512)`` and ``pk_eisenstein_hu._K_INT``, so the
#: weight applied here is the same one the σ8 normalisation applies.
_K_INT = jnp.logspace(-4.0, 3.0, 512)

#: Top-hat radius of the projection [Mpc/h].
_R8 = 8.0

#: Amplitude used when a backend needs ``ln10^{10}A_s`` and ``theta`` carries
#: only ``sigma8``.  The overall amplitude cancels in the ratio, so the value is
#: immaterial; the Planck-2018 central value keeps emulators near their training
#: centre.
_REF_LN10_10_AS = 3.044

#: Fallback classification for backends that do not declare ``has_native_z``
#: themselves.  Every backend shipped with the package now carries the attribute
#: (``tests/test_growth.py`` enforces that), so this table serves only duck-typed
#: third-party objects — it is a courtesy, not the source of truth.  Kept keyed on
#: class name so this module needs no import of the backends.
#:
#: The classification is *declared*, never probed by calling the backend: one
#: that silently ignored ``z`` would return D(z) = 1 and look perfectly exact.
_NATIVE_Z = {
    "LinearPowerSpectrum":     True,    # CAMB
    "ClassPkLinear":           True,    # CLASS
    "CosmoPowerJaxPkLinear":   True,    # emulator, trained over z in [0, 5]
    "CsstLinearPowerSpectrum": True,    # CSST CEmulator
    "EisensteinHu98PkLinear":  False,   # analytic shape, no z dependence
}

_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    """Emit ``message`` the first time ``key`` is seen in this process."""
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def resolve_mode(mode: str = None) -> str:
    """Return the growth mode: ``mode`` → ``$HOD_MOD_GROWTH`` → :data:`DEFAULT_MODE`."""
    return _resolve_mode_explicit(mode)[0]


def _resolve_mode_explicit(mode: str = None):
    """``(mode, was_requested)`` — whether a human chose it or it is the default.

    The distinction matters only for diagnostics.  ``pk_ratio`` is meaningless
    without a backend, and most callers of :func:`growth_factor` are analytic
    sites that legitimately have none (the σ8-convention ``pk_linear`` methods,
    ``c_bhattacharya13``, the NLA amplitude).  Falling back for them is normal
    operation, not a problem to report — warning at each would be a storm that
    trains the reader to ignore it.  But a caller who *asked* for ``pk_ratio``,
    by argument or by ``$HOD_MOD_GROWTH``, and cannot have it, is owed a warning.
    """
    if mode is not None:
        requested = True
    else:
        mode = os.environ.get("HOD_MOD_GROWTH")
        requested = mode is not None
        if mode is None:
            mode = DEFAULT_MODE
    mode = mode.lower()
    if mode not in GROWTH_MODES:
        raise ValueError(f"growth mode must be one of {GROWTH_MODES}, got {mode!r}")
    return mode, requested


def backend_has_native_z(pk) -> bool:
    """Whether ``pk`` can supply its own redshift dependence.

    Looks first for an explicit ``has_native_z`` attribute on the object (the
    declared contract), then falls back to :data:`_NATIVE_Z` by class name.  An
    unknown backend is treated as **not** z-aware and warns once: refusing to
    guess is what keeps a backend that quietly ignores ``z`` from being read as
    a perfect growth reference.
    """
    declared = getattr(pk, "has_native_z", None)
    if declared is not None:
        return bool(declared)
    name = type(pk).__name__
    if name in _NATIVE_Z:
        return _NATIVE_Z[name]
    _warn_once(
        f"unknown-backend:{name}",
        f"growth: P(k) backend {name!r} is not in the known-backend table, so "
        "its redshift dependence cannot be assumed; falling back to the growth "
        "ODE.  Declare `has_native_z` on the class to silence this.",
    )
    return False


def _sigma2_tophat(pk: jnp.ndarray, k: jnp.ndarray, R: float = _R8) -> jnp.ndarray:
    r"""σ²(R) = (1/2π²) ∫ P(k) W²(kR) k² dk with a real-space top-hat W.

    Duplicated from :func:`hod_mod.forecast.pk_eisenstein_hu._sigma2_tophat` on
    purpose: ``hod_mod.core`` must not import ``hod_mod.forecast`` (the reverse
    edge already exists and would close a cycle).  The two are identical and
    ``tests/test_growth.py`` asserts they stay so.
    """
    x = k * R
    w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x ** 3
    return jnp.trapezoid(pk * w ** 2 * k ** 2, k) / (2.0 * jnp.pi ** 2)


def _as_amplitude_theta(theta: dict) -> dict:
    """Strip ``sigma8`` so a backend takes its native (A_s) redshift path.

    Every σ8-convention ``pk_linear`` implements ``P(k,z) = shape × norm ×
    D(z)²`` by *calling this module* — asking it for a ratio would recurse.  The
    A_s branch instead returns whatever the network or Boltzmann solver natively
    produces at ``z``, which is exactly the quantity a growth ratio wants.  The
    overall amplitude cancels, so substituting a reference ``A_s`` is free.

    ``Omega_cdm`` is deliberately **not** synthesised when it is absent.  There
    is no single right value: the Boltzmann backends want the closed-budget
    :func:`~hod_mod.core.power_spectrum.omega_cdm`
    (``Ω_m − Ω_b − Ω_ν``), while the massless-trained emulator and EH98 want
    :func:`~hod_mod.core.power_spectrum.omega_cdm_cold_equivalent`
    (``Ω_m − Ω_b``, the neutrinos taken out again by the ``nu_ratio`` table).
    Every backend already derives the one it needs, and an injected key would
    *override* that choice — silently running CAMB and CLASS at ``Ω_m + Ω_ν``
    and poisoning the CLASS solver cache key with it.  A caller who sets
    ``Omega_cdm`` explicitly still has it honoured, unchanged.
    """
    out = {kk: vv for kk, vv in theta.items() if kk != "sigma8"}
    out.setdefault("ln10^{10}A_s", _REF_LN10_10_AS)
    return out


def _pk_at(pk, k, z, theta_as: dict, variant: str):
    """Backend spectrum at ``z`` on the cold or the total-matter field."""
    if variant == "total":
        return pk.pk_linear(k, z, theta_as)

    # 1. A backend that solves the neutrino sector exactly exposes P_cb itself.
    if hasattr(pk, "pk_linear_cb"):
        return pk.pk_linear_cb(k, z, theta_as)

    # 2. Otherwise convert P_m -> P_cb with the distilled CLASS ratio table.
    if "sum_mnu" in theta_as:
        try:
            from hod_mod.core import nu_ratio
        except ImportError:
            _warn_once(
                "no-cold-spectrum",
                "growth: variant='cold' requested with sum_mnu > 0 but neither a "
                "pk_linear_cb method nor hod_mod.core.nu_ratio is available; "
                "returning the total-matter growth instead.",
            )
            return pk.pk_linear(k, z, theta_as)
        ratio = (nu_ratio.suppression_cb(k, theta_as, z)
                 / nu_ratio.suppression_m(k, theta_as, z))
        return pk.pk_linear(k, z, theta_as) * ratio

    # 3. Massless: cold and total are the same field, exactly.
    return pk.pk_linear(k, z, theta_as)


def growth_pk_ratio(z, theta: dict, pk, variant: str = "total") -> jnp.ndarray:
    r"""D(z)/D(0) read off the backend's own spectra.

    .. math::

        \frac{D(z)}{D(0)} = \sqrt{\frac{\sigma^2_{R_8}[P(k, z)]}
                                        {\sigma^2_{R_8}[P(k, 0)]}}

    The σ(R=8) top-hat is the same weight the σ8 normalisation uses, so the
    residual scale-dependence of the growth is projected out consistently with
    how :math:`\sigma(M)` consumes it (see the module docstring).

    Costs **two** backend evaluations.  For the emulator that is ~2 ms; for CAMB
    it is two multi-second solves, which a single ``redshifts=[z, 0]`` run would
    collapse into one — that optimisation belongs to the backend, not here.
    """
    if not backend_has_native_z(pk):
        raise ValueError(
            f"growth_pk_ratio: {type(pk).__name__} declares has_native_z=False, "
            "so the ratio of its spectra is not a growth factor -- its redshift "
            "dependence comes from the growth factor it multiplies in, and "
            "reading D(z) back off it would be circular.  Use "
            "growth_factor(..., mode='ode') instead.")
    theta_as = _as_amplitude_theta(theta)
    p_z = _pk_at(pk, _K_INT, z, theta_as, variant)
    p_0 = _pk_at(pk, _K_INT, 0.0, theta_as, variant)
    return jnp.sqrt(_sigma2_tophat(p_z, _K_INT) / _sigma2_tophat(p_0, _K_INT))


def growth_factor(z, theta: dict, pk=None, mode: str = None,
                  variant: str = "total") -> jnp.ndarray:
    r"""Linear growth factor :math:`D(z)/D(0)`.

    Parameters
    ----------
    z : float or array_like
        Redshift.  ``pk_ratio`` needs a concrete value (backends are called at
        it); ``ode`` accepts an array and interpolates on one internal solve.
    theta : dict
        Cosmology.  ``Omega_m`` is required; ``w0``/``wa`` are honoured by the
        ODE, ``sum_mnu`` by ``variant="cold"``.
    pk : object, optional
        The active P(k) backend.  Required for ``pk_ratio``; ignored otherwise.
    mode : str, optional
        One of :data:`GROWTH_MODES`.  Defaults through :func:`resolve_mode`.
    variant : {"total", "cold"}
        Which field to grow.  ``"cold"`` (CDM + baryon) is what the halo sector
        wants; ``"total"`` is what the 2-halo term, the NLA amplitude and
        :math:`c(M,z)` want.  Identical at :math:`\Sigma m_\nu = 0`.

    Notes
    -----
    ``pk_ratio`` falls back to ``ode`` — warning once, naming the backend — when
    ``pk`` is ``None`` or declares no native redshift dependence (EH98).  The
    fallback is declared, never inferred from a failed call.

    ``ode`` and ``carroll`` have no neutrino sector, so ``variant="cold"`` is
    not meaningful for them and warns once when :math:`\Sigma m_\nu > 0`.
    """
    if not hasattr(theta, "get"):
        raise TypeError(
            f"growth_factor expects a cosmology dict, got {type(theta).__name__} "
            f"({theta!r}).  The signature moved from (z, omega_m) to (z, theta) "
            "so that the growth can see w0/wa and sum_mnu, which a bare Omega_m "
            "cannot express; pass {'Omega_m': ...} at minimum.")
    if variant not in ("total", "cold"):
        raise ValueError(f"variant must be 'total' or 'cold', got {variant!r}")
    mode, requested = _resolve_mode_explicit(mode)

    if mode == "pk_ratio":
        if pk is None:
            if requested:
                _warn_once(
                    "pk_ratio-no-backend",
                    "growth: mode='pk_ratio' needs a P(k) backend but none was "
                    "passed; falling back to the growth ODE.",
                )
        elif not backend_has_native_z(pk):
            _warn_once(
                f"pk_ratio-no-z:{type(pk).__name__}",
                f"growth: backend {type(pk).__name__!r} has no redshift "
                "dependence of its own; falling back to the growth ODE.",
            )
        else:
            return growth_pk_ratio(z, theta, pk, variant=variant)
        mode = "ode"

    if variant == "cold" and _has_massive_nu(theta):
        _warn_once(
            f"cold-analytic:{mode}",
            f"growth: mode={mode!r} has no neutrino sector, so variant='cold' "
            "returns the total-matter growth; use mode='pk_ratio' with a "
            "backend that resolves the cold field for D_cb(z).",
        )

    # Imported lazily: halo_mass_function is a heavy module and will import this
    # one in turn once the backends declare their growth (phase B).
    from hod_mod.core.halo_mass_function import (
        _growth_factor_cpl_jax, _growth_factor_flat_jax)

    if mode == "carroll":
        return _growth_factor_flat_jax(z, theta["Omega_m"])
    return _growth_factor_cpl_jax(z, theta["Omega_m"],
                                  theta.get("w0", -1.0), theta.get("wa", 0.0))


def _has_massive_nu(theta: dict) -> bool:
    """Whether theta carries a *concrete*, non-zero Σm_ν.

    A traced Σm_ν has no value to test, and Python-level control flow on a
    tracer is impossible — so under ``jit`` this reports False and the advisory
    warning simply does not fire.  It gates a warning, never a computation.
    """
    from hod_mod.core.power_spectrum import mnu_sum
    v = mnu_sum(theta)
    if isinstance(v, jax.core.Tracer):
        return False
    try:
        return float(v) > 0.0
    except TypeError:
        return False
