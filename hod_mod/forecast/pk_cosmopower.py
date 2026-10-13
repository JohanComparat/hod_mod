r"""Differentiable linear power spectrum from the CosmoPower-JAX emulator.

This is the CAMB-accurate, JAX-differentiable drop-in for
:class:`hod_mod.forecast.pk_eisenstein_hu.EisensteinHu98PkLinear`.  It wraps the
bundled ``mpk_lin`` neural emulator of `CosmoPower-JAX
<https://github.com/dpiras/cosmopower-jax>`_ (Piras & Spurio Mancini 2023,
arXiv:2305.06347), whose ``predict`` is a pure-JAX network — so ``jax.grad`` /
``jax.jacfwd`` flow through the cosmology, exactly what gradient-based inference
(blackjax NUTS in :mod:`hod_mod.fitting.jax_inference`) needs.

Why this over EH98?  The analytic EH98 transfer function is only ``<10%`` rms
vs CAMB and is band-aided near the fiducial by the linearized
:mod:`hod_mod.forecast.pk_camb_ratio` table.  The emulator reproduces the CAMB
*shape* to ``<0.1%`` across its whole training box (validated in
``forecast/tests/test_pk_cosmopower.py``), so NUTS posteriors are trustworthy,
not just fast.

Interface (identical duck-typed contract to ``EisensteinHu98PkLinear``):

* ``pk_linear(k, z, theta)`` — physical P(k, z) [(Mpc/h)³] for the 2-halo term,
  amplitude from ``ln10^{10}A_s``, growth from the network's own redshift.
* ``as_hmf_pk_func()`` — a ``(k, z, theta) -> P(k, 0)`` callable for
  :class:`~hod_mod.core.halo_mass_function.HaloMassFunction`, which applies its
  own growth.  Physical, not a shape.
* ``pk_shape(k, theta)`` — the amplitude-free shape, kept for callers that
  normalise the amplitude away regardless.  No longer on the σ(M) path.

``pk_linear`` used to branch on ``theta``: a ``sigma8`` key took a shape emulated
at ``z = 0`` with a reference ``A_s`` and evolved by the analytic growth ODE,
while ``ln10^{10}A_s`` took the network natively.  Different physics, chosen by
dictionary key — and since ``theta`` routinely carried *both*, the σ8 branch won
and changes to ``ln10^{10}A_s`` did nothing.  ``sigma8`` is retired; only the
native path remains.

Conventions.  The emulator is native in **1/Mpc** and **Mpc³** and takes
``[omega_b, omega_cdm, h, n_s, ln10^{10}A_s, z]`` (physical densities
``omega_x = Ω_x h²``).  We convert ``k → k/h`` [h/Mpc], ``P → P·h³``
[(Mpc/h)³].  The training box is ΛCDM with massless neutrinos; ``w0``/``wa``
do **not** enter this backend at all (unlike EH98, whose amplitude carries
``D(z=0)`` for flat CPL), and ``Σm_ν`` enters through the same CLASS-distilled
shape suppression as EH98.
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp

from hod_mod.core.amplitude import require_as_amplitude
from hod_mod.core.power_spectrum import (
    omega_cdm_cold_equivalent as _omega_cdm_cold)
# Shared with the EH98 wrapper: the σ(R=8) grid (both ultimately
# hod_mod.core.amplitude.K_INT) and the CLASS-distilled ν suppression.
from hod_mod.forecast.pk_eisenstein_hu import (
    _K_INT,
    _nu_suppression,
)

# Reference ln(10^10 A_s) for the amplitude-free :meth:`pk_shape`.  Its callers
# normalise the overall amplitude away, so the value is immaterial; the
# Planck-2018 central value keeps the network near its training centre.
_REF_LN10_10_AS = 3.044

#: Emulator training box — the emulator is a neural network and is only
#: *validated* inside the hypercube it was trained on; outside it the network
#: extrapolates with no accuracy guarantee.  Recovered from the network's own
#: standardisation stats (``param_train_mean ± √3·param_train_std``, exact for a
#: uniform training prior) and asserted against them in the tests.  Note ``h``'s
#: floor of 0.64 sits only ~0.03 below the Planck-18 fiducial 0.6736 — a sampler
#: with a wide ``h`` prior will leave the box, so set priors accordingly.
TRAINING_BOX = {
    "omega_b": (0.01875, 0.02625),      # Ω_b h²
    "omega_cdm": (0.05, 0.255),         # Ω_cdm h²
    "h": (0.64, 0.82),
    "n_s": (0.84, 1.10),
    "ln10^{10}A_s": (1.61, 3.91),
    "z": (0.0, 5.0),
}

_WARNED_OUT_OF_BOX = False


def check_training_box(params: dict, warn: bool = True) -> list:
    """Return the names in ``params`` that fall outside :data:`TRAINING_BOX`.

    ``params`` uses the *emulator's* names (``omega_b``, ``omega_cdm``, ``h``,
    ``n_s``, ``ln10^{10}A_s``, ``z``).  Values that are JAX tracers are skipped —
    a jitted/NUTS call has no concrete value to test, and Python-level control
    flow on a tracer is impossible.  Call this with concrete fiducials/prior edges
    (e.g. from a driver) to check coverage before launching a sampler.

    With ``warn`` the first out-of-box call in the process emits a single
    ``RuntimeWarning``; later ones are silent so a sampler cannot spam the log.
    """
    global _WARNED_OUT_OF_BOX
    bad = []
    for name, (lo, hi) in TRAINING_BOX.items():
        v = params.get(name)
        if v is None or isinstance(v, jax.core.Tracer):
            continue
        v = float(v)
        if not (lo <= v <= hi):
            bad.append(f"{name}={v:.5g} outside [{lo}, {hi}]")
    if bad and warn and not _WARNED_OUT_OF_BOX:
        _WARNED_OUT_OF_BOX = True
        warnings.warn(
            "CosmoPower-JAX called outside its training box; the network is "
            "extrapolating and its <0.1% CAMB accuracy no longer holds: "
            + "; ".join(bad)
            + ". (Further out-of-box calls will not be reported.)",
            RuntimeWarning, stacklevel=3,
        )
    return bad


@functools.lru_cache(maxsize=None)
def _load_emulator(probe: str = "mpk_lin"):
    """Load (and cache) the CosmoPower-JAX emulator for ``probe``.

    Import is lazy so ``hod_mod`` still imports without cosmopower-jax installed
    (mirrors the optional-``blackjax`` pattern in ``fitting.jax_inference``).
    """
    try:
        from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
    except ImportError as e:                                     # pragma: no cover
        raise ImportError(
            "CosmoPowerJaxPkLinear needs cosmopower-jax — "
            "`pip install cosmopower-jax` (or the repo's `emulator` extra)."
        ) from e
    return CosmoPowerJAX(probe=probe)


def _loglog_interp(logk, logk_grid, logp_grid):
    """log-log interpolation with power-law (constant-slope) tail extrapolation.

    ``jnp.interp`` clamps outside the grid; we instead continue the end slopes so
    the σ(R) integrals over the wide ``_K_INT`` grid (up to 10³ h/Mpc, beyond the
    emulator's ~15 h/Mpc support) keep a physical ``P ∝ k^{n_eff}`` tail.  Fully
    differentiable (``jnp.interp`` + ``jnp.where``).  ``logk_grid`` is increasing.
    """
    p = jnp.interp(logk, logk_grid, logp_grid)
    s_hi = (logp_grid[-1] - logp_grid[-2]) / (logk_grid[-1] - logk_grid[-2])
    p = jnp.where(logk > logk_grid[-1],
                  logp_grid[-1] + s_hi * (logk - logk_grid[-1]), p)
    s_lo = (logp_grid[1] - logp_grid[0]) / (logk_grid[1] - logk_grid[0])
    p = jnp.where(logk < logk_grid[0],
                  logp_grid[0] + s_lo * (logk - logk_grid[0]), p)
    return p


class CosmoPowerJaxPkLinear:
    """A_s-parameterised, JAX-differentiable CosmoPower-JAX linear P(k).

    Parameters
    ----------
    probe : str
        CosmoPower-JAX probe name for the linear matter spectrum (``"mpk_lin"``).
    """

    #: The network is trained over ``z in [0, 5]`` and :meth:`pk_linear` calls it
    #: at the requested ``z``, so it supplies its own growth.  Unconditionally
    #: now: the sigma8 branch whose redshift dependence *was* the growth ODE --
    #: and which made this attribute a statement about one branch rather than
    #: about the class -- is gone.
    has_native_z = True

    def __init__(self, probe: str = "mpk_lin"):
        self._probe = str(probe)
        self._emu = _load_emulator(self._probe)
        # native emulator wavenumbers, in 1/Mpc (a fixed jnp constant)
        self._logk_native = jnp.log(jnp.asarray(self._emu.modes))

    # -- shape spectrum, for the HMF -----------------------------------
    def pk_shape(self, k: jnp.ndarray, theta: dict) -> jnp.ndarray:
        """Emulated linear P(k) at the reference ``A_s`` and z = 0.

        Amplitude-free by convention: every remaining caller normalises it away
        (a pivot ratio, or a log-slope).  **Off the sigma(M) path** -- that is
        :meth:`as_hmf_pk_func`, which now carries the real amplitude.  When
        ``theta`` carries ``sum_mnu`` (a static dict-membership branch) the same
        CLASS-distilled massive-ν suppression as the EH98 path multiplies the
        (massless-trained) emulator output.
        """
        return self._emulate(k, theta, _REF_LN10_10_AS, 0.0)

    # -- the raw emulator call, shared by both amplitude conventions -----
    def _emulate(self, k, theta, ln10As, z):
        k = jnp.asarray(k)
        h = theta["h"]
        Ob = theta["Omega_b"]
        # The network is massless-trained, so it is evaluated at the massless-
        # equivalent cold density and corrected by the nu_ratio table; see
        # power_spectrum.omega_cdm_cold_equivalent.
        Ocdm = _omega_cdm_cold(theta)
        n_s = theta["n_s"]
        omega_b, omega_cdm = Ob * h ** 2, Ocdm * h ** 2
        check_training_box({"omega_b": omega_b, "omega_cdm": omega_cdm, "h": h,
                            "n_s": n_s, "ln10^{10}A_s": ln10As, "z": z})
        # emulator input vector [omega_b, omega_cdm, h, n_s, ln10^{10}A_s, z]
        params = jnp.stack([
            omega_b, omega_cdm, h, n_s,
            jnp.asarray(ln10As, dtype=k.dtype),
            jnp.asarray(z, dtype=k.dtype),
        ])
        pk_native = self._emu.predict(params)          # (n_modes,), Mpc³
        # 1/Mpc → h/Mpc and Mpc³ → (Mpc/h)³ (both fold h into the log grid)
        logk_grid = self._logk_native - jnp.log(h)
        logp_grid = jnp.log(pk_native) + 3.0 * jnp.log(h)
        pk = jnp.exp(_loglog_interp(jnp.log(k), logk_grid, logp_grid))
        if "sum_mnu" in theta:
            # The network is trained massless, so the correction is applied at
            # the emulated redshift rather than only at z = 0.
            pk = pk * _nu_suppression(k, theta, z)
        return pk

    # -- physical spectrum, for the 2-halo term ------------------------
    def pk_linear(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        r"""Physical P(k, z) [(Mpc/h)³], amplitude from ``ln10^{10}A_s``.

        The emulator is called with the actual ``A_s`` *and* the actual ``z``
        (it is trained over ``z in [0, 5]``), so the amplitude and the growth
        both come straight from the network -- no external growth factor, no
        renormalisation.

        There used to be a second branch here, taken first whenever ``theta``
        carried a ``sigma8``: it emulated the *shape* at ``z = 0`` with a
        reference ``A_s`` and evolved it with the analytic growth ODE.  That is
        not the same physics as this one, and which of the two ran was decided
        by a dictionary key.  Worse, ``theta`` routinely carried *both* keys, so
        the branch order silently made ``sigma8`` win and any change to
        ``ln10^{10}A_s`` -- including the one a free-cosmology sampler was
        making -- do nothing at all.
        """
        require_as_amplitude(theta, "CosmoPowerJaxPkLinear.pk_linear")
        k = jnp.asarray(k)
        return self._emulate(k, theta, theta["ln10^{10}A_s"], z)

    # -- convenience: a plain callable for make_hmf(pk_func=...) --------
    def as_hmf_pk_func(self):
        """Return a ``(k, z, theta) -> P(k, 0)`` callable for the HMF.

        The z=0 **physical** spectrum, not a shape.  ``HaloMassFunction.sigma``
        used to be handed ``pk_shape`` -- emulated at a reference ``A_s`` -- and
        rescale it to ``theta['sigma8']``; with that rescale gone the amplitude
        has to arrive here instead.  ``z`` is ignored because the HMF applies
        its own growth, and it always calls with ``0.0``.
        """
        def _pk_func(k, z, theta):
            require_as_amplitude(theta, "CosmoPowerJaxPkLinear.as_hmf_pk_func")
            return self._emulate(jnp.asarray(k), theta,
                                 theta["ln10^{10}A_s"], 0.0)
        return _pk_func

    @staticmethod
    def default_cosmology() -> dict:
        """Planck 2018 flat-ΛCDM, in the A_s convention.

        Identical to :meth:`hod_mod.core.power_spectrum.LinearPowerSpectrum.default_cosmology`
        (delegated, so the two backends can never drift apart) — part of the
        duck-typed surface the production code calls on a ``pk_lin`` object.
        """
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        return LinearPowerSpectrum.default_cosmology()
