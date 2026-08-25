"""The two pipeline flavours, as one declarative object.

The package has always had two ways to evaluate the same forward model: an
accurate one for fitting and a fast differentiable one for forecasting.  Until
now the difference was spread across a ``pk_backend: str`` constructor flag
(:mod:`hod_mod.observables.clustering`), six independent re-derivations of the
same boolean by ``getattr(fhmp, "_pk_backend", "camb") == "eh98_jax"``, a
hand-maintained twin pair of ``_pk_tables_full`` / ``_pk_tables_full_jax``, and
a wholly separate re-implementation in :mod:`hod_mod.forecast.forward_jax`.
Nothing named the flavour, so nothing could report which one produced a number,
and the two implementations drifted apart in ways only a reader comparing them
line by line would notice.

A :class:`Backend` is that name.  It carries every knob on which the two
flavours legitimately differ — the linear-P(k) solver, the growth route, the
quadrature resolutions, the gas Fourier transform, the concentration relation —
and one flag, :attr:`Backend.traced`, saying whether the assembly it drives may
be put through ``jax.jit`` / ``jax.grad`` end to end.

Two profiles ship:

``ACCURATE``
    A Boltzmann solver, the fine grids, exact quadrature.  **Not**
    differentiable end to end: CAMB and CLASS are Fortran/C, and colossus
    concentrations are numpy.  This is what fits run on, and gradient-free
    optimisers and ``emcee`` do not need more.

``FAST``
    The CosmoPower-JAX emulator, coarser grids, analytic surrogates for the
    gas Fourier transforms.  Traceable end to end, which is what a Fisher
    forecast or a NUTS chain differentiates through.

``ACCURATE``'s numerical settings deliberately reproduce the values that were
hard-coded before this module existed, so selecting it changes nothing:
``n_k = 1024`` and ``k in [1e-4, 200]`` from ``FullHaloModelPrediction``,
``n_m = 512`` from ``HODBase._N_M_GRID``, ``n_ogata = 512`` from the ``_OG_N``
Hankel table, ``n_gl = 200`` from ``GasDensityDPM``.

The accuracy cost of choosing ``FAST`` is not asserted here — it is measured,
row by row, in ``benchmarks/flavour_parity.json``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

__all__ = ["Backend", "ACCURATE", "FAST", "BACKENDS", "resolve_backend"]


@dataclass(frozen=True)
class Backend:
    """A pipeline flavour: which solvers, which grids, and whether it traces.

    Frozen because it is read inside cached and jitted paths — a backend that
    could be mutated after a cache was keyed on it would serve stale tables.
    Use :meth:`with_` to derive a variant.
    """

    #: Short identifier, used in provenance and cache keys.
    name: str

    #: May the assembly driven by this backend be ``jit``/``grad``-ed end to
    #: end?  ``False`` does **not** mean "slow" — it means a gradient taken
    #: through it would silently omit the dependence of P(k) on the cosmology,
    #: which is why the API refuses one rather than returning it.
    traced: bool

    # -- layer 1: cosmology -------------------------------------------------
    #: ``hod_mod.core.power_spectrum.default_pk_linear`` key.
    pk: str
    #: ``hod_mod.core.growth`` mode.
    growth: str

    # -- grids --------------------------------------------------------------
    n_k: int
    k_min: float
    k_max: float
    n_m: int
    m_min: float
    m_max: float

    # -- layer 5: projection quadratures ------------------------------------
    #: Line-of-sight steps for ΔΣ, and its comoving limit [Mpc/h].
    n_chi: int
    chi_max: float
    #: Line-of-sight steps for w_p.
    n_pi: int
    #: Points in the internal ΔΣ radial tabulation.
    n_r_tab: int
    #: Nodes in the Ogata J0 Hankel table.
    n_ogata: int

    # -- layer 3: sector evaluation -----------------------------------------
    #: ``"gl"`` — the Gauss-Legendre radial quadrature of ``gas/conversions``;
    #: ``"surrogate"`` — the analytic differentiable gNFW form.
    gas_ft: str
    n_gl: int
    #: Concentration relation.  Must be traceable when :attr:`traced`.
    cm_model: str
    #: Mass definition for the halo boundary.
    mdef: str

    def with_(self, **overrides) -> "Backend":
        """A copy with some fields replaced, re-validated."""
        return _validate(replace(self, **overrides))

    def __str__(self) -> str:  # pragma: no cover - provenance strings
        return (f"{self.name}(pk={self.pk}, growth={self.growth}, "
                f"n_k={self.n_k}, n_m={self.n_m}, gas_ft={self.gas_ft}, "
                f"cm={self.cm_model}, traced={self.traced})")


#: Concentration relations that are pure JAX.  Everything else routes through
#: colossus, which is numpy *and* mutates global state via ``setCosmology``.
_TRACEABLE_CM = frozenset({"dutton14"})

#: Linear-P(k) backends that are pure JAX.
_TRACEABLE_PK = frozenset({"cosmopower", "eh98"})


def _validate(b: Backend) -> Backend:
    """Reject a backend whose parts cannot deliver what its flags promise.

    This is the check that used to live in
    ``clustering._validate_eh98_backend`` — an *observables* function importing
    from *forecast* to ``isinstance``-check its collaborators, the only
    dependency edge in the package that ran backwards.  Here it is a plain
    field check in ``core``, where it belongs.
    """
    if b.traced:
        if b.pk not in _TRACEABLE_PK:
            raise ValueError(
                f"backend {b.name!r} is traced but its linear P(k) backend "
                f"{b.pk!r} is not differentiable; use one of "
                f"{sorted(_TRACEABLE_PK)}"
            )
        if b.cm_model not in _TRACEABLE_CM:
            raise ValueError(
                f"backend {b.name!r} is traced but its concentration relation "
                f"{b.cm_model!r} routes through colossus, which is numpy and "
                f"sets global cosmology state; use one of {sorted(_TRACEABLE_CM)}"
            )
        if b.gas_ft != "surrogate":
            raise ValueError(
                f"backend {b.name!r} is traced but its gas Fourier transform "
                f"{b.gas_ft!r} is the numpy Gauss-Legendre path; use 'surrogate'"
            )
    if b.k_min <= 0.0 or b.k_max <= b.k_min:
        raise ValueError(f"backend {b.name!r} has an empty k range")
    if b.m_min <= 0.0 or b.m_max <= b.m_min:
        raise ValueError(f"backend {b.name!r} has an empty mass range")
    for field in ("n_k", "n_m", "n_chi", "n_pi", "n_r_tab", "n_ogata", "n_gl"):
        if getattr(b, field) < 2:
            raise ValueError(f"backend {b.name!r} has {field} < 2")
    return b


#: Fitting.  Every numerical setting matches what was hard-coded before this
#: module existed, so selecting it reproduces pre-0.6 numbers exactly.
ACCURATE = _validate(Backend(
    name="accurate",
    traced=False,
    pk="camb",
    growth="pk_ratio",
    n_k=1024, k_min=1e-4, k_max=200.0,
    n_m=512, m_min=1e10, m_max=1e16,
    n_chi=512, chi_max=300.0, n_pi=512, n_r_tab=256, n_ogata=512,
    gas_ft="gl", n_gl=200,
    cm_model="diemer19",
    mdef="200m",
))

#: Forecasting.  Differentiable end to end; its deviation from ``ACCURATE`` is
#: measured in ``benchmarks/flavour_parity.json``, not assumed.
FAST = _validate(Backend(
    name="fast",
    traced=True,
    pk="cosmopower",
    growth="pk_ratio",
    n_k=512, k_min=1e-4, k_max=200.0,
    n_m=256, m_min=1e10, m_max=1e16,
    n_chi=256, chi_max=300.0, n_pi=256, n_r_tab=256, n_ogata=256,
    gas_ft="surrogate", n_gl=64,
    cm_model="dutton14",
    mdef="200c",
))

#: The reference flavour: an independent Boltzmann solver, for the accuracy
#: budget rather than for production.  CLASS is the only backend that treats
#: massive neutrinos exactly.
REFERENCE = _validate(ACCURATE.with_(name="reference", pk="class"))

BACKENDS = {b.name: b for b in (ACCURATE, FAST, REFERENCE)}

#: Environment variable consulted when no backend is passed explicitly.
ENV_VAR = "HOD_MOD_BACKEND"


def resolve_backend(backend=None) -> Backend:
    """Return a :class:`Backend`: argument, then ``$HOD_MOD_BACKEND``, then ``ACCURATE``.

    Accepts a :class:`Backend` (returned unchanged) or one of the names in
    :data:`BACKENDS`.  The default is ``ACCURATE`` because a wrong *fast* answer
    is harder to notice than a slow one.
    """
    if isinstance(backend, Backend):
        return backend
    if backend is None:
        backend = os.environ.get(ENV_VAR)
    if backend is None:
        return ACCURATE
    key = str(backend).lower()
    if key not in BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {sorted(BACKENDS)} "
            f"or a Backend instance"
        )
    return BACKENDS[key]
