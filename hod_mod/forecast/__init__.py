"""JAX Fisher-forecast toolkit for the ZM15 + X-ray gas & AGN pipeline.

Public entry points:

* :class:`~hod_mod.forecast.forward_jax.ForwardModel` — differentiable forward
  model producing w_p, ΔΣ, C_ℓ^{gX}, C_ℓ^{gy}, C_ℓ^{XX};
* :class:`~hod_mod.forecast.pk_eisenstein_hu.EisensteinHu98PkLinear` — the
  σ8-parameterised, JAX-differentiable EH98 linear power spectrum;
* :mod:`~hod_mod.forecast.params` — fiducial values, labels, Planck priors;
* :mod:`~hod_mod.forecast.fisher` — Jacobian, Fisher matrix, constraints,
  degeneracy and probe/scale attribution;
* :class:`~hod_mod.forecast.tier2.Tier2Forecast` — the tier-2 (z, M*)-grid
  multi-probe assembly with physical survey noise
  (:mod:`~hod_mod.forecast.noise`).
"""

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES, OBSERVABLES
from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear

__all__ = ["ForwardModel", "PARAM_NAMES", "OBSERVABLES", "EisensteinHu98PkLinear",
           "Tier2Forecast"]


def __getattr__(name):
    # lazy: Tier2Forecast pulls in noise/apec_bands machinery only when used
    if name == "Tier2Forecast":
        from hod_mod.forecast.tier2 import Tier2Forecast
        return Tier2Forecast
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
