"""JAX Fisher-forecast toolkit for the ZM15 + X-ray gas & AGN pipeline.

Public entry points:

* :class:`~hod_mod.forecast.forward_jax.ForwardModel` — differentiable forward
  model producing w_p, ΔΣ, C_ℓ^{gX}, C_ℓ^{gy}, C_ℓ^{XX};
* :class:`~hod_mod.forecast.pk_eisenstein_hu.EisensteinHu98PkLinear` — the
  σ8-parameterised, JAX-differentiable EH98 linear power spectrum;
* :mod:`~hod_mod.forecast.params` — fiducial values, labels, Planck priors;
* :mod:`~hod_mod.forecast.fisher` — Jacobian, Fisher matrix, constraints,
  degeneracy and probe/scale attribution.
"""

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES, OBSERVABLES
from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear

__all__ = ["ForwardModel", "PARAM_NAMES", "OBSERVABLES", "EisensteinHu98PkLinear"]
