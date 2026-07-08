"""Small shared numerical helpers (float32-safe log floors)."""

import jax.numpy as jnp


def safe_log(P, floor: float = 1e-30):
    """``log(max(P, floor))`` with non-finite ``P`` floored instead of propagated.

    Central helper for building log-power tables that are later interpolated
    (Hankel/Limber transforms).  Two failure modes it guards against:

    * tiny high-k tails can underflow float32 to NaN/inf inside halo-model
      integrals; those k are beyond any fitted scale, so they are floored to
      ``floor`` (≈ zero power) rather than letting one NaN poison the whole
      transform;
    * ``floor`` must stay representable in float32 (JAX's default dtype): a
      smaller value (e.g. 1e-60) underflows to 0, so ``log`` returns ``-inf``
      for an all-zero field, and a constant ``-inf`` table makes
      ``jnp.interp`` produce ``(-inf) - (-inf) = NaN`` downstream.

    Use ``floor=1e-20`` for the clustering P_gg/P_gm tables (their historical
    floor) and the default ``1e-30`` for the X-ray/tSZ cross-spectra tables.
    """
    P = jnp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.log(jnp.maximum(P, floor))
