r"""Tomographic (multi-sample) Fisher forecast.

Generalises the single-sample :class:`~hod_mod.forecast.forward_jax.ForwardModel`
to a set of stellar-mass / redshift bins that **share** the cosmological, gas,
AGN and baryon parameters but carry a **per-bin HOD**.  A single global parameter
vector

    [ (N_PARAM − 9) shared params | HOD(9) × N_bin ]

is unpacked (by a static gather index per bin) into each bin's ``PARAM_NAMES``
``ForwardModel`` parameter vector, so one ``jax.jacfwd`` of the concatenated
multi-bin data vector yields the full tomographic Jacobian — the cross-bin
sharing of cosmology is handled automatically by the chain rule.

Cross-bin covariance (adjacent z-bins share large-scale modes) plugs in at the
covariance stage; this module provides the mean-model Jacobian and the parameter
bookkeeping.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES
from hod_mod.forecast import params

_HOD_NAMES = ["lg_m1h", "lg_m0star", "beta", "delta", "gamma",
              "sigma_lnmstar", "eta", "fc", "bsat"]

# Default BGS stellar-mass-threshold / redshift bins (fit_comparat2025.SAMPLES).
DEFAULT_SAMPLES = [("S1", 0.135, 10.0), ("S2", 0.15, 10.3), ("S3", 0.18, 10.6),
                   ("S4", 0.20, 10.8), ("S5", 0.22, 11.0), ("S6", 0.24, 11.2),
                   ("S7", 0.261, 11.5)]


class TomographicForecast:
    """Shared-cosmology, per-bin-HOD multi-sample forecast.

    Parameters
    ----------
    samples : list of (label, z_eff, log10m_star_thresh)
    model_kw : forwarded to each per-bin :class:`ForwardModel` (grids, n_z, ...).
    """

    def __init__(self, samples=None, **model_kw):
        self.samples = list(samples if samples is not None else DEFAULT_SAMPLES)
        self.labels = [s[0] for s in self.samples]
        self.n_bin = len(self.samples)
        self.models = [ForwardModel(z_eff=z, log10m_star_thresh=thr, **model_kw)
                       for (_, z, thr) in self.samples]

        self.shared_names = [n for n in PARAM_NAMES if n not in _HOD_NAMES]
        self.global_names = (self.shared_names
                             + [f"{h}__{lab}" for lab in self.labels for h in _HOD_NAMES])
        gidx = {n: i for i, n in enumerate(self.global_names)}
        # static per-bin gather: global vector → the bin's 31-entry PARAM_NAMES vector
        self.gather = []
        for lab in self.labels:
            g = np.array([gidx[n] if n in self.shared_names else gidx[f"{n}__{lab}"]
                          for n in PARAM_NAMES], dtype=int)
            self.gather.append(jnp.asarray(g))

    # ---- parameters --------------------------------------------------
    def fiducial(self):
        """Global fiducial vector (shared params + replicated per-bin HOD fiducials)."""
        fmap = {n: v for n, v in zip(PARAM_NAMES, params.fiducial_vector())}
        return np.array([fmap[n.split("__")[0]] for n in self.global_names], dtype=float)

    def prior(self, add_planck=False, fix=()):
        """Regularizing prior on the global vector (shared + per-bin HOD)."""
        base = {n: s for n, s in zip(PARAM_NAMES,
                params.regularizing_prior(add_planck=add_planck, fix=fix))}
        return np.array([base[n.split("__")[0]] for n in self.global_names], dtype=float)

    # ---- data vector + Jacobian -------------------------------------
    def data_vector_fn(self, which=None):
        """Return ``f(theta_global) -> stacked data vector`` + row metadata.

        Row metadata is ``(row_bin, row_obs, row_x)``.  A single ``jax.jacfwd`` of
        ``f`` gives the tomographic Jacobian over the global parameter vector.
        """
        per_bin = [m.full_data_vector_fn(which) for m in self.models]
        row_bin, row_obs, row_x = [], [], []
        for b, (_, ro, rx) in enumerate(per_bin):
            row_bin += [self.labels[b]] * len(ro)
            row_obs += list(ro)
            row_x += list(rx)

        def f(theta_global):
            parts = [per_bin[b][0](theta_global[self.gather[b]]) for b in range(self.n_bin)]
            return jnp.concatenate(parts)
        return f, np.array(row_bin), np.array(row_obs), np.array(row_x, dtype=float)

    def scale_cut_mask(self, row_bin, row_obs, row_x, rmin):
        """Per-bin scale-cut mask (each bin uses its own χ(z_eff) for the ℓ cut)."""
        keep = np.zeros(len(row_obs), dtype=bool)
        for b, lab in enumerate(self.labels):
            sel = row_bin == lab
            keep[sel] = self.models[b].scale_cut_mask(row_obs[sel], row_x[sel], rmin)
        return keep
