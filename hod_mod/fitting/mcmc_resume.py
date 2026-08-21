"""Restore rank to a resumed emcee ensemble that has gone linearly dependent.

Why this module exists
----------------------
A long stretch-move run can collapse the walkers onto a lower-dimensional affine
subspace.  emcee then refuses to restart:

.. code-block:: text

    ValueError: Initial state has a large condition number.

(:func:`emcee.ensemble.walkers_independent` rejects a column-normalised ensemble
whose condition number exceeds ``1e8``.)  Because a checkpointed job resumes from
the backend's stored state, the chain is then **permanently** stuck: every
resubmission dies before its first step, however much walltime it is given.  This
is what froze ``bgs_full_joint_allparams_v0.3`` at 1061/4000 from 18 July, and
``bgs_zm15_thresh_joint_v0.3`` at 1909/2500 from 29 July.

Why ``skip_initial_state_check=True`` is not the fix
----------------------------------------------------
The stretch move proposes only along lines joining pairs of walkers, so it cannot
leave the affine hull the ensemble spans.  Skipping the check on a rank-15-of-28
ensemble does not recover the missing 13 directions — it samples a slice of the
posterior while looking like a healthy run.  Rank has to be put back explicitly,
which is what :func:`revive_ensemble` does; the flag is then passed only so the
check cannot veto an ensemble we have already repaired.

The jitter is scaled per parameter by the ensemble's own spread, so it is small
against the posterior width but large against the collapse.  Reviving restarts
convergence for the affected segment: treat post-revive samples as a fresh
burn-in rather than a seamless continuation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["revive_ensemble"]


def revive_ensemble(backend, lo=None, hi=None, jitter=0.05, seed=0, label=""):
    """``None`` if the stored ensemble is healthy, else re-scattered coordinates.

    Parameters
    ----------
    backend : emcee.backends.Backend
        The resumed backend; its last sample is inspected.
    lo, hi : array_like or None
        Optional per-parameter bounds to clip the re-scattered walkers into.
    jitter : float
        Re-scatter width as a fraction of each parameter's current walker spread.
    seed : int
        Seed for the re-scatter, so a repeated resume is reproducible.
    label : str
        Prefix for the log line.

    Returns
    -------
    ndarray or None
        Coordinates to pass as emcee's ``initial_state`` (emcee recomputes
        ``log_prob``), or ``None`` to let it continue from its stored state.
    """
    from emcee.ensemble import walkers_independent

    try:
        state = backend.get_last_sample()
    except Exception:                      # nothing stored yet -> nothing to revive
        return None
    coords = np.asarray(state.coords, float)
    if walkers_independent(coords):
        return None

    rng = np.random.default_rng(seed)
    spread = coords.std(axis=0)
    # a fully collapsed column has zero spread; fall back to the ensemble's
    # overall scale so those directions get re-seeded too
    step = np.where(spread > 0, spread, max(float(np.abs(coords).std()), 1e-8))
    revived = coords + jitter * step[None, :] * rng.standard_normal(coords.shape)
    if lo is not None and hi is not None:
        revived = np.clip(revived, np.asarray(lo) + 1e-6, np.asarray(hi) - 1e-6)
    ok = walkers_independent(revived)
    pre = f"[{label}] " if label else ""
    print(f"{pre}resumed ensemble was linearly dependent; re-scattered walkers "
          f"by {jitter:.0%} of their spread (rank restored: {ok}). Treat the "
          f"post-restart segment as a fresh burn-in.", flush=True)
    return revived
