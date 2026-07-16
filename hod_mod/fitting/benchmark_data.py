r"""Load the ``benchmark_observables`` JSON tree into a fittable data vector.

The companion of :mod:`hod_mod.fitting.jax_inference`: that module builds a
:class:`~hod_mod.fitting.jax_inference.MultiProbeGaussianLikelihood` from a
forward model *and* a measured ``(data, icov)`` pair, but it does not know where
the data come from.  This module reads the published-measurement tree written by
``hod_mod.scripts.data.make_benchmark_observables`` (``$HOD_MOD_DATA_DIR/
benchmark_observables/``, one JSON per reference/observable/sample) and returns
a ``(data, icov)`` aligned to a forward model's own row layout, so the two snap
together with :meth:`MultiProbeGaussianLikelihood.from_forward_model`.

Two entry points:

* :func:`load_forecast_vector` — the multi-probe *forecast* path.  Reads the
  ``simulated`` entries (forward-model fiducial + Stage-IV forecast noise) for a
  set of observables and aligns each onto the abscissae of a forecast
  :class:`~hod_mod.forecast.forward_jax.ForwardModel` via
  ``full_data_vector_fn``.  Because a ``simulated`` JSON was dumped from the
  tier-2/tier-3 grid (many (z, M*) cells) while a single ``ForwardModel`` carries
  one grid, the loader selects the JSON block whose ``zeff`` is nearest the
  model's ``z_eff`` and interpolates that block onto the model's row abscissae.
* :func:`load_observed_sample` — the real-data path.  Returns the raw
  ``(x, y, y_err)`` arrays of one ``observed`` sample (e.g. a ``wp``/``ds`` CSV
  ingest), for the production ``wp``/``ds`` model or for tests.

The tree carries only per-point errors (``y_err``), so the covariance is
diagonal; if a sibling ``<stem>_cov.json`` file is present it is loaded as a full
covariance instead.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

__all__ = ["load_forecast_vector", "load_observed_sample", "available_observables"]


# --------------------------------------------------------------------------
# tree walking
# --------------------------------------------------------------------------

def _iter_entries(root):
    """Yield ``(path, body)`` for every JSON in the tree except ``index.json``."""
    for path in glob.glob(os.path.join(root, "*", "*", "*.json")):
        try:
            with open(path) as fh:
                yield path, json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue


def available_observables(root, provenance="simulated"):
    """Map ``observable -> json path`` for every entry of the given provenance."""
    out = {}
    for path, body in _iter_entries(root):
        if body.get("provenance", {}).get("type") == provenance:
            out.setdefault(body.get("observable"), path)
    return out


def _col(data, key):
    """A JSON data column as a float array with ``null`` -> ``nan``."""
    v = data.get(key)
    if v is None:
        return None
    return np.array([np.nan if x is None else float(x) for x in v], dtype=float)


# --------------------------------------------------------------------------
# interpolation onto a target abscissa
# --------------------------------------------------------------------------

def _interp_onto(x_target, x_src, y_src):
    """Interpolate ``y_src(x_src)`` onto ``x_target``.

    Log-space in ``y`` when all values are positive (spectra/abundances span
    decades); linear otherwise.  Ends are clamped (no extrapolation blow-up).
    """
    order = np.argsort(x_src)
    xs, ys = x_src[order], y_src[order]
    positive = np.all(ys > 0)
    yy = np.log10(ys) if positive else ys
    out = np.interp(x_target, xs, yy, left=yy[0], right=yy[-1])
    return 10.0 ** out if positive else out


def _select_block(x, y, e, zeff_col, block_col, z_eff):
    """Restrict a simulated observable to the (z, M*) block nearest ``z_eff``.

    ``simulated`` entries concatenate every forecast cell; a single forward
    model carries one.  Pick the block whose mean ``zeff`` is closest to the
    model's ``z_eff`` so the interpolation stays within one self-consistent cell.
    """
    if block_col is not None:
        labels = list(dict.fromkeys(block_col))
        if zeff_col is not None:
            bz = {lb: np.nanmean(zeff_col[block_col == lb]) for lb in labels}
            best = min(labels, key=lambda lb: abs(bz[lb] - z_eff))
        else:
            best = labels[0]
        sel = block_col == best
        return x[sel], y[sel], e[sel]
    if zeff_col is not None and np.isfinite(zeff_col).any():
        zt = zeff_col[np.nanargmin(np.abs(zeff_col - z_eff))]
        sel = zeff_col == zt
        return x[sel], y[sel], e[sel]
    return x, y, e


# --------------------------------------------------------------------------
# forecast (simulated) path
# --------------------------------------------------------------------------

def load_forecast_vector(root, which, row_obs, row_x, *, z_eff=0.2,
                         provenance="simulated"):
    r"""Assemble ``(data, icov)`` aligned to a forward model's row layout.

    Parameters
    ----------
    root : str
        ``benchmark_observables`` tree root.
    which : sequence[str]
        Observables to include; each **must** have an entry of ``provenance`` in
        the tree (filter with :func:`available_observables` first).
    row_obs, row_x : array
        The model's own row metadata from
        ``ForwardModel.full_data_vector_fn(which)`` — the loader fills a value for
        every row so the result matches ``f(theta)`` exactly and drops into
        :meth:`MultiProbeGaussianLikelihood.from_forward_model` with no masking.
    z_eff : float
        Effective redshift of the forward model; picks the nearest JSON block.

    Returns
    -------
    dict with ``data`` (len == ``row_x``), ``y_err``, ``icov`` (diagonal),
    ``row_obs``, ``row_x``, and ``sources`` (observable -> json path).
    """
    row_obs = np.asarray(row_obs)
    row_x = np.asarray(row_x, dtype=float)
    data = np.full(row_x.shape, np.nan)
    err = np.full(row_x.shape, np.nan)
    avail = available_observables(root, provenance)
    sources = {}
    for obs in dict.fromkeys(which):
        path = avail.get(obs)
        if path is None:
            raise KeyError(
                f"no '{provenance}' entry for observable '{obs}' in {root}; "
                f"call available_observables() to see the {len(avail)} present.")
        with open(path) as fh:
            dat = json.load(fh)["data"]
        x = _col(dat, "x")
        y = _col(dat, "y")
        e = _col(dat, "y_err")
        if x is None or y is None or e is None:
            raise ValueError(f"{path}: missing x/y/y_err columns")
        x, y, e = _select_block(x, y, e, _col(dat, "zeff"),
                                _col_str(dat, "block"), z_eff)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
        x, y, e = x[good], y[good], e[good]
        if x.size == 0:
            raise ValueError(f"{path}: no finite rows for observable '{obs}'")
        m = row_obs == obs
        data[m] = _interp_onto(row_x[m], x, y)
        err[m] = _interp_onto(row_x[m], x, e)
        sources[obs] = path
    if not np.all(np.isfinite(data) & np.isfinite(err) & (err > 0)):
        bad = np.where(~(np.isfinite(data) & np.isfinite(err) & (err > 0)))[0]
        raise ValueError(f"{len(bad)} model rows left unfilled (rows {bad[:5]}…); "
                         "an observable in `which` had no covering block.")
    return {"data": data, "y_err": err, "icov": np.diag(1.0 / err ** 2),
            "row_obs": row_obs, "row_x": row_x, "sources": sources}


def _col_str(data, key):
    v = data.get(key)
    if v is None:
        return None
    return np.array([str(x) for x in v])


# --------------------------------------------------------------------------
# observed (real-data) path
# --------------------------------------------------------------------------

def load_observed_sample(root, reference, observable, sample=None):
    """Return ``(x, y, y_err, meta)`` for one ``observed`` entry.

    Finds ``<wl>/<tr>/<reference>__<observable>[__<sample>].json``.  The first
    numeric column is the abscissa, the second the measurement, the third its
    error (matching the CSV ingest convention ``rp_hMpc, wp/ds, *_err``); when a
    full covariance is needed, load a sibling ``*_cov`` entry separately.
    """
    stem = f"{reference}__{observable}" + (f"__{sample}" if sample else "")
    hits = glob.glob(os.path.join(root, "*", "*", stem + ".json"))
    if not hits:
        raise FileNotFoundError(f"no observed entry {stem}.json under {root}")
    with open(hits[0]) as fh:
        body = json.load(fh)
    dat = body["data"]
    cols = [k for k in dat if isinstance(dat[k], list)]
    x = _col(dat, cols[0])
    y = _col(dat, cols[1])
    e = _col(dat, cols[2]) if len(cols) > 2 else np.full_like(y, np.nan)
    good = np.isfinite(x) & np.isfinite(y)
    meta = {"path": hits[0], "reference": body.get("reference", {}),
            "sample": body.get("sample", {}), "units": body.get("units", {}),
            "columns": cols}
    return x[good], y[good], e[good], meta
