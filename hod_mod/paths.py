"""Filesystem locations for hod_mod, kept *outside* the source tree.

Generated outputs (MCMC chains, figures, caches) must never be written inside the
git repository — that is what bloats it. :func:`results_root` resolves a writable
location off the repo:

1. ``$HOD_MOD_RESULTS`` if set (use this to point runs at a project/scratch disk);
2. otherwise the per-user OS data dir, ``~/.local/share/hod_mod/results`` on Linux
   (via :mod:`platformdirs`).

Curated results are *read back* from Zenodo with :func:`hod_mod.data_io.fetch`; this
module is only about *where new outputs go*.

Examples
--------
>>> from hod_mod.paths import results_root, results_path
>>> results_root()                                  # doctest: +SKIP
PosixPath('/home/<user>/.local/share/hod_mod/results')
>>> p = results_path("benchmarks", "more2015", "flatchain.npz")   # doctest: +SKIP
>>> # p's parent directory now exists; write to p
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the code repository root.

    ``$HOD_MOD_REPO`` overrides the default, which is auto-detected from this
    file's location (``<repo>/hod_mod/paths.py``). Used to locate ``configs/``
    and the in-repo ``data/`` tree regardless of the current working directory.
    """
    env = os.environ.get("HOD_MOD_REPO")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent


def results_root() -> Path:
    """Return the writable root for generated results (never inside the repo).

    ``$HOD_MOD_RESULTS`` overrides the default
    ``platformdirs.user_data_dir("hod_mod")/results``.
    """
    env = os.environ.get("HOD_MOD_RESULTS")
    if env:
        return Path(env).expanduser()
    import platformdirs

    return Path(platformdirs.user_data_dir("hod_mod")) / "results"


def results_path(*parts: str | os.PathLike, mkdir: bool = True) -> Path:
    """Return ``results_root()/parts``, creating the parent directory by default.

    Parameters
    ----------
    *parts
        Path components below :func:`results_root` (e.g. a run name and filename).
    mkdir : bool, default True
        Create the returned path's parent directory if it does not exist.
    """
    p = results_root().joinpath(*[os.fspath(x) for x in parts])
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def data_root() -> Path:
    """Return the root for *read* data (large inputs kept off / out of the repo).

    ``$HOD_MOD_DATA_DIR`` overrides the default in-repo ``hod_mod/data`` directory.
    Use it to point at a moved / shared data mirror (e.g. the reconstructed X-ray
    energy-band ``w(θ)`` under ``$HOD_MOD_DATA_DIR/xray_bands/<basename>/``).  This
    is the same variable the :mod:`hod_mod.data_io` registry checks for a local
    mirror, so one export covers both.
    """
    env = os.environ.get("HOD_MOD_DATA_DIR")
    return Path(env).expanduser() if env else Path(__file__).parent / "data"


def data_path(*parts: str | os.PathLike, mkdir: bool = False) -> Path:
    """Return ``data_root()/parts``.

    Parameters
    ----------
    *parts
        Path components below :func:`data_root` (e.g. ``"xray_bands", basename``).
    mkdir : bool, default False
        Create the returned path's parent directory if it does not exist (for
        writers such as ``reconstruct_band_wtheta``; readers leave it False).
    """
    p = data_root().joinpath(*[os.fspath(x) for x in parts])
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sum_stat_root() -> Path:
    """Return the root of the ``sum_stat`` measurement products.

    Defaults to the ``sum_stat`` package's ``data`` directory under the user's
    ``software`` tree; set ``$HOD_MOD_SUMSTAT`` to override.
    """
    env = os.environ.get("HOD_MOD_SUMSTAT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "software" / "sum_stat" / "data"


def cache_root() -> Path:
    """Return a cache directory for compilation artifacts (e.g. JAX XLA caches).

    ``$HOD_MOD_CACHE`` overrides the default OS user-cache dir.
    """
    env = os.environ.get("HOD_MOD_CACHE")
    if env:
        return Path(env).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_dir("hod_mod"))
