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
    ``software`` tree; set ``$HOD_MOD_SUMSTAT`` to override.  The measurement
    products are published inside the sum_stat repository, so a plain clone at
    the default location is sufficient.
    """
    env = os.environ.get("HOD_MOD_SUMSTAT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "software" / "sum_stat" / "data"


_MANIFEST_CACHE: dict[Path, dict] = {}


def sum_stat_manifest(root: str | os.PathLike | None = None) -> dict:
    """Return sum_stat's ``summary.yaml`` manifest (parsed, cached per root).

    The manifest indexes every published measurement file — see sum_stat's
    ``docs/data_products.rst``.  All paths inside it are relative to *root*;
    the ``sum_stat_file*`` helpers below return them resolved.
    """
    root = Path(root).expanduser() if root is not None else sum_stat_root()
    if root not in _MANIFEST_CACHE:
        manifest_path = root / "summary.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"sum_stat manifest not found: {manifest_path}\n"
                "Clone https://github.com/JohanComparat/sum_stat at "
                "~/software/sum_stat (or set $HOD_MOD_SUMSTAT to its data/ "
                "directory). If you re-measured locally, regenerate the "
                "manifest with sum_stat's scripts/gen_summary_yaml.py."
            )
        import yaml

        _MANIFEST_CACHE[root] = yaml.safe_load(manifest_path.read_text())
    return _MANIFEST_CACHE[root]


def sum_stat_file(
    sample_id: str, statistic: str = "joint", root: str | os.PathLike | None = None
) -> Path:
    """Resolve one published measurement file: ``samples[sample_id][statistic]``.

    Parameters
    ----------
    sample_id : str
        Full sum_stat sample id, e.g.
        ``"LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238"``.
    statistic : str, default "joint"
        Manifest file key: ``"joint"``, ``"sz"``, or ``"cap"``.
    root : path-like, optional
        Override :func:`sum_stat_root`.
    """
    root = Path(root).expanduser() if root is not None else sum_stat_root()
    samples = sum_stat_manifest(root)["samples"]
    if sample_id not in samples:
        raise KeyError(
            f"unknown sum_stat sample {sample_id!r}; manifest has:\n  "
            + "\n  ".join(sorted(samples))
        )
    files = samples[sample_id]["files"]
    if statistic not in files:
        raise KeyError(
            f"sample {sample_id} has no {statistic!r} file "
            f"(available: {sorted(files)})"
        )
    return root / files[statistic]


def _by_threshold(entries: dict, mstar_min: float, what: str) -> dict:
    matches = [
        e for e in entries.values()
        if abs(e["meta"]["mstar_min"] - mstar_min) < 1e-6
    ]
    if not matches:
        avail = sorted(e["meta"]["mstar_min"] for e in entries.values())
        raise KeyError(f"no {what} sample with mstar_min={mstar_min}; available: {avail}")
    return matches[0]


def sum_stat_file_by_threshold(
    mstar_min: float, statistic: str = "joint", root: str | os.PathLike | None = None
) -> Path:
    """Resolve a BGS VLIM measurement by its M* threshold (e.g. ``10.0``).

    Deterministic manifest lookup — replaces the historical
    ``sorted(glob(...))[-1]`` pattern, whose pick silently depended on
    lexicographic filename order.
    """
    root = Path(root).expanduser() if root is not None else sum_stat_root()
    entry = _by_threshold(sum_stat_manifest(root)["samples"], mstar_min, "BGS")
    files = entry["files"]
    if statistic not in files:
        raise KeyError(
            f"BGS mstar_min={mstar_min} has no {statistic!r} file (available: {sorted(files)})"
        )
    return root / files[statistic]


def sum_stat_mass_bin_files(
    root: str | os.PathLike | None = None,
) -> list[tuple[float, float, Path]]:
    """Return the thin-mass-bin files as sorted ``(mstar_lo, mstar_hi, path)``."""
    root = Path(root).expanduser() if root is not None else sum_stat_root()
    out = [
        (b["mstar_lo"], b["mstar_hi"], root / b["file"])
        for entry in sum_stat_manifest(root)["mass_bins"].values()
        for b in entry["bins"]
    ]
    return sorted(out)


def sum_stat_mock_file(
    mstar_min: float, statistic: str = "wp", root: str | os.PathLike | None = None
) -> Path:
    """Resolve an Uchuu mock measurement by M* threshold.

    ``statistic`` is one of ``"joint"``, ``"smf"``, ``"wp"``, ``"wtheta"``.
    """
    root = Path(root).expanduser() if root is not None else sum_stat_root()
    entry = _by_threshold(sum_stat_manifest(root)["mocks"], mstar_min, "mock")
    files = entry["files"]
    if statistic not in files:
        raise KeyError(
            f"mock mstar_min={mstar_min} has no {statistic!r} file (available: {sorted(files)})"
        )
    return root / files[statistic]


def cache_root() -> Path:
    """Return a cache directory for compilation artifacts (e.g. JAX XLA caches).

    ``$HOD_MOD_CACHE`` overrides the default OS user-cache dir.
    """
    env = os.environ.get("HOD_MOD_CACHE")
    if env:
        return Path(env).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_dir("hod_mod"))
