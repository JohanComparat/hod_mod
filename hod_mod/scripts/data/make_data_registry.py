#!/usr/bin/env python
"""Generate ``hod_mod/data_io/registry.txt`` for the on-demand data fetcher.

The registry is a ``filename  hash`` table that :mod:`hod_mod.data_io.registry`
ships with the package; :func:`hod_mod.data_io.fetch` uses it to verify every
download from the Zenodo data record.

Two modes
---------
``--from-doi 10.5281/zenodo.XXXXXXX``
    *Authoritative.* Pull the exact file list and checksums straight from a
    **published** Zenodo record (via ``pooch``). Run this once the deposit is
    uploaded; it sidesteps any local/Zenodo filename-mapping guesswork.

``--from-local``
    *Provisional.* Walk the curated local files (final chains, headline
    figures, summary JSON) plus any ``--extra`` Tier-2 input directories,
    hashing each with SHA256. Use this to preview the archive and produce a
    placeholder registry *before* uploading.

Usage
-----
    # before upload — see what will be archived and stage a registry
    python hod_mod/scripts/data/make_data_registry.py --from-local --dry-run
    python hod_mod/scripts/data/make_data_registry.py --from-local

    # after upload — authoritative registry from the real record
    python hod_mod/scripts/data/make_data_registry.py \
        --from-doi 10.5281/zenodo.1234567
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path

# repo root = three levels up from hod_mod/scripts/data/<this file>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_FILE = _REPO_ROOT / "hod_mod" / "data_io" / "registry.txt"

# Curated benchmark *results* to archive (Tier 3). Globs are matched against
# paths relative to the repo root. Final posteriors + headline figures + run
# summaries only — the 60k+ intermediate .npz are deliberately excluded and
# are regenerable from these plus a committed config.
CURATED_GLOBS = [
    "results/**/flatchain.npz",
    "results/**/chain.h5",
    "results/**/*summary*.json",
    "results/**/*corner*.png",
    "results/showcase/*.png",
    "results/showcase/*.pdf",
]


def _zenodo_key(name: str) -> str:
    # Mirror of hod_mod.data_io.registry.zenodo_key: Zenodo file keys cannot
    # contain '/', so nested paths are flattened with '__'. Kept inline here so
    # this script stays importable without the package on sys.path.
    return name.replace("/", "__")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_curated(root: Path):
    """Yield (key, abspath) for every curated file under *root*."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, g.replace("**/", "*")) or
               fnmatch.fnmatch(rel, g) for g in CURATED_GLOBS):
            yield rel, path


def _iter_extra(extra_dirs: list[str]):
    """Yield (key, abspath) for every file under each --extra directory.

    The key is ``<dirname>/<relpath>`` so Tier-2 inputs land under a stable
    prefix on Zenodo (e.g. ``LSDR10_GALxEVT/...``).
    """
    for d in extra_dirs:
        base = Path(d).expanduser().resolve()
        if not base.exists():
            print(f"  ! --extra path not found, skipping: {base}")
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                key = f"{base.name}/{path.relative_to(base).as_posix()}"
                yield key, path


def from_local(extra_dirs: list[str], dry_run: bool) -> dict[str, str]:
    registry: dict[str, str] = {}
    total = 0
    print(f"Scanning curated results under {_REPO_ROOT / 'results'} ...")
    for key, path in _iter_curated(_REPO_ROOT):
        registry[_zenodo_key(key)] = f"sha256:{_sha256(path)}"
        total += path.stat().st_size
    for key, path in _iter_extra(extra_dirs):
        registry[_zenodo_key(key)] = f"sha256:{_sha256(path)}"
        total += path.stat().st_size
    print(f"  {len(registry)} files, {total / 1048576:.1f} MB total")
    if dry_run:
        for key in registry:
            print(f"    {key}")
    return registry


def from_doi(doi: str) -> dict[str, str]:
    import pooch

    print(f"Reading authoritative file list from doi:{doi} ...")
    pup = pooch.create(path=pooch.os_cache("hod_mod"),
                       base_url=f"doi:{doi}/", registry=None)
    pup.load_registry_from_doi()
    print(f"  {len(pup.registry)} files")
    return dict(pup.registry)


def write_registry(registry: dict[str, str]) -> None:
    lines = [f"{name} {h}\n" for name, h in sorted(registry.items())]
    _REGISTRY_FILE.write_text("".join(lines))
    print(f"Wrote {len(registry)} entries -> {_REGISTRY_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-doi", metavar="DOI",
                      help="published Zenodo DOI to read the registry from")
    mode.add_argument("--from-local", action="store_true",
                      help="hash local curated files (provisional registry)")
    ap.add_argument("--extra", action="append", default=[], metavar="DIR",
                    help="extra Tier-2 input directory to include (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files / hashes without writing registry.txt")
    args = ap.parse_args()

    if args.from_doi:
        registry = from_doi(args.from_doi)
    else:
        registry = from_local(args.extra, args.dry_run)

    if args.dry_run:
        print("(dry run — registry.txt not written)")
        return
    write_registry(registry)


if __name__ == "__main__":
    main()
