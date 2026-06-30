#!/usr/bin/env python
"""Upload the curated data products to a Zenodo **draft** deposition.

This is a convenience wrapper over the Zenodo REST API for the data record that
:mod:`hod_mod.data_io.registry` fetches from. It creates (or adds a new version
to) a deposition, uploads files, and writes minimal metadata — but it **never
publishes**: review and click *Publish* in the Zenodo web UI to mint the DOI.
That keeps the irreversible step in your hands.

Auth
----
Set a personal access token (scope ``deposit:write``) in the environment::

    export ZENODO_TOKEN=...          # production
    export ZENODO_TOKEN=...          # or a sandbox token with --sandbox

Usage
-----
    # dry run: list what would be uploaded
    python hod_mod/scripts/data/upload_zenodo.py --from-registry --dry-run

    # create a fresh draft on the sandbox and upload curated results
    python hod_mod/scripts/data/upload_zenodo.py --sandbox --from-registry

    # add a new version to an existing concept record
    python hod_mod/scripts/data/upload_zenodo.py --new-version 1234567 \
        --extra ~/data/zenodo/LSDR10_GALxEVT

After publishing, regenerate the shipped registry from the live record::

    python hod_mod/scripts/data/make_data_registry.py --from-doi 10.5281/zenodo.<id>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from make_data_registry import _REPO_ROOT, _iter_curated, _iter_extra  # noqa: E402

from hod_mod.data_io.registry import zenodo_key  # noqa: E402

PROD = "https://zenodo.org/api"
SANDBOX = "https://sandbox.zenodo.org/api"

METADATA = {
    "metadata": {
        "upload_type": "dataset",
        "title": "hod_mod: benchmark data and curated fit results",
        "description": (
            "Large inputs and curated benchmark results for the hod_mod HOD "
            "galaxy clustering / weak-lensing package: final posterior chains, "
            "headline figures, and external measurement inputs fetched on "
            "demand via pooch. Regenerable intermediate products are not "
            "included. See https://github.com/JohanComparat/hod_mod"
        ),
        "creators": [{"name": "Comparat, Johan"}],
        "access_right": "open",
        "license": "cc-by-4.0",
    }
}


def _session(token: str):
    import requests

    s = requests.Session()
    # Bearer header (not a ?access_token= query param) so the token never
    # appears in a URL — e.g. in a raise_for_status() error message.
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def _files_to_upload(args) -> list[tuple[str, Path]]:
    if args.from_registry:
        items = list(_iter_curated(_REPO_ROOT))
    else:
        items = []
    items += list(_iter_extra(args.extra))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", action="store_true",
                    help="use sandbox.zenodo.org (testing)")
    ap.add_argument("--from-registry", action="store_true",
                    help="include the curated results (chains/figures/summaries)")
    ap.add_argument("--extra", action="append", default=[], metavar="DIR",
                    help="extra Tier-2 input directory to upload (repeatable)")
    ap.add_argument("--new-version", metavar="RECORD_ID",
                    help="create a new version of an existing deposition")
    ap.add_argument("--deposition", metavar="RECORD_ID",
                    help="upload into an existing draft deposition (for retries)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files without contacting Zenodo")
    args = ap.parse_args()

    files = _files_to_upload(args)
    if not files:
        sys.exit("Nothing selected — pass --from-registry and/or --extra DIR.")
    total = sum(p.stat().st_size for _, p in files)
    print(f"{len(files)} files, {total / 1048576:.1f} MB:")
    for key, path in files:
        print(f"  {key}  ({path.stat().st_size / 1048576:.2f} MB)")
    if args.dry_run:
        print("(dry run — nothing uploaded)")
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set $ZENODO_TOKEN (scope deposit:write).")
    api = SANDBOX if args.sandbox else PROD
    s = _session(token)

    if args.deposition:
        r = s.get(f"{api}/deposit/depositions/{args.deposition}")
        r.raise_for_status()
        draft = r.json()
    elif args.new_version:
        r = s.post(f"{api}/deposit/depositions/{args.new_version}/actions/newversion")
        r.raise_for_status()
        draft = s.get(r.json()["links"]["latest_draft"]).json()
    else:
        r = s.post(f"{api}/deposit/depositions", json={})
        r.raise_for_status()
        draft = r.json()

    dep_id = draft["id"]
    bucket = draft["links"]["bucket"]
    print(f"Draft deposition {dep_id} ({api})")

    # Zenodo file keys cannot contain '/', so flatten nested paths (the logical
    # path round-trips via hod_mod.data_io.registry.zenodo_key on download).
    for key, path in files:
        zkey = zenodo_key(key)
        with open(path, "rb") as fh:
            up = s.put(f"{bucket}/{zkey}", data=fh)
        up.raise_for_status()
        print(f"  uploaded {zkey}")

    s.put(f"{api}/deposit/depositions/{dep_id}", json=METADATA).raise_for_status()
    print(f"\nDraft ready: {draft['links']['html']}")
    print("Review and Publish in the web UI to mint the DOI, then run "
          "make_data_registry.py --from-doi <DOI>.")


if __name__ == "__main__":
    main()
