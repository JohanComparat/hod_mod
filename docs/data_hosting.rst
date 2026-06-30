.. _data_hosting:

Data hosting and on-demand fetching
===================================

GitHub is a poor host for this package's heavy data and benchmark products
(no checksums, no DOI, restrictive LFS quotas, and history bloat — every clone
pays for tracked binaries). Instead, large files live on a **Zenodo** data
record and are pulled on demand with :func:`hod_mod.data_io.fetch`, which
verifies a SHA256 checksum and caches the file locally.

Data tiers
----------

.. list-table::
   :header-rows: 1
   :widths: 18 44 38

   * - Tier
     - What
     - Where it lives
   * - 1 — in git
     - Tiny inputs needed to *import/run* the code (AGN K-correction tables,
       BNL tables, the eROSITA response ``.npz``, digitized benchmark CSVs)
     - shipped in the wheel via ``package-data``
   * - 2 — Zenodo
     - Large external inputs (LSDR10 GALxEVT cross-correlation measurements,
       eROSITA CALDB PSF/response files)
     - Zenodo data record, fetched via :func:`hod_mod.data_io.fetch`
   * - 3 — Zenodo (curated)
     - Final posterior chains (``flatchain.npz`` / ``chain.h5``), run summaries
       and headline figures
     - Zenodo data record; the 60k+ intermediate ``.npz`` are **not** archived
       (regenerable from the chains + a committed config)

Where generated results go
--------------------------

New outputs (MCMC chains, figures, caches) are **never written inside the repo**.
:func:`hod_mod.paths.results_root` resolves a writable location off the source tree:

* ``$HOD_MOD_RESULTS`` if set (point runs at a project/scratch disk), else
* the per-user OS data dir — ``~/.local/share/hod_mod/results`` on Linux.

.. code-block:: python

   from hod_mod.paths import results_root, results_path
   out = results_path("benchmarks", "my_run", "flatchain.npz")  # parent dir created

All filesystem locations go through :mod:`hod_mod.paths` (no hardcoded paths in
the code); each helper reads an env var and falls back to a sensible default:

============================ ====================== ================================================
Env var                      Helper                 Points to
============================ ====================== ================================================
``HOD_MOD_REPO``             ``repo_root()``        code repo (``configs/``, in-repo ``data/``)
``HOD_MOD_DATA_DIR``         ``data_root()``        data repo (zenodo/erosita/legacysurvey/bands)
``HOD_MOD_SUMSTAT``          ``sum_stat_root()``    ``sum_stat`` measurement products
``HOD_MOD_RESULTS``          ``results_root()``     generated outputs (never in the repo)
``HOD_MOD_CACHE``            ``cache_root()``       JAX/XLA compilation caches
============================ ====================== ================================================

Set them once in ``~/.bashrc`` so every run and shell agrees::

   export HOD_MOD_REPO="$HOME/software/hod_mod"
   export HOD_MOD_DATA_DIR="$HOME/data"
   export HOD_MOD_SUMSTAT="$HOME/software/sum_stat/data"
   export HOD_MOD_RESULTS="$HOME/data/hod_mod_results"

Fetching data
-------------

.. code-block:: python

   from hod_mod.data_io import fetch

   # downloads from Zenodo + verifies SHA256 on first call, cache hit after
   path = fetch("results/benchmarks/more2015_logM11_12/flatchain.npz")

Resolution order:

1. ``$HOD_MOD_DATA_DIR/<name>`` — if set and the file exists, it is returned
   directly with no download (point this at an existing local mirror).
2. The pooch cache (``$HOD_MOD_DATA_CACHE`` or the OS cache dir), downloading
   from Zenodo and checking the SHA256 from the shipped ``registry.txt``.

Configuration (environment variables):

``HOD_MOD_DATA_DOI``
   Zenodo DOI of the data record. Use the **concept** DOI to always track the
   latest version, or a **version** DOI to pin a snapshot for reproducibility.

``HOD_MOD_DATA_BASEURL``
   Override the base URL entirely (a ``file://`` mirror, or a non-Zenodo object
   store). Takes precedence over the DOI.

``HOD_MOD_DATA_DIR``
   Local mirror checked before any download.

``HOD_MOD_DATA_CACHE``
   Override the pooch cache directory.

Publishing / refreshing the data record
----------------------------------------

The deposit is created and curated by hand (the publish step mints the DOI and
must stay deliberate), with two helper scripts to do the heavy lifting:

#. **Preview / stage the registry** from local curated files::

      python hod_mod/scripts/data/make_data_registry.py --from-local --dry-run

#. **Upload a draft** to Zenodo (never auto-publishes; set ``$ZENODO_TOKEN``)::

      python hod_mod/scripts/data/upload_zenodo.py --sandbox --from-registry
      python hod_mod/scripts/data/upload_zenodo.py --from-registry \
          --extra ~/data/zenodo/LSDR10_GALxEVT

   Review the draft in the Zenodo web UI and click **Publish**.

#. **Regenerate the authoritative registry** from the published record (pulls
   the exact filenames + checksums Zenodo stored)::

      python hod_mod/scripts/data/make_data_registry.py --from-doi 10.5281/zenodo.<id>

   Then set ``DEFAULT_DATA_DOI`` in :mod:`hod_mod.data_io.registry` (or rely on
   ``$HOD_MOD_DATA_DOI``) and commit the updated ``registry.txt``.

Two records, not one
--------------------

* A **software** record — enable Zenodo's GitHub integration so each GitHub
  *release* auto-mints a versioned DOI for the code + Tier-1 data.
* A **data** record — the Tier-2 inputs and Tier-3 curated results, kept
  separate so 150+ MB is not re-uploaded on every code release. This is the
  record :func:`hod_mod.data_io.fetch` points at.

Why Zenodo
----------

Free, CERN-backed, permanent storage with a DOI per version (50 GB/record,
more on request) and GitHub-release integration. Alternatives considered:
CDS/VizieR (best for *tabular* data tied to a journal article); OSF / figshare
/ Dataverse (comparable DOI hosts); Git LFS (no DOI, tight quotas, does not
solve regenerable bulk); commercial object storage such as S3/B2/GCS (great for
very large or fast-changing data via the ``HOD_MOD_DATA_BASEURL`` override, but
no DOI by itself).
