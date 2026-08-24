HPC re-run campaigns (OAR / GRICAD)
====================================

Every production number in these docs — fit posteriors, benchmark χ²/dof,
forecast ellipses, the figures under ``docs/_images/`` — is regenerated in
batch as an **OAR array-job campaign** on the GRICAD ``dahu`` cluster, driven
from the `oarsub/ <https://github.com/JohanComparat/hod_mod/tree/main/oarsub>`_
directory.  This page documents the campaign machinery; cluster-account
specifics (projects, tokens, monitoring) live in ``oarsub/README.md``.

Why campaigns
-------------

Three behaviour-changing releases have forced full re-runs:

* **v0.3** — the 0.3.0 Hankel-transform fix in ``_pk_to_xi`` moved every
  real-space observable (:math:`w_p` ≤ 19 %, :math:`\Delta\Sigma` ≤ 20 %,
  :math:`\Sigma_y` ~16 %), superseding all ≤ 0.2.3 results.
* **v0.31** — 0.3.1 swapped the default linear P(k) from CAMB to the
  CosmoPower-JAX emulator (~+2.5 % in P(k) amplitude, ≈1.3 % in
  :math:`\sigma_8`; see :doc:`cosmology`).
* **v0.4** — 0.4.0 corrected the full-joint gas sector, whose
  full-covariance prior was never applied and whose seed sat clipped on three
  bounds.  Every v0.3/v0.31 full-joint gas posterior is withdrawn rather than
  shifted, so no pin reproduces the old numbers.

Because ~47 job lines write through :func:`hod_mod.paths.results_root`, each
campaign gets its **own versioned results tree** (``…/hod_mod_results`` for
v0.3, ``…/hod_mod_results_v0.31`` for v0.31) so campaigns never overwrite each
other and a before/after diff is always possible.  Keeping the two trees also
isolates the P(k) swap: v0.3 vs v0.31 differ *only* in the backend.

Anatomy
-------

``oarsub/_campaign_env.sh``
   Sourced by every job and helper.  Resolves ``VTAG`` (default ``v0.31``),
   derives and exports ``HOD_MOD_RESULTS`` (the versioned tree),
   ``HOD_MOD_DATA_DIR``, ``HOD_MOD_SUMSTAT``, and pins ``HOD_MOD_PK_BACKEND``
   per campaign (``v0.3 → camb``, otherwise ``cosmopower``).  It also seeds
   each fresh tree with the fixed *reference inputs* (e.g. the ZM15 MAP used
   by ``--fix-zm15``) that are inputs to the campaign, not outputs of it.

   .. note::

      The backend pin reaches **Families A–C only** (they route through
      :func:`~hod_mod.core.power_spectrum.default_pk_linear`).  Family-D
      forecasts build ``ForwardModel``, which selects its P(k) correction via
      driver flags (real CAMB is not JAX-traceable), so a ``v0.3`` Family-D
      re-run still uses the emulator regardless of the variable.

``oarsub/run_job.sh`` + ``oarsub/params/*.txt``
   The generic OAR array-job wrapper and one param file per family; each line
   is a complete ``python -m …`` command that the wrapper ``eval``\ s on the
   node with the campaign environment in place.

``oarsub/submit_campaign.sh``
   One-command family submission, with ``--devel`` running just the first
   param line on the 30-minute dev partition as an end-to-end smoke test:

   .. code-block:: bash

      ./oarsub/submit_campaign.sh your-oar-project benchmarks_map --devel  # smoke
      ./oarsub/submit_campaign.sh your-oar-project all                     # everything
      VTAG=v0.3 ./oarsub/submit_campaign.sh your-oar-project production    # pinned re-run

   ============================  ==========================================  =========================
   Family                        What                                        Sizing (per array line)
   ============================  ==========================================  =========================
   A ``benchmarks_map``          literature benchmarks, MAP                  8 cores, 2 h
   A ``benchmarks_mcmc``         literature benchmarks, MCMC                 8 cores, 8 h
   B ``production``              BGS ZM15 / thresh / lsdr10 joint MCMC       16 cores, 24 h, resumable
   B ``full_joint``              BGS full-model joint (fixedzm15+allparams)  dedicated scripts
   C ``comparat2025``            Comparat+2025 X-ray w_θ MAP presets (×5)    16 cores, 4–18 h
   D ``forecasts``               tier2/3/4, stage4, sensitivity, MAP+MCMC    16 cores, 24 h
   ============================  ==========================================  =========================

Lifecycle
---------

.. code-block:: bash

   # on dahu (repo at the campaign commit, data staged):
   ./oarsub/submit_campaign.sh <PROJECT> <family> --devel   # 1-line smoke
   ./oarsub/submit_campaign.sh <PROJECT> <family>           # real submission
   oarstat -u $USER                                         # monitor

   # from the workstation, when the chains land:
   ./oarsub/pull_results.sh --go          # rsync the versioned trees back
   ./oarsub/campaign_status.sh            # preflight: every out-dir present?
   ./oarsub/collect_and_plot.sh           # refresh docs/_images/ figures

``campaign_status.sh`` audits the pulled tree *before* plotting —
``collect_and_plot.sh`` runs under ``set -euo pipefail`` and would abort
half-way (leaving ``docs/_images/`` partially refreshed) on a single missing
out-dir.  All four helpers agree on the same default ``VTAG``; override with
``VTAG=v0.3 …`` to operate on the CAMB-era tree.

VTAG semantics
--------------

=========  ==============================  ===========================  =============================
VTAG       Physics                         ``HOD_MOD_PK_BACKEND`` pin   Results tree
=========  ==============================  ===========================  =============================
``v0.3``   Hankel fix, CAMB P(k)           ``camb``                     ``…/hod_mod_results``
``v0.31``  + CosmoPower-JAX default P(k)   ``cosmopower``               ``…/hod_mod_results_v0.31``
``v0.4``   + 0.4.0 gas prior/seed fix      ``cosmopower``               ``…/hod_mod_results_v0.4``
=========  ==============================  ===========================  =============================

The pin is derived from ``VTAG`` *inside the job* rather than exported by the
caller because OAR does not propagate the submitting shell's environment to
the node — an ``HOD_MOD_PK_BACKEND=camb oarsub …`` would be silently dropped.

Closed campaigns: what v0.3 and v0.31 never produced
----------------------------------------------------

*Status 2026-08-23.*  Both campaigns are **closed as-is**.  Four production
chains never reached their step budget and one Family-D artifact never landed.
None will be resumed: 0.4.0 supersedes the gas sector of the two ``full_joint``
runs among them, so the re-run belongs to the ``v0.4`` campaign instead.

.. list-table:: Outstanding at close
   :header-rows: 1
   :widths: 32 20 20 28

   * - Artifact
     - v0.3
     - v0.31
     - Disposition
   * - ``bgs_full_joint_allparams_<VTAG>``
     - 1061/4000 steps
     - 889/4000 steps
     - superseded — gas sector invalid (0.4.0)
   * - ``bgs_zm15_thresh_joint_<VTAG>``
     - 1909/2500 steps
     - 1071/2500 steps
     - re-run under ``v0.4``
   * - ``fits/benchmark``
     - 2026-07-12, pre-campaign
     - absent
     - re-run under ``v0.4`` (``forecasts.txt``)

The chains are truncated, not corrupted — ``log_prob[:iteration]`` is finite
throughout.  The two ``allparams`` runs died after 11-12 h of wall clock at
74-98 steps/h, against the 41-54 h that 4000 steps needed; the ``zm15_thresh``
pair additionally hit the ensemble collapse that
:mod:`hod_mod.fitting.mcmc_resume` was later written for.

Everything else landed in both trees, including all **31/31** requested
literature benchmarks.  (The v0.3 ``benchmarks/`` tree has five extra top-level
directories -- one deprecated run, an archived ``version0`` snapshot and three
figure-only directories -- which are not models; the 25-vs-20 directory count is
a misleading proxy for completeness.)

``docs/_images/`` carries the **v0.3** figure set at close: the stamp reads
``v0.3``, with Family D taken from the v0.31 tree.  v0.31 was never collected,
deliberately -- the two campaigns do not produce the same *set* of figures, and
the tree is kept as the numeric before/after for the P(k) swap alone.

v0.4 campaign status
--------------------

*Status 2026-08-24 09:20, campaign in flight.*  52 job lines submitted from
**v0.5.0** (``commit=d6a0e62``) into ``~/data/hod_mod_results_v0.4``, plus a
five-job re-run of the ``--ecf`` family from ``commit=17c9026`` once the anchor
bug below was fixed.

**Landed (45 of 57):** ``bgs_full_joint_fixedzm15`` — the fit 0.4.0 exists for,
and the one that carries the acceptance test — ``bgs_comparat2025``, all ten
Family-C presets, the **re-run** ``--ecf`` five, 25 of 31 literature benchmarks,
and the tier2/tier3/tier4 forecasts.

**Still running (5):** ``bgs_zm15_joint_wp_ngal`` (313/2500 steps),
``bgs_zm15_thresh_joint`` (391/2500), ``bgs_full_joint_allparams`` (1561/4000),
``sensitivity_fisher`` and ``stage4_forecast``.  ``fits/benchmark`` and six
Zu–Mandelbaum MCMC bins are queued behind them.

The three MCMC production fits are the long pole and are the reason the campaign
is not finished: at the step rates observed in v0.3 they need 24–50 h each, and
they restarted at least once on besteffort.

.. warning::

   ``docs/_images/`` is stamped ``v0.4`` but is currently a **mix**.
   ``collect_and_plot.sh`` was run with ``ALLOW_PARTIAL=1``, so it refreshed the
   87 figures whose jobs have landed and skipped the rest — which therefore
   still show their **v0.3** renderings.  The stamp records the campaign that
   *collected*, not that every figure came from it.  Re-collect without
   ``ALLOW_PARTIAL`` once ``campaign_status.sh`` reports a clean tree.

   Family-C figures are unaffected either way: those fits emit **PDF**, and
   section C of ``collect_and_plot.sh`` copies ``*.png``, so it has never
   collected anything from them.

.. admonition:: Provenance incident, 2026-08-23 — check the commit, not the tree name
   :class: warning

   The first v0.4 submission ran on the **wrong code** and had to be killed and
   redone.  ``dahu`` was still at ``ce17a1c``, the pre-0.4.0 ``main`` tip,
   missing the gas prior/seed fix that is the campaign's entire reason to
   exist — so ``bgs_full_joint_fixedzm15`` was 909 steps into reproducing the
   very posterior 0.4.0 withdrew.

   Nothing about the *tree* revealed this.  ``VTAG=v0.4`` was passed correctly
   and the old ``_campaign_env.sh`` still routed it to
   ``…/hod_mod_results_v0.4`` with the CosmoPower pin, so v0.3-era physics was
   landing in a directory named for v0.4.  What exposed it was the **job
   name**: ``hodmod_v03_rerun`` where 0.4.1's submitter writes
   ``hodmod_v04_<family>``.  That per-campaign log stem was added for tidiness
   and earned its keep on its first outing.

   The rule this leaves: after ``pull_results.sh``, read ``commit=`` out of a
   job log header before trusting anything in the tree.  ``run_job.sh`` has
   logged it since 785c097 precisely so that a mislabelled campaign is
   detectable rather than merely unlikely.

What a "v0.3 figure" is, and what it is not
--------------------------------------------

Pages refreshed from the ``VTAG=v0.3`` campaign carry a provenance note pointing
here.  Three caveats, stated once:

* **The package version is not the campaign version.**  ``conf.py`` takes
  ``release`` from ``pyproject.toml``, whose *default* linear
  :math:`P(k)` is the CosmoPower-JAX emulator.  The v0.3 figures were produced
  with ``HOD_MOD_PK_BACKEND=camb``, pinned inside the job from ``VTAG``.  0.3.1
  in the sidebar and CAMB in the figures is deliberate, not a mismatch.

  The same holds for ``v0.4``, which runs from the **0.5.0** release.  ``VTAG``
  names the results tree and the physics generation — here 0.4.0's full-joint
  gas prior/seed correction — not the package version that happens to be
  current when the jobs are submitted.  0.5.0 adds CCL-parity work on top; none
  of it moves a campaign number, because its one behaviour change reaches only
  the ``sheth99`` / ``bhattacharya11`` bias pairings and every job line runs
  ``tinker08``.

* **The pin reaches Families A–C only.**  Family-D forecasts
  (:doc:`sensitivity_fisher`, :doc:`stage4_forecast`, :doc:`tier2_forecast`,
  :doc:`tier3_forecast`, :doc:`tier4_forecast`, :doc:`benchmark_map_mcmc`) build
  :class:`~hod_mod.forecast.forward_jax.ForwardModel`, which chooses its
  :math:`P(k)` correction from a driver flag and defaults to the emulator —
  real CAMB is not JAX-traceable.  Those pages are campaign-fresh but
  **emulator-based**, and a ``VTAG=v0.3`` re-run would not make them CAMB
  products.

* **Anything dated before 2026-07-16 is pre-campaign**, hence pre-Hankel-fix,
  hence superseded.  Such pages carry a dated *Pending v0.3 re-run* admonition
  naming the artifact they wait on.  ``campaign_status.sh`` finds them:
  ``SINCE=2026-07-16`` makes it reject an artifact that merely *exists*, which
  is how the three tier forecasts were caught still quoting their 2026-07-02/04
  runs after a job that had refreshed only their Jacobian cache.

Checking a campaign before trusting its figures
------------------------------------------------

.. code-block:: bash

   # what landed, per artifact rather than per directory
   SINCE=2026-07-16 VTAG=v0.3 HOD_MOD_RESULTS=~/data/hod_mod_results \
       ./oarsub/campaign_status.sh

   # refresh only what is complete; the rest is skipped with a reason
   SINCE=2026-07-16 VTAG=v0.3 HOD_MOD_RESULTS=~/data/hod_mod_results \
       ALLOW_PARTIAL=1 ./oarsub/collect_and_plot.sh

``collect_and_plot.sh`` stamps ``docs/_images/.campaign_vtag`` with the campaign
it collected and refuses to collect a different one on top without
``FORCE_VTAG_SWITCH=1``: the two campaigns do not produce the same *set* of
figures, so collecting one over the other leaves a silent mix.
