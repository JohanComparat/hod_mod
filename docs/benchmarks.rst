.. _benchmarks:

HOD Literature Benchmarks — WP only
=====================================

This section documents validation runs comparing **hod_mod** predictions against
published data vectors from the reference papers that introduced each HOD or CSMF
model.  For every benchmark we:

1. Use the paper's own data vector (or the best publicly available proxy).
2. Fit with the same parameter set and same cosmology as the original analysis.
3. Compare best-fit parameters and :math:`\chi^2/\text{dof}` against published values.

Pass criterion: :math:`\chi^2/\text{dof} < 2.0`.  MCMC runs are optional (``--mcmc`` flag).

Running the benchmarks
----------------------

Single model::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model more2015 --plot

All models::

    python hod_mod/scripts/benchmarks/run_all_benchmarks.py

Optional flags:

* ``--mcmc``  — run emcee sampling after the MAP fit (slow, off by default)
* ``--plot``  — save comparison figures to the output directory
* ``--output DIR``  — override the output directory from the config

Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 12 18 20

   * - Model / Paper
     - Survey
     - z\ :sub:`eff`
     - Observable
     - Status
   * - :doc:`MoreHODModel — More+2015 <benchmark_more2015>`
     - BOSS CMASS
     - 0.52
     - w\ :sub:`p`
     - **PASSED** χ²/dof=1.36
   * - :doc:`Kravtsov04HODModel — Kravtsov+2004 <benchmark_kravtsov2004>`
     - BOSS CMASS (same data)
     - 0.52
     - w\ :sub:`p`
     - **PASSED** χ²/dof=1.91
   * - :doc:`HODModel — Zheng+2007 <benchmark_zheng2007>`
     - SDSS DR3
     - 0.1
     - w\ :sub:`p`
     - **PASSED** χ²/dof=1.88 (power-law data)
   * - :doc:`ZuMandelbaum15HODModel — Zu & Mandelbaum+2015 <benchmark_zumandelbaum2015>`
     - SDSS DR7
     - 0.1
     - w\ :sub:`p` + ΔΣ
     - **PASSED** χ²/dof≈0.0; MCMC running
   * - :doc:`Guo18ICSMFModel — Guo+2018 <benchmark_guo2018>`
     - SDSS LOWZ
     - 0.1
     - w\ :sub:`p`
     - **FAILED** χ²/dof=2.89 (threshold vs bin mismatch)
   * - :doc:`Guo19ICSMFModel — Guo+2019 <benchmark_guo2019>`
     - eBOSS ELG
     - 0.80
     - w\ :sub:`p`
     - **FAILED** χ²/dof=2.94 (threshold vs bin mismatch)
   * - :doc:`Leauthaud12HODModel — Leauthaud+2012 <benchmark_leauthaud2012>`
     - COSMOS
     - 0.66
     - w\ :sub:`p` + ΔΣ
     - Needs wp data (no wp in paper)
   * - :doc:`VanUitert16CSMFModel — van Uitert+2016 <benchmark_vanutert2016>`
     - GAMA + KiDS
     - 0.18
     - w\ :sub:`p` + ΔΣ
     - Needs data transcription
   * - :doc:`Zacharegkas25HODModel — Zacharegkas+2025 <benchmark_zacharegkas2025>`
     - DES Y3
     - 0.36
     - w\ :sub:`p` + ΔΣ
     - Needs data transcription
   * - :doc:`Lange25HODModel — Lange+2025 (DESI DR1) <benchmark_lange2025>`
     - DESI BGS+LRG / DES+HSC
     - 0.25–0.70
     - w\ :sub:`p` + ΔΣ + free cosmo
     - Needs data (Zenodo upon publication)

Data transcription instructions for each NEEDS_DATA benchmark are in
``data/<survey_dir>/README_data.md``.

.. seealso::

   :ref:`benchmarks_deltasigma` — ESD-only fits for
   Leauthaud+2012, van Uitert+2016, Zu & Mandelbaum+2015.

   :ref:`benchmarks_joint` — Joint :math:`w_p + \Delta\Sigma`
   fits for Zu & Mandelbaum+2015, Zacharegkas+2025, and Lange+2025.

   :ref:`benchmark_lange2025` — DESI DR1 benchmark with free cosmological parameters
   (Ω_m, S8) and assembly bias (Lange+2025).
