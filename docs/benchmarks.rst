.. _benchmarks:

HOD Literature Benchmarks
==========================

This section documents validation runs comparing **hod_mod** predictions against
published data vectors from the reference papers that introduced each HOD model.
For every benchmark we:

1. Use the paper's own data vector (or the best publicly available proxy).
2. Fit with the same parameter set and same cosmology as the original analysis.
3. Compare best-fit parameters and :math:`\chi^2/\text{dof}` against published values.

Pass criterion: :math:`\chi^2/\text{dof} < 2.0`.  MCMC runs are optional (``--mcmc`` flag).

Running the benchmarks
----------------------

Single model::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model more2015 --plot

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
     - **PASSED** χ²/dof=1.54
   * - :doc:`Lange25HODModel — Lange+2025 (DESI DR1) <benchmark_lange2025>`
     - DESI BGS+LRG / HSC-Y3
     - 0.25–0.70
     - w\ :sub:`p` + ΔΣ + free cosmo
     - In progress (bwpd data)
