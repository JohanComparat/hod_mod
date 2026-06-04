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

More+2015 — BOSS CMASS
-----------------------

Full details: :doc:`benchmark_more2015`.

All five variants use the **MoreHODModel** on BOSS CMASS data (White+2014).
The three mass-threshold variants (logM11_12, logM11p3_12, logM11p4_12) fit
both :math:`w_p(r_p)` and :math:`\Delta\Sigma(R)` jointly.

.. list-table::
   :header-rows: 1
   :widths: 38 20 12 14 16

   * - Benchmark key
     - Survey / sample
     - Observables
     - :math:`\chi^2/\text{dof}`
     - Status
   * - ``more2015``
     - BOSS CMASS
     - :math:`w_p`
     - 1.38
     - **PASSED**
   * - ``more2015_logM11_12``
     - BOSS CMASS logM*>11.1
     - :math:`w_p + \Delta\Sigma`
     - 1.65
     - **PASSED**
   * - ``more2015_logM11p3_12``
     - BOSS CMASS logM*>11.3
     - :math:`w_p + \Delta\Sigma`
     - 1.57
     - **PASSED**
   * - ``more2015_logM11p4_12``
     - BOSS CMASS logM*>11.4
     - :math:`w_p + \Delta\Sigma`
     - 1.73
     - **PASSED**
   * - ``more2015_logM11_12_freecosmo``
     - BOSS CMASS logM*>11.1
     - :math:`w_p + \Delta\Sigma` + free cosmo
     - 1.62
     - **PASSED**

Lange+2025 — DESI DR1
----------------------

Full details: :doc:`benchmark_lange2025`.

All variants use the **Lange25HODModel** on DESI DR1 galaxy clustering (BGS, LRG)
with HSC-Y3 weak lensing.  Data are manually digitized from Figures 3–4 of
Lange+2025 (arXiv:2512.15962; accuracy ∼20–30%); replace with Zenodo tables
(record 17831718) for quantitative comparisons.

.. note::

   ESD-only fits report negative ndof (more free parameters than data points after
   scale cuts), so :math:`\chi^2/\text{dof}` is undefined.  These variants are
   listed as FAIL regardless.

.. list-table::
   :header-rows: 1
   :widths: 38 12 14 14 16 12

   * - Benchmark key
     - Sample
     - z\ :sub:`eff`
     - Observables
     - :math:`\chi^2/\text{dof}`
     - Status
   * - ``lange2025_bgs2_bwpd_wp``
     - BGS2
     - 0.25
     - :math:`w_p`
     - 9.25
     - FAIL
   * - ``lange2025_bgs2_bwpd_esd``
     - BGS2
     - 0.25
     - :math:`\Delta\Sigma`
     - — (ndof<0)
     - FAIL
   * - ``lange2025_bgs2_bwpd_hsc``
     - BGS2
     - 0.25
     - :math:`w_p + \Delta\Sigma`
     - 2.20
     - FAIL
   * - ``lange2025_bgs3_bwpd_wp``
     - BGS3
     - 0.35
     - :math:`w_p`
     - 13.54
     - FAIL
   * - ``lange2025_bgs3_bwpd_esd``
     - BGS3
     - 0.35
     - :math:`\Delta\Sigma`
     - — (ndof<0)
     - FAIL
   * - ``lange2025_bgs3_bwpd_hsc``
     - BGS3
     - 0.35
     - :math:`w_p + \Delta\Sigma`
     - 1.64
     - **PASSED**
   * - ``lange2025_lrg1_bwpd_wp``
     - LRG1
     - 0.51
     - :math:`w_p`
     - 15.96
     - FAIL
   * - ``lange2025_lrg1_bwpd_esd``
     - LRG1
     - 0.51
     - :math:`\Delta\Sigma`
     - — (ndof<0)
     - FAIL
   * - ``lange2025_lrg1_bwpd_hsc``
     - LRG1
     - 0.51
     - :math:`w_p + \Delta\Sigma`
     - 2.96
     - FAIL
   * - ``lange2025_lrg2_bwpd_wp``
     - LRG2
     - 0.70
     - :math:`w_p`
     - 24.73
     - FAIL
   * - ``lange2025_lrg2_bwpd_esd``
     - LRG2
     - 0.70
     - :math:`\Delta\Sigma`
     - — (ndof<0)
     - FAIL
   * - ``lange2025_lrg2_bwpd_hsc``
     - LRG2
     - 0.70
     - :math:`w_p + \Delta\Sigma`
     - 2.44
     - FAIL
   * - ``lange2025_all_samples_hsc_wp``
     - all
     - —
     - :math:`w_p`
     - 2.12
     - —

Primary failure drivers for Lange+2025:

1. **Digitized data** — accuracy ∼20–30% per point; replace with Zenodo tables for fair comparison.
2. **Model mismatch** — the analytical halo model cannot replicate the per-halo-concentration
   decorated HOD used in the paper (AbacusSummit N-body + Hearin+2016).
3. **S8/Ω_m prior tension** — MAP is pulled toward Planck values rather than the published
   DESI values; a wider/uniform cosmological prior would be needed.
