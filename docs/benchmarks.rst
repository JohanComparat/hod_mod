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
All fits include the beyond-linear halo bias correction
(:class:`~hod_mod.core.beyond_linear_bias.BeyondLinearBiasMead21`).

.. list-table::
   :header-rows: 1
   :widths: 38 20 12 14 16

   * - Benchmark key
     - Survey / sample
     - Observables
     - :math:`\chi^2/\text{dof}`
     - Status
   * - ``more2015_logM11_12``
     - BOSS CMASS logM*>11.1
     - :math:`w_p + \Delta\Sigma`
     - 1.974
     - **PASSED**
   * - ``more2015_logM11p3_12``
     - BOSS CMASS logM*>11.3
     - :math:`w_p + \Delta\Sigma`
     - 1.646
     - **PASSED**
   * - ``more2015_logM11p4_12``
     - BOSS CMASS logM*>11.4
     - :math:`w_p + \Delta\Sigma`
     - 1.809
     - **PASSED**
   * - ``more2015_logM11_12_freecosmo``
     - BOSS CMASS logM*>11.1
     - :math:`w_p + \Delta\Sigma` + free cosmo
     - 1.713
     - **PASSED**

Zu & Mandelbaum 2015 — SDSS DR7 (iHOD)
---------------------------------------

Full details: :doc:`benchmark_zumandelbaum2015` (threshold sample) and
:doc:`benchmark_zumandelbaum2015_multisample` (per-bin and joint fits).

All variants use the **ZuMandelbaum15HODModel** (inverse-SHMR iHOD) on SDSS DR7.
The threshold sample uses the model-anchored data vector; the per-bin samples use
:math:`w_p(r_p)` and :math:`\Delta\Sigma(R)` digitized from ZM15 Figure 6
(``data/zumandelbaum2015_sdss/``). MAP fits below (no MCMC):

.. list-table::
   :header-rows: 1
   :widths: 36 20 16 14 14

   * - Benchmark key
     - Sample
     - Observables
     - :math:`\chi^2/\text{dof}`
     - Status
   * - ``zumandelbaum2015``
     - threshold M\ :sub:`*`\ >10.2
     - :math:`w_p + \Delta\Sigma`
     - 0.062
     - **PASSED**
   * - ``zumandelbaum2015_ds``
     - threshold M\ :sub:`*`\ >10.2
     - :math:`\Delta\Sigma`
     - 0.208
     - **PASSED**
   * - ``zumandelbaum2015_bin_9p4_9p8``
     - bin [9.4–9.8]
     - :math:`w_p`
     - 1.871
     - **PASSED**
   * - ``zumandelbaum2015_bin_9p8_10p2``
     - bin [9.8–10.2]
     - :math:`w_p`
     - 1.988
     - **PASSED**
   * - ``zumandelbaum2015_bin_10p2_10p6``
     - bin [10.2–10.6]
     - :math:`w_p + \Delta\Sigma`
     - 0.757
     - **PASSED**
   * - ``zumandelbaum2015_bin_10p6_11p0``
     - bin [10.6–11.0]
     - :math:`w_p + \Delta\Sigma`
     - 1.015
     - **PASSED**
   * - ``zumandelbaum2015_bin_11p0_11p2``
     - bin [11.0–11.2]
     - :math:`w_p + \Delta\Sigma`
     - 1.528
     - **PASSED**
   * - ``zumandelbaum2015_bin_11p2_11p4``
     - bin [11.2–11.4]
     - :math:`w_p + \Delta\Sigma`
     - 3.058
     - FAIL
   * - ``zumandelbaum2015_bin_11p4_12p0``
     - bin [11.4–12.0]
     - :math:`w_p + \Delta\Sigma`
     - 1.455
     - **PASSED**

The highest-mass bin [11.2–11.4] exceeds the :math:`\chi^2/\text{dof}<2` criterion;
the others pass. Digitization of Figure 6 (∼15–20% per-point accuracy) is the main
residual source. See :doc:`benchmark_zumandelbaum2015_multisample` for the joint
all-bins fit.

Lange+2025 — DESI DR1
----------------------

Full details: benchmark_lange2025 (page not linked).

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

Comparat+2025 — Galaxy × eROSITA soft X-ray
--------------------------------------------

Full details: benchmark_comparat2025 (page not linked).

Seven stellar-mass-limited galaxy samples from LS DR10 ×
eROSITA eRASS:5 (0.5–2 keV), modelled with the iHOD
(**ZuMandelbaum15HODModel**) from Table 3 and the DPM electron density
profile (Oppenheimer+2025 Model 2) for the hot-gas contribution.
Observable: :math:`w_\theta(\theta)` on scales 9–842 kpc (physical).
Data in ``hod_mod/data/benchmarks/xray/comparat2025_wtheta_S{1..7}.csv``.

.. list-table::
   :header-rows: 1
   :widths: 10 22 18 22 18

   * - Sample
     - :math:`\log_{10}(M_*/M_\odot)`
     - :math:`z_{\rm mean}`
     - Observable
     - Status
   * - S1
     - :math:`> 10.00`
     - 0.136
     - :math:`w_\theta(\theta)`
     - data available
   * - S2
     - :math:`> 10.25`
     - 0.172
     - :math:`w_\theta(\theta)`
     - data available
   * - S3
     - :math:`> 10.50`
     - 0.205
     - :math:`w_\theta(\theta)`
     - data available
   * - S4
     - :math:`> 10.75`
     - 0.243
     - :math:`w_\theta(\theta)`
     - data available
   * - S5
     - :math:`> 11.00`
     - 0.261
     - :math:`w_\theta(\theta)`
     - data available
   * - S6
     - :math:`> 11.25`
     - 0.261
     - :math:`w_\theta(\theta)`
     - data available
   * - S7
     - :math:`> 11.50`
     - 0.261
     - :math:`w_\theta(\theta)`
     - data available

Primary failure drivers for Lange+2025:

1. **Digitized data** — accuracy ∼20–30% per point; replace with Zenodo tables for fair comparison.
2. **Model mismatch** — the analytical halo model cannot replicate the per-halo-concentration
   decorated HOD used in the paper (AbacusSummit N-body + Hearin+2016).
3. **S8/Ω_m prior tension** — MAP is pulled toward Planck values rather than the published
   DESI values; a wider/uniform cosmological prior would be needed.
