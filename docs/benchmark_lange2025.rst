.. _benchmark_lange2025:

Lange+2025 — DESI DR1 Tracer Analysis
=======================================

**Paper:** Lange et al. 2025, arXiv:2512.15962

**Survey:** DESI DR1 galaxy clustering (BGS, LRG) + DES Y3/KiDS-1000/HSC-Y3 weak lensing

**Key result:** S8 = 0.794 ± 0.023 (DES/KiDS) | 0.793 ± 0.017 (HSC), Ω_m = 0.295 ± 0.012

Data repository (upon publication): https://zenodo.org/records/17831718

Overview
--------

The DESI DR1 tracer analysis fits a "decorated HOD" with assembly bias to four galaxy
samples spanning z = 0.2–0.8:

.. list-table::
   :header-rows: 1
   :widths: 10 12 10 20 12 12

   * - Sample
     - z range
     - z\ :sub:`eff`
     - n\ :sub:`g` [h³ Mpc⁻³]
     - DES/KiDS ESD
     - HSC-Y3 ESD
   * - BGS2
     - 0.2–0.3
     - 0.25
     - 3.71 × 10\ :sup:`−3`
     - Fig 3
     - Fig 4
   * - BGS3
     - 0.3–0.4
     - 0.35
     - 1.38 × 10\ :sup:`−3`
     - Fig 3
     - Fig 4
   * - LRG1
     - 0.4–0.6
     - 0.51
     - 5.70 × 10\ :sup:`−4`
     - Fig 3
     - Fig 4
   * - LRG2
     - 0.6–0.8
     - 0.70
     - 5.69 × 10\ :sup:`−4`
     - —
     - Fig 4

Number densities for BGS from Hahn+2023 (AJ 165, 253); for LRG from Zhou+2023 (AJ 165, 58).

Observables
-----------

- **wp(rp)**: projected clustering, π\ :sub:`max` = 80 h\ :sup:`−1` Mpc,
  11 bins in rp ∈ [0.4, 63] h\ :sup:`−1` Mpc.
- **ΔΣ(R)**: excess surface density (weak lensing),
  13 bins in R ∈ [2.5, 40] h\ :sup:`−1` Mpc.
  Two surveys: DES Y3/KiDS-1000 (Figure 3) and HSC-Y3 (Figure 4).

HOD model: ``Lange25HODModel``
-------------------------------

hod_mod implements the Lange+2025 HOD as :class:`~hod_mod.galaxies.hod.Lange25HODModel`:

- **Base occupation**: Zheng+2007 centrals + Kravtsov+2004 satellites
- **f_Gamma** ∈ [0.5, 1.0]: central galaxy completeness
- **A_cen**, **A_sat** ∈ [−1, 1]: effective assembly bias amplitudes

The assembly bias is implemented analytically via a ``(b−1)/b`` kernel on the
effective galaxy bias (2-halo term). This is an approximation to the N-body
decorated HOD used in the paper; quantitative agreement is limited.

.. note::

   The Lange+2025 paper uses AbacusSummit N-body simulations with a fully
   stochastic decorated HOD (Hearin+2016). The hod_mod analytical halo model
   cannot reproduce per-halo-concentration occupation — instead, A_cen and A_sat
   shift the effective galaxy bias via the (b−1)/b assembly bias kernel.
   Parameter recovery (especially S8 and Ω_m) is the primary validation target.

Free cosmology
--------------

The benchmark frees Ω_m and S8 = σ₈ √(Ω_m/0.3) with Planck 2018 Gaussian priors:

.. math::

   \Omega_m &= 0.3153 \pm 0.0073  \quad [\text{5}\sigma \text{ bounds: } 0.279, 0.352] \\
   S_8 &= 0.832 \pm 0.0114         \quad [\text{5}\sigma \text{ bounds: } 0.775, 0.889]

The S8 → σ₈ conversion uses σ₈ = S8 √(0.3/Ω_m); the power spectrum amplitude
is updated via ln10As ≈ ln10As_fid + 2 ln(σ₈/σ₈_fid) (fast, avoids CAMB per step).

Running the benchmarks
----------------------

.. code-block:: bash

   # MAP fit: joint wp+ESD, BGS2 + HSC-Y3 lensing
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_bwpd_hsc --plot

   # MAP fit: wp-only, BGS2 (with free cosmology)
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_bwpd_wp --plot

   # MAP fit: ESD-only, BGS2 + HSC
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_bwpd_esd --plot

   # Full MCMC (slow — free cosmology, 64 walkers × 4000 steps)
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_bwpd_hsc --mcmc --plot

Available model keys (bwpd series):

* wp-only: ``lange2025_bgs2_bwpd_wp``, ``lange2025_bgs3_bwpd_wp``,
  ``lange2025_lrg1_bwpd_wp``, ``lange2025_lrg2_bwpd_wp``
* ESD-only (HSC-Y3): ``lange2025_bgs2_bwpd_esd``, ``lange2025_bgs3_bwpd_esd``,
  ``lange2025_lrg1_bwpd_esd``, ``lange2025_lrg2_bwpd_esd``
* Joint wp+ESD (HSC-Y3): ``lange2025_bgs2_bwpd_hsc``, ``lange2025_bgs3_bwpd_hsc``,
  ``lange2025_lrg1_bwpd_hsc``, ``lange2025_lrg2_bwpd_hsc``

Data status
-----------

.. note::

   Data manually digitized with WebPlotDigitizer from Figures 3–4 of Lange+2025
   (arXiv:2512.15962).  Files are in **bwpd format** (rp, rp×wp upper-envelope; R, R×ΔΣ
   upper-envelope) and stored in ``data/lange2025_desi_dr1/<SAMPLE>/``:

   * ``wp_<sample>_bwpd.csv`` — projected clustering
   * ``ds_hsc_<sample>_bwpd.csv`` — HSC-Y3 excess surface density

   DES Y3/KiDS-1000 ESD is excluded from this release (data not yet transcribed).
   Replace with the published tables from Zenodo record 17831718 when available
   (currently under review) and update CSV headers accordingly.

Published results (Table 3, combined all samples)
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Lensing
     - S8
     - Ω_m
     - σ₈
   * - DES Y3 + KiDS-1000
     - 0.794 ± 0.023
     - 0.295 ± 0.012
     - —
   * - HSC-Y3
     - 0.793 ± 0.017
     - 0.303 ± 0.010
     - —

Results
-------

MAP fit summary for all Lange+2025 benchmarks.  All fits used Powell optimizer
(≥ 5 free parameters), 50 000 iteration limit, :math:`\delta x < 10^{-5}` tolerance.
Data are digitized from Figures 3–4 of Lange+2025 (accuracy ∼20–30%); replace with
Zenodo data when available to obtain quantitative chi-squared comparisons.

.. note::

   Assembly bias parameters ``A_cen`` and ``A_sat`` converge to identical values
   (≈ −0.473) across all benchmarks, indicating a degenerate direction in the
   likelihood surface.  The analytical (b−1)/b assembly bias kernel gives equal
   contributions from central and satellite assembly bias, so the data cannot
   distinguish them from wp or ΔΣ alone.  They are effectively a single combined
   amplitude in MAP fits.

wp-only — BGS2 with free cosmology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 46.77 / 1` — **FAIL**

Free parameters: 8 HOD + Ω_m + S8 = 10 total; 11 data points.

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.819
     - 0.794
   * - Ω_m
     - 0.323
     - 0.295
   * - ``log10mmin``
     - 12.043
     - —
   * - ``log10m1``
     - 12.830
     - —
   * - ``alpha``
     - 0.500
     - —
   * - ``A_cen`` = ``A_sat``
     - −0.473
     - —

.. figure:: ../results/benchmarks/lange2025/bgs2_wp/benchmark_lange2025_bgs2_wp_wp.png
   :width: 80%
   :alt: BGS2 wp MAP fit

   MAP best-fit :math:`w_p(r_p)` vs BGS2 data (top) and residuals (bottom).

Joint wp+ESD — BGS2 × DES/KiDS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 393.7 / 15 = 26.2` — **FAIL**

Free parameters: 8 HOD + Ω_m + S8 = 10 total; 11 wp + 13 ESD + 1 ng = 25 data.

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.833
     - 0.794
   * - Ω_m
     - 0.345
     - 0.295
   * - ``log10mmin``
     - 12.283
     - —
   * - ``log10m1``
     - 13.097
     - —
   * - ``alpha``
     - 1.223
     - —

.. figure:: ../results/benchmarks/lange2025/bgs2_des/benchmark_lange2025_bgs2_des_wp.png
   :width: 80%
   :alt: BGS2×DES wp MAP fit

   MAP :math:`w_p(r_p)` vs BGS2 data.

.. figure:: ../results/benchmarks/lange2025/bgs2_des/benchmark_lange2025_bgs2_des_ds.png
   :width: 80%
   :alt: BGS2×DES ESD MAP fit

   MAP :math:`\Delta\Sigma(R)` vs BGS2 × DES/KiDS data.

Joint wp+ESD — BGS2 × HSC-Y3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 270.8 / 15 = 18.1` — **FAIL**

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.839
     - 0.793
   * - Ω_m
     - 0.349
     - 0.303
   * - ``log10mmin``
     - 12.388
     - —
   * - ``log10m1``
     - 13.097
     - —
   * - ``alpha``
     - 1.148
     - —

.. figure:: ../results/benchmarks/lange2025/bgs2_hsc/benchmark_lange2025_bgs2_hsc_wp.png
   :width: 80%
   :alt: BGS2×HSC wp MAP fit

   MAP :math:`w_p(r_p)` vs BGS2 data.

.. figure:: ../results/benchmarks/lange2025/bgs2_hsc/benchmark_lange2025_bgs2_hsc_ds.png
   :width: 80%
   :alt: BGS2×HSC ESD MAP fit

   MAP :math:`\Delta\Sigma(R)` vs BGS2 × HSC-Y3 data.

Joint wp+ESD — BGS3 × DES/KiDS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 421.4 / 15 = 28.1` — **FAIL**

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.835
     - 0.794
   * - Ω_m
     - 0.342
     - 0.295
   * - ``log10mmin``
     - 12.758
     - —
   * - ``log10m1``
     - 13.241
     - —

.. figure:: ../results/benchmarks/lange2025/bgs3_des/benchmark_lange2025_bgs3_des_wp.png
   :width: 80%
   :alt: BGS3×DES wp MAP fit

   MAP :math:`w_p(r_p)` vs BGS3 data.

.. figure:: ../results/benchmarks/lange2025/bgs3_des/benchmark_lange2025_bgs3_des_ds.png
   :width: 80%
   :alt: BGS3×DES ESD MAP fit

   MAP :math:`\Delta\Sigma(R)` vs BGS3 × DES/KiDS data.

Joint wp+ESD — BGS3 × HSC-Y3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 350.5 / 15 = 23.4` — **FAIL**

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.836
     - 0.793
   * - Ω_m
     - 0.346
     - 0.303
   * - ``log10mmin``
     - 12.763
     - —
   * - ``log10m1``
     - 13.267
     - —

.. figure:: ../results/benchmarks/lange2025/bgs3_hsc/benchmark_lange2025_bgs3_hsc_wp.png
   :width: 80%
   :alt: BGS3×HSC wp MAP fit

   MAP :math:`w_p(r_p)` vs BGS3 data.

.. figure:: ../results/benchmarks/lange2025/bgs3_hsc/benchmark_lange2025_bgs3_hsc_ds.png
   :width: 80%
   :alt: BGS3×HSC ESD MAP fit

   MAP :math:`\Delta\Sigma(R)` vs BGS3 × HSC-Y3 data.

Joint wp+ESD — LRG1 × DES/KiDS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 633.0 / 15 = 42.2` — **FAIL**

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.837
     - 0.794
   * - Ω_m
     - 0.321
     - 0.295
   * - ``log10mmin``
     - 13.001
     - —
   * - ``log10m1``
     - 13.865
     - —

.. figure:: ../results/benchmarks/lange2025/lrg1_des/benchmark_lange2025_lrg1_des_wp.png
   :width: 80%
   :alt: LRG1×DES wp MAP fit

   MAP :math:`w_p(r_p)` vs LRG1 data.

.. figure:: ../results/benchmarks/lange2025/lrg1_des/benchmark_lange2025_lrg1_des_ds.png
   :width: 80%
   :alt: LRG1×DES ESD MAP fit

   MAP :math:`\Delta\Sigma(R)` vs LRG1 × DES/KiDS data.

Joint wp+ESD — LRG1 × HSC-Y3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 618.7 / 15 = 41.2` — **FAIL**

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.838
     - 0.793
   * - Ω_m
     - 0.322
     - 0.303
   * - ``log10mmin``
     - 13.089
     - —
   * - ``log10m1``
     - 13.845
     - —

.. figure:: ../results/benchmarks/lange2025/lrg1_hsc/benchmark_lange2025_lrg1_hsc_wp.png
   :width: 80%
   :alt: LRG1×HSC wp MAP fit

   MAP :math:`w_p(r_p)` vs LRG1 data.

.. figure:: ../results/benchmarks/lange2025/lrg1_hsc/benchmark_lange2025_lrg1_hsc_ds.png
   :width: 80%
   :alt: LRG1×HSC ESD MAP fit

   MAP :math:`\Delta\Sigma(R)` vs LRG1 × HSC-Y3 data.

Joint wp+ESD — LRG2 × HSC-Y3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 672.1 / 15 = 44.8` — **FAIL**

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Parameter
     - MAP value
     - Published
   * - S8
     - 0.848
     - 0.793
   * - Ω_m
     - 0.318
     - 0.303
   * - ``log10mmin``
     - 13.101
     - —
   * - ``log10m1``
     - 13.338
     - —

.. figure:: ../results/benchmarks/lange2025/lrg2_hsc/benchmark_lange2025_lrg2_hsc_wp.png
   :width: 80%
   :alt: LRG2×HSC wp MAP fit

   MAP :math:`w_p(r_p)` vs LRG2 data.

.. figure:: ../results/benchmarks/lange2025/lrg2_hsc/benchmark_lange2025_lrg2_hsc_ds.png
   :width: 80%
   :alt: LRG2×HSC ESD MAP fit

   MAP :math:`\Delta\Sigma(R)` vs LRG2 × HSC-Y3 data.

ESD-only — BGS2 × DES/KiDS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 27.1 / 6 = 4.52` — **FAIL**

.. figure:: ../results/benchmarks/lange2025/bgs2_ds_des/benchmark_lange2025_bgs2_ds_des_ds.png
   :width: 80%
   :alt: BGS2 ESD-only DES MAP fit

   MAP :math:`\Delta\Sigma(R)` vs BGS2 × DES/KiDS data (ESD-only fit).

ESD-only — BGS2 × HSC-Y3
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\chi^2/\text{dof} = 28.3 / 6 = 4.71` — **FAIL**

.. figure:: ../results/benchmarks/lange2025/bgs2_ds_hsc/benchmark_lange2025_bgs2_ds_hsc_ds.png
   :width: 80%
   :alt: BGS2 ESD-only HSC MAP fit

   MAP :math:`\Delta\Sigma(R)` vs BGS2 × HSC-Y3 data (ESD-only fit).

Summary table
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 15 10 12 12 12

   * - Benchmark key
     - Probes
     - ndof
     - :math:`\chi^2/\text{dof}`
     - S8 MAP
     - Ω_m MAP
   * - ``lange2025_bgs2_wp``
     - wp
     - 1
     - 46.8
     - 0.819
     - 0.323
   * - ``lange2025_bgs2_des``
     - wp+ESD
     - 15
     - 26.2
     - 0.833
     - 0.345
   * - ``lange2025_bgs2_hsc``
     - wp+ESD
     - 15
     - 18.1
     - 0.839
     - 0.349
   * - ``lange2025_bgs3_des``
     - wp+ESD
     - 15
     - 28.1
     - 0.835
     - 0.342
   * - ``lange2025_bgs3_hsc``
     - wp+ESD
     - 15
     - 23.4
     - 0.836
     - 0.346
   * - ``lange2025_lrg1_des``
     - wp+ESD
     - 15
     - 42.2
     - 0.837
     - 0.321
   * - ``lange2025_lrg1_hsc``
     - wp+ESD
     - 15
     - 41.2
     - 0.838
     - 0.322
   * - ``lange2025_lrg2_hsc``
     - wp+ESD
     - 15
     - 44.8
     - 0.848
     - 0.318
   * - ``lange2025_bgs2_ds_des``
     - ESD
     - 6
     - 4.52
     - —
     - —
   * - ``lange2025_bgs2_ds_hsc``
     - ESD
     - 6
     - 4.71
     - —
     - —

All benchmarks currently fail (:math:`\chi^2/\text{dof} > 2`).  The primary limitations are:

1. **Digitized data** — accuracy ∼20–30% per point.  Replace with Zenodo tables (record 17831718)
   for a fair quantitative comparison.
2. **Model mismatch** — the analytical halo model cannot replicate per-halo-concentration
   occupation.  The decorated HOD used in the paper introduces halo-to-halo stochasticity
   that shifts wp and ΔΣ at ∼10% level in the 1-halo regime.
3. **S8/Ω_m tension with Planck priors** — the optimizer pulls toward Planck values
   (S8 ≈ 0.83, Ω_m ≈ 0.32) rather than the published DESI values (S8 ≈ 0.79, Ω_m ≈ 0.30),
   because the digitized data cannot overcome the prior penalty.  Full Zenodo data with
   proper covariance matrices and a looser / no cosmological prior would be needed.

See :ref:`benchmarks` for the benchmark suite summary.
