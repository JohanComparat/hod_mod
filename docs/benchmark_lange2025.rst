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

   # MAP fit: joint wp+ESD, BGS2 + DES/KiDS lensing
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_des --plot

   # MAP fit: joint wp+ESD, BGS2 + HSC-Y3 lensing
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_hsc --plot

   # MAP fit: wp-only, BGS2 (with free cosmology)
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_wp --plot

   # MAP fit: ESD-only, BGS2 + HSC
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_ds_hsc --plot

   # Full MCMC (slow — free cosmology, 64 walkers × 4000 steps)
   python hod_mod/scripts/benchmarks/run_benchmark.py --model lange2025_bgs2_hsc --mcmc --plot

Available model keys: ``lange2025_bgs2_des``, ``lange2025_bgs3_des``, ``lange2025_lrg1_des``,
``lange2025_bgs2_hsc``, ``lange2025_bgs3_hsc``, ``lange2025_lrg1_hsc``, ``lange2025_lrg2_hsc``,
``lange2025_bgs2_wp``, ``lange2025_bgs2_ds_des``, ``lange2025_bgs2_ds_hsc``.

Data status
-----------

.. note::

   Data digitized from ar5iv PNG renderings of Figures 3–4 (Lange+2025).
   Values extracted using power-law models anchored from visual inspection of the figures
   (wp ∝ rp\ :sup:`−1.05`, ESD ∝ R\ :sup:`−1.2`); accuracy ~20–30% per data point.
   A 1% systematic uncertainty is added in quadrature to all error bars.

   Replace ``data/lange2025_desi_dr1/*/`` with the published tables from
   Zenodo record 17831718 when available (currently under review).  After replacing,
   update the comments in each CSV header accordingly.

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
