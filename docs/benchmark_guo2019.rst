:orphan:

.. _benchmark_guo2019:

Benchmark: Guo+2019 — eBOSS ELG
==================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - ``Guo19ICSMFModel``
   * - **Paper**
     - Guo et al. 2019, ApJ 871, 147 (`arXiv:1810.05318 <https://arxiv.org/abs/1810.05318>`_)
   * - **Survey**
     - eBOSS ELG (Emission Line Galaxies), z\ :sub:`eff` ≈ 0.80
   * - **Observable**
     - :math:`w_p(r_p)`,  :math:`\pi_\mathrm{max} = 60\ h^{-1}\,\mathrm{Mpc}`
   * - **Cosmology**
     - Planck 2015: :math:`\Omega_m=0.307,\ h=0.677,\ \sigma_8=0.829,\ n_s=0.961,\ \Omega_b=0.049`
   * - **Config**
     - ``configs/benchmarks/benchmark_guo2019.yml``
   * - **Data**
     - ``data/guo2019_eboss_elg/wp_elg_z08.csv`` — digitized from Figure 4 of Guo+2019

Data Vector
-----------

Source: Guo+2019 Figure 4, eBOSS ELG clustering at :math:`0.8 < z < 0.9`.
:math:`\pi_\mathrm{max} = 20\ h^{-1}\,\mathrm{Mpc}` (Section 3.4 of the paper).
8 data points at :math:`r_p \in [0.20, 25]\ h^{-1}\,\mathrm{Mpc}`.
Digitized with ~20-30% precision from the figure.

.. warning::
   **Model–data mismatch**: ``Guo19ICSMFModel`` computes :math:`w_p` for a
   *threshold* ELG sample, but the data come from Figure 4 which shows
   the ELG clustering in a specific redshift bin.  The shape mismatch
   (model overpredicts at :math:`r_p \sim 1`–10, underpredicts at :math:`r_p \sim 25`)
   is consistent with the threshold vs. bin selection difference.

Published Parameters
--------------------

From Table 3 of Guo+2019 (ELG sample):

.. list-table::
   :header-rows: 1
   :widths: 35 25 20

   * - Parameter
     - Published value
     - 1σ error
   * - ``log10m_star0``
     - 9.5
     - —
   * - ``log10m1_shmr``
     - 11.5
     - —
   * - ``alpha_shmr``
     - 0.3
     - —
   * - ``beta_shmr``
     - 1.5
     - —
   * - ``sigma_logm_star``
     - 0.2
     - —
   * - ``log10m1_sat``
     - 12.5
     - —
   * - ``alpha_sat``
     - 1.0
     - —
   * - ``f_cen``
     - 0.8
     - —
   * - ``log10m_q``
     - 12.0
     - —

Benchmark Setup
---------------

:math:`w_p`-only fit.
Scale cuts: :math:`r_p \in [0.10, 20.0]\ h^{-1}\,\mathrm{Mpc}`.
Nine free parameters (Guo18 params + ``log10m_q``).  HMF backend: Tinker et al. 2008.

Config (``configs/benchmarks/benchmark_guo2019.yml``)::

    cosmology:
      Omega_m: 0.307
      h:       0.677
      sigma8:  0.829
      n_s:     0.961
      Omega_b: 0.0486

    model:
      hod_model:   Guo19ICSMFModel
      hmf_backend: tinker08
      z:           0.80
      pi_max:      60.0

    parameters:
      log10m_star0:      {free: true,  bounds: [9.0, 12.0],  init: 10.0}
      log10m1_shmr:      {free: true,  bounds: [10.0, 13.0], init: 11.5}
      log10m1_sat:       {free: true,  bounds: [11.0, 15.0], init: 12.5}
      f_cen:             {free: true,  bounds: [0.1, 1.0],   init: 0.5}
      log10m_q:          {free: true,  bounds: [11.0, 14.0], init: 13.1}
      alpha_shmr:          {free: false, init: 0.3}
      beta_shmr:           {free: false, init: 1.5}
      sigma_logm_star:     {free: false, init: 0.2}
      alpha_sat:           {free: false, init: 1.0}

Run command::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model guo2019 --plot

Results
-------

MAP fit: :math:`\chi^2/\text{dof} = 8.82 / 3 = 2.939`.
**Status: FAILED** (:math:`\chi^2/\text{dof} > 2.0`).

Best-fit parameters: ``log10m_star0 = 10.74``, ``log10m1_shmr = 10.00`` (lower bound),
``log10m1_sat = 15.00`` (upper bound), ``log10m_q = 11.00`` (lower bound).
The optimizer finds a degenerate solution with parameters at bounds, indicating
the threshold model cannot reproduce the observed ELG wp.

Root cause: threshold vs. bin selection mismatch (see warning above).

Conclusions
-----------

The Guo+2019 model extends the Guo+2018 ICSMF framework to star-forming/emission-line
galaxies by introducing a quenching mass scale :math:`M_q` (``log10m_q``).  Above
:math:`M_q` the central galaxy is assumed to be quenched and absent from the ELG sample.
The high redshift (:math:`z \approx 0.80`) means the linear growth factor differs
significantly from the SDSS :math:`z \approx 0.1` benchmarks.

The benchmark fails because the data comes from a specific ELG redshift bin while
the model implements a threshold selection.  The quenching mass scale ``log10m_q``
introduces an upper mass cutoff but the lower mass behaviour also differs.
For a passing benchmark, threshold-selected ELG data or a bin-selection extension
to the model would be needed.

See :ref:`benchmarks` for the full suite summary.
