:orphan:

.. _benchmark_guo2018:

Benchmark: Guo+2018 — SDSS LOWZ
=================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - ``Guo18ICSMFModel``
   * - **Paper**
     - Guo et al. 2018, ApJ 858, 30 (`arXiv:1804.01993 <https://arxiv.org/abs/1804.01993>`_)
   * - **Survey**
     - SDSS BOSS LOWZ, :math:`\log_{10}(M_*/M_\odot) > 10.0`, z\ :sub:`eff` ≈ 0.1
   * - **Observable**
     - :math:`w_p(r_p)`,  :math:`\pi_\mathrm{max} = 60\ h^{-1}\,\mathrm{Mpc}`
   * - **Cosmology**
     - Planck 2015: :math:`\Omega_m=0.307,\ h=0.677,\ \sigma_8=0.829,\ n_s=0.961,\ \Omega_b=0.049`
   * - **Config**
     - ``configs/benchmarks/benchmark_guo2018.yml``
   * - **Data**
     - ``data/guo2018_sdss/wp_mstar10_lowz.csv`` — digitized from Figure 5 of Guo+2018

Data Vector
-----------

Source: Guo+2018 Figure 5, mass bin :math:`11.0 < \log M_* / M_\odot < 11.5` (blue × symbols).
Digitized with ~20% precision; :math:`\pi_\mathrm{max} = 100\ h^{-1}\,\mathrm{Mpc}` (BOSS LOWZ).
8 data points at :math:`r_p \in [0.20, 25]\ h^{-1}\,\mathrm{Mpc}`.

.. warning::
   **Model–data mismatch**: ``Guo18ICSMFModel`` computes :math:`w_p` for a
   *threshold* sample (:math:`M_* > M_{*,\min}`), but the data come from a
   *mass-bin* sample (:math:`11.0 < \log M_* < 11.5`).  This systematic
   mismatch limits how well the model can reproduce the observed signal.
   A perfect threshold model at :math:`M_* > 11.0` would include all galaxies
   above the lower bin edge, predicting a different amplitude than the bin-selected data.

Published Parameters
--------------------

From Table 2 of Guo+2018:

.. list-table::
   :header-rows: 1
   :widths: 35 25 20

   * - Parameter
     - Published value
     - 1σ error
   * - ``log10m_star0``
     - 10.7
     - —
   * - ``log10m1_shmr``
     - 11.9
     - —
   * - ``alpha_shmr``
     - 0.3
     - —
   * - ``beta_shmr``
     - 1.5
     - —
   * - ``sigma_logm_star``
     - 0.15
     - —
   * - ``log10m1_sat``
     - 13.0
     - —
   * - ``alpha_sat``
     - 1.0
     - —
   * - ``f_cen``
     - 1.0
     - —

Benchmark Setup
---------------

:math:`w_p`-only fit.
Scale cuts: :math:`r_p \in [0.10, 20.0]\ h^{-1}\,\mathrm{Mpc}`.
Eight free parameters.  HMF backend: Tinker et al. 2008.

Config (``configs/benchmarks/benchmark_guo2018.yml``)::

    cosmology:
      Omega_m: 0.307
      h:       0.677
      sigma8:  0.829
      n_s:     0.961
      Omega_b: 0.0486

    model:
      hod_model:   Guo18ICSMFModel
      hmf_backend: tinker08
      z:           0.1
      pi_max:      60.0

    parameters:
      log10m_star0:      {free: true,  bounds: [10.0, 13.5], init: 11.5}
      log10m1_shmr:      {free: true,  bounds: [12.0, 15.0], init: 13.0}
      alpha_shmr:        {free: true,  bounds: [0.01,  2.0], init: 0.3}
      beta_shmr:         {free: true,  bounds: [0.5,   5.0], init: 1.5}
      log10m1_sat:       {free: true,  bounds: [12.0, 15.0], init: 13.5}
      f_cen:             {free: true,  bounds: [0.5,   1.0], init: 1.0}
      sigma_logm_star:   {free: false, init: 0.15}
      alpha_sat:         {free: false, init: 1.0}

Run command::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model guo2018 --plot

Results
-------

MAP fit: :math:`\chi^2/\text{dof} = 5.79 / 2 = 2.894`.
**Status: FAILED** (:math:`\chi^2/\text{dof} > 2.0`).

Best-fit parameters hit bounds: ``log10m_star0 = 13.47`` (near upper limit 13.5),
``log10m1_sat = 12.00`` (at lower limit).  The optimizer cannot find a physical
solution because the threshold model systematically overpredicts the amplitude
for physical BOSS LOWZ SHMR parameters.

Root cause: threshold vs. mass-bin selection mismatch (see warning above).

Conclusions
-----------

The Guo+2018 ICSMF model connects stellar mass to halo mass via a double power-law
SHMR and introduces a separate satellite mass scale (:math:`M_{1,\mathrm{sat}}`).
The ``f_cen`` parameter accounts for incompleteness of the central galaxy sample.
This model serves as the baseline for the eBOSS ELG extension in Guo+2019.

The benchmark fails because the digitized data comes from a mass-bin sample
while the model implements threshold selection.  A proper comparison would
require either threshold-selected data from the paper or a model extension
to support bin-selected observables.

See :ref:`benchmarks` for the full suite summary.
