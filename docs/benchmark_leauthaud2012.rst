:orphan:

.. _benchmark_leauthaud2012:

Benchmark: Leauthaud+2012 — COSMOS PHOTO_z2
============================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - ``Leauthaud12HODModel``
   * - **Paper**
     - Leauthaud et al. 2012, ApJ 744, 159 (`arXiv:1104.0928 <https://arxiv.org/abs/1104.0928>`_)
   * - **Survey**
     - COSMOS, PHOTO_z2: :math:`z \in [0.48, 0.74]`, :math:`\log_{10}(M_*/M_\odot) > 10.6`
   * - **Observable (paper)**
     - Angular clustering :math:`w(\theta)` + :math:`\Delta\Sigma(R)` + SMF
       (our benchmark uses projected :math:`w_p(r_p)` + :math:`\Delta\Sigma`)
   * - **Observable (benchmark)**
     - Joint :math:`w_p(r_p)` + :math:`\Delta\Sigma(R)`,  :math:`\pi_\mathrm{max} = 40\ h^{-1}\,\mathrm{Mpc}`
   * - **Cosmology**
     - WMAP5: :math:`\Omega_m=0.258,\ h=0.72,\ \sigma_8=0.796,\ n_s=0.963,\ \Omega_b=0.044`
   * - **Config**
     - ``configs/benchmarks/benchmark_leauthaud2012.yml``
   * - **Data**
     - ``data/leauthaud2012_cosmos/wp_photo_z2_thresh106.csv`` — **NEEDS DATA TRANSCRIPTION**
     - ``data/leauthaud2012_cosmos/ds_photo_z2_thresh106.csv`` — **NEEDS DATA TRANSCRIPTION**

Data Vector
-----------

.. important::
   Both data files are stubs with headers only.  See
   ``data/leauthaud2012_cosmos/README_data.md`` for exact transcription instructions.

Sources:

* :math:`\Delta\Sigma(R)`: Leauthaud+2012 Table A2, PHOTO_z2 column, threshold :math:`\log M_* > 10.6`.
  Units: :math:`M_\odot\,h\,\mathrm{pc}^{-2}` (as printed in the paper).
  **Convert to code units** (:math:`M_\odot\,h\,\mathrm{pc}^{-2}`) before saving — same units.
* :math:`w_p(r_p)`: from the COSMOS galaxy–galaxy clustering analysis in the same paper.

Triple-check protocol:

* :math:`R` and :math:`r_p` values match the table column headers.
* :math:`\Delta\Sigma` values are positive, decreasing with :math:`R`.
* :math:`w_p` values are positive, decreasing with :math:`r_p`.
* Confirm units: paper uses :math:`h\,M_\odot\,\mathrm{pc}^{-2}` — verify sign convention is
  :math:`\Sigma(<R) - \bar\Sigma(<R)` (positive for typical lens profiles).
* :math:`z_\mathrm{eff} = 0.66` (midpoint of :math:`z \in [0.48, 0.74]`).

Published Parameters
--------------------

From Table 5 of Leauthaud+2012, SIG MOD1, z2 column (:math:`z \in [0.48, 0.74]`):

.. note::
   The original paper fits angular correlation :math:`w(\theta)` + :math:`\Delta\Sigma` + SMF.
   Our benchmark re-fits the same HOD form using projected :math:`w_p(r_p)`, so direct
   chi²/dof comparison with the paper is not meaningful; only the parameter recovery matters.

.. list-table::
   :header-rows: 1
   :widths: 28 25 22

   * - Parameter
     - Published value
     - 1σ error
   * - ``log10m1``
     - 12.725
     - ±0.032
   * - ``log10m_star0``
     - 11.038
     - ±0.019
   * - ``beta``
     - 0.466
     - ±0.009
   * - ``delta``
     - 0.61
     - ±0.13
   * - ``gamma``
     - 1.95
     - ±0.25
   * - ``sigma_logm``
     - 0.249
     - ±0.019
   * - ``bsat`` (:math:`B_\mathrm{sat}`)
     - 9.04
     - ±0.81
   * - ``bcut`` (:math:`B_\mathrm{cut}`)
     - 1.65
     - ±0.65
   * - ``beta_sat``
     - 0.740
     - ±0.059
   * - ``beta_cut``
     - 0.59
     - ±0.28

Fixed: ``log10m_star_thresh = 10.6``, ``alpha_sat = 1.0``.

Benchmark Setup
---------------

Joint :math:`w_p + \Delta\Sigma` fit.
Scale cuts: :math:`r_p, R \in [0.05, 15.0]\ h^{-1}\,\mathrm{Mpc}`.
Six free parameters.  HMF backend: Tinker et al. 2008.

Config (``configs/benchmarks/benchmark_leauthaud2012.yml``)::

    cosmology:
      Omega_m: 0.258
      h:       0.720
      sigma8:  0.796
      n_s:     0.963
      Omega_b: 0.044

    model:
      hod_model:   Leauthaud12HODModel
      hmf_backend: tinker08
      z:           0.66
      pi_max:      40.0

    parameters:
      log10m1:       {free: true, bounds: [11.5, 14.0], init: 12.725}
      log10m_star0:  {free: true, bounds: [9.5,  12.0], init: 11.038}
      beta:          {free: true, bounds: [0.1,   1.5], init: 0.466}
      delta:         {free: true, bounds: [0.1,   2.0], init: 0.61}
      gamma:         {free: true, bounds: [0.5,   5.0], init: 1.95}
      sigma_logm:    {free: true, bounds: [0.05,  0.8], init: 0.249}
      log10m_star_thresh: {free: false, init: 10.6}
      alpha_sat:     {free: false, init: 1.0}

    joint:
      ds_file:   data/leauthaud2012_cosmos/ds_photo_z2_thresh106.csv
      ds_rp_min: 0.05
      ds_rp_max: 15.0

Run command (after data transcription)::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model leauthaud2012 --plot

Results
-------

.. note::
   Benchmark not yet run — data transcription required.
   After transcription, run the command above and paste the printed comparison table here.

Conclusions
-----------

The Leauthaud+2012 model connects stellar mass to halo mass via an SHMR parametrized
by the double power-law :math:`M_*(M_h)` relation (Behroozi+2010 form).  The joint
:math:`w_p + \Delta\Sigma` constraint breaks the degeneracy between the HOD normalization
and the satellite fraction.  The WMAP5 cosmology (:math:`\sigma_8=0.796`) is noticeably
lower than Planck18 and must be set correctly.

See :ref:`benchmarks` for the full suite summary.
