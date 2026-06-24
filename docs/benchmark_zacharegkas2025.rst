:orphan:

.. _benchmark_zacharegkas2025:

Benchmark: Zacharegkas+2025 — DES Y3
======================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - ``Zacharegkas25HODModel``
   * - **Paper**
     - Zacharegkas et al. 2025 (`arXiv:2506.22367 <https://arxiv.org/abs/2506.22367>`_)
   * - **Survey**
     - DES Year 3, lens bin (:math:`\ell=1, m=1`): :math:`z \in [0.20, 0.40]`,
       :math:`\log M_* \in [9.56, 9.98]`, :math:`z_\mathrm{eff} \approx 0.30`
   * - **Observable (paper)**
     - Angular galaxy clustering :math:`w(\theta)` + galaxy-galaxy lensing :math:`\gamma_t(\theta)`
       in 30 bins over :math:`\theta \in [0.25, 250]` arcmin
   * - **Observable (benchmark)**
     - Joint :math:`w_p(r_p)` + :math:`\Delta\Sigma(R)` using same HOD parametrization
   * - **Cosmology**
     - DES Y3 (MagLim): :math:`\Omega_m=0.339,\ h=0.6737,\ \sigma_8=0.733,\ n_s=0.9649,\ \Omega_b=0.0486`
   * - **Config**
     - ``configs/benchmarks/benchmark_zacharegkas2025.yml``
   * - **Data**
     - ``data/zacharegkas2025_des/wp_des_bin1.csv`` — **NEEDS DATA TRANSCRIPTION**
     - ``data/zacharegkas2025_des/ds_des_bin1.csv`` — **NEEDS DATA TRANSCRIPTION**

Data Vector
-----------

.. important::
   Both data files are stubs with headers only.  See
   ``data/zacharegkas2025_des/README_data.md`` for exact transcription instructions.

Source: Zacharegkas+2025 published tables or DES data release,
DES Y3 bin 1 (lowest stellar mass bin).
Units: :math:`h^{-1}\,\mathrm{Mpc}` for :math:`r_p`/:math:`R`;
:math:`M_\odot\,h\,\mathrm{pc}^{-2}` for :math:`\Delta\Sigma`.

Triple-check protocol:

* :math:`r_p` and :math:`R` bins match Table 2 or data release column headers.
* :math:`w_p > 0` and decreasing; :math:`\Delta\Sigma > 0` and decreasing with :math:`R`.
* Confirm :math:`\pi_\mathrm{max} = 100\ h^{-1}\,\mathrm{Mpc}` from the paper.
* DES Y3 cosmology (:math:`\Omega_m = 0.339`, :math:`\sigma_8 = 0.760`) is lower
  than Planck18 — verify the config cosmology matches.

Published Parameters
--------------------

**Global SHMR** from abstract of Zacharegkas+2025 (all bins combined):

.. list-table::
   :header-rows: 1
   :widths: 28 22 20 20

   * - SHMR parameter
     - Published value
     - +1σ
     - −1σ
   * - ``log10m1_shmr`` (:math:`\log M_1`)
     - 11.506
     - +0.325
     - −0.404
   * - ``log10eps`` (:math:`\log \varepsilon`)
     - −1.632
     - +0.306
     - −0.181
   * - ``alpha_shmr`` (:math:`\alpha`)
     - −1.638
     - +0.108
     - −0.099
   * - ``gamma_shmr`` (:math:`\gamma`)
     - 0.596
     - +0.251
     - −0.210
   * - ``delta_shmr`` (:math:`\delta`)
     - 3.810
     - +2.045
     - −1.811

**Per-bin HOD parameters** from Appendix Table of Zacharegkas+2025,
bin :math:`(\ell=1, m=1)` — :math:`z \in [0.20, 0.40]`, :math:`\log M_* \in [9.56, 9.98]`:

.. list-table::
   :header-rows: 1
   :widths: 28 22 20 20

   * - HOD parameter
     - Published value
     - +1σ
     - −1σ
   * - ``sigma_logm_star``
     - 0.427
     - +0.144
     - −0.165
   * - ``alpha_sat``
     - 1.378
     - +0.079
     - −0.082
   * - ``bsat`` (:math:`B_\mathrm{sat}`)
     - 14.077
     - +2.551
     - −3.090
   * - ``beta_sat``
     - 0.202
     - +0.073
     - −0.061
   * - ``bcut`` (:math:`B_\mathrm{cut}`)
     - 9.411
     - +3.051
     - −2.936
   * - ``beta_cut``
     - 0.419
     - +0.228
     - −0.244
   * - ``log10m1_shmr`` (bin)
     - 11.624
     - +0.101
     - −0.104
   * - ``log10eps`` (bin)
     - −1.068
     - +0.172
     - −0.189
   * - ``alpha_shmr`` (bin)
     - −2.007
     - +0.329
     - −0.158
   * - ``gamma_shmr`` (bin)
     - 0.208
     - +0.192
     - −0.123
   * - ``delta_shmr`` (bin)
     - 5.883
     - +0.473
     - −0.743

Fixed: ``log10m_star_lo = 9.56``, ``log10m_star_hi = 9.98``, ``f_cen = 1.0``,
``f_sat = 1.0``, :math:`\kappa_\mathrm{sat} = 1.5` (illustrative value from paper).

Benchmark Setup
---------------

Joint :math:`w_p + \Delta\Sigma` fit.
Scale cuts: :math:`r_p, R \in [0.10, 20.0]\ h^{-1}\,\mathrm{Mpc}`.
Eight free parameters.  HMF backend: Tinker et al. 2008.

Config (``configs/benchmarks/benchmark_zacharegkas2025.yml``)::

    cosmology:
      Omega_m: 0.339
      h:       0.6737
      sigma8:  0.733
      n_s:     0.9649
      Omega_b: 0.0486

    model:
      hod_model:   Zacharegkas25HODModel
      hmf_backend: tinker08
      z:           0.30
      pi_max:      100.0

    parameters:
      log10m1_shmr:    {free: true,  bounds: [10.0, 13.5], init: 11.624}
      log10eps:        {free: true,  bounds: [-3.0,  0.0], init: -1.068}
      alpha_shmr:      {free: true,  bounds: [-3.0,  0.0], init: -2.007}
      gamma_shmr:      {free: true,  bounds: [0.1,   3.0], init: 0.208}
      delta_shmr:      {free: true,  bounds: [0.5,  10.0], init: 5.883}
      sigma_logm_star: {free: true,  bounds: [0.05,  1.0], init: 0.427}
      alpha_sat:       {free: true,  bounds: [0.5,   2.5], init: 1.378}
      kappa:           {free: true,  bounds: [0.01,  5.0], init: 1.5}
      log10m_star_lo:  {free: false, init: 9.56}
      log10m_star_hi:  {free: false, init: 9.98}
      f_cen:           {free: false, init: 1.0}

    joint:
      ds_file:   data/zacharegkas2025_des/ds_des_bin1.csv
      ds_rp_min: 0.10
      ds_rp_max: 20.0

Run command (after data transcription)::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model zacharegkas2025 --plot

Results
-------

.. note::
   Benchmark not yet run — data transcription required.
   After transcription, run the command above and paste the printed comparison table here.

Conclusions
-----------

The Zacharegkas+2025 model uses a Behroozi+2013-style SHMR with five shape parameters
(:math:`\log_{10}M_1`, :math:`\log_{10}\varepsilon`, :math:`\alpha`, :math:`\gamma`, :math:`\delta`)
and adds a stellar mass bin selection for the centrals.  The DES Y3 cosmology has notably
low :math:`\Omega_m = 0.339` and :math:`\sigma_8 = 0.760` compared to Planck18; using
the correct cosmology is essential for reproducing the published halo masses.

See :ref:`benchmarks` for the full suite summary.
