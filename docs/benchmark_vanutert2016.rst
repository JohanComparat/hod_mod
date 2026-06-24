:orphan:

.. _benchmark_vanutert2016:

Benchmark: van Uitert+2016 — GAMA + KiDS
==========================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - ``VanUitert16CSMFModel``
   * - **Paper**
     - van Uitert et al. 2016, MNRAS 459, 3251 (`arXiv:1601.06791 <https://arxiv.org/abs/1601.06791>`_)
   * - **Survey**
     - GAMA (clustering) + KiDS (lensing), z\ :sub:`eff` ≈ 0.18
   * - **Sample**
     - All GAMA galaxies in 8 stellar mass bins :math:`9.39 < \log_{10}(M_*/h^{-2}M_\odot) < 11.69`
   * - **Observable (paper)**
     - :math:`\Delta\Sigma(R)` (KiDS lensing) + stellar mass function (GAMA SMF)
   * - **Observable (benchmark)**
     - Joint :math:`w_p(r_p)` + :math:`\Delta\Sigma(R)`,
       :math:`\pi_\mathrm{max} = 60\ h^{-1}\,\mathrm{Mpc}`
   * - **Cosmology**
     - Planck 2013: :math:`\Omega_m=0.315,\ h=0.673,\ \sigma_8=0.829,\ n_s=0.960,\ \Omega_b=0.049`
   * - **Config**
     - ``configs/benchmarks/benchmark_vanutert2016.yml``
   * - **Data**
     - | ``data/vanutert2016_gama/wp_bin2_104_108.csv`` — **NEEDS DATA TRANSCRIPTION**
       | ``data/vanutert2016_gama/ds_bin2_104_108.csv`` — **NEEDS DATA TRANSCRIPTION**

Data Vector
-----------

.. important::
   Both data files are stubs with headers only.  See
   ``data/vanutert2016_gama/README_data.md`` for exact transcription instructions.

Source: van Uitert+2016 Appendix B, Table B2, stellar mass bin 2
(:math:`10.4 < \log M_* < 10.8`).  Contains :math:`r_p`, :math:`w_p`, :math:`\sigma_{w_p}`
and :math:`R`, :math:`\Delta\Sigma`, :math:`\sigma_{\Delta\Sigma}`.

Units: :math:`h^{-1}\,\mathrm{Mpc}` for scales; :math:`M_\odot\,h\,\mathrm{pc}^{-2}` for
:math:`\Delta\Sigma`.

Triple-check protocol:

* :math:`R` and :math:`r_p` values match Table B2 column headers.
* :math:`w_p` and :math:`\Delta\Sigma` values match the bin 2 columns.
* Confirm sign convention: :math:`\Delta\Sigma > 0` (tangential shear positive).
* Check that pi_max used in the paper matches ``pi_max: 60.0`` in the config.

Published Parameters
--------------------

From Table 3 of van Uitert+2016, "All" sample (combined :math:`\Delta\Sigma` + SMF fit).

.. note::
   The original paper fits :math:`\Delta\Sigma` + SMF simultaneously across 8 stellar mass
   bins in the range :math:`9.39 < \log_{10}(M_*/h^{-2}M_\odot) < 11.69`.  Our benchmark
   uses projected :math:`w_p(r_p)` + :math:`\Delta\Sigma`, so direct parameter comparison
   is approximate.

.. list-table::
   :header-rows: 1
   :widths: 28 28 22

   * - Parameter
     - Published value
     - 68% credible interval
   * - ``log10m_h1``
     - 10.97
     - :math:`^{+0.34}_{-0.25}`
   * - ``log10m_star0``
     - 10.58
     - :math:`^{+0.22}_{-0.15}`
   * - ``beta1`` (:math:`\beta_1`)
     - 7.5
     - :math:`^{+3.8}_{-2.7}`
   * - ``beta2`` (:math:`\beta_2 = 10^{\log_{10}\beta_2}`)
     - 0.25
     - :math:`^{+0.04}_{-0.06}`
   * - ``sigma_c``
     - 0.20
     - :math:`^{+0.02}_{-0.03}`
   * - ``b0``
     - 0.18
     - :math:`^{+0.28}_{-0.39}`
   * - ``b1``
     - 0.83
     - :math:`^{+0.27}_{-0.23}`
   * - ``alpha_s``
     - −0.83
     - :math:`^{+0.22}_{-0.16}`
   * - ``f_sub``
     - 0.59
     - :math:`^{+0.31}_{-0.40}`
   * - ``f_conc``
     - 0.70
     - :math:`^{+0.19}_{-0.15}`

Notes: masses in :math:`h^{-1}M_\odot` (column 1) and :math:`h^{-2}M_\odot` (column 2).
:math:`f_\mathrm{sub}` and :math:`f_\mathrm{conc}` are fixed in our benchmark config.

Fixed: ``log10m_star_lo = 10.4``, ``log10m_star_hi = 10.8``.

Benchmark Setup
---------------

Joint :math:`w_p + \Delta\Sigma` fit.
Scale cuts: :math:`r_p \in [0.05, 20.0]\ h^{-1}\,\mathrm{Mpc}`;
:math:`R \in [0.05, 2.0]\ h^{-1}\,\mathrm{Mpc}`.
Eight free parameters.  HMF backend: Tinker et al. 2008.

Config (``configs/benchmarks/benchmark_vanutert2016.yml``)::

    cosmology:
      Omega_m: 0.315
      h:       0.673
      sigma8:  0.829
      n_s:     0.9603
      Omega_b: 0.0487

    model:
      hod_model:   VanUitert16CSMFModel
      hmf_backend: tinker08
      z:           0.18
      pi_max:      60.0

    parameters:
      log10m_h1:      {free: true, bounds: [10.5, 14.0], init: 11.80}
      log10m_star0:   {free: true, bounds: [9.0,  12.0], init: 10.50}
      beta1:          {free: true, bounds: [1.0,  15.0], init: 5.0}
      log10_beta2:    {free: true, bounds: [-2.0,  1.0], init: -0.5}
      sigma_c:        {free: true, bounds: [0.05,  0.8], init: 0.15}
      alpha_s:        {free: true, bounds: [-2.5, -0.1], init: -1.1}
      b0:             {free: true, bounds: [-3.0,  3.0], init: 0.0}
      b1:             {free: true, bounds: [0.1,   5.0], init: 1.5}
      log10m_star_lo: {free: false, init: 10.4}
      log10m_star_hi: {free: false, init: 10.8}

    joint:
      ds_file:   data/vanutert2016_gama/ds_bin2_104_108.csv
      ds_rp_min: 0.05
      ds_rp_max: 2.0

Run command (after data transcription)::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model vanutert2016 --plot

Results
-------

.. note::
   Benchmark not yet run — data transcription required.
   After transcription, run the command above and paste the printed comparison table here.

Conclusions
-----------

The van Uitert+2016 CSMF model uses a conditional stellar mass function (CSMF)
framework to predict :math:`w_p` and :math:`\Delta\Sigma` simultaneously for a stellar
mass bin (rather than a threshold sample).  Eight parameters control the central SHMR
shape, satellite amplitude, and satellite radial profile (b0, b1 parametrize the
satellite bias relative to NFW).

See :ref:`benchmarks` for the full suite summary.
