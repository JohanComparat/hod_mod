:orphan:

.. _benchmark_zheng2007:

Benchmark: Zheng+2007 — SDSS M\ :sub:`r` < −21
================================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - ``HODModel``
   * - **Paper**
     - Zheng et al. 2007, ApJ 667, 760 (`arXiv:astro-ph/0703457 <https://arxiv.org/abs/astro-ph/0703457>`_)
   * - **Survey**
     - SDSS DR3, :math:`M_r < -21` luminosity-threshold sample, z\ :sub:`eff` ≈ 0.1
   * - **Observable**
     - :math:`w_p(r_p)`,  :math:`\pi_\mathrm{max} = 40\ h^{-1}\,\mathrm{Mpc}`
   * - **Cosmology**
     - WMAP3: :math:`\Omega_m=0.238,\ h=0.732,\ \sigma_8=0.740,\ n_s=0.958,\ \Omega_b=0.041`
   * - **Config**
     - ``configs/benchmarks/benchmark_zheng2007.yml``
   * - **Data**
     - ``data/zheng2007_sdss/wp_mr21_sdss.csv`` — computed from Zehavi+2005 power-law fit

Data Vector
-----------

Source: Zehavi et al. 2005, ApJ 630, 1 (`arXiv:astro-ph/0408564 <https://arxiv.org/abs/astro-ph/0408564>`_),
Table 2, :math:`M_r < -21` row, power-law fit :math:`r_0 = 6.24\ h^{-1}\,\mathrm{Mpc}`,
:math:`\gamma = 1.90`, :math:`\pi_\mathrm{max} = 40\ h^{-1}\,\mathrm{Mpc}`.
Data points recomputed analytically from the power-law (not digitized from figures).
Units: :math:`h^{-1}\,\mathrm{Mpc}`.


Published Parameters
--------------------

From Table 3 of Zheng+2007 (:math:`M_r < -21` row):

.. list-table::
   :header-rows: 1
   :widths: 30 25 20

   * - Parameter
     - Published value
     - 1σ error
   * - ``log10mmin``
     - 12.78
     - ±0.10
   * - ``sigma_logm``
     - 0.68
     - ±0.15
   * - ``log10m0``
     - 11.92
     - ±0.30
   * - ``log10m1``
     - 13.88
     - ±0.08
   * - ``alpha``
     - 1.39
     - ±0.15

Benchmark Setup
---------------

Scale cuts: :math:`r_p \in [0.10, 20.0]\ h^{-1}\,\mathrm{Mpc}`.
Five free parameters.  HMF backend: Tinker et al. 2008.

Config (``configs/benchmarks/benchmark_zheng2007.yml``)::

    cosmology:
      Omega_m: 0.238
      h:       0.732
      sigma8:  0.740
      n_s:     0.958
      Omega_b: 0.041

    model:
      hod_model:   HODModel
      hmf_backend: tinker08
      z:           0.1
      pi_max:      40.0

    parameters:
      log10mmin:  {free: true,  bounds: [11.5, 14.0], init: 12.78}
      sigma_logm: {free: true,  bounds: [0.05, 1.50], init: 0.68}
      log10m0:    {free: true,  bounds: [10.0, 14.0], init: 11.92}
      log10m1:    {free: true,  bounds: [12.0, 15.0], init: 13.88}
      alpha:      {free: true,  bounds: [0.50, 2.50], init: 1.39}

Run command::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model zheng2007 --plot

Results
-------

MAP fit: :math:`\chi^2/\text{dof} = 11.30 / 6 = 1.884`.
**Status: PASSED** (:math:`\chi^2/\text{dof} < 2.0`).

.. note::
   The data vector is derived from a power-law fit to the observed :math:`w_p`,
   not the raw Zehavi+2005 data points.  This means the power-law model fits
   the data exactly at 2 effective degrees of freedom, and the HOD parameters
   are poorly constrained individually (strong degeneracy).  The chi2/dof
   should be interpreted as a consistency check, not a precision fit.

.. list-table::
   :header-rows: 1
   :widths: 25 18 18 18 18 18

   * - Parameter
     - MAP
     - Published
     - MCMC median
     - 16th pct
     - 84th pct
   * - ``log10mmin``
     - 12.700
     - 12.78 ± 0.10
     - 12.45
     - 11.81
     - 13.14
   * - ``sigma_logm``
     - 1.427 ⚠
     - 0.68 ± 0.15
     - 1.28
     - 0.97
     - 1.44
   * - ``log10m0``
     - 10.000 ⚠
     - 11.92 ± 0.30
     - 11.39
     - 10.46
     - 12.35
   * - ``log10m1``
     - 12.648 ⚠
     - 13.88 ± 0.08
     - 12.60
     - 12.21
     - 13.10
   * - ``alpha``
     - 1.017 ⚠
     - 1.39 ± 0.15
     - 1.01
     - 0.86
     - 1.17

⚠ = MAP parameter deviates by > 2σ from published value.

The MAP finds a degenerate solution; the MCMC posterior is broad and does not
recover the published values tightly.  The discrepancy is expected because a
power-law :math:`w_p` constrains only the correlation amplitude and slope
(effectively 2 degrees of freedom), while the HOD has 5 free parameters.
For a definitive reproduction test, the original Zehavi+2005 data table is needed.

MCMC Results
^^^^^^^^^^^^

MCMC run: 32 walkers × 2000 steps (500 burn-in), 64 000 samples.
Flatchain saved to ``results/benchmarks/zheng2007_sdss/flatchain.npz``.

Conclusions
-----------

The ``HODModel`` implements the standard Zheng+2007 five-parameter HOD with a sharp
central step function and a power-law satellite term.  The WMAP3 cosmology used here
has lower :math:`\sigma_8 = 0.740` than Planck18, which shifts the characteristic halo
masses and must be set correctly to reproduce the published best-fit parameters.

See :ref:`benchmarks` for the full suite summary.
