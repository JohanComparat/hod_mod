.. _benchmark_kravtsov2004:

Benchmark: Kravtsov+2004 — BOSS CMASS
=======================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - ``Kravtsov04HODModel``
   * - **Paper**
     - Kravtsov et al. 2004, ApJ 609, 35 (`arXiv:astro-ph/0308519 <https://arxiv.org/abs/astro-ph/0308519>`_)
   * - **Survey**
     - BOSS CMASS (same data as :doc:`benchmark_more2015`), z\ :sub:`eff` = 0.52
   * - **Observable**
     - :math:`w_p(r_p)`, 12 bins, :math:`r_p \in [0.18, 52]\ h^{-1}\,\mathrm{Mpc}`,
       :math:`\pi_\mathrm{max} = 60\ h^{-1}\,\mathrm{Mpc}`
   * - **Cosmology**
     - WMAP7: :math:`\Omega_m=0.272,\ h=0.704,\ \sigma_8=0.810,\ n_s=0.966,\ \Omega_b=0.044`
   * - **Config**
     - ``configs/benchmarks/benchmark_kravtsov2004.yml``
   * - **Data**
     - ``data/more2015_boss_cmass/wp_cmass_z052.csv``

Data Vector
-----------

Same 12-bin BOSS CMASS :math:`w_p(r_p)` measurement used in :doc:`benchmark_more2015`.
See that page for the full data table and triple-check log.

Kravtsov+2004 is a theoretical HOD parametrization introduced in the context of
galaxy formation simulations.  No single published observational dataset accompanies
the paper, so we validate against the same BOSS CMASS data as the More+2015 benchmark
for a direct model-to-model comparison on equal footing.

Implementation reference: ``tests/test_aum_comparison.py`` (comparison against the
AUM code by S. More, `github.com/surhudm/aum <https://github.com/surhudm/aum>`_).

HOD form
--------

The Kravtsov+2004 model parametrizes central and satellite occupation as:

.. math::

   \langle N_\mathrm{cen}(M) \rangle &= \frac{1}{2}
     \left[1 + \mathrm{erf}\!\left(\frac{\log_{10}M - \log_{10}M_\mathrm{min}}
                                         {\sigma_{\log M}}\right)\right]

   \langle N_\mathrm{sat}(M) \rangle &= \langle N_\mathrm{cen}(M) \rangle
     \left(\frac{M - M_0}{M_1}\right)^{\alpha}

Five free parameters: ``log10mmin``, ``sigma_logm``, ``log10m0``, ``log10m1``, ``alpha``.

Benchmark Setup
---------------

Scale cuts: :math:`r_p \in [0.50, 50.0]\ h^{-1}\,\mathrm{Mpc}` (11 bins).
Five free parameters.  HMF backend: Tinker et al. 2008.

Config (``configs/benchmarks/benchmark_kravtsov2004.yml``)::

    cosmology:
      Omega_m: 0.272
      h:       0.704
      sigma8:  0.810
      n_s:     0.966
      Omega_b: 0.044

    model:
      hod_model:   Kravtsov04HODModel
      hmf_backend: tinker08
      z:           0.52
      pi_max:      60.0

    parameters:
      log10mmin:  {free: true,  bounds: [12.0, 14.5], init: 13.00}
      sigma_logm: {free: true,  bounds: [0.05, 1.50], init: 0.38}
      log10m0:    {free: true,  bounds: [11.0, 15.0], init: 13.50}
      log10m1:    {free: true,  bounds: [13.0, 15.5], init: 14.00}
      alpha:      {free: true,  bounds: [0.50, 2.50], init: 1.00}

Run command::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model kravtsov2004 --plot --mcmc

Results
-------

MAP fit: :math:`\chi^2/\text{dof} = 7.65 / 4 = 1.91` (no published value to compare).
**Status: PASSED** (:math:`\chi^2/\text{dof} < 2.0`).

No published parameter comparison is available (theoretical HOD; no original dataset).
Best-fit parameters from the MAP:

.. list-table::
   :header-rows: 1
   :widths: 35 35

   * - Parameter
     - Best-fit (MAP)
   * - ``log10mmin``
     - 13.193
   * - ``sigma_logm``
     - 0.401
   * - ``log10m0``
     - 14.334
   * - ``log10m1``
     - 13.405
   * - ``alpha``
     - 1.075

.. figure:: ../results/benchmarks/kravtsov2004_cmass/benchmark_kravtsov2004_wp.png
   :width: 80%
   :alt: Kravtsov+2004 wp comparison

   Best-fit :math:`w_p(r_p)` vs. BOSS CMASS data (top) and residuals (bottom).
   MAP result with :math:`\chi^2/\text{dof} = 1.91`.

MCMC Results
^^^^^^^^^^^^

The MCMC was run with 32 walkers × 2000 steps (500 burn-in), producing a flat chain
of 64 000 samples.  Posterior statistics derived from the full chain:

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 18 18 18

   * - Parameter
     - MAP
     - Median
     - 16th pct
     - 84th pct
     - :math:`\sigma`
   * - ``log10mmin``
     - 13.193
     - 13.44
     - 12.99
     - 14.06
     - 0.48
   * - ``sigma_logm``
     - 0.401
     - 0.56
     - 0.19
     - 0.94
     - 0.32
   * - ``log10m0``
     - 14.334
     - 12.67
     - 11.54
     - 13.77
     - 1.01
   * - ``log10m1``
     - 13.405
     - 13.83
     - 13.38
     - 14.21
     - 0.37
   * - ``alpha``
     - 1.075
     - 1.34
     - 0.90
     - 1.81
     - 0.44

.. figure:: ../results/benchmarks/kravtsov2004_cmass/benchmark_kravtsov2004_wp_mcmc.png
   :width: 80%
   :alt: Kravtsov+2004 wp with MCMC band

   Projected correlation function :math:`w_p(r_p)` vs. BOSS CMASS data (Black dots).
   Solid blue: MAP model (:math:`\chi^2/\text{dof} = 1.91`).
   Dashed blue: MCMC median. Shaded blue: posterior :math:`1\sigma` band
   (16th–84th percentile over 400 chain draws).
   Bottom: residuals data/MAP :math:`-1`; band shows MCMC spread.

.. figure:: ../results/benchmarks/kravtsov2004_cmass/corner_kravtsov2004.png
   :width: 90%
   :alt: Kravtsov+2004 MCMC corner plot

   Posterior corner plot for the five HOD parameters.
   Contours show the :math:`1\sigma` and :math:`2\sigma` credible regions.
   Dashed red lines mark the MAP values.
   The ``log10m0`` marginal is broad (:math:`\sigma \approx 1.0` dex),
   reflecting a strong degeneracy with ``log10m1``:
   raising :math:`M_0` (exponential cutoff) can be compensated by raising
   :math:`M_1` (satellite mass scale) while keeping the total satellite
   fraction nearly fixed.

To regenerate the MCMC figures independently::

    python hod_mod/scripts/benchmarks/plot_kravtsov2004_mcmc.py

Comparison with More+2015
^^^^^^^^^^^^^^^^^^^^^^^^^^

Both models fit the same BOSS CMASS data with the same WMAP7 cosmology.
The figure below places them side by side.

.. figure:: ../results/benchmarks/kravtsov2004_cmass/comparison_kravtsov2004_vs_more2015_wp.png
   :width: 85%
   :alt: Model comparison Kravtsov+2004 vs More+2015

   Projected correlation function :math:`w_p(r_p)` for both HOD models vs.
   BOSS CMASS data.  Solid lines: MAP best-fits.  Shaded bands: MCMC
   :math:`1\sigma` posteriors (400 draws each).  Bottom: residuals
   model/data :math:`-1`.

The MoreHODModel achieves :math:`\chi^2/\text{dof} = 1.54` vs 1.91 for
Kravtsov+2004.  The extra flexibility of the :math:`\kappa` parameter
(a smooth Heaviside onset for the satellite term) lets MoreHODModel fit
the 1-halo to 2-halo transition more accurately than the hard exponential
cutoff at :math:`M_0` used by Kravtsov+2004.  Both models are nonetheless
statistically acceptable on this dataset.

Conclusions
-----------

The Kravtsov+2004 model fits the BOSS CMASS :math:`w_p` with :math:`\chi^2/\text{dof} = 1.91`,
slightly worse than MoreHODModel (1.54) on the same data.  The harder cutoff at
:math:`M_0` (no kappa smoothing) gives less flexibility in matching the 1-halo
to 2-halo transition, reflected in a higher :math:`\chi^2`.

The MCMC posteriors confirm the fit quality: the 1σ band in :math:`w_p` brackets
the data across all 11 fitted bins, and the MAP model lies well inside the posterior.
The dominant degeneracy is between ``log10m0`` and ``log10m1``; both satellites scales
are poorly constrained individually but their combination (the effective satellite
abundance) is well-determined.  ``log10mmin`` and ``sigma_logm`` also show the
expected central-HOD degeneracy: a narrower central step can be offset by a lower
:math:`M_\mathrm{min}`.  Despite these degeneracies the model successfully reproduces
the BOSS CMASS clustering signal within statistical uncertainties.

See :ref:`benchmarks` for the full suite summary.
