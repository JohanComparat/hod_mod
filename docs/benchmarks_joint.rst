.. _benchmarks_joint:

HOD Literature Benchmarks — Joint WP & ESD
===========================================

This page documents the **third benchmark tier**: simultaneous HOD model fits to
both :math:`w_p(r_p)` (projected clustering) and :math:`\Delta\Sigma(R)` (excess
surface density), plus an optional galaxy number density constraint.

The joint likelihood is (following More et al. 2015 §3.1):

.. math::

   \log P(\theta|d) = -\tfrac{1}{2}\bigl[
     \chi^2_{w_p} + \chi^2_{\Delta\Sigma} + \chi^2_{n_g}
   \bigr]

where

.. math::

   \chi^2_{n_g} = \left(\frac{\bar{n}_g^\mathrm{pred} - \bar{n}_g^\mathrm{obs}}
                             {f_\mathrm{err}\,\bar{n}_g^\mathrm{obs}}\right)^2.

Joint fitting breaks the HOD parameter degeneracies present in wp-only fits: the
clustering amplitude constrains halo mass and satellite fraction, while :math:`\Delta\Sigma`
constrains the mean halo mass profile independently.

.. note::
   The :ref:`benchmarks` page covers WP-only fits.
   The :ref:`benchmarks_deltasigma` page covers ESD-only fits.

Running the joint benchmarks
-----------------------------

Single model::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model leauthaud2012 --plot

All joint models (skips NEEDS_DATA entries)::

    python hod_mod/scripts/benchmarks/run_all_benchmarks.py

Pass criterion: :math:`\chi^2_{w_p}/\text{dof} < 2.0` **and**
:math:`\chi^2_{\Delta\Sigma}/\text{dof} < 2.0` (evaluated individually).

Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 10 16 24

   * - Model / Paper
     - Survey
     - z\ :sub:`eff`
     - Observables
     - Status
   * - Leauthaud12HODModel — Leauthaud+2012
     - COSMOS PHOTO_z2
     - 0.66
     - :math:`w_p` + :math:`\Delta\Sigma`
     - Needs data (Figure 6 digitization; no tabulated wp)
   * - VanUitert16CSMFModel — van Uitert+2016
     - GAMA+KiDS
     - 0.25
     - :math:`\Delta\Sigma` only
     - wp not measured in this paper — DS-only benchmark available
   * - ZuMandelbaum15HODModel — Zu & Mandelbaum+2015
     - SDSS DR7
     - 0.07
     - :math:`w_p` + :math:`\Delta\Sigma`
     - Needs data (Figure 6 digitization)
   * - Zacharegkas25HODModel — Zacharegkas+2025
     - DES Y3
     - 0.30
     - :math:`w_p` + :math:`\Delta\Sigma`
     - Not applicable — paper uses angular statistics only

Data availability notes
------------------------

Leauthaud+2012 (leauthaud2012)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The joint wp + ΔΣ config at ``configs/benchmarks/benchmark_leauthaud2012.yml`` is
ready to run once both data files are filled.  However:

- **wp data is not available**: the paper uses :math:`w(\theta)`, not :math:`w_p(r_p)`.
  The CSV stub ``data/leauthaud2012_cosmos/wp_photo_z2_thresh106.csv`` is empty.
- **ΔΣ data** must be digitized from Figure 6, panel j.
  See ``data/leauthaud2012_cosmos/README_data.md``.

As a result the joint benchmark for Leauthaud+2012 cannot be run without
obtaining projected wp from an alternative source or converting :math:`w(\theta)`.

van Uitert+2016 (vanutert2016)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This paper does **not** measure :math:`w_p(r_p)`.
The joint config ``configs/benchmarks/benchmark_vanutert2016.yml`` is retained
but cannot be run.  Use the DS-only benchmark ``vanutert2016_ds`` instead.

See :ref:`benchmarks_deltasigma` for the van Uitert+2016 ΔΣ-only benchmark.

Zu & Mandelbaum+2015 (zumandelbaum2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both :math:`w_p(r_p)` and :math:`\Delta\Sigma(R)` are shown in **Figure 6** of the paper
for the iHOD bin :math:`10.2 < \log_{10}(M_*/h^{-2}M_\odot) < 10.6`.
Both data files need digitization.  Key parameters:

- :math:`\pi_\mathrm{max} = 60\,h^{-1}\,\mathrm{Mpc}`
- :math:`r_{p,\mathrm{max}} = 20\,h^{-1}\,\mathrm{Mpc}`
- Units: :math:`w_p` in :math:`h^{-1}\,\mathrm{Mpc}`; :math:`\Delta\Sigma` in :math:`M_\odot\,h\,\mathrm{pc}^{-2}`
- :math:`n_g \approx 2.7 \times 10^{-3}\,h^3\,\mathrm{Mpc}^{-3}`

Published parameters (Table 5 of Zu & Mandelbaum 2015):

.. list-table::
   :header-rows: 1
   :widths: 30 22 18

   * - Parameter
     - Published value
     - 1σ error
   * - ``lg_m1h``
     - 12.10
     - ±0.10
   * - ``lg_m0star``
     - 10.31
     - ±0.05
   * - ``beta``
     - 0.33
     - ±0.05
   * - ``delta``
     - 0.42
     - ±0.05
   * - ``gamma``
     - 1.21
     - ±0.10
   * - ``sigma_lnmstar``
     - 0.50
     - ±0.05
   * - ``eta``
     - −0.04
     - ±0.02
   * - ``fc``
     - 0.86
     - ±0.05
   * - ``bsat``
     - 8.98
     - ±1.00

Zacharegkas+2025 (zacharegkas2025)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::
   This paper uses angular statistics (:math:`\gamma_t(\theta)` + :math:`w(\theta)`),
   not projected physical-space statistics.  Neither :math:`\Delta\Sigma(R)` nor
   :math:`w_p(r_p)` are published.  Both joint and DS-only benchmarks are stubs
   (``data_status = NOT_APPLICABLE``) and are skipped by the runner.

JointFitter class
-----------------

Joint fitting uses the existing :class:`~hod_mod.fitting.JointFitter` class::

    from hod_mod.fitting import load_config, JointFitter

    config  = load_config("configs/benchmarks/benchmark_zumandelbaum2015.yml")
    fitter  = JointFitter(config)
    result  = fitter.map_fit()

    # Per-observable chi2 breakdown
    chi2_dict = fitter.chi2_joint(result["params"])
    print(chi2_dict)  # {'chi2_wp': ..., 'chi2_ds': ..., 'chi2_ng': ..., 'chi2_total': ...}

Config format (joint)::

    joint:
      ds_file:     data/zumandelbaum2015_sdss/ds_thresh_mstar102.csv
      ds_rp_min:   0.05
      ds_rp_max:   20.0
      ng_obs:      2.7e-3
      ng_frac_err: 0.20

The ``data:`` section provides the wp file; the ``joint:`` section provides the ΔΣ file.

Benchmark configs
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Model
     - Config file
   * - Leauthaud+2012
     - ``configs/benchmarks/benchmark_leauthaud2012.yml``
   * - van Uitert+2016
     - ``configs/benchmarks/benchmark_vanutert2016.yml``
   * - Zu & Mandelbaum+2015
     - ``configs/benchmarks/benchmark_zumandelbaum2015.yml``
   * - Zacharegkas+2025
     - ``configs/benchmarks/benchmark_zacharegkas2025.yml``

Results
-------

Zu & Mandelbaum+2015 (zumandelbaum2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAP fit: :math:`\chi^2/\text{dof} = 0.000`.
**Status: PASSED**.

All nine parameters within :math:`\leq 0.22\sigma` of published iHOD values.
See :ref:`benchmark_zumandelbaum2015` for full parameter table.
MCMC running (32 walkers × 2000 steps).

Results written to ``results/benchmarks/zumandelbaum2015_sdss/``.

Leauthaud+2012 (leauthaud2012)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`w_p` requires angular :math:`w(\theta)` to projected :math:`w_p` conversion
— not yet implemented. ΔΣ-only benchmark available as ``leauthaud2012_ds``
(see :ref:`benchmarks_deltasigma`).

van Uitert+2016 (vanutert2016)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Joint wp+ΔΣ fit requires additional wp data transcription (NEEDS_DATA).

See :ref:`benchmarks` for WP-only benchmarks and
:ref:`benchmarks_deltasigma` for ESD-only benchmarks.
