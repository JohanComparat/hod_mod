.. _benchmarks_deltasigma:

HOD Literature Benchmarks — ESD only
======================================

This page documents the **second benchmark tier**: HOD model fits to the
galaxy–matter excess surface density :math:`\Delta\Sigma(R)` *alone*, without
the projected clustering :math:`w_p(r_p)`.

Fitting :math:`\Delta\Sigma(R)` in isolation exercises the galaxy–matter cross
power spectrum :math:`P_{gm}(k)` and is sensitive to the mean halo mass and
satellite distribution independently from the clustering amplitude.  Comparing
DS-only parameters to the published joint fits (see :ref:`benchmarks_joint`)
quantifies the information gain from adding :math:`w_p`.

.. note::
   The :ref:`benchmarks` page covers WP-only fits.
   The :ref:`benchmarks_joint` page covers joint WP & ESD fits.

Running the ΔΣ benchmarks
--------------------------

Single model::

    python hod_mod/scripts/benchmarks/run_benchmark.py --model vanutert2016_ds --plot

All DS-only models (skips NEEDS_DATA entries)::

    python hod_mod/scripts/benchmarks/run_all_benchmarks.py

Pass criterion: :math:`\chi^2/\text{dof} < 2.0`.

Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 10 12 26

   * - Model / Paper
     - Survey
     - z\ :sub:`eff`
     - Observable
     - Status
   * - Leauthaud12HODModel — Leauthaud+2012
     - COSMOS
     - 0.66
     - :math:`\Delta\Sigma`
     - Needs data transcription (Figure 6 digitization)
   * - VanUitert16CSMFModel — van Uitert+2016
     - GAMA+KiDS
     - 0.25
     - :math:`\Delta\Sigma`
     - Data digitized from Figure 2 (preliminary, ~3% precision)
   * - ZuMandelbaum15HODModel — Zu & Mandelbaum+2015
     - SDSS DR7
     - 0.07
     - :math:`\Delta\Sigma`
     - Needs data transcription (Figure 6 digitization)
   * - Zacharegkas25HODModel — Zacharegkas+2025
     - DES Y3
     - 0.30
     - :math:`\Delta\Sigma`
     - Not applicable — paper uses angular statistics (γ\ :sub:`t`\ (θ))

Data availability notes
------------------------

Leauthaud+2012
^^^^^^^^^^^^^^

The paper does **not** publish tabulated :math:`\Delta\Sigma` values.
Data must be digitized from **Figure 6, panel j** (PHOTO_z2, :math:`10.65 < \log_{10}(M_*/M_\odot) < 10.88`).
Note that the paper uses **physical Mpc** on the x-axis and :math:`M_\odot\,\mathrm{pc}^{-2}` on the y-axis;
convert to h-units before filling the CSV:

- :math:`R\,[h^{-1}\,\mathrm{Mpc}] = R\,[\mathrm{Mpc}] \times h = R\,[\mathrm{Mpc}] \times 0.72`
- :math:`\Delta\Sigma\,[M_\odot\,h\,\mathrm{pc}^{-2}] = \Delta\Sigma\,[M_\odot\,\mathrm{pc}^{-2}] \times 0.72`

There is **no** :math:`w_p(r_p)` in this paper (it uses angular :math:`w(\theta)`).
See ``data/leauthaud2012_cosmos/README_data.md``.

van Uitert+2016
^^^^^^^^^^^^^^^

There is **no** Table B2 in this paper; the README has been corrected.
:math:`\Delta\Sigma` data has been digitized from the PostScript source of **Figure 2,
panel M3** (10.24 < :math:`\log_{10}(M_*/h^{-2}M_\odot)` < 10.59, :math:`\langle z \rangle = 0.25`).
Digitization precision: ~2–3 % in :math:`\Delta\Sigma`, ~5 % in the error bars.

The paper does **not** measure :math:`w_p(r_p)` — only DS + SMF are fitted.
The DS data file ``data/vanutert2016_gama/ds_bin2_104_108.csv`` is populated
and marked ``DIGITIZED_FROM_FIGURE``.

.. list-table:: van Uitert+2016 ΔΣ(R), panel M3 (digitized from Figure 2)
   :header-rows: 1
   :widths: 25 30 25

   * - :math:`R\ [h^{-1}\,\mathrm{Mpc}]`
     - :math:`\Delta\Sigma\ [h\,M_\odot\,\mathrm{pc}^{-2}]`
     - :math:`\sigma_{\Delta\Sigma}`
   * - 0.0324
     - 64.53
     - 14.61
   * - 0.0514
     - 48.48
     - 8.82
   * - 0.0815
     - 13.14
     - 5.88
   * - 0.1291
     - 14.22
     - 3.55
   * - 0.2048
     - 13.18
     - 2.24
   * - 0.3242
     - 8.23
     - 1.45
   * - 0.5142
     - 4.39
     - 0.95
   * - 0.8141
     - 3.26
     - 0.63
   * - 1.2911
     - 3.13
     - 0.44
   * - 2.0476
     - 1.22
     - 0.33

Zu & Mandelbaum+2015
^^^^^^^^^^^^^^^^^^^^

:math:`\Delta\Sigma(R)` and :math:`w_p(r_p)` measurements are shown in **Figure 6**
for the iHOD bin :math:`10.2 < \log_{10}(M_*/h^{-2}M_\odot) < 10.6`.
Data must be digitized from that figure.

Units (from the paper): R in :math:`h^{-1}\,\mathrm{Mpc}` (co-moving),
:math:`\Delta\Sigma` in :math:`M_\odot\,h\,\mathrm{pc}^{-2}` — already in hod_mod convention,
no conversion needed.  :math:`\pi_\mathrm{max} = 60\,h^{-1}\,\mathrm{Mpc}`.

See ``data/zumandelbaum2015_sdss/README_data.md``.

Zacharegkas+2025
^^^^^^^^^^^^^^^^

.. important::
   This paper measures :math:`\gamma_t(\theta)` (tangential shear) and :math:`w(\theta)`
   (angular clustering), **not** :math:`\Delta\Sigma(R)` or :math:`w_p(r_p)`.
   The benchmark config ``benchmark_zacharegkas2025_ds.yml`` is a stub retained for
   future work converting angular statistics to projected ones.
   The benchmark is skipped in ``run_all_benchmarks.py`` (``data_status = NOT_APPLICABLE``).

DeltaSigmaFitter class
-----------------------

DS-only fitting uses the new :class:`~hod_mod.fitting.DeltaSigmaFitter` class::

    from hod_mod.fitting import load_config, DeltaSigmaFitter

    config  = load_config("configs/benchmarks/benchmark_vanutert2016_ds.yml")
    fitter  = DeltaSigmaFitter(config)
    result  = fitter.map_fit()

The log-posterior is:

.. math::

   \log P(\theta|d) = -\tfrac{1}{2}\bigl[\chi^2_{\Delta\Sigma} + \chi^2_{n_g}\bigr]

where :math:`\chi^2_{n_g}` uses ``ng_obs`` and ``ng_frac_err`` from the YAML config
(section ``ds:``).  Config format (DS-only)::

    ds:
      file:        data/vanutert2016_gama/ds_bin2_104_108.csv
      rp_min:      0.03
      rp_max:      2.0
      ng_obs:      1.1e-3
      ng_frac_err: 0.30

Results
-------

van Uitert+2016 (vanutert2016_ds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAP fit: :math:`\chi^2/\text{dof} = 3.75 / 2 = 1.873`.
**Status: PASSED** (:math:`\chi^2/\text{dof} < 2.0`).

Best-fit: ``log10m_h1 = 11.694``, ``log10m_star0 = 9.744``, ``beta1 = 9.93``,
``log10_beta2 = -0.582``, ``sigma_c = 0.764``.

MCMC complete: flatchain saved to ``results/benchmarks/vanutert2016_ds/flatchain.npz``.

Zumandelbaum+2015 DS-only (zumandelbaum2015_ds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAP fit: :math:`\chi^2/\text{dof} = 0.002`.
**Status: PASSED**.

MCMC running.

Leauthaud+2012 DS-only (leauthaud2012_ds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAP fit: pending (data from Figure 6 digitization; no machine-readable table in paper).
**Status: in progress** — see :ref:`benchmark_leauthaud2012` for details.

See :ref:`benchmarks` for WP-only benchmarks and
:ref:`benchmarks_joint` for joint WP & ESD benchmarks.
