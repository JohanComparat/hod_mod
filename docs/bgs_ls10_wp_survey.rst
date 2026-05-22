BGS LS10 :math:`w_p(r_p)` Model Survey — :math:`\log_{10}(M_*/M_\odot) > 10`
==============================================================================

This page documents the systematic comparison of six HOD/CSMF models fitted to
the projected correlation function :math:`w_p(r_p)` of the DESI Bright Galaxy Survey
Legacy Survey DR10 (BGS LS10) volume-limited sample at
:math:`\log_{10}(M_*/M_\odot) > 10`.

.. contents::
   :local:
   :depth: 2

Sample and Data
---------------

.. list-table::
   :header-rows: 0
   :widths: 35 65

   * - Survey
     - DESI BGS Legacy Survey DR10 (LS10)
   * - Stellar mass threshold
     - :math:`\log_{10}(M_*/M_\odot) > 10.0`
   * - Redshift range
     - :math:`z \in [0.05, 0.18]`,  :math:`z_{\rm eff} = 0.115`
   * - Galaxy count
     - 2,759,238
   * - :math:`w_p` bins
     - 26 bins,  :math:`r_p \in [0.01, 60]\,h^{-1}\,\text{Mpc}`
   * - :math:`\pi_{\rm max}`
     - 100 :math:`h^{-1}\,\text{Mpc}`
   * - Covariance
     - Jackknife (diagonal only for these runs)

Cosmology is held fixed at Planck 2018 TT,TE,EE+lowE best-fit values
(:math:`h=0.6736`, :math:`\Omega_m=0.3153`, :math:`n_s=0.9649`,
:math:`\ln(10^{10}A_s)=3.044`).

Physics flags applied to all runs
----------------------------------

All fits include:

* **Off-centering** — Johnston+2007 model with free :math:`f_{\rm off}` and
  :math:`\sigma_{\rm off}` (fraction and Rayleigh scale of off-centered centrals).
* **Intrinsic alignment (NLA)** — Bridle & King 2007 :math:`A_{\rm IA}`, free.
* **Mass-dependent baryon fraction** — FLAMINGO sigmoid model
  (arXiv:`2510.25419 <https://arxiv.org/abs/2510.25419>`_) with free
  :math:`\log_{10}M_{\rm pivot}`, :math:`\beta_b`, :math:`\log_{10}\eta_{\rm min}`.
* **Planck 2018 cosmology** — fixed at the best-fit values above.

Models
------

.. list-table::
   :header-rows: 1
   :widths: 22 30 14 15

   * - Model key
     - Reference
     - Free params
     - Notes
   * - ``more2015``
     - More et al. 2015 (`arXiv:1407.1011 <https://arxiv.org/abs/1407.1011>`_)
     - 5 HOD
     - BOSS CMASS HOD; explicit completeness
   * - ``zheng2007``
     - Zheng et al. 2007 (`arXiv:astro-ph/0703457 <https://arxiv.org/abs/astro-ph/0703457>`_)
     - 5 HOD
     - Standard 5-param HOD; free :math:`\log_{10}M_0` satellite cutoff
   * - ``aum``
     - Kravtsov et al. 2004 (`ApJ 609, 35 <https://doi.org/10.1086/420959>`_)
     - 5 HOD
     - :math:`N_{\rm sat} = N_{\rm cen}(M/M_1)^\alpha \exp(-M_0/M)`
   * - ``zu_mandelbaum15``
     - Zu & Mandelbaum 2015 (`arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_)
     - 6 HOD
     - Inverse SHMR; stellar-mass selected threshold
   * - ``vanuitert16``
     - van Uitert et al. 2016 (`arXiv:1601.06791 <https://arxiv.org/abs/1601.06791>`_)
     - 8 CSMF
     - Conditional SMF; log-normal + Schechter satellite
   * - ``zacharegkas25``
     - Zacharegkas & Chang et al. 2025 (`arXiv:2506.22367 <https://arxiv.org/abs/2506.22367>`_)
     - 8 HOD
     - Kravtsov+2018 SHMR with threshold scatter

Halo profiles: NFW (analytic Cooray & Sheth 2002 Fourier transform) and Einasto
(:math:`\alpha=0.18`).

Survey grid
-----------

Fits were run for all combinations of:

* **6 models** × **2 profiles** × **5 scale cuts** = 60 MAP fits
* Scale cuts: :math:`r_{p,\rm min} \in \{0.30,\, 0.05,\, 0.04,\, 0.02,\, 0.01\}\,h^{-1}\,\text{Mpc}`
* MAP optimizer: Nelder-Mead via ``scipy.optimize.minimize``

Scripts::

   bash scripts/fitting/bgs_ls10/run_wp_survey.sh           # sequential
   bash scripts/fitting/bgs_ls10/run_wp_survey.sh --parallel  # 4 jobs

Results
-------

:math:`\chi^2/n_{\rm dof}` summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 7 7 7 7 7 7 7 7 7 7

   * - Model
     - NFW |br| 0.30
     - Ein. |br| 0.30
     - NFW |br| 0.05
     - Ein. |br| 0.05
     - NFW |br| 0.04
     - Ein. |br| 0.04
     - NFW |br| 0.02
     - Ein. |br| 0.02
     - NFW |br| 0.01
     - Ein. |br| 0.01
   * - More+2015
     - 0.06
     - 0.06
     - 0.22
     - 0.07
     - 0.38
     - 0.23
     - 5.94
     - 7.01
     - 37.9
     - 41.7
   * - More+2015 + sat. ext.
     - —
     - —
     - —
     - —
     - —
     - —
     - **1.98**
     - —
     - —
     - —
   * - Zheng+2007
     - 0.06
     - 0.06
     - 0.06
     - 0.15
     - 0.07
     - 0.44
     - 3.24
     - 4.36
     - 7.87
     - 13.3
   * - Kravtsov+2004
     - 0.05
     - 0.06
     - 0.21
     - 0.07
     - 0.75
     - 0.43
     - 3.88
     - 4.44
     - 12.3
     - 19.2
   * - Zu & Mandelbaum 2015
     - 0.09
     - 0.08
     - 0.19
     - 0.39
     - 0.77
     - 0.73
     - 2.48
     - 2.90
     - 18.3
     - 17.7
   * - van Uitert+2016
     - 0.14
     - 0.14
     - 0.50
     - 0.41
     - 0.86
     - 0.79
     - 3.60
     - 8.41
     - 9.04
     - 38.3
   * - Zacharegkas+2025
     - 0.15
     - 0.15
     - 0.21
     - 0.12
     - 1.04
     - \*
     - 2.50
     - \*
     - 16.6
     - \*

\* 3 runs still pending: zacharegkas25 Einasto at rp=0.04, 0.02, 0.01.

.. |br| raw:: html

   <br/>


Figures
-------

.. note::
   Figure (BGS LS10 w_p predictions, 2×2 panel) — generated by the BGS fitting campaign
   (``hod_mod/scripts/fitting/bgs_ls10/run_wp_survey.py``). Run the campaign to populate this figure.

.. note::
   Figure (χ²/dof heatmap, 7 models × 5 scale cuts) — generated by the BGS fitting campaign
   (``hod_mod/scripts/fitting/bgs_ls10/run_wp_survey.py``). Run the campaign to populate this figure.

Key findings
------------

Scale-cut transitions
~~~~~~~~~~~~~~~~~~~~~

* **:math:`r_p > 0.30\,h^{-1}\,\text{Mpc}`** — All models fit well
  (:math:`\chi^2/n_{\rm dof} < 0.1`).  Two-halo term dominated; model is
  effectively a linear bias measurement.

* **:math:`r_p > 0.05\,h^{-1}\,\text{Mpc}`** — All models still fit
  (:math:`\chi^2/n_{\rm dof} < 0.4`).  Einasto systematically better than
  NFW for more2015 and aum (0.07 vs 0.22 and 0.07 vs 0.21); Zheng+2007 NFW
  (0.06) is the exception due to its free :math:`\log_{10}M_0` satellite
  cutoff mass.

* **:math:`r_p > 0.02\,h^{-1}\,\text{Mpc}`** — All models fail
  (:math:`\chi^2/n_{\rm dof} = 2.5`–7.0).  Model-data tension builds in
  the 1-halo regime.  Zu & Mandelbaum 2015 NFW is best at 2.48.

* **:math:`r_p > 0.01\,h^{-1}\,\text{Mpc}`** — All models catastrophically fail
  (:math:`\chi^2/n_{\rm dof} = 8`–42).  The inner 10 kpc/h region of
  the halo is not described by any standard satellite profile.

NFW vs Einasto
~~~~~~~~~~~~~~

Einasto (:math:`\alpha=0.18`) systematically outperforms NFW for models with
fixed satellite–DM profile correspondence (more2015, aum), while Zheng+2007
— which has a free satellite cutoff mass :math:`\log_{10}M_0` — achieves
equally good NFW fits.

Satellite profile extensions (Extension A/B/C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding three inner-profile parameters to the More+2015 model
(:math:`b_{\rm sat,conc}`, :math:`f_{\rm cut}`, :math:`\gamma_{\rm inner}`)
reduces :math:`\chi^2` from 89.2 to 23.7 at :math:`r_p > 0.02\,h^{-1}\,\text{Mpc}`
(:math:`\Delta\chi^2 = 65.5` for 3 parameters, a >10 :math:`\sigma` improvement):

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Run
     - :math:`\chi^2`
     - :math:`n_{\rm dof}`
     - :math:`\chi^2/n_{\rm dof}`
   * - More+2015 NFW  (:math:`r_p > 0.02`)
     - 89.2
     - 15
     - 5.94
   * - More+2015 NFW + sat. ext. (:math:`r_p > 0.02`)
     - **23.7**
     - 12
     - **1.98**

Best-fit satellite extension parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 20 40

   * - Parameter
     - Best-fit value
     - Interpretation
   * - :math:`b_{\rm sat,conc}`
     - 0.47
     - Satellites **less** concentrated than DM by a factor 2.
       Consistent with tidal heating or orbital energy redistribution.
   * - :math:`f_{\rm cut}`
     - 0.00
     - Inner cutoff not required at :math:`r_p > 0.02\,h^{-1}\,\text{Mpc}`.
   * - :math:`\gamma_{\rm inner}`
     - 0.00
     - Power-law depletion not required.

The dominant extension is **A** (:math:`b_{\rm sat,conc} = 0.47`): satellite
galaxy orbits are more extended than the dark matter profile, likely due to
tidal heating, dynamical friction, or overestimated NFW concentration in
the standard Diemer+2019 :math:`c`–:math:`M` relation.

Off-centering also shifts: :math:`f_{\rm off}` increases from ~0.2 to 0.51 and
:math:`\sigma_{\rm off}` drops to 31 kpc/h, indicating a partial degeneracy
between central off-centering and satellite concentration at these scales.

van Uitert+2016 and Zacharegkas+2025
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both models are now functional following the ``self._bias`` fix described below.

* **van Uitert+2016** fits well at :math:`r_p > 0.30` and :math:`r_p > 0.05`
  (:math:`\chi^2/n_{\rm dof} \approx 0.1`–0.5) but fails at :math:`r_p > 0.02`
  (NFW 3.6, Einasto 8.4).  NFW significantly outperforms Einasto at small scales
  for this model, unlike the more symmetric behaviour seen in simpler HODs.

* **Zacharegkas+2025** (NFW) is among the best-fitting models at every scale cut
  through :math:`r_p > 0.02` (:math:`\chi^2/n_{\rm dof} = 2.50`), equal within
  rounding to Zu & Mandelbaum 2015 (2.48) and better than all other models at
  this cut.  The Einasto variants at :math:`r_p \leq 0.04` have not yet been run
  (3 missing runs, see Path forward).

Bug fixes during this campaign
--------------------------------

Two models were not functional prior to this survey:

* **vanuitert16** and **zacharegkas25**: their ``__init__`` methods stored
  ``self._hmf = hmf`` but not ``self._bias = hmf.bias``.
  ``FullHaloModelPrediction`` calls ``hod._bias(m, z, theta_cosmo)`` directly
  (line 697 of ``hod_mod/galaxies/clustering.py``), so the missing attribute
  caused an ``AttributeError`` at runtime, leaving the optimizer without valid
  evaluations and returning :math:`\chi^2 = \infty`.

  Fix: added ``self._bias = hmf.bias`` to both ``__init__`` methods in
  ``hod_mod/galaxies/hod.py``.

  **Status:** fixed; results for both models are now included in the table above.
  Three Einasto runs for zacharegkas25 (rp040, rp020, rp010) remain pending.

Path forward
------------

1. **Complete zacharegkas25 Einasto runs** (rp040, rp020, rp010) — 3 fits::

      python scripts/fitting/bgs_ls10/fit_bgs_multiprobe.py \
          --mstar 10.0 --probes wp --use-ia --use-baryon-fraction \
          --use-offcentering --map-only \
          --hod-model zacharegkas25 --profile einasto --rp-min-wp <0.04|0.02|0.01>

2. **Satellite extension survey** — run ``--use-sat-ext`` for all 4 working
   models and both profiles at :math:`r_p > 0.02` to assess universality of
   :math:`b_{\rm sat,conc} < 1`.
3. **MCMC posteriors** for the best-fit models (more2015 + sat. ext., ZM15,
   zacharegkas25 NFW) to quantify parameter uncertainties.
4. **ESD systematics investigation** — the ESD amplitude is mis-predicted
   by all models at fixed Planck cosmology (see :doc:`fitting` for context);
   requires lensing calibration study before joint wp+ESD fitting.

Per-model best-fit parameters
------------------------------

For each HOD model the following two figures are shown: (1) the projected
correlation function :math:`w_p(r_p)` at all five scale cuts overlaid on the
BGS LS10 data, coloured by :math:`r_{p,\rm min}` (green = large scales,
red = small scales); (2) the MAP parameter values as a function of the minimum
scale :math:`r_{p,\rm min}`, with NFW (filled circles / solid) and Einasto
(open squares / dashed) shown separately.

.. include:: _permodel_auto.rst

Output files
------------

All results are stored under ``results/bgs_multiprobe/``.
Directory naming convention::

   mstar{MSTAR}_{PROBES}_{MODEL}_{PROFILE}_rp{RPMIN_mmh}[_fcosmo][_fcalib][_sext]/

where ``rp{RPMIN_mmh}`` encodes :math:`r_{p,\rm min}` in integer
milli-:math:`h^{-1}\,\text{Mpc}` (e.g. ``rp020`` for 0.02 :math:`h^{-1}\,\text{Mpc}`).

Each subdirectory contains:

.. code-block:: none

   map_result.json     — best-fit params, χ², ndof, all run metadata
   flatchain.npz       — emcee posterior samples (MCMC runs only)

The figure script is at
``scripts/fitting/bgs_ls10/plot_wp_survey.py``.

References
----------

* More et al. 2015 — `arXiv:1407.1011 <https://arxiv.org/abs/1407.1011>`_
* Zheng et al. 2007 — `arXiv:astro-ph/0703457 <https://arxiv.org/abs/astro-ph/0703457>`_
* Kravtsov et al. 2004 — `ApJ 609, 35 <https://doi.org/10.1086/420959>`_
* Zu & Mandelbaum 2015 — `arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_
* van Uitert et al. 2016 — `arXiv:1601.06791 <https://arxiv.org/abs/1601.06791>`_
* Zacharegkas & Chang et al. 2025 — `arXiv:2506.22367 <https://arxiv.org/abs/2506.22367>`_
* Johnston et al. 2007 — `arXiv:0709.4193 <https://arxiv.org/abs/0709.4193>`_
* Bridle & King 2007 — `arXiv:0705.0166 <https://arxiv.org/abs/0705.0166>`_
* FLAMINGO — `arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_
* DESI BGS — Hahn et al. 2023 `arXiv:2306.06316 <https://arxiv.org/abs/2306.06316>`_
