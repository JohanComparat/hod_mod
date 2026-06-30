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
     - :math:`z \in [0.05, 0.18]`,  :math:`z_{\rm eff} = 0.136`
   * - Galaxy count
     - 2,759,238
   * - :math:`w_p` bins
     - 30 data bins (:math:`r_p \in [{\sim}0.008, 60]\,h^{-1}\,\text{Mpc}`);
       17–29 used in fits (:math:`r_{p,\rm max} = 50\,h^{-1}\,\text{Mpc}`)
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
* **Beyond-linear halo bias** — Mead & Verde 2021 (arXiv:`2011.08858
  <https://arxiv.org/abs/2011.08858>`_) additive correction to the 2-halo
  galaxy–galaxy and galaxy–matter power spectra, using tabulated
  :math:`\beta^{\rm NL}(k,\nu_1,\nu_2)` from the MultiDark MDR1 N-body
  simulation.  The linear power spectrum is used for the 2-halo term throughout
  (following More+2015); the BNL correction is applied on top.
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
     - More et al. 2015 (`arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_)
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
     - 0.04
     - 0.04
     - 0.16
     - 0.20
     - 0.65
     - 0.09
     - 3.87
     - 3.39
     - 46.2
     - 51.0
   * - Zheng+2007
     - 0.04
     - 0.04
     - 0.09
     - 0.06
     - 0.50
     - 0.13
     - 2.98
     - 3.39
     - 9.56
     - 14.8
   * - Kravtsov+2004
     - 0.04
     - 0.04
     - 0.04
     - 0.06
     - 0.12
     - 0.58
     - 2.85
     - 3.31
     - 13.2
     - 15.3
   * - Zu & Mandelbaum 2015
     - 0.07
     - 0.06
     - 0.22
     - 0.31
     - 0.64
     - 0.59
     - **2.21**
     - 2.83
     - 19.2
     - 23.0
   * - van Uitert+2016
     - 0.11
     - 0.13
     - 0.39
     - 0.36
     - 0.69
     - 0.62
     - 6.22
     - 3.63
     - 12.8
     - 35.1
   * - Zacharegkas+2025
     - 0.11
     - 0.10
     - 0.22
     - 0.10
     - 0.38
     - 0.57
     - 3.14
     - 2.54
     - 23.8
     - 29.3

.. |br| raw:: html

   <br/>


Figures
-------

.. figure:: /_images/fig_wp_survey_predictions.png
   :width: 100%
   :align: center

   BGS LS10 :math:`w_p(r_p)` data (black points) and all MAP best-fit model
   predictions at each of the five scale cuts.  Each column corresponds to one
   :math:`r_{p,\rm min}` threshold (indicated by a vertical dotted line).
   Solid lines = NFW profile; dashed = Einasto.  Colours follow the model
   legend in each panel.  Lower sub-panels show the ratio
   :math:`w_p^{\rm pred} / w_p^{\rm data}`.

.. figure:: /_images/fig_wp_survey_chi2.png
   :width: 85%
   :align: center

   :math:`\chi^2/n_{\rm dof}` heatmap for all 6 models × 5 scale cuts,
   shown separately for NFW (left) and Einasto (right) profiles.
   Green cells indicate good fits; red cells indicate poor fits.

.. figure:: /_images/fig_shmr_comparison.png
   :width: 80%
   :align: center

   Stellar-to-halo mass relations inferred from the MAP fits at
   :math:`r_p > 0.05\,h^{-1}\,\text{Mpc}` (best-constrained scale cut).
   Solid lines = NFW; dashed = Einasto.  Models with an explicit SHMR
   (Zu & Mandelbaum 2015, Zacharegkas+2025, van Uitert+2016) are shown as
   continuous curves; threshold HODs (More+2015, Zheng+2007, Kravtsov+2004)
   are shown as single markers at :math:`(\log_{10}M_{\rm min},\,10.0)` —
   their effective halo-mass pivot for the :math:`\log_{10}(M_*/M_\odot)>10`
   stellar-mass threshold (dotted horizontal line).

Key findings
------------

Scale-cut transitions
~~~~~~~~~~~~~~~~~~~~~

* **:math:`r_p > 0.30\,h^{-1}\,\text{Mpc}`** — All models fit well
  (:math:`\chi^2/n_{\rm dof} \approx 0.04`–0.13).  Two-halo term dominated;
  model is effectively a linear bias measurement.

* **:math:`r_p > 0.05\,h^{-1}\,\text{Mpc}`** — All models still fit
  (:math:`\chi^2/n_{\rm dof} < 0.4`).  Einasto outperforms NFW for more2015
  (0.20 vs 0.16) and zacharegkas25 (0.10 vs 0.22); Zheng+2007 and Kravtsov+2004
  reach 0.04–0.09 with NFW.

* **:math:`r_p > 0.04\,h^{-1}\,\text{Mpc}`** — Models begin to diverge.
  Kravtsov+2004 NFW (0.12) and more2015 Einasto (0.09) are the best fits;
  more2015 NFW degrades to 0.65.

* **:math:`r_p > 0.02\,h^{-1}\,\text{Mpc}`** — All models struggle
  (:math:`\chi^2/n_{\rm dof} = 2.2`–6.2).  Model-data tension builds in the
  1-halo regime.  Zu & Mandelbaum 2015 NFW is the best model at 2.21.

* **:math:`r_p > 0.01\,h^{-1}\,\text{Mpc}`** — All models fail badly
  (:math:`\chi^2/n_{\rm dof} = 9.6`–51).  The inner 10 kpc/:math:`h`
  sub-halo regime is not described by any standard satellite profile.

NFW vs Einasto
~~~~~~~~~~~~~~

The profile comparison is model-dependent.  For more2015 at
:math:`r_p > 0.04`, Einasto (0.09) is much better than NFW (0.65), while
for Kravtsov+2004 at the same cut the ordering reverses (NFW 0.12, Einasto
0.58).  Zheng+2007 and zacharegkas25 perform similarly under both profiles
at :math:`r_p > 0.05`.  At large scales (:math:`r_p > 0.30`) all models
converge to :math:`\chi^2/n_{\rm dof} \approx 0.04`–0.13 regardless of
profile.

van Uitert+2016 and Zacharegkas+2025
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both models are fully run (all 10 combinations each) following the
``self._bias`` fix described below.

* **van Uitert+2016** fits well at :math:`r_p > 0.30` and :math:`r_p > 0.05`
  (:math:`\chi^2/n_{\rm dof} \approx 0.11`–0.39) but fails at
  :math:`r_p > 0.02` (NFW 6.22, Einasto 3.63).  Einasto significantly
  outperforms NFW for this model at small scales, opposite to simpler HODs.

* **Zacharegkas+2025** achieves the best fits at :math:`r_p > 0.04` for
  NFW (0.38) and among the best at :math:`r_p > 0.02` (NFW 3.14, Einasto
  2.54).  At :math:`r_p > 0.05`, zacharegkas25 Einasto (0.10) ties with
  Zheng+2007 Einasto (0.06) for the lowest :math:`\chi^2/n_{\rm dof}`.

Bug fixes during this campaign
--------------------------------

Two models were not functional prior to this survey:

* **vanuitert16** and **zacharegkas25**: their ``__init__`` methods stored
  ``self._hmf = hmf`` but not ``self._bias = hmf.bias``.
  ``FullHaloModelPrediction`` calls ``hod._bias(m, z, theta_cosmo)`` directly
  (in ``hod_mod/observables/clustering.py``), so the missing attribute
  caused an ``AttributeError`` at runtime, leaving the optimizer without valid
  evaluations and returning :math:`\chi^2 = \infty`.

  Fix: added ``self._bias = hmf.bias`` to both ``__init__`` methods in
  ``hod_mod/connection/hod/``.

  **Status:** fixed; results for both models are fully included in the table above.

Recommendation for joint :math:`w_p` + X-ray cross-correlation
---------------------------------------------------------------

For jointly modelling :math:`w_p(r_p)` and the galaxy × eROSITA X-ray angular
cross-correlation :math:`w(\theta)`:

**Primary: Zu & Mandelbaum 2015 NFW at** :math:`r_p > 0.02\,h^{-1}\,\text{Mpc}`

* Best :math:`\chi^2/n_{\rm dof} = 2.21` at :math:`r_p > 0.02` (best of all
  models at small scales).
* Inverse SHMR framework maps the stellar-mass threshold directly to a halo mass
  distribution — this ties naturally to the X-ray gas emissivity model via
  :math:`\varepsilon \propto n_e^2(r\,|\,M_{200})` (``GasDensityDPM``).
* Already validated for this exact cross-correlation in
  ``hod_mod/scripts/validate_comparat2025.py`` (LS DR10 × eRASS:5 soft X-ray,
  0.5–2 keV), which uses ``ZuMandelbaum15HODModel + GasDensityDPM`` across 7
  stellar-mass bins.
* 6 HOD free parameters — tractable for MCMC with a joint covariance.

**Alternative: Zacharegkas+2025 Einasto at** :math:`r_p > 0.04\,h^{-1}\,\text{Mpc}`

* :math:`\chi^2/n_{\rm dof} = 0.57` — excellent WPRP fit through the full
  1-halo transition.
* Kravtsov+2018 SHMR is physically motivated by N-body simulations and provides
  an accurate mass-dependent satellite normalisation.
* 8 HOD free parameters; Einasto profile preferred over NFW for this model.
* Trade-off: the :math:`r_p > 0.04` cut avoids the innermost 40 kpc/:math:`h`,
  which may under-constrain the satellite concentration in a joint fit.

**Not recommended: More+2015, Zheng+2007, Kravtsov+2004** for the joint fit —
these are threshold HODs without an explicit SHMR.  Connecting them to the X-ray
gas emissivity requires an independent mass–observable relation, introducing
degeneracies between the HOD and gas-profile parameters.

Path forward
------------

1. **Satellite extension survey** — run ``--use-sat-ext`` for all 6 models
   and both profiles at :math:`r_p > 0.02` to assess whether reduced satellite
   concentration (:math:`b_{\rm sat,conc} < 1`) is a universal correction::

      python scripts/fitting/bgs_ls10/fit_bgs_multiprobe.py \
          --mstar 10.0 --probes wp --use-ia --use-baryon-fraction \
          --use-offcentering --use-sat-ext --map-only \
          --hod-model <model> --profile <nfw|einasto> --rp-min-wp 0.02

2. **MCMC posteriors** for the best-fitting models (Zu & Mandelbaum 2015 NFW,
   zacharegkas25 Einasto, Kravtsov+2004 NFW at :math:`r_p > 0.02`) to quantify
   parameter uncertainties.
3. **ESD systematics investigation** — the ESD amplitude is mis-predicted
   by all models at fixed Planck cosmology (see :doc:`fitting` for context);
   requires lensing calibration study before joint :math:`w_p` + ESD fitting.

Per-model best-fit parameters
------------------------------

For each HOD model the following two figures are shown: (1) the projected
correlation function :math:`w_p(r_p)` at all five scale cuts overlaid on the
BGS LS10 data, coloured by :math:`r_{p,\rm min}` (green = large scales,
red = small scales); (2) the MAP parameter values as a function of the minimum
scale :math:`r_{p,\rm min}`, with NFW (filled circles / solid) and Einasto
(open squares / dashed) shown separately.  Physics flags active for all runs:
off-centering (:math:`f_{\rm off}`, :math:`\sigma_{\rm off}`), NLA intrinsic
alignment (:math:`A_{\rm IA}`), and mass-dependent baryon fraction
(:math:`\log_{10}M_{\rm pivot}`, :math:`\beta_b`, :math:`\log_{10}\eta_{\rm min}`).

.. include:: _permodel_auto.rst

Output files
------------

All results are stored under ``hod_mod/results/bgs_multiprobe/``.
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

* More et al. 2015 — `arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_
* Zheng et al. 2007 — `arXiv:astro-ph/0703457 <https://arxiv.org/abs/astro-ph/0703457>`_
* Kravtsov et al. 2004 — `ApJ 609, 35 <https://doi.org/10.1086/420959>`_
* Zu & Mandelbaum 2015 — `arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_
* van Uitert et al. 2016 — `arXiv:1601.06791 <https://arxiv.org/abs/1601.06791>`_
* Zacharegkas & Chang et al. 2025 — `arXiv:2506.22367 <https://arxiv.org/abs/2506.22367>`_
* Johnston et al. 2007 — `arXiv:0709.4193 <https://arxiv.org/abs/0709.4193>`_
* Bridle & King 2007 — `arXiv:0705.0166 <https://arxiv.org/abs/0705.0166>`_
* FLAMINGO — `arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_
* DESI BGS — Hahn et al. 2023 `arXiv:2208.08512 <https://arxiv.org/abs/2208.08512>`_
