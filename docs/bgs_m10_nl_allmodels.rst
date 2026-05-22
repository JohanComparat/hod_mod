BGS M10 Non-Linear 2-Halo Survey — All HOD Models × :math:`P_{\rm nl}` Backends
===================================================================================

Systematic comparison of all nine HOD/ICSMF/CLF models with the **non-linear
2-halo term** enabled, using two independent :math:`P_{\rm nl}(k)` backends
(HMcode-2020 and Aletheia).  The goal is to quantify (i) how well each occupation
model fits the BGS M10 :math:`w_p(r_p)` signal when the quasi-linear regime
(:math:`r_p \lesssim 5\,h^{-1}\,\text{Mpc}`) is modelled with :math:`P_{\rm nl}(k)`,
and (ii) how sensitive the inferred HOD parameters are to the choice of
:math:`P_{\rm nl}` emulator.

This page extends :doc:`bgs_m10_nonlinear_models`, which covered only
More+2015 (linear and HMcode) and Leauthaud+2012 (HMcode).

**Key finding**: The Aletheia emulator systematically over-predicts :math:`w_p`
relative to the data for all nine models
(:math:`\chi^2/N_{\rm dof} = 5`–27 vs. :math:`\lesssim 1` for HMcode),
pointing to a calibration or k-range issue in the Aletheia :math:`P_{\rm nl}`
at :math:`z \approx 0.12` for this cosmology.  Only HMcode results should be
used for physical interpretation.

.. contents::
   :local:
   :depth: 2

.. |br| raw:: html

   <br/>

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
     - :math:`z \in [0.05,\,0.18]`,  :math:`z_{\rm eff} = 0.115`
   * - Galaxy count
     - 2,759,238
   * - :math:`r_p` fit range
     - 16 bins,  :math:`r_p \in [0.3,\,30]\,h^{-1}\,\text{Mpc}`
   * - :math:`\pi_{\rm max}`
     - :math:`100\,h^{-1}\,\text{Mpc}`
   * - Covariance
     - Jackknife; 1% diagonal regularisation
   * - Cosmology
     - Planck 2018 fixed
       (:math:`h=0.6736`, :math:`\Omega_m=0.3153`,
       :math:`n_s=0.9649`, :math:`\ln 10^{10}A_s=3.044`)
   * - HMF
     - Tinker+2008 (analytic, JAX-differentiable)
   * - :math:`c`–:math:`M` relation
     - Diemer & Joyce 2019 (via colossus)
   * - Halo profile
     - NFW

:math:`P_{\rm nl}(k)` Backends
--------------------------------

Both backends replace the linear power spectrum in the **2-halo term only**.
The 1-halo term is identical across all runs.

.. list-table::
   :header-rows: 1
   :widths: 15 35 30 20

   * - Backend key
     - Implementation
     - Autodiff
     - Notes
   * - ``hmcode``
     - CAMB + HMcode-2020 (`arXiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_)
       via :class:`~hod_mod.cosmology.nonlinear.HALOFITSpectrum`
     - No — CAMB boundary
     - Cached in :class:`~hod_mod.cosmology.nonlinear.CachedPkNonlinear`
   * - ``aletheia``
     - Aletheia neural-network emulator
       via :class:`~hod_mod.cosmology.nonlinear.NonLinearPowerSpectrum`
     - Yes — full JAX trace via :meth:`pk_nonlinear_jax`
     - Valid :math:`k \in [0.006,\,2.0]\,h\,{\rm Mpc}^{-1}`; extrapolated outside

Model Registry
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 22 8 44

   * - Model class
     - Reference
     - Free / |br| fixed
     - Free parameters
   * - ``HODModel``
     - Zheng+2007
       `arXiv:astro-ph/0703457 <https://arxiv.org/abs/astro-ph/0703457>`_
     - 5 / 0
     - :math:`\log M_{\rm min}`,  :math:`\sigma_{\log m}`,
       :math:`\log M_0`,  :math:`\log M_1`,  :math:`\alpha`
   * - ``MoreHODModel``
     - More+2015
       `arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_
     - 5 / 2
     - :math:`\log M_{\rm min}`,  :math:`\sigma_{\log m}`,
       :math:`\log M_1`,  :math:`\alpha`,  :math:`\kappa`; |br|
       fixed: :math:`\alpha_{\rm inc}=0`, :math:`\log M_{\rm inc}=12`
   * - ``Leauthaud12HODModel``
     - Leauthaud+2012
       `arXiv:1104.0928 <https://arxiv.org/abs/1104.0928>`_
     - 5 / 5
     - :math:`\log M_1`,  :math:`\sigma_{\log m}`,
       :math:`\log M_{\rm sat}`,  :math:`\log M_{\rm cut}`,  :math:`\alpha_{\rm sat}`; |br|
       fixed SHMR shape: :math:`\log M_{*,0}=10.916`, :math:`\beta=0.457`,
       :math:`\delta=0.566`, :math:`\gamma=1.53`,
       :math:`\log M_{*,\rm thresh}=10`
   * - ``Guo18ICSMFModel``
     - Guo+2018
       `arXiv:1804.01993 <https://arxiv.org/abs/1804.01993>`_
     - 6 / 7
     - :math:`\log M_{*,0}^{\rm shmr}`,  :math:`\log M_1^{\rm shmr}`,
       :math:`\alpha_{\rm shmr}`,  :math:`\beta_{\rm shmr}`,
       :math:`\log M_1^{\rm sat}`,  :math:`\alpha_{\rm sat}`; |br|
       fixed: :math:`\sigma_{\log M_*}=0.15`, completeness limits,
       :math:`f_{\rm cen}=f_{\rm sat}=1`
   * - ``Guo19ICSMFModel``
     - Guo+2019
       `arXiv:1810.05318 <https://arxiv.org/abs/1810.05318>`_
     - 7 / 7
     - same as Guo18 + :math:`\log M_q` (quenching mass)
   * - ``Zacharegkas25HODModel``
     - Zacharegkas+2025
       `arXiv:2506.22367 <https://arxiv.org/abs/2506.22367>`_
     - 6 / 10
     - :math:`\log M_1^{\rm shmr}`,  :math:`\log\varepsilon`,
       :math:`\alpha_{\rm shmr}`,  :math:`\gamma_{\rm shmr}`,
       :math:`\delta_{\rm shmr}`,  :math:`B_{\rm sat}`; |br|
       fixed: :math:`\log M_{*,\rm lo}=10`, :math:`\log M_{*,\rm hi}=12`,
       :math:`\sigma_{\log M_*}=0.3`, :math:`f_{\rm cen}=1`, etc.
   * - ``VanUitert16CSMFModel``
     - van Uitert+2016
       `arXiv:1601.06791 <https://arxiv.org/abs/1601.06791>`_
     - 6 / 4
     - :math:`\log M_{h,1}`,  :math:`\log M_{*,0}`,
       :math:`\beta_1`,  :math:`\log\beta_2`,
       :math:`\sigma_c`,  :math:`\alpha_s`; |br|
       fixed: :math:`\log M_{*,\rm lo}=10`, :math:`\log M_{*,\rm hi}=12`,
       :math:`b_0=0`, :math:`b_1=1.5`
   * - ``ZuMandelbaum15HODModel``
     - Zu & Mandelbaum 2015
       `arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_
     - 9 / 5
     - :math:`\log M_{1h}`,  :math:`\log M_{0*}`,
       :math:`\beta`,  :math:`\delta`,  :math:`\gamma`,
       :math:`\sigma_{\ln M_*}`,  :math:`f_c`,
       :math:`b_{\rm sat}`,  :math:`\alpha_{\rm sat}`; |br|
       fixed: :math:`\log M_{*,\rm thresh}=10`, :math:`\eta=-0.04`,
       :math:`\beta_{\rm sat}=0.90`, :math:`b_{\rm cut}=0.86`,
       :math:`\beta_{\rm cut}=0.41`
   * - ``CLFModel``
     - Cacciato+2009
       `arXiv:0807.4932 <https://arxiv.org/abs/0807.4932>`_
     - 7 / 1
     - :math:`\log M_1`,  :math:`\log L_0`,
       :math:`\alpha_{\rm cen}`,  :math:`\beta_{\rm cen}`,
       :math:`\sigma_c`,  :math:`b_{\rm sat}`,  :math:`\alpha_{\rm sat}`; |br|
       fixed: :math:`\log L_{\rm lim}=9.5`

:math:`\chi^2/N_{\rm dof}` Summary
------------------------------------

:math:`N_{\rm dof} = 16 - N_{\rm free}`.  Symbols: ‡ = one or more free
parameters hit a bound; ✗ = minimiser failed to converge on physical solution.

.. list-table::
   :header-rows: 1
   :widths: 28 8 8 18 18

   * - Model
     - :math:`N_{\rm free}`
     - :math:`N_{\rm dof}`
     - :math:`\chi^2/N_{\rm dof}` |br| HMcode
     - :math:`\chi^2/N_{\rm dof}` |br| Aletheia
   * - HODModel
     - 5
     - 11
     - 0.248 ‡
     - 5.12 ✗
   * - MoreHODModel
     - 5
     - 11
     - 0.104 ‡
     - 5.28 ✗
   * - Leauthaud12HODModel
     - 5
     - 11
     - 0.205 ‡
     - 5.93 ✗
   * - Guo18ICSMFModel
     - 6
     - 10
     - **0.083** ‡
     - 5.89 ✗
   * - Guo19ICSMFModel
     - 7
     - 9
     - 0.424 ‡
     - 6.51 ✗
   * - Zacharegkas25HODModel
     - 6
     - 10
     - 0.258
     - 13.78 ✗
   * - VanUitert16CSMFModel
     - 6
     - 10
     - 3.721 ‡
     - 26.44 ✗
   * - ZuMandelbaum15HODModel
     - 9
     - 7
     - 9.129 ✗
     - 10.33 ✗
   * - CLFModel
     - 7
     - 9
     - >1000 ✗
     - >1000 ✗

Comparison Figure
-----------------

.. note::
   Figure (BGS M10 non-linear 2-halo, 3×3 HOD model grid) — generated by the BGS fitting campaign.
   Run ``hod_mod/scripts/fitting/bgs_ls10/`` to populate this figure.

Interpretation
--------------

Aletheia backend failure
~~~~~~~~~~~~~~~~~~~~~~~~

Every model returns :math:`\chi^2/N_{\rm dof} \approx 5`–27 with Aletheia,
regardless of the HOD parametrisation.  The uniform failure across all 9 models
rules out a parameter-space issue; the Aletheia emulator is systematically
mis-predicting :math:`P_{\rm nl}(k)` at :math:`z=0.115` for Planck 2018
cosmology.  Two likely causes:

1. **k-range extrapolation**: The emulator is valid only for
   :math:`k \in [0.006,\,2.0]\,h\,{\rm Mpc}^{-1}`.
   The 2-halo integral runs to :math:`k \sim 200\,h\,{\rm Mpc}^{-1}`, so the
   boost-ratio extrapolation adopted in :meth:`pk_nonlinear_jax` may
   amplify small emulator errors in the transition region.
2. **Redshift coverage**: Aletheia's training set should be checked for
   coverage at :math:`z \approx 0.12`; the emulator may be interpolating
   far from training nodes at this low redshift.

The Aletheia path remains valid for **autodiff through cosmological parameters**
(its primary use case, see :doc:`autodiff_sensitivity`) but should not be used
as a drop-in replacement for HMcode in :math:`w_p` fitting until the
extrapolation behaviour is resolved.

HMcode results — model ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All fits use :math:`r_p \in [0.3,\,30]\,h^{-1}\,\text{Mpc}` (16 bins).
The HMcode non-linear 2-halo substantially improves the fit compared to
the linear 2-halo for all models (compare with :doc:`bgs_ls10_wp_survey`).

**Well-fitting models** (:math:`\chi^2/N_{\rm dof} < 0.5`):

- **Guo18ICSMFModel** is the best-fitting model overall (0.083), though
  :math:`\log M_1^{\rm sat} = 14.5` is at the upper bound.
- **MoreHODModel** (0.104) and **Leauthaud12HODModel** (0.205) match the
  earlier three-model comparison (:doc:`bgs_m10_nonlinear_models`).
- **HODModel** (0.248) and **Zacharegkas25HODModel** (0.258) fit comparably.
- **Guo19ICSMFModel** (0.424) is marginally acceptable.

**Failing models**:

- **VanUitert16CSMFModel** (3.72): the satellite faint-end slope
  :math:`\alpha_s` hits its upper bound at :math:`-0.10` (bound is
  :math:`(-2, -0.1)`), suggesting the model wants a much shallower
  satellite CLF than physically plausible; the parameterisation needs
  review for a threshold-selected sample.
- **ZuMandelbaum15HODModel** (9.13): despite 9 free parameters the model
  cannot reproduce the data.  The fixed parameters
  (:math:`\beta_{\rm sat}`, :math:`b_{\rm cut}`, :math:`\beta_{\rm cut}`)
  likely need to be freed for a :math:`w_p`-only fit.
- **CLFModel** (:math:`\chi^2 \sim 30000`): the luminosity–mass mapping
  diverges.  The fixed luminosity threshold :math:`\log L_{\rm lim}=9.5`
  is likely inconsistent with the BGS stellar-mass threshold
  :math:`\log M_*>10`; the conversion between luminosity and stellar-mass
  thresholds needs calibration before the CLF can be applied to this sample.

Boundary hits (HMcode)
~~~~~~~~~~~~~~~~~~~~~~~

The satellite mass scale parameter is at or near its upper bound
(:math:`\log M_1 = 14.5\,M_\odot/h`) in HODModel, MoreHODModel,
Leauthaud12HODModel, Guo18ICSMFModel, and Guo19ICSMFModel.  This
is the same degeneracy identified in :doc:`bgs_m10_nonlinear_models`:
with :math:`P_{\rm nl}` already boosting the 2-halo signal, the
optimizer drives the satellite normalization to a very high mass scale
to avoid double-counting large-scale power.  A joint SMF + :math:`w_p`
fit is needed to break this degeneracy and constrain :math:`\log M_1`
independently.

Best-fit Parameters (HMcode)
-----------------------------

Boundary hits are marked with ‡.

HODModel
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_{\rm min}`
     - 12.281
     - 11.970
   * - :math:`\sigma_{\log m}`
     - 0.213
     - 0.282
   * - :math:`\log_{10} M_0`
     - 11.257
     - 11.949
   * - :math:`\log_{10} M_1` ‡
     - 14.500
     - 14.036
   * - :math:`\alpha`
     - 1.036
     - 0.500

MoreHODModel
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_{\rm min}`
     - 12.340
     - 12.099
   * - :math:`\sigma_{\log m}`
     - 0.253
     - 0.478
   * - :math:`\log_{10} M_1` ‡
     - 14.500
     - 14.142
   * - :math:`\alpha`
     - 1.183
     - 0.500
   * - :math:`\kappa`
     - 1.124
     - 0.979

Leauthaud+2012
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_1`
     - 13.170
     - 13.120
   * - :math:`\sigma_{\log m}`
     - 0.195
     - 0.727
   * - :math:`\log_{10} M_{\rm sat}` ‡
     - 14.500
     - 14.500 ‡
   * - :math:`\log_{10} M_{\rm cut}`
     - 12.571
     - 11.644
   * - :math:`\alpha_{\rm sat}`
     - 0.965
     - 0.546

Guo+2018 ICSMF
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_{*,0}`
     - 11.036
     - 10.000
   * - :math:`\log_{10} M_1^{\rm shmr}`
     - 12.975
     - 11.354
   * - :math:`\alpha_{\rm shmr}`
     - 0.258
     - 0.363
   * - :math:`\beta_{\rm shmr}`
     - 1.056
     - 1.979
   * - :math:`\log_{10} M_1^{\rm sat}` ‡
     - 14.500
     - 14.309
   * - :math:`\alpha_{\rm sat}`
     - 1.234
     - 0.500

Guo+2019 ICSMF
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_{*,0}` ‡
     - 9.500
     - 9.659
   * - :math:`\log_{10} M_1^{\rm shmr}`
     - 10.671
     - 10.757
   * - :math:`\alpha_{\rm shmr}`
     - 0.352
     - 0.347
   * - :math:`\beta_{\rm shmr}`
     - 1.091
     - 1.847
   * - :math:`\log_{10} M_1^{\rm sat}` ‡
     - 14.500
     - 14.345
   * - :math:`\alpha_{\rm sat}`
     - 1.006
     - 0.500
   * - :math:`\log_{10} M_q`
     - 11.098
     - 11.000

Zacharegkas+2025
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_1^{\rm shmr}`
     - 10.691
     - 10.948
   * - :math:`\log_{10} \varepsilon`
     - −1.824
     - −1.625
   * - :math:`\alpha_{\rm shmr}`
     - −2.755
     - −2.997 ‡
   * - :math:`\gamma_{\rm shmr}`
     - 0.654
     - 0.748
   * - :math:`\delta_{\rm shmr}`
     - 1.939
     - 1.431
   * - :math:`B_{\rm sat}`
     - 13.319
     - 14.601

van Uitert+2016
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_{h,1}`
     - 11.732
     - 10.500 ‡
   * - :math:`\log_{10} M_{*,0}`
     - 11.448
     - 11.236
   * - :math:`\beta_1`
     - 6.839
     - 5.545
   * - :math:`\log_{10}\beta_2`
     - −0.203
     - −0.510
   * - :math:`\sigma_c`
     - 0.131
     - 0.154
   * - :math:`\alpha_s` ‡
     - −0.100
     - −0.830

Zu & Mandelbaum 2015
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - HMcode
     - Aletheia
   * - :math:`\log_{10} M_{1h}`
     - 11.015
     - 11.077
   * - :math:`\log_{10} M_{0*}`
     - 9.787
     - 9.501
   * - :math:`\beta`
     - 0.301
     - 0.440
   * - :math:`\delta`
     - 0.851
     - 0.386
   * - :math:`\gamma`
     - 0.745
     - 1.399
   * - :math:`\sigma_{\ln M_*}`
     - 0.180
     - 0.397
   * - :math:`f_c`
     - 0.724
     - 0.685
   * - :math:`b_{\rm sat}`
     - 14.290
     - 12.681
   * - :math:`\alpha_{\rm sat}`
     - 0.819
     - 0.500

CLF (Cacciato+2009)
~~~~~~~~~~~~~~~~~~~

Both backends return :math:`\chi^2 > 10^4`; the best-fit parameters are
not physically meaningful.  The luminosity threshold :math:`\log L_{\rm lim}=9.5`
(fixed) is inconsistent with the BGS stellar-mass threshold
:math:`\log M_* > 10.0` at :math:`z=0.115`.
The :math:`M_*`–:math:`L` conversion needs to be calibrated before the
CLF model can be applied to this sample.

Fit runtimes
------------

.. list-table::
   :header-rows: 1
   :widths: 35 18 18

   * - Model
     - HMcode (s)
     - Aletheia (s)
   * - HODModel
     - 21
     - 16
   * - MoreHODModel
     - 19
     - 15
   * - Leauthaud12HODModel
     - 78
     - 86
   * - Guo18ICSMFModel
     - 18
     - 21
   * - Guo19ICSMFModel
     - 23
     - 21
   * - Zacharegkas25HODModel
     - 195
     - 298
   * - VanUitert16CSMFModel
     - 24
     - 18
   * - ZuMandelbaum15HODModel
     - 208
     - 162
   * - CLFModel
     - 41
     - 22

Reproducing These Results
--------------------------

.. code-block:: bash

   python scripts/fitting/bgs_ls10/fit_bgs_m10_nl_allmodels.py \
       --backend all \
       --output-dir results/bgs_m10_nl_allmodels \
       --save results/bgs_m10_nl_allmodels/fig_wp_survey.png \
       2>&1 | tee results/bgs_m10_nl_allmodels/run.log

JSON output per backend (separate from the earlier three-model comparison in
``results/bgs_m10_nonlinear_models.json``):

.. code-block:: none

   results/bgs_m10_nl_allmodels/
   ├── hmcode_results.json
   ├── aletheia_results.json
   ├── fig_wp_survey.png
   └── run.log

Path Forward
------------

1. **Diagnose Aletheia extrapolation** — compare
   :meth:`~hod_mod.cosmology.nonlinear.NonLinearPowerSpectrum.pk_nonlinear_jax`
   output against HMcode at :math:`z=0.115` to identify whether the failure
   is in the emulator body or in the boost-ratio extrapolation beyond
   :math:`k=2\,h\,{\rm Mpc}^{-1}`.

2. **Free the satellite boundary** — relax the :math:`\log M_1` upper bound
   from 14.5 to 15.5 and add an SMF constraint to regularise it.  Six of the
   nine models currently hit this boundary.

3. **VanUitert16 :math:`\alpha_s` fix** — the fit wants a shallower satellite
   CLF than the bound allows; investigate whether the double power-law CSMF
   parametrisation is appropriate for a stellar-mass threshold sample, or
   whether the :math:`b_1` parameter should be freed.

4. **ZuMandelbaum15 — free more params** — unfix :math:`\beta_{\rm sat}`,
   :math:`b_{\rm cut}`, :math:`\beta_{\rm cut}` and re-run.

5. **CLF luminosity threshold** — derive :math:`\log L_{\rm lim}` consistent
   with :math:`\log M_* > 10` at :math:`z=0.115` using the BGS mass-to-light
   ratio before re-fitting.

References
----------

* Zheng+2007 — `arXiv:astro-ph/0703457 <https://arxiv.org/abs/astro-ph/0703457>`_
* More+2015 — `arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_
* Leauthaud+2012 — `arXiv:1104.0928 <https://arxiv.org/abs/1104.0928>`_
* Guo+2018 — `arXiv:1804.01993 <https://arxiv.org/abs/1804.01993>`_
* Guo+2019 — `arXiv:1810.05318 <https://arxiv.org/abs/1810.05318>`_
* Zacharegkas+2025 — `arXiv:2506.22367 <https://arxiv.org/abs/2506.22367>`_
* van Uitert+2016 — `arXiv:1601.06791 <https://arxiv.org/abs/1601.06791>`_
* Zu & Mandelbaum 2015 — `arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_
* Cacciato+2009 — `arXiv:0807.4932 <https://arxiv.org/abs/0807.4932>`_
* Mead+2020 (HMcode) — `arXiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_
* Tinker+2008 — `arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_
* Diemer & Joyce 2019 — `arXiv:1809.07326 <https://arxiv.org/abs/1809.07326>`_
