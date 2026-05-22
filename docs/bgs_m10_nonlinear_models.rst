BGS M10 MAP Fits — Linear vs Non-Linear 2-Halo Models
======================================================

This page documents MAP (maximum a posteriori) fits of three halo model configurations
to the BGS LS10 DR10 projected correlation function :math:`w_p(r_p)` for the stellar
mass threshold :math:`\log_{10}(M_*/M_\odot) > 10.0`, :math:`z = 0.05\text{--}0.18`.

The key question is whether replacing the linear matter power spectrum in the 2-halo
term with a non-linear :math:`P_{\rm nl}(k)` (HMcode-2020 via CAMB) improves the fit
and whether this changes the inferred HOD parameters.

.. contents::
   :local:
   :depth: 2

Data and Sample
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
   * - Data vector range
     - :math:`w_p \in [3.5,\,214.0]\,h^{-1}\,\text{Mpc}`
   * - Covariance
     - Jackknife; 1% diagonal regularisation applied

Cosmology is fixed at Planck 2018 TT,TE,EE+lowE best-fit values
(:math:`h=0.6736`, :math:`\Omega_m=0.3153`, :math:`n_s=0.9649`,
:math:`\ln(10^{10}A_s)=3.044`).

Model Configurations
--------------------

Three models are compared, all sharing the same HMF (Tinker+2008), halo profile
(NFW + Diemer & Joyce 2019 :math:`c`–:math:`M` relation), and fixed cosmology.

.. list-table::
   :header-rows: 1
   :widths: 8 22 20 50

   * - Label
     - HOD model
     - 2-halo :math:`P(k)`
     - Free parameters (5 each)
   * - **M1**
     - More+2015 (MoreHODModel)
     - Linear  (CAMB)
     - :math:`\log M_{\rm min}`,  :math:`\sigma_{\log m}`,  :math:`\log M_1`,  :math:`\alpha`,  :math:`\kappa`
   * - **M2**
     - More+2015 (MoreHODModel)
     - Non-linear  (HMcode-2020)
     - same
   * - **M3**
     - Leauthaud+2012 (SHMR-based)
     - Non-linear  (HMcode-2020)
     - :math:`\log M_1`,  :math:`\sigma_{\log m}`,  :math:`\log M_{\rm sat}`,  :math:`\log M_{\rm cut}`,  :math:`\alpha_{\rm sat}`

For M3 the SHMR shape parameters are fixed to the Leauthaud+2012 Table 3 best-fit
values (:math:`\log M_{*,0}=10.916`, :math:`\beta=0.457`, :math:`\delta=0.566`,
:math:`\gamma=1.53`).

All fits use the Nelder-Mead simplex algorithm (scipy ``minimize``, 3000 iterations,
tolerances :math:`10^{-4}` in both parameter and function space).

MAP Results
-----------

Summary
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 40 15 10 12

   * - Label
     - Configuration
     - :math:`\chi^2`
     - :math:`N_{\rm dof}`
     - :math:`\chi^2/N_{\rm dof}`
   * - **M1**
     - More+2015 HOD,  linear 2-halo
     - 6.37
     - 11
     - **0.579**
   * - **M2**
     - More+2015 HOD,  HMcode-2020 nl 2-halo
     - 1.14
     - 11
     - **0.104**
   * - **M3**
     - Leauthaud+2012 HOD,  HMcode-2020 nl 2-halo
     - 2.26
     - 11
     - **0.205**

The non-linear 2-halo term provides a substantially better fit on scales
:math:`r_p \lesssim 5\,h^{-1}\,\text{Mpc}` where the transition between the
1-halo and 2-halo regimes is sensitive to the shape of :math:`P_{\rm nl}(k)`.

Best-fit Parameters — Model 1 (More+2015, linear 2-halo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Best-fit
     - Description
   * - :math:`\log_{10} M_{\rm min}`
     - 11.858
     - Halo mass at 50% central occupation
   * - :math:`\sigma_{\log m}`
     - 0.371
     - Width of central log-normal step
   * - :math:`\log_{10} M_1`
     - 13.128
     - Satellite characteristic mass
   * - :math:`\alpha`
     - 1.169
     - Satellite power-law slope
   * - :math:`\kappa`
     - 0.100 ‡
     - Satellite cutoff factor

‡ :math:`\kappa` hit the lower bound (0.10); this signals that the linear 2-halo
term requires a very sharp satellite cutoff to avoid over-predicting large-scale power.

Best-fit Parameters — Model 2 (More+2015, HMcode-2020 nl 2-halo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Best-fit
     - Description
   * - :math:`\log_{10} M_{\rm min}`
     - 12.340
     - Halo mass at 50% central occupation
   * - :math:`\sigma_{\log m}`
     - 0.253
     - Width of central log-normal step
   * - :math:`\log_{10} M_1`
     - 14.500 ‡
     - Satellite characteristic mass
   * - :math:`\alpha`
     - 1.183
     - Satellite power-law slope
   * - :math:`\kappa`
     - 1.124
     - Satellite cutoff factor

‡ :math:`\log M_1` hit the upper bound (14.50), indicating the non-linear 2-halo
term already accounts for some of the large-scale signal so the satellite contribution
is pushed to very high halo masses.  A wider prior or joint SMF+:math:`w_p` fit
is needed to break this degeneracy.

Best-fit Parameters — Model 3 (Leauthaud+2012, HMcode-2020 nl 2-halo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed SHMR shape: :math:`\log M_{*,0}=10.916`, :math:`\beta=0.457`,
:math:`\delta=0.566`, :math:`\gamma=1.53`, :math:`\log M_{*,\rm thresh}=10.0`.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Best-fit
     - Description
   * - :math:`\log_{10} M_1`
     - 13.170
     - SHMR pivot halo mass
   * - :math:`\sigma_{\log m}`
     - 0.195
     - SHMR scatter in :math:`\log M_*` at fixed :math:`M_h`
   * - :math:`\log_{10} M_{\rm sat}`
     - 14.500 ‡
     - Satellite amplitude mass scale
   * - :math:`\log_{10} M_{\rm cut}`
     - 12.571
     - Satellite exponential cutoff mass
   * - :math:`\alpha_{\rm sat}`
     - 0.965
     - Satellite power-law slope

‡ :math:`\log M_{\rm sat}` hit the upper bound.  The same degeneracy as M2 applies;
this will be resolved in a joint SMF+:math:`w_p` fit.

Comparison Figure
-----------------

.. note::
   Figure (BGS M10 MAP fits, linear vs non-linear 2-halo) — generated by the BGS fitting campaign.
   Run ``hod_mod/scripts/fitting/bgs_ls10/`` to populate this figure.

Interpretation
--------------

**Non-linear 2-halo matters at** :math:`r_p \lesssim 5\,h^{-1}\,\text{Mpc}`.
In this range the standard More+2015 prescription (linear :math:`P(k)`) under-predicts
:math:`w_p` and the Nelder-Mead minimiser compensates by driving :math:`\kappa` to its
lower bound.  Switching to :math:`P_{\rm nl}(k)` (HMcode-2020) reduces :math:`\chi^2`
by a factor of ~6 with identical HOD freedom.

**Parameter shifts between M1 and M2**:

- :math:`\log M_{\rm min}` shifts by +0.48 dex: with the non-linear power already
  providing more clustering at intermediate scales, the minimum halo mass is pulled
  toward more massive (rarer) halos.
- :math:`\sigma_{\log m}` decreases from 0.37 to 0.25: the broader step was
  absorbing scale-dependent corrections now explained by :math:`P_{\rm nl}`.
- :math:`\kappa` rises from 0.10 to 1.12 (no longer at the boundary).

**Leauthaud+2012 (M3) vs More+2015 (M2)**:  both use :math:`P_{\rm nl}` and achieve
similar fit quality (:math:`\chi^2/N_{\rm dof} = 0.205` vs 0.104).  The Leauthaud
SHMR-based central occupation yields a tighter scatter (:math:`\sigma_{\log m}=0.195`)
because the SHMR functional form partially fixes the shape of :math:`N_{\rm cen}(M)`.

Reproducing These Results
--------------------------

.. code-block:: bash

   python scripts/fitting/bgs_ls10/fit_bgs_m10_nonlinear_models.py \\
       --save results/figures/fig_bgs_m10_nonlinear_models.png \\
       --output-json results/bgs_m10_nonlinear_models.json

The script requires:

- ``hod_mod`` installed (``pip install -e .``)
- ``camb`` for the HMcode-2020 non-linear power spectrum
- ``colossus`` for the Diemer & Joyce 2019 :math:`c`–:math:`M` relation
- BGS LS10 data file at ``../sum_stat/data/BGS_Mstar10.0/`` (relative to repo root)

JSON output is saved to ``results/bgs_m10_nonlinear_models.json`` and contains
``chi2``, ``ndof``, and ``params`` for all three models.

Implementation Notes
--------------------

The non-linear 2-halo term is enabled via the ``nl_2halo`` flag in
:class:`~hod_mod.galaxies.clustering.FullHaloModelPrediction`:

.. code-block:: python

   from hod_mod.cosmology.nonlinear import HALOFITSpectrum, CachedPkNonlinear
   from hod_mod.galaxies.clustering import FullHaloModelPrediction

   pk_nl = CachedPkNonlinear(HALOFITSpectrum("mead2020"))
   pred  = FullHaloModelPrediction(pk_lin, hod, hp, pk_nl=pk_nl, nl_2halo=True)

:class:`~hod_mod.cosmology.nonlinear.CachedPkNonlinear` caches CAMB evaluations
keyed on :math:`(z, \Omega_m, \ln 10^{10}A_s, h)` so repeated calls during
optimisation do not re-run CAMB.

References
----------

- More+2015: :math:`w_p` HOD parametrisation — `arXiv:1407.1011 <https://arxiv.org/abs/1407.1011>`_
- Leauthaud+2012: SHMR-based HOD — `arXiv:1104.0928 <https://arxiv.org/abs/1104.0928>`_
- Mead+2020: HMcode non-linear power spectrum — `arXiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_
- Tinker+2008: Halo mass function — `arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_
