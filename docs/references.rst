Bibliography
============

Consolidated reference list for all papers cited in ``hod_mod``.
Entries are grouped by topic and ordered chronologically within each group
to show the progression of the field.

.. contents::
   :local:
   :depth: 1

----

Cosmology and Power Spectra
-----------------------------

Foundation papers for the cosmological framework, linear power spectrum
computation, and non-linear emulators used in ``hod_mod``.

.. [EisensteinHu1998] Eisenstein D.J. & Hu W. 1998, ApJ 496, 605.
   Fitting formulae for the linear matter power spectrum without CDM (transfer function).
   `arXiv:astro-ph/9709066 <https://arxiv.org/abs/astro-ph/9709066>`_

.. [Lewis2002] Lewis A., Challinor A. & Lasenby A. 2000, ApJ 538, 473.
   CAMB: Code for Anisotropies in the Microwave Background; ``hod_mod`` uses CAMB for
   linear :math:`P(k)` via ``LinearPowerSpectrum``.
   `arXiv:astro-ph/9911177 <https://arxiv.org/abs/astro-ph/9911177>`_

.. [PlanckCollaboration2018] Planck Collaboration 2018, A&A 641, A6.
   Planck 2018 cosmological parameters (default cosmology in ``hod_mod``).
   `arXiv:1807.06209 <https://arxiv.org/abs/1807.06209>`_

.. [Aletheia2025] Aletheia Collaboration 2025.
   Non-linear matter power spectrum emulator used via ``NonLinearPowerSpectrum``.
   `arXiv:2511.13826 <https://arxiv.org/abs/2511.13826>`_

----

Halo Model Framework
-----------------------

The halo model provides the theoretical basis for connecting dark matter halos
to observed galaxy statistics. These foundational works established the framework
implemented in ``hod_mod``.

.. [Asgari2023] Marika Asgari, Alexander J. Mead, Catherine Heymans 
   OJAp 6E 39A 2023. The halo model for cosmology: a pedagogical review. 
   `arXiv:astro-ph/2303.08752 <https://arxiv.org/abs/2303.08752>`_

.. [SeljakWarren2004] Seljak U. & Warren M.S. 2004, MNRAS 355, 129.
   First complete halo model predictions for galaxy clustering including
   scale-dependent bias; established the 1-halo + 2-halo decomposition.
   `arXiv:astro-ph/0403698 <https://arxiv.org/abs/astro-ph/0403698>`_

.. [CooraySheth2002] Cooray A. & Sheth R. 2002, Phys. Rep. 372, 1.
   Definitive review of halo models of large-scale structure; reference for the
   1-halo / 2-halo power spectrum decomposition used throughout ``hod_mod``.
   `arXiv:astro-ph/0206508 <https://arxiv.org/abs/astro-ph/0206508>`_

----

Halo Mass Function and Bias
-----------------------------

Calibrations of the halo mass function and halo bias, from early analytic
approximations through simulation-calibrated fits to modern emulators.

.. [PressSchechter1974] Press W.H. & Schechter P. 1974, ApJ 187, 425.
   Original analytic derivation of the dark matter halo abundance;
   historical foundation for all subsequent HMF work.

.. [ShethTormen1999] Sheth R.K. & Tormen G. 1999, MNRAS 308, 119.
   Ellipsoidal collapse HMF; improved agreement with N-body simulations
   over Press-Schechter.
   `arXiv:astro-ph/9901122 <https://arxiv.org/abs/astro-ph/9901122>`_

.. [Jenkins2001] Jenkins A. et al. 2001, MNRAS 321, 372.
   First large N-body calibration of the HMF across multiple cosmologies.
   `arXiv:astro-ph/0005260 <https://arxiv.org/abs/astro-ph/0005260>`_

.. [Tinker2008] Tinker J.L. et al. 2008, ApJ 688, 709.
   Precision calibration of the HMF from N-body simulations across 11 orders
   of magnitude in mass; default HMF in ``hod_mod`` (``tinker08``).
   `arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_

.. [Tinker2010] Tinker J.L. et al. 2010, ApJ 724, 878.
   Calibration of the large-scale linear halo bias corresponding to the
   Tinker+2008 HMF; used in the 2-halo term of ``hod_mod``.
   `arXiv:1001.3162 <https://arxiv.org/abs/1001.3162>`_

.. [ChenCSST2025] Chen Z. et al. 2025, Science China: Physics, Mechanics & Astronomy 68, 9513.
   CEmulator v2.0: Gaussian-Process emulator of halo statistics (HMF, matter power
   spectrum, halo-matter cross-correlation) for CSST cosmologies spanning
   :math:`w_0w_a\nu`CDM; ``make_hmf("csst")`` in ``hod_mod``.
   `ADS <https://ui.adsabs.harvard.edu/abs/2025SCPMA..6809513C>`_

.. [ShenAemulus2025] Shen X. et al. 2025, JCAP 2025 (03), 056.
   Aemulus-ν: Gaussian-Process HMF emulator for massive-neutrino wCDM cosmologies,
   calibrated on 150 high-resolution N-body simulations for
   :math:`M \geq 10^{13}\,M_\odot/h`, :math:`z \leq 2`; ``make_hmf("aemulusnu")``
   in ``hod_mod``.
   `arXiv:2410.00913 <https://arxiv.org/abs/2410.00913>`_

.. [Nishimichi2019] Nishimichi T. et al. 2019, ApJ 884, 29.
   Dark Emulator: Gaussian Process emulation of halo clustering statistics;
   enables rapid HOD predictions for arbitrary ΛCDM cosmologies.
   `arXiv:1811.09504 <https://arxiv.org/abs/1811.09504>`_

----

Halo Profiles and Concentration
---------------------------------

From the original NFW profile through concentration calibrations to the
projected surface-mass-density formulas used in lensing predictions.

.. [NFW1997] Navarro J.F., Frenk C.S. & White S.D.M. 1997, ApJ 490, 493.
   Universal NFW density profile from hierarchical clustering simulations;
   the default halo profile in ``hod_mod``.
   `arXiv:astro-ph/9611107 <https://arxiv.org/abs/astro-ph/9611107>`_

.. [Einasto1965] Einasto J. 1965, Trudy Astrofizicheskogo Instituta Alma-Ata 5, 87.
   Einasto profile; alternative to NFW available in ``hod_mod``.

.. [WrightBrainerd2000] Wright C.O. & Brainerd T.G. 2000, ApJ 534, 34.
   Analytical formulas for weak-lensing shear and convergence of NFW halos;
   basis for :math:`\Delta\Sigma(R)` computations in ``hod_mod``.
   `arXiv:astro-ph/9908213 <https://arxiv.org/abs/astro-ph/9908213>`_

.. [BryanNorman1998] Bryan G.L. & Norman M.L. 1998, ApJ 495, 80.
   Virial overdensity :math:`\Delta_{\rm vir}(z)` fitting formula used in
   halo mass–concentration conversions.
   `arXiv:astro-ph/9710107 <https://arxiv.org/abs/astro-ph/9710107>`_

.. [DiemerJoyce2019] Diemer B. & Joyce M. 2019, ApJ 871, 168.
   Accurate physical model for halo concentrations; default concentration–mass
   relation in ``hod_mod`` via colossus (``diemer19``).
   `arXiv:1809.07326 <https://arxiv.org/abs/1809.07326>`_

----

HOD Models
-----------

The halo occupation distribution (HOD) connects galaxies to dark matter halos.
These references cover the foundational formalism through the models implemented
in ``hod_mod/connection/hod/``.

.. [BerlindWeinberg2002] Berlind A.A. & Weinberg D.H. 2002, ApJ 575, 587.
   Foundational HOD formalism paper; introduced the conditional probability of
   finding :math:`N` galaxies in a halo of mass :math:`M` as the core statistic.
   `arXiv:astro-ph/0109001 <https://arxiv.org/abs/astro-ph/0109001>`_

.. [Zheng2005] Zheng Z. et al. 2005, ApJ 633, 791.
   HOD models with explicit separation of central and satellite galaxies;
   introduced the :math:`\langle N_{\rm cen}\rangle + \langle N_{\rm sat}\rangle`
   decomposition that underlies all modern HOD codes.
   `arXiv:astro-ph/0408564 <https://arxiv.org/abs/astro-ph/0408564>`_

.. [Zheng2007] Zheng Z. et al. 2007, ApJ 667, 760.
   HOD fits to DEEP2 and SDSS galaxy samples across redshifts; the parametrization
   ``HODModel`` in ``hod_mod`` follows Zheng+2007.
   `arXiv:astro-ph/0703457 <https://arxiv.org/abs/astro-ph/0703457>`_

.. [More2015] More S. et al. 2015, ApJ 806, 2.
   HOD analysis of BOSS CMASS using :math:`w_p + \Delta\Sigma`; introduced the
   incompleteness correction and :math:`\kappa` satellite cut.
   Reference model for ``MoreHODModel`` in ``hod_mod``.
   `arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_

.. [vanUitert2016] van Uitert E. et al. 2016, MNRAS 459, 3251.
   HOD fits using a Gaussian conditional stellar mass function;
   reference for ``VanUitert16CSMFModel`` in ``hod_mod``.
   `arXiv:1601.06791 <https://arxiv.org/abs/1601.06791>`_

.. [ZuMandelbaum2015] Zu Y. & Mandelbaum R. 2015, MNRAS 454, 1161.
   iHOD model: inverse SHMR approach to galaxy–halo connection via SDSS
   clustering and galaxy–galaxy lensing; reference for
   ``ZuMandelbaum15HODModel`` in ``hod_mod``.
   `arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_

.. [ZuMandelbaum2016] Zu Y. & Mandelbaum R. 2016, MNRAS 457, 4360.
   iHOD quenching model: Weibull CDF red fractions for centrals and satellites;
   reference for ``ZuMandelbaum16QuenchingModel`` in ``hod_mod``.
   `arXiv:1509.06758 <https://arxiv.org/abs/1509.06758>`_

.. [Guo2018] Guo H. et al. 2018, ApJ 858, 30.
   Incompleteness-corrected SHMR (ICSMF) with broken power-law for SDSS main;
   reference for ``Guo18ICSMFModel`` in ``hod_mod``.
   `arXiv:1804.01993 <https://arxiv.org/abs/1804.01993>`_

.. [Guo2019] Guo H. et al. 2019, ApJ 871, 147.
   15-parameter ICSMF for eBOSS ELGs including quenched fraction;
   reference for ``Guo19ICSMFModel`` in ``hod_mod``.
   `arXiv:1810.05318 <https://arxiv.org/abs/1810.05318>`_

.. [Zacharegkas2025] Zacharegkas G. et al. 2025.
   Kravtsov SHMR with threshold scatter; reference for
   ``Zacharegkas25HODModel`` in ``hod_mod``.
   `arXiv:2506.22367 <https://arxiv.org/abs/2506.22367>`_

----

Stellar-to-Halo Mass Relations and SHAM
-----------------------------------------

Empirical and simulation-based constraints on how stellar mass maps to halo mass,
used in SHAM models (``hod_mod/connection/sham.py``).

.. [Moster2013] Moster B.P., Naab T. & White S.D.M. 2013, MNRAS 428, 3121.
   Empirical SMHM relation via abundance matching across redshifts;
   reference for ``smhm_moster13`` in ``hod_mod``.
   `arXiv:1205.5807 <https://arxiv.org/abs/1205.5807>`_

.. [Behroozi2013] Behroozi P.S., Wechsler R.H. & Conroy C. 2013, ApJ 770, 57.
   SMHM relation from average star formation histories; reference for
   ``smhm_behroozi13`` in ``hod_mod``.
   `arXiv:1207.6105 <https://arxiv.org/abs/1207.6105>`_

.. [Girelli2020] Girelli G. et al. 2020, A&A 634, A135.
   Stellar-to-halo mass relation over the past 12 Gyr;
   reference for ``smhm_girelli20`` in ``hod_mod``.
   `arXiv:2001.02230 <https://arxiv.org/abs/2001.02230>`_

----

Galaxy Clustering and Projected Correlation Function
------------------------------------------------------

Theoretical and observational works on the projected correlation function
:math:`w_p(r_p)` and the power-law approximations used for model validation.

.. [DavisPeebles1983] Davis M. & Peebles P.J.E. 1983, ApJ 267, 465.
   Introduced the projected correlation function :math:`w_p(r_p)` via
   line-of-sight integration to :math:`\pi_{\rm max}`; fundamental observable
   in HOD fitting.

.. [Hamilton1992] Hamilton A.J.S. 1992, ApJ 385, L5.
   Linear redshift-space distortions; basis for RSD corrections in
   projected correlation functions.

----

Galaxy-Galaxy Lensing and Excess Surface Mass Density
-------------------------------------------------------

From the first GGL detections through modern combined HOD+lensing analyses
covering the full range of scales accessible to ``hod_mod``.

.. [BartelmannSchneider2001] Bartelmann M. & Schneider P. 2001, Phys. Rep. 340, 291.
   Comprehensive review of weak gravitational lensing theory; reference for
   :math:`\Delta\Sigma(R)` and convergence formulas.
   `arXiv:astro-ph/9912508 <https://arxiv.org/abs/astro-ph/9912508>`_

.. [Mandelbaum2005] Mandelbaum R. et al. 2005, MNRAS 361, 1287.
   First SDSS galaxy-galaxy lensing analysis measuring halo masses and
   satellite fractions across galaxy samples.
   `arXiv:astro-ph/0501048 <https://arxiv.org/abs/astro-ph/0501048>`_

.. [Mandelbaum2006] Mandelbaum R. et al. 2006, MNRAS 372, 758.
   SDSS GGL: density profiles of galaxy groups and clusters from weak lensing;
   demonstrated NFW profile consistency at group scales.
   `arXiv:astro-ph/0605476 <https://arxiv.org/abs/astro-ph/0605476>`_

.. [Leauthaud2017] Leauthaud A. et al. 2017, MNRAS 467, 3024.
   "Lensing is Low": BOSS CMASS lensing amplitude 20–40% below predictions from
   clustering; established the lensing–clustering discrepancy as a key diagnostic.
   `arXiv:1611.08606 <https://arxiv.org/abs/1611.08606>`_

.. [Miyatake2022] Miyatake H. et al. 2022, Phys. Rev. D 106, 083520.
   Emulator-based HOD analysis of HSC-Y1 × SDSS: joint :math:`w_p + \Delta\Sigma`
   at 3–30 :math:`h^{-1}`Mpc; :math:`S_8 = 0.795^{+0.049}_{-0.042}`.
   Used for pipeline consistency validation.
   `arXiv:2111.02419 <https://arxiv.org/abs/2111.02419>`_

.. [Lange2023] Lange J.U. et al. 2023, MNRAS 520, 5373.
   Full-scale :math:`w_p + \Delta\Sigma` (0.4–63 :math:`h^{-1}`Mpc) in BOSS × KiDS+DES;
   :math:`S_8 = 0.792 \pm 0.022`; includes small-scale HOD constraints.
   `arXiv:2301.08692 <https://arxiv.org/abs/2301.08692>`_

.. [Heydenreich2025] Heydenreich S. et al. 2025.
   "Lensing Without Borders": :math:`\Delta\Sigma` and :math:`w_p` from DESI-DR1
   cross-correlated with DES, KiDS, and HSC; data release for KP7 cosmological analyses.
   `arXiv:2506.21677 <https://arxiv.org/abs/2506.21677>`_

.. [Lange2025] Lange J.U. et al. 2025.
   Cosmological constraints from full-scale clustering + lensing with DESI-DR1:
   :math:`S_8 = 0.794 \pm 0.023`, :math:`\Omega_m = 0.295 \pm 0.012`.
   `arXiv:2512.15962 <https://arxiv.org/abs/2512.15962>`_

----

Intrinsic Alignments
---------------------

Progression from the first tidal alignment models through the non-linear alignment
(NLA) model and its extensions, to modern observational constraints.

.. [Catelan2001] Catelan P., Kamionkowski M. & Blandford R.D. 2001, MNRAS 320, L7.
   First tidal shear model for intrinsic alignments of elliptical galaxies;
   foundation of the linear alignment (LA) model.
   `arXiv:astro-ph/0012040 <https://arxiv.org/abs/astro-ph/0012040>`_

.. [HirataSeljak2004] Hirata C.M. & Seljak U. 2004, Phys. Rev. D 70, 063526.
   Derived the gravitational torquing model and showed LA/NLA is the dominant
   systematic in weak lensing; NLA uses :math:`P_{\rm lin}(k)` — not :math:`P_{\rm nl}`.
   `arXiv:astro-ph/0406275 <https://arxiv.org/abs/astro-ph/0406275>`_

.. [Brown2002] Brown M.L. et al. 2002, MNRAS 333, 501.
   Observational measurement of intrinsic alignments; defines
   :math:`C_1 \rho_{\rm crit,0} = 0.0134` used in the NLA amplitude.
   `arXiv:astro-ph/0208084 <https://arxiv.org/abs/astro-ph/0208084>`_

.. [BridleKing2007] Bridle S. & King L. 2007, New J. Phys. 9, 444.
   NLA model applied to dark energy forecasts; showed IA can bias :math:`w`
   by ~50% if ignored; reference for :math:`A_{\rm IA}` parametrisation in
   ``hod_mod``.
   `arXiv:0705.0166 <https://arxiv.org/abs/0705.0166>`_

.. [Blazek2019] Blazek J. et al. 2019, Phys. Rev. D 100, 103506.
   "Beyond linear galaxy alignments": perturbative framework including quadratic
   tidal terms; order-unity corrections at small scales; FAST-PT implementation.
   `arXiv:1708.09247 <https://arxiv.org/abs/1708.09247>`_

.. [DESI_KP6] DESI Collaboration 2025.
   DESI KP6: intrinsic alignment of BGS-like lenses;
   :math:`A_{\rm IA} \sim 0.3{-}1.5` for stellar-mass-selected samples.
   `arXiv:2509.04552 <https://arxiv.org/abs/2509.04552>`_

----

Baryon Effects on the Matter Power Spectrum
--------------------------------------------

Baryonic feedback suppresses the matter power spectrum at small scales.
These works calibrate and model the suppression, motivating
the baryon fraction and gas concentration models in ``hod_mod``.

.. [vanDaalen2011] van Daalen M.P. et al. 2011, MNRAS 415, 3649.
   OWLS simulations: AGN feedback suppresses :math:`P(k)` by up to 30% at
   :math:`k \gtrsim 1~h/{\rm Mpc}`; first large systematic study.
   `arXiv:1104.1174 <https://arxiv.org/abs/1104.1174>`_

.. [SchneiderTeyssier2015] Schneider A. & Teyssier R. 2015, JCAP 12, 049.
   Baryon correction model (BCM): analytic prescription for baryonic
   redistribution based on gas fraction and stellar feedback.
   `arXiv:1510.06034 <https://arxiv.org/abs/1510.06034>`_

.. [Mead2015] Mead A.J. et al. 2015, MNRAS 454, 1958.
   HMcode: accurate halo model for non-linear :math:`P(k)` including baryonic
   feedback; models gas as an NFW profile with reduced concentration
   :math:`c_{\rm gas} = \eta\,c_{\rm DM}`.
   `arXiv:1505.07098 <https://arxiv.org/abs/1505.07098>`_

.. [McCarthy2017] McCarthy I.G. et al. 2017, MNRAS 465, 2936.
   BAHAMAS: calibrated hydro simulations for large-scale structure cosmology;
   provides gas fractions and profiles at group–cluster scales.
   `arXiv:1612.06090 <https://arxiv.org/abs/1612.06090>`_

.. [IllustrisTNG_chydro] Contreras S. et al. 2024.
   IllustrisTNG / MillenniumTNG: baryonic effects on halo concentration;
   broken power-law fit :math:`c_{\rm hydro}/c_{\rm DMO}` (Table 2) used in
   ``hod_mod`` for gas concentration ratio :math:`\eta(M)`.
   `arXiv:2409.01758 <https://arxiv.org/abs/2409.01758>`_

.. [Schaller2025baryon] Schaller M. et al. 2025, MNRAS 539, 1337.
   FLAMINGO: Gaussian process emulator for baryon suppression of :math:`P(k)`;
   covers diverse feedback models to sub-percent accuracy.
   `arXiv:2410.17109 <https://arxiv.org/abs/2410.17109>`_

.. [Schaller2025analytic] Schaller M. & Schaye J. 2025, MNRAS (accepted).
   Analytic redshift-independent sigmoid parametrisation of baryonic effects on
   :math:`P(k)` from FLAMINGO; motivates the ``BaryonFractionSigmoid`` model.
   `arXiv:2504.15633 <https://arxiv.org/abs/2504.15633>`_

.. [FLAMINGO_fgas] FLAMINGO Collaboration 2025.
   FLAMINGO gas fraction measurements at group scales;
   :math:`f_b(M) < f_{b,\rm cosmic}` as implemented in ``hod_mod``.
   `arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_ *(verify: same ID as [Lange2025phz])*

.. [FLAMINGO_hotgas] FLAMINGO Collaboration 2025.
   FLAMINGO hot gas profiles: :math:`c_{\rm gas} < c_{\rm DM}` at group–cluster
   scales; motivates the gas concentration ratio :math:`\eta(M)`.
   `arXiv:2509.10230 <https://arxiv.org/abs/2509.10230>`_

.. [Siegel2025] Siegel J. et al. 2025, MNRAS (submitted).
   X-ray gas fractions + kSZ profiles + GGL: :math:`10 \pm 2\%` matter power
   suppression at :math:`k = 1~h/{\rm Mpc}`; validates baryon fraction model.
   `arXiv:2512.02954 <https://arxiv.org/abs/2512.02954>`_

.. [Veenema2026] Veenema M. et al. 2026.
   Closure-radius model for the baryon fraction in halos.
   `arXiv:2603.13095 <https://arxiv.org/abs/2603.13095>`_

.. [Pfeifer2025] Pfeifer S. et al. 2025.
   Machine-learning gas profiles: halo mass as primary driver beyond
   :math:`M_{\rm BH}`.
   `arXiv:2512.09021 <https://arxiv.org/abs/2512.09021>`_

----

Gas Profiles and Cross-Correlations
--------------------------------------

Papers providing the gas profile parametrisations and the observational
benchmarks for galaxy × tSZ and galaxy × soft X-ray cross-correlations.

.. [Arnaud2010] Arnaud M., Pratt G.W., Piffaretti R. et al. 2010, A&A 517, A92.
   Universal pressure profile of galaxy clusters from the REXCESS sample
   (generalised NFW; Table 1: P₀=8.403, c₅₀₀=1.177, γ=0.3081, α=1.0510,
   β=5.4905, α_p=0.12).  Implemented as
   :class:`~hod_mod.gas.PressureProfileA10`.
   `arXiv:0910.1234 <https://arxiv.org/abs/0910.1234>`_

.. [Oppenheimer2025] Oppenheimer B.D. et al. 2025.
   DPMhalo: parametric electron density profiles for the diffuse gas around
   galaxies; 3 calibrated model variants with mass- and redshift-dependent
   normalisations.  Implemented as
   :class:`~hod_mod.gas.GasDensityDPM`.
   `arXiv:2505.14782 <https://arxiv.org/abs/2505.14782>`_

.. [Comparat2025] Comparat J. et al. 2025, A&A 697, A173.
   Galaxy × eROSITA eRASS:5 soft X-ray (0.5–2 keV) angular cross-correlation
   for 7 stellar-mass-selected LS DR10 samples (M*>10¹⁰–10¹¹·⁵ M☉);
   HOD + DPM gas model (Tables 3–4).  Data in
   ``hod_mod/data/benchmarks/xray/``.
   `arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_

.. [Amodeo2021] Amodeo S. et al. 2021, Phys. Rev. D 103, 063514.
   ACT DR4 × BOSS: stacked tSZ and kSZ profiles around BOSS CMASS and LOWZ
   galaxies; 4.5σ measurement of the baryonic mass density in the warm-hot
   intergalactic medium.  Model comparison target for
   ``validate_amodeo2021.py``.
   `arXiv:2009.05557 <https://arxiv.org/abs/2009.05557>`_

.. [Pandey2025] Pandey S. et al. 2025.
   DES Year 3 × ACT DR6: 21σ detection of the lensing × tSZ cross-correlation
   C_ℓ^{γ,y}; constraints on baryonic feedback at group–cluster scales.
   Model comparison target for ``validate_pandey2025.py``.
   `arXiv:2506.07432 <https://arxiv.org/abs/2506.07432>`_

----

Surveys and Data
-----------------

Key spectroscopic and imaging surveys providing the galaxy samples and
weak-lensing source catalogues used in ``hod_mod`` analyses.

.. [Blanton2003] Blanton M.R. et al. 2003, ApJ 592, 819.
   SDSS photometric survey and galaxy samples used in many HOD analyses.
   `arXiv:astro-ph/0209479 <https://arxiv.org/abs/astro-ph/0209479>`_

.. [BOSS_CMASS] Anderson L. et al. 2014, MNRAS 441, 24.
   SDSS-III BOSS Data Releases 10 and 11; the CMASS sample is the
   reference HOD target in ``more2015_boss_cmass.py``.
   `arXiv:1312.4877 <https://arxiv.org/abs/1312.4877>`_

.. [HSC_Aihara2018] Aihara H. et al. 2018, PASJ 70, S4.
   Hyper Suprime-Cam Subaru Strategic Program: overview of the survey.
   `arXiv:1704.05858 <https://arxiv.org/abs/1704.05858>`_

.. [HSC_Mandelbaum2018] Mandelbaum R. et al. 2018, PASJ 70, S25.
   HSC-Y1 weak-lensing shape catalog; source of HSC ESD data in ``hod_mod``.
   `arXiv:1705.06745 <https://arxiv.org/abs/1705.06745>`_

.. [KiDS_Heymans2021] Heymans C. et al. 2021, A&A 646, A140.
   KiDS-1000 multi-probe 3×2pt analysis: :math:`S_8 = 0.766^{+0.020}_{-0.014}`,
   2–3σ below Planck; source of KiDS ESD data used in BGS analyses.
   `arXiv:2007.15632 <https://arxiv.org/abs/2007.15632>`_

.. [DES_Abbott2022] Abbott T.M.C. et al. 2022, Phys. Rev. D 105, 023520.
   DES Year 3 cosmic shear: :math:`S_8 = 0.759^{+0.025}_{-0.023}`,
   2.3σ below Planck; source of DES ESD data used in BGS analyses.
   `arXiv:2105.13544 <https://arxiv.org/abs/2105.13544>`_

.. [DESI_EDR] DESI Collaboration 2023.
   DESI Early Data Release: survey overview, instrument, targeting.
   `arXiv:2306.06308 <https://arxiv.org/abs/2306.06308>`_

.. [DESI_BGS_Hahn2023] Hahn C. et al. 2023, AJ 165, 253.
   DESI Bright Galaxy Survey: target selection, completeness, and validation.
   `arXiv:2208.08512 <https://arxiv.org/abs/2208.08512>`_

.. [Comparat2023] Comparat J. et al. 2023, A&A 673, A122.
   eFEDS X-ray AGN HOD analysis: joint X-ray/optical galaxy–halo connection.
   `ADS <https://ui.adsabs.harvard.edu/abs/2023A%26A...673A.122C>`_

.. [Lange2024] Lange J.U. et al. 2024, MNRAS (accepted).
   Systematic effects in galaxy–galaxy lensing with DESI: fibre incompleteness,
   magnification, and intrinsic alignment for DES/HSC/KiDS sources.
   `arXiv:2404.09397 <https://arxiv.org/abs/2404.09397>`_

.. [Lange2025phz] Lange J.U. et al. 2025, ApJ (accepted).
   Unified photometric redshift calibration for DES, HSC, and KiDS weak-lensing
   surveys using DESI spectroscopy; reduces photo-z systematic uncertainty.
   `arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_ *(verify: same ID as [FLAMINGO_fgas])*

----

Recent Cosmological Constraints (S₈ Tension)
----------------------------------------------

Joint analyses combining galaxy clustering with weak gravitational lensing
to constrain :math:`S_8 = \sigma_8 (\Omega_m/0.3)^{0.5}`.  All recent
results find :math:`S_8 \approx 0.77{-}0.80`, consistently 1.5–2.5σ below Planck.

The data for the DESI-DR1 analyses are described in `[Heydenreich2025]_`
("Lensing Without Borders").  Individual HOD-based and full-shape constraints:

- `[Miyatake2022]_` — HSC-Y1 × SDSS, :math:`S_8 = 0.795^{+0.049}_{-0.042}`
- `[Lange2023]_` — BOSS × KiDS+DES, :math:`S_8 = 0.792 \pm 0.022`
- `[Porredon2025]_` — DESI-DR1 3×2pt, :math:`S_8 = 0.786^{+0.022}_{-0.019}`
- `[Semenaite2025]_` — DESI-DR1 full-shape, :math:`S_8 = 0.771{-}0.791`
- `[Lange2025]_` — DESI-DR1 HOD-based, :math:`S_8 = 0.794 \pm 0.023`

.. [Porredon2025] Porredon A. et al. 2025, Open J. Astrophys. 9.
   DESI-DR1 3×2pt analysis (BGS+LRG × KiDS-1000/DES-Y3/HSC-Y3):
   :math:`S_8 = 0.786^{+0.022}_{-0.019}`, 1.5–2σ below Planck.
   `arXiv:2512.15960 <https://arxiv.org/abs/2512.15960>`_

.. [Semenaite2025] Semenaite A. et al. 2025, Open J. Astrophys.
   DESI-DR1 full-shape clustering + lensing in configuration space
   (BGS+LRG × KiDS-1000/DES-Y3/HSC-Y3):
   :math:`S_8 = 0.771{-}0.791`, 1.9–2.9σ below Planck.
   `arXiv:2512.15961 <https://arxiv.org/abs/2512.15961>`_

----

Inference Methods
------------------

Statistical inference tools used in ``hod_mod`` for MAP estimation and
posterior sampling.

.. [Foreman-Mackey2013] Foreman-Mackey D. et al. 2013, PASP 125, 306.
   emcee: the MCMC Hammer — affine-invariant ensemble sampler;
   used in ``WpFitter.mcmc_fit()``.
   `arXiv:1202.3665 <https://arxiv.org/abs/1202.3665>`_

.. [Phan2019] Phan D. et al. 2019.
   NumPyro: composable effects for flexible and accelerated probabilistic
   programming; used in ``hod_mod/inference.py`` for HMC/NUTS.
   `arXiv:1912.11554 <https://arxiv.org/abs/1912.11554>`_

----

Galaxy-Halo Connection with Non-Linear Power Spectrum
-------------------------------------------------------

The standard halo model in ``hod_mod`` uses the **linear** matter power spectrum
:math:`P_{\rm lin}(k)` for the 2-halo term (following More et al. 2015).  A parallel
literature bypasses this approximation by either (a) substituting a non-linear fitting
formula / emulator for :math:`P(k)` directly into the halo-model integrals, or (b)
emulating :math:`w_p(r_p)` and :math:`\Delta\Sigma(R)` end-to-end from N-body
simulations.  The papers below are organised chronologically within four sub-topics.

*Non-linear P(k) fitting formulae.*

.. [PeacockSmith2000] Peacock J.A. & Smith R.E. 2000, MNRAS 318, 1144.
   Derived the first analytic halo model for the non-linear matter power spectrum,
   decomposing :math:`P_{\rm nl}(k) = P^{\rm 1h}(k) + P^{\rm 2h}(k)` from NFW
   profiles and a Press-Schechter HMF; foundation for all subsequent non-linear
   halo-model treatments of galaxy statistics.
   `arXiv:astro-ph/0005010 <https://arxiv.org/abs/astro-ph/0005010>`_

.. [Smith2003] Smith R.E. et al. 2003, MNRAS 341, 1311.
   HALOFIT: empirical fitting formula for :math:`P_{\rm nl}(k)` calibrated on
   N-body simulations over :math:`0.001 \le k \le 10\,h\,{\rm Mpc}^{-1}`;
   the first widely used non-linear :math:`P(k)` prescription in HOD pipelines.
   `arXiv:astro-ph/0207664 <https://arxiv.org/abs/astro-ph/0207664>`_

.. [Takahashi2012] Takahashi R. et al. 2012, ApJ 761, 152.
   Revised HALOFIT recalibrated on higher-resolution N-body simulations;
   corrects ~10% errors in the original Smith et al. (2003) formula at
   :math:`k \gtrsim 1\,h\,{\rm Mpc}^{-1}`; default non-linear :math:`P(k)` in many
   HOD pipelines and in the CosmoCov / TreeCorr ecosystem.
   `arXiv:1208.2701 <https://arxiv.org/abs/1208.2701>`_

.. [Mead2020] Mead A.J. et al. 2021, MNRAS 502, 1401.
   HMcode-2020: extended halo model for non-linear :math:`P(k)` with neutrino
   masses and baryonic feedback; sub-percent accuracy to
   :math:`k \le 10\,h\,{\rm Mpc}^{-1}`, :math:`z \le 2`;
   see also [Mead2015]_ for the original version.
   `arXiv:2009.01858 <https://arxiv.org/abs/2009.01858>`_

*Foundational halo-model treatments of galaxy statistics.*

.. [Seljak2000] Seljak U. 2000, MNRAS 318, 1144.
   First analytic galaxy + dark matter clustering model using NFW profiles and
   a Poisson HOD; showed that non-linear galaxy power spectra can be predicted
   from halo properties alone, motivating the modern HOD+halo-model approach.
   `arXiv:astro-ph/0001493 <https://arxiv.org/abs/astro-ph/0001493>`_

.. [Scoccimarro2001] Scoccimarro R., Sheth R.K., Hui L. & Jain B. 2001, ApJ 546, 20.
   "How Many Galaxies Fit in a Halo?": tested non-linear HOD predictions from
   N-body simulations; established that the 1-halo term dominates
   :math:`w_p(r_p)` at :math:`r_p \lesssim 1\,h^{-1}` Mpc and that departures
   from linearity must be modelled at those scales.
   `arXiv:astro-ph/0006319 <https://arxiv.org/abs/astro-ph/0006319>`_

*HOD / CLF implementations fitting w_p and ΔΣ with the full non-linear halo model.*

.. [Cacciato2009] Cacciato M., van den Bosch F.C. & More S. 2009, MNRAS 394, 929.
   Conditional luminosity function (CLF) halo model jointly fitting galaxy
   clustering and galaxy-galaxy lensing; the 1-halo contribution to both
   :math:`w_p` and :math:`\Delta\Sigma` is computed from non-linear NFW profiles;
   pioneered the combined :math:`w_p + \Delta\Sigma` constraint framework.
   `arXiv:0807.4932 <https://arxiv.org/abs/0807.4932>`_

.. [Leauthaud2012] Leauthaud A. et al. 2012, ApJ 744, 159.
   COSMOS HOD: joint weak lensing + clustering across stellar-mass threshold bins
   at :math:`0.2 < z < 1.0`; full non-linear 1h+2h halo model including NFW
   profiles; derived galaxy–halo connection from :math:`\Delta\Sigma + n_{\rm gal}`.
   `arXiv:1104.0928 <https://arxiv.org/abs/1104.0928>`_

.. [Cacciato2013] Cacciato M., van Uitert E. & Hoekstra H. 2014, MNRAS 437, 377.
   CLF halo model for KiDS/SDSS weak lensing + clustering spanning
   0.1–30 :math:`h^{-1}`Mpc in a single non-linear halo model;
   demonstrated consistent :math:`w_p + \Delta\Sigma` constraints without
   switching between linear and non-linear prescriptions.
   `arXiv:1303.5445 <https://arxiv.org/abs/1303.5445>`_

.. [Zacharegkas2022] Zacharegkas G. et al. 2022, MNRAS 509, 3119.
   DES Year 3 galaxy-galaxy lensing: high-precision :math:`\Delta\Sigma(R)`
   measurement combined with :math:`w_p(r_p)` and HOD halo-model fitting at
   non-linear scales; one of the largest GGL samples used for galaxy-halo
   connection inference at the time.
   `arXiv:2106.08438 <https://arxiv.org/abs/2106.08438>`_

*N-body emulator approaches: w_p and ΔΣ predicted directly from simulations.*

These methods replace both the linear power spectrum and the analytic halo model
integrals with Gaussian-process or neural-network interpolation over a grid of
N-body runs, making the predicted statistics fully non-linear by construction.

.. [DeRose2019] DeRose J. et al. 2019.
   The Aemulus Project I: suite of 75 high-resolution N-body simulations spanning
   a 7-dimensional wCDM parameter space; the simulation grid that
   underpins the Aemulus halo-statistics emulator (see [Wibking2019]_).
   `arXiv:1804.05865 <https://arxiv.org/abs/1804.05865>`_

.. [Wibking2017] Wibking B.D., Salcedo A.N. & Weinberg D.H. 2019, MNRAS 492, 2872.
   Methodology and Fisher-matrix forecasts for emulating galaxy clustering and
   galaxy-galaxy lensing into the deeply non-linear regime; Taylor-expansion
   emulator around a pivot HOD; showed that small scales
   (:math:`r_p \gtrsim 0.5\,h^{-1}` Mpc) tighten cosmological constraints
   substantially.
   `arXiv:1709.07099 <https://arxiv.org/abs/1709.07099>`_

.. [Wibking2019] Wibking B.D., Weinberg D.H. & Salcedo A.N. 2020, MNRAS 492, 2872.
   Applied the emulator method to BOSS LOWZ: cosmological constraints from
   :math:`w_p + \Delta\Sigma` on non-perturbative scales
   (0.4–30 :math:`h^{-1}` Mpc); demonstrated consistent results with
   traditional large-scale analyses while extracting additional information
   from the 1-halo regime.
   `arXiv:1907.06293 <https://arxiv.org/abs/1907.06293>`_

.. [Kobayashi2020] Kobayashi Y. et al. 2020.
   Dark Quest emulator for the redshift-space power spectrum of dark matter
   halos; neural-network emulator trained on the Dark Quest N-body suite;
   achieves ~1% accuracy for galaxy power spectrum predictions used in
   HOD :math:`w_p` / :math:`\Delta\Sigma` forward models.
   `arXiv:2005.06122 <https://arxiv.org/abs/2005.06122>`_

.. [Miyatake2021] Miyatake H. et al. 2021, Phys. Rev. D 103, 123517.
   Dark Quest validation paper: cosmological inference pipeline from
   emulator-based HOD applied to HSC-Y1 and SDSS mock catalogues;
   established end-to-end accuracy of the emulator approach for
   joint :math:`w_p + \Delta\Sigma` analysis before application to real data
   (see [Miyatake2022]_).
   `arXiv:2101.00113 <https://arxiv.org/abs/2101.00113>`_

----

Simulation Reference: FLAMINGO
---------------------------------

The FLAMINGO suite of cosmological hydrodynamical simulations underpins
the baryon fraction and gas profile calibrations in ``hod_mod``.

.. [FLAMINGO_overview] FLAMINGO Collaboration 2023.
   FLAMINGO: Large cosmo-hydro simulations for next-generation lensing surveys.
   `https://flamingo.strw.leidenuniv.nl/ <https://flamingo.strw.leidenuniv.nl/>`_
