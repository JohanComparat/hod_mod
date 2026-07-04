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
   `arXiv:2303.08752 <https://arxiv.org/abs/2303.08752>`_

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

----

Beyond-ΛCDM Cosmology and Emulators
-------------------------------------

Dynamical dark energy, massive neutrinos and the emulator landscape — the
upgrade path for the differentiable :doc:`forecast cosmology <missing_physics>`.

.. [EisensteinHu1999] Eisenstein D.J. & Hu W. 1999, ApJ 511, 5.
   Analytic transfer functions for CDM variants including massive neutrinos;
   the fitting-function route to a differentiable ν-suppression.
   `arXiv:astro-ph/9710252 <https://arxiv.org/abs/astro-ph/9710252>`_

.. [Kiakotou2008] Kiakotou A., Elgarøy Ø. & Lahav O. 2008, Phys. Rev. D 77, 063005.
   Scale-dependent growth suppression from massive neutrinos — compact
   coefficients for the f_ν correction to D(z, k).
   `arXiv:0709.0253 <https://arxiv.org/abs/0709.0253>`_

.. [DESI_DR2_BAO] DESI Collaboration 2025, Phys. Rev. D 112, 083515.
   DESI DR2 BAO measurements: ~3σ preference for dynamical dark energy
   (w0waCDM) — the science driver for freeing w0/wa in the forecast.
   `arXiv:2503.14738 <https://arxiv.org/abs/2503.14738>`_

.. [CosmoVerse2025] Di Valentino E. et al. 2025, Phys. Dark Univ. 49, 101965.
   CosmoVerse white paper on observational tensions (H0, S8) — the context in
   which beyond-ΛCDM freedom must be marginalised, not assumed away.
   `arXiv:2504.01669 <https://arxiv.org/abs/2504.01669>`_

.. [EuclidEmulator2] Euclid Collaboration (Knabenhans M.) et al. 2021, MNRAS 505, 2840.
   EuclidEmulator2: non-linear P(k) boost emulation with massive neutrinos and
   dynamical dark energy — a distill-to-table candidate for the JAX pipeline.
   `arXiv:2010.11288 <https://arxiv.org/abs/2010.11288>`_

.. [BACCO2021] Angulo R.E. et al. 2021, MNRAS 507, 5869.
   The BACCO simulation project: cosmology-rescaled non-linear P(k) emulator
   covering σ8, w0/wa and Σm_ν.
   `arXiv:2004.06245 <https://arxiv.org/abs/2004.06245>`_

.. [MiraTitanIV] Moran K.R. et al. 2023, MNRAS 520, 3443.
   Mira-Titan IV: high-precision P(k) emulator over an 8-parameter
   w0waνCDM space.
   `arXiv:2207.12345 <https://arxiv.org/abs/2207.12345>`_

.. [CSSTEmulatorI] Chen Z. et al. 2025, Sci. China Phys. Mech. Astron. 68, 289512.
   CSST emulator I: matter P(k) to k = 10 h/Mpc at one percent in w0waνCDM;
   companion of the HMF emulator [ChenCSST2025]_ already wrapped in ``hod_mod``.
   `arXiv:2502.11160 <https://arxiv.org/abs/2502.11160>`_

.. [Goku2025] Yang Y., Bird S. & Ho M.-F. 2025, Phys. Rev. D 111, 083529.
   Goku: ten-parameter simulation suite for emulation beyond ΛCDM
   (w0, wa, Σm_ν, N_eff, running).
   `arXiv:2501.06296 <https://arxiv.org/abs/2501.06296>`_

.. [eMANTIS2024] Sáez-Casares I., Rasera Y. & Li B. 2024, MNRAS 527, 7242.
   e-MANTIS: non-linear P(k) emulator for f(R) modified gravity — the
   modified-gravity branch of the emulator landscape.
   `arXiv:2303.08899 <https://arxiv.org/abs/2303.08899>`_

.. [DUCA2025] Castro T. et al. 2025, A&A 697, A194.
   DUCA I: halo mass function calibration in dynamical dark energy
   cosmologies — the w0waCDM correction to Tinker-style HMFs.
   `arXiv:2504.07608 <https://arxiv.org/abs/2504.07608>`_

----

Precision Halo-Model Ingredients
----------------------------------

Cosmology-dependent calibrations of the mass function, bias, concentration
and the linear power spectrum (see :doc:`missing_physics`).

.. [CLASS2011] Lesgourgues J. 2011, arXiv:1104.2932.
   The CLASS Boltzmann solver — with CAMB [Lewis2002]_, the accuracy standard
   any differentiable P(k) surrogate must reproduce.
   `arXiv:1104.2932 <https://arxiv.org/abs/1104.2932>`_

.. [CosmoPower2022] Spurio Mancini A. et al. 2022, MNRAS 511, 1771.
   COSMOPOWER: neural-network emulation of CMB and matter power spectra; the
   dense-MLP architecture is ~10 lines of ``jnp`` to evaluate — the template
   for a differentiable CAMB-ratio emulator.
   `arXiv:2106.03846 <https://arxiv.org/abs/2106.03846>`_

.. [EvolutionMapping2022] Sánchez A.G. et al. 2022, MNRAS 514, 5673.
   Evolution mapping: degeneracy structure of matter clustering across
   cosmologies — compresses the emulation parameter space.
   `arXiv:2108.12710 <https://arxiv.org/abs/2108.12710>`_

.. [DiemerKravtsov2015] Diemer B. & Kravtsov A.V. 2015, ApJ 799, 108.
   Universal concentration–mass model in terms of peak height ν and local
   P(k) slope n_eff — a *genuinely cosmology-dependent* c(M) that is
   JAX-portable (both inputs live on the σ(M) grid).
   `arXiv:1407.4730 <https://arxiv.org/abs/1407.4730>`_

.. [Colossus2018] Diemer B. 2018, ApJS 239, 35.
   COLOSSUS toolkit — the numerical reference implementation for validating
   c(M) and HMF ports.
   `arXiv:1712.04512 <https://arxiv.org/abs/1712.04512>`_

.. [Despali2016] Despali G. et al. 2016, MNRAS 456, 2486.
   Universality of the virial-overdensity HMF and models for the
   non-universality of other halo definitions.
   `arXiv:1507.05627 <https://arxiv.org/abs/1507.05627>`_

.. [Comparat2017] Comparat J. et al. 2017, MNRAS 469, 4157.
   Accurate halo mass and velocity functions from the MultiDark simulations;
   one of the ``f(sigma)`` fits shipped in ``hod_mod``.
   `arXiv:1702.01628 <https://arxiv.org/abs/1702.01628>`_

.. [EuclidHMF2023] Euclid Collaboration (Castro T.) et al. 2023, A&A 671, A100.
   Percent-level HMF calibration in Λ(ν)CDM with explicit cosmology
   dependence of the fit parameters — the reference against which the
   tinker08 non-universality budget should be quoted.
   `arXiv:2208.02174 <https://arxiv.org/abs/2208.02174>`_

----

AGN Multi-Wavelength Emission and the Fundamental Plane
---------------------------------------------------------

Radio and infrared AGN emission channels and their halo statistics
(see :doc:`missing_physics`).

.. [MerloniHeinzDiMatteo2003] Merloni A., Heinz S. & di Matteo T. 2003, MNRAS 345, 1057.
   The fundamental plane of black-hole activity:
   log L_R = 0.60 log L_X + 0.78 log M_BH + 7.33 — the relation that attaches
   a radio luminosity to the Powell chain's (M_BH, L_X).
   `arXiv:astro-ph/0305261 <https://arxiv.org/abs/astro-ph/0305261>`_

.. [FalckeKordingMarkoff2004] Falcke H., Körding E. & Markoff S. 2004, A&A 414, 895.
   Jet-dominated accretion unification of low-power black holes — the
   physical basis of the radio/X-ray correlation.
   `arXiv:astro-ph/0305335 <https://arxiv.org/abs/astro-ph/0305335>`_

.. [Gultekin2019] Gültekin K. et al. 2019, ApJ 871, 80.
   Updated fundamental-plane coefficients and scatter with dynamical M_BH —
   the natural external prior on (ξ_RX, ξ_RM, b_R, σ_R).
   `arXiv:1901.02530 <https://arxiv.org/abs/1901.02530>`_

.. [Hopkins2007] Hopkins P.F., Richards G.T. & Hernquist L. 2007, ApJ 654, 731.
   Observational bolometric quasar luminosity function; the L_bol-dependent
   bolometric corrections used to map L_bol to infrared bands.
   `arXiv:astro-ph/0605678 <https://arxiv.org/abs/astro-ph/0605678>`_

.. [BestHeckman2012] Best P.N. & Heckman T.M. 2012, MNRAS 421, 1569.
   The local radio-AGN dichotomy (HERG/LERG) — the jet population that a
   fundamental-plane-only model does not capture.
   `arXiv:1201.2397 <https://arxiv.org/abs/1201.2397>`_

.. [Shen2009] Shen Y. et al. 2009, ApJ 697, 1656.
   Quasar clustering from SDSS DR5 as a function of physical properties
   (including radio loudness).
   `arXiv:0810.4144 <https://arxiv.org/abs/0810.4144>`_

.. [RetanaMontenegro2017] Retana-Montenegro E. & Röttgering H.J.A. 2017, A&A 600, A97.
   Radio-loud vs radio-quiet quasar clustering (SDSS×FIRST): halo-mass
   difference between the two populations.
   `arXiv:1611.08630 <https://arxiv.org/abs/1611.08630>`_

.. [Hale2025] Hale C.L. et al. 2025, MNRAS 544, 1323.
   Clustering of radio AGN and star-forming galaxies in the LoTSS Deep
   Fields — the current benchmark for radio-AGN halo occupation.
   `arXiv:2510.01029 <https://arxiv.org/abs/2510.01029>`_

.. [Donoso2014] Donoso E. et al. 2014, ApJ 789, 44.
   Angular clustering of WISE-selected AGN: different halos for obscured and
   unobscured populations.
   `arXiv:1309.2277 <https://arxiv.org/abs/1309.2277>`_

.. [Petter2023] Petter G.C. et al. 2023, ApJ 946, 27.
   Host halos of 1.4 million WISE obscured/unobscured quasars — the IR-side
   test of the obscuration parameter shared with the X-ray sector.
   `arXiv:2302.00690 <https://arxiv.org/abs/2302.00690>`_

.. [Comparat2019] Comparat J. et al. 2019, MNRAS 487, 2005.
   eROSITA AGN mock catalogue with empirical multi-wavelength SEDs — the
   ``hod_mod`` heritage for AGN band modelling.
   `arXiv:1901.10866 <https://arxiv.org/abs/1901.10866>`_

.. [Aird2015] Aird J. et al. 2015, MNRAS 451, 1892.
   X-ray luminosity functions of unabsorbed and absorbed AGN to z~5; the
   validation target of the Powell XLF in ``hod_mod``.
   `arXiv:1503.01120 <https://arxiv.org/abs/1503.01120>`_

----

Galaxy Morphology, Quenching and Star Formation
-------------------------------------------------

The galaxy-population physics (morphology, sSFR, quenching) missing from the
all-galaxy ZM15 connection (see :doc:`missing_physics`).

.. [WechslerTinker2018] Wechsler R.H. & Tinker J.L. 2018, ARA&A 56, 435.
   Review of the galaxy–halo connection: assembly bias, conditional
   distributions, and where morphology/SFR enter the halo model.
   `arXiv:1804.03097 <https://arxiv.org/abs/1804.03097>`_

.. [Peng2010] Peng Y. et al. 2010, ApJ 721, 193.
   Mass and environment quenching separability — the empirical form behind
   halo-mass quenching parameterisations.
   `arXiv:1003.4747 <https://arxiv.org/abs/1003.4747>`_

.. [Ilbert2013] Ilbert O. et al. 2013, A&A 556, A55.
   Quiescent and star-forming stellar mass functions since z≈4 (UltraVISTA)
   — the f_Q(M*, z) data the split model must reproduce.
   `arXiv:1301.3157 <https://arxiv.org/abs/1301.3157>`_

.. [Muzzin2013] Muzzin A. et al. 2013, ApJ 777, 18.
   SF/quiescent stellar mass functions to z = 4 (COSMOS/UltraVISTA).
   `arXiv:1303.4409 <https://arxiv.org/abs/1303.4409>`_

.. [Speagle2014] Speagle J.S. et al. 2014, ApJS 214, 15.
   Consistent star-forming main sequence over 0 < z < 6 — the μ_MS(M*, z)
   parameterisation for a continuous sSFR model.
   `arXiv:1405.2041 <https://arxiv.org/abs/1405.2041>`_

.. [Kennicutt1998] Kennicutt R.C. 1998, ARA&A 36, 189.
   Star formation in galaxies along the Hubble sequence — the SFR–line
   calibrations behind the ``loii_norm`` parameter.
   `arXiv:astro-ph/9807187 <https://arxiv.org/abs/astro-ph/9807187>`_

.. [MadauDickinson2014] Madau P. & Dickinson M. 2014, ARA&A 52, 415.
   The cosmic star-formation history — the ρ_SFR(z) data the ``sfrd``
   observable is compared against.
   `arXiv:1403.0007 <https://arxiv.org/abs/1403.0007>`_

.. [Murphy2011] Murphy E.J. et al. 2011, ApJ 737, 67.
   Extinction-free SFR diagnostics calibrated with 33 GHz free–free emission
   — the L_ν(1.4 GHz)–SFR calibration behind ``l14_sfr``.
   `arXiv:1105.4877 <https://arxiv.org/abs/1105.4877>`_

.. [KennicuttEvans2012] Kennicutt R.C. & Evans N.J. 2012, ARA&A 50, 531.
   Star formation in the Milky Way and nearby galaxies — the total-IR and
   Hα SFR calibrations behind ``lir_sfr`` and ``lha_norm``.
   `arXiv:1204.3552 <https://arxiv.org/abs/1204.3552>`_

.. [Runnoe2012] Runnoe J.C., Brotherton M.S. & Shang Z. 2012, MNRAS 422, 478.
   Updated quasar bolometric corrections — the 1450 Å and 4400 Å values
   behind ``agn_bc_uv`` / ``agn_bc_opt``.
   `arXiv:1201.5155 <https://arxiv.org/abs/1201.5155>`_

.. [Wright2010] Wright E.L. et al. 2010, AJ 140, 1868.
   The Wide-field Infrared Survey Explorer (WISE) — the 3.4/4.6/12 μm bands
   the tier-3 IR intensity maps emulate.
   `arXiv:1008.0031 <https://arxiv.org/abs/1008.0031>`_

.. [Dore2014] Doré O. et al. 2014, arXiv e-prints.
   Cosmology with the SPHEREx all-sky spectral survey — the near-IR
   intensity-mapping context of the tier-3 IR maps.
   `arXiv:1412.4872 <https://arxiv.org/abs/1412.4872>`_

.. [Bacon2018] Bacon D.J. et al. (SKA Cosmology SWG) 2020, PASA 37, e007.
   Cosmology with Phase 1 of the Square Kilometre Array (Red Book 2018) —
   the radio continuum survey spec the tier-3 radio maps emulate.
   `arXiv:1811.02743 <https://arxiv.org/abs/1811.02743>`_

.. [Behroozi2019] Behroozi P. et al. 2019, MNRAS 488, 3143.
   UniverseMachine: empirical galaxy–halo connection with per-halo SFR
   histories — the reference empirical model for SFR-resolved occupations.
   `arXiv:1806.07893 <https://arxiv.org/abs/1806.07893>`_

.. [Yang2007groups] Yang X., Mo H.J. & van den Bosch F.C. 2007, ApJ 671, 153.
   SDSS DR4 galaxy group catalogue — quenched fractions vs group halo mass.
   `arXiv:0707.4640 <https://arxiv.org/abs/0707.4640>`_

.. [Tinker2021groups] Tinker J.L. 2021, ApJ 923, 154.
   Self-calibrating halo-based group finder — the modern f_Q(M_h) data.
   `arXiv:2010.02946 <https://arxiv.org/abs/2010.02946>`_

.. [Zhang2024CGM] Zhang Y. et al. 2024, A&A 690, A268.
   eROSITA hot CGM II: L_X–mass scaling relations of central galaxies.
   `arXiv:2401.17309 <https://arxiv.org/abs/2401.17309>`_

.. [Zhang2025CGM] Zhang Y. et al. 2025, A&A 693, A197.
   eROSITA hot CGM III: star-forming vs quiescent galaxies — the measurement
   an SF/Q-split hot-gas sector must fit.
   `arXiv:2411.19945 <https://arxiv.org/abs/2411.19945>`_

.. [Truong2021] Truong N., Pillepich A. & Nelson D. 2021, MNRAS 508, 1563.
   TNG predictions linking CGM X-ray properties to galaxy sSFR.
   `arXiv:2109.06884 <https://arxiv.org/abs/2109.06884>`_

.. [Banerjee2025] Banerjee A. et al. 2025, PASA 42, 78.
   AGN vs star-forming galaxies at fixed stellar mass: colour, D4000,
   morphology and clustering differences.
   `arXiv:2310.12943 <https://arxiv.org/abs/2310.12943>`_

.. [Yang2019BHbulge] Yang G. et al. 2019, MNRAS 485, 3721.
   Black-hole growth traces bulge growth — the coupling that puts B/T into
   the M_BH–M* step of the Powell chain.
   `arXiv:1903.00003 <https://arxiv.org/abs/1903.00003>`_

.. [Ni2019] Ni Q. et al. 2019, MNRAS 490, 1135.
   BH growth vs host compactness at fixed M* — morphology as a second
   parameter of AGN occupation.
   `arXiv:1909.06382 <https://arxiv.org/abs/1909.06382>`_

----

Galaxy Luminosity Functions and SEDs
--------------------------------------

Multi-band luminosity functions and the stellar-population calibrations for a
band-resolved conditional luminosity function (see :doc:`missing_physics`).

.. [Kauffmann2003] Kauffmann G. et al. 2003, MNRAS 341, 33.
   Stellar masses and star-formation histories for 10^5 SDSS galaxies — the
   M*/L calibration anchor for mass-to-light parameterisations.
   `arXiv:astro-ph/0204055 <https://arxiv.org/abs/astro-ph/0204055>`_

.. [Faber2007] Faber S.M. et al. 2007, ApJ 665, 265.
   B-band luminosity functions to z≈1 (DEEP2/COMBO-17) and the red-sequence
   build-up.
   `arXiv:astro-ph/0506044 <https://arxiv.org/abs/astro-ph/0506044>`_

.. [Comparat2015OII] Comparat J. et al. 2015, A&A 575, A40.
   Evolution of the bright end of the [OII] luminosity function — the
   emission-line LF target for an SFR→line extension.
   `arXiv:1408.1523 <https://arxiv.org/abs/1408.1523>`_

.. [Shuntov2025] Shuntov M. et al. 2025, A&A 695, A20.
   COSMOS-Web stellar-mass assembly in relation to dark-matter halos over
   0.2 < z < 12.
   `arXiv:2410.08290 <https://arxiv.org/abs/2410.08290>`_

.. [Xu2025PAC] Xu K. et al. 2025, MNRAS 540, 1635.
   PAC in DESI: the galaxy stellar mass function into the 10^6 M_sun frontier.
   `arXiv:2503.01948 <https://arxiv.org/abs/2503.01948>`_

.. [EuclidCosmicDawn2025] Euclid Collaboration (Zalesky L.) et al. 2025.
   Cosmic Dawn Survey: galaxy stellar mass function over 0.2 < z < 6.5 on
   10 deg².
   `arXiv:2504.17867 <https://arxiv.org/abs/2504.17867>`_

----

Stellar Feedback and Galactic Winds
-------------------------------------

Supernova-driven winds and the simulation suites that calibrate them
(see :doc:`missing_physics`; AGN feedback references live in the
Baryonic-Effects group above).

.. [SomervilleDave2015] Somerville R.S. & Davé R. 2015, ARA&A 53, 51.
   Review of physical models of galaxy formation — the canonical forms of
   energy- and momentum-driven wind mass loading.
   `arXiv:1412.2712 <https://arxiv.org/abs/1412.2712>`_

.. [Muratov2015] Muratov A.L. et al. 2015, MNRAS 454, 2691.
   FIRE galactic winds: η_w ∝ V_c^{-1} (momentum) to V_c^{-2} (energy)
   mass-loading calibrations.
   `arXiv:1501.03155 <https://arxiv.org/abs/1501.03155>`_

.. [Chisholm2017] Chisholm J., Tremonti C.A., Leitherer C. & Chen Y. 2017, MNRAS 469, 4831.
   Measured mass and momentum outflow rates of photoionised galactic winds —
   external priors on the wind sector.
   `arXiv:1702.07351 <https://arxiv.org/abs/1702.07351>`_

.. [PillepichTNG2018] Pillepich A. et al. 2018, MNRAS 473, 4077.
   The IllustrisTNG galaxy-formation model — the SN-wind + AGN subgrid
   pairing modern hydro suites converge on.
   `arXiv:1703.02970 <https://arxiv.org/abs/1703.02970>`_

.. [CAMELS2023] Villaescusa-Navarro F. et al. 2023, ApJS 265, 54.
   CAMELS public data release: thousands of hydro simulations varying SN and
   AGN feedback — the validation grid for a freed feedback sector.
   `arXiv:2201.01300 <https://arxiv.org/abs/2201.01300>`_

.. [Eckert2021] Eckert D., Gaspari M., Gastaldello F. et al. 2021, Universe 7, 142.
   Feedback in galaxy groups: the mass scale where SN and AGN feedback
   prescriptions diverge most between simulations.
   `arXiv:2106.13259 <https://arxiv.org/abs/2106.13259>`_

----

Cold Gas and Neutral Hydrogen
-------------------------------

The HI halo model, cold-gas scaling relations and 21 cm data
(see :doc:`missing_physics`).

.. [VillaescusaNavarro2018HI] Villaescusa-Navarro F. et al. 2018, ApJ 866, 135.
   Ingredients for 21 cm intensity mapping: the M_HI(M_h, z) halo model and
   HI profiles from IllustrisTNG — the form proposed for ``hod_mod``.
   `arXiv:1804.09180 <https://arxiv.org/abs/1804.09180>`_

.. [Jones2018ALFALFA] Jones M.G. et al. 2018, MNRAS 477, 2.
   The ALFALFA HI mass function — the z≈0 abundance anchor for the HI sector.
   `arXiv:1802.00053 <https://arxiv.org/abs/1802.00053>`_

.. [Catinella2018xGASS] Catinella B. et al. 2018, MNRAS 476, 875.
   xGASS: cold-gas scaling relations and atomic-to-molecular ratios of local
   galaxies — the M_HI(M*, sSFR) conditional data.
   `arXiv:1802.02373 <https://arxiv.org/abs/1802.02373>`_

.. [Obuljen2019] Obuljen A. et al. 2019, MNRAS 486, 5124.
   The HI content of dark-matter halos at z≈0 from ALFALFA — the empirical
   HI HOD.
   `arXiv:1805.00934 <https://arxiv.org/abs/1805.00934>`_

.. [GuoNUM2023] Guo H. et al. 2023, ApJ 955, 57.
   NeutralUniverseMachine: empirical HI/H2 evolution on the UniverseMachine
   galaxy–halo connection (HDR reference for the cold-gas outlook).
   `arXiv:2307.07078 <https://arxiv.org/abs/2307.07078>`_

.. [Nishigaki2025] Nishigaki M. et al. 2025, ApJ 984, 135.
   ChemicalUniverseMachine: metals in the galaxy–ISM–CGM ecosystem.
   `arXiv:2503.10999 <https://arxiv.org/abs/2503.10999>`_

.. [CHIME2023] CHIME Collaboration 2023, ApJ 947, 16.
   Detection of cosmological 21 cm emission in cross-correlation with eBOSS
   tracers — the proof of principle for C_ell^{HI×g}.
   `arXiv:2202.01242 <https://arxiv.org/abs/2202.01242>`_

.. [Ponomareva2023] Ponomareva A.A. et al. 2023, MNRAS 522, 5308.
   MIGHTEE-HI: the first MeerKAT HI mass function from an untargeted
   interferometric survey.
   `arXiv:2304.13051 <https://arxiv.org/abs/2304.13051>`_

.. [SKA2019] Braun R. et al. 2019, arXiv:1912.12699.
   Anticipated performance of SKA1 — the sensitivity reference for HI
   intensity-mapping forecasts.
   `arXiv:1912.12699 <https://arxiv.org/abs/1912.12699>`_

.. [HI4PI2016] HI4PI Collaboration 2016, A&A 594, A116.
   Full-sky Galactic HI survey — used in ``hod_mod`` only as the Galactic
   absorption (N_H) template, *not* as extragalactic cold gas.
   `arXiv:1610.06175 <https://arxiv.org/abs/1610.06175>`_

----

Sensitivity-Benchmark Data Anchors
------------------------------------

The published measurements compiled on :doc:`sensitivity_benchmark` — the
"already existing observables" the forward model must reproduce, and the
current state of the art per probe.

.. [Zehavi2011] Zehavi I. et al. 2011, ApJ 736, 59.
   SDSS DR7 projected correlation functions w_p(r_p) per luminosity/colour
   sample — the reference low-z clustering benchmark.
   `arXiv:1005.2413 <https://arxiv.org/abs/1005.2413>`_

.. [Wright2025] Wright A.H. et al. 2025, A&A.
   KiDS-Legacy cosmic shear (1347 deg², nine bands):
   S8 = 0.815 (+0.016/−0.021) — the current cosmic-shear state of the art.
   `arXiv:2503.19441 <https://arxiv.org/abs/2503.19441>`_

.. [Qu2024] Qu F.J. et al. 2024, ApJ 962, 112.
   ACT DR6 CMB-lensing power spectrum: 43σ (2.3 % amplitude), matching
   Planck PR4 — the C_ell^{κκ_CMB} benchmark.
   `arXiv:2304.05202 <https://arxiv.org/abs/2304.05202>`_

.. [Kim2024] Kim J. et al. 2024, JCAP 12, 022.
   DESI LRG × ACT DR6 CMB lensing in four tomographic bins (0.4 ≤ z ≤ 1):
   38σ (50σ with Planck PR4), S8 to 2.7 % — the C_ell^{gκ_CMB} benchmark.
   `arXiv:2407.04606 <https://arxiv.org/abs/2407.04606>`_

.. [Ghirardini2024] Ghirardini V. et al. 2024, A&A.
   eRASS1 cluster-abundance cosmology (5259 clusters, 12 791 deg²):
   σ8 = 0.88 ± 0.02, S8 = 0.86 ± 0.01, Σm_ν < 0.43 eV — the live X-ray
   cluster-counts (``ncl``) benchmark.
   `arXiv:2402.08458 <https://arxiv.org/abs/2402.08458>`_

.. [DESI2024FS] DESI Collaboration 2025, JCAP 07, 028 (DESI 2024 VII).
   DR1 full-shape clustering: σ8 = 0.842 ± 0.034 alone,
   σ8 to 0.65 % with CMB — the RSD/full-shape benchmark.
   `arXiv:2411.12022 <https://arxiv.org/abs/2411.12022>`_

.. [DESIDR2] DESI Collaboration 2025, Phys. Rev. D 112, 083515.
   DR2 BAO from >14 million tracers; with CMB a 3.1σ preference for
   w0waCDM over ΛCDM — the geometric benchmark for the freed (w0, wa).
   `arXiv:2503.14738 <https://arxiv.org/abs/2503.14738>`_

.. [Schaan2021] Schaan E. et al. (ACT) 2021, Phys. Rev. D 103, 063513.
   kSZ + tSZ profiles of BOSS CMASS galaxies with ACT — the measured
   gas-density/pressure profiles a kSZ observable would be fit against.
   `arXiv:2009.05557 <https://arxiv.org/abs/2009.05557>`_

.. [CHIMEauto2025] CHIME Collaboration 2025 (preprint).
   First detection of the cosmological 21 cm auto-power spectrum at z ≈ 1
   with CHIME — the C_ell^{HI×HI} proof of principle.
   `arXiv:2511.19620 <https://arxiv.org/abs/2511.19620>`_

.. [Bonato2021] Bonato M. et al. 2021, A&A 656, A48.
   LoTSS Deep Fields: 150 MHz luminosity of star-forming galaxies as an SFR
   tracer — the measured SF radio LF behind ``l14_sfr``/``rlf``.
   `arXiv:2109.06735 <https://arxiv.org/abs/2109.06735>`_

.. [Kondapally2022] Kondapally R. et al. 2022, MNRAS 513, 3742.
   LoTSS Deep Fields: cosmic evolution of low-excitation radio galaxies to
   z ≈ 2.5 — the radio-AGN (jet-mode) LF benchmark.
   `arXiv:2204.07588 <https://arxiv.org/abs/2204.07588>`_

.. [Weaver2023] Weaver J.R. et al. 2023, A&A 677, A184.
   COSMOS2020 galaxy stellar-mass function 0.2 < z < 5.5, total and
   quiescent — the SMF + quenched-fraction benchmark for the (z, M*) grid.
   `arXiv:2212.02512 <https://arxiv.org/abs/2212.02512>`_

.. [Driver2022] Driver S.P. et al. 2022, MNRAS 513, 439.
   GAMA DR4: the low-z galaxy stellar-mass function over 250 deg² —
   the local SMF anchor.
   `arXiv:2203.08539 <https://arxiv.org/abs/2203.08539>`_

.. [Popesso2023] Popesso P. et al. 2023, MNRAS 519, 1526.
   The main sequence of star-forming galaxies over 0 < z < 6 — the
   sSFR(M*, z) benchmark behind ``ssfr_ms_*``.
   `arXiv:2203.10487 <https://arxiv.org/abs/2203.10487>`_

.. [Kulkarni2019] Kulkarni G., Worseck G. & Hennawi J.F. 2019, MNRAS 488, 1035.
   Homogenised type-1 quasar luminosity functions 0 < z < 7.5 — the
   ``qlf_uv``/``qlf_opt`` benchmark.
   `arXiv:1807.09774 <https://arxiv.org/abs/1807.09774>`_

.. [KormendyHo2013] Kormendy J. & Ho L.C. 2013, ARA&A 51, 511.
   The coevolution of supermassive black holes and their host galaxies —
   the M_BH–M_bulge census that pins the ``agn_mu_bh`` sector externally.
   `arXiv:1304.7762 <https://arxiv.org/abs/1304.7762>`_

.. [Greene2020] Greene J.E., Strader J. & Ho L.C. 2020, ARA&A 58, 257.
   The demographics of intermediate-mass and low-mass-galaxy black holes —
   extends the M_BH census to the tier-3 M* < 10¹⁰ regime.
   `arXiv:1911.09678 <https://arxiv.org/abs/1911.09678>`_

.. [Sobral2013] Sobral D. et al. 2013, MNRAS 428, 1128.
   HiZELS: Hα luminosity functions at z = 0.4–2.2 — the ``half``
   observable benchmark.
   `arXiv:1202.3436 <https://arxiv.org/abs/1202.3436>`_

.. [Wyder2005] Wyder T.K. et al. 2005, ApJ 619, L15.
   GALEX local UV luminosity functions — the low-z ``uvlf`` anchor.
   `arXiv:astro-ph/0411364 <https://arxiv.org/abs/astro-ph/0411364>`_

.. [Moustakas2013] Moustakas J. et al. 2013, ApJ 767, 50.
   PRIMUS: stellar-mass functions and quenching 0 < z < 1 — a core
   constraint of the UniverseMachine compilation.
   `arXiv:1301.1688 <https://arxiv.org/abs/1301.1688>`_

.. [Song2016] Song M. et al. 2016, ApJ 825, 5.
   UV–stellar-mass relations at z = 4–8 from CANDELS SED stacks — a
   UniverseMachine constraint re-derived in their Appendix D.
   `arXiv:1507.05636 <https://arxiv.org/abs/1507.05636>`_

.. [Finkelstein2015] Finkelstein S.L. et al. 2015, ApJ 810, 71.
   UV luminosity functions at z = 4–8 from CANDELS/HUDF — a
   UniverseMachine UVLF constraint.
   `arXiv:1410.5439 <https://arxiv.org/abs/1410.5439>`_

.. [Harikane2023] Harikane Y. et al. 2023, ApJS 265, 5.
   JWST UV luminosity functions at z ≈ 9–16 — the current high-z frontier
   of the UVLF benchmark.
   `arXiv:2208.01612 <https://arxiv.org/abs/2208.01612>`_

.. [Macquart2020] Macquart J.-P. et al. 2020, Nature 581, 391.
   The FRB dispersion-measure–redshift relation: a direct census of the
   ionised cosmic baryons — a published probe of the same hot-gas sector.
   `arXiv:2005.13161 <https://arxiv.org/abs/2005.13161>`_

.. [Kollmeier2017] Kollmeier J.A. et al. 2017, arXiv e-prints.
   SDSS-V: Pioneering Panoptic Spectroscopy — the Black Hole Mapper
   time-domain spectroscopic M_BH census now underway.
   `arXiv:1711.03234 <https://arxiv.org/abs/1711.03234>`_

.. [Powell2022] Powell M.C. et al. 2022, ApJ 938, 77 (BASS XXXVI).
   Constraining the local SMBH–halo connection by forward-modelling the
   clustering + luminosity function of Swift/BAT AGN — the AGN–halo model
   implemented as :class:`hod_mod.agn.powell.PowellAGNModel`.
   `arXiv:2209.02728 <https://arxiv.org/abs/2209.02728>`_

.. [Ananna2022] Ananna T.T. et al. 2022, ApJS 261, 9 (BASS XXX).
   Distribution functions of BASS DR2 Eddington ratios, black-hole masses
   and X-ray luminosities — the broken-power-law ERDF used by the Powell
   chain (``agn_log10_lstar``, ``agn_delta1``, ``agn_delta2``).
   `arXiv:2201.05603 <https://arxiv.org/abs/2201.05603>`_
