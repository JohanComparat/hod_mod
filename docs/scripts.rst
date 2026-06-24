Scripts
=======

All runnable entry-points live under ``scripts/``, organised by topic.
Demo scripts are self-contained and produce a matplotlib figure.
Fitting scripts read survey data and write results to ``results/``.

The legacy ``scripts/nb_*.py`` and the old batch shells (``run_fit_*.sh``)
are deprecated; use the per-campaign scripts described here instead.

---

Directory map
-------------

.. code-block:: text

    scripts/
    ├── cosmology/               demo scripts — cosmological quantities
    ├── galaxies/                demo scripts — HOD, SHAM, clustering
    ├── agn/                     demo scripts — AGN luminosity functions
    ├── fitting/
    │   ├── bgs_ls10/            HOD fitting on BGS/LS10 real data
    │   ├── mocks/               HOD fitting on Uchuu mock data
    │   ├── gama/                SMF visualisation + SHAM overlay for GAMA
    │   ├── cosmos/              SMF visualisation + SHAM overlay for COSMOS
    │   └── paper_reproductions/ reproduce published HOD fits
    ├── utils/                   batch helpers and diagnostic tools
    ├── run_pipeline.py          one-shot forward-model evaluation
    ├── run_inference.py         numpyro HMC posterior sampling
    └── fit_hod_wp.py            thin wrapper around WpFitter (legacy configs)

---

Cosmology demos
---------------

``scripts/cosmology/demo_distances.py``
    Computes and plots comoving distance :math:`\chi(z)`, angular diameter distance
    :math:`D_A(z)`, luminosity distance :math:`D_L(z)`, and lookback time for the
    Planck 2018 cosmology.
    **Runtime**: < 5 s.

``scripts/cosmology/demo_power_spectrum.py``
    Plots the linear matter power spectrum :math:`P(k,z)` from CAMB and the
    Eisenstein-Hu 1998 fitting formula, and the non-linear :math:`P(k)` from the
    Aletheia emulator.
    **Runtime**: 30–60 s (CAMB run).

``scripts/cosmology/demo_power_spectrum_diff.py``
    Computes logarithmic derivatives :math:`d\ln P / d\ln\theta_i` by finite
    differences around the Planck 2018 best-fit for each cosmological parameter.
    **Runtime**: 2–4 min (one CAMB run per parameter).

``hod_mod/scripts/cosmology/plot_nonlinear_power_spectrum.py``
    Generates ``docs/_images/fig01b_nonlinear_power_spectrum.png`` comparing all
    available non-linear P(k) backends:
    :class:`~hod_mod.cosmology.nonlinear.NonLinearPowerSpectrum` (Aletheia),
    :class:`~hod_mod.cosmology.nonlinear.HALOFITSpectrum` (HMcode-2020, Takahashi+2012),
    and :class:`~hod_mod.cosmology.nonlinear.WHMSpectrum` (WHM, when WHM-CAMB is installed).
    **Runtime**: 1–3 min (two CAMB runs + Aletheia).
    Run as: ``JAX_PLATFORMS=cpu python -m hod_mod.scripts.cosmology.plot_nonlinear_power_spectrum``

``hod_mod/scripts/cosmology/plot_hmf_bias.py``
    Generates three HMF/bias documentation figures:
    ``fig02_hmf.png`` (fiducial dn/dM and b(M) with ±3σ S8 variation),
    ``fig02a_hmf_models.png`` (six multiplicity-function models via
    ``fsigma_*`` functions in :mod:`hod_mod.cosmology.halo_mass_function`),
    ``fig02b_bias_models.png`` (Tinker+2010 bias redshift evolution).
    **Runtime**: < 60 s.
    Run as: ``JAX_PLATFORMS=cpu python -m hod_mod.scripts.cosmology.plot_hmf_bias``

``scripts/cosmology/demo_halo_mass_function.py``
    Plots :math:`dn/dM`, :math:`\sigma(M)`, and halo bias :math:`b(M)` at several
    redshifts using the Tinker+2008 multiplicity function.
    **Runtime**: < 30 s.

``scripts/cosmology/demo_halo_profiles.py``
    Plots NFW and Einasto 3D density profiles :math:`\rho(r)`, projected surface
    density :math:`\Sigma(R)`, and excess surface density :math:`\Delta\Sigma(R)`.
    **Runtime**: < 30 s.

---

Galaxy / HOD demos
------------------

``scripts/galaxies/demo_hod_models.py``
    Compares the central occupation :math:`\langle N_{\rm cen}(M)\rangle` and satellite
    occupation :math:`\langle N_{\rm sat}(M)\rangle` for the five main HOD parametrisations
    (Zheng+2007, More+2015, Guo+2018, ZM15 iHOD) as a function of halo mass.
    **Runtime**: < 10 s.

``scripts/galaxies/demo_clustering_full.py``
    Predicts :math:`w_p(r_p)` and :math:`\Delta\Sigma(R)` using
    ``FullHaloModelPrediction`` (1-halo + 2-halo) for a BOSS CMASS-like HOD.
    **Runtime**: 1–2 min.

``scripts/galaxies/plot_agn_ham_model.py``
    Verification script for :class:`~hod_mod.galaxies.agn_ham.HamAGNModel`.
    Generates four figures: the HAM luminosity mapping
    (:math:`\log L_X^{\rm hard}` vs. :math:`\log M_h` and :math:`\log M_*`),
    the hard XLF check (HAM prediction vs. Aird+2015 / Ueda+2014 references),
    the predicted soft (0.5–2 keV) XLF vs. Hasinger+2005, and the obscuration
    type-fraction curves from the Comparat+2019 model.
    **Runtime**: ~30 s (two model instantiations at ~12 s each).

``scripts/galaxies/generate_kcorr_table.py``
    Generates the X-ray K-correction grid bundled in
    ``hod_mod/data/agn/v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt``.
    Integrates an absorbed power-law spectrum (:math:`\Gamma = 1.9`,
    solar abundances, :math:`f_{\rm scat} = 0.02`) over a
    35 :math:`\times` 16 grid of :math:`(z, \log N_H)` values.
    Only needs to be re-run if the spectral model assumptions change.
    **Runtime**: < 5 s.

``scripts/galaxies/plot_kcorr_table.py``
    Visualises the K-correction grid as a 2D colour map and a set of
    fixed-redshift slices (see Figure HAM-0 in :doc:`galaxies`).
    **Runtime**: < 5 s.

``scripts/galaxies/plot_erosita_psf.py``
    Validates the analytic King-profile PSF model against the eROSITA TM CalDB
    2D PSF images (``caldb_221121v03``, soft band 0.5–2 keV, TM1–TM7).
    Produces a three-panel figure (see Figure PSF-1 in :doc:`galaxies`):

    * **Left** — azimuthally-averaged radial profiles for each TM plus their mean,
      overplotted with the fitted King profile and same-FWHM Gaussian.
    * **Centre** — fractional residuals (TM mean − King fit) / TM mean.
    * **Right** — PSF window :math:`B_\ell` in harmonic space, comparing analytic
      King, Gaussian, and numerically-transformed tabulated profiles.

    Fitted parameters: :math:`\theta_c = 8.64''`, :math:`\alpha = 1.502`,
    FWHM = 13.2'' (on-axis).
    Requires the CalDB files at
    ``/home/comparat/data/erosita/caldb_221121v03/caldb/srv-0500-2000/tm{1..7}_2dpsf_221121v03.fits``
    and ``astropy`` in the active environment.
    Output: ``results/psf/erosita_psf_king_fit.png``.
    **Runtime**: ~30 s.

---

Fitting campaigns
-----------------

LSDR10 joint fit — wp + ESD + galaxy×X-ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hod_mod/scripts/fitting/fit_joint_lsdr10.py``

Fits :math:`w_p(r_p)`, :math:`\Delta\Sigma(R)`, and :math:`w_\theta(\theta)`
(galaxy × soft X-ray) simultaneously for BGS stellar-mass-threshold samples
(S1, S3, S5, S7) from the DESI Legacy Survey DR10.

Data sources:

* **WP + ESD**: ``sum_stat`` HDF5 files via :class:`~hod_mod.data_io.sum_stat_reader.SumStatReader`
* **Galaxy × X-ray**: zenodo record 15111974, ``LSDR10_GALxEVT/Measurements_Xcorr_Stacks/XCORR/``

Model: More+2015 HOD (5 params) + DPM gas amplitude + AGN amplitude = **7 free parameters**.

.. code-block:: bash

   # MAP for all four samples
   python -m hod_mod.scripts.fitting.fit_joint_lsdr10 --sample all --mode map

   # MAP + MCMC for S1 (HSC weak lensing)
   python -m hod_mod.scripts.fitting.fit_joint_lsdr10 \
       --sample S1 --mode both --esd-survey esd_hsc

   # Custom scale cuts
   python -m hod_mod.scripts.fitting.fit_joint_lsdr10 \
       --rp-min 0.3 --rp-max 30 --R-min 0.1 --theta-min 8

Outputs per sample::

    results/fits/joint_lsdr10/{S}_map.json        # best-fit + χ²/dof
    results/fits/joint_lsdr10/{S}_chain.h5         # MCMC chain
    results/fits/joint_lsdr10/{S}_bestfit.pdf      # 3-panel fit plot
    results/fits/joint_lsdr10/{S}_corner.pdf       # posterior corner

See :ref:`timing_joint_model` for a detailed discussion of computation times
and the parallelisation strategy used for the X-ray Limber integral.

**Estimated runtime**: MAP ~15–60 min per sample; MAP+MCMC 4–24 h per sample.

BGS/LS10 real data — More+2015 HOD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input data: ``sum_stat`` HDF5 files in
``~/software/sum_stat/data/twopcf/``.
File pattern::

    LS10_VLIM_ANY_Mstar{MSTAR_LO}-12.0_z{Z_MIN}-{Z_MAX}-wp-pimax100-sys-comb.h5

Run a single stellar-mass bin::

    python scripts/fitting/bgs_ls10/fit_ls10_more2015.py --mstar 10.5 --plot

Run all six bins sequentially::

    bash scripts/fitting/bgs_ls10/run_ls10_batch.sh

CLI options:

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--mstar``
     - required
     - Stellar-mass lower edge: 9.0, 9.5, 10.0, 11.0, 11.25, 11.5
   * - ``--vary-cosmo``
     - off
     - Add Planck 2018 3σ Gaussian priors on h, Ω_m, n_s, ln10¹⁰A_s
   * - ``--map-only``
     - off
     - Stop after MAP (Nelder-Mead); skip emcee MCMC
   * - ``--plot``
     - off
     - Save diagnostic w_p bestfit figure to ``results/bgs_ls10/``

Outputs (per mass bin)::

    results/bgs_ls10/mstar{XX}/map_result.json   — best-fit params, χ²/dof
    results/bgs_ls10/mstar{XX}/flatchain.npz     — emcee posterior samples
    results/bgs_ls10/mstar{XX}/wp_bestfit.pdf    — diagnostic figure

**Estimated runtime**: MAP ~ 2 min; MAP + 500 MCMC steps ~ 30 min per bin.

Uchuu mock data — More+2015 HOD with Planck prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input data: ``sum_stat`` HDF5 files in
``~/software/sum_stat/data/mocks/twopcf/``.
File pattern::

    MOCK_VLIM_ANY_Mstar{MSTAR}_{Z_MIN}-{Z_MAX}-wp-pimax100.h5

Run a single bin::

    python scripts/fitting/mocks/fit_mocks_more2015.py --mstar 10.0 --plot

Run all nine bins::

    bash scripts/fitting/mocks/run_mocks_batch.sh

By default, Gaussian Planck 2018 3σ priors are applied to cosmological parameters.
Pass ``--wide-cosmo`` to use wide uniform priors instead (useful for prior sensitivity
checks).

Outputs::

    results/mocks/mstar{XX}/map_result.json
    results/mocks/mstar{XX}/flatchain.npz

GAMA stellar mass functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loads FITS files from ``~/software/sum_stat/data/GAMA/`` and overlays a
Moster+2013 SHAM prediction on the observed SMF::

    python scripts/fitting/gama/fit_gama_smf_wp.py
    python scripts/fitting/gama/fit_gama_smf_wp.py --output results/gama/smf_comparison.pdf

.. note::
   A full HOD-based SMF fit requires a differential conditional stellar mass
   function (CSMF) predictor that integrates the HOD over the HMF.  This component
   is planned for a future release.  The current script is a data-exploration and
   validation tool.

COSMOS stellar mass functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same approach over six COSMOS photometric redshift bins (z = 0.2–3.0)::

    python scripts/fitting/cosmos/fit_cosmos_smf_wp.py
    python scripts/fitting/cosmos/fit_cosmos_smf_wp.py --output results/cosmos/smf_comparison.pdf

Paper reproductions
~~~~~~~~~~~~~~~~~~~

``scripts/fitting/paper_reproductions/more2015_boss_cmass.py``
    Reproduces the More+2015 HOD fit to BOSS CMASS :math:`w_p(r_p)`.
    Reads ``configs/more2015_boss_cmass.yml``.
    **Runtime**: MAP ~ 3 min; MAP + MCMC (300 steps × 64 walkers) ~ 2 h.

---

Gas Profile Validation Scripts
---------------------------------

These scripts validate the halo gas profile implementations and produce
comparison figures against published results.  All figures are saved to
``hod_mod/scripts/figures/``.

``scripts/validate_arnaud2010.py``
    Validates the Arnaud+2010 universal pressure profile
    (`arXiv:0910.1234 <https://arxiv.org/abs/0910.1234>`_).

    * Checks all 6 Table 1 parameters (:math:`P_0`, :math:`c_{500}`,
      :math:`\gamma`, :math:`\alpha`, :math:`\beta`, :math:`\alpha_p`).
    * Plots the dimensionless shape :math:`p(x)`, physical
      :math:`P_e(r/R_{500c})` at :math:`z=0` and :math:`z=0.5`, the
      self-similar mass scaling :math:`Y_{\rm SZ} \propto M_{500c}^{5/3+\alpha_p}`,
      and the Fourier transform :math:`\tilde{y}(k|M)`.
    * Exits with ``PASS`` / ``FAIL`` based on parameter agreement with Table 1.

    **Runtime**: ~ 2 min.
    **Output**: ``a10_01_profile_shape.pdf``, ``a10_02_pressure_profile.pdf``,
    ``a10_03_mass_scaling.pdf``, ``a10_04_pressure_uk.pdf``::

        python -m hod_mod.scripts.validate_arnaud2010

``scripts/validate_oppenheimer2025.py``
    Validates the DPM electron density profile
    (`arXiv:2505.14782 <https://arxiv.org/abs/2505.14782>`_) for all 3
    calibrated model variants.

    * Checks the normalization :math:`n_e(0.3\,R_{200} | M_{200}=10^{12}
      M_\odot/h,\,z=0) = n_{e,03}` for each model.
    * Plots profile shapes, mass scaling (:math:`\beta`), redshift scaling,
      density FT :math:`\tilde{n}_e(k|M)`, and emissivity FT
      :math:`\tilde\varepsilon(k|M)`.

    **Runtime**: ~ 3 min.
    **Output**: ``dpm_01_profiles_3models.pdf`` through
    ``dpm_05_emissivity_uk.pdf``::

        python -m hod_mod.scripts.validate_oppenheimer2025

``scripts/validate_gas_profiles.py``
    Cross-model validation of A10 and DPM (all 3 variants) radial profiles,
    mass scaling, parameter sensitivity, and predicted X-ray/tSZ scaling
    relations (:math:`L_X`, :math:`kT_{\rm ew}`, :math:`Y\cdot D_A^2`) against
    Lovisari+2020, Bulbul+2018, Lovisari+2015, Zhang+2024 and Popesso+2024
    data.

    * Figures 4 and 6 (``gas_04_scaling_relations.pdf``,
      ``gas_06_xray_calibration.pdf``) calibrate the DPM model-2 density and
      pressure mass-slopes (:math:`\beta_n`, :math:`\beta_P`) to the
      S1 MAP best-fit of :mod:`hod_mod.scripts.fitting.fit_comparat2025`,
      rather than to an independent target — so these two figures track the
      production joint fit and should be regenerated whenever that fit is
      rerun.

    **Runtime**: ~ 3 min.
    **Output**: ``gas_01_radial_profiles.pdf`` through
    ``gas_06_xray_calibration.pdf``::

        python -m hod_mod.scripts.validate_gas_profiles

``scripts/validate_sz_xray.py``
    End-to-end validation of galaxy × tSZ and galaxy × soft X-ray
    cross-spectra using the A10 pressure profile and DPM density profile.

    Produces 7 panels: A10 pressure profile, pressure FT, :math:`P_{g,y}(k)`
    decomposition, :math:`P_{g,X}(k)` decomposition, projected
    :math:`\Sigma_y(r_p)`, angular :math:`C_\ell^{g,y}`, projected
    :math:`w_{g,X}(r_p)`.

    **Runtime**: ~ 5 min.
    **Output**: ``sz_01_*.pdf`` through ``sz_07_*.pdf``::

        python -m hod_mod.scripts.validate_sz_xray

---

Cross-Correlation Benchmark Scripts
-------------------------------------

These scripts compare hod_mod model predictions against published
galaxy × gas cross-correlation measurements.

``scripts/validate_amodeo2021.py``
    Model predictions for the Amodeo+2021 ACT DR4 × BOSS stacked tSZ benchmark
    (`arXiv:2009.05557 <https://arxiv.org/abs/2009.05557>`_).

    Computes :math:`\Sigma_y(r_p)` (projected tSZ profile) for BOSS CMASS
    (:math:`z_{\rm eff}=0.55`) and LOWZ (:math:`z_{\rm eff}=0.27`) using
    the More+2015 HOD and A10 pressure profile, plus the
    :math:`P_{g,y}(k)` 1h+2h decomposition.

    .. note::
       Measured profiles (Amodeo+2021 Fig. 4) are not included in hod_mod.
       Obtain from `<https://github.com/EmmanuelSchaan/ThumbStack>`_.

    **Runtime**: ~ 5 min.
    **Output**: ``amo21_01_sigma_y_cmass.pdf``, ``amo21_02_sigma_y_lowz.pdf``,
    ``amo21_03_pgy_decomposition.pdf``::

        python -m hod_mod.scripts.validate_amodeo2021

``scripts/validate_pandey2025.py``
    Model predictions for the Pandey+2025 DES Y3 × ACT DR6 lensing × tSZ
    benchmark (`arXiv:2506.07432 <https://arxiv.org/abs/2506.07432>`_).

    Computes :math:`C_\ell^{g,y}` via the Limber approximation for a
    DES Y3-like double-Gaussian :math:`n(z)`, shows the 1h/2h decomposition,
    and plots :math:`P_{g,y}(k)` at three representative redshifts.

    .. note::
       The measured DES Y3 × ACT DR6 :math:`C_\ell^{\gamma,y}` data vector
       is not included in hod_mod.  Obtain from the ACT DR6 data release
       (`<https://github.com/ACTCollaboration>`_).

    **Runtime**: ~ 5 min.
    **Output**: ``pand25_01_cl_gy.pdf``, ``pand25_02_cl_gy_decomposition.pdf``,
    ``pand25_03_pgy_vs_k.pdf``::

        python -m hod_mod.scripts.validate_pandey2025

``scripts/validate_comparat2025.py``
    Full benchmark reproduction for Comparat+2025 galaxy × eROSITA soft X-ray
    (`arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_, A&A 697, A173).

    For each of the 7 stellar-mass-selected samples (S1–S7):

    * Loads :math:`w_\theta(\theta)` from
      ``hod_mod/data/benchmarks/xray/comparat2025_wtheta_{S1..S7}.csv``.
    * Uses the Table 3 HOD (ZuMandelbaum15HODModel) and DPM Model 2.
    * Computes :math:`C_\ell^{g,X}` via Limber → :math:`w_\theta(\theta)`
      via Hankel transform.
    * Fits a free amplitude :math:`A_X` and reports :math:`\chi^2/\nu`.

    **Runtime**: ~ 30 min (7 samples).
    **Output**: ``comparat2025_{S1..S7}_wtheta.pdf``,
    ``comparat2025_all_samples.pdf``::

        python -m hod_mod.scripts.validate_comparat2025

---

Core pipeline scripts
---------------------

``scripts/run_pipeline.py``
    Evaluates the full ``GGAPipeline`` forward model at the default Planck 2018
    cosmology and prints :math:`n_{\rm gal}`, :math:`b_{\rm eff}`, and array shapes::

        python scripts/run_pipeline.py --z 0.3 --hmf tinker08 --plot

``scripts/run_inference.py``
    Runs numpyro NUTS HMC posterior sampling::

        python scripts/run_inference.py --num-warmup 500 --num-samples 1000

``scripts/fit_hod_wp.py``
    Thin CLI wrapper around ``WpFitter`` for use with legacy YAML configs::

        python scripts/fit_hod_wp.py configs/hod_fit_more2015_cmass.yml --plot

---

Utility scripts
---------------

``scripts/utils/gather_inputs_sum_stat.py``
    Scans the ``sum_stat`` data directory and builds a manifest of available HDF5
    files with their stellar-mass and redshift ranges.

``scripts/utils/make_result_figures.py``
    Post-processing script that reads ``results/*/flatchain.npz`` and produces
    corner plots and bestfit overlays.

``scripts/utils/measure_timing.py``
    Profiles the wall-clock time of each pipeline stage (CAMB, HMF, NFW projection,
    HOD integral, w_p Limber integral) with and without JAX JIT warm-up.
