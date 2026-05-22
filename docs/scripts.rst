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

---

Fitting campaigns
-----------------

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
