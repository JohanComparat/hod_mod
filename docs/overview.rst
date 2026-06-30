Overview
========

``hod_mod`` is a JAX-accelerated Python framework for modelling the galaxy–halo
connection (see reviews from [CooraySheth2002]_, [Asgari2023]_). 
Starting from a set of cosmological parameters and a galaxy–halo occupation model, it predicts: 
the observed projected galaxy autocorrelation function :math:`w_p(r_p)` and 
the observed galaxy–matter cross-correlation (excess surface density) :math:`\Delta\Sigma(R)`.


The forward model chain
-----------------------

The pipeline proceeds through six sequential steps:

.. code-block:: text

    Cosmological parameters θ
           │
           ▼
    1. Linear matter power spectrum  P_lin(k, z; θ)     
           │
           ▼
    2. Halo mass function  dn/dM(M, z; θ)               
       Halo bias           b(M, z; θ)
           │
           ▼
    3. Halo profiles  u(k|M) [NFW or Einasto], c-M
           │
           ▼
    4. Galaxy occupation  ⟨N_cen⟩, ⟨N_sat⟩(M; p_HOD)  [HOD / ICSMF / iHOD models]
           │
           ▼
    5. Power spectra  P_gg(k), P_gm(k)            
           ├── Galaxy clustering w_p(r_p; π_max)      
           └── Galaxy-mass lensing ΔΣ(R)              

Step 1 is the computational bottleneck (CAMB takes ~30 s).  In MCMC mode a caching
layer (``CachedPkLinear``) interpolates on a pre-computed grid, reducing per-step cost
to < 1 s.

---

Installation
------------

Requires Python ≥ 3.11, JAX ≥ 0.4, and CAMB.
The package is available on `PyPI <https://pypi.org/project/hod-mod/>`_:

.. code-block:: bash

    pip install hod-mod

For development, create and activate the conda/mamba environment, then install in editable mode:

.. code-block:: bash

    mamba env create -f environment.yml
    mamba activate hod_mod
    pip install -e .

---

Quick start
-----------

Compute the projected correlation function :math:`w_p(r_p)` with a the HOD model from [More2015]_:

.. code-block:: python

    import jax.numpy as jnp
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.core.halo_profiles import HaloProfile
    from hod_mod.connection import MoreHODModel
    from hod_mod.observables import FullHaloModelPrediction

    pk_lin = LinearPowerSpectrum()
    theta  = pk_lin.default_cosmology()       # Planck 2018 best-fit
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)

    colossus_cosmo = dict(flat=True, H0=67.36, Om0=0.31, Ob0=0.0493, sigma8=0.811, ns=0.965)
    hp = HaloProfile(colossus_cosmo, cm_relation="diemer19")

    hod    = MoreHODModel(hmf, hmf.bias)
    pred   = FullHaloModelPrediction(pk_lin, hod, hp, profile="nfw")

    rp     = jnp.logspace(-1, 1.5, 20)
    params = MoreHODModel.default_params()
    wp     = pred.wp(rp, pi_max=60.0, z=0.5, theta_cosmo=theta, hod_params=params)

``"tinker08"`` is the library's dependency-free default HMF backend, used
above for the quickstart. The project's fitting pipelines instead use
``make_hmf("csst")`` (CSSTEMU) as their baseline — see
:doc:`cosmology` for the full list of backends and why.

---

Coordinate and unit conventions
---------------------------------

All spatial quantities are in **h-units** throughout the pipeline:

=========================== ====================== =======================
Quantity                    Symbol                 Unit
=========================== ====================== =======================
Comoving separation         :math:`r, r_p`         Mpc/h
Halo mass                   :math:`M`              :math:`M_\odot/h`
Power spectrum              :math:`P(k)`           :math:`({\rm Mpc}/h)^3`
Wavenumber                  :math:`k`              :math:`h\,{\rm Mpc}^{-1}`
Galaxy number density       :math:`n_g`            :math:`({\rm Mpc}/h)^{-3}`
Stellar Mass Function       :math:`\Phi`           :math:`({\rm Mpc}/h)^{-3}\,{\rm dex}^{-1}`
=========================== ====================== =======================

---

Cosmological parameter dictionary
----------------------------------

All functions that require cosmological parameters expect a Python ``dict`` with these
keys (produced by ``LinearPowerSpectrum.default_cosmology()``):

.. code-block:: python

    theta = {
        "h":              0.6736,   # H₀ / (100 km/s/Mpc)
        "Omega_b":        0.0493,   # baryon density parameter
        "Omega_cdm":      0.2644,   # cold dark matter density
        "Omega_m":        0.3137,   # total matter = Omega_b + Omega_cdm
        "n_s":            0.9649,   # scalar spectral index
        "ln10^{10}A_s":   3.044,    # log amplitude of primordial spectrum
    }

These are the Planck 2018 TT,TE,EE+lowE+lensing best-fit values
(`Planck Collaboration 2020 <https://arxiv.org/abs/1807.06209>`_, Table 2) [PlanckCollaboration2018]_.

---

JAX conventions
---------------

The package follows JAX idioms to enable gradient-based inference:

* Use ``jnp.*`` everywhere inside hot functions; only use numpy ``np.*`` at I/O boundaries.
* Pure functions are decorated with ``@jax.jit``; class methods use
  ``@partial(jax.jit, static_argnums=(0,))``.
* Avoid Python-level ``if``/``for`` inside JIT-compiled code; use ``jax.lax.cond``
  and ``jax.lax.scan``.
* Never mutate arrays in-place (JAX arrays are immutable).

Non-JAX libraries (CAMB, colossus, aemulusnu) are called at explicit **boundaries**;
their outputs are wrapped with ``jnp.asarray()`` before entering the JAX computation
graph.

---

Repository structure
--------------------

.. code-block:: text

    hod_mod/                 organised by observable pipeline over a shared core
    ├── core/                P(k), HMF, halo profiles, distances, concentration, BNL
    ├── connection/          galaxy–halo occupation: hod/ (per-family), CLF, SHAM
    ├── gas/                 hot-gas fields: pressure, density, cooling, metallicity,
    │                        conversions, eROSITA response (X-ray + tSZ ingredients)
    ├── agn/                 AGN X-ray models: xray, ham, hod, duty_cycle
    ├── observables/         the pipelines: clustering (wp, ΔΣ), cross_spectra
    │                        (g×y tSZ + g×X engine), cross_clustering, IA, baryon frac.
    ├── fitting/             models, config, fitters (MAP + emcee), Planck prior
    ├── cli/                 unified ``hod-mod`` command (python -m hod_mod)
    └── data_io/             SumStatReader (HDF5 + FITS), wp/ΔΣ CSV loaders

    hod_mod/scripts/
    ├── cosmology/           demo scripts (P(k), HMF, profiles)
    ├── galaxies/            demo + AGN/gas plotting scripts
    ├── benchmarks/          literature benchmark runner
    └── fitting/
        ├── bgs_ls10/        BGS/LS10 fitting campaign
        ├── mocks/           Uchuu mock fitting campaign
        └── paper_reproductions/

    configs/                 YAML configurations for WpFitter
    results/                 output directory (not tracked by git)
    tests/                   pytest test suite
    data/                    data sets for testing

---

.. _acronyms:

Acronym glossary
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Acronym
     - Expansion
   * - **1h / 2h**
     - 1-halo / 2-halo term — pairs of galaxies within the same halo vs. different halos
   * - **AGN**
     - Active Galactic Nucleus
   * - **BOSS**
     - Baryon Oscillation Spectroscopic Survey (SDSS-III)
   * - **CAMB**
     - Code for Anisotropies in the Microwave Background
   * - **CDM**
     - Cold Dark Matter
   * - **CSMF**
     - Conditional Stellar Mass Function — P(M\ :sub:`*` | M\ :sub:`h`)
   * - **DES**
     - Dark Energy Survey
   * - **eBOSS**
     - Extended Baryon Oscillation Spectroscopic Survey (SDSS-IV)
   * - **EH98**
     - Eisenstein & Hu 1998 — analytical transfer function / power spectrum
   * - **ELG**
     - Emission Line Galaxy
   * - **eRASS**
     - eROSITA All-Sky Survey
   * - **GAMA**
     - Galaxy And Mass Assembly survey
   * - **GP**
     - Gaussian Process emulator
   * - **HMC**
     - Hamiltonian Monte Carlo
   * - **HMF**
     - Halo Mass Function — dn/dM or dn/d ln M
   * - **HOD**
     - Halo Occupation Distribution — P(N | M)
   * - **ICSMF**
     - Inverse Conditional Stellar Mass Function
   * - **iHOD**
     - Inverse HOD — galaxy assignment derived by inverting the SHMR (Zu & Mandelbaum 2015)
   * - **JAX**
     - Google's library for high-performance numerical computing with autodiff and JIT
   * - **JIT**
     - Just-In-Time compilation (via XLA, used by JAX)
   * - **ΛCDM**
     - Lambda Cold Dark Matter — the standard cosmological model
   * - **LRG**
     - Luminous Red Galaxy
   * - **MAP**
     - Maximum A Posteriori estimate
   * - **MCMC**
     - Markov Chain Monte Carlo
   * - **NFW**
     - Navarro-Frenk-White (1997) dark matter halo density profile
   * - **NUTS**
     - No-U-Turn Sampler — gradient-based MCMC implemented in numpyro
   * - **P(k)**
     - Matter power spectrum
   * - **SDSS**
     - Sloan Digital Sky Survey
   * - **SHAM**
     - Sub-Halo Abundance Matching
   * - **SHMR**
     - Stellar-to-Halo Mass Relation
   * - **SMF**
     - Stellar Mass Function — Φ(M\ :sub:`*`)
   * - **XLA**
     - Accelerated Linear Algebra — the compiler backend used by JAX
   * - **ΔΣ(R)**
     - Excess Surface Density — a weak gravitational lensing observable
   * - **w\ :sub:`p`\ (r\ :sub:`p`\ )**
     - Projected galaxy two-point correlation function — the clustering observable

---

Citing this work
-----------------

If you use ``hod_mod`` in published research, please cite: 
`Comparat et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.173C>`_ (A&A 697, A173)

---


