.. _timing_joint_model:

Computation time: wp + ESD + Galaxy×X-ray
==========================================

This page describes the wall-clock cost of evaluating the joint
:math:`w_p(r_p) + \Delta\Sigma(R) + w_\theta(\theta)` model used in
:ref:`bgs_ls10_wp_survey` and the joint LSDR10 analysis, and documents the
parallelisation strategy applied to keep single-evaluation time manageable.

.. contents::
   :depth: 2
   :local:

---

Overview
--------

The joint model combines three probes, each with its own computational profile:

.. list-table::
   :header-rows: 1
   :widths: 35 30 15 20

   * - Probe
     - Predictor
     - Parameters
     - Cost
   * - :math:`w_p(r_p)` — projected clustering
     - ``FullHaloModelPrediction.wp()``
     - 5 HOD
     - < 1 s (post JIT)
   * - :math:`\Delta\Sigma(R)` — weak-lensing ESD
     - ``FullHaloModelPrediction.delta_sigma()``
     - 5 HOD
     - 1–5 s
   * - :math:`w_\theta(\theta)` — galaxy × X-ray
     - ``HaloModelCrossSpectra.angular_cl_gX()``
     - 5 HOD + 2 amplitude
     - see below

The dominant bottleneck is ``angular_cl_gX``:

.. code-block:: text

    angular_cl_gX (serial)
     └── for z in z_arr (5 pts × ~180 s each):
          └── _pk_tables_gX(z, …)
               └── GasDensityDPM.emissivity_uk
                    [Gauss-Legendre 200 pts × 200 mass × 80 k = 3.2 M ops per z-pt]

At five redshift points (narrow :math:`n(z)` around :math:`z_{\rm mean}`),
this amounts to **~900 s serial** per sample.

---

Component breakdown — sample S1
---------------------------------

Sample S1: :math:`\log_{10} M_* > 10.0`, :math:`z_{\rm mean} = 0.135`.

.. list-table::
   :header-rows: 1
   :widths: 45 20 20 15

   * - Phase
     - Serial [s]
     - Parallel [s]
     - Speed-up
   * - Infrastructure build (CAMB + HMF + JAX warm-up)
     - ~20
     - —
     - —
   * - :math:`w_p(r_p)` — More+2015 HOD (post JIT)
     - < 1
     - —
     - —
   * - :math:`\Delta\Sigma(R)` — More+2015 HOD
     - 1–5
     - —
     - —
   * - :math:`w_p + \Delta\Sigma` combined
     - 2–6
     - —
     - —
   * - ``angular_cl_gX`` — :math:`N_z = 5`, :math:`N_\ell = 80`
     - ~900
     - ~180
     - ~5 ×
   * - Full joint evaluation
     - ~906
     - ~186
     - ~5 ×

.. note::
   Run ``python -m hod_mod.scripts.timing.time_joint_model --sample S1``
   to reproduce these numbers on your hardware. Results are written to
   ``results/timing/timing_joint_model.json``.

---

Speedup implementation
-----------------------

Two independent optimisations reduce ``angular_cl_gX`` from ~900 s to ~180 s.

z-loop parallelisation (ThreadPoolExecutor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each call to ``_pk_tables_gX(z_i, …)`` is independent. The serial for-loop is
replaced by a ``ThreadPoolExecutor`` map over redshift points:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    def _tables_at_z(zi):
        return self._pk_tables_gX(zi, theta_cosmo, hod_params, ...)

    with ThreadPoolExecutor(max_workers=min(n_workers, n_z)) as pool:
        raw_tables = list(pool.map(_tables_at_z, z_arr))

JAX releases the GIL during XLA-compiled computation, so thread-based
parallelism is safe and effective on CPU.  With :math:`N_z = 5` points and
at least 5 available cores, this gives the full ~5× speed-up.

Control parallelism with the ``n_workers`` argument
(``-1`` = use all available CPUs, default; ``1`` = serial):

.. code-block:: python

    cl = cross.angular_cl_gX(
        ell_arr, z_arr, nz_g, theta_cosmo, hod_params,
        psf_fwhm_arcsec=30.0,
        n_workers=-1,   # default — all CPUs
    )

ℓ-loop vectorisation (JAX vmap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Limber integral over 80 multipoles is replaced by a single batched JAX
operation.  The key step is building the Limber wave-vector table
:math:`k_{\rm Limber}(\ell, z) = (\ell + 0.5)/\chi(z)` for all
:math:`(\ell, z)` pairs at once and interpolating using ``jax.vmap``:

.. code-block:: python

    k_limber = (ell_j[:, None] + 0.5) / chi_z_j[None, :]   # (Nell, Nz)

    def _interp_one(log_k_query, log_p_table):
        return jnp.exp(jnp.interp(log_k_query, log_k_j, log_p_table))

    _interp_z    = jax.vmap(_interp_one, in_axes=(0, 0))
    _interp_ellz = jax.vmap(_interp_z,   in_axes=(0, None))

    pk_mat    = _interp_ellz(jnp.log(k_limber), log_pgX_stack)   # (Nell, Nz)
    cl_arr    = jnp.trapezoid(
        dndchi_j[None,:] * pk_mat / chi_z_j[None,:]**2,
        chi_z_j, axis=1,
    )   # (Nell,)

This replaces an 80-iteration Python loop with a single XLA-compiled call,
contributing a further ~10× reduction in Limber-integral time.

---

Shape cache
-----------

For MAP and MCMC fits (``fit_comparat2025.py``, ``fit_joint_lsdr10.py``), the
HOD-dependent angular cross-power components (gas, AGN) are cached to disk as
``results/.../shape_cache/{label}_{hash}.npz``.

The cache key is the MD5 of the label + HOD parameter values (6 decimal places)
+ ``|agn|joint`` suffix.  A typical HOD evaluation changes all five parameters,
so the cache is only hit on repeated calls with identical HOD parameters — useful
during MCMC when the accepted chain has repeated evaluations at the same point
or when re-running a fit after a crash.

Benefit: avoids re-running the ~180 s Limber integral for amplitude-only
re-evaluations.  The two amplitude parameters (``log10_A_gas``,
``log10_A_AGN``) are pure scalar multipliers on the cached shapes:

.. math::

   w_\theta(\theta) = 10^{\log A_{\rm gas}} \cdot \hat{w}_{\rm gas}(\theta)
                    + 10^{\log A_{\rm AGN}} \cdot \hat{w}_{\rm AGN}(\theta)

---

Practical runtimes for the joint fit
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Fit stage
     - Time
     - Notes
   * - Infrastructure build
     - ~20 s
     - once per session
   * - Shape computation (wtheta)
     - ~180 s
     - once per HOD proposal (cached)
   * - Single likelihood call (wp + ESD + wtheta)
     - 2–6 s
     - dominated by wp/ESD; wtheta from cache
   * - MAP (L-BFGS-B, ~100–500 iterations)
     - 15–60 min
     - depends on starting point
   * - MCMC (32 walkers × 1000 steps × n_accept)
     - 4–24 h
     - each new HOD ≈ 180 s + ~4 s wp/ESD

.. note::
   MCMC is expensive because each *new* HOD proposal requires recomputing the
   Limber integral.  The 32-walker ensemble moves across HOD space in small steps,
   so many accepted steps share cache hits, but the worst case is ~186 s per
   accepted step.

   For efficient posterior sampling, consider:

   1. Running MAP first (``--mode map``) and using the best-fit as the MCMC seed.
   2. Using a coarser :math:`n(z)` grid (:math:`N_z = 3`) or fewer multipoles
      (:math:`N_\ell = 40`) during burn-in, then full resolution in production.
   3. Fixing the HOD to the MAP best-fit and sampling only the two amplitude
      parameters (``log10_A_gas``, ``log10_A_AGN``) — each call takes < 1 s.

---

How to reproduce
-----------------

.. code-block:: bash

   # Full timing benchmark (serial + parallel phases A–F)
   python -m hod_mod.scripts.timing.time_joint_model --sample S1

   # Repeat wp/ESD measurements three times for stable means
   python -m hod_mod.scripts.timing.time_joint_model --sample S1 --n-repeat 3

   # Results
   cat results/timing/timing_joint_model.json

Joint LSDR10 fit
-----------------

The ``fit_joint_lsdr10.py`` script fits all three statistics simultaneously for
four BGS stellar-mass-threshold samples (S1, S3, S5, S7):

.. code-block:: bash

   # MAP for all four samples
   python -m hod_mod.scripts.fitting.fit_joint_lsdr10 --sample all --mode map

   # MAP + MCMC for S1 only (HSC weak-lensing ESD)
   python -m hod_mod.scripts.fitting.fit_joint_lsdr10 \
       --sample S1 --mode both \
       --esd-survey esd_hsc \
       --rp-min 0.3 --rp-max 30 \
       --R-min 0.1  --R-max 30  \
       --theta-min 8 --theta-max 300

Output files::

    results/fits/joint_lsdr10/
      {S}_map.json          best-fit params, χ²/dof per probe
      {S}_chain.h5          emcee posterior chain (requires h5py)
      {S}_bestfit.pdf       3-panel: wp | ΔΣ | wtheta (with residuals)
      {S}_corner.pdf        7-parameter posterior corner (requires corner)

.. seealso::

   :ref:`bgs_ls10_wp_survey` — wp-only BGS/LS10 fitting.
   :mod:`hod_mod.observables.cross_spectra` — ``HaloModelCrossSpectra`` API.
   :mod:`hod_mod.observables.clustering` — ``FullHaloModelPrediction`` API.
