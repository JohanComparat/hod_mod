Differentiable multi-probe inference
====================================

The production forward model is **JAX-differentiable end-to-end**.  Every
observable below is ``jax.jit`` / ``jax.jacfwd`` / ``jax.grad``-able with respect
to the cosmological *and* HOD/gas parameters, which enables gradient-based MAP
optimisation and Hamiltonian Monte Carlo (NUTS) instead of the gradient-free
Powell / ``emcee`` fits in :mod:`hod_mod.fitting.fitters`.

Two differentiable backends
---------------------------

* **Forecast surrogate** — :class:`hod_mod.forecast.forward_jax.ForwardModel`
  computes *every* probe (``wp``, ``ds``, ``cl_gy``, ``cl_gX``, ``cl_XX``,
  ``cl_kk`` cosmic shear, ``xlf``/``wp_agn`` AGN, ``smf``, CMB lensing, …) as one
  ``jacfwd``-able call in the σ8-native EH98 parameterisation.  The X-ray/tSZ legs
  are analytic *surrogates* and the galaxy ``n(z)`` is synthetic (override it with
  the ``galaxy_nz=(z_grid, nz)`` constructor argument).

* **Production, full fidelity** — the ``pk_backend="eh98_jax"`` path of
  :class:`hod_mod.observables.clustering.FullHaloModelPrediction`
  (built via :func:`hod_mod.observables.make_differentiable_prediction`) plus
  :class:`hod_mod.observables.cross_spectra.HaloModelCrossSpectra` give the real
  production amplitudes.  Differentiable observables, each validated against
  central finite differences:

  ============================  =========================================  ===========
  Observable                    Path                                       jacfwd vs FD
  ============================  =========================================  ===========
  ``wp`` / ``ΔΣ``               ``FullHaloModelPrediction``                ~1e-7
  tSZ ``cl_gy(ℓ)``              ``HaloModelCrossSpectra``                  ~3e-8
  X-ray ``cl_gX`` (density)     ``GasDensityDPM``                          ~4e-8
  X-ray ``cl_gX`` (full-APEC)   ``GasDensityDPM`` + ``ApecCoolingTable``   ~7e-6
  galaxy × AGN X-ray            ``XrayAGNModel``                           ~1e-7
  cluster × galaxy ``w_p^{cg}`` ``ClusterGalaxyCrossCorrelation``          ~7e-7
  ============================  =========================================  ===========

  CAMB stays the default backend; ``eh98_jax`` is opt-in and reproduces CAMB
  clustering to ~2 %.

Inference
---------

:mod:`hod_mod.fitting.jax_inference` wraps either backend into a Gaussian
log-posterior and drives it with gradients:

.. code-block:: python

   import jax
   from hod_mod.forecast.forward_jax import ForwardModel
   from hod_mod.fitting.jax_inference import (
       MultiProbeGaussianLikelihood, run_map_jax, run_nuts)

   fm = ForwardModel(z_eff=0.2)
   which = ["wp", "ds", "cl_gy", "cl_gX", "cl_kk", "xlf"]   # galaxies+SZ+X-ray+shear+AGN
   free  = ["Omega_m", "sigma8", "lg_m1h", "lg_m0star"]

   like, x_true = MultiProbeGaussianLikelihood.synthetic(fm, which, free, rel_err=0.05)
   res = run_map_jax(like, x0)                 # scipy L-BFGS-B with the JAX gradient
   post = run_nuts(like, res["x"])             # blackjax NUTS (optional dependency)

For the production backend, assemble the real observables with
:class:`~hod_mod.fitting.jax_inference.ProductionMultiProbeModel` and use
:meth:`MultiProbeGaussianLikelihood.synthetic_production` /
``from_production``:

.. code-block:: python

   from hod_mod.observables import make_differentiable_prediction
   from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
   from hod_mod.gas import PressureProfileA10, GasDensityDPM
   from hod_mod.fitting.jax_inference import ProductionMultiProbeModel

   pred = make_differentiable_prediction("more15")
   cross = HaloModelCrossSpectra(pred, pressure_profile=PressureProfileA10(),
                                 density_profile=GasDensityDPM(model=2))
   prod = ProductionMultiProbeModel(
       pred, cross=cross, z=0.2, rp_wp=..., rp_ds=..., ell=...,
       z_grid=z_grid, nz_g=nz, base_cosmo=cosmo, base_hod=hod,
       cosmo_params=["Omega_m", "sigma8"], hod_free=["log10mmin", "alpha"])

Practical notes
---------------

* **Run gradient work under** ``JAX_ENABLE_X64=1`` — float32 finite differences on
  the observables are noise.
* **Cosmology-dict convention.**  The differentiable backends are parameterised by
  **σ8** (keys ``Omega_m, Omega_b, h, n_s, sigma8``; optional ``sum_mnu, w0, wa``),
  *not* the ``ln10^{10}A_s`` used by the CAMB path.
* **NUTS cost.**  The Limber angular spectra (``cl_gy``/``cl_gX``/``cl_kk``) inflate
  the NUTS trajectory compile ~10×; MAP handles the full vector cheaply, but for
  NUTS prefer the projected/abundance probes (``wp``/``ds``/``xlf``/``smf``) or a
  small redshift grid.  The production 4-probe gradient compiles once (~6 min) then
  evaluates in ~1 s.
* **v1 scope of the production backend.**  Supports NFW/Einasto profiles, the baryon
  split, assembly bias, off-centering and satellite cutoffs; rejects BNL and the
  CAMB non-linear 2-halo term.
