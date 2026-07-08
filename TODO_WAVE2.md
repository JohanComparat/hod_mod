# JAX differentiability — Wave 2 (done) + Wave 3 (in progress)

Branch `feature/tests-and-jax-coverage`.

## Wave 2 — DONE

Opt-in `pk_backend="eh98_jax"` on `FullHaloModelPrediction` makes production
`wp`/`ΔΣ`/`ξ` jit/jacfwd/grad-able end-to-end (CAMB stays default).
`make_differentiable_prediction(...)` factory; `ConcentrationModel._mdef_delta_rho`.
Validated `tests/test_eh98_backend.py` (jacfwd==FD ~1e-7; eh98-vs-CAMB ~2%).

## Wave 3 — differentiable multi-probe inference + production cross-spectra port

### Done

- **Phase 1 — differentiable multi-probe inference** (`fitting/jax_inference.py`).
  `MultiProbeGaussianLikelihood` wraps the forecast `ForwardModel.data_vector_fn`
  as a `jax.value_and_grad`-able Gaussian log-posterior over a chosen subset of
  the 111 forecast params, spanning galaxy clustering + g-g lensing + tSZ + X-ray
  + cosmic shear + AGN in one call. `run_map_jax` (scipy L-BFGS-B + JAX gradient)
  and `run_nuts` (blackjax NUTS). Validated: AD grad == FD ~1e-6; MAP recovers
  injected 6-probe params at χ²/dof~0.9; NUTS recovers the posterior.
  NUTS caveat: the Limber angular spectra inflate the trajectory compile ~10×, so
  NUTS is practical on projected+abundance probes (wp/ds/xlf/smf) — MAP handles the
  full angular vector fine. `blackjax` added as optional `inference` extra.
- **Phase 2 — lifted eh98_jax restrictions.** einasto profile, baryon split
  (CDM+gas), and traceable satellite cutoffs (`satellite_nfw_uk` f_cut/gamma_inner
  now `jnp.where`, Wave-1 oracle still green). Only BNL + CAMB nl-2halo remain
  rejected. jacfwd==FD for each.
- **Phase 3 — differentiable production tSZ cross-power.** Enabling change:
  `FullHaloModelPrediction._get_halo_tables` exposes the halo tables (m, dndm,
  bias, uk, pk_lin, c, r_delta, rho_m, k) as numpy on CAMB / **traced** jnp on
  eh98. `HaloModelCrossSpectra._get_static_cache`/`_get_hod_weights` backend-aware;
  `_pressure_uk_cached` bypasses the concrete cache when traced. Ported
  `PressureProfileA10.pressure_uk`, `gas/conversions._profile_uk_gl`,
  `m200_to_m500c`, and the `angular_cl_gy/gX` Limber χ(z) to accept traced inputs.
  Result: **`cl_gy(ℓ)` (tSZ) is jacfwd-able w.r.t. cosmology** (rel 3e-8 vs FD);
  CAMB cross-spectra regressions green.

### Remaining

- **Phase 4 — X-ray (cl_gX/cl_XX) + AGN, production fidelity.** The tSZ path is
  done; the X-ray path needs: (a) `GasDensityDPM.emissivity_uk`/`density_uk`
  integrand callbacks (`_ne_grid` etc., `gas/density.py`) to accept **traced**
  `r_nodes` (they currently `np.asarray` them — fine for CAMB-concrete, breaks
  under trace); (b) the APEC cooling `ApecCoolingTable.__call__` scipy
  `RegularGridInterpolator` → `jax.scipy.interpolate.RegularGridInterpolator` (or
  reuse the JAX band tables in `forecast/apec_bands.py`) + `temperature_from_profiles`
  → jnp, to carry gradients through T(r),Z(r); (c) `XrayAGNModel.agn_emissivity_uk`
  and `HODAgnModel.nc_ns_agn`/`agn_emissivity_uk` to return jnp (drop the
  `np.asarray`/`np.power`/`np.ones` boundary casts; the Lx/occupation kernels are
  already jnp); (d) `_density_uk_cached` cache bypass under trace (like the pressure
  one). The `_pk_tables_gX/XX` assembly + Limber are already jnp, so they trace once
  fed traced inputs. Then wire the production cross-probes into
  `MultiProbeGaussianLikelihood` as a production-fidelity backend option and demo a
  joint galaxies+ΔΣ+cl_gy+cl_gX+shear+AGN MAP + NUTS. `cross_clustering._pk_table_cg`
  gets the same `_get_halo_tables` swap (nearly free — assembly already jnp).
- **Phase 5 — n(z) hook, docs, validation, close.** Real galaxy n(z) injection into
  `ForwardModel` (currently a synthetic narrow Gaussian, forward_jax.py:509-514);
  surrogate-vs-production amplitude validation on cl_gy/cl_gX;
  `docs/differentiable_inference.rst`; full `--cov` re-baseline + raise floor.

## Guardrails / oracles to keep green

`tests/test_eh98_backend.py` (incl. `TestEh98LiftedFeatures`, `TestEh98CrossSpectra`),
`tests/test_jax_inference.py`, `tests/test_wave1_regression.py`,
`TestSatelliteNfwUkNumpyOracle`, `TestHodWeightsNumpyOracle`,
`test_m500c_gradient_vs_fd`. Float32 np-vs-jnp floor: rtol≈2e-5 smooth, atol≈1e-4
oscillatory FTs. Gradient work under `JAX_ENABLE_X64=1` (`x64` marker; env var only).
