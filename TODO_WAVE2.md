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

- **Phase 4 — X-ray (cl_gX) + AGN, production fidelity — DONE.** Ported
  `GasDensityDPM._gnfw_f`/`_concentration`/`density_uk`/`emissivity_uk`/
  `emissivity_full_uk` to accept **traced** inputs; `ApecCoolingTable.__call__` is
  now a pure-jnp 2D log-log bilinear interp on the soxs-built (T,Z) grid (matches the
  former scipy RGI to 8e-15 on the interior); `temperature_from_profiles` → jnp;
  `XrayAGNModel.agn_emissivity_uk` returns jnp and the AGN conversion keeps h traced;
  `_density_uk_cached` bypasses the concrete cache under trace. Result — all X-ray
  legs jacfwd-able w.r.t. cosmology: **density-only `cl_gX`** (4e-8 vs FD),
  **full-APEC `cl_gX`** with T(r)/Z(r) cooling (7e-6), **galaxy×AGN cross** (1.4e-7).
  CAMB gas/dpm/agn regressions green. Commits `89a3d41`, `a8cffdb`, `b5ef892`.

### Wave 5 — production-fidelity inference wiring — DONE

- **`ProductionMultiProbeModel`** (`fitting/jax_inference.py`) assembles the
  differentiable production observables (`FullHaloModelPrediction.wp/delta_sigma`
  + `HaloModelCrossSpectra.angular_cl_gy/gX`) into one `value_and_grad`-able data
  vector; `MultiProbeGaussianLikelihood` generalized with `param_names` +
  `synthetic_production`. Production wp+ds MAP recovers injected params; full
  4-probe gradient compiles + is finite (steady 0.8 s/eval). Fixed three jit-only
  concretizations (c_diemer15 dynamic omega_m; _neff grid spacing; angular log_k
  jnp). Commit `8b810b2`.
- **cluster × galaxy `w_p^{cg}`** differentiable (`_get_halo_tables` swap). `99cc486`.
- **Real n(z) hook** `ForwardModel(galaxy_nz=(z,nz))`. `89b5fdc`.
- **Docs** `docs/differentiable_inference.rst`. `d9b6101`.

### Remaining (small follow-ons)

- `HODAgnModel.nc_ns_agn`/`agn_emissivity_uk` jnp (HOD-AGN, not parametric, cross);
  `cl_XX` auto-spectrum production-differentiable; `x_uk_override` emulator path.
- NUTS over the full angular-spectra vector (compile cost; MAP is fine).
- Surrogate-vs-production amplitude validation on cl_gy/cl_gX.
- Coverage floor re-baseline after the Wave-3/4/5 additions.

## Guardrails / oracles to keep green

`tests/test_eh98_backend.py` (incl. `TestEh98LiftedFeatures`, `TestEh98CrossSpectra`),
`tests/test_jax_inference.py`, `tests/test_wave1_regression.py`,
`TestSatelliteNfwUkNumpyOracle`, `TestHodWeightsNumpyOracle`,
`test_m500c_gradient_vs_fd`. Float32 np-vs-jnp floor: rtol≈2e-5 smooth, atol≈1e-4
oscillatory FTs. Gradient work under `JAX_ENABLE_X64=1` (`x64` marker; env var only).
