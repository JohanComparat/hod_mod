# Wave 2 TODO — full JAX unification of the production stack

What remains after the 2026-07 "tests + JAX coverage" wave
(branch `feature/tests-and-jax-coverage`).  Wave 1 converted every numpy
island in the per-call hot paths to jnp and regression-pinned them against
float64 numpy oracles; the codebase is now Wave-2-ready.

## The headline deliverable

Opt-in `pk_backend="eh98_jax"` on
`observables/clustering.py::FullHaloModelPrediction`, making production
wp/ΔΣ jittable and `jax.jacfwd`/`grad`-able end-to-end w.r.t. HOD +
cosmology parameters (CAMB stays the accurate default).

Design already grounded in the code (see the plan's deferred section):

- Add a parallel `_pk_tables_full_jax(z, theta_cosmo, hod_params)` returning
  the same log-table dict as `_pk_tables_full` (clustering.py:984-996 keys).
  Dispatch at the table-builder boundary — every public observable consumes
  only the log-table dict, and the downstream Hankel/π/χ integrations are
  already pure JAX.
- Build from: `forecast/pk_eisenstein_hu.EisensteinHu98PkLinear` (traceable;
  z static via `float(z)` in `growth_factor` — acceptable, gradients are
  w.r.t. cosmology + HOD), the already-JAX HMF
  (`core/halo_mass_function.py` — `dndm`/`bias` traceable with an EH98
  `pk_func`), `nfw_uk_jax` (halo_profiles.py), the module-level
  `mdef_delta_rho` (now traced-friendly, no float() casts), and traced
  `hod_params` pytrees (the jitted occupation kernels already accept traced
  values — exactly how `forward_jax` uses them).
- No `_static_cache`, no `disable_jit` in that mode — jit compilation
  caching replaces the dict cache.
- v1 validation constraints: require `ConcentrationModel` with a traceable
  c(M) (`duffy08`/`dutton14`/`klypin16` — klypin16 z-interp is now jnp);
  `profile == "nfw"`; exclude `f_cut`/`gamma_inner` (satellite_nfw_uk is
  jnp but its `if float(f_cut) > 0` branches are concrete), baryon split,
  BNL, nl_2halo, einasto — raise ValueError on static key presence.
- Convenience factory (`make_differentiable_prediction`) wiring
  EH98 + tinker08-JAX + ConcentrationModel; document the `sigma8` vs
  `ln10^10 A_s` parameter-dict split and the `camb_ratio` accuracy hook
  (`forecast/pk_camb_ratio.py` — linearized CAMB/EH98 ratio, already
  tabulated and tested).
- Tests: mirror `hod_mod/forecast/tests/test_forward_matches_production.py`
  tolerances for eh98-vs-CAMB accuracy; jit-vs-eager self-consistency;
  `jacfwd` of wp w.r.t. (sigma8, Omega_m, log10_m_min, alpha_sat) vs central
  finite differences under the `x64` marker (CI step exists).

## Follow-on items (smaller)

- **Gradient-based fitter**: once the eh98 backend exists, add a
  jax-gradient MAP path (scipy L-BFGS with `jax.grad`, or optax) next to the
  Powell/Nelder-Mead `fitting/fitters.py::run_map`. Deferred deliverable:
  a docs snippet + smoke test proving `jax.jacfwd` through `wp` works.
- **`gas/density.py` integrands → jnp**: `_profile_uk_gl` now contracts in
  jnp but `GasDensityDPM`'s integrand callbacks still compute n_e(r) in
  numpy (kept compatible on purpose — the callback receives numpy
  `r_nodes`). Porting them + `_profile_uk_gl_bands` (full-APEC band FTs)
  makes the whole X-ray emissivity FT jnp. The APEC Λ(T,Z) lookup is a
  scipy RegularGridInterpolator (init-time table) — swap for
  `jax.scipy.interpolate.RegularGridInterpolator` if the emissivity path
  should ever be differentiated (the tier-2 forecast already has its own
  distilled JAX band tables in `forecast/apec_bands.py`).
- **`forward_jax` safe_log adoption**: `core/numerics.safe_log` is adopted
  in cross_spectra + clustering; forward_jax still has ~10 inline
  `jnp.log(jnp.maximum(·, 1e-30))` sites — behavior-identical, adopt during
  the next forward_jax touch rather than as standalone churn.
- **`HODClusteringPrediction`** (2-halo-only, clustering.py:176): same
  backend treatment is nearly free once `_pk_tables_full_jax` exists.
- **w_theta under jit**: geometry setup uses concrete maxima for grid
  construction — leave eager-only in v1, revisit with static grid bounds.
- **x64 policy**: library still never sets `jax_enable_x64` (deliberate);
  gradient-based fitting should run under `JAX_ENABLE_X64=1` (env var, read
  at jax import). Documented in the x64 pytest marker + CI step.

## Wave-1 outcomes to remember while doing Wave 2

- Everything per-call in `_pk_tables_full` is jnp float32 now; the
  per-cosmology static cache stays float64 numpy. Regression oracles:
  `tests/test_wave1_regression.py` (clustering),
  `TestSatelliteNfwUkNumpyOracle` (test_cosmology.py),
  `TestPcgNumpyOracle` (test_cross_clustering.py),
  `TestHodWeightsNumpyOracle` (test_cross_spectra.py) — keep them green.
- `gas/conversions.py::m200_to_m500c` is differentiable end-to-end now
  (Newton polish after the bisection; `test_m500c_gradient_vs_fd`). Use the
  same trick (or `jax.lax.custom_root`) for any future fixed-iteration
  solver.
- Float32 tolerance floor for np-vs-jnp comparisons: rtol ≈ 2e-5 on
  smooth integrals, absolute ≈ 1e-4 on oscillatory FTs (sin(K) at K~300).
