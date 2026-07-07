# Wave 2 — status and remaining work

Wave 2's headline deliverable is **done** (branch `feature/tests-and-jax-coverage`,
2026-07): an opt-in `pk_backend="eh98_jax"` on
`observables/clustering.py::FullHaloModelPrediction` makes production
`wp`/`ΔΣ`/`ξ` `jax.jit`- and `jacfwd`/`grad`-able end-to-end w.r.t. HOD **and**
cosmology parameters, CAMB staying the default.

## Done

- `_pk_tables_full_jax` — traceable twin of `_pk_tables_full`, built entirely
  in jnp (EH98 linear P(k), analytic JAX HMF, `nfw_uk_jax`, traced
  `hod_params`; no static cache, no `disable_jit`). Dispatched at the
  table-builder boundary; every public observable works unmodified.
- `make_differentiable_prediction(...)` factory wiring
  `EisensteinHu98PkLinear` + `make_hmf("tinker08", pk_func=...)` +
  `ConcentrationModel`. Cosmology dict uses **sigma8** (not `ln10^{10}A_s`).
- `ConcentrationModel._mdef_delta_rho` — makes a ConcentrationModel drop-in for
  HaloProfile in the halo-model radius, on **both** backends (it previously
  crashed the CAMB path).
- Validated: `tests/test_eh98_backend.py` — observables physical + jit-stable;
  eh98-vs-CAMB agree to **1.7% wp / 2.3% ΔΣ** with matched c(M) (test asserts
  <5%/<6%); `jacfwd` vs central finite differences to ~1e-7 for
  (sigma8, Omega_m, log10mmin, alpha) under the `x64` marker; guard rails.
- v1 scope: standard 1h+2h assembly plus smooth `A_cen/A_sat`, off-centering
  (`p_off` + `R_off`|`sigma_off`), `b_sat_conc`; c(M) in
  duffy08/dutton14/klypin16; profile nfw. Rejects baryon split, BNL,
  non-linear 2-halo, einasto, and `f_cut`/`gamma_inner` satellite cutoffs.

## Remaining (follow-on, smaller)

- **Gradient-based fitter** — the enabling deliverable now exists, so add a
  `jax.grad` MAP path (scipy L-BFGS-B with the analytic gradient, or optax)
  beside Powell/Nelder-Mead in `fitting/fitters.py::run_map`. Smallest useful
  unit: a smoke test + docs snippet fitting `wp`/`ΔΣ` through `jax.jacfwd`.
  Run under `JAX_ENABLE_X64=1`.
- **Lift v1 restrictions** (as needed): traceable satellite cutoffs (make
  `satellite_nfw_uk`'s `f_cut`/`gamma_inner` branches `jnp.where`-based so they
  can join the eh98 path); baryon split (`baryon_fraction` is already jnp) and
  BNL under the traced backend; einasto FT (`einasto_uk` is GL-quadrature, jnp).
- **`w_theta` under jit** — its geometry setup uses concrete maxima for grid
  construction; revisit with static grid bounds if the angular observable
  needs to be differentiable (v1 targets wp/ΔΣ/ξ only).
- **`HODClusteringPrediction`** (2-halo-only) — same backend treatment is nearly
  free now that the pattern exists.
- **`gas/density.py` integrands → jnp** — `_profile_uk_gl` contracts in jnp but
  `GasDensityDPM`'s integrand callbacks still compute n_e(r) in numpy (kept
  compatible on purpose — the callback receives numpy `r_nodes`). Porting them +
  `_profile_uk_gl_bands` makes the X-ray emissivity FT jnp; the APEC Λ(T,Z)
  lookup is a scipy `RegularGridInterpolator` (init-time table) — swap for
  `jax.scipy.interpolate.RegularGridInterpolator` only if that path must be
  differentiated (the tier-2 forecast already has JAX band tables in
  `forecast/apec_bands.py`).
- **`forward_jax` safe_log adoption** — `core/numerics.safe_log` is adopted in
  cross_spectra + clustering; forward_jax still has ~10 inline
  `jnp.log(jnp.maximum(·, 1e-30))` sites (behavior-identical) — adopt on the
  next forward_jax touch, not as standalone churn.

## Guardrails / oracles to keep green

Wave-1 float64 regression oracles: `tests/test_wave1_regression.py`,
`TestSatelliteNfwUkNumpyOracle` (test_cosmology.py), `TestPcgNumpyOracle`
(test_cross_clustering.py), `TestHodWeightsNumpyOracle` (test_cross_spectra.py),
`test_m500c_gradient_vs_fd` (test_jax_conversions.py). Wave-2:
`tests/test_eh98_backend.py`. Float32 tolerance floor for np-vs-jnp: rtol≈2e-5
smooth integrals, atol≈1e-4 oscillatory FTs. Gradient work runs under
`JAX_ENABLE_X64=1` (env var only; never toggle mid-session).
