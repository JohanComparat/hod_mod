# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- **GODMAX independent SZ cross-check.** Added an independent-code validation of
  the thermal-SZ machinery against [GODMAX](https://github.com/shivampcosmo/GODMAX)
  ([Pandey2024], arXiv:2401.18072), driving both codes through the *shared*
  Battaglia+2012 electron pressure profile so the comparison isolates hod_mod's
  projection stack from the gas-physics model.
  - New `hod_mod.gas.PressureProfileBattaglia12` — the AGN-feedback GNFW pressure
    ([Battaglia2012], arXiv:1109.3711, Δ=200c), mirroring GODMAX's `Battaglia_12_16`
    (Kaiser `P_200c` amplitude, mass/z-scaled `{P0, xc, β}`, electron factor
    `f_e=(2+2X_H)/(3+5X_H)`). Reproduces an independent-quadrature reference for
    `P_e(r|M,z)` and `ỹ(k|M)` to ~1e-6.
  - New tSZ projectors in `HaloModelCrossSpectra`: `angular_cl_yy` (tSZ auto,
    via `_pk_tables_yy`) and `angular_cl_ky` (shear × tSZ, reusing `P_my` with a
    new tomographic convergence kernel `_convergence_kernel`).
  - `scripts/godmax/export_godmax_b12_reference.py` freezes the reference
    (`--source godmax` in a GODMAX env, or a self-contained `--source independent`
    NumPy build); `hod_mod/scripts/validate_godmax.py` overlays + reports residuals;
    `tests/test_godmax_comparison.py` adds self-contained machinery checks plus
    skipif reference assertions. New docs page `docs/godmax_cross_check.rst`.

## [0.3.0] — 2026-07-15

**Behaviour-changing release — minor bump, not a patch.** This corrects the Hankel transform
underlying every real-space observable, so every number the code produces moves: `w_p` by up to
19%, `ΔΣ` by up to 20%, `Σ_y` by ~16%. Results from 0.2.3 and earlier are superseded and any fit
built on them should be re-run. Pinning `~=0.2.3` will *not* pick this up, which is deliberate:
a silent 20% shift in a patch release is exactly what a minor bump exists to prevent.

- **Fixed `_pk_to_xi` (`observables.clustering`)**, which was not the Ogata double-exponential
  quadrature it claimed to be. Two coupled bugs that masked each other:
  1. the nodes were a factor `1/h = 200` too small — `x = π(hn)tanh(...)` instead of the
     `x = πn tanh(...)` documented in the file's own header comment — with a compensating `h`
     prefactor hiding it in the normalisation. The result was a plain trapezoid truncated at
     `k·r ≈ π h N = 8.04`, so the double-exponential decay of the weights (the entire point of
     the scheme) never happened;
  2. `P(k)` was extrapolated as a *constant* above the table's `k_max`. Bug 1's truncation was
     limiting the damage from bug 2, so fixing either alone was worse than fixing neither.

  Now: correct nodes (reaching `k·r ≈ 1608`) plus a power-law high-k continuation, guarded
  against a degenerate or `-inf` tail.

- **Validation.** The corrected transform agrees with an *exact analytic* transform pair to
  <1e-4 (the old one returned the **wrong sign** where `ξ ≈ 3e-4`), with `mcfit`'s and
  `hankl`'s independent FFTLog implementations to 0.12%/0.26% on a real `P_gm` and 0.04%/0.26%
  on a real `P_gy`, and — end-to-end against **AUM**, an independent C++ halo model — improves
  `w_p` from 16.1% to **2.8%** median disagreement and `ΔΣ` from 15.1% to **4.2%**. The residual
  is the known CAMB-vs-Eisenstein-Hu `P_lin` and Diemer19-vs-AUM `c(M)` difference, not numerics.

- **Tests.** `tests/test_aum_comparison.py` tolerances tightened from 60%/130% to **15%/15%** —
  at 60% the broken transform sat 21% from AUM and the test passed. Added an exact analytic
  transform pair and a node-reach guard. Recorded that `TestOgataHankel`'s `k^-2` power-law
  tests are structurally blind to node/extrapolation bugs and must not be read as coverage.

- **tSZ**: `projected_gy` gains `beam_fwhm_arcmin` (Gaussian beam via the existing
  `psf_window_ell`, applied as `B(|k|χ)` before the Abel projection — exact by projection-slice),
  plus `projected_gy_nz` for an n(z)-weighted beam, `sigma_y_theta`, and a reusable `cap_filter`
  for compensated aperture photometry. `SumStatReader` gains `sz()` for `sum_stat`'s `sz/` group.
  Fixed an unguarded `rp_centres_wp`/`rp_centres_esd` lookup that made any subset-of-statistics
  joint file fail to open.

- **Version**: `__init__.__version__` had been stale at 0.2.1 since 0.2.2/0.2.3 shipped; it now
  tracks `pyproject.toml` again.

## [0.2.3] — 2026-07-08

Maintenance release — first PyPI upload carrying the 0.2.2 differentiable
multi-probe inference + JAX lensing work (0.2.2 was tagged but never published to
PyPI).

- **CI**: fix a duty-cycle-AGN `FileNotFoundError` in the test suite and stop the
  test matrix's `fail-fast` from masking failures on other Python versions.
- **Docs**: remove the stale `REPORT_strategy.md` and `TODO_WAVE2.md` planning
  notes.

## [0.2.2] — 2026-07-08

The **differentiable multi-probe inference** track and **pure-JAX weak + strong
lensing**: the production forward model is now differentiable end-to-end, and the
whole numerical core has been ported off per-call numpy islands onto JAX.

- **Differentiable multi-probe inference** (`hod_mod.fitting.jax_inference`):
  gradient-based MAP (`run_map_jax`, scipy L-BFGS-B driven by the JAX gradient)
  and blackjax NUTS (`run_nuts`) over a `MultiProbeGaussianLikelihood`, with
  `ProductionMultiProbeModel` assembling the real observables. Two backends —
  the σ8-native forecast surrogate `forecast.forward_jax.ForwardModel` (every
  probe in one `jacfwd` call) and the full-fidelity production path
  `FullHaloModelPrediction(pk_backend="eh98_jax")` built via
  `observables.make_differentiable_prediction`; CAMB remains the default and the
  `eh98_jax` clustering reproduces it to ~2%.
- **Differentiable observables** on the `eh98_jax` backend, each validated
  against central finite differences (`jacfwd` vs FD ≲ 1e-6): tSZ `cl_gy(ℓ)`,
  X-ray `cl_gX` (density-only **and** full-APEC temperature-dependent
  emissivity), galaxy × AGN X-ray cross, and cluster × galaxy `w_p^{cg}`. Lifted
  the eh98_jax v1 restrictions (Einasto, baryon split, off-centering, traceable
  satellite cutoffs). Real galaxy `n(z)` injection hook
  (`ForwardModel(galaxy_nz=...)`) for fitting measured angular spectra.
- **Pure-JAX weak + strong lensing** (`hod_mod.observables.lensing`,
  `hod_mod.core.lensing_profiles`): sharply truncated NFW (Takada & Jain 2003),
  BMO and Hernquist profiles with no colossus/astropy/fftlog dependency, and
  `ClusterLensingPrediction` (`kappa`, `gamma_t`/ΔΣ with mis-centering + a
  Tinker10 2-halo term, `einstein_radius`, magnification, critical curves). The
  Einstein-radius solver's `jax.grad` is the exact implicit-function-theorem
  derivative. Ports the `halo_lensing` reference (Oguri et al. 2026); reproduces
  its fftlog ΔΣ to ~1% max / 0.04% median.
- **JAX-coverage waves 1–5**: all per-call numpy islands in the core, gas and
  cross-spectra assembly moved to `jnp` with pinned float64 numpy oracles;
  AD-gradient fixes through the `m200_to_m500c` / m500c bisections (Newton
  polish); shared float32-safe `core.numerics.safe_log`. Sensitivity study
  parameter-freedom robustness (31 vs 111 params). Coverage floor raised
  84 → 85 (full suite 87.6%).
- **New docs pages**: `differentiable_inference.rst`, `lensing.rst`,
  `bgs_zm15_joint_mcmc.rst`.

## [0.2.1] — 2026-07-04

The **missing-physics extension** (waves 1-4) and the **tier-3/tier-4
multi-wavelength + morphology forecasts**: the differentiable parameter
vector grows 61 -> 111, every addition fiducial-preserving and gated by an
exact invariant; three production Fisher runs (90, 102 and 111 parameters)
with physical survey noise.  Headline: sigma(Omega_m) tightens from
2.85e-4 (tier-2, 61 params) to 4.89e-5 (tier-4, 111 params) while freeing
50 additional parameters -- the multi-wavelength data self-calibrate the
astrophysics, and the intrinsic-alignment amplitude self-calibrates
through the morphology sector at no cost to the shear cosmology.

First implementation wave of the `docs/missing_physics.rst` roadmap
(parameter vector 61 → 77, all fiducial-preserving; gates in
`tests/test_missing_physics.py`):

- **Cosmology-dependent concentration**: `ForwardModel(cm_relation="diemer15")`
  uses Diemer & Kravtsov c(ν, n_eff) with Diemer & Joyce 2019 median
  parameters — differentiable through σ(M, z) and the EH98 slope, so c(M)
  finally responds to σ8/n_s/h and the beyond-ΛCDM sector.  Fixed a factor-~2
  bug in the pre-existing `core.concentration.c_diemer15` (DK15 Eq. 9 form)
  and the κ = 0.42 n_eff convention; values now match COLOSSUS anchors.
- **Beyond-ΛCDM**: `w0`, `wa`, `sum_mnu` in the vector; CPL growth ODE
  (`growth_factor` dispatcher, RK4 via `lax.scan`, <0.2% vs exact ΛCDM);
  w0/wa threaded through all forecast distances and E(z); first-order
  massive-ν suppression of the EH98 shape (σ8-anchored, exactly massless at
  the 0 eV fiducial).
- **SF/quiescent split**: `ForwardModel(sfq="sf"|"q")` — ZM16 Weibull
  quenching on the occupations (SF + Q ≡ unsplit, exact) + the
  `dlx_quenched` hot-gas offset targeted at the eROSITA CGM SF-vs-Q data.
- **AGN fundamental plane**: `rlf` observable (5 GHz radio LF) from
  (ξ_RX, ξ_RM, b_R, σ_R) on the Powell chain; identity-collapses onto the
  hard-band XLF at (1, 0, 0, 0).
- **HI sector**: VN18 M_HI(M_h) halo model, `himf` and `cl_gHI` observables
  reusing the existing kernels; C_ℓ^{gHI} ∝ M_0 exactly.
- `eps_sn` promoted (energy-closure SN coupling now free).

Wave 2 (vector 77 → 83):

- **CAMB-quality P(k)**: `forecast/pk_camb_ratio.py` — the linearized
  CAMB/EH98 shape-ratio table (fiducial + derivative rows for
  h/Ω_b/Ω_m/n_s/Σm_ν; |lnR₀|max = 3.8%), distilled once into
  `data/pk_ratio/camb_eh98_ratio.npz` and applied differentiably via
  `ForwardModel(pk_correction="camb_linear")` — spectrum and first
  derivatives CAMB-accurate near the fiducial (the Fisher requirement).
- **SN wind mass loading**: `eta_w_norm`/`alpha_w` (Muratov+15 form) coupled
  into the η(M) gas-concentration slot; exactly the tier-2 sigmoid at the
  η₀ = 0 fiducial.
- **Star-forming main sequence**: per-cell `ssfr` observable
  (`ssfr_ms_norm`/`ssfr_ms_slope` + `ssfr_ms_zs` through the standard
  evolution mechanism); **quenched-HI deficit** `dhi_quenched`.
- **Radio/HI enter the forecast**: `RadioSurvey` (LoTSS-like, νL_ν(5 GHz)
  completeness limit) and `HISurvey` (ALFALFA flux-limited HIMF + effective
  21 cm-IM noise) in `forecast/noise.py`;
  `Tier2Forecast(include_radio, include_hi, include_ssfr)` adds the radio LF
  per shell, a dedicated LOCAL (z ≤ 0.06) HIMF block — at Δz = 0.1 shell
  depths the 21 cm flux limit flags every bin, so the HIMF is local by
  design — the 21 cm × galaxy cross per cell and the sSFR datum per
  non-quenched cell; driver flags
  `--include-radio --include-hi --include-ssfr --split-sfq`.

Wave 3 (vector 83 → 90):

- **Continuous sSFR distribution**: double-lognormal p(log sSFR | M*) with
  free MS scatter `sigma_ms` and quenched offset `dssfr_q`;
  **sSFR-threshold selection** `ForwardModel(ssfr_cut=...)` (ELG-like
  samples) composable with the SF/Q split — SF-cut + Q-cut ≡ mixture-cut
  exactly (tested).
- **SFR density** `sfrd` per cell (ρ_SFR ∝ 10^{sSFR_MS} exactly; SF + Q
  partition it exactly) and the **z-resolved [OII] LF** `oiilf` — the
  Kennicutt-like `loii_norm` calibration on the SHMR + main sequence, with
  the DESI-like line-flux completeness limit in the noise.
- **Radio-loud jets** (`f_loud0`, `beta_loud`, `b_jet`): a second rlf
  component from ALL central black holes (HERG/LERG, not ERDF-tied — the
  ferdf amplitude identity of the FP-only rlf breaks by design, tested).
- **AGN infrared LF** `ilf` (`agn_bc_ir` on L_bol) — obscuration-robust by
  construction (zero `agn_fabs` response, tested): the cross-band check of
  the obscured fraction the soft-X-ray XLF is dimmed by.  `IRSurvey`
  (WISE/SPHEREx-like) noise + completeness.
- Driver flags `--include-ir` and extended `--include-ssfr` (sfrd + [OII]
  per shell); probe groups `+IR AGN`, galaxy-grid `+sfrd/oiilf`.

Tier-3 forecast (vector 90 → 102; `docs/tier3_forecast.rst`; gates in
`tests/test_forecast_tier3.py`):

- **12 SED calibrations** (`TIER3_EXTENSION`, all feeding only new
  observables): radio–FIR `l14_sfr` + `alpha_syn`, IR `lir_sfr` +
  `bir_color`, stellar M/L `ml_nir`/`ml_opt` + `dopt_q`, UV
  `luv_norm` + `tau_uv_mslope`, Hα `lha_norm`, AGN bolometric corrections
  `agn_bc_uv`/`agn_bc_opt`.  Slices `MISSING_PHYSICS`/`TIER2_EXTENSION`
  frozen at [61:90]/[31:90].
- **Radio/IR intensity maps** (SKA-like 0.95/1.4/3 GHz; WISE/SPHEREx-like
  3.4/4.9/12 μm): per-halo central νL_ν fields (SF synchrotron + FP cores +
  jets; dust + stellar continuum + AGN torus) with exact band-scaling and
  chain-rule gates; observables `cl_gR`/`cl_gI` (cells), `cl_RR`/`cl_II`/
  `cl_aR`/`cl_aI`/`cl_ag` (shells) through the new generic
  `_pk_tracer_field` kernel (bit-identical `_pk_gX_of` delegation).
- **Galaxy band LFs + AGN UV/opt LFs** via one `_lf_lognormal` kernel:
  `uvlf`, `optlf` (SF/Q mixture, exact collapse at `dopt_q=0`), `nirlf`,
  `half` (exact `oiilf` clone) and type-1 `qlf_uv`/`qlf_opt`
  (= `ilf` kernel × (1−f_abs) — the cross-band obscuration system closed).
- **Extras**: tSZ auto `cl_yy`, 21 cm auto `cl_HIHI`, X-ray cluster counts
  `ncl` (free L_X–M relation, 0.25 dex selection scatter), AGN lensing
  `ds_agn`; per-shell wide-M* `sfrd` blocks (Madau–Dickinson).
- **`Tier3Forecast`** (`forecast/tier3.py`): coarse exploratory grid
  (Δz = 0.2 to z = 2 × 0.2 dex down to M* = 10⁹, SF/Q split), two-tier
  spectroscopic completeness (wide M*_lim(z) = 10^{9+z} + deep field;
  incomplete cells skipped and reported), extended mass grid
  (`ForwardModel(log10m_min=8.5)`; default 10.0 bit-identical), AGN samples
  to z = 1.9; new noise models `SKASurvey`/`IRMapSurvey`/`BandLFSurvey`
  + `SpectroSurvey` M*/Hα limits; tier-2 assembly reused through identity
  hooks (tier-2 σ regression unchanged).
- **`--jobs N`** parallel block precompute
  (`Tier2Forecast.precompute_blocks`): spawned workers rebuild the forecast
  from its resolved ctor spec and fill the cache atomically; parallel ==
  serial byte-exact (the x64 mode propagates via `JAX_ENABLE_X64` so
  module-level constants build in the right precision).
- Driver `run_tier3_forecast.py` (probe groups `+radio/IR maps`,
  `+band LFs`, `+tSZ/HI autos`, `+clusters`, `+AGN lensing`); tier-3 figure
  prefix in `make_tier2_figures`; docs page + 6 arXiv-verified references.

Wave 4 (vector 102 → 106) — galaxy morphology, the last roadmap topic:

- **Conditional early-type fraction** `connection/morphology.py` (ZM16
  Weibull pattern): `f_early_cen(M_h; log10_M_morph, beta_morph)` + the
  satellite boost `f_morph_sat`; `ForwardModel(morph="early"|"late")`
  splits any sample with EARLY + LATE ≡ unsplit exact, composable with the
  SF/Q split (4-way partition sums exactly — tested to 1e-12).
- **`f_early` observable**: one Euclid-VIS-like early-type-fraction datum
  per (z, M*) cell (`include_morph` / `--include-morph`, default ON in
  `Tier3Forecast`); binomial + `SpectroSurvey.fmorph_err = 0.02`
  calibration-floor noise.
- **BH–bulge coupling** `mbh_bt_slope` (Yang+2019-like): the mean
  early-type fraction proxies B/T inside the Powell chain's M_BH — exactly
  the pure chain at the 0 fiducial (zero XLF morphology response, tested),
  and off-fiducial it routes the morphology parameters into the
  XLF/radio/IR LFs.
- Slice freeze `TIER3_EXTENSION = PARAM_NAMES[90:102]`; new `morphology`
  sector; wave-4 gates in `tests/test_missing_physics.py`.

Tier-4 morphology observables (vector 106 → 111):

- **Joint E/Q fractions** (`rho_morph_q`): the 4-way early/late × SF/Q
  partition uses bounded-correlation joint fractions (exact independence at
  the 0 fiducial; unity partition at any rho, tested); split-cell `f_early`
  becomes conditional (f_E|Q vs f_E|SF) so the grid itself measures rho;
  per-cell `f_early_q` = the Galaxy Zoo red-spiral/blue-elliptical census.
- **Sizes** (`log10_f_size`, `dsize_early`, `f_size_zs`): per-cell
  <log10 R_e> through Kravtsov R_e = 0.015 R_200c (centrals, 0.2 dex
  scatter) — cosmology enters via R_200c ∝ (M/rho_crit)^(1/3); exact
  unit-response and _Z_EVOL chain-rule gates.
- **Intrinsic alignments** (`a_ia`): per-cell w_g+ in the NLA form with the
  amplitude carried by f_early (KiDS-1000: IA is driven by morphology),
  reusing the ΔΣ J2 transform verbatim; exact 1/a_ia and strict
  w_g+ ∝ f_early gates — the shear IA systematic becomes self-calibrated
  through the morphology sector.
- **AGN hosts**: per-shell `f_early_agn` (early fraction among L_X-selected
  hosts) — the direct mbh_bt_slope probe (Kocevski-style bulge dominance).
- **Morphology-split lensing/clustering**: early/late (wp, ΔΣ, n_gal)
  "morph_cell" blocks per (z, M*) cell to z = 1.2 (wide-tier cells only) —
  Mandelbaum-2006 at Euclid scale.
- `Tier4Forecast` + `run_tier4_forecast.py` (probe groups incl. a
  block-kind-selected morph-split group); `noise.wgp_noise`;
  `SpectroSurvey.size_err/fmorph_agn_err`; docs page with the 14
  arXiv-verified references; gates in tests/test_forecast_tier4.py.

## [0.2.0] — 2026-07-03

The **tier-2 sensitivity study**: all 61 parameters free (nothing fixed), a
(z, M*) grid of volume-limited galaxy samples, a multi-band APEC X-ray layer,
tomographic shear, and physical survey-noise models — answering how much
cosmology and how much astrophysics an optimistic Stage-IV data scenario
teaches when the model marginalises over everything it contains.

### Added

- **Parameter vector 31 → 61** (append-only; tier-1 scripts pin the extension
  by default and gain `--free-tier2`): the 16 formerly hard-coded nuisance
  shapes (satellite HOD, baryon-sector shape, gas emissivity slopes, the full
  A10 pressure shape including `p0_pressure`, Powell Model-2 `agn_rho` and
  `agn_sig_mstar`), 7 redshift-evolution slopes applied per
  `ln[(1+z)/(1+z_pivot)]` through `ForwardModel._theta_eff` (chain-rule exact),
  and 7 X-ray spectral parameters (temperature-profile tilt, ICM metallicity
  norm/mass-slope/evolution, AGN photon index `agn_gamma`, obscured fraction
  `agn_fabs`, `agn_mu_bh_zs`).  `log10DC` is retired: `agn_emission="powell"`
  replaces the tier-1 `L∝M` surrogate in `C_ℓ^{gX}`/`C_ℓ^{XX}` with the Powell
  chain (validated against `agn/powell.py` to <1%).
- **`forecast/tier2.py` — `Tier2Forecast`**: 80 volume-limited (z, M*) cells
  (Δz = 0.1 × 0.2 dex, exact threshold-difference bin occupations) + 10
  shell blocks (soft-band AGN XLF, per-band `C_ℓ^{XX}`) + a global tomographic
  lensing block + 5 AGN-clustering blocks (`wp_agn` per 0.5-dex soft-L_X bin),
  all sharing ONE parameter vector; block-wise Jacobians with per-block npz
  caching.
- **Multi-band APEC layer** (`forecast/apec_bands.py` + shipped
  `hod_mod/data/apec_bands/*.npz`): band-integrated Λ_b(T, Z) tables distilled
  once from `ApecCoolingTable`, evaluated with differentiable bilinear
  interpolation; exact Σ_b w_b = 1 amplitude partition; Morrison & McCammon
  ISM absorption templates for the obscured AGN fraction.
- **`forecast/noise.py`** — physical survey noise: pair counts + cosmic
  variance (w_p, `wp_agn`), shape noise (ΔΣ, 5-bin Euclid+LSST tomographic
  shear at 30 arcmin⁻², f_sky = 0.5), CXB photon noise with the
  completeness-pinned Athena all-sky spec (F_lim = 2×10⁻¹⁶ erg/s/cm² — exactly
  the depth making L_X > 10⁴² complete to z = 1), Poisson XLF errors with
  automated L_lim(z) completeness flags.
- **`run_tier2_forecast`** driver (`--smoke`, `--n-bands {1,6,15}`) with the
  cosmology-vs-astrophysics decomposition (sector pinning, bits per sector,
  probe build-up), a 9-figure suite (`make_tier2_figures`), and the
  `docs/tier2_forecast.rst` page with the production numbers.  Headlines at
  R_min = 0.1 Mpc/h: σ(Ω_m) = 2.9×10⁻⁴, σ(σ_8) = 4.4×10⁻⁴ — only ×2.2/×2.7
  above the astrophysics-pinned limit; the 6-band spectra turn
  Γ_AGN/f_abs/Z_gas from prior-bound into measured (×500 on Γ_AGN vs one
  broad band).
- **`docs/missing_physics.rst`** — "What the model does not yet contain":
  eight missing-physics sectors (beyond-ΛCDM, cosmology-correct differentiable
  ingredients, AGN radio/IR + fundamental plane, morphology, sSFR, SEDs,
  stellar feedback, cold gas/HI) with concrete implementation propositions,
  constraining measurements, and 55 new title/author-verified references.
- `fisher.constraints(..., scale=)` — prior-scaled eigen-inversion for the
  61-parameter conditioning; `ForwardModel` grew `log10m_star_bin`,
  `n_shear_bins`, `xray_bands`, `xlf_band`, `agn_emission`, `agn_lx_bins`
  keywords (all backward-compatible; tier-1 defaults bit-identical, tested).
- Tests: `tests/test_forecast_tier2.py`, `tests/test_forecast_noise.py`,
  `hod_mod/forecast/tests/test_agn_matches_powell.py` (94 green across the
  forecast suites), plus grid-convergence and monotonicity audits.

## [0.1.6] — 2026-07-02

The **differentiable sensitivity / Fisher forecast** of the full
ZM15 + X-ray gas + AGN pipeline, plus the physical AGN X-ray luminosity-function
model it uses, and a large test-coverage extension of the new code.

### Added

- **`hod_mod.forecast`** — a pure-JAX forward model whose whole chain (cosmology
  included, via the analytic Eisenstein & Hu 1998 transfer function instead of
  CAMB) is one differentiable function, so the Fisher Jacobian
  `∂d/∂θ` is a single `jax.jacfwd`:
  - `forward_jax.ForwardModel` — 31-parameter → 12-observable forward model
    (`w_p`, `ΔΣ`, `C_ℓ^{gX,gy,XX,κκ,κκ_c,gκ_c,κκ_c}`, `Φ(L_X)`, `n_gal`, `Φ(M_*)`),
    with a shared hot-gas/baryon sector linking the ΔΣ/cosmic-shear baryon split to
    the X-ray and tSZ amplitudes, and an optional energy-closure baryon mode.
  - `pk_eisenstein_hu.EisensteinHu98PkLinear` — σ8-parameterised, differentiable
    EH98 linear `P(k)`.
  - `fisher` — Fisher assembly (diagonal or full analytic covariance), eigen-floored
    constraints, figure of merit, degeneracy ranking, principal directions and
    per-probe decomposition; `covariance` — analytic Gaussian covariance with the
    lensing-triplet cross-correlations; `params` — fiducials/priors/labels;
    `tomography.TomographicForecast` — shared-cosmology, per-bin-HOD multi-sample
    Jacobian.
- **`hod_mod.agn.powell.PowellAGNModel`** — the physical Powell (2022) AGN–halo
  model forward-modelling the AGN X-ray luminosity function (ZM15 SHMR → free
  `M_BH`–`M_*` relation → universal Ananna 2022 Eddington-ratio distribution),
  validated standalone against the Aird+2015 hard XLF; added as the 7-parameter
  `xlf` observable of the forecast.
- **`hod_mod.scripts.forecasts`** — `run_sensitivity_study`, `make_sensitivity_figures`,
  `make_forward_diagram` (the structure-of-the-prediction diagram) and
  `run_stage4_forecast`; `scripts.fitting.fit_powell_agn`.
- **Documentation** — two new pages wired into the toctree: `sensitivity_fisher`
  (the pedagogical scale-cut/degeneracy-breaking study, with a new appendix that
  writes out the forward model *equation by equation* and a colour-coded flow
  diagram linking the 31 equations) and `stage4_forecast` (realistic multi-survey
  error budget).

### Tests

- 59 new tests for the forecast/AGN code: `test_forecast_fisher`,
  `test_forecast_params`, `test_forecast_covariance`, `test_forecast_pk_eh98`,
  `test_forecast_tomography` and `test_forecast_forward` (fast primitives + a
  tiny-grid `ForwardModel` smoke test asserting every observable is finite and the
  exact active-fraction identity `∂lnΦ/∂log10 f_ERDF = ln 10`), plus
  `test_powell_agn` (ERDF shape, XLF positivity/decline, convolution-vs-Monte-Carlo,
  occupation/bias/emissivity). These run in the default `tests/` suite (the heavy
  production-match / finite-difference validation stays in
  `hod_mod/forecast/tests/`).

## [0.1.5] — 2026-07-01

- ``docs/conf.py`` sets ``autodoc_mock_imports`` for the heavy/optional backends
  (``camb``, ``colossus``, ``AletheiaCosmo``, ``CEmulator``, ``aemulusnu_hmf``,
  ``soxs``) so a ReadTheDocs build still succeeds if one fails to import at
  build time. JAX is intentionally left real so ``jax.jit``-decorated functions
  keep their signatures in the API docs. Local build verified: 29 pages, no
  autodoc import failures.

## [0.1.4] — 2026-07-01

- ``docs/Makefile`` now invokes ``python -m sphinx`` (via a ``PYTHON`` variable)
  instead of the bare ``sphinx-build`` console script, so ``make html`` works
  whenever the active Python has Sphinx — no dependency on the console script
  being on ``$PATH``. Override with ``make html PYTHON=/path/to/python``.

## [0.1.3] — 2026-07-01

Packaging fix so the project can be published to PyPI.

- Removed the ``csst`` / ``aemulusnu`` / ``emulators`` optional-dependency extras:
  they referenced git-only packages via direct ``git+https`` URLs, which PyPI
  forbids in uploaded metadata (400 Bad Request). These emulator backends are
  documented as manual installs instead
  (``pip install git+https://github.com/czymh/csstemu`` etc.; see
  ``docs/cosmology.rst``). No runtime behaviour change.

## [0.1.2] — 2026-07-01

Documentation now mirrors the refactored repository; all links verified.

- ``docs/scripts.rst`` rewritten to the current ``scripts/`` layout — removed
  ~15 references to scripts deleted in the refactor (demos, ``run_pipeline``/
  ``run_inference``, ``utils/``, ``gama/``/``cosmos/`` fits).
- Replaced all remaining hardcoded paths in the docs with the ``$HOD_MOD_*``
  env-var forms; fixed ``paper_reproductions/more2015_boss_cmass.py`` to read the
  real ``configs/hod_fit_more2015_cmass.yml``.
- Verified **all links** with a Sphinx build: fixed a broken ``../_images`` figure
  ref, a missing ``bgs_ls10_wp_survey`` label, a toctree entry to an excluded page,
  5 dangling ``:ref:`` cross-references, and a dead external repo link; every other
  internal cross-reference, figure, and external URL resolves.

## [0.1.1] — 2026-06-30

Repository hygiene and reproducible paths. No public-API symbol changes.

### Data & results moved out of the repo
- Curated benchmark data + results are archived on **Zenodo**
  (concept DOI ``10.5281/zenodo.21078473``) and fetched on demand with checksum
  verification via ``hod_mod.data_io.fetch`` (``pooch``).
- Generated outputs now write **outside** the repo via
  ``hod_mod.paths.results_root`` (``$HOD_MOD_RESULTS``); ~30 scripts updated.
- ``results/`` purged from git history; ``.git`` shrank 309 MB → ~31 MB.

### No hardcoded paths
- All filesystem locations resolve through ``hod_mod.paths`` helpers with env-var
  overrides: ``repo_root()`` (``$HOD_MOD_REPO``), ``data_root()``
  (``$HOD_MOD_DATA_DIR``), ``sum_stat_root()`` (``$HOD_MOD_SUMSTAT``),
  ``results_root()`` (``$HOD_MOD_RESULTS``), ``cache_root()`` (``$HOD_MOD_CACHE``).
- Removed every hardcoded ``/home/comparat`` / ``~/data`` / ``~/software/sum_stat``
  path from executable code.

### Misc
- Documentation figures moved to ``docs/_images/``; README updated to the
  refactored module paths + an environment-variable setup section.
- Removed Guix install support; added a ``pre-commit`` guard against committing
  large files or ``results/``.

## [0.1.0] — 2026-06-30

A structural refactor that reorganises the package **by observable pipeline**
(galaxy clustering + lensing, galaxy × X-ray, galaxy × thermal SZ) on top of a shared
core, instead of by ingredient type (`cosmology/` vs `galaxies/`). This is a
**clean break**: internal import paths change and there are no compatibility shims.
The top-level public API (symbol names such as `MoreHODModel`,
`FullHaloModelPrediction`, `FitConfig`) is preserved via re-exports from
``hod_mod/__init__.py``.

### Breaking changes — module move map

| Old import path | New import path |
|---|---|
| `hod_mod.cosmology.*` | `hod_mod.core.*` |
| `hod_mod.cosmology.gas_profiles` | `hod_mod.gas.{pressure,density,cooling,metallicity,conversions}` |
| `hod_mod.cosmology.erosita_response` | `hod_mod.gas.erosita_response` |
| `hod_mod.galaxies.hod` | `hod_mod.connection.hod.{base,more15,zumandelbaum15,…}` |
| `hod_mod.galaxies.{clf,sham}` | `hod_mod.connection.{clf,sham}` |
| `hod_mod.galaxies.agn*` | `hod_mod.agn.{xray,ham,hod,duty_cycle}` |
| `hod_mod.galaxies.clustering` | `hod_mod.observables.clustering` |
| `hod_mod.galaxies.cross_spectra` | `hod_mod.observables.cross_spectra` |
| `hod_mod.galaxies.cross_clustering` | `hod_mod.observables.cross_clustering` |
| `hod_mod.galaxies.intrinsic_alignment` | `hod_mod.observables.intrinsic_alignment` |
| `hod_mod.galaxies.baryon_fraction` | `hod_mod.observables.baryon_fraction` |
| `hod_mod.fitting.hod_wp` | `hod_mod.fitting.{config,models,fitters}` |

### Added

- `hod_mod/observables/` — the thin top layer mirroring the three observable
  pipelines; `cross_spectra` is the shared galaxy × tSZ and galaxy × X-ray engine.
- `hod_mod/cli/` — a single consolidated CLI front door (`python -m hod_mod <cmd>`
  and the `hod-mod` console entry point) whose subcommands (`fit`, `fit-cross`,
  `fit-joint`, `benchmark`, `predict`, `validate <target>`) delegate to the existing
  scripts. `hod-mod fit` is the recommended config-driven fitter, superseding the
  near-duplicate `fit_hod_wp` / `run_fit` / `run_fit_More15` drivers (which remain
  runnable). The ~50 scripts were not physically relocated.
- Galaxy × thermal SZ promoted to a first-class, documented pipeline (`pipeline_gal_tsz`
  doc page + worked example) built on the existing, already-tested
  `HaloModelCrossSpectra` (`P_{g,y}`, `projected_gy`, `angular_cl_gy`).

### Changed

- Three oversized modules were split along their natural class/function boundaries:
  `hod.py` (2321 lines → `connection/hod/` family package: `base`, `more15`,
  `zumandelbaum15`, `guo`, `kravtsov04`, `zacharegkas25`, `vanuitert16`,
  `leauthaud12`, `lange25`), `fitting/hod_wp.py` → `fitting/{models,config,fitters}.py`,
  and `gas_profiles.py` → `gas/{conversions,pressure,density,cooling,metallicity}.py`.
  `cross_spectra.py` is kept whole as the shared cross-correlation engine.
  `observables/clustering.py` is **deliberately not split** in this release: it is the
  critical wp/ΔΣ prediction path (with the assembly-bias fix and numpy static caches),
  it can only be regression-verified through CAMB, and it is already cleanly isolated
  behind `hod_mod.observables`.
- `m200_to_m500c` (NFW M₂₀₀→M₅₀₀c) re-implemented as a vectorised, jittable JAX
  bisection, replacing the per-halo `scipy.optimize.brentq` Python loop (matches the
  former result to 2e-7). The differentiable forward model (HOD occupation, distances,
  power spectrum, halo-profile FTs) is already pure-JAX. The MAP optimiser keeps
  `scipy.optimize` (gradient-free Powell/Nelder-Mead): its objective runs through CAMB
  and the numpy MCMC caches and is not differentiable end-to-end, so a jaxopt/optax
  swap would require a CAMB-free differentiable forward model (out of scope here).
- Documentation toctree reorganised to mirror the package: User Guide → Pipelines
  (Clustering & Lensing, Galaxy × X-ray, Galaxy × tSZ) → Benchmarks → API Reference.

### Fixed (galaxy × X-ray angular spectra)

Exposed while raising test coverage of `observables/cross_spectra.py`:

- **Threaded-JAX segfault.** `angular_cl_gX` / `angular_cl_XX` defaulted to
  `n_workers=-1`, dispatching the per-redshift `_pk_tables_gX` build across
  `os.cpu_count()` Python threads. Concurrent JAX *compilation* from threads crashes
  the interpreter. Now **serial by default** (`n_workers=1`); the opt-in threaded path
  (`n_workers>1`) does a serial warm-up compile first.
- **float32 NaN.** The `_safe_log` floor `1e-60` (and the XX block's explicit `1e-120`)
  underflows float32 to 0, so an all-zero field (e.g. the AGN leg when no AGN model is
  configured) gives `log(0)=-inf`; a constant `-inf` table then makes `jnp.interp`
  compute `(-inf)-(-inf)=NaN`, poisoning the whole Limber integral. Floor raised to a
  float32-safe `1e-30`. `angular_cl_gX` now returns finite, positive spectra.

### Tests and documentation

- Test suite updated to the new layout (all imports rewritten) and extended with new
  modules covering both the refactored code and previously-untested integration paths:
  `test_public_api` (clean-break contract), `test_cli`, `test_jax_conversions`,
  `test_config_loading` (joint/ds/fits/cosmology branches + esd reader),
  `test_power_spectrum_extra` (EH no-wiggle), `test_refactor_coverage` (baryon-fraction
  models, Lange+2025 assembly bias, eROSITA `ecf_*`, `ApecCoolingTable`, bwpd reader,
  `python -m hod_mod`), and CAMB-heavy `slow` suites: `test_fitter_integration`
  (`WpFitter.map_fit`/`sample`, `DeltaSigmaFitter`, `JointFitter`),
  `test_clustering_prediction` (ΔΣ split/components, baryon, einasto, `n_gal`),
  `test_agn_ham` (HAM abundance matching, both XLFs), and the `cross_spectra` X-ray
  angular spectra (serial==threaded regression). Heavy tests are marked `@pytest.mark.slow`
  so plain `pytest` stays fast; CI runs `pytest -m ""`.
- Fixed a pre-existing test bug (`test_emissivity_uk_scaled_by_boost`: its `_THETA`
  lacked `Omega_b`/`n_s`).
- Full-suite coverage rose from ~77% to **85%** (852 tests pass). Notable per-module
  gains: `gas/erosita_response` 22→83%, `gas/cooling` 45→94%, `agn/ham` 41→88%,
  `fitting/config` 58→98%, `observables/cross_spectra` 64→87%, `fitting/fitters`
  57→74%, with `connection/hod/lange25`, `observables/baryon_fraction` and
  `cross_clustering` at 100%. A `fail_under = 82` floor was added to
  `[tool.coverage.report]` to prevent silent regression.
- Documentation revised to match the refactor: the architecture tree, prose
  file-paths, and code examples now reference the new packages (all 24 example
  imports execute). All 154 documentation links were HTTP-verified; **three wrong
  references were corrected** (Ueda+2014 `1402.7902`→`1402.1836`; Zu & Mandelbaum 2015
  `1407.8741`→`1505.02781`; and the Ogata J₀ DOI `10.1145/1141885.1141895`, which
  actually resolved to an unrelated linear-algebra paper, → the real PRIMS DOI
  `10.2977/prims/1145474602`).

### Housekeeping

- Added `hod_mod.data.erosita` (`*.npz`) to wheel `package-data` (the DR1 response
  and ECF tables were previously unpackaged).
- Removed the one-shot scratch script `_refactor_hod.py` and the empty
  `data/to_deprecate/` directory.
- `.gitignore`: the stray `hod_mod/results/` output tree, the 365 MB untracked
  `apec_v*.fits` tables (downloaded on demand by `soxs`), and the optional vendored
  `WHM/` CAMB fork are now ignored.

## [0.0.5] — 2026-06-24

### Added

- Zu & Mandelbaum 2015 (iHOD) benchmark suite, consolidated for release:
  - Digitized SDSS DR7 data from ZM15 Figure 6 added to the repository
    (`data/zumandelbaum2015_sdss/`): 7 stellar-mass-binned `w_p(r_p)` files
    (9.4–12.0), 5 binned `\Delta\Sigma(R)` files (10.2–12.0), and the raw
    WebPlotDigitizer `Fig6_*.txt` source files, with updated `metadata.json`.
  - Regenerated the model-anchored threshold-sample data vectors
    (`wp_thresh_mstar102.csv`, `ds_thresh_mstar102.csv`).
  - MAP benchmarks (threshold, ΔΣ-only, 7 stellar-mass bins, and joint) refreshed
    via `run_benchmark.py`; figures and result JSON regenerated.
  - Benchmark documentation linked into the navigation: `benchmark_zumandelbaum2015`
    and `benchmark_zumandelbaum2015_multisample` added to the `docs/index.rst`
    Benchmark toctree and summarised in `docs/benchmarks.rst`.
- Conda-free installation with GNU Guix. New files at the repository root:
  `manifest.scm` (hermetic Python + C/Fortran toolchain + runtime libs for
  manylinux wheels), `channels.scm` (pinned Guix revision for reproducible
  `guix time-machine` builds), `requirements-guix.txt` (the validated, pinned
  Python dependency set — `camb==1.4.0`, `numpy==2.4.6`, …), and `INSTALL_GUIX.md`
  (step-by-step procedure and binary-install prerequisites).
- Installation instructions for Guix in `README.md` and `docs/overview.rst`.

### Notes

- The Guix workflow runs `pip` inside a `guix shell --container --network` and
  points `LD_LIBRARY_PATH` at `$GUIX_ENVIRONMENT/lib`, so PyPI wheels (numpy,
  scipy, h5py, jax/jaxlib, camb, …) load against the Guix interpreter without
  relying on Guix's own Python packages or clobbering glibc.
- Use `guix time-machine -C channels.scm` so the environment pins to Python 3.11:
  `camb==1.4.0` (validated, source-only) needs ≤ 3.11, and an unpinned newer camb
  on Python 3.12 fails under numpy ≥ 2.4 (``camb/model.py:691`` TypeError).
- `.venv-guix/` added to `.gitignore`.

## [0.0.2] — 2026-06-01

### Added

- Benchmark configs for More+2015 stellar-mass subsamples: `benchmark_more2015_logM11_12.yml`,
  `benchmark_more2015_logM11p3_12.yml`, `benchmark_more2015_logM11p4_12.yml`,
  and the free-cosmology variant `benchmark_more2015_logM11_12_freecosmo.yml`.
- Digitized joint wp+ESD data for More+2015 subsamples A/B/C:
  `data/more2015_boss_cmass/logM11_12/`, `logM11p3_12/`, `logM11p4_12/`.
- Benchmark configs for Lange+2025 DESI DR1 bwpd series (12 configs):
  BGS2, BGS3, LRG1, LRG2 × wp-only / ESD-only (HSC) / joint wp+ESD (HSC).
- Manually digitized (WebPlotDigitizer) data for Lange+2025 in bwpd format:
  `wp_*_bwpd.csv` and `ds_hsc_*_bwpd.csv` for all four samples.
- Documentation for More+2015 and Lange+2025 benchmarks.
- Dedicated digitization scripts and raw figure archives for both datasets.

### Changed

- Documentation: benchmark navigation reorganised — only More+2015 and Lange+2025
  are linked in the main toctree; other benchmarks exist but are not yet reachable
  from the navigation until their data and fits are validated.
- Analysis of LS10/BGS pages removed from the documentation navigation.
- Lange+2025 benchmark model keys updated to `bwpd` naming convention
  (`lange2025_bgs2_bwpd_hsc`, etc.) to reflect the new manually-digitized dataset.

## [1.0.0] — 2025-05-21

### Added

- Initial release of `hod_mod`.
- `HODBase` abstract base class: all 9 clustering HOD classes now inherit from it.
  Implements `_integrate()`, `galaxy_number_density()`, `effective_bias()`, and
  `effective_mass()` once, delegating to the `nc_ns()` extension point.  Saves
  ~1 500 lines of copy-pasted boilerplate.
- `_SINGLE_ARG_INIT` class flag on `HODBase` subclasses replaces the
  `_HOD_SINGLE_ARG` string set that was used in the original fitter.
- Unified `FitConfig` dataclass replaces the three separate config classes
  (`WpFitConfig`, `JointFitConfig`, `WpFitFITSConfig`).
- Single `load_config()` function auto-detects `joint` and `fits` YAML sections.
- `WpFitter` / `JointFitter` fitter hierarchy; `WpFitterFITS` is now an alias.
- `Kravtsov04HODModel` added to the `HOD_MODELS` dispatch dict.
- Backward-compatibility aliases: `AUMHODModel = Kravtsov04HODModel`,
  `WpFitConfig = JointFitConfig = WpFitFITSConfig = FitConfig`,
  `load_joint_config = load_fits_config = load_config`.

### Breaking changes

- All imports must use `from hod_mod.*` (the previous package name is no longer supported).
- `WpFitterFITS` is now identical to `WpFitter`; the FITS-specific class was
  merged into the base fitter via `_load_data()` dispatch.
