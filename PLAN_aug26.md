# v0.6 — layered forward model, one amplitude, one growth, free gas

Living plan for the upgrade started 2026-08-25. **Update the status markers in
this file as work lands**; it is the shared record of what is done and what is
not.

- Branch: `feature/v0.6-relayer`, cut from `feature/single-amplitude-lnAs`
- Target release: `0.6.0`; campaign tag `VTAG=v0.6`
- Breaking compatibility is explicitly sanctioned. Equivalent tests are rebuilt
  after each change rather than preserved through it.

Legend: **DONE** · **WIP** · **TODO** · **DEFERRED**

---

## 1. Status at a glance

| # | Commit | Status | Moves numbers |
|---|---|---|---|
| 0 | Reroot onto `feature/single-amplitude-lnAs` | **DONE** | — |
| 1 | Baryon split unreachable; two paths fell out of the trace (`38ed29e`) | **DONE** | yes (ΔΣ, for anyone who passed `baryon_params`) |
| 2 | Name the two flavours: `core/backend.py` (`3c42845`) | **DONE** | no |
| 3 | Amplitude residuals: `planck_prior` double-counting, migration-only `_sigma8_to_lnAs`, derived σ8/S8 in chains | **TODO** | yes (priors) |
| 4 | `core/distances.py`: radiation + massive-ν `E(z)` | **TODO** | yes (χ(z) ~1e-4) |
| 5 | `core/cosmology_state.py::CosmologySnapshot` | **TODO** | no |
| 6 | Delete Carroll+1992 | **TODO** | only for `HOD_MOD_GROWTH=carroll` |
| 7 | σ(M,z) from P(k,z); growth becomes a snapshot field | **TODO** | yes (2.4e-5 → 2.3e-4) |
| 8 | Neutrino residuals, incl. `HaloProfile` cosmology | **TODO** | **yes, large** (c(M) 7–13%) |
| 9 | `core/halo_field.py::HaloField` — owns the mass grid | **TODO** | no (assert bit-identical) |
| 10 | `observables/sectors/`: protocol, `GalaxyOccupation`, `MisCentering`, `ModelParams` | **TODO** | forecast only |
| 11 | `observables/spectra.py::pk_cross` — one 1h+2h integral | **TODO** | yes (`I_g(k)` at k ≳ 1) |
| 12 | `HotGasDPM` + `AgnSector` as peers | **TODO** | no |
| 13 | `observables/assembly.py::Assembly` | **TODO** | no |
| 14 | Referee A1a + A1b + A2: DPM normalisation and one `c_DPM` | **TODO** | **BOUNDARY** |
| 15 | `gas/dpm_params.py` registry + bounds; metallicity constructor | **TODO** | no at defaults |
| 16 | Prior in scaling-relation space | **TODO** | yes (gas posteriors) |
| 17 | Retire the `(p2, r_max)` transfer grid; all DPM parameters free | **TODO** | **BOUNDARY** |
| 18 | Matter tracer: CDM + DPM gas + stars | **TODO** | **BOUNDARY** (all lensing) |
| 19a | `FAST` wired; twin deleted; `ForwardModelLegacy` frozen as parity reference | **TODO** | no |
| 19b | `forward_jax.py` → `params_vector.py` + `observables_extra.py` | **TODO** | yes (forecast) |
| 20 | Delete the legacy prediction classes; merge the two HOD registries | **TODO** | no |
| 21 | `make_flavour_parity.py` + `benchmarks/flavour_parity.json` + tests | **TODO** | no |
| 22 | Config migration, CHANGELOG boundary, campaign tag | **TODO** | no |

---

## 2. Why

Four things block work, not merely offend taste.

**The layers are fused.** `observables/clustering.py::FullHaloModelPrediction`
(1200 lines) is simultaneously sector glue, power-spectrum assembler and
observable projector. Galaxy occupation is not a peer of the other sectors — it
is the *host*: the halo mass grid belongs to `self._hod._m_grid`
(`connection/hod/base.py:114`), and the gas, AGN and cluster legs exist only by
wrapping a galaxy predictor and reading its privates. A pure pressure
auto-spectrum still has to be handed an HOD parameter dict. Mis-centering is not
a module — twenty inlined lines duplicated across the numpy and JAX paths, plus a
second, differently-named implementation in `observables/lensing.py`. Hot gas
exists three times in three parametrisations. The same
`∫dM dndm·A(k,M)·B(k,M)` integral is hand-written eight times; `wp`/`ΔΣ` four
times.

**Two amplitude conventions coexisted and one was silently inert.** Fixed on
`feature/single-amplitude-lnAs`; see §4.

**D(z) is computed by five routes.** A dispatcher exists, but only
`HaloMassFunction._growth` reaches its accurate mode; every other site forces the
analytic ODE or silently falls back to it, and three modules bypass the
dispatcher with their own hard-wired flat-ΛCDM formula. Carroll+1992 is still
selectable and its error is Ω_m-dependent (a free-cosmology fit cannot absorb it)
with `∂lnD/∂w0` structurally zero — precisely the sensitivity a Fisher forecast
differentiates through.

**The gas sector is mostly frozen and never reaches lensing.** The production
X-ray fit varies nine DPM parameters and hard-fixes the rest, including the whole
pressure shape and the entire metallicity sector; `c_DPM = 2.772` appears as six
independent literals. The DPM gas feeds X-ray and tSZ, but the lensing leg of the
*same* cross-spectrum is a total-mass NFW — in `C_ℓ^{κy}` the two sides do not
share a gas sector.

Target: real layer boundaries (cosmology → halos → sectors → spectra →
observables), every sector a peer, one amplitude, one growth route, one gas
sector, and a fast differentiable path that is the *same code* as the accurate
path under a different backend profile — with the disagreement measured row by
row rather than assumed.

---

## 3. Defects confirmed by reproduction

Not inherited from a report; each was re-run against this tree.

| Defect | Evidence | Status |
|---|---|---|
| Free-cosmology `S8` inert | 65% intended amplitude change → `max\|P_b/P_a−1\| = 0.0` | **FIXED** on the amplitude branch (now 0.653) |
| `baryon_params` landed in `n_pi` / `chi_max` | `clustering.py:2034`, `:2054`; `wp` never raised | **FIXED** `38ed29e` |
| ECF weight broke the traced path | `cross_spectra.py:711` unconditional `np.asarray` | **FIXED** `38ed29e` |
| ZM15 inverse-SHMR gradient structurally zero | `d log10M*/d beta`: autodiff `0.0` vs central FD `0.105` (referee A7) | **TODO** — commit 19a |
| `HaloProfile` discards the caller's cosmology | `halo_profiles.py:704` `setCosmology("planck18")`; 20 call sites pass a dict that is ignored | **TODO** — commit 8 |
| `M₁₂` pivot units disagree within one function | `dpm_priors.py:92` physical M⊙ vs `:115` M⊙/h | **TODO** — commit 14 |

---

## 4. Requirement 2 — sigma8 / S8 retired (mostly already done)

`feature/single-amplitude-lnAs` (6 commits, 61 files) landed before this branch
and is its base. Verified functionally, not taken on trust: the amplitude
responds (0.653), σ8 is derived (0.6918 → 0.8895 across the same step), and
`require_as_amplitude` raises on a retired key.

Already there: `core/amplitude.py` (canonical `sigma2_tophat` replacing four
copies, `sigma8_from_theta`, `s8_from_theta`, the tripwire), `forecast/derived.py`
(Fisher propagation of derived σ8/S8), EH98 rebuilt on `eisenstein_hu_pk_phys`,
the σ8 branch deleted from CosmoPower, the σ8 anchor deleted from
`HaloMassFunction.sigma`, 37 config files migrated, the forecast vector
reindexed, and referee A20.

**Remaining (commit 3):**

- `fitting/planck_prior.py:47-75` still carries `sigma8`, `S8` *and*
  `ln10^{10}A_s`, and `planck18_log_prior` adds a term for **every** key present
  → double-counted amplitude prior. Delete the retired entries; raise if
  `params=` names one. A Planck σ8 prior, if wanted, is a prior on a *derived*
  quantity and belongs in a separate function that computes σ8 from the current
  theta — not a third independent Gaussian.
- `fitting/config.py:107-123 _sigma8_to_lnAs` must delete the `sigma8` key after
  converting and be reachable only from a `--migrate` path, never from
  `load_config`. Delete the verbatim copy at
  `scripts/benchmarks/run_benchmark_hsc_wp_lange25.py:182-190`.
- Record derived σ8/S8 as blobs at every MCMC sample so posteriors carry them
  without a post-processing pass (`fitting/fitters.py`, `fitting/jax_inference.py`).
- `CsstHaloMassFunction.sigma` — referee B12 remainder ("Ω_m ignored on the
  massless path"), close in commit 8.

---

## 5. Target structure

New package `hod_mod/model/` (or `observables/sectors/`) holds *layer machinery*
only; physics stays in its existing homes and each sector gains an adapter.

```
L0  core/                     distances, growth, amplitude, power_spectrum,
                              halo_profiles, concentration, nu_ratio
L1  core/backend.py           Backend — the flavour profile          [DONE]
    core/cosmology_state.py   CosmologySnapshot — the resolved cosmology
L2  core/halo_field.py        HaloField — OWNS the m/k grids and halo tables
L3  observables/sectors/      protocol + four peers + modifiers
       occupation.py          GalaxyOccupation   (wraps connection/)
       miscentering.py        MisCentering       (k-space modifier; was 3 copies)
       hot_gas.py             HotGasDPM          (wraps gas/; three tracer views)
       agn.py                 AgnSector          (was inline in cross_spectra)
       matter.py              MatterField        (CDM + gas + stars composite)
L4  observables/spectra.py    pk_cross / pk_auto — ONE 1h+2h integral
L5  observables/projections.py hankel_j0, abel, limber, delta_sigma
    observables/assembly.py   Assembly — the single public predictor
```

**Who owns the mass grid.** `HaloField`, built from a `Backend` and a
`CosmologySnapshot`. There are five independent owners today
(`connection/hod/base.py:114`, `connection/hod/zumandelbaum15.py:406`,
`connection/clf.py:448`, `observables/clustering.py:1799`, `agn/hod.py:290`), all
hard-coded `logspace(10,16,N)` with N ∈ {256, 512, 600}. All five delete. That
single move is what demotes galaxy occupation from host to peer.

`HODBase.nc_ns(log10m_arr, params)` keeps its exact signature — all eleven
occupation classes are unchanged. What goes is `HODBase._integrate` / `_m_grid` /
`_log10m_grid`; the three consumers move to `GalaxyOccupation` and take a
`HaloField`. `connection/hod/lange25.py:71` and `connection/clf.py:451` override
`_integrate` and need the same treatment.

**The tracer contract (L3 → L4):**

```python
@dataclass(frozen=True)
class TracerWeights:
    w_point: Array | None     # (Nk,NM) or (NM,) — already k-modulated
    w_extended: Array | None  # (Nk,NM)
    norm: Array               # n̄ for discrete tracers, 1.0 for fields
    discrete: bool            # governs the auto-spectrum pair rule
    bias_weight: Array | None # optional 2-halo override
    name: str
```

with `W(k|M) = (w_point + w_extended) / norm`.

**Parameters:** promote `core/containers.py::ParameterContainer` (already a
correct JAX pytree, currently dead code) to a `ModelParams` with namespaced
leaves `cosmology / occupation / miscentering / gas / agn / matter`. Each leaf
stays a plain dict of floats, so `nc_ns`, `pressure_uk`, `emissivity_full_uk`
need no signature change. Namespacing is what structurally kills the
`p_off`/`R_off`-smuggled-through-`hod_params` problem.

**The generic integral** replaces all eight copies (`clustering.py:1024-1094`,
`cross_spectra.py:535/610/737/783/927`, `cross_clustering.py:164`,
`core/halo_model.py:112`, `forward_jax.py:1133`). The `discrete` pair rule is the
only physics branch. It makes two silent inconsistencies explicit:

- `P_gg^2h` is currently `b_eff²·P_lin` with `ũ→1` forced at all k; the generic
  form gives the real `I_g(k)`, which differs at k ≳ 1 h/Mpc. **Numbers-moving.**
- `P_my^2h` already uses the real `I_m(k)` while `P_gm^2h` does not. The generic
  engine makes them consistent by construction.

**Mis-centering** becomes a `ProfileModifier` returning `h_c(k|M)`, with the
convention as a *static config field* rather than `if "R_off" in hod_params`
(which is why the traced and numpy paths can currently take different branches).
It then applies to AGN centrals and cluster centres too, which nothing supports
today. The real-space implementation at `lensing.py:149-193` is **kept as an
independent reference** and becomes an accuracy-budget row (k-space vs
real-space, tol 1e-3).

---

## 6. Flavours — `core/backend.py` [DONE]

`ACCURATE` (CAMB/CLASS, fine grids, numpy GL gas FT, colossus c(M), not traced)
and `FAST` (CosmoPower-JAX, coarse grids, gNFW surrogate, `dutton14`, traced).
`ACCURATE`'s settings reproduce the pre-registry constants exactly, pinned in
tests against their original homes.

**The mechanism that makes one assembly serve both:** the P(k) backend call is
hoisted out of the traced region into `CosmologySnapshot`. Under `ACCURATE` its
fields are concrete float64 arrays; under `FAST` they are tracers and the whole
thing composes under `jit`. Everything downstream is *one* implementation written
in `jnp`.

Consequences: `_pk_tables_full_jax` dies, so do the four `jax.disable_jit()`
blocks and `_validate_eh98_backend` (with the backwards `observables → forecast`
import edge). `make_differentiable_prediction` becomes `Assembly(backend=FAST)`.

**Gradients under `ACCURATE` must raise**, not silently return a gradient missing
∂P_lin/∂θ. Already enforced at the type level: `ACCURATE.with_(traced=True)`
raises.

---

## 7. Growth

One route: `pk_ratio`. `ode` survives *only* as the declared fallback for a
backend with `has_native_z = False` (EH98) and as the budget reference.
**Carroll is deleted** — `GROWTH_MODES`, `_growth_factor_flat`,
`_growth_factor_flat_jax`, the dispatch branch, and the budget rows. The
*measurement* stays in `docs/growth_factor.rst` as the recorded justification.

**Structural fix: nobody calls `growth_factor` any more.** Growth becomes a field
of `CosmologySnapshot` (`growth_total(z)`, `growth_cold(z)`, `growth_d0()`), so a
caller holding a snapshot cannot silently get the ODE. That closes the four
forcing sites and the four silent fallbacks (`forward_jax.py:2188`,
`lensing.py:269`, `intrinsic_alignment.py:56`, `agn/ham.py:510`) at once. The
forecast-HMF fallback is a one-kwarg fix per site: `make_hmf(..., pk_obj=pk)`.

**σ(M,z) from P(k,z) directly — take it now.** The ~24 signature sites are inside
the layers being rebuilt; deferring means touching them twice. It also removes
the scalar-D(z) ambiguity (at Σmν = 0.06 the k-spread of √(P(k,z)/P(k,0)) is
3.4e-3, an order of magnitude above the ODE's own 2.1e-4 error) and deletes the
load-bearing `variant="total"` warning in `HaloMassFunction.sigma`, whose
`_cold_factor` and growth multiplication both disappear. Measured effect
2.4e-5 (z=0.3) to 2.3e-4 (z=3).

---

## 8. Neutrinos

- **`hubble_e`: implement the real transition, do not document the
  approximation.** `core/distances.py:75-118` has neither radiation nor a ν
  sector. Use the standard Komatsu form,
  `Ω_ν(z) = Ω_ν,0(1+z)³ f(y)`, `f(y) = [1+(Ay)^p]^{1/p}`,
  `A = 180ζ(3)/(7π⁴)`, `p = 1.83` — <0.1% across the transition. Then delete
  `clustering._hubble_E` and `forward_jax._e2z`, and route the eight
  w0/wa-dropping `comoving_distance` calls plus `forecast/noise.py:43` through
  `core/distances`.
- `eisenstein_hu_pk_phys` gains ν suppression at *z* (matching CosmoPower; EH98
  currently applies it at z=0). `lensing._pk_lin_tab` is replaced wholesale by
  the snapshot spectrum, removing the ν-blindness, the fourth σ8 renormaliser and
  the fourth top-hat copy in one edit.
- `concentration._neff_eisenstein_hu:379` uses the bare unsuppressed shape.
- **`clustering._rho_m` (total Ω_m) vs the HMF's `omega_cb` are both correct and
  must stay different** — lensing sees neutrinos, halos form from the cold field.
  The bug is that they share one name. `HaloField` exposes `rho_cold` and
  `rho_matter`; a test asserts the declared choice at each of ~12 sites. Same
  resolution for the `f_b^cosmic` denominators.
- `halo_profiles.py:704`: build the colossus cosmology from the caller's theta,
  memoised on a rounded key, with a **derived** σ8 — a nice demonstration that
  σ8-as-output works. Measured cost of the current bug: 7–13% in c(M).
  Note colossus has no ν sector, so `diemer19` at Σmν > 0 stays approximate;
  record it and make the JAX-native relation the default for both flavours.
- `nu_ratio` must raise past 0.50 eV / z=5 rather than extrapolate flat.
- No shipped config exposes `sum_mnu`; add it to every `cosmology:` block.

---

## 9. DPM — nothing fixed

**Sequencing is not optional: the normalisation fix lands first.** Freeing the
parameters over an 11.6× normalisation error lets the sampler absorb it into the
amplitude and report a good fit at wrong physics.

1. **Referee A1a + A1b + A2 together (commit 14).** `×1e-6` →
   `k_B = 8.617333e-8 keV/K` (`gas/pressure.py:246`); the `M₁₂` pivot at four
   sites (`density.py:185`, `density.py:199`, `pressure.py:294`,
   `dpm_priors.py:115`), all dividing an M⊙/h mass by 10¹² where the paper
   defines the pivot in physical M⊙; and one `c_DPM` shared by all three
   profiles. **Check the DPMhalo source before changing anything.**
   Acceptance test: DPM Model 1 is self-similar by construction, so
   `kT/kT_vir` must be flat in mass — 0.74 with `k_B` alone, 0.96 with both, and
   all three models give ≈6.01 keV at 10¹⁵ M⊙/h. Knock-on: `_LOG10_P03_DEF`
   moves outside its own prior window; the new bounds table supersedes it. The
   pivot fix makes the gas profiles depend on `h` for the first time.
2. **Expose everything (commit 15).** `gas/metallicity.py` has *no constructor
   arguments at all*; it gains six. `gamma_n`, `gamma_P`, `alpha_out_var` (DPM
   Eq. 5, dead at 0.0 in all three models), `sigma_scatter` (DPM Eq. 6, off
   everywhere) and `_NH_RATIO` all become parameters. Declarative registry in
   `gas/dpm_params.py`; `forecast/params.py`'s existing `bounds_vector` /
   `bounds_pairs` / `log_prior_bounds` / `clip_to_bounds` are reused as-is.
3. **The prior (commit 16).** `induced_gaussian_prior` is a **square** 4→4 map
   solved with `jnp.linalg.solve` / `inv`; both fail the moment more parameters
   are freed. `scaling_map` itself is differentiable and extends mechanically
   (slopes move from `ctx.shape` into `theta_n`). **Apply the prior in
   scaling-relation space directly** — exact, no Newton solve, no Jacobian
   inversion, no cache, and automatically in the right coordinates however the
   shape parameters move. That also dissolves the prior/likelihood coordinate
   mismatch (`_scaling_ctx` uses 1.0/1.9/2.7 while the transfer grids use
   0.9/2.0). Keep `induced_gaussian_prior` for reporting and proposal
   covariance. The v0.4 kT–M inversion regression test stays.
4. **The transfer grid is what forces the freezing (commit 17).**
   `fit_xray_joint_bands.py` grids `alpha_out` and `r_max` because `G_grid`/`J`
   are precomputed on those axes, and pins the pressure shape for the same
   reason. **Measure the traced/surrogate emissivity FT cost before
   committing** — `emissivity_full_uk` is ~1.3 s per call today. If it does not
   reach ~10 ms, a trained surrogate over
   `(alpha_in, alpha_tr, alpha_out, r_max, T_0, Z)` is the fallback, which is
   exactly what `Backend.gas_ft="surrogate"` is for.

---

## 10. CDM on one side, DPM baryons on the other

One gas sector, three tracer views (`mass_uk`, `pressure_uk`, `emissivity_uk`)
from one parameter set, and

```
W_m(k|M) = (M/ρ̄_m)·[ (1 − f_gas − f_★)·ũ_DM + f_gas·ũ_gas^DPM + f_★·ũ_★ ]
```

Each `ũ` is normalised to 1 at k→0 and the fractions sum to 1, so
`W_m(k→0) = M/ρ̄_m` **exactly** — mass conservation by construction, pinned to
1e-6 in both flavours and added to `_MUST_COVER`. `ũ_★` is a point mass, which
is where the `log10_M_star_cen` term bolted onto ΔΣ post-hoc
(`clustering.py:1459-1463`) belongs.

**`f_gas` derived from the DPM density profile**,
`f_gas(M) = μ_e m_p ∫ n_e(r|M) 4πr² dr / M`. This is what makes DPM *the* gas
sector rather than one of two, and it has a payoff the current fit lacks: the
absolute `n_e` normalisation is unconstrained today (`c_total` is anchored
empirically; `n_e,0.3` ↔ `A_gas` is a documented degeneracy), so **the lensing
leg breaks the X-ray amplitude degeneracy**. Risk: nothing stops `f_gas`
exceeding `f_b^cosmic`. Do **not** clip — it kills the gradient. Reparameterise
as `f_b^cosmic · sigmoid(·)` or bound; decide by measuring over the prior volume.
`BaryonFractionSigmoid` survives for gas-free runs and as a prior on `f_gas(M)`.

**`P_gg` is unchanged** — galaxies are a discrete tracer and satellites follow
the dark matter. Verified in `38ed29e`: `wp` moves by exactly zero. What changes
is `P_gm`, `P_mm`, `ΔΣ`, `C_ℓ^{κκ}`, `C_ℓ^{κy}`, `C_ℓ^{κg}`.

**The 2-halo hack dies.** "Assign the whole 2-halo term to CDM so cdm+b = total"
(`clustering.py:1084`) is replaced by `P_gm^2h = P_lin·I_g(k)·I_m(k)` where `W_m`
already carries the split, so the decomposition follows from the same weights. At
k→0, `I_m → 1` and it reduces to `b_eff·P_lin` as before. The post-hoc scalar
fallback `ds_cdm = (1 − Ω_b/Ω_m)·ds_total` is deleted with no replacement.

**`C_ℓ^{κy}` gets fixed as a side effect** — today its lensing side is
total-matter NFW while its y side is DPM pressure (`cross_spectra.py:546-554`,
`P_my` uses bare `uk` with no `(1−f_b)` weighting), so the two legs of one
cross-spectrum do not share a gas sector.

The third gNFW gas at `forward_jax.py:904-917` is **the design target, not a
casualty**: it already does exactly this. It moves into
`gas/density.py::GasDensityDPM.mass_uk` and becomes the production
implementation.

HMcode's `mead2020_feedback` stays disconnected, now for a stateable reason: the
DPM gas *is* the feedback model and running both double-counts. Add a guard.

---

## 11. Quantifying the deviation (requirement 1)

A **second** budget file `benchmarks/flavour_parity.json`, generated by
`hod_mod/scripts/make_flavour_parity.py`, which imports `_entry` / `_dev` /
`_git_sha` verbatim from `make_accuracy_budget.py` — same schema, same
skipped-entries-are-recorded discipline, same non-zero exit. Separate from
`accuracy.json` because the two ask different questions (*is the physics right*
vs *is the fast path faithful*) and have different regeneration costs.

Metric is **max |ratio − 1|** with the median recorded alongside. The existing
forecast-vs-production test asserts `median < 0.03` on wp/ΔΣ only, with
`_FIXED_BARYON["f_b_min"]` monkey-patched to 0 — that weakness is what this
replaces.

Rows: `P_lin(k,z)` 3e-3 · `D(z)` total/cold 1e-3 · `sigma(M,z)` 2e-3 · `dn/dM`
2e-2 · `b(M,z)` 5e-3 · `c(M,z)` 3e-2 · `u_NFW` 1e-4 · `u_gas`/`ỹ`/`X̃` 1e-2 ·
`P_gg`,`P_gm` 1e-2 · `P_gy`,`P_gX`,`P_XX`,`P_yy`,`P_mm` 2e-2 · `wp` 1e-2 ·
**`ΔΣ` with baryons on** 1.5e-2 · `n_gal`,`b_eff` 5e-3 · `w(θ)`,`ξ_cg` 2e-2 ·
`C_ℓ` 3e-2 · derived `sigma8`,`S8` 1e-3.

**Gradient rows matter more than value rows** — a fast flavour whose values agree
to 1% but whose derivatives are 30% off gives a wrong Fisher matrix and nothing
else catches it. `∂ln wp/∂θ_i` and `∂ln ΔΣ/∂θ_i` for 8 representative θ,
AD-fast vs FD-accurate, tol 5e-2.

CI: `tests/test_flavour_parity.py` mirroring `test_accuracy_budget.py`, **plus an
in-process smoke subset** (wp, ΔΣ, P_gy, one gradient) — the JSON-only check goes
stale the moment someone forgets to regenerate. Also close referee B28
(`make_accuracy_budget.py:243` hard-codes the AUM path).

---

## 12. Verification

```bash
python -m pytest -q                              # full suite
python -m pytest -m x64 -q                       # gradient checks, separate process
python -m hod_mod.scripts.make_accuracy_budget   # 0 out of tolerance
python -m hod_mod.scripts.make_flavour_parity    # 0 out of tolerance
```

Four defect reproductions must flip:

1. **S8 responds** — non-zero (~0.65) where it gave exactly `0.0`. *(already flipped)*
2. **ZM15 gradient matches FD** — `d log10M*/d beta` to ~1e-6, now `0.0` vs `0.105`.
3. **`kT/kT_vir` flat in mass** for DPM Model 1 over 10¹²–10¹⁵ M⊙/h; a 10¹⁵
   cluster at 5–10 keV, not 70.
4. **c(M,z) responds to cosmology** — two theta differing in Ω_m give different
   concentrations through `HaloProfile`.

End to end: one MAP fit via `fit_benchmark_observables` against its recorded χ²,
with the shift *explained* by commits 3–8 rather than merely noted; a tier-2
forecast (Fisher finite, `ln10^{10}A_s` constrained, sane `probe_decomposition`);
`growth_factor_study` regenerating `docs/growth_factor.rst` with Carroll gone.

---

## 13. Risks

**Likely underestimated**

- **Commit 19b.** `forward_jax.py` is 2478 lines with ~111 parameters, and
  `tier2/3/4`, `tomography`, `fisher`, `covariance`, `noise` all index into its
  internals. Largest item here; expect 2×. Mitigated by the 19a/19b split and by
  keeping `ForwardModelLegacy` for a full release — which doubles as the parity
  reference, so it is not wasted.
- **Commit 17.** If the traced emissivity FT does not reach ~10 ms, freeing the
  slopes makes the fit infeasible. Measure first.
- **Commit 8.** A 7–13% change in c(M) is not a cleanup, it is a re-fit of every
  `use_free_cosmo` benchmark.
- **Commit 11.** Collapsing eight integrals to one will surface disagreements
  between duplicates that are invisible today. Adjudicate and *record* each
  delta rather than silently picking a winner.
- ~70 `make_hmf` / `HaloMassFunction(` sites need `pk_obj=`.

**Could silently produce wrong numbers**

1. **Gradients on `ACCURATE`.** Once the assembly is uniformly traceable,
   `jax.grad` would *succeed* and omit ∂P_lin/∂θ. Must raise. *(blocked at the
   type level in `3c42845`; still needs the API-boundary check in commit 19a)*
2. **The pair rule.** A wrong central self-pair exclusion changes `P_gg^1h` at
   the 1/N_c level and nothing errors. Pin with an analytic Poisson test *before*
   any consumer switches over.
3. **ρ̄ cold vs total.** Three sites, three conventions today; the generic engine
   makes it one field, so a wrong choice propagates everywhere at once.
4. **`f_gas` exceeding `f_b^cosmic`.** A `clip` gives plausible numbers with a
   dead gradient.
5. **Any surviving partial cache key.** `_cosmo_cache_key` is already known-stale
   (no `sum_mnu`/`w0`/`wa`); if any hand-picked key subset survives the refactor,
   every parity measurement above is undermined.
6. **`nu_ratio` extrapolation** past its grid returns a plausible number today.

**Deferred, with reasons**

- **Baryonification / BCM / adiabatic contraction** — the DPM gas *is* the
  feedback model for this release; a second one invites double-counting.
- **Tier-3/4 extra sectors** (HI, radio, IR, morphology, sizes, SFR) — forecast-
  only, no production twin, ~1000 lines of churn. They move as-is to
  `forecast/observables_extra.py`.
- **BNL** (`core/beyond_linear_bias.py`) — a 2-halo *correction*, not a profile;
  it does not fit the tracer abstraction. Keep as an explicit `TwoHaloCorrection`
  hook, `ACCURATE` only.
- **`HODProjectedCorrelation`, `w(θ)` Limber refinement** — delete rather than
  port.
- **Switching the default `ACCURATE` P(k) from CAMB to CLASS** — separate
  decision, separate comparability note. CLASS stays the budget *reference*.
- **The rest of `REFEREE_ACTIONS.md`.** Only A1, A2, A7, A20, B12, B28 are
  absorbed here because they block this work. A16/A17/B18 (ΔΣ inner annulus, IA
  equations, Kaiser residual) touch the layer-5 rebuild and should follow it.

**What not to do**

- Do not re-implement the σ8 retirement — it is the base of this branch.
- Do not keep `_pk_tables_full_jax` "for safety". Two hand-maintained twins is
  the failure mode this exists to remove; the parity harness is the safety net.
- Do not free the DPM parameters before the A1 normalisation fix.
- Do not use the median as a parity metric.

---

## 14. Progress log

- **2026-08-25** — Rerooted onto `feature/single-amplitude-lnAs` after verifying
  it functionally (amplitude responds 0.653; σ8 derived 0.6918 → 0.8895;
  tripwire raises). Landed `38ed29e` (baryon split reachable, keyword-only
  optional arguments, two traced-path fixes; `tests/test_baryon_plumbing.py`,
  static half verified to fail on the original code) and `3c42845`
  (`core/backend.py`, `tests/test_backend.py`, 26 tests).
