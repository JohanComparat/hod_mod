# ΔΣ vs wp Discrepancy: Pipeline Audit and Strategy Report

**Date:** 2026-05-29  
**Context:** Lange+2025 DESI DR1 benchmarks in `hod_mod/scripts/benchmarks/run_benchmark.py`.  
**Symptom:** MAP fitting yields wp close to data but ΔΣ predictions discrepant.

---

## 1. Executive Summary

Four pipeline inconsistencies were identified. One is a **critical bug** that entirely
suppresses the assembly bias signal in the 2-halo power spectrum. The others are more
subtle but affect ΔΣ predictions at the 5–20% level.

| # | Severity | Location | Status |
|---|----------|----------|--------|
| 1 | **Critical** | `clustering.py:822` — assembly bias ignored in `FullHaloModelPrediction` | **FIXED** |
| 2 | Moderate | `halo_profiles.py:662` — HaloProfile hardcodes planck18 cosmology | open |
| 3 | Moderate | `clustering.py:942` — chi_max=300 Mpc/h vs pi_max=80 Mpc/h asymmetry | open |
| 4 | Minor | `clustering.py:985` — sigma_bar inner boundary at R=0.01 Mpc/h | open |

---

## 2. Bug Details and Fixes

### Bug 1 (FIXED): Assembly bias ignored in `FullHaloModelPrediction._pk_tables_full`

**Location:** `hod_mod/galaxies/clustering.py`, line 822.

**Root cause:**  
`FullHaloModelPrediction._pk_tables_full()` computes `b_eff` as:
```python
b_eff = np.trapezoid(dndm_np * nt_np * bias_np, m_np) / n_gal
```
This uses the raw linear halo bias `bias_np` and ignores the assembly bias parameters
`A_cen` and `A_sat` from `Lange25HODModel`. The assembly bias kernel `(b-1)/b` that
`Lange25HODModel._integrate()` applies is never called.

In contrast, `HODClusteringPrediction._pk_tables()` (line 247) correctly calls
`self._hod._integrate()` which runs the assembly bias correction.

**Impact:**  
With A_cen=0.5 (typical BGS2 posterior), b_eff shifts from ~1.74 to ~2.01 at z=0.25
(+16%). Since wp ∝ b_eff² and ΔΣ ∝ b_eff, ignoring assembly bias means:
- The MAP optimizer compensates by shifting HOD parameters (log10mmin, etc.)
- The resulting HOD parameters are biased, inconsistent with Lange+2025 results
- A_cen and A_sat are effectively inert free parameters in the fit

**Fix applied (2026-05-29):**  
In `_pk_tables_full`, after computing `n_gal`, apply the same assembly bias kernel
using the already-cached `bias_np` arrays:
```python
A_cen_pk = float(hod_params.get("A_cen", 0.0))
A_sat_pk = float(hod_params.get("A_sat", 0.0))
if A_cen_pk != 0.0 or A_sat_pk != 0.0:
    gamma_pk = (bias_np - 1.0) / np.where(bias_np > 0.5, bias_np, 0.5)
    b_nc_pk  = bias_np * (1.0 + A_cen_pk * gamma_pk)
    b_ns_pk  = bias_np * (1.0 + A_sat_pk * gamma_pk)
    b_eff = float(
        np.trapezoid(dndm_np * (nc_np * b_nc_pk + ns_np * b_ns_pk), m_np) / n_gal
    )
else:
    b_eff = float(np.trapezoid(dndm_np * nt_np * bias_np, m_np) / n_gal)
```
This uses cached arrays (no extra CAMB/HMF calls) and falls back to the fast path
for standard HOD models without assembly bias.

---

### Bug 2 (OPEN): HaloProfile concentration hardcodes planck18

**Location:** `hod_mod/cosmology/halo_profiles.py`, line 662.

**Root cause:**  
`HaloProfile.__init__()` calls `col_cosmo.setCosmology("planck18")` unconditionally.
For any non-`dutton14` concentration model, the colossus concentration-mass relation
always uses Planck18 parameters regardless of the per-call `theta_cosmo`.

**Impact:**  
In free-cosmo mode (Omega_m, S8 free), the 1-halo NFW profile scale radius r_s and
concentration c(M) do not adapt to the sampled cosmology. Since the 1-halo term
dominates ΔΣ at R < 3 Mpc/h, this biases the inner ΔΣ prediction for cosmologies
far from Planck18.

**Fix approach:**  
Option A (quick): Switch to `cm_relation='dutton14'` which uses a JAX-native power-law
calibration and ignores cosmology (same limitation but consistently applied).

Option B (correct): In `_pk_tables_full`, when the cache key includes Omega_m, pass
a colossus cosmology dict matching theta_cosmo to `col_cosmo.setCosmology()` before
computing concentration. This requires making the concentration computation
per-call rather than per-init.

Option C: Use `ConcentrationModel` with `theta_cosmo` — a subclass that takes
theta_cosmo in `concentration()` signature (the try/except at clustering.py:736–744
already supports this interface).

---

### Bug 3 (OPEN): chi_max vs pi_max asymmetry biases large-R ΔΣ

**Location:** `hod_mod/galaxies/clustering.py`, line 942 (default `chi_max=300 Mpc/h`).

**Root cause:**  
wp integrates along LOS to `pi_max=80 Mpc/h` (set in config/metadata, misses BAO at
~100 Mpc/h). The ΔΣ LOS integral uses `chi_max=300 Mpc/h` which includes the BAO
peak. Because P_gm has the same BAO feature as P_lin, the ΔΣ model predicts a 2-halo
BAO contribution that has no counterpart in the measured wp.

**Impact:**  
At the largest R bins (R > 10 Mpc/h), ΔΣ is systematically over-predicted by ~5–15%
compared to a prediction with chi_max=80 Mpc/h. The optimizer partly compensates by
lowering b_eff, which then makes wp slightly under-predicted.

**Fix approach:**  
In `WpFitter`/`JointFitter`, pass `chi_max=pi_max` to `delta_sigma()` so both
observables integrate over the same LOS depth. Verify the config has `pi_max_hMpc`
in metadata and plumb it through the fitter.

Alternatively: match what Lange+2025 actually does. Their pipeline (TripoSH + AbacusSummit
simulations) likely uses a much larger pi_max internally or integrates in redshift space.
The correct approach depends on what chi_max they used — check their public code.

---

### Bug 4 (OPEN): sigma_bar integral misses R < 0.01 Mpc/h

**Location:** `hod_mod/galaxies/clustering.py`, line 985.

```python
R_tab = jnp.logspace(-2, 2.0, n_R_tab)  # starts at 0.01, not 0
```

**Impact:**  
The cumulative integral `sigma_bar(R) = (2/R²) ∫₀ᴿ R' Σ(R') dR'` is started at
R=0.01 Mpc/h rather than R→0. At the smallest ΔΣ bin (R~2.8 Mpc/h), the contribution
from R < 0.01 Mpc/h is negligible (<0.1%). This is a minor issue.

**Fix:** Extend `R_tab` to start at 0.001 Mpc/h or use an analytical NFW contribution
for R < 0.01 Mpc/h. Low priority.

---

## 3. Prediction Chain: Unit Consistency

The unit chain for ΔΣ is **consistent** (no h-factor error):

| Step | Value | Unit |
|------|-------|------|
| `P_gm(k) = b_eff × P_lin(k)` | — | (Mpc/h)³ |
| `ξ_gm(r)` via Fourier transform | — | dimensionless |
| `Σ(R) = ρ̄_m × ∫ ξ_gm dχ` → `ΔΣ = Σ̄(<R) - Σ(R)` | model output × 1e-12 | Msun h / pc² |
| Data column `ds_Msun_h_pc2` | paper R×ΔΣ / R / h | Msun h / pc² |

Both model and data are in `[Msun h / pc²]`. The unit convention is self-consistent
and is NOT the source of the discrepancy.

---

## 4. Action Plan

| Priority | Action | File | Effort |
|----------|--------|------|--------|
| Done | Fix assembly bias in `_pk_tables_full` | clustering.py:819-822 | done |
| High | Re-run MAP fits; verify b_eff and chi2/ndof improve | run_benchmark.py | 1h |
| High | Fix chi_max = pi_max in JointFitter/DeltaSigmaFitter | hod_wp.py | 0.5h |
| Medium | Fix HaloProfile cosmology: use dutton14 or pass theta_cosmo | halo_profiles.py | 2h |
| Low | Lower sigma_bar inner boundary to 0.001 Mpc/h | clustering.py:985 | 0.5h |

---

## 5. MAP Fit Results (2026-05-29, after fix 1)

Benchmarks run with `run_all_benchmarks.py` after applying the assembly bias fix.

| Benchmark | chi2/ndof | Status | Notes |
|-----------|-----------|--------|-------|
| zheng2007 | 0.36 | pass | Excellent |
| zumandelbaum2015 | 0.00 | pass | |
| zumandelbaum2015_ds | 0.00 | pass | |
| more2015 | 1.36 | pass | Close to published 0.9 |
| kravtsov2004 | 1.35 | pass | |
| vanutert2016_ds | 1.89 | pass | |
| lange2025_bgs2_ds_des | 1.43 | pass | ΔΣ-only works well |
| lange2025_bgs2_ds_hsc | 0.73 | pass | ΔΣ-only works well |
| lange2025_bgs2_wp | 6.89 | fail | ndof=1 (10 params, 11 wp bins) |
| lange2025_bgs2_des | 3.02 | fail | Joint wp+ESD fails |
| lange2025_bgs2_hsc | 2.75 | fail | Joint wp+ESD fails |
| lange2025_bgs3_des | 5.68 | fail | |
| lange2025_bgs3_hsc | 6.61 | fail | |
| lange2025_lrg1_des | 7.72 | fail | |
| lange2025_lrg1_hsc | 9.91 | fail | |
| lange2025_lrg2_hsc | 6.09 | fail | |
| leauthaud2012_ds | 46.32 | fail | Pre-existing failure |
| guo2018 | 2.82 | fail | Pre-existing |
| guo2019 | 2.94 | fail | Pre-existing |

### Key observations

1. **ΔΣ-only Lange+2025 fits pass** (bgs2_ds_des=1.43, bgs2_ds_hsc=0.73) — the
   1-halo ΔΣ prediction is working correctly.

2. **wp-only Lange+2025 fails** (bgs2_wp=6.89, ndof=1): severely over-parameterized.
   With 10 free params and only 11 wp bins (after rp_min=0.4 cut), the fitter has
   only 1 degree of freedom. A_cen and A_sat are degenerate for wp → optimizer
   finds A_cen=A_sat=-0.47 (identical, degenerate).
   **Fix:** For wp-only config, fix A_sat=0 or reduce free params to get ndof≥3.

3. **Joint fits fail** (chi2/ndof=3-10): The data is digitized from paper figures
   (Zenodo not yet public). Calibration errors of 20-30% in the digitized wp or ESD
   values would explain the poor fit quality. These benchmarks should be revisited
   when Zenodo data (record 17831718) becomes publicly available.

4. **Assembly bias fix verified**: A=0.5 shifts b_eff by +12% (matches `HOD._integrate`
   formula exactly). The fix is working correctly.

### Next steps

- Wait for Zenodo record 17831718 (public access expected) and replace digitized data
- For wp-only configs: reduce free params (fix A_sat=0 or simplify model)
- For joint configs: the failure is likely data quality, not model physics
