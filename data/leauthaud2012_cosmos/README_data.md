# Data Extraction Instructions: Leauthaud+2012

## Source paper
Leauthaud et al. 2012, ApJ 744, 159 — arXiv:1104.0928

## IMPORTANT: Corrections to original instructions

**Tables A1/A2/A3 do not exist.** The paper contains only Tables 1–5.
No machine-readable tables were deposited at VizieR (J/ApJ/744/159 not found).

**wp(rp) is not measured.** The paper uses w(θ) (angular correlation function),
not the projected wp(rp). To obtain wp, one would need to convert w(θ) via
the lens N(z) — this is non-trivial and not done in the paper.

## DeltaSigma data: Figure 6

DeltaSigma(R) is shown in **Figure 6** of Leauthaud+2012, for the PHOTO_z2 bin
(z ∈ [0.48, 0.74]) in **stellar mass bins** (panels g–m), NOT threshold samples.

The closest bin to log10(M*/Msun) > 10.6 is:
- Panel j: 10.65 < log10(M*/Msun) < 10.88  (use this)
- Panel k: 10.30 < log10(M*/Msun) < 10.65  (alternative)

### Units (from figure axis labels)
- x-axis: **physical transverse R [Mpc]** (not h^-1 Mpc, not comoving)
- y-axis: **ΔΣ [M_sun pc^-2]** (NOT h M_sun pc^-2)

### Conversion to hod_mod units
- R [h^-1 Mpc] = R [Mpc] × h = R [Mpc] × 0.72
- ds [M_sun h pc^-2] = ds [M_sun pc^-2] × h = ds [M_sun pc^-2] × 0.72

### Approximate R bins (physical Mpc, from Figure 6)
R spans roughly 0.02–20 Mpc (physical), logarithmically spaced.
After conversion: R [h^-1 Mpc] ≈ 0.014–14.4 h^-1 Mpc.

## wp data
Not available in this paper. The angular correlation w(θ) for PHOTO_z2,
M* > 10.3 threshold is shown in Figure 6, panels c–e.

**Alternative**: Use Section 5.3 of Leauthaud+2012 and their lensing-only
parameter recovery to validate the DeltaSigma-only benchmark.

## Triple-check protocol (DeltaSigma from digitized Figure 6, panel j)
1. R values in Figure 6j span ~0.02–20 physical Mpc (i.e., ~0.014–14.4 h^-1 Mpc).
2. DeltaSigma should peak at small R (~50–200 M_sun pc^-2) and decrease outward.
3. Confirm positive values throughout; cross-check 3 points vs. Figure 6.
4. After unit conversion, ds at R~0.1 h^-1 Mpc should be ~30–150 M_sun h pc^-2.

## CSV format (DeltaSigma) — after digitization and unit conversion
```
R_hMpc,ds_Msun_h_pc2,ds_err_Msun_h_pc2
0.014,XXX,XX
...
```

## STATUS: NEEDS_DATA
Data must be digitized from Figure 6, panel j of arXiv:1104.0928.
Use a digitization tool (e.g., WebPlotDigitizer) on the published PDF.
