# Data Extraction Instructions: van Uitert+2016

## Source paper
van Uitert et al. 2016, MNRAS 459, 3251 — arXiv:1601.06791

## IMPORTANT: Corrections to original instructions

**Table B2 does not exist.** Appendix B contains sensitivity tests (B1–B4),
not data tables. The DeltaSigma measurements are shown only in **Figure 2**.

**wp(rp) is not measured.** This paper fits only the galaxy–galaxy lensing
signal (DeltaSigma) plus the stellar mass function. No projected clustering is
measured. For GAMA wp(rp), consider Farrow et al. 2015 (arXiv:1508.05468).

## DeltaSigma data: Figure 2

DeltaSigma(R) is shown in **Figure 2** for 8 stellar mass bins.
Their stellar mass is in **h^-2 Msun** (not Msun).

### Mass bin closest to user's bin 2 (10.4 < log10(M*/Msun) < 10.8)
With h = 0.673 (Planck 2013):

| Paper bin | h^-2 Msun range | Msun range | N_lens | z_eff |
|-----------|-----------------|------------|--------|-------|
| M2 | 9.89–10.24 | 10.23–10.58 Msun | 19175 | 0.21 |
| M3 | 10.24–10.59 | 10.58–10.93 Msun | 24459 | 0.25 |

The benchmark config uses **panel M3** as the closest to the upper half of
the user's bin (10.4–10.8 Msun). The current CSV has been populated with
digitized M3 data.

### Units (from figure and paper)
- x-axis: **comoving R [h^-1 Mpc]**
- y-axis: **ΔΣ [h M_sun pc^-2]**

These are already in hod_mod convention — no conversion needed.

## Digitized DS data (STATUS: DIGITIZED_FROM_FIGURE, precision ~3%)
The `ds_bin2_104_108.csv` file is populated with values digitized from the
PostScript vector source of Figure 2, panel M3. Precision: ~2–3% in ΔΣ,
~5% in σ_ΔΣ. Cross-check 3 values against the published Figure 2.

## Triple-check protocol
1. R values span ~0.03–2 h^-1 Mpc (10 bins).
2. DeltaSigma peaks at ~60–70 h M_sun pc^-2 at the smallest R.
3. Values should monotonically decrease (approximately) with increasing R.
4. The third data point (R ≈ 0.08 h^-1 Mpc) may have large noise.
5. Confirm sign: ΔΣ > 0 throughout.

## wp: NOT AVAILABLE
The wp_bin2_104_108.csv file documents that wp(rp) is not measured in this
paper. Only DeltaSigma is available for this benchmark.

## Galaxy number density estimate
For M3 (Nlens = 24459, survey area ~75 deg^2, z = 0.05–0.35):
  n_g ≈ 1.1 × 10^-3 h^3 Mpc^-3 (rough estimate; not quoted in paper)
