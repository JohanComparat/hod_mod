# Data Extraction Instructions: Zu & Mandelbaum 2015

## Source paper
Zu & Mandelbaum 2015, MNRAS 454, 1161 (Paper I) — arXiv:1505.02781

## wp data
The wp measurements are from their own analysis of SDSS DR7.
Check Section 2 and Figure 2 of ZM15 for the log10(M_*) > 10.2 threshold sample.
If a machine-readable data table is available in the supplementary material
or a companion data release, use that.  Otherwise digitize from Figure 2.

## DeltaSigma data
The weak-lensing signal is from Mandelbaum et al. 2006 (MNRAS 368, 715;
arXiv:astro-ph/0509702), re-analysed for the same stellar mass threshold.
Check ZM15 Figure 2 right panels or their data release.

## Cosmology note
ZM15 use WMAP7: Omega_m=0.26, h=0.72, sigma8=0.80, ns=0.96 (from their Section 2).

## Triple-check protocol
1. wp: should span rp ~ 0.05 to 30 h^-1 Mpc, ~8-15 bins.
2. DeltaSigma: R ~ 0.05 to 20 h^-1 Mpc, ~8-10 bins.
3. wp should be ~1000 h^-1 Mpc at 0.1 Mpc/h, ~10 at 10 Mpc/h for this threshold.
4. Cross-check against Figure 2, log M_* > 10.2 panel.
5. Check that pi_max = 60 h^-1 Mpc (stated in their analysis section).

## CSV format
Same as other benchmarks — see README_data.md in more2015_boss_cmass/ for reference.
