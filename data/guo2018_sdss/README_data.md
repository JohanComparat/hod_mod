# Data Extraction Instructions: Guo+2018

## Source paper
Guo et al. 2018, ApJ 858, 30 — arXiv:1804.01993

## What to extract
**Figure 3** of Guo+2018: projected correlation function wp(rp) for their
galaxy sample.  Check if a machine-readable table is available in the paper
appendix or ADS supplementary material.

Identify the stellar mass bin corresponding to their primary analysis
(log10 M_* > 10.0 or their specific luminosity/mass cut stated in Section 2).

## Columns
- rp [h^-1 Mpc]
- wp [h^-1 Mpc]
- sigma_wp (1-sigma error, from jackknife or covariance diagonal)

## Cosmology
Planck 2015 XIII: Omega_m=0.307, h=0.677, sigma8=0.829, ns=0.961.

## Triple-check
1. rp range matches Figure 3 x-axis.
2. wp values at ~1 and ~10 Mpc/h match Figure 3 data points.
3. Units are h^-1 Mpc (standard for SDSS analyses).
