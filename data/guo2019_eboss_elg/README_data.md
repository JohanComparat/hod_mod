# Data Extraction Instructions: Guo+2019

## Source paper
Guo et al. 2019, ApJ 871, 147 — arXiv:1811.10583

## What to extract
wp(rp) for the eBOSS ELG sample at z ~ 0.8.
Check Figure 3 and any supplementary tables.  The eBOSS ELG data may also
be available from the official eBOSS data release at
https://www.sdss.org/surveys/eboss/ (DR16 value-added catalogs).

## Columns
- rp [h^-1 Mpc]
- wp [h^-1 Mpc]
- sigma_wp (1-sigma error from jackknife or bootstrap)

## Cosmology
Planck 2015 XIII: Omega_m=0.307, h=0.677, sigma8=0.829, ns=0.961.

## Triple-check
1. z_eff = 0.8 confirmed from paper abstract.
2. wp at ~1 Mpc/h should be ~100-300 h^-1 Mpc (ELGs are weakly clustered).
3. Units: h^-1 Mpc.
4. Scale range: rp ~ 0.1 to 30 h^-1 Mpc.
