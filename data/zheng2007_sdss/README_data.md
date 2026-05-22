# Data Extraction Instructions: Zheng+2007 / Zehavi+2005

## Source paper
Zehavi et al. 2005, ApJ 630, 1 — arXiv:astro-ph/0408564

## What to extract
**Table 1** of Zehavi+2005: projected correlation function wp(rp) for SDSS galaxies.

Extract the **M_r < −21** luminosity-threshold sample:
- Column 1: rp [h^-1 Mpc]
- Column labelled "M_r < -21": wp [h^-1 Mpc]
- Corresponding error column (jackknife σ)

The table has ~12 rp bins spanning rp ≈ 0.1 to 20 h^-1 Mpc.

## Triple-check protocol
1. Count rows: should be ~12 bins.
2. Verify rp is monotonically increasing.
3. Verify wp is positive and decreasing.
4. Confirm units are h^-1 Mpc (stated in paper caption).
5. Cross-check two or three values against Figure 7 of Zheng+2007
   (the M_r < -21 panel should overlay the same points).

## CSV format
```
rp_hMpc,wp_hMpc,wp_err_hMpc
0.101,XXXX,XXX
...
```

## Cosmology note
Zehavi+2005 presents measurements in their cosmology (h=0.7, flat ΛCDM).
Zheng+2007 uses WMAP3: Ωm=0.238, h=0.732, σ8=0.740, ns=0.958.
Both papers work in h^-1 Mpc units throughout, so no unit conversion is needed.
