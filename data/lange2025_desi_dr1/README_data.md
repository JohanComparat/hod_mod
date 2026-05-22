# Data: Lange+2025 DESI DR1 tracer analysis (arXiv:2512.15962)

## Source

Paper: Lange et al. 2025, "Galaxy Clustering and Weak Lensing from DESI DR1:
Tracing HOD Parameters and Cosmology with BGS and LRG Samples"
arXiv: https://arxiv.org/abs/2512.15962

Data repository (upon publication):
https://zenodo.org/records/17831718

## Samples

| Sample | z range  | z_eff | n_g [h³ Mpc⁻³]   | wp source | DES ESD source | HSC ESD source |
|--------|----------|-------|-------------------|-----------|----------------|----------------|
| BGS2   | 0.2–0.3  | 0.25  | 3.71±0.03 × 10⁻³  | Fig 3     | Fig 3          | Fig 4          |
| BGS3   | 0.3–0.4  | 0.35  | 1.38±0.01 × 10⁻³  | Fig 3     | Fig 3          | Fig 4          |
| LRG1   | 0.4–0.6  | 0.51  | 5.70±0.03 × 10⁻⁴  | Fig 3     | Fig 3          | Fig 4          |
| LRG2   | 0.6–0.8  | 0.70  | 5.69±0.03 × 10⁻⁴  | Fig 3     | none           | Fig 4          |

BGS n_g from: Hahn+2023, AJ 165, 253 (arXiv:2306.06316)
LRG n_g from: Zhou+2023, AJ 165, 58 (arXiv:2208.08515)

## Observables

- **wp(rp)**: projected correlation function, π_max = 80 h⁻¹ Mpc
  - 14 log-spaced bins, rp ∈ [0.1, 63] h⁻¹ Mpc (full measurement)
  - Analysis scale cut: rp_min = 0.4 h⁻¹ Mpc (11 bins used)
- **ΔΣ(R)**: excess surface density (weak lensing)
  - 13 log-spaced bins, R ∈ [2.5, 40] h⁻¹ Mpc (scale cut already applied)
  - Figure 3: DES Y3 + KiDS-1000 lensing (BGS2, BGS3, LRG1)
  - Figure 4: HSC-Y3 lensing (BGS2, BGS3, LRG1, LRG2)

## Units

- wp: rp in h⁻¹ Mpc, wp in h⁻¹ Mpc
- ESD: R in h⁻¹ Mpc, ΔΣ in M_⊙ h pc⁻² (paper reports M_⊙ pc⁻²; multiply by h=0.6736)

## Status

**NEEDS_DATA**: The CSV files contain correct bin positions but PLACEHOLDER values (0.0).
Fill in actual measurements from the Zenodo repository when available, or digitize
from Figures 3 and 4 of the paper.

## Column formats

wp CSV:  `rp_hMpc, wp_hMpc, wp_err_hMpc`
ESD CSV: `R_hMpc, ds_Msun_h_pc2, ds_err_Msun_h_pc2`
