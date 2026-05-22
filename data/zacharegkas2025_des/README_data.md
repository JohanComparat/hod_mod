# Data Extraction Instructions: Zacharegkas+2025

## Source paper
Zacharegkas et al. 2025 — arXiv:2506.22367

## IMPORTANT: Observable mismatch

**This paper does NOT measure wp(rp) or ΔΣ(R).**

The observables used are:
- **Angular clustering**: w(θ) — angular correlation function
- **Weak lensing**: γ_t(θ) — tangential shear (angular)

Both are measured in 30 logarithmic angular bins spanning [0.25, 250] arcmin.
The modeling is done in angular/harmonic space via Limber-approximated C_ℓ.

The hod_mod benchmarks use projected physical-space statistics (wp and ΔΣ).
To use this paper, one would need to either:
1. Convert γ_t(θ) → ΔΣ(R) using the lens n(z) and a cosmology
2. Convert w(θ) → wp(rp) using the lens n(z)
Neither conversion is done in the paper.

## DES Y3 Stellar Mass Bin 1 (lens bin ℓ=1)

From Table 1 of Zacharegkas+2025:
- Redshift range: z = (0.20, 0.40)
- Stellar mass range: log10(M_*/Msun) = (9.56, 9.98)
- N_gal = 3.00 × 10^6
- n_gal = 0.218 arcmin^-2
- Derived n_g ≈ 7 × 10^-3 h^3 Mpc^-3 (from N_gal / comoving volume)
- HOD results: log10(M_h^cen) = 11.430 (+0.065/-0.064), F_sat = 0.053 ± 0.005

## Cosmology
DES Y3 3x2pt best-fit: Omega_m=0.339, h=0.677, sigma8=0.760, ns=0.968.

## STATUS: NOT_APPLICABLE_FOR_PROJECTED_BENCHMARKS
This paper cannot directly provide wp(rp) or ΔΣ(R) data for the projected
benchmarks. The benchmark config benchmark_zacharegkas2025.yml and the
DS-only variant benchmark_zacharegkas2025_ds.yml are retained as stubs
for potential future work converting angular to projected statistics.
