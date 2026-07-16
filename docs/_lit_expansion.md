# Literature-expansion log — second iteration

Audit trail for the benchmark-observables expansion. Two passes: (A) close the
gap between the papers already cited on `sensitivity_benchmark.rst` and the JSON
tree; (B) a fresh per-observable literature search for the current best
published measurement not yet cited. Every added entry is a `placeholder` flagged
`needs_operator_extraction` (the operator digitizes/downloads the table by hand),
except Heydenreich2025 which is a public release ingestible as `observed` once
fetched.

## Pass A — page↔tree gap (10 entries)

These references were already cited on the page but had no JSON file. Added as
flagged placeholders with an extraction hint naming the exact table/figure.

| Ref | Observable | Why it belongs |
|---|---|---|
| Zehavi2011 | `wp_dr7` | SDSS DR7 w_p per luminosity/colour sample — the page's headline low-z clustering benchmark (tree only had the digitized Zehavi2005 point). |
| Mandelbaum2006 | `ds` | SDSS g-g lensing ΔΣ per L/M* bin — foundational lensing benchmark. |
| Leauthaud2017 | `ds` | CMASS "lensing is low" — the wp-vs-lensing tension test. |
| Heydenreich2025 | `ds_wp` | Lensing Without Borders: DESI-DR1×DES/KiDS/HSC public ΔΣ+w_p release. |
| DESIDR2 | `bao` | DESI DR2 BAO distances — geometry block for (w0, wa, h). |
| DESI2024FS | `fs_multipoles` | DESI DR1 full-shape multipoles — growth/RSD block. |
| Macquart2020 | `frb_dm` | FRB DM–z (Macquart relation) — absolute ionised-baryon census. |
| Ponomareva2023 | `himf_zgt0` | MIGHTEE-HI HIMF beyond z≈0 — extends local ALFALFA in redshift. |
| Muzzin2013 | `smf` | UltraVISTA total+quiescent SMF 0.2<z<4 — z-evolution anchor. |
| Greene2020 | `mbh_lowmass` | low-mass M_BH pin — companion to KormendyHo2013. |

## Pass B — fresh literature search (4 references → 3 entries)

One bounded pass over the observable list of the "Benchmark measurements per
model observable" table, targeting observables that carried only a simulated or
placeholder stand-in and where a current published measurement was *not yet
cited*. Search queries and choices:

| Observable | Query | Chosen paper | arXiv | Rationale |
|---|---|---|---|---|
| `cl_kk` (shear) | "DES Y3 cosmic shear Amon Secco 2022 cosmological constraints" | Amon 2022 + Secco 2022, *Phys. Rev. D* 105 023514/023515 | 2105.13543 / 2105.13544 | A second wide-area shear anchor beside KiDS-Legacy; S8≈0.76 at 2–3%, ~2.3σ below Planck — directly tests the shear row's robustness. |
| `cl_gy` (galaxy×tSZ) | (already in `references.rst`) | Pandey 2025, DES Y3 × ACT DR6 lensing×tSZ, 21σ | 2506.07432 | A current lensing×tSZ *cross-spectrum* benchmark beyond the ACT×CMASS stacked profiles [Amodeo2021] — constrains group–cluster feedback. |
| `cl_gI` / `wp_agn` (IR AGN) | (already in `references.rst`) | Petter 2023, WISE obscured/unobscured QSO host halos | 2302.00690 | The page states "WISE crosses measurable today"; this is the real WISE AGN clustering measurement, tied to the shared obscuration parameter. |

Pandey2025 and Petter2023 already existed in `references.rst` (uncited on this
page) — the search identified them as the right current measurements and they are
now both cited on the page and materialised in the tree. Amon2022/Secco2022 are
new to `references.rst`.

Observables deliberately *not* expanded (current cited source is already the
state of the art): `n_gal`/`smf` (COSMOS2020 Weaver2023 + GAMA Driver2022),
`ssfr` (Popesso2023), `sfrd` (Madau–Dickinson), `xlf` (Aird2015), `wp_agn`
(eRASS1 Comparat2023), `rlf` (LoTSS Kondapally2022/Bonato2021), `qlf_*`
(Kulkarni2019), `uvlf`/`half`/`oiilf` (Wyder2005/Finkelstein2015/Harikane2023,
Sobral2013, Comparat2015OII), `ncl` (eRASS1 Ghirardini2024), `cl_kCMB`/`cl_gkCMB`
(ACT DR6 Qu2024/Kim2024), `cl_gHI`/`cl_HIHI` (CHIME).

## Net effect

Tree grows 83 → 96 files (all 13 new entries flagged for operator extraction).
Updated counts propagate to `sensitivity_benchmark.rst` (§ *The benchmark data
tree*), the tree `README.md`, and `DIGITIZATION_WORKLIST.md`.
