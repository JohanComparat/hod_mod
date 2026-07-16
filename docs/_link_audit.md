# Link audit — `sensitivity_benchmark` page + benchmark-observables generator

Second-iteration verification of every citation link reachable from
`docs/sensitivity_benchmark.rst` and every `arxiv` URL emitted into the
`benchmark_observables` JSON tree by
`hod_mod/scripts/data/make_benchmark_observables.py::REFS`.

Method: the page cites 46 `[Key]_` references (all resolve in
`references.rst`); all `:doc:`/`:ref:`/internal-label cross-refs resolve
(`bench-today`, `bench-missing`, `bench-data-tree`, `data_formats`, the sibling
forecast docs). Each unique arXiv/DOI for the page-cited set and for the
divergent generator entries was fetched and its title/first-author/year checked
against the citation. Pre-2020 canonical IDs (NFW1997, EisensteinHu1998,
Planck2018, KormendyHo2013, MadauDickinson2014, Leauthaud2012, Wyder2005,
Sobral2013, Comparat2015OII, Aird2015, Kulkarni2019, Jones2018ALFALFA,
Moustakas2013, Muzzin2013, Behroozi2019, Wright2010, Dore2014, Bacon2018,
SKA2019, Kollmeier2017, Macquart2020, Greene2020, Finkelstein2015, Song2016,
Harikane2023, Weaver2023, Driver2022, Popesso2023, Ananna2022, Powell2022,
CHIME2023) were confirmed by recognised arXiv identifier.

## Bugs found and fixed

| Where | Key | Was | Now | Note |
|---|---|---|---|---|
| `references.rst` | `Amodeo2021` | `2009.05557` | `2009.05558` | `…557` is **Schaan** (kSZ); Amodeo (gas thermo, PRD 103 063514) is `…558`. Verified. |
| `references.rst` | `Zehavi2005` | *(missing)* | `astro-ph/0408569` | JSON tree cites `Zehavi2005` (digitized Fig. 8 `M_r<-21`); entry added next to `Zheng2007`. Verified. |
| generator `REFS` | `Guo2018` | `1803.07697` | `1804.01993` | `…7697` is a **condensed-matter** paper. `…1993` = Guo 2018 ApJ 858 30 (DOI 10.3847/1538-4357/aabb0e). Verified + folder metadata. |
| generator `REFS` | `Lange2025` | `2502.10230` | `2512.15962` | Authoritative per `data/lange2025_desi_dr1/*/metadata.json`; DESI-DR1 full-scale clustering+lensing. Verified. |
| generator `REFS` | `Zacharegkas2025` | `2106.08438` | `2506.22367` | `…08438` is Zacharegkas **2022** (MNRAS 509 3119). `…22367` = Zacharegkas 2025 DES-Y3 SHMR. Verified + folder metadata. |
| `sensitivity_benchmark.rst` | `wp` row | inline `arXiv:2512.15962` | `[Lange2025]_` | Redundant raw link collapsed to the citation (same paper). |

The generator's `Guo2018`/`Lange2025`/`Zacharegkas2025` JSON outputs already
carried the correct URLs (folder `metadata.json` overrides the `REFS` default via
`_ref_from_meta`); the fix corrects the hard-coded fallback so a metadata-less
ingest can never emit the wrong link.

## Web-confirmed page-cited references (spot list)

Amodeo2021 `2009.05558`; Schaan2021 `2009.05557`; Comparat2023 `2301.01388`
(eFEDS AGN, A&A 673 A122 — `references.rst` uses the equivalent ADS link);
Comparat2025 `2503.19796`; Zehavi2011 `1005.2413`; Mandelbaum2006
`astro-ph/0605476`; Wright2025 `2503.19441`; Qu2024 `2304.05202`; Kim2024
`2407.04606`; Ghirardini2024 `2402.08458`; DESIDR2 `2503.14738`; DESI2024FS
`2411.12022`; Heydenreich2025 `2506.21677`; Ponomareva2023 `2304.13051`;
Kondapally2022 `2204.07588`; CHIMEauto2025 `2511.19620`. All correct.

## Cross-reference integrity

`grep`-level check: every `[Key]_` on the page has a `.. [Key]` definition in
`references.rst` (46/46); every `:doc:`/`:ref:` target exists. Re-run after edits
with `make -C docs html` and scan for `WARNING: undefined label` /
`unknown document`.
