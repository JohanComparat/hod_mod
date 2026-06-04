# Changelog

All notable changes to this project will be documented in this file.

## [0.0.2] — 2026-06-01

### Added

- Benchmark configs for More+2015 stellar-mass subsamples: `benchmark_more2015_logM11_12.yml`,
  `benchmark_more2015_logM11p3_12.yml`, `benchmark_more2015_logM11p4_12.yml`,
  and the free-cosmology variant `benchmark_more2015_logM11_12_freecosmo.yml`.
- Digitized joint wp+ESD data for More+2015 subsamples A/B/C:
  `data/more2015_boss_cmass/logM11_12/`, `logM11p3_12/`, `logM11p4_12/`.
- Benchmark configs for Lange+2025 DESI DR1 bwpd series (12 configs):
  BGS2, BGS3, LRG1, LRG2 × wp-only / ESD-only (HSC) / joint wp+ESD (HSC).
- Manually digitized (WebPlotDigitizer) data for Lange+2025 in bwpd format:
  `wp_*_bwpd.csv` and `ds_hsc_*_bwpd.csv` for all four samples.
- Documentation for More+2015 and Lange+2025 benchmarks.
- Dedicated digitization scripts and raw figure archives for both datasets.

### Changed

- Documentation: benchmark navigation reorganised — only More+2015 and Lange+2025
  are linked in the main toctree; other benchmarks exist but are not yet reachable
  from the navigation until their data and fits are validated.
- Analysis of LS10/BGS pages removed from the documentation navigation.
- Lange+2025 benchmark model keys updated to `bwpd` naming convention
  (`lange2025_bgs2_bwpd_hsc`, etc.) to reflect the new manually-digitized dataset.

## [1.0.0] — 2025-05-21

### Added

- Initial release of `hod_mod`.
- `HODBase` abstract base class: all 9 clustering HOD classes now inherit from it.
  Implements `_integrate()`, `galaxy_number_density()`, `effective_bias()`, and
  `effective_mass()` once, delegating to the `nc_ns()` extension point.  Saves
  ~1 500 lines of copy-pasted boilerplate.
- `_SINGLE_ARG_INIT` class flag on `HODBase` subclasses replaces the
  `_HOD_SINGLE_ARG` string set that was used in the original fitter.
- Unified `FitConfig` dataclass replaces the three separate config classes
  (`WpFitConfig`, `JointFitConfig`, `WpFitFITSConfig`).
- Single `load_config()` function auto-detects `joint` and `fits` YAML sections.
- `WpFitter` / `JointFitter` fitter hierarchy; `WpFitterFITS` is now an alias.
- `Kravtsov04HODModel` added to the `HOD_MODELS` dispatch dict.
- Backward-compatibility aliases: `AUMHODModel = Kravtsov04HODModel`,
  `WpFitConfig = JointFitConfig = WpFitFITSConfig = FitConfig`,
  `load_joint_config = load_fits_config = load_config`.

### Breaking changes

- All imports must use `from hod_mod.*` (the previous package name is no longer supported).
- `WpFitterFITS` is now identical to `WpFitter`; the FITS-specific class was
  merged into the base fitter via `_load_data()` dispatch.
