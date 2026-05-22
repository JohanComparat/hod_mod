# Changelog

All notable changes to this project will be documented in this file.

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
