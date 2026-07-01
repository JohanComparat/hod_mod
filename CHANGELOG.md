# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2] — 2026-07-01

Documentation now mirrors the refactored repository; all links verified.

- ``docs/scripts.rst`` rewritten to the current ``scripts/`` layout — removed
  ~15 references to scripts deleted in the refactor (demos, ``run_pipeline``/
  ``run_inference``, ``utils/``, ``gama/``/``cosmos/`` fits).
- Replaced all remaining hardcoded paths in the docs with the ``$HOD_MOD_*``
  env-var forms; fixed ``paper_reproductions/more2015_boss_cmass.py`` to read the
  real ``configs/hod_fit_more2015_cmass.yml``.
- Verified **all links** with a Sphinx build: fixed a broken ``../_images`` figure
  ref, a missing ``bgs_ls10_wp_survey`` label, a toctree entry to an excluded page,
  5 dangling ``:ref:`` cross-references, and a dead external repo link; every other
  internal cross-reference, figure, and external URL resolves.

## [0.1.1] — 2026-06-30

Repository hygiene and reproducible paths. No public-API symbol changes.

### Data & results moved out of the repo
- Curated benchmark data + results are archived on **Zenodo**
  (concept DOI ``10.5281/zenodo.21078473``) and fetched on demand with checksum
  verification via ``hod_mod.data_io.fetch`` (``pooch``).
- Generated outputs now write **outside** the repo via
  ``hod_mod.paths.results_root`` (``$HOD_MOD_RESULTS``); ~30 scripts updated.
- ``results/`` purged from git history; ``.git`` shrank 309 MB → ~31 MB.

### No hardcoded paths
- All filesystem locations resolve through ``hod_mod.paths`` helpers with env-var
  overrides: ``repo_root()`` (``$HOD_MOD_REPO``), ``data_root()``
  (``$HOD_MOD_DATA_DIR``), ``sum_stat_root()`` (``$HOD_MOD_SUMSTAT``),
  ``results_root()`` (``$HOD_MOD_RESULTS``), ``cache_root()`` (``$HOD_MOD_CACHE``).
- Removed every hardcoded ``/home/comparat`` / ``~/data`` / ``~/software/sum_stat``
  path from executable code.

### Misc
- Documentation figures moved to ``docs/_images/``; README updated to the
  refactored module paths + an environment-variable setup section.
- Removed Guix install support; added a ``pre-commit`` guard against committing
  large files or ``results/``.

## [0.1.0] — 2026-06-30

A structural refactor that reorganises the package **by observable pipeline**
(galaxy clustering + lensing, galaxy × X-ray, galaxy × thermal SZ) on top of a shared
core, instead of by ingredient type (`cosmology/` vs `galaxies/`). This is a
**clean break**: internal import paths change and there are no compatibility shims.
The top-level public API (symbol names such as `MoreHODModel`,
`FullHaloModelPrediction`, `FitConfig`) is preserved via re-exports from
``hod_mod/__init__.py``.

### Breaking changes — module move map

| Old import path | New import path |
|---|---|
| `hod_mod.cosmology.*` | `hod_mod.core.*` |
| `hod_mod.cosmology.gas_profiles` | `hod_mod.gas.{pressure,density,cooling,metallicity,conversions}` |
| `hod_mod.cosmology.erosita_response` | `hod_mod.gas.erosita_response` |
| `hod_mod.galaxies.hod` | `hod_mod.connection.hod.{base,more15,zumandelbaum15,…}` |
| `hod_mod.galaxies.{clf,sham}` | `hod_mod.connection.{clf,sham}` |
| `hod_mod.galaxies.agn*` | `hod_mod.agn.{xray,ham,hod,duty_cycle}` |
| `hod_mod.galaxies.clustering` | `hod_mod.observables.clustering` |
| `hod_mod.galaxies.cross_spectra` | `hod_mod.observables.cross_spectra` |
| `hod_mod.galaxies.cross_clustering` | `hod_mod.observables.cross_clustering` |
| `hod_mod.galaxies.intrinsic_alignment` | `hod_mod.observables.intrinsic_alignment` |
| `hod_mod.galaxies.baryon_fraction` | `hod_mod.observables.baryon_fraction` |
| `hod_mod.fitting.hod_wp` | `hod_mod.fitting.{config,models,fitters}` |

### Added

- `hod_mod/observables/` — the thin top layer mirroring the three observable
  pipelines; `cross_spectra` is the shared galaxy × tSZ and galaxy × X-ray engine.
- `hod_mod/cli/` — a single consolidated CLI front door (`python -m hod_mod <cmd>`
  and the `hod-mod` console entry point) whose subcommands (`fit`, `fit-cross`,
  `fit-joint`, `benchmark`, `predict`, `validate <target>`) delegate to the existing
  scripts. `hod-mod fit` is the recommended config-driven fitter, superseding the
  near-duplicate `fit_hod_wp` / `run_fit` / `run_fit_More15` drivers (which remain
  runnable). The ~50 scripts were not physically relocated.
- Galaxy × thermal SZ promoted to a first-class, documented pipeline (`pipeline_gal_tsz`
  doc page + worked example) built on the existing, already-tested
  `HaloModelCrossSpectra` (`P_{g,y}`, `projected_gy`, `angular_cl_gy`).

### Changed

- Three oversized modules were split along their natural class/function boundaries:
  `hod.py` (2321 lines → `connection/hod/` family package: `base`, `more15`,
  `zumandelbaum15`, `guo`, `kravtsov04`, `zacharegkas25`, `vanuitert16`,
  `leauthaud12`, `lange25`), `fitting/hod_wp.py` → `fitting/{models,config,fitters}.py`,
  and `gas_profiles.py` → `gas/{conversions,pressure,density,cooling,metallicity}.py`.
  `cross_spectra.py` is kept whole as the shared cross-correlation engine.
  `observables/clustering.py` is **deliberately not split** in this release: it is the
  critical wp/ΔΣ prediction path (with the assembly-bias fix and numpy static caches),
  it can only be regression-verified through CAMB, and it is already cleanly isolated
  behind `hod_mod.observables`.
- `m200_to_m500c` (NFW M₂₀₀→M₅₀₀c) re-implemented as a vectorised, jittable JAX
  bisection, replacing the per-halo `scipy.optimize.brentq` Python loop (matches the
  former result to 2e-7). The differentiable forward model (HOD occupation, distances,
  power spectrum, halo-profile FTs) is already pure-JAX. The MAP optimiser keeps
  `scipy.optimize` (gradient-free Powell/Nelder-Mead): its objective runs through CAMB
  and the numpy MCMC caches and is not differentiable end-to-end, so a jaxopt/optax
  swap would require a CAMB-free differentiable forward model (out of scope here).
- Documentation toctree reorganised to mirror the package: User Guide → Pipelines
  (Clustering & Lensing, Galaxy × X-ray, Galaxy × tSZ) → Benchmarks → API Reference.

### Fixed (galaxy × X-ray angular spectra)

Exposed while raising test coverage of `observables/cross_spectra.py`:

- **Threaded-JAX segfault.** `angular_cl_gX` / `angular_cl_XX` defaulted to
  `n_workers=-1`, dispatching the per-redshift `_pk_tables_gX` build across
  `os.cpu_count()` Python threads. Concurrent JAX *compilation* from threads crashes
  the interpreter. Now **serial by default** (`n_workers=1`); the opt-in threaded path
  (`n_workers>1`) does a serial warm-up compile first.
- **float32 NaN.** The `_safe_log` floor `1e-60` (and the XX block's explicit `1e-120`)
  underflows float32 to 0, so an all-zero field (e.g. the AGN leg when no AGN model is
  configured) gives `log(0)=-inf`; a constant `-inf` table then makes `jnp.interp`
  compute `(-inf)-(-inf)=NaN`, poisoning the whole Limber integral. Floor raised to a
  float32-safe `1e-30`. `angular_cl_gX` now returns finite, positive spectra.

### Tests and documentation

- Test suite updated to the new layout (all imports rewritten) and extended with new
  modules covering both the refactored code and previously-untested integration paths:
  `test_public_api` (clean-break contract), `test_cli`, `test_jax_conversions`,
  `test_config_loading` (joint/ds/fits/cosmology branches + esd reader),
  `test_power_spectrum_extra` (EH no-wiggle), `test_refactor_coverage` (baryon-fraction
  models, Lange+2025 assembly bias, eROSITA `ecf_*`, `ApecCoolingTable`, bwpd reader,
  `python -m hod_mod`), and CAMB-heavy `slow` suites: `test_fitter_integration`
  (`WpFitter.map_fit`/`sample`, `DeltaSigmaFitter`, `JointFitter`),
  `test_clustering_prediction` (ΔΣ split/components, baryon, einasto, `n_gal`),
  `test_agn_ham` (HAM abundance matching, both XLFs), and the `cross_spectra` X-ray
  angular spectra (serial==threaded regression). Heavy tests are marked `@pytest.mark.slow`
  so plain `pytest` stays fast; CI runs `pytest -m ""`.
- Fixed a pre-existing test bug (`test_emissivity_uk_scaled_by_boost`: its `_THETA`
  lacked `Omega_b`/`n_s`).
- Full-suite coverage rose from ~77% to **85%** (852 tests pass). Notable per-module
  gains: `gas/erosita_response` 22→83%, `gas/cooling` 45→94%, `agn/ham` 41→88%,
  `fitting/config` 58→98%, `observables/cross_spectra` 64→87%, `fitting/fitters`
  57→74%, with `connection/hod/lange25`, `observables/baryon_fraction` and
  `cross_clustering` at 100%. A `fail_under = 82` floor was added to
  `[tool.coverage.report]` to prevent silent regression.
- Documentation revised to match the refactor: the architecture tree, prose
  file-paths, and code examples now reference the new packages (all 24 example
  imports execute). All 154 documentation links were HTTP-verified; **three wrong
  references were corrected** (Ueda+2014 `1402.7902`→`1402.1836`; Zu & Mandelbaum 2015
  `1407.8741`→`1505.02781`; and the Ogata J₀ DOI `10.1145/1141885.1141895`, which
  actually resolved to an unrelated linear-algebra paper, → the real PRIMS DOI
  `10.2977/prims/1145474602`).

### Housekeeping

- Added `hod_mod.data.erosita` (`*.npz`) to wheel `package-data` (the DR1 response
  and ECF tables were previously unpackaged).
- Removed the one-shot scratch script `_refactor_hod.py` and the empty
  `data/to_deprecate/` directory.
- `.gitignore`: the stray `hod_mod/results/` output tree, the 365 MB untracked
  `apec_v*.fits` tables (downloaded on demand by `soxs`), and the optional vendored
  `WHM/` CAMB fork are now ignored.

## [0.0.5] — 2026-06-24

### Added

- Zu & Mandelbaum 2015 (iHOD) benchmark suite, consolidated for release:
  - Digitized SDSS DR7 data from ZM15 Figure 6 added to the repository
    (`data/zumandelbaum2015_sdss/`): 7 stellar-mass-binned `w_p(r_p)` files
    (9.4–12.0), 5 binned `\Delta\Sigma(R)` files (10.2–12.0), and the raw
    WebPlotDigitizer `Fig6_*.txt` source files, with updated `metadata.json`.
  - Regenerated the model-anchored threshold-sample data vectors
    (`wp_thresh_mstar102.csv`, `ds_thresh_mstar102.csv`).
  - MAP benchmarks (threshold, ΔΣ-only, 7 stellar-mass bins, and joint) refreshed
    via `run_benchmark.py`; figures and result JSON regenerated.
  - Benchmark documentation linked into the navigation: `benchmark_zumandelbaum2015`
    and `benchmark_zumandelbaum2015_multisample` added to the `docs/index.rst`
    Benchmark toctree and summarised in `docs/benchmarks.rst`.
- Conda-free installation with GNU Guix. New files at the repository root:
  `manifest.scm` (hermetic Python + C/Fortran toolchain + runtime libs for
  manylinux wheels), `channels.scm` (pinned Guix revision for reproducible
  `guix time-machine` builds), `requirements-guix.txt` (the validated, pinned
  Python dependency set — `camb==1.4.0`, `numpy==2.4.6`, …), and `INSTALL_GUIX.md`
  (step-by-step procedure and binary-install prerequisites).
- Installation instructions for Guix in `README.md` and `docs/overview.rst`.

### Notes

- The Guix workflow runs `pip` inside a `guix shell --container --network` and
  points `LD_LIBRARY_PATH` at `$GUIX_ENVIRONMENT/lib`, so PyPI wheels (numpy,
  scipy, h5py, jax/jaxlib, camb, …) load against the Guix interpreter without
  relying on Guix's own Python packages or clobbering glibc.
- Use `guix time-machine -C channels.scm` so the environment pins to Python 3.11:
  `camb==1.4.0` (validated, source-only) needs ≤ 3.11, and an unpinned newer camb
  on Python 3.12 fails under numpy ≥ 2.4 (``camb/model.py:691`` TypeError).
- `.venv-guix/` added to `.gitignore`.

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
