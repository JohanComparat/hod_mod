# hod_mod

JAX-accelerated HOD galaxy clustering, weak lensing, and gas cross-correlation
predictions and fitting.

[![CI Tests](https://img.shields.io/github/actions/workflow/status/JohanComparat/hod_mod/tests.yml?branch=main&label=tests)](https://github.com/JohanComparat/hod_mod/actions)
[![Coverage](https://img.shields.io/codecov/c/github/JohanComparat/hod_mod?label=coverage)](https://codecov.io/gh/JohanComparat/hod_mod)
[![Docs](https://img.shields.io/readthedocs/hod-mod?label=docs)](https://hod-mod.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/hod-mod)](https://pypi.org/project/hod-mod/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/hod-mod/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21078473.svg)](https://doi.org/10.5281/zenodo.21078473)

## Overview

`hod_mod` is a Python 3.11+ package for forward-modelling galaxy clustering (w_p),
weak gravitational lensing (ΔΣ), and galaxy × gas cross-correlations (tSZ Compton-y,
soft X-ray) from Halo Occupation Distribution (HOD) and inverse-SHMR (iHOD) models.
All numerical code is JAX-native, enabling automatic differentiation and JIT
compilation for efficient MCMC inference.

## Install

Available on [PyPI](https://pypi.org/project/hod-mod/):

```bash
pip install hod-mod
```

For development, create and activate the conda environment then install in editable mode:

```bash
# Download the installer (Linux x86_64)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Run the installer (follow prompts, accept defaults)
bash Miniforge3-Linux-x86_64.sh

# Reload shell
source ~/.bashrc
```

```bash
mamba env create -f environment.yml
mamba activate hod_mod
pip install -e .
pre-commit install          # optional: blocks committing large files / results/
```

## Data and benchmark results

Small reference data needed to run the models ships inside the package. Large
inputs and the curated benchmark results (final MCMC chains, headline figures)
are archived on Zenodo and fetched **on demand** — the git repository stays lean.

- **Dataset:** [10.5281/zenodo.21078473](https://doi.org/10.5281/zenodo.21078473) (concept DOI — always resolves to the latest version)
- Downloads are checksum-verified and cached locally with
  [`pooch`](https://www.fatiando.org/pooch/) (a dependency, installed automatically).

```python
from hod_mod.data_io import fetch

# downloads from Zenodo + verifies the checksum on first call; cache hit afterwards
chain = fetch("results/benchmarks/more2015_logM11_12/flatchain.npz")
```

Optional configuration via environment variables:

| Variable | Purpose |
|---|---|
| `HOD_MOD_DATA_DOI` | pin a specific Zenodo version for reproducibility (default: pinned in code) |
| `HOD_MOD_DATA_DIR` | read from a local mirror directory instead of downloading |
| `HOD_MOD_DATA_CACHE` | override the download cache location |

See [docs/data_hosting.rst](docs/data_hosting.rst) for the full strategy and the
upload/registry workflow.

## Tests

```bash
pytest                             # run all tests
pytest tests/test_cosmology.py    # single module
pytest -x                         # stop on first failure
pytest -v                         # verbose output
pytest --tb=short                 # compact tracebacks
```

The test suite covers cosmology, HOD models, gas profiles, clustering predictions,
cross-spectra, data I/O, and fitting.  Tests that require optional backends
(`camb`, `colossus`) are skipped automatically if those packages are absent.

## Supported HOD models

| Class | Reference |
|---|---|
| `HODModel` | [Zheng et al. 2007](https://arxiv.org/abs/astro-ph/0703457) |
| `MoreHODModel` | [More et al. 2015](https://arxiv.org/abs/1407.1856) (BOSS CMASS) |
| `Kravtsov04HODModel` | [Kravtsov et al. 2004](https://doi.org/10.1086/420959) |
| `Guo18ICSMFModel` | [Guo et al. 2018](https://arxiv.org/abs/1804.01993) |
| `Guo19ICSMFModel` | [Guo et al. 2019](https://arxiv.org/abs/1810.05318) (eBOSS ELGs) |
| `Zacharegkas25HODModel` | [Zacharegkas et al. 2025](https://arxiv.org/abs/2506.22367) |
| `VanUitert16CSMFModel` | [van Uitert et al. 2016](https://arxiv.org/abs/1601.06791) |
| `ZuMandelbaum15HODModel` | [Zu & Mandelbaum 2015](https://arxiv.org/abs/1505.02781) (iHOD) |
| `ZuMandelbaum16QuenchingModel` | [Zu & Mandelbaum 2016](https://arxiv.org/abs/1509.06758) |
| `Leauthaud12HODModel` | [Leauthaud et al. 2012](https://arxiv.org/abs/1104.0928) |

All clustering HOD classes subclass `HODBase` (ABC) and implement `nc_ns()` and
`default_params()`.

## Gas profiles and cross-correlations

`hod_mod` predicts galaxy × gas cross-correlations using parametric electron
pressure and density profiles embedded in the same halo model framework.

**Gas profile classes** (`hod_mod.cosmology.gas_profiles`):

| Class | Physical profile | Reference |
|---|---|---|
| `PressureProfileA10` | electron pressure P_e(r\|M,z) → tSZ Compton-y | [Arnaud et al. 2010](https://arxiv.org/abs/0910.1234) |
| `GasDensityDPM` (model=1,2,3) | electron density n_e(r\|M,z) → soft X-ray ε | [Oppenheimer et al. 2025](https://arxiv.org/abs/2505.14782) |
| `m200_to_m500c` | NFW bisection: M₂₀₀ → M₅₀₀c, R₅₀₀c | — |

**Cross-spectrum observables** (`hod_mod.galaxies.cross_spectra`):

| Method | Observable | Units |
|---|---|---|
| `_pk_tables_gy` | P_{g,y}(k), P_{m,y}(k), 1h+2h | (Mpc/h)² |
| `_pk_tables_gX` | P_{g,X}(k), 1h+2h | (Mpc/h)³ cm⁻⁶ |
| `projected_gy` | Σ_y(r_p) stacked tSZ profile | dimensionless Compton-y |
| `projected_gX` | w_{g,X}(r_p) stacked X-ray profile | (Mpc/h) cm⁻⁶ |
| `angular_cl_gy` | C_ℓ^{g,y} via Limber approximation | (Mpc/h)² |
| `angular_cl_gX` | C_ℓ^{g,X} via Limber approximation | (Mpc/h) cm⁻⁶ |

```python
from hod_mod.cosmology import PressureProfileA10, GasDensityDPM
from hod_mod.galaxies.cross_spectra import HaloModelCrossSpectra

pp    = PressureProfileA10(r_max_over_r500c=5.0, n_gl=200)   # Arnaud+2010
dp    = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=200) # Oppenheimer+2025
cross = HaloModelCrossSpectra(fhmp, pressure_profile=pp, density_profile=dp)

sigma_y = cross.projected_gy(rp, z=0.5, theta_cosmo=theta, hod_params=params)
cl_gy   = cross.angular_cl_gy(ell, z_arr, nz_g, theta, params)
wgX     = cross.projected_gX(rp, z=0.5, theta_cosmo=theta, hod_params=params)
```

Benchmark data for [Comparat et al. 2025](https://arxiv.org/abs/2503.19796)
(galaxy × eROSITA 0.5–2 keV, 7 stellar-mass-selected samples, LS DR10 × eRASS:5)
is included in `hod_mod/data/benchmarks/xray/`.

## Quick start — clustering and lensing

```python
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies import MoreHODModel, FullHaloModelPrediction
import jax.numpy as jnp

pk_lin = LinearPowerSpectrum()
theta  = pk_lin.default_cosmology()
hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)

colossus_cosmo = dict(flat=True, H0=67.36, Om0=0.31, Ob0=0.0493, sigma8=0.811, ns=0.965)
hp = HaloProfile(colossus_cosmo, cm_relation="diemer19")

hod    = MoreHODModel(hmf, hmf.bias)
pred   = FullHaloModelPrediction(pk_lin, hod, hp, profile="nfw")

rp     = jnp.logspace(-1, 1.5, 20)
params = MoreHODModel.default_params()
wp     = pred.wp(rp, pi_max=60.0, z=0.5, theta_cosmo=theta, hod_params=params)
```

`"tinker08"` is the library's dependency-free default HMF backend. The
fitting pipelines under `hod_mod/scripts/fitting/` instead use
`make_hmf("csst")` (CSSTEMU) as their baseline — see
[docs/cosmology.rst](docs/cosmology.rst) for details.

## HOD fitting

Run from the repository root (paths in configs are resolved relative to it):

```python
from hod_mod.fitting import load_config, WpFitter

cfg     = load_config("configs/hod_fit_more2015_cmass.yml")
fitter  = WpFitter(cfg)
result  = fitter.map_fit()               # Nelder-Mead MAP → dict
sampler = fitter.sample()               # emcee MCMC → EnsembleSampler
chain   = sampler.get_chain(flat=True)  # shape (n_steps * n_walkers, n_free)
```

The sample data file `data/more2015_boss_cmass/wp_cmass_z052.csv` is included in
the repository (More+2015, arXiv:1407.1856, Figure 2).

## Reproducing published results

Each benchmark paper has a dedicated validation script.
Run any script from the repository root:

| Paper | Script | Observable |
|---|---|---|
| [More et al. 2015](https://arxiv.org/abs/1407.1856) | `run_benchmark.py --model more2015` | w_p(r_p) BOSS CMASS |
| [Lange et al. 2025](https://arxiv.org/abs/2512.15962) | `run_benchmark.py --model lange2025_bgs3_bwpd_hsc` | w_p + ΔΣ DESI BGS |
| [Arnaud et al. 2010](https://arxiv.org/abs/0910.1234) | `validate_arnaud2010.py` | A10 pressure profile |
| [Oppenheimer et al. 2025](https://arxiv.org/abs/2505.14782) | `validate_oppenheimer2025.py` | DPM density profile |
| [Amodeo et al. 2021](https://arxiv.org/abs/2009.05557) | `validate_amodeo2021.py` | Σ_y(r_p) BOSS CMASS tSZ |
| [Pandey et al. 2025](https://arxiv.org/abs/2506.07432) | `validate_pandey2025.py` | C_ℓ^{g,y} DES × ACT |
| [Comparat et al. 2025](https://arxiv.org/abs/2503.19796) | `validate_comparat2025.py` | w_θ(θ) LS DR10 × eROSITA |

Run clustering/lensing benchmarks:

```bash
# from repo root
python hod_mod/scripts/benchmarks/run_benchmark.py --model more2015 --plot
python hod_mod/scripts/benchmarks/run_all_benchmarks.py --plot
```

Run gas/cross-correlation validation scripts:

```bash
python -m hod_mod.scripts.validate_arnaud2010
python -m hod_mod.scripts.validate_oppenheimer2025
python -m hod_mod.scripts.validate_sz_xray
python -m hod_mod.scripts.validate_amodeo2021
python -m hod_mod.scripts.validate_pandey2025
python -m hod_mod.scripts.validate_comparat2025
```

Figures are saved to `hod_mod/scripts/figures/`.

## Citation

If you use `hod_mod` in published work, cite:

> Comparat et al. 2025, A&A 697, A173
> https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.173C

and this repository URL.  Depending on the model used, additionally cite the
relevant HOD or gas profile paper(s) from the tables above.

If you use the archived benchmark data or curated results, also cite the
dataset: [10.5281/zenodo.21078473](https://doi.org/10.5281/zenodo.21078473).

---

## License

MIT — see [LICENSE](LICENSE).
