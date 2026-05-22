# hod_mod

JAX-accelerated HOD galaxy clustering and weak lensing predictions and fitting.

[![CI Tests](https://img.shields.io/github/actions/workflow/status/JohanComparat/hod_mod/tests.yml?branch=main&label=tests)](https://github.com/JohanComparat/hod_mod/actions)
[![Coverage](https://img.shields.io/codecov/c/github/JohanComparat/hod_mod?label=coverage)](https://codecov.io/gh/JohanComparat/hod_mod)
[![Docs](https://img.shields.io/readthedocs/hod-mod?label=docs)](https://hod-mod.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/hod-mod)](https://pypi.org/project/hod-mod/)
[![Python](https://img.shields.io/pypi/pyversions/hod-mod)](https://pypi.org/project/hod-mod/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Overview

`hod_mod` is a Python 3.11+ package for forward-modelling galaxy clustering (w_p)
and weak gravitational lensing (ΔΣ) from Halo Occupation Distribution (HOD) and
inverse-SHMR (ICSMF) models.  All numerical code is JAX-native, enabling
automatic differentiation and JIT compilation for efficient MCMC inference.

## Install

Create and activate the conda environment, then install the package in editable mode:

```bash
mamba env create -f environment.yml
mamba activate hod_mod
pip install -e .
```

## Tests

```bash
pytest                      # run all tests
pytest tests/test_cosmology.py          # single module
pytest -x                   # stop on first failure
pytest -v                   # verbose output
pytest --tb=short           # compact tracebacks
```

The test suite covers cosmology, HOD models, clustering predictions, data I/O, and fitting.  Tests that require optional backends (`camb`, `colossus`) are skipped automatically if those packages are absent.

## Supported HOD models

| Class | Reference |
|---|---|
| `HODModel` | [Zheng et al. 2007](https://arxiv.org/abs/astro-ph/0703457) |
| `MoreHODModel` | [More et al. 2015](https://arxiv.org/abs/1407.1011) (BOSS CMASS) |
| `Kravtsov04HODModel` | [Kravtsov et al. 2004](https://doi.org/10.1086/420959) |
| `Guo18ICSMFModel` | [Guo et al. 2018](https://arxiv.org/abs/1707.01922) |
| `Guo19ICSMFModel` | [Guo et al. 2019](https://arxiv.org/abs/1811.10583) (eBOSS ELGs) |
| `Zacharegkas25HODModel` | [Zacharegkas et al. 2025](https://arxiv.org/abs/2506.22367) |
| `VanUitert16CSMFModel` | [van Uitert et al. 2016](https://arxiv.org/abs/1601.06791) |
| `ZuMandelbaum15HODModel` | [Zu & Mandelbaum 2015](https://arxiv.org/abs/1505.02781) (iHOD) |
| `ZuMandelbaum16QuenchingModel` | [Zu & Mandelbaum 2016](https://arxiv.org/abs/1509.06758) |
| `Leauthaud12HODModel` | [Leauthaud et al. 2012](https://arxiv.org/abs/1104.0928) |

All clustering HOD classes subclass `HODBase` (ABC) and implement `nc_ns()` and
`default_params()`.

## Quick start

```python
import hod_mod
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

The sample data file `data/more2015_boss_cmass/wp_cmass_z052.csv` is included in the repository (More+2015, arXiv:1407.1856, Figure 2).

## Citation

If you use `hod_mod` in published work please cite the papers for the HOD models
you use (see table above) and include a reference to this package.

## License

MIT — see [LICENSE](LICENSE).
