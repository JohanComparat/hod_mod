# hod_mod

JAX-accelerated HOD galaxy clustering and weak lensing predictions and fitting.

## Overview

`hod_mod` is a Python 3.11+ package for forward-modelling galaxy clustering (w_p)
and weak gravitational lensing (ΔΣ) from Halo Occupation Distribution (HOD) and
inverse-SHMR (ICSMF) models.  All numerical code is JAX-native, enabling
automatic differentiation and JIT compilation for efficient MCMC inference.

## Supported HOD models

| Class | Reference |
|---|---|
| `HODModel` | Zheng et al. 2007 |
| `MoreHODModel` | More et al. 2015 (BOSS CMASS) |
| `Kravtsov04HODModel` / `AUMHODModel` | Kravtsov et al. 2004 |
| `Guo18ICSMFModel` | Guo et al. 2018 |
| `Guo19ICSMFModel` | Guo et al. 2019 (eBOSS ELGs) |
| `Zacharegkas25HODModel` | Zacharegkas et al. 2025 |
| `VanUitert16CSMFModel` | van Uitert et al. 2016 |
| `ZuMandelbaum15HODModel` | Zu & Mandelbaum 2015 (iHOD) |
| `ZuMandelbaum16QuenchingModel` | Zu & Mandelbaum 2016/2017 |
| `Leauthaud12HODModel` | Leauthaud et al. 2012 |

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

```python
from hod_mod.fitting import load_config, WpFitter

cfg    = load_config("configs/hod_fit_more2015_cmass.yml")
fitter = WpFitter(cfg)
result = fitter.map_fit()        # Nelder-Mead MAP
chain  = fitter.mcmc_fit()       # emcee MCMC
```

## Install

```bash
pip install -e .
```

Or via the conda environment:

```bash
mamba env create -f environment.yml
mamba activate hod_mod
pip install -e .
```

## Citation

If you use `hod_mod` in published work please cite the papers for the HOD models
you use (see table above) and include a reference to this package.

## License

MIT — see [LICENSE](LICENSE).
