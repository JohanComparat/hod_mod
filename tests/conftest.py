"""Shared pytest fixtures for hod_mod tests."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax.numpy as jnp

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile


@pytest.fixture(scope="session")
def planck_cosmo():
    """Planck 2018 cosmological parameter dict."""
    return LinearPowerSpectrum().default_cosmology()


@pytest.fixture(scope="session")
def pk_lin():
    """LinearPowerSpectrum (CAMB) instance."""
    return LinearPowerSpectrum()


@pytest.fixture(scope="session")
def hmf(pk_lin):
    """Tinker+2008 HMF backed by CAMB."""
    return make_hmf("tinker08", pk_func=pk_lin.pk_linear)


@pytest.fixture(scope="session")
def halo_profile():
    """NFW HaloProfile with Diemer+2019 c(M) relation."""
    colossus_cosmo = {
        "flat": True,
        "H0": 67.36,
        "Om0": 0.3100,
        "Ob0": 0.0493,
        "sigma8": 0.8111,
        "ns": 0.9649,
    }
    return HaloProfile(colossus_cosmo, cm_relation="diemer19")
