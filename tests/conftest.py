"""Shared pytest fixtures for hod_mod tests."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

# Enable x64 BEFORE the hod_mod imports below: module-level jnp constants (the
# Gauss-Legendre nodes in core.halo_profiles, pk_eisenstein_hu._K_INT, ...) are
# materialised at first import with whatever precision is active.  In any
# multi-file run the x64-marked test modules flip the flag during collection
# anyway, so tests already execute under x64 — but by then conftest has imported
# core.halo_profiles and frozen its nodes at float32, silently degrading
# finite-difference comparisons (see test_eh98_backend lifted-features).
jax.config.update("jax_enable_x64", True)

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
