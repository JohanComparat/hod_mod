"""Integration coverage for the HAM AGN model (agn.ham). CAMB + abundance matching.

Builds the halo-abundance-matching X-ray AGN model (the heavy __init__ that maps
the hard XLF onto the HMF via the SHMR) for both XLF references and exercises its
L_X and emissivity methods. Marked slow.
"""
import warnings

import numpy as np
import pytest

from hod_mod.agn import HamAGNModel
from hod_mod.core.power_spectrum import LinearPowerSpectrum

pytestmark = pytest.mark.slow

_THETA = LinearPowerSpectrum.default_cosmology()


@pytest.fixture(scope="module")
def pk():
    return LinearPowerSpectrum()


@pytest.mark.parametrize("xlf", ["aird15", "ueda14"])
def test_ham_construction_and_lx_mapping(pk, xlf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # kcorr fallback warning
        model = HamAGNModel(pk_lin=pk, xlf=xlf)
    assert model._xlf_name == xlf

    log10_mh = np.linspace(12.0, 14.5, 15)
    lx_hard = np.asarray(model.ham_log10lx_hard(log10_mh, z=0.2))
    assert lx_hard.shape == log10_mh.shape
    assert np.all(np.isfinite(lx_hard))
    # more massive haloes host more luminous AGN (abundance-matching monotonicity)
    assert lx_hard[-1] > lx_hard[0]


def test_ham_mean_lx(pk):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HamAGNModel(pk_lin=pk)
    m_halo = 10.0 ** np.linspace(12.0, 14.5, 12)
    mean_log10lx = np.asarray(model.mean_agn_log10lx(m_halo, z=0.2))
    mean_lx      = np.asarray(model.mean_agn_lx(m_halo, z=0.2))
    assert np.all(np.isfinite(mean_log10lx)) and np.all(np.isfinite(mean_lx))
    assert np.all(mean_lx > 0)


def test_ham_emissivity_uk(pk):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HamAGNModel(pk_lin=pk)
    k = np.logspace(-2, 1, 8)
    m_halo = 10.0 ** np.linspace(12.0, 14.0, 10)
    X = np.asarray(model.agn_emissivity_uk(k, m_halo, 0.2, _THETA))
    assert np.all(np.isfinite(X))
