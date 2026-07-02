r"""Unit tests for the tomographic parameter bookkeeping
(:mod:`hod_mod.forecast.tomography`).

Builds a two-bin :class:`TomographicForecast` on tiny grids and checks the
shared-vs-per-bin parameter layout, the fiducial/prior mapping, the static
per-bin gather, and the row metadata — *without* evaluating the (heavy)
``jax.jacfwd`` data vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from hod_mod.forecast.tomography import TomographicForecast, _HOD_NAMES
from hod_mod.forecast.forward_jax import PARAM_NAMES, _IDX

_TINY = dict(n_k=16, n_m=16, n_gl=8, n_z=3, n_z_shear=3)


@pytest.fixture(scope="module")
def tomo():
    samples = [("A", 0.15, 10.0), ("B", 0.25, 10.6)]
    return TomographicForecast(samples=samples, **_TINY)


def test_layout_shared_and_per_bin(tomo):
    assert tomo.n_bin == 2
    assert tomo.labels == ["A", "B"]
    # shared = all params except the 9 HOD ones
    assert set(tomo.shared_names) == set(PARAM_NAMES) - set(_HOD_NAMES)
    # global vector = 22 shared + 9 HOD per bin
    assert len(tomo.global_names) == len(tomo.shared_names) + len(_HOD_NAMES) * tomo.n_bin
    assert "fc__A" in tomo.global_names and "fc__B" in tomo.global_names
    assert "Omega_m" in tomo.global_names               # shared, not suffixed


def test_fiducial_shared_and_replicated(tomo):
    fid = tomo.fiducial()
    assert fid.shape == (len(tomo.global_names),)
    assert np.all(np.isfinite(fid))
    gidx = {n: i for i, n in enumerate(tomo.global_names)}
    # per-bin HOD fiducials are replicated (same base fiducial for both bins)
    assert fid[gidx["fc__A"]] == fid[gidx["fc__B"]]


def test_prior_fix_and_planck(tomo):
    p = tomo.prior(add_planck=True, fix=("bsat",))
    gidx = {n: i for i, n in enumerate(tomo.global_names)}
    from hod_mod.forecast import params
    assert p[gidx["Omega_m"]] == params.PLANCK_PRIOR_SIGMA["Omega_m"]
    # a fixed HOD parameter is pinned in *every* bin
    assert p[gidx["bsat__A"]] == 1e-4 and p[gidx["bsat__B"]] == 1e-4


def test_gather_routes_shared_and_per_bin_hod(tomo):
    # a distinct value per global slot; check each bin's 31-vector is reassembled
    g = np.arange(len(tomo.global_names), dtype=float)
    gidx = {n: i for i, n in enumerate(tomo.global_names)}
    for b, lab in enumerate(tomo.labels):
        local = np.asarray(g[np.asarray(tomo.gather[b])])
        assert local.shape == (len(PARAM_NAMES),)
        # shared parameter comes from the shared slot
        assert local[_IDX["Omega_m"]] == g[gidx["Omega_m"]]
        # HOD parameter comes from this bin's slot
        assert local[_IDX["fc"]] == g[gidx[f"fc__{lab}"]]


def test_data_vector_metadata(tomo):
    f, row_bin, row_obs, row_x = tomo.data_vector_fn(which=["wp"])
    assert callable(f)
    n = len(row_bin)
    assert len(row_obs) == n and len(row_x) == n
    assert set(np.unique(row_bin)) == {"A", "B"}
    assert set(np.unique(row_obs)) == {"wp"}


def test_scale_cut_mask_shape(tomo):
    _, row_bin, row_obs, row_x = tomo.data_vector_fn(which=["wp"])
    keep = tomo.scale_cut_mask(row_bin, row_obs, row_x, rmin=1.0)
    assert keep.dtype == bool and keep.shape == (len(row_obs),)
