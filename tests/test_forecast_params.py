r"""Unit tests for the forecast parameter bookkeeping
(:mod:`hod_mod.forecast.params`): fiducial vector, prior vectors and labels.

Pure-numpy / JSON only — no forward model is built.
"""

from __future__ import annotations

import numpy as np

from hod_mod.forecast import params
from hod_mod.forecast.forward_jax import PARAM_NAMES, _IDX, N_PARAM


def test_fiducial_vector_shape_and_finiteness():
    v = params.fiducial_vector()
    assert v.shape == (N_PARAM,)
    assert np.all(np.isfinite(v))


def test_load_fiducial_has_all_param_names():
    fid = params.load_fiducial()
    for n in PARAM_NAMES:
        assert n in fid, f"missing fiducial for {n}"
    # vector ordering matches PARAM_NAMES
    v = params.fiducial_vector()
    for n in PARAM_NAMES:
        assert v[_IDX[n]] == float(fid[n])


def test_fiducial_cosmology_is_planck18_ballpark():
    fid = params.load_fiducial()
    assert 0.2 < fid["Omega_m"] < 0.4
    assert 0.7 < fid["sigma8"] < 0.9
    assert 0.6 < fid["h"] < 0.75


def test_prior_sigma_vector_only_planck_finite():
    s = params.prior_sigma_vector()
    assert s.shape == (N_PARAM,)
    finite = {PARAM_NAMES[i] for i in np.where(np.isfinite(s))[0]}
    assert finite == set(params.PLANCK_PRIOR_SIGMA)
    for n, val in params.PLANCK_PRIOR_SIGMA.items():
        assert s[_IDX[n]] == val


def test_regularizing_prior_default_is_broad():
    s = params.regularizing_prior()
    assert s.shape == (N_PARAM,)
    assert np.all(np.isfinite(s))
    for n in PARAM_NAMES:
        assert s[_IDX[n]] == params.BROAD_PRIOR_SIGMA[n]


def test_regularizing_prior_add_planck_tightens_targets():
    broad = params.regularizing_prior(add_planck=False)
    planck = params.regularizing_prior(add_planck=True)
    # the Planck prior is tighter on the two targets
    assert planck[_IDX["Omega_m"]] < broad[_IDX["Omega_m"]]
    assert planck[_IDX["sigma8"]] < broad[_IDX["sigma8"]]
    assert planck[_IDX["Omega_m"]] == params.PLANCK_PRIOR_SIGMA["Omega_m"]


def test_regularizing_prior_fix_pins_parameters():
    s = params.regularizing_prior(fix=("log10_M_pivot", "log10_eta_min"))
    assert s[_IDX["log10_M_pivot"]] == 1e-4
    assert s[_IDX["log10_eta_min"]] == 1e-4
    # a non-fixed parameter keeps its broad width
    assert s[_IDX["fc"]] == params.BROAD_PRIOR_SIGMA["fc"]


def test_latex_labels_cover_all_params():
    labs = params.latex_labels()
    assert len(labs) == N_PARAM
    assert all(isinstance(x, str) and x.startswith("$") for x in labs)
