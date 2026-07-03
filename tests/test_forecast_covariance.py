r"""Unit tests for the analytic Gaussian covariance
(:mod:`hod_mod.forecast.covariance`).

The correlation-coefficient machinery is pure numpy; :func:`gaussian_covariance`
only needs a ``.ell`` grid and a ``.predict`` returning the three lensing field
spectra, so a light stub model exercises it without building the forward model.
"""

from __future__ import annotations

import numpy as np

from hod_mod.forecast import covariance as cov_mod


def test_field_pair_mapping():
    assert cov_mod._field_pair("cl_kk") == ("k", "k")
    assert cov_mod._field_pair("cl_shear_kCMB") == ("k", "kc")
    assert cov_mod._field_pair("cl_kCMB") == ("kc", "kc")
    assert cov_mod._field_pair("not_an_observable") is None


def test_lensing_rho_diagonal_is_unity_and_bounded():
    ell = np.array([100.0, 300.0, 1000.0])
    pred = {"cl_kk": np.array([1.0, 0.5, 0.2]),
            "cl_kCMB": np.array([2.0, 1.0, 0.4]),
            "cl_shear_kCMB": np.array([1.2, 0.6, 0.25])}
    rho, names = cov_mod._lensing_rho(pred, ell)
    assert set(names) == {"cl_kk", "cl_shear_kCMB", "cl_kCMB"}
    for n in names:
        np.testing.assert_allclose(rho[(n, n)], np.ones_like(ell), rtol=1e-12)
    for i in names:
        for j in names:
            assert np.all(np.abs(rho[(i, j)]) <= 1.0 + 1e-9)
            np.testing.assert_allclose(rho[(i, j)], rho[(j, i)], rtol=1e-12)


class _StubModel:
    """Minimal ForwardModel stand-in for gaussian_covariance."""

    def __init__(self, ell, spectra):
        self.ell = np.asarray(ell)
        self._spectra = spectra

    def predict(self, theta, which):
        return {o: np.asarray(self._spectra[o]) for o in which}


def _build_case():
    ell = np.array([100.0, 300.0, 1000.0, 3000.0])
    spectra = {"cl_kk": np.array([1.0, 0.6, 0.3, 0.1]),
               "cl_kCMB": np.array([2.0, 1.2, 0.5, 0.15]),
               "cl_shear_kCMB": np.array([1.3, 0.8, 0.35, 0.11])}
    model = _StubModel(ell, spectra)
    # full data vector: a wp block (no correlations) + the three lensing spectra
    row_obs = (["wp"] * 3 + ["cl_kk"] * 4 + ["cl_shear_kCMB"] * 4 + ["cl_kCMB"] * 4)
    row_x = np.arange(len(row_obs), dtype=float)
    d0 = np.concatenate([np.array([5.0, 4.0, 3.0]),
                         spectra["cl_kk"], spectra["cl_shear_kCMB"], spectra["cl_kCMB"]])
    rel_err = np.full(len(row_obs), 0.1)
    return model, d0, np.array(row_obs), row_x, rel_err


def test_gaussian_covariance_diagonal_and_symmetry():
    model, d0, row_obs, row_x, rel_err = _build_case()
    C = cov_mod.gaussian_covariance(model, np.zeros(3), d0, row_obs, row_x, rel_err)
    n = len(d0)
    assert C.shape == (n, n)
    np.testing.assert_allclose(C, C.T, rtol=1e-12)
    # diagonal equals the validated per-bin variance
    np.testing.assert_allclose(np.diag(C), (rel_err * d0) ** 2, rtol=1e-10)


def test_gaussian_covariance_positive_definite():
    model, d0, row_obs, row_x, rel_err = _build_case()
    C = cov_mod.gaussian_covariance(model, np.zeros(3), d0, row_obs, row_x, rel_err)
    assert np.min(np.linalg.eigvalsh(C)) > 0.0


def test_gaussian_covariance_lensing_crossterms_present_others_zero():
    model, d0, row_obs, row_x, rel_err = _build_case()
    C = cov_mod.gaussian_covariance(model, np.zeros(3), d0, row_obs, row_x, rel_err)
    idx = {n: np.where(row_obs == n)[0] for n in np.unique(row_obs)}
    # kk × kCMB (same matter field) → positive off-diagonal correlation
    i0, j0 = idx["cl_kk"][0], idx["cl_kCMB"][0]
    assert C[i0, j0] > 0.0
    # wp is uncorrelated with everything off its own diagonal
    wp = idx["wp"]
    off = C[np.ix_(wp, [k for k in range(len(d0)) if k not in wp])]
    np.testing.assert_allclose(off, 0.0, atol=1e-15)
