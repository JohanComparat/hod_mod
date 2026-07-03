r"""Validation of the JAX Powell AGN chain against the numpy PowellAGNModel.

The forecast's ``_agn_kernel_parts`` / ``_agn_occupation`` / ``_wp_agn`` and the
``agn_emission="powell"`` point-source term re-express the validated numpy
:class:`hod_mod.agn.powell.PowellAGNModel` chain in JAX.  The one deliberate
difference is the SHMR: the forecast uses the ZM15 inverse SHMR (its own free
parameters) where powell.py uses Girelli20.  ``_log10ms`` is a plain attribute
of the numpy model, so these tests substitute the ZM15 mean into it and then
require the two chains to agree.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from hod_mod.forecast.forward_jax import (  # noqa: E402
    ForwardModel, _IDX, _K_H2S_FID, _inv_shmr)
from hod_mod.forecast import params  # noqa: E402


FID = params.fiducial_vector()
_Z = 0.25


@pytest.fixture(scope="module")
def model():
    return ForwardModel(z_eff=_Z, n_k=64, n_m=220, n_gl=16, n_z=3, n_z_shear=3)


@pytest.fixture(scope="module")
def powell(model):
    """Numpy PowellAGNModel on the SAME mass grid / HMF / SHMR / parameters."""
    from hod_mod.agn.powell import PowellAGNModel
    th = jnp.asarray(FID)
    c = {k: float(v) for k, v in model._cosmo(th).items()}
    fid = {n: FID[i] for n, i in _IDX.items()}
    pw = PowellAGNModel(
        c, model._hmf, z_mean=_Z,
        mbh_m=(fid["agn_mu_bh"], fid["agn_al_bh"], fid["agn_sig_bh"]),
        erdf=(fid["agn_log10_lstar"], fid["agn_delta1"], fid["agn_delta2"]),
        rho=fid["agn_rho"], sham_scatter=fid["agn_sig_mstar"],
        log10lx_min=42.0,
        log10m_min=float(model.log10m[0]), log10m_max=float(model.log10m[-1]),
        n_m=len(np.asarray(model.log10m)),
        loglam_min=float(model.loglam[0]), loglam_max=float(model.loglam[-1]),
        n_lam=len(np.asarray(model.loglam)),
    )
    pw.set_params(log10_ferdf=fid["agn_log10_ferdf"])
    # substitute the ZM15 SHMR mean (the deliberate difference vs Girelli20)
    hp = model._hod(th)
    pw._log10ms = np.asarray(_inv_shmr(model.log10m, hp["lg_m1h"], hp["lg_m0star"],
                                       hp["beta"], hp["delta"], hp["gamma"]))
    return pw


def test_occupation_matches_powell(model, powell):
    """Analytic erf bin integral == numpy convolution kernel, on-grid."""
    th = jnp.asarray(FID)
    ours = np.asarray(model._agn_occupation(th, 42.0, float(powell.loglx[-1])))
    ref, _ = powell.nc_ns_agn(np.asarray(model.log10m))
    # powell.py's discretized convolution carries an O(dlam²/σ²) error vs our
    # analytic erf integral: ~1% through the selection transition, a few
    # percent in the deep tail, ~0.1% at saturation.
    hi = ref > 1e-3
    assert np.max(np.abs(ours[hi] - ref[hi]) / ref[hi]) < 1e-2
    lo = ref > 1e-12
    assert np.max(np.abs(ours[lo] - ref[lo]) / ref[lo]) < 3e-2


def test_xlf_matches_powell(model, powell):
    th = jnp.asarray(FID)
    ours = np.asarray(model._xlf(th))
    ref = powell.xlf(np.asarray(model.loglx_xlf), band="hard")
    np.testing.assert_allclose(ours, ref, rtol=2e-2)


def test_agn_bias_matches_powell(model, powell):
    th = jnp.asarray(FID)
    H = model._halo_common(th, _Z)
    n = model._agn_occupation(th, 42.0, float(powell.loglx[-1]))
    b_ours = float(jnp.trapezoid(H["dndm"] * H["bias"] * n, model.m)
                   / jnp.trapezoid(H["dndm"] * n, model.m))
    b_ref = powell.agn_bias()
    assert abs(b_ours - b_ref) / b_ref < 1e-2


def test_ferdf_amplitude_identity(model):
    """n_agn ∝ 10^ferdf exactly (below the N≤1 clip): dln n/dferdf = ln10."""
    th = np.asarray(FID).copy()
    i = _IDX["agn_log10_ferdf"]
    H = model._halo_common(jnp.asarray(th), _Z)

    def n_agn(v):
        t = th.copy(); t[i] = v
        n = model._agn_occupation(jnp.asarray(t), 42.0, 46.0)
        return float(jnp.trapezoid(H["dndm"] * n, model.m))
    eps = 1e-4
    dln = (np.log(n_agn(th[i] + eps)) - np.log(n_agn(th[i] - eps))) / (2 * eps)
    np.testing.assert_allclose(dln, np.log(10.0), rtol=1e-6)


def test_wp_agn_shape_and_gradient(model):
    """wp_agn: finite, positive at large r_p, and jacfwd == finite differences."""
    th = jnp.asarray(FID)
    wp = np.asarray(model.predict(th, ["wp_agn"])["wp_agn"])
    nb, nr = len(model.agn_lx_bins), len(np.asarray(model.rp_wp_agn))
    assert wp.shape == (nb * nr,) and np.all(np.isfinite(wp)) and np.all(wp > 0)
    # higher-L_X bins live in more massive halos -> higher bias -> higher wp
    per_bin = wp.reshape(nb, nr)
    assert per_bin[-1, 0] > per_bin[0, 0]

    f = lambda t: model.predict(t, ["wp_agn"])["wp_agn"]
    for pname in ("agn_mu_bh", "agn_rho", "Omega_m"):
        i = _IDX[pname]
        ad = np.asarray(jax.jacfwd(f)(th))[:, i]
        eps = 1e-4 * max(abs(float(th[i])), 1.0)
        tp = np.asarray(th).copy(); tm = np.asarray(th).copy()
        tp[i] += eps; tm[i] -= eps
        fd = (np.asarray(f(jnp.asarray(tp))) - np.asarray(f(jnp.asarray(tm)))) / (2 * eps)
        assert np.median(np.abs(ad - fd) / (np.abs(fd) + 1e-30)) < 0.02, pname


def test_powell_emission_mode(model):
    """powell mode: C_gX responds to ferdf and NOT to log10DC; surrogate default
    is unchanged by the AGN sector parameters that only the Powell chain uses."""
    kw = dict(z_eff=_Z, n_k=48, n_m=48, n_gl=16, n_z=3, n_z_shear=3)
    m_pow = ForwardModel(agn_emission="powell", **kw)
    m_sur = ForwardModel(agn_emission="surrogate", **kw)
    th = jnp.asarray(FID)

    for m, active, inert in ((m_pow, "agn_log10_ferdf", "log10DC"),
                             (m_sur, "log10DC", "agn_mu_bh")):
        f = lambda t: m.predict(t, ["cl_gX"])["cl_gX"]
        J = np.asarray(jax.jacfwd(f)(th))
        d0 = np.asarray(f(th))
        assert np.abs(J[:, _IDX[active]]).max() > 1e-6 * np.abs(d0).max(), m.agn_emission
        assert np.abs(J[:, _IDX[inert]]).max() == 0.0, m.agn_emission

    # the Powell AGN mean emission is a sane, positive amplitude
    amp = np.asarray(m_pow._agn_emissivity_amp(th))
    assert np.all(np.isfinite(amp)) and np.all(amp > 0)
    assert _K_H2S_FID == pytest.approx(0.607, abs=1e-3)
