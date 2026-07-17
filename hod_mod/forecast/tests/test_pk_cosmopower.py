r"""Validation of the CosmoPower-JAX linear P(k) backend.

Four checks on :class:`hod_mod.forecast.pk_cosmopower.CosmoPowerJaxPkLinear`:

1. **σ8 self-consistency** — σ(R=8 Mpc/h) of ``pk_linear`` reproduces
   ``theta['sigma8']`` through the shared top-hat renormalisation (so the HMF and
   2-halo paths stay consistent, exactly as for the EH98 backend).
2. **Shape accuracy vs CAMB** — after σ8 renormalisation the emulator matches a
   massless-neutrino CAMB reference to well under 1% (materially better than
   EH98's few-to-ten percent), across a spread of cosmologies in the box.
3. **Differentiability** — ``jax.jacfwd`` of a ``pk_linear`` summary w.r.t. the
   σ8-native cosmology ``{Omega_m, h, sigma8, n_s, Omega_b}`` is finite and agrees
   with central finite differences.  This is the property NUTS depends on.
4. **Interface parity with EH98** — same duck-typed methods, same growth/z scaling.

``cosmopower-jax`` (and ``camb`` for check 2) are optional; the module is skipped
when they are absent so the suite still runs in a minimal environment.
"""

from __future__ import annotations

import warnings

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("cosmopower_jax")

from hod_mod.forecast import pk_cosmopower as pkcp  # noqa: E402
from hod_mod.forecast.pk_cosmopower import CosmoPowerJaxPkLinear  # noqa: E402
from hod_mod.forecast.pk_eisenstein_hu import (  # noqa: E402
    EisensteinHu98PkLinear,
    _K_INT,
    _sigma2_tophat,
)


def _theta(Omega_m=0.3153, Omega_b=0.0493, h=0.6736, n_s=0.9649, sigma8=0.8111):
    return dict(Omega_m=Omega_m, Omega_b=Omega_b, Omega_cdm=Omega_m - Omega_b,
                h=h, n_s=n_s, sigma8=sigma8)


# a handful of cosmologies spanning the emulator/forecast box
_COSMOS = [
    _theta(),
    _theta(Omega_m=0.27, sigma8=0.75, h=0.70),
    _theta(Omega_m=0.35, sigma8=0.85, h=0.65, n_s=0.98),
    _theta(Omega_m=0.31, Omega_b=0.045, sigma8=0.80, h=0.68, n_s=0.95),
]


@pytest.fixture(scope="module")
def cp():
    return CosmoPowerJaxPkLinear()


@pytest.mark.parametrize("theta", _COSMOS)
def test_sigma8_self_consistency(cp, theta):
    """σ(R=8) of pk_linear returns the requested sigma8 (renorm is consistent)."""
    s8 = float(jnp.sqrt(_sigma2_tophat(cp.pk_linear(_K_INT, 0.0, theta), _K_INT, 8.0)))
    assert abs(s8 - theta["sigma8"]) < 1e-3


def test_growth_scaling(cp):
    """pk_linear(z) / pk_linear(0) is k-independent and equals [D(z)/D(0)]²."""
    theta = _theta()
    k = jnp.logspace(-2, 0, 30)
    ratio = np.asarray(cp.pk_linear(k, 0.7, theta) / cp.pk_linear(k, 0.0, theta))
    assert np.ptp(ratio) < 1e-6          # scale-independent
    assert 0.2 < ratio.mean() < 1.0      # growth suppresses power at z>0


@pytest.mark.parametrize("theta", _COSMOS)
def test_shape_accuracy_vs_camb(theta):
    """Emulator shape matches massless-ν CAMB to <1% after σ8 renormalisation."""
    camb = pytest.importorskip("camb")
    cp = CosmoPowerJaxPkLinear()
    h = theta["h"]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=theta["Omega_b"] * h ** 2,
                       omch2=theta["Omega_cdm"] * h ** 2,
                       mnu=0.0, num_massive_neutrinos=0)
    pars.InitPower.set_params(ns=theta["n_s"], As=2.1e-9)
    pars.set_matter_power(redshifts=[0.0], kmax=10.0)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    kh, _z, pkc = res.get_matter_power_spectrum(minkh=1e-3, maxkh=3.0, npoints=300)
    pkc = pkc[0]

    k = jnp.asarray(kh)
    pk_cp = np.asarray(cp.pk_linear(k, 0.0, theta))
    # compare shapes: divide out the (renormalised) amplitude
    ratio = pk_cp / pkc
    ratio /= np.median(ratio)
    assert np.max(np.abs(ratio - 1.0)) < 0.01


def test_beats_eh98_vs_camb():
    """On the fiducial the emulator is at least ~5× closer to CAMB than EH98."""
    camb = pytest.importorskip("camb")
    theta = _theta()
    h = theta["h"]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=theta["Omega_b"] * h ** 2,
                       omch2=theta["Omega_cdm"] * h ** 2,
                       mnu=0.0, num_massive_neutrinos=0)
    pars.InitPower.set_params(ns=theta["n_s"], As=2.1e-9)
    pars.set_matter_power(redshifts=[0.0], kmax=10.0)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    kh, _z, pkc = res.get_matter_power_spectrum(minkh=1e-3, maxkh=2.0, npoints=200)
    pkc = pkc[0]
    k = jnp.asarray(kh)

    def shape_dev(backend):
        r = np.asarray(backend.pk_linear(k, 0.0, theta)) / pkc
        r /= np.median(r)
        return np.max(np.abs(r - 1.0))

    dev_cp = shape_dev(CosmoPowerJaxPkLinear())
    dev_eh = shape_dev(EisensteinHu98PkLinear())
    assert dev_cp < 0.01
    assert dev_cp < dev_eh / 5.0


def test_jacfwd_matches_finite_difference(cp):
    """Autodiff through pk_linear w.r.t. σ8-native cosmology matches FD."""
    k = jnp.logspace(-3, 0.5, 40)

    def summary(v):
        theta = dict(Omega_m=v[0], Omega_b=0.0493, Omega_cdm=v[0] - 0.0493,
                     h=v[1], n_s=v[3], sigma8=v[2])
        return jnp.sum(jnp.log(cp.pk_linear(k, 0.0, theta)))

    v0 = jnp.array([0.3153, 0.6736, 0.8111, 0.9649])
    J = np.asarray(jax.jacfwd(summary)(v0))
    assert np.all(np.isfinite(J))

    # Central-FD truncation tolerance is relative 5e-3: h shifts the log-log
    # interpolation grid (logk_grid = logk_native - log h), so a finite-step
    # stencil straddles the piecewise-linear interpolation kinks and carries a
    # larger truncation error than the other params — AD is exact (FD converges
    # to it as eps→0, verified to ~1e-8 at eps=3e-5); a missing chain-rule term
    # would still show up as an O(1) relative mismatch, far above this bound.
    eps = 1e-4
    for i in range(4):
        vp = v0.at[i].add(eps)
        vm = v0.at[i].add(-eps)
        fd = (float(summary(vp)) - float(summary(vm))) / (2 * eps)
        assert abs(fd - J[i]) <= 5e-3 * (abs(J[i]) + 1.0)


def test_interface_parity_with_eh98(cp):
    """Same duck-typed contract as the EH98 backend."""
    for meth in ("pk_shape", "pk_linear", "as_hmf_pk_func"):
        assert hasattr(cp, meth)
    theta = _theta()
    k = jnp.logspace(-3, 0, 20)
    pk_func = cp.as_hmf_pk_func()
    assert np.allclose(np.asarray(pk_func(k, 0.0, theta)),
                       np.asarray(cp.pk_shape(k, theta)))


def test_training_box_matches_network_stats():
    """TRAINING_BOX must track the network's own standardisation, not drift from it.

    The box is mean ± √3·std of a uniform training prior; if a future emulator
    ships different ranges this catches the stale constant.
    """
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
    emu = CosmoPowerJAX(probe="mpk_lin")
    mean = np.asarray(emu.param_train_mean)
    std = np.asarray(emu.param_train_std)
    lo, hi = mean - np.sqrt(3) * std, mean + np.sqrt(3) * std
    for i, name in enumerate(emu.parameters):
        declared_lo, declared_hi = pkcp.TRAINING_BOX[name]
        assert abs(lo[i] - declared_lo) < 2e-3
        assert abs(hi[i] - declared_hi) < 2e-3


def test_out_of_box_warns_once(monkeypatch):
    """First out-of-box call warns; later ones stay silent (never spam a sampler)."""
    monkeypatch.setattr(pkcp, "_WARNED_OUT_OF_BOX", False)
    with pytest.warns(RuntimeWarning, match="training box"):
        bad = pkcp.check_training_box({"h": 0.50})
    assert bad and "h=0.5" in bad[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pkcp.check_training_box({"h": 0.50})
        assert len(w) == 0            # already warned once


def test_in_box_does_not_warn(monkeypatch):
    monkeypatch.setattr(pkcp, "_WARNED_OUT_OF_BOX", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert pkcp.check_training_box({"h": 0.6736, "n_s": 0.9649}) == []
        assert len(w) == 0


def test_as_convention_matches_camb_shape(cp):
    """The A_s (CAMB/production) convention path reproduces CAMB's *shape*.

    Amplitude is deliberately not compared: production CAMB runs with its default
    ``mnu=0.06`` while the emulator is massless-trained, a ~2.5% offset that the
    A_s path (unlike the σ8 path) does not renormalise away.
    """
    camb = pytest.importorskip("camb")
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    theta = LinearPowerSpectrum.default_cosmology()
    assert "sigma8" not in theta and "ln10^{10}A_s" in theta   # production layout
    k = np.logspace(-3, 0.3, 40)
    pk_emu = np.asarray(cp.pk_linear(jnp.asarray(k), 0.0, theta))
    pk_camb = np.asarray(LinearPowerSpectrum().pk_linear(k, 0.0, theta))
    r = pk_emu / pk_camb
    r /= np.median(r)
    assert np.max(np.abs(r - 1.0)) < 0.03        # massive-nu shape difference


def test_as_convention_uses_native_redshift(cp):
    """A_s path evolves z through the network (trained z∈[0,5]), not an external D(z)."""
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    theta = LinearPowerSpectrum.default_cosmology()
    k = jnp.logspace(-2, 0, 20)
    p0 = np.asarray(cp.pk_linear(k, 0.0, theta))
    p1 = np.asarray(cp.pk_linear(k, 1.0, theta))
    assert np.all(np.isfinite(p1)) and np.all(p1 < p0)   # growth suppresses at z>0


def test_missing_amplitude_key_raises(cp):
    k = jnp.logspace(-2, 0, 5)
    with pytest.raises(KeyError, match="sigma8"):
        cp.pk_linear(k, 0.0, dict(Omega_m=0.31, Omega_b=0.049, h=0.6736, n_s=0.9649))
