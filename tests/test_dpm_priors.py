"""The native<-scaling Jacobian must reproduce the analytic DPM identities.

Guards :mod:`hod_mod.fitting.dpm_priors`, which back-propagates the GAS.py L_X-M /
kT-M priors onto the native DPM gas parameters.  If this map drifts, the band fit's
priors silently stop encoding the scaling-relation information they claim to.

The identities below are not empirical — they follow from the DPM definitions:

* ``T = P/n_e`` exactly  =>  d kt_norm/d log10 P_0.3 = +1 and
  d kt_norm/d log10 n_e,0.3 = -1 (kT constrains the *ratio*), and
  d kt_slope/d beta_P = -d kt_slope/d beta_n (kt_slope tracks beta_P - beta_n).
* ``L_X ∝ n_e²``  =>  d lx_norm/d log10 n_e,0.3 = 2.
* ``L_X ∝ Λ(T)`` with T set by the pressure  =>  d lx_slope/d beta_P > 0:
  L_X partly traces the pressure through the cooling function.  A diagonal prior
  cannot represent that.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_Z = 0.135


@pytest.fixture(scope="module")
def jac_and_names():
    import jax
    if not jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=1 (emission integral carries r_cm**2 ~ 1e49)")
    import jax.numpy as jnp
    import hod_mod.scripts.validate_gas_profiles as V
    from hod_mod.gas import m200_to_m500c
    from hod_mod.gas.cooling import ApecCoolingTable
    from hod_mod.fitting.dpm_priors import ScalingCtx, scaling_map
    from hod_mod.scripts.direct_prediction_gal_gas_agn import (
        _P_03_CAL, _BETA_P_CAL, _NE_03_CAL, _BETA_N_CAL)

    h = V._H
    m200 = np.geomspace(3e13, 2e15, 10) * h
    r200 = np.array([V._r200(m, _Z) for m in m200])
    c200 = np.array([V._c200_approx(m) for m in m200])
    m500c, r500c = m200_to_m500c(m200, c200, r200, V._rho_crit_z(_Z))
    shape = dict(a_in_n=1.0, a_tr_n=1.9, a_out_n=2.7, gamma_n=2.0,
                 a_in_p=0.3, a_tr_p=1.3, a_out_p=4.1, gamma_p=8.0 / 3.0)
    ctx = ScalingCtx(m200, r200, np.asarray(m500c), np.asarray(r500c),
                     ApecCoolingTable(emin=0.5, emax=2.0), shape, _Z,
                     float(V._ez(_Z)), h)
    theta0 = jnp.asarray([np.log10(_NE_03_CAL), _BETA_N_CAL,
                          np.log10(_P_03_CAL), _BETA_P_CAL])
    J = np.asarray(jax.jacfwd(lambda t: scaling_map(t, ctx))(theta0))
    # rows: lx_norm, lx_slope, kt_norm, kt_slope; cols: log10_ne03, beta_n, log10_p03, beta_P
    return J


def test_kt_norm_traces_the_pressure_density_ratio(jac_and_names):
    """T = P/n_e  =>  d kt_norm/d log10 P_0.3 = +1 = -d kt_norm/d log10 n_e,0.3."""
    J = jac_and_names
    d_p03, d_ne03 = J[2, 2], J[2, 0]
    assert d_p03 == pytest.approx(1.0, abs=0.02)
    assert d_ne03 == pytest.approx(-1.0, abs=0.02)
    # exactly antisymmetric: kT constrains the ratio, never P alone
    assert d_p03 == pytest.approx(-d_ne03, abs=1e-6)


def test_kt_slope_traces_beta_P_minus_beta_n(jac_and_names):
    """kt_slope ∝ (beta_P - beta_n): equal magnitude, opposite sign."""
    J = jac_and_names
    d_bp, d_bn = J[3, 3], J[3, 1]
    assert d_bp > 0 and d_bn < 0
    assert d_bp == pytest.approx(-d_bn, rel=1e-4)


def test_lx_norm_traces_ne_squared(jac_and_names):
    """L_X ∝ n_e²  =>  d lx_norm/d log10 n_e,0.3 ≈ 2 (slightly under, via T_min/Λ)."""
    J = jac_and_names
    assert J[0, 0] == pytest.approx(2.0, abs=0.15)


def test_lx_slope_feels_the_pressure_through_cooling(jac_and_names):
    """The cross-term a diagonal prior would miss: L_X partly traces P via Λ(T)."""
    J = jac_and_names
    assert abs(J[1, 3]) > 0.05, "lx_slope should respond to beta_P through Λ(T)"


def test_induced_prior_reproduces_the_scaling_prior_centre():
    """Newton must land exactly on f(theta*) = mu_s, else the prior is mis-centred."""
    import jax
    if not jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=1")
    import numpy as np
    import hod_mod.scripts.validate_gas_profiles as V
    from hod_mod.gas import m200_to_m500c
    from hod_mod.gas.cooling import ApecCoolingTable
    from hod_mod.fitting.dpm_priors import ScalingCtx, induced_gaussian_prior
    from hod_mod.scripts.direct_prediction_gal_gas_agn import (
        _P_03_CAL, _BETA_P_CAL, _NE_03_CAL, _BETA_N_CAL)
    h = V._H
    m200 = np.geomspace(3e13, 2e15, 10) * h
    r200 = np.array([V._r200(m, _Z) for m in m200])
    c200 = np.array([V._c200_approx(m) for m in m200])
    m500c, r500c = m200_to_m500c(m200, c200, r200, V._rho_crit_z(_Z))
    shape = dict(a_in_n=1.0, a_tr_n=1.9, a_out_n=2.7, gamma_n=2.0,
                 a_in_p=0.3, a_tr_p=1.3, a_out_p=4.1, gamma_p=8.0 / 3.0)
    ctx = ScalingCtx(m200, r200, np.asarray(m500c), np.asarray(r500c),
                     ApecCoolingTable(emin=0.5, emax=2.0), shape, _Z,
                     float(V._ez(_Z)), h)
    mu_s = np.array([44.7, 1.61, 0.4, 0.6])
    sig_s = np.array([0.3, 0.3, 0.2, 0.15])
    theta0 = np.array([np.log10(_NE_03_CAL), _BETA_N_CAL, np.log10(_P_03_CAL), _BETA_P_CAL])
    mu_n, cov_n, J, f_at = induced_gaussian_prior(ctx, mu_s, sig_s, theta0)
    np.testing.assert_allclose(f_at, mu_s, atol=1e-6)
    # covariance must be a valid, strongly-correlated PSD matrix
    assert np.all(np.linalg.eigvalsh(cov_n) > 0)
    sd = np.sqrt(np.diag(cov_n))
    corr = cov_n / np.outer(sd, sd)
    off = corr[np.triu_indices(4, k=1)]
    assert np.abs(off).max() > 0.5, "induced prior should be strongly correlated"
