"""The native-DPM radial band weight must reproduce the full radial emission integral.

Guards :mod:`hod_mod.fitting.dpm_bands`, which re-bases the X-ray band model onto the
native DPM gas parameters {n_e,0.3, β_n, P_0.3, β_P} — the same parameters that drive
the tSZ Σ_y, and hence the X-ray ↔ SZ coupling of the joint fit.

Ground truth is ``validate_gas_profiles._integrate_profile`` (a direct radial
∫ n_e² Λ(T(r),Z) dV with T = P/n_e).

Two facts these tests pin down, both learned the hard way:

* The **isothermal** shortcut — one Λ(kT_ew) rescaling — is wrong by ~14%, because
  Λ is non-linear in T and T has a radial profile.  That is why the J-table exists.
* The ``T_min`` cut puts a sharp edge in J(T_0); a coarse log-T_0 grid smears it and
  degrades low-mass accuracy to several %.  The grid must resolve the edge.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_Z = 0.135
_T_MIN = 0.3
_Z_METAL = 0.3


@pytest.fixture(scope="module")
def x64():
    """The emission integral carries r_cm² ~ 1e49 — float32 overflows it to inf."""
    import jax
    if not jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=1 (r_cm**2 ~ 1e49 overflows float32)")
    return True


@pytest.fixture(scope="module")
def setup(x64):
    import hod_mod.scripts.validate_gas_profiles as V
    from hod_mod.gas.cooling import ApecCoolingTable
    from hod_mod.scripts.direct_prediction_gal_gas_agn import (
        _make_pressure_variant, _make_density_variant,
        _P_03_CAL, _BETA_P_CAL, _NE_03_CAL, _BETA_N_CAL)
    dp = _make_density_variant(model=2, ne_03=_NE_03_CAL, beta=_BETA_N_CAL)
    pp = _make_pressure_variant(model=2, P_03=_P_03_CAL, beta=_BETA_P_CAL)
    broad = ApecCoolingTable(emin=0.5, emax=2.0)
    ez = float(np.sqrt(V._OM * (1 + _Z) ** 3 + (1 - V._OM)))
    return V, dp, pp, broad, ez, (_NE_03_CAL, _BETA_N_CAL, _P_03_CAL, _BETA_P_CAL)


class _ConstZ:
    def metallicity_3d(self, r, r200):
        return np.full_like(np.asarray(r, float), _Z_METAL)


def _truth_and_native(setup, ne03, p03, bn, bP, n_lt):
    """(truth Lx, native Lx) for one native-DPM parameter point."""
    import hod_mod.scripts.validate_gas_profiles as V
    from hod_mod.gas import m200_to_m500c
    from hod_mod.fitting.dpm_bands import (build_j_table, t0_of_mass,
                                           emission_measure_factor, _gnfw)
    from hod_mod.scripts.direct_prediction_gal_gas_agn import (
        _make_pressure_variant, _make_density_variant)
    V_, dp0, pp0, broad, ez, _ = setup
    masses = np.array([3e13, 1e14, 3e14]) * V_._H       # bulk X-ray-emitting masses
    lt_grid = np.linspace(-1.6, 1.6, n_lt)
    z_grid = np.array([0.15, 0.3, 0.6])

    truth, native = [], []
    for m200 in masses:
        r200 = V_._r200(m200, _Z)
        c2 = V_._c200_approx(m200)
        m500, r500 = m200_to_m500c(np.array([m200]), np.array([c2]),
                                   np.array([r200]), V_._rho_crit_z(_Z))
        dp = _make_density_variant(model=2, ne_03=ne03, beta=bn)
        pp = _make_pressure_variant(model=2, P_03=p03, beta=bP)
        lx, _, _ = V_._integrate_profile(m200, r200, float(r500[0]), _Z, pp, dp,
                                         _ConstZ(), T_min=_T_MIN)
        truth.append(lx)

        r_hi = min(float(r500[0]), 3.0 * r200)
        rs = r200 / dp0._C_DPM
        x_lo, x_hi = 0.01 * r200 / rs, r_hi / rs
        J = build_j_table(dp0, pp0, [broad], z_grid=z_grid, log10_t0_grid=lt_grid,
                          x_lo=x_lo, x_hi=x_hi, n_x=250, t_min=_T_MIN)
        t0 = t0_of_mass(np.array([m200]), p03, ne03, bP, bn, ez)
        em = emission_measure_factor(np.array([m200]), ne03, bn, ez)[0]
        xx = np.linspace(x_lo, x_hi, 250)
        shape_int = float(np.trapezoid(
            _gnfw(xx, dp0._alpha_in, dp0._alpha_tr, dp0._alpha_out) ** 2 * xx ** 2, xx))
        f_ref = _gnfw(0.3 * dp0._C_DPM, dp0._alpha_in, dp0._alpha_tr, dp0._alpha_out)
        v_shape = 4 * np.pi * (rs * (V_._MPC_CM / V_._H)) ** 3 * shape_int / f_ref ** 2
        native.append(em * v_shape * J(t0, _Z_METAL)[0, 0])
    return np.array(truth), np.array(native)


@pytest.mark.parametrize("tag,ne03,p03_mul,bn,bP", [
    ("reference", 1.260e-5, 1.0, 0.20, 0.80),
    ("ne03_up", 2.1e-5, 1.0, 0.20, 0.80),
    ("p03_up", 1.260e-5, 1.7, 0.20, 0.80),
    ("beta_n_up", 1.260e-5, 1.0, 0.32, 0.80),
    ("beta_P_up", 1.260e-5, 1.0, 0.20, 0.95),
])
def test_j_factorization_matches_radial_truth(setup, tag, ne03, p03_mul, bn, bP):
    """J_b(T_0,Z) must reproduce ∫n_e²Λ(T(r))dV as the native params vary."""
    from hod_mod.scripts.direct_prediction_gal_gas_agn import _P_03_CAL
    truth, native = _truth_and_native(setup, ne03, p03_mul * _P_03_CAL, bn, bP, n_lt=481)
    live = truth > 1e-6 * truth.max()
    assert live.any(), "T_min killed every halo — test point is degenerate"
    np.testing.assert_allclose(native[live], truth[live], rtol=5e-3)


def test_isothermal_shortcut_is_materially_wrong(setup):
    """Documents *why* the J-table exists: one Λ(kT_ew) is wrong at the ~10% level.

    If this ever starts passing at J-table accuracy, the radial temperature profile
    has been lost somewhere and the coupling is no longer DPM's.
    """
    import hod_mod.scripts.validate_gas_profiles as V
    from hod_mod.gas import m200_to_m500c
    from hod_mod.gas.cooling import ApecCoolingTable
    from hod_mod.scripts.direct_prediction_gal_gas_agn import (
        _make_pressure_variant, _make_density_variant, _P_03_CAL, _BETA_P_CAL,
        _NE_03_CAL, _BETA_N_CAL)
    V_, _, _, broad, ez, _ = setup
    m200 = 1e14 * V_._H
    r200 = V_._r200(m200, _Z)
    c2 = V_._c200_approx(m200)
    m500, r500 = m200_to_m500c(np.array([m200]), np.array([c2]),
                               np.array([r200]), V_._rho_crit_z(_Z))

    def radial(p03):
        dp = _make_density_variant(model=2, ne_03=_NE_03_CAL, beta=_BETA_N_CAL)
        pp = _make_pressure_variant(model=2, P_03=p03, beta=_BETA_P_CAL)
        return V_._integrate_profile(m200, r200, float(r500[0]), _Z, pp, dp,
                                     _ConstZ(), T_min=_T_MIN)

    lx0, kt0, _ = radial(_P_03_CAL)
    lx1, kt1, _ = radial(1.7 * _P_03_CAL)
    # isothermal prediction: Lx scales purely as Λ(kT) at the emission-weighted T
    lam0 = float(np.asarray(broad(np.array([kt0]), np.array([_Z_METAL])))[0])
    lam1 = float(np.asarray(broad(np.array([kt1]), np.array([_Z_METAL])))[0])
    iso = lx0 * lam1 / lam0
    assert abs(iso / lx1 - 1.0) > 0.02, (
        "isothermal rescaling unexpectedly accurate — radial T profile may be lost")
