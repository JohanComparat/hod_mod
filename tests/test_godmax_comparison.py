"""GODMAX independent SZ cross-check (shared Battaglia+2012 profile).

Two layers:

* **Self-contained machinery checks** (always run, no external reference):
  the B12 ``pressure_uk`` spherical FT against an independently-coded direct
  quadrature; B12 vs A10 sanity bound; C_ℓ^{yy}/C_ℓ^{κy} positivity, 1h/2h
  crossover, and the W_κ lensing kernel shape.  These validate hod_mod's tSZ
  projection machinery on their own.

* **Frozen-reference comparison** (skipped unless the reference npz is present):
  P_e(r|M,z), ỹ(k|M) and — when the reference carries them — C_ℓ^{yy}/C_ℓ^{κy}
  against a file made by ``scripts/godmax/export_godmax_b12_reference.py``.
  The committed ``independent_b12_reference.npz`` exercises the profile + FT;
  generate ``godmax_b12_reference.npz`` (``--source godmax --with-cl``) in a
  GODMAX environment for the full-stack numeric check.  Mirrors the AUM-style
  skipif pattern of ``tests/test_aum_comparison.py``.
"""
import numpy as np
import pytest

from hod_mod.gas import PressureProfileBattaglia12, PressureProfileA10
from hod_mod.gas.conversions import _RHO_CRIT0, _SIGMA_T_OVER_ME_C2, _MPC_CM

_DATA = __import__("pathlib").Path(__file__).parents[1] / "hod_mod" / "data" / "godmax"
_THETA = {"Omega_m": 0.3089, "Omega_b": 0.0486, "h": 0.6774, "n_s": 0.9667, "sigma8": 0.8159}


def _r200c_comoving(m_h, z, om):
    ez2 = om * (1.0 + z) ** 3 + (1.0 - om)
    rho = _RHO_CRIT0 * ez2 / (1.0 + z) ** 3
    return float((3.0 * m_h / (4.0 * np.pi * 200.0 * rho)) ** (1.0 / 3.0))


def _direct_ytilde(pp, k, m_h, z, theta, n=600):
    """Independent direct-trapezoid spherical FT of pp._p3d (not GL/einsum)."""
    import jax.numpy as jnp
    r200 = _r200c_comoving(m_h, z, theta["Omega_m"])
    r = np.linspace(1e-3 * r200, pp._r_max_factor * r200, n)
    pe = np.asarray(pp._p3d(jnp.asarray(r / r200), jnp.asarray(m_h), z,
                            theta["h"], theta["Omega_m"],
                            theta["Omega_b"] / theta["Omega_m"], jnp.asarray(r200)))
    out = np.array([4.0 * np.pi * np.trapz(pe * np.sinc(ki * r / np.pi) * r ** 2, r)
                    for ki in k])
    return _SIGMA_T_OVER_ME_C2 * (_MPC_CM / theta["h"]) * out


# ---------------------------------------------------------------------------
# Self-contained machinery checks
# ---------------------------------------------------------------------------

class TestB12Profile:
    def test_pressure_uk_matches_direct_ft(self):
        """hod_mod's GL/einsum FT reproduces an independent trapezoid FT."""
        pp = PressureProfileBattaglia12(r_max_over_r200c=4.0, n_gl=200)
        k = np.logspace(-2, 0.7, 12)
        for m_h in (1e13, 1e14, 5e14):
            z = 0.4
            r200 = _r200c_comoving(m_h, z, _THETA["Omega_m"])
            hod = np.asarray(pp.pressure_uk(k, np.array([m_h]), np.array([r200]),
                                            np.array([5.0]), z, _THETA))[:, 0]
            ref = _direct_ytilde(pp, k, m_h, z, _THETA)
            assert np.max(np.abs(hod / ref - 1.0)) < 3e-3

    def test_pe_magnitude_sane(self):
        """P_e near R200c of a 3e14 halo is ~1e-4 keV/cm^3 (cluster outskirts)."""
        import jax.numpy as jnp
        pp = PressureProfileBattaglia12()
        m_h = 3e14; z = 0.5
        r200 = _r200c_comoving(m_h, z, _THETA["Omega_m"])
        pe = float(pp._p3d(jnp.asarray(1.0), jnp.asarray(m_h), z, _THETA["h"],
                           _THETA["Omega_m"], _THETA["Omega_b"] / _THETA["Omega_m"],
                           jnp.asarray(r200)))
        assert 1e-5 < pe < 1e-3

    def test_electron_fraction(self):
        """f_e = (2+2X_H)/(3+5X_H) ≈ 0.518 for X_H=0.76; 0.5 for pure H (more He → higher)."""
        assert abs(PressureProfileBattaglia12(x_h=0.76)._f_e - 0.5176) < 1e-3
        assert abs(PressureProfileBattaglia12(x_h=1.0)._f_e - 0.5) < 1e-9
        assert PressureProfileBattaglia12(x_h=0.5)._f_e > PressureProfileBattaglia12(x_h=1.0)._f_e

    def test_b12_vs_a10_bounded(self):
        """Two independent physical pressure profiles agree to within ~15%."""
        k = np.logspace(-2, 0.3, 8)
        m = np.array([3e14]); z = 0.5
        r200 = np.array([_r200c_comoving(m[0], z, _THETA["Omega_m"])]); c = np.array([5.0])
        b12 = np.asarray(PressureProfileBattaglia12().pressure_uk(k, m, r200, c, z, _THETA))[:, 0]
        a10 = np.asarray(PressureProfileA10().pressure_uk(k, m, r200, c, z, _THETA))[:, 0]
        assert np.max(np.abs(b12 / a10 - 1.0)) < 0.15


@pytest.fixture(scope="module")
def cross_b12():
    from hod_mod.observables.clustering import make_differentiable_prediction
    from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
    pred = make_differentiable_prediction("more15", cm_model="dutton14", mdef="200c")
    pp = PressureProfileBattaglia12(r_max_over_r200c=4.0, n_gl=96)
    return HaloModelCrossSpectra(pred, pressure_profile=pp)


@pytest.fixture(scope="module")
def hod_params():
    from hod_mod.connection.hod import MoreHODModel
    p = MoreHODModel.default_params()
    p.update({"log10mmin": 13.03, "sigma_logm": 0.38, "log10m1": 13.80,
              "alpha": 1.17, "kappa": 0.51})
    return p


@pytest.mark.slow
class TestProjectors:
    _ELL = np.array([100.0, 300.0, 1000.0, 3000.0, 8000.0])
    _Z = np.linspace(0.05, 3.0, 7)
    _NZ = np.exp(-0.5 * ((np.linspace(0.05, 3.0, 7) - 0.6) / 0.28) ** 2)

    def test_cl_yy_positive_and_crossover(self, cross_b12, hod_params):
        cl = np.asarray(cross_b12.angular_cl_yy(self._ELL, self._Z, _THETA, hod_params))
        assert np.all(cl > 0) and np.all(np.isfinite(cl))
        t = cross_b12._pk_tables_yy(0.3, _THETA, hod_params)
        k = np.exp(np.asarray(t["log_k"]))
        p1 = np.exp(np.asarray(t["log_pyy_1h"])); p2 = np.exp(np.asarray(t["log_pyy_2h"]))
        assert np.interp(0.01, k, p2) > np.interp(0.01, k, p1)   # 2h dominates low-k
        assert np.interp(1.0, k, p1) > np.interp(1.0, k, p2)     # 1h dominates high-k

    def test_cl_ky_positive(self, cross_b12, hod_params):
        cl = np.asarray(cross_b12.angular_cl_ky(self._ELL, self._Z, self._NZ,
                                                _THETA, hod_params))
        assert np.all(cl > 0) and np.all(np.isfinite(cl))

    def test_convergence_kernel_shape(self, cross_b12):
        z = np.linspace(0.02, 3.0, 40)
        nz = np.exp(-0.5 * ((z - 0.6) / 0.25) ** 2)
        wk = np.asarray(cross_b12._convergence_kernel(z, nz, _THETA))
        assert np.all(wk >= 0)
        assert 0 < np.argmax(wk) < len(wk) - 1           # peaks between lens & source
        assert wk[-1] < wk[np.argmax(wk)] * 1e-3         # → 0 beyond the sources


# ---------------------------------------------------------------------------
# Frozen-reference comparison (skipif the npz is absent)
# ---------------------------------------------------------------------------

def _load_reference():
    for name in ("godmax_b12_reference.npz", "independent_b12_reference.npz"):
        p = _DATA / name
        if p.exists():
            return np.load(p, allow_pickle=True), name
    return None, None


_REF, _REF_NAME = _load_reference()
requires_ref = pytest.mark.skipif(_REF is None, reason="no GODMAX/B12 reference npz")


@requires_ref
class TestAgainstReference:
    def test_profile(self):
        import jax.numpy as jnp
        pp = PressureProfileBattaglia12(r_max_over_r200c=4.0, n_gl=200)
        z, m, x, Pe = _REF["prof_z"], _REF["prof_m200c_Msun"], _REF["prof_x"], _REF["prof_Pe"]
        worst = 0.0
        for i, zi in enumerate(z):
            for j, mphys in enumerate(m):
                mh = float(mphys) * _THETA["h"]
                r200 = _r200c_comoving(mh, float(zi), _THETA["Omega_m"])
                hod = np.asarray(pp._p3d(jnp.asarray(x), jnp.asarray(mh), float(zi),
                                         _THETA["h"], _THETA["Omega_m"],
                                         _THETA["Omega_b"] / _THETA["Omega_m"],
                                         jnp.asarray(r200)))
                worst = max(worst, np.max(np.abs(hod / Pe[i, j] - 1.0)))
        assert worst < 1e-3, f"P_e max frac diff {worst:.2e} vs {_REF_NAME}"

    def test_ytilde(self):
        pp = PressureProfileBattaglia12(r_max_over_r200c=4.0, n_gl=200)
        z, k, m, yt = _REF["uk_z"], _REF["uk_k"], _REF["uk_m200c_h"], _REF["uk_ytilde"]
        worst = 0.0
        for i, zi in enumerate(z):
            r200 = np.array([_r200c_comoving(mm, float(zi), _THETA["Omega_m"]) for mm in m])
            hod = np.asarray(pp.pressure_uk(k, m, r200, np.full_like(m, 5.0),
                                            float(zi), _THETA))
            worst = max(worst, np.max(np.abs(hod / yt[i] - 1.0)))
        assert worst < 3e-3, f"ytilde max frac diff {worst:.2e} vs {_REF_NAME}"

    @pytest.mark.slow
    def test_cl_yy(self, cross_b12, hod_params):
        if "cl_yy" not in _REF:
            pytest.skip(f"{_REF_NAME} carries no C_l (regenerate with --with-cl)")
        cl = np.asarray(cross_b12.angular_cl_yy(_REF["ell"], _REF["cl_z"], _THETA, hod_params))
        worst = np.max(np.abs(cl / _REF["cl_yy"] - 1.0))
        assert worst < 0.15, f"C_l^yy max frac diff {worst:.2e}"

    @pytest.mark.slow
    def test_cl_ky(self, cross_b12, hod_params):
        if "cl_ky" not in _REF:
            pytest.skip(f"{_REF_NAME} carries no C_l (regenerate with --with-cl)")
        cl = np.asarray(cross_b12.angular_cl_ky(_REF["ell"], _REF["cl_z"],
                                                _REF["nz_source"], _THETA, hod_params))
        worst = np.max(np.abs(cl / _REF["cl_ky"] - 1.0))
        assert worst < 0.20, f"C_l^ky max frac diff {worst:.2e}"
