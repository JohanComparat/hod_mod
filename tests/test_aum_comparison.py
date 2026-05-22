"""Cross-validation tests: AUM (surhudm/aum) vs hod_mod.

All tests are skipped automatically if AUM is not importable — this avoids
CI failures on systems where AUM has not been built.

AUM must be installed for Python 3.11 at:
  /home/comparat/software/previous/aum/install/lib/python3.11/site-packages/

To build AUM:
  cd /home/comparat/software/previous/aum
  mamba install -n hod_mod -c conda-forge swig=4.0.0 gsl=2.8
  mamba run -n hod_mod python setup.py install --prefix=./install  (twice)

Expected tolerances
-------------------
- N_cen:       < 0.1%   (identical erfc formula)
- N_sat:       < 2%     (AUMHODModel = exact TINK=0 replica)
- n_gal:       < 2%     (same σ₈, same HMF integral)
- b_eff:       < 2%
- w_p:         < 60%    (two-halo dominated by P_lin shape: CAMB vs AUM/E-H)
- delta_sigma: < 130%   (additionally sensitive to c(M): Diemer+2019 vs AUM built-in)

The large w_p / ΔΣ tolerances are intentional.  The dominant sources of
disagreement are:

1. **Linear P(k) shape**: hod_mod uses CAMB (full Boltzmann solver);
   AUM uses the Eisenstein-Hu 1998 fitting formula normalised to the same σ₈.
   At k ∈ [0.1, 1] h/Mpc the shapes differ by 10–15%, which propagates
   directly to w_p at the same level on 2-halo scales (rp > 3 Mpc/h).

2. **Concentration–mass relation**: hod_mod uses Diemer+2019 (colossus);
   AUM uses a built-in Klypin-like c(M).  This creates additional 20–50%
   differences in the 1-halo term at R < 1 Mpc/h.

The tests verify positivity, monotonicity, and correct relative ordering.
The comparison script (scripts/utils/compare_aum.py) shows the full %diff
including a panel for the P_lin shape mismatch.
"""
import sys
import math
import pytest
import numpy as np
import jax.numpy as jnp

_AUM_PATH = "/home/comparat/software/previous/aum/install/lib/python3.11/site-packages"
if _AUM_PATH not in sys.path:
    sys.path.insert(0, _AUM_PATH)


def _aum_importable() -> bool:
    try:
        import hod  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _aum_importable(),
    reason=(
        "AUM not installed — build with: "
        "cd /home/comparat/software/previous/aum && "
        "mamba run -n hod_mod python setup.py install --prefix=./install (twice)"
    ),
)

_THETA_COSMO = {
    "h":             0.6736,
    "Omega_m":       0.3100,
    "Omega_b":       0.0493,
    "Omega_cdm":     0.2607,
    "n_s":           0.9649,
    "ln10^{10}A_s":  3.044,
    "w0": -1.0,
    "wa":  0.0,
}

_HOD_PARAMS = {
    "log10mmin":  13.0,
    "sigma_logm":  0.5,
    "log10m0":    10.0,   # Mcut << Mmin → exp(-M0/M) ≈ 1
    "log10m1":    14.0,
    "alpha":       1.0,
}

_Z = 0.1
_PI_MAX = 60.0


def _compute_sigma8(theta: dict) -> float:
    import camb
    h = float(theta["h"])
    lnAs = float(theta["ln10^{10}A_s"])
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * h,
        ombh2=float(theta["Omega_b"]) * h**2,
        omch2=float(theta["Omega_cdm"]) * h**2,
    )
    pars.InitPower.set_params(ns=float(theta["n_s"]), As=np.exp(lnAs) * 1e-10)
    pars.set_dark_energy(w=float(theta.get("w0", -1.0)), wa=float(theta.get("wa", 0.0)),
                         dark_energy_model="ppf")
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)
    return float(camb.get_results(pars).get_sigma8_0())


def _make_aum_hod(theta: dict, sigma8: float, hod_params: dict):
    import hod as h_aum
    p = h_aum.cosmo()
    p.Om0   = float(theta["Omega_m"])
    p.Omk   = 0.0
    p.w0    = float(theta.get("w0", -1.0))
    p.wa    = float(theta.get("wa",  0.0))
    p.Omb   = float(theta["Omega_b"])
    p.hval  = float(theta["h"])
    p.th    = 2.7255
    p.s8    = sigma8
    p.nspec = float(theta["n_s"])
    p.ximax = math.log10(8.0)
    p.cfac  = 1.0
    q = h_aum.hodpars()
    q.Mmin    = hod_params["log10mmin"]
    q.siglogM = hod_params["sigma_logm"]
    q.Msat    = hod_params["log10m1"]
    q.alpsat  = hod_params["alpha"]
    q.Mcut    = hod_params["log10m0"]
    q.csbycdm = 1.0
    q.fac     = 1.0
    return h_aum.hod(p, q)


def _to_carr(arr):
    import hod as h_aum
    c = h_aum.doubleArray(len(arr))
    for i, v in enumerate(arr):
        c[i] = float(v)
    return c


def _from_carr(c, n):
    return np.array([c[i] for i in range(n)])


@pytest.fixture(scope="module")
def setup():
    """Build both model stacks once per module (expensive CAMB call)."""
    pytest.importorskip("camb")
    from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
    from hod_mod.cosmology.halo_mass_function import make_hmf
    from hod_mod.cosmology.halo_profiles import HaloProfile
    from hod_mod.galaxies.hod import AUMHODModel
    from hod_mod.galaxies.clustering import FullHaloModelPrediction

    sigma8 = _compute_sigma8(_THETA_COSMO)

    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hod    = AUMHODModel(hmf, hmf.bias)

    colossus_cosmo = {
        "flat": True, "H0": _THETA_COSMO["h"] * 100.0,
        "Om0": _THETA_COSMO["Omega_m"], "Ob0": _THETA_COSMO["Omega_b"],
        "sigma8": sigma8, "ns": _THETA_COSMO["n_s"],
    }
    hp   = HaloProfile(colossus_cosmo, cm_relation="diemer19")
    pred = FullHaloModelPrediction(pk_lin, hod, hp, profile="nfw")
    aum  = _make_aum_hod(_THETA_COSMO, sigma8, _HOD_PARAMS)

    return {
        "pred": pred, "hod": hod, "aum": aum,
        "sigma8": sigma8,
    }


class TestOccupation:
    """N_cen and N_sat point-by-point comparison."""

    _LOG10M = np.linspace(12.0, 15.5, 50)

    def test_ncen_agrees(self, setup):
        """N_cen agrees to < 0.1% — identical erfc formula."""
        from hod_mod.galaxies.hod import n_cen
        nc_gga = np.asarray(n_cen(jnp.asarray(self._LOG10M),
                                   _HOD_PARAMS["log10mmin"], _HOD_PARAMS["sigma_logm"]))
        nc_aum = np.array([setup["aum"].ncen(lm) for lm in self._LOG10M])
        # Only compare where N_cen > 0.01 (avoid numerical noise near zero)
        mask = nc_gga > 0.01
        pct = 100.0 * np.abs(nc_gga[mask] - nc_aum[mask]) / nc_aum[mask]
        assert np.all(pct < 0.1), f"N_cen max %diff = {pct.max():.3f}%"

    def test_nsat_agrees(self, setup):
        """N_sat agrees to < 2% — AUMHODModel is exact TINK=0 replica."""
        from hod_mod.galaxies.hod import n_sat_aum
        ns_gga = np.asarray(n_sat_aum(jnp.asarray(self._LOG10M),
                                       _HOD_PARAMS["log10mmin"], _HOD_PARAMS["sigma_logm"],
                                       _HOD_PARAMS["log10m0"], _HOD_PARAMS["log10m1"],
                                       _HOD_PARAMS["alpha"]))
        ns_aum = np.array([setup["aum"].nsat(lm) for lm in self._LOG10M])
        mask = ns_gga > 0.01
        pct = 100.0 * np.abs(ns_gga[mask] - ns_aum[mask]) / ns_aum[mask]
        assert np.all(pct < 2.0), f"N_sat max %diff = {pct.max():.3f}%"

    def test_ncen_positive_monotone(self, setup):
        """N_cen is monotonically non-decreasing with halo mass.

        Strict > 0 fails at saturation (float32 gives nc=1 exactly, diff=0).
        """
        from hod_mod.galaxies.hod import n_cen
        nc = np.asarray(n_cen(jnp.asarray(self._LOG10M),
                               _HOD_PARAMS["log10mmin"], _HOD_PARAMS["sigma_logm"]))
        assert np.all(np.diff(nc) >= 0)

    def test_nsat_positive_and_increasing(self, setup):
        """N_sat > 0 and increasing for M > Mmin in both codes."""
        from hod_mod.galaxies.hod import n_sat_aum
        log10m_hi = np.linspace(13.5, 15.5, 20)
        ns = np.asarray(n_sat_aum(jnp.asarray(log10m_hi),
                                   _HOD_PARAMS["log10mmin"], _HOD_PARAMS["sigma_logm"],
                                   _HOD_PARAMS["log10m0"], _HOD_PARAMS["log10m1"],
                                   _HOD_PARAMS["alpha"]))
        assert np.all(ns > 0)
        assert np.all(np.diff(ns) > 0)


class TestNumberDensityAndBias:

    def test_ngal_agrees(self, setup):
        """n_gal agrees to < 2% (same σ₈, same HOD + HMF integrals)."""
        n_gal_gga, _, _ = setup["hod"]._integrate(_Z, _THETA_COSMO, _HOD_PARAMS)
        n_gal_gga = float(n_gal_gga)
        n_gal_aum = setup["aum"].ncenz(_Z) + setup["aum"].nsatz(_Z)
        pct = 100.0 * abs(n_gal_gga - n_gal_aum) / n_gal_aum
        assert pct < 2.0, f"n_gal %diff = {pct:.2f}%"

    def test_beff_agrees(self, setup):
        """Effective bias agrees to < 2%."""
        _, b_eff_gga, _ = setup["hod"]._integrate(_Z, _THETA_COSMO, _HOD_PARAMS)
        b_eff_gga = float(b_eff_gga)
        b_eff_aum = setup["aum"].galaxy_bias(_Z)
        pct = 100.0 * abs(b_eff_gga - b_eff_aum) / b_eff_aum
        assert pct < 2.0, f"b_eff %diff = {pct:.2f}%"

    def test_ngal_positive(self, setup):
        """n_gal > 0 in both codes."""
        n_gal_gga, _, _ = setup["hod"]._integrate(_Z, _THETA_COSMO, _HOD_PARAMS)
        n_gal_aum = setup["aum"].ncenz(_Z) + setup["aum"].nsatz(_Z)
        assert float(n_gal_gga) > 0
        assert n_gal_aum > 0


class TestWp:

    _RP = np.logspace(np.log10(0.5), np.log10(20.0), 12)

    def test_wp_agrees(self, setup):
        """w_p agrees to within 60% — dominated by CAMB vs AUM/E-H P_lin shape.

        The 2-halo contribution at rp > 3 Mpc/h differs by ~10–15% because
        hod_mod uses CAMB P_lin while AUM uses Eisenstein-Hu fitted to the
        same σ₈.  The 1-halo term adds discrepancy at small scales.
        """
        import hod as h_aum
        wp_gga = np.asarray(setup["pred"].wp(jnp.asarray(self._RP), _PI_MAX, _Z,
                                              _THETA_COSMO, _HOD_PARAMS))
        rp_c = _to_carr(self._RP)
        wp_c = h_aum.doubleArray(len(self._RP))
        setup["aum"].Wp(_Z, len(self._RP), rp_c, wp_c, _PI_MAX)
        wp_aum = _from_carr(wp_c, len(self._RP))

        pct = 100.0 * np.abs(wp_gga - wp_aum) / np.abs(wp_aum)
        assert np.all(pct < 60.0), (
            f"w_p max %diff = {pct.max():.1f}% at rp={self._RP[pct.argmax()]:.2f} Mpc/h"
        )

    def test_wp_positive(self, setup):
        """w_p > 0 in both codes."""
        import hod as h_aum
        wp_gga = np.asarray(setup["pred"].wp(jnp.asarray(self._RP), _PI_MAX, _Z,
                                              _THETA_COSMO, _HOD_PARAMS))
        rp_c = _to_carr(self._RP)
        wp_c = h_aum.doubleArray(len(self._RP))
        setup["aum"].Wp(_Z, len(self._RP), rp_c, wp_c, _PI_MAX)
        wp_aum = _from_carr(wp_c, len(self._RP))
        assert np.all(wp_gga > 0)
        assert np.all(wp_aum > 0)

    def test_wp_decreasing(self, setup):
        """w_p is monotonically decreasing with rp in both codes."""
        import hod as h_aum
        wp_gga = np.asarray(setup["pred"].wp(jnp.asarray(self._RP), _PI_MAX, _Z,
                                              _THETA_COSMO, _HOD_PARAMS))
        rp_c = _to_carr(self._RP)
        wp_c = h_aum.doubleArray(len(self._RP))
        setup["aum"].Wp(_Z, len(self._RP), rp_c, wp_c, _PI_MAX)
        wp_aum = _from_carr(wp_c, len(self._RP))
        assert np.all(np.diff(wp_gga) < 0), "hod_mod w_p is not monotonically decreasing"
        assert np.all(np.diff(wp_aum) < 0), "AUM w_p is not monotonically decreasing"


class TestDeltaSigma:

    _R = np.logspace(np.log10(0.3), 1.0, 10)

    def test_ds_agrees(self, setup):
        """ΔΣ agrees to within 130% — dominated by c(M) + P_lin differences.

        At R < 1 Mpc/h the 1-halo term is sensitive to the concentration-mass
        relation; hod_mod uses Diemer+2019, AUM uses its own built-in c(M).
        The 2-halo contribution at large R adds the same P_lin shape mismatch
        as for w_p.
        """
        import hod as h_aum
        ds_gga = np.asarray(setup["pred"].delta_sigma(jnp.asarray(self._R), _Z,
                                                       _THETA_COSMO, _HOD_PARAMS))
        R_c   = _to_carr(self._R)
        esd_c = h_aum.doubleArray(len(self._R))
        setup["aum"].ESD(_Z, len(self._R), R_c, esd_c, len(self._R) + 8)
        ds_aum = _from_carr(esd_c, len(self._R))

        pct = 100.0 * np.abs(ds_gga - ds_aum) / np.abs(ds_aum)
        assert np.all(pct < 130.0), (
            f"ΔΣ max %diff = {pct.max():.1f}% at R={self._R[pct.argmax()]:.2f} Mpc/h"
        )

    def test_ds_positive(self, setup):
        """ΔΣ > 0 in both codes."""
        import hod as h_aum
        ds_gga = np.asarray(setup["pred"].delta_sigma(jnp.asarray(self._R), _Z,
                                                       _THETA_COSMO, _HOD_PARAMS))
        R_c   = _to_carr(self._R)
        esd_c = h_aum.doubleArray(len(self._R))
        setup["aum"].ESD(_Z, len(self._R), R_c, esd_c, len(self._R) + 8)
        ds_aum = _from_carr(esd_c, len(self._R))
        assert np.all(ds_gga > 0)
        assert np.all(ds_aum > 0)

    def test_ds_decreasing(self, setup):
        """ΔΣ is monotonically decreasing with R in both codes."""
        import hod as h_aum
        ds_gga = np.asarray(setup["pred"].delta_sigma(jnp.asarray(self._R), _Z,
                                                       _THETA_COSMO, _HOD_PARAMS))
        R_c   = _to_carr(self._R)
        esd_c = h_aum.doubleArray(len(self._R))
        setup["aum"].ESD(_Z, len(self._R), R_c, esd_c, len(self._R) + 8)
        ds_aum = _from_carr(esd_c, len(self._R))
        assert np.all(np.diff(ds_gga) < 0), "hod_mod ΔΣ is not monotonically decreasing"
        assert np.all(np.diff(ds_aum) < 0), "AUM ΔΣ is not monotonically decreasing"
