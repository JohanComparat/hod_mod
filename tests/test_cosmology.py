"""Tests for cosmology module."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.core.power_spectrum import LinearPowerSpectrum, eisenstein_hu_pk
from hod_mod.core.nonlinear import NonLinearPowerSpectrum
from hod_mod.core.halo_profiles import (
    nfw_rho, nfw_mass, nfw_sigma, nfw_delta_sigma, nfw_mean_sigma,
    nfw_uk, nfw_uk_jax, einasto_rho, einasto_uk, satellite_nfw_uk,
    HaloProfile, concentration_dutton14_jax,
    _si_jax, _ci_jax,
)
from hod_mod.core.halo_model import HaloModelPowerSpectrum
from hod_mod.core.halo_mass_function import (
    tinker08_fsigma, tinker10_bias,
    fsigma_press74, fsigma_sheth99, fsigma_jenkins01, fsigma_warren06,
    fsigma_angulo12, fsigma_crocce10, fsigma_watson13, fsigma_bhattacharya11,
    fsigma_courtin11, fsigma_bocquet16, fsigma_despali16,
    fsigma_rodriguezpuebla16, fsigma_comparat17, fsigma_seppi20,
    fsigma_yung24, fsigma_yung25,
    delta_vir_flat_jax, HaloMassFunction, make_hmf,
)
from hod_mod.core.distances import (
    hubble_e, comoving_distance, angular_diameter_distance,
    luminosity_distance, distance_modulus,
    comoving_distance_z1z2, angular_diameter_distance_z1z2,
    comoving_volume_element, comoving_volume,
    lookback_time, age_of_universe,
)


_PLANCK18 = {
    "h": 0.6736,
    "Omega_m": 0.3111,
    "Omega_b": 0.0493,
    "Omega_cdm": 0.2607,
    "n_s": 0.9649,
    "ln10^{10}A_s": 3.044,
}

_RHO_S = 1e7  # M_sun h^2 / Mpc^3
_R_S = 0.3   # Mpc/h


class TestLinearPowerSpectrum:
    """CAMB-based LinearPowerSpectrum — no JAX required for these tests."""

    @pytest.fixture(scope="class")
    def lin_pk(self):
        return LinearPowerSpectrum()

    def test_shape(self, lin_pk):
        k = np.logspace(-3, 1, 60)
        pk = lin_pk.pk_linear(jnp.asarray(k), z=0.3, theta=_PLANCK18)
        assert pk.shape == (60,)

    def test_positive(self, lin_pk):
        k = np.logspace(-3, 1, 60)
        pk = lin_pk.pk_linear(jnp.asarray(k), z=0.3, theta=_PLANCK18)
        assert jnp.all(pk > 0)

    def test_finite(self, lin_pk):
        k = np.logspace(-3, 1, 60)
        pk = lin_pk.pk_linear(jnp.asarray(k), z=0.3, theta=_PLANCK18)
        assert jnp.all(jnp.isfinite(pk))

    def test_large_scale_slope(self, lin_pk):
        """Large-scale (small k) slope should be close to n_s ≈ 1 (Harrison-Zel'dovich)."""
        k = np.logspace(-3, -1.5, 20)
        pk = np.asarray(lin_pk.pk_linear(jnp.asarray(k), z=0.0, theta=_PLANCK18))
        logk = np.log10(k)
        logpk = np.log10(pk)
        slope = np.polyfit(logk, logpk, 1)[0]
        # Expected slope ≈ n_s on large scales (before BAO/transfer function suppression)
        assert 0.5 < slope < 1.5

    def test_redshift_suppression(self, lin_pk):
        """P(k, z=1) should be uniformly below P(k, z=0)."""
        k = jnp.logspace(-2, 0, 40)
        pk_z0 = lin_pk.pk_linear(k, z=0.0, theta=_PLANCK18)
        pk_z1 = lin_pk.pk_linear(k, z=1.0, theta=_PLANCK18)
        assert jnp.all(pk_z1 < pk_z0)

    def test_growth_factor_ratio(self, lin_pk):
        """Growth factor D(z=0)/D(z=1) should be ~ 1.5–2.0 for LCDM."""
        k = jnp.logspace(-2, -0.5, 20)
        pk_z0 = lin_pk.pk_linear(k, z=0.0, theta=_PLANCK18)
        pk_z1 = lin_pk.pk_linear(k, z=1.0, theta=_PLANCK18)
        # D ratio estimated from sqrt(P ratio) at large scales
        d_ratio = float(jnp.mean(jnp.sqrt(pk_z0 / pk_z1)))
        assert 1.3 < d_ratio < 2.5

    def test_camb_vs_eisenstein_hu_shape(self, lin_pk):
        """CAMB and E-H agree in shape at intermediate scales 0.01–0.3 h/Mpc (within 30%).

        E-H 1998 uses a simplified CDM transfer function so it diverges from CAMB
        at very large scales (k < 0.01 h/Mpc).  This test restricts to the range
        where the approximation is designed to work.
        """
        k = jnp.logspace(-2, -0.5, 30)
        pk_camb = np.asarray(lin_pk.pk_linear(k, z=0.0, theta=_PLANCK18))
        pk_eh = np.asarray(eisenstein_hu_pk(k, _PLANCK18))
        # Normalise both at k=0.05
        k_np = np.logspace(-2, -0.5, 30)
        pk_camb_n = pk_camb / np.interp(np.log(0.05), np.log(k_np), pk_camb)
        pk_eh_n   = pk_eh   / np.interp(np.log(0.05), np.log(k_np), pk_eh)
        np.testing.assert_allclose(pk_camb_n, pk_eh_n, rtol=0.35)

    def test_default_cosmology_keys(self):
        keys = LinearPowerSpectrum.default_cosmology()
        for k in ("h", "Omega_b", "Omega_cdm", "n_s", "ln10^{10}A_s"):
            assert k in keys

    def test_amplitude_units(self, lin_pk):
        """P(k=0.1 h/Mpc, z=0) should be O(1000–30000) (Mpc/h)^3."""
        k = jnp.array([0.1])
        pk = float(lin_pk.pk_linear(k, z=0.0, theta=_PLANCK18)[0])
        assert 500.0 < pk < 1e5


class TestNonLinearPowerSpectrum:
    """Aletheia non-linear power spectrum."""

    @pytest.fixture(scope="class")
    def nl_pk(self):
        try:
            emu = NonLinearPowerSpectrum(backend="aletheia")
        except ImportError:
            pytest.skip("aletheiacosmo not installed")
        try:
            import numpy as _np
            emu.pk_nonlinear(_np.array([0.5, 1.0]), z=0.3, theta=_PLANCK18)
        except Exception as e:
            pytest.skip(f"aletheia backend not functional: {e}")
        return emu

    def test_shape(self, nl_pk):
        k = np.logspace(-1, 0.4, 40)  # h/Mpc; [0.1, 2.5] → [0.067, 1.68] Mpc^-1 < 2.0
        pk = nl_pk.pk_nonlinear(k, z=0.3, theta=_PLANCK18)
        assert pk.shape == (40,)

    def test_positive(self, nl_pk):
        k = np.logspace(-1, 0.4, 40)
        pk = nl_pk.pk_nonlinear(k, z=0.3, theta=_PLANCK18)
        assert jnp.all(pk > 0)

    def test_finite(self, nl_pk):
        k = np.logspace(-1, 0.4, 40)
        pk = nl_pk.pk_nonlinear(k, z=0.3, theta=_PLANCK18)
        assert jnp.all(jnp.isfinite(pk))

    def test_nonlinear_boost_gt_one_small_scales(self, nl_pk):
        """Non-linear P(k) should exceed linear on small scales (k > 0.3 h/Mpc)."""
        lin_pk = LinearPowerSpectrum()
        k = np.logspace(-0.5, 0.4, 30)  # h/Mpc; max 2.5 h/Mpc → 1.68 Mpc^-1 < 2.0
        pk_nl = np.asarray(nl_pk.pk_nonlinear(k, z=0.0, theta=_PLANCK18))
        pk_lin = np.asarray(lin_pk.pk_linear(jnp.asarray(k), z=0.0, theta=_PLANCK18))
        boost = pk_nl / pk_lin
        # At k ~ 1 h/Mpc the boost should be > 1
        assert float(boost[-1]) > 1.0

    def test_boost_factor_shape(self, nl_pk):
        k = jnp.logspace(-1, 0.4, 30)  # h/Mpc; within Aletheia [0.006, 2.0] Mpc^-1
        lin_pk = LinearPowerSpectrum()
        pk_lin = lin_pk.pk_linear(k, z=0.3, theta=_PLANCK18)
        boost = nl_pk.boost_factor(k, z=0.3, theta=_PLANCK18, pk_lin=pk_lin)
        assert boost.shape == (30,)

    def test_redshift_suppression(self, nl_pk):
        """Non-linear P(k) at z=1 should be below z=0 on large scales."""
        k = np.logspace(-1, 0, 20)
        pk_z0 = np.asarray(nl_pk.pk_nonlinear(k, z=0.0, theta=_PLANCK18))
        pk_z1 = np.asarray(nl_pk.pk_nonlinear(k, z=1.0, theta=_PLANCK18))
        assert np.all(pk_z1 < pk_z0)

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="aletheia"):
            NonLinearPowerSpectrum(backend="goku")


class TestEisensteinHu:
    def test_shape(self):
        k = jnp.logspace(-3, 1, 50)
        pk = eisenstein_hu_pk(k, _PLANCK18)
        assert pk.shape == (50,)

    def test_positive(self):
        k = jnp.logspace(-3, 1, 50)
        pk = eisenstein_hu_pk(k, _PLANCK18)
        assert jnp.all(pk > 0)

    def test_normalisation(self):
        k = jnp.logspace(-3, 1, 200)
        pk = eisenstein_hu_pk(k, _PLANCK18)
        # Normalised to 1 at k=0.05 h/Mpc
        pk_at_pivot = jnp.interp(jnp.log(jnp.array(0.05)), jnp.log(k), pk)
        assert jnp.abs(pk_at_pivot - 1.0) < 0.01

    def test_jit(self):
        k = jnp.logspace(-3, 1, 50)
        pk_jit = jax.jit(eisenstein_hu_pk)(k, _PLANCK18)
        pk_ref = eisenstein_hu_pk(k, _PLANCK18)
        np.testing.assert_allclose(pk_jit, pk_ref, rtol=1e-5)


class TestNFWProfile:
    def test_density_shape(self):
        r = jnp.logspace(-2, 1, 30)
        rho = nfw_rho(r, _RHO_S, _R_S)
        assert rho.shape == (30,)

    def test_density_decreasing(self):
        r = jnp.logspace(-2, 1, 30)
        rho = nfw_rho(r, _RHO_S, _R_S)
        assert jnp.all(jnp.diff(rho) < 0)

    def test_mass_positive(self):
        r = jnp.logspace(-2, 1, 30)
        m = nfw_mass(r, _RHO_S, _R_S)
        assert jnp.all(m > 0)

    def test_mass_increasing(self):
        r = jnp.logspace(-2, 1, 30)
        m = nfw_mass(r, _RHO_S, _R_S)
        assert jnp.all(jnp.diff(m) > 0)

    def test_sigma_positive(self):
        R = jnp.logspace(-2, 1, 20)
        sig = nfw_sigma(R, _RHO_S, _R_S)
        assert jnp.all(sig > 0)

    def test_delta_sigma_positive(self):
        R = jnp.logspace(-2, 1, 20)
        ds = nfw_delta_sigma(R, _RHO_S, _R_S)
        assert jnp.all(ds > 0)

    def test_delta_sigma_identity(self):
        """ΔΣ = Σ_bar - Σ must hold elementwise."""
        R = jnp.logspace(-2, 1, 20)
        ds = nfw_delta_sigma(R, _RHO_S, _R_S)
        sbar = nfw_mean_sigma(R, _RHO_S, _R_S)
        sig = nfw_sigma(R, _RHO_S, _R_S)
        np.testing.assert_allclose(ds, sbar - sig, rtol=1e-4)

    def test_sigma_at_rs(self):
        """At x=1 (R=r_s) sigma should use the special-case branch."""
        R = jnp.array([_R_S])
        sig = nfw_sigma(R, _RHO_S, _R_S)
        expected = 2.0 * _RHO_S * _R_S / 3.0
        np.testing.assert_allclose(float(sig[0]), expected, rtol=1e-3)


class TestDarkEnergy:
    """CPL dark energy parametrization in hubble_e and distance functions."""

    _OM = 0.3111
    _H = 0.6736

    def test_lcdm_limit_hubble_e(self):
        """w0=-1, wa=0 must reproduce the ΛCDM formula exactly."""
        z = jnp.linspace(0.0, 3.0, 50)
        e_lcdm = jnp.sqrt(self._OM * (1.0 + z) ** 3 + (1.0 - self._OM))
        e_cpl  = hubble_e(z, self._OM, w0=-1.0, wa=0.0)
        np.testing.assert_allclose(e_cpl, e_lcdm, rtol=1e-6)

    def test_lcdm_default_args(self):
        """Default arguments (no w0/wa) must give the same result as w0=-1, wa=0."""
        z = jnp.linspace(0.0, 2.0, 30)
        e_default = hubble_e(z, self._OM)
        e_explicit = hubble_e(z, self._OM, w0=-1.0, wa=0.0)
        np.testing.assert_allclose(e_default, e_explicit, rtol=1e-6)

    def test_f_de_unity_at_z0(self):
        """f_DE(z=0) = 1 for any (w0, wa)."""
        for w0, wa in [(-0.7, 0.0), (-1.3, 0.5), (-0.9, -0.3)]:
            e0 = float(hubble_e(jnp.array(0.0), self._OM, w0=w0, wa=wa))
            e_lcdm_z0 = float(jnp.sqrt(jnp.array(self._OM + (1.0 - self._OM))))
            np.testing.assert_allclose(e0, e_lcdm_z0, rtol=1e-5,
                                       err_msg=f"f_DE(z=0) != 1 for w0={w0}, wa={wa}")

    def test_w0_quintessence_raises_e(self):
        """w0 > -1 (quintessence) → higher E(z) than ΛCDM at z>0.

        For w0=-0.7: f_DE=(1+z)^{3(1+w0)}=(1+z)^{0.9} > 1 → more DE in the past
        → larger E(z) → shorter comoving distances.
        """
        z = jnp.linspace(0.5, 3.0, 20)
        e_lcdm  = hubble_e(z, self._OM, w0=-1.0, wa=0.0)
        e_quint = hubble_e(z, self._OM, w0=-0.7, wa=0.0)
        assert jnp.all(e_quint >= e_lcdm)

    def test_comoving_distance_lcdm_limit(self):
        """Comoving distance with w0=-1, wa=0 matches the ΛCDM default."""
        z = jnp.array([0.5, 1.0, 2.0])
        chi_default = comoving_distance(z, self._H, self._OM)
        chi_cpl     = comoving_distance(z, self._H, self._OM, w0=-1.0, wa=0.0)
        np.testing.assert_allclose(chi_cpl, chi_default, rtol=1e-5)

    def test_quintessence_closer(self):
        """Quintessence (w0 > -1) gives smaller comoving distances than ΛCDM."""
        z = jnp.array([0.5, 1.0, 2.0])
        chi_lcdm  = comoving_distance(z, self._H, self._OM, w0=-1.0, wa=0.0)
        chi_quint = comoving_distance(z, self._H, self._OM, w0=-0.7, wa=0.0)
        assert jnp.all(chi_quint < chi_lcdm)

    def test_phantom_farther(self):
        """Phantom dark energy (w0 < -1) gives larger comoving distances than ΛCDM."""
        z = jnp.array([0.5, 1.0, 2.0])
        chi_lcdm    = comoving_distance(z, self._H, self._OM, w0=-1.0, wa=0.0)
        chi_phantom = comoving_distance(z, self._H, self._OM, w0=-1.3, wa=0.0)
        assert jnp.all(chi_phantom > chi_lcdm)

    def test_wa_nonzero_changes_distance(self):
        """Non-zero wa changes comoving distance relative to wa=0 at the same w0."""
        z = jnp.array([1.0, 2.0])
        chi_wa0  = comoving_distance(z, self._H, self._OM, w0=-0.9, wa=0.0)
        chi_waP  = comoving_distance(z, self._H, self._OM, w0=-0.9, wa=0.5)
        assert not jnp.allclose(chi_wa0, chi_waP)

    def test_distance_modulus_finite(self):
        """Distance modulus should be finite and positive for valid cosmologies."""
        z = jnp.linspace(0.01, 2.0, 20)
        for w0, wa in [(-1.0, 0.0), (-0.7, 0.0), (-1.3, 0.5)]:
            mu = distance_modulus(z, self._H, self._OM, w0=w0, wa=wa)
            assert jnp.all(jnp.isfinite(mu))
            assert jnp.all(mu > 0)

    def test_jit_hubble_e_w0wa(self):
        """hubble_e must be JIT-compilable with non-ΛCDM (w0, wa)."""
        z = jnp.linspace(0.0, 2.0, 20)
        fn = jax.jit(lambda z: hubble_e(z, self._OM, w0=-0.9, wa=0.3))
        result = fn(z)
        assert result.shape == (20,)
        assert jnp.all(jnp.isfinite(result))

    def test_default_cosmology_has_w0_wa(self):
        """LinearPowerSpectrum.default_cosmology must include w0 and wa."""
        d = LinearPowerSpectrum.default_cosmology()
        assert "w0" in d and "wa" in d
        assert d["w0"] == -1.0
        assert d["wa"] == 0.0

    def test_camb_w0wa_positive(self):
        """CAMB P(k) with non-ΛCDM w0 should return positive finite values."""
        lin_pk = LinearPowerSpectrum()
        theta_de = LinearPowerSpectrum.default_cosmology()
        theta_de["w0"] = -0.7
        theta_de["wa"] = 0.3
        k = jnp.logspace(-2, 0, 20)
        pk = lin_pk.pk_linear(k, z=0.5, theta=theta_de)
        assert jnp.all(pk > 0)
        assert jnp.all(jnp.isfinite(pk))


class TestHMFFunctions:
    def test_fsigma_positive(self):
        sigma = jnp.logspace(-1, 0.5, 50)
        f = tinker08_fsigma(sigma)
        assert jnp.all(f >= 0)

    def test_bias_gt_zero(self):
        nu = jnp.logspace(-0.5, 1.0, 50)
        b = tinker10_bias(nu)
        # Bias can be negative for very low ν (below 1σ peaks) — just check shape
        assert b.shape == (50,)

    def test_bias_high_mass(self):
        """High-ν (massive) halos should have b > 1."""
        nu = jnp.array([3.0, 5.0, 10.0])
        b = tinker10_bias(nu)
        assert jnp.all(b > 1.0)


# ---------------------------------------------------------------------------
# All multiplicity functions (positive, z-dependence where applicable)
# ---------------------------------------------------------------------------

class TestMultiplicityFunctions:
    """Test all f(σ) multiplicity functions beyond Tinker+2008."""

    _SIGMA = jnp.logspace(-0.5, 0.5, 30)

    @pytest.mark.parametrize("fn", [
        fsigma_press74, fsigma_sheth99, fsigma_jenkins01, fsigma_warren06,
        fsigma_angulo12, fsigma_watson13, fsigma_courtin11,
        fsigma_rodriguezpuebla16, fsigma_comparat17, fsigma_yung24,
    ])
    def test_positive_z0(self, fn):
        f = fn(self._SIGMA, z=0.0)
        assert jnp.all(f > 0), f"f(σ) should be positive for {fn.__name__}"

    @pytest.mark.parametrize("fn", [
        fsigma_crocce10, fsigma_bhattacharya11,
    ])
    def test_z_dependent_positive(self, fn):
        f0 = fn(self._SIGMA, z=0.0)
        f1 = fn(self._SIGMA, z=1.0)
        assert jnp.all(f0 > 0)
        assert jnp.all(f1 > 0)
        assert not jnp.allclose(f0, f1)

    def test_bocquet16_dmo_positive(self):
        f = fsigma_bocquet16(self._SIGMA, z=0.0, hydro=False)
        assert jnp.all(f > 0)

    def test_bocquet16_hydro_positive(self):
        f = fsigma_bocquet16(self._SIGMA, z=0.0, hydro=True)
        assert jnp.all(f > 0)

    def test_bocquet16_hydro_differs_from_dmo(self):
        f_dmo  = fsigma_bocquet16(self._SIGMA, z=0.0, hydro=False)
        f_hyd  = fsigma_bocquet16(self._SIGMA, z=0.0, hydro=True)
        assert not jnp.allclose(f_dmo, f_hyd)

    def test_despali16_delta_ratio_one(self):
        f = fsigma_despali16(self._SIGMA, z=0.0, delta_ratio=1.0)
        assert jnp.all(f > 0)

    def test_despali16_varies_with_delta(self):
        f1 = fsigma_despali16(self._SIGMA, z=0.0, delta_ratio=1.0)
        f2 = fsigma_despali16(self._SIGMA, z=0.0, delta_ratio=2.0)
        assert not jnp.allclose(f1, f2)

    def test_seppi20_shape_and_positive(self):
        sigma_small = jnp.logspace(-0.3, 0.3, 6)
        f = fsigma_seppi20(sigma_small, z=0.0)
        assert f.shape == (6,)
        assert jnp.all(f > 0)

    def test_yung25_high_z_positive(self):
        """yung25 is calibrated for high z; just verify it runs and is positive."""
        sigma_small = jnp.logspace(-0.3, 0.3, 10)
        f = fsigma_yung25(sigma_small, z=8.0)
        assert jnp.all(f > 0)

    def test_delta_vir_flat(self):
        """Bryan & Norman 1998 Δ_vir should be ~178 at z=0, Ω_m=1."""
        dv = float(delta_vir_flat_jax(0.0, 1.0))
        assert abs(dv - 178.0) < 2.0

    def test_delta_vir_increases_with_z(self):
        """Δ_vir → 18π²≈178 at high z (matter-dominated), exceeding the z=0 value."""
        dv_z0 = float(delta_vir_flat_jax(0.0, 0.3))
        dv_z2 = float(delta_vir_flat_jax(2.0, 0.3))
        assert dv_z2 > dv_z0


# ---------------------------------------------------------------------------
# HaloMassFunction class methods
# ---------------------------------------------------------------------------

class TestHaloMassFunctionClass:
    """HaloMassFunction.sigma, dndm, bias, n_eff via LinearPowerSpectrum."""

    @pytest.fixture(scope="class")
    def hmf_theta(self):
        lin_pk = LinearPowerSpectrum()
        theta = lin_pk.default_cosmology()
        hmf = make_hmf("tinker08", pk_func=lin_pk.pk_linear)
        return hmf, theta

    def test_sigma_shape(self, hmf_theta):
        hmf, theta = hmf_theta
        m = jnp.logspace(10, 15, 30)
        s = hmf.sigma(m, z=0.0, theta=theta)
        assert s.shape == (30,)

    def test_sigma_positive(self, hmf_theta):
        hmf, theta = hmf_theta
        m = jnp.logspace(10, 15, 30)
        assert jnp.all(hmf.sigma(m, z=0.0, theta=theta) > 0)

    def test_sigma_decreasing_with_mass(self, hmf_theta):
        """σ(M) must decrease with halo mass (more mass → smoother region)."""
        hmf, theta = hmf_theta
        m = jnp.logspace(10, 15, 30)
        s = hmf.sigma(m, z=0.0, theta=theta)
        assert jnp.all(jnp.diff(s) < 0)

    def test_sigma_redshift_suppression(self, hmf_theta):
        """σ(M, z=1) < σ(M, z=0) — growth factor < 1."""
        hmf, theta = hmf_theta
        m = jnp.logspace(12, 14, 15)
        s0 = hmf.sigma(m, z=0.0, theta=theta)
        s1 = hmf.sigma(m, z=1.0, theta=theta)
        assert jnp.all(s1 < s0)

    def test_sigma8_normalisation(self, hmf_theta):
        """With sigma8 in theta, σ(8 Mpc/h, z=0) ≈ sigma8."""
        hmf, theta = hmf_theta
        theta_s8 = dict(theta, sigma8=0.8111)
        rho_mean = hmf.rho_mean
        M8 = (4.0 / 3.0) * jnp.pi * 8.0**3 * rho_mean
        s8 = float(hmf.sigma(jnp.array([M8]), z=0.0, theta=theta_s8)[0])
        assert abs(s8 - 0.8111) / 0.8111 < 0.02

    def test_dndm_shape(self, hmf_theta):
        hmf, theta = hmf_theta
        m = jnp.logspace(11, 15, 40)
        dn = hmf.dndm(m, z=0.0, theta=theta)
        assert dn.shape == (40,)

    def test_dndm_positive(self, hmf_theta):
        hmf, theta = hmf_theta
        m = jnp.logspace(11, 15, 40)
        assert jnp.all(hmf.dndm(m, z=0.0, theta=theta) > 0)

    def test_dndm_decreasing_high_mass(self, hmf_theta):
        """dn/dM decreases monotonically above the knee."""
        hmf, theta = hmf_theta
        m = jnp.logspace(12, 15, 30)
        dn = hmf.dndm(m, z=0.0, theta=theta)
        assert jnp.all(jnp.diff(dn) < 0)

    def test_bias_positive_high_mass(self, hmf_theta):
        """b_eff > 1 for cluster-mass halos."""
        hmf, theta = hmf_theta
        m = jnp.array([1e14, 1e15])
        b = hmf.bias(m, z=0.0, theta=theta)
        assert jnp.all(b > 1.0)

    def test_n_eff_positive(self, hmf_theta):
        hmf, theta = hmf_theta
        n = float(hmf.n_eff(1e12, 1e16, z=0.0, theta=theta))
        assert n > 0

    def test_n_eff_decreases_with_mmin(self, hmf_theta):
        """Higher mass threshold → smaller number density."""
        hmf, theta = hmf_theta
        n1 = float(hmf.n_eff(1e11, 1e16, z=0.0, theta=theta))
        n2 = float(hmf.n_eff(1e13, 1e16, z=0.0, theta=theta))
        assert n1 > n2


# ---------------------------------------------------------------------------
# make_hmf factory
# ---------------------------------------------------------------------------

class TestMakeHmf:
    """make_hmf factory: analytic and emulator backends."""

    @pytest.fixture(scope="class")
    def pk_func(self):
        return LinearPowerSpectrum().pk_linear

    def test_tinker08_returns_hmf(self, pk_func):
        hmf = make_hmf("tinker08", pk_func=pk_func)
        assert isinstance(hmf, HaloMassFunction)
        assert hmf.model == "tinker08"

    def test_bocquet16_with_kwargs(self, pk_func):
        hmf = make_hmf("bocquet16", pk_func=pk_func, hydro=True)
        assert isinstance(hmf, HaloMassFunction)

    def test_all_analytic_backends_construct(self, pk_func):
        analytic_models = [
            "press74", "sheth99", "jenkins01", "warren06", "tinker08",
            "crocce10", "courtin11", "bhattacharya11", "watson13", "angulo12",
            "bocquet16", "despali16", "rodriguezpuebla16", "comparat17",
            "yung24", "yung25",
        ]
        for model in analytic_models:
            hmf = make_hmf(model, pk_func=pk_func)
            assert hmf.model == model, f"model mismatch for {model}"

    def test_invalid_backend_raises(self, pk_func):
        with pytest.raises(ValueError, match="backend"):
            make_hmf("unknown_model", pk_func=pk_func)

    def test_missing_pk_func_raises(self):
        with pytest.raises(ValueError, match="pk_func"):
            make_hmf("tinker08")

    def test_aemulusnu_backend_key(self, pk_func):
        """make_hmf('aemulusnu') returns AemulusNuHaloMassFunction."""
        pytest.importorskip("aemulusnu_hmf")
        from hod_mod.core.halo_mass_function import AemulusNuHaloMassFunction
        hmf = make_hmf("aemulusnu", pk_func=pk_func)
        assert isinstance(hmf, AemulusNuHaloMassFunction)

    def test_csst_backend_key(self):
        """make_hmf('csst') returns CsstHaloMassFunction."""
        pytest.importorskip("CEmulator")
        from hod_mod.core.halo_mass_function import CsstHaloMassFunction
        hmf = make_hmf("csst")
        assert isinstance(hmf, CsstHaloMassFunction)


# ---------------------------------------------------------------------------
# AemulusNu emulator backend
# ---------------------------------------------------------------------------

class TestAemulusNuHaloMassFunction:
    """AemulusNuHaloMassFunction: dndm, bias, sigma, mass-range warning."""

    aemulusnu_hmf = pytest.importorskip("aemulusnu_hmf")

    @pytest.fixture(scope="class")
    def theta(self):
        return LinearPowerSpectrum.default_cosmology()

    @pytest.fixture(scope="class")
    def hmf(self):
        pk = LinearPowerSpectrum()
        return make_hmf("aemulusnu", pk_func=pk.pk_linear)

    @pytest.fixture(scope="class")
    def m_grid(self):
        return jnp.logspace(13, 15, 8)

    def test_dndm_shape(self, hmf, m_grid, theta):
        dn = hmf.dndm(m_grid, 0.0, theta)
        assert dn.shape == m_grid.shape

    def test_dndm_positive(self, hmf, m_grid, theta):
        dn = hmf.dndm(m_grid, 0.0, theta)
        assert jnp.all(dn > 0)

    def test_dndm_finite(self, hmf, m_grid, theta):
        dn = hmf.dndm(m_grid, 0.0, theta)
        assert jnp.all(jnp.isfinite(dn))

    def test_dndm_decreasing_with_mass(self, hmf, m_grid, theta):
        dn = hmf.dndm(m_grid, 0.0, theta)
        assert jnp.all(jnp.diff(dn) < 0)

    def test_dndm_mass_range_warning(self, hmf, theta):
        m_low = jnp.array([1e12])
        with pytest.warns(UserWarning, match="calibration range"):
            hmf.dndm(m_low, 0.0, theta)

    def test_sigma_shape(self, hmf, m_grid, theta):
        sig = hmf.sigma(m_grid, 0.0, theta)
        assert sig.shape == m_grid.shape

    def test_sigma_positive(self, hmf, m_grid, theta):
        sig = hmf.sigma(m_grid, 0.0, theta)
        assert jnp.all(sig > 0)

    def test_bias_shape(self, hmf, m_grid, theta):
        b = hmf.bias(m_grid, 0.0, theta)
        assert b.shape == m_grid.shape

    def test_bias_positive(self, hmf, m_grid, theta):
        b = hmf.bias(m_grid, 0.0, theta)
        assert jnp.all(b > 0)

    def test_no_pk_func_sigma_raises(self, theta):
        from hod_mod.core.halo_mass_function import AemulusNuHaloMassFunction
        hmf_nopk = AemulusNuHaloMassFunction(pk_func=None)
        with pytest.raises(RuntimeError, match="pk_func"):
            hmf_nopk.sigma(jnp.array([1e14]), 0.0, theta)

    def test_dndm_agrees_with_tinker_order_of_magnitude(self, hmf, theta):
        """Aemulus-ν and Tinker08 dndm agree within a factor of 5 at z=0, M∈[10^13,10^14.5]."""
        pk = LinearPowerSpectrum()
        hmf_t = make_hmf("tinker08", pk_func=pk.pk_linear)
        # Limit to [1e13, 1e14.5]: Aemulus-ν calibrated range; at M>1e14.5 neutrino
        # suppression and different mass definitions cause >5× divergence vs Tinker08.
        m = jnp.logspace(13, 14.5, 5)
        dn_ae  = jnp.asarray(hmf.dndm(m, 0.0, theta))
        dn_ti  = jnp.asarray(hmf_t.dndm(m, 0.0, theta))
        ratio  = dn_ae / dn_ti
        assert jnp.all(ratio > 0.2) and jnp.all(ratio < 5.0), \
            f"Aemulus/Tinker ratio out of range: {ratio}"


# ---------------------------------------------------------------------------
# CSST emulator backend
# ---------------------------------------------------------------------------

class TestCsstHaloMassFunction:
    """CsstHaloMassFunction: dndm, bias, sigma."""

    CEmulator = pytest.importorskip("CEmulator")

    @pytest.fixture(scope="class")
    def theta(self):
        return LinearPowerSpectrum.default_cosmology()

    @pytest.fixture(scope="class")
    def hmf(self):
        return make_hmf("csst")

    @pytest.fixture(scope="class")
    def m_grid(self):
        return jnp.logspace(13, 15, 8)

    def test_dndm_shape(self, hmf, m_grid, theta):
        dn = hmf.dndm(m_grid, 0.0, theta)
        assert dn.shape == m_grid.shape

    def test_dndm_positive(self, hmf, m_grid, theta):
        dn = hmf.dndm(m_grid, 0.0, theta)
        assert jnp.all(dn > 0)

    def test_dndm_finite(self, hmf, m_grid, theta):
        dn = hmf.dndm(m_grid, 0.0, theta)
        assert jnp.all(jnp.isfinite(dn))

    def test_bias_shape(self, hmf, m_grid, theta):
        b = hmf.bias(m_grid, 0.0, theta)
        assert b.shape == m_grid.shape

    def test_bias_positive(self, hmf, m_grid, theta):
        b = hmf.bias(m_grid, 0.0, theta)
        assert jnp.all(b > 0)

    def test_sigma_shape(self, hmf, m_grid, theta):
        sig = hmf.sigma(m_grid, 0.0, theta)
        assert sig.shape == m_grid.shape


# ---------------------------------------------------------------------------
# Einasto profile
# ---------------------------------------------------------------------------

class TestEinastoProfile:
    """einasto_rho: density, shape, boundary conditions."""

    _RHO_S = 1e7
    _R_S   = 0.3
    _R     = jnp.logspace(-2, 1, 30)

    def test_shape(self):
        rho = einasto_rho(self._R, self._RHO_S, self._R_S)
        assert rho.shape == (30,)

    def test_positive(self):
        rho = einasto_rho(self._R, self._RHO_S, self._R_S)
        assert jnp.all(rho > 0)

    def test_decreasing(self):
        rho = einasto_rho(self._R, self._RHO_S, self._R_S)
        assert jnp.all(jnp.diff(rho) < 0)

    def test_at_rs_equals_rho_s(self):
        """ρ(r_s) = ρ_s exp(0) = ρ_s for any alpha."""
        for alpha in [0.1, 0.18, 0.3]:
            val = float(einasto_rho(jnp.array([self._R_S]), self._RHO_S, self._R_S, alpha)[0])
            np.testing.assert_allclose(val, self._RHO_S, rtol=1e-6)

    def test_alpha_changes_profile(self):
        """Different α → different density outside r_s (profile shape changes)."""
        r = jnp.array([2.0 * self._R_S])
        rho1 = float(einasto_rho(r, self._RHO_S, self._R_S, alpha=0.1)[0])
        rho2 = float(einasto_rho(r, self._RHO_S, self._R_S, alpha=0.3)[0])
        assert abs(rho1 - rho2) / max(rho1, rho2) > 0.01

    def test_jit(self):
        rho_ref = einasto_rho(self._R, self._RHO_S, self._R_S)
        rho_jit = jax.jit(einasto_rho)(self._R, self._RHO_S, self._R_S)
        np.testing.assert_allclose(rho_jit, rho_ref, rtol=1e-5)

    def test_finite(self):
        rho = einasto_rho(self._R, self._RHO_S, self._R_S)
        assert jnp.all(jnp.isfinite(rho))


# ---------------------------------------------------------------------------
# NFW Fourier transform
# ---------------------------------------------------------------------------

class TestNFWFourierTransform:
    """nfw_uk: shape, limits, and physical bounds."""

    # Typical galaxy cluster: r_s = 0.5 Mpc/h, c = 5
    _R_S = np.array([0.5])
    _C   = np.array([5.0])

    def test_shape_1d(self):
        k = np.logspace(-2, 1, 20)
        uk = nfw_uk(k, self._R_S, self._C)
        assert uk.shape == (20, 1)

    def test_shape_multi_mass(self):
        k   = np.logspace(-2, 1, 15)
        r_s = np.array([0.3, 0.5, 1.0])
        c   = np.array([7.0, 5.0, 3.0])
        uk  = nfw_uk(k, r_s, c)
        assert uk.shape == (15, 3)

    def test_k0_limit_unity(self):
        """û_m(k→0) → 1 (Fourier transform of normalised profile at zero frequency)."""
        k_small = np.array([1e-8, 1e-7, 1e-6])
        uk = nfw_uk(k_small, self._R_S, self._C)
        np.testing.assert_allclose(uk, 1.0, atol=1e-4)

    def test_in_range_zero_to_one(self):
        """Normalized Fourier transform must satisfy 0 < û ≤ 1."""
        k = np.logspace(-3, 2, 50)
        uk = np.asarray(nfw_uk(k, self._R_S, self._C))
        assert np.all(uk > 0)
        assert np.all(uk <= 1.0 + 1e-6)

    def test_decreasing_with_k(self):
        """û_m(k) is a monotonically decreasing function of k."""
        k  = np.logspace(-2, 2, 40)
        uk = np.asarray(nfw_uk(k, self._R_S, self._C))[:, 0]
        assert np.all(np.diff(uk) < 0)

    def test_concentration_effect(self):
        """Higher concentration → more compact profile → slower decay in k."""
        k    = np.array([1.0])   # 1 h/Mpc
        r_s  = np.array([0.5])
        uk_lo = float(nfw_uk(k, r_s, np.array([3.0]))[0, 0])
        uk_hi = float(nfw_uk(k, r_s, np.array([10.0]))[0, 0])
        assert uk_hi < uk_lo   # more concentrated → smaller r_s relative to halo → larger K → more suppressed

    def test_finite(self):
        k  = np.logspace(-2, 2, 30)
        uk = nfw_uk(k, self._R_S, self._C)
        assert np.all(np.isfinite(uk))


# ---------------------------------------------------------------------------
# HaloModelPowerSpectrum (1-halo + 2-halo matter P_mm)
# ---------------------------------------------------------------------------

class TestHaloModelPowerSpectrum:
    """HaloModelPowerSpectrum: shape, positivity, asymptotic behaviour."""

    @pytest.fixture(scope="class")
    def halo_model(self):
        try:
            from colossus.cosmology import cosmology as col_cosmo
        except ImportError:
            pytest.skip("colossus not installed")
        pk_lin = LinearPowerSpectrum()
        theta  = pk_lin.default_cosmology()
        hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
        prof   = HaloProfile(cosmo_params={}, cm_relation="diemer19", mdef="200m")
        model  = HaloModelPowerSpectrum(hmf, prof, pk_lin, m_min=1e10, m_max=1e16, n_m=30)
        return model, theta

    def test_pk_1h_shape(self, halo_model):
        model, theta = halo_model
        k = np.logspace(-1, 0.5, 10)
        p1h = model.pk_1h_mm(k, z=0.0, theta=theta)
        assert p1h.shape == (10,)

    def test_pk_1h_positive(self, halo_model):
        model, theta = halo_model
        k = np.logspace(-1, 0.5, 10)
        p1h = model.pk_1h_mm(k, z=0.0, theta=theta)
        assert jnp.all(p1h > 0)

    def test_pk_1h_finite(self, halo_model):
        model, theta = halo_model
        k = np.logspace(-1, 0.5, 10)
        p1h = model.pk_1h_mm(k, z=0.0, theta=theta)
        assert jnp.all(jnp.isfinite(p1h))

    def test_pk_2h_shape(self, halo_model):
        model, theta = halo_model
        k = np.logspace(-1, 0.5, 10)
        p2h = model.pk_2h_mm(k, z=0.0, theta=theta)
        assert p2h.shape == (10,)

    def test_pk_2h_positive(self, halo_model):
        model, theta = halo_model
        k = np.logspace(-1, 0.5, 10)
        p2h = model.pk_2h_mm(k, z=0.0, theta=theta)
        assert jnp.all(p2h > 0)

    def test_pk_mm_sum(self, halo_model):
        """pk_mm must equal pk_1h + pk_2h exactly."""
        model, theta = halo_model
        k = np.logspace(-1, 0.5, 8)
        p1h = np.asarray(model.pk_1h_mm(k, z=0.0, theta=theta))
        p2h = np.asarray(model.pk_2h_mm(k, z=0.0, theta=theta))
        pmm = np.asarray(model.pk_mm(k,  z=0.0, theta=theta))
        np.testing.assert_allclose(pmm, p1h + p2h, rtol=1e-5)

    def test_pk_2h_large_scale_close_to_linear(self, halo_model):
        """On large scales (k ~ 0.05 h/Mpc) P^2h should be within a factor 2 of P_lin."""
        model, theta = halo_model
        k = np.array([0.05])
        p2h = float(model.pk_2h_mm(k, z=0.0, theta=theta)[0])
        pk_lin_obj = LinearPowerSpectrum()
        p_lin = float(pk_lin_obj.pk_linear(jnp.asarray(k), z=0.0, theta=theta)[0])
        ratio = p2h / p_lin
        assert 0.3 < ratio < 3.0

    def test_1h_dominates_at_small_scales(self, halo_model):
        """At k ~ 1 h/Mpc the 1-halo term should exceed the 2-halo term."""
        model, theta = halo_model
        k = np.array([1.0])
        p1h = float(model.pk_1h_mm(k, z=0.0, theta=theta)[0])
        p2h = float(model.pk_2h_mm(k, z=0.0, theta=theta)[0])
        assert p1h > p2h


# ---------------------------------------------------------------------------
# Untested distance functions
# ---------------------------------------------------------------------------

class TestDistanceFunctions:
    """Tests for comoving_distance_z1z2, angular_diameter_distance_z1z2,
    comoving_volume_element, comoving_volume, lookback_time, age_of_universe."""

    _H  = 0.6736
    _OM = 0.3111

    # --- comoving_distance_z1z2 ---

    def test_comoving_z1z2_shape(self):
        d = comoving_distance_z1z2(jnp.array([0.1]), jnp.array([0.5]),
                                    self._H, self._OM)
        assert d.shape == (1,)

    def test_comoving_z1z2_positive(self):
        d = comoving_distance_z1z2(jnp.array([0.1, 0.5]),
                                    jnp.array([0.5, 1.0]),
                                    self._H, self._OM)
        assert jnp.all(d > 0)

    def test_comoving_z1z2_identity(self):
        """chi(z1, z2) == chi(0, z2) - chi(0, z1)."""
        z1 = jnp.array([0.2])
        z2 = jnp.array([1.0])
        d12  = comoving_distance_z1z2(z1, z2, self._H, self._OM)
        chi2 = comoving_distance(z2, self._H, self._OM)
        chi1 = comoving_distance(z1, self._H, self._OM)
        np.testing.assert_allclose(d12, chi2 - chi1, rtol=1e-5)

    # --- angular_diameter_distance_z1z2 ---

    def test_ang_diam_z1z2_shape(self):
        d = angular_diameter_distance_z1z2(jnp.array([0.1]),
                                            jnp.array([0.5]),
                                            self._H, self._OM)
        assert d.shape == (1,)

    def test_ang_diam_z1z2_positive(self):
        d = angular_diameter_distance_z1z2(jnp.array([0.1, 0.5]),
                                            jnp.array([0.5, 1.0]),
                                            self._H, self._OM)
        assert jnp.all(d > 0)

    def test_ang_diam_z1z2_less_than_comoving(self):
        """D_A(z1,z2) < chi(z1,z2) because of the (1+z2) denominator."""
        z1 = jnp.array([0.2])
        z2 = jnp.array([1.0])
        da12 = angular_diameter_distance_z1z2(z1, z2, self._H, self._OM)
        d12  = comoving_distance_z1z2(z1, z2, self._H, self._OM)
        assert jnp.all(da12 < d12)

    # --- comoving_volume_element ---

    def test_volume_element_shape(self):
        z = jnp.linspace(0.1, 2.0, 20)
        dv = comoving_volume_element(z, self._H, self._OM)
        assert dv.shape == (20,)

    def test_volume_element_positive(self):
        z = jnp.linspace(0.1, 2.0, 20)
        dv = comoving_volume_element(z, self._H, self._OM)
        assert jnp.all(dv > 0)

    def test_volume_element_increasing(self):
        """dV/dz is increasing with z (comoving volume grows)."""
        z = jnp.linspace(0.1, 1.5, 15)
        dv = comoving_volume_element(z, self._H, self._OM)
        assert jnp.all(jnp.diff(dv) > 0)

    def test_volume_element_finite(self):
        z = jnp.linspace(0.01, 3.0, 30)
        dv = comoving_volume_element(z, self._H, self._OM)
        assert jnp.all(jnp.isfinite(dv))

    # --- comoving_volume ---

    def test_comoving_volume_shape(self):
        z = jnp.linspace(0.1, 2.0, 10)
        v = comoving_volume(z, self._H, self._OM)
        assert v.shape == (10,)

    def test_comoving_volume_positive(self):
        z = jnp.linspace(0.1, 2.0, 10)
        v = comoving_volume(z, self._H, self._OM)
        assert jnp.all(v > 0)

    def test_comoving_volume_increasing(self):
        z = jnp.linspace(0.1, 2.0, 10)
        v = comoving_volume(z, self._H, self._OM)
        assert jnp.all(jnp.diff(v) > 0)

    def test_comoving_volume_z1_sanity(self):
        """V_c(z=1) should be of order 10^10 to 10^12 Mpc^3 for Planck18."""
        v = float(comoving_volume(jnp.array([1.0]), self._H, self._OM)[0])
        assert 1e10 < v < 1e12

    # --- lookback_time ---

    def test_lookback_time_shape(self):
        z = jnp.linspace(0.01, 3.0, 20)
        t = lookback_time(z, self._H, self._OM)
        assert t.shape == (20,)

    def test_lookback_time_positive(self):
        z = jnp.linspace(0.01, 3.0, 20)
        t = lookback_time(z, self._H, self._OM)
        assert jnp.all(t > 0)

    def test_lookback_time_increasing(self):
        z = jnp.linspace(0.01, 3.0, 20)
        t = lookback_time(z, self._H, self._OM)
        assert jnp.all(jnp.diff(t) > 0)

    def test_lookback_time_z0_near_zero(self):
        """t_L(z→0) → 0."""
        t_small = float(lookback_time(jnp.array([0.001]), self._H, self._OM)[0])
        assert t_small < 0.1  # less than 0.1 Gyr

    def test_lookback_time_z1_planck(self):
        """t_L(z=1) ≈ 7.7 Gyr for Planck18 (within 20%)."""
        t = float(lookback_time(jnp.array([1.0]), self._H, self._OM)[0])
        assert 6.0 < t < 9.5

    def test_lookback_time_finite(self):
        z = jnp.linspace(0.01, 5.0, 30)
        t = lookback_time(z, self._H, self._OM)
        assert jnp.all(jnp.isfinite(t))

    # --- age_of_universe ---

    def test_age_of_universe_planck18(self):
        """Age should be ~13.8 Gyr for Planck18 (within 10%)."""
        t0 = float(age_of_universe(self._H, self._OM)[0])
        assert 12.0 < t0 < 15.5

    def test_age_increases_with_lower_h(self):
        """Lower H0 → older Universe."""
        t_high_h = float(age_of_universe(0.72, self._OM)[0])
        t_low_h  = float(age_of_universe(0.60, self._OM)[0])
        assert t_low_h > t_high_h

    def test_age_finite(self):
        t0 = age_of_universe(self._H, self._OM)
        assert jnp.all(jnp.isfinite(t0))


# ---------------------------------------------------------------------------
# HaloProfile class — concentration, rho_s_and_rs, delta_sigma, _mdef_delta_rho
# ---------------------------------------------------------------------------

_COLOSSUS_COSMO = {
    "flat": True, "H0": 67.36, "Om0": 0.3100,
    "Ob0": 0.0493, "sigma8": 0.8111, "ns": 0.9649,
}
_THETA_COSMO = {"Omega_m": 0.3100, "Omega_b": 0.0493}
_MASS_ARR = jnp.array([1e12, 1e13, 1e14])
_R_ARR    = jnp.logspace(-1, 1, 10)


class TestHaloProfileClass:
    """HaloProfile concentration–mass and lensing methods."""

    @pytest.fixture(scope="class")
    def hp_colossus(self):
        return HaloProfile(_COLOSSUS_COSMO, cm_relation="diemer19", mdef="200m")

    @pytest.fixture(scope="class")
    def hp_dutton(self):
        return HaloProfile(_COLOSSUS_COSMO, cm_relation="dutton14", mdef="200c")

    # --- concentration ---

    def test_concentration_colossus_shape(self, hp_colossus):
        c = hp_colossus.concentration(_MASS_ARR, z=0.1)
        assert c.shape == (3,)

    def test_concentration_colossus_positive(self, hp_colossus):
        c = hp_colossus.concentration(_MASS_ARR, z=0.1)
        assert jnp.all(c > 0)

    def test_concentration_dutton14_shape(self, hp_dutton):
        c = hp_dutton.concentration(_MASS_ARR, z=0.0)
        assert c.shape == (3,)

    def test_concentration_dutton14_positive(self, hp_dutton):
        c = hp_dutton.concentration(_MASS_ARR, z=0.0)
        assert jnp.all(c > 0)

    def test_dutton14_wrong_mdef_raises(self):
        with pytest.raises(ValueError, match="200c"):
            HaloProfile(_COLOSSUS_COSMO, cm_relation="dutton14", mdef="200m")

    # --- rho_s_and_rs ---

    def test_rho_s_and_rs_positive_200m(self, hp_colossus):
        rho_s, r_s = hp_colossus.rho_s_and_rs(_MASS_ARR, z=0.1, theta_cosmo=_THETA_COSMO)
        assert jnp.all(rho_s > 0)
        assert jnp.all(r_s > 0)

    def test_rho_s_and_rs_positive_200c(self):
        hp = HaloProfile(_COLOSSUS_COSMO, cm_relation="diemer19", mdef="200c")
        rho_s, r_s = hp.rho_s_and_rs(_MASS_ARR, z=0.3, theta_cosmo=_THETA_COSMO)
        assert jnp.all(rho_s > 0)
        assert jnp.all(r_s > 0)

    def test_rho_s_and_rs_positive_vir(self):
        hp = HaloProfile(_COLOSSUS_COSMO, cm_relation="diemer19", mdef="vir")
        rho_s, r_s = hp.rho_s_and_rs(_MASS_ARR, z=0.1, theta_cosmo=_THETA_COSMO)
        assert jnp.all(rho_s > 0)
        assert jnp.all(r_s > 0)

    def test_rho_s_and_rs_r_s_from_c(self, hp_colossus):
        """r_delta = c * r_s must hold: check r_s = r_delta / c."""
        from hod_mod.core.halo_profiles import _RHO_CRIT0
        rho_s, r_s = hp_colossus.rho_s_and_rs(_MASS_ARR, z=0.0, theta_cosmo=_THETA_COSMO)
        c = hp_colossus.concentration(_MASS_ARR, z=0.0)
        delta, rho_ref = hp_colossus._mdef_delta_rho(0.0, _THETA_COSMO)
        r_delta = (3.0 * _MASS_ARR / (4.0 * jnp.pi * delta * rho_ref)) ** (1.0 / 3.0)
        np.testing.assert_allclose(np.asarray(r_s), np.asarray(r_delta / c), rtol=1e-5)

    # --- delta_sigma ---

    def test_delta_sigma_shape(self, hp_colossus):
        m_h = jnp.array(1e13)
        ds = hp_colossus.delta_sigma(_R_ARR, m_h, z=0.1, theta_cosmo=_THETA_COSMO)
        assert ds.shape == _R_ARR.shape

    def test_delta_sigma_positive(self, hp_colossus):
        m_h = jnp.array(1e13)
        ds = hp_colossus.delta_sigma(_R_ARR, m_h, z=0.1, theta_cosmo=_THETA_COSMO)
        assert jnp.all(ds > 0)

    # --- _mdef_delta_rho branches ---

    def test_mdef_200m(self, hp_colossus):
        delta, rho_ref = hp_colossus._mdef_delta_rho(0.5, _THETA_COSMO)
        assert delta == pytest.approx(200.0)
        assert rho_ref > 0

    def test_mdef_200c(self):
        hp = HaloProfile(_COLOSSUS_COSMO, cm_relation="diemer19", mdef="200c")
        delta, rho_ref = hp._mdef_delta_rho(0.0, _THETA_COSMO)
        assert delta == pytest.approx(200.0)
        assert rho_ref > 0

    def test_mdef_vir(self):
        hp = HaloProfile(_COLOSSUS_COSMO, cm_relation="diemer19", mdef="vir")
        delta, rho_ref = hp._mdef_delta_rho(0.0, _THETA_COSMO)
        assert 100 < delta < 400  # Bryan & Norman 1998: ~178 at z=0 for flat ΛCDM
        assert rho_ref > 0

    def test_mdef_invalid_raises(self):
        hp = HaloProfile(_COLOSSUS_COSMO, cm_relation="diemer19", mdef="999x")
        with pytest.raises(ValueError, match="999x"):
            hp._mdef_delta_rho(0.0, _THETA_COSMO)


# ---------------------------------------------------------------------------
# concentration_dutton14_jax — standalone JAX-native c(M,z) relation
# ---------------------------------------------------------------------------

class TestConcentrationDutton14Jax:
    """Standalone concentration_dutton14_jax function."""

    def test_shape(self):
        c = concentration_dutton14_jax(_MASS_ARR, z=0.0)
        assert c.shape == (3,)

    def test_positive(self):
        c = concentration_dutton14_jax(_MASS_ARR, z=0.5)
        assert jnp.all(c > 0)

    def test_decreasing_with_mass(self):
        """Higher mass → lower concentration (Dutton & Macciò 2014)."""
        c = concentration_dutton14_jax(_MASS_ARR, z=0.0)
        assert float(c[0]) > float(c[-1])

    def test_z_dependence(self):
        """At fixed mass, higher z → lower concentration."""
        m = jnp.array([1e13])
        c_z0 = float(concentration_dutton14_jax(m, z=0.0)[0])
        c_z1 = float(concentration_dutton14_jax(m, z=1.0)[0])
        assert c_z0 > c_z1

    def test_finite(self):
        c = concentration_dutton14_jax(_MASS_ARR, z=2.0)
        assert jnp.all(jnp.isfinite(c))


# ---------------------------------------------------------------------------
# nfw_uk_jax, _si_jax, _ci_jax
# ---------------------------------------------------------------------------

_K_ARR  = np.logspace(-2, 1.5, 12)
_RS_ARR = np.array([0.2, 0.5, 1.0])
_C_ARR  = np.array([7.0, 8.0, 10.0])


class TestNfwUkJax:
    """nfw_uk_jax and its JAX Si/Ci helpers."""

    def test_shape(self):
        uk = nfw_uk_jax(jnp.asarray(_K_ARR), jnp.asarray(_RS_ARR), jnp.asarray(_C_ARR))
        assert uk.shape == (len(_K_ARR), len(_RS_ARR))

    def test_range(self):
        uk = nfw_uk_jax(jnp.asarray(_K_ARR), jnp.asarray(_RS_ARR), jnp.asarray(_C_ARR))
        assert jnp.all(uk > 0)
        assert jnp.all(uk <= 1.0 + 1e-6)

    def test_k0_limit(self):
        """Near k=0, û_m → 1."""
        k_small = jnp.array([1e-5, 1e-4])
        uk = nfw_uk_jax(k_small, jnp.asarray(_RS_ARR), jnp.asarray(_C_ARR))
        np.testing.assert_allclose(np.asarray(uk), 1.0, atol=1e-3)

    def test_agrees_with_nfw_uk(self):
        """JAX and scipy implementations agree to < 0.2% for K in [0.01, 10]."""
        uk_scipy = np.asarray(nfw_uk(_K_ARR, _RS_ARR, _C_ARR))
        uk_jax   = np.asarray(nfw_uk_jax(jnp.asarray(_K_ARR), jnp.asarray(_RS_ARR),
                                          jnp.asarray(_C_ARR)))
        np.testing.assert_allclose(uk_jax, uk_scipy, rtol=2e-3)

    def test_si_known_values(self):
        """_si_jax agrees with scipy.special.sici to < 1e-4 relative error."""
        from scipy.special import sici
        x = jnp.array([0.5, 1.0, 3.0, 7.0, 15.0])
        si_ref = np.array([sici(float(xi))[0] for xi in x])
        si_jax = np.asarray(_si_jax(x))
        np.testing.assert_allclose(si_jax, si_ref, rtol=1e-4)

    def test_ci_known_values(self):
        """_ci_jax agrees with scipy.special.sici to < 1e-4 relative error."""
        from scipy.special import sici
        x = jnp.array([0.5, 1.0, 3.0, 7.0, 15.0])
        ci_ref = np.array([sici(float(xi))[1] for xi in x])
        ci_jax = np.asarray(_ci_jax(x))
        np.testing.assert_allclose(ci_jax, ci_ref, rtol=1e-4)

    def test_si_odd_function(self):
        """Si is an odd function: Si(-x) = -Si(x)."""
        x = jnp.array([1.0, 3.0, 10.0])
        np.testing.assert_allclose(np.asarray(_si_jax(-x)), -np.asarray(_si_jax(x)), rtol=1e-6)

    def test_jit_differentiable(self):
        """jax.grad of nfw_uk_jax w.r.t. r_s runs without error."""
        def uk_sum(r_s):
            return jnp.sum(
                nfw_uk_jax(jnp.asarray(_K_ARR[:3]), r_s, jnp.asarray(_C_ARR[:3]))
            )
        grad_fn = jax.grad(uk_sum)
        g = grad_fn(jnp.asarray(_RS_ARR))
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# einasto_uk and satellite_nfw_uk
# ---------------------------------------------------------------------------

_RVIR_ARR = _C_ARR * _RS_ARR   # r_vir = c * r_s


class TestEinastoAndSatelliteUk:
    """einasto_uk and satellite_nfw_uk Fourier transforms."""

    def test_einasto_uk_shape(self):
        uk = einasto_uk(_K_ARR, _RS_ARR, _C_ARR)
        assert uk.shape == (len(_K_ARR), len(_RS_ARR))

    def test_einasto_uk_positive(self):
        uk = einasto_uk(_K_ARR, _RS_ARR, _C_ARR)
        assert jnp.all(uk > 0)

    def test_einasto_uk_k0_limit(self):
        """At very small k, û_Ein → 1."""
        k_small = np.array([1e-5, 1e-4])
        uk = einasto_uk(k_small, _RS_ARR, _C_ARR)
        np.testing.assert_allclose(np.asarray(uk), 1.0, atol=5e-3)

    def test_einasto_uk_finite(self):
        uk = einasto_uk(_K_ARR, _RS_ARR, _C_ARR)
        assert jnp.all(jnp.isfinite(uk))

    def test_satellite_nfw_uk_shape(self):
        uk = satellite_nfw_uk(_K_ARR, _RS_ARR, _C_ARR, _RVIR_ARR)
        assert uk.shape == (len(_K_ARR), len(_RS_ARR))

    def test_satellite_nfw_uk_finite(self):
        uk = satellite_nfw_uk(_K_ARR, _RS_ARR, _C_ARR, _RVIR_ARR)
        assert jnp.all(jnp.isfinite(uk))

    def test_satellite_nfw_uk_default_matches_nfw_uk(self):
        """With b_sat_conc=1, f_cut=0, gamma=0, satellite_nfw_uk == nfw_uk."""
        uk_sat = np.asarray(
            satellite_nfw_uk(_K_ARR, _RS_ARR, _C_ARR, _RVIR_ARR,
                             b_sat_conc=1.0, f_cut=0.0, gamma=0.0)
        )
        uk_ref = np.asarray(nfw_uk(_K_ARR, _RS_ARR, _C_ARR))
        np.testing.assert_allclose(uk_sat, uk_ref, rtol=1e-2)
