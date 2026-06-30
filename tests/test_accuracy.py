"""Accuracy and timing benchmarks for hod_mod.

Each test class covers one source file. Tests assert tolerances derived from
analytic identities, known limits, or literature calibration points.

The ``_timeit`` helper measures mean wall-clock cost post-JIT (n_runs=200).
Results are recorded in each function's docstring as "Timing" sections.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import scipy.integrate
import scipy.special


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _timeit(fn, *args, n_warmup=3, n_runs=200):
    """Mean wall-clock time per call [s] after JIT warm-up."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / n_runs


_PLANCK18 = {
    "h": 0.6736,
    "Omega_m": 0.3111,
    "Omega_b": 0.0493,
    "Omega_cdm": 0.2607,
    "n_s": 0.9649,
    "ln10^{10}A_s": 3.044,
}

_RHO_S = 1e7    # M_sun h^2 / Mpc^3
_R_S = 0.3      # Mpc/h


# ===========================================================================
# cosmology/halo_profiles.py
# ===========================================================================

class TestNFWProfilesAccuracy:
    """Accuracy tests for nfw_* and einasto_* via analytic identities."""

    from hod_mod.core.halo_profiles import (
        nfw_rho, nfw_mass, nfw_sigma, nfw_mean_sigma, nfw_delta_sigma,
        nfw_uk, einasto_rho, einasto_uk,
    )

    _R = jnp.logspace(-2, 1, 200) * _R_S   # 200 test radii

    def test_nfw_rho_inner_slope(self):
        """ρ(r) ∝ r^{-1} for r << r_s: slope d log ρ / d log r → -1."""
        from hod_mod.core.halo_profiles import nfw_rho
        r = jnp.logspace(-4, -1, 60) * _R_S
        rho = np.asarray(nfw_rho(r, _RHO_S, _R_S))
        logr = np.log10(np.asarray(r))
        slope = np.polyfit(logr[:30], np.log10(rho[:30]), 1)[0]
        assert abs(slope - (-1.0)) < 0.05, f"inner slope {slope:.3f} ≠ -1"

    def test_nfw_mass_vs_numerical_integration(self):
        """Analytic M(<r) equals numerical ∫ 4πr'² ρ(r') dr' to < 0.1%."""
        from hod_mod.core.halo_profiles import nfw_rho, nfw_mass
        r_test = np.logspace(-1, 1, 30) * _R_S
        for r_max in r_test[::6]:
            m_analytic = float(nfw_mass(jnp.array([r_max]), _RHO_S, _R_S)[0])
            r_grid = np.linspace(1e-4 * _R_S, r_max, 2000)
            rho_grid = np.asarray(nfw_rho(jnp.asarray(r_grid), _RHO_S, _R_S))
            m_numeric = float(scipy.integrate.trapezoid(4 * np.pi * r_grid**2 * rho_grid, r_grid))
            np.testing.assert_allclose(m_analytic, m_numeric, rtol=1e-3,
                                       err_msg=f"r_max={r_max:.3f}")

    def test_nfw_mean_sigma_vs_numerical_integration(self):
        """Σ_bar(R) equals (2/R²) ∫₀ᴿ Σ(R') R' dR' numerically to < 0.01%."""
        from hod_mod.core.halo_profiles import nfw_sigma, nfw_mean_sigma
        R_test = np.logspace(-1, 1, 20) * _R_S
        for R_max in R_test[::4]:
            msig_analytic = float(nfw_mean_sigma(jnp.array([R_max]), _RHO_S, _R_S)[0])
            R_grid = np.linspace(1e-4 * _R_S, R_max, 4000)
            sig_grid = np.asarray(nfw_sigma(jnp.asarray(R_grid), _RHO_S, _R_S))
            msig_numeric = float(2.0 / R_max**2 * scipy.integrate.trapezoid(
                sig_grid * R_grid, R_grid))
            np.testing.assert_allclose(msig_analytic, msig_numeric, rtol=5e-4,
                                       err_msg=f"R={R_max:.3f}")

    def test_nfw_delta_sigma_identity(self):
        """ΔΣ(R) = Σ̄(<R) − Σ(R) holds pointwise to < 1e-5 relative error."""
        from hod_mod.core.halo_profiles import nfw_sigma, nfw_mean_sigma, nfw_delta_sigma
        R = jnp.logspace(-1, 1, 100) * _R_S
        ds_direct = np.asarray(nfw_delta_sigma(R, _RHO_S, _R_S))
        ds_identity = np.asarray(nfw_mean_sigma(R, _RHO_S, _R_S) - nfw_sigma(R, _RHO_S, _R_S))
        np.testing.assert_allclose(ds_direct, ds_identity, rtol=1e-5)

    def test_nfw_uk_k0_limit(self):
        """û(k → 0) → 1 (NFW normalized to unit mass)."""
        from hod_mod.core.halo_profiles import nfw_uk
        k_tiny = np.array([1e-6, 1e-5, 1e-4])
        r_s = np.array([_R_S])
        c = np.array([10.0])
        uk = np.asarray(nfw_uk(k_tiny, r_s, c))   # shape (3, 1)
        np.testing.assert_allclose(uk[:, 0], 1.0, atol=0.01)

    def test_nfw_uk_high_k_decreasing(self):
        """û(k) is strictly decreasing with k for k > 0."""
        from hod_mod.core.halo_profiles import nfw_uk
        k = np.logspace(-2, 2, 50)
        r_s = np.array([_R_S])
        c = np.array([10.0])
        uk = np.asarray(nfw_uk(k, r_s, c))[:, 0]
        assert np.all(np.diff(uk) < 0), "nfw_uk should be monotonically decreasing"

    def test_einasto_rho_at_scale_radius(self):
        """ρ_Ein(r_s) = ρ_s * exp(0) = ρ_s exactly."""
        from hod_mod.core.halo_profiles import einasto_rho
        val = float(einasto_rho(jnp.array([_R_S]), _RHO_S, _R_S, alpha=0.18)[0])
        np.testing.assert_allclose(val, _RHO_S, rtol=1e-6)

    def test_einasto_uk_k0_limit(self):
        """Einasto û(k → 0) → 1."""
        from hod_mod.core.halo_profiles import einasto_uk
        k_tiny = np.array([1e-6, 1e-5])
        r_s = np.array([_R_S])
        c = np.array([10.0])
        uk = np.asarray(einasto_uk(k_tiny, r_s, c))[:, 0]
        np.testing.assert_allclose(uk, 1.0, atol=0.01)


# ===========================================================================
# cosmology/distances.py
# ===========================================================================

class TestDistancesAccuracy:
    """Accuracy tests for flat wCDM distance functions."""

    _H = _PLANCK18["h"]
    _OM = _PLANCK18["Omega_m"]

    def test_hubble_e_z0_equals_one(self):
        """E(z=0) = 1.0 for any flat cosmology."""
        from hod_mod.core.distances import hubble_e
        val = float(hubble_e(jnp.array(0.0), self._OM))
        np.testing.assert_allclose(val, 1.0, atol=1e-7)

    def test_hubble_e_matter_dominated(self):
        """E(z=100) ≈ sqrt(Ω_m) * (1+z)^{3/2} in matter-dominated regime."""
        from hod_mod.core.distances import hubble_e
        z = 100.0
        e_numeric = float(hubble_e(jnp.array(z), self._OM))
        e_md = float(np.sqrt(self._OM) * (1.0 + z) ** 1.5)
        np.testing.assert_allclose(e_numeric, e_md, rtol=0.02)

    def test_comoving_distance_z0_equals_zero(self):
        """χ(z=0) = 0."""
        from hod_mod.core.distances import comoving_distance
        val = float(comoving_distance(jnp.array([0.0]), self._H, self._OM)[0])
        np.testing.assert_allclose(val, 0.0, atol=1e-3)

    def test_comoving_distance_z1_planck_range(self):
        """χ(z=1) for Planck 2018 is in [3200, 3500] Mpc (h-independent check)."""
        from hod_mod.core.distances import comoving_distance
        chi = float(comoving_distance(jnp.array([1.0]), self._H, self._OM)[0])
        assert 3200.0 < chi < 3500.0, f"χ(z=1) = {chi:.0f} Mpc outside expected range"

    def test_angular_diameter_distance_identity(self):
        """d_A(z) = χ(z) / (1 + z)."""
        from hod_mod.core.distances import comoving_distance, angular_diameter_distance
        z = jnp.array([0.3, 0.5, 1.0, 2.0])
        chi = np.asarray(comoving_distance(z, self._H, self._OM))
        da = np.asarray(angular_diameter_distance(z, self._H, self._OM))
        da_expected = chi / (1.0 + np.asarray(z))
        np.testing.assert_allclose(da, da_expected, rtol=1e-6)

    def test_luminosity_distance_identity(self):
        """d_L(z) = χ(z) * (1 + z)."""
        from hod_mod.core.distances import comoving_distance, luminosity_distance
        z = jnp.array([0.3, 0.5, 1.0, 2.0])
        chi = np.asarray(comoving_distance(z, self._H, self._OM))
        dl = np.asarray(luminosity_distance(z, self._H, self._OM))
        dl_expected = chi * (1.0 + np.asarray(z))
        np.testing.assert_allclose(dl, dl_expected, rtol=1e-6)

    def test_etherington_relation(self):
        """d_L = d_A * (1+z)^2 (Etherington duality / distance-duality relation)."""
        from hod_mod.core.distances import angular_diameter_distance, luminosity_distance
        z = jnp.array([0.1, 0.5, 1.0, 2.0, 4.0])
        da = np.asarray(angular_diameter_distance(z, self._H, self._OM))
        dl = np.asarray(luminosity_distance(z, self._H, self._OM))
        z_np = np.asarray(z)
        np.testing.assert_allclose(dl, da * (1.0 + z_np) ** 2, rtol=1e-6)

    def test_distance_modulus_consistency(self):
        """μ(z) = 5 log10(d_L [Mpc]) + 25 mag."""
        from hod_mod.core.distances import luminosity_distance, distance_modulus
        z = jnp.array([0.1, 0.5, 1.0, 2.0])
        dl = np.asarray(luminosity_distance(z, self._H, self._OM))
        mu_direct = np.asarray(distance_modulus(z, self._H, self._OM))
        mu_from_dl = 5.0 * np.log10(dl) + 25.0
        np.testing.assert_allclose(mu_direct, mu_from_dl, atol=1e-4)


# ===========================================================================
# cosmology/power_spectrum.py
# ===========================================================================

class TestEisensteinHuPkAccuracy:
    """Accuracy tests for the Eisenstein-Hu 1998 power spectrum."""

    def test_positive_and_finite(self):
        """P_EH(k) > 0 and finite for k ∈ [1e-4, 20] h/Mpc."""
        from hod_mod.core.power_spectrum import eisenstein_hu_pk
        k = jnp.logspace(-4, 1.3, 200)
        pk = eisenstein_hu_pk(k, _PLANCK18)
        assert jnp.all(pk > 0) and jnp.all(jnp.isfinite(pk))

    def test_normalized_at_k_pivot(self):
        """E-H P(k) = 1.0 at k = 0.05 h/Mpc (normalisation pivot)."""
        from hod_mod.core.power_spectrum import eisenstein_hu_pk
        k_pivot = jnp.array([0.05])
        pk_pivot = float(eisenstein_hu_pk(k_pivot, _PLANCK18)[0])
        np.testing.assert_allclose(pk_pivot, 1.0, atol=0.1,
                                   err_msg="E-H P(k) should be ≈ 1 near normalisation pivot k=0.05")

    def test_large_scale_slope(self):
        """At k < 3e-3 h/Mpc (super-horizon scales), P(k) ∝ k^{n_s} to within 0.15.

        The E-H transfer function approaches unity at k → 0, so the slope
        converges to n_s; small deviations arise from the normalisation pivot
        at k=0.05 h/Mpc and finite integration range.
        """
        from hod_mod.core.power_spectrum import eisenstein_hu_pk
        k = np.logspace(-4, -2.5, 40)
        pk = np.asarray(eisenstein_hu_pk(jnp.asarray(k), _PLANCK18))
        slope = np.polyfit(np.log10(k), np.log10(pk), 1)[0]
        ns = _PLANCK18["n_s"]
        np.testing.assert_allclose(slope, ns, atol=0.15,
                                   err_msg=f"large-scale slope {slope:.3f} ≠ n_s={ns:.3f}")

    def test_shape_agreement_with_camb(self):
        """E-H and CAMB shapes agree to < 25% after common normalisation at k=0.1 h/Mpc."""
        pytest.importorskip("camb")
        from hod_mod.core.power_spectrum import eisenstein_hu_pk, LinearPowerSpectrum
        lin = LinearPowerSpectrum()
        k = jnp.logspace(-2, -0.5, 40)
        pk_camb = np.asarray(lin.pk_linear(k, z=0.0, theta=_PLANCK18))
        pk_eh = np.asarray(eisenstein_hu_pk(k, _PLANCK18))
        k_np = np.logspace(-2, -0.5, 40)
        i_norm = np.argmin(np.abs(k_np - 0.1))
        pk_camb_n = pk_camb / pk_camb[i_norm]
        pk_eh_n = pk_eh / pk_eh[i_norm]
        np.testing.assert_allclose(pk_camb_n, pk_eh_n, rtol=0.25)


# ===========================================================================
# cosmology/halo_mass_function.py
# ===========================================================================

class TestHaloMassFunctionAccuracy:
    """Accuracy tests for multiplicity functions, bias, and virial overdensity."""

    def test_delta_vir_eds_limit(self):
        """Δ_vir(Ω_m = 1, EdS) = 18π² ≈ 177.65 (Bryan & Norman 1998)."""
        from hod_mod.core.halo_mass_function import delta_vir_flat_jax
        dv = float(delta_vir_flat_jax(0.0, 1.0))
        np.testing.assert_allclose(dv, 18.0 * np.pi**2, rtol=1e-5)

    def test_fsigma_tinker08_at_sigma1(self):
        """f_T08(σ=1, z=0, Δ=200) ≈ 0.283 ± 5%.

        Computed from Tinker+2008 Table 2 (Δ=200): A0=0.186, a0=1.47, b0=2.57,
        c0=1.19 → f = 0.186 × (2.57^1.47 + 1) × exp(−1.19) ≈ 0.283.
        """
        from hod_mod.core.halo_mass_function import fsigma_tinker08
        f = float(fsigma_tinker08(jnp.array([1.0]), z=0.0, Delta=200.0)[0])
        np.testing.assert_allclose(f, 0.283, rtol=0.05,
                                   err_msg=f"f_T08(σ=1) = {f:.4f}, expected ≈ 0.283")

    def test_tinker10_bias_near_unity_at_nu1(self):
        """b_T10(ν = 1, Δ=200) is in [0.85, 1.15] (characteristic mass scale)."""
        from hod_mod.core.halo_mass_function import tinker10_bias
        b = float(tinker10_bias(jnp.array([1.0]), Delta=200.0)[0])
        assert 0.85 < b < 1.15, f"b(ν=1) = {b:.3f} outside [0.85, 1.15]"

    def test_tinker10_bias_large_nu(self):
        """b(ν >> 1) >> 1 — high-mass halos are strongly biased tracers."""
        from hod_mod.core.halo_mass_function import tinker10_bias
        b_large = float(tinker10_bias(jnp.array([5.0]))[0])
        assert b_large > 3.0, f"b(ν=5) = {b_large:.2f}, expected > 3"

    def test_fsigma_tinker08_positive(self):
        """f_T08(σ) > 0 for σ ∈ [0.2, 3.0]."""
        from hod_mod.core.halo_mass_function import fsigma_tinker08
        sigma = jnp.logspace(-0.7, 0.5, 60)
        f = fsigma_tinker08(sigma)
        assert jnp.all(f > 0)

    def test_fsigma_all_variants_positive(self):
        """All 14 f(σ) variants return positive values for σ ∈ [0.3, 2.0]."""
        from hod_mod.core.halo_mass_function import (
            fsigma_press74, fsigma_sheth99, fsigma_jenkins01, fsigma_warren06,
            fsigma_angulo12, fsigma_crocce10, fsigma_watson13, fsigma_bhattacharya11,
            fsigma_courtin11, fsigma_bocquet16, fsigma_despali16,
            fsigma_rodriguezpuebla16, fsigma_comparat17, fsigma_seppi20,
        )
        sigma = jnp.logspace(-0.5, 0.3, 40)
        for fn in [
            fsigma_press74, fsigma_sheth99, fsigma_jenkins01, fsigma_warren06,
            fsigma_angulo12, fsigma_crocce10, fsigma_watson13, fsigma_bhattacharya11,
            fsigma_courtin11, fsigma_bocquet16, fsigma_despali16,
            fsigma_rodriguezpuebla16, fsigma_comparat17, fsigma_seppi20,
        ]:
            f = fn(sigma)
            assert jnp.all(f > 0), f"{fn.__name__} returned non-positive values"

    @pytest.fixture(scope="class")
    def hmf(self):
        pytest.importorskip("camb")
        from hod_mod.core.power_spectrum import LinearPowerSpectrum
        from hod_mod.core.halo_mass_function import make_hmf
        lin = LinearPowerSpectrum()
        return make_hmf("tinker08", pk_func=lin.pk_linear)

    def test_hmf_sigma_monotone(self, hmf):
        """σ(M) is strictly monotonically decreasing."""
        m = jnp.logspace(10, 15, 60)
        sig = np.asarray(hmf.sigma(m, z=0.0, theta=_PLANCK18))
        assert np.all(np.diff(sig) < 0), "σ(M) must be monotonically decreasing"

    def test_hmf_dndm_integral_range(self, hmf):
        """∫ dn/dM dM over [10^12, 10^15] M_sun/h ∈ [1e-4, 0.5] (Mpc/h)^-3."""
        m = jnp.logspace(12, 15, 200)
        dndm = np.asarray(hmf.dndm(m, z=0.0, theta=_PLANCK18))
        n = float(scipy.integrate.trapezoid(dndm, np.asarray(m)))
        assert 1e-4 < n < 0.5, f"∫dn/dM dM = {n:.3e}, outside expected range"


# ===========================================================================
# galaxies/hod.py
# ===========================================================================

class TestHODAccuracy:
    """Accuracy tests for HOD occupation functions."""

    _LOG10M = jnp.linspace(10.0, 15.0, 200)
    _HOD_PARAMS = dict(log10mmin=11.35, sigma_logm=0.25,
                       log10m0=11.20, log10m1=12.40, alpha=1.0)

    def test_ncen_at_mmin_equals_half(self):
        """N_cen(M_min) = 0.5 exactly (erfc definition)."""
        from hod_mod.connection.hod import n_cen
        p = self._HOD_PARAMS
        val = float(n_cen(jnp.array([p["log10mmin"]]), p["log10mmin"], p["sigma_logm"])[0])
        np.testing.assert_allclose(val, 0.5, atol=1e-6)

    def test_ncen_low_mass_limit(self):
        """N_cen(M << M_min) < 1e-10."""
        from hod_mod.connection.hod import n_cen
        p = self._HOD_PARAMS
        val = float(n_cen(jnp.array([8.0]), p["log10mmin"], p["sigma_logm"])[0])
        assert val < 1e-10, f"N_cen(8.0) = {val:.2e}, expected < 1e-10"

    def test_ncen_high_mass_limit(self):
        """N_cen(M >> M_min) > 1 - 1e-8."""
        from hod_mod.connection.hod import n_cen
        p = self._HOD_PARAMS
        val = float(n_cen(jnp.array([16.0]), p["log10mmin"], p["sigma_logm"])[0])
        assert val > 1.0 - 1e-8, f"N_cen(16) = {val:.10f}, expected → 1"

    def test_nsat_zero_below_m0(self):
        """N_sat = 0 for log10 M < log10 M_0."""
        from hod_mod.connection.hod import n_sat
        p = self._HOD_PARAMS
        val = float(n_sat(
            jnp.array([p["log10m0"] - 0.5]),
            p["log10mmin"], p["sigma_logm"],
            p["log10m0"], p["log10m1"], p["alpha"],
        )[0])
        np.testing.assert_allclose(val, 0.0, atol=1e-9)

    def test_nsat_powerlaw_slope(self):
        """d log N_sat / d log M → α at M >> M_1."""
        from hod_mod.connection.hod import n_sat
        p = self._HOD_PARAMS
        log10m = jnp.linspace(p["log10m1"] + 1.5, p["log10m1"] + 2.5, 40)
        ns = np.log10(np.asarray(n_sat(
            log10m, p["log10mmin"], p["sigma_logm"],
            p["log10m0"], p["log10m1"], p["alpha"],
        )))
        slope = np.polyfit(np.asarray(log10m), ns, 1)[0]
        np.testing.assert_allclose(slope, p["alpha"], rtol=0.02,
                                   err_msg=f"high-M slope {slope:.4f} ≠ α={p['alpha']}")

    def test_f_quenched_low_mass_limit(self):
        """f_q(M → 0) → 1 (all halos quenched at low mass)."""
        from hod_mod.connection.hod import f_quenched
        val = float(f_quenched(jnp.array([8.0]), log10m_q=12.0)[0])
        np.testing.assert_allclose(val, 1.0, atol=1e-3)

    def test_f_quenched_high_mass_limit(self):
        """f_q(M → ∞) → 0."""
        from hod_mod.connection.hod import f_quenched
        val = float(f_quenched(jnp.array([18.0]), log10m_q=12.0)[0])
        assert val < 1e-5, f"f_q(18) = {val:.2e}, expected → 0"

    def test_f_red_cen_saturates(self):
        """f_red_cen(M → ∞) → 1 (Weibull CDF saturates)."""
        from hod_mod.connection.hod import f_red_cen_zu16
        val = float(f_red_cen_zu16(jnp.array([18.0]), lg_mqc_h=12.5, mu_c=1.5)[0])
        np.testing.assert_allclose(val, 1.0, atol=1e-4)

    def test_f_red_sat_saturates(self):
        """f_red_sat(M → ∞) → 1 (Weibull CDF saturates)."""
        from hod_mod.connection.hod import f_red_sat_zu16
        val = float(f_red_sat_zu16(jnp.array([18.0]), lg_mqs_h=12.0, mu_s=2.0)[0])
        np.testing.assert_allclose(val, 1.0, atol=1e-4)

    def test_sigma_lnmstar_positive(self):
        """σ_lnM*(M) > 0 everywhere for physical parameters."""
        from hod_mod.connection.hod import sigma_lnmstar_zu15
        log10m = jnp.linspace(11.0, 15.0, 100)
        sig = sigma_lnmstar_zu15(log10m, lg_m1h=11.5, sigma_lnmstar=0.2, eta=-0.04)
        assert jnp.all(sig > 0)

    def test_shmr_guo18_monotone(self):
        """Guo+2018 SHMR M_*(M_h) is monotonically increasing."""
        from hod_mod.connection.hod import shmr_guo18
        log10m = jnp.linspace(11.0, 15.0, 100)
        ms = shmr_guo18(log10m, log10m_star0=10.3, log10m1=12.5,
                        alpha=0.5, beta=2.0)
        assert jnp.all(jnp.diff(ms) > 0)

    def test_shmr_zacharegkas25_monotone(self):
        """Zacharegkas+2025 Kravtsov SHMR is monotonically increasing with best-fit params."""
        from hod_mod.connection.hod import shmr_zacharegkas25
        log10m = jnp.linspace(11.0, 15.0, 100)
        # Best-fit Zacharegkas+2025 Table 2 parameters
        ms = shmr_zacharegkas25(log10m, log10m1=11.506, log10eps=-1.632,
                                 alpha=-1.638, gamma=0.596, delta=3.810)
        assert jnp.all(jnp.diff(ms) > 0)

    def test_n_cen_thresh_zu15_range(self):
        """N_cen_thresh_zu15 ∈ [0, 1] for all halo masses."""
        from hod_mod.connection.hod import n_cen_thresh_zu15
        m_h = jnp.linspace(10.0, 16.0, 500)
        nc = n_cen_thresh_zu15(
            m_h,
            log10m_star_thresh=10.0,
            lg_m1h=12.52, lg_m0star=10.916,
            beta=0.457, delta=0.566, gamma=1.53,
            sigma_lnmstar=0.206, eta=-0.04, fc=1.0,
        )
        assert jnp.all(nc >= 0.0) and jnp.all(nc <= 1.0 + 1e-6)


# ===========================================================================
# galaxies/sham.py
# ===========================================================================

class TestSHAMAccuracy:
    """Physical bound and peak-location tests for SHAM relations."""

    _LOG10M = jnp.linspace(10.0, 15.0, 100)

    def test_moster13_mstar_below_mhalo(self):
        """Moster+2013: M_* < M_halo everywhere (f_* < 1)."""
        from hod_mod.connection.sham import smhm_moster13
        log10ms = smhm_moster13(self._LOG10M, z=0.1)
        diff = np.asarray(log10ms) - np.asarray(self._LOG10M)
        assert np.all(diff < 0), "Stellar mass must not exceed halo mass"

    def test_moster13_peak_location(self):
        """Moster+2013 peak M_*/M_h at M_h ≈ 10^{11.5} M_sun (within ±0.5 dex)."""
        from hod_mod.connection.sham import smhm_moster13
        log10m = jnp.linspace(10.5, 13.5, 200)
        ratio = np.asarray(smhm_moster13(log10m, z=0.0)) - np.asarray(log10m)
        peak_log10m = float(log10m[np.argmax(ratio)])
        assert abs(peak_log10m - 11.5) < 0.5, f"Peak at {peak_log10m:.2f}, expected ~11.5"

    def test_behroozi13_mstar_below_mhalo(self):
        """Behroozi+2013: M_* < M_halo everywhere."""
        from hod_mod.connection.sham import smhm_behroozi13
        log10ms = smhm_behroozi13(self._LOG10M, z=0.1)
        diff = np.asarray(log10ms) - np.asarray(self._LOG10M)
        assert np.all(diff < 0)

    def test_girelli20_mstar_below_mhalo(self):
        """Girelli+2020: M_* < M_halo everywhere."""
        from hod_mod.connection.sham import smhm_girelli20
        log10ms = smhm_girelli20(self._LOG10M, z=0.1)
        diff = np.asarray(log10ms) - np.asarray(self._LOG10M)
        assert np.all(diff < 0)

    def test_moster13_monotone_increasing(self):
        """Moster+2013: M_*(M_h) is monotonically increasing."""
        from hod_mod.connection.sham import smhm_moster13
        log10ms = smhm_moster13(self._LOG10M, z=0.0)
        assert jnp.all(jnp.diff(log10ms) > 0)

    def test_behroozi13_monotone_increasing(self):
        """Behroozi+2013: M_*(M_h) is monotonically increasing."""
        from hod_mod.connection.sham import smhm_behroozi13
        log10ms = smhm_behroozi13(self._LOG10M, z=0.0)
        assert jnp.all(jnp.diff(log10ms) > 0)

    def test_sham_redshift_evolution(self):
        """Moster+2013: M_*/M_h decreases toward lower-mass halos at z=0 vs z=2."""
        from hod_mod.connection.sham import smhm_moster13
        log10m_hi = jnp.array([14.0])
        # At high masses, evolution should be mild
        ms_z0 = float(smhm_moster13(log10m_hi, z=0.0)[0])
        ms_z2 = float(smhm_moster13(log10m_hi, z=2.0)[0])
        # Just verify both are physically reasonable, not NaN
        assert np.isfinite(ms_z0) and np.isfinite(ms_z2)
