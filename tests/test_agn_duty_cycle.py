"""Tests for the Lau et al. 2025 AGN model (agn_lau): the W_AGN(z) kernel
(Eq. A9), the ZM15×duty-cycle occupation, and the galaxy×AGN cross-power
wiring into HaloModelCrossSpectra."""

import os

import numpy as np
import pytest

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.connection.hod import n_cen_thresh_zu15, n_sat_thresh_zu15
from hod_mod.agn.duty_cycle import (
    DutyCycleAGNModel,
    compute_w_agn_kernel,
    load_zm15_map_params,
    w_agn_path_for,
)

_THETA = LinearPowerSpectrum.default_cosmology()


# ---------------------------------------------------------------------------
# Part 1 — W_AGN(z) kernel (Eq. A9)
# ---------------------------------------------------------------------------

class TestWAgnKernel:

    @pytest.fixture(scope="class")
    def kernel_path(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("agn_duty_cycle") / "W_AGN_S1.h5"
        return compute_w_agn_kernel(
            sample="S1", theta_cosmo=_THETA, n_z=12, n_lx=200,
            out_path=str(out),
        )

    def test_file_created_with_all_datasets(self, kernel_path):
        import h5py
        assert os.path.exists(kernel_path)
        with h5py.File(kernel_path, "r") as f:
            for key in ("z_grid", "log10LX_hard", "phi_dex", "k_eff",
                        "log10LX_soft", "S_X", "selection_mask",
                        "f_SX", "integrand_W", "W_AGN", "n_AGN", "mean_SX"):
                assert key in f, f"missing dataset {key}"
            # no optical (r-band) selection is stored: galaxies x all events
            assert "r_mag" not in f
            assert f["phi_dex"].shape == (12, 200)
            assert f["W_AGN"].shape == (12,)
            assert f.attrs["sample"] == "S1"
            # completeness assumption is recorded
            assert "f(S_X) = 1" in f.attrs["completeness"]

    def test_kernel_positive_and_finite(self, kernel_path):
        import h5py
        with h5py.File(kernel_path, "r") as f:
            W = np.asarray(f["W_AGN"])
            n = np.asarray(f["n_AGN"])
            mSX = np.asarray(f["mean_SX"])
        assert np.all(np.isfinite(W)) and np.all(np.isfinite(n))
        assert np.all(np.isfinite(mSX))
        assert np.all(W >= 0) and np.all(n >= 0)
        # near the sample mean redshift the kernel is strictly positive
        assert W.max() > 0 and n.max() > 0 and mSX.max() > 0

    def test_completeness_is_unity(self, kernel_path):
        import h5py
        with h5py.File(kernel_path, "r") as f:
            assert np.allclose(np.asarray(f["f_SX"]), 1.0)

    def test_flux_range_selection(self, kernel_path):
        import h5py
        with h5py.File(kernel_path, "r") as f:
            sel = np.asarray(f["selection_mask"])
            assert "none" in f.attrs["selection"]   # no optical r-band cut
        # the k-corrected flux range keeps most of the L grid (not none)
        assert sel.mean() > 0.0

    def test_skip_if_exists(self, kernel_path):
        # Re-running without overwrite must not rewrite the file.
        mtime0 = os.path.getmtime(kernel_path)
        path2 = compute_w_agn_kernel(
            sample="S1", theta_cosmo=_THETA, n_z=12, n_lx=200,
            out_path=kernel_path,
        )
        assert path2 == kernel_path
        assert os.path.getmtime(kernel_path) == mtime0


# ---------------------------------------------------------------------------
# Part 2 — occupation and emissivity
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agn(tmp_path_factory):
    out = tmp_path_factory.mktemp("agn_duty_cycle_model") / "W_AGN_S1.h5"
    compute_w_agn_kernel(sample="S1", theta_cosmo=_THETA, n_z=16, n_lx=300,
                         out_path=str(out))
    return DutyCycleAGNModel(sample="S1", theta_cosmo=_THETA, log10DC=-2.0,
                       w_agn_path=str(out))


class TestOccupation:

    def test_loads_13_zm15_params(self):
        p = load_zm15_map_params()
        assert len(p) == 13
        assert {"lg_m1h", "fc", "alpha_sat"} <= set(p)

    def test_threshold_is_sample_cut(self, agn):
        assert agn._log10m_star_thresh == pytest.approx(10.0)

    def test_equals_zm15_at_dc_unity(self, agn):
        # Below the high-mass cutoff the AGN occupation equals the raw ZM15.
        log10m = np.linspace(11, 13.5, 60)
        nc, ns = agn.nc_ns_agn(log10m)
        p = agn._zm15
        thr = agn._log10m_star_thresh
        nc_ref = np.asarray(n_cen_thresh_zu15(
            np.asarray(log10m), thr, p["lg_m1h"], p["lg_m0star"], p["beta"],
            p["delta"], p["gamma"], p["sigma_lnmstar"], p["eta"], p["fc"]))
        ns_ref = np.asarray(n_sat_thresh_zu15(
            np.asarray(log10m), thr, p["lg_m1h"], p["lg_m0star"], p["beta"],
            p["delta"], p["gamma"], p["sigma_lnmstar"], p["eta"], p["fc"],
            p["bsat"], p["beta_sat"], p["bcut"], p["beta_cut"], p["alpha_sat"]))
        assert np.allclose(nc, nc_ref)
        assert np.allclose(ns, ns_ref)

    def test_centrals_bounded_by_fc(self, agn):
        log10m = np.linspace(10, 16, 100)
        nc, _ = agn.nc_ns_agn(log10m)
        assert np.all(nc >= 0)
        assert nc.max() <= agn._zm15["fc"] + 1e-6

    def test_satellites_nonneg_and_grow_with_mass(self, agn):
        # Below the high-mass cutoff the satellite occupation grows with mass.
        log10m = np.linspace(12, 13.5, 50)
        _, ns = agn.nc_ns_agn(log10m)
        assert np.all(ns >= 0)
        assert ns[-1] > ns[0]

    def test_high_mass_cutoff(self, agn):
        # No AGN in the most massive halos: occupation declines from 1e14 to 0
        # at 2e14 (log10 ~ 14.301), monotonically and smoothly.
        lo, hi = agn._log10m_cut_lo, agn._log10m_cut_hi
        assert lo == pytest.approx(14.0)
        assert hi == pytest.approx(np.log10(2e14))
        # at/below 1e14 the AGN occupation equals the raw ZM15 (cutoff = 1)
        p = agn._zm15; thr = agn._log10m_star_thresh
        m_lo = np.array([13.0, 13.5, 14.0])
        nc_lo, ns_lo = agn.nc_ns_agn(m_lo)
        nc_ref = np.asarray(n_cen_thresh_zu15(
            np.asarray(m_lo), thr, p["lg_m1h"], p["lg_m0star"], p["beta"],
            p["delta"], p["gamma"], p["sigma_lnmstar"], p["eta"], p["fc"]))
        assert np.allclose(nc_lo, nc_ref)
        # at/above 2e14 the occupation is zero
        nc_hi, ns_hi = agn.nc_ns_agn(np.array([hi, 14.5, 15.0]))
        assert np.allclose(nc_hi, 0.0) and np.allclose(ns_hi, 0.0)
        # the cutoff FACTOR declines monotonically and smoothly across [lo, hi]
        from hod_mod.agn.duty_cycle import (
            _high_mass_cutoff, DutyCycleAGNModel)
        m_mid = np.linspace(lo, hi, 30)
        cut = np.asarray(_high_mass_cutoff(m_mid, lo, hi))
        assert np.all(np.diff(cut) <= 1e-9)
        assert cut[0] == pytest.approx(1.0) and cut[-1] == pytest.approx(0.0)
        # the saturated central occupation therefore also declines monotonically
        nc_mid, _ = agn.nc_ns_agn(m_mid)
        assert np.all(np.diff(nc_mid) <= 1e-9)
        # disabling the cutoff recovers a non-zero high-mass occupation; the
        # cutoff strictly reduces the occupation at high mass
        agn_no = DutyCycleAGNModel(sample="S1", theta_cosmo=_THETA,
                                   apply_high_mass_cutoff=False,
                                   w_agn_path=agn._w_agn_path)
        _, ns_no = agn_no.nc_ns_agn(np.array([14.2, 15.0]))
        _, ns_cut = agn.nc_ns_agn(np.array([14.2, 15.0]))
        assert ns_no[0] > 0 and ns_no[1] > 0
        assert np.all(ns_cut <= ns_no + 1e-12)


class TestEmissivity:

    def test_flat_in_k_and_mass(self, agn):
        k = np.logspace(-3, 1, 32)
        m = np.logspace(11, 15, 40)
        uk = agn.agn_emissivity_uk(k, m, 0.135, _THETA)
        assert uk.shape == (32, 40)
        assert np.all(np.isfinite(uk)) and np.all(uk > 0)
        assert np.allclose(uk, uk[0][None, :])     # flat in k
        assert np.allclose(uk, uk[:, 0][:, None])  # flat in M

    def test_linear_in_duty_cycle(self, agn):
        k = np.logspace(-3, 1, 8)
        m = np.logspace(11, 15, 8)
        e1 = agn.agn_emissivity_uk(k, m, 0.135, _THETA, log10DC=-2.0)
        e2 = agn.agn_emissivity_uk(k, m, 0.135, _THETA, log10DC=-1.0)
        assert np.allclose(e2 / e1, 10.0, rtol=1e-6)

    def test_tracks_mean_sx(self, agn):
        # <L_X>(z) = DC * mean_SX(z) * 4 pi d_L(z)^2
        from hod_mod.agn.duty_cycle import _luminosity_distance_cm
        z = 0.135
        dl_cm = float(_luminosity_distance_cm(z, _THETA)[0])
        expect = 10.0 ** (-2.0) * float(agn.mean_sx_at(z)) * 4.0 * np.pi * dl_cm ** 2
        assert agn.mean_agn_lx(z, log10DC=-2.0) == pytest.approx(expect, rel=1e-6)


# ---------------------------------------------------------------------------
# Part 3 — cross-spectra wiring (slow: builds the CAMB static cache)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cross(agn):
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.core.halo_profiles import HaloProfile
    from hod_mod.gas import GasDensityDPM
    from hod_mod.connection.hod import ZuMandelbaum15HODModel
    from hod_mod.observables.clustering import FullHaloModelPrediction
    from hod_mod.observables.cross_spectra import HaloModelCrossSpectra

    pk = LinearPowerSpectrum()
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=_THETA["h"] * 100.0, Om0=_THETA["Omega_m"],
                Ob0=_THETA["Omega_b"], sigma8=0.811, ns=_THETA["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    dp = GasDensityDPM(model=2, r_max_over_r200=3.0, n_gl=80)
    return HaloModelCrossSpectra(fhmp, density_profile=dp, agn_model=agn)


class TestCrossSpectra:

    def test_detects_hod_interface(self, cross):
        assert cross._agn_has_hod is True

    def test_pair_with_galaxy_flag(self, agn):
        assert getattr(agn, "_pair_with_galaxy_occupation", False) is True

    @pytest.mark.slow
    def test_agn_component_finite_and_positive(self, cross, agn):
        t = cross._pk_tables_gX(0.135, _THETA, agn.zm15_hod_params,
                                agn_kwargs={"log10DC": -2.0})
        pagn = np.exp(np.asarray(t["log_pgX_agn"]))
        pgas = np.exp(np.asarray(t["log_pgX_gas"]))
        assert np.all(np.isfinite(pagn)) and np.all(pagn > 0)
        assert np.all(np.isfinite(pgas))

    @pytest.mark.slow
    def test_agn_scales_linearly_with_dc(self, cross, agn):
        t1 = cross._pk_tables_gX(0.135, _THETA, agn.zm15_hod_params,
                                 agn_kwargs={"log10DC": -2.0})
        t2 = cross._pk_tables_gX(0.135, _THETA, agn.zm15_hod_params,
                                 agn_kwargs={"log10DC": -1.0})
        p1 = np.exp(np.asarray(t1["log_pgX_agn"]))
        p2 = np.exp(np.asarray(t2["log_pgX_agn"]))
        assert np.allclose(p2 / p1, 10.0, rtol=1e-4)
