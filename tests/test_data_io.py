"""Tests for the sum_stat data I/O layer.

All tests create synthetic in-memory data (HDF5 or FITS) and verify that
SumStatReader correctly reads, applies h-unit conversion, and returns
well-formed dicts.  No internet access or external data files are required.
"""

import io
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: create minimal synthetic HDF5 and FITS files
# ---------------------------------------------------------------------------

def make_wp_hdf5(path, h=0.6736, n_bins=10, pi_max=100.0):
    """Write a minimal single-stat wp HDF5 file."""
    import h5py
    rp_Mpc = np.logspace(-1, 1.5, n_bins)
    wp_Mpc = 100.0 * rp_Mpc ** (-0.8)
    cov_Mpc2 = np.diag((0.1 * wp_Mpc) ** 2)

    with h5py.File(path, "w") as f:
        f.attrs["created_by"] = "test"
        g = f.create_group("wp")
        g.attrs["pi_max_Mpc"]  = pi_max
        g.attrs["estimator"]   = "landy-szalay"
        g.attrs["cov_method"]  = "treecorr_variance"
        g.attrs["survey"]      = "test"
        g.attrs["n_gal"]       = 100000
        cosmo = g.create_group("cosmology")
        cosmo.attrs["name"] = "test"
        cosmo.create_dataset("H0",  data=h * 100.0)
        cosmo.create_dataset("Om0", data=0.315)
        cosmo.create_dataset("Ob0", data=0.049)
        cosmo.create_dataset("Ok0", data=0.0)
        g.create_dataset("sep_centres", data=rp_Mpc)
        g.create_dataset("xi",          data=wp_Mpc)
        g.create_dataset("cov",         data=cov_Mpc2)
        g.create_dataset("bin_edges",   data=np.linspace(rp_Mpc[0], rp_Mpc[-1], n_bins + 1))


def make_smf_hdf5(path, h=0.6736, n_bins=14):
    """Write a minimal single-stat SMF HDF5 file."""
    import h5py
    log10m = np.linspace(9.0, 12.0, n_bins)
    phi    = 1e-3 * 10 ** (-0.5 * (log10m - 10.7) ** 2)
    phi_e  = phi * 0.1
    cov    = np.diag(phi_e ** 2)

    with h5py.File(path, "w") as f:
        f.attrs["created_by"] = "test"
        g = f.create_group("smf")
        g.attrs["estimator"]  = "1/Vmax"
        g.attrs["cov_method"] = "jackknife_K100"
        g.attrs["area_deg2"]  = 1000.0
        cosmo = g.create_group("cosmology")
        cosmo.attrs["name"] = "test"
        cosmo.create_dataset("H0",  data=h * 100.0)
        cosmo.create_dataset("Om0", data=0.315)
        cosmo.create_dataset("Ob0", data=0.049)
        cosmo.create_dataset("Ok0", data=0.0)
        g.create_dataset("log10mstar_centres", data=log10m)
        g.create_dataset("phi",     data=phi)
        g.create_dataset("phi_err", data=phi_e)
        g.create_dataset("cov",     data=cov)
        g.create_dataset("bin_edges", data=np.linspace(9.0, 12.0, n_bins + 1))


def make_joint_hdf5(path, h=0.6736, n_smf=10, n_wp=10, n_ds=10):
    """Write a minimal joint SMF+wp+ESD HDF5 file."""
    import h5py
    n_tot = n_smf + n_wp + n_ds

    rp_Mpc  = np.logspace(-1, 1.2, n_wp)
    wp_Mpc  = 80.0 * rp_Mpc ** (-0.8)
    log10m  = np.linspace(9.5, 11.5, n_smf)
    phi_Mpc = 1e-3 * np.ones(n_smf)
    ds      = 100.0 * np.ones(n_ds)
    dv      = np.concatenate([phi_Mpc, wp_Mpc, ds])
    cov     = np.diag((0.1 * dv) ** 2)

    with h5py.File(path, "w") as f:
        f.attrs["created_by"] = "test"

        # smf group
        smf_grp = f.create_group("smf")
        sg = smf_grp.create_group("test_sample")
        sg.attrs["estimator"] = "1/Vmax"
        cosmo = sg.create_group("cosmology")
        cosmo.create_dataset("H0",  data=h * 100.0)
        cosmo.create_dataset("Om0", data=0.315)
        cosmo.create_dataset("Ob0", data=0.049)
        cosmo.create_dataset("Ok0", data=0.0)
        sg.create_dataset("log10mstar_centres", data=log10m)
        sg.create_dataset("phi",     data=phi_Mpc)
        sg.create_dataset("phi_err", data=phi_Mpc * 0.1)
        sg.create_dataset("cov",     data=np.diag((phi_Mpc * 0.1) ** 2))
        sg.create_dataset("bin_edges", data=np.linspace(9.5, 11.5, n_smf + 1))

        # twopcf group
        twop_grp = f.create_group("twopcf")
        wg = twop_grp.create_group("wp_test_sample")
        wg.attrs["pi_max_Mpc"] = 100.0
        cosmo2 = wg.create_group("cosmology")
        cosmo2.create_dataset("H0",  data=h * 100.0)
        cosmo2.create_dataset("Om0", data=0.315)
        cosmo2.create_dataset("Ob0", data=0.049)
        cosmo2.create_dataset("Ok0", data=0.0)
        wg.create_dataset("sep_centres", data=rp_Mpc)
        wg.create_dataset("xi",          data=wp_Mpc)
        wg.create_dataset("cov",         data=np.diag((0.1 * wp_Mpc) ** 2))
        wg.create_dataset("bin_edges",   data=np.linspace(rp_Mpc[0], rp_Mpc[-1], n_wp + 1))

        # esd group
        esd_grp = f.create_group("esd")
        eg = esd_grp.create_group("esd_test_HSC")
        eg.attrs["source_survey"] = "HSC_test"
        cosmo3 = eg.create_group("cosmology")
        cosmo3.create_dataset("H0",  data=h * 100.0)
        cosmo3.create_dataset("Om0", data=0.315)
        cosmo3.create_dataset("Ob0", data=0.049)
        cosmo3.create_dataset("Ok0", data=0.0)
        eg.create_dataset("rp_centres",  data=rp_Mpc)
        eg.create_dataset("delta_sigma", data=ds)
        eg.create_dataset("cov",         data=np.diag((0.1 * ds) ** 2))
        eg.create_dataset("bin_edges",   data=np.linspace(rp_Mpc[0], rp_Mpc[-1], n_ds + 1))

        # joint_covariance group
        jg = f.create_group("joint_covariance")
        jg.attrs["n_bins_smf"] = n_smf
        jg.attrs["n_bins_wp"]  = n_wp
        jg.attrs["n_bins_ds"]  = n_ds
        jg.create_dataset("data_vector",   data=dv)
        jg.create_dataset("cov",           data=cov)
        jg.create_dataset("err_jackknife", data=np.sqrt(np.diag(cov)))
        jg.create_dataset("mstar_centres", data=log10m)
        jg.create_dataset("rp_centres",    data=rp_Mpc)
        jg.create_dataset("jk_subsamples", data=np.tile(dv, (10, 1)))


def make_smf_fits(path, n_bins=14):
    """Write a minimal GAMA/COSMOS-style SMF FITS file."""
    from astropy.io import fits
    import astropy.table

    log10m = np.linspace(8.5, 12.0, n_bins)
    phi    = 1e-3 * 10 ** (-0.4 * (log10m - 10.5) ** 2)
    phi[phi < 0] = 0
    phi_e  = phi * 0.1

    tbl = astropy.table.Table({
        "log10mstar":    log10m,
        "phi":           phi,
        "phi_err":       phi_e,
        "log10mstar_lo": log10m - 0.125,
        "log10mstar_hi": log10m + 0.125,
    })
    tbl.write(path, format="fits", overwrite=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSumStatReaderWp:
    def test_wp_read_and_shape(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        import h5py
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_wp_hdf5(path, h=0.6736, n_bins=10)
            reader = SumStatReader.from_hdf5(path)
            d = reader.wp()
            assert "rp"  in d and "wp" in d and "cov" in d
            assert d["rp"].shape  == (10,)
            assert d["wp"].shape  == (10,)
            assert d["cov"].shape == (10, 10)
        finally:
            os.unlink(path)

    def test_wp_h_unit_conversion(self):
        """rp and wp must be multiplied by h compared to Mpc values."""
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        h = 0.6736
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_wp_hdf5(path, h=h, n_bins=5)
            reader = SumStatReader.from_hdf5(path)
            d = reader.wp()
            import h5py
            with h5py.File(path, "r") as hf:
                rp_Mpc = np.array(hf["wp/sep_centres"])
            np.testing.assert_allclose(d["rp"], rp_Mpc * h, rtol=1e-6)
        finally:
            os.unlink(path)

    def test_wp_cov_h_units(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        h = 0.6736
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_wp_hdf5(path, h=h, n_bins=5)
            reader = SumStatReader.from_hdf5(path)
            d = reader.wp()
            import h5py
            with h5py.File(path, "r") as hf:
                cov_Mpc2 = np.array(hf["wp/cov"])
            np.testing.assert_allclose(d["cov"], cov_Mpc2 * h**2, rtol=1e-6)
        finally:
            os.unlink(path)

    def test_wp_list_groups(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_wp_hdf5(path)
            reader = SumStatReader.from_hdf5(path)
            assert "wp" in reader.list_groups()
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        with pytest.raises(FileNotFoundError):
            SumStatReader.from_hdf5("/nonexistent/file.h5")


class TestSumStatReaderSmf:
    def test_smf_read(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_smf_hdf5(path, h=0.6736, n_bins=14)
            reader = SumStatReader.from_hdf5(path)
            d = reader.smf()
            assert "log10mstar" in d and "phi" in d and "cov" in d
            assert d["log10mstar"].shape == (14,)
            assert d["phi"].shape == (14,)
        finally:
            os.unlink(path)

    def test_smf_phi_h_conversion(self):
        """phi must be divided by h^3."""
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        h = 0.6736
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_smf_hdf5(path, h=h, n_bins=5)
            reader = SumStatReader.from_hdf5(path)
            d = reader.smf()
            import h5py
            with h5py.File(path, "r") as hf:
                phi_Mpc = np.array(hf["smf/phi"])
            np.testing.assert_allclose(d["phi"], phi_Mpc / h**3, rtol=1e-6)
        finally:
            os.unlink(path)


class TestSumStatReaderJoint:
    def test_joint_keys(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_joint_hdf5(path, n_smf=10, n_wp=10, n_ds=10)
            reader = SumStatReader.from_hdf5(path)
            j = reader.joint()
            assert j["data_vector"].shape == (30,)
            assert j["cov"].shape == (30, 30)
            assert j["n_bins_smf"] == 10
            assert j["n_bins_wp"]  == 10
            assert j["n_bins_ds"]  == 10
        finally:
            os.unlink(path)

    def test_joint_rp_h_conversion(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        h = 0.6736
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            make_joint_hdf5(path, h=h, n_smf=10, n_wp=10, n_ds=10)
            reader = SumStatReader.from_hdf5(path)
            j = reader.joint()
            import h5py
            with h5py.File(path, "r") as hf:
                rp_Mpc = np.array(hf["joint_covariance/rp_centres"])
            np.testing.assert_allclose(j["rp_centres"], rp_Mpc * h, rtol=1e-6)
        finally:
            os.unlink(path)


class TestSumStatReaderFITS:
    def test_fits_smf_read(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            make_smf_fits(path, n_bins=14)
            reader = SumStatReader.from_fits(path)
            d = reader.smf()
            assert "log10mstar" in d and "phi" in d
            assert d["phi"].shape[0] > 0
            # phi > 0 filtering should have been applied
            assert np.all(d["phi"] > 0)
        finally:
            os.unlink(path)

    def test_fits_smf_h_conversion(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        h = 0.6736
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            make_smf_fits(path, n_bins=10)
            reader_noh = SumStatReader.from_fits(path)
            reader_wh  = SumStatReader.from_fits(path)
            d_no = reader_noh.smf(h=None)
            d_wi = reader_wh.smf(h=h)
            np.testing.assert_allclose(d_wi["phi"], d_no["phi"] / h**3, rtol=1e-6)
        finally:
            os.unlink(path)

    def test_fits_missing_file_raises(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        with pytest.raises(FileNotFoundError):
            SumStatReader.from_fits("/nonexistent/file.fits")
