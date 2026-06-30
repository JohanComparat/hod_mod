"""Coverage for fitting.config: load_config variant branches + the esd reader."""
import numpy as np
import pytest

from hod_mod.fitting import load_config, FitConfig
from hod_mod.fitting.config import read_esd_bwpd_4col_fig3, _sigma8_to_lnAs

# Minimal valid config (model + one free parameter). Extra YAML blocks are appended
# to exercise the joint / ds / fits / cosmology branches of load_config.
_BASE = """\
model:
  hod_model: MoreHODModel
  hmf_backend: tinker08
  z: 0.2
  pi_max: 60.0

parameters:
  log10mmin:
    init: 11.35
    free: true
    bounds: [10.5, 13.5]
    prior_type: uniform
"""


def _write(tmp_path, extra=""):
    p = tmp_path / "cfg.yml"
    p.write_text(_BASE + extra)
    return str(p)


def test_minimal_loads(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert isinstance(cfg, FitConfig)
    assert cfg.hod_model == "MoreHODModel" and cfg.ds_file is None


def test_joint_section(tmp_path):
    extra = """
joint:
  ds_file: data/foo_ds.csv
  ds_rp_min: 0.3
  ds_rp_max: 25.0
  ng_obs: 5.0e-4
  ng_frac_err: 0.15
  fit_ng: true
"""
    cfg = load_config(_write(tmp_path, extra))
    assert cfg.ds_file is not None and cfg.ds_file.endswith("data/foo_ds.csv")
    assert cfg.ds_rp_min == pytest.approx(0.3)
    assert cfg.ng_obs == pytest.approx(5.0e-4)
    assert cfg.fit_ng is True


def test_ds_only_section(tmp_path):
    extra = """
ds:
  file: data/foo_ds.csv
  rp_min: 0.5
  rp_max: 30.0
  fit_ng: false
"""
    cfg = load_config(_write(tmp_path, extra))
    assert cfg.ds_file is not None and cfg.ds_rp_min == pytest.approx(0.5)
    assert cfg.fit_ng is False


def test_fits_section_sets_format_and_jk(tmp_path):
    extra = """
fits:
  jk_dir: data/jk_samples
  jk_pattern: NSIDE_08
  h: 0.70
"""
    cfg = load_config(_write(tmp_path, extra))
    assert cfg.data_format == "fits"
    assert cfg.jk_dir is not None and cfg.jk_pattern == "NSIDE_08"
    assert cfg.h_hubble == pytest.approx(0.70)


def test_cosmology_section_derives_omega_cdm_and_lnAs(tmp_path):
    extra = """
cosmology:
  Omega_m: 0.31
  Omega_b: 0.048
  h: 0.6766
  n_s: 0.9665
  sigma8: 0.81
"""
    cfg = load_config(_write(tmp_path, extra))
    cosmo = cfg.cosmology
    assert cosmo["Omega_cdm"] == pytest.approx(0.31 - 0.048)
    assert "ln10^{10}A_s" in cosmo and np.isfinite(cosmo["ln10^{10}A_s"])


def test_use_free_cosmo_flag(tmp_path):
    cfg = load_config(_write(tmp_path, "\nfitting:\n  use_free_cosmo: true\n"))
    assert cfg.use_free_cosmo is True


def test_sigma8_to_lnAs_monotonic():
    base = {"Omega_m": 0.31, "Omega_b": 0.048, "h": 0.6766, "n_s": 0.9665}
    lo = _sigma8_to_lnAs({**base, "sigma8": 0.75})
    hi = _sigma8_to_lnAs({**base, "sigma8": 0.85})
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi > lo            # higher sigma8 -> larger amplitude A_s


def test_read_esd_bwpd_4col(tmp_path):
    f = tmp_path / "ds.csv"
    R   = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    DS  = np.array([40.0, 25.0, 15.0, 7.0, 3.0])
    hi  = DS * 1.15
    lo  = DS * 0.85
    np.savetxt(str(f), np.column_stack([R, DS, hi, lo]), delimiter=",")
    Rr, DSr, err_hi, err_lo = read_esd_bwpd_4col_fig3(str(f))
    assert np.allclose(Rr, R) and np.allclose(DSr, DS)
    assert np.allclose(err_hi, hi - DS) and np.allclose(err_lo, DS - lo)
    # scale cut
    Rc, *_ = read_esd_bwpd_4col_fig3(str(f), R_min=1.0, R_max=5.0)
    assert Rc.min() >= 1.0 and Rc.max() <= 5.0
