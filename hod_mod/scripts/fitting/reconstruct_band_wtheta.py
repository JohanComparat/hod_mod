"""Reconstruct the per-sample, per-energy-band galaxy x X-ray w(theta) summaries
by merging the per-field measurements on the backup drive.

Ports ``compute_average_2PCFCross.py:get_merge_wth`` + the field selection
(eROSITA-DE footprint, KS quality, has_ACF, not clipped) from the LS DR10
clustering pipeline.  The 15 bands are 0.5-0.6 ... 1.9-2.0 keV
(folders ``<emin>_E_<emax>`` in eV).  Output (small) FITS go to a gitignored repo
dir; the broad-band sum is validated against the zenodo broad-band w(theta).

    python -m hod_mod.scripts.fitting.reconstruct_band_wtheta            # all samples
    python -m hod_mod.scripts.fitting.reconstruct_band_wtheta --sample LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
from astropy.table import Table

from hod_mod import paths
from hod_mod.paths import data_root

_GIT = str(data_root() / "legacysurvey/lsdr10_clustering_data")
_BACKUP = "/media/comparat/backup/data/data_s4_c030"
_META = os.path.join(_GIT, "data", "metadata_wtheta_per_field")
_ZENODO_XCORR = str(data_root() / "zenodo/LSDR10_GALxEVT/Measurements_Xcorr_Stacks/XCORR")
# Output root for the reconstructed band w(theta): $HOD_MOD_DATA_DIR/xray_bands
# if set, else the in-repo hod_mod/data/xray_bands (gitignored).  Keep this in
# sync with the Phase-B band loader (fit_xray_joint_bands.load_band_data).
_OUT = os.fspath(paths.data_path("xray_bands"))
_SUFFIX = "GALxEVTc030singleRRDR10"
_BANDS = [f"{lo:04d}_E_{lo+100:04d}" for lo in range(500, 2000, 100)]   # 15 bands


def _selected_srvmaps(basename):
    s4 = Table.read(os.path.join(_META, "skymap_wth_metadataLOG_4.fits"))
    cl = Table.read(os.path.join(_META, basename + "_xcorr_clipping.fits"))
    assert np.all(s4["SRVMAP"] == cl["SRVMAP"]), "metadata SRVMAP misaligned"
    is_eroDE = (((cl["OWNER"] == 2) | (cl["OWNER"] == 0))
                & (np.abs(cl["GLAT_CEN"]) > 18) & (cl["DE_CEN"] <= 32))
    is_good_KS = (is_eroDE & (s4["KS_dev_MASKED_Obs"] > 0.0)
                  & (s4["KS_dev_MASKED_Obs"] < 0.2))
    is_good_2pcf = is_good_KS & cl["has_ACF"] & (~cl["is_clipped"])
    return [int(x) for x in cl["SRVMAP"][is_good_2pcf]]


def _band_path(basename, band, srv):
    return os.path.join(_BACKUP, f"{srv:06d}", "WTH", band,
                        f"{basename}_{_SUFFIX}_{srv:06d}.wtheta.2pcf.fits")


_CNT = ["N_data", "N2_data", "N_random", "N2_random",
        "D1D2_counts", "D1R2_counts", "D2R1_counts", "R1R2_counts"]


def _merge(tables):
    """Sum the raw counts over fields and form the Landy-Szalay estimator."""
    m = Table()
    for c in ("theta_min", "theta_max", "theta_mid"):
        m[c] = tables[0][c]
    tot = {c: np.sum([np.asarray(t[c], float) for t in tables], axis=0) for c in _CNT}
    for c in _CNT:
        m[c] = tot[c]
    fN1 = tot["N_random"][0] / tot["N_data"][0]
    fN2 = tot["N2_random"][0] / tot["N2_data"][0]
    cf = np.full(len(m), np.nan)
    nz = tot["R1R2_counts"] > 0
    cf[nz] = (fN1 * fN2 * tot["D1D2_counts"][nz] - fN1 * tot["D1R2_counts"][nz]
              - fN2 * tot["D2R1_counts"][nz] + tot["R1R2_counts"][nz]) / tot["R1R2_counts"][nz]
    m["wtheta"] = cf
    with np.errstate(divide="ignore", invalid="ignore"):
        m["wtheta_err"] = np.abs(cf) * np.sqrt(
            0.01 ** 2 + sum(1.0 / np.maximum(tot[c], 1.0)
                            for c in ("D1D2_counts", "D1R2_counts",
                                      "D2R1_counts", "R1R2_counts")))
    return m, tot


def reconstruct_sample(basename, verbose=True):
    srv = _selected_srvmaps(basename)
    out_dir = os.path.join(_OUT, basename)
    os.makedirs(out_dir, exist_ok=True)
    band_counts = []
    for band in _BANDS:
        t0 = time.time()
        tabs = [Table.read(_band_path(basename, band, s)) for s in srv
                if os.path.isfile(_band_path(basename, band, s))]
        if not tabs:
            print(f"  {band}: NO field files found", flush=True); continue
        m, tot = _merge(tabs)
        m.write(os.path.join(out_dir, band + ".fits"), overwrite=True)
        band_counts.append(tot)
        if verbose:
            print(f"  {band}: {len(tabs)} fields, w(0.01deg)~"
                  f"{np.interp(0.01, m['theta_mid'], m['wtheta']):.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    # broad-band validation: sum the band counts, LS, vs zenodo broad band
    if band_counts:
        bb = Table()
        ref = Table.read(os.path.join(out_dir, _BANDS[0] + ".fits"))
        for c in ("theta_min", "theta_max", "theta_mid"):
            bb[c] = ref[c]
        tot = {c: np.sum([b[c] for b in band_counts], axis=0) for c in _CNT}
        fN1 = tot["N_random"][0] / tot["N_data"][0]
        fN2 = tot["N2_random"][0] / tot["N2_data"][0]
        cf = np.full(len(bb), np.nan); nz = tot["R1R2_counts"] > 0
        cf[nz] = (fN1*fN2*tot["D1D2_counts"][nz] - fN1*tot["D1R2_counts"][nz]
                  - fN2*tot["D2R1_counts"][nz] + tot["R1R2_counts"][nz]) / tot["R1R2_counts"][nz]
        bb["wtheta"] = cf
        bb.write(os.path.join(out_dir, "broadband_sum_0500_2000.fits"), overwrite=True)
        _validate(basename, bb)
    return out_dir


def _validate(basename, bb_sum):
    zfs = glob.glob(os.path.join(_ZENODO_XCORR, basename + "_GALxEVT_wtheta.fits"))
    if not zfs:
        print("  [validate] zenodo broad-band file not found; skipping", flush=True)
        return
    z = Table.read(zfs[0])
    th_z = np.asarray(z["theta"], float) * 3600.0           # arcsec
    th_b = np.asarray(bb_sum["theta_mid"], float) * 3600.0
    m = (th_b >= 8.0) & (th_b <= 300.0)
    wz = np.interp(th_b[m], th_z, np.asarray(z["wtheta"], float))
    wb = np.asarray(bb_sum["wtheta"], float)[m]
    rel = np.nanmedian(np.abs(wb - wz) / np.abs(wz))
    print(f"  [validate] Sum(15 bands) vs zenodo broad band over [8,300]\": "
          f"median |rel diff| = {100*rel:.1f}%  (n={m.sum()})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default=None, help="one basename (default: all 7)")
    args = ap.parse_args(argv)
    if args.sample:
        names = [args.sample]
    else:
        names = sorted(os.path.basename(f)[:-len("_xcorr_clipping.fits")]
                       for f in glob.glob(os.path.join(_META, "*_xcorr_clipping.fits")))
    print(f"reconstructing {len(names)} sample(s) x {len(_BANDS)} bands -> {_OUT}",
          flush=True)
    for bn in names:
        print(f"=== {bn} ===", flush=True)
        reconstruct_sample(bn)


if __name__ == "__main__":
    main()
