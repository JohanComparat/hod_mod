#!/usr/bin/env python
"""Convert Zu & Mandelbaum 2015 Figure 6 digitized txt files to standard CSV format.

Input txt files are in data/zumandelbaum2015_sdss/ and come in two column layouts:

  WPRP 2-col: rp_Mpchm1  wprp_Mpchm1
  WPRP 3-col: rp_Mpchm1  wprp_Mpchm1_up  wprp_Mpchm1_lo

  ESD  2-col: rp_Mpchm1  esd_Msunh_pcm2
  ESD  3-col: rp_Mpchm1  esd_Msunh_pcm2_up  esd_Msunh_pcm2_lo

Parsing rules:
  3-col: value = sqrt(up * lo)  [log-space midpoint],  err = (up - lo) / 2
  2-col WPRP: value = col2,  err = 0.15 * value
  2-col ESD:  value = col2,  err = 0.20 * value

ESD for bins 9.4-9.8 and 9.8-10.2 are NOT written (too noisy per ZM15).

Output CSV files use the standard fitting-code column names:
  wp:  rp_hMpc, wp_hMpc, wp_err_hMpc
  ds:  R_hMpc, ds_Msun_h_pc2, ds_err_Msun_h_pc2

Usage
-----
    python hod_mod/scripts/data/convert_zm15_txt_to_csv.py
"""

import os
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "zumandelbaum2015_sdss",
)

# Bin definitions: (label, lo_str, hi_str, lo_float, hi_float)
BINS = [
    ("9.4-9.8",   "9p4",  "9p8",  9.4,  9.8),
    ("9.8-10.2",  "9p8",  "10p2", 9.8,  10.2),
    ("10.2-10.6", "10p2", "10p6", 10.2, 10.6),
    ("10.6-11.0", "10p6", "11p0", 10.6, 11.0),
    ("11.0-11.2", "11p0", "11p2", 11.0, 11.2),
    ("11.2-11.4", "11p2", "11p4", 11.2, 11.4),
    ("11.4-12.0", "11p4", "12p0", 11.4, 12.0),
]

# ESD bins that should NOT be used
ESD_SKIP = {"9.4-9.8", "9.8-10.2"}


def _txt_fname(kind, lo_str, hi_str):
    lo = lo_str.replace("p", ".").replace("m", "-")
    hi = hi_str.replace("p", ".").replace("m", "-")
    return os.path.join(DATA_DIR, f"Fig6_{kind}_{lo}_M_{hi}_measurements.txt")


def read_txt(filepath, default_frac_err):
    """Read a 2- or 3-column measurement txt file.

    Returns (rp, value, err) numpy arrays.
    """
    data = np.loadtxt(filepath, comments="#")
    rp = data[:, 0]
    if data.shape[1] == 3:
        up, lo = data[:, 1], data[:, 2]
        value = np.sqrt(up * lo)        # log-space midpoint
        err   = (up - lo) / 2.0        # half linear range
    else:
        value = data[:, 1]
        err   = default_frac_err * value
    return rp, value, err


def write_wp_csv(out_path, rp, wp, wp_err):
    header = (
        "# Projected correlation function wp(rp) — Zu & Mandelbaum 2015 Fig. 6\n"
        "# Digitized with WebPlotDigitizer; midpoint in log-space where bounds given\n"
        "# rp and wp in h^-1 Mpc\n"
        "rp_hMpc,wp_hMpc,wp_err_hMpc"
    )
    rows = np.column_stack([rp, wp, wp_err])
    np.savetxt(out_path, rows, fmt="%.6f", delimiter=",", header=header, comments="")
    # numpy puts header before data but doesn't prefix '#' on first line — fix:
    with open(out_path) as f:
        content = f.read()
    lines = content.split("\n")
    # first 4 lines are the header written by numpy (without '#' prefix on col names)
    # re-write with proper CSV header
    with open(out_path, "w") as f:
        f.write("# Projected correlation function wp(rp) — Zu & Mandelbaum 2015 Fig. 6\n")
        f.write("# Digitized with WebPlotDigitizer; midpoint in log-space where bounds given\n")
        f.write("# rp and wp in h^-1 Mpc\n")
        f.write("rp_hMpc,wp_hMpc,wp_err_hMpc\n")
        for rp_i, wp_i, e_i in zip(rp, wp, wp_err):
            f.write(f"{rp_i:.6f},{wp_i:.6f},{e_i:.6f}\n")


def write_ds_csv(out_path, R, ds, ds_err):
    with open(out_path, "w") as f:
        f.write("# Excess surface density DeltaSigma(R) — Zu & Mandelbaum 2015 Fig. 6\n")
        f.write("# Digitized with WebPlotDigitizer; midpoint in log-space where bounds given\n")
        f.write("# R in h^-1 Mpc; DeltaSigma in M_sun h pc^-2\n")
        f.write("R_hMpc,ds_Msun_h_pc2,ds_err_Msun_h_pc2\n")
        for R_i, ds_i, e_i in zip(R, ds, ds_err):
            f.write(f"{R_i:.6f},{ds_i:.6f},{e_i:.6f}\n")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for label, lo_str, hi_str, lo_f, hi_f in BINS:
        # --- WPRP ---
        wp_txt  = _txt_fname("wprp", lo_str, hi_str)
        wp_csv  = os.path.join(DATA_DIR, f"wp_bin_{lo_str}_{hi_str}.csv")
        if os.path.isfile(wp_txt):
            rp, wp, wp_err = read_txt(wp_txt, default_frac_err=0.15)
            write_wp_csv(wp_csv, rp, wp, wp_err)
            print(f"[wp] {label:12s}  {len(rp):2d} pts  →  {os.path.basename(wp_csv)}")
        else:
            print(f"[wp] {label:12s}  MISSING: {wp_txt}")

        # --- ESD ---
        if label in ESD_SKIP:
            print(f"[ds] {label:12s}  SKIPPED (lowest 2 bins not used)")
            continue
        ds_txt = _txt_fname("esd", lo_str, hi_str)
        ds_csv = os.path.join(DATA_DIR, f"ds_bin_{lo_str}_{hi_str}.csv")
        if os.path.isfile(ds_txt):
            R, ds, ds_err = read_txt(ds_txt, default_frac_err=0.20)
            write_ds_csv(ds_csv, R, ds, ds_err)
            print(f"[ds] {label:12s}  {len(R):2d} pts  →  {os.path.basename(ds_csv)}")
        else:
            print(f"[ds] {label:12s}  MISSING: {ds_txt}")

    print("\nDone.")


if __name__ == "__main__":
    main()
