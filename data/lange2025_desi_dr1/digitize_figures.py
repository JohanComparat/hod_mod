#!/usr/bin/env python3
"""
Reusable figure digitizer for multi-panel log-x linear-y plots.

Axis conventions for Lange+2025 (arXiv:2512.15962):
  wp panels:   x = rp [Mpc/h],  y-axis = rp * wp    [(Mpc/h)^2]
  ESD panels:  x = R  [Mpc/h],  y-axis = R  * dS    [(Mpc/h)(M_sun/pc^2)]

Unit convention in stored CSVs (matching HOD model output):
  wp_hMpc       : wp  [h^-1 Mpc]        = (rp*wp) / rp
  ds_Msun_h_pc2 : DeltaSigma [M_sun h^-1 pc^-2]
                  = (R*dS_paper [M_sun/pc^2]) / R  / h_cosmo
                  Paper M_sun/pc^2 -> code M_sun h^-1/pc^2 : DIVIDE by h.
                  (h is a dimensionless number, not a unit label.)

Extraction method (auto-detect):
  1. Build column profile (colored pixels per column, excluding top row_skip_top rows
     which contain the panel title text).
  2. Detect peaks in the smoothed column profile -> actual x-positions.
  3. For each x-peak:
       a. x-centroid (column-weighted mean) -> rp [Mpc/h]
       b. Row density argmax + ±body_radius window centroid -> y-center
       c. Full vertical colored-pixel extent -> error bar half-height

Usage:
    python digitize_figures.py [--dry-run] [--panel N] [--compare] [--inspect]
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Axis calibration
# ---------------------------------------------------------------------------

def axis_transformation(calib_px, calib_data):
    """Least-squares linear pixel -> data transform.

    calib_px   : list of (col, row)
    calib_data : list of (data_x=log10(rp), data_y)

    Returns (sX, offX), (sY, offY)  with
        data_x = sX*col + offX
        data_y = sY*row + offY
    """
    cols = np.array([p[0] for p in calib_px], dtype=float)
    rows = np.array([p[1] for p in calib_px], dtype=float)
    dx   = np.array([p[0] for p in calib_data], dtype=float)
    dy   = np.array([p[1] for p in calib_data], dtype=float)

    A = np.column_stack([cols, np.ones(len(cols))])
    sX, offX = np.linalg.lstsq(A, dx, rcond=None)[0]

    A = np.column_stack([rows, np.ones(len(rows))])
    sY, offY = np.linalg.lstsq(A, dy, rcond=None)[0]

    return (sX, offX), (sY, offY)


# ---------------------------------------------------------------------------
# Panel configuration
# ---------------------------------------------------------------------------

@dataclass
class PanelConfig:
    """Configuration for one subplot panel."""
    fig_path: str
    col_left: int
    col_right: int
    row_top: int
    row_bot: int
    # Calibration: list of (col_px, row_px, log10_rp, y_value)
    calib: List[Tuple[int, int, float, float]]
    # Output CSV path (relative to this script's parent directory)
    output_csv: str = ""
    # Column names written to CSV
    col_x:    str = "rp_hMpc"
    col_y:    str = "y"
    col_yerr: str = "y_err"
    # Marker color (RGB) and tolerance
    data_color_rgb: Tuple[int, int, int] = (140, 200, 40)
    color_tol: int = 40
    # Search window (pixels) around each detected column peak
    search_window: int = 8
    # Rows to skip at the panel top (panel title text in tracer color)
    row_skip_top: int = 40
    # Local column threshold: apply row_skip_top only for pk_local < title_col_max
    # (title text occupies leftmost ~50-60 local cols; BAO-bump data beyond that)
    title_col_max: int = 65
    # Minimum column-peak distance for find_peaks (pixels)
    peak_distance: int = 8
    # Radius (rows) around argmax used for y-centroid
    body_radius: int = 6
    # Tight column half-width [pixels] for row-density computation (y detection).
    # Narrower than search_window to exclude adjacent-bin error-bar contamination.
    body_window: int = 4
    # If True: use topmost dense row-cluster as disc body (not argmax).
    # Needed for wp panels where error-bar caps pile up at y=0 and can beat the disc.
    # Leave False for ESD panels (symmetric error bars; argmax finds disc reliably).
    use_topmost_cluster: bool = False
    # If set: bypass column-profile peak detection and use these known x-bin values.
    # Useful for ESD panels where bins are 8 px apart and discs merge in column profile.
    force_x_bins: Optional[List[float]] = None
    # h_cosmo: 1.0 for wp (no conversion); h value for ESD (divide paper M_sun/pc^2 by h)
    h_cosmo: float = 1.0
    # Scale cut stored in metadata (not applied to extraction)
    rp_min_analysis: float = 0.0
    # Minimum x value [Mpc/h] to accept; filters false peaks at small R/rp
    x_extract_min: float = 0.0
    # Maximum x value [Mpc/h] to accept; filters model-band peaks beyond data range
    x_extract_max: float = 1e9
    # Human-readable y-axis label for comparison figures
    y_label: str = ""


# ---------------------------------------------------------------------------
# Color inspection helper
# ---------------------------------------------------------------------------

def find_data_color(img_arr, col_left, col_right, row_top, row_bot):
    """Return the 8 most common non-grey colors in the panel region."""
    panel = img_arr[row_top:row_bot+1, col_left:col_right+1, :3]
    r, g, b = panel[:,:,0].astype(int), panel[:,:,1].astype(int), panel[:,:,2].astype(int)
    gv = (np.abs(r-g) + np.abs(g-b) + np.abs(r-b)) / 3
    colorful = (gv > 20) & ((r+g+b) < 700) & ((r+g+b) > 100)
    px = panel[colorful]
    if len(px) == 0:
        return []
    rounded = (px // 20) * 20
    unique, counts = np.unique(rounded.reshape(-1, 3), axis=0, return_counts=True)
    idx = np.argsort(-counts)
    return [(tuple(unique[i]), counts[i]) for i in idx[:8]]


# ---------------------------------------------------------------------------
# Extraction engine
# ---------------------------------------------------------------------------

def extract_panel(panel: PanelConfig, img_arr: np.ndarray, verbose: bool = True):
    """Auto-detect data markers; return DataFrame with rp, y, y_err.

    Steps
    -----
    1. Build color mask for the image.
    2. Build column profile from the panel region BELOW row_skip_top rows
       (title text in tracer color would otherwise create false peaks).
    3. Gaussian-smooth + find_peaks -> actual marker x-positions.
    4. For each x-peak:
         x-centroid (column-weighted) -> rp
         Row density argmax + ±body_radius centroid -> y-center (marker body)
         Full colored-row extent -> error bar half-height
    5. Unit conversion:
         wp:  y_signal = rp_y_center / rp_val          [h^-1 Mpc]
         ESD: y_signal = rp_y_center / rp_val / h_cosmo [M_sun h^-1 pc^-2]
              (paper M_sun/pc^2 -> code M_sun h^-1/pc^2 requires DIVIDING by h)
    """
    calib_px   = [(c[0], c[1]) for c in panel.calib]
    calib_data = [(c[2], c[3]) for c in panel.calib]
    (sX, offX), (sY, offY) = axis_transformation(calib_px, calib_data)

    if verbose:
        print(f"  Calibration: sX={sX:.5f} offX={offX:.4f}  sY={sY:.5f} offY={offY:.4f}")

    # Build color mask for entire image
    r = img_arr[:, :, 0].astype(int)
    g = img_arr[:, :, 1].astype(int)
    b = img_arr[:, :, 2].astype(int)
    cr, cg, cb = panel.data_color_rgb
    tol = panel.color_tol
    color_mask = (
        (np.abs(r - cr) < tol) &
        (np.abs(g - cg) < tol) &
        (np.abs(b - cb) < tol)
    )

    # ---- Step 1: column peak positions ----
    full_panel_mask = color_mask[panel.row_top:panel.row_bot + 1,
                                  panel.col_left:panel.col_right + 1]
    col_profile = full_panel_mask.sum(axis=0).astype(float)

    # Build (pk_local, forced_xval_or_None) pairs
    if panel.force_x_bins is not None:
        # Known bin positions: use exact x value; no centroid (bins are 8 px apart,
        # adjacent discs overlap in col_profile and bias the centroid).
        peaks_info = []
        for xval in panel.force_x_bins:
            if xval < panel.x_extract_min:
                continue
            col_abs = (np.log10(xval) - offX) / sX
            pk_local = int(round(col_abs - panel.col_left))
            if 0 <= pk_local < full_panel_mask.shape[1]:
                peaks_info.append((pk_local, xval))
        if verbose:
            print(f"  {len(peaks_info)} forced x-bin positions")
    else:
        # Auto-detect: Gaussian-smooth + find_peaks on column profile
        # Must include top rows so BAO-bump data (rp*wp~160-190, near top) is detected.
        col_smooth  = gaussian_filter1d(col_profile, sigma=1.5)
        auto_peaks, _ = find_peaks(col_smooth, height=2,
                                    distance=panel.peak_distance)
        peaks_info = [(int(pk), None) for pk in auto_peaks]
        if verbose:
            print(f"  {len(peaks_info)} column peaks detected")

    records = []
    for pk_local, forced_xval in peaks_info:
        # Variable row_start: skip title-text rows only in leftmost columns.
        # Title text "BGS2/BGS3/LRG1/LRG2" occupies local col < title_col_max.
        # Beyond that, what looks like "text" is actually real BAO-bump data.
        if pk_local < panel.title_col_max:
            row_start = panel.row_top + panel.row_skip_top
        else:
            row_start = panel.row_top

        if forced_xval is not None:
            # ---- Forced path: use exact known x; refine column via col_profile ----
            # Find actual disc column near theoretical position (handles calibration
            # offsets up to ±5 px). Keep R exact regardless of column refinement.
            col_theoretical = (np.log10(forced_xval) - offX) / sX  # absolute col
            search_refine = 5
            lo_r = max(0, pk_local - search_refine)
            hi_r = min(full_panel_mask.shape[1] - 1, pk_local + search_refine)
            local_profile = col_profile[lo_r:hi_r + 1]
            if local_profile.max() >= 3:
                refined_pk = lo_r + int(np.argmax(local_profile))
                col_centroid = refined_pk + panel.col_left
            else:
                col_centroid = col_theoretical
            rp_val = forced_xval  # exact known R — never shift with centroid
            eff_body_window = panel.body_window
        else:
            # ---- Auto path: x-centroid from column profile ----
            lo   = max(0, pk_local - panel.search_window)
            hi   = min(full_panel_mask.shape[1] - 1, pk_local + panel.search_window)
            col_lo = lo + panel.col_left
            col_hi = hi + panel.col_left
            # ---- Step 2: x-centroid ----
            cw = col_profile[lo:hi + 1]
            if cw.sum() == 0:
                continue
            cols_abs = np.arange(col_lo, col_hi + 1, dtype=float)
            col_centroid = np.dot(cols_abs, cw) / cw.sum()
            rp_val = 10 ** (sX * col_centroid + offX)
            # Filter false peaks at x outside the valid range
            if rp_val < panel.x_extract_min or rp_val > panel.x_extract_max:
                continue
            eff_body_window = panel.body_window

        # ---- Step 3: row density in a window around the x-centroid ----
        # body_window (auto path) is narrower than search_window so that adjacent
        # bins' error-bar caps (which pile up near y=0 across many x positions)
        # contribute at most 1-2 px/row, while the disc body contributes 6-10 px/row.
        b_lo = max(panel.col_left, int(round(col_centroid)) - eff_body_window)
        b_hi = min(panel.col_right, int(round(col_centroid)) + eff_body_window)

        row_dens = color_mask[row_start:panel.row_bot + 1,
                               b_lo:b_hi + 1].sum(axis=1).astype(float)
        all_colored = np.where(row_dens > 0)[0]
        if len(all_colored) == 0:
            continue

        # Marker body: locate disc center and error bar extent.
        # Two strategies depending on panel type:
        #
        # WP panels (use_topmost_cluster=True): error bars run from disc DOWN to y=0;
        #   caps pile up at y=0 and would bias all_colored.max(). Use the topmost dense
        #   cluster as the disc body anchor, then walk outward stopping at ≥2 gap rows.
        #
        # ESD panels (use_topmost_cluster=False): error bars are symmetric; the marker
        #   is an open circle with an empty interior. all_colored.min/max captures the
        #   full bar extent and their midpoint is the disc centre (unbiased).
        if panel.use_topmost_cluster:
            # Find topmost dense cluster
            dense = np.where(row_dens >= 2)[0]
            if len(dense) == 0:
                dense = np.where(row_dens > 0)[0]
            if len(dense) == 0:
                continue
            run_end = dense[0]
            for i in range(1, len(dense)):
                if dense[i] <= dense[i - 1] + 2:
                    run_end = dense[i]
                else:
                    break
            body_peak = (dense[0] + run_end) // 2
            blo = max(0, body_peak - panel.body_radius)
            bhi = min(len(row_dens) - 1, body_peak + panel.body_radius)
            bd  = row_dens[blo:bhi + 1]
            ba  = (np.arange(blo, bhi + 1) + row_start).astype(float)
            if bd.sum() == 0:
                continue
            center_row = np.dot(ba, bd) / bd.sum()
            # Walk outward from body centre; stop at ≥2 consecutive zero rows
            body_row_rel = int(round(center_row - row_start))
            body_row_rel = max(0, min(body_row_rel, len(row_dens) - 1))
            bt_rel = body_row_rel
            gap_count = 0
            for r_idx in range(body_row_rel - 1, -1, -1):
                if row_dens[r_idx] > 0:
                    bt_rel = r_idx; gap_count = 0
                else:
                    gap_count += 1
                    if gap_count >= 2:
                        break
            bb_rel = body_row_rel
            gap_count = 0
            for r_idx in range(body_row_rel + 1, len(row_dens)):
                if row_dens[r_idx] > 0:
                    bb_rel = r_idx; gap_count = 0
                else:
                    gap_count += 1
                    if gap_count >= 2:
                        break
            bar_top = float(bt_rel + row_start)
            bar_bot = float(bb_rel + row_start)
        else:
            # ESD open-circle markers with symmetric error bars.
            # Disc centre = midpoint of the full colored-pixel extent.
            bar_top = float(all_colored.min() + row_start)
            bar_bot = float(all_colored.max() + row_start)
            center_row = (bar_top + bar_bot) / 2.0

        # ---- Step 4: convert to data values ----
        rp_y_center = sY * center_row + offY
        rp_y_half   = abs(sY * (bar_bot - bar_top) / 2.0)

        # wp:  y = rp_y_center / rp_val            [h^-1 Mpc]
        # ESD: y = rp_y_center / rp_val / h_cosmo  [M_sun h^-1 pc^-2]
        y_signal = rp_y_center / rp_val / panel.h_cosmo
        y_err    = rp_y_half   / rp_val / panel.h_cosmo

        records.append((rp_val, y_signal, y_err))

        if verbose:
            print(f"    rp={rp_val:.4f}  rp*y={rp_y_center:.3f}  "
                  f"y={y_signal:.4f}  yerr={y_err:.4f}")

    records.sort(key=lambda x: x[0])
    return pd.DataFrame(records, columns=[panel.col_x, panel.col_y, panel.col_yerr])


# ---------------------------------------------------------------------------
# Run panel
# ---------------------------------------------------------------------------

def run_panel(panel: PanelConfig, img_arr: np.ndarray,
              dry_run: bool = False, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"\n--- {panel.output_csv} ---")
    df = extract_panel(panel, img_arr, verbose=verbose)
    if verbose:
        print(df.to_string(index=False))
    if not dry_run and panel.output_csv:
        out = Path(panel.fig_path).parent.parent / panel.output_csv
        df.to_csv(out, index=False, float_format="%.6f")
        if verbose:
            print(f"  Wrote {out}")
    return df


# ---------------------------------------------------------------------------
# Comparison figures
# ---------------------------------------------------------------------------

def make_overlay_figure(fig_path, panels, results, output_suffix="_overlay",
                        verbose=True):
    """Overlay extracted points on the original PNG in pixel coordinates."""
    img = np.array(Image.open(fig_path))
    fig, ax = plt.subplots(figsize=(img.shape[1]/72, img.shape[0]/72), dpi=150)
    ax.imshow(img, origin="upper", aspect="auto")

    for panel, df in zip(panels, results):
        calib_px   = [(c[0], c[1]) for c in panel.calib]
        calib_data = [(c[2], c[3]) for c in panel.calib]
        (sX, offX), (sY, offY) = axis_transformation(calib_px, calib_data)

        for _, row in df.iterrows():
            xv = row[panel.col_x]
            yv = row[panel.col_y]
            ye = row[panel.col_yerr]
            if np.isnan(yv):
                continue
            # Reverse transform: stored value -> paper plotted value (rp*y)
            rp_y_cen = yv * xv * panel.h_cosmo
            rp_y_err = ye * xv * panel.h_cosmo

            col_px  = (np.log10(xv) - offX) / sX
            row_cen = (rp_y_cen - offY) / sY
            row_top = (rp_y_cen + rp_y_err - offY) / sY
            row_bot = (rp_y_cen - rp_y_err - offY) / sY

            used  = (xv >= panel.rp_min_analysis - 1e-6)
            color = "red" if used else "salmon"
            ax.plot(col_px, row_cen, "o", ms=4, color=color, mec="darkred", mew=0.4, zorder=10)
            ax.plot([col_px, col_px], [row_top, row_bot], "-", lw=0.8, color=color, zorder=9)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    plt.tight_layout(pad=0)
    out = Path(fig_path).with_suffix("").as_posix() + output_suffix + ".png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Overlay -> {out}")


def make_standalone_figure(panels, results, output_path, verbose=True):
    """Recreate the paper figure layout with extracted data only."""
    wp_pairs  = [(p, d) for p, d in zip(panels, results) if "wp"  in p.output_csv]
    esd_pairs = [(p, d) for p, d in zip(panels, results) if "ds_" in p.output_csv]

    n_col = max(len(wp_pairs), len(esd_pairs))
    n_row = (1 if wp_pairs else 0) + (1 if esd_pairs else 0)
    if n_col == 0:
        return

    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(4.0*n_col, 3.5*n_row), squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    if wp_pairs:
        for ci, (panel, df) in enumerate(wp_pairs):
            ax = axes[0, ci]
            sample = Path(panel.output_csv).parent.name
            x = df[panel.col_x].values
            y = df[panel.col_y].values
            e = df[panel.col_yerr].values
            rp_wp = x * y
            rp_wp_e = x * e
            used = x >= panel.rp_min_analysis - 1e-6
            ax.errorbar(x[used],  rp_wp[used],  rp_wp_e[used],
                        fmt="o", ms=4, capsize=2)
            if np.any(~used):
                ax.errorbar(x[~used], rp_wp[~used], rp_wp_e[~used],
                            fmt="o", ms=4, capsize=2, color="grey")
            if panel.rp_min_analysis > 0:
                ax.axvline(panel.rp_min_analysis, ls="--", color="grey", lw=0.8)
            ax.set_xscale("log"); ax.set_xlim(0.08, 70); ax.set_ylim(0, 200)
            ax.set_xlabel("$r_p$ [Mpc/h]")
            ax.set_ylabel("$r_p w_p$ [(Mpc/h)$^2$]")
            ax.set_title(sample)

    if esd_pairs:
        ri = 1 if wp_pairs else 0
        for ci, (panel, df) in enumerate(esd_pairs):
            ax = axes[ri, ci]
            sample  = Path(panel.output_csv).parent.name
            src = "DES" if "des" in panel.output_csv else "HSC"
            x = df[panel.col_x].values
            y = df[panel.col_y].values
            e = df[panel.col_yerr].values
            # Back to paper units for the plot
            ds_p = y * panel.h_cosmo   # code -> paper: multiply by h
            ds_e = e * panel.h_cosmo
            R_ds = x * ds_p
            R_ds_e = x * ds_e
            ax.errorbar(x, R_ds, R_ds_e, fmt="o", ms=4, capsize=2)
            ax.set_xscale("log"); ax.set_xlim(0.08, 70); ax.set_ylim(0, 12)
            ax.set_xlabel("$R$ [Mpc/h]")
            ax.set_ylabel(r"$R\,\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]")
            ax.set_title(f"{sample} {src}")

    plt.suptitle("Extracted data — Lange+2025 DESI DR1", y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Standalone -> {output_path}")


# ---------------------------------------------------------------------------
# Lange+2025 DESI DR1 panel configurations
# ---------------------------------------------------------------------------
#
# x-axis calibration (log-scale):
#   fig3:  col 64=0.1, 151=1, 238=10 Mpc/h   (87 px/decade)
#   fig4:  col 64=0.1, 129=1, 194=10 Mpc/h   (65 px/decade)
#
# wp y-axis ticks:
#   fig3:  row  54=150, 103=100, 153=50  (Mpc/h)^2
#   fig4:  row  12=200,  58=150, 106=100, 151=50
#
# ESD y-axis ticks:
#   fig3:  row 250=10.0, 293=7.5, 335=5.0, 380=2.5  (Mpc/h)(M_sun/pc^2)
#   fig4:  row 263=10.0, 302=7.5, 341=5.0, 379=2.5
#
# Panel title text in tracer color occupies roughly the first 40 relative rows.
# row_skip_top=45 reliably excludes it without cutting into data.
#
# ESD panels: markers are closer together -> peak_distance=5; search_window=5.

_DATA_DIR = Path(__file__).parent
_FIG3 = str(_DATA_DIR / "raw_figures/fig3.png")
_FIG4 = str(_DATA_DIR / "raw_figures/fig4.png")

# Known ESD bin centers [h^-1 Mpc] — log-spaced at ratio ~1.238 (8 px apart in fig3).
# Extends into grey scale-cut region (R < 5.5 Mpc/h) to capture data shown but excluded from fit.
# Using forced positions bypasses column-profile peak merging for closely-spaced bins.
_ESD_BINS = [1.817, 2.249, 2.781329, 3.442524, 4.260902, 5.273830, 6.527558, 8.079331,
             10.000000, 12.377263, 15.319664, 18.961550, 23.469209, 29.048457, 35.954039]

_H_COSMO = 0.6736   # Planck 2018 h for Lange+2025

# Column boundaries (absolute pixel columns in full image)
_FIG3_CL = [61,  321, 583]
_FIG3_CR = [311, 572, 834]
_FIG4_CL = [61,  257, 452, 648]
_FIG4_CR = [248, 443, 639, 834]

# Per-panel x-axis offset: relative column from col_left where rp=0.1 Mpc/h gridline sits.
# Pixel analysis of x-axis tick marks shows panel 0 at rel=3, panels 1-2 at rel=5 (+2 px shift).
# This 2-px layout difference causes ~5% rp overestimation in the uncorrected panels.
_FIG3_X0_REL = [3, 5, 5]   # for panels 0 (BGS2), 1 (BGS3), 2 (LRG1)

# Marker colors
_BGS2_COLOR = (140, 200,  40)
_BGS3_COLOR = (240, 200,   0)
_LRG1_COLOR = (240, 160,   0)
_LRG2_COLOR = (240,   0,   0)


def _calib_fig3_top(i):
    # x0: absolute column of rp=0.1 Mpc/h gridline for panel i.
    # Panels 1 and 2 are shifted +2 px right relative to panel 0 (measured from tick marks).
    x0 = _FIG3_CL[i] + _FIG3_X0_REL[i]
    return [(x0,       54, -1.0, 150.0),
            (x0 + 87, 103,  0.0, 100.0),
            (x0 +174, 153,  1.0,  50.0)]

def _calib_fig3_bot(i):
    x0 = _FIG3_CL[i] + _FIG3_X0_REL[i]
    return [(x0,       250, -1.0, 10.0),
            (x0 + 87, 335,  0.0,  5.0),
            (x0 +174, 380,  1.0,  2.5)]

def _calib_fig4_top(i):
    dx = _FIG4_CL[i] - _FIG4_CL[0]
    return [(64+dx,   12, -1.0, 200.0),
            (129+dx, 106,  0.0, 100.0),
            (194+dx, 151,  1.0,  50.0)]

def _calib_fig4_bot(i):
    dx = _FIG4_CL[i] - _FIG4_CL[0]
    return [(64+dx,  263, -1.0, 10.0),
            (129+dx, 341,  0.0,  5.0),
            (194+dx, 379,  1.0,  2.5)]


PANEL_CONFIGS = [
    # ── fig3 top: wp ──────────────────────────────────────────────────────
    # row_bot=205: y-calibration gives row~202 for y=0 -> captures all data
    # title_col_max=65: skip top rows only for col<65 (actual title text region)
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_CL[0], col_right=_FIG3_CR[0],
        row_top=5, row_bot=205, calib=_calib_fig3_top(0),
        data_color_rgb=_BGS2_COLOR, h_cosmo=1.0, rp_min_analysis=0.4,
        row_skip_top=40, title_col_max=80,
        peak_distance=8, search_window=8, body_radius=6, use_topmost_cluster=True,
        output_csv="BGS2/wp_bgs2.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
        y_label="$r_p w_p$ [(Mpc/h)$^2$]",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_CL[1], col_right=_FIG3_CR[1],
        row_top=5, row_bot=205, calib=_calib_fig3_top(1),
        data_color_rgb=_BGS3_COLOR, h_cosmo=1.0, rp_min_analysis=0.4,
        row_skip_top=40, title_col_max=80,
        peak_distance=8, search_window=8, body_radius=6, use_topmost_cluster=True,
        output_csv="BGS3/wp_bgs3.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
        y_label="$r_p w_p$ [(Mpc/h)$^2$]",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_CL[2], col_right=_FIG3_CR[2],
        row_top=5, row_bot=205, calib=_calib_fig3_top(2),
        data_color_rgb=_LRG1_COLOR, h_cosmo=1.0, rp_min_analysis=0.4,
        row_skip_top=40, title_col_max=80,
        peak_distance=8, search_window=8, body_radius=6, use_topmost_cluster=True,
        output_csv="LRG1/wp_lrg1.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
        y_label="$r_p w_p$ [(Mpc/h)$^2$]",
    ),
    # ── fig3 bottom: DES ESD ──────────────────────────────────────────────
    # force_x_bins=_ESD_BINS: snap to known log-spaced bin centers (avoids peak-merge
    # artefacts from closely-spaced discs and prevents model-curve false detections).
    # _ESD_BINS starts at 1.817 Mpc/h to capture data in the grey scale-cut region.
    # rp_min_analysis=5.5: marks the analysis scale cut (red=used, salmon=excluded).
    # x_extract_max=37: drops any peak beyond the outermost data bin.
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_CL[0], col_right=_FIG3_CR[0],
        row_top=233, row_bot=383, calib=_calib_fig3_bot(0),
        data_color_rgb=_BGS2_COLOR, h_cosmo=_H_COSMO,
        row_skip_top=45, title_col_max=65, x_extract_min=1.5, x_extract_max=37.0,
        rp_min_analysis=5.5,
        peak_distance=8, search_window=8, body_radius=6,
        force_x_bins=_ESD_BINS,
        output_csv="BGS2/ds_des_bgs2.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
        y_label=r"$R\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_CL[1], col_right=_FIG3_CR[1],
        row_top=233, row_bot=383, calib=_calib_fig3_bot(1),
        data_color_rgb=_BGS3_COLOR, h_cosmo=_H_COSMO,
        row_skip_top=45, title_col_max=65, x_extract_min=1.5, x_extract_max=37.0,
        rp_min_analysis=5.5,
        peak_distance=8, search_window=8, body_radius=6,
        force_x_bins=_ESD_BINS,
        output_csv="BGS3/ds_des_bgs3.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
        y_label=r"$R\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_CL[2], col_right=_FIG3_CR[2],
        row_top=233, row_bot=383, calib=_calib_fig3_bot(2),
        data_color_rgb=_LRG1_COLOR, h_cosmo=_H_COSMO,
        row_skip_top=45, title_col_max=65, x_extract_min=1.5, x_extract_max=37.0,
        rp_min_analysis=5.5,
        peak_distance=8, search_window=8, body_radius=6,
        force_x_bins=_ESD_BINS,
        output_csv="LRG1/ds_des_lrg1.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
        y_label=r"$R\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]",
    ),
    # ── fig4 top: LRG2 wp ─────────────────────────────────────────────────
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_CL[3], col_right=_FIG4_CR[3],
        row_top=6, row_bot=200, calib=_calib_fig4_top(3),
        data_color_rgb=_LRG2_COLOR, h_cosmo=1.0, rp_min_analysis=0.4,
        row_skip_top=40, title_col_max=80,
        peak_distance=8, search_window=8, body_radius=6, use_topmost_cluster=True,
        output_csv="LRG2/wp_lrg2.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
        y_label="$r_p w_p$ [(Mpc/h)$^2$]",
    ),
    # ── fig4 bottom: HSC ESD ──────────────────────────────────────────────
    # force_x_bins=_ESD_BINS: same strategy as DES ESD; captures grey-area data points.
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_CL[0], col_right=_FIG4_CR[0],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(0),
        data_color_rgb=_BGS2_COLOR, h_cosmo=_H_COSMO,
        row_skip_top=45, title_col_max=65, x_extract_min=1.5, x_extract_max=37.0,
        rp_min_analysis=5.5,
        peak_distance=8, search_window=8, body_radius=6,
        force_x_bins=_ESD_BINS,
        output_csv="BGS2/ds_hsc_bgs2.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
        y_label=r"$R\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]",
    ),
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_CL[1], col_right=_FIG4_CR[1],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(1),
        data_color_rgb=_BGS3_COLOR, h_cosmo=_H_COSMO,
        row_skip_top=45, title_col_max=65, x_extract_min=1.5, x_extract_max=37.0,
        rp_min_analysis=5.5,
        peak_distance=8, search_window=8, body_radius=6,
        force_x_bins=_ESD_BINS,
        output_csv="BGS3/ds_hsc_bgs3.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
        y_label=r"$R\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]",
    ),
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_CL[2], col_right=_FIG4_CR[2],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(2),
        data_color_rgb=_LRG1_COLOR, h_cosmo=_H_COSMO,
        row_skip_top=45, title_col_max=65, x_extract_min=1.5, x_extract_max=37.0,
        rp_min_analysis=5.5,
        peak_distance=8, search_window=8, body_radius=6,
        force_x_bins=_ESD_BINS,
        output_csv="LRG1/ds_hsc_lrg1.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
        y_label=r"$R\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]",
    ),
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_CL[3], col_right=_FIG4_CR[3],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(3),
        data_color_rgb=_LRG2_COLOR, h_cosmo=_H_COSMO,
        row_skip_top=45, title_col_max=65, x_extract_min=1.5, x_extract_max=37.0,
        rp_min_analysis=5.5,
        peak_distance=8, search_window=8, body_radius=6,
        force_x_bins=_ESD_BINS,
        output_csv="LRG2/ds_hsc_lrg2.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
        y_label=r"$R\Delta\Sigma$ [(Mpc/h)(M$_\odot$/pc$^2$)]",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print extracted values without writing CSV files.")
    ap.add_argument("--inspect",  action="store_true",
                    help="Print dominant marker colors per panel and exit.")
    ap.add_argument("--compare",  action="store_true",
                    help="Generate overlay and standalone comparison figures.")
    ap.add_argument("--panel",    type=int, default=None,
                    help="Process only panel INDEX (0-based).")
    args = ap.parse_args()

    loaded_imgs: dict = {}
    panels = PANEL_CONFIGS if args.panel is None else [PANEL_CONFIGS[args.panel]]

    if args.inspect:
        for i, panel in enumerate(panels):
            img_arr = loaded_imgs.setdefault(
                panel.fig_path, np.array(Image.open(panel.fig_path)))
            colors = find_data_color(
                img_arr, panel.col_left, panel.col_right, panel.row_top, panel.row_bot)
            print(f"Panel {i} ({panel.output_csv}):")
            for color, count in colors:
                print(f"  RGB{color}: {count} px")
        return

    all_results = []
    for panel in panels:
        img_arr = loaded_imgs.setdefault(
            panel.fig_path, np.array(Image.open(panel.fig_path)))
        df = run_panel(panel, img_arr, dry_run=args.dry_run, verbose=True)
        all_results.append(df)

    print("\nDone.")

    if args.compare:
        print("\nGenerating comparison figures...")
        fig_groups: dict = {}
        for panel, df in zip(panels, all_results):
            fig_groups.setdefault(panel.fig_path, []).append((panel, df))

        for fig_path, pairs in fig_groups.items():
            p_list = [x[0] for x in pairs]
            d_list = [x[1] for x in pairs]
            make_overlay_figure(fig_path, p_list, d_list)
            stem = Path(fig_path).stem
            make_standalone_figure(
                p_list, d_list,
                str(_DATA_DIR / f"raw_figures/{stem}_extracted.pdf"))


if __name__ == "__main__":
    main()
