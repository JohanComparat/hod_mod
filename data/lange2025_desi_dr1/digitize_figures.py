#!/usr/bin/env python3
"""
Reusable figure digitizer for multi-panel log-x linear-y plots.

Figures in HOD analysis papers typically show rp*wp(rp) and rp*Delta_Sigma(rp)
with a log-scale x-axis (rp in h^-1 Mpc) and a linear y-axis.

This script uses color-based marker detection to extract data points and
error bars from each subplot panel.

Usage:
    python digitize_figures.py [--dry-run] [--plot]

Configuration:
    Edit the PANEL_CONFIGS dict below to adapt to other figures.

Reuse guide:
    1. Set FIG_DIR to the directory containing the PNG figures.
    2. For each figure, define panel_col_left/right and row bounds.
    3. Identify x/y tick pixel positions and their data values by viewing
       the figures (use the --inspect flag or crop and view with PIL).
    4. Set data_color_rgb to the dominant color of the data points
       (run a quick color analysis: see find_data_color() below).
    5. Run and verify: extracted rp values should match known bin centers.

Calibration (axis_transformation):
    Each panel needs 3+ calibration points:
        (col_pixel, row_pixel, data_x_value, data_y_value)
    where data_x is log10(rp) and data_y is the LINEAR product rp*y plotted.
    Use intersections of axis tick lines (both x and y tick known) as points.

Output:
    Overwrites the CSV files listed in each panel config.
    Columns: as specified by col_x, col_y, col_yerr in each panel config.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Axis transformation (from plotdigitizer; reproduced here for clarity)
# ---------------------------------------------------------------------------

def axis_transformation(calib_px, calib_data):
    """
    Compute linear pixel -> data transformation from calibration points.

    Parameters
    ----------
    calib_px : list of (col, row)   pixel coordinates
    calib_data : list of (data_x, data_y)  known data values at those pixels

    Returns
    -------
    (sX, offX), (sY, offY)  such that
        data_x = sX * col + offX
        data_y = sY * row + offY
    """
    cols = np.array([p[0] for p in calib_px], dtype=float)
    rows = np.array([p[1] for p in calib_px], dtype=float)
    dx = np.array([p[0] for p in calib_data], dtype=float)
    dy = np.array([p[1] for p in calib_data], dtype=float)

    # Least-squares linear fits
    A = np.column_stack([cols, np.ones(len(cols))])
    sX, offX = np.linalg.lstsq(A, dx, rcond=None)[0]

    A = np.column_stack([rows, np.ones(len(rows))])
    sY, offY = np.linalg.lstsq(A, dy, rcond=None)[0]

    return (sX, offX), (sY, offY)


# ---------------------------------------------------------------------------
# Panel configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PanelConfig:
    """Configuration for one subplot panel."""
    fig_path: str
    col_left: int       # leftmost data column (x-axis left border)
    col_right: int      # rightmost data column
    row_top: int        # topmost data row (y-axis top border)
    row_bot: int        # bottommost data row
    # Calibration: list of (col, row, data_x, data_y)
    # data_x = log10(rp), data_y = rp*wp or rp*Delta_Sigma (the plotted product)
    calib: List[Tuple[int, int, float, float]]
    # Known bin positions in x (physical units, e.g. h^-1 Mpc)
    x_bins: np.ndarray
    # Color of the data points (R, G, B), tolerance ~40
    data_color_rgb: Tuple[int, int, int] = (140, 200, 40)
    color_tol: int = 40
    # Output CSV path (relative to fig_path parent)
    output_csv: str = ""
    # Column names in output CSV
    col_x: str = "rp_hMpc"
    col_y: str = "y"
    col_yerr: str = "y_err"
    # Window (half-width in pixels) to search around expected x-column
    search_window: int = 8


# ---------------------------------------------------------------------------
# Extraction engine
# ---------------------------------------------------------------------------

def find_data_color(img_arr, col_left, col_right, row_top, row_bot):
    """
    Report the most common non-grey/non-white/non-black colors in the panel.
    Use this to identify the data_color_rgb for a new figure.
    """
    panel = img_arr[row_top:row_bot+1, col_left:col_right+1, :3]
    r, g, b = panel[:, :, 0], panel[:, :, 1], panel[:, :, 2]
    gray_var = (
        np.abs(r.astype(int) - g.astype(int))
        + np.abs(g.astype(int) - b.astype(int))
        + np.abs(r.astype(int) - b.astype(int))
    ) / 3
    colorful = (
        (gray_var > 20)
        & (r.astype(int) + g.astype(int) + b.astype(int) < 700)
        & (r.astype(int) + g.astype(int) + b.astype(int) > 100)
    )
    px = panel[colorful]
    if len(px) == 0:
        return []
    rounded = (px // 20) * 20
    unique, counts = np.unique(rounded.reshape(-1, 3), axis=0, return_counts=True)
    idx = np.argsort(-counts)
    return [(tuple(unique[i]), counts[i]) for i in idx[:8]]


def extract_panel(panel: PanelConfig, img_arr: np.ndarray, verbose: bool = True):
    """
    Extract signal and error-bar for each x-bin from a single panel.

    The figure is assumed to display rp*y vs log(rp) with a linear y-axis.
    After extraction:
        y_signal = (rp*y_extracted) / rp
        y_err    = (rp*y_half_span) / rp

    Returns
    -------
    pd.DataFrame with columns [col_x, col_y, col_yerr]
    """
    calib_px = [(c[0], c[1]) for c in panel.calib]
    calib_data = [(c[2], c[3]) for c in panel.calib]
    (sX, offX), (sY, offY) = axis_transformation(calib_px, calib_data)

    if verbose:
        print(f"  Calibration: sX={sX:.5f} offX={offX:.4f}  "
              f"sY={sY:.5f} offY={offY:.4f}")

    # Color mask over the full image
    r = img_arr[:, :, 0].astype(int)
    g = img_arr[:, :, 1].astype(int)
    b = img_arr[:, :, 2].astype(int)
    cr, cg, cb = panel.data_color_rgb
    tol = panel.color_tol
    color_mask = (
        (np.abs(r - cr) < tol)
        & (np.abs(g - cg) < tol)
        & (np.abs(b - cb) < tol)
    )

    records = []
    for x_bin in panel.x_bins:
        log_x = np.log10(x_bin)
        x_col_float = (log_x - offX) / sX
        x_col = int(round(x_col_float))

        col_lo = max(panel.col_left, x_col - panel.search_window)
        col_hi = min(panel.col_right, x_col + panel.search_window)

        # Collect rows with any colored pixel in [col_lo, col_hi]
        colored_rows = []
        colored_counts = []
        for row in range(panel.row_top, panel.row_bot + 1):
            n = int(np.sum(color_mask[row, col_lo : col_hi + 1]))
            if n > 0:
                colored_rows.append(row)
                colored_counts.append(n)

        if not colored_rows:
            if verbose:
                print(f"    WARNING: no colored pixels at rp={x_bin:.3f} "
                      f"(col≈{x_col})")
            records.append((x_bin, np.nan, np.nan))
            continue

        colored_rows = np.array(colored_rows)
        colored_counts = np.array(colored_counts)

        # Marker center = row with the most pixels (densest part of circle)
        center_row = colored_rows[np.argmax(colored_counts)]

        # Error bar extent
        bar_top_row = colored_rows.min()
        bar_bot_row = colored_rows.max()

        # Convert to data y-value (the plotted product rp*y)
        rp_y_center = sY * center_row + offY
        rp_y_half = abs(sY * (bar_bot_row - bar_top_row) / 2.0)

        # Divide by x_bin to recover y (wp or Delta_Sigma)
        y_signal = rp_y_center / x_bin
        y_err = rp_y_half / x_bin

        records.append((x_bin, y_signal, y_err))

    df = pd.DataFrame(records, columns=[panel.col_x, panel.col_y, panel.col_yerr])
    return df


def run_panel(panel: PanelConfig, img_arr: np.ndarray,
              dry_run: bool = False, verbose: bool = True, make_plot: bool = False):
    """Extract data from panel and write CSV."""
    if verbose:
        print(f"\nProcessing -> {panel.output_csv}")

    df = extract_panel(panel, img_arr, verbose=verbose)

    # Print summary
    if verbose:
        print(df.to_string(index=False))

    if make_plot:
        _plot_result(panel, df)

    if not dry_run and panel.output_csv:
        out_path = Path(panel.fig_path).parent.parent / panel.output_csv
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"  Wrote {out_path}")

    return df


def _plot_result(panel: PanelConfig, df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.errorbar(df[panel.col_x], df[panel.col_y], yerr=df[panel.col_yerr],
                fmt="o", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(panel.col_x)
    ax.set_ylabel(panel.col_y)
    ax.set_title(Path(panel.output_csv).stem)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Lange+2025 DESI DR1 configuration
# ---------------------------------------------------------------------------
#
# Figures 3 and 4 from arXiv:2512.15962, stored in raw_figures/.
#
# Figure layout (both 840x480 px):
#   fig3.png: 3 cols x 2 rows  (BGS2, BGS3, LRG1)
#   fig4.png: 4 cols x 2 rows  (BGS2, BGS3, LRG1, LRG2)
#
# Top row: rp*wp [h^-2 Mpc^2] vs rp [h^-1 Mpc]
# Bottom row: rp*Delta_Sigma [10^6 M_sun/pc] vs rp [h^-1 Mpc]
#
# Calibration derived from pixel analysis of axis tick marks (see README).
# X-axis tick marks:  col 64 = rp=0.1,  col 151 = rp=1,  col 238 = rp=10  (fig3)
#                     col 64 = rp=0.1,  col 129 = rp=1,  col 194 = rp=10  (fig4)
# fig3 top Y-ticks:   row 54 = 150,  row 103 = 100,  row 153 = 50   h^-2 Mpc^2
# fig4 top Y-ticks:   row 12 = 200,  row  58 = 150,  row 106 = 100,
#                     row 151 = 50   h^-2 Mpc^2
# fig3 bot Y-ticks:   row 250 = 10.0, row 293 = 7.5, row 335 = 5.0,
#                     row 380 = 2.5   [10^6 M_sun/pc]
# fig4 bot Y-ticks:   row 263 = 10.0, row 302 = 7.5, row 341 = 5.0,
#                     row 379 = 2.5   [10^6 M_sun/pc]

_DATA_DIR = Path(__file__).parent
_FIG3 = str(_DATA_DIR / "raw_figures/fig3.png")
_FIG4 = str(_DATA_DIR / "raw_figures/fig4.png")

# Known rp bin centers (from current CSV files -- log-spaced, exact positions)
_RP_BINS = np.array([
    0.501187, 0.794328, 1.258925, 1.995262, 3.162278,
    5.011872, 7.943282, 12.589254, 19.952623, 31.622777, 50.118723,
])
# Known R bin centers for ESD (13 bins)
_R_BINS = np.array([
    2.781329, 3.442524, 4.260902, 5.273830, 6.527558,
    8.079331, 10.000000, 12.377263, 15.319664, 18.961550,
    23.469209, 29.048457, 35.954039,
])

# fig3 column offsets per panel (left borders at cols 61, 321, 583)
# fig4 column offsets per panel (left borders at cols 61, 257, 452, 648)
_FIG3_COL_LEFT = [61, 321, 583]
_FIG3_COL_RIGHT = [311, 572, 834]
_FIG4_COL_LEFT = [61, 257, 452, 648]
_FIG4_COL_RIGHT = [248, 443, 639, 834]

# x-tick column offsets relative to the panel left border (same for all panels):
#   fig3: rp=0.1 at +3px, rp=1 at +90px, rp=10 at +177px
#   fig4: rp=0.1 at +3px, rp=1 at +68px, rp=10 at +133px
# So in absolute columns for panel i:
#   fig3: tick_rp01 = col_left[i]+3 = 64, 324, 586
#   fig4: tick_rp01 = col_left[i]+3 = 64, 260, 455, 651


def _calib_fig3_top(panel_idx):
    """3 calibration points for fig3 top-row panel i."""
    dx = _FIG3_COL_LEFT[panel_idx] - _FIG3_COL_LEFT[0]  # col offset vs panel 0
    return [
        (64 + dx,  54, -1.0, 150.0),  # rp=0.1, rp*wp=150
        (151 + dx, 103,  0.0, 100.0),  # rp=1,   rp*wp=100
        (238 + dx, 153,  1.0,  50.0),  # rp=10,  rp*wp=50
    ]


def _calib_fig3_bot(panel_idx):
    """3 calibration points for fig3 bottom-row panel i."""
    dx = _FIG3_COL_LEFT[panel_idx] - _FIG3_COL_LEFT[0]
    return [
        (64 + dx,  250, -1.0, 10.0),  # rp=0.1, rp*ds=10.0
        (151 + dx, 335,  0.0,  5.0),  # rp=1,   rp*ds=5.0
        (238 + dx, 380,  1.0,  2.5),  # rp=10,  rp*ds=2.5
    ]


def _calib_fig4_top(panel_idx):
    """3 calibration points for fig4 top-row panel i."""
    dx = _FIG4_COL_LEFT[panel_idx] - _FIG4_COL_LEFT[0]
    return [
        (64 + dx,   12, -1.0, 200.0),  # rp=0.1, rp*wp=200
        (129 + dx, 106,  0.0, 100.0),  # rp=1,   rp*wp=100
        (194 + dx, 151,  1.0,  50.0),  # rp=10,  rp*wp=50
    ]


def _calib_fig4_bot(panel_idx):
    """3 calibration points for fig4 bottom-row panel i."""
    dx = _FIG4_COL_LEFT[panel_idx] - _FIG4_COL_LEFT[0]
    return [
        (64 + dx,  263, -1.0, 10.0),
        (129 + dx, 341,  0.0,  5.0),
        (194 + dx, 379,  1.0,  2.5),
    ]


# Per-sample marker colours (each tracer uses a distinct colour)
_BGS2_COLOR  = (140, 200,  40)   # lime green
_BGS3_COLOR  = (240, 200,   0)   # golden yellow
_LRG1_COLOR  = (240, 160,   0)   # orange
_LRG2_COLOR  = (240,   0,   0)   # red

PANEL_CONFIGS = [
    # ---- fig3 top row: wp ----
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_COL_LEFT[0], col_right=_FIG3_COL_RIGHT[0],
        row_top=5, row_bot=155, calib=_calib_fig3_top(0),
        x_bins=_RP_BINS, data_color_rgb=_BGS2_COLOR,
        output_csv="BGS2/wp_bgs2.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_COL_LEFT[1], col_right=_FIG3_COL_RIGHT[1],
        row_top=5, row_bot=155, calib=_calib_fig3_top(1),
        x_bins=_RP_BINS, data_color_rgb=_BGS3_COLOR,
        output_csv="BGS3/wp_bgs3.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_COL_LEFT[2], col_right=_FIG3_COL_RIGHT[2],
        row_top=5, row_bot=155, calib=_calib_fig3_top(2),
        x_bins=_RP_BINS, data_color_rgb=_LRG1_COLOR,
        output_csv="LRG1/wp_lrg1.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
    ),
    # ---- fig3 bottom row: DES ESD ----
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_COL_LEFT[0], col_right=_FIG3_COL_RIGHT[0],
        row_top=233, row_bot=383, calib=_calib_fig3_bot(0),
        x_bins=_R_BINS, data_color_rgb=_BGS2_COLOR,
        output_csv="BGS2/ds_des_bgs2.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_COL_LEFT[1], col_right=_FIG3_COL_RIGHT[1],
        row_top=233, row_bot=383, calib=_calib_fig3_bot(1),
        x_bins=_R_BINS, data_color_rgb=_BGS3_COLOR,
        output_csv="BGS3/ds_des_bgs3.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
    ),
    PanelConfig(
        fig_path=_FIG3, col_left=_FIG3_COL_LEFT[2], col_right=_FIG3_COL_RIGHT[2],
        row_top=233, row_bot=383, calib=_calib_fig3_bot(2),
        x_bins=_R_BINS, data_color_rgb=_LRG1_COLOR,
        output_csv="LRG1/ds_des_lrg1.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
    ),
    # ---- fig4 top row: wp (use only LRG2 since BGS2/3 and LRG1 are in fig3) ----
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_COL_LEFT[3], col_right=_FIG4_COL_RIGHT[3],
        row_top=6, row_bot=155, calib=_calib_fig4_top(3),
        x_bins=_RP_BINS, data_color_rgb=_LRG2_COLOR,
        output_csv="LRG2/wp_lrg2.csv",
        col_x="rp_hMpc", col_y="wp_hMpc", col_yerr="wp_err_hMpc",
    ),
    # ---- fig4 bottom row: HSC ESD ----
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_COL_LEFT[0], col_right=_FIG4_COL_RIGHT[0],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(0),
        x_bins=_R_BINS, data_color_rgb=_BGS2_COLOR,
        output_csv="BGS2/ds_hsc_bgs2.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
    ),
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_COL_LEFT[1], col_right=_FIG4_COL_RIGHT[1],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(1),
        x_bins=_R_BINS, data_color_rgb=_BGS3_COLOR,
        output_csv="BGS3/ds_hsc_bgs3.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
    ),
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_COL_LEFT[2], col_right=_FIG4_COL_RIGHT[2],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(2),
        x_bins=_R_BINS, data_color_rgb=_LRG1_COLOR,
        output_csv="LRG1/ds_hsc_lrg1.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
    ),
    PanelConfig(
        fig_path=_FIG4, col_left=_FIG4_COL_LEFT[3], col_right=_FIG4_COL_RIGHT[3],
        row_top=233, row_bot=383, calib=_calib_fig4_bot(3),
        x_bins=_R_BINS, data_color_rgb=_LRG2_COLOR,
        output_csv="LRG2/ds_hsc_lrg2.csv",
        col_x="R_hMpc", col_y="ds_Msun_h_pc2", col_yerr="ds_err_Msun_h_pc2",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Extract and print but do not write CSV files.")
    ap.add_argument("--plot", action="store_true",
                    help="Show a log-log plot for each extracted dataset.")
    ap.add_argument("--inspect", action="store_true",
                    help="Print dominant data colors in each panel and exit.")
    ap.add_argument("--panel", type=int, default=None,
                    help="Process only panel INDEX (0-based) instead of all.")
    args = ap.parse_args()

    # Cache loaded images
    loaded_imgs = {}

    panels = PANEL_CONFIGS if args.panel is None else [PANEL_CONFIGS[args.panel]]

    for i, panel in enumerate(panels):
        if panel.fig_path not in loaded_imgs:
            loaded_imgs[panel.fig_path] = np.array(Image.open(panel.fig_path))
        img_arr = loaded_imgs[panel.fig_path]

        if args.inspect:
            colors = find_data_color(
                img_arr, panel.col_left, panel.col_right,
                panel.row_top, panel.row_bot,
            )
            idx = PANEL_CONFIGS.index(panel) if args.panel is None else args.panel
            print(f"Panel {idx} ({panel.output_csv or 'unnamed'}):")
            for color, count in colors:
                print(f"  RGB{color}: {count} px")
            continue

        run_panel(panel, img_arr,
                  dry_run=args.dry_run,
                  verbose=True,
                  make_plot=args.plot)

    if not args.inspect:
        print("\nDone.")


if __name__ == "__main__":
    main()
