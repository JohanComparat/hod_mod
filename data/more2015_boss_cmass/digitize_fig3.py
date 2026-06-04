"""Digitize More+2015 arXiv:1407.1856 Figure 3: wp and ESD for 3 mass-threshold samples.

Figure 3 layout (596x596 RGBA PNG):
  Left column  (wp):  cols 44-288, y = rp*wp [(h^-1 Mpc)^2] LOG scale
  Right column (ESD): cols 356-503, y = R*DeltaSigma [(h^-1 Mpc)(M_sun/pc^2)] LOG scale
  Row 0: logM*>11.10 (crimson markers)
  Row 1: logM*>11.30 (blue markers)
  Row 2: logM*>11.40 (olive/yellow markers)

Panel boundaries (from image analysis):
  Row 0: rows 10-178
  Row 1: rows 195-363
  Row 2: rows 380-548

X-axis (log scale, same for all panels):
  wp:  col_left=44,  col_right=288, rp_min=0.1, rp_max=100 Mpc/h
  ESD: col_left=356, col_right=503, R_min=0.1,  R_max=100  Mpc/h

Y-axis calibration (log scale):
  wp  panels: ticks at rp*wp = 3000, 1000, 300, 100, 30, 10 (h^-1 Mpc)^2
              panel_row_top+11 -> 3000,  panel_row_top+141 -> 10
  ESD panels: ticks at R*DeltaSigma = 100, 30, 10, 3, 1, 0.3 (h^-1 Mpc)(M_sun/pc^2)
              panel_row_top+8 -> 100,  panel_row_top+168 -> 0.3

Unit conventions in stored CSV files:
  wp:  rp_hMpc, wp_hMpc, wp_err_hMpc
       wp [h^-1 Mpc] = (rp*wp) / rp
  ESD: R_hMpc, ds_Msun_h_pc2, ds_err_Msun_h_pc2
       DeltaSigma [M_sun h / pc^2] = R*DeltaSigma / R / h
       Paper unit (M_sun/pc^2) -> code unit (M_sun h/pc^2): DIVIDE by h.
       h_more = 0.704 (WMAP7, More+2015 cosmology)

Color masks for data circles:
  Row 0 (logM>11.10): crimson  r>150, g<50, b<80
  Row 1 (logM>11.30): blue     b>180, r<120, b>r+80
  Row 2 (logM>11.40): olive    r≈g>100, b<40

Usage
-----
  cd /home/comparat/software/hod_mod
  python data/more2015_boss_cmass/digitize_fig3.py
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_PATH = "data/more2015_boss_cmass/raw_figures/fig3.png"
OUT_DIRS = {
    "logM11_12":   "data/more2015_boss_cmass/logM11_12",
    "logM11p3_12": "data/more2015_boss_cmass/logM11p3_12",
    "logM11p4_12": "data/more2015_boss_cmass/logM11p4_12",
}

# h for unit conversion: M_sun/pc^2 -> M_sun h/pc^2 requires dividing by h
H_MORE = 0.704   # WMAP7, More+2015 Table 1

# ---- Panel layout ----
PANEL_ROWS = [(10, 178), (195, 363), (380, 548)]   # (row_top, row_bottom)
WP_COLS  = (44, 288)
ESD_COLS = (356, 503)

# ---- X-axis calibration (log, 0.1..100 Mpc/h, same for wp and ESD columns) ----
def col_to_r(col, c_left, c_right, r_min=0.1, r_max=100.0):
    return 10.0 ** (np.log10(r_min) + (col - c_left) / (c_right - c_left) * np.log10(r_max / r_min))

def r_to_col(r, c_left, c_right, r_min=0.1, r_max=100.0):
    return c_left + (np.log10(r) - np.log10(r_min)) / np.log10(r_max / r_min) * (c_right - c_left)

# ---- Y-axis calibration (log scale) ----
# wp panels: panel_row_top + 11 -> rp*wp = 3000, panel_row_top + 141 -> rp*wp = 10
WP_REL_TOP    = 11
WP_REL_BOT    = 141
WP_Y_TOP      = 3000.0
WP_Y_BOT      = 10.0

def row_to_rpwp(row, panel_row_top):
    rel = row - panel_row_top
    t = (rel - WP_REL_TOP) / (WP_REL_BOT - WP_REL_TOP)
    log_y = np.log10(WP_Y_TOP) + t * (np.log10(WP_Y_BOT) - np.log10(WP_Y_TOP))
    return 10.0 ** log_y

def rpwp_to_row(rpwp, panel_row_top):
    t = (np.log10(rpwp) - np.log10(WP_Y_TOP)) / (np.log10(WP_Y_BOT) - np.log10(WP_Y_TOP))
    return panel_row_top + WP_REL_TOP + t * (WP_REL_BOT - WP_REL_TOP)

# ESD panels: panel_row_top + 8 -> R*DS = 100, panel_row_top + 168 -> R*DS = 0.3
ESD_REL_TOP   = 8
ESD_REL_BOT   = 168
ESD_Y_TOP     = 100.0
ESD_Y_BOT     = 0.3

def row_to_rds(row, panel_row_top):
    rel = row - panel_row_top
    t = (rel - ESD_REL_TOP) / (ESD_REL_BOT - ESD_REL_TOP)
    log_y = np.log10(ESD_Y_TOP) + t * (np.log10(ESD_Y_BOT) - np.log10(ESD_Y_TOP))
    return 10.0 ** log_y

def rds_to_row(rds, panel_row_top):
    t = (np.log10(rds) - np.log10(ESD_Y_TOP)) / (np.log10(ESD_Y_BOT) - np.log10(ESD_Y_TOP))
    return panel_row_top + ESD_REL_TOP + t * (ESD_REL_BOT - ESD_REL_TOP)

# ---- Color masks ----
def build_masks(arr):
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    ri, gi, bi = r.astype(int), g.astype(int), b.astype(int)
    masks = [
        (r > 150) & (g < 50) & (b < 80) & (a > 180),               # Row 0 crimson
        (b > 160) & (r < 150) & (bi > ri + 50) & (a > 180),         # Row 1 blue (relaxed)
        (r > 100) & (g > 100) & (bi < ri - 40) & (abs(ri - gi) < 45) & (a > 180),  # Row 2 olive
    ]
    return masks


def extract_column_points(mask_2d, row_top, row_bot, col_left, col_right,
                           col_to_x, row_to_y, y_from_row,
                           min_height=5, x_distance=10, strip_half=4):
    """Extract (x, y, y_err) data points from a color mask in a panel region."""
    sub = mask_2d[row_top:row_bot, col_left:col_right]
    prof_x = sub.sum(axis=0)
    prof_x_sm = gaussian_filter1d(prof_x.astype(float), sigma=1.5)
    x_peaks, _ = find_peaks(prof_x_sm, height=min_height, distance=x_distance)

    points = []
    for xi_rel in x_peaks:
        col_abs = xi_rel + col_left
        x_val = col_to_x(col_abs)

        # Vertical centroid in ±strip_half pixel strip
        strip = sub[:, max(0, xi_rel - strip_half): xi_rel + strip_half + 1]
        row_prof = strip.sum(axis=1)
        ypks, _ = find_peaks(row_prof, height=2, distance=5)
        if len(ypks) == 0:
            ypks = [np.argmax(row_prof)]
        row_rel = ypks[np.argmax(row_prof[ypks])]
        row_abs = row_rel + row_top
        y_center = row_to_y(row_abs)

        # Error: vertical extent of colored pixels
        rows_on = np.where(row_prof > 0)[0]
        if len(rows_on) >= 2:
            y_top = y_from_row(rows_on[0]  + row_top)
            y_bot = y_from_row(rows_on[-1] + row_top)
            y_err = abs(y_top - y_bot) / 2.0
        else:
            y_err = abs(y_center) * 0.15

        points.append((x_val, y_center, y_err))
    return points


def make_overlay_figure(img_arr, all_wp, all_esd, out_path):
    """Overlay extracted data points on the original figure in pixel coordinates."""
    fig, ax = plt.subplots(figsize=(img_arr.shape[1]/72, img_arr.shape[0]/72), dpi=150)
    ax.imshow(img_arr, origin="upper", aspect="auto")

    colors = ["crimson", "dodgerblue", "olive"]
    mass_labels = [r"$\log M_*>11.1$", r"$\log M_*>11.3$", r"$\log M_*>11.4$"]

    for pi, ((wp_pts, esd_pts), r_top) in enumerate(zip(zip(all_wp, all_esd), [p[0] for p in PANEL_ROWS])):
        col = colors[pi]

        # WP points
        for (rp, wp, wp_err) in wp_pts:
            rpwp = rp * wp
            col_px = r_to_col(rp, WP_COLS[0], WP_COLS[1])
            row_cen = rpwp_to_row(rpwp, r_top)
            rpwp_hi = rpwp + rp * wp_err
            rpwp_lo = max(rpwp - rp * wp_err, 1.0)
            row_hi  = rpwp_to_row(rpwp_hi, r_top)
            row_lo  = rpwp_to_row(rpwp_lo, r_top)
            ax.plot(col_px, row_cen, "o", ms=4, color=col, mec="k", mew=0.3, zorder=10)
            ax.plot([col_px, col_px], [row_hi, row_lo], "-", lw=0.8, color=col, zorder=9)

        # ESD points
        for (R, ds, ds_err) in esd_pts:
            rds = R * ds * H_MORE   # back to paper units: M_sun/pc^2 -> multiply by H_MORE
            col_px = r_to_col(R, ESD_COLS[0], ESD_COLS[1])
            row_cen = rds_to_row(rds, r_top)
            rds_hi  = max(rds + R * ds_err * H_MORE, 0.31)
            rds_lo  = max(rds - R * ds_err * H_MORE, 0.31)
            row_hi  = rds_to_row(rds_hi, r_top)
            row_lo  = rds_to_row(rds_lo, r_top)
            ax.plot(col_px, row_cen, "s", ms=4, color=col, mec="k", mew=0.3, zorder=10)
            ax.plot([col_px, col_px], [row_hi, row_lo], "-", lw=0.8, color=col, zorder=9)

    ax.set_xlim(0, img_arr.shape[1])
    ax.set_ylim(img_arr.shape[0], 0)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Overlay -> {out_path}")


def make_standalone_figure(all_wp, all_esd, out_path):
    """Recreate paper-style figure with extracted data in data coordinates."""
    colors = ["crimson", "dodgerblue", "olive"]
    mass_labels = [r"$\log M_*>11.1$", r"$\log M_*>11.3$", r"$\log M_*>11.4$"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    for pi, ((wp_pts, esd_pts), col, lbl) in enumerate(zip(zip(all_wp, all_esd), colors, mass_labels)):
        ax_wp  = axes[0, pi]
        ax_esd = axes[1, pi]

        if wp_pts:
            rp_arr    = np.array([p[0] for p in wp_pts])
            wp_arr    = np.array([p[1] for p in wp_pts])
            wp_err    = np.array([p[2] for p in wp_pts])
            ax_wp.errorbar(rp_arr, rp_arr * wp_arr, rp_arr * wp_err,
                           fmt="o", color=col, capsize=3, label=lbl)
            ax_wp.set_xscale("log"); ax_wp.set_yscale("log")
            ax_wp.set_xlim(0.08, 120); ax_wp.set_ylim(8, 4000)
            ax_wp.set_xlabel(r"$r_p$ [h$^{-1}$ Mpc]")
            ax_wp.set_ylabel(r"$r_p w_p$ [(h$^{-1}$ Mpc)$^2$]")
            ax_wp.set_title(lbl); ax_wp.legend(fontsize=7)

        if esd_pts:
            R_arr  = np.array([p[0] for p in esd_pts])
            ds_arr = np.array([p[1] for p in esd_pts])   # M_sun h / pc^2
            de_arr = np.array([p[2] for p in esd_pts])
            # plot in paper units (M_sun/pc^2) so it looks like the paper
            ds_paper = ds_arr * H_MORE
            de_paper = de_arr * H_MORE
            ax_esd.errorbar(R_arr, R_arr * ds_paper, R_arr * de_paper,
                            fmt="s", color=col, capsize=3)
            ax_esd.set_xscale("log"); ax_esd.set_yscale("log")
            ax_esd.set_xlim(0.08, 120); ax_esd.set_ylim(0.25, 120)
            ax_esd.set_xlabel(r"$R$ [h$^{-1}$ Mpc]")
            ax_esd.set_ylabel(r"$R\,\Delta\Sigma$ [(h$^{-1}$ Mpc)(M$_\odot$ pc$^{-2}$)]")
            ax_esd.set_title(lbl)

    plt.suptitle("More+2015 Fig 3 — digitized (ESD in paper M$_\\odot$ pc$^{-2}$ units)", fontsize=11)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Standalone -> {out_path}")


def main():
    img = Image.open(IMG_PATH).convert("RGBA")
    arr = np.array(img)
    masks = build_masks(arr)

    mass_labels = ["logM11_12",   "logM11p3_12", "logM11p4_12"]
    mass_mins   = [11.10,          11.30,          11.40]
    colors_plot = ["crimson",      "dodgerblue",   "olive"]

    all_wp_pts  = []
    all_esd_pts = []

    for p_idx, (mass_label, log_mstar_min) in enumerate(zip(mass_labels, mass_mins)):
        mask = masks[p_idx]
        r_top, r_bot = PANEL_ROWS[p_idx]

        # ---- WP extraction ----
        wp_pts = extract_column_points(
            mask, r_top, r_bot, WP_COLS[0], WP_COLS[1],
            col_to_x=lambda c: col_to_r(c, WP_COLS[0], WP_COLS[1]),
            row_to_y=lambda row: row_to_rpwp(row, r_top) / col_to_r(
                int(r_to_col(col_to_r(WP_COLS[0], WP_COLS[0], WP_COLS[1]), WP_COLS[0], WP_COLS[1])),
                WP_COLS[0], WP_COLS[1]),   # placeholder; we compute wp below
            y_from_row=lambda row: row_to_rpwp(row, r_top),
            min_height=5, x_distance=10
        )
        # Recompute properly: extract (rp, rpwp_center, rpwp_err) then divide by rp
        sub_wp = mask[r_top:r_bot, WP_COLS[0]:WP_COLS[1]]
        prof_x = sub_wp.sum(axis=0)
        prof_x_sm = gaussian_filter1d(prof_x.astype(float), sigma=1.5)
        px_rel, _ = find_peaks(prof_x_sm, height=5, distance=10)

        wp_result = []
        for xi_rel in px_rel:
            col_abs = xi_rel + WP_COLS[0]
            rp = col_to_r(col_abs, WP_COLS[0], WP_COLS[1])
            strip = sub_wp[:, max(0, xi_rel - 4): xi_rel + 5]
            row_prof = strip.sum(axis=1)
            ypks, _ = find_peaks(row_prof, height=2, distance=5)
            if len(ypks) == 0:
                ypks = [np.argmax(row_prof)]
            row_rel  = ypks[np.argmax(row_prof[ypks])]
            row_abs  = row_rel + r_top
            rpwp     = row_to_rpwp(row_abs, r_top)
            wp       = rpwp / rp
            rows_on  = np.where(row_prof > 0)[0]
            if len(rows_on) >= 2:
                rpwp_hi = row_to_rpwp(rows_on[0]  + r_top, r_top)
                rpwp_lo = row_to_rpwp(rows_on[-1] + r_top, r_top)
                wp_err  = abs(rpwp_hi - rpwp_lo) / 2.0 / rp
            else:
                wp_err = wp * 0.15
            wp_result.append((rp, wp, wp_err))
        all_wp_pts.append(wp_result)

        # ---- ESD extraction ----
        sub_esd = mask[r_top:r_bot, ESD_COLS[0]:ESD_COLS[1]]
        prof_ex = sub_esd.sum(axis=0)
        prof_ex_sm = gaussian_filter1d(prof_ex.astype(float), sigma=1.5)
        ex_rel, _ = find_peaks(prof_ex_sm, height=3, distance=8)

        esd_result = []
        for xi_rel in ex_rel:
            col_abs = xi_rel + ESD_COLS[0]
            R = col_to_r(col_abs, ESD_COLS[0], ESD_COLS[1])
            strip = sub_esd[:, max(0, xi_rel - 4): xi_rel + 5]
            row_prof = strip.sum(axis=1)
            ypks, _ = find_peaks(row_prof, height=2, distance=5)
            if len(ypks) == 0:
                ypks = [np.argmax(row_prof)]
            row_rel  = ypks[np.argmax(row_prof[ypks])]
            row_abs  = row_rel + r_top
            RdS      = row_to_rds(row_abs, r_top)     # (h^-1 Mpc)(M_sun/pc^2)
            ds_paper = RdS / R if R > 0 else 0.0       # M_sun/pc^2
            ds_code  = ds_paper / H_MORE               # M_sun h / pc^2

            rows_on  = np.where(row_prof > 0)[0]
            if len(rows_on) >= 2:
                RdS_hi   = row_to_rds(rows_on[0]  + r_top, r_top)
                RdS_lo   = row_to_rds(rows_on[-1] + r_top, r_top)
                ds_err   = abs(RdS_hi - RdS_lo) / 2.0 / R / H_MORE
            else:
                ds_err = ds_code * 0.15
            esd_result.append((R, ds_code, ds_err))
        all_esd_pts.append(esd_result)

        # ---- Save CSVs ----
        out_dir = OUT_DIRS[mass_label]
        os.makedirs(out_dir, exist_ok=True)

        rp_arr     = np.array([p[0] for p in wp_result])
        wp_arr     = np.array([p[1] for p in wp_result])
        wp_err_arr = np.array([p[2] for p in wp_result])
        wp_out = os.path.join(out_dir, f"wp_{mass_label}.csv")
        header_wp = (
            f"More+2015 Fig3 panel {p_idx}: logM*>{log_mstar_min} mass-threshold sample\n"
            f"Digitized from fig3.png ({['crimson','blue','olive'][p_idx]} markers)\n"
            f"wp y-axis: log scale {WP_Y_TOP}-{WP_Y_BOT} (h^-1 Mpc)^2, "
            f"ticks at rel rows {WP_REL_TOP}..{WP_REL_BOT}\n"
            f"rp_hMpc,wp_hMpc,wp_err_hMpc"
        )
        np.savetxt(wp_out, np.column_stack([rp_arr, wp_arr, wp_err_arr]),
                   header=header_wp, delimiter=",", fmt="%.6f", comments="")
        print(f"Wrote {wp_out} ({len(rp_arr)} wp bins)")

        if esd_result:
            R_arr      = np.array([p[0] for p in esd_result])
            ds_arr     = np.array([p[1] for p in esd_result])
            ds_err_arr = np.array([p[2] for p in esd_result])
            ds_out = os.path.join(out_dir, f"ds_{mass_label}.csv")
            header_ds = (
                f"More+2015 Fig3 panel {p_idx}: logM*>{log_mstar_min} ESD\n"
                f"Digitized from fig3.png. Unit conversion: paper M_sun/pc^2 / h={H_MORE} -> M_sun h/pc^2\n"
                f"ESD y-axis: log scale {ESD_Y_TOP}-{ESD_Y_BOT} (h^-1 Mpc)(M_sun/pc^2), "
                f"ticks at rel rows {ESD_REL_TOP}..{ESD_REL_BOT}\n"
                f"R_hMpc,ds_Msun_h_pc2,ds_err_Msun_h_pc2"
            )
            np.savetxt(ds_out, np.column_stack([R_arr, ds_arr, ds_err_arr]),
                       header=header_ds, delimiter=",", fmt="%.6f", comments="")
            print(f"Wrote {ds_out} ({len(R_arr)} ESD bins)")

    # ---- Overlay figure on original image ----
    ov_path = "data/more2015_boss_cmass/raw_figures/fig3_overlay.png"
    make_overlay_figure(arr, all_wp_pts, all_esd_pts, ov_path)

    # ---- Standalone extracted-data figure ----
    standalone_path = "data/more2015_boss_cmass/raw_figures/fig3_extracted.pdf"
    make_standalone_figure(all_wp_pts, all_esd_pts, standalone_path)


if __name__ == "__main__":
    main()
