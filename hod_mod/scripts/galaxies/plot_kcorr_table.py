"""Plot the X-ray K-correction table shipped with hod_mod.

Generates docs/_images/fig_agn_ham_05_kcorr.png — a two-panel figure:
  Left:  fraction_observed(z, logNH) as a colour map in the (z, logNH) plane
  Right: fraction_observed vs logNH for several representative redshifts

Run with:
    python3 -m hod_mod.scripts.galaxies.plot_kcorr_table
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator

_TABLE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "data", "agn",
    "v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt",
)
_OUT = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "docs", "_images",
    "fig_agn_ham_05_kcorr.png",
)


def main() -> None:
    z_arr, nh_arr, frac_arr = np.loadtxt(_TABLE, unpack=True)

    itp = LinearNDInterpolator(np.column_stack([z_arr, nh_arr]), frac_arr)

    # Evaluation grids
    z_grid  = np.linspace(z_arr.min(), z_arr.max(), 200)
    nh_grid = np.linspace(nh_arr.min(), nh_arr.max(), 200)
    ZZ, NN  = np.meshgrid(z_grid, nh_grid)
    FF      = itp(ZZ, NN)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: 2D colour map ---
    ax = axes[0]
    vmin, vmax = 1e-3, 5e-1
    im = ax.pcolormesh(
        z_grid, nh_grid, np.clip(FF, vmin, vmax),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="plasma", shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(
        r"$f_{\rm obs} = L_X^{0.5{-}2\,\rm keV,\,obs}\,/\,L_X^{2{-}10\,\rm keV,\,RF}$",
        fontsize=10,
    )
    ax.set_xlabel("Redshift $z$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(N_H\,[\rm cm^{-2}])$", fontsize=12)
    ax.set_title("K-correction: 2–10 keV (RF) → 0.5–2 keV (obs)", fontsize=11)

    # Mark type boundaries
    ax.axhline(22, color="white", lw=1, ls="--", alpha=0.7)
    ax.axhline(24, color="white", lw=1, ls=":",  alpha=0.7)
    ax.text(0.02, 22.10, "Type-1 / Type-2 boundary", color="white",
            fontsize=7, transform=ax.get_yaxis_transform(), va="bottom")
    ax.text(0.02, 24.10, "Compton-thin / CT boundary", color="white",
            fontsize=7, transform=ax.get_yaxis_transform(), va="bottom")

    # --- Right: slices at fixed z ---
    ax = axes[1]
    z_slices = [0.01, 0.1, 0.5, 1.0, 2.0, 3.5]
    colors   = plt.cm.viridis(np.linspace(0.1, 0.9, len(z_slices)))
    nh_fine  = np.linspace(20, 26, 300)

    for z_s, col in zip(z_slices, colors):
        pts  = np.column_stack([np.full_like(nh_fine, z_s), nh_fine])
        frac = itp(pts)
        ax.semilogy(nh_fine, frac, color=col, lw=1.8, label=f"z = {z_s}")

    ax.axvline(22, color="grey", lw=1, ls="--", alpha=0.7)
    ax.axvline(24, color="grey", lw=1, ls=":",  alpha=0.7)
    ax.text(22.05, 5e-4, "logNH=22", color="grey", fontsize=7, rotation=90, va="bottom")
    ax.text(24.05, 5e-4, "logNH=24 (CT)", color="grey", fontsize=7, rotation=90, va="bottom")

    ax.set_xlabel(r"$\log_{10}(N_H\,[\rm cm^{-2}])$", fontsize=12)
    ax.set_ylabel(
        r"$f_{\rm obs} = L_X^{0.5{-}2\,\rm keV,\,obs}\,/\,L_X^{2{-}10\,\rm keV,\,RF}$",
        fontsize=10,
    )
    ax.set_title("K-correction slices at fixed redshift", fontsize=11)
    ax.set_ylim(3e-4, 1.0)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    ax.text(
        0.03, 0.97,
        r"XSPEC: TBabs(plcabs + pexrav + $f_{\rm scat}$PL), $\Gamma=1.9$, $f_{\rm scat}=0.02$" "\n"
        r"Solar abundances (Wilms+2000), $N_H^{\rm gal}=3\times10^{20}\,{\rm cm}^{-2}$" "\n"
        r"$f_{\rm obs}(z=0,\log N_H=20)=0.607$; CT floor $=0.0133$",
        transform=ax.transAxes, fontsize=7.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    fig.savefig(_OUT, dpi=150)
    plt.close(fig)
    print(f"Saved → {_OUT}")


if __name__ == "__main__":
    main()
