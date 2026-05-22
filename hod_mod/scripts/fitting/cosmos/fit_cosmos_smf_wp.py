"""Explore and visualise the COSMOS stellar mass function to z=3.

Loads stellar mass functions (SMF) from the COSMOS photometric redshift
catalogue across six redshift bins (z = 0.2–3.0).  Overlays a SHAM
prediction from Moster et al. 2013 evolved to each redshift bin.

COSMOS provides a unique window into galaxy assembly history at high redshift,
tracing the buildup of stellar mass from z~3 to the present.

A full HOD-based SMF fit requires a conditional stellar mass function (CSMF)
predictor — see :ref:`fitting` for the roadmap.

Inputs
------
- /path/to/sum_stat/data/COSMOS/cosmos_smf_z{zmin}_{zmax}.fits
  FITS binary table: log10mstar, phi, phi_err, log10mstar_lo, log10mstar_hi.
  Units: log10(M*/M_sun), Mpc^-3 dex^-1.

Outputs
-------
Matplotlib figure displayed interactively (or saved to ``--output`` path).

References
----------
Laigle et al. 2016, ApJS 224, 24 — COSMOS2015 catalogue (arXiv:1604.02350)
Ilbert et al. 2013, A&A 556, A55 — COSMOS SMF at z<4 (arXiv:1301.3318)
Moster et al. 2013, ApJ 770, 57 (arXiv:1205.5807) — SHMR evolution with z
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from hod_mod.data_io.sum_stat_reader import SumStatReader
from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.galaxies.sham import smhm_moster13

SUM_STAT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                 "..", "sum_stat", "data")
)

COSMOS_ZRANGES = [
    (0.2, 0.5),
    (0.5, 0.8),
    (0.8, 1.1),
    (1.1, 1.5),
    (1.5, 2.0),
    (2.0, 3.0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sum-stat-dir", default=SUM_STAT_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cosmos_dir = os.path.join(args.sum_stat_dir, "COSMOS")
    pk_lin     = LinearPowerSpectrum()
    theta      = pk_lin.default_cosmology()
    hmf        = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    h          = theta["h"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for idx, (zmin, zmax) in enumerate(COSMOS_ZRANGES):
        ax    = axes.flatten()[idx]
        fname = f"cosmos_smf_z{zmin:.1f}_{zmax:.1f}.fits"
        path  = os.path.join(cosmos_dir, fname)

        if not os.path.isfile(path):
            ax.set_title(f"COSMOS $z={zmin}$–${zmax}$ (file missing)")
            continue

        reader = SumStatReader.from_fits(path)
        data   = reader.smf(h=h)
        log10m = data["log10mstar"]
        phi    = data["phi"]
        phi_e  = data["phi_err"]
        mask   = phi > 0

        ax.errorbar(log10m[mask], np.log10(phi[mask]),
                    yerr=phi_e[mask] / (phi[mask] * np.log(10)),
                    fmt="o", ms=3, color="k", label="COSMOS 1/Vmax")

        # SHAM prediction at z_eff
        z_eff = 0.5 * (zmin + zmax)
        log10m_fine = np.linspace(9.0, 12.5, 150)
        m_h = jnp.array(10.0 ** log10m_fine)
        try:
            dndlogm = np.asarray(hmf.dndm(m_h, z_eff, theta)) * np.asarray(m_h) * np.log(10)
            log10ms = np.asarray(smhm_moster13(m_h, z_eff))
            dlgmh_dlgms = np.gradient(log10m_fine, log10ms)
            phi_pred = dndlogm * np.abs(dlgmh_dlgms)
            good = np.isfinite(phi_pred) & (phi_pred > 0)
            ax.plot(log10m_fine[good], np.log10(phi_pred[good]),
                    ls="--", color="C0", label=f"Moster+2013 $z={z_eff:.2f}$")
        except Exception:
            pass

        ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
        ax.set_ylabel(r"$\log_{10}\,\Phi\;[h^3\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        ax.set_title(f"COSMOS  $z={zmin:.1f}$–${zmax:.1f}$")
        ax.legend(fontsize=7)
        ax.set_ylim(-7, 0)

    plt.suptitle("COSMOS stellar mass function evolution (z=0.2–3)", y=1.01)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved → {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
