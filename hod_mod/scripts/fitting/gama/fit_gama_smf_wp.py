"""Explore and visualise the GAMA stellar mass function.

Loads stellar mass functions (SMF) from the Galaxy and Mass Assembly (GAMA)
survey across five redshift bins.  Overlays a stellar-to-halo mass relation
(SHMR) prediction from Moster et al. 2013.

The GAMA SMF is measured using the 1/Vmax estimator:

.. math::

    \\Phi(M_*) = \\frac{1}{\\Delta\\log M_*} \\sum_i \\frac{1}{V_{\\max,i}}

A full HOD-based SMF fit requires a conditional stellar mass function (CSMF)
predictor integrated over the halo mass function — this is a planned future
addition to the pipeline.  The current script demonstrates data loading and
comparison with the SHAM prediction.

Inputs
------
- /path/to/sum_stat/data/GAMA/gama_smf_z{zmin}_{zmax}.fits
  FITS binary table with columns: log10mstar, phi, phi_err,
  log10mstar_lo, log10mstar_hi.  Units: log10(M*/M_sun), Mpc^-3 dex^-1.

Outputs
-------
Matplotlib figure displayed interactively (or saved to ``--output`` path).

References
----------
Driver et al. 2011, MNRAS 413, 971 — GAMA survey
Baldry et al. 2012, MNRAS 421, 621 (arXiv:1111.5707) — GAMA SMF at z<0.06
Moster et al. 2013, ApJ 770, 57 (arXiv:1205.5807) — stellar-to-halo mass
"""

import argparse
import os
import glob

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

GAMA_ZRANGES = [
    (0.002, 0.060),
    (0.060, 0.100),
    (0.100, 0.200),
    (0.200, 0.300),
    (0.300, 0.500),
]


def sham_smf_prediction(z: float, log10m_arr: np.ndarray, hmf,
                        theta: dict, h: float) -> np.ndarray:
    """Predict differential SMF from Moster+2013 SHMR + HMF (numerical derivative).

    Returns phi(M*) in units h^3 Mpc^-3 dex^-1.
    """
    m_h_arr = jnp.array(10.0 ** log10m_arr)
    # HMF in h^3 Mpc^-3 d(log10 M)
    dndlogm = np.asarray(hmf.dndm(m_h_arr, z, theta)) * m_h_arr * np.log(10)

    # SHMR: M* = f(M_h)
    log10mstar = np.asarray(smhm_moster13(m_h_arr, z))

    # phi(M*) = dn/d(log10 M_h) * |d(log10 M_h)/d(log10 M*)|
    # Estimate derivative numerically
    dlgmh_dlgms = np.gradient(log10m_arr, log10mstar)
    phi = dndlogm * np.abs(dlgmh_dlgms)
    return phi


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sum-stat-dir", default=SUM_STAT_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gama_dir = os.path.join(args.sum_stat_dir, "GAMA")
    pk_lin   = LinearPowerSpectrum()
    theta    = pk_lin.default_cosmology()
    hmf      = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    h        = theta["h"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes_flat = axes.flatten()

    for idx, (zmin, zmax) in enumerate(GAMA_ZRANGES):
        ax    = axes_flat[idx]
        fname = f"gama_smf_z{zmin:.3f}_{zmax:.3f}.fits"
        path  = os.path.join(gama_dir, fname)

        if not os.path.isfile(path):
            ax.set_title(f"GAMA $z={zmin:.2f}$–${zmax:.2f}$ (file missing)")
            continue

        reader = SumStatReader.from_fits(path)
        data   = reader.smf(h=h)

        log10m = data["log10mstar"]
        phi    = data["phi"]
        phi_e  = data["phi_err"]
        mask   = phi > 0

        ax.errorbar(log10m[mask], np.log10(phi[mask]),
                    yerr=phi_e[mask] / (phi[mask] * np.log(10)),
                    fmt="o", ms=4, color="k", label="GAMA 1/Vmax")

        # Overlay SHAM SMF prediction
        log10m_fine = np.linspace(9.5, 12.5, 200)
        z_eff = 0.5 * (zmin + zmax)
        try:
            phi_pred = sham_smf_prediction(z_eff, log10m_fine, hmf, theta, h)
            good = np.isfinite(phi_pred) & (phi_pred > 0)
            ax.plot(log10m_fine[good], np.log10(phi_pred[good]),
                    color="C0", ls="--", label="Moster+2013 SHAM")
        except Exception:
            pass

        ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
        ax.set_ylabel(r"$\log_{10}\,\Phi\;[h^3\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        ax.set_title(f"GAMA  $z={zmin:.2f}$–${zmax:.2f}$")
        ax.legend(fontsize=8)
        ax.set_ylim(-6, 0)

    axes_flat[-1].axis("off")
    plt.suptitle("GAMA stellar mass function vs. Moster+2013 SHAM", y=1.01)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved → {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
