"""Tabulate the true per-component eROSITA ECF (0.5-2 keV band averages).

For each GALxEVT sample, precompute:
  * ECF_gas(T) over a temperature grid (APEC, Z=0.3 Zsun) at the sample z_mean,
  * ECF_AGN  (absorbed power law Gamma=1.9) at the same z,
folded through the TM0 **survey** ARF + RMF (``ErositaResponse``), N_H=3e20.

Also stores ``ecf_fixed`` — the GALxEVT pipeline's fixed conversion
``ARF_1keV / C = 2200 cm^2 / 1.602177e-9 erg = 1.373e12 cts/s per erg/s/cm^2``
(on-axis 1 keV/photon, no spectral model), used only to undo the displayed
``S^R_X`` pseudo-energy normalisation when going to an absolute footing.

Output: ``hod_mod/data/erosita/ecf_tables_<sample>.npz``
(kT_grid, ecf_gas, ecf_agn, ecf_fixed, z, nH, Z).

Usage:
    python -m hod_mod.scripts.galaxies.build_ecf_tables            # all samples
    python -m hod_mod.scripts.galaxies.build_ecf_tables --sample S1
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from hod_mod.gas import ErositaResponse
from hod_mod.scripts.fitting import fit_comparat2025 as F

_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "erosita"))
_ARF_1KEV = 2200.0                 # cm^2, GALxEVT pipeline on-axis value
_ERG_PER_CT = 1.602177e-9          # erg, 1 keV/photon (no spectral model)
_ECF_FIXED = _ARF_1KEV / _ERG_PER_CT   # cts/s per erg/s/cm^2 = 1.373e12
_NH = 0.03                         # 1e22 cm^-2
_Z_METAL = 0.3                     # Zsun


def build(sample, resp=None, kT_grid=None):
    resp = resp or ErositaResponse()
    z = float(F.SAMPLES[sample]["zmean"])
    kT_grid, ecf_gas, _ = resp.ecf_apec_table(z=z, nH=_NH, Z=_Z_METAL,
                                              kT_grid=kT_grid)
    ecf_agn = resp.ecf_powerlaw(1.9, z=z, nH=_NH)
    out = os.path.join(_DATA_DIR, f"ecf_tables_{sample}.npz")
    np.savez(out, kT_grid=kT_grid, ecf_gas=ecf_gas, ecf_agn=ecf_agn,
             ecf_fixed=_ECF_FIXED, z=z, nH=_NH, Z=_Z_METAL,
             arf_1keV=_ARF_1KEV, erg_per_ct=_ERG_PER_CT)
    print(f"{sample} z={z:.3f}: ECF_AGN={ecf_agn:.3e}  "
          f"ECF_gas(1keV)={np.interp(0.0, np.log10(kT_grid), ecf_gas):.3e}  "
          f"ecf_fixed={_ECF_FIXED:.3e}  -> {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default=None, help="one sample (default: all)")
    args = ap.parse_args(argv)
    resp = ErositaResponse()
    kT_grid = np.logspace(np.log10(0.1), np.log10(15.0), 24)
    samples = [args.sample] if args.sample else list(F.SAMPLES.keys())
    for s in samples:
        build(s, resp=resp, kT_grid=kT_grid)


if __name__ == "__main__":
    main()
