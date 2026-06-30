"""Build the distilled eROSITA DR1 TM0 (all-7-telescope) response artifact.

Reads the official combined TM0 response from the eROSITA-DE DR1 ``arfrmf`` page
(https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_arfrmf/), placed in
``~/data/erosita/instrument/``:

  * ARF  ``{variant}_tm0_arf_filter_2023-01-17.fits.gz``  (variant = survey | onaxis)
  * RMF  ``onaxis_tm0_rmf_2023-01-17.fits.gz``            (position-independent)

``survey`` is the vignetting-corrected, FoV/survey-averaged effective area
(~1230 cm^2 @ 1 keV) — the physically correct response for the all-sky GALxEVT
cross-correlation.  ``onaxis`` is the on-axis combined area (~2189 cm^2 @ 1 keV).

The TM0 RMF is a single MATRIX extension (single group per row); its rows sum to
1 across 0.5-2 keV (detector QE is carried by the ARF), so the distilled in-band
efficiency ``g(E) = sum_{PI in 0.5-2 keV} R(E, PI)`` is the pure redistribution
fraction.

Output: ``hod_mod/data/erosita/dr1_response_tm0_{variant}_0p5-2keV.npz``
(energ_lo, energ_hi, arf_comb [cm^2], g_inband, rmf_rowsum, band, provenance).
The survey ARF is also copied into ``hod_mod/data/erosita/`` for portability; the
RMF (~2 MB) is not bundled — re-point ``--instr-dir`` to rebuild.

Usage:
    python -m hod_mod.scripts.galaxies.build_erosita_response \
        --instr-dir ~/data/erosita/instrument --variant survey
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from astropy.io import fits

_HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "erosita"))
_DEFAULT_INSTR = os.path.expanduser("~/data/erosita/instrument")
_BAND = (0.5, 2.0)
_ARF_NAME = "{variant}_tm0_arf_filter_2023-01-17.fits.gz"
_RMF_NAME = "onaxis_tm0_rmf_2023-01-17.fits.gz"


def _read_arf(path):
    d = fits.open(path)["SPECRESP"].data
    return (d["ENERG_LO"].astype(float), d["ENERG_HI"].astype(float),
            d["SPECRESP"].astype(float))


def _g_inband(rmf_path):
    """In-band redistribution g(E)=sum_{PI in 0.5-2 keV} R(E,PI) and row sums."""
    h = fits.open(rmf_path)
    eb = h["EBOUNDS"].data
    ch_lo = eb["E_MIN"].astype(float); ch_hi = eb["E_MAX"].astype(float)
    n_ch = len(eb)
    m = h["MATRIX"]
    cmin = int(m.header.get("TLMIN4", 1))
    d = m.data
    elo = d["ENERG_LO"].astype(float); ehi = d["ENERG_HI"].astype(float)
    M = np.zeros((len(elo), n_ch))
    for k in range(len(elo)):
        fch = np.atleast_1d(d["F_CHAN"][k]); nch = np.atleast_1d(d["N_CHAN"][k])
        row = np.atleast_1d(d["MATRIX"][k]).astype(float)
        p = 0
        for f, n in zip(fch, nch):
            n = int(n)
            if n <= 0:
                continue
            f = int(f) - cmin
            M[k, f:f + n] += row[p:p + n]
            p += n
    in_band = (ch_lo >= _BAND[0]) & (ch_hi <= _BAND[1])
    return elo, ehi, M.sum(axis=1), M[:, in_band].sum(axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instr-dir", default=_DEFAULT_INSTR,
                    help="dir with {survey,onaxis}_tm0_arf/rmf .fits.gz")
    ap.add_argument("--variant", choices=("survey", "onaxis"), default="survey")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    arf_src = os.path.join(args.instr_dir, _ARF_NAME.format(variant=args.variant))
    rmf_src = os.path.join(args.instr_dir, _RMF_NAME)
    a_elo, a_ehi, arf = _read_arf(arf_src)
    r_elo, r_ehi, rowsum, g = _g_inband(rmf_src)
    assert np.allclose(a_elo, r_elo), "ARF and RMF energy grids differ"

    emid = 0.5 * (r_elo + r_ehi)
    band = (emid >= _BAND[0]) & (emid <= _BAND[1])
    print(f"TM0 {args.variant} ARF peak {arf.max():.0f} cm^2 at "
          f"{emid[arf.argmax()]:.2f} keV; A@1keV {arf[np.argmin(abs(emid-1.0))]:.0f}; "
          f"mean in-band {arf[band].mean():.0f} cm^2")
    print(f"RMF median rowsum (0.5-2 keV) {np.median(rowsum[band]):.3f} "
          f"(=1 -> QE in ARF)")

    os.makedirs(_DATA_DIR, exist_ok=True)
    shutil.copy(arf_src, os.path.join(_DATA_DIR, os.path.basename(arf_src)))
    out = args.out or os.path.join(
        _DATA_DIR, f"dr1_response_tm0_{args.variant}_0p5-2keV.npz")
    np.savez(out, energ_lo=r_elo, energ_hi=r_ehi, arf_comb=arf, g_inband=g,
             rmf_rowsum=rowsum, band=np.array(_BAND),
             provenance=np.array([f"eROSITA DR1 arfrmf TM0; ARF "
                                  f"{os.path.basename(arf_src)}; RMF "
                                  f"{_RMF_NAME}"]))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
