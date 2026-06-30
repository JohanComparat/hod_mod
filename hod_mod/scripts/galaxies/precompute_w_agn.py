"""Precompute the Lau et al. 2025 W_AGN(z) kernel (Eq. A9) for a BGS sample.

Builds the X-ray-flux-weighted AGN redshift kernel and stores it (with all
integrand components, for verification) to an h5 file.  Skips the computation
if the output file already exists (unless ``--overwrite`` is given).

Run with:
    python -m hod_mod.scripts.galaxies.precompute_w_agn --sample S1
    python -m hod_mod.scripts.galaxies.precompute_w_agn --sample S1 --overwrite
"""

from __future__ import annotations

import argparse
import logging

from hod_mod.agn.duty_cycle import compute_w_agn_kernel, BGS_SAMPLES


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S1", choices=list(BGS_SAMPLES),
                    help="LS10-BGS sample label (default S1).")
    ap.add_argument("--n-z", type=int, default=40,
                    help="Number of redshift grid points (default 40).")
    ap.add_argument("--n-lx", type=int, default=500,
                    help="Number of hard-band luminosity grid points (default 500).")
    ap.add_argument("--xlf", default="aird15", choices=["aird15", "ueda14"],
                    help="X-ray luminosity function (default aird15 LADE).")
    ap.add_argument("--out-path", default=None,
                    help="Explicit output h5 path (default results/agn_duty_cycle/W_AGN_<sample>.h5).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if the output file exists.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    path = compute_w_agn_kernel(
        sample=args.sample, n_z=args.n_z, n_lx=args.n_lx, xlf=args.xlf,
        out_path=args.out_path, overwrite=args.overwrite,
    )
    print(f"W_AGN kernel for {args.sample}: {path}")


if __name__ == "__main__":
    main()
