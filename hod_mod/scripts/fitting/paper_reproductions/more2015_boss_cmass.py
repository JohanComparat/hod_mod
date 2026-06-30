"""Reproduce the More+2015 BOSS CMASS HOD fit.

Fits the More+2015 HOD model to the BOSS CMASS projected correlation
function :math:`w_p(r_p)` at :math:`z_\\mathrm{eff} = 0.52`, following the
methodology of More et al. 2015.

The MoreHODModel adds an incompleteness correction to the standard Zheng+2007
HOD.  The central occupation becomes:

.. math::

    \\langle N_\\mathrm{cen}(M) \\rangle = \\frac{\\alpha_\\mathrm{inc}}{2}\\,
    \\mathrm{erfc}\\!\\left[\\frac{\\log_{10}M_\\mathrm{min} - \\log_{10}M}
    {\\sqrt{2}\\,\\sigma_{\\log m}}\\right]

where :math:`\\alpha_\\mathrm{inc} \\leq 1` accounts for the survey incompleteness.

Inputs
------
- data/more2015_boss_cmass/wp_cmass_z052.csv
    Three-column CSV: rp_hMpc, wp_hMpc, wp_err_hMpc.
    Source: More et al. 2015, Table 1.
- configs/more2015_boss_cmass.yml

Outputs
-------
- results/more2015_cmass/map_result.json      (MAP best-fit parameters, χ²)
- results/more2015_cmass/flatchain.npz         (emcee posterior samples)
- results/more2015_cmass/wp_bestfit.pdf        (comparison figure, with --plot)

References
----------
More et al. 2015, ApJ 806, 2 (arXiv:1407.1856)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from hod_mod.fitting import load_config, WpFitter


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG    = os.path.join(REPO_ROOT, "configs", "more2015_boss_cmass.yml")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=CONFIG,
                        help="Path to YAML config (default: configs/more2015_boss_cmass.yml)")
    parser.add_argument("--map-only", action="store_true",
                        help="Run MAP estimation only, skip MCMC.")
    parser.add_argument("--mcmc-only", action="store_true",
                        help="Run MCMC only, skip MAP.")
    parser.add_argument("--plot", action="store_true",
                        help="Save a comparison plot after MAP fit.")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    fitter  = WpFitter(cfg)

    if not args.mcmc_only:
        result = fitter.map_fit()
        print("\n=== MAP result ===")
        for name, val in zip(cfg.free_params, result["theta"]):
            print(f"  {name:20s} = {val:.4f}")
        print(f"  chi2/dof = {result['chi2']:.2f} / {result['ndof']}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        out_json = os.path.join(cfg.output_dir, "map_result.json")
        with open(out_json, "w") as fh:
            json.dump({
                "params": result["params"],
                "chi2":   result["chi2"],
                "ndof":   result["ndof"],
                "success": result["success"],
            }, fh, indent=2)
        print(f"MAP result saved → {out_json}")

        if args.plot:
            wp_pred = fitter.predict_wp(result["params"])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(fitter.rp_arr, fitter.wp_obs, fitter.wp_err,
                        fmt="o", color="k", label="BOSS CMASS")
            ax.loglog(fitter.rp_arr, wp_pred, color="C0", label="MAP fit")
            ax.set_xlabel(r"$r_p$ [Mpc/$h$]")
            ax.set_ylabel(r"$w_p$ [Mpc/$h$]")
            ax.legend()
            ax.set_title(f"More+2015 BOSS CMASS — χ²/dof = {result['chi2']:.1f}/{result['ndof']}")
            out_fig = os.path.join(cfg.output_dir, "wp_bestfit.pdf")
            plt.tight_layout()
            plt.savefig(out_fig)
            print(f"Figure saved → {out_fig}")

    if not args.map_only and cfg.method in ("emcee", "both"):
        sampler = fitter.sample(progress=True)
        flat = sampler.get_chain(flat=True)
        acc  = np.mean(sampler.acceptance_fraction)
        print(f"\nMCMC acceptance fraction: {acc:.3f}")
        print(f"Chain shape: {flat.shape}")


if __name__ == "__main__":
    main()
