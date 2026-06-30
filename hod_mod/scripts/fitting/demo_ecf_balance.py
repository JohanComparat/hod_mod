"""Show how folding the true eROSITA ECF re-weights gas vs AGN in the galaxy x
X-ray cross-correlation.

w(theta) is a count ratio, so for a *predicted* (absolute-L_X) model the gas-vs-AGN
split is set by the true per-component ECF: gas counts ~ n_e^2 Lambda ECF_gas(kT(M)),
AGN counts ~ L_X^AGN ECF_AGN(Gamma=1.9).  The AGN ECF is a single constant (fixed
spectrum); the gas ECF varies with halo temperature kT(M), so the re-weighting
ECF_gas(kT(M))/ECF_AGN depends on halo mass.  Computed directly from the tabulated
TM0-survey ECF (ErositaResponse) + the Lovisari+2020 kT-M500c relation.

Output: results/agn_duty_cycle/baseline/S1_ecf_balance.png + printed ratios.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.gas import load_ecf_tables
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts import validate_gas_profiles as vgp

_OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "results", "agn_duty_cycle", "baseline", "S1_ecf_balance.png"))


def main():
    sample = "S1"
    z = float(F.SAMPLES[sample]["zmean"])
    gas_of_T, ecf_agn, ecf_fixed = load_ecf_tables(sample)

    kT = np.logspace(np.log10(0.15), np.log10(10.0), 200)
    ecf_gas = np.array([float(gas_of_T(t)) for t in kT])

    # halo temperature kT(M) -> ECF_gas(M) (Lovisari kT-M500c)
    m200 = np.logspace(12.0, 15.0, 200) * vgp._H        # Msun/h
    r200 = vgp._r200(m200, z); c200 = vgp._c200_approx(m200)
    m500_h, _ = vgp.m200_to_m500c(m200, c200, r200, vgp._rho_crit_z(z))
    kT_M = vgp._lovisari20_kt(m500_h / vgp._H, z=z)
    ecf_gas_M = np.array([float(gas_of_T(t)) for t in kT_M])
    rew = ecf_gas_M / ecf_agn

    print(f"S1 z={z:.3f}:  ECF_AGN(Γ=1.9) = {ecf_agn:.3e} (constant)")
    print(f"  ECF_gas(kT): 0.5keV={float(gas_of_T(0.5)):.3e}  "
          f"1keV={float(gas_of_T(1.0)):.3e}  2keV={float(gas_of_T(2.0)):.3e}")
    for lm in (12.0, 13.0, 14.0, 15.0):
        i = np.argmin(np.abs(np.log10(m200 / vgp._H) - lm))
        print(f"  M200=1e{lm:.0f}: kT={kT_M[i]:.2f} keV  "
              f"ECF_gas/ECF_AGN = {rew[i]:.3f}  (gas {100*(rew[i]-1):+.0f}% vs AGN)")
    print(f"  ecf_fixed (Comparat pipeline) = {ecf_fixed:.3e}; the FIXED-ECF "
          f"measurement mis-weights gas/AGN by up to ~{100*(rew.max()-rew.min()):.0f}%.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    a.semilogx(kT, ecf_gas / 1e11, "C2-", lw=2, label=r"gas APEC, ECF$_{\rm gas}(kT)$")
    a.axhline(ecf_agn / 1e11, ls="--", color="C1", lw=2,
              label=r"AGN $\Gamma=1.9$, ECF$_{\rm AGN}$")
    a.set_xlabel(r"$kT$ [keV]")
    a.set_ylabel(r"ECF [$10^{11}$ cts/s per erg/s/cm$^2$], TM0 survey")
    a.set_title(f"{sample}: true per-component ECF (0.5-2 keV)"); a.legend(fontsize=9)
    a.grid(alpha=0.2)
    a = ax[1]
    a.semilogx(m200 / vgp._H, rew, "C3-", lw=2)
    a.axhline(1.0, ls="-", color="0.7", lw=0.8)
    a.set_xlabel(r"$M_{200}$ [$M_\odot$]")
    a.set_ylabel(r"gas/AGN re-weighting  ECF$_{\rm gas}(kT(M))$/ECF$_{\rm AGN}$")
    a.set_title("Mass-dependent gas vs AGN re-weighting"); a.grid(alpha=0.2)
    a.text(0.05, 0.1, "fixed-ECF pipeline = flat (no re-weighting)",
           transform=a.transAxes, fontsize=8, color="0.4")
    fig.tight_layout(); fig.savefig(_OUT, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved -> {_OUT}")


if __name__ == "__main__":
    main()
