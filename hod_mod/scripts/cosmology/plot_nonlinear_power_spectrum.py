"""Generate fig01b_nonlinear_power_spectrum.png for the forward-model showcase.

Compares all available non-linear matter power spectrum backends in hod_mod:

* :class:`~hod_mod.core.nonlinear.NonLinearPowerSpectrum` (Aletheia,
  arXiv:2511.13826) — ``backend='aletheia'``
* :class:`~hod_mod.core.nonlinear.HALOFITSpectrum` with
  ``halofit_version='mead2020'`` (CAMB HMcode-2020, arXiv:2009.01858)
* :class:`~hod_mod.core.nonlinear.HALOFITSpectrum` with
  ``halofit_version='takahashi'`` (CAMB Takahashi+2012)
* :class:`~hod_mod.core.nonlinear.WHMSpectrum` (Web-Halo Model,
  Brieden et al. arXiv:2508.10902) — requires WHM-CAMB fork

Output: docs/_images/fig01b_nonlinear_power_spectrum.png

Usage::

    cd /home/comparat/software/hod_mod
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.cosmology.plot_nonlinear_power_spectrum
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.nonlinear import NonLinearPowerSpectrum, HALOFITSpectrum, WHMSpectrum

_HERE    = Path(__file__).parent
_IMG_DIR = _HERE.parents[2] / "docs" / "_images"
_IMG_DIR.mkdir(parents=True, exist_ok=True)

_THETA = LinearPowerSpectrum.default_cosmology()
_Z     = 0.14
_K     = np.logspace(-2, 1, 300)  # h/Mpc

# Aletheia valid range: k ∈ [0.006, 2.0] Mpc⁻¹  (arXiv:2511.13826)
# Convert to h/Mpc: k_h = k_mpc / h
_H = float(LinearPowerSpectrum.default_cosmology()["h"])
_ALETHEIA_K_MASK = (_K >= 0.006 / _H) & (_K <= 2.0 / _H)


def _compute_pk_lin():
    pklin = LinearPowerSpectrum()
    return np.asarray(pklin.pk_linear(_K, _Z, _THETA))


def _compute_backends(pk_lin):
    results = {}

    # --- Aletheia emulator (clipped to valid k range, no extrapolation) ---
    try:
        emu  = NonLinearPowerSpectrum(backend="aletheia")
        pk   = np.asarray(emu.pk_nonlinear(_K, _Z, _THETA))
        pk_masked = np.where(_ALETHEIA_K_MASK, pk, np.nan)
        results["Aletheia (arXiv:2511.13826)"] = {
            "pk": pk_masked, "color": "C0", "ls": "-",
            "class": "NonLinearPowerSpectrum(backend='aletheia')",
            "path":  "hod_mod.core.nonlinear.NonLinearPowerSpectrum",
        }
    except Exception as e:
        print(f"[skip] Aletheia: {e}")

    # --- CAMB HMcode-2020 ---
    try:
        hm = HALOFITSpectrum(halofit_version="mead2020")
        pk = np.asarray(hm.pk_nonlinear(_K, _Z, _THETA))
        results["CAMB HMcode-2020 (arXiv:2009.01858)"] = {
            "pk": pk, "color": "C2", "ls": "-",
            "class": "HALOFITSpectrum(halofit_version='mead2020')",
            "path":  "hod_mod.core.nonlinear.HALOFITSpectrum",
        }
    except Exception as e:
        print(f"[skip] HMcode-2020: {e}")

    # --- CAMB Takahashi+2012 ---
    try:
        tak = HALOFITSpectrum(halofit_version="takahashi")
        pk  = np.asarray(tak.pk_nonlinear(_K, _Z, _THETA))
        results["CAMB Takahashi+2012 (arXiv:1208.2701)"] = {
            "pk": pk, "color": "C3", "ls": "--",
            "class": "HALOFITSpectrum(halofit_version='takahashi')",
            "path":  "hod_mod.core.nonlinear.HALOFITSpectrum",
        }
    except Exception as e:
        print(f"[skip] Takahashi: {e}")

    # --- WHM (requires WHM-CAMB fork) ---
    try:
        whm = WHMSpectrum(whm_version="brieden2023")
        pk  = np.asarray(whm.pk_nonlinear(_K, _Z, _THETA))
        results["WHM brieden2023 (arXiv:2508.10902)"] = {
            "pk": pk, "color": "C1", "ls": "-.",
            "class": "WHMSpectrum(whm_version='brieden2023')",
            "path":  "hod_mod.core.nonlinear.WHMSpectrum",
        }
        print("[ok]  WHM brieden2023")
    except RuntimeError as e:
        print(f"[skip] WHM (WHM-CAMB not installed): {e}")
    except Exception as e:
        print(f"[skip] WHM: {e}")

    return results


def make_figure():
    pk_lin = _compute_pk_lin()
    backends = _compute_backends(pk_lin)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    # — top panel: P(k) —
    ax1.loglog(_K, pk_lin, color="0.55", lw=1.5, label=r"$P_{\rm lin}$")
    for label, d in backends.items():
        ax1.loglog(_K, d["pk"], color=d["color"], ls=d["ls"], lw=2, label=label)

    ax1.set_ylabel(r"$P(k)\;[(h^{-1}\,\mathrm{Mpc})^3]$", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_title(rf"Non-linear matter power spectrum ($z = {_Z}$)", fontsize=12)
    ax1.grid(True, which="both", alpha=0.25)

    # — bottom panel: P_nl / P_lin ratio —
    _SHORT = {
        "Aletheia (arXiv:2511.13826)":          "Aletheia",
        "CAMB HMcode-2020 (arXiv:2009.01858)":  "HMcode-2020",
        "CAMB Takahashi+2012 (arXiv:1208.2701)": "Takahashi+2012",
        "WHM brieden2023 (arXiv:2508.10902)":   "WHM",
    }
    ax2.axhline(1.0, color="0.55", lw=1.0, ls="--")
    for label, d in backends.items():
        short = _SHORT.get(label, label.split(" ")[0])
        ax2.semilogx(_K, d["pk"] / pk_lin, color=d["color"], ls=d["ls"], lw=2,
                     label=short)

    ax2.set_ylabel(r"$P_{\rm nl}/P_{\rm lin}$", fontsize=12)
    ax2.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, which="both", alpha=0.25)

    # — attribution footnote —
    refs = (
        "Aletheia (arXiv:2511.13826); "
        "Mead+2020 HMcode (arXiv:2009.01858); "
        "Takahashi+2012 (arXiv:1208.2701); "
        "WHM Brieden+2025 (arXiv:2508.10902)"
    )
    fig.text(0.5, 0.01, refs, ha="center", va="bottom", fontsize=6, color="0.45")

    out = _IMG_DIR / "fig01b_nonlinear_power_spectrum.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # Print class table for documentation
    print("\nClass/function reference table:")
    print(f"{'Backend':<45} {'Class / path'}")
    print("-" * 90)
    for label, d in backends.items():
        print(f"{label:<45} {d['path']}.{d['class'].split('(')[0].split('.')[-1]}()")


if __name__ == "__main__":
    make_figure()
