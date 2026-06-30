"""Generate HMF and bias documentation figures for the forward-model showcase.

Produces three figures:

* ``fig02_hmf.png``
  :class:`~hod_mod.core.halo_mass_function.HaloMassFunction` ``dn/dM`` and
  ``b(M)`` at z=0.14 with ±3σ S8 variation (Tinker+2008/2010).

* ``fig02a_hmf_models.png``
  Comparison of six multiplicity functions implemented in
  :mod:`hod_mod.core.halo_mass_function`:
  ``fsigma_tinker08``, ``fsigma_press74``, ``fsigma_sheth99``,
  ``fsigma_warren06``, ``fsigma_bocquet16``, ``fsigma_watson13``.
  Bottom panel shows ratio to Tinker+2008.

* ``fig02b_bias_models.png``
  Redshift evolution of :meth:`~hod_mod.core.halo_mass_function.HaloMassFunction.bias`
  (Tinker+2010) at z = 0, 0.14, 0.5, 1.0.
  Bottom panel shows ratio to z=0.14.

Usage::

    cd $HOD_MOD_REPO
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.cosmology.plot_hmf_bias
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.core.power_spectrum import LinearPowerSpectrum, rho_critical_0
from hod_mod.core.halo_mass_function import HaloMassFunction

_HERE    = Path(__file__).parent
_IMG_DIR = _HERE.parents[2] / "docs" / "_images"
_IMG_DIR.mkdir(parents=True, exist_ok=True)

_THETA  = LinearPowerSpectrum.default_cosmology()
_H      = float(_THETA["h"])
_OM     = float(_THETA["Omega_m"])
_RHO_M  = rho_critical_0() * _OM
_Z_FID  = 0.14

_LOG10M = np.linspace(11, 16, 300)
_M_H    = 10 ** _LOG10M


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_hmf(model="tinker08"):
    pklin = LinearPowerSpectrum()
    return HaloMassFunction(pklin.pk_linear, rho_mean=_RHO_M, model=model)


def _s8_theta(s8):
    """Return theta dict rescaled to a given σ₈."""
    import jax.numpy as jnp
    theta = dict(_THETA)
    hmf0 = _make_hmf()
    s8_fid = float(hmf0.sigma(jnp.array([1e14]), 0.0, _THETA)[0])
    # σ₈ ≈ σ(R=8 Mpc/h) — approximate via σ(M) at characteristic mass
    # Simpler: just rescale ln10As so that P_lin ∝ σ₈²
    s8_planck = 0.832
    scale2 = (s8 / s8_planck) ** 2
    import math
    theta["ln10^{10}A_s"] = float(_THETA["ln10^{10}A_s"]) + math.log(scale2)
    return theta


# ── figure 1: fiducial HMF + bias with S8 variation ─────────────────────────

def make_fig02():
    hmf = _make_hmf("tinker08")

    # ±3σ S8 variants
    s8_fid = 0.832
    s8_hi  = s8_fid + 3 * 0.013
    s8_lo  = s8_fid - 3 * 0.013
    theta_hi = _s8_theta(s8_hi)
    theta_lo = _s8_theta(s8_lo)

    import jax.numpy as jnp
    m = jnp.array(_M_H)
    dndm_fid = np.asarray(hmf.dndm(m, _Z_FID, _THETA))
    dndm_hi  = np.asarray(hmf.dndm(m, _Z_FID, theta_hi))
    dndm_lo  = np.asarray(hmf.dndm(m, _Z_FID, theta_lo))
    bias_fid = np.asarray(hmf.bias(m, _Z_FID, _THETA))
    bias_hi  = np.asarray(hmf.bias(m, _Z_FID, theta_hi))
    bias_lo  = np.asarray(hmf.bias(m, _Z_FID, theta_lo))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # dn/dM
    ax1.semilogy(_M_H, dndm_fid, "C0-",  lw=2,   label=rf"$z={_Z_FID}$ (fiducial)")
    ax1.semilogy(_M_H, dndm_hi,  "C0--", lw=1.2, label=rf"S8 + 3σ = {s8_hi:.3f}")
    ax1.semilogy(_M_H, dndm_lo,  "C0:",  lw=1.2, label=rf"S8 − 3σ = {s8_lo:.3f}")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$M_h\;[h^{-1}\,M_\odot]$", fontsize=12)
    ax1.set_ylabel(r"$\mathrm{d}n/\mathrm{d}M\;[h^4\,M_\odot^{-1}\,\mathrm{Mpc}^{-3}]$",
                   fontsize=11)
    ax1.set_title(rf"Halo mass function (Tinker+2008, $z={_Z_FID}$)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)

    # b(M)
    ax2.loglog(_M_H, bias_fid, "C0-",  lw=2,   label=rf"$z={_Z_FID}$ (fiducial)")
    ax2.loglog(_M_H, bias_hi,  "C0--", lw=1.2, label=rf"S8 + 3σ = {s8_hi:.3f}")
    ax2.loglog(_M_H, bias_lo,  "C0:",  lw=1.2, label=rf"S8 − 3σ = {s8_lo:.3f}")
    ax2.set_xlabel(r"$M_h\;[h^{-1}\,M_\odot]$", fontsize=12)
    ax2.set_ylabel(r"$b(M)$", fontsize=12)
    ax2.set_title("Linear halo bias", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)

    fig.text(0.5, 0.01,
             "Tinker et al. 2008 (arXiv:0803.2706); Planck 2018 (arXiv:1807.06209)",
             ha="center", va="bottom", fontsize=7, color="0.45")
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = _IMG_DIR / "fig02_hmf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── figure 2: HMF model comparison ───────────────────────────────────────────

_HMF_MODELS = [
    ("tinker08", "Tinker+2008 (fiducial)", "C0", "-"),
    ("press74",  "Press+1974",             "0.5", "--"),
    ("sheth99",  "Sheth+1999",             "C4", ":"),
    ("warren06", "Warren+2006",            "C1", "-."),
    ("bocquet16","Bocquet+2016",           "C2", "-"),
    ("watson13", "Watson+2013",            "C3", "--"),
]


def make_fig02a():
    import jax.numpy as jnp
    m = jnp.array(_M_H)

    curves = {}
    for model, label, col, ls in _HMF_MODELS:
        try:
            hmf = _make_hmf(model)
            curves[label] = {
                "dndm": np.asarray(hmf.dndm(m, _Z_FID, _THETA)),
                "color": col, "ls": ls,
            }
        except Exception as e:
            print(f"[skip] {model}: {e}")

    ref_label = "Tinker+2008 (fiducial)"
    dndm_ref  = curves[ref_label]["dndm"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    for label, d in curves.items():
        ax1.semilogy(_M_H, d["dndm"], color=d["color"], ls=d["ls"], lw=2, label=label)
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\mathrm{d}n/\mathrm{d}M\;[h^4\,M_\odot^{-1}\,\mathrm{Mpc}^{-3}]$",
                   fontsize=11)
    ax1.set_title(rf"HMF model comparison ($z={_Z_FID}$)", fontsize=12)
    ax1.legend(fontsize=9, ncol=2, loc="upper right")
    ax1.grid(True, which="both", alpha=0.25)

    for label, d in curves.items():
        ax2.semilogx(_M_H, d["dndm"] / dndm_ref,
                     color=d["color"], ls=d["ls"], lw=2, label=label)
    ax2.axhline(1.0, color="k", lw=0.8, ls=":")
    ax2.set_ylabel(r"$\frac{\mathrm{d}n/\mathrm{d}M}{(\mathrm{Tinker\,+08})}$",
                   fontsize=11)
    ax2.set_xlabel(r"$M_h\;[h^{-1}\,M_\odot]$", fontsize=12)
    ax2.set_ylim(0, 2.7)
    ax2.legend(fontsize=8, ncol=2, loc="upper left")
    ax2.grid(True, which="both", alpha=0.25)

    refs = ("Tinker+2008 (arXiv:0803.2706); Press+1974; Sheth+1999; "
            "Watson+2013; Bocquet+2016")
    fig.text(0.5, 0.01, refs, ha="center", va="bottom", fontsize=6, color="0.45")

    out = _IMG_DIR / "fig02a_hmf_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── figure 3: bias redshift evolution ────────────────────────────────────────

_REDSHIFTS = [
    (0.0,  "C4", "--"),
    (0.14, "C0", "-"),
    (0.5,  "C1", "--"),
    (1.0,  "C3", "--"),
]


def make_fig02b():
    import jax.numpy as jnp
    hmf = _make_hmf("tinker08")
    m   = jnp.array(_M_H)

    curves = {}
    for z, col, ls in _REDSHIFTS:
        curves[z] = {
            "bias": np.asarray(hmf.bias(m, z, _THETA)),
            "color": col, "ls": ls,
        }

    ref_z    = 0.14
    bias_ref = curves[ref_z]["bias"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    for z, d in curves.items():
        ax1.loglog(_M_H, d["bias"], color=d["color"], ls=d["ls"], lw=2,
                   label=rf"$z={z}$")
    ax1.set_ylabel(r"$b(M)$", fontsize=12)
    ax1.set_title("Linear halo bias — redshift evolution (Tinker+2010)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.25)

    for z, d in curves.items():
        ax2.semilogx(_M_H, d["bias"] / bias_ref, color=d["color"], ls=d["ls"], lw=2,
                     label=rf"$z={z}$")
    ax2.axhline(1.0, color="k", lw=0.8, ls=":")
    ax2.set_ylabel(rf"$b(M,z)\,/\,b(M,z={ref_z})$", fontsize=11)
    ax2.set_xlabel(r"$M_h\;[h^{-1}\,M_\odot]$", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)

    refs = ("Tinker et al. 2010 (arXiv:1001.3162); "
            "Tinker et al. 2008 (arXiv:0803.2706)")
    fig.text(0.5, 0.01, refs, ha="center", va="bottom", fontsize=7, color="0.45")

    out = _IMG_DIR / "fig02b_bias_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    make_fig02()
    make_fig02a()
    make_fig02b()
