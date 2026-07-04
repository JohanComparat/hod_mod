r"""Pedagogical halo-abundance figures vs :math:`S_8` (CSST emulator HMF).

Illustrates how the halo abundance in a **half-sky, :math:`0.1<z<0.6`** survey
responds to a :math:`\pm3\sigma` shift in :math:`S_8` away from Planck 2018,
using the CSST CEmulator halo mass function.  All three cosmologies share the
Planck :math:`\Omega_m,h`; only :math:`\sigma_8` (hence :math:`S_8`) changes.

For each of two mass definitions — the emulator-native **M200c** (``FoFM200c``)
and the NFW-converted **M500c** — two figures are produced:

* ``counts``  — cumulative :math:`N(>M)` in the survey comoving volume, with
  :math:`\sqrt{N}` Poisson error bars (how many clusters we expect to find);
* ``density`` — the differential mass function
  :math:`{\rm d}n/{\rm d}\log_{10}M` [Mpc\ :sup:`-3` dex\ :sup:`-1`];

each with a bottom panel showing the ratio w.r.t. the Planck cosmology.

Masses are in :math:`M_\odot` and volumes/densities in physical Mpc (no ``h``),
using :math:`h=0.6736`.

Usage::

    JAX_PLATFORMS=cpu HOD_MOD_RESULTS=/path python -m \
        hod_mod.scripts.forecasts.show_HMF                    # both massdefs
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hod_mod.core.halo_mass_function import CsstHaloMassFunction
from hod_mod.core.halo_profiles import concentration_dutton14_jax
from hod_mod.core.distances import comoving_volume, comoving_volume_element
from hod_mod.core.power_spectrum import rho_critical_0
from hod_mod.gas.conversions import m200_to_m500c
from hod_mod.fitting.planck_prior import PLANCK18_MEANS, PLANCK18_SIGMAS

_RHO_CRIT0 = float(rho_critical_0())          # (Msun/h)/(Mpc/h)^3, z=0
_S8_FID = float(PLANCK18_MEANS["S8"])         # 0.8319
_S8_SIG = float(PLANCK18_SIGMAS["S8"])        # 0.0114
_SIG8_FID = float(PLANCK18_MEANS["sigma8"])   # 0.8111
_LNAS_FID = float(PLANCK18_MEANS["ln10^{10}A_s"])
_H = float(PLANCK18_MEANS["h"])               # 0.6736
_OM = float(PLANCK18_MEANS["Omega_m"])        # 0.3153

# (label, S8, colour) — diverging cool→neutral→warm about the Planck midpoint.
_COSMOS = [
    (r"$S_8-3\sigma$", _S8_FID - 3.0 * _S8_SIG, "#2E6FB0"),
    ("Planck 2018",    _S8_FID,                 "#222222"),
    (r"$S_8+3\sigma$", _S8_FID + 3.0 * _S8_SIG, "#C0392B"),
]
_PLANCK_IDX = 1


def _rho_crit_z(z: float, om: float = _OM) -> float:
    """Comoving critical density [(Msun/h)/(Mpc/h)^3] at redshift z (flat ΛCDM)."""
    ez2 = om * (1.0 + z) ** 3 + (1.0 - om)
    return _RHO_CRIT0 * ez2 / (1.0 + z) ** 3


def _theta_for_S8(S8: float) -> dict:
    """Planck theta dict with sigma8/A_s set to reproduce the target S8.

    S8 = sigma8 (Omega_m/0.3)^0.5, so at fixed Omega_m, sigma8 ∝ S8.  A_s is
    scaled via the sigma8^2 ∝ A_s relation (fitters.py convention).
    """
    th = {k: float(PLANCK18_MEANS[k]) for k in
          ("Omega_b", "Omega_cdm", "h", "n_s", "Omega_m")}
    sig8 = _SIG8_FID * (S8 / _S8_FID)
    th["ln10^{10}A_s"] = _LNAS_FID + 2.0 * np.log(sig8 / _SIG8_FID)
    th["sigma8"] = sig8
    return th


def _interp_loglog(xq, x, y):
    """Log-log linear interpolation (x, y strictly positive, x ascending)."""
    ly = np.log10(np.clip(y, 1e-300, None))
    return 10.0 ** np.interp(np.log10(xq), np.log10(x), ly)


def _n_gt(dndlnM, lnm):
    """Cumulative comoving number density n(>M) = ∫_M^Mmax dn/dlnM dlnM.

    Trapezoid, integrating from the high-mass end (ascending lnm).
    """
    seg = 0.5 * (dndlnM[1:] + dndlnM[:-1]) * np.diff(lnm)
    out = np.zeros_like(dndlnM)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def _dndlnM_target(hmf, theta, z, massdef, m200c_h, m_target_h):
    """dn/dlnM on the target mass grid (Msun/h) for the requested definition.

    The emulator gives dn/dlnM in FoFM200c; for M500c each halo mass is remapped
    M200c→M500c (NFW, Dutton+14 c200c) and dn/dlnM is divided by the Jacobian
    dlnM500c/dlnM200c (number conservation), then interpolated onto the target.
    """
    dndlnM200c = m200c_h * np.asarray(hmf.dndm(m200c_h, z, theta))
    if massdef == "m200c":
        mx, dy = m200c_h, dndlnM200c
    else:  # m500c
        c200c = np.asarray(concentration_dutton14_jax(m200c_h, z))
        rho_c = _rho_crit_z(z)
        r200c = (3.0 * m200c_h / (4.0 * np.pi * 200.0 * rho_c)) ** (1.0 / 3.0)
        m500c_h, _ = m200_to_m500c(m200c_h, c200c, r200c, rho_c)
        m500c_h = np.asarray(m500c_h)
        jac = np.gradient(np.log(m500c_h), np.log(m200c_h))   # dlnM500c/dlnM200c
        mx, dy = m500c_h, dndlnM200c / jac
    dndlnM_t = _interp_loglog(m_target_h, mx, dy)
    n_gt_t = _interp_loglog(m_target_h, mx, _n_gt(dy, np.log(mx)))
    return dndlnM_t, n_gt_t


def compute(massdef, args):
    """Return (m_target[Msun], per-cosmology dict) for one mass definition."""
    hmf = CsstHaloMassFunction(massdef="FoFM200c")

    # redshift shell: mid-points and per-bin comoving volume (half-sky).
    z_edges = np.linspace(args.zmin, args.zmax, args.nz + 1)
    z_mid = 0.5 * (z_edges[1:] + z_edges[:-1])
    Vc = np.asarray(comoving_volume(z_edges, _H, _OM))        # Mpc^3 (physical, <z)
    dV_phys = args.fsky * (Vc[1:] - Vc[:-1])                  # Mpc^3 per z-bin
    dV_hunits = dV_phys * _H ** 3                             # (Mpc/h)^3 per z-bin

    # target display grid [Msun] and the emulator M200c grid [Msun/h] (extends
    # well above the target top so n(>M) captures higher-mass haloes).
    m_target = np.logspace(np.log10(args.mmin), np.log10(args.mmax), args.nm)
    m_target_h = m_target * _H
    m200c_h = np.logspace(np.log10(args.mmin * _H / 1.6),
                          np.log10(args.mmax * _H / 0.5), 240)

    out = {}
    for label, S8, color in _COSMOS:
        theta = _theta_for_S8(S8)
        N_cum = np.zeros_like(m_target)          # cumulative counts in volume
        dens_acc = np.zeros_like(m_target)       # volume-weighted dn/dlnM
        for iz, z in enumerate(z_mid):
            dndlnM_t, n_gt_t = _dndlnM_target(
                hmf, theta, float(z), massdef, m200c_h, m_target_h)
            N_cum += n_gt_t * dV_hunits[iz]
            dens_acc += dndlnM_t * dV_phys[iz]
        dndlog10M = np.log(10.0) * (dens_acc / dV_phys.sum()) * _H ** 3   # Mpc^-3 dex^-1
        out[label] = dict(S8=S8, color=color, N_cum=N_cum, dndlog10M=dndlog10M)
    return m_target, out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

_MDEF_TeX = {"m200c": r"M_{200c}", "m500c": r"M_{500c}"}


def _sci_tex(x):
    """LaTeX 'a×10^b' for a threshold mass (e.g. 2e14 → '2\\times10^{14}')."""
    e = int(np.floor(np.log10(x)))
    mant = x / 10.0 ** e
    return (rf"10^{{{e}}}" if abs(mant - 1.0) < 1e-6
            else rf"{mant:g}\times10^{{{e}}}")


def _mark_threshold(ax, axr, m, res, mA, dots=False):
    """Draw a vertical marker at mass mA and list N(>mA) per cosmology.

    Returns the {label: N(>mA)} dict.  With ``dots=True`` also marks the point
    on each cumulative-count curve.
    """
    for a in (ax, axr):
        a.axvline(mA, color="0.55", ls="--", lw=1.0, zorder=0)
    NA = {}
    x0, y0, dy = 0.97, 0.96, 0.078
    ax.text(x0, y0, rf"$N(>{_sci_tex(mA)}\,M_\odot)$:", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, color="0.2")
    for i, (lab, r) in enumerate(res.items()):
        NA[lab] = float(_interp_loglog(np.array([mA]), m,
                                       np.clip(r["N_cum"], 1e-30, None))[0])
        ax.text(x0, y0 - (i + 1) * dy, rf"{lab} = {NA[lab]:,.0f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=r["color"])
        if dots:
            ax.plot([mA], [NA[lab]], "o", color=r["color"], ms=5, zorder=6,
                    mec="white", mew=0.6)
    return NA


def _two_panel_axes(figsize=(5.6, 5.4)):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.set_constrained_layout_pads(hspace=0.02, wspace=0.0)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    ax.tick_params(labelbottom=False)
    return fig, ax, axr


def _finish(fig, ax, axr, massdef, args, ylabel, ratio_ylabel):
    mtex = _MDEF_TeX[massdef]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", lw=0.5, color="0.85")
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    ax.set_title(rf"Halo abundance vs $S_8$ ($\pm3\sigma$),  "
                 rf"$f_{{\rm sky}}={args.fsky:g}$,  ${args.zmin:g}<z<{args.zmax:g}$",
                 fontsize=12)
    axr.axhline(1.0, color="0.4", lw=1.0)
    axr.set_xscale("log")
    axr.set_xlabel(rf"${mtex}\ \ [M_\odot]$")
    axr.set_ylabel(ratio_ylabel)
    axr.grid(True, which="both", ls=":", lw=0.5, color="0.85")
    fig.align_ylabels([ax, axr])


def make_figures(massdef, m, res, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    mtex = _MDEF_TeX[massdef]
    planck = res[list(res)[_PLANCK_IDX]]

    # ---- Figure 1: total number of haloes N(>M) ----
    fig, ax, axr = _two_panel_axes()
    for label, r in res.items():
        N = r["N_cum"]
        lw = 2.2 if label.startswith("Planck") else 1.6
        ax.plot(m, N, color=r["color"], lw=lw, label=label)
        # sparse Poisson error bars
        sel = slice(2, None, max(1, args.nm // 12))
        ax.errorbar(m[sel], N[sel], yerr=np.sqrt(np.clip(N[sel], 0, None)),
                    fmt="none", ecolor=r["color"], elinewidth=1.0, alpha=0.6)
        axr.plot(m, N / planck["N_cum"], color=r["color"], lw=lw)
    # Poisson detectability band around unity (relative to Planck counts)
    frac = 1.0 / np.sqrt(np.clip(planck["N_cum"], 1e-30, None))
    axr.fill_between(m, 1 - frac, 1 + frac, color="0.75", alpha=0.4, lw=0,
                     label=r"Planck $\sqrt{N}/N$")
    axr.legend(frameon=False, fontsize=10, loc="upper right")
    _mark_threshold(ax, axr, m, res, args.mthresh, dots=True)
    _finish(fig, ax, axr, massdef, args,
            rf"$N(>{mtex})$  in survey volume", r"ratio")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    axr.set_yscale("log"); axr.set_ylim(0.3, 3.0)
    axr.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
    axr.yaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2"]))
    axr.yaxis.set_minor_formatter(NullFormatter())
    p1 = os.path.join(out_dir, f"show_HMF__counts_{massdef.replace('m', 'M')}.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- Figure 2: differential space density dn/dlog10 M ----
    fig, ax, axr = _two_panel_axes()
    for label, r in res.items():
        lw = 2.2 if label.startswith("Planck") else 1.6
        ax.plot(m, r["dndlog10M"], color=r["color"], lw=lw, label=label)
        axr.plot(m, r["dndlog10M"] / planck["dndlog10M"], color=r["color"], lw=lw)
    _mark_threshold(ax, axr, m, res, args.mthresh, dots=False)
    _finish(fig, ax, axr, massdef, args,
            rf"${{\rm d}}n/{{\rm d}}\log_{{10}}{mtex}$  [Mpc$^{{-3}}$ dex$^{{-1}}$]",
            r"ratio")
    axr.set_ylim(0.5, 1.9)
    axr.set_yticks([0.5, 1.0, 1.5])
    p2 = os.path.join(out_dir, f"show_HMF__density_{massdef.replace('m', 'M')}.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    return p1, p2


def _out_dir(args):
    if args.out:
        d = args.out
    else:
        try:
            from hod_mod import paths
            d = os.fspath(paths.results_root() / "show_HMF")
        except Exception:
            d = os.path.join(os.environ.get("HOD_MOD_RESULTS", "."), "show_HMF")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--massdef", choices=["both", "m200c", "m500c"], default="both")
    ap.add_argument("--zmin", type=float, default=0.1)
    ap.add_argument("--zmax", type=float, default=0.6)
    ap.add_argument("--fsky", type=float, default=0.5)
    ap.add_argument("--mmin", type=float, default=1e13, help="M_sun (physical)")
    ap.add_argument("--mmax", type=float, default=5e15, help="M_sun (physical)")
    ap.add_argument("--mthresh", type=float, default=2e14,
                    help="M_sun (physical): cumulative N(>M) annotated on the figures")
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--nm", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = _out_dir(args)
    defs = ["m200c", "m500c"] if args.massdef == "both" else [args.massdef]
    print(f"[show_HMF] S8 = {_S8_FID:.4f} ± 3×{_S8_SIG:.4f}; "
          f"{args.zmin} < z < {args.zmax}, f_sky={args.fsky}")
    for md in defs:
        m, res = compute(md, args)
        p1, p2 = make_figures(md, m, res, args, out_dir)
        pl = res[list(res)[_PLANCK_IDX]]
        lo, hi = res[list(res)[0]], res[list(res)[2]]

        def _Ngt(r, M):
            return float(_interp_loglog(np.array([M]), m,
                                        np.clip(r["N_cum"], 1e-30, None))[0])
        print(f"[{md}] N(>{args.mthresh:.0e} Msun)  Planck={_Ngt(pl, args.mthresh):,.0f}  "
              f"(-3σ={_Ngt(lo, args.mthresh):,.0f}, +3σ={_Ngt(hi, args.mthresh):,.0f})")
        # abundance ratio at the high-mass end (S8 sensitivity is strongest there)
        j = np.searchsorted(m, 1e15)
        print(f"      dn/dlog10M ratio at 1e15 Msun:  "
              f"-3σ={lo['dndlog10M'][j]/pl['dndlog10M'][j]:.2f}, "
              f"+3σ={hi['dndlog10M'][j]/pl['dndlog10M'][j]:.2f}")
        print(f"      wrote {p1}\n            {p2}")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
