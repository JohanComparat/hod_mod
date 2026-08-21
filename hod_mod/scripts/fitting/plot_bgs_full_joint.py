"""Figures for the BGS S1 full-model joint fit (galaxies + hot gas + AGN).

Reads a fit output directory written by
:mod:`hod_mod.scripts.fitting.fit_bgs_full_joint` (``flatchain.npz`` +
``map_result.json``) and produces three figures, overlaying the **MAP** and the
**posterior-median** model on the data:

* ``<prefix>__corner.png``      — posterior corner of the free params, MAP overlaid.
* ``<prefix>__observables.png`` — wp, ESD (DES/HSC/KIDS), XLF (z=0.1, 0.4), AGN
  bias b(L_X) and the X-ray broad-band w(theta): data +/- err vs MAP + median.
* ``<prefix>__xray_bands.png``  — the 15 narrow 0.1-keV band w(theta) panels.

``prefix`` is ``bgs_full_joint_fixedzm15`` (default) or ``bgs_full_joint_allparams``
(``--free-zm15``).  Figures go into the run directory; ``--docs`` additionally
writes them into ``docs/_images/`` with the names the ``bgs_full_joint`` doc page
embeds.

Usage
-----
::

    HOD_MOD_DATA_DIR=/home/comparat/data/hod_mod_data JAX_PLATFORMS=cpu \\
        python -m hod_mod.scripts.fitting.plot_bgs_full_joint              # fixed-ZM15
    ... plot_bgs_full_joint --free-zm15                                    # all-params run
    ... plot_bgs_full_joint --out-dir /path/to/run --docs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hod_mod.paths import results_root
from hod_mod.scripts.fitting.fit_bgs_full_joint import _DEFAULT_OBS

# 15 narrow bands: 0.5-0.6 ... 1.9-2.0 keV (see fit_xray_joint_bands._BANDS).
_BAND_LABELS = [f"{lo / 1000:.1f}–{(lo + 100) / 1000:.1f} keV"
                for lo in range(500, 2000, 100)]

_MAP_C, _MED_C, _DATA_C = "tab:green", "tab:blue", "k"


def _load(out: Path):
    """Return (chain, names, theta_med, theta_map) from a fit output dir."""
    npz = np.load(out / "flatchain.npz")
    chain = np.asarray(npz["flatchain"], float)
    names = [str(n) for n in npz["param_names"]]
    theta_med = np.median(chain, axis=0)
    theta_map = None
    mj = out / "map_result.json"
    if mj.exists():
        theta_map = np.asarray(json.load(open(mj))["theta"], float)
    return chain, names, theta_med, theta_map


def fig_corner(chain, names, theta_map, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    fig = corner.corner(
        chain, labels=names,
        truths=(list(theta_map) if theta_map is not None else None),
        truth_color=_MAP_C, quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_fmt=".2f", title_kwargs=dict(fontsize=7), label_kwargs=dict(fontsize=7))
    fig.suptitle("BGS S1 full-joint posterior  (green = MAP)", fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _overlay(ax, x, y_map, y_med):
    if y_map is not None:
        ax.plot(x, y_map, color=_MAP_C, lw=1.6, label="MAP")
    ax.plot(x, y_med, color=_MED_C, lw=1.6, ls="--", label="posterior median")


_C1H, _C2H, _CPM, _CGAS, _CAGN = "tab:red", "tab:purple", "tab:brown", "tab:red", "tab:orange"


def fig_observables(J, pred_map, pred_med, comp, path, chain=None, names=None):
    """Data vs model per observable, decomposed: wp / ESD into 1-halo & 2-halo
    (ESD also the central point mass), the broad X-ray into gas & AGN.  When a
    chain is given, the AGN-bias panel also shows the model's +/-3 sigma response
    in the most bias-relevant parameter (the M_BH-M* normalisation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []  # each: a function ax -> None

    if "wp" in J.obs:
        d = J.data_gal["wp"]; cw = comp.get("wp") if comp else None
        def _wp(ax, d=d, cw=cw):
            ax.errorbar(d["rp"], d["wp"], yerr=d["err"], fmt="o", ms=3, color=_DATA_C, label="data", zorder=5)
            if pred_map is not None:
                ax.plot(d["rp"], pred_map["wp"], color=_MAP_C, lw=1.8, label="MAP total")
            ax.plot(d["rp"], pred_med["wp"], color=_MED_C, lw=1.1, ls="--", label="median total")
            if cw is not None:
                ax.plot(d["rp"], cw["1h"], color=_C1H, lw=1.1, ls=":", label="1-halo")
                ax.plot(d["rp"], cw["2h"], color=_C2H, lw=1.1, ls="-.", label="2-halo")
            ax.set(xscale="log", yscale="log", xlabel=r"$r_p$ [Mpc/$h$]", ylabel=r"$w_p$")
            ax.set_title("wp"); ax.legend(fontsize=6)
        panels.append(_wp)

    if "esd" in J.obs:
        for sv in J.data_gal["esd"]:
            d = J.data_gal["esd"][sv]; ce = comp["esd"][sv] if (comp and "esd" in comp) else None
            def _esd(ax, sv=sv, d=d, ce=ce):
                ax.errorbar(d["rp"], d["ds"], yerr=d["err"], fmt="o", ms=3, color=_DATA_C, label="data", zorder=5)
                if pred_map is not None:
                    ax.plot(d["rp"], pred_map["esd"][sv], color=_MAP_C, lw=1.8, label="MAP total")
                ax.plot(d["rp"], pred_med["esd"][sv], color=_MED_C, lw=1.1, ls="--", label="median total")
                if ce is not None:
                    ax.plot(d["rp"], ce["one_h"], color=_C1H, lw=1.1, ls=":", label="1-halo")
                    ax.plot(d["rp"], ce["two_h"], color=_C2H, lw=1.1, ls="-.", label="2-halo")
                    if np.any(np.asarray(ce["point_mass"]) > 0):
                        ax.plot(d["rp"], ce["point_mass"], color=_CPM, lw=1.0, ls=(0, (1, 1)), label="point mass")
                ax.set(xscale="log", yscale="log", xlabel=r"$r_p$ [Mpc/$h$]",
                       ylabel=r"$\Delta\Sigma$ [$M_\odot h/\mathrm{pc}^2$]")
                ax.set_title(f"ESD {sv}"); ax.legend(fontsize=6)
            panels.append(_esd)

    if "xlf" in J.obs:
        for z in J.xlf_z:
            d = J.data_agn["xlf"][z]
            def _xlf(ax, z=z, d=d):
                ax.errorbar(d["log10lx"], d["phi"], yerr=d["err"], fmt="o", ms=3, color=_DATA_C, label="data", zorder=5)
                _overlay(ax, d["log10lx"], (pred_map["xlf"][z] if pred_map else None), pred_med["xlf"][z])
                ax.set(yscale="log", xlabel=r"$\log_{10} L_X$ [erg/s, hard]", ylabel=r"$\phi$ [Mpc$^{-3}$]")
                ax.set_title(f"XLF  z={z}"); ax.legend(fontsize=6)
            panels.append(_xlf)

    if "agn_bias" in J.obs:
        d = J.data_agn["bias"]
        def _bias(ax, d=d):
            ax.errorbar(d["log10lx_soft"], d["bias"], yerr=d["bias_err"], fmt="o", ms=3,
                        color=_DATA_C, label="data", zorder=5)
            # smooth model b(L_X) at the data's median redshift, with the +/-3 sigma
            # response in agn_mu_bh (M_BH-M* norm) -- the dominant driver of host mass
            if chain is not None and names is not None and "agn_mu_bh" in names:
                idx = names.index("agn_mu_bh")
                med = np.median(chain, axis=0)
                sig = float(np.std(chain[:, idx]))
                zrep = float(np.median(d["z"]))
                lxg = np.linspace(float(np.min(d["log10lx_soft"])) - 0.2,
                                  float(np.max(d["log10lx_soft"])) + 0.3, 40)

                def _bcurve(mu, med=med, lxg=lxg, zrep=zrep):
                    v = _mp(names, med); v["agn_mu_bh"] = mu
                    pw = J._powell_at(zrep); J._apply_agn_params(pw, v)
                    return np.asarray(pw.agn_bias_of_lx(lxg, band="soft"), float)

                b0 = _bcurve(med[idx])
                bhi = _bcurve(med[idx] + 3 * sig); blo = _bcurve(med[idx] - 3 * sig)
                ax.plot(lxg, b0, color=_MED_C, lw=1.5, label="median model")
                ax.fill_between(lxg, np.minimum(blo, bhi), np.maximum(blo, bhi),
                                color=_MED_C, alpha=0.2, label=r"$\pm3\sigma$ ($\mu_{\rm BH}$)")
            if pred_map is not None:
                ax.plot(d["log10lx_soft"], pred_map["agn_bias"], "s", color=_MAP_C, ms=5, label="MAP")
            ax.plot(d["log10lx_soft"], pred_med["agn_bias"], "^", color=_MED_C, ms=5, label="median")
            ax.set(xlabel=r"$\log_{10} L_X^{\rm soft}$ [erg/s]", ylabel="halo bias")
            ax.set_title("AGN bias  b(L_X)"); ax.legend(fontsize=6)
        panels.append(_bias)

    if "xray_broad" in J.obs and hasattr(J, "data_xray_broad"):
        d = J.data_xray_broad; cb = comp.get("xray_broad") if comp else None
        def _broad(ax, d=d, cb=cb):
            yb = d["wtheta"]; pos = yb > 0
            ax.errorbar(np.asarray(d["theta_as"])[pos], yb[pos], yerr=np.asarray(d["err"])[pos],
                        fmt="o", ms=3, color=_DATA_C, label="data", zorder=5)
            if cb is not None:
                ax.plot(d["theta_as"], cb["gas"], color=_CGAS, lw=1.1, ls="-", label="gas")
                ax.plot(d["theta_as"], cb["agn"], color=_CAGN, lw=1.1, ls=":", label="AGN")
                ax.plot(d["theta_as"], cb["total"], color=_MAP_C, lw=1.6, label="total (MAP)")
            ax.set(xscale="log", yscale="log", xlabel=r"$\theta$ [arcsec]", ylabel=r"broad $w(\theta)$")
            ax.set_title("X-ray broad (0.5-2 keV)"); ax.legend(fontsize=6)
        panels.append(_broad)

    n = len(panels); ncol = min(3, n) or 1; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
    for i, pf in enumerate(panels):
        pf(axes[i // ncol][i % ncol])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("BGS S1 full-joint: data vs model, decomposed "
                 "(1-halo/2-halo for galaxies; gas/AGN for X-ray)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_xray_bands(J, comp_map, comp_med, path):
    """Per-band w(theta) in log-log, decomposed into hot-gas / AGN / total."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xb = comp_map["xray"] if comp_map else comp_med["xray"]
    th = xb["th_as"]; mask = xb["mask"]; wtheta, err = xb["wtheta"], xb["err"]
    nb = wtheta.shape[0]
    labels = (_BAND_LABELS + [f"band {i}" for i in range(nb)])[:nb]
    thm = th[mask]; xlo, xhi = thm.min() * 0.6, thm.max() * 2.0

    ncol = 4
    nrow = int(np.ceil(nb / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow),
                             squeeze=False, sharex=True)
    for b in range(nb):
        ax = axes[b // ncol][b % ncol]
        yb = wtheta[b][mask]; eb = err[b][mask]; pos = yb > 0
        ax.errorbar(thm[pos], yb[pos], yerr=eb[pos], fmt="o", ms=2.5, color=_DATA_C, lw=0.7, zorder=5)
        ax.plot(th, xb["gas"][b], color=_CGAS, lw=1.0, ls="-", label="gas")
        ax.plot(th, xb["agn"][b], color=_CAGN, lw=1.0, ls=":", label="AGN")
        ax.plot(th, xb["total"][b], color=_MAP_C, lw=1.4, label="total (MAP)")
        if comp_med is not None:
            ax.plot(th, comp_med["xray"]["total"][b], color=_MED_C, lw=1.0, ls="--", label="total (med)")
        ax.set(xscale="log", yscale="log", xlim=(xlo, xhi))
        ymax = max(float(np.nanmax(yb[pos])) if pos.any() else 1e-3, float(np.nanmax(xb["total"][b])))
        ax.set_ylim(ymax * 3e-3, ymax * 2.0)
        ax.set_title(labels[b], fontsize=8)
    axes[0][0].legend(fontsize=6)
    for j in range(nb, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("BGS S1 full-joint: narrow-band $w(\\theta)$ decomposition "
                 "(log-log; gas / AGN / total)", fontsize=12)
    fig.supxlabel(r"$\theta$ [arcsec]")
    fig.supylabel(r"$w(\theta)$")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _mp(names, theta):
    """Parameter name -> value dict."""
    return {n: float(theta[i]) for i, n in enumerate(names)}


# The gas-sector parameters the figures below read out of the chain.  They must all
# be free parameters of the band model (fit_xray_joint_bands._PARAMS) — test_full_joint
# pins that, because a silent drift here (the native-DPM re-base replaced lx_norm/
# lx_slope/kt_norm/kt_slope) only shows up as a KeyError at plotting time.
_GAS_KEYS = ("log10_ne03", "beta_n", "log10_p03", "beta_P", "p2", "r_max", "z_metal")


def _gas(mp):
    """The gas subset of a parameter dict, with a message that names the cause."""
    missing = [k for k in _GAS_KEYS if k not in mp]
    if missing:
        raise KeyError(f"gas parameters {missing} are absent from this chain — it "
                       f"predates the native-DPM re-base of the band model and cannot "
                       f"be plotted with the current code; re-run the fit.")
    return {k: float(mp[k]) for k in _GAS_KEYS}


def _dpm_profiles(mp):
    """(density, pressure) DPM profiles for the fit's native gas parameters.

    The band model varies the four native DPM parameters directly — ``n_e,0.3``,
    ``beta_n`` (density) and ``P_0.3``, ``beta_P`` (pressure, and hence
    ``T = P/n_e``) — so nothing has to be calibrated here: the posterior *is* the
    profile.  ``p2``/``r_max`` deform only the density (outer slope
    ``alpha_out = alpha_prof + 2 p2``, truncated at ``r_max R_200``); the pressure
    keeps the native DPM model-2 shape, exactly as in the likelihood.
    """
    from hod_mod.scripts import validate_gas_profiles as v
    from hod_mod.scripts.fitting import fit_xray_joint_bands as XB
    mp = _gas(mp)
    dp = v._make_density_variant(model=2, ne_03=10.0 ** mp["log10_ne03"],
                                 beta=mp["beta_n"], alpha_in=XB._ALPHA_PROF,
                                 alpha_tr=2.0,
                                 alpha_out=XB._ALPHA_PROF + 2.0 * mp["p2"])
    dp._r_max_factor = float(mp["r_max"])
    pp = v._make_pressure_variant(model=2, P_03=10.0 ** mp["log10_p03"],
                                  beta=mp["beta_P"])
    return dp, pp


def _gas_mass_grid(z, n_m=45, lm_range=(13.0, 15.4)):
    """(m200, r200, r500c, log10 M500c) for the scaling-relation panels.

    m200/r200/r500c are in the DPM's h-units [Msun/h, Mpc/h]; the returned
    log10 M500c is physical (Msun), the abscissa the literature relations use.
    """
    from hod_mod.scripts import validate_gas_profiles as v
    m200 = np.geomspace(1.2e13, 5e15, n_m) * v._H
    r200 = v._r200(m200, z)
    c200 = v._c200_approx(m200)
    m500c, r500c = v.m200_to_m500c(m200, c200, r200, v._rho_crit_z(z))
    lm = np.log10(np.asarray(m500c, float) / v._H)
    sel = (lm >= lm_range[0]) & (lm <= lm_range[1])
    return m200[sel], r200[sel], np.asarray(r500c, float)[sel], lm[sel]


def _lx_kt_of_mass(mp, grid, ez, cool, n_x=200):
    """R_500c-integrated L_X (0.5-2 keV) [erg/s] and emission-weighted kT [keV].

    Thin adapter over :func:`hod_mod.fitting.dpm_bands.lx_kt_of_mass`, which is the
    same routine ``make_xray_diagnostics`` uses — the scaling relations plotted
    here and there must be the one quantity, computed one way.
    """
    from hod_mod.fitting.dpm_bands import lx_kt_of_mass
    from hod_mod.gas.conversions import _MPC_CM
    from hod_mod.scripts import validate_gas_profiles as v
    from hod_mod.scripts.fitting import fit_xray_joint_bands as XB

    mp = _gas(mp)
    dp, pp = _dpm_profiles(mp)
    return lx_kt_of_mass(grid[0], grid[1], grid[2], dp, pp, cool,
                         ne03=10.0 ** mp["log10_ne03"], beta_n=mp["beta_n"],
                         p03=10.0 ** mp["log10_p03"], beta_P=mp["beta_P"],
                         ez=ez, h=v._H, mpc_cm=_MPC_CM,
                         z_metal=float(np.clip(mp["z_metal"], 0.05, 3.0)),
                         t_min=XB._T_MIN_XRAY, n_x=n_x)


def fig_gas(J, chain, names, theta_map, theta_med, path):
    """Hot-gas scaling relations (L_X-M, kT-M, L_X-kT) integrated from the fitted
    DPM profiles + posterior band + literature, and the radial profiles (n_e, T,
    P_e) those same parameters describe.

    Since the native-DPM re-base of the band model there are no free L_X-M / kT-M
    power laws to read off: the relations are *predictions* of the four gas
    parameters (n_e,0.3, beta_n, P_0.3, beta_P), obtained here by integrating the
    posterior's own profiles inside R_500c."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hod_mod.scripts import validate_gas_profiles as v
    from hod_mod.scripts.fitting import fit_xray_joint_bands as XB

    z = J.z
    ez = float(J._S["ez"])
    cool = J._S["cool_broad"]                      # APEC 0.5-2 keV, the fit's own table
    mp_med = _mp(names, theta_med)
    mp_map = _mp(names, theta_map) if theta_map is not None else None
    grid = _gas_mass_grid(z)
    lm = grid[3]
    m500 = 10.0 ** lm

    LX_med, kT_med = _lx_kt_of_mass(mp_med, grid, ez, cool)
    LX_map, kT_map = (_lx_kt_of_mass(mp_map, grid, ez, cool) if mp_map else (None, None))

    rng = np.random.default_rng(0)
    sub = chain[rng.choice(len(chain), size=min(400, len(chain)), replace=False)]
    curves = [_lx_kt_of_mass(_mp(names, th), grid, ez, cool) for th in sub]
    LXs = np.array([c[0] for c in curves])
    kTs = np.array([c[1] for c in curves])
    LX_lo, LX_hi = np.percentile(LXs, [16, 84], axis=0)
    kT_lo, kT_hi = np.percentile(kTs, [16, 84], axis=0)

    # A halo whose gas is everywhere below the X-ray selection cut has L_X = kT = 0.
    # On a log axis that must read as a GAP, not as a plunge to the bottom of the
    # frame, so blank the zeros and say out loud how much of the posterior is cold.
    def _pos(a):
        return None if a is None else np.where(np.asarray(a, float) > 0, a, np.nan)
    LX_med, kT_med, LX_map, kT_map = (_pos(LX_med), _pos(kT_med), _pos(LX_map), _pos(kT_map))
    LX_lo, LX_hi, kT_lo, kT_hi = (_pos(LX_lo), _pos(LX_hi), _pos(kT_lo), _pos(kT_hi))
    f_cold = float(np.mean(~np.any(LXs > 0, axis=1)))
    map_cold = LX_map is not None and not np.any(np.isfinite(LX_map))

    def _try(loader):  # literature scatter data is optional (may be unstaged on dahu)
        try:
            return loader()
        except Exception as e:
            print(f"[plot] gas literature data missing ({e.__class__.__name__}); "
                  f"plotting analytic relations only", flush=True)
            return None, None, None

    def _sc(ax, x, y, **kw):
        if x is not None:
            ax.scatter(x, y, **kw)

    M_lo, Lx_lo20, kT_lo20 = _try(v._load_lovisari20_data)
    M_bu, Lx_bu, kT_bu = _try(v._load_bulbul18)
    m_lit = np.logspace(13, 15.4, 60)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))

    ax = axes[0, 0]
    ax.fill_between(m500, LX_lo, LX_hi, color="tab:blue", alpha=0.2, label="68% posterior")
    ax.loglog(m500, LX_med, color="tab:blue", lw=2, ls="--", label="median")
    if LX_map is not None:
        ax.loglog(m500, LX_map, color="tab:green", lw=2, label="MAP")
    _sc(ax, M_lo, Lx_lo20, s=11, color="gray", alpha=.7, label="Lovisari+2020")
    _sc(ax, M_bu, Lx_bu, s=11, color="chocolate", alpha=.7, label="Bulbul+2018")
    ax.loglog(m_lit, v._lovisari20_lx(m_lit, z=z), "k--", lw=1.1, label="Lovisari+2020 fit")
    ax.loglog(m_lit, v._gas_py_lx(m_lit, z=z)[0], "k:", lw=1.1, label="Comparat+2025")
    ax.set(xlabel=r"$M_{500c}$ [$M_\odot$]", ylabel=r"$L_X$ (0.5-2 keV) [erg/s]",
           ylim=(1e40, 1e46)); ax.set_title(r"$L_X$-$M_{500c}$")
    ax.legend(fontsize=6.5); ax.grid(alpha=.2)
    if f_cold > 0 or map_cold:
        note = (f"{100 * f_cold:.0f}% of posterior samples: "
                rf"$T<T_{{\rm min}}$ ({XB._T_MIN_XRAY:g} keV) at every mass" "\n"
                "(no X-ray-emitting gas; blanks are $L_X=0$)")
        if map_cold:
            note += "\nMAP is one of them — no MAP curve to draw"
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=6.5,
                style="italic", va="bottom")

    ax = axes[0, 1]
    ax.fill_between(m500, kT_lo, kT_hi, color="tab:blue", alpha=.2)
    ax.loglog(m500, kT_med, color="tab:blue", lw=2, ls="--", label="median")
    if kT_map is not None:
        ax.loglog(m500, kT_map, color="tab:green", lw=2, label="MAP")
    _sc(ax, M_lo, kT_lo20, s=11, color="gray", alpha=.7, label="Lovisari+2020")
    _sc(ax, M_bu, kT_bu, s=11, color="chocolate", alpha=.7, label="Bulbul+2018")
    ax.loglog(m_lit, v._lovisari20_kt(m_lit, z=z), "k--", lw=1.1, label="Lovisari+2020 fit")
    ax.loglog(m_lit, v._gas_py_kt(m_lit, z=z)[0], "k:", lw=1.1, label="Comparat+2025")
    ax.set(xlabel=r"$M_{500c}$ [$M_\odot$]", ylabel=r"$kT$ [keV]"); ax.set_title(r"$kT$-$M_{500c}$")
    ax.legend(fontsize=6.5); ax.grid(alpha=.2)

    ax = axes[0, 2]
    ax.loglog(kT_med, LX_med, color="tab:blue", lw=2, ls="--", label="median")
    if LX_map is not None:
        ax.loglog(kT_map, LX_map, color="tab:green", lw=2, label="MAP")
    _sc(ax, kT_lo20, Lx_lo20, s=11, color="gray", alpha=.7, label="Lovisari+2020")
    _sc(ax, kT_bu, Lx_bu, s=11, color="chocolate", alpha=.7, label="Bulbul+2018")
    ax.set(xlabel=r"$kT$ [keV]", ylabel=r"$L_X$ [erg/s]"); ax.set_title(r"$L_X$-$kT$")
    ax.legend(fontsize=6.5); ax.grid(alpha=.2)

    # radial profiles straight from the fitted native-DPM parameters
    mp_prof = mp_map or mp_med
    dp, pp = _dpm_profiles(mp_prof)
    a10 = v.PressureProfileA10()
    x = np.logspace(-2, np.log10(3), 200)
    cols = plt.cm.viridis(np.linspace(.1, .85, 3))
    ax_ne, ax_t, ax_p = axes[1, 0], axes[1, 1], axes[1, 2]
    for log10m, c in zip([13, 14, 15], cols):
        m_p = 10.0 ** log10m; r2 = v._r200(m_p, z); rr = x * r2
        ne = dp.density_3d(rr, m_p, r2, z, v._OM)
        Pe = pp._pressure_3d(rr, m_p, r2, z, v._OM)
        T = v.temperature_from_profiles(Pe, ne)
        lbl = rf"$10^{{{log10m}}}$"
        ax_ne.loglog(x, ne, color=c, lw=2, label=lbl)
        ax_t.loglog(x, T, color=c, lw=2, label=lbl)
        ax_p.loglog(x, Pe, color=c, lw=2, label=lbl)
        c200 = v._c200_approx(m_p)
        m5, r5 = v.m200_to_m500c(np.array([m_p]), np.array([c200]), np.array([r2]), v._rho_crit_z(z))
        ax_p.loglog(x, a10._p3d(rr / float(r5[0]), float(m5[0]), z, v._H, v._OM),
                    color=c, lw=1, ls=":")
    ax_ne.set(xlabel=r"$r/R_{200}$", ylabel=r"$n_e$ [cm$^{-3}$]"); ax_ne.set_title("Electron density $n_e(r)$")
    ax_ne.legend(fontsize=7, title=r"$M_{200}\,[M_\odot/h]$"); ax_ne.grid(alpha=.2)
    ax_t.set(xlabel=r"$r/R_{200}$", ylabel=r"$T$ [keV]"); ax_t.set_title(r"Temperature $T=P_e/n_e$")
    ax_t.legend(fontsize=7); ax_t.grid(alpha=.2)
    ax_p.set(xlabel=r"$r/R_{200}$", ylabel=r"$P_e$ [keV cm$^{-3}$]")
    ax_p.set_title("Electron pressure (dotted = A10)"); ax_p.legend(fontsize=7); ax_p.grid(alpha=.2)

    fig.suptitle("BGS S1 full-joint: hot-gas scaling relations (top; integrated inside "
                 r"$R_{500c}$) and radial profiles (bottom) — both from the fitted "
                 "native-DPM gas parameters", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def fig_agn_host_smf(J, theta_map, theta_med, path):
    """Stellar-mass function of AGN host galaxies above hard-X-ray luminosity
    thresholds (model prediction, to be compared to Bongiorno+2016).

    Built from the Powell AGN occupation N_AGN(>L_X | M_halo) weighted by the HMF
    and mapped into stellar mass via the Girelli+2020 SHMR (with its scatter).
    Centrals only (the model has no satellite AGN yet)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = theta_map if theta_map is not None else theta_med
    v = {n: float(theta[i]) for i, n in enumerate(J.names)}
    pw = J._powell_at(J.z); J._apply_agn_params(pw, v)
    h = float(pw.theta["h"])
    log10ms = np.asarray(pw._log10ms, float)              # host M* [Msun] per halo
    dndlog, _ = pw._dndm_bias()                           # dn/dlog10M_halo [(Mpc/h)^-3 dex^-1]
    p = pw._p_lx_given_m() * 10.0 ** pw.log10_ferdf       # (NM, Nlx) incl. active fraction
    ms_grid = np.linspace(9.0, 12.3, 60)
    sig_ms = max(float(pw.sigma_ms), 0.05)
    G = (np.exp(-0.5 * ((ms_grid[:, None] - log10ms[None, :]) / sig_ms) ** 2)
         / (np.sqrt(2 * np.pi) * sig_ms))                 # SHMR scatter kernel (Nms, NM)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for lth, c in zip([43.0, 44.0], ["tab:blue", "tab:red"]):
        sel = pw.loglx >= lth
        n_agn = np.sum(p[:, sel], axis=1) * pw.dlx        # N_AGN(>L_th | M_halo)
        w = dndlog * n_agn                                # (NM,)
        phi = np.trapezoid(w[None, :] * G, pw.log10m, axis=1) * h ** 3  # Mpc^-3 dex^-1
        ax.plot(ms_grid, phi, color=c, lw=2, label=fr"model $\log_{{10}}L_X^{{2\text{{-}}10}}>{lth:.0f}$")
    ax.set(yscale="log", xlim=(9.0, 12.3), ylim=(1e-7, 3e-3),
           xlabel=r"$\log_{10} M_*$ [$M_\odot$]",
           ylabel=r"$\Phi_{\rm AGN}(M_*)$ [Mpc$^{-3}$ dex$^{-1}$]")
    ax.set_title("AGN host stellar-mass function")
    ax.grid(alpha=0.2); ax.legend(fontsize=9, title="compare to Bongiorno+2016")
    ax.text(0.02, 0.03, "Bongiorno+2016 data not staged locally — model prediction only\n"
            "(host M* via Girelli+2020 SHMR + scatter; centrals only)",
            transform=ax.transAxes, fontsize=7.5, style="italic", va="bottom")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _print_chi2(J, label, theta):
    bd = J.chi2_breakdown(theta)
    tot = float(sum(bd.values()))
    dof = max(J.n_data() - J.ndim, 1)
    br = {k: round(x, 1) for k, x in bd.items()}
    print(f"[plot] {label:7s} chi2/dof = {tot:.1f}/{dof} = {tot / dof:.3f}  {br}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="S1")
    ap.add_argument("--free-zm15", action="store_true",
                    help="the all-params run (bgs_full_joint_allparams)")
    ap.add_argument("--observables", nargs="+", default=None,
                    help=f"observable set the fit used (default {_DEFAULT_OBS})")
    ap.add_argument("--out-dir", default=None,
                    help="fit output dir (default results_root()/bgs_full_joint_<tag>)")
    ap.add_argument("--docs", action="store_true",
                    help="also write figures into docs/_images/ with the doc-page names")
    args = ap.parse_args(argv)

    tag = "allparams" if args.free_zm15 else "fixedzm15"
    prefix = f"bgs_full_joint_{tag}"
    out = Path(args.out_dir) if args.out_dir else results_root() / prefix
    if not (out / "flatchain.npz").exists():
        raise SystemExit(f"[plot] no flatchain.npz in {out} — run fit_bgs_full_joint first "
                         f"(or pass --out-dir).")

    chain, names, theta_med, theta_map = _load(out)
    dests = [out]
    if args.docs:
        docs_img = Path(__file__).resolve().parents[3] / "docs" / "_images"
        docs_img.mkdir(parents=True, exist_ok=True)
        dests.append(docs_img)

    # Build the model so we can overlay predictions.  Reconstruct the EXACT data
    # selection from fit_config.json (written by the driver) so the plotted data
    # match the fit; fall back to CLI/defaults for older runs.
    from hod_mod.fitting.full_joint import JointFull
    cfg = {}
    if (out / "fit_config.json").exists():
        cfg = json.load(open(out / "fit_config.json"))
    obs = args.observables or cfg.get("observables") or _DEFAULT_OBS
    print(f"[plot] building JointFull ({tag}, {len(names)} params) ...", flush=True)
    J = JointFull(sample=cfg.get("sample", args.sample),
                  free_zm15=cfg.get("free_zm15", args.free_zm15), observables=obs,
                  rp_min_wp=cfg.get("rp_min_wp", 0.5), rp_min_esd=cfg.get("rp_min_esd", 2.0),
                  esd_surveys=cfg.get("esd_surveys", ("DES", "HSC", "KIDS")),
                  esd_rp_max=cfg.get("esd_rp_max", float("inf")),
                  xlf_z=cfg.get("xlf_z", (0.1, 0.4)),
                  xlf_lx_min=cfg.get("xlf_lx_min", float("-inf")),
                  agn_bias_refs=cfg.get("agn_bias_refs"),
                  # Pre-0.4 runs recorded kt_prior_sig, a flag whose indices stopped
                  # meaning kt_norm/kt_slope at the native-DPM re-base.  It is not
                  # reconstructible as a widen factor, so old configs plot against the
                  # default prior; the chain itself is unaffected.
                  gas_prior_widen=cfg.get("gas_prior_widen"),
                  f_sys=cfg.get("f_sys", 0.05), hmf_backend=cfg.get("hmf", "tinker08"),
                  verbose=False)
    if theta_med.size != J.ndim or list(names) != list(J.names):
        # The commonest cause is a chain that predates the native-DPM re-base of the
        # band model (lx_norm/lx_slope/kt_norm/kt_slope -> log10_ne03/beta_n/
        # log10_p03/beta_P + agn_gamma): those parameters no longer exist, so the
        # model cannot be evaluated at that theta at all.
        raise SystemExit(f"[plot] chain params do not match the model.\n"
                         f"        chain ({theta_med.size}): {list(names)}\n"
                         f"        model ({J.ndim}): {list(J.names)}\n"
                         f"        wrong --observables/--free-zm15, or a pre-DPM chain "
                         f"that must be re-fitted.")
    if theta_map is not None and theta_map.size != J.ndim:
        print(f"[plot] map_result.json has {theta_map.size} params != model {J.ndim}; "
              f"ignoring the MAP overlay", flush=True)
        theta_map = None

    pred_med = J.predict(theta_med)
    pred_map = J.predict(theta_map) if theta_map is not None else None
    comp_med = J.predict_components(theta_med)
    comp_map = J.predict_components(theta_map) if theta_map is not None else None
    comp = comp_map or comp_med

    for dd in dests:
        fig_corner(chain, names, theta_map, dd / f"{prefix}__corner.png")
        fig_observables(J, pred_map, pred_med, comp, dd / f"{prefix}__observables.png",
                        chain=chain, names=names)
        if {"xray_bands", "xray_broad"} & J.obs:
            fig_xray_bands(J, comp_map, comp_med, dd / f"{prefix}__xray_bands.png")
            fig_gas(J, chain, names, theta_map, theta_med, dd / f"{prefix}__gas.png")
        if {"xlf", "agn_bias"} & J.obs:
            fig_agn_host_smf(J, theta_map, theta_med, dd / f"{prefix}__agn_host_smf.png")

    if theta_map is not None:
        _print_chi2(J, "MAP", theta_map)
    _print_chi2(J, "median", theta_med)
    print(f"[plot] figures -> {'  '.join(str(d) for d in dests)}", flush=True)


if __name__ == "__main__":
    main()
