r"""Pedagogical figures for the sensitivity / Fisher-forecast documentation page.

Tells one story in four figures:

1. ``derivative_cascade`` — the derivative of the pipeline, **step by step**: how a
   logarithmic response to a cosmological parameter propagates through
   P(k) → σ(M) → dn/dM → the galaxy power spectra → w_p / ΔΣ (the chain rule made
   visible, all via ``jax`` autodiff).
2. ``response_functions`` — the Jacobian rows ∂ln(observable)/∂θ vs scale for the
   five summary statistics: cosmological response **grows towards small scales**,
   and each statistic responds to a different set of parameters.
3. ``degeneracy_breaking`` — 1σ Fisher ellipses at a fixed small-scale cut,
   showing how adding an X-ray and/or SZ statistic **breaks** the
   cosmology↔galaxy-halo-connection degeneracy that limits clustering+lensing.
4. ``sigma_vs_scale`` — the thesis plot: σ(σ8), σ(Ω_m) vs the scale cut R_min for
   growing probe sets — the cosmological sensitivity improves towards small scales
   **only to the fuller extent that an X-ray and/or SZ statistic is included**.

Reads the Jacobian produced by ``run_sensitivity_study`` (``fisher_study.npz``)
if present, else recomputes it.  Writes PNGs into ``docs/_images/`` with the
``sensitivity_fisher__`` prefix.

Usage::

    JAX_PLATFORMS=cpu HOD_MOD_RESULTS=/path python \
      -m hod_mod.scripts.forecasts.make_sensitivity_figures
"""

from __future__ import annotations

import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES, OBSERVABLES, _IDX  # noqa: E402
from hod_mod.forecast import params, fisher  # noqa: E402

_PFX = "sensitivity_fisher__"
_PROBE_LABEL = {"wp": r"$w_p$", "ds": r"$\Delta\Sigma$", "cl_gX": r"$C_\ell^{gX}$",
                "cl_gy": r"$C_\ell^{gy}$", "cl_XX": r"$C_\ell^{XX}$"}


def _docs_images():
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    for _ in range(6):
        if os.path.isdir(os.path.join(root, "docs")):
            break
        root = os.path.dirname(root)
    d = os.path.join(root, "docs", "_images")
    os.makedirs(d, exist_ok=True)
    return d


def _load_or_compute(model, fid):
    """Return (d0, J, row_obs, row_x) — from the study npz if available."""
    try:
        from hod_mod import paths
        npz = os.fspath(paths.results_root() / "sensitivity_fisher" / "fisher_study.npz")
    except Exception:
        npz = os.path.join(os.environ.get("HOD_MOD_RESULTS", "."),
                           "sensitivity_fisher", "fisher_study.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        if list(d["param_names"]) == PARAM_NAMES:
            print(f"[fig] using Jacobian from {npz}")
            return np.asarray(d["d0"]), np.asarray(d["jacobian"]), \
                np.asarray(d["row_obs"]), np.asarray(d["row_x"])
    print("[fig] computing Jacobian fresh ...")
    f, row_obs, row_x = model.full_data_vector_fn()
    d0, J = fisher.jacobian(f, fid)
    return np.asarray(d0), np.asarray(J), row_obs, row_x


# ---------------------------------------------------------------------------
# Figure 1 — the derivative cascade (chain rule through the pipeline)
# ---------------------------------------------------------------------------

def fig_derivative_cascade(model, fid, d0, J, row_obs, row_x, out):
    th = jnp.asarray(fid)
    z = model.z_eff
    k = np.asarray(model.k)
    m = np.asarray(model.m)
    cosmo_ps = ["Omega_m", "sigma8", "h", "n_s", "Omega_b"]
    colors = {"Omega_m": "C0", "sigma8": "C3", "h": "C2", "n_s": "C4", "Omega_b": "C1"}
    lab = {"Omega_m": r"$\Omega_m$", "sigma8": r"$\sigma_8$", "h": r"$h$",
           "n_s": r"$n_s$", "Omega_b": r"$\Omega_b$"}

    # --- upstream pipeline stages (fresh autodiff): P(k) → σ(M) → dn/dM ---
    def dln_dln(fn):
        val = np.asarray(fn(th)); Jf = np.asarray(jax.jacfwd(fn)(th))
        return {p: Jf[:, PARAM_NAMES.index(p)] * fid[PARAM_NAMES.index(p)]
                / np.where(np.abs(val) > 0, val, np.inf) for p in cosmo_ps}
    dP = dln_dln(lambda t: model._pk.pk_linear(model.k, z, model._cosmo(t)))
    dS = dln_dln(lambda t: model._hmf.sigma(model.m, z, model._cosmo(t)))
    dN = dln_dln(lambda t: model._hmf.dndm(model.m, z, model._cosmo(t)))

    # --- observable log-derivatives from the saved full Jacobian ---
    def obs_resp(name):
        mrow = row_obs == name
        dv = d0[mrow]
        return row_x[mrow], {p: J[mrow, _IDX[p]] * fid[_IDX[p]]
                             / np.where(np.abs(dv) > 0, dv, np.inf) for p in cosmo_ps}

    # all twelve summary statistics (name, title, xlabel, logx)
    OBS = [("wp", r"$w_p(r_p)$", r"$r_p\,[\mathrm{Mpc}/h]$", True),
           ("ds", r"$\Delta\Sigma(R)$", r"$R\,[\mathrm{Mpc}/h]$", True),
           ("cl_gX", r"$C_\ell^{gX}$ (galaxy×X-ray)", r"$\ell$", True),
           ("cl_gy", r"$C_\ell^{gy}$ (galaxy×tSZ)", r"$\ell$", True),
           ("cl_XX", r"$C_\ell^{XX}$ (X-ray auto)", r"$\ell$", True),
           ("cl_kk", r"$C_\ell^{\kappa\kappa}$ (cosmic shear)", r"$\ell$", True),
           ("cl_kCMB", r"$C_\ell^{\kappa_c\kappa_c}$ (CMB lensing)", r"$\ell$", True),
           ("cl_gkCMB", r"$C_\ell^{g\kappa_c}$ (galaxy×CMB lens)", r"$\ell$", True),
           ("cl_shear_kCMB", r"$C_\ell^{\kappa\kappa_c}$ (shear×CMB lens)", r"$\ell$", True),
           ("xlf", r"$\Phi(L_X)$ (AGN XLF)", r"$\log_{10}L_X$", False),
           ("smf", r"$\Phi(M_*)$ (stellar-mass fn)", r"$\log_{10}M_*$", False),
           ("n_gal", r"$n_{\rm gal}$ (number density)", "", None)]

    fig, ax = plt.subplots(5, 3, figsize=(13.5, 16))
    axf = ax.ravel()

    def _panel(a, x, dd, xlabel, title, logx=True):
        for p in cosmo_ps:
            a.plot(x, dd[p], color=colors[p], label=lab[p])
        if logx:
            a.set_xscale("log")
        a.axhline(0, color="0.7", lw=0.7)
        a.set_xlabel(xlabel); a.set_title(title, fontsize=10.5)
        a.set_ylabel(r"$\partial\ln O/\partial\ln p$")

    _panel(axf[0], k, dP, r"$k\;[h/\mathrm{Mpc}]$", r"pipeline 1 — linear $P(k)$")
    _panel(axf[1], m, dS, r"$M\;[M_\odot/h]$", r"pipeline 2 — variance $\sigma(M)$")
    _panel(axf[2], m, dN, r"$M\;[M_\odot/h]$", r"pipeline 3 — mass function $dn/dM$")
    axf[0].legend(fontsize=10, loc="best")

    for a, (name, title, xlabel, logx) in zip(axf[3:], OBS):
        x, dd = obs_resp(name)
        if name == "n_gal":                        # abundance: a single number
            vals = [float(dd[p][0]) for p in cosmo_ps]
            a.bar(range(len(cosmo_ps)), vals, color=[colors[p] for p in cosmo_ps])
            a.set_xticks(range(len(cosmo_ps)))
            a.set_xticklabels([lab[p] for p in cosmo_ps], rotation=90, fontsize=8)
            a.axhline(0, color="0.7", lw=0.7)
            a.set_title(title, fontsize=10.5); a.set_ylabel(r"$\partial\ln O/\partial\ln p$")
        else:
            s = np.argsort(x)
            _panel(a, x[s], {p: dd[p][s] for p in cosmo_ps}, xlabel, title, logx=logx)

    fig.suptitle("Derivative of the pipeline, step by step: response to all five cosmological "
                 "parameters, from $P(k)$ through the twelve summary statistics", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(out, _PFX + "derivative_cascade.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)


# ---------------------------------------------------------------------------
# Figure 2 — response functions (Jacobian rows) per observable
# ---------------------------------------------------------------------------

def fig_response_functions(model, fid, d0, J, row_obs, row_x, out):
    show = ["Omega_m", "sigma8", "fc", "lg_m1h", "bsat", "lx_norm", "p2",
            "log10DC", "log10_M_pivot", "agn_log10_ferdf"]
    col = {p: f"C{i}" for i, p in enumerate(show)}
    TITLE = {"wp": r"$w_p$", "ds": r"$\Delta\Sigma$", "cl_gX": r"$C_\ell^{gX}$",
             "cl_gy": r"$C_\ell^{gy}$", "cl_XX": r"$C_\ell^{XX}$",
             "cl_kk": r"$C_\ell^{\kappa\kappa}$", "cl_kCMB": r"$C_\ell^{\kappa_c\kappa_c}$",
             "cl_gkCMB": r"$C_\ell^{g\kappa_c}$", "cl_shear_kCMB": r"$C_\ell^{\kappa\kappa_c}$",
             "xlf": r"$\Phi(L_X)$", "smf": r"$\Phi(M_*)$", "n_gal": r"$n_{\rm gal}$"}
    obs_list = ["wp", "ds", "cl_gX", "cl_gy", "cl_XX", "cl_kk", "cl_kCMB",
                "cl_gkCMB", "cl_shear_kCMB", "xlf", "smf", "n_gal"]

    def resp_of(mrow, p):
        i = PARAM_NAMES.index(p); dv = d0[mrow]
        return J[mrow, i] * fid[i] / np.where(np.abs(dv) > 0, dv, np.inf)

    fig, axes = plt.subplots(2, 6, figsize=(19, 8.8))
    for ax, o in zip(axes.ravel(), obs_list):
        mrow = (row_obs == o)
        x = row_x[mrow]
        if o == "n_gal":                            # abundance: one number per param
            names = [p for p in show if not np.allclose(resp_of(mrow, p), 0)]
            ax.bar(range(len(names)), [float(resp_of(mrow, p)[0]) for p in names],
                   color=[col[p] for p in names])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([params.PARAM_LATEX[p] for p in names], rotation=90, fontsize=7)
        else:
            s = np.argsort(x)
            for p in show:
                resp = resp_of(mrow, p)
                if np.allclose(resp, 0):
                    continue
                ax.plot(x[s], resp[s], col[p], label=params.PARAM_LATEX[p])
            if o not in ("xlf", "smf"):            # xlf/smf abscissa is already log10
                ax.set_xscale("log")
            ax.set_xlabel(r"$r_p\,[\mathrm{Mpc}/h]$" if o in ("wp", "ds")
                          else (r"$\log_{10}L_X$" if o == "xlf"
                                else (r"$\log_{10}M_*$" if o == "smf" else r"$\ell$")))
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_title(TITLE[o], fontsize=12)
    for a in (axes[0, 0], axes[1, 0]):
        a.set_ylabel(r"$\partial\ln O/\partial\ln\theta$")
    # complete legend: one entry per parameter (proxy handles), so every colour is labelled
    handles = [Line2D([0], [0], color=col[p], lw=2, label=params.PARAM_LATEX[p]) for p in show]
    fig.legend(handles=handles, fontsize=10, ncol=len(show),
               loc="lower center", bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Response functions ∂ln(observable)/∂ln(θ) for all twelve summary statistics: "
                 "the cosmological response grows towards small scales; each statistic carries "
                 "different (astro)physical information", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.955])
    p = os.path.join(out, _PFX + "response_functions.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)


# ---------------------------------------------------------------------------
# Figure 3 — degeneracy-breaking ellipses
# ---------------------------------------------------------------------------

def _ellipse(ax, cov, i, j, cx, cy, color, label, ls="-"):
    sub = cov[np.ix_([i, j], [i, j])]
    if not np.all(np.isfinite(sub)):
        return
    w, V = np.linalg.eigh(sub)
    w = np.clip(w, 0, None)
    ang = np.degrees(np.arctan2(V[1, np.argmax(w)], V[0, np.argmax(w)]))
    a, b = 2 * np.sqrt(2.30 * w[np.argmax(w)]), 2 * np.sqrt(2.30 * w[np.argmin(w)])
    e = Ellipse((cx, cy), a, b, angle=ang, fill=False, edgecolor=color, lw=2, ls=ls, label=label)
    ax.add_patch(e)


def fig_degeneracy_breaking(model, fid, d0, J, row_obs, row_x, out, rmin=0.1, rel_err=0.01):
    # pairs showing the cosmology↔(HOD, baryon) degeneracies broken by X-ray/SZ
    pairs = [("sigma8", "Omega_m"), ("sigma8", "log10_M_pivot"), ("sigma8", "bsat")]
    sets = [("wp+ds", ["wp", "ds"], "C0", "-"),
            ("+ SZ", ["wp", "ds", "cl_gy"], "C2", "--"),
            ("+ X-ray", ["wp", "ds", "cl_gX", "cl_XX"], "C1", "-."),
            ("all five", ["wp", "ds", "cl_gX", "cl_gy", "cl_XX"], "C3", "-")]
    prior = params.regularizing_prior(add_planck=False)   # bounds flat nuisances
    smask0 = model.scale_cut_mask(row_obs, row_x, rmin)
    covs = {}
    for label, obs, _, _ in sets:
        m = smask0 & np.isin(row_obs, obs)
        F = fisher.fisher_matrix(d0[m], J[m], rel_err, prior)
        covs[label], _, _ = fisher.constraints(F, ridge=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, (pa, pb) in zip(axes, pairs):
        ia, ib = PARAM_NAMES.index(pa), PARAM_NAMES.index(pb)
        cx, cy = fid[ia], fid[ib]
        for label, obs, color, ls in sets:
            _ellipse(ax, covs[label], ia, ib, cx, cy, color, label, ls)
        ax.plot(cx, cy, "k+", ms=9)
        ax.set_xlabel(params.PARAM_LATEX[pa]); ax.set_ylabel(params.PARAM_LATEX[pb])
        ax.set_title(f"{params.PARAM_LATEX[pa]} – {params.PARAM_LATEX[pb]}")
        ax.relim(); ax.autoscale_view()
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(rf"Degeneracy breaking at $R_{{\min}}={rmin}$ Mpc/h "
                 rf"(1σ, {rel_err*100:g}% errors): X-ray / SZ cross-statistics shrink and "
                 rf"de-rotate the cosmology↔galaxy-halo-connection ellipses", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out, _PFX + "degeneracy_breaking.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)


# ---------------------------------------------------------------------------
# Figure 4 — the thesis plot: sigma(cosmology) vs scale cut, per probe set
# ---------------------------------------------------------------------------

def fig_sigma_vs_scale(model, fid, d0, J, row_obs, row_x, out,
                       rmins=(0.1, 0.5, 2.5, 10.0), rel_err=0.01):
    # A cumulative data vector that grows to *all twelve* summary statistics.
    s_cl = ["wp", "ds", "cl_gy", "cl_gX", "cl_XX"]                       # + SZ + X-ray
    s_le = s_cl + ["cl_kk", "cl_kCMB", "cl_gkCMB", "cl_shear_kCMB"]      # + all lensing spectra
    sets = [("wp+ΔΣ (clustering+GG-lensing)", ["wp", "ds"], "C0", "o"),
            (r"+ SZ + X-ray ($C_\ell^{gy},C_\ell^{gX},C_\ell^{XX}$)", s_cl, "C1", "^"),
            (r"+ cosmic + CMB lensing ($C_\ell^{\kappa\kappa}$ + CMB-$\kappa$)", s_le, "C2", "s"),
            (r"all twelve (+ $n_{\rm gal}$, SMF, XLF)", list(OBSERVABLES), "C3", "D")]
    prior = params.regularizing_prior(add_planck=False)
    sig_label = {"sigma8": r"$\sigma(\sigma_8)$", "Omega_m": r"$\sigma(\Omega_m)$"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, tgt in zip(axes, ["sigma8", "Omega_m"]):
        j = PARAM_NAMES.index(tgt)
        for label, obs, color, mk in sets:
            ys = []
            for rm in rmins:
                m = model.scale_cut_mask(row_obs, row_x, rm) & np.isin(row_obs, obs)
                F = fisher.fisher_matrix(d0[m], J[m], rel_err, prior)
                _, sig, _ = fisher.constraints(F, ridge=0.0)
                ys.append(sig[j])
            gain = ys[-1] / ys[0]
            ax.loglog(rmins, ys, mk + "-", color=color, label=f"{label}  (×{gain:.1f})")
        ax.set_xlabel(r"scale cut  $R_{\min}\;[\mathrm{Mpc}/h]$  (use scales $>R_{\min}$)")
        ax.set_ylabel(sig_label[tgt])
        ax.set_title(params.PARAM_LATEX[tgt])
        ax.invert_xaxis()   # small scales (small R_min) to the right
    axes[0].legend(fontsize=8.5, title="cumulative data vector  (gain 10→0.1)")
    fig.suptitle("Cosmological sensitivity vs scale cut, data vector grown to all twelve "
                 "summary statistics (1% errors, no external prior)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, _PFX + "sigma_vs_scale.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)


# ---------------------------------------------------------------------------
# Figure 5 — baryonic feedback contaminating ΔΣ
# ---------------------------------------------------------------------------

def fig_baryon_suppression(model, fid, out):
    """Fractional ΔΣ change from the hot-gas feedback, vs feedback strength."""
    import jax.numpy as jnp
    i_mp = PARAM_NAMES.index("log10_M_pivot")
    fmp = float(fid[i_mp])

    def ds(mpiv):
        t = fid.copy(); t[i_mp] = mpiv
        return np.asarray(model.predict(jnp.asarray(t), ["ds"])["ds"])

    R = np.asarray(model.rp_ds)
    ref = ds(9.0)   # no-feedback reference (gas at cosmic fraction everywhere)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for mp, c in zip([13.0, 13.5, 14.0, 14.5], ["C0", "C1", "C2", "C3"]):
        ax.semilogx(R, 100 * (ds(mp) / ref - 1.0), "o-", color=c, ms=3,
                    label=rf"$\log_{{10}}M_{{\rm pivot}}={mp}$"
                          + ("  (fid)" if abs(mp - fmp) < 0.01 else ""))
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xlabel(r"$R\;[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$\Delta\Sigma$ change vs no feedback  [%]")
    ax.set_title("Baryonic feedback contaminates small-scale lensing")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, _PFX + "baryon_suppression.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)


# ---------------------------------------------------------------------------
# Figure 6 — X-ray/SZ recover the small-scale σ8 lost to baryon marginalisation
# ---------------------------------------------------------------------------

def fig_baryon_recovery(model, fid, d0, J, row_obs, row_x, out,
                        rmins=(0.1, 0.5, 2.5, 10.0), rel_err=0.01):
    js8 = PARAM_NAMES.index("sigma8")
    pri_fix = params.regularizing_prior(fix=("log10_M_pivot", "log10_eta_min"))
    pri_marg = params.regularizing_prior()

    def s8(obs, rm, prior):
        m = model.scale_cut_mask(row_obs, row_x, rm) & np.isin(row_obs, obs)
        F = fisher.fisher_matrix(d0[m], J[m], rel_err, prior)
        _, sig, _ = fisher.constraints(F, ridge=0.0)
        return sig[js8]

    wpds = ["wp", "ds"]; allf = ["wp", "ds", "cl_gX", "cl_gy", "cl_XX"]
    y_fix = [s8(wpds, rm, pri_fix) for rm in rmins]
    y_marg = [s8(wpds, rm, pri_marg) for rm in rmins]
    y_rec = [s8(allf, rm, pri_marg) for rm in rmins]
    y_all = [s8(list(OBSERVABLES), rm, pri_marg) for rm in rmins]      # full data vector
    y_all_fix = [s8(list(OBSERVABLES), rm, pri_fix) for rm in rmins]   # full vector, baryons known

    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    ax.loglog(rmins, y_fix, "s--", color="0.4", label="wp+ΔΣ, baryons known (ideal)")
    ax.loglog(rmins, y_marg, "o-", color="C3", label="wp+ΔΣ, baryons marginalised")
    ax.loglog(rmins, y_rec, "^-", color="C0", label="wp+ΔΣ + X-ray + SZ, baryons marg.")
    ax.loglog(rmins, y_all, "D-", color="C2", label="all twelve statistics, baryons marg.")
    ax.fill_between(rmins, y_fix, y_marg, color="C3", alpha=0.12)
    ax.annotate("baryon\ncontamination", (rmins[0], np.sqrt(y_fix[0] * y_marg[0])),
                fontsize=8, color="C3", ha="left", va="center")
    ax.set_xlabel(r"scale cut  $R_{\min}\;[\mathrm{Mpc}/h]$  (use scales $>R_{\min}$)")
    ax.set_ylabel(r"$\sigma(\sigma_8)$")
    ax.set_title("Baryons limit small-scale lensing;\nthe full data vector self-calibrates them")
    ax.invert_xaxis()
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(out, _PFX + "baryon_recovery.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)
    # numbers for the doc text
    print(f"[baryon] Rmin=0.1: sigma8 fixed={y_fix[0]:.3e} marg={y_marg[0]:.3e} "
          f"recovered={y_rec[0]:.3e}  degrade x{y_marg[0]/y_fix[0]:.2f}  recover x{y_marg[0]/y_rec[0]:.2f}")
    print(f"[baryon] Rmin=0.1 all-twelve: known={y_all_fix[0]:.3e} marg={y_all[0]:.3e} "
          f"baryon-degrade x{y_all[0]/y_all_fix[0]:.3f}  (below wp+ΔΣ ideal x{y_fix[0]/y_all[0]:.2f})")


# ---------------------------------------------------------------------------
# Figure 7 — cosmic shear: baryons bite harder than in ΔΣ
# ---------------------------------------------------------------------------

def fig_shear_vs_ds_baryons(model, fid, out):
    import jax.numpy as jnp
    i_mp = _IDX["log10_M_pivot"]

    def ds(mp):
        t = fid.copy(); t[i_mp] = mp
        return np.asarray(model.predict(jnp.asarray(t), ["ds"])["ds"])

    def kk(mp):
        t = fid.copy(); t[i_mp] = mp
        return np.asarray(model.predict(jnp.asarray(t), ["cl_kk"])["cl_kk"])

    R = np.asarray(model.rp_ds); ell = np.asarray(model.ell)
    ds0, kk0 = ds(9.0), kk(9.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mp, c in zip([13.5, 14.0, 14.5], ["C1", "C2", "C3"]):
        ax[0].semilogx(R, 100 * (ds(mp) / ds0 - 1), "o-", color=c, ms=3,
                       label=rf"$\log_{{10}}M_{{\rm pivot}}={mp}$")
        ax[1].semilogx(ell, 100 * (kk(mp) / kk0 - 1), "o-", color=c, ms=3)
    for a in ax:
        a.axhline(0, color="0.7", lw=0.7)
    ax[0].set_title(r"lensing $\Delta\Sigma(R)$"); ax[0].set_xlabel(r"$R\;[\mathrm{Mpc}/h]$")
    ax[0].set_ylabel("baryon change [%]"); ax[0].legend(fontsize=8)
    ax[1].set_title(r"cosmic shear $C_\ell^{\kappa\kappa}$"); ax[1].set_xlabel(r"$\ell$")
    fig.suptitle("Baryonic-feedback imprint on the two lensing probes: ΔΣ at small R, "
                 "cosmic shear across the high-ℓ range that carries its σ₈ signal", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, _PFX + "shear_vs_ds_baryons.png")
    fig.savefig(p, dpi=120); plt.close(fig); print("[fig]", p)


# ---------------------------------------------------------------------------
# Figure 8 — energetic closure of the baryon budget (two channels)
# ---------------------------------------------------------------------------

def fig_energy_closure(fid, out):
    import jax.numpy as jnp
    import hod_mod.forecast.forward_jax as FJ
    me = ForwardModel(n_k=64, n_m=220, energy_closure=True)
    c = me._cosmo(jnp.asarray(fid)); Om = float(c["Omega_m"]); Ob = float(c["Omega_b"]); fbc = Ob / Om
    M = np.asarray(me.m); z = me.z_eff
    ez2 = Om * (1 + z) ** 3 + (1 - Om); rho = 2.775e11 * ez2 / (1 + z) ** 3
    r200 = (3 * M / (4 * np.pi * 200 * rho)) ** (1 / 3.); v2 = FJ._G_KMS2 * M / r200
    from hod_mod.observables.baryon_fraction import BaryonFractionSigmoid
    fb_sig = np.asarray(BaryonFractionSigmoid()(jnp.asarray(M), c,
                        {"log10_M_pivot": 13.5, "beta_b": 1.5, "f_b_min": 0.01}))
    E_disp = (fbc - fb_sig) * M * v2
    dc = 10 ** fid[_IDX["log10DC"]]; eps = 10 ** (-0.5)
    L_on = 10 ** FJ._LX_ON_NORM * (M / 1e13) ** FJ._LX_ON_SLOPE
    E_agn = eps * dc * FJ._K_BOL * L_on * FJ._T_HUBBLE_S / FJ._ERG_PER_MSUN_KMS2
    from hod_mod.connection.hod.zumandelbaum15 import _mstar_from_mh_zu15
    Mstar = 10 ** np.asarray(_mstar_from_mh_zu15(jnp.log10(jnp.asarray(M)),
                             11.677, 10.477, 0.735, 0.662, 0.540))
    E_sn = FJ._EPS_SN * Mstar * FJ._E_SN_PER_MSUN
    good = fbc - fb_sig > 1e-3
    te = fid.copy(); te[_IDX["log10_M_pivot"]] = -0.5
    fb_en = np.asarray(me._fb_energy(jnp.asarray(te), c))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(M[good], (E_agn / E_disp)[good], "C3-", lw=2, label="AGN (×DC, L_X)")
    ax[0].loglog(M[good], (E_sn / E_disp)[good], "C0-", lw=2, label="stellar / SN")
    ax[0].loglog(M[good], ((E_agn + E_sn) / E_disp)[good], "k--", lw=1.5, label="total")
    ax[0].axhline(1, color="0.6", lw=1, ls=":")
    ax[0].set_xlabel(r"$M_{200}\;[M_\odot/h]$")
    ax[0].set_ylabel(r"$E_{\rm avail}/E_{\rm displace}$  (closure ratio)")
    ax[0].set_title("Feedback energy vs energy to expel the missing baryons")
    ax[0].legend(fontsize=9)
    ax[1].semilogx(M, fb_sig, "C1-", lw=2, label="phenomenological (sigmoid)")
    ax[1].semilogx(M, fb_en, "C3--", lw=2, label=r"energy-derived ($\varepsilon_{\rm couple}$, DC)")
    ax[1].axhline(fbc, color="0.6", lw=1, ls=":"); ax[1].text(2e10, fbc * 1.02, "cosmic", fontsize=8)
    ax[1].set_xlabel(r"$M_{200}\;[M_\odot/h]$"); ax[1].set_ylabel(r"$f_b(M)$")
    ax[1].set_title("Baryon fraction: fit vs predicted from AGN energetics")
    ax[1].legend(fontsize=9)
    fig.suptitle("Closing the loop: the AGN population has ample energy (×few–100) to "
                 "displace the missing baryons; SN dominates at low mass", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, _PFX + "energy_closure.png")
    fig.savefig(p, dpi=120); plt.close(fig); print("[fig]", p)


def fig_xlf_cosmology(model, fid, d0, J, row_obs, row_x, out,
                      rmins=(0.1, 0.5, 2.5, 10.0), rel_err=0.01):
    r"""Does the Powell AGN X-ray luminosity function tighten (Ω_m, σ_8)?

    Left: the XLF log-response ∂lnΦ/∂lnθ vs L_X — the cosmology signal
    (Ω_m, σ_8) is L_X-dependent, but the active fraction f_ERDF is a *flat*
    amplitude, so the XLF's cosmology amplitude is degenerate with it.  Middle:
    σ(Ω_m) improvement factor from adding the XLF, vs scale cut, for three
    external-knowledge scenarios of the AGN accretion sector.  Right: the
    (Ω_m, σ_8) 1σ ellipse at R_min=10 Mpc/h with vs without the XLF when the
    AGN sector is externally pinned.
    """
    row_obs = np.asarray(row_obs)
    is_xlf = row_obs == "xlf"
    iom, is8 = _IDX["Omega_m"], _IDX["sigma8"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # --- panel 1: XLF response functions ∂lnΦ/∂lnθ vs logLX ---
    lx = np.asarray(model.grid_of("xlf"))
    phi0 = d0[is_xlf]
    Jx = J[is_xlf]                                    # (Nlx, Nparam)
    ax = axes[0]
    # d Φ / d θ  normalised to the fiducial Φ (so all curves are O(1) and the
    # amplitude-vs-shape contrast is legible; μ_BH's huge response is off-scale).
    for nm, col in [("Omega_m", "C3"), ("sigma8", "C0"),
                    ("agn_log10_ferdf", "0.4"), ("agn_delta2", "C1")]:
        dln = Jx[:, _IDX[nm]] / phi0                  # d lnΦ / d θ
        ls = "--" if nm == "agn_log10_ferdf" else "-"
        ax.plot(lx, dln, ls, color=col, marker="o", ms=3, label=params.PARAM_LATEX[nm])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\log_{10} L_X\;[\mathrm{erg/s}]$")
    ax.set_ylabel(r"$\partial\ln\Phi/\partial\theta$")
    ax.set_title("XLF response: cosmology is $L_X$-shaped,\n$f_{\\rm ERDF}$ is flat (degenerate amplitude)")
    ax.legend(fontsize=8.5, ncol=2)

    # --- panel 2: sigma(Om) gain from +XLF vs Rmin, 3 AGN-prior scenarios ---
    scen = [("AGN sector free", params.regularizing_prior(add_planck=False), "C0", "o"),
            (r"$f_{\rm ERDF}$ pinned",
             params.regularizing_prior(add_planck=False, fix=("agn_log10_ferdf",)), "C2", "s"),
            ("AGN sector pinned",
             params.regularizing_prior(add_planck=False, fix=(
                 "agn_mu_bh", "agn_al_bh", "agn_sig_bh", "agn_log10_lstar",
                 "agn_delta1", "agn_delta2", "agn_log10_ferdf")), "C3", "^")]
    ax = axes[1]
    covs_pin = None
    for label, prior, col, mk in scen:
        gains = []
        for rm in rmins:
            sm = model.scale_cut_mask(row_obs, row_x, rm)
            cN, sN, _ = fisher.masked_constraints(d0, J, sm & ~is_xlf, rel_err, prior)
            cY, sY, _ = fisher.masked_constraints(d0, J, sm, rel_err, prior)
            gains.append(sN[iom] / sY[iom])
            if label == "AGN sector pinned" and rm == max(rmins):
                covs_pin = (cN, cY)
        ax.plot(rmins, gains, mk + "-", color=col, label=label)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"scale cut $R_{\min}\;[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$\sigma(\Omega_m)$ gain from $+$XLF")
    ax.set_title(r"XLF tightens $\Omega_m$ only when the AGN"
                 "\naccretion sector is externally known")
    ax.legend(fontsize=8.5, loc="best")

    # --- panel 3: (Om, s8) ellipse w/ vs w/o XLF, AGN pinned, Rmin=10 ---
    ax = axes[2]
    if covs_pin is not None:
        cN, cY = covs_pin
        cx, cy = fid[iom], fid[is8]
        _ellipse(ax, cN, iom, is8, cx, cy, "0.5", "6 obs (no XLF)", ls="--")
        _ellipse(ax, cY, iom, is8, cx, cy, "C3", "+ XLF", ls="-")
        ax.plot(cx, cy, "k+", ms=9)
        ax.set_xlabel(params.PARAM_LATEX["Omega_m"]); ax.set_ylabel(params.PARAM_LATEX["sigma8"])
        ax.set_title(r"$(\Omega_m,\sigma_8)$ at $R_{\min}=10$ Mpc/h"
                     "\n(AGN sector pinned, 1% errors)")
        ax.legend(fontsize=9); ax.relim(); ax.autoscale_view()

    fig.suptitle("The AGN X-ray luminosity function as a cosmology probe: its amplitude carries "
                 r"$dn/dM(\Omega_m,\sigma_8)$ but is degenerate with the active fraction", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, _PFX + "xlf_cosmology.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("[fig]", p)


def main():
    out = _docs_images()
    fid = params.fiducial_vector()
    model = ForwardModel(n_k=256, n_m=256)
    d0, J, row_obs, row_x = _load_or_compute(model, fid)
    fig_derivative_cascade(model, fid, d0, J, row_obs, row_x, out)
    fig_response_functions(model, fid, d0, J, row_obs, row_x, out)
    fig_degeneracy_breaking(model, fid, d0, J, row_obs, row_x, out)
    fig_sigma_vs_scale(model, fid, d0, J, row_obs, row_x, out)
    fig_baryon_suppression(model, fid, out)
    fig_baryon_recovery(model, fid, d0, J, row_obs, row_x, out)
    fig_shear_vs_ds_baryons(model, fid, out)
    fig_energy_closure(fid, out)
    fig_xlf_cosmology(model, fid, d0, J, row_obs, row_x, out)
    print("[done] figures in", out)


if __name__ == "__main__":
    main()
