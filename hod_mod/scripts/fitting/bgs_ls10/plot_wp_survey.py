"""Summary figure and table for the BGS M* > 10^10 wp(rp) model survey.

Reads all map_result.json files under results/bgs_multiprobe/mstar10.0_wp_*/,
rebuilds each HOD predictor at its MAP parameters, and produces two figures:

  results/showcase/fig_wp_survey_predictions.pdf  — wp(rp) data + all models
  results/showcase/fig_wp_survey_chi2.pdf         — chi²/ndof heatmap

Usage
-----
    python scripts/fitting/bgs_ls10/plot_wp_survey.py

Requires the sum_stat data file to be accessible (reads rp, wp, err from the
map_result.json data_file field).
"""

import os
import json
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import jax.numpy as jnp

# sys.path.insert removed — hod_mod is installed

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.cosmology.halo_mass_function import make_hmf
from hod_mod.cosmology.halo_profiles import HaloProfile
from hod_mod.galaxies.clustering import FullHaloModelPrediction
from hod_mod.galaxies.baryon_fraction import BaryonFractionSigmoid
from hod_mod.data_io.sum_stat_reader import SumStatReader
from hod_mod.galaxies.hod import (
    _mstar_from_mh_zu15,
    shmr_zacharegkas25,
    _mean_stellar_mass_c_vanuitert16,
)

# Import the fitting helpers
_FIT_SCRIPT = os.path.join(os.path.dirname(__file__), "fit_bgs_multiprobe.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("fit_bgs_multiprobe", _FIT_SCRIPT)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MultiProbeFitter = _mod.MultiProbeFitter
_build_theta_cosmo = _mod._build_theta_cosmo
_build_hod_params  = _mod._build_hod_params
BGS_BINS           = _mod.BGS_BINS
_find_data_file    = _mod._find_data_file
HOD_REGISTRY       = _mod.HOD_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "results", "bgs_multiprobe")
RESULT_ROOT = os.path.abspath(RESULT_ROOT)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "results", "showcase")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

SUM_STAT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                 "sum_stat", "data"))

MODEL_COLOR = {
    "more2015":        "#1f77b4",
    "zheng2007":       "#ff7f0e",
    "aum":             "#2ca02c",
    "zu_mandelbaum15": "#d62728",
    "vanuitert16":     "#8c564b",
    "zacharegkas25":   "#e377c2",
    "sext":            "#9467bd",
}
MODEL_LABEL = {
    "more2015":        "More+2015",
    "zheng2007":       "Zheng+2007",
    "aum":             "Kravtsov+2004",
    "zu_mandelbaum15": "Zu & Mandelbaum 2015",
    "vanuitert16":     "van Uitert+2016",
    "zacharegkas25":   "Zacharegkas+2025",
    "sext":            "More+2015 + sat ext.",
}
RP_MINS = [0.30, 0.05, 0.04, 0.02, 0.01]
RP_TAGS = {0.30: "rp300", 0.05: "rp050", 0.04: "rp040", 0.02: "rp020", 0.01: "rp010"}
MODELS  = ["more2015", "zheng2007", "aum", "zu_mandelbaum15", "vanuitert16", "zacharegkas25"]

# Colours for per-model overlay: large scale (green) → small scale (red)
RP_MIN_COLORS = {
    0.30: "#2ca02c",
    0.05: "#17becf",
    0.04: "#1f77b4",
    0.02: "#ff7f0e",
    0.01: "#d62728",
}

PARAM_LABEL = {
    "log10mmin":        r"$\log_{10}M_{\rm min}$",
    "sigma_logm":       r"$\sigma_{\log m}$",
    "log10m0":          r"$\log_{10}M_0$",
    "log10m1":          r"$\log_{10}M_1$",
    "alpha":            r"$\alpha$",
    "kappa":            r"$\kappa$",
    "alpha_inc":        r"$\alpha_{\rm inc}$",
    "lg_m1h":           r"$\log_{10}M_{1h}$",
    "lg_m0star":        r"$\log_{10}M_{*0}$",
    "beta":             r"$\beta$",
    "sigma_lnmstar":    r"$\sigma_{\ln M_*}$",
    "bsat":             r"$B_{\rm sat}$",
    "alpha_sat":        r"$\alpha_{\rm sat}$",
    "log10m_h1":        r"$\log_{10}M_{h1}$",
    "log10m_star0":     r"$\log_{10}M_{*0}$",
    "beta1":            r"$\beta_1$",
    "log10_beta2":      r"$\log_{10}\beta_2$",
    "sigma_c":          r"$\sigma_c$",
    "b0":               r"$b_0$",
    "b1":               r"$b_1$",
    "log10m1_shmr":     r"$\log_{10}M_1^{\rm SHMR}$",
    "log10eps":         r"$\log_{10}\varepsilon$",
    "alpha_shmr":       r"$\alpha_{\rm SHMR}$",
    "sigma_logm_star":  r"$\sigma_{\log M_*}$",
    "B_sat":            r"$B_{\rm sat}$",
    "B_cut":            r"$B_{\rm cut}$",
    "A_IA":             r"$A_{\rm IA}$",
    "f_off":            r"$f_{\rm off}$",
    "sigma_off":        r"$\sigma_{\rm off}$",
    "log10_M_pivot":    r"$\log_{10}M_{\rm pivot}$",
    "beta_b":           r"$\beta_b$",
    "log10_eta_min":    r"$\log_{10}\eta_{\rm min}$",
    "b_sat_conc":       r"$b_{\rm sat,conc}$",
    "f_cut":            r"$f_{\rm cut}$",
    "gamma_inner":      r"$\gamma_{\rm inner}$",
}

DOCS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "docs"))


def _rp_tag(rp_min):
    return f"rp{int(round(rp_min * 1000)):03d}"


def _load_all_results():
    """Return list of dicts, one per completed map_result.json."""
    records = []
    for d in sorted(os.listdir(RESULT_ROOT)):
        jf = os.path.join(RESULT_ROOT, d, "map_result.json")
        if not os.path.exists(jf):
            continue
        r = json.load(open(jf))
        if r.get("chi2", float("inf")) == float("inf"):
            continue                           # vanuitert16 etc.
        if "esd" in r.get("probes", ""):
            continue                           # wp-only survey
        r["_dir"] = d
        records.append(r)
    return records


class _CachedPkLinear:
    """Minimal CAMB cache shared across models."""
    def __init__(self, pk_lin_obj, n_k=512):
        self._base = pk_lin_obj
        self._k_ref = np.logspace(-4, 1.5, n_k)
        self._log_k_ref = np.log(self._k_ref)
        self._cache = {}

    def _key(self, z, theta):
        return (round(float(z), 4),
                round(float(theta["Omega_m"]), 5),
                round(float(theta["ln10^{10}A_s"]), 4),
                round(float(theta.get("h", 0.6736)), 4))

    def pk_linear(self, k, z, theta):
        key = self._key(z, theta)
        if key not in self._cache:
            pk_ref = np.asarray(self._base.pk_linear(self._k_ref, float(z), theta))
            self._cache[key] = np.log(np.maximum(pk_ref, 1e-50))
        log_k = np.log(np.asarray(k, dtype=float))
        return jnp.asarray(np.exp(np.interp(log_k, self._log_k_ref, self._cache[key])))


def _build_predictor_for(hod_model, profile, pk_cached):
    """Build a FullHaloModelPrediction for the given (hod_model, profile)."""
    pk_lin = LinearPowerSpectrum()
    cosmo  = pk_lin.default_cosmology()
    hmf    = make_hmf("csst")
    cfg    = HOD_REGISTRY[hod_model]
    if cfg.get("bias_arg", True):
        hod = cfg["class"](hmf, hmf.bias)
    else:
        hod = cfg["class"](hmf)
    hp   = HaloProfile(cosmo)
    bf   = BaryonFractionSigmoid()
    pred = FullHaloModelPrediction(
        pk_cached, hod, hp, baryon_fraction=bf,
        profile=profile, einasto_alpha=0.18,
    )
    return pred, cosmo


def predict_wp(pred, cosmo_default, params, rp, z=0.115, pi_max=100.0):
    tc = _build_theta_cosmo(params, cosmo_default)
    hp = _build_hod_params(params)
    return np.asarray(pred.wp(jnp.array(rp), pi_max, z, tc, hp))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_data(data_file):
    reader = SumStatReader.from_hdf5(data_file)
    jt = reader.joint_bgs(probes=("wp",))
    n_full  = 30
    rp_full = jt["rp_wp"]                                       # already Mpc/h
    wp_full = jt["data_vector"][:n_full]                        # already Mpc/h
    wp_err  = np.sqrt(np.diag(jt["cov"][:n_full, :n_full]))
    return rp_full, wp_full, wp_err


# ---------------------------------------------------------------------------
# Per-model figure helpers
# ---------------------------------------------------------------------------

def _plot_per_model_wp(model_key, predictions, predictions_at_data,
                       chi2_table, rp_data, wp_data, wp_err, rp_fine, out_dir):
    """wp(rp) overlay for one model: all (rp_min, profile) combinations."""
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.04},
    )
    has_any = False

    if rp_data is not None:
        ax.errorbar(rp_data, rp_data * wp_data, rp_data * wp_err,
                    fmt="o", color="k", ms=4, lw=1.2, zorder=2,
                    label="BGS LS10 data", capsize=2)
        axr.axhline(1.0, color="k", lw=1.2, zorder=5)
        axr.fill_between(rp_data,
                         1 - wp_err / wp_data, 1 + wp_err / wp_data,
                         color="k", alpha=0.12, zorder=0)

    for prof, ls in [("nfw", "-"), ("einasto", "--")]:
        key = (model_key, prof)
        if key not in predictions:
            continue
        for rp_min in RP_MINS:
            if rp_min not in predictions[key]:
                continue
            has_any = True
            color = RP_MIN_COLORS[rp_min]
            c2n   = chi2_table.get((model_key, prof, rp_min), float("nan"))
            lbl   = (f"$r_p>{rp_min:.2f}$  "
                     r"$\chi^2/n_{\rm dof}=$" + f"{c2n:.2f}") if prof == "nfw" else None
            lw    = 1.8 if prof == "nfw" else 1.2
            alpha = 0.92 if prof == "nfw" else 0.65
            ax.plot(rp_fine, rp_fine * predictions[key][rp_min],
                    ls=ls, lw=lw, alpha=alpha, color=color, label=lbl, zorder=5)
            if (rp_data is not None
                    and key in predictions_at_data
                    and rp_min in predictions_at_data[key]):
                ratio = predictions_at_data[key][rp_min] / wp_data
                axr.plot(rp_data, ratio, ls=ls, lw=lw, alpha=alpha, color=color, zorder=5)

    if not has_any:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$r_p\,w_p(r_p)\;[h^{-2}\,{\rm Mpc}^2]$", fontsize=11)
    ax.set_title(MODEL_LABEL.get(model_key, model_key), fontsize=11)
    ax.set_xlim(0.008, 60)

    nfw_h = Line2D([0], [0], ls="-",  lw=2.0, color="0.4", label="NFW")
    ein_h = Line2D([0], [0], ls="--", lw=1.5, color="0.4", label="Einasto")
    leg_s = ax.legend(handles=[nfw_h, ein_h],
                      fontsize=8, loc="lower right", framealpha=0.85)
    ax.add_artist(leg_s)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
              handlelength=1.8, labelspacing=0.5)

    axr.set_xscale("log")
    axr.set_yscale("log")
    axr.set_xlabel(r"$r_p\;[h^{-1}\,{\rm Mpc}]$", fontsize=11)
    axr.set_ylabel(r"$w_p^{\rm pred} / w_p^{\rm data}$", fontsize=9.5)
    axr.set_xlim(0.008, 60)
    axr.set_ylim(0.2, 8.0)
    axr.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axr.axhline(2.0, color="0.8", ls="--", lw=0.8)
    axr.axhline(0.5, color="0.8", ls="--", lw=0.8)

    base = os.path.join(out_dir, f"fig_wp_allcuts_{model_key}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=150)
        print(f"  Saved {base}.{ext}")
    plt.close(fig)


def _plot_per_model_params(model_key, records, out_dir):
    """Parameter-trend figure: each free param vs rp_min, NFW vs Einasto."""
    _COSMO = {"h", "Omega_m", "n_s", "ln10^{10}A_s"}

    param_data = {}   # (prof, rp_min) → {name: value}
    free_union = set()
    for r in records:
        if r.get("hod_model") != model_key or r.get("use_sat_ext", False):
            continue
        rp   = float(r.get("rp_min_wp", 0.3))
        prof = r.get("profile", "nfw")
        fps  = [p for p in r.get("free_params", []) if p not in _COSMO]
        free_union.update(fps)
        param_data[(prof, rp)] = {k: r["params"][k] for k in fps if k in r["params"]}

    if not param_data:
        return

    free_list = sorted(free_union)
    n, ncols  = len(free_list), 4
    nrows     = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 2.8))
    axes = np.array(axes).reshape(-1)
    fig.subplots_adjust(hspace=0.60, wspace=0.45)

    rp_x = sorted({rp for (_, rp) in param_data})
    color = MODEL_COLOR.get(model_key, "0.4")

    for ai, pname in enumerate(free_list):
        ax = axes[ai]
        for prof, marker, ls, fill in [
            ("nfw",     "o", "-",  "full"),
            ("einasto", "s", "--", "none"),
        ]:
            xs = [rp for rp in rp_x if (prof, rp) in param_data
                  and pname in param_data[(prof, rp)]]
            ys = [float(param_data[(prof, rp)][pname]) for rp in xs]
            if xs:
                lbl = "NFW" if prof == "nfw" else "Einasto"
                ax.plot(xs, ys, marker=marker, ls=ls, lw=1.5,
                        color=color, ms=6, fillstyle=fill, label=lbl)
        ax.set_xscale("log")
        ax.set_xticks(rp_x)
        ax.set_xticklabels([f"{rp:.2f}" for rp in rp_x],
                           fontsize=7, rotation=45)
        ax.set_xlabel(r"$r_{p,\rm min}$", fontsize=8)
        ax.set_title(PARAM_LABEL.get(pname, pname.replace("_", " ")),
                     fontsize=9)
    axes[0].legend(fontsize=7, loc="best")
    for ai in range(n, len(axes)):
        axes[ai].set_visible(False)

    fig.suptitle(
        MODEL_LABEL.get(model_key, model_key) + " — MAP parameters vs scale cut",
        fontsize=11)

    base = os.path.join(out_dir, f"fig_wp_params_{model_key}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=150)
        print(f"  Saved {base}.{ext}")
    plt.close(fig)


def _plot_shmr_comparison(records, chi2_table, out_dir):
    """Stellar-to-halo mass relation comparison for all models at rp>0.05.

    Explicit-SHMR models (zu_mandelbaum15, zacharegkas25, vanuitert16) are
    shown as continuous curves; threshold HODs (more2015, zheng2007, aum)
    as single markers at (log10mmin, 10.0).
    """
    log10m_h = np.linspace(11.0, 15.0, 300)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlabel(r"$\log_{10}(M_{\rm halo}\;[h^{-1}\,M_\odot])$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(M_*\;[h^{-1}\,M_\odot])$", fontsize=12)
    ax.set_title(
        r"Stellar–halo mass relation — MAP fits at $r_p > 0.05\,h^{-1}$Mpc",
        fontsize=11)
    ax.set_xlim(11.5, 15.0)
    ax.set_ylim(9.0, 12.5)

    # Threshold line at M* = 10^10 h-1 Msun
    ax.axhline(10.0, color="0.75", ls=":", lw=1.2, zorder=0)
    ax.text(14.9, 10.05, r"$M_*$ threshold", ha="right", va="bottom",
            fontsize=8, color="0.6")

    # Index records by (model, profile) at rp=0.05
    by_key = {}
    for r in records:
        m    = r.get("hod_model", "?")
        p    = r.get("profile", "nfw")
        sext = r.get("use_sat_ext", False)
        rp   = float(r.get("rp_min_wp", 0.3))
        if sext or abs(rp - 0.05) > 1e-6:
            continue
        by_key[(m, p)] = r

    _COSMO = {"h", "Omega_m", "n_s", "ln10^{10}A_s"}

    for prof, ls, label_suf in [("nfw", "-", "NFW"), ("einasto", "--", "Einasto")]:
        # --- explicit SHMR models ---
        for model in ["zu_mandelbaum15", "zacharegkas25", "vanuitert16"]:
            rec = by_key.get((model, prof))
            if rec is None:
                continue
            p   = rec["params"]
            color = MODEL_COLOR[model]
            lbl   = f"{MODEL_LABEL[model]} ({label_suf})" if prof == "nfw" else None

            if model == "zu_mandelbaum15":
                mstar = np.array([
                    float(_mstar_from_mh_zu15(lm, p["lg_m1h"], p["lg_m0star"],
                                              p["beta"], p["delta"], p["gamma"]))
                    for lm in log10m_h
                ])
            elif model == "zacharegkas25":
                mstar = np.asarray(shmr_zacharegkas25(
                    jnp.array(log10m_h),
                    p["log10m1_shmr"], p["log10eps"],
                    p["alpha_shmr"], p["gamma_shmr"], p["delta_shmr"],
                ))
            else:  # vanuitert16
                mstar = np.array([
                    float(_mean_stellar_mass_c_vanuitert16(
                        lm, p["log10m_star0"], p["log10m_h1"],
                        p["beta1"], p["log10_beta2"]))
                    for lm in log10m_h
                ])

            ax.plot(log10m_h, mstar, ls=ls, lw=2.0,
                    color=color, label=lbl, zorder=5)

        # --- threshold HODs: single marker at (Mmin, 10.0) ---
        if prof == "nfw":  # show once; NFW and Einasto Mmin are similar
            for model in ["more2015", "zheng2007", "aum"]:
                rec_nfw = by_key.get((model, "nfw"))
                rec_ein = by_key.get((model, "einasto"))
                for rec, marker, fs in [(rec_nfw, "o", "full"),
                                        (rec_ein, "s", "none")]:
                    if rec is None:
                        continue
                    mmin = float(rec["params"]["log10mmin"])
                    ax.plot(mmin, 10.0, marker=marker, ms=10,
                            color=MODEL_COLOR[model], fillstyle=fs,
                            zorder=6,
                            label=(f"{MODEL_LABEL[model]} (NFW)"
                                   if (rec is rec_nfw) else None))

    # Legend
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
              ncol=2, handlelength=2.0)

    base = os.path.join(out_dir, "fig_shmr_comparison")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=150)
        print(f"  Saved {base}.{ext}")
    plt.close(fig)


def _write_permodel_rst(models, records, chi2_table, docs_dir):
    """Generate docs/_permodel_auto.rst with one subsection per model."""
    import re as _re
    def _to_rst(s):
        return _re.sub(r'\$([^$]+)\$', r':math:`\1`', s)
    PARAM_LABEL_RST = {k: _to_rst(v) for k, v in PARAM_LABEL.items()}

    _COSMO = {"h", "Omega_m", "n_s", "ln10^{10}A_s"}

    lines = []
    for model_key in models:
        label     = MODEL_LABEL.get(model_key, model_key)
        underline = "~" * max(len(label), 4)
        lines += [label, underline, ""]

        # χ² table
        lines += [
            f".. list-table:: :math:`\\chi^2/n_{{\\rm dof}}` — ``{model_key}``",
            "   :header-rows: 1",
            "   :widths: 12 " + " ".join(["11"] * len(RP_MINS)),
            "",
            "   * - Profile",
        ]
        for rp in RP_MINS:
            lines.append(f"     - :math:`r_p>{rp:.2f}`")
        for prof in ("nfw", "einasto"):
            lines.append(f"   * - {prof.upper()}")
            for rp in RP_MINS:
                v = chi2_table.get((model_key, prof, rp), float("nan"))
                lines.append(f"     - {v:.2f}" if np.isfinite(v) else "     - —")
        lines.append("")

        # Figures
        lines += [
            f".. figure:: /_images/fig_wp_allcuts_{model_key}.png",
            "   :width: 72%",
            "   :align: center",
            "",
            f"   Best-fit :math:`w_p(r_p)` for {label} at all scale cuts.",
            "   Solid = NFW; dashed = Einasto. Colours: "
            "green = :math:`r_p>0.30`, cyan = :math:`r_p>0.05`, "
            "blue = :math:`r_p>0.04`, orange = :math:`r_p>0.02`, "
            "red = :math:`r_p>0.01`.",
            "",
            f".. figure:: /_images/fig_wp_params_{model_key}.png",
            "   :width: 95%",
            "   :align: center",
            "",
            f"   MAP parameter values vs minimum scale :math:`r_{{p,\\rm min}}`"
            f" for {label}.",
            "   Filled circles / solid = NFW; open circles / dashed = Einasto.",
            "",
        ]

        # Best-fit parameter tables (one per profile)
        param_data = {}
        free_union = set()
        for r in records:
            if r.get("hod_model") != model_key or r.get("use_sat_ext", False):
                continue
            rp   = float(r.get("rp_min_wp", 0.3))
            prof = r.get("profile", "nfw")
            fps  = [p for p in r.get("free_params", []) if p not in _COSMO]
            free_union.update(fps)
            param_data[(prof, rp)] = {k: r["params"][k] for k in fps
                                      if k in r["params"]}

        free_list = sorted(free_union)
        col_w = "22 " + " ".join(["12"] * len(RP_MINS))

        for prof in ("nfw", "einasto"):
            prof_label = "NFW" if prof == "nfw" else "Einasto"
            lines += [
                f".. list-table:: {prof_label} best-fit parameters",
                "   :header-rows: 1",
                f"   :widths: {col_w}",
                "",
                "   * - Parameter",
            ]
            for rp in RP_MINS:
                lines.append(f"     - :math:`r_p>{rp:.2f}`")
            for pname in free_list:
                plabel = PARAM_LABEL_RST.get(pname, f"``{pname}``")
                lines.append(f"   * - {plabel}")
                for rp in RP_MINS:
                    v = param_data.get((prof, rp), {}).get(pname)
                    lines.append(f"     - {v:.3g}" if v is not None else "     - —")
            lines.append("")

    outpath = os.path.join(docs_dir, "_permodel_auto.rst")
    with open(outpath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading results …")
    records = _load_all_results()
    if not records:
        print("No completed results found.")
        return

    # Shared CAMB cache
    pk_obj    = LinearPowerSpectrum()
    pk_cached = _CachedPkLinear(pk_obj)
    cosmo_def = pk_obj.default_cosmology()

    # Load data from first available record
    info = BGS_BINS[10.0]
    try:
        data_file = _find_data_file(10.0, info, SUM_STAT_DIR)
        rp_data, wp_data, wp_err = _load_data(data_file)
        print(f"  Data: {len(rp_data)} rp bins from {data_file[-60:]}")
    except Exception as e:
        print(f"  Warning: could not load data ({e}); skipping data overlay")
        rp_data = wp_data = wp_err = None

    # Fine rp grid for model predictions
    rp_fine = np.logspace(np.log10(0.008), np.log10(60), 80)

    # Build predictors (one per hod_model × profile)
    predictors = {}
    print("Building predictors …")
    for model in MODELS:
        for prof in ["nfw", "einasto"]:
            key = (model, prof)
            print(f"  {model} / {prof} …", end=" ", flush=True)
            try:
                pred, cosmo = _build_predictor_for(model, prof, pk_cached)
                predictors[key] = (pred, cosmo)
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")
    # Satellite extension predictor (more2015 NFW)
    pred_sext, cosmo_sext = _build_predictor_for("more2015", "nfw", pk_cached)
    predictors[("sext", "nfw")] = (pred_sext, cosmo_sext)

    # Group records by (hod_model, profile)
    by_key = {}
    for r in records:
        m = r.get("hod_model", "?")
        p = r.get("profile", "nfw")
        sext = r.get("use_sat_ext", False)
        key  = ("sext", "nfw") if sext else (m, p)
        by_key.setdefault(key, []).append(r)

    # Pre-compute predictions for selected rp cuts
    predictions = {}   # key → {rp_min: wp_fine array}
    print("Computing predictions …")
    for key, recs in by_key.items():
        if key not in predictors:
            continue
        pred, cosmo = predictors[key]
        predictions[key] = {}
        for r in recs:
            rp_min = float(r.get("rp_min_wp", 0.3))
            print(f"  {key[0]} / {key[1]}  rp>{rp_min:.2f} …", end=" ", flush=True)
            try:
                wp_pred = predict_wp(pred, cosmo, r["params"], rp_fine,
                                     z=r.get("z_eff", 0.115))
                predictions[key][rp_min] = wp_pred
                print("OK")
            except Exception as e:
                print(f"SKIP ({e})")

    # -----------------------------------------------------------------------
    # Shared tables (used by Figure 1 and per-model figures)
    # -----------------------------------------------------------------------

    # chi²/ndof table
    chi2_table = {}
    for r in records:
        m    = r.get("hod_model", "?")
        p    = r.get("profile", "nfw")
        sext = r.get("use_sat_ext", False)
        rp   = float(r.get("rp_min_wp", 0.3))
        chi2 = r.get("chi2", float("nan"))
        ndof = r.get("ndof", 1)
        label_m = "sext" if sext else m
        chi2_table[(label_m, p, rp)] = chi2 / ndof if ndof > 0 else float("nan")

    # Predictions at the actual data rp bins (for ratio panels)
    predictions_at_data = {}
    if rp_data is not None:
        for key, recs in by_key.items():
            if key not in predictors:
                continue
            pred_obj, cosmo_obj = predictors[key]
            predictions_at_data[key] = {}
            for r in recs:
                rp_min = float(r.get("rp_min_wp", 0.3))
                try:
                    wp_p = predict_wp(pred_obj, cosmo_obj, r["params"],
                                      rp_data, z=r.get("z_eff", 0.115))
                    predictions_at_data[key][rp_min] = np.asarray(wp_p)
                except Exception:
                    pass

    # -----------------------------------------------------------------------
    # Figure 1: wp(rp) survey — 2 rows × 5 columns (one per scale cut)
    # -----------------------------------------------------------------------
    panels = [
        (0.30, r"$r_p > 0.30$"),
        (0.05, r"$r_p > 0.05$"),
        (0.04, r"$r_p > 0.04$"),
        (0.02, r"$r_p > 0.02$"),
        (0.01, r"$r_p > 0.01$"),
    ]

    fig, all_ax = plt.subplots(
        2, len(panels), figsize=(24, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.04},
    )
    fig.subplots_adjust(wspace=0.28)

    for col, (rp_min_show, title) in enumerate(panels):
        ax  = all_ax[0, col]
        axr = all_ax[1, col]

        # ---- data (plotted first so models draw on top) ----
        if rp_data is not None:
            ax.errorbar(rp_data, rp_data * wp_data, rp_data * wp_err,
                        fmt="o", color="k", ms=3, lw=1.0, zorder=2,
                        capsize=1.5)
            axr.axhline(1.0, color="k", lw=1.2, zorder=5)
            axr.fill_between(rp_data,
                             1 - wp_err / wp_data,
                             1 + wp_err / wp_data,
                             color="k", alpha=0.12, zorder=0)

        # ---- models (zorder=5 so they overlay the data points) ----
        for model in MODELS:
            for prof, ls in [("nfw", "-"), ("einasto", "--")]:
                key = (model, prof)
                if key not in predictions or rp_min_show not in predictions[key]:
                    continue
                wp_p  = predictions[key][rp_min_show]
                lw    = 1.6 if prof == "nfw" else 1.0
                alpha = 0.92 if prof == "nfw" else 0.65
                c2n   = chi2_table.get((model, prof, rp_min_show), float("nan"))
                lbl   = (f"{MODEL_LABEL[model]}  "
                         r"$\chi^2=$" + f"{c2n:.2f}") if prof == "nfw" else None
                ax.plot(rp_fine, rp_fine * wp_p,
                        ls=ls, lw=lw, alpha=alpha, zorder=5,
                        color=MODEL_COLOR[model], label=lbl)
                if (rp_data is not None
                        and key in predictions_at_data
                        and rp_min_show in predictions_at_data[key]):
                    ratio = predictions_at_data[key][rp_min_show] / wp_data
                    axr.plot(rp_data, ratio, ls=ls, lw=lw, alpha=alpha, zorder=5,
                             color=MODEL_COLOR[model])

        # ---- satellite extension ----
        key_s = ("sext", "nfw")
        if key_s in predictions and rp_min_show in predictions[key_s]:
            wp_s = predictions[key_s][rp_min_show]
            c2n  = chi2_table.get(("sext", "nfw", rp_min_show), float("nan"))
            ax.plot(rp_fine, rp_fine * wp_s, ls="-", lw=2.2,
                    color=MODEL_COLOR["sext"], zorder=6,
                    label=(r"$+$sat.ext.  " r"$\chi^2=$" + f"{c2n:.2f}"))
            if (rp_data is not None
                    and key_s in predictions_at_data
                    and rp_min_show in predictions_at_data[key_s]):
                ratio = predictions_at_data[key_s][rp_min_show] / wp_data
                axr.plot(rp_data, ratio, ls="-", lw=2.2,
                         color=MODEL_COLOR["sext"], zorder=6)

        ax.axvline(rp_min_show,  color="0.5", ls=":", lw=1.0)
        axr.axvline(rp_min_show, color="0.5", ls=":", lw=1.0)

        # wp panel
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.008, 60)
        if col == 0:
            ax.set_ylabel(r"$r_p\,w_p\;[h^{-2}{\rm Mpc}^2]$", fontsize=10)

        # profile indicator (only on first panel to save space)
        if col == 0:
            nfw_h = Line2D([0], [0], ls="-",  lw=2, color="0.4", label="NFW")
            ein_h = Line2D([0], [0], ls="--", lw=1.5, color="0.4", label="Einasto")
            leg_s = ax.legend(handles=[nfw_h, ein_h],
                              fontsize=7, loc="lower right", framealpha=0.85)
            ax.add_artist(leg_s)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9,
                  handlelength=1.5, labelspacing=0.4)

        # residual panel
        axr.set_xscale("log")
        axr.set_yscale("log")
        axr.set_xlabel(r"$r_p\;[h^{-1}{\rm Mpc}]$", fontsize=9)
        axr.set_xlim(0.008, 60)
        axr.set_ylim(0.2, 8.0)
        if col == 0:
            axr.set_ylabel(r"pred$/$ data", fontsize=9)
        axr.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axr.axhline(2.0, color="0.8", ls="--", lw=0.8)
        axr.axhline(0.5, color="0.8", ls="--", lw=0.8)

    fig.suptitle(
        r"BGS LS10 $\log_{10}(M_*/M_\odot) > 10$ — all models, all scale cuts",
        fontsize=11, y=1.01,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"fig_wp_survey_predictions.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved {out}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2: chi²/ndof heatmap
    # -----------------------------------------------------------------------
    working_models = MODELS + ["sext"]
    rp_mins_plot   = [0.30, 0.05, 0.04, 0.02, 0.01]

    # Build chi2 matrices: shape (n_models, n_rp) for NFW and Einasto
    def _chi2_matrix(models, profile_key):
        mat = np.full((len(models), len(rp_mins_plot)), np.nan)
        for i, m in enumerate(models):
            for j, rp in enumerate(rp_mins_plot):
                k = (m, profile_key, rp)
                mat[i, j] = chi2_table.get(k, np.nan)
        return mat

    mat_nfw = _chi2_matrix(working_models, "nfw")
    mat_ein = _chi2_matrix(working_models, "einasto")

    n_rows = len(working_models)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 0.7 * n_rows + 2.5))
    fig2.subplots_adjust(wspace=0.45)

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.BoundaryNorm(
        boundaries=[0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        ncolors=cmap.N)

    row_labels = [MODEL_LABEL.get(m, m) for m in working_models]
    col_labels = [f"$r_p>{rp:.2f}$" for rp in rp_mins_plot]

    for ax2, mat, prof_title in [
        (axes2[0], mat_nfw, "NFW profile"),
        (axes2[1], mat_ein, "Einasto profile"),
    ]:
        im = ax2.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        ax2.set_xticks(range(len(rp_mins_plot)))
        ax2.set_xticklabels(col_labels, fontsize=9)
        ax2.set_yticks(range(len(working_models)))
        ax2.set_yticklabels(row_labels, fontsize=9)
        ax2.set_title(prof_title, fontsize=10)

        for i in range(len(working_models)):
            for j in range(len(rp_mins_plot)):
                val = mat[i, j]
                if np.isfinite(val):
                    txt = f"{val:.1f}" if val < 10 else f"{val:.0f}"
                    color = "white" if val > 10 else "black"
                    ax2.text(j, i, txt, ha="center", va="center",
                             fontsize=8.5, color=color, fontweight="bold")
                else:
                    ax2.text(j, i, "—", ha="center", va="center",
                             fontsize=10, color="0.5")

        cb = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cb.set_label(r"$\chi^2 / n_{\rm dof}$", fontsize=9)

    fig2.suptitle(
        r"BGS LS10 $\log_{10}(M_*/M_\odot) > 10$ — $\chi^2/n_{\rm dof}$ by model and scale cut",
        fontsize=11, y=1.02,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"fig_wp_survey_chi2.{ext}")
        fig2.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved {out}")
    plt.close(fig2)

    # -----------------------------------------------------------------------
    # Per-model figures + RST generation
    # -----------------------------------------------------------------------
    import shutil
    print("\nGenerating per-model figures …")
    for model in MODELS:
        print(f"  {model} …")
        _plot_per_model_wp(model, predictions, predictions_at_data,
                           chi2_table, rp_data, wp_data, wp_err,
                           rp_fine, OUT_DIR)
        _plot_per_model_params(model, records, OUT_DIR)

    _write_permodel_rst(MODELS, records, chi2_table, DOCS_DIR)

    print("\nGenerating SHMR comparison figure …")
    _plot_shmr_comparison(records, chi2_table, OUT_DIR)

    # Copy all new figures to docs/_images/ (Sphinx source directory)
    docs_images = os.path.join(DOCS_DIR, "_images")
    os.makedirs(docs_images, exist_ok=True)
    for model in MODELS:
        for tag in ("allcuts", "params"):
            src = os.path.join(OUT_DIR, f"fig_wp_{tag}_{model}.png")
            if os.path.exists(src):
                shutil.copy(src, docs_images)
    for name in ("fig_wp_survey_predictions.png", "fig_wp_survey_chi2.png",
                 "fig_shmr_comparison.png"):
        src = os.path.join(OUT_DIR, name)
        if os.path.exists(src):
            shutil.copy(src, docs_images)
    print(f"Copied figures → {docs_images}")

    # -----------------------------------------------------------------------
    # Print ASCII table for docs
    # -----------------------------------------------------------------------
    print()
    print("=== chi²/ndof table ===")
    header = f"{'Model':<20} {'Profile':<8} " + \
             " ".join(f"rp>{rp:.2f}" for rp in rp_mins_plot)
    print(header)
    print("-" * len(header))
    for m in working_models:
        for prof in ["nfw", "einasto"]:
            row = f"{MODEL_LABEL.get(m,m):<20} {prof:<8} "
            for rp in rp_mins_plot:
                val = chi2_table.get((m, prof, rp), float("nan"))
                if np.isfinite(val):
                    row += f"{val:8.2f} "
                else:
                    row += "      —  "
            print(row)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
