r"""Structure-of-the-prediction diagram for the sensitivity / Fisher page.

Draws the **forward model as a directed flow of equations**, in the exact order
they are evaluated in :mod:`hod_mod.forecast.forward_jax` — from the input
parameter vector, through the linear power spectrum, halo abundance/structure,
the galaxy–halo connection, the shared hot-gas/baryon and AGN sectors, the 3-D
power spectra, the projection/Limber transforms, and finally the 12-observable
data vector and the Fisher matrix.

The numbered boxes match the numbered equations in the "Structure of the
prediction" appendix of ``docs/sensitivity_fisher.rst``; colours group the
equations by physical **sector** (cosmology / halo abundance / halo structure /
galaxy–halo / hot gas / AGN / observables), using the validated categorical
palette.  A single ``jax.jacfwd`` differentiates this whole graph at once.

Writes ``docs/_images/sensitivity_fisher__forward_model_diagram.png``.

Usage::

    python -m hod_mod.scripts.forecasts.make_forward_diagram
    # slide variant (drops the in-figure title; writes ..._diagram_beamer.png):
    python -m hod_mod.scripts.forecasts.make_forward_diagram --beamer --out-dir <dir>
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_PFX = "sensitivity_fisher__"

# Validated categorical palette (dataviz reference instance), one hue per sector.
SECTOR = {
    "cosmo":  ("#2a78d6", "Cosmology  →  linear P(k)"),
    "hmf":    ("#4a3aa7", "Halo abundance  (Tinker 08/10)"),
    "prof":   ("#1baf7a", "Halo structure  (NFW)"),
    "hod":    ("#008300", "Galaxy–halo connection  (ZM15 HOD)"),
    "gas":    ("#eb6834", "Shared hot-gas / baryon sector"),
    "agn":    ("#e87ba4", "AGN accretion  (Powell 22)"),
    "obs":    ("#e34948", "Projection  →  observables"),
    "fisher": ("#52514e", "Data vector & Fisher"),
}
_INK = "#0b0b0b"
_MUTED = "#6f6d67"          # primary flow arrows (a touch darker than axis-muted)
_VIO = "#5b4fc4"           # secondary "bypass" dependency arrows (violet)
_SURFACE = "#fcfcfb"


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


def _tint(hex_color, frac=0.16):
    """Light tint of a hue over the light chart surface (fill behind ink text)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    sr, sg, sb = (int(_SURFACE.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (sr + frac * (r - sr), sg + frac * (g - sg), sb + frac * (b - sb))


def _box(ax, x, y, w, h, sector, title, eqs, title_size=11, eq_size=10.5):
    """A rounded sector-tinted node: coloured title bar + numbered equation lines."""
    hue = SECTOR[sector][0]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, edgecolor=hue, facecolor=_tint(hue),
        mutation_aspect=1.0, zorder=2))
    # coloured header strip
    ax.text(x + 0.14, y + h - 0.20, title, color=hue, fontsize=title_size,
            fontweight="bold", va="top", ha="left", zorder=3)
    line_y = y + h - 0.20 - 0.34
    for num, eq in eqs:
        if num is not None:
            ax.text(x + 0.16, line_y, num, color="#ffffff", fontsize=8.4,
                    fontweight="bold", va="center", ha="center", zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.16", fc=hue, ec="none"))
        ax.text(x + (0.42 if num is not None else 0.16), line_y, eq, color=_INK,
                fontsize=eq_size, va="center", ha="left", zorder=3)
        line_y -= 0.335
    return (x, y, w, h)


def _arrow(ax, p0, p1, color=_MUTED, style="-", lw=2.0, rad=0.0, alpha=1.0, zorder=6):
    """Draw an arrow *in front of* the boxes (zorder 6 > box zorder 2–4), with a
    white halo so the shaft/head stay legible where they cross a coloured edge."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=17, linewidth=lw,
        linestyle=style, color=color, alpha=alpha,
        connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3, zorder=zorder,
        path_effects=[pe.withStroke(linewidth=lw + 2.4, foreground=_SURFACE)]))


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def make_figure(out_dir=None, *, beamer=False):
    out_dir = out_dir or _docs_images()
    fig, ax = plt.subplots(figsize=(15.5, 13.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 13.2)
    ax.axis("off")
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    # The beamer variant drops the bold in-figure title (the slide's frame title
    # supplies it) and lifts the compact method subtitle into its place.
    if not beamer:
        ax.text(7.75, 12.95, "Structure of the prediction — the differentiable forward model",
                ha="center", va="top", fontsize=16, fontweight="bold", color=_INK)
    ax.text(7.75, (12.95 if beamer else 12.55),
            r"one JAX function $\theta \mapsto d(\theta)$;  a single $\mathtt{jax.jacfwd}$ "
            r"gives $\partial d/\partial\theta$  (numbers = equations in the appendix)",
            ha="center", va="top", fontsize=11, color=_MUTED)

    # ---- S0 : inputs -------------------------------------------------
    s0 = _box(ax, 4.25, 11.55, 7.0, 0.72, "cosmo",
              "Input parameter vector",
              [(None, r"$\theta$ = 5 cosmo | 9 HOD | 6 X-ray | 2 baryon | "
                      r"$\log_{10}$DC, $\beta_P$ | 7 AGN   (31)")],
              title_size=11, eq_size=10.5)

    # ---- S1 : linear power -------------------------------------------
    s1 = _box(ax, 4.6, 9.95, 6.3, 1.28, "cosmo",
              SECTOR["cosmo"][1],
              [("1", r"$P_{\rm shape}(k)\propto k^{n_s}\,T_{\rm EH98}(k)^2$   (Eisenstein–Hu 98)"),
               ("2", r"$P_{\rm lin}(k,z)=P_{\rm shape}\,\frac{\sigma_8^2}{\sigma^2_{\rm shape}(R_8)}"
                     r"\left[\frac{D(z)}{D(0)}\right]^2$")])

    # ---- S2 halo abundance  |  S3 halo structure ---------------------
    s2 = _box(ax, 0.35, 8.05, 5.25, 1.55, "hmf",
              SECTOR["hmf"][1],
              [("3", r"$\sigma(M,z)$  from $P_{\rm shape}$, normalised to $\sigma_8$"),
               ("4", r"$\frac{dn}{dM}=f(\sigma)\frac{\bar\rho_m}{M^2}"
                     r"\left|\frac{d\ln\sigma}{d\ln M}\right|$   (Tinker 08)"),
               ("5", r"$b(\nu)$   (Tinker 10 large-scale bias)")])
    s3 = _box(ax, 5.95, 8.05, 5.0, 1.55, "prof",
              SECTOR["prof"][1],
              [("6", r"$c_{200c}(M,z)$   (Dutton–Macciò 14)"),
               ("7", r"$r_\Delta=\left(\frac{3M}{4\pi\,200\,\rho_c(z)}\right)^{1/3}$,"
                     r"  $r_s=r_\Delta/c$"),
               ("8", r"$u(k|M)$   normalised NFW transform")])

    # ---- S4 HOD | S5 gas | S6 AGN ------------------------------------
    s4 = _box(ax, 0.35, 5.55, 5.05, 2.1, "hod",
              SECTOR["hod"][1],
              [("9", r"$M_*(M_h)$   inverse SHMR (Eq. 19, custom-jvp)"),
               ("10", r"$\sigma_{\ln M_*}(M_h)$   scatter (Eq. 20)"),
               ("11", r"$N_{\rm cen}=\frac{f_c}{2}\,{\rm erfc}\!\left[\frac{\ln M_*^{\rm th}"
                      r"-\ln M_*^c}{\sqrt{2}\,\sigma}\right]$"),
               ("12", r"$N_{\rm sat}=N_{\rm cen}(M_h/M_{\rm sat})^{\alpha}e^{-M_{\rm cut}/M_h}$"),
               ("13", r"$n_{\rm gal}=\!\int\!\frac{dn}{dM}N_{\rm tot}\,dM$,  $b_{\rm eff}$")],
              eq_size=9.9)
    s5 = _box(ax, 5.6, 5.55, 5.15, 2.1, "gas",
              SECTOR["gas"][1],
              [("14", r"$f_b(M)$   hot-gas mass fraction (sigmoid)"),
               ("15", r"$\eta(M)=c_{\rm gas}/c_{\rm DM}$   gas puffiness"),
               ("16", r"$u_{\rm gas}(k|M)$   ($c_{\rm gas}=c\,\eta$)"),
               ("17", r"$\tilde X(k|M)=\hat u_{n_e^2}\,L_X(M)\,w_{kT}\,f_b^2$"),
               ("18", r"$\tilde Y(k|M)=\hat u_P\,P_{500}(M)\,f_b$")],
              eq_size=9.9)
    s6 = _box(ax, 10.95, 5.55, 4.2, 2.1, "agn",
              SECTOR["agn"][1],
              [(None, r"$\langle\log M_{\rm BH}\rangle(M_h)$  via SHMR"),
               (None, r"$+\,M_{\rm BH}$–$M_*$ relation"),
               (None, r"ERDF $P(\lambda)$  (Ananna 22)"),
               (None, r"$L_{\rm bol}=1.26{\times}10^{38}M_{\rm BH}\lambda$"),
               ("30a", r"$\Rightarrow P(\log L_X\,|\,M_h)$")],
              eq_size=9.9)

    # ---- S7 : 3-D power spectra --------------------------------------
    s7 = _box(ax, 1.4, 3.55, 12.7, 1.45, "obs",
              "3-D power spectra   (1-halo + 2-halo)",
              [("19", r"$P_{gg},\,P_{gm}$   with baryon split "
                      r"$u_m=(1{-}f_b)u+f_b u_{\rm gas}$        "
                      r"20  $\;P_{mm}=P_{1h}^{\rm split}+P_{\rm lin}$"),
               ("21", r"$P_{gX},\ P_{gy},\ P_{XX}$   (galaxy $\times$ X-ray / tSZ, X-ray auto; "
                      r"+ AGN duty-cycle term)")],
              title_size=11.5, eq_size=10.4)

    # ---- S8 : projection & abundance ---------------------------------
    s8a = _box(ax, 0.35, 1.52, 4.55, 1.75, "obs",
               "Projection: real space",
               [("22", r"$\xi(r)=\frac{1}{2\pi^2}\!\int\!P\,j_0(kr)k^2dk$"),
                ("23", r"$w_p(r_p)=2\!\int_0^{\Pi}\!\xi_{gg}\,d\pi$"),
                ("24", r"$\Delta\Sigma=\bar\Sigma({<}R)-\Sigma(R)$")],
               title_size=10.5, eq_size=10.0)
    s8b = _box(ax, 5.05, 1.52, 4.9, 1.75, "obs",
               "Projection: Limber angular",
               [("25", r"$C_\ell=\!\int\!\frac{d\chi}{\chi^2}W_1W_2\,"
                       r"P\!\left(\frac{\ell+1/2}{\chi},z\right)$"),
                ("26", r"kernels $W_\kappa$ (shear), $W_{\kappa_{\rm CMB}}$"),
                ("27", r"$C_\ell^{gX,gy,XX,\kappa\kappa,\kappa_c\kappa_c,g\kappa_c,\kappa\kappa_c}$")],
               title_size=10.5, eq_size=10.0)
    s8c = _box(ax, 10.1, 1.52, 5.05, 1.75, "obs",
               "Abundances",
               [("28", r"$n_{\rm gal}$   (scalar; pins $f_c$)"),
                ("29", r"$\Phi(M_*)=-\,d\bar n({>}M_*)/d\log M_*$"),
                ("30", r"$\Phi(\log L_X)=f_{\rm ERDF}\!\int\!\frac{dn}{d\log M_h}"
                       r"P(\log L_X|M_h)\,d\log M_h$")],
               title_size=10.5, eq_size=9.6)

    # ---- S9 : data vector + Fisher -----------------------------------
    s9 = _box(ax, 1.7, 0.10, 12.1, 1.16, "fisher",
              "Data vector  &  Fisher matrix",
              [("31", r"$d=[\,w_p,\ \Delta\Sigma,\ C_\ell^{gX},\ C_\ell^{gy},\ C_\ell^{XX},"
                      r"\ C_\ell^{\kappa\kappa},\ C_\ell^{\kappa_c\kappa_c},\ C_\ell^{g\kappa_c},"
                      r"\ C_\ell^{\kappa\kappa_c},\ \Phi(\log L_X),\ n_{\rm gal},"
                      r"\ \Phi(M_*)\,]$   (12 observables)"),
               (None, r"$F_{ab}=\frac{1}{f^2}\sum_i\partial_a\!\ln d_i\,\partial_b\!\ln d_i$")],
              title_size=11, eq_size=10.0)

    # ---- flow arrows (drawn in front of the boxes, with a white halo) ----
    _arrow(ax, bottom(s0), top(s1))
    # linear P(k) feeds abundance and (2-halo) structure/power
    _arrow(ax, (6.4, 9.95), (3.0, 9.62))     # → S2
    _arrow(ax, (9.1, 9.95), (8.4, 9.62))     # → S3
    # abundance + structure feed the connection blocks and the power spectra
    _arrow(ax, (2.6, 8.05), (2.8, 7.68))     # S2 → S4
    _arrow(ax, (4.7, 8.05), (5.9, 7.68), rad=-0.12)  # S2 → S5 (dn/dM)
    _arrow(ax, (8.4, 8.05), (7.9, 7.68))     # S3 → S5 (u(k|M))
    # dn/dM (cosmology) also seeds the AGN/XLF sector — one clean feeder in the
    # gap above S5 (the SHMR that AGN also needs is internal to that box).
    _arrow(ax, (5.55, 8.02), (11.25, 7.68), rad=-0.10, color=_VIO)
    ax.text(8.55, 7.85, r"$dn/dM$", color=_VIO, fontsize=8.6, style="italic",
            ha="center", va="center", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.6, foreground=_SURFACE)])
    # connection blocks → 3-D power
    _arrow(ax, (2.9, 5.55), (3.0, 5.02))                      # S4 → S7
    _arrow(ax, (8.15, 5.55), (8.1, 5.02), color=SECTOR["gas"][0], lw=2.4)  # S5 → S7
    # shared baryon linkage (highlighted): the SAME f_b(M) drives the lensing
    # (P_gm / P_mm baryon split) AND the X-ray / tSZ amplitudes downstream.
    ax.text(6.35, 5.28, "same $f_b(M)$\n→ lensing  &  X-ray / tSZ",
            color=SECTOR["gas"][0], fontsize=9.0, style="italic", fontweight="bold",
            ha="center", va="center", zorder=8, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.3", fc=_tint(SECTOR["gas"][0], 0.14),
                      ec=SECTOR["gas"][0], lw=1.2))
    # AGN → XLF (abundance)
    _arrow(ax, (13.1, 5.55), (12.7, 3.29), color=SECTOR["agn"][0], rad=0.10)
    # power spectra → projections
    _arrow(ax, (3.0, 3.55), (3.0, 3.29))
    _arrow(ax, (7.5, 3.55), (7.5, 3.29))
    # abundances bypass the 3-D power spectra (n_gal / SMF come straight from the
    # HOD occupation × dn/dM); flagged by a note rather than a box-crossing arrow.
    ax.text(10.75, 3.42, "bypass $P(k)$", color=_VIO, fontsize=8.4, style="italic",
            ha="left", va="center", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.6, foreground=_SURFACE)])
    # projections → data vector / Fisher
    _arrow(ax, bottom(s8a), (2.8, 1.26), rad=0.04)
    _arrow(ax, bottom(s8b), top(s9))
    _arrow(ax, bottom(s8c), (12.4, 1.26), rad=-0.04)

    # ---- sector legend (upper-left, in the empty corner) ------------
    _short = {"cosmo": "Cosmology", "hmf": "Halo abundance", "prof": "Halo structure",
              "hod": "Galaxy–halo (HOD)", "gas": "Hot gas / baryons",
              "agn": "AGN accretion", "obs": "Observables", "fisher": "Data vector & Fisher"}
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=11,
                      markerfacecolor=_tint(c), markeredgecolor=c, markeredgewidth=1.6,
                      label=_short[key])
               for key, (c, lab) in SECTOR.items()]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.945),
                    frameon=True, fontsize=9.0, ncol=1, handletextpad=0.5,
                    labelspacing=0.42, borderpad=0.6, title="sector")
    leg.get_title().set_fontsize(9.0)
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_edgecolor("#e1e0d9")
    leg.get_frame().set_facecolor(_SURFACE)

    suffix = "_beamer" if beamer else ""
    out = os.path.join(out_dir, f"{_PFX}forward_model_diagram{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Structure-of-the-prediction forward-model diagram.")
    parser.add_argument("--beamer", action="store_true",
                        help="drop the in-figure title for use as a Beamer slide figure "
                             "(writes ..._forward_model_diagram_beamer.png)")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: docs/_images)")
    args = parser.parse_args()
    make_figure(out_dir=args.out_dir, beamer=args.beamer)
