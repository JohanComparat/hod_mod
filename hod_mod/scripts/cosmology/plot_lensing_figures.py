"""Generate docs/_images/lensing_0*.png for docs/lensing.rst.

Usage:  python hod_mod/scripts/cosmology/plot_lensing_figures.py
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_profiles import nfw_rho, nfw_sigma, nfw_delta_sigma
from hod_mod.core.lensing_profiles import (
    tnfw_rho, tnfw_sigma, tnfw_delta_sigma,
    bmo_rho, bmo_sigma, bmo_delta_sigma,
    hernquist_rho, hernquist_sigma, hernquist_delta_sigma,
    nfw_params_from_mass)
from hod_mod.observables.lensing import ClusterLensingPrediction

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "docs", "_images")
PC2 = 1e-12

theta = LinearPowerSpectrum.default_cosmology()
theta["sigma8"] = 0.81
m_h, z_l, z_s = 1e14, 0.3, 1.0

clp = ClusterLensingPrediction(profile="bmo", mdef="vir",
                               cm_relation="duffy08", tau_v=2.5)
c = float(clp.concentration(jnp.array([m_h]), z_l, theta)[0])
rho_s, r_s, r_vir = (float(v[0]) for v in nfw_params_from_mass(
    jnp.array([m_h]), jnp.array([c]), z_l, theta, "vir"))
tau = 2.5 * c
m_hern, rb_hern = m_h, 0.551 * r_vir / 4.0


def fig01_profiles():
    r = jnp.logspace(-2.5, 1.3, 300)
    R = jnp.logspace(-2.5, 1.5, 300)
    rho = {"NFW": nfw_rho(r, rho_s, r_s),
           "TJ (sharp trunc.)": tnfw_rho(r, rho_s, r_s, c),
           "BMO (smooth trunc.)": bmo_rho(r, rho_s, r_s, tau),
           "Hernquist": hernquist_rho(r, m_hern, rb_hern)}
    sig = {"NFW": nfw_sigma(R, rho_s, r_s),
           "TJ (sharp trunc.)": tnfw_sigma(R, rho_s, r_s, c),
           "BMO (smooth trunc.)": bmo_sigma(R, rho_s, r_s, tau),
           "Hernquist": hernquist_sigma(R, m_hern, rb_hern)}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for name in rho:
        axes[0].loglog(r, rho[name], label=name)
        axes[1].loglog(R, sig[name] * PC2, label=name)
    axes[0].axvline(r_vir, color="gray", ls=":", lw=1)
    axes[0].set_ylim(1e8, 1e18)
    axes[0].set_xlabel(r"$r$ [Mpc/$h$]")
    axes[0].set_ylabel(r"$\rho(r)$ [$h^2 M_\odot/{\rm Mpc}^3$]")
    axes[1].set_ylim(1e-3, 2e3)
    axes[1].set_xlabel(r"$R$ [Mpc/$h$]")
    axes[1].set_ylabel(r"$\Sigma(R)$ [$h\,M_\odot/{\rm pc}^2$]")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(r"$M_{\rm vir}=10^{14}\,M_\odot/h$, $z=0.3$ "
                 r"(dotted: $r_{\rm vir}$)", fontsize=10)
    fig.savefig(os.path.join(OUT, "lensing_01_profiles.png"), dpi=130)


def fig02_deltasigma_decomposition():
    R = jnp.logspace(-1.5, 1.3, 60)
    f_cen, s_off = 0.7, 0.4
    ds_cen = clp.delta_sigma_1h(R, m_h, z_l, theta)
    ds_off = clp.delta_sigma_off(R, m_h, z_l, theta, sigma_off=s_off)
    ds_2h = clp.delta_sigma_2h(R, m_h, z_l, theta)
    ds_tot = f_cen * ds_cen + (1 - f_cen) * ds_off + ds_2h
    plt.figure(figsize=(6.5, 4.5))
    plt.loglog(R, ds_tot * PC2, "k-", lw=2, label="total")
    plt.loglog(R, f_cen * ds_cen * PC2, "--",
               label=rf"centered 1h ($f_{{\rm cen}}={f_cen}$)")
    plt.loglog(R, (1 - f_cen) * ds_off * PC2, "-.",
               label=rf"off-centered 1h ($\sigma_{{\rm off}}={s_off}$)")
    plt.loglog(R, ds_2h * PC2, ":", label="2-halo")
    plt.xlabel(r"$R$ [Mpc/$h$]")
    plt.ylabel(r"$\Delta\Sigma$ [$h\,M_\odot/{\rm pc}^2$]")
    plt.title(r"$M=10^{14}\,M_\odot/h$, $z_l=0.3$ (BMO, $\tau_v=2.5$)")
    plt.ylim(0.05, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lensing_02_deltasigma_decomposition.png"),
                dpi=130)


def fig03_einstein_radius():
    logM = jnp.linspace(13.8, 15.8, 40)
    plt.figure(figsize=(6.5, 4.5))
    for zs_i in (0.8, 1.5, 3.0):
        theta_E = jax.vmap(
            lambda lm: clp.einstein_radius(10.0**lm, z_l, zs_i, theta)[1]
        )(logM)
        plt.plot(logM, theta_E, label=f"$z_s = {zs_i}$")
    plt.xlabel(r"$\log_{10} M_{\rm vir}\,[M_\odot/h]$")
    plt.ylabel(r"$\theta_E$ [arcsec]")
    plt.title(r"Einstein radius, $z_l = 0.3$ (BMO)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lensing_03_einstein_radius.png"), dpi=130)


def fig04_magnification():
    m15, zs_sl = 1e15, 2.0
    R = jnp.logspace(-3.2, 0.5, 400)
    arc = float(clp.arcsec_per_mpc(z_l, theta))
    r_tan, r_rad = clp.critical_curves(m15, z_l, zs_sl, theta)
    mu = clp.magnification(R, m15, z_l, zs_sl, theta)
    plt.figure(figsize=(6.5, 4))
    plt.loglog(np.asarray(R) * arc, np.abs(np.asarray(mu)), "k-")
    plt.axvline(float(r_tan) * arc, color="C3", ls="--",
                label=f"tangential (Einstein): {float(r_tan)*arc:.1f}''")
    plt.axvline(float(r_rad) * arc, color="C0", ls="--",
                label=f"radial: {float(r_rad)*arc:.1f}''")
    plt.xlabel(r"$\theta$ [arcsec]")
    plt.ylabel(r"$|\mu(\theta)|$")
    plt.title(r"$M=10^{15}\,M_\odot/h$, $z_l=0.3$, $z_s=2$")
    plt.ylim(0.05, 1e3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lensing_04_magnification.png"), dpi=130)


if __name__ == "__main__":
    fig01_profiles()
    fig02_deltasigma_decomposition()
    fig03_einstein_radius()
    fig04_magnification()
    print("wrote lensing_01..04 to", os.path.abspath(OUT))
