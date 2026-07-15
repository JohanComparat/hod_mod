"""Prototype: jaxace background cosmology vs hod_mod's differentiable layer.

Compares E(z), comoving distance chi(z), and linear growth D(z) between
``jaxace.w0waCDMCosmology`` and :mod:`hod_mod.core.distances` / ``growth_factor``,
on both VALUES and GRADIENTS (the point of staying in JAX). Written as a
one-off evaluation of whether the CosmologicalEmulators JAX packages are worth
adopting -- see the tracking note in ``docs/cosmology.rst`` (Linear Power
Spectrum section) for the conclusion.

``jaxace`` is an *optional* dependency (not part of the hod_mod environment).
Reproduce the evaluation in an isolated venv layered over the hod_mod env::

    python -m venv --system-site-packages /tmp/venv_jaxace
    /tmp/venv_jaxace/bin/python -m pip install \
        "git+https://github.com/CosmologicalEmulators/jaxace.git"
    JAX_PLATFORMS=cpu /tmp/venv_jaxace/bin/python \
        notebooks/compare_jaxace_background.py

Result on 2026-07-15 (jaxace 0.6.1): values agree to < 0.05 %, autodiff
gradients (d chi / d w0) agree to 3.5e-4.
"""
import jax
jax.config.update("jax_enable_x64", True)   # match hod_mod's x64 convention
import jax.numpy as jnp
import numpy as np

import jaxace
from hod_mod.core import distances
from hod_mod.core.halo_mass_function import growth_factor

# ---------------------------------------------------------------- fiducial
# Planck-2018-ish flat LCDM, massless neutrinos for a clean w0waCDM background.
h        = 0.6736
omega_b  = 0.02237          # Omega_b h^2  (physical, jaxace convention)
omega_c  = 0.1200           # Omega_c h^2
w0, wa   = -1.0, 0.0
Omega_m  = (omega_b + omega_c) / h**2      # hod_mod convention (total matter)
print(f"Omega_m (derived) = {Omega_m:.6f}\n")

cosmo = jaxace.w0waCDMCosmology(
    ln10As=3.044, ns=0.9649, h=h,
    omega_b=omega_b, omega_c=omega_c,
    omega_k=0.0, m_nu=0.0, w0=w0, wa=wa,
)
# hod_mod's pk/growth uses a theta dict
theta = dict(h=h, Omega_m=Omega_m, sigma8=0.8, ns=0.9649, w0=w0, wa=wa)

zgrid = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])


def reldiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    return np.abs(a - b) / denom


# ---------------------------------------------------------------- E(z)
E_ja = np.array([float(cosmo.E_z(z)) for z in zgrid])
E_hm = np.array(distances.hubble_e(jnp.asarray(zgrid), Omega_m, w0, wa))
print("E(z) = H(z)/H0")
print(f"  {'z':>5} {'jaxace':>12} {'hod_mod':>12} {'reldiff':>10}")
for z, a, b in zip(zgrid, E_ja, E_hm):
    print(f"  {z:5.2f} {a:12.6f} {b:12.6f} {reldiff(a,b):10.2e}")
print(f"  max reldiff = {reldiff(E_ja, E_hm).max():.2e}\n")

# ------------------------------------------------- comoving distance chi(z)
# Detect jaxace r_z units by comparing magnitude to hod_mod's Mpc.
chi_hm = np.array(distances.comoving_distance(jnp.asarray(zgrid[1:]), h, Omega_m, w0, wa))
r_z_ja = np.array([float(cosmo.r_z(z)) for z in zgrid[1:]])
ratio = np.median(r_z_ja / chi_hm)
unit = "Mpc" if abs(ratio - 1) < 0.05 else (f"Mpc/h (x{ratio:.3f})" if abs(ratio - h) < 0.05 else f"?(x{ratio:.3f})")
print(f"comoving distance  [jaxace r_z units look like: {unit}]")
chi_ja = r_z_ja / (h if abs(ratio - h) < 0.05 else 1.0)   # -> Mpc if needed
print(f"  {'z':>5} {'jaxace[Mpc]':>14} {'hod_mod[Mpc]':>14} {'reldiff':>10}")
for z, a, b in zip(zgrid[1:], chi_ja, chi_hm):
    print(f"  {z:5.2f} {a:14.3f} {b:14.3f} {reldiff(a,b):10.2e}")
print(f"  max reldiff = {reldiff(chi_ja, chi_hm).max():.2e}\n")

# ---------------------------------------------------------------- growth D(z)
# hod_mod growth_factor returns D(z)/D(0); normalise jaxace the same way.
D0 = float(cosmo.D_z(0.0))
D_ja = np.array([float(cosmo.D_z(z)) / D0 for z in zgrid])
D_hm = np.array([float(growth_factor(float(z), theta)) for z in zgrid])
print("linear growth D(z)/D(0)")
print(f"  {'z':>5} {'jaxace':>12} {'hod_mod':>12} {'reldiff':>10}")
for z, a, b in zip(zgrid, D_ja, D_hm):
    print(f"  {z:5.2f} {a:12.6f} {b:12.6f} {reldiff(a,b):10.2e}")
print(f"  max reldiff = {reldiff(D_ja, D_hm).max():.2e}")
print("  (hod_mod uses Carroll+92 fitting fn; jaxace integrates the ODE)\n")

# ---------------------------------------------------------------- GRADIENTS
# Clean common parameter: w0 (enters the DE sector identically in both).
z_test = 1.0

def chi_ja_of_w0(w0v):
    c = jaxace.w0waCDMCosmology(ln10As=3.044, ns=0.9649, h=h,
                                omega_b=omega_b, omega_c=omega_c,
                                omega_k=0.0, m_nu=0.0, w0=w0v, wa=wa)
    return c.r_z(z_test) / (h if unit.startswith("Mpc/h") else 1.0)

def chi_hm_of_w0(w0v):
    return distances.comoving_distance(jnp.asarray([z_test]), h, Omega_m, w0v, wa)[0]

g_ja = float(jax.grad(chi_ja_of_w0)(w0))
g_hm = float(jax.grad(chi_hm_of_w0)(w0))
print("GRADIENT d chi(z=1) / d w0   [Mpc per unit w0]  -- autodiff, both sides")
print(f"  jaxace  = {g_ja:12.3f}")
print(f"  hod_mod = {g_hm:12.3f}")
print(f"  reldiff = {reldiff(g_ja, g_hm):.2e}\n")

# jaxace growth is also differentiable (what an emulated P(k) would ride on):
g_growth = float(jax.grad(lambda w: jaxace.w0waCDMCosmology(
    ln10As=3.044, ns=0.9649, h=h, omega_b=omega_b, omega_c=omega_c,
    omega_k=0.0, m_nu=0.0, w0=w, wa=wa).D_z(z_test))(w0))
print(f"jaxace d D(z=1)/d w0 = {g_growth:.4f}  (confirms jaxace growth is autodiff-able)")
