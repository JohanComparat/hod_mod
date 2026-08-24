r"""CLASS-distilled massive-neutrino corrections to the linear power spectrum.

Two things change when :math:`\Sigma m_\nu > 0` and both are tabulated here,
from the same CLASS solves, as ratios to the massless spectrum at the same
:math:`(\Omega_{\rm m}, \Omega_{\rm b}, h, n_{\rm s}, A_{\rm s})`:

``S_m``
    :math:`P_{\rm m}(k,z;f_\nu) / P_{\rm m}(k,z;0)` -- free streaming suppresses
    the total-matter spectrum on small scales, asymptotically by about
    :math:`-8f_\nu`.

``S_cb``
    :math:`P_{\rm cb}(k,z;f_\nu) / P_{\rm m}(k,z;0)` -- the *cold* (CDM +
    baryon) spectrum, which is what :math:`\sigma(M)` and the halo mass
    function must be built from: neutrinos contribute to :math:`\Omega_{\rm m}`
    but do not cluster into halos.

Both are indexed by :math:`f_\nu = \Omega_\nu/\Omega_{\rm m}` rather than by
:math:`\Sigma m_\nu`, so the table transfers across :math:`h` and
:math:`\Omega_{\rm m}` instead of being pinned to one fiducial.  The
:math:`f_\nu = 0` row is exactly one, so the massless limit is bit-identical to
an uncorrected spectrum.

Interpolation is ``jnp.interp`` in :math:`(\ln k, z, f_\nu)`, so the correction
is differentiable with respect to :math:`\Sigma m_\nu` -- which the closed tanh
form it replaces was too, but at up to 11 percentage points of error at
0.3 eV (it saturated about 1.5 decades of :math:`k` too early).

Rebuild with ``python -m hod_mod.core.nu_ratio`` (needs ``classy``; ~1 min).
"""

from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.core.power_spectrum import omega_nu

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, "data", "pk_ratio")
_NPZ = os.path.normpath(os.path.join(_DATA_DIR, "class_nu_ratio.npz"))

#: Sigma m_nu sampling [eV] of the build grid.  Dense where the minimal normal
#: hierarchy sits (0.06 eV) and stretched out to cover a broad prior.
_MNU_GRID = np.array([0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.18, 0.25, 0.35, 0.50])
#: Redshifts extracted from each solve (one CLASS run serves all of them).
_Z_GRID = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
_K_GRID = np.logspace(-4.0, np.log10(200.0), 400)

_TABLE = None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(npz_path: str = _NPZ, verbose: bool = True) -> dict:
    """Run CLASS over the (Sigma m_nu, z) grid and cache the ratios as npz."""
    from hod_mod.core.power_spectrum import ClassPkLinear, LinearPowerSpectrum

    base = dict(LinearPowerSpectrum.default_cosmology())
    h, Om = float(base["h"]), float(base["Omega_m"])

    def spectra(mnu):
        """(P_m, P_cb) on (_Z_GRID, _K_GRID) for this neutrino mass."""
        cl = ClassPkLinear(z_max=float(_Z_GRID.max()) + 0.1, cache_size=1)
        th = dict(base, sum_mnu=float(mnu))
        th.pop("Omega_cdm", None)          # let omega_cdm() close the budget
        pm = np.array([np.asarray(cl.pk_linear(_K_GRID, float(z), th))
                       for z in _Z_GRID])
        pcb = np.array([np.asarray(cl.pk_linear_cb(_K_GRID, float(z), th))
                        for z in _Z_GRID])
        cl.clear_cache()
        return pm, pcb

    if verbose:
        print(f"[nu_ratio] massless reference ...")
    pm0, _ = spectra(0.0)

    f_nu, lnr_m, lnr_cb = [], [], []
    for mnu in _MNU_GRID:
        if verbose:
            print(f"[nu_ratio] Sigma m_nu = {mnu:.2f} eV ...")
        pm, pcb = spectra(mnu)
        f_nu.append(mnu / (93.14 * h ** 2) / Om)
        lnr_m.append(np.log(pm / pm0))
        lnr_cb.append(np.log(pcb / pm0))

    out = dict(lnk=np.log(_K_GRID), z=_Z_GRID, f_nu=np.array(f_nu),
               lnr_m=np.stack(lnr_m), lnr_cb=np.stack(lnr_cb),
               mnu=_MNU_GRID, h_fid=h, Omega_m_fid=Om)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **out)
    if verbose:
        print(f"[nu_ratio] wrote {npz_path}  "
              f"(f_nu up to {out['f_nu'].max():.4f})")
    return out


# ---------------------------------------------------------------------------
# Load / apply
# ---------------------------------------------------------------------------

def load(npz_path: str = _NPZ) -> dict:
    """Load the table (cached per process) as jnp arrays."""
    global _TABLE
    if _TABLE is None:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"{npz_path} not found — build it with "
                "`python -m hod_mod.core.nu_ratio` (needs classy).")
        d = np.load(npz_path)
        _TABLE = {"lnk": jnp.asarray(d["lnk"]), "z": jnp.asarray(d["z"]),
                  "f_nu": jnp.asarray(d["f_nu"]),
                  "lnr_m": jnp.asarray(d["lnr_m"]),
                  "lnr_cb": jnp.asarray(d["lnr_cb"])}
    return _TABLE


def _lin_weights(x, grid):
    """Index and fraction for linear interpolation of ``x`` on ``grid``.

    Clamped at both ends, matching ``jnp.interp``.  The fraction is
    differentiable in ``x``; the index is not, which is fine -- the gradient
    flows through the fraction.
    """
    i = jnp.clip(jnp.searchsorted(grid, x) - 1, 0, grid.size - 2)
    t = jnp.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0)
    return i, t


@jax.jit
def _interp3_core(cube, f_grid, z_grid, lnk_grid, lnk, z, f_nu):
    i, a = _lin_weights(f_nu, f_grid)
    at_f = (1.0 - a) * cube[i] + a * cube[i + 1]          # (n_z, n_k)
    j, b = _lin_weights(z, z_grid)
    at_z = (1.0 - b) * at_f[j] + b * at_f[j + 1]          # (n_k,)
    return jnp.interp(lnk, lnk_grid, at_z)


def _interp3(cube, tab, k, z, f_nu):
    """Interpolate ``cube`` (n_f, n_z, n_k) at (f_nu, z, ln k).

    Collapsing the two cheap axes first keeps this at
    O(n_z*n_k + n_k + n_out) rather than interpolating all n_f*n_z rows onto
    the output grid: on the 512-point HMF grid that is the difference between
    ~4 ms and ~0.1 ms per call, and sigma() calls it on every evaluation.
    """
    return _interp3_core(cube, tab["f_nu"], tab["z"], tab["lnk"],
                         jnp.log(jnp.asarray(k)), jnp.asarray(z, dtype=float),
                         jnp.asarray(f_nu, dtype=float))


def _f_nu(theta):
    """f_nu = Omega_nu / Omega_m — what the table is indexed by."""
    return omega_nu(theta) / theta["Omega_m"]


def suppression_m(k, theta, z=0.0, tab: dict = None):
    r"""$P_{\rm m}(k,z;\Sigma m_\nu) / P_{\rm m}(k,z;0)$ — exactly 1 at 0 eV."""
    tab = tab if tab is not None else load()
    return jnp.exp(_interp3(tab["lnr_m"], tab, k, z, _f_nu(theta)))


def suppression_cb(k, theta, z=0.0, tab: dict = None):
    r"""$P_{\rm cb}(k,z;\Sigma m_\nu) / P_{\rm m}(k,z;0)$ — the cold spectrum."""
    tab = tab if tab is not None else load()
    return jnp.exp(_interp3(tab["lnr_cb"], tab, k, z, _f_nu(theta)))


if __name__ == "__main__":
    build()
