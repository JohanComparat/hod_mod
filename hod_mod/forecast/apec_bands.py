r"""Band-integrated APEC tables + AGN spectral templates for the tier-2 X-ray layer.

The differentiable forecast cannot trace soxs/AtomDB, so this module distills
:class:`hod_mod.gas.cooling.ApecCoolingTable` (the machinery behind the
production ``fit_xray_joint_bands`` energy-band fit) into static
``log10 Λ_b(log10 T, log10 Z)`` grids, cached as one npz per band set under
``hod_mod/data/apec_bands/``.  The forecast then only ever does JAX bilinear
interpolation on those arrays — soxs is needed once per new band configuration.

Also provides the Morrison & McCammon (1983) ISM photoelectric cross-section
and per-band transmission templates for the obscured-AGN fraction (``agn_fabs``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "apec_bands")

# Default tier-2 band set: the validated production 15×100 eV grid
# (fit_xray_joint_bands._BAND_EDGES) merged into 6 groups over 0.5–2 keV.
DEFAULT_BANDS = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, 2.0)]
BROAD_BAND = (0.5, 2.0)

# table resolution: T range covers the kT(M500c) scaling over the full halo
# grid (the AtomDB APEC grid itself tops out near 64 keV); log-spaced.
_N_T, _T_MIN, _T_MAX = 48, 0.05, 60.0
_N_Z, _Z_MIN, _Z_MAX = 13, 0.05, 3.0


def band_tables(bands, n_T=_N_T, T_min=_T_MIN, T_max=_T_MAX,
                n_Z=_N_Z, Z_min=_Z_MIN, Z_max=_Z_MAX):
    """log10 Λ(log10 T, log10 Z) tables for ``bands`` + the 0.5–2 broad band.

    Returns a dict of numpy arrays: ``lt`` (n_T,), ``lz`` (n_Z,), ``tables``
    (n_bands+1, n_T, n_Z) with the broad band LAST, and ``edges``.  Cached to
    an npz keyed on the full configuration; building needs soxs once.
    """
    edges = [(float(a), float(b)) for a, b in list(bands)] + [BROAD_BAND]
    key = hashlib.md5(repr((edges, n_T, T_min, T_max, n_Z, Z_min, Z_max))
                      .encode()).hexdigest()[:12]
    fp = os.path.join(_DATA_DIR, f"apec_bands_{key}.npz")
    if os.path.exists(fp):
        d = np.load(fp)
        return dict(lt=d["lt"], lz=d["lz"], tables=d["tables"], edges=d["edges"])

    from hod_mod.gas.cooling import ApecCoolingTable
    T_grid = np.logspace(np.log10(T_min), np.log10(T_max), n_T)
    Z_grid = np.logspace(np.log10(Z_min), np.log10(Z_max), n_Z)
    tt, zz = np.meshgrid(T_grid, Z_grid, indexing="ij")
    tabs = []
    for lo, hi in edges:
        cb = ApecCoolingTable(emin=lo, emax=hi, n_T=n_T, T_min=T_min, T_max=T_max,
                              n_Z=n_Z, Z_min=Z_min, Z_max=Z_max)
        tabs.append(np.log10(np.maximum(cb(tt, zz), 1e-40)))   # exact node values
    out = dict(lt=np.log10(T_grid), lz=np.log10(Z_grid),
               tables=np.stack(tabs), edges=np.asarray(edges))
    os.makedirs(_DATA_DIR, exist_ok=True)
    np.savez_compressed(fp, **out)
    return out


# ---------------------------------------------------------------------------
# Morrison & McCammon (1983, ApJ 270, 119) ISM photoelectric absorption
# ---------------------------------------------------------------------------

# rows: (E_lo [keV], c0, c1, c2); sigma(E) = (c0 + c1 E + c2 E²)/E³ × 1e-24 cm²
_MM83 = np.array([
    [0.030, 17.3, 608.1, -2150.0],
    [0.100, 34.6, 267.9, -476.1],
    [0.284, 78.1, 18.8, 4.3],
    [0.400, 71.4, 66.8, -51.4],
    [0.532, 95.5, 145.8, -61.1],
    [0.707, 308.9, -380.6, 294.0],
    [0.867, 120.6, 169.3, -47.7],
    [1.303, 141.3, 146.8, -31.5],
    [1.840, 202.7, 104.7, -17.0],
    [2.471, 342.7, 18.7, 0.0],
    [3.210, 352.2, 18.7, 0.0],
    [4.038, 433.9, -2.4, 0.75],
    [7.111, 629.0, 30.9, 0.0],
    [8.331, 701.2, 25.2, 0.0],
])


def mm83_sigma(e_kev):
    """MM83 ISM photoelectric cross-section σ(E) [cm² per H atom], 0.03–10 keV."""
    e = np.asarray(e_kev, dtype=float)
    idx = np.clip(np.searchsorted(_MM83[:, 0], e, side="right") - 1, 0, len(_MM83) - 1)
    c0, c1, c2 = _MM83[idx, 1], _MM83[idx, 2], _MM83[idx, 3]
    return (c0 + c1 * e + c2 * e ** 2) / e ** 3 * 1e-24


def band_transmission(bands, nh=1e22, gamma=1.8, n_e=256):
    """Energy-flux-weighted transmission of a Γ power law through N_H, per band.

    A static template for the obscured-AGN fraction: the intra-band Γ dependence
    is second order, so the template is evaluated at the fiducial Γ and the free
    ``agn_gamma`` only moves the *band fractions*, not the transmissions.
    """
    out = []
    for lo, hi in bands:
        e = np.linspace(float(lo), float(hi), n_e)
        w = e ** (1.0 - float(gamma))               # energy-flux weighting
        t = np.exp(-mm83_sigma(e) * float(nh))
        out.append(np.trapezoid(w * t, e) / np.trapezoid(w, e))
    return np.asarray(out)
