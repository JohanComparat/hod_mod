r"""CAMB-quality linear P(k) for the forecast: a linearized EH98-ratio table.

The differentiable forecast uses the analytic EH98 shape, which deviates from
CAMB by up to ~4% around the BAO scale.  For a **Fisher** application the
requirement is precise and modest: the spectrum and its *first derivatives*
must be CAMB-accurate near the fiducial.  That is exactly what a first-order
expansion of the log shape-ratio delivers:

.. math::

    \ln R(k;\theta) \simeq \ln R_0(k)
    + \sum_i \frac{\partial \ln R}{\partial \theta_i}(k)\,
      (\theta_i - \theta_i^{\rm fid}),
    \qquad
    R = \frac{P^{\rm shape}_{\rm CAMB}}{P^{\rm shape}_{\rm EH98}}

with both shapes pivot-normalised at k = 0.05 h/Mpc (the σ8 anchoring makes
the overall normalisation irrelevant).  The EH98 side includes the forecast's
ν suppression, so the Σm_ν derivative row corrects the *residual* of the
first-order tanh form against CAMB.  Eleven CAMB evaluations (fiducial +
central/forward differences over h, Ω_b, Ω_m, n_s, Σm_ν) are distilled once
into ``hod_mod/data/pk_ratio/camb_eh98_ratio.npz`` (the apec_bands pattern);
the JAX side is two ``jnp.interp`` calls and an ``exp``.

Beyond the tabulated k range the log-ratio is edge-clamped (deep power-law
tail where EH98 is accurate).  w0/wa need no rows: they do not change the
z = 0 transfer-function shape (only growth/geometry, handled elsewhere).
"""

from __future__ import annotations

import os

import numpy as np
import jax.numpy as jnp

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "pk_ratio")
_NPZ = os.path.join(_DATA_DIR, "camb_eh98_ratio.npz")

# (parameter name, step, one_sided) — one-sided for sum_mnu (fiducial 0 eV)
_DERIV_SPEC = [("h", 0.02, False), ("Omega_b", 0.002, False),
               ("Omega_m", 0.01, False), ("n_s", 0.01, False),
               ("sum_mnu", 0.12, True)]
_K_PIVOT = 0.05
_K_GRID = np.logspace(-4.0, 1.45, 400)      # CAMB-safe k range [h/Mpc]


def build(npz_path: str = _NPZ, verbose: bool = True) -> dict:
    """Build the ratio table with CAMB (once; ~1 min) and cache it as npz."""
    import jax
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    from hod_mod.forecast.pk_eisenstein_hu import EisensteinHu98PkLinear
    from hod_mod.forecast import params

    lp = LinearPowerSpectrum()
    eh = EisensteinHu98PkLinear()
    fid = params.load_fiducial()
    base = dict(LinearPowerSpectrum.default_cosmology())

    def shapes(over: dict):
        th = dict(fid); th.update(over)
        camb_th = dict(base, Omega_m=th["Omega_m"], Omega_b=th["Omega_b"],
                       Omega_cdm=th["Omega_m"] - th["Omega_b"], h=th["h"],
                       n_s=th["n_s"], mnu=th.get("sum_mnu", 0.0))
        p_c = np.asarray(lp.pk_linear(jnp.asarray(_K_GRID), 0.0, camb_th))
        eh_th = {k: th[k] for k in ("Omega_m", "Omega_b", "h", "n_s", "sigma8")}
        eh_th["Omega_cdm"] = th["Omega_m"] - th["Omega_b"]
        eh_th["sum_mnu"] = th.get("sum_mnu", 0.0)
        p_e = np.asarray(eh.pk_shape(jnp.asarray(_K_GRID), eh_th))
        r = (p_c / np.interp(_K_PIVOT, _K_GRID, p_c)) \
            / (p_e / np.interp(_K_PIVOT, _K_GRID, p_e))
        return np.log(r)

    if verbose:
        print("[pk_camb_ratio] fiducial CAMB/EH98 shape ratio ...")
    lnr0 = shapes({})
    names, fids, rows = [], [], []
    for name, step, one_sided in _DERIV_SPEC:
        f0 = float(fid.get(name, 0.0))
        if verbose:
            print(f"[pk_camb_ratio] d ln R / d {name} (step {step}) ...")
        if one_sided:
            rows.append((shapes({name: f0 + step}) - lnr0) / step)
        else:
            rows.append((shapes({name: f0 + step})
                         - shapes({name: f0 - step})) / (2.0 * step))
        names.append(name)
        fids.append(f0)
    out = dict(lnk=np.log(_K_GRID), lnr0=lnr0, dlnr=np.stack(rows),
               names=np.array(names), fid=np.array(fids))
    os.makedirs(_DATA_DIR, exist_ok=True)
    np.savez_compressed(npz_path, **out)
    if verbose:
        print(f"[pk_camb_ratio] wrote {npz_path}  "
              f"(|lnR0|max = {np.abs(lnr0).max():.3f})")
    return out


def load(npz_path: str = _NPZ) -> dict:
    """Load the table as jnp arrays; raise with build instructions if absent."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} missing — build it once with\n"
            "  python -m hod_mod.forecast.pk_camb_ratio\n(needs CAMB).")
    z = np.load(npz_path, allow_pickle=False)
    return dict(lnk=jnp.asarray(z["lnk"]), lnr0=jnp.asarray(z["lnr0"]),
                dlnr=jnp.asarray(z["dlnr"]),
                names=[str(n) for n in z["names"]],
                fid=jnp.asarray(z["fid"]))


def apply_ratio(pk, k, theta: dict, tab: dict):
    """Multiply an EH98 shape by the linearized CAMB ratio (differentiable)."""
    lnk = jnp.log(jnp.asarray(k))
    lnr = jnp.interp(lnk, tab["lnk"], tab["lnr0"])
    for i, name in enumerate(tab["names"]):
        d = jnp.interp(lnk, tab["lnk"], tab["dlnr"][i])
        lnr = lnr + d * (theta[name] - tab["fid"][i])
    return pk * jnp.exp(lnr)


if __name__ == "__main__":
    build()
