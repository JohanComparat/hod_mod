"""Σ_y(r_p) transfer kernel — an MCMC-fast, exact re-expression of ``projected_gy``.

A live :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.projected_gy`
call costs ~60 s (halo-model mass integrals + Ogata Hankel), which is hopeless for
MCMC.  This module precomputes the **per-halo-mass projected kernel**

.. math::

    G_{sz}(r_p, M) = \\mathcal{H}\\big[P_{g,y}(k, M)\\big](r_p)\\;\\Delta M ,

so that the stacked Compton-y profile is a single weighted matrix-vector product

.. math::

    \\Sigma_y(r_p) = \\sum_M A(M)\\, G_{sz}(r_p, M).

Why this is **exact** (no interpolation grid, unlike the X-ray band transfer):
the DPM electron pressure (Oppenheimer+2025 Eq. 2) is

.. math::

    P(r, M, z) = P_{0.3}\\, f(r/R_s \\mid \\alpha)\\, E(z)^{\\gamma^P}\\, M_{12}^{\\beta^P},

in which the *shape* :math:`f` — and hence its Fourier transform — is independent of
both :math:`P_{0.3}` and :math:`\\beta^P`.  Those two enter only as an overall scale
and a per-mass power of :math:`M_{12}`.  So a kernel built once at any reference
:math:`(P_{0.3}^{\\rm ref}, \\beta_P^{\\rm ref})` rescales analytically:

.. math::

    A(M) = \\frac{P_{0.3}}{P_{0.3}^{\\rm ref}}\\;
           M_{12}^{\\,\\beta^P - \\beta_P^{\\rm ref}} .

Both the 1-halo and 2-halo terms carry the y-leg linearly, and the Ogata Hankel is
linear in :math:`P(k)`, so projecting each mass column and summing reproduces
``projected_gy``.  Measured agreement is **~3e-6 relative** (not round-off: the
log-space interpolation inside :func:`_pk_to_wp` is mildly non-linear, so
project-then-sum and sum-then-project differ at the ppm level; verified identical
in float32 and float64, which rules out precision).  That is five orders of
magnitude below the ~40% Σ_y measurement errors.  The analytic
:math:`(P_{0.3}, \\beta^P)` rescaling holds to the same ~3e-6 even far from the
reference point (checked at :math:`P_{0.3}\\times1.4`, :math:`\\beta^P+0.15`).
Both are asserted in ``tests/test_sz_transfer.py``.

The native DPM pressure parameters :math:`(P_{0.3}, \\beta^P)` are therefore free at
MCMC speed, and they are the *same* parameters that set the X-ray temperature
:math:`T = P/n_e` — which is what couples the SZ and X-ray legs of the joint fit.
"""

from __future__ import annotations

import numpy as np

from hod_mod.observables.cross_spectra import _pk_to_wp

__all__ = ["build_sz_transfer", "sz_amplitude", "predict_sigma_y"]

# DPM reference mass pivot: M_12 = M_200 / 1e12 (Msun/h), matching
# hod_mod.gas.PressureProfileDPM._pressure_3d.
_M_PIVOT = 1.0e12


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """dM measure of ``np.trapezoid`` as explicit per-node weights."""
    w = np.empty_like(x)
    w[1:-1] = (x[2:] - x[:-2]) / 2.0
    w[0] = (x[1] - x[0]) / 2.0
    w[-1] = (x[-1] - x[-2]) / 2.0
    return w


def build_sz_transfer(
    cross,
    z: float,
    theta_cosmo: dict,
    hod_params: dict,
    rp_arr: np.ndarray,
    beam_fwhm_arcmin: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the Σ_y kernel ``G_sz`` (NR, NM) and the mass grid ``m200`` (NM,).

    Mirrors :meth:`HaloModelCrossSpectra._pk_tables_gy` +
    :meth:`HaloModelCrossSpectra.projected_gy`, but *without* the mass integral, so
    the pressure amplitude/mass-slope stay analytic (see :func:`sz_amplitude`).

    The returned kernel is built at the pressure profile's own current
    ``(_P_03, _beta)``; pass those as the reference to :func:`sz_amplitude`.

    Parameters
    ----------
    cross : HaloModelCrossSpectra
        Must carry a ``pressure_profile`` (DPM recommended — see module docstring).
    z, theta_cosmo, hod_params : as for ``projected_gy``
    rp_arr : (NR,) projected comoving separations [Mpc/h]
    beam_fwhm_arcmin : float | None
        Gaussian beam FWHM of the y-map [arcmin] (ACT DR6 NILC: 1.6).  The
        measurement is beam-convolved, so this is normally mandatory.

    Returns
    -------
    G_sz : (NR, NM) kernel including the trapezoid dM measure
    m200 : (NM,) halo mass grid [Msun/h]
    """
    if cross._pp is None:
        raise RuntimeError("build_sz_transfer needs a pressure_profile on the cross model")
    rp = np.asarray(rp_arr, dtype=float)
    sc = cross._get_static_cache(float(z), theta_cosmo, hod_params)
    k_np = np.asarray(sc["k_np"], float)
    m_np = np.asarray(sc["m_np"], float)
    dndm = np.asarray(sc["dndm_np"], float)
    bias = np.asarray(sc["bias_np"], float)
    pk_lin = np.asarray(sc["pk_lin"], float)
    uk = np.asarray(sc["uk"], float)                       # (Nk, NM) NFW satellite FT
    nc, ns, n_gal, b_eff = cross._get_hod_weights(float(z), theta_cosmo, hod_params, sc)
    nc = np.asarray(nc, float)
    ns = np.asarray(ns, float)
    n_gal = float(n_gal)
    b_eff = float(b_eff)

    # y-leg FT at the profile's current (P_03, beta_P) — the reference amplitude.
    y_uk = np.asarray(
        cross._pp.pressure_uk(k_arr=k_np, m200_arr=m_np, r200_arr=np.asarray(sc["r_delta"], float),
                              c200_arr=np.asarray(sc["c_np"], float), z=float(z),
                              theta_cosmo=theta_cosmo),
        float)                                             # (Nk, NM)

    # Per-mass P_gy(k, M): 1-halo + 2-halo, exactly the integrands of _pk_tables_gy.
    Pk_M = (dndm[None, :] * (nc[None, :] + ns[None, :] * uk) * y_uk / n_gal
            + dndm[None, :] * bias[None, :] * y_uk * (b_eff * pk_lin[:, None]))   # (Nk, NM)

    log_k = np.log(k_np)
    if beam_fwhm_arcmin is not None:
        B = np.asarray(cross._beam_window_k(log_k, float(z), theta_cosmo,
                                            float(beam_fwhm_arcmin)), float)
        Pk_M = Pk_M * B[:, None]

    # Project each mass column: the Ogata Hankel is linear in P(k), so
    # H[∫dM Pk_M] == ∫dM H[Pk_M].
    G = np.empty((rp.size, m_np.size), dtype=float)
    floor = 1e-300
    for j in range(m_np.size):
        G[:, j] = _pk_to_wp(rp, log_k, np.log(np.maximum(Pk_M[:, j], floor)))
    return G * _trapezoid_weights(m_np)[None, :], m_np


def sz_amplitude(m200: np.ndarray, p03: float, beta_p: float,
                 p03_ref: float, beta_p_ref: float) -> np.ndarray:
    """Per-mass rescaling A(M) from the kernel's reference DPM pressure params.

    ``A(M) = (P_0.3 / P_0.3_ref) · M_12^(β_P − β_P_ref)`` — exact, because the DPM
    pressure *shape* does not depend on either parameter (module docstring).
    """
    m12 = np.asarray(m200, float) / _M_PIVOT
    return (float(p03) / float(p03_ref)) * m12 ** (float(beta_p) - float(beta_p_ref))


def predict_sigma_y(G_sz: np.ndarray, m200: np.ndarray, p03: float, beta_p: float,
                    p03_ref: float, beta_p_ref: float) -> np.ndarray:
    """Σ_y(r_p) = Σ_M A(M) G_sz(r_p, M) — the MCMC-fast evaluation."""
    return G_sz @ sz_amplitude(m200, p03, beta_p, p03_ref, beta_p_ref)
