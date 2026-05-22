"""Planck 2018 cosmological priors for HOD fitting.

Provides best-fit values, 1σ uncertainties, and 3σ flat bounds from the
Planck 2018 primary CMB analysis (TT,TE,EE+lowE likelihood, Table 2).

Reference
---------
Planck Collaboration 2020, A&A 641, A6
https://arxiv.org/abs/1807.06209

The primary parameters and their 68% confidence intervals are:

.. math::

    h            &= 0.6736 \\pm 0.0054 \\\\
    \\Omega_m      &= 0.3153 \\pm 0.0073 \\\\
    \\Omega_b h^2  &= 0.02237 \\pm 0.00015 \\\\
    n_s           &= 0.9649 \\pm 0.0042 \\\\
    \\ln 10^{10} A_s &= 3.044 \\pm 0.014

The 3σ flat bounds are :math:`[\\mu - 3\\sigma, \\mu + 3\\sigma]`.

Usage in YAML config
--------------------
Set ``prior_type: gaussian`` for any cosmological parameter to activate the
Gaussian prior.  The ``bounds`` field is still required and acts as hard
clipping beyond which the log-prior returns ``-inf``::

    parameters:
      h:
        free: true
        init: 0.6736
        bounds: [0.6574, 0.6898]   # 3σ hard bounds
        prior_type: gaussian
        prior_mean: 0.6736
        prior_sigma: 0.0054
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Planck 2018 TT,TE,EE+lowE best-fit values and 1σ uncertainties
# ---------------------------------------------------------------------------

PLANCK18_MEANS: dict[str, float] = {
    "h":                  0.6736,
    "Omega_m":            0.3153,
    "Omega_b":            0.0493,   # Omega_b = Omega_b*h^2 / h^2 ≈ 0.02237 / 0.6736^2
    "Omega_cdm":          0.2607,
    "n_s":                0.9649,
    "ln10^{10}A_s":       3.044,
    "sigma8":             0.8111,
    # S8 = sigma8 * sqrt(Omega_m / 0.3) — Planck 2018 value and propagated uncertainty
    "S8":                 0.8319,   # 0.8111 * sqrt(0.3153/0.3)
}

PLANCK18_SIGMAS: dict[str, float] = {
    "h":                  0.0054,
    "Omega_m":            0.0073,
    "Omega_b":            0.0008,
    "Omega_cdm":          0.0073,   # dominated by Omega_m uncertainty
    "n_s":                0.0042,
    "ln10^{10}A_s":       0.014,
    "sigma8":             0.0060,
    # Propagated from sigma(sigma8) and sigma(Omega_m) via error propagation
    "S8":                 0.0114,
}

PLANCK18_3SIGMA: dict[str, tuple[float, float]] = {
    name: (PLANCK18_MEANS[name] - 3.0 * PLANCK18_SIGMAS[name],
           PLANCK18_MEANS[name] + 3.0 * PLANCK18_SIGMAS[name])
    for name in PLANCK18_MEANS
}


# ---------------------------------------------------------------------------
# Log-prior functions
# ---------------------------------------------------------------------------

def planck18_log_prior(theta: dict, params: list | None = None) -> float:
    """Sum of Gaussian log-prior terms for Planck 2018 cosmological parameters.

    .. math::

        \\ln \\pi(\\theta) = -\\frac{1}{2} \\sum_i
        \\left( \\frac{\\theta_i - \\mu_i}{\\sigma_i} \\right)^2

    Parameters
    ----------
    theta : dict
        Parameter dict.  Only keys present in :data:`PLANCK18_MEANS` contribute.
    params : list of str, optional
        Restrict to these parameters only.  Default: all keys in
        :data:`PLANCK18_MEANS` that also appear in ``theta``.

    Returns
    -------
    float
        Log-prior value.  Returns ``-inf`` if any parameter is outside its
        3σ hard bounds.
    """
    keys = params if params is not None else list(PLANCK18_MEANS.keys())
    log_pi = 0.0
    for k in keys:
        if k not in theta or k not in PLANCK18_MEANS:
            continue
        val  = float(theta[k])
        lo, hi = PLANCK18_3SIGMA[k]
        if not (lo <= val <= hi):
            return -np.inf
        z = (val - PLANCK18_MEANS[k]) / PLANCK18_SIGMAS[k]
        log_pi -= 0.5 * z * z
    return log_pi


def gaussian_log_prior(val: float, mean: float, sigma: float,
                        lo: float = -np.inf, hi: float = np.inf) -> float:
    """Gaussian log-prior for a single parameter.

    .. math::

        \\ln \\pi(\\theta) = -\\frac{1}{2}
        \\left( \\frac{\\theta - \\mu}{\\sigma} \\right)^2

    Returns ``-inf`` if ``val`` is outside ``[lo, hi]`` (hard bounds).

    Parameters
    ----------
    val : float
    mean, sigma : float
        Gaussian mean and standard deviation.
    lo, hi : float
        Hard bounds (uniform outside returns -inf).
    """
    if not (lo <= val <= hi):
        return -np.inf
    z = (val - mean) / sigma
    return -0.5 * z * z
