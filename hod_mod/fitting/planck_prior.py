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

import jax.numpy as jnp

from hod_mod.core.amplitude import AMPLITUDE_KEY, RETIRED_AMPLITUDE_KEYS

# ---------------------------------------------------------------------------
# Planck 2018 TT,TE,EE+lowE best-fit values and 1σ uncertainties
# ---------------------------------------------------------------------------

# The **independent** directions.  A Gaussian may be placed on each of these at
# the same time, because Planck constrains them jointly and none is a function
# of the others.  Deliberately *not* here: ``Omega_cdm``, ``sigma8`` and ``S8``
# -- see :data:`PLANCK18_DERIVED_MEANS`.
PLANCK18_MEANS: dict[str, float] = {
    "h":                  0.6736,
    "Omega_m":            0.3153,
    "Omega_b":            0.0493,   # Omega_b = Omega_b*h^2 / h^2 ≈ 0.02237 / 0.6736^2
    "n_s":                0.9649,
    "ln10^{10}A_s":       3.044,
}

PLANCK18_SIGMAS: dict[str, float] = {
    "h":                  0.0054,
    "Omega_m":            0.0073,
    "Omega_b":            0.0008,
    "n_s":                0.0042,
    "ln10^{10}A_s":       0.014,
}

PLANCK18_3SIGMA: dict[str, tuple[float, float]] = {
    name: (PLANCK18_MEANS[name] - 3.0 * PLANCK18_SIGMAS[name],
           PLANCK18_MEANS[name] + 3.0 * PLANCK18_SIGMAS[name])
    for name in PLANCK18_MEANS
}

# Quantities Planck also reports, which this package treats as **outputs**.
#
# They are here so a *derived* posterior can be compared with Planck, and for
# plotting.  They are not prior directions and :func:`planck18_log_prior`
# refuses to score them:
#
# * ``Omega_cdm`` is fixed by ``Omega_m - Omega_b - Omega_nu``.  Scoring it
#   alongside ``Omega_m`` counts the matter density twice -- measured, a 1σ
#   ``Omega_m`` shift went from -2lnL = 1.0 to 2.0 once the implied
#   ``Omega_cdm`` was also present, and ``default_cosmology()`` carries all
#   three keys.
# * ``sigma8`` and ``S8`` are derived from the spectrum
#   (``core.amplitude.sigma8_from_theta`` / ``s8_from_theta``) and were retired
#   as inputs; a theta carrying either is on the old convention.
PLANCK18_DERIVED_MEANS: dict[str, float] = {
    "Omega_cdm":          0.2607,
    "sigma8":             0.8111,
    # S8 = sigma8 * sqrt(Omega_m / 0.3)
    "S8":                 0.8319,   # 0.8111 * sqrt(0.3153/0.3)
}

PLANCK18_DERIVED_SIGMAS: dict[str, float] = {
    "Omega_cdm":          0.0073,   # dominated by the Omega_m uncertainty
    "sigma8":             0.0060,
    # Propagated from sigma(sigma8) and sigma(Omega_m)
    "S8":                 0.0114,
}

_NEG_INF = float("-inf")


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
        Parameter dict.  Only keys present in :data:`PLANCK18_MEANS` -- the
        independent directions -- contribute.  Keys in
        :data:`PLANCK18_DERIVED_MEANS` are ignored rather than scored:
        ``Omega_cdm`` is a function of ``Omega_m``, ``Omega_b`` and
        ``sum_mnu``, and ``default_cosmology()`` carries all of them, so
        scoring it would count the matter density twice.
    params : list of str, optional
        Restrict to these parameters only.  Default: all keys in
        :data:`PLANCK18_MEANS` that also appear in ``theta``.  Naming a derived
        quantity here raises rather than silently double-counting.

    Returns
    -------
    float
        Log-prior value.  Returns ``-inf`` if any parameter is outside its
        3σ hard bounds.

    Raises
    ------
    ValueError
        If ``theta`` carries a retired amplitude key (``sigma8``/``S8``), or if
        ``params`` names a derived quantity.
    """
    stray = [k for k in RETIRED_AMPLITUDE_KEYS if k in theta]
    if stray:
        raise ValueError(
            f"planck18_log_prior: {', '.join(repr(x) for x in stray)} is no "
            f"longer a parameter -- {AMPLITUDE_KEY!r} is the only amplitude in "
            "this package.  sigma8 and S8 are derived from the spectrum, so a "
            "Gaussian on one of them alongside the amplitude would count the "
            "same Planck information twice.  Compare a derived posterior "
            "against PLANCK18_DERIVED_MEANS instead.")

    if params is not None:
        bad = [k for k in params if k in PLANCK18_DERIVED_MEANS]
        if bad:
            raise ValueError(
                f"planck18_log_prior: {', '.join(repr(x) for x in bad)} is a "
                "derived quantity, not an independent prior direction; scoring "
                "it alongside the parameters it is a function of double-counts "
                "the Planck constraint.  Independent directions are "
                f"{sorted(PLANCK18_MEANS)}.")
        keys = params
    else:
        keys = list(PLANCK18_MEANS.keys())

    log_pi = 0.0
    for k in keys:
        if k not in theta or k not in PLANCK18_MEANS:
            continue
        val  = float(theta[k])
        lo, hi = PLANCK18_3SIGMA[k]
        if not (lo <= val <= hi):
            return _NEG_INF
        z = (val - PLANCK18_MEANS[k]) / PLANCK18_SIGMAS[k]
        log_pi -= 0.5 * float(z * z)
    return log_pi


def gaussian_log_prior(val: float, mean: float, sigma: float,
                        lo: float = float("-inf"), hi: float = float("inf")) -> float:
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
    in_bounds = (val >= lo) & (val <= hi)
    z = (val - mean) / sigma
    return float(jnp.where(in_bounds, -0.5 * z * z, _NEG_INF))
