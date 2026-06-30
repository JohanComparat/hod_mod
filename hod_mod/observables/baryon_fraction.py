"""Mass-dependent baryon fraction models for halo-model galaxy lensing.

Within halos the baryon fraction f_b(M) = M_baryon / M_total is suppressed
below the cosmic value f_b^cosmic = Omega_b / Omega_m at group and cluster
masses by AGN and stellar feedback (FLAMINGO, BAHAMAS, TNG simulations).

Three parametric models share the common interface::

    fb_model(m_h, theta_cosmo, params) -> jnp.ndarray

and a ``default_params()`` static method.

The ``params`` dict returned by :meth:`BaryonFractionSigmoid.default_params`
also contains gas-concentration parameters (``log10_eta_min``,
``log10_M_eta``) consumed by
:meth:`~hod_mod.observables.clustering.FullHaloModelPrediction._pk_tables_full`
when splitting P_gm into CDM and gas 1-halo integrals.  They are silently
ignored by the ``BaryonFractionSigmoid`` callable itself.

References
----------
van Daalen et al. 2011, MNRAS 415, 3649
  `arXiv:1104.1174 <https://arxiv.org/abs/1104.1174>`_ — baryon suppression
McCarthy et al. 2017, MNRAS 465, 2936
  `arXiv:1612.06090 <https://arxiv.org/abs/1612.06090>`_ — BAHAMAS calibration
Schneider & Teyssier 2015, JCAP 12, 049
  `arXiv:1510.06034 <https://arxiv.org/abs/1510.06034>`_ — baryonification
FLAMINGO simulations
  `arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_ — f_gas at group scales
  `arXiv:2509.10230 <https://arxiv.org/abs/2509.10230>`_ — hot gas profiles
Veenema et al. 2026
  `arXiv:2603.13095 <https://arxiv.org/abs/2603.13095>`_ — closure-radius model
IllustrisTNG/MillenniumTNG baryonic effects on halo concentration
  `arXiv:2409.01758 <https://arxiv.org/abs/2409.01758>`_ — c_hydro/c_DMO Table 2
Mead et al. 2015
  `arXiv:1611.08606 <https://arxiv.org/abs/1611.08606>`_ — gas as NFW with
  reduced concentration
ML gas profiles (Pfeifer et al. 2025)
  `arXiv:2512.09021 <https://arxiv.org/abs/2512.09021>`_ — M_BH primary driver
Ayromlou et al. 2023
  `arXiv:2209.07390 <https://arxiv.org/abs/2209.07390>`_ — baryon budget in TNG/FLAMINGO
Zhang et al. 2025
  `arXiv:2511.17313 <https://arxiv.org/abs/2511.17313>`_ — CGM baryon budget in
  Milky Way-mass halos; observational motivation for the low-mass upturn
"""

import jax
import jax.numpy as jnp
from functools import partial


class BaryonFractionSigmoid:
    r"""Mass-dependent baryon fraction via a sigmoid in log-mass.

    .. math::

        f_b(M) = \frac{f_b^{\rm cosmic}}{1 + (M_{\rm pivot} / M)^{\beta_b}}

    Limits:

    * :math:`M \gg M_{\rm pivot}` → :math:`f_b^{\rm cosmic} = \Omega_b / \Omega_m`
      (clusters recover the cosmic baryon fraction).
    * :math:`M \ll M_{\rm pivot}` → 0
      (feedback-dominated low-mass halos are gas-poor).

    Parameters
    ----------
    (passed at call time via ``params`` dict)

    Notes
    -----
    Typical FLAMINGO values at group scale: :math:`\log_{10} M_{\rm pivot} \approx 13.5`,
    :math:`\beta_b \approx 1.5`.
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        m_h: jnp.ndarray,
        theta_cosmo: dict,
        params: dict,
    ) -> jnp.ndarray:
        r"""Baryon fraction f_b(M) ∈ [0, f_b^cosmic].

        Parameters
        ----------
        m_h : jnp.ndarray
            Halo masses [Msun/h].
        theta_cosmo : dict
            Cosmological parameters (needs ``Omega_b``, ``Omega_m``).
        params : dict
            Model parameters: ``log10_M_pivot`` [Msun/h], ``beta_b`` (> 0).

        Returns
        -------
        f_b : jnp.ndarray, same shape as m_h, values in [0, f_b^cosmic].
        """
        f_b_cosmic = theta_cosmo["Omega_b"] / theta_cosmo["Omega_m"]
        M_pivot = 10.0 ** params["log10_M_pivot"]
        f_b_min = params.get("f_b_min", 0.0)
        fb = f_b_cosmic / (1.0 + (M_pivot / m_h) ** params["beta_b"])
        return jnp.maximum(fb, f_b_min)

    @staticmethod
    def default_params() -> dict:
        """Default baryon and gas-concentration parameters.

        The sigmoid parameters (``log10_M_pivot``, ``beta_b``) are used by
        :meth:`__call__` to compute f_b(M).  The gas-concentration parameters
        (``log10_eta_min``, ``log10_M_eta``) are consumed by
        :meth:`~hod_mod.observables.clustering.FullHaloModelPrediction._pk_tables_full`
        and are silently ignored by this callable.

        Sources:

        * ``log10_M_pivot = 13.5`` — FLAMINGO f_gas measurements at group scale
          (`arXiv:2510.25419 <https://arxiv.org/abs/2510.25419>`_);
          closure-radius model (`arXiv:2603.13095 <https://arxiv.org/abs/2603.13095>`_)
        * ``beta_b = 1.5`` — sigmoid sharpness calibrated to FLAMINGO
        * ``log10_eta_min = −0.22`` — log10(0.6); IllustrisTNG group-scale
          c_hydro/c_DMO ≈ 0.6 at M ~ 10^13 Msun
          (`arXiv:2409.01758 <https://arxiv.org/abs/2409.01758>`_ Table 2)
        * ``log10_M_eta = 13.0`` — break mass M_2 from IllustrisTNG fit
          (`arXiv:2409.01758 <https://arxiv.org/abs/2409.01758>`_ Table 2)
        """
        return {
            "log10_M_pivot": 13.5,
            "beta_b":        1.5,
            "f_b_min":       0.01,   # 1% floor; CGM gas in dwarf halos
            "log10_eta_min": -0.22,  # log10(0.6), arXiv:2409.01758 Table 2
            "log10_M_eta":   13.0,   # M_2 break mass, arXiv:2409.01758 Table 2
        }


class BaryonFractionPowerLaw:
    r"""Mass-dependent baryon fraction via a power law (clipped to cosmic value).

    .. math::

        f_b(M) = \mathrm{clip}\!\left(
                 f_b^{\rm cosmic} \left(\frac{M}{M_{\rm ref}}\right)^{\alpha_b},
                 \; 0,\; f_b^{\rm cosmic}
                 \right)

    Parameters
    ----------
    (passed at call time via ``params`` dict)

    Notes
    -----
    :math:`\alpha_b = 0` recovers the constant cosmic fraction.
    Larger :math:`\alpha_b` gives stronger mass dependence.
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        m_h: jnp.ndarray,
        theta_cosmo: dict,
        params: dict,
    ) -> jnp.ndarray:
        r"""Baryon fraction f_b(M) ∈ [0, f_b^cosmic].

        Parameters
        ----------
        m_h : jnp.ndarray
            Halo masses [Msun/h].
        theta_cosmo : dict
            Cosmological parameters (needs ``Omega_b``, ``Omega_m``).
        params : dict
            Model parameters: ``log10_M_ref`` [Msun/h], ``alpha_b`` (≥ 0).
        """
        f_b_cosmic = theta_cosmo["Omega_b"] / theta_cosmo["Omega_m"]
        M_ref = 10.0 ** params["log10_M_ref"]
        raw = f_b_cosmic * (m_h / M_ref) ** params["alpha_b"]
        return jnp.clip(raw, 0.0, f_b_cosmic)

    @staticmethod
    def default_params() -> dict:
        """Default parameters: mild power-law rise toward clusters."""
        return {"log10_M_ref": 14.0, "alpha_b": 0.3}


class BaryonFractionUpturn:
    r"""Double-sigmoid baryon fraction with group-scale valley and low-mass upturn.

    .. math::

        f_b(M) = f_b^{\rm min}
                 + \frac{f_b^{\rm cosmic} - f_b^{\rm min}}{1 + (M_{\rm hi}/M)^{\beta_{\rm hi}}}
                 + \frac{f_b^{\rm lo,amp}}{1 + (M/M_{\rm lo})^{\beta_{\rm lo}}}

    Physical motivation:

    * Group-scale suppression (same as `BaryonFractionSigmoid`): AGN feedback evacuates
      gas from :math:`M \sim 10^{13}` M\ :sub:`sun`\ /h halos.
    * Low-mass upturn (:math:`M \lesssim 10^{11.5}` M\ :sub:`sun`\ /h): AGN feedback is
      weak in dwarf halos; cold CGM gas fraction rises again
      (EAGLE, IllustrisTNG, FLAMINGO; arXiv:2511.17313).
    * Non-zero floor :math:`f_b^{\rm min}`: retained CGM gas even in the deepest valley.

    Parameters
    ----------
    (passed at call time via ``params`` dict)

    Notes
    -----
    Default amplitudes are illustrative.  The upturn amplitude ``f_b_lo_amp``
    adds on top of the floor, so the low-mass asymptote is
    :math:`f_b^{\rm min} + f_b^{\rm lo,amp}`.
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        m_h: jnp.ndarray,
        theta_cosmo: dict,
        params: dict,
    ) -> jnp.ndarray:
        r"""Baryon fraction f_b(M) with group-scale valley and low-mass upturn.

        Parameters
        ----------
        m_h : jnp.ndarray
            Halo masses [Msun/h].
        theta_cosmo : dict
            Cosmological parameters (needs ``Omega_b``, ``Omega_m``).
        params : dict
            Model parameters: ``f_b_min``, ``log10_M_hi``, ``beta_hi``,
            ``f_b_lo_amp``, ``log10_M_lo``, ``beta_lo``.

        Returns
        -------
        f_b : jnp.ndarray, same shape as m_h.
        """
        f_b_cosmic = theta_cosmo["Omega_b"] / theta_cosmo["Omega_m"]
        f_b_min    = params["f_b_min"]
        M_hi       = 10.0 ** params["log10_M_hi"]
        M_lo       = 10.0 ** params["log10_M_lo"]
        sig_hi = 1.0 / (1.0 + (M_hi / m_h) ** params["beta_hi"])
        sig_lo = 1.0 / (1.0 + (m_h  / M_lo) ** params["beta_lo"])
        return f_b_min + (f_b_cosmic - f_b_min) * sig_hi + params["f_b_lo_amp"] * sig_lo

    @staticmethod
    def default_params() -> dict:
        """Default double-sigmoid valley parameters.

        Sources:

        * Group-scale pivot ``log10_M_hi = 13.5`` — same as `BaryonFractionSigmoid`
          (FLAMINGO / closure-radius model)
        * ``f_b_min = 0.01`` — 1% floor from CGM gas census
        * ``f_b_lo_amp = 0.05`` — illustrative upturn amplitude (~30% of cosmic
          f_b) in dwarf halos
        * ``log10_M_lo = 11.5`` — mass below which gas fraction rises (IllustrisTNG,
          FLAMINGO; arXiv:2511.17313 CGM survey at Milky Way mass)
        * ``beta_lo = 2.0`` — sharpness of the low-mass upturn
        """
        return {
            "f_b_min":       0.01,
            "log10_M_hi":    13.5,
            "beta_hi":       1.5,
            "f_b_lo_amp":    0.05,
            "log10_M_lo":    11.5,
            "beta_lo":       2.0,
        }


def make_baryon_fraction(model: str = "sigmoid"):
    """Factory returning the requested baryon fraction model.

    Parameters
    ----------
    model : {"sigmoid", "powerlaw", "upturn"}
        Model name (case-insensitive).  ``"upturn"`` also accepts
        ``"double_sigmoid"`` and ``"valley"``.

    Returns
    -------
    BaryonFractionSigmoid, BaryonFractionPowerLaw, or BaryonFractionUpturn instance.
    """
    model = model.lower()
    if model == "sigmoid":
        return BaryonFractionSigmoid()
    if model in ("powerlaw", "power_law", "pl"):
        return BaryonFractionPowerLaw()
    if model in ("upturn", "double_sigmoid", "valley"):
        return BaryonFractionUpturn()
    raise ValueError(
        f"Unknown baryon fraction model '{model}'. "
        "Choose 'sigmoid', 'powerlaw', or 'upturn'."
    )
