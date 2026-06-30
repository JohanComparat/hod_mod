"""ICM metallicity profile (DPM)."""
import numpy as np
from .conversions import _gnfw_f_params


# ---------------------------------------------------------------------------
# DPM metallicity profile
# ---------------------------------------------------------------------------

class MetallicityProfileDPM:
    """DPM gas metallicity profile (Oppenheimer+2025, arXiv:2505.14782, Eq. 4).

    All three DPM models share the same metallicity profile (Table 1):

    .. math::

        Z(r, M, z) = Z_0 \\, f(r/R_s \\mid \\alpha^Z)

    with :math:`\\alpha_{\\rm in}^Z = 0`, :math:`\\alpha_{\\rm tr}^Z = 0.5`,
    :math:`\\alpha_{\\rm out}^Z = 0.7`, :math:`\\beta^Z = 0`, :math:`\\gamma^Z = 0`
    (no mass or redshift dependence).  The normalisation is
    :math:`Z(0.3 R_{200}) = 0.3\\,Z_\\odot`.

    The same :data:`_C_DPM` = 2.772 scale radius convention is used.

    This profile is used by :meth:`GasDensityDPM.emissivity_full_uk` to
    evaluate the metallicity-dependent X-ray cooling function Λ(T, Z).
    """

    _C_DPM     = 2.772
    _Z_03      = 0.3   # [Z_sun] at r=0.3 R_200 (all models, no M or z dependence)
    _ALPHA_IN  = 0.0
    _ALPHA_TR  = 0.5
    _ALPHA_OUT = 0.7

    def __init__(self):
        x_ref = 0.3 * self._C_DPM
        f_ref = _gnfw_f_params(x_ref, self._ALPHA_IN, self._ALPHA_TR, self._ALPHA_OUT)
        self._Z0 = self._Z_03 / float(f_ref)   # [Z_sun]

    def metallicity_3d(self, r: np.ndarray, r200: float) -> np.ndarray:
        """Gas metallicity Z(r) [Z_sun].

        No mass or redshift dependence (β^Z = γ^Z = 0).

        Parameters
        ----------
        r    : radii [Mpc/h]
        r200 : R₂₀₀ [Mpc/h]
        """
        r_s = r200 / self._C_DPM
        x   = np.asarray(r, dtype=float) / r_s
        return self._Z0 * _gnfw_f_params(x, self._ALPHA_IN, self._ALPHA_TR, self._ALPHA_OUT)
