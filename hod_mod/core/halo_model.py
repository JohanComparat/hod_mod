"""Full halo model matter power spectrum P_mm(k) = P^{1h}_mm + P^{2h}_mm.

Implements the Asgari et al. (2023) halo model for the matter auto-power spectrum.
The 1-halo term captures intra-halo (shot-noise) clustering; the 2-halo term
recovers linear clustering on large scales.

.. math::

    P^{1h}_{mm}(k) = \\frac{1}{\\bar{\\rho}_m^2}
                     \\int M^2\\, \\hat{u}_m^2(k,M)\\, n(M)\\, dM

    P^{2h}_{mm}(k) = P_{\\rm lin}(k)\\,
                     \\left[\\frac{1}{\\bar{\\rho}_m}
                     \\int M\\, \\hat{u}_m(k,M)\\, b(M)\\, n(M)\\, dM\\right]^2

References
----------
Asgari et al. 2023, arXiv:2303.08752 — halo model review (Eqs. 34–35)
Cooray & Sheth 2002, Phys.Rep. 372, 1 — NFW window function (Eq. 11)
"""

import numpy as np
import jax
import jax.numpy as jnp

from .halo_profiles import nfw_uk, nfw_uk_jax

_RHO_CRIT0 = 2.775e11   # ρ_crit,0 in (M_sun/h) / (Mpc/h)³


class HaloModelPowerSpectrum:
    """Matter power spectrum P_mm(k) from the halo model.

    Combines a 1-halo and 2-halo term using NFW profile window functions
    and the chosen halo mass function and bias.

    .. math::

        P^{1h}_{mm}(k) = \\frac{1}{\\bar{\\rho}_m^2}
                         \\int M^2\\, \\hat{u}_m^2(k,M)\\, n(M)\\, dM

        P^{2h}_{mm}(k) = P_{\\rm lin}(k)\\,
                         \\left[\\frac{1}{\\bar{\\rho}_m}
                         \\int M\\, \\hat{u}_m(k,M)\\, b(M)\\, n(M)\\, dM\\right]^2

    where :math:`\\hat{u}_m(k,M)` is the NFW normalized Fourier transform
    (Cooray & Sheth 2002 Eq. 11, implemented in ``nfw_uk``), :math:`n(M)` is
    the halo mass function, :math:`b(M)` is the linear halo bias, and
    :math:`\\bar{\\rho}_m = \\Omega_m \\rho_{\\rm crit,0}`.

    On large scales (k → 0): :math:`\\hat{u}_m → 1` so the 2-halo integral
    → 1 (by the mass-weighted bias normalization), recovering
    :math:`P^{2h}_{mm} → P_{\\rm lin}` as expected.

    Parameters
    ----------
    hmf : HaloMassFunction
        Provides ``dndm(m, z, theta)`` and ``bias(m, z, theta)``.
    halo_profile : HaloProfile
        Provides ``rho_s_and_rs`` and ``concentration`` (colossus c–M relation).
    pk_lin : LinearPowerSpectrum
        Provides ``pk_linear(k, z, theta)`` for the 2-halo term.
    m_min, m_max : float [M_sun/h]
        Mass integration limits.
    n_m : int
        Number of log-spaced mass bins.
    """

    def __init__(
        self,
        hmf,
        halo_profile,
        pk_lin,
        m_min: float = 1e10,
        m_max: float = 1e16,
        n_m: int = 100,
    ):
        self._hmf = hmf
        self._prof = halo_profile
        self._pk_lin = pk_lin
        self._m = np.logspace(np.log10(m_min), np.log10(m_max), n_m)

    def _profile_arrays(
        self,
        z: float,
        theta: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (r_s, c) arrays for all mass bins at redshift z.

        Uses jax.vmap when the profile uses the JAX-native dutton14 c-M
        relation, giving a fully vectorised (loop-free) implementation.
        Falls back to the original scalar loop for colossus-based relations.
        """
        if getattr(self._prof, "_conc_model", None) == "dutton14":
            m_jnp = jnp.asarray(self._m)
            c_arr = self._prof.concentration(m_jnp, z)  # (NM,) — vectorised
            _, r_s_arr = jax.vmap(
                lambda mi: self._prof.rho_s_and_rs(mi.reshape(1), z, theta)
            )(m_jnp)
            return np.asarray(r_s_arr.squeeze(-1)), np.asarray(c_arr)

        r_s_arr = np.empty(len(self._m))
        c_arr   = np.empty(len(self._m))
        for i, mi in enumerate(self._m):
            mi_jnp = jnp.array([mi])
            _, r_s_i = self._prof.rho_s_and_rs(mi_jnp, z, theta)
            c_i      = self._prof.concentration(mi_jnp, z)
            r_s_arr[i] = float(r_s_i[0])
            c_arr[i]   = float(c_i[0])
        return r_s_arr, c_arr

    def pk_1h_mm(
        self,
        k_arr: np.ndarray,
        z: float,
        theta: dict,
    ) -> jnp.ndarray:
        """1-halo matter power spectrum (Asgari+2023 Eq. 34).

        .. math::

            P^{1h}_{mm}(k) = \\frac{1}{\\bar{\\rho}_m^2}
                             \\int M^2\\, \\hat{u}_m^2(k,M)\\,
                             \\frac{dn}{dM}\\, dM

        Parameters
        ----------
        k_arr : [h/Mpc], shape (Nk,)

        Returns
        -------
        p1h : [(Mpc/h)³], shape (Nk,)
        """
        rho_m_bar = _RHO_CRIT0 * float(theta["Omega_m"])
        m = self._m

        r_s_arr, c_arr = self._profile_arrays(z, theta)
        _use_jax = getattr(self._prof, "_conc_model", None) == "dutton14"
        if _use_jax:
            uk = nfw_uk_jax(jnp.asarray(k_arr), jnp.asarray(r_s_arr), jnp.asarray(c_arr))
        else:
            uk = jnp.asarray(nfw_uk(k_arr, r_s_arr, c_arr))       # (Nk, NM)
        dndm = self._hmf.dndm(jnp.asarray(m), z, theta)            # (NM,)

        integrand = (jnp.asarray(m) ** 2) * dndm                   # (NM,)
        p1h = jnp.trapezoid(uk ** 2 * integrand[None, :], jnp.asarray(m), axis=1) / rho_m_bar ** 2
        return p1h

    def pk_2h_mm(
        self,
        k_arr: np.ndarray,
        z: float,
        theta: dict,
    ) -> jnp.ndarray:
        """2-halo matter power spectrum (Asgari+2023 Eq. 35).

        .. math::

            P^{2h}_{mm}(k) = P_{\\rm lin}(k)\\,
                             \\left[\\frac{1}{\\bar{\\rho}_m}
                             \\int M\\, \\hat{u}_m(k,M)\\, b(M)\\,
                             \\frac{dn}{dM}\\, dM\\right]^2

        Parameters
        ----------
        k_arr : [h/Mpc], shape (Nk,)

        Returns
        -------
        p2h : [(Mpc/h)³], shape (Nk,)
        """
        rho_m_bar = _RHO_CRIT0 * float(theta["Omega_m"])
        m = self._m

        r_s_arr, c_arr = self._profile_arrays(z, theta)
        _use_jax = getattr(self._prof, "_conc_model", None) == "dutton14"
        if _use_jax:
            uk = nfw_uk_jax(jnp.asarray(k_arr), jnp.asarray(r_s_arr), jnp.asarray(c_arr))
        else:
            uk = jnp.asarray(nfw_uk(k_arr, r_s_arr, c_arr))        # (Nk, NM)
        dndm = self._hmf.dndm(jnp.asarray(m), z, theta)             # (NM,)
        bias = self._hmf.bias(jnp.asarray(m), z, theta)             # (NM,)

        integrand = jnp.asarray(m) * dndm * bias                    # (NM,)
        I_k = jnp.trapezoid(uk * integrand[None, :], jnp.asarray(m), axis=1) / rho_m_bar

        pk_lin = self._pk_lin.pk_linear(jnp.asarray(k_arr), z, theta)
        return pk_lin * I_k ** 2

    def pk_mm(
        self,
        k_arr: np.ndarray,
        z: float,
        theta: dict,
    ) -> jnp.ndarray:
        """Total matter power spectrum P_mm = P^{1h}_mm + P^{2h}_mm.

        Parameters
        ----------
        k_arr : [h/Mpc], shape (Nk,)

        Returns
        -------
        pk : [(Mpc/h)³], shape (Nk,)
        """
        return self.pk_1h_mm(k_arr, z, theta) + self.pk_2h_mm(k_arr, z, theta)
