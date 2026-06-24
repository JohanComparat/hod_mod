"""Beyond-linear halo bias correction — Mead & Verde (2021).

Reference: Mead & Verde, MNRAS 503, 3 (2021), arXiv:2011.08858
Data source: https://github.com/alexander-mead/BNL
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "bnl"
_DEFAULT_SNAP = "85"


class BeyondLinearBiasMead21:
    """Beyond-linear halo bias correction from Mead & Verde (2021), arXiv:2011.08858.

    Loads tabulated beta^NL(k, nu1, nu2) measured from the MultiDark MDR1
    N-body simulation (Rockstar halos, M512 mass threshold, snapshot 85 = z=0)
    and provides interpolation onto arbitrary (k, nu) grids.

    The correction enters the two-halo galaxy power spectra as an additive term:

        P_gg^{2h,BNL}(k) = b_eff^2 P_lin(k)  +  P_lin(k) * delta_gg(k)
        P_gm^{2h,BNL}(k) = b_eff   P_lin(k)  +  P_lin(k) * delta_gm(k)

    where delta_gg/gm are k-dependent double integrals over the halo mass
    function weighted by beta^NL(k, nu1, nu2).

    **Memory-efficient implementation**: instead of building the full
    (Nk, NM, NM) beta^NL matrix, we project weights onto the coarse 8-point
    peak-height grid (Nk, 8) and contract with the small (Nk, 8, 8) table.
    This reduces memory from O(Nk × NM^2) to O(Nk × NM × 8), with identical
    results because linear interpolation factorises over the two nu axes.

    The tabulated k range is [6.3e-3, 0.74] h/Mpc across 8 peak-height bins.
    beta^NL is set to 0 outside this range (large-scale and low/high-nu safety).

    Parameters
    ----------
    data_dir : path-like or None
        Directory containing the BNL .dat files.  Defaults to the bundled
        ``hod_mod/data/bnl/`` directory.
    snap : str
        Snapshot identifier suffix, e.g. "85" for z=0.
    """

    def __init__(self, data_dir=None, snap=_DEFAULT_SNAP):
        if data_dir is None:
            data_dir = _DEFAULT_DATA_DIR
        data_dir = Path(data_dir)

        # --- binstats: 8 rows x 8 cols ---
        # cols: log10(m_min) log10(m_max) log10(m) nu_min nu_max nu b rv
        bs = np.loadtxt(data_dir / f"MDR1_rockstar_{snap}_binstats.dat")
        self._nu_ref = bs[:, 5]   # representative nu per mass bin (8 values)
        self._b_ref = bs[:, 6]    # linear bias per mass bin

        # --- bnl: 1600 rows x 2 cols ---
        # Layout: 64 (8x8) mass-bin pairs x 25 k-values, ordered (0,0),(0,1),...,(7,7)
        # cols: k [h/Mpc],  1 + beta^NL
        raw = np.loadtxt(data_dir / f"MDR1_rockstar_{snap}_bnl.dat")
        # reshape (1600, 2) → (64, 25, 2) → separate k and 1+β^NL
        raw = raw.reshape(64, 25, 2)
        self._k_ref = raw[0, :, 0]                          # (25,) same for all pairs
        one_plus_bnl = raw[:, :, 1].reshape(8, 8, 25)       # (8, 8, 25)

        # beta^NL array in layout (n_k, n_nu1, n_nu2) for interpolator
        self._beta_nl_grid = one_plus_bnl.transpose(2, 0, 1) - 1.0   # (25, 8, 8)
        self._log_k_ref = np.log(self._k_ref)

        # JAX arrays for computation
        self._nu_ref_jax       = jnp.asarray(self._nu_ref)
        self._log_k_ref_jax    = jnp.asarray(self._log_k_ref)
        self._beta_nl_grid_jax = jnp.asarray(self._beta_nl_grid)

        # 3-D linear interpolator in (log_k, nu1, nu2).
        # fill_value=0 → beta^NL = 0 outside the tabulated range.
        self._interp = RegularGridInterpolator(
            (self._log_k_ref_jax, self._nu_ref_jax, self._nu_ref_jax),
            self._beta_nl_grid_jax,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    # ------------------------------------------------------------------
    # Public interpolation: arbitrary (k, nu1, nu2)
    # ------------------------------------------------------------------

    def beta_nl(
        self,
        k: np.ndarray,
        nu1: np.ndarray,
        nu2: np.ndarray,
    ) -> jnp.ndarray:
        """Interpolate beta^NL(k, nu1, nu2).

        Parameters
        ----------
        k   : array-like, shape (Nk,)   wavenumbers [h/Mpc]
        nu1 : array-like, shape (N1,)   peak heights for first halo type
        nu2 : array-like, shape (N2,)   peak heights for second halo type

        Returns
        -------
        ndarray, shape (Nk, N1, N2)
        """
        k   = jnp.asarray(k)
        nu1 = jnp.asarray(nu1)
        nu2 = jnp.asarray(nu2)

        lk = jnp.log(k)
        LK, NU1, NU2 = jnp.meshgrid(lk, nu1, nu2, indexing="ij")
        pts = jnp.stack([LK.ravel(), NU1.ravel(), NU2.ravel()], axis=1)
        return self._interp(pts).reshape(len(k), len(nu1), len(nu2))

    # ------------------------------------------------------------------
    # Coarse 8-bin matrix — cached per (k, cosmo) in clustering.py
    # ------------------------------------------------------------------

    def beta_nl_matrix(
        self,
        k_arr: np.ndarray,
        nu_arr: np.ndarray = None,
    ) -> jnp.ndarray:
        """Return the beta^NL matrix on the coarse 8-bin nu grid.

        Returns beta^NL(k, nu_ref, nu_ref) of shape (Nk, 8, 8).  This small
        table is the only quantity that needs to be cached per cosmology; the
        contraction with mass-function weights is performed in the correction
        methods via projection onto the coarse nu grid.

        ``nu_arr`` is accepted but ignored — it exists only for API symmetry
        with callers that pass the fine mass-grid peak heights.

        Parameters
        ----------
        k_arr  : (Nk,) wavenumbers [h/Mpc]
        nu_arr : ignored

        Returns
        -------
        ndarray, shape (Nk, 8, 8)
        """
        return self.beta_nl(k_arr, self._nu_ref, self._nu_ref)

    # ------------------------------------------------------------------
    # Weight projection onto coarse nu grid (internal helper)
    # ------------------------------------------------------------------

    def _project_weights(
        self,
        nu_arr: np.ndarray,
        w_eff: np.ndarray,
    ) -> jnp.ndarray:
        """Project mass-grid weights onto the 8 BNL nu bins via linear interp.

        Parameters
        ----------
        nu_arr : (NM,)     peak heights on the fine halo mass grid
        w_eff  : (Nk, NM)  profile-weighted mass-function weights per k

        Returns
        -------
        W_coarse : (Nk, 8)  projected weights
        """
        nu_arr = jnp.asarray(nu_arr)
        w_eff  = jnp.asarray(w_eff)
        NM = nu_arr.shape[0]
        nb = len(self._nu_ref)

        # Vectorised linear interpolation: find left bin index for each nu
        idx = jnp.searchsorted(self._nu_ref_jax, nu_arr, side="right") - 1
        idx = jnp.clip(idx, 0, nb - 2)

        alpha = (nu_arr - self._nu_ref_jax[idx]) / (
            self._nu_ref_jax[idx + 1] - self._nu_ref_jax[idx]
        )

        # Scatter interpolation weights into (NM, nb) matrix
        row = jnp.arange(NM)
        phi = jnp.zeros((NM, nb))
        phi = phi.at[row, idx    ].set(1.0 - alpha)
        phi = phi.at[row, idx + 1].add(alpha)

        # Zero out entries outside the tabulated nu range
        mask = (nu_arr > self._nu_ref_jax[0]) & (nu_arr < self._nu_ref_jax[-1])
        phi = phi * mask[:, None]

        # W_coarse[k, a] = Σ_i phi[i, a] * w_eff[k, i]
        return w_eff @ phi   # (Nk, NM) @ (NM, 8) → (Nk, 8)

    # ------------------------------------------------------------------
    # Two-halo correction integrals (memory-efficient)
    # ------------------------------------------------------------------

    def correction_2h_gg(
        self,
        k_arr: np.ndarray,
        nu_arr: np.ndarray,
        weights: np.ndarray,
        uk: np.ndarray,
        beta_matrix: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """Additive BNL correction to P_gg^{2h}(k) / P_lin(k).

        Computes the double mass-integral:

            delta_gg(k) = sum_i sum_j w_i(k) w_j(k) beta^NL(k, nu_i, nu_j)

        via projection onto the coarse 8-bin nu grid (memory-efficient):

            W_a(k) = sum_i phi_a(nu_i) w_i(k)
            delta_gg(k) = sum_a sum_b W_a(k) W_b(k) beta^NL(k, nu_a, nu_b)

        Parameters
        ----------
        k_arr       : (Nk,)      wavenumbers [h/Mpc]
        nu_arr      : (NM,)      peak heights on the halo mass grid
        weights     : (NM,)      dndm * N_tot * b / n_gal  (WITH trapz dm factor)
        uk          : (Nk, NM)   halo profile Fourier transform
        beta_matrix : optional (Nk, 8, 8) pre-computed coarse beta^NL
                      (use self.beta_nl_matrix() to generate once per cosmo)

        Returns
        -------
        delta_gg : (Nk,)
        """
        k_arr   = jnp.asarray(k_arr)
        nu_arr  = jnp.asarray(nu_arr)
        weights = jnp.asarray(weights)
        uk      = jnp.asarray(uk)

        w_eff    = weights[None, :] * uk                        # (Nk, NM)
        W_coarse = self._project_weights(nu_arr, w_eff)         # (Nk, 8)

        if beta_matrix is None:
            beta_matrix = self.beta_nl_matrix(k_arr)            # (Nk, 8, 8)
        beta_matrix = jnp.asarray(beta_matrix)

        return jnp.einsum("ka,kab,kb->k", W_coarse, beta_matrix, W_coarse)

    def correction_2h_gm(
        self,
        k_arr: np.ndarray,
        nu_arr: np.ndarray,
        weights_g: np.ndarray,
        weights_m: np.ndarray,
        uk_g: np.ndarray,
        uk_m: np.ndarray,
        beta_matrix: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """Additive BNL correction to P_gm^{2h}(k) / P_lin(k).

        Computes the asymmetric double mass-integral:

            delta_gm(k) = sum_i sum_j wg_i(k) wm_j(k) beta^NL(k, nu_i, nu_j)

        via projection onto the coarse 8-bin nu grid.

        Parameters
        ----------
        k_arr      : (Nk,)
        nu_arr     : (NM,)
        weights_g  : (NM,)   galaxy side: dndm * N_tot * b / n_gal  (WITH dm)
        weights_m  : (NM,)   matter side: dndm * M * b / rho_m      (WITH dm)
        uk_g       : (Nk, NM)
        uk_m       : (Nk, NM)
        beta_matrix: optional (Nk, 8, 8)

        Returns
        -------
        delta_gm : (Nk,)
        """
        k_arr     = jnp.asarray(k_arr)
        nu_arr    = jnp.asarray(nu_arr)
        weights_g = jnp.asarray(weights_g)
        weights_m = jnp.asarray(weights_m)
        uk_g      = jnp.asarray(uk_g)
        uk_m      = jnp.asarray(uk_m)

        wg_eff = weights_g[None, :] * uk_g                      # (Nk, NM)
        wm_eff = weights_m[None, :] * uk_m                      # (Nk, NM)

        Wg_coarse = self._project_weights(nu_arr, wg_eff)       # (Nk, 8)
        Wm_coarse = self._project_weights(nu_arr, wm_eff)       # (Nk, 8)

        if beta_matrix is None:
            beta_matrix = self.beta_nl_matrix(k_arr)            # (Nk, 8, 8)
        beta_matrix = jnp.asarray(beta_matrix)

        return jnp.einsum("ka,kab,kb->k", Wg_coarse, beta_matrix, Wm_coarse)
