"""Non-linear matter power spectrum via the Aletheia or CSST CEmulator, or CAMB HALOFIT."""

import numpy as np
import jax
import jax.numpy as jnp


class NonLinearPowerSpectrum:
    """Non-linear P(k, z) via the Aletheia or CSST CEmulator.

    Both backends are numpy-based; this class loads the requested emulator once
    and exposes a JAX-compatible interface by converting outputs to jnp arrays.
    Gradients through the emulator are not available — use finite differences.

    Parameters
    ----------
    backend : {"aletheia", "csst"}
        ``"aletheia"`` — Sanchez 2025 (arXiv:2511.13826), valid k ∈ [0.006, 2] Mpc^-1.
        ``"csst"`` — Chen+2025 CEmulator v2.0, valid k ∈ [0.005, 10] h/Mpc,
        z ∈ [0, 3], nonlinear via HMcode-2020 boost ratio.
    """

    def __init__(self, backend: str = "aletheia"):
        if backend not in ("aletheia", "csst"):
            raise ValueError(f"backend must be 'aletheia' or 'csst', got '{backend}'")
        self.backend = backend
        self._emu = self._load()

    def _load(self):
        if self.backend == "aletheia":
            try:
                from aletheiacosmo import AletheiaEmu
            except ImportError as e:
                raise ImportError("aletheiacosmo not installed — pip install aletheiacosmo") from e
            return AletheiaEmu()
        else:  # csst
            try:
                from CEmulator.Emulator import Pkmm_CEmulator
            except ImportError as e:
                raise ImportError("CEmulator not installed") from e
            return Pkmm_CEmulator()

    # ------------------------------------------------------------------
    # Aletheia helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cosmo_aletheia(theta: dict) -> dict:
        """Convert hod_mod theta dict to Aletheia parameter dict."""
        from aletheiacosmo import AletheiaEmu
        h    = float(theta["h"])
        lnAs = float(theta["ln10^{10}A_s"])
        return AletheiaEmu.create_cosmo_dict(
            h       = h,
            omega_b = float(theta["Omega_b"])   * h ** 2,
            omega_c = float(theta["Omega_cdm"]) * h ** 2,
            n_s     = float(theta["n_s"]),
            A_s     = np.exp(lnAs) * 1e-10,
            model   = "LCDM",
        )

    # ------------------------------------------------------------------
    # CSST helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_cosmos_csst(emu, theta: dict) -> None:
        """Push hod_mod theta dict into the CEmulator cosmology state."""
        emu.set_cosmos(
            Omegab=float(theta["Omega_b"]),
            Omegac=float(theta["Omega_cdm"]),
            H0=float(theta["h"]) * 100.0,
            As=np.exp(float(theta["ln10^{10}A_s"])) * 1e-10,
            ns=float(theta["n_s"]),
            w=float(theta.get("w0", -1.0)),
            wa=float(theta.get("wa", 0.0)),
            mnu=float(theta.get("mnu", 0.06)),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pk_nonlinear(self, k: np.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """Non-linear P(k) [(Mpc/h)^3] at redshift z.

        Parameters
        ----------
        k : array_like, h/Mpc
        z : float
        theta : dict — hod_mod cosmological parameter dict

        Notes
        -----
        For the aletheia backend the emulator is valid for k ∈ [0.006, 2.0] Mpc⁻¹.
        k values outside this range fall back to the Eisenstein-Hu linear spectrum
        multiplied by the nonlinear boost evaluated at the nearest valid boundary.
        This ensures the function is defined over any k grid used by the clustering
        module without raising a ValueError from the emulator.
        """
        if self.backend == "aletheia":
            from hod_mod.cosmology.power_spectrum import eisenstein_hu_pk
            h = float(theta["h"])
            cosmo = self._build_cosmo_aletheia(theta)
            k_arr = np.asarray(k, dtype=float)
            k_mpc = k_arr * h      # h/Mpc → Mpc^-1

            K_MIN, K_MAX = 0.006, 2.0  # Mpc^-1 valid emulator range
            k_mpc_safe = np.clip(k_mpc, K_MIN, K_MAX)
            pk_mpc3 = self._emu.get_pnl(k_mpc_safe, cosmo, float(z))
            pk_h3 = pk_mpc3 * h ** 3

            out_of_range = (k_mpc < K_MIN) | (k_mpc > K_MAX)
            if np.any(out_of_range):
                # Linear E-H at original k and at the boundary k
                pk_lin = np.asarray(eisenstein_hu_pk(jnp.asarray(k_arr), theta))
                pk_lin_safe = np.asarray(eisenstein_hu_pk(jnp.asarray(k_mpc_safe / h), theta))
                # Boost = P_nl / P_lin at boundary; use it to extend outside valid range
                boost = pk_h3 / np.where(pk_lin_safe > 0, pk_lin_safe, 1.0)
                pk_h3 = np.where(out_of_range, pk_lin * boost, pk_h3)

            return jnp.asarray(pk_h3)
        else:  # csst
            self._set_cosmos_csst(self._emu, theta)
            k_np = np.asarray(k)
            pk2d = self._emu.get_pknl(z=float(z), k=k_np)  # shape (1, len(k))
            return jnp.asarray(pk2d[0])

    def pk_nonlinear_jax(
        self, k: jnp.ndarray, z: float, theta: dict
    ) -> jnp.ndarray:
        """JAX-native P_nl(k) [(Mpc/h)^3] for the Aletheia backend.

        Unlike :meth:`pk_nonlinear`, this method keeps all array operations in JAX
        so that ``jax.grad`` / ``jax.jacobian`` can differentiate through the
        emulator with respect to both ``k`` and the HOD/cosmological parameters
        encoded inside ``theta``.

        Only available for ``backend='aletheia'``.  For CSST or HMcode use
        :class:`CachedPkNonlinear` wrapping the numpy-based method.

        Parameters
        ----------
        k : jnp.ndarray [h/Mpc]
        z : float
        theta : dict — hod_mod cosmological parameter dict

        Returns
        -------
        jnp.ndarray [(Mpc/h)^3]
        """
        if self.backend != "aletheia":
            raise RuntimeError(
                "pk_nonlinear_jax is only available for backend='aletheia'; "
                f"current backend is '{self.backend}'"
            )
        from hod_mod.cosmology.power_spectrum import eisenstein_hu_pk

        h = float(theta["h"])
        cosmo = self._build_cosmo_aletheia(theta)

        K_MIN, K_MAX = 0.006, 2.0  # Mpc^-1 valid emulator range
        k_mpc = k * h              # h/Mpc → Mpc^-1 (stays in JAX trace)
        k_mpc_safe = jnp.clip(k_mpc, K_MIN, K_MAX)

        # Emulator call: returns jnp array when given jnp input
        pk_mpc3 = self._emu.get_pnl(k_mpc_safe, cosmo, float(z))
        pk_h3 = pk_mpc3 * h ** 3

        # Out-of-range extension via E-H boost (jnp.where evaluates both branches,
        # so the gradient is defined everywhere even when some k are out-of-range)
        out_of_range = (k_mpc < K_MIN) | (k_mpc > K_MAX)
        pk_lin = eisenstein_hu_pk(k, theta)
        pk_lin_safe = eisenstein_hu_pk(k_mpc_safe / h, theta)
        boost = pk_h3 / jnp.where(pk_lin_safe > 0.0, pk_lin_safe, jnp.ones_like(pk_lin_safe))
        return jnp.where(out_of_range, pk_lin * boost, pk_h3)

    def boost_factor(
        self,
        k: jnp.ndarray,
        z: float,
        theta: dict,
        pk_lin: jnp.ndarray,
    ) -> jnp.ndarray:
        """Non-linear boost B(k, z) = P_nl / P_lin."""
        pk_nl = self.pk_nonlinear(np.asarray(k), z, theta)
        return pk_nl / pk_lin


class HALOFITSpectrum:
    """Non-linear P(k, z) via CAMB's built-in HALOFIT / HMcode variants.

    Uses CAMB (already installed) with ``NonLinear = NonLinear_pk``.
    The default variant is ``halofit_mead2020`` (HMcode-2020, arXiv:2009.01858),
    which includes a baryonic feedback option.

    Parameters
    ----------
    halofit_version : str
        CAMB HALOFIT variant name.  Common choices:

        ``"mead2020"``     — HMcode-2020 (Mead+2021, arXiv:2009.01858) **default**
        ``"mead2020_feedback"`` — HMcode-2020 with baryonic feedback
        ``"takahashi"``    — Takahashi+2012 (arXiv:1208.2701)
        ``"mead"``         — HMcode-2015 (arXiv:1602.02154)
        ``"original"``     — Smith+2003 original HALOFIT (arXiv:astro-ph/0207664)

    Notes
    -----
    Each call runs a full CAMB evaluation (~1–5 s).  Use the ``CachedPkLinear``
    pattern from CLAUDE.md to cache results in hot loops.
    Gradients through this class are not available.
    """

    def __init__(self, halofit_version: str = "mead2020"):
        self._version = halofit_version

    def pk_nonlinear(self, k: np.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """Non-linear P(k) [(h^{-1} Mpc)^3] from CAMB HALOFIT.

        Parameters
        ----------
        k : array_like, [h/Mpc]
        z : float
        theta : dict — hod_mod cosmological parameter dict

        Returns
        -------
        jnp.ndarray, shape (len(k),)
            P_nl(k) in (h^{-1} Mpc)^3, interpolated from the CAMB output grid.
        """
        import camb

        k_arr = np.asarray(k, dtype=float)
        h = float(theta["h"])
        lnAs = float(theta["ln10^{10}A_s"])

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=h * 100.0,
            ombh2=float(theta["Omega_b"])   * h ** 2,
            omch2=float(theta["Omega_cdm"]) * h ** 2,
        )
        pars.InitPower.set_params(
            As=np.exp(lnAs) * 1e-10,
            ns=float(theta["n_s"]),
        )
        pars.NonLinear = camb.model.NonLinear_pk
        pars.NonLinearModel.set_params(halofit_version=self._version)
        pars.set_matter_power(
            redshifts=[float(z)],
            kmax=float(k_arr.max()) * 1.2,
        )
        results = camb.get_results(pars)
        # get_matter_power_spectrum returns k in h/Mpc, P in (h^-1 Mpc)^3
        k_h, _, pk2d = results.get_matter_power_spectrum(
            minkh=float(k_arr.min()) * 0.8,
            maxkh=float(k_arr.max()) * 1.2,
            npoints=512,
        )
        pk_fid = pk2d[0]   # shape (512,) at z[0]
        # Interpolate log-log onto requested k grid
        log_pk = np.interp(np.log(k_arr), np.log(k_h), np.log(pk_fid))
        return jnp.asarray(np.exp(log_pk))


class CachedPkNonlinear:
    """Caching wrapper for any non-linear power spectrum backend.

    Duck-types any object exposing ``pk_nonlinear(k, z, theta) -> array_like``
    and caches the result on a fixed log-spaced k grid, keyed by
    ``(z, Omega_m, ln10As, h)``.  Subsequent calls with the same cosmology
    are returned via cheap log-log interpolation.

    Compatible backends:

    * :class:`NonLinearPowerSpectrum` with ``backend='aletheia'`` — Aletheia
      emulator (arXiv:2511.13826), fast and JAX-based.
    * :class:`NonLinearPowerSpectrum` with ``backend='csst'`` — CEmulator v2.0.
    * :class:`HALOFITSpectrum` — CAMB HALOFIT / HMcode variants.

    This mirrors the :class:`CachedPkLinear` pattern documented in CLAUDE.md;
    use it in MCMC hot loops to avoid repeated CAMB / emulator calls.

    Parameters
    ----------
    pk_nl_obj : object
        Any instance with a ``pk_nonlinear(k, z, theta)`` method.
    n_k : int
        Number of points in the internal interpolation grid
        (k ∈ [10⁻⁴, 20] h/Mpc).
    """

    def __init__(self, pk_nl_obj, n_k: int = 512):
        self._base   = pk_nl_obj
        self._k_ref  = np.logspace(-4, np.log10(20.0), n_k)
        self._lk_ref = np.log(self._k_ref)
        self._cache: dict = {}

    def _key(self, z: float, theta: dict) -> tuple:
        return (
            round(float(z), 4),
            round(float(theta["Omega_m"]), 5),
            round(float(theta["ln10^{10}A_s"]), 4),
            round(float(theta.get("h", 0.72)), 4),
        )

    def pk_nonlinear(self, k, z: float, theta: dict) -> jnp.ndarray:
        """Non-linear P_nl(k) [(Mpc/h)^3], log-log interpolated from cache.

        Parameters
        ----------
        k : array_like [h/Mpc]
        z : float
        theta : dict — hod_mod cosmological parameter dict

        Returns
        -------
        jnp.ndarray
        """
        key = self._key(z, theta)
        if key not in self._cache:
            pk_ref = np.asarray(
                self._base.pk_nonlinear(self._k_ref, float(z), theta)
            )
            self._cache[key] = np.log(np.maximum(pk_ref, 1e-50))
        lk = np.log(np.asarray(k, dtype=float))
        return jnp.asarray(np.exp(np.interp(lk, self._lk_ref, self._cache[key])))
