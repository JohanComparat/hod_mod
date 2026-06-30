"""X-ray cooling/emissivity: APEC table, temperature and cooling function."""
import numpy as np


# ---------------------------------------------------------------------------
# Module-level helpers (X-ray temperature and cooling function)
# ---------------------------------------------------------------------------

def temperature_from_profiles(
    pressure: np.ndarray,
    n_electron: np.ndarray,
) -> np.ndarray:
    """Gas temperature T = P / (n_e k_B) [keV].

    Used internally by :meth:`GasDensityDPM.emissivity_full_uk` to build the
    temperature map from DPM pressure and density profiles (Section 3.1.1 of
    arXiv:2505.14782).

    Parameters
    ----------
    pressure   : P_e [keV cm⁻³] — from :class:`PressureProfileDPM`
    n_electron : n_e [cm⁻³]     — from :class:`GasDensityDPM`

    Returns
    -------
    T : [keV]
    """
    ne_safe = np.where(np.asarray(n_electron) > 1e-40, n_electron, 1e-40)
    return np.asarray(pressure) / ne_safe


def temperature_from_dpm(
    pressure_profile,
    density_profile,
    r: np.ndarray,
    m200: float,
    r200: float,
    z: float,
    theta_cosmo: dict,
) -> np.ndarray:
    """Gas temperature T(r, M, z) [keV] from DPM pressure and density profiles.

    Uses the ideal gas law T = P_e / (n_e k_B).  Convenience wrapper that
    evaluates both DPM profiles at the same radii.

    Parameters
    ----------
    pressure_profile  : PressureProfileDPM instance
    density_profile   : GasDensityDPM instance
    r     : radii [Mpc/h]
    m200  : M₂₀₀ [Msun/h]
    r200  : R₂₀₀ [Mpc/h]
    z     : redshift
    theta_cosmo : dict with key 'Omega_m'

    Returns
    -------
    T : (Nr,) [keV]
    """
    omega_m = float(theta_cosmo["Omega_m"])
    P  = pressure_profile._pressure_3d(r, m200, r200, z, omega_m)
    ne = density_profile.density_3d(r, m200, r200, z, omega_m)
    return temperature_from_profiles(P, ne)


def xray_cooling_function(
    T_keV: np.ndarray,
    Z_solar: np.ndarray,
    alpha_T: float = 0.5,
    alpha_Z: float = 1.0,
    Lambda_0: float = 3e-23,
) -> np.ndarray:
    """Simplified power-law cooling function Λ(T, Z) [erg cm³ s⁻¹].

    .. deprecated::
        Use :class:`ApecCoolingTable` instead, which evaluates the full
        APEC plasma code tables (soxs) over the specified energy band.

    Models the 0.5–2 keV soft X-ray volume emissivity coefficient:

    .. math::

        \\Lambda(T, Z) = \\Lambda_0
            \\left(\\frac{T}{1\\,\\text{keV}}\\right)^{\\alpha_T}
            \\left(\\frac{Z}{0.3\\,Z_\\odot}\\right)^{\\alpha_Z}
    """
    import warnings
    warnings.warn(
        "xray_cooling_function is deprecated — use ApecCoolingTable instead.",
        DeprecationWarning, stacklevel=2,
    )
    T = np.asarray(T_keV,   dtype=float)
    Z = np.asarray(Z_solar, dtype=float)
    T_safe = np.where(T > 1e-6, T, 1e-6)
    Z_safe = np.where(Z > 1e-6, Z, 1e-6)
    return Lambda_0 * (T_safe / 1.0) ** alpha_T * (Z_safe / 0.3) ** alpha_Z


class ApecCoolingTable:
    """Band-integrated APEC cooling function Λ(T, Z) precomputed as a 2D table.

    Uses :class:`soxs.ApecGenerator` to compute the X-ray emission spectrum from
    the AtomDB APEC CIE plasma model for a grid of temperatures and metallicities,
    integrates each spectrum over the requested energy band, and stores the result
    as a 2D log-log interpolator for fast evaluation.

    The output follows the APEC normalization convention:

    .. math::

        \\varepsilon(r) = n_e(r)\\,n_H(r)\\,\\Lambda(T(r), Z(r))

    with :math:`n_H \\approx 0.83\\,n_e` (fully ionized solar-abundance plasma).
    The stored table gives :math:`\\Lambda_{n_e^2}(T, Z) = 0.83\\,\\Lambda_{\\rm APEC}`,
    so that :math:`\\varepsilon = n_e^2\\,\\Lambda_{n_e^2}`.

    Parameters
    ----------
    emin, emax : float
        Energy band edges [keV].  Default: 0.5–2.0 (soft X-ray).
    n_T : int
        Number of temperature grid points (log-spaced).  Default: 60.
    T_min, T_max : float
        Temperature range [keV].  Default: 0.08–20.
    n_Z : int
        Number of metallicity grid points (log-spaced).  Default: 15.
    Z_min, Z_max : float
        Metallicity range [Z_sun].  Default: 0.05–3.0.
    apec_vers : str
        APEC version string (default from soxs config, currently "3.1.3").
    nbins : int
        Number of spectral bins used for the band integration.  Default: 1000.

    Notes
    -----
    Requires ``soxs`` (``pip install pyxsim soxs``) and the APEC spectral tables,
    which are downloaded once via ``soxs.download_spectrum_tables("apec")``.

    Initialisation takes a few seconds (60×15 = 900 APEC evaluations).
    Cache the instance across calls.
    """

    _NH_RATIO = 0.83   # n_H / n_e for fully ionized solar-abundance plasma

    def __init__(
        self,
        emin: float = 0.5,
        emax: float = 2.0,
        n_T: int = 60,
        T_min: float = 0.08,
        T_max: float = 20.0,
        n_Z: int = 15,
        Z_min: float = 0.05,
        Z_max: float = 3.0,
        apec_vers: str = None,
        nbins: int = 1000,
    ):
        try:
            import soxs
            from scipy.interpolate import RegularGridInterpolator
        except ImportError as e:
            raise ImportError(
                "soxs not installed — run: pip install pyxsim soxs"
            ) from e

        self._emin = emin
        self._emax = emax

        T_grid = np.logspace(np.log10(T_min), np.log10(T_max), n_T)   # [keV]
        Z_grid = np.logspace(np.log10(Z_min), np.log10(Z_max), n_Z)   # [Z_sun]

        kwargs = dict(broadening=False)
        if apec_vers is not None:
            kwargs["apec_vers"] = apec_vers
        apec = soxs.ApecGenerator(emin, emax, nbins, **kwargs)

        # Precompute Λ_{n_e²}(T_i, Z_j)  [erg cm³ s⁻¹]
        # APEC convention: norm = 1e-14 × EM / (4π D_A²)
        # For norm=1, D_A=1 cm, z=0: EM = 4π × 1e14 cm⁻⁵
        # flux F [erg/s/cm²] = Λ_APEC × EM / (4π D_A²) = Λ_APEC × 1e14
        # → Λ_APEC = F.value / 1e14   [erg cm³/s w.r.t. n_e n_H]
        # → Λ_{n_e²} = 0.83 × Λ_APEC
        table = np.empty((n_T, n_Z))
        for i, kT in enumerate(T_grid):
            for j, Z in enumerate(Z_grid):
                spec = apec.get_spectrum(kT, Z, redshift=0.0, norm=1.0)
                table[i, j] = float(spec.total_energy_flux.value) / 1e14 * self._NH_RATIO

        self._T_grid = T_grid
        self._Z_grid = Z_grid
        self._table  = table
        self._interp = RegularGridInterpolator(
            (np.log10(T_grid), np.log10(Z_grid)),
            np.log10(np.maximum(table, 1e-60)),
            method="linear",
            bounds_error=False,
            fill_value=None,   # linear extrapolation beyond grid edges
        )

    def __call__(self, T_keV: np.ndarray, Z_solar: np.ndarray) -> np.ndarray:
        """Λ_{n_e²}(T, Z) [erg cm³ s⁻¹] by 2D log-log interpolation.

        Parameters
        ----------
        T_keV   : temperature [keV]
        Z_solar : metallicity [Z_sun]

        Returns
        -------
        Lambda : same shape as T_keV  [erg cm³ s⁻¹]
        """
        T = np.asarray(T_keV,   dtype=float)
        Z = np.asarray(Z_solar, dtype=float)
        pts = np.column_stack([
            np.log10(np.maximum(T.ravel(), 1e-6)),
            np.log10(np.maximum(Z.ravel(), 1e-6)),
        ])
        return 10.0 ** self._interp(pts).reshape(T.shape)
