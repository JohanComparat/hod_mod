"""HOD-based X-ray AGN model with modified abundance matching.

This is a third AGN X-ray model, conceptually distinct from
:class:`~hod_mod.galaxies.agn_ham.HamAGNModel` (which abundance-matches halo
mass directly to L_X) and :class:`~hod_mod.galaxies.agn.XrayAGNModel` (a
parametric L_X(M_*)).  Here AGN are placed by an explicit **halo occupation
distribution** and their luminosities are assigned by abundance matching against
a flux/optically-selected X-ray luminosity function.

Pipeline
--------
1. **AGN HOD** — a simple 5-parameter More+2015 occupation with a *constant*
   duty cycle ``f_inc`` (mass-independent), see
   :class:`~hod_mod.galaxies.hod.MoreConstFincHODModel`.  This populates halos
   with central and satellite AGN.
2. **Stellar masses** — the Zu & Mandelbaum (2015) SHMR turns the AGN-host halo
   masses into a stellar-mass distribution (centrals and satellites).
3. **Modified abundance matching** at the sample mean redshift ``z_mean``:

   - Build the luminosity distribution from the XLF (Aird+2015 by default) down
     to ``log10lx_min`` (1e39 erg/s).
   - Convert hard-band L_X to observed soft (0.5–2 keV) luminosity via the
     obscuration-weighted K-correction, then to observed flux ``FX`` with the
     luminosity distance.
   - Predict the r-band magnitude ``r_mag = a + b*log10(FX)`` (default
     ``a, b = -7, -2``) and keep ``16 ≤ r_mag ≤ 19.5``.
   - Rank-order match the selected L_X distribution onto the (f_inc-suppressed)
     AGN-host stellar-mass distribution.  Because ``f_inc`` is applied to the
     host population, the matched abundances agree mechanically — no rescaling
     of flux or luminosity is applied.

   The matching is performed deterministically on cumulative **number
   densities** (the noise-free limit of drawing a finite Monte-Carlo array;
   the sample volume cancels and only enters the absolute-count diagnostics).
4. **Outputs** — independent AGN occupations ``N_cen(AGN)``, ``N_sat(AGN)``, a
   monotonic ``log10(M_*) → log10(L_X^{0.5-2,obs})`` mapping, and the
   sample-averaged observed luminosity/flux.

The class exposes the same ``mean_agn_log10lx`` / ``mean_agn_lx`` /
``agn_emissivity_uk`` interface as the other AGN models (so it plugs into
:class:`~hod_mod.galaxies.cross_spectra.HaloModelCrossSpectra`) **plus**
``nc_ns_agn`` for the independent AGN occupation used by the X-ray
auto/cross-power spectra (following Lau et al. 2024, arXiv:2410.22397, App. A).

References
----------
More et al. 2015, ApJ 806, 2 (arXiv:1407.1856) — HOD form
Zu & Mandelbaum 2015, MNRAS 454, 1161 (arXiv:1505.02781) — SHMR
Aird et al. 2015, ApJ 815, 66 — XLF
Comparat et al. 2025, A&A 697, A173 — LS10-BGS samples S1…S7
Lau et al. 2024, arXiv:2410.22397 — X-ray power-spectrum formalism
"""

from __future__ import annotations

import logging

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d

from hod_mod.galaxies.agn_ham import (
    _XLF_FUNCS,
    _H_XLF,
    _ZU15_DEFAULT_SHMR,
    setup_kcorrection,
    mean_k_eff,
)
from hod_mod.galaxies.hod import (
    n_cen_more15_const_finc,
    n_sat_more15_const_finc,
    _mstar_from_mh_zu15,
)

_LOG = logging.getLogger(__name__)

_MPC_CM = 3.0857e24            # cm per Mpc (consistent with gas_profiles / cross_spectra)
_DEG2_PER_SKY = 41252.96125    # deg^2 in 4π sr

# LS10-BGS sample definitions (Comparat+2025 Table 1; see
# hod_mod/scripts/fitting/fit_comparat2025.py).  zmin defaults to 0.0.
BGS_SAMPLES = {
    "S1": dict(log10ms_min=10.00, z_mean=0.135, z_max=0.18),
    "S2": dict(log10ms_min=10.25, z_mean=0.162, z_max=0.22),
    "S3": dict(log10ms_min=10.50, z_mean=0.191, z_max=0.26),
    "S4": dict(log10ms_min=10.75, z_mean=0.226, z_max=0.31),
    "S5": dict(log10ms_min=11.00, z_mean=0.252, z_max=0.35),
    "S6": dict(log10ms_min=11.25, z_mean=0.255, z_max=0.35),
    "S7": dict(log10ms_min=11.50, z_mean=0.261, z_max=0.35),
}

# Default LS10 footprint solid angle [deg^2].  Only affects the absolute-count
# diagnostics (the L_X–M_* mapping is volume-independent).  Settable.
_LS10_AREA_DEG2 = 14000.0


class HODAgnModel:
    """HOD-based X-ray AGN model with flux/optically-selected abundance matching.

    Parameters
    ----------
    pk_lin : LinearPowerSpectrum, optional
        Linear power spectrum used to build the HMF.  Default: Planck 2018.
    theta_cosmo : dict, optional
        Cosmology dict.  Default: Planck 2018.
    hod_params : dict, optional
        AGN HOD parameters with keys ``log10mmin, sigma_logm, log10m1, alpha,
        kappa, f_inc``.  Default: the constant-f_inc More+2015 values
        (log10mmin=12.5, sigma_logm=0.8, alpha=0.8, log10m1=14.0, kappa=0.3,
        f_inc=0.1).  ``log10m1`` defaults to ``log10mmin + 1.5`` when omitted.
    shmr_params : dict, optional
        Zu & Mandelbaum 2015 SHMR parameters; default: Table 2 SDSS values.
    xlf : {'aird15', 'ueda14'}
        XLF reference (default ``'aird15'``).
    z_mean, z_min, z_max : float
        Sample mean redshift (where the matching is done) and redshift edges
        (for the volume diagnostic).
    sky_area_deg2 : float
        Survey solid angle [deg^2] (absolute-count diagnostic only).
    log10lx_min : float
        Faint luminosity floor for the XLF array [log10 erg/s], default 39.0.
    alpha_ox_coeffs : (a, b)
        r_mag = a + b*log10(FX).  Default (-7, -2).
    r_mag_range : (r_faint_bright, r_faint_faint)
        Optical selection window, default (16.0, 19.5).
    kcorr_path : str, optional
        Override path to the K-correction table.
    n_lx_grid : int
        Number of luminosity grid points.
    n_m_grid : int
        Number of halo-mass grid points for the host population.
    f_sat_agn : float
        Kept for interface back-compat only; the HOD path ignores it because
        ``N_sat(AGN)`` already encodes the satellite AGN content.
    """

    def __init__(
        self,
        pk_lin=None,
        theta_cosmo: dict | None = None,
        hod_params: dict | None = None,
        shmr_params: dict | None = None,
        xlf: str = "aird15",
        z_mean: float = 0.135,
        z_min: float = 0.0,
        z_max: float = 0.18,
        sky_area_deg2: float = _LS10_AREA_DEG2,
        log10lx_min: float = 39.0,
        alpha_ox_coeffs: tuple[float, float] = (-7.0, -2.0),
        r_mag_range: tuple[float, float] = (16.0, 19.5),
        kcorr_path: str | None = None,
        n_lx_grid: int = 600,
        n_m_grid: int = 600,
        f_sat_agn: float = 0.10,
        hmf=None,
    ):
        if xlf not in _XLF_FUNCS:
            raise ValueError(f"xlf must be 'aird15' or 'ueda14', got '{xlf}'")
        self._xlf_name = xlf
        self._xlf_func = _XLF_FUNCS[xlf]

        # -- Cosmology
        if theta_cosmo is None:
            from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
            theta_cosmo = LinearPowerSpectrum.default_cosmology()
        self._theta_cosmo = theta_cosmo

        # -- Power spectrum / HMF
        if pk_lin is None:
            from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
            pk_lin = LinearPowerSpectrum()
        # Reuse the caller's HMF (e.g. the CSST emulator used by the fit) for a
        # consistent abundance-match; fall back to Tinker08 only if none given.
        if hmf is not None:
            self._hmf = hmf
        else:
            from hod_mod.cosmology.halo_mass_function import make_hmf
            self._hmf = make_hmf("tinker08", pk_func=pk_lin.pk_linear)

        # -- HOD parameters (constant-f_inc More+2015)
        p = {
            "log10mmin": 12.5,
            "sigma_logm": 0.8,
            "alpha": 0.8,
            "kappa": 0.3,
            "f_inc": 0.1,
        }
        if hod_params is not None:
            p.update(hod_params)
        p.setdefault("log10m1", p["log10mmin"] + 1.5)
        self._hod_params = p

        # -- SHMR parameters
        self._shmr_params = dict(_ZU15_DEFAULT_SHMR)
        if shmr_params is not None:
            self._shmr_params.update(shmr_params)

        # -- Sample / selection configuration
        self._z_mean = float(z_mean)
        self._z_min = float(z_min)
        self._z_max = float(z_max)
        self._sky_area_deg2 = float(sky_area_deg2)
        self._log10lx_min = float(log10lx_min)
        self._alpha_ox_coeffs = (float(alpha_ox_coeffs[0]), float(alpha_ox_coeffs[1]))
        self._r_mag_range = (float(r_mag_range[0]), float(r_mag_range[1]))
        self._n_lx_grid = int(n_lx_grid)
        self._n_m_grid = int(n_m_grid)
        self._f_sat_agn = float(f_sat_agn)   # back-compat only; unused by HOD path

        # -- K-correction table
        self._kcorr_interp, self._kcorr_mode = setup_kcorrection(kcorr_path)

        # -- Run the abundance-matching pipeline (sets self._lx_am etc.)
        _LOG.info("HODAgnModel: running abundance matching at z=%.3f (xlf=%s) …",
                  self._z_mean, xlf)
        self._build_abundance_match()
        _LOG.info("HODAgnModel: done. mean log10 LX_soft=%.2f, mean FX=%.3e, "
                  "clamped host fraction=%.2f.",
                  self._mean_log10lx, self._mean_fx, self._frac_clamped)

    # ------------------------------------------------------------------
    # AGN occupation
    # ------------------------------------------------------------------

    def nc_ns_agn(self, log10m_arr, hod_params: dict | None = None):
        """Return (N_cen_AGN, N_sat_AGN) on *log10m_arr* (f_inc applied)."""
        p = hod_params if hod_params is not None else self._hod_params
        log10m = jnp.asarray(log10m_arr)
        nc = n_cen_more15_const_finc(log10m, p["log10mmin"], p["sigma_logm"],
                                     p["f_inc"])
        ns = n_sat_more15_const_finc(log10m, p["log10mmin"], p["sigma_logm"],
                                     p["log10m1"], p["alpha"], p["kappa"],
                                     p["f_inc"])
        return np.asarray(nc, dtype=np.float64), np.asarray(ns, dtype=np.float64)

    # ------------------------------------------------------------------
    # Abundance matching
    # ------------------------------------------------------------------

    def _luminosity_distance_cm(self, z: float) -> float:
        from hod_mod.cosmology.distances import luminosity_distance
        th = self._theta_cosmo
        dl_mpc = float(np.atleast_1d(np.asarray(luminosity_distance(
            jnp.atleast_1d(jnp.asarray(float(z))), th["h"], th["Omega_m"],
            th.get("w0", -1.0), th.get("wa", 0.0),
        )))[0])
        return dl_mpc * _MPC_CM

    def _comoving_volume_h3(self, z: float) -> float:
        """Comoving volume within z over 4π sr, in (Mpc/h)^3."""
        from hod_mod.cosmology.distances import comoving_volume
        th = self._theta_cosmo
        vc_mpc3 = float(np.atleast_1d(np.asarray(comoving_volume(
            jnp.atleast_1d(jnp.asarray(float(z))), th["h"], th["Omega_m"],
            th.get("w0", -1.0), th.get("wa", 0.0),
        )))[0])
        return vc_mpc3 * th["h"] ** 3   # Mpc^3 → (Mpc/h)^3

    def _build_abundance_match(self) -> None:
        z = self._z_mean
        th = self._theta_cosmo
        h = float(th["h"])
        h_factor = (_H_XLF / h) ** 3   # XLF Mpc^-3 (h=0.70) → (Mpc/h)^-3

        # ---- 1. Luminosity (hard-band) grid + XLF density ----
        log10lx_hard = np.linspace(self._log10lx_min, 47.0, self._n_lx_grid)
        dlx = np.gradient(log10lx_hard)
        phi_h3 = np.asarray(self._xlf_func(log10lx_hard, z)) * h_factor  # (Mpc/h)^-3 dex^-1

        # ---- 2. Hard → observed soft luminosity and flux ----
        k_eff = np.asarray(mean_k_eff(self._kcorr_interp, self._kcorr_mode,
                                      log10lx_hard, z))
        k_eff = np.clip(k_eff, 1e-30, 1.0)
        log10lx_soft = log10lx_hard + np.log10(k_eff)        # observed 0.5-2 keV
        dl_cm = self._luminosity_distance_cm(z)
        fx = 10.0 ** log10lx_soft / (4.0 * np.pi * dl_cm ** 2)   # erg/s/cm^2

        # ---- 3. Alpha_OX → r-band magnitude and optical selection ----
        a, b = self._alpha_ox_coeffs
        r_mag = a + b * np.log10(fx)
        r_lo, r_hi = self._r_mag_range
        sel = (r_mag >= r_lo) & (r_mag <= r_hi)

        # Cumulative selected XLF density n(>=L_X), bright → faint
        dn_lx = phi_h3 * dlx * sel
        n_lx_cumul = np.cumsum(dn_lx[::-1])[::-1]            # (Mpc/h)^-3
        self._n_lx_selected = float(n_lx_cumul.max()) if n_lx_cumul.size else 0.0

        # ---- 4. AGN-host stellar-mass population (f_inc applied) ----
        log10m = np.linspace(10.0, 15.5, self._n_m_grid)
        m = 10.0 ** log10m
        dlogm = np.gradient(log10m)
        dndm = np.asarray(self._hmf.dndm(jnp.asarray(m), z, th))   # (Mpc/h)^-3 / (Msun/h)
        nc, ns = self.nc_ns_agn(log10m)
        # number density of AGN hosts per log-mass bin
        dn_host = dndm * m * np.log(10.0) * (nc + ns) * dlogm      # (Mpc/h)^-3
        n_host_cumul = np.cumsum(dn_host[::-1])[::-1]              # n(>=M)
        self._n_agn_host = float(n_host_cumul.max()) if n_host_cumul.size else 0.0

        # Stellar mass of each host halo (Zu & Mandelbaum 2015 SHMR), monotonic
        log10mstar = np.asarray(_mstar_from_mh_zu15(
            jnp.asarray(log10m),
            self._shmr_params["lg_m1h"], self._shmr_params["lg_m0star"],
            self._shmr_params["beta"], self._shmr_params["delta"],
            self._shmr_params["gamma"],
        ))

        # ---- 5. Rank-order (abundance) matching ----
        # Map a cumulative density level → observed soft log10 L_X, using the
        # selected portion of the XLF (where dn_lx > 0).
        sel_idx = np.where(dn_lx > 0)[0]
        if sel_idx.size < 2:
            raise RuntimeError(
                "HODAgnModel: optical/flux selection retained < 2 luminosity "
                "bins — check z_mean, alpha_ox_coeffs, and r_mag_range."
            )
        x_cum = n_lx_cumul[sel_idx]            # decreasing as L_X increases
        y_lx = log10lx_soft[sel_idx]           # observed soft luminosity to assign
        # sort by cumulative density ascending (→ luminosity descending)
        order = np.argsort(x_cum)
        x_cum_s = x_cum[order]
        y_lx_s = y_lx[order]
        # dedupe identical cumulative values for a strictly increasing x
        x_cum_s, uniq = np.unique(x_cum_s, return_index=True)
        y_lx_s = y_lx_s[uniq]
        lx_bright = float(y_lx_s[0])           # smallest cumul → brightest L_X
        lx_faint = float(y_lx_s[-1])           # largest cumul → faintest L_X

        cum_to_lx = interp1d(
            x_cum_s, y_lx_s, kind="linear",
            bounds_error=False, fill_value=(lx_bright, lx_faint),
        )
        log10lx_host = cum_to_lx(n_host_cumul)   # observed soft log10 L_X per host

        # Fraction of host weight whose matched L_X is clamped at the faint
        # selection edge (hosts more abundant than the selected AGN).
        clamped = n_host_cumul > x_cum_s[-1]
        w = dn_host
        self._frac_clamped = float(np.sum(w[clamped]) / np.sum(w)) if np.sum(w) > 0 else 0.0

        # ---- 6. Monotonic M_* → L_X mapping and sample averages ----
        # log10mstar increases with halo mass; ensure strictly increasing for interp
        ms_s, ums = np.unique(log10mstar, return_index=True)
        lx_s = log10lx_host[ums]
        self._lx_am = interp1d(
            ms_s, lx_s, kind="linear",
            bounds_error=False, fill_value=(float(lx_s[0]), float(lx_s[-1])),
        )

        lx_lin = 10.0 ** log10lx_host
        fx_host = lx_lin / (4.0 * np.pi * dl_cm ** 2)
        wsum = np.sum(w)
        self._mean_lx = float(np.sum(w * lx_lin) / wsum)
        self._mean_log10lx = float(np.log10(self._mean_lx))
        self._mean_fx = float(np.sum(w * fx_host) / wsum)

        # diagnostics
        self._frac_selected = float(np.sum(sel) / sel.size)
        self._dl_cm = dl_cm
        self._volume_h3 = (
            (self._sky_area_deg2 / _DEG2_PER_SKY)
            * (self._comoving_volume_h3(self._z_max) - self._comoving_volume_h3(self._z_min))
        )
        self._n_agn_count = self._n_agn_host * self._volume_h3
        self._lx_soft_floor = lx_faint
        self._lx_soft_ceil = lx_bright

    # ------------------------------------------------------------------
    # Interface methods (parity with HamAGNModel / XrayAGNModel)
    # ------------------------------------------------------------------

    def mean_agn_log10lx(self, m_halo_arr, z: float = None, shmr_params=None, **kw):
        """log10 of the mean observed soft (0.5–2 keV) L_X per halo [erg/s].

        Maps M_halo → M_* (Zu & Mandelbaum 2015 SHMR) → L_X via the abundance
        match.  The mapping is fixed at ``z_mean``; the ``z`` argument is
        accepted for interface compatibility but ignored.
        """
        m = np.asarray(m_halo_arr, dtype=np.float64)
        sp = dict(self._shmr_params)
        if shmr_params is not None:
            sp.update(shmr_params)
        log10mstar = np.asarray(_mstar_from_mh_zu15(
            jnp.log10(jnp.asarray(m)),
            sp["lg_m1h"], sp["lg_m0star"], sp["beta"], sp["delta"], sp["gamma"],
        ))
        return np.asarray(self._lx_am(log10mstar), dtype=np.float64)

    def mean_agn_lx(self, m_halo_arr, z: float = None, shmr_params=None, **kw):
        """Mean observed soft X-ray AGN luminosity per halo [erg/s]."""
        return np.power(10.0, self.mean_agn_log10lx(m_halo_arr, z, shmr_params, **kw))

    def agn_emissivity_uk(self, k_arr, m_halo_arr, z: float, theta_cosmo: dict,
                          shmr_params=None, **kw):
        """Fourier transform of the AGN X-ray emissivity (point-source, flat in k).

        Returns the mean luminosity **per occupied AGN**, normalized ``L_X/1e43``;
        the AGN occupation weighting is applied by the cross-spectra code.

        Returns
        -------
        uk_agn : (Nk, NM) float64 ndarray [L_X / 1e43, dimensionless]
        """
        k = np.asarray(k_arr, dtype=np.float64)
        m = np.asarray(m_halo_arr, dtype=np.float64)
        Nk = k.shape[0]
        log10_lx = self.mean_agn_log10lx(m, z, shmr_params)
        lx_norm = np.power(10.0, log10_lx - 43.0)
        return np.ones((Nk, 1), dtype=np.float64) * lx_norm[None, :]

    # ------------------------------------------------------------------
    # Sample-averaged predictions
    # ------------------------------------------------------------------

    def mean_observed_lx(self) -> float:
        """Host-population-averaged observed soft (0.5–2 keV) L_X [erg/s]."""
        return self._mean_lx

    def mean_observed_fx(self) -> float:
        """Host-population-averaged observed soft (0.5–2 keV) flux [erg/s/cm^2]."""
        return self._mean_fx
