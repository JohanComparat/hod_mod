"""Duty-cycle AGN model for the galaxy × AGN X-ray-emission cross-correlation.

This module predicts the galaxy × AGN X-ray-emission cross-correlation by
populating galaxies with AGN at a free duty cycle and weighting their X-ray
emission with a luminosity-function kernel.  It follows the Appendix-A
formalism of Lau et al. 2025 (ApJ 983, 8; arXiv:2410.22397), specialized to the
LS10-BGS samples (S1: ``M* > 10^10 M_sun``, ``z_mean = 0.135``).  It is a fourth
AGN model alongside :class:`~hod_mod.agn.ham.HamAGNModel`,
:class:`~hod_mod.agn.hod.HODAgnModel`, and
:class:`~hod_mod.agn.xray.XrayAGNModel`.

Two pieces
----------
**Part 1 — the W_AGN(z) kernel (Eq. A9).**  :func:`compute_w_agn_kernel` builds
the X-ray-flux-weighted redshift kernel

.. math::
    W_\\mathrm{AGN}(z) = \\int \\mathrm{d}\\ln L_X\\,
        \\Phi_\\mathrm{AGN}(L_X, z)\\, S_X(L_X, z)\\, f(S_X)

using the Aird+2015 LADE **hard** XLF (:func:`agn_ham._aird15_lade_np`) and the
Comparat+2019 obscuration-weighted K-correction (:func:`agn_ham.mean_k_eff`) to
turn hard-band rest-frame luminosity into observed soft (0.5--2 keV) flux.  The
cross-correlation is between the galaxies and **all** X-ray events, so **no
optical (r-band) selection is applied**: the integral runs over every AGN in the
k-corrected flux range (``[1e-20, 1e-10] erg/s/cm^2``, Lau+2025).  The
completeness ``f(S_X)`` is set to 1 (unlike Lau et al. 2025, who use a logistic
flux-limit curve A11).  All integrand components are stored to an h5 file (one
per sample) for verification; the file is skipped if it already exists.

**Part 2 — :class:`DutyCycleAGNModel`.**  The AGN occupation is the Zu & Mandelbaum
2015 occupation (taken from the joint ``wp + n_gal`` MAP fit
``results/bgs_zm15_joint_wp_ngal/map_result.json``, at the sample's stellar-mass
threshold) scaled by a free duty cycle ``DC = 10**log10DC`` (``log10DC in
[-4, 0]``).  The per-AGN soft luminosity is mass-independent (flat in halo mass,
consistent with the kernel-outside-d M form of Eqs. A7/A8) and derived from the
kernel: ``<L_X>(z) = mean_SX(z) * 4 pi d_L(z)^2`` with
``mean_SX(z) = W_AGN(z) / n_AGN(z)``.  The class exposes the standard
``nc_ns_agn`` / ``agn_emissivity_uk`` interface so it plugs straight into
:class:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra`.

The duty cycle is applied **once**, as a multiplicative factor on the per-AGN
emissivity amplitude.  Because the galaxy×AGN cross-power is linear in the AGN
occupation, this is mathematically identical to scaling the AGN occupation
(``N^A = DC * N^g``, i.e. the duty cycle in front of the occupation) but
lets ``log10DC`` be threaded as a cheap ``agn_kwargs`` parameter by the fit
without rebuilding the model.  :attr:`DutyCycleAGNModel._pair_with_galaxy_occupation`
is set so the cross-spectra code uses the proper galaxy×AGN occupation product
in the 1-halo term (Eqs. A7/A8 modified for the cross).

References
----------
Lau, Bogdán, Chadayammuri et al. 2025, ApJ 983, 8 (arXiv:2410.22397) — formalism
Aird et al. 2015, MNRAS 451, 1892 — LADE XLF
Comparat et al. 2019, MNRAS 487, 2005 (arXiv:1901.10866) — obscuration / K-corr
Zu & Mandelbaum 2015, MNRAS 454, 1161 — iHOD occupation
Comparat et al. 2025, A&A 697, A173 — LS10-BGS samples S1…S7
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import jax.numpy as jnp

from hod_mod.agn.ham import (
    _XLF_FUNCS,
    _H_XLF,
    setup_kcorrection,
    mean_k_eff,
)
from hod_mod.agn.hod import _MPC_CM, BGS_SAMPLES
from hod_mod.connection.hod import n_cen_thresh_zu15, n_sat_thresh_zu15
from hod_mod.core.distances import luminosity_distance
from hod_mod.paths import results_root

_LOG = logging.getLogger(__name__)

# Default location of the joint wp+ngal ZM15 MAP fit used for the occupation.
_ZM15_MAP_JSON = str(results_root() / "bgs_zm15_joint_wp_ngal" / "map_result.json")

# Default output directory for the per-sample W_AGN(z) kernels.
_W_AGN_DIR = str(results_root() / "agn_duty_cycle")

# The 13 ZuMandelbaum+2015 occupation parameters expected in map_result.json.
_ZM15_PARAM_KEYS = (
    "lg_m1h", "lg_m0star", "beta", "delta", "gamma",
    "sigma_lnmstar", "eta", "fc",
    "bsat", "beta_sat", "bcut", "beta_cut", "alpha_sat",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_zm15_map_params(path: str | None = None) -> dict:
    """Load the 13 ZuMandelbaum+2015 occupation parameters from a MAP json.

    Parameters
    ----------
    path : str, optional
        Path to a ``map_result.json`` with a ``params`` dict.  Defaults to
        ``results/bgs_zm15_joint_wp_ngal/map_result.json``.

    Returns
    -------
    dict
        ``{key: value}`` for the 13 keys in :data:`_ZM15_PARAM_KEYS`.
    """
    p = path if path is not None else _ZM15_MAP_JSON
    with open(p) as fh:
        data = json.load(fh)
    params = data["params"]
    missing = [k for k in _ZM15_PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"map_result.json {p} missing ZM15 params: {missing}")
    return {k: float(params[k]) for k in _ZM15_PARAM_KEYS}


def _default_cosmo(theta_cosmo: dict | None) -> dict:
    if theta_cosmo is not None:
        return theta_cosmo
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    return LinearPowerSpectrum.default_cosmology()


def _luminosity_distance_cm(z, theta_cosmo: dict) -> np.ndarray:
    """Luminosity distance(s) [cm] for scalar or array z."""
    th = theta_cosmo
    z_arr = jnp.atleast_1d(jnp.asarray(np.asarray(z, dtype=float)))
    dl_mpc = np.asarray(luminosity_distance(
        z_arr, th["h"], th["Omega_m"], th.get("w0", -1.0), th.get("wa", 0.0),
    ), dtype=np.float64)
    return dl_mpc * _MPC_CM


# ---------------------------------------------------------------------------
# Part 1 — W_AGN(z) kernel (Eq. A9)
# ---------------------------------------------------------------------------

def w_agn_path_for(sample: str, out_dir: str | None = None) -> str:
    """Default h5 path for a sample's W_AGN(z) kernel."""
    d = out_dir if out_dir is not None else _W_AGN_DIR
    return os.path.join(d, f"W_AGN_{sample}.h5")


def compute_w_agn_kernel(
    sample: str = "S1",
    theta_cosmo: dict | None = None,
    z_grid: np.ndarray | None = None,
    n_z: int = 40,
    log10lx_hard: np.ndarray | None = None,
    n_lx: int = 500,
    xlf: str = "aird15",
    flux_range: tuple[float, float] = (1e-20, 1e-10),
    kcorr_path: str | None = None,
    out_path: str | None = None,
    out_dir: str | None = None,
    overwrite: bool = False,
) -> str:
    """Compute the Lau+2025 W_AGN(z) kernel (Eq. A9) and store it to h5.

    The kernel weights the AGN X-ray flux over the luminosity function at each
    redshift, with completeness ``f(S_X) = 1``.  The cross-correlation is between
    the galaxies and **all** X-ray events, so **no optical (r-band) selection is
    applied**: the integral runs over every AGN whose k-corrected soft flux falls
    in ``flux_range`` (default ``[1e-20, 1e-10] erg/s/cm^2``, the Lau+2025 range).
    All integrand components are stored for verification.  If the output file
    already exists and ``overwrite`` is False, the function returns its path
    without recomputing.

    Returns
    -------
    str
        Path to the h5 file written (or found).
    """
    import h5py

    out_path = out_path if out_path is not None else w_agn_path_for(sample, out_dir)
    if os.path.exists(out_path) and not overwrite:
        _LOG.info("W_AGN kernel already exists (%s) — skipping.", out_path)
        return out_path

    if xlf not in _XLF_FUNCS:
        raise ValueError(f"xlf must be one of {list(_XLF_FUNCS)}, got '{xlf}'")
    xlf_func = _XLF_FUNCS[xlf]

    theta_cosmo = _default_cosmo(theta_cosmo)
    h = float(theta_cosmo["h"])

    if sample not in BGS_SAMPLES:
        raise ValueError(f"Unknown sample '{sample}'. Known: {list(BGS_SAMPLES)}")
    s = BGS_SAMPLES[sample]
    z_mean = float(s["z_mean"])

    # Redshift grid covering the galaxy n(z) (LS10-BGS reach z<~0.35).
    if z_grid is None:
        z_grid = np.linspace(0.01, 0.35, n_z)
    z_grid = np.asarray(z_grid, dtype=np.float64)

    # Hard-band (2-10 keV rest-frame) luminosity grid [log10 erg/s].
    if log10lx_hard is None:
        log10lx_hard = np.linspace(40.0, 47.0, n_lx)
    log10lx_hard = np.asarray(log10lx_hard, dtype=np.float64)
    n_z, n_lx = z_grid.size, log10lx_hard.size

    f_lo, f_hi = float(flux_range[0]), float(flux_range[1])

    kcorr_interp, kcorr_mode = setup_kcorrection(kcorr_path)
    h_factor = (_H_XLF / h) ** 3   # XLF Mpc^-3 (h=0.70) -> (Mpc/h)^-3

    # Pre-allocate 2-D (z, L) component arrays + 1-D (z,) results.
    phi_dex      = np.zeros((n_z, n_lx))
    k_eff_arr    = np.zeros((n_z, n_lx))
    log10lx_soft = np.zeros((n_z, n_lx))
    s_x          = np.zeros((n_z, n_lx))
    sel_mask     = np.zeros((n_z, n_lx))
    integrand_W  = np.zeros((n_z, n_lx))
    f_sx         = np.ones((n_z, n_lx))   # completeness = 1 (user choice)

    W_AGN   = np.zeros(n_z)
    n_AGN   = np.zeros(n_z)
    mean_SX = np.zeros(n_z)

    dl_cm = _luminosity_distance_cm(z_grid, theta_cosmo)   # (n_z,)

    for i, z in enumerate(z_grid):
        # 1. XLF density (per dex), converted to (Mpc/h)^-3 dex^-1.
        phi = np.asarray(xlf_func(log10lx_hard, float(z))) * h_factor
        # 2. obscuration-weighted hard -> observed soft luminosity.
        keff = np.asarray(mean_k_eff(kcorr_interp, kcorr_mode, log10lx_hard, float(z)))
        keff = np.clip(keff, 1e-30, 1.0)
        lxs = log10lx_hard + np.log10(keff)
        # 3. observed soft flux [erg/s/cm^2].
        sx = 10.0 ** lxs / (4.0 * np.pi * dl_cm[i] ** 2)
        # 4. NO optical selection: the cross-correlation is galaxies x all X-ray
        #    events.  Integrate over every AGN in the k-corrected flux range.
        sel = (sx >= f_lo) & (sx <= f_hi)
        # 5. completeness f(S_X) = 1 (already in f_sx).
        # 6. integrate over dlog10 L (= dlnL with phi the per-dex XLF).
        integ = phi * sx * sel.astype(float) * f_sx[i]
        W_AGN[i] = np.trapezoid(integ, log10lx_hard)
        n_AGN[i] = np.trapezoid(phi * sel.astype(float) * f_sx[i], log10lx_hard)
        mean_SX[i] = W_AGN[i] / n_AGN[i] if n_AGN[i] > 0 else 0.0

        phi_dex[i]      = phi
        k_eff_arr[i]    = keff
        log10lx_soft[i] = lxs
        s_x[i]          = sx
        sel_mask[i]     = sel.astype(float)
        integrand_W[i]  = integ

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("z_grid", data=z_grid)
        f.create_dataset("log10LX_hard", data=log10lx_hard)
        # 2-D integrand components (n_z, n_lx)
        f.create_dataset("phi_dex", data=phi_dex)
        f.create_dataset("k_eff", data=k_eff_arr)
        f.create_dataset("log10LX_soft", data=log10lx_soft)
        f.create_dataset("S_X", data=s_x)
        f.create_dataset("selection_mask", data=sel_mask)
        f.create_dataset("f_SX", data=f_sx)
        f.create_dataset("integrand_W", data=integrand_W)
        # 1-D results (n_z,)
        f.create_dataset("W_AGN", data=W_AGN)
        f.create_dataset("n_AGN", data=n_AGN)
        f.create_dataset("mean_SX", data=mean_SX)
        # metadata
        f.attrs["sample"] = sample
        f.attrs["z_mean"] = z_mean
        f.attrs["z_min"] = float(z_grid.min())
        f.attrs["z_max"] = float(z_grid.max())
        f.attrs["flux_lo"] = f_lo
        f.attrs["flux_hi"] = f_hi
        f.attrs["xlf"] = xlf
        f.attrs["h"] = h
        f.attrs["Omega_m"] = float(theta_cosmo["Omega_m"])
        f.attrs["selection"] = "none (galaxies x all X-ray events); flux range only"
        f.attrs["completeness"] = "f(S_X) = 1 (constant; not Lau+2025 logistic A11)"
        f.attrs["W_AGN_units"] = "(Mpc/h)^-3 * erg/s/cm^2"
        f.attrs["n_AGN_units"] = "(Mpc/h)^-3"
        f.attrs["mean_SX_units"] = "erg/s/cm^2"
        f.attrs["band_xlf"] = "Aird+2015 LADE hard 2-10 keV rest-frame"
        f.attrs["band_obs"] = "soft 0.5-2 keV observed (via K-correction)"

    _LOG.info("Wrote W_AGN kernel for %s to %s "
              "(W_AGN(z_mean=%.3f)~%.3e, mean_SX~%.3e erg/s/cm^2).",
              sample, out_path, z_mean,
              float(np.interp(z_mean, z_grid, W_AGN)),
              float(np.interp(z_mean, z_grid, mean_SX)))
    return out_path


# ---------------------------------------------------------------------------
# High-mass cutoff for the AGN occupation
# ---------------------------------------------------------------------------

def _high_mass_cutoff(log10m, lo: float, hi: float):
    """Smooth high-mass taper applied to the AGN occupation.

    Returns 1 for ``log10m <= lo``, 0 for ``log10m >= hi``, and a cosine
    smoothstep ``0.5*(1 + cos(pi*x))`` (``x = (log10m-lo)/(hi-lo)``) in between
    — exactly 1 at ``lo`` and 0 at ``hi``, monotonically decreasing and
    C1-smooth.  Physically: X-ray AGN are not hosted by the most massive
    (cluster-scale) halos, so the occupation declines from ``10**lo`` to zero
    at ``10**hi``.
    """
    x = jnp.clip((jnp.asarray(log10m) - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * (1.0 + jnp.cos(jnp.pi * x))


# ---------------------------------------------------------------------------
# Part 2 — DutyCycleAGNModel
# ---------------------------------------------------------------------------

class DutyCycleAGNModel:
    """Lau et al. 2025 AGN model: ZM15 occupation × duty cycle + W_AGN kernel.

    Parameters
    ----------
    sample : str
        LS10-BGS sample label (default ``'S1'``); sets ``z_mean``,
        ``log10m_star_thresh`` (= ``log10ms_min``), and the W_AGN h5 file.
    theta_cosmo : dict, optional
        Cosmology dict.  Default: Planck 2018.
    hmf : optional
        Kept for interface parity with the other AGN models (unused here; the
        cross-power uses the HMF/bias from the host ``FullHaloModelPrediction``).
    zm15_params : dict, optional
        The 13 ZuMandelbaum+2015 occupation parameters.  Default: loaded from
        ``zm15_map_json``.
    zm15_map_json : str, optional
        Path to the ``map_result.json`` to read the occupation from.  Default:
        ``results/bgs_zm15_joint_wp_ngal/map_result.json``.
    log10m_star_thresh : float, optional
        Stellar-mass threshold [log10 M_sun/h] for the occupation.  Default:
        the sample's ``log10ms_min`` (S1 → 10.0).
    log10DC : float
        log10 of the AGN duty cycle ``DC = 10**log10DC`` (range [-4, 0]).
    w_agn_path : str, optional
        Path to the W_AGN h5 file.  Default: ``results/agn_duty_cycle/W_AGN_<sample>.h5``;
        computed on the fly (Part 1) if missing.
    recompute_kernel : bool
        Force recomputation of the W_AGN kernel even if the file exists.
    """

    def __init__(
        self,
        sample: str = "S1",
        theta_cosmo: dict | None = None,
        hmf=None,
        zm15_params: dict | None = None,
        zm15_map_json: str | None = None,
        log10m_star_thresh: float | None = None,
        log10DC: float = -2.0,
        w_agn_path: str | None = None,
        xlf: str = "aird15",
        flux_range: tuple[float, float] = (1e-20, 1e-10),
        kcorr_path: str | None = None,
        recompute_kernel: bool = False,
        apply_high_mass_cutoff: bool = True,
        log10m_cut_lo: float = 14.0,
        log10m_cut_hi: float = float(np.log10(2e14)),
    ):
        if sample not in BGS_SAMPLES:
            raise ValueError(f"Unknown sample '{sample}'. Known: {list(BGS_SAMPLES)}")
        self._sample = sample
        self._theta_cosmo = _default_cosmo(theta_cosmo)
        self._hmf = hmf

        s = BGS_SAMPLES[sample]
        self._z_mean = float(s["z_mean"])
        self._log10m_star_thresh = (
            float(s["log10ms_min"]) if log10m_star_thresh is None
            else float(log10m_star_thresh)
        )

        # -- ZM15 occupation parameters
        if zm15_params is not None:
            self._zm15 = {k: float(zm15_params[k]) for k in _ZM15_PARAM_KEYS}
        else:
            self._zm15 = load_zm15_map_params(zm15_map_json)

        # Full galaxy HOD param dict (occupation + threshold) for the galaxy leg.
        self.zm15_hod_params = dict(self._zm15)
        self.zm15_hod_params["log10m_star_thresh"] = self._log10m_star_thresh

        self.log10DC = float(log10DC)

        # Tell the cross-spectra code to pair the AGN emission with the GALAXY
        # occupation (proper g×AGN 1-halo cross product, Eqs. A7/A8 modified).
        self._pair_with_galaxy_occupation = True
        # Interface back-compat attribute (unused on the HOD/cross path).
        self._f_sat_agn = 0.10

        # High-mass cutoff: no X-ray AGN in the most massive (cluster) halos.
        self._apply_high_mass_cutoff = bool(apply_high_mass_cutoff)
        self._log10m_cut_lo = float(log10m_cut_lo)
        self._log10m_cut_hi = float(log10m_cut_hi)

        # -- W_AGN(z) kernel (Part 1): load or compute.
        self._w_agn_path = (
            w_agn_path if w_agn_path is not None else w_agn_path_for(sample)
        )
        if recompute_kernel or not os.path.exists(self._w_agn_path):
            compute_w_agn_kernel(
                sample=sample, theta_cosmo=self._theta_cosmo,
                xlf=xlf, flux_range=flux_range, kcorr_path=kcorr_path,
                out_path=self._w_agn_path, overwrite=recompute_kernel,
            )
        self._load_kernel()

    # ------------------------------------------------------------------
    def _load_kernel(self) -> None:
        import h5py
        with h5py.File(self._w_agn_path, "r") as f:
            self._k_z = np.asarray(f["z_grid"])
            self._k_W_AGN = np.asarray(f["W_AGN"])
            self._k_n_AGN = np.asarray(f["n_AGN"])
            self._k_mean_SX = np.asarray(f["mean_SX"])

    def mean_sx_at(self, z) -> np.ndarray:
        """Number-weighted mean observed soft flux <S_X>(z) [erg/s/cm^2]."""
        return np.interp(np.asarray(z, dtype=float), self._k_z, self._k_mean_SX)

    def w_agn_at(self, z) -> np.ndarray:
        """The W_AGN(z) kernel [ (Mpc/h)^-3 erg/s/cm^2 ] at redshift(s) z."""
        return np.interp(np.asarray(z, dtype=float), self._k_z, self._k_W_AGN)

    # ------------------------------------------------------------------
    # AGN occupation (pure ZM15; the duty cycle is applied in the emissivity).
    # ------------------------------------------------------------------
    def nc_ns_agn(self, log10m_arr, hod_params: dict | None = None):
        """Return (N_cen, N_sat) for the AGN occupation = ZM15 × high-mass cutoff.

        The occupation is the bare ZM15 occupation (DC=1; the duty cycle is
        applied as a single multiplicative amplitude inside
        :meth:`agn_emissivity_uk`, equivalent by linearity of the cross-power)
        multiplied by a smooth **high-mass cutoff** (:func:`_high_mass_cutoff`)
        so that no X-ray AGN are hosted by the most massive (cluster-scale)
        halos: both centrals and satellites decline from full at
        ``10**log10m_cut_lo`` (default 1e14) to zero at ``10**log10m_cut_hi``
        (default 2e14).  Returning the (cut) ZM15 occupation here keeps the
        cross-spectra 1-halo product ``N^g · N^A`` well-defined.
        """
        p = hod_params if hod_params is not None else self._zm15
        thr = self._log10m_star_thresh
        log10m = jnp.asarray(log10m_arr)
        nc = n_cen_thresh_zu15(
            log10m, thr, p["lg_m1h"], p["lg_m0star"], p["beta"], p["delta"],
            p["gamma"], p["sigma_lnmstar"], p["eta"], p["fc"],
        )
        ns = n_sat_thresh_zu15(
            log10m, thr, p["lg_m1h"], p["lg_m0star"], p["beta"], p["delta"],
            p["gamma"], p["sigma_lnmstar"], p["eta"], p["fc"],
            p["bsat"], p["beta_sat"], p["bcut"], p["beta_cut"], p["alpha_sat"],
        )
        if self._apply_high_mass_cutoff:
            cut = _high_mass_cutoff(log10m, self._log10m_cut_lo, self._log10m_cut_hi)
            nc = nc * cut
            ns = ns * cut
        return np.asarray(nc, dtype=np.float64), np.asarray(ns, dtype=np.float64)

    # ------------------------------------------------------------------
    # Per-AGN soft luminosity (mass-independent), carrying the duty cycle.
    # ------------------------------------------------------------------
    def mean_agn_lx(self, z: float, log10DC: float | None = None) -> float:
        """DC-weighted mean observed soft luminosity per AGN [erg/s] at z.

        ``<L_X>(z) = DC * mean_SX(z) * 4 pi d_L(z)^2``.
        """
        dc = 10.0 ** (self.log10DC if log10DC is None else float(log10DC))
        dl_cm = float(_luminosity_distance_cm(z, self._theta_cosmo)[0])
        return dc * float(self.mean_sx_at(z)) * 4.0 * np.pi * dl_cm ** 2

    def mean_agn_log10lx(self, m_halo_arr, z: float = None, shmr_params=None, **kw):
        """log10 of the (mass-independent) DC-weighted mean soft L_X [erg/s].

        Returned per element of ``m_halo_arr`` (flat in halo mass) for interface
        parity with the other AGN models.
        """
        m = np.asarray(m_halo_arr, dtype=np.float64)
        val = np.log10(self.mean_agn_lx(z, log10DC=kw.get("log10DC")))
        return np.full(m.shape, val, dtype=np.float64)

    def agn_emissivity_uk(self, k_arr, m_halo_arr, z: float, theta_cosmo: dict,
                          shmr_params=None, log10DC: float | None = None, **kw):
        """FT of the AGN X-ray emissivity (point source, flat in k and in M).

        Returns the DC-weighted mean soft luminosity per AGN, normalized
        ``L_X / 1e43``, in the same convention as
        :meth:`HODAgnModel.agn_emissivity_uk`.  The cross-spectra code applies
        the AGN occupation weighting and the unit conversion to the gas
        emissivity scale.

        Returns
        -------
        (Nk, NM) float64 ndarray
        """
        k = np.asarray(k_arr, dtype=np.float64)
        m = np.asarray(m_halo_arr, dtype=np.float64)
        lx = self.mean_agn_lx(z, log10DC=log10DC)   # erg/s, includes DC
        lx_norm = lx / 1e43
        return np.full((k.shape[0], m.shape[0]), lx_norm, dtype=np.float64)
