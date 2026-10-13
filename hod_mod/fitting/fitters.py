"""MAP + MCMC fitters (wp, joint wp+ESD, ESD-only) on the halo model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from hod_mod.core.power_spectrum import (
    LinearPowerSpectrum, default_pk_linear, mnu_sum, omega_nu)
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.core.beyond_linear_bias import BeyondLinearBiasMead21
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.connection.hod import (
    HODModel,
    Kravtsov04HODModel,
    MoreHODModel,
    Guo18ICSMFModel,
    Guo19ICSMFModel,
    Zacharegkas25HODModel,
    VanUitert16CSMFModel,
    ZuMandelbaum15HODModel,
    Leauthaud12HODModel,
    Lange25HODModel,
)
from hod_mod.connection.clf import CLFModel

from .config import FitConfig
from .models import HOD_MODELS
from hod_mod.fitting.mcmc_resume import revive_ensemble




# ---------------------------------------------------------------------------
# Log-probability helpers (standalone, callable by emcee and scipy)
# ---------------------------------------------------------------------------

def _assemble_hod_params(theta_vec, free_params, fixed_params):
    """Combine a free-parameter vector with fixed-parameter dict."""
    p = dict(fixed_params)
    for name, val in zip(free_params, theta_vec):
        p[name] = float(val)
    return p


def log_prob_wp(
    theta_vec,
    free_params: list,
    fixed_params: dict,
    param_bounds: dict,
    predictor: FullHaloModelPrediction,
    rp_arr: np.ndarray,
    wp_obs: np.ndarray,
    icov_wp: np.ndarray,
    z: float,
    theta_cosmo: dict,
    pi_max: float,
) -> float:
    """Log-posterior for wp(rp) fitting.

    log P(theta | data) = -0.5 chi^2  (uniform prior, 0 inside bounds, -inf outside)

    Parameters
    ----------
    theta_vec : array_like, shape (n_free,)
    free_params : list of str
    fixed_params : dict
    param_bounds : dict — {name: (lo, hi)}
    predictor : FullHaloModelPrediction
    rp_arr, wp_obs, icov_wp : projected data arrays and inverse covariance
    z : float — effective redshift
    theta_cosmo : dict — cosmological parameters
    pi_max : float — l.o.s. integration limit [Mpc/h]
    """
    for name, val in zip(free_params, theta_vec):
        lo, hi = param_bounds[name]
        if not (lo <= val <= hi):
            return -np.inf
    hod_params = _assemble_hod_params(theta_vec, free_params, fixed_params)
    try:
        wp_pred = np.asarray(
            predictor.wp(jnp.array(rp_arr), pi_max, z, theta_cosmo, hod_params)
        )
    except Exception:
        return -np.inf
    residual = wp_pred - wp_obs
    return -0.5 * float(residual @ icov_wp @ residual)


def log_prob_joint(
    theta_vec,
    free_params: list,
    fixed_params: dict,
    param_bounds: dict,
    predictor: FullHaloModelPrediction,
    rp_arr: np.ndarray,
    wp_obs: np.ndarray,
    icov_wp: np.ndarray,
    R_arr: np.ndarray,
    ds_obs: np.ndarray,
    icov_ds: np.ndarray,
    ng_obs: float,
    ng_frac_err: float,
    z: float,
    theta_cosmo: dict,
    pi_max: float,
) -> float:
    r"""Log-posterior for joint wp + ΔΣ + n_g fitting.

    .. math::

        \log P(\theta|d) = -\frac{1}{2}\bigl[
          \chi^2_{w_p} + \chi^2_{\Delta\Sigma} + \chi^2_{n_g}
        \bigr]
    """
    for name, val in zip(free_params, theta_vec):
        lo, hi = param_bounds[name]
        if not (lo <= val <= hi):
            return -np.inf
    hod_params = _assemble_hod_params(theta_vec, free_params, fixed_params)
    try:
        wp_pred = np.asarray(
            predictor.wp(jnp.array(rp_arr), pi_max, z, theta_cosmo, hod_params)
        )
        ds_pred = np.asarray(
            predictor.delta_sigma(jnp.array(R_arr), z, theta_cosmo, hod_params)
        )
        ng_pred = predictor.n_gal(z, theta_cosmo, hod_params)
    except Exception:
        return -np.inf
    chi2_wp = float((wp_pred - wp_obs) @ icov_wp @ (wp_pred - wp_obs))
    chi2_ds = float((ds_pred - ds_obs) @ icov_ds @ (ds_pred - ds_obs))
    chi2_ng = float(((ng_pred - ng_obs) / (ng_frac_err * ng_obs)) ** 2)
    return -0.5 * (chi2_wp + chi2_ds + chi2_ng)


# ---------------------------------------------------------------------------
# Free-cosmology helpers
# ---------------------------------------------------------------------------


class _CachedPkLinear:
    """Interpolation cache around LinearPowerSpectrum for free-cosmo MCMC.

    On the first call for a given cosmology the reference P(k) is computed on a
    fixed k grid by the wrapped backend and log-log interpolated on every
    subsequent call.  With a Boltzmann backend this takes the per-sample cost
    from seconds to <1 ms once the cache warms up.

    The key must name every parameter the spectrum depends on.  It previously
    covered only (z, Ω_m, ln10As, h), so a chain that varied Ω_b, n_s, w0/wa or
    Σm_ν would silently be served a spectrum computed at a different cosmology.
    """

    @property
    def has_native_z(self):
        """Delegate the redshift-capability declaration to the wrapped backend.

        Without this the growth dispatcher sees an unrecognised class, refuses to
        assume it is z-aware and falls back to the ODE -- precisely in the
        free-cosmology runs that paid for a Boltzmann solver.  A wrapper must
        answer for what it wraps.
        """
        from hod_mod.core.growth import backend_has_native_z
        return backend_has_native_z(self._base)

    def pk_linear_cb(self, k, z: float, theta: dict):
        """Forward the cold spectrum when the wrapped backend has one.

        Not cached: the cold path is used only by the growth ratio, which is
        called once per (cosmology, redshift), not per likelihood evaluation.
        """
        if not hasattr(self._base, "pk_linear_cb"):
            raise AttributeError(
                f"{type(self._base).__name__} has no cold spectrum to forward")
        return self._base.pk_linear_cb(k, z, theta)

    def __init__(self, pk_lin_obj, n_k: int = 512):
        self._base      = pk_lin_obj
        self._k_ref     = np.logspace(-4, 1.5, n_k)
        self._log_k_ref = np.log(self._k_ref)
        self._cache: dict = {}

    def _key(self, z: float, theta: dict) -> tuple:
        return (
            round(float(z), 4),
            round(float(theta["Omega_m"]), 5),
            round(float(theta["Omega_b"]), 6),
            round(float(theta["ln10^{10}A_s"]), 4),
            round(float(theta["n_s"]), 5),
            round(float(theta.get("h", 0.6736)), 4),
            round(float(mnu_sum(theta)), 5),
            round(float(theta.get("w0", -1.0)), 5),
            round(float(theta.get("wa", 0.0)), 5),
        )

    def pk_linear(self, k, z: float, theta: dict):
        key = self._key(z, theta)
        if key not in self._cache:
            pk_ref = np.asarray(self._base.pk_linear(self._k_ref, float(z), theta))
            self._cache[key] = np.log(np.maximum(pk_ref, 1e-50))
        log_k = np.log(np.asarray(k, dtype=float))
        return jnp.asarray(np.exp(np.interp(log_k, self._log_k_ref, self._cache[key])))


# ---------------------------------------------------------------------------
# WpFitter
# ---------------------------------------------------------------------------

class WpFitter:
    """Fit an HOD model to wp(rp) data.

    Supports ``data_format = "csv"``, ``"hdf5"``, or ``"fits"`` (FITS JK).
    For a joint wp+ΔΣ+n_g fit, use :class:`JointFitter` with a config that
    has ``ds_file`` set.  For a ΔΣ-only fit, use :class:`DeltaSigmaFitter`.

    Parameters
    ----------
    config : FitConfig
        Parsed fitting configuration (from :func:`load_config`).
    """

    def __init__(self, config: FitConfig):
        self._setup_common(config)
        self._load_data()
        self._build_predictor()
        self._build_icov()

    # ------------------------------------------------------------------
    # Common setup (reused by DeltaSigmaFitter)

    def _setup_common(self, config: FitConfig):
        self.config      = config
        self._pk_lin     = default_pk_linear()
        self.theta_cosmo = (
            config.cosmology if config.cosmology is not None
            else self._pk_lin.default_cosmology()
        )
        self._cov_wp     = None
        if config.use_free_cosmo:
            self._pk_lin = _CachedPkLinear(self._pk_lin)

    #: Cosmological parameters a ``use_free_cosmo`` fit may sample.  A whitelist
    #: rather than a pattern match, because the pattern match is what hid the
    #: bug below: the previous implementation acted on exactly ``Omega_m`` and
    #: ``S8``, so a YAML that freed ``h`` -- as
    #: ``benchmark_more2015_logM11_12_freecosmo.yml`` does, its own header
    #: claiming "5 (HOD) + 2 (Omega_m, S8) + 1 (h) = 8" -- sampled an ``h`` that
    #: never reached the cosmology dict at all.  That fit ran 7 effective
    #: parameters while reporting 8.
    _COSMO_FREE = ("Omega_m", "Omega_b", "h", "n_s", "ln10^{10}A_s",
                   "w0", "wa", "sum_mnu")

    def _theta_cosmo_call(self, all_params: dict) -> dict:
        """Build a per-call theta_cosmo when use_free_cosmo is enabled.

        Copies every free cosmological parameter straight through and re-derives
        ``Omega_cdm`` last, so that a free ``h`` or ``sum_mnu`` -- both of which
        move ``Omega_nu`` -- are accounted for.

        The amplitude is ``ln10^{10}A_s``, sampled directly.  It used to be
        ``S8``, converted here via ``sigma8 = S8 sqrt(0.3/Omega_m)`` and then a
        linearised ``ln10As += 2 ln(sigma8/sigma8_fid)``.  That conversion wrote
        only ``ln10^{10}A_s`` and left the fiducial ``sigma8`` sitting in ``tc``,
        where both the P(k) backend and the mass function preferred it -- so the
        sampled ``S8`` moved nothing.  Sampling the amplitude itself removes the
        conversion, the stale key and the precedence question together.

        When ``use_free_cosmo`` is False, returns ``self.theta_cosmo`` unchanged.
        """
        if not self.config.use_free_cosmo:
            return self.theta_cosmo
        unknown = [k for k in ("sigma8", "S8") if k in all_params]
        if unknown:
            raise ValueError(
                f"{type(self).__name__}: {unknown} is no longer a sampleable "
                "cosmological parameter -- free 'ln10^{10}A_s' instead and "
                "derive sigma8/S8 from the chain a posteriori with "
                "hod_mod.core.amplitude.s8_from_theta.")
        tc = dict(self.theta_cosmo)
        for name in self._COSMO_FREE:
            if name in all_params:
                tc[name] = float(all_params[name])
        # Last, and only now: omega_nu depends on h and sum_mnu, both of which
        # may just have moved.
        if "Omega_m" in tc:
            tc["Omega_cdm"] = (float(tc["Omega_m"]) - float(tc["Omega_b"])
                               - float(omega_nu(tc)))
        return tc

    # ------------------------------------------------------------------
    # Setup helpers

    def _load_data(self):
        fmt = self.config.data_format
        if fmt == "hdf5":
            self._load_data_hdf5()
        elif fmt == "fits":
            self._load_data_fits()
        elif fmt == "bwpd":
            self._load_data_bwpd()
        else:
            self._load_data_csv()

    def _load_data_csv(self):
        import pandas as pd
        data = pd.read_csv(self.config.data_file, comment="#")
        mask = (data["rp_hMpc"] >= self.config.rp_min) & (data["rp_hMpc"] <= self.config.rp_max)
        self.rp_arr = data["rp_hMpc"][mask].to_numpy()
        self.wp_obs = data["wp_hMpc"][mask].to_numpy()
        self.wp_err = data["wp_err_hMpc"][mask].to_numpy()
        self._cov_wp = None

    def _load_data_hdf5(self):
        from hod_mod.data_io.sum_stat_reader import SumStatReader
        reader = SumStatReader.from_hdf5(self.config.data_file)
        d    = reader.wp()
        rp   = d["rp"]
        wp   = d["wp"]
        cov  = d["cov"]
        mask = (rp >= self.config.rp_min) & (rp <= self.config.rp_max)
        self.rp_arr  = rp[mask]
        self.wp_obs  = wp[mask]
        self.wp_err  = np.sqrt(np.diag(cov)[mask])
        self._cov_wp = cov[np.ix_(mask, mask)]

    def _load_data_fits(self):
        from hod_mod.data_io.wprp_fits import load_jk_wp_auto
        cfg = self.config
        rp_h, wp_h, cov_h = load_jk_wp_auto(
            directory=cfg.jk_dir,
            pattern=cfg.jk_pattern,
            h=cfg.h_hubble,
            rp_min=cfg.rp_min,
            rp_max=cfg.rp_max,
        )
        self.rp_arr  = rp_h
        self.wp_obs  = wp_h
        self.wp_err  = np.sqrt(np.diag(cov_h))
        self._cov_wp = cov_h

    @staticmethod
    def _read_bwpd_file(path: str) -> np.ndarray:
        """Parse a manually-digitized BWPD file with mixed comma/space delimiters.

        Splits each non-comment line on any run of commas and/or whitespace,
        making the reader robust to inconsistent spacing around commas. Returns
        a (N, 3) float array with the three data columns.
        """
        import re
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = [x for x in re.split(r"[,\s]+", line) if x]
                try:
                    rows.append([float(x) for x in fields[:3]])
                except ValueError:
                    continue  # skip header rows that contain non-numeric text
        return np.array(rows)

    def _load_data_bwpd(self):
        """Load wp from a manually-digitized BWPD file.

        Expected columns (comma/whitespace-delimited, spaces allowed):
            rp_hMpc   rpwp_h2Mpc2   rpwp_upper_h2Mpc2

        - ``rp_hMpc``           : r_p in h^-1 Mpc
        - ``rpwp_h2Mpc2``       : r_p × w_p(r_p) in (h^-1 Mpc)^2
        - ``rpwp_upper_h2Mpc2`` : upper bound of r_p × w_p; uncertainty = upper − value

        Converts to code units::

            wp [h^-1 Mpc] = rpwp / rp
            wp_err        = (rpwp_upper − rpwp) / rp
        """
        d          = self._read_bwpd_file(self.config.data_file)
        rp         = d[:, 0]
        rpwp       = d[:, 1]
        rpwp_upper = d[:, 2]
        mask = (rp >= self.config.rp_min) & (rp <= self.config.rp_max)
        self.rp_arr  = rp[mask]
        self.wp_obs  = rpwp[mask] / rp[mask]
        self.wp_err  = (rpwp_upper[mask] - rpwp[mask]) / rp[mask]
        self._cov_wp = None

    def _load_ds_data_bwpd(self):
        """Load ΔΣ from a manually-digitized BWPD file.

        Expected columns (comma/whitespace-delimited, spaces allowed):
            rp_hMpc   rpDS_1e6Msunpc   rpDS_upper_1e6Msunpc

        - ``rp_hMpc``              : R in h^-1 Mpc
        - ``rpDS_1e6Msunpc``       : R × ΔΣ in 10^6 M_sun pc^-1
        - ``rpDS_upper_1e6Msunpc`` : upper bound; uncertainty = upper − value

        Unit conversion to code units (M_sun h pc^-2):
            Since 1 Mpc = 10^6 pc the h factors cancel:
            ΔΣ [M_sun h/pc^2] = (R×ΔΣ [10^6 M_sun/pc]) / (R [10^6 pc/h]) = rpDS / R
        """
        d          = self._read_bwpd_file(self.config.ds_file)
        R          = d[:, 0]
        rpDS       = d[:, 1]
        rpDS_upper = d[:, 2]
        mask = (R >= self.config.ds_rp_min) & (R <= self.config.ds_rp_max)
        self.R_arr  = R[mask]
        self.ds_obs = rpDS[mask] / R[mask]
        self.ds_err = (rpDS_upper[mask] - rpDS[mask]) / R[mask]

    @staticmethod
    def _read_bwpd_4col_file(path: str) -> np.ndarray:
        """Parse a 4-column BWPD file with asymmetric (upper + lower) bounds.

        Expected columns (comma/whitespace-delimited, spaces allowed):
            rp_hMpc   DS_central   DS_upper   DS_lower

        The DS columns store the observable (e.g. R×ΔΣ) as plotted on the
        figure y-axis, with separate upper and lower confidence bounds.
        Returns a (N, 4) float array.
        """
        import re
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = [x for x in re.split(r"[,\s]+", line) if x]
                try:
                    rows.append([float(x) for x in fields[:4]])
                except ValueError:
                    continue  # skip header rows
        return np.array(rows)

    def _load_ds_data_bwpd_4col(self):
        """Load ΔΣ from a 4-column BWPD file with asymmetric error bounds.

        Expected columns (comma/whitespace-delimited, spaces allowed):
            rp_hMpc   DS_hMsunpc2   DS_upper_hMsunpc2   DS_lower_hMsunpc2

        - ``rp_hMpc``             : R in h^-1 Mpc
        - ``DS_hMsunpc2``         : ΔΣ in h M_sun/pc^2 — directly the y-axis value
                                    as plotted in the paper figure
        - ``DS_upper_hMsunpc2``   : upper confidence bound on ΔΣ (same units)
        - ``DS_lower_hMsunpc2``   : lower confidence bound on ΔΣ (same units)

        No unit conversion needed: h M_sun/pc^2 matches the predictor output
        (``clustering.py`` returns ΔΣ in Msun h pc⁻²).

        The asymmetric errors are symmetrised as err = (upper − lower) / 2.
        """
        d        = self._read_bwpd_4col_file(self.config.ds_file)
        R        = d[:, 0]
        DS       = d[:, 1]
        DS_upper = d[:, 2]
        DS_lower = d[:, 3]
        mask = (R >= self.config.ds_rp_min) & (R <= self.config.ds_rp_max)
        self.R_arr  = R[mask]
        self.ds_obs = DS[mask]
        self.ds_err = (DS_upper[mask] - DS_lower[mask]) / 2.0

    def _build_predictor(self):
        hmf     = make_hmf(self.config.hmf_backend, pk_func=self._pk_lin.pk_linear)
        hod_cls = HOD_MODELS[self.config.hod_model]
        hod     = hod_cls(hmf) if hod_cls._SINGLE_ARG_INIT else hod_cls(hmf, hmf.bias)
        bnl     = BeyondLinearBiasMead21() if self.config.use_bnl else None
        self.predictor = FullHaloModelPrediction(
            self._pk_lin, hod, HaloProfile(self.theta_cosmo),
            bnl_model=bnl,
        )

    def _build_icov(self):
        if self._cov_wp is not None:
            cov = self._cov_wp
            reg = 0.01 * np.diag(np.diag(cov))
            self.icov_wp = np.linalg.inv(cov + reg)
        else:
            self.icov_wp = np.diag(1.0 / self.wp_err**2)

    # ------------------------------------------------------------------
    # Fixed / free partition

    @property
    def _fixed_params(self) -> dict:
        return {k: v for k, v in self.config.param_init.items()
                if k not in self.config.free_params}

    @property
    def _x0(self) -> np.ndarray:
        return np.array([self.config.param_init[p] for p in self.config.free_params])

    # ------------------------------------------------------------------
    # Log-probability

    def _prior_log_prob(self, theta_vec) -> float:
        """Log-prior: flat (uniform) or Gaussian, per parameter."""
        from hod_mod.fitting.planck_prior import gaussian_log_prior
        log_pi = 0.0
        for name, val in zip(self.config.free_params, theta_vec):
            lo, hi = self.config.param_bounds[name]
            if not (lo <= val <= hi):
                return -np.inf
            ptype = self.config.param_prior_types.get(name, "uniform")
            if ptype == "gaussian":
                mu    = self.config.param_prior_means[name]
                sigma = self.config.param_prior_sigmas[name]
                log_pi += gaussian_log_prior(val, mu, sigma, lo, hi)
        return log_pi

    def _log_prob(self, theta_vec) -> float:
        log_pi = self._prior_log_prob(theta_vec)
        if not np.isfinite(log_pi):
            return -np.inf
        hod_params  = _assemble_hod_params(
            theta_vec, self.config.free_params, self._fixed_params
        )
        theta_cosmo = self._theta_cosmo_call(hod_params)
        try:
            wp_pred = np.asarray(
                self.predictor.wp(
                    jnp.array(self.rp_arr), self.config.pi_max,
                    self.config.z, theta_cosmo, hod_params,
                )
            )
        except Exception:
            return -np.inf
        residual = wp_pred - self.wp_obs
        return log_pi - 0.5 * float(residual @ self.icov_wp @ residual)

    # ------------------------------------------------------------------
    # MAP estimation

    def map_fit(self) -> dict:
        """Maximum a-posteriori fit via Powell / Nelder-Mead.

        Returns
        -------
        dict
            Keys: ``theta``, ``params``, ``chi2``, ``ndof``, ``success``, ``message``.
            ``chi2`` is the pure data chi-squared (prior penalty excluded).
        """
        from scipy.optimize import minimize
        n_free = len(self.config.free_params)
        # Powell converges faster than Nelder-Mead in higher dimensions (≥5 params).
        method = "Powell" if n_free >= 5 else "Nelder-Mead"
        result = minimize(
            lambda x: -self._log_prob(x),
            self._x0,
            method=method,
            options={"maxiter": 50000, "xatol": 1e-5, "fatol": 1e-5,
                     "xtol": 1e-5, "ftol": 1e-5, "disp": False},
        )
        best_theta  = result.x
        best_params = _assemble_hod_params(best_theta, self.config.free_params, self._fixed_params)
        # Compute data chi2 only (excludes Gaussian prior penalty terms).
        theta_cosmo = self._theta_cosmo_call(best_params)
        try:
            wp_pred = np.asarray(
                self.predictor.wp(
                    jnp.array(self.rp_arr), self.config.pi_max,
                    self.config.z, theta_cosmo, best_params,
                )
            )
            chi2_data = float((wp_pred - self.wp_obs) @ self.icov_wp @ (wp_pred - self.wp_obs))
        except Exception:
            chi2_data = float("nan")
        return {
            "theta":   best_theta,
            "params":  best_params,
            "chi2":    chi2_data,
            "ndof":    len(self.rp_arr) - n_free,
            "success": result.success,
            "message": result.message,
        }

    # ------------------------------------------------------------------
    # Ensemble MCMC sampling

    def sample(self, initial_pos: np.ndarray | None = None, progress: bool = True):
        """Run emcee ensemble sampler.

        Burn-in and production are one continuous chain, checkpointed to
        ``<output_dir>/chain.h5`` (emcee HDF backend, flushed every step) and
        resumed automatically: a job killed by the cluster walltime picks up at
        the step it reached instead of starting over.  Before this, the chain
        lived only in memory until the single ``np.savez`` below, so an 8 h
        walltime kill discarded ~8 h of sampling and left an empty output dir —
        which is why seven of the ZM15 benchmarks never landed in the 2026-07
        campaign.  The burn-in is discarded at read-out, never on disk, so the
        saved ``flatchain.npz`` holds the same production samples as before.

        To start a chain over rather than resume it, delete ``chain.h5``: the
        backend is keyed only on (n_walkers, n_free), so it cannot tell that a
        partial chain came from different physics.

        Parameters
        ----------
        initial_pos : array_like, shape (n_walkers, n_free), optional
        progress : bool — show tqdm progress bar

        Returns
        -------
        emcee.EnsembleSampler
        """
        import emcee
        n_free    = len(self.config.free_params)
        n_walkers = self.config.n_walkers
        if initial_pos is None:
            initial_pos = self._default_initial_pos(n_walkers, n_free)

        os.makedirs(self.config.output_dir, exist_ok=True)
        backend_path = os.path.join(self.config.output_dir, "chain.h5")
        backend = emcee.backends.HDFBackend(backend_path)
        already = backend.iteration if os.path.exists(backend_path) else 0
        total   = int(self.config.n_burnin) + int(self.config.n_steps)

        sampler = emcee.EnsembleSampler(n_walkers, n_free, self._log_prob,
                                        backend=backend)
        if already >= total:
            print(f"Chain already complete ({already} >= {total} steps) → {backend_path}")
        elif already == 0:
            print(f"Sampling {total} steps ({self.config.n_burnin} burn-in + "
                  f"{self.config.n_steps} production), {n_walkers} walkers, "
                  f"checkpointing to {backend_path} …")
            sampler.run_mcmc(initial_pos, total, progress=progress)
        else:
            print(f"Resuming from step {already}/{total} "
                  f"({total - already} left) → {backend_path}")
            # A collapsed ensemble makes emcee refuse the resume outright; see
            # hod_mod.fitting.mcmc_resume for why skipping the check is not enough.
            _start = revive_ensemble(backend, None, None, label="fit")
            sampler.run_mcmc(_start, total - already, progress=progress, skip_initial_state_check=True)

        discard  = min(int(self.config.n_burnin), max(sampler.iteration - 1, 0))
        out_path = os.path.join(self.config.output_dir, "flatchain.npz")
        np.savez(out_path, flatchain=sampler.get_chain(discard=discard, flat=True),
                 param_names=np.array(self.config.free_params))
        print(f"Chain saved → {out_path}")
        return sampler

    def _default_initial_pos(self, n_walkers: int, n_free: int) -> np.ndarray:
        x0  = self._x0
        pos = np.zeros((n_walkers, n_free))
        for i, name in enumerate(self.config.free_params):
            lo, hi  = self.config.param_bounds[name]
            width   = 0.05 * (hi - lo)
            pos[:, i] = np.clip(x0[i] + np.random.randn(n_walkers) * width, lo, hi)
        return pos

    # ------------------------------------------------------------------
    # Predictions

    def predict_wp(self, params: dict) -> np.ndarray:
        """Predicted wp(rp) [Mpc/h].

        The per-call cosmology matters here: under ``use_free_cosmo`` this is a
        read-out of the *sampled* point, and it used to read ``self.theta_cosmo``
        -- the fiducial -- so it returned the same curve for every cosmology.
        The likelihood and the optimiser were always right (they go through
        ``_theta_cosmo_call``); only the read-out was wrong, which is why the
        stored chi2 tables are sound and the plotted MAP curves were not.
        """
        return np.asarray(
            self.predictor.wp(
                jnp.array(self.rp_arr), self.config.pi_max,
                self.config.z, self._theta_cosmo_call(params), params,
            )
        )

    # ------------------------------------------------------------------
    # DS data loading (shared with JointFitter and DeltaSigmaFitter)

    def _load_ds_data(self):
        if self.config.ds_format == "bwpd":
            self._load_ds_data_bwpd()
        elif self.config.ds_format == "bwpd_4col":
            self._load_ds_data_bwpd_4col()
        else:
            self._load_ds_data_csv()

    def _load_ds_data_csv(self):
        import pandas as pd
        data = pd.read_csv(self.config.ds_file, comment="#")
        mask = (
            (data["R_hMpc"] >= self.config.ds_rp_min)
            & (data["R_hMpc"] <= self.config.ds_rp_max)
        )
        self.R_arr  = data["R_hMpc"][mask].to_numpy()
        self.ds_obs = data["ds_Msun_h_pc2"][mask].to_numpy()
        self.ds_err = data["ds_err_Msun_h_pc2"][mask].to_numpy()

    def _build_icov_ds(self):
        self.icov_ds = np.diag(1.0 / self.ds_err**2)

    def chi2(self, params: dict) -> float:
        """χ² = (wp_pred - wp_obs)ᵀ icov (wp_pred - wp_obs)."""
        res = self.predict_wp(params) - self.wp_obs
        return float(res @ self.icov_wp @ res)


# ---------------------------------------------------------------------------
# JointFitter — wp + ΔΣ + n_g
# ---------------------------------------------------------------------------

class JointFitter(WpFitter):
    """Fit an HOD model to wp(rp) + ΔΣ(R) + n_g simultaneously.

    Implements the joint likelihood of More+2015 §3.1.

    Parameters
    ----------
    config : FitConfig
        Must have ``ds_file`` set to a non-empty path.
    """

    def __init__(self, config: FitConfig):
        super().__init__(config)
        self._load_ds_data()
        self._build_icov_ds()

    def _log_prob(self, theta_vec) -> float:
        log_pi = self._prior_log_prob(theta_vec)
        if not np.isfinite(log_pi):
            return -np.inf
        hod_params  = _assemble_hod_params(
            theta_vec, self.config.free_params, self._fixed_params
        )
        theta_cosmo = self._theta_cosmo_call(hod_params)
        try:
            wp_pred = np.asarray(
                self.predictor.wp(jnp.array(self.rp_arr), self.config.pi_max,
                                  self.config.z, theta_cosmo, hod_params)
            )
            ds_pred = np.asarray(
                self.predictor.delta_sigma(jnp.array(self.R_arr), self.config.z,
                                           theta_cosmo, hod_params)
            )
            if self.config.fit_ng:
                ng_pred = self.predictor.n_gal(self.config.z, theta_cosmo, hod_params)
        except Exception:
            return -np.inf
        chi2_wp = float((wp_pred - self.wp_obs) @ self.icov_wp @ (wp_pred - self.wp_obs))
        chi2_ds = float((ds_pred - self.ds_obs) @ self.icov_ds @ (ds_pred - self.ds_obs))
        if self.config.fit_ng:
            chi2_ng = float(
                ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
            )
            return log_pi - 0.5 * (chi2_wp + chi2_ds + chi2_ng)
        return log_pi - 0.5 * (chi2_wp + chi2_ds)

    def map_fit(self) -> dict:
        """Maximum a-posteriori fit for the joint wp+ΔΣ+n_g likelihood.

        Returns
        -------
        dict
            Keys: ``theta``, ``params``, ``chi2``, ``ndof``, ``success``, ``message``.
            ``chi2`` is the pure data chi-squared (prior penalty excluded).
            ``ndof`` = n_wp + n_ds + 1 (ng) − n_free_params.
        """
        from scipy.optimize import minimize
        n_free = len(self.config.free_params)
        method = "Powell" if n_free >= 5 else "Nelder-Mead"
        result = minimize(
            lambda x: -self._log_prob(x),
            self._x0,
            method=method,
            options={"maxiter": 50000, "xatol": 1e-5, "fatol": 1e-5,
                     "xtol": 1e-5, "ftol": 1e-5, "disp": False},
        )
        best_theta  = result.x
        best_params = _assemble_hod_params(best_theta, self.config.free_params, self._fixed_params)
        theta_cosmo = self._theta_cosmo_call(best_params)
        try:
            wp_pred = np.asarray(self.predictor.wp(
                jnp.array(self.rp_arr), self.config.pi_max,
                self.config.z, theta_cosmo, best_params))
            ds_pred = np.asarray(self.predictor.delta_sigma(
                jnp.array(self.R_arr), self.config.z, theta_cosmo, best_params))
            chi2_wp = float((wp_pred - self.wp_obs) @ self.icov_wp @ (wp_pred - self.wp_obs))
            chi2_ds = float((ds_pred - self.ds_obs) @ self.icov_ds @ (ds_pred - self.ds_obs))
            n_data  = len(self.rp_arr) + len(self.R_arr)
            if self.config.fit_ng:
                ng_pred   = self.predictor.n_gal(self.config.z, theta_cosmo, best_params)
                chi2_ng   = float(
                    ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
                )
                chi2_data = chi2_wp + chi2_ds + chi2_ng
                n_data   += 1
            else:
                chi2_data = chi2_wp + chi2_ds
        except Exception:
            chi2_data = float("nan")
            n_data    = len(self.rp_arr) + len(self.R_arr)
        return {
            "theta":   best_theta,
            "params":  best_params,
            "chi2":    chi2_data,
            "ndof":    n_data - n_free,
            "success": result.success,
            "message": result.message,
        }

    def predict_ds(self, params: dict) -> np.ndarray:
        """Predicted ΔΣ(R) [M_sun h pc⁻²].

        The per-call cosmology matters here: under ``use_free_cosmo`` this is a
        read-out of the *sampled* point, and it used to read ``self.theta_cosmo``
        -- the fiducial -- so it returned the same curve for every cosmology.
        The likelihood and the optimiser were always right (they go through
        ``_theta_cosmo_call``); only the read-out was wrong, which is why the
        stored chi2 tables are sound and the plotted MAP curves were not.
        """
        return np.asarray(
            self.predictor.delta_sigma(
                jnp.array(self.R_arr), self.config.z,
                self._theta_cosmo_call(params), params,
            )
        )

    def predict_ng(self, params: dict) -> float:
        """Predicted galaxy number density n̄_g [h³ Mpc⁻³].

        The per-call cosmology matters here: under ``use_free_cosmo`` this is a
        read-out of the *sampled* point, and it used to read ``self.theta_cosmo``
        -- the fiducial -- so it returned the same curve for every cosmology.
        The likelihood and the optimiser were always right (they go through
        ``_theta_cosmo_call``); only the read-out was wrong, which is why the
        stored chi2 tables are sound and the plotted MAP curves were not.
        """
        return self.predictor.n_gal(
            self.config.z, self._theta_cosmo_call(params), params)

    def chi2_joint(self, params: dict) -> dict:
        """Per-observable χ² contributions and total."""
        wp_pred = self.predict_wp(params)
        ds_pred = self.predict_ds(params)
        ng_pred = self.predict_ng(params)
        r_wp = wp_pred - self.wp_obs
        r_ds = ds_pred - self.ds_obs
        c_wp = float(r_wp @ self.icov_wp @ r_wp)
        c_ds = float(r_ds @ self.icov_ds @ r_ds)
        c_ng = float(
            ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
        )
        return {"chi2_wp": c_wp, "chi2_ds": c_ds, "chi2_ng": c_ng, "chi2_total": c_wp + c_ds + c_ng}


# ---------------------------------------------------------------------------
# DeltaSigmaFitter — ΔΣ(R) only + optional n_g constraint
# ---------------------------------------------------------------------------

class DeltaSigmaFitter(WpFitter):
    """Fit an HOD model to ΔΣ(R) data only (no wp constraint).

    Uses the galaxy–matter cross-correlation ΔΣ(R) and optionally the
    galaxy number density n_g as constraints.  The log-posterior is:

    .. code-block:: text

        log P(θ|d) = −½ (χ²_ΔΣ + χ²_n_g)

    Config requirements:

    - ``ds_file`` must point to a valid CSV with columns
      ``R_hMpc``, ``ds_Msun_h_pc2``, ``ds_err_Msun_h_pc2``.
    - ``ng_obs`` and ``ng_frac_err`` control the n_g constraint weight
      (defaults from :class:`FitConfig` apply if not set in YAML).
    - No ``data_file`` / ``data.file`` is required.

    Parameters
    ----------
    config : FitConfig
        Must have ``ds_file`` set.
    """

    def __init__(self, config: FitConfig):
        self._setup_common(config)
        self._build_predictor()
        self._load_ds_data()
        self._build_icov_ds()

    def _log_prob(self, theta_vec) -> float:
        log_pi = self._prior_log_prob(theta_vec)
        if not np.isfinite(log_pi):
            return -np.inf
        hod_params  = _assemble_hod_params(
            theta_vec, self.config.free_params, self._fixed_params
        )
        theta_cosmo = self._theta_cosmo_call(hod_params)
        try:
            ds_pred = np.asarray(
                self.predictor.delta_sigma(jnp.array(self.R_arr), self.config.z,
                                           theta_cosmo, hod_params)
            )
            if self.config.fit_ng:
                ng_pred = self.predictor.n_gal(self.config.z, theta_cosmo, hod_params)
        except Exception:
            return -np.inf
        chi2_ds = float((ds_pred - self.ds_obs) @ self.icov_ds @ (ds_pred - self.ds_obs))
        if self.config.fit_ng:
            chi2_ng = float(
                ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
            )
            return log_pi - 0.5 * (chi2_ds + chi2_ng)
        return log_pi - 0.5 * chi2_ds

    def map_fit(self) -> dict:
        """Maximum a-posteriori fit for the ΔΣ-only likelihood.

        Returns
        -------
        dict
            Keys: ``theta``, ``params``, ``chi2``, ``ndof``, ``success``, ``message``.
            ``chi2`` is the pure data chi-squared (prior penalty excluded).
        """
        from scipy.optimize import minimize
        n_free = len(self.config.free_params)
        method = "Powell" if n_free >= 5 else "Nelder-Mead"
        result = minimize(
            lambda x: -self._log_prob(x),
            self._x0,
            method=method,
            options={"maxiter": 50000, "xatol": 1e-5, "fatol": 1e-5,
                     "xtol": 1e-5, "ftol": 1e-5, "disp": False},
        )
        best_theta  = result.x
        best_params = _assemble_hod_params(best_theta, self.config.free_params, self._fixed_params)
        theta_cosmo = self._theta_cosmo_call(best_params)
        try:
            ds_pred = np.asarray(self.predictor.delta_sigma(
                jnp.array(self.R_arr), self.config.z, theta_cosmo, best_params))
            ng_pred = self.predictor.n_gal(self.config.z, theta_cosmo, best_params)
            chi2_ds = float((ds_pred - self.ds_obs) @ self.icov_ds @ (ds_pred - self.ds_obs))
            chi2_ng = float(
                ((ng_pred - self.config.ng_obs) / (self.config.ng_frac_err * self.config.ng_obs)) ** 2
            )
            chi2_data = chi2_ds + chi2_ng
        except Exception:
            chi2_data = float("nan")
        return {
            "theta":   best_theta,
            "params":  best_params,
            "chi2":    chi2_data,
            "ndof":    len(self.R_arr) + 1 - n_free,
            "success": result.success,
            "message": result.message,
        }

    def predict_ds(self, params: dict) -> np.ndarray:
        """Predicted ΔΣ(R) [M_sun h pc⁻²].

        The per-call cosmology matters here: under ``use_free_cosmo`` this is a
        read-out of the *sampled* point, and it used to read ``self.theta_cosmo``
        -- the fiducial -- so it returned the same curve for every cosmology.
        The likelihood and the optimiser were always right (they go through
        ``_theta_cosmo_call``); only the read-out was wrong, which is why the
        stored chi2 tables are sound and the plotted MAP curves were not.
        """
        return np.asarray(
            self.predictor.delta_sigma(
                jnp.array(self.R_arr), self.config.z,
                self._theta_cosmo_call(params), params,
            )
        )

    def chi2(self, params: dict) -> float:
        """χ² for ΔΣ-only (excludes n_g term)."""
        res = self.predict_ds(params) - self.ds_obs
        return float(res @ self.icov_ds @ res)


# Backward-compatibility alias (WpFitterFITS was eliminated; use WpFitter with data_format="fits")
WpFitterFITS = WpFitter
