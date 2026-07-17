"""Joint BGS M*>10 (S1) full-model likelihood: galaxies + hot gas + AGN.

Assembles a single Gaussian log-likelihood over the observable set decided in Step 1
(see docs / plan):

  galaxy (ZM15)      n_gal (SMF integral), wp(rp), ESD ΔΣ (DES+HSC+KIDS, rp>2 Mpc/h,
                     with a free central point mass ``log10_M_star_cen``)
  X-ray cross        15 narrow 0.1-keV bands (0.5–2 keV) + broad band, via the
                     PRECOMPUTED transfer-grid band model of
                     :mod:`hod_mod.scripts.fitting.fit_xray_joint_bands` (free LX–M,
                     kT–M, shape, AGN duty cycle, gas Z) — feasible for MCMC.
  AGN (Powell)       XLF Φ(L_X) at z=0.1 & 0.4 and halo bias b(L_X), forward-modelled by
                     :class:`hod_mod.agn.powell.PowellAGNModel`.

Two modes:
  * fixed-ZM15 (Step 2): the 13 ZM15 params are held at the BGS mass-bin posterior median;
    only the X-ray + AGN (+ point mass) params are free.  Galaxy predictions that do not
    depend on the point mass are precomputed once.
  * free-ZM15 (Step 3): the 13 ZM15 params are additionally free, with Gaussian priors
    from the mass-bin posterior (mean=median, σ=(p84−p16)/2).

MAP runs locally (scipy); MCMC uses a resumable emcee HDF backend (for dahu).

NOTE: run with ``HOD_MOD_DATA_DIR=/home/comparat/data/hod_mod_data`` (the complete data
root — the bare ``/home/comparat/data`` is missing ``xray_bands/``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import h5py

from hod_mod.core.power_spectrum import LinearPowerSpectrum, default_pk_linear
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.agn.powell import PowellAGNModel
from hod_mod.paths import data_root, results_root

from hod_mod.scripts.fitting import fit_xray_joint_bands as XB
from hod_mod.scripts.fitting.fit_comparat2025 import (
    _THETA_COSMO, SAMPLES, _sum_stat_path, load_esd_data, load_smf_data,
    load_wp_data, load_data as _load_xray_broad, _predict_smf,
)

_H = float(_THETA_COSMO["h"])

# ZM15 occupation keys (13) held fixed / freed together.
_ZM15_KEYS = ["lg_m1h", "lg_m0star", "beta", "delta", "gamma", "sigma_lnmstar",
              "eta", "fc", "bsat", "beta_sat", "bcut", "beta_cut", "alpha_sat"]

# Powell AGN free params (XLF + bias forward model).  Bounds + fiducials.
_AGN_PARAMS = {
    #  name             (lo,    hi,    fiducial)
    "agn_mu_bh":        (7.0,   8.5,   7.76),    # M_BH–M* norm
    "agn_log10_lstar":  (-2.5, -0.2,  float(np.log10(0.13))),  # ERDF break
    "agn_delta1":       (0.0,   1.2,   0.30),    # ERDF faint slope
    "agn_delta2":       (2.0,   5.0,   3.70),    # ERDF bright slope
    "log10_ferdf":      (-3.5,  0.0,  -1.5),     # active fraction (XLF normalisation)
}
_AGN_NAMES = list(_AGN_PARAMS)

# ESD central point mass (free, informative prior from sample mean M*).
_POINT_MASS = ("log10_M_star_cen", 10.0, 12.5, 11.3, 0.3)   # name, lo, hi, mean, sig


def load_roster26_xlf(z: float) -> tuple[np.ndarray, np.ndarray]:
    """Roster 2026 hard-band (2–10 keV) XLF: (log10 L_X, Φ [Mpc^-3 dex^-1])."""
    tag = "z0p1" if abs(z - 0.1) < 0.05 else "z0p4"
    p = data_root() / "erosita" / "observations" / "XLF_Roster26" / f"roster26_{tag}.csv"
    arr = np.genfromtxt(p, delimiter=",", comments="#")
    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr[:, 0], arr[:, 1]


def load_agn_bias_compilation(refs=None) -> dict:
    """AGN large-scale halo bias vs soft (0.5–2 keV) luminosity compilation.

    The file columns are ``redshift, LX0.5-2keV, bias, bias_err, Ref``.  If
    ``refs`` is given (e.g. ``("Comparat23", "Krumpe15")``) only rows from those
    references are kept; ``None`` keeps all.
    """
    p = data_root() / "erosita" / "observations" / "AGN_bias_halo_compilation.txt"
    arr = np.atleast_2d(np.genfromtxt(p, delimiter=",", comments="#", usecols=(0, 1, 2, 3)))
    ref = np.atleast_1d(np.char.strip(
        np.genfromtxt(p, delimiter=",", comments="#", usecols=(4,), dtype=str).astype(str)))
    keep = np.isfinite(arr).all(axis=1)
    if refs is not None:
        want = {str(r).strip() for r in refs}
        keep = keep & np.array([r in want for r in ref])
    arr = arr[keep]
    return dict(z=arr[:, 0], log10lx_soft=arr[:, 1], bias=arr[:, 2], bias_err=arr[:, 3])


def load_zm15_median() -> dict:
    p = results_root() / "bgs_zm15_joint_wp_ngal" / "posterior_summary.json"
    return json.load(open(p))


class JointFull:
    """Joint galaxies + gas + AGN likelihood for BGS S1."""

    XLF_Z = (0.1, 0.4)
    XLF_FRAC_ERR = 0.10

    def __init__(self, sample="S1", free_zm15=False,
                 observables=("ngal", "wp", "esd", "xray_bands", "xray_broad",
                              "xlf", "agn_bias"),
                 rp_min_wp=0.5, rp_min_esd=2.0, f_sys=0.05,
                 hmf_backend="tinker08", ngal_frac_err=0.05, verbose=True,
                 esd_surveys=("DES", "HSC", "KIDS"), esd_rp_max=np.inf,
                 xlf_z=(0.1, 0.4), xlf_lx_min=-np.inf, agn_bias_refs=None,
                 kt_prior_sig=None):
        self.sample_name = sample
        self.free_zm15 = bool(free_zm15)
        self.obs = set(observables)
        self.rp_min_wp = rp_min_wp
        self.rp_min_esd = rp_min_esd
        self.esd_surveys = tuple(esd_surveys)
        self.esd_rp_max = float(esd_rp_max)
        self.xlf_z = tuple(float(z) for z in xlf_z)
        self.xlf_lx_min = float(xlf_lx_min)
        self.agn_bias_refs = tuple(agn_bias_refs) if agn_bias_refs is not None else None
        self.kt_prior_sig = tuple(kt_prior_sig) if kt_prior_sig is not None else None
        self.f_sys = f_sys
        self.ngal_frac_err = ngal_frac_err
        self.z = float(SAMPLES[sample]["zmean"])
        self._log = print if verbose else (lambda *a, **k: None)

        # --- ZM15 median (+ priors for free mode) --------------------------
        summ = load_zm15_median()
        self.zm15_med = {k: float(summ[k]["med"]) for k in _ZM15_KEYS}
        self.zm15_prior_sig = {k: max(1e-3, 0.5 * (summ[k]["p84"] - summ[k]["p16"]))
                               for k in _ZM15_KEYS}

        # --- galaxy model (ZM15 + full halo model) -------------------------
        self._log("[joint] building galaxy model ...", flush=True)
        pk = default_pk_linear()
        self.hmf = make_hmf(hmf_backend, pk_func=pk.pk_linear)
        hp = HaloProfile({"H0": _THETA_COSMO["h"] * 100.0, "Om0": _THETA_COSMO["Omega_m"],
                          "Ob0": _THETA_COSMO["Omega_b"], "ns": _THETA_COSMO["n_s"],
                          "sigma8": 0.811}, cm_relation="diemer19")
        hod = ZuMandelbaum15HODModel(self.hmf, self.hmf.bias)
        self.fhmp = FullHaloModelPrediction(pk, hod, hp)
        self._infra = type("infra", (), {"fhmp": self.fhmp})()   # shim for _predict_smf

        # --- data -----------------------------------------------------------
        self._load_galaxy_data()
        self._load_agn_data()
        self._powell = {}                       # z -> PowellAGNModel (built lazily)

        # --- X-ray band sector (reuse the transfer-grid band model) --------
        if {"xray_bands", "xray_broad"} & self.obs:
            self._log("[joint] building X-ray transfer grid (cached) ...", flush=True)
            XB._apply_candidate("baseline")
            self._xb_params = XB._PARAMS
            (G_grid, log10_m500c, ez, agn_dc1, bd, mask,
             m200, r200) = XB._precompute(sample, hmf_backend)
            err = np.sqrt(bd["wtheta_err"] ** 2 + (f_sys * np.abs(bd["wtheta"])) ** 2)
            from scipy.interpolate import RegularGridInterpolator
            G_interp = RegularGridInterpolator(
                (XB._P2_GRID, XB._RMAX_GRID),
                G_grid.reshape(XB._P2_GRID.size, XB._RMAX_GRID.size, -1),
                method="linear", bounds_error=False, fill_value=None)
            cool_bands, cool_broad = XB._band_cooling()
            # Native-DPM J_b(T_0, Z | p2, r_max) — the X-ray leg now shares
            # {n_e,0.3, beta_n, P_0.3, beta_P} with the tSZ Sigma_y leg.
            j_grid, v_grid = XB._j_grid(sample)
            j_interp, v_interp = XB._make_j_interp(j_grid, v_grid)
            self._S = dict(G_interp=G_interp, nth=bd["theta_arcsec"].size,
                           log10_m500c=log10_m500c, ez=ez, agn_dc1=agn_dc1,
                           wtheta=bd["wtheta"], err=err, mask=mask,
                           th_as=bd["theta_arcsec"], fb=XB._agn_band_fractions(),
                           arf=XB._arf_band_weights(), cool_bands=cool_bands,
                           cool_broad=cool_broad, kT_lo=0.09, kT_hi=19.0,
                           c_obs_total=XB.J._c_obs_total(sample),
                           srx=float(_load_xray_broad(sample)["beckground"][0]),
                           n_pts=int(mask.sum()) * XB._NB, c_total=1.0,
                           m200=m200, r200=r200, h=float(_THETA_COSMO["h"]),
                           J_interp=j_interp, V_interp=v_interp)
            c_total, _ = XB._anchor({sample: self._S}, sample)
            self._S["c_total"] = c_total
            self._log(f"[joint]   c_total anchor = {c_total:.3e}", flush=True)
            if "xray_broad" in self.obs:
                self._load_xray_broad_data()

        # --- parameter layout ----------------------------------------------
        self._build_param_layout()

        # --- fixed-ZM15 precompute (galaxy predictions are constant) --------
        self._gal_cache = None
        if not self.free_zm15:
            self._gal_cache = self._predict_galaxy(self._hod_params(self.zm15_med))
        self._log("[joint] ready.", flush=True)

    # ------------------------------------------------------------------ data
    def _load_galaxy_data(self):
        self.data_gal = {}
        if "wp" in self.obs or "ngal" in self.obs:
            d = load_wp_data(self.sample_name, rp_min=self.rp_min_wp)
            self.data_gal["wp"] = dict(rp=d["rp"], wp=d["wp"],
                                       err=np.sqrt(np.diag(d["cov"])), pi_max=d["pi_max"])
        if "ngal" in self.obs:
            smf = load_smf_data(self.sample_name)   # (Mpc/h)^-3 dex^-1
            ngal = float(np.trapezoid(smf["phi"], smf["log10mstar"]))
            self.data_gal["ngal"] = dict(value=ngal, err=self.ngal_frac_err * ngal,
                                         grid=smf["log10mstar"])
        if "esd" in self.obs:
            self.data_gal["esd"] = {}
            for sv in self.esd_surveys:
                e = load_esd_data(self.sample_name, sv)
                m = (e["rp"] > self.rp_min_esd) & (e["rp"] < self.esd_rp_max)
                self.data_gal["esd"][sv] = dict(rp=e["rp"][m], ds=e["delta_sigma"][m],
                                                err=e["delta_sigma_err"][m])

    def _load_agn_data(self):
        self.data_agn = {}
        if "xlf" in self.obs:
            self.data_agn["xlf"] = {}
            for z in self.xlf_z:
                lx, phi = load_roster26_xlf(z)
                m = np.asarray(lx) > self.xlf_lx_min
                self.data_agn["xlf"][z] = dict(log10lx=lx[m], phi=phi[m],
                                               err=self.XLF_FRAC_ERR * phi[m])
        if "agn_bias" in self.obs:
            self.data_agn["bias"] = load_agn_bias_compilation(refs=self.agn_bias_refs)

    def _load_xray_broad_data(self):
        d = _load_xray_broad(self.sample_name)
        self.data_xray_broad = dict(theta_as=d["theta_arcsec"], wtheta=d["wtheta"],
                                    err=np.sqrt(d["wtheta_err"] ** 2 +
                                                (self.f_sys * np.abs(d["wtheta"])) ** 2))

    # ------------------------------------------------------------- params
    def _build_param_layout(self):
        names, lo, hi, x0, pri_mu, pri_sig = [], [], [], [], [], []

        def add(n, l, h, init, mu=np.nan, sig=np.inf):
            names.append(n); lo.append(l); hi.append(h); x0.append(init)
            pri_mu.append(mu); pri_sig.append(sig)

        if self.free_zm15:
            for k in _ZM15_KEYS:
                add(k, self.zm15_med[k] - 5 * self.zm15_prior_sig[k],
                    self.zm15_med[k] + 5 * self.zm15_prior_sig[k],
                    self.zm15_med[k], self.zm15_med[k], self.zm15_prior_sig[k])
        if "esd" in self.obs:
            n, l, h, mu, sig = _POINT_MASS
            add(n, l, h, mu, mu, sig)
        if {"xray_bands", "xray_broad"} & self.obs:
            mu8, sig8, bnd8 = XB._MU8, np.array(XB._SIG8, float), XB._BND8
            if self.kt_prior_sig is not None:
                # tighten the kt_norm (idx 2) / kt_slope (idx 3) Gaussian priors so
                # the kT-M relation cannot extrapolate far from the cluster scaling
                sig8[2], sig8[3] = float(self.kt_prior_sig[0]), float(self.kt_prior_sig[1])
            # In-bounds start: the band-fit's canonical seed.  _MU8 carries 0.0
            # placeholders for the flat-prior shape/DC params (p2, r_max, log10DC),
            # which are OUTSIDE their bounds — never use _MU8 as x0 for those.
            xb_x0 = [44.7, 1.61, 0.40, 0.60, 0.6, 4.0, -1.5, 0.30, 1.8]
            for i, pn in enumerate(self._xb_params):
                init = float(np.clip(xb_x0[i], bnd8[i, 0] + 1e-6, bnd8[i, 1] - 1e-6))
                add(pn, float(bnd8[i, 0]), float(bnd8[i, 1]), init,
                    float(mu8[i]), float(sig8[i]))
        if {"xlf", "agn_bias"} & self.obs:
            for pn in _AGN_NAMES:
                l, h, init = _AGN_PARAMS[pn]
                add(pn, l, h, init)

        self.names = names
        self.lo = np.array(lo); self.hi = np.array(hi)
        self.x0 = np.array(x0)
        self.pri_mu = np.array(pri_mu); self.pri_sig = np.array(pri_sig)
        self.ndim = len(names)
        self._idx = {n: i for i, n in enumerate(names)}

    def _hod_params(self, zm15: dict, log10_M_star_cen=None) -> dict:
        hp = ZuMandelbaum15HODModel.default_params()
        hp.update(zm15)
        hp["log10m_star_thresh"] = 10.0
        if log10_M_star_cen is not None:
            hp["log10_M_star_cen"] = float(log10_M_star_cen)
        return hp

    def _decode(self, theta) -> dict:
        v = {n: float(theta[i]) for i, n in enumerate(self.names)}
        zm15 = {k: v.get(k, self.zm15_med[k]) for k in _ZM15_KEYS}
        return dict(v=v, zm15=zm15)

    # -------------------------------------------------------- predictions
    def _predict_galaxy(self, hod_params) -> dict:
        out = {}
        if "ngal" in self.obs:
            # n_gal = N(>M*_lo) − N(>M*_hi) over the SMF data grid, in (Mpc/h)^-3,
            # matching ∫Φ dlog10M* of the data.  Computed via the JITTED threshold
            # occupation (nc_ns) rather than fhmp.n_gal's disable_jit path — same
            # numbers, ~1000× faster (matters for the free-ZM15 Step-3 MCMC).
            grid = self.data_gal["ngal"]["grid"]
            out["ngal"] = (self._ngal_fast(hod_params, float(grid[0]))
                           - self._ngal_fast(hod_params, float(grid[-1])))
        if "wp" in self.obs:
            rp = self.data_gal["wp"]["rp"]
            out["wp"] = np.asarray(self.fhmp.wp(rp, pi_max=self.data_gal["wp"]["pi_max"],
                                                z=self.z, theta_cosmo=_THETA_COSMO,
                                                hod_params=hod_params), float)
        if "esd" in self.obs:
            out["esd"] = {}
            for sv, d in self.data_gal["esd"].items():
                out["esd"][sv] = np.asarray(
                    self.fhmp.delta_sigma(d["rp"], z=self.z, theta_cosmo=_THETA_COSMO,
                                          hod_params=hod_params), float)
        return out

    def _ngal_fast(self, hod_params, thresh) -> float:
        """N(>M*=thresh) [(Mpc/h)^-3] via the jitted threshold occupation.

        Replicates ``FullHaloModelPrediction.n_gal`` (∫ dn/dM·(N_c+N_s) dM on the
        model's native grid) but WITHOUT the ``jax.disable_jit`` wrapper, so it is
        fast enough to call on every free-ZM15 likelihood evaluation.
        """
        fhmp = self.fhmp
        ck = fhmp._cosmo_cache_key(self.z, _THETA_COSMO)
        if ck not in fhmp._static_cache:
            fhmp._pk_tables_full(self.z, _THETA_COSMO, hod_params)
        sc = fhmp._static_cache[ck]
        grid = fhmp._hod._log10m_grid
        hp = dict(hod_params)
        hp["log10m_star_thresh"] = float(thresh)
        hp.pop("log10m_star_max", None)
        nc, ns = fhmp._hod.nc_ns(grid, hp)
        return float(np.trapezoid(np.asarray(sc["dndm_np"]) *
                                  (np.asarray(nc) + np.asarray(ns)), sc["m_np"]))

    def _powell_at(self, z):
        zk = round(float(z), 3)
        if zk not in self._powell:
            self._powell[zk] = PowellAGNModel(_THETA_COSMO, self.hmf, z_mean=zk)
        return self._powell[zk]

    def _apply_agn_params(self, pw, v):
        pw.set_params(mbh_m=(v["agn_mu_bh"], pw.al_bh, pw.sig_bh),
                      erdf=(v["agn_log10_lstar"], v["agn_delta1"], v["agn_delta2"]),
                      log10_ferdf=v["log10_ferdf"])

    # --------------------------------------------------------------- chi2
    def _chi2_galaxy(self, v, zm15) -> float:
        if self._gal_cache is not None and not self.free_zm15:
            pred = self._gal_cache
        else:
            pred = self._predict_galaxy(self._hod_params(zm15))
        chi2 = 0.0
        if "ngal" in self.obs:
            d = self.data_gal["ngal"]
            chi2 += ((pred["ngal"] - d["value"]) / d["err"]) ** 2
        if "wp" in self.obs:
            d = self.data_gal["wp"]
            err = np.sqrt(d["err"] ** 2 + (self.f_sys * np.abs(d["wp"])) ** 2)
            chi2 += float(np.sum(((pred["wp"] - d["wp"]) / err) ** 2))
        if "esd" in self.obs:
            pm = 10.0 ** v.get("log10_M_star_cen", -np.inf) if "log10_M_star_cen" in v else 0.0
            for sv, d in self.data_gal["esd"].items():
                # point-mass ΔΣ = M / (π rp²): rp[Mpc/h], want Msun h/pc² -> /1e12
                ds_pm = pm / (np.pi * (d["rp"] * 1e6) ** 2) if pm else 0.0
                model = pred["esd"][sv] + ds_pm
                chi2 += float(np.sum(((model - d["ds"]) / d["err"]) ** 2))
        return float(chi2)

    def _chi2_agn(self, v) -> float:
        chi2 = 0.0
        if "xlf" in self.obs:
            for z in self.xlf_z:
                pw = self._powell_at(z); self._apply_agn_params(pw, v)
                grid, phi = pw.xlf(band="hard")
                phi_phys = np.asarray(phi) * _H ** 3
                d = self.data_agn["xlf"][z]
                model = np.interp(d["log10lx"], grid, phi_phys, left=1e-30, right=1e-30)
                chi2 += float(np.sum(((model - d["phi"]) / d["err"]) ** 2))
        if "agn_bias" in self.obs:
            d = self.data_agn["bias"]
            for zu in np.unique(np.round(d["z"], 3)):
                pw = self._powell_at(float(zu)); self._apply_agn_params(pw, v)
                sel = np.round(d["z"], 3) == zu
                model = pw.agn_bias_of_lx(d["log10lx_soft"][sel], band="soft")
                chi2 += float(np.sum(((model - d["bias"][sel]) / d["bias_err"][sel]) ** 2))
        return float(chi2)

    def _chi2_xray(self, v) -> float:
        if not ({"xray_bands", "xray_broad"} & self.obs):
            return 0.0
        p = np.array([v[pn] for pn in self._xb_params])
        chi2 = 0.0
        if "xray_bands" in self.obs:
            chi2 += XB._chi2_sample(p, self._S)
        if "xray_broad" in self.obs and hasattr(self, "data_xray_broad"):
            gas, agn = XB._components_bands(p, self._S)         # (Nb, Nth) on band grid
            # The broad band is the AVERAGE of the single bands (each reconstructed
            # band is normalised near the broad amplitude, so their mean — not sum —
            # reproduces the broad 0.5–2 keV w(θ)).  A genuine absolute constraint.
            broad_model = np.mean(gas + agn, axis=0)
            d = self.data_xray_broad
            broad_at = np.interp(d["theta_as"], self._S["th_as"], broad_model)
            chi2 += float(np.sum(((broad_at - d["wtheta"]) / d["err"]) ** 2))
        return float(chi2)

    def chi2_breakdown(self, theta) -> dict:
        dec = self._decode(theta)
        return dict(galaxy=self._chi2_galaxy(dec["v"], dec["zm15"]),
                    xray=self._chi2_xray(dec["v"]),
                    agn=self._chi2_agn(dec["v"]))

    # ------------------------------------------------------ model for plots
    def predict(self, theta) -> dict:
        """Model curves for every active observable at ``theta``, aligned to the
        data abscissae — for best-fit / posterior-median overlays.

        Mirrors the ``_chi2_*`` methods so plotted curves are exactly what the
        likelihood sees.  Galaxy/AGN entries align to ``self.data_gal`` /
        ``self.data_agn``; the ``xray_bands`` entry carries the full ``(Nb, Ntheta)``
        gas / AGN / total on ``self._S['th_as']`` and its data ``wtheta``/``err``/``mask``.
        """
        dec = self._decode(theta); v = dec["v"]; zm15 = dec["zm15"]
        out = {}
        gal = self._predict_galaxy(self._hod_params(zm15))
        if "ngal" in self.obs:
            out["ngal"] = float(gal["ngal"])
        if "wp" in self.obs:
            out["wp"] = np.asarray(gal["wp"], float)
        if "esd" in self.obs:
            pm = 10.0 ** v["log10_M_star_cen"] if "log10_M_star_cen" in v else 0.0
            out["esd"] = {}
            for sv, d in self.data_gal["esd"].items():
                ds_pm = pm / (np.pi * (d["rp"] * 1e6) ** 2) if pm else 0.0
                out["esd"][sv] = np.asarray(gal["esd"][sv], float) + ds_pm
        if "xlf" in self.obs:
            out["xlf"] = {}
            for z in self.xlf_z:
                pw = self._powell_at(z); self._apply_agn_params(pw, v)
                grid, phi = pw.xlf(band="hard")
                phi_phys = np.asarray(phi) * _H ** 3
                d = self.data_agn["xlf"][z]
                out["xlf"][z] = np.interp(d["log10lx"], grid, phi_phys,
                                          left=1e-30, right=1e-30)
        if "agn_bias" in self.obs:
            d = self.data_agn["bias"]
            model = np.empty(np.shape(d["bias"]), float)
            for zu in np.unique(np.round(d["z"], 3)):
                pw = self._powell_at(float(zu)); self._apply_agn_params(pw, v)
                sel = np.round(d["z"], 3) == zu
                model[sel] = pw.agn_bias_of_lx(d["log10lx_soft"][sel], band="soft")
            out["agn_bias"] = model
        if {"xray_bands", "xray_broad"} & self.obs:
            p = np.array([v[pn] for pn in self._xb_params])
            gas, agn = XB._components_bands(p, self._S)
            out["xray_bands"] = dict(th_as=self._S["th_as"], gas=gas, agn=agn,
                                     total=gas + agn, wtheta=self._S["wtheta"],
                                     err=self._S["err"], mask=self._S["mask"])
            if "xray_broad" in self.obs and hasattr(self, "data_xray_broad"):
                out["xray_broad"] = np.interp(self.data_xray_broad["theta_as"],
                                              self._S["th_as"], np.mean(gas + agn, axis=0))
        return out

    def predict_components(self, theta) -> dict:
        """Per-observable component decomposition for the plots: ``wp`` and ESD
        split into 1-halo / 2-halo (ESD also carries the central point mass), and
        the X-ray cross split into hot-gas / AGN.  In fixed-ZM15 mode the galaxy
        terms are theta-independent (only the ESD point mass moves)."""
        dec = self._decode(theta); v = dec["v"]; zm15 = dec["zm15"]
        hp = self._hod_params(zm15)
        out = {}
        if "wp" in self.obs:
            d = self.data_gal["wp"]
            c = self.fhmp.wp_components(d["rp"], pi_max=d["pi_max"], z=self.z,
                                        theta_cosmo=_THETA_COSMO, hod_params=hp)
            out["wp"] = {k: np.asarray(c[k], float) for k in ("1h", "2h", "total")}
        if "esd" in self.obs:
            pm = 10.0 ** v["log10_M_star_cen"] if "log10_M_star_cen" in v else 0.0
            out["esd"] = {}
            for sv, d in self.data_gal["esd"].items():
                c = self.fhmp.delta_sigma_components(d["rp"], z=self.z,
                                                     theta_cosmo=_THETA_COSMO, hod_params=hp)
                ds_pm = (pm / (np.pi * (d["rp"] * 1e6) ** 2)) if pm else np.zeros_like(d["rp"])
                out["esd"][sv] = dict(one_h=np.asarray(c["1h"], float),
                                      two_h=np.asarray(c["2h"], float),
                                      point_mass=np.asarray(ds_pm, float),
                                      total=np.asarray(c["total"], float) + np.asarray(ds_pm, float))
        if {"xray_bands", "xray_broad"} & self.obs:
            p = np.array([v[pn] for pn in self._xb_params])
            gas, agn = XB._components_bands(p, self._S)
            out["xray"] = dict(th_as=self._S["th_as"], gas=gas, agn=agn, total=gas + agn,
                               wtheta=self._S["wtheta"], err=self._S["err"], mask=self._S["mask"])
            if hasattr(self, "data_xray_broad"):
                th = self._S["th_as"]; d = self.data_xray_broad
                out["xray_broad"] = dict(
                    theta_as=d["theta_as"], wtheta=d["wtheta"], err=d["err"],
                    gas=np.interp(d["theta_as"], th, np.mean(gas, axis=0)),
                    agn=np.interp(d["theta_as"], th, np.mean(agn, axis=0)),
                    total=np.interp(d["theta_as"], th, np.mean(gas + agn, axis=0)))
        return out

    # --------------------------------------------------------- probability
    def log_prior(self, theta) -> float:
        theta = np.asarray(theta, float)
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        fin = np.isfinite(self.pri_sig)
        d = (theta[fin] - self.pri_mu[fin]) / self.pri_sig[fin]
        return -0.5 * float(np.sum(d ** 2))

    def log_prob(self, theta) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        dec = self._decode(theta)
        chi2 = (self._chi2_galaxy(dec["v"], dec["zm15"])
                + self._chi2_xray(dec["v"]) + self._chi2_agn(dec["v"]))
        if not np.isfinite(chi2):
            return -np.inf
        return lp - 0.5 * chi2

    def n_data(self) -> int:
        n = 0
        if "ngal" in self.obs: n += 1
        if "wp" in self.obs: n += len(self.data_gal["wp"]["wp"])
        if "esd" in self.obs: n += sum(len(d["ds"]) for d in self.data_gal["esd"].values())
        if "xray_bands" in self.obs: n += self._S["n_pts"]
        if "xray_broad" in self.obs and hasattr(self, "data_xray_broad"):
            n += len(self.data_xray_broad["wtheta"])
        if "xlf" in self.obs: n += sum(len(d["phi"]) for d in self.data_agn["xlf"].values())
        if "agn_bias" in self.obs: n += len(self.data_agn["bias"]["bias"])
        return n

    # ---------------------------------------------------------------- MAP
    def map_fit(self, method="Nelder-Mead", maxiter=3000, n_starts=2):
        from scipy.optimize import minimize
        neg = lambda p: -self.log_prob(p)
        best = None
        jitters = (0.0, 0.05, -0.05, 0.1)[:max(1, n_starts)]
        for jitter in jitters:
            x = self.x0 * (1.0 + jitter) if jitter else self.x0.copy()
            x = np.clip(x, self.lo + 1e-6, self.hi - 1e-6)
            o = minimize(neg, x, method=method,
                         options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2))
            if best is None or o.fun < best.fun:
                best = o
        chi2 = -2.0 * self.log_prob(best.x) + 2.0 * self.log_prior(best.x)
        ndof = max(self.n_data() - self.ndim, 1)
        return dict(theta=best.x.tolist(), names=self.names,
                    params={n: float(best.x[i]) for i, n in enumerate(self.names)},
                    chi2=float(chi2), ndof=int(ndof), chi2_per_dof=float(chi2 / ndof),
                    breakdown=self.chi2_breakdown(best.x), success=bool(best.success))

    def sample(self, out_dir, n_walkers=48, n_burnin=500, n_steps=2000, x_start=None):
        import emcee
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        x0 = np.asarray(x_start if x_start is not None else self.x0, float)
        if x0.shape != (self.ndim,):
            # A stale/incompatible seed (e.g. a MAP saved with a different param
            # set) must not seed the walkers -> fall back to the fiducial x0.
            print(f"[sample] x_start size {x0.size} != model ndim {self.ndim}; "
                  f"ignoring it and seeding from the fiducial x0", flush=True)
            x0 = self.x0.copy()
        nw = max(n_walkers, 2 * self.ndim + 2)
        rng = np.random.default_rng(42)
        scale = np.where(np.isfinite(self.pri_sig), self.pri_sig, 0.05 * (self.hi - self.lo))
        backend = emcee.backends.HDFBackend(str(out / "chain.h5"))
        try:
            resumed = backend.iteration > 0
        except (AttributeError, OSError, KeyError):
            resumed = False
        if resumed and backend.shape[1] != self.ndim:
            # An existing chain of a different dimensionality can't be continued.
            print(f"[sample] existing chain.h5 has ndim {backend.shape[1]} != "
                  f"{self.ndim}; starting a fresh chain", flush=True)
            resumed = False
        if not resumed:
            p0 = x0[None, :] + 1e-3 * scale[None, :] * rng.standard_normal((nw, self.ndim))
            p0 = np.clip(p0, self.lo + 1e-6, self.hi - 1e-6)
            backend.reset(nw, self.ndim)
            start = p0
        else:
            start = None
            nw = backend.shape[0]
        sampler = emcee.EnsembleSampler(nw, self.ndim, self.log_prob, backend=backend)
        total = n_burnin + n_steps
        remaining = total - backend.iteration
        if remaining > 0:
            sampler.run_mcmc(start, remaining, progress=False)
        flat = sampler.get_chain(discard=min(n_burnin, backend.iteration // 2), flat=True)
        np.savez(out / "flatchain.npz", flatchain=flat, param_names=np.array(self.names))
        pct = np.percentile(flat, [16, 50, 84], axis=0)
        summ = {n: dict(med=float(pct[1, i]), lo=float(pct[1, i] - pct[0, i]),
                        hi=float(pct[2, i] - pct[1, i])) for i, n in enumerate(self.names)}
        json.dump(summ, open(out / "posterior_summary.json", "w"), indent=2)
        return dict(flat=flat, summary=summ,
                    acceptance=float(np.mean(sampler.acceptance_fraction)))
