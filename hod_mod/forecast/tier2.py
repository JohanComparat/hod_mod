r"""Tier-2 forecast assembly: the (z, M*) cell grid + AGN + global lensing blocks.

Unlike :class:`~hod_mod.forecast.tomography.TomographicForecast` (per-bin HOD
copies), the tier-2 design uses ONE shared global parameter vector — redshift
evolution is carried by the explicit ``*_zs`` slope parameters inside each
:class:`~hod_mod.forecast.forward_jax.ForwardModel` (``_theta_eff``) — so the
global vector is simply ``PARAM_NAMES`` (61 entries) and the Jacobian is the
row-stack of independent per-block Jacobians:

* **cell** blocks — volume-limited (z, M*) samples: Δz = 0.1 shells × 0.2-dex
  M* bins, each predicting ``(wp, ds, cl_gX×bands, cl_gy, cl_gkCMB, n_gal)``
  (the per-bin ``n_gal`` IS the stellar-mass-function datum, so ``smf`` is
  not a separate observable);
* **shell** blocks — per-Δz observables that do not depend on the M* split:
  the soft-band AGN XLF and the per-band X-ray auto ``cl_XX``;
* **global** block — tomographic cosmic shear (``cl_kk`` pairs), CMB lensing
  and their cross (``cl_kCMB``, ``cl_shear_kCMB``);
* **wp_agn** blocks — projected clustering of complete soft-L_X-selected AGN
  samples in 0.5-dex bins at a few redshifts.

Each block gets its own ``jax.jacfwd`` (never one monolithic Jacobian) and a
per-block npz cache, so re-runs are incremental.  Noise is physical
(:mod:`hod_mod.forecast.noise`): pair counts + cosmic variance for the
projected statistics, shape noise for lensing, CXB photon noise + the
completeness-pinned Athena spec for X-rays, Poisson counts for the XLF.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import jax
import jax.numpy as jnp

from hod_mod.forecast.forward_jax import ForwardModel, PARAM_NAMES, _IDX
from hod_mod.forecast import params, fisher, noise
from hod_mod.forecast.apec_bands import DEFAULT_BANDS

GAL_OBS = ("wp", "ds", "cl_gX", "cl_gy", "cl_gkCMB", "n_gal")
SHELL_OBS = ("xlf", "cl_XX")
GLOBAL_OBS = ("cl_kk", "cl_kCMB", "cl_shear_kCMB")

# tier-2 default grids: lighter than the tier-1 single-sample defaults because
# ~100 blocks share them (a high-res reference cell checks convergence).
DEFAULT_MODEL_KW = dict(n_k=96, n_m=128, n_gl=48, n_z=5, nz_sig=0.04)
_RP_WP = np.logspace(-1.0, 1.5, 12)
_RP_DS = np.logspace(-1.0, 1.3, 10)
_ELL = np.logspace(1.0, 3.5, 12)

_BAND_PRESETS = {
    1: [(0.5, 2.0)],
    6: DEFAULT_BANDS,
    15: [(0.5 + 0.1 * i, 0.6 + 0.1 * i) for i in range(15)],
}


class _Block:
    def __init__(self, label, kind, model, which, z_lo, z_hi,
                 m_lo=np.nan, m_hi=np.nan):
        self.label, self.kind, self.model, self.which = label, kind, model, tuple(which)
        self.z_lo, self.z_hi, self.m_lo, self.m_hi = z_lo, z_hi, m_lo, m_hi


class Tier2Forecast:
    """Shared-vector multi-block tier-2 forecast (see module docstring).

    Parameters
    ----------
    z_edges, mstar_edges : array
        Cell grid: Δz = 0.1 shells over 0 < z < 1 × 0.2-dex bins over
        10.0 ≤ log10 M* ≤ 11.6 by default (80 cells).
    n_bands : int
        X-ray energy bands over 0.5–2 keV: 1 (broad), 6 (default) or 15
        (the validated production 100 eV grid); any explicit list of
        (emin, emax) pairs is also accepted.
    n_shear_bins : int
        Tomographic shear source bins (equal-number Smail split).
    agn_lx_bins, agn_z_centers
        Soft-band L_X bins (0.5 dex, complete above 1e42) and the redshifts
        of the AGN clustering samples (Δz = 0.2 windows).
    shear, cmbl, athena, spectro : noise.py survey dataclasses
    tsz : (rN, aN, f_sky)
        The calibrated stage-4 effective tSZ noise recipe (kept in v1).
    split_sfq : bool
        Split every (z, M*) cell into star-forming and quiescent samples
        (ZM16 Weibull quenching; missing-physics extension).  Doubles the
        cell blocks; the shell X-ray observables stay unsplit (total gas).
    model_kw : forwarded to every ForwardModel (grids etc.).
    """

    def __init__(self, z_edges=None, mstar_edges=None, n_bands=6,
                 n_shear_bins=5, agn_lx_bins=None,
                 agn_z_centers=(0.1, 0.3, 0.5, 0.7, 0.9),
                 shear=None, cmbl=None, athena=None, spectro=None,
                 tsz=(0.25, 0.9, 0.30), split_sfq=False,
                 include_radio=False, include_hi=False, include_ssfr=False,
                 radio=None, hi=None, **model_kw):
        self.z_edges = np.asarray(z_edges if z_edges is not None
                                  else np.arange(0.0, 1.01, 0.1))
        self.mstar_edges = np.asarray(mstar_edges if mstar_edges is not None
                                      else np.arange(10.0, 11.61, 0.2))
        self.bands = (list(n_bands) if not np.isscalar(n_bands)
                      else _BAND_PRESETS[int(n_bands)])
        self.n_shear_bins = int(n_shear_bins)
        self.agn_lx_bins = list(agn_lx_bins) if agn_lx_bins is not None else \
            [(42.0, 42.5), (42.5, 43.0), (43.0, 43.5), (43.5, 44.0)]
        self.agn_z_centers = tuple(float(z) for z in agn_z_centers)

        self.shear = shear if shear is not None else noise.ShearSurvey()
        self.cmbl = cmbl if cmbl is not None else noise.CMBLensingSurvey()
        self.athena = athena if athena is not None else noise.AthenaAllSky()
        self.spectro = spectro if spectro is not None else noise.SpectroSurvey()
        self.tsz = tuple(tsz)
        # missing-physics wave 2: radio LF, HI (HIMF + 21 cm IM cross), MS sSFR
        self.include_radio = bool(include_radio)
        self.include_hi = bool(include_hi)
        self.include_ssfr = bool(include_ssfr)
        self.radio = radio if radio is not None else noise.RadioSurvey()
        self.hi = hi if hi is not None else noise.HISurvey()

        kw = dict(DEFAULT_MODEL_KW)
        kw.update(model_kw)
        kw.setdefault("rp_wp", _RP_WP)
        kw.setdefault("rp_ds", _RP_DS)
        kw.setdefault("ell", _ELL)
        kw.setdefault("n_z_shear", max(16, 3 * self.n_shear_bins + 1))
        self.model_kw = kw
        self.split_sfq = bool(split_sfq)
        self._spec_repr = repr((sorted(PARAM_NAMES), list(self.z_edges),
                                list(self.mstar_edges), self.bands,
                                self.n_shear_bins, self.agn_lx_bins,
                                self.agn_z_centers, self.split_sfq,
                                self.include_radio, self.include_hi,
                                self.include_ssfr,
                                {k: np.asarray(v).tolist() if hasattr(v, "__len__") else v
                                 for k, v in kw.items()}))

        cell_kw = dict(kw, xray_bands=self.bands, agn_emission="powell",
                       xlf_band="soft", agn_lx_bins=self.agn_lx_bins)
        # SF/quiescent split (missing-physics): two samples per (z, M*) cell
        # sharing the one global vector — SF + Q sum exactly to the unsplit
        # occupations, so the split only ADDs information (and the dlx_quenched
        # hot-gas offset becomes observable through the per-population cl_gX).
        sfq_variants = ("sf", "q") if self.split_sfq else (None,)
        self.blocks = []
        for i, (z1, z2) in enumerate(zip(self.z_edges[:-1], self.z_edges[1:])):
            zc = 0.5 * (z1 + z2)
            shell_model = None
            for (m1, m2) in zip(self.mstar_edges[:-1], self.mstar_edges[1:]):
                for sv in sfq_variants:
                    m = ForwardModel(z_eff=zc, log10m_star_bin=(m1, m2),
                                     sfq=sv, **cell_kw)
                    lab = f"z{zc:.2f}_m{m1:.1f}" + ("" if sv is None else f"_{sv}")
                    # wave-2 per-cell observables: 21 cm × galaxies cross for
                    # every cell; the MS mean sSFR only for non-quenched samples
                    obs = tuple(GAL_OBS)
                    if self.include_hi:
                        obs += ("cl_gHI",)
                    if self.include_ssfr and sv != "q":
                        obs += ("ssfr",)
                    self.blocks.append(_Block(lab, "cell", m, obs,
                                              z1, z2, m1, m2))
                    if shell_model is None and sv is None:
                        shell_model = m
            if shell_model is None:
                # split mode: the shell observables (xlf, TOTAL-gas cl_XX) need
                # an UNSPLIT model — the quenched L_X offset must not leak into
                # the X-ray auto of the whole shell
                shell_model = ForwardModel(
                    z_eff=zc,
                    log10m_star_bin=(self.mstar_edges[0], self.mstar_edges[1]),
                    **cell_kw)
            # shell observables (M*-independent): soft XLF + per-band cl_XX,
            # + the radio LF (fundamental plane, wave 2)
            shell_obs = tuple(SHELL_OBS)
            if self.include_radio:
                shell_obs += ("rlf",)
            self.blocks.append(_Block(f"z{zc:.2f}_shell", "shell", shell_model,
                                      shell_obs, z1, z2))
        if self.include_hi:
            # the blind HIMF is a LOCAL measurement (ALFALFA-like z ≲ 0.06):
            # at Δz = 0.1 shell depths the 21 cm flux limit flags everything
            # below ~10^11 Msun — one dedicated low-z block instead
            m_hi = ForwardModel(z_eff=0.5 * self.hi.z_himf,
                                log10m_star_bin=(self.mstar_edges[0],
                                                 self.mstar_edges[1]),
                                **cell_kw)
            self.blocks.append(_Block("hi_local", "shell", m_hi, ("himf",),
                                      0.0, self.hi.z_himf))
        gm = ForwardModel(z_eff=0.3, n_shear_bins=self.n_shear_bins,
                          z_src_mean=0.9, **kw)
        self.global_model = gm
        self.blocks.append(_Block("global_lensing", "global", gm, GLOBAL_OBS,
                                  0.0, self.z_edges[-1]))
        for zc in self.agn_z_centers:
            m = ForwardModel(z_eff=zc, **cell_kw)
            self.blocks.append(_Block(f"agn_z{zc:.2f}", "wp_agn", m,
                                      ("wp_agn",), zc - 0.1, zc + 0.1))

    # ---- parameters ---------------------------------------------------
    def fiducial(self):
        return params.fiducial_vector()

    def prior(self, add_planck=False, fix=("log10DC",)):
        """Regularizing prior; the retired log10DC is pinned by default
        (agn_emission="powell" removes it from the emissivity entirely)."""
        return params.regularizing_prior(add_planck=add_planck, fix=fix)

    # ---- data vector + Jacobian (block-wise, cached) -------------------
    def _cache_path(self, cache_dir, block, fid):
        key = hashlib.md5((self._spec_repr + block.label + repr(block.which)
                           + np.asarray(fid).tobytes().hex()).encode()
                          ).hexdigest()[:16]
        return os.path.join(cache_dir, f"tier2_{block.label}_{key}.npz")

    def _compute_block(self, block, fid):
        f, row_obs, row_x = block.model.full_data_vector_fn(list(block.which))
        d0, J = fisher.jacobian(f, fid)
        extras = {}
        if block.kind == "cell":
            extras["cl_gg"] = np.asarray(block.model.cl_gg_fiducial(fid))
        if block.kind == "wp_agn":
            th = block.model._theta_eff(jnp.asarray(fid))
            H = block.model._halo_common(th, block.model.z_eff)
            extras["n_agn"] = np.asarray([
                float(jnp.trapezoid(H["dndm"]
                      * block.model._agn_occupation_obs(th, l1, l2),
                      block.model.m))
                for (l1, l2) in self.agn_lx_bins])
        return (np.asarray(d0), np.asarray(J), np.asarray(row_obs),
                np.asarray(row_x, dtype=float), extras)

    def data_and_jacobian(self, fid, cache_dir=None, verbose=True):
        """Assemble (d0, J, meta) block by block; per-block npz caching.

        ``meta`` is a dict of per-row arrays: block, kind, zeff, z_lo, z_hi,
        m_lo, m_hi, obs, x, sub (band / L_X-bin / shear-pair sub-index).
        """
        d0s, Js, meta = [], [], {k: [] for k in
                                 ("block", "kind", "zeff", "z_lo", "z_hi",
                                  "m_lo", "m_hi", "obs", "x", "sub")}
        self._extras = {}
        for ib, b in enumerate(self.blocks):
            fp = self._cache_path(cache_dir, b, fid) if cache_dir else None
            if fp and os.path.exists(fp):
                z = np.load(fp, allow_pickle=False)
                d0, J, row_obs, row_x = z["d0"], z["J"], z["row_obs"], z["row_x"]
                extras = {k[3:]: z[k] for k in z.files if k.startswith("ex_")}
            else:
                d0, J, row_obs, row_x, extras = self._compute_block(b, fid)
                if fp:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.savez_compressed(fp, d0=d0, J=J, row_obs=row_obs,
                                        row_x=row_x,
                                        **{f"ex_{k}": v for k, v in extras.items()})
            if verbose:
                print(f"[tier2] block {ib + 1:3d}/{len(self.blocks)} "
                      f"{b.label:16s} rows={d0.size}", flush=True)
            self._extras[b.label] = extras
            d0s.append(d0); Js.append(J)
            row_obs = np.asarray(row_obs)
            meta["block"] += [b.label] * d0.size
            meta["kind"] += [b.kind] * d0.size
            meta["zeff"] += [b.model.z_eff] * d0.size
            meta["z_lo"] += [b.z_lo] * d0.size
            meta["z_hi"] += [b.z_hi] * d0.size
            meta["m_lo"] += [b.m_lo] * d0.size
            meta["m_hi"] += [b.m_hi] * d0.size
            meta["obs"] += list(row_obs)
            meta["x"] += list(np.asarray(row_x, dtype=float))
            meta["sub"] += list(self._sub_index(b.model, row_obs))
        meta = {k: np.asarray(v) for k, v in meta.items()}
        return np.concatenate(d0s), np.vstack(Js), meta

    @staticmethod
    def _sub_index(model, row_obs):
        """Sub-index within stacked observables: X-ray band, L_X bin, shear pair."""
        sub = np.full(len(row_obs), -1, dtype=int)
        for name, base in (("cl_gX", len(np.asarray(model.ell))),
                           ("cl_XX", len(np.asarray(model.ell))),
                           ("cl_kk", len(np.asarray(model.ell))),
                           ("cl_shear_kCMB", len(np.asarray(model.ell))),
                           ("wp_agn", len(np.asarray(model.rp_wp_agn)))):
            sel = np.where(row_obs == name)[0]
            if sel.size:
                sub[sel] = np.arange(sel.size) // base
        return sub

    # ---- scale cuts ----------------------------------------------------
    def scale_cut_mask(self, meta, rmin):
        """Per-block mask: r_p > rmin (projected), ℓ < χ(z_eff)/rmin (angular)."""
        keep = np.zeros(len(meta["obs"]), dtype=bool)
        for b in self.blocks:
            sel = meta["block"] == b.label
            keep[sel] = b.model.scale_cut_mask(meta["obs"][sel], meta["x"][sel], rmin)
        return keep

    # ---- physical noise -------------------------------------------------
    def noise_sigma(self, fid, d0, meta, verbose=True):
        """Per-row absolute Gaussian σ from the physical survey noise models.

        Completeness: XLF / wp_agn rows whose L_X bin dips below the Athena
        detection limit L_lim(z_hi) get σ = inf (zero weight) and are reported.
        """
        h, Om = float(fid[_IDX["h"]]), float(fid[_IDX["Omega_m"]])
        ath, sh, sp, cm = self.athena, self.shear, self.spectro, self.cmbl
        rn_y, an_y, fsky_y = self.tsz
        f_gx = min(sp.f_sky, ath.f_sky)          # galaxies × Athena overlap
        f_agn = min(sp.f_sky, ath.f_sky)         # AGN need spec-z counterparts
        nkk = sh.noise_cl(self.n_shear_bins)
        sigma = np.full(d0.size, np.inf)
        ell = np.asarray(self.global_model.ell)
        gm = self.global_model
        zs_src, nz_src = np.asarray(gm.z_shear), np.asarray(gm.nz_src)
        flagged = []

        # global-block fiducials needed by several cross-spectra
        gsel = meta["block"] == "global_lensing"
        g_obs, g_sub = meta["obs"][gsel], meta["sub"][gsel]
        g_d0 = d0[gsel]
        cl_kcmb = g_d0[g_obs == "cl_kCMB"]
        kk_auto = {i: g_d0[(g_obs == "cl_kk") & (g_sub == p)]
                   for p, (i, j) in enumerate(gm.shear_pairs) if i == j}

        # per-shell X-ray fiducials + photon noise
        shell_xx, shell_nx = {}, {}
        for b in self.blocks:
            if b.kind != "shell":
                continue
            sel = (meta["block"] == b.label) & (meta["obs"] == "cl_XX")
            nb = len(self.bands)
            shell_xx[(b.z_lo, b.z_hi)] = d0[sel].reshape(nb, -1)
            c1, c2 = noise.chi_of(b.z_lo, h, Om), noise.chi_of(b.z_hi, h, Om)
            shell_nx[(b.z_lo, b.z_hi)] = noise.athena_noise_cl_model(
                ath, b.model.xray_bands, b.model.z_eff, c1, c2, h)

        for b in self.blocks:
            bsel = meta["block"] == b.label
            obs_b = meta["obs"][bsel]
            d_b = d0[bsel]
            sig_b = np.full(d_b.size, np.inf)
            c1, c2 = noise.chi_of(b.z_lo, h, Om), noise.chi_of(b.z_hi, h, Om)
            beam2 = ath.beam(ell) ** 2

            if b.kind == "cell":
                v = noise.shell_volume(b.z_lo, b.z_hi, h, Om, sp.f_sky)
                ngal = float(d_b[obs_b == "n_gal"][0])
                n2d = noise.n2d_of(ngal, c1, c2)
                cl_gg = self._extras[b.label]["cl_gg"]
                xx = shell_xx[(b.z_lo, b.z_hi)]
                nx = shell_nx[(b.z_lo, b.z_hi)]
                for name in b.which:
                    s = obs_b == name
                    if name == "wp":
                        sig_b[s] = noise.wp_pair_sigma(
                            np.asarray(b.model.rp_wp), d_b[s], ngal, v, sp)
                    elif name == "ds":
                        sig_b[s] = noise.delta_sigma_noise(
                            np.asarray(b.model.rp_ds), d_b[s], b.model.z_eff,
                            ngal, v, h, Om, zs_src, nz_src, sh, sp)
                    elif name == "n_gal":
                        sig_b[s] = ngal * np.sqrt(1.0 / (ngal * v)
                                                  + sp.cv_rel(v) ** 2)
                    elif name == "cl_gy":
                        n_y = rn_y * (ell / 100.0) ** an_y * d_b[s]
                        sig_b[s] = np.sqrt(2.0 / noise.n_modes(ell, fsky_y)) \
                            * (d_b[s] + n_y)
                    elif name == "cl_gHI":
                        # 21 cm IM × galaxies: calibrated effective recipe
                        n_hi = self.hi.rn_im * (ell / 100.0) ** self.hi.an_im \
                            * d_b[s]
                        sig_b[s] = np.sqrt(
                            2.0 / noise.n_modes(ell, self.hi.f_sky_im)) \
                            * (d_b[s] + n_hi)
                    elif name == "ssfr":
                        sig_b[s] = sp.ssfr_err
                    elif name == "cl_gkCMB":
                        sig_b[s] = noise.knox_cross(
                            ell, d_b[s], cl_gg, 1.0 / n2d, cl_kcmb, cm.n0,
                            min(sp.f_sky, cm.f_sky))
                    elif name == "cl_gX":
                        cgx = d_b[s].reshape(len(self.bands), -1)
                        sig_b[s] = np.concatenate([
                            noise.knox_cross(ell, cgx[k], cl_gg, 1.0 / n2d,
                                             xx[k], nx[k] / beam2, f_gx)
                            for k in range(len(self.bands))])

            elif b.kind == "shell":
                v_agn = noise.shell_volume(b.z_lo, b.z_hi, h, Om, f_agn)
                l_lim = ath.l_lim(b.z_hi, h, Om)
                nx = shell_nx[(b.z_lo, b.z_hi)]
                for name in b.which:
                    s = obs_b == name
                    if name == "xlf":
                        rel = noise.xlf_relerr(d_b[s], v_agn, dloglx=0.5)
                        lx_lo = 10.0 ** (meta["x"][bsel][s] - 0.25)
                        bad = lx_lo < l_lim
                        rel = np.where(bad, np.inf, rel)
                        for xv in meta["x"][bsel][s][bad]:
                            flagged.append((b.label, "xlf", float(xv)))
                        sig_b[s] = rel * d_b[s]
                    elif name == "cl_XX":
                        cxx = d_b[s].reshape(len(self.bands), -1)
                        sig_b[s] = np.concatenate([
                            noise.knox_auto(ell, cxx[k], nx[k] / beam2, ath.f_sky)
                            for k in range(len(self.bands))])
                    elif name == "rlf":
                        # radio LF: Poisson counts over the radio×z-counterpart
                        # footprint, with the νLν(5 GHz) completeness limit
                        grid = np.asarray(meta["x"][bsel][s])
                        dlog = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
                        v_r = noise.shell_volume(b.z_lo, b.z_hi, h, Om,
                                                 self.radio.f_sky)
                        rel = noise.xlf_relerr(d_b[s], v_r, dloglx=dlog)
                        lr_lo = 10.0 ** (grid - 0.5 * dlog)
                        bad = lr_lo < self.radio.l_lim(b.z_hi, h, Om)
                        rel = np.where(bad, np.inf, rel)
                        for xv in grid[bad]:
                            flagged.append((b.label, "rlf", float(xv)))
                        sig_b[s] = rel * d_b[s]
                    elif name == "himf":
                        # blind HIMF: Poisson with the 21 cm M_HI flux limit
                        grid = np.asarray(meta["x"][bsel][s])
                        dlog = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
                        v_hi = noise.shell_volume(b.z_lo, b.z_hi, h, Om,
                                                  self.hi.f_sky)
                        rel = noise.xlf_relerr(d_b[s], v_hi, dloglx=dlog)
                        mhi_lo = 10.0 ** (grid - 0.5 * dlog)
                        bad = mhi_lo < self.hi.mhi_lim(b.z_hi, h, Om)
                        rel = np.where(bad, np.inf, rel)
                        for xv in grid[bad]:
                            flagged.append((b.label, "himf", float(xv)))
                        sig_b[s] = rel * d_b[s]

            elif b.kind == "global":
                for name in b.which:
                    s = obs_b == name
                    if name == "cl_kk":
                        parts = []
                        c_p = d_b[s].reshape(len(gm.shear_pairs), -1)
                        for p, (i, j) in enumerate(gm.shear_pairs):
                            if i == j:
                                parts.append(noise.knox_auto(
                                    ell, c_p[p], nkk, sh.f_sky))
                            else:
                                parts.append(noise.knox_cross(
                                    ell, c_p[p], kk_auto[i], nkk,
                                    kk_auto[j], nkk, sh.f_sky))
                        sig_b[s] = np.concatenate(parts)
                    elif name == "cl_kCMB":
                        sig_b[s] = noise.knox_auto(ell, d_b[s], cm.n0, cm.f_sky)
                    elif name == "cl_shear_kCMB":
                        c_i = d_b[s].reshape(self.n_shear_bins, -1)
                        sig_b[s] = np.concatenate([
                            noise.knox_cross(ell, c_i[i], kk_auto[i], nkk,
                                             cl_kcmb, cm.n0,
                                             min(sh.f_sky, cm.f_sky))
                            for i in range(self.n_shear_bins)])

            elif b.kind == "wp_agn":
                v = noise.shell_volume(b.z_lo, b.z_hi, h, Om, f_agn)
                l_lim = ath.l_lim(b.z_hi, h, Om)
                n_agn = self._extras[b.label]["n_agn"]
                rp = np.asarray(b.model.rp_wp_agn)
                w = d_b.reshape(len(self.agn_lx_bins), -1)
                parts = []
                for k, (l1, l2) in enumerate(self.agn_lx_bins):
                    if 10.0 ** l1 < l_lim:
                        flagged.append((b.label, "wp_agn", float(l1)))
                        parts.append(np.full(rp.size, np.inf))
                    else:
                        parts.append(noise.wp_pair_sigma(
                            rp, w[k], float(n_agn[k]), v, sp))
                sig_b[:] = np.concatenate(parts)

            sigma[bsel] = sig_b

        if verbose and flagged:
            print(f"[tier2] completeness: {len(flagged)} rows below "
                  f"L_lim(z_hi) got sigma=inf:")
            for lab, o, xv in flagged:
                print(f"        {lab} {o} log10Lx={xv:.2f}")
        self.completeness_flags = flagged
        return sigma
