r"""Tier-3 forecast assembly: multi-wavelength maps, band LFs, z < 2 × M* > 10⁹.

Extends :class:`~hod_mod.forecast.tier2.Tier2Forecast` (whose block/cache/noise
machinery is reused unchanged through its extension hooks) with

* a **coarse exploratory grid**: Δz = 0.2 shells over 0 < z < 2 × 0.2-dex M*
  bins over 9.0 ≤ log10 M* ≤ 11.6 (130 cells, ×2 with the SF/Q split), each
  cell on the extended mass grid (``log10m_min = 8.5``);
* **radio and IR intensity maps** (SKA-like GHz bands; WISE/SPHEREx-like μm
  bands): per-cell galaxy crosses ``cl_gR``/``cl_gI``, per-shell autos
  ``cl_RR``/``cl_II`` and AGN crosses ``cl_aR``/``cl_aI``/``cl_ag``;
* **galaxy band LFs** (``uvlf``, ``optlf``, ``nirlf``, the ``half`` Hα LF) and
  **AGN UV/optical LFs** (``qlf_uv``, ``qlf_opt``) per shell;
* a per-shell wide-M* **SFRD(z)** block (the Madau–Dickinson measurement);
* the four **extras**: tSZ auto ``cl_yy``, 21 cm auto ``cl_HIHI``, X-ray
  cluster counts ``ncl`` (per shell) and AGN lensing ``ds_agn`` (per AGN
  z-block).

**Two-tier galaxy completeness**: the wide spectroscopic survey carries a
stellar-mass limit log10 M*_lim(z) = ``mstar_lim0`` + ``mstar_lim_slope``·z;
cells below it fall back to a small deep field (``spectro_deep``), and cells
complete in neither tier are not built (recorded in ``skipped_cells``).
"""

from __future__ import annotations

import numpy as np

from hod_mod.forecast.forward_jax import ForwardModel, _IDX
from hod_mod.forecast import params, noise
from hod_mod.forecast.tier2 import Tier2Forecast, _Block

MAP_OBS = ("cl_gR", "cl_gI", "cl_RR", "cl_II", "cl_aR", "cl_aI", "cl_ag")
BANDLF_OBS = ("uvlf", "optlf", "nirlf", "half", "qlf_uv", "qlf_opt")
EXTRA_OBS = ("cl_yy", "cl_HIHI", "ncl", "ds_agn")

# clusters are selected above max(L_lim(z), 1e42) — below 1e42 the counts
# blur into the group regime the cross-spectra already constrain
_LOGL_CL_MIN = 42.0


class Tier3Forecast(Tier2Forecast):
    """Tier-3 multi-wavelength forecast (see module docstring).

    Beyond :class:`Tier2Forecast` (whose wave flags default ON here):

    Parameters
    ----------
    spectro, spectro_deep : noise.SpectroSurvey
        Wide tier (f_sky = 0.5, M*-complete to 10^{9+z}) and deep tier
        (f_sky = 0.004, flat at 10^9).
    ska, irmap : noise.SKASurvey, noise.IRMapSurvey
        Radio/IR intensity-map surveys (bands + effective noise recipes).
    lf_uv, lf_opt, lf_nir, lf_quv, lf_qopt : noise.BandLFSurvey
        Band-LF footprints and νL_ν flux limits (GALEX/Rubin/WISE-like and
        quasar-survey-like defaults).
    include_maps, include_bandlfs, include_extras : bool
        Toggle the three tier-3 observable families.
    cell_log10m_min, cell_n_m : float, int
        Mass grid of the cell/sfrd models (the 8.5 floor resolves M* = 10^9
        occupations; n_m = 256 converges to <1e-3).
    """

    def __init__(self, z_edges=None, mstar_edges=None,
                 agn_z_centers=(0.1, 0.3, 0.5, 0.7, 0.9,
                                1.1, 1.3, 1.5, 1.7, 1.9),
                 spectro=None, spectro_deep=None, ska=None, irmap=None,
                 lf_uv=None, lf_opt=None, lf_nir=None, lf_quv=None,
                 lf_qopt=None, include_maps=True, include_bandlfs=True,
                 include_extras=True, cell_log10m_min=8.5, cell_n_m=256,
                 **kw):
        # tier-3 attributes must exist BEFORE super().__init__ runs the
        # block-building loop (the hooks below consume them)
        self.include_maps = bool(include_maps)
        self.include_bandlfs = bool(include_bandlfs)
        self.include_extras = bool(include_extras)
        self.cell_log10m_min = float(cell_log10m_min)
        self.cell_n_m = int(cell_n_m)
        self.ska = ska if ska is not None else noise.SKASurvey()
        self.irmap = irmap if irmap is not None else noise.IRMapSurvey()
        self.lf_uv = lf_uv if lf_uv is not None else \
            noise.BandLFSurvey(f_sky=0.35, nulnu_lim=3.0e-16)
        self.lf_opt = lf_opt if lf_opt is not None else \
            noise.BandLFSurvey(f_sky=0.5, nulnu_lim=1.0e-16)
        self.lf_nir = lf_nir if lf_nir is not None else \
            noise.BandLFSurvey(f_sky=0.65, nulnu_lim=1.0e-15)
        self.lf_quv = lf_quv if lf_quv is not None else \
            noise.BandLFSurvey(f_sky=0.5, nulnu_lim=1.0e-14)
        self.lf_qopt = lf_qopt if lf_qopt is not None else \
            noise.BandLFSurvey(f_sky=0.5, nulnu_lim=1.0e-14)
        self.spectro_deep = spectro_deep if spectro_deep is not None else \
            noise.SpectroSurvey(f_sky=0.004, mstar_lim0=9.0,
                                mstar_lim_slope=0.0)
        if spectro is None:
            spectro = noise.SpectroSurvey(f_sky=0.5, mstar_lim0=9.0,
                                          mstar_lim_slope=1.0)
        # survey-limit constants for the cluster selection (fiducial cosmology)
        self._fid_h = float(params._FIDUCIAL_DEFAULT["h"])
        self._fid_om = float(params._FIDUCIAL_DEFAULT["Omega_m"])

        kw.setdefault("split_sfq", True)
        kw.setdefault("include_radio", True)
        kw.setdefault("include_hi", True)
        kw.setdefault("include_ssfr", True)
        kw.setdefault("include_ir", True)
        super().__init__(
            z_edges=(z_edges if z_edges is not None
                     else np.arange(0.0, 2.01, 0.2)),
            mstar_edges=(mstar_edges if mstar_edges is not None
                         else np.arange(9.0, 11.61, 0.2)),
            agn_z_centers=agn_z_centers, spectro=spectro, **kw)
        self._spec_repr += repr((
            "tier3", self.include_maps, self.include_bandlfs,
            self.include_extras, self.cell_log10m_min, self.cell_n_m,
            self.ska, self.irmap, self.lf_uv, self.lf_opt, self.lf_nir,
            self.lf_quv, self.lf_qopt, self.spectro, self.spectro_deep))
        # --jobs workers rebuild a Tier3Forecast, not the tier-2 base
        self._ctor_kwargs.update(
            spectro_deep=self.spectro_deep, ska=self.ska, irmap=self.irmap,
            lf_uv=self.lf_uv, lf_opt=self.lf_opt, lf_nir=self.lf_nir,
            lf_quv=self.lf_quv, lf_qopt=self.lf_qopt,
            include_maps=self.include_maps,
            include_bandlfs=self.include_bandlfs,
            include_extras=self.include_extras,
            cell_log10m_min=self.cell_log10m_min, cell_n_m=self.cell_n_m)

    # ---- block-building hooks ------------------------------------------
    def _keep_cell(self, z1, z2, m1, m2):
        kw = dict(log10m_min=self.cell_log10m_min, n_m=self.cell_n_m)
        if self.include_maps:
            kw.update(radio_map_bands=self.ska.bands,
                      ir_map_bands=self.irmap.bands)
        if self.spectro.complete_for(m1, z2) or \
                self.spectro_deep.complete_for(m1, z2):
            return True, kw
        return False, {}

    def _decorate_cell(self, block):
        block.spectro = (self.spectro
                         if self.spectro.complete_for(block.m_lo, block.z_hi)
                         else self.spectro_deep)

    def _cell_spectro(self, block):
        return getattr(block, "spectro", None) or self.spectro

    def _cell_extra_obs(self, sv):
        return ("cl_gR", "cl_gI") if self.include_maps else ()

    def _shell_mstar_bin(self):
        # wide galaxy sample: the AGN × galaxy cross needs a physical sample
        # (xlf / cl_XX are galaxy-independent, so this is free elsewhere)
        return (self.mstar_edges[0], self.mstar_edges[-1])

    def _shell_extra_kw(self, zc, z1, z2):
        kw = dict(log10m_min=self.cell_log10m_min, n_m=self.cell_n_m)
        if self.include_maps:
            kw.update(radio_map_bands=self.ska.bands,
                      ir_map_bands=self.irmap.bands)
        if self.include_extras:
            l_lim = self.athena.l_lim(z2, self._fid_h, self._fid_om)
            kw["logl_ncl"] = max(float(np.log10(l_lim)), _LOGL_CL_MIN)
        return kw

    def _shell_extra_obs(self):
        obs = ()
        if self.include_maps:
            obs += ("cl_RR", "cl_II", "cl_aR", "cl_aI", "cl_ag")
        if self.include_bandlfs:
            obs += BANDLF_OBS
        if self.include_extras:
            obs += ("cl_yy", "cl_HIHI", "ncl")
        return obs

    def _agn_extra_obs(self):
        return ("ds_agn",) if self.include_extras else ()

    def _extra_blocks(self):
        # per-shell wide-M* SFRD blocks: the Madau–Dickinson ρ_SFR(z)
        # measurement integrates far below the cell grid's completeness
        blocks = []
        for z1, z2 in zip(self.z_edges[:-1], self.z_edges[1:]):
            zc = 0.5 * (z1 + z2)
            m = ForwardModel(z_eff=zc, **dict(
                self._base_cell_kw, log10m_star_bin=(9.0, 12.0),
                log10m_min=self.cell_log10m_min, n_m=self.cell_n_m))
            blocks.append(_Block(f"sfrd_z{zc:.2f}", "sfrd_wide", m,
                                 ("sfrd",), z1, z2))
        return blocks

    def _block_extras(self, block, fid, extras):
        if block.kind == "shell" and self.include_maps \
                and "cl_ag" in block.which:
            # galaxy + AGN auto fiducials for the AGN-cross Knox noise
            extras["cl_gg"] = np.asarray(block.model.cl_gg_fiducial(fid))
            extras["n_gal_shell"] = np.asarray(
                [float(block.model.predict(np.asarray(fid),
                                           ["n_gal"])["n_gal"][0])])
            cls, nags = [], []
            for (l1, l2) in self.agn_lx_bins:
                cl, na = block.model.cl_aa_fiducial(fid, l1, l2)
                cls.append(np.asarray(cl))
                nags.append(na)
            extras["cl_aa"] = np.stack(cls)
            extras["n_agn"] = np.asarray(nags)

    # ---- physical noise --------------------------------------------------
    def noise_sigma(self, fid, d0, meta, verbose=True):
        """Tier-2 noise for the inherited rows, then the tier-3 families."""
        sigma = super().noise_sigma(fid, d0, meta, verbose=False)
        h, Om = float(fid[_IDX["h"]]), float(fid[_IDX["Omega_m"]])
        ath, sh, sp = self.athena, self.shear, self.spectro
        rn_y, an_y, fsky_y = self.tsz
        f_agn = min(sp.f_sky, ath.f_sky)
        ell = np.asarray(self.global_model.ell)
        gm = self.global_model
        zs_src, nz_src = np.asarray(gm.z_shear), np.asarray(gm.nz_src)
        flagged = list(self.completeness_flags)

        # per-shell fiducial map autos (the cross-noise auto legs)
        shell_rr, shell_ii = {}, {}
        if self.include_maps:
            for b in self.blocks:
                if b.kind != "shell" or "cl_RR" not in b.which:
                    continue
                key = (b.z_lo, b.z_hi)
                s = (meta["block"] == b.label) & (meta["obs"] == "cl_RR")
                shell_rr[key] = d0[s].reshape(len(self.ska.bands), -1)
                s = (meta["block"] == b.label) & (meta["obs"] == "cl_II")
                shell_ii[key] = d0[s].reshape(len(self.irmap.bands), -1)

        for b in self.blocks:
            bsel = meta["block"] == b.label
            obs_b = meta["obs"][bsel]
            d_b = d0[bsel]
            sig_b = sigma[bsel]
            c1, c2 = noise.chi_of(b.z_lo, h, Om), noise.chi_of(b.z_hi, h, Om)

            if b.kind == "cell" and self.include_maps:
                sp_b = self._cell_spectro(b)
                ngal = float(d_b[obs_b == "n_gal"][0])
                n2d = noise.n2d_of(ngal, c1, c2)
                cl_gg = self._extras[b.label]["cl_gg"]
                for name, sur, autos in (("cl_gR", self.ska,
                                          shell_rr[(b.z_lo, b.z_hi)]),
                                         ("cl_gI", self.irmap,
                                          shell_ii[(b.z_lo, b.z_hi)])):
                    s = obs_b == name
                    if not s.any():
                        continue
                    c = d_b[s].reshape(len(sur.bands), -1)
                    sig_b[s] = np.concatenate([
                        noise.knox_cross(ell, c[k], cl_gg, 1.0 / n2d,
                                         autos[k],
                                         sur.noise_cl(ell, autos[k], k),
                                         min(sp_b.f_sky, sur.f_sky))
                        for k in range(len(sur.bands))])

            elif b.kind == "shell":
                ex = self._extras[b.label]
                if self.include_maps and "cl_RR" in b.which:
                    for name, sur, autos in (("cl_RR", self.ska,
                                              shell_rr[(b.z_lo, b.z_hi)]),
                                             ("cl_II", self.irmap,
                                              shell_ii[(b.z_lo, b.z_hi)])):
                        s = obs_b == name
                        c = d_b[s].reshape(len(sur.bands), -1)
                        sig_b[s] = np.concatenate([
                            noise.knox_auto(ell, c[k],
                                            sur.noise_cl(ell, c[k], k),
                                            sur.f_sky)
                            for k in range(len(sur.bands))])
                    n2d_g = noise.n2d_of(float(ex["n_gal_shell"][0]), c1, c2)
                    n2d_a = noise.n2d_of(ex["n_agn"], c1, c2)   # (Nlx,)
                    cl_aa = ex["cl_aa"]                          # (Nlx, Nell)
                    for name, sur, autos in (("cl_aR", self.ska,
                                              shell_rr[(b.z_lo, b.z_hi)]),
                                             ("cl_aI", self.irmap,
                                              shell_ii[(b.z_lo, b.z_hi)])):
                        s = obs_b == name
                        nb = len(sur.bands)
                        c = d_b[s].reshape(len(self.agn_lx_bins), nb, -1)
                        sig_b[s] = np.concatenate([
                            noise.knox_cross(ell, c[i, k],
                                             cl_aa[i], 1.0 / n2d_a[i],
                                             autos[k],
                                             sur.noise_cl(ell, autos[k], k),
                                             min(f_agn, sur.f_sky))
                            for i in range(len(self.agn_lx_bins))
                            for k in range(nb)])
                    s = obs_b == "cl_ag"
                    c = d_b[s].reshape(len(self.agn_lx_bins), -1)
                    sig_b[s] = np.concatenate([
                        noise.knox_cross(ell, c[i], cl_aa[i], 1.0 / n2d_a[i],
                                         ex["cl_gg"], 1.0 / n2d_g, f_agn)
                        for i in range(len(self.agn_lx_bins))])
                if self.include_extras and "cl_yy" in b.which:
                    s = obs_b == "cl_yy"
                    n_y = rn_y * (ell / 100.0) ** an_y * d_b[s]
                    sig_b[s] = noise.knox_auto(ell, d_b[s], n_y, fsky_y)
                    s = obs_b == "cl_HIHI"
                    n_hi = self.hi.rn_im * (ell / 100.0) ** self.hi.an_im \
                        * d_b[s]
                    sig_b[s] = noise.knox_auto(ell, d_b[s], n_hi,
                                               self.hi.f_sky_im)
                    s = obs_b == "ncl"
                    v_cl = noise.shell_volume(b.z_lo, b.z_hi, h, Om,
                                              ath.f_sky)
                    sig_b[s] = d_b[s] * noise.poisson_relerr(d_b[s], v_cl)
                if self.include_bandlfs and "uvlf" in b.which:
                    for name, f_sky_lf, lim in (
                            ("uvlf", self.lf_uv.f_sky,
                             self.lf_uv.l_lim(b.z_hi, h, Om)),
                            ("optlf", self.lf_opt.f_sky,
                             self.lf_opt.l_lim(b.z_hi, h, Om)),
                            ("nirlf", self.lf_nir.f_sky,
                             self.lf_nir.l_lim(b.z_hi, h, Om)),
                            ("half", sp.f_sky,
                             sp.lha_lim(b.z_hi, h, Om)),
                            ("qlf_uv", self.lf_quv.f_sky,
                             self.lf_quv.l_lim(b.z_hi, h, Om)),
                            ("qlf_opt", self.lf_qopt.f_sky,
                             self.lf_qopt.l_lim(b.z_hi, h, Om))):
                        s = obs_b == name
                        grid = np.asarray(meta["x"][bsel][s])
                        dlog = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
                        v_lf = noise.shell_volume(b.z_lo, b.z_hi, h, Om,
                                                  f_sky_lf)
                        rel = noise.xlf_relerr(d_b[s], v_lf, dloglx=dlog)
                        bad = 10.0 ** (grid - 0.5 * dlog) < lim
                        rel = np.where(bad, np.inf, rel)
                        for xv in grid[bad]:
                            flagged.append((b.label, name, float(xv)))
                        sig_b[s] = rel * d_b[s]

            elif b.kind == "sfrd_wide":
                s = obs_b == "sfrd"
                sig_b[s] = sp.sfrd_rel * np.abs(d_b[s])

            elif b.kind == "wp_agn" and self.include_extras:
                s = obs_b == "ds_agn"
                if s.any():
                    v = noise.shell_volume(b.z_lo, b.z_hi, h, Om, f_agn)
                    l_lim = ath.l_lim(b.z_hi, h, Om)
                    n_agn = self._extras[b.label]["n_agn"]
                    rp = np.asarray(b.model.rp_ds)
                    dsv = d_b[s].reshape(len(self.agn_lx_bins), -1)
                    parts = []
                    for k, (l1, l2) in enumerate(self.agn_lx_bins):
                        if 10.0 ** l1 < l_lim:
                            flagged.append((b.label, "ds_agn", float(l1)))
                            parts.append(np.full(rp.size, np.inf))
                        else:
                            parts.append(noise.delta_sigma_noise(
                                rp, dsv[k], b.model.z_eff, float(n_agn[k]),
                                v, h, Om, zs_src, nz_src, sh, sp))
                    sig_b[s] = np.concatenate(parts)

            sigma[bsel] = sig_b

        if verbose and flagged:
            print(f"[tier3] completeness: {len(flagged)} rows below "
                  f"the survey limits got sigma=inf:")
            for lab, o, xv in flagged:
                print(f"        {lab} {o} x={xv:.2f}")
        self.completeness_flags = flagged
        return sigma
