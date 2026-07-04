r"""Tier-4 forecast assembly: the morphology observables of the literature.

Extends :class:`~hod_mod.forecast.tier3.Tier3Forecast` with the measurements
that pin the wave-4 morphology sector (see ``docs/tier4_forecast.rst`` for the
verified literature basis):

* **f_early_q** per cell — the joint early∩quenched fraction (Galaxy Zoo red
  spirals / blue ellipticals census) measuring the ``rho_morph_q``
  morphology–quenching correlation;
* **size** per cell — the mean ⟨log10 R_e⟩ of the centrals through the
  Kravtsov R_e ≈ 0.015 R_200c relation (+ the early-type offset), weighing
  cosmology through R_200c ∝ (M/ρ_crit)^{1/3};
* **wgp** per cell — the NLA galaxy–intrinsic-alignment cross with the
  amplitude carried by the early-type fraction (KiDS/DESI: IA is driven by
  morphology) — the shear IA systematic becomes self-calibrated;
* **f_early_agn** per shell — the bulge-dominance of X-ray AGN hosts, the
  direct probe of the ``mbh_bt_slope`` BH–bulge coupling;
* **morph_cell blocks** — early/late-split w_p, ΔΣ and n̄_g per (z, M*) cell
  to z ≤ 1.2 (the Mandelbaum-2006 morphology–halo-mass measurement at Euclid
  scale; wide spectroscopic tier only — morphology needs imaging depth).
"""

from __future__ import annotations

import numpy as np

from hod_mod.forecast.forward_jax import ForwardModel, _IDX
from hod_mod.forecast import noise
from hod_mod.forecast.tier2 import _Block
from hod_mod.forecast.tier3 import Tier3Forecast

MORPH_OBS = ("f_early_q", "size", "wgp", "f_early_agn")

_SIG_SIZE = 0.2      # lognormal scatter of R_e at fixed R_200c [dex] (Kravtsov13)


class Tier4Forecast(Tier3Forecast):
    """Tier-4 morphology forecast (see module docstring).

    Beyond :class:`Tier3Forecast` (all of whose flags default ON here):

    Parameters
    ----------
    include_morphq, include_sizes, include_ia, include_agn_morph : bool
        The per-cell f_early_q / size / w_g+ data and the per-shell AGN-host
        early fraction.
    include_morph_split : bool
        Early/late-split (wp, ds, n_gal) blocks per cell up to
        ``z_morph_max`` (wide-tier cells only).
    z_morph_max : float
        Morphological-classification depth of the imaging survey (Euclid
        VIS-like) — also where the shear sources run out for the split ΔΣ.
    """

    def __init__(self, include_morphq=True, include_sizes=True,
                 include_ia=True, include_agn_morph=True,
                 include_morph_split=True, z_morph_max=1.2, **kw):
        self.include_morphq = bool(include_morphq)
        self.include_sizes = bool(include_sizes)
        self.include_ia = bool(include_ia)
        self.include_agn_morph = bool(include_agn_morph)
        self.include_morph_split = bool(include_morph_split)
        self.z_morph_max = float(z_morph_max)
        kw.setdefault("include_morph", True)
        super().__init__(**kw)
        self._spec_repr += repr((
            "tier4", self.include_morphq, self.include_sizes,
            self.include_ia, self.include_agn_morph,
            self.include_morph_split, self.z_morph_max))
        self._ctor_kwargs.update(
            include_morphq=self.include_morphq,
            include_sizes=self.include_sizes, include_ia=self.include_ia,
            include_agn_morph=self.include_agn_morph,
            include_morph_split=self.include_morph_split,
            z_morph_max=self.z_morph_max)

    # ---- block-building hooks ------------------------------------------
    def _cell_extra_obs(self, sv):
        obs = super()._cell_extra_obs(sv)
        if self.include_sizes:
            obs += ("size",)
        if self.include_ia:
            obs += ("wgp",)
        # the joint fraction is a BASE-sample datum — one copy per (z, M*)
        # cell (attached to the SF variant; both variants would duplicate it)
        if self.include_morphq and sv != "q":
            obs += ("f_early_q",)
        return obs

    def _shell_extra_obs(self):
        obs = super()._shell_extra_obs()
        if self.include_agn_morph:
            obs += ("f_early_agn",)
        return obs

    def _extra_blocks(self):
        blocks = super()._extra_blocks()
        if not self.include_morph_split:
            return blocks
        # early/late-split clustering + lensing: light blocks (wp/ds/n_gal
        # only), wide-tier cells to z_morph_max — the morphology–halo-mass
        # measurement (Mandelbaum-2006-style at Euclid scale)
        for z1, z2 in zip(self.z_edges[:-1], self.z_edges[1:]):
            if z2 > self.z_morph_max + 1e-9:
                continue
            zc = 0.5 * (z1 + z2)
            for m1, m2 in zip(self.mstar_edges[:-1], self.mstar_edges[1:]):
                if not self.spectro.complete_for(m1, z2):
                    continue                    # imaging morphology: wide tier
                for mo in ("early", "late"):
                    m = ForwardModel(z_eff=zc, **dict(
                        self._base_cell_kw, log10m_star_bin=(m1, m2),
                        morph=mo, log10m_min=self.cell_log10m_min,
                        n_m=self.cell_n_m))
                    blk = _Block(f"z{zc:.2f}_m{m1:.1f}_{mo}", "morph_cell",
                                 m, ("wp", "ds", "n_gal"), z1, z2, m1, m2)
                    blk.spectro = self.spectro
                    blocks.append(blk)
        return blocks

    # ---- physical noise --------------------------------------------------
    def noise_sigma(self, fid, d0, meta, verbose=True):
        sigma = super().noise_sigma(fid, d0, meta, verbose=False)
        h, Om = float(fid[_IDX["h"]]), float(fid[_IDX["Omega_m"]])
        sh, sp, ath = self.shear, self.spectro, self.athena
        gm = self.global_model
        zs_src, nz_src = np.asarray(gm.z_shear), np.asarray(gm.nz_src)
        flagged = list(self.completeness_flags)

        for b in self.blocks:
            bsel = meta["block"] == b.label
            obs_b = meta["obs"][bsel]
            d_b = d0[bsel]
            sig_b = sigma[bsel]

            if b.kind == "cell":
                sp_b = self._cell_spectro(b)
                v = noise.shell_volume(b.z_lo, b.z_hi, h, Om, sp_b.f_sky)
                ngal = float(d_b[obs_b == "n_gal"][0])
                s = obs_b == "f_early_q"
                if s.any():
                    f = np.clip(d_b[s], 1e-4, 1.0 - 1e-4)
                    sig_b[s] = np.sqrt(f * (1.0 - f) / (ngal * v)
                                       + sp_b.fmorph_err ** 2)
                s = obs_b == "size"
                if s.any():
                    sig_b[s] = np.sqrt(_SIG_SIZE ** 2 / (ngal * v)
                                       + sp_b.size_err ** 2)
                s = obs_b == "wgp"
                if s.any():
                    sig_b[s] = noise.wgp_noise(
                        np.asarray(b.model.rp_ds), d_b[s], b.model.z_eff,
                        ngal, v, h, Om, sh, sp_b)

            elif b.kind == "shell" and "f_early_agn" in b.which:
                s = obs_b == "f_early_agn"
                v_agn = noise.shell_volume(b.z_lo, b.z_hi, h, Om,
                                           min(sp.f_sky, ath.f_sky))
                n3d = float(np.sum(self._extras[b.label].get(
                    "n_agn", np.array([np.nan]))))
                if np.isfinite(n3d) and n3d > 0:
                    f = np.clip(d_b[s], 1e-4, 1.0 - 1e-4)
                    sig_b[s] = np.sqrt(f * (1.0 - f) / (n3d * v_agn)
                                       + sp.fmorph_agn_err ** 2)
                else:
                    flagged.append((b.label, "f_early_agn", np.nan))

            elif b.kind == "morph_cell":
                sp_b = self._cell_spectro(b)
                v = noise.shell_volume(b.z_lo, b.z_hi, h, Om, sp_b.f_sky)
                ngal = float(d_b[obs_b == "n_gal"][0])
                s = obs_b == "wp"
                sig_b[s] = noise.wp_pair_sigma(
                    np.asarray(b.model.rp_wp), d_b[s], ngal, v, sp_b)
                s = obs_b == "ds"
                sig_b[s] = noise.delta_sigma_noise(
                    np.asarray(b.model.rp_ds), d_b[s], b.model.z_eff,
                    ngal, v, h, Om, zs_src, nz_src, sh, sp_b)
                s = obs_b == "n_gal"
                sig_b[s] = ngal * np.sqrt(1.0 / (ngal * v)
                                          + sp_b.cv_rel(v) ** 2)

            sigma[bsel] = sig_b

        if verbose and flagged:
            print(f"[tier4] completeness: {len(flagged)} rows flagged")
        self.completeness_flags = flagged
        return sigma
