"""Powell et al. 2022 (ApJ 938, 77) AGN–halo model — analytic implementation.

Unlike the repo's other AGN models (``ham``/``hod``/``xray``/``duty_cycle``), which
CONSUME an X-ray luminosity function, this model **predicts** the XLF and the AGN
clustering by forward-modelling supermassive black holes:

    M_halo --SHMR--> M_*  --M_BH-M_*--> M_BH  --ERDF--> lambda_Edd  --> L_bol --> L_X

with (optionally) an M_BH–M_halo correlation at fixed M_* ("Model 2").  Every galaxy
hosts one SMBH that accretes at a rate drawn from a UNIVERSAL Eddington-ratio
distribution (ERDF / LSAR; Ananna et al. 2022) — there is no duty-cycle parameter.

Because log M_*|M_halo and log M_BH|log M_* are Gaussian, log M_BH|M_halo is a single
Gaussian; then log L_X|M_halo = const + log M_BH + log lambda, so P(L_X|M_halo) is
that Gaussian convolved with the ERDF (one convolution per halo mass, vectorised).

Free parameters (Powell/Ananna centres):
    M_BH-M_*: norm=7.76, slope=0.67, scatter=0.33 dex   (Powell Eq. 6, Model 1)
    ERDF   : log10 lambda*=log10(0.13), delta1=0.30, delta2=3.70  (Ananna 2022)
    rho    : M_BH-M_halo correlation (0 = Model 1, >0 = Model 2)

The class exposes the same interface as ``HODAgnModel`` (``nc_ns_agn``,
``agn_emissivity_uk``, ``mean_agn_lx``) so it plugs into ``HaloModelCrossSpectra``
unchanged, plus ``xlf`` / ``agn_bias`` / ``wp`` for standalone LF + clustering.

References
----------
Powell et al. 2022, ApJ 938, 77 (arXiv:2210....);  Ananna et al. 2022, ApJS 261, 9;
Aird et al. 2015 (hard XLF, for validation) — reused from :mod:`hod_mod.agn.ham`.
"""

from __future__ import annotations

import numpy as np

from hod_mod.connection.sham import smhm_girelli20
from hod_mod.agn.ham import _HARD_TO_SOFT_RATIO

# L_bol = 1.26e38 (M_BH/Msun) lambda_Edd  [erg/s]  (Powell Eq. 7)
_LBOL_COEF = 1.26e38
# hard-band (2-10 keV) bolometric correction (Powell: L_X = L_bol/8 for 14-195 keV;
# a 2-10 keV correction of ~20 -> here a single tunable constant, default 20).
_KBOL_HARD = 20.0

_ERDF_CENTRE = (float(np.log10(0.13)), 0.30, 3.70)   # log10 lambda*, delta1, delta2
_MBH_M_CENTRE = (7.76, 0.67, 0.33)                   # norm, slope, scatter [dex]


def erdf_dloglambda(loglam, log10_lstar, delta1, delta2):
    """Ananna 2022 ERDF dN/dlog10(lambda_Edd), broken power law (UNnormalised)."""
    x = 10.0 ** (np.asarray(loglam) - log10_lstar)
    return 1.0 / (x ** delta1 + x ** delta2)


class PowellAGNModel:
    """Analytic Powell 2022 AGN–halo model (predicts XLF + clustering)."""

    def __init__(self, theta_cosmo, hmf, z_mean=0.135,
                 mbh_m=_MBH_M_CENTRE, erdf=_ERDF_CENTRE, rho=0.0,
                 sham_scatter=0.20, log10lx_min=42.0, k_hard_to_soft=_HARD_TO_SOFT_RATIO,
                 kbol_hard=_KBOL_HARD, log10m_min=10.5, log10m_max=15.5, n_m=220,
                 loglx_lo=40.0, loglx_hi=47.0, n_lx=560, n_lam=240,
                 loglam_min=-3.0, loglam_max=1.5):
        self.theta = dict(theta_cosmo)
        self.hmf = hmf
        self.z_mean = float(z_mean)
        self.set_params(mbh_m, erdf, rho)
        self.sigma_ms = float(sham_scatter)          # SHMR scatter [dex]
        self.log10lx_min = float(log10lx_min)
        self.k_h2s = float(k_hard_to_soft)           # 0.5-2 / 2-10 keV
        self.kbol = float(kbol_hard)
        # grids
        self.log10m = np.linspace(log10m_min, log10m_max, n_m)          # M_halo [Msun/h]
        self.loglx = np.linspace(loglx_lo, loglx_hi, n_lx)              # log10 L_X hard [erg/s]
        self.dlx = self.loglx[1] - self.loglx[0]
        # ERDF support: Ananna 2022 defines it over ~ -3 < log10 lambda_Edd < 1;
        # the broken power law DIVERGES as lambda->0 (∝ lambda^-delta1), so a finite
        # low-lambda cutoff is physical AND required for normalisation.
        self.loglam = np.linspace(loglam_min, loglam_max, n_lam)        # log10 lambda_Edd
        self.dlam = self.loglam[1] - self.loglam[0]
        # SHMR: M_halo(Msun/h) -> log10 M_*(Msun).  Girelli takes log10 M_halo in Msun.
        h = float(self.theta["h"])
        self._log10ms = np.asarray(smhm_girelli20(self.log10m - np.log10(h), self.z_mean))
        self._logk_lx = np.log10(_LBOL_COEF / self.kbol)               # log L_X = logk + logMBH + loglam

    def set_params(self, mbh_m=None, erdf=None, rho=None, log10_ferdf=None):
        if mbh_m is not None:
            self.mu_bh, self.al_bh, self.sig_bh = [float(x) for x in mbh_m]
        if erdf is not None:
            self.log10_lstar, self.delta1, self.delta2 = [float(x) for x in erdf]
        if rho is not None:
            self.rho = float(np.clip(rho, 0.0, 0.999))
        if log10_ferdf is not None:
            self.log10_ferdf = float(log10_ferdf)   # ERDF normalisation (active fraction)
        elif not hasattr(self, "log10_ferdf"):
            self.log10_ferdf = -1.5                  # default ~3% active (calibrated by the fit)

    # --- core: P(log L_X | M_halo) -----------------------------------------
    def _p_lx_given_m(self):
        """(NM, Nlx) density dP(log L_X | M_halo)/dlog L_X (normalised).

        Shift-invariant: log L_X - logk - mean_bh(M_halo) = (log M_BH - mean_bh) +
        log lambda ~ N(0, sig_lm) + ERDF, whose density K(t) is the SAME for every
        halo (sig_lm is halo-independent).  So K = ERDF ⊛ Gaussian is computed ONCE
        and shifted per halo — O(NM·Nlx) instead of an (NM,Nlx,Nlam) cube.
        """
        mean_bh = self.mu_bh + self.al_bh * (self._log10ms - 11.0)          # (NM,)
        # rho>0 (Model 2): part of the M_* scatter aligns with M_halo, shrinking the
        # M_BH|M_halo scatter -> steeper L_X-selection with mass.
        var_lm = self.al_bh ** 2 * self.sigma_ms ** 2 * (1.0 - self.rho) + self.sig_bh ** 2
        sig_lm = float(np.sqrt(max(var_lm, 1e-6)))
        erdf = erdf_dloglambda(self.loglam, self.log10_lstar, self.delta1, self.delta2)
        erdf = erdf / (np.sum(erdf) * self.dlam)                            # density (Nlam,)
        # Gaussian kernel on the same dlam grid (+-6 sigma), as probabilities.
        ng = int(np.ceil(6.0 * sig_lm / self.dlam))
        x = np.arange(-ng, ng + 1) * self.dlam
        gk = np.exp(-0.5 * (x / sig_lm) ** 2); gk /= gk.sum()               # sums to 1
        K = np.convolve(erdf * self.dlam, gk, mode="full") / self.dlam     # density
        t0 = self.loglam[0] - ng * self.dlam                               # K grid start
        tgrid = t0 + np.arange(K.size) * self.dlam
        # P(logLX | M_halo)[m,:] = K( logLX - logk - mean_bh[m] )
        t = self.loglx[None, :] - self._logk_lx - mean_bh[:, None]         # (NM, Nlx)
        return np.interp(t.ravel(), tgrid, K, left=0.0, right=0.0).reshape(t.shape)

    # --- predictions -------------------------------------------------------
    def _dndm_bias(self):
        # HMF depends only on (cosmo, z) which are fixed for the AGN fit -> cache.
        if getattr(self, "_dndm_cache", None) is None:
            mh = 10.0 ** self.log10m
            dndm = np.asarray(self.hmf.dndm(mh, self.z_mean, self.theta))
            bias = np.asarray(self.hmf.bias(mh, self.z_mean, self.theta))
            self._dndm_cache = (dndm * mh * np.log(10.0), bias)          # (dn/dlog10M, bias)
        return self._dndm_cache

    def xlf(self, log10lx=None, band="hard"):
        """Predicted AGN XLF Phi(L_X) [ (Mpc/h)^-3 dex^-1 ] at z_mean (centrals).

        band='hard' (2-10 keV, matches Aird15 for validation) or 'soft' (0.5-2 keV).
        """
        p = self._p_lx_given_m() * 10.0 ** self.log10_ferdf   # (NM, Nlx); ferdf=active frac
        dndlog, _ = self._dndm_bias()
        phi = np.trapezoid(dndlog[:, None] * p, self.log10m, axis=0)      # (Nlx,)  per dex
        grid = self.loglx
        if band == "soft":
            grid = self.loglx + np.log10(self.k_h2s)
        if log10lx is None:
            return grid, phi
        return np.interp(np.asarray(log10lx), grid, phi, left=0.0, right=0.0)

    def _occ_and_mean_lx(self):
        """Central AGN occupation N_cen(>L_min|M_halo) and L_X-weighted mean per halo."""
        p = self._p_lx_given_m() * 10.0 ** self.log10_ferdf   # (NM, Nlx); ferdf=active frac
        sel = self.loglx >= self.log10lx_min
        n_cen = np.sum(p[:, sel], axis=1) * self.dlx                       # <N_AGN(L_X>L_min) | M_halo>
        lx = 10.0 ** self.loglx
        mean_lx = np.sum(p[:, sel] * lx[None, sel], axis=1) * self.dlx     # <L_X * 1(>L_min)>
        return np.clip(n_cen, 0.0, 1.0), mean_lx                          # (NM,), (NM,) [hard erg/s]

    def nc_ns_agn(self, log10m_arr, hod_params=None):
        """(N_cen_AGN, N_sat_AGN) on log10m_arr — L_X-selected AGN occupation.

        v1: centrals only (dominant); satellites (subhalo MF) are a documented
        refinement -> N_sat_AGN = 0.
        """
        n_cen, _ = self._occ_and_mean_lx()
        nc = np.interp(np.asarray(log10m_arr, float), self.log10m, n_cen, left=0.0, right=n_cen[-1])
        return nc.astype(np.float64), np.zeros_like(nc, dtype=np.float64)

    def mean_agn_lx(self, z=None, band="soft"):
        """HMF-weighted mean AGN L_X per selected halo [erg/s]."""
        n_cen, mean_lx = self._occ_and_mean_lx()
        dndlog, _ = self._dndm_bias()
        num = np.trapezoid(dndlog * mean_lx, self.log10m)
        den = np.trapezoid(dndlog * n_cen, self.log10m)
        lx_hard = num / max(den, 1e-300)
        return lx_hard * (self.k_h2s if band == "soft" else 1.0)

    def agn_bias(self, z=None):
        n_cen, _ = self._occ_and_mean_lx()
        dndlog, bias = self._dndm_bias()
        num = np.trapezoid(dndlog * bias * n_cen, self.log10m)
        den = np.trapezoid(dndlog * n_cen, self.log10m)
        return float(num / max(den, 1e-300))

    def agn_bias_of_lx(self, log10lx, band="soft"):
        """Large-scale halo bias of AGN *at* luminosity L_X (differential), at ``z_mean``.

        .. math::
            b(L_X) = \\frac{\\int b(M)\\,p(L_X|M)\\,\\frac{dn}{d\\log M}\\,dM}
                          {\\int p(L_X|M)\\,\\frac{dn}{d\\log M}\\,dM}

        The ERDF normalisation (``log10_ferdf``) cancels between numerator and
        denominator, so this depends only on the M_BH–M_*/ERDF *shape* and the HMF.
        Redshift is controlled by the instance (``self.z_mean``); to predict at
        another redshift build a fresh model at that z.

        Parameters
        ----------
        log10lx : array-like
            Luminosities to evaluate [log10 erg/s].
        band : {"soft", "hard"}
            Band of the *input* luminosities.  The internal grid is hard
            (2–10 keV); ``"soft"`` (0.5–2 keV) inputs are shifted by
            ``-log10(k_h2s)`` onto the hard grid before interpolation.
        """
        p = self._p_lx_given_m()                              # (NM, Nlx), hard grid
        dndlog, bias = self._dndm_bias()                      # (NM,), (NM,)
        num = np.trapezoid(dndlog[:, None] * bias[:, None] * p, self.log10m, axis=0)
        den = np.trapezoid(dndlog[:, None] * p,                self.log10m, axis=0)
        b_of_lx = num / np.maximum(den, 1e-300)               # (Nlx,) on self.loglx
        ll = np.asarray(log10lx, dtype=float)
        if band == "soft":
            ll = ll - np.log10(self.k_h2s)                    # soft -> hard
        return np.interp(ll, self.loglx, b_of_lx, left=b_of_lx[0], right=b_of_lx[-1])

    def median_host_logmhalo(self, z=None):
        n_cen, _ = self._occ_and_mean_lx()
        dndlog, _ = self._dndm_bias()
        w = dndlog * n_cen
        cdf = np.cumsum(w) / np.sum(w)
        return float(np.interp(0.5, cdf, self.log10m))

    def agn_emissivity_uk(self, k_arr, m200_arr, z, theta_cosmo, **kw):
        """Point-source AGN emissivity FT (flat in k): L_X,soft-weighted occupation.

        Returns (Nk, NM) = <L_X_soft * N_cen(>L_min)>(M) broadcast over k, in
        L_X/1e43 [normalised luminosity], matching the HODAgnModel convention.
        """
        n_cen, mean_lx = self._occ_and_mean_lx()
        emis = np.interp(np.log10(np.asarray(m200_arr, float)), self.log10m,
                         mean_lx * self.k_h2s / 1e43, left=0.0, right=mean_lx[-1] * self.k_h2s / 1e43)
        return np.broadcast_to(emis[None, :], (len(np.asarray(k_arr)), emis.size)).copy()
