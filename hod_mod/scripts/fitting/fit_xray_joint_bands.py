"""Phase B v2 — energy-band gas fit with FREE LX–M and kT–M scaling relations.

The temperature is NO LONGER self-similar.  The gas is parametrised directly by the
(normalisation, slope) of two scaling relations, free "to a limited extent" around
the Comparat+2025 GAS.py centrals (``validate_gas_profiles._gas_py_lx``/``_gas_py_kt``):

* LX–M:  log10(LX_0.5-2 / E(z)²) = lx_norm + lx_slope·(log10 M500c − 15),  σ_LX=0.3
* kT–M:  log10(kT / E(z)^{2/3}) = kt_slope·log10 M500c + kt_norm,          σ_kT=0.2

Free params ``[lx_norm, lx_slope, kt_norm, kt_slope, p2, r_max, log10DC]`` with
informative Gaussian priors N(44.7,0.3)/N(1.61,0.3)/N(−8,0.2)/N(0.6,0.15).

Forward model (constant-kT-per-halo): the band-b emissivity of a halo is
``LX(M)·[Λ_b(kT(M))/Λ_broad(kT(M))]``, so w_b(θ) is a **cheap** analytic weight
folded through a PRECOMPUTED per-mass transfer ``G(θ,M | p2,r_max)`` (the Limber +
Hankel of the normalised n_e²-shape FT with the halo-model mass kernel; validated to
reproduce a direct ``angular_cl_gX`` call to ~5e-5).  The eROSITA per-band response
weight ``A_b`` multiplies gas and AGN; the empirical S1 anchor sets the absolute
amplitude (anchor-relative v1).  No per-eval FT ⇒ fast MCMC.

Usage:
    HOD_MOD_DATA_DIR=/home/comparat/data HOD_MOD_RESULTS=/home/comparat/data/hod_mod_results \
      JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.fit_xray_joint_bands \
      --samples S1 S2 S3 S4 --mcmc
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, lsq_linear

from hod_mod import paths
from hod_mod.core.power_spectrum import LinearPowerSpectrum
from hod_mod.core.halo_mass_function import make_hmf
from hod_mod.core.halo_profiles import HaloProfile
from hod_mod.core.distances import hubble_e, comoving_distance
from hod_mod.gas import GasDensityDPM
from hod_mod.gas.cooling import ApecCoolingTable
from hod_mod.gas.conversions import m200_to_m500c
from hod_mod.connection.hod import ZuMandelbaum15HODModel
from hod_mod.observables.clustering import FullHaloModelPrediction
from hod_mod.observables.cross_spectra import HaloModelCrossSpectra, psf_king_window_ell
from hod_mod.agn.duty_cycle import DutyCycleAGNModel
from hod_mod.scripts.fitting import fit_comparat2025 as F
from hod_mod.scripts.fitting import fit_agn_duty_cycle_baseline as B
from hod_mod.scripts.fitting import fit_xray_joint as J
from hod_mod.scripts.validate_gas_profiles import _make_density_variant, _rho_crit_z

_OUT_DIR = os.fspath(paths.results_root() / "xray_joint_bands")

_BANDS = [f"{lo:04d}_E_{lo+100:04d}" for lo in range(500, 2000, 100)]
_BAND_EDGES = [(lo / 1000.0, (lo + 100) / 1000.0) for lo in range(500, 2000, 100)]
_NB = len(_BANDS)
_GAMMA_AGN = 1.8

# profile-shape grid (the scaling relations are analytic, NOT gridded)
_ALPHA_PROF = 0.9
_P2_GRID   = np.array([0.1, 0.6, 1.2, 2.4])   # gas outer slope
_RMAX_GRID = np.array([3.0, 4.0, 5.0])         # r_max / r200
_Z_METAL   = 0.3                                # representative gas metallicity [Z_sun]

# Scaling-relation pivots.  kt_norm is PIVOTED at M500c=10^14 (near the emission-
# weighted mass) — NOT at M500c=1 — so kt_norm ≡ log10(kT/E^{2/3}) at 10^14 keV and
# does NOT trade off with kt_slope (the M500c=1 pivot amplified the slope ~10x at the
# data mass and drove kT to unphysical values).  GAS.py at 10^14: 0.6·14−8 = +0.4.
_KT_PIVOT = 14.0
_LX_PIVOT = 15.0


def kT_of_M(log10_m500c, ez, kt_slope, kt_norm):
    """kT(M500c) [keV].  kt_norm = log10(kT/E(z)^{2/3}) at M500c = 10^_KT_PIVOT."""
    return 10.0 ** (kt_slope * (np.asarray(log10_m500c) - _KT_PIVOT) + kt_norm) * ez ** (2.0 / 3.0)


def LX_of_M(log10_m500c, ez, lx_norm, lx_slope, boost=1.0):
    """LX_0.5-2(M500c) [erg/s].  lx_norm at M500c = 10^_LX_PIVOT."""
    return 10.0 ** (lx_norm + lx_slope * (np.asarray(log10_m500c) - _LX_PIVOT)) * ez ** 2 * boost


# 8 free params: LX-M(2) + kT-M(2) + shape(2) + AGN duty cycle + gas metallicity.
# Metallicity is a param so the "free_metal" candidate can vary it; the relation
# params always carry their informative Gaussian priors (limited-extent freedom).
_PARAMS   = ["lx_norm", "lx_slope", "kt_norm", "kt_slope", "p2", "r_max", "log10DC",
             "z_metal", "agn_gamma"]
_BOOST_LX  = float(np.exp(0.5 * (np.log(10.0) * 0.3) ** 2))   # log-normal mean boost
_LOG10DC_LO, _LOG10DC_HI = B._LOG10DC_LO, B._LOG10DC_HI
_Z_FID = 0.3                                                  # fiducial gas metallicity [Z_sun]

# --- candidate machinery: per-candidate Gaussian priors + bounds (8-vectors) ------
# Each candidate isolates one hypothesis for the "gas runs hot" result and writes to
# its own subfolder.  sig=inf ⇒ flat prior (bounds only); a tight sig ⇒ effectively
# fixed.  Set by _apply_candidate() into module globals used by _log_prior/_weight.
_CANDIDATES = ["baseline", "agn_fixed", "free_metal", "flat_kt"]
_MU8 = _SIG8 = _BND8 = None


def _apply_candidate(cand, dc_fix=-1.8):
    """Configure the priors/bounds for a candidate; returns the output subdir name."""
    global _MU8, _SIG8, _BND8
    inf = np.inf
    # 9th param agn_gamma: AGN/continuum photon index, free with a Gaussian prior
    # around 1.8 so the continuum (not hot gas) can absorb a flat band spectrum.
    mu  = np.array([44.7, 1.61, 0.4, 0.6, 0.0, 0.0, 0.0, _Z_FID, _GAMMA_AGN])
    sig = np.array([0.3, 0.3, 0.2, 0.15, inf, inf, inf, 0.005, 0.3])   # base: DC flat, Z fixed
    bnd = np.array([[43.2, 46.2], [0.6, 2.6], [-0.9, 1.7], [0.1, 1.3],
                    [_P2_GRID[0], _P2_GRID[-1]], [_RMAX_GRID[0], _RMAX_GRID[-1]],
                    [_LOG10DC_LO, _LOG10DC_HI], [0.05, 1.0], [1.2, 2.6]])
    if cand == "agn_fixed":          # fix AGN duty cycle to the Phase-A broad-band value
        mu[6] = dc_fix; sig[6] = 0.02; bnd[6] = [dc_fix - 0.1, dc_fix + 0.1]
    elif cand == "free_metal":       # free the gas metallicity (flat in [0.05, 1.0])
        sig[7] = inf
    elif cand == "flat_kt":          # relax the kT-norm prior -> data-driven temperature
        sig[2] = 2.0
    elif cand != "baseline":
        raise ValueError(f"unknown candidate {cand}; choose from {_CANDIDATES}")
    _MU8, _SIG8, _BND8 = mu, sig, bnd
    return "baseline" if cand == "baseline" else cand


# --- band data + cooling + AGN spectral split + eROSITA response -------------

def _basename(label):
    return F._zenodo_fname(label).name.replace("_GALxEVT_wtheta.fits", "")


def load_band_data(label):
    """Per-band reconstructed w_b(theta): (Nb, Ntheta) wtheta/err + theta grid."""
    root = paths.data_path("xray_bands", _basename(label))
    rows = []; th_deg = None
    for band in _BANDS:
        fp = os.fspath(root / (band + ".fits"))
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"missing band file: {fp}\n"
                                    f"set $HOD_MOD_DATA_DIR to the data root.")
        t = Table.read(fp)
        if th_deg is None:
            th_deg = np.asarray(t["theta_mid"], float)
        rows.append((np.asarray(t["wtheta"], float), np.asarray(t["wtheta_err"], float)))
    return dict(theta_deg=th_deg, theta_arcsec=th_deg * 3600.0,
                theta_rad=th_deg * np.pi / 180.0,
                wtheta=np.vstack([r[0] for r in rows]),
                wtheta_err=np.vstack([r[1] for r in rows]))


_COOLING_CACHE = None
def _band_cooling():
    """15 per-band + 1 broad (0.5-2 keV) ApecCoolingTable (built once)."""
    global _COOLING_CACHE
    if _COOLING_CACHE is None:
        bands = [ApecCoolingTable(emin=lo, emax=hi) for lo, hi in _BAND_EDGES]
        broad = ApecCoolingTable(emin=0.5, emax=2.0)
        _COOLING_CACHE = (bands, broad)
    return _COOLING_CACHE


def _agn_band_fractions(gamma=_GAMMA_AGN):
    """Energy-flux fraction of a Γ power-law AGN in each band (Σ f_b = 1)."""
    p = 2.0 - gamma
    def _integ(lo, hi):
        return np.log(hi / lo) if abs(p) < 1e-9 else (hi ** p - lo ** p) / p
    tot = _integ(0.5, 2.0)
    return np.array([_integ(lo, hi) / tot for lo, hi in _BAND_EDGES])


_ARF_CACHE = None
def _arf_band_weights():
    """Per-band eROSITA effective-response weight A_b = <ARF·g_inband>_band (mean 1)."""
    global _ARF_CACHE
    if _ARF_CACHE is None:
        from hod_mod.gas import erosita_response as ER
        d = np.load(ER._DEFAULT_NPZ)
        emid = 0.5 * (d["energ_lo"] + d["energ_hi"])
        resp = d["arf_comb"] * d["g_inband"]
        A = np.array([np.mean(resp[(emid >= lo) & (emid < hi)]) for lo, hi in _BAND_EDGES])
        _ARF_CACHE = A / A.mean()
    return _ARF_CACHE


# --- per-mass transfer G(theta, M) (the validated core) ---------------------

def _shape_ft(dp, sc, z, theta_cosmo):
    """Normalised n_e²-shape FT Ŝ(k,M) = emissivity_uk(k,M)/emissivity_uk(k→0,M);
    cancels the density amplitude AND mass slope (those are now LX(M))."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        em = np.asarray(dp.emissivity_uk(sc["k_np"], sc["m_np"], sc["r_delta"],
                                         float(z), theta_cosmo))
    return em / em[0:1, :]


def _build_transfer(cross, hod, theta_cosmo, z_arr, nz, th_rad, ell, dp):
    """G(θ,M) (Ntheta, NM): w_gas(θ) = weight(M) @ G.  Linear-interp Limber + Hankel
    of the halo-model mass kernel × Ŝ, with the trapezoid dM measure folded in."""
    h = float(theta_cosmo["h"]); om = float(theta_cosmo["Omega_m"])
    chi_z = np.array([float(np.asarray(comoving_distance(float(zi), h, om)).ravel()[0]) * h
                      for zi in z_arr])
    dndchi = np.asarray(nz, float) / np.trapezoid(np.asarray(nz, float), chi_z)
    ell = np.asarray(ell, float)
    k_lim = (ell[:, None] + 0.5) / chi_z[None, :]           # (Nell, Nz)
    per_z = []
    m_np = None
    for iz, zi in enumerate(z_arr):
        sc = cross._get_static_cache(float(zi), theta_cosmo, hod)
        k_np = np.asarray(sc["k_np"], float); m_np = np.asarray(sc["m_np"], float)
        dndm = np.asarray(sc["dndm_np"], float); bias = np.asarray(sc["bias_np"], float)
        pk = np.asarray(sc["pk_lin"], float); uk = np.asarray(sc["uk"], float)
        nc, ns, n_gal, b_eff = cross._get_hod_weights(float(zi), theta_cosmo, hod, sc)
        nc = np.asarray(nc, float); ns = np.asarray(ns, float)
        S = _shape_ft(dp, sc, zi, theta_cosmo)             # (Nk, NM)
        Pk = (dndm[None, :] * (nc[None, :] + ns[None, :] * uk) * S / n_gal
              + dndm[None, :] * bias[None, :] * S * (b_eff * pk[:, None]))   # (Nk, NM)
        klz = k_lim[:, iz]
        Pint = np.stack([np.interp(klz, k_np, Pk[:, j]) for j in range(Pk.shape[1])], axis=1)
        per_z.append(dndchi[iz] * Pint / chi_z[iz] ** 2)   # (Nell, NM)
    Cl_M = np.trapezoid(np.stack(per_z, 0), chi_z, axis=0)  # (Nell, NM)
    Cl_M = Cl_M * np.asarray(psf_king_window_ell(ell, F._PSF_KING_THETA_C, 1.5))[:, None]
    # vectorised Hankel: G[θ,M] = Σ_ℓ (ℓ w_ℓ / 2π) j0(ℓθ) Cl_M[ℓ,M]
    from scipy.special import j0 as _j0
    wl = np.empty_like(ell)
    wl[1:-1] = (ell[2:] - ell[:-2]) / 2.0; wl[0] = (ell[1] - ell[0]) / 2.0
    wl[-1] = (ell[-1] - ell[-2]) / 2.0
    J0w = (ell * wl / (2.0 * np.pi))[None, :] * _j0(ell[None, :] * np.asarray(th_rad)[:, None])
    G = J0w @ Cl_M                                          # (Ntheta, NM)
    wtrap = np.empty_like(m_np)
    wtrap[1:-1] = (m_np[2:] - m_np[:-2]) / 2.0; wtrap[0] = (m_np[1] - m_np[0]) / 2.0
    wtrap[-1] = (m_np[-1] - m_np[-2]) / 2.0
    return G * wtrap[None, :], m_np


def _precompute(sample, hmf_backend):
    """Build (or load) the per-sample transfer grid G(θ,M | p2,r_max) + m500c(M) +
    E(z) + AGN template + band data."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    cache = os.path.join(_OUT_DIR, f"{sample}_transfer.npz")
    bd = load_band_data(sample)
    th_as = bd["theta_arcsec"]; th_rad = bd["theta_rad"]
    mask = (th_as >= 8.0) & (th_as <= 300.0)
    if os.path.exists(cache):
        d = np.load(cache)
        if (np.array_equal(d["p2_grid"], _P2_GRID) and np.array_equal(d["rmax_grid"], _RMAX_GRID)
                and d["G_grid"].shape[-2] == th_as.size):
            return (d["G_grid"], d["log10_m500c"], float(d["ez"]), d["agn_dc1"], bd, mask)
        print(f"  [{sample}] cached transfer stale -> rebuild", flush=True)

    th = F._THETA_COSMO
    pk = LinearPowerSpectrum(); hmf = make_hmf(hmf_backend, pk_func=pk.pk_linear)
    colo = dict(flat=True, H0=th["h"] * 100.0, Om0=th["Omega_m"], Ob0=th["Omega_b"],
                sigma8=0.811, ns=th["n_s"])
    hp = HaloProfile(colo, cm_relation="diemer19")
    hod = ZuMandelbaum15HODModel(hmf, hmf.bias)
    fhmp = FullHaloModelPrediction(pk, hod, hp)
    hod_params = B._build_hod_params(sample)
    z_arr, nz = F._build_nz_fast(sample)
    zmean = float(F.SAMPLES[sample]["zmean"])
    cross = HaloModelCrossSpectra(fhmp, density_profile=GasDensityDPM(model=2))

    # AGN template (DC=1)
    agn = DutyCycleAGNModel(sample=sample, theta_cosmo=th, hmf=hmf, log10DC=0.0)
    cross_a = HaloModelCrossSpectra(fhmp, density_profile=GasDensityDPM(model=2), agn_model=agn)
    comp = cross_a.angular_cl_gX(F._ELL, z_arr, nz, th, hod_params,
                                 psf_king_theta_c_arcsec=F._PSF_KING_THETA_C,
                                 return_components=True, agn_kwargs={"log10DC": 0.0}, n_workers=1)
    agn_dc1 = F._hankel(np.asarray(comp["agn"], float), th_rad)

    # m500c(M) at zmean + E(zmean) (relation inputs; constant-per-halo)
    sc0 = cross._get_static_cache(zmean, th, hod_params)
    m_np = np.asarray(sc0["m_np"], float); c_np = np.asarray(sc0["c_np"], float)
    r_delta = np.asarray(sc0["r_delta"], float)
    m500c_h, _ = m200_to_m500c(m_np, c_np, r_delta, _rho_crit_z(zmean))
    log10_m500c = np.log10(np.asarray(m500c_h, float) / th["h"])   # Msun (physical)
    ez = float(hubble_e(zmean, th["Omega_m"]))

    G_grid = np.zeros((_P2_GRID.size, _RMAX_GRID.size, th_as.size, m_np.size))
    t0 = time.time()
    for i, p2 in enumerate(_P2_GRID):
        for j, rmax in enumerate(_RMAX_GRID):
            dp = _make_density_variant(model=2, ne_03=1e-4, beta=0.5, alpha_in=_ALPHA_PROF,
                                       alpha_tr=2.0, alpha_out=_ALPHA_PROF + 2.0 * float(p2))
            dp._r_max_factor = float(rmax)
            G_grid[i, j], _ = _build_transfer(cross, hod_params, th, z_arr, nz, th_rad, F._ELL, dp)
        print(f"  [{sample}] transfer p2={p2:.2f} ({(i+1)*_RMAX_GRID.size}/"
              f"{_P2_GRID.size*_RMAX_GRID.size}) [{time.time()-t0:.0f}s]", flush=True)
    np.savez(cache, G_grid=G_grid, log10_m500c=log10_m500c, ez=ez, agn_dc1=agn_dc1,
             p2_grid=_P2_GRID, rmax_grid=_RMAX_GRID)
    print(f"[{sample}] transfer built in {time.time()-t0:.0f}s -> {cache}", flush=True)
    return G_grid, log10_m500c, ez, agn_dc1, bd, mask


# --- model + priors ---------------------------------------------------------

def _weight_bands(p, S):
    """LX(M)·Λ_b(kT(M))/Λ_broad(kT(M))  — (Nb, NM), from the free relations."""
    lx_norm, lx_slope, kt_norm, kt_slope = p[0], p[1], p[2], p[3]
    lm = S["log10_m500c"]
    kT = np.clip(kT_of_M(lm, S["ez"], kt_slope, kt_norm), S["kT_lo"], S["kT_hi"])
    LX = LX_of_M(lm, S["ez"], lx_norm, lx_slope, boost=_BOOST_LX)
    Zc = np.full_like(kT, np.clip(p[7], 0.05, 3.0))    # gas metallicity (free in free_metal)
    lam_broad = np.maximum(np.asarray(S["cool_broad"](kT, Zc)), 1e-40)
    lam_b = np.stack([np.asarray(cb(kT, Zc)) for cb in S["cool_bands"]])   # (Nb, NM)
    return LX[None, :] * (lam_b / lam_broad[None, :])


def _components_bands(p, S):
    """ARF-weighted gas and AGN (Nb, Ntheta)."""
    G = S["G_interp"]([[p[4], p[5]]])[0].reshape(S["nth"], -1)   # (Ntheta, NM)
    gas = S["c_total"] * (_weight_bands(p, S) @ G.T)             # (Nb, Ntheta)
    fb = _agn_band_fractions(gamma=p[8]) if len(p) > 8 else S["fb"]   # free AGN photon index
    agn = 10.0 ** p[6] * S["c_obs_total"] * (fb[:, None] * S["agn_dc1"][None, :])
    return S["arf"][:, None] * gas, S["arf"][:, None] * agn


def _model_bands(p, S):
    g, a = _components_bands(p, S)
    return g + a


def _chi2_sample(p, S):
    r = (_model_bands(p, S) - S["wtheta"])[:, S["mask"]] / S["err"][:, S["mask"]]
    return float(np.sum(r ** 2))


def _bounds():
    return _BND8


def _log_prior(p):
    for v, (lo, hi) in zip(p, _BND8):
        if not (lo <= v <= hi):
            return -np.inf
    fin = np.isfinite(_SIG8)
    d = (np.asarray(p, float)[fin] - _MU8[fin]) / _SIG8[fin]
    return -0.5 * float(np.sum(d ** 2))


def _neg_log_prob(p, samples):
    lp = _log_prior(p)
    if not np.isfinite(lp):
        return 1e30
    return -lp + 0.5 * sum(_chi2_sample(p, S) for S in samples.values())


def _anchor(samples, anchor_sample):
    """Empirical gas amplitude anchor on the anchor sample: at the prior-centre
    relation params + mid shape, free (A_gas, A_AGN) lsq over the band data -> the
    A_gas that reproduces it defines c_total (density_norm≡1)."""
    S = samples[anchor_sample]
    pc = np.array([_MU8[0], _MU8[1], _MU8[2], _MU8[3], 0.6, 4.0, -1.5, _Z_FID, _GAMMA_AGN])
    St = dict(S); St["c_total"] = 1.0
    gas1, _ = _components_bands(pc, St)                # c_total=1
    agn1 = St["arf"][:, None] * St["c_obs_total"] * (St["fb"][:, None] * St["agn_dc1"][None, :])
    m = S["mask"]; w = 1.0 / S["err"][:, m].ravel()
    A = np.column_stack([gas1[:, m].ravel() * w, agn1[:, m].ravel() * w])
    res = lsq_linear(A, S["wtheta"][:, m].ravel() * w, bounds=([0, 0], [np.inf, np.inf]),
                     method="bvls")
    chi2 = float(np.sum((A @ res.x - S["wtheta"][:, m].ravel() * w) ** 2))
    return float(res.x[0]), chi2


# --- figures ----------------------------------------------------------------

def _figures_bands(tag, samples, flat, chain_full, lp_full, nburn, map_p, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ecen = np.array([0.5 * (lo + hi) for lo, hi in _BAND_EDGES])
    np_ = len(_PARAMS)
    fig, axs = plt.subplots(np_ + 1, 1, figsize=(8, 1.5 * (np_ + 1)), sharex=True)
    for i, p in enumerate(_PARAMS):
        axs[i].plot(chain_full[:, :, i], color="C0", alpha=0.12, lw=0.5)
        axs[i].set_ylabel(p, fontsize=8); axs[i].axvline(nburn, color="C3", ls=":")
    axs[-1].plot(lp_full, color="C0", alpha=0.12, lw=0.5)
    axs[-1].set_ylabel("log prob", fontsize=8); axs[-1].axvline(nburn, color="C3", ls=":")
    axs[0].set_title(f"{tag}: band-fit traces (red=burn-in {nburn})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{tag}_bands_traces.png"), dpi=110)
    plt.close(fig)
    try:
        import corner
        fig = corner.corner(flat, labels=_PARAMS, truths=list(map_p))
        fig.savefig(os.path.join(out_dir, f"{tag}_bands_corner.png"), dpi=110); plt.close(fig)
    except Exception as e:
        print(f"  (corner skipped: {e})", flush=True)
    for s, S in samples.items():
        th = S["th_as"]; wd = S["wtheta"]; err = S["err"]
        gas_m, agn_m = _components_bands(map_p, S); tot = gas_m + agn_m
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for th0, c in [(10.0, "C0"), (40.0, "C1")]:
            k = int(np.argmin(np.abs(th - th0)))
            a1.errorbar(ecen, wd[:, k], yerr=err[:, k], fmt="o", ms=3, color=c,
                        label=fr"data $\theta$≈{th[k]:.0f}″")
            a1.plot(ecen, tot[:, k], "-", color=c)
        a1.set_xlabel("band energy [keV]"); a1.set_ylabel(r"$w_b(\theta)$")
        a1.set_title(f"{s}: band spectrum (pts=data, line=model)", fontsize=9)
        a1.legend(fontsize=7); a1.axhline(0, color="grey", lw=0.5)
        m = S["mask"]
        sd = np.nansum(wd[0:4], 0); hd = np.nansum(wd[10:15], 0)
        sm = np.nansum(tot[0:4], 0); hm = np.nansum(tot[10:15], 0)
        good = m & np.isfinite(sd) & np.isfinite(hd) & (hd > 0)
        a2.plot(th[good], (sd / hd)[good], "ko", ms=3, label="data")
        a2.plot(th[good], (sm / hm)[good], "C0-", label="model")
        a2.set_xscale("log"); a2.set_xlabel(r"$\theta$ [arcsec]")
        a2.set_ylabel("soft(0.5-0.9)/hard(1.5-2.0)")
        a2.set_title(f"{s}: band ratio (temperature)", fontsize=9); a2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{s}_bands_spectrum.png"), dpi=110)
        plt.close(fig)


# --- main -------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", nargs="+", default=list(F.SAMPLES))
    ap.add_argument("--hmf", default="tinker08")
    ap.add_argument("--f-sys", type=float, default=0.05)
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--mcmc", action="store_true")
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=8000)
    ap.add_argument("--nburn", type=int, default=2000)
    ap.add_argument("--candidate", default="baseline", choices=_CANDIDATES,
                    help="hot-gas hypothesis to test; writes to its own subfolder")
    args = ap.parse_args(argv)
    tag = "_".join(args.samples)

    # candidate config: agn_fixed pins log10DC to the Phase-A broad-band value
    dc_fix = -1.8
    if args.candidate == "agn_fixed":
        af = "S1" if "S1" in args.samples else args.samples[0]
        pa = os.path.join(J._OUT_DIR, f"{af}_bb_summary.json")
        if os.path.isfile(pa):
            dc_fix = float(json.load(open(pa))["posterior"]["log10DC"]["median"])
    subdir = _apply_candidate(args.candidate, dc_fix)
    out_dir = _OUT_DIR if subdir == "baseline" else os.path.join(_OUT_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Candidate '{args.candidate}' -> {out_dir}"
          + (f"  (dc_fix={dc_fix:.2f})" if args.candidate == "agn_fixed" else ""), flush=True)

    cool_bands, cool_broad = _band_cooling()
    fb = _agn_band_fractions(); arf = _arf_band_weights()
    kT_lo, kT_hi = 0.09, 19.0
    samples = {}
    for s in args.samples:
        G_grid, log10_m500c, ez, agn_dc1, bd, mask = _precompute(s, args.hmf)
        err = np.sqrt(bd["wtheta_err"] ** 2 + (args.f_sys * np.abs(bd["wtheta"])) ** 2)
        nth = bd["theta_arcsec"].size
        G_interp = RegularGridInterpolator(
            (_P2_GRID, _RMAX_GRID), G_grid.reshape(_P2_GRID.size, _RMAX_GRID.size, -1),
            method="linear", bounds_error=False, fill_value=None)
        samples[s] = dict(G_interp=G_interp, nth=nth, log10_m500c=log10_m500c, ez=ez,
                          agn_dc1=agn_dc1, wtheta=bd["wtheta"], err=err, mask=mask,
                          th_as=bd["theta_arcsec"], fb=fb, arf=arf,
                          cool_bands=cool_bands, cool_broad=cool_broad, kT_lo=kT_lo, kT_hi=kT_hi,
                          c_obs_total=J._c_obs_total(s), srx=float(F.load_data(s)["beckground"][0]),
                          n_pts=int(mask.sum()) * _NB)
        print(f"[{s}] transfer ready, n_pts={samples[s]['n_pts']}", flush=True)

    anchor_sample = "S1" if "S1" in samples else args.samples[0]
    samples[anchor_sample]["c_total"] = 1.0
    c_total_S1, chi2a = _anchor(samples, anchor_sample)
    srx_a = samples[anchor_sample]["srx"]
    print(f"\nAnchor on {anchor_sample}: c_total={c_total_S1:.3e} (unconstrained band chi2={chi2a:.1f})",
          flush=True)
    for s, S in samples.items():
        S["c_total"] = c_total_S1 * srx_a / S["srx"]

    n_tot = sum(S["n_pts"] for S in samples.values())
    print(f"\nBand MAP over {len(samples)} samples × {_NB} bands, {n_tot} pts, "
          f"{len(_PARAMS)} shared params (LX-M + kT-M relations free) ...", flush=True)

    def nlp(p):
        return _neg_log_prob(p, samples)

    # 8-vec starts (…, log10DC, z_metal); log10DC seeded at the candidate's bound mid
    dc0 = float(np.clip(-1.5, _BND8[6, 0], _BND8[6, 1]))
    starts = [   # kt_norm = log10(kT/E^{2/3}) at M500c=10^14 (GAS.py: +0.4)
        [44.7, 1.61, 0.40, 0.60, 0.6, 4.0, dc0, _Z_FID],
        [45.0, 1.80, 0.20, 0.70, 0.3, 3.5, dc0 + 0.4, _Z_FID],
        [44.4, 1.40, 0.60, 0.50, 1.2, 4.5, dc0 - 0.4, 0.5],
        [44.7, 1.61, 0.30, 0.65, 0.6, 4.0, dc0, 0.15],
    ]
    best = None
    for q0 in starts:
        o = minimize(nlp, np.array(q0), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=6000))
        if best is None or o.fun < best.fun:
            best = o
        print(f"  start {np.round(q0,2)} -> chi2={2*o.fun:.1f}", flush=True)
    map_p = best.x
    chi2 = 2.0 * sum(_chi2_sample(map_p, S) for S in samples.values())
    ndof = max(n_tot - len(_PARAMS), 1)
    out = dict(zip(_PARAMS, [float(v) for v in map_p]))
    out["chi2"] = chi2; out["ndof"] = ndof; out["chi2_per_dof"] = chi2 / ndof
    out["chi2_per_sample"] = {s: float(_chi2_sample(map_p, S)) for s, S in samples.items()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"joint_bands_map_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== BAND MAP (free LX-M, kT-M) ===")
    for k in _PARAMS:
        print(f"  {k:10s} = {out[k]:+.4f}")
    print(f"  chi2/dof   = {chi2:.1f}/{ndof} = {chi2/ndof:.3f}")
    for s, v in out["chi2_per_sample"].items():
        print(f"    {s}: chi2={v:.1f} ({samples[s]['n_pts']} pts)", flush=True)

    if args.map_only or not args.mcmc:
        return out

    import emcee
    ndim = len(_PARAMS); nw = args.nwalkers
    rng = np.random.default_rng(42)
    p0 = map_p + 1e-3 * rng.standard_normal((nw, ndim)) * np.ptp(_bounds(), axis=1)
    p0 = np.clip(p0, _bounds()[:, 0] + 1e-6, _bounds()[:, 1] - 1e-6)
    sampler = emcee.EnsembleSampler(nw, ndim, lambda p: -nlp(p) if nlp(p) < 1e29 else -np.inf)
    print(f"\nBAND MCMC: {nw} walkers × {args.nsteps} steps ({tag}) ...", flush=True)
    t0 = time.time(); sampler.run_mcmc(p0, args.nsteps, progress=False)
    print(f"MCMC done in {time.time()-t0:.0f}s; acceptance={np.mean(sampler.acceptance_fraction):.2f}",
          flush=True)
    flat = sampler.get_chain(discard=args.nburn, flat=True)
    np.savez(os.path.join(out_dir, f"{tag}_bands_chain.npz"), flatchain=flat,
             log_prob=sampler.get_log_prob(discard=args.nburn, flat=True),
             chain=sampler.get_chain(), lp=sampler.get_log_prob(), params=_PARAMS,
             nburn=args.nburn, c_total={s: S["c_total"] for s, S in samples.items()})
    pct = np.percentile(flat, [16, 50, 84], axis=0)
    post = {p: dict(median=float(pct[1, i]), lo=float(pct[1, i] - pct[0, i]),
                    hi=float(pct[2, i] - pct[1, i])) for i, p in enumerate(_PARAMS)}
    with open(os.path.join(out_dir, f"{tag}_bands_summary.json"), "w") as fh:
        json.dump(dict(samples=args.samples, map=out,
                       acceptance=float(np.mean(sampler.acceptance_fraction)), posterior=post),
                  fh, indent=2)
    print("Posterior (median +hi -lo):", flush=True)
    for i, p in enumerate(_PARAMS):
        print(f"  {p:10s} = {pct[1,i]:.3f} +{pct[2,i]-pct[1,i]:.3f} -{pct[1,i]-pct[0,i]:.3f}",
              flush=True)
    _figures_bands(tag, samples, flat, sampler.get_chain(), sampler.get_log_prob(),
                   args.nburn, map_p, out_dir)
    print(f"Saved chain/summary/figures -> {out_dir}", flush=True)
    return out


if __name__ == "__main__":
    main()
