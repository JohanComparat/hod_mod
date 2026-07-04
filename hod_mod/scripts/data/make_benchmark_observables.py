r"""Build the benchmark-observables JSON tree in the data repository.

One JSON file per (bibliographic reference, observable, sample), organised as
``<out>/<wavelength>/<tracer>/<RefKey>__<observable>[__<sample>].json``.
Each file carries the full metadata (reference with links, units, sample
definition, provenance) and the data with uncertainties.  Three provenance
types:

* ``observed``               — real measurements ingested from the curated
                               local files (repo ``data/`` benchmark folders,
                               ``hod_mod/data/benchmarks/xray``,
                               ``hod_mod/data/xray_bands``);
* ``observed_derived_fit``   — points evaluated from a *published* fitting
                               function with published parameters (MD14 SFRD,
                               ALFALFA HIMF Schechter);
* ``simulated``              — the forward-model fiducial prediction with the
                               forecast survey noise, extracted from the
                               tier-2/tier-3 production npz.  These stand in
                               until the operator extracts the published
                               table (``needs_operator_extraction: true``).

Usage::

    python -m hod_mod.scripts.data.make_benchmark_observables \
        [--out /home/comparat/data/benchmark_observables]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RESULTS = os.environ.get("HOD_MOD_RESULTS", "/home/comparat/data/hod_mod_results")
_T2_NPZ = os.path.join(_RESULTS, "tier2_forecast", "tier2_forecast_nb6.npz")
_T3_NPZ = os.path.join(_RESULTS, "tier3_forecast", "tier3_forecast_nb6_smoke.npz")
SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# references (citation + link); keys match docs/references.rst
# --------------------------------------------------------------------------
REFS = {
    "Zehavi2005": ("Zehavi I. et al. 2005, ApJ 630, 1",
                   "https://arxiv.org/abs/astro-ph/0408569"),
    "Zheng2007": ("Zheng Z. et al. 2007, ApJ 667, 760",
                  "https://arxiv.org/abs/astro-ph/0703457"),
    "Guo2018": ("Guo H. et al. 2018, ApJ 858, 30",
                "https://arxiv.org/abs/1803.07697"),
    "Guo2019": ("Guo H. et al. 2019, ApJ 871, 147",
                "https://arxiv.org/abs/1810.05318"),
    "Leauthaud2012": ("Leauthaud A. et al. 2012, ApJ 744, 159",
                      "https://arxiv.org/abs/1104.0928"),
    "More2015": ("More S. et al. 2015, ApJ 806, 2",
                 "https://arxiv.org/abs/1407.1856"),
    "vanUitert2016": ("van Uitert E. et al. 2016, MNRAS 459, 3251",
                      "https://arxiv.org/abs/1601.06791"),
    "Zacharegkas2025": ("Zacharegkas G. et al. 2025",
                        "https://arxiv.org/abs/2106.08438"),
    "ZuMandelbaum2015": ("Zu Y. & Mandelbaum R. 2015, MNRAS 454, 1161",
                         "https://arxiv.org/abs/1505.02781"),
    "Lange2025": ("Lange J.U. et al. 2025",
                  "https://arxiv.org/abs/2502.10230"),
    "Comparat2025": ("Comparat J. et al. 2025, A&A 697, A173",
                     "https://arxiv.org/abs/2503.19796"),
    "Comparat2023": ("Comparat J. et al. 2023, A&A 673, A122",
                     "https://arxiv.org/abs/2301.01388"),
    "Comparat2015OII": ("Comparat J. et al. 2015, A&A 575, A40",
                        "https://arxiv.org/abs/1408.1523"),
    "MadauDickinson2014": ("Madau P. & Dickinson M. 2014, ARA&A 52, 415",
                           "https://arxiv.org/abs/1403.0007"),
    "Jones2018ALFALFA": ("Jones M.G. et al. 2018, MNRAS 477, 2",
                         "https://arxiv.org/abs/1802.00053"),
    "Aird2015": ("Aird J. et al. 2015, MNRAS 451, 1892",
                 "https://arxiv.org/abs/1503.01120"),
    "Kulkarni2019": ("Kulkarni G. et al. 2019, MNRAS 488, 1035",
                     "https://arxiv.org/abs/1807.09774"),
    "Sobral2013": ("Sobral D. et al. 2013, MNRAS 428, 1128",
                   "https://arxiv.org/abs/1202.3436"),
    "Wyder2005": ("Wyder T.K. et al. 2005, ApJ 619, L15",
                  "https://arxiv.org/abs/astro-ph/0411364"),
    "Driver2022": ("Driver S.P. et al. 2022, MNRAS 513, 439",
                   "https://arxiv.org/abs/2203.08539"),
    "Weaver2023": ("Weaver J.R. et al. 2023, A&A 677, A184",
                   "https://arxiv.org/abs/2212.02512"),
    "Moustakas2013": ("Moustakas J. et al. 2013, ApJ 767, 50",
                      "https://arxiv.org/abs/1301.1688"),
    "Popesso2023": ("Popesso P. et al. 2023, MNRAS 519, 1526",
                    "https://arxiv.org/abs/2203.10487"),
    "Wright2025": ("Wright A.H. et al. 2025, A&A (KiDS-Legacy)",
                   "https://arxiv.org/abs/2503.19441"),
    "Qu2024": ("Qu F.J. et al. 2024, ApJ 962, 112 (ACT DR6)",
               "https://arxiv.org/abs/2304.05202"),
    "Kim2024": ("Kim J. et al. 2024, JCAP 12, 022 (DESI LRG x ACT DR6)",
                "https://arxiv.org/abs/2407.04606"),
    "Schaan2021": ("Schaan E. et al. 2021, Phys. Rev. D 103, 063513",
                   "https://arxiv.org/abs/2009.05557"),
    "Amodeo2021": ("Amodeo S. et al. 2021, Phys. Rev. D 103, 063514",
                   "https://arxiv.org/abs/2009.05558"),
    "CHIME2023": ("CHIME Collaboration 2023, ApJ 947, 16",
                  "https://arxiv.org/abs/2202.01242"),
    "CHIMEauto2025": ("CHIME Collaboration 2025 (preprint)",
                      "https://arxiv.org/abs/2511.19620"),
    "Ghirardini2024": ("Ghirardini V. et al. 2024, A&A (eRASS1 clusters)",
                       "https://arxiv.org/abs/2402.08458"),
    "Bonato2021": ("Bonato M. et al. 2021, A&A 656, A48",
                   "https://arxiv.org/abs/2109.06735"),
    "Kondapally2022": ("Kondapally R. et al. 2022, MNRAS 513, 3742",
                       "https://arxiv.org/abs/2204.07588"),
    "Powell2022": ("Powell M.C. et al. 2022, ApJ 938, 77 (BASS XXXVI)",
                   "https://arxiv.org/abs/2209.02728"),
    "Ananna2022": ("Ananna T.T. et al. 2022, ApJS 261, 9 (BASS XXX)",
                   "https://arxiv.org/abs/2201.05603"),
    "KormendyHo2013": ("Kormendy J. & Ho L.C. 2013, ARA&A 51, 511",
                       "https://arxiv.org/abs/1304.7762"),
    "Greene2020": ("Greene J.E. et al. 2020, ARA&A 58, 257",
                   "https://arxiv.org/abs/1911.09678"),
    "Finkelstein2015": ("Finkelstein S.L. et al. 2015, ApJ 810, 71",
                        "https://arxiv.org/abs/1410.5439"),
    "Harikane2023": ("Harikane Y. et al. 2023, ApJS 265, 5",
                     "https://arxiv.org/abs/2208.01612"),
    "Song2016": ("Song M. et al. 2016, ApJ 825, 5",
                 "https://arxiv.org/abs/1507.05636"),
    "Behroozi2019": ("Behroozi P. et al. 2019, MNRAS 488, 3143",
                     "https://arxiv.org/abs/1806.07893"),
}


def _ref(key, note=None):
    cit, url = REFS[key]
    r = {"key": key, "citation": cit, "arxiv": url}
    if note:
        r["note"] = note
    return r


def _ref_from_meta(meta, fallback_key, note=None):
    """Prefer the curated folder metadata (authoritative arXiv/DOI)."""
    r = _ref(fallback_key, note)
    if meta.get("paper"):
        r["citation"] = meta["paper"]
        if "arXiv:" in meta["paper"] and not meta.get("arxiv"):
            a = meta["paper"].split("arXiv:")[1].split(",")[0].strip()
            r["arxiv"] = f"https://arxiv.org/abs/{a}"
    if meta.get("arxiv"):
        a = meta["arxiv"].replace("arXiv:", "")
        r["arxiv"] = f"https://arxiv.org/abs/{a}"
    if meta.get("doi"):
        r["doi"] = f"https://doi.org/{meta['doi']}"
    return r


def _san(x):
    """JSON-safe: numpy -> python, non-finite -> None."""
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_san(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_san(v) for v in x]
    if isinstance(x, dict):
        return {k: _san(v) for k, v in x.items()}
    return x


def write_entry(out_root, wavelength, tracer, ref_key, observable, payload,
                sample=None):
    d = os.path.join(out_root, wavelength, tracer)
    os.makedirs(d, exist_ok=True)
    stem = f"{ref_key}__{observable}" + (f"__{sample}" if sample else "")
    path = os.path.join(d, stem + ".json")
    body = {"schema_version": SCHEMA_VERSION, "wavelength": wavelength,
            "tracer": tracer, "observable": observable}
    if sample:
        body["sample_id"] = sample
    body.update(payload)
    with open(path, "w") as fh:
        json.dump(_san(body), fh, indent=1)
    rel = os.path.relpath(path, out_root)
    print("[json]", rel)
    return rel


# --------------------------------------------------------------------------
# 1. observed: curated CSV benchmark folders (repo data/)
# --------------------------------------------------------------------------
# (folder, ref_key, wavelength, tracer, [(csv, observable, sample), ...])
CSV_SETS = [
    ("zheng2007_sdss", "Zehavi2005", "optical", "galaxies",
     [("wp_mr21_sdss.csv", "wp", "mr_lt_m21")]),
    ("guo2018_sdss", "Guo2018", "optical", "galaxies",
     [("wp_mstar10_lowz.csv", "wp", "mstar10_lowz")]),
    ("guo2019_eboss_elg", "Guo2019", "optical", "galaxies",
     [("wp_elg_z08.csv", "wp", "elg_z08")]),
    ("leauthaud2012_cosmos", "Leauthaud2012", "optical", "galaxies",
     [("ds_photo_z2_thresh106.csv", "ds", "z2_thresh106")]),
    ("vanutert2016_gama", "vanUitert2016", "optical", "galaxies",
     [("wp_bin2_104_108.csv", "wp", "bin2_104_108"),
      ("ds_bin2_104_108.csv", "ds", "bin2_104_108")]),
    ("zacharegkas2025_des", "Zacharegkas2025", "optical", "galaxies",
     [("wp_des_bin1.csv", "wp", "des_bin1"),
      ("ds_des_bin1.csv", "ds", "des_bin1")]),
]


def _read_csv(path):
    """#-comments, optional un-commented header row of column names."""
    header, colnames, rows = [], None, []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            header.append(ln[1:].strip())
            continue
        parts = [p.strip() for p in ln.split(",")]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            if colnames is None and rows == []:
                colnames = parts
            # else: silently skip malformed line
    return np.asarray(rows, dtype=float), header, colnames


def _load_meta(folder):
    p = os.path.join(folder, "metadata.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def ingest_csv_sets(out_root):
    idx = []
    for folder, ref, wl, tr, files in CSV_SETS:
        base = os.path.join(_REPO, "data", folder)
        meta = _load_meta(base)
        for csv, obs, sample in files:
            fp = os.path.join(base, csv)
            if not os.path.exists(fp):
                print("[skip] missing", fp)
                continue
            arr, header, colnames = _read_csv(fp)
            if arr.size == 0:
                print(f"[skip] {folder}/{csv}: no data rows (see file header)")
                continue
            cols = (colnames or list(meta.get("columns", {}).keys()))[:arr.shape[1]] or \
                [f"col{i}" for i in range(arr.shape[1])]
            data = {c: arr[:, i] for i, c in enumerate(cols)}
            payload = {
                "reference": _ref_from_meta(meta, ref, meta.get("data_source")),
                "provenance": {
                    "type": "observed",
                    "origin": f"repo data/{folder}/{csv}",
                    "extraction_method": meta.get("extraction_method"),
                    "needs_operator_extraction": False},
                "sample": {k: meta.get(k) for k in
                           ("survey", "z_eff", "pi_max_hMpc", "cosmology")
                           if k in meta},
                "units": meta.get("columns", {}),
                "data": data,
                "notes": meta.get("notes"),
            }
            idx.append(write_entry(out_root, wl, tr, ref, obs, payload, sample))
    return idx


def ingest_multi_sample(out_root):
    """ZM15 M*-bin files, More2015 and Lange2025 per-sample folders."""
    idx = []
    # --- ZM15: wp/ds per stellar-mass bin (digitized Fig 6)
    base = os.path.join(_REPO, "data", "zumandelbaum2015_sdss")
    meta = _load_meta(base)
    for fp in sorted(glob.glob(os.path.join(base, "*_bin_*.csv"))):
        name = os.path.basename(fp)
        obs = "wp" if name.startswith("wp") else "ds"
        sample = name.split("_bin_")[1].replace(".csv", "")
        arr, header, colnames = _read_csv(fp)
        if arr.size == 0:
            print(f"[skip] {name}: no data rows")
            continue
        cols = (colnames or ["rp_hMpc", obs, obs + "_err"])[:arr.shape[1]]
        payload = {
            "reference": _ref("ZuMandelbaum2015", "Figure 6, digitized"),
            "provenance": {"type": "observed",
                           "origin": f"repo data/zumandelbaum2015_sdss/{name}",
                           "extraction_method": "model-anchored digitization "
                                                "of Figure 6 (see file header)",
                           "needs_operator_extraction": False},
            "sample": {"survey": "SDSS DR7", "z_eff": 0.1,
                       "mstar_bin_log10": sample.replace("p", ".")},
            "units": {"rp_hMpc": "h^-1 Mpc",
                      obs: "h^-1 Mpc" if obs == "wp" else "h Msun/pc^2"},
            "data": {c: arr[:, i] for i, c in enumerate(cols)},
            "notes": (meta or {}).get("notes"),
        }
        idx.append(write_entry(out_root, "optical", "galaxies",
                               "ZuMandelbaum2015", obs, payload, sample))
    # --- per-sample folders with their own metadata.json
    for folder, ref in (("more2015_boss_cmass", "More2015"),
                        ("lange2025_desi_dr1", "Lange2025")):
        base = os.path.join(_REPO, "data", folder)
        for sub in sorted(os.listdir(base)):
            subdir = os.path.join(base, sub)
            if not os.path.isdir(subdir) or sub == "raw_figures":
                continue
            meta = _load_meta(subdir)
            for fp in sorted(glob.glob(os.path.join(subdir, "*.csv"))):
                name = os.path.basename(fp)
                obs = "wp" if name.startswith("wp") else "ds"
                arr, _, colnames = _read_csv(fp)
                if arr.size == 0:
                    print(f"[skip] {folder}/{sub}/{name}: no data rows")
                    continue
                cols = (colnames or list(meta.get("columns", {}).keys()) or
                        ["rp_hMpc", obs, obs + "_err"])[:arr.shape[1]]
                payload = {
                    "reference": _ref_from_meta(meta, ref, meta.get("data_source")),
                    "provenance": {
                        "type": "observed",
                        "origin": f"repo data/{folder}/{sub}/{name}",
                        "extraction_method": meta.get("extraction_method"),
                        "needs_operator_extraction": False},
                    "sample": {k: meta.get(k) for k in
                               ("survey", "z_eff", "pi_max_hMpc", "cosmology")
                               if k in meta},
                    "units": meta.get("columns", {}),
                    "data": {c: arr[:, i] for i, c in enumerate(cols)},
                    "notes": meta.get("notes"),
                }
                idx.append(write_entry(out_root, "optical", "galaxies", ref,
                                       obs, payload, sub))
    return idx


# --------------------------------------------------------------------------
# 2. observed: Comparat+2025 X-ray cross-correlations (broad band + 16 bands)
# --------------------------------------------------------------------------

def ingest_xray(out_root):
    idx = []
    base = os.path.join(_REPO, "hod_mod", "data", "benchmarks", "xray")
    for fp in sorted(glob.glob(os.path.join(base, "comparat2025_wtheta_S*.csv"))):
        sample = os.path.basename(fp).split("_")[-1].replace(".csv", "")
        arr, header, _cn = _read_csv(fp)
        payload = {
            "reference": _ref("Comparat2025", "galaxy x soft X-ray w(theta), "
                                              "0.5-2 keV broad band"),
            "provenance": {"type": "observed",
                           "origin": f"hod_mod/data/benchmarks/xray/"
                                     f"{os.path.basename(fp)}",
                           "needs_operator_extraction": False},
            "sample": {"description": header[1] if len(header) > 1 else None,
                       "survey": "eROSITA eRASS x Legacy Survey DR10"},
            "units": {"theta_rad": "rad", "theta_deg": "deg", "wtheta": "-",
                      "wtheta_err": "-", "R_kpc": "kpc"},
            "data": {c: arr[:, i] for i, c in enumerate(
                ["theta_rad", "theta_deg", "wtheta", "wtheta_err", "R_kpc"])},
        }
        idx.append(write_entry(out_root, "xray", "galaxies", "Comparat2025",
                               "wtheta_broad", payload, sample))
    # 16-band w(theta) per volume-limited sample (per-field backup, validated)
    try:
        from astropy.io import fits
    except Exception:
        print("[skip] astropy unavailable — band FITS not ingested")
        return idx
    for sdir in sorted(glob.glob(os.path.join(
            _REPO, "hod_mod", "data", "xray_bands", "LS10_VLIM_*"))):
        sample = os.path.basename(sdir)
        bands = []
        theta_mid = None
        for fp in sorted(glob.glob(os.path.join(sdir, "*_E_*.fits"))):
            b = os.path.basename(fp).replace(".fits", "")
            elo, ehi = (int(x) / 1000.0 for x in b.split("_E_"))
            with fits.open(fp) as h:
                t = h[1].data
                theta_mid = np.asarray(t["theta_mid"], dtype=float)
                bands.append({"e_min_keV": elo, "e_max_keV": ehi,
                              "wtheta": np.asarray(t["wtheta"], dtype=float),
                              "wtheta_err": np.asarray(t["wtheta_err"],
                                                       dtype=float)})
        if theta_mid is None:
            continue
        payload = {
            "reference": _ref("Comparat2025", "energy-band w(theta), 100 eV "
                                              "bands over 0.5-2 keV"),
            "provenance": {"type": "observed",
                           "origin": f"hod_mod/data/xray_bands/{sample}",
                           "extraction_method": "reconstructed from per-field "
                                                "backups, validated against "
                                                "the Zenodo release",
                           "needs_operator_extraction": False},
            "sample": {"id": sample,
                       "survey": "eROSITA eRASS x Legacy Survey DR10"},
            "units": {"theta_mid": "rad", "wtheta": "-", "wtheta_err": "-"},
            "data": {"theta_mid": theta_mid, "bands": bands},
        }
        idx.append(write_entry(out_root, "xray", "galaxies", "Comparat2025",
                               "wtheta_bands", payload, sample))
    return idx


# --------------------------------------------------------------------------
# 3. observed_derived_fit: published fitting functions
# --------------------------------------------------------------------------

def ingest_derived(out_root):
    idx = []
    # MD14 Eq. 15 (Salpeter IMF): psi(z) [Msun/yr/Mpc^3]
    z = np.round(np.arange(0.1, 2.01, 0.2), 2)
    psi = 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)
    payload = {
        "reference": _ref("MadauDickinson2014", "Eq. 15 best-fit"),
        "provenance": {"type": "observed_derived_fit",
                       "origin": "psi(z)=0.015(1+z)^2.7/[1+((1+z)/2.9)^5.6], "
                                 "Salpeter IMF",
                       "needs_operator_extraction": True,
                       "extraction_hint": "binned compilation points: "
                                          "MD14 Table 1 / Fig. 9"},
        "sample": {"imf": "Salpeter",
                   "note": "divide by 1.64 for Chabrier"},
        "units": {"z": "-", "sfrd": "Msun yr^-1 Mpc^-3",
                  "sfrd_err": "Msun yr^-1 Mpc^-3"},
        "data": {"z": z, "sfrd": psi, "sfrd_err": 0.12 * psi},
        "notes": "12% per-shell error matches the tier-3 sfrd noise premise; "
                 "the fit itself is the published MD14 result.",
    }
    idx.append(write_entry(out_root, "multiwavelength", "galaxies",
                           "MadauDickinson2014", "sfrd", payload))
    # ALFALFA HIMF Schechter (Jones+2018): log10 M* = 9.94, alpha = -1.25,
    # phi* = 4.5e-3 Mpc^-3 dex^-1 (h=0.70 units)
    lgm = np.arange(8.0, 10.81, 0.2)
    mstar, alpha, phistar = 9.94, -1.25, 4.5e-3
    x = 10 ** (lgm - mstar)
    phi = np.log(10) * phistar * x ** (alpha + 1) * np.exp(-x)
    payload = {
        "reference": _ref("Jones2018ALFALFA", "Schechter fit, final ALFALFA"),
        "provenance": {"type": "observed_derived_fit",
                       "origin": "Schechter(log10M*=9.94+-0.01+-0.05, "
                                 "alpha=-1.25+-0.02+-0.10, "
                                 "phi*=(4.5+-0.2+-0.8)e-3)",
                       "needs_operator_extraction": True,
                       "extraction_hint": "binned HIMF points from the paper "
                                          "(1/Veff estimator)"},
        "sample": {"survey": "ALFALFA (final)", "z_eff": 0.0,
                   "h_convention": "h70; convert to model h before fitting"},
        "units": {"log10_MHI": "log10 Msun (h70^-2)",
                  "phi": "Mpc^-3 dex^-1 (h70^3)", "phi_err": "same"},
        "data": {"log10_MHI": lgm, "phi": phi, "phi_err": 0.05 * phi},
        "notes": "phi_err = 5% statistical placeholder from the phi* "
                 "fractional error; systematic ~18% dominates.",
    }
    idx.append(write_entry(out_root, "radio", "hi", "Jones2018ALFALFA",
                           "himf", payload))
    return idx


# --------------------------------------------------------------------------
# 4. simulated: forward-model fiducial + forecast noise from the npz
# --------------------------------------------------------------------------
# observable -> (npz tag, wavelength, tracer, ref for the missing data,
#                extraction hint)
SIM_SPECS = {
    "n_gal": ("t2", "optical", "galaxies", "Weaver2023",
              "COSMOS2020 SMFs (total + quiescent); GAMA DR4 [Driver2022] "
              "for the local anchor"),
    "ssfr": ("t2", "multiwavelength", "galaxies", "Popesso2023",
             "MS fit and scatter vs (M*, z) from the compilation"),
    "cl_gy": ("t2", "microwave", "galaxies", "Amodeo2021",
              "ACT x CMASS tSZ profiles (data on LAMBDA); kSZ in "
              "[Schaan2021]"),
    "cl_kk": ("t2", "optical", "lensing", "Wright2025",
              "KiDS-Legacy band powers (public data release)"),
    "cl_kCMB": ("t2", "microwave", "cmb_lensing", "Qu2024",
                "ACT DR6 lensing bandpowers (public)"),
    "cl_gkCMB": ("t2", "microwave", "cmb_lensing", "Kim2024",
                 "DESI LRG x ACT DR6 bandpowers (public likelihood)"),
    "cl_shear_kCMB": ("t2", "microwave", "cmb_lensing", "Qu2024",
                      "shear x CMB-lensing cross bandpowers (ACT/SPT x "
                      "DES/KiDS publications)"),
    "cl_XX": ("t2", "xray", "gas", "Comparat2025",
              "no published tomographic soft-band auto-spectrum; "
              "eRASS CXB fluctuation analyses are the closest data"),
    "xlf": ("t2", "xray", "agn", "Aird2015",
            "binned XLF per (L_X, z) from the paper's electronic tables"),
    "wp_agn": ("t2", "xray", "agn", "Comparat2023",
               "eRASS1 AGN clustering data points"),
    "rlf": ("t2", "radio", "agn", "Kondapally2022",
            "LERG LFs per z bin (CDS table); SF radio LF in [Bonato2021]"),
    "ilf": ("t2", "infrared", "agn", "Behroozi2019",
            "operator to select a published mid-IR AGN LF compilation"),
    "oiilf": ("t2", "optical", "galaxies", "Comparat2015OII",
              "[OII] LF compilation tables (author's own)"),
    "cl_gHI": ("t2", "radio", "hi", "CHIME2023",
               "CHIME x eBOSS stacking amplitudes"),
    "uvlf": ("t3", "uv", "galaxies", "Wyder2005",
             "GALEX local UV LF; z>1 from the MD14/UM compilations"),
    "half": ("t3", "optical", "galaxies", "Sobral2013",
             "HiZELS Halpha LF Schechter points per z"),
    "optlf": ("t3", "optical", "galaxies", "Driver2022",
              "GAMA r-band LF"),
    "nirlf": ("t3", "infrared", "galaxies", "Driver2022",
              "GAMA/WISE NIR LFs"),
    "qlf_uv": ("t3", "uv", "agn", "Kulkarni2019",
               "type-1 QLF electronic tables (1450 A)"),
    "qlf_opt": ("t3", "optical", "agn", "Kulkarni2019",
                "type-1 QLF electronic tables"),
    "cl_yy": ("t3", "microwave", "gas", "Amodeo2021",
              "Planck/ACT tSZ auto-spectrum publications"),
    "cl_HIHI": ("t3", "radio", "hi", "CHIMEauto2025",
                "CHIME 21cm auto-power measurement"),
    "ncl": ("t3", "xray", "clusters", "Ghirardini2024",
            "eRASS1 cluster dn/dz (catalogue public at MPE)"),
    "ds_agn": ("t3", "xray", "agn", "Comparat2023",
               "no published AGN-lensing per L_X bin; flag for future data"),
    "cl_gR": ("t3", "radio", "galaxies", "Bonato2021",
              "SKA-era maps required; LoTSS source catalogues meanwhile"),
    "cl_gI": ("t3", "infrared", "galaxies", "Behroozi2019",
              "WISE W1/W2/W3 maps x galaxy samples — measurable today"),
}


def _npz_rows(z, obs):
    m = np.asarray([str(o) for o in z["meta_obs"]]) == obs
    return {
        "x": np.asarray(z["meta_x"])[m],
        "y": np.asarray(z["d0"])[m],
        "y_err": np.asarray(z["sigma_noise"])[m],
        "block": [str(b) for b in np.asarray(z["meta_block"])[m]],
        "zeff": np.asarray(z["meta_zeff"])[m],
    }


def ingest_simulated(out_root):
    idx = []
    stores = {}
    for tag, path in (("t2", _T2_NPZ), ("t3", _T3_NPZ)):
        if os.path.exists(path):
            stores[tag] = np.load(path)
        else:
            print(f"[warn] {path} missing — {tag} simulated entries skipped")
    src_note = {"t2": "tier-2 production npz (90 params, full grid)",
                "t3": "tier-3 SMOKE npz (2x2-cell reduced grid — replace "
                      "with the tier-3 production npz when available)"}
    for obs, (tag, wl, tr, ref, hint) in sorted(SIM_SPECS.items()):
        if tag not in stores:
            continue
        rows = _npz_rows(stores[tag], obs)
        if rows["y"].size == 0:
            print(f"[skip] {obs}: no rows in {tag} npz")
            continue
        payload = {
            "reference": _ref(ref, "reference for the MISSING observed data"),
            "provenance": {
                "type": "simulated",
                "origin": f"forward-model fiducial + forecast survey noise, "
                          f"{src_note[tag]}",
                "needs_operator_extraction": True,
                "extraction_hint": hint},
            "sample": {"note": "rows labelled by forecast block "
                               "(cell/shell id) and z_eff"},
            "units": {"x": "observable-native (r_p [Mpc/h], ell, log10 L, "
                           "log10 M, or z depending on the observable)",
                      "y": "model-native prediction",
                      "y_err": "forecast survey noise (NOT current-survey "
                               "errors); null = masked/incomplete row"},
            "data": rows,
        }
        idx.append(write_entry(out_root, wl, tr, ref, obs, payload,
                               "simulated"))
    return idx


# --------------------------------------------------------------------------
# 5. placeholders: flagged for operator extraction, no local stand-in
# --------------------------------------------------------------------------
PLACEHOLDERS = [
    ("optical", "galaxies", "Moustakas2013", "quenched_fraction",
     "PRIMUS quenched fractions vs (M*, z), 0<z<1",
     "paper tables; also derivable from the n_gal SF/Q simulated entries"),
    ("optical", "galaxies", "Behroozi2019", "um_compilation",
     "the full UniverseMachine observational compilation "
     "(SMF/CSFR/SSFR/QF/UVLF/wp + environmental quenching)",
     "public at the UniverseMachine data release "
     "(https://bitbucket.org/pbehroozi/universemachine, obs/ directory)"),
    ("uv", "galaxies", "Finkelstein2015", "uvlf_z4_8",
     "UV LFs at z=4-8 — OUTSIDE the current model grid (z<2)",
     "paper tables; requires extending the model grid in z"),
    ("uv", "galaxies", "Harikane2023", "uvlf_z9_16",
     "JWST UV LFs at z=9-16 — outside the current model grid",
     "paper tables"),
    ("uv", "galaxies", "Song2016", "uv_mstar",
     "UV-M* relations z=4-8 — outside the current model grid",
     "paper tables"),
    ("optical", "blackholes", "KormendyHo2013", "mbh_census",
     "local M_BH-M_bulge relation (external pin for agn_mu_bh)",
     "Table 3 of the review; low-mass extension in [Greene2020]"),
    ("xray", "agn", "Ananna2022", "erdf_bhmf",
     "BASS DR2 ERDF + black-hole mass function (type 1/2)",
     "paper electronic tables"),
    ("xray", "agn", "Powell2022", "local_agn_clustering",
     "Swift/BAT local AGN clustering (the model's own validation target)",
     "paper tables / BASS DR2 products"),
    ("xray", "clusters", "Ghirardini2024", "cluster_cosmology",
     "eRASS1 cluster abundance data products",
     "catalogue + likelihood public at erosita.mpe.mpg.de/dr1"),
    ("microwave", "galaxies", "Schaan2021", "ksz_profiles",
     "ACT x CMASS kSZ profiles (gas-density observable, model extension)",
     "data products on LAMBDA / ACT DR5 release"),
]


def ingest_placeholders(out_root):
    idx = []
    for wl, tr, ref, obs, desc, hint in PLACEHOLDERS:
        payload = {
            "reference": _ref(ref),
            "provenance": {"type": "placeholder",
                           "origin": "no local data; extraction required",
                           "needs_operator_extraction": True,
                           "extraction_hint": hint},
            "description": desc,
            "data": None,
        }
        idx.append(write_entry(out_root, wl, tr, ref, obs, payload))
    return idx


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/home/comparat/data/benchmark_observables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    idx = []
    idx += ingest_csv_sets(args.out)
    idx += ingest_multi_sample(args.out)
    idx += ingest_xray(args.out)
    idx += ingest_derived(args.out)
    idx += ingest_simulated(args.out)
    idx += ingest_placeholders(args.out)
    summary = {}
    for rel in idx:
        body = json.load(open(os.path.join(args.out, rel)))
        summary[rel] = {
            "observable": body["observable"],
            "wavelength": body["wavelength"],
            "tracer": body["tracer"],
            "reference": body["reference"]["key"],
            "provenance": body["provenance"]["type"],
            "needs_operator_extraction":
                body["provenance"].get("needs_operator_extraction", False),
        }
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    n_obs = sum(1 for v in summary.values() if v["provenance"] == "observed")
    n_sim = sum(1 for v in summary.values() if v["provenance"] == "simulated")
    n_fit = sum(1 for v in summary.values()
                if v["provenance"] == "observed_derived_fit")
    n_pl = sum(1 for v in summary.values() if v["provenance"] == "placeholder")
    n_flag = sum(1 for v in summary.values()
                 if v["needs_operator_extraction"])
    print(f"[done] {len(summary)} files: {n_obs} observed, {n_fit} derived, "
          f"{n_sim} simulated, {n_pl} placeholders; "
          f"{n_flag} flagged for operator extraction -> {args.out}")


if __name__ == "__main__":
    main()
