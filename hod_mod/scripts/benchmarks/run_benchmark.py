#!/usr/bin/env python
"""
Run a single HOD literature benchmark.

Usage
-----
    python hod_mod/scripts/benchmarks/run_benchmark.py --model more2015_logM11_12 [--mcmc] [--plot]

Each benchmark loads its config from configs/benchmarks/benchmark_{model}.yml,
fits the HOD model to the paper's published data vector using the paper's
cosmology, and compares the best-fit parameters against the published values.

A JSON result file is written to the configured output directory.
Pass --mcmc to run emcee sampling after the MAP fit (off by default).
"""

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Registry: model key → (config path, published params dict, published errors)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

BENCHMARK_REGISTRY = {
    # -----------------------------------------------------------------------
    # More+2015 individual stellar-mass subsamples (Fig 3)
    # wp+ESD jointly fit; free cosmology in paper; here fixed to published MAP.
    # sigma_logm = sqrt(sigma^2); errors propagated: err = err_sigma2 / (2*sigma_logm)
    # -----------------------------------------------------------------------
    "more2015_logM11_12": {
        "config": "configs/benchmarks/benchmark_more2015_logM11_12.yml",
        "label": "More+2015 BOSS CMASS logM*>11.1 joint wp+ESD (MoreHODModel)",
        "published_params": {
            "log10mmin": (13.13, 0.13),
            "sigma_logm": (0.469, 0.13),
            "log10m1":    (14.21, 0.13),
            "alpha":      (1.13,  0.15),
            "kappa":      (1.25,  0.45),
        },
        "published_chi2_ndof": 0.8,
        "data_status": "ready",
    },
    # Full cosmological analysis: Omega_m + S8 free, h with WMAP9 prior (8 free params)
    "more2015_logM11_12_freecosmo": {
        "config": "configs/benchmarks/benchmark_more2015_logM11_12_freecosmo.yml",
        "label": "More+2015 BOSS CMASS logM*>11.1 joint wp+ESD FREE COSMO (MoreHODModel)",
        "published_params": {
            "log10mmin": (13.13, 0.13),
            "sigma_logm": (0.469, 0.13),
            "log10m1":    (14.21, 0.13),
            "alpha":      (1.13,  0.15),
            "kappa":      (1.25,  0.45),
            "Omega_m":    (0.310, 0.020),
            "S8":         (0.798, 0.044),
        },
        "published_chi2_ndof": 0.8,
        "data_status": "ready",
    },
    "more2015_logM11p3_12": {
        "config": "configs/benchmarks/benchmark_more2015_logM11p3_12.yml",
        "label": "More+2015 BOSS CMASS logM*>11.3 joint wp+ESD (MoreHODModel)",
        "published_params": {
            "log10mmin": (13.45, 0.15),
            "sigma_logm": (0.671, 0.19),
            "log10m1":    (14.51, 0.17),
            "alpha":      (1.14,  0.49),
        },
        "published_chi2_ndof": 1.3,
        "data_status": "ready",
    },
    "more2015_logM11p4_12": {
        "config": "configs/benchmarks/benchmark_more2015_logM11p4_12.yml",
        "label": "More+2015 BOSS CMASS logM*>11.4 joint wp+ESD (MoreHODModel)",
        "published_params": {
            "log10mmin": (13.68, 0.16),
            "sigma_logm": (0.889, 0.22),
            "log10m1":    (14.56, 0.25),
            "alpha":      (1.00,  0.44),
        },
        "published_chi2_ndof": 1.5,
        "data_status": "ready",
    },
    "kravtsov2004": {
        "config": "configs/benchmarks/benchmark_kravtsov2004.yml",
        "label": "Kravtsov+2004 (Kravtsov04HODModel, BOSS CMASS data)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zheng2007": {
        "config": "configs/benchmarks/benchmark_zheng2007.yml",
        "label": "Zheng+2007 SDSS M_r < -21 (HODModel)",
        "published_params": {
            "log10mmin":  (12.78, 0.10),
            "sigma_logm": (0.68, 0.15),
            "log10m0":    (11.92, 0.30),
            "log10m1":    (13.88, 0.08),
            "alpha":      (1.39, 0.15),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "leauthaud2012": {
        "config": "configs/benchmarks/benchmark_leauthaud2012.yml",
        "label": "Leauthaud+2012 COSMOS z2=[0.48,0.74] (Leauthaud12HODModel)",
        "published_params": {
            "log10m1":      (12.725, 0.032),
            "log10m_star0": (11.038, 0.019),
            "beta":         (0.466, 0.009),
            "delta":        (0.61, 0.13),
            "gamma":        (1.95, 0.25),
            "sigma_logm":   (0.249, 0.019),
        },
        "published_chi2_ndof": 1.6,
        "data_status": "NEEDS_DATA",
    },
    "vanutert2016": {
        "config": "configs/benchmarks/benchmark_vanutert2016.yml",
        "label": "van Uitert+2016 GAMA bin 2 (VanUitert16CSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "NEEDS_DATA",
    },
    "zumandelbaum2015": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015.yml",
        "label": "Zu & Mandelbaum 2015 SDSS (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h":        (12.10, 0.17),
            "lg_m0star":     (10.31, 0.10),
            "beta":          (0.33, 0.21),
            "delta":         (0.42, 0.04),
            "gamma":         (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04),
            "eta":           (-0.04, 0.02),
            "fc":            (0.86, 0.14),
            "bsat":          (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    # -----------------------------------------------------------------------
    # Zu & Mandelbaum 2015 — per-bin iHOD fits (real digitized data from Fig. 6)
    # Each bin uses log10m_star_thresh=lo, log10m_star_max=hi → bin HOD.
    # published_params = global iHOD Table 2 values (shared reference for all bins).
    # -----------------------------------------------------------------------
    "zumandelbaum2015_bin_9p4_9p8": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_bin_9p4_9p8.yml",
        "label": "ZM15 SDSS bin [9.4-9.8] wp-only (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h": (12.10, 0.17), "lg_m0star": (10.31, 0.10),
            "beta": (0.33, 0.21), "delta": (0.42, 0.04), "gamma": (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04), "eta": (-0.04, 0.02),
            "fc": (0.86, 0.14), "bsat": (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_bin_9p8_10p2": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_bin_9p8_10p2.yml",
        "label": "ZM15 SDSS bin [9.8-10.2] wp-only (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h": (12.10, 0.17), "lg_m0star": (10.31, 0.10),
            "beta": (0.33, 0.21), "delta": (0.42, 0.04), "gamma": (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04), "eta": (-0.04, 0.02),
            "fc": (0.86, 0.14), "bsat": (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_bin_10p2_10p6": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_bin_10p2_10p6.yml",
        "label": "ZM15 SDSS bin [10.2-10.6] joint wp+ESD (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h": (12.10, 0.17), "lg_m0star": (10.31, 0.10),
            "beta": (0.33, 0.21), "delta": (0.42, 0.04), "gamma": (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04), "eta": (-0.04, 0.02),
            "fc": (0.86, 0.14), "bsat": (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_bin_10p6_11p0": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_bin_10p6_11p0.yml",
        "label": "ZM15 SDSS bin [10.6-11.0] joint wp+ESD (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h": (12.10, 0.17), "lg_m0star": (10.31, 0.10),
            "beta": (0.33, 0.21), "delta": (0.42, 0.04), "gamma": (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04), "eta": (-0.04, 0.02),
            "fc": (0.86, 0.14), "bsat": (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_bin_11p0_11p2": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p0_11p2.yml",
        "label": "ZM15 SDSS bin [11.0-11.2] joint wp+ESD (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h": (12.10, 0.17), "lg_m0star": (10.31, 0.10),
            "beta": (0.33, 0.21), "delta": (0.42, 0.04), "gamma": (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04), "eta": (-0.04, 0.02),
            "fc": (0.86, 0.14), "bsat": (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_bin_11p2_11p4": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p2_11p4.yml",
        "label": "ZM15 SDSS bin [11.2-11.4] joint wp+ESD (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h": (12.10, 0.17), "lg_m0star": (10.31, 0.10),
            "beta": (0.33, 0.21), "delta": (0.42, 0.04), "gamma": (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04), "eta": (-0.04, 0.02),
            "fc": (0.86, 0.14), "bsat": (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_bin_11p4_12p0": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_bin_11p4_12p0.yml",
        "label": "ZM15 SDSS bin [11.4-12.0] joint wp+ESD (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h": (12.10, 0.17), "lg_m0star": (10.31, 0.10),
            "beta": (0.33, 0.21), "delta": (0.42, 0.04), "gamma": (1.21, 0.20),
            "sigma_lnmstar": (0.50, 0.04), "eta": (-0.04, 0.02),
            "fc": (0.86, 0.14), "bsat": (8.98, 1.18),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "guo2018": {
        "config": "configs/benchmarks/benchmark_guo2018.yml",
        "label": "Guo+2018 SDSS LOWZ (Guo18ICSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "guo2019": {
        "config": "configs/benchmarks/benchmark_guo2019.yml",
        "label": "Guo+2019 eBOSS ELG (Guo19ICSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zacharegkas2025": {
        "config": "configs/benchmarks/benchmark_zacharegkas2025.yml",
        "label": "Zacharegkas+2025 DES Y3 (Zacharegkas25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "NEEDS_DATA",
    },
    # -----------------------------------------------------------------------
    # Second benchmark tier: ΔΣ-only fits
    # -----------------------------------------------------------------------
    "leauthaud2012_ds": {
        "config": "configs/benchmarks/benchmark_leauthaud2012_ds.yml",
        "label": "Leauthaud+2012 COSMOS z2 — ΔΣ only (Leauthaud12HODModel)",
        "published_params": {
            "log10m1":      (12.725, 0.032),
            "log10m_star0": (11.038, 0.019),
            "beta":         (0.466, 0.009),
            "delta":        (0.61, 0.13),
            "gamma":        (1.95, 0.25),
            "sigma_logm":   (0.249, 0.019),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "vanutert2016_ds": {
        "config": "configs/benchmarks/benchmark_vanutert2016_ds.yml",
        "label": "van Uitert+2016 GAMA+KiDS bin M3 — ΔΣ only (VanUitert16CSMFModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zumandelbaum2015_ds": {
        "config": "configs/benchmarks/benchmark_zumandelbaum2015_ds.yml",
        "label": "Zu & Mandelbaum 2015 SDSS — ΔΣ only (ZuMandelbaum15HODModel)",
        "published_params": {
            "lg_m1h":        (12.10, 0.10),
            "lg_m0star":     (10.31, 0.05),
            "beta":          (0.33, 0.05),
            "delta":         (0.42, 0.05),
            "gamma":         (1.21, 0.10),
            "sigma_lnmstar": (0.50, 0.05),
            "eta":           (-0.04, 0.02),
            "fc":            (0.86, 0.05),
            "bsat":          (8.98, 1.00),
        },
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "zacharegkas2025_ds": {
        "config": "configs/benchmarks/benchmark_zacharegkas2025_ds.yml",
        "label": "Zacharegkas+2025 DES Y3 bin 1 — ΔΣ only (Zacharegkas25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "NOT_APPLICABLE",
    },
    # -----------------------------------------------------------------------
    # DESI DR1 tracer benchmarks — Lange+2025 (arXiv:2512.15962)
    # Decorated HOD with effective assembly bias; free Omega_m + S8
    # Data: digitized from ar5iv PNG (Fig 3/4), ~20-30% accuracy; Zenodo 17831718 when published
    # -----------------------------------------------------------------------
    "lange2025_bgs2_des": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_des.yml",
        "label": "Lange+2025 DESI BGS2 joint wp+ESD — DES/KiDS (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs3_des": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs3_des.yml",
        "label": "Lange+2025 DESI BGS3 joint wp+ESD — DES/KiDS (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg1_des": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg1_des.yml",
        "label": "Lange+2025 DESI LRG1 joint wp+ESD — DES/KiDS (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_hsc.yml",
        "label": "Lange+2025 DESI BGS2 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs3_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs3_hsc.yml",
        "label": "Lange+2025 DESI BGS3 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg1_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg1_hsc.yml",
        "label": "Lange+2025 DESI LRG1 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg2_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg2_hsc.yml",
        "label": "Lange+2025 DESI LRG2 joint wp+ESD — HSC-Y3 (Lange25HODModel)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_wp": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_wp.yml",
        "label": "Lange+2025 DESI BGS2 wp-only with free cosmo (Lange25HODModel)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_ds_des": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_ds_des.yml",
        "label": "Lange+2025 DESI BGS2 ESD-only — DES/KiDS (Lange25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_ds_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_ds_hsc.yml",
        "label": "Lange+2025 DESI BGS2 ESD-only — HSC-Y3 (Lange25HODModel)",
        "published_params": {},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    # --- manually digitized (WebPlotDigitizer) bwpd data ---
    "lange2025_bgs2_bwpd_wp": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_bwpd_wp.yml",
        "label": "Lange+2025 BGS2 wp-only — manually digitized (bwpd)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_bwpd_esd": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_bwpd_esd.yml",
        "label": "Lange+2025 BGS2 ESD-only (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs2_bwpd_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs2_bwpd_hsc.yml",
        "label": "Lange+2025 BGS2 joint wp+ESD (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs3_bwpd_wp": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs3_bwpd_wp.yml",
        "label": "Lange+2025 BGS3 wp-only — manually digitized (bwpd)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs3_bwpd_esd": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs3_bwpd_esd.yml",
        "label": "Lange+2025 BGS3 ESD-only (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_bgs3_bwpd_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_bgs3_bwpd_hsc.yml",
        "label": "Lange+2025 BGS3 joint wp+ESD (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg1_bwpd_wp": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg1_bwpd_wp.yml",
        "label": "Lange+2025 LRG1 wp-only — manually digitized (bwpd)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg1_bwpd_esd": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg1_bwpd_esd.yml",
        "label": "Lange+2025 LRG1 ESD-only (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg1_bwpd_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg1_bwpd_hsc.yml",
        "label": "Lange+2025 LRG1 joint wp+ESD (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg2_bwpd_wp": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg2_bwpd_wp.yml",
        "label": "Lange+2025 LRG2 wp-only — manually digitized (bwpd)",
        "published_params": {"S8": 0.794, "Omega_m": 0.295},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg2_bwpd_esd": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg2_bwpd_esd.yml",
        "label": "Lange+2025 LRG2 ESD-only (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
    "lange2025_lrg2_bwpd_hsc": {
        "config": "configs/benchmarks/benchmark_lange2025_lrg2_bwpd_hsc.yml",
        "label": "Lange+2025 LRG2 joint wp+ESD (HSC-Y3) — manually digitized (bwpd)",
        "published_params": {"S8": 0.793, "Omega_m": 0.303},
        "published_chi2_ndof": None,
        "data_status": "ready",
    },
}


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model_key: str, mcmc: bool = False, plot: bool = False,
                  output_dir: str | None = None,
                  force_mcmc: bool = False,
                  plot_only: bool = False) -> dict:
    """Run one benchmark and return the result dict.

    Parameters
    ----------
    model_key : str
        Key from BENCHMARK_REGISTRY (e.g. ``"more2015"``).
    mcmc : bool
        If True, run emcee sampling after MAP.
    plot : bool
        If True, save comparison figures.
    output_dir : str or None
        Override the output directory from the config.
    force_mcmc : bool
        If True, rerun MCMC even if flatchain.npz already exists.
    """
    from hod_mod.fitting import load_config, WpFitter, JointFitter, DeltaSigmaFitter

    entry = BENCHMARK_REGISTRY[model_key]
    label = entry["label"]
    published = entry["published_params"]
    pub_chi2 = entry["published_chi2_ndof"]
    status = entry["data_status"]

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"{'='*60}")

    if status not in ("ready",):
        print(f"SKIP — data not yet available (status={status})")
        # derive a directory guess for the README path
        key_base = model_key.replace("_ds", "").replace("_", "")
        print(f"  See data/{key_base}*/README_data.md for data extraction instructions.")
        return {"model": model_key, "status": "skipped", "reason": status}

    config_path = os.path.join(_REPO_ROOT, entry["config"])
    config = load_config(config_path)
    if output_dir is not None:
        config = _override_output(config, output_dir)

    os.makedirs(config.output_dir, exist_ok=True)

    # Choose fitter based on available data:
    #   ds_file + data_file → JointFitter  (third benchmark: wp + ΔΣ)
    #   ds_file only        → DeltaSigmaFitter (second benchmark: ΔΣ only)
    #   data_file only      → WpFitter     (first benchmark: wp only)
    has_wp = bool(config.data_file and os.path.isfile(config.data_file))
    has_ds = config.ds_file is not None
    if has_wp and has_ds:
        fitter = JointFitter(config)
        joint = True
    elif has_ds:
        fitter = DeltaSigmaFitter(config)
        joint = False
    else:
        fitter = WpFitter(config)
        joint = False

    # --plot-only: skip MAP, reload params from saved result
    if plot_only:
        result_file = os.path.join(config.output_dir, "benchmark_result.json")
        if not os.path.exists(result_file):
            print(f"ERROR: {result_file} not found — run without --plot-only first.")
            return {"model": model_key, "status": "error", "reason": "no saved result"}
        with open(result_file) as fh:
            saved = json.load(fh)
        params    = saved["params"]
        chi2_ndof = saved["chi2_ndof"]
        print(f"  Loaded params from {result_file} (χ²/dof={chi2_ndof:.3f})")
        ds_only = has_ds and not has_wp
        _make_plots(fitter, params, chi2_ndof, model_key, config.output_dir,
                    joint=joint, ds_only=ds_only)
        return saved

    # MAP fit
    map_result = fitter.map_fit()
    params = map_result["params"]
    chi2 = map_result["chi2"]
    ndof = map_result["ndof"]
    chi2_ndof = chi2 / ndof if ndof > 0 else float("nan")

    # Optional MCMC
    if mcmc:
        flatchain_path = os.path.join(config.output_dir, "flatchain.npz")
        if os.path.exists(flatchain_path) and not force_mcmc:
            print(f"MCMC already done — skipping ({flatchain_path}).")
            print("  Pass --force-mcmc to rerun.")
        else:
            print("Running MCMC (this may take several minutes)…")
            theta0 = np.asarray(map_result["theta"])
            n_walkers = fitter.config.n_walkers
            scale = np.maximum(np.abs(theta0) * 1e-3, 1e-4)
            rng = np.random.default_rng(42)
            initial_pos = theta0[None, :] + rng.normal(0, scale, (n_walkers, len(theta0)))
            fitter.sample(initial_pos=initial_pos)

    # Print comparison table
    _print_comparison(params, published, chi2_ndof, pub_chi2)

    # Optional plots
    ds_only = has_ds and not has_wp
    if plot:
        _make_plots(fitter, params, chi2_ndof, model_key, config.output_dir,
                    joint=joint, ds_only=ds_only)

    # Build result dict
    passes = chi2_ndof < 2.0 and not np.isnan(chi2_ndof)
    result = {
        "model": model_key,
        "label": label,
        "status": "pass" if passes else "fail",
        "chi2": float(chi2),
        "ndof": int(ndof),
        "chi2_ndof": float(chi2_ndof),
        "published_chi2_ndof": pub_chi2,
        "params": {k: float(v) for k, v in params.items()},
        "published_params": {k: list(v) for k, v in _normalize_published(published).items()},
        "param_deviations_sigma": _deviations(params, published),
    }

    out_file = os.path.join(config.output_dir, "benchmark_result.json")
    with open(out_file, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResult saved: {out_file}")
    print(f"Benchmark: {'PASSED' if passes else 'FAILED'}")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _override_output(config, output_dir):
    from dataclasses import replace
    return replace(config, output_dir=output_dir)


def _normalize_published(published: dict) -> dict:
    """Ensure every entry is a (value, error) tuple; plain floats get error=0."""
    return {
        k: v if isinstance(v, tuple) else (v, 0.0)
        for k, v in published.items()
    }


def _deviations(params: dict, published: dict) -> dict:
    published = _normalize_published(published)
    devs = {}
    for pname, (pub_val, pub_err) in published.items():
        bfit = params.get(pname, float("nan"))
        devs[pname] = float((bfit - pub_val) / pub_err) if pub_err > 0 else float("nan")
    return devs


def _print_comparison(params, published, chi2_ndof, pub_chi2):
    published = _normalize_published(published)
    print(f"\nchi2/ndof = {chi2_ndof:.3f}", end="")
    if pub_chi2 is not None:
        print(f"  (published: {pub_chi2:.2f})")
    else:
        print()
    if published:
        w = max(len(k) for k in published) + 2
        print(f"\n{'Parameter':{w}s}  {'Best-fit':>10s}  {'Published':>10s}  {'Δ/σ':>8s}")
        print("-" * (w + 34))
        for pname, (pub_val, pub_err) in published.items():
            bfit = params.get(pname, float("nan"))
            diff = (bfit - pub_val) / pub_err if pub_err > 0 else float("nan")
            diff_str = f"{diff:8.2f}σ" if not (diff != diff) else "      n/a"
            print(f"{pname:{w}s}  {bfit:10.4f}  {pub_val:10.4f}  {diff_str}")


# ─── Shared plotting helpers (single source of truth in benchmark_plots.py) ──
from hod_mod.scripts.benchmarks.benchmark_plots import (
    _COL_DATA, _COL_MAP, _COL_PUB,
    _PARAM_LATEX,
    load_flatchain  as _load_flatchain,
    mcmc_bands      as _mcmc_bands,
    add_bands       as _add_bands,
    residual_panel  as _residual_panel,
    plot_hod        as _plot_hod,
    plot_corner     as _plot_corner_fn,
)


def _plot_corner(flatchain, param_names, published, model_key, output_dir,
                 fixed_params=None):
    """Delegates to benchmark_plots.plot_corner (getdist → corner → matplotlib)."""
    _plot_corner_fn(flatchain, param_names, published, model_key, output_dir,
                    normalize_fn=_normalize_published, fixed_params=fixed_params)


def _make_plots(fitter, params, chi2_ndof, model_key, output_dir, joint, ds_only=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    entry     = BENCHMARK_REGISTRY.get(model_key, {})
    published = entry.get("published_params", {})

    # Build published-parameter dict (override MAP free-params with published values)
    pub_params = None
    if published:
        pub_params = dict(params)
        for pname, pval in _normalize_published(published).items():
            pub_params[pname] = pval[0]

    # Load MCMC flatchain if it exists
    flatchain, fc_names = _load_flatchain(output_dir)

    # Initialise all observable data as None; populated by blocks 1 & 2.
    rp = wp_obs = wp_err = wp_pred = wp_pub = wp_bands = None
    R  = ds_obs = ds_err = ds_pred = ds_pub = ds_bands = None

    # ------------------------------------------------------------------ #
    # 1.  wp figure                                                        #
    # ------------------------------------------------------------------ #
    if not ds_only and hasattr(fitter, "rp_arr"):
        rp      = np.array(fitter.rp_arr)
        wp_obs  = np.array(fitter.wp_obs)
        wp_err  = (np.sqrt(np.diag(np.linalg.inv(fitter.icov_wp)))
                   if hasattr(fitter, "icov_wp") else np.ones_like(wp_obs))
        wp_pred = np.array(fitter.predict_wp(params))
        if pub_params:
            try:
                wp_pub = np.array(fitter.predict_wp(pub_params))
            except Exception:
                pass
        wp_bands = (_mcmc_bands(fitter.predict_wp, params, flatchain, fc_names)
                    if flatchain is not None else None)

        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        _add_bands(axes[0], rp, wp_bands, _COL_MAP)
        axes[0].errorbar(rp, wp_obs, yerr=wp_err, fmt="o", ms=4,
                         color=_COL_DATA, zorder=5, label="Data")
        axes[0].loglog(rp, wp_pred, "-", color=_COL_MAP, lw=1.8,
                       label=f"MAP (χ²/dof={chi2_ndof:.2f})")
        if wp_bands is not None:
            axes[0].plot([], [], "--", color=_COL_MAP, lw=1.5, label="MCMC median ± 68/95%")
        if wp_pub is not None:
            axes[0].loglog(rp, wp_pub, "--", color=_COL_PUB, lw=1.5,
                           label="Published best-fit")
        axes[0].set_ylabel(r"$w_p(r_p)$ [$h^{-1}$ Mpc]")
        axes[0].legend(fontsize=9)
        axes[0].set_title(f"Benchmark: {model_key}", fontsize=10)
        _residual_panel(axes[1], rp, wp_obs, wp_pred, wp_err,
                        pub=wp_pub, bands=wp_bands, fmt="o")
        axes[1].set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"benchmark_{model_key}_wp.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: benchmark_{model_key}_wp.png")

    # ------------------------------------------------------------------ #
    # 2.  ΔΣ figure                                                        #
    # ------------------------------------------------------------------ #
    if (joint or ds_only) and hasattr(fitter, "predict_ds"):
        R       = np.array(fitter.R_arr)
        ds_obs  = np.array(fitter.ds_obs)
        ds_err  = (np.sqrt(np.diag(np.linalg.inv(fitter.icov_ds)))
                   if hasattr(fitter, "icov_ds") else np.ones_like(ds_obs))
        ds_pred = np.array(fitter.predict_ds(params))
        if pub_params:
            try:
                ds_pub = np.array(fitter.predict_ds(pub_params))
            except Exception:
                pass
        ds_bands = (_mcmc_bands(fitter.predict_ds, params, flatchain, fc_names)
                    if flatchain is not None else None)

        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
        _add_bands(axes2[0], R, ds_bands, _COL_MAP)
        axes2[0].errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=4,
                          color=_COL_DATA, zorder=5, label="Data")
        axes2[0].loglog(R, ds_pred, "-", color=_COL_MAP, lw=1.8,
                        label=f"MAP (χ²/dof={chi2_ndof:.2f})")
        if ds_bands is not None:
            axes2[0].plot([], [], "--", color=_COL_MAP, lw=1.5, label="MCMC median ± 68/95%")
        if ds_pub is not None:
            axes2[0].loglog(R, ds_pub, "--", color=_COL_PUB, lw=1.5,
                            label="Published best-fit")
        axes2[0].set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
        axes2[0].legend(fontsize=9)
        title_suffix = " — ΔΣ only" if ds_only else " — ΔΣ"
        axes2[0].set_title(f"Benchmark: {model_key}{title_suffix}", fontsize=10)
        _residual_panel(axes2[1], R, ds_obs, ds_pred, ds_err,
                        pub=ds_pub, bands=ds_bands, fmt="s")
        axes2[1].set_xlabel(r"$R$ [$h^{-1}$ Mpc]")
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, f"benchmark_{model_key}_ds.png"), dpi=150)
        plt.close(fig2)
        print(f"  Saved: benchmark_{model_key}_ds.png")

    # ------------------------------------------------------------------ #
    # 3.  Combined figure — always generated, layout adapts to available  #
    #     observables (1-column for single obs, 2-column for joint)       #
    # ------------------------------------------------------------------ #
    has_wp_data = rp is not None
    has_ds_data = R  is not None

    if has_wp_data or has_ds_data:
        n_cols  = int(has_wp_data) + int(has_ds_data)
        fig_w   = 6 * n_cols + 0.5
        fig3, axes3 = plt.subplots(
            2, n_cols, figsize=(fig_w, 8), sharex="col", squeeze=False,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05,
                         "wspace": 0.3 if n_cols > 1 else 0.0},
        )
        col = 0

        if has_wp_data:
            ax0, ax1 = axes3[0, col], axes3[1, col]
            _add_bands(ax0, rp, wp_bands, _COL_MAP, scale=rp)
            ax0.errorbar(rp, rp * wp_obs, yerr=rp * wp_err, fmt="o", ms=4,
                         color=_COL_DATA, zorder=5, label="Data")
            ax0.loglog(rp, rp * wp_pred, "-", color=_COL_MAP, lw=1.8,
                       label=f"MAP (χ²/dof={chi2_ndof:.2f})")
            if wp_bands is not None:
                ax0.plot([], [], "--", color=_COL_MAP, lw=1.5,
                         label="MCMC median ± 68/95%")
            if wp_pub is not None:
                ax0.loglog(rp, rp * wp_pub, "--", color=_COL_PUB, lw=1.5,
                           label="Published best-fit")
            ax0.set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}{\rm Mpc}^2$]")
            ax0.legend(fontsize=8)
            ax0.set_title(f"{model_key} — $w_p$", fontsize=10)
            _residual_panel(ax1, rp, wp_obs, wp_pred, wp_err,
                            pub=wp_pub, bands=wp_bands, fmt="o")
            ax1.set_xlabel(r"$r_p$ [$h^{-1}$ Mpc]")
            col += 1

        if has_ds_data:
            ax0, ax1 = axes3[0, col], axes3[1, col]
            _add_bands(ax0, R, ds_bands, _COL_MAP)
            ax0.errorbar(R, ds_obs, yerr=ds_err, fmt="s", ms=4,
                         color=_COL_DATA, zorder=5, label="Data")
            ax0.loglog(R, ds_pred, "-", color=_COL_MAP, lw=1.8,
                       label="MAP" if has_wp_data else f"MAP (χ²/dof={chi2_ndof:.2f})")
            if ds_bands is not None:
                ax0.plot([], [], "--", color=_COL_MAP, lw=1.5,
                         label="MCMC median ± 68/95%")
            if ds_pub is not None:
                ax0.loglog(R, ds_pub, "--", color=_COL_PUB, lw=1.5,
                           label="Published best-fit")
            ax0.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot\,h\,{\rm pc}^{-2}$]")
            ax0.legend(fontsize=8)
            ds_title = r"$\Delta\Sigma$" if has_wp_data else f"{model_key} — $\\Delta\\Sigma$"
            ax0.set_title(ds_title, fontsize=10)
            _residual_panel(ax1, R, ds_obs, ds_pred, ds_err,
                            pub=ds_pub, bands=ds_bands, fmt="s")
            ax1.set_xlabel(r"$R$ [$h^{-1}$ Mpc]")

        fig3.savefig(os.path.join(output_dir, f"benchmark_{model_key}_combined.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: benchmark_{model_key}_combined.png")

    # ------------------------------------------------------------------ #
    # 4.  HOD figure                                                       #
    # ------------------------------------------------------------------ #
    _plot_hod(fitter, params, pub_params, model_key, output_dir,
              flatchain=flatchain, param_names=fc_names)

    # ------------------------------------------------------------------ #
    # 5.  Corner plot (only when flatchain is available)                   #
    # ------------------------------------------------------------------ #
    if flatchain is not None:
        # Fixed params = everything in params that was NOT sampled by MCMC
        fixed = {k: float(v) for k, v in params.items() if k not in fc_names}
        _plot_corner(flatchain, fc_names, published, model_key, output_dir,
                     fixed_params=fixed or None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run a single HOD literature benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(BENCHMARK_REGISTRY)}",
    )
    p.add_argument("--model", required=True, choices=list(BENCHMARK_REGISTRY),
                   help="Benchmark identifier")
    p.add_argument("--mcmc", action="store_true",
                   help="Run emcee MCMC after MAP (slow)")
    p.add_argument("--force-mcmc", action="store_true",
                   help="Rerun MCMC even if flatchain.npz already exists")
    p.add_argument("--plot", action="store_true",
                   help="Save comparison figures to output dir")
    p.add_argument("--plot-only", action="store_true",
                   help="Reload saved results and regenerate figures only (skip MAP/MCMC)")
    p.add_argument("--output", default=None,
                   help="Override output directory from config")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_only = getattr(args, "plot_only", False)
    result = run_benchmark(args.model, mcmc=args.mcmc,
                           plot=args.plot or plot_only,
                           output_dir=args.output,
                           force_mcmc=args.force_mcmc,
                           plot_only=plot_only)
    sys.exit(0 if result.get("status") in ("pass", "skipped", "error") else 1)
