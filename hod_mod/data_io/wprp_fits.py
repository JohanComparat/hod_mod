"""Load projected correlation function wp(rp) from FITS files.

Handles both full measurements and jackknife / bootstrap realisations produced
by the Comparat & Macias-Perez+2025 pipeline (eROSITA × Legacy Survey DR10).

FITS column layout (HDU 1):
    rp_min, rp_max, rp_mid  [Mpc, comoving]
    wprp                    [Mpc, dimensionless projected integral]
    N_data, N_random, pimax

All public functions return distances in h-units (Mpc/h) and accept an `h`
argument (default 0.6736, Planck 2018) for the Mpc → Mpc/h conversion:

    rp_h = rp_Mpc * h
    wp_h = wp_Mpc * h

Jackknife covariance estimator (delete-one, N patches)::

    C_ij = (N-1)/N  Σ_k (wp_k^i - <wp>^i)(wp_k^j - <wp>^j)

Bootstrap covariance estimator (M realisations)::

    C_ij = 1/(M-1) Σ_k (wp_k^i - <wp>^i)(wp_k^j - <wp>^j)

For the cross-correlation files some bins are NaN (pair counts below threshold).
``load_jk_wp_cross`` handles NaN values per-bin using only valid realisations.

References
----------
Comparat & Macias-Perez+2025, A&A 700 A271
Zenodo: https://zenodo.org/records/15806800
"""

from __future__ import annotations

import os
import glob
import warnings

import numpy as np


def _read_fits_table(path: str):
    """Open a single FITS HDU-1 binary table and return its data as a numpy recarray."""
    from astropy.io import fits
    with fits.open(path) as hdul:
        data = hdul[1].data.copy()
    return data


# ---------------------------------------------------------------------------
# Single-measurement loader
# ---------------------------------------------------------------------------

def load_wp_full(
    path: str,
    h: float = 0.6736,
    rp_min: float = 0.05,
    rp_max: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single (non-JK) wp(rp) FITS file.

    Parameters
    ----------
    path : str
        Path to the FITS file containing the full-sample measurement.
    h : float
        Dimensionless Hubble constant (H0 / 100 km/s/Mpc) for unit conversion.
    rp_min, rp_max : float [Mpc/h]
        Scale cut applied after conversion.

    Returns
    -------
    rp_h : ndarray [Mpc/h]
    wp_h : ndarray [Mpc/h]
    """
    data = _read_fits_table(path)
    rp_h = data["rp_mid"] * h
    wp_h = data["wprp"] * h
    mask = (rp_h >= rp_min) & (rp_h <= rp_max) & np.isfinite(wp_h)
    return rp_h[mask], wp_h[mask]


# ---------------------------------------------------------------------------
# Galaxy auto-correlation JK loader (delete-one JK)
# ---------------------------------------------------------------------------

def load_jk_wp_auto(
    directory: str,
    pattern: str = "NSIDE_04",
    h: float = 0.6736,
    rp_min: float = 0.1,
    rp_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load galaxy auto-correlation JK realisations and compute delete-one JK covariance.

    Reads all files matching ``*{pattern}_J_*.fits`` in *directory*.

    Parameters
    ----------
    directory : str
        Path containing the JK measurement FITS files and the full measurement.
    pattern : str
        Sub-string identifying the jackknife resolution (e.g. ``'NSIDE_04'``).
    h : float
        Dimensionless Hubble constant for Mpc → Mpc/h conversion.
    rp_min, rp_max : float [Mpc/h]
        Scale cuts applied after unit conversion.

    Returns
    -------
    rp_h : ndarray, shape (Nbins,) [Mpc/h]
        Projected separation bin centres.
    wp_h : ndarray, shape (Nbins,) [Mpc/h]
        wp from the *full* measurement (not the JK mean).
    cov_h : ndarray, shape (Nbins, Nbins) [(Mpc/h)²]
        Jackknife covariance matrix.

    Notes
    -----
    The delete-one JK covariance estimator is::

        C_ij = (N-1)/N  Σ_k (wp_k^i - <wp>^i)(wp_k^j - <wp>^j)

    where N is the number of JK patches.
    """
    jk_files = sorted(glob.glob(os.path.join(directory, f"*{pattern}_J_*.fits")))
    if not jk_files:
        raise FileNotFoundError(
            f"No JK files matching *{pattern}_J_*.fits in {directory}"
        )

    full_files = [f for f in os.listdir(directory) if "_J_" not in f and f.endswith(".fits")]
    if not full_files:
        raise FileNotFoundError(f"No full-measurement FITS file found in {directory}")
    rp_full, wp_full = load_wp_full(
        os.path.join(directory, full_files[0]), h=h, rp_min=rp_min, rp_max=rp_max
    )

    wp_jk = []
    for path in jk_files:
        data = _read_fits_table(path)
        rp_h = data["rp_mid"] * h
        wp_h = data["wprp"] * h
        mask = (rp_h >= rp_min) & (rp_h <= rp_max) & np.isfinite(wp_h)
        wp_jk.append(wp_h[mask])

    wp_jk = np.array(wp_jk)  # (N_jk, Nbins)
    N = len(wp_jk)
    wp_mean = wp_jk.mean(axis=0)
    delta = wp_jk - wp_mean[None, :]
    cov = (N - 1) / N * (delta.T @ delta)
    return rp_full, wp_full, cov


# ---------------------------------------------------------------------------
# Cluster-galaxy cross-correlation JK loader (bootstrap sample covariance)
# ---------------------------------------------------------------------------

def load_jk_wp_cross(
    directory: str,
    galaxy_type: str = "ANY",
    pattern: str = "NSIDE_04",
    h: float = 0.6736,
    rp_min: float = 0.1,
    rp_max: float = 50.0,
    min_valid_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cluster-galaxy cross-correlation bootstrap realisations.

    Reads files matching ``*{galaxy_type}*{pattern}_J_*.fits`` in *directory*
    and computes a sample covariance matrix with per-bin NaN masking.

    Parameters
    ----------
    directory : str
        Directory containing the cross-correlation JK FITS files.
    galaxy_type : str
        Galaxy colour sub-sample to select: ``'ANY'``, ``'BC'``, or ``'RS'``.
    pattern : str
        Jackknife resolution sub-string (e.g. ``'NSIDE_04'``).
    h : float
        Dimensionless Hubble constant for Mpc → Mpc/h conversion.
    rp_min, rp_max : float [Mpc/h]
        Scale cuts applied after unit conversion.
    min_valid_frac : float
        Minimum fraction of non-NaN realisations required to include a bin.

    Returns
    -------
    rp_h : ndarray, shape (Nbins,) [Mpc/h]
        Projected separation bin centres for bins with sufficient coverage.
    wp_h_mean : ndarray, shape (Nbins,) [Mpc/h]
        Mean wp across all valid realisations in each bin.
    cov_h : ndarray, shape (Nbins, Nbins) [(Mpc/h)²]
        Sample covariance matrix.  Off-diagonal entries (i,j) use only
        realisations that are valid in *both* bin i and bin j.

    Notes
    -----
    The bootstrap sample covariance estimator is::

        C_ij = 1/(M-1) Σ_k (wp_k^i - <wp>^i)(wp_k^j - <wp>^j)

    where M is the number of valid (non-NaN) bootstrap realisations per pair
    of bins (i, j).
    """
    search_pattern = f"*{galaxy_type}*{pattern}_J_*.fits"
    jk_files = sorted(glob.glob(os.path.join(directory, search_pattern)))
    if not jk_files:
        raise FileNotFoundError(
            f"No files matching {search_pattern} in {directory}"
        )

    # Load all JK realisations into a (N_jk, N_all_bins) matrix with NaN preserved
    first_data = _read_fits_table(jk_files[0])
    rp_all = first_data["rp_mid"] * h
    n_bins_all = len(rp_all)
    scale_mask = (rp_all >= rp_min) & (rp_all <= rp_max)
    rp_all_cut = rp_all[scale_mask]
    n_bins = scale_mask.sum()

    N = len(jk_files)
    wp_matrix = np.full((N, n_bins), np.nan)
    for i, path in enumerate(jk_files):
        data = _read_fits_table(path)
        wp_h = data["wprp"][scale_mask] * h
        wp_matrix[i] = wp_h

    # Per-bin valid count
    valid_mask = np.isfinite(wp_matrix)  # (N, Nbins)
    valid_frac = valid_mask.mean(axis=0)

    good_bins = valid_frac >= min_valid_frac
    rp_h = rp_all_cut[good_bins]
    n_good = good_bins.sum()

    wp_matrix_good = wp_matrix[:, good_bins]  # (N, n_good)
    valid_good = valid_mask[:, good_bins]       # (N, n_good)

    # Nanmean for each bin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wp_mean = np.nanmean(wp_matrix_good, axis=0)  # (n_good,)

    # Sample covariance with per-pair valid counts
    cov = np.zeros((n_good, n_good))
    for ii in range(n_good):
        for jj in range(ii, n_good):
            both = valid_good[:, ii] & valid_good[:, jj]
            m = both.sum()
            if m < 2:
                continue
            di = wp_matrix_good[both, ii] - wp_mean[ii]
            dj = wp_matrix_good[both, jj] - wp_mean[jj]
            val = np.dot(di, dj) / (m - 1)
            cov[ii, jj] = val
            cov[jj, ii] = val

    return rp_h, wp_mean, cov
