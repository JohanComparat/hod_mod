"""Read summary statistics produced by the ``sum_stat`` package.

The ``sum_stat`` package stores measurements in two formats:

* **HDF5** — LS10/BGS two-point functions and stellar mass functions.
  See :ref:`data_formats` for the full schema.
* **FITS binary tables** — GAMA and COSMOS stellar mass / luminosity functions.

All angular and projected distances are stored in physical Mpc by ``sum_stat``.
This module converts everything to h-units (Mpc/h) on the way out, using
the Hubble constant stored in the file's ``cosmology/H0`` sub-group.

Conversion rules
----------------
* Distances:          ``r_h    = r_Mpc * h``
* Volumes:            ``V_h    = V_Mpc * h^3``
* Number densities:   ``phi_h  = phi_Mpc / h^3``
* Covariances scale as the square of the primary quantity.

Examples
--------
Read a single-stat projected correlation function file::

    reader = SumStatReader.from_hdf5(
        "/path/to/sum_stat/data/twopcf/"
        "LS10_VLIM_ANY_Mstar10.0-12.0_z0.05-0.18-wp-pimax100-sys-comb.h5"
    )
    data = reader.wp()
    rp   = data["rp"]     # (N,) array, Mpc/h
    wp   = data["wp"]     # (N,) array, Mpc/h
    cov  = data["cov"]    # (N, N) array, (Mpc/h)^2

Read a joint SMF+wp+ESD file::

    reader = SumStatReader.from_hdf5(
        "/path/to/sum_stat/data/BGS_Mstar10.00/"
        "joint_smf_wprp_deltasigma-sys-comb.h5"
    )
    jt = reader.joint()
    # jt["data_vector"], jt["cov"] — full joint data vector and covariance
    # jt["n_bins_smf"], jt["n_bins_wp"], jt["n_bins_ds"]

Read a GAMA stellar mass function::

    reader = SumStatReader.from_fits(
        "/path/to/sum_stat/data/GAMA/gama_smf_z0.060_0.100.fits"
    )
    data = reader.smf()

References
----------
* Planck Collaboration 2020, A&A 641, A6 (https://arxiv.org/abs/1807.06209)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# HDF5 helper
# ---------------------------------------------------------------------------

def _h5_h(f) -> float:
    """Extract h = H0/100 from an open HDF5 file.

    Searches two levels deep: ``f[group]["cosmology"]`` (old format) and
    ``f[group][subgroup]["cosmology"]`` (new BGS joint format where cosmology
    lives inside ``esd/{sample_name}/cosmology``).
    """
    import h5py

    for key in f.keys():
        grp = f[key]
        if not isinstance(grp, h5py.Group):
            continue
        # One level deep
        cosmo = grp.get("cosmology")
        if cosmo is not None and "H0" in cosmo:
            return float(cosmo["H0"][()]) / 100.0
        # Two levels deep (new joint format)
        for subkey in grp.keys():
            subgrp = grp[subkey]
            if not isinstance(subgrp, h5py.Group):
                continue
            cosmo = subgrp.get("cosmology")
            if cosmo is not None and "H0" in cosmo:
                return float(cosmo["H0"][()]) / 100.0
    raise KeyError("Cannot find cosmology/H0 in HDF5 file.")


def _first_subgroup(group) -> Any:
    """Return the first child group of an HDF5 group."""
    keys = list(group.keys())
    for k in keys:
        try:
            import h5py
            if isinstance(group[k], h5py.Group):
                return group[k], k
        except Exception:
            pass
    return group[keys[0]], keys[0]


# ---------------------------------------------------------------------------
# Main reader class
# ---------------------------------------------------------------------------

class SumStatReader:
    """Unified reader for ``sum_stat`` HDF5 and FITS measurement files.

    Do not instantiate directly; use the class methods:

    * :meth:`from_hdf5` — for HDF5 files from the ``twopcf``, ``lf_smf``,
      ``mocks``, or ``BGS_Mstar*`` directories.
    * :meth:`from_fits` — for FITS files from the ``GAMA`` or ``COSMOS``
      directories.
    """

    def __init__(self, path: str, fmt: str, _cache: dict):
        self._path  = path
        self._fmt   = fmt   # "hdf5" or "fits"
        self._cache = _cache

    # ------------------------------------------------------------------
    # Constructors

    @classmethod
    def from_hdf5(cls, path: str) -> "SumStatReader":
        """Open a sum_stat HDF5 file.

        Parameters
        ----------
        path : str
            Absolute or relative path to the HDF5 file.

        Returns
        -------
        SumStatReader
        """
        import h5py

        if not os.path.isfile(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        cache: dict = {}
        with h5py.File(path, "r") as f:
            h = _h5_h(f)
            cache["h"]    = h
            cache["_top"] = list(f.keys())
            cache["path"] = path
            cache["root_attrs"] = dict(f.attrs)

            # Single-stat wp file: top-level group is "wp"
            if "wp" in f:
                g = f["wp"]
                rp   = np.array(g["sep_centres"]) * h
                xi   = np.array(g["xi"])          * h
                cov  = np.array(g["cov"])         * h**2
                cache["wp"] = {
                    "rp":       rp,
                    "wp":       xi,
                    "cov":      cov,
                    "pi_max":   float(g.attrs.get("pi_max_Mpc", 100.0)),
                    "z_eff":    float(g.attrs.get("z_eff", 0.0)) if "z_eff" in g.attrs else None,
                    "survey":   str(g.attrs.get("survey", "")),
                    "n_gal":    int(g.attrs.get("n_gal", 0)) if "n_gal" in g.attrs else None,
                    "estimator": str(g.attrs.get("estimator", "")),
                    "cov_method": str(g.attrs.get("cov_method", "")),
                    "attrs":    dict(g.attrs),
                }

            # Single-stat smf file: top-level group is "smf"
            if "smf" in f and not "twopcf" in f:
                g = f["smf"]
                log10m = np.array(g["log10mstar_centres"])
                phi    = np.array(g["phi"])     / h**3
                phi_e  = np.array(g["phi_err"]) / h**3
                cov    = np.array(g["cov"])     / h**6
                cache["smf"] = {
                    "log10mstar": log10m,
                    "phi":        phi,
                    "phi_err":    phi_e,
                    "cov":        cov,
                    "estimator":  str(g.attrs.get("estimator", "")),
                    "cov_method": str(g.attrs.get("cov_method", "")),
                    "attrs":      dict(g.attrs),
                }

            # Joint file: top-level groups include "smf", "twopcf", "esd", "joint_covariance"
            if "twopcf" in f:
                twop_grp = f["twopcf"]
                subgrp, _ = _first_subgroup(twop_grp)
                rp  = np.array(subgrp["sep_centres"]) * h
                xi  = np.array(subgrp["xi"])           * h
                cov = np.array(subgrp["cov"])          * h**2
                cache["wp"] = {
                    "rp":       rp,
                    "wp":       xi,
                    "cov":      cov,
                    "pi_max":   float(subgrp.attrs.get("pi_max_Mpc", 100.0)),
                    "estimator": str(subgrp.attrs.get("estimator", "")),
                    "attrs":    dict(subgrp.attrs),
                }

            if "smf" in f and "twopcf" in f:
                smf_grp = f["smf"]
                subgrp, _ = _first_subgroup(smf_grp)
                log10m = np.array(subgrp["log10mstar_centres"])
                phi    = np.array(subgrp["phi"])     / h**3
                phi_e  = np.array(subgrp["phi_err"]) / h**3
                cov    = np.array(subgrp["cov"])     / h**6
                cache["smf"] = {
                    "log10mstar": log10m,
                    "phi":        phi,
                    "phi_err":    phi_e,
                    "cov":        cov,
                    "estimator":  str(subgrp.attrs.get("estimator", "")),
                    "attrs":      dict(subgrp.attrs),
                }

            if "esd" in f:
                esd_grp = f["esd"]
                subgrp, _ = _first_subgroup(esd_grp)
                # rp in Mpc → Mpc/h; delta_sigma in M_sun/pc² (no h conversion needed)
                rp  = np.array(subgrp["rp_centres"])  * h
                ds  = np.array(subgrp["delta_sigma"])
                cov = np.array(subgrp["cov"])
                cache["esd"] = {
                    "rp":          rp,
                    "delta_sigma": ds,
                    "cov":         cov,
                    "source_survey": str(subgrp.attrs.get("source_survey", "")),
                    "attrs":       dict(subgrp.attrs),
                }

            if "number_density" in f:
                nd_grp = f["number_density"]
                subgrp, _ = _first_subgroup(nd_grp)
                # n is a number density (Mpc^-3 → h^3 Mpc^-3): divide by h^3.
                val = np.array(subgrp["value"]) / h**3
                err = np.array(subgrp["err"])   / h**3
                cov = np.array(subgrp["cov"])   / h**6
                cache["number_density"] = {
                    "n":         float(np.ravel(val)[0]),
                    "n_err":     float(np.ravel(err)[0]),
                    "cov":       cov,
                    "estimator": str(subgrp.attrs.get("estimator", "")),
                    "attrs":     dict(subgrp.attrs),
                }

            if "joint_covariance" in f:
                jg = f["joint_covariance"]
                jg_attrs = dict(jg.attrs)

                slice_keys = [k for k in jg_attrs if k.startswith("slice_")]
                if slice_keys:
                    # New BGS joint format: one slice_<stat> index pair per measured
                    # statistic (e.g. slice_nbar, slice_wp, slice_esd_hsc).  Built
                    # dynamically so files carrying any subset of stats parse.
                    # Read raw arrays without conversion; accessor applies h-units.
                    def _parse_slice(a):
                        return (int(a[0]), int(a[1]))

                    slices = {
                        key[len("slice_"):]: _parse_slice(jg_attrs[key])
                        for key in slice_keys
                    }
                    subs_raw = np.array(jg["subsamples"]) if "subsamples" in jg else None
                    cache["joint_bgs"] = {
                        "data_vector_raw": np.array(jg["data_vector"]),
                        "cov_raw":         np.array(jg["cov"]),
                        "subsamples_raw":  subs_raw,   # (n_jk, 286) or None
                        "slices":          slices,
                        "rp_centres_wp":   np.array(jg["rp_centres_wp"]) * h,
                        "rp_centres_esd":  np.array(jg["rp_centres_esd"]) * h,
                        "h":               h,
                        "attrs":           jg_attrs,
                    }
                else:
                    # Legacy joint format: n_bins_smf/wp/ds attrs.
                    n_smf = int(jg_attrs.get("n_bins_smf", 0))
                    n_wp  = int(jg_attrs.get("n_bins_wp",  0))
                    n_ds  = int(jg_attrs.get("n_bins_ds",  0))
                    # Data vector layout: [phi_SMF (Mpc^-3) | w_p (Mpc) | DeltaSigma (M_sun/pc^2)]
                    # Convert SMF and wp sections to h-units; ΔΣ is invariant.
                    dv_raw = np.array(jg["data_vector"])
                    dv_h   = dv_raw.copy()
                    if n_smf > 0:
                        dv_h[:n_smf] = dv_raw[:n_smf] / h**3
                    if n_wp > 0:
                        dv_h[n_smf:n_smf+n_wp] = dv_raw[n_smf:n_smf+n_wp] * h

                    cov_raw = np.array(jg["cov"])
                    scales  = np.ones(n_smf + n_wp + n_ds)
                    if n_smf > 0:
                        scales[:n_smf] = 1.0 / h**3
                    if n_wp > 0:
                        scales[n_smf:n_smf+n_wp] = h
                    cov_h = cov_raw * np.outer(scales, scales)

                    cache["joint"] = {
                        "data_vector":   dv_h,
                        "cov":           cov_h,
                        "err_jackknife": np.sqrt(np.diag(cov_h)),
                        "mstar_centres": np.array(jg["mstar_centres"]),
                        "rp_centres":    np.array(jg["rp_centres"]) * h,
                        "n_bins_smf":    n_smf,
                        "n_bins_wp":     n_wp,
                        "n_bins_ds":     n_ds,
                        "attrs":         jg_attrs,
                    }

        return cls(path, "hdf5", cache)

    @classmethod
    def from_fits(cls, path: str) -> "SumStatReader":
        """Open a GAMA or COSMOS FITS SMF/LF file.

        The FITS file must have an HDU-1 binary table with at least the
        columns ``log10mstar``, ``phi``, and ``phi_err``.  Units are assumed
        to be Mpc (not Mpc/h); no h-conversion is applied because GAMA and
        COSMOS files do not embed cosmological parameters.  Pass the
        :math:`h` value explicitly when calling :meth:`smf` if needed.

        Parameters
        ----------
        path : str
            Absolute or relative path to the FITS file.

        Returns
        -------
        SumStatReader
        """
        from astropy.io import fits

        if not os.path.isfile(path):
            raise FileNotFoundError(f"FITS file not found: {path}")

        with fits.open(path) as hdul:
            data = hdul[1].data.copy()
            header = dict(hdul[1].header)

        cache: dict = {"path": path, "_fits_data": data, "_fits_header": header}
        return cls(path, "fits", cache)

    # ------------------------------------------------------------------
    # Data accessors

    def wp(self) -> dict:
        """Return the projected correlation function w_p(r_p).

        Returns
        -------
        dict with keys:

        * ``rp``        — projected separation bin centres, Mpc/h
        * ``wp``        — projected correlation function, Mpc/h
        * ``cov``       — covariance matrix, (Mpc/h)²
        * ``pi_max``    — line-of-sight integration limit, Mpc/h
        * ``estimator`` — ``'landy-szalay'`` or similar
        * ``attrs``     — raw HDF5 group attributes
        """
        if "wp" not in self._cache:
            raise KeyError(f"No wp group found in {self._path}.")
        return self._cache["wp"]

    def smf(self, h: float | None = None) -> dict:
        """Return the stellar mass function Φ(M*).

        For FITS files (GAMA, COSMOS) the covariance matrix is not stored;
        ``cov`` will be a diagonal matrix built from ``phi_err``.

        Parameters
        ----------
        h : float, optional
            Hubble constant h = H0/100 for Mpc → Mpc/h conversion of number
            densities.  Only used for FITS files; HDF5 files carry h internally.

        Returns
        -------
        dict with keys:

        * ``log10mstar`` — log10(M*/M_sun) bin centres
        * ``phi``        — Φ(M*) in h³ Mpc⁻³ dex⁻¹
        * ``phi_err``    — uncertainty
        * ``cov``        — covariance matrix, (h³ Mpc⁻³ dex⁻¹)²
        """
        if self._fmt == "fits":
            data  = self._cache["_fits_data"]
            log10m = np.array(data["log10mstar"])
            phi    = np.array(data["phi"])
            phi_e  = np.array(data["phi_err"])
            if h is not None:
                phi   /= h**3
                phi_e /= h**3
            mask = np.isfinite(phi) & (phi > 0)
            return {
                "log10mstar": log10m[mask],
                "phi":        phi[mask],
                "phi_err":    phi_e[mask],
                "cov":        np.diag(phi_e[mask]**2),
                "estimator":  "1/Vmax",
                "attrs":      self._cache.get("_fits_header", {}),
            }
        if "smf" not in self._cache:
            raise KeyError(f"No smf group found in {self._path}.")
        return self._cache["smf"]

    def esd(self) -> dict:
        """Return the excess surface mass density ΔΣ(R).

        Returns
        -------
        dict with keys:

        * ``rp``          — projected separation bin centres, Mpc/h
        * ``delta_sigma`` — ΔΣ(R) in M_sun pc⁻²
        * ``cov``         — covariance matrix, (M_sun pc⁻²)²
        * ``attrs``       — raw HDF5 group attributes
        """
        if "esd" not in self._cache:
            raise KeyError(f"No esd group found in {self._path}.")
        return self._cache["esd"]

    def number_density(self) -> dict:
        """Return the galaxy number density n of the sample.

        Returns
        -------
        dict with keys:

        * ``n``         — number density in h³ Mpc⁻³
        * ``n_err``     — uncertainty in h³ Mpc⁻³
        * ``cov``       — (1, 1) variance in (h³ Mpc⁻³)²
        * ``estimator`` — ``'sum(w_i / Vmax_i)'``
        * ``attrs``     — raw HDF5 group attributes
        """
        if "number_density" not in self._cache:
            raise KeyError(f"No number_density group found in {self._path}.")
        return self._cache["number_density"]

    def joint(self) -> dict:
        """Return the full joint data vector and covariance matrix.

        The joint data vector has layout
        ``[φ_SMF (n_smf), w_p (n_wp), ΔΣ (n_ds)]``.

        Returns
        -------
        dict with keys:

        * ``data_vector``   — (n_total,) concatenated data vector
        * ``cov``           — (n_total, n_total) joint covariance
        * ``err_jackknife`` — sqrt(diag(cov))
        * ``mstar_centres`` — log10(M*/M_sun) bin centres for SMF section
        * ``rp_centres``    — r_p bin centres [Mpc/h] for wp and ΔΣ sections
        * ``n_bins_smf``, ``n_bins_wp``, ``n_bins_ds`` — section lengths
        * ``attrs``         — raw HDF5 group attributes
        """
        if "joint" not in self._cache:
            raise KeyError(f"No joint_covariance group found in {self._path}.")
        return self._cache["joint"]

    def joint_bgs(
        self,
        probes: tuple = ("wp", "esd_hsc", "esd_des"),
    ) -> dict:
        """Extract a multi-probe sub-data-vector from the new BGS joint format.

        Selects the requested probes from the full 286-element joint data vector
        and returns the corresponding sub-block of the joint covariance matrix,
        with h-unit conversion applied:

        * ``wp``       — w_p multiplied by h (Mpc → Mpc/h); covariance × h²
        * ``esd_*``    — ΔΣ left as M_sun pc⁻² (invariant); covariance unchanged
        * Cross-terms  — scaled by h¹ (WP axis) × 1 (ESD axis) = h

        Parameters
        ----------
        probes : tuple of str
            Any subset of ``('smf', 'wp', 'esd_hsc', 'esd_des', 'esd_kids',
            'wtheta', 'knn')``.  Default: ``('wp', 'esd_hsc', 'esd_des')``.

        Returns
        -------
        dict with keys:

        * ``data_vector`` — (n_sel,) sub-vector in h-units
        * ``cov``         — (n_sel, n_sel) sub-block covariance in h-units
        * ``rp_wp``       — (30,) r_p bin centres for WP [Mpc/h]  (if wp requested)
        * ``rp_esd``      — (30,) r_p bin centres for ESD [Mpc/h] (if any esd requested)
        * ``slices_out``  — ``{probe: slice}`` mapping probe name to its position
                            in the returned ``data_vector``
        * ``h``           — embedded Hubble constant h = H0/100
        * ``attrs``       — raw HDF5 group attributes
        """
        if "joint_bgs" not in self._cache:
            raise KeyError(
                f"No BGS joint_covariance group found in {self._path}. "
                "Use joint() for the legacy format."
            )
        raw  = self._cache["joint_bgs"]
        h    = raw["h"]
        sls  = raw["slices"]

        # Probe → h-scale factor (WP: Mpc → Mpc/h; ESD: invariant; others: 1)
        _wp_probes  = {"wp", "wtheta"}
        _smf_probes = {"smf", "nbar"}   # number densities: Mpc^-3 → h^3 Mpc^-3

        indices   = []
        scales    = []
        slices_out: dict = {}
        cursor    = 0

        for probe in probes:
            if probe not in sls:
                raise ValueError(
                    f"Unknown probe '{probe}'. Choose from {list(sls.keys())}."
                )
            lo, hi = sls[probe]
            n = hi - lo
            indices.extend(range(lo, hi))
            if probe in _wp_probes:
                scales.extend([h] * n)
            elif probe in _smf_probes:
                scales.extend([1.0 / h**3] * n)
            else:
                scales.extend([1.0] * n)
            slices_out[probe] = slice(cursor, cursor + n)
            cursor += n

        idx    = np.array(indices, dtype=int)
        sc     = np.array(scales)
        dv_sub = raw["data_vector_raw"][idx] * sc
        cov_sub = raw["cov_raw"][np.ix_(idx, idx)] * np.outer(sc, sc)

        out: dict = {
            "data_vector": dv_sub,
            "cov":         cov_sub,
            "slices_out":  slices_out,
            "h":           h,
            "attrs":       raw["attrs"],
        }
        if "rp_centres_wp" in raw:
            out["rp_wp"]  = raw["rp_centres_wp"]
        if "rp_centres_esd" in raw:
            out["rp_esd"] = raw["rp_centres_esd"]
        if raw.get("subsamples_raw") is not None:
            subs_h = raw["subsamples_raw"][:, idx] * sc[np.newaxis, :]
            out["subsamples"] = subs_h   # (n_jk, n_sel)
        return out

    def list_groups(self) -> list:
        """List available statistic types in this file."""
        return [k for k in ("wp", "smf", "esd", "number_density", "joint", "joint_bgs")
                if k in self._cache]

    def attrs(self) -> dict:
        """File-level attributes (creation date, version)."""
        return self._cache.get("root_attrs", {})

    def h(self) -> float | None:
        """Hubble constant h = H0/100 embedded in the file (HDF5 only)."""
        return self._cache.get("h")
