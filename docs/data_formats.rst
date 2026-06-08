.. _data_formats:

Data Formats
============

.. contents:: Contents
   :local:
   :depth: 2

``hod_mod`` consumes two types of input data:

* **HDF5** — primary format for galaxy surveys (BGS/LS10, mocks) produced by the
  companion `sum_stat <https://github.com/JohanComparat/sum_stat>`_ package; stores
  the full covariance matrix and cosmological metadata.
* **CSV + JSON** — paper benchmark datasets bundled in ``data/{paper_name}/``; plain
  CSV files with a ``metadata.json`` sidecar describing cosmology and column meanings.

All spatial quantities in ``sum_stat`` are stored in **Mpc** (h-free).
``SumStatReader`` converts to **Mpc/h** automatically using the ``H0`` attribute
embedded in each file, so all arrays returned by the reader are already in the
h-unit system required by ``hod_mod``.

---

sum_stat HDF5 Schema
--------------------

Single-statistic files (e.g. a w_p-only measurement) use one of the following top-level
group structures.  Joint files nest these same groups together.

w_p (projected correlation function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    wp/
      sep_centres       (N_rp,)  float64   — projected separation bin centres [Mpc]
      xi                (N_rp,)  float64   — w_p values [Mpc]   (name "xi" is historical)
      cov               (N_rp, N_rp) float64 — covariance matrix [Mpc²]
      bin_edges         (N_rp+1,) float64  — bin edges [Mpc]
      cosmology/
        H0              scalar  — Hubble constant H₀ [km/s/Mpc]
        Om0             scalar  — Ω_m(z=0)
        Ob0             scalar  — Ω_b(z=0)
        Ok0             scalar  — Ω_k(z=0)
      attrs:
        pi_max_Mpc      float   — line-of-sight integration limit [Mpc]
        estimator       str     — "landy-szalay" | "hamilton"
        survey          str
        n_gal           int     — galaxy count in sample

SMF (stellar mass function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    smf/
      log10mstar_centres (N_m,) float64  — log₁₀(M*/M⊙) bin centres
      phi               (N_m,)  float64  — Φ(M*) [Mpc⁻³ dex⁻¹]
      phi_err           (N_m,)  float64  — 1σ uncertainty [Mpc⁻³ dex⁻¹]
      cov               (N_m, N_m) float64
      bin_edges         (N_m+1,) float64
      cosmology/        — same sub-group as above

ESD (excess surface density / weak-lensing ΔΣ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    esd/
      rp_centres        (N_R,)  float64  — projected radius [Mpc]
      delta_sigma       (N_R,)  float64  — ΔΣ(R) [M⊙/pc²]
      cov               (N_R, N_R) float64
      cosmology/

Joint files
~~~~~~~~~~~

A joint file combines the three statistics above and adds a pre-computed joint
covariance block:

.. code-block:: text

    smf/{sample_key}/           — SMF group (structure as above)
    twopcf/{sample_key}/        — w_p group, named "wp_{sample_key}" is also common
    esd/{sample_key}/           — ESD group
    joint_covariance/
      data_vector       (N_tot,) float64   — [phi | wp | delta_sigma]
      cov               (N_tot, N_tot)     — full joint covariance
      err_jackknife     (N_tot,)            — √diag(cov)
      mstar_centres     (N_smf,)            — same as smf/log10mstar_centres
      rp_centres        (N_wp,)             — same as twopcf/sep_centres
      attrs:
        n_bins_smf      int
        n_bins_wp       int
        n_bins_ds       int

File naming convention::

    {SURVEY}_VLIM_ANY_Mstar{MSTAR_LO}-{MSTAR_HI}_z{Z_MIN}-{Z_MAX}-{STAT}.h5

    Examples:
      LS10_VLIM_ANY_Mstar10.5-12.0_z0.05-0.18-wp-pimax100-sys-comb.h5
      MOCK_VLIM_ANY_Mstar11.39_z0.05-0.35-wp-pimax100.h5

---

Unit Conversion
---------------

``SumStatReader.from_hdf5()`` reads ``h = H0/100`` from the embedded cosmology group
and applies the following conversions before returning arrays:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Quantity
     - sum_stat unit
     - hod_mod unit
     - Conversion
   * - :math:`r_p` (separation)
     - Mpc
     - Mpc/h
     - :math:`r_p^{h} = r_p \times h`
   * - :math:`w_p` (correlation)
     - Mpc
     - Mpc/h
     - :math:`w_p^{h} = w_p \times h`
   * - Cov(:math:`w_p`)
     - Mpc²
     - (Mpc/h)²
     - :math:`C^{h} = C \times h^2`
   * - :math:`\Phi(M_*)` (SMF)
     - Mpc⁻³ dex⁻¹
     - (Mpc/h)⁻³ dex⁻¹
     - :math:`\Phi^{h} = \Phi / h^3`
   * - Cov(:math:`\Phi`)
     - Mpc⁻⁶
     - (Mpc/h)⁻⁶
     - :math:`C_\Phi^{h} = C_\Phi / h^6`
   * - :math:`\Delta\Sigma(R)` (ESD)
     - :math:`M_\odot/\mathrm{pc}^2`
     - :math:`M_\odot/\mathrm{pc}^2`
     - (invariant — pc absorbs the h)
   * - :math:`\log_{10}(M_*/M_\odot)`
     - dimensionless
     - dimensionless
     - (no change needed)

.. warning::
   The ``xi`` dataset in the ``wp/`` HDF5 group stores the **projected** correlation
   function :math:`w_p(r_p)`, not the 3D correlation function :math:`\xi(r)`.  This
   naming is historical (TreeCorr uses ``xi`` as the generic correlation variable).

---

Paper Benchmark Data (CSV + JSON)
----------------------------------

All published benchmark datasets are stored under ``data/{paper_name}/`` as plain CSV
files with an accompanying ``metadata.json``.

Directory layout
~~~~~~~~~~~~~~~~

.. code-block:: text

    data/
      guo2018_sdss/
        metadata.json
        wp_mstar10_lowz.csv
        ...
      leauthaud2012_cosmos/
        metadata.json
        ds_photo_z2_thresh106.csv
        wp_photo_z2_thresh106.csv
        ...

wp CSV (projected correlation function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    rp_hMpc           — projected separation r_p  [h⁻¹ Mpc]
    wp_hMpc           — w_p(r_p)                  [h⁻¹ Mpc]
    wp_err_hMpc       — 1σ uncertainty             [h⁻¹ Mpc]

Lines beginning with ``#`` are comments (header / provenance notes) and are ignored by
the reader.

ΔΣ CSV (excess surface density)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    R_hMpc            — projected radius R         [h⁻¹ Mpc]
    ds_Msun_h_pc2     — ΔΣ(R)                      [M⊙ h pc⁻²]
    ds_err_Msun_h_pc2 — 1σ uncertainty             [M⊙ h pc⁻²]

metadata.json
~~~~~~~~~~~~~

One JSON sidecar per dataset. Fields:

.. code-block:: text

    paper             — citation string (e.g. "Guo et al. 2018")
    arxiv             — arXiv identifier
    survey            — survey name (e.g. "SDSS BOSS LOWZ")
    sample            — description of the galaxy sample
    z_eff             — effective redshift
    pi_max_hMpc       — line-of-sight integration limit [h⁻¹ Mpc]  (wp only)
    cosmology         — dict: Omega_m, h, sigma8, n_s, Omega_b
    observable        — "wp" | "wp+ds" | …
    columns_wp        — dict mapping column names to descriptions
    columns_ds        — dict mapping column names to descriptions
    status            — "ready" | "NEEDS_DATA" |
                        "NOT_APPLICABLE_FOR_PROJECTED_BENCHMARKS"
    published_params  — dict of best-fit HOD parameters from the paper (optional)
    published_param_errors — uncertainties on published_params (optional)
    notes             — free-text remarks

---

Reading Data in Python
----------------------

.. code-block:: python

    from hod_mod.data_io.sum_stat_reader import SumStatReader

    # ── HDF5 single w_p file ──────────────────────────────────────────────────
    reader = SumStatReader.from_hdf5(
        "LS10_VLIM_ANY_Mstar10.5-12.0_z0.05-0.18-wp-pimax100-sys-comb.h5"
    )
    d = reader.wp()
    # d["rp"]    shape (N_rp,)      Mpc/h
    # d["wp"]    shape (N_rp,)      Mpc/h
    # d["cov"]   shape (N_rp, N_rp) (Mpc/h)²
    # d["pi_max"]                   float, Mpc/h

    # ── HDF5 joint file ───────────────────────────────────────────────────────
    joint_reader = SumStatReader.from_hdf5("joint_stats.h5")
    j = joint_reader.joint()
    # j["data_vector"]    shape (N_smf + N_wp + N_ds,)
    # j["cov"]            shape (N_tot, N_tot)
    # j["n_bins_smf"], j["n_bins_wp"], j["n_bins_ds"]

    # ── Paper benchmark CSV + metadata.json ───────────────────────────────────
    import json
    import pandas as pd

    meta = json.load(open("data/guo2018_sdss/metadata.json"))
    df   = pd.read_csv("data/guo2018_sdss/wp_mstar10_lowz.csv", comment="#")
    # df.columns: rp_hMpc, wp_hMpc, wp_err_hMpc
    h = meta["cosmology"]["h"]

.. automodule:: hod_mod.data_io.sum_stat_reader
   :members:
   :undoc-members:
