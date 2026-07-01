:orphan:

.. _benchmark_comparat2025:

Comparat+2025 — Galaxy × eROSITA Soft X-ray
============================================

This benchmark reproduces the galaxy × soft X-ray (0.5–2 keV) angular
cross-correlation measurements of

   **Comparat et al. 2025**, A&A 697, A173
   (`arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_)

using **hod_mod**'s full halo model for the galaxy × hot-gas cross-spectrum
:math:`P_{g,X}(k,z)`.

Overview
--------

The paper cross-correlates photometric galaxy positions from the DESI Legacy
Survey DR10 (LS DR10) with soft X-ray photon positions from the eROSITA
All-Sky Survey (eRASS:5) in the western Galactic hemisphere.  Seven
stellar-mass-limited galaxy samples are defined:

.. list-table::
   :header-rows: 1
   :widths: 8 18 12 14 18

   * - Label
     - :math:`\log_{10}(M_*/M_\odot)` cut
     - :math:`z_{\rm max}`
     - :math:`z_{\rm mean}`
     - :math:`N_{\rm gal}`
   * - S1
     - :math:`> 10.00`
     - 0.18
     - 0.136
     - 2 759 238
   * - S2
     - :math:`> 10.25`
     - 0.22
     - 0.172
     - 3 308 841
   * - S3
     - :math:`> 10.50`
     - 0.26
     - 0.205
     - 3 263 228
   * - S4
     - :math:`> 10.75`
     - 0.31
     - 0.243
     - 2 802 710
   * - S5
     - :math:`> 11.00`
     - 0.35
     - 0.261
     - 1 619 838
   * - S6
     - :math:`> 11.25`
     - 0.35
     - 0.261
     - 541 855
   * - S7
     - :math:`> 11.50`
     - 0.35
     - 0.261
     - 120 882

The observable is the Landy-Szalay angular cross-correlation
:math:`w_\theta(\theta)` between galaxy positions (data :math:`D`)
and X-ray photon positions (random photon field :math:`R_X`):

.. math::

   w_\theta(\theta) = \frac{DD - DR_X - D_XR + R_XR}{R_XR}

measured over 40 angular bins in the range
:math:`\theta \in [9\,{\rm kpc},\,842\,{\rm kpc}]` (physical).

Observable data
---------------

Data files are stored as CSV in
``hod_mod/data/benchmarks/xray/comparat2025_wtheta_{S1..S7}.csv``,
converted from the original FITS files in the Comparat+2025 analysis
archive.  Each file has columns:

* ``theta_rad`` — angular separation [rad]
* ``theta_deg`` — angular separation [deg]
* ``wtheta`` — Landy-Szalay estimator :math:`w_\theta(\theta)` [dimensionless]
* ``wtheta_err`` — jackknife error on :math:`w_\theta`
* ``R_kpc`` — physical scale at :math:`z_{\rm mean}` [kpc]

Typical amplitude: :math:`w_\theta \approx 1.6` at 9 kpc falling to
:math:`w_\theta \approx 0.027` at 842 kpc.

Model
-----

Cosmology
~~~~~~~~~

All model predictions use the Planck 2018 cosmology
(`arXiv:1807.06209 <https://arxiv.org/abs/1807.06209>`_):
:math:`\Omega_m = 0.30966`, :math:`h = 0.6766`, :math:`\sigma_8 \approx 0.811`,
:math:`n_s = 0.965`.

HOD — Zu & Mandelbaum 2015 (iHOD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The galaxy occupation is modelled with
:class:`~hod_mod.connection.hod.ZuMandelbaum15HODModel` with the tying
convention from Comparat+2025 (Table 3):

.. math::

   \log_{10}M_0 &= \log_{10}M_{\rm min} + \Delta M_0 \\
   \log_{10}M_1 &= \log_{10}M_0 + 1.0

Best-fit HOD parameters from Comparat+2025 Table 3:

.. list-table::
   :header-rows: 1
   :widths: 8 18 14 16 16

   * - Sample
     - :math:`\log_{10}M_{\rm min}`
     - :math:`\alpha_{\rm sat}`
     - :math:`\sigma_{\log m}`
     - :math:`\Delta M_0`
   * - S1
     - 12.113
     - 1.184
     - 0.666
     - 0.052
   * - S2
     - 12.260
     - 1.178
     - 0.619
     - 0.014
   * - S3
     - 12.362
     - 1.163
     - 0.538
     - 0.016
   * - S4
     - 12.327
     - 1.091
     - 0.228
     - 0.031
   * - S5
     - 12.674
     - 1.131
     - 0.202
     - 0.018
   * - S6
     - 13.096
     - 1.159
     - 0.123
     - 0.036
   * - S7
     - 13.483
     - 1.261
     - 0.100
     - 0.002

Gas profile — DPM Model 2
~~~~~~~~~~~~~~~~~~~~~~~~~~

The hot-gas component uses the DPM electron density profile
(:class:`~hod_mod.gas.GasDensityDPM`, model=2;
`Oppenheimer et al. 2025 <https://arxiv.org/abs/2505.14782>`_) with
parameters:
:math:`n_{e,03} = 4.87\times10^{-5}\,{\rm cm}^{-3}`,
:math:`\beta = 0.36`,
:math:`\alpha_{\rm in} = 1.0`,
:math:`\alpha_{\rm tr} = 2.0`,
:math:`\alpha_{\rm out} = 2.7`.

.. note::

   The free parameters ``beta_gas`` and ``beta_pressure`` (density and
   pressure mass-slopes :math:`\beta_n`, :math:`\beta_P`) are refit jointly
   with the HOD; see the S1 MAP best-fit below. ``scripts/validate_gas_profiles.py``
   (see :doc:`scripts`) regenerates the X-ray/tSZ scaling-relation figures
   (``gas_04_scaling_relations.pdf``, ``gas_06_xray_calibration.pdf``) using
   these fitted slopes, and should be rerun whenever this fit is updated.

The X-ray emissivity FT per halo:

.. math::

   \tilde\varepsilon(k|M,z) = \int_0^{3\,R_{200}}
   n_e^2(r|M,z)\,\frac{\sin(kr)}{kr}\,4\pi r^2\,dr
   \quad [({\rm Mpc}/h)^3\,{\rm cm}^{-6}]

Best-fit X-ray model parameters from Comparat+2025 Table 4:

.. list-table::
   :header-rows: 1
   :widths: 8 16 16 12 12 14 14

   * - Sample
     - :math:`\alpha_{\rm SR}`
     - :math:`\Delta M_{\rm min}`
     - :math:`w_0`
     - :math:`\chi^2/{\rm dof}`
     - :math:`L_X^{\rm ps}`
     - :math:`L_X^{\rm gas}`
   * - S1
     - 1.629
     - 0.911
     - 0.011
     - 0.918
     - 3.59
     - 7.81
   * - S2
     - 1.573
     - 1.007
     - 0.014
     - 0.830
     - 4.54
     - 11.60
   * - S3
     - 1.612
     - 1.131
     - 0.016
     - 0.752
     - 6.20
     - 15.64
   * - S4
     - 1.654
     - 0.851
     - 0.019
     - 1.083
     - 8.63
     - 18.27
   * - S5
     - 1.634
     - 1.221
     - 0.021
     - 1.103
     - 7.60
     - 42.11
   * - S6
     - 1.713
     - 2.181
     - 0.021
     - 0.865
     - 7.43
     - 137.91
   * - S7
     - 1.544
     - 2.054
     - 0.021
     - 0.788
     - 5.53
     - 401.20

Angular cross-correlation model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The angular cross-correlation is obtained from the 3D cross-power spectrum via:

1. Compute :math:`P_{g,X}(k, z_{\rm mean})` using
   :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra._pk_tables_gX`.

2. Convert to the angular power spectrum via the Limber approximation:

   .. math::

      C_\ell^{g,X} = \int \frac{d\chi}{\chi^2}\,W_g(\chi)\,
      P_{g,X}\!\left(k=\ell/\chi,\,z(\chi)\right)

   using :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.angular_cl_gX`.

3. Optionally multiply :math:`C_\ell^{g,X}` by a PSF window :math:`B_\ell`
   to account for the eROSITA instrument PSF.

   Two models are available (see :doc:`galaxies`, § *eROSITA PSF window functions*):

   * **Gaussian** — :math:`B_\ell = \exp(-\ell^2\sigma^2/2)`, controlled by
     ``psf_fwhm_arcsec``.
   * **King profile** (recommended) — :math:`B_\ell = \exp(-\ell\,\theta_c)` for
     :math:`\alpha=3/2`, fitted to the eROSITA TM CalDB (on-axis, 0.5–2 keV):
     :math:`\theta_c = 8.64''`, FWHM = 13.2''.  Avoids the truncation rebounds
     that appear when a tabulated PSF is Fourier-transformed with finite support.
     Controlled by ``psf_king_theta_c_arcsec``.

   .. figure:: _images/erosita_psf_king_fit.png
      :width: 100%
      :align: center

      PSF radial profiles (TM1–TM7), King fit residuals, and :math:`B_\ell`
      comparison.  See :doc:`galaxies` for full caption.

4. Transform :math:`C_\ell^{g,X}` to :math:`w_\theta(\theta)` via the inverse Legendre
   transform or fast Hankel transform.

A free amplitude normalization :math:`A_X` (or equivalently
:math:`\log_{10}L_X^{\rm gas}`) is introduced to account for uncertainty
in the absolute X-ray emissivity calibration.  Point-source (AGN/XRB)
and satellite components are added as separate HOD-weighted contributions
following Comparat+2025 §4.

Diagnostic predictions (not fit)
---------------------------------

``fit_comparat2025.py --plot-only`` additionally renders three model
predictions that are **not** part of the :math:`w_\theta + w_p` likelihood —
useful consistency checks, not fits.  Implemented in
:mod:`hod_mod.scripts.fitting.fit_comparat2025` (``plot_diagnostics``,
``{label}_diagnostics.pdf``).

Stellar mass function :math:`\Phi(M_*)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZuMandelbaum15HODModel galaxy number density above a stellar-mass
threshold (More+2015 Eq. 12, see :meth:`~hod_mod.observables.clustering.FullHaloModelPrediction.n_gal`) is

.. math::

   \bar n_g(>M_{*,\rm th}) = \int \frac{dn}{dM}\,
   \bigl[\langle N_{\rm cen}(M\,|\,M_{*,\rm th})\rangle +
         \langle N_{\rm sat}(M\,|\,M_{*,\rm th})\rangle\bigr]\,dM
   \quad [h^3\,{\rm Mpc}^{-3}]

The differential stellar-mass function is obtained by finite-differencing
this cumulative count over a grid of thresholds
(:func:`~hod_mod.scripts.fitting.fit_comparat2025._predict_smf`):

.. math::

   \Phi(M_*) = -\frac{d\bar n_g(>M_*)}{d\log_{10}M_*}
   \quad [{\rm Mpc}^{-3}\,{\rm dex}^{-1}]

**Unit conversion.** ``log10m_star_thresh`` is defined in
:math:`\log_{10}(M_*/[M_\odot\,h^{-1}])` (ZM15 convention), while the
sum_stat SMF is tabulated in the standard h-free convention
:math:`\log_{10}(M_*/M_\odot)`.  The threshold grid must therefore be
shifted by :math:`+\log_{10}h` before being passed to ``n_gal``, and the
resulting :math:`\bar n_g` — returned in :math:`h^3\,{\rm Mpc}^{-3}` — must
be multiplied by :math:`h^3` to match the data's physical
:math:`{\rm Mpc}^{-3}` units:

.. math::

   \Phi(M_*^{\rm phys}) = -h^3\,
   \frac{d}{d\log_{10}M_*^{\rm phys}}\,
   \bar n_g\!\bigl(>\log_{10}M_*^{\rm phys} + \log_{10}h\bigr)

Because the HOD shape parameters (:math:`M_1`, :math:`\sigma_{\ln M_*}`,
:math:`\alpha_{\rm sat}`, …) are constrained only by clustering
(:math:`w_\theta`, :math:`w_p`) and never by an SMF/:math:`n_g` term in the
likelihood, a residual normalisation offset of a few :math:`\times` between
model and data is expected even after the unit fix above.

Galaxy–galaxy lensing :math:`\Delta\Sigma(r_p)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard halo-model excess surface density (see :doc:`galaxies`, §
*Excess surface density*), evaluated at the best-fit HOD parameters with
:meth:`~hod_mod.observables.clustering.FullHaloModelPrediction.delta_sigma`:

.. math::

   \Delta\Sigma(R) = \frac{2}{R^2}\int_0^R R'\,\Sigma_{gm}(R')\,dR' - \Sigma_{gm}(R)

and overlaid against the HSC/DES/KIDS measurements from sum_stat **without
refitting** — a pure forward-model comparison.

X-ray auto-power spectrum :math:`C_\ell^{X,X}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forward-model-only prediction (no data exists yet for this observable); see
:doc:`galaxies`, § *X-ray auto-power* :math:`P_{X,X}(k)` for the full
1-halo/2-halo gas×AGN decomposition and the Limber integral with
:math:`W_X(\chi)^2`, computed via
:meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.angular_cl_XX`.

Validation script
-----------------

The script ``hod_mod/scripts/validate_comparat2025.py`` (once complete) will:

1. Load the 7 CSV data files from ``hod_mod/data/benchmarks/xray/``.
2. Set up the HOD from Table 3 for each sample.
3. Compute the halo-model :math:`w_\theta(\theta)` using DPM Model 2.
4. Compare with the measured :math:`w_\theta(\theta)` and report :math:`\chi^2/{\rm dof}`.
5. Save comparison figures to ``hod_mod/scripts/figures/comparat2025_*.pdf``.

Reproduce with::

    cd $HOD_MOD_REPO
    python -m hod_mod.scripts.validate_comparat2025

References
----------

* Comparat et al. 2025, A&A 697, A173
  (`arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_)
* Oppenheimer et al. 2025
  (`arXiv:2505.14782 <https://arxiv.org/abs/2505.14782>`_) — DPM gas profile
* Zu & Mandelbaum 2015, MNRAS 454, 1161
  (`arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_) — iHOD model
* Tinker et al. 2008 (`arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_) — HMF
* Planck Collaboration 2018 (`arXiv:1807.06209 <https://arxiv.org/abs/1807.06209>`_) — cosmology
