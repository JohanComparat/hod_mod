Galaxy × Thermal Sunyaev–Zel'dovich (tSZ)
==========================================

The thermal SZ effect measures the line-of-sight integral of the electron
**pressure** through the hot intracluster and circumgalactic medium. Cross-correlating
a galaxy sample with a Compton-:math:`y` map probes the pressure–halo connection of
the galaxies' host haloes.

This pipeline reuses the same halo-model engine as the galaxy clustering and
galaxy × X-ray pipelines: :class:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra`
wraps an existing :class:`~hod_mod.observables.clustering.FullHaloModelPrediction`
(reusing its cached halo mass function, bias, linear power spectrum and dark-matter
profile FT) and adds an electron-pressure field
(:class:`~hod_mod.gas.PressureProfileA10`, Arnaud+2010, or the DPM variant).

The model
---------

The galaxy × Compton-:math:`y` cross-power spectrum has the usual 1-halo + 2-halo
decomposition,

.. math::

   P_{g,y}(k, z) = P_{g,y}^{1h}(k,z) + P_{g,y}^{2h}(k,z),

where the galaxy leg is the occupation :math:`\langle N_g \rangle(M)` weighted by the
halo number density, and the :math:`y` leg is the Fourier transform of the electron
pressure profile :math:`\tilde{y}(k|M, z)` (dimensionless Compton-:math:`y`). The
projected and angular observables follow by:

* :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.projected_gy` —
  the stacked Compton-:math:`y` profile :math:`\Sigma_y(r_p)` via Abel projection;
* :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.angular_cl_gy` —
  the angular cross-spectrum :math:`C_\ell^{g,y}` via the Limber approximation.

Worked example
--------------

.. code-block:: python

    import numpy as np
    from hod_mod.core.power_spectrum import LinearPowerSpectrum
    from hod_mod.core.halo_mass_function import make_hmf
    from hod_mod.core.halo_profiles import HaloProfile
    from hod_mod.connection.hod import MoreHODModel
    from hod_mod.observables.clustering import FullHaloModelPrediction
    from hod_mod.observables.cross_spectra import HaloModelCrossSpectra
    from hod_mod.gas import PressureProfileA10

    theta = {"h": 0.6774, "Omega_m": 0.3089, "Omega_b": 0.0486,
             "n_s": 0.9667, "sigma8": 0.8159}

    pk_lin = LinearPowerSpectrum()
    hmf    = make_hmf("tinker08", pk_func=pk_lin.pk_linear)
    hod    = MoreHODModel(hmf, hmf.bias)
    hp     = HaloProfile()

    fhmp   = FullHaloModelPrediction(pk_lin, hod, hp)
    pp     = PressureProfileA10(r_max_over_r500c=5.0, n_gl=150)
    cross  = HaloModelCrossSpectra(fhmp, pressure_profile=pp)

    rp      = np.logspace(-1, 1.3, 20)           # Mpc/h
    z       = 0.3
    params  = hod.default_params()
    sigma_y = cross.projected_gy(rp, z, theta, params)

    # angular C_ell^{g,y} integrates over the galaxy redshift distribution n(z)
    ell     = np.logspace(2, 4, 30)
    z_arr   = np.linspace(0.2, 0.5, 16)
    nz_g    = np.exp(-0.5 * ((z_arr - 0.3) / 0.05) ** 2)
    cl_gy   = cross.angular_cl_gy(ell, z_arr, nz_g, theta, params)

The instrument beam
-------------------

A Compton-:math:`y` map is convolved with the instrument beam (ACT DR6 NILC: 1.6 arcmin FWHM),
so a measurement made from one is beam-convolved and an unbeamed model is **not the same
quantity**.  This matters most in the innermost :math:`r_p` bins, which is exactly where the
signal is.

Both observables take an optional beam:

.. code-block:: python

    # real space: Sigma_y(r_p), Gaussian beam of this FWHM in arcmin
    sigma_y = cross.projected_gy(rp, z, theta, params, beam_fwhm_arcmin=1.6)

    # harmonic space: C_ell^{g,y}, same window, FWHM in arcsec
    cl_gy   = cross.angular_cl_gy(ell, z_arr, nz_g, theta, params, psf_fwhm_arcsec=96.0)

Both apply the same Gaussian window
:math:`B_\ell = \exp(-\ell^2\sigma^2/2)` (:func:`~hod_mod.observables.cross_spectra.psf_window_ell`);
only the galaxy field is unconvolved, so :math:`B_\ell` enters to the first power.

In :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.projected_gy` the window is
applied to :math:`P_{g,y}(k)` at :math:`\ell = k\,\chi(z)` *before* the Abel projection.  Although
:math:`B(|k|\chi)` is isotropic while a beam is purely transverse, this is exact: the
line-of-sight integral samples the 3D power at :math:`k_z = 0`, where :math:`|k| = k_\perp`.

A fixed *angular* beam maps to a *comoving* scale through :math:`\chi(z)`, so its size varies
across a stack spanning a wide redshift range (for the LS10 BGS samples, :math:`\sigma_{\rm beam}`
runs from 0.04 to 0.15 Mpc/h).  Use
:meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.projected_gy_nz` to average over
n(z) when that matters.

.. important::

   **The projection was corrected in 2026-07.**  :func:`~hod_mod.observables.clustering._pk_to_xi`
   — which every real-space observable goes through, including
   :meth:`~hod_mod.observables.cross_spectra.HaloModelCrossSpectra.projected_gy`,
   ``projected_gX``, w\ :sub:`p` and ΔΣ — was silently a trapezoid truncated at k·r ≈ 8 rather
   than the Ogata double-exponential rule its own header comment documented (a factor 1/h was
   missing from the nodes).  It biased Σ\ :sub:`y` by ~16% and w\ :sub:`p`/ΔΣ by ~19%/20%.

   The corrected rule agrees with the independent FFTLog implementations in ``mcfit`` and
   ``hankl`` to 0.04%/0.26% on a real P\ :sub:`gy`, and reproduces an exact analytic transform
   pair.  **Predictions made before this are superseded.**

Aperture photometry
-------------------

Stacking analyses (Schaan et al. 2021, Amodeo et al. 2021) report a **compensated aperture
photometry** (CAP) statistic rather than the raw profile: a disk minus an equal-area ring, which
cancels any spatially flat component.  To compare against them, filter the *beamed* model:

.. code-block:: python

    from hod_mod.observables import cap_filter

    theta   = np.linspace(1e-3, 12.0, 600)          # arcmin, out past sqrt(2)*theta_d
    prof    = cross.sigma_y_theta(theta, z, theta_cosmo, params, beam_fwhm_arcmin=1.6)
    t_ap    = cap_filter(prof, theta, np.array([1.0, 2.0, 4.0]))   # integrated T_AP

:func:`~hod_mod.observables.cross_spectra.cap_filter` returns the **integrated**
:math:`T_{\rm AP} = 2Y(\theta_d) - Y(\sqrt{2}\theta_d)`, matching Schaan et al. 2021 Eqs.
(10)-(11) — not a mean :math:`y`.

The validation figures (A10 pressure profile, :math:`P_{g,y}(k)` decomposition,
:math:`\Sigma_y(r_p)`, :math:`C_\ell^{g,y}`) are produced by::

    hod-mod validate sz-xray

References
----------

* Arnaud et al. 2010, arXiv:0910.1234 — A10 generalized-NFW pressure profile.
* Amodeo et al. 2021, arXiv:2009.05558 — ACT × BOSS CMASS: gas thermodynamics from tSZ+kSZ.
* Schaan et al. 2021, arXiv:2009.05557 — the companion measurement; stacked CAP profiles.
* Pandey et al. 2025, arXiv:2506.07432 — DES Y3 shear × ACT DR6 tSZ.

API
---

.. autoclass:: hod_mod.observables.cross_spectra.HaloModelCrossSpectra
   :members: projected_gy, projected_gy_nz, sigma_y_theta, angular_cl_gy
   :noindex:
