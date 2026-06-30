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

The validation figures (A10 pressure profile, :math:`P_{g,y}(k)` decomposition,
:math:`\Sigma_y(r_p)`, :math:`C_\ell^{g,y}`) are produced by::

    hod-mod validate sz-xray

References
----------

* Arnaud et al. 2010, arXiv:0910.1234 — A10 generalized-NFW pressure profile.
* Amodeo et al. 2021, arXiv:2009.05557 — ACT × BOSS CMASS/LOWZ stacked tSZ.
* Pandey et al. 2025, arXiv:2506.07432 — DES Y3 shear × ACT DR6 tSZ.

API
---

.. autoclass:: hod_mod.observables.cross_spectra.HaloModelCrossSpectra
   :members: projected_gy, angular_cl_gy
   :noindex:
