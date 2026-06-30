.. _fitting_more2015:

Fitting with the More+2015 HOD Model
=====================================

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

``hod_mod/scripts/fitting/run_fit_More15.py`` (also ``hod-mod fit``) is a command-line tool for fitting
the :class:`~hod_mod.observables.clustering.FullHaloModelPrediction` forward model to
observational data using the `More et al. (2015) <https://arxiv.org/abs/1407.1856>`_
Halo Occupation Distribution (HOD) model.

The script is a thin wrapper around the validated fitting classes in
:mod:`hod_mod.fitting`:

- :class:`~hod_mod.fitting.WpFitter` — projected clustering :math:`w_p(r_p)` only
- :class:`~hod_mod.fitting.DeltaSigmaFitter` — excess surface density :math:`\Delta\Sigma(R)` only
- :class:`~hod_mod.fitting.JointFitter` — joint :math:`w_p + \Delta\Sigma + n_\mathrm{gal}`

The probe mode is detected automatically from the YAML configuration (see `Probe Modes`_).

Quick Start
-----------

**Step 1 — Write a configuration file** (see examples in ``configs/fitting/``)::

    label: "My galaxy sample"
    data:
      file:   path/to/wp_data.csv
      format: bwpd
      rp_min: 0.5
      rp_max: 50.0
    cosmology:
      Omega_m: 0.310
      h:       0.700
      sigma8:  0.800
      n_s:     0.965
      Omega_b: 0.045
    model:
      hod_model:   MoreHODModel
      hmf_backend: tinker08
      z:           0.50
      pi_max:      100.0
    parameters:
      log10mmin:  {free: true,  bounds: [11.0, 15.0], init: 12.5}
      sigma_logm: {free: true,  bounds: [0.05, 1.50], init: 0.4}
      log10m1:    {free: true,  bounds: [12.0, 15.5], init: 13.5}
      alpha:      {free: true,  bounds: [0.50, 2.50], init: 1.0}
      kappa:      {free: true,  bounds: [0.01, 3.00], init: 1.0}
    fitting:
      method: map
    output:
      dir: results/my_fit/

**Step 2 — Run**::

    python hod_mod/scripts/fitting/run_fit_More15.py  config.yml --map-only

**Step 3 — Check outputs** in ``results/my_fit/``::

    fit_result.json      # best-fit params, χ²/dof
    fit_wp.png           # wp data vs best-fit
    fit_combined.png     # rp·wp panel
    fit_hod.png          # N_c(M), N_s(M) occupation

YAML Configuration Reference
------------------------------

.. _fitting-yaml-data:

``data:`` block (wp)
~~~~~~~~~~~~~~~~~~~~~

Required when fitting :math:`w_p(r_p)`.

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``file``
     - string
     - —
     - Path to :math:`w_p` data file (CSV, HDF5, or FITS).
   * - ``format``
     - string
     - —
     - Data format. ``bwpd`` (More+2015-style three-column), ``hdf5``, ``fits``.
   * - ``rp_min``
     - float
     - 0.0
     - Minimum :math:`r_p` (:math:`h^{-1}` Mpc) to include in fit.
   * - ``rp_max``
     - float
     - ∞
     - Maximum :math:`r_p` (:math:`h^{-1}` Mpc) to include in fit.

``joint:`` block (ESD + n_gal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include to activate joint :math:`w_p + \Delta\Sigma + n_\mathrm{gal}` fitting.
Omitting this block makes the fit wp-only.

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``ds_file``
     - string
     - —
     - Path to :math:`\Delta\Sigma` data file.
   * - ``ds_format``
     - string
     - —
     - ESD format. ``bwpd_4col`` (R, DS, upper, lower), ``hdf5``.
   * - ``ds_rp_min``
     - float
     - 0.0
     - Minimum :math:`R` (:math:`h^{-1}` Mpc) for ESD.
   * - ``ds_rp_max``
     - float
     - ∞
     - Maximum :math:`R` (:math:`h^{-1}` Mpc) for ESD.
   * - ``ng_obs``
     - float
     - —
     - Observed galaxy number density :math:`h^3` Mpc:math:`^{-3}`.
   * - ``ng_frac_err``
     - float
     - 0.20
     - Fractional error on :math:`n_\mathrm{gal}` (Gaussian term in likelihood).

``ds:`` block (ESD-only mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ds:`` instead of ``joint:`` to fit :math:`\Delta\Sigma` only (without :math:`w_p`).
The fields are identical to the ``joint:`` block above.

``cosmology:`` block
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Field
     - Type
     - Required
     - Description
   * - ``Omega_m``
     - float
     - yes
     - Total matter density parameter at z=0.
   * - ``h``
     - float
     - yes
     - Dimensionless Hubble constant (:math:`H_0 = 100\,h` km/s/Mpc).
   * - ``sigma8``
     - float
     - yes
     - RMS matter fluctuations at 8 :math:`h^{-1}` Mpc.
   * - ``n_s``
     - float
     - yes
     - Primordial spectral index.
   * - ``Omega_b``
     - float
     - yes
     - Baryon density parameter.

``model:`` block
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``hod_model``
     - string
     - ``MoreHODModel``
     - HOD model class. Use ``MoreHODModel`` for More+2015.
   * - ``hmf_backend``
     - string
     - ``tinker08``
     - Halo mass function backend. ``tinker08`` recommended.
   * - ``z``
     - float
     - —
     - Galaxy sample effective redshift.
   * - ``pi_max``
     - float
     - 100.0
     - Line-of-sight integration limit :math:`\pi_\mathrm{max}` (Mpc/:math:`h`).
   * - ``use_bnl``
     - bool
     - ``false``
     - Enable beyond-linear halo bias (Mead & Verde 2021). Negligible at :math:`r_p > 1\,h^{-1}` Mpc.

``parameters:`` block
~~~~~~~~~~~~~~~~~~~~~~

Each entry: ``name: {free: bool, bounds: [lo, hi], init: value}``.

If ``free: false``, only ``init`` is required (the parameter is held fixed).

``fitting:`` block
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``method``
     - string
     - ``map``
     - ``map`` — MAP only; ``mcmc`` — MCMC only (starts at ``init``); ``both`` — MAP then MCMC.
   * - ``n_walkers``
     - int
     - 32
     - Number of emcee walkers.
   * - ``n_steps``
     - int
     - 2000
     - Number of MCMC steps per walker.
   * - ``n_burnin``
     - int
     - 500
     - Burn-in steps discarded before saving the flat chain.

``output:`` block
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``dir``
     - string
     - ``results/fit/``
     - Directory for all output files.

``label:`` (top-level)
~~~~~~~~~~~~~~~~~~~~~~~

Optional string identifying this fit in figure titles and JSON output.
Defaults to the YAML filename stem if absent.

``published_params:`` (top-level, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dictionary mapping parameter names to ``[best-fit, error]`` pairs.
When present, a dashed reference curve is overlaid on all data comparison plots,
and MCMC corner plots mark the reference values as vertical/horizontal lines.

Example::

    published_params:
      log10mmin:  [13.13, 0.13]
      sigma_logm: [0.469, 0.13]
      log10m1:    [14.21, 0.12]
      alpha:      [1.130, 0.09]
      kappa:      [1.250, 0.40]

Probe Modes
-----------

The fitter class is selected automatically based on which data sections appear in the YAML:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Probes in config
     - Fitter class
     - Likelihood
   * - ``data:`` only
     - :class:`~hod_mod.fitting.WpFitter`
     - :math:`\chi^2_{w_p}`
   * - ``ds:`` only (no ``data:``)
     - :class:`~hod_mod.fitting.DeltaSigmaFitter`
     - :math:`\chi^2_{\Delta\Sigma}`
   * - ``data:`` + ``joint:``
     - :class:`~hod_mod.fitting.JointFitter`
     - :math:`\chi^2_{w_p} + \chi^2_{\Delta\Sigma} + \chi^2_{n_\mathrm{gal}}`

Covariance Options
------------------

Diagonal covariance (CSV / ``bwpd`` format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``data.file`` is a CSV with ``bwpd`` or ``bwpd_4col`` format, the
covariance is diagonal: :math:`C_{ii} = \sigma_i^2` from the error column.
This is appropriate when data points are independently measured or when
off-diagonal correlations are negligible.

Full covariance (HDF5/FITS jackknife)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the data file is an HDF5/FITS jackknife catalogue, :func:`~hod_mod.fitting.load_config`
builds the full jackknife covariance matrix and applies the
`Hartlap et al. (2007) <https://arxiv.org/abs/astro-ph/0608064>`_ correction:

.. math::

   \hat{C}^{-1} = \frac{n_\mathrm{jk} - n_\mathrm{bins} - 2}{n_\mathrm{jk} - 1}\, C^{-1}_\mathrm{JK}

To use jackknife data, provide either::

    data:
      file:   path/to/wp_jackknife.h5
      format: hdf5

or the jackknife patch directory::

    fits:
      jk_dir:     data/my_survey/jk_patches/
      jk_pattern: NSIDE_04
      h:          0.6736

Command-Line Options
---------------------

.. code-block:: text

    python hod_mod/scripts/fitting/run_fit_More15.py  <config.yml>  [options]

    Positional argument:
      config              Path to YAML configuration file.

    Options:
      --map-only          Run MAP optimisation only (skip MCMC even if config says
                          method=both or method=mcmc).
      --mcmc              Run MAP then MCMC sampling (overrides method=map in config).
      --plot-only         Skip fitting; reload fit_result.json and regenerate figures.
      --output-dir DIR    Override the output directory from the config.

Output Files
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``fit_result.json``
     - JSON with label, probes, chi2, ndof, chi2/ndof, success flag, all parameter values.
   * - ``flatchain.npz``
     - MCMC flat chain (after burn-in). Arrays: ``flatchain`` (N×n_free), ``param_names``.
   * - ``fit_wp.png``
     - :math:`w_p(r_p)` data vs MAP prediction with residuals; MCMC bands if available.
   * - ``fit_ds.png``
     - :math:`\Delta\Sigma(R)` data vs MAP prediction with residuals (joint/ESD modes).
   * - ``fit_combined.png``
     - :math:`r_p\,w_p` and :math:`\Delta\Sigma` side by side with residuals.
   * - ``fit_hod.png``
     - :math:`N_c(M)` and :math:`N_s(M)` at MAP; MCMC occupation credible bands if available.
   * - ``fit_corner.png``
     - MCMC corner plot with 1σ/2σ contours; reference values from ``published_params:`` overlaid.

Parameter Reference
--------------------

More+2015 HOD Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

The More et al. (2015) model has five free parameters describing central and
satellite occupation:

.. list-table::
   :header-rows: 1
   :widths: 16 18 66

   * - Name
     - Units
     - Physical meaning
   * - ``log10mmin``
     - :math:`\log_{10}(M_\odot/h)`
     - Halo mass scale at which 50% of halos host a central galaxy.
       Controls the overall HOD amplitude and galaxy number density.
   * - ``sigma_logm``
     - dex
     - Width of the central occupation step function (log-normal scatter in SHMR).
       Larger values lower the effective halo mass threshold.
   * - ``log10m1``
     - :math:`\log_{10}(M_\odot/h)`
     - Characteristic satellite mass scale — halos above :math:`M_1` host on average one satellite.
   * - ``alpha``
     - —
     - Slope of the satellite mean occupation power law :math:`N_s \propto (M/M_1)^\alpha`.
   * - ``kappa``
     - —
     - Threshold multiplicative factor: satellites only occupy halos above :math:`\kappa\,M_\mathrm{min}`.

Off-centering Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed by default to the More+2015 published MAP values; can be freed.

.. list-table::
   :header-rows: 1
   :widths: 16 18 66

   * - Name
     - Units
     - Physical meaning
   * - ``p_off``
     - —
     - Fraction of central galaxies miscentred from the halo centre.
   * - ``R_off``
     - :math:`r_s` units
     - Off-centering radial scale in units of the NFW scale radius :math:`r_s(M)`.
       A value of 2.2 means off-centred galaxies are displaced by :math:`2.2\,r_s` on average.

Incompleteness Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``free: false`` and ``init: 1.0`` / ``13.0`` to disable incompleteness correction.

.. list-table::
   :header-rows: 1
   :widths: 16 18 66

   * - Name
     - Units
     - Physical meaning
   * - ``alpha_inc``
     - —
     - Incompleteness power-law slope. ``1.0`` = no incompleteness.
   * - ``log10m_inc``
     - :math:`\log_{10}(M_\odot/h)`
     - Halo mass scale for incompleteness onset.

Example: wp-only MAP
---------------------

Configuration file ``configs/fitting/More15_wp_example.yml``:

.. literalinclude:: ../configs/fitting/More15_wp_example.yml
   :language: yaml

Run::

    python hod_mod/scripts/fitting/run_fit_More15.py \
        configs/fitting/More15_wp_example.yml --map-only

Expected output::

    ============================================================
    run_fit_More15  [More+2015 BOSS CMASS logM11 — wp-only MAP]
      Config:   configs/fitting/More15_wp_example.yml
      Probes:   wp
      Free params (5): ['log10mmin', 'sigma_logm', 'log10m1', 'alpha', 'kappa']
      Output:   results/fitting/More15_wp_example/

    ============================================================
    Fit: More+2015 BOSS CMASS logM11 — wp-only MAP
    ============================================================
      chi2 / ndof = 42.892 / 22  →  chi2/dof = 1.950
      Optimizer:  OK

      Best-fit parameters:
        log10mmin            =   13.191  (free)
        sigma_logm           =   0.5110  (free)
        log10m1              =   14.196  (free)
        alpha                =   1.8605  (free)
        kappa                =   2.6753  (free)
        alpha_inc            =      1.0
        log10m_inc           =     13.0
        p_off                =     0.34
        R_off                =      2.2

    Result saved → results/fitting/More15_wp_example/fit_result.json

    === Generating figures ===
      Saved: results/fitting/More15_wp_example/fit_wp.png
      ...

Example: joint MAP + MCMC
--------------------------

Configuration file ``configs/fitting/More15_joint_example.yml``:

.. literalinclude:: ../configs/fitting/More15_joint_example.yml
   :language: yaml

Run MAP then MCMC sampling::

    python hod_mod/scripts/fitting/run_fit_More15.py \
        configs/fitting/More15_joint_example.yml --mcmc

Or run MAP only first to check convergence, then add MCMC::

    # Quick MAP check
    python hod_mod/scripts/fitting/run_fit_More15.py \
        configs/fitting/More15_joint_example.yml --map-only

    # Regenerate figures from saved result
    python hod_mod/scripts/fitting/run_fit_More15.py \
        configs/fitting/More15_joint_example.yml --plot-only

The joint likelihood is:

.. math::

   -2\ln\mathcal{L} = \chi^2_{w_p} + \chi^2_{\Delta\Sigma} + \chi^2_{n_\mathrm{gal}}

where

.. math::

   \chi^2_{n_\mathrm{gal}} = \left(\frac{n_\mathrm{gal}^\mathrm{pred} - n_\mathrm{gal}^\mathrm{obs}}
                              {f_\mathrm{err}\,n_\mathrm{gal}^\mathrm{obs}}\right)^2

References
----------

- More, S., van den Bosch, F. C., Cacciato, M., et al. 2015, ApJ, 806, 2.
  `arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_
- Tinker, J. L., Kravtsov, A. V., Klypin, A., et al. 2008, ApJ, 688, 709.
  `arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_
- Hartlap, J., Simon, P., & Schneider, P. 2007, A&A, 464, 399.
  `arXiv:astro-ph/0608064 <https://arxiv.org/abs/astro-ph/0608064>`_
- Mead, A. J., & Verde, L. 2021, MNRAS, 503, 3095.
  `arXiv:2009.10724 <https://arxiv.org/abs/2009.10724>`_
