HOD Fitting Module
==================

The ``hod_mod.fitting`` module provides two-stage HOD fitting of observed galaxy
clustering data: a fast MAP estimate via Nelder-Mead optimisation, followed by an
MCMC posterior exploration with emcee.

---

Statistical framework
---------------------

Given observed :math:`w_p` values :math:`\mathbf{d}` with covariance matrix
:math:`\mathbf{C}`, the log-likelihood under a Gaussian noise model is

.. math::

   \ln\mathcal{L}(\boldsymbol{\theta}) =
   -\frac{1}{2}\left(\mathbf{d} - \mathbf{m}(\boldsymbol{\theta})\right)^T
   \mathbf{C}^{-1}
   \left(\mathbf{d} - \mathbf{m}(\boldsymbol{\theta})\right)

where :math:`\mathbf{m}(\boldsymbol{\theta})` is the model prediction for parameter
vector :math:`\boldsymbol{\theta}`.  In terms of a reduced chi-squared:

.. math::

   \chi^2_\nu = \frac{1}{\nu}
   \left(\mathbf{d} - \mathbf{m}\right)^T \mathbf{C}^{-1}
   \left(\mathbf{d} - \mathbf{m}\right)

where :math:`\nu = N_{\rm data} - N_{\rm free}` is the number of degrees of freedom.

The log-posterior is

.. math::

   \ln P(\boldsymbol{\theta} | \mathbf{d}) =
   \ln\mathcal{L}(\boldsymbol{\theta}) + \ln\pi(\boldsymbol{\theta})

where :math:`\ln\pi(\boldsymbol{\theta})` is the log-prior (see :ref:`priors` below).

Covariance matrix
~~~~~~~~~~~~~~~~~

When data are loaded from a ``sum_stat`` HDF5 file, the full covariance matrix from the
file is used.  A 1% diagonal regularisation is applied before inversion to guard against
numerical singularities:

.. math::

   \mathbf{C}_{\rm reg} = \mathbf{C} + 0.01 \cdot \mathrm{diag}(\mathbf{C})

When data are loaded from a CSV file (legacy), only diagonal errors are available and
:math:`C_{ij} = \sigma_i^2 \delta_{ij}`.

.. _priors:

Prior distributions
-------------------

**Uniform prior** (default for HOD parameters)

For each free parameter :math:`\theta_i` with bounds :math:`[l_i, u_i]`:

.. math::

   \ln\pi(\theta_i) = \begin{cases} 0 & l_i \le \theta_i \le u_i \\ -\infty & \text{otherwise}\end{cases}

**Gaussian prior** (optional, for cosmological parameters)

For each parameter with Gaussian prior :math:`\mathcal{N}(\mu_i, \sigma_i^2)`:

.. math::

   \ln\pi(\theta_i) = \begin{cases}
   -\dfrac{(\theta_i - \mu_i)^2}{2\sigma_i^2} & l_i \le \theta_i \le u_i \\
   -\infty & \text{otherwise}
   \end{cases}

where hard bounds :math:`[l_i, u_i]` are applied in addition.  When
``param_prior_types[name] = "gaussian"`` the fitter adds this term on top of the
chi-squared.

---

Planck 2018 cosmological prior
--------------------------------

(`hod_mod.fitting.planck_prior`)

The Planck 2018 TT,TE,EE+lowE best-fit values and 1σ uncertainties
(`Planck Collaboration 2020 <https://arxiv.org/abs/1807.06209>`_, Table 2) [PlanckCollaboration2018]_ are
encoded in ``PLANCK18_MEANS`` and ``PLANCK18_SIGMAS``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 25 20

   * - Parameter
     - Symbol
     - Best-fit
     - 1σ
     - 3σ range
   * - Hubble constant
     - :math:`h`
     - 0.6736
     - 0.0054
     - [0.6574, 0.6898]
   * - Matter density
     - :math:`\Omega_m`
     - 0.3153
     - 0.0073
     - [0.2934, 0.3372]
   * - Baryon density
     - :math:`\Omega_b`
     - 0.0493
     - 0.0006
     - [0.0475, 0.0511]
   * - Spectral index
     - :math:`n_s`
     - 0.9649
     - 0.0042
     - [0.9523, 0.9775]
   * - Log amplitude
     - :math:`\ln 10^{10}A_s`
     - 3.044
     - 0.014
     - [3.002, 3.086]
   * - :math:`\sigma_8`
     - :math:`\sigma_8`
     - 0.8111
     - 0.0060
     - [0.7931, 0.8291]

The 3σ bounds are used as hard truncation limits for the Gaussian priors.  The
function ``planck18_log_prior(theta)`` returns the sum of all Gaussian log-prior terms;
it returns :math:`-\infty` if any parameter is outside its 3σ range.

Usage in fitting scripts:

.. code-block:: python

    from hod_mod.fitting.planck_prior import PLANCK18_MEANS, PLANCK18_SIGMAS
    from hod_mod.fitting.hod_wp import WpFitConfig, WpFitter

    cfg = WpFitConfig(
        ...
        param_prior_types  = {"h": "gaussian", "Omega_m": "gaussian"},
        param_prior_means  = {"h": PLANCK18_MEANS["h"],
                              "Omega_m": PLANCK18_MEANS["Omega_m"]},
        param_prior_sigmas = {"h": PLANCK18_SIGMAS["h"],
                              "Omega_m": PLANCK18_SIGMAS["Omega_m"]},
    )

---

MAP estimation
--------------

``WpFitter.map_fit()`` minimises :math:`-\ln P` using the Nelder-Mead simplex
algorithm (via ``scipy.optimize.minimize``).  Nelder-Mead is gradient-free and robust
to the discontinuous derivatives that arise from hard prior bounds.  The result is the
MAP (maximum a posteriori) point estimate.

A good starting point is critical: ``param_init`` in the config should be set to a
physically plausible value (e.g. set ``log10mmin`` close to the expected characteristic
halo mass for the sample).

---

MCMC posterior sampling
-----------------------

``WpFitter.mcmc_fit()`` uses the `emcee <https://emcee.readthedocs.io>`_ ensemble
sampler (`Foreman-Mackey et al. 2013 <https://arxiv.org/abs/1202.3665>`_) [Foreman-Mackey2013]_.  The
default configuration uses 64 walkers initialised in a Gaussian ball around the MAP
estimate.

Convergence diagnostics:

* **Acceptance fraction**: should be between 0.2 and 0.5.  If too low, the proposal
  scale ``moves`` parameter needs tuning.
* **Integrated autocorrelation time** :math:`\hat{\tau}`: the chain is considered
  converged when the number of steps exceeds :math:`50\hat{\tau}`.  Access via
  ``sampler.get_autocorr_time()``.
* **Gelman-Rubin statistic** :math:`\hat{R}`: for multi-chain runs, :math:`\hat{R} < 1.1`
  indicates convergence.

---

Configuration
-------------

Fitting is driven by a ``WpFitConfig`` dataclass (or equivalently a YAML file parsed
by ``load_config``):

.. code-block:: yaml

    data_file:    /path/to/data.h5
    data_format:  hdf5          # "csv" or "hdf5"
    rp_min:       0.3           # Mpc/h
    rp_max:       30.0          # Mpc/h
    hod_model:    MoreHODModel
    hmf_backend:  csst           # pipeline baseline (default if omitted); use
                                  # tinker08 to reproduce literature results
    z:            0.15
    pi_max:       100.0         # Mpc/h

    free_params:    [log10mmin, sigma_logm, log10m1, alpha]
    param_bounds:
      log10mmin:  [11.0, 13.5]
      sigma_logm: [0.1,  1.0]
      log10m1:    [12.0, 15.0]
      alpha:      [0.5,  2.0]
    param_init:
      log10mmin:  12.5
      sigma_logm: 0.38
      log10m1:    13.5
      alpha:      1.0
      kappa:      1.0
      alpha_inc:  1.0
      log10m_inc: 12.0

    # Optional: Gaussian cosmological priors
    param_prior_types:  {h: gaussian}
    param_prior_means:  {h: 0.6736}
    param_prior_sigmas: {h: 0.0054}

    output_dir: results/bgs_ls10/mstar10.5/

---

Usage example
-------------

.. code-block:: python

    from hod_mod.fitting.hod_wp import WpFitter, load_config

    cfg    = load_config("configs/hod_fit_more2015_cmass.yml")
    fitter = WpFitter(cfg)
    result = fitter.map_fit()           # Nelder-Mead MAP estimate
    print(result.x, result.fun)

    sampler = fitter.mcmc_fit()         # emcee MCMC posterior
    flat    = sampler.get_chain(flat=True, discard=100, thin=5)

.. automodule:: hod_mod.fitting.hod_wp
   :members:
   :undoc-members:

.. automodule:: hod_mod.fitting.planck_prior
   :members:
   :undoc-members:

---

.. rubric:: Key references

Lensing and galaxy–matter cross-correlation:
[BartelmannSchneider2001]_, [Mandelbaum2005]_, [Mandelbaum2006]_, [Leauthaud2017]_,
[Miyatake2022]_, [Lange2023]_, [Heydenreich2025]_, [Lange2025]_.

Intrinsic alignments:
[Catelan2001]_, [HirataSeljak2004]_, [Brown2002]_, [BridleKing2007]_,
[Blazek2019]_, [DESI_KP6]_.

Surveys:
[Blanton2003]_, [BOSS_CMASS]_, [HSC_Aihara2018]_, [HSC_Mandelbaum2018]_,
[KiDS_Heymans2021]_, [DES_Abbott2022]_, [DESI_EDR]_, [DESI_BGS_Hahn2023]_,
[Comparat2023]_, [Lange2024]_, [Lange2025phz]_.

Inference:
[Phan2019]_.
