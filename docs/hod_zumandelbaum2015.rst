.. _hod_zumandelbaum2015:

Zu & Mandelbaum 2015 iHOD Model — SDSS, X-ray & BGS
======================================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - :class:`~hod_mod.galaxies.hod.ZuMandelbaum15HODModel`
   * - **Paper**
     - Zu & Mandelbaum 2015, MNRAS 454, 1161
       (`arXiv:1505.02781 <https://arxiv.org/abs/1505.02781>`_)
   * - **SHMR basis**
     - Behroozi, Converse & Wechsler 2010, ApJ 717, 379
       (`arXiv:1001.0015 <https://arxiv.org/abs/1001.0015>`_)
   * - **Survey (benchmark)**
     - SDSS DR7, :math:`\log_{10}(M_*/h^{-2}M_\odot) > 10.2`,
       :math:`z_\mathrm{eff} \approx 0.1`
   * - **Observable (benchmark)**
     - Joint :math:`w_p(r_p) + \Delta\Sigma(R)`,
       :math:`\pi_\mathrm{max} = 60\,h^{-1}\,\mathrm{Mpc}`,
       11 bins each (:math:`r_p,\,R \in [0.05,\,20]\,h^{-1}\,\mathrm{Mpc}`)
   * - **X-ray extension**
     - BGS LS10 × eROSITA (0.5–2 keV),
       Comparat et al. 2025 (`arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_)
   * - **Code**
     - :mod:`hod_mod.galaxies.hod` (lines 1431–1735),
       :mod:`hod_mod.galaxies.clustering`,
       :mod:`hod_mod.scripts.fitting.fit_comparat2025`

----

Cosmological framework
-----------------------

Both HOD models share the same backbone; see also :ref:`hod_more2015`.

.. rubric:: Cosmological parameters

Six base parameters :math:`\boldsymbol{\theta} = (\Omega_m,\,\Omega_b,\,h,\,n_s,\,\ln 10^{10}A_s,\,\sigma_8)`
define the linear matter power spectrum :math:`P_\mathrm{lin}(k, z)` via
`CAMB <https://camb.readthedocs.io>`_ (:class:`~hod_mod.cosmology.LinearPowerSpectrum`).

Benchmark cosmology (ZM15-specific):
:math:`\Omega_m = 0.260,\ h = 0.720,\ \sigma_8 = 0.770,\ n_s = 0.960,\ \Omega_b = 0.044`.

BGS/LS10 X-ray fits use Planck 2018:
:math:`\Omega_m = 0.315,\ h = 0.674,\ \sigma_8 = 0.811,\ n_s = 0.965`.

.. rubric:: Halo mass function

Tinker et al. 2008 (`arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_),
:math:`\Delta = 200\rho_m`, via
:func:`~hod_mod.cosmology.halo_mass_function.make_hmf`
with ``backend="tinker08"`` (default).

Alternative emulator backends — ``"csst"`` (Chen+2025,
`SCPMA 2025 <https://ui.adsabs.harvard.edu/abs/2025SCPMA..6809513C>`_)
and ``"aemulusnu"`` (Shen+2025,
`arXiv:2410.00913 <https://arxiv.org/abs/2410.00913>`_) — expose the
same interface; see :doc:`cosmology` for details.

.. note::

   The BGS×eROSITA fit (:mod:`hod_mod.scripts.fitting.fit_comparat2025`) uses
   the **CSST emulator** HMF by default, and the *same* HMF instance is reused by
   the AGN models (``HamAGNModel``/``HODAgnModel`` accept an ``hmf=`` argument)
   so the galaxy clustering and the AGN abundance-match share one consistent
   mass function rather than each defaulting to Tinker08.

.. rubric:: Linear halo bias

Tinker et al. 2010 (`arXiv:1001.3162 <https://arxiv.org/abs/1001.3162>`_), with
the **beyond-linear halo bias** correction of Mead & Verde 2021
(`arXiv:2011.08858 <https://arxiv.org/abs/2011.08858>`_;
:class:`~hod_mod.cosmology.beyond_linear_bias.BeyondLinearBiasMead21`) applied
to the 2-halo galaxy terms.  The BGS×eROSITA fit passes this ``bnl_model`` into
:class:`~hod_mod.galaxies.clustering.FullHaloModelPrediction` so the non-linear
scale-dependent bias is used consistently in :math:`w_p` and :math:`w_\theta`.
Effective bias:

.. math::

   b_\mathrm{eff}(z) =
   \frac{\displaystyle\int \mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}\,
         \langle N_\mathrm{tot}(M)\rangle\,b(M,z)}
        {\bar{n}_g}

.. rubric:: NFW profile and Fourier transform

NFW profile (Navarro, Frenk & White 1997,
`arXiv:astro-ph/9508025 <https://arxiv.org/abs/astro-ph/9508025>`_);
Fourier transform from Cooray & Sheth 2002
(`arXiv:astro-ph/0206508 <https://arxiv.org/abs/astro-ph/0206508>`_),
concentration–mass from Diemer & Joyce 2019
(`arXiv:1809.07326 <https://arxiv.org/abs/1809.07326>`_).

----

Zu & Mandelbaum 2015 iHOD model
---------------------------------

Unlike traditional HOD models that parametrise :math:`N(M_h)` directly, the
*inverse HOD* (iHOD) derives the halo occupation from the stellar-to-halo
mass relation (SHMR) by inversion: the mean stellar mass
:math:`M_*^\mathrm{c}(M_h)` is obtained by inverting the SHMR, and the
occupation functions follow from the log-normal scatter around that mean.
This is equivalent to the conventional HOD (cHOD) at the 1–2% level for
:math:`w_p` and :math:`\Delta\Sigma` (verified in
``results/benchmarks/zumandelbaum2015_sdss/comparison_ihod_chod_wp_ds.png``).

Implementation: :class:`~hod_mod.galaxies.hod.ZuMandelbaum15HODModel`
whose :meth:`~hod_mod.galaxies.hod.ZuMandelbaum15HODModel.nc_ns` method
returns :math:`(N_c,\,N_s)` arrays consumed by
:class:`~hod_mod.galaxies.clustering.FullHaloModelPrediction`.

.. admonition:: Pedagogical figures and tutorial notebook
   :class: tip

   Each equation below is illustrated by a figure produced by
   ``hod_mod/scripts/benchmarks/illustrate_zumandelbaum2015_equations.py``
   (run it to regenerate the panels, or pass ``--to-notebook`` to export the
   interactive tutorial ``notebooks/zumandelbaum2015_equations.ipynb``). The
   occupation equations (Eqs. 19–22) are cosmology-independent and additionally
   show their analytic ``jax.grad`` parameter sensitivity; the clustering
   observables (:math:`b_\mathrm{eff}`, :math:`P(k)`, :math:`w_p`,
   :math:`\Delta\Sigma`) carry a bottom panel with the **logarithmic
   sensitivity** :math:`d\ln O/d\ln p` to :math:`\Omega_m`, :math:`\sigma_8`
   and :math:`h` (central finite differences, since CAMB lies outside the JAX
   graph), with the top panel overlaying :math:`+2\%` variants.

.. rubric:: Step 1 — SHMR forward direction (ZM15 Eq. 19)

The SHMR is written as **halo mass as a function of stellar mass**
(Behroozi et al. 2010, ZM15 Eq. 19):

.. math::

   \log_{10} M_h(M_*) = \log_{10} M_1
   + \beta\,\log_{10}\!\frac{M_*}{M_{*,0}}
   + \frac{1}{\ln 10}\!\left[
       \frac{(M_*/M_{*,0})^\delta}{1 + (M_*/M_{*,0})^{-\gamma}}
       - \frac{1}{2}
     \right]

where :math:`M_1 = 10^{\mathtt{lg\_m1h}}` and
:math:`M_{*,0} = 10^{\mathtt{lg\_m0star}}` are the pivot halo and stellar
masses.
Implemented in :func:`~hod_mod.galaxies.hod._mh_from_mstar_zu15`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq01_shmr_forward.png
   :width: 90%
   :align: center

   SHMR forward direction (Eq. 19). Sweeping the low-mass slope
   :math:`\beta` and transition sharpness :math:`\gamma` pivots the relation
   about :math:`(M_{*,0}, M_1)`. Lower panel: the analytic
   :math:`\partial\log_{10}M_h/\partial\beta` from ``jax.grad``.

.. rubric:: Step 2 — SHMR inversion: mean stellar mass at fixed :math:`M_h`

The iHOD requires :math:`M_*(M_h)` — the mean stellar mass of a central
galaxy in a halo of mass :math:`M_h`.  Because Eq. 19 has no closed-form
inverse, it is numerically inverted by 60 iterations of JAX-compatible
bisection over :math:`\log_{10}(M_*/M_\odot) \in [4, 13]`:

.. math::

   M_*^\mathrm{c}(M_h) = \mathrm{SHMR}^{-1}(M_h)

implemented in :func:`~hod_mod.galaxies.hod._mstar_from_mh_zu15`.
The 60-iteration bisection converges to :math:`\lesssim 10^{-17}` dex
precision; in practice the output is accurate to machine precision for any
:math:`M_h \in [10^{10},\,10^{15.5}]\,M_\odot/h`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq02_shmr_inverse.png
   :width: 90%
   :align: center

   SHMR inversion. The mean central stellar mass
   :math:`M_*^\mathrm{c}(M_h)` from the JAX bisection. Lower panel: the
   forward-of-inverse round-trip residual sits at the :math:`\sim10^{-14}` dex
   floor, confirming machine-precision inversion.

.. rubric:: Step 3 — Mass-dependent scatter in :math:`\ln M_*` (ZM15 Eq. 20)

The log-normal scatter in stellar mass at fixed halo mass varies with
:math:`M_h`:

.. math::

   \sigma_{\ln M_*}(M_h) =
   \begin{cases}
   \sigma_{\ln m_*} & M_h \leq M_1 \\
   \sigma_{\ln m_*} + \eta\,\log_{10}(M_h/M_1) & M_h > M_1
   \end{cases}

:math:`\eta < 0` encodes decreasing scatter toward cluster-scale halos
(ZM15 best fit :math:`\eta = -0.04`).
Implemented in :func:`~hod_mod.galaxies.hod.sigma_lnmstar_zu15`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq03_scatter.png
   :width: 90%
   :align: center

   Mass-dependent scatter (Eq. 20). The scatter is flat below the pivot
   :math:`M_1` and tilts with slope :math:`\eta` above it. Lower panel:
   :math:`\partial\sigma/\partial\eta` from ``jax.grad`` switches on exactly at
   :math:`M_1`.

.. rubric:: Step 4 — Central occupation (ZM15 Eq. 21)

:math:`M_{*,\mathrm{th}}` is the **stellar-mass threshold** that defines
the galaxy sample: only galaxies with stellar mass
:math:`M_* \geq M_{*,\mathrm{th}}` are counted.
It is set by the parameter ``log10m_star_thresh``
(:math:`= \log_{10}(M_{*,\mathrm{th}}/M_\odot)`).

**Key iHOD step.** Assuming :math:`\ln M_*\,|\,M_h` is Gaussian
with mean :math:`\ln M_*^\mathrm{c}(M_h)` and standard deviation
:math:`\sigma_{\ln M_*}(M_h)`, the probability that the central galaxy
satisfies :math:`M_* > M_{*,\mathrm{th}}` is:

.. math::

   \langle N_\mathrm{cen}(M_h \,|\, M_{*,\mathrm{th}})\rangle =
   \frac{f_c}{2}\,
   \mathrm{erfc}\!\left[
     \frac{\ln M_{*,\mathrm{th}} - \ln M_*^\mathrm{c}(M_h)}
          {\sqrt{2}\,\sigma_{\ln M_*}(M_h)}
   \right]

:math:`f_c \leq 1` is a central-galaxy completeness fraction
(see note below).  At the characteristic halo mass
:math:`M_h = M_\mathrm{min} \equiv \mathrm{SHMR}^{-1}(M_{*,\mathrm{th}})`,
the argument of erfc vanishes and
:math:`\langle N_\mathrm{cen}\rangle = f_c/2`.
Implemented in :func:`~hod_mod.galaxies.hod.n_cen_thresh_zu15`.

.. note::

   In the original ZM15 paper, :math:`f_c` is defined as the *satellite
   concentration ratio* (:math:`c_\mathrm{sat} = f_c\,c_\mathrm{dm}`,
   Table 2, Section 4.4) and does **not** appear in
   :math:`\langle N_\mathrm{cen}\rangle`.  This implementation repurposes
   :math:`f_c` as a *central-galaxy completeness fraction* — an extension
   useful for flux-limited samples.  The ZM15 best-fit value (0.86) is
   adopted as the default.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq04_central.png
   :width: 90%
   :align: center

   Central occupation (Eq. 21). Raising the threshold
   :math:`M_{*,\mathrm{th}}` shifts the erfc step to higher halo mass; the grey
   markers sit at :math:`f_c/2` at :math:`M_\mathrm{min}`. Lower panel: the
   ``jax.grad`` threshold sensitivity peaks at the transition mass.

.. rubric:: Step 5 — Satellite mass scales (ZM15 Eq. 22 auxiliary)

The characteristic halo mass :math:`M_\mathrm{min}` for the threshold
sample is obtained **directly from the SHMR** (no iteration needed here —
the forward direction Eq. 19 is used):

.. math::

   \log_{10} M_\mathrm{min} =
   \mathtt{\_mh\_from\_mstar\_zu15}(\log_{10} M_{*,\mathrm{th}})

Then the satellite mass scale and cut-off mass follow:

.. math::

   M_\mathrm{sat} &= B_\mathrm{sat}
     \left(\frac{M_\mathrm{min}}{10^{12}\,h^{-1}M_\odot}\right)^{\!\beta_\mathrm{sat}}
     \times 10^{12}\,h^{-1}M_\odot \\[4pt]
   M_\mathrm{cut} &= B_\mathrm{cut}
     \left(\frac{M_\mathrm{min}}{10^{12}\,h^{-1}M_\odot}\right)^{\!\beta_\mathrm{cut}}
     \times 10^{12}\,h^{-1}M_\odot

computed inside :func:`~hod_mod.galaxies.hod.n_sat_thresh_zu15` before the
power-law occuption is evaluated.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq05_satellite_scales.png
   :width: 90%
   :align: center

   Satellite mass scales (Step 5). :math:`M_\mathrm{sat}` (per-satellite
   mass) and :math:`M_\mathrm{cut}` (truncation mass) as functions of the
   threshold, tied to :math:`M_\mathrm{min}` through the SHMR; dashed curves
   vary :math:`B_\mathrm{sat}` and :math:`\beta_\mathrm{cut}`.

.. rubric:: Step 6 — Satellite occupation (ZM15 Eq. 22)

.. math::

   \langle N_\mathrm{sat}(M_h \,|\, M_{*,\mathrm{th}})\rangle =
   \langle N_\mathrm{cen}(M_h)\rangle \times
   \left(\frac{M_h}{M_\mathrm{sat}}\right)^{\!\alpha_\mathrm{sat}}
   \exp\!\left(-\frac{M_\mathrm{cut}}{M_h}\right)

The satellite occupation inherits :math:`\langle N_\mathrm{cen}\rangle` as
a prefactor, so it vanishes for halos that are too light to host a central
galaxy above the threshold.
Implemented in :func:`~hod_mod.galaxies.hod.n_sat_thresh_zu15`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq06_satellite.png
   :width: 90%
   :align: center

   Satellite and total occupation (Eq. 22). Satellites inherit the central
   prefactor (vanishing in light halos), rise as a power law of slope
   :math:`\alpha_\mathrm{sat}`, and are cut below :math:`M_\mathrm{cut}`. Lower
   panel: the ``jax.grad`` :math:`\alpha_\mathrm{sat}` sensitivity grows toward
   massive halos.

.. rubric:: Step 7 — Stellar-mass bin HOD (used in joint fit)

When fitting **stellar-mass bins** rather than threshold samples (as in
:mod:`hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint`), the bin HOD
is computed by subtraction of two threshold HODs at the bin edges
:math:`[M_{*,\mathrm{lo}},\,M_{*,\mathrm{hi}})`:

.. math::

   \langle N(M_h \,|\, M_{*,\mathrm{lo}} \leq M_* < M_{*,\mathrm{hi}})\rangle =
   \langle N(M_h \,|\, M_{*,\mathrm{th}} = M_{*,\mathrm{lo}})\rangle -
   \langle N(M_h \,|\, M_{*,\mathrm{th}} = M_{*,\mathrm{hi}})\rangle

This is activated by passing ``log10m_star_max`` in ``hod_params`` to
:meth:`~hod_mod.galaxies.hod.ZuMandelbaum15HODModel.nc_ns`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq07_bin_hod.png
   :width: 90%
   :align: center

   Stellar-mass-bin HOD (Step 7). The occupation of a stellar-mass bin is
   the difference of the two threshold HODs at the bin edges.

.. rubric:: Step 8 — Standard halo-model integrals

Once :math:`(N_c(M_h),\,N_s(M_h))` are in hand, all clustering
predictions follow the standard halo-model framework — shared with the
More+2015 path in :class:`~hod_mod.galaxies.clustering.FullHaloModelPrediction`:

.. math::

   \bar{n}_g &= \int \mathrm{d}M_h\,\frac{\mathrm{d}n}{\mathrm{d}M_h}\,
               \bigl[N_c(M_h) + N_s(M_h)\bigr] \\[4pt]
   P_{gg}^\mathrm{1h}(k) &= \frac{1}{\bar{n}_g^2}
   \int \mathrm{d}M_h\,\frac{\mathrm{d}n}{\mathrm{d}M_h}
   \bigl[N_s^2\,\tilde{u}^2 + 2\,N_c\,N_s\,\tilde{u}\bigr] \\[4pt]
   P_{gg}^\mathrm{2h}(k) &= b_\mathrm{eff}^2\,P_\mathrm{lin}(k)

The difference from a pure cHOD model is that
:math:`N_c(M_h)` and :math:`N_s(M_h)` are derived from the SHMR inversion
(Steps 1–4) rather than fitted directly in halo-mass space.
For the BGS samples analysed here, the iHOD and cHOD predictions agree
to ≲1–2% (see comparison figure in the SDSS benchmark section below).

:meth:`~hod_mod.galaxies.clustering.FullHaloModelPrediction.wp`,
:meth:`~hod_mod.galaxies.clustering.FullHaloModelPrediction.delta_sigma`,
and :meth:`~hod_mod.galaxies.clustering.FullHaloModelPrediction.n_gal`
all follow this path.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq08_effective_bias.png
   :width: 90%
   :align: center

   Effective bias — the first cosmology-dependent quantity. The HOD-weighted
   bias integrand :math:`M\,\frac{dn}{dM}\,N_\mathrm{tot}\,b(M)` for the
   fiducial cosmology and three :math:`+2\%` variants, with
   :math:`b_\mathrm{eff}` quoted in the legend. Lower panel: the logarithmic
   sensitivity :math:`d\ln(\mathrm{integrand})/d\ln p` to :math:`\Omega_m`,
   :math:`\sigma_8` and :math:`h` (central finite differences).

----

Power spectra and summary statistics
--------------------------------------

The power spectra and projected statistics follow the same formalism as
:ref:`hod_more2015`.

**1-halo power spectra** (galaxy auto + galaxy–matter):

.. math::

   P_{gg}^\mathrm{1h}(k) &= \frac{1}{\bar{n}_g^2}
   \int \mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}
   \bigl[\langle N_s^2\rangle\,\tilde{u}^2 + 2\,\langle N_c\rangle\,\langle N_s\rangle\,\tilde{u}\bigr]\\
   P_{gm}^\mathrm{1h}(k) &= \frac{1}{\bar{n}_g}
   \int \mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}
   \bigl[\langle N_c\rangle + \langle N_s\rangle\,\tilde{u}\bigr]
   \frac{M}{\bar{\rho}_m}\,\tilde{u}

**2-halo power spectra**:

.. math::

   P_{gg}^\mathrm{2h}(k) = b_\mathrm{eff}^2\,P_\mathrm{lin}(k),
   \qquad
   P_{gm}^\mathrm{2h}(k) = b_\mathrm{eff}\,P_\mathrm{lin}(k)

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq09_power_spectra.png
   :width: 90%
   :align: center

   Galaxy auto / galaxy–matter power spectra. The 1-halo term dominates at
   high :math:`k`, the 2-halo term at low :math:`k`. Lower panel: the
   logarithmic sensitivity :math:`d\ln P_{gg}/d\ln p` to :math:`\Omega_m`,
   :math:`\sigma_8` and :math:`h`. The :math:`\sigma_8` curve is nearly flat at
   :math:`\simeq2` (:math:`P\propto\sigma_8^2`), while :math:`\Omega_m` and
   :math:`h` are scale-dependent (central finite differences).

**Projected correlation function** (:math:`\pi_\mathrm{max} = 60\,h^{-1}\,\mathrm{Mpc}` for SDSS):

.. math::

   w_p(r_p) = 2\int_0^{\pi_\mathrm{max}}
   \xi_{gg}\!\left(\sqrt{r_p^2 + \pi^2}\right)\mathrm{d}\pi

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq10_wp.png
   :width: 90%
   :align: center

   Projected correlation function :math:`w_p(r_p)`. Top panel overlays the
   fiducial with three :math:`+2\%` variants. Lower panel: the logarithmic
   sensitivity :math:`d\ln w_p/d\ln p` to :math:`\Omega_m`, :math:`\sigma_8`
   and :math:`h`, each with a distinct scale dependence (central finite
   differences).

**Excess surface density**:

.. math::

   \Delta\Sigma(R) =
   \frac{2}{R^2}\int_0^R R'\,\Sigma_{gm}(R')\,\mathrm{d}R' - \Sigma_{gm}(R)
   \quad [\mathrm{M}_\odot\,h\,\mathrm{pc}^{-2}]

:math:`\xi(r)` is computed via the Ogata (2005) :math:`j_0` Hankel transform
(DOI:`10.1145/1141885.1141895 <https://doi.org/10.1145/1141885.1141895>`_).

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq11_delta_sigma.png
   :width: 90%
   :align: center

   Excess surface mass density :math:`\Delta\Sigma(R)`. Because
   :math:`\Sigma_{gm}\propto\bar\rho_m\propto\Omega_m`, the signal is especially
   sensitive to :math:`\Omega_m` — visible in the lower panel, the logarithmic
   sensitivity :math:`d\ln\Delta\Sigma/d\ln p` to :math:`\Omega_m`,
   :math:`\sigma_8` and :math:`h` (central finite differences).

.. rubric:: Predicting the stellar mass function :math:`\Phi(M_*)`

Because the iHOD occupation functions are defined as a function of a
stellar-mass *threshold* :math:`M_{*,\mathrm{th}}` rather than a stellar-mass
*bin*, the model does not return :math:`\Phi(M_*)` directly. It is obtained
by first computing the **cumulative** galaxy number density above a
threshold, then differentiating numerically.

**Step 1 — cumulative number density.** For a fixed set of iHOD parameters
:math:`(\log_{10}M_1,\,\log_{10}M_{*,0},\,\beta,\,\delta,\,\gamma,\,
\sigma_{\ln M_*},\,\eta,\,f_c,\,B_\mathrm{sat},\,\beta_\mathrm{sat},\,
B_\mathrm{cut},\,\beta_\mathrm{cut},\,\alpha_\mathrm{sat})`, the central and
satellite occupation functions above threshold
(:math:`\langle N_\mathrm{cen}(M_h\,|\,M_{*,\mathrm{th}})\rangle`,
:math:`\langle N_\mathrm{sat}(M_h\,|\,M_{*,\mathrm{th}})\rangle` from
Eqs. 21–22 above) are integrated against the halo mass function:

.. math::

   \bar n_g(>M_{*,\mathrm{th}}) = \int \mathrm{d}M_h\,\frac{\mathrm{d}n}{\mathrm{d}M_h}\,
   \bigl[\langle N_\mathrm{cen}(M_h\,|\,M_{*,\mathrm{th}})\rangle +
         \langle N_\mathrm{sat}(M_h\,|\,M_{*,\mathrm{th}})\rangle\bigr]
   \quad [h^3\,\mathrm{Mpc}^{-3}]

implemented in
:meth:`~hod_mod.galaxies.clustering.FullHaloModelPrediction.n_gal`
(its ``hod_params["log10m_star_thresh"]`` entry *is*
:math:`\log_{10}M_{*,\mathrm{th}}`, in :math:`\log_{10}(M_\odot\,h^{-1})`
— see the parameter table below).

**Step 2 — finite-difference derivative.** Evaluating
:math:`\bar n_g(>M_{*,\mathrm{th}})` on a grid of thresholds and
differentiating gives the differential stellar mass function:

.. math::

   \Phi(M_*) = -\frac{\mathrm{d}\bar n_g(>M_*)}{\mathrm{d}\log_{10}M_*}
   \quad [h^3\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]

i.e. :math:`\Phi(M_*)\,\mathrm{d}\log_{10}M_*` is the number density of
galaxies with stellar mass in :math:`[M_*,\,M_* + \mathrm{d}M_*]`. This is
implemented by a centred finite difference
(:func:`numpy.gradient`) over the threshold scan — no closed-form derivative
of Eqs. 21–22 is taken, so the same code path works for any iHOD
parameter set without re-deriving the SHMR algebra.

**Units (h convention).** The sum_stat SMF data :math:`\Phi` is tabulated in
:math:`h^3\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}` (= :math:`(\mathrm{Mpc}/h)^{-3}\,\mathrm{dex}^{-1}`;
see :meth:`~hod_mod.data_io.sum_stat_reader.SumStatReader.smf`), the **same**
numeric convention as the halo model's native comoving density
:math:`\mathrm{d}n/\mathrm{d}M\;[(\mathrm{Mpc}/h)^{-3}/(M_\odot/h)]`.  The model
:math:`\bar n_g` is therefore returned directly in :math:`(\mathrm{Mpc}/h)^{-3}`
with **no** :math:`h^3` factor (the mass axis is still converted to the SHMR
convention by the :math:`+\log_{10}h` threshold shift).  An earlier version
multiplied by :math:`h^3`, under-predicting the SMF by :math:`h^3\approx0.31`
(:func:`~hod_mod.scripts.fitting.fit_comparat2025._predict_smf`).

.. rubric:: SMF entry in the likelihood — number density only

The binned :math:`\Phi(M_*)` is **not** fit shape-by-shape.  Two issues make
that pathological for the LS10-BGS samples: (i) the joint SMF+wp jackknife
covariance is numerically near-singular (condition number :math:`\sim10^{16}`,
SMF variances :math:`\sim10^{-13}` vs :math:`w_p` :math:`\sim10^{2}`), so its
dense inverse is meaningless; and (ii) the iHOD over-predicts the high-mass SMF
tail (:math:`\gtrsim 3\text{--}30\times` at :math:`\log_{10}M_*\gtrsim11.5`,
where the small :math:`z<0.18` volume is also unreliable), which — with tiny
per-bin errors — railed :math:`f_c` and the threshold against their bounds.

Instead the joint galaxy-sector likelihood uses **diagonal** uncertainties on
:math:`w_p(r_p)` plus a single **overall number-density** constraint,

.. math::

   \bar n_g = \int \mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}\,
   \bigl[\langle N_c\rangle + \langle N_s\rangle\bigr]
   \quad [(\mathrm{Mpc}/h)^{-3}],

compared to :math:`\int \Phi\,\mathrm{d}\log_{10}M_*` from the data with a
systematic floor (:math:`f_\mathrm{sys}`, default 5%).  This is robust to the
high-mass shape mismatch while still pinning the overall galaxy abundance.
:math:`\Phi(M_*)` itself (h³-correct) is retained for the diagnostic panels.

----

Parameter table
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 12 12 12 22 18

   * - Parameter
     - Symbol
     - Default
     - Fitted?
     - ZM15 Table 2 (:math:`\pm 1\sigma`)
     - Units
   * - ``log10m_star_thresh``
     - :math:`\log_{10} M_{*,\mathrm{th}}`
     - 10.2
     - Yes
     - — (sample-specific)
     - :math:`\log_{10}(M_\odot h^{-2})`
   * - ``lg_m1h``
     - :math:`\log_{10} M_1`
     - 12.10
     - Yes
     - :math:`12.10 \pm 0.17`
     - :math:`\log_{10}(M_\odot h^{-1})`
   * - ``lg_m0star``
     - :math:`\log_{10} M_{*,0}`
     - 10.31
     - Yes
     - :math:`10.31 \pm 0.10`
     - :math:`\log_{10}(M_\odot h^{-2})`
   * - ``beta``
     - :math:`\beta`
     - 0.33
     - Yes
     - :math:`0.33 \pm 0.21`
     - —
   * - ``delta``
     - :math:`\delta`
     - 0.42
     - Yes
     - :math:`0.42 \pm 0.04`
     - —
   * - ``gamma``
     - :math:`\gamma`
     - 1.21
     - Yes
     - :math:`1.21 \pm 0.20`
     - —
   * - ``sigma_lnmstar``
     - :math:`\sigma_{\ln M_*}`
     - 0.50
     - Yes
     - :math:`0.50 \pm 0.04`
     - —
   * - ``eta``
     - :math:`\eta`
     - −0.04
     - Yes
     - :math:`-0.04 \pm 0.02`
     - —
   * - ``fc``
     - :math:`f_c`
     - 0.86
     - Yes
     - :math:`0.86 \pm 0.14`
     - —
   * - ``bsat``
     - :math:`B_\mathrm{sat}`
     - 8.98
     - Yes
     - :math:`8.98 \pm 1.18`
     - —
   * - ``beta_sat``
     - :math:`\beta_\mathrm{sat}`
     - 0.90
     - **Fixed**
     - —
     - —
   * - ``bcut``
     - :math:`B_\mathrm{cut}`
     - 0.86
     - **Fixed**
     - —
     - —
   * - ``beta_cut``
     - :math:`\beta_\mathrm{cut}`
     - 0.41
     - **Fixed**
     - —
     - —
   * - ``alpha_sat``
     - :math:`\alpha_\mathrm{sat}`
     - 1.00
     - Fixed (SDSS); Yes (BGS)
     - —
     - —

----

SDSS DR7 benchmark
-------------------

Source: ``results/benchmarks/zumandelbaum2015_sdss/benchmark_result.json``.
Data digitised from ZM15 Figure 6 (WebPlotDigitizer);
reference data also from Mandelbaum et al. 2006
(`arXiv:astro-ph/0509702 <https://arxiv.org/abs/astro-ph/0509702>`_).

:math:`\chi^2/\mathrm{dof} \approx 1.75 \times 10^{-6}` — effectively zero.
This near-perfect agreement is by construction: the data are extracted from the
published figure using the published model parameters, so the MAP recovers
those parameters to within numerical precision.

.. list-table::
   :header-rows: 1
   :widths: 24 18 24 14

   * - Parameter
     - MAP
     - Published (:math:`\pm 1\sigma`)
     - Deviation
   * - ``lg_m1h``
     - 12.1000
     - :math:`12.10 \pm 0.17`
     - :math:`0.00\sigma`
   * - ``lg_m0star``
     - 10.3100
     - :math:`10.31 \pm 0.10`
     - :math:`0.00\sigma`
   * - ``beta``
     - 0.3300
     - :math:`0.33 \pm 0.21`
     - :math:`0.00\sigma`
   * - ``delta``
     - 0.4177
     - :math:`0.42 \pm 0.04`
     - :math:`0.06\sigma`
   * - ``gamma``
     - 1.2106
     - :math:`1.21 \pm 0.20`
     - :math:`0.00\sigma`
   * - ``sigma_lnmstar``
     - 0.5001
     - :math:`0.50 \pm 0.04`
     - :math:`0.00\sigma`
   * - ``eta``
     - −0.0400
     - :math:`-0.04 \pm 0.02`
     - :math:`0.00\sigma`
   * - ``fc``
     - 0.9066
     - :math:`0.86 \pm 0.14`
     - :math:`0.33\sigma`
   * - ``bsat``
     - 8.9801
     - :math:`8.98 \pm 1.18`
     - :math:`0.00\sigma`

.. figure:: ../results/benchmarks/zumandelbaum2015_sdss/benchmark_zumandelbaum2015_combined.png
   :width: 95%
   :alt: ZM15 combined wp and delta sigma

   MAP fit to SDSS DR7 :math:`\log_{10}(M_*/h^{-2}M_\odot) > 10.2`.
   *Top*: :math:`w_p(r_p)`.  *Bottom*: :math:`\Delta\Sigma(R)`.

.. figure:: ../results/benchmarks/zumandelbaum2015_sdss/benchmark_zumandelbaum2015_hod.png
   :width: 70%
   :alt: ZM15 HOD occupation functions

   HOD occupation functions vs halo mass at the MAP parameters.

.. figure:: ../results/benchmarks/zumandelbaum2015_sdss/comparison_ihod_chod_wp_ds.png
   :width: 80%
   :alt: iHOD vs cHOD comparison

   Comparison of iHOD (ZuMandelbaum15) and cHOD (conventional) predictions
   for :math:`w_p` and :math:`\Delta\Sigma`.
   Differences are at the 1–2% level — both approaches produce statistically
   equivalent predictions for these observables.

For the 7-bin multi-sample iHOD fit (all stellar-mass bins simultaneously,
joint :math:`\chi^2/\mathrm{dof} = 2.34`), see
:ref:`benchmark_zumandelbaum2015_multisample`.

----

X-ray cross-correlation extension
-----------------------------------

For the BGS × eROSITA analysis
(Comparat et al. 2025, `arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_),
the ZM15 iHOD is extended with a gas density component and an AGN component.

.. rubric:: Gas density profile (GasDensityDPM)

Reference: Oppenheimer et al. 2025
(`arXiv:2505.14782 <https://arxiv.org/abs/2505.14782>`_),
implemented in :class:`~hod_mod.cosmology.gas_profiles.GasDensityDPM`.

The electron density profile uses a generalised NFW shape
(arXiv:2505.14782 Eq. 1):

.. math::

   f(x \,|\, \boldsymbol{\alpha}) =
   x^{-\alpha_\mathrm{in}}\,
   \bigl(1 + x^{\alpha_\mathrm{tr}}\bigr)^{(\alpha_\mathrm{in}-\alpha_\mathrm{out})/\alpha_\mathrm{tr}}

where :math:`x = r/R_s` and :math:`R_s = R_{200}/c(M,z)`.
The concentration :math:`c(M,z)` follows Diemer & Joyce 2019
(`arXiv:1809.07326 <https://arxiv.org/abs/1809.07326>`_),
via :func:`~hod_mod.cosmology.concentration.c_diemer15`,
the same relation used for the NFW galaxy profile.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq12_gnfw_shape.png
   :width: 80%
   :align: center

   gNFW shape function :math:`f(x|\boldsymbol{\alpha})` shared by all three DPM
   profiles, varying the inner and outer slopes. Cosmology-independent.

The electron density (arXiv:2505.14782 Eq. 3):

.. math::

   n_e(r, M, z) = n_{e,0.3}\,
   \frac{f(r/R_s)}{f(0.3\,R_{200}/R_s)}\,
   E(z)^\gamma\,
   \left(\frac{M_{200}}{10^{12}\,h^{-1}M_\odot}\right)^{\!\beta_\mathrm{gas}}

Default model 2 parameters (Table 1 of arXiv:2505.14782):
:math:`n_{e,0.3} = 4.87 \times 10^{-5}` cm\ :sup:`-3`,
:math:`\alpha_\mathrm{in} = 1.0`,
:math:`\alpha_\mathrm{tr} = 1.9`,
:math:`\alpha_\mathrm{out} = 2.7`,
:math:`\beta_\mathrm{gas} = 0.36` (mass slope; **free in fitting**),
:math:`\gamma = 2.0`.

.. rubric:: Gas pressure profile (PressureProfileDPM)

The electron pressure uses the same gNFW shape with mass-dependent outer slope
(arXiv:2505.14782 Eq. 5) and mass/redshift scaling (Eq. 2):

.. math::

   P(r, M, z) = P_{0.3}\,
   \frac{f(r/R_s \,|\, \alpha_\mathrm{out}(M))}{f(0.3\,R_{200}/R_s)}\,
   E(z)^{\gamma^P}\,
   \left(\frac{M_{200}}{10^{12}\,h^{-1}M_\odot}\right)^{\!\beta^P}

Model 2 parameters: :math:`P_{0.3} = 115` meV cm\ :sup:`-3`,
:math:`\beta^P = 0.85`, :math:`\gamma^P = 8/3`.

.. rubric:: Gas metallicity profile (MetallicityProfileDPM)

A gNFW metallicity profile (no mass or redshift dependence):

.. math::

   Z(r) = Z_0\,f(r/R_s \,|\, \boldsymbol{\alpha}^Z),
   \qquad Z(0.3\,R_{200}) = 0.3\,Z_\odot

with :math:`\alpha^Z_\mathrm{in}=0`, :math:`\alpha^Z_\mathrm{tr}=0.5`,
:math:`\alpha^Z_\mathrm{out}=0.7` (Table 1 of arXiv:2505.14782).

.. rubric:: X-ray emissivity — APEC cooling function

All three DPM profiles are evaluated at every Gauss-Legendre quadrature node.
The temperature at radius :math:`r` is derived from the ideal gas law:

.. math::

   T(r, M, z) = \frac{P(r, M, z)}{n_e(r, M, z)} \quad [\mathrm{keV}]

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq13_gas_profiles.png
   :width: 75%
   :align: center

   DPM electron density :math:`n_e(r)` (top) and temperature
   :math:`T(r)=P/n_e` (bottom) for three halo masses, showing the
   :math:`M_{12}^{\beta}` mass scaling. Cosmology-independent (fixed cosmology).

The 0.5–2 keV soft X-ray emissivity per unit volume is:

.. math::

   \varepsilon(r, M, z) =
   n_e^2(r, M, z)\;\Lambda_{n_e^2}\!\bigl(T(r),\,Z(r)\bigr)

where :math:`\Lambda_{n_e^2}(T, Z) = 0.83\,\Lambda_{\rm APEC}(T, Z)` is
the band-integrated APEC cooling function (AtomDB, via `soxs
<https://hea-www.cfa.harvard.edu/soxs/>`_ + pyXSIM), precomputed over a
log-spaced :math:`(T, Z)` grid at initialisation and evaluated by 2D
log-log interpolation at runtime
(:class:`~hod_mod.cosmology.gas_profiles.ApecCoolingTable`).
The factor 0.83 converts from the :math:`n_e n_H` APEC convention to
:math:`n_e^2` (:math:`n_H \approx 0.83\,n_e` for solar-abundance plasma).

Reference values (0.5–2 keV, AtomDB 3.1.3, ``abund_table="angr"``):

.. list-table::
   :header-rows: 1
   :widths: 12 12 25

   * - :math:`T` [keV]
     - :math:`Z` [:math:`Z_\odot`]
     - :math:`\Lambda_{n_e^2}` [erg cm\ :sup:`3` s\ :sup:`-1`]
   * - 0.5
     - 0.3
     - :math:`\approx 8.0\times10^{-24}`
   * - 1.0
     - 0.3
     - :math:`\approx 7.6\times10^{-24}`
   * - 2.0
     - 0.3
     - :math:`\approx 5.0\times10^{-24}`
   * - 1.0
     - 1.0
     - :math:`\approx 1.9\times10^{-23}`

An overall amplitude :math:`A_\mathrm{gas}` is a free parameter fitted
jointly with the HOD parameters.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq14_cooling.png
   :width: 95%
   :align: center

   *Left*: the gNFW metallicity profile :math:`Z(r)`. *Right*: the
   band-integrated APEC cooling function :math:`\Lambda_{n_e^2}(T,Z)` (0.5–2 keV,
   AtomDB) vs temperature for three metallicities. Cosmology-independent.

.. rubric:: HAM AGN model (HamAGNModel)

References: Aird et al. 2015, ApJ 815, 66
(`arXiv:1503.01120 <https://arxiv.org/abs/1503.01120>`_) — LADE hard XLF;
Comparat et al. 2019, A&A 622, A12
(`arXiv:1901.10866 <https://arxiv.org/abs/1901.10866>`_) — obscuration model;
implemented in :class:`~hod_mod.galaxies.agn_ham.HamAGNModel`.

The HAM pipeline assigns a hard X-ray luminosity to each halo by
abundance-matching the cumulative halo number density (from the iHOD SHMR)
to the cumulative AGN number density from the Aird+2015 LADE hard XLF.

**LADE hard XLF** (Aird+2015, 2–10 keV):

.. math::

   \Phi(L_X, z) = \frac{k(z)}{(L_X/L_s(z))^{\gamma_1}
                   + (L_X/L_s(z))^{\gamma_2}}
   \quad [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]

with luminosity-dependent density evolution (LADE):

.. math::

   k(z) = 10^{-4.03 - 0.19(1+z)}, \qquad
   L_s(z) = 10^{44.84} \left[
     \left(\frac{1+2}{1+z}\right)^{3.87}
     + \left(\frac{1+2}{1+z}\right)^{-2.12}
   \right]^{-1}

and slopes :math:`\gamma_1 = 0.48`, :math:`\gamma_2 = 2.27`
(Comparat+2019 eqs. 2–3 fit to Aird+2015).

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq16_xlf.png
   :width: 70%
   :align: center

   LADE hard X-ray luminosity function :math:`\Phi(L_X,z)` (Aird+2015) at four
   redshifts, showing the luminosity-dependent density evolution.

**Abundance matching**: at each :math:`(M_h, z)`, the iHOD cumulative number
density :math:`n_\mathrm{gal}(>M_*(M_h), z)` multiplied by the AGN duty
cycle :math:`f_\mathrm{DC}(z)` is matched to :math:`n_\mathrm{AGN}(>L_X, z)`
from the XLF, yielding :math:`\langle L_X^\mathrm{hard}(M_h, z)\rangle`.
This table is precomputed at initialisation on a 2D :math:`(z, M_h)` grid.

**Duty cycle** :math:`f_\mathrm{DC}(z)`:

.. list-table::
   :header-rows: 1
   :widths: 10 12

   * - :math:`z`
     - :math:`f_\mathrm{DC}`
   * - 0.00
     - 0.038
   * - 0.25
     - 0.097
   * - 0.75
     - 0.40
   * - :math:`\geq 1.75`
     - 0.50

**Obscuration model** (Comparat+2019 eqs. 4–11):

*Compton-thick luminosity threshold* — the CT boundary shifts with redshift:

.. math::

   L_{ll}(z) = 41.5 + 1.5\,\arctan(5z)

*Compton-thick fraction* (log :math:`N_H \geq 24`, Comparat+2019 eq. 4):

.. math::

   f_{CT}(\log L_X,\,z) = 0.30\left[0.5
     + 0.5\,\mathrm{erf}\!\left(\frac{L_{ll}(z) - \log L_X}{0.25}\right)\right]

*Bright-end obscured fraction* (Comparat+2019 eq. 5) — floor from CT plus a
redshift-dependent boost:

.. math::

   f_1(\log L_X,\,z) = f_{CT}(\log L_X,\,z) + 0.01
     + 0.3\,\mathrm{erf}\!\left(\frac{z}{4}\right)

*Faint-end obscured fraction* (Comparat+2019 eq. 6) — rising toward low
luminosities:

.. math::

   f_2(\log L_X) = 0.9\,\sqrt{\frac{41}{\log_{10}(L_X\,/\,\mathrm{erg\,s}^{-1})}}

*Transition luminosity* (crossover between the two regimes):

.. math::

   L_t(z) = 43.2 + 1.2\,\mathrm{erf}(z)

*Blending weight* (smooth interpolation between :math:`f_1` and :math:`f_2`):

.. math::

   w(\log L_X,\,z) = 0.5
     + 0.5\,\mathrm{erf}\!\left(\frac{L_t(z) - \log L_X}{0.6}\right)

*Total obscured fraction* (log :math:`N_H > 22`, type-2 + CT,
Comparat+2019 eq. 11):

.. math::

   f_\mathrm{obsc}(\log L_X,\,z)
     = \mathrm{clip}\!\left[f_1 + (f_2 - f_1)\,w,\;0,\;1\right]

*Type fractions* derived from :math:`f_\mathrm{obsc}` and :math:`f_{CT}`:

.. math::

   f_{\mathrm{type\text{-}1}} = 1 - f_\mathrm{obsc}, \qquad
   f_{\mathrm{type\text{-}2}} = f_\mathrm{obsc} - f_{CT}, \qquad
   f_{\mathrm{CT}} \text{ as above}

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq17_obscuration.png
   :width: 95%
   :align: center

   *Left*: AGN obscured (:math:`\log N_H>22`) and Compton-thick fractions vs
   :math:`\log L_X` at two redshifts. *Right*: the AGN duty cycle
   :math:`f_\mathrm{DC}(z)`. Cosmology-independent.

**Hard-to-soft conversion**: the effective K-correction is averaged over the
three type classes using the precomputed absorption table
(``hod_mod/data/agn/v3_fraction_observed_A15_RF_hard_Obs_soft_fscat_002.txt``,
generated by integrating an absorbed power-law with :math:`\Gamma=1.9` and
:math:`f_\mathrm{scat}=0.02` over a :math:`35\times16` grid of
:math:`(z,\log N_H)` values):

.. math::

   k_\mathrm{eff}(\log L_X,\,z)
     = f_{\mathrm{type\text{-}1}}\,k(z,\,N_H=10^{20})
     + f_{\mathrm{type\text{-}2}}\,k(z,\,N_H=10^{23})
     + f_{CT}\,k(z,\,N_H=10^{25})

The hard-to-soft ratio for unobscured AGN at :math:`z=0` is 0.607.

**Mean soft X-ray luminosity**: combining HAM hard luminosity, K-correction,
duty cycle, and log-normal scatter boost:

.. math::

   \langle L_X^\mathrm{soft}(M_h,z)\rangle
     = \langle L_X^\mathrm{hard}\rangle_\mathrm{HAM}
       \times\,k_\mathrm{eff}\,
       \times\,f_\mathrm{DC}(z)\,
       \times\,\exp\!\left(\frac{\sigma_\mathrm{dex}^2\,\ln^2\!10}{2}\right)

with log-normal scatter :math:`\sigma_\mathrm{dex} = 0.8` dex in
:math:`L_X` at fixed :math:`M_h`.

AGN are unresolved point sources, so their angular template is the PSF
(King profile, :math:`\theta_c = 8.64` arcsec on-axis eROSITA):

.. math::

   \mathrm{PSF}_\mathrm{King}(\theta) =
   \left[1 + \left(\frac{\theta}{\theta_c}\right)^2\right]^{-\alpha_\mathrm{King}},
   \quad \alpha_\mathrm{King} = 1.5

.. rubric:: HOD AGN model (HODAgnModel)

References: More et al. 2015, ApJ 806, 2
(`arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_) — HOD form;
Aird et al. 2015 — XLF; Lau et al. 2024
(`arXiv:2410.22397 <https://arxiv.org/abs/2410.22397>`_) — X-ray power-spectrum
formalism; implemented in :class:`~hod_mod.galaxies.agn_hod.HODAgnModel`.

This is an alternative, conceptually distinct AGN component.  Rather than
abundance-matching halo mass directly to :math:`L_X` (as
:class:`~hod_mod.galaxies.agn_ham.HamAGNModel` does), it places AGN with an
explicit HOD and then assigns luminosities by a flux/optically-selected
abundance match.  Unlike the HAM model it provides its **own** AGN occupation
:math:`N_\mathrm{cen}^\mathrm{AGN}(M)`, :math:`N_\mathrm{sat}^\mathrm{AGN}(M)`,
which drives an occupation-weighted X-ray auto/cross power spectrum.

**1. AGN HOD with a constant duty cycle.** A 5-parameter More+2015 occupation
with a mass-independent duty cycle :math:`f_\mathrm{inc}` multiplying the whole
occupation uniformly (:class:`~hod_mod.galaxies.hod.MoreConstFincHODModel`):

.. math::

   N_\mathrm{cen}^\mathrm{AGN}(M) &= f_\mathrm{inc}\,\tfrac{1}{2}\,
   \mathrm{erfc}\!\left[\frac{\log_{10}M_\mathrm{min} - \log_{10}M}{\sigma_{\log M}}\right]\\
   N_\mathrm{sat}^\mathrm{AGN}(M) &= N_\mathrm{cen}^\mathrm{AGN}(M)
   \left(\frac{M - \kappa M_\mathrm{min}}{M_1}\right)^{\alpha}
   \quad (M > \kappa M_\mathrm{min})

with defaults :math:`\log_{10}M_\mathrm{min}=12.5`, :math:`\sigma_{\log M}=0.8`,
:math:`\alpha=0.8`, :math:`\log_{10}M_1=14.0\,(=\log_{10}M_\mathrm{min}+1.5)`,
:math:`\kappa=0.3`, :math:`f_\mathrm{inc}=0.1`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq18_agn_hod.png
   :width: 70%
   :align: center

   AGN HOD occupation :math:`N_c^\mathrm{AGN}`, :math:`N_s^\mathrm{AGN}` and the
   total, with the mass-independent duty cycle :math:`f_\mathrm{inc}` scaling the
   whole occupation. Cosmology-independent.

**2. Host stellar masses.** The Zu & Mandelbaum 2015 SHMR (above) maps each
AGN-host halo mass to a stellar mass for centrals and satellites.

**3. Modified abundance matching** (at the sample mean redshift):

- From the Aird+2015 XLF, build the luminosity distribution down to
  :math:`\log_{10}L_X = 39`.
- Convert hard-band :math:`L_X` to observed soft (0.5–2 keV) luminosity via the
  obscuration-weighted K-correction (shared with HamAGNModel), then to flux
  :math:`F_X = L_X^\mathrm{soft}/(4\pi d_L^2)`.
- Predict the r-band magnitude :math:`r = -7 - 2\log_{10}F_X` and keep
  :math:`16 \le r \le 19.5`.
- Rank-order match the selected :math:`L_X` distribution onto the
  (:math:`f_\mathrm{inc}`-suppressed) AGN-host stellar-mass distribution.  The
  matching is done on cumulative number **densities** (the volume cancels), so
  it is deterministic — the noise-free limit of drawing a finite array.

This yields a monotonic :math:`\log_{10}M_* \to \log_{10}L_X^{0.5-2,\mathrm{obs}}`
mapping (``mean_agn_log10lx``) and the sample-averaged observed luminosity/flux
(``mean_observed_lx`` / ``mean_observed_fx``).

**4. Occupation-weighted X-ray power spectra** (Lau+2024 App. A).  In
:meth:`~hod_mod.galaxies.cross_spectra.HaloModelCrossSpectra._pk_tables_gX` and
``_pk_tables_XX`` the AGN luminosity is placed by the AGN occupation —
central at the halo centre, satellites on the NFW profile — instead of the
galaxy HOD point-source weighting used for HamAGNModel/XrayAGNModel.  The
auto-power AGN term then carries the central–satellite and satellite–satellite
pair structure, luminosity-weighted, plus the gas×AGN cross term.  This branch
activates automatically when the supplied ``agn_model`` exposes ``nc_ns_agn``.

.. admonition:: Duty cycle and the abundance match

   The AGN-host number density scales with :math:`f_\mathrm{inc}`.  At
   :math:`f_\mathrm{inc}=0.1` it (:math:`\sim 3\times10^{-4}\,(h/\mathrm{Mpc})^3`
   for S1) is :math:`\sim 30\times` larger than the optically/flux-selected AGN
   density (:math:`\sim 10^{-5}`), so most low-mass hosts match to the faint
   selection edge (``_frac_clamped`` :math:`\approx 0.97`).  Lowering
   :math:`f_\mathrm{inc}` brings the two abundances into agreement — for S1 the
   faint-edge clamped fraction falls to :math:`\approx 0.66` at
   :math:`f_\mathrm{inc}=0.01` and to :math:`0` near :math:`f_\mathrm{inc}\approx
   0.003`, while the predicted mean observed :math:`\log_{10}L_X` rises from
   :math:`42.5` to :math:`43.4`.  The default for the fits is therefore
   :math:`f_\mathrm{inc}=0.01` (configurable); the clamped fraction is reported
   as a diagnostic.

.. rubric:: Use in the BGS×eROSITA fit (fit_comparat2025)

When run with ``--agn-model hod`` (the default), the AGN term of the
:math:`w_\theta` fit in :mod:`hod_mod.scripts.fitting.fit_comparat2025` is the
Hankel transform of the **predicted** HODAgnModel cross-power
:math:`C_\ell^{gX,\mathrm{agn}}` (PSF-convolved), not a free-shape King PSF.
A per-sample ``HODAgnModel`` is built from the sample's :math:`z_\mathrm{mean}`,
:math:`z_\mathrm{max}` and ``--agn-finc`` (default 0.01).  The free parameter
:math:`\log_{10}A_\mathrm{AGN}` then acts as a **fudge factor** on this
physically-predicted amplitude (the PSF normalisation in the cross-correlation),
rather than absorbing the entire AGN flux scale.  The legacy free-amplitude King
PSF behaviour remains available via ``--agn-model ham``.

A demo driver is provided in
:mod:`hod_mod.scripts.galaxies.run_agn_hod_cl`
(``--sample S1`` / ``--sample S5``).

.. rubric:: Prediction pipeline — from profiles to :math:`w_\theta(\theta)`

The six steps below go from the DPM profile parameters to the observable
angular cross-correlation.  All steps are implemented in
:class:`~hod_mod.galaxies.cross_spectra.HaloModelCrossSpectra`.

**Step 1 — Emissivity profile Fourier transform** :math:`\tilde{X}(k|M,z)`

The key quantity linking the 3D profile to the halo model is the spherical
Fourier transform of the per-halo emissivity (implemented in
:meth:`~hod_mod.cosmology.gas_profiles.GasDensityDPM.emissivity_full_uk`):

.. math::

   \tilde{X}(k|M,z) = 4\pi \int_0^{r_\mathrm{max}}
   n_e^2(r|M,z)\;\Lambda_{\mathrm{APEC}}\!\bigl(T(r|M,z),\,Z(r)\bigr)\;
   j_0(kr)\;r^2\,\mathrm{d}r

where :math:`j_0(x) = \sin(x)/x`.
:math:`T(r) = P(r)/n_e(r)` [keV] from the ideal gas law (Step 1 of
:meth:`~hod_mod.cosmology.gas_profiles.temperature_from_profiles`).
The radial integral uses 200-point Gauss-Legendre quadrature up to
:math:`r_\mathrm{max} = 3\,R_{200}`.

At :math:`k \to 0`, :math:`\tilde{X}(0|M,z)` equals the total
halo emissivity :math:`L_X^\mathrm{gas}(M,z)/\Lambda_\mathrm{ref}` in
volume units :math:`[\mathrm{Mpc}/h]^3\,\mathrm{cm}^{-6}`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq15_emissivity.png
   :width: 95%
   :align: center

   *Left*: X-ray emissivity profile :math:`\varepsilon(r)=n_e^2\,\Lambda(T,Z)`
   for three halo masses. *Right*: the spherical Fourier transform
   :math:`\tilde X(k|M,z)` that links the 3D emissivity to the halo model.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq20_emissivity_sensitivity.png
   :width: 75%
   :align: center

   Cosmology sensitivity of the emissivity Fourier transform
   :math:`\tilde X(k|M)` at :math:`M_{200}=10^{14}\,M_\odot/h`. Lower panel: the
   logarithmic sensitivity :math:`d\ln\tilde X/d\ln p` to :math:`\Omega_m`,
   :math:`\sigma_8` and :math:`h` (through the concentration and :math:`E(z)`
   scaling; central finite differences) — the cosmology dependence that
   propagates into :math:`P_{gX}`, :math:`C_\ell^{gX}` and :math:`w_\theta`.

**Step 2 — 3D galaxy × X-ray power spectrum** :math:`P_{gX}(k,z)`

The halo model splits :math:`P_{gX}` into 1-halo and 2-halo contributions
(:meth:`~hod_mod.galaxies.cross_spectra.HaloModelCrossSpectra._pk_tables_gX`):

*1-halo term* (galaxies and X-ray emission in the same halo):

.. math::

   P_{gX}^{1h}(k,z) = \frac{1}{\bar{n}_g}
   \int \frac{\mathrm{d}n}{\mathrm{d}M}
   \Bigl[N_c(M)\,+\,N_s(M)\,\tilde{u}(k|M)\Bigr]\,
   \tilde{X}(k|M,z)\;\mathrm{d}M

where :math:`\tilde{u}(k|M)` is the NFW dark-matter profile Fourier transform,
and :math:`N_c,\,N_s` are the central/satellite occupation from
:class:`~hod_mod.galaxies.hod.ZuMandelbaum15HODModel`.

*2-halo term* (galaxies in one halo, X-ray emission from a different halo,
correlated by large-scale structure):

.. math::

   P_{gX}^{2h}(k,z) = b_\mathrm{eff}\,P_\mathrm{lin}(k,z)
   \int \frac{\mathrm{d}n}{\mathrm{d}M}\,b(M)\,\tilde{X}(k|M,z)\;\mathrm{d}M

The total is :math:`P_{gX} = P_{gX}^{1h} + P_{gX}^{2h}`.

.. admonition:: Note on 1-halo vs 2-halo amplitude

   At angular scales :math:`\theta \in [8'',\,300'']` (i.e.
   :math:`k \approx 1.3`–:math:`49\,h\,\mathrm{Mpc}^{-1}` at
   :math:`z=0.135`), the **1-halo term dominates** the prediction by
   factors of 9–12 000 over the 2-halo term.  Both terms are computed
   in full; the 2-halo only overtakes the 1-halo at
   :math:`\theta \gtrsim 1^\circ` (outside the data range).

   The predicted :math:`w_\theta(\theta)` can appear "2-halo-like"
   (a smooth power-law slope) because the dominant signal traces
   the *outer gNFW profile* at :math:`r > R_{200}` (where :math:`\alpha_\mathrm{out}=2.7`
   gives :math:`n_e^2 \propto r^{-5.4}`, a steep power law), plus the
   contribution of satellite galaxies sitting in X-ray bright clusters.

**Step 3 — Mass-slope tilt** (free parameters :math:`\beta_\mathrm{gas}`,
:math:`\beta_P`)

After computing :math:`\tilde{X}` at the DPM reference slopes,
two multiplicative tilts are applied in mass space (no re-integration needed):

.. math::

   \tilde{X}(k|M) \;\longrightarrow\;
   \tilde{X}(k|M)
   \times \left(\frac{M}{10^{12}\,h^{-1}M_\odot}\right)^{\!2(\beta_\mathrm{gas}-\beta_\mathrm{gas}^0)}
   \times \left(\frac{M}{10^{12}\,h^{-1}M_\odot}\right)^{\!0.5(\beta_P - \beta_P^0)}

with :math:`\beta_\mathrm{gas}^0 = 0.36` and :math:`\beta_P^0 = 0.85` (DPM model 2).
These replace full re-integration and make the fitting fast (cached :math:`\tilde{X}`
reused across MCMC steps).

**Step 4 — X-ray window function** :math:`W_X(\chi)` and Limber integral

The galaxy × X-ray cross-spectrum via the Limber approximation
(Loverde & Afshordi 2008, `arXiv:0809.5112 <https://arxiv.org/abs/0809.5112>`_):

.. math::

   C_\ell^{gX} = \int \frac{\mathrm{d}\chi}{\chi^2}\;
   \underbrace{n_g(z)\,\frac{\mathrm{d}z}{\mathrm{d}\chi}}_{W_g(\chi)}\;
   P_{gX}\!\left(\frac{\ell+\tfrac{1}{2}}{\chi},\,z(\chi)\right)

There is **no separate** :math:`W_X(\chi)` window function.
The X-ray emissivity response is entirely encoded in :math:`P_{gX}(k,z)` via
:math:`\tilde{X}(k|M,z)`: the halo model automatically integrates the
emissivity profile over all halos at each redshift.
Symbolically, writing :math:`W_X(\chi) = \langle\varepsilon(z)\rangle/\langle S_X\rangle`
would require knowing the mean background emissivity; instead the halo model
computes :math:`P_{gX}` directly, bypassing that normalisation.

The Limber integrand is evaluated on a grid of :math:`N_z` redshift slices
spanning the galaxy :math:`n_g(z)`, with :math:`k_\mathrm{Limber} = (\ell + \tfrac{1}{2})/\chi(z)`.
The :math:`\ell` grid spans :math:`\ell \in [10,\,10^5]` on 160 log-spaced points;
:math:`\ell_\mathrm{max} = 10^5` resolves :math:`\theta = 8''`
(:math:`\ell \approx 25\,800` at :math:`z_\mathrm{mean} = 0.135`).

**Step 5 — PSF convolution**

The gas :math:`C_\ell^{gX}` is multiplied by the eROSITA PSF window
in :math:`\ell`-space (King profile, :math:`\theta_c = 8.64''`, :math:`\alpha = 1.5`):

.. math::

   C_\ell^{gX} \;\longrightarrow\; C_\ell^{gX}\,B_\ell,
   \qquad
   B_\ell = C \bigl(\ell\,\theta_c\bigr)^{\alpha - 1/2}
            K_{\alpha-1/2}(\ell\,\theta_c)

where :math:`K_\nu` is the modified Bessel function and :math:`C` normalises
:math:`B_0 = 1`.  At :math:`\ell = 25\,800` (:math:`\theta_c = 8.64'' = 4.19\times10^{-5}\,\mathrm{rad}`),
:math:`\ell\,\theta_c \approx 1.08` so :math:`B_\ell \approx 0.5`: the PSF
suppresses but does not eliminate signal at :math:`\theta = 8''`.

.. figure:: ../results/benchmarks/zumandelbaum2015_equations/eq19_psf.png
   :width: 95%
   :align: center

   *Left*: the eROSITA King PSF :math:`\mathrm{PSF}_\mathrm{King}(\theta)`.
   *Right*: the corresponding beam window :math:`B_\ell` that multiplies the gas
   :math:`C_\ell^{gX}`. Instrument response — cosmology-independent.

**Step 6 — Angular correlation function and full model**

The PSF-convolved :math:`C_\ell` is Hankel-transformed to the angular
correlation function:

.. math::

   w_\theta(\theta) = \frac{1}{2\pi}
   \int_0^\infty \ell\,C_\ell^{gX}\,B_\ell\,J_0(\ell\theta)\;\mathrm{d}\ell

The full model at the 31 data bins :math:`\theta \in [8'',\,300'']` is:

.. math::

   w_\theta^\mathrm{model}(\theta) =
   A_\mathrm{gas}\,s_\mathrm{gas}(\theta)
   + A_\mathrm{AGN}\,\mathrm{PSF}_\mathrm{King}(\theta)

where :math:`s_\mathrm{gas}(\theta)` is the Hankel transform of
:math:`C_\ell^\mathrm{gas}` (both 1h and 2h included), and
:math:`A_\mathrm{gas}`,\,:math:`A_\mathrm{AGN}` are free dimensionless
amplitudes.  AGN are treated as unresolved point sources so their template is
:math:`\mathrm{PSF}_\mathrm{King}(\theta)` directly (no halo-model
:math:`C_\ell^{gX,\mathrm{AGN}}`), avoiding a spurious 2-halo hump at large
:math:`\theta` in the AGN component.

.. figure:: ../_images/direct_prediction_S1_fig7_wtheta-1.png
   :width: 90%
   :alt: BGS S1 w_theta decomposition

   Angular cross-correlation :math:`w_\theta(\theta)` for BGS S1
   (:math:`\log_{10}M_* > 10`, :math:`z_\mathrm{mean} = 0.135`).
   Gas (blue), AGN (orange), and total (black) model components vs eROSITA data (grey).

----

BGS X-ray fit parameters (8 free parameters)
----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 14 20 20 20

   * - Parameter
     - Symbol
     - Prior range
     - Units
     - Description
   * - ``log10_A_gas``
     - :math:`\log_{10} A_\mathrm{gas}`
     - :math:`[-2,\,12]`
     - —
     - gas amplitude (absorbs :math:`\Lambda_\mathrm{eff}` and unit conversion)
   * - ``beta_gas``
     - :math:`\beta_\mathrm{gas}`
     - :math:`[0,\,0.8]`
     - —
     - gas density mass slope (overrides DPM calibrated value)
   * - ``beta_pressure``
     - :math:`\beta_P`
     - :math:`[0,\,2]`
     - —
     - pressure profile mass slope (for future tSZ extension)
   * - ``log10_A_AGN``
     - :math:`\log_{10} A_\mathrm{AGN}`
     - :math:`[-5,\,15]`
     - —
     - AGN amplitude (absorbs duty cycle and unit conversion)
   * - ``log10m_star_thresh``
     - :math:`\log_{10} M_{*,\mathrm{th}}`
     - :math:`[9,\,12]`
     - :math:`\log_{10}(M_\odot)`
     - stellar-mass threshold of the BGS sample
   * - ``sigma_lnmstar``
     - :math:`\sigma_{\ln M_*}`
     - :math:`[0.01,\,1.5]`
     - —
     - scatter in :math:`\ln M_*` at fixed :math:`M_h`
   * - ``lg_m1h``
     - :math:`\log_{10} M_1`
     - :math:`[9.5,\,14]`
     - :math:`\log_{10}(M_\odot h^{-1})`
     - SHMR pivot halo mass
   * - ``alpha_sat``
     - :math:`\alpha_\mathrm{sat}`
     - :math:`[0.5,\,2.5]`
     - —
     - satellite occupation power-law slope

All SHMR shape parameters (:math:`\beta,\,\delta,\,\gamma,\,\eta,\,f_c,\,B_\mathrm{sat}`)
are held at their ZM15 published best-fit values during BGS fitting.

Complete model parameter inventory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below covers every named parameter of the three model components.
Column **Status** uses: **Free** = fitted in the BGS 8-parameter run;
*Fixable* = currently fixed but physically meaningful to free;
Fixed = hard-coded or calibrated from simulation / external data.

.. list-table::
   :header-rows: 1
   :widths: 30 10 14 12 34

   * - Parameter / symbol
     - Value
     - Status
     - Component
     - Notes

   * - :math:`\log_{10} A_\mathrm{gas}`
     - (fitted)
     - **Free**
     - Gas
     - Amplitude of the emissivity template; absorbs overall normalisation uncertainty

   * - :math:`\beta_\mathrm{gas}` (``beta_gas``)
     - (fitted)
     - **Free**
     - Gas
     - Mass slope of :math:`n_e`; overrides DPM model-2 calibrated value 0.36

   * - :math:`\beta_P` (``beta_pressure``)
     - (fitted)
     - **Free**
     - Gas
     - Mass slope of pressure profile; sets :math:`T(r) \propto M^{\beta_P - \beta_n}` at fixed shape

   * - :math:`\log_{10} A_\mathrm{AGN}`
     - (fitted)
     - **Free**
     - AGN
     - Amplitude of the AGN PSF template; absorbs duty-cycle and unit-conversion uncertainty

   * - :math:`\log_{10} M_{*,\mathrm{th}}` (``log10m_star_thresh``)
     - 10.0 (S1)
     - **Free**
     - iHOD
     - Stellar-mass threshold; determines effective :math:`M_\mathrm{min}` of the sample

   * - :math:`\sigma_{\ln M_*}` (``sigma_lnmstar``)
     - 0.50
     - **Free**
     - iHOD
     - SHMR scatter; controls satellite richness and the width of the 1-halo gas peak

   * - :math:`\log_{10} M_1` (``lg_m1h``)
     - 12.10
     - **Free**
     - iHOD
     - SHMR pivot halo mass; sets :math:`\langle M_h\rangle` of the galaxy sample

   * - :math:`\alpha_\mathrm{sat}` (``alpha_sat``)
     - 1.00
     - **Free**
     - iHOD
     - Satellite occupation slope; affects the 2-halo to 1-halo transition

   * - :math:`n_{e,0.3}` (``ne_03``)
     - :math:`4.87\times10^{-5}\ \mathrm{cm}^{-3}`
     - *Fixable*
     - Gas
     - Density amplitude at :math:`0.3\,R_{200}` (degenerate with :math:`A_\mathrm{gas}`; freeing it would allow a physical normalisation)

   * - :math:`P_{0.3}`
     - :math:`115\ \mathrm{meV\,cm}^{-3}`
     - *Fixable*
     - Gas
     - Pressure amplitude; sets :math:`T \propto P/n_e`; freeing it changes the spectral shape via :math:`\Lambda_\mathrm{APEC}(T, Z)`

   * - :math:`\alpha_\mathrm{out}` (density)
     - 2.7
     - *Fixable*
     - Gas
     - Outer gNFW slope; controls how quickly the gas profile truncates beyond :math:`R_{200}`

   * - :math:`Z_0` (metallicity amplitude)
     - :math:`Z(0.3\,R_{200}) = 0.3\,Z_\odot`
     - *Fixable*
     - Gas
     - Normalisation of the metallicity profile; shifts :math:`\Lambda_\mathrm{APEC}` by ~30% if doubled

   * - :math:`\sigma_\mathrm{dex}` (AGN scatter)
     - 0.8
     - *Fixable*
     - AGN
     - Log-normal scatter in :math:`L_X` at fixed :math:`M_h`; partly degenerate with :math:`A_\mathrm{AGN}`

   * - :math:`f_\mathrm{sat,AGN}`
     - 0.10
     - *Fixable*
     - AGN
     - Fraction of satellite galaxies hosting AGN; affects the angular scale dependence of the AGN term

   * - :math:`\log_{10} M_{*,0}` (``lg_m0star``)
     - 10.31
     - Fixed (ZM15)
     - iHOD
     - SHMR pivot stellar mass; held at ZM15 Table 2 value

   * - :math:`\beta` (``beta``)
     - 0.33
     - Fixed (ZM15)
     - iHOD
     - Low-mass SHMR power-law slope

   * - :math:`\delta` (``delta``)
     - 0.42
     - Fixed (ZM15)
     - iHOD
     - High-mass SHMR transition exponent

   * - :math:`\gamma` (``gamma``)
     - 1.21
     - Fixed (ZM15)
     - iHOD
     - High-mass SHMR power-law slope

   * - :math:`\eta` (``eta``)
     - −0.04
     - Fixed (ZM15)
     - iHOD
     - Mass-dependent scatter slope

   * - :math:`f_c` (``fc``)
     - 0.86
     - Fixed (ZM15)
     - iHOD
     - Central galaxy fraction

   * - :math:`B_\mathrm{sat}` (``bsat``)
     - 8.98
     - Fixed (ZM15)
     - iHOD
     - Satellite halo-mass normalisation

   * - :math:`\beta_\mathrm{sat}`, :math:`\beta_\mathrm{cut}`
     - 0.90, 0.41
     - Fixed (ZM15)
     - iHOD
     - Satellite and cut-off mass-scaling slopes

   * - :math:`B_\mathrm{cut}` (``bcut``)
     - 0.86
     - Fixed (ZM15)
     - iHOD
     - Central cut-off normalisation

   * - :math:`\alpha_\mathrm{in},\,\alpha_\mathrm{tr}` (density)
     - 1.0, 1.9
     - Fixed (DPM)
     - Gas
     - Inner slope and transition steepness of the gNFW density profile

   * - :math:`\gamma_n` (density redshift)
     - 2.0
     - Fixed (DPM)
     - Gas
     - Redshift scaling exponent :math:`E(z)^{\gamma_n}`

   * - :math:`\gamma^P,\,\alpha^P_\mathrm{out}(M)`
     - 8/3, mass-dep.
     - Fixed (DPM)
     - Gas
     - Pressure profile redshift slope and (mass-dependent) outer slope

   * - :math:`\alpha^Z_\mathrm{in},\,\alpha^Z_\mathrm{tr},\,\alpha^Z_\mathrm{out}`
     - 0, 0.5, 0.7
     - Fixed (DPM)
     - Gas
     - Metallicity profile shape parameters

   * - :math:`f_\mathrm{DC}(z)` (duty cycle table)
     - 0.038–0.50
     - Fixed (C19)
     - AGN
     - AGN duty cycle at six redshift nodes (Comparat+2019 Table 3)

   * - Obscuration constants (:math:`L_{ll}`, :math:`f_{CT}`, etc.)
     - see eqs. above
     - Fixed (C19)
     - AGN
     - All numerical constants in Comparat+2019 eqs. 4–11

   * - K-correction table (:math:`k(z, N_H)`)
     - precomputed
     - Fixed (C19)
     - AGN
     - Absorbed power-law (:math:`\Gamma=1.9,\,f_\mathrm{scat}=0.02`) integrated over :math:`35\times16` grid

   * - :math:`c(M,z)` concentration
     - Diemer+2019
     - Fixed
     - Gas
     - Diemer & Joyce 2019 mass–concentration relation used for all gas profiles

----

BGS LS10 MAP results — samples S1–S3
--------------------------------------

Fitting script:
:mod:`hod_mod.scripts.fitting.fit_comparat2025`
(``--mode map``, ``--sample S1``).
Data: BGS LS10 galaxy catalogue × eROSITA all-sky soft X-ray (0.5–2 keV)
+ BGS LS10 clustering :math:`w_p(r_p)` from
Comparat et al. 2025 (`arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_).

.. list-table::
   :header-rows: 1
   :widths: 8 16 12 18 12 18 12

   * - Sample
     - :math:`\log_{10} M_*^\mathrm{min}`
     - :math:`z_\mathrm{mean}`
     - :math:`N_\mathrm{gal}`
     - :math:`\chi^2/\mathrm{dof}`
     - n\ :sub:`pts` (:math:`w_\theta` + :math:`w_p`)
     - Status
   * - S1
     - 10.00
     - 0.135
     - 2 759 238
     - **3.85**
     - 57 (31 + 26)
     - elevated
   * - S2
     - 10.25
     - 0.162
     - 3 308 841
     - **0.07**
     - 31 (:math:`w_\theta` only)
     - good
   * - S3
     - 10.50
     - 0.191
     - 3 263 228
     - **23.17**
     - 57 (31 + 26)
     - poor fit

.. list-table:: MAP parameters
   :header-rows: 1
   :widths: 30 18 18 18

   * - Parameter
     - S1
     - S2
     - S3
   * - ``log10_A_gas``
     - 3.988
     - 6.802
     - 3.626
   * - ``beta_gas``
     - 0.244
     - 0.232
     - 0.618
   * - ``beta_pressure``
     - 0.891
     - 0.909
     - 2.000 (at bound)
   * - ``log10_A_AGN``
     - 0.086
     - 0.132
     - 0.364
   * - ``log10m_star_thresh``
     - 9.546
     - 10.234
     - 9.000 (at bound)
   * - ``sigma_lnmstar``
     - 0.562
     - 0.500
     - 0.871
   * - ``lg_m1h``
     - 9.988
     - 12.061
     - 9.500 (at bound)
   * - ``alpha_sat``
     - 0.924
     - 1.054
     - 0.822

.. note::
   S1 column refreshed 2026-06-16 with the current code
   (``fit_comparat2025.py``); the previous S1 entries here (:math:`\chi^2/\mathrm{dof}=23.00`,
   ``log10_A_gas=7.760``, ``log10_A_AGN=0.231``) were generated before
   same-day fixes to the gas/AGN cross-spectra code and are stale. S2 and S3
   have not been refreshed and may likewise be out of date relative to the
   current code.

.. note::
   S1 (:math:`\chi^2/\mathrm{dof}=3.85`) and S3 (:math:`\chi^2/\mathrm{dof}=23.17`,
   not yet refreshed) remain :math:`\gg 1`.
   The ZM15 iHOD model was calibrated on SDSS
   (:math:`z \approx 0.1`, small-scale :math:`r_p \gtrsim 0.05\,h^{-1}\,\mathrm{Mpc}`).
   BGS LS10 spans higher redshifts and includes scales
   :math:`r_p < 0.1\,h^{-1}\,\mathrm{Mpc}` where fibre-collision corrections and
   non-linear satellite dynamics may introduce systematic offsets.
   S2 (:math:`\chi^2/\mathrm{dof} = 0.07`) is fitted with :math:`w_\theta` only
   (no :math:`w_p` data available for this sample), which accounts for the
   much lower :math:`\chi^2`.

   S3 ``beta_pressure`` and ``lg_m1h`` reach their prior boundaries —
   this sample requires wider priors or a different model.

AGN luminosity calibration check
---------------------------------

``log10_A_AGN`` is a free linear amplitude on a normalized PSF template
(:func:`~hod_mod.scripts.fitting.fit_comparat2025._psf_template`), fit
directly against the dimensionless :math:`w(\theta)` data. By construction it
has **no built-in link** to :class:`~hod_mod.galaxies.agn_ham.HamAGNModel`'s
predicted mean soft X-ray AGN luminosity — so "``log10_A_AGN`` close to 0"
is not, by itself, evidence that the model's AGN luminosity is correct.

:mod:`hod_mod.scripts.fitting.audit_agn_lx_comparat2025` builds that missing
link using the background-subtraction technique of Comparat et al. 2025
(:math:`S_X^G(R) = (1+w(R))\times S_R^X`, their Eq. 3 and Table 2): the
fitted AGN amplitude is converted into a physical excess surface-brightness
profile and integrated over area to give a mean AGN luminosity, which is then
compared to (a) :meth:`HamAGNModel.mean_agn_lx` evaluated with the
central-galaxy-occupation-weighted halo population (matching the paper's
"AGN only in centrals" assumption), and (b) Comparat+2025 Table 4's
independently deduced point-source luminosity.

.. list-table:: AGN luminosity audit — sample S1 (refreshed 2026-06-16)
   :header-rows: 1
   :widths: 30 20 20

   * - Quantity
     - Value
     - Notes
   * - :math:`L_X` implied by the fit
     - :math:`3.08\times10^{40}` erg/s
     - via :math:`A_\mathrm{AGN}\times\mathrm{PSF}(\theta)\times S_R^X`
   * - :math:`L_X` deduced, Comparat+2025 Table 4
     - :math:`3.59\times10^{40}` erg/s
     - independent measurement; ratio to fit-implied = 0.86

The fit-implied value agrees with the paper's independently-deduced value to
14% — the background-subtraction conversion (
:mod:`hod_mod.scripts.fitting.audit_agn_lx_comparat2025`) is working
correctly. The background-value table used (Table 2) calibrates the
Davis-Peebles stacking estimator, while ``wtheta`` here is the Landy-Szalay
estimator — the paper shows these agree to ~5-10% over 20-500 kpc (proper)
and diverge outside that range, a systematic worth keeping in mind.

**``HamAGNModel`` calibration against Table 4 (all 7 samples).**
:mod:`hod_mod.scripts.fitting.calibrate_ham_agn_lx` fits 3 free parameters
added to ``mean_agn_log10lx`` — ``scatter_lx`` (overrides the constructor's
scatter, cheap since it never touches the abundance-matching precompute),
``log10_A_kcorr`` (rescales the K-correction, clamped ≤1), ``log10_A_dc``
(rescales the duty cycle used in population-averaging only, not the one
baked into the abundance-matching table) — against Table 4, using
``_TABLE3``-default HOD parameters throughout (decoupled from any w(θ)
MAP-fit staleness).

.. list-table:: Raw HamAGNModel vs. Table 4, before calibration (2026-06-16)
   :header-rows: 1
   :widths: 10 14 16 16 12

   * - Sample
     - floor_fraction
     - :math:`L_X` raw [erg/s]
     - :math:`L_X` paper [erg/s]
     - ratio
   * - S1
     - 0.868
     - :math:`4.24\times10^{39}`
     - :math:`3.59\times10^{40}`
     - 0.12
   * - S2
     - 0.000
     - :math:`2.77\times10^{41}`
     - :math:`4.54\times10^{40}`
     - 6.10
   * - S3
     - 0.000
     - :math:`4.62\times10^{41}`
     - :math:`6.20\times10^{40}`
     - 7.45
   * - S4
     - 0.000
     - :math:`8.69\times10^{41}`
     - :math:`8.63\times10^{40}`
     - 10.06
   * - S5
     - 0.000
     - :math:`1.74\times10^{42}`
     - :math:`7.60\times10^{40}`
     - 22.92
   * - S6
     - 0.000
     - :math:`3.46\times10^{42}`
     - :math:`7.43\times10^{40}`
     - 46.58
   * - S7
     - 0.000
     - :math:`7.39\times10^{42}`
     - :math:`5.53\times10^{40}`
     - 133.54

Two distinct, independent problems, not one:

1. **S1 (only): floor-saturated, not a calibration target.** 86.8% of S1's
   HOD-:math:`N_\mathrm{cen}`-weighted population sits at ``HamAGNModel``'s
   physical luminosity floor (``_LOG10_LX_MIN_PHYSICAL = 40.0`` dex). This
   is not a numerical-resolution artifact: the Aird+2015 faint-end slope
   makes the abundance-matched luminosity diverge (verified directly, no
   convergence extending the grid down to 26 dex) for halos as low-mass as
   S1's stellar-mass threshold (9.56, the lowest of the 7) pulls in. 40.0
   dex was chosen as a physically-motivated minimum (roughly the
   conventional low-luminosity-AGN/XRB boundary), not derived from data, so
   S1's prediction is dominated by that assumption and excluded from the
   fit below.

2. **S2–S7: a clean, monotonically growing over-prediction with stellar-mass
   threshold (6× at S2 up to 134× at S7) that 3 flat multiplicative
   parameters cannot absorb** — confirmed by running the calibration:

.. list-table:: Calibration result (excluding S1; all 7 gives nearly identical parameters)
   :header-rows: 1
   :widths: 30 18

   * - Parameter
     - Best fit
   * - ``scatter_lx``
     - 0.550 dex (default 0.8)
   * - ``log10_A_kcorr``
     - −0.197
   * - ``log10_A_dc``
     - −0.351

.. list-table:: HamAGNModel after calibration vs. Table 4 (2026-06-16)
   :header-rows: 1
   :widths: 10 16 16 12 16

   * - Sample
     - :math:`L_X` calibrated [erg/s]
     - :math:`L_X` paper [erg/s]
     - ratio
     - :math:`L_X` fit-implied [erg/s]
   * - S1
     - :math:`4.90\times10^{38}`
     - :math:`3.59\times10^{40}`
     - 0.01
     - :math:`3.08\times10^{40}`
   * - S2
     - :math:`3.20\times10^{40}`
     - :math:`4.54\times10^{40}`
     - 0.71
     - :math:`1.80\times10^{40}`
   * - S3
     - :math:`5.34\times10^{40}`
     - :math:`6.20\times10^{40}`
     - 0.86
     - :math:`5.26\times10^{40}`
   * - S4
     - :math:`1.01\times10^{41}`
     - :math:`8.63\times10^{40}`
     - 1.16
     - :math:`2.71\times10^{40}`
   * - S5
     - :math:`2.02\times10^{41}`
     - :math:`7.60\times10^{40}`
     - 2.65
     - n/a (stale)
   * - S6
     - :math:`4.00\times10^{41}`
     - :math:`7.43\times10^{40}`
     - 5.39
     - :math:`1.82\times10^{41}`
   * - S7
     - :math:`8.54\times10^{41}`
     - :math:`5.53\times10^{40}`
     - 15.45
     - :math:`1.69\times10^{41}`

The calibration brings S2–S4 within ~30% of Table 4, but the residual
**grows systematically with stellar-mass threshold** for S5–S7 (2.7× to
15×) — none of the 3 fitted parameters is pinned at its bound, so this isn't
"the optimizer wants to go further and can't"; a flat amplitude genuinely
cannot reproduce a trend that grows over two decades in mass. The most
likely explanation: this codebase weights ``HamAGNModel`` by the **full
galaxy-sample** HOD occupation (``_TABLE3``, the same threshold used for
:math:`w_p(r_p)` clustering), while Comparat+2025's Table 4 values are
deduced using **Comparat+2023's AGN-specific HOD** (a much higher effective
mass threshold than the full galaxy sample, with a free ``ΔM_min`` shift) —
i.e. the two are integrating over different halo populations, increasingly
so at the high-mass end. Adopting an AGN-specific threshold for this
weighting (rather than rescaling ``HamAGNModel``'s amplitude further) is the
recommended next step; not implemented here. See
``results/fits/comparat2025/ham_agn_calibration.json`` for full per-sample
detail and :mod:`hod_mod.scripts.fitting.calibrate_ham_agn_lx`.

.. figure:: ../results/benchmarks/zumandelbaum2015_sdss/benchmark_zumandelbaum2015_wp.png
   :width: 85%
   :alt: ZM15 wp benchmark

   SDSS DR7 :math:`w_p(r_p)` at MAP (ZM15 published parameters).

.. figure:: ../results/benchmarks/zumandelbaum2015_sdss/benchmark_zumandelbaum2015_ds.png
   :width: 85%
   :alt: ZM15 delta sigma benchmark

   SDSS DR7 :math:`\Delta\Sigma(R)` at MAP (ZM15 published parameters).
