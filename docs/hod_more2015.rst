.. _hod_more2015:

More+2015 HOD Model — BOSS CMASS & BGS
========================================

.. list-table::
   :widths: 25 75

   * - **Model class**
     - :class:`~hod_mod.connection.hod.MoreHODModel` (alias ``More2015HODModel``)
   * - **Paper**
     - More et al. 2015, ApJ 806, 2
       (`arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_,
       `DOI:10.1088/0004-637X/806/1/2 <https://doi.org/10.1088/0004-637X/806/1/2>`_)
   * - **Primary survey**
     - BOSS CMASS, :math:`z_\mathrm{eff} = 0.52`
   * - **Observable**
     - Joint :math:`w_p(r_p) + \Delta\Sigma(R)`
   * - **Code**
     - :mod:`hod_mod.connection.hod` (lines 420–536),
       :mod:`hod_mod.observables.clustering` (:class:`~hod_mod.observables.clustering.FullHaloModelPrediction`)

----

Cosmological framework
-----------------------

Both HOD models in this package share the same halo-model backbone.
All quantities below feed into the HOD occupation integrals.

.. rubric:: Cosmological parameters

The six base parameters :math:`\boldsymbol{\theta} = (\Omega_m,\,\Omega_b,\,h,\,n_s,\,\ln 10^{10}A_s,\,\sigma_8)`
define the linear matter power spectrum :math:`P_\mathrm{lin}(k, z)`, computed by
`CAMB <https://camb.readthedocs.io>`_ via :class:`~hod_mod.core.LinearPowerSpectrum`.

Fiducial values used for BOSS CMASS benchmarks:
:math:`\Omega_m = 0.310,\ h = 0.703,\ \sigma_8 = 0.785,\ n_s = 0.964,\ \Omega_b = 0.0451`.

.. rubric:: Halo mass function

Tinker et al. 2008 (`arXiv:0803.2706 <https://arxiv.org/abs/0803.2706>`_):

.. math::

   \frac{\mathrm{d}n}{\mathrm{d}M}(M, z)

Implemented via :func:`~hod_mod.core.halo_mass_function.make_hmf` with
``backend="tinker08"`` (default), overdensity :math:`\Delta = 200\rho_m`.
Units: :math:`h^4\,\mathrm{Mpc}^{-3}\,M_\odot^{-1}`.

Alternative emulator backends — ``"csst"`` (Chen+2025,
`SCPMA 2025 <https://ui.adsabs.harvard.edu/abs/2025SCPMA..6809513C>`_)
and ``"aemulusnu"`` (Shen+2025,
`arXiv:2410.00913 <https://arxiv.org/abs/2410.00913>`_) — expose the
same interface; see :doc:`cosmology` for details.

.. rubric:: Linear halo bias

Tinker et al. 2010 (`arXiv:1001.3162 <https://arxiv.org/abs/1001.3162>`_),
:math:`b(M, z)`. The effective galaxy bias is:

.. math::

   b_\mathrm{eff}(z) =
   \frac{\displaystyle\int \mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}\,
         \langle N_\mathrm{tot}(M)\rangle\,b(M,z)}
        {\bar{n}_g}

.. rubric:: NFW profile and Fourier transform

Dark-matter halos follow the NFW profile
(Navarro, Frenk & White 1997, `arXiv:astro-ph/9508025 <https://arxiv.org/abs/astro-ph/9508025>`_):

.. math::

   \rho(r \,|\, M) = \frac{\rho_s}{(r/r_s)(1 + r/r_s)^2},
   \qquad r_s = \frac{r_{200}}{c(M,z)}

Concentration–mass relation: Diemer & Joyce 2019
(`arXiv:1809.07326 <https://arxiv.org/abs/1809.07326>`_),
accessed via :class:`~hod_mod.observables.clustering.HaloProfile` with ``cm_relation="diemer19"``.

The normalised NFW Fourier transform (Cooray & Sheth 2002, Eq. 11,
`arXiv:astro-ph/0206508 <https://arxiv.org/abs/astro-ph/0206508>`_):

.. math::

   \tilde{u}(k \,|\, M) =
   \frac{4\pi\rho_s r_s^3}{M}\Bigl[
     \sin(k r_s)\bigl(\mathrm{Si}((1+c)k r_s) - \mathrm{Si}(k r_s)\bigr)
   - \frac{\sin(c k r_s)}{(1+c)k r_s}
   + \cos(k r_s)\bigl(\mathrm{Ci}((1+c)k r_s) - \mathrm{Ci}(k r_s)\bigr)
   \Bigr]

.. rubric:: Galaxy number density

.. math::

   \bar{n}_g(z) = \int_{M_\mathrm{min}}^{M_\mathrm{max}}
   \frac{\mathrm{d}n}{\mathrm{d}M}\,\langle N_\mathrm{tot}(M)\rangle\,\mathrm{d}M

Mass grid: 512 log-spaced points, :math:`M \in [10^{10},\,10^{16}]\,h^{-1}M_\odot`.

----

More+2015 HOD model
--------------------

Reference: More et al. 2015, ApJ 806, 2
(`arXiv:1407.1856 <https://arxiv.org/abs/1407.1856>`_).
Implemented in :class:`~hod_mod.connection.hod.MoreHODModel`
(``hod_mod/connection/hod/more15.py``).

.. rubric:: Incompleteness function

The BOSS CMASS sample has a colour–magnitude selection that reduces completeness
at the low-mass end. More+2015 model this with a linear ramp:

.. math::

   f_\mathrm{inc}(M) = \mathrm{clip}\!\left(
     1 + \alpha_\mathrm{inc}\,\bigl(\log_{10} M - \log_{10} M_\mathrm{inc}\bigr),
     \;0,\;1\right)

Default: :math:`\alpha_\mathrm{inc} = 1.0` (fixed), :math:`\log_{10} M_\mathrm{inc} = 13.0` (fixed).

.. rubric:: Central occupation

.. math::

   \langle N_\mathrm{cen}(M)\rangle =
   \frac{f_\mathrm{inc}(M)}{2}\,
   \mathrm{erfc}\!\left[\frac{\log_{10} M_\mathrm{min} - \log_{10} M}{\sigma_{\log M}}\right]

The step-function threshold :math:`M_\mathrm{min}` is broadened by scatter
:math:`\sigma_{\log M}` (in dex, base-10).
At :math:`M = M_\mathrm{min}`, :math:`\langle N_\mathrm{cen}\rangle = f_\mathrm{inc}/2`.

.. rubric:: Satellite occupation

.. math::

   \langle N_\mathrm{sat}(M)\rangle =
   \langle N_\mathrm{cen}(M)\rangle \times
   \left(\frac{M - \kappa\,M_\mathrm{min}}{M_1}\right)^\alpha
   \quad \text{for } M > \kappa\,M_\mathrm{min},
   \quad \text{else } 0

Satellites live in halos that first contain at least one central galaxy;
their mean number rises as a power law :math:`\alpha` above the threshold
:math:`\kappa\,M_\mathrm{min}`.

.. rubric:: Off-centering of central galaxies

A fraction :math:`p_\mathrm{off}` of centrals are displaced from the halo centre
(Johnston et al. 2007, `arXiv:0709.4193 <https://arxiv.org/abs/0709.4193>`_;
More+2015 §3.3).  In Fourier space (mass-dependent width):

.. math::

   \langle N_\mathrm{cen}^\mathrm{eff}(k \,|\, M)\rangle =
   \langle N_\mathrm{cen}(M)\rangle\,
   \bigl[(1 - p_\mathrm{off})
   + p_\mathrm{off}\,e^{-k^2 (R_\mathrm{off}\,r_s(M))^2/2}\bigr]

where :math:`r_s(M) = r_{200}(M)/c(M)`.
Fixed values: :math:`p_\mathrm{off} = 0.34`, :math:`R_\mathrm{off} = 2.2`.

----

Power spectra
--------------

.. rubric:: 1-halo terms

Galaxy–galaxy (More+2015 Eq. 9):

.. math::

   P_{gg}^\mathrm{1h}(k) = \frac{1}{\bar{n}_g^2}
   \int \mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}
   \left[
     \langle N_s^2\rangle\,\tilde{u}^2(k|M)
     + 2\,\langle N_c\rangle\,\langle N_s\rangle\,\tilde{u}(k|M)
   \right]

Galaxy–matter (More+2015 Eq. 13):

.. math::

   P_{gm}^\mathrm{1h}(k) = \frac{1}{\bar{n}_g}
   \int \mathrm{d}M\,\frac{\mathrm{d}n}{\mathrm{d}M}
   \left[\langle N_c^\mathrm{eff}(k|M)\rangle + \langle N_s\rangle\,\tilde{u}(k|M)\right]
   \frac{M}{\bar{\rho}_m}\,\tilde{u}(k|M)

For the satellite term, a Poisson satellite distribution gives
:math:`\langle N_s^2\rangle = \langle N_s\rangle^2 + \langle N_s\rangle`.

.. rubric:: 2-halo terms

.. math::

   P_{gg}^\mathrm{2h}(k) &= b_\mathrm{eff}^2\,P_\mathrm{lin}(k)
   \quad [+ \delta P_\mathrm{BNL}(k)]\\
   P_{gm}^\mathrm{2h}(k) &= b_\mathrm{eff}\,P_\mathrm{lin}(k)

The beyond-linear halo bias (BNL) correction :math:`\delta P_\mathrm{BNL}` follows
Mead & Verde 2021 (`arXiv:2109.15266 <https://arxiv.org/abs/2109.15266>`_),
tabulated from the MultiDark MDR1 simulation, implemented in
:class:`~hod_mod.core.beyond_linear_bias.BeyondLinearBiasMead21`.

Total:

.. math::

   P_{gg}(k) = P_{gg}^\mathrm{1h}(k) + P_{gg}^\mathrm{2h}(k),\qquad
   P_{gm}(k) = P_{gm}^\mathrm{1h}(k) + P_{gm}^\mathrm{2h}(k)

----

Summary statistics
-------------------

.. rubric:: 3D correlation function

The galaxy auto-correlation function :math:`\xi_{gg}(r)` and galaxy–matter
cross-correlation :math:`\xi_{gm}(r)` are obtained from the respective power spectra
via the Ogata (2005) double-exponential :math:`j_0` Hankel transform
(DOI:`10.2977/prims/1145474602 <https://doi.org/10.2977/prims/1145474602>`_):

.. math::

   \xi(r) = \frac{1}{2\pi^2}\int_0^\infty k^2\,P(k)\,j_0(kr)\,\mathrm{d}k

.. rubric:: Projected correlation function

.. math::

   w_p(r_p) = 2\int_0^{\pi_\mathrm{max}}
   \xi_{gg}\!\left(\sqrt{r_p^2 + \pi^2}\right)\mathrm{d}\pi,
   \qquad \pi_\mathrm{max} = 100\,h^{-1}\,\mathrm{Mpc}

More+2015 use :math:`\pi_\mathrm{max} = 80\,h^{-1}\,\mathrm{Mpc}` (set via ``pi_max``
in :meth:`~hod_mod.observables.clustering.FullHaloModelPrediction.wp`).

.. rubric:: Excess surface mass density

The galaxy–matter lensing signal:

.. math::

   \Sigma_{gm}(R) = 2\int_0^\infty \xi_{gm}\!\left(\sqrt{R^2 + \chi^2}\right)
   \bar{\rho}_m\,\mathrm{d}\chi

.. math::

   \Delta\Sigma(R) =
   \bar{\Sigma}_{gm}(<R) - \Sigma_{gm}(R)
   = \frac{2}{R^2}\int_0^R R'\,\Sigma_{gm}(R')\,\mathrm{d}R'
   - \Sigma_{gm}(R)

Units: :math:`M_\odot\,h\,\mathrm{pc}^{-2}`.
Implemented in
:meth:`~hod_mod.observables.clustering.FullHaloModelPrediction.delta_sigma`.

----

Parameter table
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 12 18 22

   * - Parameter
     - Symbol
     - Default
     - Fitted?
     - Prior / fixed value
     - Units
   * - ``log10mmin``
     - :math:`\log_{10} M_\mathrm{min}`
     - 13.03
     - Yes
     - :math:`[11,\,15]`
     - :math:`\log_{10}(M_\odot h^{-1})`
   * - ``sigma_logm``
     - :math:`\sigma_{\log M}`
     - 0.38
     - Yes
     - :math:`[0.01,\,2]`
     - dex (base 10)
   * - ``log10m1``
     - :math:`\log_{10} M_1`
     - 14.00
     - Yes
     - :math:`[11,\,16]`
     - :math:`\log_{10}(M_\odot h^{-1})`
   * - ``alpha``
     - :math:`\alpha`
     - 1.0
     - Yes
     - :math:`[0.1,\,3]`
     - —
   * - ``kappa``
     - :math:`\kappa`
     - 1.0
     - Yes
     - :math:`[0.01,\,5]`
     - —
   * - ``alpha_inc``
     - :math:`\alpha_\mathrm{inc}`
     - 1.0
     - **Fixed**
     - 1.0
     - —
   * - ``log10m_inc``
     - :math:`\log_{10} M_\mathrm{inc}`
     - 13.0
     - **Fixed**
     - 13.0
     - :math:`\log_{10}(M_\odot h^{-1})`
   * - ``p_off``
     - :math:`p_\mathrm{off}`
     - 0.34
     - **Fixed**
     - 0.34
     - —
   * - ``R_off``
     - :math:`R_\mathrm{off}`
     - 2.2
     - **Fixed**
     - 2.2
     - :math:`r_s` units

----

BOSS CMASS benchmarks
----------------------

Three stellar-mass threshold subsamples from More+2015 Figure 3 are reproduced.
Full MAP results and MCMC corner plots are in :ref:`benchmark_more2015`.

Data digitised from Figure 3 of More+2015 using WebPlotDigitizer;
stored in ``data/more2015_boss_cmass/``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 15 20

   * - Variant
     - :math:`\log_{10} M_*^\mathrm{min}`
     - :math:`\chi^2`
     - dof
     - :math:`\chi^2/\mathrm{dof}`
   * - ``logM11_12``
     - 11.1
     - 71.06
     - 36
     - **1.967** (pub. 0.8)
   * - ``logM11p3_12``
     - 11.3
     - 57.60
     - 35
     - **1.646** (pub. 1.3)
   * - ``logM11p4_12``
     - 11.4
     - 63.30
     - 35
     - **1.809** (pub. 1.5)
   * - ``logM11_12_freecosmo``
     - 11.1 + free :math:`\Omega_m,S_8`
     - 35.70
     - 33
     - **1.082**

.. rubric:: Primary benchmark: ``logM11_12`` (MAP parameters)

.. list-table::
   :header-rows: 1
   :widths: 22 16 24 18

   * - Parameter
     - MAP
     - Published (:math:`\pm 1\sigma`)
     - Deviation
   * - ``log10mmin``
     - 13.134
     - :math:`13.13 \pm 0.13`
     - :math:`+0.03\sigma`
   * - ``sigma_logm``
     - 0.458
     - :math:`0.469 \pm 0.13`
     - :math:`-0.09\sigma`
   * - ``log10m1``
     - 14.168
     - :math:`14.21 \pm 0.13`
     - :math:`-0.32\sigma`
   * - ``alpha``
     - 1.841
     - :math:`1.13 \pm 0.15`
     - :math:`+4.74\sigma`
   * - ``kappa``
     - 3.000
     - :math:`1.25 \pm 0.45`
     - :math:`+3.89\sigma`

.. note::
   ``alpha`` and ``kappa`` tensions are driven by a near-degenerate likelihood valley.
   MCMC medians agree much better: ``alpha`` = 1.928 (±0.19), ``kappa`` = 1.862 (+0.79/−1.03).
   All mass-scale parameters agree within :math:`0.32\sigma`.

.. rubric:: Variant ``logM11p3_12`` (MAP)

.. list-table::
   :header-rows: 1
   :widths: 22 16 24 18

   * - Parameter
     - MAP
     - Published (:math:`\pm 1\sigma`)
     - Deviation
   * - ``log10mmin``
     - 13.549
     - :math:`13.45 \pm 0.15`
     - :math:`+0.66\sigma`
   * - ``sigma_logm``
     - 0.616
     - :math:`0.671 \pm 0.19`
     - :math:`-0.29\sigma`
   * - ``log10m1``
     - 14.548
     - :math:`14.51 \pm 0.17`
     - :math:`+0.22\sigma`
   * - ``alpha``
     - 2.361
     - :math:`1.14 \pm 0.49`
     - :math:`+2.49\sigma`
   * - ``kappa``
     - 0.148
     - not published
     - —

.. rubric:: Variant ``logM11p4_12`` (MAP)

.. list-table::
   :header-rows: 1
   :widths: 22 16 24 18

   * - Parameter
     - MAP
     - Published (:math:`\pm 1\sigma`)
     - Deviation
   * - ``log10mmin``
     - 14.166
     - :math:`13.68 \pm 0.16`
     - :math:`+3.04\sigma`
   * - ``sigma_logm``
     - 0.875
     - :math:`0.889 \pm 0.22`
     - :math:`-0.06\sigma`
   * - ``log10m1``
     - 14.390
     - :math:`14.56 \pm 0.25`
     - :math:`-0.68\sigma`
   * - ``alpha``
     - 1.602
     - :math:`1.00 \pm 0.44`
     - :math:`+1.37\sigma`
   * - ``kappa``
     - 1.675
     - not published
     - —

.. rubric:: Free-cosmology variant ``logM11_12_freecosmo`` (MAP)

Three additional free parameters with Planck 2018 priors:
:math:`\Omega_m = 0.310 \pm 0.020`, :math:`S_8 \equiv \sigma_8\sqrt{\Omega_m/0.3} = 0.798 \pm 0.044`,
:math:`h = 0.703` (fixed at the published value).

.. list-table::
   :header-rows: 1
   :widths: 22 16 24

   * - Parameter
     - MAP
     - Planck prior centre
   * - :math:`\Omega_m`
     - 0.281
     - :math:`0.310 \pm 0.020`
   * - :math:`S_8`
     - 0.778
     - :math:`0.798 \pm 0.044`
   * - ``log10mmin``
     - 13.163
     - —
   * - ``sigma_logm``
     - 0.508
     - —
   * - ``log10m1``
     - 14.224
     - —
   * - ``alpha``
     - 2.018
     - —
   * - ``kappa``
     - 2.920
     - —

----

Benchmark figures (``logM11_12``)
-----------------------------------

.. figure:: _images/benchmarks__more2015_logM11_12__benchmark_more2015_logM11_12_combined.png
   :width: 95%
   :alt: more2015 combined wp and delta sigma

   MAP fit to BOSS CMASS logM*>11.1.
   *Top panel*: :math:`w_p(r_p)`.
   *Bottom panel*: :math:`\Delta\Sigma(R)`.
   Orange: published More+2015 parameters.  Blue: MAP.  Grey: data.

.. figure:: _images/benchmarks__more2015_logM11_12__benchmark_more2015_logM11_12_hod.png
   :width: 70%
   :alt: more2015 HOD occupation functions

   HOD occupation functions :math:`\langle N_c(M)\rangle`, :math:`\langle N_s(M)\rangle`,
   and :math:`\langle N_\mathrm{tot}(M)\rangle`.
   Solid: MAP. Dashed + band: MCMC median ± 1σ.  Orange: published values.

.. figure:: _images/benchmarks__more2015_logM11_12__benchmark_more2015_logM11_12_corner.png
   :width: 95%
   :alt: more2015 MCMC corner plot

   MCMC posterior corner plot (32 walkers × 2000 steps, 500 burn-in = 48 000 samples).
   Contours: 68% and 95% credible regions.  Orange lines: published More+2015 values.

----

BGS LS10 — preliminary results (S4–S7)
----------------------------------------

The BGS LS10 cross-correlation :math:`w_\theta(\theta)` (galaxy × eROSITA soft X-ray)
and :math:`w_p(r_p)` were fitted jointly for higher stellar-mass samples
(Comparat et al. 2025, `arXiv:2503.19796 <https://arxiv.org/abs/2503.19796>`_).
Samples S4–S7 used :class:`~hod_mod.connection.hod.MoreHODModel` parameters.

.. list-table::
   :header-rows: 1
   :widths: 8 16 12 18 12 14 14 14

   * - Sample
     - :math:`\log_{10} M_*^\mathrm{min}`
     - :math:`z_\mathrm{mean}`
     - :math:`N_\mathrm{gal}`
     - :math:`\chi^2/\mathrm{dof}`
     - n\ :sub:`pts`
     - ``log10mmin``
     - ``log10m1``
   * - S4
     - 10.75
     - 0.226
     - 2 802 710
     - 316.60
     - 31
     - 12.327
     - 13.358
   * - S5
     - 11.00
     - 0.252
     - 1 619 838
     - 242.79
     - 57
     - 12.674
     - 13.692
   * - S6
     - 11.25
     - 0.255
     - 541 855
     - 314.32
     - 31
     - 13.096
     - 14.132
   * - S7
     - 11.50
     - 0.261
     - 120 882
     - MAP failed
     - 57
     - —
     - —

.. note::
   For S4–S6, the gas amplitude ``log10_A_gas`` converges at its lower bound (−2.0),
   indicating that the gas component is not detected in these samples at current data quality.
   :math:`\chi^2/\mathrm{dof} \gg 1` reflects a combination of model inadequacy,
   data systematics, and the gas non-detection.
   These results are preliminary; see :ref:`hod_zumandelbaum2015` for the
   lower-mass samples fitted with the iHOD model.
