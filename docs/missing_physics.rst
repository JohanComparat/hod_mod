What the model does not yet contain
===================================

*Missing physics and implementation propositions.*

The other forecast pages catalogue extensions that stay **within** the current
physics: :doc:`sensitivity_fisher` lists the parameters that could be freed and
the observables the pipeline already knows how to compute, and
:doc:`tier2_forecast` states the caveats of the 61-parameter study.  This page
covers what is **not in the model at all**: eight sectors of physics the
pipeline cannot yet describe, each with (i) a precise statement of where the
code stands, (ii) a concrete implementation proposition — module names,
equations, the exact parameter-vector entries it would append, and whether it
is native JAX or needs the :mod:`~hod_mod.forecast.apec_bands`
"distill-to-table" pattern — (iii) the measurements that would constrain the
new parameters, and (iv) key references.  References were assembled from the
2026 HDR bibliography first and supplemented from the literature; every arXiv
link below was resolved and checked against title and first author.

.. contents::
   :local:
   :depth: 1

.. admonition:: Implementation status (2026-07, branch ``feature/missing-physics``)

   Three implementation waves grew the parameter vector 61 → **90** and
   delivered sectors 1–3, 5, 7 and 8 of this page (beyond-ΛCDM growth and
   neutrinos, the cosmology-dependent c(ν, n_eff), the AGN fundamental
   plane with jets and the infrared band, the SF/quiescent split with a
   continuous sSFR distribution and the [OII] LF, wind mass loading, and
   the HI sector), plus the linearized CAMB/EH98 ratio for sector 2 —
   each with exact tested invariants and survey-noise wiring into the
   tier-2 forecast.  **The full account lives in**
   :doc:`missing_physics_implementation`.  Still open: ν/HMF emulator
   distillation beyond first order, multi-band SEDs/CLF (route 1/2), and
   morphology.

Summary and suggested order
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 22 16 12 24 6

   * - Topic
     - Current state
     - Proposed module
     - New params
     - Primary constraining data
     - Effort
   * - Beyond-ΛCDM
     - w0/wa in geometry only (:mod:`hod_mod.core.distances`); ΛCDM growth + EH98 P(k)
     - ``forecast/growth_ode.py`` + ν tables
     - 3 (+2)
     - DESI BAO/FS, 3×2pt, cluster counts + CMB lensing
     - M
   * - Precision ingredients
     - universal f(σ)/bias; fixed-cosmology c(M); EH98 shape
     - wire :func:`~hod_mod.core.concentration.c_diemer15`; ``forecast/pk_camb_ratio.py``
     - 0 (correctness)
     - Euclid×eROSITA masses, small-scale ΔΣ, DESI full shape
     - S–M
   * - AGN radio/IR/FP
     - X-ray only (:mod:`hod_mod.agn`)
     - ``agn/multiband.py``
     - 4 (+1)
     - LoTSS, VLASS/RACS, WISE/SPHEREx clustering + LFs
     - M
   * - Galaxy morphology
     - nothing
     - ``connection/morphology.py``
     - 3 (+1)
     - Euclid VIS morphologies, COSMOS-Web, group catalogues
     - L
   * - sSFR / SF-vs-Q
     - binary red/blue exists, not in forecast
     - (z, M*, SF/Q) cells in ``forecast/tier2.py``
     - 7
     - f_Q(M*,z), split w_p/ΔΣ, eROSITA/SO SF-vs-Q stacks
     - M
   * - SEDs / multi-band LFs
     - single-band CLF (:mod:`hod_mod.connection.clf`)
     - ``connection/multiband.py``
     - 4
     - COSMOS-Web/Euclid/GAMA LFs, colour-split clustering
     - L
   * - Stellar feedback
     - fixed ε_SN = 0.1 energy channel
     - promote ``eps_sn``; wind loading in η(M)
     - 3
     - low-mass SMF, group L_X–M, kSZ profiles
     - S–M
   * - Cold gas / HI
     - nothing (gas sector is hot-phase only)
     - ``gas/hi.py``
     - 3 (+1)
     - ALFALFA/MIGHTEE HIMF, xGASS, CHIME×DESI
     - M

**Suggested order.**  Two items are nearly free and should come first: wiring
the existing cosmology-dependent :func:`~hod_mod.core.concentration.c_diemer15`
into the forecast, and promoting the supernova coupling ``eps_sn`` (the
documented "constant → ``theta``" pattern).  The highest science return per
effort is then the beyond-ΛCDM block — a growth ODE plus a neutrino-suppression
table unlock the :math:`\Sigma m_\nu` target that :doc:`sensitivity_fisher`
names as the headline.  Next come the three medium extensions that mostly reuse
tier-2 infrastructure: the SF/quiescent cell split (the eROSITA CGM data to fit
it already exist), the fundamental-plane AGN sector (the Powell chain already
carries :math:`M_{\rm BH}` and :math:`L_X` per halo), and the HI sector (fully
analytic).  The CAMB-ratio emulator and HMF correction tables are systematic-
error insurance that can proceed in parallel.  The two heavy lifts — multi-band
SEDs and morphology — are best scheduled against the maturity of the Euclid
deep-survey catalogues they would be fitted to.

----

Cosmology beyond ΛCDM
---------------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

The differentiable forecast is strictly flat-ΛCDM: the linear power spectrum is
the Eisenstein & Hu transfer function
(:class:`~hod_mod.forecast.pk_eisenstein_hu.EisensteinHu98PkLinear`) and the
growth factor is the Carroll et al. (1992) fitting formula, which takes
**only** :math:`\Omega_m` (``_growth_factor_flat_jax`` in
:mod:`hod_mod.core.halo_mass_function`).  Ironically, the *geometry* is already
more general: :mod:`hod_mod.core.distances` implements the full CPL dark energy
:math:`w(a) = w_0 + w_a(1-a)` in JAX, and the non-differentiable production
paths (CAMB in :mod:`hod_mod.core.power_spectrum`, the CSST emulator in
:mod:`hod_mod.core.nonlinear`) accept ``w0``, ``wa`` and ``mnu``.  The gap is
therefore precisely *growth and the P(k) shape inside the JAX path* — nothing
else.

.. seealso::
   The five candidate parameters (:math:`w_0, w_a, \Sigma m_\nu, \Omega_k,
   \alpha_s`) and their place in the roadmap are already identified in
   :doc:`sensitivity_fisher` ("Parameters that could be freed", Phase 3);
   :doc:`tier2_forecast` lists "EH98 cosmology" among its caveats.  This
   section adds the *how*.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **Growth ODE** — new module ``hod_mod/forecast/growth_ode.py`` solving

  .. math::

     D'' + \left[\frac{3}{a} + \frac{E'(a)}{E(a)}\right] D'
     - \frac{3}{2}\,\frac{\Omega_m(a)}{a^2}\,D = 0

  with a **fixed-grid RK4 in** :math:`\ln a` (64–128 steps via
  ``jax.lax.scan`` — jit/vmap/jacfwd-safe, no extra dependency), taking
  :math:`E(a)` from the CPL expressions already in
  :mod:`hod_mod.core.distances`.  It replaces ``_growth_factor_flat_jax``
  behind a keyword of :class:`~hod_mod.core.halo_mass_function.HaloMassFunction`
  and :class:`~hod_mod.forecast.pk_eisenstein_hu.EisensteinHu98PkLinear`.
  Regression gate: at :math:`w_0=-1, w_a=0` the ODE reproduces Carroll+1992 to
  better than 0.1%.
* **Massive neutrinos**, two routes: (i) the Eisenstein & Hu (1999) massive-ν
  transfer function or the Kiakotou et al. scale-dependent growth-suppression
  coefficients — native JAX, zero infrastructure; (ii) a CAMB ratio table
  :math:`P(k;\Sigma m_\nu)/P(k;0)` over :math:`(k, \Sigma m_\nu, z)`, built
  once in numpy and evaluated with JAX multilinear interpolation — the
  :mod:`~hod_mod.forecast.apec_bands` pattern verbatim.  Route (ii) is the
  accuracy path; route (i) the fallback.
* **Non-linear boost beyond ΛCDM**: distill the CSST emulator (already wrapped
  in :mod:`hod_mod.core.nonlinear`) or EuclidEmulator2/BACCO boosts
  :math:`B(k, z; w_0, w_a, \Sigma m_\nu)` into an npz grid + JAX
  interpolation.  The experimental ``jax.pure_callback`` bridge in
  :mod:`hod_mod.core.nonlinear` is **not differentiable** and therefore cannot
  serve the Fisher pipeline — the table route is the correct one.
* **HMF response**: to first order the cosmology dependence rides on
  :math:`\sigma(M, z)` for free; the explicit w0waCDM correction is calibrated
  by DUCA [DUCA2025]_.
* **New parameters** (append-only): ``w0`` (fiducial −1), ``wa`` (0), and
  ``sum_mnu`` — introduced at fiducial 0 eV first, so every existing prediction
  is bit-identical, then moved to the Planck baseline 0.06 eV as a deliberate,
  documented model upgrade (the σ8-anchoring convention keeps the amplitude
  fixed; only the small-scale shape shifts by the ~0.5% EH99 suppression).
  ``Omega_k`` and ``alpha_s`` follow the same pattern when needed.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* DESI DR2 (and later) BAO + full-shape multipoles — the current ~3σ w0waCDM
  preference [DESI_DR2_BAO]_ is the science driver.
* Euclid/Rubin tomographic 3×2pt — the growth-vs-geometry split that the
  pipeline's own shear + clustering blocks provide.
* eROSITA cluster abundance dn/dz + SO CMB lensing — the
  :math:`\Sigma m_\nu` combination; re-running the tier-2 vector with the
  extended cosmology quantifies exactly how much the astrophysics
  marginalization costs it.

Key references
~~~~~~~~~~~~~~

DESI DR2 BAO evidence for dynamical dark energy [DESI_DR2_BAO]_
(`arXiv:2503.14738 <https://arxiv.org/abs/2503.14738>`_); the CosmoVerse
tensions white paper [CosmoVerse2025]_
(`arXiv:2504.01669 <https://arxiv.org/abs/2504.01669>`_); Planck 2018 baseline
[PlanckCollaboration2018]_.  Massive-ν transfer functions: [EisensteinHu1999]_
(`arXiv:astro-ph/9710252 <https://arxiv.org/abs/astro-ph/9710252>`_) and
[Kiakotou2008]_ (`arXiv:0709.0253 <https://arxiv.org/abs/0709.0253>`_).
Emulator landscape: EuclidEmulator2 [EuclidEmulator2]_
(`arXiv:2010.11288 <https://arxiv.org/abs/2010.11288>`_), BACCO [BACCO2021]_
(`arXiv:2004.06245 <https://arxiv.org/abs/2004.06245>`_), Mira-Titan IV
[MiraTitanIV]_ (`arXiv:2207.12345 <https://arxiv.org/abs/2207.12345>`_), the
CSST emulators [CSSTEmulatorI]_
(`arXiv:2502.11160 <https://arxiv.org/abs/2502.11160>`_) and [ChenCSST2025]_,
Goku [Goku2025]_ (`arXiv:2501.06296 <https://arxiv.org/abs/2501.06296>`_),
Aletheia [Aletheia2025]_, e-MANTIS for f(R) [eMANTIS2024]_
(`arXiv:2303.08899 <https://arxiv.org/abs/2303.08899>`_); the w0waCDM halo
mass function [DUCA2025]_
(`arXiv:2504.07608 <https://arxiv.org/abs/2504.07608>`_).

----

Cosmology-correct, differentiable halo-model ingredients
--------------------------------------------------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

Every halo-model ingredient in the JAX path is differentiable, but their
*cosmology dependence* is incomplete in a specific, documentable way.  The
fifteen ``f(sigma)`` multiplicity functions in
:mod:`hod_mod.core.halo_mass_function` carry cosmology only through
:math:`\sigma(M, z)` — their fit coefficients are universal.  The large-scale
bias is Tinker et al. (2010) [Tinker2010]_ as a function of peak height only.
Every concentration–mass relation in :mod:`hod_mod.core.concentration` is a
fit frozen at its calibration cosmology, and the forecast uses the Planck13-
frozen ``concentration_dutton14_jax``.  The linear P(k) is EH98, a few-percent
approximation to CAMB [Lewis2002]_ / CLASS [CLASS2011]_.  Crucially, a
cosmology-dependent concentration model **already exists in the package**:
:func:`~hod_mod.core.concentration.c_diemer15` implements the Diemer &
Kravtsov (2015) c(ν, n_eff) model with Diemer & Joyce (2019) parameters
[DiemerJoyce2019]_ — but its spectral-slope helper is numpy
(``np.gradient``/``np.interp``) and it is not wired into the forecast.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **Concentration (the near-free fix)**: port ``_neff_eisenstein_hu`` to jnp
  (a ten-line change: ``jnp.gradient`` + ``jnp.interp`` on the EH98 grid the
  forecast already tabulates) and swap

  .. math::

     c_{200c}(\nu, n_{\rm eff}) = (\phi_0 + \phi_1 n_{\rm eff})
     \left(\frac{\nu}{\eta_0 + \eta_1 n_{\rm eff}}\right)^{-\alpha}
     \left[1 + \left(\frac{\nu}{\eta_0 + \eta_1 n_{\rm eff}}\right)^{\beta}\right]

  for Dutton14 behind a ``cm_relation`` flag of
  :class:`~hod_mod.forecast.forward_jax.ForwardModel`.  Both inputs (ν from
  :math:`\sigma(M, z)`, :math:`n_{\rm eff} = {\rm d}\ln P/{\rm d}\ln k`) live
  on grids the model computes anyway, so concentration finally responds to
  :math:`\sigma_8, n_s, h` — and to Section 1's :math:`w_0, w_a, \Sigma m_\nu`
  through growth.  Regression: match COLOSSUS [Colossus2018]_ at the fiducial
  cosmology.
* **HMF**: quote the tinker08 non-universality error budget against the
  Euclid percent-level calibration [EuclidHMF2023]_; where it matters (cluster
  masses, high z), distill the already-wrapped GP emulators
  (``make_hmf("aemulusnu")`` [ShenAemulus2025]_, ``make_hmf("csst")``
  [ChenCSST2025]_) into a multiplicative correction grid
  :math:`f_{\rm emu}/f_{\rm tinker08}(\sigma, z; {\rm cosmology})` — new
  module ``forecast/hmf_tables.py``, apec_bands pattern.
* **Bias**: Tinker10-through-ν is first-order cosmology-correct already; an
  emulated correction from DarkQuest/Aemulus [Nishimichi2019]_ is possible but
  **explicitly low priority** — the ingredient least in need of repair.
* **CAMB-quality P(k)**: a *ratio* emulator
  :math:`T_{\rm CAMB}(k)/T_{\rm EH98}(k)` over a Sobol design in
  :math:`(h, \Omega_b, \Omega_m, n_s, \Sigma m_\nu, w_0, w_a)`, stored as a
  multilinear table or a small dense MLP whose weights load from npz and
  evaluate in ~10 lines of ``jnp`` (the CosmoPower architecture
  [CosmoPower2022]_ without the framework); it multiplies the EH98 shape and
  preserves the existing σ8 anchoring — new module
  ``forecast/pk_camb_ratio.py``.  Evolution mapping [EvolutionMapping2022]_
  compresses the parameter space the design must cover.
* **New parameters: none.**  This section buys correctness of derivatives the
  vector already contains, not new freedom.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* Cluster mass calibration (Euclid lensing × eROSITA) — the HMF accuracy test.
* Small-scale ΔΣ (HSC/Rubin) and satellite kinematics — the c(M) test the
  model's own :math:`\Delta\Sigma(R < 1\,{\rm Mpc}/h)` block performs.
* DESI full-shape multipoles — the P(k)-shape validation.

Key references
~~~~~~~~~~~~~~

HMF calibrations: [Tinker2008]_, [Despali2016]_
(`arXiv:1507.05627 <https://arxiv.org/abs/1507.05627>`_), [Comparat2017]_
(`arXiv:1702.01628 <https://arxiv.org/abs/1702.01628>`_), the Euclid
Λ(ν)CDM calibration [EuclidHMF2023]_
(`arXiv:2208.02174 <https://arxiv.org/abs/2208.02174>`_) and the emulators
[ShenAemulus2025]_, [ChenCSST2025]_.  Bias: [Tinker2010]_.  Concentration:
[DiemerKravtsov2015]_ (`arXiv:1407.4730 <https://arxiv.org/abs/1407.4730>`_),
[DiemerJoyce2019]_, COLOSSUS [Colossus2018]_
(`arXiv:1712.04512 <https://arxiv.org/abs/1712.04512>`_).  P(k): CAMB
[Lewis2002]_, CLASS [CLASS2011]_
(`arXiv:1104.2932 <https://arxiv.org/abs/1104.2932>`_), CosmoPower
[CosmoPower2022]_ (`arXiv:2106.03846 <https://arxiv.org/abs/2106.03846>`_),
halofit [Smith2003]_, HMcode [Mead2020]_, DarkQuest [Nishimichi2019]_,
evolution mapping [EvolutionMapping2022]_
(`arXiv:2108.12710 <https://arxiv.org/abs/2108.12710>`_).

----

AGN: radio, infrared, and the fundamental plane
-----------------------------------------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

The AGN sector (:mod:`hod_mod.agn`: ``ham``, ``hod``, ``xray``,
``duty_cycle``, ``powell``) terminates exclusively in X-ray luminosity.  There
is no radio or infrared emission channel and no fundamental-plane relation
anywhere in the package — nor, notably, in the HDR itself, which treats radio
AGN only through radio-loud/quiet clustering.  The decisive asset is that the
Powell chain — :math:`M_h \to M_* \to M_{\rm BH} \to \lambda_{\rm Edd} \to
L_{\rm bol} \to L_X` — already runs inside the differentiable forecast
(``_agn_kernel_parts`` / ``_agn_occupation`` / ``_xlf`` in
:mod:`hod_mod.forecast.forward_jax`), so **the black-hole mass and X-ray
luminosity of every halo are already on the grid**; other bands attach with no
new infrastructure.

.. seealso::
   AGN *clustering as an observable* (and what it does for the
   :math:`M_{\rm BH}`–halo correlation) is discussed in
   :doc:`sensitivity_fisher` ("Observables that could be added"); the tier-2
   study implements it in X-rays.  This section is about the missing *bands*.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **Fundamental plane of black-hole activity** — new module
  ``hod_mod/agn/multiband.py`` plus a mirrored kernel in ``forward_jax``:

  .. math::

     \log L_R = \xi_{RX}\,\log L_X + \xi_{RM}\,\log M_{\rm BH} + b_R,
     \qquad \sigma_R \simeq 0.88\ {\rm dex}

  with Merloni, Heinz & di Matteo (2003) coefficients
  :math:`(\xi_{RX}, \xi_{RM}, b_R) = (0.60, 0.78, 7.33)` as fiducials and the
  Gültekin et al. (2019) re-calibration as the external prior.  Applied per
  :math:`(M_h, \lambda_{\rm Edd})` cell after the existing L_X step; the
  lognormal scatter convolves on the same grid the ERDF convolution already
  uses.  A radio luminosity function ``rlf`` observable follows from the same
  weighted-histogram machinery as ``_xlf``.
* **Infrared**: :math:`L_{\rm IR}` from an :math:`L_{\rm bol}`-dependent
  bolometric correction (Hopkins et al. 2007 form).  The existing tier-2
  obscuration parameter ``agn_fabs`` then does double duty: obscuration
  *suppresses* soft X-rays and *boosts* IR re-emission — one parameter tested
  in two bands, a genuine cross-band consistency check of the obscured
  fraction that WISE obscured/unobscured halo-mass measurements probe from
  the clustering side.
* **Radio-loud jets**: the fundamental plane describes the radio-quiet/core
  emission; the jetted HERG/LERG population needs a second component with a
  loud fraction :math:`f_{\rm loud}(M_{\rm BH})` — flagged as stage two
  [BestHeckman2012]_.
* **Clustering**: radio/IR AGN angular cross-correlations
  :math:`C_\ell^{{\rm AGN}\times g}` reuse the existing galaxy Limber kernels
  with the AGN occupation weights.
* **New parameters** (touch only new observables — existing predictions are
  untouched by construction): ``agn_xi_rx`` (0.60), ``agn_xi_rm`` (0.78),
  ``agn_b_r`` (7.33), ``agn_sig_r`` (0.88), optionally ``agn_bc_ir``.  Native
  JAX; only a trivial power-law radio k-correction is needed.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* LoTSS Deep Fields radio LF + clustering of radio AGN [Hale2025]_; wide-area
  counts from VLASS (`arXiv:1907.01981 <https://arxiv.org/abs/1907.01981>`_)
  and ASKAP/RACS (`arXiv:2012.00747 <https://arxiv.org/abs/2012.00747>`_);
  MeerKAT MIGHTEE continuum in the deep fields.
* WISE-selected AGN clustering and obscured/unobscured halo masses
  [Donoso2014]_, [Petter2023]_
  (WISE mission: `arXiv:1008.0031 <https://arxiv.org/abs/1008.0031>`_);
  SPHEREx all-sky IR AGN counts
  (`arXiv:2511.02985 <https://arxiv.org/abs/2511.02985>`_).
* Quasar clustering vs radio loudness [Shen2009]_, [RetanaMontenegro2017]_.
* Fundamental-plane normalisation and scatter priors from [Gultekin2019]_.

Key references
~~~~~~~~~~~~~~

The fundamental plane: [MerloniHeinzDiMatteo2003]_
(`arXiv:astro-ph/0305261 <https://arxiv.org/abs/astro-ph/0305261>`_),
[FalckeKordingMarkoff2004]_
(`arXiv:astro-ph/0305335 <https://arxiv.org/abs/astro-ph/0305335>`_),
[Gultekin2019]_ (`arXiv:1901.02530 <https://arxiv.org/abs/1901.02530>`_).
Bolometric corrections: [Hopkins2007]_
(`arXiv:astro-ph/0605678 <https://arxiv.org/abs/astro-ph/0605678>`_).  Radio
AGN populations and clustering: [BestHeckman2012]_
(`arXiv:1201.2397 <https://arxiv.org/abs/1201.2397>`_), [Shen2009]_
(`arXiv:0810.4144 <https://arxiv.org/abs/0810.4144>`_),
[RetanaMontenegro2017]_
(`arXiv:1611.08630 <https://arxiv.org/abs/1611.08630>`_), [Hale2025]_
(`arXiv:2510.01029 <https://arxiv.org/abs/2510.01029>`_), LoTSS DR2
(`arXiv:2202.11733 <https://arxiv.org/abs/2202.11733>`_).  IR AGN:
[Donoso2014]_ (`arXiv:1309.2277 <https://arxiv.org/abs/1309.2277>`_),
[Petter2023]_ (`arXiv:2302.00690 <https://arxiv.org/abs/2302.00690>`_).
Multi-wavelength AGN mocks: [Comparat2019]_
(`arXiv:1901.10866 <https://arxiv.org/abs/1901.10866>`_); XLF baseline
[Aird2015]_ (`arXiv:1503.01120 <https://arxiv.org/abs/1503.01120>`_).

----

Galaxy morphology
-----------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

**Implemented (wave 4)** — :mod:`hod_mod.connection.morphology` provides the
conditional early-type fraction below; ``ForwardModel(morph="early"|"late")``
splits any sample, the per-cell ``f_early`` observable enters the tier
forecasts (``--include-morph``), and ``mbh_bt_slope`` couples the bulge
proxy into the Powell chain.  See
:doc:`missing_physics_implementation` (wave 4).  The assembly-bias hook
(item 3 below) remains open.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **Conditional early-type fraction** — new module
  ``hod_mod/connection/morphology.py`` mirroring the
  :class:`~hod_mod.connection.hod.zumandelbaum15.ZuMandelbaum16QuenchingModel`
  pattern:

  .. math::

     f_{\rm early,cen}(M_h) = 1 - \exp\!\left[-\left(M_h/M_{\rm morph}
     \right)^{\beta_{\rm morph}}\right],

  with a satellite boost ``f_morph_sat``.
* **Coupling to the black-hole sector**: insert the bulge fraction into the
  Powell chain's :math:`M_{\rm BH}`–:math:`M_*` step
  (:math:`M_{\rm BH} \propto (B/T \cdot M_*)^{\alpha}`; Yang et al. 2019) —
  one slope parameter, and morphology becomes *testable through the X-ray
  luminosity function* the model already predicts.
* **Coupling to assembly**: the cosmology-dependent :math:`c(M, z)` of
  Section 2 provides a formation-time proxy that orders :math:`B/T` at fixed
  halo mass — the natural (and falsifiable) assembly-bias hook
  [WechslerTinker2018]_.
* **Tier-2 integration**: either morphology-split cells (the same mechanism
  as the SF/Q split below) or, far cheaper, a new observable block
  :math:`f_{\rm early}(M_*, z)` fitted alongside the cell grid.
* **New parameters**: ``log10_M_morph`` (~12.5), ``beta_morph`` (~0.8),
  ``f_morph_sat``, optionally ``mbh_bt_slope``.  Native JAX; new observables
  only, so existing predictions are untouched.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* Euclid VIS morphological catalogues: :math:`f_{\rm early}(M_*, z)` and
  morphology-split :math:`w_p` / :math:`\Delta\Sigma`
  (mission: `arXiv:2405.13491 <https://arxiv.org/abs/2405.13491>`_).
* COSMOS-Web and HSC visual/ML morphologies at higher z.
* Morphology–halo-mass trends in group catalogues [Yang2007groups]_,
  [Tinker2021groups]_.

Key references
~~~~~~~~~~~~~~

Galaxy–halo connection review [WechslerTinker2018]_
(`arXiv:1804.03097 <https://arxiv.org/abs/1804.03097>`_); AGN–host morphology
at fixed M*: [Banerjee2025]_
(`arXiv:2310.12943 <https://arxiv.org/abs/2310.12943>`_); black-hole–bulge
coevolution [Yang2019BHbulge]_
(`arXiv:1903.00003 <https://arxiv.org/abs/1903.00003>`_) and compactness
dependence [Ni2019]_ (`arXiv:1909.06382 <https://arxiv.org/abs/1909.06382>`_);
the Euclid morphology programme
(`arXiv:1110.3193 <https://arxiv.org/abs/1110.3193>`_).

----

Specific star formation rate: star-forming vs quiescent
--------------------------------------------------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

Binary red/blue machinery exists —
:class:`~hod_mod.connection.hod.zumandelbaum15.ZuMandelbaum16QuenchingModel`
(Weibull red fractions for centrals and satellites [ZuMandelbaum2016]_) and
the Guo et al. (2019) :func:`~hod_mod.connection.hod.guo.f_quenched`
[Guo2019]_ — but neither is connected to the differentiable forecast, and
there is **no continuous sSFR variable** anywhere: no main sequence, no
conditional :math:`p({\rm sSFR}\,|\,M_*)`.  Every tier-2 cell mixes
star-forming and quiescent galaxies, although their gas contents (the model's
central subject) are known to differ at fixed mass.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **Cell split**: extend the tier-2 (z, M*) grid
  (:class:`~hod_mod.forecast.tier2.Tier2Forecast`, 80 cells) to
  (z, M*, SF/Q).  Per-cell occupations become :math:`N_c f_Q` and
  :math:`N_c (1 - f_Q)` with the ZM16 Weibull

  .. math::

     f_{\rm Q,cen}(M_h) = 1 - \exp\!\left[-\left(M_h/M_{\rm q}^{\rm cen}
     \right)^{\mu_{\rm q}}\right]

  re-implemented in a few lines of ``jnp`` inside ``forward_jax``.  The exact
  invariant — SF + Q cells sum to the unsplit prediction — is the regression
  test, in the same spirit as the tier-2 band-additivity test.
* **Continuous sSFR**: the double-lognormal conditional

  .. math::

     p(\log {\rm sSFR}\,|\,M_*, z) = f_Q\,\mathcal{N}(\mu_Q, \sigma_Q)
     + (1 - f_Q)\,\mathcal{N}\big(\mu_{\rm MS}(M_*, z), \sigma_{\rm MS}\big)

  with the star-forming main sequence :math:`\mu_{\rm MS}` parameterised à la
  Speagle et al. (2014), its redshift evolution using the existing tier-2
  ``_zs`` slope mechanism.
* **The physically interesting coupling**: let the hot-gas sector differ
  between SF and Q at fixed halo mass — a single offset ``dlx_quenched`` on
  the L_X–M relation (and optionally on the gas-concentration floor).  This
  is precisely the eROSITA CGM star-forming-vs-quiescent measurement of Zhang
  et al. (2025), and it plugs into the existing :math:`C_\ell^{gX}` machinery
  with split cells.
* **New parameters**: ``log10_Mq_cen``, ``mu_q_cen``, ``log10_Mq_sat``,
  ``ssfr_ms_norm``, ``ssfr_ms_slope``, ``ssfr_ms_zs``, ``dlx_quenched`` (0 —
  fiducial preserves the unsplit gas sector).  All analytic, native JAX.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* Quenched fractions :math:`f_Q(M_*, z)`: COSMOS/UltraVISTA [Ilbert2013]_,
  [Muzzin2013]_ and the Euclid deep surveys.
* SF/Q-split :math:`w_p` and :math:`\Delta\Sigma` (DESI BGS × Rubin) — the
  halo-mass difference of the two populations.
* SF/Q-split X-ray and tSZ stacks: eROSITA [Zhang2024CGM]_, [Zhang2025CGM]_
  and SO/ACT — the gas-sector offset.
* Group-catalogue quenched fractions vs :math:`M_h` [Yang2007groups]_,
  [Tinker2021groups]_.

Key references
~~~~~~~~~~~~~~

The eROSITA CGM SF/Q measurements [Zhang2025CGM]_
(`arXiv:2411.19945 <https://arxiv.org/abs/2411.19945>`_) and scaling relations
[Zhang2024CGM]_ (`arXiv:2401.17309 <https://arxiv.org/abs/2401.17309>`_), with
the TNG prediction [Truong2021]_
(`arXiv:2109.06884 <https://arxiv.org/abs/2109.06884>`_).  Quenching
phenomenology: [Peng2010]_
(`arXiv:1003.4747 <https://arxiv.org/abs/1003.4747>`_); SF/Q mass functions
[Ilbert2013]_ (`arXiv:1301.3157 <https://arxiv.org/abs/1301.3157>`_),
[Muzzin2013]_ (`arXiv:1303.4409 <https://arxiv.org/abs/1303.4409>`_); the main
sequence [Speagle2014]_
(`arXiv:1405.2041 <https://arxiv.org/abs/1405.2041>`_); empirical SFR–halo
models [Behroozi2019]_
(`arXiv:1806.07893 <https://arxiv.org/abs/1806.07893>`_); group catalogues
[Yang2007groups]_ (`arXiv:0707.4640 <https://arxiv.org/abs/0707.4640>`_),
[Tinker2021groups]_ (`arXiv:2010.02946 <https://arxiv.org/abs/2010.02946>`_);
the in-package quenching models [ZuMandelbaum2016]_, [Guo2019]_.

----

Galaxy spectral energy distributions: multi-band luminosity functions
---------------------------------------------------------------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

The package has a single-band conditional luminosity function
(:mod:`hod_mod.connection.clf`, Cacciato et al. [Cacciato2009]_,
[Cacciato2013]_) and the forecast predicts a stellar mass function — but there
is no galaxy SED, no filter/passband machinery, no k-correction (the only
k-correction table in the package is the AGN X-ray one), and therefore no
multi-band luminosity function.  The model cannot yet predict what Rubin,
Euclid or SPHEREx actually observe: magnitudes through filters.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **Route 1 (empirical, first)** — per-band CLF in
  ``hod_mod/connection/multiband.py``: extend :mod:`~hod_mod.connection.clf`
  to a list of bands with band-dependent central relations
  :math:`L_{c,b}(M_h)` and Schechter satellites, driven by mass-to-light
  ratios

  .. math::

     M_*/L_b = f\big(M_*,\ {\rm SF/Q}\big)

  parameterised per band with two stellar-population templates (red/blue),
  consuming the SF/Q label of the previous section — colours then emerge from
  the model instead of being assumed.
* **Route 2 (template distillation)** — synthesise a 2–3-component SED basis
  (old + young + dust screen) through the real filter curves once in numpy,
  cache :math:`L_b(M_*, {\rm sSFR}, z)` npz grids, and evaluate with JAX
  interpolation — the :mod:`~hod_mod.forecast.apec_bands` distill-to-table
  pattern verbatim; k-corrections come out by construction.
* **New observables**: per-band luminosity functions :math:`\Phi(M_b, z)` via
  the same cumulative-difference machinery as the forecast SMF; colour-split
  clustering; emission-line LFs ([OII]) flagged as a later extension needing
  an SFR→line calibration [Comparat2015OII]_.
* **New parameters** (two bands initially): ``ml_r_norm``, ``ml_r_slope``,
  ``ml_blue_offset``, ``tau_dust``.  New observables only — existing
  predictions untouched.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* COSMOS-Web stellar-mass/luminosity assembly [Shuntov2025]_ and the Euclid
  Cosmic Dawn SMF [EuclidCosmicDawn2025]_.
* Local band LFs: SDSS [Blanton2003]_, GAMA; B-band evolution to z≈1
  [Faber2007]_; Rubin ugrizy LFs to come.
* The DESI PAC stellar mass function into the :math:`10^6 M_\odot` frontier
  [Xu2025PAC]_ — the low-mass anchor.
* [OII] luminosity functions [Comparat2015OII]_ for the emission-line
  extension.

Key references
~~~~~~~~~~~~~~

CLF formalism [Cacciato2009]_, [Cacciato2013]_; M*/L calibrations
[Kauffmann2003]_
(`arXiv:astro-ph/0204055 <https://arxiv.org/abs/astro-ph/0204055>`_); band LFs
[Blanton2003]_, [Faber2007]_
(`arXiv:astro-ph/0506044 <https://arxiv.org/abs/astro-ph/0506044>`_); modern
mass functions [Shuntov2025]_
(`arXiv:2410.08290 <https://arxiv.org/abs/2410.08290>`_), [Xu2025PAC]_
(`arXiv:2503.01948 <https://arxiv.org/abs/2503.01948>`_),
[EuclidCosmicDawn2025]_
(`arXiv:2504.17867 <https://arxiv.org/abs/2504.17867>`_); emission-line LFs
[Comparat2015OII]_ (`arXiv:1408.1523 <https://arxiv.org/abs/1408.1523>`_);
empirical multi-wavelength mocks [Comparat2019]_.

----

Stellar feedback
----------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

The energy-closure mode of the forecast contains a single supernova channel,
:math:`E_{\rm SN} = \varepsilon_{\rm SN} \cdot M_*(M_h) \cdot e_{\rm SN}` with
the coupling frozen at :math:`\varepsilon_{\rm SN} = 0.1` (``_EPS_SN`` in
:mod:`hod_mod.forecast.forward_jax`).  There is no wind model, no mass
loading, no kinetic/thermal split; the low-mass shape of the gas sector is
carried by the phenomenological :math:`\eta(M)` gas-concentration sigmoid and
the baryon-fraction pivot.  Hydro-calibrated matter-power boosts (HMcode
[Mead2020]_) exist in :mod:`hod_mod.core.nonlinear` but are disconnected from
the halo-level feedback narrative.

.. seealso::
   Promoting fixed constants into the parameter vector is the documented
   "cheap extension" pattern of :doc:`sensitivity_fisher`; ``eps_sn`` is
   simply its next instance.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **Step 1 (near-free)**: promote ``_EPS_SN`` → ``eps_sn`` in the parameter
  vector (fiducial 0.1 — bit-identical predictions), giving the energy
  closure a free SN coupling alongside the AGN one.
* **Step 2 — wind mass loading**:

  .. math::

     \eta_w(M_h) = \eta_0 \left(\frac{V_c(M_h)}{V_0}\right)^{-\alpha_w},
     \qquad \alpha_w = 1\ (\text{momentum-driven}) \ldots 2\ (\text{energy-driven})

  with the FIRE calibration [Muratov2015]_ as the fiducial anchor, coupled
  into the existing :math:`\eta(M)` gas-concentration slot of ``forward_jax``
  — same code slot, physical meaning: SN-driven winds set how puffed-out the
  low-mass hot gas is.  Fiducials for :math:`(\eta_0, \alpha_w)` are fitted
  once to reproduce the current sigmoid so fiducial predictions move by
  <0.1%.
* **Step 3**: tie :math:`E_{\rm SN}` to a group-scale entropy/temperature
  floor extending the energy closure, so the SN sector talks to the
  :math:`L_X`–:math:`M` and :math:`kT`–:math:`M` relations at the low-mass
  end where SN and AGN prescriptions diverge most between simulations
  [Eckert2021]_.
* **Validation** (not a runtime dependency): compare the freed feedback
  sector against the CAMELS feedback-variation grid [CAMELS2023]_ and
  FLAMINGO [FLAMINGO_overview]_.
* **New parameters**: ``eps_sn`` (0.1), ``eta_w_norm``, ``alpha_w``.  Native
  JAX.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* The low-mass slope of the stellar mass function (COSMOS-Web, Euclid deep,
  PAC [Xu2025PAC]_) — the integral constraint on wind mass loading.
* eROSITA group-scale :math:`L_X`–:math:`M` [Zhang2024CGM]_ and the group
  baryon census [Eckert2021]_.
* kSZ gas-ejection profiles (ACT/SO × DESI) — the momentum budget.
* Down-the-barrel outflow rates as external priors [Chisholm2017]_.

Key references
~~~~~~~~~~~~~~

Review [SomervilleDave2015]_
(`arXiv:1412.2712 <https://arxiv.org/abs/1412.2712>`_); wind calibrations
[Muratov2015]_ (`arXiv:1501.03155 <https://arxiv.org/abs/1501.03155>`_) and
measurements [Chisholm2017]_
(`arXiv:1702.07351 <https://arxiv.org/abs/1702.07351>`_); subgrid
implementations [PillepichTNG2018]_
(`arXiv:1703.02970 <https://arxiv.org/abs/1703.02970>`_); variation suites
[CAMELS2023]_ (`arXiv:2201.01300 <https://arxiv.org/abs/2201.01300>`_),
[FLAMINGO_overview]_; the group-scale divergence [Eckert2021]_
(`arXiv:2106.13259 <https://arxiv.org/abs/2106.13259>`_).

----

Cold gas and neutral hydrogen
-----------------------------

Where the model stands
~~~~~~~~~~~~~~~~~~~~~~

Nothing.  The gas sector (:mod:`hod_mod.gas`) is entirely hot-phase — X-ray
emissivity, tSZ pressure, ICM metallicity.  Neutral hydrogen appears in the
package in exactly one role: the Galactic absorption column of the tier-2
obscuration template — which is foreground, not astrophysics of the modeled
halos.  There is no :math:`M_{\rm HI}(M_h)`, no HI profile, no 21 cm
observable, no molecular phase.

Proposed implementation
~~~~~~~~~~~~~~~~~~~~~~~

* **HI halo model** — new module ``hod_mod/gas/hi.py`` with the
  Villaescusa-Navarro et al. (2018) form:

  .. math::

     M_{\rm HI}(M_h, z) = M_0 \left(\frac{M_h}{M_{\rm min}}\right)^{\alpha}
     \exp\!\left[-\left(\frac{M_{\rm min}}{M_h}\right)^{0.35}\right]

  plus an exponential/altered-NFW HI profile :math:`u_{\rm HI}(k|M)` for the
  small scales.
* **Observables reusing existing machinery**: :math:`\Omega_{\rm HI}(z)` and
  the HI bias :math:`b_{\rm HI}` are integrals over the same HMF grid; the HI
  mass function follows from the conditional-scatter pattern of the forecast
  SMF; and **21 cm intensity-mapping cross-correlations**
  :math:`C_\ell^{{\rm HI}\times g}` are the existing
  :math:`C_\ell^{gX}` Limber machinery with an HI brightness-temperature
  kernel — the :math:`\bar T_b(z)` prefactor is analytic.
* **Coupling to the SF/Q split**: quenched centrals are HI-poor — one offset
  parameter conditioned on the Section-5 label, following the
  NeutralUniverseMachine phenomenology [GuoNUM2023]_.
* **New parameters**: ``log10_M0_hi``, ``log10_Mmin_hi``, ``alpha_hi``
  (fiducials from [VillaescusaNavarro2018HI]_; z-evolution through the
  existing tier-2 ``_zs`` slope mechanism; optionally ``dhi_quenched``).
  Fully analytic, native JAX; new observables only.
* **Honest scoping note**: HI4PI [HI4PI2016]_ is *Galactic* HI — it stays in
  its foreground-template role and is not evidence about the halo HI content.

Constraining measurements
~~~~~~~~~~~~~~~~~~~~~~~~~

* The z≈0 HI mass function: ALFALFA [Jones2018ALFALFA]_ and the first
  interferometric HIMF from MeerKAT MIGHTEE-HI [Ponomareva2023]_.
* The empirical HI HOD from ALFALFA×SDSS [Obuljen2019]_.
* Conditional gas fractions :math:`M_{\rm HI}(M_*, {\rm sSFR})` from xGASS
  [Catinella2018xGASS]_ — the data for the SF/Q coupling.
* 21 cm intensity-mapping cross-correlations: the CHIME×eBOSS detection
  [CHIME2023]_ today, CHIME/MeerKLASS×DESI next, SKA1 forecasts [SKA2019]_.
* :math:`\Omega_{\rm HI}(z)` from DLA surveys as the high-z anchor.

Key references
~~~~~~~~~~~~~~

The HI halo model [VillaescusaNavarro2018HI]_
(`arXiv:1804.09180 <https://arxiv.org/abs/1804.09180>`_) and HI HOD
[Obuljen2019]_ (`arXiv:1805.00934 <https://arxiv.org/abs/1805.00934>`_);
abundances [Jones2018ALFALFA]_
(`arXiv:1802.00053 <https://arxiv.org/abs/1802.00053>`_), [Ponomareva2023]_
(`arXiv:2304.13051 <https://arxiv.org/abs/2304.13051>`_); scaling relations
[Catinella2018xGASS]_
(`arXiv:1802.02373 <https://arxiv.org/abs/1802.02373>`_); empirical evolution
models [GuoNUM2023]_ (`arXiv:2307.07078 <https://arxiv.org/abs/2307.07078>`_),
[Nishigaki2025]_ (`arXiv:2503.10999 <https://arxiv.org/abs/2503.10999>`_);
21 cm detections and forecasts [CHIME2023]_
(`arXiv:2202.01242 <https://arxiv.org/abs/2202.01242>`_), [SKA2019]_
(`arXiv:1912.12699 <https://arxiv.org/abs/1912.12699>`_); the Galactic
foreground map [HI4PI2016]_
(`arXiv:1610.06175 <https://arxiv.org/abs/1610.06175>`_).
