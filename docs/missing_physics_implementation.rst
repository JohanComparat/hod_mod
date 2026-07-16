The extended model: implementation of the missing physics
==========================================================

*The implemented counterpart of* :doc:`missing_physics`.

Between 2026-07-03 and 2026-07-04 (branch ``feature/missing-physics``) the
propositions of :doc:`missing_physics` were implemented in four waves,
growing the differentiable parameter vector from the tier-2 61 entries to
**90** (waves 1–3), then to **102** (the :doc:`tier3_forecast` SED
calibrations), **106** (wave 4, galaxy morphology) and **111**
(:doc:`tier4_forecast`, the morphology observables) — every addition
fiducial-preserving, so the :doc:`tier2_forecast`
predictions are reproduced exactly (or to a stated tolerance) at the fiducial,
and every mechanism carries an exact invariant that is enforced by
``tests/test_missing_physics.py``.  This page documents the architecture, the
physics of each sector with its equations and parameters, the survey noise
models, the invariants, usage, and the approximations deliberately made.

.. contents::
   :local:
   :depth: 2

Architecture
------------

Three design rules govern every extension:

1. **Append-only, fiducial-preserving parameters.**  New entries land at the
   end of :data:`~hod_mod.forecast.forward_jax.PARAM_NAMES`
   (``MISSING_PHYSICS = PARAM_NAMES[61:]``, 29 names) with fiducials equal to
   the previously hard-coded behaviour: promoted constants keep their values,
   switches sit at their "off" value (``eta_w_norm = 0``, ``sum_mnu = 0`` eV,
   ``dlx_quenched = 0``), and physical relations take their literature
   centres.  Tier-1/tier-2 scripts pin the whole extension by default
   (``fix=params.TIER2_EXTENSION``).
2. **Static dispatch, no traced branches.**  Where behaviour must change with
   the parameter *set* rather than parameter *values*, the branch is on
   dictionary membership or a constructor flag — static under JAX tracing.
   The growth factor switches to the CPL ODE because the forecast cosmology
   dict *has* a ``w0`` key; production ΛCDM dicts keep the Carroll formula
   bit-identical.  Sample definitions (``sfq``, ``ssfr_cut``,
   ``cm_relation``, ``pk_correction``) are constructor arguments.
3. **Distill-to-table for non-traceable dependencies.**  Anything that cannot
   be traced (CAMB, soxs/APEC) is evaluated once in numpy, stored as an npz
   shipped with the package, and consumed through differentiable ``jnp``
   interpolation — the pattern established by
   :mod:`~hod_mod.forecast.apec_bands` and extended by
   :mod:`~hod_mod.forecast.pk_camb_ratio`.

The parameter block
~~~~~~~~~~~~~~~~~~~

.. list-table:: The 29 ``MISSING_PHYSICS`` parameters (fiducial → prior width)
   :header-rows: 1
   :widths: 16 12 10 14 48

   * - Name
     - Fiducial
     - Prior σ
     - Sector
     - Enters through / constrained by
   * - ``eps_sn``
     - 0.1
     - 0.3
     - baryon
     - SN channel of the energy closure; low-mass SMF, group L_X–M
   * - ``w0``, ``wa``
     - −1, 0
     - 1.0, 2.0
     - cosmology
     - CPL growth ODE + all Limber distances + E(z); BAO/FS, 3×2pt
   * - ``sum_mnu`` [eV]
     - 0
     - 0.25
     - cosmology
     - ν suppression of the P(k) shape (σ8-anchored); cluster counts + CMB lensing
   * - ``log10_Mq_cen``, ``mu_q_cen``
     - 11.78, 0.41
     - 1.0, 1.0
     - sfq
     - ZM16 central quenching Weibull; f_Q(M*, z), split w_p/ΔΣ
   * - ``log10_Mq_sat``, ``mu_q_sat``
     - 12.19, 0.24
     - 1.0, 1.0
     - sfq
     - ZM16 satellite quenching; group catalogues
   * - ``dlx_quenched`` [dex]
     - 0
     - 1.0
     - sfq
     - L_X–M offset of quenched samples; eROSITA SF/Q CGM stacks
   * - ``agn_xi_rx``, ``agn_xi_rm``
     - 0.60, 0.78
     - 0.3, 0.3
     - agn
     - fundamental-plane slopes; radio LF + FP priors
   * - ``agn_b_r``, ``agn_sig_r``
     - 7.33, 0.88
     - 2.0, 0.3
     - agn
     - FP zero point and scatter
   * - ``log10_M0_hi``, ``log10_Mmin_hi``, ``alpha_hi``
     - 10.63, 12.3, 0.24
     - 1, 1, 0.3
     - coldgas
     - M_HI(M_h); HIMF, HI HOD, 21 cm crosses
   * - ``eta_w_norm``, ``alpha_w``
     - 0, 1.0
     - 0.5, 0.7
     - baryon
     - wind mass loading in η(M); kSZ, group baryon census
   * - ``ssfr_ms_norm``, ``ssfr_ms_slope``, ``ssfr_ms_zs``
     - −9.9, −0.25, 0
     - 0.5, 0.3, 2.0
     - sfq
     - the star-forming main sequence + evolution; MS measurements
   * - ``dhi_quenched`` [dex]
     - 0
     - 1.0
     - coldgas
     - HI deficit of quenched centrals; xGASS conditional gas fractions
   * - ``sigma_ms``, ``dssfr_q`` [dex]
     - 0.30, −1.6
     - 0.15, 0.5
     - sfq
     - the double-lognormal p(log sSFR | M*); sSFR distributions
   * - ``loii_norm``
     - 40.85
     - 0.3
     - sfq
     - log L_[OII]/SFR calibration [Kennicutt1998]_; [OII] LFs
   * - ``f_loud0``, ``beta_loud``, ``b_jet``
     - 0.01, 2.0, 41.0
     - 0.05, 1.5, 2.0
     - agn
     - radio-loud jet population; bright-end radio LF
   * - ``agn_bc_ir``
     - −0.52
     - 0.3
     - agn
     - L_IR(6 μm)/L_bol; IR AGN LF and counts

Wave 1 — cosmology, concentration, quenching, fundamental plane, HI
--------------------------------------------------------------------

Beyond-ΛCDM growth and neutrinos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear growth factor is obtained from the growth ODE in
:math:`x = \ln a`,

.. math::

   D'' + \Big[2 + \frac{{\rm d}\ln E}{{\rm d}\ln a}\Big] D'
   - \tfrac{3}{2}\,\Omega_m(a)\,D = 0,
   \qquad
   E^2(a) = \Omega_m a^{-3} + (1-\Omega_m)\,
   a^{-3(1+w_0+w_a)}\,e^{-3 w_a (1-a)},

integrated by a fixed-grid RK4 through ``jax.lax.scan`` (160 steps from
:math:`\ln a = -7`; jit/vmap/jacfwd-safe, no extra dependency) with the
matter-domination growing mode :math:`D \propto a` as the initial condition
(:func:`~hod_mod.core.halo_mass_function._growth_factor_cpl_jax`).  The
dispatcher :func:`~hod_mod.core.halo_mass_function.growth_factor` selects it
whenever the cosmology dict carries a ``w0``/``wa`` key; at
:math:`(w_0, w_a) = (-1, 0)` it reproduces the Carroll et al. (1992) fitting
formula to a few :math:`\times 10^{-4}` and the exact ΛCDM quadrature
:math:`D \propto E \int {\rm d}a\,(aE)^{-3}` to <0.2% (both tested).  The CPL
parameters also propagate through **geometry**: every Limber comoving
distance and every :math:`E(z)` factor in the forecast (halo definitions,
L_X/kT/P500 self-similar scalings, the energy closure) uses the CPL
expressions of :mod:`hod_mod.core.distances` /
``ForwardModel._e2z``.

Massive neutrinos suppress the *shape* of the EH98 spectrum
(:func:`~hod_mod.forecast.pk_eisenstein_hu._nu_suppression`):

.. math::

   \frac{P_\nu(k)}{P_0(k)} = 1 - 8 f_\nu \cdot
   \tfrac{1}{2}\Big[1 + \tanh\frac{\ln (k/k_{\rm nr})}{1.5}\Big],
   \quad
   f_\nu = \frac{\Sigma m_\nu / 93.14\,{\rm eV}}{\Omega_m h^2},

with :math:`k_{\rm nr} \simeq 0.018\sqrt{\Omega_m m_{\nu,i}/1\,{\rm eV}}`
h/Mpc (three degenerate species) and a small mass floor inside the square
root keeping the ``jacfwd`` column finite at the massless fiducial.  Because
the suppression multiplies the shape used *everywhere* (the HMF path, the σ8
anchor and the 2-halo term), σ8 keeps its measured-amplitude meaning and only
the shape responds — exactly massless (bit-identical) at
:math:`\Sigma m_\nu = 0`.  The wave-2 CAMB-ratio derivative row (below)
corrects the residual of this first-order form.

Cosmology-dependent concentration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ForwardModel(cm_relation="diemer15")`` replaces the Planck13-frozen
Dutton14 fit with the Diemer & Kravtsov / Diemer & Joyce model
[DiemerKravtsov2015]_, [DiemerJoyce2019]_,

.. math::

   c_{\rm 200c} = \frac{c_{\min}}{2}
   \left[\left(\frac{\nu}{\nu_{\min}}\right)^{-\alpha}
       + \left(\frac{\nu}{\nu_{\min}}\right)^{\beta}\right],
   \qquad
   \begin{aligned}
   c_{\min} &= \phi_0 + \phi_1\, n_{\rm eff},\\
   \nu_{\min} &= \eta_0 + \eta_1\, n_{\rm eff},
   \end{aligned}

with the DJ19 median parameters (φ₀, φ₁, η₀, η₁, α, β) =
(6.58, 1.27, 7.28, 1.56, 1.08, 1.77), peak height
:math:`\nu = 1.686/\sigma(M, z)` from the JAX σ(M) machinery, and
:math:`n_{\rm eff} = {\rm d}\ln P/{\rm d}\ln k` evaluated at
:math:`k_R = \kappa\, 2\pi/R(M)` with the DJ19 calibration **κ = 0.42**
(``ForwardModel._neff_eh98``, differentiable).  Both inputs carry cosmology,
so c(M) finally responds to σ8/n_s/h — and, through the growth factor, to
w0/wa/Σm_ν.

.. important::

   Implementing this exposed a factor-~2 bug in the pre-existing
   :func:`hod_mod.core.concentration.c_diemer15`: the second power law was
   applied as :math:`c_{\min} x^{-\alpha}(1+x^{\beta})` (an effective
   :math:`\beta-\alpha` exponent, no ½) instead of the DK15 Eq. 9 form above,
   and the n_eff helper used κ = 1.  Both are fixed (core and forecast);
   values now match the COLOSSUS [Colossus2018]_ anchors —
   :math:`c_{\rm 200c}(10^{12}, z{=}0) = 8.8` (~9.5),
   :math:`c_{\rm 200c}(10^{15}) = 4.7` (~5.0) — enforced by
   ``test_dk15_concentration_is_cosmology_dependent``.

The SF/quiescent split
~~~~~~~~~~~~~~~~~~~~~~

``ForwardModel(sfq="sf"|"q")`` weights the occupations with the Zu &
Mandelbaum halo-quenching Weibull fractions [ZuMandelbaum2016]_
(:func:`~hod_mod.connection.hod.zumandelbaum15.f_red_cen_zu16` /
``f_red_sat_zu16``, the in-repo MAP fiducials):

.. math::

   f_{\rm Q}(M_h) = 1 - \exp\!\left[-\big(M_h/M_{\rm q}\big)^{\mu_{\rm q}}\right],

separately for centrals and satellites.  By construction the star-forming and
quenched samples **sum exactly to the unsplit sample** — the regression
invariant, tested to :math:`10^{-12}` on occupations, abundances and the SFR
density.  ``dlx_quenched`` shifts the L_X–M relation of quenched samples only
(``_lx_kt_of``), turning the eROSITA CGM star-forming-vs-quiescent
measurements [Zhang2025CGM]_ into a one-parameter test through the split-cell
:math:`C_\ell^{gX}`.  In the tier-2 assembly, ``Tier2Forecast(split_sfq=True)``
doubles the cell blocks while the shell X-ray observables (XLF, band
:math:`C_\ell^{XX}`) keep a dedicated **unsplit** model, so the quenched
offset cannot leak into the total-gas X-ray auto.

The fundamental plane of black-hole activity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The radio luminosity attaches to the Powell chain's per-halo
:math:`(M_{\rm BH}, L_X)` through [MerloniHeinzDiMatteo2003]_ (priors from
[Gultekin2019]_):

.. math::

   \log L_R = \xi_{RX} \log L_X + \xi_{RM} \log M_{\rm BH} + b_R .

Because the :math:`M_{\rm BH}` scatter :math:`\sigma_{lm}` is the *same*
deviate in :math:`L_X` and :math:`M_{\rm BH}`, it enters the radio kernel
with coefficient :math:`(\xi_{RX}+\xi_{RM})`, and the FP scatter adds in
quadrature:

.. math::

   \sigma_R^{\rm eff} = \sqrt{(\xi_{RX}+\xi_{RM})^2 \sigma_{lm}^2 + \sigma_R^2}.

The ``rlf`` observable (5 GHz νL_ν luminosity function) reuses the XLF
kernel machinery; the exact identity — at
:math:`(\xi_{RX}, \xi_{RM}, b_R, \sigma_R) = (1, 0, 0, 0)` and jets off, the
rlf **equals the hard-band XLF** on the same abscissas — is tested to
:math:`10^{-10}`.

The HI sector
~~~~~~~~~~~~~

The Villaescusa-Navarro et al. halo model [VillaescusaNavarro2018HI]_:

.. math::

   M_{\rm HI}(M_h) = M_0 \left(\frac{M_h}{M_{\min}}\right)^{\alpha}
   \exp\!\left[-\left(\frac{M_{\min}}{M_h}\right)^{0.35}\right],

feeding (i) the ``himf`` HI mass function through a 0.35 dex lognormal
conditional (the SMF/XLF kernel pattern), and (ii) the 21 cm × galaxy cross
``cl_gHI`` — the :math:`C_\ell^{gX}` machinery with the NFW-distributed HI
mass as the tracer field, so :math:`C_\ell^{g{\rm HI}} \propto M_0`
**exactly** (tested).  ``dhi_quenched`` (wave 2) dims the HI around quenched
centrals, following the NeutralUniverseMachine phenomenology [GuoNUM2023]_.

Supernova coupling
~~~~~~~~~~~~~~~~~~

``eps_sn`` promotes the formerly fixed :math:`\varepsilon_{\rm SN} = 0.1` of
the energy-closure SN channel into the vector.  Note for testing: at the
default fiducial the ``log10_M_pivot`` slot (reinterpreted as
:math:`\log_{10}\varepsilon_{\rm AGN}` in closure mode) saturates the energy
budget's ``min()``, so exercising the SN derivative requires a physical
coupling (the gate uses :math:`\log_{10}\varepsilon_{\rm AGN} = -2`).

Wave 2 — CAMB-quality P(k), winds, main sequence, radio/HI surveys
-------------------------------------------------------------------

The linearized CAMB ratio
~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   **Status (0.3.1):** the "CAMB-quality P(k)" goal is now met by the
   CosmoPower-JAX emulator (:mod:`hod_mod.forecast.pk_cosmopower`), the
   package-wide default via
   :func:`~hod_mod.core.power_spectrum.default_pk_linear` and
   ``ForwardModel(pk_correction="cosmopower")``.  The ratio table described
   below remains available as the lighter ``pk_correction="camb_linear"``
   Fisher correction.

A Fisher forecast needs the spectrum and its **first derivatives** to be
accurate at the fiducial — nothing more.  That reduces "CAMB-quality P(k)"
from an emulator problem to eleven CAMB evaluations
(:mod:`hod_mod.forecast.pk_camb_ratio`):

.. math::

   \ln R(k;\theta) \simeq \ln R_0(k) + \sum_i
   \frac{\partial \ln R}{\partial \theta_i}(k)\,(\theta_i - \theta_i^{\rm fid}),
   \qquad
   R = \frac{P^{\rm shape}_{\rm CAMB}}{P^{\rm shape}_{\rm EH98}},

with both shapes pivot-normalised at k = 0.05 h/Mpc and derivative rows for
(h, Ω_b, Ω_m, n_s, Σm_ν).  The fiducial ratio confirms the expected EH98
error: :math:`\max|\ln R_0| = 3.8\%` around the BAO scale.  The Σm_ν row is
computed against the EH98 shape *including* the wave-1 tanh suppression, so
it corrects that form's residual to CAMB accuracy.  w0/wa need no rows (they
do not alter the z = 0 transfer shape).  The table ships as
``hod_mod/data/pk_ratio/camb_eh98_ratio.npz``; evaluation is two
``jnp.interp`` calls and an ``exp``, applied inside ``pk_shape`` so the HMF,
σ8 anchor and 2-halo term stay mutually consistent.  Opt-in:
``ForwardModel(pk_correction="camb_linear")``; rebuild with
``python -m hod_mod.forecast.pk_camb_ratio`` (needs CAMB).  Beyond the
tabulated k range the log-ratio is edge-clamped (deep power-law tail).

Wind mass loading
~~~~~~~~~~~~~~~~~

The Muratov-style [Muratov2015]_ SN wind,

.. math::

   \eta_w(M_h) = \eta_{w,0} \left(\frac{V_c(M_h)}{200\,{\rm km/s}}\right)^{-\alpha_w},
   \qquad
   \eta_{\rm eff}(M) = \frac{\eta_{\rm sigmoid}(M)}{1 + \eta_w(M)},

couples into the existing gas-concentration slot: stronger winds puff out the
low-mass hot gas, which the ΔΣ baryon split, the X-ray emissivity and the tSZ
pressure all see.  :math:`\alpha_w` interpolates the momentum-driven (1) to
energy-driven (2) scalings [SomervilleDave2015]_.  At the fiducial
:math:`\eta_{w,0} = 0` the tier-2 sigmoid is reproduced **bit-identically**
(tested), and the :math:`\eta_{w,0}` derivative is live; :math:`\alpha_w` is
flat exactly at the fiducial (the prior bounds it) — a documented property,
not a bug.

Surveys: radio, HI, and the local-HIMF lesson
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~hod_mod.forecast.noise.RadioSurvey` (LoTSS-like:
:math:`f_{\rm sky} = 0.13`, a νL_ν(5 GHz)-equivalent detection threshold of
:math:`3\times10^{-22}` erg/s/cm² driving the L_lim(z) completeness of the
radio LF) and :class:`~hod_mod.forecast.noise.HISurvey` (ALFALFA-like:
:math:`f_{\rm sky} = 0.16`, the standard 21 cm mass–flux relation
:math:`M_{\rm lim} = 2.36\times10^5 d_L^2 S_{\rm int}`, plus an effective
noise-to-signal recipe for the 21 cm intensity-mapping cross) enter
``Tier2Forecast(include_radio=True, include_hi=True)``.

One design lesson worth recording: attaching the HIMF to the Δz = 0.1
shells fails *physically* — at :math:`z_{\rm hi} = 0.3` the flux limit gives
:math:`M_{\rm lim} \approx 2.5\times10^{11} M_\odot`, flagging every bin.
Blind HIMFs are local measurements, so the assembly builds one dedicated
``hi_local`` block (z ≤ 0.06), matching how ALFALFA [Jones2018ALFALFA]_ and
MIGHTEE-HI [Ponomareva2023]_ actually operate.

Wave 3 — continuous sSFR, [OII], jets, infrared
------------------------------------------------

The continuous sSFR distribution and selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conditional sSFR distribution is the double lognormal

.. math::

   p(\log {\rm sSFR}\,|\,M_*) =
   f_{\rm Q}\,\mathcal{N}\big(\mu_{\rm MS} + \Delta_{\rm Q},\, \sigma_{\rm Q}\big)
   + (1-f_{\rm Q})\,\mathcal{N}\big(\mu_{\rm MS},\, \sigma_{\rm MS}\big),

with the Speagle-like main sequence [Speagle2014]_
:math:`\mu_{\rm MS} = {\rm ssfr\_ms\_norm} + {\rm ssfr\_ms\_slope}\,
(\log M_* - 10.5)` (evolution through the standard ``_zs`` mechanism), free
MS scatter ``sigma_ms`` and quenched offset ``dssfr_q``
(:math:`\sigma_{\rm Q} = 0.5` dex fixed).  ``ForwardModel(ssfr_cut=...)``
selects sSFR-thresholded samples (ELG-like) by the per-component Gaussian
survival :math:`S = \tfrac12\,{\rm erfc}[({\rm cut}-\mu)/\sqrt{2}\sigma]`
applied inside the occupation (``_sfq_weights``).  The selection **composes**
with the SF/Q split: SF-cut + Q-cut ≡ mixture-cut exactly, and a cut at
:math:`-\infty` recovers the unsplit sample exactly (both tested).

Two observables consume it:

* ``sfrd`` — the cell's SFR density
  :math:`\rho_{\rm SFR} = \sum_{\rm pop} n_{\rm pop}\,\langle{\rm SFR}\rangle_{\rm pop}`
  with the lognormal means
  :math:`\langle{\rm SFR}\rangle = 10^{\mu + \log M_*} e^{(\sigma\ln 10)^2/2}`;
  :math:`\rho_{\rm SFR} \propto 10^{\rm ssfr\_ms\_norm}` **exactly** and SF+Q
  partition it exactly (both tested).  Data: the cosmic star-formation
  history [MadauDickinson2014]_.
* ``oiilf`` — the z-resolved [OII] luminosity function through the
  Kennicutt-like calibration [Kennicutt1998]_
  :math:`\log L_{\rm [OII]} = {\rm loii\_norm} + \log{\rm SFR}` on the ZM15
  SHMR + main sequence, with kernel width
  :math:`\sqrt{\sigma_{\rm MS}^2 + \sigma_{\rm [OII]}^2 +
  (1+{\rm slope})^2\sigma_{M_*}^2}` and the star-forming fraction as weight
  (centrals-only v1).  Data: the [OII] LFs of [Comparat2015OII]_.  A designed
  degeneracy is made explicit and testable: the ``loii_norm`` and
  ``ssfr_ms_norm`` Jacobian columns are *identical* (both shift the same
  abscissa) — only SFR-side data (``sfrd``, ``ssfr``) break it.

Radio-loud jets
~~~~~~~~~~~~~~~

The fundamental plane describes the radiatively efficient population.  The
jetted HERG/LERG population [BestHeckman2012]_ is a second ``rlf`` component
from **all** central black holes — deliberately *not* tied to the ERDF-active
fraction:

.. math::

   f_{\rm loud}(M_{\rm BH}) = \min\!\big[f_{\rm loud,0}\,
   10^{\beta_{\rm loud}(\log M_{\rm BH} - 8)},\, 1\big],
   \qquad
   \log L_R^{\rm jet} \sim \mathcal{N}\big(b_{\rm jet} + (\log M_{\rm BH} - 8),\,
   \sigma_{\rm jet}\big).

Consequently the :math:`\Phi_R \propto 10^{f_{\rm ERDF}}` amplitude identity
of the FP-only rlf **breaks by design** once jets are on — tested in both
directions (exact with :math:`f_{\rm loud,0} = 0`; a deficit with jets on).

The infrared AGN luminosity function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ilf`` maps the Powell chain's bolometric output to 6 μm,
:math:`L_{\rm IR} = 10^{\rm agn\_bc\_ir}\,L_{\rm bol}` ([Hopkins2007]_-style
correction), with **no** obscuration suppression: the IR LF is
obscuration-robust by construction (the ``agn_fabs`` Jacobian column is
exactly zero — tested), so together with the f_abs-dimmed soft-X-ray XLF it
is the cross-band consistency check of the obscured fraction, the
WISE-selected halo statistics [Donoso2014]_, [Petter2023]_ probe from the
clustering side.  Noise: :class:`~hod_mod.forecast.noise.IRSurvey`
(WISE/SPHEREx-like, νL_ν(6 μm) completeness limit).

Wave 4 — galaxy morphology
--------------------------

The last roadmap topic with no code at all.  Vector 102 → 106
(:data:`~hod_mod.forecast.forward_jax.WAVE4_MORPHOLOGY`, sector
``morphology``); the ``TIER3_EXTENSION`` slice is frozen at [90:102].

The conditional early-type fraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`hod_mod.connection.morphology` mirrors the ZM16 halo-quenching
pattern: a Weibull early-type fraction of centrals,

.. math::

    f_{\rm early,c}(M_h) = 1 - \exp\!\left[-\left(M_h/M_{\rm morph}
    \right)^{\beta_{\rm morph}}\right],

with fiducial :math:`\log_{10} M_{\rm morph} = 12.5`,
:math:`\beta_{\rm morph} = 0.8`, and a satellite boost toward early types
:math:`f_{\rm early,s} = f_{\rm early,c} + f_{\rm morph,sat}
(1 - f_{\rm early,c})` (environmental transformation; ∈ [0, 1] by
construction).  ``ForwardModel(morph="early"|"late")`` weights the
occupations exactly like the SF/Q split — EARLY + LATE ≡ unsplit, and the
four-way SF/Q × early/late partition sums exactly to the unsplit sample
(both tested to 1e-12; morphology and quenching selections are treated as
independent at fixed M_h, a documented simplification).

The f_early observable
~~~~~~~~~~~~~~~~~~~~~~

The roadmap's cheap route: one Euclid-VIS-like datum
:math:`f_{\rm early}(M_*, z)` per (z, M*) cell — the occupation-weighted
mean early-type fraction of the cell's sample
(``Tier2Forecast(include_morph=True)`` / ``--include-morph``; default ON in
:class:`~hod_mod.forecast.tier3.Tier3Forecast`).  Noise is binomial counting
over the cell (negligible for wide cells) plus a morphological-calibration
floor (``SpectroSurvey.fmorph_err`` = 0.02 absolute).  Data:
Euclid VIS morphologies, COSMOS-Web/HSC at higher z, group-catalogue
morphology–halo-mass trends [Yang2007groups]_, [Tinker2021groups]_.

The black-hole–bulge coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mbh_bt_slope`` inserts the bulge proxy into the Powell chain
(:math:`M_{\rm BH} \propto (B/T \cdot M_*)`-like coevolution
[Yang2019BHbulge]_), with :math:`B/T` proxied by the mean early-type
fraction of the halo:

.. math::

    \langle \log M_{\rm BH} \rangle \mathrel{+}= {\rm mbh\_bt\_slope}
    \cdot \log_{10}\!\big[f_{\rm early,c}(M_h) + 10^{-4}\big] .

At the ``mbh_bt_slope = 0`` fiducial the chain is **exactly** unchanged
(zero morphology response of the XLF — tested), while the
``mbh_bt_slope`` Jacobian column is live at the fiducial and, off-fiducial,
routes ``log10_M_morph``/``beta_morph`` into the XLF, radio and IR LFs —
morphology becomes testable through the AGN sector, as the roadmap
proposed.  (The assembly-bias hook — ordering B/T by formation time through
the cosmology-dependent c(M) — remains a documented refinement.)

Observables and noise: the full menu
------------------------------------

.. list-table:: Opt-in observables added by the three waves
   :header-rows: 1
   :widths: 10 22 16 30 22

   * - Name
     - Physical content
     - Block / flag
     - Noise
     - Completeness cut
   * - ``rlf``
     - 5 GHz AGN radio LF (FP + jets)
     - shell / ``--include-radio``
     - Poisson over the radio footprint
     - νL_ν ≥ L_lim(z_hi), LoTSS-like
   * - ``ilf``
     - 6 μm AGN IR LF
     - shell / ``--include-ir``
     - Poisson over the IR footprint
     - νL_ν ≥ L_lim(z_hi), WISE/SPHEREx-like
   * - ``oiilf``
     - [OII] emission-line LF
     - shell / ``--include-ssfr``
     - Poisson over the spectroscopic volume
     - L ≥ 4π d_L² F_[OII],lim (DESI-like)
   * - ``himf``
     - HI mass function
     - ``hi_local`` (z ≤ 0.06) / ``--include-hi``
     - Poisson, ALFALFA footprint
     - M_HI ≥ 2.36×10⁵ d_L² S_int
   * - ``cl_gHI``
     - 21 cm × galaxy cross
     - cell / ``--include-hi``
     - Knox with the effective IM recipe
     - —
   * - ``ssfr``
     - mean MS log sSFR of the cell
     - non-Q cell / ``--include-ssfr``
     - 0.05 dex absolute
     - —
   * - ``f_early``
     - early-type fraction of the cell
     - cell / ``--include-morph``
     - binomial + 0.02 calibration floor
     - —
   * - ``sfrd``
     - SFR density of the cell
     - non-Q cell / ``--include-ssfr``
     - 12% relative (MD14-like)
     - —

All abundance-type observables (``rlf``, ``ilf``, ``oiilf``, ``himf``,
``ssfr``, ``sfrd``) bypass the r_p/ℓ scale cuts; ``cl_gHI`` follows the
angular cut of its cell.  Rows failing a completeness limit receive σ = ∞
(zero Fisher weight) and are reported by the driver.

Exact invariants (the test contract)
------------------------------------

Every mechanism ships with at least one exact identity in
``tests/test_missing_physics.py`` — the extension's regression contract:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Invariant
     - Test
   * - Growth ODE ≡ Carroll (ΛCDM) to <0.5%; ≡ exact quadrature to <0.2%
     - ``test_growth_ode_lcdm_matches_carroll_and_exact``
   * - :math:`\partial D/\partial w_0 \ne 0`, correct sign
     - ``test_growth_responds_to_dark_energy``
   * - ν suppression ≡ 1 at Σm_ν = 0 (bit-identical); small scales only
     - ``test_nu_suppression_shape``
   * - ``_c_dk15`` ≡ core ``c_diemer15``; COLOSSUS anchors; ∂c/∂σ8 ≠ 0 (and = 0 for Dutton14)
     - ``test_dk15_*``
   * - SF + Q ≡ unsplit (occupations, n_gal, ρ_SFR) to 1e-12
     - ``test_sfq_split_sums_to_unsplit``, ``test_sfrd_observable``
   * - ``dlx_quenched``/``dhi_quenched`` touch ONLY quenched samples
     - ``test_d*_quenched_only_touches_*``
   * - rlf ≡ hard-band XLF at (ξ_RX, ξ_RM, b_R, σ_R) = (1,0,0,0), jets off
     - ``test_rlf_identity_with_xlf``
   * - Φ ∝ 10^f_ERDF exactly for FP-rlf and ilf; broken (deficit) by jets
     - ``test_rlf_amplitude_identity...``, ``test_ilf_observable``
   * - :math:`C_\ell^{g{\rm HI}} \propto M_0` exactly (∂ln C/∂log₁₀M₀ = ln 10)
     - ``test_hi_sector``
   * - SF-cut + Q-cut ≡ mixture-cut; cut → −∞ ≡ no cut
     - ``test_ssfr_cut_selection``
   * - ρ_SFR ∝ 10^{sSFR_MS} exactly
     - ``test_sfrd_observable``
   * - ∂Φ_[OII]/∂loii_norm ≡ ∂Φ_[OII]/∂ssfr_ms_norm (the designed degeneracy)
     - ``test_oiilf_observable``
   * - ilf has exactly zero ``agn_fabs`` response (obscuration-robust)
     - ``test_ilf_observable``
   * - wind η₀ = 0 ≡ tier-2 sigmoid bit-identically
     - ``test_wind_loading``
   * - CAMB ratio: bounded (<10%), differentiable, moves ∂/∂n_s
     - ``test_camb_ratio_correction``
   * - tier-2 assembly: split cells partition n_gal; shell/`hi_local` block
       structure; all new rows carry noise
     - ``test_tier2_split_sfq_assembly``, ``test_tier2_wave*_observables``

Usage
-----

Forward model (all opt-in; every default reproduces tier-2)::

    from hod_mod.forecast.forward_jax import ForwardModel

    m = ForwardModel(
        z_eff=0.35,
        cm_relation="diemer15",        # cosmology-dependent c(nu, n_eff)
        pk_correction="camb_linear",   # linearized CAMB/EH98 ratio
        sfq="sf",                      # or "q", or None
        ssfr_cut=-10.5,                # ELG-like sSFR threshold [log10 yr^-1]
    )
    out = m.predict(theta, ["wp", "rlf", "ilf", "oiilf", "himf",
                            "cl_gHI", "ssfr", "sfrd"])

Tier-2 forecast with everything enabled::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier2_forecast \
        --rmin 0.1 0.5 2.5 --n-bands 6 --split-sfq \
        --include-radio --include-hi --include-ssfr --include-ir

Rebuild the CAMB-ratio table after changing the fiducial (needs CAMB)::

    python -m hod_mod.forecast.pk_camb_ratio

Approximations and fixed constants
----------------------------------

Deliberate simplifications, each a candidate for promotion later:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Constant / choice
     - Value
     - Meaning and caveat
   * - :math:`\sigma_{\rm Q}`
     - 0.5 dex
     - width of the quenched sSFR lognormal (fixed)
   * - :math:`\sigma_{\rm jet}`
     - 0.7 dex
     - jet-luminosity scatter at fixed M_BH; :math:`\xi_{\rm jet} = 1` fixed
   * - :math:`\sigma_{\rm [OII]}`
     - 0.2 dex
     - extra [OII]-calibration scatter beyond the MS width
   * - :math:`\sigma_{M_{\rm HI}}`
     - 0.35 dex
     - HI-mass conditional scatter of the HIMF kernel
   * - ν suppression
     - first order
     - the −8f_ν tanh form + the CAMB-ratio derivative row; scale-dependent
       *growth* of ν is neglected
   * - CAMB ratio
     - linearized
     - valid near the fiducial (a Fisher application); rebuild for a new
       fiducial; edge-clamped beyond k ≈ 28 h/Mpc
   * - :math:`\alpha_w`
     - flat at fiducial
     - the wind slope decouples exactly at :math:`\eta_{w,0} = 0`
       (prior-bounded)
   * - [OII]/AGN kernels
     - centrals only
     - the Powell-chain convention; satellites are a documented refinement
   * - 21 cm IM noise
     - effective (rN, aN)
     - the calibrated-recipe precedent (tSZ); a physical T_sys model is the
       upgrade
   * - N_H template, ZM16 forms, V₀ = 200 km/s, z_pivot = 0.3
     - fixed
     - inherited tier-2 / wave conventions

Results
-------

The 90-parameter production forecast (six X-ray bands, SF/Q-split cells,
radio + IR + [OII] + HI observables — 177 blocks, 22 539 data rows) extends
the :doc:`tier2_forecast` decomposition.  Headline numbers at
:math:`r_\mathrm{min} = 0.1\,h^{-1}` Mpc:

.. list-table::
   :header-rows: 1
   :widths: 22 20 22 14

   * - Parameter
     - All 90 free
     - Astrophysics pinned
     - Degradation
   * - :math:`\Omega_\mathrm{m}`
     - :math:`1.82\times10^{-4}`
     - :math:`8.01\times10^{-5}`
     - ×2.3
   * - :math:`\sigma_8`
     - :math:`2.93\times10^{-4}`
     - :math:`1.22\times10^{-4}`
     - ×2.4
   * - :math:`w_0`
     - :math:`3.17\times10^{-3}`
     - :math:`1.46\times10^{-3}`
     - ×2.2
   * - :math:`w_a`
     - :math:`1.36\times10^{-2}`
     - :math:`6.22\times10^{-3}`
     - ×2.2
   * - :math:`\sum m_\nu` [eV]
     - :math:`3.23\times10^{-5}`
     - —
     - —

Despite 29 additional free parameters, the 90-parameter run *beats* the
61-parameter tier-2 baseline (:math:`\sigma(\Omega_\mathrm{m}) = 2.85\times
10^{-4}`, :math:`\sigma(\sigma_8) = 4.39\times10^{-4}`) by ~35%: the new
per-cell observables (sSFR, SFRD, :math:`C_\ell^{g\mathrm{HI}}`) and the
radio/[OII]/IR/HI shells add more information than the new freedom removes.
The cumulative probe attribution is dominated by galaxy grid → +lensing →
+X-ray/tSZ (:math:`3.1 \to 2.7 \to 1.9 \times 10^{-4}` on
:math:`\Omega_\mathrm{m}`), with the wave observables contributing
percent-level refinements thereafter.  The three analytically flat
directions (``eps_sn`` at the saturated energy-closure fiducial,
``alpha_w`` at the wind-off fiducial, ``dssfr_q`` without an sSFR cut)
return ``inf`` marginals exactly as designed — they are prior-bounded, not
data-constrained.  The dominant residual degeneracies are the intra-sector
ERDF triangle (``agn_sig_bh``–``agn_rho``–``agn_sig_mstar``,
:math:`|\rho| = 1.000`) and the gas metallicity-evolution pair
(``z_gas_norm``–``z_gas_zs``, :math:`\rho = -1.000`); the worst-constrained
Fisher directions are combinations of ``z_gas_norm``/``kt_slope``/``kt_zs``
and the jet duo ``f_loud0``/``beta_loud``.  Completeness pruning drops
49 rows (high-z [OII]/IR/radio/X-ray LF bins below the flux limits and the
two lowest HIMF masses).  SUMMARY, npz products, per-sector information
gains and the nine diagnostic figures are written to
``$HOD_MOD_RESULTS/tier2_forecast/`` (``*_nb6``).

API reference
-------------

.. automodule:: hod_mod.forecast.noise
   :members: ShearSurvey, CMBLensingSurvey, AthenaAllSky, RadioSurvey,
             IRSurvey, HISurvey, SpectroSurvey, athena_noise_cl_model,
             wp_pair_sigma, delta_sigma_noise, xlf_relerr

.. automodule:: hod_mod.forecast.pk_camb_ratio
   :members: build, load, apply_ratio

.. automodule:: hod_mod.forecast.tier2
   :members: Tier2Forecast

.. automodule:: hod_mod.forecast.tier3
   :members: Tier3Forecast

.. automodule:: hod_mod.forecast.apec_bands
   :members: band_tables, band_transmission, mm83_sigma
