Tier-4 forecast: the morphology observables
===========================================

Wave 4 gave the model a morphology *sector* (the conditional early-type
fraction, the sample split, the BH–bulge coupling — see
:doc:`missing_physics_implementation`, "Wave 4").  The tier-4 study adds the
*measurements* the literature actually delivers for it, and asks:

    **When galaxy morphology is measured at Euclid scale — fractions, joint
    morphology–quenching censuses, sizes, AGN-host demographics,
    morphology-split lensing and intrinsic alignments — what does it teach
    the halo model, and does the IA self-calibration pay off for cosmology?**

Five parameters join the vector (**111 in total**,
:data:`~hod_mod.forecast.forward_jax.TIER4_MORPHOLOGY`), all
fiducial-preserving; together with the four wave-4 entries they form the
nine-parameter ``morphology`` sector.

Reproduce with::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier4_forecast \
        --jobs 6
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier4_forecast --smoke

.. contents::
   :local:
   :depth: 2

The literature basis
--------------------

Every observable is anchored to a measurement programme (all arXiv links
API-verified before inclusion — the house rule):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Measurement
     - Key references
   * - f_early(M*, z) at scale
     - Euclid Q1 morphology catalogue (378k galaxies, →100M)
       [EuclidQ1Morph2025]_; Zoobot ML morphologies [EuclidZoobot2024]_;
       Galaxy Zoo heritage [Skibba2009]_
   * - f_early evolution to z ≳ 3
     - JWST: CEERS [Kartaltepe2023]_, [Ferreira2023]_; COSMOS-Web Hubble
       sequence [COSMOSWeb2025Hubble]_
   * - Morphology-split lensing
     - SDSS halo masses vs morphology [Mandelbaum2006]_
   * - Morphology–quenching joint census
     - passive red spirals [Masters2010]_; two quenching pathways
       [Schawinski2014]_; causality [Bluck2022]_
   * - Sizes ↔ haloes
     - R_e ≈ 0.015 R_200c [Kravtsov2013]_; size–mass evolution
       [vanderWel2014]_
   * - Intrinsic alignments ↔ morphology
     - KiDS-1000: IA driven by morphology more fundamentally than colour
       [Georgiou2025]_
   * - AGN-host morphology
     - BH–bulge coevolution [Yang2019BHbulge]_, [KormendyHo2013]_; AGN vs
       inactive hosts at fixed M* [Banerjee2025]_

The morphology sector: all nine parameters
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 10 10 10 54

   * - Parameter
     - Fiducial
     - Prior
     - Wave
     - Meaning
   * - ``log10_M_morph``
     - 12.5
     - 1.0
     - 4
     - early-type Weibull transition mass (centrals)
   * - ``beta_morph``
     - 0.8
     - 0.5
     - 4
     - Weibull shape (transition sharpness)
   * - ``f_morph_sat``
     - 0.2
     - 0.3
     - 4
     - satellite boost toward early types,
       :math:`f_{E,s} = f_{E,c} + f_{\rm morph,sat}(1 - f_{E,c})`
   * - ``mbh_bt_slope``
     - 0.0
     - 0.5
     - 4
     - BH–bulge coupling:
       :math:`\langle\log M_{\rm BH}\rangle \mathrel{+}=
       {\rm slope}\cdot\log_{10}[f_{E,c}(M_h) + 10^{-4}]`
       (0 = the pure Powell chain, exactly)
   * - ``rho_morph_q``
     - 0.0
     - 0.3
     - t4
     - morphology–quenching correlation at fixed M_h:
       :math:`f_{E\cap Q} = f_E f_Q + \rho\sqrt{f_E(1-f_E)f_Q(1-f_Q)}`
       (ρ = 0 is the wave-4 independence, exactly; the square root carries
       a 1e-24 floor so the Jacobian stays finite where the Weibulls
       saturate to exactly 0/1)
   * - ``log10_f_size``
     - −1.824
     - 0.3
     - t4
     - log10(R_e/R_200c) — the Kravtsov 0.015 relation
   * - ``dsize_early``
     - −0.20
     - 0.3
     - t4
     - early-type size offset at fixed M* [dex]
   * - ``f_size_zs``
     - 0.0
     - 2.0
     - t4
     - size-ratio evolution (a ``_Z_EVOL`` pair with ``log10_f_size``:
       :math:`\partial d/\partial{\rm zs} = x_{\rm evol}\,
       \partial d/\partial{\rm base}` by the chain rule)
   * - ``a_ia``
     - 2.0
     - 1.0
     - t4
     - NLA IA amplitude carried by the early types
       (:math:`A_{\rm eff} = a_{\rm IA}\, f_{\rm early}`)

Fixed constants: ``_SIG_SIZE`` = 0.2 dex (the Kravtsov scatter, in
``forecast/tier4.py``); the NLA :math:`C_1\rho_{\rm crit}` = 0.0134
convention is absorbed into the ``a_ia`` normalisation.

Architecture: where everything lives
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Location
     - Contents
   * - ``connection/morphology.py``
     - ``f_early_cen(log10m_h, lg_m_morph, beta_morph)`` and
       ``f_early_sat(..., f_morph_sat)`` — jitted Weibull fractions
       (the ZM16 ``f_red_*_zu16`` pattern), ∈ [0, 1] by construction.
   * - ``forecast/forward_jax.py``
     - ``_morph_fractions`` (marginal f_E per cen/sat);
       ``_morph_q_joint`` (joint f_{E∩Q}, tier 4);
       ``_occ_sample`` — the 4-way (early/late × SF/Q) partition uses the
       joint fractions when BOTH ``morph`` and ``sfq`` are set, the
       marginal weights otherwise; ``_occ_sfq`` = the pre-morphology
       sample; ``_f_early`` (conditional on SF/Q-selected samples),
       ``_f_early_q``, ``_size_mean``, ``_f_early_agn``, ``_wgp``;
       the ``mbh_bt_slope`` term inside ``_agn_kernel_parts``;
       slices ``WAVE4_MORPHOLOGY = PARAM_NAMES[102:106]``,
       ``TIER4_MORPHOLOGY = PARAM_NAMES[106:]``.
   * - ``forecast/tier2.py`` (hooks, unchanged behaviour)
     - ``include_morph`` flag adds ``f_early`` to every cell;
       ``_cell_extra_obs`` / ``_shell_extra_obs`` / ``_extra_blocks`` /
       ``_cell_spectro`` / ``_block_extras`` are the extension points the
       tier-3/4 subclasses consume.
   * - ``forecast/tier4.py``
     - :class:`~hod_mod.forecast.tier4.Tier4Forecast`  — flags
       ``include_morphq`` / ``include_sizes`` / ``include_ia`` /
       ``include_agn_morph`` / ``include_morph_split`` (all default ON),
       ``z_morph_max = 1.2``; builds the ``morph_cell`` blocks and the
       tier-4 noise override.
   * - ``forecast/noise.py``
     - ``wgp_noise()`` (shape noise per annulus, the ΔΣ geometry without
       the Σ_crit weight); ``SpectroSurvey.fmorph_err`` = 0.02,
       ``size_err`` = 0.02 dex, ``fmorph_agn_err`` = 0.05.
   * - ``scripts/forecasts/run_tier4_forecast.py``
     - The driver: tier-3 probe groups + ``+f_early(z,M*)``,
       ``+E∩Q joint``, ``+sizes``, ``+AGN hosts``,
       ``+morph-split wp/dS`` (selected by block KIND via the
       ``__kind__morph_cell`` sentinel — its rows are named wp/ds/n_gal),
       ``+IA w_g+``; ``--jobs``, ``--smoke``, ``--no-morph-split``,
       ``--no-ia``.
   * - ``tests/test_forecast_tier4.py``
     - The 13 gates (below); wave-4 gates live in
       ``tests/test_missing_physics.py``.

The observables
---------------

**Joint fractions and the 4-way split.**  With both ``morph`` and ``sfq``
set, the occupation weights are the joint-partition set
:math:`\{f_{E\cap Q},\ f_E - f_{E\cap Q},\ f_Q - f_{E\cap Q},\
1 - f_E - f_Q + f_{E\cap Q}\}` (× the sSFR-cut survivals), which sums to
unity by construction — EARLY+LATE+SF/Q additivity is exact at *any* ρ.
The SF/Q-split cells' ``f_early`` data are **conditional** fractions:
:math:`f_{E|Q} = f_{E\cap Q}/f_Q` and
:math:`f_{E|SF} = (f_E - f_{E\cap Q})/(1 - f_Q)` — at ρ = 0 both reduce to
the marginal, so the existing split grid itself measures ρ (at ρ = 0.2 a
z = 0.4 cell separates to f_E|Q ≈ 0.30 vs f_E|SF ≈ 0.06).  The per-cell
``f_early_q`` datum is the base-sample joint fraction — the red-spiral /
blue-elliptical census — attached to ONE variant per (z, M*) cell (the SF
one) to avoid duplicating a sample-independent number.

**Sizes.**  Per cell (centrals only — satellite sizes do not follow the
host halo radius):

.. math:: \langle\log_{10} R_e\rangle = \log_{10} f_{\rm size}
    + \langle\log_{10} R_{200c}\rangle_{N_c}
    + \Delta_{\rm size}^{E}\, f_{\rm early}^{\rm cell},

with :math:`R_{200c}` the comoving ``r_delta`` of the halo grid [Mpc/h]
(the constant unit offset to physical kpc is absorbed by
``log10_f_size``).  Cosmology enters through
:math:`R_{200c} \propto (M/\rho_{\rm crit})^{1/3}`.

**Intrinsic alignments.**  Per cell, on the ``rp_ds`` grid:

.. math:: w_{g+}(r_p) = a_{\rm IA}\, f_{\rm early}^{\rm cell}\;
    \frac{0.0134\,\Omega_m}{D(z)}\;
    \Delta\Sigma\text{-transform}\big[b_g P_{\rm lin}\big],

reusing ``_delta_sigma`` (the J₂-type transform) verbatim and the
CPL-aware growth dispatcher for D(z).  Because the amplitude is carried by
:math:`f_{\rm early}`, the IA contamination of the tomographic shear block
shares parameters with the morphology sector — the forecast quantifies IA
*self-calibration*.

**AGN hosts.**  Per shell: the early fraction among soft
L_X = 10⁴²–10⁴⁴ AGN,
:math:`f_E^{\rm AGN} = \int dM\, \frac{dn}{dM}\, N_{\rm AGN} f_{E,c}
\big/ \int dM\, \frac{dn}{dM}\, N_{\rm AGN}` with the observed-band
occupation ``_agn_occupation_obs``.  With ``mbh_bt_slope`` > 0 the Powell
chain shifts AGN into early-type-rich haloes — the Kocevski-style
bulge-dominance trend, measured directly.

**Morphology-split clustering + lensing.**  Early/late (w_p, ΔΣ, n̄_g)
blocks per (z, M*) cell — [Mandelbaum2006]_ at Euclid scale.  Kind
``morph_cell``; built only where the WIDE spectroscopic tier is complete
(imaging morphology needs depth) and z_hi ≤ ``z_morph_max`` = 1.2 (where
the shear sources also run out): the production grid yields 104 of the
naive 156 blocks.

The exact gates
---------------

Every mechanism carries an invariant enforced by
``tests/test_forecast_tier4.py`` (13 tests) or the wave-4 section of
``tests/test_missing_physics.py``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Invariant
     - Meaning
   * - ``f_EQ(ρ=0) ≡ f_E·f_Q`` (rtol 1e-14)
     - tier 4 collapses to wave-4 independence exactly at the fiducial
   * - ∂f_EQ/∂ρ = √(f_E(1−f_E)f_Q(1−f_Q)) analytically
     - the correlation lever arm is exact
   * - 4-way partition sums to the unsplit n_gal at ρ ∈ {0, ±0.x}
     - unity partition by construction, not by tuning
   * - f_E|Q > f_E|SF at ρ > 0
     - the conditional split carries the ρ signal
   * - 0 < f_early_q < f_early < 1, ∂/∂ρ > 0
     - joint bounded by the marginals
   * - ∂⟨log R⟩/∂``log10_f_size`` = 1 exactly; ``f_size_zs`` chain rule
       = x_evol; nonzero (Ωm, h) response; sizes grow with the M* bin
     - the size observable is a calibrated ruler
   * - ∂ln w_g+/∂``a_ia`` = 1/a_ia exactly;
       w_g+(θ₂)/w_g+(θ₁) ≡ f_early(θ₂)/f_early(θ₁) under
       morphology-parameter changes
     - strict NLA amplitude and strict morphology-carried scaling
   * - ∂f_early_agn/∂``mbh_bt_slope`` > 0 at the fiducial;
       responds to ``log10_M_morph``
     - the BH–bulge coupling is observable
   * - w_g+ obeys the projected r_p scale cut
     - bookkeeping
   * - tiny ``Tier4Forecast`` end-to-end: finite d0/J, positive noise,
       every family present, ``morph_cell`` rows ∈ {wp, ds, n_gal},
       cache round-trip identical
     - assembly
   * - EARLY+LATE ``morph_cell`` n_gal ≡ the unsplit cell total
     - the split blocks add information, not amplitude
   * - parallel == serial BYTE-identical (fresh-interpreter subprocess;
       batched pools)
     - the --jobs machinery is exact

Usage
-----

Model level::

    from hod_mod.forecast.forward_jax import ForwardModel
    m = ForwardModel(z_eff=0.4, log10m_star_bin=(10.0, 10.4),
                     morph="early", sfq="q")     # the E∩Q sample
    out = m.predict(theta, ["wp", "ds", "n_gal"])
    m2 = ForwardModel(z_eff=0.4, log10m_star_bin=(10.0, 10.4))
    out2 = m2.predict(theta, ["f_early", "f_early_q", "size",
                              "wgp", "f_early_agn"])

Forecast level::

    from hod_mod.forecast.tier4 import Tier4Forecast
    t4 = Tier4Forecast()                  # all tier-2/3/4 flags default ON
    d0, J, meta = t4.data_and_jacobian(t4.fiducial(), cache_dir=cache)
    sigma = t4.noise_sigma(t4.fiducial(), d0, meta)
    # morph-split rows: meta["kind"] == "morph_cell"

Noise
-----

.. list-table::
   :header-rows: 1
   :widths: 18 48 34

   * - Row
     - σ model
     - Completeness
   * - ``f_early``, ``f_early_q``
     - binomial √(f(1−f)/N) ⊕ ``fmorph_err`` = 0.02 floor
     - wide/deep spectro tier of the cell
   * - ``size``
     - ``_SIG_SIZE``/√N ⊕ ``size_err`` = 0.02 dex floor
     - idem
   * - ``wgp``
     - :func:`~hod_mod.forecast.noise.wgp_noise` — shape noise per
       annulus (ΔΣ geometry, no Σ_crit weight) ⊕ CV floor
     - idem
   * - ``f_early_agn``
     - binomial over N_AGN(shell) ⊕ ``fmorph_agn_err`` = 0.05 floor
     - flagged where the shell lacks the AGN extras
   * - ``morph_cell`` wp/ds/n_gal
     - the standard pair-count / shape-noise / count models
     - wide tier only, z ≤ 1.2 (blocks not built otherwise)

The w_g+ recipe is *effective* (no IA–clustering cross-covariance) — the
tSZ/IM documentation precedent; the AGN-host floor is larger than the
galaxy one because point-source light complicates host classification.

Operational record
------------------

* Developed in a git worktree on branches ``feature/wave4-morphology`` →
  ``feature/tier4-morphology`` while the tier-3 production ran: the
  ``--jobs`` workers **re-import the package source at every batch
  respawn**, so in-place edits during a run are forbidden.
* The vector growth invalidates all previous per-block caches
  (``PARAM_NAMES`` enters ``_spec_repr``) — a tier-4 production is a fresh
  full run.
* Production: 396 blocks (260 SF/Q cells, 104 morph-split, 10 shells,
  global lensing, 10 AGN, 10 SFRD, hi_local), ``--jobs 6`` batched pools
  (a fresh executor per chunk of jobs × 4 blocks — CPython 3.11's
  ``max_tasks_per_child`` respawn deadlocks; batching gives the same
  per-worker memory bound), ~4.5 h wall on 6 workers / 64 GB.

Results (2026-07-04 run)
------------------------

396 blocks, 65 431 rows (260 SF/Q cells + **104** morph-split blocks — the
wide-tier completeness gate trims the naive 156: at z_hi = 1.2 imaging
morphology only survives above M* ≈ 10^{10.2}).  At
:math:`r_\mathrm{min} = 0.1\,h^{-1}` Mpc, all 111 free:

.. list-table::
   :header-rows: 1
   :widths: 22 20 22 14

   * - Parameter
     - All 111 free
     - Astrophysics pinned
     - Degradation
   * - :math:`\Omega_\mathrm{m}`
     - :math:`4.89\times10^{-5}`
     - :math:`3.33\times10^{-5}`
     - ×1.5
   * - :math:`\sigma_8`
     - :math:`7.23\times10^{-5}`
     - :math:`5.25\times10^{-5}`
     - ×1.4
   * - :math:`w_0`
     - :math:`7.23\times10^{-4}`
     - :math:`4.58\times10^{-4}`
     - ×1.6
   * - :math:`w_a`
     - :math:`3.35\times10^{-3}`
     - :math:`2.10\times10^{-3}`
     - ×1.6

Answers to the three headline questions:

* **The morphology sector is fully measured** (84 bits of information; no
  prior-bound directions): the Weibull transition at the sub-permille level
  (σ(``log10_M_morph``) = 5.4×10⁻⁴, σ(``beta_morph``) = 7.2×10⁻⁴), the
  satellite boost and joint E∩Q correlation at a few ×10⁻³
  (σ(``rho_morph_q``) = 3.0×10⁻³), and the size relation to 1.9×10⁻³ dex
  with its early-type offset at 5.0×10⁻⁴ (the ``log10_f_size``–
  ``dsize_early`` pair is the sector's strongest internal degeneracy,
  ρ = +0.996).
* **σ(mbh_bt_slope) = 7.7×10⁻⁴** — the BH–bulge coupling is constrained
  ~650× beyond its prior by the AGN-host early fractions and the
  morphology-coupled AGN LFs: the coevolution question becomes decisively
  testable in this data scenario.
* **The IA amplitude self-calibrates to σ(a_ia) ≈ 0.010** (0.5% of the
  fiducial A_IA = 2) — the dominant weak-lensing systematic is pinned by
  the morphology sector *for free*, with no cost to the shear cosmology.

On the cosmology side, the new information comes almost entirely from the
**morph-split w_p/ΔΣ blocks**, which tighten
:math:`\sigma(\Omega_\mathrm{m})` by 17% (5.89 → 4.89×10⁻⁵) and
:math:`\sigma(\sigma_8)` by 9% over the tier-3 baseline — the scalar
morphology fractions and sizes constrain the sector itself rather than
cosmology, exactly as designed.  The tier progression now reads
:math:`\sigma(\Omega_\mathrm{m}) = 2.85 \to 1.82 \to 0.59 \to 0.49
\times 10^{-4}` across tier-2(61) → tier-2(90) → tier-3(102) →
tier-4(111).  The tier-3 conditioning caveat stands
(:math:`\Sigma m_\nu` and the agn-sector determinant hit the prior-scaled
eigenvalue floor at this dynamic range).  SUMMARY, npz and the
``tier4_forecast__*`` figure suite live in
``$HOD_MOD_RESULTS/tier4_forecast/``.

Continuation points
-------------------

Open items, in rough priority order, with their entry points:

1. **Eigenvalue-floor fix** — ``fisher.constraints`` drops prior-scaled
   eigenvalues below ``rcond``·λ_max; at tier-3/4 dynamic range
   (λ_max > 10¹⁰ in prior units) this sweeps up prior-bounded directions
   (Σm_ν → ∞, agn-sector slogdet → NaN at r_min = 0.1/0.5).  Principled
   fix: in prior-scaled space every eigenvalue is ≥ ~1 by construction, so
   FLOOR at 1 instead of dropping.  One function
   (``forecast/fisher.py::constraints``); changes all tiers' reported σ
   slightly — rerun the SUMMARYs from the cached npz (no model
   recomputation needed: the npz stores d0/J).
2. **Morphology evolution** ``log10_M_morph_zs`` — one ``_Z_EVOL`` pair +
   a prior; the JWST f_early(z) anchors ([Kartaltepe2023]_,
   [COSMOSWeb2025Hubble]_) are the data.  Trivial with the existing
   mechanism; adds the 112th parameter.
3. **Satellites in the morphology-coupled kernels** — ``_size_mean``,
   ``_f_early_agn`` and the AGN chain are centrals-only (the Powell
   convention).  A satellite term needs a subhalo size/BH prescription.
4. **Assembly-bias hook** — order B/T at fixed M_h by formation time via
   the cosmology-dependent c(M) (``cm_relation="diemer15"`` provides the
   proxy); the falsifiable Wechsler–Tinker link, roadmap item 3.
5. **IA refinements** — luminosity scaling
   (:math:`A_{\rm IA} \propto (L/L_0)^{\beta}`), a blue-galaxy tidal-torque
   term, and the IA–shear cross-covariance in the noise model
   (currently the effective shape-noise recipe).
6. **Morphology → SED coupling** — early types carry higher M/L: a
   ``dml_early`` offset in the tier-3 NIR/optical calibrations would tie
   the two sectors (one parameter, `_lf_lognormal` weights).
7. **Data-side** — the Euclid Q1 catalogue [EuclidQ1Morph2025]_ is public:
   a real f_early(M*, z) measurement against a Q1-matched selection would
   ground the fiducials before the next production.

API reference
-------------

.. automodule:: hod_mod.connection.morphology
   :members: f_early_cen, f_early_sat

.. automodule:: hod_mod.forecast.tier4
   :members: Tier4Forecast

.. autofunction:: hod_mod.forecast.noise.wgp_noise
   :no-index:
