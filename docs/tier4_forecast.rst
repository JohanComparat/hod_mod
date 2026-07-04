Tier-4 forecast: the morphology observables
===========================================

Wave 4 gave the model a morphology *sector* (the conditional early-type
fraction, the sample split, the BH–bulge coupling — see
:doc:`missing_physics_implementation`).  The tier-4 study adds the
*measurements* the literature actually delivers for it, and asks:

    **When galaxy morphology is measured at Euclid scale — fractions, joint
    morphology–quenching censuses, sizes, AGN-host demographics,
    morphology-split lensing and intrinsic alignments — what does it teach
    the halo model, and does the IA self-calibration pay off for cosmology?**

Five parameters join the vector (**111 in total**,
:data:`~hod_mod.forecast.forward_jax.TIER4_MORPHOLOGY`), all
fiducial-preserving.

Reproduce with::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier4_forecast \
        --jobs 6
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier4_forecast --smoke

The literature basis
--------------------

Every observable below is anchored to a measurement programme (all arXiv
links API-verified):

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

The five parameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 16 12 12 60

   * - Parameter
     - Fiducial
     - Prior
     - Meaning
   * - ``rho_morph_q``
     - 0.0
     - 0.3
     - morphology–quenching correlation at fixed M_h:
       :math:`f_{E\cap Q} = f_E f_Q + \rho\sqrt{f_E(1-f_E)f_Q(1-f_Q)}`
       (ρ = 0 is the wave-4 independence, exactly)
   * - ``log10_f_size``
     - −1.824
     - 0.3
     - log10(R_e/R_200c) — the Kravtsov 0.015 relation
   * - ``dsize_early``
     - −0.20
     - 0.3
     - early-type size offset at fixed M* [dex]
   * - ``f_size_zs``
     - 0.0
     - 2.0
     - size-ratio evolution (the ``_Z_EVOL`` mechanism)
   * - ``a_ia``
     - 2.0
     - 1.0
     - NLA IA amplitude carried by the early types
       (:math:`A_{\rm eff} = a_{\rm IA} f_{\rm early}`)

The observables
---------------

**Joint fractions and the 4-way split.**  The (early/late × SF/Q) partition
now uses the joint fractions with ``rho_morph_q`` — the four weights
partition unity by construction, EARLY+LATE+SF/Q sums remain exact at any ρ
(tested), and the SF/Q-split cells' ``f_early`` data become *conditional*
fractions (f_E|Q vs f_E|SF), so the split grid itself measures ρ.  The
per-cell ``f_early_q`` datum is the direct red-spiral / blue-elliptical
census.

**Sizes.**  ⟨log10 R_e⟩ per cell through R_e = f_size·R_200c (centrals
only, 0.2 dex Kravtsov scatter, ``dsize_early``·f_early offset).  Because
R_200c ∝ (M/ρ_crit)^{1/3}, galaxy sizes weigh cosmology — a clean, if
weak, new handle.  Exact gates: ∂⟨log R⟩/∂log10_f_size = 1;
the ``f_size_zs`` chain rule.

**Intrinsic alignments.**  w_g+(r_p) per cell in the NLA form with the
amplitude carried by the early-type fraction — implemented by feeding
:math:`a_{\rm IA} f_{\rm early}\,0.0134\,\Omega_m/D(z)\; b_g P_{\rm lin}`
through the existing ΔΣ (J₂-type) transform verbatim.  Exact gates:
∂ln w_g+/∂a_ia = 1/a_ia; w_g+ strictly ∝ f_early (morphology-parameter
changes rescale both identically — tested).  The payoff: the IA
contamination of the tomographic shear block shares parameters with the
morphology sector, so the forecast quantifies IA *self-calibration*.

**AGN hosts.**  ``f_early_agn`` per shell — the early fraction among
(soft) L_X = 10⁴²–10⁴⁴ AGN hosts.  With ``mbh_bt_slope`` > 0 the Powell
chain shifts AGN into early-type-rich haloes, so this datum measures the
BH–bulge coupling directly (the Kocevski-style bulge-dominance trend).

**Morphology-split clustering + lensing.**  Early/late (w_p, ΔΣ, n̄_g)
blocks per (z, M*) cell — the [Mandelbaum2006]_ measurement at Euclid
scale.  Wide-tier cells only, to ``z_morph_max = 1.2`` (imaging morphology
+ shear-source depth); block kind ``morph_cell``.

Noise
-----

Binomial counting + calibration floors for the fractions
(``SpectroSurvey.fmorph_err`` = 0.02, ``fmorph_agn_err`` = 0.05), the
Kravtsov scatter over √N + a 0.02 dex floor for sizes
(``SpectroSurvey.size_err``), shape noise per annulus for w_g+
(:func:`~hod_mod.forecast.noise.wgp_noise` — the ΔΣ geometry without the
Σ_crit weight), and the standard pair-count/shape-noise models for the
morph-split blocks.

Production configuration
------------------------

The tier-4 run is THE morphology-enabled production (the 102-parameter
tier-3 run is its baseline): the tier-3 block set (fresh cache — the vector
grew) plus ~150 light morph-split blocks and the per-cell/shell morphology
rows; 111 parameters, ``--jobs 6`` batched pools.  Results are folded in
here when the run completes, with three headline questions: the morphology
sector's information content, σ(``mbh_bt_slope``) from the AGN-host data,
and how much the IA self-calibration protects (or costs) the shear
cosmology.
