Sensitivity benchmark: the existing observables the model must reproduce
=========================================================================

The :doc:`sensitivity pages <sensitivity_fisher>` forecast what the forward
model *will* measure under stated survey premises.  This page asks the prior
question, in the spirit of the UniverseMachine programme [Behroozi2019]_:

    **Which already-published measurements must the forward model reproduce
    before its forecasts mean anything — and, given the signal-to-noise of
    the data that exist today, which part of the full model can be fit
    right now?**

It has three parts: the benchmark compilation itself (the UniverseMachine
set, then the multi-wavelength expansion matched to this pipeline's
observables), the *run-today* assessment (§ :ref:`bench-today`), and the
*missing-data* table — what is absent, which experiment delivers it, and
when (§ :ref:`bench-missing`).  Every entry cites a published measurement;
the reference list with links is on :doc:`references`.

.. admonition:: The data live in the data repository

   The machine-readable companion of this page is
   ``$HOD_MOD_DATA_DIR/benchmark_observables/`` — one JSON per
   (reference, observable, sample), organised per wavelength and tracer,
   with full metadata, uncertainties and provenance.  See
   :ref:`bench-data-tree` below for the layout, the file schema and the
   operator workflow.

.. contents::
   :local:
   :depth: 2

----

The UniverseMachine precedent
-----------------------------

UniverseMachine [Behroozi2019]_ fits an empirical galaxy–halo model to a
deliberately *complete* compilation of one-point and two-point statistics
(their Table 1; all datasets re-normalised to uniform stellar-population,
dust and SFH assumptions, their Appendix C):

.. list-table:: The UniverseMachine constraint set.
   :header-rows: 1
   :widths: 30 14 56

   * - Observable
     - Redshift
     - Data sources (as compiled in [Behroozi2019]_)
   * - Stellar-mass functions
     - 0 < z < 4
     - SDSS, PRIMUS [Moustakas2013]_, UltraVISTA [Muzzin2013]_, CANDELS,
       ZFOURGE
   * - Cosmic SFR density
     - 0 < z < 10
     - UV / 24 μm / radio / Hα / SED compilations
   * - Specific SFRs
     - 0 < z < 8
     - same surveys as the CSFR compilation
   * - Quenched fractions
     - 0 < z < 3.5
     - Bauer et al. 2013, [Moustakas2013]_, [Muzzin2013]_
   * - UV luminosity functions
     - 4 < z < 10
     - [Finkelstein2015]_, Bouwens et al. 2016
   * - Galaxy autocorrelations (all / SF / quenched)
     - z ≈ 0; z ≈ 0.5
     - SDSS (re-measured); PRIMUS (Coil et al. 2017)
   * - Massive × MW-mass cross-correlation
     - z ≈ 0
     - SDSS (constrains satellite disruption)
   * - Environmental quenching of centrals
     - z ≈ 0
     - SDSS (quenched fraction vs. neighbour density)
   * - UV–M* relations
     - 4 < z < 8
     - [Song2016]_ (re-derived SED stacks)
   * - IRX–UV dust relations
     - 4 < z < 7
     - Bouwens et al. 2016 (ALMA)

Two lessons carry over.  First, *completeness is the constraint*: no single
statistic pins the galaxy–halo connection; the compilation does.  Second,
*re-normalisation to shared assumptions* (IMF, SPS, dust) is part of the
data model — the same discipline the tier-2/3 pages apply through their
SED-calibration parameters (``ml_nir``, ``luv_norm``, …).

The expanded multi-wavelength benchmark set
-------------------------------------------

The forward model predicts strictly more than the UniverseMachine set: gas,
AGN and lensing observables join the stellar ones.  For every observable in
the tier-1/2/3 vector, the table lists the current best *published*
measurement and its precision — this is the data vector an actual fit (as
opposed to a Fisher forecast) would use today.

.. list-table:: Benchmark measurements per model observable.
   :header-rows: 1
   :widths: 20 34 46

   * - Model observable
     - Current best measurement
     - Precision / signal-to-noise today
   * - ``wp`` per (z, M*) cell
     - SDSS DR7 per-sample w_p [Zehavi2011]_; DESI DR1 small-scale
       clustering + lensing (`arXiv:2512.15962
       <https://arxiv.org/abs/2512.15962>`_)
     - few-% per point at z < 0.2; DESI extends to z ≈ 1 (LRG)
   * - ``ds`` (ΔΣ)
     - SDSS [Mandelbaum2006]_, [Leauthaud2017]_; DESI×DES/KiDS lensing
       [Heydenreich2025]_ [Lange2025]_
     - 5–15 % per radial bin per sample
   * - ``n_gal`` / SMF (grid densities)
     - GAMA DR4 low-z SMF [Driver2022]_; COSMOS2020 to z = 5.5
       [Weaver2023]_
     - few-% (wide, low z) to ~0.1 dex (deep, high z)
   * - SF/quenched split (``sfq`` sector)
     - COSMOS2020 quiescent SMFs [Weaver2023]_; PRIMUS quenched fractions
       [Moustakas2013]_
     - ~0.05–0.1 dex per (z, M*) bin
   * - sSFR main sequence (``ssfr_ms_*``)
     - MS compilation 0 < z < 6 [Popesso2023]_
     - ~0.1 dex normalisation, ~0.2 dex scatter
   * - ``sfrd``
     - Madau–Dickinson compilation [MadauDickinson2014]_; LoTSS radio view
     - ~12 % per Δz shell (the tier-3 noise assumption is current-data)
   * - shear tomography (``cl_kk``)
     - KiDS-Legacy, 1347 deg² [Wright2025]_ (+ DES Y3 joint)
     - S8 to 2.3 %
   * - ``cl_kCMB`` (CMB-lensing auto)
     - ACT DR6 [Qu2024]_ / Planck PR4
     - 43σ (2.3 % amplitude)
   * - ``cl_gkCMB``
     - DESI LRG × ACT DR6 + Planck, 4 tomographic bins [Kim2024]_
     - 38–50σ; S8 to 2.7 %
   * - ``cl_gy`` (galaxy × tSZ)
     - ACT × BOSS CMASS stacked tSZ + kSZ profiles [Schaan2021]_
       [Amodeo2021]_
     - ~10σ-level profile detections per sample
   * - ``cl_gX`` (galaxy × soft X-ray, bands)
     - eROSITA × legacy galaxy samples: band-resolved X-ray profiles and
       scaling relations [Comparat2025]_
     - L_X–M slope to ±0.09 (one broad band + coarse sub-bands)
   * - ``cl_XX`` (X-ray auto, bands)
     - no published tomographic soft-band auto at the assumed depth
     - — (eRASS-depth CXB fluctuation analyses only)
   * - ``xlf`` (z-resolved AGN XLF)
     - Chandra/deep-fields compilation 0 < z < 7 [Aird2015]_
     - ~0.1 dex at L_X > 10⁴³; faint end (10⁴²) pencil-beam only at z ≈ 1
   * - ``wp_agn`` / AGN bias
     - eROSITA eRASS1 AGN clustering [Comparat2023]_; local BASS AGN
       [Powell2022]_
     - bias per L_X bin at ~10 % (broad L_X bins)
   * - AGN sector external pins
     - M_BH census [KormendyHo2013]_ [Greene2020]_; BASS DR2 ERDF
       [Ananna2022]_
     - μ_BH to ~0.1 dex (local); ERDF break/slopes measured at z ≈ 0
   * - ``qlf_uv`` / ``qlf_opt``
     - homogenised type-1 QLF 0 < z < 7.5 [Kulkarni2019]_
     - ~0.05–0.1 dex over the bright end
   * - ``rlf`` (radio LFs)
     - LoTSS Deep: SF radio LF / L₁₅₀–SFR [Bonato2021]_, jet-mode AGN LF to
       z ≈ 2.5 [Kondapally2022]_
     - ~0.1 dex per (L, z) bin
   * - ``cl_gR`` / ``cl_RR`` (radio maps)
     - no µJy-depth wide-area maps at 0.95–3 GHz yet (SKA premise)
     - —
   * - ``ilf`` / ``cl_gI`` (IR)
     - WISE all-sky 3.4/4.6/12 μm imaging [Wright2010]_ (the tier-3 band
       choice); SPHEREx spectral maps arriving [Dore2014]_
     - WISE crosses measurable now; spectro-IR from 2026–27
   * - ``uvlf``
     - GALEX local [Wyder2005]_; z = 4–8 [Finkelstein2015]_; z = 9–16 JWST
       [Harikane2023]_
     - ~0.1–0.2 dex per bin
   * - ``half`` (Hα LF)
     - HiZELS z = 0.4–2.2 [Sobral2013]_
     - ~0.1 dex per bin
   * - ``oiilf``
     - [OII] LF compilation to z ≈ 1.6 [Comparat2015OII]_
     - ~0.1 dex per bin
   * - ``himf``
     - ALFALFA final HIMF [Jones2018ALFALFA]_; MeerKAT MIGHTEE-HI
       [Ponomareva2023]_
     - knee mass ±0.01 (stat), slope ±0.02 — *local only*
   * - ``cl_gHI`` (21 cm × galaxies)
     - CHIME × eBOSS stacking [CHIME2023]_
     - 11σ (z ≈ 0.8–1.4, coarse scales)
   * - ``cl_HIHI`` (21 cm auto)
     - first CHIME auto-power detection [CHIMEauto2025]_
     - detection-level (foreground-limited)
   * - ``ncl`` (cluster counts)
     - eRASS1 abundances, 5259 clusters / 12 791 deg² [Ghirardini2024]_
     - σ8 = 0.88 ± 0.02, S8 = 0.86 ± 0.01 — a *live* benchmark (and a
       ~2σ-high S8, i.e. a real test, not a formality)
   * - geometry for (w0, wa, h)
     - DESI DR2 BAO [DESIDR2]_; DR1 full-shape [DESI2024FS]_
     - w0waCDM preferred at 3.1σ (with CMB); σ8 = 0.842 ± 0.034 alone

.. _bench-today:

What part of the model can be run today
---------------------------------------

Matching the table against the forecast premises, sector by sector:

* **Tier-1 (31 parameters, 12 observables) — runnable in full.**  Every
  tier-1 observable has a published counterpart at usable S/N: clustering
  and g-g lensing [Zehavi2011]_ [Mandelbaum2006]_, abundances [Driver2022]_,
  shear [Wright2025]_, CMB lensing and its crosses [Qu2024]_ [Kim2024]_,
  tSZ [Schaan2021]_, X-ray cross [Comparat2025]_ and XLF [Aird2015]_.  The
  attainable cosmology is today's LSS state of the art — S8 at the 2–3 %
  level (KiDS 2.3 %, LRG×κ_CMB 2.7 %, eRASS1 1.2 %) — not the 0.1 % of the
  Stage-IV error model; the *structure* of the study (degeneracy breaking,
  baryon calibration) is testable now, the headline precision is not.

* **Tier-2 galaxy grid — roughly the z < 0.5 half.**  The premise is a
  volume-limited M* > 10¹⁰ grid to z = 1 over f_sky = 0.5.  Today DESI
  (~14 000 deg²) with 4MOST now in survey operations covers wide-area
  spectroscopy to z ≈ 0.5–0.6 for these masses; beyond that the samples
  are colour-selected (LRG) or deep-pencil-beam (COSMOS).  About half the
  80 (z, M*) cells — and with COSMOS2020-style deep fields, the SF/Q split
  [Weaver2023]_ — can be populated with real data now; the z-evolution
  slopes lose roughly half their lever arm.

* **X-ray band spectroscopy — collapses to the 1-band control.**  eROSITA
  provides the all-sky broad band and coarse sub-bands at
  t·A_eff two orders of magnitude short of the Athena premise
  (F_lim ≈ 2×10⁻¹⁶ erg/s/cm²); the tier-2 ``--n-bands 1`` control run
  documents exactly what survives: Γ_AGN, f_abs and the ICM-metallicity
  triple fall back to their priors, kT_norm becomes flat.  The gas sector
  is fittable today at broad-band level [Comparat2025]_; its
  *spectroscopic* refinement is not.

* **Shear/CMB-lensing sector — at ~4× the forecast noise.**  KiDS-Legacy's
  1347 deg² is f_sky ≈ 0.033 against the assumed 0.5, so cosmic-variance
  errors are ~3.9× the tier-2 assumption; ACT DR6 κκ is already within
  reach of the assumed S4-like noise floor.  Rubin (survey started
  30 June 2026) and Euclid DR1 close the shear gap from late 2026.

* **AGN sector — runnable at z < 1 with today's pins.**  XLF [Aird2015]_,
  clustering [Comparat2023]_ [Powell2022]_, the local M_BH census
  [KormendyHo2013]_ [Greene2020]_ and the BASS ERDF [Ananna2022]_ exist;
  the per-L_X-bin completeness to L_X = 10⁴² at z = 1 (the Athena premise)
  does not — eROSITA is complete there only above ~10⁴³·⁵.

* **Radio / IR / HI (tier-3) — LFs yes, maps partly.**  The radio and
  [OII]/Hα/UV/QLF luminosity functions all have published counterparts
  ([Bonato2021]_ [Kondapally2022]_ [Comparat2015OII]_ [Sobral2013]_
  [Wyder2005]_ [Kulkarni2019]_), so the SED-calibration parameters are
  fittable now.  Of the map crosses, the IR ones are available today —
  the tier-3 3.4/4.9/12 μm bands are the WISE bands [Wright2010]_ — while
  the 0.95–3 GHz radio maps at µJy depth await the SKA, and the HI sector
  has only the local HIMF [Jones2018ALFALFA]_ plus low-S/N 21 cm crosses
  [CHIME2023]_ [CHIMEauto2025]_.

* **Extended cosmology — geometry yes, growth machinery not yet.**  DESI
  DR2 BAO already delivers the (w0, wa) geometry [DESIDR2]_ and full-shape
  σ8 [DESI2024FS]_; on the model side the freed (w0, wa, Σm_ν) act through
  the CPL growth ODE only, so a differentiable P(k)-shape upgrade remains
  the prerequisite for fitting them against these data (see the tier-2
  caveats).

Summed up: **all 12 tier-1 observables and roughly two thirds of the
tier-2/3 data rows have published counterparts today**, at per-point errors
3–30× the end-of-decade assumptions; the sectors with *no* data path today
are the multi-band X-ray spectroscopy, the deep radio-map crosses, and
tomographic 21 cm.

.. _bench-missing:

What is missing, and which experiment delivers it
-------------------------------------------------

.. list-table:: Missing data → delivering experiment → expected availability.
   :header-rows: 1
   :widths: 30 40 30

   * - Missing ingredient
     - Experiment / release
     - When
   * - Volume-limited M* > 10¹⁰ grid to z = 1, wide sky
     - Euclid DR1 (1900 deg² spectroscopy + photometry, 21 Oct 2026);
       DESI extended operations; 4MOST surveys (in operations since
       Q2 2026)
     - 2026 →
   * - Stage-IV shear at f_sky ≈ 0.4–0.5
     - Rubin LSST (survey started 30 June 2026; DR1 ≈ one year of survey
       + one year of processing); Euclid wide releases; Roman (launch
       30 Aug 2026) for the high-z calibration
     - 2027–2030
   * - S4-like CMB lensing + tSZ depth over half the sky
     - Simons Observatory full facility (operating since 2025); enhanced
       LAT upgrade (30 000 extra detectors)
     - upgrade complete ≈ 2028
   * - Multi-band X-ray spectra at F_lim ≈ 2×10⁻¹⁶, all-sky; AGN
       completeness L_X > 10⁴² to z = 1
     - NewAthena (ESA adoption Q1 2027)
     - launch 2037
   * - µJy radio intensity maps at 0.95–3 GHz
     - SKA-Mid (science verification 2029; Cycle-0 PI observations 2032);
       SKA-Low science verification 2027 [Bacon2018]_
     - 2029–2032
   * - All-sky spectro-photometric IR maps (102 bands)
     - SPHEREx (launched Mar 2025; weekly quick releases since Jul 2025;
       full-depth all-sky spectral maps over the 2-yr mission)
       [Dore2014]_
     - 2026–2027
   * - HIMF beyond z ≈ 0.05 and tomographic 21 cm
     - MeerKAT/MIGHTEE-HI [Ponomareva2023]_ deep pointings; SKA-Mid HI
       surveys [SKA2019]_
     - late 2020s → 2030s
   * - Time-domain M_BH census at scale (breaks the σ_lm triple)
     - SDSS-V Black Hole Mapper [Kollmeier2017]_ (ongoing)
     - ongoing → late 2020s

Literature-grounded extensions
------------------------------

Following the citation graph of the papers behind this benchmark (each
proposal is a *published measurement* the pipeline does not yet predict, or
a published relation it does not yet exploit):

* **kSZ profiles.**  ACT × CMASS kSZ measurements exist now [Schaan2021]_
  and were converted into gas thermodynamic profiles in [Amodeo2021]_ —
  a momentum-weighted (n_e × v) observable that breaks the
  density–temperature degeneracy of the X-ray/tSZ pair.  The model already
  carries n_e(r|M); it needs only the linear-theory velocity weight.
* **FRB dispersion measures.**  The Macquart relation [Macquart2020]_ is a
  published, direct census of the same ionised baryons the ``f_b(M)``
  sector redistributes — an absolute calibration of Σ n_e that neither
  X-ray (n_e²) nor tSZ (n_e T) provides.
* **BAO / full-shape multipoles.**  With (w0, wa, h) now free, the DESI
  DR2 BAO distances [DESIDR2]_ and DR1 full-shape multipoles [DESI2024FS]_
  are the natural geometric data block — contingent on the P(k) upgrade.
* **Tomographic galaxy × CMB-lensing data.**  The published 4-bin LRG ×
  κ_CMB data vector [Kim2024]_ is exactly the model's ``cl_gkCMB`` — a
  benchmark fit that needs no new machinery at all.
* **External AGN pins.**  The local M_BH census [KormendyHo2013]_
  [Greene2020]_ and the BASS DR2 ERDF [Ananna2022]_ pin
  (μ_BH, σ_BH, λ*, δ₁, δ₂) outside the σ_lm kernel — the published route
  to breaking the tier-2 σ_lm triple, with SDSS-V [Kollmeier2017]_
  extending the census.
* **Environmental quenching and satellite-disruption statistics.**  The
  UniverseMachine appendices publish central-quenching-vs-density and
  massive×MW cross-correlations [Behroozi2019]_ that discriminate
  one-halo conformity and satellite disruption — observables the SF/Q
  split (``sfq`` sector) could predict with a per-halo environment
  kernel.
* **UV–M* + dust relations.**  The z = 4–8 UV–M* stacks [Song2016]_ and
  IRX-type relations directly constrain ``luv_norm``/``tau_uv_mslope``
  outside 0 < z < 2 — the published path for extending the grid in
  redshift, together with the JWST UVLF frontier [Harikane2023]_.
* **Jet-mode AGN demographics.**  The LoTSS LERG luminosity functions and
  their quiescent/SF host split [Kondapally2022]_ benchmark the wave-3
  radio-loud sector (``f_loud0, beta_loud, b_jet``) as a function of host
  type, not just in aggregate.
* **Cluster-count cosmology as a consistency test.**  eRASS1 abundances
  [Ghirardini2024]_ give S8 = 0.86 ± 0.01 — ~2σ *above* the lensing
  probes; the model's ``ncl`` observable with its free L_X–M relation is
  positioned to test whether scaling-relation freedom absorbs the offset.
* **21 cm auto-spectrum.**  The first CHIME auto-power detection
  [CHIMEauto2025]_ upgrades ``cl_HIHI`` from forecast-only to
  benchmarkable.

.. _bench-data-tree:

The benchmark data tree
-----------------------

The compilation above is materialised as a JSON tree in the data
repository, ``$HOD_MOD_DATA_DIR/benchmark_observables/`` — **83 files**,
one per (bibliographic reference, observable, sample):

.. code-block:: text

   benchmark_observables/
     README.md                # schema + operator workflow
     index.json               # every file with provenance + extraction flag
     <wavelength>/<tracer>/<RefKey>__<observable>[__<sample>].json

with ``wavelength`` ∈ {radio, infrared, optical, uv, xray, microwave,
multiwavelength} and ``tracer`` ∈ {galaxies, agn, clusters, hi, gas,
lensing, cmb_lensing, blackholes}.  The tree is generated (and
regenerated after any change) by::

    python -m hod_mod.scripts.data.make_benchmark_observables \
        --out $HOD_MOD_DATA_DIR/benchmark_observables

File schema (``schema_version`` 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Content
   * - ``wavelength``, ``tracer``, ``observable``, ``sample_id``
     - position in the taxonomy; ``observable`` uses the model's names
       (``wp``, ``ds``, ``xlf``, ``himf``, …)
   * - ``reference``
     - ``key`` (matching the entries on :doc:`references`), ``citation``,
       ``arxiv`` link, optional ``doi`` and ``note``.  For datasets
       ingested from the curated ``data/{paper}`` folders the arXiv/DOI
       come from the folder's ``metadata.json`` (authoritative)
   * - ``provenance``
     - ``type`` (see the classes below), ``origin`` (file path or
       formula), optional ``extraction_method``, and the
       ``needs_operator_extraction`` flag with its ``extraction_hint``
   * - ``sample``
     - survey, redshift, selection and the measurement's own cosmology
   * - ``units``
     - per-column units (h-conventions stated where they differ from the
       model's)
   * - ``data``
     - column arrays including uncertainties; ``null`` marks a masked or
       non-finite entry

Provenance classes
~~~~~~~~~~~~~~~~~~

* **observed (45)** — real measurements ingested from the curated local
  sources: the nine benchmark folders under the repo's ``data/``
  (Zehavi/SDSS ``wp``, Guo 2018/2019, Leauthaud 2012 ΔΣ, More 2015 CMASS
  ×3 samples, van Uitert 2016 ΔΣ, Zacharegkas 2025 DES, Zu & Mandelbaum
  2015 per-M*-bin ``wp``/ΔΣ, Lange 2025 DESI DR1 BGS/LRG — see
  :ref:`data_formats` for the folder convention), the [Comparat2025]_
  broad-band w(θ) for S1–S7, and the 15×100 eV energy-band w(θ) FITS
  sets for all seven volume-limited samples (the 16th per-sample FITS,
  the broad-band sum, is already represented by the broad-band entries).
* **observed_derived_fit (2)** — points evaluated from a *published*
  fitting function with published parameters: the [MadauDickinson2014]_
  SFRD (their Eq. 15, Salpeter IMF) and the ALFALFA HIMF Schechter fit
  [Jones2018ALFALFA]_.  The binned points behind the fits remain flagged
  for extraction.
* **simulated (26)** — the forward-model fiducial prediction with the
  *forecast* survey noise, dumped from the tier-2 production npz
  (90 parameters, full grid) or the tier-3 smoke npz (reduced grid — to
  be refreshed from the tier-3 production npz once that run completes).
  These are stand-ins: ``y_err`` is the Stage-IV noise model, **not**
  current-survey errors, and every one names the published table that
  should replace it (e.g. the [Aird2015]_ electronic XLF tables, the
  KiDS-Legacy band powers [Wright2025]_, the [Kulkarni2019]_ QLF
  tables).
* **placeholder (10)** — no local stand-in is meaningful (data outside
  the model grid such as the z > 4 UV LFs, scalar relations such as the
  M_BH census, or external catalogues such as the eRASS1 cluster
  products); reference + description + extraction hint only.

Operator worklist
~~~~~~~~~~~~~~~~~

**38 files carry** ``needs_operator_extraction: true`` (all simulated
and placeholder entries plus the two derived fits).  ``index.json``
lists them::

    python -c "
    import json
    idx = json.load(open('index.json'))
    for k, v in idx.items():
        if v['needs_operator_extraction']:
            print(v['provenance'].ljust(22), k)"

When a published table has been extracted, replace the entry *in place*
(same file name), set ``provenance.type = 'observed'`` with the
``extraction_method``, record the source table/figure in
``provenance.origin``, and clear the flag.  One curated source is an
intentional empty placeholder: van Uitert 2016 measures only lensing, so
its ``wp`` CSV documents the GAMA clustering alternatives in its header
instead of data.

Relation to the other pages
---------------------------

:doc:`sensitivity_fisher` establishes *where the information lives* under
flat errors; :doc:`stage4_forecast` prices it with Stage-IV noise;
:doc:`tier2_forecast` / :doc:`tier3_forecast` free the full parameter
vector.  This page grounds all four: the same data vector, restricted to
the rows published today at their measured S/N, is the fit that can be run
*now* — and the missing-data table is the schedule on which the forecasts
become fits.
