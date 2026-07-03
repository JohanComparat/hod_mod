Tier-3 forecast: multi-wavelength maps, band LFs, z < 2, M* > 10⁹
==================================================================

The :doc:`tier-2 study <tier2_forecast>` freed every parameter of the
X-ray/tSZ/lensing forward model on a z < 1, M* > 10¹⁰ galaxy grid.  The
tier-3 study extends that design to the full multi-wavelength survey
landscape of the coming decade:

    **When the same halo-model chain also predicts the radio and infrared
    sky, the UV/optical/NIR luminosity functions and the cosmic
    star-formation history — over 0 < z < 2 and down to M* = 10⁹ — what do
    the combined maps and counts teach us about cosmology and about every
    piece of the galaxy–halo connection?**

Twelve SED-calibration parameters join the vector (**102 in total**,
:data:`~hod_mod.forecast.forward_jax.TIER3_EXTENSION`); nothing else is
fixed.  All twelve feed only new observables, so the fiducial predictions of
every tier-1/2 observable are bit-identical (the append-only house rule).

Reproduce with::

    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier3_forecast \
        --jobs 8
    # fast end-to-end check (2x2 cells, tiny grids, ~4 min):
    JAX_PLATFORMS=cpu python -m hod_mod.scripts.forecasts.run_tier3_forecast --smoke

The extended sample grid
------------------------

* **Coarse exploratory grid**: Δz = 0.2 shells over 0 < z < 2 × 0.2-dex
  volume-limited M* bins over 9.0 ≤ log10 M* ≤ 11.6 (130 cells), each split
  into star-forming and quiescent samples (260 cell blocks).  The per-block
  caches are incremental, so the grid can be refined later without
  recomputation of unchanged blocks.
* **Mass-grid floor**: cells integrate haloes from log10 M_h = 8.5 (the
  ``log10m_min`` :class:`~hod_mod.forecast.forward_jax.ForwardModel` argument;
  the tier-1/2 default of 10.0 is bit-identical) — M_h(M* = 10⁹) ≈ 2.4×10¹⁰
  sits well inside the grid, and the (9.0, 9.2) bin abundance converges to
  <10⁻³ at ``n_m = 256``.
* **Two-tier completeness**: the wide spectroscopic survey (f_sky = 0.5)
  carries a stellar-mass limit log10 M*_lim(z) = 9.0 + 1.0·z (the
  magnitude-limited trend); cells below it fall back to a deep field
  (f_sky = 0.004, complete at 10⁹ at all z) with correspondingly larger
  noise, and cells complete in neither tier are not built (reported as
  ``skipped_cells``).  AGN clustering samples extend to z = 1.9
  (``agn_z_centers``).

The twelve SED calibrations
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 12 12 60

   * - Parameter
     - Fiducial
     - Prior
     - Meaning
   * - ``l14_sfr``
     - 28.20
     - 0.20
     - log10 L_ν(1.4 GHz)/SFR [erg s⁻¹ Hz⁻¹ / M_⊙ yr⁻¹], the radio–FIR
       calibration [Murphy2011]_
   * - ``alpha_syn``
     - 0.70
     - 0.15
     - synchrotron spectral index; scales the SF and jet emission across the
       radio bands (FP cores stay flat)
   * - ``lir_sfr``
     - 43.41
     - 0.20
     - log10 L_IR/SFR (the Kennicutt–Murphy total-IR calibration
       [KennicuttEvans2012]_)
   * - ``bir_color``
     - 0.0
     - 0.5
     - dust-SED color tilt across the IR bands (inert at the 4.9 μm anchor)
   * - ``ml_nir``
     - 33.20
     - 0.30
     - log10 νL_ν(3.4 μm)/M* — the stellar M/L behind the NIR LF and the
       IR-map stellar continuum
   * - ``ml_opt``
     - 33.10
     - 0.30
     - log10 νL_ν(r)/M* of star-forming centrals
   * - ``dopt_q``
     - −0.20
     - 0.30
     - quiescent r-band offset [dex] (the optical LF is the SF/Q mixture)
   * - ``luv_norm``
     - 42.65
     - 0.30
     - log10 νL_ν(1500 Å)/SFR including the mean attenuation
   * - ``tau_uv_mslope``
     - 0.0
     - 0.3
     - UV-attenuation slope versus log M* (tilts the UV LF)
   * - ``lha_norm``
     - 41.27
     - 0.20
     - log10 L_Hα/SFR ([Kennicutt1998]_; the ``half`` LF is the exact
       ``oiilf`` kernel clone)
   * - ``agn_bc_uv``
     - −0.62
     - 0.30
     - AGN bolometric correction to 1450 Å [Runnoe2012]_
   * - ``agn_bc_opt``
     - −0.72
     - 0.30
     - AGN bolometric correction to 4400 Å [Runnoe2012]_

New observables
---------------

**Radio/IR intensity maps** (``radio_map_bands`` = 0.95, 1.4, 3.0 GHz —
SKA-like [Bacon2018]_; ``ir_map_bands`` = 3.4, 4.9, 12 μm — WISE/SPHEREx-like
[Wright2010]_ [Dore2014]_).  The per-halo central νL_ν fields combine

* SF synchrotron: :math:`10^{\mathrm{l14\_sfr}}\,\nu_{1.4}\,
  \langle\mathrm{SFR}\rangle(M)\,(\nu/1.4)^{1-\alpha_{\rm syn}}` with the
  ⟨SFR⟩(M) lognormal mean built from the main sequence, the SHMR and the
  ZM16 quenched fraction (the ``_oiilf`` width algebra);
* fundamental-plane cores (the ``_rlf`` FP moments, flat spectrum) and
  radio-loud jets (5 GHz anchored, ``alpha_syn``-scaled);
* IR dust (∝ SFR, color-tilted), stellar continuum (∝ M*·Υ_NIR) and the AGN
  torus (the ``_ilf`` chain — **no** f_abs: IR is obscuration-robust).

Observables: ``cl_gR``/``cl_gI`` per cell (galaxy × map, band-major),
``cl_RR``/``cl_II`` (band autos), ``cl_aR``/``cl_aI`` (AGN × map, L_X-bin ×
band) and ``cl_ag`` (AGN × galaxy) per shell.  Exact gates: the SF-only band
ratio is :math:`(\nu/1.4)^{1-\alpha_{\rm syn}}` to machine precision;
``∂ln C^{gR}/∂l14_sfr = ln 10`` (cross) and ``2 ln 10`` (auto);
``alpha_syn`` is exactly inert at the 1.4 GHz anchor.

**Galaxy band LFs + AGN UV/opt LFs** (per shell, via one ``_lf_lognormal``
kernel):  ``uvlf`` (SF, attenuation-tilted), ``optlf`` (SF/Q mixture —
collapses exactly to a single lognormal at ``dopt_q = 0``), ``nirlf`` (all
centrals, pure SHMR probe), ``half`` (≡ ``oiilf`` at matched normalisation),
and the type-1 AGN LFs ``qlf_uv``/``qlf_opt`` = the ``_ilf`` kernel ×
(1 − f_abs) — completing the cross-band obscuration system
(UV/opt ∝ 1−f_abs, IR ∝ 1, soft X-ray = the N_H transmission mixture;
``∂Φ/∂f_abs = −Φ/(1−f_abs)`` exactly).

**SFRD(z)**: a per-shell wide-M* (10⁹–10¹²) ``sfrd`` block — the
Madau–Dickinson measurement [MadauDickinson2014]_ at 12 % per shell.

**The four extras**:

* ``cl_yy`` — tSZ auto-spectrum (the pressure field squared;
  ``∂ln C_yy/∂P₀ = 2/P₀`` exactly);
* ``cl_HIHI`` — 21 cm auto-power (the HI field squared; ∝ M₀² exactly);
* ``ncl`` — X-ray cluster counts above max(L_lim(z), 10⁴²) erg/s through the
  free L_X–M relation with a fixed 0.25 dex selection scatter: the classic
  dn/dz probe, cosmology through dn/dM;
* ``ds_agn`` — AGN galaxy–galaxy lensing ΔΣ per L_X bin (the central-AGN
  tracer against the baryon-split matter field), directly weighing the
  L_X-selected host haloes.

Survey noise
------------

Physical where the tier-2 machinery applies (pair counts, shape noise, CXB
photon statistics, Poisson LFs with L_lim(z) completeness flags); the radio
and IR maps use the calibrated effective ``(rn, an)`` recipe
(:class:`~hod_mod.forecast.noise.SKASurvey`,
:class:`~hod_mod.forecast.noise.IRMapSurvey` — the tSZ / 21 cm IM precedent;
a physical T_sys/confusion model is the documented upgrade).  Band LFs get
:class:`~hod_mod.forecast.noise.BandLFSurvey` footprints with νL_ν flux
limits; the AGN crosses build their Knox noise from the fiducial AGN auto
(``cl_aa_fiducial``) plus shot noise.

Parallel execution
------------------

``--jobs N`` fills the per-block Jacobian cache with N spawned workers
(:meth:`~hod_mod.forecast.tier2.Tier2Forecast.precompute_blocks`): each
worker rebuilds the forecast from its resolved constructor spec, computes
blocks into atomic tempfiles, and the parent assembles serially from the
cache.  The parallel == serial invariant is byte-exact (tested in a fresh
interpreter; the workers inherit the parent's x64 mode through the
``JAX_ENABLE_X64`` environment variable so that module-level constants are
built in the right precision).

Verification
------------

``tests/test_forecast_tier3.py`` gates every feature with an exact identity
(band ratios, chain rules, kernel clones, mixture additivity, shift
invariance, mass-grid convergence, two-tier completeness assignment,
parallel == serial, cache round-trip), and the tier-2 regression suites run
unchanged through the new extension hooks.
