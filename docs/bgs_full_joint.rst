BGS S1 full-model joint fit: galaxies + hot gas + AGN
=====================================================

This page documents the **full-model joint fit** for the BGS :math:`M_\star>10`
(``S1``) sample: a single Gaussian likelihood that ties together the galaxy
clustering/lensing sector, the hot-gas soft-X-ray cross sector, and the AGN
X-ray sector, and samples it with a resumable MCMC on GRICAD/dahu.  It sits on
top of the galaxy-only fit of :doc:`bgs_zm15_joint_mcmc` — that fit's posterior
*anchors* the galaxy sector here — and folds in the X-ray band model of
:doc:`xray_joint_fit` and the Powell AGN model (:doc:`sensitivity_fisher`).

.. contents::
   :local:
   :depth: 2

----

Setup
-----

**Model sectors.**  One log-likelihood assembles three forward models on the
shared halo-model engine (linear :math:`P(k)` from
:func:`~hod_mod.core.power_spectrum.default_pk_linear` — the fits shown here
ran on the CAMB backend; since 0.3.1 the package default is the CosmoPower-JAX
emulator, see :doc:`cosmology` — + Tinker08 HMF + Zu & Mandelbaum
2015 occupation):

* **Galaxies (ZM15).**  :math:`n_\mathrm{gal}` (SMF integral), projected
  clustering :math:`w_p(r_p)`, and excess surface density
  :math:`\Delta\Sigma(r_p)` (with a free central point mass).
* **Hot gas (X-ray).**  The 15 narrow 0.1-keV bands (0.5–2 keV) and the broad
  band of the galaxy × soft-X-ray cross-correlation (Comparat 2025), via the
  precomputed transfer-grid band model of :doc:`xray_joint_fit`.
* **AGN (Powell).**  The X-ray luminosity function :math:`\phi(L_X)` (Roster
  2026) and the AGN halo bias :math:`b(L_X)`, forward-modelled by
  :class:`~hod_mod.agn.powell.PowellAGNModel`.

**Data selection.**  The fitted data are trimmed to the most-robust points (all
selections are ``fit_bgs_full_joint`` CLI options, recorded per run in
``fit_config.json``):

* lensing: only the **HSC** :math:`\Delta\Sigma` on :math:`2<r_p<8\,\mathrm{Mpc}/h`
  (DES and KIDS dropped);
* XLF: only the :math:`z=0.1` slice with :math:`\log_{10}L_X>41`;
* AGN halo bias: only the **Comparat+2023** and **Krumpe+2015** points;
* X-ray: the broad band **and** all 15 narrow bands are kept.

**Two steps.**  The galaxy sector is expensive, so the ZM15 parameters are
handled two ways (each has its own OAR job):

* **Step 2 — fixed ZM15** (this page).  The 13 ZM15 parameters are held at the
  :doc:`mass-bin posterior <bgs_zm15_joint_mcmc>` median, so the galaxy
  predictions that do not depend on the point mass are precomputed **once**; each
  likelihood evaluation is then :math:`\sim0.01\,\mathrm{s}` and the MCMC is fast.
  **15 free parameters.**
* **Step 3 — free ZM15**.  The 13 ZM15 parameters are additionally free with
  Gaussian priors from the mass-bin posterior (mean = median,
  :math:`\sigma=(p_{84}-p_{16})/2`).  **28 free parameters**; the galaxy sector is
  recomputed every evaluation, so this is the heavier run.

**Free parameters (Step 2, 15).**

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Parameter
     - Prior
     - Role
   * - ``log10_M_star_cen``
     - :math:`\mathcal N(11.3, 0.3)`
     - ESD central point mass (:math:`\Delta\Sigma` inner boundary)
   * - ``log10_ne03`` / ``beta_n``
     - flat
     - DPM electron-density normalisation at :math:`0.3\,r_{500c}` and its mass slope
   * - ``log10_p03`` / ``beta_P``
     - tight :math:`\mathcal N` (:math:`\sigma=0.05/0.04`)
     - DPM electron-pressure normalisation at :math:`0.3\,r_{500c}` and its mass
       slope.  :math:`L_X` and :math:`k_BT` are no longer free power laws: since
       the native-DPM re-base they are *derived* by integrating the profiles, so
       the scaling relations shown below are predictions, not fitted lines.
   * - ``p2`` / ``r_max``
     - flat
     - hot-gas density-profile shape (transfer grid)
   * - ``log10DC``
     - flat
     - AGN duty cycle
   * - ``z_metal``
     - :math:`\mathcal N`
     - gas metallicity
   * - ``agn_gamma``
     - :math:`\mathcal N(1.8, 0.3)`
     - AGN/continuum photon index — frees the band spectral shape (new)
   * - ``agn_mu_bh``
     - flat
     - :math:`M_{\rm BH}`–:math:`M_\star` normalisation
   * - ``agn_log10_lstar``
     - flat
     - ERDF break luminosity
   * - ``agn_delta1`` / ``agn_delta2``
     - flat
     - ERDF faint / bright slopes
   * - ``log10_ferdf``
     - flat
     - AGN / ERDF normalisation

**Sampler.**  A local gradient-free MAP (scipy Nelder-Mead) seeds a resumable
`emcee <https://emcee.readthedocs.io>`_ ensemble sampler with an HDF backend
(``chain.h5``, flushed every step), so a besteffort/idempotent OAR job continues
exactly where it left off.  The production Step-2 job uses 48 walkers,
500 burn-in + 2000 steps.

**Commands.**  Stage the inputs to dahu, submit, then make the figures locally:

.. code-block:: bash

   # 1. stage data + the ZM15 posterior + the apec cache to dahu (run locally)
   bash oarsub/rsync_data_to_dahu.sh

   # 2. submit the fixed-ZM15 job (MAP then MCMC; a re-submit resumes the chain).
   #    The job script applies the data selection + tight kT prior via CLI flags:
   #      --esd-surveys HSC --esd-rp-max 8.0 --xlf-z 0.1 --xlf-lx-min 41.0
   #      --agn-bias-refs Comparat23 Krumpe15 --kt-prior-sig 0.05 0.04
   oarsub --project your-oar-project -S ./oarsub/fit_bgs_full_joint_fixedzm15_mcmc.sh

   # 3. pull the run back, then plot (MAP + posterior-median overlays; --docs
   #    writes the figures embedded below).  The plotter reads fit_config.json so
   #    the plotted data match the fit's selection automatically.
   rsync -avz DAHU:data/hod_mod_results/bgs_full_joint_fixedzm15/ \
       $HOD_MOD_RESULTS/bgs_full_joint_fixedzm15/
   JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.plot_bgs_full_joint --docs

Data vs. MAP / posterior-median model
-------------------------------------

For every fitted observable, the figure shows the data with error bars, the
**MAP** model (green) and the **posterior-median** model (blue dashed), each
panel decomposed into its physical components.

.. figure:: _images/bgs_full_joint_fixedzm15__observables.png
   :width: 100%
   :alt: Data vs MAP and posterior-median model for wp, ESD, XLF, AGN bias and the broad X-ray band.

   Galaxy, AGN and broad-X-ray observables.  ``wp`` and the HSC ``ESD``
   (:math:`2<r_p<8`) are split into **1-halo / 2-halo** (ESD also the central
   point mass); ``XLF`` (:math:`z=0.1`, :math:`\log_{10}L_X>41`) and ``AGN bias``
   (Comparat+2023, Krumpe+2015) constrain the Powell AGN sector — the bias panel
   shows the model :math:`\pm3\sigma` response in the host-mass normalisation
   :math:`\mu_{\rm BH}`; the broad :math:`w(\theta)` (log-log) is split into
   **gas** and **AGN**.

.. figure:: _images/bgs_full_joint_fixedzm15__xray_bands.png
   :width: 100%
   :alt: The 15 narrow 0.5-2 keV band w(theta) panels in log-log, decomposed into gas/AGN/total.

   The 15 narrow 0.1-keV bands of the galaxy × soft-X-ray cross-correlation
   (0.5–2.0 keV), in **log-log** with each panel decomposed into **gas / AGN /
   total**.  The energy dependence of :math:`w(\theta)` is a gas-temperature
   tracer (via the band-resolved cooling function); the AGN dominates
   :math:`\theta\lesssim20''` and the gas the outer profile.  The 9 X-ray
   parameters are fit jointly across all bands.

.. admonition:: Correction (2026-08-21) — the gas sector of this fit is invalid
   :class: warning

   The X-ray/gas numbers below were written from the v0.3 posterior in good
   faith.  Three defects in :mod:`hod_mod.fitting.full_joint`, found while
   preparing v0.4, mean that posterior does not constrain the gas sector:

   * **The informative gas prior was never applied.**  ``pri_sig`` is diagonal
     by construction and carries ``inf`` for the four native-DPM gas
     parameters, so the full-covariance prior — whose information lives almost
     entirely in its correlations (ρ from −0.95 to +0.84) — contributed
     nothing.  The gas sector ran on its uniform bounds alone.  Where this page
     says the intervals are *prior-dominated*, the sharper and worse truth is
     that they are **bounds-dominated**.
   * **The gas seed was the wrong distribution's centre** — the
     scaling-relation prior centre used where the induced-prior centre belongs,
     474 units of prior :math:`\chi^2` away and clipped onto three bounds,
     which Nelder-Mead did not escape.
   * **``--kt-prior-sig`` pinned the wrong parameters.**  After the native-DPM
     re-base the two slots it wrote onto are ``log10_p03`` and ``beta_P``, not
     ``kt_norm``/``kt_slope``.  It is replaced in v0.4 by ``--gas-prior-widen``,
     a scalar inflation of the whole 4×4.

   The visible symptom is an **inverted** :math:`k_BT`–:math:`M` relation:
   :math:`\beta_P - \beta_n = -0.41` at the posterior median against a prior
   centred on :math:`+0.6`.  The galaxy and AGN sectors are unaffected.  This
   page will be rewritten once the fit is re-run with the corrected prior and
   seed; until then treat every gas-sector number and the
   *Hot-gas scaling relations* figures below as withdrawn.

Goodness of fit (campaign ``VTAG=v0.3``)
----------------------------------------

.. list-table:: Step-2 MAP, ``bgs_full_joint_fixedzm15_v0.3``
   :header-rows: 1
   :widths: 34 22 22 22

   * - Sector
     - :math:`\chi^2`
     - share
     - note
   * - galaxies (:math:`n_{\rm gal}, w_p, \Delta\Sigma`)
     - 2.11
     - 0.002 %
     - ZM15 held fixed
   * - AGN (XLF + bias)
     - 53.55
     - 0.05 %
     - —
   * - X-ray (broad + 15 bands)
     - 97 985.73
     - **99.9 %**
     - carries the entire misfit
   * - **total**
     - **98 041.40** / 526 dof
     - —
     - :math:`\chi^2/\mathrm{dof} = 186.4`

Two things follow, and neither is cosmetic.

* **The X-ray band sector is not fitted.**  The galaxy and AGN sectors are
  essentially perfect (:math:`\chi^2 = 2.1` and :math:`53.6`); all 98 000 units of
  :math:`\chi^2` sit in the X-ray bands.  A :math:`\chi^2/\mathrm{dof}` of 186 is
  not a measurement — the band model is missing something, and the posterior
  below should be read as "where the sampler went", not as a constraint.
* **The MAP and the posterior disagree.**  Re-evaluating at the posterior median
  gives :math:`\chi^2/\mathrm{dof} = 2805` (X-ray 1 475 528), a factor 15 worse
  than the MAP.  The chain is not sampling a basin around the MAP; several
  parameters sit on prior bounds (``p2`` at its 0.1 floor while the MAP is at
  0.637; ``r_max``, ``agn_delta2`` and ``log10_M_star_cen`` at their ceilings;
  ``agn_mu_bh`` at its floor).  ``agn_gamma`` is one of them: the chain rails on
  its :math:`\Gamma=1.2` floor (median 1.224) while the MAP sits at 1.448, so
  the ":math:`\Gamma` settles hard at :math:`\approx1.5`" reading below holds at
  the MAP but not in the posterior.  Quote the MAP, and treat the 68 % intervals
  in ``posterior_summary.json`` as prior-dominated.

Why the AGN photon index is free
--------------------------------

A per-band spectral diagnostic (``bandmodel_diagnostic.py``, staged with the run
products) shows the observed cross-correlation band spectrum is **flat and
featureless** — it lacks the Fe-L line peak at 0.7–1.2 keV that :math:`\sim1`-keV
thermal gas would imprint, and it is nearly identical at all angular scales.
Because APEC only reproduces a flat spectrum at :math:`k_BT\gtrsim8\,\mathrm{keV}`,
a thermal-gas-only model is driven to an unphysically hot :math:`k_BT`, and even a
tight :math:`k_BT`-:math:`M` prior cannot hold it (the band likelihood overrides
it at :math:`\sim3.6\sigma`).

The fix is to give the **continuum** its own spectral degree of freedom: the AGN
photon index ``agn_gamma`` (fixed at :math:`\Gamma=1.8`) is freed.  The AGN then
absorbs the flat spectral shape (:math:`\Gamma` settles hard, :math:`\approx1.5`),
the band :math:`\chi^2` drops by :math:`\sim210` — *these two numbers come from the
2026-07 free-vs-fixed diagnostic pair, not from the v0.3 run archived here, which
stores only the* :math:`\Gamma`-*free fit* — and with the tight
:math:`k_BT`-prior the **posterior** :math:`k_BT`-:math:`M` relation tracks the
resolved-cluster scaling instead of soaring an order of magnitude above it.  A
residual remains: even :math:`\Gamma`-free, the bands cannot fully pin the
:math:`k_BT` *normalisation* (the density–temperature–continuum degeneracy is not
fully broken by soft-X-ray emission alone) — the tSZ (pressure-weighted) cross is
the natural next constraint.

Hot-gas scaling relations and radial profiles
---------------------------------------------

The X-ray sector is parametrised by explicit :math:`L_X`–:math:`M_{500c}` and
:math:`k_BT`–:math:`M_{500c}` power laws plus a density-profile shape
(:math:`p_2`, :math:`r_{\max}`).  The top row shows those recovered scaling
relations against cluster/group literature; the bottom row shows the DPM radial
profiles whose density normalisation is calibrated to the fit's own
:math:`L_X`–:math:`M` relation.

.. figure:: _images/bgs_full_joint_fixedzm15__gas.png
   :width: 100%
   :alt: Hot-gas L_X-M, kT-M and L_X-kT scaling relations and n_e, T, P_e radial profiles.

   **Top:** :math:`L_X`–:math:`M_{500c}`, :math:`k_BT`–:math:`M_{500c}` and
   :math:`L_X`–:math:`k_BT` — the MAP (green) and posterior-median (blue) fit
   relations with the 68 % posterior band, over-plotted with Lovisari+2020 and
   Bulbul+2018 cluster/group samples and the Lovisari+2020 / Comparat+2025
   (GAS.py) relations.  With ``agn_gamma`` free (previous section) the posterior
   :math:`k_BT`–:math:`M_{500c}` **tracks the Lovisari/Bulbul cluster cloud**
   rather than sitting an order of magnitude above it; the slope is physical.
   **Bottom:** electron density :math:`n_e(r)`, temperature :math:`T(r)=P_e/n_e`
   and electron pressure :math:`P_e(r)` at
   :math:`M_{200}=10^{13,14,15}\,M_\odot/h` (dotted = Arnaud+2010 pressure), with
   :math:`n_e` calibrated to the fit's :math:`L_X`–:math:`M`.

AGN host stellar-mass function
------------------------------

A derived prediction: the stellar-mass function of AGN host galaxies above a
hard-X-ray luminosity threshold, built from the Powell central-AGN occupation
:math:`N_{\rm AGN}(>L_X\,|\,M_{\rm halo})` weighted by the HMF and mapped into
stellar mass through the Girelli+2020 SHMR (with scatter).

.. figure:: _images/bgs_full_joint_fixedzm15__agn_host_smf.png
   :width: 75%
   :alt: Predicted stellar-mass function of AGN host galaxies above L_X thresholds.

   AGN host stellar-mass function for two hard-band luminosity thresholds
   (model, centrals only).  Intended for comparison with Bongiorno+2016 — those
   data are not yet staged locally, so this is currently the model prediction.

Parameter posteriors
--------------------

.. figure:: _images/bgs_full_joint_fixedzm15__corner.png
   :width: 95%
   :alt: Corner plot of the 15 free parameters with the MAP overlaid in green.

   The 15-parameter posterior (green lines = MAP).  The X-ray relation
   parameters (incl. the free ``agn_gamma``), the AGN ERDF parameters and the ESD
   point mass are recovered jointly; the ``lx_norm``–``log10DC`` and
   ``agn_gamma``–``kt_norm`` directions show the expected gas-vs-AGN /
   continuum-vs-temperature degeneracies.

Where the outputs are stored
----------------------------

The job writes into ``$HOD_MOD_RESULTS/bgs_full_joint_fixedzm15/`` (the
all-params Step-3 run uses ``bgs_full_joint_allparams/``):

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Content
   * - ``fit_config.json``
     - the exact data selection (surveys, cuts, references) — read back by the
       plotter so the figures match the fit
   * - ``map_result.json``
     - MAP parameters, :math:`\chi^2`/dof, per-sector breakdown, optimiser status
       (written as soon as MAP finishes; its presence makes a re-submit skip MAP)
   * - ``chain.h5``
     - the resumable emcee HDF backend (continue by re-submitting)
   * - ``flatchain.npz``
     - the flattened post-burn-in chain (``flatchain``, ``param_names``) — the
       plotter's input
   * - ``posterior_summary.json``
     - per-parameter median and :math:`\pm1\sigma` (16/50/84)

The figures above are produced by
:mod:`hod_mod.scripts.fitting.plot_bgs_full_joint`; ``--docs`` writes them into
``docs/_images/bgs_full_joint_fixedzm15__{observables,xray_bands,gas,agn_host_smf,corner}.png``.
