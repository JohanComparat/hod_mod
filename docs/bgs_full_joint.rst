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
shared halo-model engine (CAMB :math:`P(k)` + Tinker08 HMF + Zu & Mandelbaum
2015 occupation):

* **Galaxies (ZM15).**  :math:`n_\mathrm{gal}` (SMF integral), projected
  clustering :math:`w_p(r_p)`, and excess surface density
  :math:`\Delta\Sigma(r_p)` for DES/HSC/KIDS (:math:`r_p>2\,\mathrm{Mpc}/h`, with
  a free central point mass).
* **Hot gas (X-ray).**  The 15 narrow 0.1-keV bands (0.5–2 keV) and the broad
  band of the galaxy × soft-X-ray cross-correlation (Comparat 2025), via the
  precomputed transfer-grid band model of :doc:`xray_joint_fit`.
* **AGN (Powell).**  The X-ray luminosity function :math:`\phi(L_X)` at
  :math:`z=0.1` and :math:`0.4` (Roster 2026) and the AGN halo bias
  :math:`b(L_X)`, forward-modelled by
  :class:`~hod_mod.agn.powell.PowellAGNModel`.

**Two steps.**  The galaxy sector is expensive, so the ZM15 parameters are
handled two ways (each has its own OAR job):

* **Step 2 — fixed ZM15** (this page).  The 13 ZM15 parameters are held at the
  :doc:`mass-bin posterior <bgs_zm15_joint_mcmc>` median, so the galaxy
  predictions that do not depend on the point mass are precomputed **once**; each
  likelihood evaluation is then :math:`\sim0.01\,\mathrm{s}` and the MCMC is fast.
  **14 free parameters.**
* **Step 3 — free ZM15**.  The 13 ZM15 parameters are additionally free with
  Gaussian priors from the mass-bin posterior (mean = median,
  :math:`\sigma=(p_{84}-p_{16})/2`).  **27 free parameters**; the galaxy sector is
  recomputed every evaluation, so this is the heavier run.

**Free parameters (Step 2, 14).**

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Parameter
     - Prior
     - Role
   * - ``log10_M_star_cen``
     - :math:`\mathcal N(11.3, 0.3)`
     - ESD central point mass (:math:`\Delta\Sigma` inner boundary)
   * - ``lx_norm`` / ``lx_slope``
     - flat / :math:`\mathcal N`
     - :math:`L_X`–:math:`M_{500c}` relation (normalisation, slope)
   * - ``kt_norm`` / ``kt_slope``
     - flat / :math:`\mathcal N`
     - :math:`k_BT`–:math:`M_{500c}` relation
   * - ``p2`` / ``r_max``
     - flat
     - hot-gas density-profile shape (transfer grid)
   * - ``log10DC``
     - flat
     - AGN duty cycle
   * - ``z_metal``
     - :math:`\mathcal N`
     - gas metallicity
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

   # 2. submit the fixed-ZM15 job (MAP then MCMC; a re-submit resumes the chain)
   oarsub --project your-oar-project -S ./oarsub/fit_bgs_full_joint_fixedzm15_mcmc.sh

   # 3. pull the run back, then plot (MAP + posterior-median overlays; --docs
   #    writes the figures embedded below)
   rsync -avz DAHU:data/hod_mod_results/bgs_full_joint_fixedzm15/ \
       $HOD_MOD_RESULTS/bgs_full_joint_fixedzm15/
   JAX_PLATFORMS=cpu python -m hod_mod.scripts.fitting.plot_bgs_full_joint --docs

Data vs. MAP / posterior-median model
-------------------------------------

For every fitted observable, the figure shows the data with error bars, the
**MAP** model (green) and the **posterior-median** model (blue dashed).

.. figure:: _images/bgs_full_joint_fixedzm15__observables.png
   :width: 100%
   :alt: Data vs MAP and posterior-median model for wp, ESD, XLF, AGN bias and the broad X-ray band.

   Galaxy, AGN and broad-X-ray observables.  ``wp`` and ``ESD`` (DES/HSC/KIDS)
   constrain the galaxy–halo connection (ZM15 fixed) and the central point mass;
   ``XLF`` (:math:`z=0.1,\,0.4`) and ``AGN bias`` constrain the Powell AGN sector;
   the broad :math:`w(\theta)` anchors the absolute hot-gas amplitude.

.. figure:: _images/bgs_full_joint_fixedzm15__xray_bands.png
   :width: 100%
   :alt: The 15 narrow 0.5-2 keV band w(theta) panels, data vs MAP and median models.

   The 15 narrow 0.1-keV bands of the galaxy × soft-X-ray cross-correlation
   (0.5–2.0 keV).  The energy dependence of :math:`w(\theta)` encodes the gas
   temperature (via the band-resolved cooling function) and the AGN contribution;
   the 8 X-ray parameters are fit jointly across all bands.

Parameter posteriors
--------------------

.. figure:: _images/bgs_full_joint_fixedzm15__corner.png
   :width: 95%
   :alt: Corner plot of the 14 free parameters with the MAP overlaid in green.

   The 14-parameter posterior (green lines = MAP).  The X-ray relation
   parameters, the AGN ERDF parameters and the ESD point mass are recovered
   jointly; the ``lx_norm``–``log10DC`` and ERDF slope directions show the
   expected gas-vs-AGN and shape degeneracies.

Where the outputs are stored
----------------------------

The job writes into ``$HOD_MOD_RESULTS/bgs_full_joint_fixedzm15/`` (the
all-params Step-3 run uses ``bgs_full_joint_allparams/``):

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Content
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

The three figures above are produced by
:mod:`hod_mod.scripts.fitting.plot_bgs_full_joint`; ``--docs`` writes them into
``docs/_images/bgs_full_joint_fixedzm15__{observables,xray_bands,corner}.png``.
