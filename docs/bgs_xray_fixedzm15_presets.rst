Family C — fixed-ZM15 X-ray :math:`w(\theta)` preset fits
=========================================================

This page reports the **Comparat+2025 fixed-ZM15 X-ray preset fits**, the family
the re-run campaigns call *Family C*.  Each preset holds the galaxy sector at the
frozen ZM15 reference MAP and fits the galaxy × soft-X-ray angular
cross-correlation :math:`w(\theta)` alone, varying a different slice of the gas
and AGN model.  The model itself is documented in
:doc:`benchmark_comparat2025`; the preset definitions and the frozen ZM15
parameters live in :doc:`hod_zumandelbaum2015`; the campaign machinery is
:doc:`oarsub_campaign`.

.. contents::
   :local:
   :depth: 2

----

Setup
-----

**Five presets, five jobs.**  Each is one dedicated OAR script calling
:mod:`hod_mod.scripts.fitting.fit_comparat2025`:

.. code-block:: bash

   VTAG=v0.4 ./oarsub/submit_campaign.sh <PROJECT> comparat2025
   VTAG=v0.4 ./oarsub/submit_campaign.sh <PROJECT> comparat2025_ecf   # --ecf twins

Each script takes the campaign tag as its first positional argument and the
literal ``ecf`` as an optional second, writing to
``$HOD_MOD_RESULTS/fits/comparat2025_fixedZM15_<preset>[_ecf]_<VTAG>/``.  All
five run ``--sample S1 --fix-zm15 <ZM15_JSON> --mode map``, are MAP
(``scipy`` L-BFGS-B) and are **not resumable** — the walltime must cover the
whole optimisation.

**Data.**  A single sample, S1 (:math:`\log_{10}M_\star>10`), 31 :math:`w(\theta)`
points over :math:`8''<\theta<300''`, with a 5 % systematic floor
(``--f-sys 0.05``).  The SMF, :math:`\bar n_g`, :math:`w_p` and :math:`\Delta\Sigma`
panels that appear in the diagnostic figures are **fixed predictions, not fitted
data**, in this mode.

**What each preset frees.**  From ``_FREE_PRESETS`` and ``_PARAM_REGISTRY`` in
``fit_comparat2025.py``.  Parameters marked *rebuild* force a DPM profile or AGN
occupation rebuild on every likelihood evaluation, which is what makes these jobs
hours rather than minutes.

.. list-table::
   :header-rows: 1
   :widths: 16 8 10 46 20

   * - Preset
     - N
     - AGN model
     - Free parameters beyond the four base ones
     - Cost
   * - ``gas-shape``
     - 6
     - ``hod``
     - ``alpha_out_gas``, ``alpha_out_pressure``
     - rebuild
   * - ``gas-temp``
     - 8
     - ``hod``
     - + ``gamma_gas``, ``log10_P_03``
     - rebuild
   * - ``gas-full``
     - 14
     - ``hod``
     - + ``alpha_in_gas``, ``alpha_tr_gas``, ``log10_ne_03``, ``alpha_in_pressure``, ``alpha_tr_pressure``, ``Z_0``
     - rebuild
   * - ``agn-occ``
     - 8
     - ``hod`` (required)
     - ``f_inc``, ``log10mmin_agn``, ``sigma_logm_agn``, ``alpha_agn``
     - rebuild
   * - ``agn-lum``
     - 7
     - ``ham`` (required)
     - ``scatter_lx``, ``log10_A_kcorr``, ``log10_A_dc``
     - cheap

The four base parameters common to every preset are ``log10_A_gas``,
``beta_gas``, ``beta_pressure`` and ``log10_A_AGN``.  ``agn-lum`` and ``agn-occ``
are pinned to a specific ``--agn-model`` by ``_PRESET_REQUIRES_AGN`` and are
rejected at argument-parse time under any other.

Goodness of fit (campaign ``VTAG=v0.4``)
----------------------------------------

.. list-table:: S1 MAP, 31 :math:`w(\theta)` points, with the ``--ecf`` twin
   :header-rows: 1
   :widths: 16 8 8 14 12 14 12 16

   * - Preset
     - AGN
     - dof
     - :math:`\chi^2`
     - :math:`\chi^2/\mathrm{dof}`
     - :math:`\chi^2` (ecf)
     - /dof (ecf)
     - converged?
   * - ``gas-shape``
     - ``hod``
     - 25
     - 104.04
     - 4.161
     - 208.70
     - 8.348
     - yes / yes
   * - ``gas-temp``
     - ``hod``
     - 23
     - 100.93
     - 4.388
     - 207.98
     - 9.043
     - yes / **no**
   * - ``gas-full``
     - ``hod``
     - 17
     - 102.38
     - 6.022
     - 227.45
     - 13.379
     - yes / **no**
   * - ``agn-occ``
     - ``hod``
     - 23
     - 131.57
     - 5.720
     - 226.79
     - 9.861
     - **no** / **no**
   * - ``agn-lum``
     - ``ham``
     - 24
     - **92.28**
     - **3.845**
     - 226.79
     - 9.450
     - **no** / **no**

.. warning::

   The last column is not decoration.  L-BFGS-B reports ``success=False`` for
   **6 of the 10** runs, including both non-ECF AGN presets.  Every number on
   this page is "where the optimiser stopped", and for those six that is not the
   same as "the MAP".

Best non-ECF fit is ``agn-lum`` at :math:`\chi^2/\mathrm{dof}=3.845`; adding gas
profile freedom does not improve on it, and ``gas-full`` — the richest preset —
is the *worst* of the five.  No preset reaches :math:`\chi^2/\mathrm{dof}\sim1`:
the known residual shape mismatch (the gas model too flat in :math:`\theta`, the
AGN PSF template overshooting at :math:`8''`) is not addressed by any of them.

Stability across campaigns
--------------------------

Family C is almost completely insensitive to the two behaviour-changing releases
that forced the v0.3 and v0.31 campaigns — the 0.3.0 Hankel-transform fix and the
CAMB → CosmoPower-JAX default :math:`P(k)` swap:

.. list-table:: :math:`\chi^2/\mathrm{dof}`, non-ECF
   :header-rows: 1
   :widths: 22 20 20 20 18

   * - Preset
     - v0.3 (CAMB)
     - v0.31 (CosmoPower)
     - v0.4
     - spread
   * - ``gas-shape``
     - 4.086
     - 4.042
     - 4.161
     - 2.9 %
   * - ``gas-temp``
     - 4.441
     - 4.391
     - 4.388
     - 1.2 %
   * - ``gas-full``
     - 6.076
     - 6.020
     - 6.022
     - 0.9 %
   * - ``agn-occ``
     - 5.770
     - 5.721
     - 5.720
     - 0.9 %
   * - ``agn-lum``
     - 3.875
     - 3.845
     - 3.845
     - 0.8 %

That is expected — :math:`w(\theta)` at :math:`8''`–:math:`300''` is dominated by
the 1-halo term and the instrument response, neither of which the Hankel fix or
the linear :math:`P(k)` moves much — but it is worth stating, because it means
**Family C results from the three campaigns are directly comparable**.

The one exception is ``gas-shape``, whose *parameters* moved in v0.4 even though
its :math:`\chi^2` barely did: ``log10_A_gas`` :math:`-0.73 \to +0.09`,
``beta_pressure`` :math:`0.95 \to 1.18`, ``alpha_out_gas`` :math:`2.80 \to 2.63`.
A different point in a flat direction, i.e. a degeneracy, not a change in
physics.

.. _familyc-inert:

Which parameters actually constrain
-----------------------------------

.. warning::

   Most of the parameters these presets nominally free **never move**.  Their
   fitted values equal the ``_PARAM_REGISTRY`` seed to machine precision, in
   *all three* campaigns.

.. list-table:: Non-ECF v0.4 runs; "frozen" = fitted value equals the registry seed to <1e-9
   :header-rows: 1
   :widths: 16 10 10 46

   * - Preset
     - free
     - frozen
     - which
   * - ``gas-shape``
     - 6
     - 0
     - —
   * - ``gas-temp``
     - 8
     - 1
     - ``log10_P_03``
   * - ``gas-full``
     - 14
     - 2
     - ``log10_P_03``, ``log10_ne_03``
   * - ``agn-occ``
     - 8
     - **6**
     - ``f_inc``, ``log10mmin_agn``, ``sigma_logm_agn``, ``alpha_agn``, ``beta_gas``, ``beta_pressure``
   * - ``agn-lum``
     - 7
     - **5**
     - ``scatter_lx``, ``log10_A_kcorr``, ``log10_A_dc``, ``beta_gas``, ``beta_pressure``

So the two AGN presets fit **two parameters each** — the gas and AGN amplitudes —
and nothing else.  Every AGN-sector parameter they were built to constrain sits
at its starting value, and so do the two gas mass-slope tilts.  ``log10_A_kcorr``
and ``log10_A_dc`` additionally have their seed *on* their upper bound, so they
could only ever have moved downward.

The two gas normalisations that freeze (``log10_P_03``, ``log10_ne_03``) are
exactly the parameters the ZM15 page's *Cost and identifiability* note predicts
to be degenerate flat directions in a :math:`w(\theta)`-only fit — a normalisation
is not separable from the amplitude that multiplies it.

The two AGN presets freeze for **two different reasons**, both now measured.

``agn-lum`` — the parameters have no effect at all
   Perturbing ``scatter_lx``, ``log10_A_kcorr`` and ``log10_A_dc`` across their
   entire allowed ranges changes the predicted :math:`w(\theta)` by
   :math:`\max|\Delta| = 0` — exactly zero, in both the gas and the AGN leg.
   Under ``--agn-model ham`` the AGN template is
   :func:`_psf_template`, a **normalised** King PSF: ``_predict_shape`` does
   compute the AGN cross-power that these three parameters feed, and then
   discards it.  Nor could it be otherwise for a point source — a luminosity
   parameter can only rescale a normalised template, and that scaling is
   already free as :math:`\log_{10}A_\mathrm{AGN}`.  The preset is degenerate
   by construction, not merely unconstrained by the data.

``agn-occ`` — the parameters work, the optimiser does not
   The opposite case.  Perturbing the occupation parameters moves the AGN
   template substantially: ``f_inc`` 0.01 → 0.05 changes it by 244 %,
   ``sigma_logm_agn`` 0.8 → 1.2 by 66 %, ``log10mmin_agn`` 12.5 → 13.0 by 30 %
   (``alpha_agn`` is weak, 0.6 %).  The likelihood is not flat.  L-BFGS-B simply
   fails: it reports ``ABNORMAL`` termination after **189 function evaluations
   with zero successful iterations**, leaving every parameter — amplitudes
   included — at its starting value.  Rescaling the parameters to the unit cube
   does not help, so the cause is not the shared finite-difference step.

Both presets terminate the same way — ``ABNORMAL`` at ``nit=0``, with
``nfev=189`` (``agn-occ``) and ``nfev=168`` (``agn-lum``) — which is what a
line search does when the Hessian is singular or near-singular.  For
``agn-lum`` the cause is explicit: three exactly-flat directions.  For
``agn-occ`` it is near-degeneracy rather than exact — ``f_inc`` is *almost* a
pure rescaling of the AGN template (the ratio to the fiducial varies only
9–44 % across :math:`\theta`), so it is close to, but not identical with,
:math:`\log_{10}A_\mathrm{AGN}`.

Two remedies were tried and **rejected on measurement**, recorded here so they
are not retried blind:

* *Rescaling the parameters to the unit cube* (the textbook fix for a shared
  finite-difference step).  It left ``agn-occ`` bit-identical — still
  ``ABNORMAL``, ``nit=0``, ``nfev=189`` — and *regressed* ``gas-shape`` from
  4.161/converged to 4.86/``ABNORMAL``.  Reverted.
* *Powell* (derivative-free, so indifferent to conditioning).  It did not
  converge within a 10-minute budget on the cheapest AGN preset, against 2.5
  minutes for L-BFGS-B, so it could not be validated here.

**Read this as: ``agn-occ`` and ``agn-lum`` currently add no constraint over the
4-parameter ``all`` preset.**  Their :math:`\chi^2` differences reflect the AGN
*model* (``hod`` vs ``ham``), not the extra freedom.  ``agn-lum`` cannot be
repaired by a better optimiser at all — its parameters are degenerate by
construction; ``agn-occ`` plausibly can, but needs a reparametrisation that
separates ``f_inc`` from the AGN amplitude, which is a design change rather
than a tuning one.

The run now says so out loud: ``run_map`` reports the optimiser's message,
iteration and evaluation counts, and any objective failures, and records them in
``S1_map.json`` as ``opt_message`` / ``opt_nit`` / ``opt_nfev`` /
``obj_failures``.  Selecting a preset whose parameters cannot act under the
chosen ``--agn-model`` prints a warning naming them.

The ``--ecf`` variant
---------------------

``--ecf`` folds the validated eROSITA TM0 flux→count ECF into both legs and
anchors the remaining sample-independent geometry on S1, so that
``log10_A_gas``/``log10_A_AGN`` become :math:`O(1)` residuals instead of
absorbing the whole :math:`\sim10^8` model→counts chain
(:math:`\Lambda_{\rm eff}`, Mpc→cm, :math:`1/4\pi`, ECF, sr→arcsec²).

.. admonition:: The v0.4 ``--ecf`` runs are invalid — the AGN component was switched off
   :class: warning

   Every ``--ecf`` fit in the table above was made with **no AGN component at
   all**, and must not be quoted.

   The anchor is measured once on S1 by a least-squares solve for the two
   conversion constants.  As shipped in 0.4.0 that solve weighted by a bare
   :math:`1/|\sigma|`, omitting the ``f_sys`` floor the likelihood itself
   applies.  That hands the fit to the tiny-error large-:math:`\theta` bins,
   where the gas/AGN split is degenerate, and drives the AGN coefficient
   *negative*; a non-negativity box then clipped it to exactly ``0.0``.  Since
   the model applies it as ``A_AGN = 10**log10_A_AGN * c_agn``, a zero
   conversion removes the AGN term for **every** value of the amplitude.

   Three signatures follow, all visible above and in the stored results:

   * :math:`\chi^2/\mathrm{dof}` roughly doubles for every preset — gas alone
     must now fit the AGN-dominated :math:`\theta\lesssim20''` points.
   * ``agn-occ_ecf`` and ``agn-lum_ecf`` have **bit-identical**
     :math:`\chi^2 = 226.79257920358373` despite using different AGN models.
   * ``log10_A_AGN`` is reported as exactly :math:`-3.3139` by all four ``hod``
     presets.  This is **not** a bound — the range is :math:`(-5, 15)` — and not
     a coincidence: with a zero conversion the direction is exactly flat, so
     L-BFGS-B returns its **seed**, which is computed before any preset-specific
     parameter exists and is therefore identical across presets.

   **Fixed after the campaign.**  The anchor now uses the likelihood's own error
   definition (jackknife :math:`\oplus` the ``f_sys`` floor) and an
   unconstrained solve; a degenerate anchor raises instead of being cached; and
   the cache carries the measurement ``scheme``, so an anchor written the old way
   is re-measured rather than silently reused.  The MAP seed is also computed in
   residual units — it was in absolute units, so under ``--ecf`` it started ~12
   decades off and was clipped to a meaningless :math:`-3`.

   Verified end-to-end on the ``amps`` preset (S1, ``hod``), which is the same
   two amplitudes without the expensive profile rebuilds:

   .. list-table::
      :header-rows: 1
      :widths: 30 18 26 26

      * - run
        - :math:`\chi^2/\mathrm{dof}`
        - :math:`\log_{10}A_\mathrm{gas}`
        - :math:`\log_{10}A_\mathrm{AGN}`
      * - no ``--ecf``
        - 4.537
        - :math:`-0.359`
        - :math:`+8.410`
      * - ``--ecf``, **fixed**
        - **4.556**
        - **:math:`-0.0000`**
        - **:math:`-0.0000`**
      * - ``--ecf``, as shipped
        - 8.348
        - :math:`+4.465`
        - :math:`-3.314`

   The re-measured anchor is :math:`c_{\rm gas}=7.819\times10^{-13}`,
   :math:`c_{\rm AGN}=+4.854\times10^{-4}` — both strictly positive.  ``--ecf``
   is a reparametrisation, so agreeing with the non-ECF :math:`\chi^2/\mathrm{dof}`
   to 0.4 % *is* the correctness criterion, and the amplitudes now sit at 0
   instead of absorbing the conversion chain.  (The anchor is per AGN model; the
   ``ham`` value differs.)

   The five ``--ecf`` presets still need re-running before this page can report
   their results.

Where the outputs are stored
----------------------------

``$HOD_MOD_RESULTS/fits/comparat2025_fixedZM15_<preset>[_ecf]_<VTAG>/``:

.. list-table::
   :widths: 30 70

   * - ``S1_map.json``
     - MAP parameters, :math:`\chi^2`, dof, the scale cuts, the AGN model, and
       the 13 frozen ZM15 parameters.  The only machine-readable output.
   * - ``S1_bestfit.pdf``
     - :math:`w(\theta)` with the gas/AGN decomposition and residual ratio, plus
       the (unfitted) :math:`w_p` prediction.
   * - ``S1_diagnostics.pdf``
     - SMF, :math:`\bar n_g`, SHMR, :math:`\Delta\Sigma` and the X-ray
       auto-power — all fixed predictions in this mode.
   * - ``all_samples_bestfit.pdf``
     - Multi-sample overview; degenerates to a single panel here, since Family C
       fits S1 only.

.. note::

   **No figures on this page.**  These fits emit PDF, while
   ``collect_and_plot.sh`` collects ``*.png``, so no Family-C figure has ever
   reached ``docs/_images``.  Adding a PNG path is tracked separately; the
   ``ecf__`` filename prefix is already reserved in the collector.
   ``S1_gas_diagnostics.pdf`` is additionally absent from every run — that
   plotter fails and its exception is swallowed with a warning.
