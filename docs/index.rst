hod_mod: Galaxy Clustering, Lensing and X-ray / tSZ Cross-Correlations
======================================================================

**hod_mod** is a JAX-accelerated Python package for predicting and fitting
galaxy–halo observables. It is organised around three observable **pipelines**
that share one core halo-model engine:

* **Galaxy clustering & lensing** — :math:`w_p(r_p)` and :math:`\Delta\Sigma(R)`
  from the More+2015 and Zu & Mandelbaum 2015 models.
* **Galaxy × X-ray** — soft X-ray emission from hot gas and AGN (ongoing).
* **Galaxy × thermal SZ** — Compton-:math:`y` cross-correlation (new).

The package layout mirrors this: ``core`` (cosmology + halo model), ``connection``
(galaxy–halo occupation), ``gas`` and ``agn`` (the fields), and ``observables``
(the pipelines), with ``fitting`` and the ``hod-mod`` CLI on top.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   overview
   forward_model_showcase
   data_formats
   data_hosting
   scripts

.. toctree::
   :maxdepth: 2
   :caption: Pipeline — Galaxy Clustering & Lensing

   hod_more2015
   hod_zumandelbaum2015
   fitting_more2015

.. toctree::
   :maxdepth: 2
   :caption: Pipeline — Galaxy × X-ray

   direct_prediction_gal_gas
   xray_joint_fit

.. toctree::
   :maxdepth: 2
   :caption: Pipeline — Galaxy × thermal SZ

   pipeline_gal_tsz

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   benchmark_more2015
   benchmark_zumandelbaum2015_multisample

.. toctree::
   :maxdepth: 1
   :caption: BGS Analysis

   bgs_ls10_wp_survey
   bgs_comparat2025_mstar10_joint_fit
   timing_joint_model

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   cosmology
   halos
   galaxies
   fitting
   data_io
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
