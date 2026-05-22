Halos
=====

Halo profiles, lensing quantities, and concentration–mass relations.
The NFW profile math and derivations are in :doc:`cosmology` under **Halo Profiles**;
this page provides the full API reference for the halo sub-package.

---

Concentration–Mass Relations
-----------------------------

(`hod_mod.cosmology.concentration`)

Multiple calibrations of :math:`c(M, z)` are available as standalone functions
and via the :class:`~hod_mod.cosmology.concentration.ConcentrationModel` wrapper:

* **Duffy+2008** — fitted to the Millennium simulation at :math:`z = 0\text{–}2`
* **Dutton+2014** — based on Planck-normalised ΛCDM N-body runs
* **Klypin+2016** — MultiDark-Planck simulation calibration
* **Bhattacharya+2013** — calibrated against cluster lensing data
* **Diemer+2015** — uses the effective slope of P(k) via colossus
* **Diemer+2019** (default in :class:`~hod_mod.cosmology.halo_profiles.HaloProfile`) — updated colossus calibration

.. automodule:: hod_mod.cosmology.concentration
   :members:
   :undoc-members:

---

Halo Profiles
-------------

(`hod_mod.cosmology.halo_profiles`)

.. automodule:: hod_mod.cosmology.halo_profiles
   :members:
   :undoc-members:

---

Halo Model Power Spectrum
--------------------------

(`hod_mod.cosmology.halo_model`)

.. automodule:: hod_mod.cosmology.halo_model
   :members:
   :undoc-members:
