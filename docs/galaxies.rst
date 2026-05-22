Galaxies Module
===============

The ``hod_mod.galaxies`` sub-package implements the galaxy–halo connection: how many
galaxies of a given type reside in halos of mass :math:`M`, and what clustering and
lensing signals they produce.

---

HOD Models
----------

(`hod_mod.galaxies.hod`)

A Halo Occupation Distribution (HOD) specifies the probability :math:`P(N|M)` that a
halo of mass :math:`M` contains :math:`N` galaxies of a given type.  The mean occupation
factorises into centrals and satellites:

.. math::

   \langle N(M) \rangle = \langle N_{\rm cen}(M) \rangle
                        + \langle N_{\rm sat}(M) \rangle

Because a halo can only host a central if :math:`N_{\rm cen} \geq 1`, one assumes
:math:`\langle N_{\rm sat}(M) \rangle \propto \langle N_{\rm cen}(M) \rangle` at the
low-mass end.

Zheng+2007
~~~~~~~~~~

`Zheng et al. 2007 <https://arxiv.org/abs/astro-ph/0703457>`_ [Zheng2007]_ introduced the standard
parametrisation used for luminosity-selected galaxies:

.. math::

   \langle N_{\rm cen}(M) \rangle = \frac{1}{2}\left[1 + {\rm erf}
   \left(\frac{\log_{10}M - \log_{10}M_{\rm min}}{\sigma_{\log M}}\right)\right]

.. math::

   \langle N_{\rm sat}(M) \rangle = \langle N_{\rm cen}(M) \rangle
   \left(\frac{M - M_0}{M_1}\right)^\alpha

Free parameters: :math:`\log_{10}M_{\rm min}`, :math:`\sigma_{\log M}`,
:math:`\log_{10}M_0`, :math:`\log_{10}M_1`, :math:`\alpha`.

More+2015 (BOSS CMASS)
~~~~~~~~~~~~~~~~~~~~~~

`More et al. 2015 <https://arxiv.org/abs/1407.1011>`_ [More2015]_ extended Zheng+2007 with a linear
incompleteness function to model the colour-selected BOSS CMASS sample:

.. math::

   \langle N_{\rm cen}(M) \rangle = \frac{\alpha_{\rm inc}}{2}
   \left[1 + {\rm erf}\left(\frac{\log_{10}M - \log_{10}M_{\rm min}}
   {\sigma_{\log M}}\right)\right]

.. math::

   \langle N_{\rm sat}(M) \rangle = \langle N_{\rm cen}(M) \rangle
   \left(\frac{M - \kappa M_{\rm min}}{M_1}\right)^\alpha

Additional free parameters: :math:`\alpha_{\rm inc}` (incompleteness amplitude),
:math:`\kappa` (satellite-mass threshold as fraction of :math:`M_{\rm min}`).

Zu & Mandelbaum 2015 iHOD
~~~~~~~~~~~~~~~~~~~~~~~~~

`Zu & Mandelbaum 2015 <https://arxiv.org/abs/1505.02781>`_ [ZuMandelbaum2015]_ (Paper I) inverted the
standard HOD: instead of assigning galaxies to halos, they specify the
stellar-to-halo mass relation (SHMR) and derive the occupation from it.

The inverse SHMR (Eq. 19 of ZM15) gives halo mass as a function of stellar mass:

.. math::

   \log_{10} M_h(M_*) = \log_{10} M_1 + \beta \log_{10}\left(\frac{M_*}{M_{*,0}}\right)
   + \frac{(M_*/M_{*,0})^\delta}{1 + (M_*/M_{*,0})^{-\gamma}} - \frac{1}{2}

The forward SHMR :math:`M_*(M_h)` is obtained by bisection inversion.

The mass-dependent scatter (Eq. 20) is

.. math::

   \sigma_{\ln M_*}(M_h) = \sigma_0 + (\sigma_\infty - \sigma_0)
   \left[1 - \frac{2}{\pi}\arctan\left(\frac{\log_{10}M_h - \log_{10}M_\eta}
   {\eta}\right)\right]

The threshold central occupation (Eq. 21) is

.. math::

   \langle N_{\rm cen}(M_h | M_{*,{\rm th}}) \rangle =
   \frac{1}{2}{\rm erfc}\left[
   \frac{\ln M_{*,{\rm th}} - \ln M_*(M_h)}{\sqrt{2}\,\sigma_{\ln M_*}(M_h)}
   \right]

See also: `Zu & Mandelbaum 2016 <https://arxiv.org/abs/1509.06374>`_ [ZuMandelbaum2016]_ (Paper II,
galaxy quenching) and `2017 <https://arxiv.org/abs/1703.09219>`_ (Paper III, red/blue
fractions).

.. automodule:: hod_mod.galaxies.hod
   :members:
   :undoc-members:
   :show-inheritance:

---

Stellar-to-Halo Mass Relations
--------------------------------

(`hod_mod.galaxies.sham`)

Sub-halo abundance matching (SHAM) assumes a monotonic mapping between stellar mass
:math:`M_*` and halo peak circular velocity (or mass) :math:`M_h`.

Moster+2013
~~~~~~~~~~~

`Moster et al. 2013 <https://ui.adsabs.harvard.edu/abs/2013ApJ...770...57M>`_ [Moster2013]_ fitted a
double power-law SHMR with redshift-evolving parameters to abundance matching in the
Millennium and Millennium II simulations:

.. math::

   \frac{M_*(M_h, z)}{M_h} =
   2A(z)\left[\left(\frac{M_h}{M_1(z)}\right)^{-\beta(z)}
   + \left(\frac{M_h}{M_1(z)}\right)^{\gamma(z)}\right]^{-1}

with redshift evolution:
:math:`\log_{10} M_1(z) = M_{10} + M_{11} z/(1+z)`,
:math:`A(z) = A_{10} + A_{11} z/(1+z)`,
:math:`\beta(z) = \beta_{10} + \beta_{11} z/(1+z)`,
:math:`\gamma(z) = \gamma_{10} + \gamma_{11} z/(1+z)`.

Girelli+2020
~~~~~~~~~~~~~

`Girelli et al. 2020 <https://doi.org/10.1051/0004-6361/201936329>`_ [Girelli2020]_ (A&A 634, A135)
fitted a similar double power-law SHMR to COSMOS photometric data up to :math:`z=4`:

.. math::

   \frac{M_*(M_h, z)}{M_h} =
   \frac{2A(z)}{(M_h/M_A)^{-\beta(z)} + (M_h/M_A)^{\gamma(z)}}

with :math:`\log_{10}M_A = B + z\mu`, :math:`A = C(1+z)^\nu`,
:math:`\gamma = D(1+z)^\eta`, :math:`\beta = Fz + E`.

.. automodule:: hod_mod.galaxies.sham
   :members:
   :undoc-members:

---

Clustering
----------

(`hod_mod.galaxies.clustering`)

Projected correlation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The projected correlation function is the line-of-sight projection of the 3D
galaxy–galaxy correlation function :math:`\xi_{gg}(r)`:

.. math::

   w_p(r_p) = 2\int_0^{\pi_{\rm max}} \xi_{gg}(r_p, \pi)\,d\pi
            = 2\int_0^{\pi_{\rm max}} \xi_{gg}\!\left(\sqrt{r_p^2 + \pi^2}\right)d\pi

In Fourier space (Limber approximation for the power spectrum):

.. math::

   \xi_{gg}(r) = \frac{1}{2\pi^2}\int_0^\infty P_{gg}(k)\,\frac{\sin(kr)}{kr}\,k^2\,dk

The galaxy power spectrum in the halo model (see :doc:`cosmology`) is

.. math::

   P_{gg}(k) = P^{1h}_{gg}(k) + P^{2h}_{gg}(k)

with:

.. math::

   P^{1h}_{gg}(k) = \frac{1}{n_g^2}\int \frac{dn}{dM}
   \left[\langle N_{\rm cen} N_{\rm sat}\rangle u(k|M)
   + \langle N_{\rm sat}(N_{\rm sat}-1)\rangle u^2(k|M)\right]dM

.. math::

   P^{2h}_{gg}(k) = \frac{P_{\rm lin}(k)}{n_g^2}
   \left[\int \frac{dn}{dM}\,b(M)\,\langle N(M)\rangle\,u(k|M)\,dM\right]^2

The galaxy number density is

.. math::

   n_g = \int \langle N(M)\rangle \frac{dn}{dM}\,dM.

Excess surface density
~~~~~~~~~~~~~~~~~~~~~~

The galaxy–matter power spectrum is

.. math::

   P_{gm}(k) = P^{1h}_{gm}(k) + P^{2h}_{gm}(k)

with

.. math::

   P^{1h}_{gm}(k) = \frac{1}{n_g \bar{\rho}_m}\int \frac{dn}{dM}\, M\,
   \langle N(M)\rangle\, u^2(k|M)\,dM

The projected galaxy–matter correlation is

.. math::

   \Sigma_{gm}(R) = \bar{\rho}_m \int \xi_{gm}\!\left(\sqrt{R^2+\ell^2}\right)d\ell

and the weak-lensing excess surface density is

.. math::

   \Delta\Sigma(R) = \bar{\Sigma}_{gm}(<R) - \Sigma_{gm}(R)
   = \frac{2}{R^2}\int_0^R R'\,\Sigma_{gm}(R')\,dR' - \Sigma_{gm}(R).

Usage example:

.. code-block:: python

    from hod_mod.galaxies.clustering import FullHaloModelPrediction

    pred = FullHaloModelPrediction(pk_lin, hod, halo_profile, profile='nfw')
    wp   = pred.wp(rp, pi_max=60., z=0.1, theta_cosmo=theta, hod_params=p)
    ds   = pred.delta_sigma(R, z=0.1, theta_cosmo=theta, hod_params=p)

.. automodule:: hod_mod.galaxies.clustering
   :members:
   :undoc-members:

---

.. rubric:: Key references

[BerlindWeinberg2002]_, [Zheng2005]_, [vanUitert2016]_, [Guo2018]_, [Guo2019]_,
[Zacharegkas2025]_, [Behroozi2013]_, [DavisPeebles1983]_, [Hamilton1992]_.

---

Baryon Fraction
---------------

(`hod_mod.galaxies.baryon_fraction`)

Mass-dependent baryon fraction and gas concentration models for baryonic
suppression of the matter power spectrum and halo profiles.

.. automodule:: hod_mod.galaxies.baryon_fraction
   :members:
   :undoc-members:

---

Cross-Clustering
----------------

(`hod_mod.galaxies.cross_clustering`)

Galaxy–galaxy and galaxy–matter cross-clustering predictions for multi-tracer
analyses.

.. automodule:: hod_mod.galaxies.cross_clustering
   :members:
   :undoc-members:

---

Intrinsic Alignments
--------------------

(`hod_mod.galaxies.intrinsic_alignment`)

Non-linear alignment (NLA) model for intrinsic alignments of galaxy shapes
with the tidal field, used in joint :math:`w_p + \Delta\Sigma` analyses.

.. automodule:: hod_mod.galaxies.intrinsic_alignment
   :members:
   :undoc-members:
