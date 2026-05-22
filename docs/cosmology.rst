Cosmology Module
================

The ``hod_mod.cosmology`` sub-package computes the fundamental cosmological
quantities that drive the forward model: the matter power spectrum, the halo mass
function, halo density profiles, and geometric distances.

---

Distances and Volumes
---------------------

(`hod_mod.cosmology.distances`)

The comoving distance to redshift :math:`z` is

.. math::

   \chi(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

where :math:`E(z) = H(z)/H_0 = [\Omega_m(1+z)^3 + \Omega_\Lambda]^{1/2}` for a flat
:math:`\Lambda`CDM cosmology.  The angular diameter distance is
:math:`D_A = \chi/(1+z)` and the luminosity distance is
:math:`D_L = \chi (1+z)`.

The comoving volume element per steradian is

.. math::

   \frac{dV}{dz\,d\Omega} = \frac{c}{H_0} \frac{\chi^2(z)}{E(z)}.

.. automodule:: hod_mod.cosmology.distances
   :members:
   :undoc-members:

---

Linear Power Spectrum
---------------------

(`hod_mod.cosmology.power_spectrum`)

The dimensionless power spectrum is

.. math::

   \Delta^2(k, z) \equiv \frac{k^3 P(k, z)}{2\pi^2}.

For :math:`\Delta^2 \ll 1` the density field is in the linear regime.

**CAMB backend** (default)

The full Boltzmann code CAMB (`Lewis, Challinor & Lasenby 2000
<https://arxiv.org/abs/astro-ph/0205436>`_) [Lewis2002]_ is invoked via
``LinearPowerSpectrum.pk_linear(k, z, theta)``.  A single CAMB run takes ~30 s; for
MCMC use the ``CachedPkLinear`` wrapper that interpolates on a pre-computed
:math:`k`-grid keyed on :math:`(\Omega_m, h, \ln 10^{10}A_s, z)`.

**Eisenstein-Hu 1998 fitting formula** (fast)

`Eisenstein & Hu 1998 <https://arxiv.org/abs/astro-ph/9709066>`_ [EisensteinHu1998]_ provide an accurate
analytical approximation to the transfer function:

.. math::

   P_{\rm EH}(k) \propto k^{n_s} T^2(k)

where the transfer function :math:`T(k)` captures baryonic acoustic oscillations
through a fitting formula involving the baryon-to-matter ratio, the matter-radiation
equality scale, and the Silk damping scale.  Implemented as ``eisenstein_hu_pk(k, theta)``
in JAX; differentiable with respect to all cosmological parameters.

**Growth factor**

Linear growth is encoded as :math:`P(k,z) = D^2(z) P(k,0)` with the growth factor

.. math::

   D(z) = \frac{5\Omega_m}{2} H(z) / H_0
   \int_z^\infty \frac{(1+z')}{[H(z')/H_0]^3}\,dz'.

.. automodule:: hod_mod.cosmology.power_spectrum
   :members:
   :undoc-members:

---

Non-linear Power Spectrum
--------------------------

(`hod_mod.cosmology.nonlinear`)

The non-linear power spectrum :math:`P_{\rm nl}(k,z)` is computed by the
Aletheia emulator (`Contreras et al. 2023 <https://arxiv.org/abs/2305.00015>`_) [Aletheia2025]_,
a neural-network emulator trained on a suite of N-body simulations spanning a wide
cosmological parameter space including massive neutrinos and dynamical dark energy.

.. automodule:: hod_mod.cosmology.nonlinear
   :members:
   :undoc-members:

---

Halo Mass Function
------------------

(`hod_mod.cosmology.halo_mass_function`)

The comoving number density of halos per unit logarithmic mass is

.. math::

   \frac{dn}{d\ln M} = f(\sigma) \frac{\bar{\rho}_m}{M}
   \left|\frac{d\ln\sigma^{-1}}{d\ln M}\right|

where :math:`\bar{\rho}_m` is the mean comoving matter density and
:math:`\sigma^2(M, z)` is the variance of the linear density field smoothed on the
Lagrangian radius :math:`R = (3M/4\pi\bar{\rho}_m)^{1/3}`:

.. math::

   \sigma^2(M, z) = \frac{D^2(z)}{2\pi^2}
   \int_0^\infty P_{\rm lin}(k, 0)\, W^2(kR)\, k^2\, dk

with the top-hat window function :math:`W(x) = 3(\sin x - x\cos x)/x^3`.

**Tinker+2008 multiplicity function** (default)

`Tinker et al. 2008 <https://arxiv.org/abs/0803.2706>`_ [Tinker2008]_ calibrated the multiplicity
function :math:`f(\sigma)` against N-body simulations for halos defined by a fixed
overdensity :math:`\Delta = 200` relative to the mean background:

.. math::

   f(\sigma) = A\left[1 + \left(\frac{\sigma}{b}\right)^{-a}\right]
   \exp\left(-\frac{c}{\sigma^2}\right)

with best-fit parameters :math:`A=0.186`, :math:`a=1.47`, :math:`b=2.57`,
:math:`c=1.19` at :math:`z=0`.  Redshift evolution of the parameters is also given in
Table 2 of that paper.

**Halo bias**

The linear halo bias relates the halo overdensity to the matter overdensity on large
scales.  Using the Tinker+2010 prescription
(`Tinker et al. 2010 <https://arxiv.org/abs/1001.3162>`_) [Tinker2010]_:

.. math::

   b(M, z) = 1 - A_b \frac{\nu^{a_b}}{\nu^{a_b} + \delta_c^{a_b}}
              + B_b\,\nu^{b_b} + C_b\,\nu^{c_b}

where :math:`\nu = \delta_c / \sigma(M,z)` is the peak height and
:math:`\delta_c \approx 1.686` is the linear collapse threshold.

The effective galaxy bias is obtained by weighting over the occupation-weighted HMF:

.. math::

   b_{\rm eff} = \frac{\int b(M)\,\langle N(M)\rangle\,\frac{dn}{dM}\,dM}
                      {\int \langle N(M)\rangle\,\frac{dn}{dM}\,dM}

**Available backends**: ``"tinker08"`` (JAX-native, differentiable), ``"aemulus"``
(AemulusNu GP emulator), ``"csst"`` (CSST neural-net emulator).

.. automodule:: hod_mod.cosmology.halo_mass_function
   :members:
   :undoc-members:

---

Halo Profiles
-------------

(`hod_mod.cosmology.halo_profiles`)

**NFW profile**

`Navarro, Frenk & White 1997 <https://arxiv.org/abs/astro-ph/9508025>`_ [NFW1997]_ showed that
the radial density profiles of dark matter halos are well described by

.. math::

   \rho_{\rm NFW}(r) = \frac{\rho_s}{(r/r_s)(1 + r/r_s)^2}

where :math:`r_s = r_{200}/c` is the scale radius and :math:`c` is the
concentration parameter.  The characteristic density is

.. math::

   \rho_s = \frac{M_{200}}{4\pi r_s^3 \left[\ln(1+c) - c/(1+c)\right]}.

**Einasto profile**

`Einasto 1965 <https://ui.adsabs.harvard.edu/abs/1965TrAlm...5...87E>`_ [Einasto1965]_ proposed a
power-law logarithmic slope profile:

.. math::

   \rho_{\rm Ein}(r) = \rho_{-2} \exp\left\{
   -\frac{2}{\alpha_E}\left[\left(\frac{r}{r_{-2}}\right)^{\alpha_E} - 1\right]
   \right\}

where :math:`r_{-2}` is the radius where the logarithmic slope equals :math:`-2` and
:math:`\alpha_E \approx 0.18` for typical halos.

**Concentration–mass relation**

The concentration is obtained from colossus using the Diemer & Joyce 2019 relation
(`Diemer & Joyce 2019 <https://arxiv.org/abs/1809.07326>`_) [DiemerJoyce2019]_:

.. math::

   c(M, z) = c_0 \left(\frac{M}{M_{\rm piv}}\right)^{-\kappa_c}
   \left(1 + z\right)^{-\mu_c}

**Projected quantities**

The surface mass density is the line-of-sight projection:

.. math::

   \Sigma(R) = 2 \int_0^\infty \rho\!\left(\sqrt{R^2 + \ell^2}\right) d\ell

The mean surface density within radius :math:`R` is

.. math::

   \bar{\Sigma}(<R) = \frac{2}{R^2} \int_0^R R'\,\Sigma(R')\,dR'

and the excess surface density (the weak-lensing observable) is

.. math::

   \Delta\Sigma(R) = \bar{\Sigma}(<R) - \Sigma(R).

**Fourier transform** of the NFW profile (for the halo model):

.. math::

   u(k|M) = \frac{4\pi\rho_s r_s^3}{M}
   \left[\cos(kr_s)\left({\rm Ci}((1+c)kr_s) - {\rm Ci}(kr_s)\right)
   + \sin(kr_s)\left({\rm Si}((1+c)kr_s) - {\rm Si}(kr_s)\right)
   - \frac{\sin(ckr_s)}{(1+c)kr_s}\right]

where Ci and Si are the cosine and sine integrals.

.. automodule:: hod_mod.cosmology.halo_profiles
   :members:
   :undoc-members:

---

Matter Power Spectrum (Halo Model)
-----------------------------------

(`hod_mod.cosmology.halo_model`)

The 1-halo + 2-halo decomposition of the nonlinear matter power spectrum is
(`Cooray & Sheth 2002 <https://ui.adsabs.harvard.edu/abs/2002PhR...372....1C>`_) [CooraySheth2002]_:

.. math::

   P_{\rm mm}(k) = P^{1h}_{\rm mm}(k) + P^{2h}_{\rm mm}(k)

The 1-halo term (pairs within the same halo) dominates at
:math:`k \gtrsim 1\,h\,{\rm Mpc}^{-1}`:

.. math::

   P^{1h}_{\rm mm}(k) = \int \frac{dn}{dM}
   \left(\frac{M}{\bar{\rho}_m}\right)^2 u^2(k|M)\,dM

The 2-halo term (pairs in different halos) dominates at large scales:

.. math::

   P^{2h}_{\rm mm}(k) = P_{\rm lin}(k)
   \left[\int \frac{dn}{dM} b(M)\,\frac{M}{\bar{\rho}_m}\, u(k|M)\,dM\right]^2

.. automodule:: hod_mod.cosmology.halo_model
   :members:
   :undoc-members:

---

.. rubric:: Key references

[PressSchechter1974]_, [ShethTormen1999]_, [Jenkins2001]_, [Nishimichi2019]_,
[SeljakWarren2004]_, [WrightBrainerd2000]_, [BryanNorman1998]_.
