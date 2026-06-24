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

**Tinker+2008 multiplicity function** (library default; the project's
fitting pipelines use ``"csst"`` as their baseline instead — see
*Emulator backends* below)

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

**Analytic backends** (all JAX-native, differentiable)

``make_hmf(backend, pk_func=pk_lin.pk_linear)`` accepts any key in
``_FSIGMA_MODELS``, including ``"tinker08"`` (the library's
dependency-free default when no backend is requested), ``"bocquet16"``,
``"yung25"``, and 14 others.

**Emulator backends**

Two optional emulator backends replace the analytic HMF with a Gaussian-Process
or neural-net prediction trained on N-body suites.  Both expose the same
``.dndm()``, ``.sigma()``, ``.bias()``, ``.n_eff()`` interface and are
selected via ``make_hmf(backend)``.
Halo bias always falls back to Tinker+2010 (neither emulator provides it).

The project's fitting pipelines (``hod_mod/scripts/fitting/*.py`` and the
corresponding ``configs/hod_fit_*.yml`` / ``configs/fitting/*_example.yml``)
use ``"csst"`` (CSSTEMU) as their baseline HMF backend rather than the
library default ``"tinker08"``, since its wide calibration range
(:math:`M \in [10^{10},10^{16}]\,M_\odot/h`, :math:`z \leq 3`) covers the
full mass range these pipelines integrate over. ``"aemulusnu"`` is *not*
used as a pipeline baseline: it is only calibrated for
:math:`M \geq 10^{13}\,M_\odot/h`, so HOD samples with non-negligible
occupation below that mass (e.g. low stellar-mass-threshold samples) get
silently extrapolated, dominating the integrated predictions with
unreliable values (see :class:`~hod_mod.cosmology.halo_mass_function.AemulusNuHaloMassFunction`).
Literature-validation scripts (``validate_*.py``, ``configs/benchmarks/*.yml``)
intentionally stay on ``"tinker08"`` to match the HMF used by the papers
they reproduce.

.. list-table::
   :header-rows: 1
   :widths: 18 22 18 22 20

   * - Backend key
     - Class
     - Paper
     - Calibration range
     - Extra dependency
   * - ``"csst"``
     - :class:`~hod_mod.cosmology.halo_mass_function.CsstHaloMassFunction`
     - [ChenCSST2025]_
       (`SCPMA 2025 <https://ui.adsabs.harvard.edu/abs/2025SCPMA..6809513C>`_)
     - :math:`M \in [10^{10},10^{16}]\,M_\odot/h`,
       :math:`z \leq 3`,
       :math:`\Omega_m,h,n_s,A_s,w_0,w_a,m_\nu,\Omega_b`
     - ``pip install git+https://github.com/czymh/csstemu``
   * - ``"aemulusnu"``
     - :class:`~hod_mod.cosmology.halo_mass_function.AemulusNuHaloMassFunction`
     - [ShenAemulus2025]_
       (`JCAP 2025 <https://arxiv.org/abs/2410.00913>`_)
     - :math:`M \geq 10^{13}\,M_\odot/h`,
       :math:`z \leq 2`,
       :math:`w_0w_a\nu{\rm CDM}`
     - ``pip install git+https://github.com/DelonShen/aemulusnu_hmf``

Both packages have numpy ≥ 2.0 and scipy ≥ 1.11 incompatibilities that
are patched automatically at import time by ``halo_mass_function.py``
(``scipy.integrate.simps`` alias, ``np.trapz`` alias, GPR predict squeeze,
``Cosmology.get_Omegam`` scalar fix).

**Usage**::

    from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
    from hod_mod.cosmology.halo_mass_function import make_hmf

    pk  = LinearPowerSpectrum()

    # Library default: analytic, differentiable, no extra dependency
    hmf = make_hmf("tinker08", pk_func=pk.pk_linear)

    # CSST emulator (pk_func ignored; uses CSST PkLin internally) —
    # this is the baseline backend used by the fitting pipelines
    hmf = make_hmf("csst")

    # Aemulus-ν emulator (requires pk_func for bias; M >= 1e13 Msun/h only)
    hmf = make_hmf("aemulusnu", pk_func=pk.pk_linear)

    # All expose the same interface:
    dn  = hmf.dndm(m_grid, z, theta)   # [h^4 Mpc^{-3} (M_sun/h)^{-1}]
    b   = hmf.bias(m_grid, z, theta)   # Tinker 2010 (all backends)

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

Gas Profiles
------------

(`hod_mod.cosmology.gas_profiles`)

Two parametric halo gas profiles are provided for computing galaxy ×
tSZ (thermal Sunyaev-Zel'dovich Compton-:math:`y`) and galaxy × soft X-ray
cross-correlations within the halo model.

.. _sec-m200-to-m500c:

M\ :sub:`200` → M\ :sub:`500c` conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the A10 pressure profile and the DPM density profile use overdensity
:math:`\Delta = 500c`.  The static halo model cache stores halo masses and
radii at :math:`\Delta = 200m`.  The helper function
:func:`~hod_mod.cosmology.gas_profiles.m200_to_m500c` performs the
conversion analytically using the NFW enclosed-mass formula
(`Navarro, Frenk & White 1997 <https://arxiv.org/abs/astro-ph/9611107>`_):

.. math::

   M_{\rm NFW}(r | M_{200}, c_{200}) = 4\pi\rho_s r_s^3
   \left[\ln\!\left(1 + \frac{r}{r_s}\right) - \frac{r/r_s}{1 + r/r_s}\right]

A bisection (``scipy.optimize.brentq``) finds :math:`r_{500c}` such that
:math:`M_{\rm NFW}(r_{500c}) = (4\pi/3)\,500\,\rho_{\rm crit}(z)\,r_{500c}^3`
and returns :math:`(M_{500c},\,R_{500c})`.

Arnaud+2010 Pressure Profile (A10) — for tSZ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference: `Arnaud et al. 2010 <https://arxiv.org/abs/0910.1234>`_,
A&A 517, A92, Table 1.

The generalised NFW (gNFW) shape function:

.. math::

   p(x) = \frac{P_0}{\left(c_{500}\,x\right)^\gamma
           \left[1 + \left(c_{500}\,x\right)^\alpha\right]^{(\beta-\gamma)/\alpha}}

where :math:`x = r/R_{500c}`.  Universal parameters (A10 Table 1):
:math:`P_0 = 8.403`, :math:`c_{500} = 1.177`, :math:`\gamma = 0.3081`,
:math:`\alpha = 1.0510`, :math:`\beta = 5.4905`, :math:`\alpha_p = 0.12`.

Physical electron pressure (A10, Eq. 11):

.. math::

   P_e(r|M_{500c},z) = 1.65\times10^{-3}\,h(z)^{8/3}
   \left[\frac{M_{500c}}{3\times10^{14}\,h_{70}^{-1}M_\odot}\right]^{2/3+\alpha_p}
   p\!\left(\frac{r}{R_{500c}}\right) \quad [h_{70}^2\,{\rm keV\,cm}^{-3}]

where :math:`h(z) = H(z)/H_0`.

The Fourier transform of the y-profile per halo:

.. math::

   \tilde{y}(k|M,z) = \frac{\sigma_T}{m_e c^2}
   \int_0^{r_{\rm max}} P_e(r|M,z)\,\frac{\sin(kr)}{kr}\,4\pi r^2\,dr
   \quad [({\rm Mpc}/h)^2]

where :math:`\sigma_T = 6.6524\times10^{-25}\,{\rm cm}^2` and
:math:`m_e c^2 = 511\,{\rm keV}`.
The integral is computed via Gauss-Legendre quadrature with
:math:`r_{\rm max} = 5\,R_{500c}` and 200 nodes by default.

DPM Electron Density Profile — for soft X-ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference: `Oppenheimer et al. 2025 <https://arxiv.org/abs/2505.14782>`_,
arXiv:2505.14782.

The gNFW-shaped density profile:

.. math::

   f(x|\boldsymbol{\alpha}) = x^{-\alpha_{\rm in}}
   \left(1 + x^{\alpha_{\rm tr}}\right)^{(\alpha_{\rm in}-\alpha_{\rm out})/\alpha_{\rm tr}}

where :math:`x = r/R_s` with scale radius :math:`R_s = R_{200}/c_{\rm DPM}`,
:math:`c_{\rm DPM} = 2.772`.

Electron density:

.. math::

   n_e(r|M_{200},z) = n_{e,03}\left(\frac{M_{200}}{10^{12}M_\odot}\right)^\beta
   E(z)^\gamma\,f\!\left(\frac{r}{R_s}\right)

where :math:`n_{e,03}` is the normalisation at :math:`r = 0.3\,R_{200}` for
:math:`M_{200}=10^{12}\,M_\odot/h` at :math:`z=0`.  Three calibrated DPM variants
are provided (model=1, 2, 3; see `Oppenheimer et al. 2025
<https://arxiv.org/abs/2505.14782>`_, Table 2).

The X-ray emissivity Fourier transform per halo is

.. math::

   \tilde{\varepsilon}(k|M,z) =
   \int_0^{r_{\rm max}} n_e^2(r|M,z)\,\frac{\sin(kr)}{kr}\,4\pi r^2\,dr
   \quad [({\rm Mpc}/h)^3\,{\rm cm}^{-6}]

with :math:`r_{\rm max} = 3\,R_{200}` and 200 GL nodes.

.. automodule:: hod_mod.cosmology.gas_profiles
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
