"""Linear matter power spectrum backends.

CLASS (:class:`ClassPkLinear`) is the reference; CAMB
(:class:`LinearPowerSpectrum`) is the second Boltzmann solver kept for
cross-checks; :func:`eisenstein_hu_pk` is the analytic, differentiable
fallback.  :func:`default_pk_linear` selects between them and defaults to
the CosmoPower-JAX emulator, which lives in ``hod_mod.forecast``.
"""

from collections import OrderedDict

import numpy as np
import jax
import jax.numpy as jnp


def rho_critical_0() -> float:
    """Critical matter density at z=0 for H₀ = 100 km/s/Mpc, in h-units.

    .. math::

        \\rho_{\\mathrm{crit},0} = \\frac{3H_{100}^2}{8\\pi G}
        \\approx 2.775\\times10^{11}\\;(M_\\odot/h)\\,(\\mathrm{Mpc}/h)^{-3}

    In h-unit conventions the h² from :math:`H_0 = 100h` km/s/Mpc cancels the
    :math:`h^{-3}` from the comoving volume, so this quantity is independent of h.
    The mean matter density follows as
    :math:`\\bar{\\rho}_m = \\Omega_m\\,\\rho_{\\mathrm{crit},0}`.

    Physical constants used:
    :math:`G = 6.67430\\times10^{-11}` m³ kg⁻¹ s⁻²,
    1 Mpc = 3.085677581×10²² m,
    1 M⊙ = 1.989×10³⁰ kg.

    Returns
    -------
    rho_crit0 : float, (Msun/h) / (Mpc/h)³
    """
    G_SI    = 6.67430e-11           # m³ kg⁻¹ s⁻²
    Mpc_m   = 3.085677581e22        # m Mpc⁻¹
    Msun_kg = 1.989e30              # kg Msun⁻¹
    H100_SI = 1e5 / Mpc_m           # 100 km/s/Mpc in s⁻¹
    rho_SI  = 3.0 * H100_SI**2 / (8.0 * np.pi * G_SI)  # kg m⁻³
    return float(rho_SI * Mpc_m**3 / Msun_kg)


# ---------------------------------------------------------------------------
# Massive neutrinos
# ---------------------------------------------------------------------------

#: Ω_ν h² = Σm_ν / 93.14 eV — the standard closure relation for three
#: degenerate species at N_eff = 3.044 (Mangano et al. 2005).
NU_DENOM_EV = 93.14

_WARNED_MNU_ALIAS = False


def mnu_sum(theta):
    """Σm_ν [eV] from a cosmology dict; 0.0 when the key is absent.

    ``"sum_mnu"`` is the canonical spelling across the package.  ``"mnu"`` is
    accepted as a deprecated alias (it was the spelling the CSST and Aemulus-ν
    paths used to read, though nothing ever wrote it) and warns once.

    Traced values pass through unchanged, so this is safe under ``jax.jit``.
    """
    global _WARNED_MNU_ALIAS
    if "sum_mnu" in theta:
        return theta["sum_mnu"]
    if "mnu" in theta:
        if not _WARNED_MNU_ALIAS:
            import warnings
            warnings.warn(
                "cosmology key 'mnu' is a deprecated alias for 'sum_mnu'; "
                "update the dict to use 'sum_mnu'.",
                DeprecationWarning, stacklevel=2)
            _WARNED_MNU_ALIAS = True
        return theta["mnu"]
    return 0.0


def omega_nu(theta):
    """Ω_ν = Σm_ν / (93.14 eV · h²) — a density parameter, not ω_ν = Ω_ν h².

    A ``theta`` with no neutrino sector returns ``0.0`` **without reading**
    ``h``.  That is not a convenience: ``omega_cb``/``omega_cdm`` are called on
    cosmology dicts that legitimately carry only the parameters their consumer
    needs -- ``observables/baryon_fraction`` is handed ``{Omega_b, Omega_m}`` --
    and requiring ``h`` to compute a quantity that is identically zero turned
    those into a ``KeyError``.  The branch is on dict-key membership, which is
    static under ``jit``.
    """
    if "sum_mnu" not in theta and "mnu" not in theta:
        return 0.0
    return mnu_sum(theta) / (NU_DENOM_EV * theta["h"] ** 2)


def omega_cb(theta):
    """Ω_cb = Ω_m − Ω_ν, the *cold* (CDM + baryon) density parameter.

    This is the density that sets the Lagrangian radius of a halo and the
    normalisation of the mass function: neutrinos contribute to Ω_m but do not
    cluster into halos.  See the "Massive neutrinos" section of the cosmology
    documentation page for the cold-versus-total convention.
    """
    return theta["Omega_m"] - omega_nu(theta)


def omega_cdm_cold_equivalent(theta):
    """Omega_cdm for a backend that has no neutrino sector of its own.

    EH98 and the CosmoPower network are massless by construction, so the
    spectrum they should return is the massless one at the *same* total
    Omega_m -- which is exactly the reference
    :mod:`hod_mod.core.nu_ratio` measured its ratios against.  The neutrinos
    are therefore lumped back into the cold sector here and taken out again by
    the ratio.  Feeding these backends the true (reduced) Omega_cdm *and* then
    applying the ratio counts the cdm -> nu transfer twice: 1.16% against CLASS
    instead of 0.14%.
    """
    return theta["Omega_m"] - theta["Omega_b"]


def omega_cdm(theta):
    """Ω_cdm = Ω_m − Ω_b − Ω_ν, or the explicit ``Omega_cdm`` entry if present.

    Planck's Ω_m already *includes* the minimal neutrino mass, so subtracting
    Ω_ν here is what keeps the budget closed: adding Σm_ν on top of a fixed
    Ω_cdm would silently raise Ω_m instead.
    """
    if "Omega_cdm" in theta:
        return theta["Omega_cdm"]
    return theta["Omega_m"] - theta["Omega_b"] - omega_nu(theta)


def default_pk_linear(backend: str = None):
    """Construct the default linear-P(k) backend for the production stack.

    Returns a :class:`~hod_mod.forecast.pk_cosmopower.CosmoPowerJaxPkLinear` — the
    CosmoPower-JAX neural emulator — which reproduces CAMB's *shape* to <0.1% while
    being JAX-differentiable and ~instant (no 30 s CAMB call per cosmology).

    .. warning::

       The emulator is trained on **massless neutrinos**, whereas :class:`LinearPowerSpectrum`
       runs CAMB with its default ``mnu=0.06`` eV.  In the A_s convention used here
       nothing renormalises that away, so the emulator's P(k) is **~2.5% higher in
       amplitude** (≈1.3% in σ8-equivalent) than the CAMB backend it replaced.  This
       is a physics difference, not emulator error — the two agree to ~0.06% when CAMB
       is also run massless.  Results produced before this became the default are not
       directly comparable; see the emulator's ``TRAINING_BOX`` for its validity range.

    ``backend`` (or ``$HOD_MOD_PK_BACKEND``) overrides: ``"cosmopower"`` (default),
    ``"class"`` for the CLASS reference backend, or ``"camb"`` for the second
    Boltzmann solver.  Both Boltzmann paths cost seconds per cosmology and are
    not differentiable, so they are for reference and validation work, not for
    chains.
    """
    import os
    backend = (backend or os.environ.get("HOD_MOD_PK_BACKEND", "cosmopower")).lower()
    if backend == "camb":
        return LinearPowerSpectrum()
    if backend == "class":
        return ClassPkLinear()
    if backend == "cosmopower":
        from hod_mod.forecast.pk_cosmopower import CosmoPowerJaxPkLinear
        return CosmoPowerJaxPkLinear()
    raise ValueError(
        f"pk backend must be 'cosmopower', 'class' or 'camb', got {backend!r}")


def _interp_pk_table(k, kh, pk):
    """Log-log interpolate a tabulated P(k) onto the requested k grid.

    Shared by the two Boltzmann backends so a CAMB table and a CLASS table are
    resampled by exactly the same rule and can be differenced directly.
    """
    return jnp.power(
        10.0,
        jnp.interp(jnp.log(k), jnp.log(jnp.asarray(kh)),
                   jnp.log10(jnp.maximum(jnp.asarray(pk), 1e-50))),
    )


class LinearPowerSpectrum:
    """Linear P(k, z) computed with CAMB.

    Parameters
    ----------
    (none — no pre-trained weights required)
    """

    #: CAMB solves at whatever redshift it is asked for, so ``growth`` can be
    #: read straight off its own spectra (see :mod:`hod_mod.core.growth`).
    has_native_z = True

    def __init__(self):
        try:
            import camb
        except ImportError as e:
            raise ImportError("camb not installed — pip install camb") from e
        self._camb = camb

    def _camb_results(self, z: float, theta: dict):
        """Run CAMB and return results object.  Internal helper for pk_linear*."""
        h = float(theta["h"])
        lnAs = float(theta["ln10^{10}A_s"])
        w0 = float(theta.get("w0", -1.0))
        wa = float(theta.get("wa", 0.0))

        pars = self._camb.CAMBparams()
        pars.set_cosmology(
            H0=100.0 * h,
            ombh2=float(theta["Omega_b"]) * h**2,
            omch2=float(omega_cdm(theta)) * h**2,
            mnu=float(mnu_sum(theta)),
            num_massive_neutrinos=3,
            nnu=3.044,
        )
        pars.InitPower.set_params(
            ns=float(theta["n_s"]),
            As=np.exp(lnAs) * 1e-10,
        )
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
        pars.set_matter_power(redshifts=[float(z)], kmax=200.0)
        return self._camb.get_results(pars)

    def _interp_pk(self, k, kh, pk2d):
        """Log-log interpolate a CAMB P(k) table onto the requested k grid."""
        return _interp_pk_table(k, kh, pk2d[0])

    def pk_linear(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """Linear P(k) [(Mpc/h)^3] at redshift z (total matter).

        Supports CPL dark energy via ``w0`` and ``wa`` keys in ``theta``
        (defaults to ΛCDM if absent).  CAMB uses the PPF dark energy model
        (Hu & Sawicki 2007) which remains accurate for :math:`w < -1`.

        Parameters
        ----------
        k : array_like, h/Mpc
        z : float
        theta : dict  — keys: h, Omega_b, Omega_cdm, n_s, ln10^{10}A_s,
                        w0 (default -1), wa (default 0)
        """
        results = self._camb_results(float(z), theta)
        kh, _, pk2d = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=200.0, npoints=1024
        )
        return self._interp_pk(k, kh, pk2d)

    def pk_linear_cdm(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """CDM auto-power spectrum P_CDM(k) [(Mpc/h)^3] at redshift z.

        .. math::

            P_{\\rm CDM}(k) = T_{\\rm cdm}^2(k)\\,P_{\\rm prim}(k)

        where :math:`T_{\\rm cdm}` is the CAMB CDM transfer function.
        To recover total matter: :math:`f_b P_b + f_c P_{\\rm CDM} \\approx P_{\\rm tot}`
        (exact in linear theory when cross-spectrum terms dominate, valid within ~5%).

        Parameters
        ----------
        k : array_like, h/Mpc
        z : float
        theta : dict — same keys as :meth:`pk_linear`
        """
        results = self._camb_results(float(z), theta)
        kh, _, pk2d = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=200.0, npoints=1024,
            var1="delta_cdm", var2="delta_cdm",
        )
        return self._interp_pk(k, kh, pk2d)

    def pk_linear_b(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """Baryon auto-power spectrum P_b(k) [(Mpc/h)^3] at redshift z.

        .. math::

            P_b(k) = T_b^2(k)\\,P_{\\rm prim}(k)

        Baryonic BAO wiggles are more pronounced here than in the total spectrum.

        Parameters
        ----------
        k : array_like, h/Mpc
        z : float
        theta : dict — same keys as :meth:`pk_linear`
        """
        results = self._camb_results(float(z), theta)
        kh, _, pk2d = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=200.0, npoints=1024,
            var1="delta_baryon", var2="delta_baryon",
        )
        return self._interp_pk(k, kh, pk2d)

    @staticmethod
    def default_cosmology() -> dict:
        """Planck 2018 TT,TE,EE+lowE best-fit values (flat ΛCDM, Σm_ν = 0.06 eV).

        The budget closes by construction: ``Omega_b + Omega_cdm + Omega_nu ==
        Omega_m`` exactly.  Planck's Ω_m = 0.3100 already includes the minimal
        normal-hierarchy neutrino mass, so Ω_cdm is 0.2593, not 0.2607 —
        carrying 0.2607 *and* a massive neutrino would put the spectrum at
        Ω_m ≈ 0.3114 while ρ̄, E(z) and D(z) still used 0.3100.
        """
        return {
            "h": 0.6736,
            "Omega_b": 0.0493,
            "Omega_cdm": 0.3100 - 0.0493 - 0.06 / (NU_DENOM_EV * 0.6736 ** 2),
            "Omega_m": 0.3100,
            "n_s": 0.9649,
            "ln10^{10}A_s": 3.044,
            "sum_mnu": 0.06,
            "w0": -1.0,
            "wa": 0.0,
        }


#: CLASS ``N_ur`` for three degenerate massive species sharing N_eff = 3.044
#: (the recipe in CLASS's own ``explanatory.ini``).  Using the massless value
#: 3.044 alongside ``deg_ncdm = 3`` would put N_eff near 5 and inflate the
#: early-time radiation density -- a ~14% error in sigma8.
_CLASS_N_UR_3NU = 0.00441
_CLASS_N_UR_MASSLESS = 3.044


class ClassPkLinear:
    """Linear P(k, z) computed with CLASS (Blas, Lesgourgues & Tram 2011).

    The reference backend.  CLASS is an independent Boltzmann solver, so
    differencing it against CAMB bounds the numerical error of the reference
    itself, and unlike every other backend here it treats massive neutrinos
    exactly rather than through a fitting form.  It also exposes the *cold*
    (CDM + baryon) spectrum through :meth:`pk_linear_cb`, which is what the
    halo sector needs: neutrinos contribute to Omega_m but do not cluster into
    halos.

    One CLASS solve costs a few seconds and is entirely dominated by
    ``compute()`` -- pulling P(k) out afterwards is milliseconds.  The solver
    object is therefore cached per cosmology, so ``pk_linear`` and
    ``pk_linear_cb`` at many redshifts all share a single solve.

    Parameters
    ----------
    z_max : float
        Largest redshift the cached solve can serve (CLASS ``z_max_pk``).
    cache_size : int
        How many distinct cosmologies to keep solved at once.  Evicted solvers
        are freed with ``struct_cleanup()``/``empty()``.
    """

    #: CLASS solves at arbitrary redshift, and its cached solver returns many
    #: redshifts from one ``compute()`` -- so a growth ratio here is nearly free.
    has_native_z = True

    #: Internal table grid -- identical to :class:`LinearPowerSpectrum` so the
    #: two Boltzmann backends can be differenced without regridding.
    _KMIN, _KMAX, _NK = 1e-4, 200.0, 1024

    def __init__(self, z_max: float = 5.0, cache_size: int = 2):
        try:
            import classy
        except ImportError as e:
            raise ImportError("classy not installed — pip install classy") from e
        self._classy = classy
        self._z_max = float(z_max)
        self._cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()
        self._kh = np.logspace(np.log10(self._KMIN), np.log10(self._KMAX), self._NK)

    # -- CLASS configuration -------------------------------------------
    @staticmethod
    def _key(theta: dict) -> tuple:
        """Cache key: everything the CLASS solve depends on."""
        return (round(float(theta["h"]), 8),
                round(float(theta["Omega_b"]), 9),
                round(float(omega_cdm(theta)), 9),
                round(float(theta["n_s"]), 8),
                round(float(theta["ln10^{10}A_s"]), 8),
                round(float(mnu_sum(theta)), 8),
                round(float(theta.get("w0", -1.0)), 8),
                round(float(theta.get("wa", 0.0)), 8),
                round(float(theta.get("T_cmb", 2.7255)), 6))

    def _settings(self, theta: dict) -> dict:
        h = float(theta["h"])
        mnu = float(mnu_sum(theta))
        pars = {
            "output":          "mPk",
            "h":               h,
            "omega_b":         float(theta["Omega_b"]) * h ** 2,
            "omega_cdm":       float(omega_cdm(theta)) * h ** 2,
            "n_s":             float(theta["n_s"]),
            "A_s":             float(np.exp(float(theta["ln10^{10}A_s"])) * 1e-10),
            "P_k_max_h/Mpc":   self._KMAX,
            "z_max_pk":        self._z_max,
            "T_cmb":           float(theta.get("T_cmb", 2.7255)),
        }
        if mnu > 0.0:
            pars.update({"N_ncdm": 1, "deg_ncdm": 3.0, "m_ncdm": mnu / 3.0,
                         "N_ur": _CLASS_N_UR_3NU})
        else:
            pars["N_ur"] = _CLASS_N_UR_MASSLESS
        w0 = float(theta.get("w0", -1.0))
        wa = float(theta.get("wa", 0.0))
        if w0 != -1.0 or wa != 0.0:
            pars.update({"Omega_Lambda": 0.0, "w0_fld": w0, "wa_fld": wa,
                         "use_ppf": "yes"})
        return pars

    def _solver(self, theta: dict):
        """The cached CLASS instance for this cosmology, solving on a miss."""
        key = self._key(theta)
        hit = self._cache.pop(key, None)
        if hit is None:
            hit = self._classy.Class()
            hit.set(self._settings(theta))
            hit.compute()
            while len(self._cache) >= self._cache_size:
                _, stale = self._cache.popitem(last=False)
                stale.struct_cleanup()
                stale.empty()
        self._cache[key] = hit          # re-insert as most-recently used
        return hit

    def clear_cache(self) -> None:
        """Free every cached CLASS solve."""
        while self._cache:
            _, stale = self._cache.popitem()
            stale.struct_cleanup()
            stale.empty()

    # -- spectra --------------------------------------------------------
    def _table(self, z: float, theta: dict, cold: bool) -> np.ndarray:
        cosmo = self._solver(theta)
        h = float(theta["h"])
        want_cb = cold and float(mnu_sum(theta)) > 0.0
        pk_of = cosmo.pk_cb_lin if want_cb else cosmo.pk_lin
        return np.array([pk_of(ki * h, float(z)) for ki in self._kh]) * h ** 3

    def pk_linear(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """Total-matter linear P(k) [(Mpc/h)^3] at redshift z.

        Parameters
        ----------
        k : array_like, h/Mpc
        z : float
        theta : dict — keys: h, Omega_b, n_s, ln10^{10}A_s; optional
                Omega_cdm (else Omega_m - Omega_b - Omega_nu), sum_mnu
                (default 0), w0 (-1), wa (0), T_cmb (2.7255)
        """
        return _interp_pk_table(k, self._kh, self._table(z, theta, cold=False))

    def pk_linear_cb(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """Cold (CDM + baryon) linear P(k) [(Mpc/h)^3] at redshift z.

        This is the spectrum the halo mass function and halo bias should be
        built from when Sigma m_nu > 0.  It reduces to :meth:`pk_linear`
        exactly in the massless limit.
        """
        return _interp_pk_table(k, self._kh, self._table(z, theta, cold=True))

    def as_hmf_pk_func(self):
        """Return a ``(k, z, theta) -> P_m`` callable for ``make_hmf``.

        Total matter, like every other backend: ``HaloMassFunction`` converts to
        the cold spectrum itself, uniformly, through the ``nu_ratio`` table.
        Returning ``pk_linear_cb`` here would apply that conversion twice.
        :meth:`pk_linear_cb` remains the exact cold spectrum for direct use and
        is what the table is validated against.
        """
        def _pk_func(k, z, theta):
            return self.pk_linear(k, z, theta)
        return _pk_func

    @staticmethod
    def default_cosmology() -> dict:
        """Same Planck 2018 dictionary the CAMB backend uses."""
        return LinearPowerSpectrum.default_cosmology()


@jax.jit
def eisenstein_hu_pk(k: jnp.ndarray, theta: dict) -> jnp.ndarray:
    """Eisenstein & Hu (1998) transfer function with BAO wiggles.

    Implements eqs. (2)–(7), (10)–(24) of EH98.  The total transfer function is
    :math:`T(k) = f_b T_b(k) + f_c T_c(k)` and the power spectrum is
    :math:`P(k) \\propto k^{n_s} T(k)^2`, normalised to unity at
    :math:`k = 0.05\\,h\\,\\mathrm{Mpc}^{-1}`.

    Parameters
    ----------
    k : array_like, h/Mpc
    theta : dict — keys: h, Omega_m, Omega_b, n_s; optional T_cmb (K, default 2.7255)

    Accuracy
    --------
    Normalised to P(k=0.05 h/Mpc) = 1.0 by construction.  Shape agrees with
    CAMB P(k) to < 10% rms for k ∈ [0.01, 0.3] h/Mpc (Planck 2018, z=0).
    Large-scale slope d log P / d log k ≈ n_s to < 0.15 for k < 0.003 h/Mpc
    (2026-04-23).

    Timing
    ------
    ~ 242 µs / call  (JIT-compiled, N=200 wavenumbers, CPU x86-64, 2026-04-23).
    """
    h    = theta["h"]
    om   = theta["Omega_m"]
    ob   = theta["Omega_b"]
    ns   = theta["n_s"]
    Tcmb = theta.get("T_cmb", 2.7255)

    oc   = omega_cdm_cold_equivalent(theta)   # massless-trained: nu lumped into cdm
    fb   = ob / om          # baryon fraction
    fc   = oc / om          # CDM fraction
    omh2 = om * h * h
    obh2 = ob * h * h
    th27  = Tcmb / 2.7
    th272 = th27 * th27
    th274 = th272 * th272

    kh = k * h             # Mpc^-1 (paper uses k in Mpc^-1 internally)

    # Eq 2: matter-radiation equality redshift
    z_eq = 2.5e4 * omh2 / th274
    # Eq 3: equality wavenumber [Mpc^-1]
    k_eq = 7.46e-2 * omh2 / th272
    # Eq 4: drag epoch redshift
    b1d = 0.313 * omh2 ** (-0.419) * (1.0 + 0.607 * omh2 ** 0.674)
    b2d = 0.238 * omh2 ** 0.223
    z_d = 1291.0 * omh2 ** 0.251 / (1.0 + 0.659 * omh2 ** 0.828) * (1.0 + b1d * obh2 ** b2d)
    # Eq 5: baryon-to-photon momentum ratio at drag epoch and equality
    R_d  = 31.5e3 * obh2 / th274 / z_d
    R_eq = 31.5e3 * obh2 / th274 / z_eq
    # Eq 6: sound horizon at drag epoch [Mpc]
    s = (2.0 / (3.0 * k_eq)) * jnp.sqrt(6.0 / R_eq) * jnp.log(
        (jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_d + R_eq)) / (1.0 + jnp.sqrt(R_eq))
    )
    # Eq 7: Silk damping scale [Mpc^-1]
    k_silk = 1.6 * obh2 ** 0.52 * omh2 ** 0.73 * (1.0 + (10.4 * omh2) ** (-0.95))

    # Eq 10: scaled wavenumber
    q = kh / (13.41 * k_eq)

    # Eq 11: CDM suppression alpha_c
    a1      = (46.9 * omh2) ** 0.670 * (1.0 + (32.1 * omh2) ** (-0.532))
    a2      = (12.0 * omh2) ** 0.424 * (1.0 + (45.0 * omh2) ** (-0.582))
    alpha_c = a1 ** (-fb) * a2 ** (-(fb ** 3))
    # Eq 12: CDM shift beta_c
    b1c    = 0.944 / (1.0 + (458.0 * omh2) ** (-0.708))
    b2c    = (0.395 * omh2) ** (-0.0266)
    beta_c = 1.0 / (1.0 + b1c * (fc ** b2c - 1.0))

    # Eq 15: G(y) function
    y   = (1.0 + z_eq) / (1.0 + z_d)
    G_y = y * (-6.0 * jnp.sqrt(1.0 + y) + (2.0 + 3.0 * y) * jnp.log(
        (jnp.sqrt(1.0 + y) + 1.0) / (jnp.sqrt(1.0 + y) - 1.0)
    ))
    # Eq 14: baryon suppression alpha_b
    alpha_b = 2.07 * k_eq * s * (1.0 + R_d) ** (-0.75) * G_y
    # Eq 24: beta_b
    beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * jnp.sqrt((17.2 * omh2) ** 2 + 1.0)
    # Eq 23: node shift parameter
    beta_node = 8.41 * omh2 ** 0.435
    # Eq 22: effective sound horizon
    s_tilde = s / (1.0 + (beta_node / (kh * s)) ** 3) ** (1.0 / 3.0)

    def _T0(q_val, ac, bc):
        """Eq 19–20: pressureless T0 with suppression (ac, bc)."""
        C = 14.2 / ac + 386.0 / (1.0 + 69.9 * q_val ** 1.08)
        L = jnp.log(jnp.e + 1.8 * bc * q_val)
        return L / (L + C * q_val ** 2)

    # Eq 18: CDM interpolation weight
    f   = 1.0 / (1.0 + (kh * s / 5.4) ** 4)
    # Eq 17: CDM transfer function
    T_c = f * _T0(q, 1.0, beta_c) + (1.0 - f) * _T0(q, alpha_c, beta_c)

    # Eq 21: baryon transfer function
    T0_11   = _T0(q, 1.0, 1.0)
    j0_tilde = jnp.sinc(kh * s_tilde / jnp.pi)   # sin(x)/x
    T_b = (
        T0_11 / (1.0 + (kh * s / 5.2) ** 2)
        + alpha_b / (1.0 + (beta_b / (kh * s)) ** 3) * jnp.exp(-(kh / k_silk) ** 1.4)
    ) * j0_tilde

    # Eq 16: density-weighted total transfer function
    T  = fb * T_b + fc * T_c
    pk = k ** ns * T ** 2
    pk0 = jnp.interp(jnp.log(jnp.array(0.05)), jnp.log(k), pk)
    return pk / pk0


@jax.jit
def eisenstein_hu_pk_phys(k: jnp.ndarray, theta: dict) -> jnp.ndarray:
    """Eisenstein & Hu (1998) matter power spectrum in physical (Mpc/h)³ units.

    Same transfer function as :func:`eisenstein_hu_pk` but returns
    :math:`P(k)` in physical :math:`(\\mathrm{Mpc}/h)^3` units with the
    correct amplitude derived from the primordial curvature spectrum via the
    Poisson equation in conformal-Newtonian gauge:

    .. math::

        P(k_h, z=0) = D^2(z=0)\\,\\frac{8\\pi^2}{25}
                      \\frac{(c/H_{100})^4}{\\Omega_m^2}
                      A_s \\left(\\frac{h}{k_*}\\right)^{n_s-1}
                      k_h^{n_s}\\,T^2(k_h)

    where :math:`c/H_{100} = 2997.924\\;\\mathrm{Mpc}/h`,
    :math:`k_* = 0.05\\;\\mathrm{Mpc}^{-1}` (CAMB pivot, physical),
    and :math:`A_s = e^{\\ln10^{10}A_s}\\times10^{-10}`.

    :math:`D(z=0)` is the linear growth factor normalised so that
    :math:`D \\to a` during matter domination (EdS limit). For
    :math:`\\Omega_m < 1` flat :math:`\\Lambda`\\CDM, :math:`D(z=0) < 1`
    because dark energy suppresses growth after matter–:math:`\\Lambda` equality.
    This factor is **not** encoded in the EH98 transfer function shape and
    must be included in the amplitude.  It is computed via the exact numerical
    integral

    .. math::

        D(z=0) = \\frac{5\\Omega_m}{2}
                 \\int_0^1 \\frac{\\mathrm{d}a}{[a\\,H(a)/H_0]^3}

    with :math:`H(a) = H_0\\sqrt{\\Omega_m a^{-3} + 1 - \\Omega_m}`.

    Parameters
    ----------
    k : array_like, h/Mpc
    theta : dict — keys: h, Omega_m, Omega_b, n_s, ln10^{10}A_s;
                   optional Omega_cdm (defaults to Omega_m − Omega_b),
                   optional T_cmb [K] (default 2.7255)

    Returns
    -------
    P(k) : jnp.ndarray, (Mpc/h)³

    Accuracy
    --------
    :math:`\\sigma_8` computed from this spectrum agrees with CAMB to
    :math:`< 1\\%` for Planck 2018 parameters.  Residual discrepancies
    reflect the EH98 transfer-function shape error (not the amplitude
    formula).
    """
    h    = theta["h"]
    om   = theta["Omega_m"]
    ob   = theta["Omega_b"]
    ns   = theta["n_s"]
    Tcmb = theta.get("T_cmb", 2.7255)
    lnAs = theta["ln10^{10}A_s"]

    # A_s from ln10^{10}A_s: theta["ln10^{10}A_s"] = ln(10^{10} A_s), so A_s = exp(lnAs)*1e-10
    A_s = jnp.exp(lnAs) * 1e-10

    # Use Omega_cdm directly when present so jax.grad flows through it
    oc   = omega_cdm_cold_equivalent(theta)   # massless-trained: nu lumped into cdm
    fb   = ob / om
    fc   = oc / om
    omh2 = om * h * h
    obh2 = ob * h * h
    th27  = Tcmb / 2.7
    th272 = th27 * th27
    th274 = th272 * th272

    kh = k * h  # Mpc^-1

    z_eq = 2.5e4 * omh2 / th274
    k_eq = 7.46e-2 * omh2 / th272
    b1d = 0.313 * omh2 ** (-0.419) * (1.0 + 0.607 * omh2 ** 0.674)
    b2d = 0.238 * omh2 ** 0.223
    z_d = 1291.0 * omh2 ** 0.251 / (1.0 + 0.659 * omh2 ** 0.828) * (1.0 + b1d * obh2 ** b2d)
    R_d  = 31.5e3 * obh2 / th274 / z_d
    R_eq = 31.5e3 * obh2 / th274 / z_eq
    s = (2.0 / (3.0 * k_eq)) * jnp.sqrt(6.0 / R_eq) * jnp.log(
        (jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_d + R_eq)) / (1.0 + jnp.sqrt(R_eq))
    )
    k_silk = 1.6 * obh2 ** 0.52 * omh2 ** 0.73 * (1.0 + (10.4 * omh2) ** (-0.95))
    q = kh / (13.41 * k_eq)
    a1      = (46.9 * omh2) ** 0.670 * (1.0 + (32.1 * omh2) ** (-0.532))
    a2      = (12.0 * omh2) ** 0.424 * (1.0 + (45.0 * omh2) ** (-0.582))
    alpha_c = a1 ** (-fb) * a2 ** (-(fb ** 3))
    b1c    = 0.944 / (1.0 + (458.0 * omh2) ** (-0.708))
    b2c    = (0.395 * omh2) ** (-0.0266)
    beta_c = 1.0 / (1.0 + b1c * (fc ** b2c - 1.0))
    y   = (1.0 + z_eq) / (1.0 + z_d)
    G_y = y * (-6.0 * jnp.sqrt(1.0 + y) + (2.0 + 3.0 * y) * jnp.log(
        (jnp.sqrt(1.0 + y) + 1.0) / (jnp.sqrt(1.0 + y) - 1.0)
    ))
    alpha_b = 2.07 * k_eq * s * (1.0 + R_d) ** (-0.75) * G_y
    beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * jnp.sqrt((17.2 * omh2) ** 2 + 1.0)
    beta_node = 8.41 * omh2 ** 0.435
    s_tilde = s / (1.0 + (beta_node / (kh * s)) ** 3) ** (1.0 / 3.0)

    def _T0(q_val, ac, bc):
        C = 14.2 / ac + 386.0 / (1.0 + 69.9 * q_val ** 1.08)
        L = jnp.log(jnp.e + 1.8 * bc * q_val)
        return L / (L + C * q_val ** 2)

    f   = 1.0 / (1.0 + (kh * s / 5.4) ** 4)
    T_c = f * _T0(q, 1.0, beta_c) + (1.0 - f) * _T0(q, alpha_c, beta_c)
    T0_11    = _T0(q, 1.0, 1.0)
    j0_tilde = jnp.sinc(kh * s_tilde / jnp.pi)
    T_b = (
        T0_11 / (1.0 + (kh * s / 5.2) ** 2)
        + alpha_b / (1.0 + (beta_b / (kh * s)) ** 3) * jnp.exp(-(kh / k_silk) ** 1.4)
    ) * j0_tilde
    T = fb * T_b + fc * T_c

    # Growth factor D(z=0) normalized so D→a as a→0 (EdS limit).
    # For Ω_m<1 this is < 1 — the suppression is absent from the EH98 transfer
    # function (which is shape-only) and must be folded into the amplitude.
    #
    # This was `ol = 1.0 - om` and a ΛCDM quadrature, with no w0/wa anywhere.
    # Harmless while sigma8 pinned the z=0 amplitude and divided this straight
    # back out; fatal under A_s, where it IS the absolute amplitude and would
    # make dsigma8/dw0 and dsigma8/dwa identically zero while both are free
    # parameters.  growth_d0_eds keeps the ΛCDM value bit-identical and adds the
    # CPL response; see its docstring.
    #
    # Deferred import: hod_mod.core.halo_mass_function imports this module at
    # module scope, so the growth import must not be hoisted.  Inside a
    # @jax.jit function it runs at trace time only.
    from hod_mod.core.growth import growth_d0_eds
    D_z0 = growth_d0_eds(theta)

    # Physical amplitude: P = D²(z=0) × (8π²/25) × (c/H₁₀₀)⁴/Ω_m² × A_s × (h/k_*)^{n_s−1}
    # c/H₁₀₀ = 2997.924 Mpc/h;  k_* = 0.05 Mpc^{-1} (physical pivot)
    _C_H100 = 2997.924  # Mpc/h
    _K_PIVOT = 0.05     # Mpc^{-1}
    A_amp = D_z0 ** 2 * (8.0 * jnp.pi ** 2 / 25.0) * (_C_H100 ** 4) / om ** 2 * A_s * (h / _K_PIVOT) ** (ns - 1.0)
    return A_amp * k ** ns * T ** 2


@jax.jit
def eisenstein_hu_pk_nowiggle(k: jnp.ndarray, theta: dict) -> jnp.ndarray:
    """Eisenstein & Hu (1998) no-wiggle (smooth) power spectrum.

    Implements eqs. (26), (28)–(31) of EH98.  Captures the baryon-induced shape
    suppression through an effective shape parameter :math:`\\Gamma_{\\rm eff}(k)`,
    without acoustic oscillations.  Useful as a smooth reference spectrum.

    :math:`P(k) \\propto k^{n_s} T_0(q_{\\rm eff})^2`, normalised to unity at
    :math:`k = 0.05\\,h\\,\\mathrm{Mpc}^{-1}`.

    Parameters
    ----------
    k : array_like, h/Mpc
    theta : dict — keys: h, Omega_m, Omega_b, n_s; optional T_cmb (K, default 2.7255)
    """
    h    = theta["h"]
    om   = theta["Omega_m"]
    ob   = theta["Omega_b"]
    ns   = theta["n_s"]
    Tcmb = theta.get("T_cmb", 2.7255)

    fb   = ob / om
    omh2 = om * h * h
    obh2 = ob * h * h
    th27  = Tcmb / 2.7
    th272 = th27 * th27

    kh = k * h             # Mpc^-1

    # Eq 26: approximate sound horizon [Mpc]
    s = 44.5 * jnp.log(9.83 / omh2) / jnp.sqrt(1.0 + 10.0 * obh2 ** 0.75)

    # Eq 31: alpha_Gamma (shape suppression amplitude)
    alpha_gamma = (
        1.0
        - 0.328 * jnp.log(431.0 * omh2) * fb
        + 0.38  * jnp.log(22.3  * omh2) * fb ** 2
    )

    # Eq 30: effective shape parameter [h/Mpc]
    Gamma_eff = om * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * kh * s) ** 4))

    # Eq 28: scaled wavenumber (dimensionless)
    q = k * th272 / Gamma_eff

    # Eq 29: zero-baryon transfer function
    L0 = jnp.log(2.0 * jnp.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T  = L0 / (L0 + C0 * q ** 2)

    pk  = k ** ns * T ** 2
    pk0 = jnp.interp(jnp.log(jnp.array(0.05)), jnp.log(k), pk)
    return pk / pk0


class CsstLinearPowerSpectrum:
    """Linear P(k, z) via the CSST CEmulator (Chen+2025, v2.0).

    The emulator is initialised once on instantiation.  Cosmology is set via
    ``set_cosmos`` before each call to ``pk_linear``.  k is in h/Mpc and
    output P(k) is in (Mpc/h)^3.

    Parameter ranges (will raise ValueError if exceeded):

    * Omega_b  ∈ [0.04, 0.06]
    * Omega_m  ∈ [0.24, 0.40]
    * H0       ∈ [60, 80] (inferred as h * 100)
    * n_s      ∈ [0.92, 1.00]
    * A_s      ∈ [1.7e-9, 2.5e-9]
    * w0       ∈ [−1.3, −0.7]
    * wa       ∈ [−0.5, 0.5]
    * m_nu     ∈ [0, 0.3] eV  (default 0.06 eV when not in theta)
    """

    #: The CSST emulator is trained over redshift, so it can supply its own
    #: growth.  Note its stated validity caps (z <= 3, Sigma m_nu <= 0.3 eV).
    has_native_z = True

    def __init__(self):
        try:
            from CEmulator.Emulator import CBaseEmulator
        except ImportError as e:
            raise ImportError("CEmulator not installed — pip install CEmulator") from e
        self._emu = CBaseEmulator()

    @staticmethod
    def _set_cosmos(emu, theta: dict) -> None:
        """Push hod_mod theta dict into the CEmulator cosmology state."""
        import numpy as np
        emu.set_cosmos(
            Omegab=float(theta["Omega_b"]),
            Omegac=float(omega_cdm(theta)),
            H0=float(theta["h"]) * 100.0,
            As=np.exp(float(theta["ln10^{10}A_s"])) * 1e-10,
            ns=float(theta["n_s"]),
            w=float(theta.get("w0", -1.0)),
            wa=float(theta.get("wa", 0.0)),
            mnu=float(mnu_sum(theta)),
        )

    def pk_linear(self, k: jnp.ndarray, z: float, theta: dict) -> jnp.ndarray:
        """Linear P(k) [(Mpc/h)^3] at redshift z via the CSST emulator.

        Parameters
        ----------
        k : array_like, h/Mpc — interpolated onto emulator k-grid
        z : float — must lie in [0, 3]
        theta : dict — hod_mod cosmological parameter dict
        """
        import numpy as np
        self._set_cosmos(self._emu, theta)
        k_np = np.asarray(k)
        pk2d = self._emu.get_pklin(z=float(z), k=k_np)  # shape (1, len(k))
        return jnp.asarray(pk2d[0])
