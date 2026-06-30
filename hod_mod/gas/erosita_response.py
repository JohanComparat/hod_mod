"""eROSITA DR1 energy-conversion factor (ECF) from the real ARF + RMF.

Folds a rest-frame source spectrum through the combined TM1-7 effective area
(ARF) and the 0.5-2 keV in-band redistribution efficiency g(E) of the RMF, with
Galactic absorption, to give the **ECF**:

    ECF = (observed eROSITA 0.5-2 keV count rate)
          / (intrinsic rest-frame 0.5-2 keV energy flux)      [cts/s per erg/s/cm^2]

The ECF is distance-independent: the numerator (observed band count rate) and
denominator (rest-frame band energy flux) share one spectral normalisation, so
1/(4 pi d_L^2) cancels.  Only the K-correction (the (1+z) band shift), Galactic
absorption, and the instrument response survive — exactly the fixed, known eROSITA
specifications.  The cosmological surface-brightness dimming is applied by the
cross-power projection, not here.

The combined response is distilled in
``hod_mod/data/erosita/dr1_response_tm1-7_0p5-2keV.npz`` (see
``scripts/galaxies/build_erosita_response.py``).  Validated: the AGN power-law
(Gamma=1.9, N_H=3e20) gives 1/ECF = 1.05e-12 erg/s/cm^2 per cts/s, matching the
standard eROSITA 0.5-2 keV count-rate-to-flux conversion.
"""
from __future__ import annotations

import os

import numpy as np

_KEV2ERG = 1.602176634e-9
_DEFAULT_NPZ = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "erosita",
    "dr1_response_tm0_survey_0p5-2keV.npz"))
_ECF_TABLE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "erosita"))


def load_ecf_tables(sample: str):
    """Load the precomputed per-component ECF tables for a GALxEVT sample.

    Returns ``(ecf_gas_interp, ecf_agn, ecf_fixed)`` where ``ecf_gas_interp`` is a
    callable ``T_keV -> ECF_gas(T)`` [cts/s per erg/s/cm²] (log-T interpolation,
    flat extrapolation), ``ecf_agn`` is the AGN power-law ECF, and ``ecf_fixed``
    is the GALxEVT pipeline's fixed conversion ``ARF_1keV/C``.  Built by
    ``scripts/galaxies/build_ecf_tables.py``.
    """
    from scipy.interpolate import interp1d
    d = np.load(os.path.join(_ECF_TABLE_DIR, f"ecf_tables_{sample}.npz"))
    kT = d["kT_grid"]; eg = d["ecf_gas"]
    interp = interp1d(np.log10(kT), eg, bounds_error=False,
                      fill_value=(float(eg[0]), float(eg[-1])))
    gas = lambda T_keV: interp(np.log10(np.clip(np.asarray(T_keV, float), 1e-3, None)))
    return gas, float(d["ecf_agn"]), float(d["ecf_fixed"])


class ErositaResponse:
    """Combined TM1-7 eROSITA DR1 response → energy-conversion factors.

    Parameters
    ----------
    response_npz : str | None
        Path to the distilled response artifact (ARF + in-band RMF efficiency).
    """

    def __init__(self, response_npz: str | None = None):
        d = np.load(response_npz or _DEFAULT_NPZ)
        self.energ_lo = d["energ_lo"]
        self.energ_hi = d["energ_hi"]
        self.e_obs = 0.5 * (self.energ_lo + self.energ_hi)   # detector grid [keV]
        self.de_obs = self.energ_hi - self.energ_lo
        self.arf = d["arf_comb"]                              # cm^2 (TM1-7)
        self.g = d["g_inband"]                                # 0.5-2 keV fraction
        self.band = tuple(float(x) for x in d["band"])
        self._tbabs_cache: dict = {}

    # -- absorption ---------------------------------------------------------
    def _transmission(self, nH: float) -> np.ndarray:
        """tbabs transmission at the observed detector energies (cached)."""
        key = round(float(nH), 6)
        if key not in self._tbabs_cache:
            try:
                from soxs.spectra import get_tbabs_absorb
                T = np.asarray(get_tbabs_absorb(self.e_obs, nH), dtype=float)
            except Exception:
                # Morrison & McCammon-like fallback (rarely used)
                sigma = 2.0e-22 * (self.e_obs) ** -2.5
                T = np.exp(-nH * 1e22 * sigma)
            self._tbabs_cache[key] = np.clip(T, 0.0, 1.0)
        return self._tbabs_cache[key]

    # -- core folding -------------------------------------------------------
    def ecf_from_rest(self, e_rest, s_rest, z: float, nH: float = 0.03,
                      absorb: bool = True) -> float:
        """ECF for a rest-frame photon spectrum ``s_rest = dN/dE`` (any norm).

        e_rest : keV ; nH : 1e22 cm^-2 ; returns cts/s per erg/s/cm^2.
        """
        e_rest = np.asarray(e_rest, float); s_rest = np.asarray(s_rest, float)
        m = (e_rest >= self.band[0]) & (e_rest <= self.band[1])
        de_r = np.gradient(e_rest)
        F_X = np.sum(s_rest[m] * e_rest[m] * _KEV2ERG * de_r[m])      # erg/s/cm^2
        s_at = np.interp((1.0 + z) * self.e_obs, e_rest, s_rest,
                         left=0.0, right=0.0)
        phot_obs = s_at * (1.0 + z)                                  # redshifted
        if absorb:
            phot_obs = phot_obs * self._transmission(nH)
        CR = np.sum(phot_obs * self.arf * self.g * self.de_obs)      # cts/s
        return float(CR / F_X)

    # -- AGN: absorbed power law -------------------------------------------
    def ecf_powerlaw(self, photon_index: float = 1.9, z: float = 0.0,
                     nH: float = 0.03, absorb: bool = True) -> float:
        e = np.logspace(np.log10(0.05), np.log10(4.0), 4000)
        return self.ecf_from_rest(e, e ** (-photon_index), z, nH, absorb)

    # -- gas: APEC plasma ---------------------------------------------------
    def ecf_apec(self, kT: float, Z: float = 0.3, z: float = 0.0,
                 nH: float = 0.03, apec=None, absorb: bool = True) -> float:
        if apec is None:
            import soxs
            apec = soxs.ApecGenerator(0.05, 4.0, 6000, apec_vers="3.1.3",
                                      broadening=False)
        sp = apec.get_spectrum(kT, Z, 0.0, 1.0)        # rest-frame, z=0
        return self.ecf_from_rest(sp.emid.value, sp.flux.value, z, nH, absorb)

    def ecf_apec_table(self, z: float, nH: float = 0.03, Z: float = 0.3,
                       kT_grid=None):
        """Return (kT_grid, ecf_grid) and a log-kT interpolator for the gas ECF."""
        from scipy.interpolate import interp1d
        import soxs
        if kT_grid is None:
            kT_grid = np.logspace(np.log10(0.1), np.log10(15.0), 24)
        apec = soxs.ApecGenerator(0.05, 4.0, 6000, apec_vers="3.1.3",
                                  broadening=False)
        ecf = np.array([self.ecf_apec(kT, Z, z, nH, apec=apec) for kT in kT_grid])
        interp = interp1d(np.log10(kT_grid), ecf, bounds_error=False,
                          fill_value=(ecf[0], ecf[-1]))
        return kT_grid, ecf, interp
