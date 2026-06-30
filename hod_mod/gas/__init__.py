"""Hot-gas / ICM fields shared by the X-ray and thermal-SZ cross-correlations.

Pressure profiles (Arnaud+2010, DPM) feed the tSZ Compton-y signal; the gas
density, cooling/emissivity (APEC) and metallicity profiles feed the soft X-ray
emissivity.  The eROSITA instrument response converts intrinsic emission to
observed count rates.  These objects are consumed by
:class:`hod_mod.observables.cross_spectra.HaloModelCrossSpectra`.
"""

from .conversions import (
    m200_to_m500c,
    _gnfw_f_params,
    _RHO_CRIT0,
    _MPC_CM,
    _SIGMA_T_OVER_ME_C2,
)
from .pressure import PressureProfileA10, PressureProfileDPM
from .density import GasDensityDPM
from .cooling import (
    temperature_from_dpm,
    temperature_from_profiles,
    xray_cooling_function,
    ApecCoolingTable,
)
from .metallicity import MetallicityProfileDPM
from .erosita_response import ErositaResponse, load_ecf_tables

__all__ = [
    "PressureProfileA10",
    "PressureProfileDPM",
    "GasDensityDPM",
    "MetallicityProfileDPM",
    "temperature_from_dpm",
    "temperature_from_profiles",
    "xray_cooling_function",
    "ApecCoolingTable",
    "m200_to_m500c",
    "ErositaResponse",
    "load_ecf_tables",
]
