"""Public-API / clean-break contract tests for the 0.1.0 pipeline layout.

Asserts that every curated symbol is reachable from its new package and from the
top-level ``hod_mod`` namespace, and that the old ``cosmology``/``galaxies``
packages no longer exist (clean break, no shims).
"""
import importlib

import pytest

PACKAGE_EXPORTS = {
    "hod_mod.core": [
        "LinearPowerSpectrum", "NonLinearPowerSpectrum", "WHMSpectrum",
        "HaloMassFunction", "make_hmf", "HaloProfile", "nfw_uk", "einasto_rho",
        "HaloModelPowerSpectrum", "BeyondLinearBiasMead21",
        "comoving_distance", "angular_diameter_distance", "luminosity_distance",
    ],
    "hod_mod.gas": [
        "PressureProfileA10", "PressureProfileDPM", "GasDensityDPM",
        "MetallicityProfileDPM", "temperature_from_dpm", "temperature_from_profiles",
        "xray_cooling_function", "ApecCoolingTable", "m200_to_m500c",
        "ErositaResponse", "load_ecf_tables",
    ],
    "hod_mod.connection": [
        "HODBase", "HODModel", "MoreHODModel", "Kravtsov04HODModel", "AUMHODModel",
        "Guo18ICSMFModel", "Guo19ICSMFModel", "Zacharegkas25HODModel",
        "VanUitert16CSMFModel", "ZuMandelbaum15HODModel", "ZuMandelbaum16QuenchingModel",
        "Leauthaud12HODModel", "SHAMModel", "CLFModel", "VanDenBosch13CLFModel",
        "BASILISKCLFModel",
    ],
    "hod_mod.connection.hod": [
        "MoreHODModel", "ZuMandelbaum15HODModel", "Lange25HODModel",
        "n_cen", "n_sat", "n_total", "shmr_guo18", "sigma_lnmstar_zu15",
        "n_cen_thresh_zu15", "f_quenched",
    ],
    "hod_mod.agn": [
        "XrayAGNModel", "HamAGNModel", "obscured_fraction", "HODAgnModel",
        "DutyCycleAGNModel", "compute_w_agn_kernel", "load_zm15_map_params",
    ],
    "hod_mod.observables": [
        "HODClusteringPrediction", "FullHaloModelPrediction",
        "NonLinearHaloModelPrediction", "HODProjectedCorrelation",
        "projected_correlation_function", "HaloModelCrossSpectra",
        "psf_window_ell", "psf_king_profile", "psf_king_window_ell",
        "ClusterGalaxyCrossCorrelation", "NLAModel", "TATTModel",
        "make_baryon_fraction",
    ],
    "hod_mod.fitting": [
        "FitConfig", "WpFitConfig", "JointFitConfig", "WpFitFITSConfig",
        "load_config", "load_joint_config", "load_fits_config",
        "WpFitter", "JointFitter", "DeltaSigmaFitter", "WpFitterFITS", "HOD_MODELS",
    ],
}

TOP_LEVEL_EXPORTS = [
    "LinearPowerSpectrum", "HaloMassFunction", "make_hmf", "HaloProfile",
    "HODBase", "MoreHODModel", "ZuMandelbaum15HODModel", "Leauthaud12HODModel",
    "HODClusteringPrediction", "FullHaloModelPrediction", "SHAMModel", "CLFModel",
    "FitConfig", "WpFitter", "JointFitter", "load_config",
]


@pytest.mark.parametrize("pkg,symbols", PACKAGE_EXPORTS.items())
def test_package_exports(pkg, symbols):
    mod = importlib.import_module(pkg)
    missing = [s for s in symbols if not hasattr(mod, s)]
    assert not missing, f"{pkg} is missing {missing}"


def test_top_level_namespace():
    import hod_mod
    missing = [s for s in TOP_LEVEL_EXPORTS if not hasattr(hod_mod, s)]
    assert not missing, f"hod_mod top-level missing {missing}"


def test_version_is_0_1_x():
    import hod_mod
    assert hod_mod.__version__.startswith("0.1.")


@pytest.mark.parametrize("old", ["hod_mod.cosmology", "hod_mod.galaxies"])
def test_old_packages_removed(old):
    """Clean break: the pre-refactor package paths must not import."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)
