"""Best-fit MAP parameter loader for showcase and fitting scripts.

Tries to load MAP results from ``results/bgs_multiprobe/``.  Falls back to
each model's ``default_params()`` when no fitted result is on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from hod_mod.cosmology.power_spectrum import LinearPowerSpectrum
from hod_mod.galaxies.hod import (
    HODModel,
    Kravtsov04HODModel,
    MoreHODModel,
    Guo18ICSMFModel,
    Guo19ICSMFModel,
    Zacharegkas25HODModel,
    VanUitert16CSMFModel,
    ZuMandelbaum15HODModel,
    Leauthaud12HODModel,
)

_DEFAULT_THETA = LinearPowerSpectrum.default_cosmology()

_MODEL_CLASS = {
    "zheng2007":       HODModel,
    "aum":             HODModel,
    "kravtsov2004":    Kravtsov04HODModel,
    "more2015":        MoreHODModel,
    "guo2018":         Guo18ICSMFModel,
    "guo2019":         Guo19ICSMFModel,
    "zacharegkas25":   Zacharegkas25HODModel,
    "vanuitert16":     VanUitert16CSMFModel,
    "zu_mandelbaum15": ZuMandelbaum15HODModel,
    "leauthaud12":     Leauthaud12HODModel,
}

_RESULTS_ROOT = Path(__file__).parents[4] / "results" / "bgs_multiprobe"


def load_map_params(key: str) -> tuple[dict, dict]:
    """Return (theta_cosmo, hod_params) for a given model key.

    Loads from ``results/bgs_multiprobe/mstar10.0_wp_{key}_nfw_rp300/map_result.json``
    when available; otherwise returns Planck 2018 defaults.

    Parameters
    ----------
    key : str
        Model identifier, e.g. ``"zheng2007"``, ``"more2015"``.

    Returns
    -------
    theta : dict
        Cosmological parameter dict (same schema as
        :meth:`~hod_mod.cosmology.power_spectrum.LinearPowerSpectrum.default_cosmology`).
    hod_params : dict
        HOD parameter dict (same schema as the model's ``default_params()``).
    """
    json_path = _RESULTS_ROOT / f"mstar10.0_wp_{key}_nfw_rp300" / "map_result.json"

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        theta = {k: v for k, v in data.items() if k in _DEFAULT_THETA}
        theta = {**_DEFAULT_THETA, **theta}
        hod_params = {k: v for k, v in data.items() if k not in _DEFAULT_THETA}
        return theta, hod_params

    cls = _MODEL_CLASS.get(key)
    if cls is None:
        raise KeyError(f"Unknown model key '{key}'. Available: {list(_MODEL_CLASS)}")

    return dict(_DEFAULT_THETA), cls.default_params()
