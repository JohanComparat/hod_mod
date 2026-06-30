"""HOD model dispatch table (name -> class) used by the fitters."""
from hod_mod.connection.hod import (
    HODModel, Kravtsov04HODModel, MoreHODModel, Guo18ICSMFModel, Guo19ICSMFModel,
    Zacharegkas25HODModel, VanUitert16CSMFModel, ZuMandelbaum15HODModel,
    Leauthaud12HODModel, Lange25HODModel,
)
from hod_mod.connection.clf import CLFModel



HOD_MODELS: dict = {
    "HODModel":                  HODModel,
    "Kravtsov04HODModel":        Kravtsov04HODModel,
    "MoreHODModel":              MoreHODModel,
    "Guo18ICSMFModel":           Guo18ICSMFModel,
    "Guo19ICSMFModel":           Guo19ICSMFModel,
    "Zacharegkas25HODModel":     Zacharegkas25HODModel,
    "VanUitert16CSMFModel":      VanUitert16CSMFModel,
    "ZuMandelbaum15HODModel":    ZuMandelbaum15HODModel,
    "Leauthaud12HODModel":       Leauthaud12HODModel,
    "Lange25HODModel":           Lange25HODModel,
    "CLFModel":                  CLFModel,
}
