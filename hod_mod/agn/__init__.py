"""AGN X-ray emission models for the galaxy × X-ray cross-correlation.

Four interchangeable prescriptions for placing X-ray-emitting AGN in haloes:
a direct L_X(M) model, halo-abundance matching (HAM), an HOD-based occupation,
and a duty-cycle model.
"""

from .xray import XrayAGNModel
from .ham import HamAGNModel, obscured_fraction
from .hod import HODAgnModel
from .duty_cycle import DutyCycleAGNModel, compute_w_agn_kernel, load_zm15_map_params

__all__ = [
    "XrayAGNModel",
    "HamAGNModel",
    "obscured_fraction",
    "HODAgnModel",
    "DutyCycleAGNModel",
    "compute_w_agn_kernel",
    "load_zm15_map_params",
]
