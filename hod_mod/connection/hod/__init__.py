"""Halo Occupation Distribution (HOD) models in JAX.

Implements four HOD/ICSMF parametrisations:

**Zheng+2007** (ApJ 667, 760) — standard erfc HOD:

.. math::

    \\langle N_{\\rm cen}(M) \\rangle = \\frac{1}{2}\\,
    {\\rm erfc}\\!\\left[\\frac{\\log_{10}M_{\\rm min} - \\log_{10}M}{\\sigma_{\\log M}}\\right]

    \\langle N_{\\rm sat}(M) \\rangle = \\langle N_{\\rm cen}(M) \\rangle
    \\left(\\frac{M - M_0}{M_1}\\right)^\\alpha

**More+2015** (arXiv:1407.1856) — BOSS CMASS HOD with linear incompleteness function:

.. math::

    f_{\\rm inc}(M) = \\min\\!\\left[1,\\,\\max\\!\\left(0,\\,
        1 + \\alpha_{\\rm inc}\\,(\\log_{10}M - \\log_{10}M_{\\rm inc})\\right)\\right]

    \\langle N_{\\rm cen}(M) \\rangle = \\frac{f_{\\rm inc}(M)}{2}\\,
    {\\rm erfc}\\!\\left[\\frac{\\log_{10}M_{\\rm min} - \\log_{10}M}{\\sigma_{\\log M}}\\right]

    \\langle N_{\\rm sat}(M) \\rangle = \\langle N_{\\rm cen}(M) \\rangle
    \\left(\\frac{M - \\kappa M_{\\rm min}}{M_1}\\right)^\\alpha

**Guo+2018** (ApJ 858, 30) — Incomplete Conditional Stellar Mass Function (ICSMF)
using a broken power-law stellar-to-halo mass relation:

.. math::

    \\langle M_*(M) \\rangle = M_{*0}
    \\left(\\frac{M}{M_1}\\right)^{\\alpha+\\beta}
    \\left(1 + \\frac{M}{M_1}\\right)^{-\\beta}

Completeness functions (separate for centrals I and satellites II):

.. math::

    c(M_*) = \\frac{f}{2}\\left[1 + {\\rm erf}
    \\left(\\frac{\\log_{10}M_* - \\log_{10}M_{*,\\rm min}}{\\sigma_c}\\right)\\right]

**Guo+2019** (ApJ 871, 147) — ICSMF for eBOSS ELGs with quenched fraction:

.. math::

    f_q(M) = \\frac{1}{1 + M/M_q}, \\qquad f_{\\rm sf}(M) = 1 - f_q(M)

**Zu & Mandelbaum 2015** (arXiv:1505.02781, Paper I) — iHOD stellar-to-halo mass relation
with Behroozi+2010 inverse SHMR, mass-dependent log-normal scatter, and power-law
satellite occupation:

.. math::

    M_h = M_1 \\left(\\frac{M_*}{M_{*0}}\\right)^\\beta
    \\exp\\!\\left[\\frac{(M_*/M_{*0})^\\delta}{1+(M_*/M_{*0})^{-\\gamma}} - \\frac{1}{2}\\right]

    \\langle N_{\\rm sat}^{>M_*}\\rangle(M_h) = \\langle N_{\\rm cen}^{>M_*}\\rangle(M_h)
    \\left(\\frac{M_h}{M_{\\rm sat}}\\right)^{\\alpha_{\\rm sat}}
    \\exp\\!\\left(-\\frac{M_{\\rm cut}}{M_h}\\right)

**Zu & Mandelbaum 2016/2017** (arXiv:1509.06374, 1703.09219) — halo quenching model:

.. math::

    f_{\\rm red,c}(M_h) = 1 - \\exp\\!\\left[-\\left(M_h/M_h^{qc}\\right)^{\\mu_c}\\right]

    f_{\\rm red,s}(M_h) = 1 - \\exp\\!\\left[-\\left(M_h/M_h^{qs}\\right)^{\\mu_s}\\right]
"""

from .base import n_cen, n_sat, n_total, HODBase, HODModel
from .kravtsov04 import (
    n_sat_kravtsov04, n_sat_aum, n_total_kravtsov04, n_total_aum,
    Kravtsov04HODModel, AUMHODModel,
)
from .more15 import (
    _incompleteness_more15, n_cen_more15, n_sat_more15, n_total_more15, MoreHODModel,
    n_cen_more15_const_finc, n_sat_more15_const_finc, MoreConstFincHODModel,
)
from .guo import (
    shmr_guo18, completeness_guo18, n_cen_guo18, n_sat_guo18, n_total_guo18,
    Guo18ICSMFModel, f_quenched, f_starforming,
    n_cen_guo19, n_sat_guo19, n_total_guo19, Guo19ICSMFModel,
)
from .zacharegkas25 import (
    _f_kravtsov, shmr_zacharegkas25, _inverse_shmr_z25,
    n_cen_thresh_z25, n_sat_thresh_z25, n_cen_bin_z25, n_sat_bin_z25, n_total_bin_z25,
    Zacharegkas25HODModel,
)
from .vanuitert16 import (
    _mean_stellar_mass_c_vanuitert16, n_cen_vanuitert16, n_sat_vanuitert16,
    n_total_vanuitert16, VanUitert16CSMFModel,
)
from .zumandelbaum15 import (
    _mh_from_mstar_zu15, _mstar_from_mh_zu15, sigma_lnmstar_zu15,
    n_cen_thresh_zu15, n_sat_thresh_zu15, n_total_thresh_zu15, ZuMandelbaum15HODModel,
    f_red_cen_zu16, f_red_sat_zu16, ZuMandelbaum16QuenchingModel,
)
from .leauthaud12 import (
    _mh_from_mstar_leauthaud12, _mstar_from_mh_leauthaud12,
    n_cen_leauthaud12, n_sat_leauthaud12, Leauthaud12HODModel,
)
from .lange25 import Lange25HODModel
