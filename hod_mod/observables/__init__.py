"""Observable pipelines built on the core engine and the galaxy/gas/AGN models.

Three families share one halo-model backbone:

* **Galaxy clustering & lensing** — :class:`HODClusteringPrediction`,
  :class:`FullHaloModelPrediction` predict :math:`w_p(r_p)` and
  :math:`\\Delta\\Sigma(R)`, with intrinsic-alignment and baryon-fraction systematics.
* **Galaxy × X-ray** and **Galaxy × thermal SZ** — :class:`HaloModelCrossSpectra`
  is the shared cross-power engine (``P_{g,y}`` for tSZ, ``P_{g,X}`` for soft X-ray)
  with Limber/Abel projections.
* **Cluster × galaxy** — :class:`ClusterGalaxyCrossCorrelation`.
"""

from .clustering import (
    HODClusteringPrediction,
    FullHaloModelPrediction,
    NonLinearHaloModelPrediction,
    HODProjectedCorrelation,
    projected_correlation_function,
)
from .cross_spectra import (
    HaloModelCrossSpectra,
    psf_window_ell,
    psf_king_profile,
    psf_king_window_ell,
)
from .cross_clustering import ClusterGalaxyCrossCorrelation
from .intrinsic_alignment import NLAModel, TATTModel
from .baryon_fraction import (
    make_baryon_fraction,
    BaryonFractionSigmoid,
    BaryonFractionPowerLaw,
    BaryonFractionUpturn,
)

__all__ = [
    "HODClusteringPrediction",
    "FullHaloModelPrediction",
    "NonLinearHaloModelPrediction",
    "HODProjectedCorrelation",
    "projected_correlation_function",
    "HaloModelCrossSpectra",
    "psf_window_ell",
    "psf_king_profile",
    "psf_king_window_ell",
    "ClusterGalaxyCrossCorrelation",
    "NLAModel",
    "TATTModel",
    "make_baryon_fraction",
    "BaryonFractionSigmoid",
    "BaryonFractionPowerLaw",
    "BaryonFractionUpturn",
]
