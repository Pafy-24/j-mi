"""RM-managed MI gradient worker.

The worker owns model execution and child-facet gradients only. Optimizer,
scheduler, campaign state and result selection deliberately live in Trainer.
"""

from .engine import FacetGradientEngine, GradientResult
from .service import MIWorkerService

__all__ = ["FacetGradientEngine", "GradientResult", "MIWorkerService"]
