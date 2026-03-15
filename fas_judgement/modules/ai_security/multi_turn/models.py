"""
Multi-Turn Domain Models
-------------------------
WHY: Re-exports the dataclass models from service.py under a stable
     models.py name. This follows the DDD convention of having explicit
     model files so callers have a stable import path:
       `from .models import Session, Turn, PhaseResult`

LAYER: Module/Models — re-exports from service.
"""

from .service import (
    Session,
    Turn,
    Mode,
    PhaseResult,
    ADVANCE_ON,
    RETRY_ON,
    PIVOT_ON,
    ABORT_ON,
    PHASE_DESCRIPTIONS,
    CATEGORY_GOALS,
)

__all__ = [
    "Session",
    "Turn",
    "Mode",
    "PhaseResult",
    "ADVANCE_ON",
    "RETRY_ON",
    "PIVOT_ON",
    "ABORT_ON",
    "PHASE_DESCRIPTIONS",
    "CATEGORY_GOALS",
]
