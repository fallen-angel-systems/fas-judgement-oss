"""
Core Enums
----------
WHY: Enums live in the domain core because they represent business
     vocabulary — the shared language between all layers.
     Using str-enums lets them serialise directly to JSON without
     a custom encoder, and means DB rows can be compared as strings.

LAYER: Core domain — no imports from other fas_judgement modules.
"""

from enum import Enum


# === SECTION: VERDICT ENUMS === #

class VerdictType(str, Enum):
    """Outcome of a single-shot attack attempt."""
    BYPASS = "BYPASS"      # Model complied — the attack worked
    BLOCKED = "BLOCKED"    # Model refused / safety filter caught it
    PARTIAL = "PARTIAL"    # Model partially engaged — worth investigating
    ERROR = "ERROR"        # Network/HTTP error, not a model decision


class Severity(str, Enum):
    """Severity of a security finding, mirroring CVSS-style vocabulary."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


# === SECTION: MULTI-TURN ENUMS === #

class PhaseResult(str, Enum):
    """
    Score assigned to a single turn in a multi-turn attack sequence.
    The orchestrator uses these to decide whether to advance, retry, or abort.

    Ordering (best → worst for attacker):
      COMPLY > PARTIAL > DEFLECT > REFUSE > DETECT
    """
    COMPLY = "comply"     # Target shared sensitive info / accepted premise
    PARTIAL = "partial"   # Target engaged but held back the most sensitive details
    DEFLECT = "deflect"   # Target changed subject without explicit refusal
    REFUSE = "refuse"     # Target explicitly declined
    DETECT = "detect"     # Target identified this as an attack


class Difficulty(str, Enum):
    """Relative difficulty of an attack category for the red team."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


# === SECTION: TIER ENUMS === #

class Tier(str, Enum):
    """User subscription tier — drives feature access throughout the app."""
    FREE = "free"
    PRO = "pro"
    ELITE_HOME = "elite_home"
    ELITE_BUSINESS = "elite_business"
    ADMIN = "admin"

    def is_elite(self) -> bool:
        return self in (Tier.ELITE_HOME, Tier.ELITE_BUSINESS, Tier.ADMIN)

    def is_pro_or_above(self) -> bool:
        return self != Tier.FREE
