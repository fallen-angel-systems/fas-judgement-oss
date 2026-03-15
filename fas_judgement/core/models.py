"""
Core Domain Models
------------------
WHY: Dataclasses here are the "nouns" of the system — Pattern, ScanResult,
     Finding, etc. They're plain Python with no ORM / framework dependency.

     Using dataclasses (rather than Pydantic) for domain objects means the
     core can be unit-tested without FastAPI. Pydantic models live in
     http/schemas.py (the HTTP boundary).

LAYER: Core domain — imports only stdlib and core enums.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .enums import VerdictType, Severity, PhaseResult


# === SECTION: SINGLE-SHOT ATTACK MODELS === #

@dataclass
class Pattern:
    """
    A single prompt-injection attack pattern from the library.

    WHY `tier` field: patterns are gated by subscription. The service layer
    checks the user's tier against pattern.tier to decide what to expose.
    """
    id: str
    text: str
    category: str
    tier: str = "free"
    name: str = ""
    severity: str = "medium"
    technique: str = ""
    description: str = ""
    source: str = "library"   # "library" | "custom" | "submission"

    def is_locked_for(self, user_tier: str) -> bool:
        """Return True if this pattern requires a higher tier than the user has."""
        tier_order = {"free": 0, "pro": 1, "elite": 2, "elite_home": 2, "elite_business": 2, "admin": 3}
        return tier_order.get(self.tier, 0) > tier_order.get(user_tier, 0)


@dataclass
class Finding:
    """
    A single attack result that produced a noteworthy verdict (BYPASS/PARTIAL).
    Used by the report generator.
    """
    pattern_id: str
    pattern_text: str
    category: str
    verdict: VerdictType
    response_body: str
    response_status: int
    response_time_ms: float
    llm_reason: str = ""
    session_id: str = ""


@dataclass
class ScanResult:
    """
    Result of scanning one pattern against a target.
    Produced by the runner, consumed by the scorer and persisted by the repo.
    """
    pattern_id: str
    pattern_text: str
    category: str
    request_body: str
    response_status: int
    response_body: str
    response_time_ms: float
    verdict: VerdictType
    llm_reason: str = ""
    session_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class AttackSession:
    """
    Tracks aggregate stats for a single-shot attack run.
    Persisted in the `sessions` table.
    """
    id: str
    target_url: str
    method: str
    total_patterns: int
    blocked: int = 0
    partial: int = 0
    bypassed: int = 0
    errors: int = 0
    user_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def bypass_rate(self) -> float:
        if self.total_patterns == 0:
            return 0.0
        return round((self.bypassed / self.total_patterns) * 100, 1)


# === SECTION: MULTI-TURN MODELS === #

@dataclass
class Turn:
    """Single turn in a multi-turn conversation."""
    turn_number: int
    phase: int
    attack_message: str
    target_response: Optional[str] = None
    score: Optional[PhaseResult] = None
    score_reason: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class MultiTurnSession:
    """
    Full state of a multi-turn attack session.
    This is the aggregate root for the multi-turn domain.
    """
    id: str
    category: str
    category_name: str
    mode: str = "auto"
    target_url: Optional[str] = None
    target_name: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = ""
    current_phase: int = 1
    max_phases: int = 5
    turns: list = field(default_factory=list)
    used_messages: dict = field(default_factory=dict)
    status: str = "active"
    retries_this_phase: int = 0
    max_retries_per_phase: int = 3
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    findings: list = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def current_phase_key(self) -> str:
        return f"phase_{self.current_phase}"


# === SECTION: LICENSE MODELS === #

@dataclass
class LicenseInfo:
    """Deserialized license validation result returned from the license server."""
    valid: bool
    tier: str
    features: dict = field(default_factory=dict)
    expires_at: Optional[str] = None
    seats: int = 1
    seats_used: int = 0
    error: Optional[str] = None
