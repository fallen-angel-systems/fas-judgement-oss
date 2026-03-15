"""
Core Abstract Interfaces
------------------------
WHY: Abstract base classes define the contracts that modules must fulfil.
     This enforces the Dependency Inversion Principle:
       - High-level modules (HTTP layer) depend on abstractions, not concretions
       - Low-level modules (HTTP transport, Ollama scorer) implement the ABCs

     Net result: you can swap the HTTP transport for a Discord transport
     without touching the orchestrator or HTTP routes.

LAYER: Core domain — imports only stdlib and core models.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from .models import ScanResult


# === SECTION: TRANSPORT ABC === #

class BaseTransport(ABC):
    """
    Abstract transport: anything that can send an attack message and
    return the target's response.

    Concrete implementations live in fas_judgement/transport/.
    """
    transport_type: str = "unknown"

    @abstractmethod
    async def send(self, message: str) -> str:
        """Send attack message to the target. Returns the target's response text."""
        ...

    @abstractmethod
    async def check_connection(self) -> dict:
        """
        Verify the target is reachable.
        Returns a dict with at least {"connected": bool}.
        """
        ...

    def to_dict(self) -> dict:
        """Serialise transport config for session export."""
        return {"type": self.transport_type}


# === SECTION: SCANNER ABC === #

class BaseScanner(ABC):
    """
    Abstract scanner: sends a list of patterns to a target via a transport
    and yields ScanResult objects.
    """

    @abstractmethod
    async def run(
        self,
        patterns: list,
        transport: BaseTransport,
        session_id: str,
        delay_ms: int = 200,
    ) -> AsyncIterator[ScanResult]:
        """
        Execute patterns against a target.
        Yields ScanResult as each pattern completes.
        WHY AsyncIterator: SSE streaming — results go to the browser
        as they arrive rather than buffering the whole run.
        """
        ...


# === SECTION: SCORER ABC === #

class BaseScorer(ABC):
    """
    Abstract scorer: classifies a (attack, response) pair as BYPASS/BLOCKED/PARTIAL.

    WHY abstract: we have two implementations:
      1. KeywordScorer   — fast, deterministic, no external calls
      2. LLMScorer       — Ollama-based, smarter but slower
    The scanner can swap scorers without caring about the underlying mechanism.
    """

    @abstractmethod
    async def classify(
        self,
        attack_text: str,
        response_text: str,
        status_code: int,
    ) -> tuple[str, str]:
        """
        Returns (verdict, reason).
        verdict ∈ {"BYPASS", "BLOCKED", "PARTIAL", "ERROR"}
        reason  = human-readable explanation (for the report)
        """
        ...


# === SECTION: MODULE ABC === #

class BaseModule(ABC):
    """
    Abstract module: a pluggable feature group (e.g. AI Security, Campaigns).

    Modules register their own HTTP routers, which get mounted into the
    FastAPI app at startup. This follows the Vertical Slice / Feature Module
    pattern — each module owns its own routes, services, and data.
    """
    name: str = "unknown"
    description: str = ""

    @abstractmethod
    def get_router(self):
        """Return the FastAPI APIRouter for this module's endpoints."""
        ...

    def on_startup(self):
        """Optional hook called during app startup."""
        pass

    def on_shutdown(self):
        """Optional hook called during app shutdown."""
        pass
