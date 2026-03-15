"""
HTTP Dependencies (OSS Edition)
---------------------------------
WHY: Shared dependencies for route handlers — currently just the rate limiter.

     OSS simplification: no JWT, no OAuth, no API key auth, no user lookup.
     You're running this on your own machine — you ARE the user.
     The rate limiter here is global (not per-user) to prevent accidental
     abuse of external targets.

LAYER: HTTP — imports FastAPI, stdlib.
"""

import time
from typing import Optional


# === SECTION: RATE LIMITER === #

class RateLimiter:
    """
    In-memory global rate limiter for attack and scan endpoints.

    WHY global (not per-user): OSS has no users. Limits exist to prevent
    accidental hammering of a target or a misconfigured delay value.

    WHY in-memory (not Redis): single-instance local deployment.
    If you somehow need distributed rate limiting, you're probably using
    the wrong product — check out the Elite hosted version.
    """

    def __init__(self):
        # Track active attack sessions (session_id → True)
        self.active_attacks: set = set()
        # Timestamps of recent attacks (sliding window)
        self.attack_history: list = []
        # Timestamps of recent scans
        self.scan_history: list = []

        self.MAX_CONCURRENT_ATTACKS = 3
        self.MAX_ATTACKS_PER_HOUR = 20       # Generous for local use
        self.MAX_SCANS_PER_MINUTE = 15
        self.MIN_DELAY_MS = 100              # SSRF mitigation: never go below 100ms

    def _clean_timestamps(self, timestamps: list, window_seconds: float) -> list:
        """Remove timestamps older than the sliding window."""
        cutoff = time.time() - window_seconds
        return [t for t in timestamps if t > cutoff]

    def can_attack(self) -> tuple[bool, Optional[str]]:
        """Check if a new attack can start. Returns (allowed, reason_if_not)."""
        if len(self.active_attacks) >= self.MAX_CONCURRENT_ATTACKS:
            return False, f"Maximum {self.MAX_CONCURRENT_ATTACKS} concurrent attacks. Wait for one to finish."

        self.attack_history = self._clean_timestamps(self.attack_history, 3600)
        if len(self.attack_history) >= self.MAX_ATTACKS_PER_HOUR:
            return False, f"Maximum {self.MAX_ATTACKS_PER_HOUR} attacks per hour. Try again in a bit."

        return True, None

    def start_attack(self, session_id: str) -> None:
        """Register a session as active."""
        self.active_attacks.add(session_id)
        self.attack_history.append(time.time())

    def end_attack(self, session_id: str) -> None:
        """Deregister a session when it completes or is aborted."""
        self.active_attacks.discard(session_id)

    def can_scan(self) -> tuple[bool, Optional[str]]:
        """Check if a scan is allowed."""
        self.scan_history = self._clean_timestamps(self.scan_history, 60)
        if len(self.scan_history) >= self.MAX_SCANS_PER_MINUTE:
            return False, f"Maximum {self.MAX_SCANS_PER_MINUTE} scans per minute. Slow down."
        return True, None

    def record_scan(self) -> None:
        """Record a scan timestamp."""
        self.scan_history.append(time.time())

    def enforce_min_delay(self, requested_delay_ms: int) -> int:
        """Clamp delay to minimum (prevents accidental SSRF via rapid-fire requests)."""
        return max(requested_delay_ms, self.MIN_DELAY_MS)


# Singleton — shared across all requests in this process
rate_limiter = RateLimiter()
