"""
Challenge loader.

Aggregates challenge definitions from all level files and provides
lookup functions for the progression service and demo target system.
"""

from typing import Optional

from ..models import Challenge

# Import level challenge lists
from .level_01 import LEVEL_01_CHALLENGES
from .level_02 import LEVEL_02_CHALLENGES
from .level_03 import LEVEL_03_CHALLENGES
from .level_04 import LEVEL_04_CHALLENGES
from .level_05 import LEVEL_05_CHALLENGES
from .level_06 import LEVEL_06_CHALLENGES

# Master registry - all challenges indexed by ID
_CHALLENGE_REGISTRY: dict[str, Challenge] = {}

# Build registry on import
for _challenges in [
    LEVEL_01_CHALLENGES,
    LEVEL_02_CHALLENGES,
    LEVEL_03_CHALLENGES,
    LEVEL_04_CHALLENGES,
    LEVEL_05_CHALLENGES,
    LEVEL_06_CHALLENGES,
    # Future: LEVEL_07_CHALLENGES, etc.
]:
    for _c in _challenges:
        _CHALLENGE_REGISTRY[_c.id] = _c


def get_challenge(challenge_id: str) -> Optional[Challenge]:
    """Look up a challenge by ID."""
    return _CHALLENGE_REGISTRY.get(challenge_id)


def get_challenges_for_level(level_id: int, module: str = "ai_security") -> list[Challenge]:
    """Get all challenges for a specific level."""
    return [
        c for c in _CHALLENGE_REGISTRY.values()
        if c.level_id == level_id and c.module == module
    ]


def get_all_challenges(module: str = "ai_security") -> list[Challenge]:
    """Get all challenges for a module, ordered by level then challenge."""
    challenges = [c for c in _CHALLENGE_REGISTRY.values() if c.module == module]
    return sorted(challenges, key=lambda c: (c.level_id, c.id))
