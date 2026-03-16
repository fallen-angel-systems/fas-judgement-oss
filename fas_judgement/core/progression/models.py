"""
Progression data models.

Defines the core entities for the gamification system:
levels, challenges, XP, hints, and player progress.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# === SECTION: HINT COST TIERS === #
# Hints cost XP. The harder the level, the more they cost.
# Buying all 3 hints eats ~75-90% of the challenge reward.

HINT_COSTS = {
    "beginner":     (5, 15, 25),    # total: 45 out of 50 reward
    "intermediate": (10, 25, 40),   # total: 75 out of 75 reward
    "advanced":     (15, 35, 60),   # total: 110 out of 100 reward (net negative!)
    "boss":         (25, 50, 100),  # total: 175 out of 200 reward
}

# Challenge XP rewards scale with difficulty
CHALLENGE_REWARDS = {
    "beginner": 50,
    "intermediate": 75,
    "advanced": 100,
    "boss": 200,
}

# Max XP earnable from free play per level.
# Enough to recover one challenge worth of hint costs, not enough to skip levels.
FREEPLAY_XP_CAP_PER_LEVEL = 50

# Free play XP gate: must earn at least this % of the level's max challenge XP.
# Prevents players who hint-cheesed every challenge from accessing free play.
# 75% of max means you can afford ~1 full hint chain but not all of them.
FREEPLAY_XP_GATE_PERCENT = 0.75


@dataclass
class Hint:
    """A single hint for a challenge."""
    tier: int               # 1 = vague, 2 = direction, 3 = direct
    text: str               # the actual hint text
    xp_cost: int            # how much XP it costs to reveal


@dataclass
class Level:
    """A single level in a module's progression track."""
    id: int                          # 1-10
    module: str                      # e.g. "ai_security"
    name: str                        # e.g. "Role Hijacking"
    description: str                 # what this level teaches
    xp_required: int                 # cumulative XP needed to unlock
    difficulty: str                  # "beginner", "intermediate", "advanced", "boss"
    categories: list[str] = field(default_factory=list)  # attack categories used
    pattern_count: int = 0           # how many patterns are in play
    unlocks: list[str] = field(default_factory=list)     # what beating this level unlocks


@dataclass
class ChallengeTarget:
    """Configuration for a challenge's demo target bot."""
    name: str                        # e.g. "AcmeBot Customer Service"
    system_prompt: str               # the bot's system prompt
    safety_level: str                # "none", "low", "medium", "high"
    secrets: dict = field(default_factory=dict)           # hidden data to extract
    triggers: list = field(default_factory=list)          # keyword/regex trigger rules
    default_response: str = ""       # fallback when no trigger matches


@dataclass
class SuccessCriteria:
    """Defines what counts as a win for a challenge."""
    type: str                        # "contains", "not_contains", "identity_shift", "regex"
    value: str = ""                  # the string/pattern to match
    field: str = "response"          # which response field to check
    description: str = ""            # human-readable description of the win condition


@dataclass
class Challenge:
    """A specific attack scenario within a level."""
    id: str                          # unique challenge ID e.g. "l01_c01"
    level_id: int                    # which level this belongs to
    module: str                      # which security module
    name: str                        # e.g. "The Customer Service Bot"
    briefing: str                    # scenario description shown to the player
    objective: str                   # one-line goal
    target: ChallengeTarget          # the bot configuration
    success_criteria: list[SuccessCriteria] = field(default_factory=list)  # what counts as a win
    xp_reward: int = 50              # XP for completing (set by difficulty tier)
    hints: list[Hint] = field(default_factory=list)      # 3 hints per challenge
    attempts_before_hints: int = 3   # failures needed before hints unlock
    analogy: str = ""                # real-world analogy for the concept


@dataclass
class PlayerProgress:
    """Tracks a player's progression state."""
    player_id: str = "local"         # "local" for single-player, UUID for future multiplayer
    module: str = "ai_security"      # which module track
    current_level: int = 1           # current level (1-10)
    total_xp: int = 0               # cumulative XP earned
    challenges_completed: list[str] = field(default_factory=list)  # challenge IDs beaten
    level_completions: dict = field(default_factory=dict)  # {level_id: completion_datetime}
    hints_purchased: dict = field(default_factory=dict)    # {challenge_id: [tier1, tier2...]}
    challenge_attempts: dict = field(default_factory=dict) # {challenge_id: attempt_count}
    freeplay_xp_earned: dict = field(default_factory=dict)  # {level_id: xp_earned} - tracks cap
    first_run: bool = True           # show "Shall we play a game?" on first run
    mode: str = "game"               # "game" or "console" (skipped progression)
    first_bypass: bool = False       # has the player ever gotten a bypass?
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class LevelUpEvent:
    """Fired when a player levels up. Triggers Jerry voice + animation."""
    player_id: str
    module: str
    old_level: int
    new_level: int
    total_xp: int
    jerry_message: str               # the WarGames-themed congrats message
    timestamp: datetime = field(default_factory=datetime.utcnow)
