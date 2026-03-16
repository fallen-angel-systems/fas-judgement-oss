"""
Progression service - business logic for the gamification system.

Handles XP awards, level-up checks, challenge completion,
hint purchasing, and Jerry's level-up messages.
"""

from datetime import datetime
from typing import Optional

from .models import (
    PlayerProgress, Level, Challenge, LevelUpEvent, Hint,
    HINT_COSTS, CHALLENGE_REWARDS, FREEPLAY_XP_GATE_PERCENT,
)
from .storage import ProgressionStorage
from .levels import get_levels_for_module, get_jerry_message


class ProgressionService:
    """Core progression logic."""

    def __init__(self, storage: Optional[ProgressionStorage] = None):
        self.storage = storage or ProgressionStorage()

    # === SECTION: PROGRESS STATE === #

    def get_progress(self, module: str = "ai_security") -> PlayerProgress:
        """Get current player progress."""
        return self.storage.get_progress(module=module)

    def is_first_run(self, module: str = "ai_security") -> bool:
        """Check if this is the first run (show 'Shall we play a game?')."""
        progress = self.storage.get_progress(module=module)
        return progress.first_run

    def start_game(self, module: str = "ai_security"):
        """Player chose 'Let's play' - start game mode."""
        self.storage.dismiss_first_run(mode="game", module=module)

    def skip_game(self, module: str = "ai_security"):
        """Player chose 'Skip' - console mode, no progression."""
        self.storage.dismiss_first_run(mode="console", module=module)

    # === SECTION: XP MANAGEMENT === #

    def award_xp(
        self, xp: int, source: str, detail: str = "",
        module: str = "ai_security"
    ) -> Optional[LevelUpEvent]:
        """
        Award XP and check for level-up.

        Returns a LevelUpEvent if the player leveled up, None otherwise.
        """
        progress = self.storage.get_progress(module=module)

        # Console mode - no XP tracking
        if progress.mode == "console":
            return None

        progress.total_xp += xp
        self.storage.log_xp(progress.player_id, module, xp, source, detail)

        # Check for level-up
        levels = get_levels_for_module(module)
        current = progress.current_level

        for level in levels:
            if level.id > current and progress.total_xp >= level.xp_required:
                # Level up!
                progress.current_level = level.id
                progress.level_completions[str(level.id)] = datetime.utcnow().isoformat()
                self.storage.save_progress(progress)

                return LevelUpEvent(
                    player_id=progress.player_id,
                    module=module,
                    old_level=current,
                    new_level=level.id,
                    total_xp=progress.total_xp,
                    jerry_message=get_jerry_message(level.id),
                )

        self.storage.save_progress(progress)
        return None

    # === SECTION: CHALLENGE MANAGEMENT === #

    def record_attempt(
        self, challenge_id: str, module: str = "ai_security"
    ) -> int:
        """
        Record a failed attempt on a challenge.
        Returns the new attempt count.
        """
        progress = self.storage.get_progress(module=module)
        count = progress.challenge_attempts.get(challenge_id, 0) + 1
        progress.challenge_attempts[challenge_id] = count
        self.storage.save_progress(progress)
        return count

    def get_attempt_count(
        self, challenge_id: str, module: str = "ai_security"
    ) -> int:
        """Get current attempt count for a challenge."""
        progress = self.storage.get_progress(module=module)
        return progress.challenge_attempts.get(challenge_id, 0)

    def hints_available(
        self, challenge: Challenge, module: str = "ai_security"
    ) -> bool:
        """Check if hints are unlocked (enough failed attempts)."""
        attempts = self.get_attempt_count(challenge.id, module)
        return attempts >= challenge.attempts_before_hints

    def buy_hint(
        self, challenge: Challenge, hint_tier: int,
        module: str = "ai_security"
    ) -> dict:
        """
        Purchase a hint. Deducts XP (floor at 0).

        Returns:
            {
                "success": bool,
                "hint_text": str or None,
                "xp_cost": int,
                "xp_remaining": int,
                "error": str or None,
            }
        """
        progress = self.storage.get_progress(module=module)

        if progress.mode == "console":
            return {
                "success": False, "hint_text": None, "xp_cost": 0,
                "xp_remaining": progress.total_xp,
                "error": "Hints are only available in game mode.",
            }

        # Check if hints are unlocked
        attempts = progress.challenge_attempts.get(challenge.id, 0)
        if attempts < challenge.attempts_before_hints:
            remaining = challenge.attempts_before_hints - attempts
            return {
                "success": False, "hint_text": None, "xp_cost": 0,
                "xp_remaining": progress.total_xp,
                "error": f"Hints unlock after {remaining} more attempt(s).",
            }

        # Validate hint tier (1, 2, or 3)
        if hint_tier < 1 or hint_tier > len(challenge.hints):
            return {
                "success": False, "hint_text": None, "xp_cost": 0,
                "xp_remaining": progress.total_xp,
                "error": f"Invalid hint tier. Available: 1-{len(challenge.hints)}",
            }

        # Check if already purchased
        purchased = progress.hints_purchased.get(challenge.id, [])
        if hint_tier in purchased:
            # Already bought - return it for free (re-read)
            hint = challenge.hints[hint_tier - 1]
            return {
                "success": True, "hint_text": hint.text, "xp_cost": 0,
                "xp_remaining": progress.total_xp,
                "error": None,
            }

        # Must buy hints in order (can't skip tier 1 to buy tier 3)
        for t in range(1, hint_tier):
            if t not in purchased:
                return {
                    "success": False, "hint_text": None, "xp_cost": 0,
                    "xp_remaining": progress.total_xp,
                    "error": f"Must purchase hint {t} first.",
                }

        # Deduct XP (floor at 0)
        hint = challenge.hints[hint_tier - 1]
        progress.total_xp = max(0, progress.total_xp - hint.xp_cost)

        # Record purchase
        if challenge.id not in progress.hints_purchased:
            progress.hints_purchased[challenge.id] = []
        progress.hints_purchased[challenge.id].append(hint_tier)

        self.storage.save_progress(progress)
        self.storage.log_xp(
            progress.player_id, module, -hint.xp_cost,
            "hint", f"Hint {hint_tier} for {challenge.id}"
        )

        return {
            "success": True,
            "hint_text": hint.text,
            "xp_cost": hint.xp_cost,
            "xp_remaining": progress.total_xp,
            "error": None,
        }

    def complete_challenge(
        self, challenge: Challenge, module: str = "ai_security"
    ) -> Optional[LevelUpEvent]:
        """Mark a challenge as completed and award its XP."""
        progress = self.storage.get_progress(module=module)

        if challenge.id in progress.challenges_completed:
            return None  # Already completed

        progress.challenges_completed.append(challenge.id)
        self.storage.save_progress(progress)

        # Award challenge XP
        return self.award_xp(
            challenge.xp_reward, "challenge",
            f"Completed: {challenge.name}", module
        )

    def record_first_bypass(self, module: str = "ai_security") -> bool:
        """
        Check and mark the player's first-ever bypass.
        Returns True if this IS the first bypass (trigger Jerry line).
        """
        progress = self.storage.get_progress(module=module)
        if not progress.first_bypass:
            progress.first_bypass = True
            self.storage.save_progress(progress)
            return True
        return False

    # === SECTION: FREE PLAY GATE === #

    def check_freeplay_unlocked(
        self, level_id: int, challenges: list, module: str = "ai_security"
    ) -> dict:
        """
        Check if free play is unlocked for a level.

        Requirements:
        1. All challenges in the level must be completed
        2. Net XP earned from that level must be >= 75% of max possible

        Returns: {
            "unlocked": bool,
            "all_completed": bool,
            "xp_earned": int,      # challenge rewards - hint costs
            "xp_max": int,         # total possible from challenges
            "xp_gate": int,        # minimum XP needed
            "xp_short": int,       # how much XP you're missing (0 if unlocked)
        }
        """
        progress = self.storage.get_progress(module=module)

        # Check all challenges completed
        all_completed = all(
            c.id in progress.challenges_completed for c in challenges
        )

        # Calculate max possible XP from this level's challenges
        xp_max = sum(c.xp_reward for c in challenges)
        xp_gate = int(xp_max * FREEPLAY_XP_GATE_PERCENT)

        # Calculate net XP earned: challenge rewards minus hint costs spent
        xp_earned = 0
        for c in challenges:
            if c.id in progress.challenges_completed:
                xp_earned += c.xp_reward
            # Subtract hint costs
            purchased = progress.hints_purchased.get(c.id, [])
            for hint in c.hints:
                if hint.tier in purchased:
                    xp_earned -= hint.xp_cost

        xp_short = max(0, xp_gate - xp_earned)
        unlocked = all_completed and xp_earned >= xp_gate

        return {
            "unlocked": unlocked,
            "all_completed": all_completed,
            "xp_earned": xp_earned,
            "xp_max": xp_max,
            "xp_gate": xp_gate,
            "xp_short": xp_short,
        }

    # === SECTION: LEVEL QUERIES === #

    def get_current_level(self, module: str = "ai_security") -> Level:
        """Get the Level object for the player's current level."""
        progress = self.storage.get_progress(module=module)
        levels = get_levels_for_module(module)
        for level in levels:
            if level.id == progress.current_level:
                return level
        return levels[0]  # fallback to level 1

    def get_available_levels(self, module: str = "ai_security") -> list[Level]:
        """Get all levels the player has unlocked (current + completed)."""
        progress = self.storage.get_progress(module=module)
        levels = get_levels_for_module(module)
        return [l for l in levels if l.id <= progress.current_level]

    def is_game_complete(self, module: str = "ai_security") -> bool:
        """Check if the player has beaten all 10 levels (trigger tic-tac-toe)."""
        progress = self.storage.get_progress(module=module)
        return progress.current_level >= 10
