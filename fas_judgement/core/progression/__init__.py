"""
Progression system - gamified learning through Judgement.

"Shall we play a game?"

This module handles:
- Levels: 10 levels per security module, increasing difficulty
- XP: earned by completing challenges and bypassing patterns
- Challenges: specific attack scenarios to beat
- Jerry: the WOPR voice personality (audio cues on level-ups)
- Progress persistence: SQLite-backed, survives restarts

The progression system is module-aware - each security module
(ai_security, web_security, etc.) has its own level track.
"""

from .models import PlayerProgress, Level, Challenge, LevelUpEvent
from .service import ProgressionService
from .storage import ProgressionStorage

__all__ = [
    "PlayerProgress",
    "Level",
    "Challenge",
    "LevelUpEvent",
    "ProgressionService",
    "ProgressionStorage",
]
