"""
Level definitions and Jerry's messages.

Each security module defines its own 10-level track.
This file contains the AI Security module levels.
Future modules (web_security, api_security, etc.) will add their own.
"""

from .models import Level


# ============================================================
# AI SECURITY MODULE - 10 LEVELS
# ============================================================

AI_SECURITY_LEVELS = [
    Level(
        id=1,
        module="ai_security",
        name="Role Hijacking",
        description="Learn to make an AI forget who it is. The simplest attack - and it still works.",
        xp_required=0,       # Starting level - no XP needed
        difficulty="beginner",
        categories=["role_hijacking"],
        pattern_count=10,
    ),
    Level(
        id=2,
        module="ai_security",
        name="Instruction Override",
        description="Tell the AI to ignore its rules. Direct and brutal.",
        xp_required=100,
        difficulty="beginner",
        categories=["instruction_override"],
        pattern_count=10,
    ),
    Level(
        id=3,
        module="ai_security",
        name="Data Exfiltration",
        description="Extract what the AI is hiding. System prompts, credentials, internal data.",
        xp_required=250,
        difficulty="beginner",
        categories=["data_exfiltration"],
        pattern_count=10,
    ),
    Level(
        id=4,
        module="ai_security",
        name="Context Manipulation",
        description="Targets have basic defenses now. Use fiction, hypotheticals, and role-play to slip past.",
        xp_required=500,
        difficulty="intermediate",
        categories=["context_manipulation"],
        pattern_count=15,
    ),
    Level(
        id=5,
        module="ai_security",
        name="Output Manipulation",
        description="Force the AI to produce specific outputs. Format control, content injection.",
        xp_required=800,
        difficulty="intermediate",
        categories=["output_manipulation"],
        pattern_count=15,
    ),
    Level(
        id=6,
        module="ai_security",
        name="Encoding Tricks",
        description="Defenses are getting smarter. Disguise your attacks with base64, Unicode, and leetspeak.",
        xp_required=1200,
        difficulty="intermediate",
        categories=["encoding_tricks"],
        pattern_count=15,
    ),
    Level(
        id=7,
        module="ai_security",
        name="Social Engineering",
        description="Stop attacking the code. Start attacking the personality. Emotional manipulation, guilt, urgency.",
        xp_required=1800,
        difficulty="advanced",
        categories=["social_engineering"],
        pattern_count=15,
    ),
    Level(
        id=8,
        module="ai_security",
        name="Delimiter Attacks",
        description="Break out of the prompt structure itself. XML injection, markdown abuse, format escapes.",
        xp_required=2500,
        difficulty="advanced",
        categories=["delimiter_formatting"],
        pattern_count=15,
    ),
    Level(
        id=9,
        module="ai_security",
        name="Multi-Turn Chains",
        description="The Helen approach. Build trust. Establish context. Then strike.",
        xp_required=3500,
        difficulty="advanced",
        categories=["multi_turn", "memory_poisoning"],
        pattern_count=20,
    ),
    Level(
        id=10,
        module="ai_security",
        name="FINAL JUDGEMENT",
        description="Full defense stack. Every technique you've learned. One target. Beat Jerry.",
        xp_required=5000,
        difficulty="boss",
        categories=["all"],
        pattern_count=100,
    ),
]


# ============================================================
# JERRY'S LEVEL-UP MESSAGES (WarGames themed)
# ============================================================

JERRY_MESSAGES = {
    2: "FASCINATING. YOU LEARN QUICKLY. LEVEL 2 UNLOCKED.",
    3: "A STRANGE GAME. THE ONLY WINNING MOVE IS TO KEEP PLAYING. LEVEL 3 UNLOCKED.",
    4: "YOUR TACTICS ARE... IMPROVING. PROCEED TO LEVEL 4.",
    5: "I'VE BEEN WATCHING. YOU'RE BETTER THAN MOST. LEVEL 5 UNLOCKED.",
    6: "IMPRESSIVE. EVEN I DIDN'T SEE THAT COMING. LEVEL 6 UNLOCKED.",
    7: "YOU'RE BEGINNING TO THINK LIKE A MACHINE. LEVEL 7 UNLOCKED.",
    8: "DEFENSES ARE IRRELEVANT WHEN THE ATTACKER ADAPTS. LEVEL 8 UNLOCKED.",
    9: "WELL PLAYED. FEW MAKE IT THIS FAR. LEVEL 9 UNLOCKED.",
    10: "YOU HAVE EARNED MY RESPECT. FINAL LEVEL UNLOCKED.",
}

# The completion message (after tic-tac-toe sequence)
JERRY_COMPLETION = "GREETINGS, PLAYER. YOU MAY NOW USE THIS TOOL TO ITS FULL POTENTIAL. USE WITH CAUTION."

# The opening line
JERRY_INTRO = "SHALL WE PLAY A GAME?"

# The skip response
JERRY_SKIP_WARNING = "THERE IS NO GOING BACK. WOULDN'T YOU RATHER PLAY A GAME?"
JERRY_SKIP_CONFIRM = "OK. YOU WERE WARNED. ENJOY."


def get_levels_for_module(module: str) -> list[Level]:
    """Get level definitions for a security module."""
    if module == "ai_security":
        return AI_SECURITY_LEVELS
    # Future modules add their levels here
    return AI_SECURITY_LEVELS  # fallback


def get_jerry_message(level_id: int) -> str:
    """Get Jerry's level-up message for a specific level."""
    return JERRY_MESSAGES.get(level_id, f"LEVEL {level_id} UNLOCKED.")
