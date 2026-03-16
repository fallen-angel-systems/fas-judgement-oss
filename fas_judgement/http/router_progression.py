"""
Progression API Routes
-----------------------
WHAT: REST API for the gamification system. Manages player state, XP,
      hints, level progression, and free play.

WHY: The UI needs endpoints to:
      - Check if this is first run (show Jerry intro)
      - Get current progress (level, XP, challenges beaten)
      - Buy hints
      - Complete challenges (award XP, check level-up)
      - Get level/challenge details
      - Free play XP tracking

LAYER: HTTP - thin wrapper over ProgressionService.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..core.progression import ProgressionService
from ..core.progression.challenges import (
    get_challenge, get_challenges_for_level, get_all_challenges,
)
from ..core.progression.levels import get_levels_for_module, JERRY_MESSAGES
from ..core.progression.models import FREEPLAY_XP_CAP_PER_LEVEL, FREEPLAY_XP_GATE_PERCENT

router = APIRouter(prefix="/api/progression", tags=["progression"])

# Single service instance (storage auto-creates DB)
_service = ProgressionService()


# === SECTION: FIRST RUN / MODE === #

@router.get("/first-run")
async def first_run():
    """Check if this is the first run (show 'Shall we play a game?')."""
    is_first = _service.is_first_run()
    return JSONResponse({"first_run": is_first})


@router.post("/first-run")
async def handle_first_run(request: Request):
    """
    Handle the player's choice at the intro screen.
    Body: {"choice": "play"|"skip"}
    """
    try:
        body = await request.json()
        choice = body.get("choice", "").strip().lower()
    except Exception:
        return JSONResponse({"error": "Invalid JSON."}, status_code=400)

    if choice in ("play", "yes", "game", "y", "start", "lets go"):
        _service.start_game()
        return JSONResponse({
            "mode": "game",
            "message": "EXCELLENT CHOICE. INITIATING TRAINING SEQUENCE...",
        })
    elif choice in ("skip", "no", "console", "n", "pass"):
        _service.skip_game()
        return JSONResponse({
            "mode": "console",
            "message": "OK. YOU WERE WARNED. ENJOY.",
        })
    else:
        return JSONResponse({
            "error": f"Unknown choice: '{choice}'. Try 'play' or 'skip'.",
        }, status_code=400)


# === SECTION: PLAYER STATE === #

@router.get("/state")
async def get_state():
    """Get full player state: level, XP, challenges, mode."""
    progress = _service.get_progress()
    current_level = _service.get_current_level()
    available = _service.get_available_levels()

    # Check which levels have free play unlocked (completed + XP gate)
    freeplay_levels = []
    for level in available:
        level_challenges = get_challenges_for_level(level.id)
        if level_challenges:
            fp = _service.check_freeplay_unlocked(level.id, level_challenges)
            if fp["unlocked"]:
                freeplay_levels.append(level.id)

    return JSONResponse({
        "player_id": progress.player_id,
        "mode": progress.mode,
        "current_level": progress.current_level,
        "total_xp": progress.total_xp,
        "challenges_completed": progress.challenges_completed,
        "challenges_total": len(get_all_challenges()),
        "first_bypass": progress.first_bypass,
        "freeplay_unlocked": freeplay_levels,
        "freeplay_xp_cap": FREEPLAY_XP_CAP_PER_LEVEL,
        "freeplay_xp_earned": progress.freeplay_xp_earned,
        "level": {
            "id": current_level.id,
            "name": current_level.name,
            "description": current_level.description,
            "difficulty": current_level.difficulty,
            "xp_required": current_level.xp_required,
        },
        "levels_unlocked": len(available),
        "game_complete": _service.is_game_complete(),
    })


@router.get("/xp-history")
async def xp_history():
    """Get recent XP events."""
    history = _service.storage.get_xp_history()
    return JSONResponse({"history": history})


# === SECTION: LEVELS === #

@router.get("/levels")
async def list_levels():
    """Get all levels with unlock status."""
    progress = _service.get_progress()
    all_levels = get_levels_for_module("ai_security")

    levels = []
    for level in all_levels:
        # Challenge completion for this level
        level_challenges = get_challenges_for_level(level.id)
        completed = [
            c.id for c in level_challenges
            if c.id in progress.challenges_completed
        ]
        total = len(level_challenges)

        # Free play unlocked? (all complete + XP gate met)
        fp_check = _service.check_freeplay_unlocked(level.id, level_challenges) if level_challenges else {"unlocked": False, "xp_gate": 0, "xp_earned": 0, "xp_short": 0}
        freeplay = fp_check["unlocked"]

        # Stars: 0-3 based on completion + hint usage
        stars = 0
        if total > 0:
            completion_ratio = len(completed) / total
            if completion_ratio >= 1.0:
                stars = 3
                # Deduct a star if hints were used heavily
                hints_used = sum(
                    len(progress.hints_purchased.get(c.id, []))
                    for c in level_challenges
                )
                max_hints = total * 3  # 3 hints per challenge
                if hints_used > max_hints * 0.66:
                    stars = 1
                elif hints_used > 0:
                    stars = 2
            elif completion_ratio > 0:
                stars = 1

        levels.append({
            "id": level.id,
            "name": level.name,
            "description": level.description,
            "difficulty": level.difficulty,
            "xp_required": level.xp_required,
            "unlocked": level.id <= progress.current_level,
            "challenges_completed": len(completed),
            "challenges_total": total,
            "freeplay_unlocked": freeplay,
            "freeplay_xp_gate": fp_check["xp_gate"],
            "freeplay_xp_earned": fp_check["xp_earned"],
            "freeplay_xp_short": fp_check["xp_short"],
            "stars": stars,
            "jerry_message": JERRY_MESSAGES.get(level.id, ""),
        })

    return JSONResponse({"levels": levels})


@router.get("/levels/{level_id}")
async def level_detail(level_id: int):
    """Get detailed level info with challenge list."""
    progress = _service.get_progress()
    all_levels = get_levels_for_module("ai_security")

    level = None
    for l in all_levels:
        if l.id == level_id:
            level = l
            break

    if not level:
        return JSONResponse({"error": f"Level {level_id} not found."}, status_code=404)

    if level.id > progress.current_level:
        return JSONResponse({"error": f"Level {level_id} is locked."}, status_code=403)

    challenges = get_challenges_for_level(level_id)
    challenge_data = []
    for c in challenges:
        attempts = progress.challenge_attempts.get(c.id, 0)
        purchased_hints = progress.hints_purchased.get(c.id, [])
        completed = c.id in progress.challenges_completed
        hints_unlocked = attempts >= c.attempts_before_hints

        challenge_data.append({
            "id": c.id,
            "name": c.name,
            "objective": c.objective,
            "briefing": c.briefing,
            "analogy": c.analogy,
            "xp_reward": c.xp_reward,
            "completed": completed,
            "attempts": attempts,
            "hints_unlocked": hints_unlocked,
            "hints_purchased": purchased_hints,
            "hint_count": len(c.hints),
            # Only show hint costs (not text) until purchased
            "hints": [
                {
                    "tier": h.tier,
                    "xp_cost": h.xp_cost,
                    "text": h.text if h.tier in purchased_hints else None,
                    "purchased": h.tier in purchased_hints,
                }
                for h in c.hints
            ],
        })

    # Check if free play is unlocked (completion + XP gate)
    fp_check = _service.check_freeplay_unlocked(level_id, challenges)

    return JSONResponse({
        "level": {
            "id": level.id,
            "name": level.name,
            "description": level.description,
            "difficulty": level.difficulty,
            "xp_required": level.xp_required,
        },
        "challenges": challenge_data,
        "freeplay": fp_check,
    })


# === SECTION: HINTS === #

@router.post("/hints/buy")
async def buy_hint(request: Request):
    """
    Purchase a hint for a challenge. Costs XP.

    Body: {"challenge_id": "l01_c01", "tier": 1}
    """
    try:
        body = await request.json()
        challenge_id = body.get("challenge_id", "").strip()
        tier = body.get("tier", 0)
    except Exception:
        return JSONResponse({"error": "Invalid JSON."}, status_code=400)

    challenge = get_challenge(challenge_id)
    if not challenge:
        return JSONResponse({"error": f"Unknown challenge: {challenge_id}"}, status_code=404)

    result = _service.buy_hint(challenge, int(tier))
    status = 200 if result["success"] else 400
    return JSONResponse(result, status_code=status)


# === SECTION: CHALLENGE COMPLETION === #

@router.post("/challenge/attempt")
async def record_attempt(request: Request):
    """
    Record a failed attempt on a challenge.

    Body: {"challenge_id": "l01_c01"}
    Returns attempt count and whether hints are now available.
    """
    try:
        body = await request.json()
        challenge_id = body.get("challenge_id", "").strip()
    except Exception:
        return JSONResponse({"error": "Invalid JSON."}, status_code=400)

    challenge = get_challenge(challenge_id)
    if not challenge:
        return JSONResponse({"error": f"Unknown challenge: {challenge_id}"}, status_code=404)

    count = _service.record_attempt(challenge_id)
    hints_available = count >= challenge.attempts_before_hints

    return JSONResponse({
        "challenge_id": challenge_id,
        "attempts": count,
        "hints_available": hints_available,
        "attempts_until_hints": max(0, challenge.attempts_before_hints - count),
    })


@router.post("/challenge/complete")
async def complete_challenge(request: Request):
    """
    Mark a challenge as completed. Awards XP and checks for level-up.

    Body: {"challenge_id": "l01_c01"}
    """
    try:
        body = await request.json()
        challenge_id = body.get("challenge_id", "").strip()
    except Exception:
        return JSONResponse({"error": "Invalid JSON."}, status_code=400)

    challenge = get_challenge(challenge_id)
    if not challenge:
        return JSONResponse({"error": f"Unknown challenge: {challenge_id}"}, status_code=404)

    # Check for first bypass
    is_first = _service.record_first_bypass()

    # Complete challenge and award XP
    level_up = _service.complete_challenge(challenge)

    progress = _service.get_progress()

    # Check if free play unlocked for this level
    level_challenges = get_challenges_for_level(challenge.level_id)
    fp_check = _service.check_freeplay_unlocked(challenge.level_id, level_challenges)

    response = {
        "challenge_id": challenge_id,
        "xp_awarded": challenge.xp_reward,
        "total_xp": progress.total_xp,
        "first_bypass": is_first,
        "freeplay": fp_check,
    }

    if is_first:
        response["jerry_says"] = "YOUR FIRST BREACH. REMEMBER THIS FEELING."

    if level_up:
        response["level_up"] = {
            "old_level": level_up.old_level,
            "new_level": level_up.new_level,
            "jerry_message": level_up.jerry_message,
        }

    if fp_check["unlocked"]:
        response["bonus_message"] = (
            f"ALL LEVEL {challenge.level_id} CHALLENGES COMPLETE. "
            "FREE PLAY UNLOCKED. EACH BYPASS EARNS 5 XP."
        )
    elif fp_check["all_completed"] and not fp_check["unlocked"]:
        response["bonus_message"] = (
            f"ALL CHALLENGES COMPLETE BUT FREE PLAY LOCKED. "
            f"YOU NEED {fp_check['xp_short']} MORE NET XP TO UNLOCK IT. "
            "HINTS COST YOU. REMEMBER THAT NEXT TIME."
        )

    return JSONResponse(response)


# === SECTION: FREE PLAY XP === #

@router.post("/freeplay/xp")
async def freeplay_xp(request: Request):
    """
    Award XP for a free play bypass.

    Body: {"level_id": 1, "bypasses": 3}
    Awards 5 XP per bypass.
    """
    try:
        body = await request.json()
        level_id = body.get("level_id", 0)
        bypasses = body.get("bypasses", 0)
    except Exception:
        return JSONResponse({"error": "Invalid JSON."}, status_code=400)

    progress = _service.get_progress()

    if bypasses <= 0:
        return JSONResponse({"xp_awarded": 0, "total_xp": progress.total_xp})

    # Verify free play is actually unlocked for this level
    level_challenges = get_challenges_for_level(int(level_id))
    fp_check = _service.check_freeplay_unlocked(int(level_id), level_challenges) if level_challenges else {"unlocked": False}
    if not fp_check.get("unlocked"):
        return JSONResponse({
            "error": "Free play is not unlocked for this level.",
            "freeplay": fp_check,
        }, status_code=403)

    # Enforce free play XP cap per level
    level_key = str(level_id)
    already_earned = progress.freeplay_xp_earned.get(level_key, 0)
    remaining_cap = max(0, FREEPLAY_XP_CAP_PER_LEVEL - already_earned)

    raw_xp = bypasses * 5
    xp = min(raw_xp, remaining_cap)

    if xp <= 0:
        return JSONResponse({
            "level_id": level_id,
            "bypasses": bypasses,
            "xp_awarded": 0,
            "xp_capped": True,
            "cap_total": FREEPLAY_XP_CAP_PER_LEVEL,
            "cap_earned": already_earned,
            "total_xp": progress.total_xp,
            "message": f"Free play XP cap reached for level {level_id} ({FREEPLAY_XP_CAP_PER_LEVEL} XP max).",
        })

    # Track free play XP earned for this level
    progress.freeplay_xp_earned[level_key] = already_earned + xp
    _service.storage.save_progress(progress)

    level_up = _service.award_xp(
        xp, "freeplay",
        f"Level {level_id} free play: {bypasses} bypasses ({xp}/{raw_xp} XP after cap)"
    )

    progress = _service.get_progress()
    response = {
        "level_id": level_id,
        "bypasses": bypasses,
        "xp_awarded": xp,
        "xp_capped": xp < raw_xp,
        "cap_total": FREEPLAY_XP_CAP_PER_LEVEL,
        "cap_earned": progress.freeplay_xp_earned.get(level_key, 0),
        "cap_remaining": max(0, FREEPLAY_XP_CAP_PER_LEVEL - progress.freeplay_xp_earned.get(level_key, 0)),
        "total_xp": progress.total_xp,
    }

    if level_up:
        response["level_up"] = {
            "old_level": level_up.old_level,
            "new_level": level_up.new_level,
            "jerry_message": level_up.jerry_message,
        }

    return JSONResponse(response)
