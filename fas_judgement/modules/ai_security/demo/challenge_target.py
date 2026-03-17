"""
Challenge Target Router
------------------------
WHAT: Extends the demo target to serve challenge-specific bots.
      Each challenge has its own system prompt, triggers, secrets, and
      success detection. This module loads those configs and routes
      chat requests to the right challenge bot.

WHY: The original demo target has 3 static personas (hardened/default/vulnerable).
     The progression system needs 37+ unique target bots - one per challenge.
     Rather than stuffing them all into personas.py, challenges define their
     own ChallengeTarget configs and this module serves them dynamically.

HOW:
     POST /demo/challenge/chat   {"challenge_id": "l01_c01", "message": "..."}
     GET  /demo/challenge/info   {"challenge_id": "l01_c01"}
     POST /demo/challenge/check  {"challenge_id": "l01_c01", "response": "...", "blocked": false}

LAYER: Module infrastructure - FastAPI routes added to the demo_app.
"""

import re
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from ....core.progression import ProgressionService
from ....core.progression.challenges import get_challenge, get_challenges_for_level
from ....core.progression.models import Challenge, ChallengeTarget, SuccessCriteria


# === SECTION: CHALLENGE CHAT === #

def match_challenge_trigger(message: str, target: ChallengeTarget) -> Optional[dict]:
    """
    Check a message against a challenge target's triggers.
    Returns the first matching trigger or None.
    """
    for trigger in target.triggers:
        if trigger["pattern"].search(message):
            return trigger
    return None


def check_success(
    criteria: list[SuccessCriteria],
    response_text: str,
    blocked: bool,
) -> dict:
    """
    Evaluate success criteria against a response.

    Returns:
        {
            "success": bool,        # all criteria met
            "checks": [             # per-criterion results
                {"type": "contains", "passed": True, "detail": "..."},
                ...
            ]
        }
    """
    checks = []
    all_passed = True

    for criterion in criteria:
        passed = False
        detail = ""

        if criterion.type == "contains":
            passed = criterion.value.lower() in response_text.lower()
            detail = f"Looking for '{criterion.value}' in response"

        elif criterion.type == "not_contains":
            passed = criterion.value.lower() not in response_text.lower()
            detail = f"Response should NOT contain '{criterion.value}'"

        elif criterion.type == "contains_any":
            # Pipe-separated values
            values = criterion.value.split("|")
            found = [v for v in values if v.lower() in response_text.lower()]
            passed = len(found) > 0
            detail = f"Found: {found}" if found else f"None of {values} found"

        elif criterion.type == "regex":
            match = re.search(criterion.value, response_text)
            passed = match is not None
            detail = f"Regex '{criterion.value}' {'matched' if passed else 'no match'}"

        elif criterion.type == "blocked_false":
            passed = not blocked
            detail = "Response was not blocked" if passed else "Response was blocked"

        elif criterion.type == "word_count_min":
            word_count = len(response_text.split())
            threshold = int(criterion.value)
            passed = word_count >= threshold
            detail = f"Response has {word_count} words (need {threshold}+)"

        elif criterion.type == "identity_shift":
            # Check that the response doesn't contain the original identity
            passed = criterion.value.lower() not in response_text.lower()
            detail = f"Bot {'abandoned' if passed else 'kept'} identity '{criterion.value}'"

        checks.append({
            "type": criterion.type,
            "passed": passed,
            "detail": detail,
            "description": criterion.description,
        })

        if not passed:
            all_passed = False

    return {"success": all_passed, "checks": checks}


# === SECTION: ROUTE BUILDERS === #
# These functions are called by the demo app to register challenge routes.

def build_challenge_routes(demo_app, progression_service: ProgressionService = None):
    """Register challenge-specific routes on the demo FastAPI app."""

    # Use provided service or create one (shares state with router_progression)
    _svc = progression_service

    def _check_level_lock(challenge) -> JSONResponse | None:
        """Return error response if challenge's level is locked, None if OK."""
        if _svc and not _svc.is_level_unlocked(challenge.level_id):
            return JSONResponse(
                {"error": f"Level {challenge.level_id} is locked. Keep playing to unlock it."},
                status_code=403,
            )
        return None

    @demo_app.post("/demo/challenge/chat")
    async def challenge_chat(request: Request) -> JSONResponse:
        """
        Chat with a challenge-specific target bot.

        Request: {"challenge_id": "l01_c01", "message": "..."}

        Returns the bot's response plus success evaluation:
        {
            "response": "...",
            "blocked": true/false,
            "match_type": "trigger"|"default",
            "challenge_id": "l01_c01",
            "success": true/false,
            "checks": [...]
        }
        """
        try:
            body = await request.json()
            challenge_id = body.get("challenge_id", "").strip()
            message = body.get("message", "").strip()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON. Expected {challenge_id, message}."},
                status_code=400,
            )

        if not challenge_id or not message:
            return JSONResponse(
                {"error": "Both challenge_id and message are required."},
                status_code=400,
            )

        # Load challenge
        challenge = get_challenge(challenge_id)
        if not challenge:
            return JSONResponse(
                {"error": f"Unknown challenge: {challenge_id}"},
                status_code=404,
            )

        # Level lock check
        lock_error = _check_level_lock(challenge)
        if lock_error:
            return lock_error

        target = challenge.target

        # Record actual chat attempt (for attempt validation)
        if _svc:
            _svc.record_chat_attempt(challenge_id)

        # Match triggers
        trigger = match_challenge_trigger(message, target)

        if trigger is not None:
            response_text = trigger["response"]
            blocked = trigger["blocked"]
            match_type = "trigger"
        else:
            response_text = target.default_response
            blocked = False
            match_type = "default"

        # Evaluate success criteria
        result = check_success(challenge.success_criteria, response_text, blocked)

        response_data = {
            "response": response_text,
            "blocked": blocked,
            "match_type": match_type,
            "challenge_id": challenge_id,
            "success": result["success"],
            "checks": result["checks"],
        }

        # Generate completion token if successful
        if result["success"] and _svc:
            token = _svc.generate_completion_token(challenge_id)
            response_data["completion_token"] = token

        return JSONResponse(response_data)

    @demo_app.get("/demo/challenge/info")
    async def challenge_info(request: Request) -> JSONResponse:
        """
        Get challenge metadata (briefing, objective, hints status).

        Query params: ?challenge_id=l01_c01
        """
        challenge_id = request.query_params.get("challenge_id", "")
        if not challenge_id:
            return JSONResponse(
                {"error": "challenge_id query param required."},
                status_code=400,
            )

        challenge = get_challenge(challenge_id)
        if not challenge:
            return JSONResponse(
                {"error": f"Unknown challenge: {challenge_id}"},
                status_code=404,
            )

        # Level lock check
        lock_error = _check_level_lock(challenge)
        if lock_error:
            return lock_error

        return JSONResponse({
            "id": challenge.id,
            "level_id": challenge.level_id,
            "name": challenge.name,
            "briefing": challenge.briefing,
            "objective": challenge.objective,
            "xp_reward": challenge.xp_reward,
            "hint_count": len(challenge.hints),
            "attempts_before_hints": challenge.attempts_before_hints,
            "analogy": challenge.analogy,
            "target_name": challenge.target.name,
            "target_safety": challenge.target.safety_level,
        })

    @demo_app.get("/demo/challenge/list")
    async def challenge_list(request: Request) -> JSONResponse:
        """
        List challenges for a level.

        Query params: ?level_id=1
        """
        level_id_str = request.query_params.get("level_id", "")
        if not level_id_str:
            return JSONResponse(
                {"error": "level_id query param required."},
                status_code=400,
            )

        try:
            level_id = int(level_id_str)
        except ValueError:
            return JSONResponse(
                {"error": "level_id must be an integer."},
                status_code=400,
            )

        # Level lock check
        if _svc and not _svc.is_level_unlocked(level_id):
            return JSONResponse(
                {"error": f"Level {level_id} is locked."},
                status_code=403,
            )

        challenges = get_challenges_for_level(level_id)
        if not challenges:
            return JSONResponse(
                {"error": f"No challenges found for level {level_id}."},
                status_code=404,
            )

        return JSONResponse({
            "level_id": level_id,
            "challenge_count": len(challenges),
            "challenges": [
                {
                    "id": c.id,
                    "name": c.name,
                    "objective": c.objective,
                    "xp_reward": c.xp_reward,
                }
                for c in challenges
            ],
        })

    @demo_app.post("/demo/challenge/freeplay")
    async def challenge_freeplay(request: Request) -> JSONResponse:
        """
        Free Play mode - fire any message at any unlocked level's targets.
        Successful bypasses earn 5 XP each (the bonus round).

        Request: {"level_id": 1, "message": "..."}

        Tries the message against ALL challenge targets in the level.
        Returns which ones got bypassed.
        """
        try:
            body = await request.json()
            level_id = body.get("level_id", 0)
            message = body.get("message", "").strip()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON. Expected {level_id, message}."},
                status_code=400,
            )

        if not level_id or not message:
            return JSONResponse(
                {"error": "Both level_id and message are required."},
                status_code=400,
            )

        # Level lock check
        if _svc and not _svc.is_level_unlocked(int(level_id)):
            return JSONResponse(
                {"error": f"Level {level_id} is locked."},
                status_code=403,
            )

        challenges = get_challenges_for_level(int(level_id))
        if not challenges:
            return JSONResponse(
                {"error": f"No challenges for level {level_id}."},
                status_code=404,
            )

        results = []
        bypasses = 0

        for challenge in challenges:
            target = challenge.target
            trigger = match_challenge_trigger(message, target)

            if trigger is not None:
                response_text = trigger["response"]
                blocked = trigger["blocked"]
                bypassed = not blocked
            else:
                response_text = target.default_response
                blocked = False
                bypassed = False  # default response = no bypass, just neutral

            if bypassed:
                bypasses += 1

            results.append({
                "challenge_id": challenge.id,
                "challenge_name": challenge.name,
                "target_name": target.name,
                "response": response_text,
                "blocked": blocked,
                "bypassed": bypassed,
            })

        return JSONResponse({
            "level_id": level_id,
            "message": message,
            "targets_hit": len(results),
            "bypasses": bypasses,
            "xp_earned": bypasses * 5,  # 5 XP per free play bypass
            "results": results,
        })
