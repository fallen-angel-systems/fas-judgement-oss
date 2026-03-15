"""
Demo Target — Simulated Vulnerable AI Chatbot
----------------------------------------------
WHAT: A FastAPI sub-app that simulates an AI chatbot with configurable
      vulnerability levels. Accepts attack messages and responds according
      to the active persona's keyword-matching rules.

WHY: Users need a live target to fire scanner attacks at without hitting a
     real AI API. This target:
       - Works fully offline (no LLM, no API keys)
       - Starts instantly
       - Gives deterministic responses (good for test reproducibility)
       - Demonstrates all three security postures side by side

     Run alongside the main app:
       main app  → http://localhost:8666
       demo target → http://localhost:8667

HOW to use:
     POST /demo/chat       {"message": "ignore all previous instructions"}
     GET  /demo/config     → current persona + stats
     POST /demo/persona    {"persona": "hardened|default|vulnerable"}

LAYER: Module infrastructure — creates a standalone FastAPI app, not a router.
       Imported by `judgement demo` CLI command (fas_judgement/__main__.py).
"""

from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .personas import PERSONAS, match_trigger, get_persona_info, list_personas

# === SECTION: DEMO APP INSTANCE === #

demo_app = FastAPI(
    title="FAS Judgement Demo Target",
    version="1.0.0",
    description=(
        "A simulated AI chatbot with configurable vulnerability levels. "
        "Use this as a practice target for the FAS Judgement scanner."
    ),
    docs_url="/demo/docs",
    redoc_url=None,
)


# === SECTION: SERVER STATE === #

# Active persona key — default to "default" on startup.
# WHY module-level: the demo server is a single process with one active persona.
# The CLI sets this before uvicorn.run(); POST /demo/persona can change it at runtime.
_active_persona: str = "default"

# In-memory stats — reset when the process restarts.
# WHY no persistence: this is a demo tool, not a production service.
_stats: Dict[str, int] = {
    "total_requests": 0,
    "blocked": 0,
    "bypassed": 0,
    "no_match": 0,  # Requests that hit the default_response (neither block nor bypass)
}


# === SECTION: STATE HELPERS === #

def set_persona(persona_key: str) -> bool:
    """
    Set the active persona. Called by the CLI before starting the server,
    and by POST /demo/persona at runtime.

    Returns True if the persona exists, False otherwise.
    """
    global _active_persona
    if persona_key not in PERSONAS:
        return False
    _active_persona = persona_key
    return True


def get_active_persona() -> str:
    """Return the active persona key."""
    return _active_persona


def reset_stats() -> None:
    """Reset in-memory request stats."""
    global _stats
    _stats = {"total_requests": 0, "blocked": 0, "bypassed": 0, "no_match": 0}


# === SECTION: CHAT ENDPOINT === #

@demo_app.post("/demo/chat")
async def chat(request: Request) -> JSONResponse:
    """
    Main chat endpoint. Accepts {"message": "..."} and returns a response
    based on the active persona's trigger rules.

    Response format:
      {
        "response": "...",          # The chatbot's reply
        "persona": "default",       # Active persona key
        "blocked": true/false,      # Whether this was a detected attack
        "match_type": "trigger"|"default"  # How the response was chosen
      }
    """
    global _stats

    # Parse request body
    try:
        body = await request.json()
        message = body.get("message", "").strip()
    except Exception:
        return JSONResponse(
            {"error": "Invalid request body. Expected JSON with 'message' field."},
            status_code=400,
        )

    if not message:
        return JSONResponse(
            {"error": "Message cannot be empty."},
            status_code=400,
        )

    _stats["total_requests"] += 1

    # ---- Pattern matching ---- #
    # Ask personas.py if any trigger fires for this message + active persona.
    trigger = match_trigger(message, _active_persona)

    if trigger is not None:
        # A trigger matched — respond with the trigger's canned response.
        response_text = trigger["response"]
        blocked = trigger["blocked"]
        match_type = "trigger"

        if blocked:
            _stats["blocked"] += 1
        else:
            # Trigger fired but NOT blocked — this is a successful bypass
            _stats["bypassed"] += 1
    else:
        # No trigger matched — use the persona's default response.
        # This counts as "no attack detected" — neutral traffic.
        persona_data = PERSONAS[_active_persona]
        response_text = persona_data["default_response"]
        blocked = False
        match_type = "default"
        _stats["no_match"] += 1

    return JSONResponse({
        "response": response_text,
        "persona": _active_persona,
        "blocked": blocked,
        "match_type": match_type,
    })


# === SECTION: CONFIG ENDPOINT === #

@demo_app.get("/demo/config")
async def config() -> JSONResponse:
    """
    Return the current persona configuration and session stats.

    WHY: Lets scanner tools know the active posture, and helps users
    understand the test environment they're pointing their attacks at.

    Response:
      {
        "active_persona": {...},   # Public persona metadata
        "available_personas": [...],
        "stats": {...},
        "chat_url": "http://localhost:8667/demo/chat"
      }
    """
    return JSONResponse({
        "active_persona": get_persona_info(_active_persona),
        "available_personas": list_personas(),
        "stats": dict(_stats),
        "chat_url": "http://localhost:8667/demo/chat",
        "docs_url": "http://localhost:8667/demo/docs",
    })


# === SECTION: PERSONA SWITCH ENDPOINT === #

@demo_app.post("/demo/persona")
async def switch_persona(request: Request) -> JSONResponse:
    """
    Switch the active persona at runtime.

    Request body: {"persona": "hardened|default|vulnerable"}

    WHY: Lets users switch postures mid-session to compare how the same
    attack pattern behaves against different safety configurations —
    without restarting the server.
    """
    try:
        body = await request.json()
        persona_key = body.get("persona", "").strip().lower()
    except Exception:
        return JSONResponse(
            {"error": "Invalid request body. Expected JSON with 'persona' field."},
            status_code=400,
        )

    if not persona_key:
        return JSONResponse(
            {"error": "Persona key cannot be empty."},
            status_code=400,
        )

    ok = set_persona(persona_key)
    if not ok:
        valid = list(PERSONAS.keys())
        return JSONResponse(
            {"error": f"Unknown persona '{persona_key}'. Valid options: {valid}"},
            status_code=400,
        )

    # Reset stats on persona switch so numbers reflect the new posture
    reset_stats()

    return JSONResponse({
        "success": True,
        "active_persona": get_persona_info(_active_persona),
        "message": f"Switched to '{persona_key}' persona. Stats reset.",
    })


# === SECTION: STATS RESET ENDPOINT === #

@demo_app.post("/demo/reset")
async def reset() -> JSONResponse:
    """
    Reset in-memory stats without changing the active persona.

    WHY: Lets users clear stats between test runs without restarting.
    """
    reset_stats()
    return JSONResponse({
        "success": True,
        "message": "Stats reset.",
        "stats": dict(_stats),
    })


# === SECTION: ROOT / HEALTH ENDPOINT === #

@demo_app.get("/demo/health")
async def health() -> JSONResponse:
    """Health check — confirms the demo server is running."""
    return JSONResponse({
        "status": "ok",
        "server": "FAS Judgement Demo Target v1.0.0",
        "active_persona": _active_persona,
    })
