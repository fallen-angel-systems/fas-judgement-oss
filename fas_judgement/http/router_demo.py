"""
Demo Target Management Router
-------------------------------
WHY: Thin HTTP layer for demo target lifecycle management. All business
     logic lives in the demo service (modules/ai_security/demo/service.py).
     This router only handles:
       - HTTP request/response serialisation
       - Status code decisions
       - Calling the service

ENDPOINTS:
  POST /api/demo/start   — Spawn demo target subprocess on port 8667
  POST /api/demo/stop    — Kill the demo subprocess
  GET  /api/demo/status  — Check if demo is running + health

LAYER: HTTP — outermost ring. Imports service, returns JSONResponse.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..modules.ai_security.demo.service import (
    start as demo_start,
    stop as demo_stop,
    status as demo_status,
)

router = APIRouter()


# === SECTION: ENDPOINTS === #

@router.post("/api/demo/start")
async def start_demo(request: Request) -> JSONResponse:
    """
    Start the demo target.

    Request body (optional): {"persona": "hardened|default|vulnerable"}
    """
    # Parse optional persona from body
    try:
        body = await request.json()
        persona = body.get("persona", "default").strip().lower()
    except Exception:
        persona = "default"

    result = await demo_start(persona)

    status_code = 200 if result.success else 400 if result.error and "Unknown persona" in result.error else 500
    return JSONResponse(result.to_dict(), status_code=status_code)


@router.post("/api/demo/stop")
async def stop_demo(request: Request) -> JSONResponse:
    """Stop the demo target subprocess."""
    result = await demo_stop()
    return JSONResponse(result.to_dict())


@router.get("/api/demo/status")
async def get_demo_status(request: Request) -> JSONResponse:
    """Check demo target status and health."""
    result = await demo_status()
    return JSONResponse(result.to_dict())
