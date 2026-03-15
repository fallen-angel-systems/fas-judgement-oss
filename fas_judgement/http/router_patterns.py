"""
Patterns Router (OSS Edition)
-------------------------------
WHY: Pattern endpoints for the OSS version:
  - GET /api/patterns — return all patterns (no tier gating server-side;
    the license client handles tier-based loading on startup)
  - POST /api/patterns/submit — proxy to hosted Judgement API for review

     OSS simplifications vs Elite:
       ❌ No custom patterns CRUD (no user accounts to own them)
       ❌ No changelog endpoint (no Guardian scoring pipeline)
       ❌ No leaderboard (no user identities)
       ❌ No submissions list (proxy-only, no local DB storage)
       ✅ Full pattern library returned (what the license client loaded)
       ✅ Pattern submission proxied to hosted API

LAYER: HTTP — imports FastAPI, httpx, patterns service.
"""

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config import SUBMIT_API_URL
from ..utils.license import load_patterns

router = APIRouter(tags=["patterns"])


# === SECTION: PATTERN LIBRARY === #

@router.get("/api/patterns")
async def get_patterns():
    """
    Return the full pattern library available to this install.

    WHY no tier gating here: the license client (utils/license.py) already
    determined which patterns to load at startup. load_patterns() returns
    the right set based on the cached license state.

    Free tier → 100 bundled patterns
    Elite tier → 1000+ synced patterns from cached file
    """
    patterns = load_patterns()
    categories: dict = {}
    for p in patterns:
        cat = p.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    return {"patterns": patterns, "categories": categories, "total": len(patterns)}


# === SECTION: PATTERN SUBMISSION (PROXY) === #

@router.post("/api/patterns/submit")
async def submit_pattern(request: Request):
    """
    Proxy a community pattern submission to the hosted Judgement API.

    WHY proxy (not local): pattern deduplication, Guardian scoring, and
    review workflow all live server-side. The OSS app is just a submission
    client — it forwards the payload and returns the server's response.
    """
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    if len(text) > 10000:
        return JSONResponse(
            {"error": "Pattern too long (max 10,000 chars)"}, status_code=400
        )

    submission = {
        "text": text,
        "category": body.get("category", "auto"),
        "target_type": body.get("target_type", "chatbot"),
        "description": body.get("description", ""),
        "submitter_name": body.get("submitter_name", "anonymous"),
        "source": "oss",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{SUBMIT_API_URL}/api/patterns/submit",
                json=submission,
            )
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(
                {"error": "Submission failed", "status": resp.status_code},
                status_code=resp.status_code,
            )
    except Exception as e:
        return JSONResponse(
            {"error": f"Could not reach submission server: {str(e)[:200]}"},
            status_code=503,
        )
