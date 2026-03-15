"""
Core Router (OSS Edition)
--------------------------
WHY: Infrastructure endpoints — tier info, settings, presets, sessions,
     reports, and LLM/MCP connectivity tests.

     OSS simplifications vs Elite:
       ❌ No user_id scoping on presets/sessions (single-user, no accounts)
       ❌ No admin checks
       ❌ No display_name user DB updates in /api/settings
       ✅ All core endpoints for the frontend UI

LAYER: HTTP — imports FastAPI, aiosqlite, config.
"""

import json
import uuid
from datetime import datetime

import aiosqlite
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import DB_PATH, OLLAMA_URL, OLLAMA_MODEL
from ..utils.security import is_safe_url, parse_curl_command

router = APIRouter(tags=["core"])


# === SECTION: TIER INFO === #

@router.get("/api/tier")
async def get_tier_info():
    """
    Return current tier and feature flags for the frontend.

    WHY: The frontend adapts its UI based on tier (shows upgrade prompts,
    enables/disables multi-turn tab, etc.). The license client on disk
    is the source of truth.
    """
    try:
        from ..utils.license import get_tier, get_features, load_patterns
        tier = get_tier()
        features = get_features()
        patterns = load_patterns()
        is_elite = tier in ("elite_home", "elite_business")
        return {
            "tier": tier,
            "tier_display": tier.replace("_", " ").title() if tier != "free" else "Free",
            "features": features,
            "pattern_count": len(patterns),
            "is_elite": is_elite,
        }
    except Exception:
        # Fallback: always return a valid response so the frontend doesn't break
        return {
            "tier": "free",
            "tier_display": "Free",
            "features": {"multi_turn": False, "teams": False},
            "pattern_count": 100,
            "is_elite": False,
        }


# === SECTION: LICENSE ACTIVATION (from UI) === #


@router.post("/api/license/activate")
async def activate_license_endpoint(request: Request):
    """
    Activate a license key from the frontend Settings panel.

    WHY in the OSS build: Users can paste their Elite license key in the
    browser UI instead of stopping the server and running the CLI command.
    The license client handles validation + pattern sync.
    """
    try:
        from ..utils.license import validate_license, sync_patterns
        body = await request.json()
        key = body.get("key", "").strip().upper()
        if not key:
            return JSONResponse({"valid": False, "error": "No key provided"}, status_code=400)

        result = validate_license(key)
        if result.get("valid"):
            # Sync patterns immediately after successful validation
            sync = sync_patterns(key)
            result["patterns_synced"] = sync.get("success", False)
            result["pattern_count"] = sync.get("count", 0)

        return result
    except Exception as e:
        return JSONResponse({"valid": False, "error": str(e)}, status_code=500)


@router.post("/api/license/deactivate")
async def deactivate_license_endpoint():
    """Deactivate the current license from the frontend Settings panel."""
    try:
        from ..utils.license import deactivate
        deactivate()
        return {"status": "deactivated", "tier": "free"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# === SECTION: CURL PARSER === #

@router.post("/api/parse-curl")
async def parse_curl(request: Request):
    """Parse a pasted cURL command into scan configuration fields."""
    body = await request.json()
    curl_str = body.get("curl", "").strip()
    if not curl_str:
        return JSONResponse({"error": "curl command required"}, status_code=400)
    try:
        return parse_curl_command(curl_str)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# === SECTION: PRESETS === #

@router.get("/api/presets")
async def list_presets():
    """List all saved target presets."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM presets ORDER BY name")
        return {"presets": [dict(r) for r in await cursor.fetchall()]}


@router.post("/api/presets")
async def save_preset(request: Request):
    """Save a new target preset."""
    body = await request.json()
    preset_id = str(uuid.uuid4())[:8]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO presets (id, name, target_url, method, headers, payload_field,
               payload_template, delay_ms, timeout_s, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                preset_id,
                body.get("name", "Untitled"),
                body.get("target_url", ""),
                body.get("method", "POST"),
                json.dumps(body.get("headers", {})) if isinstance(body.get("headers"), dict) else body.get("headers", "{}"),
                body.get("payload_field", "message"),
                body.get("payload_template", ""),
                body.get("delay_ms", 1000),
                body.get("timeout_s", 10),
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()
    return {"id": preset_id, "status": "saved"}


@router.delete("/api/presets/{preset_id}")
async def delete_preset(preset_id: str):
    """Delete a preset by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM presets WHERE id=?", (preset_id,))
        await db.commit()
    return {"status": "deleted"}


# === SECTION: SESSIONS === #

@router.delete("/api/sessions")
async def clear_sessions():
    """Clear all sessions and results."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM results")
        await db.execute("DELETE FROM sessions")
        await db.commit()
    return {"status": "cleared"}


@router.get("/api/sessions")
async def list_sessions():
    """List recent sessions (most recent first, limit 50)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 50"
        )
        return {"sessions": [dict(r) for r in await cursor.fetchall()]}


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session with its full result set."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        session = await cursor.fetchone()
        if not session:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        cursor = await db.execute(
            "SELECT * FROM results WHERE session_id=? ORDER BY id", (session_id,)
        )
        results = await cursor.fetchall()
        return {"session": dict(session), "results": [dict(r) for r in results]}


@router.get("/api/sessions/{session_id}/report")
async def generate_report(
    session_id: str,
    format: str = "markdown",
    client_name: str = "Target Organization",
    product_name: str = "",
    assessor: str = "Fallen Angel Systems",
    classification: str = "CONFIDENTIAL",
    scope_notes: str = "",
):
    """Generate a professional report for a session (markdown, html, json, sarif)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        session = await cursor.fetchone()
        if not session:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        cursor = await db.execute(
            "SELECT * FROM results WHERE session_id=? ORDER BY id", (session_id,)
        )
        all_results = [dict(r) for r in await cursor.fetchall()]
        session = dict(session)

    config = {
        "client_name": client_name,
        "product_name": product_name,
        "assessor_name": assessor,
        "classification": classification,
        "scope_notes": scope_notes,
    }

    from ..core.report import (
        generate_professional_html,
        generate_json_report,
        generate_sarif_report,
        generate_markdown_report,
    )

    if format == "html":
        return HTMLResponse(generate_professional_html(session, all_results, config))
    elif format == "json":
        return JSONResponse(generate_json_report(session, all_results))
    elif format == "sarif":
        return JSONResponse(generate_sarif_report(session, all_results))
    else:
        md = generate_markdown_report(session, all_results)
        return {"report": md, "session": session}


# === SECTION: SETTINGS === #

@router.get("/api/settings")
async def get_settings():
    """Get all saved settings (KV store)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM settings")
        rows = await cursor.fetchall()
        return {"settings": {r["key"]: r["value"] for r in rows}}


@router.post("/api/settings")
async def save_settings(request: Request):
    """Save settings (key-value pairs). All values stored as strings."""
    body = await request.json()
    async with aiosqlite.connect(DB_PATH) as db:
        for k, v in body.items():
            await db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (k, str(v)),
            )
        await db.commit()
    return {"status": "saved"}


# === SECTION: LLM / MCP CONNECTIVITY === #

@router.post("/api/settings/llm-test")
async def test_llm(request: Request):
    """Test Ollama connectivity and model availability."""
    body = await request.json()
    url = body.get("ollama_url", OLLAMA_URL)
    model = body.get("ollama_model", OLLAMA_MODEL)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return {
                    "status": "connected",
                    "models": models,
                    "model_found": any(model in m for m in models),
                }
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)[:200]}


@router.post("/api/settings/mcp-test")
async def test_mcp(request: Request):
    """Test MCP endpoint connectivity."""
    body = await request.json()
    url = body.get("mcp_url", "")
    if not url:
        return {"status": "error", "message": "No MCP URL provided"}
    if not is_safe_url(url):
        return {"status": "error", "message": "URL points to a private/internal address"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json={"text": "test", "action": "ping"})
            if resp.status_code == 200:
                return {"status": "connected"}
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)[:200]}
