"""
Scan & Attack Router (OSS Edition)
------------------------------------
WHY: The attack endpoint fires patterns at a target and streams results via SSE.
     The scan-target endpoint auto-detects AI endpoint configuration.

     OSS simplifications vs Elite:
       ❌ No per-user rate limiting (global rate limiter instead)
       ❌ No user_id tracked on sessions
       ✅ Full attack logic preserved (pattern selection, SSE streaming, DB storage)
       ✅ scan-target auto-detection preserved
       ✅ Client disconnect detection (aborted event)

LAYER: HTTP — imports FastAPI, httpx, aiosqlite, SSE.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

import aiosqlite
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..config import DB_PATH, OLLAMA_URL, OLLAMA_MODEL, ALLOW_LOCAL_TARGETS
from .deps import rate_limiter
from ..utils.security import is_safe_url
from ..modules.ai_security.scanner.scorer import classify_verdict, llm_classify
from ..utils.license import load_patterns

router = APIRouter(tags=["scan"])


# === SECTION: ATTACK ENDPOINT === #

@router.post("/api/attack")
async def start_attack(request: Request):
    """
    Fire prompt injection patterns at a target and stream results via SSE.

    WHY SSE: real-time progress updates without WebSocket complexity.
    Each pattern result is yielded immediately — the frontend sees results
    as they happen, not batched at the end.

    Flow:
      1. Rate limit check (global)
      2. Validate target URL (SSRF protection)
      3. Load & filter patterns (by category, severity, custom data)
      4. Create session in DB
      5. Stream results as SSE: session → result* → complete|aborted
    """
    # Global rate limit (no per-user in OSS)
    allowed, reason = rate_limiter.can_attack()
    if not allowed:
        return JSONResponse({"error": reason}, status_code=429)

    body = await request.json()
    target_url = body.get("target_url", "")
    method = body.get("method", "POST").upper()
    headers_raw = body.get("headers", {})
    payload_field = body.get("payload_field", "message")
    categories = body.get("categories", [])
    pattern_ids = body.get("pattern_ids", [])
    delay_ms = rate_limiter.enforce_min_delay(body.get("delay_ms", 200))
    timeout_s = body.get("timeout_s", 10)
    max_patterns = body.get("max_patterns", 0)
    error_cutoff = body.get("error_cutoff", 5)

    if not target_url:
        return JSONResponse({"error": "target_url required"}, status_code=400)

    if not is_safe_url(target_url, allow_local=ALLOW_LOCAL_TARGETS):
        return JSONResponse(
            {"error": "Target URL points to a private/internal address"},
            status_code=400,
        )

    severity_filter = body.get("severity_filter", [])
    category_limits = body.get("category_limits", {})
    include_custom = body.get("include_custom", False)

    # === Pattern Selection ===
    # WHY complex: users can filter by category, severity, per-category limits,
    # specific IDs, and merge custom patterns from browser localStorage.
    patterns = load_patterns()
    if categories:
        patterns = [p for p in patterns if p["category"] in categories]
    elif not categories and include_custom:
        # No library categories selected but custom requested → skip library
        patterns = []
    if pattern_ids:
        patterns = [p for p in patterns if p["id"] in pattern_ids]

    if severity_filter:
        patterns = [p for p in patterns if p.get("severity", "medium") in severity_filter]

    # Per-category limits: sample N patterns from each category
    if category_limits and not pattern_ids:
        import random
        limited = []
        by_cat: dict = {}
        for p in patterns:
            by_cat.setdefault(p.get("category", "unknown"), []).append(p)
        for cat, cat_patterns in by_cat.items():
            limit = category_limits.get(cat)
            if limit and 0 < limit < len(cat_patterns):
                limited.extend(random.sample(cat_patterns, limit))
            else:
                limited.extend(cat_patterns)
        patterns = limited

    # Merge custom patterns from browser localStorage (never stored server-side)
    # WHY client-side only: custom patterns are per-user, ephemeral, potentially
    # sensitive. Storing them server-side adds privacy/security complexity with
    # no benefit for a single-user local tool.
    custom_patterns_data = body.get("custom_patterns_data", [])
    if include_custom and custom_patterns_data:
        for cp in custom_patterns_data[:500]:
            cp_text = (cp.get("text") or "").strip()
            if not cp_text or len(cp_text) > 10000:
                continue
            patterns.append({
                "id": cp.get("id", "cp-unknown"),
                "text": cp_text,
                "category": (cp.get("category") or "custom").strip(),
                "tier": "custom",
                "source": "custom",
            })

    # Legacy custom_patterns format (backward compat with v2.0.0)
    custom_patterns = body.get("custom_patterns", [])
    if custom_patterns and not custom_patterns_data:
        for cp in custom_patterns:
            if isinstance(cp, dict) and cp.get("text"):
                patterns.append({
                    "id": cp.get("id", f"MY-{len(patterns)+1}"),
                    "text": cp["text"],
                    "category": "my_patterns",
                })

    if not patterns:
        return JSONResponse({"error": "No patterns match the selection"}, status_code=400)

    if max_patterns and max_patterns > 0:
        patterns = patterns[:max_patterns]

    session_id = str(uuid.uuid4())[:8]

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (id, target_url, method, created_at, total_patterns) VALUES (?, ?, ?, ?, ?)",
            (session_id, target_url, method, datetime.utcnow().isoformat(), len(patterns)),
        )
        await db.commit()

    # Register with rate limiter
    rate_limiter.start_attack(session_id)

    async def event_generator():
        """
        SSE generator: fires each pattern and yields result events.

        WHY generator: FastAPI's EventSourceResponse expects an async generator.
        Each yield becomes an SSE event the frontend processes in real-time.
        """
        stats = {"blocked": 0, "partial": 0, "bypassed": 0, "errors": 0}
        consecutive_errors = 0
        aborted = False

        yield {
            "event": "session",
            "data": json.dumps({"session_id": session_id, "total": len(patterns)}),
        }

        async with httpx.AsyncClient(timeout=timeout_s, verify=False) as client:
            for i, pattern in enumerate(patterns):
                # Check if client disconnected mid-attack
                if await request.is_disconnected():
                    aborted = True
                    break

                # Build the payload (template or simple field)
                payload_template = body.get("payload_template", "")
                if payload_template:
                    try:
                        payload = json.loads(
                            payload_template.replace(
                                "{{PAYLOAD}}",
                                pattern["text"].replace('"', '\\"').replace("\n", "\\n"),
                            )
                        )
                    except json.JSONDecodeError:
                        payload = {payload_field: pattern["text"]}
                else:
                    payload = {payload_field: pattern["text"]}

                start = time.time()
                try:
                    if method == "POST":
                        resp = await client.post(target_url, json=payload, headers=headers_raw)
                    elif method == "GET":
                        resp = await client.get(
                            target_url,
                            params={payload_field: pattern["text"]},
                            headers=headers_raw,
                        )
                    else:
                        resp = await client.request(method, target_url, json=payload, headers=headers_raw)

                    elapsed = (time.time() - start) * 1000
                    response_text = resp.text[:5000]
                    verdict = classify_verdict(response_text, resp.status_code)
                    llm_reason = ""
                    if body.get("use_llm", False):
                        llm_verdict, llm_reason = await llm_classify(pattern["text"], response_text)
                        if llm_verdict:
                            verdict = llm_verdict
                            llm_reason = llm_reason or ""
                    status_code = resp.status_code

                except httpx.TimeoutException:
                    elapsed = timeout_s * 1000
                    response_text = "TIMEOUT"
                    verdict = "ERROR"
                    status_code = 0
                except Exception as e:
                    elapsed = (time.time() - start) * 1000
                    response_text = str(e)[:500]
                    verdict = "ERROR"
                    status_code = 0

                verdict_key = verdict.lower() if verdict.lower() in stats else "errors"
                stats[verdict_key] += 1

                # MCP analysis — optional external scoring
                mcp_result = ""
                if body.get("use_mcp") and body.get("mcp_url"):
                    try:
                        mcp_resp = await client.post(
                            body["mcp_url"],
                            json={
                                "payload": pattern["text"][:500],
                                "response": response_text[:500],
                                "verdict": verdict,
                                "category": pattern["category"],
                            },
                            timeout=10,
                        )
                        if mcp_resp.status_code == 200:
                            mcp_data = mcp_resp.json()
                            mcp_result = mcp_data.get(
                                "analysis", mcp_data.get("result", str(mcp_data)[:200])
                            )
                            if mcp_data.get("verdict", "").upper() in ("BLOCKED", "BYPASS", "PARTIAL"):
                                verdict = mcp_data["verdict"].upper()
                    except Exception:
                        mcp_result = "MCP error"

                result = {
                    "index": i + 1,
                    "total": len(patterns),
                    "pattern_id": pattern["id"],
                    "pattern_text": pattern["text"][:100],
                    "category": pattern["category"],
                    "status_code": status_code,
                    "response_preview": response_text[:300],
                    "response_time_ms": round(elapsed, 1),
                    "verdict": verdict,
                    "llm_reason": llm_reason if body.get("use_llm") else "",
                    "mcp_result": mcp_result if body.get("use_mcp") else "",
                    "stats": stats.copy(),
                }

                # Error cutoff: auto-stop after N consecutive errors
                if verdict_key == "errors":
                    consecutive_errors += 1
                    if error_cutoff and consecutive_errors >= error_cutoff:
                        result["cutoff"] = True
                        yield {"event": "result", "data": json.dumps(result)}
                        yield {
                            "event": "error_cutoff",
                            "data": json.dumps({
                                "message": f"Stopped: {consecutive_errors} consecutive errors. "
                                           f"Target may be rate limiting or key may be invalid.",
                                "stats": stats.copy(),
                            }),
                        }
                        break
                else:
                    consecutive_errors = 0

                yield {"event": "result", "data": json.dumps(result)}

                # Persist result to DB
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute(
                        """INSERT INTO results
                           (session_id, pattern_id, pattern_text, category, request_body,
                            response_status, response_body, response_time_ms, verdict, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            session_id, pattern["id"], pattern["text"], pattern["category"],
                            json.dumps(payload), status_code, response_text, elapsed, verdict,
                            datetime.utcnow().isoformat(),
                        ),
                    )
                    await db.commit()

                if delay_ms > 0 and i < len(patterns) - 1:
                    await asyncio.sleep(delay_ms / 1000)

        # Finalise session stats
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE sessions SET blocked=?, partial=?, bypassed=?, errors=? WHERE id=?",
                (stats["blocked"], stats["partial"], stats["bypassed"], stats["errors"], session_id),
            )
            await db.commit()

        # Release rate limiter slot
        rate_limiter.end_attack(session_id)

        if aborted:
            yield {
                "event": "aborted",
                "data": json.dumps({
                    "session_id": session_id,
                    "stats": stats,
                    "message": "Attack stopped — client disconnected",
                }),
            }
        else:
            yield {
                "event": "complete",
                "data": json.dumps({"session_id": session_id, "stats": stats}),
            }

    return EventSourceResponse(event_generator())


# === SECTION: SCAN TARGET (AUTO-DETECT) === #

@router.post("/api/scan-target")
async def scan_target(request: Request):
    """
    Auto-detect AI endpoint configuration by probing a target URL.

    WHY: Most users don't know the exact payload format their AI endpoint
    expects. This scans for common patterns (OpenAI, Anthropic, custom)
    and returns a ready-to-use config.

    Flow:
      1. Rate limit check
      2. Determine if URL is a webpage or API endpoint
      3. If webpage: look for chat widget indicators in HTML
      4. If API: probe with different payload formats
      5. Try common AI endpoint paths (/v1/chat/completions, /api/chat, etc.)
    """
    allowed, reason = rate_limiter.can_scan()
    if not allowed:
        return JSONResponse({"error": reason}, status_code=429)
    rate_limiter.record_scan()

    body = await request.json()
    url = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)

    if not is_safe_url(url, allow_local=ALLOW_LOCAL_TARGETS):
        return JSONResponse(
            {"error": "Target URL points to a private/internal address"}, status_code=400
        )

    result = {
        "method": "POST",
        "field": "message",
        "headers": {},
        "payload_template": "",
        "detected_provider": "",
        "endpoints_found": [],
        "confidence": "low",
        "info": "",
    }

    common_paths = ["/chat", "/v1/chat/completions", "/api/chat", "/completions", "/generate", "/ask"]
    probe_payloads = [
        ("messages", {"messages": [{"role": "user", "content": "hello"}]}),
        ("prompt", {"prompt": "hello"}),
        ("text", {"text": "hello"}),
        ("message", {"message": "hello"}),
        ("input", {"input": "hello"}),
        ("query", {"query": "hello"}),
    ]

    async with httpx.AsyncClient(timeout=10, verify=False, follow_redirects=True) as client:
        # Step 1: Determine if URL is a webpage or API
        is_webpage = False
        try:
            resp = await client.get(url)
            ct = resp.headers.get("content-type", "")
            if "text/html" in ct:
                is_webpage = True
        except Exception as e:
            return JSONResponse(
                {"error": f"Could not reach target: {str(e)[:200]}"}, status_code=400
            )

        # Step 2: Webpage — scan for AI chat indicators
        if is_webpage:
            html_lower = resp.text.lower()
            indicators = []
            if any(x in html_lower for x in ["chatwidget", "chat-widget", "intercom", "drift", "crisp", "tidio", "zendesk"]):
                indicators.append("chat widget detected")
            if any(x in html_lower for x in ["openai", "api.openai.com", "anthropic", "claude"]):
                indicators.append("AI API references found")
            if any(x in html_lower for x in ["/v1/chat/completions", "/api/chat", "/completions"]):
                indicators.append("AI endpoint paths in source")
            if "<iframe" in html_lower:
                indicators.append("iframe embeds found")
            result["info"] = (
                "Webpage with AI indicators: " + ", ".join(indicators)
                if indicators
                else "Webpage detected but no AI chat indicators found."
            )
            return result

        # Step 3: API endpoint — probe with payload formats
        base_url = url.rstrip("/")

        async def probe_endpoint(probe_url):
            """Try different payload formats against an endpoint."""
            for field_name, payload in probe_payloads:
                try:
                    r = await client.post(
                        probe_url, json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    if r.status_code < 500:
                        resp_text = r.text[:2000].lower()
                        provider = ""
                        if any(x in resp_text for x in ['"choices"', '"model"', "gpt-", "openai"]):
                            provider = "openai"
                        elif any(x in resp_text for x in ['"completion"', "anthropic", "claude"]):
                            provider = "anthropic"
                        elif any(x in resp_text for x in ['"response"', '"output"', '"text"', '"result"', '"generated']):
                            provider = "custom"

                        is_auth_error = r.status_code in (401, 403)
                        is_validation = r.status_code == 422
                        is_success = r.status_code < 400 and (provider or len(r.text.strip()) > 2)

                        if is_auth_error:
                            resp_raw = r.text[:2000]
                            if "x-api-key" in resp_raw.lower():
                                result["headers"] = {
                                    "Content-Type": "application/json",
                                    "X-API-Key": "YOUR_API_KEY",
                                }
                            else:
                                result["headers"] = {
                                    "Content-Type": "application/json",
                                    "Authorization": "Bearer YOUR_API_KEY",
                                }

                        if is_success or is_auth_error or is_validation:
                            template = ""
                            if field_name == "messages":
                                template = '{"messages":[{"role":"user","content":"{{PAYLOAD}}"}]}'
                            elif field_name != "message":
                                template = '{"%s":"{{PAYLOAD}}"}' % field_name
                            return field_name, template, provider, r.status_code
                except Exception:
                    pass
            return None

        # Probe directly first
        direct = await probe_endpoint(base_url)
        if direct:
            field_name, template, provider, status = direct
            result["field"] = field_name
            result["payload_template"] = template
            result["detected_provider"] = provider or "unknown"
            result["endpoints_found"].append(base_url)
            result["confidence"] = "high" if provider else "medium"
            auth_note = " (API key required)" if status in (401, 403) else ""
            result["info"] = f"Detected {provider or 'API'} at {base_url}, {field_name} format{auth_note}"
            if not result.get("headers"):
                result["headers"] = {"Content-Type": "application/json"}
            return result

        # Probe common paths
        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        for path in common_paths:
            probe_url = origin + path
            try:
                found = await probe_endpoint(probe_url)
                if found:
                    result["endpoints_found"].append(probe_url)
                    if not result["detected_provider"]:
                        field_name, template, provider, status = found
                        result["field"] = field_name
                        result["payload_template"] = template
                        result["detected_provider"] = provider or "unknown"
                        result["confidence"] = "medium" if provider else "low"
                        result["info"] = f"Detected {provider or 'unknown'} API at {probe_url}, {field_name} format"
            except Exception:
                pass

        if result["endpoints_found"]:
            return result

        result["info"] = "Could not auto-detect config. Please configure manually."
        return result
