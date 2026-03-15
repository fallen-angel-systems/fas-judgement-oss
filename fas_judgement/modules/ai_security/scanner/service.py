"""
Attack Scanner Service
-----------------------
WHY: The service layer orchestrates the full attack run — pattern filtering,
     rate limiting, SSE streaming, result persistence. It's the "use case"
     layer in clean architecture terms.

     HTTP routes call this service; the service never imports FastAPI.
     This separation means the attack logic can be called from:
       - The HTTP endpoint (SSE streaming)
       - A CLI batch mode (future)
       - Integration tests (mocked dependencies)

LAYER: Module/Scanner service — imports core, modules, utils, config.
       No FastAPI imports.
SOURCE: Extracted from server.py /api/attack (lines ~800–1017).
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import AsyncIterator

import aiosqlite
import httpx

from ....config import DB_PATH
from .runner import execute_request, build_payload
from .scorer import classify_verdict, llm_classify


# === SECTION: PATTERN LOADING === #

def load_patterns_from_file(patterns_path) -> list:
    """Load all patterns from the JSON file."""
    import json
    if patterns_path.exists():
        with open(patterns_path) as f:
            return json.load(f)
    return []


def filter_patterns(
    patterns: list,
    categories: list = None,
    pattern_ids: list = None,
    severity_filter: list = None,
    category_limits: dict = None,
    include_custom: bool = False,
    custom_patterns_data: list = None,
    max_patterns: int = 0,
) -> list:
    """
    Apply user-specified filters to the pattern list.

    WHY many parameters: the /api/attack endpoint exposes fine-grained
    filter controls so users can target specific attack vectors without
    running the full 34k library.
    """
    import random

    if categories:
        patterns = [p for p in patterns if p["category"] in categories]
    elif not categories and include_custom:
        # User wants only custom patterns, not library patterns
        patterns = []

    if pattern_ids:
        patterns = [p for p in patterns if p["id"] in pattern_ids]

    if severity_filter:
        patterns = [p for p in patterns if p.get("severity", "medium") in severity_filter]

    # Per-category random sampling
    if category_limits and not pattern_ids:
        limited = []
        by_cat = {}
        for p in patterns:
            by_cat.setdefault(p.get("category", "unknown"), []).append(p)
        for cat, cat_patterns in by_cat.items():
            limit = category_limits.get(cat)
            if limit and limit > 0 and limit < len(cat_patterns):
                limited.extend(random.sample(cat_patterns, limit))
            else:
                limited.extend(cat_patterns)
        patterns = limited

    # Merge custom patterns from the browser (never stored server-side)
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
                "source": "custom"
            })

    if max_patterns and max_patterns > 0:
        patterns = patterns[:max_patterns]

    return patterns


# === SECTION: ATTACK EXECUTION === #

async def run_attack_stream(
    session_id: str,
    patterns: list,
    target_url: str,
    method: str,
    headers_raw: dict,
    payload_field: str,
    payload_template: str,
    delay_ms: int,
    timeout_s: float,
    use_llm: bool,
    use_mcp: bool,
    mcp_url: str,
    error_cutoff: int,
    user_id: str,
    request,  # FastAPI Request — for disconnect detection
) -> AsyncIterator[dict]:
    """
    Core attack execution generator — yields SSE events as each pattern runs.

    WHY AsyncIterator: SSE streaming. The HTTP route wraps this in
    EventSourceResponse and sends events to the browser in real-time.
    """
    stats = {"blocked": 0, "partial": 0, "bypassed": 0, "errors": 0}
    consecutive_errors = 0
    aborted = False

    yield {"event": "session", "data": json.dumps({"session_id": session_id, "total": len(patterns)})}

    async with httpx.AsyncClient(timeout=timeout_s, verify=False) as client:
        for i, pattern in enumerate(patterns):
            # Check client disconnect to avoid wasting resources
            if await request.is_disconnected():
                aborted = True
                break

            payload = build_payload(pattern["text"], payload_field, payload_template)
            status_code, response_text, elapsed = await execute_request(
                client, target_url, method, payload, headers_raw, timeout_s
            )

            verdict = classify_verdict(response_text, status_code)
            llm_reason = ""

            if use_llm and verdict != "ERROR":
                llm_verdict, llm_reason = await llm_classify(pattern["text"], response_text)
                if llm_verdict:
                    verdict = llm_verdict
                    llm_reason = llm_reason or ""

            # Track stats
            stats_key = verdict.lower() if verdict.lower() in stats else "errors"
            stats[stats_key] += 1

            result = {
                "index": i + 1,
                "total": len(patterns),
                "pattern_id": pattern["id"],
                "pattern_text": pattern["text"][:100],
                "category": pattern["category"],
                "status_code": status_code,
                "response_preview": response_text[:300],
                "response_time_ms": elapsed,
                "verdict": verdict,
                "llm_reason": llm_reason if use_llm else "",
                "stats": stats.copy()
            }

            # Optional MCP analysis
            if use_mcp and mcp_url:
                try:
                    mcp_resp = await client.post(mcp_url, json={
                        "text": pattern["text"], "verdict": verdict,
                        "response": response_text[:500], "category": pattern["category"]
                    }, timeout=5)
                    if mcp_resp.status_code == 200:
                        result["mcp_analysis"] = mcp_resp.json()
                except Exception:
                    pass

            # Error cutoff tracking
            if verdict == "ERROR":
                consecutive_errors += 1
                if error_cutoff and consecutive_errors >= error_cutoff:
                    result["cutoff"] = True
                    yield {"event": "result", "data": json.dumps(result)}
                    yield {"event": "error", "data": json.dumps({
                        "message": f"Auto-stopped: {consecutive_errors} consecutive errors"
                    })}
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
                    (session_id, pattern["id"], pattern["text"], pattern["category"],
                     json.dumps(payload), status_code, response_text, elapsed, verdict,
                     datetime.utcnow().isoformat())
                )
                await db.commit()

            if delay_ms > 0 and i < len(patterns) - 1:
                await asyncio.sleep(delay_ms / 1000)

    # Save final session stats
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET blocked=?, partial=?, bypassed=?, errors=? WHERE id=?",
            (stats["blocked"], stats["partial"], stats["bypassed"], stats["errors"], session_id)
        )
        await db.commit()

    if aborted:
        yield {"event": "aborted", "data": json.dumps({
            "session_id": session_id, "stats": stats,
            "message": "Attack stopped — client disconnected"
        })}
    else:
        yield {"event": "complete", "data": json.dumps({"session_id": session_id, "stats": stats})}
