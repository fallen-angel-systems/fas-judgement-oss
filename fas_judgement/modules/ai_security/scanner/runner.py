"""
HTTP Request Runner
--------------------
WHY: The runner handles the low-level HTTP execution of each attack request.
     Isolating it here means:
       - The service layer stays clean (no httpx boilerplate)
       - The runner can be mocked in tests without spinning up a real HTTP server
       - SSE timeout/error handling lives in one place

LAYER: Module/Scanner — imports httpx, stdlib. No FastAPI imports.
SOURCE: Extracted from server.py /api/attack event_generator (lines ~870–990).
"""

import json
import time
from typing import AsyncIterator

import httpx


async def execute_request(
    client: httpx.AsyncClient,
    target_url: str,
    method: str,
    payload: dict,
    headers: dict,
    timeout_s: float = 10.0,
) -> tuple[int, str, float]:
    """
    Execute a single HTTP attack request.

    Returns (status_code, response_text, elapsed_ms).
    On timeout/error: returns (0, error_message, elapsed_ms).

    WHY return a tuple rather than raising: the service layer needs to
    distinguish between "request errored" (ERROR verdict) and "model refused"
    (BLOCKED verdict). Raising would conflate the two.
    """
    start = time.time()
    try:
        method_upper = method.upper()
        if method_upper == "POST":
            resp = await client.post(target_url, json=payload, headers=headers)
        elif method_upper == "GET":
            # GET: use the payload_field as a query param
            resp = await client.get(target_url, params=payload, headers=headers)
        else:
            resp = await client.request(method_upper, target_url, json=payload, headers=headers)

        elapsed = (time.time() - start) * 1000
        return resp.status_code, resp.text[:5000], round(elapsed, 1)

    except httpx.TimeoutException:
        elapsed = timeout_s * 1000
        return 0, "TIMEOUT", round(elapsed, 1)
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return 0, str(e)[:500], round(elapsed, 1)


def build_payload(
    pattern_text: str,
    payload_field: str,
    payload_template: str,
) -> dict:
    """
    Build the request body for a single attack pattern.

    If a payload_template is provided, replace {{PAYLOAD}} with the pattern text.
    Otherwise, use {payload_field: pattern_text} as a simple dict.

    WHY template support: some APIs require complex JSON bodies
    (e.g. OpenAI-style messages arrays). The template lets users configure
    the exact shape of the request.
    """
    if payload_template:
        try:
            # Safely substitute the placeholder without breaking JSON
            safe_text = pattern_text.replace('"', '\\"').replace('\n', '\\n')
            return json.loads(payload_template.replace("{{PAYLOAD}}", safe_text))
        except json.JSONDecodeError:
            pass
    return {payload_field: pattern_text}
