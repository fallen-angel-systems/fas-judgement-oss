"""
Ollama Connection Helpers
--------------------------
WHY: Ollama connectivity checks are used in two places — the settings UI
     and the multi-turn scorer. Extracting them to utils prevents duplication
     and keeps both callers free of httpx boilerplate.

LAYER: Utils — imports httpx (external) and config.
"""

import httpx

from ..config import OLLAMA_URL


# === SECTION: CONNECTION CHECK === #

async def check_ollama_connection(
    url: str = None,
    model: str = None,
    timeout: float = 10.0,
) -> dict:
    """
    Verify Ollama is running and optionally check that a specific model
    is available.

    Returns a dict:
      {"status": "connected", "models": [...], "model_found": True/False}
      {"status": "error", "message": "..."}
    """
    url = url or OLLAMA_URL
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                result = {
                    "status": "connected",
                    "url": url,
                    "models": models,
                }
                if model:
                    result["model_found"] = any(model in m for m in models)
                return result
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)[:200]}
