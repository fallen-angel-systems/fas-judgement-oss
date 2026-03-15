"""
Ollama Transport
----------------
WHY: Ollama transport lets users test their local models as targets — useful
     for testing a custom model's safety boundaries before deploying it.

     The Ollama /api/generate endpoint accepts a `context` array that carries
     conversation state (token IDs). We save and pass this back each turn to
     maintain multi-turn conversation memory without needing the full message
     history.

LAYER: Transport (infrastructure) — imports httpx.
SOURCE: Extracted from multi-turn-engine/transport.py OllamaTransport class.
"""

from typing import Optional

import httpx

from .base import BaseTransport


class OllamaTransport(BaseTransport):
    """Send attacks via Ollama (for testing local models as targets)."""

    transport_type = "local"

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:14b",
        system_prompt: str = "",
        timeout: float = 60.0,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        # Ollama-specific: context carries conversation token history
        self.context: list = []

    async def send(self, message: str) -> str:
        """Send attack to a local Ollama model."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "model": self.model,
                "prompt": message,
                "stream": False,
            }
            if self.system_prompt:
                payload["system"] = self.system_prompt
            if self.context:
                payload["context"] = self.context

            resp = await client.post(f"{self.ollama_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Save context for next turn (Ollama-specific state)
            self.context = data.get("context", [])
            return data.get("response", "")

    async def check_connection(self) -> dict:
        """Check Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                return {
                    "connected": True,
                    "url": self.ollama_url,
                    "model": self.model,
                    "model_available": self.model in models,
                    "all_models": models,
                }
        except Exception as e:
            return {"connected": False, "url": self.ollama_url, "error": str(e)}

    def to_dict(self) -> dict:
        return {
            "type": "local",
            "ollama_url": self.ollama_url,
            "model": self.model,
            "system_prompt": self.system_prompt,
        }
