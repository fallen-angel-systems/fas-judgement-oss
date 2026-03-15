"""
HTTP Transport
--------------
WHY: The HTTP transport is the primary delivery mechanism for single-shot
     and multi-turn attacks against REST/chat API endpoints.

     Key design decisions:
       - Persistent httpx.AsyncClient with cookie jar: multi-turn sessions
         that set cookies (e.g. session auth) retain them across turns.
       - Conversation ID tracking: many AI APIs return a conversation_id
         that must be passed back on subsequent messages. We auto-detect
         common field names so the user doesn't need to configure this.

LAYER: Transport (infrastructure) — imports httpx, json, stdlib.
SOURCE: Extracted from multi-turn-engine/transport.py HTTPTransport class.
"""

import json
from typing import Optional

import httpx

from .base import BaseTransport


class HTTPTransport(BaseTransport):
    """
    Send attacks to HTTP API endpoints.

    Maintains a persistent client with cookie jar and optional conversation ID
    for multi-turn state. The same httpx.AsyncClient is reused across sends
    so session cookies, auth tokens, and conversation context survive.
    """

    transport_type = "http"

    def __init__(
        self,
        target_url: str,
        method: str = "POST",
        body_field: str = "message",
        response_field: str = "response",
        headers: Optional[dict] = None,
        timeout: float = 30.0,
        conversation_id_field: str = "",
        session_id_field: str = "",
    ):
        self.target_url = target_url
        self.method = method
        self.body_field = body_field
        self.response_field = response_field
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.conversation_id_field = conversation_id_field
        self.session_id_field = session_id_field
        # Persistent state across turns
        self._client: Optional[httpx.AsyncClient] = None
        self._conversation_id: Optional[str] = None
        self._turn_count: int = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the persistent HTTP client with cookie jar."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def send(self, message: str) -> str:
        """
        Send attack message to the HTTP endpoint.
        Maintains session state (cookies, conversation ID) across turns.
        """
        client = await self._get_client()
        self._turn_count += 1

        # Build request body
        body = {self.body_field: message}

        # Include conversation/session ID if the target API uses one
        if self.conversation_id_field and self._conversation_id:
            body[self.conversation_id_field] = self._conversation_id
        if self.session_id_field and self._conversation_id:
            body[self.session_id_field] = self._conversation_id

        if self.method.upper() == "POST":
            resp = await client.post(self.target_url, json=body, headers=self.headers)
        else:
            resp = await client.get(self.target_url, params=body, headers=self.headers)

        resp.raise_for_status()
        data = resp.json()

        # Capture conversation/session ID from response for subsequent turns
        for id_field in [self.conversation_id_field, self.session_id_field]:
            if id_field and id_field in data:
                self._conversation_id = str(data[id_field])

        # Auto-detect common ID fields if we don't have one yet
        for auto_field in ("conversation_id", "session_id", "thread_id", "chat_id"):
            if auto_field in data and not self._conversation_id:
                self._conversation_id = str(data[auto_field])

        if self.response_field in data:
            return str(data[self.response_field])

        return json.dumps(data)

    async def check_connection(self) -> dict:
        """Ping the target endpoint."""
        try:
            client = await self._get_client()
            resp = await client.options(self.target_url)
            return {"connected": True, "url": self.target_url, "status_code": resp.status_code}
        except Exception as e:
            return {"connected": False, "url": self.target_url, "error": str(e)}

    async def close(self):
        """Cleanup persistent client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def to_dict(self) -> dict:
        return {
            "type": "http",
            "target_url": self.target_url,
            "method": self.method,
            "body_field": self.body_field,
            "response_field": self.response_field,
            "conversation_id_field": self.conversation_id_field,
            "session_id_field": self.session_id_field,
            "turn_count": self._turn_count,
        }
