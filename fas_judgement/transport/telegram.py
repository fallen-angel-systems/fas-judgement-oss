"""
Telegram Transport
------------------
WHY: Telegram bots are widely deployed for customer support and internal
     tooling. Long-polling via getUpdates is the standard approach for
     bots that don't have webhooks configured.

LAYER: Transport (infrastructure) — imports httpx, asyncio.
SOURCE: Extracted from multi-turn-engine/transport.py TelegramTransport class.
"""

import asyncio
import time

import httpx

from .base import BaseTransport


class TelegramTransport(BaseTransport):
    """Send attacks to Telegram bots via the Bot API."""

    transport_type = "telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 30.0):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._last_update_id = 0

    async def send(self, message: str) -> str:
        """Send message to chat, poll for target bot response."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Flush pending updates first
            await client.get(
                f"{self.base_url}/getUpdates",
                params={"offset": -1, "limit": 1, "timeout": 0},
            )
            # Send the attack message
            send_resp = await client.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": message},
            )
            send_resp.raise_for_status()
            sent_msg_id = send_resp.json()["result"]["message_id"]

            # Poll for response (long polling)
            start = time.time()
            while time.time() - start < self.timeout:
                updates_resp = await client.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 3,
                        "allowed_updates": '["message"]',
                    },
                )
                updates_resp.raise_for_status()
                for update in updates_resp.json().get("result", []):
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    if (
                        str(msg.get("chat", {}).get("id")) == str(self.chat_id)
                        and msg.get("message_id", 0) > sent_msg_id
                        and not msg.get("from", {}).get("is_bot") is False
                    ):
                        return msg.get("text", "[non-text response]")

        return f"[TIMEOUT: No response within {int(self.timeout)}s]"

    async def check_connection(self) -> dict:
        """Verify bot token and chat access."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                me_resp = await client.get(f"{self.base_url}/getMe")
                me_resp.raise_for_status()
                bot_info = me_resp.json().get("result", {})
                chat_resp = await client.get(
                    f"{self.base_url}/getChat", params={"chat_id": self.chat_id}
                )
                chat_resp.raise_for_status()
                chat_info = chat_resp.json().get("result", {})
                return {
                    "connected": True,
                    "bot_name": bot_info.get("username"),
                    "chat_title": chat_info.get("title") or chat_info.get("first_name"),
                    "chat_id": self.chat_id,
                }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def to_dict(self) -> dict:
        return {"type": "telegram", "chat_id": self.chat_id}
