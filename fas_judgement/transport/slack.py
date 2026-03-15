"""
Slack Transport
---------------
WHY: Slack bots (Bolt apps) are deployed in many enterprise environments.
     We use the Web API to post messages and poll conversations.history
     for bot responses.

LAYER: Transport (infrastructure) — imports httpx, asyncio.
SOURCE: Extracted from multi-turn-engine/transport.py SlackTransport class.
"""

import asyncio
import time

import httpx

from .base import BaseTransport


class SlackTransport(BaseTransport):
    """Send attacks to Slack bots via the Web API."""

    transport_type = "slack"

    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        target_bot_id: str = "",
        timeout: float = 30.0,
    ):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.target_bot_id = target_bot_id
        self.timeout = timeout
        self.base_url = "https://slack.com/api"

    async def send(self, message: str) -> str:
        """Send message to channel, poll for target bot response."""
        headers = {"Authorization": f"Bearer {self.bot_token}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            send_resp = await client.post(
                f"{self.base_url}/chat.postMessage",
                json={"channel": self.channel_id, "text": message},
                headers=headers,
            )
            send_resp.raise_for_status()
            send_data = send_resp.json()
            if not send_data.get("ok"):
                return f"[SEND ERROR: {send_data.get('error', 'unknown')}]"
            sent_ts = send_data["ts"]

            start = time.time()
            while time.time() - start < self.timeout:
                await asyncio.sleep(1.5)
                hist_resp = await client.get(
                    f"{self.base_url}/conversations.history",
                    params={"channel": self.channel_id, "oldest": sent_ts, "limit": 10},
                    headers=headers,
                )
                hist_resp.raise_for_status()
                for msg in hist_resp.json().get("messages", []):
                    if msg.get("ts") == sent_ts:
                        continue
                    if self.target_bot_id:
                        if msg.get("user") == self.target_bot_id or msg.get("bot_id"):
                            return msg.get("text", "[non-text response]")
                    else:
                        if msg.get("bot_id") or msg.get("subtype") == "bot_message":
                            return msg.get("text", "[non-text response]")

        return f"[TIMEOUT: No response within {int(self.timeout)}s]"

    async def check_connection(self) -> dict:
        """Verify token and channel access."""
        headers = {"Authorization": f"Bearer {self.bot_token}"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                auth_resp = await client.post(f"{self.base_url}/auth.test", headers=headers)
                auth_resp.raise_for_status()
                auth_data = auth_resp.json()
                if not auth_data.get("ok"):
                    return {"connected": False, "error": auth_data.get("error", "auth failed")}
                ch_resp = await client.get(
                    f"{self.base_url}/conversations.info",
                    params={"channel": self.channel_id},
                    headers=headers,
                )
                ch_resp.raise_for_status()
                ch_data = ch_resp.json()
                return {
                    "connected": True,
                    "bot_name": auth_data.get("user"),
                    "team": auth_data.get("team"),
                    "channel_name": ch_data.get("channel", {}).get("name"),
                }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def to_dict(self) -> dict:
        return {"type": "slack", "channel_id": self.channel_id, "target_bot_id": self.target_bot_id}
