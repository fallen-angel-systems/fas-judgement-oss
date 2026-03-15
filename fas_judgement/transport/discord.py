"""
Discord Transport
-----------------
WHY: Discord bots are a common AI deployment target. This transport sends
     attack messages to a Discord channel and polls for a bot's response.

     Design: we poll for messages in the channel after sending, filtering
     for the target bot's user ID. Timeout is configurable (default 30s)
     to allow for slow bot responses.

LAYER: Transport (infrastructure) — imports httpx, asyncio.
SOURCE: Extracted from multi-turn-engine/transport.py DiscordTransport class.
"""

import asyncio
import time

import httpx

from .base import BaseTransport


class DiscordTransport(BaseTransport):
    """Send attacks to Discord bots via the Discord API."""

    transport_type = "discord"

    def __init__(
        self,
        bot_token: str,
        target_bot_id: str,
        channel_id: str,
        timeout: float = 30.0,
    ):
        self.bot_token = bot_token
        self.target_bot_id = target_bot_id
        self.channel_id = channel_id
        self.timeout = timeout
        self.base_url = "https://discord.com/api/v10"

    async def send(self, message: str) -> str:
        """Send message to channel, wait for target bot's response."""
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Send the attack message
            send_resp = await client.post(
                f"{self.base_url}/channels/{self.channel_id}/messages",
                json={"content": message},
                headers=headers,
            )
            send_resp.raise_for_status()
            sent_id = send_resp.json()["id"]

            # Poll for target bot response (up to timeout)
            start = time.time()
            while time.time() - start < self.timeout:
                await asyncio.sleep(1.5)
                msgs_resp = await client.get(
                    f"{self.base_url}/channels/{self.channel_id}/messages",
                    params={"after": sent_id, "limit": 10},
                    headers=headers,
                )
                msgs_resp.raise_for_status()
                for msg in msgs_resp.json():
                    if msg["author"]["id"] == self.target_bot_id:
                        return msg["content"]

        return f"[TIMEOUT: No response from target bot within {int(self.timeout)}s]"

    async def check_connection(self) -> dict:
        """Verify bot token and channel access."""
        headers = {"Authorization": f"Bot {self.bot_token}"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                me_resp = await client.get(f"{self.base_url}/users/@me", headers=headers)
                me_resp.raise_for_status()
                bot_user = me_resp.json()
                ch_resp = await client.get(
                    f"{self.base_url}/channels/{self.channel_id}", headers=headers
                )
                ch_resp.raise_for_status()
                channel = ch_resp.json()
                return {
                    "connected": True,
                    "bot_name": bot_user.get("username"),
                    "channel_name": channel.get("name"),
                    "target_bot_id": self.target_bot_id,
                }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def to_dict(self) -> dict:
        return {"type": "discord", "target_bot_id": self.target_bot_id, "channel_id": self.channel_id}
