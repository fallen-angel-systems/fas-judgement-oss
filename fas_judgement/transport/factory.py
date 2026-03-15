"""
Transport Factory
-----------------
WHY: A factory function keeps transport creation logic in one place.
     When the UI sends a transport config dict, the factory turns it into
     the correct Transport instance. Adding a new transport type only
     requires adding one branch here.

LAYER: Transport (infrastructure).
SOURCE: Extracted from multi-turn-engine/transport.py create_transport().
"""

from .base import BaseTransport
from .http import HTTPTransport
from .ollama import OllamaTransport
from .discord import DiscordTransport
from .telegram import TelegramTransport
from .slack import SlackTransport
from .website import WebsiteTransport


def create_transport(config: dict) -> BaseTransport:
    """
    Factory: create a Transport from a config dict (from the UI).

    Expected keys vary by type:
      http:     target_url, body_field, response_field, headers, conversation_id_field
      local:    ollama_url, model, system_prompt
      discord:  bot_token, target_bot_id, channel_id
      telegram: bot_token, chat_id
      slack:    bot_token, channel_id, target_bot_id
      website:  url, input_selector, send_selector, response_selector

    Raises ValueError for unknown transport types.
    """
    transport_type = config.get("transport", "http")

    if transport_type == "http":
        return HTTPTransport(
            target_url=config.get("target_url", ""),
            body_field=config.get("body_field", "message"),
            response_field=config.get("response_field", "response"),
            headers=config.get("headers"),
            conversation_id_field=config.get("conversation_id_field", ""),
            session_id_field=config.get("session_id_field", ""),
        )
    elif transport_type == "local":
        return OllamaTransport(
            ollama_url=config.get("ollama_url", "http://localhost:11434"),
            model=config.get("model", "qwen2.5:14b"),
            system_prompt=config.get("system_prompt", ""),
        )
    elif transport_type == "discord":
        return DiscordTransport(
            bot_token=config.get("bot_token", ""),
            target_bot_id=config.get("target_bot_id", ""),
            channel_id=config.get("channel_id", ""),
        )
    elif transport_type == "telegram":
        return TelegramTransport(
            bot_token=config.get("bot_token", ""),
            chat_id=config.get("chat_id", ""),
        )
    elif transport_type == "slack":
        return SlackTransport(
            bot_token=config.get("bot_token", ""),
            channel_id=config.get("channel_id", ""),
            target_bot_id=config.get("target_bot_id", ""),
        )
    elif transport_type == "website":
        return WebsiteTransport(
            url=config.get("url", ""),
            input_selector=config.get("input_selector", 'input[type="text"]'),
            send_selector=config.get("send_selector", 'button[type="submit"]'),
            response_selector=config.get("response_selector", ".chat-message:last-child"),
        )
    else:
        raise ValueError(f"Unknown transport type: {transport_type}")
