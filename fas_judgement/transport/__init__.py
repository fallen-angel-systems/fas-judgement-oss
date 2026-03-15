"""
Transport Layer
---------------
WHY: Transport is the adapter between the attack orchestrator and the
     target system. Different targets require different delivery mechanisms
     (HTTP API, Discord bot, Telegram bot, Slack bot, website widget).

     By making each transport a standalone class implementing BaseTransport,
     we can:
       1. Swap transports without changing orchestrator logic
       2. Test orchestrators with a fake/mock transport
       3. Add new transports (e.g. WhatsApp) without touching existing code

LAYER: Infrastructure adapter (outermost ring in DDD terms).
       Imports from core/interfaces but NOT from http/ or modules/.
"""

from .base import BaseTransport
from .factory import create_transport
from .http import HTTPTransport
from .ollama import OllamaTransport
from .discord import DiscordTransport
from .telegram import TelegramTransport
from .slack import SlackTransport
from .website import WebsiteTransport

__all__ = [
    "BaseTransport",
    "HTTPTransport",
    "OllamaTransport",
    "DiscordTransport",
    "TelegramTransport",
    "SlackTransport",
    "WebsiteTransport",
    "create_transport",
]
