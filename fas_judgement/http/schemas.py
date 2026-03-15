"""
HTTP Schemas (OSS Edition)
---------------------------
WHY: Pydantic models for request/response validation.
     OSS version is minimal — no auth payloads, no Stripe webhooks.

LAYER: HTTP — imports only pydantic.
"""

from typing import Any, Optional
from pydantic import BaseModel


class AttackRequest(BaseModel):
    """Request body for POST /api/attack."""
    target_url: str
    method: str = "POST"
    headers: dict[str, str] = {}
    payload_field: str = "message"
    payload_template: str = ""
    categories: list[str] = []
    pattern_ids: list[str] = []
    delay_ms: int = 200
    timeout_s: int = 10
    max_patterns: int = 0
    error_cutoff: int = 5
    use_llm: bool = False
    use_mcp: bool = False
    mcp_url: str = ""
    severity_filter: list[str] = []
    category_limits: dict[str, int] = {}
    include_custom: bool = False
    custom_patterns_data: list[dict] = []


class PresetRequest(BaseModel):
    """Request body for POST /api/presets."""
    name: str = "Untitled"
    target_url: str = ""
    method: str = "POST"
    headers: Any = {}
    payload_field: str = "message"
    payload_template: str = ""
    delay_ms: int = 1000
    timeout_s: int = 10


class SettingsRequest(BaseModel):
    """Request body for POST /api/settings (free-form KV pairs)."""
    # Accepts arbitrary key-value pairs — validated at the route level.
    pass


class PatternSubmitRequest(BaseModel):
    """Request body for POST /api/patterns/submit."""
    text: str
    category: str = "auto"
    target_type: str = "chatbot"
    description: str = ""
    submitter_name: str = "anonymous"
