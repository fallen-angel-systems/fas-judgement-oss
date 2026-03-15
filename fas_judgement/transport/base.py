"""
Transport Base Class
--------------------
WHY: Re-exports BaseTransport from core/interfaces so transport modules
     can do `from .base import BaseTransport` rather than crossing into
     the core layer explicitly.

     This also means if we ever need transport-specific base behaviour
     (e.g. retry logic, logging), it goes here without touching core.

LAYER: Transport (infrastructure adapter layer).
"""

from ..core.interfaces import BaseTransport

__all__ = ["BaseTransport"]
