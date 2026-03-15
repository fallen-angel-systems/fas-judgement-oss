"""
AI Security Module Entry Point
-------------------------------
WHY: The AiSecurityModule class is the registry entry point for this module.
     It implements BaseModule so the registry can discover and mount it
     without knowing its internals.

LAYER: Application module — imports from core/interfaces and sub-packages.
"""

from ...core.interfaces import BaseModule


class AiSecurityModule(BaseModule):
    """
    AI Security Module: provides single-shot attack scanning,
    multi-turn attack sessions, and pattern management.
    """
    name = "ai_security"
    description = "Prompt injection attack scanning and multi-turn social engineering"

    def get_router(self):
        """
        Return a combined FastAPI router for all AI Security endpoints.

        WHY combined: the HTTP app only calls get_router() once per module.
        We compose the sub-routers here rather than in app.py.
        """
        from fastapi import APIRouter
        from .multi_turn.routes import router as mt_router

        router = APIRouter()
        router.include_router(mt_router, prefix="/api/multi-turn")
        return router
