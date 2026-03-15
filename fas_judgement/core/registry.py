"""
Module Registry
---------------
WHY: Centralising module registration prevents the app.py from importing
     each module directly, which would create tight coupling. Instead,
     modules register themselves, and app.py asks the registry for all
     available routers.

     This is the standard "plugin registry" pattern — you can add a new
     module by registering it here without touching app.py.

LAYER: Core infrastructure concern — bridges the core and HTTP layers.
       Imports BaseModule from core; concrete modules register at import time.
"""

from typing import List, Type

from .interfaces import BaseModule

# === SECTION: REGISTRY STATE === #

# The registry is a simple list because module order matters for router mounting.
_registered_modules: List[BaseModule] = []


# === SECTION: REGISTRY API === #

def register(module: BaseModule) -> None:
    """Register a module instance with the global registry."""
    _registered_modules.append(module)


def get_all_modules() -> List[BaseModule]:
    """Return all registered module instances."""
    return list(_registered_modules)


def get_all_routers():
    """
    Convenience: return FastAPI routers from all registered modules.
    Used by app.py to mount everything in one loop.
    """
    return [m.get_router() for m in _registered_modules]
