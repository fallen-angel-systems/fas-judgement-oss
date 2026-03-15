"""
AI Security Module — Package Init + Auto-Registration
------------------------------------------------------
WHAT: This file is the public entry point for the ai_security module package.
      When Python imports `fas_judgement.modules.ai_security`, this __init__.py
      runs automatically — and we use that moment to register the module
      with the global registry.

WHY: The "self-registration on import" pattern is how we achieve loose coupling
     between modules and app.py. The app never needs to know about AiSecurityModule
     directly — it just imports this package and then asks the registry for all
     routers.

     Future modules (e.g. modules/web_security/) follow the same convention:
     their __init__.py registers them, and app.py imports the package.
     Zero changes to app.py required.

LAYER: Application module boundary — bridges the module and core layers.
"""

from .module import AiSecurityModule
from ...core import registry

# === SECTION: AUTO-REGISTRATION === #

# Instantiate and register when this package is imported.
# WHY a module-level instance: the registry needs a concrete object (not a
# class) because module state (e.g. router config) may differ per instance.
_module = AiSecurityModule()
registry.register(_module)
