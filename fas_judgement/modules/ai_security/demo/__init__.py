"""
Demo Sub-Package
-----------------
WHAT: The demo sub-package provides a locally-runnable simulated AI chatbot
      that users can target with scanner attacks for practice and testing.

WHY: Lets users experience the full attack-scan-report workflow without
     needing an OpenAI/Anthropic API key or an external target to hit.
     The demo target uses deterministic keyword matching so results are
     reproducible and work offline.

USAGE:
     judgement demo                  → starts demo server (default persona)
     judgement demo hardened         → hardened persona (~90% block rate)
     judgement demo vulnerable       → vulnerable persona (~10% block rate)

LAYER: Module/Demo — sub-package of ai_security module.

COMPONENTS:
     target.py    — FastAPI app (the simulated chatbot server itself)
     personas.py  — Persona definitions and trigger matching logic
     service.py   — Subprocess lifecycle management (start/stop/status)
"""
