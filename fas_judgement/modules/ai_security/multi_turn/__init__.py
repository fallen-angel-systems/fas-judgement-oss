"""
Multi-Turn Attack Sub-Package
------------------------------
WHY: Multi-turn social engineering is architecturally separate from
     single-shot scanning. It has its own:
       - Session state machine (phases, retries, detection)
       - Scoring logic (LLM-based with keyword fallback)
       - Persistence (SQLite via SessionStore)
       - API routes

LAYER: AI Security sub-module.
"""
