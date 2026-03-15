"""
FAS Judgement OSS — Application Configuration
----------------------------------------------
WHY: All env vars and constants in one place — no scattered os.getenv() calls
     across 20 files. Tests monkeypatch here; production sets env vars.

     OSS version is deliberately minimal: no OAuth, no Stripe, no JWT,
     no admin IDs, no email sending. Your local machine needs none of that.

LAYER: Config (infrastructure) — NO imports from other fas_judgement modules
       to avoid circular dependencies.
"""

import os
from pathlib import Path

# === SECTION: PATHS === #

# DB lives in the package directory so data survives upgrades.
_HERE = Path(__file__).parent

DB_PATH = Path(os.environ.get("JUDGEMENT_DB", str(_HERE / "judgment.db")))

# Patterns bundled with the OSS release (the free 100).
# utils/license.py overrides this path if an Elite license is active.
PATTERNS_PATH = (
    _HERE / "modules/ai_security/patterns/data/single-shot/patterns.json"
)
if not PATTERNS_PATH.exists():
    # Fallback: legacy flat layout (compat for dev installs)
    PATTERNS_PATH = _HERE.parent / "judgement" / "patterns.json"


# === SECTION: AI / OLLAMA === #

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")


# === SECTION: NETWORK SAFETY === #

# Allow scanning localhost/private IP targets (demo bot, local AI endpoints).
# WHY default True for OSS: the whole point of self-hosted Judgement is scanning
# local targets. The hosted app (VPS) sets this to False via env.
ALLOW_LOCAL_TARGETS = os.environ.get("ALLOW_LOCAL_TARGETS", "true").lower() in ("true", "1", "yes")


# === SECTION: PATTERN SUBMISSION PROXY === #

# Where OSS pattern submissions are proxied to (hosted Judgement API).
SUBMIT_API_URL = os.environ.get(
    "JUDGEMENT_SUBMIT_URL", "https://judgement-app.fallenangelsystems.com"
)


# === SECTION: ASCII BANNER === #

from fas_judgement import __version__ as _v

BANNER = (
    "\n"
    "       ██╗██╗   ██╗██████╗  ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗\n"
    "       ██║██║   ██║██╔══██╗██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝\n"
    "       ██║██║   ██║██║  ██║██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║\n"
    "  ██   ██║██║   ██║██║  ██║██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║\n"
    "  ╚█████╔╝╚██████╔╝██████╔╝╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║\n"
    "   ╚════╝  ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝\n"
    f"              OSS v{_v} -- Open-Source Prompt Injection Attack Console\n"
)
