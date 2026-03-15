"""
Pattern Service
----------------
WHY: Pattern loading and filtering is used in multiple places — the /api/attack
     endpoint, the /api/patterns endpoint, and the report generator. Centralising
     it here prevents duplication and ensures consistent tier-gating logic.

LAYER: Module/Service — imports json, stdlib, config. No HTTP imports.
SOURCE: Extracted from server.py lines 520–540 (load_patterns, filter logic).
"""

import json
from pathlib import Path

from ....config import PATTERNS_PATH


# === SECTION: PATTERN LOADING === #

def load_patterns(patterns_path: Path = None) -> list:
    """
    Load the full pattern library from disk.

    WHY default parameter: tests can pass a custom path; production uses
    the config default. Both cases share the same loading logic.
    """
    path = patterns_path or PATTERNS_PATH
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def get_pattern_categories(patterns: list) -> dict:
    """Return a dict of {category: count} from a pattern list."""
    cats = {}
    for p in patterns:
        cat = p.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    return cats


def filter_by_tier(patterns: list, user_tier: str) -> list:
    """
    Filter patterns based on user's subscription tier.

    Tier ordering: free (0) < pro (1) < elite (2)
    A pattern with tier="elite" is hidden from free/pro users.
    """
    tier_order = {"free": 0, "pro": 1, "elite": 2, "elite_home": 2, "elite_business": 2, "admin": 3}
    user_level = tier_order.get(user_tier, 0)
    return [p for p in patterns if tier_order.get(p.get("tier", "free"), 0) <= user_level]


def build_pattern_response(patterns: list, user_tier: str) -> dict:
    """
    Build the /api/patterns response dict for a given tier.

    WHY not expose raw patterns: the response omits attack text for locked
    patterns (only shows 30-char preview) to prevent free-tier scraping.
    """
    tier_order = {"free": 0, "pro": 1, "elite": 2, "elite_home": 2, "elite_business": 2, "admin": 3}
    user_level = tier_order.get(user_tier, 0)

    categories = {}
    result_patterns = []
    for p in patterns:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1
        pattern_tier = p.get("tier", "free")
        pattern_level = tier_order.get(pattern_tier, 0)
        locked = pattern_level > user_level

        entry = {
            "id": p["id"],
            "category": p["category"],
            "tier": pattern_tier,
        }
        # Add optional metadata fields if present
        for field in ("name", "severity", "technique"):
            if p.get(field):
                entry[field] = p[field]

        if locked:
            entry["preview"] = p.get("text", "")[:30]
            entry["locked"] = True
        else:
            entry["preview"] = p.get("text", "")[:60]
            entry["locked"] = False
            if p.get("description"):
                entry["description"] = p["description"]

        result_patterns.append(entry)

    return {"patterns": result_patterns, "categories": categories, "total": len(patterns)}
