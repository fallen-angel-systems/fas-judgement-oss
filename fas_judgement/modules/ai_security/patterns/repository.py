"""
Pattern Repository
-------------------
WHY: Pattern file storage and changelog management are persistence concerns.
     Isolating them here means the service can be tested without touching
     the filesystem.

     The changelog auto-detection logic (comparing patterns.json hash against
     last known state) is also here — it's a data management operation,
     not business logic.

LAYER: Module/Repository — imports stdlib, aiosqlite.
SOURCE: Direct port of judgement/pattern_changelog.py.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import aiosqlite

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pattern_changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    date TEXT NOT NULL,
    change_type TEXT NOT NULL,
    category TEXT,
    count INTEGER DEFAULT 0,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

# Seed the initial changelog entries
SEED_ENTRIES = [
    {
        "version": "1.0.0",
        "date": "2026-02-11",
        "change_type": "added",
        "category": None,
        "count": 64,
        "description": "Initial pattern library: 64 patterns across 8 categories (jailbreak, system_prompt_extraction, data_exfiltration, indirect_injection, encoding_evasion, social_engineering, privilege_escalation, multilingual)."
    },
    {
        "version": "1.1.0",
        "date": "2026-02-19",
        "change_type": "added",
        "category": None,
        "count": 30,
        "description": "Volt red team review: 30 new patterns across 10 categories including multi-turn setups, context window manipulation, completion attacks, and structured output exploitation."
    },
    {
        "version": "2.0.0",
        "date": "2026-02-24",
        "change_type": "added",
        "category": None,
        "count": 906,
        "description": "Major expansion: 1000 total patterns. Added advanced encoding chains, multi-language coverage (20+ languages), tool/function calling abuse, and community-submitted patterns."
    },
    {
        "version": "2.0.0",
        "date": "2026-02-24",
        "change_type": "improved",
        "category": "jailbreak",
        "count": 0,
        "description": "Improved sophistication of existing jailbreak patterns. Added context-aware variants that adapt to target responses."
    },
    {
        "version": "2.1.0",
        "date": "2026-02-26",
        "change_type": "added",
        "category": None,
        "count": 0,
        "description": "Added professional report generation (HTML, Markdown, JSON, SARIF). Reports include CWE/OWASP references, executive summaries, and per-finding recommendations."
    },
    {
        "version": "2.1.0",
        "date": "2026-02-26",
        "change_type": "added",
        "category": None,
        "count": 0,
        "description": "Added API key authentication for programmatic access (Pro/Elite). Bearer token support for CI/CD integration."
    },
]


async def seed_changelog(db):
    """Seed initial changelog entries if table is empty."""
    cursor = await db.execute("SELECT COUNT(*) FROM pattern_changelog")
    count = (await cursor.fetchone())[0]
    if count == 0:
        for entry in SEED_ENTRIES:
            await db.execute(
                "INSERT INTO pattern_changelog (version, date, change_type, category, count, description, created_at) VALUES (?,?,?,?,?,?,?)",
                (entry["version"], entry["date"], entry["change_type"], entry.get("category"),
                 entry["count"], entry["description"], datetime.utcnow().isoformat())
            )
        await db.commit()


# --- Auto-detection ---

CREATE_PATTERN_STATE_SQL = """
CREATE TABLE IF NOT EXISTS pattern_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    file_hash TEXT NOT NULL,
    total_count INTEGER NOT NULL,
    category_counts TEXT NOT NULL,
    last_version TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _hash_file(path: str) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_categories(patterns: list) -> dict:
    """Count patterns per category."""
    counts = {}
    for p in patterns:
        cat = p.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items()))


def _bump_version(version: str, change_size: int) -> str:
    """Bump version based on change size. Major add = minor bump, small tweak = patch bump."""
    parts = version.split(".")
    if len(parts) != 3:
        return "2.2.0"
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    if change_size >= 50:
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


async def auto_detect_changes(db, patterns_path: str):
    """
    Compare current patterns.json against last known state.
    If changed, auto-generate changelog entries.
    """
    path = Path(patterns_path)
    if not path.exists():
        return

    # Create state table
    await db.execute(CREATE_PATTERN_STATE_SQL)

    # Load current patterns
    try:
        with open(path) as f:
            patterns = json.load(f)
        if isinstance(patterns, dict):
            patterns = patterns.get("patterns", [])
    except Exception:
        return

    current_hash = _hash_file(str(path))
    current_counts = _count_categories(patterns)
    current_total = len(patterns)

    # Get last known state
    cursor = await db.execute("SELECT * FROM pattern_state WHERE id = 1")
    row = cursor and await cursor.fetchone()

    if row is None:
        # First run — save state, no changelog entry needed (seed covers history)
        await db.execute(
            "INSERT INTO pattern_state (id, file_hash, total_count, category_counts, last_version, updated_at) VALUES (1,?,?,?,?,?)",
            (current_hash, current_total, json.dumps(current_counts), "2.1.0", datetime.utcnow().isoformat())
        )
        await db.commit()
        print(f"  Pattern state initialized: {current_total} patterns, {len(current_counts)} categories")
        return

    # Compare
    old_hash = row[1]  # file_hash
    old_total = row[2]  # total_count
    old_counts = json.loads(row[3])  # category_counts
    last_version = row[4]  # last_version

    if current_hash == old_hash:
        # No changes
        return

    # Changes detected! Generate changelog entries
    print(f"  Pattern changes detected! Old: {old_total} patterns, New: {current_total} patterns")

    new_version = _bump_version(last_version, abs(current_total - old_total))
    today = datetime.utcnow().strftime("%Y-%m-%d")
    now = datetime.utcnow().isoformat()
    entries = []

    # Overall count change
    diff = current_total - old_total
    if diff > 0:
        entries.append({
            "version": new_version, "date": today, "change_type": "added",
            "category": None, "count": diff,
            "description": f"Added {diff} new patterns. Total library: {current_total} patterns."
        })
    elif diff < 0:
        entries.append({
            "version": new_version, "date": today, "change_type": "removed",
            "category": None, "count": abs(diff),
            "description": f"Removed {abs(diff)} patterns. Total library: {current_total} patterns."
        })

    # Per-category changes
    all_cats = set(list(old_counts.keys()) + list(current_counts.keys()))
    for cat in sorted(all_cats):
        old_c = old_counts.get(cat, 0)
        new_c = current_counts.get(cat, 0)
        if old_c == 0 and new_c > 0:
            entries.append({
                "version": new_version, "date": today, "change_type": "added",
                "category": cat, "count": new_c,
                "description": f"New category: {cat} ({new_c} patterns)."
            })
        elif new_c == 0 and old_c > 0:
            entries.append({
                "version": new_version, "date": today, "change_type": "removed",
                "category": cat, "count": old_c,
                "description": f"Removed category: {cat} ({old_c} patterns removed)."
            })
        elif new_c > old_c:
            entries.append({
                "version": new_version, "date": today, "change_type": "added",
                "category": cat, "count": new_c - old_c,
                "description": f"Added {new_c - old_c} patterns to {cat} (now {new_c} total)."
            })
        elif new_c < old_c:
            entries.append({
                "version": new_version, "date": today, "change_type": "removed",
                "category": cat, "count": old_c - new_c,
                "description": f"Removed {old_c - new_c} patterns from {cat} (now {new_c} total)."
            })

    # If file hash changed but counts are same, patterns were modified
    if not entries:
        entries.append({
            "version": new_version, "date": today, "change_type": "updated",
            "category": None, "count": 0,
            "description": f"Updated pattern content. Total library unchanged at {current_total} patterns."
        })

    # Write entries
    for e in entries:
        await db.execute(
            "INSERT INTO pattern_changelog (version, date, change_type, category, count, description, created_at) VALUES (?,?,?,?,?,?,?)",
            (e["version"], e["date"], e["change_type"], e.get("category"), e["count"], e["description"], now)
        )
        print(f"  Changelog: [{e['change_type']}] {e['description']}")

    # Update state
    await db.execute(
        "UPDATE pattern_state SET file_hash=?, total_count=?, category_counts=?, last_version=?, updated_at=? WHERE id=1",
        (current_hash, current_total, json.dumps(current_counts), new_version, now)
    )
    await db.commit()
    print(f"  Pattern state updated to v{new_version}")
