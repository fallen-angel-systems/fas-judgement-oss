"""
Core Repository — Database Initialisation (OSS Edition)
---------------------------------------------------------
WHY: Schema is defined here so all modules share a single source of truth.
     CREATE TABLE IF NOT EXISTS makes every startup safe and idempotent.

     OSS schema is intentionally smaller than Elite:
       ✅ sessions, presets, results, settings, pattern_submissions
       ❌ users, oauth_accounts, pending_upgrades, custom_patterns, api_keys
          (no accounts, no login, no payment flow on your own machine)

LAYER: Core infrastructure — imports only aiosqlite and config.
"""

import aiosqlite

from ..config import DB_PATH


# === SECTION: TABLE DEFINITIONS === #

SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    target_url TEXT,
    method TEXT,
    created_at TEXT,
    total_patterns INTEGER DEFAULT 0,
    blocked INTEGER DEFAULT 0,
    partial INTEGER DEFAULT 0,
    bypassed INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0
)
"""
# WHY no user_id: OSS is single-user. Sessions belong to whoever's running
# the server — no need to track ownership.

PRESETS_DDL = """
CREATE TABLE IF NOT EXISTS presets (
    id TEXT PRIMARY KEY,
    name TEXT,
    target_url TEXT,
    method TEXT,
    headers TEXT,
    payload_field TEXT,
    payload_template TEXT DEFAULT '',
    delay_ms INTEGER DEFAULT 1000,
    timeout_s INTEGER DEFAULT 10,
    created_at TEXT
)
"""

RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    pattern_id TEXT,
    pattern_text TEXT,
    category TEXT,
    request_body TEXT,
    response_status INTEGER,
    response_body TEXT,
    response_time_ms REAL,
    verdict TEXT,
    created_at TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
)
"""

PATTERN_SUBMISSIONS_DDL = """
CREATE TABLE IF NOT EXISTS pattern_submissions (
    id TEXT PRIMARY KEY,
    category TEXT,
    text TEXT,
    target_type TEXT,
    description TEXT,
    submitter_name TEXT,
    status TEXT DEFAULT 'pending',
    submitted_at TEXT
)
"""
# WHY no user_id/guardian_score: OSS submissions are proxied to the hosted
# API which handles scoring and deduplication server-side.

SETTINGS_DDL = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
)
"""


# === SECTION: DB INIT === #

async def init_db(db_path=None) -> None:
    """
    Create all OSS tables if they don't exist. Safe to call on every startup.

    WHY deferred imports: modules/patterns/repository.py is only needed at
    init time, not at import time. Deferred import avoids circular imports
    during package load.
    """
    path = db_path or DB_PATH

    async with aiosqlite.connect(path) as db:
        for ddl in [
            SESSIONS_DDL,
            PRESETS_DDL,
            RESULTS_DDL,
            PATTERN_SUBMISSIONS_DDL,
            SETTINGS_DDL,
        ]:
            await db.execute(ddl)
        await db.commit()

        # Seed the pattern changelog (no-op if already seeded)
        from ..modules.ai_security.patterns.repository import (
            CREATE_TABLE_SQL as CHANGELOG_TABLE_SQL,
            seed_changelog,
            auto_detect_changes,
        )
        await db.execute(CHANGELOG_TABLE_SQL)
        await db.commit()
        await seed_changelog(db)

    # Auto-detect pattern changes after DB is ready
    from ..config import PATTERNS_PATH
    async with aiosqlite.connect(path) as db:
        await auto_detect_changes(db, str(PATTERNS_PATH))
