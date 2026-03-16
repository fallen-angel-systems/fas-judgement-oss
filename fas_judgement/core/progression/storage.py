"""
Progression persistence - SQLite-backed progress tracking.

Stores player progress locally so it survives restarts.
Uses the same SQLite approach as multi-turn session storage.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import PlayerProgress


# Default DB location: ~/.fas-judgement/progression.db
DEFAULT_DB_PATH = Path.home() / ".fas-judgement" / "progression.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_progress (
    player_id TEXT NOT NULL,
    module TEXT NOT NULL,
    current_level INTEGER DEFAULT 1,
    total_xp INTEGER DEFAULT 0,
    challenges_completed TEXT DEFAULT '[]',
    level_completions TEXT DEFAULT '{}',
    hints_purchased TEXT DEFAULT '{}',
    challenge_attempts TEXT DEFAULT '{}',
    freeplay_xp_earned TEXT DEFAULT '{}',
    first_run INTEGER DEFAULT 1,
    first_bypass INTEGER DEFAULT 0,
    mode TEXT DEFAULT 'game',
    created_at TEXT,
    updated_at TEXT,
    PRIMARY KEY (player_id, module)
);

CREATE TABLE IF NOT EXISTS xp_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    module TEXT NOT NULL,
    xp_amount INTEGER NOT NULL,
    source TEXT NOT NULL,
    detail TEXT,
    timestamp TEXT NOT NULL
);
"""

# Migration: add columns if they don't exist (safe for existing DBs)
MIGRATIONS = [
    "ALTER TABLE player_progress ADD COLUMN hints_purchased TEXT DEFAULT '{}'",
    "ALTER TABLE player_progress ADD COLUMN challenge_attempts TEXT DEFAULT '{}'",
    "ALTER TABLE player_progress ADD COLUMN freeplay_xp_earned TEXT DEFAULT '{}'",
    "ALTER TABLE player_progress ADD COLUMN first_bypass INTEGER DEFAULT 0",
]


class ProgressionStorage:
    """SQLite-backed progression persistence."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)
            # Run migrations (ignore errors for already-existing columns)
            for migration in MIGRATIONS:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # Column already exists

    def get_progress(self, player_id: str = "local", module: str = "ai_security") -> PlayerProgress:
        """Get or create player progress for a module."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM player_progress WHERE player_id = ? AND module = ?",
                (player_id, module)
            ).fetchone()

            if row:
                return PlayerProgress(
                    player_id=row["player_id"],
                    module=row["module"],
                    current_level=row["current_level"],
                    total_xp=row["total_xp"],
                    challenges_completed=json.loads(row["challenges_completed"]),
                    level_completions=json.loads(row["level_completions"]),
                    hints_purchased=json.loads(row["hints_purchased"]),
                    challenge_attempts=json.loads(row["challenge_attempts"]),
                    freeplay_xp_earned=json.loads(row["freeplay_xp_earned"]),
                    first_run=bool(row["first_run"]),
                    first_bypass=bool(row["first_bypass"]),
                    mode=row["mode"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                    updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                )

            # First time - create fresh progress
            now = datetime.utcnow().isoformat()
            conn.execute(
                """INSERT INTO player_progress 
                   (player_id, module, current_level, total_xp, challenges_completed, 
                    level_completions, hints_purchased, challenge_attempts,
                    freeplay_xp_earned, first_run, first_bypass, mode, created_at, updated_at)
                   VALUES (?, ?, 1, 0, '[]', '{}', '{}', '{}', '{}', 1, 0, 'game', ?, ?)""",
                (player_id, module, now, now)
            )
            return PlayerProgress(
                player_id=player_id,
                module=module,
                first_run=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

    def save_progress(self, progress: PlayerProgress):
        """Save player progress."""
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE player_progress SET
                   current_level = ?, total_xp = ?, challenges_completed = ?,
                   level_completions = ?, hints_purchased = ?, challenge_attempts = ?,
                   freeplay_xp_earned = ?, first_run = ?, first_bypass = ?, mode = ?, updated_at = ?
                   WHERE player_id = ? AND module = ?""",
                (
                    progress.current_level,
                    progress.total_xp,
                    json.dumps(progress.challenges_completed),
                    json.dumps(progress.level_completions),
                    json.dumps(progress.hints_purchased),
                    json.dumps(progress.challenge_attempts),
                    json.dumps(progress.freeplay_xp_earned),
                    int(progress.first_run),
                    int(progress.first_bypass),
                    progress.mode,
                    now,
                    progress.player_id,
                    progress.module,
                )
            )

    def log_xp(self, player_id: str, module: str, xp: int, source: str, detail: str = ""):
        """Log an XP gain event for history/analytics."""
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO xp_log (player_id, module, xp_amount, source, detail, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (player_id, module, xp, source, detail, now)
            )

    def dismiss_first_run(self, player_id: str = "local", module: str = "ai_security", mode: str = "game"):
        """Mark first run as dismissed. Called after 'Shall we play a game?' choice."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE player_progress SET first_run = 0, mode = ?, updated_at = ? WHERE player_id = ? AND module = ?",
                (mode, datetime.utcnow().isoformat(), player_id, module)
            )

    def get_xp_history(self, player_id: str = "local", module: str = "ai_security", limit: int = 50) -> list:
        """Get recent XP events for display."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM xp_log WHERE player_id = ? AND module = ? ORDER BY timestamp DESC LIMIT ?",
                (player_id, module, limit)
            ).fetchall()
            return [dict(r) for r in rows]
