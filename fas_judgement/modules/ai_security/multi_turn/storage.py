"""
Multi-Turn Session Storage
----------------------------
WHY: SQLite-backed persistence keeps sessions alive across server restarts.
     The SessionStore uses a synchronous sqlite3 connection (not aiosqlite)
     because the Orchestrator is designed to be called from sync contexts
     in tests, not just async HTTP handlers.

LAYER: Module/Storage — stdlib sqlite3 only.
SOURCE: Direct port of multi-turn-engine/storage.py.
"""
import json
import sqlite3
from pathlib import Path
from typing import Optional

from .service import Session, Turn, Mode, PhaseResult


DEFAULT_DB_PATH = Path.home() / ".fas-judgement" / "sessions.db"


class SessionStore:
    """SQLite-backed session persistence for multi-turn attack sessions."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                category_name TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'auto',
                target_url TEXT,
                target_name TEXT,
                ollama_url TEXT DEFAULT 'http://localhost:11434',
                ollama_model TEXT DEFAULT '',
                current_phase INTEGER DEFAULT 1,
                max_phases INTEGER DEFAULT 5,
                status TEXT DEFAULT 'active',
                retries_this_phase INTEGER DEFAULT 0,
                max_retries_per_phase INTEGER DEFAULT 3,
                used_messages TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                phase INTEGER NOT NULL,
                attack_message TEXT NOT NULL,
                target_response TEXT,
                score TEXT,
                score_reason TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                phase INTEGER NOT NULL,
                attack TEXT NOT NULL,
                response TEXT,
                severity TEXT DEFAULT 'medium',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        """)
        conn.commit()

    def save_session(self, session: Session) -> None:
        """Insert or update a session.

        Uses INSERT on first save, UPDATE on subsequent to avoid
        triggering ON DELETE CASCADE on the foreign keys.
        """
        conn = self._get_conn()
        used_json = json.dumps(
            {k: list(v) for k, v in session.used_messages.items()}
        )
        # Try UPDATE first (avoids CASCADE delete from INSERT OR REPLACE)
        cursor = conn.execute("""
            UPDATE sessions SET
                category = ?, category_name = ?, mode = ?, target_url = ?,
                target_name = ?, ollama_url = ?, ollama_model = ?,
                current_phase = ?, max_phases = ?, status = ?,
                retries_this_phase = ?, max_retries_per_phase = ?,
                used_messages = ?, updated_at = datetime('now')
            WHERE id = ?
        """, (
            session.category, session.category_name,
            session.mode.value, session.target_url, session.target_name,
            session.ollama_url, session.ollama_model,
            session.current_phase, session.max_phases, session.status,
            session.retries_this_phase, session.max_retries_per_phase,
            used_json, session.id,
        ))
        if cursor.rowcount == 0:
            # First save — INSERT
            conn.execute("""
                INSERT INTO sessions
                (id, category, category_name, mode, target_url, target_name,
                 ollama_url, ollama_model, current_phase, max_phases, status,
                 retries_this_phase, max_retries_per_phase, used_messages,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                session.id, session.category, session.category_name,
                session.mode.value, session.target_url, session.target_name,
                session.ollama_url, session.ollama_model,
                session.current_phase, session.max_phases, session.status,
                session.retries_this_phase, session.max_retries_per_phase,
                used_json, session.created_at,
            ))
        conn.commit()

    def save_turn(self, session_id: str, turn: Turn) -> None:
        """Save a turn to the database."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO turns
            (session_id, turn_number, phase, attack_message, target_response,
             score, score_reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, turn.turn_number, turn.phase, turn.attack_message,
            turn.target_response,
            turn.score.value if turn.score else None,
            turn.score_reason, turn.timestamp,
        ))
        conn.commit()

    def update_turn(self, session_id: str, turn: Turn) -> None:
        """Update an existing turn (after scoring)."""
        conn = self._get_conn()
        conn.execute("""
            UPDATE turns SET
                target_response = ?,
                score = ?,
                score_reason = ?
            WHERE session_id = ? AND turn_number = ?
        """, (
            turn.target_response,
            turn.score.value if turn.score else None,
            turn.score_reason,
            session_id, turn.turn_number,
        ))
        conn.commit()

    def save_finding(self, session_id: str, finding: dict) -> None:
        """Save a finding."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO findings (session_id, turn, phase, attack, response, severity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id, finding["turn"], finding["phase"],
            finding["attack"], finding.get("response", "")[:500],
            finding.get("severity", "medium"),
        ))
        conn.commit()

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session and its turns/findings from the database."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None

        # Rebuild used_messages
        used_raw = json.loads(row["used_messages"])
        used_messages = {k: set(v) for k, v in used_raw.items()}

        session = Session(
            id=row["id"],
            category=row["category"],
            category_name=row["category_name"],
            mode=Mode(row["mode"]),
            target_url=row["target_url"],
            target_name=row["target_name"],
            ollama_url=row["ollama_url"],
            ollama_model=row["ollama_model"],
            current_phase=row["current_phase"],
            max_phases=row["max_phases"],
            status=row["status"],
            retries_this_phase=row["retries_this_phase"],
            max_retries_per_phase=row["max_retries_per_phase"],
            used_messages=used_messages,
            created_at=row["created_at"],
        )

        # Load turns
        turn_rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number",
            (session_id,)
        ).fetchall()
        for tr in turn_rows:
            turn = Turn(
                turn_number=tr["turn_number"],
                phase=tr["phase"],
                attack_message=tr["attack_message"],
                target_response=tr["target_response"],
                score=PhaseResult(tr["score"]) if tr["score"] else None,
                score_reason=tr["score_reason"],
                timestamp=tr["timestamp"],
            )
            session.turns.append(turn)

        # Load findings
        finding_rows = conn.execute(
            "SELECT * FROM findings WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
        for fr in finding_rows:
            session.findings.append({
                "turn": fr["turn"],
                "phase": fr["phase"],
                "attack": fr["attack"],
                "response": fr["response"],
                "severity": fr["severity"],
            })

        return session

    def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List sessions with optional status filter."""
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT id, category_name, mode, status, current_phase, created_at "
                "FROM sessions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, category_name, mode, status, current_phase, created_at "
                "FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()

        return [
            {
                "id": r["id"],
                "category": r["category_name"],
                "mode": r["mode"],
                "status": r["status"],
                "phase": r["current_phase"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its turns/findings."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
